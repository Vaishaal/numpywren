import time
import random
from timeit import default_timer as timer
import string
import concurrent.futures as fs

from numpywren import compiler, job_runner, kernels
from numpywren.matrix import BigMatrix
from numpywren.alg_wrappers import cholesky, tsqr, gemm, qr, bdfac
import numpy as np
from numpywren.matrix_init import shard_matrix
from numpywren.algs import *
from numpywren.compiler import lpcompile, walk_program, find_parents, find_children
from numpywren import config
from numpywren import lambdapack as lp
import numpywren as npw
import pywren
import pywren.wrenconfig as wc
import multiprocessing as mp
import boto3
import json

def test_cholesky():
    X = np.random.randn(64, 64)
    A = X.dot(X.T) + np.eye(X.shape[0])
    shard_size = 32
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("job_runner_test", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    program, meta =  cholesky(A_sharded)
    executor = fs.ProcessPoolExecutor(1)
    print("starting program")
    program.start()
    future = job_runner.lambdapack_run(program, timeout=60, idle_timeout=6)
    program.wait()
    program.free()
    L_sharded = meta["outputs"][0]
    L_npw = L_sharded.numpy()
    L = np.linalg.cholesky(A)
    assert(np.allclose(L_npw, L))
    print("great success!")

def test_cholesky_timeouts():
    X = np.random.randn(64, 64)
    A = X.dot(X.T) + np.eye(X.shape[0])
    shard_size = 8
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("job_runner_test", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    program, meta =  cholesky(A_sharded)
    executor = fs.ProcessPoolExecutor(1)
    print("starting program")
    program.start()
    future = executor.submit(job_runner.lambdapack_run, program, timeout=10, idle_timeout=6)
    time.sleep(15)
    print("poop")
    assert(int(program.get_up()) == 0)
    program.free()
    print("great success!")


def test_cholesky_multiprocess():
    X = np.random.randn(128, 128)
    A = X.dot(X.T) + 1e9*np.eye(X.shape[0])
    shard_size = 8
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("job_runner_test", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    program, meta =  cholesky(A_sharded)
    executor = fs.ProcessPoolExecutor(8)
    print("starting program")
    program.start()
    futures = []
    for i in range(8):
        future = executor.submit(job_runner.lambdapack_run, program, timeout=25)
        futures.append(future)
    print("Waiting for futures")
    fs.wait(futures)
    [f.result() for f in futures]
    futures = []
    for i in range(8):
        future = executor.submit(job_runner.lambdapack_run, program, timeout=25)
        futures.append(future)
    print("Waiting for futures..again")
    fs.wait(futures)
    [f.result() for f in futures]
    print("great success!")
    return 0


def test_cholesky_lambda():
    X = np.random.randn(128, 128)
    A = X.dot(X.T) + np.eye(X.shape[0])
    shard_size = 128
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("job_runner_test", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    program, meta =  cholesky(A_sharded)
    executor = fs.ProcessPoolExecutor(1)
    print("starting program")
    program.start()
    pwex = pywren.default_executor()
    futures = pwex.map(lambda x: job_runner.lambdapack_run(program, timeout=60, idle_timeout=6), range(16))
    pywren.wait(futures)
    print("RESULTSSS")
    print([f.result() for f in futures])
    futures = pwex.map(lambda x: job_runner.lambdapack_run(program, timeout=60, idle_timeout=6), range(16))
    program.wait()
    #program.free()
    L_sharded = meta["outputs"][0]
    L_npw = L_sharded.numpy()
    L = np.linalg.cholesky(A)
    assert(np.allclose(L_npw, L))
    print("great success!")

def test_cholesky_multi_repeats():
    ''' Insert repeated instructions into PC queue avoid double increments '''

    print("RUNNING MULTI")
    np.random.seed(1)
    size = 256
    shard_size = 30
    repeats = 15
    total_repeats = 150
    np.random.seed(2)
    print("Generating X")
    X = np.random.randn(size, 128)
    print("Generating A")
    A = X.dot(X.T) + size*np.eye(X.shape[0])
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("cholesky_test_A_{0}".format(int(time.time())), shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    program, meta =  cholesky(A_sharded)
    states = compiler.walk_program(program.program.remote_calls)
    L_sharded = meta["outputs"][0]
    L_sharded.free()
    pwex = pywren.default_executor()
    executor = pywren.lambda_executor
    config = npw.config.default()
    print("PROGRAM HASH", program.hash)
    cores = 1
    program.start()
    jobs = []

    for c in range(cores):
        p = mp.Process(target=job_runner.lambdapack_run, args=(program,), kwargs={'timeout':3600, 'pipeline_width':3})
        jobs.append(p)
        p.start()

    np.random.seed(0)
    while(program.program_status() == lp.PS.RUNNING):
        sqs = boto3.resource('sqs', region_name=program.control_plane.region)
        time.sleep(0.5)
        waiting = 0
        running = 0
        for i, queue_url in enumerate(program.queue_urls):
            client = boto3.client('sqs')
            print("Priority {0}".format(i))
            attrs = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible'])['Attributes']
            print(attrs)
            waiting += int(attrs["ApproximateNumberOfMessages"])
            running += int(attrs["ApproximateNumberOfMessagesNotVisible"])
        print("SQS QUEUE STATUS Waiting {0}, Running {1}".format(waiting, running))
        for i in range(repeats):
            p = program.get_progress()
            if (p is None):
                continue
            else:
                p = int(p)
            pc = int(np.random.choice(min(p, len(states)), 1))
            node = states[pc]
            queue = sqs.Queue(program.queue_urls[0])
            total_repeats -= 1
            if (total_repeats > 0):
                print("Malicilously enqueueing node ", pc, node, total_repeats)
                queue.send_message(MessageBody=json.dumps(node))
            time.sleep(1)
    #for p in jobs:
    #    p.join()
    program.wait()
    program.free()
    L_npw = L_sharded.numpy()
    L = np.linalg.cholesky(A)
    z = np.argmax(np.abs(L - L_npw))
    assert(np.allclose(L_npw, L))


if __name__ == "__main__":
    #test_cholesky_multi_repeats()
    #test_cholesky()
    test_cholesky_timeouts()
    #test_cholesky_lambda()
