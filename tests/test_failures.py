from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner
from numpywren import compiler
from numpywren.matrix_init import shard_matrix
import numpywren.wait
import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest
import concurrent.futures as fs
import time
import os
import boto3
import multiprocessing as mp
import numpywren as npw
import json

class FailureTests(unittest.TestCase):
    ''' Lambdapack operations must be idempotent and fault tolerant '''

    def test_cholesky_multi_repeats(self):
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
        instructions,trailing, L_sharded= compiler._chol(A_sharded)
        all_nodes = instructions.unroll_program()
        L_sharded.free()
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = npw.config.default()
        pywren_config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=pywren_config, config=config, eager=True)
        print("PROGRAM HASH", program.hash)
        cores = 1
        program.start()
        jobs = []

        for c in range(cores):
            p = mp.Process(target=job_runner.lambdapack_run, args=(program,), kwargs={'timeout':3600, 'pipeline_width':5})
            jobs.append(p)
            p.start()

        np.random.seed(0)
        try:
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
                    pc = int(np.random.choice(min(p, len(all_nodes)), 1))
                    node = all_nodes[pc]
                    queue = sqs.Queue(program.queue_urls[0])
                    total_repeats -= 1
                    if (total_repeats > 0):
                        print("Malicilously enqueueing node ", pc, node, total_repeats)
                        queue.send_message(MessageBody=json.dumps(node))
                    time.sleep(1)
        #for p in jobs:
        #    p.join()
        except:
            pass


        print("Program status")
        print(program.program_status())
        for node in all_nodes:
            edge_sum = lp.get(program.control_plane.client, program._node_edge_sum_key(*node))
            if (edge_sum == None):
                edge_sum = 0
            edge_sum = int(edge_sum)
            parents = program.program.get_parents(*node)
            children = program.program.get_children(*node)
            indegree = len(parents)
            node_status =  program.get_node_status(*node)
            redis_str = "Node: {0}, Edge Sum: {1}, Indegree: {2}, Node Status {3}".format(node, edge_sum, indegree, node_status)
            if (edge_sum != indegree):
                print(redis_str)
                for p in parents:
                    p_status = program.get_node_status(*p)
                    edge_key = program._edge_key(p[0], p[1], node[0], node[1])
                    edge_value = lp.get(program.control_plane.client, edge_key)
                    child_str = "Parent Node: {0}, Parent Status: {1}, Edge Key: {2}".format(p, p_status, edge_value)
                    print(child_str)
            #assert(edge_sum == indegree)
        program.free()
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        z = np.argmax(np.abs(L - L_npw))
        assert(np.allclose(L_npw, L))



    def test_cholesky_multi_failures(self):
        ''' Insert repeated instructions into PC queue avoid double increments '''

        print("RUNNING MULTI")
        np.random.seed(1)
        size = 256
        shard_size = 64
        failures =  4
        np.random.seed(1)
        print("Generating X")
        X = np.random.randn(size, 128)
        print("Generating A")
        A = X.dot(X.T) + size*np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions,trailing, L_sharded= compiler._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        pywren_config = pwex.config
        config = npw.config.default()
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=pywren_config, config=config, eager=False)
        cores = 16
        program.start()
        jobs = []

        for c in range(cores):
            p = mp.Process(target=job_runner.lambdapack_run, args=(program,), kwargs={'timeout':3600, 'pipeline_width':4})
            jobs.append(p)
            p.start()

        np.random.seed(0)
        while(program.program_status() == lp.PS.RUNNING):
            sqs = boto3.resource('sqs', region_name=program.control_plane.region)
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
            time.sleep(10)
            if (np.random.random() > 0.65):
                for i in range(failures):
                    core = int(np.random.choice(cores, 1)[0])
                    print("Maliciously Killing a job!")
                    jobs[core].terminate()
                    p = mp.Process(target=job_runner.lambdapack_run, args=(program,), kwargs={'timeout':3600, 'pipeline_width':4})
                    p.start()
                    jobs[core] = p

        for p in jobs:
            p.join()

        print("Program status")
        print(program.program_status())
        program.free()
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        print(L_npw)
        print(L)
        print("MAX ", np.max(np.abs(L - L_npw)))
        assert(np.allclose(L_npw, L))
