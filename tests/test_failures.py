from numpywren.matrix import BigMatrix, BigSymmetricMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner
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

class FailureTests(unittest.TestCase):
    ''' Lambdapack operations must be idempotent and fault tolerant '''

    def test_cholesky_multi_repeats(self):
        ''' Insert repeated instructions into PC queue avoid double increments '''

        print("RUNNING MULTI")
        np.random.seed(1)
        size = 256
        shard_size = 64
        repeats = 18
        np.random.seed(2)
        print("Generating X")
        X = np.random.randn(size, 128)
        print("Generating A")
        A = X.dot(X.T) + size*np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A_{0}".format(int(time.time())), shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions,L_sharded,trailing = lp._chol(A_sharded)
        L_sharded.free()
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config, eager=True)
        print("PROGRAM HASH", program.hash)
        cores = 16
        program.start()
        jobs = []

        for c in range(cores):
            p = mp.Process(target=job_runner.lambdapack_run, args=(program,), kwargs={'timeout':3600, 'pipeline_width':5})
            jobs.append(p)
            p.start()

        np.random.seed(0)
        while(program.program_status() == lp.PS.RUNNING):
            sqs = boto3.resource('sqs', region_name='us-west-2')
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
                max_pc = program.get_max_pc()
                print("Max PC is {0}".format(max_pc))
                if (max_pc == 0): continue
                pc = int(np.random.choice(max_pc, 1)[0])
                inst_block = program.inst_blocks[pc]
                priority = inst_block.priority
                queue = sqs.Queue(program.queue_urls[priority])
                print("Malicilously enqueueing node ", pc)
                queue.send_message(MessageBody=str(pc))
                time.sleep(1)

        for p in jobs:
            p.join()

        print("Program status")
        print(program.program_status())
        program.free()
        profiled_blocks = program.get_all_profiling_info()
        print(lp.perf_profile(profiled_blocks))
        for pc,profiled_block in enumerate(profiled_blocks):
            total_time = 0
            actual_time = profiled_block.end_time - profiled_block.start_time
            for instr in profiled_block.instrs:
                if (instr.end_time == None or instr.start_time == None):
                    continue
                total_time += instr.end_time - instr.start_time
            print("Block {0} total_time {1} pipelined time {2}".format(pc, total_time, actual_time))
        time.sleep(1)
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        print(L_npw)
        print(L)
        print("MAX ", np.max(np.abs(L - L_npw)))
        z = np.argmax(np.abs(L - L_npw))
        print("L_local max value", L.ravel()[z])
        print("L_npw wrong value", L_npw.ravel()[z])
        assert(np.allclose(L_npw, L))
        for pc in range(len(program.inst_blocks)):
            edge_sum = lp.get(program._node_edge_sum_key(pc))
            if (edge_sum == None):
                edge_sum = 0
            edge_sum = int(edge_sum)
            indegree = len(program.parents[pc])
            node_status =  program.get_node_status(pc)
            redis_str = "PC: {0}, Edge Sum: {1}, Indegree: {2}, Node Status {3}".format(pc, edge_sum, indegree, node_status)
            if (edge_sum != indegree):
                print(redis_str)
            assert(edge_sum == indegree)



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
        instructions,L_sharded,trailing = lp._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config, eager=True)
        cores = 16
        program.start()
        jobs = []

        for c in range(cores):
            p = mp.Process(target=job_runner.lambdapack_run, args=(program,), kwargs={'timeout':3600, 'pipeline_width':4})
            jobs.append(p)
            p.start()

        np.random.seed(0)
        while(program.program_status() == lp.PS.RUNNING):
            sqs = boto3.resource('sqs', region_name='us-west-2')
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
        profiled_blocks = program.get_all_profiling_info()
        print(lp.perf_profile(profiled_blocks))
        for pc,profiled_block in enumerate(profiled_blocks):
            total_time = 0
            actual_time = profiled_block.end_time - profiled_block.start_time
            for instr in profiled_block.instrs:
                if (instr.end_time == None or instr.start_time == None):
                    continue
                total_time += instr.end_time - instr.start_time
            print("Block {0} total_time {1} pipelined time {2}".format(pc, total_time, actual_time))
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        print(L_npw)
        print(L)
        print("MAX ", np.max(np.abs(L - L_npw)))
        assert(np.allclose(L_npw, L))
