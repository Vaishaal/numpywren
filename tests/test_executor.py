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

class LambdapackExecutorTest(unittest.TestCase):
    def test_cholesky_single(self):
        X = np.random.randn(4,4)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=A.shape, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions,L_sharded,trailing = lp._chol(A_sharded)
        executor = pywren.lambda_executor
        config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config)
        program.start()
        job_runner.lambdapack_run(program)
        program.wait()
        program.free()
        print("Program status")
        print(program.program_status())


    def test_cholesky_multi(self):
        print("RUNNING MULTI")
        np.random.seed(1)
        size = 128
        shard_size = 64
        np.random.seed(1)
        print("Generating X")
        X = np.random.randn(size, 128)
        print("Generating A")
        A = X.dot(X.T) + np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions,L_sharded,trailing = lp._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config)
        print(program)
        program.start()
        job_runner.lambdapack_run(program)
        program.wait()
        print("Program status")
        print(program.program_status())
        program.free()
        profiled_blocks = program.get_all_profiling_info()
        print(lp.perf_profile(profiled_blocks))

    def test_cholesky_lambda_single(self): 
        print("RUNNING single lambda")
        np.random.seed(1)
        size = 128
        shard_size = 128
        np.random.seed(1)
        print("Generating X")
        X = np.random.randn(size, 128)
        print("Generating A")
        A = X.dot(X.T) + np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions,L_sharded,trailing = lp._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config)
        print(program)
        program.start()
        num_cores = 1
        while (program.program_status() == lp.EC.RUNNING):
            futures = pwex.map(lambda x: job_runner.lambdapack_run(program), range(num_cores))
            pywren.wait(futures)
            [f.result() for f in futures]
        job_runner.lambdapack_run(program)
        program.wait()
        print("Program status")
        print(program.program_status())
        program.free()
        profiled_blocks = program.get_all_profiling_info()
        print(lp.perf_profile(profiled_blocks))

    def test_cholesky_lambda_multi(self): 
        print("RUNNING single lambda")
        np.random.seed(1)
        size = 128
        shard_size = 32
        np.random.seed(1)
        print("Generating X")
        X = np.random.randn(size, 128)
        print("Generating A")
        A = X.dot(X.T) + np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions,L_sharded,trailing = lp._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config)
        print(program)
        program.start()
        num_cores = 1000
        print("Spinning up {0} workers and pre-provisioning DynamoDB".format(num_cores))
        client = boto3.client('dynamodb')
        #client.update_table(TableName='lambdapack', ProvisionedThroughput={ 'ReadCapacityUnits': min(num_cores*2, int(1e4)), 'WriteCapacityUnits': min(num_cores*2, int(1e4))})
        #print("Sleeping while update propagates")
        #time.sleep(10)

        all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program), range(num_cores))
        while (program.program_status() == lp.EC.RUNNING):
            print("Waiting...")
            dones, not_dones = numpywren.wait.wait(all_futures, numpywren.wait.ALWAYS)
            [f.result() for f in dones]
            if (num_cores - len(not_dones) > 0):
                futures = pwex.map(lambda x: job_runner.lambdapack_run(program), range(num_cores - len(not_dones)))
                all_futures += futures
            time.sleep(5)
        program.wait()
        print("Program status")
        print(program.program_status())
        program.free()
        profiled_blocks = program.get_all_profiling_info()
        print(lp.perf_profile(profiled_blocks))
        #client.update_table(TableName='lambdapack', ProvisionedThroughput={ 'ReadCapacityUnits':100, 'WriteCapacityUnits':100})
        #time.sleep(10)
        print("Sleeping while update propagates")
