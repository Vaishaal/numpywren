from numpywren.matrix import BigMatrix, BigSymmetricMatrix
from numpywren import matrix_utils, uops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest
import concurrent.futures as fs
import time
import multiprocessing
import os
from importlib import reload
from numpywren import lambdapack as lp
from numpywren import binops 
from collections import defaultdict
import seaborn as sns
import boto3
import numpywren
import numpywren.wait
from numpywren import job_runner

redis_env ={"REDIS_IP": os.environ.get("REDIS_IP", ""), "REDIS_PASS": os.environ.get("REDIS_PASS", "")}
def program_state_inspect(program):
    executor = fs.ThreadPoolExecutor(64)
    futures = []
    for i in range(len(program.inst_blocks)):
        futures.append(executor.submit(program.get_profiling_info, i))
    fs.wait(futures)
    max_state = max([f.result() for f in futures if f.result() == lp.PS.SUCCESS])
    running = sum([1 for f in futures if f.result() == lp.PS.RUNNING])
    return max_state, running

def num_running_program_state(program):
    return sum([1 for i in range(len(program.inst_blocks)) if program.inst_block_status(i) == lp.PS.RUNNING] + [0])

D = 65536
X = np.random.randn(int(D), 1)
print("Generating X")
shard_size = 4096
shard_sizes = (shard_size, 1)
X_sharded = BigMatrix("cholesky_test_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
shard_matrix(X_sharded, X)
pwex = pywren.default_executor()
print("Generating matrix")
XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, overwrite=False)
XXT_sharded = BigSymmetricMatrix("gemm(BigMatrix(cholesky_test_X), BigMatrix(cholesky_test_X).T)")
XXT_sharded.lambdav = D
instructions,L_sharded,trailing = lp._chol(XXT_sharded)
print("NUmber of instructions", len(instructions))
L_sharded.free()
instructions  = instructions
print(L_sharded.key)
print(L_sharded.bucket)
print("Block idxs exist total", len(L_sharded.block_idxs))
print("Block idxs exist not before", len(L_sharded.block_idxs_not_exist))
executor = pywren.default_executor
config = pwex.config
program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config, num_priorities=1, eager=True)
print("program.hash", program.hash)
print("LONGEST PATH ", program.longest_path)
t = time.time()
program.start()
num_cores = 512
executor = fs.ProcessPoolExecutor(num_cores)
all_futures  = []
'''
for c in range(num_cores):
    all_futures.append(executor.submit(job_runner.lambdapack_run, program, 3))
'''
t = time.time()
all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=3, cache_size=10), range(num_cores), extra_env=redis_env)
time.sleep(10)
last_run = time.time()
while(program.program_status() == lp.PS.RUNNING):
    max_pc = program.get_max_pc()
    print("Max PC is {0}".format(max_pc))
    time.sleep(5)
    waiting = 0
    running = 0
    for i, queue_url in enumerate(program.queue_urls):
        client = boto3.client('sqs')
        print("Priority {0}".format(i))
        attrs = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible'])['Attributes']
        print(attrs)
        waiting += int(attrs["ApproximateNumberOfMessages"])
        running += int(attrs["ApproximateNumberOfMessagesNotVisible"])
    if ((waiting > 10 or running == 0) and (time.time() - last_run) > 10):
        print("Looks like there are jobs spinning up more...!")
        last_run = time.time()
        more_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=3), range(waiting), extra_env=redis_env)
e = time.time()
print(program.program_status())
print("PROGRAM STATUS ", program.program_status())
print("PROGRAM HASH", program.hash)
print("Block idxs exist after", len(L_sharded.block_idxs_exist))
print("Took {0} seconds".format(e - t))
print("Downloading L_npw")
L_npw = L_sharded.numpy()
XXT = XXT_sharded.numpy()
L = np.linalg.cholesky(XXT)
print("DOWLOADING XXT")
print(L_npw)
print(L)
assert(np.allclose(L_npw, L))

print("Downloading L_npw")
L_npw = L_sharded.numpy()
print("DOWLOADING XXT")
print(L_npw)
print(L)
assert(np.allclose(L_npw, L))

