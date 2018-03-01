
import asyncio
import asyncio
import aiobotocore
import io
import numpy as np
import boto3
import concurrent.futures as fs
import time
import pywren

async def download_data():
    t = time.time()
    loop = asyncio.get_event_loop()
    session = aiobotocore.get_session(loop=loop)
    client = session.create_client('s3', use_ssl=False, verify=False)
    print("Starting download")
    resp = await client.get_object(Bucket=bucket, Key=in_key)
    async with resp['Body'] as stream:
        matrix_bytes = await stream.read()
    bio = io.BytesIO(matrix_bytes)
    e = time.time()
    return np.load(bio), e - t

async def upload_data(xy):
    t = time.time()
    loop = asyncio.get_event_loop()
    session = aiobotocore.get_session(loop=loop)
    client = session.create_client('s3', use_ssl=False, verify=False)
    bio = io.BytesIO()
    np.save(bio, xy)
    print("Starting upload")
    resp = await client.put_object(Bucket=bucket, Key=out_key, Body=bio.getvalue())
    e = time.time()
    print("Upload took ", e - t)

    return e - t


async def matmul_async(x, y, tp, loop):
    x, x_time = await x
    y, y_time = await y
    print("Starting MatMul")
    return await loop.run_in_executor(tp, matmul, x, y), x_time, y_time


def matmul(x,y):
    t = time.time()
    m = x.dot(y)
    e = time.time()
    print("Matmul took ", e - t)
    return m, e - t


async def time_pipeline(tp, loop, queue_url):
    session = aiobotocore.get_session(loop=loop)
    sqs_client = session.create_client('sqs', use_ssl=False)
    messages = await sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
    print("DEQUEING", int(messages["Messages"][0]["Body"]))
    loop.stop()
    return 0
    x, x_time  = await download_data()
    y, y_time = await download_data()
    xy, matmul_time = await loop.run_in_executor(tp, matmul, x, y)
    upload_time = await upload_data(xy)
    return upload_time + matmul_time + x_time + y_time







in_key = "test_key_in"
out_key = "out_key_in"
bucket = "pictureweb"
pipeline_width = 5
client = boto3.client('sqs')
queue_url = client.create_queue(QueueName="asynciobench")["QueueUrl"]

import time
import concurrent.futures as fs
import boto3
from numpywren import lambdapack as lp
import traceback
from multiprocessing.dummy import Pool as ThreadPool
import aiobotocore
import asyncio


class LambdaPackExecutor(object):
    def __init__(self, program, loop, pipeline_width=5):
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.loop = loop
        self.program = program
        self.parent_children = {}
        self.ret_status_map = {}
        self.pc_block_map = {}
        self.block_ends= set()

    async def run(self, pc, computer=None):
        t = time.time()
        self.program.pre_op(pc)
        instrs = self.program.inst_blocks[pc].instrs
        # first instruction in every instruction block is executable!
        for i,inst in enumerate(instrs[0:-1]):
            self.parent_children[inst] = instrs[i+1]
        for i,inst in enumerate(instrs):
            self.pc_block_map[inst] = pc
        ret_codes = []
        loop = self.loop
        runtimes = []
        try:
            for instr in instrs:
                start = time.time()
                instr.executor = computer
                res = await instr()
                instr.executor = None
                end = time.time()
                runtimes.append(end - start)
            start = time.time()
            self.program.post_op(pc, lp.EC.SUCCESS)
            end = time.time()
            post_op_time = end - start
        except Exception as e:
            traceback.print_exc()
            self.program.post_op(pc, lp.EC.EXCEPTION)
            self.loop.stop()
            raise
        e = time.time()

def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=2):
    loop = asyncio.get_event_loop()
    print("running loop")
    loop.create_task(check_program_state(program, loop))
    def wrapper(coro, loop):
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return
    executor = fs.ThreadPoolExecutor(pipeline_width)
    computer = fs.ThreadPoolExecutor(1)
    for i in range(pipeline_width):
        coro = lambdapack_run_async(loop, program, computer)
        loop.run_in_executor(executor, wrapper, coro, loop)
    res = loop.run_forever()
    print("loop end")
    loop.close()
    return 0

async def reset_msg_visibility(msg, queue_url, loop, timeout, lock):
    try:
        session = aiobotocore.get_session(loop=loop)
        sqs_client = session.create_client('sqs', use_ssl=False)
        while(lock.locked()):
            receipt_handle = msg["ReceiptHandle"]
            res = await sqs_client.change_message_visibility(VisibilityTimeout=timeout*2, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        receipt_handle = msg["ReceiptHandle"]
        print("Deleting message ", msg["Body"])
        await sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except Exception as e:
        print(e)
    return 0

async def check_program_state(program, loop):
    while(True):
        #TODO make this an s3 access as opposed to DD access since we don't *really need* atomicity here
        #TODO make this coroutine friendly
        s = program.program_status()
        if(s != lp.EC.RUNNING):
            break
        # DD is expensive so sleep alot
        await asyncio.sleep(10)
    print("Closing loop")
    loop.stop()

async def lambdapack_run_async(loop, program, computer, pipeline_width=1, msg_vis_timeout=10):
    print("RUN LAMBDA RUN ASYNC")
    session = aiobotocore.get_session(loop=loop)
    sqs_client = session.create_client('sqs', use_ssl=False)
    lmpk_executor = LambdaPackExecutor(program, loop)
    try:
        while(True):
            asyncio.sleep(1)
            messages = await sqs_client.receive_message(QueueUrl=program.queue_url, MaxNumberOfMessages=1)
            if ("Messages" not in messages):
                continue
            msg = messages["Messages"][0]
            pc = int(msg["Body"])
            lock = asyncio.Lock()
            await lock.acquire()
            coro = reset_msg_visibility(msg, program.queue_url, loop, msg_vis_timeout, lock)
            loop.create_task(coro)
            z = await lmpk_executor.run(pc, computer=computer)
            lock.release()
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise
    return

















