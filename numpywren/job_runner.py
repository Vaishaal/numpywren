
import asyncio
import aiobotocore
import io
import numpy as np
import boto3
import concurrent.futures as fs
import time
import pywren
from numpywren import lambdapack as lp
import traceback
from multiprocessing.dummy import Pool as ThreadPool
import logging


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
        print("STARTING INSTRUCTION")
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
            tb = traceback.format_exc()
            self.program.post_op(pc, lp.EC.EXCEPTION, tb=tb)
            self.loop.stop()
            raise
        e = time.time()

def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=30):
    logging.basicConfig(level=logging.DEBUG)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(check_program_state(program, loop))
    computer = fs.ThreadPoolExecutor(1)
    for i in range(pipeline_width):
        coro = lambdapack_run_async(loop, program, computer)
        loop.create_task(coro)
    res = loop.run_forever()
    print("loop end")
    loop.close()
    return 0

async def reset_msg_visibility(msg, queue_url, loop, timeout, lock):
    try:
        session = aiobotocore.get_session(loop=loop)
        while(lock.locked()):
            receipt_handle = msg["ReceiptHandle"]
            async with session.create_client('sqs', use_ssl=False,  region_name='us-west-2') as sqs_client:
                res = await sqs_client.change_message_visibility(VisibilityTimeout=30, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                await asyncio.sleep(10)
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
    session = aiobotocore.get_session(loop=loop)
    lmpk_executor = LambdaPackExecutor(program, loop)
    try:
        while(True):
            await asyncio.sleep(1)
            # go from high priority -> low priority
            for queue_url in program.queue_urls[::-1]:
                async with session.create_client('sqs', use_ssl=False,  region_name='us-west-2') as sqs_client:
                    messages = await sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
                if ("Messages" not in messages):
                    continue
                else:
                    # note for loops in python leak scope so when this breaks
                    # messages = messages
                    # queue_url= messages
                    break
            if ("Messages" not in messages):
                continue
            msg = messages["Messages"][0]
            receipt_handle = msg["ReceiptHandle"]
            pc = int(msg["Body"])
            print("creating lock")
            lock = asyncio.Lock()
            await lock.acquire()
            print("got locklock")
            coro = reset_msg_visibility(msg, queue_url, loop, msg_vis_timeout, lock)
            loop.create_task(coro)
            z = await lmpk_executor.run(pc, computer=computer)
            lock.release()
            print("releasing lock")
            async with session.create_client('sqs', use_ssl=False,  region_name='us-west-2') as sqs_client:
                await sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise
    return

















