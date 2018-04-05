
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
from pywren.serialize import serialize
import pickle
import gc


class LRUCache(object):
    def __init__(self, max_items=10):
        self.cache = {}
        self.key_order = []
        self.max_items = max_items

    def __setitem__(self, key, value):
        self.cache[key] = value
        self._mark(key)

    def __getitem__(self, key):
        try:
            value = self.cache[key]
        except KeyError:
            # Explicit reraise for better tracebacks
            raise KeyError
        self._mark(key)
        return value

    def __contains__(self, obj):
        return obj in self.cache

    def _mark(self, key):
        if key in self.key_order:
            self.key_order.remove(key)

        self.key_order.insert(0, key)
        if len(self.key_order) > self.max_items:
            remove = self.key_order[self.max_items]
            del self.cache[remove]
            for c in range(10):
                gc.collect()
            self.key_order.remove(remove)

class LambdaPackExecutor(object):
    def __init__(self, program, loop, cache):
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.loop = loop
        self.program = program
        self.cache = cache
        self.block_ends= set()

    async def run(self, pc, computer=None):
        print("STARTING INSTRUCTION ", pc)
        pcs = [pc]
        for pc in pcs:
            t = time.time()
            node_status = self.program.get_node_status(pc)
            self.program.set_max_pc(pc)
            self.program.inst_blocks[pc].start_time = time.time()
            instrs = self.program.inst_blocks[pc].instrs
            next_pc = None
            try:
                if (node_status == lp.NS.READY or node_status == lp.NS.RUNNING):
                    for instr in instrs:
                        instr.executor = computer
                        instr.cache = self.cache
                        res = await instr()
                        instr.cache = None
                        instr.executor = None
                    next_pc = self.program.post_op(pc, lp.PS.SUCCESS)
                elif (node_status == lp.NS.POST_OP):
                    print("Re-running POST OP")
                    next_pc = self.program.post_op(pc, lp.PS.SUCCESS)
                elif (node_status == lp.NS.NOT_READY):
                    print("THIS SHOULD NEVER HAPPEN OTHER THAN DURING TEST")
                    continue
                elif (node_status == lp.NS.FINISHED):
                    print("HAHAH CAN'T TRICK ME... I'm just going to rage quit")
                    continue
                else:
                    raise Exception("Unknown state")
                if (next_pc != None):
                    pcs.append(next_pc)
            except fs._base.TimeoutError as e:
                self.program.decr_up(1)
                raise
            except Exception as e:
                self.program.decr_up(1)
                traceback.print_exc()
                tb = traceback.format_exc()
                self.program.post_op(pc, lp.PS.EXCEPTION, tb=tb)
                raise
        e = time.time()

def calculate_busy_time(rtimes):
    pairs = [[(item[0], 1), (item[1], -1)] for sublist in rtimes for item in sublist]
    events = sorted([item for sublist in pairs for item in sublist])
    running = 0
    wtimes = []
    current_start = 0
    for event in events:
        if running == 0 and event[1] == 1:
            current_start = event[0]
        if running == 1 and event[1] == -1:
            wtimes.append([current_start, event[0]])
        running += event[1]
    return wtimes


def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=30, cache_size=5, timeout=200):
    lambda_start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    #loop.create_task(check_program_state(program, loop))
    computer = fs.ThreadPoolExecutor(1)
    cache = LRUCache(max_items=cache_size)
    tasks = []
    for i in range(pipeline_width):
        # all the async tasks share 1 compute thread and a io cache
        coro = lambdapack_run_async(loop, program, computer, cache, timeout=timeout)
        tasks.append(loop.create_task(coro))
    results = loop.run_until_complete(asyncio.gather(*tasks))
    print("loop end")
    loop.close()
    lambda_stop = time.time()
    program.decr_up(1)
    return {"up_time": [lambda_start, lambda_stop],
            "exec_time": calculate_busy_time(results)}

async def reset_msg_visibility(msg, queue_url, loop, timeout, lock):
    try:
        session = aiobotocore.get_session(loop=loop)
        while(lock.locked()):
            receipt_handle = msg["ReceiptHandle"]
            async with session.create_client('sqs', use_ssl=False,  region_name='us-west-2') as sqs_client:
                res = await sqs_client.change_message_visibility(VisibilityTimeout=3, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            await asyncio.sleep(1)
    except Exception as e:
        print(e)
    return 0

async def check_program_state(program, loop):
    while(True):
        #TODO make this an s3 access as opposed to DD access since we don't *really need* atomicity here
        #TODO make this coroutine friendly
        s = program.program_status()
        if(s != lp.PS.RUNNING):
            break
        # DD is expensive so sleep alot
        await asyncio.sleep(10)
    print("Closing loop")
    loop.stop()

async def lambdapack_run_async(loop, program, computer, cache, pipeline_width=1, msg_vis_timeout=10, timeout=200):
    session = aiobotocore.get_session(loop=loop)
    # every pipelined worker gets its own copy of program so we don't step on eachothers toes!
    serializer = serialize.SerializeIndependent()
    byte_string = serializer([program])[0][0]
    program = pickle.loads(byte_string)
    lmpk_executor = LambdaPackExecutor(program, loop, cache)
    start_time = time.time()
    running_times = []
    last_message_time = time.time()
    try:
        while(True):
            await asyncio.sleep(0)
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
                if time.time() - last_message_time > 10:
                    return running_times
                continue
            last_message_time = time.time()
            start_processing_time = time.time()
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
                print("Job done...Deleting message for {0}".format(pc))
                await sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            '''
            sqs_client = boto3.client('sqs')
            sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            '''
            end_processing_time = time.time()
            running_times.append((start_processing_time, end_processing_time))
            current_time = time.time()
            if (current_time - start_time > timeout):
                return running_times
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise
    return

















