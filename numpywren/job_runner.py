
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
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LRUCache(object):
    def __init__(self, max_items=10):
        self.cache = {}
        self.key_access_times = {}
        self.max_items = max_items

    def __setitem__(self, key, value):
        if (key in self.cache):
            if (not np.allclose(value, self.cache[key])):
                print( bcolors.WARNING + "OMG THIS IS AN ERROR THIS SHOULDN'T HAPPEN")
        print("Key", key)
        print("new value", value)
        print("old value", self.cache.get(key))
        self.cache[key] = value
        self.key_access_times[key] = time.time()
        if (len(self.cache.keys()) > self.max_items):
            self.evict()


    def __getitem__(self, key):
        value = self.cache[key]
        self.key_access_times[key] = time.time()
        return value

    def __contains__(self, obj):
        return obj in self.cache

    def evict(self):
        key_to_evict = min(self.cache.keys(), key=lambda x: self.key_access_times[x])
        del self.cache[key_to_evict]
        del self.key_access_times[key_to_evict]






class LambdaPackExecutor(object):
    def __init__(self, program, loop, cache, read_lock, write_lock):
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.loop = loop
        self.program = program
        self.cache = cache
        self.read_lock = read_lock
        self.write_lock = write_lock
        self.block_ends= set()

    async def run(self, pc, computer=None):
        print("STARTING INSTRUCTION ", pc)
        program = self.program
        pcs_to_ret = []
        while pc != None:
            # hack copy program for every pc
            serializer = serialize.SerializeIndependent()
            byte_string = serializer([self.program])[0][0]
            program = pickle.loads(byte_string)

            t = time.time()
            node_status = program.get_node_status(pc)
            program.set_max_pc(pc)
            program.inst_blocks[pc].start_time = time.time()
            instrs = program.inst_blocks[pc].instrs
            next_pc = None
            try:
                if (node_status == lp.NS.READY):
                    for instr in instrs:
                        instr.executor = computer
                        instr.cache = self.cache
                        #instr.cache = None
                        instr.read_lock = self.read_lock
                        instr.write_lock = self.write_lock
                        res = await instr()
                        instr.cache = None
                        instr.executor = None
                        instr.read_lock = None
                        instr.write_lock = None
                    next_pc = program.post_op(pc, lp.PS.SUCCESS)
                    pcs_to_ret.append(pc)
                    pc = next_pc
                elif (node_status == lp.NS.POST_OP):
                    print("Re-running POST OP")
                    next_pc = program.post_op(pc, lp.PS.SUCCESS)
                    pcs_to_ret.append(pc)
                    pc = next_pc
                elif (node_status == lp.NS.NOT_READY):
                    print("THIS SHOULD NEVER HAPPEN OTHER THAN DURING TEST")
                    pc = None
                    continue
                elif (node_status == lp.NS.FINISHED):
                    print("HAHAH CAN'T TRICK ME... I'm just going to rage quit")
                    pc = None
                    continue
                else:
                    raise Exception("Unknown state")
            except RuntimeError as e:
                pass
            except fs._base.CancelledError as e:
                pass
            except Exception as e:
                traceback.print_exc()
                tb = traceback.format_exc()
                self.program.post_op(pc, lp.PS.EXCEPTION, tb=tb)
                raise
        e = time.time()
        return pcs_to_ret


def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=30, cache_size=20, timeout=200):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(check_program_state(program, loop))
    computer = fs.ThreadPoolExecutor(1)
    cache = LRUCache(max_items=cache_size)
    #cache = {}

    read_lock = asyncio.Lock()
    write_lock = asyncio.Lock()
    for i in range(pipeline_width):
        # all the async tasks share 1 compute thread and a io cache
        coro = lambdapack_run_async(loop, program, computer, cache, timeout=timeout, read_lock=read_lock, write_lock=write_lock)
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
        if(s != lp.PS.RUNNING):
            break
        # DD is expensive so sleep alot
        await asyncio.sleep(10)
    print("Closing loop")
    loop.stop()

async def lambdapack_run_async(loop, program, computer, cache, pipeline_width=1, msg_vis_timeout=10, timeout=200, read_lock=None, write_lock=None):
    session = aiobotocore.get_session(loop=loop)
    # every pipelined worker gets its own copy of program so we don't step on eachothers toes!
    serializer = serialize.SerializeIndependent()
    byte_string = serializer([program])[0][0]
    program = pickle.loads(byte_string)
    lmpk_executor = LambdaPackExecutor(program, loop, cache, read_lock, write_lock)
    start_time = time.time()

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
            pcs = await lmpk_executor.run(pc, computer=computer)
            lock.release()
            print("releasing lock")
            for _pc in pcs:
                program.set_node_status(_pc, lp.NS.FINISHED)
            async with session.create_client('sqs', use_ssl=False,  region_name='us-west-2') as sqs_client:
                print("Job done...Deleting message for {0}".format(pc))
                await sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            current_time = time.time()
            if (current_time - start_time > timeout):
                return
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise
    return

















