import asyncio
import concurrent.futures as fs
import gc
import logging
from multiprocessing.dummy import Pool as ThreadPool
import os
import pickle
import random
import sys
import time
import traceback
import tracemalloc
import aiobotocore
import botocore
import boto3
import json
import numpy as np
from numpywren import lambdapack as lp
import pywren
from pywren.serialize import serialize
import redis
import sympy
import hashlib


REDIS_CLIENT = None
METADATABUCKET = "numpywrentop500testmetadata"
logger = logging.getLogger(__name__)

def mem():
   mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
   return mem_bytes/(1024.**3)

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

    #@profile
    async def run(self, expr_idx, var_values, computer=None, profile=True):
        operator_refs = [(expr_idx, var_values)]
        operator_refs_to_ret = []
        profile_bytes_to_ret = []
        for expr_idx, var_values in operator_refs:
            print("RUNNING, {0}, {1}".format(expr_idx, var_values))
            expr = self.program.program.get_expr(expr_idx)
            logger.debug("STARTING INSTRUCTION {0}, {1}, {2}".format(expr_idx, var_values,  expr))
            t = time.time()
            node_status = self.program.get_node_status(expr_idx, var_values)
            #print(node_status)
            operator_expr = self.program.program.get_expr(expr_idx)
            inst_block = operator_expr.eval_operator(var_values, hash=self.program.hash)
            inst_block.start_time = time.time()
            instrs = inst_block.instrs
            next_operator = None
            if (len(instrs) != len(set(instrs))):
                raise Exception("Duplicate instruction in instruction stream")
            try:
                if (node_status == lp.NS.READY or node_status == lp.NS.RUNNING):
                    self.program.set_node_status(expr_idx, var_values, lp.NS.RUNNING)
                    operator_refs_to_ret.append((expr_idx, var_values))
                    for instr in instrs:
                        instr.executor = computer
                        instr.cache = self.cache
                        #print("START: {0},  PC: {1}, time: {2}, pid: {3}".format(instr, pc, time.time(), os.getpid()))
                        if (instr.run):
                            e_str = ("EXCEPTION: Same machine replay instruction: " + str(instr) +
                                     " REF: {0}, time: {1}, pid: {2}".format((expr_idx, var_values), time.time(), os.getpid()))
                            raise Exception(e_str)
                        instr.start_time = time.time()
                        if (isinstance(instr, lp.RemoteRead)):
                              await self.program.begin_read()
                        if (isinstance(instr, lp.RemoteWrite)):
                              await self.program.begin_write()
                        if (isinstance(instr, lp.RemoteReturn) or isinstance(instr, lp.RemoteRead) or isinstance(instr, lp.RemoteWrite)):
                           res = await instr(redis_client=self.program.control_plane.client)
                        else:
                           res = await instr()
                        instr.end_time = time.time()
                        flops = int(instr.get_flops())
                        read_size = instr.read_size
                        write_size = instr.write_size
                        self.program.incr_flops(flops)
                        self.program.incr_read(read_size)
                        self.program.incr_write(write_size)
                        sys.stdout.flush()

                        instr.run = True
                        instr.cache = None
                        instr.executor = None
                    for instr in instrs:
                        instr.run = False
                        instr.result = None

                    next_operator, log_bytes = self.program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    if (next_operator is not None):
                     profile_bytes_to_ret.append(log_bytes)
                    profile_bytes_to_ret.append(log_bytes)
                elif (node_status == lp.NS.POST_OP):
                    operator_refs_to_ret.append((expr_idx, var_values))
                    logger.warning("node: {0}:{1} finished work skipping to post_op...".format(expr_idx, var_values))
                    next_operator, log_bytes = self.program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    if (next_operator is not None):
                     profile_bytes_to_ret.append(log_bytes)
                elif (node_status == lp.NS.NOT_READY):
                   logger.warning("node: {0}:{1} not ready skipping...".format(expr_idx, var_values))

                   continue
                elif (node_status == lp.NS.FINISHED):
                   logger.warning("node: {0}:{1} finished post_op skipping...".format(expr_idx, var_values))
                   continue
                else:
                    raise Exception("Unknown status: {0}".format(node_status))
                if next_operator is not None:
                    operator_refs.append(next_operator)
            except fs._base.TimeoutError as e:
                #self.program.decr_up(1)
                raise
            except RuntimeError as e:
                #self.program.decr_up(1)
                raise
            except Exception as e:
                instr.run = True
                instr.cache = None
                instr.executor = None
                #self.program.decr_up(1)
                traceback.print_exc()
                tb = traceback.format_exc()
                self.program.post_op(expr_idx, var_values, lp.PS.EXCEPTION, inst_block, tb=tb)
                raise
        e = time.time()
        return list(zip(operator_refs_to_ret, profile_bytes_to_ret))


def calculate_busy_time(rtimes):
    #pairs = [[(item[0], 1), (item[1], -1)] for sublist in rtimes for item in sublist]
    pairs = [[(item[0], 1), (item[1], -1)] for item in rtimes]
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


async def check_failure(loop, program, failure_key):
    global REDIS_CLIENT
    if (REDIS_CLIENT == None):
       REDIS_CLIENT = program.control_plane.client
    while (True):
      f_key = REDIS_CLIENT.get(failure_key)
      if (f_key is not None):
         logger.error("FAIL FAIL FAIL FAIL FAIL")
         logger.error(f_key)
         loop.stop()
      await asyncio.sleep(5)


def lambdapack_run_with_failures(failure_key, program, pipeline_width=5, msg_vis_timeout=60, cache_size=5, timeout=230, idle_timeout=230, msg_vis_timeout_jitter=15):
    program.incr_up(1)
    lambda_start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    computer = fs.ThreadPoolExecutor(1)
    if (cache_size > 0):
        cache = LRUCache(max_items=cache_size)
    else:
        cache = None
    shared_state = {}
    shared_state["busy_workers"] = 0
    shared_state["done_workers"] = 0
    shared_state["pipeline_width"] = pipeline_width
    shared_state["running_times"] = []
    shared_state["last_busy_time"] = time.time()
    loop.create_task(check_program_state(program, loop, shared_state, timeout, idle_timeout))
    loop.create_task(check_failure(loop, program, failure_key))
    tasks = []
    for i in range(pipeline_width):
        # all the async tasks share 1 compute thread and a io cache
        coro = lambdapack_run_async(loop, program, computer, cache, shared_state=shared_state, timeout=timeout, msg_vis_timeout=msg_vis_timeout, msg_vis_timeout_jitter=msg_vis_timeout_jitter)
        tasks.append(loop.create_task(coro))
    #results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.run_forever()
    loop.close()
    lambda_stop = time.time()
    program.decr_up(1)
    logger.debug("Loop end program status: {0}".format(program.program_status()))
    return {"up_time": [lambda_start, lambda_stop],
            "exec_time": calculate_busy_time(shared_state["running_times"])}

def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=60, cache_size=5, timeout=200, idle_timeout=200, msg_vis_timeout_jitter=15):
    program.incr_up(1)
    lambda_start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    computer = fs.ThreadPoolExecutor(1)
    program.control_plane.cache()
    tot_messages = []
    if (cache_size > 0):
        cache = LRUCache(max_items=cache_size)
    else:
        cache = None
    m = hashlib.md5()
    profiles = {}
    shared_state = {}
    shared_state["busy_workers"] = 0
    shared_state["done_workers"] = 0
    shared_state["pipeline_width"] = pipeline_width
    shared_state["all_operator_refs"] = []
    shared_state["profiles"] = profiles
    shared_state["running_times"] = []
    shared_state["last_busy_time"] = time.time()
    shared_state["tot_messages"]  = []
    loop.create_task(check_program_state(program, loop, shared_state, timeout, idle_timeout))
    tasks = []
    for i in range(pipeline_width):
        # all the async tasks share 1 compute thread and a io cache
        coro = lambdapack_run_async(loop, program, computer, cache, shared_state=shared_state, timeout=timeout, msg_vis_timeout=msg_vis_timeout, msg_vis_timeout_jitter=msg_vis_timeout_jitter)
        tasks.append(loop.create_task(coro))
    #results = loop.run_until_complete(asyncio.gather(*tasks))
    #return results
    loop.run_forever()
    print("loop end")
    loop.close()
    lambda_stop = time.time()
    profile_bytes = pickle.dumps(profiles)
    m.update(profile_bytes)
    p_key = m.hexdigest()
    p_key = "{0}/{1}/{2}".format("lambdapack", program.hash, p_key)
    client = boto3.client('s3', region_name=program.control_plane.region)
    backoff = 1
    while(True):
       try:
         new_loop = asyncio.new_event_loop()
         res = new_loop.run_until_complete(asyncio.ensure_future(program.begin_write(), loop=new_loop))
         client.put_object(Bucket=METADATABUCKET, Key=p_key, Body=profile_bytes)
         break
       except botocore.exceptions.ClientError:
         time.sleep(backoff)
         backoff *= 2
         pass
    program.decr_up(1)
    print(program.program_status())
    return {"up_time": [lambda_start, lambda_stop],
            "exec_time": calculate_busy_time(shared_state["running_times"]),
            "executed_messages": shared_state["tot_messages"],
            "operator_refs": shared_state["all_operator_refs"]}

async def reset_msg_visibility(msg, queue_url, loop, timeout, timeout_jitter, lock):
    assert(timeout > timeout_jitter)
    chances = 3
    while(lock[0] == 1 and chances > 0):
        try:
            receipt_handle = msg["ReceiptHandle"]
            operator_ref = tuple(json.loads(msg["Body"]))
            sqs_client = boto3.client('sqs')
            res = sqs_client.change_message_visibility(VisibilityTimeout=60, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            await asyncio.sleep(50)
            chances -= 1
        except Exception as e:
            print("PC: {0} Exception in reset msg vis ".format(operator_ref) + str(e))
            await asyncio.sleep(10)
    operator_ref = tuple(json.loads(msg["Body"]))
    print("Exiting msg visibility for {0}".format(operator_ref))
    return 0

async def check_program_state(program, loop, shared_state, timeout, idle_timeout):
    start_time = time.time()
    while(True):
        if shared_state["busy_workers"] == 0:
            if time.time() - start_time > timeout:
                break
            if time.time() - shared_state["last_busy_time"] >  idle_timeout:
                break
        #TODO make this an s3 access as opposed to DD access since we don't *really need* atomicity here
        #TODO make this coroutine friendly
        s = program.program_status()
        if(s != lp.PS.RUNNING):
           print("program status is ", s)
           break
        await asyncio.sleep(30)
    print("Closing loop from program")
    loop.stop()


#@profile
async def lambdapack_run_async(loop, program, computer, cache, shared_state, pipeline_width=1, msg_vis_timeout=60, timeout=200, msg_vis_timeout_jitter=15):
    global REDIS_CLIENT
    print("timeout is ", timeout)
    #print("LAMBDAPACK_RUN_ASYNC")
    session = aiobotocore.get_session(loop=loop)
    lmpk_executor = LambdaPackExecutor(program, loop, cache)
    start_time = time.time()
    running_times = shared_state['running_times']
    if (REDIS_CLIENT == None):
       REDIS_CLIENT = program.control_plane.client
    redis_client = REDIS_CLIENT
    try:
        while(True):
            current_time = time.time()
            if ((current_time - start_time) > timeout):
                print("Hit timeout...returning now")
                shared_state["done_workers"] += 1
                return
            await asyncio.sleep(0)
            # go from high priority -> low priority
            for queue_url in program.queue_urls[::-1]:
                async with session.create_client('sqs', use_ssl=False,  region_name=program.control_plane.region) as sqs_client:
                    messages = await sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
                if ("Messages" not in messages):
                    continue
                else:
                    # note for loops in python leak scope so when this breaks
                    # messages = messages
                    # queue_url= messages
                    break
            if ("Messages" not in messages):
                #if time.time() - last_message_time > 10:
                #    return running_times
                continue
            shared_state["busy_workers"] += 1
            redis_client.incr("{0}_busy".format(program.hash))
            #last_message_time = time.time()
            start_processing_time = time.time()
            msg = messages["Messages"][0]
            receipt_handle = msg["ReceiptHandle"]
            # if we don't finish in 75s count as a failure
            #res = sqs_client.change_message_visibility(VisibilityTimeout=100, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            operator_ref = json.loads(msg["Body"])
            shared_state["tot_messages"].append(operator_ref)
            operator_ref = (operator_ref[0], {sympy.Symbol(key): val for key, val in operator_ref[1].items()})
            redis_client.set(msg["MessageId"], str(time.time()))
            print("creating lock")
            lock = [1]
            coro = reset_msg_visibility(msg, queue_url, loop, msg_vis_timeout, msg_vis_timeout_jitter, lock)
            loop.create_task(coro)
            all_operator_refs = shared_state["all_operator_refs"]
            operator_refs = await lmpk_executor.run(*operator_ref, computer=computer)
            profiles = shared_state["profiles"]
            for operator_ref, p_info in operator_refs:
                logger.debug("Marking {0} as done".format(operator_ref))
                program.set_node_status(*operator_ref, lp.NS.FINISHED)
                all_operator_refs.append(operator_ref)
                profiles[str(operator_ref)] = p_info
            async with session.create_client('sqs', use_ssl=False,  region_name=program.control_plane.region) as sqs_client:
                lock[0] = 0
                await sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

            end_processing_time = time.time()
            running_times.append((start_processing_time, end_processing_time))
            shared_state["busy_workers"] -= 1
            redis_client.decr("{0}_busy".format(program.hash))
            shared_state["last_busy_time"] = time.time()
            current_time = time.time()
    except Exception as e:
        #print(e)
        traceback.print_exc()
        raise
    return

















