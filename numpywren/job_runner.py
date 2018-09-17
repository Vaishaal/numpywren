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
    def __init__(self, program, loop, cache, read_queue):
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.loop = loop
        self.program = program
        self.cache = cache
        self.block_ends= set()
        self.read_queue = read_queue

    #@profile
    async def run(self, expr_idx, var_values, computer=None, profile=True):
        operator_refs = [(expr_idx, var_values)]
        event = asyncio.Event()
        operator_refs_to_ret = []
        profile_bytes_to_ret = []
        for expr_idx, var_values in operator_refs:
            try:
               t = time.time()
               node_status = self.program.get_node_status(expr_idx, var_values)
               inst_block = self.program.program.eval_expr(expr_idx, var_values)
               inst_block.start_time = time.time()
               print(f"Running JOB={(expr_idx, var_values)}")
               instrs = inst_block.instrs
               next_operator = None
            except:
               tb = traceback.format_exc()
               traceback.print_exc()
               self.program.handle_exception("EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
               raise

            if (len(instrs) != len(set(instrs))):
                raise Exception("Duplicate instruction in instruction stream")
            try:
                if (node_status == lp.NS.READY or node_status == lp.NS.RUNNING):
                    if (node_status == lp.NS.RUNNING):
                       program.incr_repeated_compute()

                    self.program.set_node_status(expr_idx, var_values, lp.NS.RUNNING)
                    operator_refs_to_ret.append((expr_idx, var_values))
                    print("adding to read queue")
                    await self.read_queue.put((expr_idx, var_values, inst_block, 0, event))
                    print("added to read queue")
                    await event.wait()
                    for instr in instrs:
                        instr.run = False
                        instr.result = None
                    instr.post_op_start = time.time()
                    next_operator, log_bytes = await self.loop.run_in_executor(computer, self.program.post_op, expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    #next_operator, log_bytes = self.program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    instr.post_op_end = time.time()
                    profile_bytes_to_ret.append(log_bytes)
                    if next_operator is not None:
                         operator_refs.append(next_operator)
                elif (node_status == lp.NS.POST_OP):
                    program.incr_repeated_post_op()
                    operator_refs_to_ret.append((expr_idx, var_values))
                    logger.warning("node: {0}:{1} finished work skipping to post_op...".format(expr_idx, var_values))
                    next_operator, log_bytes = await self.loop.run_in_executor(computer, self.program.post_op, expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    profile_bytes_to_ret.append(log_bytes)
                    if next_operator is not None:
                        operator_refs.append(next_operator)
                elif (node_status == lp.NS.NOT_READY):
                   program.not_ready_incr()
                   logger.warning("node: {0}:{1} not ready skipping...".format(expr_idx, var_values))

                   continue
                elif (node_status == lp.NS.FINISHED):
                   program.repeated_finish_incr()
                   logger.warning("node: {0}:{1} finished post_op skipping...".format(expr_idx, var_values))
                   continue
                else:
                    raise Exception("Unknown status: {0}".format(node_status))
                if next_operator is not None:
                    operator_refs.append(next_operator)
            except fs._base.TimeoutError as e:
                self.program.decr_up(1)
                raise
            except RuntimeError as e:
                self.program.decr_up(1)
                raise
            except Exception as e:
                instr.run = True
                instr.cache = None
                instr.executor = None
                self.program.decr_up(1)
                traceback.print_exc()
                tb = traceback.format_exc()
                self.program.post_op(expr_idx, var_values, lp.PS.EXCEPTION, inst_block, tb=tb)
                raise
        e = time.time()
        res = list(zip(operator_refs_to_ret, profile_bytes_to_ret))
        #print('======\n'*10)
        #print("operator refs", operator_refs_to_ret)
        #print("profile bytes to ret", profile_bytes_to_ret)
        #print("RETURING ", res)
        #print('======\n'*10)
        return res


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


def lambdapack_run_with_failures(failure_key, program, pipeline_width=5, msg_vis_timeout=60, cache_size=0, timeout=200, idle_timeout=60, msg_vis_timeout_jitter=15, compute_threads=1):
    program.incr_up(1)
    lambda_start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    computer = fs.ThreadPoolExecutor(compute_threads)
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

async def read(read_queue, compute_queue, program):
   while (True):
      print("started reader..")
      val = await read_queue.get()
      print("got a value")
      print(val)
      expr_idx, var_values, inst_block, i, event = val
      print("here")
      assert i == 0
      for i, instr in enumerate(inst_block.instrs):
         #print("inst_block, i", inst_block, i)
         if (not isinstance(instr, lp.RemoteRead)):
            break
         else:
            try:
               await instr()
               read_size = instr.read_size
               program.incr_read(read_size)
            except:
                print("EXCEPTION")
                instr.run = True
                instr.cache = None
                instr.executor = None
                program.decr_up(1)
                traceback.print_exc()
                tb = traceback.format_exc()
                self.program.handle_exception("READ_EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
                raise

      await compute_queue.put((expr_idx, var_values, inst_block, i, event))
      await asyncio.sleep(0)

async def compute(compute_queue, write_queue, program):
   print("started computer..")
   while (True):
      val = await compute_queue.get()
      expr_idx, var_values, inst_block, i, event = val
      assert isinstance(inst_block.instrs[i], lp.RemoteCall)
      instr = inst_block.instrs[i]
      try:
         await instr()
         flops = int(instr.get_flops())
         program.incr_flops(flops)
         print("finished compute")
      except:
          instr.run = True
          instr.cache = None
          instr.executor = None
          traceback.print_exc()
          tb = traceback.format_exc()
          program.handle_exception("COMPUTE EXCEPITION", tb=tb, expr_idx=expr_idx, var_values=var_values)
          raise
      print("adding something to write queue")
      await write_queue.put((expr_idx, var_values, inst_block, i+1, event))
      await asyncio.sleep(0)

async def write(write_queue, program):
   while (True):
      print("started writer..")
      val = await write_queue.get()
      print("got write value")
      expr_idx, var_values, inst_block, i, event = val
      print('got here')
      for i in range(i, len(inst_block.instrs)):
         assert isinstance(inst_block.instrs[i], lp.RemoteWrite)
         instr = inst_block.instrs[i]
         print('here here')
         try:
            await instr()
            write_size = instr.write_size
            program.incr_write(write_size)
         except:
             instr.run = True
             instr.cache = None
             instr.executor = None
             traceback.print_exc()
             tb = traceback.format_exc()
             program.handle_exception("COMPUTE EXCEPITION", tb=tb, expr_idx=expr_idx, var_values=var_values)
             raise
      event.set()
      await asyncio.sleep(0)





#@profile
def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=60, cache_size=5, timeout=200, idle_timeout=60, msg_vis_timeout_jitter=15, compute_threads=1):
    program.incr_up(1)
    lambda_start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    computer = fs.ThreadPoolExecutor(compute_threads)
    program.control_plane.cache()

    read_queue = asyncio.Queue(pipeline_width)
    compute_queue = asyncio.Queue(pipeline_width)
    write_queue = asyncio.Queue(pipeline_width)
    post_op_queue = asyncio.Queue(pipeline_width)

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
    loop.create_task(read(read_queue, compute_queue, program))
    loop.create_task(compute(compute_queue, write_queue, program))
    loop.create_task(write(write_queue, program))

    tasks = []
    for i in range(pipeline_width):
        # all the async tasks share 1 compute thread and a io cache
        coro = lambdapack_run_async(loop, program, computer, cache, shared_state=shared_state, timeout=timeout, msg_vis_timeout=msg_vis_timeout, msg_vis_timeout_jitter=msg_vis_timeout_jitter, read_queue=read_queue)
        tasks.append(loop.create_task(coro))
    loop.run_forever()
    print("loop end")
    loop.close()
    lambda_stop = time.time()
    profile_bytes = pickle.dumps(profiles)
    m.update(profile_bytes)
    p_key = m.hexdigest()
    p_key = "{0}/{1}/{2}".format("lambdapack", program.hash, p_key)
    #print('='*10)
    #print("PROFILES WAS ", profiles)
    #print("Writing log obj.... to {0}".format(p_key))
    #print('='*10)
    client = boto3.client('s3', region_name=program.control_plane.region)
    client.put_object(Bucket=program.bucket, Key=p_key, Body=profile_bytes)
    return {"up_time": [lambda_start, lambda_stop],
            "exec_time": calculate_busy_time(shared_state["running_times"]),
            "executed_messages": shared_state["tot_messages"],
            "operator_refs": shared_state["all_operator_refs"],
            "log" : profile_bytes}


async def reset_msg_visibility(msg, queue_url, loop, timeout, timeout_jitter, lock):
    assert(timeout > timeout_jitter)
    num_tries = 0
    while(lock[0] == 1):
        try:
            if (num_tries > 2):
               break
            receipt_handle = msg["ReceiptHandle"]
            operator_ref = tuple(json.loads(msg["Body"]))
            sqs_client = boto3.client('sqs')
            res = sqs_client.change_message_visibility(VisibilityTimeout=45, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            num_tries += 1
            await asyncio.sleep(50)

        except Exception as e:
            print("PC: {0} Exception in reset msg vis ".format(operator_ref) + str(e))
            await asyncio.sleep(10)
    operator_ref = tuple(json.loads(msg["Body"]))
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
        await asyncio.sleep(idle_timeout)
    #print("Closing loop from program")
    loop.stop()


#@profile
async def lambdapack_run_async(loop, program, computer, cache, shared_state, read_queue, pipeline_width=1, msg_vis_timeout=60, timeout=200, msg_vis_timeout_jitter=15):
    global REDIS_CLIENT
    session = aiobotocore.get_session(loop=loop)
    lmpk_executor = LambdaPackExecutor(program, loop, cache, read_queue)
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
                loop.stop()
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
            #res = sqs_client.change_message_visibility(VisibilityTimeout=1800, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            operator_ref = json.loads(msg["Body"])
            print("Operator ref", operator_ref)
            shared_state["tot_messages"].append(operator_ref)
            redis_client.set(msg["MessageId"], str(time.time()))
            #print("creating lock")
            lock = [1]
            coro = reset_msg_visibility(msg, queue_url, loop, msg_vis_timeout, msg_vis_timeout_jitter, lock)
            loop.create_task(coro)
            all_operator_refs = shared_state["all_operator_refs"]
            all_operator_refs = shared_state["all_operator_refs"]
            operator_refs= await lmpk_executor.run(*operator_ref, computer=computer)
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

















