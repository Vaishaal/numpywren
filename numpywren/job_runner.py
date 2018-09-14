import asyncio
import concurrent.futures as fs
import gc
import logging
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread
import os
import pickle
import random
import sys
import time
import traceback
import tracemalloc
from multiprocessing import Process, Queue


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
                    self.program.set_node_status(expr_idx, var_values, lp.NS.RUNNING)
                    operator_refs_to_ret.append((expr_idx, var_values))
                    for instr in instrs:
                        instr.executor = computer
                        instr.cache = self.cache
                        if (instr.run):
                            e_str = ("EXCEPTION: Same machine replay instruction: " + str(instr) +
                                     " REF: {0}, time: {1}, pid: {2}".format((expr_idx, var_values), time.time(), os.getpid()))
                            raise Exception(e_str)
                        instr.start_time = time.time()
                        if (isinstance(instr, lp.RemoteWrite)):
                           res = await instr(self.program.block_sparse)
                        else:
                           res = await instr()
                        instr.end_time = time.time()
                        flops = int(instr.get_flops())
                        instr.flops = flops
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
                    instr.post_op_start = time.time()
                    #next_operator, log_bytes = self.program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    next_operator, log_bytes = await self.loop.run_in_executor(computer, self.program.post_op, expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    instr.post_op_end = time.time()
                    profile_bytes_to_ret.append(log_bytes)
                    profile_bytes_to_ret.append(log_bytes)
                    if next_operator is not None:
                         operator_refs.append(next_operator)
                elif (node_status == lp.NS.POST_OP):
                    operator_refs_to_ret.append((expr_idx, var_values))
                    logger.warning("node: {0}:{1} finished work skipping to post_op...".format(expr_idx, var_values))
                    next_operator, log_bytes = await self.loop.run_in_executor(computer, self.program.post_op, expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    #next_operator, log_bytes = self.program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
                    profile_bytes_to_ret.append(log_bytes)
                    if next_operator is not None:
                        operator_refs.append(next_operator)
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
    res = program.decr_up(1)
    print("======"*10)
    print("program decr up number", res)
    #logger.debug("Loop end program status: {0}".format(program.program_status()))
    return {"up_time": [lambda_start, lambda_stop],
            "exec_time": calculate_busy_time(shared_state["running_times"])}


PROGRAM_CHECK_INTERVAL = 10
def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=60, cache_size=5, timeout=200, idle_timeout=60, msg_vis_timeout_jitter=15, compute_threads=1):
    program.incr_up(1)
    lambda_start = time.time()
    compute_queue = Queue(pipeline_width)
    write_queue = Queue(pipeline_width)
    post_op_queue = Queue(pipeline_width)
    finish_queue = Queue(pipeline_width)
    log_queue = Queue()
    reader = Process(target=lambdapack_reader, args=(write_queue, post_op_queue))
    computer = Process(target=lambdapack_compute, args=(compute_queue, write_queue))
    writer = Process(target=lambdapack_writer, args=(write_queue, post_op_queue))
    post_op = Process(target=lambdapack_post_op, args=(post_op_queue, finish_queue, log_queue))
    reader.start()
    computer.start()
    writer.start()
    post_op.start()
    sqs_client = boto3.client('sqs', use_ssl=False, region_name=program.control_plane.region)
    all_msgs = []
    msg_info_map = {}
    finisher = Thread(target=lambdapack_finish, args=(finish_queue))
    last_p_check = 0
    while (time.time() < lambda_start + timeout):
       if ((time.time() - last_p_check) > PROGRAM_CHECK_INTERVAL):
          s = program.program_status()
          last_p_check = time.time()
          if(s != lp.PS.RUNNING):
            print("program status is ", s)
            with finished.get_lock():
               finished.value = True
            break
       messages = sqs_client.receive_message(QueueUrl=program.queue_urls[0], MaxNumberOfMessages=1)
       if ("Messages" not in messages):
          time.sleep(1)
          continue
       try:
          operator_ref = json.loads(msg["Body"])
        except ValueError:
            tb = traceback.format_exc()
            traceback.print_exc()
            self.handle_exception("JSON_PARSING_EXCEPTION", tb=tb, expr_idx=-1, var_values={})
            raise

         msg = messages["Messages"][0]
         stop = [0]
         msg_stop_map[(expr_idx, str(var_values)] = (program.queue_urls[0], stop, msg)
         t = Thread(target=reset_msg_visibility, args=(msg_vis_timeout, queue_url, receipt_handle, stop))
         t.start()
         expr_idx, var_values = operator_ref
         node_status = self.program.get_node_status(expr_idx, var_values)
         inst_block = self.program.program.eval_expr(expr_idx, var_values)
         if (node_status == lp.NS.READY or node_status == lp.NS.RUNNING):
            #start off with read
            read_queue.put((expr_idx, var_values, inst_block, 0))
         elif (node_status == lp.NS.POST_OP):
            # enqueue directly onto post_op queue
            post_op_queue.put(inst_block)
         elif (node_status == lp.NS.NOT_READY):
            logger.warning("node: {0}:{1} not ready skipping...".format(expr_idx, var_values))
            pass
         elif (node_status == lp.NS.FINISHED):
            logger.warning("node: {0}:{1} finished post_op skipping...".format(expr_idx, var_values))
         else:
            raise Exception("Unknown status: {0}".format(node_status))
    profiled_inst_blocks = {}
    while (not log_queue.empty):
      expr_idx, var_values, log = log_queue.get()
    profiled_inst_blocks[(expr_idx, str(var_values))] = inst_block

    profile_bytes = pickle.dumps(logs)
    m.update(profile_bytes)
    # double store the logs because they are *SO GOD DAMN IMPORTANT*
    p_key = m.hexdigest()
    p_key = "{0}/{1}/{2}".format("lambdapack", program.hash, p_key)
    client = boto3.client('s3', region_name=program.control_plane.region)
    client.put_object(Bucket=program.bucket, Key=p_key, Body=profile_bytes)
    res = program.decr_up(1)
    lambda_stop = time.time()
    return {"up_time": [lambda_start, lambda_stop],
            "exec_time": calculate_busy_time(shared_state["running_times"]),
            "executed_messages": shared_state["tot_messages"],
            "operator_refs": shared_state["all_operator_refs"]
            "logs":  profiled_inst_blocks}

def lambdapack_read(read_queue, compute_queue, program, finished):
   # do read and then pass on to compute_queue
   while(True):
      expr_idx, var_values, inst_block, i = read_queue.get()
      assert i == 0
      for i,instr in enumerate(instrs):
         if (isinstance(instr, lp.RemoteRead)):
            instr()
         elif (isinstance(instr, lp.RemoteWrite)):
            assert False
         elif (isinstance(instr, lp.RemoteCall)):
            #serialize and send forward
            compute_queue.put((expr_idx, var_values, inst_block, i))
            break

def lambdapack_compute(compute_queue, write_queue, program, finished):
   while(True):
      expr_idx, var_values, inst_block, i = compute_queue.get()
      assert(isinstance(inst_block.instrs[i], lp.RemoteCall))
      try:
         inst_block.instrs[i]()
      except:
         tb = traceback.format_exc()
         traceback.print_exc()
         program.handle_exception("COMPUTE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
         with finished.get_lock():
            finished.value = True
      assert len(inst_block.instrs) > i + 1
      assert isinstance(inst_block.instrs[i+1], lp.RemoteWrite)
      write_queue.put((expr_idx, var_values, inst_block, i+1))

def lambdapack_write(write_queue, post_op_queue):
   while(True):
      expr_idx, var_values, inst_block, i = write_queue.get()
      for (j in range(i, len(inst_block.instrs))):
         assert(isinstance(inst_block.instrs[j], lp.RemoteWrite))
         try:
            inst_block.instrs[j]()
         except:
            tb = traceback.format_exc()
            traceback.print_exc()
            program.handle_exception("COMPUTE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
            with finished.get_lock():
               finished.value = True
      post_op_queue.put((expr_idx, var_values, inst_block, j))


def lambdapack_post_op(post_op_queue, finish_queue):
   while(True):
      expr_idx, var_values, inst_block, i = post_op_queue.get()
      assert(len(inst_block) == i + 1)
      try:
         next_operator, log_bytes = self.program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
         finish_queue.put((next_operator, log_bytes))
      except:
         program.handle_exception("WRITE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
         with finished.get_lock():
            finished.value = True


def reset_msg_visibility(msg, msg_vis_timeout, queue_url, stop):
    while(True):
        try:
            receipt_handle = msg["ReceiptHandle"]
            operator_ref = tuple(json.loads(msg["Body"]))
            sqs_client = boto3.client('sqs')
            res = sqs_client.change_message_visibility(VisibilityTimeout=msg_vis_timeout, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            time.sleep((msg_vis_timeout/2))
            if (stop[0]):
               break
        except Exception as e:
            print("PC: {0} Exception in reset msg vis ".format(operator_ref) + str(e))
            time.sleep(msg_vis_timeout/2)
    return 0

def lambdapack_finish(finish_queue, msg_info_map):
   while (True):
      expr_idx, var_values, inst_block, i = finish_queue.get()
      print("Marking {0} as done".format((expr_idx, var_values)))
      program.set_node_status(expr_idx, var_values, lp.NS.FINISHED)
      queue_url, msg, stop = msg_info_map[(expr_idx, str(var_values)]
      stop[0] = True
      receipt_handle = msg["ReceiptHandle"]
      sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)





















