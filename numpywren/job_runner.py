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
from multiprocessing import Process, Queue, Manager, Value
#from queue import Queue
import queue

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

PROGRAM_CHECK_INTERVAL = 10

def lambdapack_run(program, pipeline_width=1, msg_vis_timeout=60, cache_size=5, timeout=200, idle_timeout=60, msg_vis_timeout_jitter=15, compute_threads=1):
    program.incr_up(1)
    lambda_start = time.time()
    read_queue = Queue(pipeline_width)
    compute_queue = Queue(pipeline_width)
    write_queue = Queue(pipeline_width)
    post_op_queue = Queue(pipeline_width*2)
    finish_queue = Queue(pipeline_width*2)
    log_queue = Queue()
    manager = Manager()
    msg_info_map  = manager.dict()
    finished = Value('i', 0)
    deadline = lambda_start + timeout
    Thread = Process
    reader = Thread(target=lambdapack_read, args=(read_queue, compute_queue, program, finished, deadline), daemon=True)
    computer = Thread(target=lambdapack_compute, args=(compute_queue, write_queue, program, finished), daemon=True)
    writer = Thread(target=lambdapack_write, args=(write_queue, post_op_queue, program, finished, deadline), daemon=True)
    post_op = Thread(target=lambdapack_post_op, args=(post_op_queue, finish_queue, log_queue, program, finished), daemon=True)
    reader.start()
    computer.start()
    writer.start()
    post_op.start()
    sqs_client = boto3.client('sqs', use_ssl=False, region_name=program.control_plane.region)
    all_msgs = []
    finisher = Thread(target=lambdapack_finish, args=(finish_queue, msg_info_map, program, finished), daemon=True)
    finisher.start()
    last_p_check = 0
    queue_url = program.queue_urls[0]
    print("starting job runner...")
    while (True):
       reset_msg_visibility(msg_info_map, msg_vis_timeout, queue_url)
       if (finished.value):
          break
       if (time.time() >= lambda_start + timeout):
          print("job runner process timeout....")
          finished.value = True
          break
       if ((time.time() - last_p_check) > PROGRAM_CHECK_INTERVAL):
          s = program.program_status()
          last_p_check = time.time()
          if(s != lp.PS.RUNNING):
            #print("program status is ", s)
            finished.value = True
            break
       if (not read_queue.full()):
          try:
            messages = sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
          except:
             break
          #TODO: Multiple prioritiy queues
          queue_url = program.queue_urls[0]
          if ("Messages" not in messages):
             print("no messages...")
             time.sleep(1)
             continue
          else:
             msg = messages["Messages"][0]

          try:
             operator_ref = json.loads(msg["Body"])
          except ValueError:
               tb = traceback.format_exc()
               traceback.print_exc()
               program.handle_exception("JSON_PARSING_EXCEPTION", tb=tb, expr_idx=-1, var_values={})
               raise

          msg = messages["Messages"][0]
          print("running msg ", msg)
          expr_idx, var_values = operator_ref
          print("Expr IDX", expr_idx)
          print("VAR VALUES", var_values)
          msg_info_map[(expr_idx, str(var_values))] = (msg, queue_url, time.time())
          node_status = program.get_node_status(expr_idx, var_values)
          inst_block = program.program.eval_expr(expr_idx, var_values)
          inst_block.start_time = time.time()
          print("node_status", node_status)
          if (node_status == lp.NS.READY or node_status == lp.NS.RUNNING):
               print("adding to read_queue")
               try:
                  read_queue.put((expr_idx, var_values, inst_block, 0), timeout=1, block=False)
               except queue.Full:
                  continue

               print("added to read_queue")
          elif (node_status == lp.NS.POST_OP):
               print("skipping directly to post op")
               try:
                  post_op_queue.put((expr_idx, var_values, inst_block, len(inst_block.instrs) - 1))
                  full_queue = False
               except queue.Full:
                  full_queue = True
                  continue
          elif (node_status == lp.NS.NOT_READY):
               print("node: {0}:{1} not ready skipping...".format(expr_idx, var_values))
               pass
          elif (node_status == lp.NS.FINISHED):
               print("node: {0}:{1} finished post_op skipping...".format(expr_idx, var_values))
          else:
               raise Exception("Unknown status: {0}".format(node_status))
    finished.value= True
    print("LOG QUEUE SIZE", log_queue.qsize())
    profiled_inst_blocks = {}
    while (True):
      try:
         expr_idx, var_values, inst_block = log_queue.get(block=False)
         profiled_inst_blocks[(expr_idx, str(var_values))] = inst_block
      except queue.Empty:
         print("log queue empty...")
         break
      except:
         print("some other exception occurred!!!")
         break
    try:
      profile_bytes = pickle.dumps(profiled_inst_blocks)
    except:
       print("PROFILE BYTES PICKLE FAILED..")

    print("number of profiled inst blocks", len(profiled_inst_blocks))
    m = hashlib.md5()
    m.update(profile_bytes)
    p_key = m.hexdigest()
    p_key = "{0}/{1}/{2}".format("lambdapack", program.hash, p_key)
    client = boto3.client('s3', region_name=program.control_plane.region)
    client.put_object(Bucket=program.bucket, Key=p_key, Body=profile_bytes)
    res = program.decr_up(1)
    lambda_stop = time.time()
    return {"up_time": [lambda_start, lambda_stop],
            "logs":  profiled_inst_blocks}
    return

def lambdapack_read(read_queue, compute_queue, program, finished, deadline):
   #print("READER STARTED..")
   # do read and then pass on to compute_queue
   while(True):
      t = time.time()
      val = read_queue.get()
      if val == -1:
         return
      e = time.time()
      expr_idx, var_values, inst_block, i = val
      inst_block.read_queue_start = t
      inst_block.read_queue_end = e
      inst_block.read_start = time.time()
      assert i == 0
      if (time.time() >= deadline):
         print("job runner process timeout....")
         finished.value = True
         break

      for i,instr in enumerate(inst_block.instrs):
         if (isinstance(instr, lp.RemoteRead)):
            try:
               instr()
               program.incr_read(instr.size)
            except:
               raise
               tb = traceback.format_exc()
               traceback.print_exc()
               program.handle_exception("READ EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
               finished.value= True
         elif (isinstance(instr, lp.RemoteWrite)):
            assert False
         elif (isinstance(instr, lp.RemoteCall)):
            #serialize and send forward
            inst_block.read_end = time.time()
            compute_queue.put((expr_idx, var_values, inst_block, i))
            break

def lambdapack_compute(compute_queue, write_queue, program, finished):
   #print("COMPUTER STARTED..")
   while(True):
      t = time.time()
      val = compute_queue.get()
      e = time.time()
      if val == -1:
         #print("compute returns")
         return
      expr_idx, var_values, inst_block, i = val
      inst_block.compute_queue_start = t
      inst_block.compute_queue_end = e
      #print(f"{expr_idx}, {var_values}, {inst_block}[{i}] in COMPUTE")
      assert(isinstance(inst_block.instrs[i], lp.RemoteCall))
      try:
         inst_block.compute_start = time.time()
         inst_block.instrs[i]()
         inst_block.compute_end = time.time()
         program.incr_flops(int(inst_block.instrs[i].get_flops()))
      except:
         raise
         tb = traceback.format_exc()
         traceback.print_exc()
         program.handle_exception("COMPUTE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
         finished.value = True
      assert len(inst_block.instrs) > i + 1
      assert isinstance(inst_block.instrs[i+1], lp.RemoteWrite)
      write_queue.put((expr_idx, var_values, inst_block, i+1))

def lambdapack_write(write_queue, post_op_queue, program, finished, deadline):
   #print("WRITER STARTED..")
   while(True):
      t = time.time()
      val = write_queue.get()
      e = time.time()
      if val == -1:
         #print("write returning")
         return
      expr_idx, var_values, inst_block, i = val
      inst_block.write_queue_start = t
      inst_block.write_queue_end = e
      inst_block.write_start = time.time()
      #print(f"{expr_idx}, {var_values}, {inst_block}[{i}] in WRITE")
      for j in range(i, len(inst_block.instrs)):
         assert(isinstance(inst_block.instrs[j], lp.RemoteWrite))
         try:
            inst_block.instrs[j]()
            program.incr_write(inst_block.instrs[j].size)
            if (time.time() >= deadline):
               print("job runner process timeout....")
               finished.value = True
               break
         except:
            raise
            tb = traceback.format_exc()
            traceback.print_exc()
            program.handle_exception("WRITE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
            finished.value = True
      inst_block.write_end = time.time()
      post_op_queue.put((expr_idx, var_values, inst_block, len(inst_block.instrs) - 1))



def lambdapack_post_op(post_op_queue, finish_queue, log_queue, program, finished):
   #print("POSTOP STARTED..")
   while(True):
      t = time.time()
      val = post_op_queue.get()
      e = time.time()
      if val == -1:
         #print("post_op returning")
         return
      expr_idx, var_values, inst_block, i = val
      inst_block.post_op_queue_start = t
      inst_block.post_op_queue_end = e
      #print(f"{expr_idx}, {var_values}, {inst_block}[{i}] in POST_OP")
      assert(len(inst_block.instrs) - 1 == i)
      try:
         inst_block.post_op_start = time.time()
         next_operator, profiled_block = program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
         inst_block.post_op_end = time.time()
         print("post_op took", inst_block.post_op_end - inst_block.post_op_start)
         finish_queue.put((expr_idx, var_values, inst_block, i))
         #print("submitting log object...")
         log_queue.put((expr_idx, var_values, profiled_block))
      except:
         program.handle_exception("post op EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
         finished.value = True


def reset_msg_visibility(msg_info_map, msg_vis_timeout, queue_url):
   for key, val in list(msg_info_map.items()):
        try:
            last_reset_time = val[-1]
            msg = val[0]
            queue_url = val[1]
            curr_time = time.time()
            if (last_reset_time - curr_time  > msg_vis_timeout/3):
               operator_ref = tuple(json.loads(msg["Body"]))
               if (finished.value):
                  #print(f"reset msg for {operator_ref} finished quitting..")
                  break
               receipt_handle = msg["ReceiptHandle"]
               sqs_client = boto3.client('sqs')
               res = sqs_client.change_message_visibility(VisibilityTimeout=msg_vis_timeout, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
               msg_info_map[msg] = (msg, queue_url, time.time())
               time.sleep((msg_vis_timeout/2))
        except Exception as e:
            finished.value = True
   return 0

def lambdapack_finish(finish_queue, msg_info_map, program, finished):
   try:
      sqs_client = boto3.client('sqs', region_name='us-west-2')
      while (True):
         if (finished.value):
            break
         val = finish_queue.get()
         if (val == -1):
            return
         expr_idx, var_values, inst_block, i = val
         inst_block.finish_start = time.time()
         assert(len(inst_block.instrs) == i + 1)
         print("Marking {0} as done".format((expr_idx, var_values)))
         program.set_node_status(expr_idx, var_values, lp.NS.FINISHED)
         msg, queue_url, reset_time = msg_info_map[(expr_idx, str(var_values))]
         receipt_handle = msg["ReceiptHandle"]
         try:
            del msg_info_map[(expr_idx, str(var_values))]
         except KeyError:
            print("key error..")
            pass
         sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
         inst_block.finish_end = time.time()
   except:
      finished.value = True
      raise






















