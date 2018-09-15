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
#from multiprocessing import Process
from queue import Queue
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
    post_op_queue = Queue(pipeline_width)
    finish_queue = Queue(pipeline_width)
    log_queue = Queue()
    finished = [0]
    reader = Thread(target=lambdapack_read, args=(read_queue, compute_queue, program, finished))
    computer = Thread(target=lambdapack_compute, args=(compute_queue, write_queue, program, finished))
    writer = Thread(target=lambdapack_write, args=(write_queue, post_op_queue, program, finished))
    post_op = Thread(target=lambdapack_post_op, args=(post_op_queue, finish_queue, log_queue, program, finished))
    reader.start()
    computer.start()
    writer.start()
    post_op.start()
    sqs_client = boto3.client('sqs', use_ssl=False, region_name=program.control_plane.region)
    all_msgs = []
    msg_info_map = {}
    finisher = Thread(target=lambdapack_finish, args=(finish_queue,msg_info_map, program, finished))
    finisher.start()
    last_p_check = 0
    queue_url = program.queue_urls[0]
    while (True):
       reset_msg_visibility(msg_info_map, msg_vis_timeout, queue_url)
       if (time.time() >= lambda_start + timeout):
          print("job runner process timeout....")
          break
       if ((time.time() - last_p_check) > PROGRAM_CHECK_INTERVAL):
          s = program.program_status()
          last_p_check = time.time()
          if(s != lp.PS.RUNNING):
            #print("program status is ", s)
            finished[0] = True
            break
       try:
         messages = sqs_client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
       except:
          break
       #TODO: Multiple prioritiy queues
       queue_url = program.queue_urls[0]
       if ("Messages" not in messages):
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
       #print("running msg ", msg)
       expr_idx, var_values = operator_ref
       #print("Expr IDX", expr_idx)
       #print("VAR VALUES", var_values)
       msg_info_map[(expr_idx, str(var_values))] = (msg, queue_url, time.time())
       node_status = program.get_node_status(expr_idx, var_values)
       inst_block = program.program.eval_expr(expr_idx, var_values)
       #print("node_status", node_status)
       if (node_status == lp.NS.READY or node_status == lp.NS.RUNNING):
            #print("adding to read_queue")
            read_queue.put((expr_idx, var_values, inst_block, 0))
            #print("added to read_queue")
       elif (node_status == lp.NS.POST_OP):
            post_op_queue.put(inst_block)
       elif (node_status == lp.NS.NOT_READY):
            logger.warning("node: {0}:{1} not ready skipping...".format(expr_idx, var_values))
            pass
       elif (node_status == lp.NS.FINISHED):
            logger.warning("node: {0}:{1} finished post_op skipping...".format(expr_idx, var_values))
       else:
            raise Exception("Unknown status: {0}".format(node_status))
    profiled_inst_blocks = {}
    finished[0]= True
    print("LOG QUEUE SIZE", log_queue.qsize())
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
    read_queue.put(-1)
    write_queue.put(-1)
    compute_queue.put(-1)
    post_op_queue.put(-1)
    finish_queue.put(-1)
    #print("returning..")
    return {"up_time": [lambda_start, lambda_stop],
            "logs":  profiled_inst_blocks}

def lambdapack_read(read_queue, compute_queue, program, finished):
   #print("READER STARTED..")
   # do read and then pass on to compute_queue
   while(True):
      val = read_queue.get()
      if val == -1:
         return
      expr_idx, var_values, inst_block, i = val
      assert i == 0
      for i,instr in enumerate(inst_block.instrs):
         if (isinstance(instr, lp.RemoteRead)):
            try:
               instr()
            except:
               tb = traceback.format_exc()
               traceback.print_exc()
               program.handle_exception("READ EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
               finished[0]= True
         elif (isinstance(instr, lp.RemoteWrite)):
            assert False
         elif (isinstance(instr, lp.RemoteCall)):
            #serialize and send forward
            compute_queue.put((expr_idx, var_values, inst_block, i))
            break

def lambdapack_compute(compute_queue, write_queue, program, finished):
   #print("COMPUTER STARTED..")
   while(True):
      val = compute_queue.get()
      if val == -1:
         #print("compute returns")
         return
      expr_idx, var_values, inst_block, i = val
      #print(f"{expr_idx}, {var_values}, {inst_block}[{i}] in COMPUTE")
      assert(isinstance(inst_block.instrs[i], lp.RemoteCall))
      try:
         inst_block.instrs[i]()
      except:
         tb = traceback.format_exc()
         traceback.print_exc()
         program.handle_exception("COMPUTE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
         finished[0]= True
      assert len(inst_block.instrs) > i + 1
      assert isinstance(inst_block.instrs[i+1], lp.RemoteWrite)
      write_queue.put((expr_idx, var_values, inst_block, i+1))

def lambdapack_write(write_queue, post_op_queue, program, finished):
   #print("WRITER STARTED..")
   while(True):
      val = write_queue.get()
      if val == -1:
         #print("write returning")
         return
      expr_idx, var_values, inst_block, i = val
      #print(f"{expr_idx}, {var_values}, {inst_block}[{i}] in WRITE")
      for j in range(i, len(inst_block.instrs)):
         assert(isinstance(inst_block.instrs[j], lp.RemoteWrite))
         try:
            inst_block.instrs[j]()
         except:
            tb = traceback.format_exc()
            traceback.print_exc()
            program.handle_exception("COMPUTE EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
            finished[0] = True
      post_op_queue.put((expr_idx, var_values, inst_block, j))


def lambdapack_post_op(post_op_queue, finish_queue, log_queue, program, finished):
   #print("POSTOP STARTED..")
   while(True):
      val = post_op_queue.get()
      if val == -1:
         #print("post_op returning")
         return
      expr_idx, var_values, inst_block, i = val
      #print(f"{expr_idx}, {var_values}, {inst_block}[{i}] in POST_OP")
      assert(len(inst_block.instrs) == i + 1)
      try:
         next_operator, profiled_block = program.post_op(expr_idx, var_values, lp.PS.SUCCESS, inst_block)
         finish_queue.put((expr_idx, var_values, inst_block, i))
         print("submitting log object...")
         log_queue.put((expr_idx, var_values, profiled_block))
      except:
         program.handle_exception("post op EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
         finished[0] = True


def reset_msg_visibility(msg_info_map, msg_vis_timeout, queue_url):
   for key, val in list(msg_info_map.items()):
        try:
            last_reset_time = val[-1]
            msg = val[0]
            queue_url = val[1]
            curr_time = time.time()
            if (last_reset_time - curr_time  > msg_vis_timeout/3):
               operator_ref = tuple(json.loads(msg["Body"]))
               if (finished[0]):
                  #print(f"reset msg for {operator_ref} finished quitting..")
                  break
               receipt_handle = msg["ReceiptHandle"]
               sqs_client = boto3.client('sqs')
               res = sqs_client.change_message_visibility(VisibilityTimeout=msg_vis_timeout, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
               msg_info_map[msg] = (msg, queue_url, time.time())
               time.sleep((msg_vis_timeout/2))
        except Exception as e:
            pass
   return 0

def lambdapack_finish(finish_queue, msg_info_map, program, finished):
   sqs_client = boto3.client('sqs')
   while (True):
      if (finished[0]):
         break
      val = finish_queue.get()
      if (val == -1):
         return
      expr_idx, var_values, inst_block, i = val
      assert(len(inst_block.instrs) == i + 1)
      print("Marking {0} as done".format((expr_idx, var_values)))
      program.set_node_status(expr_idx, var_values, lp.NS.FINISHED)
      msg, queue_url, reset_time = msg_info_map[(expr_idx, str(var_values))]
      receipt_handle = msg["ReceiptHandle"]
      del msg_info_map[(expr_idx, str(var_values))]
      sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)





















