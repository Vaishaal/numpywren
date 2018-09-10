import abc
from collections import defaultdict
import concurrent.futures as fs
import copy
from enum import Enum
import gc
import hashlib
import math
from multiprocessing import Process
import os
import pickle
import time
import traceback
import sys

import aiobotocore
import aiohttp
import asyncio
import boto3
import botocore
import botocore.exceptions
import json
import numpy as np
import pywren
from pywren.serialize import serialize
import pywren.wrenconfig as wc
import sympy
import redis
import scipy.linalg
import dill
import redis.exceptions
import logging
from .matrix import BigMatrix
from .matrix_utils import load_mmap, chunk, generate_key_name_uop, generate_key_name_binop, constant_zeros
from . import control_plane, matrix
from . import utils


try:
  DEFAULT_CONFIG = wc.default()
except:
  DEFAULT_CONFIG = {}

logger = logging.getLogger(__name__)


class RemoteInstructionOpCodes(Enum):
    S3_LOAD = 0
    S3_WRITE = 1
    GENERIC = 3
    RET = 4


class NodeStatus(Enum):
    NOT_READY = 0
    READY = 1
    RUNNING = 2
    POST_OP = 3
    FINISHED = 4

class EdgeStatus(Enum):
    NOT_READY = 0
    READY = 1

class ProgramStatus(Enum):
    SUCCESS = 0
    RUNNING = 1
    EXCEPTION = 2
    NOT_STARTED = 3

def put(client, key, value, s3=False, s3_bucket=""):
    #TODO: fall back to S3 here
    if (s3):
      # flush write to S3
      raise Exception("Not Implemented")
    backoff = 1
    val = None
    while(True):
      try:
        val = client.set(key, value)
        break
      except redis.exceptions.TimeoutError:
        time.sleep(backoff)
        backoff *= 2
    return val

def upload(key, bucket, data):
    client = boto3.client('s3')
    client.put_object(Bucket=bucket, Key=key, Body=data)

def get(client, key, s3=False, s3_bucket=""):
    #TODO: fall back to S3 here
    if (s3):
      # read from S3
      raise Exception("Not Implemented")
    else:
      if (client is None):
        raise Exception("No redis client up")
      backoff = 1
      val = None
      while(True):
        try:
          val = client.get(key)
          break
        except redis.exceptions.TimeoutError:
          time.sleep(backoff)
          backoff *= 2
      return val




def incr(client, key, amount=1, s3=False, s3_bucket=""):
    #TODO: fall back to S3 here
    if (s3):
      # read from S3
      raise Exception("Not Implemented")
    else:
      if (client is None):
        raise Exception("No redis client up")
      backoff = 1
      val = None
      while(True):
        try:
          val = client.incr(key, amount=amount)
          break
        except redis.exceptions.TimeoutError:
          time.sleep(backoff)
          backoff *= 2
      return val

def decr(client, key, amount, s3=False, s3_bucket=""):
    #TODO: fall back to S3 here
    if (s3):
      # read from S3
      raise Exception("Not Implemented")
    else:
      if (client is None):
        raise Exception("No redis client up")
      backoff = 1
      val = None
      while(True):
        try:
          val = client.decr(key, amount=amount)
          break
        except redis.exceptions.TimeoutError:
          time.sleep(backoff)
          backoff *= 2
      return val



def conditional_increment(client, key_to_incr, condition_key):
  ''' Crucial atomic operation needed to insure DAG correctness
      @param key_to_incr - increment this key
      @param condition_key - only do so if this value is 1
      @param ip - ip of redis server
      @param value - the value to bind key_to_set to
    '''
  res = 0

  r = client
  backoff = 1
  success = False
  while (True):
    try:
      with r.pipeline() as pipe:
        while True:
          try:
            pipe.watch(condition_key)
            pipe.watch(key_to_incr)
            current_value = pipe.get(key_to_incr)
            if (current_value is None):
              current_value = 0
            current_value = int(current_value)
            condition_val = pipe.get(condition_key)
            if (condition_val is None):
              condition_val = 0
            condition_val = int(condition_val)
            res = current_value
            if (condition_val == 0):
              pipe.multi()
              pipe.incr(key_to_incr)
              pipe.set(condition_key, 1)
              t_results = pipe.execute()
              res = int(t_results[0])
              assert(t_results[1])
            break
          except redis.WatchError as e:
            continue
        break
    except redis.exceptions.TimeoutError:
        time.sleep(backoff)
        backoff *= 2
  return res


OC = RemoteInstructionOpCodes
NS = NodeStatus
ES = EdgeStatus
PS = ProgramStatus



class RemoteInstruction(object):
    def __init__(self, i_id):
        self.id = i_id
        self.ret_code = -1
        self.start_time = None
        self.end_time = None
        self.type = None
        self.executor = None
        self.cache = None
        self.run = False
        self.read_size = 0
        self.write_size = 0

    def get_flops(self):
      return 0

    def clear(self):
        self.result = None

class RemoteRead(RemoteInstruction):
    def __init__(self, i_id, matrix, *bidxs):
        super().__init__(i_id)
        self.i_code = OC.S3_LOAD
        self.matrix = matrix
        self.bidxs = bidxs
        self.result = None
        self.cache_hit = False
        self.MAX_READ_TIME = 10
        self.read_size = np.product(self.matrix.shard_sizes)*np.dtype(self.matrix.dtype).itemsize

    #@profile
    async def __call__(self):
        loop = asyncio.get_event_loop()
        self.start_time = time.time()
        t = time.time()
        print("reading...", self.bidxs)
        if (self.result is None):
            cache_key = (self.matrix.key, self.matrix.bucket, self.matrix.true_block_idx(*self.bidxs))
            if (self.cache != None and cache_key in self.cache):
              t = time.time()
              self.result = self.cache[cache_key]
              self.cache_hit = True
              self.size = sys.getsizeof(self.result)
              e = time.time()
            else:
              t = time.time()
              backoff = 0.2
              #print(f"Reading from {self.matrix} at {self.bidxs}")
              while (True):
                try:
                  self.result = await asyncio.wait_for(self.matrix.get_block_async(loop, *self.bidxs), self.MAX_READ_TIME)
                  print("Read ", self.result)
                  break
                except (asyncio.TimeoutError, aiohttp.client_exceptions.ClientPayloadError, fs._base.CancelledError, botocore.exceptions.ClientError):
                  await asyncio.sleep(backoff)
                  backoff *= 2
                  pass
              self.size = sys.getsizeof(self.result)
              if (self.cache != None):
                self.cache[cache_key] = self.result
              e = time.time()
        self.end_time = time.time()
        e = time.time()
        print(f"Read took {e - t} seconds")
        return self.result

    def clear(self):
        self.result = None
    def __str__(self):
        bidxs_str = ""
        for x in self.bidxs:
            bidxs_str += str(x)
            bidxs_str += " "
        return "{0} = S3_LOAD {1} {2} {3}".format(self.id, self.matrix, len(self.bidxs), bidxs_str.strip())

class RemoteWrite(RemoteInstruction):
    def __init__(self, i_id, matrix, data_loc, data_idx, *bidxs):
        super().__init__(i_id)
        self.i_code = OC.S3_WRITE
        self.matrix = matrix
        self.bidxs = bidxs
        self.data_loc = data_loc
        self.data_idx = data_idx

        self.result = None
        self.MAX_WRITE_TIME = 10
        self.write_size = np.product(self.matrix.shard_sizes)*np.dtype(self.matrix.dtype).itemsize

    #@profile
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        t = time.time()
        loop = asyncio.get_event_loop()
        self.start_time = time.time()
        if (self.result is None):
            cache_key = (self.matrix.key, self.matrix.bucket, self.matrix.true_block_idx(*self.bidxs))
            if (self.cache != None):
              # write to the cache
              self.cache[cache_key] = self.data_loc[self.data_idx]
            backoff = 0.2
            #print(f"Writing to {self.matrix} at {self.bidxs}")
            while (True):
              try:
                self.result = await asyncio.wait_for(self.matrix.put_block_async(self.data_loc[self.data_idx], loop, *self.bidxs), self.MAX_WRITE_TIME)
                break
              except (asyncio.TimeoutError, aiohttp.client_exceptions.ClientPayloadError, fs._base.CancelledError, botocore.exceptions.ClientError) as e:
                  await asyncio.sleep(backoff)
                  backoff *= 2
                  pass
            self.size = sys.getsizeof(self.data_loc[self.data_idx])
            self.ret_code = 0
        self.end_time = time.time()
        e = time.time()
        print(f"Write took {e - t} seconds")
        return self.result

    def clear(self):
        self.result = None

    def __str__(self):
        bidxs_str = ""
        for x in self.bidxs:
            bidxs_str += str(x)
            bidxs_str += " "
        return "{0} = S3_WRITE {1} {2} {3} {4}".format(self.id, self.matrix, len(self.bidxs), bidxs_str.strip(), self.data_idx)


class RemoteCall(RemoteInstruction):
    def __init__(self, i_id, compute, argv_instr, num_outputs, symbols, **kwargs):
        super().__init__(i_id)
        self.i_code = OC.GENERIC
        self.results = [None for x in range(num_outputs)]
        self.kwargs = kwargs
        self.compute = compute
        self.symbols = symbols
        self.argv_instr = argv_instr
    #@profile
    async def __call__(self, prev=None):
        t = time.time()
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        #@profile
        def compute():
          self.start_time = time.time()
          # TODO: we shouldn't need to squeeze here.
          pyarg_list = []
          for arg in self.argv_instr:
            if (isinstance(arg, RemoteRead)):
              pyarg_list.append(arg.result)
            elif (isinstance(arg, float)):
              pyarg_list.append(arg)
          #print("CALLING COMPUTE", self.compute)
          results = self.compute(*pyarg_list, **self.kwargs)
          if (isinstance(results, tuple) and len(results) != len(self.results)):
            raise Exception("Expected {0} results, got {1}".format(len(self.results), len(results)))
          elif (isinstance(results, tuple)):
            for i,r in enumerate(results):
              self.results[i] = results[i]
          else:
            self.results[0] = results
          self.ret_code = 0
          self.end_time = time.time()
          return self.results
        res = await loop.run_in_executor(self.executor, compute)
        e = time.time()
        print(f"Compute {self.compute} took {e - t} seconds")
        return res

    def get_flops(self):
      if getattr(self.compute, "flops", None) is not None:
        return self.compute.flops(*argv_instr)
      else:
        return 0

    def __str__(self):
        out_str = []
        for i,z in enumerate(self.results):
          out_str.append("{0}".format(i + len(self.symbols)))
        out_str = ",".join(out_str)
        return "{1} = {0}({2}, **kwargs)".format(self.compute, out_str, ",".join(self.symbols))


class RemoteReturn(RemoteInstruction):
    def __init__(self, i_id):
        super().__init__(i_id)
        self.i_code = OC.RET
        self.result = None
    async def __call__(self, client, return_hash):
      logger.debug("RETURNING...")
      print("RETURNING....")
      loop = asyncio.get_event_loop()
      self.start_time = time.time()
      if (self.result == None):
        print("RETURNING VALUE ", PS.SUCCESS.value)
        put(client, return_hash, PS.SUCCESS.value)
        print("GET VALUE ", get(client, return_hash))
        self.size = sys.getsizeof(PS.SUCCESS.value)
      self.end_time = time.time()
      return self.result

    def clear(self):
        self.result = None

    def __name__(self):
      return "RemoteReturn"

    def __str__(self):
        return "RET"

class InstructionBlock(object):
    block_count = 0
    def __init__(self, instrs, label=None, priority=0):
        self.instrs = instrs
        self.label = label
        self.priority = priority
        if (self.label == None):
            self.label = "%{0}".format(InstructionBlock.block_count)
        InstructionBlock.block_count += 1

    def __call__(self):
        val = [x() for x in self.instrs]
        return val

    def __str__(self):
        out = ""
        out += self.label
        out += "\n"
        for inst in self.instrs:
            out += "\t"
            out += str(inst)
            out += "\n"
        return out
    def clear(self):
      [x.clear() for x in self.instrs]

    def total_flops(self):
      return sum([getattr(x, "flops", 0) for x in self.instrs])

    def total_io(self):
      return sum([getattr(x, "size", 0) for x in self.instrs])

    def __copy__(self):
        return InstructionBlock(self.instrs.copy(), self.label)


class LambdaPackProgram(object):
    '''Sequence of instruction blocks that get executed
       on stateless computing substrates
       Maintains global state information
    '''
    def __init__(self, program, config, num_priorities=1, eager=False):
        self.config = config
        self.config = config
        self.bucket = matrix.DEFAULT_BUCKET
        self.program = program
        self.max_priority = num_priorities - 1
        self.eager = eager
        cpid = control_plane.get_control_plane_id(config=config)
        if (cpid is None):
          raise Exception("No active control planes")
        self.control_plane = control_plane.get_control_plane(config=config)
        hashed = hashlib.sha1()
        #HACK to have interpretable runs
        self.hash = str(int(time.time()))
        self.up = 'up' + self.hash
        self.set_up(0)
        client = boto3.client('sqs', region_name=self.control_plane.region)
        self.queue_urls = []
        for i in range(num_priorities):
          queue_url = client.create_queue(QueueName=self.hash + str(i))["QueueUrl"]
          self.queue_urls.append(queue_url)
          client.purge_queue(QueueUrl=queue_url)
        put(self.control_plane.client, self.hash, PS.NOT_STARTED.value)

    def _node_key(self, expr_idx, var_values):
      return "{0}_{1}".format(self.hash, self._node_str(expr_idx, var_values))

    def _node_edge_sum_key(self, expr_idx, var_values):
      return "{0}_{1}_edgesum".format(self.hash, self._node_str(expr_idx, var_values))

    def _edge_key(self, expr_idx1, var_values1, expr_idx2, var_values2):
      return "{0}_{1}_{2}".format(self.hash, self._node_str(expr_idx1, var_values1),
                                  self._node_str(expr_idx2, var_values2))

    def _node_str(self, expr_idx, var_values):
        var_strs = sorted(["{0}:{1}".format(key, value) for key, value in var_values.items()])
        return "{0}_({1})".format(expr_idx, "-".join(var_strs))

    def get_node_status(self, expr_idx, var_values):
      s = get(self.control_plane.client, self._node_key(expr_idx, var_values))
      if (s == None):
        s = 0
      return NS(int(s))

    def set_node_status(self, expr_id, var_values, status):
      put(self.control_plane.client, self._node_key(expr_id, var_values), status.value)
      return status

    def set_profiling_info(self, inst_block, expr_idx, var_values):
        byte_string = dill.dumps(inst_block)
        client = boto3.client('s3', region_name=self.control_plane.region)
        client.put_object(Bucket=self.bucket, Key="lambdapack/{0}/{1}_{2}".format(self.hash, expr_idx, var_values), Body=byte_string)

    def post_op(self, expr_idx, var_values, ret_code, inst_block, tb=None):
        # need clean post op logic to handle
        # replays
        # avoid double increments
        # failures at ANY POINT

        # for each dependency2
        # post op needs to ATOMICALLY check dependencies
        t = time.time()
        try:
          post_op_start = time.time()
          children = self.program.find_children(expr_idx, var_values)
          node_status = self.get_node_status(expr_idx, var_values)
          # if we had 2 racing tasks and one finished no need to go through rigamarole
          # of re-enqueeuing children
          if (node_status == NS.FINISHED):
            return
          self.set_node_status(expr_idx, var_values, NS.POST_OP)
          if (ret_code == PS.EXCEPTION and tb != None):
            self.handle_exception(" EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
          ready_children = []
          #print(f"ALL CHILDREN {children}")
          for child in children:
              my_child_edge = self._edge_key(expr_idx, var_values, *child)
              child_edge_sum_key = self._node_edge_sum_key(*child)
              # redis transaction should be atomic
              tp = fs.ThreadPoolExecutor(1)
              val_future = tp.submit(conditional_increment, self.control_plane.client, child_edge_sum_key, my_child_edge)
              done, not_done = fs.wait([val_future], timeout=60)
              if len(done) == 0:
                raise Exception("Redis Atomic Set and Sum timed out!")
              val = val_future.result()
              print(child[1])
              parents = self.program.find_parents(child[0], child[1])
              num_child_parents = len(parents)
              if ((val == num_child_parents) and self.get_node_status(*child) != NS.FINISHED):
                self.set_node_status(*child, NS.READY)
                ready_children.append(child)
          print("Ready children", ready_children)

          if self.eager and ready_children:
              # TODO: Re-add priorities here
              next_operator = ready_children[-1]
              del ready_children[-1]
          else:
              next_operator = None
          # move the highest priority job thats ready onto the local task queue
          # this is JRK's idea of dynamic node fusion or eager scheduling
          # the idea is that if we do something like a local cholesky decomposition
          # we would run its highest priority child *locally* by adding the instructions to the local instruction queue
          # this has 2 key benefits, first we completely obliviete scheduling overhead between these two nodes but also because of the local LRU cache the first read of this node will be saved this will translate
          client = boto3.client('sqs', region_name=self.control_plane.region)
          assert (expr_idx, var_values) not in ready_children
          for child in ready_children:
            # TODO: Re-add priorities here
            message_body = json.dumps([int(child[0]), {str(key): int(val) for key, val in child[1].items()}])
            resp = client.send_message(QueueUrl=self.queue_urls[0], MessageBody=message_body)

          inst_block.end_time = time.time()
          inst_block.clear()
          post_op_end = time.time()
          post_op_time = post_op_end - post_op_start
          self.incr_progress()
          self.set_profiling_info(inst_block, expr_idx, var_values)
          e = time.time()
          print(f"Post op for {expr_idx, var_values}, took {e - t} seconds")
          terminator = self.program.is_terminator(expr_idx)
          if (terminator):
            return_key = self.hash + "_return"
            return_edge = self._edge_key(expr_idx, var_values, return_key, {})
            tp = fs.ThreadPoolExecutor(1)
            val_future = tp.submit(conditional_increment, self.control_plane.client, return_key, return_edge)
            done, not_done = fs.wait([val_future], timeout=60)
            if len(done) == 0:
              raise Exception("Redis Atomic Set and Sum timed out!")
            val = val_future.result()
            print("Num finished terminators", val,  "num terminators total", self.program.num_terminators)
            if (val == self.program.num_terminators):
              self.return_success()
          return next_operator
        except Exception as e:
            tb = traceback.format_exc()
            traceback.print_exc()
            self.handle_exception("POST OP EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)

    def start(self):
        put(self.control_plane.client, self.hash, PS.RUNNING.value)
        sqs = boto3.resource('sqs')
        for starter in self.program.starters:
          self.set_node_status(*starter, NS.READY)
          # TODO readd priorities here
          queue = sqs.Queue(self.queue_urls[0])
          queue.send_message(MessageBody=json.dumps([starter[0], {str(key): val for key, val in starter[1].items()}]))
        return 0


    def stop(self):
        client = boto3.client('s3')
        client.put_object(Key="lambdapack/" + self.hash + "/EXCEPTION.DRIVER.CANCELLED", Bucket=self.bucket, Body="cancelled by driver")
        e = PS.EXCEPTION.value
        put(self.control_plane.client, self.hash, e)

    def return_success(self):
      print("RETURNING....")
      put(self.control_plane.client, self.hash, PS.SUCCESS.value)



    def handle_exception(self, error, tb, expr_idx, var_values):
        client = boto3.client('s3')
        client.put_object(Key="lambdapack/" + self.hash + "/EXCEPTION.{0}".format(self._node_str(expr_idx, var_values)), Bucket=self.bucket, Body=tb + str(error))
        e = PS.EXCEPTION.value
        put(self.control_plane.client, self.hash, e)

    def program_status(self):
      status = get(self.control_plane.client, self.hash)
      return PS(int(status))

    def incr_up(self, amount):
      incr(self.control_plane.client, self.up, amount)

    def incr_progress(self):
      incr(self.control_plane.client, "{0}_progress".format(self.hash))


    def incr_flops(self, amount):
      if (amount > 0):
        incr(self.control_plane.client, "{0}_flops".format(self.hash), amount)

    def incr_read(self, amount):
      if (amount > 0):
        incr(self.control_plane.client,"{0}_read".format(self.hash), amount)

    def incr_write(self, amount):
      if (amount > 0):
        incr(self.control_plane.client,"{0}_write".format(self.hash), amount)

    def decr_flops(self, amount):
      if (amount > 0):
        decr(self.control_plane.client,"{0}_flops".format(self.hash), amount)

    def decr_read(self, amount):
      if (amount > 0):
        decr(self.control_plane.client,"{0}_read".format(self.hash), amount)

    def decr_write(self, amount):
      if (amount > 0):
        decr(self.control_plane.client,"{0}_write".format(self.hash), amount)

    def decr_up(self, amount):
      decr(self.control_plane.client,self.up, amount)

    def get_up(self):
      return get(self.control_plane.client, self.up)

    def get_flops(self):
      return get(self.control_plane.client, "{0}_flops".format(self.hash))

    def get_read(self):
      return get(self.control_plane.client, "{0}_read".format(self.hash))

    def get_write(self):
      return get(self.control_plane.client, "{0}_write".format(self.hash))

    def get_progress(self):
      return get(self.control_plane.client, "{0}_progress".format(self.hash))

    def set_up(self, value):
      put(self.control_plane.client,self.control_plane.client, self.up, value)

    def wait(self, sleep_time=1):
        status = self.program_status()
        # TODO reinstate status change
        while (status == PS.RUNNING):
              time.sleep(sleep_time)
              status = self.program_status()
              print("Program status is ", status)

    def free(self):
        for queue_url in self.queue_urls:
          client = boto3.client('sqs')
          client.delete_queue(QueueUrl=queue_url)

    def get_all_profiling_info(self):
        return [self.get_profiling_info(i) for i in range(self.num_inst_blocks) if i is not None]

    def get_profiling_info(self, expr_idx, var_values):
        try:
          client = boto3.client('s3')
          byte_string = client.get_object(Bucket=self.bucket, Key="lambdapack/{0}/{1}_{2}".format(self.hash, expr_idx, var_values))["Body"].read()
          return dill.loads(byte_string)
        except:
          raise

