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

from .matrix import BigMatrix
from .matrix_utils import load_mmap, chunk, generate_key_name_uop, generate_key_name_binop, constant_zeros
from . import control_plane
from . import utils


try:
  DEFAULT_CONFIG = wc.default()
except:
  DEFAULT_CONFIG = {}


class RemoteInstructionOpCodes(Enum):
    S3_LOAD = 0
    S3_WRITE = 1
    SYRK = 2
    TRSM = 3
    GEMM = 4
    ADD = 5
    SUB = 6
    CHOL = 7
    INVRS = 8
    RET = 9
    BARRIER = 10
    EXIT = 11


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
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        self.start_time = time.time()
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
              while (True):
                try:
                  self.result = await asyncio.wait_for(self.matrix.get_block_async(loop, *self.bidxs), self.MAX_READ_TIME)
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
    def __init__(self, i_id, matrix, data_instr, *bidxs):
        super().__init__(i_id)
        self.i_code = OC.S3_WRITE
        self.matrix = matrix
        self.bidxs = bidxs
        self.data_instr = data_instr
        self.result = None
        self.MAX_WRITE_TIME = 10
        self.write_size = np.product(self.matrix.shard_sizes)*np.dtype(self.matrix.dtype).itemsize

    #@profile
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        self.start_time = time.time()
        if (self.result is None):
            cache_key = (self.matrix.key, self.matrix.bucket, self.matrix.true_block_idx(*self.bidxs))
            if (self.cache != None):
              # write to the cache
              self.cache[cache_key] = self.data_instr.result
            backoff = 0.2
            while (True):
              try:
                self.result = await asyncio.wait_for(self.matrix.put_block_async(self.data_instr.result, loop, *self.bidxs), self.MAX_WRITE_TIME)
                break
              except (asyncio.TimeoutError, aiohttp.client_exceptions.ClientPayloadError, fs._base.CancelledError, botocore.exceptions.ClientError) as e:
                  await asyncio.sleep(backoff)
                  backoff *= 2
                  pass
            self.size = sys.getsizeof(self.data_instr.result)
            self.ret_code = 0
        self.end_time = time.time()
        return self.result

    def clear(self):
        self.result = None

    def __str__(self):
        bidxs_str = ""
        for x in self.bidxs:
            bidxs_str += str(x)
            bidxs_str += " "
        return "{0} = S3_WRITE {1} {2} {3} {4}".format(self.id, self.matrix, len(self.bidxs), bidxs_str.strip(), self.data_instr.id)


class RemoteIdentity(RemoteInstruction):
    def __init__(self, i_id, argv_instr, **kwargs):
        self.argv = argv_instr
        pass
    async def __call__(self, prev=None):
        self.result = self.argv_instr[0].result
        return self.result

class RemoteSYRK(RemoteInstruction):
    def __init__(self, i_id, argv_instr, **kwargs):
        super().__init__(i_id)
        self.i_code = OC.SYRK
        assert len(argv_instr) == 3
        self.argv = argv_instr
        self.result = None
    #@profile
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        #@profile
        def compute():
          self.start_time = time.time()
          if (self.result is None):
            # TODO: we shouldn't need to squeeze here.
            old_block = self.argv[0].result
            real_shape = old_block.shape
            block_2 = np.squeeze(self.argv[1].result)
            block_1 = np.squeeze(self.argv[2].result)
            res = old_block - (block_2.dot(block_1.T)).reshape(real_shape)
            self.result = res
            self.flops = old_block.size + 2*block_2.shape[0]*block_2.shape[1]*block_1.shape[0]
          else:
            raise Exception("Same Machine Replay instruction... ")
          self.ret_code = 0
          self.end_time = time.time()
          return self.result

        return await loop.run_in_executor(self.executor, compute)

    def get_flops(self):
      old_block = self.argv[0].result
      block_2 = self.argv[1].result
      block_1 = self.argv[2].result
      self.flops = old_block.size + 2*block_2.shape[0]*block_2.shape[1]*block_1.shape[0]
      return self.flops


    def __str__(self):
        return "{0} = SYRK {1} {2} {3}".format(self.id, self.argv[0].id,  self.argv[1].id,  self.argv[2].id)

class RemoteAdd(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.ADD
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        def compute():
          self.start_time = time.time()
          if self.result is None:
              block_1 = self.argv[0].result
              block_2 = self.argv[1].result
              self.result = block_1 + block_2
              self.flops = block_1.shape[0]*block_1.shape[1]
              self.ret_code = 0
          self.end_time = time.time()
          return self.result
        return await loop.run_in_executor(self.executor, compute)

    def __str__(self):
        return "{0} = ADD {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)


class RemoteSub(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.SUB
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        def compute():
          self.start_time = time.time()
          if self.result is None:
              block_1 = self.argv[0].result
              block_2 = self.argv[1].result
              self.result = block_1 - block_2
              self.flops = block_1.shape[0]*block_1.shape[1]
              self.ret_code = 0
          self.end_time = time.time()
          return self.result
        return await loop.run_in_executor(self.executor, compute)

    def __str__(self):
        return "{0} = SUB {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)

class RemoteSUM(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.GEMM
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        def compute():
          self.start_time = time.time()
          if (self.result == None):
              self.result = np.sum(self.argv)
              self.flops = block_1.shape[0]*block_1.shape[1]*len(self.argv)
              self.ret_code = 0
          self.end_time = time.time()
          return self.result
        return await loop.run_in_executor(self.executor, compute)

    def __str__(self):
        return "{0} = GEMM {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)


class RemoteGEMM(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.GEMM
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        def compute():
          self.start_time = time.time()
          if (self.result == None):
              block_1 = self.argv[0].result
              block_2 = self.argv[1].result
              self.result = block_1.dot(block_2)
              self.flops = 2*block_1.shape[0]*block_1.shape[1]*block_2.shape[1]
              self.ret_code = 0
          self.end_time = time.time()
          return self.result
        return await loop.run_in_executor(self.executor, compute)

    def __str__(self):
        return "{0} = GEMM {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)

class RemoteTRSM(RemoteInstruction):
    def __init__(self, i_id, argv_instr, lower=False, right=True, **kwargs):
        super().__init__(i_id)
        self.i_code = OC.TRSM
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
        self.lower = lower
        self.right = right
    #@profile
    async def __call__(self, prev=None):
      if (prev != None):
        await prev
      loop = asyncio.get_event_loop()
      #@profile
      def compute():
          self.start_time = time.time()
          if self.result is None:
              A_block = self.argv[0].result
              B_block = self.argv[1].result
              # TODO: this is a transpose hack for cholesky
              A_block_shape = A_block.shape
              A_block = A_block.squeeze()
              self.result = scipy.linalg.blas.dtrsm(1.0, A_block.T, B_block, side=int(self.right),lower=int(self.lower))
              A_block.reshape(A_block_shape)
              self.flops =  A_block.shape[0] * B_block.shape[0] * B_block.shape[1]
              self.ret_code = 0
          else:
            raise Exception("Same Machine Replay instruction...")
          self.end_time = time.time()
          return self.ret_code
      return await loop.run_in_executor(self.executor, compute)

    def clear(self):
        self.result = None

    def get_flops(self):
      L_bb = self.argv[1].result
      col_block = self.argv[0].result
      self.flops =  col_block.shape[1] * L_bb.shape[0] * L_bb.shape[1]
      return self.flops

    def __str__(self):
        return "{0} = TRSM {1} {2} {3} {4}".format(self.id, self.argv[0].id,  self.argv[1].id, self.lower, self.right)

class RemoteCholesky(RemoteInstruction):
    def __init__(self, i_id, argv_instr, **kwargs):
        super().__init__(i_id)
        self.i_code = OC.CHOL
        assert len(argv_instr) == 1
        self.argv = argv_instr
        self.result = None
    #@profile
    async def __call__(self, prev=None):
      if (prev != None):
          await prev
      loop = asyncio.get_event_loop()
      #@profile
      def compute():
          self.start_time = time.time()
          s = time.time()
          if (self.result is None):
              L_bb = self.argv[0].result
              self.result = np.linalg.cholesky(L_bb)
              self.flops = 1.0/3.0*(L_bb.shape[0]**3) + 2.0/3.0*(L_bb.shape[0])
              self.ret_code = 0
          else:
            raise Exception("Same Machine Replay instruction...")
          e = time.time()
          self.end_time = time.time()
          return self.result
      return await loop.run_in_executor(self.executor, compute)

    def clear(self):
        self.result = None

    def get_flops(self):
        L_bb = self.argv[0].result
        self.flops = 1.0/3.0*(L_bb.shape[0]**3)
        return self.flops

    def __str__(self):
        return "{0} = CHOL {1}".format(self.id, self.argv[0].id)


class RemoteReturn(RemoteInstruction):
    def __init__(self, i_id, *args, **kwargs):
        super().__init__(i_id)
        self.i_code = OC.RET
        self.result = None
        self.return_loc = kwargs["hash"]
    async def __call__(self, client, prev=None):
      if (prev != None):
          await prev
      loop = asyncio.get_event_loop()
      self.start_time = time.time()
      if (self.result == None):
        put(client, self.return_loc, PS.SUCCESS.value)
        self.size = sys.getsizeof(PS.SUCCESS.value)
      self.end_time = time.time()
      return self.result

    def clear(self):
        self.result = None

    def __str__(self):
        return "{0} = RET {1}".format(self.id, self.return_loc)

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
        return 0

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
    def __init__(self, program, config, executor=pywren.default_executor, pywren_config=DEFAULT_CONFIG,
                 num_priorities=1, io_rate=3e7, flop_rate=20e9, eager=False,
                 num_program_shards=5000):
        pwex = executor(config=pywren_config)
        self.config = config
        self.pywren_config = pywren_config
        self.config = config
        self.executor = executor
        self.bucket = pywren_config['s3']['bucket']
        self.program = program
        self.max_priority = num_priorities - 1
        self.io_rate = io_rate
        self.flop_rate = flop_rate
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
        try:
          post_op_start = time.time()
          children = self.program.get_children(expr_idx, var_values)
          node_status = self.get_node_status(expr_idx, var_values)
          # if we had 2 racing tasks and one finished no need to go through rigamarole
          # of re-enqueeuing children
          if (node_status == NS.FINISHED):
            return
          self.set_node_status(expr_idx, var_values, NS.POST_OP)
          if (ret_code == PS.EXCEPTION and tb != None):
            self.handle_exception(" EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)
          ready_children = []
          for child in children:
              operator_expr = self.program.get_expr(child[0])
              self.control_plane.client.set("{0}_sqs_meta".format(self._edge_key(expr_idx, var_values, *child)), "STILL IN POST OP")
              my_child_edge = self._edge_key(expr_idx, var_values, *child)
              child_edge_sum_key = self._node_edge_sum_key(*child)
              # redis transaction should be atomic
              tp = fs.ThreadPoolExecutor(1)
              val_future = tp.submit(conditional_increment, self.control_plane.client, child_edge_sum_key, my_child_edge)
              done, not_done = fs.wait([val_future], timeout=60)
              if len(done) == 0:
                raise Exception("Redis Atomic Set and Sum timed out!")
              val = val_future.result()
              if (operator_expr == self.program.return_expr):
                num_child_parents = self.program.num_terminators
              else:
                num_child_parents = len(self.program.get_parents(child[0], child[1]))

              if ((val == num_child_parents) and self.get_node_status(*child) != NS.FINISHED):
                self.set_node_status(*child, NS.READY)
                ready_children.append(child)

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
            message_body = json.dumps([int(child[0]), {key.name: int(val) for key, val in child[1].items()}])
            resp = client.send_message(QueueUrl=self.queue_urls[0], MessageBody=message_body)
            self.control_plane.client.set("{0}_sqs_meta".format(self._edge_key(expr_idx, var_values, *child)), str(resp))

          inst_block.end_time = time.time()
          inst_block.clear()
          post_op_end = time.time()
          post_op_time = post_op_end - post_op_start
          self.incr_progress()
          self.set_profiling_info(inst_block, expr_idx, var_values)
          return next_operator
        except Exception as e:
            tb = traceback.format_exc()
            traceback.print_exc()
            print("Exception raised...")
            self.handle_exception("POST OP EXCEPTION", tb=tb, expr_idx=expr_idx, var_values=var_values)

    def start(self):
        put(self.control_plane.client, self.hash, PS.RUNNING.value)
        sqs = boto3.resource('sqs')
        for starter in self.program.starters:
          self.set_node_status(*starter, NS.READY)
          # TODO readd priorities here
          queue = sqs.Queue(self.queue_urls[0])
          queue.send_message(MessageBody=json.dumps([starter[0], {key.name: val for key, val in starter[1].items()}]))
        return 0

    def stop(self):
        client = boto3.client('s3')
        print("Stopping program")
        client.put_object(Key="lambdapack/" + self.hash + "/EXCEPTION.DRIVER.CANCELLED", Bucket=self.bucket, Body="cancelled by driver")
        e = PS.EXCEPTION.value
        put(self.control_plane.client, self.hash, e)

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
        # while (status == PS.RUNNING):
        #     time.sleep(sleep_time)
        #     status = self.program_status()

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

