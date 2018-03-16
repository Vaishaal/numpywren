import numpywren
import numpywren.matrix
from .matrix import BigMatrix, BigSymmetricMatrix, Scalar
from .matrix_utils import load_mmap, chunk, generate_key_name_uop, generate_key_name_binop, constant_zeros
import numpy as np
import pywren
from pywren.serialize import serialize
from numpywren import matrix_utils, uops
import pytest
import numpy as np
import pywren
import pywren.wrenconfig as wc
import unittest
import time
import time
from enum import Enum
import boto3
import hashlib
import copy
import concurrent.futures as fs
import sys
import botocore
import scipy.linalg
import traceback
import pickle
from collections import defaultdict
import aiobotocore
import asyncio
import redis

try:
  DEFAULT_CONFIG = wc.default()
except:
  DEFAULT_CONFIG = {}


class LocalExecutor(object):
  def __init__(self, procs=32, config=DEFAULT_CONFIG):
    self.procs = procs
    self.executor = fs.ProcessPoolExecutor(max_workers=procs)
    self.config = DEFAULT_CONFIG

  def call_async(self, f, *args, **kwargs):
    return self.executor.submit(f, *args, **kwargs)

  def map(self, f, arg_list, **kwargs):
    futures = []
    for a in arg_list:
      futures.append(self.call_async(f, a))
    return futures

class RemoteInstructionOpCodes(Enum):
    S3_LOAD = 0
    S3_WRITE = 1
    SYRK = 2
    TRSM = 3
    GEMM = 4 
    ADD = 4 
    SUB = 5 
    CHOL = 6 
    INVRS = 7 
    RET = 8 
    EXIT = 9 


class RemoteInstructionExitCodes(Enum):
    SUCCESS = 0
    RUNNING = 1
    EXCEPTION = 2
    REPLAY = 3
    NOT_STARTED = 4

class RemoteInstructionTypes(Enum):
  IO = 0
  COMPUTE = 1

class RemoteProgramState(object):
  ''' Host integers on dynamodb '''
  def __init__(self, key, ip="127.0.0.1"):
    self.key = key
    self.ip = ip

  def put(self, value):
    assert isinstance(value, int)
    redis_client = redis.StrictRedis(self.ip, port=6379, db=0)
    redis_client.set(self.key, value)

  def clear_all(self):
    redis_client = redis.StrictRedis(self.ip, port=6379, db=0)
    redis_client.flushall()

  def get(self):
    redis_client = redis.StrictRedis(self.ip, port=6379, db=0)
    return int(redis_client.get(self.key))

  def incr(self, inc=1):
    t = time.time()
    redis_client = redis.StrictRedis(self.ip, port=6379, db=0)
    final_val = redis_client.incr(self.key)
    e = time.time()
    print("{0} took {1}".format(self.key, e - t))
    return final_val




EC = RemoteInstructionExitCodes
OC = RemoteInstructionOpCodes
IT = RemoteInstructionTypes
RPS = RemoteProgramState


class RemoteInstruction(object):
    def __init__(self, i_id):
        self.id = i_id
        self.ret_code = -1
        self.start_time = None
        self.end_time = None
        self.type = None
        self.executor = None

    def clear(self):
        self.result = None

    def __deep_copy__(self, memo):
        return self

class RemoteLoad(RemoteInstruction):
    def __init__(self, i_id, matrix, *bidxs):
        super().__init__(i_id)
        self.i_code = OC.S3_LOAD
        self.matrix = matrix
        self.bidxs = bidxs
        self.result = None

    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        self.start_time = time.time()
        if (self.result == None):
            self.result = await self.matrix.get_block_async(loop, *self.bidxs)
            self.size = sys.getsizeof(self.result)
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

    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        self.start_time = time.time()
        if (self.result == None):
            self.result = await self.matrix.put_block_async(self.data_instr.result, loop, *self.bidxs)
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


class RemoteSYRK(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.SYRK
        assert len(argv_instr) == 3
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
        if (prev != None):
          await prev
        loop = asyncio.get_event_loop()
        def compute():
          self.start_time = time.time()
          if (self.result == None):
              old_block = self.argv[0].result
              block_2 = self.argv[1].result
              block_1 = self.argv[2].result
              old_block -= block_2.dot(block_1.T)
              self.result = old_block
              self.flops = old_block.size + 2*block_2.shape[0]*block_2.shape[1]*block_1.shape[0]
              self.ret_code = 0
          self.end_time = time.time()
          return self.result
        return await loop.run_in_executor(self.executor, compute)

    def __str__(self):
        return "{0} = SYRK {1} {2} {3}".format(self.id, self.argv[0].id,  self.argv[1].id,  self.argv[2].id)

class RemoteSub(RemoteInstruction):
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
              self.result = block_1 - block_2 
              self.flops = block_1.shape[0]*block_1.shape[1]
              self.ret_code = 0
          self.end_time = time.time()
          return self.result
        return await loop.run_in_executor(self.executor, compute)

    def __str__(self):
        return "{0} = SUB {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)


class RemoteGemm(RemoteInstruction):
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
    def __init__(self, i_id, argv_instr, lower=False, right=False):
        super().__init__(i_id)
        self.i_code = OC.TRSM
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
        self.lower = lower
        self.right = right
    async def __call__(self, prev=None):
      if (prev != None):
        await prev
      loop = asyncio.get_event_loop()
      def compute():
          self.start_time = time.time()
          if (self.result == None):
              A_block = self.argv[0].result
              B_block = self.argv[1].result
              self.result = scipy.linalg.blas.dtrsm(1.0, A_block, B_block, side=int(self.right),lower=int(self.lower))
              self.flops =  A_block.shape[0] * B_block.shape[0] * B_block.shape[1]
              self.ret_code = 0
          self.end_time = time.time()
          return self.ret_code
      return await loop.run_in_executor(self.executor, compute)

    def clear(self):
        self.result = None

    def __str__(self):
        return "{0} = TRSM {1} {2} {3} {4}".format(self.id, self.argv[0].id,  self.argv[1].id, self.lower, self.right)

class RemoteCholesky(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.CHOL
        assert len(argv_instr) == 1
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
      if (prev != None):
          await prev
      loop = asyncio.get_event_loop()
      def compute():
          self.start_time = time.time()
          s = time.time()
          if (self.result == None):
              L_bb = self.argv[0].result
              self.result = np.linalg.cholesky(L_bb)
              self.flops = 1.0/3.0*(L_bb.shape[0]**3) + 2.0/3.0*(L_bb.shape[0])
              self.ret_code = 0
          e = time.time()
          self.end_time = time.time()
          return self.result
      return await loop.run_in_executor(self.executor, compute)

    def clear(self):
        self.result = None

    def __str__(self):
        return "{0} = CHOL {1}".format(self.id, self.argv[0].id)

class RemoteInverse(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.INVRS
        assert len(argv_instr) == 1
        self.argv = argv_instr
        self.result = None
    async def __call__(self, prev=None):
      if (prev != None):
          await prev
      loop = asyncio.get_event_loop()
      def compute():
          self.start_time = time.time()
          if (self.result == None):
              L_bb = self.argv[0].result
              self.result = np.linalg.inv(L_bb)
              self.flops = 2.0/3.0*(L_bb.shape[0]**3)
              self.ret_code = 0
          self.end_time  = time.time()
          return self.result
      return await loop.run_in_executor(self.executor, compute)


    def clear(self):
        self.result = None

    def __str__(self):
        return "{0} = INVRS {1}".format(self.id, self.argv[0].id)


class RemoteReturn(RemoteInstruction):
    def __init__(self, i_id, return_loc):
        super().__init__(i_id)
        self.i_code = OC.RET
        self.return_loc = return_loc
        self.result = None
    async def __call__(self, prev=None):
      if (prev != None):
          await prev
      loop = asyncio.get_event_loop()
      self.start_time = time.time()
      if (self.result == None):
        self.return_loc.put(EC.SUCCESS.value)
        self.size = sys.getsizeof(EC.SUCCESS.value)
      self.end_time = time.time()
      return self.result

    def clear(self):
        self.result = None

    def __str__(self):
        return "{0} = RET {1}".format(self.id, self.return_loc)

class InstructionBlock(object):
    block_count = 0
    def __init__(self, instrs, label=None):
        self.instrs = instrs
        self.label = label
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
    def __copy__(self):
        return InstructionBlock(self.instrs.copy(), self.label)


class LambdaPackProgram(object):
    '''Sequence of instruction blocks that get executed
       on stateless computing substrates
       Maintains global state information
    '''

    def __init__(self, inst_blocks, executor=pywren.default_executor, pywren_config=DEFAULT_CONFIG):
        pwex = executor(config=pywren_config)
        self.pywren_config = pywren_config
        self.executor = executor
        self.bucket = pywren_config['s3']['bucket']
        self.inst_blocks = [copy.copy(x) for x in inst_blocks]
        self.program_string = "\n".join([str(x) for x in inst_blocks])
        program_string = "\n".join([str(x) for x in self.inst_blocks])
        hashed = hashlib.sha1()
        hashed.update(program_string.encode())
        hashed.update(str(time.time()).encode())
        self.hash = hashed.hexdigest()
        self.ret_status = RPS(self.hash)
        self.ret_status.clear_all()
        client = boto3.client('sqs', region_name='us-west-2')
        self.queue_url = client.create_queue(QueueName=self.hash)["QueueUrl"]
        client.purge_queue(QueueUrl=self.queue_url)
        self.children, self.parents = self._io_dependency_analyze(self.inst_blocks)
        self.starters = []
        self.terminators = []
        max_i_id = max([inst.id for inst_block in self.inst_blocks for inst in inst_block.instrs])
        self.remote_return = RemoteReturn(max_i_id + 1, self.ret_status)
        self.return_block = InstructionBlock([self.remote_return], label="EXIT")
        self.inst_blocks.append(self.return_block)

        self.pc = max_i_id + 1
        self.block_return_statuses = []
        self.block_ready_statuses = []
        for i, (children, parents, inst_block) in enumerate(zip(self.children, self.parents, self.inst_blocks)):
            if len(children) == 0:
                self.terminators.append(i)
            if len(parents) == 0:
                self.starters.append(i)
            block_hash = hashlib.sha1((self.hash + str(i)).encode()).hexdigest()
            block_ret_status  = RPS(block_hash)
            block_return = RemoteReturn(self.pc + 1, block_ret_status)
            block_ready_hash = block_hash + "_ready"
            block_ready_status = RPS(block_ready_hash)
            self.inst_blocks[i].instrs.append(block_return)
            self.block_return_statuses.append(block_ret_status)
            self.block_ready_statuses.append(block_ready_status)

        for i in self.terminators:
          self.children[i].add(len(self.inst_blocks) - 1)
        self.children.append(set())
        self.parents.append(self.terminators)
        self.block_return_statuses.append(self.ret_status)
        self.ret_ready_status = RPS(self.hash + "_ready")
        self.block_ready_statuses.append(self.ret_ready_status)

    def pre_op(self, i):
        # 1. check if program has terminated
        # 2. check if this instruction_block has executed successfully
        # 3. check if parents are not completed
        # if any of the above are False -> exit
        try:
          print("RUNNING " , i)
          program_status = self.program_status().value
          if (i == len(self.inst_blocks) - 1):
            # special case final block
            print("RETURN BLOCK")
            pass
          self.set_inst_block_status(i, EC.RUNNING)
          self.inst_blocks[i].start_time = time.time()
          if (program_status != EC.RUNNING.value):
            return i, self.inst_blocks[i], EC.EXCEPTION
        except Exception as e:
            print("EXCEPTION ", e)
            self.handle_exception(e)
            traceback.print_exc()
            raise

    def post_op(self, i, ret_code):
        try:
          children = self.children[i]
          parents = self.parents[i]
          self.set_inst_block_status(i, EC(ret_code))
          self.inst_blocks[i].clear()
          child_futures = []
          ready_children = []
          for child in children:
            val = self.block_ready_statuses[child].incr()
            print("op {0} Child {1}, parents {2} ready_val {3}".format(i, child, self.parents[child], val))
            print(self.block_ready_statuses[child].key)
            if (val >= len(self.parents[child])):
              ready_children.append(child)
          sqs = boto3.resource('sqs')
          queue = sqs.Queue(self.queue_url)
          self.inst_blocks[i].end_time = time.time()
          self.set_profiling_info(i)
          self.inst_blocks[i].end_time = time.time()
          inst_block = self.inst_blocks[i]
          pipelined_time = inst_block.end_time - inst_block.start_time
          full_times = [inst.end_time  - inst.start_time for inst in inst_block.instrs]
          print("Finished {0}, all children are {1}, ready children are {2}".format(i, children, ready_children))
          for child in ready_children:
            print("Adding {0} to sqs queue".format(child))
            queue.send_message(MessageBody=str(child))
            pass
        except Exception as e:
            print("POST OP EXCEPTION ", e)
            self.handle_exception(e)
            traceback.print_exc()
            raise

    async def post_op_async(self, i, ret_code, loop=None):
        if (loop == None):
          loop = asyncio.get_event_loop()
        try:
          children = self.children[i]
          parents = self.parents[i]
          await self.set_inst_block_status_async(i, EC(ret_code), loop=loop)
          self.inst_blocks[i].clear()
          child_futures = []
          ready_children = []
          for child in children:
            val = await self.block_ready_statuses[child].incr_async(loop=loop)
            if (val >= len(self.parents[child])):
              ready_children.append(child)
          session = aiobotocore.get_session(loop=loop)
          sqs_client = session.create_client('sqs', use_ssl=False)
          self.inst_blocks[i].end_time = time.time()
          inst_block = self.inst_blocks[i]
          pipelined_time = inst_block.end_time - inst_block.start_time
          full_times = [inst.end_time  - inst.start_time for inst in inst_block.instrs]

          print("BLOCK {0} DONE BLOCK TOOK {1} seconds wall clock for {2} seconds of compute".format(i, pipelined_time, np.sum(full_times)))
          await self.set_profiling_info(i, loop)
          for child in ready_children:
            print("Adding {0} to sqs queue".format(child))
            await sqs_client.send_message(QueueUrl=self.queue_url, MessageBody=str(child))

        except Exception as e:
            print("EXCEPTION ", e)
            self.handle_exception(e)
            traceback.print_exc()
            raise

    def start(self):
        self.ret_status.put(EC.RUNNING.value)
        sqs = boto3.resource('sqs')
        queue = sqs.Queue(self.queue_url)
        for starter in self.starters:
          print("Enqueuing ", starter)
          queue.send_message(MessageBody=str(starter))

        return 0

    def handle_exception(self, error):
        e = EC.EXCEPTION.value
        self.ret_status.put(e)

    def program_status(self):
      status = self.ret_status.get()
      if (status == None):
        return EC.NOT_STARTED
      else:
        return EC(status)



    def wait(self, sleep_time=1):
        status = self.program_status()
        while (status == EC.RUNNING):
            time.sleep(sleep_time)
            status = self.program_status()

    def free(self):
        client = boto3.client('sqs')
        client.delete_queue(QueueUrl=self.queue_url)

    def get_all_profiling_info(self):
        client = boto3.client('s3')
        return [self.get_profiling_info(i) for i in range(len(self.inst_blocks))]

    def get_profiling_info(self, pc):
        client = boto3.client('s3')
        byte_string = client.get_object(Bucket=self.bucket, Key="{0}/{1}".format(self.hash, pc))["Body"].read()
        return pickle.loads(byte_string)

    def set_profiling_info(self, pc):
        inst_block = self.inst_blocks[pc]
        serializer = serialize.SerializeIndependent()
        byte_string = serializer([inst_block])[0][0]
        client = boto3.client('s3')
        print("TYPE IS ", type(byte_string))
        client.put_object(Bucket=self.bucket, Key="{0}/{1}".format(self.hash, pc), Body=byte_string)


    async def set_profiling_info_async(self, pc, loop=None):
        if (loop == None):
          loop = asyncio.get_event_loop()
        inst_block = self.inst_blocks[pc]
        serializer = serialize.SerializeIndependent()
        byte_string = serializer([inst_block])[0][0]
        session = aiobotocore.get_session(loop=loop)
        client = session.create_client('s3', use_ssl=False)
        client = boto3.client('s3')
        print("TYPE IS ", type(byte_string))
        client.put_object(Bucket=self.bucket, Key="{0}/{1}".format(self.hash, pc), Body=byte_string)

    def inst_block_status(self, i):
      status = self.block_return_statuses[i].get()
      if (status == None):
        return EC.NOT_STARTED
      else:
        return EC(status)

    def set_inst_block_status(self, i, status):
        return self.block_return_statuses[i].put(status.value)

    async def set_inst_block_status_async(self, i, status, loop):
        return await self.block_return_statuses[i].put_async(status.value, loop)

    def _io_dependency_analyze(self, instruction_blocks):
        all_forward_dependencies = [set() for i in range(len(instruction_blocks))]
        all_backward_dependencies = [set() for i in range(len(instruction_blocks))]
        for i, inst_0 in enumerate(instruction_blocks):
            # find all places inst_0 reads
            deps = []
            for inst in inst_0.instrs:
              if isinstance(inst, RemoteLoad) or isinstance(inst, RemoteLoad):
                    deps.append(inst)
            deps_managed = set()
            for j, inst_1 in enumerate(instruction_blocks):
                 for inst in inst_1.instrs:
                    if isinstance(inst, RemoteWrite) or isinstance(inst, RemoteWrite):
                        for d in deps:
                            if (d.matrix == inst.matrix and d.bidxs == inst.bidxs):
                                # this is a dependency
                                if d in deps_managed:
                                    raise Exception("Each load should correspond to exactly one write")
                                deps_managed.add(d)
                                all_forward_dependencies[j].add(i)
                                all_backward_dependencies[i].add(j)
        return all_forward_dependencies, all_backward_dependencies


    def __str__(self):
      return "\n".join([str(i) + "\n" + str(x) + "children: \n" + str(self.children[i]) + "\n parents: \n" + str(self.parents[i]) for i,x in enumerate(self.inst_blocks)])


def make_column_update(pc, L_out, L_in, b0, b1, label=None):
    L_load = RemoteLoad(pc, L_in, b0, b1)
    pc += 1
    L_bb_load = RemoteLoad(pc, L_out.T, b1, b1)
    pc += 1
    trsm = RemoteTRSM(pc, [L_bb_load, L_load], lower=False, right=True)
    pc += 1
    write = RemoteWrite(pc, L_out, trsm, b0, b1)
    return InstructionBlock([L_load, L_bb_load, trsm, write], label=label), 4

def make_low_rank_update(pc, L_out, L_prev, L_final,  b0, b1, b2, label=None):
    old_block_load = RemoteLoad(pc, L_prev, b1, b2)
    pc += 1
    block_1_load = RemoteLoad(pc, L_final, b1, b0)
    pc += 1
    block_2_load = RemoteLoad(pc, L_final, b2, b0)
    pc += 1
    syrk = RemoteSYRK(pc, [old_block_load, block_1_load, block_2_load])
    pc += 1
    write = RemoteWrite(pc, L_out, syrk, b1, b2)
    return InstructionBlock([old_block_load, block_1_load, block_2_load, syrk, write], label=label), 5

def make_remote_trisolve(pc, X, A, B, b0, b1, lower=False, label=None): 
    block_A = RemoteLoad(pc, A, b0, b0)
    pc += 1
    block_B = RemoteLoad(pc, B, b0, b0)
    pc += 1
    trsm = RemoteTRSM(pc, [block_A, block_B], lower=lower, right=False)
    pc += 1
    write_trisolve = RemoteWrite(pc, X, trsm, b0, b1)
    pc += 1
    return InstructionBlock([block_A, block_B, trsm, write_trisolve], label=label), 4 

def make_local_cholesky(pc, L_out, L_in, b0, label=None):
    block_load = RemoteLoad(pc, L_in, b0, b0)
    pc += 1
    cholesky = RemoteCholesky(pc, [block_load])
    pc += 1
    write_diag = RemoteWrite(pc, L_out, cholesky, b0, b0)
    pc += 1
    return InstructionBlock([block_load, cholesky, write_diag], label=label), 3


def make_remote_gemm(pc, XY, X, Y, b0, b1, b2, label=None):
    # download row_b0[b2]
    # download col_b1[b2]
    # compute row_b0[b2].T.dot(col_b1[b2])

    block_0_load = RemoteLoad(pc, X, b0, b1)
    pc += 1
    block_1_load = RemoteLoad(pc, Y, b1, b2)
    pc += 1
    matmul = RemoteGemm(pc, [block_0_load, block_1_load])
    pc += 1
    write_out = RemoteWrite(pc, XY, matmul, b0, b1)
    pc += 1
    return InstructionBlock([block_0_load, block_1_load, matmul, write_out], label=label), 4


def _gemm(X, Y,out_bucket=None, tasks_per_job=1):
    reduce_idxs = Y._block_idxs(axis=1)
    if (out_bucket == None):
        out_bucket = X.bucket

    root_key = generate_key_name_binop(X, Y, "gemm")

    if (X.key == Y.key and (X.transposed ^ Y.transposed)):
        XY = BigSymmetricMatrix(root_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]])
    else:
        XY = BigMatrix(root_key, shape=(X.shape[0], Y.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], Y.shard_sizes[0]])

    num_out_blocks = len(XY.blocks)
    num_jobs = int(num_out_blocks/float(tasks_per_job))
    block_idxs_to_map = list(set(XY.block_idxs))
    chunked_blocks = list(chunk(list(chunk(block_idxs_to_map, tasks_per_job)), num_jobs))
    all_futures = []

    for i, c in enumerate(chunked_blocks):
        print("Submitting job for chunk {0} in axis 0".format(i))
        s = time.time()
        futures = pwex.map(pywren_run, c)
        e = time.time()
        print("Pwex Map Time {0}".format(e - s))
        all_futures.append((i,futures))
    return instruction_blocks



def _trisolve(A, B, lower=False, out_bucket=None):
    if out_bucket is None:
        out_bucket = A.bucket
    out_key = generate_key_name_binop(A, B, "trisolve")
    # generate output matrix
    X = BigMatrix(out_key,
                  shape=(B.shape[0], B.shape[1]),
                  bucket=out_bucket,
                  shard_sizes=[B.shard_sizes[0], B.shard_sizes[1]],
                  parent_fn=constant_zeros,
                  write_header=True)

    scratch = []
    for i,j0 in enumerate(B._block_idxs(1)):
        col_size = min(B.shard_sizes[1], B.shape[1] - B.shard_sizes[1]*j0)
        scratch.append(BigMatrix(out_key + "_{0}_scratch".format(i),
                       shape=[A.shape[0], col_size * len(B._block_idxs(0))],
                       bucket=out_bucket,
                       shard_sizes=[A.shard_sizes[0], col_size],
                       parent_fn=constant_zeros,
                       write_header=True))
    pc = 0
    all_instructions = []
    for i in B._block_idxs(1):
        block_idxs = A._block_idxs(0) 
        if not lower:
            block_idxs = reversed(block_idxs) 
        for j in block_idxs:
            if lower:
                start = 0
                end = j
            else:
                start = j + 1
                end = len(A._block_idxs(1))
            for k in A._block_idxs(1)[start:end]:
                instructions, count = make_remote_gemm(pc, scratch[i], A, X, j, k, i,
                                                       label="gemm_{0}_{1}_{2}".format(i, j, k))
                all_instructions.append(instructions)
                pc += count

            acc_load = RemoteLoad(pc, B, j, i)
            pc += 1
            sub_instructions = [acc_load]
            for k in A._block_idxs(1)[start:end]:
                block_load = RemoteLoad(pc, scratch[i], j, k)
                pc += 1
                acc_load = RemoteSub(pc, [acc_load, block_load])
                pc += 1
                sub_instructions += [block_load, acc_load]
            write_out = RemoteWrite(pc, scratch[i], acc_load, j, j)
            pc += 1
            sub_instructions.append(write_out)
            all_instructions.append(InstructionBlock(sub_instructions, label="sub_{0}_{1}".format(i, j)))
            instructions, count = make_remote_trisolve(pc, X, A, scratch[i], j, i, lower=lower, label="trisolve") 
            all_instructions.append(instructions)
            pc += count
    return all_instructions, X, scratch 


def _chol(X, out_bucket=None):
    if (out_bucket == None):
        out_bucket = X.bucket
    out_key = generate_key_name_uop(X, "chol")
    # generate output matrix
    L = BigMatrix(out_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]], parent_fn=constant_zeros, write_header=True)
    # generate intermediate matrices
    trailing = [X]
    all_blocks = list(L.block_idxs)
    block_idxs = sorted(X._block_idxs(0))

    for i,j0 in enumerate(X._block_idxs(0)):
        L_trailing = BigMatrix(out_key + "_{0}_trailing".format(i),
                       shape=(X.shape[0], X.shape[0]),
                       bucket=out_bucket,
                       shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]],
                       parent_fn=constant_zeros)
        block_size =  min(X.shard_sizes[0], X.shape[0] - X.shard_sizes[0]*j0)
        trailing.append(L_trailing)
    trailing.append(L)
    all_instructions = []

    pc = 0
    par_block = 0
    for i in block_idxs:
        instructions, count = make_local_cholesky(pc, trailing[-1], trailing[i], i, label="local")
        all_instructions.append(instructions)
        pc += count
        par_count = 0
        parallel_block = []
        for j in block_idxs[i+1:]:
            instructions, count = make_column_update(pc, trailing[-1], trailing[i], j, i, label="parallel_block_{0}_job_{1}".format(par_block, par_count))
            all_instructions.append(instructions)
            pc += count
            par_count += 1
        #all_instructions.append(PywrenInstructionBlock(pwex, parallel_block))
        par_block += 1
        par_count = 0
        parallel_block = []
        for j in block_idxs[i+1:]:
            for k in block_idxs[i+1:]:
                if (k > j): continue
                instructions, count = make_low_rank_update(pc, trailing[i+1], trailing[i], trailing[-1], i, j, k, label="parallel_block_{0}_job_{1}".format(par_block, par_count))
                all_instructions.append(instructions)
                pc += count
                par_count += 1
        #all_instructions.append(PywrenInstructionBlock(pwex, parallel_block))
    return all_instructions, trailing[-1], trailing[:-1]


def perf_profile(blocks, num_bins=100):
    READ_INSTRUCTIONS = [OC.S3_LOAD]
    WRITE_INSTRUCTIONS = [OC.S3_WRITE, OC.RET]
    COMPUTE_INSTRUCTIONS = [OC.SYRK, OC.TRSM, OC.INVRS, OC.CHOL]
    # first flatten into a single instruction list
    instructions = [inst for block in blocks for inst in block.instrs]
    start_times = [inst.start_time for inst in instructions]
    end_times = [inst.end_time for inst in instructions]

    abs_start = min(start_times)
    last_end = max(end_times)
    tot_time = (last_end - abs_start)
    bins = np.linspace(0, tot_time, tot_time)
    total_flops_per_sec = np.zeros(len(bins))
    read_bytes_per_sec = np.zeros(len(bins))
    write_bytes_per_sec = np.zeros(len(bins))
    runtimes = []

    for i,inst in enumerate(instructions):
        duration = inst.end_time - inst.start_time
        if (inst.i_code in READ_INSTRUCTIONS):
            start_time = inst.start_time - abs_start
            end_time = inst.end_time - abs_start
            start_bin, end_bin = np.searchsorted(bins, [start_time, end_time])
            size = inst.size
            bytes_per_sec = size/duration
            gb_per_sec = bytes_per_sec/1e9
            read_bytes_per_sec[start_bin:end_bin]  += gb_per_sec

        if (inst.i_code in WRITE_INSTRUCTIONS):
            start_time = inst.start_time - abs_start
            end_time = inst.end_time - abs_start
            start_bin, end_bin = np.searchsorted(bins, [start_time, end_time])
            size = inst.size
            bytes_per_sec = size/duration
            gb_per_sec = bytes_per_sec/1e9
            write_bytes_per_sec[start_bin:end_bin]  += gb_per_sec

        if (inst.i_code in COMPUTE_INSTRUCTIONS):
            start_time = inst.start_time - abs_start
            end_time = inst.end_time - abs_start
            start_bin, end_bin = np.searchsorted(bins, [start_time, end_time])
            flops = inst.flops
            flops_per_sec = flops/duration
            gf_per_sec = flops_per_sec/1e9
            total_flops_per_sec[start_bin:end_bin]  += gf_per_sec
        runtimes.append(end_time - start_time)
    optimes = defaultdict(int)
    opcounts = defaultdict(int)
    offset = instructions[0].start_time
    for inst, t in zip(instructions, runtimes):
      opcounts[inst.i_code] += 1
      optimes[inst.i_code] += t
      IO_INSTRUCTIONS = [OC.S3_LOAD, OC.S3_WRITE, OC.RET]
      if (inst.i_code not in IO_INSTRUCTIONS):
        flops = inst.flops/1e9
      else:
        flops = None
      print("{0}  {1}  {2} {3} gigaflops".format(str(inst), inst.start_time - offset, inst.end_time - offset, flops))
    for k in optimes.keys():
      print("{0}: {1}s".format(k, optimes[k]/opcounts[k]))
    return read_bytes_per_sec, write_bytes_per_sec, total_flops_per_sec, bins , instructions, runtimes



