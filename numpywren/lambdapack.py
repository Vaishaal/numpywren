import numpywren
import numpywren.matrix
from .matrix import BigMatrix, BigSymmetricMatrix, Scalar
from .matrix_utils import load_mmap, chunk, generate_key_name_uop, constant_zeros
import numpy as np
import pywren
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

try:
  DEFAULT_CONFIG = wc.default()
except:
  DEFAULT_CONFIG = {}


class LocalExecutor(object):
  def __init__(self, procs=32, config=DEFAULT_CONFIG):
    self.procs = procs
    self.executor = fs.ThreadPoolExecutor(max_workers=procs)
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
    CHOL = 4
    INVRS = 5
    RET = 6
    EXIT = 7


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
  def __init__(self, key, table_name="lambdapack"):
    self.key = {"id": {"S":key}}
    self.table_name = table_name

  def put(self, value):
    assert isinstance(value, int)
    item = self.key.copy()
    item["val"] = {"N": str(value)}
    client = boto3.client('dynamodb', region_name='us-west-2')
    client.put_item(TableName=self.table_name, Item=item)

  def get(self):
    client = boto3.client('dynamodb', region_name='us-west-2')
    resp = client.get_item(TableName=self.table_name, Key=self.key, ConsistentRead=True)
    if "Item" in resp.keys():
      return int(resp["Item"]["val"]["N"])
    else:
      return None

  def incr(self, inc=1):
    client = boto3.client('dynamodb', region_name='us-west-2')
    assert isinstance(inc, int)
    done = False
    while (not done):
      try:
        old_val = self.get()
        if (old_val is None):
          update_value = {":newval":{"N":str(inc)}}
          update = "ADD val :newval"
          cond = "attribute_not_exists(Id)"
          client.update_item(TableName=self.table_name, Key=self.key, UpdateExpression=update, ExpressionAttributeValues=update_value, ConditionExpression=cond)
          final_val = inc
          done = True
        else:
          update_value = {":newval":{"N":str(old_val+inc)}, ":oldval":{"N":str(old_val)}}
          update = "SET val = :newval"
          cond = "val = :oldval"
          client.update_item(TableName=self.table_name, Key=self.key, UpdateExpression=update, ExpressionAttributeValues=update_value, ConditionExpression=cond)
          done = True
          final_val = old_val + inc
      except botocore.exceptions.ClientError as e:
        # Ignore the ConditionalCheckFailedException, bubble up
        # other exceptions.
        if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
          raise
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

    def __call__(self):
        self.start_time = time.time()
        if (self.result == None):
            self.result = self.matrix.get_block(*self.bidxs)
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
        return "{0} = S3_LOAD {1} {2}".format(self.id, self.matrix, len(self.bidxs), bidxs_str.strip())

class RemoteWrite(RemoteInstruction):
    def __init__(self, i_id, matrix, data_instr, *bidxs):
        super().__init__(i_id)
        self.i_code = OC.S3_WRITE
        self.matrix = matrix
        self.bidxs = bidxs
        self.data_instr = data_instr
        self.result = None

    def __call__(self):
        self.start_time = time.time()
        if (self.result == None):
            self.result = self.matrix.put_block(self.data_instr.result, *self.bidxs)
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
    def __call__(self):
        self.start_time = time.time()
        if (self.result == None):
            old_block = self.argv[0].result
            block_2 = self.argv[1].result
            block_1 = self.argv[2].result
            self.result = old_block - block_2.dot(block_1.T)
            self.flops = old_block.size + 2*block_2.shape[0]*block_2.shape[1]*block_1.shape[0]
            self.ret_code = 0
        self.end_time = time.time()
        return self.result

    def clear(self):
        self.result = None


    def __str__(self):
        return "{0} = SYRK {1} {2} {3}".format(self.id, self.argv[0].id,  self.argv[1].id,  self.argv[2].id)

class RemoteTRSM(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.TRSM
        assert len(argv_instr) == 2
        self.argv = argv_instr
        self.result = None
    def __call__(self):
        self.start_time = time.time()
        if (self.result == None):
            L_bb_inv = self.argv[1].result
            col_block = self.argv[0].result
            self.result = col_block.dot(L_bb_inv.T)
            self.flops = 2*col_block.shape[0]*col_block.shape[1]*L_bb_inv.shape[0]
            self.ret_code = 0
        self.end_time = time.time()
        return self.result

    def clear(self):
        self.result = None

    def __str__(self):
        return "{0} = TRSM {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)

class RemoteCholesky(RemoteInstruction):
    def __init__(self, i_id, argv_instr):
        super().__init__(i_id)
        self.i_code = OC.CHOL
        assert len(argv_instr) == 1
        self.argv = argv_instr
        self.result = None
    def __call__(self):
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
    def __call__(self):
        self.start_time = time.time()
        if (self.result == None):
            L_bb = self.argv[0].result
            self.result = np.linalg.inv(L_bb)
            self.flops = 2.0/3.0*(L_bb.shape[0]**3)
            self.ret_code = 0
        self.end_time  = time.time()
        return self.result

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
    def __call__(self):
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
          self.children[i].append(len(self.inst_blocks) - 1)
        self.children.append([])
        self.parents.append(self.terminators)
        self.block_return_statuses.append(self.ret_status)
        self.ret_ready_status = RPS(self.hash + "_ready")
        self.block_ready_statuses.append(self.ret_ready_status)


    def pywren_func(self, i):
        # 1. check if program has terminated
        # 2. check if this instruction_block has executed successfully
        # 3. check if parents are not completed
        # if any of the above are False -> exit
        try:
          print("RUNNING " , i)
          children = self.children[i]
          parents = self.parents[i]
          program_status = self.program_status().value
          if (i == len(self.inst_blocks) - 1):
            # special case final block
            print("RETURN BLOCK")
            pass
          self.set_inst_block_status(i, EC.RUNNING)
          if (program_status != EC.RUNNING.value):
            return i, self.inst_blocks[i], EC.EXCEPTION

          ret_code = self.inst_blocks[i]()
          self.set_inst_block_status(i, EC(ret_code))
          self.inst_blocks[i].clear()
          child_futures = []
          pwex = self.executor(config=self.pywren_config)
          ready_children = []
          for child in children:
            val = self.block_ready_statuses[child].incr()
            if (val >= len(self.parents[child])):
              ready_children.append(child)
          child_futures = pwex.map(self.pywren_func, ready_children, extra_env={"OMP_NUM_THREADS": "1"})
          return i, self.inst_blocks[i], child_futures
        except Exception as e:
            print("EXCEPTION ", e)
            self.handle_exception(e)
            raise

    def start(self):
        self.ret_status.put(EC.RUNNING.value)
        pwex = self.executor(config=self.pywren_config)
        print(pwex.config)
        print(self.starters)
        self.futures = pwex.map(self.pywren_func, self.starters, extra_env={"OMP_NUM_THREADS": "1"})
        return self.futures

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


    def unwind(self):
      if (self.program_status() != EC.SUCCESS):
        raise Exception("Unwind can only be called when program has succeeded")
      results = []
      for f in self.futures:
        results += self.__unwind_recursive(f)
      return sorted(results, key=lambda x: x[0])

    def __unwind_recursive(self, future):
      index, inst_block, children = future.result()
      if (isinstance(children, RemoteInstructionExitCodes)):
        return []
      results = [(index, inst_block)]
      for c in children:
        results += self.__unwind_recursive(c)
      return results

    def inst_block_status(self, i):
      status = self.block_return_statuses[i].get()
      if (status == None):
        return EC.NOT_STARTED
      else:
        return EC(status)

    def set_inst_block_status(self, i, status):
        return self.block_return_statuses[i].put(status.value)

    def _io_dependency_analyze(self, instruction_blocks):
        all_forward_dependencies = [[] for i in range(len(instruction_blocks))]
        all_backward_dependencies = [[] for i in range(len(instruction_blocks))]
        for i, inst_0 in enumerate(instruction_blocks):
            # find all places inst_0 reads
            deps = []
            for inst in inst_0.instrs:
                if isinstance(inst, RemoteLoad):
                    deps.append(inst)
            deps_managed = set()
            for j, inst_1 in enumerate(instruction_blocks):
                 for inst in inst_1.instrs:
                    if isinstance(inst, RemoteWrite):
                        for d in deps:
                            if (d.matrix == inst.matrix and d.bidxs == inst.bidxs):
                                # this is a dependency
                                if d in deps_managed:
                                    raise Exception("Each load should correspond to exactly one write")
                                deps_managed.add(d)
                                all_forward_dependencies[j].append(i)
                                all_backward_dependencies[i].append(j)
        return all_forward_dependencies, all_backward_dependencies


    def __str__(self):
        return "\n".join([str(x) for x in self.inst_blocks])


def make_column_update(pc, L_out, L_in, L_bb_inv, b0, b1, label=None):
    L_load = RemoteLoad(pc, L_in, b0, b1)
    pc += 1
    L_bb_inv_load = RemoteLoad(pc, L_bb_inv, 0, 0)
    pc += 1
    trsm = RemoteTRSM(pc, [L_load, L_bb_inv_load])
    pc += 1
    write = RemoteWrite(pc, L_out, trsm, b0, b1)
    return InstructionBlock([L_load, L_bb_inv_load, trsm, write], label=label), 4

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

def make_local_cholesky_and_inverse(pc, L_out, L_bb_inv_out, L_in, b0, label=None):
    block_load = RemoteLoad(pc, L_in, b0, b0)
    pc += 1
    cholesky = RemoteCholesky(pc, [block_load])
    pc += 1
    inverse = RemoteInverse(pc, [cholesky])
    pc += 1
    write_diag = RemoteWrite(pc, L_out, cholesky, b0, b0)
    pc += 1
    write_inverse = RemoteWrite(pc, L_bb_inv_out, inverse, 0, 0)
    pc += 1
    return InstructionBlock([block_load, cholesky, inverse, write_diag, write_inverse], label=label), 5


def make_remote_gemm(pc, XY, X, Y, b0, b1, b2, label=None):
    # download row_b0[b2]
    # download col_b1[b2]
    # compute row_b0[b2].T.dot(col_b1[b2])

    block_0_load = RemoteLoad(pc, X, b0, b2)
    pc += 1
    block_1_load = RemoteLoad(pc, X, b1, b2)
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



def _chol(X, out_bucket=None):
    if (out_bucket == None):
        out_bucket = X.bucket
    out_key = generate_key_name_uop(X, "chol")
    # generate output matrix
    L = BigMatrix(out_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]], parent_fn=constant_zeros, write_header=True)
    # generate intermediate matrices
    trailing = [X]
    cholesky_diag_inverses = []
    all_blocks = list(L.block_idxs)
    block_idxs = sorted(X._block_idxs(0))

    for i,j0 in enumerate(X._block_idxs(0)):
        L_trailing = BigMatrix(out_key + "_{0}_trailing".format(i),
                       shape=(X.shape[0], X.shape[0]),
                       bucket=out_bucket,
                       shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]],
                       parent_fn=constant_zeros)
        block_size =  min(X.shard_sizes[0], X.shape[0] - X.shard_sizes[0]*j0)
        L_bb_inv = BigMatrix(out_key + "_({0},{0})_inv".format(i),
                             shape=(block_size, block_size),
                             bucket=out_bucket,
                             shard_sizes=[block_size, block_size],
                             parent_fn=constant_zeros)

        trailing.append(L_trailing)
        cholesky_diag_inverses.append(L_bb_inv)
    trailing.append(L)
    all_instructions = []

    pc = 0
    par_block = 0
    for i in block_idxs:
        L_inv = cholesky_diag_inverses[i]
        instructions, count = make_local_cholesky_and_inverse(pc, trailing[-1], L_inv, trailing[i], i, label="local")
        all_instructions.append(instructions)
        pc += count
        par_count = 0
        parallel_block = []
        for j in block_idxs[i+1:]:
            instructions, count = make_column_update(pc, trailing[-1], trailing[i], L_inv, j, i, label="parallel_block_{0}_job_{1}".format(par_block, par_count))
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



