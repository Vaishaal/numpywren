import numpywren
import numpywren.matrix
from numpywren.matrix import BigMatrix, BigSymmetricMatrix, Scalar
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

try:
  DEFAULT_CONFIG = wc.default()
except:
  DEFAULT_CONFIG = {}



class RemoteInstructionOpCodes(Enum):
    S3_LOAD = 0
    S3_WRITE = 1
    SYRK = 2
    TRSM = 3
    CHOL = 4
    INVRS = 5
    RET = 6


class RemoteInstructionExitCodes(Enum):
    SUCCESS = 0
    RUNNING = 1
    EXCEPTION = 2
    REPLAY = 3
    NOT_STARTED = 4

EC = RemoteInstructionExitCodes
OC = RemoteInstructionOpCodes


class RemoteInstruction(object):
    def __init__(self, i_id):
        self.id = i_id
        self.ret_code = -1

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
        if (self.result == None):
            self.result = self.matrix.get_block(*self.bidxs)
            self.ret_code = 0
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
        if (self.result == None):
            self.result = self.matrix.put_block(self.data_instr.result, *self.bidxs)
            self.ret_code = 0
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
        if (self.result == None):
            old_block = self.argv[0].result
            block_2 = self.argv[1].result
            block_1 = self.argv[2].result
            self.result = old_block - block_2.dot(block_1.T)
            self.ret_code = 0
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
        if (self.result == None):
            L_bb_inv = self.argv[1].result
            col_block = self.argv[0].result
            self.result = col_block.dot(L_bb_inv.T)
            self.ret_code = 0
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
        if (self.result == None):
            L_bb = self.argv[0].result
            self.result = np.linalg.cholesky(L_bb)
            self.ret_code = 0
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
        if (self.result == None):
            L_bb = self.argv[0].result
            self.result = np.linalg.inv(L_bb)
            self.ret_code = 0
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
        if (self.result == None):
          self.return_loc.put(EC.SUCCESS.value)
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
        self.bucket = pwex.config['s3']['bucket']
        self.inst_blocks = [copy.copy(x) for x in inst_blocks]
        self.program_string = "\n".join([str(x) for x in inst_blocks])
        program_string = "\n".join([str(x) for x in self.inst_blocks])
        hashed = hashlib.sha1()
        hashed.update(program_string.encode())
        self.hash = hashed.hexdigest()

        self.ret_matrix = Scalar(self.hash, dtype=np.uint8)
        # delete it if it exists
        self.ret_matrix.free()
        self.ret_matrix.parent_fn =  matrix_utils.make_constant_parent(EC.NOT_STARTED.value)
        self.children, self.parents = self._io_dependency_analyze(self.inst_blocks)
        self.starters = []
        self.terminators = []

        max_i_id = max([inst.id for inst_block in self.inst_blocks for inst in inst_block.instrs])


        self.remote_return = RemoteReturn(max_i_id + 1, self.ret_matrix)
        self.pc = max_i_id + 1
        self.block_return_matrices = []
        for i, (children, parents, inst_block) in enumerate(zip(self.children, self.parents, self.inst_blocks)):
            if len(children) == 0:
                self.terminators.append(i)
            if len(parents) == 0:
                self.starters.append(i)
            if (len(children) > 0):
               block_hash = hashlib.sha1((self.hash + str(i)).encode()).hexdigest()
               block_ret_matrix  = Scalar(block_hash, dtype=np.uint8)
               block_ret_matrix.free()
               block_ret_matrix.put(EC.NOT_STARTED.value)
               block_return = RemoteReturn(self.pc + 1, block_ret_matrix)
               self.inst_blocks[i].instrs.append(block_return)
            else:
               block_ret_matrix = self.ret_matrix
            self.block_return_matrices.append(block_ret_matrix)
        for i in self.terminators:
            self.inst_blocks[i].instrs.append(self.remote_return)


    def pywren_func(self, i):
        children = self.children[i]
        parents = self.parents[i]
        # 1. check if program has terminated
        # 2. check if this instruction_block has executed successfully
        # 3. check if parents are not completed
        # if any of the above are False -> exit
        try:
          program_status = self.program_status().value
          parents_status = [self.inst_block_status(i).value for i in parents]
          my_status = self.inst_block_status(i).value
          if (sum(parents_status) != EC.SUCCESS.value):
              return self.inst_blocks[i], EC.RUNNING
          elif (program_status != EC.RUNNING.value):
              return self.inst_blocks[i], EC.EXCEPTION
          elif (my_status != EC.NOT_STARTED.value):
              return self.inst_blocks[i], EC.REPLAY
          self.set_inst_block_status(i, EC.RUNNING)
          ret_code = self.inst_blocks[i]()
          self.set_inst_block_status(i, EC(ret_code))
          self.inst_blocks[i].clear()
          child_futures = []
          pwex = self.executor(config=self.pywren_config)
          for child in children:
              child_futures.append(pwex.call_async(self.pywren_func, child))
          return self.inst_blocks[i], child_futures
        except Exception as e:
            self.handle_exception(e)
            raise

    def start(self):
        self.ret_matrix.parent_fn =  matrix_utils.make_constant_parent(EC.RUNNING.value)
        pwex = pywren.default_executor()
        futures = []
        for i in self.starters:
            futures.append(pwex.call_async(self.pywren_func, i))
        return futures

    def handle_exception(self, error):
        e = EC.EXCEPTION.value
        self.ret_matrix.put(e)

    def program_status(self):
        return EC(self.ret_matrix.get())


    def wait(self, sleep_time=1):
        status = self.program_status()
        while (status == EC.RUNNING):
            time.sleep(sleep_time)
            status = self.program_status()

    def inst_block_status(self, i):
        return EC(self.block_return_matrices[i].get())

    def set_inst_block_status(self, i, status):
        return self.block_return_matrices[i].put(status.value)

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



