import time
import concurrent.futures as fs

class LambdaPackExecutor(object):
    def __init__(self, program, pipeline_width):
        self.instruction_queue = []
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.parent_children = {}
        self.ret_status_map = {}
        self.computer = fs.ThreadPoolExecutor(1)
        self.writer = fs.ThreadPoolExecutor(1)
        self.reader = fs.ThreadPoolExecutor(1)
        self.scheduler_thread = fs.ThreadPoolExecutor(1)
        self.scheduler_thread.run(self.scheduler)

    def push(self, pc, local_ret_status):
        while(True):
            if (len(self.instruction_queue) <= pipeline_width):
                # unpack instructions
                insts = self.program.inst_blocks[pc].instrs
                # first instruction in every instruction block is executable!
                self.instruction_queue.append(insts[0])
                for i,inst in instrs[0:-1]:
                    self.parent_children[inst] = self.inst[i+1]
                self.ret_status_map[insts[-1]] = local_ret_status
                break
            else:
                time.sleep(1)

    def scheduler(self):
        while(True):
            # by invariant inforced in push this should always be fine to run
            if (len(self.instruction_queue) == 0):
                time.sleep(1)
            instr = self.instruction_queue.pop(0)
            if (instr.i_code == lp.OC.S3_WRITE):
                self.writer.run(self,runner, instr)
            elif (instr.i_code == lp.OC.S3_READ):
                self.reader.run(self,runner, instr)
            elif (instr.i_code == lp.OC.RET):
                self.writer.run(self.runner, instr)
            else:
                self.computer.run(instr)
            time.sleep(1)

    def runner(self, op):
        # do the thang
        op()
        if op in self.ret_status_map:
            # mark as done so we release the SQS message
            self.ret_status_map[op][0] = 0
        if op in self.parent_children:
            self.insts.append(self.parent_children[op])

def reset_msg_visibility(msg, not_done, timeout):
    print("Starting message visibility resetter")
    while(not_done[0]):
        time.sleep(timeout/2)
        msg.change_visibility(VisibilityTimeout=timeout)
    msg.delete()
    return 0

def lambdapack_run(sqs_queue, program, pipeline_width=5, msg_vis_timeout=2):
    start_time = time.time()
    executor = LambdaPackExecutor()
    reset_thread = fs.ThreadPoolExecutor(pipeline_width)
    while(True):
        msg = sqs_queue.dequeue()
        pc = int(msg)
        local_ret_status = [1]
        reset_thread.run(reset_msg_visibility, msg, local_ret_status, msg_vis_timeout)
        executor.push(pc, local_ret_status)
    end_time = time.time()




