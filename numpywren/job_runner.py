import time
import concurrent.futures as fs
import boto3
from numpywren import lambdapack as lp
import traceback

class LambdaPackExecutor(object):
    def __init__(self, program, pipeline_width=5):
        self.instruction_queue = []
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.pipeline_width = pipeline_width
        self.program = program
        self.parent_children = {}
        self.ret_status_map = {}
        self.pc_block_map = {}
        self.block_ends= set()
        self.computer = fs.ThreadPoolExecutor(1)
        self.writer = fs.ThreadPoolExecutor(1)
        self.reader = fs.ThreadPoolExecutor(1)
        self.scheduler_thread = fs.ThreadPoolExecutor(1)
        self.scheduler_thread.submit(self.scheduler)

    def push(self, pc, local_ret_status):
        while(True):
            if (len(self.instruction_queue) <= self.pipeline_width):
                # unpack instructions
                self.program.pre_op(pc)
                instrs = self.program.inst_blocks[pc].instrs
                # first instruction in every instruction block is executable!
                for i,inst in enumerate(instrs[0:-1]):
                    self.parent_children[inst] = instrs[i+1]
                for i,inst in enumerate(instrs):
                    self.pc_block_map[inst] = pc

                self.ret_status_map[pc] = local_ret_status
                self.block_ends.add(instrs[-1])
                self.instruction_queue.append(instrs[0])
                break
            else:
                time.sleep(1)

    def scheduler(self):
        print("Scheduler started")
        while(True):
            try:
                print("Scheduler queue: ", self.instruction_queue)
                # by invariant inforced in push this should always be fine to run
                if (len(self.instruction_queue) == 0):
                    time.sleep(0.1)
                    if ((int(time.time()) % 5) == 0):
                        if(program.program_status() != lp.EC.RUNNING):
                            break
                    continue
                instr = self.instruction_queue.pop(0)
                if (instr.i_code == lp.OC.S3_WRITE):
                    self.writer.submit(self.runner, instr)
                elif (instr.i_code == lp.OC.S3_LOAD):
                    self.reader.submit(self.runner, instr)
                elif (instr.i_code == lp.OC.RET):
                    self.writer.submit(self.runner, instr)
                else:
                    print("Submitting to computer")
                    self.computer.submit(self.runner, instr)
            except Exception as e:
                print("Scheduler error")
                print("ERRROR IS ", e)
                traceback.print_exc()

    def runner(self, op):
        try:
            # do the thang
            print("Running", op)
            op()
            print("op finished", op)
            if op in self.block_ends:
                # mark as done so we release the SQS message
                pc = self.pc_block_map[op]
                self.ret_status_map[pc][0] = 0
                self.program.post_op(pc, lp.EC.SUCCESS)
                self.block_ends.remove(op)
            if op in self.parent_children:
                print("locally enqueueing children")
                self.instruction_queue.append(self.parent_children[op])
                del self.parent_children[op]
            print(self.instruction_queue)
        except Exception as e:
            print("Runner error")
            print(e)
            pc = self.pc_block_map[op]
            self.program.post_op(pc, lp.EC.EXCEPTION)
            self.ret_status_map[pc][0] = 0
            traceback.print_exc()

def reset_msg_visibility(msg, not_done, timeout):
    print("Starting message visibility resetter")
    while(not_done[0]):
        time.sleep(1)
        msg.change_visibility(VisibilityTimeout=10)
    msg.delete()
    return 0

def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=2):
    sqs = boto3.resource('sqs')
    sqs_queue = sqs.Queue(program.queue_url)
    start_time = time.time()
    executor = LambdaPackExecutor(program)
    reset_thread = fs.ThreadPoolExecutor(pipeline_width)
    while(True):
        time.sleep(0.1)
        if ((int(time.time()) % 5) == 0):
            if(program.program_status() != lp.EC.RUNNING):
                break
        messages = sqs_queue.receive_messages(MaxNumberOfMessages=1)
        if (len(messages) == 0):
            continue
        msg = messages[0]
        pc = int(msg.body)
        local_ret_status = [1]
        reset_thread.submit(reset_msg_visibility, msg, local_ret_status, msg_vis_timeout)
        executor.push(pc, local_ret_status)
    end_time = time.time()




