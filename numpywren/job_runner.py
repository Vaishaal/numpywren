import time
import concurrent.futures as fs
import boto3
from numpywren import lambdapack as lp
import traceback
from multiprocessing.dummy import Pool as ThreadPool
import aiobotocore
import asyncio


class LambdaPackExecutor(object):
    def __init__(self, program, loop, pipeline_width=5):
        self.read_executor = None
        self.write_executor = None
        self.compute_executor = None
        self.loop
        self.program = program
        self.parent_children = {}
        self.ret_status_map = {}
        self.pc_block_map = {}
        self.block_ends= set()
        self.computer = fs.ThreadPoolExecutor(1)
        self.reader = fs.ThreadPoolExecutor(1)
        self.scheduler_thread = fs.ThreadPoolExecutor(1)
        self.scheduler_thread.submit(self.scheduler)


    async def run(self, pc, local_ret_status):
        self.program.pre_op(pc, self.loop)
        instrs = self.program.inst_blocks[pc].instrs
        # first instruction in every instruction block is executable!
        for i,inst in enumerate(instrs[0:-1]):
            self.parent_children[inst] = instrs[i+1]
        for i,inst in enumerate(instrs):
            self.pc_block_map[inst] = pc
        ret_codes = []
        try:
            for instr in instrs:
                if (instr.i_code == lp.OC.S3_WRITE):
                    ret_code = await self.loop.run_in_executor(self.writer, instr)
                elif (instr.i_code == lp.OC.S3_LOAD):
                    ret_code = await self.loop.run_in_executor(self.reader, instr)
                elif (instr.i_code == lp.OC.RET):
                    ret_code = await self.loop.run_in_executor(self.writer, instr)
                else:
                    ret_code = await self.loop.run_in_executor(self.computer, instr)
                if ret_code != 0: raise
            self.program.post_op(pc, lp.EC.SUCCESS)
        except Exception as e:
            self.program.post_op(pc, lp.EC.EXCEPTION)
            traceback.print_exc()
            local_ret_status[0] = 0
            self.loop.stop()
            raise
        local_ret_status[0] = 0
        return

def lambdapack_run(program, pipeline_width=5, msg_vis_timeout=2):
    loop = asyncio.get_event_loop()
    loop.create_task(lambdapack_run_async, loop, program, pipeline_width, msg_vis_timeout)
    loop.create_task(check_program_state, program, loop)
    loop.run_forever()
    loop.close()


async def reset_msg_visibility(msg, queue_url, client, not_done, timeout):
    print("Starting message visibility resetter for msg ")
    while(not_done[0]):
        receipt_handle = msg["ReceiptHandle"]
        await client.change_message_visibility(VisibilityTimeout=timeout*2, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        asyncio.sleep(timeout)
    receipt_handle = msg["ReceiptHandle"]
    await client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    return 0

async def check_program_state(program, loop):
    while(True):
        #TODO make this an s3 access as opposed to DD access since we don't *really need* atomicity here
        #TODO make this coroutine friendly
        if(program.program_status() != lp.EC.RUNNING):
            loop.stop()
        # DD is expensive so sleep alot
        asyncio.sleep(10)

async def lambdapack_run_async(loop, program, pipeline_width=5, msg_vis_timeout=2):
    session = aiobotocore.get_session(loop=loop)
    sqs_client = session.create_client('sqs')
    lmpk_executor = LambdaPackExecutor(program, loop)
    reset_thread = fs.ThreadPoolExecutor(pipeline_width)
    workers = fs.ThreadPoolExecutor(pipeline_width)

    while(True):
        asyncio.sleep(0.1)
        messages = await sqs_client.receive_messages(QueueUrl=program.queue_url, MaxNumberOfMessages=1)
        if (len(messages) == 0):
            continue
        msg = messages[0]
        pc = int(msg["Body"])
        # start the asyncio loop
        local_ret_status = [1]
        loop.run_in_executor(reset_thread, reset_msg_visibility, program.queue_url, sqs_client, msg, local_ret_status, timeout)
        loop.run_in_executor(workers, lmpk_executor.run, pc, local_ret_status)


















