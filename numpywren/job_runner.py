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
        self.loop = loop
        self.program = program
        self.parent_children = {}
        self.ret_status_map = {}
        self.pc_block_map = {}
        self.block_ends= set()
        self.computer = fs.ThreadPoolExecutor(1)
        self.reader = fs.ThreadPoolExecutor(1)
        self.writer = fs.ThreadPoolExecutor(1)


    async def run(self, pc, local_ret_status):
        print("In Run for {0}".format(pc))
        self.program.pre_op(pc)
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
                    print("Waiting on write")
                    res = await self.loop.run_in_executor(self.writer, instr)
                    print("Write done")
                elif (instr.i_code == lp.OC.S3_LOAD):
                    print("Waiting on read")
                    res = await self.loop.run_in_executor(self.reader, instr)
                    print("Read done")
                elif (instr.i_code == lp.OC.RET):
                    print("Waiting on return")
                    res = await self.loop.run_in_executor(self.writer, instr)
                    print("Return done")
                else:
                    print("Waiting on compute")
                    res = await self.loop.run_in_executor(self.computer, instr)
                    print("Compute done")
                    res = await self.loop.run_in_executor(self.computer, instr)
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
    loop.create_task(lambdapack_run_async(loop, program, pipeline_width, msg_vis_timeout))
    loop.create_task(check_program_state(program, loop))
    loop.run_forever()
    loop.close()


async def reset_msg_visibility(msg, queue_url, client, not_done, timeout, all_msgs):
    print("Starting message visibility resetter for msg ")
    while(not_done[0]):
        receipt_handle = msg["ReceiptHandle"]
        await client.change_message_visibility(VisibilityTimeout=timeout*2, QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        await asyncio.sleep(timeout)
    receipt_handle = msg["ReceiptHandle"]
    await client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    del all_msgs[receipt_handle]
    return 0

async def check_program_state(program, loop):
    while(True):
        #TODO make this an s3 access as opposed to DD access since we don't *really need* atomicity here
        #TODO make this coroutine friendly
        if(program.program_status() != lp.EC.RUNNING):
            break
        # DD is expensive so sleep alot
        await asyncio.sleep(10)
    loop.stop()

async def lambdapack_run_async(loop, program, pipeline_width=5, msg_vis_timeout=2):
    session = aiobotocore.get_session(loop=loop)
    sqs_client = session.create_client('sqs')
    lmpk_executor = LambdaPackExecutor(program, loop)
    reset_thread = fs.ThreadPoolExecutor(pipeline_width)
    workers = fs.ThreadPoolExecutor(pipeline_width)
    all_messages = {}
    while(True):
        print("Main Loop")
        await asyncio.sleep(0.1)
        if (len(list(all_messages.keys())) >= pipeline_width):
            print("Full queue")
            print(all_messages)
            continue
        messages = await sqs_client.receive_message(QueueUrl=program.queue_url, MaxNumberOfMessages=1)
        # if the queue is empty
        if ("Messages" not in messages):
            continue
        msg = messages['Messages'][0]
        all_messages[msg["ReceiptHandle"]] = msg
        pc = int(msg["Body"])
        # start the asyncio loop
        local_ret_status = [1]
        print("scheduling message reset visibility")
        coro = reset_msg_visibility(msg, program.queue_url, sqs_client, local_ret_status, msg_vis_timeout, all_messages)
        loop.create_task(coro)
        loop.create_task(lmpk_executor.run(pc, local_ret_status))


















