import argparse
from numpywren import lambdapack as lp
import pywren
import concurrent.futures as fs
import hashlib
import numpy as np
from numpywren.matrix import BigMatrix
from numpywren.matrix_init import shard_matrix
from numpywren import job_runner
import numpywren.binops as binops
from pywren.serialize import serialize
import os
import time
import boto3
import redis
import pickle
import os
import hashlib
import matplotlib
# so plots work in headless mode
matplotlib.use('Agg')
import seaborn as sns
from pylab import plt
import logging
import copy
import pywren.wrenconfig as wc


REDIS_ADDR = os.environ.get("REDIS_ADDR", "127.0.0.1")
REDIS_PASS = os.environ.get("REDIS_PASS", "")
REDIS_PORT = os.environ.get("REDIS_PORT", "9001")
INFO_FREQ = 5

''' OSDI numpywren optimization effectiveness experiments '''

def run_experiment(problem_size, shard_size, pipeline, priority, lru, eager, truncate, max_cores, start_cores, trial, launch_granularity, timeout, log_granularity, autoscale_policy, standalone):
    # set up logging
    logger = logging.getLogger()
    for key in logging.Logger.manager.loggerDict:
        logging.getLogger(key).setLevel(logging.CRITICAL)
    logger.setLevel(logging.DEBUG)
    arg_bytes = pickle.dumps((problem_size, shard_size, pipeline, priority, lru, eager, truncate, max_cores, start_cores, trial, launch_granularity, timeout, log_granularity, autoscale_policy))
    arg_hash = hashlib.md5(arg_bytes).hexdigest()
    log_file = "optimization_experiments/{0}.log".format(arg_hash)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logging to {0}".format(log_file))

    X = np.random.randn(problem_size, 1)
    if standalone:
        redis_env ={"REDIS_ADDR": os.environ.get("REDIS_ADDR", ""), "REDIS_PASS": os.environ.get("REDIS_PASS", ""), "AWS_ACCESS_KEY_ID" : "AKIAIV3ENRQOI3FET2YA", "AWS_SECRET_ACCESS_KEY": "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", "OMP_NUM_THREADS":"1"}
        config = wc.default()
        config['runtime']['s3_bucket'] = 'pictureweb'
        config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-numpywren_avx512.tar.gz'
        pwex = pywren.standalone_executor(config=config)
    else:
        redis_env ={"REDIS_ADDR": os.environ.get("REDIS_ADDR", ""), "REDIS_PASS": os.environ.get("REDIS_PASS", "")}
        config = wc.default()
        config['runtime']['s3_bucket'] = 'pictureweb'
        config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz'
        pwex = pywren.default_executor(config=config)

    shard_sizes = [shard_size, 1]
    X_sharded = BigMatrix("cholesky_test_{0}_{1}".format(problem_size, shard_size), shape=X.shape, shard_sizes=shard_sizes, write_header=True)
    shard_matrix(X_sharded, X)
    print("Generating PSD matrix...")
    t = time.time()
    XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, overwrite=False)
    e = time.time()
    print("GEMM took {0}".format(e - t))
    XXT_sharded.lambdav = problem_size*10
    instructions ,L_sharded,trailing= lp._chol(XXT_sharded)
    pipeline_width = args.pipeline
    if (priority):
        num_priorities = 5
    else:
        num_priorities = 1
    if (lru):
        cache_size = 5
    else:
        cache_size = 0

    REDIS_CLIENT = redis.StrictRedis(REDIS_ADDR, port=REDIS_PORT, password=REDIS_PASS, db=0, socket_timeout=5)

    if (truncate is not None):
        instructions = instructions[:truncate]
    config = pwex.config

    program = lp.LambdaPackProgram(instructions, executor=pywren.lambda_executor, pywren_config=config, num_priorities=num_priorities, eager=eager)


    done_counts = []
    ready_counts = []
    post_op_counts = []
    not_ready_counts = []
    running_counts = []
    sqs_invis_counts = []
    sqs_vis_counts = []
    up_workers_counts = []
    busy_workers_counts = []
    times = []
    flops = []
    reads = []
    writes = []
    print("LRU", lru)
    print("eager", eager)
    exp = {}
    exp["redis_done_counts"] = done_counts
    exp["redis_ready_counts"] = ready_counts
    exp["redis_post_op_counts"] = post_op_counts
    exp["redis_not_ready_counts"] = not_ready_counts
    exp["redis_running_counts"] = running_counts
    exp["sqs_invis_counts"] = sqs_invis_counts
    exp["sqs_vis_counts"] = sqs_vis_counts
    exp["busy_workers"] = busy_workers_counts
    exp["up_workers"] = up_workers_counts
    exp["times"] = times
    exp["lru"] = lru
    exp["priority"] = priority
    exp["eager"] = eager
    exp["truncate"] = truncate
    exp["max_cores"] = max_cores
    exp["problem_size"] = problem_size
    exp["shard_size"] = shard_size
    exp["pipeline"] = pipeline
    exp["flops"] = flops
    exp["reads"] = reads
    exp["writes"] = writes
    exp["trial"] = trial
    exp["launch_granularity"] = launch_granularity
    exp["log_granularity"] = log_granularity
    exp["autoscale_policy"] = autoscale_policy 
    exp["standalone"] = standalone 


    logger.info("Longest Path: {0}".format(program.longest_path))
    program.start()
    t = time.time()
    logger.info("Starting with {0} cores".format(start_cores))
    all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(start_cores), extra_env=redis_env)
   # print([f.result() for f in all_futures])
    start_time = time.time()
    last_run_time = start_time

    while(program.program_status() == lp.PS.RUNNING):
        curr_time = int(time.time() - start_time)
        max_pc = program.get_max_pc()
        times.append(int(time.time()))
        time.sleep(log_granularity)
        waiting = 0
        running = 0
        for i, queue_url in enumerate(program.queue_urls):
            client = boto3.client('sqs')
            attrs = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible'])['Attributes']
            waiting += int(attrs["ApproximateNumberOfMessages"])
            running += int(attrs["ApproximateNumberOfMessagesNotVisible"])
        sqs_invis_counts.append(running)
        sqs_vis_counts.append(waiting)
        busy_workers = REDIS_CLIENT.get("{0}_busy".format(program.hash))
        if (busy_workers == None):
            busy_workers = 0
        else:
            busy_workers = int(busy_workers)
        up_workers = program.get_up()

        if (up_workers == None):
            up_workers = 0
        else:
            up_workers = int(up_workers)
        up_workers_counts.append(up_workers)
        busy_workers_counts.append(busy_workers)

        logger.debug("Waiting: {0}, Currently Processing: {1}".format(waiting, running))
        logger.debug("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))
        if ((curr_time % INFO_FREQ) == 0):
            logger.info("Max PC is {0}".format(max_pc))
            logger.info("Waiting: {0}, Currently Processing: {1}".format(waiting, running))
            logger.info("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))

        #print("{5}: Not Ready: {0}, Ready: {1}, Running: {4}, Post OP: {2},  Done: {3}".format(not_ready_count, ready_count, post_op_count, done_count, running_count, curr_time))
        current_gflops = program.get_flops()
        if (current_gflops is None):
            current_gflops = 0
        else:
            current_gflops = int(current_gflops)/1e9

        flops.append(current_gflops)
        current_gbytes_read = program.get_read()
        if (current_gbytes_read is None):
            current_gbytes_read = 0
        else:
            current_gbytes_read = int(current_gbytes_read)/1e9

        reads.append(current_gbytes_read)
        current_gbytes_write = program.get_write()
        if (current_gbytes_write is None):
            current_gbytes_write = 0
        else:
            current_gbytes_write = int(current_gbytes_write)/1e9
        writes.append(current_gbytes_write)
        #print("{0}: Total GFLOPS {1}, Total GBytes Read {2}, Total GBytes Write {3}".format(curr_time, current_gflops, current_gbytes_read, current_gbytes_write))

        time_since_launch = time.time() - last_run_time
        if (autoscale_policy == "dynamic"):
            if (time_since_launch > launch_granularity and up_workers < np.ceil(waiting*0.5/pipeline_width) and up_workers < max_cores):
                cores_to_launch = int(min(np.ceil(waiting/pipeline_width) - up_workers, max_cores - up_workers))
                logger.info("launching {0} new tasks....".format(cores_to_launch))
                new_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(cores_to_launch), extra_env=redis_env)
                last_run_time = time.time()
                # check if we OOM-erred
               # [x.result() for x in all_futures]
                all_futures.extend(new_futures)
        elif (autoscale_policy == "constant_timeout"):
            if (time_since_launch > (0.99*timeout)):
                cores_to_launch = max_cores
                logger.info("launching {0} new tasks....".format(cores_to_launch))
                new_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(cores_to_launch), extra_env=redis_env)
                last_run_time = time.time()
                # check if we OOM-erred
               # [x.result() for x in all_futures]
                all_futures.extend(new_futures)
        else:
            raise Exception("unknown autoscale policy")


    exp["all_futures"] = all_futures
    doubles = 0

    for pc in range(program.num_inst_blocks):
        run_count = REDIS_CLIENT.get("{0}_{1}_start".format(program.hash, pc))
        if (run_count is None):
            run_count = 0
        else:
            run_count = int(run_count)

        if (run_count != 1):
            logger.warn("PC: {0}, Run Count: {1}".format(pc, run_count))
            doubles += 1

    print("Number of repeats: {0}".format(doubles))
    e = time.time()
    time.sleep(10)
    logger.info(program.program_status())
    logger.info("PROGRAM STATUS " + str(program.program_status()))
    logger.info("PROGRAM HASH " + str(program.hash))
    logger.info("Took {0} seconds".format(e - t))
    # collect in
    executor = fs.ThreadPoolExecutor(72)
    futures = []
    for i in range(0,program.num_inst_blocks,1):
        futures.append(executor.submit(program.get_profiling_info, i))
    res = fs.wait(futures)
    profiled_blocks = [f.result() for f in futures]
    serializer = serialize.SerializeIndependent()
    byte_string = serializer([profiled_blocks])[0][0]
    exp["profiled_block_pickle_bytes"] = byte_string

    read,write,total_flops,bins, instructions, runtimes = lp.perf_profile(profiled_blocks, num_bins=100)
    flop_rate = sum(total_flops)/max(bins)
    exp["flop_rate"] = flop_rate
    print("Average Flop rate of {0}".format(flop_rate))
    # save other stuff
    try:
        os.mkdir("optimization_experiments/")
    except FileExistsError:
        pass
    exp_bytes = pickle.dumps(exp)
    dump_path = "optimization_experiments/{0}.pickle".format(arg_hash)
    print("Dumping experiment pickle to {0}".format(dump_path))
    with open(dump_path, "wb+") as f:
        f.write(exp_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run OSDI optimization effectiveness experiments')
    parser.add_argument("problem_size", type=int)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument('--truncate', type=int, default=None)
    parser.add_argument('--max_cores', type=int, default=32)
    parser.add_argument('--start_cores', type=int, default=32)
    parser.add_argument('--pipeline', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=200)
    parser.add_argument('--autoscale_policy', type=str, default="constant_timeout")
    parser.add_argument('--log_granularity', type=int, default=1)
    parser.add_argument('--launch_granularity', type=int, default=60)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--priority', action='store_true')
    parser.add_argument('--lru', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--standalone', action='store_true')
    args = parser.parse_args()
    run_experiment(args.problem_size, args.shard_size, args.pipeline, args.priority, args.lru, args.eager, args.truncate, args.max_cores, args.start_cores, args.trial, args.launch_granularity, args.timeout, args.log_granularity, args.autoscale_policy, args.standalone)



