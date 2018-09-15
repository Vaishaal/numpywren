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
import os
import time
import boto3
import redis
import pickle
import os
import hashlib
import logging
import copy
import pywren.wrenconfig as wc
from numpywren import compiler
from numpywren.alg_wrappers import cholesky, tsqr, gemm, qr
import numpywren as npw
import dill
import traceback



INFO_FREQ = 5

def parse_int(x):
    if x is None: return 0
    return int(x)



''' NSDI qr effectiveness experiments '''

def run_experiment(problem_size, shard_size, pipeline, num_priorities, lru, eager, truncate, max_cores, start_cores, trial, launch_granularity, timeout, log_granularity, autoscale_policy, standalone, warmup, verify, matrix_exists, read_limit, write_limit, compute_threads_per_worker):
    # set up logging
    invoke_executor = fs.ThreadPoolExecutor(1)
    logger = logging.getLogger()
    region = wc.default()["account"]["aws_region"]
    print("REGION", region)
    for key in logging.Logger.manager.loggerDict:
        logging.getLogger(key).setLevel(logging.CRITICAL)
    logger.setLevel(logging.DEBUG)
    arg_bytes = pickle.dumps((problem_size, shard_size, pipeline, num_priorities, lru, eager, truncate, max_cores, start_cores, trial, launch_granularity, timeout, log_granularity, autoscale_policy, read_limit, write_limit))
    arg_hash = hashlib.md5(arg_bytes).hexdigest()
    log_file = "{0}.log".format(arg_hash)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logging to {0}".format(log_file))
    if standalone:
        extra_env ={"AWS_ACCESS_KEY_ID" : os.environ["AWS_ACCESS_KEY_ID"].strip() , "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"].strip(), "OMP_NUM_THREADS":"1", "AWS_DEFAULT_REGION":region}
        config = wc.default()
        config['runtime']['s3_bucket'] = 'numpywrenpublic'
        key = "pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz"
        config['runtime']['s3_key'] = key
        pwex = pywren.standalone_executor(config=config)
    else:
        extra_env = {"AWS_DEFAULT_REGION":region}
        config = wc.default()
        config['runtime']['s3_bucket'] = 'numpywrenpublic'
        key = "pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz"
        config['runtime']['s3_key'] = key
        print(config)
        pwex = pywren.default_executor(config=config)

    if (not matrix_exists):
        X = np.random.randn(problem_size, 1)
        shard_sizes = [shard_size, 1]
        X_sharded = BigMatrix("qr_test_{0}_{1}".format(problem_size, shard_size), shape=X.shape, shard_sizes=shard_sizes, write_header=True, autosqueeze=False, bucket="numpywrennsdi2")
        shard_matrix(X_sharded, X)
        print("Generating PSD matrix...")
        t = time.time()
        print(X_sharded.shape)
        XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, overwrite=False)
        e = time.time()
        print("GEMM took {0}".format(e - t))
    else:
        X_sharded = BigMatrix("qr_test_{0}_{1}".format(problem_size, shard_size), autosqueeze=False, bucket="numpywrennsdi2")
        key_name = binops.generate_key_name_binop(X_sharded, X_sharded.T, "gemm")
        XXT_sharded = BigMatrix(key_name, hash_keys=False, bucket="numpywrensdi2")
    XXT_sharded.lambdav = problem_size*10
    t = time.time()
    program, meta = qr(XXT_sharded)
    pipeline_width = args.pipeline
    if (lru):
        cache_size = 5
    else:
        cache_size = 0
    pywren_config = pwex.config
    e = time.time()
    print("Program compile took {0} seconds".format(e - t))
    print("program.hash", program.hash)
    REDIS_CLIENT = program.control_plane.client
    done_counts = []
    ready_counts = []
    post_op_counts = []
    not_ready_counts = []
    running_counts = []
    sqs_invis_counts = []
    sqs_vis_counts = []
    up_workers_counts = []
    busy_workers_counts = []
    read_objects = []
    write_objects = []
    all_read_timeouts = []
    all_write_timeouts = []
    all_redis_timeouts = []
    times = [time.time()]
    flops = [0]
    reads = [0]
    writes = [0]
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
    exp["priority"] = num_priorities 
    exp["eager"] = eager
    exp["truncate"] = truncate
    exp["max_cores"] = max_cores
    exp["problem_size"] = problem_size
    exp["shard_size"] = shard_size
    exp["pipeline"] = pipeline
    exp["flops"] = flops
    exp["reads"] = reads
    exp["writes"] = writes
    exp["read_objects"] = read_objects
    exp["write_objects"] = write_objects
    exp["read_timeouts"] = all_read_timeouts
    exp["write_timeouts"] = all_write_timeouts 
    exp["redis_timeouts"] = all_redis_timeouts 
    exp["trial"] = trial
    exp["launch_granularity"] = launch_granularity
    exp["log_granularity"] = log_granularity
    exp["autoscale_policy"] = autoscale_policy
    exp["standalone"] = standalone
    exp["program"] = program
    exp["time_steps"] = 1
    exp["failed"] = False


    program.start()
    t = time.time()
    logger.info("Starting with {0} cores".format(start_cores))
    all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(start_cores), extra_env=extra_env)
    start_time = time.time()
    last_run_time = start_time
    print(program.program_status())
    print("QUEUE URLS", len(program.queue_urls))
    total_lambda_epochs = start_cores
    try:
        while(program.program_status() == lp.PS.RUNNING):
            time.sleep(log_granularity)
            curr_time = int(time.time() - start_time)
            p = program.get_progress()
            if (p is None):
                print("no progress...")
                continue
            else:
               p = int(p)
            times.append(int(time.time()))
            max_pc = p
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

            logger.debug("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))
            if ((curr_time % INFO_FREQ) == 0):
                logger.info("Waiting: {0}, Currently Processing: {1}".format(waiting, running))
                logger.info("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))

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

            gflops_rate = flops[-1]/(times[-1] - times[0])
            greads_rate = reads[-1]/(times[-1] - times[0])
            gwrites_rate = writes[-1]/(times[-1] - times[0])
            b = XXT_sharded.shard_sizes[0]
            current_objects_read = (current_gbytes_read*1e9)/(b*b*8)
            current_objects_write = (current_gbytes_write*1e9)/(b*b*8)
            read_objects.append(current_objects_read)
            write_objects.append(current_objects_write)
            read_rate = read_objects[-1]/(times[-1] - times[0])
            write_rate = write_objects[-1]/(times[-1] - times[0])

            avg_workers = np.mean(up_workers_counts)
            smooth_len = 10
            if (len(flops) > smooth_len + 5):
                gflops_rate_5_min_window = (flops[-1] - flops[-smooth_len])/(times[-1] - times[-smooth_len])
                gread_rate_5_min_window = (reads[-1] - reads[-smooth_len])/(times[-1] - times[-smooth_len])
                gwrite_rate_5_min_window = (writes[-1] - writes[-smooth_len])/(times[-1] - times[-smooth_len])
                read_rate_5_min_window = (read_objects[-1] - read_objects[-smooth_len])/(times[-1] - times[-smooth_len])
                write_rate_5_min_window = (write_objects[-1] - write_objects[-smooth_len])/(times[-1] - times[-smooth_len])
                workers_5_min_window = np.mean(up_workers_counts[-smooth_len:])
            else:
                gflops_rate_5_min_window =  "N/A"
                gread_rate_5_min_window = "N/A"
                gwrite_rate_5_min_window = "N/A"
                workers_5_min_window = "N/A"
                read_rate_5_min_window = "N/A"
                write_rate_5_min_window = "N/A"

            read_timeouts = int(parse_int(REDIS_CLIENT.get("s3.timeouts.read")))
            write_timeouts = int(parse_int(REDIS_CLIENT.get("s3.timeouts.write")))
            redis_timeouts = int(parse_int(REDIS_CLIENT.get("redis.timeouts")))
            all_read_timeouts.append(read_timeouts)
            all_write_timeouts.append(write_timeouts)
            all_redis_timeouts.append(redis_timeouts)
            read_timeouts_fraction = read_timeouts/(current_objects_read+1e-8)
            write_timeouts_fraction = write_timeouts/(current_objects_write+1e-8)
            print("=======================================")
            print("Max PC is {0}".format(max_pc))
            print("Waiting: {0}, Currently Processing: {1}".format(waiting, running))
            print("{2}: Up Workers: {0}, Busy Workers: {1}".format(up_workers, busy_workers, curr_time))
            print("{0}: Total GFLOPS {1}, Total GBytes Read {2}, Total GBytes Write {3}".format(curr_time, current_gflops, current_gbytes_read, current_gbytes_write))
            print("{0}: Average GFLOPS rate {1}, Average GBytes Read rate {2}, Average GBytes Write  rate {3}, Average Worker Count {4}".format(curr_time, gflops_rate, greads_rate, gwrites_rate, avg_workers))
            print("{0}: Average read txns/s {1}, Average write txns/s {2}".format(curr_time, read_rate, write_rate))
            print("{0}: smoothed GFLOPS rate {1}, smoothed GBytes Read rate {2}, smoothed GBytes Write  rate {3}, smoothed Worker Count {4}".format(curr_time, gflops_rate_5_min_window, gread_rate_5_min_window, gwrite_rate_5_min_window, workers_5_min_window))
            print("{0}: smoothed read txns/s {1}, smoothed write txns/s {2}".format(curr_time, read_rate_5_min_window, write_rate_5_min_window))
            print("{0}: Read timeouts: {1}, Write timeouts: {2}, Redis timeouts: {3}  ".format(curr_time, read_timeouts, write_timeouts, redis_timeouts))
            print("{0}: Read timeouts fraction: {1}, Write timeouts fraction: {2}".format(curr_time, read_timeouts_fraction, write_timeouts_fraction))
            print("=======================================")

            time_since_launch = time.time() - last_run_time
            if (time_since_launch > (0.85*timeout)):
                break
            exp["time_steps"] += 1
    except KeyboardInterrupt:
        exp["failed"] = True
        program.stop()
        pass
    except Exception as e:
        traceback.print_exc()
        exp["failed"] = True
        program.stop()
        raise
        pass
    print("killing program...")
    print(program.program_status())
    exp["all_futures"] = all_futures
    exp_bytes = dill.dumps(exp)
    client = boto3.client('s3')
    client.put_object(Key="lambdapack/{0}/runtime.pickle".format(program.hash), Body=exp_bytes, Bucket=program.bucket)
    print("=======================")
    print("=======================")
    print("Execution Summary:")
    print("Executed Program ID: {0}".format(program.hash))
    print("Program Success: {0}".format((not exp["failed"])))
    print("Problem Size: {0}".format(exp["problem_size"]))
    print("Shard Size: {0}".format(exp["shard_size"]))
    print("Total Execution time: {0}".format(times[-1] - times[0]))
    print("Average Flop Rate (GFlop/s): {0}".format(exp["flops"][-1]/(times[-1] - times[0])))
    with open("/tmp/last_run", "w+") as f:
        f.write(program.hash)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run OSDI optimization effectiveness experiments')
    parser.add_argument("problem_size", type=int)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument('--truncate', type=int, default=0)
    parser.add_argument('--max_cores', type=int, default=32)
    parser.add_argument('--start_cores', type=int, default=32)
    parser.add_argument('--pipeline', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=140)
    parser.add_argument('--write_limit', type=int, default=1e6)
    parser.add_argument('--read_limit', type=int, default=1e6)
    parser.add_argument('--autoscale_policy', type=str, default="constant_timeout")
    parser.add_argument('--log_granularity', type=int, default=5)
    parser.add_argument('--launch_granularity', type=int, default=60)
    parser.add_argument('--compute_threads_per_worker', type=int, default=1)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--num_priorities', type=int, default=1)
    parser.add_argument('--lru', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--standalone', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--matrix_exists', action='store_true')
    args = parser.parse_args()
    run_experiment(args.problem_size, args.shard_size, args.pipeline, args.num_priorities, args.lru, args.eager, args.truncate, args.max_cores, args.start_cores, args.trial, args.launch_granularity, args.timeout, args.log_granularity, args.autoscale_policy, args.standalone, args.warmup, args.verify, args.matrix_exists, args.write_limit, args.read_limit, args.compute_threads_per_worker)



