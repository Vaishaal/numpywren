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
import numpywren as npw
import dill



INFO_FREQ = 5


''' OSDI numpywren optimization effectiveness experiments '''

def run_experiment(problem_size, shard_size, pipeline, priority, lru, eager, truncate, max_cores, start_cores, trial, launch_granularity, timeout, log_granularity, autoscale_policy, standalone, warmup, verify):
    # set up logging
    logger = logging.getLogger()
    region = wc.default()["account"]["aws_region"]
    for key in logging.Logger.manager.loggerDict:
        logging.getLogger(key).setLevel(logging.CRITICAL)
    logger.setLevel(logging.DEBUG)
    arg_bytes = pickle.dumps((problem_size, shard_size, pipeline, priority, lru, eager, truncate, max_cores, start_cores, trial, launch_granularity, timeout, log_granularity, autoscale_policy))
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

    X = np.random.randn(problem_size, 1)
    if standalone:
        extra_env ={"AWS_ACCESS_KEY_ID" : os.environ["AWS_ACCESS_KEY_ID"], "AWS_SECRET_ACCESS_KEY": os.environ["AWS_ACCESS_KEY_ID"], "OMP_NUM_THREADS":"1", "AWS_DEFAULT_REGION":region}
        config = wc.default()
        config['runtime']['s3_bucket'] = 'numpywrenpublic'
        config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz'
        pwex = pywren.standalone_executor(config=config)
    else:
        extra_env = {"AWS_DEFAULT_REGION":region}
        config = wc.default()
        config['runtime']['s3_bucket'] = 'numpywrenpublic'
        config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-numpywren.tar.gz'
        pwex = pywren.default_executor(config=config)
    if (warmup):
        def warmup_fn(x):
            time.sleep(200)
        futures = pwex.map(warmup_fn, range(max_cores))
        pywren.wait(futures)
    shard_sizes = [shard_size, 1]
    X_sharded = BigMatrix("cholesky_test_{0}_{1}".format(problem_size, shard_size), shape=X.shape, shard_sizes=shard_sizes, write_header=True, autosqueeze=False)
    shard_matrix(X_sharded, X)
    print("Generating PSD matrix...")
    t = time.time()
    print(X_sharded.shape)
    XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, overwrite=False)
    e = time.time()
    print("GEMM took {0}".format(e - t))
    XXT_sharded.lambdav = problem_size*10
    print(XXT_sharded.get_block(0,0))
    if (verify):
        A = XXT_sharded.numpy()
        print("Computing local cholesky")
        L = np.linalg.cholesky(A)

    instructions, trailing, L_sharded = compiler._chol(XXT_sharded)
    pipeline_width = args.pipeline
    if (priority):
        num_priorities = 5
    else:
        num_priorities = 1
    if (lru):
        cache_size = 5
    else:
        cache_size = 0

    pywren_config = pwex.config
    config = npw.config.default()
    program = lp.LambdaPackProgram(instructions, executor=pywren.lambda_executor, pywren_config=pywren_config, num_priorities=num_priorities, eager=eager, config=config)
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
    exp["program"] = program
    exp["time_steps"] = 1
    exp["failed"] = False


    program.start()
    t = time.time()
    logger.info("Starting with {0} cores".format(start_cores))
    #job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout)
    all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(start_cores), extra_env=extra_env)
    # print([f.result() for f in all_futures])
    start_time = time.time()
    last_run_time = start_time
    try:

        while(program.program_status() == lp.PS.RUNNING):
            curr_time = int(time.time() - start_time)
            p = program.get_progress()
            if (p is None):
               continue
            else:
               p = int(p)
            max_pc = p
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

            gflops_rate = flops[-1]/(times[-1] - times[0])

            print("{0}: Total GFLOPS {1}, Total GBytes Read {2}, Total GBytes Write {3}, Gflops Rate {4}".format(curr_time, current_gflops, current_gbytes_read, current_gbytes_write, gflops_rate))

            time_since_launch = time.time() - last_run_time
            if (autoscale_policy == "dynamic"):
                if (time_since_launch > launch_granularity and up_workers < np.ceil(waiting*0.5/pipeline_width) and up_workers < max_cores):
                    cores_to_launch = int(min(np.ceil(waiting/pipeline_width) - up_workers, max_cores - up_workers))
                    logger.info("launching {0} new tasks....".format(cores_to_launch))
                    new_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(cores_to_launch), extra_env=extra_env)
                    last_run_time = time.time()
                    # check if we OOM-erred
                   # [x.result() for x in all_futures]
                    all_futures.extend(new_futures)
            elif (autoscale_policy == "constant_timeout"):
                if (time_since_launch > (0.99*timeout)):
                    cores_to_launch = max_cores
                    logger.info("launching {0} new tasks....".format(cores_to_launch))
                    new_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=timeout), range(cores_to_launch), extra_env=extra_env)
                    last_run_time = time.time()
                    # check if we OOM-erred
                   # [x.result() for x in all_futures]
                    all_futures.extend(new_futures)
            else:
                raise Exception("unknown autoscale policy")
            exp["time_steps"] += 1
        if (verify):
            L_sharded_local = L_sharded.numpy()
            print("max diff", np.max(np.abs(L_sharded_local - L)))
    except KeyboardInterrupt:
        exp["failed"] = True
        program.stop()
        pass
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
    parser.add_argument('--timeout', type=int, default=200)
    parser.add_argument('--autoscale_policy', type=str, default="constant_timeout")
    parser.add_argument('--log_granularity', type=int, default=1)
    parser.add_argument('--launch_granularity', type=int, default=60)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--priority', action='store_true')
    parser.add_argument('--lru', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--standalone', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    run_experiment(args.problem_size, args.shard_size, args.pipeline, args.priority, args.lru, args.eager, args.truncate, args.max_cores, args.start_cores, args.trial, args.launch_granularity, args.timeout, args.log_granularity, args.autoscale_policy, args.standalone, args.warmup, args.verify)



