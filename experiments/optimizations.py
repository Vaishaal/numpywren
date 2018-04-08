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



''' OSDI numpywren optimization effectiveness experiments '''

TIMEOUT = 180
EST_FLOP_RATE = 24
def run_experiment(problem_size, shard_size, pipeline, priority, lru, eager, truncate, cores):
    X = np.random.randn(problem_size, 1)
    pwex = pywren.default_executor()
    shard_sizes = [shard_size, 1]
    X_sharded = BigMatrix("cholesky_test_{0}_{1}".format(problem_size, shard_size), shape=X.shape, shard_sizes=shard_sizes, write_header=True)
    shard_matrix(X_sharded, X)
    print("Generating PSD matrix...")
    XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, overwrite=False)
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

    if (truncate is not None):
        instructions = instructions[:truncate]
    config = pwex.config

    program = lp.LambdaPackProgram(instructions, executor=pywren.lambda_executor, pywren_config=config, num_priorities=num_priorities, eager=eager)
    redis_env ={"REDIS_IP": os.environ.get("REDIS_IP", ""), "REDIS_PASS": os.environ.get("REDIS_PASS", "")}


    done_counts = []
    ready_counts = []
    post_op_counts = []
    not_ready_counts = []
    running_counts = []
    sqs_invis_counts = []
    sqs_vis_counts = []

    exp = {}
    exp["redis_done_counts"] = done_counts
    exp["redis_ready_counts"] = ready_counts
    exp["redis_post_op_counts"] = post_op_counts
    exp["redis_not_ready_counts"] = not_ready_counts
    exp["redis_running_counts"] = running_counts
    exp["sqs_invis_counts"] = sqs_invis_counts
    exp["sqs_vis_counts"] = sqs_vis_counts
    exp["lru"] = lru
    exp["priority"] = priority
    exp["eager"] = eager
    exp["truncate"] = truncate
    exp["cores"] = cores
    exp["problem_size"] = problem_size
    exp["shard_size"] = shard_size
    exp["pipeline"] = pipeline

    print("Longest Path: {0}".format(program.longest_path))
    program.start()
    t = time.time()
    all_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=TIMEOUT), range(cores), extra_env=redis_env)
    start_time = time.time()
    last_run = time.time()
    times = []
    while(program.program_status() == lp.PS.RUNNING):
        max_pc = program.get_max_pc()
        print("Max PC is {0}".format(max_pc))
        time.sleep(5)
        waiting = 0
        running = 0
        done_count = 0
        ready_count = 0
        post_op_count = 0
        not_ready_count = 0
        running_count = 0
        for pc in range(len(program.inst_blocks)):
            ns = program.get_node_status(pc)
            done_count += (ns  == lp.NS.FINISHED)
            not_ready_count += (ns  == lp.NS.NOT_READY)
            post_op_count += (ns  == lp.NS.POST_OP)
            ready_count += (ns == lp.NS.READY)
            running_count += (ns == lp.NS.RUNNING)
        print("{5}: Not Ready: {0}, Ready: {1}, Running: {4}, Post OP: {2},  Done: {3}".format(not_ready_count, ready_count, post_op_count, done_count, running_count, int(time.time() - start_time)))
        done_counts.append(done_count)
        not_ready_counts.append(not_ready_count)
        post_op_counts.append(post_op_count)
        ready_counts.append(ready_count)
        running_counts.append(running_counts)
        times.append(time.time())

        if (time.time() - last_run > TIMEOUT - 10):
            new_futures = pwex.map(lambda x: job_runner.lambdapack_run(program, pipeline_width=pipeline_width, cache_size=cache_size, timeout=TIMEOUT), range(cores), extra_env=redis_env)
            print(new_futures)
            print("previous status")
            pywren.wait(all_futures)
            print([f.result() for f in all_futures])
            all_futures = new_futures
            last_run = time.time()
        for i, queue_url in enumerate(program.queue_urls):
            client = boto3.client('sqs')
            attrs = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible'])['Attributes']
            waiting += int(attrs["ApproximateNumberOfMessages"])
            running += int(attrs["ApproximateNumberOfMessagesNotVisible"])
        if (ready_count > 0 and waiting == 0 and running == 0):
            print("BAD STATE...printing redis state")
            for pc in range(len(program.inst_blocks)):
                #ns = program.get_node_status(pc)
                #done_count += (ns  == lp.NS.FINISHED)
                #not_ready_count += (ns  == lp.NS.NOT_READY)
                #post_op_count += (ns  == lp.NS.POST_OP)
                #ready_count += (ns == lp.NS.READY)
                #running_count += (ns == lp.NS.RUNNING)
                print("PC: {0}, Edge Sum: {1}, Indegree: {2}, Node Status {3}".format(pc, lp.get(program._node_edge_sum_key(pc)), len(program.parents[pc]), program.get_node_status(pc)))
                #pywren.wait(all_futures)

        sqs_invis_counts.append(running)
        sqs_vis_counts.append(waiting)

        print("SQS INVIS : {0},  SQS VIS {1}".format(running, waiting))
    e = time.time()
    print(program.program_status())
    print("PROGRAM STATUS ", program.program_status())
    print("PROGRAM HASH", program.hash)
    print("Took {0} seconds".format(e - t))
    time.sleep(5)
    # collect in
    executor = fs.ThreadPoolExecutor(72)
    futures = []
    for i in range(0,len(program.inst_blocks),1):
        futures.append(executor.submit(program.get_profiling_info, i))
    res = fs.wait(futures)
    profiled_blocks = [f.result() for f in futures]
    serializer = serialize.SerializeIndependent()
    byte_string = serializer([profiled_blocks])[0][0]
    exp["profiled_block_pickle_bytes"] = byte_string

    arg_bytes = pickle.dumps((problem_size, shard_size, pipeline, priority, lru, eager, truncate, cores))
    arg_hash = hashlib.md5(arg_bytes).hexdigest()
    read,write,total_flops,bins, instructions, runtimes = lp.perf_profile(profiled_blocks, num_bins=100)
    sns.tsplot(total_flops, time=(bins - min(bins)), condition="Observed Performance")
    c = sns.palettes.color_palette()[1]
    plt.xlabel("Time since start")
    plt.ylabel("aggregate Gflops/s")
    plt.hlines(EST_FLOP_RATE*cores,0,10000, label="Theoretical Performance", color=c)
    lgd = plt.legend(bbox_to_anchor=(1.5, 0.8))
    plt.savefig("optimization_experiments/flop_rate_{0}.pdf".format(arg_hash), bbox_extra_artists=(lgd,), bbox_inches='tight')

    flop_rate = sum(total_flops)/max(bins)
    exp["flop_rate"] = flop_rate
    print("Average Flop rate of {0}".format(flop_rate))
    # save other stuff
    print("Args Hash is {0}".format(arg_hash))
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
    parser.add_argument('--cores', type=int, default=16)
    parser.add_argument('--pipeline', type=int, default=1)
    parser.add_argument('--priority', action='store_true')
    parser.add_argument('--lru', action='store_true')
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    run_experiment(args.problem_size, args.shard_size, args.pipeline, args.priority, args.lru, args.eager, args.truncate, args.cores)



