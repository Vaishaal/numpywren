import pywren
import numpywren as npw
from numpywren import lambdapack as lp
from numpywren import control_plane
import argparse
import time
import concurrent.futures as fs

def warmup_fn(x, start, control_plane, warmup_sleep):
    client = control_plane.client
    lp.incr(client, str(start))
    time.sleep(warmup_sleep)
    lp.decr(client, str(start))

def delayed_exec(request, start, control_plane, warmup_sleep):
    pwex = pywren.default_executor()
    return pwex.map(lambda x: warmup_fn(x, start, control_plane, warmup_sleep), range(request))


def warmup(control_plane, num_lambdas, warmup_rate, warmup_sleep, start_lambdas):
    print(f"Starting warmup...to {num_lambdas} ")
    start = time.time()
    pwex = pywren.default_executor()
    request = min(num_lambdas, start_lambdas)
    print(f"requesting {request} lambdas...")
    executor = fs.ProcessPoolExecutor(2)
    future = executor.submit(delayed_exec, request, start, control_plane, warmup_sleep)
    curr_lambda_count = 0
    last_launch = start
    iter_count = 0
    while(True):
        curr_time = time.time()
        iter_count += 1
        val = lp.get(control_plane.client, str(start))
        if (val is None):
            val = 0
        else:
            val = int(val)
        time.sleep(1)
        curr_lambda_count = max(val, curr_lambda_count)
        if (curr_lambda_count >= num_lambdas):
            break
        if (iter_count % 10 == 0):
            print(f"Iteration {iter_count},  Num Lambdas: {val}")
        if (curr_time - last_launch > 0.8*warmup_sleep):
            last_launch = curr_time
            additional_lambdas = warmup_rate*int((curr_time - start)/60)
            request = min(num_lambdas, start_lambdas + additional_lambdas)
            print(f"requesting {request} lambdas...")
            future = executor.submit(delayed_exec, request, start, control_plane, warmup_sleep)
            curr_lambda_count = 0
    print(f"{num_lambdas} acquired!")














if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warmup lambdas")
    parser.add_argument("--num_lambdas", default=10000, type=int)
    parser.add_argument("--warmup_rate", default=500, type=int, help="How many lambdas do you want to increase per minute")
    parser.add_argument("--warmup_sleep", default=60, type=int, help="How many lambdas do you want to increase per minute")
    parser.add_argument("--start", default=3000, type=int, help="How many lambdas do you want to increase per minute")
    config = npw.config.default()
    control_plane = control_plane.get_control_plane(config=config)
    client = control_plane.client
    args = parser.parse_args()
    warmup(control_plane, args.num_lambdas, args.warmup_rate, args.warmup_sleep, args.start)



