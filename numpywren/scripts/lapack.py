import numpywren as npw
import boto3
import shutil
import io
import click
import os
import tarfile
import urllib
import subprocess
import concurrent.futures as fs

LAPACK_VERSION = "3.8.0"
LAPACK_URL = f"http://www.netlib.org/lapack/lapack-{LAPACK_VERSION}.tar.gz"

@click.group()
def lapack():
    pass



@click.command()
@click.option('--force', is_flag=True)
def download_runtime(force):
    if (not os.path.exists("/tmp/condaruntime/") or force):
        shutil.rmtree("/tmp/condaruntime/", ignore_errors=True)
        config = npw.config.default()
        bucket = config['runtime']['bucket']
        key = config['runtime']['s3_key']
        client = boto3.client('s3')
        tar_bytes = client.get_object(Key=key, Bucket=bucket)["Body"].read()
        obj = io.BytesIO(tar_bytes)
        tar = tarfile.open(fileobj=obj)
        tar.extractall("/tmp/")

@click.command()
@click.option('--force', is_flag=True)
def download_lapack(force):
    if (not os.path.exists("/tmp/lapack/") or force):
        shutil.rmtree("/tmp/lapack/", ignore_errors=True)
        response = urllib.request.urlopen(LAPACK_URL)
        data = response.read()
        obj = io.BytesIO(data)
        tar = tarfile.open(fileobj=obj)
        tar.extractall("/tmp/", )
        shutil.move(f"/tmp/lapack-{LAPACK_VERSION}", "/tmp/lapack")


@click.command()
@click.argument('function')
@click.option('--full_name', is_flag=True)

@click.option('--num_shards', default=100)
@click.option('--num_threads', default=32)
@click.option('--blas', is_flag=True)
def export_function(function, full_name, num_shards, num_threads, blas):
    config = npw.config.default()
    #bucket = config['s3']['bucket']
    bucket = "numpywrenpublic"
    client = boto3.client('s3')
    if (not full_name):
        function_full = function + ".f"
    else:
        function_full = function
        function = function.replace('.f', '')
    if (blas):
        subprocess.check_output(f"cd /tmp/; /tmp/condaruntime/bin/f2py -c /tmp/lapack/BLAS/SRC/{function_full} -m {function} -L/tmp/condaruntime/lib/ -lmkl_rt", shell=True)
    else:
        subprocess.check_output(f"cd /tmp/; /tmp/condaruntime/bin/f2py -c /tmp/lapack/SRC/{function_full} -m {function} -L/tmp/condaruntime/lib/ -lmkl_rt", shell=True)
    out_name = f"{function}.cpython-36m-x86_64-linux-gnu.so"
    print("bucketpotato", bucket)
    tp = fs.ThreadPoolExecutor(num_threads)
    futures = []
    with open(f"/tmp/{out_name}", 'rb') as f:
        f_bytes = f.read()
        client.put_object(Bucket=bucket, Key=f"lapack/{out_name}", Body=f_bytes)
        for i in range(1, num_shards):
            future = tp.submit(client.copy_object, Bucket=bucket, Key=f"lapack/{out_name}_{i}", CopySource=f"{bucket}/lapack/{out_name}")
            futures.append(future)
    fs.wait(futures)
    [f.result() for f in futures]


lapack.add_command(download_runtime)
lapack.add_command(download_lapack)
lapack.add_command(export_function)

