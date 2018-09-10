import numpywren as npw
from numpywren import control_plane as cp
import argparse
import click
import os
import pywren
from pywren import wrenconfig
import boto3
import json
import re
import redis
import time
import yaml
from numpywren.matrix_utils import key_exists, list_all_keys
import boto3
import click
import concurrent.futures as fs
import io
import numpywren as npw
import os
import pandas as pd
import shutil
import subprocess
import tarfile
import urllib

GIT_URL = "https://github.com/Vaishaal/numpywren"
NUMPYWREN_SETUP =\
'''
Welcome to numpywren setup!
numpywren is software that allows for seamless large scale linear algebra computation. Please note
that numpywren is currently research grade software not intended for production use of any sort.
Are you okay with this?
'''


SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_username():
    return pwd.getpwuid(os.getuid())[0]

def check_overwrite_function(filename):
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        return click.confirm("{} already exists, would you like to overwrite?".format(filename))
    return True


def check_valid_lifespan(lifespan):
    try:
        int(lifespan)
        return True
    except ValueError:
        return False


def click_validate_prompt(message, default, validate_func=lambda x: True,
                          fail_msg="", max_attempts=5):
    """
    Click wrapper that repeats prompt until acceptable answer
    """
    attempt_num = 0
    while True:
        res = click.prompt(message, default)
        if validate_func(res):
            return res
        else:
            attempt_num += 1
            if attempt_num >= max_attempts:
                raise Exception("Too many invalid answers")
            if fail_msg != "":
                click.echo(fail_msg.format(res))

def check_bucket_exists(s3bucket):
    """
    This is the recommended boto3 way to check for bucket
    existence:
    http://boto3.readthedocs.io/en/latest/guide/migrations3.html
    """
    s3 = boto3.resource("s3")
    exists = True
    try:
        s3.meta.client.head_bucket(Bucket=s3bucket)
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            exists = False
        else:
            raise e
    return exists

def check_valid_bucket_name(bucket_name):
    return True



@click.group()
@click.option('--filename', default=npw.config.get_default_config_filename())
@click.pass_context
def cli(ctx, filename):
    ctx.obj = {'config_filename' : filename}

@click.group()
def control_plane():
    """
    control_plane subcommand
    """
    pass


@click.command()
def launch():
    t = time.time()
    click.echo("Launching instance...")
    info = cp.launch_and_provision_redis()
    ip = info["public_ip"]
    click.echo("Waiting for redis")
    config = npw.config.default()
    rc = config["control_plane"]
    password = rc["password"]
    port= rc["port"]
    redis_client = redis.StrictRedis(host=ip, port=port, password=password)
    while (True):
        try:
            redis_client.ping()
            break
        except redis.exceptions.ConnectionError as e:
            pass
    e = time.time()
    click.echo("redis launch took {0} seconds".format(e - t))
    cp.set_control_plane(info, config=config)

@click.command()
def list():
    config = npw.config.default()
    client = boto3.client('s3')
    rc = config["control_plane"]
    prefix = rc["control_plane_prefix"].strip("/")
    bucket = config["s3"]["bucket"]
    keys = list_all_keys(prefix=prefix, bucket=bucket)
    dicts = []
    for i,key in enumerate(keys):
        dicts.append(json.loads(client.get_object(Key=key, Bucket=config["s3"]["bucket"])["Body"].read()))
    if (len(dicts) > 0):
        # maybe custom pretty printing here, but pandas does a god enough job
        click.echo(pd.DataFrame(dicts))
    else:
        click.echo("No control planes found")



@click.command()
@click.argument('idx', default=0)
def terminate(idx):
    config = npw.config.default()
    client = boto3.client('s3')
    rc = config["control_plane"]
    prefix = rc["control_plane_prefix"].strip("/")
    bucket = config["s3"]["bucket"]
    keys = list_all_keys(prefix=prefix, bucket=bucket)
    if (idx >= len(keys)):
        click.echo("idx must be less that number of total control planes")
        return
    key = keys[idx]
    info = json.loads(client.get_object(Key=key, Bucket=config["s3"]["bucket"])["Body"].read())
    instance_id = info['id']
    ec2_client = boto3.client('ec2')
    click.echo("terminating control plane {0}".format(idx))
    resp = ec2_client.terminate_instances(InstanceIds=[instance_id])
    client.delete_object(Key=key, Bucket=bucket)


@click.command()
@click.argument('idx', default=0)
def info(idx):
    config = npw.config.default()
    client = boto3.client('s3')
    rc = config["control_plane"]
    password = rc["password"]
    port= rc["port"]
    prefix = rc["control_plane_prefix"].strip("/")
    bucket = config["s3"]["bucket"]
    keys= list_all_keys(prefix=prefix, bucket=bucket)
    if (idx >= len(keys)):
        click.echo("idx must be less that number of total control planes")
        return
    key = keys[idx]
    info = json.loads(client.get_object(Key=key, Bucket=config["s3"]["bucket"])["Body"].read())
    host = info["public_ip"]
    redis_client = redis.StrictRedis(host=host, port=port, password=password)
    while (True):
        try:
            info =  redis_client.info()
            for (k,v) in info.items():
                print("{0}: {1}".format(k,v))
            break
        except redis.exceptions.ConnectionError as e:
            click.echo("info failed.")
            pass


@click.command()
@click.argument('idx', default=0)
def ping(idx):
    config = npw.config.default()
    client = boto3.client('s3')
    rc = config["control_plane"]
    password = rc["password"]
    port= rc["port"]
    prefix = rc["control_plane_prefix"].strip("/")
    bucket = config["s3"]["bucket"]
    keys= list_all_keys(prefix=prefix, bucket=bucket)
    if (idx >= len(keys)):
        click.echo("idx must be less that number of total control planes")
        return
    key = keys[idx]
    info = json.loads(client.get_object(Key=key, Bucket=config["s3"]["bucket"])["Body"].read())
    host = info["public_ip"]
    redis_client = redis.StrictRedis(host=host, port=port, password=password)
    while (True):
        try:
            redis_client.ping()
            click.echo("successful ping!")
            break
        except redis.exceptions.ConnectionError as e:
            click.echo("ping failed.")
            pass





control_plane.add_command(launch)
control_plane.add_command(list)
control_plane.add_command(ping)
control_plane.add_command(info)
control_plane.add_command(terminate)

@click.command()
@click.pass_context
def test(ctx):
    """
    test numpywren
    """
    click.echo("Test")

@click.command()
@click.pass_context
def setup(ctx):
    """
    setup numpywren
    """
    ctx.invoke(interactive_setup)


def test_pywren():
    pwex = pywren.default_executor()
    def hello_world(_):
        return "Hello world"
    fut = pwex.call_async(hello_world, None)
    res = fut.result()

def create_role(config, role_name):
    """
    Creates the IAM profile used by numpywren.
    """

    iam = boto3.resource('iam')
    iamclient = boto3.client('iam')
    pywren_config = wrenconfig.default()
    roles = [x for x in iamclient.list_roles()["Roles"] if x["RoleName"] == role_name]
    if (len(roles) == 0):
        json_policy= json.dumps(npw.config.basic_role)
        iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json_policy)
    AWS_ACCOUNT_ID = pywren_config['account']['aws_account_id']
    AWS_REGION = pywren_config['account']['aws_region']
    more_json_policy = json.dumps(npw.config.basic_permissions)
    more_json_policy = more_json_policy.replace("AWS_ACCOUNT_ID", str(AWS_ACCOUNT_ID))
    more_json_policy = more_json_policy.replace("AWS_REGION", AWS_REGION)

    iam.RolePolicy(role_name, '{}-more-permissions'.format(role_name)).put(
    PolicyDocument=more_json_policy)

def create_instance_profile(config, instance_profile_name):
    instance_profile_name = config['iam']['instance_profile_name']
    role_name = config['iam']['role_name']
    iamclient = boto3.client('iam')
    profiles = [x for x in iamclient.list_instance_profiles()['InstanceProfiles'] if x['InstanceProfileName'] == instance_profile_name]
    if (len(profiles) == 0):
        iam = boto3.resource('iam')
        iam.create_instance_profile(InstanceProfileName=instance_profile_name)
        instance_profile = iam.InstanceProfile(instance_profile_name)
        instance_profile.add_role(RoleName=role_name)


@click.command()
@click.pass_context
def interactive_setup(ctx):

    '''
    Take the following setup
    1) First check if pywren works by running pywren test
    2) Create config file
    3) Create auxillary ``cron lambda"
       a) Check for ``unused redis"
    '''


    def ds(key):
        """
        Debug suffix for defaults. For automated testing,
        automatically adds a suffix to each default
        """
        return "{}{}".format(key, suffix)

    ok = click.confirm(NUMPYWREN_SETUP, default=True)
    if (not ok):
        return
    click.echo("Testing pywren is correctly installed...")
    try:
        test_pywren()
    except Exception as e:
        click.echo("Looks like there is something wrong with your pywren setup. Please make sure the command\
                    pywren test_function returns sucessfully")
        raise
    pywren_config = wrenconfig.default()
    pywren_bucket = pywren_config["s3"]["bucket"]
    # if config file exists, ask before overwriting
    config_filename = click_validate_prompt(
        "Location for config file: ",
        default=npw.config.get_default_home_filename())

    overwrite = check_overwrite_function(config_filename)
    config_filename = os.path.expanduser(config_filename)

    s3_bucket = click_validate_prompt(
        "numpywren requires an s3 bucket to store all data. " + \
            "What s3 bucket would you like to use?",
        default=pywren_bucket,
        validate_func=check_valid_bucket_name)
    create_bucket = False
    if not check_bucket_exists(s3_bucket):
        create_bucket = click.confirm(
            "Bucket does not currently exist, would you like to create it?", default=True)

    click.echo("numpywren prefixes every object it puts in S3 with a particular prefix.")
    prefix = click_validate_prompt(
        "numpywren s3 prefix: ",
        default=npw.config.AWS_S3_PREFIX_DEFAULT)
    if (overwrite):
        default_yaml = yaml.safe_load(open(os.path.join(SOURCE_DIR, "../default_config.yaml")))
    else:
        default_yaml = yaml.safe_load(open(config_filename))

    default_yaml["s3"]["bucket"] = s3_bucket
    default_yaml["s3"]["prefix"] = prefix
    default_yaml["iam"]["role_name"] = npw.config.AWS_ROLE_DEFAULT
    default_yaml["iam"]["instance_profile_name"] = npw.config.AWS_INSTANCE_PROFILE_DEFAULT
    try:
        ec2_client = boto3.client('ec2')
        response = ec2_client.describe_key_pairs()
        key_pairs = [x['KeyName'] for x in response["KeyPairs"]]
        key_pair = key_pairs[0]
    except:
        raise
        click.echo("Error in acquiring ec2 key pair, perhaps you don't have any setup?")
        return

    default_yaml["control_plane"]["ec2_ssh_key"] = key_pair
    config_advanced = click.confirm(
        "Would you like to configure advanced numpywren properties?", default=False)
    if (config_advanced):
        lifespan = int(click_validate_prompt("How many days would you like numpywren to temporarily store data on S3 (default is 1 day, which translates to roughly $0.72 per TB)", default=default_yaml["s3"]["lifespan"], validate_func=check_valid_lifespan))
        default_yaml["s3"]["lifespan"] = lifespan

        runtime_bucket = click_validate_prompt("Which bucket would you like pywren to load the python runtime from", default=default_yaml["runtime"]["bucket"], validate_func=check_valid_bucket_name)
        runtime_key = click_validate_prompt("What is the runtime key in above bucket", default=default_yaml["runtime"]["s3_key"])
        default_yaml["runtime"]["bucket"] = runtime_bucket
        default_yaml["runtime"]["s3_key"] = runtime_key
        role_name = click_validate_prompt("What would you like to name the numpywren iam role which will allow numpywren executors to access your AWS resources", default=default_yaml["iam"]["role_name"])
        default_yaml["iam"]["role_name"] = role_name
        instance_profile_name= click_validate_prompt("What would you like to name the numpywren iam instance profile which will allow numpywren executors to access your AWS resources", default=default_yaml["iam"]["instance_profile_name"])
        default_yaml["iam"]["instance_profile_name"] = instance_profile_name
        ec2_ssh_key = click_validate_prompt("Pick a valid ec2 ssh key pair", default=default_yaml["control_plane"]["ec2_ssh_key"])
        default_yaml["control_plane"]["ec2_ssh_key"] = ec2_ssh_key
    else:
        role_name = default_yaml["iam"]["role_name"]
        instance_profile_name = default_yaml["iam"]["instance_profile_name"]

    create_role(default_yaml, role_name)
    create_instance_profile(default_yaml, instance_profile_name)
    lifespan = default_yaml["s3"]["lifespan"]
    s3Client = boto3.client('s3')
    s3Client.put_bucket_lifecycle_configuration(
        Bucket=s3_bucket,
        LifecycleConfiguration={
            'Rules': [
                {
                    'Status': 'Enabled',
                    'Expiration':{'Days': lifespan},
                    'Filter': { 'Prefix': prefix }
                },
            ]
        })
    open(config_filename, "w+").write(yaml.dump(default_yaml, default_flow_style=False))

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
    bucket = config['s3']['bucket']
    bucket = 'numpywrenpublic'
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
    print("bucket", bucket)
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




cli.add_command(control_plane)
cli.add_command(test)
cli.add_command(setup)
cli.add_command(lapack)


def main():
    return cli() # pylint: disable=no-value-for-parameter

