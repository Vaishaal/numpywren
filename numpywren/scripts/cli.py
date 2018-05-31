import numpywren as npw
from numpywren import redis_utils
import argparse
import os
import click
import pywren
from pywren import wrenconfig
import re
import yaml
import boto3
import json
import redis
import time

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
    # Validates bucketname
    # Based on http://info.easydynamics.com/blog/aws-s3-bucket-name-validation-regex
    # https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html
    bucket_regex = re.compile(r"""^([a-z]|(\d(?!\d{0,2}\.\d{1,3}\.\d{1,3}\.\d{1,3})))
                                   ([a-z\d]|(\.(?!(\.|-)))|(-(?!\.))){1,61}[a-z\d\.]$""", re.X)
    if re.match(bucket_regex, bucket_name):
        return True
    return False



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
    ip = redis_utils.launch_and_provision_redis()
    click.echo("Waiting for redis")
    config = npw.config.default()
    rc = config["redis"]
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





control_plane.add_command(launch)

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




cli.add_command(control_plane)
cli.add_command(test)
cli.add_command(setup)


def main():
    return cli() # pylint: disable=no-value-for-parameter

