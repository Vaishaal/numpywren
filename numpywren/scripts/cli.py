import numpywren as npw
import argparse
import os
import click
import pywren
from pywren import wrenconfig
import re
import yaml
import boto3


SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_username():
    return pwd.getpwuid(os.getuid())[0]

def check_overwrite_function(filename):
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        return click.confirm("{} already exists, would you like to overwrite?".format(filename))
    return True

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

@click.command()
@click.pass_context
def control_plane(ctx):
    """
    control_plane subcommand
    """
    click.echo("control_plane")

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


    click.echo("This is the numpywren interactive setup script")
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
        default=npw.config.get_default_home_filename(),
        validate_func=check_overwrite_function)
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

    default_yaml = yaml.safe_load(open(os.path.join(SOURCE_DIR, "../default_config.yaml")))
    print(default_yaml)
    default_yaml["s3"]["bucket"] = s3_bucket
    default_yaml["s3"]["prefix"] = prefix
    open(config_filename, "w+").write(yaml.dump(default_yaml, default_flow_style=False))


def create_config(bucket_name, bucket_prefix):
    """
    Create a config file initialized with the defaults, and
    put it in your ~/.numpywren_config

    """
    filename = ctx.obj['config_filename']
    # copy default config file

    # FIXME check if it exists
    default_yaml = open(os.path.join(SOURCE_DIR, "../default_config.yaml")).read()

    client = boto3.client("sts")
    account_id = client.get_caller_identity()["Account"]

    # perform substitutions -- get your AWS account ID and auto-populate

    default_yaml = default_yaml.replace('AWS_ACCOUNT_ID', account_id)
    default_yaml = default_yaml.replace('AWS_REGION', aws_region)
    default_yaml = default_yaml.replace('pywren_exec_role', lambda_role)
    default_yaml = default_yaml.replace('pywren1', function_name)
    default_yaml = default_yaml.replace('BUCKET_NAME', bucket_name)
    default_yaml = default_yaml.replace('pywren.jobs', bucket_prefix)
    default_yaml = default_yaml.replace('pywren-queue', sqs_queue)
    default_yaml = default_yaml.replace('pywren-standalone', standalone_name)
    if pythonver not in pywren.wrenconfig.default_runtime:
        print('No matching runtime package for python version ', pythonver)
        print('Python 2.7 runtime will be used for remote.')
        pythonver = '2.7'

    runtime_bucket = 'pywren-public-{}'.format(aws_region)
    default_yaml = default_yaml.replace("RUNTIME_BUCKET",
                                        runtime_bucket)
    k = pywren.wrenconfig.default_runtime[pythonver]

    default_yaml = default_yaml.replace("RUNTIME_KEY", k)

    # print out message about the stuff you need to do
    if os.path.exists(filename) and not force:
        raise ValueError("{} already exists; not overwriting (did you need --force?)".format(
            filename))

    open(filename, 'w').write(default_yaml)
    click.echo("new default file created in {}".format(filename))
    click.echo("lambda role is {}".format(lambda_role))



cli.add_command(control_plane)
cli.add_command(test)
cli.add_command(setup)


def main():
    return cli() # pylint: disable=no-value-for-parameter

