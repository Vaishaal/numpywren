import redis
import pywren.ec2standalone as ec2standalone
import pywren.wrenconfig as wc
import numpywren as npw
import os
import base64
import boto3
import time


def sd(filename):
    """
    get the file in the redis dir
    """
    return os.path.join(npw.SOURCE_DIR, 'redis_files', filename)

def b64s(string):
    """
    Base-64 encode a string and return a string
    """
    return base64.b64encode(string.encode('utf-8')).decode('ascii')


def create_security_group(name="numpywren.group"):
    #TODO figure out how to security
    client = boto3.client('ec2')
    groups = [x for x in client.describe_security_groups()['SecurityGroups'] if x['GroupName'] == name]

    if (len(groups) == 0):
         group = client.create_security_group(GroupName=name, Description=name)
         group_id = group['GroupId']
         resp = client.authorize_security_group_ingress(GroupId=group_id,IpProtocol="tcp",CidrIp="0.0.0.0/0",FromPort=0,ToPort=9999)
    else:
        group = groups[0]
    return group['GroupId']



def _create_instances(num_instances,
                      region,
                      spot_price,
                      ami,
                      key_name,
                      instance_type,
                      block_device_mappings,
                      security_group_ids,
                      ebs_optimized,
                      instance_profile,
                      availability_zone,
                      user_data):

    ''' Function graciously borrowed from Flintrock ec2 wrapper
        https://raw.githubusercontent.com/nchammas/flintrock/00cce5fe9d9f741f5999fddf2c7931d2cb1bdbe8/flintrock/ec2.py
    '''

    ec2 = boto3.resource(service_name='ec2', region_name=region)
    spot_requests = []
    try:
        if spot_price:
            print("Requesting {c} spot instances at a max price of ${p}...".format(
                c=num_instances, p=spot_price))
            client = ec2.meta.client


            LaunchSpecification = {
                'ImageId': ami,
                'KeyName': key_name,
                'InstanceType': instance_type,
                'SecurityGroupIds': security_group_ids,
                'EbsOptimized': ebs_optimized,
                #'IamInstanceProfile' : instance_profile,
                'UserData' : b64s(user_data)}
            if availability_zone is not None:
                LaunchSpecification['Placement'] = {"AvailabilityZone":availability_zone}
            if block_device_mappings is not None:
                LaunchSpecification['BlockDeviceMappings'] = block_device_mappings

            spot_requests = client.request_spot_instances(
                SpotPrice=str(spot_price),
                InstanceCount=num_instances,
                LaunchSpecification=LaunchSpecification)['SpotInstanceRequests']

            request_ids = [r['SpotInstanceRequestId'] for r in spot_requests]
            pending_request_ids = request_ids

            while pending_request_ids:
                print("{grant} of {req} instances granted. Waiting...".format(
                    grant=num_instances - len(pending_request_ids),
                    req=num_instances))
                time.sleep(30)
                spot_requests = client.describe_spot_instance_requests(
                    SpotInstanceRequestIds=request_ids)['SpotInstanceRequests']

                failed_requests = [r for r in spot_requests if r['State'] == 'failed']
                if failed_requests:
                    failure_reasons = {r['Status']['Code'] for r in failed_requests}
                    raise Exception(
                        "The spot request failed for the following reason{s}: {reasons}"
                        .format(
                            s='' if len(failure_reasons) == 1 else 's',
                            reasons=', '.join(failure_reasons)))

                pending_request_ids = [
                    r['SpotInstanceRequestId'] for r in spot_requests
                    if r['State'] == 'open']

            print("All {c} instances granted.".format(c=num_instances))

            cluster_instances = list(
                ec2.instances.filter(
                    Filters=[
                        {'Name': 'instance-id', 'Values': [r['InstanceId'] for r in spot_requests]}
                    ]))
        else:
            # Move this to flintrock.py?
            print("Launching {c} instance{s}...".format(
                c=num_instances,
                s='' if num_instances == 1 else 's'))

            # TODO: If an exception is raised in here, some instances may be
            #       left stranded.

            LaunchSpecification = {
                "MinCount" : num_instances,
                "MaxCount" : num_instances,
                "ImageId" : ami,
                "KeyName" : key_name,
                "InstanceType" : instance_type,
                "SecurityGroupIds" : security_group_ids,
                "EbsOptimized" : ebs_optimized,
                #"IamInstanceProfile" :   instance_profile,
                "InstanceInitiatedShutdownBehavior" :  'terminate',
                "UserData" :  user_data}
            if block_device_mappings is not None:
                LaunchSpecification['BlockDeviceMappings'] = block_device_mappings

            cluster_instances = ec2.create_instances(**LaunchSpecification)

        time.sleep(10)  # AWS metadata eventual consistency tax.
        return cluster_instances
    except (Exception, KeyboardInterrupt) as e:
        if not isinstance(e, KeyboardInterrupt):
            print(e)
        if spot_requests:
            request_ids = [r['SpotInstanceRequestId'] for r in spot_requests]
            if any([r['State'] != 'active' for r in spot_requests]):
                print("Canceling spot instance requests...")
                client.cancel_spot_instance_requests(
                    SpotInstanceRequestIds=request_ids)
            # Make sure we have the latest information on any launched spot instances.
            spot_requests = client.describe_spot_instance_requests(
                SpotInstanceRequestIds=request_ids)['SpotInstanceRequests']
            instance_ids = [
                r['InstanceId'] for r in spot_requests
                if 'InstanceId' in r]
            if instance_ids:
                cluster_instances = list(
                    ec2.instances.filter(
                        Filters=[
                            {'Name': 'instance-id', 'Values': instance_ids}
                        ]))
        raise Exception("Launch failure")


def launch_and_provision_redis(port=6379, password="potato", spot_price=0.0):
    config = npw.config.default()
    pywren_config = wc.default()
    rc = config["redis"]
    ami = rc["target_ami"]
    instance_type = rc["ec2_instance_type"]
    key_name = rc["ec2_ssh_key"]
    aws_region = pywren_config['account']['aws_region']
    availability_zone = rc.get("availability_zone", None)
    redis_conf = open(sd("redis.conf")).read()
    template_file = sd("redis.cloudinit.template")
    user_data = open(template_file, 'r').read()
    cloud_agent_conf = open(sd("cloudwatch-agent.config"),
                            'r').read()
    cloud_agent_conf_64 = b64s(cloud_agent_conf)
    redis_conf_b64 = b64s(redis_conf.format(port=port, password=password))
    redis_init_b64 = b64s(open(sd("redis_init_script")).read().format(port=port))
    user_data = user_data.format(redis_init=redis_init_b64, cloud_agent_conf=cloud_agent_conf_64, redis_conf=redis_conf_b64, aws_region=aws_region)
    instance_profile_dict = None
    group_id = create_security_group()
    instances = _create_instances(1, aws_region, spot_price, ami=ami, instance_type=instance_type, block_device_mappings=None, security_group_ids=[group_id], ebs_optimized=True, availability_zone=None, instance_profile=instance_profile_dict, user_data=user_data, key_name=key_name)
    inst = instances[0]
    inst.reload()
    inst.create_tags(
        Resources=[
            inst.instance_id
        ],
        Tags=[
            {
                'Key': 'Name',
                'Value': 'numpywren.control_plane'
            },
        ]
    )








