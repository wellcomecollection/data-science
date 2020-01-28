#!/usr/bin/env python
import json
import math
import os
import stat

import boto3
import pandas as pd
import paramiko
from halo import Halo


class Controller():
    def __init__(self, set_up_ssh=False):
        self.hal_dir = os.path.expanduser('~/.hal/')
        self.config = json.load(
            open(os.path.join(self.hal_dir, 'config.json'))
        )
        self.credentials = self.get_credentials()

        self.key_path = os.path.expanduser(self.config['key_path'])
        self.key_name = os.path.basename(self.key_path)

        boto3.setup_default_session(region_name='eu-west-1')

        self.client = boto3.client(
            'ec2',
            aws_access_key_id=self.credentials['AccessKeyId'],
            aws_secret_access_key=self.credentials['SecretAccessKey'],
            aws_session_token=self.credentials['SessionToken']
        )
        self.resource = boto3.resource(
            'ec2',
            aws_access_key_id=self.credentials['AccessKeyId'],
            aws_secret_access_key=self.credentials['SecretAccessKey'],
            aws_session_token=self.credentials['SessionToken']
        )

        # if we already have instances running, assign them to the controller
        self.volume = self.get_volume(self.config['VolumeId'])
        self.instance, self.ssh = self.get_existing_instance(set_up_ssh)

    def get_existing_instance(self, set_up_ssh):
        '''
        Returns any data science instances which are already running
        '''
        instances = self.resource.instances.filter(Filters=[
            {'Name': 'tag:datascience', 'Values': ['true']},
            {'Name': 'tag:user_name', 'Values': [self.config['user_name']]},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ])

        instance, ssh = None, None
        if len(list(instances)) == 1:
            instance = list(instances)[0]
            if set_up_ssh:
                ssh = self.set_up_ssh(instance)

        return instance, ssh

    def get_credentials(self):
        sts = boto3.client('sts')
        assumed_role_object = sts.assume_role(
            RoleArn='arn:aws:iam::964279923020:role/data-developer',
            RoleSessionName='AssumeRoleSession1'
        )
        return assumed_role_object['Credentials']

    def get_spot_price(self, instance_type, multiplier=1.1):
        '''
        Returns a bid base on the most recent spot price history for 
        the specified instance_type
        '''
        response = self.client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            MaxResults=1,
            ProductDescriptions=['Linux/UNIX (Amazon VPC)'],
            AvailabilityZone='eu-west-1b'
        )

        base_price = float(response['SpotPriceHistory'][0]['SpotPrice'])
        # multiply base price by factor (default 1.1x), rounded up to 2dp
        bid_price = math.ceil(base_price * multiplier * 100.0) / 100.0
        return bid_price

    def construct_spot_launch_spec(self, instance_type):
        '''
        Builds a launch specification for a spot instance request with a 
        specified instance type and a small EBS volume.
        '''
        launch_spec = {
            'ImageId': self.config['ImageId'],
            'InstanceType': instance_type,
            'KeyName': self.key_name,
            'IamInstanceProfile': {
                'Arn': self.config['instance_profile_arn']
            },
            'NetworkInterfaces': [{
                'DeviceIndex': 0,
                'SubnetId': self.config['SubnetId'],
                'Groups': self.config['Groups'],
                'AssociatePublicIpAddress': True
            }],
        }
        return launch_spec

    @Halo(text='\nCreating instance')
    def create_instance(self, instance_type, instance_name, spot_price=None):
        '''
        Makes a spot request for an specified instance with a dynamically 
        calculated spot price based on the most recent available data
        '''
        launch_spec = self.construct_spot_launch_spec(instance_type)
        if not spot_price:
            spot_price = self.get_spot_price(instance_type)

        request_id = self.client.request_spot_instances(
            LaunchSpecification=launch_spec,
            InstanceInterruptionBehavior='terminate',
            SpotPrice=str(spot_price),
            Type='one-time'
        )['SpotInstanceRequests'][0]['SpotInstanceRequestId']

        self.client.get_waiter('spot_instance_request_fulfilled').wait(
            SpotInstanceRequestIds=[request_id],
            WaiterConfig={'Delay': 3, 'MaxAttempts': 20}
        )

        instance_id = self.client.describe_spot_instance_requests(
            SpotInstanceRequestIds=[request_id]
        )['SpotInstanceRequests'][0]['InstanceId']

        self.client.get_waiter('instance_running').wait(
            InstanceIds=[instance_id],
            WaiterConfig={'Delay': 3, 'MaxAttempts': 20}
        )

        self.name_instance(instance_name, instance_id)
        self.instance = self.get_instance(instance_id)
        return self.instance.id, spot_price

    def describe_instances(self):
        '''
        Displays the status of any instances which haven't been terminated
        '''
        instances = self.resource.instances.filter(Filters=[{
            'Name': 'instance-state-name',
            'Values': ['pending', 'running', 'stopping', 'stopped']
        }])

        data = [{
            'id': i.instance_id,
            'instance_name': [t['Value'] for t in i.tags if t['Key'] == 'Name'][0],
            'user_name':[t['Value'] for t in i.tags if t['Key'] == 'user_name'][0],
            'state': i.state['Name']
        } for i in list(instances)]

        if len(data) == 0:
            return 'There are currently no running instances!'
        else:
            return pd.DataFrame(data)

    def get_instance(self, instance_id):
        '''
        Returns a specified EC2 instance object which can be subsequently
        manipulated
        '''
        response = self.resource.instances.filter(Filters=[{
            'Name': 'instance-id',
            'Values': [instance_id]
        }])
        return next(iter(response))

    @Halo(text='\nStarting instance')
    def start_instance(self, instance_id=None):
        instance_id = instance_id or self.instance.id
        self.get_instance(instance_id).start()

    @Halo(text='\nStopping instance')
    def stop_instance(self, instance_id=None):
        instance_id = instance_id or self.instance.id
        self.get_instance(instance_id).stop()

    @Halo(text='\nTerminating instance')
    def terminate_instance(self, instance_id=None):
        instance_id = instance_id or self.instance.id
        self.get_instance(instance_id).terminate()

    def name_instance(self, instance_name, instance_id=None):
        instance_id = instance_id or self.instance.id
        self.client.create_tags(
            Resources=[instance_id],
            Tags=[
                {'Key': 'Name', 'Value': instance_name},
                {'Key': 'datascience', 'Value': 'true'},
                {'Key': 'user_name', 'Value': self.config['user_name']}
            ]
        )

    @Halo(text='\nCreating ssh connection to instance')
    def set_up_ssh(self, instance=None):
        '''
        Return a paramiko ssh connection objected connected to instance
        '''
        instance = instance or self.instance
        key = paramiko.RSAKey.from_private_key_file(
            self.key_path, password=self.config['password']
        )
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=instance.public_ip_address,
            username='ec2-user',
            pkey=key,
            password=self.config['password']
        )
        client.raise_stderr = True
        self.ssh = client
        return client

    def execute_command(self, command):
        '''
        Run a command on the remote instance via ssh
        '''
        _, stdout, stderr = self.ssh.exec_command(command)
        stdout_str = stdout.read().decode()
        stderr_str = stderr.read().decode()
        if stdout.channel.recv_exit_status() != 0:
            raise Exception(stderr_str)
        if self.ssh.raise_stderr:
            if stderr_str:
                raise Exception(stderr_str)
            return stdout_str
        return stdout_str, stderr_str

    def get_volume(self, volume_id):
        '''
        Returns a specified EBS volume object which can be subsequently
        manipulated
        '''
        response = self.resource.volumes.filter(Filters=[{
            'Name': 'volume-id',
            'Values': [volume_id]
        }])
        return next(iter(response))

    @Halo(text='\nAttaching storage volume')
    def attach_volume(self):
        self.volume.attach_to_instance(
            Device='/dev/xvdf',
            InstanceId=self.instance.id
        )
        self.client.get_waiter('volume_in_use').wait(
            VolumeIds=[self.volume.id],
            WaiterConfig={'Delay': 3, 'MaxAttempts': 20}
        )

    @Halo(text='\nMounting storage volume')
    def mount_volume(self):
        commands = [
            'sudo mkdir /storage',
            'sudo mount /dev/xvdf /storage'
        ]
        for command in commands:
            self.execute_command(command)

    @Halo(text='\nDetaching storage volume')
    def detach_volume(self):
        self.execute_command('sudo umount -d /dev/xvdf')
        self.volume.detach_from_instance()
        self.client.get_waiter('volume_available').wait(
            VolumeIds=[self.volume.id],
            WaiterConfig={'Delay': 3, 'MaxAttempts': 20}
        )

    def start_jupyterlab(self):
        os.system(
            f'ssh ec2-user@{self.instance.public_ip_address} '
            f'-i {self.key_path} -NfL 8888:localhost:8888'
        )

        self.execute_command(
            'jupyter lab --no-browser --port=8888 --notebook-dir="/storage" &'
        )
        self.execute_command('')

    def open_connection_to_instance(self):
        '''
        Replaces the current python process with an ssh process connected 
        to the specified instance.
        '''
        os.system(
            f'ssh ec2-user@{self.instance.public_ip_address} '
            f'-i {self.key_path} -L 8888:localhost:8888'
        )

    def send_file(self, local_path, remote_path):
        '''
        Sends a file or directory from the local machine to the remote instance
        '''
        sftp = self.ssh.open_sftp()
        if os.path.isfile(local_path):  # local_path is a file
            sftp.put(local_path, remote_path)

        elif os.path.isdir(local_path):  # local_path is a dir
            try:
                sftp.stat(remote_path)
            except FileNotFoundError:  # if remote dir does not exist
                sftp.mkdir(remote_path)  # create it

            for file_name in os.listdir(local_path):
                sftp.put(
                    localpath=os.path.join(local_path, file_name),
                    remotepath=os.path.join(remote_path, file_name)
                )

        else:
            raise ValueError(
                f'"{local_path} is neither a file or a directory!"'
            )

    def get_file(self, local_path, remote_path):
        '''
        Gets a file or directory from the remote instance and saves a local copy
        '''
        # get remote path metadata
        sftp = self.ssh.open_sftp()
        remote_path_attr = sftp.lstat(remote_path)

        if stat.S_ISREG(remote_path_attr.st_mode):  # remote_path is a file
            base_path = os.path.dirname(local_path)
            if not os.path.exists(base_path):
                os.mkdir(base_path)

            with open(local_path, 'w'):
                sftp.get(remote_path, local_path)

        elif stat.S_ISDIR(remote_path_attr.st_mode):  # remote_path is a dir
            if not os.path.exists(local_path):
                os.mkdir(local_path)

            for file_name in sftp.listdir(remote_path):
                sftp.get(
                    localpath=os.path.join(local_path, file_name),
                    remotepath=os.path.join(remote_path, file_name)
                )

        else:
            raise ValueError(
                f'"{remote_path} is neither a file or a directory!"'
            )

    def fix_dns(self):
        self.execute_command('sudo chmod a+rwx /etc/resolv.conf')
        self.execute_command(
            'printf "nameserver 169.254.169.253" >> /etc/resolv.conf'
        )
