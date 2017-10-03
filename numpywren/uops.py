import boto3
import itertools
import numpy as np
from .matrix import BigSymmetricMatrix, BigMatrix
from .matrix_utils import load_mmap, chunk, generate_key_name
import concurrent.futures as fs
import math
import os
import pywren
from pywren.executor import Executor
import time

# this one is hard
def reshard(pwex, X, new_shard_sizes, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# These have some dependencies
def argmin(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def argmax(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def min(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def max(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def norm(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# these have no dependencies
def abs(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def neg(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sum(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def square(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def power(pwex, X, k, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sqrt(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def prod(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sin(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def cos(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def tan(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def exp(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sign(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def elemwise_uop_func(pwex, X, f, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError




