
from . import config
from . import matrix
from . import binops
from . import compiler
import os
import logging

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('numpywren')
ch = logging.StreamHandler()
try:
    conf = config.default()
    log_level = conf['logging']['level']
    if (log_level == ''):
        log_level = logging.DEBUG
    elif (log_level == 'ERROR'):
        log_level = logging.ERROR
    elif (log_level == 'WARNING'):
        log_level = logging.WARNING
    elif (log_level == 'INFO'):
        log_level = logging.INFO
    else:
        raise Exception("Unsupported loglevel")
except:
    log_level = logging.DEBUG
    pass
logger.setLevel(logging.DEBUG)
logging.getLogger('boto').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('multyvac.dependency-analyzer').setLevel(logging.CRITICAL)
f_stream = open("/tmp/numpywren.log", "w+")
logging.basicConfig(stream=f_stream, level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)




from sys import platform
if platform == "linux" or platform == "linux2":
    TMP_DIR = "/dev/shm/"
elif platform == "darwin":
    TMP_DIR = "/tmp/"
elif platform == "win32":
    TMP_DIR = "/tmp/"
else:
    raise Exception("Unsupported platform")
