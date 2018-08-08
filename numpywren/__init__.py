from . import config
from . import matrix
from . import binops
from . import compiler
import os
import logging

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
print("POOOP")
logger = logging.getLogger('numpywren')
ch = logging.StreamHandler()
conf = config.default()
log_level = conf['logging']['level']
if (log_level == 'DEBUG'):
    log_level = logging.DEBUG
elif (log_level == 'ERROR'):
    log_level = logging.ERROR
elif (log_level == 'WARNING'):
    log_level = logging.WARNING
elif (log_level == 'INFO'):
    log_level = logging.INFO
else:
    raise Exception("Unsupported loglevel")

logger.setLevel(log_level)
logging.getLogger('boto').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
print("log level ", log_level)
logging.basicConfig(level=logging.CRITICAL)





from sys import platform
if platform == "linux" or platform == "linux2":
    TMP_DIR = "/dev/shm/"
elif platform == "darwin":
    TMP_DIR = "/tmp/"
elif platform == "win32":
    TMP_DIR = "/tmp/"
else:
    raise Exception("Unsupported platform")
