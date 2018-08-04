from . import config
from . import matrix
from . import binops
from . import compiler
import os

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

from sys import platform
if platform == "linux" or platform == "linux2":
    TMP_DIR = "/dev/shm/"
elif platform == "darwin":
    TMP_DIR = "/tmp/"
elif platform == "win32":
    TMP_DIR = "/tmp/"
else:
    raise Exception("Unsupported platform")
