import os
import ctypes
import platform

libname = '/lib/librhd2k.so'
if platform.system() == 'Windows' or 'CYGWIN' in platform.system():
    libname =  '/lib/librhd2k.dll'
rhd2klib = ctypes.cdll.LoadLibrary(os.path.dirname( os.path.dirname( __file__ ) ) + libname)

from .rhd2000evalboard import Rhd2000EvalBoard
from .rhd2000datablock import Rhd2000DataBlock
from .rhd2000registers import Rhd2000Registers
from .dataqueue import DataQueue
from .vector import VectorInt
from .ofstream import Ofstream
from .constants import *
