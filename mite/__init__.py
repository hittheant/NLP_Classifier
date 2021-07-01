import platform
os_sys = platform.system().lower()

# aesthetic imports
import matplotlib

import numpy
numpy.set_printoptions( precision = 3, suppress = True )

# timing functions
if os_sys == 'linux':
    import ctypes
    libc = ctypes.CDLL('libc.so.6')

    class Timespec( ctypes.Structure ):
        _fields_ = [ ('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long) ]
    libc.nanosleep.argtypes = [ctypes.POINTER(Timespec), ctypes.POINTER(Timespec)]
    nanosleep_req = Timespec()
    nanosleep_rem = Timespec()

    def ns_sleep( ns ):
        nanosleep_req.tv_sec = int( ns / 1e9 )
        nanosleep_req.tv_nsec = int( ns % 1e9 )
        libc.nanosleep( nanosleep_req, nanosleep_rem )

    def us_sleep( us ):
        libc.usleep( int( us ) )
else:
    import time
    def ns_sleep( ns ): time.sleep( ns * 1e-9 )
    def us_sleep( us ): time.sleep( us * 1e-6 )