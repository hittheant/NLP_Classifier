# NOTE: This interface requires that the Myo Armband be previously set up using MyoConnect on
#       a Windows or Mac PC.'

import re
import sys

import time
import struct
import ctypes
import queue

import serial
from serial.tools.list_ports import comports

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from .. import ns_sleep
from ..utils import Quaternion as quat
from . import AbstractBaseInput

class MyoArmband(AbstractBaseInput):
    """ Python interface for a Myo Armband from Thalmic Labs  """
    CONNECTION_TIMEOUT = 10.0

    MYOHW_ORIENTATION_SCALE = 16384.0
    MYOHW_ACCELEROMETER_SCALE = 2048.0
    MYOHW_GYROSCOPE_SCALE = 16.0

    ''' Static methods for data manipulation '''
    @staticmethod
    def pack( fmt, *args ):
        """
        Packs data into byte structure using the supplied format string
        
        Parameters
        ----------
        fmt : str
            Designates the format of the given data
        args : int, float, str, bool
            The data to pack

        Returns
        -------
        bytes
            The input data interpreted in binary
        """
        return struct.pack( '<' + fmt, *args )

    @staticmethod 
    def unpack( fmt, *args ):
        """
        Unpacks byte structure as data according to supplied format string

        Parameters
        ----------
        fmt : str
            Designates the format of the given data
        args: bytes
            The data to unpack

        Returns
        -------
        int, float, str, bool
            The byte data interpreted as python base-types
        """
        return struct.unpack( '<' + fmt, *args )

    @staticmethod
    def multichr( ords ):
        """
        Converts input into appropriate unicode representation

        Parameters
        ----------
        ords : iterable of ints
            The ASCII codes to convert into characters

        Returns
        -------
        str (Python 2), bytes (Python 3)
            A string representation of the input data
        """
        if sys.version_info[0] >= 3: return bytes( ords )
        else: return ''.join( map( chr, ords ) )

    @staticmethod
    def multiord( b ):
        """
        Converts input into appropriate ASCII digits

        Parameters
        ----------
        b : str
            Unicode characters to translate
        
        Returns
        -------
        list
            ASCII digits of each input character
        """
        if sys.version_info[0] >= 3: return list( b )
        else: return list( map( ord, b ) )

    class Packet( object ):
        """ Private class to wrap Myo Armband bluetooth packet structure """
        def __init__( self, ords ):
            """
            Constructor

            Parameters
            ----------
            ords : iterable of ints
                The raw data in the Myo Armband bluetooth packet

            Returns
            -------
            obj
                A Packet object
            """
            self.typ = ords[0]
            self.cls = ords[2]
            self.cmd = ords[3]
            self.payload = MyoArmband.multichr( ords[4:] )

        def __repr__(self):
            """
            Defines how packet content should be structured when represented as a string

            Returns
            -------
            str
                Packet data structured in a readable manner
            """
            return 'Packet(%02X, %02X, %02X, [%s])' % \
                (self.typ, self.cls, self.cmd,
                ' '.join('%02X' % b for b in MyoArmband.multiord(self.payload)))
    
    class MyoBT( object ):
        """ Private class that implements the non-Myo-specific details of the Bluetooth protocol """
        def __init__(self, tty ):
            """
            Constructor

            Parameters
            ----------
            tty : str
                The serial port that the bluetooth data is received on
            
            Returns
            -------
            obj
                A bluetooth interface object
            """
            self.ser = serial.Serial( port = tty, baudrate = 9600, dsrdtr = 1 )
            self.buf = []
            self.handlers = []

        # internal data-handling methods
        def recv_packet( self, timeout = None ):
            """
            Receives and processes incoming bluetooth packets

            Parameters
            ----------
            timeout : float or None
                Time (seconds) to wait until giving up on receiving a packet

            Returns
            -------
            Packet or None
                Myo bluetooth packet (if received)
            """
            t0 = time.time()
            self.ser.timeout = None
            while timeout is None or time.time() < t0 + timeout:
                if timeout is not None: self.ser.timeout = max( 0, t0 + timeout - time.time() )
                c = self.ser.read()
                if not c: return None

                ret = self.proc_byte( ord( c ) )
                if ret:
                    if ret.typ == 0x80: self.handle_event( ret )
                    return ret

        def proc_byte( self, c ):
            """
            Processes incoming bytestream (one byte at a time)
            
            This function is a helper function for recv_packet

            Parameters
            ----------
            c : byte
                individual byte to process

            Returns
            -------
            Packet or None
                Myo bluetooth packet (once done processing)
            """
            if not self.buf:
                if c in [ 0x00, 0x80, 0x08, 0x88 ]: # [ BLE response packet, BLE event packet, Wifi response packet, Wifi event packet ]
                    self.buf.append( c )
                return None
            elif len( self.buf ) == 1:
                self.buf.append( c )
                self.packet_len = 4 + ( self.buf[ 0 ] & 0x07 ) + self.buf[ 1 ]
                return None
            else: self.buf.append( c )

            if self.packet_len and len( self.buf ) == self.packet_len:
                p = MyoArmband.Packet( self.buf )
                self.buf = []
                return p
            return None

        def handle_event( self, p ):
            """
            Applies all registered event handlers to the given packet

            Parameters
            ----------
            p : Packet
                The data packet to handle
            """
            for h in self.handlers: h( p )

        def add_handler( self, h ): 
            """
            Add the given event handler to the list of handlers

            Parameters
            ----------
            h : function
                The function to add to the handler list
            """
            self.handlers.append( h )

        def remove_handler( self, h ):
            """
            Remove the given event handler from the list of handlers (if it exists)

            Parameters
            ----------
            h : function
                The function to remove from the handler list
            """
            try:
                self.handlers.remove( h )
            except ValueError: pass

        def wait_event( self, cls, cmd ):
            """
            Waits for a Packet to be received with the specified cls and cmd fields

            Parameters
            ----------
            cls : byte
                The wanted cls field for the Packet object
            cmd : byte
                The wanted cmd field for the Packet object
            
            Returns
            -------
            Packet
                A bluetooth packet with the correct cls and cmd fields
            """
            res = [ None ]
            def h( p ):
                if ( p.cls, p.cmd ) == ( cls, cmd ): res[0] = p
            self.add_handler( h )
            while res[0] is None: self.recv_packet()
            self.remove_handler( h )
            return res[0]

        # specific BLE commands
        def connect( self, addr ):
            """
            Send a connection request to a Myo Armband

            Parameters
            ----------
            addr : str
                The mac address of the armband we want to connect to

            Returns
            -------
            Packet
                The response packet to our request
            """
            return self.send_command( 6, 3, MyoArmband.pack( '6sBHHHH', MyoArmband.multichr( addr ), 0, 6, 6, 64, 0 ) )

        def get_connections( self ):
            """
            Get a list of all connected Myo Armbands

            Returns
            -------
            Packet
                The response packet to our request
            """
            return self.send_command( 0, 6 )

        def discover( self ):
            """
            Start Myo Armband BLE discovery

            Returns
            -------
            Packet
                The response packet to our request
            """
            return self.send_command( 6, 2, b'\x01' )

        def end_scan( self ):
            """
            Stop scan for available Myo Armbands

            Returns
            -------
            Packet
                The response packet to our request
            """
            return self.send_command( 6, 4 )

        def disconnect( self, h ):
            """
            Disconnect from the specified Myo Armband

            Parameters
            ----------
            h : byte
                The connection handle to the wanted Myo Armband
            
            Returns
            -------
            Packet
                The response packet to our request
            """
            return self.send_command( 3, 0, MyoArmband.pack( 'B', h ) )

        def read_attr( self, con, attr ):
            """
            Reads the specific attribute of the specified Myo Armband

            Parameters
            ----------
            con : byte
                Connection handle of the wanted Myo Armband
            attr : bytes
                Attribute to read

            Returns
            -------
            Packet
                Bluetooth packet containing the queried attribute value
            """
            self.send_command( 4, 4, MyoArmband.pack( 'BH', con, attr ) )
            return self.wait_event( 4, 5 )

        def write_attr( self, con, attr, val ):
            """
            Writes the specific value to the specified attribute of the specified Myo Armband

            Parameters
            ----------
            con : byte
                Connection handle of the wanted Myo Armband
            attr : bytes
                Attribute to write to
            val : byte
                Value to write to the attribute

            Returns
            -------
                Bluetooth packet confirming success (or failure) of attribute write
            """
            self.send_command( 4, 5, MyoArmband.pack( 'BHB', con, attr, len( val ) ) + val )
            return self.wait_event( 4, 1 )

        def send_command( self, cls, cmd, payload = b'', wait_resp = True ):
            """
            Sends a command packet to all connected Myo Armbands

            Parameters
            ----------
            cls : byte
                The class of the command packet
            cmd : byte
                The type of command to be sent
            payload : bytes
                The data payload to be sent
            wait_resp : bool
                Flag to determine if we should wait for a response

            Returns
            -------
            Packet
                Response to our command
            """
            s = MyoArmband.pack( '4B', 0, len( payload ), cls, cmd ) + payload
            self.ser.write( s )

            while True:
                p = self.recv_packet()
                if p.typ == 0: return p
                self.handle_event( p )
    
    """ Myo-specific communication protocol """

    def __init__(self, name = 'MyoArmband', com = '', mac = 'e0:f2:99:e7:60:40', srate = 200.0, raw = True):
        """ 
        Constructor 
        
        Parameters
        ----------
        name : str
            The unique device handle used to refer to the connected hardware
        com : str
            The named communication port for the serial connection (e.g. COM9 on Windows, /dev/ttyACM0 on linux)
        mac : str
            The MAC address of the MyoArmband
        srate : float
            The sampling rate used to poll the device (maximum is 200 Hz)
        raw : bool
            Flag to determine whether to receive raw (True) or mean absolute value (False) data

        Returns
        -------
        obj
            A MyoArmband interface object
        """
        self._name = name
        self._mac = bytes.fromhex( ''.join( mac.split(':') ) )
        self._raw = raw

        # set device variables
        self._channelcount = 12
        self._speriod = 1.0 / srate
        
        # bluetooth connection variables
        self._bt = None
        self._conn = None

        # raw EMG callback variables
        self._emg_time = mp.Value( ctypes.c_double, 0.0 )
        self._emg_value = mp.Array( ctypes.c_byte, 16 )
        
        # synchronization events
        self._exit_event = mp.Event()
        self._conn_event = mp.Event()

        # streaming variables
        self._state = np.zeros( ( self._channelcount, ) )
        self._state[8:] = np.array( [ 1.0, 0.0, 0.0, 0.0 ] ) # initialize quaternion
        self._state_buffer = mp.Queue()
        self._state_buffer_max = int( 5 * srate )

        self._stream_event = mp.Event()
        self._print_event = mp.Event()
        self._streamer = None

        # viewing variables
        self._view_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        self._streamer = mp.Process( target = self._connect, args = ( com, ) )
        self._streamer.start()
        
        self._conn_event.wait()

    def __del__(self):
        """ 
        Destructor 

        This cleans up any spawned child processes / resources for the interface
        """
        try:
            if self._streamer.is_alive:
                self._stream_event.clear()
                self._exit_event.set()
                self._streamer.join()
        except AttributeError: pass # never got to make the I/O thread
        try:
            if self._viewer.is_alive: self.hide()
        except AttributeError: pass # no viewer exists currently

    def _detect_tty(self):
        """
        Detects the serial port that the BlueGiga BLE dongle is attached to (linux only)

        Returns
        -------
        str or None
            Serial port of BLE dongle
        """
        for p in comports():
            if re.search( r'PID=2458:0*1', p[2] ): return p[0]
        return None

    def _write_attr( self, attr, val ):
        """
        Writes value to specific attribute

        Parameters
        ----------
        attr : bytes
            The attribute to write to
        val : byte
            The value to write
        """
        if self._conn is not None:
            self._bt.write_attr( self._conn, attr, val )

    def _read_attr( self, attr ):
        """
        Reads value of specific attribute

        Parameters
        ----------
        attr : bytes
            The attribute to read from

        Returns
        -------
        byte or None
            The value of the attribute or None if no Myo is connected
        """
        if self._conn is not None:
            return self._bt.read_attr( self._conn, attr )
        return None

    def _connect(self, com):
        """
        Connect to the specified Myo Armband
        
        Parameters
        ----------
        com : str
            Serial port that the BlueGiga dongle is connected to

        Raises
        ------
        RuntimeError
            BlueGiga BLE dongle is not attached (or cannot be detected)
        ValueError
            No MyoArmband with the specified MAC address is available

        Notes
        -----
        This function is the target of the child polling process
        """
        try:
            # standard initialization stuff
            if not len( com ): com = self._detect_tty()
            if com is None: raise RuntimeError( 'Cannot discover BlueGiga dongle!' )
                
            self._bt = MyoArmband.MyoBT( com )

            # connection
        
            # stop everything from before
            self._bt.end_scan()
            for i in range( 0, 3 ): self._bt.disconnect( i )
            
            # start scanning
            addr = None
            self._bt.discover()
            t0 = time.time()
            while ( time.time() - t0 < MyoArmband.CONNECTION_TIMEOUT ):
                p = self._bt.recv_packet()
                if p.payload.endswith( b'\x06\x42\x48\x12\x4a\x7f\x2c\x48\x47\xb9\xde\x04\xa9\x01\x00\x06\xd5' ):
                    if p.payload[2:8] == self._mac[::-1]:
                        addr = list( MyoArmband.multiord( p.payload[2:8] ) )
                        break
            if addr is None: raise ValueError( 'Could not find Myo Armband with MAC address', self._mac )
            self._bt.end_scan()

            # connect and wait for status event
            conn_pkt = self._bt.connect( addr )
            self._conn = MyoArmband.multiord( conn_pkt.payload )[-1]
            self._bt.wait_event( 3, 0 )

            # get firmware version
            fw = self._read_attr( 0x17 )
            _, _, _, _, v0, _, _, _ = MyoArmband.unpack( 'BHBBHHHH', fw.payload )
            if v0 == 0: # old firmware
                self._write_attr( 0x19, b'\x01\x02\x00\x00' )
                # subscribe to EMG data notifications
                self._write_attr( 0x2f, b'\x01\x00' )
                self._write_attr( 0x2c, b'\x01\x00' )
                self._write_attr( 0x32, b'\x01\x00' )
                self._write_attr( 0x35, b'\x01\x00' )

                
                self._write_attr( 0x28, b'\x01\x00' )  # enable EMG data
                self._write_attr( 0x1d, b'\x01\x00' )  # enable IMU data

                # set up streaming parameters
                C = 1000
                emg_hz = 50
                emg_smooth = 100
                imu_hz = 50
                self._write_attr( 0x19, MyoArmband.pack( 'BBBBHBBBBB', 2, 9, 2, 1, C, emg_smooth, C // emg_hz, imu_hz, 0, 0 ) )
            else:
                # name = self.__read_attr( 0x03 )         # get device name, can use this later
                self._write_attr( 0x1d, b'\x01\x00' )  # enable imu
                # self.__write_attr( 0x24, b'\x02\x00' )  # enable on/off arm notifications

                # enable emg streaming
                self._write_attr( 0x2c, b'\x01\x00' )  # subscribe to EmgData0Characteristic
                self._write_attr( 0x2f, b'\x01\x00' )  # subscribe to EmgData1Characteristic
                self._write_attr( 0x32, b'\x01\x00' )  # subscribe to EmgData2Characteristic
                self._write_attr( 0x35, b'\x01\x00' )  # subscribe to EmgData3Characteristic

                if self._raw: 
                    self._write_attr( 0x19, b'\x01\x03\x02\x01\x00' )
                else:                      # hidden functionality, no guarantees moving forward
                    self._write_attr( 0x28, b'\x01\x00' )
                    self._write_attr( 0x19, b'\x01\x03\x01\x01\x00' )

            # self.__write_attr( 0x12, b'\x01\x10' )       # enable battery notifications
            self._write_attr( 0x19, MyoArmband.pack( '3B', 9, 1, 1 ) ) # turn off sleep

            def handle_data( p ):
                if ( p.cls, p.cmd ) != ( 4, 5 ): return
                _, attr, _ = MyoArmband.unpack( 'BHB', p.payload[:4] )
                pay = p.payload[5:]
                if attr == 0x27:                                                    # filtered EMG
                    self._state[:8] = MyoArmband.unpack( '8HB', pay )[:8]
                elif attr == 0x2b or attr == 0x2e or attr == 0x31 or attr == 0x34:  # raw EMG
                    self._emg_value[:] = struct.unpack( '<16b', pay )
                    self._emg_time.value = time.time()
                elif attr == 0x1c:                                                  # IMU
                    self._state[8:] = np.array( MyoArmband.unpack( '10h', pay )[:4] ) / MyoArmband.MYOHW_ORIENTATION_SCALE
                else: pass                                                          # unsupported

            # stream data
            self._bt.add_handler( handle_data )
            self._conn_event.set()
            while not self._exit_event.is_set():                   # whle we are not exiting
               if self._stream_event.is_set(): self._stream()     # stream when asked
        finally:
            self._conn_event.set()
            self._disconnect()

            # drain all processing queues so we can join
            self.flush()
            empty = False
            while not empty:
                try: self._plot_buffer.get( timeout = 1e-3 )
                except queue.Empty: empty = True

    def _disconnect(self):
        """
        Disconnect from the connected Myo Armband
        """
        if self._conn is not None:
            self._bt.disconnect( self._conn )

    def _read(self):
        """ 
        Reads a single sample from the MyoArmband

        While this function does not return anything, it sets the __state variable
        to the last measured sensor readings.
        """
        self._bt.recv_packet( self._speriod )
        if self._raw:
            idx = 8 * ( ( time.time() - self._emg_time.value ) > 5e-3 )
            self._state[:8] = self._emg_value[idx:idx+8]

        while self._state_buffer.qsize() > self._state_buffer_max: 
            self._state_buffer.get( timeout = 1e-3 )
        self._state_buffer.put( self._state.copy(), timeout = 1e-3 )

        if self._print_event.is_set(): print( self._name, ':', self._state )
        if self._view_event.is_set(): self._plot_buffer.put( self._state )
            
    def _stream(self):
        """ 
        Streams data from the MyoArmband at the specified sampling rate 
        """
        t = time.time()
        while self._stream_event.is_set():
            t = t + self._speriod
            self._read()
            while max( t - time.time(), 0 ): ns_sleep( 1e2 )

    def _plot(self):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self._name )
        
        # orientation plots
        orientations = []
        orient_colors = 'rgb'
        orient_titles = [ 'ROLL', 'PITCH', 'YAW' ]
        for i in range( 0, 3 ):
            ax = gui.add_subplot( 2, 3, i + 1, 
                 projection = 'polar', aspect = 'equal' )
            ax.plot( np.linspace( 0, 2*np.pi, 100 ), 
                     np.ones( 100 ), color = orient_colors[ i ],
                     linewidth = 2.0 )
            orientations.append( ax.plot( np.zeros( 2 ), np.linspace( 0, 1, 2 ), 
                                         color = orient_colors[ i ], linewidth = 2.0  ) )
            ax.set_rticks( [] )
            ax.set_rmax( 1 )
            ax.set_xlabel( orient_titles[ i ] )
            ax.grid( True )

        # line plot
        emg_plots = []
        if self._raw: emg_offsets = np.array( [ 128, 384, 640, 896, 1152, 1408, 1664, 1920 ] )
        else: emg_offsets = np.array( [ 500, 1500, 2500, 3500, 4500, 5500, 6500, 7500 ] )

        ax = gui.add_subplot( 2, 1, 2 )
        num_emg_samps = int( 5 * np.round( 1.0 / self._speriod ) )
        for i in range( 0, 8 ):
            xdata = np.linspace( 0, 1, num_emg_samps )
            ydata = emg_offsets[ i ] * np.ones( num_emg_samps )
            emg_plots.append( ax.plot( xdata, ydata ) )
        if self._raw: ax.set_ylim( 0, 2048 )
        else: ax.set_ylim( 0, 9000 )
        ax.set_xlim( 0, 1 )
        ax.set_yticks( emg_offsets.tolist() )
        ax.set_yticklabels( [ 'EMG01', 'EMG02', 'EMG03', 'EMG04', 'EMG05', 'EMG06', 'EMG07', 'EMG08' ] )
        ax.set_xticks( [] )  

        plt.tight_layout()
        plt.show( block = False )
        while self._view_event.is_set():
            try:
                data = []
                while self._plot_buffer.qsize() > 0: data.append( self._plot_buffer.get() )
                if data:
                    # concate to get a block of data
                    data = np.vstack( data )

                    # update orientation data
                    angles = quat.to_euler( data[-1, 8:] )
                    for i in range( 0, 3 ):
                        tdata = np.ones( 2 ) * angles[ i ]
                        rdata = np.linspace( 0, 1, 2 )
                        orientations[ i ][ 0 ].set_data( tdata, rdata )
                
                    # update electrophysiological data
                    for i in range( 0, 8):
                        ydata = emg_plots[ i ][ 0 ].get_ydata()
                        ydata = np.append( ydata, data[ :, i ] + emg_offsets[ i ] )
                        ydata = ydata[-num_emg_samps:]
                        emg_plots[ i ][ 0 ].set_ydata( ydata )
                plt.pause( 0.005 )
            except: self._view_event.clear()
        plt.close( gui )

    @property
    def name(self):
        """
        The unique handle for the device
        
        Returns
        -------
        str
            The specified unique handle of this device interface
        """ 
        return self._name

    @property
    def state(self):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        try:
            return self._state_buffer.get( timeout = 1e-3 )
        except queue.Empty:
            return None

    @property
    def speriod(self):
        """
        The sampling period for the device
        
        Returns
        -------
        float
            The sampling period of the device
        """ 
        return self._speriod

    @property
    def channelcount(self):
        """
        The channel count for the device

        Returns
        -------
        int
            The number of channels per sensor measurement
        """ 
        return self._channelcount

    def run(self, display = False):
        """ 
        Starts the acquisition process of the Myo Armband 
        
        Parameters
        ----------
        display : bool
            Flag determining whether sensor measurements should be printed to console (True) or not (False)
        """
        if not self._stream_event.is_set():
            if display: self._print_event.set()
            else: self._print_event.clear()
            self._stream_event.set()

    def flush(self):
        """
        Dispose of all previous data
        """
        empty = False
        while not empty:
            try: self._state_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True

    def stop(self):
        """
        Stops the acquisition process of the MyoArmband
        """
        if self._stream_event.is_set():
            self._stream_event.clear()

    def view(self):
        """ 
        Launches the GUI viewer of the MyoArmband 
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot )
            self._viewer.start()
            
    def hide(self):
        """
        Closes the GUI viewer of the MyoArmband
        """
        if self._view_event.is_set():
            self._view_event.clear()
            self._viewer.join()

    def vibrate(self, length):
        """
        Vibrates the Myo Armband

        Parameters
        ----------
        length : byte
            The strength of vibration (0 - 255)

        Notes
        -----
        This functionality is experimental and has yet to be fully tested
        """
        if length in range( 1, 4 ):
            self._write_attr( 0x19, MyoArmband.pack( '3B', 3, 1, length ) )

if __name__ == '__main__':
    import sys
    import inspect
    import argparse

    # helper function for booleans
    def str2bool( v ):
        if v.lower() in [ 'yes', 'true', 't', 'y', '1' ]: return True
        elif v.lower() in [ 'no', 'false', 'n', 'f', '0' ]: return False
        else: raise argparse.ArgumentTypeError( 'Boolean value expected!' )

    # parse commandline entries
    class_init = inspect.getargspec( MyoArmband.__init__ )
    arglist = class_init.args[1:]   # first item is always self
    defaults = class_init.defaults
    parser = argparse.ArgumentParser()
    for arg in range( 0, len( arglist ) ):
        try: tgt_type = type( defaults[ arg ][ 0 ] )
        except: tgt_type = type( defaults[ arg ] )
        if tgt_type is bool:
            parser.add_argument( '--' + arglist[ arg ], 
                             type = str2bool, nargs = '?',
                             action = 'store', dest = arglist[ arg ],
                             default = defaults[ arg ] )
        else:
            parser.add_argument( '--' + arglist[ arg ], 
                                type = tgt_type, nargs = '+',
                                action = 'store', dest = arglist[ arg ],
                                default = defaults[ arg ] )
    args = parser.parse_args()
    for arg in range( 0, len( arglist ) ):
        attr = getattr( args, arglist[ arg ] )
        if isinstance( attr, list ) and not isinstance( defaults[ arg ], list ):
            setattr( args, arglist[ arg ], attr[ 0 ]  )

    # create interface
    myo = MyoArmband( name = args.name, com = args.com, mac = args.mac, srate = args.srate, raw = args.raw )
    myo.run( display = False )
    myo.view()

    while True:
        state = myo.state
        if state is not None:
            print( state )