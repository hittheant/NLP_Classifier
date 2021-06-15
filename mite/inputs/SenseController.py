import bluetooth

import time
import queue

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from .. import ns_sleep
from . import AbstractBaseInput

class SenseController(AbstractBaseInput):
    # Packet commands
    CMD_CONNECT = 0x80
    CMD_START_SIGNALS = 0x8C
    CMD_STOP_SIGNALS = 0x8D

    # Packet messages
    MSG_ACK          = 0x00
    MSG_SIGNAL_DATA  = 0x05
    MSG_CONNECT_DATA = 0x06

    # Streaming
    STREAM_RATE_MS_DEFAULT = 1

    # Signal types
    STREAM_TYPE_EMG     = 1
    STREAM_TYPE_ENVELOP = 2
    STREAM_TYPE_RAW     = 3

    # Rx buffer data
    RX_BUFFER_DEFAULT = 20
    RX_BUFFER_MIN     = 5
    RX_BUFFER_MAX     = 1000

    class Packet:
        def __init__(self, cmd = 0, length = 0, data = []):
            self.cmd = cmd
            self.length = length
            self.data = data.copy()
            self.checksum = 0

    @staticmethod
    def checksum( array ):
        chk = 0
        for item in array: chk += item
        return chk & 0xFF

    # TX commands to send to controller
    @staticmethod
    def tx_pack( cmd, length, data ):
        txba = bytearray()
        txba.append( 0xFF )
        txba.append( length + 2 )
        txba.append( cmd )

        # data loop
        for dta in data: 
            txba.append( dta)

        txba.append( SenseController.checksum( txba ) )
        return bytes( txba )

    @staticmethod
    def tx_connect():
        dta = bytearray()
        for n in range( 0, 4 ): 
            dta.append( 0 )
        return SenseController.tx_pack( SenseController.CMD_CONNECT, len( dta ), dta )

    @staticmethod
    def tx_ack( cmd ):
        dta = bytearray()
        dta.append( cmd )
        return SenseController.tx_pack( SenseController.MSG_ACK, len( dta ), dta )

    def __init__(self, name = 'Sense', mac = 'ec:fe:7e:1b:8e:bd', num_electrodes = 8, srate = 1000, mode = 'emg' ):
        # device variable
        self._mac = mac
        self._name = name
        self._channelcount = num_electrodes
        self._speriod = 1.0 / srate

        if mode == 'emg': self._stream_type = SenseController.STREAM_TYPE_EMG
        elif mode == 'envelope': self._stream_type = SenseController.STREAM_TYPE_ENVELOP
        elif mode == 'raw': self._stream_type = SenseController.STREAM_TYPE_RAW
        else: raise RuntimeError( 'Unsupported streaming mode!', mode )

        # hardware communication variables
        self._bt = None
        self._rx_state = 0x00
        self._rx_pkt = SenseController.Packet()
        self._rx_pkt_buffer = []
        self._rx_data_buffer = []
        self._rx_buffer_size = SenseController.RX_BUFFER_DEFAULT

        # state variables
        self._state = np.zeros( ( self._channelcount, ) )
        self._state_buffer = mp.Queue()
        self._state_buffer_max = int( 5 * srate )

        # streaming variables
        self._exit_event = mp.Event()
        self._conn_event = mp.Event()
        self._stream_event = mp.Event()
        self._print_event = mp.Event()

        # viewing variables
        self._view_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        self._streamer = mp.Process( target = self._connect )
        self._streamer.start()

        self._conn_event.wait()
    
    def __del__(self):
        """
        Destructor

        Stops the Sense controller and closes communication.
        """
        # try:
        #     # stop core streaming
        #     dta = bytearray()
        #     dta.append( ( int( 1e3 * self._speriod ) >> 8 & 0xFF ) )
        #     dta.append( ( int( 1e3 * self._speriod ) & 0xFF ) )
        #     dta.append( self._stream_type )
        #     self._tx_send( SenseController.tx_pack( SenseController.CMD_STOP_SIGNALS, len( dta ), dta ) )

        #     self._bt.close()                       # close communication
        # except (AttributeError, OSError): pass # did not open the bluetooth communication
        try:
            if self._streamer.is_alive:
                self._stream_event.clear()
                self._exit_event.set()
                self._streamer.join()
        except AttributeError: pass # never got to make the I/O thread
        try:
            if self._viewer.is_alive: self.hide()
        except AttributeError: pass # no viewer exists currently

    def _tx_send( self, pkt ):
        """
        """
        self._bt.send( pkt )

    def _rx_push( self, pkt ):
        """
        """
        self._rx_pkt_buffer.append( SenseController.Packet( pkt.cmd, pkt.length, pkt.data ) )

    def _rx_parse( self, rxb ):
        """
        Parses the RX bytes
        """
        for b in rxb:
            if self._rx_state == 0x00:
                if b == 0xFF:
                    self._rx_state = 0x01
                    self._rx_pkt.checksum = b
                    self._rx_pkt.data.clear()

            elif self._rx_state == 0x01:
                if b == 0xAA: 
                    self._rx_state = 0x02
                else: 
                    self._rx_state = 0x00
            elif self._rx_state == 0x02: 
                self._rx_state = 0x03
            elif self._rx_state == 0x03:
                self._rx_state = 0x04
                self._rx_pkt.length = b - 2
                self._rx_pkt.checksum += b
            elif self._rx_state == 0x04:
                self._rx_pkt.cmd = b
                self._rx_pkt.checksum += b

                if self._rx_pkt.length > 0:
                    self._rx_state = 0x05
                else:
                    self._rx_state = 0x06
            elif self._rx_state == 0x05:
                self._rx_pkt.data.append( b )
                self._rx_pkt.checksum += b
                if len( self._rx_pkt.data ) >= self._rx_pkt.length:
                    self._rx_state = 0x06
            elif self._rx_state == 0x06:
                if ( self._rx_pkt.checksum & 0xFF ) == b:
                    self._rx_push( self._rx_pkt )
                self._rx_state = 0x00

    def _rx_pkt_handler( self ):
        pkt = self._rx_pkt_buffer.pop( 0 )

        # don't ACK streams
        if pkt.cmd != SenseController.MSG_SIGNAL_DATA:
            self._tx_send( SenseController.tx_ack( pkt.cmd ) )
        else:
            if len( self._rx_data_buffer ) < self._rx_buffer_size: # NOTE: will drop data if it gets filled
                self._rx_data_buffer.append( pkt.data )

    def _connect(self):
        """
        Connect to specified Sense controller

        Notes
        -----
        This function is the target of the the child polling process
        """
        try:
            # connect to device
            self._bt = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
            self._bt.connect( ( self._mac, 1 ) )
            self._bt.setblocking( False )
            self._tx_send( SenseController.tx_connect() )
            
            # start core streaming
            dta = bytearray()
            dta.append( ( int( 1e3 * self._speriod ) >> 8 & 0xFF ) )
            dta.append( ( int( 1e3 * self._speriod ) & 0xFF ) )
            dta.append( self._stream_type )

            self._conn_event.set()
            while not self._exit_event.is_set():                       # whle we are not exiting
                if self._stream_event.is_set():
                    self._tx_send( SenseController.tx_pack( SenseController.CMD_START_SIGNALS, len( dta ), dta ) )  # send stream command to controller
                    self._stream()                                                                                  # stream when asked
                    self._tx_send( SenseController.tx_pack( SenseController.CMD_STOP_SIGNALS, len( dta ), dta ) )   # stop stream command to controller
                self._rx_data_buffer.clear()

        finally:
            # close bluetooth if we opened it
            if self._bt is not None:
                try:
                    dta = bytearray()
                    dta.append( ( int( 1e3 * self._speriod ) >> 8 & 0xFF ) )
                    dta.append( ( int( 1e3 * self._speriod ) & 0xFF ) )
                    dta.append( self._stream_type )
                    self._tx_send( SenseController.tx_pack( SenseController.CMD_STOP_SIGNALS, len( dta ), dta ) )   # stop stream command to controller
                    self._bt.close()
                except OSError: pass                                                                                # did not connect to controller

            # drain all processing queues so we can join
            self.flush()
            empty = False
            while not empty:
                try: self._plot_buffer.get( timeout = 1e-3 )
                except queue.Empty: empty = True
            
            # signal connection so main process isn't held up
            self._conn_event.set()

    def _read(self):
        """ 
        Reads a single sample from the Sense controller

        While this function does not return anything, it sets the _state variable
        to the last measured sensor readings.
        """
        # populate the RX buffer
        data = None
        while data is None:
            # read bluetooth data if needed
            try: 
                rx = self._bt.recv( 255 )
                # print( 'INCOMING BYTES:', len( rx ) )
                self._rx_parse( rx )
            except:
                pass

            # get the oldest packet in rx buffer (if there is one)
            while len( self._rx_pkt_buffer ) > 0:
                # print( 'BUFFER SIZE:', len( self._rx_pkt_buffer ) )
                self._rx_pkt_handler()

                if len( self._rx_data_buffer ) > 0:
                    data = self._rx_data_buffer.pop( 0 )

                    for i in range( 1, 2 * self._channelcount, 2 ):
                        self._state[i//2] = np.int16( ( data[i] << 8 ) + data[i+1] )
        
                    # store data in device state
                    while self._state_buffer.qsize() > self._state_buffer_max: 
                        self._state_buffer.get( timeout = 1e-3 )
                    self._state_buffer.put( self._state.copy(), timeout = 1e-3 )

                    if self._print_event.is_set(): print( self._name, ':', self._state )
                    if self._view_event.is_set(): self._plot_buffer.put( self._state )

    def _stream(self):
        """ 
        Streams data from the Sense controller at the specified sampling rate 
        
        Notes
        -----
        This function is the target of the the child polling process
        """
        t = time.time()
        while self._stream_event.is_set():
            self._read()

    def _plot(self):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self._name )

        # line plot
        emg_plots = []
        emg_offsets = (2 ** 16 - 1 ) * np.arange( 0, self._channelcount ) + ( 2 ** 16 / 2 )

        ax = gui.add_subplot( 111 )
        num_emg_samps = int( 5 * np.round( 1.0 / self._speriod ) )
        for i in range( 0, 8 ):
            xdata = np.linspace( 0, 1, num_emg_samps )
            ydata = emg_offsets[ i ] * np.ones( num_emg_samps )
            emg_plots.append( ax.plot( xdata, ydata ) )
        
        ax.set_ylim( 0, emg_offsets[-1] + ( 2 ** 16 / 2 ) )
        ax.set_xlim( 0, 1 )
        ax.set_yticks( emg_offsets.tolist() )
        ax.set_yticklabels( [ 'EMG01', 'EMG02', 'EMG03', 'EMG04', 'EMG05', 'EMG06', 'EMG07', 'EMG08' ] )
        ax.set_xticks( [] )  

        plt.tight_layout()
        plt.show( block = False )
        while self._view_event.is_set():
            try:
                data = []
                while self._plot_buffer.qsize() > 0:
                    data.append( self._plot_buffer.get() )
                    
                if data:
                    # concatenate to get a block of data
                    data = np.vstack( data )
                
                    # update electrophysiological data
                    for i in range( 0, self._channelcount ):
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
        Starts the acquisition process of the Sense controller
        
        Parameters
        ----------
        display : bool
            Flag determining whether sensor measurements should be printed to console (True) or not (False)
        """
        if not self._stream_event.is_set():
            self._stream_event.set()
            if display: self._print_event.set()
            else: self._print_event.clear()

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
        Stops the acquisition process of the Sense controller
        """
        if self._stream_event.is_set():
            self._stream_event.clear()

    def view(self):
        """ 
        Launches the GUI viewer of the Sense data 
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot )
            self._viewer.start()

    def hide(self):
        """ 
        Closes the GUI viewer of the Sense data
        """
        if self._view_event.is_set():
            self._view_event.clear()
            self._viewer.join()

if __name__ == "__main__":
    import sys
    import inspect
    import argparse

    # helper function for booleans
    def str2bool( v ):
        if v.lower() in [ 'yes', 'true', 't', 'y', '1' ]: return True
        elif v.lower() in [ 'no', 'false', 'n', 'f', '0' ]: return False
        else: raise argparse.ArgumentTypeError( 'Boolean value expected!' )

    # parse commandline entries
    class_init = inspect.getargspec( SenseController.__init__ )
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

    ibt = SenseController( name = args.name, mac = args.mac, num_electrodes = args.num_electrodes, srate = args.srate, mode = args.mode )
    ibt.run( display = False )
    ibt.view()

    while True:
        state = ibt.state
        if state is not None:
            print( state )