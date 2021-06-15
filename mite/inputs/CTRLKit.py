import json
import asyncio
import websockets

import os
import sys
import subprocess

import ctypes
import queue
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt

from .. import ns_sleep
from . import AbstractBaseInput

class CTRLKit(AbstractBaseInput):
    CTRL_NUM_CHANNELS = 16
    # CTRL_SAMPLES_PER_PACKET = 16 # NOTE: This is no longer enforced
    CTRL_MAX_CONNECTION_ATTEMPTS = 5

    def __init__(self, name = 'CTRL-EMG', port = 'ws://localhost:9999', num_samples = 64):
        """ 
        Constructor 
        
        Parameters
        ----------
        name : str
            The unique device handle used to refer to the connected hardware
        port : str
            The websocket address of the CTRL kit
        srate : float
            The sampling rate used to poll the device (maximum is 2000 Hz)
        num_samples : int
            The number of samples to return in one read

        Returns
        -------
        obj
            A CTRLKit interface object
        """
        srate = 2000.0

        # device variables
        self._name = name
        self._port = port
        self._channelcount = CTRLKit.CTRL_NUM_CHANNELS
        self._speriod = 1.0 / srate

        self._num_samples = int( num_samples )
        self._state = np.zeros( ( self._num_samples, self._channelcount ) )
        self._state_buffer = mp.Queue()
        self._state_buffer_max = int( 5 * srate )
        
        # streaming variables
        self._conn_event = mp.Event()
        self._exit_event = mp.Event()
        self._stream_event = mp.Event()
        self._print_event = mp.Event()
        self._streamer = None

        # packeting variables
        self._newdata_event = mp.Event()
        self._samples_in_buffer = mp.Value( 'i', 0 )

        # viewing variables
        self._view_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        # start streaming service
        platform = sys.platform.lower()
        ctrl_root = os.path.abspath( os.path.dirname( __file__ ) )
        if platform == 'linux': service_path = ctrl_root + '/external/ctrlkit/linux/bin/ctrlservice'
        elif platform == 'darwin': service_path = ctrl_root + '/external/ctrlkit/mac/bin/ctrlservice'
        elif platform == 'win32': service_path = ctrl_root + '/external/ctrlkit/windows/ctrlservice.exe'
        else: raise RuntimeError( 'Unsupported platform!', platform )
        
        self._ctrl_service = subprocess.Popen( [ service_path ],
                                                 shell = True,
                                                 stdin = subprocess.DEVNULL,
                                                 stdout = subprocess.DEVNULL,
                                                 stderr = subprocess.DEVNULL,
                                                 close_fds = True )

        # connect to device
        self._streamer = mp.Process( target = self._stream )
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
        except AttributeError: pass # never got to make the I/O process
        try:
            self._ctrl_service.kill()  # TODO: Test that this kills the subprocess
        except AttributeError: pass
        try:
            if self._viewer.is_alive: self.hide()
        except AttributeError: pass # no viewer exists currently

    def _stream(self):
        """
        Create asyncio event loop to stream data
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete( self._read() )
        loop.close()

        # clean processing queues so we can join
        self.flush()
        empty = False
        while not empty:
            try: self._plot_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True


    async def _read(self):
        """
        Connect and read data from a CTRL kit device asynchronously
        """
        for _ in range( CTRLKit.CTRL_MAX_CONNECTION_ATTEMPTS ):
            try:
                async with websockets.connect( self._port ) as websocket:
                    start_msg = { 'api_version' : '0.10',
                                  'api_request' : { 'request_id'           : 1,
                                                    'start_stream_request' : { 'stream_id'      : self._name,
                                                                               'app_id'         : 'MITE',
                                                                               'raw_emg_target' : {} } } }
                    await websocket.send( json.dumps( start_msg ) )
                    response = await websocket.recv()

                    self._conn_event.set()

                    buffer = []
                    while not self._exit_event.is_set():
                        if self._stream_event.is_set():
                            try:
                                msg = await websocket.recv()
                            except websockets.exceptions.ConnectionClosed:
                                break

                            data = json.loads( msg )
                            for sample in data['stream_batch']['raw_emg_batch']['samples']:
                                # emg = 0.01 * ( np.array( sample['raw_emg'] ) - 2047 ) # NOTE: This is APL conversion, doesn't work
                                buffer.append( 1e4 * np.array( sample['raw_emg'] ) )

                            samples = np.vstack( buffer )
                            self._samples_in_buffer.value = samples.shape[0]

                            if self._samples_in_buffer.value >= self._num_samples:
                                try:
                                    self._state_buffer.put( samples[:self._num_samples,:], timeout = 1e-3 )
                                except queue.Full:
                                    self._state_buffer.get()
                                    self._state_buffer.put( samples[:self._num_samples,:], timeout = 1e-3 )
                                if self._view_event.is_set(): self._plot_buffer.put( samples[:self._num_samples,:] )
                                if self._print_event.is_set(): print( self._name, ':', samples[0,:] )

                                buffer = [ samples[-self._num_samples,:] ]
                                self._samples_in_buffer.value -= self._num_samples
                    return   # break out of connection attempts
            except OSError: 
                ns_sleep( 3e9 ) # connect once every 3 s
        self._conn_event.set()
        raise RuntimeError( 'Maximum connection attempts reached without response!', CTRLKit.CTRL_MAX_CONNECTION_ATTEMPTS )

    def _plot(self):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self._name )
        
        offset = np.arange( self._channelcount ) * 20 + 10
        ax = gui.add_subplot( 111 )

        plots = []
        num_emg_samples = int( 5 * np.round( 1.0 / self.__speriod ) )
        for i in range( self._channelcount ):
            xdata = np.linspace( 0, 1, num_emg_samples )
            ydata = offset[i] * np.ones( num_emg_samples )
            plots.append( ax.plot( xdata, ydata ) )
        ax.set_xlim( 0, 1 )
        # ax.set_ylim( 0, XXX )
        ax.set_yticks( offset.tolist() )
        ax.set_yticklabels( [ 'EMG%02d' % (i+1) for i in range( self._channelcount ) ] )
        ax.set_xticks( [] )

        plt.tight_layout()
        plt.show( block = False )
        while self._view_event.is_set():
            try:
                data = []
                while self._plot_buffer.qsize() > 0: data.append( self._plot_buffer.get() )
                if data:
                    # concatenate to get a block of data
                    data = np.vstack( data )

                    # update EMG data
                    for i in range( 0, self._channelcount ):
                        ydata = plots[i][0].get_ydata()
                        ydata = np.append( ydata, data[:,i] + offset[i] )
                        ydata = ydata[-num_emg_samples:]
                        plots[i][0].set_ydata( ydata )
                plt.pause( 0.001 )
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
        Starts the acquisition process of the CTRL kit
        
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
        Stops the acquisition process of the CTRL kit
        """
        if self._stream_event.is_set():
            self._stream_event.clear()
        while self._state_buffer.qsize() > 0: self._state_buffer.get()

    def view(self):
        """ 
        Launches the GUI viewer of the CTRL kit
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot )
            self._viewer.start()
            
    def hide(self):
        """
        Closes the GUI viewer of the CTRL kit
        """
        if self._view_event.is_set():
            self._view_event.clear()
            self._viewer.join()

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
    class_init = inspect.getargspec( CTRLKit.__init__ )
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
    ctrl = CTRLKit( name = args.name, port = args.port, num_samples = args.num_samples )
    ctrl.run( display = False )
    ctrl.view()

    while True:
        state = ctrl.state
        if state is not None:
            print( state[0,:] )