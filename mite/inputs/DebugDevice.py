import time
import ctypes
import queue
import numpy as np

import multiprocessing as mp
import matplotlib.pyplot as plt

from .. import ns_sleep
from . import AbstractBaseInput

class DebugDevice(AbstractBaseInput):
    """ Python implementation of a fake hardware device for debug purposes """
    DEFAULT_BUFFER_TIME = 5.0
    DEFAULT_MIN_FREQ = 10.0
    DEFAULT_MAX_FREQ = 500.0
    DEFAULT_ACTIVATION_PROB = 0.75

    ENVELOPE_AMP_RANGE = (2.0, 9.0)
    ENVELOPE_RISE_RANGE = ( -200.0, -500.0 )
    ENVELOPE_DECAY_RANGE = (-1.0, -1.0 / 25.0)
    ENVELOPE_WIDTH_RANGE = (1.0, 5.0)

    @staticmethod
    def compute_envelope( amp, exp_rise, exp_fall, width ):
        """ 
        Computes the amplitudes for an activation envelope of EMG
        
        Parameters
        ----------
        amp : float
            The maximum amplitude of the activation envelope
        exp_rise : float
            The exponential growth factor for the start of activation
        exp_fall : float
            The exponential decay factor for the majority of the envelope
        width : float
            The width of the activation envelope (in samples)

        Returns
        -------
        numpy.ndarray
            The magnitudes for each sample of the activation envelope
        """
        x0 = np.linspace( 0.0, 0.2, np.round( 0.2 * width ) )
        xf = np.linspace( 0.0, 0.8, np.round( 0.8 * width ) )

        growth = 1.0 - np.exp( exp_rise * x0 )
        decay = np.exp( exp_fall * xf ) * growth[-1]
        envelope = amp * np.hstack( [ growth, decay ] )
        
        return envelope

    def __init__(self, name = 'Debug', num_channels = 8, srate = 100.0):
        """ 
        Constructor

        Parameters
        ----------
        name : str
            The unique device handle used to refer to the debug hardware
        num_channels : int
            The number of channels for this device
        srate : float
            The sampling rate used to poll the device

        Returns
        -------
        obj
            A DebugDevice interface object
        """
        # device variables
        self.__name = name
        self.__channelcount = num_channels
        self.__speriod = 1.0 / srate
        self.__state = np.zeros( ( self.__channelcount, ) )

        # signal variables
        self.__signal_buffer = [ None ] * self.__channelcount
        self.__buffercount = np.inf
        self.__buffersize = int( DebugDevice.DEFAULT_BUFFER_TIME * srate )
        self.__min_frequency = DebugDevice.DEFAULT_MIN_FREQ
        self.__max_frequency = DebugDevice.DEFAULT_MAX_FREQ

        # streaming variables
        self.__stream_event = mp.Event()
        self.__print_event = mp.Event()
        self.__streamer = None

        # new data / buffer signaling
        self.__state_buffer = mp.Queue( maxsize = int( 5 * srate ) )

        # viewing variables
        self.__view_event = mp.Event()
        self.__plot_buffer = mp.Queue()
        self.__viewer = None

    def __del__(self):
        """
        Destructor

        This cleans up any spawned child processes / resources for the interface
        """
        try:
            if self.__streamer.is_alive(): self.stop()
        except AttributeError: pass
        try:
            if self.__viewer.is_alive(): self.hide()
        except AttributeError: pass

    def __generate_signal(self):
        """ 
        Generates the EMG signal buffer for each channel
        """
        for i in range( 0, self.__channelcount ):

            self.__signal_buffer[i] = np.random.randn( self.__buffersize )

            # decide if we are activating in this buffer
            activate = np.random.random() > DebugDevice.DEFAULT_ACTIVATION_PROB
            if activate:
                amplitude = np.random.uniform( DebugDevice.ENVELOPE_AMP_RANGE[0], DebugDevice.ENVELOPE_AMP_RANGE[1] )
                growth = np.random.uniform( DebugDevice.ENVELOPE_RISE_RANGE[0], DebugDevice.ENVELOPE_RISE_RANGE[1] )
                decay = np.random.uniform( DebugDevice.ENVELOPE_DECAY_RANGE[0], DebugDevice.ENVELOPE_DECAY_RANGE[1] )
                width = int( np.random.uniform( DebugDevice.ENVELOPE_WIDTH_RANGE[0], DebugDevice.ENVELOPE_WIDTH_RANGE[1] ) / self.__speriod )

                j = np.random.randint( 0, self.__buffersize - width )
                envelope = DebugDevice.compute_envelope( amplitude, growth, decay, width )
                self.__signal_buffer[i][j:j+width] *= envelope

        self.__buffercount = 0

    def __read(self):
        """ 
        Reads a single sample from the debug device 
        
        While this function does not return anything, it sets the __state variable
        to the last measured sensor readings.
        """
        if self.__buffercount >= self.__buffersize: self.__generate_signal()
        signal = np.array( [ self.__signal_buffer[i][self.__buffercount] for i in range( self.__channelcount ) ] )
        
        try:
            self.__state_buffer.put( signal, timeout = 1e-3 )
        except queue.Full:
            self.__state_buffer.get()
            self.__state_buffer.put( signal, timeout = 1e-3 )
        
        self.__buffercount += 1
        
        if self.__view_event.is_set(): self.__plot_buffer.put( signal )
        if self.__print_event.is_set(): print( self.__name, ':', signal )
        
    def __stream(self, seed):
        """ 
        Streams data from the debug device at the specified sampling rate

        Parameters
        ----------
        seed : int
            The seed to initialize the random number generator

        Notes
        -----
        This function is the target of the the child polling process
        """
        np.random.seed( seed )
        t = time.time()
        while self.__stream_event.is_set():
            t = t + self.__speriod
            self.__read()
            while max( t - time.time(), 0 ): ns_sleep( 1e2 )

    def __plot(self):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self.__name )

        # n_rows = np.ceil( np.sqrt( self.__channelcount ) )
        # n_cols = np.ceil( np.sqrt( self.__channelcount ) )
        n_rows = 8
        n_cols = np.ceil( self.__channelcount / 8 )
        
        ax = []
        nsamps = int( 5.0 / self.__speriod )
        signal = np.zeros( ( nsamps, self.__channelcount ) )
        for i in range( 0, self.__channelcount ):
            ax.append( gui.add_subplot( n_rows, n_cols, i+1 ) )
            
            ax[i].plot( signal[:,i], linewidth = 2 )
            ax[i].set_xticks( [] )

            ax[i].set_ylim( -10, 10 )
            ax[i].set_ylabel( 'Ch.%02d' % (i+1) )
        plt.tight_layout()

        plt.show( block = False )
        while self.__view_event.is_set():
            try:
                data = []
                while self.__plot_buffer.qsize() > 0: data.append( self.__plot_buffer.get() )
                if data:
                    # concatenate to get a block of data
                    data = np.vstack( data )

                    # update generated signal data
                    for i in range( 0, self.__channelcount ):
                        ydata = ax[i].lines[0].get_ydata()
                        ydata = np.append( ydata, data[:,i] )
                        ax[i].lines[0].set_ydata( ydata[-nsamps:] )

                plt.pause( 0.001 )
            except: self.__view_event.clear() 
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
        return self.__name

    @property
    def state(self): 
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last generated sensor readings
        """
        try:
            return self.__state_buffer.get( timeout = 1e-3 )
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
        return self.__speriod

    @property
    def channelcount(self): 
        """
        The channel count for the device

        Returns
        -------
        int
            The number of channels per sensor measurement
        """ 
        return self.__channelcount

    def run(self, display = False):
        """ 
        Starts the acquisition process of the debug device 
        
        Parameters
        ----------
        display : bool
            Flag determining whether sensor measurements should be printed to console (True) or not (False)
        """
        if not self.__stream_event.is_set():
            self.__stream_event.set()
            if display: self.__print_event.set()
            else: self.__print_event.clear()
            
            self.__streamer = mp.Process( target = self.__stream, args = ( int( 1e3 * np.random.rand() ), ) )
            self.__streamer.start()

    def stop(self):
        """
        Stops the acquisition process of the debug device
        """
        if self.__stream_event.is_set():
            self.__stream_event.clear()
            self.__streamer.join()

    def view(self):
        """ 
        Launches the GUI viewer of the debug device data 
        """
        if not self.__view_event.is_set():
            self.__view_event.set()
            self.__viewer = mp.Process( target = self.__plot )
            self.__viewer.start()
            
    def hide(self):
        """
        Closes the GUI viewer of the debug device data
        """
        if self.__view_event.is_set():
            self.__view_event.clear()
            self.__viewer.join()

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
    class_init = inspect.getargspec( DebugDevice.__init__ )
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

    dbg = DebugDevice( num_channels = args.num_channels, srate = args.srate )
    dbg.run( display = False )
    dbg.view()

    while True:
        state = dbg.state
        if state is not None:
            print( state )