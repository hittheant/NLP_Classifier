# NOTE: On Linux, run this command in terminal
#       sudo ln -s <libpath>/libudev.so.1 <libpath>/libudev.so.0
#       <libpath> = /lib/x86_64-linux-gnu on Ubuntu
#       undo link with -- rm <libpath>/libudev.so.0

import os, sys
modpath = os.path.join(os.path.dirname( os.path.realpath( __file__ ) ), 'external', 'rhd2000')
sys.path.append( modpath )
import rhd2k

import time
import ctypes
import queue

import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import RectBivariateSpline
from scipy import signal

from .. import ns_sleep
from . import AbstractBaseInput

class IntanRHD2000(AbstractBaseInput):
    """ Python implementation of the Intan RHD2000 evaluation board"""
    BUFFER_SIZE = int( 60 )
    SAMPLERATE_MAPPING = { 1000 : rhd2k.Rhd2000EvalBoard.SampleRate1000Hz,
                           1250 : rhd2k.Rhd2000EvalBoard.SampleRate1250Hz, 
                           1500 : rhd2k.Rhd2000EvalBoard.SampleRate1500Hz,
                           2000 : rhd2k.Rhd2000EvalBoard.SampleRate2000Hz,
                           2500 : rhd2k.Rhd2000EvalBoard.SampleRate2500Hz,
                           3000 : rhd2k.Rhd2000EvalBoard.SampleRate3000Hz,
                           3333 : rhd2k.Rhd2000EvalBoard.SampleRate3333Hz,
                           4000 : rhd2k.Rhd2000EvalBoard.SampleRate4000Hz,
                           5000 : rhd2k.Rhd2000EvalBoard.SampleRate5000Hz,
                           6250 : rhd2k.Rhd2000EvalBoard.SampleRate6250Hz,
                           8000 : rhd2k.Rhd2000EvalBoard.SampleRate8000Hz,
                           10000 : rhd2k.Rhd2000EvalBoard.SampleRate10000Hz,
                           12500 : rhd2k.Rhd2000EvalBoard.SampleRate12500Hz,
                           15000 : rhd2k.Rhd2000EvalBoard.SampleRate15000Hz,
                           20000 : rhd2k.Rhd2000EvalBoard.SampleRate20000Hz,
                           25000 : rhd2k.Rhd2000EvalBoard.SampleRate25000Hz,
                           30000 : rhd2k.Rhd2000EvalBoard.SampleRate30000Hz }
    DATAPORT_MAPPING = { 'A1' : rhd2k.Rhd2000EvalBoard.PortA1,
                         'B1' : rhd2k.Rhd2000EvalBoard.PortB1,
                         'C1' : rhd2k.Rhd2000EvalBoard.PortC1,
                         'D1' : rhd2k.Rhd2000EvalBoard.PortD1, 
                         'A2' : rhd2k.Rhd2000EvalBoard.PortA2,
                         'B2' : rhd2k.Rhd2000EvalBoard.PortB2,
                         'C2' : rhd2k.Rhd2000EvalBoard.PortC2,
                         'D2' : rhd2k.Rhd2000EvalBoard.PortD2 }
    ELECTRODE_MAPPING = [ 24, 25, 26, 27, 28, 29, 30, 31, \
                          23, 22, 21, 20, 19, 18, 17, 16, \
                           8,  9, 10, 11, 12, 13, 14, 15, \
                           7,  6,  5,  4,  3,  2,  1,  0 ]

    @staticmethod
    def microvolts( data ): return 0.195 * ( data - 32768 )
    
    def __init__(self, name = 'RHD2000', ports = ['A1','B1','C1','D1'], channels_per_port = 32, \
                 lower_cutoff = 1.0, upper_cutoff = 500.0, dc_offset = 10.0, comb = 60.0, srate = 1250 ):
        """ 
        Constructor

        Parameters
        ----------
        name : str
            The unique identifier for this hardware interface
        ports : iterable of str { 'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2' }
            The hardware ports electrodes are connected to
        channels_per_port : int or iterable of int
            The number of contacts per electrode
        lower_cutoff : float
            The lower cutoff frequency for the in-built bandpass filter
        upper_cutoff : float
            The higher cutoff frequency for the in-built bandpass filter
        dc_offset : float
            The DC offset frequency (10x lower_cutoff for best linearity)
        comb : float
            The base frequency for an IIR digital comb filter (NOT YET IMPLEMENTED)
        srate : int { 1000, 1250, 1500, 2000, 2500, 3000, 3333, 4000, 5000, 6250, 8000, 10000, 12500, 15000, 20000, 25000, 30000 }
            The sampling rate for the hardware interface
        
        Returns
        -------
        obj
            An IntanRHD2000 interface object
        """
        # Four ports: A1, B1, C1, D1
        # can use 8 to 64 electrodes per port (currently using 32)
        # sampling rate: 1250.0 Hz
        # buffer size: 60 samples
        # notch filter: 60 Hz

        # hardware variables
        assert( set( ports ).issubset( ['A1','B1','C1','D1','A2','B2','C2','D2'] ) )
        self._ports = tuple( ports )

        if isinstance( channels_per_port, int ):
            channels = ( ( channels_per_port, ) * len( self._ports ) )
        else:
            assert( len( channels_per_port ) == len( self._ports ) )
            channels = channels_per_port
        self._samplingrate = int( srate )

        self._board = None
        self._rhdqueue = None
        self._amp_channels = channels

        # device variables
        self._name = name
        self._channelcount = int( np.asarray( channels ).sum() )
        self._speriod = 1.0 / srate
        
        self._state = np.zeros( ( IntanRHD2000.BUFFER_SIZE, self._channelcount ) )
        self._state_buffer = mp.Queue()
        self._state_buffer_max = int( srate * 5.0 / IntanRHD2000.BUFFER_SIZE )

        # frequency variables
        self._bandwidth = ( lower_cutoff, upper_cutoff )
        self._dc_offset = dc_offset

        if comb is not None: 
            # assert( notch == 50.0 or notch == 60.0 )
            center = comb / ( srate / 2.0 )
            qfactor = 30
            b, a = signal.iirnotch( center, qfactor )
            zi = np.matlib.repmat( signal.lfilter_zi( b, a ), self._channelcount, 1 ).T
            self._comb = { 'a' : a, 'b' : b, 'z' : zi }
        else: self._comb = None

        # synchronization variables
        self._exit_event = mp.Event()
        self._conn_event = mp.Event()

        # streaming variables
        self._stream_event = mp.Event()
        self._print_event = mp.Event()
        self._streamer = None

        # viewing variables
        self._view_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        # start child process
        self._streamer = mp.Process( target = self._connect )
        self._streamer.start()
        self._conn_event.wait()

    def __del__(self):
        """
        Destructor
        """
        try:
            self._exit_event.set()
            if self._streamer.is_alive(): self.stop()
        except AttributeError: pass
        try:
            if self._viewer.is_alive(): self.hide()
        except AttributeError: pass

    def _connect(self):
        """ 
        Connect and initialize the Intan RHD2000 FPGA
        
        Notes
        -----
        This function is the target of the child polling process
        """
        try:
            curdir = os.getcwd()    # NOTE: Temporary fix for not finding okFrontPanel dynamic lib
            os.chdir(os.path.join(os.path.dirname( os.path.realpath( __file__ ) ), 'external', 'rhd2000', 'lib')) # NOTE: Temporary fix for not finding okFrontPanel dynamic lib
            self._board = rhd2k.Rhd2000EvalBoard()        # create board
            self._board.open()                            # open board
            os.chdir( curdir )      # NOTE: Temporary fix for not finding okFrontPanel dynamic lib
            self._board.uploadFpgaBitfile(os.path.join(os.path.dirname( os.path.realpath( __file__ ) ), 'external', 'rhd2000', 'main.bit'))     # upload main.bit
            self._board.initialize()                      # initialize

            # enable or disable each of the eight available USB data streams (0-7)
            all_ports = ( 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2' )
            for i in range( len( all_ports ) ):
                port = all_ports[i]
                if port in self._ports:
                    self._board.enableDataStream( i, True )
                    self._board.setDataSource( i, IntanRHD2000.DATAPORT_MAPPING[port] )
                else:
                    self._board.enableDataStream( i, False )

            # set sampling rate
            self._board.setSampleRate( IntanRHD2000.SAMPLERATE_MAPPING[ self._samplingrate ] )

            # set the delay for sampling each SPI port based on the length of the cable between the FPGA and the RHD2000 chip (in meters)
            # should be updated after any changes are made to the sampling rate
            for port in self._ports:
                if 'A' in port: self._board.setCableLengthMeters( rhd2k.Rhd2000EvalBoard.PortA, 1.0 )
                elif 'B' in port: self._board.setCableLengthMeters( rhd2k.Rhd2000EvalBoard.PortB, 1.0 )
                elif 'C' in port: self._board.setCableLengthMeters( rhd2k.Rhd2000EvalBoard.PortC, 1.0 )
                else: self._board.setCableLengthMeters( rhd2k.Rhd2000EvalBoard.PortD, 1.0 )

            # initialize data queue for streaming data
            self._rhdqueue = rhd2k.dataqueue.DataQueue()

            # set frequency parameters
            chip_registers = rhd2k.Rhd2000Registers( self._board.getSampleRate() )
            dsp_cutoff_freq = chip_registers.setDspCutoffFreq( self._dc_offset )    # remove DC offset (10x LPF for best linearity)
            chip_registers.setLowerBandwidth( self._bandwidth[0] )                  # amplifier lower bandwidth
            chip_registers.setUpperBandwidth( self._bandwidth[1] )                  # amplifier upper bandwidth

            # initialize and upload command list with ADC calibration to AuxCmd3 RAM Bank 0
            cmd_list = rhd2k.VectorInt()
            cmd_seq_len = chip_registers.createCommandListRegisterConfig( cmd_list, True )
            self._board.uploadCommandList( cmd_list, rhd2k.Rhd2000EvalBoard.AuxCmd3, 0 )
            self._board.selectAuxCommandLength( rhd2k.Rhd2000EvalBoard.AuxCmd3, 0, cmd_seq_len - 1 )

            # upload command list version with no ADC calibration to AuxCmd3 RAM Bank 1
            chip_registers.setFastSettle( False )
            cmd_seq_len = chip_registers.createCommandListRegisterConfig( cmd_list, False )
            self._board.uploadCommandList( cmd_list, rhd2k.Rhd2000EvalBoard.AuxCmd3, 1 )
            self._board.selectAuxCommandLength( rhd2k.Rhd2000EvalBoard.AuxCmd3, 0, cmd_seq_len - 1 )

            # select command bank with ADC calibration to initialize SPI ports
            for port in self._ports:
                if 'A' in port: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortA, rhd2k.Rhd2000EvalBoard.AuxCmd3, 0 )
                elif 'B' in port: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortB, rhd2k.Rhd2000EvalBoard.AuxCmd3, 0 )
                elif 'C' in port: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortC, rhd2k.Rhd2000EvalBoard.AuxCmd3, 0 )
                else: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortD, rhd2k.Rhd2000EvalBoard.AuxCmd3, 0 )

            # since out longest command sequence is BUFFER_SIZE commands, run the SPI interface for BUFFER_SIZE samples
            self._board.setMaxTimeStep( IntanRHD2000.BUFFER_SIZE )
            self._board.setContinuousRunMode( False )

            # Start SPI interface and wait for the run to complete
            self._board.run()
            while self._board.isRunning(): pass

            # remove collected samples from the FIFO queue
            self._board.readDataBlocks( 1, self._rhdqueue )
            self._rhdqueue.pop()
            self._board.setMaxTimeStep( 0 ) # when we say stop, we mean it

            # now that ADC calibration is complete, switch to the command bank with no ADC calibration
            for port in self._ports:
                if 'A' in port: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortA, rhd2k.Rhd2000EvalBoard.AuxCmd3, 1 )
                elif 'B' in port: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortB, rhd2k.Rhd2000EvalBoard.AuxCmd3, 1 )
                elif 'C' in port: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortC, rhd2k.Rhd2000EvalBoard.AuxCmd3, 1 )
                else: self._board.selectAuxCommandBank( rhd2k.Rhd2000EvalBoard.PortD, rhd2k.Rhd2000EvalBoard.AuxCmd3, 1 )

            # indicate which LEDs to turn on
            leds = [ 0 ] * 8
            for i in range( len( all_ports ) ):
                port = all_ports[i]
                if port in self._ports: leds[i] = 1
            self._board.setLedDisplay( leds )

            # indicate that connection is complete
            self._conn_event.set()
            
            # stream data
            while not self._exit_event.is_set():
                if self._stream_event.is_set(): 
                    self._board.setContinuousRunMode( True )
                    self._board.run()
                    
                    self._stream()      # stream when asked

                    self._board.setContinuousRunMode( False )
                    self._board.flush()
                # else:
                #     self._exit_event.set()  # exit if board stops running for whatever reason
        finally:
            # turn off LED
            self._board.setLedDisplay( [ 0 ] * 8 )
            
            # clear buffers so child processes can join
            self.flush()
            empty = False
            while not empty:
                try:
                    self._plot_buffer.get( timeout = 1e-3 )
                except queue.Empty:
                    empty = True
            
            # set connection event so constructor doesn't hang
            self._conn_event.set()

    def _read(self):
        """ 
        Reads a single sample from the Intan RHD2000 FPGA

        Notes
        -----
        While this function does not return anything, it sets the _state variable
        to the last measured sensor readings.
        """
        new_data = self._board.readDataBlocks( 1, self._rhdqueue ) # poll the device
        if new_data:
            front = self._rhdqueue.front()
            amp_prefilter = np.zeros( ( self._channelcount, IntanRHD2000.BUFFER_SIZE ) )
            for i in range( len( self._ports ) ):
                stream = front.amplifierData[i]
                for j in range( self._amp_channels[i] ):
                    channel = stream[j]
                    idx = i * self._amp_channels[i] + j
                    for t in range( IntanRHD2000.BUFFER_SIZE ):
                        amp_prefilter[idx,t] = IntanRHD2000.microvolts( channel[t] )
            self._rhdqueue.pop()
            data = amp_prefilter.T

            # block = self._rhdqueue.front()
            # buff = []
            # for i in range( 0, len( self._ports ) ):
            #     for j in range( 0, self._amp_channels[ i ] ):
            #         val = block.readAmplifier( i, j )
            #         tmp = np.frombuffer( ( ctypes.c_int32 * IntanRHD2000.BUFFER_SIZE ).from_address( \
            #                                ctypes.addressof( val.contents ) ), dtype = np.int32 ).copy()
            #         buff.append( tmp )
            # self._rhdqueue.pop()
            # data = IntanRHD2000.microvolts( np.transpose( np.vstack( buff ) ) )
            
            # apply physical remap

            # apply post-processing kernels (if any)

            # apply digital comb filter (if any)
            # if self._comb is not None:
            #     for i in range( self._channelcount ):
            #         data[:,i], self._comb['z'][:,i] = signal.lfilter( self._comb['b'], self._comb['a'], \
            #                                                             data[:,i], zi = self._comb['z'][:,i] )

            # store data in device state buffer
            while self._state_buffer.qsize() > self._state_buffer_max: 
                self._state_buffer.get( timeout = 1e-3 )
            self._state_buffer.put( data.copy(), timeout = 1e-3 )

            # display data
            if self._print_event.is_set(): print( self._name, ':', data[0,:] )
            if self._view_event.is_set(): self._plot_buffer.put( data )

    def _stream(self):
        """ 
        Streams data from the Intan RHD2000 FPGA at the specified sampling rate
        """
        while self._stream_event.is_set():
            self._read()

    def _plot(self, imshape):
        """
        Generates a visualization of the incoming data in real-time

        Parameters
        ----------
        imshape : tuple (height, width)
            The dimensions of the plotted image. height x width must equal n_channels
        """
        # initialize the figure
        gui, ax = plt.subplots()
        gui.canvas.set_window_title( self._name )
        plt.subplots_adjust( bottom = 0.15 )

        vmin = IntanRHD2000.microvolts( 0 )
        vmax = IntanRHD2000.microvolts( 2 ** 16 - 1 )
        img = ax.imshow( np.zeros( imshape ), cmap = 'jet', vmin = vmin, vmax = vmax )
        plt.axis( 'off' )

        axfilter = plt.axes( [0.25, 0.05, 0.65, 0.03], facecolor = 'lightgoldenrodyellow' )
        sfilt = Slider( axfilter, 'Multiplier', 1.0, 50.0, valinit = 1.0 )

        # stream
        plt.show( block = False )
        x =  np.linspace( 0, imshape[0], imshape[0] )
        y =  np.linspace( 0, imshape[1], imshape[1] )
        while self._view_event.is_set():
            try:
                data = None
                while self._plot_buffer.qsize() > 0: data = self._plot_buffer.get()
                if data is not None:
                    # compute the mean absolute value of the latest block
                    data = np.mean( np.abs( data ), axis = 0 )

                    # remap data for appropriate display
                    # idx1 = 0
                    # for i in range( 0, len( self._ports ) ):
                    #     idx2 = idx1 + self._amp_channels[i]
                    #     data[idx1:idx2] = ( data[idx1:idx2] )[IntanRHD2000.ELECTRODE_MAPPING]
                    #     idx1 = idx2
                    
                    data = np.reshape( data, ( imshape[1], imshape[0] ) ).T     # complicated reshape to match our HDEMG array 
                    # data = data / ( 0.0975 / 2 )                                    # scale between 0 and 1
                    
                    # smooth data for better visualization
                    if sfilt.val != 1.0:
                        interp = RectBivariateSpline( x, y, data )
                        xnew = np.linspace( 0, imshape[0], np.round( sfilt.val * imshape[0] ) )
                        ynew = np.linspace( 0, imshape[1], np.round( sfilt.val * imshape[1] ) )
                        data = interp( xnew, ynew )
                    
                    # update plot
                    img.set_data( data )
                plt.pause( 0.05 )
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
        numpy.ndarray (n_samples, n_channels)
            The last block of measured sensor readings
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
        Starts the acquisition process of the Intan RHD2000 FPGA
        
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
        Stops the acquisition process of the Intan RHD2000 FPGA
        """
        if self._stream_event.is_set():
            self._stream_event.clear()
            self._streamer.join()

    def view(self, imshape = (8, 16) ):
        """ 
        Launches the GUI viewer of the Intan RHD2000 FPGA data

        Parameters
        ----------
        imshape : tuple (height, width)
            The shape of the plotting image 
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot, args = ( imshape, ) )
            self._viewer.start()
            
    def hide(self):
        """ 
        Closes the GUI viewer of the Intan RHD2000 FPGA data
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
    class_init = inspect.getargspec( IntanRHD2000.__init__ )
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

    intan = IntanRHD2000( name = args.name, ports = args.ports, channels_per_port = args.channels_per_port,
                          lower_cutoff = args.lower_cutoff, upper_cutoff = args.upper_cutoff, dc_offset = args.dc_offset,
                          comb = args.comb, srate = args.srate )
    
    intan.run( display = False )
    intan.view()

    while True:
        state = intan.state
        if state is not None:
            print( state[0,:] )