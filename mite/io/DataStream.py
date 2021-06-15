import copy
import time
import collections
import numpy as np
import multiprocessing as mp

from .. import ns_sleep

class DataStream:
    """ A Python implementation of a datastream of various hardware devices """
    def __init__(self, hardware = None):
        """ 
        Constructor

        Parameters
        ----------
        hardware : iterable of AbstractBaseInput
            Hardware devices to stream from

        Returns
        -------
        obj
            A DataStream object

        Raises
        ------
        ValueError
            Hardware passed is not a valid interface
        """
        if not ( isinstance( hardware, collections.Sequence ) and not isinstance( hardware, str ) ):
            raise ValueError( "Pass a sequence of hardware devices!" )

        self.__hardware = hardware
        self.__speriod = np.inf                             # initialize sampling rate
        self.__state = { 'Time' : [] }                      # initialize empty dictionary
        for dev in self.__hardware:
            self.__state.update( { dev.name : [] } )        # add device name as key to dictionary
            if dev.speriod < self.__speriod:            
                self.__speriod = dev.speriod                # update writer's sampling period if greater than device's
        
        self.__recording = mp.Event()   # event set when we are recording
        self.__flushing = mp.Event()    # event set when we are asked to flush the buffer
        self.__queue = mp.Queue()
        self.__recorder = None

    def __record(self):
        """
        Records data from each peice of hardware
        """
        # local dictionary to hold streams
        local_dict = { 'Time' : [] }
        for dev in self.__hardware: 
            local_dict.update( { dev.name : [] } )

        while self.__recording.is_set():
            t = time.perf_counter() + self.__speriod
            local_dict['Time'].append( time.perf_counter() )
            for dev in self.__hardware:
                local_dict[ dev.name ].append( dev.state )
            while max( t - time.perf_counter(), 0 ): ns_sleep( 1e2 )

            # print( self.__flushing.is_set() )
            if self.__flushing.is_set():
                self.__queue.put( copy.deepcopy( local_dict ) )
                local_dict['Time'] = []
                for dev in self.__hardware: local_dict[ dev.name ] =  []
                while self.__flushing.is_set(): ns_sleep( 1e2 ) # wait until flush operation is done
        
        if len( local_dict['Time'] ):
            self.__queue.put( copy.deepcopy( local_dict ) )         # push this in case this is ended before a flush is asked

    def start(self):
        """
        Start recording from each peice of hardware (if not already)
        """
        if not self.__recording.is_set():
            while not self.__queue.empty(): self.__queue.get()      # clear queue, just in case
            for dev in self.__hardware: dev.run( display = False )  # start each device
            self.__recording.set()                                  # signal recording is happening
            self.__recorder = mp.Process( target = self.__record )
            self.__recorder.start()

    def stop(self):
        """
        Stop recording from each peice of hardware (if currently recording)
        """
        if self.__recording.is_set():
            self.__recording.clear()
            # time.sleep( 0.5 )   # need to wait for queue to fill before join
            # self.__recorder.join()
            for dev in self.__hardware: dev.stop()

    # def peek( self ):
    #     output = self.__state.copy()       # copy to keep atomicity
    #     output[ 'Time' ] = np.array( output[ 'Time' ] )
    #     for dev in self.__hardware:
    #         key = dev.name
    #         output[ key ] = np.vstack( output[ key ] )
    #     return output

    def flush(self):
        """
        Flush all data collected since last flush

        Returns
        -------
        dict
            Dictionary containing timestamped hardware samples
            Keys include 'Time' and the device unique id (queried using the 'name' property)
        """
        self.__flushing.set()
        output = self.__queue.get()
        output[ 'Time' ] = np.array( output[ 'Time' ] )
        for dev in self.__hardware:
            output[ dev.name ] = np.vstack( output[ dev.name ] )
        self.__flushing.clear()
        return output

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ..inputs import DebugDevice

    dbg = DebugDevice(srate = 100.0)
    stream = DataStream( hardware = [ dbg ] )
    
    stream.start()
    print( 'Start Recording...', flush = True, end = '' )
    for i in range( 0, 3 ):
        print( '%d...' % ( 3 - i ), flush = True, end = '' )
        ns_sleep( 1e9 )
    stream.stop()
    print( 'Stop Recording!' )

    data = stream.flush()
    print( data['Debug'].shape )
    print( data['Debug'] )

    ax = plt.figure().add_subplot( 111 )
    x = np.arange( data['Debug'].shape[0] )

    for i in range( 0, dbg.channelcount ):
        y = data['Debug'][:,i] + ( 2 * i )
        ax.plot( x, y )
    plt.show()