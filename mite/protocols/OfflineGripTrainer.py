import matplotlib.image as mpimg

import copy
import random
import os.path

from ..io import DataStream
from . import AbstractBaseProtocol

class OfflineGripTrainer(AbstractBaseProtocol):
    """ A Python implementation of a protocol to collect grip data in an offline fashion """
    def __init__(self, num_trials = 3, delay = 2, duration = 3, cues = [ 'rest', 'open', 'power', 'pronate', 'supinate', 'tripod', 'index' ], hardware = None):
        """
        Constructor

        Parameters
        ----------
        num_trials : int
            The number of trials for each grip cue
        delay : float
            The delay between cue presentation and the start of data acquisition (in sec)
        duration : float
            The amount of time data is collected for (in sec)
        cues : iterable of str
            The grip cues that will be presented {'rest', 'open', 'power', 'pronate', 'supinate', 'tripod', 'index'}
        hardware : iterable of AbstractBaseInput
            The hardware interfaces to collect data from

        Returns
        -------
        obj
            An OfflineGripTrainer protocol object

        Raises
        ------
        RuntimeError
            Grip cue photos cannot be found
            No collection hardware was specified
        """
        self.__num_trials = num_trials
        self.__duration = duration
        self.__delay = delay

        if cues:
            self.__cues = cues
            self.__cue_images = {}
            for cue in self.__cues:
                cue_path = os.path.join(os.path.dirname( os.path.abspath( __file__ ) ), 'cues/grip/')
                img = mpimg.imread( cue_path + cue + '.jpg' )
                self.__cue_images.update( { cue : img } )
        else: raise RuntimeError( 'No cues specified!' )

        if hardware:
            self.__hardware = hardware 
            self.__stream = DataStream(hardware)
        else: raise RuntimeError( 'No hardware connected!' )

        self.__trainer = None
        self.__queue = mp.Queue()

    def __train(self):
        """
        Specifies the protocol logic
        """
        data = []
        cue_fig = plt.figure()
        plt.ion()
        self.__stream.start()
        for trial in range( 0, self.__num_trials ):
            random.shuffle( self.__cues )
            data.append( {} )                                       # append an empty dictionary
            for cue in self.__cues:                                 # randomized cues
                plt.imshow( self.__cue_images[ cue ] )              # show cue
                plt.axis( 'off' )
                plt.show( block = False )
                plt.pause( self.__delay )                           # wait
                # self.__stream.start()                               # start recording
                self.__stream.flush()

                t = time.time() + self.__duration
                while max( t - time.time(), 0 ): ns_sleep( 1e6 )
                # self.__stream.stop()                                # stop recording
                data[ trial ].update( { cue : self.__stream.flush() } )
        plt.close( cue_fig )
        self.__stream.stop()
        self.__queue.put( copy.deepcopy( data ) )

    def view(self):
        """
        Launch the GUI viewers for each connected hardware interface
        """
        for hw in self.__hardware: hw.view()
    
    def hide(self):
        """
        Hide the GUI viewers for each conected hardware interface
        """
        for hw in self.__hardware: hw.hide()

    def start(self):
        """
        Start the grip protocol

        Notes
        -----
        This launches a child process to run the protocol logic in
        """
        self.__trainer = mp.Process( target = self.__train )
        self.__trainer.start()
        while self.__queue.empty(): pass # time.sleep( 0.001 ) # ns_sleep( 1e3 )
        # time.sleep( self.__num_trials * len( self.__cues ) * ( self.__duration + self.__delay ) + 0.5 )
        data = self.__queue.get()
        self.__trainer.join() # wait for process to finish
        return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ..inputs.DebugDevice import *
    
    # interface
    dbg = DebugDevice( num_channels = 1, srate = 100.0 )
    cues = [ 'rest', 'open', 'power', 'pronate', 'supinate', 'tripod' ]

    trainer = OfflineTrainer( num_trials = 1, delay = 1, duration = 3, cues = cues, hardware = [ dbg ] )
    data = trainer.start()

    fig = plt.figure()
    for i in range( len( cues ) ):
        ax = fig.add_subplot( len( cues ), 1, i+1 )
        ax.plot( data[0][cues[i]]['Time'] - data[0][cues[i]]['Time'][0], data[0][cues[i]]['Debug'] )
        ax.set_ylabel( cues[i] )
        if i == ( len( cues ) - 1 ): ax.set_xlabel( 'Time (Sec)' )
        else: ax.set_xticks( [] )
        if i == 0: ax.set_title( 'Offline Training Data' )
    plt.show()