import os
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from . import AbstractBaseOutput

class GripCueVisualizer(AbstractBaseOutput):
    """ A Python implementation of a grip cue visualizer """
    def __init__(self):
        """
        Constructor

        Returns
        -------
        obj
            A GripCueVisualizer interface object
        """
        self.__fig = None
        self.__cue_images = {}

        cue_path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'protocols/cues/grip/' )
        for f in os.listdir( cue_path ):
            if f.endswith( '.jpg' ):
                img = mpimg.imread( cue_path + f )
                self.__cue_images.update( { os.path.splitext( f )[0] : img } )    # remove extension from filename to use as key

    def publish( self, msg ):
        """
        Publish commanded grip cue to an image figure

        Parameters
        ----------
        msg : str
            The name of the output movement class to send
        """
        if self.__fig is None or not plt.fignum_exists(self.__fig.number): 
            self.__fig = plt.figure()
            self.__fig.canvas.set_window_title('Cue Visualizer')
            plt.axis( 'off' )
            plt.show( block = False )
        try:
            plt.imshow( self.__cue_images[msg] )
            self.__fig.canvas.draw()
            self.__fig.canvas.flush_events()
            plt.pause( 1e-9 )
        except KeyError:
            raise RuntimeWarning( 'Invalid grip cue!', msg )

if __name__ == '__main__':
    gcv = GripCueVisualizer()
    moves = [ 'rest', 'open', 'power', 'pronate', 'supinate', 'tripod', 'index' ]
    
    print( '----- Movement Commands -----' )
    print( '| 00  -----  REST           |' )
    print( '| 01  -----  OPEN           |' )
    print( '| 02  -----  POWER          |' )
    print( '| 03  -----  PRONATE        |' )
    print( '| 04  -----  SUPINATE       |' )
    print( '| 05  -----  TRIPOD         |' )
    print( '| 06  -----  INDEX POINT    |' )
    print( '-----------------------------' )
    print( '| Press [Q] to quit!        |' )
    print( '-----------------------------' )

    done = False  
    while not done:
        cmd = input( 'Command: ' )
        if cmd.lower() == 'q':
            done = True
        else:
            try:
                idx = int( cmd )
                if idx in range( 0, len( moves ) ):
                    gcv.publish( moves[ idx ] )
            except ValueError:
                pass
    print( 'Bye-bye!' )