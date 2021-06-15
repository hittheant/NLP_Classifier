import socket
import subprocess

from . import AbstractBaseOutput

class MyoTrain(AbstractBaseOutput):
    """ A Python implementation of a MyoTrain interface """
    MOVEMENT_DICTIONARY = { 'rest'         :  0, 'open'         :  1, 'trigger'       :  2,
                            'column'       :  3, 'index'        :  4, 'key'           :  5,
                            'mouse'        :  6, 'park_thumb'   :  7, 'power'         :  8,
                            'pinch_closed' :  9, 'pinch_open'   : 10, 'tripod'        : 11,
                            'tripod_open'  : 12, 'pronate'      : 35, 'supinate'      : 36,
                            'elbow_bend'   : 45, 'elbow_extend' : 46  }

    @staticmethod
    def create_packet( movement, velocity ):
        """
        Constructs a UDP packet following the MyoTrain protocol

        Parameters
        ----------
        
        """
        # movement          --> movement class to perform
        # velocity          --> speed at which to do the movement
        # compound_movement --> if false, returns to rest after move (set 1)
        # active            --> can the hand move (set to 1)
        # max_cue_length    --> how long each cue can last for (set to 15)
        return bytes( [ movement, 0, 0, 0, velocity, 1, 1, 15 ] )
    
    def __init__( self, path = 'MyoTrain_R.exe', ip = '127.0.0.1', port = 9027 ):
        """
        Constructor

        Parameters
        ----------
        path : str
            The path to the MyoTrain executable to launch
        ip : str
            The IP address the MyoTrain executable is running on
        port : int
            The UDP port that the MyoTrain executable is listening on

        Returns
        -------
        obj
            A MyoTrain interface object
        """
        self.__addr = ( ip, port )
        self.__udp = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
        self.__venv = subprocess.Popen( [ path ], shell = False,
                                        stdin = subprocess.DEVNULL,
                                        stdout = subprocess.DEVNULL,
                                        stderr = subprocess.DEVNULL )

    def __del__( self ):
        """
        Destructor

        Closes the MyoTrain executable.
        """
        try: self.__venv.kill()
        except AttributeError: pass

    def publish( self, msg, speed = 1 ):
        """
        Publish output to the MyoTrain virtual environment

        Parameters
        ----------
        msg : str
            The name of the output movement class to send
        speed : float (0, 1)
            The speed at which to complete the movement class
        """
        if self.__venv.poll() is None: # virtual environment is on
            try:
                move = MyoTrain.MOVEMENT_DICTIONARY[ msg ]
                speed = int( 255 * min( [ max( [ 0.0, speed ] ), 1.0 ] ) )
                pkt = MyoTrain.create_packet( move, speed )
                self.__udp.sendto( pkt, self.__addr )
            except KeyError:
                pass

if __name__ == '__main__':
    import inspect
    import argparse

    # helper function for booleans
    def str2bool( v ):
        if v.lower() in [ 'yes', 'true', 't', 'y', '1' ]: return True
        elif v.lower() in [ 'no', 'false', 'n', 'f', '0' ]: return False
        else: raise argparse.ArgumentTypeError( 'Boolean value expected!' )

    # parse commandline entries
    class_init = inspect.getargspec( MyoTrain.__init__ )
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

    vhand = MyoTrain( args.path, args.ip, args.port )
    moves = [ 'rest', 'open', 'trigger', 'column', 'index_point',
              'key', 'mouse', 'park_thumb', 'power', 'pinch_closed',
              'pinch_open', 'tripod', 'tripod_open', 'pronate', 'supinate',
              'elbow_bend', 'elbow_extend' ]
    
    print( '------------ Movement Commands ------------' )
    print( '| 00  -----  REST                         |' )
    print( '| 01  -----  HAND OPEN                    |' )
    print( '| 02  -----  TRIGGER GRASP                |' )
    print( '| 03  -----  COLUMN GRASP                 |' )
    print( '| 04  -----  INDEX_POINT                  |' )
    print( '| 05  -----  KEY GRASP                    |' )
    print( '| 06  -----  MOUSE GRASP                  |' )
    print( '| 07  -----  PARK THUMB                   |' )
    print( '| 08  -----  POWER GRASP                  |' )
    print( '| 09  -----  FINE PINCH CLOSED            |' )
    print( '| 10  -----  FINE PINCH OPEN              |' )
    print( '| 11  -----  TRIPOD PINCH CLOSED          |' )
    print( '| 12  -----  TRIPOD PINCH OPEN            |' )
    print( '| 13  -----  WRIST PRONATION              |' )
    print( '| 14  -----  WRIST SUPINATION             |' )
    print( '| 15  -----  ELBOW BEND                   |' )
    print( '| 16  -----  ELBOW EXTEND                 |' )
    print( '-------------------------------------------' )
    print( '| Press [Q] to quit!                      |' )
    print( '-------------------------------------------' )

    done = False  
    while not done:
        cmd = input( 'Command: ' )
        if cmd.lower() == 'q':
            done = True
        else:
            try:
                idx = int( cmd )
                if idx in range( 0, len( moves ) ):
                    vhand.publish( moves[ idx ] )
            except ValueError:
                pass
    print( 'Bye-bye!' )