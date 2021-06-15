import os
import sys
import copy
import time

import numpy as np
import multiprocessing as mp

import matplotlib
matplotlib.use( 'QT5Agg' )

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import AbstractBaseProtocol

class TargetAchievementControl(AbstractBaseProtocol):
    """ A Python implementation of a 1DOF target achievement control task """
    PLOT_RESOLUTION = 100
    def __init__(self, velocity = 0.2, timeout = 3, dwell = 1, target_range = (0.25, 0.75), eps = 0.05):
        """
        Constructor

        Parameters
        ----------
        velocity : float
            The velocity of the target movement
        timeout : float
            The timeout for the target achievement
        dwell : float
            The amount of continuous time inside target to register a successful task
        target_range : tuple of floats
            The minimum and maximum possible target values
        eps : float
            The width of the target value

        Returns
        -------
        obj
            A TargetAchievementControl protocol object
        """
        self.__velocity = velocity
        
        self.__timeout = timeout
        self.__dwell = dwell

        self.__range = target_range
        self.__threshold = eps
        
        self.__tac = None
        self.__tac_event = mp.Event()
        self.__start_event = mp.Event()

        self.__inqueue = mp.Queue()
        self.__outqueue = mp.Queue()
        self.__results = None

    def __del__(self):
        """
        Destructor

        This cleans up any child processes / resources spawned by this object
        """
        try:
            if self.__tac.is_alive:
                pass
        except AttributeError: pass

    def __protocol(self, n_trials, fileno):
        """
        The logic for this protocol

        Parameters
        ----------
        n_trials : int
            The number of trials to run
        fileno : int
            The filenumber to pipe stdin to (allows printing to console)
        """
        sys.stdin = os.fdopen(fileno)
        np.random.seed( int( time.perf_counter() ) )


        fig = plt.figure( figsize = ( 7.5, 7.5 ) )
        fig.canvas.set_window_title('Target Achievement Control')

        ax = fig.add_subplot( 111, projection = 'polar' )
        ax.grid( False )
        ax.set_xticks( [] )
        ax.set_yticks( [] )
        ax.set_rlim( 0.0, 1.0 )

        theta = np.linspace( 0, 2 * np.pi, TargetAchievementControl.PLOT_RESOLUTION )

        upper_threshold = ax.plot( theta, np.zeros( TargetAchievementControl.PLOT_RESOLUTION ), 
                                   color = 'g', linestyle = '--' )
        lower_threshold = ax.plot( theta, np.zeros( TargetAchievementControl.PLOT_RESOLUTION ), 
                                   color = 'g', linestyle = '--' )
        target_circle = ax.plot( theta, np.zeros( TargetAchievementControl.PLOT_RESOLUTION ), 
                                 color = 'r', linewidth = 4 )

        plt.show( block = False )
        
        results = []
        for i in range( n_trials ):
            target_radius = np.random.uniform( self.__range[0], self.__range[1] )
            upper_threshold[0].set_ydata( ( target_radius + self.__threshold ) * np.ones( TargetAchievementControl.PLOT_RESOLUTION ) )
            lower_threshold[0].set_ydata( ( target_radius - self.__threshold ) * np.ones( TargetAchievementControl.PLOT_RESOLUTION ) )
            
            ax.collections.clear()
            ax.fill_between( theta, target_radius - self.__threshold, target_radius + self.__threshold, color = 'g', alpha = 0.2 )

            current_radius = 0.0
            print( 'Target is %.2f' % target_radius )

            data = { 'time' : [], 'radius' : [], 'movement' : [], 'target' : target_radius, 'success' : False }
            movement = None

            input( 'Press ENTER to begin task...' )# (%02d/%02d)...' % (i+1, n_trials) )
            self.__start_event.set()

            dt = 0
            timeout_timer = time.perf_counter()
            completion_timer = timeout_timer
            while time.perf_counter() < timeout_timer + self.__timeout and not data['success']:
                loop_timer = time.perf_counter()
                while not self.__inqueue.empty(): movement = self.__inqueue.get()   # None -- invalid movement, do nothing
                                                                                    #    0 -- rest
                                                                                    #    1 -- agonist movement
                                                                                    #   -1 -- antagonist movement
                if movement is not None:
                    if movement != 0:  # not predicting rest
                        current_radius += ( movement * self.__velocity * dt )
                        current_radius = np.minimum( 1.0, np.maximum( 0.0, current_radius ) )
                        target_circle[0].set_ydata( current_radius * np.ones( TargetAchievementControl.PLOT_RESOLUTION ) )
                        completion_timer = time.perf_counter()
                    else:           # predicting rest
                        current_velocity = 0.0
                        if current_radius <= ( target_radius + self.__threshold ) and current_radius >= ( target_radius - self.__threshold ):
                            if time.perf_counter() > completion_timer + self.__dwell:
                                data['success'] = True
                else: completion_timer = time.perf_counter()
                
                data['time'].append( time.perf_counter() )
                data['radius'].append( current_radius )
                data['movement'].append( movement )

                plt.pause( 0.001 )
                dt = time.perf_counter() - loop_timer
            
            data['time'] = np.array( data['time'] ) - data['time'][0]
            data['radius'] = np.array( data['radius'] )
            data['movement'] = np.array( data['movement'] )
        
            results.append( data )
        sys.stdin.close()

        self.__outqueue.put( results )
        self.__tac_event.clear()        # done with TAC
        self.__start_event.clear()      # next task is ready to start

    def move(self, movement):
        """
        Send a movement cue to TAC protocol

        Parameters
        ----------
        movement : int
            The desired movement direction for the target {0 (rest), 1 (expand), -1 (shrink)}
        """
        if self.__tac_event.is_set():
            self.__inqueue.put(movement)

    def start(self, n_trials = 1):
        """
        Begin the protocol

        Parameters
        ----------
        n_trials : int
            The number of trials in the protocol
        """
        if not self.__tac_event.is_set():
            self.__tac_event.set()
            fn = sys.stdin.fileno() # get file descriptor for stdin
            self.__tac = mp.Process( target = self.__protocol, args = (n_trials, fn) )
            self.__tac.start()

    @property
    def started(self):
        """
        Checks to see if the protocol is currently running

        Returns
        -------
        bool
            True if protocol subprocess is still running, False else
        """
        return self.__start_event.is_set()

    @property
    def done(self):
        """
        Checks to see if the protocol has been run previously

        Returns
        -------
        bool
            True if the protocol has been run (and results are queued), False else
        """
        return not self.__tac_event.is_set()

    @property
    def results(self):
        """
        Gets the results from all previously run TAC tasks (since last call)

        Returns
        -------
        iterable of dicts
            A list of dictionaries containing the task results from each trial

        Notes
        -----
        A result dictionary contains the following keys:
        'time'      : timestamps for the trial (numpy.ndarray)
        'radius'    : target value at each timestamp (numpy.ndarray)
        'movement'  : movement command at each timestamp (numpy.ndarray)
        'target'    : the desired target value for the trial (float)
        'success'   : whether the task was successful or not (bool)
        """
        while not self.__outqueue.empty(): 
            self.__results = copy.deepcopy( self.__outqueue.get() )
        return self.__results

if __name__ == '__main__':
    tac = TargetAchievementControl()
    tac.start( n_trials = 2 )
    while not tac.done: time.sleep( 0.1 )

    print( tac.results )