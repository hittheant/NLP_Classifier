import numpy as np
import filterpy.kalman

from .. import AbstractBaseFilter

class KalmanFilter(AbstractBaseFilter):
    def __init__(self, state_transition_matrix, observation_matrix,
                 initial_state = None, process_covariance = None, process_noise = None,
                 observation_noise = None, state_control_matrix = None):
        """
        A Kalman filter optimal estimator

        Parameters
        ----------
        state_transition_matrix : numpy.ndarray (n_state_dims, n_state_dims)
            The state transition matrix, F
        observation_matrix : numpy.ndarray (n_measure_dims, n_state_dims)
            The measure observation matrix, H
        initial_state : numpy.ndarray (n_state_dims,) or None
            The initial state of the Kalman filter internal estimator
        process_covariance : numpy.ndarray (n_state_dims, n_state_dims) or None
            The covariance matrix for the internal process, P
        process_noise : numpy.ndarray (n_state_dims, n_state_dims) or None
            The noise inherent to the internal process, P
        observation_noise : numpy.ndarray (n_measure_dims, n_measure_dims) or None
            The noise inherent to the observation process, Q
        state_control_matrix : numpy.ndarray (n_state_dims, n_control_dims) or None
            The state control matrix for inputs, B
        
        Returns
        -------
        obj
            A KalmanFilter object
        """
        state_dim = state_transition_matrix.shape[0]
        observe_dim = observation_matrix.shape[0]
        
        self.__kf = filterpy.kalman.KalmanFilter(dim_x = state_dim, dim_z = observe_dim)
        
        self.__kf.x = np.zeros( state_dim ) if initial_state is None else initial_state
        self.__kf.P = np.eye( state_dim ) if process_covariance is None else process_covariance
        self.__kf.Q = np.eye( state_dim ) if process_noise is None else process_noise

        self.__kf.H = observation_matrix
        self.__kf.R = np.eye( observe_dim ) if observation_noise is None else observation_noise

        self.__kf.F = state_transition_matrix
        self.__kf.B = 0 if state_control_matrix is None else state_control_matrix

    def filter(self, x):
        """
        Compute the optimal estimator given the input data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_state_dims)
            Input data to filter

        Returns
        -------
        numpy.ndarray (n_samples, n_state_dims)
            Filtered output data
        """
        if len( x.shape ) == 1: x.resize( ( x.shape[0], 1 ) ) # samples x dimensions
        ret = []
        for sample in x:
            sample.resize( 1, sample.shape[0] )
            self.__kf.predict()
            self.__kf.update( z = sample )
            ret.append( self.__kf.x )
        return np.vstack( ret )

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_steps = 1e3
    t = np.linspace( 0, 3 * np.pi, n_steps )
    Ts = t[1] - t[0]

    xtruth = np.sin( t )
    xmeas = xtruth + 0.5 * np.random.randn( int( n_steps ) )
    
    # position-velocity model for a sinusoid
    F = np.array( [ [ 1.0, 1.0 ], [ 0.0, 1.0 ] ] )
    H = np.array( [ [ 1.0, 0.0 ] ] )
    P = np.array( [ [ 1e3, 0 ], [ 0, 1e3 ] ] )
    R = np.array( [ [ 1e2 ] ] )
    Q = np.array( [ [ Ts ** 3 / 3.0, Ts ** 2 / 2.0 ], 
                    [ Ts ** 2 / 2.0, Ts ] ] )

    filt = KalmanFilter( state_transition_matrix = F,
                         observation_matrix = H,
                         process_covariance = P,
                         observation_noise = R,
                         process_noise = Q )
    xfilt = filt.filter( xmeas )

    fig = plt.figure()
    ax = fig.add_subplot( 111 )
    ax.plot( t, xmeas, lw = 2, label = 'observations' )
    ax.plot( t, xfilt[:,0], lw = 2, label = 'estimates' )
    ax.plot( t, xtruth, lw = 3, label = 'truth', color = 'black' )
    
    ax.set_xlabel( 'Time' )
    ax.set_ylabel( 'Process' )
    ax.set_title( 'Kalman Filter' )

    plt.legend( loc = 'lower right' )
    plt.show()