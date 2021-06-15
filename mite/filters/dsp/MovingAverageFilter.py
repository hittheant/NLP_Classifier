import numpy as np
from .. import AbstractBaseFilter

class MovingAverageFilter(AbstractBaseFilter):
    def __init__(self, winsize):
        """ 
        Constructor

        Parameters
        ----------
        winsize : int
            The size of the averaging windows (in samples)

        Returns
        -------
        A MovingAverageFilter object 

        Notes
        -----
        This is a causal filter
        """
        self.__winsize = winsize
        self.__buffer = []

    def filter(self, x):
        """ 
        Computes the moving average of the raw input data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            The data to average

        Returns
        -------
        numpy.ndarray (n_samples, n_channels)
            The averaged data
        """
        ret = []
        for sample in x:
            self.__buffer.append( sample )
            if len( self.__buffer ) == self.__winsize:
                window = np.vstack( self.__buffer[-self.__winsize:] )
                ret.append( np.mean( window, axis = 0 ) )
                self.__buffer.pop( 0 )
            else: ret.append( sample )
        return np.vstack( ret )

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = np.linspace( 0, 1, 1000 )
    data = np.sin( 2 * np.pi * t ) + 0.5 * np.random.rand( 1000 )

    avg = MovingAverageFilter( winsize = 10 )
    features = avg.filter( data )
    
    fig = plt.figure()
    ax = fig.add_subplot( 111 )

    ax.plot( t, data, 'r' )
    ax.plot( t, features, 'k' )
    
    ax.set_title( 'Moving Average Filter' )
    ax.set_xlabel( 'Time (Sec)' )
    ax.set_ylabel( 'Measurements' )

    plt.show()