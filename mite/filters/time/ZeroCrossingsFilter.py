import numpy as np
from mite.filters import AbstractBaseFilter

class ZeroCrossingsFilter(AbstractBaseFilter):
    """ A Python implementation of a zero crossings filter """
    def __init__(self, eps = 1e-6):
        """  
        Constructor

        Parameters
        ----------
        eps : float
            Threshold to count a zero-crossing

        Returns
        -------
        A ZeroCrossingsFilter object
        """
        self.__eps = eps

    def filter(self, x):
        """
        Compute zero crossings of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            Filtered output data
        """
        zc = np.dstack( [ np.abs( x[1:,:] ) > self.__eps, 
                          np.abs( x[:-1,:] ) > self.__eps, 
                          np.multiply( x[1:,:], x[:-1,:] ) < 0 ] )
        return np.sum( np.sum( zc, axis = 2 ) == 3, axis = 0 )

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 ) - 0.5
    zc = ZeroCrossingsFilter(eps = 1e-6)
    features = zc.filter( data )
    print( features )