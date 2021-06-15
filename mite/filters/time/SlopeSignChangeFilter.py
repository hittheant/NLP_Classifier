import numpy as np
from .. import AbstractBaseFilter

class SlopeSignChangeFilter(AbstractBaseFilter):
    """ A Python implementation of a slope sign change filter """
    def __init__(self, eps = 1e-6):
        """
        Constructor

        Parameters
        ----------
        eps : float
            Threshold to count a sign-change in the slope

        Returns
        -------
        obj
            A SlopeSignChangeFilter object
        """
        self.__eps = eps

    def filter(self, x):
        """
        Compute slope sign change of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            Filtered output data
        """
        ddt = np.gradient( x, axis = 0 )
        ssc = np.dstack( [ np.abs( ddt[1:,:] ) > self.__eps, 
                           np.abs( ddt[:-1,:] ) > self.__eps, 
                           np.multiply( ddt[1:,:], ddt[:-1,:] ) < 0 ] )
        return np.sum( np.sum( ssc, axis = 2 ) == 3, axis = 0 )

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 ) - 0.5
    ssc = SlopeSignChangeFilter(eps = 1e-6)
    features = ssc.filter( data )
    print( features )