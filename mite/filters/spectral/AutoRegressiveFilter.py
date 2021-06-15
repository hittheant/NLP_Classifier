import numpy as np
import spectrum as sp
from .. import AbstractBaseFilter

class AutoRegressiveFilter(AbstractBaseFilter):
    """ A Python implementation of an autoregressive spectral filter """
    def __init__(self, order = 6):
        """ 
        Constructor 
        
        Parameters
        ----------
        order : int
            The order of autoregressive features using the Burg method

        Returns
        -------
        obj
            An AutoRegressiveFilter object
        """
        self.__order = order

    def filter(self, x):
        """ 
        Compute AR coefficients of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels x order,)
            Filtered output data
        """
        n_channels = x.shape[1]
        feat = np.zeros( n_channels * self.__order )
        for chan in range( 0, n_channels ):
            idx1 = chan * self.__order
            idx2 = idx1 + self.__order
            feat[ idx1:idx2 ] = np.abs( sp.arburg( x[ :, chan ], self.__order )[0] )
        return feat

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 )
    ar = AutoRegressiveFilter( order = 6 )
    features = ar.filter( data )
    print( features )