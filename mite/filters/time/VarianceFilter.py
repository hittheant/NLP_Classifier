import numpy as np
from mite.filters import AbstractBaseFilter

class VarianceFilter(AbstractBaseFilter):
    """ A Python implementation of a variance filter """
    def __init__(self):
        """
        Constructor

        Returns
        -------
        obj
            A VarianceFilter object
        """
        pass

    def filter(self, x):
        """
        Compute variance of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            Filtered output data
        """
        return np.var( x, axis = 0 )

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 ) - 0.5
    var = VarianceFilter()
    features = var.filter( data )
    print( features )