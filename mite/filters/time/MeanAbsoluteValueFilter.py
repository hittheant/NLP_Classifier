import numpy as np
from mite.filters import AbstractBaseFilter

class MeanAbsoluteValueFilter(AbstractBaseFilter):
    """ A Python implementation of a mean absolute value filter """
    def __init__(self):
        """
        Constructor

        Returns
        -------
        obj
            A MeanAbsoluteValueFilter object
        """
        pass

    def filter(self, x):
        """
        Compute mean absolute value of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            Filtered output data
        """
        return np.mean( np.abs( x ), axis = 0 )

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 ) - 0.5
    mav = MeanAbsoluteValueFilter()
    features = mav.filter( data )
    print( features )