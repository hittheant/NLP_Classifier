import numpy as np
from mite.filters import AbstractBaseFilter

class WaveformLengthFilter(AbstractBaseFilter):
    """ A Python implementation of a waveform length filter """
    def __init__(self):
        """
        Constructor

        Returns
        -------
        obj
            A WaveformLengthFilter object
        """
        pass

    def filter(self, x):
        """
        Compute waveform length of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            Filtered output data
        """
        return np.sum( np.abs( np.diff( x, axis = 0 ) ), axis = 0 )

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 ) - 0.5
    wl = WaveformLengthFilter()
    features = wl.filter( data )
    print( features )