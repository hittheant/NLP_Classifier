import numpy as np
from . import AbstractBaseFilter

class SequentialFilter(AbstractBaseFilter):
    """ A Python implementation of a sequential aggregate filter """
    def __init__(self, filters = []):
        """
        Constructor

        Parameters
        ----------
        filters : iterable of AbstractBaseFilter
            Processing filters to apply in sequential order

        Returns
        -------
        obj
            A SequentialFilter object
        """
        self.__filters = filters

    def filter(self, x):
        """
        Applies filtering functions to input raw data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray
            Filtered output data
        """
        for f in self.__filters: x = f.filter( x )
        return x

if __name__ == '__main__':
    from .time import MeanAbsoluteValueFilter, WaveformLengthFilter
    
    data = np.random.rand( 1000, 8 ) - 0.5
    mav = MeanAbsoluteValueFilter()
    wl = WaveformLengthFilter()

    filt = SequentialFilter( filters = [ mav, wl ] )
    features = filt.filter( data )
    print( features )