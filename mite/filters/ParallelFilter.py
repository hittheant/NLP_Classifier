import numpy as np
from . import AbstractBaseFilter

class ParallelFilter(AbstractBaseFilter):
    """ A Python implementation of a parallel aggregate filter """
    def __init__(self, filters = []):
        """
        Constructor

        Parameters
        ----------
        filters : iterable of AbstractBaseFilter
            Processing filters to apply in parallel

        Returns
        -------
        obj
            A ParallelFilter object
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
        ret = []
        for f in self.__filters: ret.append( f.filter( x ) )
        return np.hstack( ret )

if __name__ == '__main__':
    from .time import MeanAbsoluteValueFilter, VarianceFilter, WaveformLengthFilter, ZeroCrossingsFilter, SlopeSignChangeFilter
    
    data = np.random.rand( 1000, 8 ) - 0.5
    mav = MeanAbsoluteValueFilter()
    var = VarianceFilter()
    wl = WaveformLengthFilter()
    zc = ZeroCrossingsFilter()
    ssc = SlopeSignChangeFilter()

    filt = ParallelFilter( filters = ( mav, var, wl, zc, ssc ) )
    features = filt.filter( data )
    print( features )