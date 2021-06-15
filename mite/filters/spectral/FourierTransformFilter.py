import numpy as np
from mite.filters import AbstractBaseFilter

class FourierTransformFilter(AbstractBaseFilter):
    """ A Python implementation of a fast Fourier transform filter """
    def __init__(self, fftlen = 256):
        """ 
        Constructor 

        Parameters
        ----------
        fftlen : int
            The length of the fast Fourier transform window (should be a power of 2)

        Returns
        -------
        obj
            A FourierTransformFilter object

        Notes
        -----
        This filter only returns the first half of the magnitude coefficients.
        This is because for a real-valued input signal, the magnitude coefficients are symmetric.
        """
        self.__fftlen = fftlen

    def filter(self, x):
        """ 
        Compute FFT magnitude coefficients of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels x fftlen / 2,)
            Filtered output data
        """
        feat = np.fft.fft( x, n = self.__fftlen, axis = 0 )
        return np.abs( feat[0:int(self.__fftlen/2.0)] ).flatten()


if __name__ == '__main__':
    data = np.random.rand( 1024, 8 )
    fft = FourierTransformFilter(fftlen = 256)
    features = fft.filter( data )
    print( features.shape )