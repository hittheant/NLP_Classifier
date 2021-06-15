import numpy as np
from scipy.signal import butter, lfilter

from ...filters import AbstractBaseFilter

class LowPassFilter(AbstractBaseFilter):
    """ A Python implementation of a Butterworth lowpass filter """
    def __init__(self, cutoff, fs, order = 5):
        """
        Constructor

        Parameters
        ----------
        cutoff : float
            Cutoff frequency (in Hz)
        fs : float
            Sampling frequency (in Hz)
        order : int
            Butterworth filter order

        Returns
        -------
        obj
            A LowPassFilter object
        """  
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter( order, normal_cutoff, btype = 'lowpass', analog = False )

        self.__butter = ( b, a )

    def filter(self, x):
        """
        Lowpass filters the input data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            data to be filtered

        Returns
        -------
        numpy.ndarray (n_samples, n_channels)
            filtered data
        """
        return lfilter( self.__butter[0], self.__butter[1], x, axis = 0 )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    order = 6
    fs = 30.0
    cutoff = 3.667

    lpf = LowPassFilter(cutoff = cutoff, fs = fs, order = order)
    w, h = freqz( lpf._LowPassFilter__butter[0], lpf._LowPassFilter__butter[1], worN = 8000 )
    plt.subplot( 2, 1, 1 )
    plt.plot( 0.5 * fs * w / np.pi, np.abs( h ), 'b' )
    plt.plot( cutoff, 0.5 * np.sqrt( 2 ), 'ko' )
    plt.axvline( cutoff, color = 'k' )
    plt.xlim( 0, 0.5 * fs )
    plt.title( 'Lowpass Filter Frequency Response' )
    plt.xlabel( 'Frequency (Hz)' )
    plt.grid()

    T = 5.0
    n = int( T * fs )
    t = np.linspace(0, T, n, endpoint = False)
    data = np.sin( 1.2 * 2 * np.pi * t ) + 1.5 * np.cos( 9 * 2 * np.pi * t ) + 0.5 * np.sin( 12.0 * 2 * np.pi * t )
    
    
    y = lpf.filter( data )

    plt.subplot( 2, 1, 2 )
    plt.plot( t, data, 'r-', label = 'data' )
    plt.plot( t, y, 'k-', linewidth = 2, label = 'filtered data')
    plt.xlabel( 'Time (sec)')
    plt.grid()
    plt.legend(frameon = False)

    plt.subplots_adjust( hspace = 0.35 )
    plt.show()