import numpy as np
from scipy.signal import butter, lfilter

from ...filters import AbstractBaseFilter

class BandStopFilter(AbstractBaseFilter):
    """ A Python implementation of a Butterworth bandstop filter """
    def __init__(self, low_cutoff, high_cutoff, fs, order = 5):
        """
        Constructor

        Parameters
        ----------
        low_cutoff : float
            Lowerbound cutoff frequency (in Hz)
        high_cutoff : float
            Higherbound cutoff frequency (in Hz)
        fs : float
            Sampling frequency (in Hz)
        order : int
            Butterworth filter order

        Returns
        -------
        obj
            A BandStopFilter object
        """  
        nyq = 0.5 * fs
        lo_normal_cutoff = low_cutoff / nyq
        hi_normal_cutoff = high_cutoff / nyq
        b, a = butter( order, [ lo_normal_cutoff, hi_normal_cutoff], btype = 'bandstop', analog = False )

        self.__sampling_freq = fs
        self.__butter = ( b, a )

    def filter(self, x):
        """
        Bandstop filters the input data

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
    lowcutoff = 8.667
    highcutoff = 9.333

    bsf = BandStopFilter(low_cutoff = lowcutoff, high_cutoff = highcutoff, fs = fs, order = order)
    w, h = freqz( bsf._BandStopFilter__butter[0], bsf._BandStopFilter__butter[1], worN = 8000 )
    plt.subplot( 2, 1, 1 )
    plt.plot( 0.5 * fs * w / np.pi, np.abs( h ), 'b' )
    plt.plot( lowcutoff, 0.5 * np.sqrt( 2 ), 'ko' )
    plt.axvline( lowcutoff, color = 'k' )
    plt.plot( highcutoff, 0.5 * np.sqrt( 2 ), 'ko' )
    plt.axvline( highcutoff, color = 'k' )
    plt.xlim( 0, 0.5 * fs )
    plt.title( 'Bandstop Filter Frequency Response' )
    plt.xlabel( 'Frequency (Hz)' )
    plt.grid()

    T = 5.0
    n = int( T * fs )
    t = np.linspace(0, T, n, endpoint = False)
    data = 0.25 * t + np.sin( 1.2 * 2 * np.pi * t ) + 1.5 * np.cos( 9 * 2 * np.pi * t ) + 0.5 * np.sin( 12.0 * 2 * np.pi * t )
    
    
    y = bsf.filter( data )

    plt.subplot( 2, 1, 2 )
    plt.plot( t, data, 'r-', label = 'data' )
    plt.plot( t, y, 'k-', linewidth = 2, label = 'filtered data')
    plt.xlabel( 'Time (sec)')
    plt.grid()
    plt.legend(frameon = False)

    plt.subplots_adjust( hspace = 0.35 )
    plt.show()