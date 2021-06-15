import numpy as np
from scipy.signal import iirnotch, lfilter

from ...filters import AbstractBaseFilter

class NotchFilter(AbstractBaseFilter):
    """ A Python implementation of an IIR notch filter """
    def __init__(self, cutoff, fs, Q = 30):
        """
        Constructor

        Parameters
        ----------
        cutoff : float
            Frequency to remove (in Hz)
        Q : float
            Quality factor that characterizes how narrow the notch is

        Returns
        -------
        obj
            A NotchFilter object
        """  
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = iirnotch( normal_cutoff, Q )

        self.__sampling_freq = fs
        self.__notch = ( b, a )

    def filter(self, x):
        """
        Notch filters the input data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            data to be filtered

        Returns
        -------
        numpy.ndarray (n_samples, n_channels)
            filtered data
        """
        return lfilter( self.__notch[0], self.__notch[1], x, axis = 0 )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    Q = 30
    fs = 30.0
    cutoff = 9.0

    notch = NotchFilter(cutoff = cutoff, Q = Q, fs = fs)
    w, h = freqz( notch._NotchFilter__notch[0], notch._NotchFilter__notch[1], worN = 8000 )
    plt.subplot( 2, 1, 1 )
    plt.plot( 0.5 * fs * w / np.pi, np.abs( h ), 'b' )
    plt.axvline( cutoff, color = 'k' )
    plt.xlim( 0, 0.5 * fs )
    plt.title( 'Notch Filter Frequency Response' )
    plt.xlabel( 'Frequency (Hz)' )
    plt.grid()


    T = 5.0
    n = int( T * fs )
    t = np.linspace(0, T, n, endpoint = False)
    data = np.sin( 1.2 * 2 * np.pi * t ) + 1.5 * np.cos( 9 * 2 * np.pi * t ) + 0.5 * np.sin( 12.0 * 2 * np.pi * t )
    
    y = notch.filter( data )
    plt.subplot( 2, 1, 2 )
    plt.plot( t, data, 'r-', label = 'data' )
    plt.plot( t, y, 'k-', linewidth = 2, label = 'filtered data')
    plt.xlabel( 'Time (sec)')
    plt.grid()
    plt.legend(frameon = False)
    
    plt.subplots_adjust( hspace=0.35 )
    plt.show()