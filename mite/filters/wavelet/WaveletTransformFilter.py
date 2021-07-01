import pywt
import numpy as np
from mite.filters import AbstractBaseFilter


class WaveletTransformFilter(AbstractBaseFilter):
    """ A Python implementation of a wavelet transform filter """
    def __init__(self, scales=range(120, 128), wavelet_name='morl'):
        self.scales = scales
        self.wavelet_name = wavelet_name

    def filter(self, x):
        result = []
        coeff, freq = pywt.cwt(x, self.scales, self.wavelet_name, 1)
        result.extend(coeff.flatten())
        return result


if __name__ == '__main__':
    data = np.random.rand( 100, 6 )
    cwt = WaveletTransformFilter()
    features = cwt.filter( data )
    print( np.array(features).shape )