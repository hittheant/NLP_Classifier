import pywt
import numpy as np
from mite.filters import AbstractBaseFilter


class WaveletTransformFilter(AbstractBaseFilter):
    """ A Python implementation of a wavelet transform filter """
    def __init__(self, wavelet_name='db8'):
        self.wavelet_name = wavelet_name

    def filter(self, x):
        result = []
        x = np.transpose(x)
        (cA, cD) = pywt.dwt(x, self.wavelet_name, 1)
        result.extend(cA.flatten())
        result.extend(cD.flatten())
        return result


if __name__ == '__main__':
    data = np.random.rand(100, 6)
    cwt = WaveletTransformFilter()
    features = cwt.filter(data)
    print(np.array(features).shape)