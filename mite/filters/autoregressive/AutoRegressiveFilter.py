from mite.filters import AbstractBaseFilter
from statsmodels.tsa.ar_model import AutoReg
import numpy as np


class AutoRegressiveFilter(AbstractBaseFilter):
    """ A Python implementation of a wavelet transform filter """
    def __init__(self, lags=6):
        self.lags = lags

    def filter(self, x):
        x = np.transpose(x)
        result = []
        for channel in range(len(x)):
            model = AutoReg(x[channel], self.lags, old_names=False)
            model_fit = model.fit()
            result.extend(model_fit.params)
        return result


if __name__ == '__main__':
    data = np.random.rand(100, 6)
    ar = AutoRegressiveFilter()
    features = ar.filter(data)
    print(np.array(features).shape)