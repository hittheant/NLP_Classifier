import pywt
import numpy as np


def cwt_transform(train_size, train_signal, scales = range(1,128), waveletname = 'morl'):
    train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))
    for ii in range(0, train_size):
        if ii % 1000 == 0:
            print(ii)
        for jj in range(0, 9):
            signal = train_signal[ii, :, jj]
            coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
            coeff_ = coeff[:, :127]
            train_data_cwt[ii, :, :, jj] = coeff_
