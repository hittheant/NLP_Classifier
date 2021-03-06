import numpy as np
from tqdm import tqdm
from mite.filters.ParallelFilter import ParallelFilter
from mite.filters.time import MeanAbsoluteValueFilter, VarianceFilter, WaveformLengthFilter, SlopeSignChangeFilter, \
    ZeroCrossingsFilter
from mite.filters.wavelet.WaveletTransformFilter import WaveletTransformFilter
from mite.filters.autoregressive.AutoRegressiveFilter import AutoRegressiveFilter
from mite.filters.spectral.FourierTransformFilter import FourierTransformFilter
from sklearn.feature_selection import f_classif
from scipy.signal import butter, sosfilt, sosfreqz, iircomb, lfilter
from sklearn.decomposition import PCA


def class_bin(y, window, n_classes=3, low=2.49, high=2.51):
    y = np.mean(y.reshape(-1, window), axis=1)
    y_bin = np.zeros(np.shape(y))
    if n_classes == 3:
        y_bin[np.logical_and(y > low, y < high)] = 1
        y_bin[y >= high] = 2
    else:
        y_bin[y >= high] = 1
    # bins = np.linspace(np.min(y), np.max(y), num=n_classes)
    return y_bin


def feature_extract(data, featureset, window_size, shift_size):
    if featureset == 'td5':
        data_filter = ParallelFilter(filters=[MeanAbsoluteValueFilter(),
                                      VarianceFilter(),
                                      WaveformLengthFilter(),
                                      SlopeSignChangeFilter(),
                                      ZeroCrossingsFilter()])
    elif featureset == 'ar':
        data_filter = AutoRegressiveFilter()
    elif featureset == 'td5ar':
        data_filter = ParallelFilter(filters=[MeanAbsoluteValueFilter(),
                                              VarianceFilter(),
                                              WaveformLengthFilter(),
                                              SlopeSignChangeFilter(),
                                              ZeroCrossingsFilter(),
                                              AutoRegressiveFilter()])
    elif featureset == 'dwt':
        data_filter = WaveletTransformFilter()
    elif featureset == 'ft':
        data_filter = FourierTransformFilter()
    else:
        return -1

    features = []
    print("Extracting " + featureset + " features:")
    for sample in tqdm(range(0, np.shape(data)[1], shift_size)):
        window = data[:, sample:sample + window_size]  # grab device measurements
        features.append(data_filter.filter(np.transpose(window)))  # extract features

    features = np.vstack(np.array(features))
    '''
    if featureset == 'dwt' or featureset == 'ft':
        pca = PCA(n_components=np.shape(data)[0]*4)
        pca.fit(features)
        features = pca.transform(features)
    '''
    return features


def anova_test(features, y):
    f, p = f_classif(features, y)
    print(f)
    print(p)
    return f, p


def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


def comb_filter(data, fs, f0=60.0, Q=30):
    b, a = iircomb(f0, Q, ftype='notch', fs=fs)
    y = lfilter(b, a, data)
    return y
