import matplotlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mite.filters import ParallelFilter
from mite.filters.time import MeanAbsoluteValueFilter, VarianceFilter, WaveformLengthFilter, SlopeSignChangeFilter, \
    ZeroCrossingsFilter
from mite.filters.wavelet.WaveletTransformFilter import WaveletTransformFilter
from mite.filters.autoregressive.AutoRegressiveFilter import AutoRegressiveFilter
from mite.filters.spectral.FourierTransformFilter import FourierTransformFilter
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

# importing dataset
hf = h5py.File('s15data.mat', 'r')
data = hf.get('dat')
data = np.array(data)

outfile = open("output.txt", "a")

y = data[8, :]
y_means = np.mean(y.reshape(-1, 100), axis=1)
diff = (np.max(y) - np.min(y))/3
bins = np.array([np.min(y), np.min(y) + diff, np.min(y) + 2*diff, np.max(y)])
y_binned = np.digitize(y_means, bins)
y = y_binned
y_means = []
y_binned = []

sampling_rate = 10240

td5 = ParallelFilter(filters=[MeanAbsoluteValueFilter(),
                              VarianceFilter(),
                              WaveformLengthFilter(),
                              SlopeSignChangeFilter(),
                              ZeroCrossingsFilter()])
ar = AutoRegressiveFilter()
cwt = WaveletTransformFilter()
ft = FourierTransformFilter()

window = []
td5_features = []
ar_features = []
ft_features = []
cwt_features = []

for sample in tqdm(range(0, np.shape(data)[1], 100)):
    window = data[0:6, sample:sample+100]  # grab device measurements
    td5_features.append(td5.filter(np.transpose(window)))  # extract features
    ar_features.append(ar.filter(np.transpose(window)))  # extract features
    ft_features.append(ft.filter(np.transpose(window)))  # extract features

# TD5 Featureset
td5_features = np.vstack(np.array(td5_features))
f, p = f_classif(td5_features, y)
outfile.write("TD5 Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

# AR Featureset
ar_features = np.vstack(np.array(ar_features))
f, p = f_classif(ar_features, y)
outfile.write("AR Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

# Fourier Transform Featureset
ft_features = np.vstack(np.array(ft_features))
pca = PCA(n_components=16)
pca.fit(ft_features)
ft_features = pca.transform(ft_features)

f, p = f_classif(ft_features, y)
outfile.write("Fourier Transform Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

td5_features = []
ar_features = []
ft_features = []
cwt_features = []
pca = []

for sample in tqdm(range(0, np.shape(data)[1], 100)):
    window = data[0:6, sample:sample+100]
    cwt_features.append(cwt.filter(np.transpose(window)))  # extract features

# Wavelet Transform Featureset
cwt_features = np.vstack(np.array(cwt_features))
pca = PCA(n_components=16)
pca.fit(cwt_features)
cwt_features = pca.transform(cwt_features)

f, p = f_classif(cwt_features, y)
outfile.write("Wavelet Transform Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

y_ds = data[8, ::10]
y_ds_means = np.mean(y_ds.reshape(-1, 50), axis=1)
diff = (np.max(y_ds) - np.min(y_ds))/3
bins = np.array([np.min(y_ds), np.min(y_ds) + diff, np.min(y_ds) + 2*diff, np.max(y_ds)])
y_ds_binned = np.digitize(y_ds_means, bins)
y_ds = y_ds_binned
ds_data = data[::10]

data = []
y = []
y_ds_means = []
y_ds_binned = []

outfile.write("Downsampled Data:")
window = []

for sample in tqdm(range(0, np.shape(ds_data)[1], 50)):
    window = ds_data[0:6, sample:sample+50]  # grab device measurements
    td5_features.append(td5.filter(np.transpose(window)))  # extract features
    ar_features.append(ar.filter(np.transpose(window)))  # extract features
    ft_features.append(ft.filter(np.transpose(window)))  # extract features

# TD5 Featureset
td5_features = np.vstack(np.array(td5_features))
f, p = f_classif(td5_features, y_ds)
outfile.write("TD5 Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

# AR Featureset
ar_features = np.vstack(np.array(ar_features))

f, p = f_classif(ar_features, y_ds)
outfile.write("AR Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

# Fourier Transform Featureset
ft_features = np.vstack(np.array(ft_features))
pca = PCA(n_components=16)
pca.fit(ft_features)
ft_features = pca.transform(ft_features)

f, p = f_classif(ft_features, y_ds)
outfile.write("Fourier Transform Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

td5_features = []
ar_features = []
ft_features = []
cwt_features = []
pca = []

for sample in tqdm(range(0, np.shape(ds_data)[1], 50)):
    window = ds_data[0:6, sample:sample+50]  # grab device measurements
    cwt_features.append(cwt.filter(np.transpose(window)))  # extract features

# Wavelet Transform Featureset
cwt_features = np.vstack(np.array(cwt_features))
pca = PCA(n_components=16)
pca.fit(cwt_features)
cwt_features = pca.transform(cwt_features)

f, p = f_classif(cwt_features, y_ds)
outfile.write("Wavelet Transform Featureset")
for i in range(len(f)):
    outfile.write('Feature %d: %f %f' % (i, f[i], p[i]))

outfile.close()
'''
fig = plt.figure()
for i in range(7):
    ax = fig.add_subplot(7, 1, i + 1)
    ax.plot(tfeat, features[:, i])

    # ax.set_ylim( 0.35, 0.65 )
    ax.set_ylabel('Ch %02d' % (i + 1))

    if i == 0:
        ax.set_title('Mean Absolute Value')
    elif i == 6:
        ax.set_xlabel('Feature Number')
    else:
        ax.set_xticks([])
plt.show()
'''
