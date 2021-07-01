import matplotlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mite.filters import ParallelFilter
from mite.filters.time import MeanAbsoluteValueFilter, VarianceFilter, WaveformLengthFilter, SlopeSignChangeFilter, \
    ZeroCrossingsFilter
from mite.filters.wavelet.WaveletTransformFilter import WaveletTransformFilter

matplotlib.use('QT5Agg')


# importing dataset
hf = h5py.File('s15data.mat', 'r')
data = hf.get('dat')
data = np.array(data)

y = data[8, :]
y_means = np.mean(y.reshape(-1, 100), axis=1)
diff = (np.max(y) - np.min(y))/3
bins = np.array([np.min(y), np.min(y) + diff, np.min(y) + 2*diff, np.max(y)])
y_binned = np.digitize(y_means, bins)

ds_data = data[0:10:-1]
sampling_rate = 10240

# processing variables
window_size = 100
window_step = 100

td5 = ParallelFilter(filters=[MeanAbsoluteValueFilter(),
                              VarianceFilter(),
                              WaveformLengthFilter(),
                              SlopeSignChangeFilter(),
                              ZeroCrossingsFilter()])

cwt = WaveletTransformFilter()

window = []
tfeat = []
features = []

sample = 0
t = 0
for sample in tqdm(range(0, 1*sampling_rate, 100)):
    window = data[0:6, sample:sample+100]  # grab device measurements
    features.append(cwt.filter(np.transpose(window)))  # extract features
    t = (sample + 100) / sampling_rate
    tfeat.append(t)  # save timestamp

raw = data[0:sample]
traw = range(sample)
tfeat = np.hstack(tfeat)
features = np.array(features)
features = np.vstack(features)

print('Done!')

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
