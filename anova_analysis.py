import h5py
import numpy as np
from utils import class_bin, feature_extract, anova_test, butter_bandpass_filter, comb_filter

# importing dataset
hf = h5py.File('s15data.mat', 'r')
raw_data = hf.get('dat')
raw_data = np.array(raw_data)
fs = 10240

data = raw_data
data[0:6, :] = butter_bandpass_filter(raw_data[0:6, :], 50, (fs - 1)/2, fs)
data[0:6, :] = comb_filter(data[0:6, :], fs, f0=(fs/round(fs/60)))
y = class_bin(raw_data[8, :], 500)

# TD5 Featureset
print("TD5 Featureset")
td5_features = feature_extract(data[0:6, :], 'td5', 500, 500)
f, p = anova_test(td5_features, y)

# AR Featureset
print("AR Featureset")
ar_features = feature_extract(data[0:6, :], 'ar', 500, 500)
f, p = anova_test(ar_features, y)

# Fourier Transform Featureset
print("FT Featureset")
ft_features = feature_extract(data[0:6, :], 'ft', 500, 500)
anova_test(ft_features, y)

# Wavelet Transform Featureset
print("WT Featureset")
dwt_features = feature_extract(data[0:6, :], 'dwt', 500, 500)
anova_test(dwt_features, y)

fs_ds = fs / 10
ds_data = raw_data[:, ::10]
ds_data[0:6, :] = butter_bandpass_filter(raw_data[0:6, :], 50, (fs_ds - 1)/2, fs_ds)
data[0:6, :] = comb_filter(data[0:6, :], fs_ds, f0=(fs_ds/round(fs_ds/60)))
y_ds = class_bin(ds_data[8, :], 50)
data = []
y = []

print("Downsampled Data:")

# TD5 Featureset
print("TD5 Featureset")
td5_features = feature_extract(ds_data[0:6, :], 'td5', 50, 50)
f, p = anova_test(td5_features, y_ds)

# AR Featureset
print("AR Featureset")
ar_features = feature_extract(ds_data[0:6, :], 'ar', 50, 50)
f, p = anova_test(ar_features, y_ds)

# Fourier Transform Featureset
print("FT Featureset")
ft_features = feature_extract(ds_data[0:6, :], 'ft', 50, 50)
f, p = anova_test(ft_features, y_ds)

# Wavelet Transform Featureset
print("WT Featureset")
dwt_features = feature_extract(ds_data[0:6, :], 'dwt', 50, 50)
f, p = anova_test(dwt_features, y_ds)
