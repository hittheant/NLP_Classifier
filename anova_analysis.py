import h5py
import numpy as np
from utils import class_bin, feature_extract, anova_test

# importing dataset
hf = h5py.File('s15data.mat', 'r')
data = hf.get('dat')
data = np.array(data)
y = class_bin(data[8, :], 500)

sampling_rate = 10240

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
ar_features = feature_extract(data[0:6, :], 'dwt', 500, 500)
anova_test(ar_features, y)

ds_data = data[:, ::10]
y_ds = class_bin(ds_data[8, :], 50)
data = []
y = []

print("Downsampled Data:")

# TD5 Featureset
print("TD5 Featureset")
td5_features = feature_extract(ds_data[0:6, :], 'td5', 50, 50)
f, p = anova_test(td5_features, y)

# AR Featureset
print("AR Featureset")
ar_features = feature_extract(ds_data[0:6, :], 'ar', 50, 50)
f, p = anova_test(ar_features, y)

# Fourier Transform Featureset
print("FT Featureset")
ft_features = feature_extract(ds_data[0:6, :], 'ft', 50, 50)
f, p = anova_test(ft_features, y)

# Wavelet Transform Featureset
print("WT Featureset")
dwt_features = feature_extract(ds_data[0:6, :], 'dwt', 50, 50)
f, p = anova_test(dwt_features, y)
