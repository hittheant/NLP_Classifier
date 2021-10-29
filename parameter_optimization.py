import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mite.models.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from mite.models.SupportVectorMachine import SupportVectorMachine
from mite.models.MultiLayerPerceptron import MultiLayerPerceptron
from utils import class_bin, feature_extract, butter_bandpass_filter, comb_filter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def run_trainer(data_dir='./s15data.mat', model='lda', featureset='td5',
                downsampling=1, window_size=500, shift_size=500, emg_indices=6,
                force_index=8, fs=10240):
    hf = h5py.File(data_dir, 'r')
    data = hf.get('dat')
    data = np.array(data)
    datalength = (np.shape(data)[1]//(window_size * downsampling)) * \
                 (window_size * downsampling)
    data = data[:, :datalength:downsampling]
    fs = fs / downsampling
    data[0:emg_indices, :] = butter_bandpass_filter(data[0:emg_indices, :], 50, (fs - 1) / 2, fs)
    data[0:emg_indices, :] = comb_filter(data[0:emg_indices, :], fs, f0=(fs / round(fs / 60)))
    y = class_bin(data[force_index, :], shift_size)
    x = feature_extract(data[0:emg_indices, :], featureset, window_size, shift_size)

    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, shuffle=True)

    print("Training model...")
    if model == 'lda':
        mdl = LinearDiscriminantAnalysis(Xtrain, ytrain)
    elif model == 'svm':
        mdl = SupportVectorMachine(Xtrain, ytrain, regressor=False)
    elif model == 'mlp':
        mdl = MultiLayerPerceptron(Xtrain, ytrain, regressor=False)

    yhat = mdl.predict(Xtest)

    conf_mat = confusion_matrix(yhat, ytest)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    # p, r, f = precision_recall_fscore_support(ytest, yhat)
    return acc


if __name__ == '__main__':
    fsets = ['td5', 'ar', 'td5ar', 'ft', 'dwt']
    windows = range(80, 500, 10)
    models = ['lda', 'mlp']
    downsampling = [1, 10]

    # for model in models:
    for model in models:
        print(model)
        accs = []
        for sampling in downsampling:
            for window in tqdm(windows):
                acc_row = []
                for fset in fsets:
                    acc = run_trainer(model=model, featureset=fset,
                                      window_size=window, shift_size=window,
                                      downsampling=sampling)
                    acc_row.append(acc)
                accs.append(acc_row)
            np.save(model + str(sampling) + '_accs.npy', accs)
