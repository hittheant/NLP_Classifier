import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mite.models.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from mite.models.SupportVectorMachine import SupportVectorMachine
from mite.models.MultiLayerPerceptron import MultiLayerPerceptron
from utils import class_bin, feature_extract
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def run_trainer(data_dir='./s15data.mat', model='lda', featureset='td5',
                downsampling=1, window_size=500, shift_size=500, emg_indices=6,
                force_index=8):
    hf = h5py.File(data_dir, 'r')
    data = hf.get('dat')
    data = np.array(data)
    datalength = (np.shape(data)[1]//window_size) * window_size
    data = data[:, :datalength:downsampling]
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
    windows = range(200, 1000, 100)
    models = ['lda', 'svm', 'mlp']

    # for model in models:
    model = 'lda'
    print(model)
    accs = []
    for window in windows:
        acc_row = []
        for fset in fsets:
            acc = run_trainer(model=model, featureset=fset,
                              window_size=window, shift_size=window)
            acc_row.append(acc)
        accs.append(acc_row)
    print(accs)
