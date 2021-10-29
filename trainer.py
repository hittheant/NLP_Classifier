import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from mite.models.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from mite.models.SupportVectorMachine import SupportVectorMachine
from mite.models.MultiLayerPerceptron import MultiLayerPerceptron
from argparse import ArgumentParser
from utils import class_bin, feature_extract, butter_bandpass_filter, comb_filter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--data_dir', required=False, type=str, default='./s15data.mat',
                        help='path to mat file')
    parser.add_argument('--model', required=False, type=str, default='mlp',
                        help='type of model to train')
    parser.add_argument('--featureset', required=False, type=str, default='td5',
                        help='featureset to be extracted')
    parser.add_argument('--downsampling', required=False, type=int, default=1,
                        help='level of downsampling')
    parser.add_argument('--window_size', required=False, type=int, default=500,
                        help='window size for feature extraction')
    parser.add_argument('--shift_size', required=False, type=int, default=500,
                        help='step size of windows')
    parser.add_argument('--emg_indices', required=False, type=int, default=6,
                        help='maximum index of EMG data columns')
    parser.add_argument('--force_index', required=False, type=int, default=8,
                        help='force data to be classified')
    args = parser.parse_args()

    hf = h5py.File(args.data_dir, 'r')
    data = hf.get('dat')
    data = np.array(data)
    datalength = (np.shape(data)[1]//args.window_size) * args.window_size
    fs = 10240 / args.downsampling
    data = data[:, :datalength:args.downsampling]
    data[0:args.emg_indices, :] = butter_bandpass_filter(data[0:args.emg_indices, :], 50, fs/2, fs)
    data[0:args.emg_indices, :] = comb_filter(data[0:args.emg_indices, :], fs, f0=(fs / round(fs / 60)))
    y = class_bin(data[args.force_index, :], args.shift_size)
    x = feature_extract(data[0:args.emg_indices, :], args.featureset,
                        args.window_size, args.shift_size)

    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, shuffle=True)

    print("Training model...")
    if args.model == 'lda':
        mdl = LinearDiscriminantAnalysis(Xtrain, ytrain)
    elif args.model == 'svm':
        mdl = SupportVectorMachine(Xtrain, ytrain, regressor=False)
    elif args.model == 'mlp':
        mdl = MultiLayerPerceptron(Xtrain, ytrain, regressor=False)

    yhat = mdl.predict(Xtest)

    conf_mat = confusion_matrix(yhat, ytest)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    print('Overall accuracy: {} %'.format(acc * 100))
    print(conf_mat)
    print("Precision, recall and fscore:")
    print(precision_recall_fscore_support(ytest, yhat))


