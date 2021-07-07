import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mite.models.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from argparse import ArgumentParser
from utils import class_bin, feature_extract
from mite.utils.Metrics import confusion_matrix

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='path to mat file')
    parser.add_argument('--save_dir', required=True, type=str,
                        help='path to model save directory')
    parser.add_argument('--model', required=False, type=str, default='lda',
                        help='type of model to train')
    parser.add_argument('--featureset', required=False, type=str, default='td5',
                        help='featureset to be extracted')
    parser.add_argument('--downsampling', required=False, type=int, default=1,
                        help='level of downsampling')
    parser.add_argument('--window_size', required=False, type=int, default=100,
                        help='window size for feature extraction')
    parser.add_argument('--shift_size', required=False, type=int, default=100,
                        help='step size of windows')
    parser.add_argument('--emg_indices', required=False, type=int, default=6,
                        help='maximum index of EMG data columns')
    parser.add_argument('--force_index', required=False, type=int, default=8,
                        help='force data to be classified')
    args = parser.parse_args()

    hf = h5py.File(args.data_dir, 'r')
    data = hf.get('dat')
    data = np.array(data)
    data = data[:, ::args.downsampling]
    y = class_bin(data[args.force_index], args.shift_size)
    x = feature_extract(data[0:args.emg_indices, :], args.featureset,
                        args.window_size, args.shift_size)

    Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, data.target, test_size=0.33)
    mdl = LinearDiscriminantAnalysis(Xtrain, ytrain)
    yhat = mdl.predict(Xtest)

    fig = plt.figure(figsize=(10.0, 5.0))
    cm = confusion_matrix(ytest, yhat, labels=data.target_names, show=False)
    fig.set_title('EMG Dataset Classification')
    plt.tight_layout()
    plt.show()
