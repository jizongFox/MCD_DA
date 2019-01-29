from scipy.io import loadmat
import numpy as np
from ..utils.utils import dense_to_one_hot
import sys

sys.path.insert(0, 'MCD_DA/classification/data')


def load_svhn():
    svhn_train = loadmat('MCD_DA/classification/data/train_32x32.mat')
    svhn_test = loadmat('MCD_DA/classification/data/test_32x32.mat')
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test
