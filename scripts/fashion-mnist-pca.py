"""Perform PCA on Fashion-MNIST.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import numpy as np

from pnslib import utils
from pnslib import ml

# load data
(train_x, train_y, test_x, test_y) = utils.binary_fashion_mnist_load(
    flatten=True)

train_x = train_x.astype("float32")/255.
mean_train_x = np.mean(train_x, axis=0)
train_x -= mean_train_x
test_x = train_x.astype("float32")/255.
test_x -= mean_train_x

print (train_x.dtype)
print (test_x.dtype)

# perform PCA on training dataset
train_x, R, n_retained = ml.pca(train_x)

# perform PCA on testing dataset
test_x = ml.pca_fit(test_x, R, n_retained)

print (train_x.shape)
print (test_x.shape)
