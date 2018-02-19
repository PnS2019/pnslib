"""Demonstrate the usage of Fashion-MNIST.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import matplotlib.pyplot as plt

import pnslib.utils as utils

(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load("full")

print (train_x.shape)
print (train_y.shape)
print (test_x.shape)
print (test_y.shape)

plt.figure()

for idx in range(100):
    plt.imshow(test_x[idx, ..., 0], cmap="gray")
    plt.show()
