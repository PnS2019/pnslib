"""Generate data for Binary Classification.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

import matplotlib.pyplot as plt

import pnslib.utils as utils

train_x, train_y, test_x, test_y = utils.generate_binary_data()

# class_1
train_x_c1 = train_x[train_y[:, 0] == 0]
train_x_c2 = train_x[train_y[:, 0] == 1]

plt.figure()
plt.plot(
    train_x_c1[:, 0], train_x_c1[:, 1], ".r",
    train_x_c2[:, 0], train_x_c2[:, 1], ".g",
    test_x[:, 0], test_x[:, 1], ".b")
plt.show()
