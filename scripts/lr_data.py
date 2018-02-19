"""Generate data for Linear Regression.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

import matplotlib.pyplot as plt

import pnslib.utils as utils

train_x, train_y, test_x, test_y = utils.generate_lr_data()

plt.figure()
plt.plot(train_x, train_y, ".r", test_x, test_y, ".g")
plt.show()


# with custom function
def sample_function(x):
    return x**2+x+3


train_x, train_y, test_x, test_y = utils.generate_lr_data(
    function=sample_function)

plt.figure()
plt.plot(train_x, train_y, ".r", test_x, test_y, ".g")
plt.show()
