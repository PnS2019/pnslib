"""Demo for Linear Regression.

Linear Regression on the function y = 3*x+2

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt

import pnslib.utils as utils


# get dataset
def sample_function(x):
    return 3*x+2


train_x, train_y, test_x, test_y = utils.generate_lr_data(
    num_data=1000,
    function=sample_function)

x = Input(shape=(1,))
y = Dense(1)(x)

model = Model(x, y)

model.summary()

model.compile(loss="mean_squared_error",
              optimizer="sgd",
              metrics=["mse"])


num_epochs = 10
batch_size = 16

plt.figure()
plt.plot(train_x, train_y, ".r", test_x, test_y, ".g")
plt.show()
for idx in xrange(num_epochs):
    print ("Training %d epochs" % (idx))
    weight_list = model.get_weights()

    if idx % 2 == 0:
        plt.plot(test_x, test_y, ".g",
                 test_x, test_x*weight_list[0][0]+weight_list[1][0])
        plt.title("Epoch %d" % (idx))
        plt.show()

    for batch_idx in xrange(train_x.shape[0]//batch_size):
        loss = model.train_on_batch(
            x=train_x[batch_idx*batch_size:(batch_idx+1)*batch_size],
            y=train_y[batch_idx*batch_size:(batch_idx+1)*batch_size])
        print ("Training Loss: %.2f" % (loss[0]))

    for batch_idx in xrange(test_x.shape[0]//batch_size):
        loss = model.test_on_batch(
            x=test_x[batch_idx*batch_size:(batch_idx+1)*batch_size],
            y=test_y[batch_idx*batch_size:(batch_idx+1)*batch_size])
        print ("Testing Loss: %.2f" % (loss[0]))
