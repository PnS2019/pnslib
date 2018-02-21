"""Demo for Logistic Regression.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np

import pnslib.utils as utils


# get dataset
def sample_function(x):
    return 3*x+2


train_x, train_y, test_x, test_y = utils.generate_binary_data(
        num_data=5000)
test_x_c1 = test_x[test_y[:, 0] == 0]
test_x_c2 = test_x[test_y[:, 0] == 1]

x = Input(shape=(2,))
y = Dense(1, activation="sigmoid")(x)

model = Model(x, y)

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


num_epochs = 100
batch_size = 128

plt.figure()
plt.plot(
    test_x_c1[:, 0], test_x_c1[:, 1], ".r",
    test_x_c2[:, 0], test_x_c2[:, 1], ".g")
plt.show()

boundary_x = np.linspace(-5, 10)
for idx in xrange(num_epochs):
    print ("Training %d epochs" % (idx))

    if idx % 20 == 0 or idx == num_epochs-1:
        weight_list = model.get_weights()
        a = -weight_list[0][0]/weight_list[0][1]
        boundary_y = a*boundary_x - (weight_list[1][0]/weight_list[0][1])
        plt.plot(
            test_x_c1[:, 0], test_x_c1[:, 1], ".r",
            test_x_c2[:, 0], test_x_c2[:, 1], ".g",
            boundary_x, boundary_y)

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
