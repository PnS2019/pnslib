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

x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))
boundary_x = np.c_[xx.ravel(), yy.ravel()]
print (np.c_[xx.ravel(), yy.ravel()].shape)
for idx in xrange(num_epochs):
    print ("Training %d epochs" % (idx))

    if idx % 20 == 0 or idx == num_epochs-1:
        boundary_y = model.predict(boundary_x)
        boundary_y = boundary_y.reshape(xx.shape)
        weight_list = model.get_weights()
        plt.plot(
            test_x_c1[:, 0], test_x_c1[:, 1], ".r",
            test_x_c2[:, 0], test_x_c2[:, 1], ".g")
        plt.contour(xx, yy, boundary_y, cmap=plt.cm.Paired)
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
