"""Testing skimage.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import

import numpy as np
from skimage.io import imread
from skimage.filters import threshold_adaptive

import matplotlib.pyplot as plt

# read image
img = imread("Lenna.png", as_grey=True)

th1 = img > 0.5

th2 = img > threshold_adaptive(img, 11, method="mean", offset=2/255.)

th3 = img > threshold_adaptive(img, 11, method="gaussian", offset=2/255.)


titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
