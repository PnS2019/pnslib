"""Testing skimage.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import

import numpy as np
from skimage.io import imread
from skimage.filters import threshold_adaptive
from skimage.filters import laplace
from skimage.filters import sobel_h, sobel_v
from skimage.feature import canny

import matplotlib.pyplot as plt

# read image
img = imread("Lenna.png", as_grey=True)

edges = canny(img)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
