"""Testing for OpenCV.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Lenna.png")

kernel = np.ones((11, 11), np.float32)/121
dst = cv2.filter2D(img, -1, kernel)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
