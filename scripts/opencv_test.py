"""Testing for OpenCV.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import

import cv2
import numpy as np

img = cv2.imread('Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# for opencv 3.x and above
# you will need to run the following code to install
# pip install opencv-contrib-python
sift = cv2.xfeatures2d.SIFT_create()
# for raspberry pi
sift = cv2.SIFT()

kp = sift.detect(gray, None)

# for opencv 3.x and above
cv2.drawKeypoints(gray, kp, img)
# for Raspberry Pi
img = cv2.drawKeypoints(gray, kp)

cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
