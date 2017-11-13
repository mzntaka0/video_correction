# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import time

import cv2
import imutils
from bpdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from controllers.face_detection import black_mask





image_path = 'storage/image/gaikoku.jpg'
img = cv2.imread(image_path)
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.set_xlabel('Hue')
#ax.set_ylabel('Saturation')
#ax.set_zlabel('Value')
#ax.plot(img[:, :, 0], img[:, :, 1], img[:, :, 2], 'o')
#plt.show()
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('img', img)
masked_face = black_mask(img)
masked_face = cv2.cvtColor(masked_face, cv2.COLOR_RGB2GRAY)
masked_pixels = img_hsv[np.where(masked_face==0)[0], np.where(masked_face==0)[1]]
masked_pixels_bgr = img[np.where(masked_face==0)[0], np.where(masked_face==0)[1]]
masked_pixels_gray = gray[np.where(masked_face==0)[0], np.where(masked_face==0)[1]]
print(np.histogram(masked_pixels_gray, bins=64))
x = np.histogram(masked_pixels_gray, bins=64)
plt.bar(x[1][:-1], x[0])
plt.show()
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Value')
ax.plot(masked_pixels_bgr[:, 0], masked_pixels_bgr[:, 1], masked_pixels_bgr[:, 2], 'o')
plt.show()
print(np.histogram(masked_pixels_gray, bins=32))
#fig = plt.figure()
#ax = fig.add_subplot(111)

#H = ax.hist2d(masked_pixels[:, 0], masked_pixels[:, 1], bins=128)
#fig.colorbar(H[3], ax=ax)
#plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Value')
ax.plot(masked_pixels[:, 0], masked_pixels[:, 1], masked_pixels[:, 2], 'o')
plt.show()
masked_face = np.where(masked_face <= 0, 255, 0)
cv2.imwrite('mask.png', masked_face)
mask = cv2.imread('mask.png', 0)
bgr = cv2.split(img)
bgra = cv2.merge(bgr + [mask])
bgra = cv2.LUT(bgra, make_tone_curve(zero_one_sigmoid))
gray_bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_bgra', gray_bgra)

print(masked_face)
cv2.imshow('masked', masked_face)


lut_img = cv2.LUT(img, make_tone_curve(zero_one_sigmoid))
#cv2.imshow('img_lut', lut_img)
lut_inv_img = cv2.LUT(lut_img, make_tone_curve(inv_zero_one_sigmoid))
#masked_face = lut_img[np.where(lut_inv_img == [255, 0, 0])]
#print(masked_face)
#masked_face = cv2.cvtColor(masked_face, cv2.COLOR_BGR2GRAY)

#print(np.histogram(masked_face, bins=64))
#print(masked_face)

lut_inv_img = cv2.cvtColor(lut_inv_img, cv2.COLOR_RGB2BGR)
#cv2.imshow('inv_lut', lut_inv_img)

set_trace()


y = gamma_lookuptable(2.2)
print(len(y))
plt.scatter(np.arange(len(y)), y)
plt.show()

s = time.time()
img = black_mask(img)
e = time.time()
print(e - s)


if __name__ == '__main__':
    pass

