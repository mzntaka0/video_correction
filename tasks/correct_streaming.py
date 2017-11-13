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

def tone_func(x):
    return 255 * ((1 / np.pi) * np.arcsin(2*x - 1) + 1 / 2)


def zero_one_sigmoid(x, a=5.0):
    return 255*((1/2)*(1 + ((1 - np.exp(-a*(2*x-1)))) / (1 + np.exp(-a*(2*x - 1))) * (((1 + np.exp(-a)))/(1-np.exp(-a)))))

def inv_zero_one_sigmoid(x, a=4.0):
    return 255*(1 - ((np.log((1 - _X(x, a)) / (1 + _X(x, a))) + a) / (2.0 * a)))

def make_tone_curve(func):
    return np.array([int(np.round(func(x/256))) for x in np.arange(256)]).reshape(-1, 1).astype(np.uint8)


def tone_func_contrast(x):
    return 255 * (np.sin(np.pi*x - np.pi/2) + 1) / 2

def _X(x, a):
    return (2*x - 1) * ((1 - np.exp(-a)) / (1 + np.exp(-a)))

def gamma_func(i, gamma=2.2):
    return 255 * pow(float(i) / 255, 1.0 / gamma)

def gamma_lookuptable(gamma):
    return np.vectorize(
            gamma_func
            )(np.arange(256, dtype=np.uint8).reshape(-1, 1), gamma)

if __name__ == '__main__':
    image_path = 'storage/image/gaikoku.jpg'
    img = cv2.imread(image_path)
    ##fig = plt.figure()
    ##ax = Axes3D(fig)
    ##ax.set_xlabel('Hue')
    ##ax.set_ylabel('Saturation')
    ##ax.set_zlabel('Value')
    ##ax.plot(img[:, :, 0], img[:, :, 1], img[:, :, 2], 'o')
    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #skeleton = imutils.skeletonize(gray, size=(3, 3))
    #cv2.imshow('ske', skeleton)
    #set_trace()
    ##cv2.imshow('img', img)
    #masked_face = black_mask(img)
    #masked_face = cv2.cvtColor(masked_face, cv2.COLOR_RGB2GRAY)
    #masked_pixels = img_hsv[np.where(masked_face==0)[0], np.where(masked_face==0)[1]]
    #masked_pixels_bgr = img[np.where(masked_face==0)[0], np.where(masked_face==0)[1]]
    #masked_pixels_gray = gray[np.where(masked_face==0)[0], np.where(masked_face==0)[1]]
    #print(np.histogram(masked_pixels_gray, bins=64))
    #x = np.histogram(masked_pixels_gray, bins=64)
    #plt.bar(x[1][:-1], x[0])
    #plt.show()
    ##fig = plt.figure()
    ##ax = Axes3D(fig)
    ##ax.set_xlabel('Hue')
    ##ax.set_ylabel('Saturation')
    ##ax.set_zlabel('Value')
    ##ax.plot(masked_pixels_bgr[:, 0], masked_pixels_bgr[:, 1], masked_pixels_bgr[:, 2], 'o')
    ##plt.show()
    #print(np.histogram(masked_pixels_gray, bins=32))
    ##fig = plt.figure()
    ##ax = fig.add_subplot(111)

    ##H = ax.hist2d(masked_pixels[:, 0], masked_pixels[:, 1], bins=128)
    ##fig.colorbar(H[3], ax=ax)
    ##plt.show()

    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.set_xlabel('Hue')
    #ax.set_ylabel('Saturation')
    #ax.set_zlabel('Value')
    #ax.plot(masked_pixels[:, 0], masked_pixels[:, 1], masked_pixels[:, 2], 'o')
    #plt.show()
    #masked_face = np.where(masked_face <= 0, 255, 0)
    #cv2.imwrite('mask.png', masked_face)
    #mask = cv2.imread('mask.png', 0)
    #bgr = cv2.split(img)
    #bgra = cv2.merge(bgr + [mask])
    #bgra = cv2.LUT(bgra, make_tone_curve(zero_one_sigmoid))
    #gray_bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray_bgra', gray_bgra)

    #print(masked_face)
    #cv2.imshow('masked', masked_face)


    #lut_img = cv2.LUT(img, make_tone_curve(zero_one_sigmoid))
    ##cv2.imshow('img_lut', lut_img)
    #lut_inv_img = cv2.LUT(lut_img, make_tone_curve(inv_zero_one_sigmoid))
    ##masked_face = lut_img[np.where(lut_inv_img == [255, 0, 0])]
    ##print(masked_face)
    ##masked_face = cv2.cvtColor(masked_face, cv2.COLOR_BGR2GRAY)
    #
    ##print(np.histogram(masked_face, bins=64))
    ##print(masked_face)

    #lut_inv_img = cv2.cvtColor(lut_inv_img, cv2.COLOR_RGB2BGR)
    ##cv2.imshow('inv_lut', lut_inv_img)

    #set_trace()


    #y = gamma_lookuptable(2.2)
    #print(len(y))
    #plt.scatter(np.arange(len(y)), y)
    #plt.show()

    #s = time.time()
    #img = black_mask(img)
    #e = time.time()
    #print(e - s)

    cap = cv2.VideoCapture(0)

    fig, ax = plt.subplots()

    while True:
        ret, frame = cap.read()
        #try:
        #    frame = black_mask(frame)
        #except IndexError:
        #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.LUT(frame, make_tone_curve(zero_one_sigmoid).astype(np.uint8))
        frame = cv2.LUT(frame, gamma_lookuptable(2.0).astype(np.uint8))
        ax.imshow(frame)
        plt.pause(0.05)
        plt.cla()
    cap.release()
    cv2.destroyAllWindows()
