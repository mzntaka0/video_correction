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
