# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import time

import cv2
from bpdb import set_trace
import numpy as np
import matplotlib.pyplot as plt

from tasks.face_detection import black_mask

def tone_func(x):
    return 255 * ((1 / np.pi) * np.arcsin(2*x - 1) + 1 / 2)


def make_tone_curve(func):
    return np.array([int(np.round(func(x/256))) for x in np.arange(256)]).reshape(-1, 1).astype(np.uint8)


def tone_func_contrast(x):
    return 255 * (np.sin(np.pi*x - np.pi/2) + 1) / 2


def gamma_func(i, gamma=2.2):
    return 255 * pow(float(i) / 255, 1.0 / gamma)

def gamma_lookuptable(gamma):
    return np.vectorize(
            gamma_func
            )(np.arange(256, dtype=np.uint8).reshape(-1, 1), gamma)

if __name__ == '__main__':
    image_path = 'storage/image/72d780808e3bd6d05b244cbf44280c89.jpg'
    img = cv2.imread(image_path, 1)
    #cv2.imshow('img', img)
    #print(make_tone_curve(tone_func_contrast))
    #print(np.arange(256).reshape(-1, 1))
    #lut_img = cv2.LUT(img, make_tone_curve(tone_func_contrast))
    #cv2.imshow('lut', lut_img)
    #set_trace()


    y = gamma_lookuptable(2.2)
    print(len(y))
    plt.scatter(np.arange(len(y)), y)
    plt.show()

    s = time.time()
    img = black_mask(img)
    e = time.time()
    print(e - s)

    cap = cv2.VideoCapture(0)

    fig, ax = plt.subplots()

    while True:
        ret, frame = cap.read()
        try:
            frame = black_mask(frame)
        except IndexError:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pass

        #frame = cv2.LUT(frame, gamma_lookuptable(1.5).astype(np.uint8))
        frame = cv2.LUT(frame, make_tone_curve(tone_func_contrast).astype(np.uint8))
        ax.imshow(frame)
        plt.pause(0.05)
        plt.cla()
    cap.release()
    cv2.destroyAllWindows()
