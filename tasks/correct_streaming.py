# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import time

import cv2
import imutils
from bpdb import set_trace
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from controllers.face_detection import black_mask
from tasks.train import AlexNet





def tone_func(x):
    return 255 * ((1 / np.pi) * np.arcsin(2*x - 1) + 1 / 2)

def zero_one_sigmoid(x, a):
    return 255*((1/2)*(1 + ((1 - np.exp(-a*(2*x-1)))) / (1 + np.exp(-a*(2*x - 1))) * (((1 + np.exp(-a)))/(1-np.exp(-a)))))

def inv_zero_one_sigmoid(x, a):
    return 255*(1 - ((np.log((1 - _X(x, a)) / (1 + _X(x, a))) + a) / (2.0 * a)))

def _X(x, a):
    return (2*x - 1) * ((1 - np.exp(-a)) / (1 + np.exp(-a)))

def make_contrast_curve(func, a):
    return np.array([int(np.round(func(x/256, a))) for x in np.arange(256)]).reshape(-1, 1).astype(np.uint8)


def tone_func_contrast(x):
    return 255 * (np.sin(np.pi*x - np.pi/2) + 1) / 2


def gamma_func(i, gamma=2.2):
    return 255 * pow(float(i) / 255, 1.0 / gamma)

def gamma_lookuptable(gamma):
    return np.vectorize(
            gamma_func
            )(np.arange(256, dtype=np.uint8).reshape(-1, 1), gamma)


def basic_convert(frame, gamma, a):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.LUT(frame, make_contrast_curve(zero_one_sigmoid, a=a).astype(np.uint8))
    frame = cv2.LUT(frame, gamma_lookuptable(gamma=gamma).astype(np.uint8))
    return frame


def streaming(gamma=None, a=None):
    gamma = 0.0
    a = 0.0
    model = AlexNet()
    model.load_state_dict(torch.load('result/pytorch/epoch-0.model'))
    cap = cv2.VideoCapture(0)
    fig, ax = plt.subplots()
    _, pred_frame = cap.read()
    pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2HSV)
    pred_frame = cv2.resize(pred_frame, (256, 256))
    output = model(Variable(torch.Tensor(np.array([pred_frame]).astype(np.float32)).view(1, 3, 256, 256)))
    gamma, a = output.data.numpy()[0][0]
    print('gamma: {}, a: {}'.format(gamma, a))
    gamma_list = list()
    a_list = list()
    cnt = 0
    while True:
        cnt += 1
        ret, frame = cap.read()
        #try:
        #    frame = black_mask(frame)
        #    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #except IndexError:
        #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    pass

        output = model(Variable(torch.Tensor(np.array([pred_frame]).astype(np.float32)).view(1, 3, 256, 256)))
        print(output)
        gamma0, a0 = output.data.numpy()[0][0]
        gamma_list.append(gamma0)
        a_list.append(a0)
        gamma = np.dot(np.array(gamma_list), np.linspace(0.1, 1.0, len(gamma_list))) / len(gamma_list)
        a = np.dot(np.array(a_list), np.linspace(0.1, 1.0, len(a_list))) / len(a_list)
        print('gamma: {}, a: {}'.format(gamma, a))
        frame = basic_convert(frame, gamma=gamma, a=a)
        ax.imshow(frame)
        plt.pause(0.05)
        plt.cla()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # prediction targets
    gamma = 1.5 
    a = 4.0

    streaming(gamma, a)
