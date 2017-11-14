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
    frame = cv2.LUT(frame, make_contrast_curve(zero_one_sigmoid, a=a).astype(np.uint8))
    frame = cv2.LUT(frame, gamma_lookuptable(gamma=gamma).astype(np.uint8))
    return frame

def basic_convert_inv(frame, gamma, a):
    frame = cv2.LUT(frame, make_contrast_curve(inv_zero_one_sigmoid, a=a).astype(np.uint8))
    frame = cv2.LUT(frame, gamma_lookuptable(gamma=gamma).astype(np.uint8))
    return frame

def streaming(gamma=None, a=None, weight_max=1.0, mean_average=True, dpi=150):
    gamma = 0.0
    a = 0.0
    model = AlexNet()
    model.load_state_dict(torch.load('result/pytorch/epoch-0.model'))
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(0)
    fig, axes = plt.subplots(1, 2, dpi=dpi)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0, wspace=0.0)
    for ax in axes:
        ax.tick_params(labelbottom='off', bottom='off')
        ax.tick_params(labelleft='off', left='off')
        ax.set_xticklabels([])
        ax.axis('off')
    #plt.subplots_adjust(left=None, right=None, top=None, wspace=0, hspace=0)
    #_, pred_frame = cap.read()
    #pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2HSV)
    #pred_frame = cv2.resize(pred_frame, (256, 256))
    #output = model(Variable(torch.Tensor(np.array([pred_frame]).astype(np.float32)).view(1, 3, 256, 256)))
    #gamma, a = output.data.numpy()[0][0]
    #print('gamma: {}, a: {}'.format(gamma, a))
    black_array = np.array([0, 0, 0])
    gamma_list = list()
    a_list = list()
    while True:
        ret, frame = cap.read()
        raw_frame = frame
        try:
            masked_frame = black_mask(raw_frame)
            mask = cv2.inRange(masked_frame, black_array, black_array)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            #frame = cv2.LUT(frame, gamma_lookuptable(1.5))
            filter_frame = basic_convert(frame, gamma=2.5, a=2.0)
            filter_frame = cv2.GaussianBlur(filter_frame, (25, 25), 6.0)
            raw_frame = cv2.addWeighted(raw_frame, 0.9, filter_frame, 0.8, 2.0)
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        except IndexError or UnboundLocalError:
            pass
        #output = model(Variable(torch.Tensor(np.array([frame]).astype(np.float32)).view(1, 3, 256, 256)))
        #gamma0, a0 = output.data.numpy()[0][0]
        #if mean_average:
        #    gamma_list.append(gamma0)
        #    a_list.append(a0)
        #    gamma = np.dot(np.array(gamma_list), np.linspace(0.1, weight_max, len(gamma_list))) / len(gamma_list)
        #    a = np.dot(np.array(a_list), np.linspace(0.1, weight_max, len(a_list))) / len(a_list)
        #    print('gamma: {}, a: {}'.format(gamma, a))
        #else:
        #    gamma = gamma0
        #    a = a0
        #    print('gamma: {}, a: {}'.format(gamma, a))
        #if a > 6.0:
        #    a = 6.0
        #if gamma > 4.0:
        #    gamma = 4.0
        #frame = basic_convert(frame, gamma=gamma, a=a)
        axes[0].imshow(frame)
        axes[1].imshow(raw_frame)
        plt.pause(0.05)
        plt.cla()

        if len(gamma_list) > 100:
            gamma_list = [gamma]
            a_list = [a]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    streaming(weight_max=1.0, mean_average=True, dpi=150)
