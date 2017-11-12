# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def zero_one_sigmoid(x, a):
    return (1/2)*(1 + ((1 - np.exp(-a*(2*x-1)))) / (1 + np.exp(-a*(2*x - 1))) * (((1 + np.exp(-a)))/(1-np.exp(-a))))


def _X(x, a):
    return (2*x - 1) * ((1 - np.exp(-a)) / (1 + np.exp(-a)))

def inv_zero_one_sigmoid(x, a):
    return 1 - ((np.log((1 - _X(x, a)) / (1 + _X(x, a))) + a) / (2.0 * a))

if __name__ == '__main__':
    a = np.arange(0, 4, 0.1)
    x = np.arange(0, 1, 0.01)
    #for i in a:
    #    y = inv_zero_one_sigmoid(x, i)
    #    plt.scatter(x, y)
    #    plt.xlim(0, 1)
    #    plt.ylim(0, 1)
    #    plt.savefig('storage/image/zero_one_sigmoid/{}_inv.png'.format(i))
    #    plt.cla()
    a = 3.0
    line_x = np.linspace(0, 1)
    plt.plot(line_x, line_x, 'r-')
    plt.scatter(x, inv_zero_one_sigmoid(x, a))
    plt.scatter(x, zero_one_sigmoid(x, a))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


