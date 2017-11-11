# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import numpy as np


class Geometry:

    def __init__(self):
        pass


class LineGeometry(Geometry):
    """
    a*x + b*y + c = 0
    """

    def __init__(self, line_points):
        self.line_points = np.array(line_points, dtype=np.float64)
        self._sub = self.line_points[1] - self.line_points[0]
        self.a = np.array(-self._sub[1] / self._sub[0], dtype=np.float64)
        self.b = 1.0
        self.c = -self.line_points[1][1] + self.line_points[1][0] * self._sub[1] / self._sub[0]

    def f(self, x):
        return -self.a*x - self.c

    def normal_vector(self):
        return np.array([self.a, self.b]) / np.sqrt(self.a**2 + self.b**2)

    def d(self, p):
        p = np.array(p, dtype=np.float64)
        return 2.0 * np.abs(self.a*p[0] + self.b*p[1] + self.c) / np.sqrt(self.a**2 + self.b**2)

    def _additional_vector(self, p):
        return self.d(p) * self.normal_vector()

    def line_symmetry(self, p):
        p = np.array(p, dtype=np.float64)
        if self.f(p[0]) > p[1]:
            return np.round(p + self._additional_vector(p))
        else:
            return np.round(p - self._additional_vector(p))

    def imshow(self):
        pass




if __name__ == '__main__':
    line_geometry = LineGeometry([[1, 1], [2, 2]])
    print(line_geometry.f(3))
    print(np.sqrt(line_geometry.normal_vector()[0]**2 + line_geometry.normal_vector()[1]**2))
    print(line_geometry.line_symmetry([0, 3]))
