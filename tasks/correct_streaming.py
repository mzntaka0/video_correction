# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import time

import cv2
from bpdb import set_trace

from tasks.face_detection import black_mask




if __name__ == '__main__':
    image_path = 'storage/image/72d780808e3bd6d05b244cbf44280c89.jpg'
    img = cv2.imread(image_path)

    s = time.time()
    img = black_mask(img)
    e = time.time()
    print(e - s)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        try:
            frame = black_mask(frame)
            cv2.imshow('camera capture', frame)
        except IndexError:
            cv2.imshow('camera capture', frame)
    cap.release()
    cv2.destroyAllWindows()
