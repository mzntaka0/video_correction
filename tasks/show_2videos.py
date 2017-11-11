# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import cv2

ESC_KEY = 27
INTERVAL= 33
FRAME_RATE = 30

ORG_WINDOW_NAME = "org"
GRAY_WINDOW_NAME = "gray"

ORG_FILE_NAME = "storage/video/city_1/city_1.MOV"
CRD_FILE_NAME = "storage/video/city_1/city_1_2.mov"

crd = cv2.VideoCapture(CRD_FILE_NAME)
org = cv2.VideoCapture(ORG_FILE_NAME)
end_flag, c_frame = org.read()
end_flag, g_frame = crd.read()
height, width, channels = c_frame.shape

cv2.namedWindow(ORG_WINDOW_NAME)
cv2.namedWindow(GRAY_WINDOW_NAME)


while end_flag == True:

    cv2.imshow(ORG_WINDOW_NAME, c_frame)
    cv2.imshow(GRAY_WINDOW_NAME, g_frame)


    key = cv2.waitKey(INTERVAL)
    if key == ESC_KEY:
        break

    end_flag, c_frame = org.read()

cv2.destroyAllWindows()
org.release()
crd.release()


if __name__ == '__main__':
    pass

