# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import numpy as np
import cv2

HSV_MIN = np.array([0, 30, 60])
HSV_MAX = np.array([20, 150, 255])

cap = cv2.VideoCapture(0)

# 初期背景を設定するため画像取得
ret, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()

    # 背景画像との差分を取る
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background)

    # 肌色検出がしやすいようにBGRからHSVに変換
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv_mask = cv2.inRange(hsv_image, HSV_MIN, HSV_MAX)

    # 色相が肌色に近い部分を抽出
    cv2.imshow('Capture', frame)
    cv2.imshow('hsv_mask', hsv_mask)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    elif key == ord('b'):
        # bを押したらその時点でのキャプチャを背景とする
        ret, frame = cap.read()
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    pass

