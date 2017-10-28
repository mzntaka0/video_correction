# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import cv2
import numpy as np

# 定数定義
ESC_KEY = 27     # Escキー
INTERVAL = 33     # 待ち時間
FRAME_RATE = 30

ORG_WINDOW_NAME = "org"
GRAY_WINDOW_NAME = "gray"

ORG_FILE_NAME = "/Users/yoheiono/PycharmProjects/3mim/movie/IMG_0884.MOV"
GRAY_FILE_NAME = "/Users/yoheiono/PycharmProjects/3mim/movie/IMG_0884_change.MOV"

# 元ビデオファイル読み込み
org = cv2.VideoCapture(ORG_FILE_NAME)

# 保存ビデオファイルの準備
fps = int(org.get(7))   # 総フレーム数を取得
end_flag, c_frame = org.read()
height, width, channels = c_frame.shape
rec = cv2.VideoWriter(GRAY_FILE_NAME, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (width, height), False)

pixels = height * width

# ガンマ関数数式（明るさ補正）
lookuptable = np.ones((256, 1), dtype='uint8') * 0
gamma = 2.0

for i in range(256):
    lookuptable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

# Video to frames
image_dir = "/Users/yoheiono/PycharmProjects/3mim/img/"
image_file = "img_%s.png"
cnt = 0
min = 100
max = 200
tone_curve = np.arange(256, dtype=np.uint8)

# 動画を1フレームごとに画像に保存
while (org.isOpened()):
    flag, frame = org.read()  # Capture frame-by-frame
    if flag == False:  # ファイルオープンに成功している間True
        break
    cv2.imwrite(image_dir + image_file % str(cnt).zfill(3), frame)  # フレームごとに画像を保存
    print('Save', image_dir + image_file % str(cnt).zfill(3))
    pict = cv2.imread(image_dir + image_file % str(cnt).zfill(3))
    pict = cv2.LUT(pict, lookuptable)
    for i in range(0, min):
        tone_curve[i] = 0

    for i in range(min, max):
        tone_curve[i] = 255 * (i - min) / (max - min)

    for i in range(max, 255):
        tone_curve[i] = 255
    pict = cv2.LUT(pict, tone_curve)
    cv2.imwrite(image_dir + image_file % str(cnt).zfill(3), pict)
    cnt += 1


# 画像から動画の作成
i = 0
for i in range(0, fps-1):
    img = cv2.imread(image_dir + "img_" + str(i).zfill(3) + ".png")
    img = cv2.resize(img, (width, height))
    rec.write(img)

rec.release()


# 終了処理
cv2.destroyAllWindows()
org.release()
rec.release()


if __name__ == '__main__':
    pass

