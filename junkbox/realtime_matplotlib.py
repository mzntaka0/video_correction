# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import cv2
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

predictor_path = "shape_predictor_68_face_landmarks.dat"

# download trained model
if not os.path.isfile(predictor_path):
    os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    os.system("bunzip2 shape_predictor_68_face_landmarks.dat.bz2")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(frame, 1)

    # for each detected face
    for d in dets:
        # draw bounding box of the detected face
        #rect = patches.Rectangle((d.left(), d.top()), d.width(), d.height(), fill=False)
        #ax.add_patch(rect)

        # draw landmarks
        parts = predictor(frame, d).parts()
        ax.scatter([point.x for point in parts], [point.y for point in parts])

        for k, point in enumerate(parts):
            ax.text(point.x, point.y, k)

        ax.imshow(frame)
        plt.pause(0.1)
        plt.cla()


if __name__ == '__main__':
    pass

