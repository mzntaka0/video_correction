# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2

from controllers.face_detection import get_masked_img, get_face_landmarks

def load_training_dict(pickle_path):
    with open(pickle_path) as f:
        training_img_dict = pickle.load(f)

def cut_face(img_path):
    img = cv2.imread(img_path)
    face_landmarks_list = get_face_landmarks(img)
    print(face_landmarks_list)
    chin_points = np.array(face_landmarks_list[0]['chin'])
    eyebrow_points = np.array(face_landmarks_list[0]['right_eyebrow'] + face_landmarks_list[0]['left_eyebrow'])
    x = sorted(chin_points, key=lambda x: x[0], reverse=False)
    xmin = x[0]
    xmax = x[-1]
    ymax = sorted(chin_points, key=lambda x: x[1], reverse=False)[-1]
    ymin = sorted(eyebrow_points, key=lambda x: x[1], reverse=False)[0]
    print(xmin, xmax, ymin, ymax)
    trimmed_img = img[xmin:xmax, ymin:ymax]
    plt.imshow(trimmed_img)





if __name__ == '__main__':
    img_path = './storage/image/076.jpg'
    cut_face(img_path)

