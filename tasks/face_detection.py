# -*- coding: utf-8 -*-
"""
"""
import os
import sys
from imutils import face_utils
import imutils
import numpy as np
import collections
import dlib
import cv2

#def face_remap(shape):
#   remapped_image = shape.copy()
#   # left eye brow
#   remapped_image[17] = shape[26]
#   remapped_image[18] = shape[25]
#   remapped_image[19] = shape[24]
#   remapped_image[20] = shape[23]
#   remapped_image[21] = shape[22]
#   # right eye brow
#   remapped_image[22] = shape[21]
#   remapped_image[23] = shape[20]
#   remapped_image[24] = shape[19]
#   remapped_image[25] = shape[18]
#   remapped_image[26] = shape[17]
#   # neatening
#   remapped_image[27] = shape[0]
#
#   return remapped_image

img_path = "storage/image/076.jpg"

def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image

"""
MAIN CODE STARTS HERE
"""
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(img_path)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

out_face = np.zeros_like(image)

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
print(detector.__dict__)
predictor = dlib.shape_predictor()

# detect faces in the grayscale image
rects = detector(gray, 1)
print(rects.__dict__)

# loop over the face detections
for (i, rect) in enumerate(rects):
   """
   Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
   """
   shape = predictor(gray, rect)
   print(shape.part(0))
   shape = face_utils.shape_to_np(shape)

   #initialize mask array
   remapped_shape = np.zeros_like(shape)
   feature_mask = np.zeros((image.shape[0], image.shape[1]))

   # we extract the face
   remapped_shape = face_remap(shape)
   cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
   feature_mask = feature_mask.astype(np.bool)
   out_face[feature_mask] = image[feature_mask]
   cv2.imshow("mask_inv", out_face)
   cv2.imwrite("out_face.png", out_face)


if __name__ == '__main__':
    pass

