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

def recognize_face(img_path):
    image = face_recognition.load_image_file("storage/image/bijin/0000.jpg")
    coords_order = [3, 0, 1, 2]
    face_locations = face_recognition.face_locations(image, model='hog')
    face_landmarks_list = face_recognition.face_landmarks(image)
    print('face_landmarks: {}'.format(faface_locationsce_landmarks_list))
    x, y, w, h = face_locations[0][3], face_locations[0][0], face_locations[0][1], face_locations[0][2]
    cv2.rectangle(image, (x, y), (w, h), (255, 0, 0))
    cv2.imshow('a', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return face_landmarks_list




    
img_path = "storage/image/gaikoku.jpg"


from PIL import Image, ImageDraw
from controllers.geometry import LineGeometry
import face_recognition

image = face_recognition.load_image_file(img_path)
#face_locations = face_recognition.face_locations(image, model='hog')
#print(face_locations)
#x, y, w, h = face_locations[0][3], face_locations[0][0], face_locations[0][1], face_locations[0][2]
#cv2.rectangle(image, (x, y), (w, h), (255, 0, 0))
#cv2.imshow('a', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Load the jpg file into a numpy array

# Find all facial features in all the faces in the image
gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
gray_line = gray.reshape(-1,)
hist = np.histogram(gray_line, bins=64)
print(hist)
cv2.imshow('gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list[0]['chin'])
line_points = np.array(face_landmarks_list[0]['chin'][0], face_landmarks_list[0]['chin'][-1])
line_geometry = LineGeometry(line_points)




raw_pil_image = Image.fromarray(image)
for face_landmarks in face_landmarks_list:
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')

    ## Make the eyebrows into a nightmare
    #d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    #d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    #d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    #d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    ## Gloss the lips
    #d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    #d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    #d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    #d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    ### Sparkle the eyes
    #d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    #d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    ### Apply some eyeliner
    #d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    #d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    #face_landmarks['chin'].append((880, 239))

    d.polygon(face_landmarks['chin'], fill=(255, 255, 255, 255))

    pil_image.show()

img = np.asarray(pil_image)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(np.where(gray > 254))
raw_img = np.asarray(raw_pil_image)
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
masked = raw_gray[np.where(gray > 254)]
hist = np.histogram(masked, 64)
print(len(hist[0]), len(hist[1]))
print(hist)
plt.scatter(hist[1][:-1], hist[0])
plt.show()
cv2.imshow('masked', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


recognize_face(img_path)
image = Image.open(img_path)
#image = to_pil(cca.automatic_color_equalization(from_pil(image)))
image = to_pil(cca.stretch(cca.gray_world(from_pil(image))))
corrected_img_path = 'storage/image/colorcorrected_{}'.format(os.path.split(img_path)[1])
image.save(corrected_img_path, 'JPEG')
icrr = ImageGammaCorrection(corrected_img_path, gamma=1.8) 
img = icrr.fit_gamma(icrr.img)
cv2.imshow('gamma', img)
cv2.waitKey(0)
cv2.destroyAllWindows()







image = face_recognition.load_image_file("storage/image/bijin/0000.jpg")
#image = Image.open(img_path)
face_locations = face_recognition.face_locations(image, model='hog')
print(face_locations)
x, y, w, h = face_locations[0][3], face_locations[0][0], face_locations[0][1], face_locations[0][2]
cv2.rectangle(image, (x, y), (w, h), (255, 0, 0))
cv2.imshow('a', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
to_pil(cca.stretch(cca.gray_world(from_pil(image)))).show()
vcrr = VideoGammaCorrection()

image = to_pil(cca.automatic_color_equalization(from_pil(image)))
image.save('colorcorrected.jpg', 'JPEG')
cv2.rectangle(image, (x, y), (w, h), (255, 0, 0))
cv2.imshow('a', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



if __name__ == '__main__':
    pass

