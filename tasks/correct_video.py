# -*- coding: utf-8 -*-
"""
"""
import os
import sys

import colorcorrect
from bpdb import set_trace
import face_recognition
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import numpy as np

from controllers.video_correction import VideoGammaCorrection
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil


class ImageGammaCorrection:

    def __init__(self, img_path, gamma=2.2, max_=240, min_=10):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise ValueError('the image has not been loaded.')
        self.hparams = dict()
        self.hparams['gamma'] = gamma
        self.hparams['max_'] = max_
        self.hparams['min_'] = min_
        self.gamma_list = np.arange(1.0, 2.51, 0.01)
        self.gamma_lookuptable = self._gamma_lookuptable()
        self.tone_curve = self._make_tone_curve()

    def _gamma_func(self, i, gamma=2.2):
        return 255 * pow(float(i) / 255, 1.0 / gamma)

    def _gamma_lookuptable(self):
        return np.vectorize(
                self._gamma_func
                )(np.arange(256, dtype=np.uint8).reshape(-1, 1), self.hparams['gamma'])

    def gamma_val(self, pix_val):
        if not 0 <= pix_val < 256:
            raise ValueError('Out of value. expected: 0 <= pix_val < 256')
        return self.gamma_lookuptable[pix_val][0]

    def update_gamma(self, gamma):
        self.hparams['gamma'] = gamma
        self.gamma_lookuptable =  np.vectorize(
                self._gamma_func
                )(np.arange(256, dtype=np.uint8).reshape(-1, 1), self.hparams['gamma'])

    def plot_gamma(self):
        pix = np.arange(len(self.gamma_lookuptable))
        plt.scatter(pix, self.gamma_lookuptable.reshape(-1,))
        plt.show()

    def plot_tone_curve(self):
        pix = np.arange(len(self.tone_curve))
        plt.scatter(pix, self.tone_curve.reshape(-1,))
        plt.show()


    def _make_tone_curve(self):
        min_ = self.hparams['min_']
        max_ = self.hparams['max_']
        tone_curve = np.arange(256, dtype=np.uint8)
        between_val = lambda i: 255 * (i - min_) / (max_ - min_)

        tone_curve = np.vectorize(between_val)(tone_curve)
        tone_curve = np.where(tone_curve < min_, 0, tone_curve)
        tone_curve = np.where(max_ <= tone_curve, 255, tone_curve)
        return tone_curve.reshape(-1, 1)

    def _check_none(self, **kwargs):
        for key, val in kwargs.items():
            if val == None and self.hparams[key] == None:
                print('Please set hyper params. Try again.')
                sys.exit(1)

    def set_hparams(self, gamma=None, max_=None, min_=None):
        self._check_none(gamma=gamma, max_=max_, min_=min_)
        self.hparams['gamma'] = gamma
        self.hparams['max_'] = max_
        self.hparams['min_'] = min_
        self.gamma_lookuptable =  np.vectorize(
                self._gamma_func
                )(np.arange(256, dtype=np.uint8).reshape(-1, 1), self.hparams['gamma'])
        self.tone_curve = self._make_tone_curve()
        print('Hyper parameters have been set as: {}'.format(self.hparams))

    def fit_gamma(self, img):
        return np.round(cv2.LUT(img, self.gamma_lookuptable)).astype(np.uint8)

    def fit_tone_curve(self, img):
        return np.round(cv2.LUT(img, self.tone_curve)).astype(np.uint8)

    def make_reverse_gamma_image_list(self):
        img_list = list()
        img_dir, img_name = os.path.split(self.img_path)
        corrected_img_dir = os.path.join(
                img_dir,
                'gamma_corrected_' + img_name.replace(os.path.splitext(self.img_path)[1], '')
                )
        if not os.path.exists(corrected_img_dir):
            os.makedirs(corrected_img_dir)
        for gamma in self.gamma_list:
            reciprocal_gamma = 1.0 / gamma
            self.set_hparams(gamma=reciprocal_gamma, max_=240, min_=10)
            corrected_img = self.fit_gamma(self.img)
            img_list.append(corrected_img)
            corrected_img_name = 'reciprocal_gamma_{:.2f}__'.format(gamma) + img_name
            cv2.imwrite(os.path.join(corrected_img_dir, corrected_img_name), corrected_img)
        return img_list




img_path = "storage/image/bijin/0000.jpg"
img_dir_path = ['storage/image/bijin/', 'storage/image/agejo/']
for img_dir in img_dir_path:
    for img_name in os.listdir(img_dir):
        try:
            icrr = ImageGammaCorrection(os.path.join(img_dir, img_name))
        except ValueError:
            continue
        icrr.make_reverse_gamma_image_list()
set_trace()
icrr = ImageGammaCorrection(img_path)

img_list = icrr.make_reverse_gamma_image_list()



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
