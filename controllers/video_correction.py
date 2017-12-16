# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class Video:
    """
    Super class of video.
    Provide to load video

    Args:
        - video_file_path[str]: the path of target video file
    """

    def __init__(self, video_file_path):
        self._video_file_path = video_file_path
        self._data = cv2.VideoCapture(video_file_path)
        self.frame_num = int(self._data.get(7))
        if self.frame_num == 0:
            print("Couldn't read the video file. Please try again.")
            sys.exit(1)
        self.fps = round(self._data.get(cv2.CAP_PROP_FPS) )
        self.flag, self.frame = self._data.read()
        self.shape = self.frame.shape



class VideoGammaCorrection(Video):
    
    def __init__(self, video_file_path, split=True, frame_dir=None, gamma=2.2, min_=10, max_=240, fr=30):
        super(VideoGammaCorrection, self).__init__(video_file_path)
        if frame_dir:
            self.frame_dir = frame_dir
        else:
            self.frame_dir = os.path.join(os.path.split(self._video_file_path)[0], 'frames')
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)
        if split:
            self._video_split()
        self.corrected_frames_dir = None
        self.hparams = dict()
        self.hparams['gamma'] = gamma
        self.hparams['max_'] = max_
        self.hparams['min_'] = min_
        self.gamma_lookuptable = self._gamma_lookuptable()
        self.tone_curve = self._make_tone_curve()
        self.FRAME_RATE = fr
        
    def _video_split(self, ss=0, r=30):
        if os.listdir(self.frame_dir):
            print('frames path: {}'.format(self.frame_dir))
            ans = input('The frames already exist. Make frames again?[y/n]')
            if ans != 'y':
                return 
        cmd = [
                'ffmpeg', 
                '-ss', str(ss),
                '-r', str(r),
                '-i', self._video_file_path,
                os.path.join(self.frame_dir, 'img_%04d.png')
                ]
        subprocess.call(cmd)

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

    def fit(self):
        video_dir = os.path.split(self._video_file_path)[0]
        self.corrected_frames_dir = os.path.join(
                os.path.split(self._video_file_path)[0],
                'corrected_frames'
                )
        print(len(self.tone_curve))
        if not os.path.exists(self.corrected_frames_dir):
            os.makedirs(self.corrected_frames_dir)
        for image_file_name in tqdm(os.listdir(self.frame_dir)):
            image_file_path = os.path.join(video_dir, 'frames', image_file_name)
            pict = cv2.imread(image_file_path)
            if pict is None:
                print('The image has not been loaded. Set accurate file path.')
                sys.exit(1)
            try:
                pict = np.round(cv2.LUT(pict, self.gamma_lookuptable)).astype(np.uint8)
            except cv2.error:
                print('!!gamma correction has been passed because of error.')
            #try:
            #    pict = cv2.LUT(pict, self.tone_curve)
            #except cv2.error:
            #    print('!!tone curve correction has been passed because of error.')
            cv2.imwrite(
                    os.path.join(
                        self.corrected_frames_dir,
                        image_file_name
                        ),
                    pict
                    )
        self._make_corrected_video()
        self._extract_sound()
        self._merge_video_sound()
        self._rm_raw_files()

    def _make_corrected_video(self):
        image_file = "img_%04d.png"
        video_dir, video_name = os.path.split(self._video_file_path)
        cmd = [
                'ffmpeg',
                '-framerate', str(self.FRAME_RATE),
                '-i', os.path.join(self.corrected_frames_dir, image_file),
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(self.FRAME_RATE),
                os.path.join(video_dir, 'raw_corrected_' + video_name)
                ]
        subprocess.call(cmd)

    def _extract_sound(self):
        cmd = [
                'ffmpeg',
                '-y',
                '-i', self._video_file_path,
                '-ab', '256k',
                os.path.splitext(self._video_file_path)[0] + '.mp3'
                ]
        subprocess.call(cmd)

    def _merge_video_sound(self):
        video_dir, video_name = os.path.split(self._video_file_path)
        corrected_video_dir = os.path.join(video_dir, 'corrected_videos')
        if not os.path.exists(corrected_video_dir):
            os.makedirs(corrected_video_dir)
        cmd = [
                'ffmpeg',
                '-i', os.path.join(video_dir, 'raw_corrected_' + video_name),
                '-i', os.path.splitext(self._video_file_path)[0] + '.mp3',
                '-map', '0:0',
                '-map', '1:0',
                os.path.join(corrected_video_dir,
                    'gamma_{}__max_{}__min_{}__'.format(
                        self.hparams['gamma'],
                        self.hparams['max_'],
                        self.hparams['min_']
                        ) + video_name
                    )
                ]
        subprocess.call(cmd)

    def _rm_raw_files(self):
        video_dir, video_name = os.path.split(self._video_file_path)
        rm_paths = list()
        rm_paths.append(os.path.join(video_dir, 'raw_corrected_' + video_name))
        rm_paths.append(os.path.splitext(self._video_file_path)[0] + '.mp3')
        rm_paths.append(os.path.join(video_dir, 'corrected_frames'))
        for path in rm_paths:
            subprocess.call(['rm', '-r', path])
        

    def run(self, gamma=None, max_=None, min_=None):
        self.set_hparams(gamma=gamma, max_=max_, min_=min_)
        self.fit()
        self._data.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    movie_dir = "storage/video/sample001/"
    movie_name = "sample001.mp4"
    BEFORE_FILE_NAME = os.path.join(movie_dir, movie_name) 
    FRAME_RATE = 30

    vcrr = VideoGammaCorrection(BEFORE_FILE_NAME)
    vcrr.run(gamma=2.0, max_=240, min_=10)
