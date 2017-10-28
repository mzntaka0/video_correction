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
    super class of video.
    provide to load video

    args:
        - video_file_path[str]: the path of target video file
    """

    def __init__(self, video_file_path):
        self._video_file_path = video_file_path
        self._data = cv2.VideoCapture(video_file_path)
        self.fps = int(self._data.get(7))
        self.frame_dir = None
        if self.fps == 0:
            print("Couldn't read the video file. Please try again.")
            sys.exit(1)
        self.flag, self.frame = self._data.read()
        self.shape = self.frame.shape



class VideoGammaCorrection(Video):
    
    def __init__(self, video_file_path, split=True, frame_save_dir=None, gamma=2.2, min_=10, max_=240):
        super(VideoGammaCorrection, self).__init__(video_file_path)
        if frame_save_dir:
            self.frame_save_dir = frame_save_dir
            self.corrected_frames_dir = None
        else:
            self.frame_save_dir = os.path.split(self._video_file_path)[0]
        if split:
            self._video_split()
        self.hparams = dict()
        self.hparams['gamma'] = gamma
        self.hparams['max_'] = min_
        self.hparams['min_'] = max_
        self.gamma_lookuptable = self._gamma_lookuptable()
        self.tone_curve = self._make_tone_curve()
        

    def _video_split(self, ss=0, r=30):
        video_dir = os.path.split(self._video_file_path)[0]
        self.frame_dir = os.path.join(video_dir, 'frames')
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)
        elif os.listdir(self.frame_dir):
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
        return np.vectorize(self._gamma_func)(np.arange(256, dtype=np.uint8), self.hparams['gamma'])

    def gamma_val(self, pix_val):
        if not 0 <= pix_val < 256:
            raise ValueError('Out of value. expected: 0 <= pix_val < 256')
        return self.gamma_lookuptable[pix_val]

    def update_gamma(self, gamma):
        self.hparams['gamma'] = gamma
        self.gamma_lookuptable =  np.vectorize(self._gamma_func)(np.arange(256), self.hparams['gamma'])

    def plot_gamma(self):
        pix = np.arange(len(self.gamma_lookuptable))
        plt.scatter(pix, self.gamma_lookuptable)
        plt.show()


    def _make_tone_curve(self):
        min_ = self.hparams['min_']
        max_ = self.hparams['max_']
        tone_curve = np.arange(256, dtype=np.uint8)
        between_val = lambda i: 255 * (i - min_) / (max_ - min_)

        tone_curve = np.vectorize(between_val)(tone_curve)
        tone_curve = np.where(tone_curve < min_, 0, tone_curve)
        tone_curve = np.where(max_ <= tone_curve, 255, tone_curve)
        return tone_curve

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
                )(np.arange(256), self.hparams['gamma'])
        self.tone_curve = self._make_tone_curve()
        print('Set hyper params as: {}'.join(self.hparams))

    def fit(self):
        self.corrected_frames_dir = os.path.join(
                os.path.split(self._video_file_path)[0],
                'corrected_frames'
                )
        if not os.path.exists(self.corrected_frames_dir):
            os.makedirs(self.corrected_frames_dir)
        for image_file_name in tqdm(os.listdir(self.frame_dir)):
            pict = cv2.imread(image_file_name)
            pict = cv2.LUT(pict, self.gamma_lookuptable)
            pict = cv2.LUT(pict, self.tone_curve)
            cv2.imwrite(
                    os.path.join(
                        self.corrected_frames_dir,
                        image_file_name
                        )
                    )
        self._make_corrected_video()
        self._extract_sound()
        self._merge_video_sound()

    def _make_corrected_video(self):
        image_file = "img_%s.png"
        video_dir, video_name = os.path.split(self._video_file_path)
        cmd = [
                'ffmpeg',
                '-framerate', str(FRAME_RATE),
                '-i', os.path.join(self.corrected_frames_dir, image_file),
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(FRAME_RATE),
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
        cmd = [
                'ffmpeg',
                '-i', os.path.join(video_dir, 'raw_corrected_' + video_name),
                '-i', os.path.splitext(self._video_file_path)[0] + '.mp3',
                '-map', '0:0',
                '-map', '1:0',
                os.path.join(video_dir, 'gamma_corrected_' + video_name)
                ]
        subprocess.call(cmd)
        

    def run(self, gamma=None, max_=None, min_=None):
        self.set_hparams(gamma=gamma, max_=max_, min_=min_)
        self.fit()



# 元ビデオファイル読み込み
#org = cv2.VideoCapture(BEFORE_FILE_NAME)

# 保存ビデオファイルの準備
#fps = int(org.get(7))   # 総フレーム数を取得

# 読み込めなかった場合のエラー
#if fps == 0:
#    print("ファイルが読み込めませんでした")

#end_flag, c_frame = org.read()
#height, width, channels = c_frame.shape
#pixels = height * width # 画素数取得

# ガンマ関数数式（明るさ補正）
#lookuptable = np.ones((256, 1), dtype='uint8') * 0
#gamma = 1.01 # 学習させたデータを反映させる変数になる

#for i in range(256):
#    lookuptable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
#
## 動画を1フレームごとに画像に保存
#image_dir = "/Users/yoheiono/PycharmProjects/3mim/img/"
#image_file = "img_%s.png"
#os.system("ffmpeg -ss 0 -r 30 -i " + movie_dir + movie_name + " " + image_dir + "img_%03d.png") #
#
## 画像1枚ずつ明るさとコントラストを補正する
#cnt = 0
#min = 10 # 学習させたデータを反映させる変数になる
#max = 240 # 学習させたデータを反映させる変数になる
#
#tone_curve = np.arange(256, dtype=np.uint8)
#while (org.isOpened()):
#    flag, frame = org.read()
#    if flag == False:  # ファイルオープンに成功している間True
#        break
#    pict = cv2.imread(image_dir + image_file % str(cnt).zfill(3))
#    pict = cv2.LUT(pict, lookuptable)
#    for i in range(0, min): #トーンカーブでコントラスト調整
#        tone_curve[i] = 0
#
#    for i in range(min, max):
#        tone_curve[i] = 255 * (i - min) / (max - min)
#
#    for i in range(max, 255):
#        tone_curve[i] = 255
#    pict = cv2.LUT(pict, tone_curve)
#    cv2.imwrite(image_dir + image_file % str(cnt).zfill(3), pict)
#    cnt += 1
#
## 画像から動画を作成
#motion = "ffmpeg -framerate " + str(FRAME_RATE) + " -i " + image_dir + "img_%03d.png -vcodec libx264 -pix_fmt yuv420p -r " + str(FRAME_RATE) + " " + movie_dir + "sample001_2.mp4"
#os.system(motion)
#
## 元データから音声を抽出(256kでmp3形式で保存)
#sound = "ffmpeg -y -i " + movie_dir + movie_name + " -ab 256k " + movie_dir + "sample001.mp3"
#os.system(sound)
#
## 動画と音声を結合
#mov = "ffmpeg -i " + movie_dir + "sample001_2.mp4" + " -i " + movie_dir + "sample001.mp3 -map 0:0 -map 1:0 " + movie_dir + "output.mp4"
#os.system(mov)
#
## 終了処理
#cv2.destroyAllWindows()
#org.release()

if __name__ == '__main__':
    movie_dir = "storage/video/sample001/"
    movie_name = "sample001.mp4"
    BEFORE_FILE_NAME = os.path.join(movie_dir, movie_name) 
    FRAME_RATE = 30

    vcrr = VideoGammaCorrection(BEFORE_FILE_NAME, split=False)
    vcrr.run(gamma=2.2, max_=10, min_=240)
