# -*- coding: utf-8 -*-
"""
"""
import os
import sys

from controllers.video_correction import VideoGammaCorrection



if __name__ == '__main__':
    movie_dir = "storage/video/city_1/"
    movie_name = "city_1.MOV"
    BEFORE_FILE_NAME = os.path.join(movie_dir, movie_name) 
    FRAME_RATE = 30

    vcrr = VideoGammaCorrection(BEFORE_FILE_NAME)
    vcrr.run(gamma=2.0, max_=240, min_=10)

