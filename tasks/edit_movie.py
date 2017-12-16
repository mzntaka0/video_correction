# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import subprocess
from collections import OrderedDict


import cv2

from controllers.video_correction import Video


class CropVideo:
    """
    this class manipulate the videos on the disk using ffmpeg, not putting on the memory(in case of huge video data.)
    Args:
        - video_path[str]: the target video file path
    """

    def __init__(self, video_path, cropped_dir=None):
        self._target_video = video_path
        if cropped_dir:
            self.cropped_dir = cropped_dir
        else:
            _splited_path = os.path.split(self._target_video)
            self.cropped_dir = os.path.join(_splited_path[0], os.path.splitext(_splited_path[1])[0] + '_cropped')
        if not os.path.exists(self.cropped_dir):
            os.makedirs(self.cropped_dir)
        self.cropped_video_dict = OrderedDict()  # {cropped_video_name: cropped_video_path, ...}
        self.video_extension = ['.mov', '.mp4']


    # todo: have to decide whether convert to 30[fps] or adjust to the video fps
    # todo: which is better argument of duration or end
    # todo: which is better argument of tuple or separate args about crop time
    # todo: where should cropped video be placed(temporally assume it's better to put where the original video be)
    def crop(self, video_name, start_time, duration, ext='mov'):
        """
        crop video using ffmpeg

        input:
            - video_name[str]: video name cropping
            #- duration[tuple]: ('HH:mm:ss', 'HH:mm:ss') ((start, duration)) specify time to crop

        output:
            - cropped video to the directory
        """
        _output_path = os.path.join(self.cropped_dir, '{}.{}'.format(video_name, ext))
        cmd = [
                'ffmpeg', 
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', self._target_video,
                _output_path
                ]
        subprocess.call(cmd)
        self.cropped_video_dict[video_name] = _output_path


    def get_duration(start_time, end_time):
        pass

    def _reorder_cropped_video_dict(self, order_list):
        reorderd_dict = OrderedDict()
        for video_name in order_list:
            try:
                reorderd_dict[video_name] = self.cropped_video_dict[video_name1]
            except KeyError:
                ans = input('The video name is not in the cropped videos. Continue?(else Exit)[y/n]')
                if not ans == 'y':
                    sys.exit(0)
                else:
                    continue
        self.cropped_video_dict = reorderd_dict


    def render(self, rendered_name=None, order_list=None, ext='mov'):
        """
        save the video chained cropped videos, after cropping these

        input:
            - order_list[list](option): video_name list which you wanna order to render
        """
        if order_list:
            self._reorder_cropped_video_dict(order_list)

        rendering_video_num = len(self.cropped_video_dict)
        rendering_video_cmd = list()
        for video_path in self.cropped_video_dict.values():
            rendering_video_cmd.append('-i')
            rendering_video_cmd.append(video_path)

        _splited_path = os.path.split(self._target_video)
        rendered_dir = os.path.join(_splited_path[0], os.path.splitext(_splited_path[1])[0] + '_rendered')
        if not os.path.exists(rendered_dir):
            os.makedirs(rendered_dir)
        rendered_name = os.path.splitext(_splited_path[1])[0] + 'rendered_' + str(len(os.listdir(rendered_dir)))
        _output_path = os.path.join(rendered_dir, rendered_name)
        print(_output_path)
        cmd = ['ffmpeg'] + rendering_video_cmd + ['-filter_complex', 'concat=n={}:v=1:a=1'.format(rendering_video_num), '{}.{}'.format(_output_path, ext)] 
        print(' '.join(cmd))
        subprocess.call(cmd)


if __name__ == '__main__':
    video_path = 'storage/tv_9dw.mov'
    cropvideo = CropVideo(video_path)
    cropvideo.crop('test', '00:00:10', '00:00:3')
    cropvideo.crop('test2', '00:00:20', '00:00:5')
    cropvideo.crop('test3', '00:00:50', '00:00:15')
    print(cropvideo.cropped_video_dict)
    cropvideo.render()
