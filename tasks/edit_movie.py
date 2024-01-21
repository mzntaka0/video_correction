# -*- coding: utf-8 -*-
"""
"""
import os
import sys
import subprocess
from multiprocessing import Process, Queue, Lock, set_start_method
from collections import OrderedDict
try:
    from bpdb import set_trace
except ImportError:
    from pdb import set_trace

import cv2

from controllers.video_correction import Video
from controllers.video_edit_utils import string_to_ffmpeg_format
#from submodule.sixcns.tasks.transcribe_streaming_mic import streaming


class ProjectBase:
    """
    [super class]
    when you start new project to edit videos, this class will be the director of whole operations.
    this class provides the initialization of the project. make initial directories following.
       - raws: raw video data not edited
       - cropped: cropped video data
       - rendered: rendered video data

    Args:
        - project_name[str]: the project name what you want to make
        - project_path[str](option): the path which you want to start project
    """

    def __init__(self, project_name, project_path=os.path.abspath('storage/projects')):
        self.project_name = project_name
        self.project_path = project_path
        self.project_root = os.path.join(project_path, project_name)

    def init(self):
        if not os.path.exists(self.project_root):
            os.makedirs(self.project_root)
            self._project_dirs()
            print('Please put videos to {}/raw'.format(self.project_name))
            ans = input('Continue?[y/n]: ')
            if not ans == 'y':
                sys.exit(0)
        else:
            print('Loading the project "{}". Enjoy Editing.'.format(self.project_name))
        
    def _project_dirs(self):
        os.makedirs(os.path.join(self.project_root, 'raw'))
        os.makedirs(os.path.join(self.project_root, 'cropped'))
        os.makedirs(os.path.join(self.project_root, 'rendered'))


class Project(ProjectBase):
    """
    this class provides the project editing 
    """

    def __init__(self, project_name, project_path=os.path.abspath('storage/projects')):
        super(Project, self).__init__(project_name, project_path=project_path)
        self.video_extension = ['.mov', '.mp4']

    def crop_mode(self, video_name, ext='mov'):
        video_path = os.path.join(self.project_root, 'raw', '{}.{}'.format(video_name, ext))
        if not os.path.exists(video_path):
            raise FileNotFoundError('the video "{}" does not exists.'.format(os.path.split(video_path)[1]))
        cropped_dir = os.path.join(self.project_root, 'cropped', video_name + '_cropped')
        return CropVideo(video_path, cropped_dir=cropped_dir)

    def render_mode(self):
        pass



class CropVideo:
    """
    this class operates the videos on the disk using ffmpeg, not putting on the memory(in case of huge video data.)
    Args:
        - video_path[str]: the target video file path
        - cropped_dir[str](option): just in case you want to specify the path of cropped videos.
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
        cmd = ['ffmpeg'] + rendering_video_cmd + ['-filter_complex', 'concat=n={}:v=1:a=1'.format(rendering_video_num), '{}.{}'.format(_output_path, ext)] 
        subprocess.call(cmd)


if __name__ == '__main__':
    # Audio recording parameters
    #RATE = 16000
    #CHUNK = int(RATE / 10)  # 100ms
    #q = Queue()
    #l = Lock()
    #p = Process(target=streaming, args=(q,l,))
    #p.start()
    #while True:
    #    pass
    #    #print('test: {}'.format(q.get()))
    #    #query = q.get()
    #    #print(query)
    #    #time_query = string_to_ffmpeg_format(query)
    #    #print(time_query)
    #    
    #p.terminate()
    #p.join()
    project = Project('test')
    project.init()
    project.crop_mode('pedistrians')
    project.crop_mode('pedistrians').crop('test', '00:00:10', '00:00:3')
    project.crop_mode('pedistrians').crop('test2', '00:00:20', '00:00:5')
    project.crop_mode('pedistrians').crop('test3', '00:00:50', '00:00:15')
