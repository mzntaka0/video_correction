# -*- coding: utf-8 -*-
"""
"""
import os
import sys
# -*- coding: shift-jis -*-
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""

# [START import_libraries]
from __future__ import division

import os
import re
import sys
import subprocess
import time
import yaml
from collections import OrderedDict
from multiprocessing import Process, Queue
try:
    from bpdb import set_trace
except ImportError:
    from pdb import set_trace

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import cv2

from video_edit_utils import string_to_ffmpeg_format

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

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

    def render_mode(self, video_name, order_list=None, ext='mov'):
        video_path = os.path.join(self.project_root, 'raw', '{}.{}'.format(video_name, ext))
        if not os.path.exists(video_path):
            raise FileNotFoundError('the video "{}" does not exists.'.format(os.path.split(video_path)[1]))
        cropped_dir = os.path.join(self.project_root, 'cropped', video_name + '_cropped')
        rendered_dir = os.path.join(self.project_root, 'rendered', video_name + '_rendered')
        return CropVideo(video_path, cropped_dir=cropped_dir, rendered_dir=rendered_dir)


class CropVideo:
    """
    this class operates the videos on the disk using ffmpeg, not putting on the memory(in case of huge video data.)
    Args:
        - video_path[str]: the target video file path
        - cropped_dir[str](option): just in case you want to specify the path of cropped videos.
    """

    def __init__(self, video_path, cropped_dir=None, rendered_dir=None):
        self._target_video = video_path
        if cropped_dir:
            self.cropped_dir = cropped_dir
        else:
            _splited_path = os.path.split(self._target_video)
            self.cropped_dir = os.path.join(_splited_path[0], os.path.splitext(_splited_path[1])[0] + '_cropped')
        if rendered_dir:
            self.rendered_dir = rendered_dir
        else:
            _splited_path = os.path.split(self._target_video)
            self.rendered_dir = os.path.join(_splited_path[0], os.path.splitext(_splited_path[1])[0] + '_rendered')
        if not os.path.exists(self.cropped_dir):
            os.makedirs(self.cropped_dir)
        if not os.path.exists(self.rendered_dir):
            os.makedirs(self.rendered_dir)
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
                reorderd_dict[video_name] = self.cropped_video_dict[video_name]
            except KeyError:
                ans = input('The video name is not in the cropped videos. Continue?(else Exit)[y/n]')
                if not ans == 'y':
                    sys.exit(0)
                else:
                    continue
        self.cropped_video_dict = reorderd_dict


    def render(self, rendered_name=None, cropped_video_dict=None, order_list=None, rendered_dir=None, ext='mov'):
        """
        save the video chained cropped videos, after cropping these

        input:
            - order_list[list](option): video_name list which you wanna order to render
        """
        if order_list:
            self._reorder_cropped_video_dict(order_list)

        rendering_video_cmd = list()

        if cropped_video_dict:
            rendering_video_num = len(cropped_video_dict)
            for video_path in cropped_video_dict.values():
                print(video_path)
                rendering_video_cmd.append('-i')
                rendering_video_cmd.append(video_path)
        else:
            rendering_video_num = len(self.cropped_video_dict)
            for video_path in self.cropped_video_dict.values():
                rendering_video_cmd.append('-i')
                rendering_video_cmd.append(video_path)

        _splited_path = os.path.split(self._target_video)
        rendered_dir = os.path.join(_splited_path[0].replace('raw/', ''), 'rendered', os.path.splitext(_splited_path[1])[0].replace('raw/', '') + '_rendered')
        if not os.path.exists(rendered_dir):
            os.makedirs(rendered_dir)
        rendered_name = os.path.splitext(_splited_path[1])[0].replace('raw/', '') + 'rendered_' + str(len(os.listdir(rendered_dir)))
        _output_path = os.path.join(rendered_dir, rendered_name)
        cmd = ['ffmpeg'] + rendering_video_cmd + ['-filter_complex', 'concat=n={}:v=1:a=1'.format(rendering_video_num), '{}.{}'.format(_output_path, ext)]
        print(cmd)
        subprocess.call(cmd)
        subprocess.call('open {}.{}'.format(_output_path, ext), shell=True)


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)
# [END audio_stream]


def listen_print_loop(responses, q=None, l=None):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            num_chars_printed = len(transcript)
        else:
            # output text
            sys.stdout = open('text.txt', 'w')
            print(transcript + overwrite_chars)
            sys.stdout.close()
            # restart stdout
            sys.stdout = sys.__stdout__
            sys.stdout.flush()

            if l:
                l.acquire()
            if q:
                q.put(transcript + overwrite_chars)
            if l:
                l.release()

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit|end)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0


def streaming(q=None, l=None):
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'ja-JP'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use.
            script = listen_print_loop(responses, q)
    except:
        streaming(q)



if __name__ == '__main__':
    #stream()
    project = Project('test')
    project.init()
    cropping = project.crop_mode('tmim_sample_movie')
    q = Queue()
    p = Process(target=streaming, args=(q,))
    p.start()
    cnt = 0
    #try:
    while True:
        queue = q.get()
        if queue == '�����_�[':
            video_dict = cropping.cropped_video_dict
            print(video_dict)
            project.render_mode('tmim_sample_movie').render(cropped_video_dict=video_dict)
        try:
            time_queue = string_to_ffmpeg_format(queue)
            print(time_queue)
            cropping.crop('test{}'.format(cnt), time_queue[0], time_queue[1])
            cnt += 1
        except:
            pass
        if queue == '�G���h':
            break
    #except Exception as e:
    #    print('Stopping...Try again. {}'.format(e))
    #    pass

    p.terminate()
    p.join()


if __name__ == '__main__':
    pass

