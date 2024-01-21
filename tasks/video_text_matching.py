# -*- coding: utf-8 -*-
"""
"""
import os
import re
import sys
import json
from glob import glob
import subprocess
import datetime
import pandas as pd
from multiprocessing import Process, Queue
try:
    from bpdb import set_trace
except:
    from pdb import set_trace

import googleapiclient

from controllers.audioRecognition import post_audio, parse_speech_recognition_response
from tasks.transcribe_streaming_mic import streaming


def video2audio(video_path, ext='wav'):
    audio_path = '{}.{}'.format(os.path.splitext(video_path)[0], ext)
    cmd = 'ffmpeg -i {} {}'.format(video_path, audio_path)
    subprocess.call(cmd, shell=True)
    return audio_path


def split_audio(audio_path, duration='15'):
    splited_path = os.path.splitext(audio_path)[0] + 'splited_{}s'.format(duration)
    if not os.path.exists(splited_path):
        os.makedirs(splited_path)
    else:
        ans = input('The dirctory {} already exists. pass?[y/n]'.format(splited_path))
        if ans == 'y':
            return  splited_path

    cmd = 'ffmpeg -i {} -c copy -map 0 -f segment -segment_time {} -segment_list {}/out.csv {}/out%03d.wav'.format(
            audio_path,
            duration,
            splited_path,
            splited_path
            )
    subprocess.call(cmd, shell=True)
    return splited_path

def post_audio_respectively(splited_path, post_limit=10):
    script_dict = dict()
    for i, audio_file in enumerate(os.listdir(splited_path)):
        if not 'wav' in audio_file:
            continue
        if i > post_limit:
            break
        try:
            print(os.path.join(splited_path.replace('mono_', ''), audio_file.replace('mono_', '')))
            script_dict[audio_file] = [script[0] for script in parse_speech_recognition_response(
                    post_audio(
                        os.path.join(splited_path.replace('mono_', ''), audio_file.replace('mono_', '')),
                        API_KEY,
                        JSON_CONFIG
                        )
                    )
                    ]
        except googleapiclient.errors.HttpError as e:
            print('{} has been passed'.format(os.path.join(splited_path, audio_file)))
            print(e)
            pass
    return script_dict

def save_script_dict(splited_path, script_dict):
    script_len = len(script_dict)
    with open(os.path.join(splited_path, 'script_{}.json'.format(script_len)), 'w') as f:
        json.dump(script_dict, f)


def get_script(audio_path, API_KEY, JSON_CONFIG):
    response = post_audio(audio_path, API_KEY, JSON_CONFIG)
    clause_list = parse_speech_recognition_response(response)
    return clause_list


def string_to_word_format(string):
    splited = string.split('から')
    start = splited[0]
    print('splited: {}'.format(splited))
    print('start: {}'.format(start))
    if 'まで' in splited[1]:
        end = splited[1].split('まで')[0]
    else:
        end = splited[1]
    print('end: {}'.format(end))
    return (start, end)


def crop_from_word(video_path, splited_path, start_list, end_list, ext='mp4'):
    df = pd.read_csv(
            os.path.join(splited_path, 'out.csv'),
            header=None
            )
    start = sorted(start_list, key=lambda s: re.findall(r'[0-9]', s))[0].replace('mono_', '')
    end = sorted(end_list, key=lambda s: re.findall(r'[0-9]', s))[-1].replace('mono_', '')
    print('start: {}'.format(start))
    print('end: {}'.format(end))

    render_list = list()
    start_num = int(re.findall(r'(\d+)', start)[-1])
    end_num = int(re.findall(r'(\d+)', end)[-1])
    render_len = len(range(start_num, end_num + 1))
    _output_path = os.path.join(splited_path, 'rendered')
    if not os.path.exists(_output_path):
        os.makedirs(_output_path)
    for i in range(start_num, end_num + 1):
        render_list.append('-i')
        render_list.append(os.path.join(splited_path, 'out{0:03d}.wav'.format(i)))
    #cmd = ['ffmpeg'] + render_list + ['-filter_complex', 'concat=n={}:v=1:a=1'.format(render_len), '{}.{}'.format(_output_path, 'trimmed_{}_{}'.format(start_num, end_num), ext)]
    #print(' '.join(cmd))
    #subprocess.call(cmd)


    start_time = df[df[0] == start][1].values[0]
    end_time = df[df[0] == end][2].values[0]
    if start_time > end_time:
        start_time = df[df[0] == start][2]
        end_time = df[df[0] == end][1]

    duration_raw = datetime.timedelta(seconds=end_time - start_time)
    duration = duration_raw.seconds
    print('start time: {}'.format(start_time))
    print('end time: {}'.format(end_time))
    print('duration {}'.format(duration))

    _output = os.path.join(_output_path, 'trimmed_{:.0f}_{:.0f}.{}'.format(start_time, end_time, ext))
    cmd = [
            'ffmpeg', 
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(end_time),
            _output
            ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    subprocess.call('open {}'.format(_output), shell=True)

def search_script_json(splited_path, script_num=None):
    json_files = glob(os.path.join(splited_path, '*.json'))
    mono_files = glob(os.path.join(splited_path, 'mono_*.wav'))
    for mono in mono_files:
        os.remove(mono)
    print(json_files)
    if json_files:
        if not script_num:
            json_name = json_files[0]
        else:
            json_name = list(filter(lambda f: str(script_num) in f, json_files))[0]
        with open(json_name, 'r') as f:
            script_dict = json.load(f)
        return script_dict
    else:
        print('json not found')
        return None


    



if __name__ == '__main__':
    JSON_CONFIG = 'config/request_local.json'
    video_path = 'resources/YoutubeDL/ai.mp4'
    audio_path = 'resources/YoutubeDL/ai.wav'
    splited_path = split_audio(audio_path)
    script_dict = search_script_json(splited_path)
    if not script_dict:
        script_dict = post_audio_respectively(splited_path, post_limit=20)
        save_script_dict(splited_path, script_dict)
    print(sorted(script_dict.items(), key=lambda x: re.findall(r'[0-9]', x[0])))
    q = Queue()
    p = Process(target=streaming, args=(q,))
    p.start()
    start_list = list()
    end_list = list()
    while True:
        queue = q.get()
        print('queue: {}'.format(queue))
        try:
            crop_queue = string_to_word_format(queue)
            print(crop_queue)
            for key, val in script_dict.items():
                if crop_queue[0] in val[0]:
                    start_list.append(key)
                elif crop_queue[1] in val[0]:
                    end_list.append(key)
        except:
            pass
        if queue == 'エンド':
            break
        if start_list and end_list:
            print('got query. start cropping...')
            print('start_list: {}'.format(start_list))
            print('end_list: {}'.format(end_list))
            break
        else:
            print('did not match scripts with {}'.format(crop_queue))

    p.terminate()
    p.join()
    print(video_path)
    crop_from_word(video_path, splited_path, start_list, end_list)
