#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ******************************************************
#         @author: Haifeng CHEN - optical.dlz@gmail.com
# @date (created): 2016-05-12 14:25
#           @file: recorder.py
#          @brief: audio recording utilities
#       @internal:
#        revision: 4
#   last modified: 2016-07-12 09:54:51
# *****************************************************

import os, sys, time, threading
import pyaudio, math, wave, audioop
from collections import deque


def get_noise_level(stream, nsamples=50, chunk=1024):
    """
        Gets average audio intensity of your mic sound.
        The average is the avg of the 20% largest intensities recorded.
    """
    values = [math.sqrt(abs(audioop.avg(stream.read(chunk), 4)))
              for x in range(nsamples)]
    values = sorted(values, reverse=True)
    r = sum(values[:int(nsamples * 0.2)])/int(nsamples * 0.2)
    return r


def listen_for_signal(stream, event=None, cb_fun=None, **kwargs):
    """ listen for signal from mic """
    # get parameters
    max_len = kwargs.get('max_len', 5)
    chunk = kwargs.get('chunk', 1024)
    rate = kwargs.get('rate', 48000)
    silence_ = kwargs.get('silence_limit', 1.0)
    prev_ = kwargs.get('prev_audio', 0.5)
    thr = kwargs.get('threshold', 2500)
    rel = int(rate/chunk)
    max_niter = max_len * rel
    slid_win = deque(maxlen=int(silence_ * rel))
    prev_audio = deque(maxlen=int(prev_ * rel))
    started = False
    audio = []
    # current chunk of audio data
    cur_data = ''
    # maximal data length
    niter = 0
    # counter for callback function
    counter = 0
    while True:
        if niter >= max_niter:
            break
        if event is not None:
            if event.is_set():
                break
        cur_data = stream.read(chunk)
        avg_val = math.sqrt(abs(audioop.avg(cur_data, 4)))
        slid_win.append(avg_val)
        if cb_fun is not None and counter % 8 == 0:
            cb_fun('intensity', int(avg_val))
        if sum([x > thr for x in slid_win]):
            if not started:
                started = True
                if cb_fun is not None:
                    cb_fun('status', 1)
            niter += 1
            audio.append(cur_data)
        elif started:
            audio = list(prev_audio) + audio
            # started = False
            # slid_win = deque(maxlen=silence_ * rel)
            # prev_audio = deque(maxlen=prev_ * rel)
            # audio = []
            break
        else:
            prev_audio.append(cur_data)
        counter += 1

    output_dir = kwargs.get('output_dir', 'records')
    status_code = 0
    if event is None:
        save_wave(audio, output_dir, rate=rate)
    elif not event.is_set():
        save_wave(audio, output_dir, rate=rate)
    else:
        status_code = 2

    if cb_fun is not None:
        cb_fun('status', status_code)

    return audio


def save_wave(data, output_dir, dsize=2, rate=48000):
    """ Saves mic data to WAV file. Returns filename of saved file """
    t = time.localtime(time.time())
    sub_folder = os.path.join(output_dir, time.strftime('%m-%d-%Y', t))
    filename = os.path.join(sub_folder,
                            'output_' + time.strftime('%H-%M-%S', t))
    # writes data to WAV file
    data = bytes.join(b'', data)
    try:
        os.makedirs(sub_folder)
    except:
        pass    
    try:
        wf = wave.open(filename + '.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(dsize)
        wf.setframerate(rate)
        wf.writeframes(data)
        wf.close()
        return filename + '.wav'
    except:
        return ''
