#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ******************************************************
#         @author: Haifeng CHEN - optical.dlz@gmail.com
# @date (created): 2016-05-12 16:35
#           @file: analyzer.py
#          @brief:
#       @internal:
#        revision: 4
#   last modified: 2017-04-10 13:54:52
# *****************************************************

import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft


def calc_powerspectrum(data, window, wsize=512):
    """
    Calculate power spectrum of the input data

    Parameters:
        data: 1D numpy vector
        window: The type of window to create. See scipy.signal.get_window for
        more details
        wsize: The number of samples in the window.
    """
    if not data.ndim == 1:
        raise ValueError('Only 1D numpy data is supported')
    if wsize > len(data):
        return None
    # normalize the audio data first
    d_norm = data.astype('float')/np.max(data)
    w = get_window(window, wsize, fftbins=True)
    half = wsize // 2
    n = 0
    start = 0
    power_spectrum = np.zeros(wsize)
    while start + wsize <= len(d_norm):
        # apply window function
        d = np.multiply(d_norm[start:start+wsize], w)
        # normalization for numpy fft - norm='ortho'
        ps = np.abs(fft(d))**2
        power_spectrum[0:half] += ps[0:half]
        # maintaine ...
        start += half
        n += 1
    # get decimal value
    with np.errstate(divide='ignore'):
        # normalize and convert to db, window size may not be needed if
        # fft output was divided by 1/sqrt(N)
        processed = 10 * np.log10(power_spectrum/wsize/n)
    inf_idx = np.isneginf(processed)
    processed[inf_idx] = 0
    # normalize to 0dB peak
    processed -= np.max(processed)
    return processed

def cubic_interplote(d, x):
    """
    Finds the degree-three polynomial which best fits these points and
    returns the value of this polynomilal at given value x.    
    """
    if not len(d) == 4:
        raise ValueError('Insufficient number of data')
    a = d[0]/-6.0 + d[1]/2.0 - d[2]/2.0 + d[3]/6.0
    b = d[0] - 5.0*d[1]/2.0 + 2.0*d[2] - d[3]/2.0
    c = -11.0*d[0]/6.0 + 3.0*d[1] - 3.0*d[2]/2.0 + d[3]/3.0
    ed = d[0]
    
    xx = x * x
    xxx = xx * x
    
    return a * xxx + b * xx + c * x + ed
    
def cubic_maximize(d):
    """
    Finds the maximal value by cubic interpolate.
    Returns (x, y), position and maximum value.
    """
    # find coefficients of cubic
    if not len(d) == 4:
        raise ValueError('Insufficient number of data')
    a = d[0]/-6.0 + d[1]/2.0 - d[2]/2.0 + d[3]/6.0
    b = d[0] - 5.0*d[1]/2.0 + 2.0*d[2] - d[3]/2.0
    c = -11.0*d[0]/6.0 + 3.0*d[1] - 3.0*d[2]/2.0 + d[3]/3.0
    ed = d[0]
    # derivative    
    da = 3 * a
    db = 2 * b
    dc = c
    # find zeros of derivative using quadratic equation
    discriminat = db * db - 4 * da * dc
    if discriminat < 0:
        return (None, None)  # error
    
    x1 = (-db + np.sqrt(discriminat)) / (2 * da)
    x2 = (-db - np.sqrt(discriminat)) / (2 * da)
    # the on which corresponds to local maximum in the 
    # cubic is the one we want -- the one with a negative
    # second derivative
    dda = 2 * da
    ddb = db
    if dda * x1 + ddb < 0:
        return (x1, a*x1*x1*x1 + b*x1*x1 + c*x1 + ed)
    else:
        return (x2, a*x2*x2*x2 + b*x2*x2 + c*x2 + ed)
