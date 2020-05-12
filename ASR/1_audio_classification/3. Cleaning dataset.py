# -*- coding: utf-8 -*-
"""
Created on Tue May 12 07:43:48 2020

@author: Susmoy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 12 00:06:21 2020

@author: Susmoy
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import os
import tqdm
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle("Time Series", size=16)
    i=0
    
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle("Fourier Transforms", size=16)
    i=0
    
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle("Filter Bank Coefficients", size=16)
    i=0
    
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle("Mel Frequency Ceptrum Coefficients", size=16)
    i=0
    
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


def envelope(y, rate, threshold):
    mask = [] # true false values
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

main_dir = "C://Users//Susmoy//Desktop//Jarvis//ASR//1_audio_classification"
dataDir = os.path.join(main_dir, "wavfiles")

signals = {}
fft = {}
fbank = {}
mfccs = {}


for dirs in os.listdir(dataDir):
    #print(dirs)
    for i in os.listdir(os.path.join(dataDir, dirs)):
        
        path = os.path.join(dataDir, dirs)
        path = os.path.join(path, i)
        #print(path)
        signal, rate = librosa.load(path, sr=44100)
        
        #Cleaning data
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        
        signals[dirs] = signal
        
        # calculating fft, mfcc, filterbank
        fft[dirs] = calc_fft(signal, rate)
        
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[dirs] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[dirs] = mel
        break


plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

# Cleaning dataset
if len(os.listdir('clean')) == 0:
    for dirs in os.listdir(dataDir):
    #print(dirs)
        for i in os.listdir(os.path.join(dataDir, dirs)):
            path = os.path.join(dataDir, dirs)
            path = os.path.join(path, i)
            #print(path)
            signal, rate = librosa.load(path, sr=16000)
            mask = envelope(signal, rate, 0.0005)
            cleanPath = os.path.join(main_dir, 'clean')
            wavfile.write(filename='clean/'+i, rate = rate, data=signal[mask])
