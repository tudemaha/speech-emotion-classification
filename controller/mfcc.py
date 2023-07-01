import numpy as np
import math
import librosa
import streamlit as st

@st.cache_data
def create_mfcc(df):
    mfccs = []

    for path in df.Path:
        y, sr = librosa.load(path, sr = 16000)
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 30)
        mfccs.append(mfcc)
    
    return mfccs

@st.cache_data
def create_audio_mfcc(librosa_audio):
    mfccs = []
    
    for audio in librosa_audio:
        mfcc = librosa.feature.mfcc(y = audio, sr = 16000, n_mfcc = 30)
        mfccs.append(mfcc)

    return mfccs

@st.cache_resource
def resize_mfcc(array):
    new_mfcc = np.zeros((30, 80))
    for i in range(30):
        for j in range(80):
            try:
                new_mfcc[i][j] = array[i][j]
            except IndexError:
                pass
    
    return new_mfcc

@st.cache_data
def create_resized_mfcc(mfccs):
    new_mfccs = []
    sum = 0

    for mfcc in mfccs:
        new_mfccs.append(resize_mfcc(mfcc))
        sum += mfcc.shape[1]
    
    averate_column = math.ceil(sum / 2000)

    return new_mfccs, averate_column