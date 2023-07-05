# import necessary modules
import numpy as np
import math
import librosa
import streamlit as st

# function to create mfccs from the dataframe
@st.cache_data
def create_mfcc(df):
    # prepare empty list to store the mfccs
    mfccs = []

    # loop through the dataframe and create mfccs from the audio
    for path in df.Path:
        y, sr = librosa.load(path, sr = 22550)
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 30)
        mfccs.append(mfcc)
    
    # return the mfccs
    return mfccs

# function to create mfccs from the uploaded audio
@st.cache_data
def create_audio_mfcc(librosa_audio):
    # prepare empty list to store the mfccs
    mfccs = []
    
    # loop through the uploaded audio and create mfccs from the audio
    for audio in librosa_audio:
        mfcc = librosa.feature.mfcc(y = audio, sr = 22550, n_mfcc = 30)
        mfccs.append(mfcc)

    # return the mfccs
    return mfccs

# function to resize the mfccs
@st.cache_resource
def resize_mfcc(array):
    # create empty array with shape (30, 80)
    new_mfcc = np.zeros((30, 110))
    # loop through the array and copy the value to the new array
    # if the mfcc length less than 80, the rest of the array will be filled with 0
    # if the mfcc length more than 80, the rest of the array will be ignored
    for i in range(30):
        for j in range(110):
            try:
                new_mfcc[i][j] = array[i][j]
            except IndexError:
                pass
    
    # return the new resized mfcc
    return new_mfcc

# function to create resized mfccs from the mfccs
@st.cache_data
def create_resized_mfcc(mfccs):
    # prepare empty list to store the resized mfccs
    new_mfccs = []
    # prepare variable to store the sum of mfccs column
    sum = 0

    # loop through the mfccs and create resized mfccs from the mfccs
    # also calculate the sum of mfccs column
    for mfcc in mfccs:
        new_mfccs.append(resize_mfcc(mfcc))
        sum += mfcc.shape[1]
    
    # calculate the average of mfccs column
    averate_column = math.ceil(sum / 2000)

    # return the resized mfccs and the average of mfccs column
    return new_mfccs, averate_column

# code by @tudemaha