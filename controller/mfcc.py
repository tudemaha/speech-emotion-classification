import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def create_mfcc(df):
    mfccs = []

    for path in df.Path:
        y, sr = librosa.load(path, sr = 16000)
        mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 30)
        mfccs.append(mfcc)
    
    return mfccs

def resize_mfcc(array):
    new_mfcc = np.zeros((30, 80))
    for i in range(30):
        for j in range(80):
            try:
                new_mfcc[i][j] = array[i][j]
            except IndexError:
                pass
    
    return new_mfcc

def show_mfcc(df_mfcc_path, array_mfcc):
    fig, ax = plt.subplots(nrows = 1, ncols =  2, figsize = (20, 4))

    y, sr = librosa.load(df_mfcc_path, sr = 16000)
    x_mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 30)
    librosa.display.specshow(x_mfcc, sr = sr, x_axis = "time", norm = Normalize(vmin = -50, vmax = 50), ax = ax[0])
    ax[0].set_title("Before")

    librosa.display.specshow(array_mfcc, sr = sr, x_axis = "time", norm = Normalize(vmin = -50, vmax = 50), ax = ax[1])
    ax[1].set_title("After")

    plt.suptitle(df_mfcc_path, size = 14)

    st.pyplot(fig)