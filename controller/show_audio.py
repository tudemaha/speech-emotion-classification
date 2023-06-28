import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import librosa
import random

def show_random_plot(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 4))
    index = random.randint(0, df.shape[0])

    y, sr = librosa.load(df.Path[index], sr = 16000)

    librosa.display.waveshow(y, sr = sr, ax = ax[0])
    ax[0].set_title("Waveform")

    f0, _, _, = librosa.pyin(y, sr = sr, fmin = 50, fmax = 1500, frame_length = 2048)
    timepoints = np.linspace(0, df.Duration[index], num = len(f0), endpoint = False)
    x_stft = np.abs(librosa.stft(y))
    x_stft = librosa.amplitude_to_db(x_stft, ref = np.max)
    librosa.display.specshow(x_stft, sr = sr, x_axis = "time", y_axis = "log", ax = ax[1])
    ax[1].plot(timepoints, f0, color = "cyan", linewidth = 4)
    ax[1].set_title("Spectrogram with Fundamental Frequency")

    x_mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)
    librosa.display.specshow(x_mfcc, sr = sr, x_axis = "time", norm = Normalize(vmin = -50, vmax = 50), ax = ax[2])
    ax[2].set_title("MFCC")

    plt.suptitle(df.Path[index], size = 14)
    plt.tight_layout()
    
    st.pyplot(fig)