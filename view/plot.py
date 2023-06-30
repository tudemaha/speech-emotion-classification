import pandas as pd
import numpy as np
import random
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

from controller.cnn import predict

def distribution(dataset):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))

    dataset.groupby(["Label"]).size().plot(kind = "bar", ax = ax[0])
    ax[0].set_title("Dataset Emotion Distribution", size = 14)
    ax[0].set_ylabel("number of samples")

    sns.violinplot(x = dataset["Label"], y = dataset["Duration"], linewidth = 1, ax = ax[1])
    ax[1].set_title("Duration Distribution per Emotion")
    ax[1].set_ylabel("duration (second)")
    
    st.pyplot(fig)

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


def plot_history(history):
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 5))
    plt.suptitle("CNN with MFCCs", size = 14)

    results = pd.DataFrame(history.history)
    results[["loss", "val_loss"]].plot(ax = ax[0])
    val_loss_mean = np.mean(history.history["val_loss"][-3:])
    ax[0].set_title(f"Validation loss {round(val_loss_mean, 3)} (mean last 3)")

    results[["accuracy", "val_accuracy"]].plot(ax = ax[1])
    val_accuracy_mean = np.mean(history.history["val_accuracy"][-3:])
    ax[1].set_title(f"Validation accuracy {round(val_accuracy_mean, 3)} (mean last 3)")

    st.pyplot(fig)

def plot_confusion_matrix(model, x_te, y_te):
    labels = {"sad": 0, "happy": 1}

    pred = predict(model, x_te)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels = labels, ax = ax[0])
    ax[0].set_title("Confusion Matrix (count)", size = 14)

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels = labels, normalize = "true", ax = ax[1])
    ax[1].set_title("Confusion Matrix (ratio)", size = 14)

    st.pyplot(fig)
