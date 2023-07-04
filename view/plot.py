# import necessary modules
import pandas as pd
import numpy as np
import random
import librosa
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.colors import Normalize

# function to plot dataset distribution
def distribution(dataset):
    # create figure and axes (left: emotion distribution, right: duration distribution)
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))

    # group the dataset by label and plot the distribution
    dataset.groupby(["Label"]).size().plot(kind = "bar", ax = ax[0])
    ax[0].set_title("Dataset Emotion Distribution", size = 14)
    ax[0].set_ylabel("number of samples")

    # group the dataset by label and plot the duration distribution using violin plot
    sns.violinplot(x = dataset["Label"], y = dataset["Duration"], linewidth = 1, ax = ax[1])
    ax[1].set_title("Duration Distribution per Emotion")
    ax[1].set_ylabel("duration (second)")
    
    # show the plot
    st.pyplot(fig)

# function to plot random dataset picked
def show_random_plot(df):
    # create figure and axes (left: waveform, middle: spectrogram, right: mfcc)
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 4))
    # pick random index from dataset
    index = random.randint(0, df.shape[0])

    # load the audio file based on the path
    y, sr = librosa.load(df.Path[index], sr = 22050)

    # plot the waveform
    librosa.display.waveshow(y, sr = sr, ax = ax[0])
    ax[0].set_title("Waveform")

    # plot the spectrogram with fundamental frequency
    f0, _, _, = librosa.pyin(y, sr = sr, fmin = 50, fmax = 1500, frame_length = 2048)
    timepoints = np.linspace(0, df.Duration[index], num = len(f0), endpoint = False)
    x_stft = np.abs(librosa.stft(y))
    x_stft = librosa.amplitude_to_db(x_stft, ref = np.max)
    librosa.display.specshow(x_stft, sr = sr, x_axis = "time", y_axis = "log", ax = ax[1])
    ax[1].plot(timepoints, f0, color = "cyan", linewidth = 4)
    ax[1].set_title("Spectrogram with Fundamental Frequency")

    # plot the mfcc
    x_mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 20)
    librosa.display.specshow(x_mfcc, sr = sr, x_axis = "time", norm = Normalize(vmin = -50, vmax = 50), ax = ax[2])
    ax[2].set_title("MFCC")

    # show the path of picked audio file as title
    plt.suptitle(df.Path[index], size = 14)
    # use tight layout to prevent overlapping
    plt.tight_layout()
    
    # show the plot
    st.pyplot(fig)

# function to plot mfcc before and after equalization
def show_mfcc(df_mfcc_path, array_mfcc):
    # create figure and axes (left: mfcc before augmentation, right: mfcc after equalization)
    fig, ax = plt.subplots(nrows = 1, ncols =  2, figsize = (20, 4))

    # load the audio file based on the path
    y, sr = librosa.load(df_mfcc_path, sr = 22050)
    # plot the mfcc before augmentation
    x_mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 30)
    librosa.display.specshow(x_mfcc, sr = sr, x_axis = "time", norm = Normalize(vmin = -50, vmax = 50), ax = ax[0])
    ax[0].set_title("Before")

    # plot the mfcc after equalization
    librosa.display.specshow(array_mfcc, sr = sr, x_axis = "time", norm = Normalize(vmin = -50, vmax = 50), ax = ax[1])
    ax[1].set_title("After")

    # show the path of picked audio file as title
    plt.suptitle(df_mfcc_path, size = 14)

    # show the plot
    st.pyplot(fig)


# function to plot the model's training history
def plot_history(history):
    # create figure and axes (left: loss, right: accuracy)
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 5))
    plt.suptitle("CNN with MFCCs", size = 14)

    # make dataframe from training history
    results = pd.DataFrame(history.history)
    # plot the loss and validation loss for last 3 epochs
    results[["loss", "val_loss"]].plot(ax = ax[0])
    val_loss_mean = np.mean(history.history["val_loss"][-3:])
    ax[0].set_title(f"Validation loss {round(val_loss_mean, 3)} (mean last 3)")

    # plot the accuracy and validation accuracy for last 3 epochs
    results[["accuracy", "val_accuracy"]].plot(ax = ax[1])
    val_accuracy_mean = np.mean(history.history["val_accuracy"][-3:])
    ax[1].set_title(f"Validation accuracy {round(val_accuracy_mean, 3)} (mean last 3)")

    # show the plot
    st.pyplot(fig)

# function to plot the confusion matrix
def confusion_matrix(pred, y_te):
    # prepare numbered labels from string labels
    labels = {"sad": 0, "happy": 1}

    # create figure and axes (left: confusion matrix (count), right: confusion matrix (ratio))
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    # plot the confusion matrix (count)
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels = labels, ax = ax[0])
    ax[0].set_title("Confusion Matrix (count)", size = 14)

    # plot the confusion matrix (ratio)
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels = labels, normalize = "true", ax = ax[1])
    ax[1].set_title("Confusion Matrix (ratio)", size = 14)

    # show the plot
    st.pyplot(fig)
