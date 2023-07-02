# import necessary modules
import numpy as np
import streamlit as st

# import app module (model and controller)
from model.load_audio import load
from controller.mfcc import create_audio_mfcc, create_resized_mfcc
from controller.cnn import make_prediction

# function to check/predict the uploaded audio
def checking(uploaded_files):
    # load the uploaded audio into librosa format
    audio_librosa = load(uploaded_files)
    # create mfccs from the audio
    mfccs = create_audio_mfcc(audio_librosa)
    # create resized mfccs from the mfccs (equalize the length of mfccs)
    new_mfccs, _ = create_resized_mfcc(mfccs)

    # convert the mfccs into numpy array
    new_mfccs = np.array([i for i in new_mfccs])
    # normalize the mfccs
    new_mfccs = (new_mfccs - st.session_state["x_mean"]) / st.session_state["x_std"]
    # add channel dimension to the mfccs
    new_mfccs = new_mfccs[..., None]

    # load the model from session state
    model = st.session_state["model"]
    # make prediction from the model
    pred = make_prediction(model, new_mfccs)

    # return the prediction
    return pred