import numpy as np
import streamlit as st

from model.load_audio import load
from controller.mfcc import create_audio_mfcc, create_resized_mfcc
from controller.cnn import make_prediction

def checking(uploaded_files):
    audio_librosa = load(uploaded_files)
    mfccs = create_audio_mfcc(audio_librosa)
    new_mfccs, _ = create_resized_mfcc(mfccs)

    new_mfccs = np.array([i for i in new_mfccs])
    new_mfccs = (new_mfccs - st.session_state["x_mean"]) / st.session_state["x_std"]
    new_mfccs = new_mfccs[..., None]

    model = st.session_state["model"]
    pred = make_prediction(model, new_mfccs)

    return pred