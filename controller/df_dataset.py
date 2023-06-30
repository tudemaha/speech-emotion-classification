import pandas as pd
import streamlit as st

from model.load_dataset import load

@st.cache_data
def dataset_df(path):
    dataset_dict = load(path)
    df = pd.DataFrame(dataset_dict)
    df.columns = ["Path", "Label", "Duration"]
    return df