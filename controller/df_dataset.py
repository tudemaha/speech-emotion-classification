# import necessary modules
import pandas as pd
import streamlit as st

# import app module (model)
from model.load_dataset import load

# function to create dataframe from dataset
@st.cache_data
def dataset_df(path):
    # load the dataset
    dataset_dict = load(path)
    # create dataframe from the dataset
    df = pd.DataFrame(dataset_dict)
    # rename the columns
    df.columns = ["Path", "Label", "Duration"]
    # return the dataframe
    return df