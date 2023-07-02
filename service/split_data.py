# import neccessary modules
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

# function to split the dataset into train, validation, and test set
@st.cache_data
def make_train_test_split(x, y):
    # replace string label with number
    y["Label"].replace({"sad": 0, "happy": 1}, inplace = True)
    # get the values from dataframe
    y = y.Label.values

    # copy the dataset input
    x = x.copy()

    # split the dataset into train and test set (80:20)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 0)
    # split the train set into train and validation set (80:20)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size = 0.2, shuffle = True, random_state = 0)

    # convert the dataset into numpy array
    x_tr = np.array([i for i in x_tr])
    x_va = np.array([i for i in x_va])
    x_te = np.array([i for i in x_te])

    # get the mean and standard deviation of the train set
    tr_mean = np.mean(x_tr, axis = 0)
    tr_std = np.std(x_tr, axis = 0)

    # save the mean and standard deviation of the train set to session state
    st.session_state["x_mean"] = tr_mean
    st.session_state["x_std"] = tr_std

    # normalize the dataset
    x_tr = (x_tr - tr_mean) / tr_std
    x_va = (x_va - tr_mean) / tr_std
    x_te = (x_te - tr_mean) / tr_std

    # add channel dimension to the dataset
    x_tr = x_tr[..., None]
    x_va = x_va[..., None]
    x_te = x_te[..., None]

    # return the splitted dataset
    return x_tr, y_tr, x_va, y_va, x_te, y_te
