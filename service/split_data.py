from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

@st.cache_data
def make_train_test_split(x, y):
    y["Label"].replace({"sad": 0, "happy": 1}, inplace = True)
    y = y.Label.values

    x = x.copy()

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 0)
    x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size = 0.2, shuffle = True, random_state = 0)

    x_tr = np.array([i for i in x_tr])
    x_va = np.array([i for i in x_va])
    x_te = np.array([i for i in x_te])

    tr_mean = np.mean(x_tr, axis = 0)
    tr_std = np.std(x_tr, axis = 0)

    st.session_state["x_mean"] = tr_mean
    st.session_state["x_std"] = tr_std

    x_tr = (x_tr - tr_mean) / tr_std
    x_va = (x_va - tr_mean) / tr_std
    x_te = (x_te - tr_mean) / tr_std

    x_tr = x_tr[..., None]
    x_va = x_va[..., None]
    x_te = x_te[..., None]

    return x_tr, y_tr, x_va, y_va, x_te, y_te
