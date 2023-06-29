import streamlit as st
import math
import random

from controller.df_dataset import dataset_df
from controller.dataset_spread import distribution
from controller.show_audio import show_random_plot
from controller.mfcc import create_mfcc, resize_mfcc, show_mfcc
from service.split_data import make_train_test_split
from controller.cnn_train import train, test, plot_history
from controller.plot_conf_matrix import plot_confusion_matrix

# title for the web
title = 'Machine Modeling - Speech Emotion Classification'

# setup the web configuration
st.set_page_config(layout='wide', page_title=title, menu_items={
    'About': f"""
    ### {title}
    Made with :heart: by Group 1 in Class A  
    Introduction to Multimedia Data Processing Subject
    """
})

st.title("Machine Modeling")

st.write("### Training Dataset")
df = dataset_df('dataset')
st.dataframe(df, 500)

st.write("Datased have been loaded, let's preprocess them!")
st.write(st.session_state)
st.button('Preprocessing', key = 'preprocessing')

def preprocessing():
    st.write("### Sample Distribution")
    distribution(df)

    st.write("### Random Pick Sample")
    happy_df = df.loc[df["Label"] == "happy"].reset_index(drop = True)
    sad_df = df.loc[df["Label"] == "sad"].reset_index(drop = True)

    st.write("#### Happy")
    show_random_plot(happy_df)
    st.write("#### Sad")
    show_random_plot(sad_df)

    mfccs = create_mfcc(df)
    new_mfccs = []
    sum = 0

    st.write("#### Average MFCCs Column")
    for mfcc in mfccs:
        new_mfccs.append(resize_mfcc(mfcc))
        sum += mfcc.shape[1]
    st.write(math.ceil(sum / 1600))

    index = random.randint(0, df.shape[0])
    st.write("#### MFCC Comparison")
    show_mfcc(df.Path[index], new_mfccs[index])

# x_tr, y_tr, x_va, y_va, x_te, y_te = make_train_test_split(new_mfccs, df)

# st.write("#### Training Process")
# trained_model, history = train(x_tr, y_tr, x_va, y_va)
# plot_history(history)

# st.write("#### Testing Process")
# test(trained_model, x_te, y_te)

# st.write("### Confusion Matrix")
# plot_confusion_matrix(trained_model, x_te, y_te)