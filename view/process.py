import streamlit as st
import math
import random

from controller.df_dataset import dataset_df
from controller.dataset_spread import distribution
from controller.show_audio import show_random_plot
from controller.mfcc import create_mfcc, resize_mfcc, show_mfcc

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
df = dataset_df('dataset/training')
st.dataframe(df, 500)

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