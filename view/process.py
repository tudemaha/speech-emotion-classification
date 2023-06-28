import streamlit as st
from controller.df_dataset import dataset_df
from controller.dataset_spread import distribution
from controller.show_audio import show_random_plot

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