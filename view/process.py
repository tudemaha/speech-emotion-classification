import streamlit as st
import random

from controller.df_dataset import dataset_df
from controller.mfcc import create_mfcc, create_resized_mfcc
from controller.cnn import train, test
from service.split_data import make_train_test_split
from view.plot import distribution, show_random_plot, plot_confusion_matrix, show_mfcc, plot_history

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

if "preprocessing" not in st.session_state: st.session_state["preprocessing"] = False
if "train" not in st.session_state: st.session_state["train"] = False
if "test" not in st.session_state: st.session_state["test"] = False

def start():
    st.title("Machine Modeling")

    st.write("### Training Dataset")
    global df
    df = dataset_df('dataset')
    st.dataframe(df, 500)

    st.write("Dataset have been loaded, let's preprocess them!")
    if st.button("Preprocessing"):
        st.session_state["preprocessing"] = True
    
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
    global new_mfccs
    new_mfccs, average_cols = create_resized_mfcc(mfccs)

    st.write("### Average MFCCs Column")
    st.metric("Average", average_cols)

    index = random.randint(0, df.shape[0])
    st.write("### MFCC Comparison")
    show_mfcc(df.Path[index], new_mfccs[index])

def start_train():
    global x_te, y_te
    global trained_model

    x_tr, y_tr, x_va, y_va, x_te, y_te = make_train_test_split(new_mfccs, df)

    st.write("#### MFCCs Distribution")
    col1, col2, col3 = st.columns(3)
    col1.metric("Train", str(x_tr.shape))
    col2.metric("Validation", str(x_va.shape))
    col3.metric("Test", str(x_te.shape))

    st.write("#### Training Process")
    trained_model, history = train(x_tr, y_tr, x_va, y_va)
    plot_history(history)

def start_test():
    st.write("#### Testing Process")
    loss, accuracy = test(trained_model, x_te, y_te)

    col1, col2 = st.columns(2)
    col1.metric("Loss", "{:.2f}".format(loss))
    col2.metric("Accuracy", "{:.2f}%".format(accuracy))

    st.write("### Confusion Matrix")
    plot_confusion_matrix(trained_model, x_te, y_te)

if __name__ == "__main__":
    start()

if st.session_state["preprocessing"]:
    preprocessing()
    st.write("The dataset have been preprocessed, let's train them using CNN!")
    if st.button("Training"):
        st.session_state["train"] = True

if st.session_state["train"]:
    start_train()
    st.write("Training model fisinh, let's check the accuration with testing dataset!")
    if st.button("Testing"):
        st.session_state["test"] = True

if st.session_state["test"]:
    start_test()