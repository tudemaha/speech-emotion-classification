import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def distribution(dataset):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))

    dataset.groupby(["Label"]).size().plot(kind = "bar", ax = ax[0])
    ax[0].set_title("Dataset Emotion Distribution", size = 14)
    ax[0].set_ylabel("number of samples")

    sns.violinplot(x = dataset["Label"], y = dataset["Duration"], linewidth = 1, ax = ax[1])
    ax[1].set_title("Duration Distribution per Emotion")
    ax[1].set_ylabel("duration (second)")
    
    st.pyplot(fig)