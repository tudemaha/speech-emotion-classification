from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st
import matplotlib.pyplot as plt
from service.confucion_matrix import confusion_matrix

def plot_confusion_matrix(model, x_te, y_te):
    labels = {"sad": 0, "happy": 1}

    pred = confusion_matrix(model, x_te)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels = labels, ax = ax[0])
    ax[0].set_title("Confusion Matrix (count)", size = 14)

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels = labels, normalize = "true", ax = ax[1])
    ax[1].set_title("Confusion Matrix (ratio)", size = 14)

    st.pyplot(fig)
