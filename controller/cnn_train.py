from service.cnn import create_model
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

def train(x_tr, y_tr, x_va, y_va):
    model = create_model(x_tr)
    model.compile(optimizer = "Adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    earlystopping_cb = keras.callbacks.EarlyStopping(patience = 5)

    history = model.fit(
        x = x_tr,
        y = y_tr,
        epochs = 100,
        batch_size = 32,
        validation_data = (x_va, y_va),
        callbacks = [earlystopping_cb]
    )

    return model, history

def test(model, x_te, y_te):
    loss_te, accuracy_te = model.evaluate(x_te, y_te)

    st.write("Test Loss: {:.2f}".format(loss_te))
    st.write("Test Accuracy: {:.2f}%".format(100 * accuracy_te))

def plot_history(history):
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (20, 5))
    plt.suptitle("CNN with MFCCs", size = 14)

    results = pd.DataFrame(history.history)
    results[["loss", "val_loss"]].plot(ax = ax[0])
    val_loss_mean = np.mean(history.history["val_loss"][-3:])
    ax[0].set_title(f"Validation loss {round(val_loss_mean, 3)} (mean last 3)")

    results[["accuracy", "val_accuracy"]].plot(ax = ax[1])
    val_accuracy_mean = np.mean(history.history["val_accuracy"][-3:])
    ax[1].set_title(f"Validation accuracy {round(val_accuracy_mean, 3)} (mean last 3)")

    st.pyplot(fig)