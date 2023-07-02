# import necessary modules
import numpy as np
from tensorflow import keras
from sklearn.metrics import recall_score, precision_score, f1_score
import streamlit as st

# import app module (service)
from service.cnn import create_model

# function to train the model
@st.cache_resource
def train(x_tr, y_tr, x_va, y_va):
    # create the model
    model = create_model(x_tr)
    # prepare the Adam optimizer
    optimizer = keras.optimizers.Adam()
    # compile the model
    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    # prepare early stopping callback
    earlystopping_cb = keras.callbacks.EarlyStopping(patience = 5)

    # train the model
    history = model.fit(
        x = x_tr,
        y = y_tr,
        epochs = 100,
        batch_size = 32,
        validation_data = (x_va, y_va),
        callbacks = [earlystopping_cb]
    )

    # return the model and history
    return model, history

# function to test the model
def test(model, x_te, y_te):
    # evaluate the model
    loss_te, accuracy_te = model.evaluate(x_te, y_te)

    # return the loss and accuracy
    return loss_te, accuracy_te

# function to make prediction
def make_prediction(model, x_te):
    # make prediction from trained model
    predictions = model.predict(x_te)

    # prepare empty list to store the prediction
    pred = []
    # loop through the predictions and append the prediction to the list
    for i in predictions:
        pred.append(np.argmax(i))
    
    # return the prediction
    return pred

# function to predict the test set
def predict(model, x_te, y_te):
    # make prediction
    pred = make_prediction(model, x_te)
    # calculate the precision, recall, and f1 score
    precision = precision_score(y_te, pred, average = "weighted")
    recall = recall_score(y_te, pred, average = "weighted")
    f1 = f1_score(y_te, pred, average = "weighted")
    
    # return the prediction, precision, recall, and f1 score
    return pred, precision, recall, f1

# code by @tudemaha