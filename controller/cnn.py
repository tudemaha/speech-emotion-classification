import numpy as np
from tensorflow import keras
from sklearn.metrics import recall_score, precision_score, f1_score
import streamlit as st

from service.cnn import create_model

@st.cache_resource
def train(x_tr, y_tr, x_va, y_va):
    model = create_model(x_tr)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    earlystopping_cb = keras.callbacks.EarlyStopping(patience = 5)

    history = model.fit(
        x = x_tr,
        y = y_tr,
        epochs = 100,
        batch_size = 32,
        validation_data = (x_va, y_va),
        callbacks = [earlystopping_cb]
    )

    st.session_state["model"] = model
    return model, history

def test(model, x_te, y_te):
    loss_te, accuracy_te = model.evaluate(x_te, y_te)

    return loss_te, accuracy_te

def make_prediction(model, x_te):
    predictions = model.predict(x_te)

    pred = []
    for i in predictions:
        pred.append(np.argmax(i))
    
    return pred

def predict(model, x_te, y_te):
    pred = make_prediction(model, x_te)
    precision = precision_score(y_te, pred, average = "weighted")
    recall = recall_score(y_te, pred, average = "weighted")
    f1 = f1_score(y_te, pred, average = "weighted")
        
    return pred, precision, recall, f1