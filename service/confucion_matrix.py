import numpy as np

def confusion_matrix(model, x_te):
    predictions = model.predict(x_te)

    pred = []

    for i in predictions:
        pred.append(np.argmax(i))

    return pred