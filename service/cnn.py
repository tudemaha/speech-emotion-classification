from tensorflow import keras
from keras.layers import (Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D)
from keras import initializers

def create_model(x_tr):
    model = keras.Sequential()
    model.add(Conv2D(filters = 64, kernel_size = 5, strides = (2, 2), activation = "relu", input_shape = x_tr.shape[1:]))
    model.add(MaxPool2D(pool_size = 2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = 4, strides = (2, 1), activation = "relu"))
    model.add(MaxPool2D(pool_size = 2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units = 7, activation = "softmax"))
    model.summary()

    return model