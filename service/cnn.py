# import necessary modules
from tensorflow import keras
from keras.layers import (Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D)

# function to create the model (CNN)
def create_model(x_tr):
    # create sequential model from keras
    model = keras.Sequential()
    # add first convolutional layer, max pooling layer, and batch normalization layer
    model.add(Conv2D(filters = 64, kernel_size = 5, strides = (2, 2), activation = "tanh", input_shape = x_tr.shape[1:]))
    model.add(MaxPool2D(pool_size = 2))
    model.add(BatchNormalization())
    # add second convolutional layer, max pooling layer, and batch normalization layer
    model.add(Conv2D(filters = 32, kernel_size = 4, strides = (2, 1), activation = "tanh"))
    model.add(MaxPool2D(pool_size = 2))
    model.add(BatchNormalization())
    # add flatten layer
    model.add(Flatten())
    # add first dense layer, dropout layer
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = "relu"))
    # add second dense layer, dropout layer
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = "relu"))
    # add third dense layer, dropout layer
    model.add(Dropout(0.5))
    model.add(Dense(units = 2, activation = "sigmoid"))

    # show the summary of the model
    model.summary()

    # return the model
    return model