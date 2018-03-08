import keras
from keras.models import Model, Sequential
from keras.layers import Reshape, Dense, Input, Conv2D, Dropout, MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization


def two_class():
    input_size = (300, 300)
    model = Sequential([
        Conv2D(16, (1, 1), input_shape=input_size, activation="elu"),
        Conv2D(256, (3, 3), activation="elu"),
        Conv2D(256, (3, 3), activation="elu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="elu"),
        Conv2D(128, (3, 3), activation="elu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="elu"),
        Conv2D(128, (3, 3), activation="elu"),
        Conv2D(128, (3, 3), activation="elu"),
        Conv2D(11, (1, 1), activation="relu")
        Flatten(),
        Dense(900, activation="elu"),
        Dropout(.25),
        Dense(900, activation="elu"),
        Reshape((30, 30, 1)),
        Conv2D(11, (1, 1), activation="relu"),
        Flatten(),
        Dense(400, activation="elu"),
        Dropout(.25),
        Dense(400, activation="elu"),
        Reshape()(20, 20, 1)),
        Conv2D(11, (1, 1), activation="relu"),
        Dense(100, activation="elu"),
        Dropout(.25),
        Dense(100, activation="relu"),
        Reshape((10, 10, 1)),
        Conv2D(11, (1, 1), activation="relu"),
        AveragePooling2D((2, 2)),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])
    return model
