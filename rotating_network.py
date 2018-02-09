import data
# Now import real libs
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Keras
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, InceptionResNetV2, Xception

im_size = (224, 224, 3)


from generator import CustomImageDataGenerator

def preprocess():
    """
    Returns
    ------------
    training_generator
    test_generator
    mean for normalization
    std for normalization
    """
    dataset = data.load("../train", im_size)
    x, y = zip(*dataset)
    r = data.onehot_label(y)
    y = list(map(lambda k: r[k], y))
    x, m, s = data.normalize(x)
    (x_train, y_train), (x_test, y_test) = data.train_val_test_split((x, y), prc_test=.3, random_state=42)
    training_generator = CustomImageDataGenerator(
            x_train[0].shape,
            .5,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=80,
            width_shift_range=.3,
            height_shift_range=.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.5,
            shear_range=0.5,
            fill_mode="reflect"
    )
    test_generator = CustomImageDataGenerator(
            x_train[0].shape,
            0,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=.0,
            height_shift_range=.0,
            horizontal_flip=False,
            vertical_flip=False,
            zoom_range=0,
            shear_range=0,
            fill_mode="reflect"
    )
    return training_generator, (x_train, y_train), test_generator, (x_test, y_test), m, s


def build_model():
    rot0 = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )

    rot72 = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )

    rot144 = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )

    rot216 = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )

    concat = keras.layers.concatenate([rot0, rot72, rot144, rot216])
    norm = BatchNormalization()(concat)
    drop = Dropout(.25)(norm)
    dense = Dense(128, activation="relu")(drop)
    drop = Dropout(.25)(dense)
    out = Dense(12, activation="softmax")(drop)
    model = Model(input=[rot0.input, rot72.input, rot144.input, rot216.input], output=model)
    return model
