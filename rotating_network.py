import data
# Now import real libs
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Keras
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, InceptionResNetV2, Xception

im_size = (224, 224, 3)


from generator import MultipleInputData

def rotate_90(im):
    return scipy.ndimage.rotate(im, 90)

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
    training_generator = MultipleInputData(
            nb=4,
            transf=rotate_90,
            image_shape=x_train[0].shape,
            prob_transfo=.5,
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
    test_generator = MultipleInputData(
            nb=4,
            transf=rotate_90,
            image_shape=x_train[0].shape,
            prob_transfo=0,
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
    model = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )
    model = MaxPooling2D(pool_size=(7, 7))(model)
    model = Flatten()(model)

    model = Model(input=[model.input], output=model.output)

    rot0_in = Input(shape=im_size, name="rot0")
    rot0 = model(rot0_in)

    rot90_in = Input(shape=im_size, name="rot90")
    rot90 = model(rot90_in)

    rot180_in = Input(shape=im_size, name="rot180")
    rot180 = model(rot180_in)

    rot270_in = Input(shape=im_size, name="rot270")
    rot270 = model(rot270_in)

    concat = keras.layers.concatenate([rot0, rot90, rot180, rot270])
    norm = BatchNormalization()(concat)
    drop = Dropout(.25)(norm)
    dense = Dense(128, activation="relu")(drop)
    drop = Dropout(.25)(dense)
    dense = Dense(128, activation="relu")(drop)
    out = Dense(12, activation="softmax")(dense)

    model = Model(input=[rot0.input, rot90.input, rot180.input, rot270.input], output=model)
    return model
