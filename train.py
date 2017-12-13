import keras
import keras.backend as K

import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from random import shuffle
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

IM_SIZE = (160, 160)


def imread(path):
    return cv2.imread(path)[...,::-1]

def imlabel(path):
    return path.split('/')[-2]

L = []

im_train = glob('./train/*/*.png')

for im_path in tqdm(im_train):
    feats = cv2.resize(imread(im_path), IM_SIZE)
    label = imlabel(im_path)
    L.append((feats, label))

x_data, x_label = zip(*L)

#one hot encoding of labels
classes = list(set(x_label))

def one_hot(c, n):
    o = [0] * n
    o[c] = 1
    return o

one_hot_encoder = {c: one_hot(i, len(classes)) for i, c in enumerate(classes)}

x_label = np.array(list(map(lambda s: one_hot_encoder[s], x_label)))


#split data
data = list(zip(x_data, x_label))

shuffle(data)

x_data, x_label = zip(*data)

x_train, x_test = x_data[:int(len(x_data) * .9)],  x_data[int(len(x_data) * .9):]
y_train, y_test = x_label[:int(len(x_data) * .9)], x_label[int(len(x_data) * .9):]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)


from keras.applications import VGG16

vgg = VGG16(
    input_shape=(IM_SIZE[0], IM_SIZE[0], 3),
    include_top=False,
    weights='imagenet'
)


partial_vgg = vgg.get_layer('block5_pool').output
model = GlobalMaxPooling2D()(partial_vgg)
model = Dense(len(classes), activation='sigmoid')(model)

model = Model(input=[vgg.input], output=model)

model.compile(loss='binary_crossentropy', optimizer='nadam')

check = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.5f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='max')

print(model.summary())


training_generator = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False, 
        rotation_range=180,
        width_shift_range=.1,
        height_shift_range=.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2
)

validation_generator = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False, 
        rotation_range=180,
        width_shift_range=.1,
        height_shift_range=.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2
)


h = model.fit_generator(
    training_generator.flow(x_train, y_train),
    steps_per_epoch=len(x_train) / 32,
    validation_data=validation_generator.flow(x_test, y_test),
    validation_steps=len(x_test) / 32,
    epochs=20000,
    verbose=0,
    callbacks=[early, check]
)

model.save("vgg_transfert_learning.h5")








