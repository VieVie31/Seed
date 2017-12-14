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

from skimage import transform, io

from sklearn.utils import class_weight


IM_SIZE = (75, 75, 3)

def imread(path):
    return io.imread(path)

def imlabel(path):
    return path.split('/')[-2]

L = []

im_train = glob('./train/*/*.png')

for im_path in tqdm(im_train):
    feats = transform.resize(imread(im_path), IM_SIZE)
    label = imlabel(im_path)
    L.append((feats, label))

x_data, x_label = zip(*L)

#one hot encoding of labels
classes = sorted(list(set(x_label)))

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


partial_vgg = vgg.get_layer('block2_pool').output

model = Conv2D(64, (3, 3), activation='elu')(partial_vgg)
model = Conv2D(64, (3, 3), activation='elu')(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(64, (3, 3), activation='elu')(model)
model = Conv2D(64, (3, 3), activation='elu')(model)
model = MaxPooling2D((2, 2))(model)

model = Flatten()(model)

model = Dropout(.25)(model)
model = Dense(len(classes), activation='softmax')(model)

model = Model(input=[vgg.input], output=model)

to_freeze = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2']
for t_f in to_freeze:
    model.get_layer(t_f).trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

check = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.5f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

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

cw = class_weight.compute_class_weight('balanced', np.unique(y_train.argmax(1)), y_train.argmax(1))

h = model.fit_generator(
    training_generator.flow(x_train, y_train),
    class_weight=cw,
    steps_per_epoch=len(x_train) / 32,
    validation_data=validation_generator.flow(x_test, y_test),
    validation_steps=len(x_test) / 32,
    epochs=2000,
    callbacks=[early, check]
)

model.save("vgg_transfert_learning.h5")
