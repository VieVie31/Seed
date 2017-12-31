import keras
import xgboost as xgb

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import data

IM_SIZE = im_size = (224, 224, 3)

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
    (x_train, y_train), (x_test, y_test) = data.train_val_test_split((x, y))
    training_generator = ImageDataGenerator(
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
    test_generator = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=.0,
            height_shift_range=.0,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.5,
            shear_range=0.5,
            fill_mode="reflect"
    )
    return training_generator, (x_train, y_train), test_generator, (x_test, y_test), m, s

features_extractor = keras.models.load_model("xception_fine_tunned.h5")
features_extractor = Model(features_extractor.input, features_extractor.layers[-2].output)

train_gen, (x_train, y_train), test_gen, (x_test, y_test), mean, std = preprocess()
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print("Mean :", mean, "Std :", std)

x_train_features = features_extractor.predict(x_train)
x_test_features  = features_extractor.predict(x_test)


gbm = xgb.XGBClassifier(
    max_depth=3,
    n_estimators=300,
    learning_rate=0.05
).fit(x_train_features, y_train.argmax(1))

#the accuracy may be biaised because the feature extractor
#was learned with the training datas...
predictions = gbm.predict(x_test_features)
print('acc : ', (predictions == y_test.argmax(1)).mean())

#save the xgboost model
import pickle

pickle.dump(gbm, open("xgboost_model.mdl", "wb"))


### Making the submission
import os
from tqdm import tqdm
from glob import glob
from random import shuffle
from collections import Counter
from skimage import transform, io

def imread(path):
    return io.imread(path)


#load the test images
L = []
im_test = glob('../test/*.png')
for im_path in tqdm(im_test):
    feats = transform.resize(imread(im_path), IM_SIZE)
    L.append(feats)

x_data = np.array(L)
x_data = (x_data - mean) / std

#load the classes values
def imlabel(path):
    return path.split('/')[-2]


L = []
im_train = glob('../train/*/*.png')
for im_path in tqdm(im_train):
    label = imlabel(im_path)
    L.append(label)

classes = sorted(list(set(L)))

#extract the submissions features
x_data_features = features_extractor.predict(x_data)

#make the xgboost prediction
predicted_classes = gbm.predict(x_data_features)


#build the submission file
s = "file,species\n"
for p, c in zip(im_test, predicted_classes):
    s += p.split('/')[-1] + ',' + classes[c] + '\n'

f = open("submission.csv", 'w')
f.write(s)
f.close()




