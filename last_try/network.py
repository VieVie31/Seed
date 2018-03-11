import keras
import keras.backend as K

import os

import numpy as np

from tqdm import tqdm
from glob import glob
from random import shuffle
from collections import Counter
from skimage import transform, io

#use cpu for test...
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mean = 0.275734672004
std  = 0.12763850954
IM_SIZE = (224, 224, 3)
print("STARTING")
def imread(path):
    return io.imread(path)

#load the test images
L = []
im_test = glob('../../test/*.png')
for im_path in tqdm(im_test):
    feats = transform.resize(imread(im_path), IM_SIZE)
    L.append(feats)

x_data = np.array(L)
x_data = (x_data - mean) / std

#load the classes values
def imlabel(path):
    return path.split('/')[-2]

L = []
im_train = glob('../../train/*/*.png')
for im_path in tqdm(im_train):
    label = imlabel(im_path)
    L.append(label)
classes = sorted(list(set(L)))


disc = sorted(["Loose Silky-bent", "Black-grass"])

#load model
general_model = keras.models.load_model("go.hdf5")
discri_model = keras.models.load_model("discri.hdf5")
print("PREDICTION")
#make predictiona
predicted_classes = []
i = 0
for x in x_data:
    pred = model.predict(x)
    clas = classes[pred.argmax(1)]
    if clas in disc:
        pred2 = discri_model.predict(x)
        predicted_classes.append(disc[pred2])
        i += 1 if clas != disc[pred2] else 0
    else:
        predicted_classes.append(pred.argmax(1))
print(i, "changements effectuesd")

#build the submission file
s = "file,species\n"
for p, c in zip(im_test, predicted_classes):
    s += p.split('/')[-1] + ',' + classes[c] + '\n'

f = open("submission.csv", 'w')
f.write(s)
f.close()
