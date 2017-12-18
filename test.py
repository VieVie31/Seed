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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mean = 0.275887342968
std = 0.127649436617
IM_SIZE = (160, 160, 3)

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


#load model
model = keras.models.load_model("totry.hdf5")

#make prediction
pred = model.predict(x_data)

predicted_classes = pred.argmax(1)


#build the submission file
s = "file,species\n"
for p, c in zip(im_test, predicted_classes):
    s += p.split('/')[-1] + ',' + classes[c] + '\n'

f = open("submission.csv", 'w')
f.write(s)
f.close()




