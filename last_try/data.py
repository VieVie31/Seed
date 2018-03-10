import os
import random
import collections
from glob import glob
from tqdm import tqdm

import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split

def load_specific(dirs, im_size):
    images = []
    imlabel = lambda path: path.split('/')[-2]

    for i, d in enumerate(dirs):
        impath = glob(d + "/*png")
        print("Directory :", d)
        for i in tqdm(impath):
            f = transform.resize(io.imread(i), im_size)
            images.append((f, i))
    return images


def load(dir_path, im_size):
    """Load a directory of images.

    Parameters
    -------------
    dir_path : str
        Directory path. Structure is dir_path/classes/images

    im_size : tuple
        Shape for each image (don't forget channels)
    """
    imlabel = lambda path: path.split('/')[-2]

    impath = glob(dir_path + "/*/*png")
    images = []
    for i in tqdm(impath):
        f = transform.resize(io.imread(i), im_size)
        images.append((f, imlabel(i)))
    return images


def onehot_label(labels):
    """Return dict {class_name: onehot_encoding}.
    Keep labels sorted.
    """
    l = sorted(set(labels))
    r = {}
    for i, c in enumerate(l):
        t = [0] * len(l)
        t[i] = 1
        r[c] = t
    return r


def train_val_test_split(dataset, prc_test=0.2, prc_val=0, random_state=None):
    x, y = dataset
    x = np.array(x)
    y = np.array(y)
    X_train, X_test,  Y_train, Y_test = train_test_split(*dataset, test_size=prc_test, random_state=random_state)

    if prc_val:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=prc_val, random_state=random_state)
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    else:
        return (X_train, Y_train), (X_test, Y_test)


def normalize(data):
    data = np.array(data)
    m = np.mean(data)
    s = np.std(data)
    return (data - m) / s, m, s
