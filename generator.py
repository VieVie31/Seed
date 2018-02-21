from keras.preprocessing.image import ImageDataGenerator
import dropelets
import numpy as np


class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, image_shape, prob_transfo, *args, **kwargs):
        super(CustomImageDataGenerator, self).__init__(*args, **kwargs)
        self.shape = image_shape
        self.prob = prob_transfo

    def random_transform(self, x, seed=None):
        x = super(CustomImageDataGenerator, self).random_transform(x, seed=seed)
        if np.random.random() < self.prob:
            x = dropelets.gaussian(x, np.random.uniform(.5, 10),np.int64([np.random.uniform(.05 * s, .1 * s) for s in self.shape[:-1]]))

        if np.random.random() < self.prob:
            x = dropelets.rotate(x, np.random.uniform(0, 360), range(len(x.shape)-1),np.int64([np.random.uniform(.05 * s, .1 * s) for s in self.shape[:-1]]))

        if np.random.random() < self.prob:
            x = dropelets.zoom(x, [np.random.uniform(.8, 1.2)]*len(x.shape[:-1]) + [1],np.int64([np.random.uniform(.05 * s, .1 * s) for s in self.shape[:-1]]))

        x = dropelets.dropout(x, np.random.uniform(0, .1), 0)

        return x

class MultipleInputData(CustomImageDataGenerator):
    """docstring for MultipleInputData."""
    def __init__(self, nb, transf, *args, **kwargs):
        super(MultipleInputData, self).__init__(*args, **kwargs)
        self.nb = nb
        self.transf = transf

    def random_transform(self, x, seed=None):
        X = [x]
        for i in range(self.nb-1):
            X.append(self.transf(X[-1]))
        X = list(map(super(MultipleInputData, self).random_transform, X, [seed]*len(X)))
        return X
