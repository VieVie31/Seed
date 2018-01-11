from keras.preprocessing.image import ImageDataGenerator
import dropelets
import numpy as np

def random_generator_generator(min, max):
    def uniform_generator(min, max):
        while True:
            yield np.random.uniform(min, max)
    return uniform_generator(min, max)


sigma_gen = np.random.uniform(.5, 10)
angle_gen = np.random.uniform(0, 360)
zoom_gen = np.random.uniform(.8, 1.2)
drop_gen = np.random.uniform(0, .1)

def window_generator_generator(shape):
    def window_generator(shape):
        while True:
            yield np.int64([np.random.uniform(.05 * s, .1 * s) for s in shape[:-1]])
    return window_generator(shape)


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
