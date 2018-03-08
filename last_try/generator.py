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

class RotatingGenerator(CustomImageDataGenerator):
    """docstring for MultipleInputData."""
    def __init__(self, angle, *args, **kwargs):
        super(RotatingGenerator, self).__init__(*args, **kwargs)
        self.angle = angle

    def random_transform(self, x, seed=None):
        x = scipy.ndimage.rotate(im, self.angle)
        x = super(RotatingGenerator, self).random_transform(x, seed=seed)
        return x


def multipleInputGenerator(x, y, generators):
    gen1 = generators[0].flow(x, y)
    gen_o = [gen.flow(x) for gen in generators[1:]]
    while True:
        x1 = gen1.next()
        x_o = [gen.next() for gen in gen_o]
        yield [x1[0]] + x_o, x1[1]
