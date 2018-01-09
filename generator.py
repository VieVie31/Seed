from keras.preprocessing.image import ImageDataGenerator
import dropelets


def random_generator_generator(min, max):
    def uniform_generator(min, max):
        while True:
            yield np.random.uniform(min, max)
    return uniform_generator(min, max)


sigma_gen = random_generator_generator(.5, 10)
angle_gen = random_generator_generator(0, 360)
zoom_gen = random_generator_generator(.8, 1.2)
drop_gen = random_generator_generator(0, .1)

def window_generator_generator(shape):
    def window_generator(shape):
        while True:
            yield np.int64([np.random.uniform(.05 * s, .1 * s) for s in shape[:-1]])
    return window_generator(shape)


class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, image_shape, prob_transfo, *args, **kwargs):
        super(CustomImageDataGenerator, self).__init__(*args, **kwargs)
        self.window_gen = window_generator_generator(image_shape)
        self.prob = prob_transfo

    def random_transform(self, x, seed=None):
        x = super(CustomImageDataGenerator, self).random_transform(x, seed=seed)
        if np.random.random() < self.prob:
            x = dropelets.gaussian(x, next(sigma_gen), next(self.window_gen))

        if np.random.random() < self.prob:
            x = dropelets.rotate(x, next(angle_gen), range(len(x.shape)-1), next(self.window_gen))

        if np.random.random() < self.prob:
            x = dropelets.zoom(x, [next(zoom_gen) for i in range(len(x.shape) - 1)] + [1], next(self.window_gen))

        x = dropelets.dropout(x, next(drop_gen), 0)

        return x
