import data
# Now import real libs
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# Keras
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adadelta, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, InceptionResNetV2, Xception

im_size = (300, 300, 3)

import rotating_network as rn
from generator import CustomImageDataGenerator

def preprocess():
    """
    Returns
    ------------
    training_generator
    test_generator
    mean for normalization
    std for normalization
    """
    dataset = data.load("../../train", im_size)
    x, y = zip(*dataset)
    r = data.onehot_label(y)
    y = list(map(lambda k: r[k], y))
    x, m, s = data.normalize(x)
    (x_train, y_train), (x_test, y_test) = data.train_val_test_split((x, y), prc_test=.2, random_state=42)
    training_generator = CustomImageDataGenerator(
            x_train[0].shape,
            .5,
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
    test_generator = CustomImageDataGenerator(
            x_train[0].shape,
            0,
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=.0,
            height_shift_range=.0,
            horizontal_flip=False,
            vertical_flip=False,
            zoom_range=0,
            shear_range=0,
            fill_mode="reflect"
    )
    return training_generator, (x_train, y_train), test_generator, (x_test, y_test), m, s

def build_model():
    x_model = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet',
        pooling="avg"
    )
    partial_model = x_model.output
    for l in x_model.layers:
        try:
            x_model.get_layer(l).trainable = False
        except:
            pass
    model = BatchNormalization()(partial_model)
    model = Dropout(.5)(model)
    model = Dense(512, activation="elu")(model)
    model = Dropout(.5)(model)
    model = Dense(512, activation="elu")(model)
    model = Dropout(.5)(model)
    model = Dense(12, activation='softmax')(model)
    model = Model(input=[x_model.input], output=model)
    return model

# Call functions
train_gen, (x_train, y_train), test_gen, (x_test, y_test), mean, std = preprocess()
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print("Mean :", mean, "Std :", std)

model = build_model()
print(model.summary())

# Prepare fit
cw = {i: (y_train.argmax(1) == i).sum() + (y_test.argmax(1) == i).sum() for i in range(12)}
tot = sum([v for v in cw.values()])
cw = {k: v / tot * 100 for k, v in cw.items()}
print(cw)
batch_size = 32

# Fit
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.hdf5", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

print("TRAINING THE MODEL")

model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])
h = model.fit_generator(
    train_gen.flow(x_train, y_train),
    class_weight=cw,
    validation_data=test_gen.flow(x_test, y_test),
    epochs=200,
    callbacks=[early, check],
    use_multiprocessing=True,
    workers=8,
    max_queue_size=5,
)
with open("pre_history.txt", "w") as f:
    f.write(str(h.history))
