import data
# Now import real libs
import numpy as np

# Keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, InceptionResNetV2, Xception

im_size = (224, 224, 3)


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
    dataset = data.load("../train", im_size)
    x, y = zip(*dataset)
    r = data.onehot_label(y)
    y = list(map(lambda k: r[k], y))
    x, m, s = data.normalize(x)
    (x_train, y_train), (x_test, y_test) = data.train_val_test_split((x, y))
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
            .5,
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


# Build model
def build_model():
    x_model = Xception(
        input_shape=im_size,
        include_top=False,
        weights='imagenet'
    )
    partial_model = x_model.layers[-1].output
    model = MaxPooling2D((7, 7))(partial_model)
    model = Flatten()(model)
    model = Dense(12, activation='softmax')(model)
    model = Model(input=[x_model.input], output=model)
    return model
    """
    vgg = VGG16(
        input_shape=im_size,
        include_top=False,
        weights='imagenet'
    )

    partial_vgg = vgg.get_layer('block4_pool').output

    model = Conv2D(64, (3, 3), activation='elu')(partial_vgg)
    model = Conv2D(64, (3, 3), activation='elu')(model)
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(64, (3, 3), activation='elu')(model)
    model = Conv2D(64, (3, 3), activation='elu')(model)
    model = GlobalAveragePooling2D()(model)

    model = Dropout(.25)(model)
    model = Dense(12, activation='softmax')(model)

    model = Model(input=[vgg.input], output=model)

    to_freeze = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2']#, 'block3_conv1', 'block3_conv2', 'block3_conv3']
    for t_f in to_freeze:
        model.get_layer(t_f).trainable = False
    return model
    """

# Call functions
train_gen, (x_train, y_train), test_gen, (x_test, y_test), mean, std = preprocess()
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print("Mean :", mean, "Std :", std)

model = build_model()
print(model.summary())
# Compile
opt = Adadelta()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Prepare fit

check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#  cw = class_weight.compute_class_weight('balanced', np.unique(y_train.argmax(1)), y_train.argmax(1))
cw = {i: (y_train.argmax(1) == i).sum() + (y_test.argmax(1) == i).sum() for i in range(12)}
tot = sum([v for v in cw.values()])
cw = {k: v / tot * 100 for k, v in cw.items()}
print(cw)
batch_size = 32

# Fit
h = model.fit_generator(
    train_gen.flow(x_train, y_train),
    class_weight=cw,
    validation_data=test_gen.flow(x_test, y_test),
    epochs=2000,
    callbacks=[early, check]
)

model.save("xception_transfert_learning.h5")
