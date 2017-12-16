import data
# Now import real libs
import numpy as np

# Keras
import keras
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator


# Data
im_size = (224, 224, 3)

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
    print(r)
    y = list(map(lambda k: r[k], y))
    x, m, s = data.normalize(x)
    (x_train, y_train), (x_test, y_test) = data.train_val_test_split((x, y))

    training_generator = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center=False, 
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False, 
            rotation_range=180,
            width_shift_range=.1,
            height_shift_range=.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2
    )

    test_generator = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center=False, 
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False, 
            rotation_range=180,
            width_shift_range=.1,
            height_shift_range=.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2
    )
    return training_generator, (x_train, y_train), test_generator, (x_test, y_test), m, s


# Build model
from keras.applications import VGG16
def build_model():
	vgg = VGG16(
	    input_shape=im_size,
	    include_top=False,
	    weights='imagenet'
	)


	partial_vgg = vgg.get_layer('block3_pool').output

	model = Conv2D(64, (3, 3), activation='elu')(partial_vgg)
	model = Conv2D(64, (3, 3), activation='elu')(model)
	model = MaxPooling2D((2, 2))(model)

	model = Conv2D(64, (3, 3), activation='elu')(model)
	model = Conv2D(64, (3, 3), activation='elu')(model)
	model = MaxPooling2D((2, 2))(model)

	model = Flatten()(model)

	model = Dropout(.25)(model)
	model = Dense(12, activation='softmax')(model)

	model = Model(input=[vgg.input], output=model)

	to_freeze = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3'] #, 'block4_conv1', 'block4_conv2', 'block4_conv3']
	for t_f in to_freeze:
	    model.get_layer(t_f).trainable = False
	return model


# Call functions
model = build_model()
train_gen, (x_train, y_train), test_gen, (x_test, y_test), mean, std = preprocess()
y_train = np.asarray(y_train)

print("Mean :", mean, "Std :", std)


print(model.summary())
# Compile
from sklearn.metrics import f1_score
opt = Adadelta()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Prepare fit
from sklearn.utils import class_weight

check = ModelCheckpoint("weights.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#cw = class_weight.compute_class_weight('balanced', np.unique(y_train.argmax(1)), y_train.argmax(1))
cw = {i: (y_train.argmax(1) == i).mean()*100 for i in range(12)}
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

model.save("vgg_transfert_learning.h5")
