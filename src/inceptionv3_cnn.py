import numpy as np
import shutil

from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, regularizers
from keras import applications
import os

# dimensions of our images.
from generate_data import project_path
from read_data import get_train_generator, get_test_generator, BATCH_SIZE

inception_v3_width, inception_v3_height = 299, 299

temp_folder_path = os.path.join(project_path, "tmp", "inception_v3")
top_model_weights_path = os.path.join(temp_folder_path, "bottleneck_model.h5")
bottleneck_train_x_path = os.path.join(temp_folder_path, "bottleneck_train_x.npy")
bottleneck_train_y_path = os.path.join(temp_folder_path, "bottleneck_train_y.npy")
bottleneck_test_x_path = os.path.join(temp_folder_path, "bottleneck_test_x.npy")
bottleneck_test_y_path = os.path.join(temp_folder_path, "bottleneck_test_y.npy")


num_train_samples = 2000
num_validation_samples = 800


def append_array(original, new):
    if original is None:
        return new
    else:
        return np.vstack((original, new))


def save_bottlebeck_features(pretrained_model=applications.InceptionV3(include_top=False, weights='imagenet'),
                             num_train_batches=2000,
                             num_test_batches=200):
    train_y = None
    test_y = None
    bottleneck_train_x = None
    bottleneck_test_x = None

    generator = get_train_generator(width=inception_v3_width, height=inception_v3_height)
    i = 0
    for batch_x, batch_y in generator:
        if i == 0:
            print(batch_x.shape)
        batch_x = applications.inception_v3.preprocess_input(batch_x)
        bottleneck_batch_x = pretrained_model.predict(batch_x)
        bottleneck_train_x = append_array(bottleneck_train_x, bottleneck_batch_x)
        train_y = append_array(train_y, batch_y)

        i += 1
        if i >= num_train_batches:
            break

    np.save(open(bottleneck_train_x_path, "wb"), bottleneck_train_x)
    np.save(open(bottleneck_train_y_path, "wb"), train_y)

    generator = get_test_generator(width=inception_v3_width, height=inception_v3_height)
    i = 0
    for batch_x, batch_y in generator:
        batch_x = applications.inception_v3.preprocess_input(batch_x)
        bottleneck_batch_x = pretrained_model.predict(batch_x)
        bottleneck_test_x = append_array(bottleneck_test_x, bottleneck_batch_x)
        test_y = append_array(test_y, batch_y)
        i += 1
        if i >= num_test_batches:
            break

    np.save(open(bottleneck_test_x_path, 'wb'), bottleneck_test_x)
    np.save(open(bottleneck_test_y_path, 'wb'), test_y)


def train_top_model():
    train_data = np.load(open(bottleneck_train_x_path, "rb"))
    train_labels = np.load(open(bottleneck_train_y_path, "rb"))

    validation_data = np.load(open(bottleneck_test_x_path, "rb"))
    validation_labels = np.load(open(bottleneck_test_y_path, "rb"))

    regularizer = None# regularizers.l2(0.001)
    dropout_rate = 0.5

    print("Input shape: " + str(train_data.shape[1:]))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(2048,
                    kernel_regularizer=regularizer,
                    activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(144,
                    activation='softmax'))

    model.compile(optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.001),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    earlystopper = EarlyStopping(monitor='loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')

    model.fit(train_data, train_labels,
              epochs=4000,
              batch_size=BATCH_SIZE,
              validation_data=(validation_data, validation_labels),
              callbacks=[earlystopper])
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
# train_top_model()
