import numpy as np
import shutil

from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.optimizers import Adadelta, SGD
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
fine_tuned_model_weights_path = os.path.join(temp_folder_path, "fine_tuned_model.h5")
bottleneck_train_x_path = os.path.join(temp_folder_path, "bottleneck_train_x.npy")
bottleneck_train_y_path = os.path.join(temp_folder_path, "bottleneck_train_y.npy")
bottleneck_test_x_path = os.path.join(temp_folder_path, "bottleneck_test_x.npy")
bottleneck_test_y_path = os.path.join(temp_folder_path, "bottleneck_test_y.npy")


num_train_samples = 2000
num_validation_samples = 800


def get_pretrained_model():
    return applications.InceptionV3(include_top=False, weights='imagenet', pooling="avg")


def append_array(original, new):
    if original is None:
        return new
    else:
        return np.vstack((original, new))


def save_bottlebeck_features(
        pretrained_model=get_pretrained_model(),
        num_train_batches=100,
        num_test_batches=10):
    train_y = None
    test_y = None
    bottleneck_train_x = None
    bottleneck_test_x = None

    generator = get_train_generator(width=inception_v3_width, height=inception_v3_height, batch_size=3200)
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
        else:
            print("\rTraining data transformation finished: " + str(i / num_train_batches), end=' ')

    np.save(open(bottleneck_train_x_path, "wb"), bottleneck_train_x)
    np.save(open(bottleneck_train_y_path, "wb"), train_y)

    generator = get_test_generator(width=inception_v3_width, height=inception_v3_height, batch_size=3200)
    i = 0
    for batch_x, batch_y in generator:
        batch_x = applications.inception_v3.preprocess_input(batch_x)
        bottleneck_batch_x = pretrained_model.predict(batch_x)
        bottleneck_test_x = append_array(bottleneck_test_x, bottleneck_batch_x)
        test_y = append_array(test_y, batch_y)
        i += 1
        if i >= num_test_batches:
            break
        else:
            print("\rTesting data transformation finished: " + str(i / num_test_batches), end=' ')

    np.save(open(bottleneck_test_x_path, 'wb'), bottleneck_test_x)
    np.save(open(bottleneck_test_y_path, 'wb'), test_y)


def train_top_model():
    train_data = np.load(open(bottleneck_train_x_path, "rb"))
    train_labels = np.load(open(bottleneck_train_y_path, "rb"))

    validation_data = np.load(open(bottleneck_test_x_path, "rb"))
    validation_labels = np.load(open(bottleneck_test_y_path, "rb"))

    print("Input shape: " + str(train_data.shape[1:]))

    model = get_top_model_architecture(train_data.shape[1:])

    model.compile(optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.001),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    earlystopper = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=3, verbose=0, mode='auto')

    model.fit(train_data, train_labels,
              epochs=15,
              batch_size=BATCH_SIZE,
              validation_data=(validation_data, validation_labels),
              callbacks=[earlystopper])
    model.save_weights(top_model_weights_path)


def get_top_model_architecture(input_shape):
    regularizer = None  # regularizers.l2(0.001)
    dropout_rate = 0.3
    model = Sequential()
    model.add(Dense(2048,
                    input_shape=input_shape,
                    kernel_regularizer=regularizer,
                    activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(144,
                    activation='softmax'))
    return model


def fine_tuning_cnn():
    pretrained_model = get_pretrained_model()

    top_model = get_top_model_architecture(pretrained_model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    model = Model(inputs=pretrained_model.input, outputs=top_model(pretrained_model.output))

    for layer in model.layers[:152]:
        layer.trainable = False
    for layer in model.layers[152:]:
        layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    print(model.summary())

    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    earlystopper = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='auto')

    model.fit_generator(
        get_train_generator(width=inception_v3_width, height=inception_v3_height, preprocessing_function=applications.inception_v3.preprocess_input),
        steps_per_epoch=100,
        epochs=500,
        validation_data=get_test_generator(width=inception_v3_width, height=inception_v3_height, preprocessing_function=applications.inception_v3.preprocess_input),
        validation_steps=25,
        workers=7,
        callbacks=[earlystopper])

    model.save_weights(fine_tuned_model_weights_path)


# save_bottlebeck_features()
# train_top_model()
fine_tuning_cnn()

# 1*2048 FC with DO=0.5
# Epoch 9/4000
# loss: 0.5608
# categorical_accuracy: 0.8485
# val_loss: 1.2125
# val_categorical_accuracy: 0.6659

# 2*2048 FC with DO=0.5
# Epoch 11/4000
# loss: 0.5862
# categorical_accuracy: 0.8227
# val_loss: 1.1443
# val_categorical_accuracy: 0.6599


# 3*2048 FC with DO=0.5
# Epoch 9/4000
# loss: 0.8936
# categorical_accuracy: 0.7240
# val_loss: 1.2141
# val_categorical_accuracy: 0.6431

# 3*2048 FC with DO=0.3 l2=0.001
# loss: 1.2823
# categorical_accuracy: 0.8518
# val_loss: 1.9469
# val_categorical_accuracy: 0.6623

#
# 100/100 [==============================] - 44s - loss: 0.1055 - categorical_accuracy: 0.9818 - val_loss: 0.9617 - val_categorical_accuracy: 0.7263
# Epoch 37/500
# 100/100 [==============================] - 44s - loss: 0.1006 - categorical_accuracy: 0.9834 - val_loss: 0.9885 - val_categorical_accuracy: 0.7302
# Epoch 38/500
# 100/100 [==============================] - 44s - loss: 0.1078 - categorical_accuracy: 0.9825 - val_loss: 0.9695 - val_categorical_accuracy: 0.7276
# Epoch 39/500
# 100/100 [==============================] - 44s - loss: 0.1089 - categorical_accuracy: 0.9815 - val_loss: 0.9799 - val_categorical_accuracy: 0.7315
# Epoch 40/500
# 100/100 [==============================] - 44s - loss: 0.0977 - categorical_accuracy: 0.9837 - val_loss: 1.0562 - val_categorical_accuracy: 0.7147
# Epoch 41/500
# 100/100 [==============================] - 44s - loss: 0.1119 - categorical_accuracy: 0.9793 - val_loss: 0.9928 - val_categorical_accuracy: 0.7341
# Epoch 42/500
# 100/100 [==============================] - 44s - loss: 0.0976 - categorical_accuracy: 0.9847 - val_loss: 1.0014 - val_categorical_accuracy: 0.7289
# Epoch 43/500
# 100/100 [==============================] - 44s - loss: 0.0852 - categorical_accuracy: 0.9869 - val_loss: 1.0299 - val_categorical_accuracy: 0.7302
# Epoch 44/500
# 100/100 [==============================] - 44s - loss: 0.0988 - categorical_accuracy: 0.9818 - val_loss: 0.9582 - val_categorical_accuracy: 0.7341
