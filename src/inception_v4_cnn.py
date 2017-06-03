import numpy as np
import shutil



from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.optimizers import Adadelta, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, regularizers, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import applications
import os

# dimensions of our images.
from generate_data import project_path
from keras_inceptionV4.inception_v4 import inception_v4
import keras_inceptionV4
from read_data import get_train_generator, get_test_generator, BATCH_SIZE

inception_v4_width, inception_v4_height = 299, 299

temp_folder_path = os.path.join(project_path, "tmp", "inception_v4")
top_model_weights_path = os.path.join(temp_folder_path, "bottleneck_model.h5")
fine_tuned_model_weights_path = os.path.join(temp_folder_path, "fine_tuned_model.h5")
bottleneck_train_x_path = os.path.join(temp_folder_path, "bottleneck_train_x.npy")
bottleneck_train_y_path = os.path.join(temp_folder_path, "bottleneck_train_y.npy")
bottleneck_test_x_path = os.path.join(temp_folder_path, "bottleneck_test_x.npy")
bottleneck_test_y_path = os.path.join(temp_folder_path, "bottleneck_test_y.npy")


num_train_samples = 2000
num_validation_samples = 800


def get_pretrained_inception_v4():
    model0 = inception_v4(num_classes=1001, dropout_keep_prob=0.5, include_top=True, weights='imagenet')
    model = Model(inputs=model0.input, outputs=model0.layers[-4].output)
    return model


def append_array(original, new):
    if original is None:
        return new
    else:
        return np.vstack((original, new))


def save_bottlebeck_features(
        num_train_batches=10,
        num_test_batches=1):
    train_y = None
    test_y = None
    bottleneck_train_x = None
    bottleneck_test_x = None

    pretrained_model=get_pretrained_inception_v4()
    generator = get_train_generator(width=inception_v4_width, height=inception_v4_height, batch_size=320,
                                    preprocessing_function=keras_inceptionV4.inception_v4.preprocess_input)
    print(pretrained_model.summary())
    i = 0
    for batch_x, batch_y in generator:
        if i == 0:
            print(batch_x.shape)
        bottleneck_batch_x = pretrained_model.predict(batch_x)
        bottleneck_train_x = append_array(bottleneck_train_x, bottleneck_batch_x)
        print(np.average(bottleneck_batch_x[0]))
        train_y = append_array(train_y, batch_y)

        i += 1
        if i >= num_train_batches:
            break
        else:
            print("\rTraining data transformation finished: " + str(i / num_train_batches), end=' ')

    np.save(open(bottleneck_train_x_path, "wb"), bottleneck_train_x)
    np.save(open(bottleneck_train_y_path, "wb"), train_y)

    generator = get_test_generator(width=inception_v4_width, height=inception_v4_height, batch_size=320,
                                   preprocessing_function=keras_inceptionV4.inception_v4.preprocess_input)
    i = 0
    for batch_x, batch_y in generator:
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

    print(model.summary())

    model.compile(optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.001),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    earlystopper = EarlyStopping(monitor='loss', min_delta=1e-2, patience=10, verbose=0, mode='auto')
    # earlystopper = EarlyStopping(monitor='loss', min_delta=2e-2, patience=1, verbose=0, mode='auto')

    model.fit(train_data, train_labels,
              epochs=300,
              batch_size=BATCH_SIZE,
              validation_data=(validation_data, validation_labels),
              callbacks=[earlystopper])
    # print(model.get_weights())
    model.save_weights(top_model_weights_path)


#     Epoch 82/300
# 31685/31685 [==============================] - 5s - loss: 0.1569 - categorical_accuracy: 0.9734 - val_loss: 1.1443 - val_categorical_accuracy: 0.6818
# Epoch 83/300
# 31685/31685 [==============================] - 5s - loss: 0.1582 - categorical_accuracy: 0.9714 - val_loss: 1.1432 - val_categorical_accuracy: 0.6796
# Epoch 84/300
# 31685/31685 [==============================] - 5s - loss: 0.1572 - categorical_accuracy: 0.9730 - val_loss: 1.1434 - val_categorical_accuracy: 0.6818
# Epoch 85/300
# 31685/31685 [==============================] - 5s - loss: 0.1565 - categorical_accuracy: 0.9726 - val_loss: 1.1417 - val_categorical_accuracy: 0.6811


def get_top_model_architecture(input_shape):
    dropout_rate = 0.3
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1536, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(144,
                    activation='softmax'))
    return model


def fine_tuning_cnn():
    pretrained_model = get_pretrained_inception_v4()

    top_model = get_top_model_architecture(pretrained_model.output_shape[1:])
    # print(top_model.get_weights())
    top_model.load_weights(top_model_weights_path)
    # print(top_model.get_weights())

    model = Model(inputs=pretrained_model.input, outputs=top_model(pretrained_model.output))

    for layer in model.layers[:469]:
        layer.trainable = False
    for layer in model.layers[469:]:
        layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    print(model.summary())

    model.compile(optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    earlystopper = EarlyStopping(monitor='loss', min_delta=1e-3, patience=10, verbose=0, mode='auto')

    model.fit_generator(
        get_train_generator(width=inception_v4_width, height=inception_v4_height,
                            preprocessing_function=keras_inceptionV4.inception_v4.preprocess_input),
        steps_per_epoch=100,
        epochs=500,
        validation_data=get_test_generator(width=inception_v4_width, height=inception_v4_height,
                                           preprocessing_function=keras_inceptionV4.inception_v4.preprocess_input),
        validation_steps=25,
        workers=7,
        callbacks=[earlystopper])

    model.save_weights(fine_tuned_model_weights_path)


# save_bottlebeck_features()
# train_top_model()
fine_tuning_cnn()















# exam wrong results
#
# g = get_train_generator(width=inception_v4_width, height=inception_v4_height,
#                         preprocessing_function=keras_inceptionV4.inception_v4.preprocess_input)
#
# for batch_x, batch_y in g:
#     data_x, data_y = batch_x, batch_y
#     break;
# validation_data = np.load(open(bottleneck_train_x_path, "rb"))
# pretrained_model = get_pretrained_inception_v4()
# predicted = pretrained_model.predict(data_x)
# for i in range(10):
#     print("predicted: " + str(np.average(predicted[i])))
#     print(predicted[i].shape)
#     print("saved: " + str(np.average(validation_data[i])))
#     print(validation_data[i].shape)
