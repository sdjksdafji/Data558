import keras
import numpy as np
import shutil
import csv

import scipy
from keras.applications.imagenet_utils import decode_predictions
from keras.callbacks import EarlyStopping
from keras.engine import Model
from keras.optimizers import Adadelta, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, regularizers, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import applications
import os

# dimensions of our images.
from keras.regularizers import l2

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
        num_train_batches=100,
        num_test_batches=10):
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
    dropout_rate = 0.5
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(144,
                    kernel_regularizer=l2(0.01),
                    activation='softmax'))
    return model


def predict(model_path="fine_tuned_model_bak_7.h5"):
    model_weights_path = os.path.join(temp_folder_path, model_path)

    pretrained_model = get_pretrained_inception_v4()
    top_model = get_top_model_architecture(pretrained_model.output_shape[1:])
    model = Model(inputs=pretrained_model.input, outputs=top_model(pretrained_model.output))

    model.load_weights(model_weights_path)

    generator = get_train_generator()
    index_to_class = {v: k for k, v in generator.class_indices.items()}

    with open(os.path.join(temp_folder_path, "prediction", "train_prediction.csv"), "w") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Id", "Prediction"])
        with open(os.path.join(project_path, "data", "train_images.csv")) as inputFile:
            reader = csv.reader(inputFile)
            header = True
            for row in reader:
                if header:
                    header = False
                    continue
                subPath = row[0]
                id = row[1]
                img = load_img(os.path.join(project_path, "data", subPath))
                x = img_to_array(img)
                x = scipy.misc.imresize(x, (inception_v4_width, inception_v4_height, 3))
                x = keras_inceptionV4.inception_v4.preprocess_input(x)
                x = x.reshape((1,) + x.shape)
                y = model.predict(x)
                writer.writerow([id, int(index_to_class[np.argmax(y[0])][:3])])

    with open(os.path.join(temp_folder_path, "prediction", "test_prediction.csv"), "w") as outputFile:
        writer = csv.writer(outputFile)
        writer.writerow(["Id", "Prediction"])
        with open(os.path.join(project_path, "data", "test_images.csv")) as inputFile:
            reader = csv.reader(inputFile)
            header = True
            for row in reader:
                if header:
                    header = False
                    continue
                subPath = row[0]
                id = row[1]
                img = load_img(os.path.join(project_path, "data", subPath))
                x = img_to_array(img)
                x = scipy.misc.imresize(x, (inception_v4_width, inception_v4_height, 3))
                x = keras_inceptionV4.inception_v4.preprocess_input(x)
                x = x.reshape((1,) + x.shape)
                y = model.predict(x)
                writer.writerow([id, int(index_to_class[np.argmax(y[0])][:3])])


def fine_tuning_cnn():
    pretrained_model = get_pretrained_inception_v4()

    top_model = get_top_model_architecture(pretrained_model.output_shape[1:])
    # print(top_model.get_weights())
    top_model.load_weights(top_model_weights_path)
    # print(top_model.get_weights())

    model = Model(inputs=pretrained_model.input, outputs=top_model(pretrained_model.output))

    for layer in model.layers[:292]:
        layer.trainable = False
    for layer in model.layers[292:]:
        layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    print(model.summary())

    model.compile(optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.002),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    earlystopper = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='auto')

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
# predict()














# exam wrong results

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



# id 8 do 0.5 l2 0.01 freeze 292 ready
# Epoch 153/500
# 100/100 [==============================] - 76s - loss: 1.3298 - categorical_accuracy: 0.9953 - val_loss: 2.1294 - val_categorical_accuracy: 0.7982
# Epoch 154/500
# 100/100 [==============================] - 76s - loss: 1.3179 - categorical_accuracy: 0.9963 - val_loss: 2.1294 - val_categorical_accuracy: 0.7917
# Epoch 155/500
# 100/100 [==============================] - 75s - loss: 1.3295 - categorical_accuracy: 0.9950 - val_loss: 2.1114 - val_categorical_accuracy: 0.8008
# Epoch 156/500
# 100/100 [==============================] - 76s - loss: 1.3178 - categorical_accuracy: 0.9947 - val_loss: 2.1060 - val_categorical_accuracy: 0.8060
# Epoch 157/500
# 100/100 [==============================] - 76s - loss: 1.3097 - categorical_accuracy: 0.9981 - val_loss: 2.1086 - val_categorical_accuracy: 0.8047


# id 7 do 0.5 l2 0.01 freeze 228 ready
# Epoch 26/500
# 100/100 [==============================] - 85s - loss: 1.4975 - categorical_accuracy: 0.9887 - val_loss: 2.7963 - val_categorical_accuracy: 0.7782
# Epoch 27/500
# 100/100 [==============================] - 85s - loss: 1.4862 - categorical_accuracy: 0.9906 - val_loss: 2.8116 - val_categorical_accuracy: 0.7782
# Epoch 28/500
# 100/100 [==============================] - 85s - loss: 1.4689 - categorical_accuracy: 0.9881 - val_loss: 2.7653 - val_categorical_accuracy: 0.7912
# Epoch 29/500
# 100/100 [==============================] - 85s - loss: 1.4580 - categorical_accuracy: 0.9906 - val_loss: 2.7616 - val_categorical_accuracy: 0.7938

# id 6 do 0.5 l2 0.01 freeze 292 ready
# Epoch 22/500
# 100/100 [==============================] - 77s - loss: 1.7170 - categorical_accuracy: 0.9762 - val_loss: 2.6444 - val_categorical_accuracy: 0.7951
# Epoch 23/500
# 100/100 [==============================] - 77s - loss: 1.7176 - categorical_accuracy: 0.9700 - val_loss: 2.6312 - val_categorical_accuracy: 0.7886
# Epoch 24/500
# 100/100 [==============================] - 77s - loss: 1.6939 - categorical_accuracy: 0.9781 - val_loss: 2.6877 - val_categorical_accuracy: 0.8029
# Epoch 25/500
# 100/100 [==============================] - 77s - loss: 1.6794 - categorical_accuracy: 0.9744 - val_loss: 2.6494 - val_categorical_accuracy: 0.7977
# Epoch 26/500
# 100/100 [==============================] - 76s - loss: 1.6570 - categorical_accuracy: 0.9781 - val_loss: 2.7061 - val_categorical_accuracy: 0.7977
# Epoch 27/500
# 100/100 [==============================] - 77s - loss: 1.6347 - categorical_accuracy: 0.9803 - val_loss: 2.6429 - val_categorical_accuracy: 0.8054
# Epoch 28/500
# 100/100 [==============================] - 76s - loss: 1.6176 - categorical_accuracy: 0.9803 - val_loss: 2.6699 - val_categorical_accuracy: 0.8054


# id 5 do 0.5 l2 0.01 freeze 356 ready
# Epoch 27/500
# 100/100 [==============================] - 68s - loss: 1.9432 - categorical_accuracy: 0.9591 - val_loss: 2.7531 - val_categorical_accuracy: 0.7678
# Epoch 28/500
# 100/100 [==============================] - 68s - loss: 1.9173 - categorical_accuracy: 0.9553 - val_loss: 2.7540 - val_categorical_accuracy: 0.7834
# Epoch 29/500
# 100/100 [==============================] - 68s - loss: 1.9281 - categorical_accuracy: 0.9618 - val_loss: 2.7175 - val_categorical_accuracy: 0.7795
# Epoch 30/500
# 100/100 [==============================] - 67s - loss: 1.8798 - categorical_accuracy: 0.9618 - val_loss: 2.7404 - val_categorical_accuracy: 0.7678
# Epoch 31/500
# 100/100 [==============================] - 68s - loss: 1.8782 - categorical_accuracy: 0.9637 - val_loss: 2.7286 - val_categorical_accuracy: 0.7847
# Epoch 32/500
# 100/100 [==============================] - 67s - loss: 1.8569 - categorical_accuracy: 0.9672 - val_loss: 2.7304 - val_categorical_accuracy: 0.7964
# Epoch 33/500
# 100/100 [==============================] - 68s - loss: 1.8456 - categorical_accuracy: 0.9700 - val_loss: 2.7218 - val_categorical_accuracy: 0.7717
# Epoch 34/500
# 100/100 [==============================] - 67s - loss: 1.8337 - categorical_accuracy: 0.9662 - val_loss: 2.7170 - val_categorical_accuracy: 0.7860
# Epoch 35/500
# 100/100 [==============================] - 67s - loss: 1.8171 - categorical_accuracy: 0.9762 - val_loss: 2.7165 - val_categorical_accuracy: 0.7808

# id 4 do 0.95 l2 0.01 freeze 356 ready

# id 3 do 0.75 l2 0.01 freeze 401 ready
# Epoch 50/500
# 100/100 [==============================] - 63s - loss: 6.8813 - categorical_accuracy: 0.9265 - val_loss: 7.2370 - val_categorical_accuracy: 0.7665
# Epoch 51/500
# 100/100 [==============================] - 63s - loss: 6.8257 - categorical_accuracy: 0.9237 - val_loss: 7.2028 - val_categorical_accuracy: 0.7704
# Epoch 52/500
# 100/100 [==============================] - 63s - loss: 6.7524 - categorical_accuracy: 0.9281 - val_loss: 7.0937 - val_categorical_accuracy: 0.7834
# Epoch 53/500
# 100/100 [==============================] - 62s - loss: 6.6950 - categorical_accuracy: 0.9290 - val_loss: 7.0680 - val_categorical_accuracy: 0.7613
# Epoch 54/500
# 100/100 [==============================] - 62s - loss: 6.6492 - categorical_accuracy: 0.9337 - val_loss: 7.0447 - val_categorical_accuracy: 0.7639
# Epoch 55/500
# 100/100 [==============================] - 63s - loss: 6.5696 - categorical_accuracy: 0.9331 - val_loss: 6.9699 - val_categorical_accuracy: 0.7523
# Epoch 56/500
# 100/100 [==============================] - 63s - loss: 6.5349 - categorical_accuracy: 0.9252 - val_loss: 6.9263 - val_categorical_accuracy: 0.7678
# Epoch 57/500
# 100/100 [==============================] - 63s - loss: 6.4680 - categorical_accuracy: 0.9311 - val_loss: 6.8331 - val_categorical_accuracy: 0.7743
# Epoch 58/500
# 100/100 [==============================] - 63s - loss: 6.4209 - categorical_accuracy: 0.9412 - val_loss: 6.8609 - val_categorical_accuracy: 0.7575



# do 0.0 l2 0.05 freeze 401
# Epoch 19/500
# 100/100 [==============================] - 63s - loss: 5.1257 - categorical_accuracy: 0.9649 - val_loss: 5.5381 - val_categorical_accuracy: 0.7367
# Epoch 20/500
# 100/100 [==============================] - 62s - loss: 4.5669 - categorical_accuracy: 0.9656 - val_loss: 5.1969 - val_categorical_accuracy: 0.7289
# Epoch 21/500
# 100/100 [==============================] - 63s - loss: 4.0698 - categorical_accuracy: 0.9684 - val_loss: 4.9275 - val_categorical_accuracy: 0.7367
# Epoch 22/500
# 100/100 [==============================] - 63s - loss: 3.6477 - categorical_accuracy: 0.9743 - val_loss: 4.8017 - val_categorical_accuracy: 0.7393
# Epoch 23/500
# 100/100 [==============================] - 63s - loss: 3.2978 - categorical_accuracy: 0.9703 - val_loss: 4.7120 - val_categorical_accuracy: 0.7380
# Epoch 24/500
# 100/100 [==============================] - 63s - loss: 3.0015 - categorical_accuracy: 0.9778 - val_loss: 4.6231 - val_categorical_accuracy: 0.7704
# Epoch 25/500
# 100/100 [==============================] - 63s - loss: 2.7593 - categorical_accuracy: 0.9778 - val_loss: 4.5697 - val_categorical_accuracy: 0.7691
# Epoch 26/500
# 100/100 [==============================] - 63s - loss: 2.5629 - categorical_accuracy: 0.9787 - val_loss: 4.5920 - val_categorical_accuracy: 0.7665
# Epoch 27/500
# 100/100 [==============================] - 63s - loss: 2.3927 - categorical_accuracy: 0.9822 - val_loss: 4.6321 - val_categorical_accuracy: 0.7613
# Epoch 28/500
# 100/100 [==============================] - 63s - loss: 2.2419 - categorical_accuracy: 0.9850 - val_loss: 4.7200 - val_categorical_accuracy: 0.7030




# id 2 do 0.5 l2 0.05 freeze :401
# Epoch 27/500
# 100/100 [==============================] - 63s - loss: 3.0181 - categorical_accuracy: 0.9634 - val_loss: 5.1234 - val_categorical_accuracy: 0.7601
# Epoch 28/500
# 100/100 [==============================] - 62s - loss: 2.8350 - categorical_accuracy: 0.9718 - val_loss: 5.1644 - val_categorical_accuracy: 0.7536
# Epoch 29/500
# 100/100 [==============================] - 63s - loss: 2.6630 - categorical_accuracy: 0.9712 - val_loss: 5.1193 - val_categorical_accuracy: 0.7562
# Epoch 30/500
# 100/100 [==============================] - 63s - loss: 2.5499 - categorical_accuracy: 0.9716 - val_loss: 5.2282 - val_categorical_accuracy: 0.7562
# Epoch 31/500
# 100/100 [==============================] - 63s - loss: 2.4097 - categorical_accuracy: 0.9818 - val_loss: 5.2772 - val_categorical_accuracy: 0.7471
# Epoch 32/500
# 100/100 [==============================] - 62s - loss: 2.3373 - categorical_accuracy: 0.9778 - val_loss: 5.3688 - val_categorical_accuracy: 0.7406
# Epoch 33/500
# 100/100 [==============================] - 62s - loss: 2.2299 - categorical_accuracy: 0.9787 - val_loss: 5.3991 - val_categorical_accuracy: 0.7250
# Epoch 34/500
# 100/100 [==============================] - 62s - loss: 2.1680 - categorical_accuracy: 0.9825 - val_loss: 5.3371 - val_categorical_accuracy: 0.7510
# Epoch 35/500
# 100/100 [==============================] - 62s - loss: 2.1181 - categorical_accuracy: 0.9809 - val_loss: 5.3834 - val_categorical_accuracy: 0.7575



# id 1 do 0.5 l2 0.05 freeze 401 ready
# Epoch 82/500
# 100/100 [==============================] - 63s - loss: 9.7132 - categorical_accuracy: 0.9750 - val_loss: 9.9817 - val_categorical_accuracy: 0.7510
# Epoch 83/500
# 100/100 [==============================] - 63s - loss: 9.6381 - categorical_accuracy: 0.9728 - val_loss: 9.8932 - val_categorical_accuracy: 0.7588
# Epoch 84/500
# 100/100 [==============================] - 63s - loss: 9.5496 - categorical_accuracy: 0.9731 - val_loss: 9.8119 - val_categorical_accuracy: 0.7588
# Epoch 85/500
# 100/100 [==============================] - 63s - loss: 9.4708 - categorical_accuracy: 0.9753 - val_loss: 9.7834 - val_categorical_accuracy: 0.7497
# Epoch 86/500
# 100/100 [==============================] - 62s - loss: 9.4057 - categorical_accuracy: 0.9700 - val_loss: 9.7353 - val_categorical_accuracy: 0.7380
# Epoch 87/500
# 100/100 [==============================] - 62s - loss: 9.3267 - categorical_accuracy: 0.9725 - val_loss: 9.6421 - val_categorical_accuracy: 0.7523
# Epoch 88/500
# 100/100 [==============================] - 62s - loss: 9.2474 - categorical_accuracy: 0.9803 - val_loss: 9.5628 - val_categorical_accuracy: 0.7497
# Epoch 89/500
# 100/100 [==============================] - 62s - loss: 9.1718 - categorical_accuracy: 0.9784 - val_loss: 9.5667 - val_categorical_accuracy: 0.7406



# id 0 do 0.4 l2 0.001 freeze :435
# Epoch 24/500
# 100/100 [==============================] - 61s - loss: 2.2096 - categorical_accuracy: 0.9784 - val_loss: 3.1368 - val_categorical_accuracy: 0.7160
# Epoch 25/500
# 100/100 [==============================] - 61s - loss: 2.1679 - categorical_accuracy: 0.9775 - val_loss: 3.1439 - val_categorical_accuracy: 0.7160
# Epoch 26/500
# 100/100 [==============================] - 61s - loss: 2.1111 - categorical_accuracy: 0.9794 - val_loss: 3.0569 - val_categorical_accuracy: 0.7458
# Epoch 27/500
# 100/100 [==============================] - 61s - loss: 2.0678 - categorical_accuracy: 0.9815 - val_loss: 3.1297 - val_categorical_accuracy: 0.7069
# Epoch 28/500
# 100/100 [==============================] - 61s - loss: 2.0267 - categorical_accuracy: 0.9840 - val_loss: 3.1815 - val_categorical_accuracy: 0.7445
# Epoch 29/500
# 100/100 [==============================] - 61s - loss: 1.9913 - categorical_accuracy: 0.9837 - val_loss: 3.1725 - val_categorical_accuracy: 0.7419
# Epoch 30/500
# 100/100 [==============================] - 61s - loss: 1.9687 - categorical_accuracy: 0.9865 - val_loss: 3.2787 - val_categorical_accuracy: 0.7393



#　do 0.0 l2 0.025 freeze 401
# Epoch 16/500
# 100/100 [==============================] - 62s - loss: 4.9781 - categorical_accuracy: 0.9687 - val_loss: 5.3865 - val_categorical_accuracy: 0.7367
# Epoch 17/500
# 100/100 [==============================] - 62s - loss: 4.3903 - categorical_accuracy: 0.9719 - val_loss: 5.0095 - val_categorical_accuracy: 0.7380
# Epoch 18/500
# 100/100 [==============================] - 62s - loss: 3.9050 - categorical_accuracy: 0.9740 - val_loss: 4.7762 - val_categorical_accuracy: 0.7432
# Epoch 19/500
# 100/100 [==============================] - 62s - loss: 3.5067 - categorical_accuracy: 0.9744 - val_loss: 4.6804 - val_categorical_accuracy: 0.7406
# Epoch 20/500
# 100/100 [==============================] - 62s - loss: 3.1509 - categorical_accuracy: 0.9778 - val_loss: 4.4713 - val_categorical_accuracy: 0.7613
# Epoch 21/500
# 100/100 [==============================] - 62s - loss: 2.8669 - categorical_accuracy: 0.9781 - val_loss: 4.5445 - val_categorical_accuracy: 0.7289
# Epoch 22/500
# 100/100 [==============================] - 62s - loss: 2.6047 - categorical_accuracy: 0.9797 - val_loss: 4.4828 - val_categorical_accuracy: 0.7341
# Epoch 23/500
# 100/100 [==============================] - 62s - loss: 2.4091 - categorical_accuracy: 0.9844 - val_loss: 4.5325 - val_categorical_accuracy: 0.7367
# Epoch 24/500
# 100/100 [==============================] - 62s - loss: 2.2333 - categorical_accuracy: 0.9841 - val_loss: 4.5492 - val_categorical_accuracy: 0.7575
# Epoch 25/500
# 100/100 [==============================] - 62s - loss: 2.0705 - categorical_accuracy: 0.9872 - val_loss: 4.6194 - val_categorical_accuracy: 0.728