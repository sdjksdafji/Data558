import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
IMG_H, IMG_W = 224, 224


def get_train_generator(training_data_dir="/home/sdjksdafji/Documents/others/Data558/training_data",
                        preprocessing_function=None,
                        width=IMG_W,
                        height=IMG_H,
                        batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")
    return train_datagen.flow_from_directory(
        training_data_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical')


def get_test_generator(training_data_dir="/home/sdjksdafji/Documents/others/Data558/testing_data",
                       preprocessing_function=None,
                       width=IMG_W,
                       height=IMG_H,
                       batch_size=BATCH_SIZE):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    return test_datagen.flow_from_directory(
        training_data_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical')


def show_image(image, title=None):
    image = image.astype(np.uint8)
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    plt.show()
