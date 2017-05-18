import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 16


def get_train_generator(training_data_dir="/home/sdjksdafji/Documents/others/Data558/training_data"):
    train_datagen = ImageDataGenerator(
        rotation_range=160,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest")
    return train_datagen.flow_from_directory(
        training_data_dir,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


def get_test_generator(training_data_dir="/home/sdjksdafji/Documents/others/Data558/testing_data"):
    test_datagen = ImageDataGenerator()
    return test_datagen.flow_from_directory(
        training_data_dir,
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


def show_image(image, title=None):
    image = image.astype(np.uint8)
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    plt.show()
