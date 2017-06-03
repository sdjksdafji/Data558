import os
import random

import shutil

import zlib

project_path = "/home/sdjksdafji/Documents/others/Data558/"
original_data_dir = "data/train"
train_data_dir = "training_data"
test_data_dir = "testing_data"

NUM_EACH_CLASS = 30


def generate_training_testing_split(num_of_testing=3):

    train_data_path = os.path.join(project_path, train_data_dir)
    test_data_path = os.path.join(project_path, test_data_dir)
    original_data_path = os.path.join(project_path, original_data_dir)

    print("Removing old data if exists ...")
    shutil.rmtree(train_data_path, ignore_errors=True)
    shutil.rmtree(test_data_path, ignore_errors=True)

    print("Copying training data ...")
    shutil.copytree(original_data_path, train_data_path)

    print("Copying testing data ...")
    shutil.copytree(original_data_path, test_data_path)

    def get_split():
        is_training = [True] * NUM_EACH_CLASS
        for i in range(num_of_testing):
            is_training[i] = False
        random.shuffle(is_training)
        return is_training

    is_training_dict = dict()

    print("Generating training data ...")
    for root, dirs, filenames in os.walk(train_data_path):
        if len(dirs) != 0:
            continue
        assert len(filenames) == NUM_EACH_CLASS
        is_training_dict[root] = get_split()
        i = 0
        for filename in filenames:
            data_point_path = os.path.join(root, filename)
            if not is_training_dict[root][i]:
                os.remove(data_point_path)
            i += 1

    print("Generating testing data ...")
    for root, dirs, filenames in os.walk(test_data_path):
        if len(dirs) != 0:
            continue
        assert len(filenames) == NUM_EACH_CLASS
        i = 0
        for filename in filenames:
            data_point_path = os.path.join(root, filename)
            if is_training_dict[root.replace("testing_data", "training_data")][i]:
                os.remove(data_point_path)
            i += 1


# generate_training_testing_split()