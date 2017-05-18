import os
import random

import shutil

import zlib


def generate_training_testing_split(split=(8, 2)):
    project_path = "/home/sdjksdafji/Documents/others/Data558/"
    original_data_dir = "data/train"
    train_data_dir = "training_data"
    test_data_dir = "testing_data"

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

    hash_salt = str(random.random())
    training_split, testing_split = split

    print("Generating training data ...")
    for root, dirs, filenames in os.walk(train_data_path):
        for filename in filenames:
            data_point_path = os.path.join(root, filename)
            hash_value = zlib.adler32(str.encode(filename + hash_salt))
            if hash_value % (training_split + testing_split) >= training_split:
                os.remove(data_point_path)

    print("Generating testing data ...")
    for root, dirs, filenames in os.walk(test_data_path):
        for filename in filenames:
            data_point_path = os.path.join(root, filename)
            hash_value = zlib.adler32(str.encode(filename + hash_salt))
            if hash_value % (training_split + testing_split) < training_split:
                os.remove(data_point_path)
