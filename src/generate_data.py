import os
from os.path import join, getsize

import shutil

project_path = "/home/sdjksdafji/Documents/others/Data558/"
train_data_dir="training_data"
test_data_dir="testing_data"

shutil.rmtree(project_path + train_data_dir)
print("hi")
# for root, dirs, files in os.walk('/home/sdjksdafji/Downloads/'):
#     print(root, "consumes", end=" ")
#     print(sum(getsize(join(root, name)) for name in files), end=" ")
#     print("bytes in", len(files), "non-directory files")