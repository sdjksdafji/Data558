from keras.applications.imagenet_utils import preprocess_input

from read_data import get_train_generator, show_image

import numpy as np

import matplotlib.pyplot as plt

ii = 0
for x_batch, y_batch in get_train_generator():
    x_batch_cp = np.copy(x_batch)
    prepro_batch_x = preprocess_input(x_batch_cp)
    print(x_batch.shape)
    print(prepro_batch_x.shape)
    print(y_batch.shape)
    for i in range(x_batch.shape[0]):
        print("Label: " + str(y_batch[i]))
        plt.imshow(x_batch[i])
        plt.show()
        plt.imshow(prepro_batch_x[i])
        plt.show()
        # show_image(x_batch[i], "Batch:" + str(ii) + ", Image:" + str(i))
    ii += 1
    if ii > 20:
        break
