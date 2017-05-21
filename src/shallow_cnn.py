from keras.callbacks import EarlyStopping
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adadelta

from read_data import get_train_generator, get_test_generator, IMG_H, IMG_W

# generate_training_testing_split()

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(IMG_H, IMG_W, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(144))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
              metrics=['categorical_accuracy'])

train_generator = get_train_generator()
test_generator = get_test_generator()

print(model.summary())

earlystopper = EarlyStopping(monitor='loss', min_delta=1e-3, patience=20, verbose=0, mode='auto')

model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=500,
    validation_data=test_generator,
    validation_steps=25,
    workers=7,
    callbacks=[earlystopper])



# result:
# Epoch 258/500
# loss: 0.9328
# categorical_accuracy: 0.7513
# val_loss: 4.3313
# val_categorical_accuracy: 0.3126
