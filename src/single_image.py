from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from read_data import show_image

image_path = "/home/sdjksdafji/Documents/others/Data558/data/train/" \
             "001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg"

datagen = ImageDataGenerator(
    rotation_range=160,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

img = load_img(image_path)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# print original image
show_image(x[0], "Original")

# print manipulated images
i = 0
for batch in datagen.flow(x, batch_size=1):
    show_image(batch[0], "Manipulated")
    i += 1
    if i > 20:
        break
