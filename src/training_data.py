from read_data import get_train_generator, show_image

ii = 0
for x_batch, y_batch in get_train_generator():
    print(x_batch.shape)
    print(y_batch.shape)
    for i in range(x_batch.shape[0]):
        print("Label: " + str(y_batch[i]))
        show_image(x_batch[i], "Batch:" + str(ii) + ", Image:" + str(i))
    ii += 1
    if ii > 20:
        break
