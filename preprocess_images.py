from PIL import Image
import os
import numpy as np

IMG_PATH = "abstract_art_512"


def preprocess_and_save_images():
    new_height = 96
    new_width = 96
    paths = [os.path.join(IMG_PATH, file) for file in os.listdir(IMG_PATH)]
    resized_images = []

    for path in paths:
        image = Image.open(path).resize((new_width, new_height), Image.ANTIALIAS)
        resized_images.append(np.asarray(image))

    training_data = np.reshape(resized_images, (-1, new_width, new_height, 3))
    training_data = training_data.astype(np.float32)
    training_data = training_data / 127.5 - 1.

    np.save("training_data" + str(new_width) + ".npy", training_data)
    print("Training data saved to: training_data" + str(new_width) + ".npy")


if __name__ == "__main__":
    preprocess_and_save_images()
