import os

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dropout, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import numpy as np
from PIL import Image
import time


def load_training_set(path, batch_size):
    training_data = np.load(path)
    training_set = tf.data.Dataset.from_tensor_slices(training_data).shuffle(9000).batch(batch_size)
    return training_set


def save_images(epoch, seed, image_res, generator, rows, cols):
    margin = 16
    image_array = np.full((
        margin + (rows * (image_res + margin)),
        margin + (cols * (image_res + margin)), 3),
        255, dtype=np.uint8)

    generated_images = generator.predict(seed)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(rows):
        for col in range(cols):
            r = row * (image_res + 16) + margin
            c = col * (image_res + 16) + margin
            image_array[r:r + image_res, c:c + image_res] \
                = generated_images[row * cols + col] * 255

    output_path = os.path.join("generator_images", 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"train-{epoch}.png")
    im = Image.fromarray(image_array)
    im.save(filename)


class ArtGen:
    def __init__(self, training_data, batch_size=16, image_res=128, seed_size=100, learning_rate=1.5e-4):
        self.training_data = training_data
        self.image_res = image_res
        self.seed_size = seed_size
        self.batch_size = batch_size

        self.generator = None
        self.build_generator()

        self.discriminator = None
        self.build_discriminator()

        self.generator_optimizer = Adam(learning_rate, 0.5)
        self.discriminator_optimizer = Adam(learning_rate, 0.5)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(4 * 4 * 256, activation="relu", input_dim=self.seed_size))
        model.add(Reshape((4, 4, 256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # Output resolution, additional upsampling
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D(size=(4, 4)))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # Final CNN layer
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        self.generator = model

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(self.image_res, self.image_res, 3),
                         padding="same"))
        model.add(Activation("relu"))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        self.discriminator = model

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        real_loss = BinaryCrossentropy()(tf.ones_like(real_output), real_output)
        fake_loss = BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    def generator_loss(fake_output):
        return BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def hms_string(seconds):
        hours = seconds // (60 * 60)
        seconds %= (60 * 60)
        minutes = seconds // 60
        seconds %= 60
        return "%02i:%02i:%02i" % (hours, minutes, seconds)

    @tf.function
    def train_step(self, images, pause=0):
        seed = tf.random.normal([self.batch_size, self.seed_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(seed, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            if pause != 2:
                self.generator_optimizer.apply_gradients(zip(
                    gradients_of_generator, self.generator.trainable_variables))
            if pause != 1:
                self.discriminator_optimizer.apply_gradients(zip(
                    gradients_of_discriminator,
                    self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, epochs):
        seed = np.random.normal(0, 1, (7 * 4, self.seed_size))
        start = time.time()
        # pause = 1 will pause the discriminator, 2 will pause the generator
        pause = 0
        save_images(-1, seed, self.image_res, self.generator, 4, 7)
        for epoch in range(epochs):
            epoch_start = time.time()

            gen_loss_list = []
            disc_loss_list = []

            for image_batch in self.training_data:
                t = self.train_step(image_batch, pause)
                gen_loss_list.append(t[0])
                disc_loss_list.append(t[1])

            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)

            # pause the discriminator learning if it is outlearning the generator and vice versa
            if g_loss / d_loss > 3:
                pause = 1
            elif d_loss / g_loss > 3:
                pause = 2
            else:
                pause = 0

            epoch_time = self.hms_string(time.time() - epoch_start)
            total_time = self.hms_string(time.time() - start)
            print(f'Epoch {epoch + 1}, gen_loss={g_loss}, disc_loss={d_loss},' \
                  f'Time: {epoch_time}, total_time={total_time}')

            seed = np.random.normal(0, 1, (7 * 4, self.seed_size))
            save_images(epoch, seed, self.image_res, self.generator, 4, 7)

            if epoch % 500 == 0:
                self.generator.save(os.path.join("saved_models", "generator_" + str(epoch) + ".h5"))
                self.discriminator.save(os.path.join("saved_models", "discriminator_" + str(epoch) + ".h5"))

        elapsed = time.time() - start
        print(f'Training time: {self.hms_string(elapsed)}')


def main():
    batch_size = 16
    image_res = 128
    epochs = 2000
    training_data = load_training_set("training_data" + str(image_res) + ".npy", batch_size)
    art_generator = ArtGen(training_data, batch_size)
    art_generator.train(epochs)


if __name__ == "__main__":
    main()
