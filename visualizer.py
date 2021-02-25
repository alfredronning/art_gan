import tkinter as tk
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import numpy as np


class Visualizer:
    def __init__(self, master, model_path):
        self.root = master
        self.generator = load_model(model_path, compile=False)
        self.sliders = [tk.Scale(from_=100, to=-100, command=self.live_sliders) for _ in range(100)]
        for i in range(25):
            self.sliders[i].grid(row=0, column=i+1)
        for i in range(25, 50):
            self.sliders[i].grid(row=1, column=i - 24)
        for i in range(50, 75):
            self.sliders[i].grid(row=2, column=i - 49)
        for i in range(75, 100):
            self.sliders[i].grid(row=3, column=i - 74)
        self.panel = None
        self.generate_from_sliders()

        # button to change image
        button = tk.Button(master, text="Generate", command=self.generate_from_sliders)
        button.grid(row=1, column=0)

        random_button = tk.Button(master, text="Random Seed", command=self.random_seed)
        random_button.grid(row=2, column=0)

    def random_seed(self):
        seed = tf.random.normal([1, 100]).numpy()[0]
        for i in range(100):
            self.sliders[i].set(seed[i]*25)
        self.generate_from_sliders()


    def live_sliders(self, event):
        self.generate_from_sliders()


    def get_slider_values(self):
        return [[slider.get()/25 for slider in self.sliders]]


    def generate_from_sliders(self):
        input_tensor = tf.convert_to_tensor(self.get_slider_values(), dtype=None, dtype_hint=None, name=None)
        output = self.generator(input_tensor, training=False)
        image_arr = output.numpy()[0, :, :, :]
        image_arr = (0.5 * image_arr + 0.5) * 255
        img = Image.fromarray(image_arr.astype(np.uint8))
        img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        if not self.panel:
            self.panel = tk.Label(self.root, image=img)
        else:
            self.panel.configure(image=img)
        self.panel.image = img
        self.panel.grid(row=0, column=0)

    def display_image(self, img_array):
        plt.imshow(img_array[0, :, :, 0])
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    visualizer = Visualizer(root, "saved_models/generator_0.h5")
    root.mainloop()
