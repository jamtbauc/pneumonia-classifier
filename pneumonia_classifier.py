import numpy as np
import os
from PIL import Image, ImageOps
import random
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

def load_images():
    folder = '/home/jamtbauc/workspace/pneumonia/resized'

    x_list = []
    y_list = []

    with os.scandir(folder) as subfolders:
        for sub in subfolders:
            print("Loading images from folder {}...".format(sub.name))
            with os.scandir(sub) as files:
                for file in files:
                    image = Image.open(file.path)
                    img_data = np.asarray(image)
                    
                    x_list.append(img_data)
                    
                    if sub.name == "normal":
                        y_list.append(0)
                    else:
                        y_list.append(1)

        print("Finished loading images.")
    
    return x_list, y_list

def run_cnn():
    x_list, y_list = load_images()

    random.Random(42).shuffle(x_list)
    random.Random(42).shuffle(y_list)

    x_data = np.asarray(x_list)
    y_data = np.asarray(y_list)

    assert x_data.shape == (5856, 100, 100)
    assert y_data.shape == (5856, )

if __name__ == "__main__":
    run_cnn()