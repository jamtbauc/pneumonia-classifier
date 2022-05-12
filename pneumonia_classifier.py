import numpy as np
import os
from PIL import Image, ImageOps
import random
from sklearn.model_selection import train_test_split
import sys

from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

def load_images():
    # create list for input images
    x_list = []
    # create list for output labels
    y_list = []
    
    # open current project directory
    with os.scandir(os.path.join(sys.path[0], "resized")) as subfolders:
        # folder contains two subfolders
        for sub in subfolders:
            print("Loading images from folder {}...".format(sub.name))
            # load all files from each folder
            with os.scandir(sub) as files:
                for file in files:
                    # use Pillow library to convert each image to numpy array
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