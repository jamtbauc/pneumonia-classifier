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
    with os.scandir(os.path.join(sys.path[0], "all")) as subfolders:
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

def reshape_input(x_list, y_list, num_classes):
    random.Random(42).shuffle(x_list)
    random.Random(42).shuffle(y_list)

    x_data = np.asarray(x_list)
    y_data = np.asarray(y_list)

    assert x_data.shape == (5856, 100, 100)
    assert y_data.shape == (5856, )

    # Split dataset into train, val and test datasets
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.20, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)

    # define image dimensions
    img_rows, img_cols = 100

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    ### normalize the input
    x_train = x_train.astype('float32') # Copy of the array, cast to a specified type.
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_val /= 255
    x_test /= 255

    ### convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_val, x_test, y_train, y_val, y_test, input_shape

def design_cnn(input_shape, num_classes):
    ### design the model 
    model = Sequential()

    model.add(Conv2D(filters=32,
                    kernel_size=(3,3),
                    strides=(2,2),
                    padding="valid",
                    activation='relu',
                    input_shape=input_shape,
                    kernel_regularizer="l2",
                    bias_regularizer="l2"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),(2,2),activation="relu",kernel_regularizer="l2",bias_regularizer="l2"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

def run_cnn():
    #initialize number of classes
    num_classes = 2

    # load images
    x_list, y_list = load_images()

    # normalize input arrays and convert class vectors to binary class matrices
    x_train, x_val, x_test, y_train, y_val, y_test, input_shape = reshape_input(x_list, y_list, 2)

    # design model
    initial_model = design_cnn(input_shape, num_classes)

    ### train the model
    #model.compile(loss=keras.losses.categorical_crossentropy,  # https://keras.io/api/losses/
    #            optimizer=keras.optimizers.SGD(),  # https://keras.io/api/optimizers/
    #            metrics=['accuracy'])

    #model.fit(x_train, y_train,
     #       batch_size=batch_size,
      #      epochs=epochs,
       #     verbose=1,
        #    validation_data=(x_val, y_val))

    #score = model.evaluate(x_test, y_test, verbose=0)

    #print('Test accuracy:', score[1])

if __name__ == "__main__":
    run_cnn()