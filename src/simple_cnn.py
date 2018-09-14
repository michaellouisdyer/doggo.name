# Simple CNN with multi-GPU support
import os

import tensorflow as tf
from keras import models
from keras.callbacks import TensorBoard
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import nadam
from PIL import ImageFile
from tensorflow.python.lib.io import file_io

from ModelMGPU import ModelMGPU
from run_model import create_generators, save_class_names, train

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_model(input_size, n_categories):
    """
    Create a simple baseline CNN
    """

    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    model.add(Convolution2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    # transition to an mlp
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    project_name = 'simple_cnn'

    target_size = (224, 224)
    input_size = target_size + (3,)

    train_folder = 'data/train'
    validation_folder = 'data/validation'
    n_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(train_folder))

    # Initialize project name for saving model weights and stats
    project_name = 'simple-cnn'
    # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
    batch_size = 32
    epochs = 20  # number of iteration the algorithm gets trained.
    augmentation_strength = 0.2
    GPUS = True

    # Important to utilize CPUs even when training on multi-GPU instances as the CPUs can be a bottle neck when feeding to the GPUs
    CPUS = 16

    save_class_names(train_folder, project_name)

    model = create_model(input_size, n_categories)

    # Initialize a dictionary with the layer name and corresponding learning rate ratio

    # Initialize optimizer and compile the model
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    if GPUS:
        # Intialize a multi-GPU model utilizing keras.utils.multi_gpu_model and then overriding the save and load models so that we can save in a format that the model can be read as a serial model
        model = ModelMGPU(model)

    # Create data generators
    train_datagen, validation_datagen, train_generator, validation_generator, nTrain, nVal = create_generators(
        augmentation_strength, target_size, batch_size, train_folder, validation_folder, preprocessing_function=None)

    train(model, train_datagen, validation_datagen, train_generator,
          validation_generator, epochs, batch_size,  project_name, nTrain, nVal, CPUS)
