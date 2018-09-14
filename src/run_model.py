# transfer model
import itertools
import os
import pickle
from glob import glob

import keras
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

from Adam_lr_mult import Adam_lr_mult
from ModelMGPU import ModelMGPU

# Load truncated files to avoid errors when running
ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_class_names(train_folder, project_name):
    """
    Keras sorts the class folders alphabetically, let's save these for later use
    """
    names = [os.path.basename(x) for x in glob(train_folder + '/*')]
    class_names = sorted(names)
    with open(f'{project_name}-class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)


def add_new_last_layer(base_model, n_categories):
    """
    Takes a base model and adds a pooling and a softmax output based on the number of categories
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_categories, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def update_base_model(model):
    """
    Prepare base model for transfer learning
    """
    n_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(train_folder))
    model = add_new_last_layer(base_model, n_categories)
    return model


def train(model, train_datagen, validation_datagen, train_generator, validation_generator, epochs, batch_size, project_name, nTrain, nVal, CPUS=3):
    """
    Trains the model
    """

    # Set up log dir and create tensorboard callback
    logdir = './{}/train'.format(project_name)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=0, batch_size=batch_size, write_graph=True, embeddings_freq=0)

    # Create a callback for each model checkpoint
    mc = keras.callbacks.ModelCheckpoint('models/'+project_name+'.{epoch:02d}-{val_loss:.2f}.hdf5',
                                         monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nTrain/batch_size,
                                  use_multiprocessing=True,
                                  workers=CPUS,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  validation_steps=nVal,
                                  callbacks=[tensorboard, mc])

    model.save('models/{}end.h5'.format(project_name))


def create_learn_rate_dict(model):
    """
    Since we're using a custom optimizer with a different learning rate for each layer, we need to initialize a dictionary of layer names and weights. Here I'm using one lower value for all but the last layer
    """
    base_layer_learn_ratio = 0.1
    final_layer_learn_ratio = 1
    layer_mult = dict(zip([layer.name for layer in model.layers],
                          itertools.repeat(base_layer_learn_ratio)))
    layer_mult[model.layers[-1].name] = final_layer_learn_ratio
    return layer_mult


def create_generators(augmentation_strength, target_size, batch_size, train_folder, validation_folder, preprocessing_function=preprocess_input):

    # Get number of training images and number of validation images
    nTrain = sum(len(files) for _, _, files in os.walk(train_folder))
    nVal = sum(len(files) for _, _, files in os.walk(validation_folder))

    # Set parameters for processing and augmenting images
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=30,
        width_shift_range=augmentation_strength,
        height_shift_range=augmentation_strength,
        shear_range=augmentation_strength,
        zoom_range=augmentation_strength,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
    )

    # Setup pipeline to give images to the fit_generator
    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        validation_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return train_datagen, validation_datagen, train_generator, validation_generator, nTrain, nVal


if __name__ == '__main__':

    # Initialize project name for saving model weights and stats
    project_name = 'Xception-Dog-With-Weights'
    # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
    batch_size = 32
    epochs = 20  # number of iteration the algorithm gets trained.
    augmentation_strength = 0.2
    GPUS = True

    # Important to utilize CPUs even when training on multi-GPU instances as the CPUs can be a bottle neck when feeding to the GPUs
    CPUS = 16

    train_folder = 'data/train'
    validation_folder = 'data/test'

    save_class_names(train_folder, project_name)

    target_size = (299, 299)
    input_size = target_size + (3,)
    base_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=input_size)

    model = update_base_model(base_model)

    # Initialize a dictionary with the layer name and corresponding learning rate ratio
    layer_mult = create_learn_rate_dict(model)

    # Initialize optimizer and compile the model
    adam_with_lr_multipliers = Adam_lr_mult(multipliers=layer_mult)
    model.compile(optimizer=adam_with_lr_multipliers,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    if GPUS:
        # Intialize a multi-GPU model utilizing keras.utils.multi_gpu_model and then overriding the save and load models so that we can save in a format that the model can be read as a serial model
        model = ModelMGPU(model)

    # Create data generators
    train_datagen, validation_datagen, train_generator, validation_generator, nTrain, nVal = create_generators(
        augmentation_strength, target_size, batch_size, train_folder, validation_folder)

    train(model, train_datagen, validation_datagen, train_generator,
          validation_generator, epochs, batch_size, project_name, nTrain, nVal, CPUS)
