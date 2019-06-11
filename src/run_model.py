# transfer model
import argparse
import itertools
import os

import joblib
import keras
import numpy as np
import pandas as pd
from keras.applications import MobileNet, Xception
from keras.applications.xception import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from keras.utils.generic_utils import get_custom_objects
from PIL import ImageFile

from src.Adam_lr_mult import Adam_lr_mult
from src.ModelMGPU import ModelMGPU

# Load truncated files to avoid errors when running
ImageFile.LOAD_TRUNCATED_IMAGES = True

get_custom_objects().update({"Adam_lr_mult": Adam_lr_mult})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="model save name")
    return parser.parse_args()


class ClassificationNet(object):
    """Keras Image Classifier with added methods to create directory datagens and evaluate on holdout set
        """

    def __init__(self,  project_name, target_size, train_folder,
                 validation_folder, holdout_folder, optimizer='Adam',
                 model_fxn=None, augmentation_strength=0.1, preprocessing=None,
                 batch_size=16, GPUS=None, metrics=['accuracy']):
        """
        Initialize class with basic attributes

        Args:
        project_name (str): project name, used for saving models
        target_size (tuple(int, int)): size of images for input
        augmentation_strength (float): strength for image augmentation transforms
        batch_size(int): number of samples propogated throught network
        preprocessing(function(img)): image preprocessing function

            """
        self.project_name = project_name
        self.target_size = target_size
        self.input_size = self.target_size + (3,)  # target size with color chennels
        self.train_datagen = ImageDataGenerator()
        self.validation_datagen = ImageDataGenerator()
        self.augmentation_strength = augmentation_strength
        self.train_generator = None
        self.validation_generator = None
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.class_names = None
        self.training_weight_dict = None
        self.GPUS = GPUS
        self.metrics = metrics

        if not os.path.exists('models/' + self.project_name):
            os.makedirs("models/" + self.project_name)

        self._training_init(train_folder, validation_folder, holdout_folder, optimizer, model_fxn)

    def _init_data(self, train_folder, validation_folder, holdout_folder):
        """
        Initializes class data

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            holdout_folder(str): folder containing holdout data
            """
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.holdout_folder = holdout_folder

        self.n_train = sum(len(files) for _, _, files in os.walk(
            self.train_folder))  # : number of training samples

        self.n_val = sum(len(files) for _, _, files in os.walk(
            self.validation_folder))  # : number of validation samples

        self.n_holdout = sum(len(files) for _, _, files in os.walk(
            self.holdout_folder))  # : number of holdout samples

        self.n_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(
            self.train_folder))  # : number of categories

        self.set_class_names()  # : text representation of classes

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def _create_generators(self):
        """
        Create generators to read images from directory
            """

        # Set parameters for processing and augmenting images
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing,
            brightness_range=[0.2, 0.8],
            horizontal_flip=True,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength / 4,
            height_shift_range=self.augmentation_strength / 4,
            shear_range=self.augmentation_strength / 4,
            zoom_range=self.augmentation_strength / 4
        )
        # no need for augmentation on validation images
        self.validation_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing
        )

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        self.validation_generator = self.validation_datagen.flow_from_directory(
            self.validation_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

        self.holdout_generator = self.validation_datagen.flow_from_directory(
            self.holdout_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

    def create_compiled_model(self, optimizer, model_fxn):
        model = model_fxn(self.input_size, self.n_categories)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=self.metrics)

        return model

    def make_callbacks(self):
            # Initialize tensorboard for monitoring
        tensorboard = keras.callbacks.TensorBoard(log_dir="models/" + self.project_name,
                                                  histogram_freq=0, batch_size=self.batch_size,
                                                  write_graph=True, embeddings_freq=0)

        # Initialize model checkpoint to save best model
        self.savename = 'models/'+self.project_name+"/"+self.project_name+'.hdf5'
        mc = keras.callbacks.ModelCheckpoint(self.savename,
                                             monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=False, mode='auto', period=1)
        self.callbacks = [mc, tensorboard]

    def _training_init(self, train_folder, validation_folder, holdout_folder, optimizer, model_fxn):
        self._init_data(train_folder, validation_folder, holdout_folder)
        self._create_generators()
        self.make_callbacks()
        # self.model = self.create_compiled_model(optimizer, model_fxn)
        self.model = self.create_compiled_model(optimizer, model_fxn)

    def fit(self, epochs):
        """
        Fits the CNN to the data, then saves and predicts on best model

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            holdout_folder(str): folder containing holdout data
            model_fxn(function): function that returns keras Sequential classifier
            optimizer(keras optimizer): optimizer for training
            epochs(int): number of times to pass over data

        Returns:
            str: file path for best model
            """

        if self.GPUS:
            # Intialize a multi-GPU model utilizing keras.utils.multi_gpu_model and
            # then overriding the save and load models so that we can save in a
            # format that the model can be read as a serial model
            self.model = ModelMGPU(self.model)

        history = self.model.fit_generator(self.train_generator,
                                           class_weight=self.training_weight_dict,
                                           steps_per_epoch=self.n_train/self.batch_size,
                                           epochs=epochs,
                                           validation_data=self.validation_generator,
                                           validation_steps=self.n_val/self.batch_size,
                                           callbacks=self.callbacks)

        best_model = load_model(self.savename)
        self.model = best_model
        print("Evaluating model")
        accuracy = self.evaluate_model(best_model)
        print("accuracy")
        return self.savename

    def evaluate_model(self, model):
        """
        evaluates model on holdout data
        Args:
            model (keras classifier model): model to evaluate
        Returns:
            list(float): metrics returned by the model, typically [loss, accuracy]
            """

        metrics = model.evaluate_generator(self.holdout_generator,
                                           steps=self.n_holdout/self.batch_size,
                                           use_multiprocessing=True,
                                           verbose=1)
        print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
        return metrics

    def print_model_layers(self, model, indices=0):
        """
        prints model layers and whether or not they are trainable

        Args:
            model (keras classifier model): model to describe
            indices(int): layer indices to print from
        Returns:
            None
            """

        for i, layer in enumerate(model.layers[indices:]):
            print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

    def process_img(self, img_path):
        """
        Loads image from filename, preprocesses it and expands the dimensions because the
        model predict function expects a batch of images, not one image
        Args:
            img_path (str): file to load
        Returns:
            np.array: preprocessed image
        """
        original = load_img(filename, target_size=self.target_size)
        numpy_image = self.preprocessing(img_to_array(original))
        image_batch = np.expand_dims(numpy_image, axis=0)

        return image_batch

    def model_predict(self, img_path, model):
        """
        Uses an image and a model to return the names and the predictions of the top 3 classes

        Args:
            img_path (str): file to load
            model (keras classifier model): model to use for prediction

        Returns:
            str: top 3 predictions
            """
        im = self.process_img(img_path)
        preds = model.predict(im)
        top_3 = preds.argsort()[0][::-1][:3]  # sort in reverse order and return top 3 indices
        top_3_names = self.class_names[top_3]
        top_3_percent = preds[0][[top_3]]*100
        top_3_text = '\n'.join([f'{name}: {percent:.2f}%' for name,
                                percent in zip(top_3_names, top_3_percent)])
        return top_3_text

    def set_class_names(self):
        """
        Sets the class names, sorted by alphabetical order
        """
        self.category_df = pd.DataFrame([(len(files), os.path.basename(dirname))
                                         for dirname, _, files in os.walk(self.train_folder)]).drop(0)

        self.category_df.columns = ["n_images", "class"]
        self.category_df = self.category_df.sort_values("class")
        print(self.category_df)
        weights = self.category_df["n_images"]/self.category_df["n_images"].sum()
        self.training_weight_dict = dict(zip(range(weights.shape[0]), weights.values))
        self.class_names = list(self.category_df["class"])
        joblib.dump(self.class_names, 'models/'+self.project_name + "/class_names.joblib")


class LayerTransferClassificationNet(ClassificationNet):
    """Keras Image Classifier with added methods to create directory datagens and evaluate on holdout set
        """

    def create_learn_rate_dict(self, model, base_layer_learn_ratio=0.1,
                               final_layer_learn_ratio=1):
        """
        Since we're using a custom optimizer with a different learning rate for each layer,
        we need to initialize a dictionary of layer names and weights.
        Here I'm using one lower value for all but the last layer
        """
        layer_mult = dict(zip([layer.name for layer in model.layers],
                              itertools.repeat(base_layer_learn_ratio)))
        layer_mult[model.layers[-1].name] = final_layer_learn_ratio
        return layer_mult

    def add_new_last_layer(self, base_model, n_categories):
        """
        Takes a base model and adds a pooling and a softmax output based on the number of categories
        """
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(n_categories, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def update_base_model(self, model):
        """
        Prepare base model for transfer learning
        """
        model = self.add_new_last_layer(model, self.n_categories)
        return model

    def create_compiled_model(self, optimizer, model_fxn):

        model = self.update_base_model(model_fxn)
        layer_mult = self.create_learn_rate_dict(model)
        adam_with_lr_multipliers = optimizer(multipliers=layer_mult)
        model.compile(optimizer=adam_with_lr_multipliers,
                      loss='categorical_crossentropy', metrics=self.metrics)
        return model


def main(args):
    base_path = 'data'

    train_folder = f'{base_path}/train'
    validation_folder = f'{base_path}/validation'
    holdout_folder = f'{base_path}/holdout'

    target_size = (299, 299)
    epochs = 20
    batch_size = 32

    model_fxn = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=target_size + (3,))
    opt = Adam_lr_mult

    lt_model = LayerTransferClassificationNet(args.model_name, target_size=target_size, train_folder=train_folder,
                                              validation_folder=validation_folder, holdout_folder=holdout_folder,
                                              model_fxn=model_fxn, optimizer=opt, augmentation_strength=0.4,
                                              preprocessing=preprocess_input,  batch_size=batch_size)

    lt_model.fit(epochs)


if __name__ == '__main__':
    main(parse_args())
