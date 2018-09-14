# Evaluate model performance on a holdout set
import os
import pickle

import numpy as np
from keras.applications.xception import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from sklearn.metrics import classification_report, confusion_matrix

from Adam_lr_mult import Adam_lr_mult
from run_model import create_learn_rate_dict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def evaluate_model(model, holdout_generator, holdout_datagen, holdout_folder, batch_size, CPUS=4):
    """
    Evaluate accuracy on holdout set
    """
    accuracy = model.evaluate_generator(holdout_generator,
                                        steps=nHoldout/batch_size,
                                        use_multiprocessing=True,
                                        workers=CPUS,
                                        verbose=1)
    print(accuracy)
    with open(project_name+'.txt', 'w') as f:
        f.write(repr(accuracy))


def predict_classes(model, holdout_generator, batch_size, CPUS):
    Y_pred = model.predict_generator(
        holdout_generator, use_multiprocessing=True, workers=CPUS, verbose=1)
    """
    Create confusion matrix and classification report for holdout set
    """
    # Take the predicted label for each observation
    y_pred = np.argmax(Y_pred, axis=1)

    # Create confusion matrix and save it
    cm = confusion_matrix(holdout_generator.classes, y_pred)
    with open(project_name + '_cm.txt', 'wb') as f:
        pickle.dump(cm, f)

    # Create classification report and save it
    with open('doggo-app/models/class_names.pkl', 'rb') as f:
        class_names = np.array(pickle.load(f))

    class_report = classification_report(
        holdout_generator.classes, y_pred, target_names=class_names)
    print(classification_report)
    with open(project_name + '_cr.txt', 'w') as f:
        f.write(repr(class_report))


if __name__ == '__main__':

    # Load your trained model
    MODEL_PATH = 'models/simple-cnn.01-16.12.hdf5'

    model = load_model(MODEL_PATH, custom_objects={'Adam_lr_mult': Adam_lr_mult})

    project_name = 'my_model_eval'
    # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
    batch_size = 32
    CPUS = 16

    holdout_folder = 'data/validation'

    n_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(holdout_folder))
    nHoldout = sum(len(files) for _, _, files in os.walk(holdout_folder))
    target_size = (224, 224)
    input_size = target_size + (3,)

    holdout_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    holdout_generator = holdout_datagen.flow_from_directory(
        holdout_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    layer_mult = create_learn_rate_dict(model)

    # Compile model with custom optimizer
    adam_with_lr_multipliers = Adam_lr_mult(multipliers=layer_mult)

    model.compile(optimizer=adam_with_lr_multipliers,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    evaluate_model(model, holdout_generator, holdout_datagen, holdout_folder, batch_size, CPUS)

    predict_classes(model, holdout_generator, batch_size, CPUS)
