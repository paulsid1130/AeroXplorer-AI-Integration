import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
import random
from PIL import Image
import io
import shutil
import sys
import cv2
import base64

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files

import pathlib

def main(images):
    
    imArray = []
    for image_path in images:
        '''
        num = 0
        jpg_recovered = base64.decodeString(image_path)
        lol = "img" + str(num) + ".jpg"
        f = open(lol, "w")
        f.write(jpg_recovered)
        f.close()
        '''
        image = cv2.read(image_path)
        imArray.append(image)

    base_model = tf.keras.models.load_model('horizon.keras')
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    train_dir = os.path()
    BATCH_SIZE = 16
    IMG_SIZE = (150, 225)
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    feature_batch_average = global_average_layer(feature_batch)

if __name__ == "__main__":
    # sys.argv[0] is the script name, so we pass the rest to the main function
    main(sys.argv[1:])

def split_data(source, training, validation, split_size):
    all_files = []
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        if os.path.getsize(file_path) > 0:
            all_files.append(filename)
        else:
            print(f'Skipped {filename} because it is zero length')

    shuffled_set = random.sample(all_files, len(all_files))
    training_length = int(len(shuffled_set) * split_size)
    validation_length = len(shuffled_set) - training_length
    training_set = shuffled_set[:training_length]
    validation_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = os.path.join(source, filename)
        destination = os.path.join(training, filename)
        shutil.copyfile(this_file, destination)

    for filename in validation_set:
        this_file = os.path.join(source, filename)
        destination = os.path.join(validation, filename)
        shutil.copyfile(this_file, destination)


def prepare_data(base_dir):
    accepted_dir = os.path.join(base_dir, 'Accepted')
    rejected_dir = os.path.join(base_dir, 'Rejected')
    train_accepted_dir = os.path.join(base_dir, 'train/accepted')
    os.makedirs(train_accepted_dir, exist_ok=True)
    validation_accepted_dir = os.path.join(base_dir, 'validation/accepted')
    os.makedirs(validation_accepted_dir, exist_ok=True)

    train_rejected_dir = os.path.join(base_dir, 'train/rejected')
    os.makedirs(train_rejected_dir, exist_ok=True)
    validation_rejected_dir = os.path.join(base_dir, 'validation/rejected')
    os.makedirs(validation_rejected_dir, exist_ok=True)

    #80% training, 20%validation
    split_size = 0.8
    split_data(accepted_dir, train_accepted_dir, validation_accepted_dir, split_size)
    split_data(rejected_dir, train_rejected_dir, validation_rejected_dir, split_size)


def create_generators(base_dir, batch_size=16, target_size=(150, 225)):
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(base_dir, 'validation'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator
