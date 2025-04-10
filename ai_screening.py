import matplotlib.pyplot as plt
#import cv2
import numpy as np
import PIL
import tensorflow as tf
import os
import random
from PIL import Image
import io
import shutil

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pathlib

def main(method_name, image_path):
    try:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")
        
        if method_name == 'expose':
            result = exposure_model(image_path)
            print(result)
        elif method_name == 'horizon':
            result = horizon_model(image_path)
            print(result)
        elif method_name == 'crop':
            result = crop_model(image_path)
            print(result)
        else:
            raise ValueError(f"Method '{method_name}' is not recognized.")
    except Exception as e:
        print(f"An error occurred: {e}")

def exposure_model(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(100, 150))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        model = tf.keras.models.load_model('expose.keras')
        try:
            prediction = model.predict(img_array)[0]
        except Exception as e:
            return f"Prediction error in expo_model: {e}"
        if prediction >= 0.5:
            return 'Rejected'
        else:
            return 'Accepted'
    except Exception as e:
        return f"An error occurred in exposure_model: {e}"

def horizon_model(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(150, 225))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        model = tf.keras.models.load_model('horizon.keras')
        try:
            prediction = model.predict(img_array)[0]
        except Exception as e:
            return f"Prediction error in horizon_model: {e}"
        if prediction >= 0.5:
            return 'Rejected'
        else:
            return 'Accepted'
    except Exception as e:
        return f"An error occurred in horizon_model: {e}"

def crop_model(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(100, 150))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        model = tf.keras.models.load_model('content.keras')
        try:
            prediction = model.predict(img_array)[0]
        except Exception as e:
            return f"Prediction error in crop_model: {e}"
        if prediction >= 0.5:
            return 'Rejected'
        else:
            return 'Accepted'
    except Exception as e:
        return f"An error occurred in crop_model: {e}"

#main('ASDAS', 'IMG_8214.jpg')
#main('expose', 'IMG_8214.jpg')
#main('horizon', 'IMG_8214.jpg')
#$command = escapeshellcmd("python3 /ai_screening.py . " . $methodName . " " . $imagePath)

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

base_dir = ''
prepare_data(base_dir)
train_generator, validation_generator = create_generators(base_dir)

class_names = train_generator.classes

checkpoint_path = "training_1/cp.ckpt.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def build_model(input_shape=(150, 225, 3)):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs=9):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[cp_callback]
    )
    return history

model = build_model()
history = train_model(model, train_generator, validation_generator)
model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(9)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

airplane_url = "https://cdn.aeroxplorer.com/large/ANPaDn0wLlFNqZDAhkho.jpg"
airplane_path = tf.keras.utils.get_file('ANPaDn0wLlFNqZDAhkho', origin=airplane_url)

img = tf.keras.utils.load_img(
    airplane_path, target_size=(150, 225)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)