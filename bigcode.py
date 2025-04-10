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

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files

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
    

if __name__ == "__main__":
    # sys.argv[0] is the script name, so we pass the rest to the main function
    main(sys.argv[1], sys.argv[2])
