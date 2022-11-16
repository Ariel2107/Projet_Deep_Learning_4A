import os
#import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
#from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf
import pathlib

# Definition du sizing des images
IMG_HEIGHT = 200
IMG_WIDTH = 200
batch = 2

# Import du dataset d'entraînement
training_dir = pathlib.Path("C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Train")
training_datasets = tf.keras.preprocessing.image_dataset_from_directory(
    training_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch
)

# Import du dataset de test
test_dir = pathlib.Path("C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Tests")
test_datasets = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch
)

class_names = training_datasets.class_names
print(class_names)

