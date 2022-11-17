import os
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
# from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf
import pathlib
from PIL import Image
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from keras.preprocessing import image

# Definition du sizing des images
IMG_HEIGHT = 200
IMG_WIDTH = 200
batch = 3
nb_class = 3

# Import du dataset d'entraînement
print("The training dataset contains :")
training_dir = pathlib.Path("C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Train")
training_datasets = tf.keras.preprocessing.image_dataset_from_directory(
    training_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch
)
class_names = training_datasets.class_names
print(class_names)

# Import du dataset de test
print("The testing dataset contains :")
test_dir = pathlib.Path("C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Tests")
test_datasets = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch
)

class_names = test_datasets.class_names
print(class_names)
"""
plt.figure(figsize=(10, 10))
for images, labels in training_datasets.take(1):
  for i in range(batch):
    ax = plt.subplot(1, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()
"""

# Construction du modèle Convnet
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(128, 4, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 4, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 4, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16, 4, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dense(nb_class, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'], )

tensorboard_callback = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(
    training_datasets,
    validation_data=test_datasets,
    epochs=15,
    callbacks=[tensorboard_callback]
)

model.summary()

# Predicting Cats
print("Cats :")
imgs_path = "C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Tests/Cats/"
for img_path in os.listdir(imgs_path):
    img_to_predict = Image.open(os.path.join(imgs_path, img_path)).resize((IMG_WIDTH, IMG_HEIGHT))
    img_to_predict = np.expand_dims(img_to_predict, axis=0)
    res = model.predict(img_to_predict)
    print(model.predict(img_to_predict))
    res_class = np.argmax(res, axis=1)
    print(res_class)

# Predicting Dogs
print("Dogs :")
imgs_path = "C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Tests/Dogs/"
for img_path in os.listdir(imgs_path):
    img_to_predict = Image.open(os.path.join(imgs_path, img_path)).resize((IMG_WIDTH, IMG_HEIGHT))
    img_to_predict = np.expand_dims(img_to_predict, axis=0)
    res = model.predict(img_to_predict)
    print(model.predict(img_to_predict))
    res_class = np.argmax(res, axis=1)
    print(res_class)

# Predicting Foxs
print("Foxs :")
imgs_path = "C:/Users/ariel/OneDrive/Bureau/ESGI/M1/S1/DeepLearning/Projet/Datasets/Tests/Foxs/"
for img_path in os.listdir(imgs_path):
    img_to_predict = Image.open(os.path.join(imgs_path, img_path)).resize((IMG_WIDTH, IMG_HEIGHT))
    img_to_predict = np.expand_dims(img_to_predict, axis=0)
    res = model.predict(img_to_predict)
    print(model.predict(img_to_predict))
    res_class = np.argmax(res, axis=1)
    print(res_class)
