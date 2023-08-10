# This source code can be used to reproduce the results of article "Intermediate Task Fine-Tuning in Cancer Classification"

# params-------------------------------
imagenet = False # If True, initial weights are taken from a model pretrained with ImageNet dataset
model_folder = '/shared/PatoUTN/entrenamiento/Ejecuciones_atlas/Ejecucion_006/models/epoch_6' # initial model if imagenet == True
conf = 'C5'  # valid values: 'C1', 'C2', 'C3', 'C4', 'C5'
dataset_loader_seed = 123
epochs = 30
# -------------------------------------

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import datetime
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from classification_models.tfkeras import Classifiers

from sklearn.utils import class_weight

def preprocess(images, labels):
    return preprocess_input(images), labels

def expand_d(images, labels):
     return tf.expand_dims(images, axis=0), tf.expand_dims(labels, axis=0)

def save_confusion_matrix(ref_dataset, trained_model, filename):
     predictions = trained_model.predict(ref_dataset)
     y_pred = np.argmax(predictions, axis = 1)
     unbatch_dataset = ref_dataset.unbatch()
     labels = list(unbatch_dataset.map(lambda x, y: y))
     y_true = np.argmax(labels, axis=1)
     conf_mat = tf.math.confusion_matrix(y_true, y_pred, num_classes=5)
     np.save(filename, conf_mat.numpy())
     print(conf_mat.numpy())
     return conf_mat

def calculate_balanced_accuracy(conf_mat):
     cm = conf_mat.numpy()
     acclist = []
     total = cm.sum()
     for idx in range(cm.shape[0]):
          TP = cm[idx,idx]
          FP = cm[:,idx].sum() - TP
          FN = cm[idx,:].sum() - TP
          TN = total - TP - FN - FP
          acc = .5 * (TP/(TP+FN) + TN/(TN+FP))
          print(acc)
          acclist.append(acc)
     print('acc OVA media: ' + str(np.array(acclist).sum()/(cm.shape[0])))
     print('acc tradicional: ' + str(cm.diagonal().sum()/total))

start_time = datetime.datetime.now()
hyperparam = {}
hyperparam["PixelRangeShear"] = 5;         # max. xy translation (in pixels) for image augmenter

train_img_dir = '/shared/DATASETS/DeepHisto/train/'
test_img_dir = '/shared/DATASETS/DeepHisto/test/'

img_height = 224
img_width = 224

train_images = tf.keras.utils.image_dataset_from_directory(
  train_img_dir,
  labels='inferred',
  seed = dataset_loader_seed,
  image_size=(img_height, img_width),
  batch_size=256,
  label_mode='categorical'
  )

validation_images = tf.keras.utils.image_dataset_from_directory(
  test_img_dir,
  labels='inferred',
  image_size=(img_height, img_width),
  batch_size=256,
  label_mode='categorical',
  shuffle=False
  )

# class_weight
class_weights = {}
cant_x_clase = np.zeros(len(train_images.class_names))
for i, clase in enumerate(train_images.class_names):
  dir_list = os.listdir(train_img_dir + '/' + clase + '/')
  print (clase + ": " + str(len(dir_list)))
  cant_x_clase[i] = len(dir_list)
for i in range(len(cant_x_clase)):
  class_weights[i] = cant_x_clase.sum()/(cant_x_clase[i] * len(cant_x_clase))
print("class weights:")
print(class_weights)


hw_factor = hyperparam["PixelRangeShear"]/224
flip_layer = layers.RandomFlip("horizontal_and_vertical")
translation_layer = layers.RandomTranslation(height_factor=hw_factor, width_factor=hw_factor)
resizing_layer = layers.Resizing(img_height, img_width)

train_images_r = train_images.map(lambda x, y: (resizing_layer(x), y))
train_images_prep = train_images_r.map(preprocess)

validation_images_r = validation_images.map(lambda x, y: (resizing_layer(x), y))
validation_images_prep = validation_images_r.map(preprocess)

class_names = train_images.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_set_cache = train_images_prep.prefetch(buffer_size=AUTOTUNE)
validation_set_cache = validation_images_prep.prefetch(buffer_size=AUTOTUNE)

train_set_cache_da = train_set_cache.map(lambda x, y: (flip_layer(x), y))
train_set_cache_da = train_set_cache_da.map(lambda x, y: (translation_layer(x), y))

new_layer_output = Dense(len(class_names), activation='softmax', name='predictions')

if imagenet:
     ResNet18, preprocess_input = Classifiers.get('resnet18')
     model_orig = ResNet18((224, 224, 3), weights='imagenet')
     model = Model(model_orig.input, new_layer_output(model_orig.layers[-3].output))
else:
     model_orig = keras.models.load_model(model_folder)
     model = Model(model_orig.input, new_layer_output(model_orig.layers[-2].output))

conflayer = {'C1': -1,
             'C2': 27,
             'C3': 46,
             'C4': 65,
             'C5': 86}
ldx = 0
for layer in model.layers:
     ldx = ldx + 1
     if ldx < conflayer[conf]:
          layer.trainable = False
          print("no trainable - " + layer.name)
     else:
          print("trainable - " + layer.name)

model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
       
end_time = datetime.datetime.now()
print("Start time: {0} || End time: {1}".format(start_time, end_time))

continuar = True
ep_acum = 0
while continuar:
     start_time = datetime.datetime.now()
     for ep in range(epochs):
          model_filepath = 'models/epoch_' + str(ep_acum)
          model.fit(train_set_cache_da, validation_data=validation_set_cache, epochs=1, validation_freq=1, 
	                class_weight=class_weights)

          model.save(model_filepath)
          conf_mat = save_confusion_matrix(validation_set_cache, model, 'epoch_' + str(ep_acum) + '_cm')
          calculate_balanced_accuracy(conf_mat)
          ep_acum = ep_acum + 1

     end_time = datetime.datetime.now()
     print("Para " + str(epochs) + " epochs -- Start time: {0} || End time: {1}".format(start_time, end_time))

     cuanto = ''
     seguir = input("Continue? Y/n: ")
     if seguir == 'Y' or seguir == 'y':
          cuanto = input("How many more epochs? ")
          epochs = int(cuanto)
     else:
          continuar = False

