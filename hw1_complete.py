#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# Test comment
## 

def build_model1():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(10, activation=None)
  ])
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10, activation=None)
  ])
  return model

def build_model3():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.SeparableConv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10, activation=None)
  ])
  
  return model

def build_model50k():
  model = model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.BatchNormalization(),

    layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),


    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation=None)
  ])
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  # Split a validation set from the training data (assuming the data is already shuffled)
  _size = x_train.shape[0]
  val_size = int(0.2 * _size)
  train_images = x_train[:-val_size] / 255.0
  train_labels = y_train[:-val_size]
  val_images = x_train[-val_size:] / 255.0
  val_labels = y_train[-val_size:]
  test_images = x_test / 255.0
  test_labels = y_test 
  
  ########################################


  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  print("Model 1 Summary:")
  print(model1.summary())

  train_history1 = model1.fit(train_images, train_labels, 
                              validation_data=(val_images, val_labels), 
                              epochs=30)
  
  plt.plot(train_history1.epoch, train_history1.history['accuracy'], label='train_accuracy')
  plt.plot(train_history1.epoch, train_history1.history['val_accuracy'], label='val_accuracy')
  plt.legend()

  print("Training accuracy: ", train_history1.history['accuracy'][-1])
  print("Validation accuracy: ", train_history1.history['val_accuracy'][-1])
  print("Test Accuracy: ", model1.evaluate(test_images, test_labels)[1])
  

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  print("Model 2 Summary:")
  print(model2.summary())
  train_history2 = model2.fit(train_images, train_labels, 
                                validation_data=(val_images, val_labels), 
                                epochs=30)

  plt.figure()
  plt.plot(train_history2.epoch, train_history2.history['accuracy'], label='train_accuracy')
  plt.plot(train_history2.epoch, train_history2.history['val_accuracy'], label='val_accuracy')
  plt.legend()

  print("Training accuracy: ", train_history2.history['accuracy'][-1])
  print("Validation accuracy: ", train_history2.history['val_accuracy'][-1])
  print("Test Accuracy: ", model2.evaluate(test_images, test_labels)[1])
  
  ## Test the model with a picture
  test_img = np.array(keras.utils.load_img(
    './dog.png',
    grayscale=False,
    color_mode='rgb',
    target_size=(32,32)))
  print("Prediction:", model2.predict(test_img.reshape(1, 32, 32, 3)))

  ## Build and train model 3
  model3 = build_model3()
  model3.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  print("Model 3 Summary:")
  print(model3.summary())
  train_history3 = model3.fit(train_images, train_labels, 
                                validation_data=(val_images, val_labels), 
                                epochs=30)

  plt.figure()
  plt.plot(train_history3.epoch, train_history3.history['accuracy'], label='train_accuracy')
  plt.plot(train_history3.epoch, train_history3.history['val_accuracy'], label='val_accuracy')
  plt.legend()

  print("Training accuracy: ", train_history3.history['accuracy'][-1])
  print("Validation accuracy: ", train_history3.history['val_accuracy'][-1])
  print("Test Accuracy: ", model3.evaluate(test_images, test_labels)[1])

  ## Build and train the 50k model
  model50k = build_model50k()
  model50k.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  print("50k Model Summary")
  print(model50k.summary())

  train_history50k = model50k.fit(train_images, train_labels, 
                                validation_data=(val_images, val_labels), 
                                epochs=15)

  plt.figure()
  plt.plot(train_history50k.epoch, train_history50k.history['accuracy'], label='train_accuracy')
  plt.plot(train_history50k.epoch, train_history50k.history['val_accuracy'], label='val_accuracy')
  plt.legend()

  print("Training accuracy: ", train_history50k.history['accuracy'][-1])
  print("Validation accuracy: ", train_history50k.history['val_accuracy'][-1])
  
  loss, acc = model50k.evaluate(test_images, test_labels)
  print("Test Accuracy: ", acc)

  model50k.save("best_model.h5")

