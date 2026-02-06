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
  model = None # Add code to define model 1.
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
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
  train_images = x_train[:-val_size]
  train_labels = y_train[:-val_size]
  val_images = x_train[-val_size:]
  val_labels = y_train[-val_size:]
  test_images = x_test
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

  print("Training accuracy: ", train_history1.history['accuracy'][-1])
  print("Validation accuracy: ", train_history1.history['val_accuracy'][-1])
  print("Test Accuracy: ", model1.evaluate(test_images, test_labels, verbose=2)[1])
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
