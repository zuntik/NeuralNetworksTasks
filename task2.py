import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Data
# Character a...z
# 26 classes

def load_isolet():
  # Loads the isolet dataset
  # Returns:
  # X....feature vectors (training set), X[i,:] is the i-th example
  # C....target classes
  # X_tst...feature vectors (test set)
  # C_tst...classes (test set)
  
  import pickle as pckl  # to load dataset
  import pylab as pl     # for graphics
  #from numpy import *    

  pl.close('all')   # closes all previous figures

  # Load dataset
  file_in = open('isolet_crop_train.pkl','rb')
  isolet_data = pckl.load(file_in) # Python 3
  #isolet_data = pckl.load(file_in, encoding='bytes') # Python 3
  file_in.close()
  X = isolet_data[0]   # input vectors X[i,:] is i-th example
  C = isolet_data[1]   # c.lasses C[i] is class of i-th example

  file_in = open('isolet_crop_test.pkl','rb')
  isolet_test = pckl.load(file_in) # Python 3
  file_in.close()

  X_tst = isolet_test[0]   # input vectors X[i,:] is i-th example
  C_tst = isolet_test[1]   # classes C[i] is class of i-th example

  return (X, C, X_tst, C_tst)

x_training, class_training, x_test, class_test = load_isolet()

#number of images on the training set
print("\nNumber of images on the training set")
print(x_training.shape)
#number of labels on the training set
print("\nNumber of classes on the training set")
print(len(class_training))
# Each label is an intenger between 0 and 9
print("\nEach class is a number between 1 and 26")
print(class_training)

#Number of images and labels on the test set
print("\nNumber of images on the test set")
print(x_test.shape)
print("\nNumber of classes on the test set")
print(len(class_test))