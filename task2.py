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

isolate_data, isolate_data_class, isolate_test, isolate_test_class = load_isolet()



print("\n--DATA SET--")
print("\nWe have 6238 character and classes (each character belongs to a class)")
print("Each character is represented with 300 attributes ranging between -1 and 1")
print(isolate_data.shape)

print("\nEach class is represented by a number between 1 and 26")
print(isolate_data_class)

print("\n--TEST SET--")
print("\nWe have 1559 characters and classes, we have 300 attributes")
print(isolate_test.shape)

# To train the network we have to convert the classes in one out of 26 vectors
# We pass the previos data and write the dimension of the new vector (26 values per vector)
isolate_data_class_binary = np.eye(26)[isolate_data_class-1]
# Now we have 1559 binary vectors each one with a range from 1 to 26
print(isolate_data_class_binary)


#--Normalize data--
# normalize_data = (data - min(data)) / (max(data) - min(data))
isolate_data = ((isolate_data +1) / 2)

# look model.fit documentation, you can put bacht dim there

size_of_batch = 40
# the number of attributes is analogous the number of pixels of an input image
num_attributes = 300
# one class per leter
num_classes = 26


# Set the variables

W = tf.Variable(np.random.randn(num_attributes,num_classes), trainable=True)
b = tf.Variable(np.zeros(num_classes),trainable=True)

x = tf.placeholder(shape=(None,num_attributes),dtype=tf.float64)
y = tf.nn.softmax(tf.matmul(x,W) + b)


y_ = tf.placeholder(shape=(None,num_classes),dtype=tf.float64)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 


# Create an op that will initialize the variable on demand
init = tf.global_variables_initializer()
sess = tf.Session()


# init the variables to start
sess.run(init)

# Create lists for monitoring
test_error_list = []
train_error_list = []


# the total number of steps is given by the number of training examples / size
# of each batch
#   because the number of training examples is not a multiple of the size of 
#   each batch, we must do a whole division and the number of examples for the
#   last step will be the remainder of the division
for i in range(isolate_data.shape[0] // size_of_batch):
    # Compute a gradient step
    #   for this to be a stochastic gradient descent, the step must be 
    #   calculated of a different batch each time
    sess.run(train_step, feed_dict={x:isolate_data[i*size_of_batch:(i+1)*size_of_batch] , y_:isolate_data_class_binary[i*size_of_batch:(i+1)*size_of_batch] } )

    # Compute the losses on training and testing sets for monitoring
    train_err = sess.run(cross_entropy, feed_dict={x:isolate_data[i*size_of_batch:(i+1)*size_of_batch] , y_:isolate_data_class_binary[i*size_of_batch:(i+1)*size_of_batch]})
    test_err = sess.run(cross_entropy, feed_dict={x:isolate_data[i*size_of_batch:(i+1)*size_of_batch] , y_:isolate_data_class_binary[i*size_of_batch:(i+1)*size_of_batch]})
    test_error_list.append(test_err)
    train_error_list.append(train_err)



