import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
isolate_test_class_binary = np.eye(26)[isolate_test_class-1]
# Now we have 1559 binary vectors each one with a range from 1 to 26
print(isolate_data_class_binary)


#--Normalize data--
# normalize_data = (data - min(data)) / (max(data) - min(data))
isolate_data = ((isolate_data +1) / 2)

# look model.fit documentation, you can put bacht dim there

# the number of attributes is analogous the number of pixels of an input image
num_attributes = isolate_test.shape[1]
# one class per leter
num_classes = isolate_data_class_binary.shape[1]


###############################################################################
# Define the network
###############################################################################
# There we define the number of hidden layers as well as the number of nodes
# each hidden layer has
# user input is only here
# uncomment the network to use
# notes:
#     y is always the last layer and x is always the first for the rest
#     of the code to work
###############################################################################
# basic scenario: one layer and cross entropy 

# W = tf.Variable(np.random.randn(num_attributes,num_classes), trainable=True)
# b = tf.Variable(np.zeros(num_classes),trainable=True)
# 
# # define how to link the variables
# x = tf.placeholder(shape=(None,num_attributes),dtype=tf.float64)
# y = tf.nn.softmax(tf.matmul(x,W) + b)

# 2 layers scenario

# the number of neuros of the hidden layer is the only thing we can change
num_hidden_neurons = 30

W = tf.Variable(np.random.randn(num_attributes,num_hidden_neurons), trainable=True)
b = tf.Variable(np.zeros(num_hidden_neurons),trainable=True)

W_out = tf.Variable(np.random.randn(num_hidden_neurons, num_classes) ,trainable=True)
b_out = tf.Variable(np.zeros(num_classes))

x = tf.placeholder(shape=(None,num_attributes),dtype=tf.float64)
z = tf.nn.tanh(tf.matmul(x,W) + b)
y = tf.nn.softmax(tf.matmul(z,W_out) + b_out )


###############################################################################


y_ = tf.placeholder(shape=(None,num_classes),dtype=tf.float64)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 


# Create an op that will initialize the variable on demand
init = tf.global_variables_initializer()
sess = tf.Session()


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# init the variables to start
sess.run(init)

# Create lists for monitoring
test_error_list = []
train_error_list = []

test_acc_list = []
train_acc_list = []

# we must apply the stochastic gradient descent

# doing number of examples / number of batches iterations will not be enough
# to get a sufficiently small error so we must iterate over the batches several
# times
num_iterations = 700
size_of_a_batch = 40

X_batch_list = np.array_split(isolate_data, size_of_a_batch)
labels_batch_list = np.array_split(isolate_data_class_binary, size_of_a_batch)

for _ in itertools.repeat(None, num_iterations):

    # Run gradient steps over each minibatch
    for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, y_:labels_minibatch})# Compute a gradient step

    # Compute the errors over the whole dataset
    train_err = sess.run(cross_entropy, feed_dict={x:isolate_data, y_:isolate_data_class_binary})
    test_err = sess.run(cross_entropy, feed_dict={x:isolate_test, y_:isolate_test_class_binary})
    
    # Compute the acc over the whole dataset
    train_acc = sess.run(accuracy, feed_dict={x:isolate_data, y_:isolate_data_class_binary})
    test_acc = sess.run(accuracy, feed_dict={x:isolate_test, y_:isolate_test_class_binary})

    # Put it into the lists
    test_error_list.append(test_err)
    train_error_list.append(train_err)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)


fig,ax_list = plt.subplots(1,2)
ax_list[0].plot(train_error_list, color='blue', label='training', lw=2)
ax_list[0].plot(test_error_list, color='green', label='testing', lw=2)
ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)

ax_list[0].set_title('Cross-entropy')
ax_list[0].set_xlabel('Training epoch')
ax_list[0].set_ylabel('Cross-entropy')
ax_list[1].set_title('Accuracy')
ax_list[1].set_xlabel('Training epoch')
ax_list[1].set_ylabel('Accuracy')
plt.legend(loc=2)

plt.show()
