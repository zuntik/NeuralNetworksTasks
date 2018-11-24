import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd

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
#print(isolate_data[0])

print("\nEach class is represented by a number between 1 and 26")
print(isolate_data_class)

print("\n--TEST SET--")
print("\nWe have 1559 characters and classes, we have 300 attributes")
print(isolate_test.shape)

# To train the network we have to convert the classes in one out of 26 vectors
isolate_data_class_binary = np.eye(26)[isolate_data_class-1]
isolate_test_class_binary = np.eye(26)[isolate_test_class-1]
# Now we have 1559 binary vectors each one with a range from 1 to 26
print(isolate_data_class_binary)


#--Normalize data--
# normalize_data = (data - min(data)) / (max(data) - min(data))
isolate_data = ((isolate_data +1) / 2)
isolate_test = ((isolate_test +1) / 2)

# look model.fit documentation, you can put bacht dim there
#300 filas
#programar 1 neurona (entrada hiden output)

# mini bacth of size 40, with 300 features
# 300/40 = 7,5 ~ 7 neurons input

#-- TWO LAYERS --
# First layer connects the data with 7 neurons and the output layer connect the 7 neurons with the 26 classifiers
W_hid = tf.Variable(rd.randn(300,7),trainable=True)
b_hid = tf.Variable(np.zeros(7),trainable=True)

W_out = tf.Variable(rd.randn(7,26),trainable=True)
b_out = tf.Variable(np.zeros(26),trainable=True)

# El input de la primera capa, tenemos 300 col, es decir cada "caracter" tiene 300 dimensiones o caracteristicas
# La dimenson de las filas se inicializa en None pero si queremos recibir mas de un caracter introduciriamos ese numero en vez de None
x = tf.placeholder(shape=(None,300),dtype=tf.float64)

y = tf.nn.sigmoid(tf.matmul(x,W_hid) + b_hid)
z = tf.nn.softmax(tf.matmul(y,W_out) + b_out)

# Placeholder
z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# We create a variable for later initialize all variables
init = tf.initialize_all_variables()
# Create the session
sess = tf.Session()
# Re init variables to start from scratch
sess.run(init)

# Create some list to monitor how error decreases
test_loss_list = []
train_loss_list = []

test_acc_list = []
train_acc_list = []

# Create minibtaches to train faster
k_batch = 40
X_batch_list = np.array_split(isolate_data,k_batch)
labels_batch_list = np.array_split(isolate_data_class_binary,k_batch)

for k in range(700):
    # Run gradient steps over each minibatch
    for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})
        
    # Compute the errors over the whole dataset
    train_loss = sess.run(cross_entropy, feed_dict={x:isolate_data, z_:isolate_data_class_binary})
    test_loss = sess.run(cross_entropy, feed_dict={x:isolate_test, z_:isolate_test_class_binary})
    
    # Compute the acc over the whole dataset
    train_acc = sess.run(accuracy, feed_dict={x:isolate_data, z_:isolate_data_class_binary})
    test_acc = sess.run(accuracy, feed_dict={x:isolate_test, z_:isolate_test_class_binary})
    
    # Put it into the lists
    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    
    if np.mod(k,50) == 0:
        print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))




fig,ax_list = plt.subplots(1,2)
ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)
ax_list[0].plot(test_loss_list, color='green', label='testing', lw=2)
ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)

ax_list[0].set_xlabel('training iterations')
ax_list[1].set_xlabel('training iterations')
ax_list[0].set_ylabel('Cross-entropy')
ax_list[1].set_ylabel('Accuracy')
plt.legend(loc=2)
plt.show()




