# Authors         MatrikulNumber 
# Javier Bezares  11805954
# Thomas Berry    11806027


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import nn18_ex2_load

isolate_data, isolate_data_class, isolate_test, isolate_test_class = nn18_ex2_load.load_isolet()

# Recieves a placeholder x, and the number of inputs and outputs
# returns the W matrix, the bias and the sigmoid fuction
def layer(x, n_input, n_output, activation):
  W = tf.Variable(rd.randn(n_input,n_output)/np.sqrt(n_input),trainable=True)
  b = tf.Variable(np.zeros(n_output),trainable=True)
  y = activation(tf.matmul(x,W) + b)
  return y

def residual_block_layer(x, n_input, n_output, activation1, activation2):
  # the number of inputs has to equal the number of outputs
  # check if the input and output are the same
  assert(n_input==n_output)
  W1 = tf.Variable( rd.randn(n_input, n_output)/np.sqrt(n_input), trainable=True)
  b1 = tf.Variable(np.zeros(n_output), trainable=True)
  f = activation1(tf.matmul(x,W1) + b1)

  W2 = tf.Variable(rd.randn(n_input, n_output)/np.sqrt(n_input), trainable=True)
  b2 = tf.Variable(np.zeros(n_output), trainable=True)
  y = activation2(x + tf.matmul(f,W2) + b2)
  return y

#print("\n--DATA SET--")
#print("\nWe have 6238 character and classes (each character belongs to a class)")
#print("Each character is represented with 300 attributes ranging between -1 and 1")
#print(isolate_data.shape)

#print("\nEach class is represented by a number between 1 and 26")
#print(isolate_data_class)

#print("\n--TEST SET--")
#print("\nWe have 1559 characters and classes, we have 300 attributes")
#print(isolate_test.shape)

# To train the network we have to convert the classes in one out of 26 vectors
isolate_data_class_binary = np.eye(26)[isolate_data_class-1]
isolate_test_class_binary = np.eye(26)[isolate_test_class-1]
# Now we have 1559 binary vectors each one with a range from 1 to 26
#print(isolate_data_class_binary)


#--Normalize data--
# normalize_data = (data - min(data)) / (max(data) - min(data))
isolate_data = ((isolate_data +1) / 2)
isolate_test = ((isolate_test +1) / 2)



#num_neuron = 150

x = tf.placeholder(shape=(None, 300),dtype=tf.float64)

#-- 9 LAYERS --
#y = layer(x, 300, 40, tf.nn.tanh)
#y2 = layer(y, 40, 40, tf.nn.tanh)
#y3 = layer(y2, 40, 40, tf.nn.tanh)
#y4 = layer(y3, 40, 40, tf.nn.tanh)
#y5 = layer(y4, 40, 40, tf.nn.tanh)
#y6 = layer(y5, 40, 40, tf.nn.tanh)
#y7 = layer(y6, 40, 40, tf.nn.tanh)
#y8 = layer(y7, 40, 40, tf.nn.tanh)
#z = layer(y8, 40, 26, tf.nn.softmax)


#--relu + resnet 
y = layer(x,300, 40, tf.nn.relu)
y2 = residual_block_layer(y, 40, 40,  tf.nn.relu, tf.nn.relu)
y3 = residual_block_layer(y2, 40, 40, tf.nn.relu, tf.nn.relu)
y4 = residual_block_layer(y3, 40, 40, tf.nn.relu, tf.nn.relu)
y5 = residual_block_layer(y4, 40, 40, tf.nn.relu, tf.nn.relu)
z = layer(y5, 40, 26, tf.nn.softmax)


z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
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
k_batch = 20
X_batch_list = np.array_split(isolate_data,k_batch)
labels_batch_list = np.array_split(isolate_data_class_binary,k_batch)

for k in range(1000):
  # Run gradient steps over each minibatch
  for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
      sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})
        
  # Compute the errors over the whole dataset
  train_loss = sess.run(cross_entropy, feed_dict={x:isolate_data, z_:isolate_data_class_binary})
  test_loss = sess.run(cross_entropy, feed_dict={x:isolate_test, z_:isolate_test_class_binary})
    
  # Compute the acc over the whole dataset
  train_acc = sess.run(accuracy, feed_dict={x:isolate_data, z_:isolate_data_class_binary})
  test_acc = sess.run(accuracy, feed_dict={x:isolate_test, z_:isolate_test_class_binary})
    
  #if test_loss_list != []:
    #if  test_loss > test_loss_list[-1]:
      #break

  # Put it into the lists
  test_loss_list.append(test_loss)
  train_loss_list.append(train_loss)
  test_acc_list.append(test_acc)
  train_acc_list.append(train_acc)
    
  if np.mod(k,100) == 0:
    print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))
    #print('iteration {} test loss: {:.3f}'.format(k+1,test_loss))
    print('iteration {} train accuracy: {:.3f}'.format(k+1,train_acc))
    #print('iteration {} train loss: {:.3f}'.format(k+1,train_loss))


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




