import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import nn18_ex2_load

isolate_data, isolate_data_class, isolate_test, isolate_test_class = nn18_ex2_load.load_isolet()

# Recieves a placeholder x, and the number of inputs and outputs
# returns the W matrix, the bias and the sigmoid fuction
def layer(x, n_input, n_output):
  W = tf.Variable(rd.randn(n_input,n_output),trainable=True)
  b = tf.Variable(np.zeros(n_output),trainable=True)
  y = tf.nn.sigmoid(tf.matmul(x,W) + b)
  return (W, b, y)

# Is like the function layer but we return a softmax function instead of a sigmoid
def last_layer(x, n_input, n_output):
  W = tf.Variable(rd.randn(n_input,n_output),trainable=True)
  b = tf.Variable(np.zeros(n_output),trainable=True)
  z = tf.nn.softmax(tf.matmul(x,W) + b)
  return (W, b, z)

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
isolate_data_class_binary = np.eye(26)[isolate_data_class-1]
isolate_test_class_binary = np.eye(26)[isolate_test_class-1]
# Now we have 1559 binary vectors each one with a range from 1 to 26
print(isolate_data_class_binary)


#--Normalize data--
# normalize_data = (data - min(data)) / (max(data) - min(data))
isolate_data = ((isolate_data +1) / 2)
isolate_test = ((isolate_test +1) / 2)


n_input = 300
num_neuron = 100

x = tf.placeholder(shape=(None, 300),dtype=tf.float64)

#-- 2 LAYERS --
W_hid , b_hid, y = layer(x, 300, num_neuron)
W_out , b_out, z = last_layer(y, num_neuron, 26)

#-- 3 LAYERS --
#W_hid , b_hid, y = layer(x, 300, num_neuron)
#W_hid2 , b_hid2, y2 = layer(y, num_neuron, num_neuron)
#W_out , b_out, z = last_layer(y2, num_neuron, 26)

#-- 4 LAYERS --
#W_hid , b_hid, y = layer(x, 300, num_neuron)
#W_hid2 , b_hid2, y2 = layer(y, num_neuron, num_neuron)
#W_hid3 , b_hid3, y3 = layer(y2, num_neuron, num_neuron)
#W_out , b_out, z = last_layer(y3, num_neuron, 26)


z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
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

for k in range(500):
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

# checkpoont , save summary




