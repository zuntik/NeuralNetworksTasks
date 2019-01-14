#!/usr/bin/env python3

# Classify delayed XOR in a {0,1} string, in input strings of VARIABLE length

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from grammar import *

# ----------------------------------------------------------------------
# parameters

# dimetion of the one hot letters is 7 
y_dim = 7

# sequence_length is no longer necessary
#sequence_length = 20
num_train, num_valid, num_test = 5000, 500, 500

# our number of hidden units is 14 not 20
num_hidden = 20

# the batch size for the stochastic gradient descent is not defined, for now it
# can be 1% of the total, 50, which will allow 100 changes to the weights per
# epoch
batch_size = 50
learning_rate = 0.1
# max epoch is necessary because the error may never be small enough on the
# validadtion samples
max_epoch = 300

# ----------------------------------------------------------------------

# Data generation


words = [ make_embedded_reber() for i in range(num_train + num_valid + num_test) ]
max_len = len(max(words,key=len))
sl, words_one_hot, next_chars_one_hot = zip(*[ \
        (len(i) ,\
        np.pad(str_to_vec(i),((0,max_len-len(i)),(0,0)),mode='constant'), \
        np.pad(str_to_next_embed(i),((0,max_len-len(i)),(0,0)),mode='constant')) \
        for i in words ])

X_train = np.array( words_one_hot[0:num_train] )
y_train = np.array( next_chars_one_hot[0:num_train] )
sl_train = sl[0:num_train]
X_valid = np.array( words_one_hot[num_train:num_train+num_valid] )
y_valid = np.array( next_chars_one_hot[num_train:num_train+num_valid] )
sl_valid = sl[num_train: num_train+num_valid]
X_test = np.array( words_one_hot[-num_test:] )
y_test = np.array( next_chars_one_hot[-num_test:] )
sl_test = sl[-num_test:]



# placeholder for the sequence length of the examples
seq_length = tf.placeholder(tf.int32, [None])

# input tensor shape: number of examples, input length, dimensionality of each input
# at every time step, one bit is shown to the network
# we one hotted the data, so the output has 7
X = tf.placeholder(tf.float32, [None, max_len, 7])

# output tensor shape: number of examples, dimensionality of each output
# Binary output at end of sequence
# in our exercise we have a "many to many" scenario and each output 
# is a vector
y = tf.placeholder(tf.float32, [None, max_len, 7])

cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
# Cells are one fully connected recurrent layer with num_hidden neurons
# Activation function can be defined as second argument.
# Standard activation function is tanh for BasicRNN and GRU

# we want to a trainable output layer 
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, y_dim)

# only use outputs, ignore states
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_length) # NEW
# tf.nn.dynamic_rnn(cell, inputs, ...)
# Creates a recurrent neural network specified by RNNCell cell.
# Performs fully dynamic unrolling of inputs.
# Returns:
# outputs: The RNN output Tensor shaped: [batch_size, max_time, cell.output_size].


# define loss, minimizer and error
# the cross_entropy will have to be redefined to include error
#  for all of the time steps
#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y)
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=y))



# we still want to reduce cross_entropy which by now has been redefined
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


mistakes = tf.not_equal(y, tf.maximum(tf.sign(outputs), 0))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# split data into batches

num_batches = int(X_train.shape[0] / batch_size)
X_train_batches = np.array_split(X_train, num_batches)
y_train_batches = np.array_split(y_train, num_batches)
sl_train_batches = np.array_split(sl_train, num_batches)

# train

error_train_ = []
error_valid_ = []

for n in range(max_epoch):
    if n%100 == 0:
        print('training epoch {0:d}'.format(n+1))

    for X_train_cur, y_train_cur, sl_train_cur in zip(X_train_batches, y_train_batches, sl_train_batches):
        sess.run(train_step, feed_dict={X: X_train_cur, y: y_train_cur, seq_length: sl_train_cur})
        # We also need to feed the current sequence length
    error_train = sess.run(error, {X: X_train, y: y_train, seq_length: sl_train})
    error_valid = sess.run(error, {X: X_valid, y: y_valid, seq_length: sl_valid})

    if n%100 == 0:
        print('  train:{0:.3g}, valid:{1:.3g}'.format(error_train, error_valid))

    error_train_ += [error_train]
    error_valid_ += [error_valid]

    if error_train == 0:
        break

error_test = sess.run(error, {X: X_test, y: y_test, seq_length: sl_test})
error_valid = sess.run(error, {X: X_test, y: y_test, seq_length: sl_test})
error_train = sess.run(error, {X: X_test, y: y_test, seq_length: sl_test})
print('-'*70)
print('train error after epoch {0:d}: {1:.3f}'.format(n+1, error_train))
print('validation error after epoch {0:d}: {1:.3f}'.format(n+1, error_valid))
print('test error after epoch {0:d}: {1:.3f}'.format(n+1, error_test))

sess.close()

plt.figure()

plt.plot(np.arange(n+1), error_train_, label='training error')
plt.plot(np.arange(n+1), error_valid_, label='validation error')
plt.axhline(y=error_test, c='C2', linestyle='--', label='test error')
plt.xlabel('epoch')
plt.xlim(0, n)
plt.legend(loc='best')
plt.tight_layout()

plt.show()

