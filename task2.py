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


# Initial weight first layer. Connect 7 neurons (first layer) to 
# la matriz de pesos en el nxm siendo "n" el numero de datos(o neuronas) de entrada y "m" el de neuronas de salida
# el bias tiene que ser del tamaño de la salida ( es decir "m")
# al principio tendriamos (300, 7) 300 datos de entrada para 7 neuronas de salida
W = tf.Variable(rd.randn(300,26),trainable=True)
b = tf.Variable(np.zeros(26),trainable=True)

# El input de la primera capa, tenemos 300 col, es decir cada "caracter" tiene 300 dimensiones o caracteristicas
# La dimenson de las filas se inicializa en None pero si queremos recibir mas de un caracter introduciriamos ese numero en vez de None
x = tf.placeholder(shape=(None,300),dtype=tf.float64)
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Placeholder
y_ = tf.placeholder(shape=(None,26),dtype=tf.float64)
# Este es el error medio por caracter
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# We create a variable for later initialize all variables
init = tf.initialize_all_variables()

# Session is responsible of building the inital graph
# We create the graph  
sess = tf.Session()

# Re init variables to start from scratch
sess.run(init)

# Create lists for monitoring
test_error_list = []
train_error_list = []

for k in range(20):
    # Compute a gradient step
    sess.run(train_step, feed_dict={x:isolate_data, y_:isolate_data_class_binary} )
    
    # Compute the losses on training and testing sets for monitoring
    train_err = sess.run(cross_entropy, feed_dict={x:isolate_data, y_:isolate_data_class_binary})
    test_err = sess.run(cross_entropy, feed_dict={x:isolate_test, y_:isolate_test_class_binary})
    test_error_list.append(test_err)
    train_error_list.append(train_err)



fig,ax = plt.subplots(1)
ax.plot(train_error_list, color='blue', label='training', lw=2)
ax.plot(test_error_list, color='green', label='testing', lw=2)
ax.set_xlabel('Training epoch')
ax.set_ylabel('Cross-entropy')
plt.legend()
plt.show()




