#importing necessary libraries

import tensorflow as tf
import tensorflow.keras as keras

#downloading the dataset
mnist = tf.keras.datasets.mnist

#dividing the dataset into training and test data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

#This line shoes the first image in the training dataset
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()

#This line shows the correspoding label for the first image printed above
print(y_train[0])

#reshaping the dataset
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#Checking the shape of the training data
x_train.shape

#initialising the CNN model
classifier = tf.keras.models.Sequential()

#Adding the Convulution, MaxPooling and Flattening Layers to the CNN model
classifier.add(tf.keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(tf.keras.layers.Flatten())

#Addign 3 dense or hidden layers
classifier.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
classifier.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
classifier.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Adding the output layer
classifier.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Compiling out Fitting the classification model to the dataset
classifier.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, [y_train], epochs = 10)

#Determing the Accuracy score 
val_loss, val_acc = classifier.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
