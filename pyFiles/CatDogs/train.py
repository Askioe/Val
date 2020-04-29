import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time



X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = np.array(X/255.0)
y = np.array(y)

dense_layers = [0, 1, 3]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]



for dense_layer in dense_layers:
	for layer_size in layer_sizes: 
		for conv_layer in conv_layers:
			NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{time.time}'
			log_dir = "logs/" + NAME
			tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

			model = Sequential()

			model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
			model.add(Activation("relu"))				#layer 1 
			model.add(MaxPooling2D(pool_size=(2,2)))

			for  l in range(conv_layer-1):
				model.add(Conv2D(layer_size, (3,3)))
				model.add(Activation("relu"))	# layer 2 
				model.add(MaxPooling2D(pool_size=(2,2)))


			model.add(Flatten())
			for l in range(dense_layer):
				model.add(Dense(layer_size))	
				model.add(Activation('relu'))

			model.add(Dense(1))
			model.add(Activation('sigmoid'))	#layer 4 output
			model.compile(loss="binary_crossentropy",
				optimizer="adam",
				metrics=['accuracy'])


			model.fit(X, y, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard_callback])

model.save("64x3-CNN.model")
