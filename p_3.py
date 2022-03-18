import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

import pickle
import time

#NAME = "Cats-vs-Dogs-CNN-64x2-{}".format(int(time.time()))


pickle_in = open("X.pickle","r+b")
x = pickle.load(pickle_in)

pickle_in = open("y.pickle","r+b")
y = pickle.load(pickle_in)

# print("-------------------------")
# print(np.array(x).shape)

x=np.array(x/255.0)
y=np.array(y)


dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
            print(NAME)
            tensorBoard = TensorBoard(log_dir='logs/{}'.format(NAME))

            #-------model begin--------
            model = Sequential()
            
            model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

            for l in range(dense_layer):
                model.add(Dense(512))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            model.fit(x, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorBoard])
 
model.save('64x3-CNN.model')





