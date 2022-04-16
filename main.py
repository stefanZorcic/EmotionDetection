

from tensorflow.keras import datasets, layers, models
import numpy as np
import datetime

import keras_tuner as kt
from keras_tuner import RandomSearch
from tensorflow import keras

import matplotlib.image as img
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import os
import cv2


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = []

#Importing data

emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]

temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for i in range(len(emotions)):
    files = [f for f in listdir("archive/{}".format(emotions[i])) if isfile(join("archive/{}".format(emotions[i]), f))]
    for x in files:
        temp[i] = 1.0
        image1=img.imread("archive/{}/{}".format(emotions[i], x))
        image = cv2.resize(image1,(50,50))
        #image = image/255
        data.append([image, np.array(temp,dtype="float32")])
        temp[i] = 0.0


#Spliting test and training data

from sklearn.model_selection import train_test_split

training_data, testing_data = train_test_split(data, test_size=0.1, random_state=25)

test_images = []
train_images = []
test_labels = []
train_labels = []

for x in training_data:
    train_images.append(np.array(x[0],dtype="float32"))
    train_labels.append(np.array(x[1],dtype="float32"))

for x in testing_data:
    test_images.append(np.array(x[0]))
    test_labels.append(np.array(x[1]))

train_images = np.array(train_images,dtype="float32").reshape(-1, 50, 50, 1)
train_labels = np.array(train_labels,dtype="float32")
test_images = np.array(test_images,dtype="float32").reshape(-1, 50, 50, 1)
test_labels = np.array(test_labels,dtype="float32")
data=[]
training_data=[]
testing_data=[]


print(train_images.shape,train_labels.shape)


#Creating the model

nums=[1,2,3,4,5]
denses=[1,2,3]
sizes=[16,32,64,128]

'''
for num in nums:
    for size in sizes:
        for dense in denses:

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"+"con"+str(num)+"size"+str(size)+"dense"+str(dense))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            model = models.Sequential()
            model.add(layers.Conv2D(size, (3, 3), activation='softmax', input_shape=(50, 50, 1)))
            model.add(layers.MaxPooling2D((2, 2)))

            for i in range(1,num):
                model.add(layers.Conv2D(size, (3, 3), activation='softmax'))
                model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Flatten())
            for i in range(dense):
                model.add(layers.Dense(size, activation='softmax'))
            model.add(layers.Dense(8, activation='softmax'))

            model.summary()

            #Testing model

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            print(len(test_images),len(test_labels))
            print(len(train_images),len(train_labels))

            #history = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))
            print(train_labels.shape)
            model.fit(train_images,train_labels,batch_size=64, epochs=50, validation_split=0.1, callbacks=[tensorboard_callback])

'''
'''
con=3
pool=2



with tf.device('/GPU:0'):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"pool{}con{}64".format(pool, con)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = models.Sequential()
    model.add(layers.Conv2D(64, (con, con), activation='softmax', input_shape=(50, 50, 1)))
    model.add(layers.MaxPooling2D((pool, pool)))

    model.add(layers.Conv2D(64, (con, con), activation='softmax'))
    model.add(layers.MaxPooling2D((pool, pool)))

    model.add(layers.Conv2D(32, (con, con), activation='softmax'))
    model.add(layers.MaxPooling2D((pool, pool)))


    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Dense(8, activation='softmax'))

    model.summary()

    #Testing model

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(len(test_images),len(test_labels))
    print(len(train_images),len(train_labels))

    #history = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))
    print(train_labels.shape)
    model.fit(train_images,train_labels,batch_size=64, epochs=100, validation_split=0.1, callbacks=[tensorboard_callback])
'''

con=3
pool=2


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"50x50con32-16-16-5-2dense32".format(pool, con)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = models.Sequential()
model.add(layers.Conv2D(16, (con, con), activation='relu', input_shape=(50, 50, 1)))
model.add(layers.MaxPooling2D((pool, pool)))

model.add(layers.Conv2D(16, (con, con), activation='relu'))
model.add(layers.MaxPooling2D((pool, pool)))

model.add(layers.Conv2D(16, (con, con), activation='relu'))
model.add(layers.MaxPooling2D((pool, pool)))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#print(len(test_images),len(test_labels))
#print(len(train_images),len(train_labels))

#history = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))
#print(train_labels.shape)
model.fit(train_images,train_labels,batch_size=200, epochs=55, validation_split=0.1, callbacks=[tensorboard_callback])
model.save("finalModel")


'''
Best:
batch=200
16 3,3 2,2 relu con
16 3,3 2,2 relu con
16 3,3 2,2 relu con
16 relu dense
8 relu dense
'''