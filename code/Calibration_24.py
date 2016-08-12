# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:16:08 2016

@author: sunbo
"""
from __future__ import print_function
import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 45
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 24, 24
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

def train_24calibration_net(X_train, y_train):
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    X_train = X_train.astype('float32')
    X_train /= 255
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(3, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_split=0.1)
    json_string = model.to_json()
    open('../model/24calibration_architecture.json', 'w').write(json_string)
    model.save_weights('../model/24calibration_weights.h5')
def get_24calibration():
    model = model_from_json(open('../model/24calibration_architecture.json').read())
    model.load_weights('../model/24calibration_weights.h5')
    return model