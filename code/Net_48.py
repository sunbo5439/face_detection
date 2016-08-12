# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:16:08 2016

@author: sunbo
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Merge, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.layers.normalization import BatchNormalization

batch_size = 128
nb_classes = 2
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5


def train_48net(X_train_1,X_train_2, y_train):
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    X_train_1 = X_train_1.astype('float32')
    X_train_2 = X_train_2.astype('float32')
    X_train_1 /= 255
    X_train_2 /= 255
    model1 = Sequential()
    model1.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                             border_mode='valid',
                             input_shape=(3, 48,48)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model1.add(BatchNormalization(mode=2))
    model1.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model1.add(BatchNormalization(mode=2))
    model1.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model1.add(Activation('relu'))
    model1.add(Flatten())

    model2 = Sequential()
    model2.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                             border_mode='valid',
                             input_shape=(3, 24,24)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model2.add(Flatten())
    model2.add(Dense(128))
    model2.add(Activation('relu'))

    model = Sequential();
    model.add(Merge([model1, model2], mode='concat'))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit([X_train_1, X_train_2], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_split=0.1)
    json_string = model.to_json()
    open('../model/48net_architecture.json', 'w').write(json_string)
    model.save_weights('../model/48net_weights.h5')

def get_48net():
    model = model_from_json(open('../model/48net_architecture.json').read())
    model.load_weights('../model/48net_weights.h5')
    return model