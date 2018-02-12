#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 08:11:09 2018
Keras Base
@author: jakub
"""

from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from dataset_coco import Dataset
from theanoFunctions import softmax2, logSoftmax2

numClasses = 29
networkScale = 8

imageSize = 256
targetSize = 14
batchSize = 24

convFilters0 = 32
convFilters1 = 64
convFilters2 = 128
convFilters3 = 256

inputShape = (1, imageSize, imageSize)

model = Sequential()
model.add(Conv2D(filters=convFilters0, kernel_size=(5,5),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters0, kernel_size=(5,5),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilters1, kernel_size=(3,3),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters1, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilters2, kernel_size=(3,3),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters2, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilters3, kernel_size=(3,3),
                 activation='relu', input_shape=inputShape))
model.add(Conv2D(filters=convFilters3, kernel_size=(3,3),
                 activation='relu'))

lastConv = Conv2D(filters=2*numClasses, kernel_size=(1,1),
                 activation='linear')
model.add()
model.add(Reshape(()))