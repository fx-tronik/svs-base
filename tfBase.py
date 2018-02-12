#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:01:48 2018
Tensorflow base
@author: jakub
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

numClasses = 29
networkScale = 8

imageSize = 256
targetSize = 24
batchSize = 3

convFilters0 = 32
convFilters1 = 64
convFilters2 = 128
convFilters3 = 256

numEpochs = 10

def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    inputLayer = tf.transpose(x_dict['images'], [0,2,3,1])
    
    conv11 = tf.layers.conv2d(
            inputs=inputLayer,
            filters=convFilters0,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(
            inputs=conv11,
            filters=convFilters0,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2])
    
    conv21 = tf.layers.conv2d(
            inputs=pool1,
            filters=convFilters1,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
    conv22 = tf.layers.conv2d(
            inputs=conv21,
            filters=convFilters1,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2])
    
    conv31 = tf.layers.conv2d(
            inputs=pool2,
            filters=convFilters2,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
    conv32 = tf.layers.conv2d(
            inputs=conv31,
            filters=convFilters2,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2, 2])
    
    conv41 = tf.layers.conv2d(
            inputs=pool3,
            filters=convFilters3,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
    conv42 = tf.layers.conv2d(
            inputs=conv41,
            filters=convFilters3,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
    #linear output?!
    conv5 = tf.layers.conv2d(
            inputs=conv42,
            filters= 2 * numClasses,
            kernel_size=[1, 1],
            padding="valid",
            activation=tf.nn.)



if __name__ == "__main__":
  tf.app.run()

