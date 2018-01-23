#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:04:12 2018
utils function
@author: jakub
"""
import theano.tensor as T

def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def softmax2(x):
    e_x = T.exp(x - x.max(axis=2, keepdims=True))
    return e_x / e_x.sum(axis=2, keepdims=True)

def logSoftmax(x):
    xdev = x - x.max(axis=1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def logSoftmax2(x):
    xdev = x - x.max(axis=2, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=2, keepdims=True))

def categoricalCrossentropyLogdomain(logPredictions, targets):
    return -T.sum(targets * logPredictions, axis=1)

def categoricalCrossentropyLogdomain2(logPredictions, targets):
    return -T.sum(targets * logPredictions, axis=2)
