#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:51:43 2018
Neural network base class
@author: jakub
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import cv2
import time
from theanoFunctions import categoricalCrossentropyLogdomain2
from logger import Logger


class nnBase(object):
    def buildNN(self, modelFile, inputVar):
        raise NotImplementedError
        
    @staticmethod
    def getNetworkScale():
        raise NotImplementedError
        
    @staticmethod
    def getReceptiveFieldWidth(self):
        raise NotImplementedError
        
    @staticmethod
    def getReceptiveFieldHeight(self):
        raise NotImplementedError
        
    @staticmethod
    def getNumClasses(self):
        raise NotImplementedError
        
    def getNetworkName(self):
        raise NotImplementedError
        
    def __init__(self, modelWeights=None, train=True):
        self.inputVar = T.tensor4('input')
        self.network = self.buildNN(modelWeights, self.inputVar, train=train)
    def compileTrainFunctions(self, learningRate=0.0001):
        print("Compiling functions...")
        # Prepare Theano variables for inputs and targets
        inputVar = self.inputVar
        tensor5 = T.TensorType('float32', (False,)*5)
        targetVar = tensor5('targets')
        weightVar = T.tensor4('weights')
        
        model = self.network
        prediction = L.get_output(model)
        loss = categoricalCrossentropyLogdomain2(prediction, targetVar)
        loss = loss * weightVar
        loss = loss.mean()
        params = L.get_all_params(model, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=learningRate)
        
        valPrediction =L.get_output(model, deterministic=True)
        valLoss = categoricalCrossentropyLogdomain2(valPrediction, targetVar)
        valLoss = valLoss * weightVar
        valLoss = valLoss.mean()
        valAcc = T.sum(T.eq(T.argmax(valPrediction, axis=2), \
                            T.argmax(targetVar, axis=2))*weightVar, 
                      dtype=theano.config.floatX)
        binTar = targetVar > 0.5
        binTar = binTar[:,:,0]
        binPred = T.exp(valPrediction) > 0.5
        binPred = binPred[:,:,0]
        TPs = binTar * binPred
        TPs = T.cast(TPs.sum(), 'float32')
        FNs = binTar * (1 - binPred)
        FNs = T.cast(FNs.sum(), 'float32')
        FPs = (1 - binTar) * binPred
        FPs = T.cast(FPs.sum(), 'float32')
        pre = T.switch(T.gt(TPs + FNs, 0.0), TPs / (TPs + FNs), 0.0)
        rec = T.switch(T.gt(TPs + FPs, 0.0), TPs / (TPs + FPs), 0.0)
        Fsc = T.switch(T.and_(T.gt(pre, 0.0), T.gt(rec, 0.0)),
                       2 * (pre * rec) / (pre + rec),0.0)
        
        valAcc /= T.sum(weightVar)  
        
        trainFn = theano.function([inputVar, targetVar, weightVar], loss, 
                                  updates=updates, allow_input_downcast=True)
        valFn = theano.function([inputVar, targetVar, weightVar], 
                                 [valLoss, valAcc, pre, rec, Fsc], 
                                 allow_input_downcast=True)
        self.trainFn = trainFn
        self.valFn = valFn
        
    def compileTestFunctions(self):
        print("Compiling functions...")
        inputVar = self.inputVar
        model = self.network
        prediction = L.get_output(model, deterministic=True)
        forwardFn = theano.function([inputVar], prediction, 
                                    allow_input_downcast=True)
        self.forwardFn = forwardFn
    def train(self, dataset, numEpochs=100):
        if not hasattr(self, 'trainFn'):
            self.compileTrainFunctions()
        logger = Logger(self.networkName)
        logger.paramsToTrack(['trainLoss', 'valLoss', 'valAcc'])
        #best val
        bestVal = 100.0
        trainFn = self.trainFn
        testFn = self.valFn
        # Launch the training loop:
        print 'Starting training...'
        logger.logStart()
        try:
            for epoch in range(numEpochs):
                # In each epoch, do a full pass over the training data:
                trainErr = 0
                trainBatches = 0
                startTime = time.time()
                
                for batch in dataset.iterateMinibatches():
                    inputs, targets, weights = batch
                    trainErr += trainFn(inputs, targets, weights)
                    trainBatches += 1
                
                # And a full pass over the validation data:
                valErr = 0
                valBatches = 0
                valAcc = 0
                valRec = 0.0
                valPre = 0.0
                valFsc = 0.0
                for batch in dataset.iterateMinibatches(True):
                    inputs, targets, weights = batch
                    err, acc, pre, rec, Fsc = testFn(inputs, targets, weights)
                    valErr += err
                    valAcc += acc
                    valRec += rec
                    valPre += pre
                    valFsc = Fsc
                    valBatches += 1
                logger.processEpoch(trainErr / trainBatches, valErr / valBatches, 
                                valAcc / valBatches)
                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(epoch + 1, numEpochs, time.time() - startTime))
                print("  training loss:\t\t{:.6f}".format(trainErr / trainBatches))
                print("  validation loss:\t\t{:.6f}".format(valErr / valBatches))
                print("  validation accuracy:\t\t{:.2f} %".format(valAcc / valBatches * 100))
                print("  validation precissi:\t\t{:.2f} %".format(valPre / valBatches * 100))
                print("  validation recall  :\t\t{:.2f} %".format(valRec / valBatches * 100))
                print("  validation F1Score :\t\t{:.2f} %".format(valFsc / valBatches * 100))
                
                if ((valErr / valBatches) < bestVal) :
                    bestVal = valErr / valBatches
                    # save network            
                    np.savez('models/model_{}.npz'.format(self.getNetworkName()), lasagne.layers.get_all_param_values(self.network))
                    print('Model saved, lowest val {:.6f}'.format(bestVal))
        except KeyboardInterrupt:
            print 'Training stopped'
            dataset.endDataset()
            logger.logEnd(success=False)
            failed = True
            
        if not failed:
            dataset.endDataset()
            logger.logEnd()
            
                
        
        