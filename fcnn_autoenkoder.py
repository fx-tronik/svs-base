#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:48:33 2018
Convolutional neural network class
@author: jakub
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
from logger import Logger
import time

from theanoFunctions import softmax2, logSoftmax2, categoricalCrossentropyLogdomain2
from nn_base import nnBase

class fcnn(nnBase):
    networkName = 'fc_autoenkoder_v00'
    
    numClasses = 29
    networkScale = 8

    imageSize = 256
    targetSize = 24
    batchSize = 24
    
    convFilters0 = 32
    convFilters1 = 64
    convFilters2 = 128
    convFilters3 = 256
    
    def __init__(self, modelWeights=None, train=True):
        self.inputVar = T.tensor4('input')
        self.network, self.decoder = self.buildNN(modelWeights, self.inputVar, train=train)
        
    def buildNN(self, modelFile, inputVar, train=True):
        print 'Model building'
        pad = 0 if train else 'same'
        nonlin = lasagne.nonlinearities.rectify
        bn = L.batch_norm
        network = L.InputLayer(shape=(None, 1, None, None),
                input_var=inputVar)
        # Convolutional layers #0
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters0, 
                                filter_size=(5, 5),nonlinearity=nonlin, 
                                pad=pad))
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters0, 
                                filter_size=(5, 5),nonlinearity=nonlin, 
                                pad=pad))

        # Max-pooling layer
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layers #1
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters1, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad))
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters1, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad))

        # Max-pooling layer
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layers #2
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters2, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad))
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters2, 
                        filter_size=(3, 3),nonlinearity=nonlin, 
                        pad=pad))

        # Max-pooling layer
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        # Convolutional layers #3
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters3, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad))
        network = bn(L.Conv2DLayer(network, num_filters=self.convFilters3, 
                                filter_size=(3, 3),nonlinearity=nonlin, 
                                pad=pad))
        #Dekoder branch
        sp1 = L.TransposedConv2DLayer(network, num_filters=128, filter_size=(5, 5), stride=2)
        sp2 = L.TransposedConv2DLayer(sp1, num_filters=32, filter_size=(7, 7), stride=2)
        sp3 = L.TransposedConv2DLayer(sp2, num_filters=8, filter_size=(13, 13), stride=2)
        sp4 = L.TransposedConv2DLayer(sp3, num_filters=1, filter_size=(32, 32))

        # Final convolutional layer
        lastNonlin = logSoftmax2 if train else softmax2
        network = L.Conv2DLayer(network, num_filters=2*self.numClasses, 
                                filter_size=(1, 1), nonlinearity=lastNonlin)
        network = L.ReshapeLayer(network, ([0], self.numClasses, 2, [2], [3]))
        network = L.NonlinearityLayer(network, lastNonlin)
        if modelFile:
            modelWeights = np.load(modelFile)
            modelWeights = modelWeights[modelWeights.keys()[0]]
            L.set_all_param_values(network, modelWeights)
            print 'ZAŁADOWANO WAGI'

        return network, sp4
    @staticmethod
    def getNumClasses(self):
        return self.numClasses
    
    @staticmethod
    def getNetworkScale(self):
        return self.networkScale 
    
    def getNetworkName(self):
        return self.networkName
    
    
    def compileTrainFunctions(self, learningRate=0.0001):
        print("Compiling functions...")
        # Prepare Theano variables for inputs and targets
        inputVar = self.inputVar
        tensor5 = T.TensorType('float32', (False,)*5)
        targetVar = tensor5('targets')
        weightVar = T.tensor4('weights')
        
        model = self.network
        decoder = self.decoder
        prediction = L.get_output(model)
        decoded = L.get_output(decoder)
        decoderLoss = lasagne.objectives.squared_error(decoded, inputVar).mean()
        loss = categoricalCrossentropyLogdomain2(prediction, targetVar)
        loss = loss * weightVar
        loss = loss.mean()
        loss = loss + decoderLoss
    
        params = L.get_all_params(model, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=learningRate)
        
        valPrediction =L.get_output(model, deterministic=True)
        valLoss = categoricalCrossentropyLogdomain2(valPrediction, targetVar)
        valLoss = valLoss * weightVar
        valLoss = valLoss.mean()
        valAcc = T.sum(T.eq(T.argmax(valPrediction, axis=2), \
                            T.argmax(targetVar, axis=2))*weightVar, 
                      dtype=theano.config.floatX)
        valAcc /= T.sum(weightVar)  
        
        trainFn = theano.function([inputVar, targetVar, weightVar], loss, 
                                  updates=updates, allow_input_downcast=True)
        valFn = theano.function([inputVar, targetVar, weightVar], 
                                 [valLoss, valAcc], allow_input_downcast=True)
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
                for batch in dataset.iterateMinibatches(True):
                    inputs, targets, weights = batch
                    err, acc = testFn(inputs, targets, weights)
                    valErr += err
                    valAcc += acc
                    valBatches += 1
                logger.processEpoch(trainErr / trainBatches, valErr / valBatches, 
                                valAcc / valBatches)
                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(epoch + 1, numEpochs, time.time() - startTime))
                print("  training loss:\t\t{:.6f}".format(trainErr / trainBatches))
                print("  validation loss:\t\t{:.6f}".format(valErr / valBatches))
                print("  validation accuracy:\t\t{:.4f} %".format(valAcc / valBatches * 100))
                
                if ((valErr / valBatches) < bestVal) :
                    best_val = valErr / valBatches
                    # save network            
                    np.savez('models/model_{}.npz'.format(self.getNetworkName()), lasagne.layers.get_all_param_values(self.network))
                print("lowest val %08f"%(best_val))
        except KeyboardInterrupt:
            print 'Training stopped'
            dataset.endDataset()
            logger.logEnd(success=False)
            failed = True
            
        if not failed:
            dataset.endDataset()
            logger.logEnd()
       
if __name__ == "__main__":
    from dataset_coco import Dataset
    dataDir = '/home/jakub/data/fxi/coco'
    #dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
    #                   batchSize=8)
    net = fcnn(train = True)
    #net.train(dataset)
    
    #dataset.endDataset()
    
    sp = L.InputLayer((8,256, 24, 24)) 
    #skok wymiaru *1.5 (1.5 **2 = 2,25), wymiar zmienia się o filter-1
    #redukcja wymiaru filtra 2, zwiększenie wymiaru obrazu 2.25 (mały niedobór danych)
    sp1 = L.TransposedConv2DLayer(sp, num_filters=128, filter_size=(13, 13))
    sp2 = L.TransposedConv2DLayer(sp1, num_filters=64, filter_size=(19, 19))
    sp3 = L.TransposedConv2DLayer(sp2, num_filters=32, filter_size=(28, 28))
    sp4 = L.TransposedConv2DLayer(sp3, num_filters=16, filter_size=(41, 41))
    sp5 = L.TransposedConv2DLayer(sp4, num_filters=8, filter_size=(61, 61))
    sp6 = L.TransposedConv2DLayer(sp5, num_filters=1, filter_size=(76, 76))
    print sp1.output_shape
    print sp2.output_shape
    print sp3.output_shape
    print sp4.output_shape
    print sp5.output_shape    
    print sp6.output_shape  
    
    sp1 = L.TransposedConv2DLayer(sp, num_filters=128, filter_size=(5, 5), stride=2)
    sp2 = L.TransposedConv2DLayer(sp1, num_filters=32, filter_size=(7, 7), stride=2)
    sp3 = L.TransposedConv2DLayer(sp2, num_filters=8, filter_size=(13, 13), stride=2)
    sp4 = L.TransposedConv2DLayer(sp3, num_filters=1, filter_size=(32, 32))

    print sp1.output_shape
    print sp2.output_shape
    print sp3.output_shape
    print sp4.output_shape
