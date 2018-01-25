#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:57:46 2018
Test dataset
@author: jakub
"""
import cv2
import numpy as np

def testDataset(dataset, classes=[0]):
    catNms = dataset.sCatNms + dataset.kpCatNms
    def flattenImage(inputs, targets, masks, cl, bId):
        scale=8
        inImage = inputs[bId,0]
        imSize = inImage.shape[0]
        maImage = masks[bId, cl]
        taImage = targets[bId, cl, 0]
        taSize = taImage.shape[0]
        margin = (imSize - (taSize * scale)) / 2
        #if np.max(taImage) > 1.0:
        #    taImage /= np.max(taImage)
        #if np.max(maImage) > 1.0: 
        #    maImage /= np.max(maImage)
        maImage = cv2.resize(maImage, None, fx=scale, fy=scale)
        taImage = cv2.resize(taImage, None, fx=scale, fy=scale)
        maImageCanvas = np.zeros((imSize, imSize), dtype=np.float32)
        taImageCanvas = np.zeros((imSize, imSize), dtype=np.float32)
        taImageCanvas[margin:-margin, margin:-margin] = taImage
        maImageCanvas[margin:-margin, margin:-margin] = maImage
        image1 = 0.5 * inImage + 0.5 * taImageCanvas
        image2 = 0.5 * inImage + 0.5 * maImageCanvas
        return image1, image2
    def showImage(im1, im2, cl, bId):
        winTarName = 'targets_{}'.format(catNms[cl])
        winMasName = 'masks_{}'.format(catNms[cl])
        cv2.namedWindow(winTarName, cv2.WINDOW_NORMAL)
        cv2.namedWindow(winMasName, cv2.WINDOW_NORMAL)
        cv2.imshow(winTarName, im1)
        cv2.imshow(winMasName, im2)
        cv2.moveWindow(winMasName, 1000, 1000)
        cv2.waitKey(0)
        cv2.destroyWindow(winTarName)
        cv2.destroyWindow(winMasName)

    for cl in classes:
        for inputs, targets, masks in dataset.iterateMinibatches(val=False):
            for bId in range(inputs.shape[0]):
                if np.max(targets[bId, cl,0]) > 0.0:
                    im1, im2 = flattenImage(inputs, targets, masks, cl, bId)
                    showImage(im1, im2, cl, bId)