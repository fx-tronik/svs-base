#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:49:05 2018
Test script
@author: jakub
"""
import numpy as np
import cv2
def imageFloat2Int(detMap):
    return (255 * detMap).astype(np.uint8) 

def blobDetector(clDet):
    clDet = imageFloat2Int(clDet)
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.blobColor = 255
    params.minDistBetweenBlobs = 1
    detector = cv2.SimpleBlobDetector_create(params)
    
    kps = detector.detect(clDet)
    kpsImg = np.zeros_like(clDet)
    for kp in kps:
        kpsImg = cv2.circle(kpsImg, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/10), 128)
    def boxesFromMap(detMap, thresholds, catNms):
    #transforms detection maps into bonding boxes
    #detMap in form (batch, numClasses, 2, targetSize, targetSize)
    #BBs in form (tl, br, classId, probability)
    detMap = detectionThr(detMap, thresholds, catNms)
    bSize = detMap.shape[0]
    sh = detMap.shape
    for bId in range(bSize):
        for catId in range(len(catNms)):
            clDet = detMap[bId, catId]
def detectionThr(detMap, thresholds, catNms):
    bSize = detMap.shape[0]
    for  bId in range(bSize):
        for catId, catNm in enumerate(catNms):
            clDet = detMap[bId, catId, 0]
            clDet[clDet > thrsholds[catNm]] = 1.0
            clDet[clDet < thrsholds[catNm]] = 0.0
    return detMap[:,:,0]

def detectionScale(detMap, scale = 8, imSize = 256):
    #return only positive part!
    bSize = detMap.shape[0]
    sh = detMap.shape
    tarSize = sh[-1]
    if len(sh) == 5:
        detMap = detMap[:,:,0]
        sh = detMap.shape
    scaledDetMap = np.zeros((sh[0], sh[1], imSize, imSize), dtype=np.float32)
    margin = int(0.5 * (imSize - tarSize * scale))
    for bId in range(bSize):
        for catId in range(len(catNms)):
            clDet = detMap[bId, catId]
            clDetCanvas = np.zeros((imSize, imSize), dtype=np.float32)
            clDetCanvas[margin:margin+scale * tarSize, margin:margin+scale * tarSize]\
                            =cv2.resize(clDet, None, fx = scale, fy = scale) 
            scaledDetMap[bId, catId] = clDetCanvas
    return scaledDetMap


if __name__ == "__main__":
    from dataset_coco import Dataset
    dataDir = '/home/jakub/data/fxi/coco'
    dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                       batchSize=8)
    catNms = dataset.sCatNms + dataset.kpCatNms
    thresholds = {catNm:0.5 for catNm in catNms}
    #put special thresholds here:
    thresholds['person'] = 0.75
    detMap = targets
    boxes = boxesFromMap(detMap, thrsholds, catNms)
    

