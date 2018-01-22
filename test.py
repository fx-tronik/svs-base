#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:49:05 2018
Test script
@author: jakub
"""
import numpy as np
import cv2
import datetime
import os

allCats = np.load('categories.npz')
allCats = list(allCats[allCats.keys()[0]])
supCats = allCats[:12]
kpCats = allCats[12:]
imId = 0
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
        
def contourDetector(clDet):
    clDet = imageFloat2Int(clDet)
    clDet2, contours, hierarchy = cv2.findContours(clDet,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def boxesFromMap(detMap, thresholds, catNms):
    #transforms detection maps into bonding boxes
    #detMap in form (batch, numClasses, 2, targetSize, targetSize)
    #BBs in form (tl, br, classId, probability)
    global allCats
    detMap = detectionThr(detMap, thresholds, catNms)
    bSize = detMap.shape[0]
    cntsPerClass = []
    for bId in range(bSize):
        for i, catNm in enumerate(catNms):
            catId = allCats.index(catNm)
            clDet = detMap[bId, catId]
            cnts = contourDetector(clDet)
            if cnts:
                cntsPerClass.append((bId, catId, cnts))
    boxes = []
    for bId, clId, cnts in cntsPerClass:
        for cnt in cnts:
            boxes.append((bId, clId, cv2.boundingRect(cnt)))
    return cntsPerClass, boxes
def detectionThr(detMap, thresholds, catNms):
    bSize = detMap.shape[0]
    for  bId in range(bSize):
        for catId, catNm in enumerate(catNms):
            clDet = detMap[bId, catId, 0]
            clDet[clDet > thresholds[catNm]] = 1.0
            clDet[clDet < thresholds[catNm]] = 0.0
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

def boxesScale(boxes, scale=8, tarSize=24, imSize=256):
    #scale rects and contours
    margin = int(0.5 * (imSize - tarSize * scale))
    scaledBoxes = []
    for bId, clId, box in boxes:
        scaledBox = np.array(box) * scale 
        scaledBox[:2] += margin
        scaledBoxes.append((bId, clId, tuple(scaledBox)))
    return scaledBoxes

def cntsScale(cntsPerClass, scale=8, tarSize=24, imSize=256):
    #scale rects and contours
    margin = int(0.5 * (imSize - tarSize * scale))
    scaledCntsPerClass = []
    for bId, clId, cnts in cntsPerClass:
        scaledCnts = []
        for cnt in cnts:
            scaledCnt = np.array(cnt) * scale + margin
            scaledCnts.append(scaledCnt)
        scaledCntsPerClass.append((bId, clId, scaledCnts))
    return scaledCntsPerClass
def filterCnts(cntsPerCat, bId, catNms):
    global allCats
    cntsToReturn = []
    for cntBId, cntCat, cnts in cntsPerCat:
        if cntBId == bId and cntCat in  [allCats.index(cat) for cat in catNms]:
            cntsToReturn.append(( cntBId, cntCat, cnts))
    return cntsToReturn
def drawCnts(image, cntsPerCat, scale=8, tarSize=24, imSize=256):
    cntsImage = np.zeros_like(image)
    cntsPerCat = cntsScale(cntsPerCat, scale, tarSize, imSize)
    for _, _, cnts in cntsPerCat:
        cntsImage = cv2.drawContours(cntsImage, cnts, -1, (0, 255, 0), 1)
    image[np.where(cntsImage[:,:,1] == 255)] = [0, 255, 0]
    return image
        
def drawBoxes(image, boxesPerCat, scale=8, tarSize=24, imSize=256):
    boxesImage = np.zeros_like(image)
    boxesPerCat = boxesScale(boxesPerCat, scale, tarSize, imSize)
    for _,_, box in boxesPerCat:
        x,y,w,h = box
        boxesImage = cv2.rectangle(boxesImage,(x,y),(x+w,y+h),(255,255,0),1)
    image[np.where(boxesImage[:,:,1] == 255)] = [255, 255, 0]
    return image
def saveResults(inputs, detMap, thresholds, scale = 8, 
                drawBoxesCats=['person'], drawCntsCats=['nose'], 
                outputDir='/home/jakub/data/results'):
    global imId
    imSize = inputs.shape[-1]
    tarSize = detMap.shape[-1]
    bSize = inputs.shape[0]
    assert bSize == detMap.shape[0]

    catsToShow = drawBoxesCats + drawCntsCats
    cntsPerCat, boxesPerCat = boxesFromMap(detMap,thresholds, catsToShow)
    for bId in range(bSize):
        image = imageFloat2Int(inputs[bId, 0])
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cntsPerIm = filterCnts(cntsPerCat, bId, catNms = drawCntsCats)
        boxesPerIm = filterCnts(boxesPerCat, bId, catNms = drawBoxesCats)
        image = drawCnts(image, cntsPerIm, scale=scale, imSize=imSize, tarSize=tarSize)
        image = drawBoxes(image, boxesPerIm, scale=scale, imSize=imSize, tarSize=tarSize)
        imId +=1
        cv2.imwrite(os.path.join(outputDir, '{}.png'.format(imId)), image)
        
    

if __name__ == "__main__":
    from dataset_coco import Dataset
    dataDir = '/home/jakub/data/coco'
    outputDir = '/home/jakub/data/results'
    now =datetime.datetime.now()
    outputDir = os.path.join(outputDir, now.strftime("%Y-%m-%d %H:%M"))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=24, 
                       batchSize=8)
    catNms = dataset.sCatNms + dataset.kpCatNms
    thresholds = {catNm:0.5 for catNm in catNms}
    #put special thresholds here:
    thresholds['person'] = 0.75
    for epoch in range(15):
        for inputs, targets, masks in dataset.iterateMinibatches(val=False):
            saveResults(inputs, targets, thresholds, drawBoxesCats=['person'],
                        drawCntsCats=kpCats, outputDir=outputDir)
    dataset.endDataset()

