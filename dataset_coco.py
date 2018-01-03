#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:01:20 2018
Plik przygotowujący obrazy ze zbioru COCO przy użyciu API i narzędzi Kerasa
@author: jakub
"""
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import cv2
from multiprocessing import Process, Queue
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

class Dataset:
    def __init__(self, dataDir, imageSize, targetSize, batchSize):
        self.dataDir = dataDir
        self.imageSize = imageSize
        self.targetSize = targetSize
        self.batchSize = batchSize
        self.initImages()
        self.initWorkers()
    def initImages(self):
        #change with final dataset!
        trainType, valType = 'val2017', 'val2017'
        types = [trainType, valType]
        annFiles = ['{}/annotations/instances_{}.json'.format(dataDir, dataType) \
                    for dataType in types]
        annFiles += ['{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType) \
                    for dataType in types]
        cocos = [COCO(annFile) for annFile in annFiles]
        cats = coco.loadCats(cocos[0].getCatIds())
        self.sCatNms = sorted(list(set([cat['supercategory'] for cat in cats])))
        self.kpCatNms =  cocos[2].loadCats(1)[0]['keypoints']
    def generateTargets(self):
        pass
    def generateMask(self):
        pass
    def batchGenerator(self, queue, val=False):
        while(1):
            queue.put((1,2,3))
            print '123 added', val
    def initWorkers(self):
        self.trainQ = None
        self.valQ = None
        self.trainQ = Queue(12)
        self.valQ = Queue(5)
        self.trainP = Process(target=self.batchGenerator, args=(self.trainQ, False))
        self.trainP.start()
        self.valP = Process(target=self.batchGenerator, args=(self.valQ, True))
        self.valP.start()
        
if __name__ == "__main__":
    dataDir = '/home/jakub/data/fxi/coco'
    dataset  = Dataset(dataDir=dataDir, imageSize=256, targetSize=40, 
                       batchSize=32)


    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    # initialize COCO api for instance annotations
    coco = COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print 'COCO categories: \n{}\n'.format(' '.join(nms))
    
    nms = set([cat['supercategory'] for cat in cats])
    print 'COCO supercategories: \n{}'.format(' '.join(nms))
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person','cat'])
    #catIds = coco.getCatIds()
    imgIds = coco.getImgIds(catIds=catIds )
    #imgIds = coco.getImgIds(imgIds = [324158])
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    img_path = os.path.join(dataDir, dataType, img['file_name'])
    
    I = io.imread(img_path)
    plt.axis('off')
    plt.imshow(I)
    plt.show()
    
    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    
    # initialize COCO api for person keypoints annotations
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
    coco_kps=COCO(annFile)
    
    plt.imshow(I); plt.axis('off')
    ax = plt.gca()
    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    coco_kps.showAnns(anns)
    polygons = []
    mask = np.zeros((I.shape[0], I.shape[1]), dtype = np.float32)
    
    for ann in anns:
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    poly = np.expand_dims(poly, 1).astype(np.int32)
                    mask = cv2.drawContours(mask, [poly], 0, 1.0,-1)
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                sks = np.array(coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
                kpn = coco_kps.loadCats(ann['category_id'])[0]['keypoints']
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
