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
import random
from multiprocessing import Process, Queue
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
#cat - category
#img - image
#ann - annotation

class Dataset:
    def __init__(self, dataDir, imageSize, targetSize, batchSize,
                 minLabelArea = 256., maxUpscale = 2., minUnderscale = 0.6):
        self.dataDir = dataDir
        self.imageSize = imageSize
        self.targetSize = targetSize
        self.batchSize = batchSize
        self.minLabelArea = minLabelArea
        self.maxUpscale = maxUpscale
        self.minUnderscale = minUnderscale
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
        cats = cocos[0].loadCats(cocos[0].getCatIds())
        sCatNms = sorted(list(set([cat['supercategory'] for cat in cats])))
        kpCatNms = cocos[2].loadCats(1)[0]['keypoints']
        weights = np.ones(29) #class weights (person and keypoints more important)
        weights[sCatNms.index('person')] = 2.0
        weights[-len(kpCatNms):] = 2.0
        borders = [False for _ in range(29)]
        borders[sCatNms.index('person')] = 'strict'
        borders[-len(kpCatNms):] = ['fuzzy' for _ in range(len(kpCatNms))]
        self.sCatNms = sCatNms
        self.kpCatNms = kpCatNms
        self.cocos = cocos
        self.types = types
        self.cats = cats
        self.weights = weights
        self.borders = borders
    def addSuperCat(self, ann):
        cats = self.cats
        annSupCat = [cat['supercategory'] for cat in cats \
                     if cat['id'] == ann['category_id']][0]
        ann['supercategory_id'] = self.sCatNms.index(annSupCat)
        
    def generateInstancesTargets(self, img, anns, coco):
        sCatNms = self.sCatNms
        w, h = img['width'], img['height']
        targets = [np.zeros((h, w), dtype = np.float32) for sCat in sCatNms]
        for ann in anns:
            sCatId = ann['supercategory_id']
            targets[sCatId] += coco.annToMask(ann).astype(np.float32)
        instTargets = np.stack(targets)
    def generateKeypointsTargets(self, img, anns, coco, tarDim=2):
        #generate target from person keypoints annotation
        #targets are point, but extended circles with radius tarDim
        kpCatNms = self.kpCatNms
        w, h = img['width'], img['height']
        targets = [np.zeros((h, w), dtype = np.float32) for kpCat in kpCatNms]
        for ann in anns:
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                #sks = np.array(coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
                #kpn = coco_kps.loadCats(ann['category_id'])[0]['keypoints']
                kps = np.array(ann['keypoints'])
                x = kps[0::3]
                y = kps[1::3]
                v = kps[2::3]
                for i, (xCor, yCor, vis) in enumerate(zip(x, y, v)):
                    if np.all([xCor, yCor, vis]):
                        '''
                        vis = 0 -> not visible nor labeled
                        vis = 1 -> not visible but labeled
                        vis = 2 -> visible and labeled
                        '''
                        print i, xCor, yCor, tarDim, vis
                        cv2.circle(targets[i], (xCor, yCor),tarDim, vis, -1)
        kpTargets = np.stack(targets)
        
            
        
    def generateMask(self, totalTargets, borderW=5, 
                     strictBMlp=4.0, fuzzyBMlp = 0.0):
        '''
        strictBorder - weight on border more important than inside of instance
                       borders more precise, harder to train
        fuzzyBorder - zero weight on border - easier to train
        '''
        weights = self.weights
        borders = self.borders
        masks = np.copy(totalTargets)
        masks *= weights.reshape((-1, 1, 1))
        kernel = np.ones((borderW, borderW),np.float32)
        for mask, border in zip(masks, borders):
            if border:
                dilation = cv2.dilate(mask,kernel,iterations = 1)
                outBorder = dilation - mask
            mask[mask == 0] = 1.0
            if border == 'strict':
                mask[outBorder > 0] = strictBMlp
            elif border == 'fuzzy':
                mask[outBorder > 0] = fuzzyBMlp
        
    def scaleStride(self, img, anns):
        #generate random scale and stride to select patch from image
        #to select top left corner:
        scale = 1.0
        stride = (0, 0)
        maxUpscale = self.maxUpscale
        minUnderscale = self.minUnderscale
        minLabelArea = self.minLabelArea
        imageSize = self.imageSize
        scale = minUnderscale + (maxUpscale - minUnderscale)*random.random()
        w, h = img['width'], img['height']
        minImgLabel = np.min([ann['area'] for ann in anns])
        #Label area should be bigger than minLabelArea - 256 (16px x 16px)
        scale = max([scale, minLabelArea / minImgLabel])
        wScaled, hScaled = [int(dim * scale) for dim in [w, h]]
        stride = [np.random.randint(0, dim - imageSize) for dim in [wScaled, hScaled]]
        return scale, stride

    def cutPatch(self, I, scale = 1.0, stride = [0, 0]):
        # TODO!
        if len(np.shape(I)) == 2:
            I = np.expand_dims(I, axis = 0)
        
    def batchGenerator(self, queue, val=False):
        cocos = self.cocos
        dataDir = self.dataDir
        types = self.types
        workingCocos = cocos[val::2] #instances and person keypoints
        dataType = types[val]
        imgIds = workingCocos[0].getImgIds()
        seed = 7
        random.seed(seed)
        random.shuffle(imgIds)
        imgIter = 0        
        while(1):
            imgId = imgIds[imgIter]
            img = workingCocos[0].loadImgs(imgId)[0] #only one image
            annIds = workingCocos[0].getAnnIds(imgIds=img['id'])
            anns = workingCocos[0].loadAnns(annIds)
            for ann in anns:
                self.addSuperCat(ann)
            imgPath = os.path.join(dataDir, dataType, img['file_name'])
            I = cv2.imread(imgPath, 0)
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.imshow('test', I)
            cv2.waitKey(1000)
            cv2.destroyWindow('test')
            imgIter+=1
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
    for ann in anns:
        dataset.addSuperCat(ann)
    coco.showAnns(anns)
    
    # initialize COCO api for person keypoints annotations
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
    coco_kps=COCO(annFile)
    
    plt.imshow(I); plt.axis('off')
    ax = plt.gca()
    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    for ann in anns:
        dataset.addSuperCat(ann)
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
                kpNms = coco_kps.loadCats(ann['category_id'])[0]['keypoints']
                kps = np.array(ann['keypoints'])
                x = kps[0::3]
                y = kps[1::3]
                v = kps[2::3]
                
