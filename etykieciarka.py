#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Thu Feb  1 07:18:32 2018
Aplikacja pozwalająca na opisywanie punktów kluczowych człowieka
Nie używa środka i skali, gdyż wydają się niepotrzebne, ewentualnie można je 
obliczyć z punktów kluczowych
@author: jakub
'''
import cv2
import numpy as np
import glob
import os
import re
from annotations import annotations

def loadImages(dataDir):
    extensions = ['jpg', 'png', 'bmp', 'jpeg']
    extensions += [x.upper() for x in extensions]
    filelist = []
    for ext in extensions:
        filelist.extend(glob.glob(os.path.join(dataDir, '*.{}'.format(ext))))
    return filelist
def loadImage(fileName):
    return cv2.imread(fileName)
def addFrame(image, margin = 100):
    h, w, ch = image.shape
    framedImg = 255 * np.ones((h+2*margin, w+2*margin, ch), dtype = np.uint8)
    framedImg[margin:-margin, margin:-margin] = image
    return framedImg
def writeText(image, margin, text):
    color = (0,0,255)
    h, w, ch = image.shape
    ht = h - 10
    wt = w / 2 -10
    return cv2.putText(image, text, (wt, ht),cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                       2, lineType=cv2.LINE_AA)
clicked = False
sx, sy = 0, 0 #start coordinates for bbox
anns = annotations()
POSE_COCO_BODY_PARTS = ['nose',
                        'neck',
                        'rshoulder',
                        'relbow',
                        'rwrist',
                        'lshoulder',
                        'lelbow',
                        'lwrist',
                        'rhip',
                        'rknee',
                        'rankle',
                        'lhip',
                        'lknee',
                        'lankle',
                        'reye',
                        'leye',
                        'rear',
                        'lear']
MODES = [                'NEW',
                         'SHIFT',
                         'DELETE',
                         'MASK',
                         'BBOX']
bodyPart = 0
bbox = None
windowName = 'Etykieciarka'
margin = 100
winW, winH = 1200, 800
dataDir = os.path.expanduser('~/data/fxtest')
mode = 0
def selectMode(lastChars):
    global POSE_COCO_BODY_PARTS, MODES, MODES_KEYS
    assert( len(lastChars) <= 3)  
    mode = 0
    bodyPart = 0 
    if len(lastChars) > 0:
        modeIds = [x[0] for x in MODES]
        if lastChars[-1] in modeIds:
            mode = modeIds.index(lastChars[-1])
        else:
            result = []
            for l in POSE_COCO_BODY_PARTS:
                match = re.search('^{}'.format(lastChars),l, re.IGNORECASE)
                if match:
                    result += [l]
            if result:
                print result
                bodyPart = POSE_COCO_BODY_PARTS.index(result[0])
                print bodyPart
    return mode, bodyPart

def mouseCallback(event,x,y,flags,param):
    global clicked, annotations, sx, sy, mode, anns, bbox
    if MODES[mode] == 'NEW':
        pass
    if MODES[mode] == 'SHIFT':
        pass
    if MODES[mode] == 'DELETE':
        pass
    if MODES[mode] == 'MASK':
        pass
    if MODES[mode] == 'BBOX':
        if event == cv2.EVENT_LBUTTONDOWN:
            sx, sy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            anns.newBBox(sx, sy, x, y, bbox)
            bbox = anns.getCurAnnId()
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print 'Click: {} {}'.format(x, y)
            


filelist = loadImages(dataDir)
for fileName in filelist:
    bbox = None
    image = loadImage(fileName)
    fImage = addFrame(image, margin)
    tImage = writeText(fImage, margin, 'Tryb:')
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, winW, winH)
    cv2.setMouseCallback(windowName, mouseCallback)
    lastChars = ''
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if chr(k).isdigit():
            print k
            bbox = int(chr(k))
            anns.setCurAnnId(bbox)
        elif k!=27 and k!=255:
            lastChars += chr(k)
            lastChars = lastChars[-3:]
            mode, bodyPart = selectMode(lastChars)
        elif k == 27:
            break
        stStr = 'Tryb: {}'.format(MODES[mode])
        if mode == 0:
            stStr += ' {}'.format(POSE_COCO_BODY_PARTS[bodyPart])
        if bbox >=0:
            stStr += ' BBOX: {}'.format(bbox)
        tImage = writeText(np.copy(fImage), margin, stStr)
        cv2.imshow(windowName, tImage)
cv2.destroyWindow(windowName)