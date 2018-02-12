#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 06:28:01 2018
CameraIP support
@author: jakub
"""

import base64
import time
import urllib2

import cv2
import numpy as np
class ipCamera(object):

    def __init__(self, url, user=None, password=None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame

#indie = ipCamera('http://203.153.33.101:86/')
#frame = indie.get_frame()

#vcap = cv2.VideoCapture('http://203.153.33.101:86/SnapshotJPEG?Resolution=640x480&Quality=Clarity&Count=-522640221')
#ret, frame = vcap.read()
cameras = {}
#PANASONIC
cameras['indie'] = 'http://203.153.33.101:86/SnapshotJPEG?Resolution=640x480&Quality=Clarity'
cameras['osaka'] = 'http://202.215.124.190:83/SnapshotJPEG?Resolution=640x480&Quality=Clarity'
cameras['saitama'] = 'http://153.167.126.151/SnapshotJPEG?Resolution=640x480&Quality=Clarity'
cameras['tavastia'] = 'http://91.153.184.52:8082/SnapshotJPEG?Resolution=640x480&Quality=Clarity'
#AXIS
cameras['freiburg'] = 'http://80.153.33.108:90/mjpg/video.mjpg'
#SONY
cameras['chipperfield'] = 'http://81.149.216.162:8040/oneshotimage1'
cameras['thuringen'] = 'http://82.144.57.102/oneshotimage.jpg'
#VIVOTEK
cameras['languedoc'] = 'http://80.14.208.115:8055/cgi-bin/viewer/video.jpg?resolution=640x480'
#LINKSYS
cameras['attiki'] = 'http://79.129.108.146:1024/img/snapshot.cgi?size=3'
toTest = 'thuringen'
vcap = cv2.VideoCapture(cameras[toTest])
ret, frame = vcap.read()
if ret:
    cv2.imwrite('./cameras/{}.png'.format(toTest), frame)