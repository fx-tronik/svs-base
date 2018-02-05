#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:47:06 2018
Adnotacje do etykieciarki
@author: jakub
"""

import numpy as np
def search_dictionaries(key, value, list_of_dictionaries):
    return [i for i, element in enumerate(list_of_dictionaries) if element[key] == value]
class annotations:
    def __init__(self):
        data = []
        curAnnId = None
        self.data = data
        self.curAnnId = curAnnId
    def getCurAnnId(self):
        return self.curAnnId
    def setCurAnnId(self, curAnnId):
        self.curAnnId = curAnnId
    def getAnnsNum(self):
        return len(self.data)
    def newBBox(self, sx, sy, x, y, bboxId = None):
        tlx = np.min([x, sx])
        tly = np.min([y, sy])
        brx = np.max([x, sx])
        bry = np.max([y, sy])
        if bboxId is None:
            print 'notID', bboxId
            bboxId = len(self.data)
        ann = {'bbox':[tlx, tly, brx, bry], 'id': bboxId}
        if bboxId not in [anx['id'] for anx in self.data]:
            self.data.append(ann)
            print 'Dodano nowy ann'
        else:
            idToOverwrite = search_dictionaries('id', bboxId, self.data)            
            assert len(idToOverwrite) == 1
            self.data[idToOverwrite[0]] = ann
            print 'Nadpisano ann w boksie {}'.format(bboxId)
        self.curAnnId = bboxId
        

if __name__ == '__main__':
    anns = annotations()
    print anns.getAnnsNum()     
    anns.newBBox(100,200,200,100)
    print 'Aktualny BBOX', anns.getCurAnnId()
    anns.newBBox(100,200,200,100, 0)
    print 'Aktualny BBOX', anns.getCurAnnId()
    anns.newBBox(100,200,200,100, 7)
    print 'Aktualny BBOX', anns.getCurAnnId()
    data = anns.data