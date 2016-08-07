# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:11:56 2016

@author: sunbo
"""
from PIL import Image, ImageDraw
import numpy as np
import Net_12, Net_24, Net_48, Calibration_12, Calibration_24, Calibration_48
import os
import nms
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


XML_RESULT='result.xml'

ls = [0.83, 0.91, 1.0, 1.1, 1.21]
lx = [-0.17, 0, 0.17]
ly = [-0.17, 0, 0.17]

net_12 = Net_12.get_12net()
net_24 = Net_24.get_24net()
net_48 = Net_48.get_48net()
calibration_12 = Calibration_12.get_12calibration()
calibration_24 = Calibration_24.get_24calibration()
#calibration_48 = Calibration_48.get_48calibration()


def getImageNames(inputFolder):
    imageNames = os.listdir(inputFolder)
    return imageNames


def doit(imageFolder,imageName,resultFolder):
    origin_image=Image.open(imageFolder+ imageName)
    width, height = origin_image.size
    stride = 10
    length_stard,length_end = 100,250
    window12 = []
    window24 = []
    window48 = []
    proba_12=[]
    proba_24=[]
    proba_48=[]
    for length in range(length_stard,length_end,20):
        for i in range(0, height-length, stride):
            for j in range(0, width-length, stride):
                left, top, right, below = j, i, j + length, i + length
                box = (left, top, right, below)
                image = origin_image.crop(box).resize((12, 12))
                x_12 = getNpArray(image)
                prob_a = net_12.predict_proba(x_12, batch_size=32, verbose=0)
                if prob_a[0][1] < 0.5:
                    continue
                label_b = calibration_12.predict_proba(x_12, batch_size=32, verbose=0)
                sn, xn, yn = getSXY(label_b[0])
                x, y, w, h = left, top, right - left, below - top
                x, y, w, h = int(x - xn * w / sn), int(y - yn * h / sn), int(w / sn), int(h / sn)
                x,y,w,h=max(0,x),max(0,y),max(0,w),max(0,h)
                window12.append((x, y, x + w, y + h))
                #proba_12.append(prob_a)
        #window12 = nms.non_max_suppression_slow(np.array(window12), 0.95).tolist()
        #window12=nwindow12.tolist()
    for length in range(length_stard, length_end, 20):
        for box in window12:
            image12 = origin_image.crop(box).resize((12, 12))
            image24 = origin_image.crop(box).resize((24, 24))
            x_24 = getNpArray(image24)
            x_12 = getNpArray(image12)
            prob_a = net_24.predict_proba([x_24, x_12], batch_size=32, verbose=0)
            if prob_a[0][1] < 0.5:
                continue
            label_b = calibration_24.predict_proba(x_24, batch_size=32, verbose=0)
            sn, xn, yn = getSXY(label_b[0])
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            x, y, w, h = int(x - xn * w / sn), int(y - yn * h / sn), int(w / sn), int(h / sn)
            x,y,w,h=max(0,x),max(0,y),max(0,w),max(0,h)
            # (left,top,right,below)=(x,y,x+w,y+h)
            window24.append((x, y, x + w, y + h))
        #window24 = nms.non_max_suppression_slow(np.array(window24), 0.9).tolist()
        #window12 = nwindow12.tolist()
            #proba_24.append(prob_a)
    for length in range(length_stard, length_end, 20):
        for box in window24:
            image48 = origin_image.crop(box).resize((48, 48))
            image24 = origin_image.crop(box).resize((24, 24))
            x_48 = getNpArray(image48)
            x_24 = getNpArray(image24)
            prob_a = net_48.predict_proba([x_48, x_24], batch_size=32, verbose=0)
            if prob_a[0][1] < 0.5:
                continue
            '''
            label_b = calibration_48.predict_proba(x_48, batch_size=32, verbose=0)
            sn, xn, yn = getSXY(label_b[0])
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            x, y, w, h = int(x - xn * w / sn), int(y - yn * h / sn), int(w / sn), int(h / sn)
            x,y,w,h=max(0,x),max(0,y),max(0,w),max(0,h)
            # (left,top,right,below)=(x,y,x+w,y+h)
            window48.append((x, y, x + w, y + h))
            '''
            window48.append(box)  ######################################################################
            #proba_48.append(prob_a)
    #window48 = nms.non_max_suppression_slow(np.array(window48), 0.5)
    for box in window48:
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        draw = ImageDraw.Draw(origin_image)
        draw.rectangle([x0, y0, x1, y1], outline=128)
    #setResult(window48,imageName)
    origin_image.save(resultFolder+'result_'+imageName)


def getSXY(label):
    top_size = 6
    index_sorted = np.argsort(label)
    top = index_sorted[-1 * top_size:]
    s, x, y = 0., 0., 0.
    for index in top:
        i = index / 9
        index %= 9
        j = index / 3
        k = index % 3
        s += ls[i]
        x += lx[j]
        y += ly[k]
    s /= top_size
    x /= top_size
    y /= top_size
    return s, x, y


def getNpArray(image):
    r, g, b = image.split()
    curIm = np.array([np.array(r), np.array(g), np.array(b)])
    curIm = curIm.astype('float32')
    curIm /= 256
    x = np.array([curIm])
    return x
    
    
def setResult(boxs,imageName):
    tree = ET.parse(XML_RESULT)
    root = tree.getroot()
    items = root.find('Items')
    subItem=ET.SubElement(items, 'Item')
    subItem.set('imageName',imageName)
    for box in boxs:
        label=ET.SubElement(subItem, 'Label')
        label.set('id','0')
        label.set('score','0')
        label.set('l',str(box[0]))
        label.set('t',str(box[1]))
        label.set('r',str(box[2]))
        label.set('b',str(box[3]))
    tree.write(XML_RESULT)
    