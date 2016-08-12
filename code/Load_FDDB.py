import os
from PIL import Image
import numpy as np

pictureFold='../originalPics/'
annotationFold='../FDDB-folds/'
faceFold='../face/'
modelFold='../model/'
backgroundFold='../background/'

def extractFace(pictureFold,annotationFold,faceFold,backgroundFold):
    annotationFiles = os.listdir(annotationFold)
    count=0;
    for file in annotationFiles:
        f = open(annotationFold + file, "r")
        while True:
            line = f.readline()
            if line:
                imageName=line.rstrip('\n')+'.jpg'
                im = Image.open(pictureFold + imageName)
                size=im.size
                width,height=size[0],size[1]
                nb_face = (int)(f.readline())
                for i in range(nb_face):
                    line = f.readline().split(' ')
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y = int(float(line[0])), int(
                        float(line[1])), float(line[2]), int(float(line[3])), int(float(line[4]))
                    x0, y0, x1, y1 = center_x - minor_axis_radius, center_y - major_axis_radius, center_x + minor_axis_radius, center_y + major_axis_radius
                    x0,y0, x1, y1 =max(x0,0),max(y0,0),min(x1,width),min(y1,height)
                    nIm = im.crop((x0, y0, x1, y1))
                    nIm.save(faceFold + 'image'+str(count)+'.jpg')
                    count=count+1;
            else:
                break
        f.close()


def saveTrainData(pictureFold, annotationFold):
    annotationFiles = os.listdir(annotationFold)
    x_train_12_a = []
    y_train_12_a = []
    x_train_12_b = []
    y_train_12_b = []

    x_train_24_a = []
    y_train_24_a = []
    x_train_24_b = []
    y_train_24_b = []

    x_train_48_a = []
    y_train_48_a = []
    x_train_48_b = []
    y_train_48_b = []
    ls = [0.83, 0.91, 1.0, 1.1, 1.21]
    lx = [-0.17, 0, 0.17]
    ly = [-0.17, 0, 0.17]
    #count=0;
    for file in annotationFiles:
        f = open(annotationFold + file, "r")
        while True:
            line = f.readline()
            if line:
                imageName = line.rstrip('\n') + '.jpg'
                im = Image.open(pictureFold + imageName)
                size = im.size
                width, height = size[0], size[1]
                nb_face = (int)(f.readline())
                mode = im.mode
                if mode != 'RGB':
                    im=im.convert("RGB")
                for i in range(nb_face):
                    line = f.readline().split(' ')
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y = int(float(line[0])), int(
                        float(line[1])), float(line[2]), int(float(line[3])), int(float(line[4]))
                    x0, y0, x1, y1 = center_x - minor_axis_radius, center_y - major_axis_radius, center_x + minor_axis_radius, center_y + major_axis_radius
                    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
                    nIm = im.crop((x0, y0, x1, y1))
                    # 12-net positive trainning data
                    r, g, b = nIm.resize((12, 12)).split()
                    curIm = np.array([np.array(r), np.array(g), np.array(b)])
                    x_train_12_a.append(curIm)
                    y_train_12_a.append(1)
                    # 24-net positive trainning data
                    r, g, b = nIm.resize((24, 24)).split()
                    curIm = np.array([np.array(r), np.array(g), np.array(b)])
                    x_train_24_a.append(curIm)
                    y_train_24_a.append(1)
                    # 48-net positive trainning data
                    r, g, b = nIm.resize((48, 48)).split()
                    curIm = np.array([np.array(r), np.array(g), np.array(b)])
                    x_train_48_a.append(curIm)
                    y_train_48_a.append(1)
                    for i in range(5):
                        sn=ls[i]
                        for j in range(3):
                            xn =  lx[j]
                            for k in range(3):
                                yn = ly[k]
                                x, y, w, h = x0, y0, x1 - x0, y1 - y0
                                w, h=w *sn, h * sn
                                x, y= x + xn * w /sn, y + yn * h /sn
                                box = (int(x), int(y), int(x + w), int(y + h))
                                #12-calibration-net trainning data
                                newIm = im.crop(box).resize((12, 12))
                                r, g, b = newIm.split()
                                curIm = np.array([np.array(r), np.array(g), np.array(b)])
                                x_train_12_b.append(curIm)
                                y_train_12_b.append(i * 9 + j * 3 + k)
                                # 24-calibration-net trainning data
                                newIm = im.crop(box).resize((24, 24))
                                r, g, b = newIm.split()
                                curIm = np.array([np.array(r), np.array(g), np.array(b)])
                                x_train_24_b.append(curIm)
                                y_train_24_b.append(i * 9 + j * 3 + k)
                                # 48-calibration-net trainning data
                                newIm = im.crop(box).resize((48,48))
                                r, g, b = newIm.split()
                                curIm = np.array([np.array(r), np.array(g), np.array(b)])
                                x_train_48_b.append(curIm)
                                y_train_48_b.append(i * 9 + j * 3 + k)
                                #im.crop(box).save("../face2/image-48-cali-"+str(count)+'.jpg')
                                #count=count+1
            else:
                break
        f.close()
    #12-net negative trainning data
    negativeImages,negativelabels=getNegativeTrainData(backgroundFold,12,600)
    x_train_12_a.extend(negativeImages)
    y_train_12_a.extend(negativelabels)
    # 24-net negative trainning data
    negativeImages, negativelabels = getNegativeTrainData(backgroundFold, 24, 600)
    x_train_24_a.extend(negativeImages)
    y_train_24_a.extend(negativelabels)
    negativeImages, negativelabels = getNegativeTrainData(backgroundFold, 48, 600)
    # 48-net negative trainning data
    x_train_48_a.extend(negativeImages)
    y_train_48_a.extend(negativelabels)
    np.save('../model/x_train_12_a.npy', np.array(x_train_12_a))
    np.save('../model/y_train_12_a.npy', np.array(y_train_12_a))
    np.save('../model/x_train_24_a.npy', np.array(x_train_24_a))
    np.save('../model/y_train_24_a.npy', np.array(y_train_24_a))
    np.save('../model/x_train_48_a.npy', np.array(x_train_48_a))
    np.save('../model/y_train_48_a.npy', np.array(y_train_48_a))
    np.save('../model/x_train_12_b.npy', np.array(x_train_12_b))
    np.save('../model/y_train_12_b.npy', np.array(y_train_12_b))
    np.save('../model/x_train_24_b.npy', np.array(x_train_24_b))
    np.save('../model/y_train_24_b.npy', np.array(y_train_24_b))
    np.save('../model/x_train_48_b.npy', np.array(x_train_48_b))
    np.save('../model/y_train_48_b.npy', np.array(y_train_48_b))


def getNegativeTrainData(backgroundFold, size,length):
    images = []
    labels = []
    imageNames = os.listdir(backgroundFold)
    for imName in imageNames:
        im = Image.open(backgroundFold + imName)
        if im.mode != 'RGB':
            im = im.convert("RGB")
        hight,width = im.size
        for i in range(0,width-length,length):
            for j in range(0,hight-length,length):
                box = (j , i , j  + length, i + length)
                nIm = im.crop(box).resize((size, size))
                r, g, b = nIm.split()
                curIm = np.array([np.array(r), np.array(g), np.array(b)])
                images.append(curIm)
                labels.append(0)
    return images, labels

def loadFromFile():
    x_train_12_a = np.load('../model/x_train_12_a.npy')
    y_train_12_a = np.load('../model/y_train_12_a.npy')
    x_train_12_b = np.load('../model/x_train_12_b.npy')
    y_train_12_b = np.load('../model/y_train_12_b.npy')

    x_train_24_a = np.load('../model/x_train_24_a.npy')
    y_train_24_a = np.load('../model/y_train_24_a.npy')
    x_train_24_b = np.load('../model/x_train_24_b.npy')
    y_train_24_b = np.load('../model/y_train_24_b.npy')

    x_train_48_a = np.load('../model/x_train_48_a.npy')
    y_train_48_a = np.load('../model/y_train_48_a.npy')
    x_train_48_b = np.load('../model/x_train_48_b.npy')
    y_train_48_b = np.load('../model/y_train_48_b.npy')
    return (x_train_12_a, y_train_12_a), (x_train_12_b, y_train_12_b), \
           (x_train_24_a, y_train_24_a), (x_train_24_b, y_train_24_b), \
           (x_train_48_a, y_train_48_a), (x_train_48_b, y_train_48_b)



#saveTrainData(pictureFold, annotationFold)