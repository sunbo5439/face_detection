# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:49:46 2016

@author: sunbo
"""
import Scanner
import Net_12,Net_24,Net_48,Calibration_12,Calibration_24,Calibration_48
import Load_FDDB

testImageFolder='../test_image/'
testImageFolder2='../test_image2/'
resultFolder='../result/'


(x_train_12_a, y_train_12_a), (x_train_12_b, y_train_12_b), (x_train_24_a, y_train_24_a), (x_train_24_b, y_train_24_b), (x_train_48_a, y_train_48_a), (x_train_48_b, y_train_48_b)=Load_FDDB.loadFromFile()
#  trainning the neural worknets
'''
print 'trainning 12-net.....................................................'
Net_12.train_12net(x_train_12_a, y_train_12_a)
print 'trainning 24-net.....................................................'
Net_24.train_24net(x_train_24_a,x_train_12_a, y_train_24_a)
print 'trainning 48-net.....................................................'
Net_48.train_48net(x_train_48_a,x_train_24_a, y_train_48_a)

print 'trainning 12-calibration_net.........................................'
Calibration_12.train_12calibration_net(x_train_12_b, y_train_12_b)
print 'trainning 24-calibration_net.........................................'
Calibration_24.train_24calibration_net(x_train_24_b, y_train_24_b)

print 'trainning 48-calibration_net.........................................'
Calibration_48.train_48calibration_net(x_train_48_b[:100], y_train_48_b[:100])

net_12 = Net_12.get_12net()
net_24 = Net_24.get_24net()
net_48 = Net_48.get_48net()
calibration_12 = Calibration_12.get_12calibration()
calibration_24 = Calibration_24.get_24calibration()
calibration_48 = Calibration_48.get_48calibration()

'''
imageNames=Scanner.getImageNames(testImageFolder2)
imageNames.sort()
for imageName in imageNames:
    print ('processing image:'+imageName)
    Scanner.doit(testImageFolder2,imageName,resultFolder)



print '----end----'