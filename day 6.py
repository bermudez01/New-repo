#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:27:32 2016

@author: Ray
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 

#%%
from sklearn.datasets import load_digits 

data = load_digits()

data.keys()

print data['DESCR']

#%%

plt.gray()
plt.imshow(data.images[-4], cmap='summer')
#plt.imshow(data.images[-4], cmap='color')

#%% Normalization process
from sklearn import preprocessing
x=preprocessing.normalize(data['data'])
y=data.target

#%% Cross validation function
from sklearn import cross_validation
x_train, x_test, y_train, y_test= cross_validation.train_test_split(x,y, test_size=0.3, random_state=0)

#%% Navie Baised Class
from sklearn.naive_bayes import GaussianNB
GNB= GaussianNB()
GNB.fit(x_train,y_train)

print GNB.score(x_test, y_test)
#%% Decision Tree Class
from sklearn.tree import DecisionTreeClassifier
dct= DecisionTreeClassifier()
dct.fit(x_train, y_train)

print dct.score(x_test, y_test)

#%% SVM
from sklearn.svm import LinearSVC 
SVC= LinearSVC()
SVC.fit(x_train, y_train)

print SVC.score(x_test, y_test)

#%% K Nearest Neibors 
from sklearn.neighbors import KNeighborsClassifier
NN=KNeighborsClassifier()
NN.fit(x_train, y_train)

print NN.score(x_test, y_test)

#%% COnfistion matrix
from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test, NN.predict(x_test))

print cf

#%% Skimage
import skimage
from skimage import io
from skimage import transform

x_four= skimage.io.imread('mnisteight.jpg')

from skimage import color

x_four= skimage.color.rgb2grey(x_four)

plt.imshow(x_four)

x_fournew= skimage.transform.downscale_local_mean(x_four, (416, 234))
x_four_re=np.reshape(x_fournew,(1,64))
print NN.predict_proba(x_four_re)

print NN.predict(x_four_re)
