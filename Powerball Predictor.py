#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:19:30 2016

@author: Ray
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 

#%%

data = pd.read_csv("/Users/undergroundking8o8/Downloads/Powerball winnums-text.csv")
#retry
data = pd.read_csv("/Users/undergroundking8o8/Downloads/Powerball winnums-text2.csv")
data.columns

data['WB1'].hist()
data['WB2'].hist()
data['WB3'].hist()
data['WB4'].hist()
data['WB5'].hist()
data['PB'].hist()
data['PP'].hist()

data[['WB1', 'WB2', 'WB3','WB4', 'WB5']].max()
data[['WB1', 'WB2', 'WB3','WB4', 'WB5']].min()

#%%

#%% Cross validation function

x = data[['Draw Date']]
#next()
#for row in x:
#    x.append(int(row[0].split('/')[0]))
#return 
#x.lstrip('/')
#x['Draw'][1::]
#
#if "/" in x:
#    (s_l, slash, s_r) = x.partition("/")
#x2= x.replace('/%', "")
#del x['PP']
data2=[]
x2 = data['Draw Date'].map(lambda x: x.lstrip('/').rstrip('/'))

y=data[['WB1', 'WB2', 'WB3','WB4', 'WB5','PB']]

from sklearn import cross_validation
x_train, x_test, y_train, y_test= cross_validation.train_test_split(x,y, test_size=0.3, random_state=0)



#%% works
from sklearn.ensemble import ExtraTreesClassifier
ETC=ExtraTreesClassifier()
ETC.fit(x_train, y_train)
#Fit
ETC.fit(x, y)

print ETC.score(x_test, y_test)

#predicts numbers bases on month entered 
print ETC.predict(11)
p= ETC.predict(test)

#%%
from sklearn.ensemble import RandomForestRegressor
ranf = RandomForestRegressor()
ranf.fit(x, y)
print(ranf.score(x_test, y_test))
# Fit the model using all the available data
ranf.fit(x_train, y_train)
print ranf.predict(11)

#%%
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(x_train, y_train)

#predicts numbers bases on month entered 
print clf.predict(11)

#%% SVR
from sklearn.svm import SVR

svr_lin = SVR(kernel= 'linear', C=1e3)
svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma = 2)

svr_lin.fit(x_train, y_train)
svr_poly.fit(y_train, x_train)
svr_rbf.fit(y_train, x_train)

print svr_lin.predict(11)

print svr_lin.predict(y_train)

plt.scatter(y_train, x_train, color='black', lable='Data')
plt.plot(x_train, svr_lin.predict(x_train), color='blue', lable='Lin Model')