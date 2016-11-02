#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:12:50 2016

@author: Ray
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 

boston = load_boston()

boston.keys()

boston.data.shape

print boston.DESCR

bos=pd.DataFrame(boston.data)

bos.columns= boston.feature_names

bos['PRICE'] = boston.target

X=np.asarray(bos['AGE'])
x=np.reshape(x,(506,1))
Y=np.asarray(bos['PRICE'])
y=np.reshape(y,(506,1))

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

lm= LinearRegression()
lm.fit(x,y)

#Quick prediction from function
lm.predict([80])

#B0 intercept
lm.intercept_

#B1 slope
lm.coef_

#Test the fit of variables provided 
lm.score(x,y)

ax=plt.scatter(x,y)
ax=plt.plot(x, lm.predict(x))

#%% Multipulenomial regression 
X1=np.asarray(bos['RM'])
x1=np.reshape(x1,(506,1))

X=np.asarray(bos['AGE'])
x2=np.reshape(X,(506,1))

#X_train= np.concatenate(x1,x2)

X_train=np.asarray([x1,x2])
X_train=np.reshape(X_train,(506,2))

xnew=np.concatenate((x1,x2), axis=1)

lm.fit(xnew,y)

lm.predict([6.5,80])

lm.coef_

lm.intercept_

lm.score(new, y)

#%% K Nearest Neighbors

from sklearn.neighbors import KNeighborsRegressor
kreg=KNeighborsRegressor()
kreg.fit(xnew,y)

kreg.score(xnew,y)
