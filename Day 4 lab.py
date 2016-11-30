#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:59:33 2016

@author: Ray

This code is for stock prediction one machine learning challenge
Given a dataset that consists of the (simulated) daily open­to­close changes of a set of 10
stocks: S1, S2, …, S10. Stock S1 trades in the U.S. as part of the S&P 500 Index, while stocks S2, S3, …,
S10 trade in Japan, as part of the Nikkei Index.
Your task is to build a model to forecast S1, as a function of S1, S2, …, S10​. You should build your
model using the first 50 rows of the dataset. The remaining 50 rows of the dataset have values for S1
missing: you will use your model to fill these in.
"""
#%% Import libraries

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
#%% Read data and preprocessing

data=pd.read_csv('/Users/undergroundking8o8/Documents/Data Sci Code/stock_returns_base150.csv')

#takes first 50 values from first column
y=np.asarray(data['S1'][0:50])

#Takes all 100 values of the rest of the table
x=data[['S2','S3','S4','S5','S6','S7','S10']]
xdata=x.as_matrix()
#extracts first 50 of the rest of columns
x=xdata[0:50,:]
x_prediction=xdata[50:100,:]

#%% Cross validation to evaluate the classifier random state 0 picks first, Test _SIze is amount to test
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(x,y, test_size=0.7, random_state=0)

#First find out what regession types/algorethems are avaliable 

#decision tree algorythem created using random normal distributions
ranf = RandomForestRegressor()
ranf.fit(X_train, Y_train)
print(ranf.score(X_val, Y_val))
# Fit the model using all the available data
ranf.fit(X_train, Y_train)



p = ranf.predict(x_prediction)

#%%
from sklearn import linear_model

bayr = linear_model.BayesianRidge()
bayr.fit(X_train, Y_train)
print (bayr.score(X_val, Y_val))

bayr.predict(x_prediction)
#%% 2. Replace the nan values in S1 with the predicted values for s1 from the random-forest regressor we used in class
#data.insert(50, 'p')
#
#data[0:50]
#data.S1.insert[50:] = p
#data.S1[50::]
#
#np.append(data, p, axis=0)

data['S1'][50:100] = p
#%%Replace the columns of s5 from rows 50-100 with nan just like how s1 was initially.

#data.empty((50, 5))
#np.delete(data, 5, 0)
#
#new_data = data.iloc[:-50, :].copy()

data['S5'][50:100] = float('NaN')

#%% Perform regression using at least 5 random regression algorithms but use a function to calculate score and print out the same.
#S5 prediction 
ys5=np.asarray(data['S5'][0:50])

#Takes all 100 values of the rest of the table
xs5=data[['S2','S3','S4','S1','S6','S7','S10']]
xs5data=xs5.as_matrix()
#extracts first 50 of the rest of columns
xs5=xs5data[0:50,:]
xs5_prediction=xs5data[50:100,:]

Xs5_train, Xs5_val, Ys5_train, Ys5_val = cross_validation.train_test_split(xs5,ys5, test_size=0.7, random_state=0)

#RandomForestRegressor() 90%

RFR = RandomForestRegressor()
RFR.fit(Xs5_train, Ys5_train)
print(RFR.score(Xs5_val, Ys5_val))
# Fit the model using all the available data
RFR.fit(Xs5_train, Ys5_train)

RFR.predict(xs5_prediction)

#ARD 96%
from sklearn import linear_model

ARD = linear_model.ARDRegression()
ARD.fit(Xs5_train, Ys5_train)
print (ARD.score(Xs5_val, Ys5_val))

ARD.predict(xs5_prediction)

#LR 93%
from sklearn import linear_model

LR = linear_model.LinearRegression()
LR.fit(Xs5_train, Ys5_train)
print (LR.score(Xs5_val, Ys5_val))

LR.predict(xs5_prediction)

#RANSACRegressor 86%
from sklearn import linear_model

RAN = linear_model.RANSACRegressor()
RAN.fit(Xs5_train, Ys5_train)
print (RAN.score(Xs5_val, Ys5_val))

RAN.predict(xs5_prediction)

#TheilSenRegressor 85%
from sklearn import linear_model

TSR = linear_model.TheilSenRegressor()
TSR.fit(Xs5_train, Ys5_train)
print (TSR.score(Xs5_val, Ys5_val))

TSR.predict(xs5_prediction)

#%%
Best model is ADR = ARDRegression() with 96%
