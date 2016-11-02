#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:26:25 2016

@author: Ray
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import sklearn.datasets
#%% 
data=sklearn.datasets.load_breast_cancer()
x=data['data']
y=data['target']

#%%
from sklearn import cross_validation
x_train, x_test, y_train, y_test= cross_validation.train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(x_train, y_train)                         

clf.predict(x_test)
clf.score(x_test, y_test)
