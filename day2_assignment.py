#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 13:13:42 2016

@author: Ray
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#%% Pandas
#1
#Declaring a pandas dataframe
df = pd.DataFrame({'example1' : [123, 116, 122, 110, 175, 126, 125, 111, 118, 117], 
#'example2' : [75, 87, 49, 68, 75, 84, 98, 92],
 #'example3' : [55, 47, 38, 66, 56, 64, 44, 39] 
 })
print df

print df.mean()

print df.median()

print df.mode()

print df["example1"].var()

print df.std()

#%%
#IQR
print "50% Quartile:"
print df.quantile(.50) 
print "Median (red line of the box)"
print df.median()


print"25% (bottom of the box)"
print df.quantile(0.25)
print"75% (top of the box)"
print df.quantile(0.75)


df.plot(kind='box')
df['example1'].plot(kind='box')

print df.describe()

df.corr()

print df['example1'].max()
print df.max()

print df.min()
print df.median()

#%% Matrix
#define matrix
A = np.array([[1, 2, -1], [1, 1, 1], [1, 2, 1]])
B = np.array([42, -2, 2])

#Inverse of A
A_inverse = np.linalg.inv(A)

X = A_inverse * B 

#Matrix Multiplication
z=np.dot(A_inverse,B)

print X
print z

#%% 3.

data = pd.read_csv("/Users/undergroundking8o8/Downloads/weather_year.csv")
data.columns
data['min_temp']
data.date
data.date.head()
data['date'].tail()
data['mean_temp'].hist()

data.columns = ["date", "max_temp", "mean_temp", "min_temp", "max_dew",
                "mean_dew", "min_dew", "max_humidity", "mean_humidity",
                "min_humidity", "max_pressure", "mean_pressure",
                "min_pressure", "max_visibilty", "mean_visibility",
                "min_visibility", "max_wind", "mean_wind", "min_wind",
                "precipitation", "cloud_cover", "events", "wind_dir"]

warm_days = data[data.min_temp >= 60]
warm_date = warm_days['date']

#==============================================================================
# data.warm_days.hist()
# 
#==============================================================================
warm_days.hist('min_temp')

warm_days.tail().plot()

warm_days.max_temp.tail().plot()
warm_days.max_temp.tail().plot(kind="bar", rot=10)

ax = warm_days.max_temp.plot(title="Min and Max Temperatures")
warm_days.min_temp.plot(style="red", ax=ax)
ax.set_ylabel("Temperature (F)")
ax.set_xlabel("Day")

# 4. 

warm_days.corr()

warm_days.corr().min 

new_corr =corr_data

warm_days.corr().min_temp.plot()

x = warm_days
y = warm_date

warm_days['date'].tail()

plot.scatter(x,y, label='Warm Day Corr', color='k')
