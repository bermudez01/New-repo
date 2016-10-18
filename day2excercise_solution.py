# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 00:32:30 2016

@author: anandpreshob
"""

#%% Matrix
#define matrix
A = np.array([[1, 2, -1], [1, 1, 1], [1, 2, 1]])
B = np.array([42, -2, 2])

#Inverse of A
A_inverse = np.linalg.inv(A)

# Here you have to perform matrix multiplication
# You had done element-wise multiplication instead
X = np.dot(A_inverse,B)

# Checking the answer
print np.dot(A,X)
#%% 3.

data = pd.read_csv("Downloads/weather_year.csv")
data.columns
data.columns = ["date", "max_temp", "mean_temp", "min_temp", "max_dew",
                "mean_dew", "min_dew", "max_humidity", "mean_humidity",
                "min_humidity", "max_pressure", "mean_pressure",
                "min_pressure", "max_visibilty", "mean_visibility",
                "min_visibility", "max_wind", "mean_wind", "min_wind",
                "precipitation", "cloud_cover", "events", "wind_dir"]

warm_days = data[data.min_temp >= 60]

warm_days.mean_temp.hist()
ax = warm_days.max_pressure.plot(title="Min and Max Pressure")
warm_days.min_pressure.plot(style="red", ax=ax)
ax.set_ylabel("Temperature (F)")

# 4. 
corr_data=warm_days.corr()
# We will discuss how I found the min value and the max value
# The minimum correlation is between max temperature and min humidity
fig1 = plot.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(warm_days.max_temp,warm_days.min_humidity)
ax1.set_xlabel('max temperature')
ax1.set_ylabel('min humidity')
# The Maximum Correlation is between the max pressure and min pressure
fig1 = plot.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(warm_days.max_pressure,warm_days.min_pressure)
ax1.set_xlabel('max pressure')
ax1.set_ylabel('min pressure')