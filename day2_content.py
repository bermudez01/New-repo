# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 18:36:36 2016
HOL for day 2
@author: anandpreshob
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#%% OOP for Python
class Employee:
   'Common base class for all employees'
   empCount = 0

   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1
   
   def displayCount(self):
     print "Total Employee %d" % Employee.empCount

   def displayEmployee(self):
      print "Name : ", self.name,  ", Salary: ", self.salary

"This would create first object of Employee class"
emp1 = Employee("Zara", 2000)
"This would create second object of Employee class"
emp2 = Employee("Manni", 5000)
emp1.displayEmployee()
emp2.displayEmployee()
print "Total Employee %d" % Employee.empCount
#%%
#Inheritance
class Parent:        # define parent class
   parentAttr = 100
   def __init__(self):
      print "Calling parent constructor"

   def parentMethod(self):
      print 'Calling parent method'

   def setAttr(self, attr):
      Parent.parentAttr = attr

   def getAttr(self):
      print "Parent attribute :", Parent.parentAttr

class Child(Parent): # define child class
   def __init__(self):
      print "Calling child constructor"

   def childMethod(self):
      print 'Calling child method'

c = Child()          # instance of child
c.childMethod()      # child calls its method
c.parentMethod()     # calls parent's method
c.setAttr(200)       # again call parent's method
c.getAttr()          # again call parent's method
#%% Introduction to Numpy
a = np.array([1, 2, 3])           # Create a rank 1 array
print type(a)                     # Prints "numpy.ndarray"
print a.shape                     # Prints "(3,)"
print a[0], a[1], a[2]            # Prints "1 2 3"
a[0] = 5                          # Change an element of the array
print a                           # Prints "[5, 2, 3]"
print (len(a))

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print b.shape                     # Prints "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   # Prints "1 2 4"

#Creating some standard Arrays
zero_matrix=np.zeros([3,3])
print(zero_matrix)

ones_matrix=np.ones([3,3])
print(ones_matrix)

constants_matrix= np.full([3,3],4)
print(constants_matrix)

identity_matrix= np.eye(3)
print(identity_matrix)

random_matrix=np.random.random([3,4])
print(random_matrix)

# Array Slicing Example
print(random_matrix[:,2:])
print(random_matrix[0:2,0:3])

a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(a[1][1])

# Boolean Array
b= a>5
print(b)

# Array Math
x=np.array([[1,2,3],[4,5,6]])
y=np.array([[4,5,6],[1,2,3]])

# Element wise operations
print(x+y)
#print np.add(x,y)

print(x-y)
#print np.subtract(x,y)

print(x*y)
#print np.multiply(x,y)

print(x/y)
#print np.divide(x,y)

print np.sqrt(x)

#Reshaping matrix
y=np.reshape(y,[3,2])
print y

#Matrix Multiplication
z=np.dot(x,y)
print z

# Sum of elements in matrix
print np.sum(x)          # Computes the sum of all the elements;
print np.sum(x, axis=0)  # Compute sum of each column; 
print np.sum(x, axis=1)  # Compute sum of each row; 

#Transpose of a matrix
print z.T

# Inverse of a matrix
print np.linalg.inv(z)

# Determinant of a matrix
print np.linalg.det(z)

#%% Pandas
# Declaring a pandas dataframe
df = pd.DataFrame({'example1' : [18, 24, 17, 21, 24, 16, 29, 18], 
'example2' : [75, 87, 49, 68, 75, 84, 98, 92],
 'example3' : [55, 47, 38, 66, 56, 64, 44, 39] })
print df

print df.mean()

print df['example2'].max()
print df.max()

print df.min()

print df.median()

print df.mode()

print df["example1"].var()

print df.std()

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
df['example3'].plot(kind='box')

print df.describe()

df.corr()

#%% Reading data
data = pd.read_csv("/Users/undergroundking8o8/Downloads/weather_year.csv")
data.columns
data['Max TemperatureF']
data.EDT
data.EDT.head()
data['EDT'].tail()
data['Mean TemperatureF'].hist()

data.columns = ["date", "max_temp", "mean_temp", "min_temp", "max_dew",
                "mean_dew", "min_dew", "max_humidity", "mean_humidity",
                "min_humidity", "max_pressure", "mean_pressure",
                "min_pressure", "max_visibilty", "mean_visibility",
                "min_visibility", "max_wind", "mean_wind", "min_wind",
                "precipitation", "cloud_cover", "events", "wind_dir"]

freezing_days = data[data.max_temp <= 32]

data.max_temp.tail().plot()
data.max_temp.tail().plot(kind="bar", rot=10)

ax = data.max_temp.plot(title="Min and Max Temperatures")
data.min_temp.plot(style="red", ax=ax)
ax.set_ylabel("Temperature (F)")

#Writing the data
data.to_csv("data/weather-mod.csv")
data.to_csv("data/weather-mod.tsv", sep="\t")