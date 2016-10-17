#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 19:18:35 2016

@author: Ray
"""

#%% 
#LIST
family_members=['Sally','Rey','Raymond','Ralph','Ryan','Raisa']
print(family_members)

#2
print family_members[5]

#3
print len(family_members[0])

#4
family_members[2] = 'Ray'

#5
family_members.insert(6,'Tron')

#6
print family_members[6].lower()

#7
sorted(family_members, key=str.lower, reverse=True)
print family_members

#%%
