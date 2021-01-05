#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:33:52 2020

@author: gabrielleangeloro
"""
from PersistenceLandscapeGrid import PersistenceLandscapeGrid
import numpy as np

dgmP = [np.array([[2,6],[4,10]])]
 
P1 = PersistenceLandscapeGrid(0, 10, 11, dgmP, 0)
P1.compute_landscape()

print(f'P1 is {P1.funct_values} ')

#%%

P2 = PersistenceLandscapeGrid(0, 10, 6, dgmP, 0)
P2.compute_landscape()
print(f'P2 is {P2.funct_values} ')

#%%

P3 = PersistenceLandscapeGrid(0, 10, 21, dgmP, 0)
P3.compute_landscape()
print(f'P3 is {P3.funct_values} ')

#%%

dgmQ = [np.array([[2,6],[2,6],[4,10]])]

Q = PersistenceLandscapeGrid(0, 10, 11, dgmQ, 0)
Q.compute_landscape()
print(f'Q is {Q.funct_values} ')

#%%