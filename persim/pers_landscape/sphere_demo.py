#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:55:33 2020

@author: gabrielleangeloro
"""
import numpy as np
import random

from tadasets import dsphere

from ripser import ripser
from PersistenceLandscapeGrid import PersistenceLandscapeGrid, lc_grid
from auxiliary import linear_combination
from visualization import plot_landscape

from sklearn import preprocessing
#%%

# sph2: list of 100 runs of sampling S2
sph2 = []
for i in range(100):
    sph2.append( preprocessing.scale(dsphere(n=100, d=2, r=1)) ) #preprocessing.scale to normalize samples

# sph3: list of 100 runs of sampling S3
sph3 = []
for i in range(100):
    sph3.append( preprocessing.scale(dsphere(n=100, d=3, r=1)) )

#%%

#sph2_dgm: list of 100 diagrams for 100 sampled points on S2
sph2_dgm = [ripser(sphere, maxdim=2)['dgms'] for sphere in sph2]


#sph3_dgm: list of 100 diagrams for 100 sampled points on S3
sph3_dgm = [ripser(sphere, maxdim=2)['dgms']for sphere in sph3]
#%%

#sph2_PL1, sph2_PL2: list of 100 landscapes for 100 sampled points on S3 in degree 1 and 2 
sph2_PL1 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=0, compute=True) 
            for diagram in sph2_dgm]
#%%
sph2_PL1 = [PersistenceLandscapeGrid(start = 0, stop = 8, num_dims=500,dgms=diagram , hom_deg=0, compute=True) 
            for diagram in sph2_dgm]

#%%
sph2_PL2 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=0, compute=True) 
            for diagram in sph2_dgm]
#%%

#sph3_PL1, sph3_PL2: list of 100 landscapes for 100 sampled points on S3 in degree 1 and 2 
sph3_PL1 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=0, compute=True) 
            for diagram in sph3_dgm]
#%%
sph3_PL2 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=0, compute=True) 
            for diagram in sph3_dgm]







