# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:36:53 2020

@author: mikec
"""

#%% Imports
import numpy as np
import random
import concurrent.futures
import time

from ripser import ripser
from PersistenceLandscapeExact import PersistenceLandscapeExact
from PersistenceLandscapeGrid import PersistenceLandscapeGrid
from visualization import plot_landscape


from tadasets import dsphere
#%%
np.random.seed(42)
data = np.random.random_sample((1000,2))
diagrams = ripser(data)['dgms']

start_time = time.perf_counter()
sph2_ple = PersistenceLandscapeExact(diagrams=diagrams, homological_degree=1)
end_time = time.perf_counter()
print(f'Time to initialize the PLE class was {end_time-start_time} s')
start_time = time.perf_counter()
sph2_ple.compute_landscape()
end_time = time.perf_counter()
print(f'Time to compute landscape exact was {end_time-start_time} s')
# start_time = time.perf_counter()
# sph2_plg = PersistenceLandscapeGrid(start=0, stop = 1, num_dims=500, diagrams=diagrams, homological_degree=1)
# end_time = time.perf_counter()
# print(f'Time to initialize the PLG class was {end_time-start_time} s')
# start_time = time.perf_counter()
# sph2_plg.compute_landscape()
# end_time = time.perf_counter()
# print(f'Time to compute the PLG class was {end_time-start_time} s')

