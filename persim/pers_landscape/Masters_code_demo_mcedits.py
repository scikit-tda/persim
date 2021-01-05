#!/usr/bin/env python
# coding: utf-8

# ## Experiemnt: $S^2$ vs. $S^3$ 

# This notebook executes an experiemnt to see if persistence landscapes can tell the difference between a sphere in dimension 2 and 3.

# In[1]:

# import miscellaneous tools
import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer

# import data sets
from tadasets import dsphere
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

# import Ripser
from ripser import ripser
from persim import plot_diagrams

# import PyLandscapes and nessasary functions
from PersistenceLandscapeGrid import PersistenceLandscapeGrid, snap_PL, average_grid
from PersistenceLandscapeExact import PersistenceLandscapeExact
from auxiliary import linear_combination
from visualization import plot_landscape


#  Sample 100 points from $S^2$ and $S^3$, 100 times and normalize

# In[2]:


# sph2: list of 100 runs of sampling S2
sph2 = []
for i in range(100):
    sph2.append( preprocessing.scale(dsphere(n=100, d=2, r=1)) ) #preprocessing.scale to normalize samples

# sph3: list of 100 runs of sampling S3
sph3 = []
for i in range(100):
    sph3.append( preprocessing.scale(dsphere(n=100, d=3, r=1)) )


# Compute persistence diagram for $S^2$ and $S^3$

# In[3]:


#sph2_dgm: list of 100 diagrams for 100 sampled points on S2
sph2_dgm = [ripser(sphere, maxdim=2)['dgms'] for sphere in sph2]


#sph3_dgm: list of 100 diagrams for 100 sampled points on S3
sph3_dgm = [ripser(sphere, maxdim=2)['dgms']for sphere in sph3]


# Compute persistence landscape for $S^2$ and $S^3$

# In[4]:


sph2_PL1 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=0) for diagram in sph2_dgm]

for x in sph2_PL1:
    x.compute_landscape(verbose=True)

# In[5]: 
z = average_grid(sph2_PL1)


# In[11]: COMMENTED HERE


#sph2_PL1, sph2_PL2: list of 100 landscapes for 100 sampled points on S3 in degree 1 and 2 
sph2_PL1 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=1, compute=True) for diagram in sph2_dgm]
sph2_PL2 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=2, compute=True) for diagram in sph2_dgm]
#%%
#sph3_PL1, sph3_PL2: list of 100 landscapes for 100 sampled points on S3 in degree 1 and 2 
sph3_PL1 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=1, compute=True) for diagram in sph3_dgm]

#%%
sph3_PL2 = [PersistenceLandscapeGrid(dgms=diagram , hom_deg=2, compute=True) for diagram in sph3_dgm]


# Average the 100 landscapes for $S^2$ and $S^3$ in dimension 1 and 2

# In[31]:


#avg2_hom1, avg2_hom2: average landscape for the 100 samples of S2 in degree 1 and 2 
avg2_hom1 = linear_combination(sph2_PL1,100*[1/100])
avg2_hom2 = linear_combination(sph2_PL2,100*[1/100])

#avg3_hom1, avg3_hom2: average landscape for the 100 samples of S3 in degree 1 and 2 
avg3_hom1 = linear_combination(sph3_PL1,100*[1/100])
avg3_hom2 = linear_combination(sph3_PL2,100*[1/100])


# # Compute the difference in sup norms between the average landscape of $S^2$ and $S^3$ in dimension 1 and 2

# # In[32]:


# #diff_hom1, diff_hom2: difference between average landscapes in degree 1 and 2 of S2 and S3
# true_diff_hom1 = (avg2_hom1 - avg3_hom1).sup_norm()
# true_diff_hom2 = (avg2_hom2 - avg3_hom2).sup_norm()


# # Plot average landscape in degree 1 for S2 and S3 and difference between them 

# # In[33]:


# # plot avg S^2
# plot_landscape(avg2_hom1) 


# # In[ ]:


# # plot avg S^3
# plot_landscape(avg3_hom1) 


# # In[ ]:


# # plot diff
# plot_landscape(true_diff_hom_1)


# # ### Run permutation test for homological degree 1

# # In[ ]:


# #PL1: persistence landscapes in degree 1 from S2 and S3
# PL1 = []
# PL1.extend(sph2_PL1)
# PL1.extend(sph3_PL1)
# PL1 = np.array(PL1) #cast as array in order to index with a list


# for run in range(100):
#     # shuffle labels for 200 landscapes
#     A_indices = random.sample(range(100), 50)
#     B_indices = [_ for _ in range(100) if _ not in A_indices]
#     A_PL1 = PL1[A_indices]
#     B_PL1 = PL1[B_indices]
    
#     # take average of landscape with label A and label B resp.
#     avg_A_PL1 = linear_combination(A_PL1,100*[1/100])
#     avg_B_PL1 = linear_combination(B_PL1,100*[1/100])
    
#     shuffled_diff_hom1 = (avg_A_PL1 - avg_B_PL1).sup_norm() #compute shuffled diff
    
#     # count differences more extreme than true diff
#     more_extreme = 0
#     if np.abs(shuffled_diff_hom1) > np.abs(true_diff_hom1):
#         more_extreme += 1

# print(f'{more_extreme} of the relabeled persistence landscapes'
#       'had difference more extreme than the true differnce')


# # ### Run permutation test for homological degree 2

# # In[ ]:


# #PL2: persistence landscapes in degree 1 from S2 and S3
# PL2 = []
# PL2.extend(sph2_PL2)
# PL2.extend(sph3_PL2)
# PL2 = np.array(PL2) #cast as array in order to index with a list


# for run in range(100):
#     # shuffle labels for 200 landscapes
#     A_indices = random.sample(range(100), 50)
#     B_indices = [_ for _ in range(100) if _ not in A_indices]
#     A_PL2 = PL2[A_indices]
#     B_PL2 = PL2[B_indices]
    
#     # take average of landscape with label A and label B resp.
#     avg_A_PL2 = linear_combination(A_PL2,100*[1/100])
#     avg_B_PL2 = linear_combination(B_PL2,100*[1/100])
    
#     shuffled_diff_hom2 = (avg_A_PL2 - avg_B_PL2).sup_norm() #compute shuffled diff
    
#     # count differences more extreme than true diff
#     more_extreme = 0
#     if np.abs(shuffled_diff_hom2) > np.abs(true_diff_hom2):
#         more_extreme += 1

# print(f'{more_extreme} of the relabeled persistence landscapes'
#       'had difference more extreme than the true differnce')


# # ### For homological degree 1 and 2 there was no relabeling that resulted in persistence landscape difference that was more extreme than that of $S^2$ and $S^3$. So we conclude that the difference between $S^2$ and $S^3$ detected by persistence landscapes was significant.

# # 
