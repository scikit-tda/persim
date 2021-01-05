"""
Script to re-create results from Bubenik/Dlotko sphere experiment.

We will sample 100 points from the 2-sphere and the 3-sphere, construct and
compute VR PH (in degrees 1 and 2), construct landscapes. 
This entire process is repeated 1000 times. In each homological degree, we will
average over these 1000 runs to get a total of 4 average persistence landscapes,
indexed on the dimension of the sphere and degree of homology. 

Fix the degree of homology for a moment (e.g., homological_degree = 1).
We take the supremum norm of the difference of 
the avg PL in dim-1 for S^2 and the avg PL in dim-1 for S^3.
This sup norm is the true difference. We then shuffle the labels of `S^2` versus `S^3`
among these 2000 landscapes, recompute averages with respect to the new labelling,
and take the sup norm of the difference of these averages. If the sup norm of the 
modified shuffling is larger than the sup norm of the true difference, we consider
this shuffling significant and add one to `significant`. The p-value is then 
`significant` divided by the number of times we do the shuffling, which is 10,000 times.

This is a simplified version of the experiment from Bubenik Dlotko. We are
sampling directly from the spheres (with the metric from the ambient
Euclidean space) and not re-scaling based on distances or sampling from a Gaussian
distribution as done there. While this isn't necessarily difficult to implement
we just have yet to do it. Hence, the results from this script are biased and we only
use them to a) explore our full workflow, b) get a feel for the statistics of landscapes,
and c) debug our visualization methods.

UPDATE: Without parallelization, the above parameters simply take too long to run.
We'll drop everything by an order of magnitude or two.
"""
#%% Imports
import numpy as np
import random
import concurrent.futures
from operator import itemgetter

from ripser import ripser
from PersistenceLandscapeGrid import PersistenceLandscapeGrid, snap_PL
from visualization import plot_landscape_grid

from tadasets import dsphere
import time

#%% Construct a list of 100 landscapes from randomly sampled points.
sph2_dgm_list = []
sph3_dgm_list = []
sph2_pl1_list = []
sph2_pl2_list = []
sph3_pl1_list = []
sph3_pl2_list = []
#sph2_b = 100.
#sph2_d = -100.
#sph3_b = 100.
#sph3_d = -100.

#%%
for i in range(100):
    sph2_pts = dsphere(n=100, d=2, r=1)
    sph2_dgm = ripser(sph2_pts, maxdim=2)['dgms']
    #if min(sph2_dgm[1],key=itemgetter(0))[0] < sph2_b:
    #    sph2_b = min(sph2_dgm[1],key=itemgetter(0))[0]
    #if max(sph2_dgm[1],key=itemgetter(1))[1] > sph2_d:
    #    sph2_d = max(sph2_dgm[1],key=itemgetter(1))[1]
    sph2_dgm_list.append(sph2_dgm)
    
    sph3_pts = dsphere(n=100, d=3, r=1)
    sph3_dgm = ripser(sph3_pts, maxdim=2)['dgms']
    #if min(sph3_dgm[1],key=itemgetter(0))[0] < sph3_b:
    #    sph3_b = min(sph3_dgm[1],key=itemgetter(0))[0]
    #if max(sph3_dgm[1],key=itemgetter(1))[1] > sph3_d:
    #    sph3_d = max(sph3_dgm[1],key=itemgetter(1))[1]
    sph3_dgm_list.append(sph3_dgm)
#%%

start_time = time.perf_counter()
for i in range(100):
    sph2_pts = dsphere(n=100, d=2, r=1)
    sph2_dgm = ripser(sph2_pts, maxdim=2)['dgms']
    sph2_pl1 = PersistenceLandscapeGrid(num_dims=500,
                                        dgms=sph2_dgm, 
                                        hom_deg=1,compute=True)
    sph2_pl1_list.append(sph2_pl1)
    
    sph3_pts = dsphere(n=100, d=3, r=1)
    sph3_dgm = ripser(sph3_pts, maxdim=2)['dgms']
    sph3_pl1 = PersistenceLandscapeGrid(num_dims=500,
                                        dgms=sph3_dgm, hom_deg=1, compute=True)
    #sph3_pl1.compute_landscape()
    sph3_pl1_list.append(sph3_pl1)
    #sph3_pl2 = PersistenceLandscapeGrid(diagrams=sph3_dgm, homological_degree=2)
    #sph3_pl2.compute_landscape()
    #sph3_pl2_list.append(sph3_pl2)
    
end_time = time.perf_counter()
print(f'The time it took for non-parallelized landscapes is {end_time-start_time} s')
#%% snap them

sph2_pl1_list = snap_PL(sph2_pl1_list)
sph3_pl1_list = snap_PL(sph3_pl1_list)

#%% Construct the true average landscape
avg_sph2_pl1 = sph2_pl1_list[0]
avg_sph3_pl1 = sph3_pl1_list[0]

for i in range(1,100):
    avg_sph2_pl1 += sph2_pl1_list[i]
    avg_sph3_pl1 += sph3_pl1_list[i]

avg_sph2_pl1=avg_sph2_pl1/100.
avg_sph3_pl1=avg_sph3_pl1/100.

#true_diff_pl = avg_sph2_pl1 - avg_sph3_pl1
#true_diff = true_diff_pl.sup_norm()

#%% Plot them

plot_landscape_grid(avg_sph2_pl1, title='Average landscape in degree 1 for $S^2$')

#%% plot avg S^3
plot_landscape_grid(avg_sph3_pl1, title='Average landscape in degree 1 for $S^3$')
#%% plot diff
#plot_landscape_grid(true_diff_pl, title='Difference of average landscapes in degree 1 for $S^2$ and $S^3$')

#%% Run the permutation test
# We do stratified shuffling, so each new group will still have 1000 entries each. This
# is what is done in the Bubenik/Dlotko paper.
# This could be done with sklearn.model_selection.permutation_test_score, but
# we don't have the transformer implemented yet.

comb_pl_list = sph2_pl1_list + sph3_pl1_list
significant = 0

for run in range(100):
    A_indices = random.sample(range(200), 100)
    B_indices = [_ for _ in range(200) if _ not in A_indices]
    
    A_pl_list = [comb_pl_list[i] for i in A_indices]
    B_pl_list = [comb_pl_list[j] for j in B_indices]
    
    A_sum = A_pl_list[0]
    B_sum = B_pl_list[0]
    for i in range(99):
        A_sum += A_pl_list[i+1]
        B_sum += B_pl_list[i+1]
    
    A_avg = A_sum/100.
    B_avg = B_sum/100.
    
    AB_diff = A_avg - B_avg
    if (AB_diff.sup_norm() > true_diff): significant += 1


print(f'Significant is {significant}') # Significant = 0
#%% do multiprocessing
# Write a function which does the above

import multiprocessing as mp


def construct_random_landscape(_):
    sph2_pts = dsphere(n=100, d=2, r=1)
    sph2_dgm = ripser(sph2_pts, maxdim=2)['dgms']
    pl = PersistenceLandscapeGrid(start=_b,stop=_d,num_dims=500,
                                    diagrams=sph2_dgm,homological_degree=1)
    pl.compute_landscape()
    return pl

pool = mp.Pool(mp.cpu_count()-3)
start_time = time.perf_counter()
results = pool.map(construct_random_landscape, [i for i in range(500)])
pool.close()
end_time = time.perf_counter()
print(f'The time it took the parallelized version was {end_time-start_time} s')