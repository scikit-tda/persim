#!/usr/bin/python3

###############################################################
# Persistent Landscape Demo
# Demonstration of persistence landscape add-on
#
# Calder Sheagren
# University of Toronto
# calder.sheagren@sri.utoronto.ca
# Date: December 23, 2020 
###############################################################

import matplotlib.pyplot as plt

import tadasets
from ripser import Rips
import persim

sphere = tadasets.dsphere(n=20, d=4, ambient=12, noise=.02)
rips = Rips(maxdim=2)
dgms = rips.fit_transform(sphere)

plt.figure()
persim.plot_diagrams(dgms)

landscapes = persim.to_landscape(dgms)

persim.plot_landscape(landscapes)


