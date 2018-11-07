"""
    Very basic tests of the plotting functionality
"""

import numpy as np

import persim
import persim.plot

def test_bottleneck_matching():
    dgm1 = np.array([
        [0.1, 0.2],
        [0.2, 0.4]
    ])
    dgm2 = np.array([
        [0.1, 0.2],
        [0.3, 0.45]
    ])

    d, (matching, D) = persim.bottleneck(dgm1, dgm2, matching=True)
    persim.plot.bottleneck_matching(dgm1, dgm2, matching, D)

def test_plot_labels():
    dgm1 = np.array([
        [0.1, 0.2],
        [0.2, 0.4]
    ])
    dgm2 = np.array([
        [0.1, 0.2],
        [0.3, 0.45]
    ])

    d, (matching, D) = persim.bottleneck(dgm1, dgm2, matching=True)
    persim.plot.bottleneck_matching(dgm1, dgm2, matching, D, labels=["X", "Y"])