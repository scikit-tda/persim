"""
    Implementation of the "multiscale heat kernel" (CVPR 2015), 

    Author: Chris Tralie

"""

import numpy as np

__all__ = ["heat"]

def evalHeatKernel(dgm1, dgm2, sigma):
    """
    Evaluate the continuous heat-based kernel between dgm1 and dgm2 (more correct than L2 on the discretized version above but may be slower because can't exploit fast matrix multiplication when evaluating many, many kernels)
    """
    kSigma = 0
    I1 = np.array(dgm1)
    I2 = np.array(dgm2)
    for i in range(I1.shape[0]):
        p = I1[i, 0:2]
        for j in range(I2.shape[0]):
            q = I2[j, 0:2]
            qc = I2[j, 1::-1]
            kSigma += np.exp(-(np.sum((p - q) ** 2)) / (8 * sigma)) - np.exp(
                -(np.sum((p - qc) ** 2)) / (8 * sigma)
            )
    return kSigma / (8 * np.pi * sigma)


def heat(dgm1, dgm2, sigma=0.4):
    """
    Return the pseudo-metric between two diagrams based on the continuous
    heat kernel as described in "A Stable Multi-Scale Kernel for Topological Machine Learning" by Jan Reininghaus, Stefan Huber, Ulrich Bauer, and Roland Kwitt (CVPR 2015)

    Parameters
    -----------

    dgm1: np.array (m,2)
        A persistence diagram
    dgm2: np.array (n,2)
        A persistence diagram
    sigma: float
        Heat diffusion parameter (larger sigma makes blurrier)
    Returns
    --------

    dist: float
        heat kernel distance between dgm1 and dgm2

    """
    return np.sqrt(
        evalHeatKernel(dgm1, dgm1, sigma)
        + evalHeatKernel(dgm2, dgm2, sigma)
        - 2 * evalHeatKernel(dgm1, dgm2, sigma)
    )
