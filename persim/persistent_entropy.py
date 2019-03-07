"""

    Implementation of persistent entropy

    Author: cimagroup (University of Seville) 

"""

import numpy as np


__all__ = ["persistent_entropy"]

def pentropy(dgms,inf = 0,valInf = 100 , norm = 0):
    """
    Perform the persistent entropy values of a family of persistence barcodes (or persistence diagrams).
    Assumes that the input diagrams are from a determined dimension. If the infinity bars have any meaning
    in your experiment and you want to keep them, remember to give the value you desire to valInf.

    Parameters
    -----------
    dgms: array of arrays of birth/death pairs of a persistence barcode of a determined dimension.
    inf: if 0 the infinity bars are removed.
         if 1 the infinity bars remain.
    valInf: substitution value to infinity.
    norm: if 0 the persistence barcodes are not normalized.
          if 1 the persistence barcodes are normalized between 0 and 1.
    
    Returns
    --------

    ps: array of persistent entropy values corresponding for each persistence barcode.

    """
    # Step 1: Remove infinity bars if inf = 1. If inf = 0, infinity value is substituted by valInf.

    if inf == 0:
        dgms = [(dgm[dgm[:,1] !=np.inf]) for dgm in dgms]
    else:
        dgms = np.where(dgms==np.inf,valInf,dgms)
            
    # Step 2: Normalization of the persistence barcodes if norm = 1

    if norm == 1:
        dgms = (dgms-np.min(dgms))/(np.max(dgms)-np.min(dgms))
        
    # Step 3: Persistent entropy computation.
    ps = []
    for dgm in dgms:
        l = dgm[:,1]-dgm[:,0]
        L = np.sum(l)
        p = l/L
        E = -np.sum(p*np.log(p))
        ps.append(E)
    return np.array(ps)
