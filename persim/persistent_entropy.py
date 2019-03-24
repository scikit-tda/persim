"""
    
    
    Implementation of persistent entropy

    Author: Eduardo Paluzo Hidalgo (cimagroup, University of Seville)
    contact: epaluzo@us.es

"""

import numpy as np


__all__ = ["persistent_entropy"]


def persistent_entropy(
    dgms, keep_inf=False, val_inf=None, normalize=False
):
    """
    Perform the persistent entropy values of a family of persistence barcodes (or persistence diagrams).
    Assumes that the input diagrams are from a determined dimension. If the infinity bars have any meaning
    in your experiment and you want to keep them, remember to give the value you desire to val_Inf.

    Parameters
    -----------
    dgms:   
        array of arrays of birth/death pairs of a persistence barcode of a determined dimension.
    keep_inf: bool, default False
        if False, the infinity bars are removed.
        if True, the infinity bars remain.
    val_inf: float, default None
        substitution value to infinity.
    normalize: bool, default False
        if False, the persistent entropy values are not normalized.
        if True, the persistent entropy values are normalized.
          
    Returns
    --------

    ps: array of persistent entropy values corresponding to each persistence barcode.

    """
    # Step 1: Remove infinity bars if keep_inf = False. If keep_inf = True, infinity value is substituted by val_inf.

    if keep_inf == False:
        dgms = [(dgm[dgm[:, 1] != np.inf]) for dgm in dgms]
    if keep_inf == True:
        if val_inf != None:
            dgms = [
                np.where(dgm == np.inf, val_inf, dgm)
                for dgm in dgms
            ]
        else:
            raise Exception(
                "Remember: You need to provide a value to infinity bars if you want to keep them."
            )

    # Step 2: Persistent entropy computation.
    ps = []
    for dgm in dgms:
        l = dgm[:, 1] - dgm[:, 0]
        L = np.sum(l)
        p = l / L
        E = -np.sum(p * np.log(p))
        if normalize == True:
            E = E / np.log(len(l))
        ps.append(E)

    return np.array(ps)
