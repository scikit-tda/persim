#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    
    The persistent entropy has been defined in [1]. A precursor of this definition was given in [2]
    to measure how different bars of the barcode are in length.
    
    [1] M. Rucco, F. Castiglione, E. Merelli, M. Pettini, Characterisation of the
    idiotypic immune network through persistent entropy, in: Proc. Complex, 2015.
    [2] H. Chintakunta, T. Gentimis, R. Gonzalez-Diaz, M.-J. Jimenez,
    H. Krim, An entropy-based persistence barcode, Pattern Recognition
    48 (2) (2015) 391–401.
        
    Implementation of persistent entropy

    Author: Eduardo Paluzo Hidalgo (cimagroup, University of Seville)
    contact: epaluzo@us.es

"""

from __future__ import division
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
    dgms: ndarray (n_pairs, 2) or list of diagrams   
        array or list of arrays of birth/death pairs of a persistence barcode of a determined dimension.
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

    ps: ndarray (n_pairs,)
        array of persistent entropy values corresponding to each persistence barcode.

    """

    if isinstance(dgms, list) == False:
        dgms = [dgms]

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
        if all(l > 0):
            L = np.sum(l)
            p = l / L
            E = -np.sum(p * np.log(p))
            if normalize == True:
                E = E / np.log(len(l))
            ps.append(E)
        else:
            raise Exception("A bar is born after dying")

    return np.array(ps)
