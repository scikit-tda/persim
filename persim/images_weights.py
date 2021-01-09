"""
Weight Functions for PersistenceImager() transformer:

A valid weight function is a Python function of the form 

weight(birth, persistence, **kwargs) 

defining a scalar-valued function over the birth-persistence plane, where birth and persistence are assumed to be numpy arrays of equal length. To ensure stability, functions should vanish continuously at the line persistence = 0.
"""
import numpy as np

def linear_ramp(birth, pers, low=0.0, high=1.0, start=0.0, end=1.0):
    """ Continuous peicewise linear ramp function which is constant below and above specified input values.
    
    Parameters
    ----------
    birth : (N,) numpy.ndarray
        Birth coordinates of a collection of persistence pairs.
    pers : (N,) numpy.ndarray
        Persistence coordinates of a collection of persistence pairs.
    low : float
        Minimum weight. 
    high : float
       Maximum weight.
    start : float
        Start persistence value of linear transition from low to high weight.
    end : float
        End persistence value of linear transition from low to high weight.
    
    Returns
    -------
    (N,) numpy.ndarray
        Weights at the persistence pairs.
    """
    try:
        n = len(birth)
    except:
        n = 1
        birth = [birth]
        pers = [pers]

    w = np.zeros((n,))
    for i in range(n):
        if pers[i] < start:
            w[i] = low
        elif pers[i] > end:
            w[i] = high
        else:
            w[i] = (pers[i] - start) * (high - low) / (end - start) + low

    return w

def persistence(birth, pers, n=1.0):
    """ Continuous monotonic function which weight a persistence pair (b,p) by p^n for some n > 0.
    
    Parameters
    ----------
    birth : (N,) numpy.ndarray
        Birth coordinates of a collection of persistence pairs.
    pers : (N,) numpy.ndarray
        Persistence coordinates of a collection of persistence pairs.
    n : positive float
        Exponent of persistence weighting function.
    
    Returns
    -------
    (N,) numpy.ndarray
        Weights at the persistence pairs.
    """
    return pers ** n