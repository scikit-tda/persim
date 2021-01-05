"""
Define a base Persistence Landscape class
"""
from abc import ABC, abstractmethod
import numpy as np


class PersistenceLandscape(ABC):
    """
    The base Persistence Landscape class

    This is the base persistence landscape class and should not be
    called directly. The subclasses `PersistenceLandscapeGrid` or
    `PersistenceLandscapeExact` should instead be called.

    Parameters
    ----------
    dgms: list[list]
        A list of birth-death pairs in `hom_deg`
    """

    def __init__(self, dgms: list = [], hom_deg: int = 0) -> None:
        if not isinstance(hom_deg, int):
            raise TypeError("hom_deg must be an integer")
        if hom_deg < 0:
            raise ValueError('hom_deg must be positive')
        if not isinstance(dgms, (list, tuple, np.ndarray)):
            raise TypeError("dgms must be a list, tuple, or numpy array")
        self.hom_deg = hom_deg

    # We force landscapes to have arithmetic and norms, 
    # this is the whole reason for using them.
    
    @abstractmethod
    def p_norm(self, p: int = 2) -> float:
        pass

    @abstractmethod
    def sup_norm(self) -> float:
        pass
        
    @abstractmethod
    def __add__(self, other):
        if self.hom_deg != other.hom_deg:
            raise ValueError("Persistence landscapes must be of same homological degree")
    
    @abstractmethod
    def __neg__(self):
        pass
    
    @abstractmethod
    def __sub__(self, other):
        pass
    
    @abstractmethod
    def __mul__(self, other):
        if not isinstance(other, (int,float)):
            raise TypeError("Can only multiply persistence landscapes by real numbers")
            
    @abstractmethod
    def __truediv__(self, other):
        if other == 0.:
            raise ValueError("Cannot divide by zero")
        
