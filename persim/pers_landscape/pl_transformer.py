"""
    Implementation of scikit-learn transformers for persistence
    landscapes.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from PersistenceLandscapeExact import PersistenceLandscapeExact


# To get the following functionality to work, I think we need 
# to split off the compute landscape function from within the 
# PL class.

class PL_exact(BaseEstimator, TransformerMixin):
    """ A scikit-learn transformer class for exact persistence landscapes. The transform
    method returns the list of critical pairs for the landscape. For a vectorized
    encoding of the landscape, using the PL_grid transformer.
    """
    def __init__(self,homological_degree:int = 0):
        self.homological_degree = homological_degree
        
    def fit(self,X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Add check that X is the output of a PH calculation.
        # return X[self.homological_degree].compute_landscape()
        result = PersistenceLandscapeExact(diagrams=X, homological_degree=self.homological_degree)
        result.compute_landscape()
        return result.critical_pairs
    
class PL_grid(BaseEstimator, TransformerMixin):
    """ A scikit-learn transformer for grid persistence landscapes.
    """
    def __init__(self,homological_degree:int = 0):
        self.homological_degree = homological_degree
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X[self.homological_degree].compute_landscape()
