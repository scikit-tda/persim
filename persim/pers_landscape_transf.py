"""
    Implementation of scikit-learn transformers for persistence
    landscapes.

    authors: Gabrielle Angeloro, Michael Catanzaro
"""
from operator import itemgetter
from sklearn.base import BaseEstimator, TransformerMixin
from .pers_landscape_exact import PersLandscapeExact
from .pers_landscape_approx import PersLandscapeApprox


__all__ = ["PLE", "PLA"]


class PLE(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer class for exact persistence landscapes. The transform
    method returns the list of critical pairs for the landscape. For a vectorized
    encoding of the landscape, using the PL_grid transformer.
    """

    def __init__(self, hom_deg: int = 0):
        self.hom_deg = hom_deg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Add check that X is the output of a PH calculation.
        # return X[self.homological_degree].compute_landscape()
        result = PersLandscapeExact(dgms=X, hom_deg=self.hom_deg)
        return result.critical_pairs


class PLA(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer for grid persistence landscapes."""

    def __init__(
        self,
        hom_deg: int = 0,
        start: float = None,
        stop: float = None,
        num_steps: int = 500,
    ):
        self.hom_deg = hom_deg
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.start is None:
            _start = min(X, key=itemgetter(0))[0]
        else:
            _start = self.start
        if self.stop is None:
            _stop = max(X, key=itemgetter(1))[1]
        else:
            _stop = self.stop
        result = PersLandscapeApprox(
            diagrams=X,
            start=_start,
            stop=_stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
        )
        return result.values
