"""
    Implementation of scikit-learn transformers for persistence
    landscapes.
"""
from operator import itemgetter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .approximate import PersLandscapeApprox

__all__ = ["PersistenceLandscaper"]


class PersistenceLandscaper(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer for converting persistence diagrams into persistence landscapes.

    Parameters
    ----------
    hom_deg : int
        Homological degree of persistence landscape.

    start : float, optional
        Starting value of approximating grid.

    stop : float, optional
        Stopping value of approximating grid.

    num_steps : int, optional
        Number of steps of approximating grid.

    flatten : bool, optional
        Determines if the resulting values are flattened.


    Examples
    --------
    First instantiate the PersistenceLandscaper::

        >>> from persim import PersistenceLandscaper
        >>> pl = PersistenceLandscaper(hom_deg=0, num_steps=10, flatten=True)
        >>> print(pl)

        PersistenceLandscaper(hom_deg=1,num_steps=10)

    The `fit()` method is first called on a list of (-,2) numpy.ndarrays to determine the `start` and `stop` parameters of the approximating grid::

        >>> ex_dgms = [np.array([[0,3],[1,4]]),np.array([[1,4]])]
        >>> pl.fit(ex_dgms)

        PersistenceLandscaper(hom_deg=0, start=0, stop=4, num_steps=10)

    The `transform()` method will then compute the values of the landscape functions on the approximated grid. The `flatten` flag determines if the output should be a flattened numpy array::

        >>> ex_pl = pl.transform(ex_dgms)
        >>> ex_pl

        array([0.        , 0.44444444, 0.88888889, 1.33333333, 1.33333333,
       1.33333333, 1.33333333, 0.88888889, 0.44444444, 0.        ,
       0.        , 0.        , 0.        , 0.44444444, 0.88888889,
       0.88888889, 0.44444444, 0.        , 0.        , 0.        ])
    """

    def __init__(
        self,
        hom_deg: int = 0,
        start: float = None,
        stop: float = None,
        num_steps: int = 500,
        flatten: bool = False,
    ):
        self.hom_deg = hom_deg
        self.start = start
        self.stop = stop
        self.num_steps = num_steps
        self.flatten = flatten

    def __repr__(self):
        if self.start is None or self.stop is None:
            return f"PersistenceLandscaper(hom_deg={self.hom_deg}, num_steps={self.num_steps})"
        else:
            return f"PersistenceLandscaper(hom_deg={self.hom_deg}, start={self.start}, stop={self.stop}, num_steps={self.num_steps})"

    def fit(self, X: np.ndarray, y=None):
        """Find optimal `start` and `stop` parameters for approximating grid.

        Parameters
        ----------

        X : list of (-,2) numpy.ndarrays
            List of persistence diagrams.
        y : Ignored
            Ignored; included for sklearn compatibility.
        """
        # TODO: remove infinities
        _dgm = X[self.hom_deg]
        if self.start is None:
            self.start = min(_dgm, key=itemgetter(0))[0]
        if self.stop is None:
            self.stop = max(_dgm, key=itemgetter(1))[1]
        return self

    def transform(self, X: np.ndarray, y=None):
        """Construct persistence landscape values.

        Parameters
        ----------

        X : list of (-,2) numpy.ndarrays
            List of persistence diagrams
        y : Ignored
            Ignored; included for sklearn compatibility.

        Returns
        -------

        numpy.ndarray
            Persistence Landscape values sampled on approximating grid.
        """
        result = PersLandscapeApprox(
            dgms=X,
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
        )
        if self.flatten:
            return (result.values).flatten()
        else:
            return result.values
