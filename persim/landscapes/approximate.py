"""
    Persistence Landscape Approximate class
"""

import numpy as np
from operator import itemgetter
from .base import PersLandscape
from .auxiliary import union_vals, ndsnap_regular, _p_norm

__all__ = ["PersLandscapeApprox"]


class PersLandscapeApprox(PersLandscape):
    """
    Persistence Landscape Approximate class.

    This class implements an approximate version of Persistence Landscape,
    given by sampling the landscape functions on a grid. This version is only
    an approximation to the true landscape, but given a fine enough grid, this
    should suffice for most applications. If an exact calculation with no
    approximation is desired, consider `PersLandscapeExact`. Computations with this
    class are much faster compared to `PersLandscapeExact` in general.

    The optional parameters `start`, `stop`, `num_steps` determine the approximating
    grid that the values are sampled on. If both `dgms` and `values` are passed but `start`
    and `stop` are not, `start` and `stop` will be determined by `dgms`.


    Parameters
    ----------
    start : float, optional
        The start parameter of the approximating grid.

    stop : float, optional
        The stop parameter of the approximating grid.

    num_steps : int, optional
        The number of dimensions of the approximation, equivalently the
        number of steps in the grid.

    dgms : list of (-,2) numpy.ndarrays, optional
        Nested list of numpy arrays, e.g., [array( array([:]), array([:]) ),..., array([:])].
        Each entry in the list corresponds to a single homological degree.
        Each array represents the birth-death pairs in that homological degree. This is
        precisely the output format from ripser.py: ripser(data_user)['dgms'].

    hom_deg : int
        Represents the homology degree of the persistence diagram.

    values : numpy.ndarray, optional
        The approximated values of the landscapes, indexed by depth.

    compute : bool, optional
        Flag determining whether landscape functions are computed upon instantiation.

    Examples
    --------
    Define a persistence diagram and instantiate the landscape::

        >>> from persim import PersLandscapeApprox
        >>> import numpy as np
        >>> pd = [ np.array([[0,3],[1,4]]), np.array([[1,4]]) ]
        >>> pla = PersLandscapeApprox(dgms=pd, hom_deg=0); pla

        Approximate persistence landscape in homological degree 0 on grid from 0 to 4 with 500 steps

    The `start` and `stop` parameters will be determined to be as tight as possible from `dgms` if they are not passed. They can be passed directly if desired::

        >>> wide_pl = PersLandscapeApprox(dgms=pd, hom_deg=0, start=-1, stop=3.1415, num_steps=1000); wide_pl

        Approximate persistence landscape in homological degree 0 on grid from -1 to 3.1415 with 1000 steps

    The approximating values are stored in the `values` attribute::

        >>> wide_pl.values

        array([[0.        , 0.        , 0.        , ..., 0.00829129, 0.00414565,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ]])

    Arithmetic methods are implemented for approximate landscapes, so they can be summed, subtracted, and admit scalar multiplication.

    .. note:: Landscapes must have the same grid parameters (`start`, `stop`, and `num_steps`) before any arithmetic is involved. See the `snap_PL` function for a method that will snap a list of landscapes onto a common grid.

        >>> pla - wide_pl

        ValueError: Start values of grids do not coincide

        >>> from persim import snap_pl
        >>> [snapped_pla, snapped_wide_pl] = snap_pl([pla,wide_pl])
        >>> print(snapped_pla, snapped_wide_pl)

        Approximate persistence landscape in homological degree 0 on grid from -1 to 4 with 1000 steps Approximate persistence landscape in homological degree 0 on grid from -1 to 4 with 1000 steps

        >>> sum_pl = snapped_pla + snapped_wide_pl; sum_pl.values
        array([[0.        , 0.        , 0.        , ..., 0.01001001, 0.00500501,
        0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ]])

    Approximate landscapes are sliced by depth and slicing returns the approximated values in those depths::

        >>> sum_pl[0]

        array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
               1.50150150e-02, 1.00100100e-02, 5.00500501e-03, 0.00000000e+00])

    Norms are implemented for all `p` including the supremum norm::

        >>> sum_pl.p_norm(p=5)

        12.44665414332285

        >>> sum_pl.sup_norm()

        2.9958943943943943

    """

    def __init__(
        self,
        start: float = None,
        stop: float = None,
        num_steps: int = 500,
        dgms: list = [],
        hom_deg: int = 0,
        values=np.array([]),
        compute: bool = True,
    ) -> None:
        super().__init__(dgms=dgms, hom_deg=hom_deg)
        if not dgms and values.size == 0:
            raise ValueError("dgms and values cannot both be emtpy")
        if dgms:  # diagrams are passed
            self.dgms = dgms[self.hom_deg]
            # remove infity values
            self.dgms = self.dgms[~np.any(self.dgms == np.inf, axis=1)]
            # calculate start and stop
            if start is None:
                start = min(self.dgms, key=itemgetter(0))[0]
            if stop is None:
                stop = max(self.dgms, key=itemgetter(1))[1]
        elif values.size > 0:  # values passed, diagrams weren't
            self.dgms = dgms
            if start is None:
                raise ValueError("start parameter must be passed if values are passed")
            if stop is None:
                raise ValueError("stop parameter must be passed if values are passed")
            if start > stop:
                raise ValueError("start must be less than or equal to stop")
        self.start = start
        self.stop = stop
        self.values = values
        self.max_depth = len(self.values)
        self.num_steps = num_steps
        if compute:
            self.compute_landscape()

    def __repr__(self) -> str:
        return (
            "Approximate persistence landscape in homological "
            f"degree {self.hom_deg} on grid from {self.start} to {self.stop}"
            f" with {self.num_steps} steps"
        )

    def compute_landscape(self, verbose: bool = False) -> list:
        """Computes the approximate landscape values

        Parameters
        ----------
        verbose : bool, optional
            If true, progress messages are printed during computation.

        """

        verboseprint = print if verbose else lambda *a, **k: None

        if self.values.size:
            verboseprint("values was stored, exiting")
            return
        verboseprint("values was empty, computing values")
        # make grid
        grid_values, step = np.linspace(
            self.start, self.stop, self.num_steps, retstep=True
        )
        # grid_values = list(grid_values)
        # grid = np.array([[x,y] for x in grid_values for y in grid_values])
        bd_pairs = self.dgms

        # snap birth-death pairs to grid
        bd_pairs_grid = ndsnap_regular(bd_pairs, *(grid_values, grid_values))

        # make grid dictionary
        index = list(range(self.num_steps))
        dict_grid = dict(zip(grid_values, index))

        # initialze W to a list of 2m + 1 empty lists
        W = [[] for _ in range(self.num_steps)]

        # for each birth death pair
        for ind_in_bd_pairs, bd in enumerate(bd_pairs_grid):
            [b, d] = bd
            ind_in_Wb = dict_grid[b]  # index in W
            ind_in_Wd = dict_grid[d]  # index in W
            mid_pt = (
                ind_in_Wb + (ind_in_Wd - ind_in_Wb) // 2
            )  # index half way between, rounded down

            # step through by x value
            j = 0
            # j in (b, b+d/2]
            for _ in range(ind_in_Wb, mid_pt):
                j += 1
                # j*step: adding points from a line with slope 1
                W[ind_in_Wb + j].append(j * step)
            j = 0
            # j in (b+d/2, d)
            for _ in range(mid_pt + 1, ind_in_Wd):
                j += 1
                W[ind_in_Wd - j].append(j * step)
        # sort each list in W
        for i in range(len(W)):
            W[i] = sorted(W[i], reverse=True)
        # calculate k: max length of lists in W
        K = max([len(_) for _ in W])

        # initialize L to be a zeros matrix of size K x (2m+1)
        L = np.array([np.zeros(self.num_steps) for _ in range(K)])

        # input Values from W to L
        for i in range(self.num_steps):
            for k in range(len(W[i])):
                L[k][i] = W[i][k]
        # check if L is empty
        if not L.size:
            L = np.array(["empty"])
            print("Bad choice of grid, values is empty")
        self.values = L
        self.max_depth = len(L)
        return

    def values_to_pairs(self):
        """Converts function values to ordered pairs and returns them"""
        self.compute_landscape()
        grid_values = list(np.linspace(self.start, self.stop, self.num_steps))
        result = []
        for vals in self.values:
            pairs = list(zip(grid_values, vals))
            result.append(pairs)
        return np.array(result)

    def __add__(self, other):
        """Computes the sum of two approximate persistence landscapes

        Parameters
        ----------
        other : PersLandscapeApprox
            The other summand.
        """
        super().__add__(other)
        if self.start != other.start:
            raise ValueError("Start values of grids do not coincide")
        if self.stop != other.stop:
            raise ValueError("Stop values of grids do not coincide")
        if self.num_steps != other.num_steps:
            raise ValueError("Number of steps of grids do not coincide")
        self_pad, other_pad = union_vals(self.values, other.values)
        return PersLandscapeApprox(
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
            values=self_pad + other_pad,
        )

    def __neg__(self):
        """Negates an approximate persistence landscape"""
        return PersLandscapeApprox(
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
            values=np.array([-1 * depth_array for depth_array in self.values]),
        )
        pass

    def __sub__(self, other):
        """Computes the difference of two approximate persistence landscapes

        Parameters
        ----------
        other : PersLandscapeApprox
            The landscape to be subtracted.
        """
        return self + -other

    def __mul__(self, other: float):
        """Multiplies an approximate persistence landscape by a real scalar

        Parameters
        ----------
        other : float
            The real scalar to be multiplied.
        """
        super().__mul__(other)
        return PersLandscapeApprox(
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
            values=np.array([other * depth_array for depth_array in self.values]),
        )

    def __rmul__(self, other: float):
        """Multiplies an approximate persistence landscape by a real scalar

        Parameters
        ----------
        other : float
            The real scalar factor.
        """
        return self.__mul__(other)

    def __truediv__(self, other: float):
        """Divides an approximate persistence landscape by a non-zero real scalar

        Parameters
        ----------
        other : float
            The non-zero real scalar divisor.
        """
        super().__truediv__(other)
        return (1.0 / other) * self

    def __getitem__(self, key: slice) -> list:
        """
        Returns a list of values corresponding in range specified by
        depth

        Parameters
        ----------
        key : slice object

        Returns
        -------
        list
            The values of the landscape function corresponding
        to depths given by key
        """
        self.compute_landscape()
        return self.values[key]

    def p_norm(self, p: int = 2) -> float:
        """
        Returns the L_{`p`} norm of an approximate persistence landscape

        Parameters
        ----------
        p: float, default 2
            value p of the L_{`p`} norm
        """
        super().p_norm(p=p)
        return _p_norm(p=p, critical_pairs=self.values_to_pairs())

    def sup_norm(self) -> float:
        """
        Returns the supremum norm of an approximate persistence landscape

        """
        return np.max(np.abs(self.values))
