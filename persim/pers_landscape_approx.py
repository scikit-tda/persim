"""
    Persistence Landscape Approximate class

    authors: Gabrielle Angeloro, Michael Catanzaro
"""

from __future__ import annotations
import numpy as np
from operator import itemgetter, attrgetter
from .pers_landscape import PersLandscape
from .pers_landscape_aux import union_vals, ndsnap_regular

__all__ = ["PersLandscapeApprox", "snap_PL", "lc_approx", "average_approx"]


class PersLandscapeApprox(PersLandscape):
    """
    Persistence Landscape Approximate class.

    This class implements an approximate version of Persistence Landscape,
    given by sampling the landscape functions on a grid. This version is only
    an approximation to the true landscape, but given a fine enough grid, this
    should suffice for most applications. If an exact calculation with no
    approximation is desired, consider `PersLandscapeExact`.

    The default parameters for start and stop favor dgms over values. That
    is, if both dgms and values are passed but start and stop are not, the
    start and stop values will be determined by dgms.

    Parameters
    ----------
    start : float, optional
        The start parameter of the approximating grid.

    stop : float, optional
        The stop parameter of the approximating grid.

    num_steps : int, optional
        The number of dimensions of the approximation, equivalently the
        number of steps in the grid.

    dgms : list[list]
        A list lists of birth-death pairs for each homological degree.

    hom_deg : int
        represents the homology degree of the persistence diagram.

    vales

    Methods
    -------


    Examples
    --------


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
                raise ValueError("start parameter must be passed if values are passed.")
            if stop is None:
                raise ValueError("stop parameter must be passed if values are passed.")
        self.start = start
        self.stop = stop
        self.values = values
        self.num_steps = num_steps
        if compute:
            self.compute_landscape()

    def __repr__(self) -> str:
        return (
            "The persistence landscape in homological "
            f"degree {self.hom_deg} on grid from {self.start} to {self.stop}"
            f" with {self.num_steps} steps"
        )

    def compute_landscape(self, verbose: bool = False) -> list:

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
        return

    def values_to_pairs(self):
        """Converts function values to ordered pairs and returns them."""
        self.compute_landscape()
        grid_values = list(np.linspace(self.start, self.stop, self.num_steps))
        result = []
        for vals in self.values:
            pairs = list(zip(grid_values, vals))
            result.append(pairs)
        return np.array(result)

    def __add__(self, other: PersLandscapeApprox) -> PersLandscapeApprox:
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

    def __neg__(self) -> PersLandscapeApprox:
        return PersLandscapeApprox(
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
            values=np.array([-1 * depth_array for depth_array in self.values]),
        )
        pass

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other: float) -> PersLandscapeApprox:
        super().__mul__(other)
        return PersLandscapeApprox(
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
            values=np.array([other * depth_array for depth_array in self.values]),
        )

    def __rmul__(self, other: float) -> PersLandscapeApprox:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> PersLandscapeApprox:
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
        return np.sum([np.linalg.norm(depth, p) for depth in self.values])

    def sup_norm(self) -> float:
        return np.max(np.abs(self.values))


############################################
# End PersLandscapeApprox class definition #
############################################


def snap_PL(
    pls: list, start: float = None, stop: float = None, num_steps: int = None
) -> list:
    """Snap a list of PersLandscapeApprox tpes to a common grid

    Given a list `l` of PersLandscapeApprox types, convert them to a list
    where each entry has the same start, stop, and num_steps. This puts each
    entry of `l` on the same grid, so they can be added, averaged, etc.
    This assumes they're all of the same homological degree.

    If the user
    does not specify the grid parameters, they are computed as tightly as
    possible from the input list `l`.
    """
    if start is None:
        start = min(pls, key=attrgetter("start")).start
    if stop is None:
        stop = max(pls, key=attrgetter("stop")).stop
    if num_steps is None:
        num_steps = max(pls, key=attrgetter("num_steps")).num_steps
    grid = np.linspace(start, stop, num_steps)
    k = []
    for pl in pls:
        snapped_landscape = []
        for funct in pl:
            # snap each function and store
            snapped_landscape.append(
                np.array(
                    np.interp(grid, np.linspace(pl.start, pl.stop, pl.num_steps), funct)
                )
            )
        # store snapped persistence landscape
        k.append(
            PersLandscapeApprox(
                start=start,
                stop=stop,
                num_steps=num_steps,
                values=np.array(snapped_landscape),
                hom_deg=pl.hom_deg,
            )
        )
    return k


def lc_approx(
    landscapes: list,
    coeffs: list,
    start: float = None,
    stop: float = None,
    num_steps: int = None,
) -> PersLandscapeApprox:
    """Compute the linear combination of a list of PersLandscapeApprox objects.

    This uses vectorized arithmetic from numpy, so it should be faster and
    more memory efficient than a naive for-loop.

    Parameters
    -------
    landscapes: list
        a list of PersLandscapeApprox objects

    coeffs: list
        a list of the coefficients defining the linear combination

    start: float
        starting value for the common grid for PersLandscapeApprox objects
    in `landscapes`

    stop: float
        last value in the common grid for PersLandscapeApprox objects
    in `landscapes`

    num_steps: int
        number of steps on the common grid for PersLandscapeApprox objects
    in `landscapes`

    Returns
    -------
    PersLandscapeApprox:
        The specified linear combination of PersLandscapeApprox objects
    in `landscapes`

    """
    pl = snap_PL(landscapes, start=start, stop=stop, num_steps=num_steps)
    return np.sum(np.array(coeffs) * np.array(pl))


def average_approx(
    landscapes: list, start: float = None, stop: float = None, num_steps: int = None
) -> PersLandscapeApprox:
    """Compute the average of a list of PersLandscapeApprox objects.

    Parameters
    -------
    landscapes: list
        a list of PersLandscapeApprox objects

    start: float, optional
        starting value for the common grid for PersLandscapeApprox objects
        in `landscapes`

    stop: float, optional
        last value in the common grid for PersLandscapeApprox objects
        in `landscapes`

    num_steps: int
        number of steps on the common grid for PersLandscapeApprox objects
        in `landscapes`

    Returns
    -------
    PersLandscapeApprox:
        The specified average of PersLandscapeApprox objects in `landscapes`
    """
    return lc_approx(
        landscapes=landscapes,
        coeffs=[1.0 / len(landscapes) for _ in landscapes],
        start=start,
        stop=stop,
        num_steps=num_steps,
    )
