"""
    Tools for working with exact and approximate persistence landscapes.
"""

import itertools
import numpy as np
from operator import itemgetter, attrgetter

from .approximate import PersLandscapeApprox
from .exact import PersLandscapeExact

__all__ = [
    "vectorize",
    "snap_PL",
    "lc_approx",
    "average_approx",
]


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


def vectorize(
    l: PersLandscapeExact, start: float = None, stop: float = None, num_dims: int = 500
) -> PersLandscapeApprox:
    """Converts a `PersLandscapeExact` type to a `PersLandscapeApprox` type.

    Parameters
    ----------
    start: float, default None
        start value of grid
    if start is not inputed, start is assigned to minimum birth value

    stop: float, default None
        stop value of grid
    if stop is not inputed, stop is assigned to maximum death value

    num_dims: int, default 500
        number of points starting from `start` and ending at `stop`

    """

    l.compute_landscape()
    if start is None:
        start = min(l.critical_pairs, key=itemgetter(0))[0]
    if stop is None:
        stop = max(l.critical_pairs, key=itemgetter(0))[0]
    grid = np.linspace(start, stop, num_dims)
    result = []
    # creates sequential pairs of points for each lambda in critical_pairs
    for depth in l.critical_pairs:
        xs, ys = zip(*depth)
        result.append(np.interp(grid, xs, ys))
    return PersLandscapeApprox(
        start=start,
        stop=stop,
        num_dims=num_dims,
        hom_deg=l.hom_deg,
        values=np.array(result),
    )
