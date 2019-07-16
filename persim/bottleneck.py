"""

    Implementation of the bottleneck distance

    Author: Chris Tralie

"""

import numpy as np

from bisect import bisect_left
from hopcroftkarp import HopcroftKarp

__all__ = ["bottleneck"]


def bottleneck(dgm1, dgm2, matching=False):
    """
    Perform the Bottleneck distance matching between persistence diagrams.
    Assumes first two columns of S and T are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).

    See the `distances` notebook for an example of how to use this.

    Parameters
    -----------
    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching infromation and cross-similarity matrix

    Returns
    --------

    d: float
        bottleneck distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)
    """

    return_matching = matching

    S = np.array(dgm1)
    S = S[np.isfinite(S[:, 1]), :]
    T = np.array(dgm2)
    T = T[np.isfinite(T[:, 1]), :]

    N = S.shape[0]
    M = T.shape[0]

    # Step 1: Compute CSM between S and T, including points on diagonal
    # L Infinity distance
    Sb, Sd = S[:, 0], S[:, 1]
    Tb, Td = T[:, 0], T[:, 1]
    D1 = np.abs(Sb[:, None] - Tb[None, :])
    D2 = np.abs(Sd[:, None] - Td[None, :])
    DUL = np.maximum(D1, D2)

    # Put diagonal elements into the matrix, being mindful that Linfinity
    # balls meet the diagonal line at a diamond vertex
    D = np.zeros((N + M, N + M))
    D[0:N, 0:M] = DUL
    UR = np.max(D) * np.ones((N, N))
    np.fill_diagonal(UR, 0.5 * (S[:, 1] - S[:, 0]))
    D[0:N, M::] = UR
    UL = np.max(D) * np.ones((M, M))
    np.fill_diagonal(UL, 0.5 * (T[:, 1] - T[:, 0]))
    D[N::, 0:M] = UL

    # Step 2: Perform a binary search + Hopcroft Karp to find the
    # bottleneck distance
    N = D.shape[0]
    ds = np.unique(D.flatten())
    ds = ds[ds > 0]
    ds = np.sort(ds)
    bdist = ds[-1]
    matching = {}
    while len(ds) >= 1:
        idx = 0
        if len(ds) > 1:
            idx = bisect_left(range(ds.size), int(ds.size / 2))
        d = ds[idx]
        graph = {}
        for i in range(N):
            graph["%s" % i] = {j for j in range(N) if D[i, j] <= d}
        res = HopcroftKarp(graph).maximum_matching()
        if len(res) == 2 * N and d < bdist:
            bdist = d
            matching = res
            ds = ds[0:idx]
        else:
            ds = ds[idx + 1::]

    if return_matching:
        matchidx = [(i, matching["%i" % i]) for i in range(N)]
        return bdist, (matchidx, D)
    else:
        return bdist
