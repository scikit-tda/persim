"""

    Implementation of the bottleneck distance using binary
    search and the Hopcroft-Karp algorithm

    Author: Chris Tralie

"""

import numpy as np

from bisect import bisect_left
from hopcroftkarp import HopcroftKarp
import warnings

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
    M = min(S.shape[0], S.size)
    if S.size > 0:
        S = S[np.isfinite(S[:, 1]), :]
        if S.shape[0] < M:
            warnings.warn(
                "dgm1 has points with non-finite death times;"+
                "ignoring those points"
            )
            M = S.shape[0]
    T = np.array(dgm2)
    N = min(T.shape[0], T.size)
    if T.size > 0:
        T = T[np.isfinite(T[:, 1]), :]
        if T.shape[0] < N:
            warnings.warn(
                "dgm2 has points with non-finite death times;"+
                "ignoring those points"
            )
            N = T.shape[0]

    if M == 0:
        S = np.array([[0, 0]])
        M = 1
    if N == 0:
        T = np.array([[0, 0]])
        N = 1

    # Step 1: Compute CSM between S and T, including points on diagonal
    # L Infinity distance
    Sb, Sd = S[:, 0], S[:, 1]
    Tb, Td = T[:, 0], T[:, 1]
    D1 = np.abs(Sb[:, None] - Tb[None, :])
    D2 = np.abs(Sd[:, None] - Td[None, :])
    DUL = np.maximum(D1, D2)

    # Put diagonal elements into the matrix, being mindful that Linfinity
    # balls meet the diagonal line at a diamond vertex
    D = np.zeros((M + N, M + N))
    D[0:M, 0:N] = DUL
    UR = np.max(D) * np.ones((M, M))
    np.fill_diagonal(UR, 0.5 * (S[:, 1] - S[:, 0]))
    D[0:M, N::] = UR
    UL = np.max(D) * np.ones((N, N))
    np.fill_diagonal(UL, 0.5 * (T[:, 1] - T[:, 0]))
    D[M::, 0:N] = UL

    # Step 2: Perform a binary search + Hopcroft Karp to find the
    # bottleneck distance
    M = D.shape[0]
    ds = np.sort(np.unique(D.flatten()))
    bdist = ds[-1]
    matching = {}
    while len(ds) >= 1:
        idx = 0
        if len(ds) > 1:
            idx = bisect_left(range(ds.size), int(ds.size / 2))
        d = ds[idx]
        graph = {}
        for i in range(M):
            graph["%s" % i] = {j for j in range(M) if D[i, j] <= d}
        res = HopcroftKarp(graph).maximum_matching()
        if len(res) == 2 * M and d <= bdist:
            bdist = d
            matching = res
            ds = ds[0:idx]
        else:
            ds = ds[idx + 1::]

    if return_matching:
        matchidx = [(i, matching["%i" % i]) for i in range(M)]
        return bdist, (matchidx, D)
    else:
        return bdist
