"""

    Implementation of the Wasserstein distance using
    the Hungarian algorithm

    Author: Chris Tralie

"""
import numpy as np
from sklearn import metrics
from scipy import optimize
import warnings

__all__ = ["wasserstein"]


def wasserstein(dgm1, dgm2, matching=False):
    """
    Perform the Wasserstein distance matching between persistence diagrams.
    Assumes first two columns of dgm1 and dgm2 are the coordinates of the persistence
    points, but allows for other coordinate columns (which are ignored in
    diagonal matching).

    See the `distances` notebook for an example of how to use this.

    Parameters
    ------------

    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: bool, default False
        if True, return matching information and cross-similarity matrix

    Returns 
    ---------

    d: float
        Wasserstein distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)

    """

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
    # Compute CSM between S and dgm2, including points on diagonal
    DUL = metrics.pairwise.pairwise_distances(S, T)

    # Put diagonal elements into the matrix
    # Rotate the diagrams to make it easy to find the straight line
    # distance to the diagonal
    cp = np.cos(np.pi/4)
    sp = np.sin(np.pi/4)
    R = np.array([[cp, -sp], [sp, cp]])
    S = S[:, 0:2].dot(R)
    T = T[:, 0:2].dot(R)
    D = np.zeros((M+N, M+N))
    np.fill_diagonal(D, 0)
    D[0:M, 0:N] = DUL
    UR = np.inf*np.ones((M, M))
    np.fill_diagonal(UR, S[:, 1])
    D[0:M, N:N+M] = UR
    UL = np.inf*np.ones((N, N))
    np.fill_diagonal(UL, T[:, 1])
    D[M:N+M, 0:N] = UL

    # Step 2: Run the hungarian algorithm
    matchi, matchj = optimize.linear_sum_assignment(D)
    matchdist = np.sum(D[matchi, matchj])

    if matching:
        matchidx = [(i, j) for i, j in zip(matchi, matchj)]
        ret = np.zeros((len(matchidx), 3))
        ret[:, 0:2] = np.array(matchidx)
        ret[:, 2] = D[matchi, matchj]
        # Indicate diagonally matched points
        ret[ret[:, 0] >= M, 0] = -1
        ret[ret[:, 1] >= N, 1] = -1
        # Exclude diagonal to diagonal
        ret = ret[ret[:, 0] + ret[:, 1] != -2, :] 
        return matchdist, ret

    return matchdist
