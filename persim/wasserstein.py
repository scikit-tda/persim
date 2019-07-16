import numpy as np
from sklearn import metrics
from scipy import optimize

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
        if True, return matching infromation and cross-similarity matrix

    Returns 
    ---------

    d: float
        Wasserstein distance between dgm1 and dgm2
    (matching, D): Only returns if `matching=True`
        (tuples of matched indices, (N+M)x(N+M) cross-similarity matrix)

    """

    # Step 1: Compute CSM between S and dgm2, including points on diagonal
    N = dgm1.shape[0]
    M = dgm2.shape[0]
    # Handle the cases where there are no points in the diagrams
    if N == 0:
        dgm1 = np.array([[0, 0]])
        N = 1
    if M == 0:
        dgm2 = np.array([[0, 0]])
        M = 1
    DUL = metrics.pairwise.pairwise_distances(dgm1, dgm2)

    # Put diagonal elements into the matrix
    # Rotate the diagrams to make it easy to find the straight line
    # distance to the diagonal
    cp = np.cos(np.pi/4)
    sp = np.sin(np.pi/4)
    R = np.array([[cp, -sp], [sp, cp]])
    dgm1 = dgm1[:, 0:2].dot(R)
    dgm2 = dgm2[:, 0:2].dot(R)
    D = np.zeros((N+M, N+M))
    D[0:N, 0:M] = DUL
    UR = np.max(D)*np.ones((N, N))
    np.fill_diagonal(UR, dgm1[:, 1])
    D[0:N, M:M+N] = UR
    UL = np.max(D)*np.ones((M, M))
    np.fill_diagonal(UL, dgm2[:, 1])
    D[N:M+N, 0:M] = UL
    D = D.tolist()

    # Step 2: Run the hungarian algorithm
    matchidx = optimize.linear_sum_assignment(D)[0]

    matchidx = [(i, matchidx[i]) for i in range(len(matchidx))]
    matchdist = 0
    for pair in matchidx:
        (i, j) = pair
        matchdist += D[i][j]

    if matching:
        return matchdist, (matchidx, D)

    return matchdist
