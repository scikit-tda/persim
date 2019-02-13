import numpy as np
from scipy.spatial.distance import cityblock

__all__ = ["sliced_wasserstein"]

def sliced_wasserstein(PD1, PD2, M=50):
    """ Implementation of Sliced Wasserstein distance as described in 
        Sliced Wasserstein Kernel for Persistence Diagrams by Mathieu Carriere, Marco Cuturi, Steve Oudot (https://arxiv.org/abs/1706.03358)


        Parameters
        -----------
        
        PD1: np.array size (m,2)
            Persistence diagram
        PD2: np.array size (n,2)
            Persistence diagram
        M: int, default is 50
            Iterations to run approximation.

        Returns
        --------
        sw: float
            Sliced Wasserstein distance between PD1 and PD2
    """

    diag_theta = np.array(
        [np.cos(0.25 * np.pi), np.sin(0.25 * np.pi)], dtype=np.float32
    )

    l_theta1 = [np.dot(diag_theta, x) for x in PD1]
    l_theta2 = [np.dot(diag_theta, x) for x in PD2]

    if (len(l_theta1) != PD1.shape[0]) or (len(l_theta2) != PD2.shape[0]):
        raise ValueError("The projected points and origin do not match")

    PD_delta1 = [[np.sqrt(x ** 2 / 2.0)] * 2 for x in l_theta1]
    PD_delta2 = [[np.sqrt(x ** 2 / 2.0)] * 2 for x in l_theta2]

    # i have the input now to compute the sw
    sw = 0
    theta = 0.5
    step = 1.0 / M
    for i in range(M):
        l_theta = np.array(
            [np.cos(theta * np.pi), np.sin(theta * np.pi)], dtype=np.float32
        )

        V1 = [np.dot(l_theta, x) for x in PD1] + [np.dot(l_theta, x) for x in PD_delta2]

        V2 = [np.dot(l_theta, x) for x in PD2] + [np.dot(l_theta, x) for x in PD_delta1]

        sw += step * cityblock(sorted(V1), sorted(V2))
        theta += step

    return sw
