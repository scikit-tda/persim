import matplotlib.pyplot as plt
import numpy as np

from .visuals import plot_diagrams

__all__ = ["bottleneck_matching", "wasserstein_matching"]


def bottleneck_matching(I1, I2, matchidx, D, labels=["dgm1", "dgm2"], ax=None):
    """ Visualize bottleneck matching between two diagrams

    Parameters
    ===========

    I1: array
        A diagram
    I2: array
        A diagram
    matchidx: tuples of matched indices
        if input `matching=True`, then return matching
    D: array
        cross-similarity matrix
    labels: list of strings
        names of diagrams for legend. Default = ["dgm1", "dgm2"], 
    ax: matplotlib Axis object
        For plotting on a particular axis.

    """

    plot_diagrams([I1, I2], labels=labels, ax=ax) 
    cp = np.cos(np.pi / 4)
    sp = np.sin(np.pi / 4)
    R = np.array([[cp, -sp], [sp, cp]])
    if I1.size == 0:
        I1 = np.array([[0, 0]])
    if I2.size == 0:
        I2 = np.array([[0, 0]])
    I1Rot = I1.dot(R)
    I2Rot = I2.dot(R)
    dists = [D[i, j] for (i, j) in matchidx]
    (i, j) = matchidx[np.argmax(dists)]
    if i >= I1.shape[0] and j >= I2.shape[0]:
        return
    if i >= I1.shape[0]:
        diagElem = np.array([I2Rot[j, 0], 0])
        diagElem = diagElem.dot(R.T)
        plt.plot([I2[j, 0], diagElem[0]], [I2[j, 1], diagElem[1]], "g")
    elif j >= I2.shape[0]:
        diagElem = np.array([I1Rot[i, 0], 0])
        diagElem = diagElem.dot(R.T)
        plt.plot([I1[i, 0], diagElem[0]], [I1[i, 1], diagElem[1]], "g")
    else:
        plt.plot([I1[i, 0], I2[j, 0]], [I1[i, 1], I2[j, 1]], "g")


def wasserstein_matching(I1, I2, matchidx, labels=["dgm1", "dgm2"]):
    plot_diagrams([I1, I2], labels=labels)
    cp = np.cos(np.pi / 4)
    sp = np.sin(np.pi / 4)
    R = np.array([[cp, -sp], [sp, cp]])
    if I1.size == 0:
        I1 = np.array([[0, 0]])
    if I2.size == 0:
        I2 = np.array([[0, 0]])
    I1Rot = I1.dot(R)
    I2Rot = I2.dot(R)
    for index in matchidx:
        (i, j) = index
        if i >= I1.shape[0] and j >= I2.shape[0]:
            continue
        if i >= I1.shape[0]:
            diagElem = np.array([I2Rot[j, 0], 0])
            diagElem = diagElem.dot(R.T)
            plt.plot([I2[j, 0], diagElem[0]], [I2[j, 1], diagElem[1]], "g")
        elif j >= I2.shape[0]:
            diagElem = np.array([I1Rot[i, 0], 0])
            diagElem = diagElem.dot(R.T)
            plt.plot([I1[i, 0], diagElem[0]], [I1[i, 1], diagElem[1]], "g")
        else:
            plt.plot([I1[i, 0], I2[j, 0]], [I1[i, 1], I2[j, 1]], "g")
