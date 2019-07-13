import matplotlib.pyplot as plt
import numpy as np

from .visuals import plot_diagrams

__all__ = ["bottleneck_matching", "wasserstein_matching"]


import numpy as np

import matplotlib.pyplot as plt

default_palette = ["#54DCC6", "#7991F2", "#CF1259", "#FDBB5D", "#DD7596"]


def plot_diagrams_custom(diagrams, labels, palette=None, ax=None):

    ax = ax or plt.subplot(111, aspect='equal', xticks=[], yticks=[])
    palette = palette or default_palette

    # labels = [
    #         "$H_0$",
    #         "$H_1$",
    #         "$H_2$",
    #         "$H_3$"
    # ]

    size=20
    xlabel, ylabel = "Birth", "Death"

    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
    x_r = ax_max - ax_min

    # Give plot a nice buffer on all sides.
    # ax_range=0 when only one point,
    buffer = x_r / 5

    x_down = ax_min - buffer / 2
    x_up = ax_max + buffer

    y_down, y_up = x_down, x_up

    yr = y_up - y_down
    ax.plot([x_down, x_up], [x_down, x_up], "--", c='k')

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label, color in zip(diagrams, labels, palette):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, c=color, edgecolor=color, alpha=1.0, zorder=999)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.ticklabel_format()
    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')
    ax.legend(loc="lower right")



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


    Examples
    ==========

    bn_matching, (matchidx, D) = persim.bottleneck(A_h1, B_h1, matching=True)
    persim.bottleneck_matching(A_h1, B_h1, matchidx, D)

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


def wasserstein_matching(I1, I2, matchidx, palette=None, labels=["dgm1", "dgm2"], colors=None, ax=None):
    """ Visualize bottleneck matching between two diagrams

    Parameters
    ===========

    I1: array
        A diagram
    I2: array
        A diagram
    matchidx: tuples of matched indices
        if input `matching=True`, then return matching
    labels: list of strings
        names of diagrams for legend. Default = ["dgm1", "dgm2"], 
    ax: matplotlib Axis object
        For plotting on a particular axis.

    Examples
    ==========

    bn_matching, (matchidx, D) = persim.wasserstien(A_h1, B_h1, matching=True)
    persim.wasserstein_matching(A_h1, B_h1, matchidx, D)

    """


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

    plot_diagrams_custom([I1, I2], labels=labels, palette=palette,ax=ax)