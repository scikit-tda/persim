"""
    Auxilary functions for working with persistence diagrams.
"""

import itertools
import numpy as np


def union_vals(A, B):
    """Helper function for summing grid landscapes.

    Extends one list to the length of the other by padding with zero lists.
    """
    diff = A.shape[0] - B.shape[0]
    if diff < 0:
        # B has more entries, so pad A
        A = np.pad(A, pad_width=((0, np.abs(diff)), (0, 0)))
        return A, B
    elif diff > 0:
        # A has more entries, so pad B
        B = np.pad(B, pad_width=((0, diff), (0, 0)))
        return A, B
    else:
        return A, B


def union_crit_pairs(A, B):
    """Helper function for summing landscapes.

    Computes the union of two sets of critical pairs.
    """
    result_pairs = []
    A.compute_landscape()
    B.compute_landscape()
    # zip functions in landscapes A and B and pad with None
    for a, b in list(itertools.zip_longest(A.critical_pairs, B.critical_pairs)):
        # B had more functions
        if a is None:
            result_pairs.append(b)
        # A had more functions
        elif b is None:
            result_pairs.append(a)
        # A, B > pos_to_slope_interp > sum_slopes > slope_to_pos_interp
        else:
            result_pairs.append(
                slope_to_pos_interp(
                    sum_slopes(
                        pos_to_slope_interp(a),
                        pos_to_slope_interp(b),
                    )
                )
            )
    return result_pairs


def pos_to_slope_interp(l: list) -> list:
    """Convert positions of critical pairs to (x-value, slope) pairs.

    Intended for internal use. Inverse function of `slope_to_pos_interp`.

    Result
    ------
    list
        [(xi,mi)] for i in len(function in landscape)
    """

    output = []
    # for sequential pairs in landscape function
    for [[x0, y0], [x1, y1]] in zip(l, l[1:]):
        slope = (y1 - y0) / (x1 - x0)
        output.append([x0, slope])
    output.append([l[-1][0], 0])
    return output


def slope_to_pos_interp(l: list) -> list:
    """Convert positions of (x-value, slope) pairs to critical pairs.

    Intended
    for internal use. Inverse function of `pos_to_slope_interp`.

    Result
    ------
    list
        [(xi, yi)]_i for i in len(function in landscape)
    """
    output = [[l[0][0], 0]]
    # for sequential pairs in [(xi,mi)]_i
    for [[x0, m], [x1, _]] in zip(l, l[1:]):
        # uncover y0 and y1 from slope formula
        y0 = output[-1][1]
        y1 = y0 + (x1 - x0) * m
        output.append([x1, y1])
    return output


def sum_slopes(a: list, b: list) -> list:
    """
    Sum two piecewise linear functions, each represented as a list
    of pairs (xi,mi), where each xi is the x-value of critical pair and
    mi is the slope. The input should be of the form of the output of the
    `pos_to_slope_interp' function.

    Result
    ------
    list

    """
    result = []
    am, bm = 0, 0  # initialize slopes
    while len(a) > 0 or len(b) > 0:
        if len(a) == 0 or (len(a) > 0 and len(b) > 0 and a[0][0] > b[0][0]):
            # The next critical pair comes from list b.
            bx, bm = b[0]
            # pop b0
            b = b[1:]
            result.append([bx, am + bm])
        elif len(b) == 0 or (len(a) > 0 and len(b) > 0 and a[0][0] < b[0][0]):
            # The next critical pair comes from list a.
            ax, am = a[0]
            # pop a0
            a = a[1:]
            result.append([ax, am + bm])
        else:
            # The x-values of two critical pairs coincide.
            ax, am = a[0]
            bx, bm = b[0]
            # pop a0 and b0
            a, b = a[1:], b[1:]
            result.append([ax, am + bm])
    return result


def ndsnap_regular(points, *grid_axes):
    """Snap points to the 2d grid determined by grid_axes"""
    # https://stackoverflow.com/q/8457645/717525
    snapped = []
    for i, ax in enumerate(grid_axes):
        diff = ax[:, np.newaxis] - points[:, i]
        best = np.argmin(np.abs(diff), axis=0)
        snapped.append(ax[best])
    return np.array(snapped).T


def _p_norm(p: float, critical_pairs: list = []):
    """
    Compute `p` norm of interpolated piecewise linear function defined from list of
    critical pairs.
    """
    result = 0.0
    for l in critical_pairs:
        for [[x0, y0], [x1, y1]] in zip(l, l[1:]):
            if y0 == y1:
                # horizontal line segment
                result += (np.abs(y0) ** p) * (x1 - x0)
                continue
            # slope is well-defined
            slope = (y1 - y0) / (x1 - x0)
            b = y0 - slope * x0
            # segment crosses the x-axis
            if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
                z = -b / slope
                ev_x1 = (slope * x1 + b) ** (p + 1) / (slope * (p + 1))
                ev_x0 = (slope * x0 + b) ** (p + 1) / (slope * (p + 1))
                ev_z = (slope * z + +b) ** (p + 1) / (slope * (p + 1))
                result += np.abs(ev_x1 + ev_x0 - 2 * ev_z)
            # segment does not cross the x-axis
            else:
                ev_x1 = (slope * x1 + b) ** (p + 1) / (slope * (p + 1))
                ev_x0 = (slope * x0 + b) ** (p + 1) / (slope * (p + 1))
                result += np.abs(ev_x1 - ev_x0)
    return (result) ** (1.0 / p)
