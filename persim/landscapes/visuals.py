"""
    Visualization methods for plotting persistence landscapes.
"""

import itertools
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .base import PersLandscape
from .exact import PersLandscapeExact
from .approximate import PersLandscapeApprox

__all__ = ["plot_landscape", "plot_landscape_simple"]


def plot_landscape(
    landscape: PersLandscape,
    num_steps: int = 3000,
    color="default",
    alpha: float = 0.8,
    title=None,
    labels=None,
    padding: float = 0.0,
    ax=None,
    depth_range=None,
):
    """
    A 3-dimensional plot of a persistence landscape.

    If the user wishes to modify the plot beyond the provided parameters, they
    should create a matplotlib.pyplot figure axis first and then pass it as the
    optional 'ax' parameter. This allows for easy modification of the plots
    after creation.

    Warning: This function is quite slow, especially for large landscapes.

    Parameters
    ----------
    landscape: PersLandscape,
        The persistence landscape to be plotted.

    num_steps: int, default 3000
        The number of sampled points that are plotted.

    color, defualt cm.viridis
        The color scheme for shading of landscape functions.

    alpha: float, default 0.8
        The transparency of shading.

    title: string
        The title of the plot.

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.
        Note that the second entry corresponds to the depth axis of the landscape.

    padding: float, default 0.0
        The amount of empty space or margin shown to left and right of the
        axis within the figure.

    ax: matplotlib axis, default = None
        An optional parameter allowing the user to pass a matplotlib axis for later modification.

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """
    if isinstance(landscape, PersLandscapeApprox):
        return plot_landscape_approx(
            landscape=landscape,
            num_steps=num_steps,
            color=color,
            alpha=alpha,
            title=title,
            labels=labels,
            padding=padding,
            ax=ax,
            depth_range=depth_range,
        )
    if isinstance(landscape, PersLandscapeExact):
        return plot_landscape_exact(
            landscape=landscape,
            num_steps=num_steps,
            color=color,
            alpha=alpha,
            title=title,
            labels=labels,
            padding=padding,
            ax=ax,
            depth_range=depth_range,
        )


def plot_landscape_simple(
    landscape: PersLandscape,
    alpha=1,
    padding=0.1,
    num_steps=1000,
    title=None,
    ax=None,
    labels=None,
    depth_range=None,
):
    """A 2-dimensional plot of the persistence landscape.

    This is a faster plotting
    utility than the standard plotting, but is recommended for smaller landscapes
    for ease of visualization.

    If the user wishes to modify the plot beyond the provided parameters, they
    should create a matplotlib.figure axis first and then pass it as the optional 'ax'
    parameter. This allows for easy modification of the plots after creation.

    Parameters
    ----------
    landscape: PersLandscape
        The landscape to be plotted.

    alpha: float, default 1
        The transparency of shading.

    padding: float, default 0.1
        The amount of empty space or margin shown to left and right of the
        landscape functions.

    num_steps: int, default 1000
        The number of sampled points that are plotted. Only used for plotting
        PersLandscapeApprox classes.

    title: string
        The title of the plot.

    ax: matplotlib axis, default = None
        The axis to plot on.

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.

    """
    if isinstance(landscape, PersLandscapeExact):
        return plot_landscape_exact_simple(
            landscape=landscape,
            alpha=alpha,
            padding=padding,
            title=title,
            ax=ax,
            labels=labels,
            depth_range=depth_range,
        )
    if isinstance(landscape, PersLandscapeApprox):
        return plot_landscape_approx_simple(
            landscape=landscape,
            alpha=alpha,
            padding=padding,
            num_steps=num_steps,
            title=title,
            ax=ax,
            labels=labels,
            depth_range=depth_range,
        )


def plot_landscape_exact(
    landscape: PersLandscapeExact,
    num_steps: int = 3000,
    color="default",
    alpha=0.8,
    title=None,
    labels=None,
    padding: float = 0.0,
    ax=None,
    depth_range=None,
):
    """
    A 3-dimensional plot of the exact persistence landscape.

    Warning: This function is quite slow, especially for large landscapes.

    Parameters
    ----------
    landscape: PersLandscapeExact,
        The persistence landscape to be plotted.

    num_steps: int, default 3000
        The number of sampled points that are plotted.

    color, default cm.viridis
        The color scheme for shading of landscape functions.

    alpha: float, default 0.8
        The transparency of the shading.

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.
        Note that the second entry corresponds to the depth axis of the landscape.

    padding: float, default 0.0
        The amount of empty grid shown to left and right of landscape functions.

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.

    """
    fig = plt.figure()
    plt.style.use(color)
    ax = fig.add_subplot(projection="3d")
    landscape.compute_landscape()
    # itemgetter index selects which entry to take max/min wrt.
    # the hanging [0] or [1] takes that entry.
    crit_pairs = list(itertools.chain.from_iterable(landscape.critical_pairs))
    min_crit_pt = min(crit_pairs, key=itemgetter(0))[0]  # smallest birth time
    max_crit_pt = max(crit_pairs, key=itemgetter(0))[0]  # largest death time
    max_crit_val = max(crit_pairs, key=itemgetter(1))[1]  # largest peak of landscape
    min_crit_val = min(crit_pairs, key=itemgetter(1))[1]  # smallest peak of landscape
    norm = mpl.colors.Normalize(vmin=min_crit_val, vmax=max_crit_val)
    scalarMap = mpl.cm.ScalarMappable(norm=norm)
    # x-axis for grid
    domain = np.linspace(min_crit_pt, max_crit_pt, num=num_steps)
    # for each landscape function
    if not depth_range:
        depth_range = range(landscape.max_depth + 1)
    for depth, l in enumerate(landscape):
        if depth not in depth_range:
            continue
        # sequential pairs in landscape
        xs, zs = zip(*l)
        image = np.interp(domain, xs, zs)
        for x, z in zip(domain, image):
            if z == 0.0:
                # plot a single point here?
                continue  # moves to the next iterable in for loop
            if z > 0.0:
                ztuple = [0, z]
            elif z < 0.0:
                ztuple = [z, 0]
            # for coloring https://matplotlib.org/3.1.0/tutorials/colors/colormapnorms.html
            ax.plot(
                [x, x],  # plotting a line to get shaded function
                [depth, depth],
                ztuple,
                linewidth=0.5,
                alpha=alpha,
                # c=colormap(norm(z)))
                c=scalarMap.to_rgba(z),
            )
            ax.plot([x], [depth], [z], "k.", markersize=0.1)
    ax.set_ylabel("depth")
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    if title:
        plt.title(title)
    ax.margins(padding)
    ax.view_init(10, 90)
    return fig


def plot_landscape_exact_simple(
    landscape: PersLandscapeExact,
    alpha=1,
    padding=0.1,
    title=None,
    ax=None,
    labels=None,
    depth_range=None,
):
    """
    A 2-dimensional plot of the persistence landscape. This is a faster plotting
    utility than the standard plotting, but is recommended for smaller landscapes
    for ease of visualization.

    Parameters
    ----------
    landscape: PersLandscape
        The landscape to be plotted.

    alpha: float, default 1
        The transparency of shading.

    padding: float, default 0.1
        The amount of empty space or margin shown to left and right of the
        landscape functions.

    title: string
        The title of the plot.

    ax: matplotlib axis, default = None
        The axis to plot on.

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """
    ax = ax or plt.gca()
    landscape.compute_landscape()
    if not depth_range:
        depth_range = range(landscape.max_depth + 1)
    for depth, l in enumerate(landscape):
        if depth not in depth_range:
            continue
        ls = np.array(l)
        ax.plot(ls[:, 0], ls[:, 1], label=f"$\lambda_{{{depth}}}$", alpha=alpha)
    ax.legend()
    ax.margins(padding)
    if title:
        ax.set_title(title)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    return ax


def plot_landscape_approx(
    landscape: PersLandscapeApprox,
    num_steps: int = 3000,
    color="default",
    alpha=0.8,
    title=None,
    labels=None,
    padding: float = 0.0,
    ax=None,
    depth_range=None,
):
    """
    A plot of the approximate persistence landscape.

    Warning: This function is quite slow, especially for large landscapes.

    Parameters
    ----------
    num_steps: int, default 3000
        number of sampled points that are plotted

    color, defualt cm.viridis
        color scheme for shading of landscape functions

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.
        Note that the second entry corresponds to the depth axis of the landscape.

    alpha, default 0.8
        transparency of shading

    padding: float, default 0.0
        amount of empty grid shown to left and right of landscape functions

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """
    fig = plt.figure()
    plt.style.use(color)
    ax = fig.add_subplot(projection="3d")
    landscape.compute_landscape()
    # TODO: RE the following line: is this better than np.concatenate?
    #       There is probably an even better way without creating an intermediary.
    _vals = list(itertools.chain.from_iterable(landscape.values))
    min_val = min(_vals)
    max_val = max(_vals)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    scalarMap = mpl.cm.ScalarMappable(norm=norm)
    # x-axis for grid
    domain = np.linspace(landscape.start, landscape.stop, num=num_steps)
    # for each landscape function
    if not depth_range:
        depth_range = range(landscape.max_depth + 1)
    for depth, l in enumerate(landscape):
        if depth not in depth_range:
            continue
        # sequential pairs in landscape
        # xs, zs = zip(*l)
        image = np.interp(
            domain,
            np.linspace(
                start=landscape.start, stop=landscape.stop, num=landscape.num_steps
            ),
            l,
        )
        for x, z in zip(domain, image):
            if z == 0.0:
                # plot a single point here?
                continue  # moves to the next iterable in for loop
            if z > 0.0:
                ztuple = [0, z]
            elif z < 0.0:
                ztuple = [z, 0]
            # for coloring https://matplotlib.org/3.1.0/tutorials/colors/colormapnorms.html
            ax.plot(
                [x, x],  # plotting a line to get shaded function
                [depth, depth],
                ztuple,
                linewidth=0.5,
                alpha=alpha,
                # c=colormap(norm(z)))
                c=scalarMap.to_rgba(z),
            )
            ax.plot([x], [depth], [z], "k.", markersize=0.1)
    ax.set_ylabel("depth")
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    ax.margins(padding)
    if title:
        plt.title(title)
    ax.view_init(10, 90)
    return fig


def plot_landscape_approx_simple(
    landscape: PersLandscapeApprox,
    alpha=1,
    padding=0.1,
    num_steps=1000,
    title=None,
    ax=None,
    labels=None,
    depth_range=None,
):
    """
    A 2-dimensional plot of the persistence landscape. This is a faster plotting
    utility than the standard plotting, but is recommended for smaller landscapes
    for ease of visualization.

    Parameters
    ----------
    landscape: PersLandscape
        The landscape to be plotted.

    alpha: float, default 1
        The transparency of shading.

    padding: float, default 0.1
        The amount of empty space or margin shown to left and right of the
        landscape functions.

    num_steps: int, default 1000
        The number of sampled points that are plotted. Only used for plotting
        PersLandscapeApprox classes.

    title: string
        The title of the plot.

    ax: matplotlib axis, default = None
        The axis to plot on.

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """

    ax = ax or plt.gca()
    landscape.compute_landscape()
    if not depth_range:
        depth_range = range(landscape.max_depth + 1)
    for depth, l in enumerate(landscape):
        if depth not in depth_range:
            continue
        domain = np.linspace(landscape.start, landscape.stop, num=len(l))
        ax.plot(domain, l, label=f"$\lambda_{{{depth}}}$", alpha=alpha)
    ax.legend()
    ax.margins(padding)
    if title:
        ax.set_title(title)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    return ax
