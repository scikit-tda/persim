"""
    Visualization methods for plotting persistence landscapes.
"""

import itertools
from operator import itemgetter
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from .base import PersLandscape
from .exact import PersLandscapeExact
from .approximate import PersLandscapeApprox

mpl.rcParams["text.usetex"] = True

__all__ = ["plot_landscape", "plot_landscape_simple"]


def plot_landscape(
    landscape: PersLandscape,
    num_steps: int = 3000,
    colormap="default",
    title=None,
    labels=None,
    ax=None,
    depth_range=None,
):
    """
    Plot landscape functions
    """
    if isinstance(landscape, PersLandscapeApprox):
        return plot_landscape_approx(
            landscape=landscape,
            num_steps=num_steps,
            colormap=colormap,
            title=title,
            labels=labels,
            ax=ax,
            depth_range=depth_range,
        )
    if isinstance(landscape, PersLandscapeExact):
        return plot_landscape_exact(
            landscape=landscape,
            num_steps=num_steps,
            colormap=colormap,
            title=title,
            labels=labels,
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
    """
    plot landscape functions.
    """
    if isinstance(landscape, PersLandscapeExact):
        return plot_landscape_exact_simple(
            landscape=landscape,
            alpha=alpha,
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
    colormap="default",
    alpha=0.8,
    labels=None,
    padding: float = 0.1,
    depth_padding: float = 0.7,
    title=None,
    ax=None,
    depth_range=None,
):
    """
    A plot of the exact persistence landscape.

    Warning: This function is quite slow, especially for large landscapes.

    Parameters
    ----------
    landscape: PersLandscapeExact,
        The persistence landscape to be plotted.
        
    num_steps: int, default 3000
        number of sampled points that are plotted

    color, defualt cm.viridis
        color scheme for shading of landscape functions

    alpha: float, default 0.8
        transparency of shading
        
    labels: list[string],
        A list of strings specifying labels for the coordinate axes. 
        Note that the second entry corresponds to the depth axis of the landscape.

    padding: float, default 0.1
        amount of empty grid shown to left and right of landscape functions

    depth_padding: float, default = 0.7
        amount of space between sequential landscape functions
        
    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.

    """
    fig = plt.figure()
    plt.style.use(colormap)
    ax = fig.gca(projection="3d")
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
    domain = np.linspace(
        min_crit_pt - padding * 0.1, max_crit_pt + padding * 0.1, num=num_steps
    )
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
                [depth_padding * depth, depth_padding * depth],
                ztuple,
                linewidth=0.5,
                alpha=alpha,
                # c=colormap(norm(z)))
                c=scalarMap.to_rgba(z),
            )
            ax.plot([x], [depth_padding * depth], [z], "k.", markersize=0.1)
    ax.set_ylabel("depth")
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    if title:
        plt.title(title)
    ax.set_yticks(np.arange(0, depth * landscape.max_depth + 1, depth_padding))
    ax.set_yticklabels(range(landscape.max_depth + 1))
    ax.view_init(10, 90)
    plt.show()


def plot_landscape_exact_simple(
    landscape: PersLandscapeExact,
    alpha=1,
    title=None,
    ax=None,
    labels=None,
    depth_range=None,
):
    """
    A simple plot of the persistence landscape. This is a faster plotting utility than the standard plotting, but is recommended for smaller landscapes for ease of visualization.


    Parameters
    ----------
    alpha, default 1
        transparency of shading
        
    labels: list[string],
        A list of strings specifying labels for the coordinate axes.
        
    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """
    ax = ax or plt.gca()
    landscape.compute_landscape()
    crit_pairs = list(itertools.chain.from_iterable(landscape.critical_pairs))
    min_crit_pt = min(crit_pairs, key=itemgetter(0))[0]  # smallest birth time
    max_crit_pt = max(crit_pairs, key=itemgetter(0))[0]  # largest death time
    max_crit_val = max(crit_pairs, key=itemgetter(1))[1]  # largest peak of landscape
    min_crit_val = min(crit_pairs, key=itemgetter(1))[1]  # smallest peak of landscape
    # for each landscape function

    if not depth_range:
        depth_range = range(landscape.max_depth + 1)
    for depth, l in enumerate(landscape):
        if depth not in depth_range:
            continue
        ls = np.array(l)
        ax.plot(ls[:, 0], ls[:, 1], label=f"$\lambda_{{{depth}}}$", alpha=alpha)
    ax.legend()
    if title:
        ax.set_title(title)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


def plot_landscape_approx(
    landscape: PersLandscapeApprox,
    num_steps: int = 3000,
    colormap="default",
    labels=None,
    alpha=0.8,
    padding: float = 0.1,
    depth_padding: float = 0.7,
    title=None,
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

    padding: float, default 0.1
        amount of empty grid shown to left and right of landscape functions

    depth_padding: float, default = 0.7
        amount of space between sequential landscape functions

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plt.style.use(colormap)
    landscape.compute_landscape()
    # TODO: RE the following line: is this better than np.concatenate?
    #       There is probably an even better way without creating an intermediary.
    _vals = list(itertools.chain.from_iterable(landscape.values))
    min_val = min(_vals)
    max_val = max(_vals)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    scalarMap = mpl.cm.ScalarMappable(norm=norm)
    # x-axis for grid
    domain = np.linspace(
        landscape.start - padding * 0.1, landscape.stop + padding * 0.1, num=num_steps
    )
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
                [depth_padding * depth, depth_padding * depth],
                ztuple,
                linewidth=0.5,
                alpha=alpha,
                # c=colormap(norm(z)))
                c=scalarMap.to_rgba(z),
            )
            ax.plot([x], [depth_padding * depth], [z], "k.", markersize=0.1)
    ax.set_ylabel("depth")
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    ax.set_yticks(np.arange(0, depth * landscape.max_depth + 1, depth_padding))
    ax.set_yticklabels(range(landscape.max_depth + 1))
    if title:
        plt.title(title)
    ax.view_init(10, 90)
    plt.show()


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
    A simple plot of the persistence landscape. This is a faster plotting utility than the standard plotting, but is recommended for smaller landscapes for ease of visualization.

    Parameters
    ----------
    alpha, default 1
        transparency of shading

    padding: float, default 0.1
        amount of empty grid shown to left and right of landscape functions

    num_steps: int, default 1000
        number of sampled points that are plotted

    labels: list[string],
        A list of strings specifying labels for the coordinate axes.

    depth_range: slice, default = None
        Specifies a range of depths to be plotted. The default behavior is to plot all.
    """
    ax = ax or plt.gca()

    landscape.compute_landscape()

    # TODO: RE the following line: is this better than np.concatenate?
    #       There is probably an even better way without creating an intermediary.
    _vals = list(itertools.chain.from_iterable(landscape.values))
    min_val = min(_vals)
    max_val = max(_vals)

    if not depth_range:
        depth_range = range(landscape.max_depth + 1)
    for depth, l in enumerate(landscape):
        if depth not in depth_range:
            continue
        # instantiate depth-specific domain
        domain = np.linspace(
            landscape.start - padding * 0.1, landscape.stop + padding * 0.1, num=len(l)
        )

        ax.plot(domain, l, label=f"$\lambda_{{{depth}}}$", alpha=alpha)
    ax.legend()
    if title:
        ax.set_title(title)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
