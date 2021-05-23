import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_diagrams", "bottleneck_matching", "wasserstein_matching"]


def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()

def plot_a_bar(p, q, c='b', linestyle='-'):
    plt.plot([p[0], q[0]], [p[1], q[1]], c=c, linestyle=linestyle, linewidth=1)

def bottleneck_matching(dgm1, dgm2, matching, labels=["dgm1", "dgm2"], ax=None):
    """ Visualize bottleneck matching between two diagrams

    Parameters
    ===========

    dgm1: Mx(>=2) 
        array of birth/death pairs for PD 1
    dgm2: Nx(>=2) 
        array of birth/death paris for PD 2
    matching: ndarray(Mx+Nx, 3)
        A list of correspondences in an optimal matching, as well as their distance, where:
        * First column is index of point in first persistence diagram, or -1 if diagonal
        * Second column is index of point in second persistence diagram, or -1 if diagonal
        * Third column is the distance of each matching
    labels: list of strings
        names of diagrams for legend. Default = ["dgm1", "dgm2"], 
    ax: matplotlib Axis object
        For plotting on a particular axis.


    Examples
    ==========

    dist, matching = persim.bottleneck(A_h1, B_h1, matching=True)
    persim.bottleneck_matching(A_h1, B_h1, matching)

    """
    ax = ax or plt.gca()

    plot_diagrams([dgm1, dgm2], labels=labels, ax=ax)
    cp = np.cos(np.pi / 4)
    sp = np.sin(np.pi / 4)
    R = np.array([[cp, -sp], [sp, cp]])
    if dgm1.size == 0:
        dgm1 = np.array([[0, 0]])
    if dgm2.size == 0:
        dgm2 = np.array([[0, 0]])
    dgm1Rot = dgm1.dot(R)
    dgm2Rot = dgm2.dot(R)
    max_idx = np.argmax(matching[:, 2])
    for idx, [i, j, d] in enumerate(matching):
        i = int(i)
        j = int(j)
        linestyle = '--'
        linewidth = 1
        c = 'C2'
        if idx == max_idx:
            linestyle = '-'
            linewidth = 2
            c = 'C3'
        if i != -1 or j != -1: # At least one point is a non-diagonal point
            if i == -1:
                diagElem = np.array([dgm2Rot[j, 0], 0])
                diagElem = diagElem.dot(R.T)
                plt.plot([dgm2[j, 0], diagElem[0]], [dgm2[j, 1], diagElem[1]], c, linewidth=linewidth, linestyle=linestyle)
            elif j == -1:
                diagElem = np.array([dgm1Rot[i, 0], 0])
                diagElem = diagElem.dot(R.T)
                ax.plot([dgm1[i, 0], diagElem[0]], [dgm1[i, 1], diagElem[1]], c, linewidth=linewidth, linestyle=linestyle)
            else:
                ax.plot([dgm1[i, 0], dgm2[j, 0]], [dgm1[i, 1], dgm2[j, 1]], c, linewidth=linewidth, linestyle=linestyle)


def wasserstein_matching(dgm1, dgm2, matching, labels=["dgm1", "dgm2"], ax=None):
    """ Visualize bottleneck matching between two diagrams

    Parameters
    ===========

    dgm1: array
        A diagram
    dgm2: array
        A diagram
    matching: ndarray(Mx+Nx, 3)
        A list of correspondences in an optimal matching, as well as their distance, where:
        * First column is index of point in first persistence diagram, or -1 if diagonal
        * Second column is index of point in second persistence diagram, or -1 if diagonal
        * Third column is the distance of each matching
    labels: list of strings
        names of diagrams for legend. Default = ["dgm1", "dgm2"], 
    ax: matplotlib Axis object
        For plotting on a particular axis.

    Examples
    ==========

    bn_matching, (matchidx, D) = persim.wasserstien(A_h1, B_h1, matching=True)
    persim.wasserstein_matching(A_h1, B_h1, matchidx, D)

    """
    ax = ax or plt.gca()

    cp = np.cos(np.pi / 4)
    sp = np.sin(np.pi / 4)
    R = np.array([[cp, -sp], [sp, cp]])
    if dgm1.size == 0:
        dgm1 = np.array([[0, 0]])
    if dgm2.size == 0:
        dgm2 = np.array([[0, 0]])
    dgm1Rot = dgm1.dot(R)
    dgm2Rot = dgm2.dot(R)
    for [i, j, d] in matching:
        i = int(i)
        j = int(j)
        if i != -1 or j != -1: # At least one point is a non-diagonal point
            if i == -1:
                diagElem = np.array([dgm2Rot[j, 0], 0])
                diagElem = diagElem.dot(R.T)
                plt.plot([dgm2[j, 0], diagElem[0]], [dgm2[j, 1], diagElem[1]], "g")
            elif j == -1:
                diagElem = np.array([dgm1Rot[i, 0], 0])
                diagElem = diagElem.dot(R.T)
                ax.plot([dgm1[i, 0], diagElem[0]], [dgm1[i, 1], diagElem[1]], "g")
            else:
                ax.plot([dgm1[i, 0], dgm2[j, 0]], [dgm1[i, 1], dgm2[j, 1]], "g")

    plot_diagrams([dgm1, dgm2], labels=labels, ax=ax)
