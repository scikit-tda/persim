import numpy as np

import matplotlib.pyplot as plt

import persim
from persim import plot_diagrams

from persim.landscapes import (
    plot_landscape,
    plot_landscape_simple,
    PersLandscapeExact,
    PersLandscapeApprox,
)

"""

    Testing visualization is a little more difficult, but still necessary. An example of how to get started:
    > https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib

    ```
    def test_plot_square2():
        f, ax = plt.subplots()
        x, y = [0, 1, 2], [0, 1, 2]
        plot_square(x, y)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(y_plot, np.square(y))

    ```


    Notes
    -----

    ax.get_children() gives all the pieces in the plot, very useful
    Scatter data is of type `PathCollection` and will be a child of ax.

"""


class TestPlotting:
    def test_single(self):
        """Most just test this doesn't crash"""
        diagram = np.array([[0, 1], [1, 1], [2, 4], [3, 5]])

        f, ax = plt.subplots()
        plot_diagrams(diagram, show=False)

        x_plot, y_plot = ax.lines[0].get_xydata().T

        assert x_plot[0] <= np.min(diagram)
        assert x_plot[1] >= np.max(diagram)

        # get PathCollection
        pathcols = [
            child
            for child in ax.get_children()
            if child.__class__.__name__ == "PathCollection"
        ]
        assert len(pathcols) == 1

    def test_multiple(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, show=False)

        pathcols = [
            child
            for child in ax.get_children()
            if child.__class__.__name__ == "PathCollection"
        ]

        assert len(pathcols) == 2
        np.testing.assert_array_equal(pathcols[0].get_offsets(), diagrams[0])
        np.testing.assert_array_equal(pathcols[1].get_offsets(), diagrams[1])

    def test_plot_only(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]
        f, ax = plt.subplots()
        plot_diagrams(diagrams, legend=False, show=False, plot_only=[1])

    def test_legend_true(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, legend=True, show=False)
        legend = [
            child for child in ax.get_children() if child.__class__.__name__ == "Legend"
        ]

        assert len(legend) == 1

    def test_legend_false(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, legend=False, show=False)
        legend = [
            child for child in ax.get_children() if child.__class__.__name__ == "Legend"
        ]

        assert len(legend) == 0

    def test_set_title(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, title="my title", show=False)
        assert ax.get_title() == "my title"

        f, ax = plt.subplots()
        plot_diagrams(diagrams, show=False)
        assert ax.get_title() == ""

    def test_default_square(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, show=False)
        diagonal = ax.lines[0].get_xydata()

        assert diagonal[0, 0] == diagonal[0, 1]
        assert diagonal[1, 0] == diagonal[1, 1]

    def test_default_label(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, show=False)

        assert ax.get_ylabel() == "Death"
        assert ax.get_xlabel() == "Birth"

    def test_lifetime(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, lifetime=True, show=False)

        assert ax.get_ylabel() == "Lifetime"
        assert ax.get_xlabel() == "Birth"

        line = ax.get_lines()[0]
        np.testing.assert_array_equal(line.get_ydata(), [0, 0])

    def test_lifetime_removes_birth(self):
        diagrams = [
            np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, lifetime=True, show=False)

        pathcols = [
            child
            for child in ax.get_children()
            if child.__class__.__name__ == "PathCollection"
        ]

        modded1 = diagrams[0]
        modded1[:, 1] = diagrams[0][:, 1] - diagrams[0][:, 0]
        modded2 = diagrams[1]
        modded2[:, 1] = diagrams[1][:, 1] - diagrams[1][:, 0]
        assert len(pathcols) == 2
        np.testing.assert_array_equal(pathcols[0].get_offsets(), modded1)
        np.testing.assert_array_equal(pathcols[1].get_offsets(), modded2)

    def test_infty(self):
        diagrams = [
            np.array([[0, np.inf], [1, 1], [2, 4], [3, 5]]),
            np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
        ]

        f, ax = plt.subplots()
        plot_diagrams(diagrams, legend=True, show=False)

        # Right now just make sure nothing breaks


class TestMatching:
    def test_bottleneck_matching(self):
        dgm1 = np.array([[0.1, 0.2], [0.2, 0.4]])
        dgm2 = np.array([[0.1, 0.2], [0.3, 0.45]])

        d, matching = persim.bottleneck(dgm1, dgm2, matching=True)
        persim.bottleneck_matching(dgm1, dgm2, matching)

    def test_plot_labels(self):
        dgm1 = np.array([[0.1, 0.2], [0.2, 0.4]])
        dgm2 = np.array([[0.1, 0.2], [0.3, 0.45]])

        d, matching = persim.bottleneck(dgm1, dgm2, matching=True)
        persim.bottleneck_matching(dgm1, dgm2, matching, labels=["X", "Y"])


class TestLandscapePlots:
    diagrams = [
        np.array([[0, 1], [1, 1], [2, 4], [3, 5]]),
        np.array([[0.5, 3], [2, 4], [4, 5], [10, 15]]),
    ]

    # Test to ensure plots are created

    def test_simple_plots(self):
        plot_landscape_simple(PersLandscapeApprox(dgms=self.diagrams))
        plot_landscape_simple(PersLandscapeExact(dgms=self.diagrams, hom_deg=1))

    def test_plots(self):
        plot_landscape(PersLandscapeExact(dgms=self.diagrams))
        plot_landscape(PersLandscapeApprox(dgms=self.diagrams, hom_deg=1))
