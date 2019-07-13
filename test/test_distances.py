import numpy as np
import scipy.sparse as sps

import pytest

from persim import bottleneck, wasserstein
from persim import sliced_wasserstein
from persim import heat
from persim import gromov_hausdorff, gromov_hausdorff


class TestBottleneck:
    def test_single(self):
        d = bottleneck(
            np.array([[0.5, 1]]),
            np.array([[0.5, 1.1]])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.001)

    def test_some(self):
        d = bottleneck(
            np.array([
                [0.5, 1],
                [0.6, 1.1]
            ]),
            np.array([
                [0.5, 1.1],
                [0.6, 1.3]
            ])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.2, 0.001)

    def test_diagonal(self):
        d = bottleneck(
            np.array([
                [10.5, 10.5],
                [10.6, 10.5],
                [10.3, 10.3]
            ]),
            np.array([
                [0.5, 1.0],
                [0.6, 1.2],
                [0.3, 0.7]
            ])
        )

        # I expect this to be 0.6
        assert d == pytest.approx(0.3, 0.001)

    def test_different_size(self):
        d = bottleneck(
            np.array([
                [0.5, 1],
                [0.6, 1.1]
            ]),
            np.array([
                [0.5, 1.1]
            ])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.001)

    def test_matching(self):
        dgm1 = np.array([
            [0.5, 1],
            [0.6, 1.1]
        ])
        dgm2 = np.array([
            [0.5, 1.1],
            [0.6, 1.1],
            [0.8, 1.1],
            [1.0, 1.1],
        ])

        d, (m, D) = bottleneck(
            dgm1, dgm2,
            matching=True
        )

        # These are very loose bounds
        assert len(m) == len(dgm1) + len(dgm2)
        assert D.shape == (len(dgm1) + len(dgm2), len(dgm1) + len(dgm2))


class TestWasserstein:
    def test_single(self):
        d = wasserstein(
            np.array([[0.5, 1]]),
            np.array([[0.5, 1.1]])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.001)

    def test_some(self):
        d = wasserstein(
            np.array([
                [0.5, 1],
                [0.6, 1.1]
            ]),
            np.array([
                [0.5, 1.1],
                [0.6, 1.3]
            ])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.3, 0.001)


class TestSliced:
    def test_single(self):
        d = sliced_wasserstein(
            np.array([[0.5, 1]]),
            np.array([[0.5, 1.1]])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.01)

    def test_some(self):
        d = sliced_wasserstein(
            np.array([
                [0.5, 1],
                [0.6, 1.1]
            ]),
            np.array([
                [0.5, 1.1],
                [0.6, 1.2]
            ])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.19, 0.02)

    def test_different_size(self):
        d = sliced_wasserstein(
            np.array([
                [0.5, 1],
                [0.6, 1.1]
            ]),
            np.array([
                [0.6, 1.2]
            ])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.314, 0.1)


class TestHeat:
    def test_compare(self):
        """ lets at least be sure that large distances are captured """
        d1 = heat(
            np.array([[0.5, 1]]),
            np.array([[0.5, 1.1]])
        )
        d2 = heat(
            np.array([[0.5, 1]]),
            np.array([[0.5, 1.5]])
        )

        # These are very loose bounds
        assert d1 < d2


class TestModifiedGromovHausdorff:
    def test_single_point(self):
        A_G = sps.csr_matrix(
            ([1]*4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        A_H = sps.csr_matrix(([], ([], [])), shape=(1, 1))
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 1
        assert ub == 1

    def test_isomorphic(self):
        A_G = sps.csr_matrix(
            ([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        A_H = sps.csr_matrix(
            ([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 0
        assert ub == 0

    def test_cliques(self):
        A_G = sps.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        A_H = sps.csr_matrix(
            ([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 0.5
        assert ub == 0.5

    def test_same_size(self):
        A_G = sps.csr_matrix(
            ([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        A_H = sps.csr_matrix(
            ([1]*4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 0.5
        assert ub == 0.5

    def test_many_graphs(self):
        A_G = sps.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        A_H = sps.csr_matrix(
            ([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        A_I = sps.csr_matrix(
            ([1]*4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        lbs, ubs = gromov_hausdorff([A_G, A_H, A_I])

        np.testing.assert_array_equal(lbs, np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))
        np.testing.assert_array_equal(ubs, np.array(
            [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))
