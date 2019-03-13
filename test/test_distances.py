import numpy as np
import scipy.sparse as sps

import pytest

from persim import bottleneck
from persim import sliced_wasserstein
from persim import heat
from persim import gromov_hausdorff_between_graphs, gromov_hausdorff

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
        dgm2 =  np.array([
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
        assert D.shape  == (len(dgm1) + len(dgm2), len(dgm1) + len(dgm2))



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
    def test_two_graphs(self):
        A_G = sps.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        A_H = sps.csr_matrix(([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        lb, ub = gromov_hausdorff_between_graphs(A_G, A_H)

        assert lb == 0.5
        assert ub == 0.5

    def test_many_graphs(self):
        A_G = sps.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        A_H = sps.csr_matrix(([1]*6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4))
        A_I = sps.csr_matrix(([1]*4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        lbs, ubs = gromov_hausdorff_between_graphs([A_G, A_H, A_I])

        np.testing.assert_array_equal(lbs, np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))
        np.testing.assert_array_equal(ubs, np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))

    def test_single_point(self):
        D_X = np.array([[0, 2, 20], [2, 0, 20], [20, 2, 0]])
        D_Y = np.array([[0]])
        lb, ub = gromov_hausdorff(D_X, D_Y)

        assert lb == 10
        assert ub == 10

    def test_isomorphic(self):
        D_X = np.array([[0, 2], [2, 0]])
        D_Y = np.array([[0, 2], [2, 0]])
        lb, ub = gromov_hausdorff(D_X, D_Y)

        assert lb == 0
        assert ub == 0

    def test_cliques(self):
        D_X = np.array([[0, 1], [1, 0]])
        D_Y = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
        lb, ub = gromov_hausdorff(D_X, D_Y)

        assert lb == 0.5
        assert ub == 0.5

    def test_same_size(self):
        D_X = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
        D_Y = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
        lb, ub = gromov_hausdorff(D_X, D_Y)

        assert lb == 0.5
        assert ub == 0.5
