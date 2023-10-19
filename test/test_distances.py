import numpy as np
import pytest
import scipy.sparse as sps

from persim import (bottleneck, gromov_hausdorff, heat, sliced_wasserstein,
                    wasserstein)


class TestBottleneck:
    def test_single(self):
        d = bottleneck(np.array([[0.5, 1]]), np.array([[0.5, 1.1]]))

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.001)

    def test_some(self):
        d = bottleneck(
            np.array([[0.5, 1], [0.6, 1.1]]), np.array([[0.5, 1.1], [0.6, 1.3]])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.2, 0.001)

    def test_diagonal(self):
        d = bottleneck(
            np.array([[10.5, 10.5], [10.6, 10.5], [10.3, 10.3]]),
            np.array([[0.5, 1.0], [0.6, 1.2], [0.3, 0.7]]),
        )

        # I expect this to be 0.6
        assert d == pytest.approx(0.3, 0.001)

    def test_different_size(self):
        d = bottleneck(np.array([[0.5, 1], [0.6, 1.1]]), np.array([[0.5, 1.1]]))
        assert d == 0.25

    def test_matching(self):
        dgm1 = np.array([[0.5, 1], [0.6, 1.1]])
        dgm2 = np.array(
            [
                [0.5, 1.1],
                [0.6, 1.1],
                [0.8, 1.1],
                [1.0, 1.1],
            ]
        )

        d, m = bottleneck(dgm1, dgm2, matching=True)
        u1 = np.unique(m[:, 0])
        u1 = u1[u1 >= 0]
        u2 = np.unique(m[:, 1])
        u2 = u2[u2 >= 0]
        assert u1.size == dgm1.shape[0] and u2.size == dgm2.shape[0]

    def test_matching_to_self(self):
        # Matching a diagram to itself should yield 0
        pd = np.array(
            [
                [0.0, 1.71858561],
                [0.0, 1.74160683],
                [0.0, 2.43430877],
                [0.0, 2.56949258],
                [0.0, np.inf],
            ]
        )
        dist = bottleneck(pd, pd)
        assert dist == 0

    def test_single_point_same(self):
        dgm = np.array([[0.11371516, 4.45734882]])
        dist = bottleneck(dgm, dgm)
        assert dist == 0

    def test_2x2_bisect_bug(self):
        dgm1 = np.array([[6, 9], [6, 8]])
        dgm2 = np.array([[4, 10], [9, 10]])
        dist = bottleneck(dgm1, dgm2)
        assert dist == 2

    def test_one_empty(self):
        dgm1 = np.array([[1, 2]])
        empty = np.array([[]])
        dist = bottleneck(dgm1, empty)
        assert dist == 0.5

    def test_inf_deathtime(self):
        dgm = np.array([[1, 2]])
        empty = np.array([[0, np.inf]])
        with pytest.warns(
            UserWarning, match="dgm1 has points with non-finite death"
        ) as w:
            dist1 = bottleneck(empty, dgm)
        with pytest.warns(
            UserWarning, match="dgm2 has points with non-finite death"
        ) as w:
            dist2 = bottleneck(dgm, empty)
        assert (dist1 == 0.5) and (dist2 == 0.5)

    def test_repeated(self):
        # Issue #44
        G = np.array([[0, 1], [0, 1]])
        H = np.array([[0, 1]])
        dist = bottleneck(G, H)
        assert dist == 0.5


class TestWasserstein:
    def test_single(self):
        d = wasserstein(np.array([[0.5, 1]]), np.array([[0.5, 1.1]]))

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.001)

    def test_some(self):
        d = wasserstein(
            np.array([[0.6, 1.1], [0.5, 1]]), np.array([[0.5, 1.1], [0.6, 1.3]])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.3, 0.001)

    def test_matching_to_self(self):
        # Matching a diagram to itself should yield 0
        pd = np.array(
            [[0.0, 1.71858561], [0.0, 1.74160683], [0.0, 2.43430877], [0.0, 2.56949258]]
        )
        dist = wasserstein(pd, pd)
        assert dist == 0

    def test_single_point_same(self):
        dgm = np.array([[0.11371516, 4.45734882]])
        dist = wasserstein(dgm, dgm)
        assert dist == 0

    def test_one_empty(self):
        dgm1 = np.array([[1, 2]])
        empty = np.array([])
        dist = wasserstein(dgm1, empty)
        assert np.allclose(dist, np.sqrt(2) / 2)

    def test_inf_deathtime(self):
        dgm = np.array([[1, 2]])
        empty = np.array([[0, np.inf]])
        with pytest.warns(
            UserWarning, match="dgm1 has points with non-finite death"
        ) as w:
            dist1 = wasserstein(empty, dgm)
        with pytest.warns(
            UserWarning, match="dgm2 has points with non-finite death"
        ) as w:
            dist2 = wasserstein(dgm, empty)
        assert (np.allclose(dist1, np.sqrt(2) / 2)) and (
            np.allclose(dist2, np.sqrt(2) / 2)
        )

    def test_repeated(self):
        dgm1 = np.array([[0, 10], [0, 10]])
        dgm2 = np.array([[0, 10]])
        dist = wasserstein(dgm1, dgm2)
        assert dist == pytest.approx(5 * np.sqrt(2))

    def test_matching(self):
        dgm1 = np.array([[0.5, 1], [0.6, 1.1]])
        dgm2 = np.array(
            [
                [0.5, 1.1],
                [0.6, 1.1],
                [0.8, 1.1],
                [1.0, 1.1],
            ]
        )

        d, m = wasserstein(dgm1, dgm2, matching=True)
        u1 = np.unique(m[:, 0])
        u1 = u1[u1 >= 0]
        u2 = np.unique(m[:, 1])
        u2 = u2[u2 >= 0]
        assert u1.size == dgm1.shape[0] and u2.size == dgm2.shape[0]


class TestSliced:
    def test_single(self):
        d = sliced_wasserstein(np.array([[0.5, 1]]), np.array([[0.5, 1.1]]))

        # These are very loose bounds
        assert d == pytest.approx(0.1, 0.01)

    def test_some(self):
        d = sliced_wasserstein(
            np.array([[0.5, 1], [0.6, 1.1]]), np.array([[0.5, 1.1], [0.6, 1.2]])
        )

        # These are very loose bounds
        assert d == pytest.approx(0.19, 0.02)

    def test_different_size(self):
        d = sliced_wasserstein(np.array([[0.5, 1], [0.6, 1.1]]), np.array([[0.6, 1.2]]))

        # These are very loose bounds
        assert d == pytest.approx(0.314, 0.1)

    def test_single_point_same(self):
        dgm = np.array([[0.11371516, 4.45734882]])
        dist = sliced_wasserstein(dgm, dgm)
        assert dist == 0


class TestHeat:
    def test_compare(self):
        """lets at least be sure that large distances are captured"""
        d1 = heat(np.array([[0.5, 1]]), np.array([[0.5, 1.1]]))
        d2 = heat(np.array([[0.5, 1]]), np.array([[0.5, 1.5]]))

        # These are very loose bounds
        assert d1 < d2

    def test_single_point_same(self):
        dgm = np.array([[0.11371516, 4.45734882]])
        dist = heat(dgm, dgm)
        assert dist == 0


class TestModifiedGromovHausdorff:
    def test_single_point(self):
        A_G = sps.csr_matrix(([1] * 4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        A_H = sps.csr_matrix(([], ([], [])), shape=(1, 1))
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 1
        assert ub == 1

    def test_isomorphic(self):
        A_G = sps.csr_matrix(
            ([1] * 6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4)
        )
        A_H = sps.csr_matrix(
            ([1] * 6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4)
        )
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 0
        assert ub == 0

    def test_cliques(self):
        A_G = sps.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        A_H = sps.csr_matrix(
            ([1] * 6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4)
        )
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 0.5
        assert ub == 0.5

    def test_same_size(self):
        A_G = sps.csr_matrix(
            ([1] * 6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4)
        )
        A_H = sps.csr_matrix(([1] * 4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        lb, ub = gromov_hausdorff(A_G, A_H)

        assert lb == 0.5
        assert ub == 0.5

    def test_many_graphs(self):
        A_G = sps.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        A_H = sps.csr_matrix(
            ([1] * 6, ([0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])), shape=(4, 4)
        )
        A_I = sps.csr_matrix(([1] * 4, ([0, 0, 1, 2], [1, 3, 2, 3])), shape=(4, 4))
        lbs, ubs = gromov_hausdorff([A_G, A_H, A_I])

        np.testing.assert_array_equal(
            lbs, np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        )
        np.testing.assert_array_equal(
            ubs, np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        )
