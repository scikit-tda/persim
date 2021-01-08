import pytest
import numpy as np

from persim import PersLandscapeExact
from persim import PersLandscapeApprox


class TestPersLandscapeExact:
    def test_pl_critical_pairs(self):
        """
        Test critical pairs computation

        """
        # example from Peter & Pavel's paper
        P = PersLandscapeExact(
            dgms=[
                np.array([[1.0, 5.0], [2.0, 8.0], [3.0, 4.0], [5.0, 9.0], [6.0, 7.0]])
            ],
            hom_deg=0,
        )
        P.compute_landscape()

        # duplicate bars
        Q = PersLandscapeExact(dgms=[np.array([[1, 5], [1, 5], [3, 6]])], hom_deg=0)
        Q.compute_landscape()

        assert P.critical_pairs == [
            [
                [1.0, 0],
                [3.0, 2.0],
                [3.5, 1.5],
                [5.0, 3.0],
                [6.5, 1.5],
                [7.0, 2.0],
                [9.0, 0],
            ],
            [[2.0, 0], [3.5, 1.5], [5.0, 0], [6.5, 1.5], [8.0, 0]],
            [[3.0, 0], [3.5, 0.5], [4.0, 0], [6.0, 0], [6.5, 0.5], [7.0, 0]],
        ]

        assert Q.critical_pairs == [
            [[1, 0], [3.0, 2.0], [4.0, 1.0], [4.5, 1.5], [6, 0]],
            [[1, 0], [3.0, 2.0], [4.0, 1.0], [4.5, 1.5], [6, 0]],
            [[3, 0], [4.0, 1.0], [5, 0]],
        ]

    def test_pl_hom_degree(self):
        """
        Test homological degree
        """
        P = PersLandscapeExact(
            dgms=[
                np.array([[1.0, 5.0], [2.0, 8.0], [3.0, 4.0], [5.0, 9.0], [6.0, 7.0]])
            ],
            hom_deg=0,
        )
        assert P.hom_deg == 0

    def test_p_norm(self):
        """
        Test p-norms
        """
        P = PersLandscapeExact(
            critical_pairs=[[[0, 0], [1, 1], [2, 1], [3, 1], [4, 0]]], hom_deg=0
        )
        negP = PersLandscapeExact(
            critical_pairs=[[[0, 0], [1, -1], [2, -1], [3, -1], [4, 0]]], hom_deg=0
        )
        assert P.sup_norm() == 1
        assert P.p_norm(p=2) == pytest.approx(np.sqrt(2 + (2.0 / 3.0)))
        assert P.p_norm(p=5) == pytest.approx((2 + (1.0 / 3.0)) ** (1.0 / 5.0))
        assert P.p_norm(p=113) == pytest.approx((2 + (1.0 / 57.0)) ** (1.0 / 113.0))
        assert negP.sup_norm() == 1
        assert negP.p_norm(p=2) == pytest.approx(np.sqrt(2 + (2.0 / 3.0)))
        assert negP.p_norm(p=5) == pytest.approx((2 + (1.0 / 3.0)) ** (1.0 / 5.0))
        assert negP.p_norm(p=113) == pytest.approx((2 + (1.0 / 57.0)) ** (1.0 / 113.0))


class TestPersLandscapeApprox:
    def test_pl_funct_values(self):
        """
        Test PersistenceLandscape
        """
        diagrams = [np.array([[2, 6], [4, 10]])]
        P1 = PersLandscapeApprox(0, 10, 11, diagrams, homological_degree=0)
        P2 = PersLandscapeApprox(0, 10, 6, diagrams, homological_degree=0)
        P3 = PersLandscapeApprox(0, 10, 21, diagrams, homological_degree=0)

        P1.compute_landscape()
        P2.compute_landscape()
        P3.compute_landscape()

        self.assertEqual(
            P1.funct_values,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ).all()

        self.assertEqual(P2.funct_values, np.array([[0.0, 0.0, 2.0, 2.0, 2.0, 0.0]]))

        self.assertEqual(
            P3.funct_values,
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        1.0,
                        1.5,
                        2.0,
                        1.5,
                        1.0,
                        1.5,
                        2.0,
                        2.5,
                        3.0,
                        2.5,
                        2.0,
                        1.5,
                        1.0,
                        0.5,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        1.0,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
        )
