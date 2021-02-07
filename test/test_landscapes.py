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
        diagrams1 = [np.array([[2, 6], [4, 10]])]
        P1 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams1, hom_deg=0
        )
        P2 = PersLandscapeApprox(
            start=0, stop=10, num_steps=6, dgms=diagrams1, hom_deg=0
        )
        P3 = PersLandscapeApprox(
            start=0, stop=10, num_steps=21, dgms=diagrams1, hom_deg=0
        )
        
        #duplicate bars
        diagrams2 = [np.array([[2, 6], [2, 6], [4, 10]])]
        Q2 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams2, hom_deg=0
        )
        
        #edge case: bars start same value
        diagrams3 = [np.array([[3, 5], [3, 7]])]
        Q3 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams3, hom_deg=0
        )
        
        #edge case: bars end same value
        diagrams4 = [np.array([[2,6], [4,6]])]
        Q4 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams4, hom_deg=0
        )
        assert (
            P1.values
            == np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (P2.values == np.array([[0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])).all()

        assert (
            P3.values
            == np.array(
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
            )
        ).all()
        
        
        assert (
            Q2.values
            == np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        
        assert (
            Q3.values
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        
        assert (
            Q4.values
            == np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        
        
        
        
        
