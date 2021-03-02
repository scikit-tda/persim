import pytest
import numpy as np

from persim.landscapes import PersLandscapeExact
from persim.landscapes import PersLandscapeApprox
from persim.landscapes import PersistenceLandscaper
from persim.landscapes import vectorize, snap_pl, lc_approx, average_approx
from persim.landscapes import death_vector


class TestPersLandscapeExact:
    def test_exact_empty(self):
        with pytest.raises(ValueError):
            PersLandscapeExact()

    def test_exact_hom_deg(self):
        P = PersLandscapeExact(dgms=[np.array([[1.0, 5.0]])], hom_deg=0,)
        assert P.hom_deg == 0
        with pytest.raises(ValueError):
            PersLandscapeExact(hom_deg=-1)

    def test_exact_critical_pairs(self):
        assert not PersLandscapeExact(
            dgms=[np.array([[0, 3]])], compute=False
        ).critical_pairs
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

        np.testing.assert_array_equal(
            P.critical_pairs,
            [
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
            ],
        )

        np.testing.assert_array_equal(
            Q.critical_pairs,
            [
                [[1, 0], [3.0, 2.0], [4.0, 1.0], [4.5, 1.5], [6, 0]],
                [[1, 0], [3.0, 2.0], [4.0, 1.0], [4.5, 1.5], [6, 0]],
                [[3, 0], [4.0, 1.0], [5, 0]],
            ],
        )

    def test_exact_add(self):
        with pytest.raises(ValueError):
            PersLandscapeExact(
                critical_pairs=[[[0, 0], [1, 1]]], hom_deg=0
            ) + PersLandscapeExact(critical_pairs=[[[0, 0], [1, 1]]], hom_deg=1)
        Q = PersLandscapeExact(
            critical_pairs=[[[0, 0], [1, 1], [2, 2]]]
        ) + PersLandscapeExact(critical_pairs=[[[0, 0], [0.5, 1], [1.5, 3]]])
        np.testing.assert_array_equal(
            Q.critical_pairs, [[[0, 0], [0.5, 1.5], [1, 3], [1.5, 4.5], [2, 5]]]
        )

    def test_exact_neg(self):
        P = PersLandscapeExact(
            critical_pairs=[[[0, 0], [1, 1], [2, 1], [3, 1], [4, 0]]]
        )
        assert (-P).critical_pairs == [[[0, 0], [1, -1], [2, -1], [3, -1], [4, 0]]]
        Q = PersLandscapeExact(critical_pairs=[[[0, 0], [5, 0]]])
        assert (-Q).critical_pairs == Q.critical_pairs

    def test_exact_mul(self):
        P = PersLandscapeExact(
            critical_pairs=[[[0, 0], [1, 2], [2, 0], [3, -1], [4, 0]]]
        )
        np.testing.assert_array_equal(
            (4 * P).critical_pairs, [[[0, 0], [1, 8], [2, 0], [3, -4], [4, 0]]]
        )
        np.testing.assert_array_equal(
            (P * (-1)).critical_pairs, [[[0, 0], [1, -2], [2, 0], [3, 1], [4, 0]]]
        )

    def test_exact_div(self):
        P = PersLandscapeExact(
            critical_pairs=[[[0, 0], [1, 2], [2, 0], [3, -1], [4, 0]]]
        )
        np.testing.assert_array_equal(
            (P / 2).critical_pairs, [[[0, 0], [1, 1], [2, 0], [3, -0.5], [4, 0]]]
        )
        with pytest.raises(ValueError):
            P / 0

    def test_exact_get_item(self):
        P = PersLandscapeExact(dgms=[np.array([[1, 5], [1, 5], [3, 6]])])
        np.testing.assert_array_equal(
            P[1], [[1, 0], [3, 2], [4, 1], [4.5, 1.5], [6, 0]]
        )
        with pytest.raises(IndexError):
            P[3]

    def test_exact_norm(self):
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
    def test_approx_empty(self):
        with pytest.raises(ValueError):
            PersLandscapeApprox()

    def test_approx_compute_flag(self):
        assert (
            PersLandscapeApprox(dgms=[np.array([[0, 4]])], compute=False).values.size
            == 0
        )

    def test_approx_hom_deg(self):
        with pytest.raises(IndexError):
            PersLandscapeApprox(dgms=[np.array([[2, 6], [4, 10]])], hom_deg=2)
        with pytest.raises(ValueError):
            PersLandscapeApprox(hom_deg=-1)
        assert (
            PersLandscapeApprox(
                dgms=[np.array([[2, 6], [4, 10]]), np.array([[1, 5], [4, 6]])],
                hom_deg=1,
            ).hom_deg
            == 1
        )

    def test_approx_grid_params(self):
        with pytest.raises(ValueError):
            PersLandscapeApprox(values=np.array([[2, 6], [4, 10]]), start=1)
        with pytest.raises(ValueError):
            PersLandscapeApprox(values=np.array([[2, 6], [4, 10]]), stop=11)
        with pytest.raises(ValueError):
            PersLandscapeApprox(values=np.array([[2, 6], [4, 10]]), start=3, stop=2)
        P = PersLandscapeApprox(dgms=[np.array([[0, 3], [1, 4]])], num_steps=100)
        assert P.start == 0
        assert P.stop == 4
        assert P.num_steps == 100
        Q = PersLandscapeApprox(values=np.array([[2, 6], [4, 10]]), start=0, stop=5)
        assert Q.start == 0
        assert Q.stop == 5

    def test_approx_compute_landscape(self):
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

        # duplicate bars
        diagrams2 = [np.array([[2, 6], [2, 6], [4, 10]])]
        Q2 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams2, hom_deg=0
        )

        # edge case: bars start same value
        diagrams3 = [np.array([[3, 5], [3, 7]])]
        Q3 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams3, hom_deg=0
        )

        # edge case: bars end same value
        diagrams4 = [np.array([[2, 6], [4, 6]])]
        Q4 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams4, hom_deg=0
        )
        np.testing.assert_array_equal(
            P1.values,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            P2.values, np.array([[0.0, 0.0, 2.0, 2.0, 2.0, 0.0]])
        )

        np.testing.assert_array_equal(
            P3.values,
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

        np.testing.assert_array_equal(
            Q2.values,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            Q3.values,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            Q4.values,
            np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_approx_values_to_pairs(self):
        diagrams1 = [np.array([[2, 6], [4, 10]])]
        P1 = PersLandscapeApprox(
            start=0, stop=10, num_steps=11, dgms=diagrams1, hom_deg=0
        )
        np.testing.assert_array_equal(
            P1.values_to_pairs(),
            np.array(
                [
                    [
                        [0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 1],
                        [4, 2],
                        [5, 1],
                        [6, 2],
                        [7, 3],
                        [8, 2],
                        [9, 1],
                        [10, 0],
                    ],
                    [
                        [0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [4, 0],
                        [5, 1],
                        [6, 0],
                        [7, 0],
                        [8, 0],
                        [9, 0],
                        [10, 0],
                    ],
                ]
            ),
        )

    def test_approx_add(self):
        with pytest.raises(ValueError):
            PersLandscapeApprox(
                dgms=[np.array([[0, 1]])], start=0, stop=2
            ) + PersLandscapeApprox(dgms=[np.array([[0, 1]])], start=1, stop=2)
        with pytest.raises(ValueError):
            PersLandscapeApprox(
                dgms=[np.array([[0, 1]])], start=0, stop=2
            ) + PersLandscapeApprox(dgms=[np.array([[0, 1]])], start=0, stop=3)
        with pytest.raises(ValueError):
            PersLandscapeApprox(
                dgms=[np.array([[0, 1]])], start=0, stop=2, num_steps=100
            ) + PersLandscapeApprox(
                dgms=[np.array([[0, 1]])], start=1, stop=2, num_steps=200
            )
        with pytest.raises(ValueError):
            PersLandscapeApprox(
                dgms=[np.array([[0, 1], [1, 2]])], hom_deg=0
            ) + PersLandscapeApprox(
                dgms=[np.array([[0, 1], [1, 2]]), np.array([[1, 3]])], hom_deg=1
            )

        Q = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, dgms=[np.array([[1, 4]])]
        ) + PersLandscapeApprox(start=0, stop=5, num_steps=6, dgms=[np.array([[2, 5]])])
        np.testing.assert_array_equal(Q.values, np.array([[0, 0, 1, 2, 1, 0]]))

    def test_approx_neg(self):
        P = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, dgms=[np.array([[0, 5], [1, 3]])]
        )
        np.testing.assert_array_equal(
            (-P).values, np.array([[0, -1, -2, -2, -1, 0], [0, 0, -1, 0, 0, 0]])
        )

    def test_approx_sub(self):
        P = PersLandscapeApprox(
            start=0,
            stop=5,
            num_steps=6,
            values=np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 0, 0, 0]]),
        )
        Q = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, values=np.array([[0, 1, 1, 1, 1, 0]])
        )
        np.testing.assert_array_equal(
            (P - Q).values, np.array([[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]])
        )

    def test_approx_mul(self):
        P = PersLandscapeApprox(
            start=0,
            stop=5,
            num_steps=6,
            values=np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 0, 0, 0]]),
        )
        np.testing.assert_array_equal(
            (3 * P).values, np.array([[0, 3, 6, 6, 3, 0], [0, 0, 3, 0, 0, 0]])
        )

    def test_approx_div(self):
        P = PersLandscapeApprox(
            start=0,
            stop=5,
            num_steps=6,
            values=np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 0, 0, 0]]),
        )
        with pytest.raises(ValueError):
            P / 0
        np.testing.assert_array_equal(
            (P / 2).values, np.array([[0, 0.5, 1, 1, 0.5, 0], [0, 0, 0.5, 0, 0, 0]])
        )

    def test_approx_get_item(self):
        P = PersLandscapeApprox(
            start=0,
            stop=5,
            num_steps=6,
            values=np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 0, 0, 0]]),
        )
        with pytest.raises(IndexError):
            P[3]
        np.testing.assert_array_equal(P[1], np.array([0, 0, 1, 0, 0, 0]))

    def test_approx_norm(self):
        P = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, values=np.array([[0, 1, 1, 1, 1, 0]]),
        )
        assert P.p_norm(p=2) == pytest.approx(np.sqrt(3 + (2.0 / 3.0)))
        assert P.sup_norm() == 1.0
        Q = PersLandscapeApprox(
            start=0,
            stop=5,
            num_steps=6,
            values=np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 1, 0, 0]]),
        )
        assert Q.p_norm(p=3) == pytest.approx((16 + 3.0 / 2.0) ** (1.0 / 3.0))
        assert Q.sup_norm() == 2.0


class TestAuxiliary:
    def test_vectorize(self):
        P = PersLandscapeExact(dgms=[np.array([[0, 5], [1, 4]])])
        Q = vectorize(P, start=0, stop=5, num_steps=6)
        assert Q.hom_deg == P.hom_deg
        assert Q.start == 0
        assert Q.stop == 5
        np.testing.assert_array_equal(
            Q.values, np.array([[0, 1, 2, 2, 1, 0], [0, 0, 1, 1, 0, 0]])
        )
        R = vectorize(P)
        assert R.start == 0
        assert R.stop == 5

    def test_snap_PL(self):
        P = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, values=np.array([[0, 1, 1, 1, 1, 0]]),
        )
        [P_snapped] = snap_pl([P], start=0, stop=10, num_steps=11)
        assert P_snapped.hom_deg == P.hom_deg
        assert P_snapped.start == 0
        assert P_snapped.stop == 10
        assert P_snapped.num_steps == 11
        np.testing.assert_array_equal(
            P_snapped.values, np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
        )
        Q = PersLandscapeApprox(
            start=1, stop=6, num_steps=6, values=np.array([[0, 1, 2, 2, 1, 0]])
        )
        [P_snap, Q_snap] = snap_pl([P, Q], start=0, stop=10, num_steps=11)
        assert P_snap.start == 0
        assert Q_snap.stop == 10
        assert P_snap.num_steps == Q_snap.num_steps
        np.testing.assert_array_equal(
            Q_snap.values, np.array([[0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0]])
        )

    def test_lc_approx(self):
        P = PersLandscapeApprox(
            start=0,
            stop=10,
            num_steps=11,
            values=np.array([[0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]]),
        )
        Q = PersLandscapeApprox(
            start=0,
            stop=10,
            num_steps=11,
            values=np.array([[0, 1, 2, 2, 1, 0, 1, 2, 3, 2, 0]]),
        )
        lc = lc_approx([P, Q], [2, -1])
        np.testing.assert_array_equal(
            lc.values, np.array([[0, 1, 0, 0, 1, 0, -1, 0, -1, -2, 0]])
        )

    def test_average_approx(self):
        P = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, values=np.array([[0, 1, 2, 3, 2, 1]])
        )
        Q = PersLandscapeApprox(
            start=0, stop=5, num_steps=6, values=np.array([[0, -1, -2, -1, 0, 1]])
        )
        np.testing.assert_array_equal(
            average_approx([P, Q]).values, np.array([[0, 0, 0, 1, 1, 1]])
        )

    def test_death_vector(self):
        dgms = [np.array([[0, 4], [0, 1], [0, 10]])]
        np.testing.assert_array_equal(death_vector(dgms), [10, 4, 1])


class TestTransformer:
    def test_persistenceimager(self):
        pl = PersistenceLandscaper(hom_deg=0, num_steps=5, flatten=True)
        assert pl.hom_deg == 0
        assert not pl.start
        assert not pl.stop
        assert pl.num_steps == 5
        assert pl.flatten
        dgms = [np.array([[0, 3], [1, 4]]), np.array([[1, 4]])]
        pl.fit(dgms)
        assert pl.start == 0
        assert pl.stop == 4.0
        np.testing.assert_array_equal(
            pl.transform(dgms),
            np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,]),
        )
        pl2 = PersistenceLandscaper(hom_deg=1, num_steps=4)
        assert pl2.hom_deg == 1
        pl2.fit(dgms)
        assert pl2.start == 1.0
        assert pl2.stop == 4.0
        np.testing.assert_array_equal(pl2.transform(dgms), [[0.0, 1.0, 1.0, 0.0]])
        pl3 = PersistenceLandscaper(hom_deg=0, num_steps=5, flatten=True)
        np.testing.assert_array_equal(
            pl3.fit_transform(dgms),
            np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,]),
        )
