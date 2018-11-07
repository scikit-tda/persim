import numpy as np

import pytest

from persim import bottleneck
from persim import sliced_wasserstein
from persim import heat

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
