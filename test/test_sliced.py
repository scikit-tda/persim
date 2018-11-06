import numpy as np

import pytest

from persim import sliced_wasserstein


def test_sliced_single():
    d = sliced_wasserstein(
        np.array([[0.5, 1]]),
        np.array([[0.5, 1.1]])
    )

    # These are very loose bounds
    assert d == pytest.approx(0.1, 0.01)

def test_sliced_some():
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

