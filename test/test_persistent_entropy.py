import numpy as np

import pytest

from persim import persistent_entropy as pe


class TestPersistentEntropy:
    def test_one_diagram(self):
        dgm = np.array([[0, 1], [0, 3], [2, 4]])
        p = pe.persistent_entropy(dgm)

        # An upper bound of persistent entropy is the logarithm of the number of bars.
        assert p < np.log(len(dgm))

    def test_diagrams(self):
        dgms = [
            np.array([[0, 1], [0, 3], [2, 4]]),
            np.array([[2, 5], [3, 8]]),
            np.array([[0, 10]]),
        ]
        p = pe.persistent_entropy(dgms)
        # An upper bound of persistent entropy is the logarithm of the number of bars.
        assert all(p < np.log(3))

    def test_diagrams_inf(self):
        dgms = [
            np.array([[0, 1], [0, 3], [2, 4]]),
            np.array([[2, 5], [3, 8]]),
            np.array([[0, np.inf]]),
        ]
        p = pe.persistent_entropy(dgms, keep_inf=True, val_inf=10)
        # An upper bound of persistent entropy is the logarithm of the number of bars.
        assert all(p < np.log(3))

    def test_diagram_one_bar(self):
        dgm = np.array([[-1, 2]])
        p = pe.persistent_entropy(dgm)
        assert all(p == 0)
