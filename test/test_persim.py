import pytest

import numpy as np
from persimmon import PersImage

def test_landscape():
    bds = np.array([[1,1],[1,2]])

    ldsp = PersImage.to_landscape(bds)

    np.testing.assert_array_equal(ldsp, [[1,0],[1,1]])

