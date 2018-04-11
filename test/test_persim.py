import pytest

import numpy as np
from persimmon import PersImage

def test_landscape():
    bds = np.array([[1,1],[1,2]])

    ldsp = PersImage.to_landscape(bds)

    np.testing.assert_array_equal(ldsp, [[1,0],[1,1]])

class TestWeighting:
    def test_zero_on_xaxis(self):
        pim = PersImage()

        wf = pim.weighting()

        assert wf([1,0]) == 0
        assert wf([100,0]) == 0

    def test_scales(self):
        pim = PersImage()

        wf = pim.weighting(np.array([[0,1],[1,2],[3,4]]))

        assert wf([1,0]) == 0
        assert wf([1,4]) == 1
        assert wf([1,2]) == .5
    


