import pytest

import numpy as np
from persim import PersImage

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
    

class TestKernels:
    def test_kernel_mean(self):
        pim = PersImage()
        kf = pim.kernel()

        data = np.array([[0,0]])
        assert kf(np.array([[0,0]]), [0,0]) >= kf(np.array([[1,1]]), [0,0]), "decreasing away"
        assert kf(np.array([[0,0]]), [1,1]) == kf(np.array([[1,1]]), [0,0]), "symmetric"


class TestTransforms:
    def test_n_pixels(self):
        pim = PersImage(pixels=9)
        diagram = np.array([[0,1], [1,1],[3,5]])
        img = pim.transform(diagram)

        assert img.shape == (3,3)

    def test_multiple_diagrams(self):
        pim = PersImage(pixels=9)
        
        diagram1 = np.array([[0,1], [1,1],[3,5]])
        diagram2 = np.array([[0,1], [1,1],[3,6]])
        imgs = pim.transform([diagram1, diagram2])

        assert len(imgs) == 2
        assert imgs[0].shape == imgs[1].shape