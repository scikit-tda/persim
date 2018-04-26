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
        assert wf([99, 1.4]) == 1.4

    def test_scales(self):
        pim = PersImage()

        wf = pim.weighting(np.array([[0,1],[1,2],[3,4]]))

        assert wf([1,0]) == 0
        assert wf([1,4]) == 1
        assert wf([1,2]) == .5
    

class TestKernels:
    def test_kernel_mean(self):
        pim = PersImage()
        kf = pim.kernel(2)

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

@pytest.mark.skip()
class TestIntegration:
    """ We can't just take the center point, we need to integrate over the surface.

        It will be changing, so we need to ensure it works correctly.
    """
    def test_integrate_constant(self):
        intr = Integrator()
        
        f = lambda center:  1
        assert np.allclose(intr.integrate(f, [0], 2), 1 * (2*2)**2)
        
        f = lambda center:  2
        assert np.allclose(intr.integrate(f, [0], 2), 2 * (2*2)**2)
        
        f = lambda center:  3
        assert np.allclose(intr.integrate(f, [0], 2), 3 * (2*2)**2)

    def test_integrate_(self):
        # I am not sure these are correct
        intr = Integrator()
        
        f = lambda center:  center[0]
        
        assert np.allclose(intr.integrate(f, [0,10], 2), 32)
        
        f = lambda center:  center[0]
        assert np.allclose(intr.integrate(f, [1,10], 2), 32)
        
        f = lambda center:  center[0]
        assert np.allclose(intr.integrate(f, [2, 10], 2), 32)

    def test_coplanar(self):
        intr = Integrator()
        cube = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])

 
        vol = intr._convex_hull_volume_bis(cube)
        assert np.allclose(vol, 0)

    def test_convex_hull_vol(self):
        intr = Integrator()

        cube = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                           [0,0,1],[1,0,1],[0,1,1],[1,1,1]])

        vol = intr._convex_hull_volume_bis(cube)
        assert np.allclose(vol, 1)

        vol = intr._convex_hull_volume_bis(cube*2)
        assert np.allclose(vol, 8)
