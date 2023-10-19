import numpy as np
import pytest

from persim import PersistenceImager, images_kernels, images_weights

# ----------------------------------------
# New PersistenceImager Tests
# ----------------------------------------


def test_empty_diagram():
    dgm = np.zeros((0, 2))
    persimgr = PersistenceImager(pixel_size=0.1)
    res = persimgr.transform(dgm)
    np.testing.assert_array_equal(res, np.zeros((10, 10)))


def test_empty_diagram_list():
    dgms1 = [np.array([[2, 3]]), np.zeros((0, 2))]
    persimgr1 = PersistenceImager(pixel_size=0.1)
    res1 = persimgr1.transform(dgms1)
    np.testing.assert_array_equal(res1[1], np.zeros((10, 10)))

    dgms2 = [np.zeros((0, 2)), np.array([[2, 3]])]
    persimgr2 = PersistenceImager(pixel_size=0.1)
    res2 = persimgr2.transform(dgms2)
    np.testing.assert_array_equal(res2[0], np.zeros((10, 10)))

    dgms3 = [np.zeros((0, 2)), np.zeros((0, 2))]
    persimgr3 = PersistenceImager(pixel_size=0.1)
    res3 = persimgr3.transform(dgms3)
    np.testing.assert_array_equal(res3[0], np.zeros((10, 10)))
    np.testing.assert_array_equal(res3[1], np.zeros((10, 10)))


def test_birth_range_setter():
    persimgr = PersistenceImager(birth_range=(0, 1), pers_range=(0, 2), pixel_size=1)
    persimgr.birth_range = (0.0, 4.5)

    np.testing.assert_equal(persimgr.pixel_size, 1)
    np.testing.assert_equal(persimgr._pixel_size, 1)
    np.testing.assert_equal(persimgr.pers_range, (0, 2))
    np.testing.assert_equal(persimgr._pers_range, (0, 2))
    np.testing.assert_equal(persimgr.birth_range, (-0.25, 4.75))
    np.testing.assert_equal(persimgr._birth_range, (-0.25, 4.75))
    np.testing.assert_equal(persimgr.width, 5)
    np.testing.assert_equal(persimgr._width, 5)
    np.testing.assert_equal(persimgr.height, 2)
    np.testing.assert_equal(persimgr._height, 2)
    np.testing.assert_equal(persimgr.resolution, (5, 2))
    np.testing.assert_equal(persimgr._resolution, (5, 2))
    np.testing.assert_array_equal(
        persimgr._bpnts, [-0.25, 0.75, 1.75, 2.75, 3.75, 4.75]
    )
    np.testing.assert_array_equal(persimgr._ppnts, [0.0, 1.0, 2.0])


def test_pers_range_setter():
    persimgr = PersistenceImager(birth_range=(0, 1), pers_range=(0, 2), pixel_size=1)
    persimgr.pers_range = (-1.5, 4.5)

    np.testing.assert_equal(persimgr.pixel_size, 1)
    np.testing.assert_equal(persimgr._pixel_size, 1)
    np.testing.assert_equal(persimgr.pers_range, (-1.5, 4.5))
    np.testing.assert_equal(persimgr._pers_range, (-1.5, 4.5))
    np.testing.assert_equal(persimgr.birth_range, (0, 1))
    np.testing.assert_equal(persimgr._birth_range, (0, 1))
    np.testing.assert_equal(persimgr.width, 1)
    np.testing.assert_equal(persimgr._width, 1)
    np.testing.assert_equal(persimgr.height, 6)
    np.testing.assert_equal(persimgr._height, 6)
    np.testing.assert_equal(persimgr.resolution, (1, 6))
    np.testing.assert_equal(persimgr._resolution, (1, 6))
    np.testing.assert_array_equal(
        persimgr._ppnts, [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    )
    np.testing.assert_array_equal(persimgr._bpnts, [0.0, 1.0])


def test_pixel_size_setter():
    persimgr = PersistenceImager(birth_range=(0, 1), pers_range=(0, 2), pixel_size=1)
    persimgr.pixel_size = 0.75

    np.testing.assert_equal(persimgr.pixel_size, 0.75)
    np.testing.assert_equal(persimgr._pixel_size, 0.75)
    np.testing.assert_equal(persimgr.birth_range, (-0.25, 1.25))
    np.testing.assert_equal(persimgr._birth_range, (-0.25, 1.25))
    np.testing.assert_equal(persimgr.pers_range, (-0.125, 2.125))
    np.testing.assert_equal(persimgr._pers_range, (-0.125, 2.125))
    np.testing.assert_equal(persimgr.height, 2.25)
    np.testing.assert_equal(persimgr._height, 2.25)
    np.testing.assert_equal(persimgr.width, 1.5)
    np.testing.assert_equal(persimgr._width, 1.5)
    np.testing.assert_equal(persimgr.resolution, (2, 3))
    np.testing.assert_equal(persimgr._resolution, (2, 3))
    np.testing.assert_array_equal(persimgr._ppnts, [-0.125, 0.625, 1.375, 2.125])
    np.testing.assert_array_equal(persimgr._bpnts, [-0.25, 0.5, 1.25])


def test_fit_diagram():
    persimgr = PersistenceImager(birth_range=(0, 1), pers_range=(0, 2), pixel_size=1)
    dgm = np.array([[1, 2], [4, 8], [-1, 5.25]])
    persimgr.fit(dgm)

    np.testing.assert_equal(persimgr.pixel_size, 1)
    np.testing.assert_equal(persimgr._pixel_size, 1)
    np.testing.assert_equal(persimgr.birth_range, (-1, 4))
    np.testing.assert_equal(persimgr._birth_range, (-1, 4))
    np.testing.assert_equal(persimgr.pers_range, (0.625, 6.625))
    np.testing.assert_equal(persimgr._pers_range, (0.625, 6.625))
    np.testing.assert_equal(persimgr.height, 6)
    np.testing.assert_equal(persimgr._height, 6)
    np.testing.assert_equal(persimgr.width, 5)
    np.testing.assert_equal(persimgr._width, 5)
    np.testing.assert_equal(persimgr.resolution, (5, 6))
    np.testing.assert_equal(persimgr._resolution, (5, 6))
    np.testing.assert_array_equal(
        persimgr._ppnts, [0.625, 1.625, 2.625, 3.625, 4.625, 5.625, 6.625]
    )
    np.testing.assert_array_equal(persimgr._bpnts, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])


def test_fit_diagram_list():
    persimgr = PersistenceImager(birth_range=(0, 1), pers_range=(0, 2), pixel_size=1)
    dgms = [np.array([[1, 2], [4, 8], [-1, 5.25]]), np.array([[1, 2], [2, 3], [3, 4]])]
    persimgr.fit(dgms)

    np.testing.assert_equal(persimgr.pixel_size, 1)
    np.testing.assert_equal(persimgr._pixel_size, 1)
    np.testing.assert_equal(persimgr.birth_range, (-1, 4))
    np.testing.assert_equal(persimgr._birth_range, (-1, 4))
    np.testing.assert_equal(persimgr.pers_range, (0.625, 6.625))
    np.testing.assert_equal(persimgr._pers_range, (0.625, 6.625))
    np.testing.assert_equal(persimgr.height, 6)
    np.testing.assert_equal(persimgr._height, 6)
    np.testing.assert_equal(persimgr.width, 5)
    np.testing.assert_equal(persimgr._width, 5)
    np.testing.assert_equal(persimgr.resolution, (5, 6))
    np.testing.assert_equal(persimgr._resolution, (5, 6))
    np.testing.assert_array_equal(
        persimgr._ppnts, [0.625, 1.625, 2.625, 3.625, 4.625, 5.625, 6.625]
    )
    np.testing.assert_array_equal(persimgr._bpnts, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])


def test_mixed_pairs():
    """This test is inspired by gh issue #3 by gh user muszyna25.
    Integer diagrams return nan values.
    This does not work: dgm = [[0, 2], [0, 6], [0, 8]];
    This one works fine: dgm = [[0.0, 2.0], [0.0, 6.0], [0.0, 8.0]];
    """
    persimgr = PersistenceImager()

    dgm = [[0, 2], [0, 6], [0, 8]]
    dgm2 = [[0.0, 2.0], [0.0, 6.0], [0.0, 8.0]]
    dgm3 = [[0.0, 2], [0.0, 6.0], [0, 8.0e0]]

    res = persimgr.transform(dgm)
    res2 = persimgr.transform(dgm2)
    res3 = persimgr.transform(dgm3)

    np.testing.assert_array_equal(res, res2)
    np.testing.assert_array_equal(res, res3)


def test_parameter_exceptions():
    def construct_imager(param_dict):
        pimgr = PersistenceImager(**param_dict)

    np.testing.assert_raises(ValueError, construct_imager, {"birth_range": 0})
    np.testing.assert_raises(ValueError, construct_imager, {"birth_range": ("str", 0)})
    np.testing.assert_raises(ValueError, construct_imager, {"birth_range": (0, 0, 0)})
    np.testing.assert_raises(ValueError, construct_imager, {"pers_range": 0})
    np.testing.assert_raises(ValueError, construct_imager, {"pers_range": ("str", 0)})
    np.testing.assert_raises(ValueError, construct_imager, {"pers_range": (0, 0, 0)})
    np.testing.assert_raises(ValueError, construct_imager, {"pixel_size": "str"})
    np.testing.assert_raises(ValueError, construct_imager, {"weight": 0})
    np.testing.assert_raises(ValueError, construct_imager, {"weight": "invalid_weight"})
    np.testing.assert_raises(ValueError, construct_imager, {"kernel": 0})
    np.testing.assert_raises(ValueError, construct_imager, {"kernel": "invalid_kernel"})
    np.testing.assert_raises(ValueError, construct_imager, {"weight_params": 0})
    np.testing.assert_raises(ValueError, construct_imager, {"kernel_params": 0})


class TestWeightFunctions:
    def test_zero_on_birthaxis(self):
        persimgr = PersistenceImager(
            weight=images_weights.linear_ramp,
            weight_params={"low": 0.0, "high": 1.0, "start": 0.0, "end": 1.0},
        )
        wf = persimgr.weight
        wf_params = persimgr.weight_params
        np.testing.assert_equal(wf(1, 0, **wf_params), 0)

        persimgr = PersistenceImager(
            weight=images_weights.persistence, weight_params={"n": 2}
        )
        wf = persimgr.weight
        wf_params = persimgr.weight_params
        np.testing.assert_equal(wf(1, 0, **wf_params), 0)

    def test_linear_ramp(self):
        persimgr = PersistenceImager(
            weight=images_weights.linear_ramp,
            weight_params={"low": 0.0, "high": 5.0, "start": 0.0, "end": 1.0},
        )

        wf = persimgr.weight
        wf_params = persimgr.weight_params

        np.testing.assert_equal(wf(1, 0, **wf_params), 0)
        np.testing.assert_equal(wf(1, 1 / 5, **wf_params), 1)
        np.testing.assert_equal(wf(1, 1, **wf_params), 5)
        np.testing.assert_equal(wf(1, 2, **wf_params), 5)

        persimgr.weight_params = {"low": 0.0, "high": 5.0, "start": 0.0, "end": 5.0}
        wf_params = persimgr.weight_params

        np.testing.assert_equal(wf(1, 0, **wf_params), 0)
        np.testing.assert_equal(wf(1, 1 / 5, **wf_params), 1 / 5)
        np.testing.assert_equal(wf(1, 1, **wf_params), 1)
        np.testing.assert_equal(wf(1, 5, **wf_params), 5)

        persimgr.weight_params = {"low": 0.0, "high": 5.0, "start": 1.0, "end": 5.0}
        wf_params = persimgr.weight_params

        np.testing.assert_equal(wf(1, 0, **wf_params), 0)
        np.testing.assert_equal(wf(1, 1, **wf_params), 0)
        np.testing.assert_equal(wf(1, 5, **wf_params), 5)

        persimgr.weight_params = {"low": 1.0, "high": 5.0, "start": 1.0, "end": 5.0}
        wf_params = persimgr.weight_params
        np.testing.assert_equal(wf(1, 0, **wf_params), 1)
        np.testing.assert_equal(wf(1, 1, **wf_params), 1)
        np.testing.assert_equal(wf(1, 2, **wf_params), 2)

    def test_persistence(self):
        persimgr = PersistenceImager(
            weight=images_weights.persistence, weight_params={"n": 1.0}
        )

        wf = persimgr.weight
        wf_params = persimgr.weight_params

        x = np.random.rand()
        np.testing.assert_equal(wf(1, x, **wf_params), x)

        persimgr.weight_params = {"n": 1.5}
        wf_params = persimgr.weight_params

        np.testing.assert_equal(wf(1, x, **wf_params), x**1.5)


class TestKernelFunctions:
    def test_gaussian(self):
        kernel = images_kernels.gaussian
        kernel_params = {"mu": [1, 1], "sigma": np.array([[1, 0], [0, 1]])}
        np.testing.assert_almost_equal(
            kernel(np.array([1]), np.array([1]), **kernel_params), 1 / 4, 8
        )

        kernel = images_kernels.bvn_cdf
        kernel_params = {
            "mu_x": 1.0,
            "mu_y": 1.0,
            "sigma_xx": 1.0,
            "sigma_yy": 1.0,
            "sigma_xy": 0.0,
        }
        np.testing.assert_almost_equal(
            kernel(np.array([1]), np.array([1]), **kernel_params), 1 / 4, 8
        )

        kernel_params = {
            "mu_x": 1.0,
            "mu_y": 1.0,
            "sigma_xx": 1.0,
            "sigma_yy": 1.0,
            "sigma_xy": 0.5,
        }
        np.testing.assert_almost_equal(
            kernel(np.array([1]), np.array([1]), **kernel_params), 1 / 3, 8
        )

        kernel_params = {
            "mu_x": 1.0,
            "mu_y": 1.0,
            "sigma_xx": 1.0,
            "sigma_yy": 2.0,
            "sigma_xy": 0.0,
        }
        np.testing.assert_almost_equal(
            kernel(np.array([1]), np.array([0]), **kernel_params), 0.11987503, 8
        )

        kernel_params = {
            "mu_x": 1.0,
            "mu_y": 1.0,
            "sigma_xx": 1.0,
            "sigma_yy": 2.0,
            "sigma_xy": 1.0,
        }
        np.testing.assert_equal(
            kernel(np.array([1]), np.array([1]), **kernel_params), 0.375
        )

    def test_norm_cdf(self):
        np.testing.assert_equal(images_kernels.norm_cdf(0), 0.5)
        np.testing.assert_almost_equal(
            images_kernels.norm_cdf(1), 0.8413447460685429, 8
        )

    def test_uniform(self):
        kernel = images_kernels.uniform
        kernel_params = {"width": 3, "height": 1}

        np.testing.assert_equal(
            kernel(np.array([-1]), np.array([-1]), mu=(0, 0), **kernel_params), 0
        )
        np.testing.assert_equal(
            kernel(np.array([3]), np.array([1]), mu=(0, 0), **kernel_params), 1
        )
        np.testing.assert_equal(
            kernel(np.array([5]), np.array([5]), mu=(0, 0), **kernel_params), 1
        )

    def test_sigma(self):
        kernel = images_kernels.gaussian
        kernel_params1 = {"sigma": np.array([[1, 0], [0, 1]])}
        kernel_params2 = {"sigma": [[1, 0], [0, 1]]}
        np.testing.assert_equal(
            kernel(np.array([1]), np.array([1]), **kernel_params1),
            kernel(np.array([1]), np.array([1]), **kernel_params2),
        )


class TestTransformOutput:
    def test_lists_of_lists(self):
        persimgr = PersistenceImager(
            birth_range=(0, 3), pers_range=(0, 3), pixel_size=1
        )
        dgm = [[0, 1], [1, 1], [3, 5]]
        img = persimgr.transform(dgm)

        np.testing.assert_equal(img.shape, (3, 3))

    def test_n_pixels(self):
        persimgr = PersistenceImager(
            birth_range=(0, 5), pers_range=(0, 3), pixel_size=1
        )
        dgm = np.array([[0, 1], [1, 1], [3, 5]])
        img = persimgr.transform(dgm)

        np.testing.assert_equal(img.shape, (5, 3))

        img = persimgr.fit_transform(dgm)
        np.testing.assert_equal(img.shape, (3, 2))

    def test_multiple_diagrams(self):
        persimgr = PersistenceImager(
            birth_range=(0, 5), pers_range=(0, 3), pixel_size=1
        )

        dgm1 = np.array([[0, 1], [1, 1], [3, 5]])
        dgm2 = np.array([[0, 1], [1, 1], [3, 6], [1, 1]])
        imgs = persimgr.transform([dgm1, dgm2])

        np.testing.assert_equal(len(imgs), 2)
        np.testing.assert_equal(imgs[0].shape, imgs[1].shape)

        imgs = persimgr.fit_transform([dgm1, dgm2])
        np.testing.assert_equal(len(imgs), 2)
        np.testing.assert_equal(imgs[0].shape, imgs[1].shape)
        np.testing.assert_equal(imgs[0].shape, (3, 3))
