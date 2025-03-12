from __future__ import division

import copy
from itertools import product
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
from deprecated.sphinx import deprecated
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from sklearn.base import TransformerMixin

from persim import images_kernels, images_weights

__all__ = ["PersImage", "PersistenceImager"]


@deprecated(
    reason="""Replaced with the class :class:`persim.PersistenceImager`.""",
    version="0.1.5",
)
class PersImage(TransformerMixin):
    """Initialize a persistence image generator.

    Parameters
    ----------

        pixels : pair of ints like (int, int)
            Tuple representing number of pixels in return image along x and y axis.
        spread : float
            Standard deviation of gaussian kernel.
        specs : dict
            Parameters for shape of image with respect to diagram domain. This is used if you would like images to have a particular range. Shaped like::

                {
                    "maxBD": float,
                    "minBD": float
                }

        kernel_type : string or ...
            TODO: Implement this feature.
            Determine which type of kernel used in the convolution, or pass in custom kernel. Currently only implements Gaussian.
        weighting_type : string or ...
            TODO: Implement this feature.
            Determine which type of weighting function used, or pass in custom weighting function.
            Currently only implements linear weighting.
    """

    def __init__(
        self,
        pixels=(20, 20),
        spread=None,
        specs=None,
        kernel_type="gaussian",
        weighting_type="linear",
        verbose=True,
    ):
        self.specs = specs
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread
        self.nx, self.ny = pixels

        if verbose:
            print(
                'PersImage(pixels={}, spread={}, specs={}, kernel_type="{}", weighting_type="{}")'.format(
                    pixels, spread, specs, kernel_type, weighting_type
                )
            )

    def transform(self, diagrams):
        """Convert diagram or list of diagrams to a persistence image.

        Parameters
        -----------

        diagrams : list of or singleton diagram, list of pairs. [(birth, death)]
            Persistence diagrams to be converted to persistence images. It is assumed they are in (birth, death) format. Can input a list of diagrams or a single diagram.

        """
        # if diagram is empty, return empty image
        if len(diagrams) == 0:
            return np.zeros((self.nx, self.ny))
        # if first entry of first entry is not iterable, then diagrams is singular and we need to make it a list of diagrams
        try:
            singular = not isinstance(diagrams[0][0], Iterable)
        except IndexError:
            singular = False

        if singular:
            diagrams = [diagrams]

        dgs = [np.copy(diagram) for diagram in diagrams]
        landscapes = [PersImage.to_landscape(dg) for dg in dgs]

        if not self.specs:
            self.specs = {
                "maxBD": np.max(
                    [
                        np.max(np.vstack((landscape, np.zeros((1, 2)))))
                        for landscape in landscapes
                    ]
                    + [0]
                ),
                "minBD": np.min(
                    [
                        np.min(np.vstack((landscape, np.zeros((1, 2)))))
                        for landscape in landscapes
                    ]
                    + [0]
                ),
            }
        imgs = [self._transform(dgm) for dgm in landscapes]

        # Make sure we return one item.
        if singular:
            imgs = imgs[0]

        return imgs

    def _transform(self, landscape):
        # Define an NxN grid over our landscape
        maxBD = self.specs["maxBD"]
        minBD = min(self.specs["minBD"], 0)  # at least show 0, maybe lower

        # Same bins in x and y axis
        dx = maxBD / (self.ny)
        xs_lower = np.linspace(minBD, maxBD, self.nx)
        xs_upper = np.linspace(minBD, maxBD, self.nx) + dx

        ys_lower = np.linspace(0, maxBD, self.ny)
        ys_upper = np.linspace(0, maxBD, self.ny) + dx

        weighting = self.weighting(landscape)

        # Define zeros
        img = np.zeros((self.nx, self.ny))

        # Implement this as a `summed-area table` - it'll be way faster
        spread = self.spread if self.spread else dx
        for point in landscape:
            x_smooth = norm.cdf(xs_upper, point[0], spread) - norm.cdf(
                xs_lower, point[0], spread
            )
            y_smooth = norm.cdf(ys_upper, point[1], spread) - norm.cdf(
                ys_lower, point[1], spread
            )
            img += np.outer(x_smooth, y_smooth) * weighting(point)
        img = img.T[::-1]
        return img

    def weighting(self, landscape=None):
        """Define a weighting function,
        for stability results to hold, the function must be 0 at y=0.
        """

        # TODO: Implement a logistic function
        # TODO: use self.weighting_type to choose function

        if landscape is not None:
            if len(landscape) > 0:
                maxy = np.max(landscape[:, 1])
            else:
                maxy = 1

        def linear(interval):
            # linear function of y such that f(0) = 0 and f(max(y)) = 1
            d = interval[1]
            return (1 / maxy) * d if landscape is not None else d

        def pw_linear(interval):
            """This is the function defined as w_b(t) in the original PI paper

            Take b to be maxy/self.ny to effectively zero out the bottom pixel row
            """

            t = interval[1]
            b = maxy / self.ny

            if t <= 0:
                return 0
            if 0 < t < b:
                return t / b
            if b <= t:
                return 1

        return linear

    def kernel(self, spread=1):
        """This will return whatever kind of kernel we want to use.
        Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1
        """
        # TODO: use self.kernel_type to choose function

        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)

        return gaussian

    @staticmethod
    def to_landscape(diagram):
        """Convert a diagram to a landscape
        (b,d) -> (b, d-b)
        """
        diagram[:, 1] -= diagram[:, 0]

        return diagram

    def show(self, imgs, ax=None):
        """Visualize the persistence image"""

        ax = ax or plt.gca()

        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")


class PersistenceImager(TransformerMixin):
    """Transformer which converts persistence diagrams into persistence images.

    Parameters
    ----------
    birth_range : pair of floats
        Range of persistence pair birth values covered by the persistence image (default: (0.0, 1.0)).
    pers_range : pair of floats
        Range of persistence pair persistence (death-birth) values covered by the persistence image (default: (0.0, 1.0)).
    pixel_size : float
        Dimensions of each square pixel (default: 0.2).
    weight : callable or str in ['persistence', 'linear_ramp']
        Function which weights the birth-persistence plane (default: 'persistence').
    weight_params : dict
        Arguments needed to specify the weight function (default: {'n': 1.0}).
    kernel : callable or str in ['gaussian', 'uniform']
        Cumulative distribution function defining the kernel (default: 'gaussian').
    kernel_params : dict
        Arguments needed to specify the kernel function (default: {'sigma': [[1.0, 0.0], [0.0, 1.0]]}).


    Example
    -------
    First instantiate a PersistenceImager() object::

        > from persim import PersistenceImager
        > pimgr = PersistenceImager(pixel_size=0.2, birth_range=(0,1))


    Printing a PersistenceImager() object will print its hyperparameters::

        > print(pimgr)

        PersistenceImager(birth_range=(0.0, 1.0), pers_range=(0.0, 1.0), pixel_size=0.2, weight=persistence, weight_params={'n': 1.0}, kernel=gaussian, kernel_params={'sigma': [[1.0, 0.0], [0.0, 1.0]]})


    PersistenceImager() attributes can be adjusted at or after instantiation. Updating attributes of a PersistenceImager() object will automatically update all other dependent attributes::

        > pimgr.pixel_size = 0.1
        > pimgr.birth_range = (0, 2)
        > print(pimgr)
        > print(pimgr.resolution)

        PersistenceImager(birth_range=(0.0, 2.0), pers_range=(0.0, 1.0), pixel_size=0.1, weight=persistence, weight_params={'n': 1.0}, kernel=gaussian, kernel_params={'sigma': [[1.0, 0.0], [0.0, 1.0]]})
        (20, 10)


    The `fit()` method can be called on one or more (-,2) numpy.ndarrays to automatically determine the miniumum birth and persistence ranges needed to capture all persistence pairs. The ranges and resolution are automatically adjusted to accomodate the specified pixel size. The option `skew=True` specifies that the diagram is currently in birth-death coordinates and must first be transformed to birth-persistence coordinates::

        > import numpy as np
        > pimgr = PersistenceImager(pixel_size=0.5)
        > pdgms = [np.array([[0.5, 0.8], [0.7, 2.2], [2.5, 4.0]]),
                   np.array([[0.1, 0.2], [3.1, 3.3], [1.6, 2.9]]),
                   np.array([[0.2, 1.5], [0.4, 0.6], [0.2, 2.6]])]
        > pimgr.fit(pdgms, skew=True)
        > print(pimgr)
        > print(pimgr.resolution)

        PersistenceImager(birth_range=(0.1, 3.1), pers_range=(-8.326672684688674e-17, 2.5), pixel_size=0.5, weight=persistence, weight_params={'n': 1.0}, kernel=gaussian, kernel_params={'sigma': [[1.0, 0.0], [0.0, 1.0]]})
        (6, 5)


    The `transform()` method can then be called on one or more (-,2) numpy.ndarrays to generate persistence images from diagrams. The option `skew=True` specifies that the diagrams are currently in birth-death coordinates and must first be transformed to birth-persistence coordinates::

        > pimgs = pimgr.transform(pdgms, skew=True)
        > pimgs[0]

        array([[0.03999068, 0.05688393, 0.06672051, 0.06341749, 0.04820814],
               [0.04506697, 0.06556791, 0.07809764, 0.07495246, 0.05730671],
               [0.04454486, 0.06674611, 0.08104366, 0.07869919, 0.06058808],
               [0.04113063, 0.0636504 , 0.07884635, 0.07747833, 0.06005714],
               [0.03625436, 0.05757744, 0.07242608, 0.07180125, 0.05593626],
               [0.02922239, 0.04712024, 0.05979033, 0.05956698, 0.04653357]])


    Notes
    -----
    [1] Adams et. al., "Persistence Images: A Stable Vector Representation of Persistent Homology," Journal of Machine Learning Research, vol. 18, pp. 1-35, 2017. http://www.jmlr.org/papers/volume18/16-337/16-337.pdf
    """

    def __init__(
        self,
        birth_range=None,
        pers_range=None,
        pixel_size=None,
        weight=None,
        weight_params=None,
        kernel=None,
        kernel_params=None,
    ):
        """PersistenceImager constructor method"""
        # set defaults
        if birth_range is None:
            birth_range = (0.0, 1.0)
        if pers_range is None:
            pers_range = (0.0, 1.0)
        if pixel_size is None:
            pixel_size = 0.2
        if weight is None:
            weight = images_weights.persistence
        if kernel is None:
            kernel = images_kernels.gaussian
        if weight_params is None:
            weight_params = {"n": 1.0}
        if kernel_params is None:
            kernel_params = {"sigma": [[1.0, 0.0], [0.0, 1.0]]}

        # validate parameters
        self._validate_parameters(
            birth_range=birth_range,
            pers_range=pers_range,
            pixel_size=pixel_size,
            weight=weight,
            weight_params=weight_params,
            kernel=kernel,
            kernel_params=kernel_params,
        )

        self.weight, self.kernel = self._ensure_callable(weight=weight, kernel=kernel)
        self.weight_params = weight_params
        self.kernel_params = kernel_params
        self._pixel_size = pixel_size
        self._birth_range = birth_range
        self._pers_range = pers_range
        self._width = birth_range[1] - birth_range[0]
        self._height = pers_range[1] - pers_range[0]
        self._resolution = (
            int(self._width / self._pixel_size),
            int(self._height / self._pixel_size),
        )
        self._create_mesh()

    @property
    def width(self):
        """
        Persistence image width.

        Returns
        -------
        width : float
            The width of the region of the birth-persistence plane covered by the persistence image in birth units.
        """
        return self._width

    @property
    def height(self):
        """
        Persistence image height.

        Returns
        -------
        height : float
            The height of the region of the birth-persistence plane covered by the persistence image in persistence units.
        """
        return self._height

    @property
    def resolution(self):
        """
        Persistence image resolution.

        Returns
        -------
        resolution : pair of ints (width, height)
            The number of pixels along each dimension of the persistence image, determined by the birth and persistence ranges and the pixel size.
        """
        return self._resolution

    @property
    def pixel_size(self):
        """
        Persistence image square pixel dimensions.

        Returns
        -------
        pixel_size : float
            The width (and height) in birth/persistence units of each square pixel in the persistence image.
        """
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, val):
        self._pixel_size = val
        self._width = (
            int(np.ceil((self.birth_range[1] - self.birth_range[0]) / self.pixel_size))
            * self.pixel_size
        )
        self._height = (
            int(np.ceil((self.pers_range[1] - self.pers_range[0]) / self.pixel_size))
            * self.pixel_size
        )
        self._resolution = (
            int(self.width / self.pixel_size),
            int(self.height / self.pixel_size),
        )
        self._create_mesh()

    @property
    def birth_range(self):
        """
        Range of birth values covered by the persistence image.

        Returns
        -------
        birth_range : pair of floats (min. birth, max. birth)
            The minimum and maximum birth values covered by the persistence image.
        """
        return self._birth_range

    @birth_range.setter
    def birth_range(self, val):
        self._birth_range = val
        self._width = (
            int(np.ceil((self.birth_range[1] - self.birth_range[0]) / self.pixel_size))
            * self._pixel_size
        )
        self._resolution = (
            int(self.width / self.pixel_size),
            int(self.height / self.pixel_size),
        )
        self._create_mesh()

    @property
    def pers_range(self):
        """
        Range of persistence values covered by the persistence image.

        Returns
        -------
        pers_range : pair of floats (min. persistence, max. persistence)
            The minimum and maximum persistence values covered by the persistence image.
        """
        return self._pers_range

    @pers_range.setter
    def pers_range(self, val):
        self._pers_range = val
        self._height = (
            int(np.ceil((self.pers_range[1] - self.pers_range[0]) / self.pixel_size))
            * self._pixel_size
        )
        self._resolution = (
            int(self.width / self.pixel_size),
            int(self.height / self.pixel_size),
        )
        self._create_mesh()

    def __repr__(self):
        import pprint as pp

        params = tuple(
            [
                self.birth_range,
                self.pers_range,
                self.pixel_size,
                self.weight.__name__,
                pp.pformat(self.weight_params),
                self.kernel.__name__,
                pp.pformat(self.kernel_params),
            ]
        )
        repr_str = (
            "PersistenceImager(birth_range=%s, pers_range=%s, pixel_size=%s, weight=%s, weight_params=%s, kernel=%s, kernel_params=%s)"
            % params
        )
        return repr_str

    def _ensure_callable(self, weight=None, kernel=None):
        valid_weights_dict = {
            "persistence": images_weights.persistence,
            "linear_ramp": images_weights.linear_ramp,
        }
        valid_kernels_dict = {
            "gaussian": images_kernels.gaussian,
            "uniform": images_kernels.uniform,
        }

        if isinstance(weight, str):
            weight = valid_weights_dict[weight]

        if isinstance(kernel, str):
            kernel = valid_kernels_dict[kernel]

        return weight, kernel

    def _validate_parameters(
        self,
        birth_range=None,
        pers_range=None,
        pixel_size=None,
        weight=None,
        weight_params=None,
        kernel=None,
        kernel_params=None,
    ):
        valid_weights = ["persistence", "linear_ramp"]
        valid_kernels = ["gaussian", "uniform"]

        # validate birth_range
        if isinstance(birth_range, tuple):
            if len(birth_range) != 2:
                raise ValueError("birth_range must be a pair: (min. birth, max. birth)")
            elif (not isinstance(birth_range[0], (int, float))) or (
                not isinstance(birth_range[1], (int, float))
            ):
                raise ValueError(
                    "birth_range must be a pair of numbers: (min. birth, max. birth)"
                )
        else:
            raise ValueError("birth_range must be a tuple")

        # validate pers_range
        if isinstance(pers_range, tuple):
            if len(pers_range) != 2:
                raise ValueError(
                    "pers_range must be a pair: (min. persistence, max. persistence)"
                )
            elif (not isinstance(pers_range[0], (int, float))) or (
                not isinstance(pers_range[1], (int, float))
            ):
                raise ValueError(
                    "pers_range must be a pair of numbers: (min. persistence, max. persistence)"
                )
        else:
            raise ValueError("pers_range must be a tuple")

        # validate pixel_size
        if not isinstance(pixel_size, (int, float)):
            raise ValueError("pixel_size must be an int or float")

        # validate weight
        if not callable(weight):
            if isinstance(weight, str):
                if weight not in ["persistence", "linear_ramp"]:
                    raise ValueError(
                        "weight must be callable or a str in %s" % valid_weights
                    )
            else:
                raise ValueError("weight must be callable or a str")

        # validate weight_params
        if not isinstance(weight_params, dict):
            raise ValueError("weight_params must be a dict")

        # validate kernel
        if not callable(kernel):
            if isinstance(kernel, str):
                if kernel not in valid_kernels:
                    raise ValueError(
                        "kernel must be callable or a str in %s" % valid_kernels
                    )
            else:
                raise ValueError("kernel must be callable or a str")

        # validate kernel_params
        if not isinstance(kernel_params, dict):
            raise ValueError("kernel_params must be a dict")

    def _create_mesh(self):
        # padding around specified image ranges as a result of incommensurable ranges and pixel width
        db = self._width - (self._birth_range[1] - self._birth_range[0])
        dp = self._height - (self._pers_range[1] - self._pers_range[0])

        # adjust image ranges to accommodate incommensurable ranges and pixel width
        self._birth_range = (
            self._birth_range[0] - db / 2,
            self._birth_range[1] + db / 2,
        )
        self._pers_range = (self._pers_range[0] - dp / 2, self._pers_range[1] + dp / 2)

        # construct linear spaces defining pixel locations
        self._bpnts = np.linspace(
            self._birth_range[0],
            self._birth_range[1] + self._pixel_size,
            self._resolution[0] + 1,
            endpoint=False,
            dtype=np.float64,
        )
        self._ppnts = np.linspace(
            self._pers_range[0],
            self._pers_range[1] + self._pixel_size,
            self._resolution[1] + 1,
            endpoint=False,
            dtype=np.float64,
        )

    def fit(self, pers_dgms, skew=True):
        """Choose persistence image range parameters which minimally enclose all persistence pairs across one or more persistence diagrams.

        Parameters
        ----------
        pers_dgms : one or an iterable of (-,2) numpy.ndarrays
            Collection of one or more persistence diagrams.
        skew : boolean
            Flag indicating if diagram(s) need to first be converted to birth-persistence coordinates (default: True).
        """
        min_birth = np.inf
        max_birth = -np.inf
        min_pers = np.inf
        max_pers = -np.inf

        # convert to a list of diagrams if necessary
        pers_dgms, singular = self._ensure_iterable(pers_dgms)

        # loop over diagrams to determine the maximum extent of the pairs contained in the birth-persistence plane
        for pers_dgm in pers_dgms:
            pers_dgm = np.copy(pers_dgm)
            if skew:
                pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]

            min_b, min_p = pers_dgm.min(axis=0)
            max_b, max_p = pers_dgm.max(axis=0)

            if min_b < min_birth:
                min_birth = min_b

            if min_p < min_pers:
                min_pers = min_p

            if max_b > max_birth:
                max_birth = max_b

            if max_p > max_pers:
                max_pers = max_p

        self.birth_range = (min_birth, max_birth)
        self.pers_range = (min_pers, max_pers)

    def transform(self, pers_dgms, skew=True, n_jobs=None):
        """Transform a persistence diagram or an iterable containing a collection of persistence diagrams into
        persistence images.

        Parameters
        ----------
        pers_dgms : one or an iterable of (-,2) numpy.ndarrays
            Collection of one or more persistence diagrams.
        skew : boolean
            Flag indicating if diagram(s) need to first be converted to birth-persistence coordinates (default: True).
        n_jobs : int
            Number of cores to use to transform a collection of persistence diagrams into persistence images (default: None, uses a single core).

        Returns
        -------
        list
            Collection of numpy.ndarrays encoding the persistence images in the same order as pers_dgms.
        """
        if n_jobs is not None:
            parallelize = True
        else:
            parallelize = False

        # if diagram is empty, return empty image
        if len(pers_dgms) == 0:
            return np.zeros(self.resolution)

        # convert to a list of diagrams if necessary
        pers_dgms, singular = self._ensure_iterable(pers_dgms)

        if parallelize:
            pers_imgs = Parallel(n_jobs=n_jobs)(
                delayed(_transform)(
                    pers_dgm,
                    skew,
                    self.resolution,
                    self.weight,
                    self.weight_params,
                    self.kernel,
                    self.kernel_params,
                    self._bpnts,
                    self._ppnts,
                )
                for pers_dgm in pers_dgms
            )
        else:
            pers_imgs = [
                _transform(
                    pers_dgm,
                    skew=skew,
                    resolution=self.resolution,
                    weight=self.weight,
                    weight_params=self.weight_params,
                    kernel=self.kernel,
                    kernel_params=self.kernel_params,
                    _bpnts=self._bpnts,
                    _ppnts=self._ppnts,
                )
                for pers_dgm in pers_dgms
            ]

        if singular:
            pers_imgs = pers_imgs[0]

        return pers_imgs

    def fit_transform(self, pers_dgms, skew=True):
        """Choose persistence image range parameters which minimally enclose all persistence pairs across one or more persistence diagrams and transform the persistence diagrams into persistence images.

        Parameters
        ----------
        pers_dgms : one or an iterable of (-,2) numpy.ndarray
            Collection of one or more persistence diagrams.
        skew : boolean
            Flag indicating if diagram(s) need to first be converted to birth-persistence coordinates (default: True).


        Returns
        -------
        list
            Collection of numpy.ndarrays encoding the persistence images in the same order as pers_dgms.
        """
        pers_dgms = copy.deepcopy(pers_dgms)

        # fit imager parameters
        self.fit(pers_dgms, skew=skew)

        # transform diagrams to images
        pers_imgs = self.transform(pers_dgms, skew=skew)

        return pers_imgs

    def _ensure_iterable(self, pers_dgms):
        # if first entry of first entry is not iterable, then diagrams is singular and we need to make it a list of diagrams
        try:
            singular = not isinstance(pers_dgms[0][0], Iterable)
        except IndexError:
            singular = False

        if singular:
            pers_dgms = [pers_dgms]

        return pers_dgms, singular

    def plot_diagram(self, pers_dgm, skew=True, ax=None, out_file=None):
        """Plot a persistence diagram.

        Parameters
        ----------
        pers_dgm : (-,2) numpy.ndarray
            A persistence diagram.
        skew : boolean
            Flag indicating if diagram needs to first be converted to birth-persistence coordinates (default: True).
        ax : matplotlib.Axes
            Instance of a matplotlib.Axes object in which to plot (default: None, generates a new figure)
        out_file : str
            Path and file name including extension to save the figure (default: None, figure not saved).

        Returns
        -------
        matplotlib.Axes
            The matplotlib.Axes which contains the persistence diagram
        """
        pers_dgm = np.copy(pers_dgm)

        if skew:
            pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]
            ylabel = "persistence"
        else:
            ylabel = "death"

        # setup plot range
        plot_buff_frac = 0.05
        bmin = np.min((np.min(pers_dgm[:, 0]), np.min(self._bpnts)))
        bmax = np.max((np.max(pers_dgm[:, 0]), np.max(self._bpnts)))
        b_plot_buff = (bmax - bmin) * plot_buff_frac
        bmin -= b_plot_buff
        bmax += b_plot_buff

        pmin = np.min((np.min(pers_dgm[:, 1]), np.min(self._ppnts)))
        pmax = np.max((np.max(pers_dgm[:, 1]), np.max(self._ppnts)))
        p_plot_buff = (pmax - pmin) * plot_buff_frac
        pmin -= p_plot_buff
        pmax += p_plot_buff

        ax = ax or plt.gca()
        ax.set_xlim(bmin, bmax)
        ax.set_ylim(pmin, pmax)

        # compute reasonable line width for pixel overlay (initially 1/50th of the width of a pixel)
        linewidth = (
            (1 / 50 * self.pixel_size)
            * 72
            * plt.gcf().bbox_inches.width
            * ax.get_position().width
            / np.min((bmax - bmin, pmax - pmin))
        )

        # plot the persistence image grid
        if skew:
            hlines = np.column_stack(
                np.broadcast_arrays(
                    self._bpnts[0], self._ppnts, self._bpnts[-1], self._ppnts
                )
            )
            vlines = np.column_stack(
                np.broadcast_arrays(
                    self._bpnts, self._ppnts[0], self._bpnts, self._ppnts[-1]
                )
            )
            lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
            line_collection = LineCollection(lines, color="black", linewidths=linewidth)
            ax.add_collection(line_collection)

        # plot persistence diagram
        ax.scatter(pers_dgm[:, 0], pers_dgm[:, 1])

        # plot diagonal if necessary
        if not skew:
            min_diag = np.min((np.min(ax.get_xlim()), np.min(ax.get_ylim())))
            max_diag = np.min((np.max(ax.get_xlim()), np.max(ax.get_ylim())))
            ax.plot([min_diag, max_diag], [min_diag, max_diag])

        # fix and label axes
        ax.set_aspect("equal")
        ax.set_xlabel("birth")
        ax.set_ylabel(ylabel)

        # optionally save figure
        if out_file:
            plt.savefig(out_file, bbox_inches="tight")

        return ax

    def plot_image(self, pers_img, ax=None, out_file=None):
        """Plot a persistence image.

        Parameters
        ----------
        pers_img : (M,N) numpy.ndarray
            A persistence image, as output by PersistenceImager().transform()
        ax : matplotlib.Axes
            Instance of a matplotlib.Axes object in which to plot (default: None, generates a new figure)
        out_file : str
            Path and file name including extension to save the figure (default: None, figure not saved).

        Returns
        -------
        matplotlib.Axes
            The matplotlib.Axes which contains the persistence image
        """
        ax = ax or plt.gca()
        ax.matshow(pers_img.T, **{"origin": "lower"})

        # fix and label axes
        ax.set_xlabel("birth")
        ax.set_ylabel("persistence")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # optionally save figure
        if out_file:
            plt.savefig(out_file, bbox_inches="tight")

        return ax


def _transform(
    pers_dgm,
    skew=True,
    resolution=None,
    weight=None,
    weight_params=None,
    kernel=None,
    kernel_params=None,
    _bpnts=None,
    _ppnts=None,
):
    """Transform a persistence diagram into a persistence image.

    Parameters
    ----------
    pers_dgm : (-,2) numpy.ndarray
        A persistence diagrams.
    skew : boolean
        Flag indicating if diagram(s) need to first be converted to birth-persistence coordinates (default: True).
    resolution : pair of ints
        The number of pixels along the birth and persistence axes in the persistence image.
    weight : callable
        Function which weights the birth-persistence plane.
    weight_params : dict
        Arguments needed to specify the weight function.
    kernel : callable
        Cumulative distribution function defining the kernel.
    kernel_params : dict
        Arguments needed to specify the kernel function.
    _bpnts : (N,) numpy.ndarray
        The birth coordinates of the persistence image pixel locations.
    _ppnts : (M,) numpy.ndarray
        The persistence coordinates of the persistence image pixel locations.

    Returns
    -------
    numpy.ndarray
        (M,N) numpy.ndarray encoding the persistence image corresponding to pers_dgm.
    """
    pers_dgm = np.copy(pers_dgm)
    pers_img = np.zeros(resolution)
    n = pers_dgm.shape[0]
    general_flag = True

    # if necessary convert from birth-death coordinates to birth-persistence coordinates
    if skew:
        pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]

    # compute weights at each persistence pair
    wts = weight(pers_dgm[:, 0], pers_dgm[:, 1], **weight_params)

    # handle the special case of a standard, isotropic Gaussian kernel
    if kernel == images_kernels.gaussian:
        general_flag = False
        sigma = kernel_params["sigma"]

        # sigma is specified by a single variance
        if isinstance(sigma, (int, float)):
            sigma = np.array([[sigma, 0.0], [0.0, sigma]], dtype=np.float64)

        if sigma[0][0] == sigma[1][1] and sigma[0][1] == 0.0:
            sigma = np.sqrt(sigma[0][0])
            for i in range(n):
                ncdf_b = images_kernels.norm_cdf((_bpnts - pers_dgm[i, 0]) / sigma)
                ncdf_p = images_kernels.norm_cdf((_ppnts - pers_dgm[i, 1]) / sigma)
                curr_img = ncdf_p[None, :] * ncdf_b[:, None]
                pers_img += wts[i] * (
                    curr_img[1:, 1:]
                    - curr_img[:-1, 1:]
                    - curr_img[1:, :-1]
                    + curr_img[:-1, :-1]
                )
        else:
            general_flag = True

    # handle the general case
    if general_flag:
        bb, pp = np.meshgrid(_bpnts, _ppnts, indexing="ij")
        bb = bb.flatten(order="C")
        pp = pp.flatten(order="C")
        for i in range(n):
            curr_img = np.reshape(
                kernel(bb, pp, mu=pers_dgm[i, :], **kernel_params),
                (resolution[0] + 1, resolution[1] + 1),
                order="C",
            )
            pers_img += wts[i] * (
                curr_img[1:, 1:]
                - curr_img[:-1, 1:]
                - curr_img[1:, :-1]
                + curr_img[:-1, :-1]
            )

    return pers_img
