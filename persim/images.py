from __future__ import division
from itertools import product
import collections

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import scipy.spatial as spatial
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin

__all__ = ["PersImage"]

class PersImage(TransformerMixin):
    """ Initialize a persistence image generator.

    Parameters
    -----------

    pixels : pair of ints like (int, int)
        Tuple representing number of pixels in return image along x and y axis.
    spread : float
        Standard deviation of gaussian kernel
    specs : dict
        Parameters for shape of image with respect to diagram domain. This is used if you would like images to have a particular range. Shaped like 
        ::
        
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


    Usage
    ------


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
        """ Convert diagram or list of diagrams to a persistence image.

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
            singular = not isinstance(diagrams[0][0], collections.Iterable)
        except IndexError:
            singular = False

        if singular:
            diagrams = [diagrams]

        dgs = [np.copy(diagram, np.float64) for diagram in diagrams]
        landscapes = [PersImage.to_landscape(dg) for dg in dgs]

        if not self.specs:
            self.specs = {
                "maxBD": np.max([np.max(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
                "minBD": np.min([np.min(np.vstack((landscape, np.zeros((1, 2))))) 
                                 for landscape in landscapes] + [0]),
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
        """ Define a weighting function, 
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
            """ This is the function defined as w_b(t) in the original PI paper

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
        """ This will return whatever kind of kernel we want to use.
            Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1
        """
        # TODO: use self.kernel_type to choose function

        def gaussian(data, pixel):
            return mvn.pdf(data, mean=pixel, cov=spread)

        return gaussian

    @staticmethod
    def to_landscape(diagram):
        """ Convert a diagram to a landscape
            (b,d) -> (b, d-b)
        """
        diagram[:, 1] -= diagram[:, 0]

        return diagram

    def show(self, imgs, ax=None):
        """ Visualize the persistence image

        """

        ax = ax or plt.gca()

        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            ax.imshow(img, cmap=plt.get_cmap("plasma"))
            ax.axis("off")
