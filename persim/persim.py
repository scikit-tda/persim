from itertools import product

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import scipy.spatial as spatial
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator


class PersImage(BaseEstimator):
    def __init__(
        self,
        pixels=20 * 20,
        spread=None,
        specs=None,
        kernel_type="gaussian",
        weighting_type="linear",
    ):
        """ Initialize a persistence image generator.

            pixels

        """

        self.specs = specs
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread

        self.pixels = pixels
        self.N = int(np.sqrt(pixels))

        print(
            'PersImage(pixels={}, spread={}, specs={}, kernel_type="{}", weighting_type="{}")'.format(
                pixels, spread, specs, kernel_type, weighting_type
            )
        )

    def transform(self, diagrams):
        """ Convert diagram or list of diagrams to a persistence image
        """

        # TODO: allow for nonsquare images
        assert int(np.sqrt(self.pixels)) == np.sqrt(
            self.pixels
        ), "Pixels must be a square"

        if type(diagrams) is not list:
            dg = np.copy(diagrams)  # keep original diagram untouched
            landscape = PersImage.to_landscape(dg)

            if not self.specs:
                self.specs = {"maxBD": np.max(landscape), "minBD": np.min(landscape)}

            imgs = self._transform(landscape)
        else:
            dgs = [np.copy(diagram) for diagram in diagrams]
            landscapes = [PersImage.to_landscape(dg) for dg in dgs]

            if not self.specs:
                self.specs = {
                    "maxBD": np.max([np.max(landscape) for landscape in landscapes]),
                    "minBD": np.min([np.min(landscape) for landscape in landscapes]),
                }
            imgs = [self._transform(dgm) for dgm in landscapes]

        return imgs

    def _transform(self, landscape):
        """ Convert single diagram to a persistence image
        """

        N = self.N

        # Define an NxN grid over our landscape
        maxBD = self.specs["maxBD"]
        minBD = min(self.specs["minBD"], 0)  # at least show 0, maybe lower

        # Same bins in x and y axis
        dx = maxBD / (N)
        xs_lower = np.linspace(minBD, maxBD, N)
        xs_upper = np.linspace(minBD, maxBD, N) + dx

        ys_lower = np.linspace(0, maxBD, N)
        ys_upper = np.linspace(0, maxBD, N) + dx

        weighting = self.weighting(landscape)

        # Define zeros
        img = np.zeros((N, N))

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
            maxy = np.max(landscape[:, 1])

        def linear(interval):
            # linear function of y such that f(0) = 0 and f(max(y)) = 1
            d = interval[1]
            return (1 / maxy) * d if landscape is not None else d

        def pw_linear(interval):
            """ This is the function defined as w_b(t) in the original PI paper

                Take b to be maxy/self.N to effectively zero out the bottom pixel row
            """

            t = interval[1]
            b = maxy / self.N

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

    def show(self, imgs):
        """ Visualize the persistence image
        """

        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            plt.imshow(img, cmap=plt.get_cmap("plasma"))
            # plt.show()
