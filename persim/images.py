from __future__ import division
from itertools import product
import collections

from joblib import Parallel, delayed
from multiprocessing import Pool

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import scipy.spatial as spatial
from scipy.special import erfc
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.base import TransformerMixin

__all__ = ["PersImage", "PersistenceImager"]

"""
Implements class used to convert persistence diagrams into a finite-dimensional representation known as a persistence images, 
as introduced in Adams et al, 2017 (http://www.jmlr.org/papers/volume18/16-337/16-337.pdf) [PI].
"""

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

        dgs = [np.copy(diagram) for diagram in diagrams]
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


class PersistenceImager(TransformerMixin):
    """ Initialize a persistence image generator used to convert persistence diagrams into persistence images

    Parameters
    -----------

    birth_range : tuple specifying lower and upper birth values of the persistence image
    pers_range : tuple specifying lower and upper persistence values of the persistence image
    pixel_size : size of each square pixel
    weight : function to weight the birth-persistence plane
    weight_params : arguments needed to specify the weight function
    kernel : cumulative distribution function of kernel
    kernel_params : arguments needed to specify the kernel (cumulative distribution) function


    Usage
    ------

    First instantiate a PersistenceImager() object:
    ```
    >>> from persim import PersistenceImager
    >>> pimgr = PersistenceImager(pixel_size=0.2, birth_range=(0,1))
    ```
    
    Printing a PersistenceImager() object will print its hyperparameters (defaults in this case):
    ```
    >>> print(pimgr)

    PersistenceImager object:
      pixel size: 0.2
      resolution: (5, 5)
      birth range: (0, 1)
      persistence range: (0, 1)
      weight: persistence
      kernel: bvncdf
      weight parameters: {n: 1.0}
      kernel parameters: {sigma: [[ 1.  0.]
                                  [ 0.  1.]]}
    
    PersistenceImager() attributes can be adjusted at or after instantiation. Updating attributes of a PersistenceImager() object will automatically update all other dependent attributes.
    ```
    >>> pimgr.pixel_size = 0.1
    >>> pimgr.birth_range = (0, 2)
    >>> print(pimgr)
    
    PersistenceImager object: 
      pixel size: 0.1 
      resolution: (20, 10)  <---
      birth range: (0, 2) 
      persistence range: (0, 1) 
      weight: persistence 
      kernel: bvncdf 
      weight parameters: {n: 1.0} 
      kernel parameters: {sigma: [[1. 0.]
                                  [0. 1.]]}
    ```
    
    The `fit()` method can be called on one or more (*,2) numpy arrays to automatically determine the miniumum birth and persistence ranges needed to capture all persistence pairs. The ranges and resolution are automatically adjusted to accomodate the specified pixel size.
    
    ```
    >>> import numpy as np
    >>> pimgr = PersistenceImager(pixel_size=0.5)
    >>> pdgms = [np.array([[0.5, 0.8], [0.7, 2.2], [2.5, 4.0]]),
                 np.array([[0.1, 0.2], [3.1, 3.3], [1.6, 2.9]]),
                 np.array([[0.2, 1.5], [0.4, 0.6], [0.2, 2.6]])]
    >>> pimgr.fit(pdgms, skew=True)
    >>> pimgr
    
    PersistenceImager object: 
      pixel size: 0.5 
      resolution: (6, 5)                      <---
      birth range: (0.1, 3.1)                 <---
      persistence range: (-8.32667e-17, 2.5)  <---
      weight: persistence 
      kernel: bvncdf 
      weight parameters: {n: 1.0} 
      kernel parameters: {sigma: [[1. 0.]
                                  [0. 1.]]}
    
    ```
    
    The `transform()` method can then be called on one or more (*,2) numpy arrays to generate persistence images from diagrams. The option `skew=True` specifies that the diagrams are currently in birth-death coordinates and must first be transformed to birth-persistence coordinates.
    ```
    >>> pimgs = pimgr.transform(pdgms, skew=True)
    >>> pimgs[0]

        array([[0.03999068, 0.05688393, 0.06672051, 0.06341749, 0.04820814],
               [0.04506697, 0.06556791, 0.07809764, 0.07495246, 0.05730671],
               [0.04454486, 0.06674611, 0.08104366, 0.07869919, 0.06058808],
               [0.04113063, 0.0636504 , 0.07884635, 0.07747833, 0.06005714],
               [0.03625436, 0.05757744, 0.07242608, 0.07180125, 0.05593626],
               [0.02922239, 0.04712024, 0.05979033, 0.05956698, 0.04653357]])
    ```
    
    The option `skew=True` specifies that the diagram is currently in birth-death coordinates and must first be transformed to birth-persistence coordinates.
    """
    
    def __init__(self, birth_range=None, pers_range=None, pixel_size=None,
                 weight=None, weight_params=None, kernel=None, kernel_params=None):
        """
        Class for transforming persistence diagrams into persistence images
        """
        # set defaults
        if birth_range is None:
            birth_range = (0.0, 1.0)
        if pers_range is None:
            pers_range = (0.0, 1.0)
        if pixel_size is None:
            pixel_size = np.min([pers_range[1] - pers_range[0], birth_range[1] - birth_range[0]]) * 0.2
        if weight is None:
            weight = persistence
        if kernel is None:
            kernel = bvncdf
        if weight_params is None:
            weight_params = {'n': 1.0}
        if kernel_params is None:
            kernel_params = {'sigma': np.array([[1.0, 0.0], [0.0, 1.0]])}
       
        self.weight = weight
        self.weight_params = weight_params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self._pixel_size = pixel_size
        self._birth_range = birth_range
        self._pers_range = pers_range
        self._width = birth_range[1] - birth_range[0]
        self._height = pers_range[1] - pers_range[0]
        self._resolution = (int(self._width / self._pixel_size), int(self._height / self._pixel_size))
        self._create_mesh()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def resolution(self):
        return self._resolution

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, val):
        self._pixel_size = val
        self._width = int(np.ceil((self.birth_range[1] - self.birth_range[0]) / self.pixel_size)) * self.pixel_size
        self._height = int(np.ceil((self.pers_range[1] - self.pers_range[0]) / self.pixel_size)) * self.pixel_size
        self._resolution = (int(self.width / self.pixel_size), int(self.height / self.pixel_size))
        self._create_mesh()

    @property
    def birth_range(self):
        return self._birth_range

    @birth_range.setter
    def birth_range(self, val):
        self._birth_range = val
        self._width = int(np.ceil((self.birth_range[1] - self.birth_range[0]) / self.pixel_size)) * self._pixel_size
        self._resolution = (int(self.width / self.pixel_size), int(self.height / self.pixel_size))
        self._create_mesh()

    @property
    def pers_range(self):
        return self._pers_range

    @pers_range.setter
    def pers_range(self, val):
        self._pers_range = val
        self._height = int(np.ceil((self.pers_range[1] - self.pers_range[0]) / self.pixel_size)) * self._pixel_size
        self._resolution = (int(self.width / self.pixel_size), int(self.height / self.pixel_size))
        self._create_mesh()

    def __repr__(self):
        repr_str = 'PersistenceImager object: \n' +\
                   '  pixel size: %g \n' % self.pixel_size +\
                   '  resolution: (%d, %d) \n' % self.resolution +\
                   '  birth range: (%g, %g) \n' % self.birth_range +\
                   '  persistence range: (%g, %g) \n' % self.pers_range +\
                   '  weight: %s \n' % self.weight.__name__ +\
                   '  kernel: %s \n' % self.kernel.__name__ +\
                   '  weight parameters: %s \n' % dict_print(self.weight_params) +\
                   '  kernel parameters: %s' % dict_print(self.kernel_params)
        return repr_str

    def _create_mesh(self):
        # padding around specified image ranges as a result of incommensurable ranges and pixel width
        db = self._width - (self._birth_range[1] - self._birth_range[0])
        dp = self._height - (self._pers_range[1] - self._pers_range[0])

        # adjust image ranges to accommodate incommensurable ranges and pixel width
        self._birth_range = (self._birth_range[0] - db / 2, self._birth_range[1] + db / 2)
        self._pers_range = (self._pers_range[0] - dp / 2, self._pers_range[1] + dp / 2)
        # construct linear spaces defining pixel locations
        self._bpnts = np.linspace(self._birth_range[0], self._birth_range[1] + self._pixel_size,
                                           self._resolution[0] + 1, endpoint=False, dtype=np.float64)
        self._ppnts = np.linspace(self._pers_range[0], self._pers_range[1] + self._pixel_size,
                                           self._resolution[1] + 1, endpoint=False, dtype=np.float64)

    def fit(self, pers_dgms, skew=True):
        """
        Automatically choose persistence images parameters based on one or more persistence diagrams
        :param pers_dgms: one or an iterable of (N,2) numpy arrays encoding a persistence diagram
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        """
        min_birth = np.Inf
        max_birth = -np.Inf
        min_pers = np.Inf
        max_pers = -np.Inf
        
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
        """
        Transform a persistence diagram or a iterable containing a collection of persistence diagrams into 
        persistence images using the parameters specified in the PersistenceImager object instance
        :param pers_dgms: one or an iterable of (N,2) numpy arrays encoding a persistence diagram
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        :return: Python list of numpy arrays encoding the persistence images
        """
        if n_jobs is not None:
            parallelize = True
            if n_jobs == -1:
                n_jobs = None
        else:
            parallelize = False
            
        # if diagram is empty, return empty image
        if len(pers_dgms) == 0:
            return np.zeros(self.resolution)
        
        # convert to a list of diagrams if necessary 
        pers_dgms, singular = self._ensure_iterable(pers_dgms)
        
        # TODO: Parallellize over collection of diagrams
        if parallelize:
            fxn = lambda pers_dgm: self._transform(pers_dgm, skew=skew)
            pool = Pool(n_jobs)
            pers_imgs = pool.map(fxn, pers_dgms)
            #pers_imgs = Parallel(n_jobs=n_jobs)(delayed(self._transform)(pers_dgm, skew=skew) for pers_dgm in pers_dgms)
            pool.close()
        else:
            pers_imgs = [self._transform(pers_dgm, skew=skew) for pers_dgm in pers_dgms]      
        
        if singular:
            pers_imgs = pers_imgs[0]
        
        return pers_imgs

    def fit_transform(self, pers_dgms, skew=True):
        """
        Automatically choose persistence image parameters based on a collection of persistence diagrams and transform
        the collection of diagrams into images using the parameters specified in the PersistenceImager object instance
        :param pers_dgms: iterable of (N,2) numpy arrays encoding persistence diagrams
        :param skew: Boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        :return: Python list of numpy arrays encoding the persistence images
        """
        pers_dgms = np.copy(pers_dgms)

        # fit imager parameters
        self.fit(pers_dgms, skew=skew)

        # transform diagrams to images
        self.transform(pers_dgms, skew=skew)

        return pers_imgs
       
    def _transform(self, pers_dgm, skew=True):
        """
        Transform a persistence diagram to a persistence image using the parameters specified in the PersistenceImager
        object instance
        :param pers_dgm: (N,2) numpy array of persistence pairs encoding a persistence diagram
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        :return: numpy array encoding the persistence image
        """
        pers_dgm = np.copy(pers_dgm)
        pers_img = np.zeros(self.resolution)
        n = pers_dgm.shape[0]
        general_flag = True

        # if necessary convert from birth-death coordinates to birth-persistence coordinates
        if skew:
            pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]

        # compute weights at each persistence pair
        wts = self.weight(pers_dgm[:, 0], pers_dgm[:, 1], **self.weight_params)

        # handle the special case of a standard, isotropic Gaussian kernel
        if self.kernel == bvncdf:
            general_flag = False
            sigma = self.kernel_params['sigma']

            # sigma is specified by a single variance
            if isinstance(sigma, (int, float)):
                sigma = np.array([[sigma, 0.0], [0.0, sigma]], dtype=np.float64)

            if (sigma[0, 0] == sigma[1, 1] and sigma[0, 1] == 0.0):
                sigma = np.sqrt(sigma[0, 0])
                for i in range(n):
                    ncdf_b = _norm_cdf((self._bpnts - pers_dgm[i, 0]) / sigma)
                    ncdf_p = _norm_cdf((self._ppnts - pers_dgm[i, 1]) / sigma)
                    curr_img = ncdf_p[None, :] * ncdf_b[:, None]
                    pers_img += wts[i]*(curr_img[1:, 1:] - curr_img[:-1, 1:] - curr_img[1:, :-1] + curr_img[:-1, :-1])
            else:
                general_flag = True

        # handle the general case
        if general_flag:
            bb, pp = np.meshgrid(self._bpnts, self._ppnts, indexing='ij')
            bb = bb.flatten(order='C')
            pp = pp.flatten(order='C')
            for i in range(n):
                curr_img = np.reshape(self.kernel(bb, pp, mu=pers_dgm[i, :], **self.kernel_params),
                                      (self.resolution[0]+1, self.resolution[1]+1), order='C')
                pers_img += wts[i]*(curr_img[1:, 1:] - curr_img[:-1, 1:] - curr_img[1:, :-1] + curr_img[:-1, :-1])

        return pers_img

    def _ensure_iterable(self, pers_dgms):
        # if first entry of first entry is not iterable, then diagrams is singular and we need to make it a list of diagrams
        try:
            singular = not isinstance(pers_dgms[0][0], collections.Iterable)
        except IndexError:
            singular = False

        if singular:
            pers_dgms = [pers_dgms]
            
        return pers_dgms, singular

    def plot_diagram(self, pers_dgm, skew=True, ax=None, out_file=None):
        """
        Plot a persistence diagram
        :param pers_dgm: An (N,2) numpy array encoding a persistence diagram
        :param skew: boolean flag indicating if diagram needs to be converted to birth-persistence coordinates
                     (default: True)
        :param out_file: optional path to save the persistence diagram 
        """
        pers_dgm = np.copy(pers_dgm)

        if skew:
            pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]
            ylabel = 'persistence'
        else:
            ylabel = 'death'

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
        linewidth = (1/50 * self.pixel_size) * 72 * plt.gcf().bbox_inches.width * ax.get_position().width / \
                    np.min((bmax - bmin, pmax - pmin))

        # plot the persistence image grid
        if skew:
            hlines = np.column_stack(np.broadcast_arrays(self._bpnts[0], self._ppnts, self._bpnts[-1], self._ppnts))
            vlines = np.column_stack(np.broadcast_arrays(self._bpnts, self._ppnts[0], self._bpnts, self._ppnts[-1]))
            lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
            line_collection = LineCollection(lines, color='black', linewidths=linewidth)
            ax.add_collection(line_collection)       

        # plot persistence diagram
        ax.scatter(pers_dgm[:, 0], pers_dgm[:, 1])

        # plot diagonal if necessary
        if not skew:
            min_diag = np.min((np.min(ax.get_xlim()), np.min(ax.get_ylim())))
            max_diag = np.min((np.max(ax.get_xlim()), np.max(ax.get_ylim())))
            ax.plot([min_diag, max_diag], [min_diag, max_diag])

        # fix and label axes
        ax.set_aspect('equal')
        ax.set_xlabel('birth')
        ax.set_ylabel(ylabel)

        # optionally save figure
        if out_file:
            plt.savefig(out_file, bbox_inches='tight')

    def plot_image(self, pers_img, ax=None, out_file=None):
        """
        Plot a persistence image
        :param pers_img: (N,K) numpy array encoding a persistence image, e.g. output of PersistenceImager.transform()
        :param out_file: optional path to save the persistence diagram 
        """
        ax = ax or plt.gca()
        ax.matshow(pers_img.T, **{'origin': 'lower'})

        # fix and label axes
        ax.set_xlabel('birth')
        ax.set_ylabel('persistence')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        # optionally save figure
        if out_file:
            plt.savefig(out_file, bbox_inches='tight')


def dict_print(dict_in):
    # print dictionary contents in human-readable format
    if dict_in is None:
        str_out = 'None'
    else:
        str_out = []
        for key, val in dict_in.items():
            str_out.append('%s: %s' % (key, str(val)))
        str_out = '{' + ', '.join(str_out) + '}'

    return str_out


"""
Kernel functions:

A valid kernel is a python function of the form kernel(x, y, mu=(birth, persistence), **kwargs) defining a 
cumulative distribution function such that kernel(x, y) = P(X <= x, Y <=y), where x and y are numpy arrays of equal length. 

The required parameter mu defines the dependance of the kernel on the location of a persistence pair and is usually 
taken to be the mean of the probability distribution function associated to kernel CDF.
"""

def bvncdf(birth, pers, mu=None, sigma=None):
    """
    Optimized bivariate normal cumulative distribution function for computing persistence images using a Gaussian kernel
    :param birth: birth-coordinate(s) of pixel corners
    :param pers: persistence-coordinate(s) of pixel corners
    :param mu: (2,)-numpy array specifying x and y coordinates of distribution means (birth-persistence pairs)
    :param sigma: (2,2)-numpy array specifying distribution covariance matrix or numeric if distribution is isotropic
    :return: P(X <= birth, Y <= pers)
    """
    if mu is None:
        mu = np.array([0.0, 0.0], dtype=np.float64)
    if sigma is None:
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    if sigma[0, 1] == 0.0:
        return _sbvn_cdf(birth, pers,
                         mu_x=mu[0], mu_y=mu[1], sigma_x=sigma[0, 0], sigma_y=sigma[1, 1])
    else:
        return _bvn_cdf(birth, pers,
                        mu_x=mu[0], mu_y=mu[1], sigma_xx=sigma[0, 0], sigma_yy=sigma[1, 1], sigma_xy=sigma[0, 1])


def _norm_cdf(x):
    """
    Univariate normal cumulative distribution function with mean 0.0 and standard deviation 1.0
    :param x: value at which to evaluate the cdf (upper limit)
    :return: P(X <= x), for X ~ N[0,1]
    """
    return erfc(-x / np.sqrt(2.0)) / 2.0


def _sbvn_cdf(x, y, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0):
    """
    Standard bivariate normal cumulative distribution function with specified mean and variances
    :param x: x-coordinate(s) at which to evaluate the cdf (upper limit)
    :param y: y-coordinate(s) at which to evaluate the cdf (upper limit)
    :param mu_x: x-coordinate of mean of bivariate normal
    :param mu_y: y-coordinate of mean of bivariate normal
    :param sigma_x: variance in x
    :param sigma_y: variance in y
    :return: P(X <= x, Y <= y)
    """
    x = (x - mu_x) / np.sqrt(sigma_x)
    y = (y - mu_y) / np.sqrt(sigma_y)
    return _norm_cdf(x) * _norm_cdf(y)


def _bvn_cdf(x, y, mu_x=0.0, mu_y=0.0, sigma_xx=1.0, sigma_yy=1.0, sigma_xy=0.0):
    """
    Bivariate normal cumulative distribution function with specified mean and covariance matrix based on the Matlab
    implementations by Thomas H. Jørgensen (http://www.tjeconomics.com/code/) and Alan Genz
    (http://www.math.wsu.edu/math/faculty/genz/software/matlab/bvnl.m) using the approach described by Drezner
    and Wesolowsky (https://doi.org/10.1080/00949659008811236)
    :param x: x-coordinate(s) at which to evaluate the cdf (upper limit)
    :param y: y-coordinate(s) at which to evaluate the cdf (upper limit)
    :param mu_x: x-coordinate of mean of bivariate normal
    :param mu_y: y-coordinate of mean of bivariate normal
    :param sigma_xx: variance in x
    :param sigma_yy: variance in y
    :param sigma_xy: covariance of x and y
    :return: P(X <= x, Y <= y)
    """
    dh = -(x - mu_x) / np.sqrt(sigma_xx)
    dk = -(y - mu_y) / np.sqrt(sigma_yy)

    hk = np.multiply(dh, dk)
    r = sigma_xy / np.sqrt(sigma_xx * sigma_yy)

    lg, w, x = _gauss_legendre_quad(r)

    dim1 = np.ones((len(dh),), dtype=np.float64)
    dim2 = np.ones((lg,), dtype=np.float64)
    bvn = np.zeros((len(dh),), dtype=np.float64)

    if abs(r) < 0.925:
        hs = (np.multiply(dh, dh) + np.multiply(dk, dk)) / 2.0
        asr = np.arcsin(r)
        sn1 = np.sin(asr * (1.0 - x) / 2.0)
        sn2 = np.sin(asr * (1.0 + x) / 2.0)
        dim1w = np.outer(dim1, w)
        hkdim2 = np.outer(hk, dim2)
        hsdim2 = np.outer(hs, dim2)
        dim1sn1 = np.outer(dim1, sn1)
        dim1sn2 = np.outer(dim1, sn2)
        sn12 = np.multiply(sn1, sn1)
        sn22 = np.multiply(sn2, sn2)
        bvn = asr * np.sum(np.multiply(dim1w, np.exp(np.divide(np.multiply(dim1sn1, hkdim2) - hsdim2,
                                                               (1 - np.outer(dim1, sn12))))) +
                           np.multiply(dim1w, np.exp(np.divide(np.multiply(dim1sn2, hkdim2) - hsdim2,
                                                               (1 - np.outer(dim1, sn22))))), axis=1) / (4 * np.pi) \
              + np.multiply(_norm_cdf(-dh), _norm_cdf(-dk))
    else:
        if r < 0:
            dk = -dk
            hk = -hk

        if abs(r) < 1:
            opmr = (1.0 - r) * (1.0 + r)
            sopmr = np.sqrt(opmr)
            xmy2 = np.multiply(dh - dk, dh - dk)
            xmy = np.sqrt(xmy2)
            rhk8 = (4.0 - hk) / 8.0
            rhk16 = (12.0 - hk) / 16.0
            asr = -1.0 * (np.divide(xmy2, opmr) + hk) / 2.0

            ind = asr > 100
            bvn[ind] = sopmr * np.multiply(np.exp(asr[ind]),
                                           1.0 - np.multiply(np.multiply(rhk8[ind], xmy2[ind] - opmr),
                                                             (1.0 - np.multiply(rhk16[ind], xmy2[ind]) / 5.0) / 3.0)
                                           + np.multiply(rhk8[ind], rhk16[ind]) * opmr * opmr / 5.0)

            ind = hk > -100
            ncdfxmyt = np.sqrt(2.0 * np.pi) * _norm_cdf(-xmy / sopmr)
            bvn[ind] = bvn[ind] - np.multiply(np.multiply(np.multiply(np.exp(-hk[ind] / 2.0), ncdfxmyt[ind]), xmy[ind]),
                                              1.0 - np.multiply(np.multiply(rhk8[ind], xmy2[ind]),
                                                                (1.0 - np.multiply(rhk16[ind], xmy2[ind]) / 5.0) / 3.0))
            sopmr = sopmr / 2
            for ix in [-1, 1]:
                xs = np.multiply(sopmr + sopmr * ix * x, sopmr + sopmr * ix * x)
                rs = np.sqrt(1 - xs)
                xmy2dim2 = np.outer(xmy2, dim2)
                dim1xs = np.outer(dim1, xs)
                dim1rs = np.outer(dim1, rs)
                dim1w = np.outer(dim1, w)
                rhk16dim2 = np.outer(rhk16, dim2)
                hkdim2 = np.outer(hk, dim2)
                asr1 = -1.0 * (np.divide(xmy2dim2, dim1xs) + hkdim2) / 2.0

                ind1 = asr1 > -100
                cdim2 = np.outer(rhk8, dim2)
                sp1 = 1.0 + np.multiply(np.multiply(cdim2, dim1xs), 1.0 + np.multiply(rhk16dim2, dim1xs))
                ep1 = np.divide(np.exp(np.divide(-np.multiply(hkdim2, (1.0 - dim1rs)),
                                                 2.0 * (1.0 + dim1rs))), dim1rs)
                bvn = bvn + np.sum(np.multiply(np.multiply(np.multiply(sopmr, dim1w), np.exp(np.multiply(asr1, ind1))),
                                               np.multiply(ep1, ind1) - np.multiply(sp1, ind1)), axis=1)
            bvn = -bvn / (2.0 * np.pi)

        if r > 0:
            bvn = bvn + _norm_cdf(-np.maximum(dh, dk))
        elif r < 0:
            bvn = -bvn + np.maximum(0, _norm_cdf(-dh) - _norm_cdf(-dk))

    return bvn


def _gauss_legendre_quad(r):
    """
    Return weights and abscissae for the Legendre-Gauss quadrature integral approximation
    :param r: correlation
    :return:
    """
    if np.abs(r) < 0.3:
        lg = 3
        w = np.array([0.1713244923791705, 0.3607615730481384, 0.4679139345726904])
        x = np.array([0.9324695142031522, 0.6612093864662647, 0.2386191860831970])
    elif np.abs(r) < 0.75:
        lg = 6
        w = np.array([.04717533638651177, 0.1069393259953183, 0.1600783285433464,
                      0.2031674267230659, 0.2334925365383547, 0.2491470458134029])
        x = np.array([0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
                      0.5873179542866171, 0.3678314989981802, 0.1252334085114692])
    else:
        lg = 10
        w = np.array([0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
                      0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
                      0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
                      0.1527533871307259])
        x = np.array([0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
                      0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
                      0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
                      0.07652652113349733])

    return lg, w, x


"""
Weight functions:

A valid weight is a python function of the form weight(birth, persistence, **kwargs) defining a scalar-valued function over the birth-persistence plane, where birth and persistence are numpy arrays of equal length. To ensure stability, functions should vanish continuously at the
line persistence = 0 (see [PI] for details).
"""

def linear_ramp(birth, pers, low=0.0, high=1.0, start=0.0, end=1.0):
    """
    Continuous peicewise linear ramp function which is constant below and above specified input values
    :param birth: birth coordinates
    :param pers: persistence coordinates
    :param low: minimal weight
    :param high: maximal weight
    :param start: start persistence value of linear transition from low to high weight
    :param end: end persistence value of linear transition from low to high weight
    :return: weight at persistence pair
    """
    n = birth.shape[0]
    w = np.zeros((n,))
    for i in range(n):
        if pers[i] < start:
            w[i] = low
        elif pers[i] > end:
            w[i] = high
        else:
            w[i] = (pers[i] - start) * (high - low) / (end - start) + low

    return w

def persistence(birth, pers, n=1.0):
    """
    Continuous monotonic function which weight a persistence pair (b,p) by p^n for some n > 0
    :param birth: birth coordinates
    :param pers: persistence coordinates
    :param n: positive float
    :return: weight at persistence pair
    """
    return pers ** n