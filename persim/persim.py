from itertools import product

import numpy as np
from scipy.stats import multivariate_normal as mvn
import scipy.spatial as spatial
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator

class PersImage(BaseEstimator):
    def __init__(self, kernel_type="gaussian", weighting_type="linear", pixels=20*20, spread=None):
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread
        
        assert int(np.sqrt(pixels)) == np.sqrt(pixels), "Pixels must be a square"
        self.pixels = pixels
        self.N = int(np.sqrt(pixels))
    
    def transform(self, diagrams):
        """ Convert diagram or list of diagrams to a persistence image
        """

        if type(diagrams) is not list:
            dg = np.copy(diagrams) # keep original diagram untouched
            landscape = PersImage.to_landscape(dg)
            specs = {
                "maxBD": np.max(landscape),
                "minBD": np.min(landscape)
            }

            imgs = self._transform(landscape, specs)
        else:
            dgs = [np.copy(diagram) for diagram in diagrams]
            landscapes = [PersImage.to_landscape(dg) for dg in dgs]
            specs = {
                'maxBD': np.max([np.max(landscape) for landscape in landscapes]),
                'minBD': np.min([np.min(landscape) for landscape in landscapes])
            }
            imgs = [self._transform(dgm, specs) for dgm in landscapes]
        
        return imgs

    def _transform(self, landscape, specs):
        """ Convert single diagram to a persistence image
        """
        # import pdb; pdb.set_trace()
        N = self.N

        # Define an NxN grid over our landscape
        maxBD = specs['maxBD']
        minBD = min(specs['minBD'], 0)

        dx = maxBD / (2 * N) 
        xs = np.linspace(minBD, maxBD, N) + dx
        ys = np.linspace(minBD, maxBD, N) + dx
        grid = np.array(list(product(xs, reversed(ys))))
        
        weighting = self.weighting(landscape)

        # maxBD seems to be a reasonable variance in practice.
        kernel = self.kernel(spread=self.spread if self.spread else dx)

        # Define zeros
        img = np.zeros(len(grid))

        # weights for each data point
        weights = np.apply_along_axis(weighting, 1, landscape) 

        # Construct a Guassian surface 
        def p_surface(point):
            """ Function defining the persistence surface  
            """
            smoothing = kernel(landscape, point)
            return np.dot(smoothing, weights)

        integrator = Integrator()
        for i, pixel in enumerate(grid):    
            img[i] = integrator.integrate(p_surface, pixel, dx)

        img = img.reshape((N,N)).T
        return img
    
    def weighting(self, landscape=None):
        ''' Define a weighting function, 
                for stability results to hold, the function must be 0 at y=0.    
        ''' 
        
        # TODO: Implement a logistic function
        # TODO: use self.weighting_type to choose function    

        if landscape is not None:
            maxy = np.max(landscape[:,1])

        def linear(interval):
            # linear function of y such that f(0) = 0 and f(max(y)) = N
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

        # if self.kernel_type == "linear":
            # return linear
        return linear
    
    def kernel(self, spread=1):
        """ This will return whatever kind of kernal we want to use.
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
        diagram[:,1] -= diagram[:,0]
            
        return diagram
    
    def show(self, imgs):
        """ Visualize the persistence image
        """

        if type(imgs) is not list:
            imgs = [imgs]

        for i, img in enumerate(imgs):
            plt.imshow(img, cmap=plt.get_cmap('plasma'))
            plt.show()


class Integrator():
    def integrate(self, f, center, dx):
        """ Integrate f over a square centered at center with radius dx.
        """

        # rectangle in the center
        # height = f(center)
        # area = 2*dx * 2*dx
        # return height * area

        if type(f) is not np.vectorize:
            f = np.vectorize(f, signature='(m)->()')

        # Trapazoid rule using the volume of a convex hull of 8 corners
        corners_ = np.array([[ dx, dx], 
                             [ dx,-dx], 
                             [-dx, dx], 
                             [-dx,-dx]]) + center

        corners = np.zeros((8,3))
        corners[:4,:2] = corners_
        corners[4:,:2] = corners_
        corners[4:,2] = f(corners_)

        vol = self._convex_hull_volume_bis(corners)
        return vol

    # https://stackoverflow.com/questions/24733185/volume-of-convex-hull-with-qhull-from-scipy
    def _tetrahedron_volume(self, a, b, c, d):
        return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

    def _convex_hull_volume_bis(self, pts):
        """ Calculate volume of convex hull """
        try:
            ch = spatial.ConvexHull(pts)
            simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                        ch.simplices))
            tets = ch.points[simplices]
            return np.sum(self._tetrahedron_volume(tets[:, 0], tets[:, 1],
                                                   tets[:, 2], tets[:, 3]))


        except spatial.qhull.QhullError:
            # Points are coplanar, probably because f=0 for all points
            return 0

    

