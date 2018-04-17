from itertools import product

import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator

class PersImage(BaseEstimator):
    def __init__(self, kernel_type="gaussian", weighting_type="linear", pixels=20*20, spread=None):
        self.kernel_type = kernel_type
        self.weighting_type = weighting_type
        self.spread = spread
        
        assert int(np.sqrt(pixels)) == np.sqrt(pixels), "Pixels must be a square"
        self.pixels = pixels
    
    def transform(self, diagrams):
        """ Convert diagram or list of diagrams to a persistence image
        """

        if type(diagrams) is not list:
            dg = np.copy(diagrams) # keep original diagram untouched
            landscape = PersImage.to_landscape(dg)
            specs = {"maxBD": np.max(landscape)}
            # import pdb; pdb.set_trace()
            imgs = self._transform(landscape, specs)
        else:
            dgs = [np.copy(diagram) for diagram in diagrams]
            landscapes = [PersImage.to_landscape(dg) for dg in dgs]
            specs = {'maxBD': np.max([np.max(landscape) for landscape in landscapes])}
            # import pdb; pdb.set_trace()
            imgs = [self._transform(dgm, specs) for dgm in landscapes]
        
        return imgs

    def _transform(self, landscape, specs):
        """ Convert single diagram to a persistence image
        """

        N = int(np.sqrt(self.pixels))

        # Define an NxN grid over our landscape
        maxBD = specs['maxBD']
        dx = maxBD / (2 * N) 
        xs = np.linspace(0, maxBD, N) + dx
        ys = np.linspace(0, maxBD, N) + dx
        grid = np.array(list(product(xs, reversed(ys))))
        
        weighting = self.weighting()#landscape)

        # maxBD seems to be a reasonable variance in practice.
        kernel = self.kernel(spread=self.spread if self.spread else maxBD)

        # Define zeros
        img = np.zeros(len(grid))

        # weights for each data point
        weights = np.apply_along_axis(weighting, 1, landscape) 
        # import pdb; pdb.set_trace()
        for i, pixel in enumerate(grid):    
            # pixel defines bottom left corner, compute w.r.t center
            smoothing = kernel(landscape, pixel)
            img[i] = np.dot(smoothing, weights)

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

