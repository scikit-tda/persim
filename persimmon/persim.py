from itertools import product

import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

class PersImage(BaseEstimator):
    def __init__(self, diagram, is_landscape=False, pixels=20*20):
        self.diagram = diagram # store original diagram
        
        dg = np.copy(diagram)
        self.landscape = PersImage.to_landscape(dg) if not is_landscape else dg

        assert int(np.sqrt(pixels)) == np.sqrt(pixels), "Pixels must be a square"
        self.pixels = pixels
    
    def transform(self):
        """ 

        
        """


        assert self.landscape is not None, "Must generate a landscape first"
        
        data = self.landscape
        N = int(np.sqrt(self.pixels))

        weighting = self.weighting()
        kernel = self.kernel()

        # Define an NxN grid over our landscape
        maxBD = np.max(data)
        dx = maxBD / (2 * N) 
        xs = np.linspace(0, maxBD, N) + dx
        ys = np.linspace(0, maxBD, N) + dx
        grid = np.array(list(product(xs, reversed(ys))))
        
        # Define zeros
        img = np.zeros(len(grid))

        # weights for each data point
        weights = np.apply_along_axis(weighting, 1, data) 
        
        for i, pixel in enumerate(grid):    
            # pixel defines bottom left corner, compute w.r.t center
            smoothing = kernel(data, pixel)
            img[i] = np.dot(smoothing, weights)

        img = img.reshape((N,N)).T
        return img
    
    def weighting(self):
        ''' Define a weighting function, 
                for stability results to hold, the function must be 0 at y=0.    
        ''' 
        
        # TODO: Implement a logistic function
        
        maxy = np.max(self.landscape[:,1])
        
        def linear(interval):
            # linear function of y such that f(0) = 0 and f(max(y)) = N
            d = interval[1]
            assert d >= 0, "Should not be defined for values below y=0"
            return len(self.landscape[:,1]) / maxy * d
            
        return linear
    
    def kernel(self, kernel_type="gaussian"):
        """ This will return whatever kind of kernal we want to use.
            Must have signature (ndarray size NxM, ndarray size 1xM) -> ndarray size Nx1
        """
        
        def gaussian(data, pixel):
            cov = 0.2
            return mvn.pdf(data, mean=pixel, cov=cov)
        
        return gaussian
    
    @staticmethod
    def to_landscape(diagram):
        """ Convert a diagram to a landscape
            (b,d) -> (b, d-b)
        """        
        diagram[:,1] -= diagram[:,0]
            
        return diagram
    
    def show(self, img):
        """ Visualize the persistence image
        """
        maxBD= np.max(self.landscape)
        plt.imshow(img, extent=(0,maxBD,0,maxBD))

