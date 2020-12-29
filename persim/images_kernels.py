"""
Kernel functions for PersistenceImager() transformer:

A valid kernel is a Python function of the form 

kernel(x, y, mu=(birth, persistence), **kwargs) 

defining a cumulative distribution function(CDF) such that kernel(x, y) = P(X <= x, Y <=y), where x and y are numpy arrays of equal length. 

The required parameter mu defines the dependance of the kernel on the location of a persistence pair and is usually taken to be the mean of the probability distribution function associated to kernel CDF.
"""
import numpy as np
from scipy.special import erfc

def uniform(x, y, mu=None, width=1, height=1):
    w1 = np.maximum(x - (mu[0] - width/2), 0)
    h1 = np.maximum(y - (mu[1] - height/2), 0)
    
    w = np.minimum(w1, width)
    h = np.minimum(h1, height)

    return w*h / (width*height)


def gaussian(birth, pers, mu=None, sigma=None):
    """ Optimized bivariate normal cumulative distribution function for computing persistence images using a Gaussian kernel.
    
    Parameters
    ----------
    birth : (M,) numpy.ndarray
        Birth coordinate(s) of pixel corners.
    pers : (N,) numpy.ndarray
        Persistence coordinates of pixel corners.
    mu : (2,) numpy.ndarray
        Coordinates of the distribution mean (birth-persistence pairs).
    sigma : float or (2,2) numpy.ndarray 
        Distribution's covariance matrix or the equal variances if the distribution is standard isotropic.
    
    Returns
    -------
    float
        Value of joint CDF at (birth, pers), i.e.,  P(X <= birth, Y <= pers).
    """
    if mu is None:
        mu = np.array([0.0, 0.0], dtype=np.float64)
    if sigma is None:
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    if sigma[0][1] == 0.0:
        return sbvn_cdf(birth, pers,
                         mu_x=mu[0], mu_y=mu[1], sigma_x=sigma[0][0], sigma_y=sigma[1][1])
    else:
        return bvn_cdf(birth, pers,
                        mu_x=mu[0], mu_y=mu[1], sigma_xx=sigma[0][0], sigma_yy=sigma[1][1], sigma_xy=sigma[0][1])


def norm_cdf(x):
    """ Univariate normal cumulative distribution function (CDF) with mean 0.0 and standard deviation 1.0.
    
    Parameters
    ----------
    x : float
        Value at which to evaluate the CDF (upper limit).
    
    Returns
    -------
    float
        Value of CDF at x, i.e., P(X <= x), for X ~ N[0,1].
    """
    return erfc(-x / np.sqrt(2.0)) / 2.0


def sbvn_cdf(x, y, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0):
    """ Standard bivariate normal cumulative distribution function with specified mean and variances.
    
    Parameters
    ----------
    x : float or numpy.ndarray of floats
        x-coordinate(s) at which to evaluate the CDF (upper limit).
    y : float or numpy.ndarray of floats 
        y-coordinate(s) at which to evaluate the CDF (upper limit).
    mu_x : float
        x-coordinate of the mean.
    mu_y : float
        y-coordinate of the mean.
    sigma_x : float
        Variance in x.
    sigma_y : float
        Variance in y.
    
    Returns
    -------
    float
        Value of joint CDF at (x, y), i.e.,  P(X <= birth, Y <= pers).
    """
    x = (x - mu_x) / np.sqrt(sigma_x)
    y = (y - mu_y) / np.sqrt(sigma_y)
    return norm_cdf(x) * norm_cdf(y)


def bvn_cdf(x, y, mu_x=0.0, mu_y=0.0, sigma_xx=1.0, sigma_yy=1.0, sigma_xy=0.0):
    """ Bivariate normal cumulative distribution function with specified mean and covariance matrix.
    
    Parameters
    ----------
    x : float or numpy.ndarray of floats
        x-coordinate(s) at which to evaluate the CDF (upper limit).
    y : float or numpy.ndarray of floats 
        y-coordinate(s) at which to evaluate the CDF (upper limit).
    mu_x : float
        x-coordinate of the mean.
    mu_y : float
        y-coordinate of the mean.
    sigma_x : float
        Variance in x.
    sigma_y : float
        Variance in y.
    sigma_xy : float
        Covariance of x and y.
        
    Returns
    -------
    float
        Value of joint CDF at (x, y), i.e.,  P(X <= birth, Y <= pers).
        
    Notes
    -----
    Based on the Matlab implementations by Thomas H. Jørgensen (http://www.tjeconomics.com/code/) and Alan Genz (http://www.math.wsu.edu/math/faculty/genz/software/matlab/bvnl.m) using the approach described by Drezner and Wesolowsky (https://doi.org/10.1080/00949659008811236).
    """
    dh = -(x - mu_x) / np.sqrt(sigma_xx)
    dk = -(y - mu_y) / np.sqrt(sigma_yy)

    hk = np.multiply(dh, dk)
    r = sigma_xy / np.sqrt(sigma_xx * sigma_yy)

    lg, w, x = gauss_legendre_quad(r)

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
              + np.multiply(norm_cdf(-dh), norm_cdf(-dk))
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
            ncdfxmyt = np.sqrt(2.0 * np.pi) * norm_cdf(-xmy / sopmr)
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
            bvn = bvn + norm_cdf(-np.maximum(dh, dk))
        elif r < 0:
            bvn = -bvn + np.maximum(0, norm_cdf(-dh) - norm_cdf(-dk))

    return bvn


def gauss_legendre_quad(r):
    """ Return weights and abscissae for the Legendre-Gauss quadrature integral approximation.
    
    Parameters
    ----------
    r : float
        Correlation
    
    Returns
    -------
    tuple
        Number of points in the Gaussian quadrature rule, quadrature weights, and quadrature points.
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