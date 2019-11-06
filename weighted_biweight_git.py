"""
From Maja Niemeyer https://github.com/majanie/intensity-mapping.git
"""


import numpy as np
#from astropy.stats.funcs import median_absolute_deviation
import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation


def biweight_location_weights(data, weights, c=6.0, M=None, axis=None):

    data = np.asanyarray(data).astype(np.float64)
    weights = np.asanyarray(weights).astype(np.float64)
   
    data[weights==0] = np.nan 
    weights[~np.isfinite(data)] = np.nan
    
    if (data.shape!=weights.shape):
        raise ValueError("data.shape != weights.shape")

    if M is None:
        M = np.nanmedian(data, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = data - M

    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis, ignore_nan=True)
    #madweights = median_absolute_deviation(weights, axis=axis)

    if axis is None and mad == 0.:
        return M  # return median if data is a constant array
    
    #if axis is None and madweights == 0:
    #    madweights = 1.

    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
        const_mask = (mad == 0.)
        mad[const_mask] = 1.  # prevent divide by zero

    #if axis is not None:
    #    madweights = np.expand_dims(madweights, axis=axis)
    #    const_mask = (madweights == 0.)
    #    madweights[const_mask] = 1.  # prevent divide by zero

    cmadsq = (c*mad)**2
    
    factor = 0.5
    weights  = weights/np.nanmedian(weights)*factor 
    
    u = d / (c * mad)

    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    #print("number of excluded points ", len(mask[mask]))
    
    u = (1 - u ** 2) ** 2
    
    weights[~np.isfinite(weights)] = 0
    
    u = u + weights**2
    u[weights==0] = 0
    d[weights==0] = 0
    u[mask] = 0

    # along the input axis if data is constant, d will be zero, thus
    # the median value will be returned along that axis
    return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)
