"""
From Maja Niemeyer https://github.com/majanie/intensity-mapping.git
"""
try:
    from elixer import global_config as G
except:
    import global_config as G

log = G.Global_Logger('logger')
log.setlevel(G.LOG_LEVEL)

import numpy as np
#from astropy.stats.funcs import median_absolute_deviation
import random
import astropy.units as u
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation, biweight_scale
from scipy.stats import bayes_mvs



def conf_interval(num_samples,sd,conf=0.95):
    """
    mean +/- error  ... this is the +/- error part as 95% (or other) confidence interval (assuming normal distro)

    :param num_samples:
    :param sd: standard deviation
    :param conf:
    :return:
    """

    if num_samples < 30:
        return None

    #todo: put in other values
    if conf == 0.68:
        t = 1.0
    elif conf == 0.95:
        t = 1.96
    elif conf == 0.99:
        t = 2.576
    else:
        log.debug("todo: need to handle other confidence intervals: ", conf)
        return None

    return t * sd / np.sqrt(num_samples)


def bootstrap_confidence_interval(data,confidence=0.68,num_bootstraps=-1, boostrap_selection_size=-1):
    """
    bootstrap the data (usually from an MC run) to normalize about a bw "avg" and take confidence interval of that
    (the MC run for PLAE/POII is often very skewed)

    :param data:
    :param confidence:
    :param num_bootstraps:
    :return: avg of averages (bw loc) and confidence interval
    """

    try:
        datasize = len(data)

        if num_bootstraps < 1:
            num_bootstraps = datasize

        if boostrap_selection_size < 1:
            boostrap_selection_size = datasize

        log.debug(f"Bootstrap confidence interval. elements: {len(data)}, conf: {confidence}, "
                 f"num bootstraps: {num_bootstraps}, selection size: {boostrap_selection_size}")
        #bw_scale = []
        bw_loc = []

        for i in range(num_bootstraps):
            try:
                subset = np.random.choice(data, boostrap_selection_size, replace=True)
                bw_loc.append(biweight_location(subset)) #avg
                #bw_scale.append(biweight_location(subset))
            except:
                log.debug("Exception (1) in bootstrap_confidence_interval",exc_info=True)

        loc = biweight_location(bw_loc)
        scale = biweight_scale(bw_loc)
        ci = conf_interval(len(bw_loc),scale*np.sqrt(num_bootstraps),conf=confidence)
        log.debug(f"Boostrap confidence interval results: bw_loc: {loc}  ci: {ci}")

        return loc,ci
    except:
        log.debug("Exception (2) in bootstrap_confidence_interval",exc_info=True)
        return None,None


def biweight_location_errors(data, errors, c=6.0, M=None, axis=None):
    """
    wrapper for biweight location weights where the caller passes in an error array rather than weight array
    s|t large errors are down-weighted
    :param data:
    :param errors:
    :param c:
    :param M:
    :param axis:
    :return:
    """
    try:
        errors = np.asanyarray([x.value for x in errors]).astype(np.float64)
    except:
        try:
            errors = np.asanyarray(errors).astype(np.float64)
        except:  # could be units attached
            log.warning("Exception in weighted_biweight:biweight_location_errors", exc_info=True)
            raise

    errors = np.nan_to_num(errors) #any nan will become zero and then those and anyother zero errors become infinite
    errors[errors==0] = np.inf #that way, the weight becomes zero ...
    #we might think of zero error as what should be maximumly weighted, but here we assume that
    #a zero error means no error recorded and it should get no weighting
    weights = 1./errors

    return biweight_location_weights(data,weights,c,M,axis)

def biweight_location_weights(data, weights, c=6.0, M=None, axis=None):

    saved_units = None

    try: #usually coming in with units attached, so remove them (if not, fall to the exception case and treat w/o units)
        saved_units = data[0].unit
        data = np.asanyarray([x.value for x in data]).astype(np.float64)
        weights = np.asanyarray([x.value for x in weights]).astype(np.float64)
    except:
        try:
            data = np.asanyarray(data).astype(np.float64)
            weights = np.asanyarray(weights).astype(np.float64)
        except:  # could be units attached
            log.warning("Exception in weighted_biweight",exc_info=True)
            raise

    data[weights==0] = np.nan 
    weights[~np.isfinite(data)] = np.nan

    if np.count_nonzero(np.isnan(weights)) > (0.1 * len(data)):
        raise ValueError("too many invalid weights")

    # if np.count_nonzero(np.isnan(weights)) > 0:
    #     print(f"some invalid weights {np.count_nonzero(np.isnan(weights))} < {0.1 * len(data)}")

    # if not np.any(weights):
    #     raise ValueError("no valid weights")
    #
    # if np.all(np.isnan(weights)):
    #     raise ValueError("no valid weights")

    #np.count_nonzero(np.isnan(weights))

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
    bwl = None
    try:
        bwl = M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)
    except: #just for logging (note: could trap and call regular biweight, but might want to not hide the fail)
        if not np.isnan(M): #hide the print if this is the common issue, but still raise the exception so caller knows it failed
            log.warning("Exception in weighted_biweight", exc_info=True)
        raise #re-raise so caller will get fail notice

    #restore units (if provided)
    if (saved_units is not None) and (bwl is not None):
        bwl *= saved_units

    return bwl
