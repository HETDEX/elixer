from __future__ import print_function
import logging
import numpy as np


try:
    from elixer import global_config as G
except:
    import global_config as G

log = G.Global_Logger('utilities')
log.setlevel(G.logging.DEBUG)

def angular_distance(ra1,dec1,ra2,dec2):
    #distances are expected to be relatively small, so will use the median between the decs for curvature
    dist = -1.
    try:
        dec_avg = 0.5*(dec1 + dec2)
        dist = np.sqrt((np.cos(np.deg2rad(dec_avg)) * (ra2 - ra1)) ** 2 + (dec2 - dec1) ** 2)
    except:
        log.debug("Invalid angular distance.",exc_info=True)

    return dist * 3600. #in arcsec