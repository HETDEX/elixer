from __future__ import print_function
import logging
import numpy as np
from astropy.visualization import ZScaleInterval


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

def get_vrange(vals,contrast=0.25):
    vmin = None
    vmax = None

    try:
        zscale = ZScaleInterval(contrast=contrast,krej=2.5) #nsamples=len(vals)
        vmin, vmax = zscale.get_limits(values=vals )
        log.info("Vrange = %f, %f" %(vmin,vmax))
    except:
        log.info("Exception in utilities::get_vrange:",exc_info =True)

    return vmin,vmax

# def get_vrange(vals,scale=1.0,contrast=1.0):
#     vmin = None
#     vmax = None
#     if scale == 0:
#         scale = 1.0
#
#     try:
#         zscale = ZScaleInterval(contrast=1.0,krej=2.5) #nsamples=len(vals)
#         vmin,vmax = zscale.get_limits(values=vals)
#         vmin = vmin/scale
#         vmax = vmax/scale
#         log.info("HETDEX (zscale) vrange = (%f, %f) raw range = (%f, %f)" %(vmin,vmax,np.min(vals),np.max(vals)))
#     except:
#         log.info("Exception in utilities::get_vrange:",exc_info =True)
#
#     return vmin, vmax