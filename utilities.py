from __future__ import print_function
import logging
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy import units
import math


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


class Measurement:
    def __init__(self):
        self.description = "" #some text description

        self.value = None
        self.error_pos = None
        self.error_neg = None

        self.apu = None #astropy units (also joke: Auxiliary Power Unit)
        self.power = None #the exponent as in 10^-17, this would be -17

    def avg_error(self):
        if (self.error_neg is not None) and (self.error_pos is not None):
            return 0.5 * abs(self.error_pos) + abs(self.error_neg)
        else:
            return None


    def sq_sym_err(self): #symmetric error
        unc = self.avg_error()
        if (unc is not None) and (self.value is not None):
            return (unc/self.value)**2
        else:
            return 0.0


    #only for errors of the form X*Y/Z
    def compute_mult_error(self,m_array):
        """
        Should only be a few in m_array, and will just loop over, so it does not really matter
        if it is a list or an array

        REMINDER: this is just the sqrt part, so still need to multiply by the calculation value

        :param m_array:
        :return:
        """
        sum = 0.0 #*unc/val)**2
        for m in m_array:
            sum += m.sq_sym_err()

        return np.sqrt(sum)



def unc_str(tup): #helper, formats a string with exponents and uncertainty
    s = ""
    if len(tup) == 2:
        tup = (tup[0],tup[1],tup[1])
    try:
        flux = ("%0.2g" % tup[0]).split('e')
        unc = ("%0.2g" % (0.5 * (abs(tup[1]) + abs(tup[2])))).split('e')

        if len(flux) == 2:
            fcoef = float(flux[0])
            fexp = float(flux[1])
        else:
            fcoef =  float(flux[0])
            fexp = 0

        if len(unc) == 2:
            ucoef = float(unc[0])
            uexp = float(unc[1])
        else:
            ucoef = float(unc[0])
            uexp = 0

        if (fexp < 4) and (fexp > -4):
            s = '%0.2f($\pm$%0.2f)' % (fcoef* 10 ** (fexp), ucoef * 10 ** (uexp ))
        else:# fexp != 0:
            s = '%0.2f($\pm$%0.2f)e%d' % (fcoef, ucoef * 10 ** (uexp - fexp), fexp)
        #else:
        #    s = '%0.2f($\pm$%0.2f)' % (fcoef, ucoef * 10 ** (uexp - fexp))
    except:
        log.warning("Exception in unc_str()", exc_info=True)

    return s

    #todo: more accessors
    # get the value * units, calculate the error with two mea


def is_in_ellipse(xp,yp,xc,yc,a,b,angle):
    """
    :param xp: x coord of point
    :param yp: y coord of point
    :param xc: x coord of ellipse center
    :param yc: y coord of ellipse center
    :param a: major axis (radius)
    :param b: minor axis (radius)
    :param angle: rotation angle in radians from positive x axis (counter clockwise)
    Assumes lower left is 0,0
    :return:
    """

    #translate to center ellipse at 0,0
    xp = xp - xc
    yp = yp - yc
    xc,yc = 0,0

    #rotate to major axis along x
    angle = 2.*math.pi - angle   #want in clock-wise from positive x axis (rotation matrix below is clock-wise)
    cosa = math.cos(angle)
    sina = math.sin(angle)

    #xt = transformed xp coord where major axis is positive x-axis and minor is positive y-axis
    #yt = transformed yp coord
    xt = xp*cosa-yp*sina
    yt = xp*sina+yp*cosa

    #np.sqrt(xt+yt) would now be the distance from the center
    #essentially stretching x and y axis s|t ellipse becomes a unit circle, then if the distance (or distance squared,
    #as coded here) is less than 1 it is inside (if == 1 it is on the ellipse or circle)
    inside=((xt*xt)/(a*a))+((yt*yt)/(b*b))

    if inside <= 1:
        return True
    else:
        return False

def saferound(value,precision,fail=0):
    try:
        if value is None:
            return fail
        elif np.isnan(value):
            return fail
        else:
            return np.round(value,precision)
    except:
        return fail
