from __future__ import print_function
import logging
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy import units
from astropy.coordinates import SkyCoord
import math
import os.path as op
import tarfile as tar

import sqlite3
from sqlite3 import Error

try:
    from elixer import global_config as G
except:
    import global_config as G

log = G.Global_Logger('utilities')
log.setlevel(G.LOG_LEVEL)


def id_from_coord(ra,dec):
    """
    make an int64 id number from the ra and dec. Each provides 7 digits ([3].[4]) with leading zeros and no decimal
    and a leading 9 is prepended
    if the dec is negative, prend a 1 infront of the dec(e.g. values -0.0001 to -90 are 1000001 to 1900000)
    :param ra: float as decimal degrees
    :param dec: float as decimal degrees
    :return: int64
    """
    try:
        id = np.int64(9e14 + int(ra * 1e4) * 1e7 + int(abs(dec * 1e4)))
        if dec < 0:
            id += int(1e6)
        return id
    except:
        return None


def coord2deg(coord_str):
    """
    take the coordinate string and covert it to degrees

    assumes we are either already in decimal degrees or as hourangle + deg

    :param coord_str: as a string as ra dec (space separated)
    :return:
    """
    ra,dec = None,None
    try:
        if ":" in coord_str.split()[0]:
            #assume hour angle
            c = SkyCoord(coord_str,frame="icrs",unit=(units.hourangle, units.deg))
            ra = c.ra.value
            dec = c.dec.value
        else: #might already be in decimal degrees
            c = SkyCoord(coord_str, frame="icrs", unit=(units.deg, units.deg))
            ra = c.ra.value
            dec = c.dec.value
    except:
        pass

    return ra, dec

def angular_distance(ra1,dec1,ra2,dec2):
    """
    :param ra1: decimal degrees
    :param dec1: decimal degrees
    :param ra2:  decimal degrees
    :param dec2: decimal degrees
    :return:  distance between the two positions (in decimal arcsec)
    """
    dist = -1
    try:
        if (ra1 is None) or (dec1 is None) or (ra2 is None) or (dec2 is None):
            return -1

        c1 = SkyCoord(ra=ra1 * units.deg, dec=dec1* units.deg)
        c2 = SkyCoord(ra=ra2 * units.deg, dec=dec2* units.deg)
        dist = c1.separation(c2).value * 3600.
    except:
        log.info("Exception in utilities::angular_distance:",exc_info =True)

    return dist #in arcsec

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
    NOTICE: this is COORDINATE DISTANCE ... does not attempt to adjust for underlying metric (i.e does not adjust
    for projection on sphere with RA distances as function of cos(declination).).

    :param xp: x coord of point
    :param yp: y coord of point
    :param xc: x coord of ellipse center
    :param yc: y coord of ellipse center
    :param a: major axis (radius)
    :param b: minor axis (radius)
    :param angle: rotation angle in radians from positive x axis (counter clockwise)
    Assumes lower left is 0,0
    :return: True/False, distance to barycenter
    """

    try:
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

        dist_to_barycenter = np.sqrt(xt*xt+yt*yt)

        #np.sqrt(xt+yt) would now be the distance from the center
        #essentially stretching x and y axis s|t ellipse becomes a unit circle, then if the distance (or distance squared,
        #as coded here) is less than 1 it is inside (if == 1 it is on the ellipse or circle)
        inside=((xt*xt)/(a*a))+((yt*yt)/(b*b))

        return inside <= 1, dist_to_barycenter

    except:
        log.debug("Exception in utilities.",exc_info=True)
        return None, None


def dist_to_ellipse(xp,yp,xc,yc,a,b,angle):
    """
    Find the distance to the nearest point ON the ellipse for point OUTSIDE the ellipse.

    NOTICE: this is COORDINATE DISTANCE ... does not attempt to adjust for underlying metric (i.e does not adjust
    for projection on sphere with RA distances as function of cos(declination).).

    :param xp: x coord of point
    :param yp: y coord of point
    :param xc: x coord of ellipse center
    :param yc: y coord of ellipse center
    :param a: major axis (radius)
    :param b: minor axis (radius)
    :param angle: rotation angle in radians from positive x axis (counter clockwise)
    Assumes lower left is 0,0
    :return: outside (True/False), distance to curve, distance to barycenter, nearest point on curve as (x,y)
            where outside also proxies for success ... if True, this should have worked, if False, not
    """

    try:
        #translate to center ellipse at 0,0
        original_xc = xc
        original_yc = yc
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


        dist_to_barycenter = np.sqrt(xt*xt+yt*yt)
        dist_to_curve = None
        inside = (((xt * xt) / (a * a)) + ((yt * yt) / (b * b))) <= 1

        if inside:
            log.info("Point to ellipse, point is INSIDE the ellipse.")
            return not inside, None,dist_to_barycenter,(None,None)

        #calculate and return distance
        #no analytical solution? so approximate
        #modified from Johannes Peter (https://gist.github.com/JohannesMP)
        px = abs(xt)
        py = abs(yt)

        tx = 0.707
        ty = 0.707

        for x in range(0, 3):
            x = a * tx
            y = b * ty

            ex = (a * a - b * b) * tx ** 3 / a
            ey = (b * b - a * a) * ty ** 3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = math.hypot(rx, ry)
            q = math.hypot(qx, qy)

            tx = min(1, max(0, (qx * r / q + ex) / a))
            ty = min(1, max(0, (qy * r / q + ey) / b))
            t = math.hypot(tx, ty)
            tx /= t
            ty /= t

        pt_on_curve =  (math.copysign(a * tx, xt), math.copysign(b * ty, yt))
        curve_to_barycenter = np.sqrt(pt_on_curve[0] * pt_on_curve[0] + pt_on_curve[1] * pt_on_curve[1])
        dist_to_curve = dist_to_barycenter - curve_to_barycenter

        #transform pt_on_curve back to ORIGINAL coordinate system
        #inverse rotation
        xt = pt_on_curve[0]*cosa+pt_on_curve[1]*sina
        yt = -1*pt_on_curve[0]*sina+pt_on_curve[1]*cosa
        #translate back
        xt += original_xc
        yt += original_yc
        pt_on_curve = (xt,yt)

        return not inside, dist_to_curve,dist_to_barycenter,pt_on_curve
    except:
        log.debug("Exception in utilities.",exc_info=True)
        return False,None, None, None


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



#this is very painful (time costly)
def open_file_from_tar(tarfn,fqfn=None,fn=None):
    """

    one of fqfn must be supplied

    :param tarfn: tar filename (fully qualified if necessary
    :param fqfn: fully qualified filename inside the tarfile
    :param fn:  filename only inside the tarfile (i.e. tar-file internal path is unknown)
    :return:
    """

    try:

        if fqfn is None and fn is None:
            log.debug(f"utilities, open tarfile, no subfile specified")
            return None

        if not op.exists(tarfn):
            log.debug(f"utilities, open tarfile {tarfn} does not exist")
            return None

        if not tar.is_tarfile(tarfn):
            log.debug(f"utilities, open tarfile {tarfn} is not a tar file")
            return None

        tarfile = tar.open(name=tarfn)

        if fqfn is not None:
            file = tarfile.extractfile(fqfn)
        elif fn is not None:
            file = tarfile.extractfile(fn)

        if file is not None:
            return file

    except:
        log.debug(f"Exception attempting to fetch sub-file {fqfn} or {fn} from {tarfn}")
        return None


def open_sqlite_file(sqlfn,key):
    """

    :param sqlfn:
    :param key:
    :return:
    """

    try:
        conn = None
        try:
            conn = sqlite3.connect(sqlfn)
        except:
            log.info(f"Exception attempting to fetch file {key} from {sqlfn}",exc_info=True)

        if conn is None:
            log.info(f"Error. SQLite connection is invalid")

        cursor = conn.cursor()

        sql_read_blob = """SELECT blobvalue from blobtable where blobname = ?"""

        cursor.execute(sql_read_blob, (str(blobname),))
        blobvalue = cursor.fetchall()
        # get back a list of tuples (each list entry is one row, the tuple is the row, so
        # we want the 1st entry (detectid is unique, so should be one or none, and the second column which is the image)
        cursor.close()
        conn.close()

        return blobvalue
    except:
        log.info(f"Exception attempting to fetch file {key} from {sqlfn}",exc_info=True)


def intersection_area(R_in, R_out,d, r=G.Fiber_Radius):
    """
    Get the overlap area of fiber and annulus. All units are arcsec

    credit: mostly taken from:
    https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/

    :param R_in: inner radius of the annulus (arcsec)
    :param R_out: outer radius of the annulus (arcsec)
    :param d: fiber center distance to center of annulus (arcsec)
    :param r: fiber radius
    :return: area (in arcsec**2) of fiber inside the annulus
    """

    def overlap(R,d,r=G.Fiber_Radius):
        """
        Treat one Radius of the annulus as a circle
        :param R: Annulus radius (inner or outer)
        :param d: fiber center distance to center of annulus (arcsec)
        :param r: fiber radius
        :return:
        """

        if d <= abs(R-r):
            # One circle is entirely enclosed in the other.
            return np.pi * min(R, r)**2
        if d >= r + R:
            # The circles don't overlap at all.
            return 0

        r2, R2, d2 = r**2, R**2, d**2
        alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
        beta = np.arccos((d2 + R2 - r2) / (2*d*R))
        return (r2*alpha + R2*beta - 0.5*(r2*np.sin(2*alpha) + R2 * np.sin(2*beta)))

    try:
        area = overlap(R_out,d,r) - overlap(R_in,d,r)
    except:
        #just assume the whole (r could be junk)
        try:
            area = np.pi *r *r
        except:
            area = 0.0

    return area

def intersection_fraction(R_in, R_out,d, r=G.Fiber_Radius):
    """
    Get the fractional overlap area of fiber and annulus. All units are arcsec

    :param R_in: inner radius of the annulus (arcsec)
    :param R_out: outer radius of the annulus (arcsec)
    :param d: fiber center distance to center of annulus (arcsec)
    :param r: fiber radius
    :return: fraction of the fiber area inside the annulus
    """

    return intersection_area(R_in, R_out,d,r) / (np.pi *r *r)


def reformat_input_id(input_id):
    """
    take an input_id string and convert to the form used in the h5 files
    :param input_id:
    :return:
    """

    try:
        toks = input_id.split("_")

        id = toks[0].replace('v','')[0:8] + "v" + toks[0].replace('v','')[8:] + "_" + str(int(toks[1])).zfill(3) + \
             "_" + str(int(toks[2])).zfill(3) + "_" + str(int(toks[3])).zfill(3) + "_" + str(int(toks[4])).zfill(3)

        return id
    except:
        return None


def gaussian_uncertainty(val,x,mu,mu_err,sigma,sigma_err,A, A_err, y, y_err):
    """
    Returns error (uncertiainty) on a Gaussian output value

    :param val: the value of the output of the Gaussian (the y value, not be be confused with the y-offset that is
                listed here as y)
    :param x:   the x coordinates fed in
    :param mu:
    :param mu_err:
    :param sigma:
    :param sigma_err:
    :param A:
    :param A_err:
    :param y:
    :param y_err:
    :return:
    """

    sr2pi = np.sqrt(2*np.pi)

    def partial_mu(x,mu,sigma,A):
        return A * (x-mu) / (sigma**3 * sr2pi) * np.exp(-np.power((x - mu) / sigma, 2.) / 2.)

    def partial_sigma(x,mu,sigma,A):
        return A * (sigma**2 - (x-mu)**2)/(sigma**4 * sr2pi) * np.exp(-np.power((x - mu) / sigma, 2.) / 2.)

    def partial_A(x,mu,sigma,A):
        return 1/(sigma*sr2pi)* np.exp(-np.power((x - mu) / sigma, 2.) / 2.)

    def partial_y():
        return 1

    return np.sqrt( (partial_mu(x,mu,sigma,A)*mu_err)**2 +
                    (partial_sigma(x,mu,sigma,A)*sigma_err)**2 +
                    (partial_A(x,mu,sigma,A)*A_err)**2 +
                    (partial_y()*y+y_err)**2   )

def getnearpos(array,value):
    """
    Nearest, but works best (with less than and greater than) if monotonically increasing. Otherwise,
    lt and gt are (almost) meaningless

    :param array:
    :param value:
    :return: nearest index, nearest index less than the value, nearest index greater than the value
            None if there is no less than or greater than
    """
    if type(array) == list:
        array = np.array(array)

    idx = (np.abs(array-value)).argmin()

    if array[idx] == value:
        lt = idx
        gt = idx
    elif array[idx] < value:
        lt = idx
        gt = idx + 1
    else:
        lt = idx - 1
        gt = idx

    if lt < 0:
        lt = None

    if gt > len(array) -1:
        gt = None


    return idx, lt, gt


