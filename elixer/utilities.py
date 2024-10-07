from __future__ import print_function
import logging
import os

import numpy as np
from astropy.visualization import ZScaleInterval
from astropy import units
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.table import Table
import math
import os
import os.path as op
import shutil
import subprocess
import glob
import tarfile as tar
from datetime import date
import fnmatch
from tqdm import tqdm

from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as C
#from astropy import units as u

import sqlite3
from sqlite3 import Error

try:
    from elixer import global_config as G
except:
    import global_config as G

log = G.Global_Logger('utilities')
log.setlevel(G.LOG_LEVEL)


def comological_volume(solid_angle,z1,z2,cosmo=None):
    """
    compute a simple volume between two redshifts given an area to project through and a cosmology

    #note: if want a proper volume, since this is a projection through an area, would need two factors of 1+z
    #      for the two transverse components of the solid angle
    #      This is NOT the same as placing a co-moving volume at some redshift, as that would be three factors of 1+z

    :param solid_angle: in sq.degree
    :param z1: low redshift
    :param z2: high redshift
    :return: comoving volume in Mpc3 or None
    """

    try:
        # remember, quad can't handle units, so only pass around values
        def comoving_vol_dz(z):  # co-moving volume in dz slice, Mpc per sr
            #basically Hogg eq 28, but with comoving distance instead of angular diameter distance * (1+z)**2
            return  np.power(cosmo.comoving_distance(z).to("Mpc").value, 2) / cosmo.efunc(z)

        if cosmo is None:
            cosmo = FlatLambdaCDM(H0=70., Om0=0.3, Tcmb0=2.725)

        DH = (C.c.to("km/s") / cosmo.H0).value  # Hubble distance, c/H0 in km/s/Mpc
               # area  (arcsec to sr)           *  c/H0   * integration of comoving volume between to redshifts
               # (sr)                             (1/Mpc)    (Mpc / sr)
        return solid_angle * (np.pi / 180.) ** 2 * DH      * quad(comoving_vol_dz, z1, z2)[0]
                                                             #just want 1st element from quad (second is the error)

    except:
        log.debug("Exception! in utilities::cosmological_volume")

    return None

def fracdiff(x1,x2,x1e=0.0,x2e=0.0):
    """
    return the fractional differnce between the two numbers (x1,x2)

    :param x1:
    :param x2:
    :param x1e:  optional error on x1
    :param x2e:  optional error on x2
    :return:
    """

    try:
        delta = 2.0 * abs((x1-x2)/(x1+x2))

        x1x2sq = (x1+x2)**2
        d_x1 = 4*x1/x1x2sq
        d_x2 = 4*x2/x1x2sq #technically -4* ... but it won't matter below

        delta_err = np.sqrt((d_x1*x1e)**2 + (d_x2*x2e)**2)

        return delta, delta_err
    except:
        return None, None


def find_fplane(date): #date as yyyymmdd string
    """Locate the fplane file to use based on the observation date

        Parameters
        ----------
            date : string
                observation date as YYYYMMDD

        Returns
        -------
            fully qualified filename of fplane file
    """
    #todo: validate date

    filepath = G.FPLANE_LOC
    fplane = None
    if filepath[-1] != "/":
        filepath += "/"
    files = glob.glob(filepath + "fplane*.txt")

    if len(files) > 0:
        target_file = filepath + "fplane" + date + ".txt"

        if target_file in files: #exact match for date, use this one
            fplane = target_file
        else:                   #find nearest earlier date
            files.append(target_file)
            files = sorted(files)
            #sanity check the index
            i = files.index(target_file)-1
            if i < 0: #there is no valid fplane
                log.info("Warning! No valid fplane file found for the given date. Will use oldest available.", exc_info=True)
                i = 0
            fplane = files[i]
    else:
        log.error("Error. No fplane files found.", exc_info = True)

    return fplane

def build_fplane_dicts(fqfn):
    """Build the dictionaries maping IFUSLOTID, SPECID and IFUID

        Parameters
        ----------
        fqfn : string
            fully qualified file name of the fplane file to use

        Returns
        -------
            ifuslotid to specid, ifuid dictionary
            specid to ifuid dictionary
        """
    # IFUSLOT X_FP   Y_FP   SPECID SPECSLOT IFUID IFUROT PLATESC
    if fqfn is None:
        log.error("Error! Cannot build fplane dictionaries. No fplane file.", exc_info=True)
        return {},{}

    ifuslot, specid, ifuid = np.loadtxt(fqfn, comments='#', usecols=(0, 3, 5), dtype = int, unpack=True)
    ifuslot_dict = {}
    cam_ifu_dict = {}
    cam_ifuslot_dict = {}

    for i in range(len(ifuslot)):
        if (ifuid[i] < 900) and (specid[i] < 900):
            ifuslot_dict[str("%03d" % ifuslot[i])] = [str("%03d" % specid[i]),str("%03d" % ifuid[i])]
            cam_ifu_dict[str("%03d" % specid[i])] = str("%03d" % ifuid[i])
            cam_ifuslot_dict[str("%03d" % specid[i])] = str("%03d" % ifuslot[i])

    return ifuslot_dict, cam_ifu_dict, cam_ifuslot_dict


def id_from_coord(ra,dec,shot=None):
    """
    limited to 19 digits with leading char 8 to fit in a signed int64

    given ra, dec and optional shot returns a close to unique value (to nearest 0.36 arcsec in ra and dec) with the
    last 2 digits of the year and the sum of the day of year and the shotID (+366).
    Form:
    first digit: 8 if Dec is non-negative, 7 if negative
    next 7: ra as [3],[4] in decimal degrees, zero filled
    next 6: dec as [2],[4] in decimal degrees, zero filled
    next 2: last two of year, zero filled
    next 3: day of year (001 to 366) + observation number + 366

    :param ra: decimal degree
    :param dec: decimal degree
    :param shot: int or string datevobs
    :return: None or 19 digit np.int64
    """

    try:
        idstr = '7' if dec < 0 else '8'
        idstr += str(int(round(ra*1e4))).zfill(7)
        idstr +=  str(int(abs(round(dec*1e4)))).zfill(6)

        if shot is not None:
            try:
                idstr += str(shot)[2:4] #last 2 digits of year ... that will be okay for a while
                day = date(int(str(shot)[0:4]),int(str(shot)[4:6]),int(str(shot)[6:8])).timetuple().tm_yday #day of year, starts with 1
                obs = int(str(shot)[-3:]) + 366 #so there is no overlap with day of year; also starts with 1
                idstr += str(min(999,obs+day)) #smallest possible value is 367 (Jan 01 + observation 001 = 001 + 365 + 001)
            except:
                idstr += "00000"
        else:
            idstr += "00000"

        return np.int64(idstr)

    except: # Exception as e:
        #print(e)
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

def weighted_mean(vals,weights):
    """
    weighted mean using the weights
    :param vals:
    :param errs:
    :return:
    """

    try: #might not be same length? could have nans in vals that are not in weights
        sel = np.logical_not( np.isnan(vals) | np.isnan(weights))
        return np.sum(vals[sel] * weights[sel])/np.nansum(weights[sel])
    except:
        return None

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
# def open_file_from_tar(tarfn,fqfn=None,fn=None):
#     """
#
#     one of fqfn must be supplied
#
#     :param tarfn: tar filename (fully qualified if necessary
#     :param fqfn: fully qualified filename inside the tarfile
#     :param fn:  filename only inside the tarfile (i.e. tar-file internal path is unknown)
#     :return:
#     """
#
#     try:
#
#         if fqfn is None and fn is None:
#             log.debug(f"utilities, open tarfile, no subfile specified")
#             return None
#
#         if not op.exists(tarfn):
#             log.debug(f"utilities, open tarfile {tarfn} does not exist")
#             return None
#
#         if not tar.is_tarfile(tarfn):
#             log.debug(f"utilities, open tarfile {tarfn} is not a tar file")
#             return None
#
#         tarfile = tar.open(name=tarfn)
#
#         if fqfn is not None:
#             file = tarfile.extractfile(fqfn)
#         elif fn is not None:
#             file = tarfile.extractfile(fn)
#
#         if file is not None:
#             return file
#
#     except:
#         log.debug(f"Exception attempting to fetch sub-file {fqfn} or {fn} from {tarfn}")
#         return None


def open_file_from_tar(tarfn, fqfn=None, fn=None,workdir=None): #, close_tar=True):
    """

    one of fqfn must be supplied
    Notice: There is some file I/O, so you may want to change to a temporary directory (see change_wd())

    :param tarfn: tar filename (fully qualified if necessary) or a tarfile handle
    :param fqfn: fully qualified filename inside the tarfile
    :param fn:  filename only inside the tarfile (i.e. tar-file internal path is unknown)
    #:param close_tar: if True, explcitly close the tarfile handle.
    :return: file handle to the fits file and its path
    """

    try:

        if fqfn is None and fn is None:
            log.debug(f"utilities, open tarfile, no subfile specified")
            return None, None

        if isinstance(tarfn, str):
            if not op.exists(tarfn):
                log.debug(f"utilities, open tarfile {tarfn} does not exist")
                return None, None

            if not tar.is_tarfile(tarfn):
                log.debug(f"utilities, open tarfile {tarfn} is not a tar file")
                return None, None

            _tarfile = tar.open(name=tarfn)
        else:
            _tarfile = tar.open(fileobj=tarfn)

        file_path = None
        if fqfn is not None:
            file = _tarfile.extractfile(fqfn)
        elif fn is not None:

            all_fqfn = np.array(_tarfile.getnames())
            if '?' in fn or '*' in fn:  # wildcards
                match_fn = fnmatch.filter(all_fqfn, fn)
                if len(match_fn) == 0:
                    file = None
                    log.debug(f"No matching files in tar.")
                elif len(match_fn) == 1:
                    file_path = match_fn[0]
                    log.debug(f"Found 1 match: {file_path}")
                    file = _tarfile.extractfile(file_path)

                else:
                    log.debug(f"{len(match_fn)} matching files in tar: {match_fn}")
                    file = None
            else:
                sel = np.array([fn in fqfn for fqfn in all_fqfn])
                ct = np.count_nonzero(sel)
                if ct == 0:  # no matches
                    file = None
                    log.debug("No matching files in tar.")
                elif ct == 1:
                    file = _tarfile.extractfile(all_fqfn[sel][0])
                else:  # more than one
                    log.debug(f"{ct} matching files in tar: {all_fqfn[sel]}")
                    file = None

        #NO. This is a stream. Need to keep it open. Let the GC take care of it.
        # try:
        #     if close_tar:
        #         _tarfile.close()
        # except:
        #     pass

        return file, file_path

    except Exception as E:
        log.error(f"Exception attempting to fetch sub-file {fqfn} or {fn} from {tarfn}:\n {E}")
        return None, None

def list_shot_types(yyyymmdd,count_exp=False,path="/work/03946/hetdex/maverick/"):
    """

    :param yyyymmdd:
    :param count_exp: if True, report the number of exposures and total number of files. Takes much longer to run.
    :param path: 
    :return:
    """

    try:

        #T = Table(dtype=[('type', str), ('shotid',int), ('num_exp', int), ('num_fits', int), ('path',str),('file',str)])
        T = Table(dtype=[('type', str), ('shotid', int), ('num_exp', int), ('num_fits', int), ('file', str)])

        if type(yyyymmdd) is int:
            yyyymmdd = str(yyyymmdd)

        tarfns = glob.glob(os.path.join(path, yyyymmdd, "virus/*.tar"))

        stl = []
       # pth = []
        typ = []
        nex = []
        nfi = []
        sid = []

        for fn in tqdm(tarfns):
            with tar.open(fn, 'r') as tarf:
                nextname = tarf.next().path

                stl.append(os.path.basename(fn))
                typ.append(nextname[-8:-5])
                #pth.append(os.path.dirname(nextname))
               # print(nextname.split("/"))
                sid.append(int(nextname.split("/")[0][5:]))

                if count_exp:
                    all_names = tarf.getnames()  # this is SLOW
                    nfi.append(len(all_names))
                    lx = [n.split("/")[1] for n in all_names]
                    nex.append(len(np.unique(lx)))
                else:
                    nex.append(-1)
                    nfi.append(-1)

        #sort by shotid
        st = np.argsort(stl)
        stl = np.array(stl)[st]
       # pth = np.array(pth)[st]
        typ = np.array(typ)[st]
        sid = np.array(sid)[st]
        nfi = np.array(nfi)[st]
        nex = np.array(nex)[st]

        #for t, f, x, i,s,p in zip(typ, stl, nex, nfi,sid,pth):
        #    T.add_row([t,s,x,i,p,f])
        for t, f, x, i,s in zip(typ, stl, nex, nfi,sid):
            T.add_row([t,s,x,i,f])

        return T

    except Exception as E:
        log.error(f"Exception in list_shot_types\n{E}")
        return None




def get_ifus_in_shot(date,shot):
    """
    Grab the tar file for the datevshot and enumerate the IFUs. Return list of IFU SLOT ID integers

    :param date:
    :param shot:

    :return:
    """

    try:
        ifulist = []

        tarfn = op.join(G.HETDEX_WORK_TAR_BASEPATH, str(date), f"virus/virus{str(shot).zfill(7)}.tar")
        if op.exists(tarfn):
            log.debug(f"Using {G.HETDEX_WORK_TAR_BASEPATH} basepath ...")
        elif op.exists(op.join(G.HETDEX_CORRAL_TAR_BASEPATH, f"{date}.tar")):
            log.debug(f"Using {G.HETDEX_CORRAL_TAR_BASEPATH} basepath ...")
            # we need to fetch a sub-tar file
            tarfn, file_path = open_file_from_tar(tarfn=op.join(G.HETDEX_CORRAL_TAR_BASEPATH, f"{date}.tar"),
                                         fqfn=op.join(str(date), f"virus/virus{str(shot).zfill(7)}.tar"))
                                         #close_tar=False) #need to keep it open, at least for now
        else:
            log.debug("No viable path.")

        if isinstance(tarfn, str):
            tarfile = tar.open(name=tarfn)
        else:
            tarfile = tar.open(fileobj=tarfn)

        all_fqfn = np.array(tarfile.getnames())

        for fn in all_fqfn:
            ifu = int(fn.split("_")[1][0:3])
            ifulist.append(ifu)

        tarfile.close()
        ifulist = np.unique(ifulist)

    except Exception as E:
        log.error(f"Exception attempting to process raw multifits file (get_ifus_in_shot()):\n {E}")

    return ifulist


def get_multifits(date,shot,exp,ifuslot=None,amp=None,longfn=None,flatfile_path=None,raw=False,calmonth=None,
                  workdir="./",clean=0):
    """
    load a single multi*fits file from original raw data

    Notice: There is some file I/O, so you may want to change to a temporary directory (see change_wd())

    !Warning! This operates in its own directory. If you are running multiple calls in paralle with raw=False,
             you MUST run each under its own unique directory. (see change_wd())

    Must always supply date, shot, exp and either ifuid and amp OR longfn (example below)

    Note: first load of the base .tar file can be costly, especially if it is from the corral-repl path
          which has tarred up all data (and not just HETDEX) for an entire day, but subsequent extractions
          are much faster as the tar file is cached (e.g. if calling for other IFUs or other exposures on the same
          datevshot)

    :param date:   (int) as YYYYMMDD
    :param shot:   (int)
    :param exp:    (int) exposure ID
    :param ifuslot:  (int)
    :param amp:    (str) one of "LL","LU","RL","RU"
    :param longfn: (str) example: "20230103T105730.2_105LL_sci.fits" ... a "raw" file
    :param raw:    (bool) If True, return as a skysubtracted but otherwise unprocessed (raw) frame from the telescope.
                          If False (default), rectify and return processed frame.
    :param calmonth: (int) or (str) should be of form YYYYMM. The first 6 digits are used.
    :param workdir: set the working directory (default is the cwd, "./")
    :param clean: clean up after vred; 0 = No, 1 = vred files (but leave the multifits), 2 = everything
                  note: the working directory is only deleted if clean == 2 AND the directory was created by this func

    :return: stream handle to the fits file (raw or processed) (ExFileObject or BufferedReader)
    """

    try:

        #todo: could add more parameter validation and have nicer error handling
        multifits = None

        #must supply ifuslot and amp OR longfn
        if ifuslot is None or amp is None:
            if longfn is None:
                #give feedback
                log.warning("Must supply date,shot,exp AND ifuslot, amp OR longfn")
                return False
            else:
                multifn = f"virus{str(shot).zfill(7)}/exp{str(exp).zfill(2)}/virus/{longfn}"
                #assuming this is of the standard form, get the ifuslot and amp ... if this fails, just let it bomb
                #and have the outer try trap it and inform the caller
                toks = longfn.split("_")
                ifuslot = int(toks[1][0:3])
                amp = toks[1][3:5]
        else:
            multifn = f"virus{str(shot).zfill(7)}/exp{str(exp).zfill(2)}/virus/*_{str(ifuslot).zfill(3)}{amp}_*.fits"

        if flatfile_path is not None:
            try:
                files = glob.glob(op.join(flatfile_path,f"{date}/virus/virus{str(shot).zfill(7)}/exp{str(exp).zfill(2)}/"
                                                        f"virus/multi_*_{str(ifuslot).zfill(3)}_*_{amp}.fits"))
                if len(files)==0:
                    log.debug(f"Not found as flat file. Moving on to check tar files.")
                elif len(files)==1:
                    log.debug(f"Found as flat file: {files[0]}")
                    multifits = open(files[0],"rb")
                    return multifits
                else:
                    log.debug(f"Unexpected number of matches {len(files)} found. Ignoring and moving on to check tar files.")

            except Exception as E:
                log.error(f"Exception checking for flat file:\n {E}")

        tarfile = op.join(G.HETDEX_WORK_TAR_BASEPATH, str(date), f"virus/virus{str(shot).zfill(7)}.tar")
        if op.exists(tarfile):
            log.debug(f"Using {G.HETDEX_WORK_TAR_BASEPATH} basepath ...")
        elif op.exists(op.join(G.HETDEX_CORRAL_TAR_BASEPATH, f"{date}.tar")):
            log.debug(f"Using {G.HETDEX_CORRAL_TAR_BASEPATH} basepath ...")
            # we need to fetch a sub-tar file
            tarfile, file_path = open_file_from_tar(tarfn=op.join(G.HETDEX_CORRAL_TAR_BASEPATH, f"{date}.tar"),
                                         fqfn=op.join(str(date), f"virus/virus{str(shot).zfill(7)}.tar"))
                                         #close_tar=False) #need to keep it open, at least for now
            # in this case, tarfile is no longer a filename but an actual file ... either way works
        else:
            log.debug("No viable path.")

        #this is the raw fits ... needs to be processed ... (e.g. rback script)
        multifits, file_path = open_file_from_tar(tarfile, fn=multifn)#,close_tar=close_tar)

        if raw: #we're done. caller only wanted the unprocessed file
            return multifits

        #else continue to processing
        try:

            clean_files = [] #if cleaning, always clean
            clean_multi = [] #just the mutlifite
            clean_wd = False

            if not op.isdir(workdir):
                os.mkdir(workdir)
                if clean > 1:
                    clean_wd = True

            hdulist = fits.open(multifits)
            clean_files.append(op.join(workdir, "in.fits"))
            hdulist.writeto(op.join(workdir,"in.fits"), overwrite=True)  # work on this with Karl's vred
            hdulist.close()

            # vred needs a vred.in file
            clean_files.append(op.join(workdir,"vred.in"))
            with open(op.join(workdir,"vred.in"), "w+") as f:
                if calmonth is None:
                    f.write(f"{str(date)[0:6]} 2\n")  # always use '2' ???

                else:
                    f.write(f"{str(calmonth)[0:6]} 2\n")  # always use '2' ???


            # vred also wants 'list' (seems to just use it to set the NAME0 card in the HDU, but pukes if it (list) is not there)
            # I don't know how important (if at all, this card is)
            clean_files.append(op.join(workdir,"list"))
            with open(op.join(workdir,"list"), "w+") as f:
                f.write(f"{str(date)}/virus/{file_path}\n")

            try:
                cpfiles = glob.glob(os.path.join(G.HETDEX_VRED_FIBERLOC_BASEPATH,f"fiber_loc_???_{str(ifuslot).zfill(3)}_???_{amp}.txt"))

                if len(cpfiles) > 0:
                    #there might be more than one match, a different specid for the given ifuslot at different times
                    #just copy them all and let vred choose the right one
                    for fn in cpfiles:
                        clean_files.append(op.join(workdir, os.path.basename(fn)))
                        try:
                            shutil.copy(fn, workdir)
                        except Exception as E:
                            log.warning(f"Unable to copy fiber loc file {fn}. Not fatal.\n {E}")
                else:# len(cpfiles) == 0:
                    log.warning(f"Unable to copy fiber loc file. No file found. Not fatal.")

            except Exception as E:
                log.warning(f"Unable to copy fiber loc file. Not fatal.\n {E}")

            # print ...make full path to vred? or dynamic
            clean_files.append(op.join(workdir, "vred"))
            if not os.path.isfile(op.join(workdir,"vred")):
                log.debug("Copying vred ...")
                shutil.copy(G.HETDEX_VRED_FQFN, op.join(workdir,"vred"))
                log.debug("Done copy")

            #see if there is any left over output from the last run
            try:
                #must remove out5.fits and out otherwise vred will complain and abend on next run in same location
                clean_files.append(op.join(workdir, "out5.fits"))
                clean_files.append(op.join(workdir, "out"))
                os.remove(op.join(workdir,"out5.fits"))
                os.remove(op.join(workdir,"out"))
                #also remove any of the multi fits that we are going to replace
                rmfiles = glob.glob(op.join(workdir,f"multi*{str(ifuslot).zfill(3)}*{amp}.fits"))
                for fn in rmfiles:
                    os.remove(fn)
            except:# Exception as E:
                pass #log.debug(f"**** {E}")


            #vred does not like it if the output files already exist, so clean them
            try:
                files = glob.glob(op.join(workdir,f"multi_*_{str(ifuslot).zfill(3)}_*_{amp}.fits"))
                for fn in files:
                    os.remove(fn)
            except:
                pass

            # run vred
            #p1 = subprocess.run([op.join(workdir,"vred")],cwd=workdir)
            p1 = subprocess.run(["vred"], cwd=workdir)

            if p1.returncode == 0:
                #need the multifits name:
                files = glob.glob(op.join(workdir,f"multi_*_{str(ifuslot).zfill(3)}_*_{amp}.fits"))
                if len(files)==0:
                    log.error(f"Failure to process raw file. multi*fits output not found.")
                    multifits = None
                elif len(files)==1:
                    #with open(files[0],"rb") as f:
                    #    multifits = f.read()
                    clean_multi.append(files[0])
                    multifits =  open(op.realpath(files[0]),"rb")
                else:
                    log.error(f"Failure to process raw file. Unexpected number of matches {len(files)} found.")
                    multifits = None
            else:
                log.error(f"Failure to process multi*fits file. vred rc = {p1.returncode}:")
                multifits = None
                clean_multi += files
        except Exception as E:
            log.error(f"Exception attempting to process raw multifits file (get_multifits()):\n {E}")


        if clean:
            if clean > 0:
                for fn in clean_files:
                    try:
                        os.remove(fn)
                        log.debug(f"removed {fn}")
                    except:
                        log.debug(f"failed to remove {fn}")
            if clean > 1:
                for fn in clean_multi:
                    try:
                        os.remove(fn)
                        log.debug(f"removed {fn}")
                    except:
                        log.debug(f"failed to remove {fn}")

                if clean_wd:
                    shutil.rmtree(workdir)

        return multifits

    except Exception as E:
        log.error(f"Exception attempting to load single multifits file (get_multifits()):\n {E}")

        try:
            if clean:
                if clean > 0:
                    for fn in clean_files:
                        try:
                            os.remove(fn)
                            log.debug(f"removed {fn}")
                        except:
                            log.debug(f"failed to remove {fn}")
                if clean > 1:
                    for fn in clean_multi:
                        try:
                            os.remove(fn)
                            log.debug(f"removed {fn}")
                        except:
                            log.debug(f"failed to remove {fn}")

                    if clean_wd:
                        shutil.rmtree(workdir)
        except:
            pass

        return None


def run_vred(date,shot,exp,ifuslot,amp,calmonth=None,workdir="./",clean=0):
    """

    based on get_multifits, but focused on just setting up and running Karl's vred for all exposures under one shot

    this is a forced RUN .. ignore any existing multifits



    load a single multi*fits file from original raw data

    Notice: There is some file I/O, so you may want to change to a temporary directory (see change_wd())

    !Warning! This operates in its own directory. If you are running multiple calls in paralle with raw=False,
             you MUST run each under its own unique directory. (see change_wd())

    Must always supply date, shot, exp and either ifuid and amp OR longfn (example below)

    Note: first load of the base .tar file can be costly, especially if it is from the corral-repl path
          which has tarred up all data (and not just HETDEX) for an entire day, but subsequent extractions
          are much faster as the tar file is cached (e.g. if calling for other IFUs or other exposures on the same
          datevshot)

    :param date:   (int) as YYYYMMDD
    :param shot:   (int)
    :param exp:    (int) exposure ID
    :param ifuslot:  (int)
    :param amp:    (str) one of "LL","LU","RL","RU"
    :param calmonth: (int) or (str) should be of form YYYYMM. The first 6 digits are used.
    :param workdir: set the working directory (default is actual cwd "./")
    :param clean: clean up after vred; 0 = No, 1 = vred files (but leave the multifits), 2 = everything
                note: the working directory is only deleted if clean == 2 AND the directory was created by this func

    :return: stream handle to the fits file (raw or processed) (ExFileObject or BufferedReader)
    """

    try:

        #todo: could add more parameter validation and have nicer error handling
        multifits = None

        #must supply ifuslot and amp OR longfn
        if ifuslot is None or amp is None:
                #give feedback
            log.warning("Must supply date,shot,exp AND ifuslot, amp OR longfn")
            return False
        else:
            multifn = f"virus{str(shot).zfill(7)}/exp{str(exp).zfill(2)}/virus/*_{str(ifuslot).zfill(3)}{amp}_*.fits"

        tarfile = op.join(G.HETDEX_WORK_TAR_BASEPATH, str(date), f"virus/virus{str(shot).zfill(7)}.tar")

        if op.exists(tarfile):
            log.debug(f"Using {G.HETDEX_WORK_TAR_BASEPATH} basepath ...")
        elif op.exists(op.join(G.HETDEX_CORRAL_TAR_BASEPATH, f"{date}.tar")):
            log.debug(f"Using {G.HETDEX_CORRAL_TAR_BASEPATH} basepath ...")
            # we need to fetch a sub-tar file
            tarfile, file_path = open_file_from_tar(tarfn=op.join(G.HETDEX_CORRAL_TAR_BASEPATH, f"{date}.tar"),
                                         fqfn=op.join(str(date), f"virus/virus{str(shot).zfill(7)}.tar"))
                                         #close_tar=False) #need to keep it open, at least for now
            # in this case, tarfile is no longer a filename but an actual file ... either way works
        else:
            log.debug("No viable path.")

        #this is the raw fits ... needs to be processed ... (e.g. rback script)
        multifits, file_path = open_file_from_tar(tarfile, fn=multifn)#,close_tar=close_tar)

        try:
            clean_files = []  # if cleaning, always clean
            clean_multi = []  # just the mutlifite
            clean_wd = False

            if not op.isdir(workdir):
                os.mkdir(workdir)
                if clean > 1:
                    clean_wd = True

            hdulist = fits.open(multifits)
            clean_files.append(op.join(workdir, "in.fits"))
            hdulist.writeto("in.fits", overwrite=True)  # work on this with Karl's vred

            try:
                specid_str = hdulist[0].header['SPECID']
                ifuid_str = hdulist[0].header['IFUID']
                log.debug(f"Full IFU ID: {specid_str}_{str(ifuslot).zfill(3)}_{ifuid_str}")
            except:
                specid_str = "???"
                ifuid_str = "???"
                log.debug("Using wildcards for specid and ifuid")

            hdulist.close()

            # vred needs a vred.in file
            clean_files.append(op.join(workdir, "vred.in"))
            with open(op.join(workdir,"vred.in"), "w+") as f:
                if calmonth is None:
                    f.write(f"{str(date)[0:6]} 2\n")  # always use '2' ???
                else:
                    f.write(f"{str(calmonth)[0:6]} 2\n")  # always use '2' ???

            # vred also wants 'list' (seems to just use it to set the NAME0 card in the HDU, but pukes if it (list) is not there)
            # I don't know how important (if at all, this card is)
            clean_files.append(op.join(workdir, "list"))
            with open(op.join(workdir,"list"), "w+") as f:
                f.write(f"{str(date)}/virus/{file_path}\n")

            try:
                #can we get the extactly right one with specid_ifuslot_ifuid ??
                #since we have the date, shot, there can be only one
                cpfiles = glob.glob(os.path.join(G.HETDEX_VRED_FIBERLOC_BASEPATH,f"fiber_loc_{specid_str}_{str(ifuslot).zfill(3)}_{ifuid_str}_{amp}.txt"))

                if len(cpfiles) > 0:
                    #there might be more than one match, a different specid for the given ifuslot at different times
                    #just copy them all and let vred choose the right one
                    for fn in cpfiles:
                        try:
                            shutil.copy(fn, workdir)
                            clean_files.append(op.join(workdir, os.path.basename(fn)))
                        except Exception as E:
                            log.warning(f"Unable to copy fiber loc file {fn}. Not fatal.\n {E}")
                else:# len(cpfiles) == 0:
                    log.warning(f"Unable to copy fiber loc file. No file found. Not fatal.")

            except Exception as E:
                log.warning(f"Unable to copy fiber loc file. Not fatal.\n {E}")

            # print ...make full path to vred? or dynamic
            clean_files.append(op.join(workdir, "vred"))
            if not os.path.isfile(op.join(workdir,"vred")):
                log.debug("Copying vred ...")
                shutil.copy(G.HETDEX_VRED_FQFN, op.join(workdir,"vred"))
                log.debug("Done copy")

            #see if there is any left over output from the last run
            try:
                #must remove out5.fits and out otherwise vred will complain and abend on next run in same location
                clean_files.append(op.join(workdir, "out5.fits"))
                clean_files.append(op.join(workdir, "out"))
                os.remove(op.join(workdir,"out5.fits"))
                os.remove(op.join(workdir,"out"))
                #also remove any of the multi fits that we are going to replace
                rmfiles = glob.glob(op.join(workdir,f"multi*{str(ifuslot).zfill(3)}*{amp}.fits"))
                for fn in rmfiles:
                    os.remove(fn)
            except:# Exception as E:
                pass #log.debug(f"**** {E}")

            # vred does not like it if the output files already exist, so clean them
            try:
                files = glob.glob(op.join(workdir, f"multi_*_{str(ifuslot).zfill(3)}_*_{amp}.fits"))
                for fn in files:
                    os.remove(fn)
            except:
                pass

            # run vred
            log.debug("Executing vred ...")
            #p1 = subprocess.run([op.join(workdir,"vred")])
            p1 = subprocess.run(["vred"], cwd=workdir)
            #p1 = subprocess.run([op.join(workdir, "vred")], cwd=workdir)

            if p1.returncode == 0:
                #need the multifits name:
                files = glob.glob(op.join(workdir,f"multi_*_{str(ifuslot).zfill(3)}_*_{amp}.fits"))
                if len(files)==0:
                    log.error(f"Failure to process raw file. multi*fits output not found.")
                    multifits = None
                elif len(files)==1:
                    #with open(files[0],"rb") as f:
                    #    multifits = f.read()
                    clean_multi.append(op.join(workdir, files[0]))
                    multifits =  open(op.realpath(files[0]),"rb")
                else:
                    log.error(f"Failure to process raw file. Unexpected number of matches {len(files)} found.")
                    multifits = None
                    clean_multi += files
            else:
                log.error(f"Failure to process multi*fits file. vred rc = {p1.returncode}:")
                multifits = None
        except Exception as E:
            log.error(f"Exception attempting to process raw multifits file (get_multifits()):\n {E}")


        if clean:
            if clean > 0:
                for fn in clean_files:
                    try:
                        os.remove(fn)
                        log.debug(f"removed {fn}")
                    except:
                        log.debug(f"failed to remove {fn}")
            if clean > 1:
                for fn in clean_multi:
                    try:
                        os.remove(fn)
                        log.debug(f"removed {fn}")
                    except:
                        log.debug(f"failed to remove {fn}")

                if clean_wd:
                    shutil.rmtree(workdir)

        return multifits

    except Exception as E:
        log.error(f"Exception attempting to load single multifits file (get_multifits()):\n {E}")

        try:
            if clean:
                if clean > 0:
                    for fn in clean_files:
                        try:
                            os.remove(fn)
                            log.debug(f"removed {fn}")
                        except:
                            log.debug(f"failed to remove {fn}")
                if clean > 1:
                    for fn in clean_multi:
                        try:
                            os.remove(fn)
                            log.debug(f"removed {fn}")
                        except:
                            log.debug(f"failed to remove {fn}")

                    if clean_wd:
                        shutil.rmtree(workdir)
        except:
            pass

        return None

    #end run_vred


def change_wd(workdir,create=True):
    """

    :param workdir:
    :param create:
    :return: True if successful, False, if fail; previous wd, new wd (workdir)
    """
    try:
        prev_cwd, cwd = None, None
        if workdir is not None:
            prev_cwd = os.getcwd()
            if cwd != os.path.realpath(workdir):
                if not op.isdir(workdir):
                    if create:
                        try:
                            os.mkdir(workdir)
                        except Exception as E:
                            log.error(f"Cannot create {workdir} \n {E}")
                            return False, prev_cwd, prev_cwd
                    else:
                        log.error(f"Directory does not exist and not set to create: {workdir}")
                        return False, prev_cwd, prev_cwd
                try:
                    os.chdir(workdir)
                    cwd = os.getcwd()
                except Exception as E:
                    log.error(f"Cannot cd to {workdir} \n {E}")
                    return False, prev_cwd, prev_cwd

        return True, prev_cwd, cwd

    except Exception as E:
        log.error(f"Exception in change_wd\n {E}")
        return False, None, None



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


def simple_linear_interp(x1,y1,x2,y2,x,clip=False):
    """

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x: the point you want to evaluate
    :param clip: if true, returned value must lie between y1 and y2 (basically becomes like sigmoid_linear_interp)
    :return:
    """
    try:
        #y = m x + b
        m = (y2-y1)/(x2-x1)
        b = y1 - m * x1

        if clip:
            return np.clip(m * x + b, min(y1,y2),max(y1,y2))
        else:
            return m * x + b
    except:
        return None


def sigmoid_linear_interp(x1, y1, x2, y2, x):
    """
    kind of like a sigmoid in that is is constant (or nearly) away from the thresholds, but the transition
    here is linear instead

    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x: the point you want to evaluate
    :return:
    """
    try:

        if x1 < x2:  #expected
            xp1 = x1
            xp2 = x2
            yp1 = y1
            yp2 = y2
        else:      #flip order
            xp1 = x2
            xp2 = x1
            yp1 = y2
            yp2 = y1

        if x < xp1:
            return yp1
        elif x > xp2:
            return yp2
        else:
            return simple_linear_interp(xp1, yp1, xp2, yp2, x)
    except:
        return None

