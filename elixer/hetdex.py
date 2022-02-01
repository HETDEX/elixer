
try:
    from elixer import global_config as G
    from elixer import line_prob
    from elixer import hetdex_fits
    from elixer import fiber as elixer_fiber
    from elixer import ifu as elixer_ifu  # only using to locate panacea files (elixer only uses individual fibers, not entire IFUs)
    from elixer import spectrum as elixer_spectrum
    from elixer import observation as elixer_observation
    from elixer import spectrum_utilities as SU
    from elixer import weighted_biweight
    from elixer import utilities as utils
    from elixer import shot_sky
    from elixer import galaxy_mask
    from elixer import cat_sdss #for the z-catalog
    from elixer import cat_gaia_dex
except:
    import global_config as G
    import line_prob
    import hetdex_fits
    import fiber as elixer_fiber
    import ifu as elixer_ifu  # only using to locate panacea files (elixer only uses individual fibers, not entire IFUs)
    import spectrum as elixer_spectrum
    import observation as elixer_observation
    import spectrum_utilities as SU
    import weighted_biweight
    import utilities as utils
    import shot_sky
    import galaxy_mask
    import cat_sdss #for the z-catalog
    import cat_gaia_dex


from hetdex_tools.get_spec import get_spectra as hda_get_spectra
from hetdex_api.shot import get_fibers_table as hda_get_fibers_table

from astropy.coordinates import SkyCoord
import astropy.units as U

import matplotlib
#matplotlib.use('agg')
import time
import dateutil.parser

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import io
from PIL import Image

from astropy.io import fits as pyfits
from astropy.coordinates import Angle
from astropy import units as units
from astropy.stats import sigma_clipped_stats

#from astropy.stats import biweight_midvariance
from astropy.modeling.models import Moffat2D, Gaussian2D
from astropy.visualization import ZScaleInterval
from photutils import CircularAperture, aperture_photometry
from scipy.ndimage.filters import gaussian_filter

#from scipy.stats import skew, kurtosis

import glob
import re
from pyhetdex.cure.distortion import Distortion
import pyhetdex.tools.files.file_tools as ft
from pyhetdex.het.ifu_centers import IFUCenter
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection import TangentPlane as TP
import os
import fnmatch
import os.path as op
from copy import copy, deepcopy

import tables

#todo: write a class wrapper for log
#an instance called log that has functions .Info, .Debug, etc
#they all take a string (the message) and the exc_info flag
# inside they prepend a time to the string and then pass along to the regular logger

#log = G.logging.getLogger('hetdex_logger')
#log.setLevel(G.logging.DEBUG)

log = G.Global_Logger('hetdex_logger')
log.setlevel(G.LOG_LEVEL)

CONFIG_BASEDIR = G.CONFIG_BASEDIR
VIRUS_CONFIG = G.VIRUS_CONFIG #op.join(CONFIG_BASEDIR,"virus_config")
FPLANE_LOC = G.FPLANE_LOC #op.join(CONFIG_BASEDIR,"virus_config/fplane")
IFUCEN_LOC = G.IFUCEN_LOC #op.join(CONFIG_BASEDIR,"virus_config/IFUcen_files")
DIST_LOC = G.DIST_LOC #op.join(CONFIG_BASEDIR,"virus_config/DeformerDefaults")
PIXFLT_LOC = G.PIXFLT_LOC #op.join(CONFIG_BASEDIR,"virus_config/PixelFlats/20161223")

PLOT_SUMMED_SPECTRA = True #zoom-in plot of top few fibers
MAX_2D_CUTOUTS = 4 #~ 5x3 of 2d cutouts  +1 for the summed cutout

SIDE = elixer_fiber.SIDE
#!!! REMEBER, Y-axis runs 'down':  Python 0,0 is top-left, DS9 is bottom-left
#!!! so in DS9 LU is 'above' LL and RL is 'above' RU
AMP  = elixer_fiber.AMP #["LU","LL","RL","RU"] #in order from bottom to top
AMP_OFFSET = elixer_fiber.AMP_OFFSET# {"LU":1,"LL":113,"RL":225,"RU":337}



#lifted from Greg Z.
dist_thresh = 2.  # Fiber Distance (arcsecs)
#todo: change to full width of frame?? (and change xl in build dict to this value, not the difference)
FRAME_WIDTH_X = 1032 #1024
AMP_HEIGHT_Y = 1032
xw = 24  # image width in x-dir
yw = 10  # image width in y-dir
#contrast1 = 0.9  # convolved image  # from Greg
#contrast2 = 0.5  # regular image # from Greg
contrast1 = 1.0  # convolved image # using normal zscale
contrast2 = 0.5  # regular image
contrast3 = 0.8  # regular image still has sky

res = [3, 9]
ww = xw * 1.9  # wavelength width
lyc_ww = 34 * 1.9 #for LyC wavelength width (2D cutouts)  = 1/2*(68AA wide) * 1.9 AA per pix at z=3.5

# number of pixels to either side of peak to fit to gaussian
#this value comes from looking at many detections and all but the widest will fit (and even the wide ones will
#be mostly captured) ... making it too wide picks up too much noise and throws the fit (especially for weaker S/N)
PEAK_PIXELS = 4

FLUX_CONVERSION_measured_w = [3000., 3500., 3540., 3640., 3740., 3840., 3940., 4040., 4140., 4240., 4340., 4440., 4540., 4640., 4740., 4840.,
     4940., 5040., 5140.,
     5240., 5340., 5440., 5500., 6000.]
FLUX_CONVERSION_measured_f = [1.12687e-18, 1.12687e-18, 9.05871e-19, 6.06978e-19, 4.78406e-19, 4.14478e-19, 3.461e-19, 2.77439e-19, 2.50407e-19,
     2.41462e-19, 2.24238e-19, 2.0274e-19, 1.93557e-19, 1.82048e-19, 1.81218e-19, 1.8103e-19, 1.81251e-19,
     1.80744e-19, 1.85613e-19, 1.78978e-19, 1.82547e-19, 1.85056e-19, 2.00788e-19, 2.00788e-19]

FLUX_CONVERSION_w_grid = np.arange(3000.0, 6000.0, 1.0)
FLUX_CONVERSION_f_grid = np.interp(FLUX_CONVERSION_w_grid, FLUX_CONVERSION_measured_w, FLUX_CONVERSION_measured_f)

FLUX_CONVERSION_DICT = dict(zip(FLUX_CONVERSION_w_grid,FLUX_CONVERSION_f_grid))



def gmag_vote_thresholds(wave):
    """
        Max idx = 30145, slope = 0.0007507, intercept: 20.450
        LyA Accuracy: 0.8827, Contamination: 0.0858
        OII Accuracy: 0.0000, Contamination: 0.2716

        Max idx = 48398, slope = 0.001211, intercept: 18.980
        LyA Accuracy: 0.7941, Contamination: 0.0311
        OII Accuracy: 0.4109, Contamination: 0.5000

        LyA above: 559, OII above: 18
        LyA between: 79, OII between: 42
        LyA below: 23, OII below: 59

    :param wave: wavelength of the emission line (as OII or LyA)
    :return: gband minimum mag for LyA vote, gband maxomum mag for OII vote
    """

    try:
        #DESI set about 660 LAE and 120 OII ... no correction
        # bright_gmag = 0.0007507 * wave + 20.450
        # faint_gmag = 0.001211 * wave + 18.980

        #DESI set about 660 LAE and 120 OII ... OII x5 to balance numbers
        bright_gmag = 0.0007507 * wave + 20.450
        faint_gmag = 0.0007507 * wave + 21.460

        return bright_gmag, faint_gmag
    except:
        log.warning("Exception in hetdex.py gmag_vote_thresholds().", exc_info=True)
        return 99.9,-99.9

def adjusted_mag_zero(mag_zero, z):
    """
    tweak the defined mag zero (where the LAE vs OII vote is zero weighted) by redshift
    Sets the zero at z = 2.5 and makes it a little brighter toward z = 2.0 [16% brighter] and fainter toward z = 3.5 [22% fainter]
    Uses factor of 1+z

    Per Andrew's paper (Leung+2017), sec 4.3, 4.3, at LyA z 2.5- 3.5, the equivalent of OII, the OII colors get
    a bit more red with weaker continuum and results in larger OII equivalent widths ... so the "danger" mag
    gets a little fainter

    with a sample of LAE and OII, get a linear fit like:   mag_zero = 0.0015 * wavelength + 17.2
    (trying to maximize accuracy (as (not Lya + missed Lya) / elixer LyA ) and minimize LyA contamination (as not LyA / elixer LyA)

    :param mag_zero:
    :param z:
    :return:
    """

    ##todo: for now (2021-12-14) just leave as is. Do not make a correction until we better understand this empirically
    #return mag_zero

    try:
        #this is very close to the more correct version of -2.5 log (f0/
        #at 7.5 (or 1+z)**3 cubed this is far too strong
        if z > 0:
            # adjust = 2.5 * np.log10((1+z) / 3.5) # 3.5 = 1 + 2.5 # 7.5x instead of 2.5x since want to use (1+z)**3
            #from DESI comparison; roughly 23.2 @ 3727 to 24.2 @4500 to 25.4 @ 5500
            #adjust = ((G.LyA_rest * (1+z)) - 4500.0) * G.LAE_MAG_SLOPE

            #use the faint slope from gmag_vote_thresholds
            bright, faint = gmag_vote_thresholds(G.LyA_rest * (1+z))
            return faint
            #adjust = faint - mag_zero

        else:
            adjust = 0
        #per above, this effect really only starts z > 2.5 (or OII z > 0.14)
        return mag_zero + adjust
    except:
        log.debug("Exception in hetdex.py adjust_mag_zero().", exc_info=True)
        return mag_zero

def flux_conversion(w): #electrons to ergs at wavelength w
    if w is None:
        return 0.0
    w = round(w)

    if w in FLUX_CONVERSION_DICT.keys():
        return FLUX_CONVERSION_DICT[w]
    else:
        log.error("ERROR! Unable to find FLUX CONVERSION entry for %f" %w)
        return 0.0


def blank_pixel_flat(xh=48,xl=0,yh=20,yl=0,vmin_pix = 0.9,vmax_pix = 1.1):
    # todo: this is really sloppy ... make a better/more efficient pattern
    #remember, this is inverted (so vmax_pix = white, vmin_pix = black
    pix_x = xh - xl + 1
    pix_y = yh - yl + 1
    pix_blank = np.zeros((pix_y, pix_x)) #yes, y then x and the same below
    pix_blank += vmax_pix
    v_avg = (vmax_pix + vmin_pix) / 2.
    try:
        for x in range(pix_x / 2):
            for y in range(pix_y / 2):
                pix_blank[y * 2, x * 2] = v_avg

        #borders
        pix_blank[0,:] = vmin_pix
        pix_blank[pix_y-1,:] = vmin_pix
        pix_blank[:,0] = vmin_pix
        pix_blank[:,pix_x-1] = vmin_pix
    except:
        pass

    return pix_blank


def flip_amp(amp=None, ampname=None, buf=None):
    """
    With HDR2 + only applies to pixel flats

    REMINDER: python is row, column (not X,Y) and orders from the "top down"
    so fliplr (left-right) is the second index (columns) which is the x coord
    and flipud (up-down) is the first index (row) or the y coord
    (row = Y, column = X)

    :param amp:
    :param ampname:
    :param buf:
    :return:
    """

    if (amp is None) or (buf is None):
        log.error("Amp designation or buffer is None.")
        return None

    try:
        #sanity check buf for correct size
        y,x = np.shape(buf)

        if (x != 1032) and (y != 1032):
            log.error('Amp buffer wrrong shape: (%d,%d). Expecting (1032,1032).' %(x,y))
            return None

        if (amp.upper() == 'LL') or (amp.upper() == 'RU'):
            pass #no need to bother logging
            #log.debug('Will not flip amp (%s). Already in correct orientation.' % amp)
        elif (amp.upper() == 'LU') or (amp.upper() == 'RL'):
            log.debug('Will flip amp (%s). Reverse X and Reverse Y.' % amp)
            buf =  np.fliplr(np.flipud(buf))
        else:
            log.warning("Unexpected AMP designation: %s" % amp)
            return None

        #this is essentially the  Config0 vs Config1 ... info
        #AMPNAME is not the same as the AMP (it comes from either [AMPNAME] or [AMPLIFIE] and has a somewhat different meaning
        if (ampname is not None) and ((ampname.upper() == 'LR') or (ampname.upper() == 'UL')):
            log.debug('Will flip amp (%s), ampname(%s). Reverse Y [config 0/1].' % (amp,ampname))
            buf = np.fliplr(buf)
    except:
        log.error("Exception in flip_amp.",exc_info=True)
        return None

    return buf


#lifted from Greg Z. make_visualization_detect.py
def get_w_as_r(seeing, gridsize, rstep, rmax, profile_name='moffat'):
    fradius = 0.75  # VIRUS
    if profile_name == 'moffat':
        alpha = 2.5  # hard coded in Cure
        gamma = seeing / 2.0 / np.sqrt(np.power(2.0, (1.0 / alpha)) - 1.0)
        profile = Moffat2D(alpha=alpha, gamma=gamma)
    else:
        sigma = seeing / 2.3548
        profile = Gaussian2D(x_stddev=sigma, y_stddev=sigma)
    x = np.linspace(-1 * (rmax + fradius + 0.5), (rmax + fradius + 0.5), gridsize)
    X, Y = np.meshgrid(x, x)
    Z = profile(X.ravel(), Y.ravel()).reshape(X.shape)
    Z /= np.sum(Z.ravel() * (x[1] - x[0]) ** 2)
    nstep = int(rmax / rstep) + 1
    r = np.linspace(0, rmax, nstep)
    xloc = np.interp(r, x, np.arange(len(x)))
    yloc = np.interp(np.zeros((nstep,)), x, np.arange(len(x)))
    positions = [xloc, yloc]
    apertures = CircularAperture(positions, r=fradius)
    phot_table = aperture_photometry(Z, apertures)
    return r, np.array(phot_table['aperture_sum'])

# def gaussian(x,x0,sigma,a=1.0):
#     if (x is None) or (x0 is None) or (sigma is None):
#         return None
#
#     return a*np.exp(-np.power((x - x0)/sigma, 2.)/2.)

def gaussian(x,x0,sigma,a=1.0,y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None
    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2. * np.pi * sigma ** 2.)) + y

def rms(data, fit):
    #sanity check
    if (data is None) or (fit is None) or (len(data) != len(fit)) or any(np.isnan(data)) or any(np.isnan(fit)):
        return None

    mx = max(data)

    if mx <= 0:
        return None

    d = np.array(data)/mx
    f = np.array(fit)/mx

    return np.sqrt(((f - d) ** 2).mean())

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

    filepath = FPLANE_LOC
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


class EmissionLine():
    def __init__(self,name,w_rest,plot_color,solution=True,z=0):
        self.name = name
        self.w_rest = w_rest
        self.w_obs = w_rest * (1.0 + z)
        self.z = z
        self.color = plot_color
        self.solution = solution #True = can consider this as the target line

    def redshift(self,z):
        self.z = z
        self.w_obs = self.w_rest * (1.0 + z)
        return self.w_obs





#mostly copied from Greg Z. make_visualization_dectect.py
class Dither():
    '''HETDEX dither file'''

    # needs open and parse (need to find the FITS files associated with it
    # need RA,DEC of fiber centers (get from pyhetdex)
    def __init__(self, dither_file):
        self.basename = []
        self.deformer = []
        self.dx = []
        self.dy = []
        self.seeing = []
        self.norm = []
        self.airmass = []
        self.dither_path = None

        self.dither_id_str = []
        self.dither_date = [] #string before T ... ie. in yyyymmdd
        self.dither_time = [] #string after T without decimal
        self.dither_time_extended = []
        self.dither_idx = []

        self.read_dither(dither_file)

    def get_dither_index(self,date,time,time_ex):
        if (date is not None) and (time is not None):
            for i in range(len(self.dither_id_str)):
                if (date == self.dither_date[i]) and (time == self.dither_time[i]):
                    #can assume that all dither_time_extended are populated or all are None
                    if (time_ex is not None) and (self.dither_time_extended[i] is not None):
                        if (date == self.dither_date[i]) and (time_ex == self.dither_time_extended[i]):
                            return self.dither_idx[i]
                        #else keep looking ... times match, but not extended time
                    else:
                        return self.dither_idx[i]

        return None



    def dither_id_str_split(self,dit_str,dit_idx=None):
        if dit_str is not None:
            #get rid of quotes, slashes, spaces
            self.dither_id_str.append(dit_str)
            dit_str = dit_str.replace("\\","")
            dit_str = dit_str.replace("\'","")
            #assume yyyymmddThhmmss.s format?

            if len(dit_str) < 18:
                log.error("Dither ID string not as expected: %s" % self.dither_id_str)
                return
            else:
                self.dither_idx.append(dit_idx)
                self.dither_date.append(dit_str[0:8])
                #next should be 'T'
                self.dither_time.append(dit_str[9:15]) #not the .# not always there
                if dit_str[15] == ".":
                    self.dither_time_extended.append(dit_str[9:17])
                else:
                    self.dither_time_extended.append(None)


    def read_dither(self, dither_file):
        try:
            self.dither_path = op.dirname(dither_file)

            if dither_file[-4:] == ".mch":
                with open(dither_file, 'r') as f:
                    f = ft.skip_comments(f)
                    dit_idx = -1
                    for l in f:
                        dit_idx += 1
                        try:
                            #there are inconsitencies in all.mch so can't just get fixed position values
                            elim = l.split()
                        except ValueError:  # skip empty or incomplete lines
                            pass

                        #used later to match up with a dither number
                        self.dither_id_str_split(elim[0],dit_idx)
                        #get the first two floats
                        val1 = None
                        val2 = None
                        for i in range(len(elim)):
                            val1 = None
                            val2 = None
                            try:
                                val1 = float(elim[i])
                                val2 = float(elim[i+1])
                            except:
                                continue

                            if (val1 is not None) and (val2 is not None):
                                self.dx.append(float(val1))
                                self.dy.append(float(val2))
                                break

            else: #Cure style
                with open(dither_file, 'r') as f:
                    f = ft.skip_comments(f)
                    for l in f:
                        try:
                            _bn, _d, _x, _y, _seeing, _norm, _airmass = l.split()
                        except ValueError:  # skip empty or incomplete lines
                            pass
                        self.basename.append(_bn)
                        self.deformer.append(_d)
                        self.dx.append(float(_x))
                        self.dy.append(float(_y))
                        self.seeing.append(float(_seeing))
                        self.norm.append(float(_norm))
                        self.airmass.append(float(_airmass))


                        #todo: similar to panacea style split the dither id string??
        except:
            log.error("Unable to read dither file: %s :" %dither_file, exc_info=True)




class DetObj:
    '''mostly a container for an emission line or continuum detection from detect_line.dat or detect_cont.dat file'''

    def __init__(self,tokens,emission=True,line_number=None,fcs_base=None,fcsdir=None,basic_only=False):
        #fcs_base is a basename of a single fcs directory, fcsdir is the entire FQdirname
        #fcsdir is more specific
        #skip NR (0)
        self.elixer_version = G.__version__
        self.elixer_datetime = time.strftime("%Y-%m-%d %H:%M:%S")

        self.phot_z_votes = []
        self.cluster_parent = 0 #detectid of anohter HETDEX source that is the cluster (specifically, redshift) for this object
        self.cluster_z = -1
        self.cluster_qz = -1
        self.cluster_list = None
        self.flags = 0 #bit mapped flags (32bit) to warn consumer about problems with the detection or its classification
                       #flags are defined in global_config.py
        self.needs_review = np.int8(0) #set to 1 if a manual review is needed
                                       #stored as Int8Col so can have -128 to 127; anticipated values:
                                       # 0 = none needed, 1 = review needed, 2= review done
                                       #with possible expansion to severity? or other
        self.full_flag_check_performed = False

        self.matched_cats = [] #list of catalogs in which this object appears (managed outside this class, in elixer.py)
        self.status = 0
        self.annulus = None
        self.target_wavelength = None
        self.syn_obs = None #SyntheticObservation
        #self.plot_dqs_fit = False
        #self.dqs = None #scaled score
        #self.dqs_raw = None #Detection Quality Score (raw score)
        self.type = 'unk'
        self.entry_id = None #e.g. line number or identifier (maybe not a number) from rsp output
        self.id = None

        #defunct for a while and removed in 1.5.07
        #still in place for CURE calls
        self.x = None  #sky x and y?
        self.y = None
        self.ifu_x = None
        self.ifu_y = None
        self.known_z = None #a fixed, "known" redshift passed in on the command line
        self.galaxy_mask = None #GalaxyMask object to be populated by the owner of this DetObj (usually a hetdex obj)
        self.galaxy_mask_z = None #array of redshifts for galaxies with which the detection might be associated
        self.galaxy_mask_d25 = None #array (corresponding to galaxy_mask_z) of minimum integer D25 defined ellipse scale
                                    #that still holds the coordinates
        self.galaxy_mask_flag = False #set to True if found in galaxy mask

        self.w = 0.0
        self.w_unc = 0.0
        self.la_z = 0.0
        self.dataflux = 0.0
        self.modflux = 0.0
        self.fluxfrac = 1.0
        self.estflux = -1.0#-1
        self.estflux_unc = 0.0
        self.estflux_h5 = -1.0 #from the h5 file (Karl's estimates)
        self.estflux_h5_unc = 0.0

        self.sigma = 0.0 #also doubling as sn (see @property sn farther below)
        self.sigma_unc = 0.0
        self.snr = None
        self.snr_unc = 0.0
        self.chi2 = 0.0
        self.chi2_unc = 0.0
        self.chi2s = 0.0
        self.chi2w = 0.0
        self.gammq = 0.0
        self.gammq_s = 0.0
        #self.eqw = 0.0
        self.eqw_obs = 0.0
        self.eqw_obs_unc = 0.0

        self.eqw_line_obs = None #uses HETDEX continuum estimate around the emission line
        self.eqw_line_obs_unc = None
        self.eqw_sdss_obs = None #uses HETDEX full width spectrum passed through sdss g-band filter
        self.eqw_sdss_obs_unc = None

        self.cont = -9999
        self.cont_cgs = -9999
        self.cont_cgs_unc = 0.0

        self.cont_cgs_narrow = -9999 #kept for reporting, not used elsewhere
        self.cont_cgs_narrow_unc = 0.0 #kept for reporting, not used elsewhere
        #in most cases, hetdex_xxx will be updated ... this is for explicit reporting in HDF5 file
        #and is not used anywhere else
        self.hetdex_cont_cgs = self.cont_cgs #as read from the detections file (right around the emission line)
        self.hetdex_cont_cgs_unc = self.cont_cgs_unc

        self.fwhm = -1.0 #in angstroms
        self.fwhm_unc = 0.0
        self.panacea = False

        self.ifuslot = None
        self.wra = None
        self.wdec = None

        self.num_hits = 0

        self.fibers = []
        self.fibers_sorted = False

        # list of fibers that are adjacent to the detection fibers (in self.fibers)
        # over all included amps and exposures BUT are not included in fibers itself
        # (it has no overlap with self.fibers)
        self.ccd_adjacent_fibers = []
        self.ccd_adjacent_single_fiber_brightest_mag = None
        self.central_single_fiber_mag = None

        self.grossly_negative_spec = False

        self.outdir = None
        self.calfib_noise_estimate = None
        self.num_duplicate_central_pixels = 0 #used in classification, if the number is high, more likely to be spurious

        #flux calibrated data (from Karl's detect and calibration)
        self.fcsdir = None
        self.pdf_name = None
        self.hdf5_detectname = None #detectname column in HDF5 representation (just for reference)
        self.hdf5_shot_dir = None #used only in the forced extraction, otherwise is specified as part of data release
        #the hdf5_detectid is the same as the self.entry_id, (see propert hdf5_detectid())

        self.line_gaussfit_parms = None #in load_fluxcalibrated_spectra becomes a 4 tuple (mu, sigma, Amplitude, y, dx)
                                        #where dx is the bin width for the amplitude (used if input data is
                                        #flux instead of flux density or flux/dx)
        self.line_gaussfit_unc = None

        self.sumspec_wavelength = []
        self.sumspec_counts = []
        self.sumspec_flux = []
        self.sumspec_flux_unit_scale = G.HETDEX_FLUX_BASE_CGS #cgs
        self.sumspec_fluxerr = []

        self.sumspec_apcor = []

        self.sumspec_wavelength_zoom = []
        self.sumspec_counts_zoom = []
        self.sumspec_flux_zoom = []
        self.sumspec_fluxerr_zoom = []
        self.sumspec_2d_zoom = []

        self.rvb = None #spectrum_utilities pseudo color dictionary (see red_vs_blue(...))
        self.spec_obj = elixer_spectrum.Spectrum() #use for classification, etc

        self.p_lae = None #from Andrew Leung
        self.p_oii = None
        self.p_lae_oii_ratio = None
        self.p_lae_oii_ratio_range = None

        self.bad_amp_dict = None

        #computed directly from HETDEX spectrum (3600AA-5400AA)
        self.hetdex_gmag = None
        self.hetdex_gmag_unc = None
        self.hetdex_gmag_cgs_cont = None
        self.hetdex_gmag_cgs_cont_unc = None
        self.eqw_hetdex_gmag_obs = None
        self.eqw_hetdex_gmag_obs_unc = None
        self.hetdex_gmag_p_lae_oii_ratio = None
        self.hetdex_gmag_p_lae_oii_ratio_range = None

        self.sdss_gmag = None #using speclite to estimate
        self.sdss_gmag_unc = None  # not computed anywhere yet
        self.sdss_cgs_cont = None
        self.sdss_cgs_cont_unc = None
        self.sdss_gmag_p_lae_oii_ratio = None
        self.sdss_gmag_p_lae_oii_ratio_range = None

        #chosen from one of the above
        self.best_gmag = None
        self.best_gmag_unc = None
        self.best_gmag_cgs_cont = None
        self.best_gmag_cgs_cont_unc = None
        self.best_eqw_gmag_obs = None
        self.best_eqw_gmag_obs_unc = None
        self.best_gmag_p_lae_oii_ratio = None
        self.best_gmag_p_lae_oii_ratio_range = None
        self.using_best_gmag_ew = False
        self.best_gmag_selected = "" #to be filled in with sdss or hetdex later

        self.duplicate_fibers_removed = 0 # -1 is detected, but not removed, 0 = none found, 1 = detected and removed
        self.duplicate_fiber_cutout_pair_weight =0 #any two of top 3 fiber 2D cutouts are identical

        self.image_2d_fibers_1st_col = None
        self.image_1d_emission_fit = None
        self.image_cutout_fiber_pos = None

        #survey info
        self.survey_shotid = None
        self.survey_fwhm_gaussian = None #HDR1
        self.survey_fwhm_moffat = None #HDR1
        self.survey_fwhm = None #HDR2+
        self.survey_response = None
        self.dither_norm = 0.0 #todo: max/min for dithers?
        self.amp_stats = 0.0 #todo:???
        self.survey_fieldname = None

        self.multiline_z_minimum_flag = False #False == multiline no good solution, True = 1 good solution

        self.aperture_details_list = []
        self.neighbors_sep = None #one (or none) entry from aperture_details_list for a single catalog + filter for SEP objects
                                  #the list is neighbors_sep.sep_objects
        self.bid_target_list = []

        self.classification_dict = {'scaled_plae':None,
                                    'plae_hat':None,
                                    'plae_hat_hi':None, #+ confidence interval (usually .68)
                                    'plae_hat_lo':None, #- confidence interval (usually .68)
                                    'plae_hat_sd':None,
                                    'size_in_psf':None, #to be filled in with info to help make a classification judgement
                                    'diam_in_arcsec':None, #to be filled in with info to help make a classification judgement
                                    'spurious_reason': None}

        self.extraction_aperture=None
        self.extraction_ffsky=False

        self.best_z = None
        self.best_p_of_z = None

        #colors in AB mag (from the photometric imaging and HETDEX-g, if no g imaging available)
        #ideally should be from the same survey (but will use different surveys to populate if a single survey does not
        #cover the requisite bands)
        self.best_img_u_mag  = None #"best" value for u as determined in cat_base::build_cat_summary_pdf_section()
        self.best_img_u_ra = None #ra, dec used to decide for colors that the objects are the same
        self.best_img_u_dec = None
        self.best_img_u_cat = None

        self.best_img_g_mag  = None #value, bright error, faint error
        self.best_img_g_ra = None
        self.best_img_g_dec = None
        self.best_img_g_cat = None

        self.best_img_v_mag  = None #value, bright error, faint error
        self.best_img_v_ra = None
        self.best_img_v_dec = None
        self.best_img_v_cat = None

        self.best_img_r_mag  = None #value, bright error, faint error
        self.best_img_r_ra = None
        self.best_img_r_dec = None
        self.best_img_r_cat = None

        self.color_gr = [None,None,None] #g-r color as color, blue max, red_max:
                                         #blue = -99 (means lower limit on blue), red = 99  means lower limit on red
        self.color_ug = [None,None,None] #might also serve as a drop out indicator; would be super red if drop out
                                         #between u and g bands
        self.color_ur = [None,None,None]

        self.best_counterpart = None #selected in cat_base::build_cat_summary_pdf_section

        if emission:
            self.type = 'emis'
            # actual line number from the input file
            # there are 3 numbers: the detect ID (from the detect file), the entity ID (from the composite file)
            # (those two are usually the same, unless CURE is used), and the line number from the input file
            # (e.g. the t7all or t5cut, etc)
            self.line_number = line_number
            if (tokens is not None) and (len(tokens) > 0):
                self.entry_id = int(tokens[0])
                self.id = int(tokens[1]) #detect id (not line number)

                #if (line_number is not None) and (self.entry_id == self.id):
                #    #could be happenstance or could be an old file
                    #if it is just happenstance, the line_number should also be the same
                #    self.entry_id = line_number

                self.x = float(tokens[2]) #sky x
                self.y = float(tokens[3]) #sky y
                self.w = float(tokens[4]) #wavelength
                self.la_z = float(tokens[5])
                self.dataflux = float(tokens[6])
                self.modflux = float(tokens[7])
                self.fluxfrac = float(tokens[8])
                #self.estflux = self.dataflux * G.FLUX_CONVERSION/self.fluxfrac #estimated flux in cgs f_lambda
                self.estflux = self.dataflux * flux_conversion(self.w) / self.fluxfrac  # estimated flux in cgs f_lambda
                #for safety
                if self.fluxfrac == 0:
                    self.fluxfrac = 1.0

                self.sigma = float(tokens[9])
                if tokens[10] == '1':
                    self.chi2 = 666
                else:
                    self.chi2 = float(tokens[10])
                self.chi2s = float(tokens[11])
                self.chi2w = float(tokens[12])
                self.gammq = float(tokens[13])
                self.gammq_s = float(tokens[14])
                self.eqw_obs = float(tokens[15])
                self.cont = float(tokens[16]) #replaced by idx ~ 25 (1st value after the last fiber listed)

                try:
                    if len(tokens) > 17: #this is probably an all ifu panacea version
                        self.panacea = True
                        self.ifuslot = str(tokens[17][-3:]) #ifu093 -> 093
                        if len(tokens) > 18:  # has the rest
                            try:
                                self.wra = float(tokens[18])
                                self.wdec = float(tokens[19])
                            except:
                                self.wra = None
                                self.wdec = None
                                if ('***' in tokens[18]) and ('***' in tokens[19]):
                                    pass
                                else:
                                    log.error("Exception parsing tokens.",exc_info=True)

                            start = 20
                            num_of_fibers = 0
                            for i in range(start,len(tokens)): #there are fibers and other stuff to follow
                                if not self.parse_fiber(tokens[i]): #this was not a fiber descriptor
                                    break
                                else:
                                    num_of_fibers += 1 #will need to know for SN reads and fiber RA,DEC positions

                            try:
                                self.cont = float(tokens[i])
                            except:
                                self.cont = None
                                if '***' in tokens[i]:
                                    pass
                                else:
                                    log.error("Exception parsing tokens.", exc_info=True)

                            start = i+1
                            for i in range(start,min(len(tokens),start+num_of_fibers)): #these are in the same order as fibers
                                try:
                                    sn = float(tokens[i])
                                except:
                                    sn = -999
                                    if '***' in tokens[i]:
                                        pass
                                    else:
                                        log.error("Exception parsing tokens.", exc_info=True)
                                for f in self.fibers:
                                    if f.sn is None:
                                        f.sn = sn
                                        break

                            start = i+1
                            fib_idx = 0
                            if (len(tokens) - start) >= (2*num_of_fibers): #this probably has the RA and Decs
                                for i in range(start,min(len(tokens),start+2*num_of_fibers),2):
                                    #could have "666" fibers, which are not added to the list of fibers
                                    #so check before attempting to add ... if the fiber does not exist in the list
                                    #we expect the value of ra and dec to also be 666, as a sanity check
                                    #but we still need to iterate over these values to parse the file correctly
                                    if fib_idx < len(self.fibers):
                                        try:
                                            self.fibers[fib_idx].ra = float(tokens[i])
                                            self.fibers[fib_idx].dec = float(tokens[i+1])
                                        except:
                                            self.fibers[fib_idx].ra = None
                                            self.fibers[fib_idx].dec = None
                                            log.error("Exception parsing tokens.", exc_info=True)
                                        fib_idx += 1
                                    else: #we are out of fibers, must be junk ...
                                        #sanity check
                                        try:
                                            if (float(tokens[i]) != 666.) or (float(tokens[i+1]) != 666.):
                                                log.warning("Warning! line file parsing may be off. Expecting 666 for "
                                                            "ra and dec but got: %s , %s " %(tokens[i],tokens[i+1]))
                                        except:
                                            pass

                except:
                    log.info("Error parsing tokens from emission line file.",exc_info=True)

                # karl is not using the line number ... is using the entry_id
                #this is a combination of having a line file AND an rsp1 direcotry
                if fcsdir is not None:
                    self.fcsdir = fcsdir
                elif (fcs_base is not None) and (line_number is not None):
                    self.fcsdir = fcs_base + str(self.entry_id).zfill(3)

                ### this is the total flux under the line in ergs s^-1 cm^-2   (not per Hz or per Angstrom)


                # ? units of dataflux? (counts per AA or per 1.9xAA?) need to convert to equivalent units with cont
                #   counts are okay, I think, if cont is in counts / AA  (or per 2AA?), else convert to cgs
                # ? assuming dataflux is NOT per AA (that is, the wavelength has been multiplied out ...
                #                                  ... is this the total flux under the line?)
                # ? cont is sometimes less than zero? that makes no sense?
                # ? does dataflux already have the fluxfrac adjustment in it? Right now not getting a fluxfrac so set to 1.0
                #
                # ** note: for the bid targets in the catalog, the line flux is this dataflux/fluxfrac converted to cgs
                #         (again, assuming it is the total flux, not per AA)
                #         and the continuum flux is the f606w converted from janskys to cgs
                #   ?? is there a better estimate for the continuum for the bid targets?

                # if self.cont <= 0, set to floor value (need to know virus limit ... does it vary with detector?)
                if (self.cont <= 0.0) or (self.cont == 666):
                    self.cont = G.CONTINUUM_FLOOR_COUNTS  # floor (
                # use the conversion factor around the line
                self.cont_cgs = self.cont * flux_conversion(self.w)

                if (self.eqw_obs == -300) and (self.dataflux != 0) and (self.fluxfrac != 0):
                    # this is the approximation vs EW = integration of (F_cont - F_line) / F_line dLambda
                    # these are all in counts (but equivalent of wFw)
                    self.eqw_obs = abs(self.dataflux / self.fluxfrac / self.cont)

            #end if tokens > 0
            elif fcsdir is not None: #expectation is that we have an rsp1 style directory instead
                self.fcsdir = fcsdir
                try: #for consistency with Karl's naming, the entry_id here is the _xxx number
                    #try as the last number
                    # toks = os.path.basename(fcsdir).split("_")
                    # possible_ids = []
                    # for s in toks:
                    #     try:
                    #         id = int(s)
                    #         possible_ids.append(id)
                    #     except:
                    #         pass
                    #
                    # if len(possible_ids) > 0:
                    #     self.entry_id = possible_ids[-1] #get the last one
                    # else:
                    #     log.debug("No detection ID from basename: %s" %fcsdir)

                    self.entry_id = os.path.basename(fcsdir).split("_",1)[1] #split on 1st '_' ... everything after is ID
                    if self.entry_id.isdigit(): #digits only
                        self.entry_id = int(self.entry_id)

                    #self.entry_id = int(os.path.basename(fcsdir).split("_")[-1]) #assumes 20170322v011_005
                except:
                    log.debug("No detection ID from basename")

        else:
            self.type = 'cont'
            if (tokens is not None) and (len(tokens) > 0):
                #basically, from an input line file
                self.id = int(tokens[0])
                self.x = float(tokens[1])
                self.y = float(tokens[2])
                self.sigma = float(tokens[3])
                self.fwhm = float(tokens[4])
                self.a = float(tokens[5])
                self.b = float(tokens[6])
                self.pa = float(tokens[7])
                self.ir1 = float(tokens[8])
                self.ka = float(tokens[9])
                self.kb = float(tokens[10])
                self.xmin = float(tokens[11])
                self.xmax = float(tokens[12])
                self.ymin = float(tokens[13])
                self.ymax = float(tokens[14])
                self.zmin = float(tokens[15])
                self.zmax = float(tokens[16])

        self.ra = None  # calculated value
        self.dec = None  # calculated value
        self.nearest_fiber = None
        self.fiber_locs = None #built later, tuples of Ra,Dec of fiber centers

        if fcsdir is not None:
            filename = os.path.basename(fcsdir)
            # expect the id to be in the filename
            if str(self.entry_id) in filename:
                self.pdf_name = filename  #w/o the extension

        # dont do this here ... only call after we know we are going to keep this DetObj
        # as this next step takes a while
        #self.load_fluxcalibrated_spectra()

    def get_phot_z_vote(self):
        try:
            if len(self.phot_z_votes) > 0:
                z = np.mean(self.phot_z_votes) #all weighted equally
                #todo: later see if can get the acutal PDFs or uncertainties to weigh the average
                if -0.1 < z < 0.7: #low Z ... favors OII
                    if abs((self.w/G.OII_rest -1.0)-z) < 0.5:
                        #consistent
                        return self.w/G.OII_rest -1.0
                elif 1.7 < z < 3.7: #high z .... favors LAE
                    if abs((self.w/G.LyA_rest -1.0)-z) < 0.5:
                        #consistent
                        return self.w/G.LyA_rest -1.0
                else: #no vote; don't trust in mid-z range, or z > 4
                    return -1
        except:
            return -1

        return -1

    def unique_sep_neighbors(self):
        """
            #need to refine the aperture details list
            #there can be multiple filters and multiple catalog/instruments
            #we only want ONE instance of each neighbor

            So. We iterate over the aperture_details_list.
            First find the deepest imaging with g and then r band. Deep wins over band then g over r.

        :return: list of unique sep_objects (to be used to add spectra and save to h5 file)
        """

        #list of dicts of RA, Dec, catalog, filter, size ,etc
        # keep preferentially, deeper imaging, g then r band (g because of HETDEX in g as most representative of size in similar band)

        best_idx = -1
        best_limit = 0
        best_filter = 'x'

        try:
            for i,cat in enumerate(self.aperture_details_list):
                if cat['filter_name'].lower() in ['g','r','f606w'] and cat['mag_limit'] < 99 and cat['mag_limit'] >= best_limit:
                    if best_filter == 'g':
                        pass # g prefered over r
                    else:
                        best_idx = i
                        best_limit = cat['mag_limit']
                        best_filter = cat['filter_name']

            #now we should have the best catalog to check
            if best_idx < 0: #none were found
                return []
            else:
                #we are going to add to this dictionary and it will no longer be the same as the others
                #so make a copy to work on
                self.neighbors_sep = deepcopy(self.aperture_details_list[best_idx])
                #this is a single object with the list as a property: .sep_objects

                self.neighbors_sep
                return self.neighbors_sep

            # #these should already be unique, but there can be overlapping ellipses
            # #for our purposes, point sources with overlapping ellipse must be very close on sky and single RA, Dec extraction
            # #would be fine but the deblending code should handle this okay anyway, so we will just treat them as
            # #unique/discrete sources
            # for s in self.aperture_details_list[best_idx]['sep_objects']:
            #     #since these are in the same imaging, the x, y is okay to identify or could use ra, dec

        except:
            log.error("Exception! Exception in DetObj::unique_neighbor_coords().",exc_info=True)

        return []

    @property
    def hdf5_detectid(self):
        #for now .... may be smarter than this in the future
        #HDF5 (internal) detectid (INT64)
        return self.entry_id

    @property
    def sn(self):
        if self.snr is not None:
            return self.snr
        else:
            return self.sigma

    @property
    def my_ra(self):
        if self.wra is not None:
            return self.wra
        else:
            return self.ra

    @property
    def my_dec(self):
        if self.wdec is not None:
            return self.wdec
        else:
            return self.dec


    def set_best_filter_mag(self,band,mag,mag_bright,mag_faint,ra=None,dec=None,catalog=None):
        """

        :param mag:
        :param mag_bright:
        :param mag_faint:
        :param ra: decimal degrees (used to check that when calculating colors, it is the same object)
        :param dec:
        :return:
        """

        mag_list = [mag,mag_bright,mag_faint]
        try:
            if band.lower() in ['u']:
                self.best_img_u_mag = mag_list
                self.best_img_u_ra = ra
                self.best_img_u_dec = dec
                self.best_img_u_cat = catalog
            elif band.lower() in ['g','f435w']:
                self.best_img_g_mag = mag_list
                self.best_img_g_ra = ra
                self.best_img_g_dec = dec
                self.best_img_g_cat = catalog
            elif band.lower() in ['v']:
                self.best_img_v_mag = mag_list
                self.best_img_v_ra = ra
                self.best_img_v_dec = dec
                self.best_img_v_cat = catalog
            elif band.lower() in ['r','f606w']:
                self.best_img_r_mag = mag_list
                self.best_img_r_ra = ra
                self.best_img_r_dec = dec
                self.best_img_r_cat = catalog
        except:
           log.warning(f"Exception in DetObj::get_filter_ccolors().",exc_info=True)




    def get_filter_colors(self):
        """
        Populate the filter colors from the available bands

        Has to be run AFTER imaging section is complete
        :return: None
        """

        def compute_color(blue,red):
            """
            each as 3-tuple [mag, mag_bright, mag_faint]

            :param blue:
            :param red:
            :return:
            """
            color = [None,None,None]
            try:
                if blue is None or red is None:
                    return color

                color[0] = blue[0] - red[0]

                #bright ends of mags, does not make sense that they would not be populated
                #most blue ... so max blue and min red
                if blue[1] is None or red[2] is None:
                    color[1] = None
                else:
                    color[1] = blue[1] - red[2]

                #most red ... so min blue and max red
                if blue[2] is None or red[1] is None:
                    color[2] = None
                else:
                    color[2] = blue[2] - red[1]

            except:
                log.warning("Exception in DetObj::get_filter_colors::compute_color()",exc_info=True)

            if color[1] < -10:
                color[1] = -99

            if color[2] > 10:
                color[2] = 99

            return color

        try:
            #if object barycenters are withing 0.1" will assume same object (a value of -1 is returned if the distance
            # could not be computed, i.e. if either coordinate is None)

            if  (self.best_img_u_cat == self.best_img_r_cat) and \
                (-1 < utils.angular_distance(self.best_img_u_ra,self.best_img_u_dec,self.best_img_r_ra,self.best_img_r_dec) < 0.1):
                self.color_ur = compute_color(self.best_img_u_mag,self.best_img_r_mag)

            if  (self.best_img_u_cat == self.best_img_g_cat) and \
                (-1 < utils.angular_distance(self.best_img_u_ra,self.best_img_u_dec,self.best_img_g_ra,self.best_img_g_dec) < 0.1):
                self.color_ug = compute_color(self.best_img_u_mag,self.best_img_g_mag)

            if  (self.best_img_g_cat == self.best_img_r_cat) and \
                (-1 < utils.angular_distance(self.best_img_g_ra,self.best_img_g_dec,self.best_img_r_ra,self.best_img_r_dec) < 0.1):
                self.color_gr = compute_color(self.best_img_g_mag,self.best_img_r_mag)

            #if self.best_img_g_mag is not None:
            #  >>> note: was the above few lines for self.color_xx
            # DO NOT USE anything other than same fixed sized apertures on same objects
            # else: #we will use the DEX-g (not ideal, but better than nothing
            #     if self.best_gmag_unc is None:
            #         self.best_gmag_unc = 0
            #
            #     if self.best_gmag < G.HETDEX_CONTINUUM_MAG_LIMIT:
            #         g_substitue = [self.best_gmag, self.best_gmag-self.best_gmag_unc, self.best_gmag+self.best_gmag_unc]
            #     else:
            #         g_substitue = [G.HETDEX_CONTINUUM_MAG_LIMIT,
            #                        min(G.HETDEX_CONTINUUM_MAG_LIMIT,self.best_gmag-self.best_gmag_unc),
            #                        99]
            #
            #     self.color_ug = compute_color(self.best_img_u_mag,g_substitue)
            #     self.color_gr = compute_color(g_substitue,self.best_img_r_mag)

            #for now at least, not bothering with 'v' band or other bands
        except:
            log.warning(f"Exception in DetObj::get_filter_ccolors().",exc_info=True)


    def flag_check(self):
        """
        walk through a bunch of conditions and set/update self.flags
        :return:
        """

        def mags_compatibile(band1,mag1,mag1_bright, mag1_faint, band2, mag2, mag2_bright, mag2_faint):
            """
            see if the mags overlap and are reasonable given the bands (can be r and/or g)
            :param mag1:
            :param mag1_bright:
            :param mag1_faint:
            :param mag2:
            :param mag2_bright:
            :param mag2_faint:
            :param band:
            :return:
            """

            pass


        def want_band(filter):
            if filter.lower() in ['g','r','f606w']:
                return True
            else:
                return False


        try:
            ##################################
            # check for no imaging
            # can skip many other checks if there is no imaging
            ##################################
            no_imaging = False
            if (self.aperture_details_list is None) or len(self.aperture_details_list)==0:
                #either there is no imaging or the imaging is so bad that we could not even
                #get any apertures, so it is just as good as having no imaging

                self.flags |= G.DETFLAG_NO_IMAGING
                no_imaging = True
                log.info(f"Detection Flag set for {self.entry_id} : DETFLAG_NO_IMAGING")



            #get the catalog match; useful in various places later
            #cat_sel == catalog bid target selected
            cat_name = None
            cat_filter = None
            cat_sel = None

            try:
                cat_sel = np.where([d.selected for d in self.bid_target_list])[0]
                if len(cat_sel) == 1:
                    cat_sel = cat_sel[0]
                    cat_name = self.bid_target_list[cat_sel].catalog_name
                    cat_filter = self.bid_target_list[cat_sel].bid_filter.lower()
            except:
                pass


            if not no_imaging and self.survey_fwhm is not None:

                shot_psf = SU.get_psf(self.survey_fwhm,3.0,0.0) #assume a 3" radius aperture centered at 0.0
                #this is used later

                ##################################
                #check DEX g-mag
                #DETFLAG_DEX_GMAG_INCONSISTENT
                ##################################

                #out of whack if imaging is bright and DEX-g is faint (or at limit)
                # or DEX-g is better than limit but imaging is faint or not found
                # (Keep in mind that the DEX-g is from, usually, a 3" aperture and the imaging aperture is often variable
                # and may be elliptical, so there are certainly obvious sources of discrepencies)
                if self.best_gmag is not None:

                    dex_g_depth = G.HETDEX_CONTINUUM_MAG_LIMIT #24.5
                    bright_skip = 22.0 #if both are brighter than this, skip the logic ... it cannot make any difference
                                       #and we can miss more flux at brighter mags since the aperture does not grow enough
                                       #and there are no real aperture corrections applied to the imaging aperture

                    if self.best_gmag < dex_g_depth:
                        dex_g_limit = False
                    else:
                        dex_g_limit = True

                    #check to see if consistent with other g-mags and maybe r-mag
                    #we will be fairly generous and allow 0.5 mag variation?
                    new_flag = 0

                    self.aperture_details_list = np.array(self.aperture_details_list)
                    #get deepest g and deepest r
                    sel_g = np.array([d['filter_name'].lower() in ['g'] for d in self.aperture_details_list])
                    if np.sum(sel_g) > 0:
                        deep_g = np.max([d['mag_limit'] for d in self.aperture_details_list[sel_g]])
                    else:
                        deep_g = -1

                    sel_r = np.array([d['filter_name'].lower() in ['r','f606w'] for d in self.aperture_details_list])
                    if np.sum(sel_r) > 0:
                        deep_r = np.max([d['mag_limit'] for d in self.aperture_details_list[sel_r]])
                    else:
                        deep_r = -1

                    #sel_depths = [d['mag_limit'] for d in self.aperture_details_list]

                    #array of False, True where in sel_g AND d['mag_limit'] == deep_r
                    sel_g = np.array([d['filter_name'].lower() in ['g'] and d['mag_limit'] == deep_g for d in self.aperture_details_list])
                    sel_r = np.array([d['filter_name'].lower() in ['r','f606w'] and d['mag_limit'] == deep_r for d in self.aperture_details_list])

                    for d in self.aperture_details_list[sel_g | sel_r]:
                        if d['mag'] is not None:
                            if (d['mag'] < bright_skip) and (self.best_gmag < bright_skip):
                                continue

                            img_limit = False
                            if d['mag_limit'] is not None and (20 < d['mag_limit'] < 35):
                                if (d['mag'] >= d['mag_limit']) or ('fail_mag_limit' in d.keys() and d['fail_mag_limit']):
                                    img_limit = True

                            if d['filter_name'].lower() in ['g']: #allow 0.5 variation?
                                allowed_diff = 0.8
                            elif d['filter_name'].lower() in ['r','f606w']: #different filter so, maybe allow 1.5?
                                allowed_diff = 1.3 #noting
                            else:
                                continue #only checking g or r


                            #and modify by distance to barycenter (think of case where we are on the edge ... the
                            #DEX PSF weighted aperture won't weight the flux the same as being centered

                            try:
                                if d['sep_obj_idx'] is not None:
                                    dist = d['sep_objects'][d['sep_obj_idx']]['dist_baryctr']
                                    radius = 0.5 * np.sqrt( d['sep_objects'][d['sep_obj_idx']]['a'] *
                                                            d['sep_objects'][d['sep_obj_idx']]['b'] )

                                    frac,ujy,flux = SU.check_overlapping_psf(self.best_gmag,d['mag'],shot_psf,dist,effective_radius=radius)
                                    adjusted_mag  = SU.ujy2mag(ujy) #the mag we would expect to measure if the object
                                                                    #were a pointsource at the distance from the DEX center

                                    #DEX radius is usally 3", and PSF weighted and to be very accurate we should
                                    #use the PSF for this shot and figure how much of the object (based in its barycenter
                                    #and radius) is captured in the DEX fiber set (with fiber weights)
                                    #this is not perfect, and assumes the object is a point source, but is a bit
                                    #of a correction for this flag
                                else:
                                    adjusted_mag = d['mag']
                            except:
                                adjusted_mag = d['mag']

                            #limit the flagging be allowing the adjusted mag to be which ever is closer to the g-mag
                            if abs(adjusted_mag-self.best_gmag) > abs(d['mag']-self.best_gmag):
                                adjusted_mag = d['mag']

                            if dex_g_limit and img_limit:
                                pass #we're done, both are at their limit
                            elif dex_g_limit and (adjusted_mag > dex_g_depth): #self.best_gmag):
                                pass #also okay ... hit DEX limit and the imaging mag is fainter than the dex limit
                            elif img_limit and (self.best_gmag > adjusted_mag):
                                pass #also okay ... hit the imaging limit (probably SDSS or maybe PanSTARRS) and DEX is fainter
                            elif dex_g_limit and ( (adjusted_mag+allowed_diff) < dex_g_depth):
                                #bad ... hit the Dex limit but imaging mag is much brighter
                                new_flag = G.DETFLAG_DEX_GMAG_INCONSISTENT
                            elif img_limit and ((self.best_gmag+allowed_diff) < adjusted_mag):
                                #bad ... hit the imging limit but DEX mag is much brighter
                                new_flag = G.DETFLAG_DEX_GMAG_INCONSISTENT
                            elif abs(adjusted_mag-self.best_gmag) > allowed_diff:
                                #neither at limit, but they disagree
                                new_flag = G.DETFLAG_DEX_GMAG_INCONSISTENT
                            # elif d['filter_name'].lower() in ['g']:
                            #     #we're done ... we matched g to g ... no need to check 'r'
                            #     break

                            if new_flag:
                                self.flags |= new_flag
                                log.info(f"Detection Flag set for {self.entry_id} : DETFLAG_DEX_GMAG_INCONSISTENT")
                                #break #already been flagged ... no need to flag the same thing again
                                #don't break ... can be undone if 'g' matches 'g'
                            elif d['filter_name'].lower() in ['g']:
                                if self.flags & G.DETFLAG_DEX_GMAG_INCONSISTENT:
                                    self.flags |= G.DETFLAG_DEX_GMAG_INCONSISTENT
                                    log.info(f"Detection Flag un-set for {self.entry_id}: g bands match")
                                break #now we DO want to break


                ######################################################
                # check large Line Flux, but nothing in imaging
                ######################################################
                try:
                    if (self.estflux > 1.5e-16) or (self.cont_cgs > 1.0e-18) or (self.best_gmag < 24.0):
                        #from the HETDEX data, we expect to see something in the imaging
                        new_flag = G.DETFLAG_COUNTERPART_NOT_FOUND
                        expect_to_find_any = False
                        for d in self.aperture_details_list:
                            if d['filter_name'].lower() in ['g','r','f606w']:
                                if d['mag_limit'] is not None and (24 < d['mag_limit']):
                                    if self.best_gmag is not None and self.best_gmag < d['mag_limit']:
                                        expect_to_find_any = True
                                    #imaging qualifies, there should be something
                                    if d['sep_objects'] is not None:
                                        #check distances (dist_curve < 1.0 or -1 (we are inside the curve))
                                        if np.any(np.array([x['dist_curve'] for x in d['sep_objects']]) < 1.0):
                                            new_flag = 0 #we did find something, so break
                                            break
                                        #todo: make better with a counterpart match (including catalog)
                                        #todo: so we would not count this if there is something faint, but several arcsecs away

                        if new_flag and expect_to_find_any: #non-match still possible ... check the catalogs
                            try:
                                for d in self.bid_target_list[1:]: #skip #0 as that is the Elixer entry
                                    if d.bid_filter.lower() in ['g','r','f606w']:
                                        if d.distance < 1.5:
                                            new_flag = 0 #we did find something, so break
                                            break
                            except:
                                log.debug("Exception in flag_check()",exec_info=True)

                        if new_flag and expect_to_find_any:
                            self.flags |= new_flag
                            log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_COUNTERPART_NOT_FOUND")

                except:
                    log.warning(f"Exception in flag_check. DETFLAG_COUNTERPART_NOT_FOUND. estflux = {self.extflux}"
                                f", cont_cgs = {self.cont_cgs}, best_gmag = {self.best_gmag}")


                ######################################################
                # check for large neighbor in SEP objects
                # G.DETFLAG_LARGE_NEIGHBOR
                ######################################################
                new_flag = 0
                for d in self.aperture_details_list:
                    if not want_band(d['filter_name']) or (d['sep_objects'] is None):
                        continue

                    for s in d['sep_objects']:
                        if (s['selected'] is False) and (s['mag'] < 23) and (s['mag'] < self.best_gmag) and \
                            (s['a'] > 4.0) and (s['dist_curve'] < s['a']):
                            log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_LARGE_NEIGHBOR")
                            self.flags |= G.DETFLAG_LARGE_NEIGHBOR
                            #only need one for this to trip
                            break

                ######################################################
                # check for no SEP ellipse wthin 0.5"
                # NOTE: specfically this only applies to SEP ellipses ... if there are NO ellipses this flag is skipped
                # and is covered by the NO_COUNTERPART flag
                ######################################################
                new_flag = 0
                for d in self.aperture_details_list:
                    if not want_band(d['filter_name']):
                        continue
                    if d['sep_obj_idx'] is None:
                        continue

                    #have to have at least one set of SEP objects, otherwise this flag makes no sense
                    new_flag = G.DETFLAG_DISTANT_COUNTERPART
                    for s in d['sep_objects']:
                        if s['dist_curve'] < 0.5:
                            # done, we are inside at least one ellipse (negative distance) OR within 0.5"
                            new_flag = 0
                            break

                    if new_flag == 0:
                        break

                if new_flag:
                    self.flags |= new_flag
                    log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_DISTANT_COUNTERPART")


                ######################################################
                # check for multiple SEP ellipse wthin 1.5"
                # NOTE: specfically this only applies to SEP ellipses ... if there are NO ellipses this flag is skipped
                ######################################################
                new_flag = 0

                for d in self.aperture_details_list:
                    if not want_band(d['filter_name']):
                        continue
                    if d['sep_obj_idx'] is None:
                        continue

                    count = 0
                    for s in d['sep_objects']:
                        if s['dist_curve'] < 1.5:
                            # done, we are inside at least one ellipse (negative distance) OR within 0.5"
                            count += 1

                    if count > 1:
                        new_flag = G.DETFLAG_BLENDED_SPECTRA
                        break

                if new_flag:
                    self.flags |= new_flag
                    log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_BLENDED_SPECTRA")

                ###############################################################
                #Check the selected catalog counterpart for mag compatibility
                # if there is a catalog counterpart, check the aperture mags
                #   for each filter matching the catalog filter
                ###############################################################

                if cat_sel is not None:
                    #we don't always use the catalog and the same imaging
                    ap_sel = None
                    if cat_sel:
                        try:
                            filter_list = []
                            band = None
                            if cat_filter in ['g']:
                                filter_list = ['g']
                                band = 'g'
                            elif cat_filter in ['r','f606w']:
                                filter_list = ['r','f606w']
                                band = 'r'

                            #choose the aperture details that match up with the catalog
                            #ap_sel = np.where([x['catalog_name'] for x in self.aperture_details_list]==cat_name) * \
                            #         np.where([x['filter_name'] for x in self.aperture_details_list]==cat_filter)
                            ap_sel =  [x['filter_name'].lower() in filter_list for x in self.aperture_details_list] #may be several

                            for ad in np.array(self.aperture_details_list)[ap_sel]:
                                #check sep object
                                if (ad['sep_obj_idx'] is not None) and (ad['sep_obj_idx'] >= 0):
                                    ap_mag = ad['sep_objects'][ad['sep_obj_idx']]['mag']
                                    ap_mag_faint = ad['sep_objects'][ad['sep_obj_idx']]['mag_faint']
                                    ap_mag_bright = ad['sep_objects'][ad['sep_obj_idx']]['mag_bright']

                                elif (ad['elixer_aper_idx'] is not None) and (ad['elixer_aper_idx'] >= 0):
                                    ap_mag = ad['elixer_apertures'][ad['elixer_aper_idx']]['mag']
                                    ap_mag_faint = ad['elixer_apertures'][ad['elixer_aper_idx']]['mag_faint']
                                    ap_mag_bright = ad['elixer_apertures'][ad['elixer_aper_idx']]['mag_bright']
                                else:
                                    ap_mag = None
                                    ap_mag_faint = None
                                    ap_mag_bright = None

                                if ap_mag is not None:
                                    pass
                                    #if (ap_mag_bright) and ()

                        except:
                            log.info("Error in DetObj.flag_check()",exc_info=True)

                    #if self.aperture_details_list is not None:
                    #        pass




                ######################################################
                # check mag depth ... must be at least 25.0
                ######################################################
                new_flag = G.DETFLAG_POOR_IMAGING
                max_band_mag = 99.9
                for d in self.aperture_details_list:
                    if not want_band(d['filter_name']):
                        continue

                    if not d['fail_mag_limit']:
                        max_band_mag = min(d['mag'],max_band_mag)

                    if d['mag_limit'] >= 24.5:
                        # done, we are inside at least one ellipse (negative distance) OR within 0.5"
                        new_flag = 0
                        break

                if new_flag and (self.best_gmag > 23.0) and (max_band_mag > 23.0): #no need to set if the object is already very bright
                    self.flags |= new_flag
                    log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_POOR_IMAGING")

                #todo: DEX-g is 24.5 or brighter (maybe 24 or brighter) and imaging depth is 24.5 or fainter
                #todo: but no extracted objects are found? and/or the aperture mag is at limit?


                #todo: check top solution (if there is one) and if unaccounted for line score is high, add "blended flag"
                #todo: or if no top solution but there are two or more strong lines, add "blended flag"


                #todo: Not sure if this one is needed
                #todo: if the various DEX-g magnitudes (wide-fit, full-spec, SDSS-g spect) are very different, add
                #todo:   "inconsistent dex flag" ??? (or some other name since already used for mismatch to imaging


                #todo: if multiple "strong" solutions, even if one is picked, set a "confused multi-line solution flag"

                #todo: if  0.3 < P(LyA) < 0.7 AND Q(z) < 0.7 set "uncertain z flag" .... IS THIS NECESSARY? Q(z) already
                #todo: says it is uncertain.

            #end if not no_imaging

            ###########################################
            #check for unmatched lines
            #DETFLAG_BLENDED_SPECTRA
            ###########################################

            #even if this solution is not shown due to not being unique, it is still the top scoring
            #solution
            if self.spec_obj.solutions is not None and len(self.spec_obj.solutions)> 0:
                #want to be really clear condition here, so both have to be > MULTILINE_FULL_SOLUTION_SCORE
                #not just > MAX_OK_UNMATCHED_LINES_SCORE
                if (self.spec_obj.solutions[0].unmatched_lines_score > (G.MULTILINE_MIN_SOLUTION_SCORE-1)) and \
                   (self.spec_obj.solutions[0].score > G.MULTILINE_FULL_SOLUTION_SCORE):
                    self.flags |= G.DETFLAG_BLENDED_SPECTRA
                    log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_BLENDED_SPECTRA")





            ###########################################
            #check for unmatched lines
            #DETFLAG_UNCERTAIN_CLASSIFICATION
            ###########################################

            #even if this solution is not shown due to not being unique, it is still the top scoring
            #solution
            if self.spec_obj.solutions is not None and len(self.spec_obj.solutions)> 0:
                #want to be really clear condition here, so both have to be > MULTILINE_FULL_SOLUTION_SCORE
                #not just > MAX_OK_UNMATCHED_LINES_SCORE
                questionable_solutions = [1549.0,1909.0,2799.0,G.OII_rest,5007.0]
                if (0.5 < self.spec_obj.solutions[0].scale_score < 0.9) and \
                   (self.spec_obj.solutions[0].z != self.best_z) and (self.spec_obj.solutions[0].central_rest in questionable_solutions):

                    self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                    log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_UNCERTAIN_CLASSIFICATION")



            #########################################
            #set so we don't re-check
            ########################################
            self.full_flag_check_performed = True

            if self.flags != 0: #todo: may add more logic and only trigger on certain flags?
                self.needs_review = 1
                log.info(f"Detection Flag: Setting REVIEW flag for {self.entry_id}")
            else:
                log.info(f"Detection Flag: NO review requested for {self.entry_id}")
        except:
            log.error("Exception! Exception in DetObj.flag_check()",exc_info=True)

    def best_redshift(self):
        """
        Sort of a P(z), but primitive

        :return: z and P(z) for that z
        """

        try:
            #aka P(LyA)
            scaled_plae_classification = self.classification_dict['scaled_plae']
            p = abs(0.5 - scaled_plae_classification)/0.5 #so, peaks near 0 and 1 and is zero at 0.5 (this is a confidence in classification)
            plya_for_oii = 0.7 #with no other evidence other than P(LyA) that favors OII, since it can be other lines
                               #not just OII, rescale by this factor when assuming OII
            rest = 0
            multiline_sol_diag = 0 #some diagnostic info on a potential multiline soluition
            #is the multiline solution (which has been updated with catalog phot-z and spec-z)
            #consistent with lowz or high-z?

            #the problem is that the spec_obj solution might be weak or even if not, out voted
            #by the P(LyA) classifictionm but just using that with P(LyA) as the confidence
            #then could be incongruous (in that they don't belong together)

            if self.spec_obj and self.spec_obj.solutions and len(self.spec_obj.solutions) > 0 \
                    and (self.spec_obj.solutions[0].scale_score > 0.7):
                #todo: maybe also check that the [0] position score is >= G.MULTILINE_MIN_SOLUTION_SCORE ??
                #these are in rank order, highest score 1st
                agree = False #P(LyA) and multiline agree
                unsure = False #P(LyA) is near 0.500
                num_solutions = len(self.spec_obj.solutions)

                primary_solution = False

                if (num_solutions > 1) and (self.spec_obj.solutions[1].frac_score > 0):
                    if (self.spec_obj.solutions[0].frac_score / self.spec_obj.solutions[1].frac_score) > 1.9:
                        primary_solution = True
                else:
                    primary_solution = True

                #keep the first that agrees
                for idx,sol in enumerate(self.spec_obj.solutions):
                    z = sol.z
                    rest = sol.central_rest
                    score = sol.score
                    scale_score = sol.scale_score

                    #voting vs line solution mis-match
                    #so 0.4 to 0.6 is no-man's land, but the prob or confidence will be very low anyway
                    #keep the z, but reduce the p(z)

                    #there is some directionality too
                    #large P(LyA) points to Lya and a spectific z
                    #but near 0 P(LyA) just points to not LyA

                    if 0.4 < scaled_plae_classification < 0.6: #never going to get an answer from P(LyA)
                        agree = True
                        unsure = True
                        break
                    elif ((scaled_plae_classification < 0.4) and (rest == G.LyA_rest)) or \
                         ((scaled_plae_classification > 0.6) and (rest != G.LyA_rest)):
                        #voting vs line solution mis-match

                        #if the next solution is much weaker just stop, otherwise keep going
                        if (idx+1) < num_solutions:
                            if  (self.spec_obj.solutions[idx+1].frac_score < 0.2) or \
                                (self.spec_obj.solutions[idx+1].score < G.MULTILINE_MIN_SOLUTION_SCORE * 0.6 ):
                                #allow some room below the minium since we are also matching to P(LyA), say 60% of minimum?
                                break #we've reached a point where there are no good scores
                            else:
                                continue
                        else:
                            continue #keep looking for a match
                    else:
                        if SU.map_multiline_score_to_confidence(scale_score) < 0.4: #low confidence
                            if z > 1.8 and self.best_gmag is not None and self.best_gmag < 23.5: #"high-z"
                                agree = False
                        elif SU.map_multiline_score_to_confidence(scale_score) < 0.6: #iffy confidence
                            if z > 1.8 and self.best_gmag is not None and self.best_gmag < 23.5: #"high-z"
                                if self.best_gmag < 23.0:
                                    agree = False
                                else:
                                    agree = True
                                    unsure = True
                                    self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                        else:
                            agree = True #or rather, they don't disagree
                        break

                #sol is at the last solution from at the break
                if agree:
                    if unsure:
                        #all the confidence now comes from the multiline score
                        if (num_solutions > 1) and (self.spec_obj.solutions[1].frac_score > 0):
                            if (self.spec_obj.solutions[0].frac_score / self.spec_obj.solutions[1].frac_score) > 1.9:
                                #notice: 1.9 is used instead of 2.0 due to rounding
                                #i.e. if only two solutions at 0.66 and 0.34 ==> 0.66/0.34 = 1.94
                                p = SU.map_multiline_score_to_confidence(sol.scale_score)
                            else:
                                p = SU.map_multiline_score_to_confidence(sol.scale_score) * sol.frac_score
                        else:
                            p = SU.map_multiline_score_to_confidence(sol.scale_score)

                        log.info(f"Q(z): Multiline solution[{idx}], score {scale_score}, frac {sol.frac_score}. "
                                 f"P(LyA) uncertain {scaled_plae_classification}. Set to z: {z} with Q(z): {p}")

                        self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                    else:
                        #basic agreement P(LyA) favors NOT LyA and multiline is not a LyA solution (less supportive)
                        #or P(LyA) favors LyA and so does multiline (more supportitve)
                        p = SU.map_multiline_score_to_confidence(scale_score)
                        if scaled_plae_classification > 0.5: #more supportive case
                            p = 0.5 * (p + scaled_plae_classification) #half from the P(LyA) and half from the scale_score

                        multiline_sol_diag = 1 #good and agree
                        log.info(f"P(z): Multiline solution[{idx}] {self.spec_obj.solutions[idx].name} score {scale_score} "
                                 f"and P(LyA) {scaled_plae_classification} agree. Set to z: {z} with Q(z): {p}")
                else: #use the 1st (highes score) that disagrees
                    sol = self.spec_obj.solutions[0]
                    z = sol.z
                    rest = sol.central_rest
                    score = sol.score
                    pscore = SU.map_multiline_score_to_confidence(sol.scale_score)

                    #if this is bright and broad, this may be an AGN ...
                    possible_agn = False
                    line_velocity = 0
                    try:
                        line_velocity = max(self.spec_obj.central_eli.fit_sigma *2.355,self.fwhm)/self.w * 3e5
                        if line_velocity > G.FWHM_TYPE1_AGN_VELOCITY_THRESHOLD: #AGN?
                            possible_agn = True
                    except:
                        pass

                    #P(LyA) and the multi-line score disagree
                    #the P(LyA) and scale_score are roughly on the same 0-1 scaling so we will choose the LyA favoring
                    # solution as THE solution and subtract away the dissent
                    if (scaled_plae_classification < 0.4) and (rest == G.LyA_rest):
                        #Not LyA vs LyA
                        #if the LyA Solution is strong, subtract off the p/2 weight

                        #test for weak LyA multi-line vs strong P(LyA) near 0.0
                        # for_lya = pscore - p
                        # for_oii = p

                        if possible_agn:
                            p = pscore
                            z = self.w / G.LyA_rest - 1.0
                            log.info(f"Q(z): Multiline solution favors LyA {pscore}. "
                                     f"P(LyA) does not {scaled_plae_classification}. Large velocity, possible AGN. Set to LyA z:{z} with Q(z): {p}")

                        elif pscore > p: #multi-line solution "stronger" than P(LyA)
                            p = max(0.05,pscore - p)
                            z = self.w / G.LyA_rest - 1.0

                            log.info(f"Q(z): Multiline solution favors LyA {pscore}. "
                                     f"P(LyA) does not {scaled_plae_classification}. Set to LyA z:{z} with Q(z): {p}")
                        else:
                            p = max(0.05,p-pscore)
                            z = self.w / G.OII_rest - 1.0

                            log.info(f"Q(z): Multiline solution favors LyA {pscore}. "
                                 f"P(LyA) does not {scaled_plae_classification}. Set to OII z:{z} with Q(z): {p}")

                    elif (scaled_plae_classification > 0.6) and (rest != G.LyA_rest):
                        #LyA vs Not
                        #voting vs line solution mis-match
                        #so 0.4 to 0.6 is no-man's land, but the prob or confidence will be very low anyway
                        #keep the z, but reduce the p(z)
                        if pscore > p:
                            p = max(0.05,pscore - p)
                            log.info(f"Q(z): Multiline solution favors z = {sol.z}; {pscore}. "
                                     f"P(LyA) favors LyA {scaled_plae_classification}. Set to z:{z} with Q(z): {p}")
                        else:
                            p = max(0.05,p - pscore)
                            z = self.w / G.LyA_rest - 1.0
                            log.info(f"Q(z): Multiline solution favors z = {sol.z}; {pscore}. "
                                     f"P(LyA) favors LyA {scaled_plae_classification}. Set to LyA z:{z} with Q(z): {p}")

                    elif pscore < 0.6: #we are ignoring the Q(z) of the multiline solution as too inconsistent
                        #we will use the P(Lya)
                        self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                        p = max(0.05,p - pscore)

                        if scaled_plae_classification < 0.5:
                            z = self.w / G.OII_rest - 1.0

                            if possible_agn: #not likely OII given the velocity width, though could still be broadend
                                p = min(0.2,p/2.0)
                                log.info(f"Q(z): Multiline solution rejected as weak and inconsistent. "
                                         f"P(LyA) favors OII {scaled_plae_classification}, but large velocity. Set to OII z:{z} with Q(z): {p}")
                            else:

                                log.info(f"Q(z): Multiline solution rejected as weak and inconsistent. "
                                         f"P(LyA) favors OII {scaled_plae_classification}. Set to OII z:{z} with Q(z): {p}")
                        else:
                            z= self.w / G.LyA_rest - 1.0

                            log.info(f"Q(z): Multiline solution rejected as weak and inconsistent. "
                                     f"P(LyA) favors LyA {scaled_plae_classification}. Set to LyA z:{z} with Q(z): {p}")


                    else: #odd place ... this should not happen
                        #we will use the P(Lya)
                        self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                        if scaled_plae_classification < 0.5:
                            z = self.w / G.OII_rest - 1.0
                        else:
                            z= self.w / G.LyA_rest - 1.0

                        log.warning(f"Q(z): unexpected outcome. Multiline solutions present, but cannot match. No Q(z) set.")


                try:
                    if sol.separation > 2.0: #this came from an external catalog
                        self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                        self.flags |= G.DETFLAG_EXT_CAT_QUESTIONABLE_Z
                except:
                    pass

            #else there are no multi-line classification solutions
            elif scaled_plae_classification < 0.3: #not LyA ... could still be high-z

                #usually OII execpt if REALLY broad, then more likely MgII or CIV. but have to mark as OII, just lower the Q(z)
                try:
                    broad = self.fwhm/self.w * 3e5 > 1000
                except:
                    broad = self.fhwm > 16.0

                if broad:
                    z = self.w / G.OII_rest - 1.0
                    rest = G.OII_rest
                    p = min(p/2.,0.1) #remember, this is just NOT LyA .. so while OII is the most common, it is hardly the only solution

                    sbl_z,sbl_name = self.spec_obj.single_broad_line_redshift(self.w,self.fwhm)
                    if sbl_z is not None:
                        log.info(f"Q(z): no multiline solutions. Really broad ({self.fwhm:0.1f}AA), so not likely OII. "
                                 f"P(LyA) favors NOT LyA. Set to single broadline ({sbl_name}) z:{sbl_z:04f} with Q(z): {p}.")
                        z = sbl_z
                    else:
                        log.info(f"Q(z): no multiline solutions. Really broad ({self.fwhm:0.1f}AA), so not likely OII, but not other good solution. "
                                 f"P(LyA) favors NOT LyA. Set to OII z:{z:04f} with Q(z): {p}. Could still be AGN with LyA or CIV, CIII or MgII alone.")


                    self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION

                else:
                    z = self.w / G.OII_rest - 1.0
                    rest = G.OII_rest
                    p = plya_for_oii*p/2. #remember, this is just NOT LyA .. so while OII is the most common, it is hardly the only solution
                         #so the highest possible would tbe 50%: P(LyA) = 0.0 ==> abs(0.5-0.0)/0.5/2. = 0.5

                    #limit p to a maximum
                    if self.flags & G.DETFLAG_DEX_GMAG_INCONSISTENT:
                        p = min(p,0.1)
                    else:
                        p = min(p,0.8)

                    #p might already be lower because of above check, but if there is a large neighbor and no other limit
                    #reduce p
                    if self.flags & G.DETFLAG_LARGE_NEIGHBOR:
                        p = min(p,0.4)

                    log.info(f"Q(z): no multiline solutions. P(LyA) favors NOT LyA. Set to OII z:{z} with Q(z): {p}")
            elif scaled_plae_classification > 0.7:
                z= self.w / G.LyA_rest - 1.0
                rest = G.LyA_rest

                #limit p to a maximum
                if self.flags & G.DETFLAG_DEX_GMAG_INCONSISTENT:
                    p = min(p,0.1)
                else:
                    p = min(p,0.8)

                #p might already be lower because of above check, but if there is a large neighbor and no other limit
                #reduce p
                #Not here ... the large neighbor only drives us toward NOT LyA; so if says LyA, then no need to lower
                #the p value
                # if self.flags & G.DETFLAG_LARGE_NEIGHBOR:
                #     p = min(p,0.4)

                log.info(f"Q(z): no multiline solutions. P(LyA) favors LyA. Set to LyA z:{z} with Q(z): {p}")
            else: #we are in no-man's land
                if scaled_plae_classification < 0.5:
                    z = self.w / G.OII_rest - 1.0
                else:
                    z = self.w / G.LyA_rest - 1.0

                #limit p to a maximum
                if self.flags & G.DETFLAG_DEX_GMAG_INCONSISTENT:
                    p = min(p,0.1)
                else:
                    p = min(p,0.25)

                #p might already be lower because of above check, but if there is a large neighbor and no other limit
                #reduce p
                #... the large neighbor only drives us toward NOT LyA; so if says LyA, then no need to lower
                #the p value
                if  (z < 0.6) and (self.flags & G.DETFLAG_LARGE_NEIGHBOR):
                    p = min(p,0.4)

                log.info(f"Q(z): no multiline solutions, no strong P(LyA). z:{z} with Q(z): {p}")


            #sanity check --- override negative z
            if z < 0: #unless this is a star, kick it out
                z = self.w / G.LyA_rest - 1.0
                if scaled_plae_classification > 0.5:
                    p = scaled_plae_classification
                else:
                    p = max(0.01,0.5 - scaled_plae_classification) #todo: need to figure a better value (not much else it can be than LyA)

            #check that if the z > 1.9 (probably means LAE) but the g or r mag is in the questionabale range
            #and the eqivalent widht is also questionable (with error between 15-25)
            #we are already in the case where there are no mulit-line solutions, so this is a single emission line
            try:
                ew_low = (self.best_eqw_gmag_obs - self.best_eqw_gmag_obs_unc) / (1+z)
                ew_hi = (self.best_eqw_gmag_obs + self.best_eqw_gmag_obs_unc) / (1+z)
                if z > 1.9 and \
                        ( (self.best_gmag is not None and 20 < self.best_gmag < adjusted_mag_zero(G.LAE_G_MAG_ZERO,z)) or \
                          (self.best_img_r_mag is not None and 20 < self.best_img_r_mag[0] < adjusted_mag_zero(G.LAE_R_MAG_ZERO,z)) ) and \
                        ( (G.LAE_EW_MAG_TRIGGER_MIN < ew_hi) and (ew_low < G.LAE_EW_MAG_TRIGGER_MAX)):

                    #set a flag and lower Q(z)
                    p = min(p,0.4) #very rough ...figure 50/50 LAE vs OII with a few other possibilities?
                    self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                    log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_UNCERTAIN_CLASSIFICATION (in best_redshift). "
                             f"EW uncertain: {ew_low:0.2f} - {ew_hi:0.2f}")
            except:
                log.debug("Exception sanity checking best_z in hetdex::DetObj::best_redshift()",exc_info=True)

            #check aperture correction at spectrum midpoint ... can be excessive
            #yes ... the clustering could undo this and that is okay
            try:
                apcor = self.sumspec_apcor[515]
                if apcor < 0.9 and multiline_sol_diag < 1 and self.best_gmag > 23.0:
                    log.info(f"Modifying Q(z) by x{apcor*apcor:0.2f} due to high aperture correction {apcor:0.2f}")
                    p *= apcor * apcor
            except:
                pass

            if self.cluster_parent != 0 and z == self.cluster_z and self.cluster_qz > 0:
                log.info(f"Clustering. Setting Q(z) to cluster parent Q(z): {self.cluster_qz:0.2f} ")
                p = self.cluster_qz

            #last checks
            try:
                #if there is no imaging and the DEX spectrum continuum is highly uncertain ...
                if (self.flags & G.DETFLAG_NO_IMAGING) and ((self.best_gmag_cgs_cont_unc is None) or \
                    (self.best_gmag_cgs_cont_unc / self.best_gmag_cgs_cont) > 0.9) and multiline_sol_diag < 1:
                    #if there is no multiline solution ...
                    #really faint, severely limit the quality
                    if self.best_gmag  > G.HETDEX_CONTINUUM_MAG_LIMIT:
                        p = min(p,0.1)
                    else:
                        p = min(p,0.2)
            except:
                pass

            try:
                if self.snr is not None and 0 < self.snr < 5.5:
                    p *= np.exp(self.snr-5.5)

            except:
                pass

            if p <= 0.1 and not (self.flags & G.DETFLAG_UNCERTAIN_CLASSIFICATION):
                self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                log.info(f"Detection Flag set for {self.entry_id}: DETFLAG_UNCERTAIN_CLASSIFICATION (in best_redshift)")

            self.best_z = z
            self.best_p_of_z = p

            return z,p
        except:
            log.warning("Exception! in hetdex.py DetObj::best_redshift()",exc_info=True)
            return 0,0

    def check_clustering_redshift(self):
        """
        Similar to check_spec_solutions_vs_catalog_counterparts(), but uses only other HETDEX/ELiXer detections
        as described in the cluster_list
        :param cluster_list:
        :return:
        """

        try:
            if self.cluster_list is None or len(self.cluster_list) == 0:
                log.info("No clustering list to check.")
                return

            #check the cluster_list for THIS detectid
            sel = [c['detectid']==self.hdf5_detectid for c in self.cluster_list]
            if np.sum(sel) == 0:
                log.info(f"No matching entry for ({self.hdf5_detectid}) in clustering list.")
                return
            elif np.sum(sel) > 1:
                log.error(f"Unexpected multiple hits ({np.sum(sel)}) for ({self.hdf5_detectid}) in clustering list.")
                return

            #else we have exacly one hit
            #get as index
            i = np.argmax(sel)
            z = self.cluster_list[i]['neighbor_z']
            self.cluster_z = z
            self.cluster_qz = self.cluster_list[i]['neighbor_qz']
            cluster_flag = self.cluster_list[i]['flag']

            #get the matching line (should be exactly one)
            lines = self.spec_obj.match_lines(self.w,z,z_error=0.001,aa_error=None,allow_absorption=False)
            if lines is None or len(lines) == 0: #unexpected
                log.error("No lines returned to match clustering redshift.")
                return

            #should only be one, but just to be safe, enforce as the highest ranking (lowest rank value) line
            line = lines[np.argmin([l.rank for l in lines])]
            boost = G.CLUSTER_SCORE_BOOST
            central_eli = self.spec_obj.central_eli

            #check the existing solutions ... if there is a corresponding z solution, boost its score
            new_solution = True
            for s in self.spec_obj.solutions:
                if s.central_rest == line.w_rest:
                    new_solution = False
                    log.info(f"Boosting existing solution:  {line.name}({line.w_rest}) + {boost}")
                    s.score += boost


            if new_solution:
                sol = elixer_spectrum.Classifier_Solution()
                sol.z = z
                sol.best_pz = self.cluster_list[i]['neighbor_qz'] #assign parent's Q(z) [only here if there was no other solution already]
                sol.central_rest = line.w_rest
                sol.name = line.name
                sol.color = line.color
                sol.emission_line = deepcopy(line)
                #add the central line info for flux, etc to line (basic info)
                if central_eli is not None:
                    sol.emission_line.line_score = central_eli.line_score
                    sol.emission_line.flux = central_eli.line_flux
                    sol.emission_line.flux_err = central_eli.line_flux_err
                    sol.emission_line.snr = central_eli.snr

                sol.emission_line.w_obs = self.w
                sol.emission_line.solution = True
                sol.prob_noise = 0
                #sol.score = boost

                if self.spec_obj.consistency_checks(sol):
                    sol.score = boost
                    sol.lines.append(sol.emission_line) #have to add as if it is an extra line
                    #otherwise the scaled score gets knocked way down
                    log.info(f"Clustering: Adding new solution {line.name}({line.w_rest}): score = {boost}")
                else: # reduce the weight ... but allow to conitnue??
                    sol.score = G.MULTILINE_MIN_SOLUTION_SCORE
                    log.info(f"Rejected catalog new solution {line.name}({line.w_rest}). Failed consistency check. Solution score set to {sol.score}")


                self.spec_obj.solutions.append(sol)
                self.flags != cluster_flag #e.g. if neighbors with matching lines have different z, then mark as uncertain

            self.cluster_parent = self.cluster_list[i]['neighborid']
            self.spec_obj.rescore()
        except:
            log.warning("Exception! in hetdex.py DetObj::check_clustering_redshift()",exc_info=True)


    def check_spec_solutions_vs_catalog_counterparts(self):
        """
        Needs to have already performed all the catalog matches
        Uses self.bid_target_list to match up any spec-z and phot-z vs rest-frame line solutions
        (similar to checking vs SDSS redshift)

        Don't need to check the RA, Dec as that is already done in the catalog matching

        """
        try:
            if (self.bid_target_list is None) or not G.CHECK_ALL_CATALOG_BID_Z:
                return

            #allow each catalog, each match to be scored independently
            #s|t multiple hits to the same spec_z and/or phot_z each boost that solution
            #(usually a form of confirmation)
            list_z = []
            possible_lines = []

            for b in self.bid_target_list:
                if not (b.selected or b.distance < 1.0): #has to be the selected counterpart
                    continue

                #sanity check ... is there a g or r mag that is at least roughly consistent with the DEX g mag?
                if self.best_gmag is not None and self.best_gmag > 0 and b.bid_mag is not None and b.bid_mag > 0:
                    if not (abs(b.bid_mag-self.best_gmag) < 1.5 or b.bid_mag > 24.5 and self.best_gmag > 24.5):
                        continue

                #check spec-z (higher boost)
                if b.spec_z is not None and b.spec_z > -0.02:
                    list_z.append({'z':b.spec_z,'z_err':0.01,'boost':G.ALL_CATATLOG_SPEC_Z_BOOST,'name':b.catalog_name,
                                   'mag':b.bid_mag,'filter':b.bid_filter,'distance':b.distance,'type':'s'})

                #then check phot-z (lower boost)
                #allowed to have the smaller of 0.5 or 20% error in z
                if b.phot_z is not None and b.phot_z > -0.02:
                    list_z.append({'z':b.phot_z,'z_err':min(0.25, b.phot_z * 0.2),'boost':G.ALL_CATATLOG_PHOT_Z_BOOST,'name':b.catalog_name,
                                   'mag':b.bid_mag,'filter':b.bid_filter,'distance':b.distance,'type':"p"})

            for bid in list_z:
                boost = bid['boost']
                z = bid['z']

                #line = self.spec_obj.match_line(self.w,z,allow_absorption=True)
                if bid['z_err'] > 0.1 or bid['type']=='p': #phot_z
                    max_rank = 3
                    allow_absorption = False
                    phot_z_only = True
                else: #spec_z
                    max_rank = 4
                    allow_absorption = True
                    phot_z_only = False

                lines = self.spec_obj.match_lines(self.w,z,z_error=bid['z_err'],aa_error=None,
                                                  allow_absorption=allow_absorption,max_rank=max_rank)

                try:
                    central_eli = self.spec_obj.central_eli
                    line_score = max(0,self.spec_obj.central_eli.line_score)
                except:
                    central_eli = None
                    line_score = 0.

                for line in lines:
                    log.info(f"{bid['name']} possible z match: {line.name} {line.w_rest} z={z} rank={line.rank}")
                    possible_lines.append(line)

                    #note: if d25 is > GALAXY_MASK_D25_SCORE_NORM, this actually starts reducing the boost as we
                    # get farther from the body of the galaxy mask
                    #lines like OII, OIII, H_beta, LyA, CIV are all low rank lines and get high boosts
                    #where H_eta, CaII, NaI are higher rank (weaker lines) and get lower boosts (and are not
                    #likely to be the HETDEX detection line anyway)

                    rank_scale = 1.0 if line.rank <= 2 else 1.0/(line.rank-1.0)

                    boost = boost * rank_scale * (1.0 if bid['z_err'] > 0.1 else 2.0) + line_score

                    #check the existing solutions ... if there is a corresponding z solution, boost its score
                    new_solution = True
                    for s in self.spec_obj.solutions:
                        if s.central_rest == line.w_rest:
                            new_solution = False
                            log.info(f"Boosting ({'phot-z' if phot_z_only else 'spec-z'}) existing solution:  {line.name}({line.w_rest}) + {boost}")
                            s.score += boost


                    if new_solution and (line.solution):

                        #see if we can confirm a multi-line solution at this redshift
                        #chances are we would have already found it earlier if it existed, but may as well check

                        #if there were a multi-line solution consistent with this redshift, we MUST have already found
                        #it, so don't bother checking again (note: that is the else case)
                        #instead just see if a single line solution is allowable.

                        if not phot_z_only:
                            if self.spec_obj.single_emission_line_redshift(line,self.w):
                                boost /= 2.0 #cut in half for a new solution (as opposed to boosting an existing solution)
                                sol = elixer_spectrum.Classifier_Solution()
                                sol.z = self.w/line.w_rest - 1.0
                                sol.central_rest = line.w_rest
                                sol.name = line.name
                                sol.color = line.color
                                sol.emission_line = deepcopy(line)
                                #add the central line info for flux, etc to line (basic info)
                                if central_eli is not None:
                                    sol.emission_line.line_score =central_eli.line_score
                                    sol.emission_line.flux = central_eli.line_flux
                                    sol.emission_line.flux_err = central_eli.line_flux_err
                                    sol.emission_line.snr = central_eli.snr

                                sol.emission_line.w_obs = self.w
                                sol.emission_line.solution = True
                                sol.prob_noise = 0
                                #sol.score = boost

                                if self.spec_obj.consistency_checks(sol):
                                    sol.score = boost
                                    #if one of the primaries ... z:[1.88,3.53] = LyA or z:[0,0.49], then this needs an
                                    #extra boost since it is only a single line
                                    if (0 < sol.z < 0.49) or (1.88 < sol.z < 3.53):
                                        sol.score += sol.emission_line.line_score
                                    sol.lines.append(sol.emission_line) #have to add as if it is an extra line
                                    #otherwise the scaled score gets knocked way down
                                    log.info(f"Catalog z: Adding new solution {line.name}({line.w_rest}): score = {sol.score}")
                                else: # reduce the weight ... but allow to conitnue??
                                    sol.score = G.MULTILINE_MIN_SOLUTION_SCORE
                                    log.info(f"Minimum catalog new solution {line.name}({line.w_rest}). Failed consistency check. Solution score set to {sol.score}")


                                self.spec_obj.solutions.append(sol)
                        else:
                            self.phot_z_votes.append(z)



            if possible_lines: #one or more possible matches
                #todo: future, instead of adding a label, set a flag to appear in the HDF5 file
                #self.spec_obj.add_classification_label(label,prepend=True)
                # rescore the solutions from above boosts
                self.spec_obj.rescore()


        except:
            log.warning("Exception! in hetdex.py DetObj::check_spec_solutions_vs_catalog_counterparts()",exc_info=True)

    def check_transients_and_flags(self): #,meteors=True,galaxy_mask=True):
        """
        Check for meteors
        Check if in galaxy mask
        :return:
        """

        if G.CHECK_FOR_METEOR:
            self.check_for_meteor()

        try:
            possible_lines = []
            flag_galaxy_mask = False
            if G.CHECK_GALAXY_MASK and self.galaxy_mask:
                log.debug("Checking position against galaxy mask ...")
                #get list of possible redshifts
                self.galaxy_mask_z, self.galaxy_mask_d25 = self.galaxy_mask.redshift(self.my_ra,self.my_dec)
                #get list of possible lines from list of emission lines
                #spec = elixer_spectrum.Spectrum()
                if ( (self.galaxy_mask_z is not None) and (len(self.galaxy_mask_z) > 0)) and (self.spec_obj is not None): #is not None or Empty

                    try:
                        central_eli = self.spec_obj.central_eli
                        line_score = max(0,self.spec_obj.central_eli.line_score)
                    except:
                        central_eli = None
                        line_score = 0.

                    for z,d25 in zip(self.galaxy_mask_z,self.galaxy_mask_d25):
                        lines = self.spec_obj.match_lines(self.w,z,aa_error=5.0) #emission only
                        for line in lines:
                            #specific check for OIII 4959 or 5007
                            if line.w_rest == 4959: #this is more dodgy ... 5007 might be strong but fail to match
                                if SU.check_oiii(z,self.sumspec_flux,self.sumspec_fluxerr,self.sumspec_wavelength,
                                              cont=self.cont_cgs,cont_err=self.cont_cgs_unc) == 0:
                                    #explicit NO
                                    log.info(f"[rejected] Galaxy mask possible line match (failed flux ratio test): {line.name} {line.w_rest} z={z} rank={line.rank} D25={d25}")
                                    continue

                            log.info(f"Galaxy mask possible line match: {line.name} {line.w_rest} z={z} rank={line.rank} D25={d25}")
                            possible_lines.append(line)

                            #note: if d25 is > GALAXY_MASK_D25_SCORE_NORM, this actually starts reducing the boost as we
                            # get farther from the body of the galaxy mask
                            #lines like OII, OIII, H_beta, LyA, CIV are all low rank lines and get high boosts
                            #where H_eta, CaII, NaI are higher rank (weaker lines) and get lower boosts (and are not
                            #likely to be the HETDEX detection line anyway)

                            rank_scale = 1.0 if line.rank <= 2 else 1.0/(line.rank-1.0)

                            boost = G.GALAXY_MASK_SCORE_BOOST * rank_scale * G.GALAXY_MASK_D25_SCORE_NORM/d25 + line_score

                            #check the existing solutions ... if there is a corresponding z solution, boost its score
                            new_solution = True
                            for s in self.spec_obj.solutions:
                                if s.central_rest == line.w_rest:
                                    new_solution = False
                                    log.info(f"Boosting existing solution:  {line.name}({line.w_rest}) + {boost}")
                                    s.score += boost
                                    s.galaxy_mask_d25 = d25

                            if new_solution and (line.solution):
                                log.info(f"Galaxy mask: Adding new solution {line.name}({line.w_rest}): score = {boost}")
                                sol = elixer_spectrum.Classifier_Solution()
                                sol.z = self.w/line.w_rest - 1.0
                                sol.central_rest = line.w_rest
                                sol.name = line.name
                                sol.color = line.color
                                sol.emission_line = deepcopy(line)
                                #add the central line info for flux, etc to line (basic info)
                                if central_eli is not None:
                                    sol.emission_line.line_score =central_eli.line_score
                                    sol.emission_line.flux = central_eli.line_flux
                                    sol.emission_line.flux_err = central_eli.line_flux_err
                                    sol.emission_line.snr = central_eli.snr
                                sol.emission_line.w_obs = self.w
                                sol.emission_line.solution = True
                                sol.prob_noise = 0
                                sol.lines.append(sol.emission_line) #have to add as if it is an extra line
                                #otherwise the scaled score gets knocked way down
                                sol.score = boost
                                sol.galaxy_mask_d25 = d25

                                self.spec_obj.solutions.append(sol)


                if possible_lines: #one or more possible matches
                    #todo: future, instead of adding a label, set a flag to appear in the HDF5 file
                    self.galaxy_mask_flag = True
                    self.spec_obj.add_classification_label("gal",prepend=True)
                    # rescore the solutions from above boosts
                    self.spec_obj.rescore()

        except:
            log.warning("Failed to check galaxy mask",exc_info=True)

        sdss_hit = False
        if not flag_galaxy_mask and G.CHECK_SDSS_Z_CATALOG:     #don't bother with SDSS catalog if already flagged in galaxy mask
            try:
                #catalog is bound to the class, so don't need an object
                log.debug("Checking against SDSS z-catalog...")
                z_sdss, z_err_sdss,sep_sdss, label_sdss = cat_sdss.SDSS.redshift(self.my_ra,self.my_dec)

                #duplicated code above, should refactor
                #
                #could hit more than one object ... if at the same z, then the same solution will get
                #multiple boosts (that is okay I think)
                #if at different z then different solutions could get a boost and would win based on the
                #  scoring
                if (z_sdss is not None) and (self.spec_obj is not None) and (len(z_sdss) > 0): #is not None or Empty
                    log.debug(f"{len(z_sdss)} matching SDSS records found.")
                    try:
                        central_eli = self.spec_obj.central_eli
                        line_score = max(0,self.spec_obj.central_eli.line_score)
                    except:
                        central_eli = None
                        line_score = 0.

                    for z,z_err,sep,label in zip(z_sdss,z_err_sdss,sep_sdss,label_sdss):
                        if z_err > 0.5 or ((z!= 0) and (z_err > 0.01) and (z_err/z > 0.2)):
                            continue #error too high

                        if z_err > 0.1: #phot_z
                            max_rank = 3
                            allow_absorption = False
                        else: #spec_z
                            max_rank = 4
                            allow_absorption = True

                        #mix of photz and specz (and can't tell?)
                        lines = self.spec_obj.match_lines(self.w,z,z_error=max(0.01,z_err),
                                                        allow_absorption=allow_absorption,max_rank=max_rank) #emission only
                        #in this case there should be exactly one, since only a single z
                        if lines and len(lines)>0:
                            for line in lines:
                                log.info(f"SDSS z-catalog possible line match: {line.name} {line.w_rest} z={z} rank={line.rank} sep={sep}")
                                possible_lines.append(line)

                                #note: if d25 is > GALAXY_MASK_D25_SCORE_NORM, this actually starts reducing the boost as we
                                # get farther from the body of the galaxy mask
                                #lines like OII, OIII, H_beta, LyA, CIV are all low rank lines and get high boosts
                                #where H_eta, CaII, NaI are higher rank (weaker lines) and get lower boosts (and are not
                                #likely to be the HETDEX detection line anyway)

                                #emission only
                                rank_scale = 1.0 if line.rank <= 2 else 1.0/(line.rank-1.0)

                                z_scale = 1.0 if z_err > 0.1 else 2.0
                                #todo: at this point we do not have the imaging and we do not know the size
                                #of our candidates, so we will just assume point sources here and with a typical seeing/size
                                #ideally we would base this on whether the SDSS coords are inside our candidate ellipse or
                                #how far outside
                                sep_scale = 1.0 if sep < 2.0 else 1/(sep - 1.0)
                                #for low error z, bump up a little more
                                boost = G.SDSS_SCORE_BOOST * rank_scale * z_scale * sep_scale + line_score

                                #check the existing solutions ... if there is a corresponding z solution, boost its score
                                new_solution = True
                                for s in self.spec_obj.solutions:
                                    if s.central_rest == line.w_rest:
                                        new_solution = False
                                        log.info(f"Boosting existing solution:  {line.name}({line.w_rest}) + {boost}")
                                        s.score += boost
                                        self.spec_obj.add_classification_label(label,prepend=True)

                                if new_solution and (line.solution) and self.spec_obj.single_emission_line_redshift(line,self.w):
                                    boost /= 2.0 #cut in half for a new solution (as opposed to boosting an existing solution)
                                    log.info(f"SDSS z: Adding new solution {line.name}({line.w_rest}): score = {boost}")
                                    sol = elixer_spectrum.Classifier_Solution()
                                    sol.z = self.w/line.w_rest - 1.0
                                    sol.central_rest = line.w_rest
                                    sol.name = line.name
                                    sol.color = line.color
                                    sol.emission_line = deepcopy(line)
                                    #add the central line info for flux, etc to line (basic info)
                                    if central_eli is not None:
                                        sol.emission_line.line_score =central_eli.line_score
                                        sol.emission_line.flux = central_eli.line_flux
                                        sol.emission_line.flux_err = central_eli.line_flux_err
                                        sol.emission_line.snr = central_eli.snr
                                    sol.emission_line.w_obs = self.w
                                    sol.emission_line.solution = True
                                    sol.prob_noise = 0
                                    sol.lines.append(sol.emission_line) #have to add as if it is an extra line
                                                                        #otherwise the scaled score gets knocked way down
                                    sol.score = boost
                                    sol.separation = sep_sdss

                                    self.spec_obj.solutions.append(sol)
                                    if sep_scale == 1.0:
                                        self.spec_obj.add_classification_label(label,prepend=True)

                    if possible_lines: #one or more possible matches
                        sdss_hit = True
                        #todo: future, instead of adding a label, set a flag to appear in the HDF5 file
                        #todo: also this only makes sense IF we end up selecting this as the solution
                        # rescore the solutions from above boosts
                        self.spec_obj.rescore()


                else:
                    log.debug("No matching SDSS records found.")

            except:
                log.warning("Failed to check SDSS z-catalog",exc_info=True)


        #if not flag_galaxy_mask and not sdss_hit and G.CHECK_GAIA_DEX_CATALOG:
        if False:
            try:
                log.debug("Checking against GAiA_DEX catalog...")
                rows = cat_gaia_dex.GAIA_DEX.query_catalog(self.my_ra,self.my_dec,error=5.0)

                #todo: check rows for a match with gmag and distance
                #todo and see if there is measured proper motion (then is likely a star)

            except:
                log.warning("Failed to check GAIA_DEX catalog",exc_info=True)

    def check_for_meteor(self):
        """
        examine the exposures (individually), stacking the calibrated spectra (over each fiber) in each exposure
        and check the residual (max - (sum of the others))
        I expect a normal detection to have nothing significant, but a meteor is bright and will appear in only one exposure
        :return:
        """

        #biweight stack or median or mean or sum?
        #since we really want this to stand out, I'm thinking sum

        #how many exposures? usually 3, but not always
        try:

            #todo: check to see if HETDEX position is actually inside the IFU ... if it is off the IFU (or even at the
            # very edge, the dither ratio can be a false trigger (since two of the dithers may have no fibers on
            # the position and even the nearest fiber may not cover the position)
            off_edge = False
            try:
                if (self.ifu_x is not None) and (self.ifu_y is not None) and \
                        not ((-24.15 <= self.ifu_x <= 24.15) or (-24.24 <= self.ifu_y <= 24.24)):
                    log.info(f"DetObj::check_for_meteor(). Position is off IFU edge ({self.ifu_x}, {self.ifu_y})")
                    off_edge = True
                    return 0 #todo: maybe still check with a modified trigger or dis-allow certain conditions?
            except:
                pass


            dither_norm_x = 1.0
            if (np.isnan(self.dither_norm) or (self.dither_norm > 2.0)):
                # log.info(f"DetObj::check_for_meteor(). Cannot check for meteor due to bad dither normalization ({self.dither_norm})")
                # return 0
                if np.isnan(self.dither_norm):
                    dither_norm_x = 3.0 #multiplier on the ratio triggers
                else:
                    dither_norm_x = self.dither_norm
                log.info(f"DetObj::check_for_meteor(). Dither norm limitations in place due to bad dither normalization ({self.dither_norm})")




            num_exp = len(np.unique([f.expid for f in self.fibers]))
            if num_exp < 2:
                log.info(f"DetObj::check_for_meteor(). Cannot check for meteor due to insufficient exposures ({num_exp})")
                return 0

            #indices that cover common meteor lines (MgI, Al, CaI, CaII) into CALFIB_WAVEGRID
            #these are a bit broader ranges than the more specific waves that are list later on in this function
            #and are used just to check the exposure vs exposure ratios
            common_line_waves = np.concatenate( (np.arange(3570,3590,2),
                                                 np.arange(3715,3745,2),
                                                 np.arange(3824,3844,2),
                                                 np.arange(3852,3864,2),
                                                 np.arange(3926,3942,2),
                                                 np.arange(3960,3976,2),
                                                 np.arange(4210,4250,2),
                                                 np.arange(4400,4450,2),
                                                 np.arange(5160,5220,2)))

            common_line_idx = np.searchsorted(G.CALFIB_WAVEGRID, common_line_waves)

            #planning ahead a bit, not quite sure how to use this yet
            #this is a an average flux density in areas not commonly covered by meteor emission lines
            #so this can indicate a lot of continuum
            # a REAALY bright meteor would have continuum too, but this is from the PSF weighted spectra and the other
            # contributing exposures would normally have very little
            exclude_common_line_idx = np.setdiff1d(np.arange(len(G.CALFIB_WAVEGRID)),common_line_idx)
            avg_exclude_flux_density = np.sum(self.sumspec_flux[exclude_common_line_idx])/(G.FLUX_WAVEBIN_WIDTH*len(exclude_common_line_idx))
            avg_exclude_flux_density_err = np.sum(self.sumspec_fluxerr[exclude_common_line_idx])/(G.FLUX_WAVEBIN_WIDTH*len(exclude_common_line_idx))
            min_avg_exclude_flux_density = avg_exclude_flux_density - avg_exclude_flux_density_err

            if min_avg_exclude_flux_density > 4.0: #erg/s/cm2/AA  #> 10e-17 in HETDEX CGS
                #definite detection of continuum
                log.info(f"DetObj::check_for_meteor(). Bright continuum ({min_avg_exclude_flux_density}) excluding common meteor lines. Not a meteor.")
                return 0


            exp = np.zeros((num_exp,len(G.CALFIB_WAVEGRID)))
            exp_err = np.zeros((num_exp,len(G.CALFIB_WAVEGRID)))

            #fibers are sorted by weight

            #using the weight_dict to find the exposure with the most weight really does not do anything
            #since it is based on the PSF model and the expected flux in each fiber from that and the weighted center
            # and NOT the actual recorded flux in each fiber.

            # exposures = np.unique([f.expid for f in self.fibers])-1 #numbered (1,2,3) not (0,1,2)
            # weight_dict = dict(zip(exposures,np.zeros(len(exposures))))

            for f in self.fibers:
                idx = f.panacea_idx
                exp[f.expid-1] += f.fits.calfib[idx]
                exp_err[f.expid-1] += f.fits.calfibe[idx]
                # weight_dict[f.expid-1] += f.relative_weight

            # try:
            #     max_exposure_weight = max(weight_dict.values())
            # except:
            #     max_exposure_weight = 0
            #


            #max_fiber_weight picks up on definitely odd/wrong stuff, but not meteors
            # try:
            #     max_fiber_weight = max([x.relative_weight for x in self.fibers])
            # except:
            #     max_fiber_weight = 0

            #sum over JUST the COMMON areas
            common_sum = np.zeros(num_exp)
            common_sume = np.zeros(num_exp)
            for i in range(num_exp):
                common_sum[i] = np.nansum(exp[i][common_line_idx])
                common_sume[i] = np.sqrt(np.nansum(exp_err[i][common_line_idx]**2))

            # which has the minimum (so can subtract off)
            #forces the lowest to be zero and keeps everything non-negative
            #but can really inflate the max/#2 ratio if #2 and #3 are really close s|t #2-#3 is a small number
            #and #1 is just a little larger, can start down the meteor path erroneously
            #SO .... only do this IF the min is negative
            if min(common_sum) < 0:
                common_sum -= min(common_sum)

            # which exposure has the maximum
            cmx_expid = np.argmax(common_sum) + 1
            cmx_sum = common_sum[cmx_expid - 1]

            # which has the minimum
            cmn_expid = np.argmin(common_sum) + 1
            cmn_sum = common_sum[cmn_expid - 1]

            # which is second highest (normally, the only one left, but there could be more than 3 exposures
            strip_sum = copy(common_sum)
            strip_sum[cmx_expid - 1] = -99999.
            cn2_expid = np.argmax(strip_sum) + 1
            cn2_sum = common_sum[cn2_expid - 1]

            near_bright_obj = False

            if True:
                #this can be tripped up by being near a bright object and exposures that move
                #closer to it can then become the largest (overall) just from continuum
                ###############################
                # Now for the entire spectrum
                ###############################

                #now, just sum across each (expect 1 exposure to stand out
                ssum = np.zeros(num_exp)
                ssume = np.zeros(num_exp)
                for i in range(num_exp):
                    ssum[i] = np.nansum(exp[i])
                    ssume[i] = np.sqrt(np.nansum(exp_err[i]**2))

                #which has the minimum (so can subtract off)
                ssum -= min(ssum)

                #which exposure has the maximum
                mx_expid = np.argmax(ssum)+1
                mx_sum = ssum[mx_expid-1]

                #which has the minimum
                mn_expid = np.argmin(ssum)+1
                mn_sum = ssum[mn_expid-1]

                #which is second highest (normally, the only one left, but there could be more than 3 exposures
                strip_sum = copy(ssum)
                strip_sum[mx_expid-1] = -99999.
                n2_expid = np.argmax(strip_sum)+1
                n2_sum = ssum[n2_expid-1]

                #median and std for later
                # med = np.median(ssum)
                # std = np.std(ssum)
                # #effectively, this is the num of std the maximum is above the next hightest (which will be the
                # #median of the three values
                # if std != 0:
                #     std_above = (mx_sum-med)/std
                # else:
                #     std_above = 0

                #make sure the two methods ID the same
                # BUT we could be near a bright object and then its scattered light could flip these around
                # so ignore this and just use the 'common' values at the end
                if (cmx_expid != mx_expid) or (cmn_expid != mn_expid):
                    log.info("DetObj::check_for_meteor expids do not match. Could be near a bright object OR not a meteor.")
                    near_bright_obj = True
                #     return 0

            if False:
                #sum over wavebins
                others_combined = np.zeros(len(G.CALFIB_WAVEGRID))
                others_combined_err = np.zeros(len(G.CALFIB_WAVEGRID))
                sel = np.where(np.arange(num_exp)!=(mx_expid-1))[0]
                #others_combined += ssum[i]

                #"maximum" case others
                others_sum = np.nansum(ssum[sel])
                others_sum_err = np.sqrt(np.nansum(ssume[sel]**2))

                if others_sum < 0: #just slide up
                    #the 50 is partly arbitrary and partly experimental
                    others_sum += others_sum_err
                    mx_sum += others_sum_err #or maybe the mx_sum's own error?
                    if others_sum < 0: #still less than zero
                        others_sum = others_sum_err #assume should be zero + error

                #others_sum = np.nansum(others_combined)

                #how does highest sum [-1] it compare to the next highest [-2]
                #next_largest = sorted(ssum)[-2]

                if others_sum > 25000.0: #sort of arbitrarily large value to exclude stars
                    return 0

                if others_sum == 0: #almost impossible unless there is a problem
                    log.debug("DetObj::check_for_meteor zero sums")
                    return 0
            else:
                #use the second highest as reference ... should be no way for it to be less than zero
                #since we subtract off the lowest sum
                if n2_sum > 25000.0:  # sort of arbitrarily large value to exclude stars
                    return 0

                if n2_sum == 0:  # almost impossible unless there is a problem
                    log.debug("DetObj::check_for_meteor zero sums")
                    return 0

                if n2_sum < 0:  #impossible?
                    log.debug("DetObj::check_for_meteor negative second sum")
                    return 0


            try: #if the errors are too big, just cannot trust anything
                if (cmx_sum / common_sume[cmx_expid-1]) < 1.0: #1.25: #2.0:
                    log.debug(f"DetObj:check_for_meteor sum error too high {cmx_sum} +/- {common_sume[cmx_expid-1]}")
                    return 0
            except:
                log.debug(f"DetObj:check_for_meteor sum (fail) error too high {cmx_sum} +/- {common_sume[cmx_expid - 1]}")
                return 0

            meteor = 0
            spec_ratio_trigger = 3.0 * dither_norm_x
            full_spec_ratio_trigger = 5.0 * dither_norm_x
            bright_obj_spec_ratio_trigger = 2.5 * dither_norm_x

            if cmx_sum > cn2_sum > 0:
                full_ratio = mx_sum / n2_sum
                common_ratio = cmx_sum / cn2_sum
                #spec_ratio = max(full_ratio,common_ratio)
                #spec_ratio = std_above
                spec_ratio = cmx_sum / cn2_sum
               # if ((full_ratio > 2 ) or (common_ratio > 4)): #maybe need more checking
                #minimum gate check, just to warrant addtional steps
                pos = []
                log.debug(f"Meteor check. full_ratio = {full_ratio:0.2f}, spec_ratio = {spec_ratio:0.2f}, near_bright = {near_bright_obj}")
                if ( (full_ratio > full_spec_ratio_trigger) or (spec_ratio > spec_ratio_trigger ) or \
                        ((near_bright_obj and (spec_ratio > bright_obj_spec_ratio_trigger)))):
                    #or \
                    #    ( (max_fiber_weight > 0.5) and (full_ratio > 1.5) ) ): #maybe need more checking
                    #merge in with the existing all found lines
                    try:
                        waves = [x.fit_x0 for x in self.spec_obj.all_found_lines]
                        #if we already have 4 or more lines from the PSF weighted spectra, these
                        #are going to be the brighter/stronger lines anyway,
                        #there is no need to run the "no fit" lines and risk a shotgun match
                        if len(waves) < 4:
                            pos = elixer_spectrum.sn_peakdet_no_fit(G.CALFIB_WAVEGRID, exp[cmx_expid - 1],
                                                                    exp_err[cmx_expid - 1],
                                                                    dx=3, rx=2, dv=3.0, dvmx=5.0)
                            quick_waves = G.CALFIB_WAVEGRID[pos]
                            for q in quick_waves:
                                if not np.any(np.isclose(q, waves, atol=6.0)):
                                    waves.append(q)

                            if not np.any(np.isclose(self.w,waves,atol=6.0)):
                                waves.append(self.w)
                    except:
                        log.info("Exception in DetObj::check_for_meteor()",exc_info=True)
                        waves = [] #this will basically fail out below

                    waves = np.array(sorted(waves))

                    bright_mg_line = np.where(  ((waves >= 3826) & (waves <= 3840)) |
                                                ((waves >= 5170) & (waves <= 5186)))[0]
                    common_lines = np.where( ((waves >= 3570) & (waves <= 3590)) |
                                             ((waves >= 3715) & (waves <= 3740)) |
                                             ((waves >= 3826) & (waves <= 3840)) |
                                             ((waves >= 3852) & (waves <= 3864)) |
                                             ((waves >= 3928) & (waves <= 3937)) |
                                             ((waves >= 3965) & (waves <= 3971)) |
                                             ((waves >= 4224) & (waves <= 4230)) |
                                             ((waves >= 5170) & (waves <= 5186))  )[0]

                    weighted_line_count = len(bright_mg_line) * 2 + len(common_lines)

                    #other lines CaII at 3934
                    #            Al    around 3968 (or Al and FeI 3962,3978)? (kinda weak)
                    #            CaI  at 4227
                    #            MgI  at 5173,5184
                    #            FeI  at 5330  (weak)

                    if len(waves) > 30:
                        #too many to trust
                        meteor = 0
                    elif len(common_lines) > 3: #got most of the common lines
                        if (len(waves) < 10):  # check for total waves (too many results in shotgun match)
                            meteor = 5 #common trigger
                            log.debug("+++++ meteor condition 7")
                        elif len(waves) < 20:  # getting close to shotgun
                            meteor = 3
                            log.debug("+++++ meteor condition 8")
                        elif len(waves) < 30:  # getting close to shotgun
                            meteor = 2
                            log.debug("+++++ meteor condition 8")
                    #full ratio in one exposure
                    elif (spec_ratio > spec_ratio_trigger):
                        #one or more of the Mg lines and  2 or more common lines (which can include MgI)
                        if (len(bright_mg_line) > 0) and (len(common_lines) > 1):
                            if (len(waves) < 10):  # check for total waves (too many results in shotgun match)
                                meteor = 3 #common trigger
                                log.debug("+++++ meteor condition 1")
                            elif len(waves) < 20: #getting close to shotgun
                                meteor = 2
                                log.debug("+++++ meteor condition 2")
                            elif len(waves) < 30: #getting close to shotgun
                                meteor = 1
                                log.debug("+++++ meteor condition 2")

                        elif (full_ratio > (0.75 * full_spec_ratio_trigger)):
                            if (len(bright_mg_line) > 0) : #only got a bright line
                                #if (len(waves) < 4):
                                #    log.debug("+++++ no enough other lines for meteor. condition 3")
                                if (len(waves) < 10):  # check for total waves (too many results in shotgun match)
                                    meteor = 2 #occasional trigger
                                    log.debug("+++++ meteor condition 3")
                                elif len(waves) < 20:  # getting close to shotgun
                                    meteor = 1
                                    log.debug("+++++ meteor condition 4")
                                elif len(waves) < 30:  # getting close to shotgun
                                    meteor = 0.5
                                    log.debug("+++++ meteor condition 4")
                            elif len(common_lines) > 1: #no bright lines, but 2+ common lines
                                    meteor = 1
                                    log.debug("+++++ meteor condition 5")

                    #reduced ratio but near a bright object
                    elif near_bright_obj and (spec_ratio > bright_obj_spec_ratio_trigger) and \
                            (full_spec_ratio_trigger > 0.75 * full_spec_ratio_trigger):
                        if (len(bright_mg_line) > 0) and (len(common_lines) > 1):
                            if (len(waves) < 10):  # check for total waves (too many results in shotgun match)
                                meteor = 2
                                log.debug("+++++ meteor condition 1b")
                            elif len(waves) < 20:
                                meteor = 1
                                log.debug("+++++ meteor condition 2b")
                            elif len(waves) < 30:
                                meteor = 0.5
                        elif (len(bright_mg_line) > 0):  # only got a bright line
                            if (len(waves) < 10):  # check for total waves (too many results in shotgun match)
                                meteor = 1.5
                                log.debug("+++++ meteor condition 3b")
                            elif len(waves) < 20:  # getting close to shotgun
                                meteor = 0.5
                                log.debug("+++++ meteor condition 4b")
                            elif len(waves) < 30:
                                meteor = 0.25
                        elif len(common_lines) > 1: #no bright lines, but 2+ common lines
                            if len(waves) < 10:
                                meteor = 0.5
                                log.debug("+++++ meteor condition 5b")
                            elif len(waves) < 30:
                                meteor = 0.25
                                log.debug("+++++ meteor condition 5b")
                            else:
                                meteor = 0 #don't trust it
                    # elif max_fiber_weight > 0.5 and ( (weighted_line_count > 3) or (8 < len(waves) < 20)):
                    #     log.debug("+++++ meteor condition 6a")
                    #     meteor = 0.5
                    else:
                        log.debug("Failed to meet additional meteor criteria. Likely not a meteor.")

                    #final check
                    if (meteor == 0)  \
                            and (((spec_ratio > 20) and (full_ratio > 5)) or ((spec_ratio > 40) and (full_ratio > 2.5))) \
                            and (len(waves) <= 20):
                        if len(common_lines) > 0: #got at least one
                            meteor = 1
                            log.debug("+++++ meteor condition 1c")
                    # CANNOT DO THIS : far too easy for AGN to fall into this
                    #     else:
                    #         meteor = 0.5
                    #         log.debug("+++++ meteor condition 1d")


                    if meteor > 0:
                        if spec_ratio > 5.0 or full_ratio > 3.0 or meteor >= 2:
                            self.spec_obj.add_classification_label("meteor",replace=True)
                        else:
                            self.spec_obj.add_classification_label("meteor",prepend=True)
                        self.spec_obj.meteor_strength = meteor
                        pos = np.array(pos)
                        if len(pos) > 0:
                            log.info(f"Meteor: Detection likely a meteor. Exp# {mx_expid} at x{spec_ratio:0.1f}, lines at {G.CALFIB_WAVEGRID[pos]}")
                        else:
                            log.info(f"Meteor: Detection likely a meteor. Exp# {mx_expid} at x{spec_ratio:0.1f}")
                        return 1
                else:
                    log.debug(f"Did not trigger initial meteor check. Targetted spec ratio {spec_ratio}. Full ratio {full_ratio}.")

            #for test
            if False:
                plt.plot(G.CALFIB_WAVEGRID,exp[0])
                plt.plot(G.CALFIB_WAVEGRID, exp[1])
                plt.plot(G.CALFIB_WAVEGRID, exp[2])
                plt.savefig("meteor_test.png")

                plt.figure(figsize=(15, 2))
                plt.plot(G.CALFIB_WAVEGRID, exp[mx_expid - 1] - others_combined)
                for p in pos:
                    plt.axvline(G.CALFIB_WAVEGRID[p])
                plt.savefig("meteor_test_sub.png")

        except:
            log.debug("Exception in hetdex::DetObj::check_for_meteor",exc_info=True)

        return 0


    def aggregate_classification(self):
        """
        Use all info available, weighted, to make a classification as to LAE or not LAE
        ?map to 0.0 (not LAE) to 1.0 (LAE)
        :return: plae, reason string
        """

        #assume this IS an LAE (so that's the NULL)

        #using some Bayesian language, but not really Bayesian, yet
        #assume roughly half of all detections are LAEs (is that reasonable?)

        vote_info = {} #dictionary of vote, weights, and info for a debugging HDF5 ClassificationExtraFeatures table

        def plae_poii_midpoint(obs_wave):
            """
            changes the 50/50 mid point of PLAE/POII based on the observed wavelength (or equivalently, on the
            redshift assuming LyA)
            Leung+2015 suggests improved performance in their simulations (see Table 3) by using
            PLAE/POII = 1.38 for z < 2.5 and 10.3 for z > 2.5
            :param obs_wave:
            :return:
            """

            #until we have good experimental data, just return 1.0 as the 50/50 midpoint
            #return 1.0

            try:
                #start with Andrew's binary condition
                # if obs_wave is not None and 3400.0 < obs_wave < 5600.0:
                #     if obs_wave < 4254: # z(LyA) = 2.5
                #         return 1.38
                #     else:
                #         return 10.3
                # else:
                #     return 1.0


                #modification on Leung+2015, rather than a hard binary case, linearly evolve with wavelength

                low_wave = 4000
                low_thresh = 1.0
                high_wave = 4500
                high_thresh = 10.0

                if obs_wave < low_wave:
                    return low_thresh
                elif obs_wave > high_wave:
                    return high_thresh
                else:
                    slope = (high_thresh-low_thresh)/(high_wave-low_wave)
                    inter = low_thresh - slope*low_wave
                    return slope * obs_wave + inter
            except:
                log.warning("Exception! Exception in plae_poii_midpoint. Set midpoint as 1.0", exc_info=True)


        def plae_gaussian_weight(plae_poii,obs_wave=None):
            #the idea here as the the closer the PLAE/POII is to 1 (a 50/50 chance) the lower the weight (tends to 0)
            # but extreme values (getting closer to 0.001 or 1000) the weight goes closer to a full value of 1
            # by a value of 20 (or 1/20) essentially at 1.0
            try:
                if plae_poii < 1:
                    plae_poii = 1.0/plae_poii

                #adjust the Gaussian center based on the observed wavelength
                if obs_wave is None:
                    mu = 1.0
                else:
                    mu = plae_poii_midpoint(obs_wave)
                sigma = G.PLAE_POII_GAUSSIAN_WEIGHT_SIGMA #5.0 #s|t by 0.1 or 10 you get 80% weight but near 1 you gain nothing
                return 1 - np.exp(-((plae_poii - mu)/(np.sqrt(2)*sigma))**2.)
            except: #could be exactly zero through either a fluke or a data issue
                log.warning("Exception! Exception in plae_gaussian_weight. Set weight to 0.0", exc_info=True)
                return 0


        def mag_gaussian_weight(mag_zero, mag, mag_bright, mag_faint):
            #todo: make use of error (bright, faint)
            #the idea here as the the closer the mag is to the zero point, the closer to 50/50
            #just for clarity
            try:
                mu = mag_zero
                sigma = G.LAE_MAG_SIGMA
                return 1-np.exp(-((mag - mu)/(np.sqrt(2)*sigma))**2.)
            except:
                return 0

        self.get_filter_colors()

        reason = ""
        base_assumption = 0.5 #not used yet
        likelihood = []
        weight = []
        var = [] #right now, this is just weight based, all variances are set to 1
        prior = [] #not using Bayes yet, so this is ignored

        #both are lists of dictionaries with keys: kpc, weight, likelihood
        diameter_lae = [] #physical units assuming a redshift consistent with LAE
        diameter_not_lae = [] #physical units assuming a redshift NOT consistend with LAE

        scaled_prob_lae = None
        fig = None

        #rules based


        ######################################
        # Angular Size (simplified)
        # basically if large and not super bright, probably OII
        # if large and bright could be OII or an AGN
        # if small could be either (no vote)
        # if REALLY small, probably LyA, but will limit to the large and not AGN case
        ######################################

        try:
            #assume the dictionary is populated ... if not, just except and move on
            if self.classification_dict['size_in_psf'] is not None and \
                    self.classification_dict['size_in_psf'] > 1.2: #we will call this resolved


                vote_info['size_in_psf'] = self.classification_dict['size_in_psf']
                vote_info['diam_in_arcsec'] = self.classification_dict['diam_in_arcsec']

                z_oii = self.w/G.OII_rest - 1
                diam = SU.physical_diameter(z_oii,self.classification_dict['diam_in_arcsec'])
                vote_info['oii_size_in_kpc'] = diam

                if diam < 6.0:# or self.classification_dict['diam_in_arcsec'] < 2.0: #usually we fall here
                    #small to medium
                    #probably LyA ... have to have really great seeing ... check the redshift
                    if  diam < 3.0: #0.5 kpc pretty clean
                        #vote FOR LyA
                        w = 0.25
                        likelihood.append(1.0)
                        weight.append(w)
                        var.append(1)
                        prior.append(base_assumption)

                        #set a base weight (will be adjusted later)
                        #diameter_lae.append({"z":z,"kpc":diam,"weight":w,"likelihood":lk})
                        log.info(
                            f"{self.entry_id} Aggregate Classification, angular size ({self.classification_dict['diam_in_arcsec']:0.2})\", added physical size. Unlikely OII:"
                            f" z(OII)({z_oii:#.4g}) kpc({diam:#.4g}) weight({w:#.5g}) likelihood({lk:#.5g})")

                        vote_info['size_in_psf_vote'] = likelihood[-1]
                        vote_info['size_in_psf_weight'] = weight[-1]
                    elif diam < 6.0: # and self.w < 4500:
                        #vote FOR LyA, but weaker, around 5:1 LyA
                        w = 0.05
                        likelihood.append(1.0)
                        weight.append(w)
                        var.append(1)
                        prior.append(base_assumption)

                        #set a base weight (will be adjusted later)
                        #diameter_lae.append({"z":z,"kpc":diam,"weight":w,"likelihood":lk})
                        log.info(
                            f"{self.entry_id} Aggregate Classification, angular size ({self.classification_dict['diam_in_arcsec']:0.2})\", added physical size. Unlikely OII:"
                            f" z(OII)({z_oii:#.4g}) kpc({diam:#.4g}) weight({w:#.5g}) likelihood({lk:#.5g})")

                        vote_info['size_in_psf_vote'] = likelihood[-1]
                        vote_info['size_in_psf_weight'] = weight[-1]

                    else:
                        log.info(f"{self.entry_id} Aggregate Classification angular size ({self.classification_dict['diam_in_arcsec']:0.2})\" no vote. Intermediate size.")

                elif self.classification_dict['diam_in_arcsec'] < 5.0:
                    #Meidum Big
                    #on the larger side ... generally more likely to be OII, but could be an AGN
                    #check on the FHWM for Type 1 AGN
                    if self.fwhm-self.fwhm_unc > 15: #about 1000 km/s at 4500AA
                        #weak vote FOR LyA (assuming AGN)
                        likelihood.append(1.0)
                        weight.append(0.25)
                        var.append(1)
                        prior.append(base_assumption)
                        log.info(f"{self.entry_id} Aggregate Classification: angular size ({self.classification_dict['diam_in_arcsec']:0.2})\" + consistent with AGN:"
                                 f" lk({likelihood[-1]}) weight({weight[-1]})")
                        vote_info['size_in_psf_vote'] = likelihood[-1]
                        vote_info['size_in_psf_weight'] = weight[-1]
                    elif self.fwhm+self.fwhm_unc < 10 : #not likely an AGN and otheriwse too big to be LyA
                        #vote for OII (or not LyA)
                        likelihood.append(0.0)

                        weight.append(0.33)
                        #weight.append(0.25)
                        var.append(1)
                        prior.append(base_assumption)
                        log.info(f"{self.entry_id} Aggregate Classification: angular size ({self.classification_dict['diam_in_arcsec']:0.2})\""
                                 f" lk({likelihood[-1]}) weight({weight[-1]})")
                        vote_info['size_in_psf_vote'] = likelihood[-1]
                        vote_info['size_in_psf_weight'] = weight[-1]
                    else:
                        log.info(f"{self.entry_id} Aggregate Classification angular size ({self.classification_dict['diam_in_arcsec']:0.2})\" no vote. Intermediate size. Not AGN, but large-ish FWHM.")

                else: # self.classification_dict['diam_in_arcsec'] > 5.0: #unless an AGN this is probably OII
                    #REALLY big
                    if self.fwhm-self.fwhm_unc > 15: #about 1000 km/s at 4500AA
                        #weak vote FOR LyA (assuming AGN)
                        likelihood.append(1.0)
                        weight.append(0.25)
                        var.append(1)
                        prior.append(base_assumption)
                        vote_info['size_in_psf_vote'] = likelihood[-1]
                        vote_info['size_in_psf_weight'] = weight[-1]
                        log.info(f"{self.entry_id} Aggregate Classification: angular size ({self.classification_dict['diam_in_arcsec']:0.2})\" + consistent with AGN:"
                                 f" lk({likelihood[-1]}) weight({weight[-1]})")
                    elif self.fwhm+self.fwhm_unc < 12 :
                        #vote for OII
                        likelihood.append(0.0)
                        weight.append(0.25)
                        var.append(1)
                        prior.append(base_assumption)
                        vote_info['size_in_psf_vote'] = likelihood[-1]
                        vote_info['size_in_psf_weight'] = weight[-1]
                        log.info(f"{self.entry_id} Aggregate Classification: angular size ({self.classification_dict['diam_in_arcsec']:0.2})\":"
                                 f" lk({likelihood[-1]}) weight({weight[-1]})")
                    else:
                        #no vote
                        log.info(f"{self.entry_id} Aggregate Classification angular size ({self.classification_dict['diam_in_arcsec']:0.2})\" no vote. Large size, intermediate line FWHM.")
            else:
                    log.info(f"{self.entry_id} Aggregate Classification angular size no vote (unresolved) or no size info.")
        except:
            log.warning(f"{self.entry_id} Aggregate Classification angular size exception.",exc_info=True)

        if False: #old physical size

            #################################
            # Physical Size Votes
            #################################


            # todo: use assumed redshift (OII vs LyA) and translate to physical size
            #
            # setup ... most of range is consistent with either, so If consistent, weight as zero (adds nothing)
            # if INCONSISTENT, weight in favor of the other solution
            #


            try: #similar to what follows, if the size in psf is not at least 1 (then it is smaller than the psf and the
                 # size cannot be (properly) determined (limited by psf)
                 # setting minimum size in PSF to be 1.2x PSF as "resolved" ... unresolved does not get a vote
                 #todo: except think about this ... if unresolved and PSF is small, then should that get a
                 # vote if everything under that size favors one object classification?
                if isinstance(self.classification_dict['size_in_psf'],float) and \
                     (self.classification_dict['size_in_psf'] > 1.2) and \
                     isinstance(self.classification_dict['diam_in_arcsec'],float):

                    # if 'base_psf' in self.classification_dict.keys() and isinstance(self.classification_dict['base_psf'],float):
                    #     base_psf = self.classification_dict['base_psf']
                    # else:
                    #     base_psf = 99.9
                    #
                    # if self.classification_dict['size_in_psf'] < 1.2: #give a little slop
                    #     unresolved = True
                    # else:
                    #     unresolved = False

                    #get all z
                    lae_z = [self.w / G.LyA_rest - 1.0]
                    if self.w >= G.OII_rest:
                        not_lae_z = [self.w / G.OII_rest - 1.0]
                    else:
                        not_lae_z = []

                    if (self.spec_obj is not None) and (self.spec_obj.solutions is not None):
                        for s in self.spec_obj.solutions:
                            if s.z > 1.8:  # suggesting LAE consistent
                                lae_z.append(s.z)
                            else:
                                not_lae_z.append(s.z)


                    for z in lae_z:
                        if z in [x['z'] for x in diameter_lae]:
                            #no need to add it again, but might re-evaluate the weight
                            continue

                        diam = SU.physical_diameter(z ,self.classification_dict['diam_in_arcsec'])
                        # max_diam =  SU.physical_diameter(z ,base_psf) #diameter assuming really unresolved and we are using the PSF

                        if diam is not None and diam > 0:
                            #todo: set the likelihood (need distro of sizes)
                            #typical half-light radius of order 1kpc (so full diamter something like 4-8 kpc and up)
                            #if AGN maybe up to 30-40kpc
                            if diam > 40.0:  #just too big, favor not LAE (unless QSO/AGN)
                                lk = 0.1 #could still be though ....
                                w = 1.0
                            # elif 30.0 < diam <= 40:
                            #     #pretty big, favors OII at lower-z, but not very definitive
                            #     #could maybe get some QSO in here?
                            #     lk = 0.0
                            #     w = 0.5
                            # elif 25.0 < diam <= 30:
                            #     #pretty big, maybe favors OII at lower-z, but not very definitive
                            #     #could get some QSO in here (brightness/width would help over rule this)
                            #     lk = 0.0
                            #     w = 0.1
                            elif 25 < diam <= 40:
                                lk = -2./75 * diam + 7./6. #from 0.5 to .1 linearly
                                #w = 2./75. * diam - 17./30. #scaled from 0.1 to 0.5 linearly from diam 25 to 40
                                w = 3./50. * diam - 7./5. #scaled from 0.1 to 1.0 linearly from diam 25 to 40
                            elif 20.0 < diam <= 25:
                                #pretty big, maybe favors OII at lower-z, but not very definitive
                                #could get some QSO in here (brightness/width would help over rule this)
                                lk = 0.5
                                w = 0.0
                            elif 15.0 < diam <= 20:
                                #not very useful
                                lk = 1.0
                                w = 0.0
                            elif 7.0 < diam <=15.0: #in the range of typical seeing FWHM from the ground
                                lk = 1.0
                                w = 0.1 #does not add much info, but is consistent
                            elif 3.0 < diam <= 7.0:
                                lk = 1.0
                                w = 0.25
                            else: #very small, highly consistent with LAE (small boost)
                                lk = 1.0
                                w = 0.5

                            likelihood.append(lk)
                            weight.append(w)
                            var.append(1)
                            prior.append(base_assumption)

                            #set a base weight (will be adjusted later)
                            diameter_lae.append({"z":z,"kpc":diam,"weight":w,"likelihood":lk})
                            log.info(
                                 f"{self.entry_id} Aggregate Classification, added physical size:"
                                 f" z({z:#.4g}) kpc({diam:#.4g}) weight({w:#.5g}) likelihood({lk:#.5g})")

                    for z in not_lae_z:
                        if z in [x['z'] for x in diameter_not_lae]:
                            #no need to add it again, but might re-evaluate the weight
                            continue

                        diam = SU.physical_diameter(z, self.classification_dict['diam_in_arcsec'])
                        # max_diam =  SU.physical_diameter(z ,base_psf) #diameter assuming really unresolved and we are using the PSF

                        if diam is not None and diam > 0:
                            # todo: set the likelihood (need distro of sizes)
                            if diam < 0.1:  #just too small, favor LAE
                                lk = 1.0 # ***(this is likelihood of LAE, so 1-likelihood not LAE)
                                #so saying here that is very UNLIKELY this is a low-z object based on small physical size
                                #so it is more LIKELY it is a high-z object
                                w = 0.75
                            #could be planetary nebula, etc (very small)
                            # elif diam > 10.0:
                            #     lk = 0
                            #     w = 0.5
                            # elif diam > 2.0:
                            #     lk = 0.0
                            #     w = 0.25
                            # elif diam > 1.0:
                            #     lk = 0.0
                            #     w = 0.1
                            elif diam > 2.0: #start walking up as a likely vote for nearby, starting at 2kpc up to 30kpc
                                  #vote is for OII, so  0.0
                                  #weight creeps up
                                lk = 0.0
                                w = min(0.5,diam/30.0)
                            else: #no info, favors either
                                lk = 0.5
                                w = 0.0
                            #set a base weight (will be adjusted later)

                            likelihood.append(lk)
                            weight.append(w)
                            var.append(1)
                            prior.append(base_assumption)
                            diameter_not_lae.append({"z":z,"kpc":diam,"weight":w,"likelihood":lk})
                            log.info(
                                f"{self.entry_id} Aggregate Classification, added physical size:"
                                f" z({z:#.4g}) kpc({diam:#.4g}) weight({w:#.5g}) likelihood({lk:#.5g})")


                    #weights and likihoods updated below with additional info
                     #then a winner in each class is chosen near the end
            except:
                log.debug(f"{self.entry_id} Aggregate Classification physical size exception.",exc_info=True)


        #
        #object extent in terms of approximate PSF ... if very inconsistent with PSF, then not likely to be LAE
        # Mostly a BOOLEAN value (yes or no, LAE)
        if False: #todo: discontinue this? now replaced with physical size
            try:
                #basiclly, 0 to 0.5 (if size > 5x PSF, probability that is LAE --> 0, if < 2x PSF holds at 0.5)
                scale = 0.5 #no info ... 50/50 chance of LAE
                consistent_with_lae_psf = 1.5
                inconsistent_with_lae_psf = 5.0
                if (self.classification_dict['size_in_psf'] is not None) and \
                        (self.classification_dict['size_in_psf'] > consistent_with_lae_psf): #greater than twice a "sloppy" PSF, inconsistent with point source
                    scale = scale * (1.0 - (self.classification_dict['size_in_psf'] - consistent_with_lae_psf) /
                                            (inconsistent_with_lae_psf-consistent_with_lae_psf))
                    if scale < 0.01: #completely inconsistent with point source
                        scale = 0.01
                        likelihood.append(scale)
                        weight.append(0.5) #opinion ... if not consistent with point-source, very unlikley to be LAE
                        var.append(1) #no variance to use
                        prior.append(base_assumption)
                        log.debug(f"{self.entry_id} Aggregate Classification: PSF scale inconsistent with point source: lk({likelihood[-1]}) weight({weight[-1]})")
                    elif scale > 0.5: #completely consistent with point source
                        scale = 0.5 #max at 0.5 (basically no info ... as likely to be LAE as not)
                        likelihood.append(scale)
                        weight.append(0.01)  # opinion ... if consistent with point-source, does not really mean anything
                        var.append(1.)  # no variance to use
                        prior.append(base_assumption)
                        log.info(
                            f"{self.entry_id} Aggregate Classification: PSF scale fully consistent with point source: lk({likelihood[-1]}) weight({weight[-1]})")
                    else: #scale is between 0.01 and 0.5,
                        var.append(1.)
                        likelihood.append(scale)
                        weight.append(0.5-scale)
                        prior.append(base_assumption)
                        log.info(
                            f"{self.entry_id} Aggregate Classification: PSF scale marginally inconsistent with point source: lk({likelihood[-1]}) weight({weight[-1]})")
                else:
                    #really adds no information ... as likely to be OII as LAE if consistent with point-source
                    pass
                    # likelihood.append(0.5)
                    # var.append(1.)
                    # weight.append(0.1)
                    # prior.append(base_assumption)
            except:
                log.debug("Exception in aggregate_classification for size_in_psf",exc_info=True)

        #
        # Elixer solution finder
        # not just a best solution, but look at all (could be, say, both OIII (5007) and HBeta are really high, so
        # can't distinguish between them, but in either case, is very inconsistent with LAE
        # Mostly a BOOLEAN value (yes or no, LAE)
        try:
            if (self.spec_obj is not None) and (self.spec_obj.solutions is not None):
                for s in self.spec_obj.solutions:
                    bonus_weight = 1.0 #multiplier
                    #if this is a really high score and there are 2 or more additional lines (3+ total with the main line)
                    #or a slightly lower score and more lines, then
                    # this basically wins (boost the weight way up)
                    if (s.score / G.MULTILINE_FULL_SOLUTION_SCORE > 8 and (len(s.lines) > 1)) or \
                       (s.score / G.MULTILINE_FULL_SOLUTION_SCORE > 4 and (len(s.lines) > 2)) or \
                       (G.MULTILINE_USE_CONSISTENCY_CHECKS and (s.score / G.MULTILINE_FULL_SOLUTION_SCORE > 2) and (s.frac_score > 0.75)):# and s):
                        bonus_weight = min(s.score / G.MULTILINE_FULL_SOLUTION_SCORE,10.0)  #up to 10x bonus

                    #and an additional bonus IF in a galaxy mask
                    #smallest non-zero value is 1.0 (unless an oddly small D25 scale is set in global_config)
                    #the smaller the number, the larger the bonus
                    #DD: 2021-02-24 ... moved to the check_transients_and_flags logic and boost to the score there
                    # if s.galaxy_mask_d25 and (s.galaxy_mask_d25 > 0):
                    #     bonus_weight += 2.0/s.galaxy_mask_d25

                    if s.score >= G.MULTILINE_MIN_SOLUTION_SCORE: #only consider somewhat probable scores
                        #split between z > 1.8 ==> LAE and < 1.8 ==>not LAE
                        #if s.z  > 1.8: #suggesting LAE consistent
                        if s.central_rest == G.LyA_rest:
                            likelihood.append(1.0) #s.scale_score)
                            weight.append(s.scale_score * bonus_weight)#0.8)  # opinion ... has multiple lines, so the score is reasonable
                            #must also have CIV or HeII, etc as possible matches
                            var.append(1)  #todo: ? could do something like the spectrum noise?
                            prior.append(base_assumption)
                            log.info(
                                f"{self.entry_id} Aggregate Classification: high-z solution: z({s.z}) lk({likelihood[-1]}) "
                                f"weight({weight[-1]}) score({s.score}) scaled score({s.scale_score})")

                        else: #suggesting NOT LAE consistent
                            likelihood.append(0.0) #1-s.scale_score)
                            weight.append(s.scale_score * bonus_weight) #1.0)  # opinion ... has multiple lines, so the score is reasonable
                            var.append(1)  #todo: ? could do something like the spectrum noise?
                            prior.append(base_assumption)
                            log.info(
                                f"{self.entry_id} Aggregate Classification: low-z solution: z({s.z}) lk({likelihood[-1]}) "
                                f"weight({weight[-1]}) score({s.score}) scaled score({s.scale_score})")
                    else: #low score, but can still impact
                        #this can be a problem for items like 1000637691 which get reduced score, but is clearly a non-LAE multi-line
                        #like an HII region; in theory clustering of emission lines would catch this

                        #w = s.score / G.MULTILINE_MIN_SOLUTION_SCORE * s.scale_score
                        #sigmoid x scale_score
                        w = 1.0 / (1.0 + np.exp(-25.0 * (s.score/25.0 - .75))) * s.scale_score

                        #w = 0.5 * (max(s.scale_score -0.5, 0)) #min(s.score / G.MULTILINE_MIN_SOLUTION_SCORE, s.scale_score)

                        if s.central_rest == G.LyA_rest:
                        #if s.z > 1.8:  # suggesting LAE consistent
                           # likelihood.append(s.scale_score)
                           # weight.append(0.8 * w)  # opinion ... has multiple lines, so the score is reasonable
                            likelihood.append(1.0)  # s.scale_score)
                            weight.append(w * bonus_weight)
                            # must also have CIV or HeII, etc as possible matches
                            var.append(1)  # todo: ? could do something like the spectrum noise?
                            prior.append(base_assumption)
                            log.info(
                                f"{self.entry_id} Aggregate Classification: high-z weak solution: z({s.z}) lk({likelihood[-1]}) "
                                f"weight({weight[-1]}) score({s.score}) scaled score({s.scale_score})")
                        else:  # suggesting NOT LAE consistent
                            #likelihood.append(1. - s.scale_score)
                            #weight.append(1.0 * w)  # opinion ... has multiple lines, so the score is reasonable
                            likelihood.append(0.0)  # weak solution so push likelihood "down" but not zero (maybe 0.2 or 0.25)?
                            weight.append(w * bonus_weight)
                            var.append(1)  # todo: ? could do something like the spectrum noise?
                            prior.append(base_assumption)
                            log.info(
                                f"{self.entry_id} Aggregate Classification: low-z weak solution: z({s.z}) lk({likelihood[-1]}) "
                                f"weight({weight[-1]}) score({s.score}) scaled score({s.scale_score})")

                    # does this match with a physical size from above?
                    try:
                        idx = np.where([x['z'] for x in diameter_lae] == s.z)[0]  # should be exactly one
                        if (idx is not None) and (len(idx) == 1):
                            idx = idx[0]
                            diameter_lae[idx]['weight'] = weight[-1]
                            log.info(
                                f"{self.entry_id} Aggregate Classification, updated physical size: "
                                f"z({diameter_lae[idx]['z']:#.4g}) "
                                f"kpc({diameter_lae[idx]['kpc']:#.4g}) "
                                f"weight({diameter_lae[idx]['weight']:#.2g}) "
                                f"likelihood({diameter_lae[idx]['likelihood']:#.2g})")
                    except:
                        log.debug(f"{self.entry_id} Aggregate Classification exception", exc_info=True)

        except:
            log.debug("Exception in aggregate_classification for ELiXer Combine ALL Continuumsolution finder",exc_info=True)


        #unmatched solutions scoring (basically any lines other than LyA)
        try:
            #should only (and always) be exactly one
            try:
                lya_sol = self.spec_obj.solutions[np.where(np.array(
                    [x.central_rest for x in self.spec_obj.solutions]) == G.LyA_rest)[0][0]]

                if (lya_sol.unmatched_lines_score > G.MAX_OK_UNMATCHED_LINES_SCORE) \
                    and (lya_sol.unmatched_lines_count > G.MAX_OK_UNMATCHED_LINES):
                    #there are significant unaccounted for lines ... this pushes DOWN the possibility that this is LAE
                    #but does not impact Non-LAE solutions
                    var.append(1.)
                    likelihood.append(0.0) #so, NOT LAE
                    weight.append(min(1.0,lya_sol.unmatched_lines_score/G.MULTILINE_FULL_SOLUTION_SCORE)) #up to 1.0
                    prior.append(base_assumption)
                    log.info(
                        f"{self.entry_id} Aggregate Classification: Significant unmatched lines penalize LAE: lk({likelihood[-1]})"
                        f" weight({weight[-1]})")
            except:
                try:
                    if (self.spec_obj is not None) and \
                            (self.spec_obj.unmatched_solution_score > G.MAX_OK_UNMATCHED_LINES_SCORE) \
                            and (self.spec_obj.unmatched_solution_count > G.MAX_OK_UNMATCHED_LINES):
                        # there are significant unaccounted for lines ... this pushes DOWN the possibility that this is LAE
                        # but does not impact Non-LAE solutions
                        # (assumes LyA is the only line ... if CIV or other detected, there will be a bonus from the above code
                        var.append(1.)
                        likelihood.append(0.0)  # so, NOT LAE
                        weight.append(min(1.0,
                                          self.spec_obj.unmatched_solution_score / G.MULTILINE_FULL_SOLUTION_SCORE))  # up to 1.0
                        prior.append(base_assumption)
                        log.info(
                            f"{self.entry_id} Aggregate Classification: Significant unmatched lines penalize LAE: lk({likelihood[-1]})"
                            f" weight({weight[-1]})")

                except:
                    log.debug("Exception in aggregate_classification", exc_info=True)
        except:
            log.debug("Exception in aggregate_classification",exc_info=True)

        ############################################
        # best PLAE/POII
        #
        # Mostly a distribution value
        ###########################################
        try:
            if (self.classification_dict['plae_hat'] is not None) and (self.classification_dict['plae_hat'] != -1) \
            and (self.classification_dict['plae_hat_sd'] is not None):
                #scale ranges from 0.999 (LAE) to 0.001 (not LAE)
                #logic is simple based on PLAE/POII interpreations to mean #LAE/(#LAE + #OII) where #LAE is a fraction and #OII == 1
                #so PLAE/POII = 1000 --> 1000/(1000+1) = 0.999, PLAE/POII == 1.0 --> (1/(1+1)) = 0.5, PLAE/POII = 0.001 --> 0.001/(0.001 +1) = 0.001
                #2022-01-27 ... DD let the weight handle all the uncertainty. Any vote that is near the midpoint will
                #get close to a zero weight anyway, so just make the vote purely binary
                # mid = self.classification_dict['plae_hat'] / plae_poii_midpoint(self.w)
                # plae_vote = mid / (mid + 1.0)
                #

                midpoint = plae_poii_midpoint(self.w)
                if self.classification_dict['plae_hat'] > midpoint:
                    plae_vote = 1.0
                else:
                    plae_vote = 0.0

                vote_info['plae_poii_combined_midpoint'] = midpoint
                vote_info['plae_poii_combined'] = self.classification_dict['plae_hat']
                vote_info['plae_poii_combined_hi'] = self.classification_dict['plae_hat_hi']
                vote_info['plae_poii_combined_lo'] = self.classification_dict['plae_hat_lo']

                # lower_plae = max(0.001, self.classification_dict['plae_hat_lo'])#self.classification_dict['plae_hat']-self.classification_dict['plae_hat_sd'])
                # scale_plae_lo =  scale_plae_hat - lower_plae / (lower_plae + 1.0)
                #
                # higher_plae = min(1000.0, self.classification_dict['plae_hat_hi'])#self.classification_dict['plae_hat']-self.classification_dict['plae_hat_sd'])
                # scale_plae_hi = higher_plae / (higher_plae + 1.0) - scale_plae_hat
                # scale_plae_sd = 0.5 * (scale_plae_hi + scale_plae_lo)

                scale_plae_lo = self.classification_dict['plae_hat_lo'] / (self.classification_dict['plae_hat_lo'] + 1.0)
                scale_plae_hi = self.classification_dict['plae_hat_hi'] / (self.classification_dict['plae_hat_hi'] + 1.0)
                scale_plae_sd = 0.5 * (scale_plae_hi - scale_plae_lo)

                likelihood.append(plae_vote)
                prior.append(base_assumption)


                #scale the weight by the difference between the scaled PLAE and one SD below (or above)
                # the closer they are to each other, the closer to the full weight you'd get)
                #weight.append(0.7 * (1.0 - scale_plae_sd))  # opinion, not quite as strong as multiple lines

                # the plae_gaussian_weight scales the weighting to zero at plae_hat == 1 (50/50 ... so no info)
                # and increases the weight as you move toward 0.001 or 1000 over a gaussian (though max weight
                # of 1.0 is hit around 0.05 or 20.0
                # At the end, the insertion of a 0.5 vote with (1-sum(weights)) should not matter
                # as, if this is sitting near a zero weight and is the only vote, that means we are sitting
                # near PLAE/POII ~ 1 which is 0.5 P(LyA)
                w = plae_gaussian_weight(self.classification_dict['plae_hat'],obs_wave=self.w) * (1.0 - scale_plae_sd)


                try:
                    #if this is on the +1 side (not the 0.001 side), and if the stdev is high, larger than the actual
                    #estimated value, then beyond just the 68% conf interval, this is highly uncertain and we should
                    #reduce the voting weight
                    if plae_vote > 0.5 and self.classification_dict['plae_hat'] /  self.classification_dict['plae_hat_sd'] < 1.0:
                        w *= (self.classification_dict['plae_hat'] /  self.classification_dict['plae_hat_sd'])**2
                # if if plae_vote > 0.5: #for LyA
                #     if self.classification_dict['plae_hat'] - self.classification_dict['plae_hat_sd'] < midpoint:
                #         #high uncertainty; not just 68% confidence
                #         #reduce the weight
                #         w *=
                except:
                    pass
                weight.append(w)
                var.append(1)  # todo: use the sd (scaled?) #can't use straight up here since the variances are not
                               # on the same scale

                vote_info['plae_poii_combined_vote'] = likelihood[-1]
                vote_info['plae_poii_combined_weight'] = weight[-1]

                log.info( f"{self.entry_id} Aggregate Classification: MC PLAE/POII from combined continuum: "
                          f"lk({likelihood[-1]}) weight({weight[-1]})")


                #sanity check vs 20AA cut
                #appears only to favor LAE for low EW and low z, never appears to favor OII at higher EW
                try:
                    if self.w > (G.OII_rest-1.0):
                        #start with the line flux
                        if self.spec_obj:
                            ew_combined_continuum = self.spec_obj.estflux
                        else:
                            ew_combined_continuum = self.estflux #the lineflux

                        zp1 = self.w/G.LyA_rest #z + 1
                        #then divide by the averaged continuum
                        ew_combined_continuum /= self.classification_dict['continuum_hat']
                        #then to the LyA restframe
                        ew_combined_continuum /= zp1

                        #very rough, not following any actual distribution right now
                        #this is meant as a flag or warning if the PLAE/POII seems to have failed expectations
                        #lae_z < 2.4  is oii_z < 0.1
                        if ew_combined_continuum and ( 0 < ew_combined_continuum < 20) and (zp1 < 3.4) and \
                                (self.classification_dict['plae_hat'] > 3.0):
                            #this is unexpected
                            log.info(f"Unexpected PLAE/POII {self.classification_dict['plae_hat']} for low-z OII {G.LyA_rest*zp1/G.OII_rest - 1.0} and small EW {ew_combined_continuum}. Applying 20AA sanity check.")
                            #here we are only tweaking the weight for the NOT LyA binary condition vote of 0.0
                            #where the weight is 0 at EW == 20 and falls to 1.0 by EW == 10

                            likelihood.append(0.0)
                            prior.append(base_assumption)
                            weight.append(min(2.0,20.0/ew_combined_continuum - 1.0))
                            var.append(1)
                            log.info(f"{self.entry_id} Aggregate Classification: 20AA sanity PLAE/POII from combined continuum: "
                                     f"lk({likelihood[-1]}) weight({weight[-1]})")


                        if False:
                            #if this object is moderately bright and the EW is well above or below 20AA, give that hard EW
                            #cut its own vote .... using ONLY the best_g mag
                            if self.best_gmag is not None and self.best_gmag < G.HETDEX_CONTINUUM_MAG_LIMIT:
                                if ew_combined_continuum < 15.0:
                                    likelihood.append(0.0)
                                    prior.append(base_assumption)
                                    weight.append(0.5) #todo make weight depend on magnitude and EW ...
                                    var.append(0.5)
                                    log.info(f"{self.entry_id} Aggregate Classification: 20AA hard cut vote from best g-mag and combined EW ({ew_combined_continuum:0.1f}): "
                                             f"lk({likelihood[-1]}) weight({weight[-1]})")
                                elif ew_combined_continuum > 25.0:
                                    likelihood.append(1.0)
                                    prior.append(base_assumption)
                                    weight.append(0.5)
                                    var.append(1)
                                    log.info(f"{self.entry_id} Aggregate Classification: 20AA hard cut vote from best g-mag and combined EW ({ew_combined_continuum:0.1f}): "
                                                 f"lk({likelihood[-1]}) weight({weight[-1]})")


                except:
                    log.warning("Exception appling 20AA sanity check",exc_info=True)

                #todo: handle the uncertainty on plae_hat
                #plae_hat_sd = self.classification_dict['plae_hat_sd']
        except:
            log.debug("Exception in aggregate_classification for best PLAE/POII",exc_info=True)


        try:
            lower_mag = self.best_gmag + self.best_gmag_unc #want best value + the error (so on the fainter side)
        except:
            lower_mag  = 99


        #################################################
        # Asymmetric Flux vote
        # compare blue side to red side of line center
        # (maybe marginally effective? ... resolution not high enough)
        #################################################
        try:
            #can fail if too close to the edge, in which case, this should not vote anyway
            #assuming no errors or similar errors on red side and blue side
            #this fails if LyA blue is really strong (high escape fraction)
            #really want to check just right near the line and want at least 3 wavebins
            #if self.snr is not None and self.snr > 6.0 and self.fwhm > 8.0:
            if True: #for now always do this as I want the info, but only vote if the condition is met

                line_width = max(3,round(self.fwhm /2.355/G.FLUX_WAVEBIN_WIDTH))

                line_center_idx,*_ = SU.getnearpos(self.sumspec_wavelength,self.w) #get closest wavebin "center"
                left_edge = self.sumspec_wavelength[line_center_idx] - 1.0
                center_blue_frac = (self.w-left_edge)/G.FLUX_WAVEBIN_WIDTH
                centerflux = self.sumspec_flux[line_center_idx]
                centerflux_err = self.sumspec_fluxerr[line_center_idx]

                lineflux_red = np.sum(self.sumspec_flux[line_center_idx+1:line_center_idx+2+line_width]) + centerflux * (1-center_blue_frac)
                lineflux_blue = np.sum(self.sumspec_flux[line_center_idx-1-line_width:line_center_idx]) + centerflux * center_blue_frac

                #note: SUPER unlikely, but we could get a zero flux, and a divide by zero, but that will trap in
                #try/except and is not worth worrying about

                #what about uncertainty??
                lineflux_red_err = np.sqrt(np.sum(self.sumspec_fluxerr[line_center_idx+1:line_center_idx+2+line_width]**2) +
                                   (centerflux_err * (1-center_blue_frac))**2)
                lineflux_blue_err = np.sqrt(np.sum(self.sumspec_fluxerr[line_center_idx-1-line_width:line_center_idx]**2) +
                                    (centerflux_err * center_blue_frac)**2)


                rat = lineflux_red/lineflux_blue
                rat_err = rat * np.sqrt((lineflux_red_err/lineflux_red)**2 + (lineflux_blue_err/lineflux_blue)**2)
                did_vote = True

                vote_info['rb_flux_asym'] = rat
                vote_info['rb_flux_asym_err'] = rat_err

                if self.snr is not None and self.snr > 6.0 and self.fwhm > 8.0:
                    if rat_err/rat > 0.5 and ((rat-rat_err) < 1.0 and (rat+rat_err) > 1.0):
                        log.info(f"{self.entry_id} Aggregate Classification: asymmetric line flux (r/b) {rat:0.2f}  +/- {rat_err:0.3f} no vote.")
                    # elif rat > 1.33:
                    #     likelihood.append(1.0)
                    #     weight.append(0.25)
                    #     prior.append(base_assumption)
                    #     var.append(1)
                    elif rat > 1.2: #seems to be pretty good separation above 1.2
                        likelihood.append(1.0)
                        weight.append(0.25)
                        prior.append(base_assumption)
                        var.append(1)
                    #from data, looks like we more blue than red is possible even for LyA
                    # elif rat < 0.70:
                    #     likelihood.append(0.0)
                    #     weight.append(0.25)
                    #     prior.append(base_assumption)
                    #     var.append(1)
                    # elif rat < 0.80:
                    #     likelihood.append(0.0)
                    #     weight.append(0.1)
                    #     prior.append(base_assumption)
                    #     var.append(1)

                    else:
                        did_vote = False
                else:
                    did_vote = False

                if did_vote:
                    vote_info['rb_flux_asym_vote'] = likelihood[-1]
                    vote_info['rb_flux_asym_weight'] = weight[-1]
                    log.info(f"{self.entry_id} Aggregate Classification: asymmetric line flux (r/b) {rat:0.2f} +/- {rat_err:0.3f} vote: "
                             f"lk({likelihood[-1]}) weight({weight[-1]})")
                else:
                    log.info(f"{self.entry_id} Aggregate Classification: asymmetric line flux (r/b) {rat:0.2f}  +/- {rat_err:0.3f} no vote.")
            else:
                log.info(f"{self.entry_id} Aggregate Classification: asymmetric line flux low snr {self.snr:0.2f} or fwhm {self.fwhm:0.2f} no vote.")
        except:
            log.debug("Exception in r/b line asymmetry vote.",exc_info=True)


        #################################################
        # line FWHM vote (really broad is not likely OII)
        # max 0.5 weight
        #################################################
        try: #sometimes there is no central_eli (if we just can't get a fit)

            if self.fwhm is not None:
                vote_line_sigma = self.fwhm / 2.355
                vote_line_sigma_unc = self.fwhm_unc / 2.355
            else:
                vote_line_sigma = self.spec_obj.central_eli.fit_sigma
                vote_line_sigma_unc = self.spec_obj.central_eli.fit_sigma_err

            vote_info['line_sigma'] = vote_line_sigma
            vote_info['line_sigma_err'] = vote_line_sigma_unc

            if (vote_line_sigma - vote_line_sigma_unc) > G.LINEWIDTH_SIGMA_TRANSITION: #unlikely OII (FHWM 16.5)
                likelihood.append(1) #vote kind of FOR LyA (though could be CIV, MgII, other)
                weight.append(min(vote_line_sigma / G.LINEWIDTH_SIGMA_TRANSITION - 1.0, 0.5)) #limit to 0.5 max
                var.append(1)
                prior.append(base_assumption)
                vote_info['line_sigma_vote'] = likelihood[-1]
                vote_info['line_sigma_weight'] = weight[-1]
                log.info(f"{self.entry_id} Aggregate Classification: line FWHM vote (not OII): lk({likelihood[-1]}) weight({weight[-1]})")
        except:
            pass


        ###################################
        # Straight EqWidth vote
        # does NOT consider OII or LyA distros
        ###################################
        try:

            delta_thresh = 0
            rat_thresh = 0

            vote_info['ew_rest_lya_combined'] = self.classification_dict['combined_eqw_rest_lya']
            vote_info['ew_rest_lya_combined_err'] = self.classification_dict['combined_eqw_rest_lya_err']

            if self.classification_dict['combined_eqw_rest_lya'] > 20.:
                #will be an LAE vote
                likelihood.append(1)
                try:
                    #delta_thresh = abs(self.classification_dict['combined_eqw_rest_lya'] - self.classification_dict['combined_eqw_rest_lya_err'] - 20.0)
                    rat_thresh = (self.classification_dict['combined_eqw_rest_lya'] - self.classification_dict['combined_eqw_rest_lya_err'])/20.0
                except:
                    pass
                #weight.append(max(0.1,min(0.5,delta_thresh*0.1))) #delta thresh
                #rat_thresh can be negative or less than one
                if rat_thresh < 1.0: #this is no-vote or minimum vote case
                    if rat_thresh < 0:
                        weight.append(0) #error is larger than the Ew
                    else:
                        #error pushes below 20AA, so minimum vote
                        weight.append(0.1)
                else:
                    weight.append(max(0.1,min(0.5,(rat_thresh-1.0)))) #rat thresh
            else:
                #will be an OII vote
                likelihood.append(0)
                try:
                    #delta_thresh = abs(self.classification_dict['combined_eqw_rest_lya'] + self.classification_dict['combined_eqw_rest_lya_err'] - 20.0)
                    rat_thresh = 20/(self.classification_dict['combined_eqw_rest_lya'] + self.classification_dict['combined_eqw_rest_lya_err'])
                except:
                    pass
                #since compress toward EW = 0, this has slightly higher scoring
                #plus OII can have large EW, but LyA below 20 is REALLY rare
                #weight.append(max(0.1,min(0.5,delta_thresh*0.25))) #delta thresh version
                #rat_thresh cannot be negative, but can be less than one
                if rat_thresh < 1.0: #this is no-vote or minimum vote case
                    #error pushes below 20AA, so minimum vote
                    weight.append(0.1)
                else:
                    weight.append(max(0.1,min(0.5,(rat_thresh-1.0)*0.5))) #rat thresh


            var.append(1)
            prior.append(base_assumption)
            vote_info['ew_rest_lya_combined_vote'] = likelihood[-1]
            vote_info['ew_rest_lya_combined_weight'] = weight[-1]
            log.info(f"{self.entry_id} Aggregate Classification: straight combined line EW "
                     f"{self.classification_dict['combined_eqw_rest_lya']} +/- {self.classification_dict['combined_eqw_rest_lya_err']} vote: "
                     f"lk({likelihood[-1]}) weight({weight[-1]})")
        except:
            pass


        ###################################
        # Phot-z vote
        # max 0.5 vote
        ###################################
        phot_z = self.get_phot_z_vote()
        if -0.1 < phot_z < 0.7:
            likelihood.append(0.0)
            weight.append(0.5)
            var.append(1)
            prior.append(base_assumption)
            log.info(f"{self.entry_id} Aggregate Classification: phot_z vote (low-z): lk({likelihood[-1]}) weight({weight[-1]})")
        elif 1.7 < phot_z < 3.7:
            likelihood.append(1.0)
            weight.append(0.5)
            var.append(1)
            prior.append(base_assumption)
            log.info(f"{self.entry_id} Aggregate Classification: phot_z vote (high-z): lk({likelihood[-1]}) weight({weight[-1]})")
        else:
            log.info(f"{self.entry_id} Aggregate Classification: phot_z vote - no vote.")



        #########################################
        # Straight DEX g-mag vote
        # simple vote based only on the HETDEX g-mag
        # low weight
        #########################################

        if True:
            try:
                if ('combined_eqw_rest_lya' in self.classification_dict) and \
                        self.classification_dict['combined_eqw_rest_lya'] is not None:
                    ew = self.classification_dict['combined_eqw_rest_lya']
                else:
                    ew = None

                if ('combined_eqw_rest_lya_err' in self.classification_dict) and \
                            self.classification_dict['combined_eqw_rest_lya_err'] is not None:
                    ew_err = self.classification_dict['combined_eqw_rest_lya_err']
                else:
                    ew_err = 0

                if self.best_gmag is not None and self.w > G.OII_rest:
                    g = min(self.best_gmag,G.HETDEX_CONTINUUM_MAG_LIMIT)

                    if self.best_gmag_unc is not None:
                        g_unc = self.best_gmag_unc
                    else:
                        g_unc = 0

                    g_bright = None
                    g_faint = None

                    gmag_bright_thresh, gmag_faint_thresh = gmag_vote_thresholds(self.w)
                    try:
                        if g == G.HETDEX_CONTINUUM_MAG_LIMIT:
                            g_bright = g
                        else:
                            g_bright = g - g_unc
                        g_faint = g + g_unc
                    except:
                        pass

                    g_str = f"{g:0.2f} ({g_faint:0.2f},{g_bright:0.2f})"
                    g_thresh_str = f"{gmag_faint_thresh:0.2f}-{gmag_bright_thresh:0.2f}"

                    vote_info['dex_gmag'] = g
                    vote_info['dex_gmag_bright'] = g_bright
                    vote_info['dex_gmag_faint'] = g_faint
                    vote_info['gmag_thresh_bright'] = gmag_bright_thresh
                    vote_info['gmag_thresh_faint'] = gmag_faint_thresh


                    if g_bright > gmag_faint_thresh: #vote for LyA
                        likelihood.append(1.0)
                        weight.append(0.5) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                        var.append(1)
                        prior.append(base_assumption)
                        vote_info['dex_gmag_vote'] = likelihood[-1]
                        vote_info['dex_gmag_weight'] = weight[-1]
                        log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                 f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                    elif g_faint < gmag_bright_thresh: #vote for OII
                        #Still up to 25-30% could be LyA ... so weigt very little
                        #check on the EW if solidly LyA then vote LyA, etc
                        if (ew is not None):
                            if (ew-ew_err) > 80:
                                #could be LyA
                                likelihood.append(1.0)
                                weight.append(0.25) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            elif (ew-ew_err) > 30:
                                #could be LyA
                                likelihood.append(1.0)
                                weight.append(0.1) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            elif (ew+ew_err) < 15:
                                #could be LyA
                                likelihood.append(0.0)
                                weight.append(0.5) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            else: #no vote
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag no vote. Mag in unclear region. mag ({g_str}), thresh ({g_thresh_str})")
                        else: #no EW info ... assume OII as more likely, but limited weight
                            likelihood.append(0.0)
                            weight.append(0.1)
                            var.append(1)
                            prior.append(base_assumption)
                            vote_info['dex_gmag_vote'] = likelihood[-1]
                            vote_info['dex_gmag_weight'] = weight[-1]
                            log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                     f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                    elif (g > gmag_faint_thresh and g_faint > gmag_bright_thresh) or \
                            ((gmag_bright_thresh < g < gmag_faint_thresh) and (g_bright < gmag_bright_thresh and g_faint < gmag_faint_thresh)): #small vote for LyA
                        #error straddles no-man's land and fainter limit
                        if (ew is not None):
                            if (ew-ew_err) > 80:
                                #could be LyA
                                likelihood.append(1.0)
                                weight.append(0.5) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            elif (ew-ew_err) > 30:
                                #could be LyA
                                likelihood.append(1.0)
                                weight.append(0.3) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            elif (ew+ew_err) < 15:
                                #could be LyA
                                likelihood.append(0.0)
                                weight.append(0.25) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            else: #no vote
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag no vote. Mag in unclear region. mag ({g_str}), thresh ({g_thresh_str})"
                                         f" with unclear EW: {ew:0.1f} +/- {ew_err:0.1f}")
                        else: #no EW info ... assume OII as more likely, but limited weight
                            likelihood.append(1.0)
                            weight.append(0.1)
                            var.append(1)
                            prior.append(base_assumption)
                            vote_info['dex_gmag_vote'] = likelihood[-1]
                            vote_info['dex_gmag_weight'] = weight[-1]
                            log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                     f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                    elif (g < gmag_bright_thresh and g_bright < gmag_faint_thresh) or \
                        ((gmag_bright_thresh < g < gmag_faint_thresh) and (g_bright < gmag_bright_thresh and g_faint < gmag_faint_thresh)):
                        #small vote for OII
                        #error straddles no-man's land and brighter limit where the fiducial g is on the bright end or in between
                        if (ew is not None):
                            if (ew-ew_err) > 80:
                                #could be LyA
                                likelihood.append(1.0)
                                weight.append(0.30) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            elif (ew-ew_err) > 30:
                                #could be LyA
                                likelihood.append(1.0)
                                weight.append(0.15) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            elif (ew+ew_err) < 15:
                                #could be LyA
                                likelihood.append(0.0)
                                weight.append(0.4) #this COULD become more of a ratio between #LyA / #OII at this magbin and wavebin
                                var.append(1)
                                prior.append(base_assumption)
                                vote_info['dex_gmag_vote'] = likelihood[-1]
                                vote_info['dex_gmag_weight'] = weight[-1]
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                         f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")
                            else: #no vote
                                log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag no vote. Mag in unclear region. mag ({g_str}), thresh ({g_thresh_str})"
                                         f" with unclear EW: {ew:0.1f} +/- {ew_err:0.1f}")
                        else: #no EW info ... assume OII as more likely, but limited weight
                            likelihood.append(0.0)
                            weight.append(0.05)
                            var.append(1)
                            prior.append(base_assumption)
                            vote_info['dex_gmag_vote'] = likelihood[-1]
                            vote_info['dex_gmag_weight'] = weight[-1]
                            log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                     f"weight({weight[-1]}): mag ({g_str}), thresh ({g_thresh_str})")

                    else: #g is in between OR error straddles both ... either way, no vote
                        log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag no vote. Mag in unclear region. mag ({g_str}), thresh ({g_thresh_str})")

            except:
                pass




        ###################################
        # general magnitude + EW vote
        # requires an EW outside of a range
        # can see really bright (22 mag "normal" LAE, and faint 26+ "normal" OII, but are rare)
        ##################################
        if False:
            if self.w > G.OII_rest:
                mg_z = self.w/G.LyA_rest -1 #redshift used to adjust the mag zeros
                #counterpart magnitude (if a counterpart was automatically identified)
                #see cat_base::build_cat_summary_pdf_section

                #we want to consider these when the equivalent width is near 20AA

                counterpart_filter = None
                if self.best_counterpart is not None and self.best_counterpart.bid_filter is not None:
                    if self.best_counterpart.bid_filter.lower() in ['r','f606w']:
                        mag_zero = G.LAE_R_MAG_ZERO
                        counterpart_filter = 'r'
                    else:
                        mag_zero = G.LAE_G_MAG_ZERO
                        counterpart_filter = 'g'

                    mag_zero = adjusted_mag_zero(mag_zero, mg_z)

                    ew = 999
                    unc = 0
                    if self.best_counterpart.bid_ew_lya_rest is not None:
                        ew = self.best_counterpart.bid_ew_lya_rest
                        if self.best_counterpart.bid_ew_lya_rest_err is not None:
                            unc =  self.best_counterpart.bid_ew_lya_rest_err

                    if (ew-unc) < G.LAE_EW_MAG_TRIGGER_MAX and (ew+unc) > G.LAE_EW_MAG_TRIGGER_MIN:
                        w = 0.5 * mag_gaussian_weight(mag_zero,self.best_counterpart.bid_mag,
                                                      self.best_counterpart.bid_mag_err_bright,self.best_counterpart.bid_mag_err_faint)

                        if self.best_counterpart.bid_mag < mag_zero:
                            likelihood.append(0.0)
                        else:
                            likelihood.append(1.0)

                        weight.append(w)
                        var.append(1)
                        prior.append(base_assumption)
                        log.info(f"{self.entry_id} Aggregate Classification: counterpart {self.best_counterpart.bid_filter.lower()}-mag vote "
                                 f"{self.best_counterpart.bid_mag:0.2f} : lk({likelihood[-1]}) "
                                 f"weight({weight[-1]})")
                    else:
                        counterpart_filter = None #undo the filter vote status

                #partly included in PLAE/POII (as continuum estiamtes)
                #using just g and r and each gets 1/2 vote
                #todo: not often, but if we have u-band, would expect it to be faint
                #for g:  24.5+ increasingly favors LAE
                #
                try:
                    if self.best_img_g_mag is not None and self.best_img_g_mag[0] is not None:
                        if counterpart_filter == 'g':
                            w = 0.25
                        else:
                            w = 0.5

                        ew = 999
                        unc = 0

                        if self.best_eqw_gmag_obs is not None:
                            ew = self.best_eqw_gmag_obs / (self.w /G.LyA_rest)
                            if self.best_eqw_gmag_obs_unc is not None:
                                unc = self.best_eqw_gmag_obs_unc  / (self.w /G.LyA_rest)


                        if (ew-unc) < G.LAE_EW_MAG_TRIGGER_MAX and (ew+unc) > G.LAE_EW_MAG_TRIGGER_MIN:
                            w = w * mag_gaussian_weight(adjusted_mag_zero(G.LAE_G_MAG_ZERO,mg_z),self.best_img_g_mag[0],
                                                    self.best_img_g_mag[1],self.best_img_g_mag[2])
                            if self.best_img_g_mag[0] < adjusted_mag_zero(G.LAE_G_MAG_ZERO,mg_z):
                                likelihood.append(0.0)
                            else:
                                likelihood.append(1.0)
                            weight.append(w)
                            var.append(1)
                            prior.append(base_assumption)
                            log.info(f"{self.entry_id} Aggregate Classification: aperture g-mag vote {self.best_img_g_mag[0]:0.2f} : lk({likelihood[-1]}) "
                                     f"weight({weight[-1]})")
                    elif self.best_gmag is not None:
                        g = min(self.best_gmag,G.HETDEX_CONTINUUM_MAG_LIMIT)
                        g_bright = None
                        g_faint = None
                        try:
                            if g == G.HETDEX_CONTINUUM_MAG_LIMIT:
                                g_bright = g
                            else:
                                g_bright = g - self.best_gmag_unc
                            g_faint = g + self.best_gmag_unc
                        except:
                            pass

                        ew = 999
                        unc = 0
                        if self.best_eqw_gmag_obs is not None:
                            ew = self.best_eqw_gmag_obs / (self.w / G.LyA_rest)
                            if self.best_eqw_gmag_obs_unc is not None:
                                unc = self.best_eqw_gmag_obs_unc / (self.w / G.LyA_rest)

                        if (ew-unc) < G.LAE_EW_MAG_TRIGGER_MAX and (ew+unc) > G.LAE_EW_MAG_TRIGGER_MIN:
                            w = 0.25 * mag_gaussian_weight(adjusted_mag_zero(G.LAE_G_MAG_ZERO,mg_z),g,g_bright,g_faint)

                            if g < adjusted_mag_zero(G.LAE_G_MAG_ZERO,mg_z):
                                likelihood.append(0.0)
                            else:
                                likelihood.append(1.0)
                            weight.append(w)
                            var.append(1)
                            prior.append(base_assumption)
                            log.info(f"{self.entry_id} Aggregate Classification: DEX g-mag vote {g:0.2f} : lk({likelihood[-1]}) "
                                     f"weight({weight[-1]})")
                except:
                    pass

                try:
                    if self.best_img_r_mag is not None and self.best_img_r_mag[0] is not None:
                        # if counterpart_filter == 'r':
                        #    w = 0.25
                        # else:
                        #    w = 0.5

                        w = 0.5

                        ew = 999
                        unc = 0
                        cont = SU.mag2cgs(self.best_img_r_mag[0],6500.0)
                        cont_unc = None
                        if self.best_img_r_mag[1] is not None:
                            cont_unc = abs(cont - SU.mag2cgs(self.best_img_r_mag[1],6500.0))
                        ew,unc = SU.lya_ewr(self.estflux,self.estflux_unc,self.w,'r',cont,cont_unc)
                        if np.isnan(ew):
                            ew = 999
                        if np.isnan(unc):
                            unc = 0

                        if (ew-unc) < G.LAE_EW_MAG_TRIGGER_MAX and (ew+unc)> G.LAE_EW_MAG_TRIGGER_MIN:
                            w = w * mag_gaussian_weight(adjusted_mag_zero(G.LAE_R_MAG_ZERO,mg_z),self.best_img_r_mag[0],
                                                    self.best_img_r_mag[1],self.best_img_r_mag[2])
                            if self.best_img_r_mag[0] < adjusted_mag_zero(G.LAE_R_MAG_ZERO,mg_z):
                                likelihood.append(0.0)
                            else:
                                likelihood.append(1.0)
                            weight.append(w)
                            var.append(1)
                            prior.append(base_assumption)
                            log.info(f"{self.entry_id} Aggregate Classification: aperture r-mag vote {self.best_img_r_mag[0]:0.2f} : lk({likelihood[-1]}) "
                                     f"weight({weight[-1]})")
                except:
                    pass

                #basic magnitude sanity checks
                if lower_mag < 18.0: #the VERY BRIGHTEST QSOs in the 2 < z < 4 are 17-18 mag
                    likelihood.append(0.1)  # weak solution so push likelihood "down" but not zero (maybe 0.2 or 0.25)?
                    weight.append(max(2.0,max(weight)))
                    var.append(1)  # todo: ? could do something like the spectrum noise?
                    prior.append(base_assumption)
                    log.info(f"{self.entry_id} Aggregate Classification: gmag too bright {self.best_gmag} to be LAE (AGN): lk({likelihood[-1]}) "
                        f"weight({weight[-1]})")
                elif lower_mag < 23.0:
                    try:
                        min_fwhm = self.fwhm - (0 if ((self.fwhm_unc is None) or (np.isnan(self.fwhm_unc))) else self.fwhm_unc)
                        min_thresh = max( ((23.0 - lower_mag) + 8.0), 8.0) #just in case something weird

                        #the -25.0 and -0.8 are from some trial and error plotting to get the shape I want
                        #runs 0 to 1.0 and drops off very fast from 1.0 toward 0.0
                        # (by ratio of 0.8 were at y=0.5, by 0.6 y ~ 0.0)
                        sigmoid = 1.0 / (1.0 + np.exp(-25.0 * (min_fwhm/min_thresh - 0.8)))
                        if min_fwhm < min_thresh:
                            #unless this is an AGN, this is very unlikely
                            likelihood.append(min(0.5, sigmoid))  # weak solution so push likelihood "down" but not zero (maybe 0.2 or 0.25)?
                            weight.append(1.0 * (1.0-sigmoid))
                            var.append(1)  # todo: ? could do something like the spectrum noise?
                            prior.append(base_assumption)
                            log.info(f"{self.entry_id} Aggregate Classification: gmag too bright {self.best_gmag} for fwhm {self.fwhm}: lk({likelihood[-1]}) "
                                      f"weight({weight[-1]})")
                    except:
                        log.debug("Exception in aggregate_classification for best PLAE/POII", exc_info=True)


        ########################################################
        #check for specific incompatible classification labels
        ########################################################

        if "Meteor" in self.spec_obj.classification_label:
            likelihood.append(0.0)
            weight.append(self.spec_obj.meteor_strength)
            var.append(1)  # todo: ? could do something like the spectrum noise?
            prior.append(base_assumption)
            log.info( f"{self.entry_id} Aggregate Classification: Meteor indicated: lk({likelihood[-1]}) weight({weight[-1]})")

        ##########################
        # single line OIII 5007
        # * replaced by special handling in spectrum::classify_with_additional_lines
        ##########################

        # try:
        #     #5006.83 in air
        #     #basically, tend to be very narrow with a fair amount of flux (so moderately high eqw)
        #     #this is a safety against possible OIII-5007 as a single line,
        #     # (which is NOT part of the PLAE/POII comparision)
        #     if (self.spec_obj is not None) and (len(self.spec_obj.solutions) == 0) and (self.w > 5006.5):
        #         if self.spec_obj.fwhm + self.spec_obj.fwhm_unc < 6.0: #getting suspicious
        #             if (self.eqw_hetdex_gmag_obs) and (not np.isnan(self.eqw_hetdex_gmag_obs)) and \
        #                 self.eqw_hetdex_gmag_obs / (self.w/1216.) > 100.0:
        #                 if (self.eqw_line_obs) and (not np.isnan(self.eqw_line_obs)) and \
        #                 self.eqw_line_obs / (self.w/1216.) > 35.0:
        #                     #todo: need other / better criteria (maybe an EW distro for OIII 5007) from HETDEX?
        #                     #todo: masking (of low-z galaxies could also help a lot)
        #
        #                     likelihood.append(0.0)
        #                     weight.append(1.0)
        #                     var.append(1)
        #                     prior.append(base_assumption)
        #                     log.info(
        #                         f"Aggregate Classification: OIII-5007 possible: lk({likelihood[-1]}) weight({weight[-1]})")
        #
        #
        # except:
        #     pass

        #todo: "Star"


        #todo: chi2 vs S/N  (is it a real line)
        #  need to include (for poor S/N) if there is a faint catalog object right under the reticle (say within 0.5")
        #a question of fit ...

        #check for bad pixel flats
        bad_pixflt_weight = 0
        for fidx in range(len(self.fibers)): #are in decreasing order of weight

            if (self.fibers[fidx].pixel_flat_center_ratio < G.MIN_PIXEL_FLAT_CENTER_RATIO) or \
               (self.fibers[fidx].pixel_flat_center_avg < G.PIXEL_FLAT_ABSOLUTE_BAD_VALUE):
                bad_pixflt_weight += self.fibers[fidx].relative_weight
                likelihood.append(0)
                weight.append(1.0 + self.fibers[fidx].relative_weight) #more central fibers make this more likely to trigger
                var.append(1)
                prior.append(0)
                log.info(f"{self.entry_id} Aggregate Classification: bad pixel flat for fiber #{fidx+1}. lk({likelihood[-1]}) "
                         f"weight({weight[-1]}) relative fiber weight({self.fibers[fidx].relative_weight})")

        #check for duplicate pixel positions
        #NO! there are valid (good) conditions where there ARE duplicate positions but it is good and correct
        #(dithering CAN produce different fibers on the same pixel where this is good signal in each dither)
        # if self.num_duplicate_central_pixels > G.MAX_NUM_DUPLICATE_CENTRAL_PIXELS: #out of the top (usually 4) fibers
        #     likelihood.append(0)
        #     weight.append(self.num_duplicate_central_pixels)  # more central fibers make this more likely to trigger
        #     var.append(1)
        #     prior.append(0)
        #     log.info(
        #         f"{self.entry_id} Aggregate Classification: bad duplicate central pixels: {self.num_duplicate_central_pixels}. lk({likelihood[-1]}) weight({weight[-1]})")

        #don't just drive down the PLAE, make it negative as a flag
        if bad_pixflt_weight > 0.5:
            # likelihood.append(-1)
            # weight.append(999)  # more central fibers make this more likely to trigger
            # var.append(1)
            # prior.append(0)
            reason = "(bad pixel flat)"
            scaled_prob_lae = -1
            self.classification_dict['scaled_plae'] = scaled_prob_lae
            self.classification_dict['spurious_reason'] = reason
            log.info(f"{self.entry_id} Aggregate Classification: bad pixel flat dominates. Setting PLAE to -1 (spurious)")
        elif self.duplicate_fiber_cutout_pair_weight > 0.0:
            reason = "(duplicate 2D fibers)"
            scaled_prob_lae = -1
            self.classification_dict['scaled_plae'] = scaled_prob_lae
            self.classification_dict['spurious_reason'] = reason
            log.info(f"{self.entry_id} Aggregate Classification: duplicate 2D fibers "
                     f"(min weight {self.duplicate_fiber_cutout_pair_weight}). Setting PLAE to -1 (spurious)")
        elif self.grossly_negative_spec:
            reason = "(negative spectrum)"
            scaled_prob_lae = -1
            self.classification_dict['scaled_plae'] = scaled_prob_lae
            self.classification_dict['spurious_reason'] = reason
            log.info(f"{self.entry_id} Aggregate Classification: grossly negative spectrum. Setting PLAE to -1 (spurious)")
        #shot conditions
        elif (self.survey_response < 0.08) or (self.survey_fwhm > 3.0) or \
                (np.isnan(self.dither_norm) or (self.dither_norm > 3.0)):
            if self.survey_response < 0.05: #this alone means we're done
                 reason = "(poor throughput)"
                 scaled_prob_lae = -1
                 self.classification_dict['scaled_plae'] = scaled_prob_lae
                 self.classification_dict['spurious_reason'] = reason
                 log.info(f"{self.entry_id} Aggregate Classification: poor throughput {self.survey_response}. Setting PLAE to -1 (spurious)")
            elif (np.isnan(self.dither_norm) or (self.dither_norm > 3.0)):
                reason = "(bad dither norm)"
                scaled_prob_lae = -1
                self.classification_dict['scaled_plae'] = scaled_prob_lae
                self.classification_dict['spurious_reason'] = reason
                log.info(
                    f"{self.entry_id} Aggregate Classification: poor throughput {self.survey_response}. Setting PLAE to -1 (spurious)")
            else:
                bool_sum  = (self.survey_response < 0.08) + (self.survey_fwhm > 3.0) +  (np.isnan(self.dither_norm) or (self.dither_norm > 3.0))
                if bool_sum > 1:
                    reason = "(poor shot)"
                    scaled_prob_lae = -1
                    self.classification_dict['scaled_plae'] = scaled_prob_lae
                    self.classification_dict['spurious_reason'] = reason
                    log.info(
                        f"{self.entry_id} Aggregate Classification: poor shot F: {self.survey_fwhm} T: {self.survey_response}  N:{self.dither_norm}. Setting PLAE to -1 (spurious)")
                else:
                    log.info(
                        f"{self.entry_id} Aggregate Classification: poor shot F: {self.survey_fwhm} T: {self.survey_response}  N:{self.dither_norm}, but did not trigger spurious.")


        # check for duplicate pixel positions
        # elif self.num_duplicate_central_pixels > G.MAX_NUM_DUPLICATE_CENTRAL_PIXELS:  # out of the top (usually 4) fibers
        #     reason = "(duplicate pixels)"
        #     scaled_prob_lae = -1
        #     self.classification_dict['scaled_plae'] = scaled_prob_lae
        #     self.classification_dict['spurious_reason'] = reason
        #     log.info(f"Aggregate Classification: bad duplicate central pixels. Setting PLAE to -1 (spurious)")

        if scaled_prob_lae != -1:
            #
            # Combine them all
            #

            try:
                if len(weight) > 0 and np.sum(weight) < 1.0:
                    tot_weight = np.sum(weight)
                    log.info(f"Low voting weight ({tot_weight}). Adding in 0.5 vote at {1.0-tot_weight} weight.")
                    likelihood.append(0.5)
                    weight.append(1.0 - np.sum(weight))
                    var.append(1.0)
                    prior.append(0.5)

                    if weight[-1] >= 0.5:
                        self.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION
                        self.flags |= G.DETFLAG_FOLLOWUP_NEEDED
            except:
                pass

            likelihood = np.array(likelihood)
            weight = np.array(weight)
            var = np.array(var) #right now, this is just weight based, all variances are set to 1
            prior = np.array(prior) #not using Bayes yet, so this is ignored

            logstring = f"P(Lya) weighted voting. {len(weight)} votes as (vote x weight)\n"
            for i in range(len(weight)):
                logstring += f"({likelihood[i]:0.4f}x{weight[i]})  "
            # for l_vote,l_weight,l_var,l_prior in zip(likelihood,weight,var,prior):
            #     logstring += f"({l_vote:0.4f}x{l_weight})  "
            log.debug(logstring)

            #sanity check ... no negative weights allowed
            sel = [weight > 0] #skip any negative weights and no need to bother with zero weights
            if np.any([weight < 0]):
                log.warning("Warning! One or more negative weights ignored.")

            try:
                if len(likelihood) > 0:
                    scaled_prob_lae = np.sum(likelihood[sel]*weight[sel]/var[sel])/np.sum(weight[sel]/var[sel]) #/ len(likelihood)

                    #while can be arbitrarily close to 0 or 1, will crop to 0.001 to 0.999
                    if 0.0 <= scaled_prob_lae < 0.001:
                        scaled_prob_lae = 0.001
                    elif 0.999 < scaled_prob_lae <= 1.0:
                        scaled_prob_lae = 0.999

                    self.classification_dict['scaled_plae'] = scaled_prob_lae
                    log.info(f"{self.entry_id} Scaled P(LyA) {scaled_prob_lae:0.4f}")
                    #print(f"{self.entry_id} Scaled Prob(LyA) {scaled_prob_lae:0.4f}")
            except:
                log.debug("Exception in aggregate_classification final combination",exc_info=True)


        self.vote_info = vote_info
        return scaled_prob_lae, reason


    def combine_all_plae(self,use_continuum=True):
        """
        Combine (all) PLAE/POII ratios into a single, best 'guess' value with errors

        "Average" using ~ inverse variance* AND rules weighting

        * these are not normal distros (those near 1000 and 0.001 are truncated AND they can be highly skewed, so we
        are using the averaged (68%) confidence interval for each as a proxy for (std dev)

        #todo: consider combining in Bayesian fashion and include extra info if available (like spec-z, phot-z for catalog matches?)

        :return:
        """

        def avg_var(plae,plae_min, plae_max):
            #can be truncated near 1000 or 0.001
            #pseduo_sd = 0.5 * ((plae-plae_min) + (plae_max - plae))
            pseduo_sd = 0.5 * (plae_max-plae_min)
            if pseduo_sd == 0: #min==max
                if plae_max > 999 or plae_max < 0.002: #same, but truncated to be the same
                    return 0.00001 #minimum truncated value squared
                else:
                    return plae * plae
            else:
                return pseduo_sd*pseduo_sd



        if use_continuum:
            continuum_hat, continuum_sd_hat, size_in_psf, diam_in_arcsec = self.combine_all_continuum()

            self.classification_dict['combined_eqw_rest_lya'] = self.estflux / continuum_hat / (self.w/G.LyA_rest)
            self.classification_dict['combined_eqw_rest_lya_err'] =  self.classification_dict['combined_eqw_rest_lya'] *\
                                                                     np.sqrt((self.estflux_unc/self.estflux)**2 + (continuum_sd_hat/continuum_hat)**2)


#feed into MC PLAE
            p_lae_oii_ratio, p_lae, p_oii, plae_errors =  line_prob.mc_prob_LAE(
                                                            wl_obs=self.w,
                                                            lineFlux=self.estflux,lineFlux_err=self.estflux_unc,
                                                            continuum=continuum_hat,continuum_err=continuum_sd_hat,
                                                            c_obs=None, which_color=None,
                                                            addl_fluxes=[], addl_wavelengths=[],
                                                            sky_area=None,cosmo=None, lae_priors=None,
                                                            ew_case=None, W_0=None,z_OII=None, sigma=None)




            try:
                plae_sd = plae_errors['ratio'][3]
                log.debug(f"{self.entry_id} Combine ALL PLAE: MC plae({p_lae_oii_ratio}) sd({plae_sd})")
            except:
                try:
                    plae_sd = np.sqrt(avg_var(plae_errors['ratio'][0],plae_errors['ratio'][1],plae_errors['ratio'][2]))
                except:
                    plae_sd = None
                    log.debug(f"{self.entry_id} Combine ALL PLAE: MC plae({p_lae_oii_ratio}) sd({plae_sd})")

            try:
                self.classification_dict['plae_hat'] = p_lae_oii_ratio
                self.classification_dict['plae_hat_hi'] = plae_errors['ratio'][2]
                self.classification_dict['plae_hat_lo'] = plae_errors['ratio'][1]
                self.classification_dict['plae_hat_sd'] = plae_sd
                self.classification_dict['plae_hat_err'] = max((plae_errors['ratio'][2] - p_lae_oii_ratio),
                                                               (p_lae_oii_ratio - plae_errors['ratio'][1]))
            except:
                #log.debug("Exception in hetdex::combine_all_plae()",exc_info=True)
                log.info(f"Unusable classifiction_dict in hetdex::combine_all_plae(). {self.classification_dict}")
                self.classification_dict['plae_hat'] = -1
                self.classification_dict['plae_hat_hi'] = -1
                self.classification_dict['plae_hat_lo'] = -1
                self.classification_dict['plae_hat_sd'] = -1
                self.classification_dict['plae_hat_err'] = -1

            try:
                log.info(f"{self.entry_id} Combine ALL PLAE: MC plae {self.classification_dict['plae_hat']} : "
                         f"high ({self.classification_dict['plae_hat_hi']}) "
                         f"low ({self.classification_dict['plae_hat_lo']})")
            except:
                pass

            try:
                self.classification_dict['size_in_psf'] = size_in_psf
                self.classification_dict['diam_in_arcsec'] = diam_in_arcsec
            except:
                log.debug("Exception in hetdex::combine_all_plae()", exc_info=True)
                self.classification_dict['size_in_psf'] = -1
                self.classification_dict['diam_in_arcsec'] = -1

            try:
                self.classification_dict['continuum_hat'] = continuum_hat
                self.classification_dict['continuum_hat_err'] = continuum_sd_hat
            except:
                log.debug("Exception in hetdex::combine_all_plae()", exc_info=True)
                self.classification_dict['continuum_hat'] = -1
                self.classification_dict['continuum_hat_err'] = -1


            return p_lae_oii_ratio, plae_sd, size_in_psf, diam_in_arcsec

        else:
            log.error("ERROR. Unexpected call to combine_all_plae using old method not combining continua.")
            return None,None,None,None
        #
        # older method comparing PLAE/POII instead of using composit spectrum as continuum estimate
        #
        #
        # plae = [] #plae/poii ratio
        # variance = [] #variance or variance proxy
        # weight = [] #rules based weights
        #
        # best_guess_extent = [] #from source extractor (f606w, g, r only)
        # base_psf = []
        # size_in_psf = None #estimate of extent in terms of PSF (sort-of)
        #
        # #todo: evaluate each PLAE/POII ratio available and set the variance (proxy) and weight before summing up
        #
        # #HETDEX PLAE
        # #what criteria to set weight? maybe really low chi^2 is good weight?
        # try:
        #     if (self.hetdex_cont_cgs is not None) and (self.hetdex_cont_cgs_unc is not None):
        #         hetdex_cont_limit =  G.HETDEX_CONTINUUM_FLUX_LIMIT #8.5e-19 #don't trust below this
        #         if (self.hetdex_cont_cgs - self.hetdex_cont_cgs_unc) > hetdex_cont_limit:
        #             w = 0.2 #never very high
        #             p = self.p_lae_oii_ratio_range[0]
        #             if len(self.p_lae_oii_ratio_range > 3):
        #                 v = self.p_lae_oii_ratio_range[3] * self.p_lae_oii_ratio_range[3]
        #             else:
        #                 v = avg_var(self.p_lae_oii_ratio_range[0], self.p_lae_oii_ratio_range[1], self.p_lae_oii_ratio_range[2])
        #
        #             # only add if p and v both successful
        #             plae.append(p)
        #             variance.append(v)
        #             weight.append(w)
        #             log.debug(f"{self.entry_id} Combine ALL PLAE: Added HETDEX: plae({p:#.4g}) var({v:#.4g}) weight({w:#.2f})")
        #         else:
        #             log.debug(f"{self.entry_id} Combine ALL PLAE: Failed HETDEX estimate")
        #         #else:
        #         #    w = 0.0 #don't trust it
        # except:
        #     log.debug("Exception handling HETDEX PLAE/POII in DetObj:combine_all_plae",exc_info=True)
        #
        #
        # #SDSS gmag PLAE
        # try:
        #     if (self.sdss_cgs_cont is not None) and (self.sdss_cgs_cont_unc is not None):
        #         # set weight to zero if gmag > 25
        #         # set to low value if gmag > 24
        #         # set to good value if gmag < 24
        #         cgs_24 = 1.35e-18  #1.35e-18 cgs ~ 24.0 mag in g-band
        #         cgs_24p5 = 8.52e-19 #8.52e-19 cgs ~ 24.5 mag in g-band, get full marks at 24mag and fall to zero by 24.5
        #         cgs_25 = 5.38e-19 #g-mag
        #         #if (self.sdss_cgs_cont - self.sdss_cgs_cont_unc) > cgs_24p5:
        #         #    frac = (self.sdss_cgs_cont - cgs_24)/(cgs_24 - cgs_24p5)
        #         if (self.sdss_cgs_cont - self.sdss_cgs_cont_unc) > cgs_25:
        #             frac = (self.sdss_cgs_cont - cgs_24) / (cgs_24 - cgs_25)
        #             # at 24mag this is zero, fainter goes negative, at 24.5 it is -1.0
        #             if frac > 0: #full weight
        #                 w = 1.0
        #             else:
        #                 w = 1.0 + frac #linear drop to zero at 8.52e-19
        #         else:
        #             w = 0.0
        #
        #         if w > 0.0:
        #             p = self.sdss_gmag_p_lae_oii_ratio_range[0]
        #
        #             if len(self.sdss_gmag_p_lae_oii_ratio_range > 3):
        #                 v = self.sdss_gmag_p_lae_oii_ratio_range[3] * self.sdss_gmag_p_lae_oii_ratio_range[3]
        #             else:
        #                 v = avg_var(self.sdss_gmag_p_lae_oii_ratio_range[0],
        #                         self.sdss_gmag_p_lae_oii_ratio_range[1], self.sdss_gmag_p_lae_oii_ratio_range[2])
        #
        #             # only add if p and v both successful
        #             plae.append(p)
        #             variance.append(v)
        #             weight.append(w)
        #             log.debug(f"{self.entry_id} Combine ALL PLAE: Added SDSS gmag: plae({p:#.4g}) var({v:#.4g}) weight({w:#.2f})")
        #         else:
        #             log.debug(f"{self.entry_id} Combine ALL PLAE: Failed SDSS gmag estimate")
        #
        # except:
        #     log.debug("Exception handling SDSS gmag PLAE/POII in DetObj:combine_all_plae", exc_info=True)
        #
        # #Best forced aperture PLAE
        # #how to set?? maybe always the same?
        # #iterate over all, g,r,606w all count with equal weight
        # try:
        #     filters = ['f606w','g','r']
        #     for a in self.aperture_details_list: #has both forced aperture and sextractor
        #         try:
        #             if (a['filter_name'] is not None) and (a['filter_name'] in filters):
        #                 #todo: ELiXer force aperture (no PLAE/POII right now, just counts and mag
        #                 # this would be forced_aperture = a['elixer_apertures'][a['elixer_aper_idx']]
        #
        #                 try:
        #                     #use the first radius for the aperture (x2 for diameter)
        #                     #is based on the reported average PSF, so this is marginally okay
        #                     if a['elixer_apertures'] is not None:
        #                         base_psf.append(a['elixer_apertures'][0]['radius']*2.0) #not quite the PSF, but similar
        #                         log.debug(f"{self.entry_id} Combine ALL PLAE: Added base psf: "
        #                                   f"{a['elixer_apertures'][0]['radius']*2.0} arcsec,"
        #                                   f" filter ({a['filter_name']})")
        #                 except:
        #                     log.debug("Exception handling base_psf in DetObj:combin_all_plae", exc_info=True)
        #
        #                 #todo: source extractor objects (no PLAE/POII right now, just counts and mag
        #                 # this would be sextractor = a['sep_objects'][a['sep_obj_idx']]
        #                 # adjust weight based on distance to barycenter if inside ellipse or to edge of ellipse if not
        #
        #                 #todo: this should be re-done in terms of the imaging catalog PSF
        #                 try:
        #                     if (a['sep_objects'] is not None) and (a['sep_obj_idx'] is not None):
        #                         best_guess_extent.append(a['sep_objects'][a['sep_obj_idx']]['a'])
        #                         log.debug(f"{self.entry_id} Combine ALL PLAE: Added best guess extent added: "
        #                               f"{a['sep_objects'][a['sep_obj_idx']]['a']:#.2g} arcsec,"
        #                               f" filter ({a['filter_name']})")
        #                 except:
        #                     log.debug("Exception handling best_guess_extent in DetObj:combin_all_plae",exc_info=True)
        #
        #                 #for now, just using the "best" selected by ELiXer
        #                 if  a['aperture_plae'] is not None:
        #                     p = a['aperture_plae']
        #                     v = avg_var(a['aperture_plae'],a['aperture_plae_min'],a['aperture_plae_max'])
        #                     w = 1.0
        #
        #                     #only add if p and v both successful
        #                     plae.append(p)
        #                     variance.append(v)
        #                     weight.append(w)
        #                     log.debug(f"{self.entry_id} Combine ALL PLAE: Added best guess filter: plae({p:#.4g}) "
        #                               f"var({v:#.4g}) weight({w:#.2f}) filter({a['filter_name']})")
        #         except:
        #             log.debug("Exception handling individual forced aperture photometry PLAE/POII in DetObj:combine_all_plae",
        #                       exc_info=True)
        # except:
        #     log.debug("Exception handling forced aperture photometry PLAE/POII in DetObj:combine_all_plae", exc_info=True)
        #
        # #?? Best catalog match PLAE ?? #do I even want to try to use this one?
        #
        # try:
        #     best_guess_extent = np.array(best_guess_extent)
        #     base_psf = np.array(base_psf)
        #
        #     if len(best_guess_extent) == len(base_psf) > 0:
        #         size_in_psf = np.mean(best_guess_extent/base_psf) #usually only 1 or 2, so std makes no sense
        #         best_guess_extent = np.mean(best_guess_extent)
        #
        # except:
        #     pass
        #
        # #sum up
        # try:
        #     plae = np.array(plae)
        #     variance = np.array(variance)
        #     weight = np.array(weight)
        #     #v2 = variance*variance
        #
        #     plae_hat = np.sum(plae * weight / variance) / np.sum(weight / variance)
        #     plae_hat_sd = np.sqrt(np.sum(weight*variance)/np.sum(weight))
        #
        #     #todo: what about plae_hat uncertainty?
        #     #todo: should I (instead of inverse variance) sample all like 250-1000 times from random distro
        #     #todo: and sum only on weights, then take the mean and std, of the resulting histogram as the plae_hat and sd?
        #
        #
        #     #todo: what about extra info now for Bayesian analysis? update our "beliefs" based on
        #     #todo: obect extent? (consistent or inconsistent with PSF and thus with a point-source?)
        #     #todo: slope of whole spectra or spectral features that would preclude LAE?
        #     #todo: ELiXer line finder strongly suggesting mutlitple lines and NOT LAE?
        #
        #     log.debug(f"{self.entry_id} Combine ALL PLAE: Final estimate: plae_hat({plae_hat}) plae_hat_sd({plae_hat_sd}) "
        #               f"size in psf ({size_in_psf})")
        #
        #     self.classification_dict['plae_hat'] = plae_hat
        #     self.classification_dict['plae_hat_sd'] = plae_hat_sd
        #     self.classification_dict['size_in_psf'] = size_in_psf
        #     self.classification_dict['diam_in_arcsec'] = best_guess_extent
        #
        #     return plae_hat, plae_hat_sd, size_in_psf, best_guess_extent
        # except:
        #
        #     log.debug("Exception handling estimation in DetObj:combine_all_plae", exc_info=True)
        #
        # return None, None, None, None



    def combine_all_continuum(self):
        """
        Combine (all) continuum estimates into a single, best 'guess' value with errors

        "Average" using ~ inverse variance* AND rules weighting

        * these are not normal distros (those near 1000 and 0.001 are truncated AND they can be highly skewed, so we
        are using the averaged (68%) confidence interval for each as a proxy for (std dev)

        #todo: consider combining in Bayesian fashion and include extra info if available (like spec-z, phot-z for catalog matches?)

        :return:
        """

        def avg_var(val,val_min, val_max):
            # assumes 1-sigma error and symmetric (Gaussian-like) error
            try:
                pseduo_sd = 0.5 * abs(val_max-val_min)
                if (pseduo_sd == 0) or (abs(np.log10(val)-np.log10(pseduo_sd)) > 3.): #assume a 33% std deviation (or (.1*var) ... so variance == (0.1*val)**2)
                    log.debug("hetdex::combine_all_continuum, pseudo-sd == 0, so setting to value")
                    return val * val #0.11 is ~ 0.33**2
                else:
                    return pseduo_sd*pseduo_sd
            except:
                return val * val #0.11 is ~ 0.33**2

        continuum = []
        variance = [] #variance or variance proxy
        weight = [] #rules based weights
        cont_type = [] #what was the input
        nondetect = []
        deep_non_detect = 0 #deepest g or r band imaging without a detection
        deep_detect = 0 #deepest g or r band imaging with a detection  (source extractor only)

        best_guess_extent = [] #from source extractor (f606w, g, r only)
        best_guess_maglimit = []
        base_psf = []
        size_in_psf = None #estimate of extent in terms of PSF (sort-of)

        gmag_at_limit = False

        # set weight to zero if gmag > 25
        # set to low value if gmag > 24
        # set to good value if gmag < 24
        # cgs_24g = 1.35e-18  # 1.35e-18 cgs ~ 24.0 mag in g-band
        # cgs_24p5g = 8.52e-19  # 8.52e-19 cgs ~ 24.5 mag in g-band, get full marks at 24mag and fall to zero by 24.5
        # cgs_25g = 5.38e-19 #
        # cgs_26g = 2.14e-19
        # cgs_27g = 8.52e-20
        # cgs_28g = 3.39e-20
        # cgs_29g = 1.35e-20
        # cgs_30g = 5.38e-21
        #
        # cgs_24r = 6.47e-19 #  24.0 mag in r-band
        # cgs_24p5r = 4.08-19  # 8.52e-19 cgs ~ 24.5 mag in r-band, get full marks at 24mag and fall to zero by 24.5
        # cgs_25r = 2.58e-19 #
        # cgs_26r = 1.03-19
        # cgs_27r = 4.08e-20
        # cgs_28r = 1.63e-20
        # cgs_29r = 6.47e-21
        # cgs_30r = 2.58e-21

        cgs_faint_limit = 1e-20 #HSC much fainter, but this is just a nominal value to use for error

        num_cat_match = 0 #number of catalog matched objects
        cat_idx = -1

        #
        # separation between LAE and OII is around 24 mag (so if flux limit is reached below that, there
        # really is no information to be gained).
        # For now, setting weight to zero if well below 24mag (could alternately kick the error way up
        # since the true value could be at the limit out to say, 28 mag or so)
        #
        #

        # Best full width gmag continuum (HETDEX full width or SDSS gmag)
        got_hd_gmag = True
        try:
            #all at 4500AA
            # cgs_25 = 5.38e-19
            # cgs_24p5 = 8.52e-19
            # cgs_24 = 1.35e-18
            # cgs_23p5 = 2.14e-18

            cgs_limit = G.HETDEX_CONTINUUM_FLUX_LIMIT # cgs_24p5
            cgs_fc =  G.HETDEX_CONTINUUM_FLUX_LIMIT * 1.2 #start downweighting at 20% brighter than hard limit

            if not self.best_gmag_cgs_cont_unc: #None or 0.0
                log.debug(f"{self.entry_id} Combine ALL Continuum: HETDEX wide estimate has no uncertainty. "
                          f"Setting to limit {cgs_limit}")
                self.best_gmag_cgs_cont_unc = cgs_limit #set to the mag limit (so, we're saying the
                #mag is whatever we measured +/- the limit (so still a very large error and will get down weighted)

            if (self.best_gmag_cgs_cont is not None) and (self.best_gmag_cgs_cont_unc is not None) and \
               (self.best_gmag_cgs_cont > 0) and (self.best_gmag_cgs_cont_unc > 0):

                #use a sigmoid s|t when the flux-error is at the flux limit you get at vote of 1
                #when brighter than the flux limit, the vote moves up to a max of 4 by 1.2x the limit
                #as we go fainter than the limit we rapidly fall to zero by 10% past the flux limit

                rat = (self.best_gmag_cgs_cont - self.best_gmag_cgs_cont_unc) / G.HETDEX_CONTINUUM_FLUX_LIMIT
                w = 1.0 / (1.0 + np.exp(-40 * (rat -0.015) + 40.5)) * 4.0
                continuum.append(self.best_gmag_cgs_cont)
                weight.append(w)

                cont_type.append('hdw') #HETDEX wide

                if w > 0.9:
                    nondetect.append(0)
                    variance.append(self.best_gmag_cgs_cont_unc * self.best_gmag_cgs_cont_unc)
                    gmag_at_limit = False
                else:
                    variance.append((cgs_limit-cgs_faint_limit)**2)
                    nondetect.append(1)
                    gmag_at_limit = True

                log.info(
                        f"{self.entry_id} Combine ALL Continuum: Added best spectrum gmag estimate ({continuum[-1]}) "
                        f"sd({np.sqrt(variance[-1])}) weight({weight[-1]})")

                if False: #old way
                    #estimate - error is still better than the limit, so it gets full marks
                    if (self.best_gmag_cgs_cont - self.best_gmag_cgs_cont_unc) > cgs_limit: #good, full marks
                        continuum.append(self.best_gmag_cgs_cont)
                        variance.append(self.best_gmag_cgs_cont_unc * self.best_gmag_cgs_cont_unc)

                        if (self.best_gmag_cgs_cont - self.best_gmag_cgs_cont_unc) > cgs_fc:
                            weight.append(4.0) #best measure of flux (right on top of our target) so boost
                            #typically 4 other estiamtes: narrow, wide (this one), 1+ forced photometery, 1+ catalog
                            #so make this one 4x to dominated (aside from big error)
                        else:
                            weight.append(1.0)

                        cont_type.append('hdw') #HETDEX wide
                        nondetect.append(0)

                        log.debug(
                            f"{self.entry_id} Combine ALL Continuum: Added best spectrum gmag estimate ({continuum[-1]:#.4g}) "
                            f"sd({np.sqrt(variance[-1]):#.4g}) weight({weight[-1]:#.2f})")

                    #estimate is better than the limit, but with error hits the limit, so reduced marks
                    elif (self.best_gmag_cgs_cont > cgs_limit): #mean value is in range, but with error is out of range
                        frac = (self.best_gmag_cgs_cont - cgs_fc) / (cgs_fc - cgs_limit)
                        # at 24mag this is zero, fainter goes negative, at 24.5 it is -1.0
                        if frac > 0:  # full weight
                            w = 1.0
                        else: #going to get at least 0.3
                            w = max(0.5, 1.0 + frac)  # linear drop to zero at cgs_limit

                        continuum.append(self.best_gmag_cgs_cont)
                        variance.append(self.best_gmag_cgs_cont_unc * self.best_gmag_cgs_cont_unc)
                        weight.append(w)
                        cont_type.append('hdw')  # HETDEX wide
                        nondetect.append(0)

                        log.debug(
                            f"{self.entry_id} Combine ALL Continuum: Added best spectrum gmag estimate ({continuum[-1]:#.4g}) "
                            f"sd({np.sqrt(variance[-1]):#.4g}) weight({weight[-1]:#.2f})")
                    else:  # going to use the lower limit, totally out of range
                        gmag_at_limit = True
                        continuum.append(cgs_limit)
                        variance.append((cgs_limit-cgs_faint_limit)**2)  # ie. sd of ~ 1/3 * cgs_limit
                        weight.append(0.5)  # never very high (a little better than HETDEX narrow continuum weight)
                        cont_type.append('hdw')  # HETDEX wide
                        nondetect.append(1)

                        log.debug(
                            f"{self.entry_id} Combine ALL Continuum: best spectrum gmag estimate fainter than limit. Setting to lower limit "
                            f"({continuum[-1]:#.4g}) "
                            f"sd({np.sqrt(variance[-1]):#.4g}) weight({weight[-1]:#.2f})")
            else:
                got_hd_gmag = False
        except:
            got_hd_gmag = False
            log.debug("Exception handling best spectrum gmag continuum in DetObj:combine_all_continuum",
                      exc_info=True)


        #HETDEX Continuum (near emission line), not very reliable only use if we did not get a HETDEX gmag
        if not got_hd_gmag:
            try:
                if (self.hetdex_cont_cgs is not None) and (self.hetdex_cont_cgs_unc is not None):
                    #hetdex_cont_limit = 2.0e-18 #don't trust below this
                    if (self.hetdex_cont_cgs - self.hetdex_cont_cgs_unc) > G.HETDEX_CONTINUUM_FLUX_LIMIT:
                        continuum.append(self.hetdex_cont_cgs)
                        variance.append(self.hetdex_cont_cgs_unc*self.hetdex_cont_cgs_unc)
                        weight.append(0.2) #never very high
                        cont_type.append('hdn') #HETDEX-narrow
                        nondetect.append(0)
                        log.debug(f"{self.entry_id} Combine ALL Continuum: Added HETDEX estimate ({continuum[-1]:#.4g}) "
                                  f"sd({np.sqrt(variance[-1]):#.4g}) weight({weight[-1]:#.2f})")
                    else: #set as lower limit ... too far to be meaningful
                        continuum.append(G.HETDEX_CONTINUUM_FLUX_LIMIT)
                        # set to itself as a big error (basically, 100% error)
                        variance.append(G.HETDEX_CONTINUUM_FLUX_LIMIT * G.HETDEX_CONTINUUM_FLUX_LIMIT)

                        #does it make any sense to attempt to use the calculated error in this case? it is the
                        #error on a value that is rejected as meaningless (below the limit or negative)
                        # if self.hetdex_cont_cgs_unc > 0:
                        #     variance.append(self.hetdex_cont_cgs_unc*self.hetdex_cont_cgs_unc)
                        # else:
                        #     variance.append(G.HETDEX_CONTINUUM_FLUX_LIMIT * G.HETDEX_CONTINUUM_FLUX_LIMIT) #set to itself as a big error

                        weight.append(0.0) #never very high
                        cont_type.append('hdn')
                        nondetect.append(1)
                        log.debug(f"{self.entry_id} Combine ALL Continuum: Failed HETDEX estimate, setting to lower limit  "
                                  f"({continuum[-1]:#.4g}) sd({np.sqrt(variance[-1]):#.4g}) weight({weight[-1]:#.2f})")

            except:
                log.debug("Exception handling HETDEX continuum in DetObj:combine_all_continuum",exc_info=True)


        #Best forced aperture PLAE
        #how to set?? maybe always the same?
        #iterate over all, g,r,606w all count with equal weight
        aperture_radius = 0.0
        try:
            filters = ['f606w','g','r']
            for a in self.aperture_details_list: #has both forced aperture and sextractor
                try:
                    if (a['filter_name'] is not None) and (a['filter_name'] in filters):
                        #todo: ELiXer force aperture (no PLAE/POII right now, just counts and mag
                        # this would be forced_aperture = a['elixer_apertures'][a['elixer_aper_idx']]

                        this_psf = None
                        matched_sep = False
                        any_sep = False
                        elix_aper_radius = -1
                        try:
                            #use the first radius for the aperture (x2 for diameter)
                            #is based on the reported average PSF, so this is marginally okay
                            if a['elixer_apertures'] is not None:
                                #base_psf.append(a['elixer_apertures'][0]['radius']*2.0) #not quite the PSF, but similar
                                #the zeroth aperture (start) is defined to be FWHM/2 - 0.5
                                this_psf = 2.0*(a['elixer_apertures'][0]['radius']-0.5)
                                #only add if there is a matching source extractor radius
                                # base_psf.append(this_psf)
                                #
                                # log.debug(f"{self.entry_id} Combine ALL Continuum: Added base psf: "
                                #           f"{a['elixer_apertures'][0]['radius']*2.0} arcsec,"
                                #           f" filter ({a['filter_name']})")


                                if a['elixer_aper_idx'] is not None:
                                    elix_aper_radius = a['elixer_apertures'][a['elixer_aper_idx']]['radius']
                        except:
                            log.debug("Exception handling base_psf in DetObj:combin_all_continuum", exc_info=True)

                        #todo: source extractor objects (no PLAE/POII right now, just counts and mag
                        # this would be sextractor = a['sep_objects'][a['sep_obj_idx']]
                        # adjust weight based on distance to barycenter if inside ellipse or to edge of ellipse if not

                        #todo: this should be re-done in terms of the imaging catalog PSF
                        try:
                            if (a['sep_objects'] is not None):
                                any_sep = len(a['sep_objects']) > 0
                                if (a['sep_obj_idx'] is not None):
                                    matched_sep = True
                                    best_guess_extent.append(a['sep_objects'][a['sep_obj_idx']]['a'])
                                    best_guess_maglimit.append(a['mag_limit'])
                                    base_psf.append(this_psf)

                                    log.debug(f"{self.entry_id} Combine ALL Continuum: Added base psf: "
                                          f"{this_psf} arcsec, filter ({a['filter_name']})")
                                    log.debug(f"{self.entry_id} Combine ALL Continuum: Added best guess extent: "
                                          f"{a['sep_objects'][a['sep_obj_idx']]['a']:#.2g} arcsec,"
                                          f" {a['catalog_name']} filter ({a['filter_name']})")

                                    if a['mag_limit'] is not None and a['mag_limit'] > deep_detect:
                                        deep_detect = a['mag_limit']

                                else: #there are SEP object(s), but they are too far away, so the apeture under/near
                                      #the reticle is probably empty .... this actually favors LAE (essentially would
                                      #be a non-detect)

                                    if a['mag_limit'] is not None and a['mag_limit'] > deep_non_detect:
                                        deep_non_detect = a['mag_limit']

                        except:
                            log.debug("Exception handling best_guess_extent in DetObj:combin_all_plae",exc_info=True)

                        if a['mag'] is not None:
                            # if a['filter_name'] == 'f606w':
                            #     lam = 5777. #AA
                            # elif  a['filter_name'] == 'g':
                            #     lam = 4770.
                            # elif a['filter_name'] == 'r':
                            #         lam = 6231.
                            # else:
                            #     log.error(f"Unexpected filter {a['mag']} in DetObj::combine_all_continuum")
                            #     lam = 4500. #set to HETDEX mid-point


                            #technically this is *WRONG* ... the wavelength should be the filter's iso wavelength
                            #and this is especially off when using 'r' band, but this is how the PLAE/POII is used
                            #(with the continuum based at the observed wavelength)
                            lam = self.w #to be consistent with the use in PLAE/POII

                            cgs_24 = SU.mag2cgs(24,lam) #24 - 25 chosen as the questionable zone for LAE vs OII
                            cgs_25 = SU.mag2cgs(25,lam)
                            cgs_26 = SU.mag2cgs(26,lam)

                            #todo: find max (fainted) mag limit and norm to that??

                            if a['fail_mag_limit']:
                                #mag is set to the (safety) mag limit
                                if a['mag'] < 24.0:
                                    #no useful information
                                    log.info(f"Combine ALL continuum: mag ({a['mag']}) at mag limit ({a['mag_limit']}) brighter than 24,"
                                             f" no useful information.")
                                    #just skip it
                                else:
                                    cont = SU.mag2cgs(a['mag_limit'], lam)
                                    #if a['mag_err'] is not None:
                                    try:
                                        #set hi to the bright limit
                                        cont_hi = SU.mag2cgs(a['mag_limit'],
                                                             lam)  # SU.mag2cgs(a['mag_bright'],lam)
                                        #set low to 1 mag fainter?
                                        cont_lo = SU.mag2cgs(a['mag_limit'] + 2.5*np.log10(1.2),  #20% error
                                                             lam)  # SU.mag2cgs(a['mag_faint'],lam)
                                        cont_var = avg_var(cont, cont_lo, cont_hi)
                                    except:
                                        #assume could be at limit or out to mag 30? mag 29?
                                        cont_hi = SU.mag2cgs(cgs_faint_limit,lam)
                                        count_lo =  SU.mag2cgs(cgs_faint_limit + 2.5*np.log10(1.2),lam) #20% error
                                        cont_var = avg_var(cont, count_lo, cont_hi)  # treat as a bogus zero error

                                    #scaled to 0-1.0 from mag24 (==0) to mag25(==1.0), linearly (prob should be more sigmoid-like)
                                    m = 1.0 / (cgs_26-cgs_24) #slope: y =  0 at cgs24 and 1 at cgs26
                                    b = 1 - m * cgs_26
                                    w = m * cont + b

                                    if w < 0:
                                        w = 0.0
                                        log.info(f"Combine ALL continuum: mag ({a['mag']}) at mag limit ({a['mag_limit']}) brigher than 24,"
                                                 f" zero weight applied.")
                                    else:
                                        w = min(1.0,w)
                                        log.info(f"Combine ALL continuum: mag ({a['mag']}) at mag limit ({a['mag_limit']}) fainter than 24,"
                                                 f" scaled weight {w} applied.")

                                    # w = min((cont - cgs_25) / (cgs_24-cgs_25),1.0)
                                    # if w < 0:
                                    #     w = 1.0
                                    #     log.info(f"Combine ALL continuum: mag ({a['mag']}) at mag limit much fainter than 24,"
                                    #              f" full weight {w} applied.")
                                    # else:
                                    #     w = max(0.1,w)
                                    #     log.info(f"Combine ALL continuum: mag ({a['mag']}) at mag limit fainter than 24,"
                                    #              f" scaled weight {w} applied.")

                                    variance.append(cont_var)
                                    continuum.append(cont)
                                    weight.append(w)
                                    cont_type.append("a"+a['filter_name'])
                                    aperture_radius = a['radius']
                                    nondetect.append(1)

                            else:
                                cont = SU.mag2cgs(a['mag'],lam)
                                if a['mag_err'] is not None:
                                    cont_hi = SU.mag2cgs(a['mag']-a['mag_err'],lam)#SU.mag2cgs(a['mag_bright'],lam)
                                    cont_lo =  SU.mag2cgs(a['mag']+a['mag_err'],lam)#SU.mag2cgs(a['mag_faint'],lam)
                                    cont_var = avg_var(cont,cont_lo,cont_hi)
                                else:
                                    cont_var = avg_var(cont, cont, cont) #treat as a bogus zero error

                                w = 1.0
                                if (a['mag_err'] is None) or (a['mag_err']==0):
                                    w = 0.2 #probably below the flux limit, so weight low
                                else:
                                    #what about ELiXer Aperture that is huge where SDSS/HETDEX gmag at limit?
                                    #
                                    # Basically, the gmag is below flux limit, and the apertures are not increasing
                                    # in mag as they grow, so we are on something faint, maybe near a bright object
                                    # so lots of sky; can't use this estimate for continuum so toss it out
                                    if gmag_at_limit and (a['sep_obj_idx'] is None) and \
                                            (a['radius'] is not None) and (a['radius'] > 2.0):
                                        #check the radii and see if there is much change in the first few
                                        if (a['elixer_apertures'] is not None) and len(a['elixer_apertures']) > 5:
                                            if np.nanstd([x['mag'] for x in a['elixer_apertures']][0:5]) < 0.1:
                                                #it is just not growing
                                                #most likely this is just sky or nearby neighbor, down weight as unreliable
                                                w = 0.0


                                #do we want to use this?
                                #if there are SEP apertures, but we are not using them
                                #and the elixer apeture is maxed out and much brighter than the DEX-g
                                #then we are likely looking at a bright object on which we are NOT centered
                                #and it is throwing off the aperture magnitude
                                if self.best_gmag_cgs_cont is not None:
                                    cont_check = max(G.HETDEX_CONTINUUM_FLUX_LIMIT,self.best_gmag_cgs_cont)
                                else:
                                    cont_check = None
                                if any_sep and (not matched_sep) and (elix_aper_radius > 2.5) and \
                                        (cont_check is not None) and (cont/cont_check > 5.0 ):
                                    #don't use this one and set the flag
                                    self.flags |= G.DETFLAG_LARGE_NEIGHBOR
                                    log.info(f"{self.entry_id} Combine ALL Continuum: {a['catalog_name']}-{a['filter_name']} Grossly inconsistent photometric aperture"
                                             f" magnitude with suggestion of bright, offset object. Excluding from consideration.")
                                else:
                                    weight.append(w)
                                    variance.append(cont_var)
                                    continuum.append(cont)
                                    cont_type.append("a" + a['filter_name'])
                                    nondetect.append(0)
                                    aperture_radius = a['radius']

                                log.debug(
                                    f"{self.entry_id} Combine ALL Continuum: Added imaging estimate ({continuum[-1]:#.4g}) "
                                    f"sd({np.sqrt(variance[-1]):#.4g}) weight({weight[-1]:#.2f}) filter({a['filter_name']})")
                except:
                    log.debug("Exception handling individual forced aperture photometry continuum in DetObj:combine_all_continuum",
                              exc_info=True)
        except:
            log.debug("Exception handling forced aperture photometry continuum in DetObj:combine_all_continuum", exc_info=True)

        #todo: ?? Best catalog match PLAE ?? #do I even want to try to use this one?
        #todo: this is a problem:
        # you can have, say 3 matches inside the aperture, one near the center --- bright, clearly OII
        # two others inside the aperture, but faint, possibly LAEs
        # even pushing down the weights, if the two faint ones have low variance, they will dominate
        #
        #todo: Furthermore, this is really a blending of probabilities ... we want the probability that our object
        # is an LAE, but *THIS* combines that with the probability that each catalog object is our object, so
        # it is not a clean probability anymore (though the invidual probabilities that a catalog object is our
        # object are independent)
        #
        #todo: So, the most correct thing to do is report each independently, or select the single best one, or
        # ignore all of them and just use the aperture (which may be best anyway since, if we do choose the best one
        # it effectively brings that magnitude estimate in twice (once for the aperture and once for the catalog object
        # that ostensibly used that (or similar) aperture

        if self.best_counterpart is not None and self.best_counterpart.bid_filter is not None:
            if self.best_counterpart.bid_filter.lower() in ['g','r','f606w']:
                weight.append(1)
                continuum.append(self.best_counterpart.bid_flux_est_cgs)
                variance.append(max( self.best_counterpart.bid_flux_est_cgs*self.best_counterpart.bid_flux_est_cgs*0.04,
                                     self.best_counterpart.bid_flux_est_cgs_unc*self.best_counterpart.bid_flux_est_cgs_unc))

                cont_type.append("c" + self.best_counterpart.bid_filter.lower())
                nondetect.append(0)
                log.debug(
                    f"{self.entry_id} Combine ALL Continuum: Added catalog bid target estimate"
                    f" ({continuum[-1]:#.4g}) sd({np.sqrt(variance[-1]):#.4g}) "
                    f"weight({weight[-1]:#.2f}) filter({self.best_counterpart.bid_filter.lower()}) dist({self.best_counterpart.distance})")

        if False: #choosing the last argument as this is a mix of probabilites and even in the "best" case is an over counting
            try:
                #which aperture filter?
                coord_dict = {} #key = catalog +_ + filter  values: ra, dec
                for a in self.aperture_details_list:
                    if (a['catalog_name'] is None) or (a['filter_name'] is None):
                        continue
                    else:
                        coord_dict[a['catalog_name']+'_'+a['filter_name']] = {'ra': a['ra'],'dec':a['dec']}

                #set the weight = 1.0 / number of possible matches
                sel = np.where(np.array([x.bid_dec for x in self.bid_target_list]) != 666)
                sel = np.where(np.array([x.distance for x in np.array(self.bid_target_list)[sel]]) < aperture_radius)

                num_cat_match = len(sel[0])

                if num_cat_match > 1:
                    catmatch_weight = 1.0/len(sel[0])
                else:
                    catmatch_weight = 1.0

                #nearest (that is not the explicit aperture position)
                catalog_target_list_distances = [x.distance for x in self.bid_target_list if x.bid_dec != 666]
                if len(catalog_target_list_distances) > 0:
                    nearest_distance = np.min(catalog_target_list_distances)

                    for b in self.bid_target_list:
                        if (b.bid_dec == 666): #this is a measured aperture, not a catalog match
                            continue

                        #if there are multiple bid targets, which one(s) to use?
                        #center could be off (large object), so distance is not ideal, nor is the prob_match
                        #maybe IF the center is INSIDE the selected ExtractedObject?

                        if b.distance < aperture_radius:
                            #"best" filter already chosen, so just use it

                            if (b.bid_flux_est_cgs is not None) and (b.bid_flux_est_cgs_unc is not None) and \
                                (b.bid_flux_est_cgs > 0) and (b.bid_flux_est_cgs_unc > 0):

                                #push down the weight as ratio of nearest_distance to the current match distance
                                #the closest gets full share of its weight and the others scale down from there
                                weight.append(catmatch_weight*nearest_distance/b.distance)

                                #set a minimum variance of ~ 20% for ground (maybe less for hubble)
                                #b.bid_flux_est_cgs_unc is the s.d. so need to square for variance
                                variance.append(max( b.bid_flux_est_cgs*b.bid_flux_est_cgs*0.04,
                                                        b.bid_flux_est_cgs_unc*b.bid_flux_est_cgs_unc))
                                continuum.append(b.bid_flux_est_cgs)
                                cont_type.append("c" + a['filter_name'])
                                nondetect.append(0)
                                log.debug(
                                    f"{self.entry_id} Combine ALL Continuum: Added catalog bid target estimate"
                                    f" ({continuum[-1]:#.4g}) sd({np.sqrt(variance[-1]):#.4g}) "
                                    f"weight({weight[-1]:#.2f}) filter({b.bid_filter}) dist({b.distance})")

                        #old

                        #just use b.bid_filter info? b.bid_flux_est_cgs, b.bid_flux_est_cgs_unc?
                        if False:
                        ##for f in b.filters:
                            if (b.catalog_name is None) or (f.filter is None):
                                continue

                            if f.filter.lower() not in ['g','r','f606w']: #should only use one ...
                                continue

                            key = b.catalog_name + "_" + f.filter.lower()

                            if key in coord_dict.keys():
                                if utils.angular_distance(b.bid_ra,b.bid_dec,coord_dict[key]['ra'],coord_dict[key]['dec']) < 1.0:
                                    if b.bid_flux_est_cgs is not None:
                                        cont = b.bid_flux_est_cgs
                                        if b.bid_flux_est_cgs_unc is not None:
                                            cont_var = b.bid_flux_est_cgs_unc * b.bid_flux_est_cgs_unc
                                            if cont_var == 0:
                                                cont_var = cont * cont
                                        else:
                                            cont_var = cont * cont

                                        weight.append(1.0)
                                        variance.append(cont_var)
                                        continuum.append(cont)

                                        cat_idx = len(continuum)

                                        log.debug(
                                            f"{self.entry_id} Combine ALL Continuum: Added catalog bid target estimate"
                                            f" ({continuum[-1]:#.4g}) sd({np.sqrt(variance[-1]):#.4g}) "
                                            f"weight({weight[-1]:#.2f}) filter({key})")
                                    break #only use one

            except:
                log.debug("Exception handling catalog bid-target continuum in DetObj:combine_all_continuum", exc_info=True)
                
        try:
            best_guess_extent = np.array(best_guess_extent)
            base_psf = np.array(base_psf)

            # this is really only meaningful IF it is the SEP aperture ... the elixer aperture can grow in-between
            # faint sources and yield a meaningless answer
            if deep_detect >= deep_non_detect:
                if (len(best_guess_extent) == len(base_psf)) and (len(base_psf) > 0):
                    if np.std(best_guess_extent) > np.min(best_guess_extent):
                        #the extents are very different ... likely due to imaging differences (ground may blend objects
                        # so SEP may give a much larger aperture) so, just use the deepest in this case
                        base_psf = base_psf[np.argmax(best_guess_maglimit)]
                        best_guess_extent = best_guess_extent[np.argmax(best_guess_maglimit)]
                        size_in_psf = best_guess_extent/base_psf
                    else:
                        size_in_psf = np.mean(best_guess_extent/base_psf) #usually only 1 or 2, so std makes no sense
                        best_guess_extent = np.mean(best_guess_extent)
                        base_psf = np.max(base_psf)

        except:
            pass


        #remove the non-detects except for the deepest or if fainter than the faintest positive detection
        try:
            nondetect=np.array(nondetect)
            sel = nondetect == 1
            if np.sum(sel) >= 1:

                log.info("Removing all non-detects other than deepest...")
                continuum = np.array(continuum)
                deepest = np.min(continuum[sel])

                try:
                    faint_detect = np.min(continuum[nondetect==0])
                except:
                    faint_detect = -9e99

                #now reselect to all detects and the deepest non-detect
                if deepest > faint_detect:
                    sel = (nondetect == 0) | (continuum==deepest)
                else:
                    log.info(f"Removing deepest non-detect {deepest:0.4g} since fainter than faintest positive detection {faint_detect:0.4g} ...")
                    sel = (nondetect == 0)

                log.info(f"Removed {np.array(cont_type)[np.invert(sel)]}")

                continuum = np.array(continuum)[sel]
                variance = np.array(variance)[sel]
                weight = np.array(weight)[sel]
                cont_type  = np.array(cont_type)[sel]
                nondetect = np.array(nondetect)[sel]
        except:
            log.info("Exception trimming nondetects from combine_all_continuum",exc_info=True)

        #sum up
        try:
            continuum = np.array(continuum)
            variance = np.array(variance)
            weight = np.array(weight)

            #clean up: don't use if sd (or sqrt variance) is larger than the actual measurement
            for i in range(len(continuum)):
                try:
                    #let it be a little larger, mostly the sampling will still work and give positive values
                    if np.sqrt(variance[i]) > (1.3 *continuum[i]):
                        if len(continuum) > 2:
                            weight[i] = 0
                            log.info(f"Error (sd) ({np.sqrt(variance[i])}): {variance[i]} too high vs continuum {continuum[i]}. Weight zeroed.")
                        else:
                            log.info(f"Warning (sd) ({np.sqrt(variance[i])}): {variance[i]} too high vs continuum {continuum[i]}, but weight kept as too few to zero out.")
                except:
                    if len(continuum) > 2:
                        weight[i] = 0
                        log.info(f"Exception checking variance {variance[i]} vs continuum {continuum[i]}. Weight zeroed.")
                    else:
                        log.info(
                            f"Exception checking variance {variance[i]} vs continuum {continuum[i]}. but weight kept as too few to zero out.")

            #remove any zero weights
            sel = np.where(weight==0)
            continuum = np.delete(continuum,sel)
            variance = np.delete(variance,sel)
            weight = np.delete(weight,sel)

            #if more than 3, use weighted biweight to down play any outliers (even if lower variance)
            if len(continuum) > 2:
                try:
                    log.debug(f"{self.entry_id} Using biweight clipping in hetdex::combin_all_continuum()...")

                    #first, regular biweight
                    continuum_hat = weighted_biweight.biweight_location(continuum)
                    continuum_sd_hat = weighted_biweight.biweight_scale(continuum)

                    #then clip (pretty aggressively)
                    diff = abs(continuum - continuum_hat)/continuum_sd_hat
                    original_continuum = copy(continuum) #for logging below
                    sigma = 1.5
                    sel = np.where(diff < sigma)

                    if len(sel[0]) > 1: #keep at least 2 of 3 (or 2 of 4)
                        continuum = continuum[sel]
                        weight = weight[sel]
                        variance = variance[sel]

                        #for logging
                        sel = np.where(diff >= 1.5)
                        if len(sel[0]) > 0:
                            log.debug(f"{self.entry_id} Removed {len(sel[0])} estimate(s) {original_continuum[sel]} sigma({diff[sel]}). (clipped at {sigma} sigma)")
                    else:
                        log.debug(f"{self.entry_id} cut too severe. Unable to remove any.")

                    #then (standard) inverse variance
                    continuum_hat = np.sum(continuum * weight / variance) / np.sum(weight / variance)
                    continuum_sd_hat = np.sqrt(np.sum(weight * variance) / np.sum(weight))


                    ## or do we just want to average as Gaussians
                    # which would be sum of mu and sum of the variances  (divided by N for mean?)

                    # continuum_hat = weighted_biweight.biweight_location_errors(continuum, errors=variance)
                    # continuum_sd_hat = weighted_biweight.biweight_scale(continuum)
                except:
                    log.debug("Exception using biweight clipping in hetdex::combine_all_contiuum(). "
                              "Switching to full array inverse variance",exc_info=True)
                    continuum_hat = np.sum(continuum * weight / variance) / np.sum(weight / variance)
                    continuum_sd_hat = np.sqrt(np.sum(weight * variance) / np.sum(weight))
            else:
                #v2 = variance*variance
                log.debug(f"{self.entry_id} Using inverse variance in hetdex::combin_all_continuum()...")
                continuum_hat = np.sum(continuum * weight / variance) / np.sum(weight / variance)
                continuum_sd_hat = np.sqrt(np.sum(weight*variance)/np.sum(weight))

            #todo: what about plae_hat uncertainty?
            #todo: should I (instead of inverse variance) sample all like 250-1000 times from random distro
            #todo: and sum only on weights, then take the mean and std, of the resulting histogram as the plae_hat and sd?


            #todo: what about extra info now for Bayesian analysis? update our "beliefs" based on
            #todo: object extent? (consistent or inconsistent with PSF and thus with a point-source?)
            #todo: slope of whole spectra or spectral features that would preclude LAE?
            #todo: ELiXer line finder strongly suggesting mutlitple lines and NOT LAE?

            log.debug(f"{self.entry_id} Combine ALL Continuum: Final estimate: continuum_hat({continuum_hat}) continuum_sd_hat({continuum_sd_hat}) "
                      f"size in psf ({size_in_psf})")

            log.debug(f"{self.entry_id} Combine ALL Continuum: Final estimate (mag @ 4500AA):"
                      f" {-2.5*np.log10(SU.cgs2ujy(continuum_hat,4500.0)/(3631.0*1e6)):0.2f} "
                      f"({-2.5*np.log10(SU.cgs2ujy(continuum_hat+continuum_sd_hat,4500.0)/(3631.0*1e6)):0.2f},"
                      f"{-2.5*np.log10(SU.cgs2ujy(continuum_hat-continuum_sd_hat,4500.0)/(3631.0*1e6)):0.2f})")


            self.classification_dict['continuum_hat'] = continuum_hat
            self.classification_dict['continuum_sd_hat'] = continuum_sd_hat
            self.classification_dict['size_in_psf'] = size_in_psf
            self.classification_dict['base_psf'] = base_psf #the PSF from which the measure came OR the max of those that contributed to the measure
            self.classification_dict['diam_in_arcsec'] = best_guess_extent

            return continuum_hat, continuum_sd_hat, size_in_psf, best_guess_extent
        except:

            log.debug("Exception handling estimation in DetObj:combine_all_continuum", exc_info=True)

        return None, None, None, None

    def multiline_solution_score(self):
        '''

        :return: bool (True) if the solution is good
                 float best solution score
        '''
        #todo: gradiate score a bit and add other conditions
        #todo: a better way of some kind rather than a by-hand scoring

        if (self.spec_obj is not None) and (self.spec_obj.solutions is not None) and (len(self.spec_obj.solutions) > 0):
            sols = self.spec_obj.solutions
            # need to tune this
            # score is the sum of the observed eq widths
            if  (self.spec_obj.solutions[0].score >= G.MULTILINE_MIN_SOLUTION_SCORE) and \
                (self.spec_obj.solutions[0].scale_score >= G.MULTILINE_MIN_SOLUTION_CONFIDENCE) and \
                (self.spec_obj.solutions[0].frac_score > 0.5):# and \
                #(len(self.spec_obj.solutions[0].lines) >= G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY):

                if (len(self.spec_obj.solutions) == 1) or \
                    ((len(self.spec_obj.solutions) > 1) and \
                      (self.spec_obj.solutions[0].score / self.spec_obj.solutions[1].score > G.MULTILINE_MIN_NEIGHBOR_SCORE_RATIO)):

                    self.multiline_z_minimum_flag = True
                    return True, self.spec_obj.solutions[0].scale_score, SU.map_multiline_score_to_confidence(self.spec_obj.solutions[0].scale_score)
                #not a clear winner, but the top is a display (or primary) line and the second is not, so use the first
                elif (self.spec_obj.solutions[0].emission_line.display is True) and \
                        (self.spec_obj.solutions[0].emission_line.display is False):

                    log.debug("multiline_solution_score, using display line over non-display line")
                    self.multiline_z_minimum_flag = True
                    return True, self.spec_obj.solutions[0].scale_score,SU.map_multiline_score_to_confidence(self.spec_obj.solutions[0].scale_score)

            if G.MULTILINE_ALWAYS_SHOW_BEST_GUESS:
                self.multiline_z_minimum_flag = False
                return False, self.spec_obj.solutions[0].scale_score,SU.map_multiline_score_to_confidence(self.spec_obj.solutions[0].scale_score)

        return False, 0.0, 0.0



    def bad_amp(self,fatal=True, fiber=None):

        #fatal default True ... stop and return; if false, delete the offending fiber but keep going
        #fiber ... apply to a single fiber, not the list of self.fibers
        #iterate through amps (or maybe just the top (highest amp?))
        #if in the bad amp list for the date range, log and return TRUE
        #log.warning("" )

        #bad_amp_dict = dictionary of lists of tuples
        #key = ifuslot  value = list that contains tuples of (start_date, end_date)

        if G.INCLUDE_ALL_AMPS:
            return False

        rc = False

        #read in the file that populates the bad_amp_dict
        if self.bad_amp_dict is None:
            self.bad_amp_dict = {}
            try:
                with open(G.BAD_AMP_LIST) as f:
                    for line in f:
                        if len(line) < 3:
                            continue
                        elif line[0] == '#':  # a comment
                            continue
                        else:  # data line
                            toks = line.split()

                            if len(toks) < 4:
                                log.warning("Unexpected line in BAD_AMP_LIST: %s" %line)
                            else:

                                start_date = None
                                stop_date = None
                                amp = None

                                if toks[1].upper() in elixer_fiber.AMP:
                                    amp = toks[1].upper()
                                else:
                                    log.warning("Unexpected AMP in BAD_AMP_LIST: %s" %line)
                                    continue


                                if toks[2].lower() != 'none':
                                    try:
                                        date = dateutil.parser.parse(toks[2])
                                        start_date = "%d%02d%02d" %(date.year,date.month,date.day)
                                    except:
                                        log.warning("Invalid date in BAD_AMP_LIST: %s" %line)
                                        continue

                                if toks[3].lower() != 'none':
                                    try:
                                        date = dateutil.parser.parse(toks[3])
                                        stop_date = "%d%02d%02d" %(date.year,date.month,date.day)
                                    except:
                                        log.warning("Invalid date in BAD_AMP_LIST: %s" %line)
                                        continue


                                key = "%s_%s" %(toks[0],amp.upper())

                                if key in self.bad_amp_dict.keys():
                                    self.bad_amp_dict[key].append((start_date,stop_date))
                                else:
                                    self.bad_amp_dict[key] = [(start_date,stop_date)]

            except:
                log.warning("Could not consume BAD_AMP_LIST %s" %G.BAD_AMP_LIST)


        if len(self.bad_amp_dict) == 0: #no bad amps listed
            return rc

        for i in range(len(self.fibers)-1,-1,-1):
            #ifuslot, ifuid, specid -- all strings
            key =  "%s_%s" %(self.fibers[i].ifuslot,self.fibers[i].amp.upper())
            if key in self.bad_amp_dict.keys():

                for t in self.bad_amp_dict[key]:
                    if (t[0] is None) or (self.fibers[i].dither_date > t[0]):
                        if (t[1] is None) or  (self.fibers[i].dither_date < t[1]):
                            if fatal:
                                log.warning("Bad amp (%s) encountered (fatal)." %key)
                                rc = True
                                break
                            else:
                                log.info("Bad amp (%s) encountered (non-fatal). Offending fiber removed." %(key))
                                del self.fibers[i]
                                rc = True

        return rc

        # rsp1 (when t5all was provided and we want to load specific fibers for a single detection)

    def load_fluxcalibrated_spectra(self):

        self.panacea = True  # if we are here, it can only be panacea

        # todo: get the line width as well

        del self.sumspec_wavelength[:]
        del self.sumspec_counts[:]
        del self.sumspec_flux[:]
        del self.sumspec_fluxerr[:]
        del self.sumspec_wavelength_zoom[:]
        del self.sumspec_counts_zoom[:]
        del self.sumspec_flux_zoom[:]
        del self.sumspec_fluxerr_zoom[:]

        if self.fcsdir is None:
            return

        # note the 20170615v09 might be v009 ... inconsistent

        if not op.isdir(self.fcsdir):
            log.debug("Cannot find flux calibrated spectra directory: " + self.fcsdir + " . Trying alternate (extra 0)")
            toks = self.fcsdir.split('v')
            toks[-1] = 'v0' + toks[-1]
            self.fcsdir = "".join(toks)

            if not op.isdir(self.fcsdir):
                # still did not find ...
                log.error("Cannot find flux calibrated spectra directory: " + self.fcsdir)
                self.status = -1
                return

        basename = op.basename(self.fcsdir)

        log.info("Loading HETDEX data (flux calibrated spectra, etc) for %s" % self.fcsdir)

        # get basic info
        # if self.annulus is None:
        file = op.join(self.fcsdir, "out2")
        try:
            with open(file, 'r') as f:
                f = ft.skip_comments(f)
                count = 0
                for l in f:  # should be exactly one line (but takes the last one if more than one)
                    count += 1
                    if count > 1:
                        log.warning("Unexepcted number of lines in %s" % (file))

                    toks = l.split()
                    self.wra = float(toks[0])
                    self.wdec = float(toks[1])
                    self.x = -999
                    self.y = -999
        except:
            if self.annulus is None:
                log.error("Fatal: Cannot read out2 file: %s" % file, exc_info=True)
                self.status = -1
                return
            else:
                self.wra = self.ra
                self.wdec = self.dec

        # get summary information (res file)
        # wavelength   wavelength(best fit) counts sigma  ?? S/N   cont(10^-17)
        # wavelength   wavelength(best fit) flux(10^-17) sigma  ?? S/N   cont(10^-17)
        if self.annulus is None:
            try:
                file = op.join(self.fcsdir, basename + "mc.res")

                if not op.exists(file):
                    log.warning("Warning. Could not load/parse: %s . Will try alternate location." % file)

                    if self.fcsdir[-1] == '/':
                        updir, _ = op.split(op.dirname(self.fcsdir))
                    else:
                        updir, _ = op.split(self.fcsdir)

                    file = op.join(updir, "fitres", basename + ".res")
                    if not op.exists(file):
                        log.warning("Warning. Could not load/parse: %s" % file)
                        file = None
            except:
                file = None
                log.warning("Warning. Could not load/parse Gaussian parm file.", exc_info=True)

            # dustin@z50:/lines/$ cat header.cat
            # ID RA DEC wavelength S/N chi^2 amplitude sigma_line continuum
            # dustin@z50:/lines/$ cat header.res
            # RA DEC wavelength S/N chi^2 amplitude sigma_line continuum

            if file is not None:
                try:
                    with open(file, 'r') as f:
                        f = ft.skip_comments(f)
                        for l in f:  # should be exactly one line (but takes the last one if more than one)
                            toks = l.split()
                            if len(toks) == 8:

                                log.info(
                                    "Using reduced info (old form) *.res file. Uncertainties not included in file.")
                                # gaussfit order: mu, sigma, A, y
                                # 0  1    2           3   4    5          6         7
                                # RA DEC wavelength S/N chi^2 amplitude sigma_line continuum

                                #line_gaussfit_parms
                                self.line_gaussfit_parms = (
                                float(toks[2]), float(toks[6]), float(toks[5])*G.FLUX_WAVEBIN_WIDTH, float(toks[7]),
                                    G.FLUX_WAVEBIN_WIDTH)
                                self.estflux = self.line_gaussfit_parms[2]/self.line_gaussfit_parms[4] * G.HETDEX_FLUX_BASE_CGS
                                self.cont_cgs = self.line_gaussfit_parms[3] * G.HETDEX_FLUX_BASE_CGS

                                self.fwhm = 2.35 * float(toks[6])

                                if self.cont_cgs <= 0:
                                    self.cont_cgs = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(float(toks[2]))
                                    self.cont_cgs_unc = 0.0  # so show no uncertainty ... a give away that this is a floor

                                if float(toks[7]) != 0:
                                    self.eqw_obs = abs(self.estflux / self.cont_cgs)
                            elif len(toks) == 14:  # xxx_mc.res version
                                #          0  1   2     3     4   5     6    7       8   9      10   11    12   13
                                # mc.res: RA DEC wave d_wave amp d_amp sigma d_sigma con d_con ston d_ston chi d_chi

                                self.wra = float(toks[0])
                                self.wdec = float(toks[1])

                                # mu, sigma, Amplitude, y, dx (see definition)
                                self.line_gaussfit_parms = (
                                float(toks[2]), float(toks[6]), float(toks[4])*G.FLUX_WAVEBIN_WIDTH, float(toks[8]),
                                    G.FLUX_WAVEBIN_WIDTH)


                                self.line_gaussfit_unc = (
                                float(toks[3]), float(toks[7]), float(toks[5])*G.FLUX_WAVEBIN_WIDTH, float(toks[9]),0.0)
                                #extra 0.0 at the end is just for symmetry ... the dx has no associated uncertainty

                                self.chi2 = float(toks[12])
                                self.chi2_unc = float(toks[13])

                                self.fwhm = 2.35 * float(toks[6])

                                self.estflux = self.line_gaussfit_parms[2]/self.line_gaussfit_parms[4] * G.HETDEX_FLUX_BASE_CGS
                                self.estflux_unc = self.line_gaussfit_unc[2] * G.HETDEX_FLUX_BASE_CGS

                                self.cont_cgs = self.line_gaussfit_parms[3] * G.HETDEX_FLUX_BASE_CGS
                                self.cont_cgs_unc = self.line_gaussfit_unc[3] * G.HETDEX_FLUX_BASE_CGS

                                if self.cont_cgs <= 0:
                                    self.cont_cgs = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(float(toks[2]))
                                    self.cont_cgs_unc = 0.0  # so show no uncertainty ... a give away that this is a floor

                                self.eqw_obs = abs(self.estflux / self.cont_cgs)
                                self.eqw_obs_unc = abs(self.eqw_obs * np.sqrt(
                                    (self.estflux_unc / self.estflux) ** 2 +
                                    (self.cont_cgs_unc / self.cont_cgs) ** 2))


                            else:
                                log.error("Error. Unexpected number of columns in %s" % file)
                except:
                    log.warning("Warning. Could not load/parse: %s" % file)

            # correction for pixels to AA (impacts only flux)
            # this section replaced by extra parameter in line_gaussfit_parms
            # if self.line_gaussfit_parms is not None:
            #     try:
            #         # have to make a list and back again since tuples are immutable
            #         parms = list(self.line_gaussfit_parms)
            #         parms[2] *= G.FLUX_WAVEBIN_WIDTH
            #         self.line_gaussfit_parms = tuple(parms)
            #
            #     except:
            #         log.warning("hetdex::load_fluxcalibrated_spectra() unable to correct pixels to AA", exc_info=True)

            # if self.line_gaussfit_unc is not None:
            #     try:
            #         parms = list(self.line_gaussfit_unc)
            #         parms[2] *= G.FLUX_WAVEBIN_WIDTH
            #         self.line_gaussfit_unc = tuple(parms)
            #     except:
            #         log.warning("hetdex::load_fluxcalibrated_spectra() unable to correct pixels to AA", exc_info=True)

            file = op.join(self.fcsdir, basename + "_2d.res")
            try:
                with open(file, 'r') as f:
                    f = ft.skip_comments(f)
                    count = 0
                    for l in f:  # should be exactly one line (but takes the last one if more than one)
                        count += 1
                        if count > 1:
                            log.warning("Unexepcted number of lines in %s" % (file))
                        toks = l.split()
                        # w = float(toks[1])  # sanity check against self.w
                        self.w = float(toks[1])

                        if self.w == 0.0:  # seeing some 2d.res files are all zeroes
                            log.error("Invalid *_2d.res file. Empty content.: %s " % file)
                            # self.status = -1
                            self.w = -1.0

                            self.estflux = -0.0001
                            self.snr = 0.01
                            self.fwhm = 0.0
                            self.cont_cgs = -0.0001
                            self.eqw_obs = -1.0
                        else:
                            # as is, way too large ... maybe not originally calculated as per angstrom? so divide by wavelength?
                            # or maybe units are not right or a miscalculation?
                            # toks2 is in counts
                            if self.estflux <= 0:
                                self.estflux = float(toks[2]) * self.sumspec_flux_unit_scale  # e.g. * 10**(-17)
                                # print("Warning! Using old flux conversion between counts and flux!!!")
                                # self.estflux = float(toks[2]) * flux_conversion(self.w)
                                self.cont_cgs = float(toks[6]) * self.sumspec_flux_unit_scale  # e.g. 10**(-17)

                                self.fwhm = 2.35 * float(
                                    toks[3])  # here sigma is the gaussian sigma, the line width, not significance
                            self.snr = float(toks[5])

                            # todo: need a floor for cgs (if negative)
                            # for now only
                            if self.cont_cgs <= 0.:
                                print("Warning! Using predefined floor for continuum")
                                self.cont_cgs = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(self.w)

                            if self.eqw_obs <= 0:
                                self.eqw_obs = self.estflux / self.cont_cgs
                                self.eqw_obs_unc = abs(self.eqw_obs * np.sqrt(
                                    (self.estflux_unc / self.estflux) ** 2 +
                                    (self.cont_cgs_unc / self.cont_cgs) ** 2))

            except:
                log.error("Cannot read *2d.res file: %s" % file, exc_info=True)

        # build the mapping between the tmpxxx files and fibers
        # l1 file
        multi = []
        idstr = []

        file = op.join(self.fcsdir, "l1")
        try:
            if G.python2():
                out = np.genfromtxt(file, dtype=None, usecols=(4, 8))
            else:
                out = np.genfromtxt(file, dtype=None, usecols=(4, 8),encoding=None)
            if out.size < 3:  # two columns ... so if the size comes back as 2, only one row in place and need to make 2d
                a = []
                a.append(out)
                out = np.array(a)

            multi = np.array(map(lambda x: x.split(".")[0], out[:, 0]))
            idstr = out[:, 1]
        except:
            log.error("Cannot read l1: %s" % file, exc_info=True)

        # if don't already have fibers, build them
        # l6 file has the paths to the multi*fits file
        # l1 file has the bulk info
        # RA, Dec, X, Y, multi(?ixy), exp, ???, ???, obs_string, obs_date, shot #
        # tmp1xx.files are in order, so first entry is tmp101.dat

        file1 = op.join(self.fcsdir, "l1")
        file6 = op.join(self.fcsdir, "l6")

        if (self.fibers is None) or (len(self.fibers) == 0):
            try:
                if G.python2():
                    out1 = np.genfromtxt(file1, dtype=None)
                    out6 = np.genfromtxt(file6, dtype=None)
                else:
                    out1 = np.genfromtxt(file1, dtype=None,encoding=None)
                    out6 = np.genfromtxt(file6, dtype=None,encoding=None)

                # if len(out6.shape) == 0: #only 1D
                # or could say if out6.size == 1:
                if out6.size == 1:
                    a = []
                    a.append(out6)
                    out6 = np.array(a)

                # if len(out1.shape) == 0:
                if out1.size == 1:
                    a = []
                    a.append(out1)
                    out1 = np.array(a)

                if len(out6) != len(out1):
                    # if out6.size != out1.size: #better since out1, out6 are ndarrays
                    log.error("Error. Unmatched file lengths for l1 and l6 under %s" % (self.fcsdir))
                    return

                for i in range(len(out6)):
                    ra = float(out1[i][0])
                    dec = float(out1[i][1])
                    sky_x = float(out1[i][2])  # IFU x,y location of this fiber
                    sky_y = float(out1[i][3])
                    multi_name = out1[i][4]
                    exp = int(out1[i][5].lstrip("exp"))
                    time_ex = out1[i][8]
                    ymd = out1[i][9]
                    shot_num = out1[i][10]

                    toks = multi_name.split("_")  # example "multi_038_096_014_RL_041.ixy"
                    specid = toks[1]  # string
                    ifu_slot = toks[2]
                    ifuid = toks[3]
                    side = toks[4]
                    fib_id = toks[5].split(".")[0]

                    # 20171220T094210.0_020_095_004_RU_089
                    idstring = time_ex + "_" + specid + "_" + ifu_slot + "_" + ifuid + "_" + side + "_" + fib_id

                    # just build up from the idstring
                    fiber = elixer_fiber.Fiber(idstring)
                    if fiber is not None:
                        fiber.ra = ra
                        fiber.dec = dec
                        fiber.obsid = int(shot_num)
                        fiber.expid = exp
                        fiber.detect_id = self.id
                        fiber.center_x = sky_x
                        fiber.center_y = sky_y
                        # add the fiber (still needs to load its fits file)
                        # we already know the path to it ... so do that here??
                        fiber.fits_fn = out6[i]

                        if self.annulus is None:
                            fits = hetdex_fits.HetdexFits(fiber.fits_fn, None, None, exp - 1, panacea=True)
                            fits.obs_date = fiber.dither_date
                            fits.obs_ymd = fits.obs_date
                            fits.obsid = fiber.obsid
                            fits.expid = fiber.expid
                            fits.amp = fiber.amp
                            fits.side = fiber.amp[0]
                            fiber.fits = fits
                            # self.sci_fits.append(fits)

                        # not here ... this is for the entire detection not just this fiber
                        # self.wra = ra
                        # self.wdec = dec
                        # self.x = sky_x
                        # self.y = sky_y

                        self.fibers.append(fiber)
            except:
                log.error("Cannot read l1,l6 files to build fibers: %s" % self.fcsdir, exc_info=True)

            # build fibers from this directory (and get the multi*fits files into the list)
            # there is a file that links all this together
            # basically, go through the fibers in the directory
            # if the associated multi*fits file is not already in the list, append it
            # construct the fiber and append to the list

        # todo: get the tmpxxx.dat files for the relative through put for each fiber (by wavelength)
        # these are the spectra cutout sections .. so need to weight each wavelength for each fiber
        # in the cutout
        # maybe add another array ... relative throughput

        # get the weights
        try:

            # this file may not be there ... if we are doing an annulus, that does not matter
            if self.annulus is None:
                file = op.join(self.fcsdir, "list2")
                # just a simple check on the length
                try:

                    if G.python2():
                        tempw = np.genfromtxt(file, dtype=np.str, usecols=0)
                    else:
                        tempw = np.genfromtxt(file, dtype=np.str, usecols=0,encoding=None)
                    # if len(tempw.shape) == 0:
                    if tempw.size == 1:
                        a = []
                        a.append(tempw)
                        tempw = np.array(a)

                    if len(tempw) != len(multi):
                        log.error("Cannot match up list2 and l1 files")
                        good = False
                    else:
                        good = True
                except:
                    log.error("Cannot load list2")
                    good = False
            else:
                good = True

            if good:
                # tmp = out[:,0]
                if self.annulus is None:
                    out = np.loadtxt(file, dtype=np.float, usecols=(1, 2))

                    if out.size < 3:  # two columns ... so if the size comes back as 2, only one row in place and need to make 2d
                        a = []
                        a.append(out)
                        out = np.array(a)

                    keep = out[:, 0]
                    w = out[:, 1]  # weights
                else:
                    keep = np.zeros(len(self.fibers))
                    w = np.zeros(len(keep))  # weights

                # sum up the weights where keep == 0
                norm = np.sum(w[np.where(keep == 0)])
                subset_norm = 0.0
                # max_weight = -999.9

                # todo: if only using a subset of fibers (e.g. from a line file), how do we interpret the weights?
                # as here, as a subset??

                # not optimal for rsp1 organization, but if a line file was provided and the source of the fibers
                # is not from rsp1, this is necessary (and it works either way ... and there are usually just a few
                # fibers, so not that bad)
                for f in self.fibers:
                    f.relative_weight = 0.0

                    for i in range(len(out)):
                        try:
                            if keep[i] == 0.:
                                # find which fiber, if any, in set this belongs to
                                if (f.multi == multi[i]) and (f.scifits_idstring == idstr[i]):
                                    f.relative_weight += w[
                                        i]  # usually there is only one, so this is 0 + weight = weight
                                    subset_norm += w[i]

                                    # if w[i] > max_weight:
                                    #    max_weight = w[i]

                                    # now, get the tmp file for the thoughput
                                    tmpfile = op.join(self.fcsdir, "tmp" + str(i + 101) + ".dat")

                                    log.info("Loading spectrum from: %s" % tmpfile)

                                    tmp_out = np.loadtxt(tmpfile, dtype=np.float)
                                    cols = np.shape(tmp_out)[1]
                                    # 0 = wavelength, 1=counts?, 2=flux? 3,4,5 = throughput 6=cont?

                                    # due to call to rsp1a2, might not be 3500 to 5500 and there can be junk
                                    # on either end, so slice the arrays read in
                                    mn_idx = elixer_spectrum.getnearpos(tmp_out[:, 0], 3500.0)
                                    mx_idx = elixer_spectrum.getnearpos(tmp_out[:, 0], 5500.0)

                                    f.fluxcal_central_emis_wavelengths = tmp_out[:, 0][mn_idx:mx_idx + 1]
                                    if f.fluxcal_central_emis_wavelengths[0] == f.fluxcal_central_emis_wavelengths[1]:
                                        log.warning(
                                            "Invalid wavelengths in hetdex::DetObj::load_flux_calibrated_spectra. Skipping ...")
                                        f.fluxcal_central_emis_wavelengths = []
                                        continue

                                    f.fluxcal_central_emis_counts = tmp_out[:, 1][mn_idx:mx_idx + 1]
                                    f.fluxcal_central_emis_flux = tmp_out[:, 2][mn_idx:mx_idx + 1]

                                    f.fluxcal_central_emis_thru = (tmp_out[:, 3][mn_idx:mx_idx + 1]) * \
                                                                  (tmp_out[:, 4][mn_idx:mx_idx + 1]) * \
                                                                  (tmp_out[:, 5][mn_idx:mx_idx + 1])

                                    if cols == 9:
                                        try:
                                            f.fluxcal_central_emis_fluxerr = tmp_out[:, 8][mn_idx:mx_idx + 1]
                                        except:
                                            log.error("Exception loading fluxerr.", exc_info=True)
                                            f.fluxcal_central_emis_fluxerr = []
                                    elif cols == 7:
                                        try:
                                            f.fluxcal_central_emis_fluxerr = tmp_out[:, 6][mn_idx:mx_idx + 1]
                                        except:
                                            log.error("Exception loading fluxerr.", exc_info=True)
                                            f.fluxcal_central_emis_fluxerr = []
                                    else:
                                        log.warning("Warning! Unexpected number of columns (%d) in tmpxxx.dat." % cols)
                                    # f.fluxcal_emis_cont = tmp_out[:,6][mn_idx:mx_idx+1]

                                    # sanity check ... could be all zeros
                                    if np.max(f.fluxcal_central_emis_flux) == 0:  # prob all zero
                                        log.info("Clearing all fiber with all zero flux values ...")
                                        # would be best to remove it
                                        f.clear(bad=True)

                            # else:
                            #    # find which fiber, if any, in set this belongs to
                            #    if (f.multi == multi[i]) and (f.scifits_idstring == idstr[i]):
                            #        f.relative_weight += 0.0
                        except:
                            log.error("Error loading fiber from %s" % tmpfile, exc_info=True)
                            if self.annulus is None:
                                log.error("Fatal. Cannot load flux calibrated spectra", exc_info=True)
                                print("Fatal. Cannot load flux calibrated spectra")
                                self.status = -1
                                return

                    # f.relative_weight /= norm #note: some we see are zero ... they do not contribute

                # subselect only fibers without all zero fluxes
                select = np.where(np.array([f.bad for f in self.fibers]) == False)
                self.fibers = list(np.array(self.fibers)[select])

                if self.annulus is None:
                    for f in self.fibers:
                        if subset_norm != 0:
                            f.relative_weight /= subset_norm
                        # f.relative_weight /= max_weight

                # todo: how is Karl using the weights (sum to 100% or something else?)
                # now, remove the zero weighted fibers, then sort
                if self.annulus is None:
                    self.fibers = [x for x in self.fibers if x.relative_weight > 0]
                self.fibers.sort(key=lambda x: x.relative_weight, reverse=True)  # highest weight is index = 0
                self.fibers_sorted = True

        except:
            log.error("Fatal. Cannot read list2: %s" % file, exc_info=True)
            print("Fatal. Cannot read list2: %s" % file)
            self.status = -1
            return

        # avg_ra = 0.0
        # avg_dec = 0.0
        # for f in self.fibers:
        #    avg_ra += f.ra * f.relative_weight
        #    avg_dec += f.dec * f.relative_weight

        # get the full flux calibrated spectra
        if self.annulus is None:
            file = op.join(self.fcsdir, basename + "specf.dat")
            try:
                size = os.stat(file).st_size
                if size == 0:
                    print("eid(%s) *specf.res file is zero length (no VIRUS data available?): %s" % (
                    str(self.entry_id), file))
                    log.error("eid(%s) *specf.res file is zero length (no VIRUS data available?): %s" % (
                    str(self.entry_id), file))
                    self.status = -1
                    return
            except:
                pass

            # 2018-08-30 new order in specf and spece is to be removed
            # wavelength flux flux_err counts counts_err       with flux and flux_err in cgs x10^-17
            try:
                out = np.loadtxt(file, dtype=None)

                self.sumspec_wavelength = out[:, 0]

                if self.sumspec_wavelength[0] == self.sumspec_wavelength[1]:
                    # this happens when there is rsp output, but there were no fibers actually found
                    print("Invalid wavelengths reported")
                    log.error("Invalid wavelengths in hetdex::DetObj::load_flux_calibrated_spectra.")
                    self.status = -1
                    return

                col1 = out[:, 1]
                col2 = out[:, 2]
                col3 = out[:, 3]
                col4 = out[:, 4]

                # there are various old version out there ... try to sort them out
                if (max(col2) < 0.00001) and (max(col4) <= 1.):  # <1= since might not have error
                    # wave cts flux(e-17) cts_err flux_err(e-17)
                    self.sumspec_flux = col2 * 1e17
                    if max(col4) < 1.:
                        self.sumspec_fluxerr = col4 * 1e17
                    self.sumspec_counts = col1
                    # not using col3, counts_err
                else:  # current newest version 2018-08-30
                    self.sumspec_flux = col1
                    self.sumspec_fluxerr = col2

                    # sometimes have the scientific notation (though this should no longer be the case)
                    if max(self.sumspec_flux) < 0.00001:
                        self.sumspec_flux *= 1e17

                    if max(self.sumspec_fluxerr) < 0.00001:
                        self.sumspec_fluxerr *= 1e17

                    self.sumspec_counts = col3  # still using for upper right zoomed in cutout
                    # self.sumspec_counts_err = out[:, 4] #not using these in elixer anymore

                # get the zoomed in part (slice around the central wavelength)

                if self.w is not None and self.w != 0:
                    idx = elixer_spectrum.getnearpos(self.sumspec_wavelength, self.w)
                else:
                    try:
                        idx = np.argmax(self.sumspec_flux)
                    except:
                        idx = 0

                left = idx - 25  # 2AA steps so +/- 50AA
                right = idx + 25

                if left < 0:
                    left = 0
                if right > len(self.sumspec_flux):
                    right = len(self.sumspec_flux)

                # these are on the 2AA grid (old spece had 2AA steps but, the grid was centered on the main wavelength)
                # this grid is not centered but is on whole 2AA (i.e. 3500.00, 3502.00, ... not 3500.4192, 3502.3192, ...)
                self.sumspec_wavelength_zoom = self.sumspec_wavelength[left:right]
                self.sumspec_flux_zoom = self.sumspec_flux[left:right]
                self.sumspec_fluxerr_zoom = self.sumspec_fluxerr[left:right]
                self.sumspec_counts_zoom = self.sumspec_counts[left:right]


            except:
                log.error("Fatal. Cannot read *specf.dat file: %s" % file, exc_info=True)
                print("Fatal. Cannot read *specf.dat file: %s" % file)
                self.status = -1
                return

            # try:
            #     #                (0)    (1)              (2)                  (3)     (4)
            #     #could be (new) wave, flux (cgs/1e-17), flux_err (cgs/1e-17), counts, counts_err
            #     #or       (old) wave, counts, flux (cgs/1e-17), counts_err, flux_err (cgs/1e-17)
            #     #     AND flux may or may not have the E-17 notation
            #
            #
            #     out = np.loadtxt(file, dtype=None)
            #
            #     self.sumspec_wavelength = out[:,0]
            #
            #     if self.sumspec_wavelength[0] == self.sumspec_wavelength[1]:
            #         print("Invalid wavelengths reported")
            #         log.error("Invalid wavelengths in hetdex::DetObj::load_flux_calibrated_spectra.")
            #         self.status = -1
            #         return
            #
            #     self.sumspec_counts = out[:, 1]
            #
            #     if max(self.sumspec_counts) < 1.0: # new order
            #         self.sumspec_flux = self.sumspec_counts * 1e17
            #         self.sumspec_fluxerr = out[:, 2]
            #         self.sumspec_counts = out[:, 3]
            #     else: #old order
            #         self.sumspec_flux = out[:, 2]
            #         if max(self.sumspec_flux) < 1:
            #             self.sumspec_flux *= 1e17
            #             self.sumspec_fluxerr = out[:, 4]
            #         else:
            #             self.sumspec_fluxerr = out[:, 4]
            #
            #     if abs(np.max(self.sumspec_fluxerr)) < 0.00001: #can assume has the e-17 or e-18 notation
            #         self.sumspec_fluxerr *= 1e17
            #
            #     #reminder data scientific notation, so mostly e-17 or e-18
            #
            #     #get flux error (not yet in this file)
            #     #self.sumspec_fluxerr = out[:,6]  * 1e17
            #     #self.sumspec_fluxerr = np.full_like(self.sumspec_flux,1.0) #i.e. 0.5x10^-17 cgs
            #     #np.random.seed(1138)
            #     #self.sumspec_fluxerr = np.random.random(len(self.sumspec_flux)) # just for test
            #
            # except:
            #     log.error("Fatal. Cannot read *specf.dat file: %s" % file, exc_info=True)
            #     print("Fatal. Cannot read *specf.dat file: %s" % file)
            #     self.status = -1
            #     return

            # get the zoomed in flux calibrated spectra
            #                (0)    (1)              (2)                  (3)     (4)
            # could be (new) wave, flux (cgs/1e-17), flux_err (cgs/1e-17), counts, counts_err
            # or       (old) wave, counts, flux (cgs/1e-17), counts_err, flux_err (cgs/1e-17)
            #     AND flux may or may not have the E-17 notation

            if False:
                file = op.join(self.fcsdir, basename + "spece.dat")
                try:
                    out = np.loadtxt(file, dtype=None)

                    # self.sumspec_wavelength_zoom = out[:, 0]
                    # self.sumspec_counts_zoom = out[:, 1]
                    # self.sumspec_flux_zoom = out[:, 2]  * 1e17
                    # #todo: get flux error (not yet in this file)
                    # #self.sumspec_fluxerr_zoom = out[:,6]  * 1e17
                    # #self.sumspec_fluxerr_zoom = np.full_like(self.sumspec_flux_zoom, 0.5)  # i.e. 0.5x10^-17 cgs

                    self.sumspec_wavelength_zoom = out[:, 0]
                    if self.sumspec_wavelength_zoom[0] == self.sumspec_wavelength_zoom[1]:
                        print("Invalid wavelengths reported")
                        log.error("Invalid wavelengths in hetdex::DetObj::load_flux_calibrated_spectra.")
                        self.status = -1
                        return

                    self.sumspec_counts_zoom = out[:, 1]

                    if max(self.sumspec_counts_zoom) < 1.0:  # new order
                        self.sumspec_flux_zoom = self.sumspec_counts_zoom * 1e17
                        self.sumspec_fluxerr_zoom = out[:, 2]  # * 1e17
                        self.sumspec_counts_zoom = out[:, 3]
                    else:  # old order
                        self.sumspec_flux_zoom = out[:, 2]
                        if max(self.sumspec_flux_zoom) < 1:
                            self.sumspec_flux_zoom *= 1e17
                            self.sumspec_fluxerr_zoom = out[:, 4]  # * 1e17
                        else:
                            self.sumspec_fluxerr_zoom = out[:, 4]

                    if abs(np.max(self.sumspec_fluxerr_zoom)) < 0.00001:  # can assume has the e-17 or e-18 notation
                        self.sumspec_fluxerr_zoom *= 1e17

                except:  # no longer consider this fatal. Using the specf data, but will use spece if it is there
                    pass
                    # log.error("Cannot read *_spece.res file: %s" % file, exc_info=True)
                    # print("Cannot read *_spece.res file: %s" % file)
                    # self.status = -1
                    # return

            # get zoomed 2d cutout
            # fits file
            file = op.join(self.fcsdir, basename + ".fits")
            try:
                f = pyfits.open(file)
                self.sumspec_2d_zoom = f[0].data
                f.close()
            except:
                log.warning("Warning (not fatal). Could not read file " + file)  # , exc_info=True) #not fatal

        # check the fiber(s) for bad amps ... if found, we're done ... discontinue
        if self.annulus is None:
            if self.bad_amp(fatal=True):
                self.status = -1
                log.warning("Warning (fatal). Bad amp(s) in the fiber list.")
                print("Warning (fatal). Bad amp(s) in the fiber list.")
                return
        else:  # annulus case, just remove the offending fibers
            self.bad_amp(fatal=False)
            if len(self.fibers) == 0:
                log.warning("Warning (fatal). Bad amp(s) in the fiber list. Number of remaining fibers below minimum.")
                print("Warning (fatal). Bad amp(s) in the fiber list. Number of remaining fibers below minimum.")
                self.status = -1
                return

        # set_spectra(self, wavelengths, values, errors, central, estflux=None, eqw_obs=None)
        # self.spec_obj.set_spectra(self.sumspec_wavelength,self.sumspec_counts,self.sumspec_fluxerr,self.w)
        self.spec_obj.identifier = "eid(%s)" % str(self.entry_id)
        self.spec_obj.plot_dir = self.outdir

        if self.annulus is None:
            self.spec_obj.set_spectra(self.sumspec_wavelength, self.sumspec_flux, self.sumspec_fluxerr, self.w,
                                      values_units=-17, estflux=self.estflux, estflux_unc=self.estflux_unc,
                                      eqw_obs=self.eqw_obs, eqw_obs_unc=self.eqw_obs_unc,
                                      continuum_g=self.best_gmag_cgs_cont,continuum_g_unc=self.best_gmag_cgs_cont_unc)
            # print("DEBUG ... spectrum peak finder")
            # if G.DEBUG_SHOW_GAUSS_PLOTS:
            #    self.spec_obj.build_full_width_spectrum(show_skylines=True, show_peaks=True, name="testsol")
            # print("DEBUG ... spectrum peak finder DONE")

            #update DEX-g based continuum and EW
            try:
                self.best_gmag_cgs_cont *= self.spec_obj.gband_continuum_correction()
                self.best_gmag_cgs_cont_unc *= self.spec_obj.gband_continuum_correction()

                self.best_eqw_gmag_obs = self.estflux / self.best_gmag_cgs_cont
                self.best_eqw_gmag_obs_unc = abs(self.best_eqw_gmag_obs * np.sqrt(
                    (self.estflux_unc / self.estflux) ** 2 +
                    (self.best_gmag_cgs_cont_unc / self.best_gmag_cgs_cont) ** 2))

                log.info(f"Update best DEX-g continuum x{self.spec_obj.gband_continuum_correction():0.2f} and EW; cont {self.best_gmag_cgs_cont} +/- {self.best_gmag_cgs_cont_unc}" )
            except:
                log.error("Exception! Excpetion updating DEX-g continuum.",exc_info=True)

            # update with MY FIT results?
            if G.REPORT_ELIXER_MCMC_FIT or self.eqw_obs == 0:
                log.info("Using ELiXer MCMC Fit for line flux, continuum, EW, and SNR")
                try:
                    self.estflux = self.spec_obj.central_eli.mcmc_line_flux
                    self.eqw_obs = self.spec_obj.central_eli.mcmc_ew_obs[0]
                    self.cont_cgs = self.spec_obj.central_eli.mcmc_continuum
                    # self.snr = self.spec_obj.central_eli.mcmc_snr
                    self.snr = self.spec_obj.central_eli.snr
                    self.snr_unc = self.spec_obj.central_eli.snr_err

                    self.spec_obj.estflux = self.estflux
                    self.spec_obj.eqw_obs = self.eqw_obs

                    # self.estflux = self.spec_obj.central_eli.line_flux
                    # self.cont = self.spec_obj.central_eli.cont
                    # self.eqw_obs = self.estflux / self.cont
                    # self.snr = self.spec_obj.central_eli.snr
                except:
                    log.warning("No MCMC data to update core stats in hetdex::load_flux_calibrated_spectra")

            self.spec_obj.classify(known_z=self.known_z)  # solutions can be returned, also stored in spec_obj.solutions
            self.rvb = SU.red_vs_blue(self.w, self.sumspec_wavelength,
                                      self.sumspec_flux / G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS,
                                      self.sumspec_fluxerr / G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS, self.fwhm)
        else:
            self.syn_obs = elixer_observation.SyntheticObservation()
            if self.wra:
                self.syn_obs.ra = self.wra
                self.syn_obs.dec = self.wdec
            else:
                self.syn_obs.ra = self.ra
                self.syn_obs.dec = self.dec
            self.syn_obs.target_wavelength = self.target_wavelength
            self.syn_obs.annulus = self.annulus
            self.syn_obs.w = self.target_wavelength
            self.syn_obs.survey_shotid = self.survey_shotid

            #get all the fibers inside the outer annulus radius
            self.syn_obs.get_aperture_fibers()


    # end load_flux_calibrated_spectra

    def load_hdf5_shot_info(self,hdf5_fn,shotid):
        try:
            id = np.int64(copy(shotid))
        except:
            log.error(f"Exception converting shotid {shotid} to int type",exc_info=True)
            msg = "+++++ %s" %str(shotid)
            log.error(msg)
            self.status = -1
            return

        log.debug("Loading shot info from HDF5 ...")

        with tables.open_file(hdf5_fn, mode="r") as h5_survey:
            survey = h5_survey.root.Survey

            try:
                rows = survey.read_where("shotid==id")
            except:
                log.error("Exception in hetdex::DetObj::load_hdf5_shot_info reading rows from Survey table",
                          exc_info=True)
                rows = None

            if (rows is None) or (rows.size != 1):
                log.error(f"Problem loading info for shot {shotid} from {hdf5_fn}. Setting out of range values for shot.")
                self.survey_fwhm = 999
                self.survey_fwhm_moffat = 999
                self.survey_response = 0
                self.survey_fieldname = "---"

                return

            row = rows[0] #should only be the one row

            #fill out shot info
            self.survey_shotid = row['shotid'] #redundant, already have it

            try: #new in HDR2
                self.survey_fwhm = row['fwhm_virus']
            except:
                try: #older HDR1
                    self.survey_fwhm_gaussian = row['fwhm_gaussian']
                except:
                    pass

            try:
                self.survey_fwhm_moffat = row['fwhm_moffat']
            except:
                pass

            self.survey_response = row['response_4540']
            try:
                self.survey_fieldname = row['field'].decode()
            except:
                self.survey_fieldname = row['field']

            self.dither_norm = -1.0
            #self.dither_norm_high_expid = -1
            try:
                relflux_virus = row['relflux_virus']
                self.dither_norm = np.max(relflux_virus) / np.min(relflux_virus)
               # self.dither_norm_high_expid = np.argmax(relflux_virus)
            except:
                self.dither_norm = -1.0

            #relflux_virus

        return



    def find_ccd_adjacent_fibers(self):
        """
        Iterate over the self.fibers list of fibers and build up self.ccd_adjacent_fibers,
        by shot and exposure

        :return: list of dictionaries with basic fiber info and flux and error
        """

        #return self.dummy_find_ccd_adjacent_fibers()

        adjacent_fibers = []
        adjacent_amp_fits = []

        all_shots = np.unique(np.array([f.shotid for f in self.fibers]))
        for shot in all_shots:
            fibers_in_shot = np.array(self.fibers)[np.where([f.shotid == shot for f in self.fibers])]
            all_exposures = np.unique([f.expid for f in fibers_in_shot])
            for exp in all_exposures:
                _fibers = np.array(fibers_in_shot)[np.where([f.expid == exp for f in fibers_in_shot])]
                if len(_fibers) == 0:
                    continue

                base_multiframe = _fibers[0].multi[:-6] #strip off the amp and fiber

                #IFU info for all is the same now (amp could be different, but IFUID, SLOTID, etc same)

                #first build detection fibers
                det_fiber_nums = np.unique(np.array([f.number_in_ccd for f in self.fibers]))
                adjacent_fiber_numbers = []
                for fibnum in det_fiber_nums:
                    plus_one = fibnum + 1
                    minus_one = fibnum - 1

                    if not (plus_one in det_fiber_nums) and not (plus_one in adjacent_fiber_numbers) and (plus_one < 449):
                        #adjacent fiber
                        f112, amp = elixer_fiber.ccd_fiber_number_to_amp(plus_one)
                        adj_multiframe = base_multiframe + amp

                        #get the calfib, calfibe, ffsky_calfib from _fibers[x].fits
                        #will need to match to the multiframe and use f112-1 as the index
                        all_fits = [f.fits for f in _fibers if f.fits.multiframe == adj_multiframe]

                        if len(all_fits) == 0:
                            #try the adjacent_fits
                            all_fits = [f for f in adjacent_amp_fits if f.multiframe == adj_multiframe]

                        #there can be multiples returned, but they all point to the same fits
                        if len(all_fits) == 0: #we don't have this one already (probably moved to another amp)
                            try:
                                #so, go get this one
                                #must be on the same CCD so get a fits we have already loaded
                                #(just match w/o the amp)
                                all_fits = [f.fits for f in _fibers if f.fits.multiframe[:-3] == adj_multiframe[:-3]]
                                if len(all_fits) == 0:
                                    #this should not ever happen, but just log and move on
                                    log.warning("Unexpected! unable to find CCD in hetdex::DetObj::find_ccd_adjacent_fibers")
                                    continue

                                #we have at least on on the same CCD
                                fits = hetdex_fits.HetdexFits(empty=True)
                                fits.filename = all_fits[0].filename  # mfits_name #todo: fix to the corect path
                                fits.multiframe = adj_multiframe
                                fits.panacea = True
                                fits.hdf5 = True
                                fits.obsid = all_fits[0].obsid
                                fits.expid = all_fits[0].expid
                                fits.specid = all_fits[0].specid
                                fits.ifuslot = all_fits[0].ifuslot
                                fits.ifuid = all_fits[0].ifuid
                                fits.amp = amp
                                fits.side = amp[0]

                                #now read
                                fits.read_hdf5()
                                if fits.okay:
                                    all_fits = [fits]
                                    adjacent_amp_fits.append(fits)
                                else:
                                    continue

                            except:
                                continue
                        try:
                            calfib = all_fits[0].calfib[f112-1]
                            calfibe = all_fits[0].calfibe[f112-1]
                            ffsky_calfib = all_fits[0].ffsky_calfib[f112 - 1] #number -1 == index
                        except:
                            log.warning(f"Unable to collect CCD-adjacent fiber info for {adj_multiframe} {f112}")
                            calfib = []
                            calfibe = []
                            ffsky_calfib = []

                        adjacent_fibers.append({"shotid":shot,"expid":exp,
                                                "f448":plus_one,"f112":f112,
                                                "amp":amp,"multiframe": adj_multiframe,
                                                "calfib":calfib,"calfibe":calfibe,"ffsky_calfib":ffsky_calfib})
                        adjacent_fiber_numbers.append(plus_one)

                    if not (minus_one in det_fiber_nums) and not (minus_one in adjacent_fiber_numbers) and (minus_one > 0):
                        # adjacent fiber
                        f112, amp = elixer_fiber.ccd_fiber_number_to_amp(minus_one)
                        adj_multiframe = base_multiframe + amp

                        # get the calfib, calfibe, ffsky_calfib from _fibers[x].fits
                        # will need to match to the multiframe and use f112-1 as the index
                        all_fits = [f.fits for f in _fibers if f.fits.multiframe == adj_multiframe]
                        # there can be multiples returned, but they all point to the same fits

                        if len(all_fits) == 0:
                            #try the adjacent_fits
                            all_fits = [f for f in adjacent_amp_fits if f.multiframe == adj_multiframe]

                        #there can be multiples returned, but they all point to the same fits
                        if len(all_fits) == 0: #we don't have this one already (probably moved to another amp)
                            try:
                                #so, go get this one
                                #must be on the same CCD so get a fits we have already loaded
                                #(just match w/o the amp)
                                all_fits = [f.fits for f in _fibers if f.fits.multiframe[:-3] == adj_multiframe[:-3]]
                                if len(all_fits) == 0:
                                    #this should not ever happen, but just log and move on
                                    log.warning("Unexpected! unable to find CCD in hetdex::DetObj::find_ccd_adjacent_fibers")
                                    continue

                                #we have at least on on the same CCD
                                fits = hetdex_fits.HetdexFits(empty=True)
                                fits.filename = all_fits[0].filename  # mfits_name #todo: fix to the corect path
                                fits.multiframe = adj_multiframe
                                fits.panacea = True
                                fits.hdf5 = True
                                fits.obsid = all_fits[0].obsid
                                fits.expid = all_fits[0].expid
                                fits.specid = all_fits[0].specid
                                fits.ifuslot = all_fits[0].ifuslot
                                fits.ifuid = all_fits[0].ifuid
                                fits.amp = amp
                                fits.side = amp[0]

                                #now read
                                fits.read_hdf5()
                                if fits.okay:
                                    all_fits = [fits]
                                    adjacent_amp_fits.append(fits)
                                else:
                                    continue

                            except:
                                continue
                        try:
                            calfib = all_fits[0].calfib[f112 - 1]
                            calfibe = all_fits[0].calfibe[f112 - 1]
                            ffsky_calfib = all_fits[0].ffsky_calfib[f112 - 1]
                        except:
                            log.warning(f"Unable to collect CCD-adjacent fiber info for {adj_multiframe} {f112}")
                            calfib = []
                            calfibe = []
                            ffsky_calfib = []

                        adjacent_fibers.append({"shotid":shot, "expid":exp,
                                                "f448":minus_one, "f112":f112,
                                                "amp":amp,"multiframe": adj_multiframe,
                                                "calfib":calfib,"calfibe":calfibe,"ffsky_calfib":ffsky_calfib})


                        adjacent_fiber_numbers.append(minus_one)

        return adjacent_fibers



    def dummy_find_ccd_adjacent_fibers(self):
        """
        Iterate over the self.fibers list of fibers and build up self.ccd_adjacent_fibers,
        by shot and exposure

        :return: list of dictionaries with basic fiber info and flux and error
        """
        adjacent_fibers = []

        all_shots = np.unique(np.array([f.shotid for f in self.fibers]))

        for shot in all_shots:
            fibers_in_shot = np.array(self.fibers)[np.where([f.shotid == shot for f in self.fibers])]
            all_exposures = np.unique([f.expid for f in fibers_in_shot])
            for exp in all_exposures:
                _fibers = np.array(fibers_in_shot)[np.where([f.expid == exp for f in fibers_in_shot])]
                if len(_fibers) == 0:
                    continue

                base_multiframe = _fibers[0].multi[:-6] #strip off the amp and fiber

                #IFU info for all is the same now (amp could be different, but IFUID, SLOTID, etc same)

                #first build detection fibers
                det_fiber_nums = np.unique(np.array([f.number_in_ccd for f in self.fibers]))
                adjacent_fiber_numbers = []
                amp = "LU"
                adj_multiframe = base_multiframe + "LU"

                # get the calfib, calfibe, ffsky_calfib from _fibers[x].fits
                # will need to match to the multiframe and use f112-1 as the index
                all_fits = [f.fits for f in _fibers if f.fits.multiframe == adj_multiframe]
                for fibnum in np.arange(0,112):
                    plus_one = fibnum
                    f112 = fibnum

                    try:
                        calfib = all_fits[0].calfib[fibnum]
                        calfibe = all_fits[0].calfibe[fibnum]
                        ffsky_calfib = all_fits[0].ffsky_calfib[fibnum]  # number -1 == index
                    except:
                        log.warning(f"Unable to collect CCD-adjacent fiber info for {adj_multiframe} {fibnum}",exc_info=True)
                        calfib = []
                        calfibe = []
                        ffsky_calfib = []

                    adjacent_fibers.append({"shotid": shot, "expid": exp,
                                            "f448": plus_one, "f112": f112,
                                            "amp": amp, "multiframe": adj_multiframe,
                                            "calfib": calfib, "calfibe": calfibe, "ffsky_calfib": ffsky_calfib})

        return adjacent_fibers


    def calc_ccd_adjacent_fiber_magnitudes(self,fiber_dict_array,ffsky=False):
        """

        :param fiber_dict_array:
        :param ffsky:
        :return:
        """

        if ffsky:
            flux_key = 'ffsky_calfib'
        else:
            flux_key = 'calfib'
        bright_mag = 99.9
        try:
            for f in fiber_dict_array:
                try:
                    f["gmag"], f["cgs_cont"], f["gmag_unc"], f["cgs_cont_unc"] = \
                                    elixer_spectrum.get_hetdex_gmag(f[flux_key] / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                                    G.CALFIB_WAVEGRID,
                                                                    f['calfibe'] / 2.0 * G.HETDEX_FLUX_BASE_CGS)
                except:
                    f["gmag"] = 99.9
                    f["cgs_cont"] = 0.0
                    f["gmag_unc"] = 0.0
                    f["cgs_cont_unc"] = 0.0
        except:
            return None, None

        #and find the brightest
        try:
            #cgs is always populated, but gmag might not be
            all_cgs = [f["cgs_cont"] for f in fiber_dict_array]
            bright_mag = -2.5 * np.log10(SU.cgs2ujy(max(all_cgs),4500)/(3631*1e6))

            # for f in fiber_dict_array:
            #     if f['expid'] == 2 and f['cgs_cont'] > 0:
            #         print(f['f448'],-2.5 * np.log10(SU.cgs2ujy(f['cgs_cont'],4500)/(3631*1e6)) )


        except:
            pass

        log.debug(f"CCD adjacent single fiber brightest mag {bright_mag}")
        return fiber_dict_array, bright_mag


    def calc_central_single_fiber_magnitude(self,ffsky=False):
        """

        :param fiber_dict_array:
        :param ffsky:
        :return:
        """

        central_fiber = None
        gmag = 99.9
        try:

            #find the central (highest weighted fiber)
            #fibers should already be sorted
            if self.fibers_sorted:
                central_fiber = self.fibers[0]
            else:
                try:
                    central_fiber = self.fibers[np.argmax([x.raw_weight for x in self.fibers])]
                except:
                    log.debug("Could not locate central fiber")
                    return gmag

            try:
                if ffsky:
                    flux  = central_fiber.fits.ffsky_calfib[central_fiber.panacea_idx]
                else:
                    flux  = central_fiber.fits.calfib[central_fiber.panacea_idx]

                calfibe = central_fiber.fits.calfibe[central_fiber.panacea_idx]

                gmag, cgs_cont, gmag_unc, cgs_cont_unc = \
                                elixer_spectrum.get_hetdex_gmag(flux / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                                G.CALFIB_WAVEGRID,
                                                                calfibe/ 2.0 * G.HETDEX_FLUX_BASE_CGS)

                log.debug(f"Central single fiber brightest mag {gmag}")
            except:
                log.warning("Failed to get single central fiber magnitude",exc_info=True)
        except:
            log.warning("Failed to get single central fiber magnitude", exc_info=True)

        return gmag

    def neighbor_forced_extraction(self,sep_obj,filter='x',catalog_name=None,allow_flat_spectrum=True):
        """
        Forced extraction, (like DetObj forced extraction) but just update the neighbor dictionary
        :param sep_obj:
        :return:
        """
        apt = []

        ra_fix = 0
        dec_fix = 0

        #assume a correction for the ra and dec based on the HETDEX object and its position relative to the imaging
        #This assumes the best_counterpart IS the correct counterpart and the offset between it and the HETDEX coord
        #is due to the astrometric and/or WCS error and differences between the two catalogs and not an emission line offset
        try:
            if catalog_name is not None and filter is not None and filter != 'x':
                if (self.best_counterpart is not None) and (self.best_counterpart.catalog_name == catalog_name) \
                        and (self.best_counterpart.bid_filter.lower() == filter.lower()):
                    if self.best_counterpart.distance < 1.5:
                        if self.wra is not None:
                            ra = self.wra
                            dec = self.wdec
                        else:
                            ra = self.ra
                            dec = self.dec
                        ra_fix =  ra - self.best_counterpart.bid_ra  #this order s|t HETDEX_ra = bid_ra + ra_fix
                        dec_fix = dec - self.best_counterpart.bid_dec
                    else:
                        log.info("Best counterpart distance exceeds limit and will not apply RA, Dec adjustment in neighbor_forced_extraction.")

        except:
            log.warning("Unable to make catalog coordinate correction in neighbor_forced_extraction.",exc_info=True)

        try:
            if ra_fix or dec_fix:
                log.info(f"Applying RA, Dec correction to move from {catalog_name} to HETDEX coord. ({ra_fix},{dec_fix})")

            log.info(f"Fetching spectrum for neighbor: RA,Dec ({sep_obj['ra']+ra_fix:0.5f},{sep_obj['dec']+dec_fix:0.5f})")
            aper = self.extraction_aperture if self.extraction_aperture is not None else 3.0

            coord = SkyCoord(ra=(sep_obj['ra'] + ra_fix) * U.deg, dec=(sep_obj['dec'] + dec_fix) * U.deg)
            apt = hda_get_spectra(coord, survey=f"hdr{G.HDR_Version}", shotid=self.survey_shotid,
                                  ffsky=self.extraction_ffsky, multiprocess=G.GET_SPECTRA_MULTIPROCESS, rad=aper,
                                  tpmin=0.0,fiberweights=False) #don't need the fiber weights
        except:
            log.info("hetdex.py forced_extraction(). Exception calling HETDEX_API get_spectra",exc_info=True)

        try:
            #always make the flat flux
            #Don't bother ... have moved to just computing as needed
            # sep_obj['flat_flux']= 1/G.HETDEX_FLUX_BASE_CGS * G.FLUX_WAVEBIN_WIDTH * SU.make_fnu_flat_spectrum(sep_obj['mag'],filter.lower(),G.CALFIB_WAVEGRID)
            #
            # sep_obj['flat_flux_err'] = 1/G.HETDEX_FLUX_BASE_CGS *  G.FLUX_WAVEBIN_WIDTH * \
            #                            (SU.make_fnu_flat_spectrum(sep_obj['mag_bright'],filter.lower(),G.CALFIB_WAVEGRID) - \
            #                             SU.make_fnu_flat_spectrum(sep_obj['mag_faint'],filter.lower(),G.CALFIB_WAVEGRID))
            #
            # sep_obj['wave'] = G.CALFIB_WAVEGRID

            if len(apt) == 0:

                sep_obj['flux'] = np.zeros(len(G.CALFIB_WAVEGRID))  #in 1e-17 units (like HDF5 read)
                sep_obj['flux_err'] = np.zeros(len(G.CALFIB_WAVEGRID))

                sep_obj['dex_g_mag'] = 99.9
                sep_obj['dex_g_mag_err'] = 99.9


            else:
                # returned from get_spectra as flux density (per AA), so multiply by wavebin width to match the HDF5 reads
                sel_nan = np.isnan(apt['spec'][0]) #where the flux is NaN
                #have to put in zeros, SDSS gmag does not handle NaNs
                #and the MC calls generate bad data if error is set to a huge value, so leave NaNs to 0 flux, 0 error
                #and just understand that if there are many of them, the magnitude can be off
                sep_obj['flux'] = np.nan_to_num(apt['spec'][0]) * G.FLUX_WAVEBIN_WIDTH   #in 1e-17 units (like HDF5 read)
                sep_obj['flux_err'] = np.nan_to_num(apt['spec_err'][0]) * G.FLUX_WAVEBIN_WIDTH
                sep_obj['flux_err'][sel_nan] = 0 #flux error gets a zero where it was NaN or where flux was NaN

                sep_obj['dex_g_mag'], _, sep_obj['dex_g_mag_err'], _ = \
                    elixer_spectrum.get_sdss_gmag(sep_obj['flux'] / G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS,
                                                  G.CALFIB_WAVEGRID,
                                                  sep_obj['flux_err'] / G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS)

                if sep_obj['dex_g_mag'] is None  or np.isnan(sep_obj['dex_g_mag']):
                        sep_obj['dex_g_mag'], _, sep_obj['dex_g_mag_err'], _ = \
                        elixer_spectrum.get_hetdex_gmag(sep_obj['flux'] / G.FLUX_WAVEBIN_WIDTH  * G.HETDEX_FLUX_BASE_CGS,
                                                        G.CALFIB_WAVEGRID,
                                                        sep_obj['flux_err'] / G.FLUX_WAVEBIN_WIDTH  * G.HETDEX_FLUX_BASE_CGS)

                if sep_obj['dex_g_mag'] is None or np.isnan(sep_obj['dex_g_mag']):
                    sep_obj['dex_g_mag'] = 99.9
                    sep_obj['dex_g_mag_err'] = 99.9

        except:
            log.error("Exception! Exception fetching neighbor spectra.",exc_info=True)


    def forced_extraction(self):
        """

        :param basic_only:
        :return:
        """
        apt = []

        try:
            coord = SkyCoord(ra=self.ra * U.deg, dec=self.dec * U.deg)
            apt = hda_get_spectra(coord, survey=f"hdr{G.HDR_Version}", shotid=self.survey_shotid,
                                  ffsky=self.extraction_ffsky, multiprocess=G.GET_SPECTRA_MULTIPROCESS, rad=self.extraction_aperture,
                                  tpmin=0.0,fiberweights=True)
        except:
            log.info("hetdex.py forced_extraction(). Exception calling HETDEX_API get_spectra",exc_info=True)

        try:
            if len(apt) == 0:
                #print(f"No spectra for ra ({self.ra}) dec ({self.dec})")
                log.info(f"No spectra for ra ({self.ra}) dec ({self.dec})")
                self.status = -1
                return

            # returned from get_spectra as flux density (per AA), so multiply by wavebin width to match the HDF5 reads
            self.sumspec_flux = np.nan_to_num(apt['spec'][0]) * G.FLUX_WAVEBIN_WIDTH   #in 1e-17 units (like HDF5 read)
            self.sumspec_fluxerr = np.nan_to_num(apt['spec_err'][0]) * G.FLUX_WAVEBIN_WIDTH
            self.sumspec_wavelength = np.array(apt['wavelength'][0])
            try: #name change in HDR3
                self.sumspec_apcor =  np.array(apt['apcor'][0]) #this is the apcor ... the fiber_weights are the PSF weights
            except:
                self.sumspec_apcor =  np.array(apt['weights'][0]) #this is the apcor ... the fiber_weights are the PSF weights


            #get fiber weights if available
            fiber_weights = None
            try:
                fiber_weights = apt['fiber_weights'][0] #as array of ra,dec,weight
            except:
                pass


            #get the per shot sky residual (if configured to do so)
            if G.SUBTRACT_HETDEX_SKY_RESIDUAL:
                residual_spec, residual_spec_err = shot_sky.get_shot_sky_residual(self.survey_shotid) #this is as a single averaged fiber

            if not self.w:

                # find the "best" wavelength to use as the central peak
                #spectrum = elixer_spectrum.Spectrum()
                if self.spec_obj is None:
                    self.spec_obj = elixer_spectrum.Spectrum()

                log.info("Scanning for anchor line ...")
                #self.all_found_lines = elixer_spectrum.peakdet(self.sumspec_wavelength,self.sumspec_flux,self.sumspec_fluxerr,values_units=-17)
                w = self.spec_obj.find_central_wavelength(self.sumspec_wavelength,self.sumspec_flux,
                                                                          self.sumspec_fluxerr,-17,return_list=False)
                if w is not None and (3400.0 < w < 5600.0):
                    self.w = w
                    self.target_wavelength = w
                    log.info(f"Anchor line set: {self.w:0.2f}")
                else:

                    try:
                        self.w = self.sumspec_wavelength[np.nanargmax(self.sumspec_flux)]
                        self.target_wavelength = self.w
                        log.info(f"Cannot identify a suitable target wavelength. Setting to maximum flux value ({self.w}).")
                    except:
                        #print("Cannot identify a suitable target wavelength. Arbitrarly setting to 4500.0 for report.")
                        log.info("Cannot identify a suitable target wavelength. Arbitrarly setting to 4500.0 for report.")
                        self.w = 4500.0
                        self.target_wavelength = 4500.0
            else:
                log.info(f"Using predefined anchor line {self.w:0.2f}")

            if self.w is not None and self.w != 0:
                idx = elixer_spectrum.getnearpos(self.sumspec_wavelength, self.w)
            else:
                try:
                    idx = np.argmax(self.sumspec_flux)
                except:
                    idx = 0
            left = idx - 25  # 2AA steps so +/- 50AA
            right = idx + 25

            if left < 0:
                left = 0
            if right > len(self.sumspec_flux):
                right = len(self.sumspec_flux)

            # these are on the 2AA grid (old spece had 2AA steps but, the grid was centered on the main wavelength)
            # this grid is not centered but is on whole 2AA (i.e. 3500.00, 3502.00, ... not 3500.4192, 3502.3192, ...)
            self.sumspec_wavelength_zoom = self.sumspec_wavelength[left:right]
            self.sumspec_flux_zoom = self.sumspec_flux[left:right]
            self.sumspec_fluxerr_zoom = self.sumspec_fluxerr[left:right]
            self.sumspec_counts_zoom = self.sumspec_counts[left:right]


            #set basic spec_obj info
            self.spec_obj.identifier = "eid(%s)" % str(self.entry_id)
            self.spec_obj.plot_dir = self.outdir

            #need individual fibers so can set noise
            #at some point this may come from get_spectra
            #for now, lets just get the closest fibers to RA, Dec and sort by distance

            if self.survey_shotid is None:
                #get shot for RA, Dec
                log.error("Required survey_shotid is None in DetObj::forced_extraction")
                self.status = -1
                return

            ftb = hda_get_fibers_table(self.survey_shotid,coord,
                                       radius=self.extraction_aperture * U.arcsec,
                                       #radius=60.0 * U.arcsec,
                                       survey=f"hdr{G.HDR_Version}")

            #build list of fibers and sort by distance (as proxy for weight)
            count = 0
            subset_norm_weight = 0
            num_fibers = len(ftb)
            for row in ftb:
                count += 1
                specid = row['specid']
                ifuslot = row['ifuslot']
                ifuid = row['ifuid']
                amp = row['amp']
                #date = str(row['date'])

                # expected to be "20180320T052104.2"
                #time_ex = row['timestamp'][9:]
                #time = time_ex[0:6]  # hhmmss

                #yyyymmddxxx
                date = str(self.survey_shotid//1000) #chop off the shotid part and keep the date part
                time = "000000"
                time_ex = "000000.0"


                mfits_name = row['multiframe']  # similar to multi*fits style name
                #fiber_index is a string here: '20190104025_3_multi_315_021_073_RL_030'
                fiber_index = row['fibidx'] #int(row['fiber_id'][-3:])  (fiber_id is a number, fibidx is index)
                try:
                    obsid = row['fiber_id'][8:11]
                except:
                    try:
                        obsid = str(self.survey_shotid)[8:11]
                    except:
                        log.error("Unable to determin observation ID.")

                idstring = date + "v" + time_ex + "_" + specid + "_" + ifuslot + "_" + ifuid + "_" + amp + "_" #leave off the fiber for the moment

                log.debug("Building fiber %d of %d (%s e%d) ..." % (count, num_fibers,idstring + str(fiber_index+1),int(row['expnum'])))
                idstring += str(fiber_index) #add the fiber index (zero based)

                fiber = elixer_fiber.Fiber(idstring=idstring,specid=specid,ifuslot=ifuslot,ifuid=ifuid,amp=amp,
                                           date=date,time=time,time_ex=time_ex, panacea_fiber_index=fiber_index,
                                           detect_id=self.id)

                if fiber is not None:
                    duplicate = False
                    fiber.ra = row['ra']
                    fiber.dec = row['dec']
                    fiber.obsid = obsid #int(row['obsind']) #notice: obsind vs obsid
                    fiber.shotid = date + obsid #both strings
                    fiber.expid = int(row['expnum'])  # integer now
                    fiber.detect_id = self.id
                    fiber.center_x = row['ifux']
                    fiber.center_y = row['ifuy']

                    #don't have weights used, so use distance to the provided RA, Dec as a sorting substitute
                    #fiber.raw_weight = row['weight']
                    fiber.distance = utils.angular_distance(fiber.ra,fiber.dec,self.ra,self.dec)

                    try:
                        #in degrees, so this is less than 0.1"
                        fw_idx = np.where( (abs(fiber_weights[:,0] - fiber.ra) < 0.00003) &
                                       (abs(fiber_weights[:,1] - fiber.dec) < 0.00003 ))[0]

                        if len(fw_idx) == 1:
                            fiber.raw_weight = fiber_weights[fw_idx,2]
                            subset_norm_weight += fiber.raw_weight
                    except:
                        pass

                    # check that this is NOT a duplicate
                    for i in range(len(self.fibers)):
                        if fiber == self.fibers[i]:
                            log.warning(
                                "Warning! Duplicate Fiber in detectID %s: %s . idx %d == %d. Duplicate will not be processed." %
                                (str(self.hdf5_detectid), fiber.idstring, i, count - 1))
                            duplicate = True
                            duplicate_count += 1
                            break

                    if duplicate:
                        continue  # continue on to next fiber

                    # fiber.relative_weight = row['weight']
                    # add the fiber (still needs to load its fits file)
                    # we already know the path to it ... so do that here??

                    # full path to the HDF5 fits equivalent (or failing that the panacea fits file?)
                    fiber.fits_fn = fiber.find_hdf5_multifits(loc=self.hdf5_shot_dir)

                    # fiber.fits_fn = get_hetdex_multifits_path(fiber.)

                    # now, get the corresponding FITS or FITS equivalent (HDF5)
                    #if self.annulus is None:
                    if True:
                        fits = hetdex_fits.HetdexFits(empty=True)
                        # populate the data we need to read the HDF5 file
                        fits.filename = fiber.fits_fn  # mfits_name #todo: fix to the corect path
                        fits.multiframe = mfits_name
                        fits.panacea = True
                        fits.hdf5 = True

                        fits.obsid = str(fiber.obsid).zfill(3)
                        fits.expid = int(fiber.expid)
                        fits.specid = str(fiber.specid).zfill(3)
                        fits.ifuslot = str(fiber.ifuslot).zfill(3)
                        fits.ifuid = str(fiber.ifuid).zfill(3)
                        fits.amp = fiber.amp
                        fits.side = fiber.amp[0]

                        fits.obs_date = fiber.dither_date
                        fits.obs_ymd = fits.obs_date

                        # now read the HDF5 equivalent
                        #if we don't already have it
                        already_read = False
                        for fi in self.fibers:
                            #same filename (same DateVshot + observation) + same (IFU address) + same exposure
                            if (fi.fits.filename == fits.filename) and (fi.fits.multiframe == fits.multiframe) \
                                and (fi.fits.expid == fits.expid):
                                fiber.fits = fi.fits
                                already_read = True
                                break

                        if not already_read:
                            fits.read_hdf5()
                             # check if it is okay
                            if fits.okay:
                                fiber.fits = fits
                            else:
                                log.error("HDF5 multi-fits equivalent is not okay ...")

                    self.fibers.append(fiber)


            if subset_norm_weight > 0:
                for f in self.fibers:
                    f.relative_weight = f.raw_weight / subset_norm_weight

                self.fibers.sort(key=lambda x: x.relative_weight, reverse=True)  # highest weight is index =
            else:
                self.fibers.sort(key=lambda x: x.distance, reverse=False)  # highest weight is index = 0

            self.fibers_sorted = True

            #build a noise estimate over the top 4 fibers (amps)?
            try:
                good_idx = np.where([x.fits for x in self.fibers])[0]  # some might be None, so get those that are not
                good_idx = good_idx[0:min(len(good_idx), 4)]

                all_calfib = np.concatenate([self.fibers[i].fits.calfib for i in good_idx], axis=0)

                # use the std dev of all "mostly empty" (hence sigma=3.0) or "sky" fibers as the error
                mean, median, std = sigma_clipped_stats(all_calfib, axis=0, sigma=3.0)
                self.calfib_noise_estimate = std
                if not G.MULTILINE_USE_ERROR_SPECTRUM_AS_NOISE:
                    self.spec_obj.noise_estimate = self.calfib_noise_estimate
                    self.spec_obj.noise_estimate_wave = G.CALFIB_WAVEGRID

            except:
                log.info("Could not build DetObj calfib_noise_estimate", exc_info=True)
                self.calfib_noise_estimate = np.zeros(len(G.CALFIB_WAVEGRID))

                try:
                    log.info("Setting spectrum noise estimate to error spectrum")
                    self.spec_obj.noise_estimate = self.sumspec_fluxerr
                    self.spec_obj.noise_estimate_wave = G.CALFIB_WAVEGRID
                except:
                    log.info("Could not set spectrum noise_estimate to sumpsec_fluxerr", exc_info=True)

            #my own fitting
            try:

                self.spec_obj.set_spectra(self.sumspec_wavelength, self.sumspec_flux, self.sumspec_fluxerr, self.w,
                                          values_units=-17, estflux=self.estflux, estflux_unc=self.estflux_unc,
                                          eqw_obs=self.eqw_obs, eqw_obs_unc=self.eqw_obs_unc,
                                          estcont=self.cont_cgs, estcont_unc=self.cont_cgs_unc,
                                          continuum_g=self.best_gmag_cgs_cont,continuum_g_unc=self.best_gmag_cgs_cont_unc)

                #update DEX-g based continuum and EW
                try:
                    self.best_gmag_cgs_cont *= self.spec_obj.gband_continuum_correction()
                    self.best_gmag_cgs_cont_unc *= self.spec_obj.gband_continuum_correction()

                    self.best_eqw_gmag_obs = self.estflux / self.best_gmag_cgs_cont
                    self.best_eqw_gmag_obs_unc = abs(self.best_eqw_gmag_obs * np.sqrt(
                        (self.estflux_unc / self.estflux) ** 2 +
                        (self.best_gmag_cgs_cont_unc / self.best_gmag_cgs_cont) ** 2))

                    log.info(f"Update best DEX-g continuum x{self.spec_obj.gband_continuum_correction():0.2f} and EW; cont {self.best_gmag_cgs_cont} +/- {self.best_gmag_cgs_cont_unc}" )
                except:
                    log.error("Exception! Excpetion updating DEX-g continuum.",exc_info=True)

                if self.spec_obj.central_eli is not None:

                    #update the central wavelength
                    log.info(f"Central Wavelength updated from {self.w} to {self.spec_obj.central_eli.fit_x0}")
                    self.w = self.spec_obj.central_eli.fit_x0

                    if self.spec_obj.central_eli.mcmc_line_flux is None:

                        self.estflux = self.spec_obj.central_eli.fit_line_flux
                        self.estflux_unc = self.spec_obj.central_eli.fit_line_flux_err

                        self.cont_cgs = self.spec_obj.central_eli.fit_continuum
                        self.cont_cgs_unc = 0 # not computed correctly right now self.spec_obj.central_eli.fit_continuum_err

                        self.eqw_obs = self.spec_obj.central_eli.eqw_obs
                        self.eqw_obs_unc = 0 #self.eqw_obs * np.sqrt( (self.estflux_unc/self.estflux)**2 + (self.cont_cgs_unc/self.cont_cgs)**2)

                        # self.snr = self.spec_obj.central_eli.mcmc_snr
                        self.snr = self.spec_obj.central_eli.snr
                        self.snr_unc = 0

                        self.chi2 = self.spec_obj.central_eli.fit_chi2
                        self.chi2_unc = 0.0

                    else: #we have mcmc data

                        self.estflux = self.spec_obj.central_eli.mcmc_line_flux
                        self.estflux_unc = 0.5 * (self.spec_obj.central_eli.mcmc_line_flux_tuple[1] +
                                                  self.spec_obj.central_eli.mcmc_line_flux_tuple[2])

                        self.eqw_obs = self.spec_obj.central_eli.mcmc_ew_obs[0]
                        self.eqw_obs_unc = 0.5 * (self.spec_obj.central_eli.mcmc_ew_obs[1] +
                                                  self.spec_obj.central_eli.mcmc_ew_obs[2])

                        self.cont_cgs = self.spec_obj.central_eli.mcmc_continuum
                        self.cont_cgs_unc = 0.5*(self.spec_obj.central_eli.mcmc_continuum_tuple[1] +
                                                 self.spec_obj.central_eli.mcmc_continuum_tuple[2])
                        # self.snr = self.spec_obj.central_eli.mcmc_snr
                        self.snr = self.spec_obj.central_eli.mcmc_snr
                        self.snr_unc = self.spec_obj.central_eli.mcmc_snr_err

                        try:
                            if self.spec_obj.central_eli.mcmc_chi2 is not None and \
                               not np.isnan(self.spec_obj.central_eli.mcmc_chi2) and \
                               self.spec_obj.central_eli.mcmc_chi2 > 0:
                                self.chi2 = self.spec_obj.central_eli.mcmc_chi2
                            else:
                                self.chi2 = self.spec_obj.central_eli.fit_chi2
                            self.chi2_unc = 0.0
                        except:
                            pass


                    self.spec_obj.estflux = self.estflux
                    self.spec_obj.eqw_obs = self.eqw_obs

                    self.snr = self.spec_obj.central_eli.snr #row['sn']
                    self.snr_unc = self.spec_obj.central_eli.snr_err #row['sn']
                    #self.snr_unc = 0.0 #row['sn_err']

                    self.sigma = self.spec_obj.central_eli.fit_sigma #row['linewidth']  # AA
                    self.sigma_unc = self.spec_obj.central_eli.fit_sigma_err #0.0 #row['linewidth_err']
                    if (self.sigma_unc is None) or (self.sigma_unc < 0.0):
                        self.sigma_unc = 0.0
                    self.fwhm = 2.35 * self.sigma
                    self.fwhm_unc = 2.35 * self.sigma_unc

                    self.estflux_h5 = self.estflux
                    self.estflux_h5_unc = self.estflux_unc

                else:
                    log.warning("No MCMC data to update core stats in hetdex::load_flux_calibrated_spectra(). spec_obj.central_eli is None.")

            except:
                log.warning("No MCMC data to update core stats in hetdex::load_flux_calibrated_spectra")




            #todo: then update the values on record
            # mu, sigma, Amplitude, y, dx   (dx is the bin width if flux instead of flux/dx)
            #continuum does NOT get the bin scaling
            self.line_gaussfit_parms = (self.w,self.sigma,self.estflux*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
                                        self.cont_cgs/G.HETDEX_FLUX_BASE_CGS,
                                        G.FLUX_WAVEBIN_WIDTH) #*2.0 for Karl's bin width
            self.line_gaussfit_unc = (self.w_unc,self.sigma_unc,self.estflux_unc*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
                                      self.cont_cgs_unc/G.HETDEX_FLUX_BASE_CGS, 0.0)

            # used just below to choose between the two
            sdss_okay = 0
            hetdex_okay = 0

            # sum over entire HETDEX spectrum to estimate g-band magnitude
            try:
                self.hetdex_gmag, self.hetdex_gmag_cgs_cont, self.hetdex_gmag_unc, self.hetdex_gmag_cgs_cont_unc = \
                    elixer_spectrum.get_hetdex_gmag(self.sumspec_flux / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                    self.sumspec_wavelength,
                                                    self.sumspec_fluxerr / 2.0 * G.HETDEX_FLUX_BASE_CGS)

                log.debug(f"HETDEX spectrum gmag {self.hetdex_gmag} +/- {self.hetdex_gmag_unc}")
                log.debug(f"HETDEX spectrum cont {self.hetdex_gmag_cgs_cont} +/- {self.hetdex_gmag_cgs_cont_unc}")

                if (self.hetdex_gmag_cgs_cont is not None) and (self.hetdex_gmag_cgs_cont != 0) and not np.isnan(
                        self.hetdex_gmag_cgs_cont):
                    if (self.hetdex_gmag_cgs_cont_unc is None) or np.isnan(self.hetdex_gmag_cgs_cont_unc):
                        self.hetdex_gmag_cgs_cont_unc = 0.0
                        hetdex_okay = 1
                    else:
                        hetdex_okay = 2

                    self.eqw_hetdex_gmag_obs = self.estflux / self.hetdex_gmag_cgs_cont
                    self.eqw_hetdex_gmag_obs_unc = abs(self.eqw_hetdex_gmag_obs * np.sqrt(
                        (self.estflux_unc / self.estflux) ** 2 +
                        (self.hetdex_gmag_cgs_cont_unc / self.hetdex_gmag_cgs_cont_unc) ** 2))

                if (self.hetdex_gmag is None) or np.isnan(self.hetdex_gmag):
                    hetdex_okay = 0
            except:
                hetdex_okay = 0
                log.error("Exception computing HETDEX spectrum gmag", exc_info=True)

            # feed HETDEX spectrum through SDSS gband filter
            try:
                # reminder needs erg/s/cm2/AA and sumspec_flux in ergs/s/cm2 so divied by 2AA bin width
                #                self.sdss_gmag, self.cont_cgs = elixer_spectrum.get_sdss_gmag(self.sumspec_flux/2.0*1e-17,self.sumspec_wavelength)
                if False:
                    self.sdss_gmag, self.sdss_cgs_cont = elixer_spectrum.get_sdss_gmag(
                        self.sumspec_flux / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                        self.sumspec_wavelength)
                    self.sdss_cgs_cont_unc = np.sqrt(np.sum(self.sumspec_fluxerr ** 2)) / len(
                        self.sumspec_fluxerr) * G.HETDEX_FLUX_BASE_CGS

                    log.debug(f"SDSS spectrum gmag {self.sdss_gmag} +/- {self.sdss_gmag_unc}")
                    log.debug(f"SDSS spectrum cont {self.sdss_cgs_cont} +/- {self.sdss_cgs_cont_unc}")

                else:
                    self.sdss_gmag, self.sdss_cgs_cont, self.sdss_gmag_unc, self.sdss_cgs_cont_unc = \
                        elixer_spectrum.get_sdss_gmag(self.sumspec_flux / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                      self.sumspec_wavelength,
                                                      self.sumspec_fluxerr / 2.0 * G.HETDEX_FLUX_BASE_CGS)

                    log.debug(f"SDSS spectrum gmag {self.sdss_gmag} +/- {self.sdss_gmag_unc}")
                    log.debug(f"SDSS spectrum cont {self.sdss_cgs_cont} +/- {self.sdss_cgs_cont_unc}")

                if (self.sdss_cgs_cont is not None) and (self.sdss_cgs_cont != 0) and not np.isnan(self.sdss_cgs_cont):
                    if (self.sdss_cgs_cont_unc is None) or np.isnan(self.sdss_cgs_cont_unc):
                        self.sdss_cgs_cont_unc = 0.0
                        sdss_okay = 1
                    else:
                        sdss_okay = 2

                    self.eqw_sdss_obs = self.estflux / self.sdss_cgs_cont
                    self.eqw_sdss_obs_unc = abs(self.eqw_sdss_obs * np.sqrt(
                        (self.estflux_unc / self.estflux) ** 2 +
                        (self.sdss_cgs_cont_unc / self.sdss_cgs_cont) ** 2))

                if (self.sdss_gmag is None) or np.isnan(self.sdss_gmag):
                    sdss_okay = 0

            except:
                sdss_okay = 0
                log.error("Exception computing SDSS g-mag", exc_info=True)

            # choose the best
            #even IF okay == 0, still record the probably bogus value (when
            #actually using the values elsewhere they are compared to a limit and the limit is used if needed

            if (hetdex_okay == sdss_okay) and (self.hetdex_gmag is not None) and (self.sdss_gmag is not None) and \
                        abs(self.hetdex_gmag - self.sdss_gmag) < 1.0: #use both as an average? what if they are very different?
                #make the average
                avg_cont = 0.5 * (self.hetdex_gmag_cgs_cont + self.sdss_cgs_cont)
                avg_cont_unc =  np.sqrt(self.hetdex_gmag_cgs_cont_unc**2 + self.sdss_cgs_cont_unc**2) #error on the mean


                self.best_gmag_selected = 'mean'
                self.best_gmag = -2.5*np.log10(SU.cgs2ujy(avg_cont,4500.00) / 1e6 / 3631.)
                mag_faint = -2.5*np.log10(SU.cgs2ujy(avg_cont-avg_cont_unc,4500.00) / 1e6 / 3631.)
                mag_bright = -2.5*np.log10(SU.cgs2ujy(avg_cont+avg_cont_unc,4500.00) / 1e6 / 3631.)
                self.best_gmag_unc = 0.5 * (mag_faint-mag_bright)

                self.best_gmag_cgs_cont = avg_cont
                self.best_gmag_cgs_cont_unc = avg_cont_unc

                self.best_eqw_gmag_obs = self.estflux / self.best_gmag_cgs_cont
                self.best_eqw_gmag_obs_unc = abs(self.best_eqw_gmag_obs * np.sqrt(
                    (self.estflux_unc / self.estflux) ** 2 +
                    (self.best_gmag_cgs_cont_unc / self.best_gmag_cgs_cont) ** 2))

                log.debug("Using mean of HETDEX full width gmag and SDSS gmag.")
                log.info(f"Mean spectrum gmag {self.best_gmag:0.2f} +/- {self.best_gmag_unc:0.3f}; cont {self.best_gmag_cgs_cont} +/- {self.best_gmag_cgs_cont_unc}" )


            elif hetdex_okay >= sdss_okay > 0 and not np.isnan(self.hetdex_gmag_cgs_cont) and (self.hetdex_gmag_cgs_cont is not None):
                self.best_gmag_selected = 'hetdex'
                self.best_gmag = self.hetdex_gmag
                self.best_gmag_unc = self.hetdex_gmag_unc
                self.best_gmag_cgs_cont = self.hetdex_gmag_cgs_cont
                self.best_gmag_cgs_cont_unc = self.hetdex_gmag_cgs_cont_unc
                self.best_eqw_gmag_obs = self.eqw_hetdex_gmag_obs
                self.best_eqw_gmag_obs_unc = self.eqw_hetdex_gmag_obs_unc
                log.debug("Using HETDEX full width gmag over SDSS gmag.")
            elif sdss_okay > 0 and not np.isnan(self.sdss_cgs_cont) and (self.sdss_cgs_cont is not None):
                self.best_gmag_selected = 'sdss'
                self.best_gmag = self.sdss_gmag
                self.best_gmag_unc = self.sdss_gmag_unc
                self.best_gmag_cgs_cont = self.sdss_cgs_cont
                self.best_gmag_cgs_cont_unc = self.sdss_cgs_cont_unc
                self.best_eqw_gmag_obs = self.eqw_sdss_obs
                self.best_eqw_gmag_obs_unc = self.eqw_sdss_obs_unc
                log.debug("Using SDSS gmag over HETDEX full width gmag")
            else: #something catastrophically bad
                log.debug("No full width spectrum g-mag estimate is valid.")
                self.best_gmag_selected = 'limit'
                self.best_gmag = G.HETDEX_CONTINUUM_MAG_LIMIT
                self.best_gmag_unc = 0
                self.best_gmag_cgs_cont = G.HETDEX_CONTINUUM_FLUX_LIMIT
                self.best_gmag_cgs_cont_unc = 0
                self.best_eqw_gmag_obs = self.estflux / self.best_gmag_cgs_cont
                self.best_eqw_gmag_obs_unc = 0

            try:
                diff = abs(self.hetdex_gmag - self.sdss_gmag)
                unc = self.hetdex_gmag_unc + self.sdss_gmag_unc
                if (hetdex_okay == sdss_okay) and \
                        ((self.hetdex_gmag < G.HETDEX_CONTINUUM_MAG_LIMIT) or (self.sdss_gmag <  G.HETDEX_CONTINUUM_MAG_LIMIT)) and \
                        ((diff > unc) and (diff > 0.5)):
                    self.flags |= G.DETFLAG_DEXSPEC_GMAG_INCONSISTENT
                    log.info(f"DEX spectrum gmag disagree by {diff/unc:0.1f}x uncertainty. "
                             f"Dex g {self.hetdex_gmag:0.2f} +/- {self.hetdex_gmag_unc:0.3f} "
                             f"vs SDSS g {self.sdss_gmag:0.2f} +/- {self.sdss_gmag_unc:0.3f}")
            except:
                pass

            try:
                self.hetdex_cont_cgs = self.cont_cgs
                self.hetdex_cont_cgs_unc = self.cont_cgs_unc

                if self.cont_cgs == -9999:  # still unset ... weird?
                    log.warning("Warning! HETDEX continuum estimate not set. Using best gmag for estimate(%g +/- %g)."
                                % (self.best_gmag_cgs_cont, self.best_gmag_cgs_cont_unc))

                    self.cont_cgs_narrow = self.cont_cgs
                    self.cont_cgs_narrow_unc = self.cont_cgs_unc
                    self.cont_cgs = self.best_gmag_cgs_cont
                    self.cont_cgs_unc = self.best_gmag_cgs_cont_unc
                    self.using_best_gmag_ew = True
                elif self.cont_cgs <= 0.0:
                    log.warning("Warning! (narrow) continuum <= 0.0. Using best gmag for estimate (%g +/- %g)."
                                % (self.best_gmag_cgs_cont, self.best_gmag_cgs_cont_unc))
                    self.cont_cgs_narrow = self.cont_cgs
                    self.cont_cgs_narrow_unc = self.cont_cgs_unc
                    self.cont_cgs = self.best_gmag_cgs_cont
                    self.cont_cgs_unc = self.best_gmag_cgs_cont_unc
                    self.using_best_gmag_ew = True
            except:
                pass

            self.spec_obj.classify(known_z=self.known_z)  # solutions can be returned, also stored in spec_obj.solutions

            self.rvb = SU.red_vs_blue(self.w, self.sumspec_wavelength,
                                      self.sumspec_flux / G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS,
                                      self.sumspec_fluxerr / G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS, self.fwhm)

            if self.annulus:
                self.syn_obs = elixer_observation.SyntheticObservation()
                if self.wra:
                    self.syn_obs.ra = self.wra
                    self.syn_obs.dec = self.wdec
                else:
                    self.syn_obs.ra = self.ra
                    self.syn_obs.dec = self.dec

                self.syn_obs.target_wavelength = self.target_wavelength
                self.syn_obs.annulus = self.annulus
                self.syn_obs.fibers_all = self.fibers
                self.syn_obs.w = self.target_wavelength
                self.syn_obs.survey_shotid = self.survey_shotid
                # get all the fibers inside the outer annulus radius
                self.syn_obs.get_aperture_fibers()

        except:
            log.error("Exception in hetdex.py forced_extraction.",exc_info=True)
            self.status = -1

        try:
            try:
                self.ccd_adjacent_fibers, self.ccd_adjacent_single_fiber_brightest_mag = \
                        self.calc_ccd_adjacent_fiber_magnitudes(self.find_ccd_adjacent_fibers())

                self.central_single_fiber_mag = self.calc_central_single_fiber_magnitude()
            except:
                pass
        except:
            pass


    def load_hdf5_fluxcalibrated_spectra(self,hdf5_fn,id,basic_only=False):
        """

        :return:
        """

        try:
            id = int(id)
        except:
            log.error("Exception converting id to int type",exc_info=True)
            msg = "+++++ %s" %str(id)
            log.error(msg)
            self.status = -1
            return


        log.debug("Loading flux calibrated data from HDF5 ...")
        self.panacea = True #if we are here, it can only be panacea

        del self.sumspec_wavelength[:]
        del self.sumspec_counts[:]
        del self.sumspec_flux[:]
        del self.sumspec_fluxerr[:]
        del self.sumspec_wavelength_zoom[:]
        del self.sumspec_counts_zoom[:]
        del self.sumspec_flux_zoom[:]
        del self.sumspec_fluxerr_zoom[:]

        log.debug("Loading base detection data from HDF5 ...")
        with tables.open_file(hdf5_fn,mode="r") as h5_detect:

            detection_table = h5_detect.root.Detections
            fiber_table = h5_detect.root.Fibers
            spectra_table = h5_detect.root.Spectra
            #spectra_table.cols.detectid.create_csindex()

            #get the multi-fits equivalent info
            #can't use "detectid==detectid" ... context is confused
            try:
                rows = detection_table.read_where("detectid==id")
            except:
                log.error("Exception in hetdex::DetObj::load_hdf5_fluxcalibrated_spectra reading rows from detection_table",
                          exc_info=True)
                rows = None

            if rows is None:
                self.status = -1
                log.error(f"Problem loading detectid {id}. None returned.")
                return
            elif rows.size != 1:
                self.status = -1
                log.error(f"Problem loading detectid {id}. {rows.size} rows returned.")
                return

            row = rows[0] #should only be the one row

            #could be more than one? ... fibers from different dates, or across amps at least
            #or just the highest weight fiber?
            #mfits_name = row['multiframe']

            #set the pdf name (w/o the .pdf extension
            if G.python2():
                self.pdf_name = row['inputid']
                try:
                    self.hdf5_detectname = row['detectname']
                except:
                    pass  #unimportant, but some versions don't have this column
            else:
                self.pdf_name = row['inputid'].decode()
                try:
                    self.hdf5_detectname = row['detectname'].decode()
                except:
                    pass #unimportant, but some versions don't have this column


            ############################
            #get basic detection info
            ############################
            self.w = row['wave']
            self.w_unc = row['wave_err']
            self.wra = row['ra']
            self.wdec = row['dec']
            self.survey_shotid = row['shotid']

            self.ifu_x = row['x_ifu']
            self.ifu_y = row['y_ifu']

            if basic_only: #we're done, this is all we need
                return




            #todo: need the Sky X,Y ?
            #self.x = #Sky X (IFU-X)
            #self.y =

            self.chi2 = row['chi2']
            self.chi2_unc = row['chi2_err']

            self.snr = row['sn']
            self.snr_unc = row['sn_err']

            self.sigma = row['linewidth']#AA
            self.sigma_unc = row['linewidth_err']
            if (self.sigma_unc is None) or (self.sigma_unc < 0.0):
                self.sigma_unc = 0.0
            self.fwhm = 2.35 * self.sigma
            self.fwhm_unc = 2.35 * self.sigma_unc

            self.estflux = row['flux']
            self.estflux_unc = row['flux_err']

            self.estflux_h5 = self.estflux
            self.estflux_h5_unc = self.estflux_unc


            self.cont_cgs = row['continuum'] #units of e-17 set below
            self.cont_cgs_unc = row['continuum_err']

            #bad idea (leaving here just as a reminder)... setting as a small value leads to artifically
            #huge EW (even with correspodingly huge error)
            # if (self.cont_cgs == 0.) and (self.cont_cgs_unc != 0.):
            #     log.debug("HETDEX continuum estimate == 0.0; setting to 0.01 e-17")
            #     self.cont_cgs = 0.01





            # mu, sigma, Amplitude, y, dx   (dx is the bin width if flux instead of flux/dx)
            #continuum does NOT get the bin scaling
            self.line_gaussfit_parms = (self.w,self.sigma,self.estflux*G.FLUX_WAVEBIN_WIDTH,self.cont_cgs,
                                        G.FLUX_WAVEBIN_WIDTH) #*2.0 for Karl's bin width
            self.line_gaussfit_unc = (self.w_unc,self.sigma_unc,self.estflux_unc*G.FLUX_WAVEBIN_WIDTH,self.cont_cgs_unc,
                                      0.0)

            log.debug(f"DEX Gaussian parms: x0 = {self.w:0.2f}AA, sigma = {self.sigma:0.3f}, A = {self.estflux*G.FLUX_WAVEBIN_WIDTH:0.2f}e-17, "
                      f"y = {self.cont_cgs:0.2f}e-17")

            #fix bin-width scaling issue for area parameter
            # if False:
            #     #todo: deal with this at a later point ...??
            #     #the data is erg s^-1 cm^-2 (not per AA)
            #     #if integrate under curve, then get flux * Length (AA) not a flux
            #     #so would then need to divide by the dx (or bin width) to get back to a flux
            #     #and that would be the dx originally used to calculate the data points (or 2.0 AA)
            #     if self.line_gaussfit_parms is not None:
            #         try:
            #             # have to make a list and back again since tuples are immutable
            #             parms = list(self.line_gaussfit_parms)
            #             parms[2] *= 2.0
            #             self.line_gaussfit_parms = tuple(parms)
            #
            #         except:
            #             log.warning("hetdex::load_hdf5_fluxcalibrated_spectra() unable to correct pixels to AA", exc_info=True)

            self.estflux *= G.HETDEX_FLUX_BASE_CGS #now as erg s^-1 cm^-2  .... NOT per AA
            #each data point IS the integrated line flux over the width of the bin
            self.estflux_unc *= G.HETDEX_FLUX_BASE_CGS

            self.cont_cgs *= G.HETDEX_FLUX_BASE_CGS
            self.cont_cgs_unc *= G.HETDEX_FLUX_BASE_CGS

            #ignoring date, datevobs, fiber_num, etc that apply to the top weighted fiber or observation, etc
            #since we want ALL the fibers and will load them after the core spectra info

            ############################################
            #get the weighted and summed spectra info
            ############################################
            log.debug("Loading summed spectra data from HDF5 ...")
            rows = spectra_table.read_where("detectid==id")
            if (rows is None) or (rows.size != 1):
                self.status = -1
                log.error("Problem loading detectid. Multiple rows or no rows in Spectra table.")
                return
            row = rows[0]

            self.sumspec_wavelength = row['wave1d']
            self.sumspec_counts = row['counts1d']#not really using this anymore
            #self.sumspec_countserr #not using this

            self.sumspec_flux = row['spec1d'] #DOES NOT have units attached, but is 10^17 (so *1e-17 to get to real units)
            self.sumspec_fluxerr = row['spec1d_err']
            self.sumspec_apcor = row['apcor'] #aperture correction

            # # #test:
            # print("!!!!!!!!!!!!!!!!!!!!!!! TEST: REMOVE ME !!!!!!!!!!!!!!!!!!!!!")
            # self.sumspec_flux = self.sumspec_flux * 2.0  / self.sumspec_apcor /  self.sumspec_apcor
            # # self.sumspec_flux = row['spec1d_nc']


            #sanity check:
            try:
                sel = np.where(np.array(self.sumspec_flux) > 0)[0]
                if len(sel) < (0.1 * len(self.sumspec_flux )): #pretty weak check, but if it fails something is very wrong
                    self.grossly_negative_spec = True
            except:
                pass

            #this is HETDEX data only (using line continuum estimate)(a bit below will optionally use sdss)
            if self.cont_cgs != 0:
                self.eqw_obs = self.estflux / self.cont_cgs

                self.eqw_obs_unc = abs(self.eqw_obs * np.sqrt(
                    (self.estflux_unc / self.estflux) ** 2 +
                    (self.cont_cgs_unc / self.cont_cgs) ** 2))

                #save for detailed reporting
                self.eqw_line_obs = self.eqw_obs
                self.eqw_line_obs_unc = self.eqw_obs_unc


            #used just below to choose between the two
            sdss_okay = 0
            hetdex_okay = 0

            # sum over entire HETDEX spectrum to estimate g-band magnitude
            try:
                self.hetdex_gmag, self.hetdex_gmag_cgs_cont, self.hetdex_gmag_unc,self.hetdex_gmag_cgs_cont_unc = \
                    elixer_spectrum.get_hetdex_gmag(self.sumspec_flux / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                    self.sumspec_wavelength,
                                                    self.sumspec_fluxerr / 2.0 * G.HETDEX_FLUX_BASE_CGS)

                log.debug(f"HETDEX spectrum gmag {self.hetdex_gmag} +/- {self.hetdex_gmag_unc}")
                log.debug(f"HETDEX spectrum cont {self.hetdex_gmag_cgs_cont} +/- {self.hetdex_gmag_cgs_cont_unc}")


                if (self.hetdex_gmag_cgs_cont is not None) and (self.hetdex_gmag_cgs_cont != 0) and not np.isnan(self.hetdex_gmag_cgs_cont):
                    if (self.hetdex_gmag_cgs_cont_unc is None) or np.isnan(self.hetdex_gmag_cgs_cont_unc):
                        self.hetdex_gmag_cgs_cont_unc = 0.0
                        hetdex_okay = 1
                    else:
                        hetdex_okay = 2

                    self.eqw_hetdex_gmag_obs = self.estflux / self.hetdex_gmag_cgs_cont
                    self.eqw_hetdex_gmag_obs_unc = abs(self.eqw_hetdex_gmag_obs * np.sqrt(
                        (self.estflux_unc / self.estflux) ** 2 +
                        (self.hetdex_gmag_cgs_cont_unc / self.hetdex_gmag_cgs_cont_unc) ** 2))

                if (self.hetdex_gmag is None) or np.isnan(self.hetdex_gmag):
                    hetdex_okay = 0
            except:
                hetdex_okay = 0
                log.error("Exception computing HETDEX spectrum gmag",exc_info=True)

            #feed HETDEX spectrum through SDSS gband filter
            try:
                #reminder needs erg/s/cm2/AA and sumspec_flux in ergs/s/cm2 so divied by 2AA bin width
#                self.sdss_gmag, self.cont_cgs = elixer_spectrum.get_sdss_gmag(self.sumspec_flux/2.0*1e-17,self.sumspec_wavelength)
                if False:
                    self.sdss_gmag, self.sdss_cgs_cont = elixer_spectrum.get_sdss_gmag(self.sumspec_flux / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                                                    self.sumspec_wavelength)
                    self.sdss_cgs_cont_unc = np.sqrt(np.sum(self.sumspec_fluxerr**2))/len(self.sumspec_fluxerr)*G.HETDEX_FLUX_BASE_CGS

                    log.debug(f"SDSS spectrum gmag {self.sdss_gmag} +/- {self.sdss_gmag_unc}")
                    log.debug(f"SDSS spectrum cont {self.sdss_cgs_cont} +/- {self.sdss_cgs_cont_unc}")

                else:
                    self.sdss_gmag, self.sdss_cgs_cont, self.sdss_gmag_unc, self.sdss_cgs_cont_unc =\
                        elixer_spectrum.get_sdss_gmag(self.sumspec_flux / 2.0 * G.HETDEX_FLUX_BASE_CGS, self.sumspec_wavelength,
                                                        self.sumspec_fluxerr / 2.0 * G.HETDEX_FLUX_BASE_CGS)

                    log.debug(f"SDSS spectrum gmag {self.sdss_gmag} +/- {self.sdss_gmag_unc}")
                    log.debug(f"SDSS spectrum cont {self.sdss_cgs_cont} +/- {self.sdss_cgs_cont_unc}")

                if (self.sdss_cgs_cont is not None) and (self.sdss_cgs_cont != 0) and not np.isnan(self.sdss_cgs_cont):
                    if (self.sdss_cgs_cont_unc is None) or np.isnan(self.sdss_cgs_cont_unc):
                        self.sdss_cgs_cont_unc = 0.0
                        sdss_okay = 1
                    else:
                        sdss_okay = 2

                    self.eqw_sdss_obs = self.estflux / self.sdss_cgs_cont
                    self.eqw_sdss_obs_unc = abs(self.eqw_sdss_obs * np.sqrt(
                        (self.estflux_unc / self.estflux) ** 2 +
                        (self.sdss_cgs_cont_unc / self.sdss_cgs_cont) ** 2))

                if (self.sdss_gmag is None) or np.isnan(self.sdss_gmag):
                    sdss_okay = 0

            except:
                sdss_okay = 0
                log.error("Exception computing SDSS g-mag",exc_info=True)


            #choose the best
            # even IF okay == 0, still record the probably bogus value (when
            # actually using the values elsewhere they are compared to a limit and the limit is used if needed
            if (hetdex_okay == sdss_okay) and (self.hetdex_gmag is not None) and (self.sdss_gmag is not None) and \
                abs(self.hetdex_gmag - self.sdss_gmag) < 1.0: #use both as an average? what if they are very different?
                #make the average
                avg_cont = 0.5 * (self.hetdex_gmag_cgs_cont + self.sdss_cgs_cont)
                avg_cont_unc =  np.sqrt(self.hetdex_gmag_cgs_cont_unc**2 + self.sdss_cgs_cont_unc**2) #error on the mean

                self.best_gmag_selected = 'mean'
                self.best_gmag = -2.5*np.log10(SU.cgs2ujy(avg_cont,4500.00) / 1e6 / 3631.)
                mag_faint = -2.5*np.log10(SU.cgs2ujy(avg_cont-avg_cont_unc,4500.00) / 1e6 / 3631.)
                mag_bright = -2.5*np.log10(SU.cgs2ujy(avg_cont+avg_cont_unc,4500.00) / 1e6 / 3631.)
                self.best_gmag_unc = 0.5 * (mag_faint-mag_bright)

                self.best_gmag_cgs_cont = avg_cont
                self.best_gmag_cgs_cont_unc = avg_cont_unc

                self.best_eqw_gmag_obs = self.estflux / self.best_gmag_cgs_cont
                self.best_eqw_gmag_obs_unc = abs(self.best_eqw_gmag_obs * np.sqrt(
                    (self.estflux_unc / self.estflux) ** 2 +
                    (self.best_gmag_cgs_cont_unc / self.best_gmag_cgs_cont) ** 2))

                log.debug("Using mean of HETDEX full width gmag and SDSS gmag.")
                log.info(f"Mean spectrum gmag {self.best_gmag:0.2f} +/- {self.best_gmag_unc:0.3f}; cont {self.best_gmag_cgs_cont} +/- {self.best_gmag_cgs_cont_unc}" )

            elif hetdex_okay >= sdss_okay > 0 and not np.isnan(self.hetdex_gmag_cgs_cont) and (self.hetdex_gmag_cgs_cont is not None):
                self.best_gmag_selected = 'hetdex'
                self.best_gmag = self.hetdex_gmag
                self.best_gmag_unc = self.hetdex_gmag_unc
                self.best_gmag_cgs_cont = self.hetdex_gmag_cgs_cont
                self.best_gmag_cgs_cont_unc = self.hetdex_gmag_cgs_cont_unc
                self.best_eqw_gmag_obs = self.eqw_hetdex_gmag_obs
                self.best_eqw_gmag_obs_unc = self.eqw_hetdex_gmag_obs_unc
                log.debug("Using HETDEX full width gmag over SDSS gmag.")
            elif sdss_okay > 0 and not np.isnan(self.sdss_cgs_cont) and (self.sdss_cgs_cont is not None):
                self.best_gmag_selected = 'sdss'
                self.best_gmag = self.sdss_gmag
                self.best_gmag_unc = self.sdss_gmag_unc
                self.best_gmag_cgs_cont = self.sdss_cgs_cont
                self.best_gmag_cgs_cont_unc = self.sdss_cgs_cont_unc
                self.best_eqw_gmag_obs = self.eqw_sdss_obs
                self.best_eqw_gmag_obs_unc = self.eqw_sdss_obs_unc
                log.debug("Using SDSS gmag over HETDEX full width gmag")
            else:
                log.debug("No full width spectrum g-mag estimate is valid.")
                self.best_gmag_selected = 'limit'
                self.best_gmag = G.HETDEX_CONTINUUM_MAG_LIMIT
                self.best_gmag_unc = 0
                self.best_gmag_cgs_cont = G.HETDEX_CONTINUUM_FLUX_LIMIT
                self.best_gmag_cgs_cont_unc = 0
                self.best_eqw_gmag_obs =  self.estflux / self.best_gmag_cgs_cont
                self.best_eqw_gmag_obs_unc = 0

            try:
                diff = abs(self.hetdex_gmag - self.sdss_gmag)
                unc = self.hetdex_gmag_unc + self.sdss_gmag_unc
                if (hetdex_okay == sdss_okay) and \
                    ((self.hetdex_gmag < G.HETDEX_CONTINUUM_MAG_LIMIT) or (self.sdss_gmag <  G.HETDEX_CONTINUUM_MAG_LIMIT)) and \
                    ((diff > unc) and (diff > 0.5)):
                    self.flags |= G.DETFLAG_DEXSPEC_GMAG_INCONSISTENT
                    log.info(f"DEX spectrum gmag disagree by {diff/unc:0.1f}x uncertainty. "
                             f"Dex g {self.hetdex_gmag:0.2f} +/- {self.hetdex_gmag_unc:0.3f} "
                             f"vs SDSS g {self.sdss_gmag:0.2f} +/- {self.sdss_gmag_unc:0.3f}")
            except:
                pass

            try:
                self.hetdex_cont_cgs = self.cont_cgs
                self.hetdex_cont_cgs_unc = self.cont_cgs_unc

                if self.cont_cgs == -9999: #still unset ... weird?
                    log.warning("Warning! HETDEX continuum estimate not set. Using best gmag for estimate(%g +/- %g)."
                                %(self.best_gmag_cgs_cont,self.best_gmag_cgs_cont_unc))

                    self.cont_cgs_narrow = self.cont_cgs
                    self.cont_cgs_narrow_unc = self.cont_cgs_unc
                    self.cont_cgs = self.best_gmag_cgs_cont
                    self.cont_cgs_unc = self.best_gmag_cgs_cont_unc
                    self.using_best_gmag_ew = True
                elif self.cont_cgs <= 0.0:
                    log.warning("Warning! HETDEX continuum <= 0.0. Using best gmag for estimate (%g +/- %g)."
                                %(self.best_gmag_cgs_cont,self.best_gmag_cgs_cont_unc))
                    self.cont_cgs_narrow = self.cont_cgs
                    self.cont_cgs_narrow_unc = self.cont_cgs_unc
                    self.cont_cgs = self.best_gmag_cgs_cont
                    self.cont_cgs_unc = self.best_gmag_cgs_cont_unc
                    self.using_best_gmag_ew = True
            except:
                pass

            #todo: should check for cont < 0? if so, try my MCMCfit?
            if self.cont_cgs != 0:
                self.eqw_obs = self.estflux / self.cont_cgs

                self.eqw_obs_unc = abs(self.eqw_obs * np.sqrt(
                    (self.estflux_unc / self.estflux) ** 2 +
                    (self.cont_cgs_unc / self.cont_cgs) ** 2))

                self.spec_obj.est_g_cont = self.cont_cgs

            if (G.CONTINUUM_RULES or self.cont_cgs > G.CONTNIUUM_RULES_THRESH) and (self.w is None) or (self.w == 0.0):

                # find the "best" wavelength to use as the central peak
                #spectrum = elixer_spectrum.Spectrum()
                if self.spec_obj is None:
                    self.spec_obj = elixer_spectrum.Spectrum()

                #self.all_found_lines = elixer_spectrum.peakdet(self.sumspec_wavelength,self.sumspec_flux,self.sumspec_fluxerr,values_units=-17)
                w = self.spec_obj.find_central_wavelength(self.sumspec_wavelength,self.sumspec_flux,
                                                                          self.sumspec_fluxerr,-17,return_list=False)
                if w is not None and (3400.0 < w < 5600.0):
                    self.w = w
                    self.target_wavelength = w
                    if (self.spec_obj.central_eli is not None):
                        self.sigma = self.spec_obj.central_eli.fit_sigma
                        self.estflux = self.spec_obj.central_eli.line_flux
                        self.cont_cgs = self.spec_obj.central_eli.cont
                        self.cont_cgs_unc = self.spec_obj.central_eli.cont_err

                        if self.spec_obj.central_eli.mcmc_x0:
                            self.w_unc = (self.spec_obj.central_eli.mcmc_x0[1] + self.spec_obj.central_eli.mcmc_x0[2])/2.0
                        if self.spec_obj.central_eli.mcmc_sigma:
                            self.sigma_unc = (self.spec_obj.central_eli.mcmc_sigma[1] + self.spec_obj.central_eli.mcmc_sigma[2])/2.0
                        self.estflux_unc = self.spec_obj.central_eli.line_flux_err

                        self.line_gaussfit_parms = (self.w,self.sigma,self.estflux*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
                                                    self.cont_cgs/G.HETDEX_FLUX_BASE_CGS,G.FLUX_WAVEBIN_WIDTH) #*2.0 for Karl's bin width
                        self.line_gaussfit_unc = (self.w_unc,self.sigma_unc,self.estflux_unc*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
                                                  self.cont_cgs_unc/G.HETDEX_FLUX_BASE_CGS, 0.0)

                else:

                    try:
                        self.w = self.sumspec_wavelength[np.nanargmax(self.sumspec_flux)]
                        self.target_wavelength = self.w
                        log.info(f"Cannot identify a suitable target wavelength. Setting to maximum flux value ({self.w}).")
                    except:
                        #print("Cannot identify a suitable target wavelength. Arbitrarly setting to 4500.0 for report.")
                        log.info("Cannot identify a suitable target wavelength. Arbitrarly setting to 4500.0 for report.")
                        self.w = 4500.0
                        self.target_wavelength = 4500.0


            if self.w is not None and self.w != 0:
                idx = elixer_spectrum.getnearpos(self.sumspec_wavelength, self.w)
            else:
                try:
                    idx = np.argmax(self.sumspec_flux)
                except:
                    idx = 0
            left = idx - 25  # 2AA steps so +/- 50AA
            right = idx + 25

            if left < 0:
                left = 0
            if right > len(self.sumspec_flux):
                right = len(self.sumspec_flux)

            # these are on the 2AA grid (old spece had 2AA steps but, the grid was centered on the main wavelength)
            # this grid is not centered but is on whole 2AA (i.e. 3500.00, 3502.00, ... not 3500.4192, 3502.3192, ...)
            self.sumspec_wavelength_zoom = self.sumspec_wavelength[left:right]
            self.sumspec_flux_zoom = self.sumspec_flux[left:right]
            self.sumspec_fluxerr_zoom = self.sumspec_fluxerr[left:right]
            self.sumspec_counts_zoom = self.sumspec_counts[left:right]


            #######################################
            #get individual fiber info
            #######################################
            #   idstring=None,specid=None,ifuslot=None,ifuid=None,amp=None,date=None,time=None,time_ex=None,
             #    panacea_fiber_index=-1, detect_id = -1):

            log.debug(f"Loading base fiber data from HDF5 ({id})...")
            rows = fiber_table.read_where("detectid == id")
            subset_norm = 0.0 #for the relative weights

            num_fibers = rows.size
            count = 0
            duplicate_count = 0

            for row in rows:
                count += 1

                if G.python2():
                    specid = row['specid']
                    ifuslot = row['ifuslot']
                    ifuid = row['ifuid']
                    amp = row['amp']
                    date = str(row['date'])

                    #expected to be "20180320T052104.2"
                    time_ex = row['timestamp'][9:]
                    time = time_ex[0:6] #hhmmss
                    mfits_name = row['multiframe'] #similar to multi*fits style name
                    fiber_index = row['fibnum'] -1  #1-112 #panacea index is 0-111
                else:
                    specid = row['specid'].decode()
                    ifuslot = row['ifuslot'].decode()
                    ifuid = row['ifuid'].decode()
                    amp = row['amp'].decode()
                    date = str(row['date'])

                    # expected to be "20180320T052104.2"
                    time_ex = row['timestamp'][9:].decode()
                    time = time_ex[0:6]  # hhmmss
                    mfits_name = row['multiframe'].decode()  # similar to multi*fits style name
                    fiber_index = row['fibnum'] - 1  # 1-112 #panacea index is 0-111

                #print("***** ", mfits_name, specid, ifuslot, ifuid, amp, row['expnum'])

                #sanity check
                try:
                    if mfits_name[-2:] != amp:
                        log.warning("hetdex.py amp string comparision mismatch: %s != %s",(mfits_name[-2:],amp))

                except:
                    log.debug("hetdex.py amp string comparision failed",exec_info=True)



                idstring = date + "v" + time_ex + "_" + specid + "_" + ifuslot + "_" + ifuid + "_" + amp + "_" #leave off the fiber for the moment
                log.debug("Building fiber %d of %d (%s e%d) ..." % (count, num_fibers,idstring + str(fiber_index+1),int(row['expnum'])))
                idstring += str(fiber_index) #add the fiber index (zero based)

                fiber = elixer_fiber.Fiber(idstring=idstring,specid=specid,ifuslot=ifuslot,ifuid=ifuid,amp=amp,
                                           date=date,time=time,time_ex=time_ex, panacea_fiber_index=fiber_index,
                                           detect_id=id)

                if fiber is not None:
                    duplicate = False
                    fiber.ra = row['ra']
                    fiber.dec = row['dec']
                    fiber.obsid = int(row['obsid'])
                    fiber.shotid = date + str(fiber.obsid).zfill(3)
                    fiber.expid = int(row['expnum']) # integer now
                    #fiber.shotid =
                    fiber.detect_id = id
                    fiber.center_x = row['x_ifu']
                    fiber.center_y = row['y_ifu']
                    fiber.raw_weight = row['weight']

                    #check that this is NOT a duplicate
                    for i in range(len(self.fibers)):
                        if fiber == self.fibers[i]:
                            log.warning("Warning! Duplicate Fiber in detectID %s: %s . idx %d == %d. Duplicate will not be processed." %
                                        (str(self.hdf5_detectid),fiber.idstring, i, count-1))
                            duplicate = True
                            duplicate_count += 1
                            break

                    if duplicate:
                        continue #continue on to next fiber


                    subset_norm += fiber.raw_weight
                    #fiber.relative_weight = row['weight']
                    # add the fiber (still needs to load its fits file)
                    # we already know the path to it ... so do that here??

                    #full path to the HDF5 fits equivalent (or failing that the panacea fits file?)
                    fiber.fits_fn = fiber.find_hdf5_multifits(loc=op.dirname(hdf5_fn))

                    #fiber.fits_fn = get_hetdex_multifits_path(fiber.)

                    #now, get the corresponding FITS or FITS equivalent (HDF5)
                    #if self.annulus is None:
                    if True: #may want this anyway to plot up the central object
                        fits = hetdex_fits.HetdexFits(empty=True)
                        #populate the data we need to read the HDF5 file
                        fits.filename = fiber.fits_fn #mfits_name #todo: fix to the corect path
                        fits.multiframe = mfits_name
                        fits.panacea = True
                        fits.hdf5 = True

                        fits.obsid = str(fiber.obsid).zfill(3)
                        fits.expid = int(fiber.expid)
                        fits.specid = str(fiber.specid).zfill(3)
                        fits.ifuslot = str(fiber.ifuslot).zfill(3)
                        fits.ifuid = str(fiber.ifuid).zfill(3)
                        fits.amp = fiber.amp
                        fits.side = fiber.amp[0]

                        fits.obs_date = fiber.dither_date
                        fits.obs_ymd = fits.obs_date

                        # now read the HDF5 equivalent
                        #if we don't already have it
                        already_read = False
                        for fi in self.fibers:
                            #same filename (same DateVshot + observation) + same (IFU address) + same exposure
                            try:
                                if (fi.fits is not None) and (fi.fits.filename == fits.filename) and \
                                   (fi.fits.multiframe == fits.multiframe) and (fi.fits.expid == fits.expid):
                                    fiber.fits = fi.fits
                                    already_read = True
                                    break
                            except:
                                log.error("Exception in DetObj.load_hdf5_fluxcalibrated_spectra",exc_info=True)

                        if not already_read:
                            fits.read_hdf5()
                             # check if it is okay
                            if fits.okay:
                                fiber.fits = fits
                            else:
                                log.error("HDF5 multi-fits equivalent is not okay ...")

                    self.fibers.append(fiber)

        if duplicate_count != 0:
            print("Warning! Duplicate Fibers found %d / %d" %(duplicate_count, count))
            log.warning("Warning! Duplicate Fibers found %d / %d" %(duplicate_count, count))
            self.duplicate_fibers_removed = 1

        #more to do here ... get the weights, etc (see load_flux_calibrated spectra)
        if True: #go ahead and do this anyway, may want to plot central object info/spectra
        #if self.annulus is None: #the weights are meaningless for an annulus report
            if subset_norm != 0:
                for f in self.fibers:
                    f.relative_weight = f.raw_weight/subset_norm

            #sort the fibers by weight
            self.fibers = [x for x in self.fibers if x.relative_weight > 0]
            self.fibers.sort(key=lambda x: x.relative_weight, reverse=True)  # highest weight is index = 0
            self.fibers_sorted = True

            # self.fibers.sort(key=lambda x: x.relative_weight, reverse=False)  # highest weight is index = 0
            # for f in self.fibers:
            #     print(f"{f.ra:0.5f} {f.dec:0.5f} \t ----- \t {f.raw_weight:0.2f} \t {f.relative_weight:0.2f}")
            #
            #



            #build a noise estimate over the top 4 fibers (amps)?
            try:
                good_idx = np.where([x.fits for x in self.fibers])[0] #some might be None, so get those that are not
                good_idx = good_idx[0:min(len(good_idx),4)]

                all_calfib = np.concatenate([self.fibers[i].fits.calfib for i in good_idx],axis=0)

                #use the std dev of all "mostly empty" (hence sigma=3.0) or "sky" fibers as the error
                mean, median, std = sigma_clipped_stats(all_calfib, axis=0, sigma=3.0)
                self.calfib_noise_estimate = std
                if not G.MULTILINE_USE_ERROR_SPECTRUM_AS_NOISE:
                    self.spec_obj.noise_estimate = self.calfib_noise_estimate
                    self.spec_obj.noise_estimate_wave = G.CALFIB_WAVEGRID

            except Exception as e:
                self.calfib_noise_estimate = np.zeros(len(G.CALFIB_WAVEGRID))
                try:
                    if e.args[0].find("ValueError: need at least one array to concatenate"):
                        log.info("Could not build DetObj calfib_noise_estimate")
                    else:
                        log.info(f"Could not build DetObj calfib_noise_estimate: {e}" )
                except:
                    log.info("Could not build DetObj calfib_noise_estimate", exc_info=True)


        self.spec_obj.identifier = "eid(%s)" %str(self.entry_id)
        self.spec_obj.plot_dir = self.outdir

        # if True:
        #     print("***** REMOVE ME ******")
        #
        #     def compute_model(x,mu, sigma, A, y):
        #         try:
        #             return A * (np.exp(-np.power((x - mu) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y
        #         except:
        #             return np.nan
        #
        #     bin_width = 2.0
        #     sigma_width = 2.0
        #     lineflux_adjust = 0.955 #ie. as sigma goes down... 1-sigma = 0.682, 2-sigma = 0.954, 3-sigma = 0.996
        #     left,*_ = SU.getnearpos(self.sumspec_wavelength,self.w-self.sigma*sigma_width)
        #     right,*_ = SU.getnearpos(self.sumspec_wavelength,self.w+self.sigma*sigma_width)
        #
        #     if self.sumspec_wavelength[left] - (self.w-self.sigma*sigma_width) < 0:
        #         left += 1 #less than 50% overlap in the left bin, so move one bin to the red
        #     if self.sumspec_wavelength[right] - (self.w+self.sigma*sigma_width) > 0:
        #         right -=1 #less than 50% overlap in the right bin, so move one bin to the blue
        #
        #     #lastly ... could combine, but this is easier to read
        #     right += 1 #since the right index is not included in slice
        #
        #     # if left > 0:
        #     #     left -= 1
        #     #right = min(right+1,len(self.sumspec_wavelength)) #+2
        #     #at 4 sigma the mcmc_A[0] is almost identical to the model_fit (as you would expect)
        #     #note: if choose to sum over model fit, remember that this is usually over 2AA wide bins, so to
        #     #compare to the error data, need to multiply the model_sum by the bin width (2AA)
        #     #(or noting that the Area == integrated flux x binwidth)
        #
        #
        #     model_fit = compute_model(self.sumspec_wavelength[left:right],
        #                               self.line_gaussfit_parms[0],
        #                               self.line_gaussfit_parms[1],
        #                               self.line_gaussfit_parms[2],
        #                               self.line_gaussfit_parms[3])
        #
        #
        #     data_err = copy(self.sumspec_fluxerr[left:right])
        #     data_err[data_err<=0] = np.nan #Karl has 0 value meaning it is flagged and should be skipped
        #
        #     print(f"***** SN Test: target {self.snr} at {self.w} for sigma = {self.sigma} and flux = {self.estflux}")
        #     # print(f"***** SN Test: {np.sum(self.sumspec_flux[left:right]-self.cont_cgs*1e17)/np.sqrt(np.sum(data_err**2))}")
        #     # print(f"***** SN Test: {(lineflux_adjust*self.estflux*1e17)/np.sqrt(np.sum((self.sumspec_fluxerr[left:right]/self.sumspec_flux[left:right])**2))}")
        #     #     #also pretty close ... error propogation for division ... data flux / data error as S/N per 2AA pixel
        #     # print()
        #     print(f"***** SN Test: {np.sum(model_fit-self.line_gaussfit_parms[3])/np.sqrt(np.nansum(data_err**2))} ... continuum sub")
        #     # #reminder to myself ...in my MCMC the AREA is 2x, so where I divide by 2 there, I multiply by 2 here
        #     # #since the flux is already corrected (or sqrt(2) as is the case since it logically belongs on the sum)
        #     # print(f"***** SN Test: {self.estflux*1e17/np.sqrt(np.nansum(data_err**2))*np.sqrt(2)}") #doing best
        #     #
        #     # print("++++++++++++++")
        #     # print(f"***** SN Test: {lineflux_adjust*self.estflux*1e17/np.sqrt(np.nansum(data_err**2))} ... straight")
        #     # print(f"***** SN Test: {np.sum(model_fit)/np.sqrt(np.nansum(data_err**2))} ... straight model")
        #     # print(f"***** SN Test: {lineflux_adjust*self.estflux*1e17/np.sqrt(np.nansum((apcor*data_err)**2))/(np.nanmean(apcor))} ... apcor")
        #     # print("++++++++++++++")
        #     #
        #     # print(f"***** SN Test: {lineflux_adjust*self.estflux*1e17/(np.nansum(data_err))}")
        #     # print(f"***** SN Test: {lineflux_adjust*2*self.estflux*1e17/np.sqrt(np.nansum(data_err))}") #*2 since in 2AA bins?
        #     # print()
        #
        #     #this is just wrong (the integrated flux already handle the removal of the y-offset
        #     # print(f"***** SN Test: {(self.estflux*1e17-self.cont_cgs*1e17)/(np.sum(data_err))}")
        #     # print(f"***** SN Test: {(self.estflux*1e17-self.cont_cgs*1e17)/np.sqrt(np.sum(data_err**2))*np.sqrt(2)}")
        #     # print(f"***** SN Test: {(self.estflux*1e17-self.cont_cgs*1e17)/np.sqrt(np.sum(data_err**2))}")


        if True: #go ahead and do this anyway, may want to plot central object info/spectra
        #if self.annulus is None:
            self.spec_obj.set_spectra(self.sumspec_wavelength,self.sumspec_flux,self.sumspec_fluxerr, self.w,
                                      values_units=-17, estflux=self.estflux, estflux_unc=self.estflux_unc,
                                      eqw_obs=self.eqw_obs,eqw_obs_unc=self.eqw_obs_unc,
                                      estcont=self.cont_cgs,estcont_unc=self.cont_cgs_unc,
                                      fwhm=self.fwhm,fwhm_unc=self.fwhm_unc,
                                      continuum_g=self.best_gmag_cgs_cont,continuum_g_unc=self.best_gmag_cgs_cont_unc)
            # print("DEBUG ... spectrum peak finder")
            # if G.DEBUG_SHOW_GAUSS_PLOTS:
            #    self.spec_obj.build_full_width_spectrum(show_skylines=True, show_peaks=True, name="testsol")
            # print("DEBUG ... spectrum peak finder DONE")

            #update DEX-g based continuum and EW
            try:
                self.best_gmag_cgs_cont *= self.spec_obj.gband_continuum_correction()
                self.best_gmag_cgs_cont_unc *= self.spec_obj.gband_continuum_correction()

                self.best_eqw_gmag_obs = self.estflux / self.best_gmag_cgs_cont
                self.best_eqw_gmag_obs_unc = abs(self.best_eqw_gmag_obs * np.sqrt(
                    (self.estflux_unc / self.estflux) ** 2 +
                    (self.best_gmag_cgs_cont_unc / self.best_gmag_cgs_cont) ** 2))

                log.info(f"Update best DEX-g continuum x{self.spec_obj.gband_continuum_correction():0.2f} and EW; cont {self.best_gmag_cgs_cont} +/- {self.best_gmag_cgs_cont_unc}" )
            except:
                log.error("Exception! Excpetion updating DEX-g continuum.",exc_info=True)

    #update with MY FIT results?
            central_wave_volatile = False
            if (self.spec_obj is not None and self.spec_obj.central_eli is not None) and \
                (G.REPORT_ELIXER_MCMC_FIT or (self.eqw_obs == 0) or G.CONTINUUM_RULES):
                log.info("Using ELiXer MCMC Fit for line flux, continuum, EW, and SNR")
                central_wave_volatile = True #CAN change te central wave after classifing
                try:
                    self.estflux = self.spec_obj.central_eli.mcmc_line_flux
                    self.eqw_obs = self.spec_obj.central_eli.mcmc_ew_obs[0]
                    self.cont_cgs = self.spec_obj.central_eli.mcmc_continuum
                    #self.snr = self.spec_obj.central_eli.mcmc_snr
                    self.snr = self.spec_obj.central_eli.snr
                    self.snr_unc = self.spec_obj.central_eli.snr_err

                    self.spec_obj.estflux = self.estflux
                    self.spec_obj.eqw_obs = self.eqw_obs

                    #self.estflux = self.spec_obj.central_eli.line_flux
                    #self.cont = self.spec_obj.central_eli.cont
                    #self.eqw_obs = self.estflux / self.cont
                    #self.snr = self.spec_obj.central_eli.snr

                    #if no mcmc try the fit?
                    self.w = self.spec_obj.central_eli.mcmc_x0[0]
                    self.w_unc = 0.5 * (self.spec_obj.central_eli.mcmc_x0[1] + self.spec_obj.central_eli.mcmc_x0[2])
                    self.sigma = self.spec_obj.central_eli.mcmc_sigma[0]
                    self.sigma_unc = 0.5 * (self.spec_obj.central_eli.mcmc_sigma[1]+self.spec_obj.central_eli.mcmc_sigma[2])
                    self.chi2 = self.spec_obj.central_eli.mcmc_chi2
                    self.estflux = self.spec_obj.central_eli.mcmc_line_flux
                    self.estflux_unc = 0.5 * (self.spec_obj.central_eli.mcmc_line_flux_tuple[1] + self.spec_obj.central_eli.mcmc_line_flux_tuple[2])


                    self.cont_cgs = self.spec_obj.central_eli.mcmc_continuum
                    self.cont_cgs_unc = 0.5 * (self.spec_obj.central_eli.mcmc_continuum_tuple[1]+self.spec_obj.central_eli.mcmc_continuum_tuple[2])
                    self.cont_cgs_narrow = self.spec_obj.central_eli.mcmc_continuum
                    self.cont_cgs_unc_narrow = self.cont_cgs_unc

                    self.snr = self.spec_obj.central_eli.mcmc_snr
                    self.snr_unc = self.spec_obj.central_eli.mcmc_snr_err

                    self.eqw_obs = self.spec_obj.central_eli.mcmc_ew_obs[0]
                    self.eqw_obs_unc = 0.5 * (self.spec_obj.central_eli.mcmc_ew_obs[1]+self.spec_obj.central_eli.mcmc_ew_obs[2])

                    #fluxes are in e-17
                    self.line_gaussfit_parms = (self.w,self.sigma,self.estflux*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
                                                self.cont_cgs/G.HETDEX_FLUX_BASE_CGS,G.FLUX_WAVEBIN_WIDTH)
                    self.line_gaussfit_unc = (self.w_unc,self.sigma_unc,self.estflux_unc*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
                                              self.cont_cgs_unc/G.HETDEX_FLUX_BASE_CGS,0.0)


                except:
                    log.warning("No MCMC data to update core stats in hetdex::load_flux_calibrated_spectra",exc_info=True)

            self.spec_obj.classify(known_z=self.known_z,continuum_limit=max(self.best_gmag_cgs_cont, G.HETDEX_CONTINUUM_FLUX_LIMIT),
                                   continuum_limit_err=self.best_gmag_cgs_cont_unc) #solutions can be returned, also stored in spec_obj.solutions

            # if central_wave_volatile and (self.spec_obj.central_eli.w_obs != self.w):
            #     try:
            #         self.estflux = self.spec_obj.central_eli.mcmc_line_flux
            #         self.eqw_obs = self.spec_obj.central_eli.mcmc_ew_obs[0]
            #         self.cont_cgs = self.spec_obj.central_eli.mcmc_continuum
            #         #self.snr = self.spec_obj.central_eli.mcmc_snr
            #         self.snr = self.spec_obj.central_eli.snr
            #         self.snr_unc = self.spec_obj.central_eli.snr_err
            #
            #         self.spec_obj.estflux = self.estflux
            #         self.spec_obj.eqw_obs = self.eqw_obs
            #
            #         #self.estflux = self.spec_obj.central_eli.line_flux
            #         #self.cont = self.spec_obj.central_eli.cont
            #         #self.eqw_obs = self.estflux / self.cont
            #         #self.snr = self.spec_obj.central_eli.snr
            #
            #         #if no mcmc try the fit?
            #         self.w = self.spec_obj.central_eli.mcmc_x0[0]
            #         self.w_unc = 0.5 * (self.spec_obj.central_eli.mcmc_x0[1] + self.spec_obj.central_eli.mcmc_x0[2])
            #         self.sigma = self.spec_obj.central_eli.mcmc_sigma[0]
            #         self.sigma_unc = 0.5 * (self.spec_obj.central_eli.mcmc_sigma[1]+self.spec_obj.central_eli.mcmc_sigma[2])
            #         self.chi2 = self.spec_obj.central_eli.mcmc_chi2
            #         self.estflux = self.spec_obj.central_eli.mcmc_line_flux
            #         self.estflux_unc = 0.5 * (self.spec_obj.central_eli.mcmc_line_flux_tuple[1] + self.spec_obj.central_eli.mcmc_line_flux_tuple[2])
            #
            #
            #         self.cont_cgs = self.spec_obj.central_eli.mcmc_continuum
            #         self.cont_cgs_unc = 0.5 * (self.spec_obj.central_eli.mcmc_continuum_tuple[1]+self.spec_obj.central_eli.mcmc_continuum_tuple[2])
            #         self.cont_cgs_narrow = self.spec_obj.central_eli.mcmc_continuum
            #         self.cont_cgs_unc_narrow = self.cont_cgs_unc
            #
            #         self.snr = self.spec_obj.central_eli.mcmc_snr
            #         self.snr_unc = self.spec_obj.central_eli.mcmc_snr_err
            #
            #         self.eqw_obs = self.spec_obj.central_eli.mcmc_ew_obs[0]
            #         self.eqw_obs_unc = 0.5 * (self.spec_obj.central_eli.mcmc_ew_obs[1]+self.spec_obj.central_eli.mcmc_ew_obs[2])
            #
            #         #fluxes are in e-17
            #         self.line_gaussfit_parms = (self.w,self.sigma,self.estflux*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
            #                                     self.cont_cgs/G.HETDEX_FLUX_BASE_CGS,G.FLUX_WAVEBIN_WIDTH)
            #         self.line_gaussfit_unc = (self.w_unc,self.sigma_unc,self.estflux_unc*G.FLUX_WAVEBIN_WIDTH/G.HETDEX_FLUX_BASE_CGS,
            #                                   self.cont_cgs_unc/G.HETDEX_FLUX_BASE_CGS,0.0)
            #
            #
            #     except:
            #         log.warning("Exception! Unable to update central line following classification.",exc_info=True)
            #

        if (self.w is None or self.w == 0) and (self.spec_obj is not None):
            try:
                self.w = self.spec_obj.central
            except:
                pass


        self.rvb = SU.red_vs_blue(self.w,self.sumspec_wavelength,
                                      self.sumspec_flux/G.FLUX_WAVEBIN_WIDTH*G.HETDEX_FLUX_BASE_CGS,
                                      self.sumspec_fluxerr/G.FLUX_WAVEBIN_WIDTH*G.HETDEX_FLUX_BASE_CGS,self.fwhm)

        if self.annulus:
            self.syn_obs = elixer_observation.SyntheticObservation()
            self.syn_obs.extraction_ffsky = self.extraction_ffsky
            if self.wra:
                self.syn_obs.ra = self.wra
                self.syn_obs.dec = self.wdec
            else:
                self.syn_obs.ra = self.ra
                self.syn_obs.dec = self.dec

            self.syn_obs.target_wavelength = self.target_wavelength
            self.syn_obs.annulus = self.annulus
            #self.syn_obs.fibers_all = self.fibers
            self.syn_obs.w = self.target_wavelength
            self.syn_obs.survey_shotid = self.survey_shotid
            #get all the fibers inside the outer annulus radius
            self.syn_obs.get_aperture_fibers()

        try:
            self.ccd_adjacent_fibers, self.ccd_adjacent_single_fiber_brightest_mag = \
                self.calc_ccd_adjacent_fiber_magnitudes(self.find_ccd_adjacent_fibers())
            self.central_single_fiber_mag = self.calc_central_single_fiber_magnitude()
        except:
            pass

        return
    #nd load_hdf5

    def get_probabilities(self):

        #if we have a flux calibrated, already classified spectra, use that
        #(this is the normal case)
        if self.spec_obj.p_lae_oii_ratio is not None:
            self.p_lae = self.spec_obj.p_lae
            self.p_oii = self.spec_obj.p_oii
            self.p_lae_oii_ratio = self.spec_obj.p_lae_oii_ratio
            self.p_lae_oii_ratio_range = self.spec_obj.p_lae_oii_ratio_range

        #otherwise, build with what we've got (note: does not have additional lines, in this case)
        elif self.w < 10: #really < 3500 is nonsense
            log.info("Warning. Cannot calculate p(LAE) ... no wavelength defined.")
            #can't calculate ... no wavelength
            self.p_lae = -1
            self.p_oii = -1
            self.p_lae_oii_ratio = -1
        else:
            ratio, self.p_lae, self.p_oii, plae_errors = line_prob.mc_prob_LAE(wl_obs=self.w,
                                                               lineFlux=self.estflux,
                                                               lineFlux_err=self.estflux_unc,
                                                               continuum=self.cont_cgs,
                                                               continuum_err=self.cont_cgs_unc,
                                                               c_obs=None, which_color=None,
                                                               addl_fluxes=[], addl_wavelengths=[],
                                                               sky_area=None,
                                                               cosmo=None, lae_priors=None,
                                                               ew_case=None, W_0=None,
                                                               z_OII=None, sigma=None)

            if (self.p_lae is not None) and (self.p_lae > 0.0):
                if (self.p_oii is not None) and (self.p_oii > 0.0):
                    self.p_lae_oii_ratio = self.p_lae /self.p_oii
                else:
                    self.p_lae_oii_ratio = float('inf')
            else:
                self.p_lae_oii_ratio = 0.0

            self.p_lae_oii_ratio = min(line_prob.MAX_PLAE_POII,self.p_lae_oii_ratio) #cap to MAX
            try:
                if plae_errors:
                        self.p_lae_oii_ratio_range = plae_errors['ratio']
            except:
                pass


        # check the BEST gmag version
        if self.best_gmag_p_lae_oii_ratio is None:
            try:

                if self.spec_obj:
                    addl_wavelengths = self.spec_obj.addl_wavelengths
                    addl_fluxes = self.spec_obj.addl_fluxes
                    addl_errors = self.spec_obj.addl_fluxerrs
                else:
                    addl_wavelengths = []
                    addl_fluxes = []
                    addl_errors = []

                best_ratio, best_p_lae, best_p_oii, plae_errors = line_prob.mc_prob_LAE(wl_obs=self.w,
                                                                                              lineFlux=self.estflux,
                                                                                              lineFlux_err=self.estflux_unc,
                                                                                              continuum=self.best_gmag_cgs_cont,
                                                                                              continuum_err=self.best_gmag_cgs_cont_unc,
                                                                                              # todo: get error est for sdss gmag
                                                                                              c_obs=None,
                                                                                              which_color=None,
                                                                                              addl_fluxes=[],
                                                                                              addl_wavelengths=[],
                                                                                              sky_area=None,
                                                                                              cosmo=None,
                                                                                              lae_priors=None,
                                                                                              ew_case=None, W_0=None,
                                                                                              z_OII=None, sigma=None)

                self.best_gmag_p_lae_oii_ratio = best_ratio
                try:
                    if plae_errors:
                        if (len(plae_errors['ratio']) >= 3):
                            #[0] = biweight location, [1] is low (or minus the scale), [2] is high,
                            #[3] is adjusted std dev or 1/2 * (16% to 84% range)
                            self.best_gmag_p_lae_oii_ratio_range = plae_errors['ratio']
                except:
                    pass


            except:
                log.info("Exception in hetdex.py DetObj::get_probabilities() for best gmag PLAE/POII: ",
                         exc_info=True)


        # #check the sdss_gmag version
        # if self.sdss_gmag_p_lae_oii_ratio is None:
        #     try:
        #         if self.spec_obj:
        #             addl_wavelengths = self.spec_obj.addl_wavelengths
        #             addl_fluxes = self.spec_obj.addl_fluxes
        #             addl_errors = self.spec_obj.addl_fluxerrs
        #         else:
        #             addl_wavelengths = []
        #             addl_fluxes = []
        #             addl_errors = []
        #
        #         sdss_ratio, sdss_p_lae, sdss_p_oii, plae_errors = line_prob.mc_prob_LAE(wl_obs=self.w,
        #                                                                            lineFlux=self.estflux,
        #                                                                            lineFlux_err=self.estflux_unc,
        #                                                                            continuum=self.sdss_cgs_cont,
        #                                                                            continuum_err=self.sdss_cgs_cont_unc,
        #                                                                            c_obs=None, which_color=None,
        #                                                                            addl_fluxes=[], addl_wavelengths=[],
        #                                                                            sky_area=None,
        #                                                                            cosmo=None, lae_priors=None,
        #                                                                            ew_case=None, W_0=None,
        #                                                                            z_OII=None, sigma=None)
        #
        #         self.sdss_gmag_p_lae_oii_ratio = sdss_ratio
        #         try:
        #             if plae_errors and (len(plae_errors['ratio']) > 2):
        #                 self.sdss_gmag_p_lae_oii_ratio_range = plae_errors['ratio']
        #         except:
        #             pass
        #
        #
        #     except:
        #         log.info("Exception in hetdex.py DetObj::get_probabilities() for sdss_gmag PLAE/POII: ", exc_info=True)
        #
        # # check the hetdex version
        # if self.hetdex_gmag_p_lae_oii_ratio is None:
        #     try:
        #
        #         if self.spec_obj:
        #             addl_wavelengths = self.spec_obj.addl_wavelengths
        #             addl_fluxes = self.spec_obj.addl_fluxes
        #             addl_errors = self.spec_obj.addl_fluxerrs
        #         else:
        #             addl_wavelengths = []
        #             addl_fluxes = []
        #             addl_errors = []
        #
        #         hetdex_ratio, hetdex_p_lae, hetdex_p_oii, plae_errors = line_prob.mc_prob_LAE(wl_obs=self.w,
        #                                                                                 lineFlux=self.estflux,
        #                                                                                 lineFlux_err=self.estflux_unc,
        #                                                                                 continuum=self.hetdex_gmag_cgs_cont,
        #                                                                                 continuum_err=self.hetdex_gmag_cgs_cont_unc,
        #                                                                                 # todo: get error est for sdss gmag
        #                                                                                 c_obs=None,
        #                                                                                 which_color=None,
        #                                                                                 addl_fluxes=[],
        #                                                                                 addl_wavelengths=[],
        #                                                                                 sky_area=None,
        #                                                                                 cosmo=None,
        #                                                                                 lae_priors=None,
        #                                                                                 ew_case=None, W_0=None,
        #                                                                                 z_OII=None, sigma=None)
        #
        #         self.hetdex_gmag_p_lae_oii_ratio = hetdex_ratio
        #         try:
        #             if plae_errors and (len(plae_errors['ratio']) == 3):
        #                 self.hetdex_gmag_p_lae_oii_ratio_range = plae_errors['ratio']
        #         except:
        #             pass
        #
        #
        #     except:
        #         log.info("Exception in hetdex.py DetObj::get_probabilities() for hetdex full width PLAE/POII: ",
        #                  exc_info=True)
        #
        # #assign the "best"
        # if self.best_gmag_selected == "sdss":
        #     self.best_gmag_p_lae_oii_ratio = self.sdss_gmag_p_lae_oii_ratio
        #     self.best_gmag_p_lae_oii_ratio_range =  self.sdss_gmag_p_lae_oii_ratio_range
        # else:
        #     self.best_gmag_p_lae_oii_ratio = self.hetdex_gmag_p_lae_oii_ratio
        #     self.best_gmag_p_lae_oii_ratio_range =  self.hetdex_gmag_p_lae_oii_ratio_range
        #
        #


    def parse_fiber(self,fiber):
        #this might be a fiber id string or something else from a detection line, so check the toks before moving on
        if fiber is None:
            return False
        #20170326T105655.6_032_094_028_LU_032

        toks = fiber.split("_")

        if len(toks) != 6:
            if (len(toks) == 1) and (toks[0] == "666"):
                    return True #this is an "ignore" flag, but still continue as if it were a fiber
            else:
                pass #stop bothering with this ... it is always there
                #log.warning("Unexpected fiber id string: %s" % fiber)
            return False

        newfiber = elixer_fiber.Fiber(fiber, detect_id=self.id)

        if newfiber is not None:
            self.fibers.append(newfiber)
            return True

        #essentially the else path
        idstring = fiber #toks[0] #ie. 20170326T105655.6

        dither_date = idstring[0:8]
        # next should be 'T'
        dither_time = idstring[9:15]  # not the .# not always there
        if idstring[15] == ".":
            dither_time_extended = idstring[9:17]
        else:
            dither_time_extended = None

        specid = toks[1]
        ifuslot = toks[2]
        ifuid = toks[3]
        amp = toks[4]
        #fiber_idx = toks[5] #note: this is the INDEX from panacea, not the relative fiberm but karl adds 1 to it
        # (ie. fiber #1 = index 111, fiber #2 = index 110 ... fiber #112 = index 0)
        fiber_idx = int(toks[5])-1


        #validate info
        if (ifuslot != self.ifuslot):
            log.error("Mismatched fiber id string. Does not match expected ifuslot id %s vs %s"
                      % (ifuslot,self.ifuslot))
            return True #this was still a fiber, just not one that is valid

        self.fibers.append(elixer_fiber.Fiber(idstring,specid,ifuslot,ifuid,amp,dither_date,dither_time,dither_time_extended,
                                 fiber_idx,self.id))

        return True

    def get_dither_number_for_fibers(self):
        #put fiber names in order (alphabetical also == time order)
        #assign dithers by that?
        pass


class FitsSorter:
#just a container for organization
    def __init__(self,fits=None,dist=0.0,loc=-1,side=None,dither=None,sn=None,fiber=None):
        self.fits = fits
        self.dist = dist
        self.loc = loc
        self.side = side
        self.dither = dither
        self.fiber_sn = sn
        self.fiber = fiber


class HETDEX:

    def __init__(self,args,fcsdir_list=None,hdf5_detectid_list=[],basic_only=False,cluster_list=None):
        #fcsdir may be a single dir or a list
        if args is None:
            log.error("Cannot construct HETDEX object. No arguments provided.")
            return None

        if args.annulus is not None:
            self.annulus = args.annulus
        else:
            self.annulus = None

        if args.wavelength is not None:
            self.target_wavelength = args.wavelength
        else:
            self.target_wavelength = None

        # if args.score:
        #     self.plot_dqs_fit = True
        # else:
        #     self.plot_dqs_fit = False

        self.output_filename = args.name
        self.dispatch_id = None
        if args.dispatch is not None:
            try:
                self.dispatch_id = int(args.dispatch.split("_")[1])
            except:
                pass

        if args.recover:
            self.recover = True
        else:
            self.recover = False

        #if a manual name or detectID is set from a --coords file:
        try:
            self.manual_name = int(args.manual_name) #args.manual_name might not exist, if so, no manual_name is set
            if self.manual_name <= 0:
                self.manual_name = None
        except:
            self.manual_name = None

        self.cluster_list = cluster_list
        self.multiple_observations = False #set if multiple observations are used (rather than a single obsdate,obsid)
        self.ymd = None
        self.target_ra = args.ra
        self.target_dec = args.dec
        self.target_err = args.error
        if args.ffsky:
            self.extraction_ffsky = args.ffsky
        else:
            self.extraction_ffsky = False

        try:
            self.known_z = args.known_z
        except:
            self.known_z = None

        try:
            self.explicit_extraction = args.explicit_extraction
        except:
            if args.aperture and args.ra and args.dec:
                self.explicit_extraction = True
            else:
                self.explicit_extraction = False

        if args.ra is not None:
            self.tel_ra = args.ra
        else:
            self.tel_ra = None

        if args.dec is not None:
            self.tel_dec = args.dec
        else:
            self.tel_dec = None

        if args.par is not None:
            self.parangle = args.par
        else:
            self.parangle = None

        if args.ifuslot is not None:
            self.ifu_slot_id = str(args.ifuslot).zfill(3)
        else:
            self.ifu_slot_id = None
        self.ifu_id = None
        self.specid = None
        self.obsid = None
        self.ifu_corner_ra = None
        self.ifu_corner_dec = None
        self.ifu_theta = None

        self.dither_fn = args.dither
        self.detectline_fn = args.line
        self.sigma = args.sigma
        self.chi2 = args.chi2
        if args.id is not None:
            self.emis_det_id = args.id.split(",") #might be a list?
        else:
            self.emis_det_id = None
        self.dither = None #Dither() obj
        self.fplane_fn = None
        self.fplane = None


        self.rot = None #calculated from PARANGLE in FITS header
        self.ifux = None #offset (in arcsecs) to the (0",0") position of IFU (read from fplane file)
        self.ifuy = None #offset (in arcsecs) to the (0",0") position of IFU (read from fplane file)
        self.tangentplane = None

        #not sure will need these ... right now looking at only one IFU
        self.ifuslot_dict = None
        self.cam_ifu_dict = None
        self.cam_ifuslot_dict = None

        self.ifu_ctr = None
        self.dist = {}

        self.emis_list = [] #list of qualified emission line detections

        self.sci_fits_path = args.path
        self.sci_fits = []
        self.status = 0

        self.plot_fibers = args.fibers
        self.min_fiber_sn = args.sn

        self.fcs_base = None
        self.fcsdir = None
        self.fcsdir_list = []
        #if we were not provided with a specific fcsdir, use the root level dir (if provided) in the args list
        #and prepend what is expected to be the appropriate subdir root (based on the naming of the output dir)
        if (fcsdir_list == None) or (len(fcsdir_list) == 0):
            if args.fcsdir is not None:
                self.fcsdir = args.fcsdir
                self.fcs_base = op.join(args.fcsdir,self.output_filename+"_")
        else: #we were given a specific directory, so use that
            self.fcsdir_list = fcsdir_list


        #HDF5 Stuff (might be empty)
        self.hdf5_detectid_list = None
        self.hdf5_detect_fqfn = None  # string ... the fully qualified filename
        self.hdf5_detect = None  # the actual HDF5 representation loaded
        self.hdf5_survey_fqfn = None

        if (hdf5_detectid_list is not None) and (len(hdf5_detectid_list) > 0):
            self.hdf5_detectid_list = hdf5_detectid_list
            self.hdf5_detect_fqfn = args.hdf5  #string ... the fully qualified filename
            self.hdf5_detect = None #the actual HDF5 representation loaded
            #for now
            self.hdf5_survey_fqfn = G.HDF5_SURVEY_FN

        if args.cure:
            self.panacea = False
        else:
            self.panacea = True

        if args.ylim is not None:
            self.ylim = args.ylim  # for full 1D plot
        else:
            self.ylim = None
        self.emission_lines = elixer_spectrum.Spectrum().emission_lines

        self.galaxy_mask = galaxy_mask.GalaxyMask()

        #parse the dither file
        #use to build fits list
        if self.dither_fn is not None:
            self.dither = Dither(self.dither_fn)
        elif (args.cure):
            #are we done? must have a dither file?
            log.error("Cannot construct HETDEX object. No dither file provided.")
            return None

        # read the detect line file if specified. Build a list of targets based on sigma and chi2 cuts
        build_fits_list = True
        if self.explicit_extraction: #needs to come first for logic to work
            self.hdf5_detect_fqfn = args.hdf5  #string ... the fully qualified filename
            self.hdf5_detect = None #the actual HDF5 representation loaded
            self.hdf5_survey_fqfn = G.HDF5_SURVEY_FN
            self.make_extraction(args.aperture,args.ffsky,args.shotid,basic_only=basic_only)
            build_fits_list = False
        elif (args.obsdate is None) and (self.detectline_fn is not None):  # this is optional
            self.read_detectline(force=True)
        elif (self.fcsdir is not None) or (len(self.fcsdir_list) > 0):
            #we have either fcsdir and fcs_base or fcsdir_list
            #consume the rsp1 style directory(ies)
            #DetObj(s) will be built here (as they are built, downstream from self.read_detectline() above
            self.read_fcsdirs()
            build_fits_list = False
        elif (self.hdf5_detectid_list is not None):  #Usual case option (but needs to come last for logic to work)
            self.read_hdf5_detect(basic_only=basic_only)
            build_fits_list = False

        if build_fits_list and not self.explicit_extraction:
            if (args.obsdate is None):
                if self.build_multi_observation_panacea_fits_list():
                    self.multiple_observations = True
                else:
                    self.status = -1
                    return
            else:
                #open and read the fits files specified in the dither file
                #need the exposure date, IFUSLOTID, SPECID, etc from the FITS files
                if not self.build_fits_list(args):
                    #fatal problem
                    self.status = -1
                    return

        if not self.panacea:
            #get ifu centers
            self.get_ifu_centers(args)

            #get distortion info
            self.get_distortion(args)

        #build fplane (find the correct file from the exposure date collected above)
        #for possible future use (could specify fplane_fn on commandline)
        if (self.fplane_fn is None) and (args.obsdate is not None):
            self.fplane_fn = find_fplane(self.ymd)

        if self.fplane_fn is not None:
            self.fplane = FPlane(self.fplane_fn)
            self.ifuslot_dict, self.cam_ifu_dict, self.cam_ifuslot_dict = build_fplane_dicts(self.fplane_fn)


        #read the detect line file if specified. Build a list of targets based on sigma and chi2 cuts
        #older style with obsdate and obsid specified on command line and the detect line file applied to a single
        #observation
        if (self.detectline_fn is not None) and (len(self.emis_list) == 0): #this is optional
            self.read_detectline(force=False)

        #assign dither indices to fibers for each emission object
        if build_fits_list:
            if self.dither:
                for e in self.emis_list:
                    for f in e.fibers:
                        f.dither_idx = self.dither.get_dither_index(f.dither_date,f.dither_time,f.dither_time_extended)
                        #get centers
                        for s in self.sci_fits:
                            #dither index should not matter, but if these are combined across much time, it is possible
                            #that the centers could have changed
                            if (s.dither_index == f.dither_idx) and (s.amp == f.amp):
                                #f.center_x = s.fiber_centers[f.number - 1,0]
                                #f.center_y = s.fiber_centers[f.number - 1,1]
                                f.center_x = s.fiber_centers[f.panacea_idx, 0]
                                f.center_y = s.fiber_centers[f.panacea_idx, 1]
                                break



        if args.annulus is None:
            #calculate the RA and DEC of each emission line object
            #remember, we are only using a single IFU per call, so all emissions belong to the same IFU

            #if ra and dec were passed in, use them instead of tel_ra and tel_dec

            #note: rot = 360-(90 + 1.8 + PARANGLE) so, PARANGLE = 360 -(90+1.8+rot)
            #the 1.8 constant is under some investigation (have also seen 1.3)

            #if PARANGLE is specified on the command line, use it instead of the FITS PARANGLE
            #360. - (90+1.3+args.rot)) from DetectWebpage
            build_coords = False
            if args.rot is not None:
                self.rot = float(args.rot)
            elif args.par is not None:
                self.rot = 360. - (90. + 1.3 + args.par)
            elif self.parangle:
                self.rot = 360. - (90. + 1.3 + self.parangle)

            if (args.ra is not None) and (args.dec is not None) and (self.rot is not None):
                self.tangentplane = TP(args.ra, args.dec, self.rot)
                build_coords = True
                log.debug("Calculating object RA, DEC from commandline RA=%f , DEC=%f , ROT=%f" \
                          % (args.ra, args.dec, self.rot))
            elif (self.tel_ra and self.tel_dec and self.rot):
                self.tangentplane = TP(self.tel_ra, self.tel_dec, self.rot)
                build_coords = True
                log.debug("Calculating object RA, DEC from: TELRA=%f , TELDEC=%f , PARANGLE=%f , ROT=%f" \
                      % (self.tel_ra, self.tel_dec, self.parangle, self.rot))

            if build_coords:
                #wants the slot id as a 0 padded string ie. '073' instead of the int (73)
                #ifu center
                self.ifux = self.fplane.by_ifuslot(self.ifu_slot_id).x
                self.ifuy = self.fplane.by_ifuslot(self.ifu_slot_id).y

                #reminder, we use the weighted ra and dec (e.wra, e.wdec) if available
                for e in self.emis_list: #yes this right: x + ifuy, y + ifux
                    e.ra, e.dec = self.tangentplane.xy2raDec(e.x + self.ifuy, e.y + self.ifux)
                    log.info("Emission Detect ID #%d RA=%f , Dec=%f" % (e.id,e.ra,e.dec))

    #end HETDEX::__init__()

    def unc_str(self, tuple): #helper, formats a string with exponents and uncertainty
        s = ""
        if len(tuple) == 2:
            tuple = (tuple[0],tuple[1],tuple[1])
        try:
            if np.isnan(tuple[0]) or tuple[0] is None:
                return 'nan'

            flux = ("%0.2g" % tuple[0]).split('e')
            unc = ("%0.2g" % (0.5 * (abs(tuple[1]) + abs(tuple[2])))).split('e')

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
            log.warning("Exception in HETDEX::unc_str()", exc_info=True)

        return s

    def rotation_matrix(self, theta=0.0, deg=True):
        # Returns a rotation matrix for CCW rotation
        # if deg is False, theta is in radians
        if deg:
            rad = theta * np.pi / 180.0
        else:
            rad = theta
        s = np.sin(rad)
        c = np.cos(rad)
        return np.array([[c, -s], [s, c]])

    def build_ifu_astrometry(self):

        if self.ifu_corner_ra is not None:
            return #already built

        #want the position of the lower left corner and the rotation relative to celestrial north

        #todo: the rotation should be from the parangle (assuming the ifus are aligned with the center)

        #cure:
        #xfiber = self.ifu_ctr.xifu[side][loc] + self.dither.dx[dither]
        #yfiber = self.ifu_ctr.yifu[side][loc] + self.dither.dy[dither]

        #panacea
        #xfiber = fits.fiber_centers[loc][0] + self.dither.dx[dither]
        #yfiber = fits.fiber_centers[loc][1] + self.dither.dy[dither]
        #self.sci_fits

        if self.panacea:
            #find the right fits for dither 0 LU (to get fiber 1 and fiber 19)
            # find the right fits for dither 0 RU (to get fiber 430)
            #fibers seem ordered from 0 to 111 (backward) fiber 1 = index 111
            lu0 = None #want fibers at indices 0 and 18
            ru0 = None #want fiber 430 which is index 112-94 (94 is 430-337+1 the relative fiber #)
            for fits in self.sci_fits:
                if fits.dither_index == 0:
                    if fits.amp == 'LU':
                        lu0 = fits
                        continue
                    elif fits.amp == 'RU':
                        ru0 = fits
                        continue

            #I think we have the x,y coords of the corner fibers as they appear on sky?

            #self.tangentplane = TP(args.ra, args.dec, self.rot)
            #todo: test remove
            #self.tangentplane = TP(0.355419, 20.170376, 116.575)
            self.tangentplane = TP(self.tel_ra, self.tel_dec, self.rot)
            #self.tangentplane = TP(self.tel_ra, 0.0, self.rot)
            #self.tangentplane = TP(0.0  , self.tel_dec,self.rot)
            self.ifux = self.fplane.by_ifuslot(self.ifu_slot_id).x
            self.ifuy = self.fplane.by_ifuslot(self.ifu_slot_id).y

          #  self.ifux = 250.7
          #  self.ifuy = 150.3

            #test
            #ifux = self.ifux
            #ifuy = self.ifuy

            #self.ifux  = 0
            #self.ifuy = 0

            #fiber 1
            corner1_x = self.ifuy + lu0.fiber_centers[112-1][0] #+ 0 for dither
            corner1_y = self.ifux + lu0.fiber_centers[112-1][1] #+ 0 for dither

            #corner1_x, corner1_y = np.dot(self.rotation_matrix(self.rot, deg=True),
            #                                np.array([corner1_x, corner1_y]).transpose())


            self.ifu_corner_ra, self.ifu_corner_dec = self.tangentplane.xy2raDec(corner1_x, corner1_y)

            #self.ifu_corner_ra, self.ifu_corner_dec = np.dot(self.rotation_matrix(self.rot, deg=True),
            #                                        np.array([self.ifu_corner_ra, self.ifu_corner_dec]).transpose())

            #fiber 19
            corner19_x = self.ifuy + lu0.fiber_centers[112-19][0]  # + 0 for dither
            corner19_y = self.ifux + lu0.fiber_centers[112-19][1]  # + 0 for dither
            #corner19_x, corner19_y = np.dot(self.rotation_matrix(self.rot-90.0, deg=True),
            #                         np.array([corner19_x, corner19_y]).transpose())
            bot_ra, bot_dec = self.tangentplane.xy2raDec(corner19_x, corner19_y)


            #fiber 430
            corner430_x = self.ifuy + ru0.fiber_centers[112-94][0]  # + 0 for dither
            corner430_y = self.ifux + ru0.fiber_centers[112-94][1]  # + 0 for dither
            #corner430_x, corner430_y = np.dot(self.rotation_matrix(self.rot, deg=True),
            #                                np.array([corner430_x, corner430_y]).transpose())
            top_ra, top_dec = self.tangentplane.xy2raDec(corner430_x, corner430_y)


            #delta y / delta x  or delta dec / delta ra
            self.ifu_theta = np.arctan2( (top_dec - self.ifu_corner_dec), (top_ra - self.ifu_corner_ra))
            double_check   = np.arctan2( (bot_dec - self.ifu_corner_dec), (bot_ra - self.ifu_corner_ra))

            sanity_check_1 = np.sqrt( (top_dec - self.ifu_corner_dec)**2 + (top_ra - self.ifu_corner_ra)**2 )*3600
            sanity_check_2 = np.sqrt( (bot_dec - self.ifu_corner_dec)**2 + (bot_ra - self.ifu_corner_ra)**2)*3600

            sanity_check_3 = np.sqrt((corner430_x - corner1_x) ** 2 + (corner430_y - corner1_y) ** 2)
            sanity_check_4 = np.sqrt((corner19_x - corner1_x) ** 2 + (corner19_y - corner1_y) ** 2)

            top_angle_xy = np.arctan2((corner430_y - corner1_y),(corner430_x - corner1_x))*180/np.pi
            bot_angle_xy = np.arctan2((corner19_y - corner1_y),(corner19_x - corner1_x))*180/np.pi

            top_angle_rd = np.arctan2((top_dec - self.ifu_corner_dec), (top_ra - self.ifu_corner_ra)) * 180 / np.pi
            bot_angle_rd = np.arctan2((bot_dec - self.ifu_corner_dec), (bot_ra - self.ifu_corner_ra)) * 180 / np.pi

            #todo: the above two should match (at least very closely) but are waaaaay off

            #self.ifux = ifux
            #self.ifuy = ifuy

        else: #this is cure

        #ifu_ctr 0,0 is bottom and left
            corner1_x = self.ifuy + self.ifu_ctr.xifu['L'][0] #+ self.dither.dx[dither]
            corner1_y = self.ifux + self.ifu_ctr.yifu['L'][0] #+ self.dither.dy[dither]
            self.ifu_corner_ra, self.ifu_corner_dec = self.tangentplane.xy2raDec(corner1_x, corner1_y)


            corner19_x = self.ifuy + self.ifu_ctr.xifu['L'][18] #+ self.dither.dx[dither]
            corner19_y = self.ifux + self.ifu_ctr.yifu['L'][18] #+ self.dither.dy[dither]
            bot_ra, bot_dec = self.tangentplane.xy2raDec(corner19_x, corner19_y)

            #430-1 - 224
            corner430_x = self.ifuy + self.ifu_ctr.xifu['R'][205] #+ self.dither.dx[dither]
            corner430_y = self.ifux + self.ifu_ctr.yifu['R'][205] #+ self.dither.dy[dither]
            top_ra, top_dec = self.tangentplane.xy2raDec(corner430_x, corner430_y)

            # delta y / delta x  or delta dec / delta ra
            self.ifu_theta = np.arctan2((top_dec - self.ifu_corner_dec), (top_ra - self.ifu_corner_ra))
            double_check = np.arctan2((bot_dec - self.ifu_corner_dec), (bot_ra - self.ifu_corner_ra))

            sanity_check_1 = np.sqrt((top_dec - self.ifu_corner_dec) ** 2 + (top_ra - self.ifu_corner_ra) ** 2) * 3600
            sanity_check_2 = np.sqrt((bot_dec - self.ifu_corner_dec) ** 2 + (bot_ra - self.ifu_corner_ra) ** 2) * 3600

            sanity_check_3 = np.sqrt((corner430_x - corner1_x) ** 2 + (corner430_y - corner1_y) ** 2)
            sanity_check_4 = np.sqrt((corner19_x - corner1_x) ** 2 + (corner19_y - corner1_y) ** 2)

            top_angle_xy = np.arctan2((corner430_y - corner1_y), (corner430_x - corner1_x)) * 180 / np.pi
            bot_angle_xy = np.arctan2((corner19_y - corner1_y), (corner19_x - corner1_x)) * 180 / np.pi

            top_angle_rd = np.arctan2((top_dec - self.ifu_corner_dec), (top_ra - self.ifu_corner_ra)) * 180 / np.pi
            bot_angle_rd = np.arctan2((bot_dec - self.ifu_corner_dec), (bot_ra - self.ifu_corner_ra)) * 180 / np.pi


        log.info("IFU (slot ID %s) lower left corner (dither 1) RA=%f , Dec=%f , Rot=%f" %
                 (self.ifu_slot_id, self.ifu_corner_ra, self.ifu_corner_dec, self.ifu_theta))

        return


    def get_ifu_centers(self,args):
        # if using the panacea variant, don't do this ... read the centers from the panacea fits file
        #   this is already done ... stored in HETDEXFits self.fiber_centers (ie. for each multi_*.fits file)
        #   see self.fiber_centers
        if not self.panacea:
            if args.ifu is not None:
                try:
                    self.ifu_ctr = IFUCenter(args.ifu)
                except:
                    log.error("Unable to open IFUcen file: %s" % (args.ifu), exc_info=True)
            else:
                ifu_fn = op.join(IFUCEN_LOC, "IFUcen_VIFU" + self.ifu_slot_id + ".txt")
                log.info("No IFUcen file provided. Look for CAM specific file %s" % (ifu_fn))
                try:
                    self.ifu_ctr = IFUCenter(ifu_fn)
                except:
                    ifu_fn = op.join(IFUCEN_LOC, "IFUcen_HETDEX.txt")
                    log.info("Unable to open CAM Specific IFUcen file. Look for generic IFUcen file.")
                    try:
                        self.ifu_ctr = IFUCenter(ifu_fn)
                    except:
                        log.error("Unable to open IFUcen file.", exc_info=True)

            if self.ifu_ctr is not None:
                #need this to be numpy array later
                for s in SIDE:
                    self.ifu_ctr.xifu[s] = np.array(self.ifu_ctr.xifu[s])
                    self.ifu_ctr.yifu[s] = np.array(self.ifu_ctr.yifu[s])


    def get_distortion(self,args):
        if not self.panacea:
            if args.dist is not None:
                try:
                    self.dist['L'] = Distortion(args.dist + '_L.dist')
                    self.dist['R'] = Distortion(args.dist + '_R.dist')
                except:
                    log.error("Unable to open Distortion files: %s" % (args.dist), exc_info=True)
            else:
                dist_base = op.join(DIST_LOC, "mastertrace_twi_" + self.specid)
                log.info("No distortion file base provided. Look for CAM specific file %s" % (dist_base))
                try:
                    self.dist['L'] = Distortion(dist_base + '_L.dist')
                    self.dist['R'] = Distortion(dist_base + '_R.dist')
                except:
                    ifu_fn = op.join(IFUCEN_LOC, "IFUcen_HETDEX.txt")
                    log.info("Unable to open CAM Specific twi dist files. Look for generic dist files.")
                    dist_base = op.join(DIST_LOC, "mastertrace_" + self.specid)
                    try:
                        self.dist['L'] = Distortion(dist_base + '_L.dist')
                        self.dist['R'] = Distortion(dist_base + '_R.dist')
                    except:
                        log.error("Unable to open distortion files.", exc_info=True)


    def build_fits_list(self,args=None):
        #read in all fits
        #get the key fits header values

        #only one dither object, but has array of (3) for each value
        #these are in "dither" order
        #read first cure-style fits to get header info needed to find the multi_*.fits files

        if (not self.dither) or (len(self.dither.basename) < 1): #we just got dx,dy from all.mch, need other info to continue
            exit_string = None
            result = False
            if args is None:
                exit_string = "Insufficient information provided. Limited dither info. Fatal."
            elif not self.panacea:
                exit_string = "Insufficient dither information provided for Cure processing. Fatal."
            elif (args.obsdate is None) or (args.obsid is None):
                exit_string = "Insufficient dither information provided. Missing obsdate or obsid."
            elif (args.specid is None) and (args.ifuid is None) and (args.ifuslot is None):
                exit_string = "Insufficient dither information provided. Must supply at least one of: " \
                              "specid, ifuid, ifuslot."
            else:
                #check args.obsdate
                result = True
                if len(args.obsdate) != 8:
                    exit_string = "Insufficient information provided. Invalid obsdate. Fatal."
                    result = False
                else:
                    try:
                        f = int(args.obsdate)
                    except:
                        exit_string = "Insufficient information provided. Invalid obsdate. Fatal."
                        result = False

            if not result:
                print (exit_string)
                log.error(exit_string)
                return False

            #build up path and filename from args info
            self.ymd = args.obsdate
            self.obsid = args.obsid
            wildcard = False
            if args.ifuid is not None:
                self.ifu_id = str(args.ifuid).zfill(3)
            else:
                self.ifu_id = '???'
                wildcard = True
            if args.ifuslot is not None:
                self.ifu_slot_id = str(args.ifuslot).zfill(3)
            else:
                self.ifu_slot_id = '???'
                wildcard = True
            if args.specid is not None:
                self.specid = str(args.specid).zfill(3)
            else:
                self.specid = '???'
                wildcard = True

            # leaves off the  LL.fits etc
            multi_fits_basename = "multi_" + self.specid + "_" + self.ifu_slot_id + "_" + self.ifu_id + "_"
            # leaves off the exp01/virus/
            multi_fits_basepath = op.join(G.PANACEA_RED_BASEDIR, self.ymd, "virus",
                                          "virus" + str(self.obsid).zfill(7))

            # see if path is good and read in the panacea fits
            dit_idx = 0
            path = op.join(multi_fits_basepath, "exp" + str(dit_idx + 1).zfill(2), "virus")

            if wildcard:
                if op.isdir(path):
                    fn = op.join(path, multi_fits_basename + "LU.fits")
                    files = glob.glob(fn)
                    #there should be exactly one match
                    if len(files) != 1:
                        exit_string = "Insufficient information provided. Unable to identify panacea science files. Fatal."
                        print(exit_string)
                        log.error(exit_string)
                        return False
                    else:
                        try:
                            toks = op.basename(files[0]).split('_')
                            self.specid = toks[1]
                            self.ifu_slot_id = toks[2]
                            self.ifu_id = toks[3]
                            multi_fits_basename = "multi_" + self.specid + "_" + self.ifu_slot_id + "_" \
                                                  + self.ifu_id + "_"
                        except:
                            exit_string = "Insufficient information provided. Unable to construct panacea science " \
                                          "file names. Fatal."
                            print(exit_string)
                            log.error(exit_string)
                            return False
                else: #invalid path:
                    print("Invalid path to panacea science fits: %s" %path)
                    log.error("Invalid path to panacea science fits: %s" %path)

            while op.isdir(path):
                for a in AMP:
                    fn = op.join(path, multi_fits_basename + a + ".fits")
                    self.sci_fits.append(hetdex_fits.HetdexFits(fn, None, None, dit_idx, panacea=True))
                    self.dither.basename.append(multi_fits_basename)

                    #todo: sanity check : make sure obsid and obsdate from command line match those in the fits
                  #  fits = self.sci_fits[-1]
                  #  if fits.obs_ymd != args.obsdate

                # next exposure
                dit_idx += 1
                path = op.join(multi_fits_basepath, "exp" + str(dit_idx + 1).zfill(2), "virus")

        elif self.panacea:
            dit_idx = 0
            for b in self.dither.basename:
                ext = ['_L.fits', '_R.fits']
                for e in ext:
                    fn = b + e
                    if (self.sci_fits_path is not None):
                        fn = op.join(self.sci_fits_path, op.basename(fn))
                    else:
                        if not op.exists(fn):
                            log.debug("Science files not found from cwd.\n%s" % (fn))
                            fn = op.join(self.dither.dither_path, fn)
                            log.debug("Trying again from dither path.\n%s" % (fn))

                    if not op.exists(fn):
                        log.error("Fatal. Cannot find science files from dither.")
                        print("Fatal. Cannot find science files from dither.")
                        return False

                    hdf = hetdex_fits.HetdexFits(fn, None, None, dit_idx)

                    if hdf.obs_ymd is not None:  # assume a good read on a fits file
                        self.ymd = hdf.obs_ymd
                        self.ifu_id = hdf.ifuid
                        self.ifu_slot_id = hdf.ifuslot
                        self.specid = hdf.specid
                        self.obsid = hdf.obsid
                        break

                        #okay, we got one, kill the loops
                if self.ymd is not None:
                    break
            #now build the path to the multi_*.fits and the file basename

            if self.specid is None:
                #clearly there was some problem
                log.error("Unable to build panacea file info from base science fits.")
                return False

            #leaves off the  LL.fits etc
            multi_fits_basename = "multi_" + self.specid + "_" + self.ifu_slot_id + "_" + self.ifu_id + "_"
            #leaves off the exp01/virus/
            multi_fits_basepath = op.join(G.PANACEA_RED_BASEDIR,self.ymd,"virus","virus"+str(self.obsid).zfill(7))

            #see if path is good and read in the panacea fits
            dit_idx = 0
            path = op.join(multi_fits_basepath, "exp" + str(dit_idx + 1).zfill(2), "virus")

            while op.isdir(path):
                for a in AMP:
                    fn  = op.join(path, multi_fits_basename + a + ".fits" )
                    self.sci_fits.append(hetdex_fits.HetdexFits(fn, None, None, dit_idx,panacea=True))

                #next exposure
                dit_idx += 1
                path = op.join(multi_fits_basepath, "exp" + str(dit_idx + 1).zfill(2), "virus")

        else: #cure style
            dit_idx = 0
            for b in self.dither.basename:
                ext = ['_L.fits','_R.fits']
                for e in ext:
                    fn = b + e
                    e_fn = op.join(op.dirname(b), "e." + op.basename(b)) + e
                    fe_fn = op.join(op.dirname(b), "Fe" + op.basename(b)) + e
                    if (self.sci_fits_path is not None):
                        fn = op.join(self.sci_fits_path, op.basename(fn))
                        e_fn = op.join(self.sci_fits_path, op.basename(e_fn))
                        fe_fn = op.join(self.sci_fits_path, op.basename(fe_fn))
                    else:
                        if not op.exists(fn):
                            log.debug("Science files not found from cwd.\n%s" % (fn))
                            fn = op.join(self.dither.dither_path, fn)
                            e_fn = op.join(self.dither.dither_path, e_fn)
                            fe_fn = op.join(self.dither.dither_path, fe_fn)
                            log.debug("Trying again from dither path.\n%s" % (fn))

                    if not op.exists(fn):
                        log.error("Fatal. Cannot find science files from dither.")
                        print("Fatal. Cannot find science files from dither.")
                        return False

                    self.sci_fits.append(hetdex_fits.HetdexFits(fn,e_fn,fe_fn,dit_idx))
                dit_idx += 1

        #all should have the same observation date in the headers so just use first
        if len(self.sci_fits) > 0:
            if not self.panacea: #these are already set in this case and the panacea fits does not contain it
                self.ymd = self.sci_fits[0].obs_ymd
                self.obsid = self.sci_fits[0].obsid

            self.ifu_id = self.sci_fits[0].ifuid
            self.ifu_slot_id = self.sci_fits[0].ifuslot
            self.specid = self.sci_fits[0].specid
            self.tel_ra = self.sci_fits[0].tel_ra
            self.tel_dec = self.sci_fits[0].tel_dec
            self.parangle = self.sci_fits[0].parangle


        if (self.tel_dec is None) or (self.tel_ra is None):
            log.error("Fatal. Cannot determine RA and DEC from FITS.", exc_info=True)
            return False
        return True


    def find_first_file(self,pattern, path):
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    return op.join(root, name)
        return None

    def build_multi_observation_panacea_fits_list(self):
        if len(self.emis_list) > 0:
            log.info("Building list of reduced fits files ...")

        dit_idx = 0
        for det in self.emis_list:
            log.debug("Searching for reduced fits file for detection ID %d " %(det.id))
            for fib in det.fibers:
                log.debug("Searching for fits files matching %s* " % (fib.idstring))
                #find the matching raw science file
                #from the path, get the observation date, obsid, expid
                #from those find the panacea file

                try:
                    path = op.join(G.PANACEA_RED_BASEDIR,fib.dither_date,"virus")
                except:
                    continue

                if not op.exists(path):
                    log.error("Cannot locate reduced data for %s" %(fib.idstring))
                    continue

                #we are at the top of the observation date ... have to search all subfolders for the idstring
                #like: '*20170326T111126.2*'

                #using assumption that there maybe multiples, but they belong to different IFUs and Amps
                #but all to the same observation date, obsid, and expid
                scifile = self.find_first_file("*"+fib.scifits_idstring+"*",path)
                if not scifile:
                    #try again with the defualt
                    default_path = op.join(G.PANACEA_RED_BASEDIR_DEFAULT, fib.dither_date, "virus")
                    if not op.exists(default_path):
                        log.error("Cannot locate reduced data for %s" % (fib.idstring))
                        continue
                    scifile = self.find_first_file("*" + fib.scifits_idstring + "*", default_path)
                #old method did not find it, try using the fits headers
                if scifile:
                    try:
                        obsid = scifile.split("virus/virus")[1].split("/")[0]
                        expid = scifile.split("/exp")[1].split("/")[0]

                        fib.expid = int(expid)
                        fib.obsid = int(obsid)

                        log.debug("Found reduction folder for file: " + scifile)
                    except:
                        log.error("Cannot locate reduction data for %s" % (fib.idstring))
                        continue
                else:
                    #only using the ifu package to get the path. elixer does not use entire IFU/observations
                    #just the specific fibers for each detection
                    info = elixer_ifu.find_panacea_path_info(fib.dither_date,
                                                              fib.dither_time_extended,
                                                              fib.dither_time,
                                                              G.PANACEA_RED_BASEDIR)

                    if (len(info) == 0) and (G.PANACEA_RED_BASEDIR != G.PANACEA_RED_BASEDIR_DEFAULT):
                        #try again with the default
                        info = elixer_ifu.find_panacea_path_info(fib.dither_date,
                                                                  fib.dither_time_extended,
                                                                  fib.dither_time,
                                                                  G.PANACEA_RED_BASEDIR_DEFAULT)
                    if len(info) > 0:
                        obsid = info['obsid']
                        expid = info['expid']

                        fib.expid = int(expid)
                        fib.obsid = int(obsid)

                        log.debug("Found reduction folder for file: " + info['path'])
                    else:
                        log.error("Cannot locate reduction data for %s" % (fib.idstring))
                        continue

                #now build the panace fits path
                path = op.join(G.PANACEA_RED_BASEDIR,fib.dither_date,"virus","virus"+obsid,"exp"+expid,"virus")

                if not op.exists(path):
                    log.error("Cannot locate panacea reduction data for %s" %(fib.idstring))
                    continue

                # now build the path to the multi_*.fits and the file basename
                # leaves off the  LL.fits etc
                multi_fits_basename = "multi_" + fib.specid + "_" + fib.ifuslot + "_" + fib.ifuid + "_"
                # leaves off the exp01/virus/
                multi_fits_basepath = op.join(G.PANACEA_RED_BASEDIR, fib.dither_date, "virus", "virus" + str(fib.obsid).zfill(7))

                # see if path is good and read in the panacea fits
                path = op.join(multi_fits_basepath, "exp" + str(fib.expid).zfill(2), "virus")
                if op.isdir(path):
                    fn = op.join(path, multi_fits_basename + fib.amp + ".fits")

                    if op.isfile(fn): # or op.islink(fn):
                        log.debug("Found reduced panacea file: " + fn)

                        fits = hetdex_fits.HetdexFits(fn, None, None, dit_idx, panacea=True)
                        fits.obs_date = fib.dither_date
                        fits.obs_ymd = fits.obs_date
                        fits.obsid = fib.obsid
                        fits.expid = fib.expid
                        fits.amp = fib.amp
                        fits.side = fib.amp[0]
                        fib.fits = fits
                        self.sci_fits.append(fits)
                    elif op.islink(fn):
                        log.error("Cannot open <" + fn + ">. Currently do not properly handle files as soft-links. "
                                                         "The path, however, can contain soft-linked directories.")
                    else:
                        log.error("Designated reduced panacea file does not exist: " + fn)
                else:
                    log.error("Cannot locate panacea reduction data for %s" % (path))
                    continue

        if len(self.sci_fits) > 0:
            return True
        else:
            return False



    def make_extraction(self,aperture,ffsky=False,shotid=None,basic_only=False):
        """

        :param basic_only:
        :return:
        """
        try:

            e = DetObj(None, emission=True, basic_only=basic_only)
            if e is not None:
                e.cluster_list = self.cluster_list
                e.galaxy_mask = self.galaxy_mask
                e.annulus = self.annulus
                e.target_wavelength = self.target_wavelength
                e.ra = self.target_ra
                e.dec = self.target_dec
                if self.known_z is not None:
                    e.known_z = self.known_z

                e.survey_shotid = shotid
                e.extraction_aperture = aperture
                e.extraction_ffsky = ffsky
                if (self.target_wavelength is not None) and (3400 < self.target_wavelength < 5600):
                    e.w = self.target_wavelength

                # just internal (ELiXer) numbering here
                G.UNIQUE_DET_ID_NUM += 1

                if self.hdf5_detectid_list is not None:
                    if len(self.hdf5_detectid_list)==1:
                        try:
                            d = np.int64(self.hdf5_detectid_list[0])
                            e.entry_id = d
                            e.id = d
                            #only get basic info to populate coords, etc
                            e.load_hdf5_fluxcalibrated_spectra(self.hdf5_detect_fqfn, d, basic_only=True)
                            e.ra = e.wra
                            e.dec = e.wdec
                            #keep the target wavelength, if provided
                            if (self.target_wavelength is not None) and (3460 < self.target_wavelength < 5540):
                                e.w = self.target_wavelength
                                e.target_wavelength = self.target_wavelength
                        except:
                            log.error(f"Skipping invalid detectid: {d}")
                            return None
                    elif len(self.hdf5_detectid_list)==0: #also expected, could just be an ra, dec
                        pass
                    else: #unexpdected
                        print(f"Unexpected # of detectids: {self.hdf5_detectid_list}")


                #a name was set on the --coords file AND the shotid was specified (so there will be 0 or 1 corresponding detection)
                #(if shotid is 0 or None, then we have to search all shots for RA, Dec and the name/ID may not be unique
                #so we default back to the UNIQUE_DET_ID_NUM
                if self.manual_name is not None:
                    if (e.survey_shotid is not None) and (e.survey_shotid != 0):
                        e.id = self.manual_name
                        e.entry_id = e.id
                    else:
                        log.warning(f"Manual ID/name {self.manual_name} specified for detection, but shotid/datevshot "
                                    f"not provided. Will default back to dispatch_id naming.")

                if e.id is None or e.survey_shotid is None:
                    if self.dispatch_id is not None:
                        e.id = np.int64(99e8 + self.dispatch_id * 1e4 + G.UNIQUE_DET_ID_NUM)
                        #so, like a hetdex detectid but starting with 99
                    elif e.entry_id is None:
                        e.id = G.UNIQUE_DET_ID_NUM

                    e.entry_id = e.id  # don't have an official one

                if e.outdir is None:
                    e.outdir = self.output_filename

                if self.recover: #this is a recover operation and we can now check for the .pdf
                    try:
                        if os.path.isfile(os.path.join(e.outdir, str(e.id) + ".pdf")) or \
                           os.path.isfile(os.path.join(e.outdir,e.outdir + "_" + str(e.id).zfill(3) + ".pdf")):

                            log.info(f"Already processed ({e.ra},{e.dec}) shot ({e.survey_shotid}). Will skip (recovery)." )
                            e.status = -1
                            return
                    except:
                        log.info("Exception checking reovery path. Will (re)build forced extraction report.")

                #a place to check for multiframe files if local and HDR locations fail
                #overloading use of --hdf5 invocation parameter (since this is a forced extraction
                # the detection file that would have been specified in --hdf5 is meaningless, so using
                # that paramater, if provided, as the possible path
                if self.hdf5_detect_fqfn is not None:
                    if op.isfile(self.hdf5_detect_fqfn):
                        e.hdf5_shot_dir = op.dirname(self.hdf5_detect_fqfn)
                    elif op.isdir(self.hdf5_detect_fqfn):
                        e.hdf5_shot_dir = self.hdf5_detect_fqfn

                if basic_only:
                    self.emis_list.append(e) #still need to append to list so the neighborhood report will generate
                    return

                e.forced_extraction()

                if e.survey_shotid and (e.status >= 0):

                    if e.w != self.target_wavelength:
                        log.info(f"Updating Central Wavelength (Target Wavelength) from {self.target_wavelength} to {e.w}")
                        self.target_wavelength = e.w
                    e.load_hdf5_shot_info(self.hdf5_survey_fqfn,  e.survey_shotid)

                if e.status >= 0:
                    e.check_transients_and_flags()

                    self.emis_list.append(e)# moved higher up to always be appended
                    if self.target_wavelength is None or self.target_wavelength == 0:
                        self.target_wavelength = e.target_wavelength

                else:
                    # regardless of the result, still append the DetObj so we will get the imaging cutouts
                    self.emis_list.append(e)
                    log.info("Unable to build full report for eid(%s). HETDEX section will not be generated." % (str(e.entry_id)))
        except:
            log.error("Exception in hetdex.py make_extraction.",exc_info=True)
            self.status = -1


    def read_hdf5_detect(self,basic_only=False):
        """
        Consume the HDF5 version of detections for the list of detections passed in
        (though there should only be one detectID at this point)
        :return:
        """

        #clear out any existing detections
        if len(self.emis_list) > 0:
            del self.emis_list[:]

        for d in self.hdf5_detectid_list:
            try:
                d = np.int64(d)
            except:
                log.error(f"Skipping invalid detectid: {d}")
                continue
            #build an empty Detect Object and then populate
            e = DetObj(None, emission=True,basic_only=basic_only)
            if e is not None:
                e.cluster_list = self.cluster_list
                e.galaxy_mask = self.galaxy_mask
                if self.known_z is not None:
                    e.known_z = self.known_z

                e.entry_id = d #aka detect_id from HDF5
                e.annulus = self.annulus
                e.target_wavelength = self.target_wavelength
                e.ra = self.target_ra
                e.dec = self.target_dec
                e.extraction_ffsky = self.extraction_ffsky

                #just internal (ELiXer) numbering here
                G.UNIQUE_DET_ID_NUM += 1
                e.id = G.UNIQUE_DET_ID_NUM

                if e.outdir is None:
                    e.outdir = self.output_filename

                #todo: load the HDF5 data here ...
                #e.load_fluxcalibrated_spectra()
                e.load_hdf5_fluxcalibrated_spectra(self.hdf5_detect_fqfn,d,basic_only=basic_only)
                #need the shotid from the detection
                if e.survey_shotid and (e.status >= 0):
                    e.load_hdf5_shot_info(self.hdf5_survey_fqfn, e.survey_shotid)

                if e.status >= 0:
                    e.check_transients_and_flags()
                    self.emis_list.append(e)
                else:
                    log.info("Unable to continue with eid(%s). No report will be generated." % (str(e.entry_id)))

    def read_fcsdirs(self):
        # we have either fcsdir and fcs_base or fcsdir_list
        # consume the rsp1 style directory
        # get fibers, get multi*fits files (may preclude the call to build_fits_list() just below)
        # note: at this point we are dealing with exactly one detection
        # base this on DetObj::load_fluxcalibrated_spectra()
        # DetObj(s) will be built here (as they are built, downstream from self.read_detectline() above
        if len(self.emis_list) > 0:
            del self.emis_list[:]

        if len(self.fcsdir_list) > 0: #we have a list of fcsdirs, one DetObj for each
            for d in self.fcsdir_list: #this is the typical case
                #build up the tokens that DetObj needs
                toks = None
                e = DetObj(toks, emission=True, fcsdir=d)

                if e is not None:
                    e.cluster_list = self.cluster_list
                    e.galaxy_mask = self.galaxy_mask
                    if self.known_z is not None:
                        e.known_z = self.known_z
                    e.annulus = self.annulus
                    e.target_wavelength = self.target_wavelength
                    e.ra = self.target_ra
                    e.dec = self.target_dec

                    G.UNIQUE_DET_ID_NUM += 1
                    #for consistency with Karl's namine, the entry_id is the _xxx number at the end
                    if e.entry_id is None or e.entry_id == 0:
                        e.entry_id = G.UNIQUE_DET_ID_NUM

                    e.id = G.UNIQUE_DET_ID_NUM
                    if e.outdir is None:
                        e.outdir = self.output_filename
                    e.load_fluxcalibrated_spectra()
                    if e.status >= 0:
                        e.check_transients_and_flags()
                        self.emis_list.append(e)
                    else:
                        log.info("Unable to continue with eid(%s). No report will be generated." %(str(e.entry_id)))
        elif (self.fcs_base is not None and self.fcsdir is not None): #not the usual case
            toks = None
            e = DetObj(toks, emission=True, fcs_base=self.fcs_base,fcsdir=self.fcsdir)

            if e is not None:
                e.cluster_list = self.cluster_list
                e.galaxy_mask = self.galaxy_mask
                if self.known_z is not None:
                    e.known_z = self.known_z
                e.annulus = self.annulus
                e.target_wavelength = self.target_wavelength
                e.ra = self.target_ra
                e.dec = self.target_dec

                G.UNIQUE_DET_ID_NUM += 1
                if e.entry_id is None or e.entry_id == 0:
                    e.entry_id = G.UNIQUE_DET_ID_NUM

                e.id = G.UNIQUE_DET_ID_NUM
                e.load_fluxcalibrated_spectra()
                if e.status >= 0:
                    e.check_transients_and_flags()
                    self.emis_list.append(e)

        #and if everything is None, nothing will work anyway

    def read_detectline(self,force=False):
        #emission line or continuum line

        #todo: rsp1 ... here, new option to build from rsp1 directories
        #todo:          an elif statement that calls to a new read_emisline replacement (read_rspdir?)
        #todo:              it will read (one or multiple subdirs) and build DetObj
        #todo:                  DetObj needs a new __init__ for this case (tokens are wrong for it)
        #todo: a single directory (detection) or multiple?
        #todo:      given a --fcsdir and no --line, look under for rsp1 output and run for each
        #todo:          so, if given a specific detection subdir, only that one woould be run

        #todo:      build this single or multiple as a call WITH the SUBDIR specified, so call
        #todo:          something else first to build a list of SUBDIRs to use (the idea is that a future
        #todo:          version will let the user specify which SUBDIRs (some subset) and then no downstream
        #todo:          changes are needed)

        #todo: something like: build_rspdirs_to_use() #returns a list of directories
        #todo:                 ingest_rspdirs (list of dirs) # fills out DetObj for each
        #todo:                      parse_rspdir(single dir,DetObj): populates DetObj and fibers, etc



        #todo: **** maybe rename fcsdir to rspdir? or in calls make above rspdir fcsdir for consistencey ***


        if '_cont.dat' in self.detectline_fn:
            self.read_contline() #for now, just applies to cure, which is not forced (ignore IFUSlot ID matching)
        else:
            self.read_emisline(force)


    def read_contline(self):
        # open and read file, line at a time. Build up list of continuum objects
        #this is a "cheat" use emis_list for emissions or continuum detections
        if len(self.emis_list) > 0:
            del self.emis_list[:]
        try:
            with open(self.detectline_fn, 'r') as f:
                f = ft.skip_comments(f)
                for l in f:
                    toks = l.split()
                    e = DetObj(toks,emission=False,fcs_base=self.fcs_base)
                    if e is None:
                        continue

                    e.cluster_list = self.cluster_list
                    e.galaxy_mask = self.galaxy_mask
                    if self.known_z is not None:
                        e.known_z = self.known_z

                    if e.ifuslot is not None:
                        if e.ifuslot != self.ifu_slot_id:
                            #this emission line does not belong to the IFU we are working on
                            #log.debug("Continuum detection IFU (%s) does not match current working IFU (%s)" %
                            #         (e.ifuslot,self.ifu_slot_id))
                            continue

                    if self.emis_det_id is not None:
                        if str(e.id) in self.emis_det_id:
                            if (self.ifu_slot_id is not None):
                                if (str(e.ifuslot) == str(self.ifu_slot_id)):
                                    e.check_transients_and_flags()

                                    self.emis_list.append(e)
                            else: #if is 'none' so they all go here ... must assume same IFU
                                e.check_transients_and_flags()
                                self.emis_list.append(e)
                    else:
                        if (self.ifu_slot_id is not None):
                            if (str(e.ifuslot) == str(self.ifu_slot_id)):
                                e.check_transients_and_flags()
                                self.emis_list.append(e)
                        else:
                            e.check_transients_and_flags()
                            self.emis_list.append(e)
        except:
            log.error("Cannot read continuum objects.", exc_info=True)

    def read_emisline(self,force=False):
        #open and read file, line at a time. Build up list of emission line objects

        if len(self.emis_list) > 0:
            del self.emis_list[:]
        try:
            with open(self.detectline_fn, 'r') as f:
                f = ft.skip_comments(f)
                line_counter = 0
                for l in f:
                    line_counter += 1
                    toks = l.split()
                    e = DetObj(toks,emission=True,line_number=line_counter,fcs_base=self.fcs_base)

                    if e is None:
                        continue

                    e.cluster_list = self.cluster_list
                    e.galaxy_mask = self.galaxy_mask
                    if self.known_z is not None:
                        e.known_z = self.known_z

                    #e.plot_dqs_fit = self.plot_dqs_fit

                    if self.panacea and (e.sn < self.min_fiber_sn): #pointless to add, nothing will plot
                        continue

                    if not force:
                        if e.ifuslot is not None:
                            if e.ifuslot != self.ifu_slot_id:
                                #this emission line does not belong to the IFU we are working on
                                #log.debug("Emission detection IFU (%s) does not match current working IFU (%s)" %
                                #         (e.ifuslot,self.ifu_slot_id))
                                continue

                    #only assign a detection to THIS HETDEX object if they are in the same IFU
                    #todo: what about combined (across multiple IFUs) (maybe define as ifu000)?
                    if self.emis_det_id is not None:
                        if str(e.id) in self.emis_det_id:
                            if (self.ifu_slot_id is not None):
                                if (str(e.ifuslot) == str(self.ifu_slot_id)):
                                    e.check_transients_and_flags()
                                    self.emis_list.append(e)
                            else: #if is 'none' so they all go here ... must assume same IFU
                                e.check_transients_and_flags()
                                self.emis_list.append(e)
                    else:
                        if (e.sigma >= self.sigma) and (e.chi2 <= self.chi2):
                            if (self.ifu_slot_id is not None):
                                if (str(e.ifuslot) == str(self.ifu_slot_id)):
                                    e.check_transients_and_flags()
                                    self.emis_list.append(e)
                            else:
                                e.check_transients_and_flags()
                                self.emis_list.append(e)
        except:
            log.error("Cannot read emission line objects.", exc_info=True)


        for e in self.emis_list:
            e.load_fluxcalibrated_spectra()

        return


    def get_sci_fits(self,dither,side,amp=None):
        for s in self.sci_fits:
            if ((s.dither_index == dither) and (s.side == side)):
                if (amp is not None) and (s.amp is not None):
                    if amp == s.amp:
                        return s
                else:
                    return s
        return None

    def get_emission_detect(self,detectid):
        for e in self.emis_list:
            if e.id == detectid:
                return e
        return None


    def build_hetdex_annulus_data_page(self,pages,detectid):
        e = self.get_emission_detect(detectid) #this is a DetObj
        if e is None:
            log.error("Could not identify correct emission to plot. Detect ID = %d" % detectid)
            return None

        if e.status < 0:
            log.info(f"Bad DetObj status ({e.status}). Will not build HETDEX report section.")
            return None

        print ("Bulding HETDEX annulus header for Detect ID #%d" %detectid)

        e.syn_obs.build_complete_emission_line_info_dict()
        if len(e.syn_obs.eli_dict) == 0:  # problem, we're done
            log.error("Problem building complete EmissionLineInfo dictionary.")
            return None

        radius_step = G.Fiber_Radius * 2.0
        radii_vector = np.arange(e.annulus[0]+radius_step,e.annulus[1]+radius_step,radius_step)
        eli_vector = [None] * len(radii_vector)
        snr_vector = [0] * len(radii_vector)
        best_eli = None
        best_radius = 0
        best_fiber_count = 0

        #loop here, incrementing the radius and rebuilding annulus_fibers
        print("LOOPING ....")

        for i in range(len(radii_vector)):
            outer_radius = radii_vector[i]
            inner_radius = e.annulus[0] #todo: option to also shift the inner_radius

            e.syn_obs.annulus_fibers(inner_radius=inner_radius,outer_radius=outer_radius,
                                     #empty=True,
                                     empty=False,
                                     central_wavelength=self.target_wavelength)

            # now sub select the annulus fibers (without signal) (populate obs.fibers_work
            #e.syn_obs.annulus_fibers(empty=True)
            if len(e.syn_obs.fibers_work) > 0:
                e.syn_obs.sum_fibers()

                spec_obj = elixer_spectrum.Spectrum()#todo: needs spec_obj.identifier and spec_obj.plot_dir

                spec_obj.set_spectra(e.syn_obs.sum_wavelengths, e.syn_obs.sum_values,
                                       e.syn_obs.sum_errors, e.syn_obs.target_wavelength,
                                       values_units=e.syn_obs.units) #,fit_min_sigma=2.0)

                eli_vector[i] = spec_obj.central_eli #might be none

                #dummy:
                if eli_vector[i] is not None and eli_vector[i].snr is not None:

                    snr_vector[i] = eli_vector[i].snr

                    if best_eli is None or (best_eli.snr <= eli_vector[i].snr):
                        best_eli = eli_vector[i]
                        best_radius = outer_radius
                        best_fiber_count = len(e.syn_obs.fibers_work)

                    print("LOOP (%d): %0.1f-%0.1f [#fib %d] SNR = %0.2f (score = %0.2f)"
                          % (i,inner_radius,outer_radius,len(e.syn_obs.fibers_work), eli_vector[i].snr, eli_vector[i].line_score))
                    log.info("Annulus loop(%d): %0.1f-%0.1f [#fib %d] SNR = %0.2f (score = %0.2f)"
                             % (i,inner_radius, outer_radius, len(e.syn_obs.fibers_work), eli_vector[i].snr, eli_vector[i].line_score))
                else:
                    print("LOOP (%d): %0.1f-%0.1f [#fib %d] SNR = None"
                          % (i,inner_radius, outer_radius,len(e.syn_obs.fibers_work)))
                    log.info("Annulus loop(%d): %0.1f-%0.1f [#fib %d] SNR = None"
                             % (i, inner_radius, outer_radius, len(e.syn_obs.fibers_work)))

            else:
                log.warning("No fibers inside annulus.")

        # do I need to classify? Is there any point (maybe pass in a fixed redshift?)
        # maybe find the max line_score in the vector and just run classify on that one?

        if (best_eli is not None) and (best_eli.snr > 0):
            log.info("Rebuilding best score ...")
            e.syn_obs.best_radius = best_radius
            e.syn_obs.annulus_fibers(inner_radius=inner_radius, outer_radius=best_radius,empty=True,
                                     central_wavelength=self.target_wavelength)
            e.syn_obs.sum_fibers()

            e.spec_obj.set_spectra(e.syn_obs.sum_wavelengths, e.syn_obs.sum_values,
                                   e.syn_obs.sum_errors, e.syn_obs.target_wavelength,
                                   values_units=e.syn_obs.units)

            #don't think there is any point in calling classify?

            #e.spec_obj.classify()  # solutions can be returned, also stored in spec_obj.solutions

        figure_sz_y = G.GRID_SZ_Y * 2
        fig = plt.figure(figsize=(G.ANNULUS_FIGURE_SZ_X, figure_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        plt.gca().axis('off')

        #two columns info then plot (radius vs line_score)
        gs = gridspec.GridSpec(2,3)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)


        if e.pdf_name is not None:
            title = "\n%s\n" % (e.pdf_name)
        elif self.output_filename is not None:
            title = "\n%s_%s.pdf\n" % (self.output_filename, str(e.entry_id).zfill(3))
        else:
            title = ""

        title += "Obs: Synthetic (%0.2f to %0.2f\")\n" %(e.annulus[0],e.annulus[1])

        if (e.entry_id is not None) and (e.id is not None):
            title += "ID (%s), Entry# (%d)" % (str(e.entry_id), e.id)
            if e.line_number is not None:
                title += ", Line# (%d)" % (e.line_number)

        if (e.wra is not None) and (e.wdec is not None):  # weighted RA and Dec
            ra = e.wra
            dec = e.wdec
        else:
            ra = e.ra
            dec = e.dec

        try:
            title += "\nRA,Dec (%f,%f)\n$\lambda$ = %g$\AA$ \n" % (e.syn_obs.ra, e.syn_obs.dec, e.syn_obs.w)

            if e.spec_obj.estflux is not None:
                title += "LineFlux = %0.3g  \n" % e.spec_obj.estflux

            if e.spec_obj.eqw_obs is not None:
                title += "EW_obs = %g$\AA$ \n" % e.spec_obj.eqw_obs

            if (e.spec_obj.central_eli is not None) and (e.spec_obj.central_eli.snr is not None):
                title += "S/N = %0.3g at %0.1f\" with %d fibers \n" \
                         %(e.spec_obj.central_eli.snr,best_radius,best_fiber_count)

            title += "Area = %0.1f (%0.1f) sq.arcsec\n" %(np.pi * (best_radius**2 - self.annulus[0]**2),
                                                      len(e.syn_obs.fibers_work) * np.pi * G.Fiber_Radius ** 2)


        except:
            log.error("Exception setting title.",exc_info=True)


        plt.subplot(gs[0,0]) #all (again, probably won't end up using grid_spec for this case)
        plt.text(0, 1.0, title, ha='left', va='top', fontproperties=font)
        if not G.ZEROTH_ROW_HEADER:
            plt.suptitle(time.strftime("%Y-%m-%d %H:%M:%S") +
                     "  Version " + G.__version__ +"  ", fontsize=8,x=1.0,y=0.98,
                     horizontalalignment='right',verticalalignment='top')
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        #SNR vs Radius
        plt.subplot(gs[1,0])
        plt.subplots_adjust(left=0.05, right=0.95, top=.9, bottom=0.2)
        plt.bar(radii_vector,snr_vector,color='b')
        plt.ylabel("SNR")
        plt.xlabel("Radius [arcsec]")

        #pages.append(fig)
        #plt.close('all')


        #Gauss Plot
        try:
            if (e.spec_obj.central_eli is not None) and (e.spec_obj.central_eli.gauss_plot_buffer is not None):
                plt.subplot(gs[:, 1:])
                plt.gca().set_frame_on(False)
                plt.gca().axis('off')
                #plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

                e.spec_obj.central_eli.gauss_plot_buffer.seek(0)
                im = Image.open(e.spec_obj.central_eli.gauss_plot_buffer)
                plt.imshow(im, interpolation='none')
                #pages.append(fig)
                #plt.close('all')
        except:
            log.error("Exception adding gauss fit plot to report.",exc_info=True)

        pages.append(fig)
        plt.close('all')

        try: #full width spectrum plot
            fig = plt.figure(figsize=(G.ANNULUS_FIGURE_SZ_X, 2.0))
            #plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
            plt.gca().axis('off')

            buf = e.spec_obj.build_full_width_spectrum(wavelengths=e.syn_obs.sum_wavelengths,
                                                       counts=e.syn_obs.sum_values,
                                                       errors=e.syn_obs.sum_errors,
                                                       values_units = e.syn_obs.units,
                                                       central_wavelength=e.syn_obs.w,
                                                       show_skylines=True, show_peaks=True, name=None,
                                                       annotate=False,figure=fig,show_line_names=False)

            if buf is not None:
                buf.seek(0)
                im = Image.open(buf)
                plt.imshow(im, interpolation='none')  # needs to be 'none' else get blurring

                pages.append(fig) #append the second part to its own page to be merged later
            plt.close()
        except:
            log.warning("Failed to build full width spec/cutout image.", exc_info=True)

        return pages

    def build_hetdex_data_page(self,pages,detectid):

        e = self.get_emission_detect(detectid) #this is a DetObj
        if e is None:
            log.error("Could not identify correct emission to plot. Detect ID = %d" % detectid)
            return None

        if e.status < 0:
            log.info(f"Bad DetObj status ({e.status}). Will not build HETDEX report section.")
            return None

        print ("Bulding HETDEX header for Detect ID #%d" %detectid)

        #self.build_ifu_astrometry()

        if G.SINGLE_PAGE_PER_DETECT:
            figure_sz_y = G.GRID_SZ_Y
        else:
            figure_sz_y = 3 * G.GRID_SZ_Y

        fig = plt.figure(figsize=(G.FIGURE_SZ_X, figure_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        #plt.tight_layout()
        plt.gca().axis('off')
        # 2x2 grid (but for sizing, make more fine)

        #4 columns ... 3 wider, 1 narrower (for scattered light) .. so make 40 steps
        #text, 2D cutous, scattered light, 1D (small) plot
        #column groups are not the same size
        if G.SINGLE_PAGE_PER_DETECT:
            gs = gridspec.GridSpec(2, 100)
        else:
            if G.SHOW_ALL_1D_SPECTRA:
                gs = gridspec.GridSpec(5, 100)#, wspace=0.25, hspace=0.5)
            else:
                gs = gridspec.GridSpec(3, 100)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        sci_files = ""
        if self.dither:
            for s in self.dither.basename:
                if not( op.basename(s) in sci_files):
                    sci_files += "  " + op.basename(s) + "*.fits\n"
        else:
            sci_files = "multiple fiber specific"

        title = r""
        if e.pdf_name is not None:
            title = "\nID: %s (%s)\n" % (str(e.entry_id),e.pdf_name)
        elif self.output_filename is not None:
            title += "ID: %s (%s_%s.pdf)\n" % (str(e.entry_id),self.output_filename, str(e.entry_id).zfill(3))
        #else:
            #title += "" #todo: start with filename

        try:
            title += "Obs: " + e.fibers[0].dither_date + "v" + str(e.fibers[0].obsid).zfill(3) + "_" + \
            str(e.fibers[0].detect_id)
            if e.extraction_aperture is not None: #this was a forced extraction
                title += f" ({e.extraction_aperture:0.1f}\""
                if e.extraction_ffsky:
                    title += " ff)"
                else:
                    title += " lo)"
        except:
            log.debug("Exception building observation string.",exc_info=True)

        #really pointless ... the id is plainly available above
        # if (e.entry_id is not None) and (e.id is not None):
        #     title += "ID (%s), Entry# (%d)" %(str(e.entry_id), e.id)
        #     if e.line_number is not None:
        #         title += ", Line# (%d)" % (e.line_number)

        #if e.entry_id is not None:
        #    title += " (Line #%d)" % e.entry_id

        #actual line number from the input file
        #there are 3 numbers: the detect ID (from the detect file), the entity ID (from the composite file)
        # (those two are usually the same, unless CURE is used), and the line number from the input file
        # (e.g. the t7all or t5cut, etc)
        #if e.line_number is not None:
        #    title += " (Line #%d)" % e.line_number

        if (e.wra is not None) and (e.wdec is not None):  # weighted RA and Dec
            ra = e.wra
            dec = e.wdec
        else:
            ra = e.ra
            dec = e.dec

        datakeep = self.build_data_dict(e)
        if (datakeep is None) or ((datakeep['status'] < 0) and self.recover):
            e.status = -1
            return None

        e.get_probabilities()

        if e.w > 0:
            la_z = e.w / G.LyA_rest - 1.0
            oii_z = e.w / G.OII_rest - 1.0
        else:
            la_z = 1
            oii_z = 1

        estflux_str = "%0.3g" %(e.estflux)
        estcont_str = "%0.3g" %(e.cont_cgs)
        eqw_lya_str = "%0.3g" %(e.eqw_obs/(1.0+la_z))
        try:
            estcont_gmag_str = self.unc_str((e.best_gmag_cgs_cont, e.best_gmag_cgs_cont_unc))
            eqw_lya_gmag_str = "w: " + self.unc_str((e.best_eqw_gmag_obs/(1.0 + la_z),e.best_eqw_gmag_obs_unc/(1.0 + la_z)))
        except:
            estcont_gmag_str = 'nan'
            eqw_lya_gmag_str = 'nan'

        if G.REPORT_ELIXER_MCMC_FIT:
            try:
                if e.spec_obj.central_eli is not None:
                    estflux_str = e.spec_obj.central_eli.flux_unc
                    estcont_str = e.spec_obj.central_eli.cont_unc
                    #comment out for now, need to check error propogation and EW fit from MCMC ... instead use original version
                    eqw_lya_str = e.spec_obj.central_eli.eqw_lya_unc
                else:
                    log.warning("Warning. e.spec_obj.central_eli is None in HETDEX::build_hetdex_data_page()")
            except:
                log.error("Exception setting uncertainty strings",exc_info=True)

        elif e.line_gaussfit_parms is not None:
            if e.line_gaussfit_unc is not None:
                #the /2.0 to deal with Karl's 2AA bin width
                estflux_str = self.unc_str((e.line_gaussfit_parms[2]/e.line_gaussfit_parms[4] *G.HETDEX_FLUX_BASE_CGS,
                                            e.line_gaussfit_unc[2]/e.line_gaussfit_parms[4]*G.HETDEX_FLUX_BASE_CGS))
                #estcont_str = self.unc_str((e.line_gaussfit_parms[3]*1e-17,e.line_gaussfit_unc[3]*1e-17))

                if e.using_best_gmag_ew:
                    estcont_str = self.unc_str((e.cont_cgs, e.cont_cgs_unc))
                else:
                    estcont_str = self.unc_str((e.cont_cgs, e.line_gaussfit_unc[3] * G.HETDEX_FLUX_BASE_CGS))
                eqw_lya_str = self.unc_str((e.eqw_obs/(1.0 + la_z),e.eqw_obs_unc/(1.0 + la_z)))
            else:
                log.info("e.line_gaussfit_unc is None. Cannot report uncertainties in flux or EW.")
                # the /2.0 to deal with Karl's 2AA bin width
                estflux_str = "%0.3g" %(e.line_gaussfit_parms[2]/e.line_gaussfit_parms[4]*G.HETDEX_FLUX_BASE_CGS)
                #estcont_str = "%0.3g" %(e.line_gaussfit_parms[3]*1e-17)
                estcont_str = "%0.3g" % (e.cont_cgs)

        if e.using_best_gmag_ew: 
            #using as an indicator that the narrow cgs flux was not valid (probably negative)
            #and was replaced throughout with the wider estimate, but we will fall back and report the
            #original narrow estimate here
            estcont_str = self.unc_str((e.cont_cgs_narrow,e.cont_cgs_narrow_unc))
            #estcont_str = "N/A"

        if self.ymd and self.obsid:
            if not G.ZOO:
                title +="\n"\
                    "ObsDate %s  ObsID %s IFU %s  CAM %s\n" \
                    "Science file(s):\n%s" \
                    "RA,Dec (%f,%f) \n" \
                    "$\lambda$ = %g$\AA$  FWHM = %0.1f($\pm$%0.1f)$\AA$\n" \
                    "LineFlux = %s" \
                    % (self.ymd, self.obsid, self.ifu_slot_id,self.specid,sci_files, ra, dec, e.w,e.fwhm,e.fwhm_unc,
                       estflux_str )

                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux,e.fluxfrac)
                else:
                    title += "\n"

                title +=  "Cont(n) = %s" %(estcont_str)
                if estcont_gmag_str:
                    title += "\nCont(w) = %s" %(estcont_gmag_str)

                if e.best_gmag is not None:
                    if e.using_best_gmag_ew:
                        #title += " (gmag %0.2f *)\n" % (e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$ *)" % (e.best_gmag,
                                                                                e.best_gmag+e.best_gmag_unc,
                                                                                e.best_gmag-e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f *)" % (e.best_gmag)
                    else:
                        #title += " (gmag %0.2f)\n" %(e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$)" % (
                            e.best_gmag, e.best_gmag + e.best_gmag_unc, e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f)" %(e.best_gmag)
                    title += "\n"

                else:
                    title += "\n"

                if eqw_lya_gmag_str:
                    title += "EWr = %s (%s)$\AA$\n" % (eqw_lya_str,eqw_lya_gmag_str)
                else:
                    title += "EWr = %s$\AA$\n" %(eqw_lya_str)

            else:  #this if for zooniverse, don't show RA and DEC or Probabilitie
                title += "\n" \
                     "ObsDate %s  ObsID %s IFU %s  CAM %s\n" \
                     "Science file(s):\n%s" \
                     "$\lambda$ = %g$\AA$  FWHM = %0.1f($\pm$%0.1f)$\AA$\n" \
                     "LineFlux = %s" \
                             % (self.ymd, self.obsid, self.ifu_slot_id, self.specid, sci_files, e.w,e.fwhm,e.fwhm_unc,
                                estflux_str)  # note: e.fluxfrac gauranteed to be nonzero
                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux, e.fluxfrac)
                else:
                    title += "\n"

                title += "Cont(n) = %s" % (estcont_str)
                if estcont_gmag_str:
                    title += "\nCont(w) = %s" %(estcont_gmag_str)

                if e.best_gmag is not None:
                    if e.using_best_gmag_ew:
                        #title += " (gmag %0.2f *)\n" % (e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$ *)" % (e.best_gmag,
                                                                                e.best_gmag + e.best_gmag_unc,
                                                                                e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f *)" % (e.best_gmag)
                    else:
                        #title += " (gmag %0.2f)\n" %(e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$)" % (e.best_gmag,
                                                                                e.best_gmag + e.best_gmag_unc,
                                                                                e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f)" % (e.best_gmag)

                    title += "\n"
                else:
                    title += "\n"

                if eqw_lya_gmag_str:
                    title += "EWr = %s (%s)$\AA$\n" % (eqw_lya_str,eqw_lya_gmag_str)
                else:
                    title += "EWr = %s$\AA$\n" %(eqw_lya_str)

               # title += "EstCont = %s  \nEW_r(LyA) = %s$\AA$\n" % (estcont_str, eqw_lya_str)


        else:
            if not G.ZOO:
                title += "\nPrimary Spec_Slot_IFU_AMP: %s_%s_%s_%s\n" % (e.fibers[0].specid, e.fibers[0].ifuslot,
                                                                     e.fibers[0].ifuid,e.fibers[0].amp)

                if e.survey_fwhm > 3.0:
                    title += f"F=*{e.survey_fwhm:0.1f}\"*  "
                else:
                    title += f"F={e.survey_fwhm:0.1f}\"  "

                if e.survey_response < 0.08:
                    title +=f"T=*{e.survey_response:0.3f}!  "
                else:
                    title += f"T={e.survey_response:0.3f}  "

                if e.dither_norm > 3.0:
                    title += f"N=*{e.dither_norm:0.2f}!  "
                else:
                    title += f"N={e.dither_norm:0.2f}  "

                if (e.sumspec_apcor is not None) and (len(e.sumspec_apcor) == 1036):
                    title += f"A={e.sumspec_apcor[515]:0.2f}  "
                else:
                    title += f"A=---  "


                #title += f"A={e.amp_stats:0.2f}"
                title += "\n"

                title += "RA,Dec (%f,%f) \n" \
                     "$\lambda$ = %g$\AA$  FWHM = %0.1f($\pm$%0.1f)$\AA$\n" \
                     "LineFlux = %s" \
                     %(ra, dec, e.w,e.fwhm,e.fwhm_unc, estflux_str)

                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux,e.fluxfrac)
                else:
                    title += "\n"
                #title +=  "EstCont = %s  \nEW_r(LyA) = %s$\AA$\n" % (estcont_str, eqw_lya_str)

                title += "Cont(n) = %s" % (estcont_str)
                if estcont_gmag_str:
                    title += "\nCont(w) = %s" %(estcont_gmag_str)

                if e.best_gmag is not None:
                    if e.using_best_gmag_ew:
                        #title += " (gmag %0.2f *)\n" % (e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$ *)" % (e.best_gmag,
                                                                                e.best_gmag + e.best_gmag_unc,
                                                                                e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f *)" % (e.best_gmag)
                    else:
                        #title += " (gmag %0.2f)\n" %(e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$)" % (e.best_gmag,
                                                                                e.best_gmag + e.best_gmag_unc,
                                                                                e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f)" % (e.best_gmag)
                    title += "\n"
                else:
                    title += "\n"

                if eqw_lya_gmag_str:
                    title += "EWr = %s (%s)$\AA$\n" % (eqw_lya_str,eqw_lya_gmag_str)
                else:
                    title += "EWr = %s$\AA$\n" %(eqw_lya_str)


            else: #this if for zooniverse, don't show RA and DEC or probabilities

                #title += "\nPrimary IFU SpecID (%s) SlotID (%s)\n" % (e.fibers[0].specid, e.fibers[0].ifuslot)
                title += "\nPrimary Spec_Slot_IFU_AMP: %s_%s_%s_%s\n" % (e.fibers[0].specid, e.fibers[0].ifuslot,
                                                                         e.fibers[0].ifuid,e.fibers[0].amp)

                if e.survey_fwhm > 3.0:
                    title += f"F=*{e.survey_fwhm:0.1f}\"*  "
                else:
                    title += f"F={e.survey_fwhm:0.1f}\"  "

                if e.survey_response < 0.08:
                    title +=f"T=*{e.survey_response:0.3f}!  "
                else:
                    title += f"T={e.survey_response:0.3f}  "

                if e.dither_norm > 3.0:
                    title += f"N=*{e.dither_norm:0.2f}!  "
                else:
                    title += f"N={e.dither_norm:0.2f}  "

                if (e.sumspec_apcor is not None) and (len(e.sumspec_apcor) == 1036):
                    title += f"A={e.sumspec_apcor[515]:0.2f}  " #idx 515 is 4500AA
                else:
                    title += f"A=---  "

                # title += f"A={e.amp_stats:0.2f}"
                title += "\n"

                title += "$\lambda$ = %g$\AA$  FWHM = %0.1f($\pm$%0.1f)$\AA$\n" \
                         "LineFlux = %s" \
                         % (e.w, e.fwhm, e.fwhm_unc, estflux_str)

                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux,e.fluxfrac)
                else:
                    title += "\n"
                #title +=  "EstCont = %s  \nEW_r(LyA) = %s$\AA$\n" % (estcont_str, eqw_lya_str)
                title += "Cont(n) = %s" % (estcont_str)
                if estcont_gmag_str:
                    title += "\nCont(w) = %s" %(estcont_gmag_str)

                if e.best_gmag is not None:
                    if e.using_best_gmag_ew:
                        #title += " (gmag %0.2f *)\n" % (e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$ *)" % (e.best_gmag,
                                                                                e.best_gmag + e.best_gmag_unc,
                                                                                e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f *)" % (e.best_gmag)
                    else:
                        #title += " (gmag %0.2f)\n" %(e.best_gmag)
                        if e.best_gmag_unc:
                            title += r" (gmag $%0.2f\ ^{%0.2f}_{%0.2f}$)" % (e.best_gmag,
                                                                                e.best_gmag + e.best_gmag_unc,
                                                                                e.best_gmag - e.best_gmag_unc)
                        else:
                            title += " (gmag %0.2f)" % (e.best_gmag)
                    title += "\n"
                else:
                    title += "\n"

                #title += "EW_r(LyA) = %s$\AA$\n" % (eqw_lya_str)
                if eqw_lya_gmag_str:
                    title += "EWr = %s (%s)$\AA$\n" % (eqw_lya_str,eqw_lya_gmag_str)
                else:
                    title += "EWr = %s$\AA$\n" %(eqw_lya_str)

        if self.panacea:
            snr = e.sigma
            snr_unc = 0.0
            if (e.snr is not None) and (e.snr != 0.0):
                snr = e.snr
                snr_unc = e.snr_unc
            title += "S/N = %0.1f($\pm$%0.1f) " % (snr,snr_unc)
        else:
            title += "$\sigma$ = %g " % (e.sigma)

        if (e.chi2 is not None) and (e.chi2 != 666) and (e.chi2 != 0):
            title += " $\chi^2$ = %0.1f($\pm$%0.1f)" % (e.chi2,e.chi2_unc)


        #if e.dqs is None:
        #    e.dqs_score() #not doing dqs anymore
      #  title += "  Score = %0.1f (%0.2f)" % (e.dqs,e.dqs_raw)

        if not G.ZOO:
            if e.p_lae_oii_ratio_range is not None:
                # title += r"\nP(LAE)/P(OII) = %.4g $^{%.4g}_{%.4g}$" % \
                #          (round(e.p_lae_oii_ratio,3),round(e.p_lae_oii_ratio_range[2],3),round(e.p_lae_oii_ratio_range[1],3))

                try:
                    title += "\n" + r'P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$' % (round(e.p_lae_oii_ratio_range[0], 3),
                                                                                round(e.p_lae_oii_ratio_range[2], 3),
                                                                                round(e.p_lae_oii_ratio_range[1], 3))
                except:
                    title += "\n" + "P(LAE)/P(OII): %.4g" % (round(e.p_lae_oii_ratio,3))
            else:
                title += "\n" + r'P(LAE)/P(OII): $N/A\ ^{N/A}_{N/A}$'


            if (not e.using_best_gmag_ew) and (e.best_gmag_p_lae_oii_ratio_range is not None):
                try:
                    title += " (w: $%.4g\ ^{%.4g}_{%.4g}$)" % (round(e.best_gmag_p_lae_oii_ratio_range[0], 3),
                                                                  round(e.best_gmag_p_lae_oii_ratio_range[2], 3),
                                                                  round(e.best_gmag_p_lae_oii_ratio_range[1], 3))
                except:
                    log.debug("SDSS gmag PLAE title exception",exc_info=True)
                    title += " (w %.4g)" % round(e.best_gmag_p_lae_oii_ratio,3)

            if G.DISPLAY_PSEUDO_COLOR:
                if e.rvb is not None:
                    #debug version
                    # title += "\nColor = %0.03g (%0.3g,%0.3g) [%0.3g,%0.3g] {%d}" \
                    #          %(e.rvb['color'],e.rvb['color_err'][0],e.rvb['color_err'][1],
                    #            e.rvb['color_range'][0],e.rvb['color_range'][1],e.rvb['flag'])
                    # title += "\nColor = %0.03g [%0.3g,%0.3g]" \
                    #      % (e.rvb['color'], e.rvb['color_range'][0], e.rvb['color_range'][1])

                    title += "\nr/b: %0.2f($\pm$%0.2f),%0.2f($\pm$%0.2f),%0.2f($\pm$%0.2f)" \
                             % (min(e.rvb['red_flux_density_ujy'],999),
                                min(e.rvb['red_flux_density_err_ujy'],999),
                                min(e.rvb['blue_flux_density_ujy'],999),
                                min(e.rvb['blue_flux_density_err_ujy'],999),
                                min(e.rvb['ratio'],999),
                                min(e.rvb['ratio_err'],999))

            #if (e.dqs is not None) and (e.dqs_raw is not None):
            #    title += "  Score = %0.1f (%0.2f)" % (e.dqs, e.dqs_raw)



        if e.w > 0:
            #title = title + "\nLy$\\alpha$ Z = %g" % la_z
            title = title + "\nLyA z = %0.4f" % la_z
            if (oii_z > 0):
                title = title + "  OII z = %0.4f" % oii_z
            else:
                title = title + "  OII z = N/A"

        if not G.ZOO:
            good, scale_score, p_score = e.multiline_solution_score()

            #pick best eqw observered to use
            if (e.eqw_sdss_obs is not None) and (e.eqw_sdss_obs_unc is not None):
                l_eqw_obs =  e.eqw_sdss_obs
            elif (e.eqw_hetdex_gmag_obs is not None) and (e.eqw_hetdex_gmag_obs_unc is not None):
                l_eqw_obs =  e.eqw_hetdex_gmag_obs
            elif (e.eqw_line_obs is not None) and (e.eqw_line_obs_unc is not None):
                l_eqw_obs = e.eqw_line_obs
            else:
                l_eqw_obs = e.eqw_obs

            if ( good ):
                # strong solution
                sol = datakeep['detobj'].spec_obj.solutions[0]
                title += "\n*Q(%0.2f) %s(%d) z = %0.4f  EW_r = %0.1f$\AA$" %(p_score, sol.name, int(sol.central_rest),sol.z,
                                                                            l_eqw_obs/(1.0+sol.z))
            elif (scale_score > G.MULTILINE_MIN_WEAK_SOLUTION_CONFIDENCE):
                #weak solution ... for display only, not acceptabale as a solution
                #do not set the solution (sol) to be recorded
                sol = datakeep['detobj'].spec_obj.solutions[0]
                title += "\nQ(%0.2f) %s(%d) z = %0.4f  EW_r = %0.1f$\AA$" % \
                         ( p_score, sol.name, int(sol.central_rest), sol.z,l_eqw_obs / (1.0 + sol.z))
            else:
                title += "\n" #just to keep the spacing
            #    log.info("No singular, strong emission line solution.")


        #plt.subplot(gs[0:2, 0:3])
        plt.subplot(gs[0:2, 0:25])
        plt.text(0, 0.5, title, ha='left', va='center', fontproperties=font)
        if not G.ZEROTH_ROW_HEADER:
            plt.suptitle(time.strftime("%Y-%m-%d %H:%M:%S") +
                     "  Version " + G.__version__ +"  ", fontsize=8,x=1.0,y=0.98,
                     horizontalalignment='right',verticalalignment='top')
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if datakeep is not None:
            img_y = None
            if datakeep['xi']:
                try:
                    if G.LyC: #want this in a different position
                        plt.subplot(gs[0:2,64:91])
                    else:
                        plt.subplot(gs[0:2,28:59])

                    plt.gca().axis('off')
                    buf,img_y = self.build_2d_image(datakeep)

                    buf.seek(0)
                    im = Image.open(buf)
                    plt.imshow(im,interpolation='none') #needs to be 'none' else get blurring
                    #gs.tight_layout()

                    if G.ZOO_MINI:
                        e.image_2d_fibers_1st_col, _ = self.build_2d_image_1st_column_only(datakeep)
                except:
                    log.warning("Failed to build 2D cutout image.", exc_info=True)

                # if G.LyC:
                #     try:
                #         self.build_2d_LyC_image(datakeep,e.w/G.LyA_rest-1.0)
                #     except:
                #         pass

                # update emission with the ra, dec of all fibers
                # needs to be here, after build_2d_image so the 'index' and 'color' exist for assignment
                try:
                    e.fiber_locs = list(
                        zip(datakeep['ra'], datakeep['dec'], datakeep['color'], datakeep['index'], datakeep['d'],
                            datakeep['fib'],datakeep['ifux'],datakeep['ifuy']))
                except:
                    log.error("Error building fiber_locs", exc_info=True)

                if G.LyC:
                    try:
                        plt.subplot(gs[0:2,27:64])
                        plt.gca().axis('off')

                        buf,_ = self.build_2d_LyC_image(datakeep,e.w/G.LyA_rest-1.0)

                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im, interpolation='none')
                    except:
                        log.warning("Failed to build (LyCon) 2D cutout image.", exc_info=True)


                    try:
                        plt.subplot(gs[0:2,91:])
                        plt.gca().axis('off')
                        if img_y is not None:
                            buf = self.build_scattered_light_image(datakeep,img_y,key='scatter_lyc')
                        else:
                            buf = self.build_scattered_light_image(datakeep,key='scatter_lyc')

                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im,interpolation='none') #needs to be 'none' else get blurring
                    except:
                        log.warning("Failed to 2D cutout image.", exc_info=True)

                else:

                    try:
                        plt.subplot(gs[0:2,59:67])
                        plt.gca().axis('off')
                        if img_y is not None:
                            buf = self.build_scattered_light_image(datakeep,img_y,key='scatter_sky')
                        else:
                            buf = self.build_scattered_light_image(datakeep,key='scatter_sky')

                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im,interpolation='none') #needs to be 'none' else get blurring
                    except:
                        log.warning("Failed to 2D cutout image.", exc_info=True)


                    try:
                        plt.subplot(gs[0:2,67:75])
                        plt.gca().axis('off')
                        if img_y is not None:
                            buf = self.build_scattered_light_image(datakeep,img_y,key='scatter')
                        else:
                            buf = self.build_scattered_light_image(datakeep,key='scatter')

                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im,interpolation='none') #needs to be 'none' else get blurring
                    except:
                        log.warning("Failed to 2D cutout image.", exc_info=True)

                    #try:
                    #    plt.subplot(gs[0:2, 58:80])
                    #    plt.gca().axis('off')
                    #
                    #    buf = self.build_relative_fiber_locs(e)
                    #    buf.seek(0)
                    #    im = Image.open(buf)
                    #    plt.imshow(im, interpolation='none')  # needs to be 'none' else get blurring
                    #except:
                    #    log.warning("Failed to build relative fiber positions image.", exc_info=True)

                    try:
                        plt.subplot(gs[0:2,75:])
                        plt.gca().axis('off')
                        buf = self.build_spec_image(datakeep,e.w, dwave=1.0)
                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im,interpolation='none')#needs to be 'none' else get blurring

                        if G.ZOO_MINI:
                            e.image_1d_emission_fit = self.build_spec_image(datakeep, e.w, dwave=1.0,unlabeled=True)

                    except:
                        log.warning("Failed to build spec image.",exc_info = True)


                if G.SINGLE_PAGE_PER_DETECT:
                    #make the first part is own (temporary) page (to be merged later)
                    pages.append(fig)
                    plt.close('all')
                    try:
                        if G.SHOW_ALL_1D_SPECTRA:
                            figure_sz_y = figure_sz_y*2.0
                        fig = plt.figure(figsize=(G.FIGURE_SZ_X, figure_sz_y))
                        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                        plt.gca().axis('off')
                        buf = self.build_full_width_spectrum(datakeep, e.w)
                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im, interpolation='none')  # needs to be 'none' else get blurring

                        pages.append(fig) #append the second part to its own page to be merged later
                        plt.close()
                    except:
                        log.warning("Failed to build full width 1D spec/cutout image.", exc_info=True)

                    if G.PLOT_FULLWIDTH_2D_SPEC:
                        try:
                            fig = plt.figure(figsize=(G.FIGURE_SZ_X,G.GRID_SZ_Y))
                            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                            plt.gca().axis('off')
                            buf = self.build_full_width_2D_cutouts(datakeep, e.w/(1.0+G.LyA_rest))

                            buf.seek(0)
                            im = Image.open(buf)
                            plt.imshow(im, interpolation='none')  # needs to be 'none' else get blurring

                            pages.append(fig)  # append the second part to its own page to be merged later
                            plt.close()

                        except:
                            log.warning("Failed to build full width 2D spec/cutout image.", exc_info=True)

                else: #join this to the hetdex page
                    try:
                        plt.subplot(gs[2:, :])
                        plt.gca().axis('off')
                        buf = self.build_full_width_spectrum(datakeep, e.w)
                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im, interpolation='none')  # needs to be 'none' else get blurring
                    except:
                        log.warning("Failed to build full width spec/cutout image.", exc_info=True)
            else:
                pages.append(fig) #append what we have (done with a blank as well so the sizes are correct)
                plt.close('all')
                fig = plt.figure(figsize=(G.FIGURE_SZ_X, figure_sz_y))
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                plt.gca().axis('off')
                plt.text(0.5, 0.5, "Unable to locate or import reduced data (FITS)", ha='center', va='center', fontproperties=font)
                pages.append(fig)  # append the second part to its own page to be merged later
                plt.close()
        else:
            pages.append(fig)  # append what we have (done with a blank as well so the sizes are correct)
            plt.close('all')
            fig = plt.figure(figsize=(G.FIGURE_SZ_X, figure_sz_y))
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.gca().axis('off')
            plt.text(0.5, 0.5, "Unable to locate or import reduced data (FITS)", ha='center', va='center',
                     fontproperties=font)
            pages.append(fig)  # append the second part to its own page to be merged later
            plt.close()

        #safety check
        # update emission with the ra, dec of all fibers
        if e.fiber_locs is None:
            try:
                e.fiber_locs = list(
                    zip(datakeep['ra'], datakeep['dec'], datakeep['color'], datakeep['index'], datakeep['d'],
                        datakeep['fib']))
            except:
                log.error("Error building fiber_locs", exc_info=True)

        #always forcing SINGLE_PAGE_PER_DETECT as of version 1.1.0
        #if not G.SINGLE_PAGE_PER_DETECT:
        #    pages.append(fig)
        #    plt.close()
        #else, the pages were appended invidivually
        return pages

    def get_vrange(self,vals,scale=1.0,contrast=1.0):
        vmin = None
        vmax = None
        if scale == 0:
            scale = 1.0

        try:
            zscale = ZScaleInterval(contrast=1.0,krej=2.5) #nsamples=len(vals)
            vmin,vmax = zscale.get_limits(values=vals)
            vmin = vmin/scale
            vmax = vmax/scale
            log.info("HETDEX (zscale) vrange = (%f, %f) raw range = (%f, %f)" %(vmin,vmax,np.min(vals),np.max(vals)))
        except:
            log.info("Exception in hetdex::get_vrange:",exc_info =True)

        return vmin, vmax

    def clean_data_dict(self,datadict=None):
        if datadict is not None:
            dd = datadict
            for k in dd.keys():
                del dd[k][:]
        else:
            dd = {}
            dd['status'] =  0
            dd['detobj'] =  None
            dd['dit'] = []
            dd['side'] = []
            dd['amp'] = []
            dd['date'] = []
            dd['obsid'] = []
            dd['expid'] = []
            dd['fib'] = []
            dd['fib_idx1'] = []
            dd['ifu_slot_id'] = []
            dd['ifu_id'] = []
            dd['spec_id'] = []
            dd['xi'] = []
            dd['yi'] = []
            dd['xl'] = []
            dd['yl'] = []
            dd['xh'] = []
            dd['yh'] = []
            dd['ifux'] = []
            dd['ifuy'] = []
            dd['ds9_x'] = []
            dd['ds9_y'] = []
            dd['ds9_x_lyc'] = []
            dd['ds9_y_lyc'] = []
            dd['x_2d_lyc'] = []
            dd['y_2d_lyc'] = []
            dd['sn'] = []
            dd['fiber_sn'] = []
            #dd['wscore'] = [] #former dqs_score ... not using anymore (1.4.0a11+)
            dd['scatter'] = []
            dd['scatter_sky'] = []
            dd['scatter_lyc'] = []
            dd['d'] = []
            dd['dx'] = []
            dd['dy'] = []
            dd['im'] = []
            dd['fw_im'] = [] #full width (1024)
            dd['fxl'] = []
            dd['fxh'] = []
            dd['vmin1'] = []
            dd['vmax1'] = []
            dd['vmin2'] = []
            dd['vmax2'] = []
            dd['vmin3'] = []
            dd['vmax3'] = []
            dd['err'] = []
            dd['fw_err'] = []
            dd['pix'] = []
            dd['fw_pix'] = []
            dd['fiber_chi2'] = []
            dd['fiber_rms']= []
            dd['spec'] = []
            dd['specwave'] = []
            dd['fw_spec']  = []
            dd['fw_specwave'] = []

            #these are single entry for entire detection (i.e. summed up)
            dd['sumspec_wave'] = []
            dd['sumspec_cnts'] = []
            dd['sumspec_flux'] = []
            dd['sumspec_ferr'] = []
            dd['sumspec_2d'] = []
            dd['sumspec_cnts_zoom'] = []
            dd['sumspec_wave_zoom'] = []
            dd['sumspec_flux_zoom'] = []
            dd['sumspec_ferr_zoom'] = []

            #these are per fiber in the detection
            dd['fiber_weight'] = []
            dd['thruput'] = []
            dd['fluxcal_wave'] = []
            dd['fluxcal_cnts'] = []
            dd['fluxcal_flux'] = []
            dd['fluxcal_fluxerr'] = []
            dd['fluxcal_cont'] = []

            dd['cos'] = []
            dd['ra'] = []
            dd['dec'] = []
            dd['color'] = []
            dd['index'] = []
            dd['primary_idx'] = None #index in datakeep of the primary fiber
        return dd


    def build_data_dict(self,detobj):
        datakeep = None
        try:
            if self.panacea:
                datakeep = self.build_panacea_hetdex_data_dict(detobj)
            else:
                datakeep = self.build_hetdex_data_dict(detobj)
        except:
            log.error("Error! Cannot build datakeep.",exc_info=True)

        #detobj.dqs_score() #force_recompute=True) #not doing dqs score anymore
        if datakeep is None:
            log.error("Error! Cannot build datakeep.")
        return datakeep


    def build_hetdex_data_dict(self,e):#e is the emission detection to use
        if e is None:
            return None

        if e.type != 'emis':
            return None

        #basically cloned from Greg Z. make_visualization_detect.py; adjusted a bit for this code base
        datakeep = self.clean_data_dict()
        datakeep['detobj'] = e
        sort_list = []

        for side in SIDE:  # 'L' and 'R'
            for dither in range(len(self.dither.dx)):  # so, dither is 0,1,2
                dx = e.x - self.ifu_ctr.xifu[side] - self.dither.dx[dither]  # IFU is my self.ifu_ctr
                dy = e.y - self.ifu_ctr.yifu[side] - self.dither.dy[dither]

                d = np.sqrt(dx ** 2 + dy ** 2)

                # all locations (fiber array index) within dist_thresh of the x,y sky coords of the detection
                locations = np.where(d < dist_thresh)[0]

                for loc in locations:
                    sort_list.append(FitsSorter(None, d[loc], loc, side, dither))

                # sort from farthest to nearest ... yes, weird, but necessary for compatibility with
                # some cloned code f
                sort_list.sort(key=lambda x: x.dist, reverse=True)

        #this is for one side of one dither of one ifu
        #for loc in locations:
        for item in sort_list:
            side = item.side
            amp = None
            dither = item.dither
            loc = item.loc
            fiber = None
            #datakeep['d'].append(item.dist)  # distance (in arcsec) of fiber center from object center
            sci = self.get_sci_fits(dither, side)
            datakeep['fiber_sn'].append(item.fiber_sn)


            max_y, max_x = sci.data.shape

            #used later
            datakeep['color'].append(None)
            datakeep['index'].append(None)

            datakeep['dit'].append(dither + 1)
            datakeep['side'].append(side)

            f0 = self.dist[side].get_reference_f(loc + 1)
            xi = self.dist[side].map_wf_x(e.w, f0)
            yi = self.dist[side].map_wf_y(e.w, f0)

            #this the fiber_num for the side (1-224)
            fiber_num = self.dist[side].map_xy_fibernum(xi, yi)

            datakeep['fib'].append(fiber_num)
            xfiber = self.ifu_ctr.xifu[side][loc] + self.dither.dx[dither]
            yfiber = self.ifu_ctr.yifu[side][loc] + self.dither.dy[dither]
            xfiber += self.ifuy #yes this is correct xfiber gets ifuy
            yfiber += self.ifux
            datakeep['ifux'].append(self.ifu_ctr.xifu[side][loc])
            datakeep['ifuy'].append(self.ifu_ctr.yifu[side][loc])
            ra, dec = self.tangentplane.xy2raDec(xfiber, yfiber)
            datakeep['ra'].append(ra)
            datakeep['dec'].append(dec)
            xl = int(np.round(xi - xw))
            xh = int(np.round(xi + xw))
            yl = int(np.round(yi - yw))
            yh = int(np.round(yi + yw))

            datakeep['ds9_x'].append(1. + (xl + xh) / 2.)
            datakeep['ds9_y'].append(1. + (yl + yh) / 2.)

            xl = max(xl,0)
            xh = min(xh,max_x)
            yl = max(yl,0)
            yh = min(yh,max_y)

            # cure does not build specific fibers (don't know the info until here), so build now for the _fib.txt file
            try:
                fiber = elixer_fiber.Fiber(op.basename(sci.filename), str(self.specid), str(self.ifu_slot_id), str(self.ifu_id),
                              None, str(self.ymd), "None", "None", -1,e.id)

                if fiber:
                    #could parse the filename and get dither_time and dither_time_extended
                    #but they are not used right now
                    #fiber.scifits_idstring =
                    fiber.emis_x = int(xi[0])
                    fiber.emis_y = int(yi[0])
                    fiber.dither_idx = dither
                    fiber.expid = dither+1
                    fiber.fits = sci
                    fiber.side = side
                    fiber.obsid = sci.obsid
                    fiber.center_x = xfiber
                    fiber.center_y = yfiber
                    fiber.ra = ra
                    fiber.dec = dec
                    fiber.number_in_side = fiber_num[0]
                    if side == 'L':
                        fiber.number_in_ccd = fiber.number_in_side
                        if fiber.number_in_side < 113:
                            fiber.number_in_amp = fiber.number_in_side
                            fiber.amp = 'LU'
                        else:
                            fiber.number_in_amp = fiber.number_in_side - 112
                            fiber.amp = 'LL'
                    else:
                        fiber.number_in_ccd = fiber.number_in_side + 224
                        if fiber.number_in_side < 113:
                            fiber.number_in_amp = fiber.number_in_side
                            fiber.amp = 'RL'
                        else:
                            fiber.number_in_amp = fiber.number_in_side - 112
                            fiber.amp = 'RU'

                    amp = fiber.amp

                    # if e.wra:
                    #     fiber.dqs_score(e.wra,e.wdec)
                    # else:
                    #     fiber.dqs_score(e.ra,e.dec)
                    # datakeep['wscore'].append(fiber.dqs)

                    d = self.emis_to_fiber_distance(e, fiber)
                    if d is not None:
                        datakeep['d'].append(d)
                    else:
                        datakeep['d'].append(item.dist)

                    e.fibers.append(fiber)

            except:
                datakeep['d'].append(item.dist)
                #this is minor, so just a debug log
                log.debug("Error building fiber object for cure data in hetdex::build_hetdex_data_dict.", exc_info=True)


            # update ... +/- 3 fiber heights (yw) (with 2 gaps in between, so 5)
            datakeep['scatter'].append(sci.data[max(0, yl - 5 * yw):min(max_y, yh + 5 * yw),xl:xh])
                                      # max(0, xi - 10):min(max_x, xi + 10)])

            datakeep['xi'].append(xi)
            datakeep['yi'].append(yi)
            datakeep['xl'].append(xl)
            datakeep['yl'].append(yl)
            datakeep['xh'].append(xh)
            datakeep['yh'].append(yh)
            datakeep['fxl'].append(0)
            datakeep['fxh'].append(FRAME_WIDTH_X-1)
            #datakeep['d'].append(d[loc]) #distance (in arcsec) of fiber center from object center
            datakeep['sn'].append(e.sigma)

            #also get full x width data
            #would be more memory effecien to just grab full width,
            #  then in original func, slice as below

            if sci is not None:
                datakeep['im'].append(sci.data[yl:yh,xl:xh])
                datakeep['fw_im'].append(sci.data[yl:yh, 0:FRAME_WIDTH_X-1])

                z1, z2 = self.get_vrange(sci.data[yl:yh, xl:xh], scale=contrast1)
                log.debug("2D cutout zscale1 (smoothed) = %f, %f  for D,S,F = %d, %s, %d"
                          % (z1, z2, dither + 1, side, fiber_num))

                datakeep['vmin1'].append(z1)
                datakeep['vmax1'].append(z2)

                z1, z2 = self.get_vrange(sci.data[yl:yh, xl:xh], scale=contrast2)
                log.debug("2D cutout zscale2 (image) = %f, %f  for D,S,F = %d, %s, %d"
                          % (z1, z2, dither + 1, side, fiber_num))

                datakeep['vmin2'].append(z1)
                datakeep['vmax2'].append(z2)

                datakeep['err'].append(sci.err_data[yl:yh, xl:xh])

            #OLD ... using joined AMPS in SIDE
            # pix_fn = op.join(PIXFLT_LOC,'pixelflat_cam%s_%s.fits' % (sci.specid, side))
            # # specid (cam) in filename might not have leading zeroes
            # if not op.exists(pix_fn) and (sci.specid[0] == '0'):
            #     log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
            #     pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (sci.specid.lstrip("0"), sci.side))
            #
            # if op.exists(pix_fn):
            #     datakeep['pix'].append(pyfits.open(pix_fn)[0].data[yl:yh,xl:xh])
            # else:
            #     # todo: this is really sloppy ... make a better/more efficient pattern
            #     log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
            #     pix_x = xh - xl + 1
            #     pix_y = yh - yl + 1
            #     pix_blank = np.zeros((pix_y, pix_x))
            #     try:
            #         for x in range(pix_x / 2):
            #             for y in range(pix_y / 2):
            #                 pix_blank[y * 2, x * 2] = 999
            #     except:
            #         pass
            #     datakeep['pix'].append(deepcopy(pix_blank))


            load_blank = False

            pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (sci.specid, amp))
            # specid (cam) in filename might not have leading zeroes
            if not op.exists(pix_fn) and (sci.specid[0] == '0'):
                log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (sci.specid.lstrip("0"), amp))

            if op.exists(pix_fn):
                buf = flip_amp(amp,sci.ampname,pyfits.open(pix_fn)[0].data)
                if buf is not None:
                    datakeep['pix'].append(buf[yl:yh, xl:xh])
                else:
                    load_blank = True
            else:
                load_blank = True

            if load_blank:
                # todo: this is really sloppy ... make a better/more efficient pattern
                log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)

                datakeep['pix'].append(deepcopy(blank_pixel_flat(xh,xl,yh,yl)))

               # pix_x = xh - xl + 1
               # pix_y = yh - yl + 1
               # pix_blank = np.zeros((pix_y, pix_x))
               # try:
               #     for x in range(pix_x / 2):
               #         for y in range(pix_y / 2):
               #             pix_blank[y * 2, x * 2] = 999
               # except:
               #     pass
               # datakeep['pix'].append(deepcopy(pix_blank))


            #cosmic removed (but will assume that is the original data)
            #datakeep['cos'].append(fits.open(cos_fn)[0].data[yl:yh, xl:xh])

            #fiber extracted
            if len(sci.fe_data) > 0 and (sci.fe_crval1 is not None) and (sci.fe_cdelt1 is not None):
                nfib, xlen = sci.fe_data.shape
                wave = np.arange(xlen)*sci.fe_cdelt1 + sci.fe_crval1

                #this sometimes produces arrays of different lengths (+/- 1) [due to rounding?]
                #which causes problems later on, so just get the nearst point to the target wavelenght
                #and a fixed number of surrounding pixels
                #Fe_indl = np.searchsorted(wave,e.w-ww,side='left')
                #Fe_indh = np.searchsorted(wave,e.w+ww,side='right')

                center = np.searchsorted(wave,e.w,side='left')
                Fe_indl = center - int(round(ww))
                Fe_indh = center + int(round(ww))

                max_y, max_x = sci.fe_data.shape
                Fe_indl = max(Fe_indl, 0)
                Fe_indh = min(Fe_indh, max_x)

                datakeep['spec'].append(sci.fe_data[loc,Fe_indl:(Fe_indh+1)])
                datakeep['specwave'].append(wave[Fe_indl:(Fe_indh+1)])

                if fiber:
                    Fe_indl = center - PEAK_PIXELS
                    Fe_indh = center + PEAK_PIXELS

                    if (Fe_indl) < 0:
                        fiber.central_wave_pixels_bad = abs(Fe_indl)
                        fiber.central_emis_counts = np.zeros(fiber.central_wave_pixels_bad)
                        fiber.central_emis_wavelengths = np.zeros(fiber.central_wave_pixels_bad)

                        fiber.central_emis_counts = np.concatenate(
                            (fiber.central_emis_counts, sci.fe_data[loc, 0:(Fe_indh + 1)]))
                        fiber.central_emis_wavelengths = np.concatenate(
                            (fiber.central_emis_wavelengths, wave[0:(Fe_indh + 1)]))
                    elif Fe_indh >= max_x:
                        fiber.central_wave_pixels_bad = Fe_indh - max_x + 1
                        fiber.central_emis_counts = np.zeros(fiber.central_wave_pixels_bad)
                        fiber.central_emis_wavelengths = np.zeros(fiber.central_wave_pixels_bad)
                        fiber.central_emis_counts = np.concatenate(
                            (sci.fe_data[loc, Fe_indl:(max_x)], fiber.central_emis_counts))
                        fiber.central_emis_wavelengths = np.concatenate(
                            (wave[Fe_indl:(max_x)], fiber.central_emis_wavelengths))

                    # if (Fe_indh == (max_x)) or (Fe_indl == 0):
                    #    log.info("Peak too close to wavelength edge for fiber %s" % fiber.idstring)

                    else:
                        fiber.central_emis_counts = sci.fe_data[loc, Fe_indl:(Fe_indh + 1)]
                        fiber.central_emis_wavelengths = wave[Fe_indl:(Fe_indh + 1)]

                datakeep['fw_spec'].append(sci.fe_data[loc,:]) #never modified downstream
                datakeep['fw_specwave'].append(wave[:]) #never modified downstream

                # todo: set the weights correctly
                datakeep['fiber_weight'].append(1.0)

                #this is CURE for now, do not ever have this data
                if len(datakeep['sumspec_wave']) == 0:
                    #there is only ONE fluxcalibrated spectra for the entire detection (not one per fiber)
                    datakeep['sumspec_wave'] = e.sumspec_wavelength
                    datakeep['sumspec_cnts'] = e.sumspec_counts
                    datakeep['sumspec_flux'] = e.sumspec_flux
                    datakeep['sumspec_ferr'] = e.sumspec_fluxerr

        return datakeep


    #build_panacea_hetdex_data_dict
    def build_panacea_hetdex_data_dict(self, e):  # e is the emission detection to use
        if e is None:
            return None

        if e.type != 'emis':
            return None

        # basically cloned from Greg Z. make_visualization_detect.py; adjusted a bit for this code base
        datakeep = self.clean_data_dict()
        datakeep['detobj'] = e
        sort_list = []

        if len(e.fibers) > 0:
            #use fiber list rather than distance
            for f in e.fibers:
                dither = f.dither_idx
                if f.fits:
                    #this is the multi ifu/fiber case (no dither file, etc)
                    fits = f.fits
                else:#this is the older case with one ifu and dithers within a single observation
                    fits = self.get_sci_fits(dither,f.side,f.amp)

                if (fits is None) or (fits.data is None):
                    log.error("Error! Could not find appropriate fits file for fiber: %s"
                              % (f.idstring))
                    continue
                #look at specific fibers

                if e.wra:
                    ra = e.wra
                    dec = e.wdec
                elif e.ra:
                    ra = e.ra
                    dec = e.dec
                else:
                    ra = None
                    dec = None

                if (ra is not None) and (f.ra is not None):
                    try:
                        d = np.sqrt((np.cos(np.deg2rad(dec))*(ra - f.ra))**2 + (dec - f.dec)**2)*3600
                    except:
                        if f.ra and f.dec:
                            log.error("Missing required emission line (#%d) coordinates." % e.id)
                        elif e.wra and e.wdec:
                            log.error("Missing required fiber (%s) coordinates." % f.idstring)
                        else:
                            log.error("Missing required fiber (%s) and/or emission line (#%d) coords."
                                  % (f.idstring,e.id))
                        continue
                elif self.dither: #unweighted location
                            dx = e.x - f.center_x - self.dither.dx[dither]  # just floats, not arrays like below
                            dy = e.y - f.center_y - self.dither.dy[dither]
                            d = np.sqrt(dx ** 2 + dy ** 2)
                else:
                    log.error("Cannot compute fiber distances. Missing mandatory information.")
                    continue
                #turn fiber number into a location. Fiber Number 1 is at the top
                #which is loc (or index) 111
                #so loc = 112 - Fiber Number
             #   loc = f.number_in_amp-1
                loc = f.panacea_idx

                sort_list.append(FitsSorter(fits,d,loc,sn=f.sn,fiber=f))

            #we want these in the order given, but they print in reverse, so invert the order
            #sort_list.sort(key=lambda x: x.dist, reverse=True)
            sort_list = sort_list[::-1]

        else: #use fibers w/in 2"

            for fits in self.sci_fits:

                if (fits is None) or (fits.data is None):
                    log.error("Error! Invalid or empty fits for detection ID %d", e.id)
                    continue

                dither = fits.dither_index  # 0,1,2

                #we must have a dither file in this case
                # e.x and e.y are the sky x and y
                # dx and dy then are the distances of each fiber center from the sky location of the source
                dx = e.x - fits.fiber_centers[:, 0] - self.dither.dx[dither]
                dy = e.y - fits.fiber_centers[:, 1] - self.dither.dy[dither]

                d = np.sqrt(dx ** 2 + dy ** 2)

                # all locations (fiber array index) within dist_thresh of the x,y sky coords of the detection
                locations = np.where(d < dist_thresh)[0]

                for loc in locations:
                    sort_list.append(FitsSorter(fits,d[loc],loc))

            # sort from farthest to nearest ... yes, weird, but necessary for compatibility with
            # some cloned code f
            sort_list.sort(key=lambda x: x.dist,reverse=True)



        if len(sort_list) == 0: #nothing available (this should not happen normally, but is seen sometimes with filesystem problems)
        #if True:
         #   print("***** DEBUG ***** TURN THIS OFF *****")
            e.status = -1
            datakeep['status'] = -1
            return datakeep

        #for loc in locations:
        for item in sort_list:
            try:
                fits = item.fits

                if not fits:
                    #something is very wrong
                    log.error("Unexpected None fits in hetdex::build_panacea_hetdex_data_dict")

                dither = fits.dither_index  # 0,1,2 or more
                loc = item.loc
                fiber = item.fiber

                if fiber is None: # did not find it? impossible?
                    log.error("Error! Cannot identify fiber in HETDEX:build_panacea_hetdex_data_dict().")
                    fiber = elixer_fiber.Fiber(0,0,0,0,'XX',"","","",-1,-1)


                log.debug("Building data dict for " + fits.filename)
                datakeep['date'].append(fiber.dither_date) #already a str

                #reminder fiber might not have obsid or expid set (fiber is built before fits files are found)
                #and in some versions of the line file, obsid and expid are not available until after fits are found
                if fiber.obsid:
                    datakeep['obsid'].append(str(fiber.obsid))
                else:
                    fiber.obsid = self.obsid
                    datakeep['obsid'].append(str(self.obsid))

                if fiber.expid:
                    datakeep['expid'].append(str(fiber.expid))
                else:
                    fiber.expid = dither + 1
                    datakeep['expid'].append(str(fiber.expid))

                datakeep['fib_idx1'].append(str(fiber.panacea_idx+1))
                datakeep['ifu_slot_id'].append(str(fiber.ifuslot).zfill(3))
                datakeep['ifu_id'].append(str(fiber.ifuid).zfill(3))
                datakeep['spec_id'].append(str(fiber.specid).zfill(3))
                datakeep['fiber_sn'].append(item.fiber_sn)

                max_y, max_x = fits.data.shape

                # used laterrange(len(fits.wave_data[loc,:]))
                datakeep['color'].append(None)
                datakeep['index'].append(None)
                datakeep['dit'].append(dither + 1)

                datakeep['side'].append(fits.side)
                datakeep['amp'].append(fits.amp)

                #lowest number fiber is at the top, not the bottom
                #loc runs from the bottom and is zero based
                #so flip ... nominally:  112 - (loc+1) + offset for the amp
                if fiber.number_in_ccd == -1:
                    fiber.number_in_ccd = len(fits.fe_data) - (loc+1) + AMP_OFFSET[fits.amp]
                datakeep['fib'].append(fiber.number_in_ccd)

                if fiber.ra is None: #then there must be a dither file
                    xfiber = fits.fiber_centers[loc][0] + self.dither.dx[dither]
                    yfiber = fits.fiber_centers[loc][1] + self.dither.dy[dither]
                    xfiber += self.ifuy  # yes this is correct xfiber gets ifuy
                    yfiber += self.ifux
                    #ra and dec of the center of the fiber (loc)
                    ra, dec = self.tangentplane.xy2raDec(xfiber, yfiber)
                    datakeep['ra'].append(ra)
                    datakeep['dec'].append(dec)
                    fiber.ra = ra
                    fiber.dec = dec
                else: #only true in some panacea cases (if provided in detect line file)
                    datakeep['ra'].append(fiber.ra)
                    datakeep['dec'].append(fiber.dec)

                datakeep['ifux'].append(fiber.center_x)
                datakeep['ifuy'].append(fiber.center_y)

                d = self.emis_to_fiber_distance(e,fiber)
                if d is not None:
                    datakeep['d'].append(d)
                else:
                    datakeep['d'].append(item.dist)

                # if e.wra:
                #     fiber.dqs_score(e.wra, e.wdec)
                # else:
                #     fiber.dqs_score(e.ra, e.dec)
                # datakeep['wscore'].append(fiber.dqs)

                #don't think I am going to need any cutouts from the fits files, so lets save time and space and not load any
                #if e.syn_obs is None: #no .. actually need some of this ...
                #REMINDER: we are going from x,y to row, column (so x coord is the column number, and y is row)
                x_2D = np.interp(e.w,fits.wave_data[loc,:],range(len(fits.wave_data[loc,:])))
                y_2D = np.interp(x_2D,range(len(fits.trace_data[loc,:])),fits.trace_data[loc,:])

                if np.isnan(x_2D) or np.isnan(y_2D):
                    log.error(f"Invalid coordinates in hetdex.py build_panacea_hetdex_data_dict: x_2D,y_2D = {x_2D},{y_2D}")

                if G.LyC:
                    try:
                        x_LyC_2D = np.interp(895.0 * e.w/G.LyA_rest, fits.wave_data[loc, :], range(len(fits.wave_data[loc, :])))
                        y_LyC_2D = np.interp(x_LyC_2D, range(len(fits.trace_data[loc, :])), fits.trace_data[loc, :])

                        x_LyC_2D = int(np.round(x_LyC_2D))
                        y_LyC_2D = int(np.round(y_LyC_2D))
                    except:
                        x_LyC_2D = -1
                        y_LyC_2D = -1

                    datakeep['x_2d_lyc'].append(x_LyC_2D)
                    datakeep['y_2d_lyc'].append(y_LyC_2D)

                if not np.isnan(x_2D):
                    fiber.emis_x = int(x_2D)
                else:
                    fiber.emis_x = -1

                if not np.isnan(y_2D):
                    fiber.emis_y = int(y_2D)
                else:
                    fiber.emis_y = -1

                try:
                    log.info("Detect # %d, Fiber %s, Cam(%s), ExpID(%d) CCD X,Y = (%d,%d)" %
                             (e.id,fiber.idstring,fiber.specid,fiber.expid,int(x_2D),int(y_2D)))
                except:
                    pass

                try:
                    xl = int(np.round(x_2D - xw))
                    xh = int(np.round(x_2D + xw))
                    datakeep['ds9_x'].append(1. + (xl + xh) / 2.)
                except:
                    datakeep['ds9_x'].append(-1)

                try:
                    yl = int(np.round(y_2D - yw))
                    yh = int(np.round(y_2D + yw))
                    datakeep['ds9_y'].append(1. + (yl + yh) / 2.)
                except:
                    datakeep['ds9_y'].append(-1)


                #################################################
                #load pixel flat ... needed to modify data image
                # more pixel flat stuff a bit later on, to make the smaller cutout images
                #################################################

                #temporary hardcoded special cases for HDR2.1
                if (int(fits.specid) == 38):
                    log.info("PixelFlat special case for CAM038...")
                    # special case, the date matters
                    if fits.obs_ymd <= 20170101:
                        pix_fn = op.join(PIXFLT_LOC, 'cam038_20170101/pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))
                    elif fits.obs_ymd <= 20181001:
                        pix_fn = op.join(PIXFLT_LOC, 'cam038_20181001/pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))
                    else:
                        pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))

                elif (int(fits.specid == 51)):
                    # special case, the date matters
                    log.info("PixelFlat special case for CAM051...")
                    if fits.obs_ymd <= 20170101:
                        pix_fn = op.join(PIXFLT_LOC, 'cam051_20170101/pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))
                    elif fits.obs_ymd <= 20180705:
                        pix_fn = op.join(PIXFLT_LOC, 'cam051_20180705/pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))
                    else:
                        pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))

                else:
                    pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))
                # specid (cam) in filename might not have leading zeroes
                if not op.exists(pix_fn) and (fits.specid[0] == '0'):
                    log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                    pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid.lstrip("0"), fits.amp))

                if op.exists(pix_fn):
                    try:
                        pixflat_hdu = pyfits.open(pix_fn)
                        if pixflat_hdu:
                            try:
                                ampname = pixflat_hdu[0].header['ampname']
                            except:
                                log.info("AMPNAME missing from pixel flat header. Do not know if config 0/1 issue...")
                                ampname = None

                            pixel_flat_buf = flip_amp(fits.amp, ampname,pixflat_hdu[0].data)
                            pixflat_hdu.close()
                        else:
                            pixel_flat_buf = None
                            load_blank = True

                    except:
                        log.info("AMPNAME missing from pixel flat header. Do not know if config 0/1 issue...")
                        pixel_flat_buf = None
                        load_blank = True



                    #per Karl 20181012, zero out where the pixel flat is less than zero (indicates an error and a
                    #section to be ignored ... don't want it showing up as 'hot' data later on

                    #20200803 ... this is no longer necessary (do NOT mask out data due to pixel flat)
                    # if pixel_flat_buf is not None:
                    #     fits.data_sky[np.where(pixel_flat_buf <= 0)] = 0
                    #     fits.data[np.where(pixel_flat_buf <= 0)] = 0
                else:
                    pixel_flat_buf = None
                    load_blank = True

                # blank_xl = xl
                # blank_xh = xh
                # blank_yl = yl
                # blank_yh = yh
                # blank = np.zeros((yh-yl+1,xh-xl+1))
                # scatter_blank = np.zeros((yh - yl + 1 + 10*yw, xh - xl + 1)) #10*yw because +/- 5*yw in height

                if G.LyC:

                    xl_lyc = int(np.round(x_LyC_2D - xw)) #This IS a position
                    xh_lyc = int(np.round(x_LyC_2D + xw))
                    yl_lyc = int(np.round(y_LyC_2D - yw))
                    yh_lyc = int(np.round(y_LyC_2D + yw))
                    datakeep['ds9_x_lyc'].append(1. + (xl_lyc + xh_lyc) / 2.)
                    datakeep['ds9_y_lyc'].append(1. + (yl_lyc + yh_lyc) / 2.)

                    blank_xl = xl_lyc
                    blank_xh = xh_lyc
                    blank_yl = yl_lyc
                    blank_yh = yh_lyc
                    blank = np.zeros((yh_lyc - yl_lyc + 1, xh_lyc - xl_lyc + 1))
                    scatter_blank = np.zeros((yh_lyc - yl_lyc + 1 + 10 * yw, xh_lyc - xl_lyc + 1))  # 10*yw because +/- 5*yw in height

                    xl_lyc = max(xl_lyc, 0)
                    xh_lyc = min(xh_lyc, max_x - 1)
                    yl_lyc = max(yl_lyc, 0)
                    yh_lyc = min(yh_lyc, max_y - 1)

                    scatter_blank_bot = 5 * yw - (yl_lyc - max(0, yl_lyc - 5 * yw))  # start copy position in scatter_blank
                    scatter_blank_height = min(max_y - 1, yh_lyc + 5 * yw) - max(0, yl_lyc - 5 * yw)  # number of pixels to copy

                    scatter_blank[scatter_blank_bot:scatter_blank_bot + scatter_blank_height + 1,
                    (xl_lyc - blank_xl):(xl_lyc - blank_xl) + (xh_lyc - xl_lyc) + 1] = \
                        fits.data[max(0, yl_lyc - 5 * yw):min(max_y - 1, yh_lyc + 5 * yw) + 1, xl_lyc:xh_lyc + 1]

                    datakeep['scatter_lyc'].append(deepcopy(scatter_blank))

                blank_xl = xl
                blank_xh = xh
                blank_yl = yl
                blank_yh = yh
                blank = np.zeros((yh - yl + 1, xh - xl + 1))
                scatter_blank = np.zeros((yh - yl + 1 + 10 * yw, xh - xl + 1))  # 10*yw because +/- 5*yw in height

                xl = max(xl, 0)
                xh = min(xh, max_x-1)
                yl = max(yl, 0)
                yh = min(yh, max_y-1)

                scatter_blank_bot = 5 * yw - (yl - max(0, yl - 5 * yw)) #start copy position in scatter_blank
                scatter_blank_height = min(max_y-1, yh + 5 * yw) - max(0, yl - 5 * yw)   #number of pixels to copy

                scatter_blank[scatter_blank_bot:scatter_blank_bot + scatter_blank_height +1,
                             (xl - blank_xl):(xl - blank_xl) + (xh - xl) + 1] = \
                    fits.data[max(0, yl - 5 * yw):min(max_y-1, yh + 5 * yw) + 1, xl:xh + 1]

                datakeep['scatter'].append(deepcopy(scatter_blank))


                #now with the sky NOT subtracted ... the indices are the same, just a different fits image
                scatter_blank[scatter_blank_bot:scatter_blank_bot + scatter_blank_height + 1,
                                (xl - blank_xl):(xl - blank_xl) + (xh - xl) + 1] = \
                    fits.data_sky[max(0, yl - 5 * yw):min(max_y - 1, yh + 5 * yw) + 1, xl:xh + 1]
                datakeep['scatter_sky'].append(deepcopy(scatter_blank))

                datakeep['xi'].append(x_2D)
                datakeep['yi'].append(y_2D)

                datakeep['xl'].append(blank_xl)
                datakeep['yl'].append(blank_yl)
                datakeep['xh'].append(blank_xh)
                datakeep['yh'].append(blank_yh)

                datakeep['fxl'].append(0)
                datakeep['fxh'].append(FRAME_WIDTH_X - 1)

                datakeep['sn'].append(e.sigma)

                blank[(yl-blank_yl):(yl-blank_yl)+(yh-yl)+1, (xl-blank_xl):(xl-blank_xl)+(xh-xl)+1] = \
                    fits.data[yl:yh+1, xl:xh+1]

                datakeep['im'].append(deepcopy(blank))
                datakeep['fw_im'].append(fits.data[yl:yh, 0:FRAME_WIDTH_X - 1])

                z1, z2 = self.get_vrange(fits.data[yl:yh, xl:xh],scale=contrast1)
                log.debug("2D cutout zscale1 (smoothed) = %f, %f  for D,S,F = %d, %s, %d"
                          %(z1,z2,dither+1,fits.side,fiber.number_in_ccd))

                # z1,z2 = self.get_vrange(sci.data[yl:yh,xl:xh])
                datakeep['vmin1'].append(z1)
                datakeep['vmax1'].append(z2)

                z1, z2 = self.get_vrange(fits.data[yl:yh, xl:xh],scale=contrast2)
                log.debug("2D cutout zscale2 (image) = %f, %f  for D,S,F = %d, %s, %d"
                          %(z1,z2,dither+1,fits.side,fiber.number_in_ccd))

                datakeep['vmin2'].append(z1)
                datakeep['vmax2'].append(z2)

                try:
                    z1, z2 = self.get_vrange(fits.data_sky[yl:yh, xl:xh], scale=contrast3)
                    log.debug("2D cutout zscale3 (image) = %f, %f  for D,S,F = %d, %s, %d"
                              % (z1, z2, dither + 1, fits.side, fiber.number_in_ccd))

                    datakeep['vmin3'].append(z1)
                    datakeep['vmax3'].append(z2)
                except:
                    log.error("Could net get contrast stretch for sky NOT subtracted 2D spectra")

                blank[(yl-blank_yl):(yl-blank_yl)+(yh-yl)+1,(xl-blank_xl):(xl-blank_xl)+(xh-xl)+1] = \
                    fits.err_data[yl:yh + 1,xl:xh + 1]

                datakeep['err'].append(deepcopy(blank))

                if G.LyC:
                    datakeep['fw_err'].append(deepcopy(fits.err_data[yl:yh, 0:FRAME_WIDTH_X - 1]))


                #OLD ... using side
                # pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid, fits.side))
                # #specid (cam) in filename might not have leading zeroes
                # if not op.exists(pix_fn) and (fits.specid[0] == '0'):
                #     log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                #     pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid.lstrip("0"), fits.side))
                #
                # if op.exists(pix_fn):
                #     blank[(yl - blank_yl):(yl - blank_yl) + yh + 1, (xl - blank_xl):(xl - blank_xl) + (xh-xl) + 1] = \
                #         pyfits.open(pix_fn)[0].data[yl:yh + 1, xl:xh + 1]
                #
                #     datakeep['pix'].append(deepcopy(blank))
                # else:
                #     #todo: this is really sloppy ... make a better/more efficient pattern
                #     log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                #     pix_x = xh - xl + 1
                #     pix_y = yh - yl + 1
                #     pix_blank = np.zeros((pix_y, pix_x))
                #     try:
                #         for x in range(pix_x/2):
                #             for y in range (pix_y/2):
                #                 pix_blank[y*2,x*2] = 999
                #     except:
                #         pass
                #     datakeep['pix'].append(deepcopy(pix_blank))


                #########################################################
                # pixel flats continued ... build smaller cutout piece
                #########################################################
                load_blank = False

                if pixel_flat_buf is not None: #loaded a few dozen lines above
                    blank[(yl - blank_yl):(yl - blank_yl) + (yh - yl) + 1,
                    (xl - blank_xl):(xl - blank_xl) + (xh - xl) + 1] = \
                        pixel_flat_buf[yl:yh + 1, xl:xh + 1]
                    datakeep['pix'].append(deepcopy(blank))
                    datakeep['fw_pix'].append(deepcopy(pixel_flat_buf[yl:yh, 0:FRAME_WIDTH_X - 1]))

                    #note: check for bad flat in the plot creation where the owning fiber is better known
                else:
                    load_blank = True

                if load_blank:
                    # todo: this is really sloppy ... make a better/more efficient pattern
                    log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                    datakeep['pix'].append(deepcopy(blank_pixel_flat(xh, xl, yh, yl)))


                #1D spectrum (spec is counts, specwave is the corresponding wavelength)
                wave = fits.wave_data[loc,:]
                # this sometimes produces arrays of different lengths (+/- 1) [due to rounding?]
                # which causes problems later on, so just get the nearst point to the target wavelenght
                # and a fixed number of surrounding pixels

                #Fe_indl = np.searchsorted(wave, e.w - ww, side='left')
                #Fe_indh = np.searchsorted(wave, e.w + ww, side='right')
                #want say, approx +/- 50 angstroms

                center = np.searchsorted(wave, e.w, side='left')

                #for the chi2 and rms of the fiber profile
                #grab center pixel (wavelength) +/- 2
                ch_indl = max(center - 2, 0)
                ch_indh = min(center + 2, max_x)

                datakeep['fiber_chi2'].append(fits.fiber_chi2[loc, ch_indl:(ch_indh + 1)])
                datakeep['fiber_rms'].append(fits.fiber_rms[loc, ch_indl:(ch_indh + 1)])

                Fe_indl = center - int(round(ww))
                Fe_indh = center + int(round(ww))

                max_y, max_x = fits.fe_data.shape
                Fe_indl = max(Fe_indl, 0)
                Fe_indh = min(Fe_indh, max_x)

                #fe_data is "sky_subtracted" ... the counts
                #wave is "wavelength" ... the corresponding wavelength

                if len(fiber.fluxcal_central_emis_wavelengths) > 0:
                    log.info("Using flux-calibrated spectra for central emission.")
                    datakeep['spec'].append(fiber.fluxcal_central_emis_counts)
                    datakeep['specwave'].append(fiber.fluxcal_central_emis_wavelengths)


                else:
                    log.info("Using panacea fiber-extracted data for central emission.")
                    datakeep['spec'].append(fits.fe_data[loc, Fe_indl:(Fe_indh+1)])
                    datakeep['specwave'].append(wave[Fe_indl:(Fe_indh+1)])

                if fiber:
                    Fe_indl = center - PEAK_PIXELS
                    Fe_indh = center + PEAK_PIXELS

                    if (Fe_indl) < 0:
                        fiber.central_wave_pixels_bad = abs(Fe_indl)
                        fiber.central_emis_counts = np.zeros(fiber.central_wave_pixels_bad)
                        fiber.central_emis_wavelengths = np.zeros(fiber.central_wave_pixels_bad)

                        fiber.central_emis_counts = np.concatenate(
                            (fiber.central_emis_counts,fits.fe_data[loc,0:(Fe_indh+1)]))
                        fiber.central_emis_wavelengths = np.concatenate(
                            (fiber.central_emis_wavelengths, wave[0:(Fe_indh + 1)]))
                    elif Fe_indh >= max_x:
                        fiber.central_wave_pixels_bad = Fe_indh - max_x + 1
                        fiber.central_emis_counts = np.zeros(fiber.central_wave_pixels_bad)
                        fiber.central_emis_wavelengths = np.zeros(fiber.central_wave_pixels_bad)
                        fiber.central_emis_counts = np.concatenate(
                            (fits.fe_data[loc, Fe_indl:(max_x)],fiber.central_emis_counts))
                        fiber.central_emis_wavelengths = np.concatenate(
                        (wave[Fe_indl:(max_x)], fiber.central_emis_wavelengths))

                    #if (Fe_indh == (max_x)) or (Fe_indl == 0):
                    #    log.info("Peak too close to wavelength edge for fiber %s" % fiber.idstring)

                    else:
                        fiber.central_emis_counts = fits.fe_data[loc,Fe_indl:(Fe_indh+1)]
                        fiber.central_emis_wavelengths = wave[Fe_indl:(Fe_indh+1)]

                datakeep['fw_spec'].append(fits.fe_data[loc, :])
                datakeep['fw_specwave'].append(wave[:])

                #todo: set the weights correctly
                datakeep['fiber_weight'].append(fiber.relative_weight)
                datakeep['thruput'].append(fiber.fluxcal_central_emis_thru)

                datakeep['fluxcal_wave'].append(fiber.fluxcal_central_emis_wavelengths)
                datakeep['fluxcal_cnts'].append(fiber.fluxcal_central_emis_counts)
                datakeep['fluxcal_flux'].append(fiber.fluxcal_central_emis_flux)
                datakeep['fluxcal_fluxerr'].append(fiber.fluxcal_central_emis_fluxerr)
                datakeep['fluxcal_cont'].append(fiber.fluxcal_emis_cont)

                if len(datakeep['sumspec_wave']) == 0:
                    # there is only ONE summed fluxcalibrated spectra for the entire detection (not one per fiber)
                    datakeep['sumspec_wave'] = e.sumspec_wavelength
                    datakeep['sumspec_cnts'] = e.sumspec_counts
                    datakeep['sumspec_flux'] = e.sumspec_flux
                    datakeep['sumspec_ferr'] = e.sumspec_fluxerr

                    datakeep['sumspec_2d'] = e.sumspec_2d_zoom
                    datakeep['sumspec_cnts_zoom'] = e.sumspec_counts_zoom
                    datakeep['sumspec_wave_zoom'] = e.sumspec_wavelength_zoom
                    datakeep['sumspec_flux_zoom'] = e.sumspec_flux_zoom
                    datakeep['sumspec_ferr_zoom'] = e.sumspec_fluxerr_zoom

                    #make Karl's 2d cutout the shame shape, keeping its zero
                    #y_2D, x_2D = np.shape(blank)
            except Exception as ex:
                if type(ex) is ValueError:
                    log.error(f"Exception building datakeep. Not fatal (ValueError). {e.entry_id}", exc_info=True)
                else:
                    log.error("Exception building datakeep. Not fatal.",exc_info=True)
        #end for loop

        return datakeep


    def make_fiber_colors(self,num_colors,total_fibers):
        #this is a weird setup and is necessary only for historical reasons
        #  the color array originally fit the fiber arrangement, but was co-opted and grown and never refactored
        #  so this is neccessary for continued compatibility with various old sections of code

        #in general the setup is:
        # (color idx) (fiber no.)  ~ color
        #   0           n           gray
        #   ...         n-1,n-2...  gray
        #   x           4           red
        #   x+1         3           yellow
        #   x+2         2           green
        #   x+3         1           blue
        #   x+4 (n+1)   --          gray (pad)  (for plotting organization)
        #   x+5 (n+2)   --          gray (pad)  (for plotting organization)

        #get num_colors as color, the rest are grey
        #norm = plt.Normalize()
        #colors = plt.cm.hsv(norm(np.arange(num_colors)))
        #colors = [[1,0,0,1],[.98,.96,0,1],[0,1,0,1],[0,0,1,1]] #red, yellow, green, blue
        colors = [[1, 0, 0, 1], [1.0, 0.65, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]  # red, orange, green, blue
        if num_colors < 4:
            colors = colors[(4-num_colors):] #keep the last colors
        pad = [[0,0,0,0]]*2
        if total_fibers > num_colors:
            cf = 0.3
            alpha = 0.6
            greys = [[cf,cf,cf,alpha]]*(total_fibers-num_colors)
            colors = np.vstack((greys, colors,pad))
        return colors


    #2d spectra cutouts (one per fiber)
    def build_2d_image(self,datakeep):

        #not dynamic, but if we are going to add a combined 2D spectra cutout at the top, set this to 1
        add_summed_image = 1 #note: adding w/cosmics masked out
        frac_y_separator = 0.2 #add a separator between the colored fibers and the black (summed) fiber at this
                               #fraction of the cell height

        detobj = datakeep['detobj']

        cmap = plt.get_cmap('gray_r')

        colors = self.make_fiber_colors(min(4,len(datakeep['ra'])),len(datakeep['ra']))# + 2 ) #the +2 is a pad in the call
        num_fibers = len(datakeep['xi'])
        num_to_display = min(MAX_2D_CUTOUTS,num_fibers) + add_summed_image  #for the summed images
        # bordbuff = 0.005
        # borderxl = 0.06 #leave room on the left for labels (weight, chi2, fiber#)
        # borderxr = 0.16 #leave room on right for position and fiber info
        # borderyb = 0.06
        # borderyt = 0.16 #leave room at the top for labels

        bordbuff = 0.01
        borderxl = 0.06
        borderxr = 0.12
        borderyb = 0.00
        borderyt = 0.15

        #the +1 for the summed image
        dx = (1. - borderxl - borderxr) / 3.
        dy = (1. - borderyb - borderyt) / (num_to_display)
        dx1 = (1. - borderxl - borderxr) / 3.
        dy1 = (1. - borderyb - borderyt - (num_to_display) * bordbuff) / (num_to_display)
        Y = (yw / dy) / (xw / dx) * 5. + frac_y_separator * dy #+ 0.2 as a separator at the top

        Y = max(Y,0.8) #set a minimum size

        fig = plt.figure(figsize=(5, Y), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        # previously sorted in order from largest distances to smallest
        ind = list(range(len(datakeep['d'])))

        #assume all the same shape
        summed_image = np.zeros(datakeep['im'][ind[0]].shape)

        #sanity check for the top few fiber cutouts as identical
        duplicate_weight = 0
        try:
            #trip on any one of the top 3 as identical, checked in weight order
            if not np.any(datakeep['im'][-1] - datakeep['im'][-2]):
                #blue and green fiber cutouts are the same
                log.info("Probable spurious detection. Duplicate fiber cutouts detected (blue and green).")
                duplicate_weight += datakeep['fiber_weight'][-1] + datakeep['fiber_weight'][-2]
            elif not np.any(datakeep['im'][-1] - datakeep['im'][-3]):
                #blue and yellow fiber cutouts are the same
                log.info("Probable spurious detection. Duplicate fiber cutouts detected (blue and yellow).")
                duplicate_weight += datakeep['fiber_weight'][-1] + datakeep['fiber_weight'][-3]
            elif not np.any(datakeep['im'][-2] - datakeep['im'][-3]):
                #green and yellow fiber cutouts are the same
                log.info("Probable spurious detection. Duplicate fiber cutouts detected (green and yellow).")
                duplicate_weight += datakeep['fiber_weight'][-2] + datakeep['fiber_weight'][-3]

            if duplicate_weight != 0:
                detobj.duplicate_fiber_cutout_pair_weight = duplicate_weight

        except:
            log.warning("Excpetion comparing fiber cutouts in hetdex.py build_2d_image()",exc_info=True)

        #need i to start at zero
        #building from bottom up

        top_pixels = []
        grid_idx = -1
        plot_label = ""
        plot_label_color = "k"
        for i in range(num_fibers+add_summed_image):
            make_display = False
            plot_label = ""
            plot_lable_color = "k"
            if i < num_fibers:
                pcolor = colors[i] #, 0:4] #keep the 4th value (alpha value) ... need that to lower the alpha of the greys
                datakeep['color'][ind[i]] = pcolor
                datakeep['index'][ind[i]] = num_fibers -i

                if i > (num_fibers - num_to_display):#we are in reverse order (building the bottom cutout first)
                    make_display = True
                    grid_idx += 1
                    is_a_fiber = True
                    ext = list(np.hstack([datakeep['xl'][ind[i]], datakeep['xh'][ind[i]],
                                          datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))

                    # set the hot (cosmic) pixel values to zero then employ guassian_filter
                    a = datakeep['im'][ind[i]]
                    a = np.ma.masked_where(datakeep['err'][ind[i]] == -1, a)
                    a = np.ma.filled(a, 0.0)

                    summed_image += a * datakeep['fiber_weight'][ind[i]]

                    # GF = gaussian_filter(datakeep['im'][ind[i]], (2, 1))
                    GF = gaussian_filter(a, (2, 1))

                    gauss_vmin = datakeep['vmin1'][ind[i]]
                    gauss_vmax = datakeep['vmax1'][ind[i]]

                    pix_image = datakeep['pix'][ind[i]]
                    #bad_pix_image = datakeep['pix'][ind[i]][:]
                    image = datakeep['im'][ind[i]]  # im can be the cosmic removed version, depends on G.PreferCosmicCleaned
                    bad_pix_value = -99999

                    # check for bad flat in the center (where emission line would be)
                    # check 9 central pixels (using "blank")
                    try:
                        flat = pix_image.flatten()
                        nonzero = len(np.where(flat != 0)[0])
                        if nonzero > 0 and float(nonzero)/float(len(flat)) > 0.5:
                            cy,cx = np.array(np.shape(pix_image))//2
                            #cntr = pix_image[cy - 1:cy + 2, cx - 2:cx + 3]  # center 3 high, 5 wide
                            #cntr = pix_image[cy-2:cy+3,cx-2:cx+3]   #center 25
                            #cntr = pix_image[cy - 3:cy + 4, cx - 3:cx + 4]  # center 49
                            cntr = pix_image[cy - 1:cy + 2, cx - 1:cx + 2]  # center 9 (3x3)

                            # nan_pix_image = copy(pix_image) #makes no difference to 4 or 5 decimal places
                            # nan_pix_image[cy - 1:cy + 2, cx - 1:cx + 2] = np.nan

                            #bad_pix_value = np.nanmedian(pix_image) * G.MIN_PIXEL_FLAT_CENTER_RATIO
                            if G.PIXEL_FLAT_ABSOLUTE_BAD_VALUE > -1: #based on a fixed value
                                bad_pix_value = G.PIXEL_FLAT_ABSOLUTE_BAD_VALUE

                                #Either condition below will trip bad flat
                                #if the average of the center (-the error) is below the absolute bad value
                                #OR if the mean of the center is less than a fraction of the median of the rest

                                #mean - 1std dev ... should it be -2x std dev?
                                cntr_avg = np.nanmean(cntr) - np.nanstd(cntr)
                                detobj.fibers[len(detobj.fibers) - i - 1].pixel_flat_center_avg = cntr_avg
                                if cntr_avg < G.PIXEL_FLAT_ABSOLUTE_BAD_VALUE:
                                    #could be bad
                                    log.info("Possible bad pixel flat at emission line position")

                                # yes, I want mean for cntr and median for the whole cutout
                                cntr_ratio = np.nanmean(cntr) / np.nanmedian(pix_image)
                                detobj.fibers[len(detobj.fibers) - i - 1].pixel_flat_center_ratio = cntr_ratio
                                # sorting is different, need to reverse
                                if cntr_ratio < G.MIN_PIXEL_FLAT_CENTER_RATIO:
                                    # could be bad
                                    log.info("Possible bad pixel flat at emission line position")

                            else: #based on standard deviation
                                mask_pix_image = np.ma.masked_where((pix_image == 0)|(np.isnan(pix_image)), pix_image)
                                bad_pix_value = np.ma.median(mask_pix_image) - \
                                                G.MARK_PIXEL_FLAT_DEVIATION*np.ma.std(mask_pix_image)

                                #here, just for reference
                                cntr_avg = np.nanmean(cntr) - np.nanstd(cntr)
                                detobj.fibers[len(detobj.fibers) - i - 1].pixel_flat_center_avg = cntr_avg

                                #yes, I want mean for cntr and median for the whole cutout
                                cntr_ratio = np.nanmean(cntr) / np.nanmedian(pix_image)
                                #sorting is different, need to reverse
                                detobj.fibers[len(detobj.fibers)-i-1].pixel_flat_center_ratio = cntr_ratio
                                if cntr_ratio < G.MIN_PIXEL_FLAT_CENTER_RATIO:
                                    #could be bad
                                    log.info("Possible bad pixel flat at emission line position")
                    except:
                        log.debug("Exception checking for bad pixel flat",exc_info=True)

                    #update pix_image to a mask to use later when ploting
                    #pix_image = np.ma.masked_where((pix_image < bad_pix_value) & (pix_image != 0), pix_image)
                    pix_image = np.ma.masked_where(pix_image < bad_pix_value, pix_image)

                    #check for same pixel (as a string for easy compare (all integer values)
                    #after the loop, make sure they are unique
                    top_pixels.append(f"{datakeep['ds9_x'][ind[i]]},{datakeep['ds9_y'][ind[i]]}" )

                    cmap1 = cmap
                    cmap1.set_bad(color=[0.2, 1.0, 0.23])
                    image = np.ma.masked_where(datakeep['err'][ind[i]] == -1, image)
                    #image = np.ma.masked_where(datakeep['im'][ind[i]] == 0, image) #just testing
                    img_vmin = datakeep['vmin2'][ind[i]]
                    img_vmax = datakeep['vmax2'][ind[i]]

                    #plot_label = str(num_fibers-i)
                    #plot_label = str("%0.2f" % datakeep['fiber_weight'][ind[i]]).lstrip('0') #save space, kill leading 0
                    plot_label = str("%0.2f" % datakeep['fiber_weight'][ind[i]])
                    plot_label_color = 'k'
                    try:
                        max_chi2 = max(datakeep['fiber_chi2'][ind[i]])
                        if max_chi2 > 4.0:
                            plot_label_color = 'red'
                        #plot_label += "\n"+ r"$\chi^2$" + "%0.1f " % max_chi2
                        if max_chi2 > 10.0:
                            plot_label += "\n" + "%0.1f " % max_chi2
                        else:
                            plot_label += "\n" + "%0.2f " % max_chi2
                    except:
                        pass
                    #plot_label += "\n" + "#" + str(datakeep['fib'][ind[i]]).zfill(3)
                    plot_label += "\n" + str(datakeep['fib'][ind[i]]).zfill(3)

                    #the last one is the top one and is the primary
                    datakeep['primary_idx'] = ind[i]
                else: #not one of the top 4 ... but still add to the sum
                    a = datakeep['im'][ind[i]]
                    a = np.ma.masked_where(datakeep['err'][ind[i]] == -1, a)
                    a = np.ma.filled(a, 0.0)

                    summed_image += a * datakeep['fiber_weight'][ind[i]]

            else: #this is the top image (the sum)
                is_a_fiber = False
                make_display = True
                grid_idx += 1
                pcolor = 'k'
                ext = None
                pix_image = None #blank_pixel_flat() #per Karl, just leave it blank, no pattern
                plot_label = "SUM"

                GF = gaussian_filter(summed_image, (2, 1))
                image = summed_image
                img_vmin, img_vmax = self.get_vrange(summed_image, scale=contrast2)
                gauss_vmin, gauss_vmax =  self.get_vrange(summed_image, scale=contrast1)

            if make_display:

                if is_a_fiber:
                    y_separator = 0.0
                else:
                    y_separator = frac_y_separator * dy

                borplot = plt.axes([borderxl + 0. * dx, borderyb + grid_idx * dy + y_separator, 3 * dx, dy])
                smplot = plt.axes([borderxl + 2. * dx - bordbuff / 3., borderyb + grid_idx * dy + bordbuff / 2. + y_separator, dx1, dy1])
                pixplot = plt.axes(
                    [borderxl + 1. * dx + 1 * bordbuff / 3., borderyb + grid_idx * dy + bordbuff / 2.  + y_separator, dx1, dy1])
                imgplot = plt.axes([borderxl + 0. * dx + bordbuff / 2., borderyb + grid_idx * dy + bordbuff / 2. + y_separator, dx1, dy1])
                autoAxis = borplot.axis()

                rec = plt.Rectangle((autoAxis[0] + bordbuff / 2., autoAxis[2] + bordbuff / 2.),
                                    (autoAxis[1] - autoAxis[0]) * (1. - bordbuff),
                                    (autoAxis[3] - autoAxis[2]) * (1. - bordbuff), fill=False, lw=3,
                                    color=pcolor, zorder=1)
                rec = borplot.add_patch(rec)
                borplot.set_xticks([])
                borplot.set_yticks([])
                borplot.axis('off')

                smplot.imshow(GF,
                              origin="lower", cmap=cmap,
                              interpolation="none", vmin=gauss_vmin,
                              vmax=gauss_vmax,
                              extent=ext)

                smplot.set_xticks([])
                smplot.set_yticks([])
               # smplot.axis(ext)
                smplot.axis('off')

                if pix_image is not None:
                    #symmetric about 1.0 (even though nominal max is just over 1 and the lower can be very low)
                    #AND yes, these are fixed values ... we want the pixel flats to be uniform across all reports
                    vmin_pix = 0.8
                    vmax_pix = 1.2
                    pix_cmap = plt.get_cmap('gray')
                    pix_cmap.set_bad(color=[1.0, 0.2, 0.2])
                    pixplot.imshow(pix_image,
                                   origin="lower", cmap=pix_cmap,
                                   interpolation="none", vmin=vmin_pix, vmax=vmax_pix,
                                   extent=ext) #vmin=0.9, vmax=1.1


                    if G.MARK_PIXEL_FLAT_CENTER:
                        #set circle here ... at center with radius 3 pix? maybe a 0.5 alpha filled circle
                        #corresponds to the emision line position
                        cx = (pixplot.get_xlim()[1]+pixplot.get_xlim()[0])/2.0
                        cy = (pixplot.get_ylim()[1]+pixplot.get_ylim()[0])/2.0

                        circ = mpatches.Circle((cx,cy), radius=4,
                                                       facecolor='gold', fill=True, alpha=0.2,
                                                       edgecolor='none', linestyle="solid")

                        pixplot.add_patch(circ)

                #still need to always turn off the axis
                pixplot.set_xticks([])
                pixplot.set_yticks([])
                # pixplot.axis(ext)
                pixplot.axis('off')

                imgplot.imshow(image,
                               origin="lower", cmap=cmap1,
                               vmin=img_vmin,
                               vmax=img_vmax,
                               interpolation="none",extent=ext)

                imgplot.set_xticks([])
                imgplot.set_yticks([])
               # imgplot.axis(ext)
                imgplot.axis('off')

                if i < num_fibers:
                    xi = datakeep['xi'][ind[i]]
                    yi = datakeep['yi'][ind[i]]
                    xl = int(np.round(xi - ext[0] - res[0] / 2.))
                    xh = int(np.round(xi - ext[0] + res[0] / 2.))
                    yl = int(np.round(yi - ext[2] - res[0] / 2.))
                    yh = int(np.round(yi - ext[2] + res[0] / 2.))

                    sn = datakeep['fiber_sn'][ind[i]]

                    if sn is None:
                        if self.panacea:
                            sn = -99 #so will fail the check and not print

                        else: #this only works (relatively well) for Cure
                            S = np.where(datakeep['err'][ind[i]][yl:yh, xl:xh] < 0, 0., datakeep['im'][ind[i]][yl:yh, xl:xh]).sum()
                            N = np.sqrt(np.where(datakeep['err'][ind[i]][yl:yh, xl:xh] < 0, 0.,
                                                 datakeep['err'][ind[i]][yl:yh, xl:xh] ** 2).sum())
                            if N != 0:
                                sn = S / N
                            else:
                                sn = 0.0


                    borplot.text(-0.265, .2, plot_label,
                            transform=imgplot.transAxes, fontsize=8, color=plot_label_color, #colors[i, 0:3],
                            verticalalignment='bottom', horizontalalignment='left')

                #if self.multiple_observations:
                #add the fiber info to the right of the images
                if not G.ZOO:
                    if is_a_fiber:
                        if self.panacea:
                            #dither and fiber position, etc generally meaningless in this case
                            #as there is no good way to immediately go back and find the source image
                            #so just show S/N and distance (and make bigger)
                            # if sn > 0:
                            #     if abs(sn) < 1000:
                            #         borplot.text(1.05, .75, 'SN: %0.2f' % (sn),
                            #                      transform=smplot.transAxes, fontsize=8, color='r',
                            #                      verticalalignment='bottom', horizontalalignment='left')
                            #     else:
                            #         borplot.text(1.05, .75, 'SN: %.1E' % (sn),
                            #                      transform=smplot.transAxes, fontsize=8, color='r',
                            #                      verticalalignment='bottom', horizontalalignment='left')

                            # distance (in arcsec) of fiber center from object center

                            borplot.text(1.05, .85, '%0.2f"' % (datakeep['d'][ind[i]]),
                                         transform=smplot.transAxes, fontsize=6, color='k',
                                         verticalalignment='bottom', horizontalalignment='left')

                            borplot.text(1.05, .65, '(%d, %d)' % (datakeep['ds9_x'][ind[i]], datakeep['ds9_y'][ind[i]]),
                                         transform=smplot.transAxes, fontsize=6, color='k',
                                         verticalalignment='bottom', horizontalalignment='left')

                            try:
                                #l3 = datakeep['date'][ind[i]] + "-" + datakeep['obsid'][ind[i]] + "-" + datakeep['expid'][ind[i]]

                                l3 = datakeep['date'][ind[i]] + "\nv" + str(datakeep['obsid'][ind[i]]).zfill(3) + "_" + \
                                     str(datakeep['expid'][ind[i]]).zfill(2)

                                #!!! multi*fits is <specid>_<ifuslot>_<ifuid> !!!
                                #!!! so do NOT change from spec_id
                                #!!! note: having all three identifiers makes the string too long so leave as is
                                l4 = datakeep['spec_id'][ind[i]] + "_" + datakeep['amp'][ind[i]] + "_" + \
                                     str(datakeep['fib_idx1'][ind[i]]).zfill(3) #+ "#" + str(datakeep['fib'][ind[i]]).zfill(3)

                                borplot.text(1.05, .25, l3,
                                             transform=smplot.transAxes, fontsize=6, color='k',
                                             verticalalignment='bottom', horizontalalignment='left')
                                borplot.text(1.05, .05, l4,
                                             transform=smplot.transAxes, fontsize=6, color='k',
                                             verticalalignment='bottom', horizontalalignment='left')
                            except:
                                log.error("Exception building extra fiber info.", exc_info=True)

                        else:
                            if sn > 0:
                                borplot.text(1.05, .75, 'S/N = %0.2f' % (sn),
                                            transform=smplot.transAxes, fontsize=6, color='r',
                                            verticalalignment='bottom', horizontalalignment='left')
                            #distance (in arcsec) of fiber center from object center
                            # borplot.text(1.05, .55, 'D(") = %0.2f %0.1f' % (datakeep['d'][ind[i]],datakeep['wscore'][ind[i]]),
                            #             transform=smplot.transAxes, fontsize=6, color='r',
                            #             verticalalignment='bottom', horizontalalignment='left')
                            borplot.text(1.05, .55, 'D(") = %0.2f' % (datakeep['d'][ind[i]]),
                                        transform=smplot.transAxes, fontsize=6, color='r',
                                        verticalalignment='bottom', horizontalalignment='left')
                            borplot.text(1.05, .35, 'X,Y = %d,%d' % (datakeep['xi'][ind[i]], datakeep['yi'][ind[i]]),
                                        transform=smplot.transAxes, fontsize=6, color='b',
                                        verticalalignment='bottom', horizontalalignment='left')
                            borplot.text(1.05, .15, 'D,S,F = %d,%s,%d' % (datakeep['dit'][ind[i]], datakeep['side'][ind[i]],
                                                                         datakeep['fib'][ind[i]]),
                                        transform=smplot.transAxes, fontsize=6, color='b',
                                        verticalalignment='bottom', horizontalalignment='left')
                    else:
                        borplot.text(1.05, .35, '\nWeighted\nSum',
                                     transform=smplot.transAxes, fontsize=8, color='k',
                                     verticalalignment='bottom', horizontalalignment='left')

                if grid_idx == (num_to_display-1): #(num + add_summed_image - 1):
                    smplot.text(0.5, 1.3, 'Smoothed',
                                transform=smplot.transAxes, fontsize=8, color='k',
                                verticalalignment='top', horizontalalignment='center')
                    pixplot.text(0.5, 1.3, 'Pixel Flat',
                                 transform=pixplot.transAxes, fontsize=8, color='k',
                                 verticalalignment='top', horizontalalignment='center')


                    if (detobj is not None) and (detobj.duplicate_fibers_removed != 0):
                        if detobj.duplicate_fibers_removed == -1:
                            warnmsg = "!!!Duplicate fibers\nNOT removed!!!"
                        elif detobj.duplicate_fibers_removed == 1:
                            warnmsg = "Duplicate fibers\nremoved"

                        pixplot.text(0.5, 0.75, warnmsg,
                                     transform=pixplot.transAxes, fontsize=8, color='k',
                                     verticalalignment='top', horizontalalignment='center')

                    imgplot.text(0.5, 1.3, '2D Spec',
                                 transform=imgplot.transAxes, fontsize=8, color='k',
                                 verticalalignment='top', horizontalalignment='center')



        #check for unique pixels (could be useful at some point in the future)
        try:
            if len(top_pixels) > 1:
                _,c = np.unique(top_pixels,return_counts=True)
                # save that info for the classifier
                detobj.num_duplicate_central_pixels = np.sum(c[np.where(c>1)])
        except:
            pass

        buf = io.BytesIO()
       # plt.tight_layout()#pad=0.1, w_pad=0.5, h_pad=1.0)
        plt.savefig(buf, format='png', dpi=300)

        if G.ZOO_CUTOUTS:
            try:
                e = datakeep['detobj']

                if e.pdf_name is not None:
                    fn = e.pdf_name.rstrip(".pdf") +  "_zoo_2d_fib.png"
                else:
                    fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_2d_fib.png"
                fn = op.join( e.outdir, fn)
                plt.savefig(fn,format="png",dpi=300)
            except:
                log.error("Unable to write zoo_2d_fib image to disk.",exc_info=True)

        plt.close(fig)
        return buf, Y

    # 2d spectra cutouts (one per fiber) 1st column only
    def build_2d_image_1st_column_only(self, datakeep):

        #always plot 5 (1 sum and 4 fibers ... if fibers are empty, plot as zero)

        # not dynamic, but if we are going to add a combined 2D spectra cutout at the top, set this to 1
        add_summed_image = 1  # note: adding w/cosmics masked out
        frac_y_separator = 0.2  # add a separator between the colored fibers and the black (summed) fiber at this
        # fraction of the cell height

        detobj = datakeep['detobj']

        cmap = plt.get_cmap('gray_r')

        colors = self.make_fiber_colors(min(4, len(datakeep['ra'])),
                                        len(datakeep['ra']))  # + 2 ) #the +2 is a pad in the call
        num_fibers = len(datakeep['xi'])

        num_to_display = 5 #min(MAX_2D_CUTOUTS, num_fibers) + add_summed_image  # for the summed images

        borderbuff = 0.01

        #just to keep the code similar to the build_2d_image from which it was cloned
        borderxl = 0.00#0.05
        borderxr = 0.00#0.15

        borderyb = 0.00 #0.05
        borderyt = 0.00 #0.15

        # the +1 for the summed image
        dx = 1.0
        dy = (0.96) / num_to_display  #- 5*borderbuff #- borderbuff #0.96 so have a little extra room to separate off the summed image
        dx1 = dx #1.0 - borderbuff
        dy1 = dy - borderbuff #dy - 2*borderbuff #(0.96 - (num_to_display) * borderbuff ) / (num_to_display)
        #dy1 = 1.0 / num_to_display
        #(1. - borderyb - borderyt - (num_to_display) * bordbuff) / (num_to_display)
        Y = 2.0

        fig = plt.figure(figsize=(0.9, Y), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        # previously sorted in order from largest distances to smallest
        ind = list(range(len(datakeep['d'])))

        # assume all the same shape
        summed_image = np.zeros(datakeep['im'][ind[0]].shape)

        # need i to start at zero
        # building from bottom up
        grid_idx = -1
        for i in range(num_fibers + add_summed_image):
            make_display = False
            if i < num_fibers:
                pcolor = colors[i]
                     # , 0:4] #keep the 4th value (alpha value) ... need that to lower the alpha of the greys
                datakeep['color'][ind[i]] = pcolor
                datakeep['index'][ind[i]] = num_fibers - i

                if i > (num_fibers - num_to_display):  # we are in reverse order (building the bottom cutout first)
                    make_display = True
                    grid_idx += 1
                    is_a_fiber = True
                    ext = list(np.hstack([datakeep['xl'][ind[i]], datakeep['xh'][ind[i]],
                                          datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))

                    # set the hot (cosmic) pixel values to zero then employ guassian_filter
                    a = datakeep['im'][ind[i]]
                    a = np.ma.masked_where(datakeep['err'][ind[i]] == -1, a)
                    a = np.ma.filled(a, 0.0)

                    summed_image += a * datakeep['fiber_weight'][ind[i]]

                    image = datakeep['im'][ind[i]]
                          # im can be the cosmic removed version, depends on G.PreferCosmicCleaned

                    cmap1 = cmap
                    cmap1.set_bad(color=[0.2, 1.0, 0.23])
                    image = np.ma.masked_where(datakeep['err'][ind[i]] == -1, image)
                    img_vmin = datakeep['vmin2'][ind[i]]
                    img_vmax = datakeep['vmax2'][ind[i]]

                    # plot_label = str(num_fibers-i)
                    # plot_label = str("%0.2f" % datakeep['fiber_weight'][ind[i]]).lstrip('0') #save space, kill leading 0

                    # the last one is the top one and is the primary
                    datakeep['primary_idx'] = ind[i]

            else:  # this is the top image (the sum)
                is_a_fiber = False
                make_display = True
                grid_idx += 1
                pcolor = 'k'
                ext = None

                image = summed_image
                img_vmin, img_vmax = self.get_vrange(summed_image, scale=contrast2)

            if make_display:

                if is_a_fiber:
                    y_separator = 0.0
                else:
                    y_separator = frac_y_separator * dy

                borplot = plt.axes([borderxl + 0. * dx, borderyb + grid_idx * dy + y_separator, dx, dy])

                imgplot = plt.axes(
                    [borderxl + 0. * dx + borderbuff / 2., borderyb + grid_idx * dy + borderbuff / 2. + y_separator,
                     dx1, dy1])

                autoAxis = borplot.axis()


                imgplot.imshow(image,
                               origin="lower", cmap=cmap1,
                               vmin=img_vmin,
                               vmax=img_vmax,
                               interpolation="none", extent=ext,zorder=1)

                imgplot.set_xticks([])
                imgplot.set_yticks([])
                imgplot.axis('off')

                rec = plt.Rectangle((autoAxis[0] + borderbuff / 2., autoAxis[2] + borderbuff / 2.),
                                    (autoAxis[1] - autoAxis[0]) * (1. - borderbuff),
                                    (autoAxis[3] - autoAxis[2]) * (1. - borderbuff), fill=False, lw=3,
                                    color=pcolor, zorder=9)
                rec = borplot.add_patch(rec)
                borplot.set_xticks([])
                borplot.set_yticks([])
                borplot.axis('off')

        buf = io.BytesIO()
        # plt.tight_layout()#pad=0.1, w_pad=0.5, h_pad=1.0)
        plt.savefig(buf, format='png', dpi=300,transparent=True)

        if G.ZOO_CUTOUTS:
            try:
                e = datakeep['detobj']

                if e.pdf_name is not None:
                    fn = e.pdf_name.rstrip(".pdf") + "_zoo_2d_fib_col1.png"
                else:
                    fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_2d_fib_col1.png"
                fn = op.join(e.outdir, fn)
                plt.savefig(fn, format="png", dpi=300,transparent=True)
            except:
                log.error("Unable to write zoo_2d_fib image to disk.", exc_info=True)

        plt.close(fig)
        return buf, Y
        #end build_2d_image_1st_column_only


    # +/- 3 fiber sizes on CCD (not spacially adjacent fibers)
    def build_scattered_light_image(self, datakeep, img_y = 3, key='scatter'):

            if not key in ['scatter','scatter_sky','scatter_lyc']:
                log.error("Invalid key for build_scattered_light_image: %s" % key)
                return None

            #cmap = plt.get_cmap('gray_r')
            norm = plt.Normalize()
            colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))
            num = len(datakeep[key])

            # which is largest SN (should be first, but confirm)
            ind = list(range(len(datakeep[key])))

            datakeep_idx = datakeep['primary_idx']

            if datakeep_idx is None:
                datakeep_idx = 0
                for i in range(num):
                    if datakeep['fiber_sn'][ind[i]] > datakeep['fiber_sn'][ind[datakeep_idx]]:
                        datakeep_idx = i
                datakeep['color'][datakeep_idx] = colors[datakeep_idx, 0:3]

            datakeep['primary_idx'] = datakeep_idx

            if key == 'scatter':
                cmap = plt.get_cmap('gray_r')
                vmin = datakeep['vmin2'][ind[datakeep_idx]]
                vmax = datakeep['vmax2'][ind[datakeep_idx]]
                title = "Clean Image"

            elif key == 'scatter_sky':
                cmap = plt.get_cmap('gray_r')
                vmin = datakeep['vmin3'][ind[datakeep_idx]]
                vmax = datakeep['vmax3'][ind[datakeep_idx]]
                title = "With Sky"
            elif key == 'scatter_lyc':
                cmap = plt.get_cmap('gray_r')
                vmin = datakeep['vmin2'][ind[datakeep_idx]]
                vmax = datakeep['vmax2'][ind[datakeep_idx]]
                title = "Cleaned (LyC)"
            else: #not possible ... just here for sanity and future expansion
                log.error("Invalid key for build_scattered_light_image: %s" % key)
                return None

            # bordbuff = 0.01
            # borderxl = 0.05
            # borderxr = 0.15
            # borderyb = 0.05
            # borderyt = 0.15

            bordbuff = 0.01
            borderxl = 0.00
            borderxr = 0.00
            borderyb = 0.00
            borderyt = 0.15

            dx = (1. - borderxl - borderxr)
            dy = (1. - borderyb - borderyt)

            #5/3. is to keep the scale (width) same as the 2D cutouts next to this plot
            img_y = max(img_y,3) #set a minimum size (height)
            fig = plt.figure(figsize=(5/3., img_y), frameon=False)
            plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)


            imgplot = plt.axes([0. + bordbuff + borderxl,0 + bordbuff + borderyb, dx, dy])
            autoAxis = imgplot.axis()
            imgplot.set_xticks([])
            imgplot.set_yticks([])

            if not G.ZOO:
                #plt.title("CCD Region of Main Fiber\n(%d,%d)"
                if key == 'scatter_lyc':
                    plt.title("%s\nx, y: %d, %d"
                              % (title, datakeep['ds9_x_lyc'][ind[datakeep_idx]], datakeep['ds9_y_lyc'][ind[datakeep_idx]]),
                              fontsize=12)
                else:
                    plt.title("%s\nx, y: %d, %d"
                          % (title,datakeep['ds9_x'][ind[datakeep_idx]], datakeep['ds9_y'][ind[datakeep_idx]]),
                              fontsize=12)
            else:
                plt.title("%s" % (title), fontsize=12)

            #combined_image = np.concatenate((datakeep['scatter_sky'][ind[datakeep_idx]],
            #                                 datakeep['scatter'][ind[datakeep_idx]]),axis=1)

            imgplot.imshow(datakeep[key][ind[datakeep_idx]],
            #imgplot.imshow(combined_image,
                           origin="lower", cmap=cmap,
                           vmin=vmin,
                           vmax=vmax,
                           interpolation="none")  # , extent=ext)


            borplot = plt.axes([0. + borderxl,0 + borderyb, dx + bordbuff, dy + bordbuff])
            #autoAxis = borplot.axis()
            borplot.set_xticks([])
            borplot.set_yticks([])
            borplot.axis('off')

            h,w = datakeep[key][ind[datakeep_idx]].shape
            rec = plt.Rectangle([0, 0],
                                (w-1),
                                (h-1), fill=False, lw=3,
                                color=datakeep['color'][datakeep_idx], zorder=9)

            rec = imgplot.add_patch(rec)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)

            if G.ZOO_CUTOUTS:
                try:
                    e = datakeep['detobj']
                    zoo_num = "ccd_sub"
                    if key == "scatter_sky":
                        zoo_num = "ccd_sky"

                    if e.pdf_name is not None:
                        fn = e.pdf_name.rstrip(".pdf") + "_zoo_" + zoo_num \
                         + ".png"
                    else:
                        fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_" + zoo_num \
                         + ".png"
                    fn = op.join(e.outdir, fn)
                    plt.savefig(fn, format="png", dpi=300)
                except:
                    log.error("Unable to write zoo_%s image to disk." %(zoo_num), exc_info=True)

            plt.close(fig)
            return buf



    def build_spec_image_fit_gaussian(self,datakeep,cwave,dwave=1.0,unlabeled=False):

        #scatter plot (with errors) the data points within the zoom-in region
        #and overplot the Gaussian fit from the 4 parameters

        try:
            parms = datakeep['detobj'].line_gaussfit_parms
            flux = datakeep['sumspec_flux_zoom']
            flux_err = datakeep['sumspec_ferr_zoom']
            wave_data = datakeep['sumspec_wave_zoom']
        except:
            log.warning("Warning. No zoom-in Gaussian fit parameters.")
            return None

        fig = plt.figure(figsize=(5, 3))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        #plt.title("Fitted Emission")

        rm = 0.2
        specplot = plt.axes([0.1, 0.1, 0.8, 0.8])
        wave_grid = np.arange(cwave - ww, cwave + ww + dwave, 0.1)
        #wave_grid = np.arange(cwave - ww, cwave + ww + dwave + 2.0, 2.0)  # 0.1)
        fit_spec = gaussian(x=wave_grid,x0=parms[0],sigma=parms[1],a=parms[2],y=parms[3])

        # mn = min(mn, min(summed_spec))
        # mx = max(mx, max(summed_spec))
        # ran = mx - mn
        #
        # if mn > 0:  # next 10 below the mn
        #     min_y = int(mn) / 10 * 10 - 10
        # else:
        #     min_y = int(mn) / 10 * 10


        #get ymin, ymax ... should completely cover the error bars (from the central half of the plot)
        #don't worry about possibly wild data points outside the central region

        #!!! NOTE: Karl's data is evaluated in the center of each wavelength bin and has its value
        #set s|t the area is correct (that is, s|t it is a flux (erg s^-1 cm^-2) NOT a flux density (
        # (erg s^-1 cm^-2 bin^-1)
        #essentially then, what I really need to do is evaluate the Gaussian at the center of each bin
        # s|t that my bins are the same as Karl's (or smaller is okay) and then multiply by the bin width ratio
        # ... e.g. if Karl is on 2AA bins and I am using 0.1AA steps. multiply the Gaussian value by 0.1/2. = 1/20
        # NEED to think about that some more ... need the area under the Gaussian to match over the same width
        # in wavelength

        try:
            width = len(flux)
            left = int(width/2) - int(width/4)
            right = int(width/2) + int(width/4) + 1
            y_min = np.min(flux[left:right] - flux_err[left:right])
            y_max = np.max(flux[left:right] + flux_err[left:right])

            #for adding the zero line for reference
            if y_min > 0:
                y_min = 0
            elif y_max < 0: #really bad
                y_max = 0

            y_bump = (y_max - y_min)*0.05 #5% of the range
            specplot.set_ylim(bottom=y_min-y_bump, top=y_max+y_bump)
        except:
            log.debug("Could not set model spectra fit scale to data.", exc_info=True)

        specplot.plot(wave_grid, fit_spec, c='k', lw=2, linestyle="solid", alpha=0.7, zorder=0)
        specplot.errorbar(wave_data,flux,yerr=flux_err,fmt='.',zorder=9)
        #add the zero line
        specplot.axhline(y=0,linestyle='solid',alpha=0.5,color='k',zorder=9)
        if unlabeled:
            specplot.set_yticklabels([])
            specplot.set_xticklabels([])
        else:
            specplot.text(0.075, 0.95, r"e$^{-17}$x2$\AA$", horizontalalignment='center',
                          verticalalignment='center', transform=specplot.transAxes)

        # log.debug("Spec Plot max count = %f , min count = %f" % (mx, mx))
        # specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], ls='--', c=[0.3, 0.3, 0.3])
        # specplot.axis([cwave - ww, cwave + ww, min_y, mx + ran / 20])


        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        if G.ZOO_CUTOUTS:
            try:
                e = datakeep['detobj']
                if e.pdf_name is not None:
                    fn = e.pdf_name.rstrip(".pdf") + "_zoo_1d_sum.png"
                else:
                    fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_1d_sum.png"
                fn = op.join(e.outdir, fn)
                plt.savefig(fn, format="png", dpi=300,transparent=True)
            except:
                log.error("Unable to write zoo_1d_sum image to disk.", exc_info=True)

        plt.close(fig)
        return buf

    #upper right panel (zoomed in spectrum image)
    def build_spec_image(self,datakeep,cwave, dwave=1.0,unlabeled=False):

        try:
            result = None
            if datakeep['detobj'].line_gaussfit_parms is not None:
                result =  self.build_spec_image_fit_gaussian(datakeep,cwave,dwave=1.0,unlabeled=unlabeled)

            #if the Gaussian fit plot worked, use it ... otherwise make the old plot
            if result is not None:
                return result
            else:
                log.warning("Call failed to HETDEX::build_spec_image_fit_gaussian. Attempting old style.")
        except:
            log.warning("Call failed to HETDEX::build_spec_image_fit_gaussian. Attempting old style.")

        #cwave = central wavelength, dwave = delta wave (wavestep in AA)

        fig = plt.figure(figsize=(5, 3))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        #plt.gca().set_title("Fitted Emission")
        #norm = plt.Normalize()
        #colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))
        colors = self.make_fiber_colors(min(4, len(datakeep['ra'])),
                                        len(datakeep['ra']))  # + 2 ) #the +2 is a pad in the call

        rm = 0.2
        specplot = plt.axes([0.1, 0.1, 0.8, 0.8])
        bigwave = np.arange(cwave - ww, cwave + ww + dwave, dwave)
        summed_spec = np.zeros(bigwave.shape)
        mn = 100.0
        mx = 0.0

        #these are plotted from bottom to top
        #we want, then, the LAST (plot_fibers) to be ploted

        #want ALL ind even if only goint to plot spectra for a subset
        ind = range(len(datakeep['d']))

        #previously sorted in order from largest distances to smallest
        N = len(datakeep['xi'])
        if self.plot_fibers is not None:
            stop = max(N - self.plot_fibers-1,-1)
        else:
            if self.panacea:
                stop = max(N -4 -1,-1) #plot the top 4 if not specified
            else:
                stop = -1 #plot them all

        if PLOT_SUMMED_SPECTRA:
            alpha = 0.7 #for individual fibers
            linewidth = 1.25 #for individual fibers
        else:
            alpha = 1.0 #for individual fibers
            linewidth = 2.0 #for individual fibers


        key_wave  = 'specwave'
        key_data = 'spec'

        if (datakeep['fluxcal_wave'][N-1] is not None) and (len(datakeep['fluxcal_wave'][N-1]) > 0):
            key_wave = 'fluxcal_wave'
            key_data = 'fluxcal_cnts'


        # # todo: set the weights correctly
        # datakeep['fiber_weight'].append(fiber.relative_weight)
        # datakeep['thruput'].append(fiber.fluxcal_central_emis_thru)
        #
        # datakeep['fluxcal_wave'].append(fiber.fluxcal_central_emis_wavelengths)
        # datakeep['fluxcal_cnts'].append(fiber.fluxcal_central_emis_counts)
        # datakeep['fluxcal_flux'].append(fiber.fluxcal_central_emis_flux)
        # datakeep['fluxcal_cont'].append(fiber.fluxcal_emis_cont)

        #print("temporary set weight == 1.0")
        #for i in range(N-1,-1,-1):
        #    datakeep['fiber_weight'][ind[i]] = 1.0

        try:
            for i in range(N-1,stop,-1): #stop is not included
                #regardless of the number if the sn is below the threshold, skip it
                if (datakeep['fiber_sn'][i] is not None) and (datakeep['fiber_sn'][i] < self.min_fiber_sn):
                    continue

                if datakeep['color'][i] is None:
                    datakeep['color'][i] = colors[i]  # , 0:3]


                #todo: fluxcal_wave
                #specwave is directly from panacea and is not calibrated

                specplot.step(datakeep[key_wave][ind[i]], datakeep[key_data][ind[i]],linestyle="solid",
                              where='mid', color=datakeep['color'][i], alpha=alpha,linewidth=linewidth,zorder=i)

                if datakeep['fiber_weight'][ind[i]] > 0.1:
                    summed_spec += (np.interp(bigwave, datakeep[key_wave][ind[i]],
                                          datakeep[key_data][ind[i]]) * datakeep['fiber_weight'][ind[i]])

                #this is for plotting purposes, so don't need them in the next loop (not plotted)
                mn = np.min([mn, np.min(datakeep[key_data][ind[i]])])
                mx = np.max([mx, np.max(datakeep[key_data][ind[i]])])

            #now for the rest
            for i in range(stop,-1,-1):
                if datakeep['fiber_weight'][ind[i]] > 0.1:
                    summed_spec += (np.interp(bigwave, datakeep[key_wave][ind[i]],
                                          datakeep[key_data][ind[i]]) * datakeep['fiber_weight'][ind[i]])

               #summed_spec += (np.interp(bigwave, datakeep['specwave'][ind[i]],
               #                           datakeep['spec'][ind[i]] / datakeep['thruput'][ind[i]] *
                #                          datakeep['fiber_weight'][ind[i]]))

            if PLOT_SUMMED_SPECTRA:

                #use Karl's externally summed spectra if available
                if (datakeep['sumspec_wave'] is not None) and (len(datakeep['sumspec_wave']) > 0):
                    summed_spec = np.interp(bigwave, datakeep['sumspec_wave'],datakeep['sumspec_cnts'])

                #specplot.step(bigwave, summed_spec, c='k', where='mid', lw=2,linestyle="solid",alpha=1.0,zorder=0)
                specplot.plot(bigwave, summed_spec, c='k', lw=2, linestyle="solid", alpha=1.0, zorder=0)
                mn = min(mn, min(summed_spec))
                mx = max(mx, max(summed_spec))

            ran = mx - mn

            if mn > 0: #next 10 below the mn
                min_y = int(mn)/10*10-10
            else:
                min_y = int(mn)/10*10

            #per Karl 20181012 no longer need this restriction
            #min_y = max(min_y, -0.25*mx)#-20) #but not below -20

            log.debug("Spec Plot max count = %f , min count = %f" %(mx,mx))
            specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], ls='--', c=[0.3, 0.3, 0.3])
            specplot.axis([cwave - ww, cwave + ww, min_y, mx + ran / 20])

            if unlabeled:
                specplot.axis('off')

        except:
            log.warning("Unable to build cutout spec plot. Datakeep info:\n"
                        "IFUSLOTID = %s\n"
                        "Dither = %s\n"
                        "SIDE = %s\n"
                        "AMP = %s\n"
                        "Fiber = %d\n"
                        "Wavelength = %f\n"
                        "i = %d\n"
                        "ind[i] = %d\n"
                        "len(ind) = %d\n"
                        "len(dict) = %d\n"
                        "len(specwave) = %d\n"
                        "len(spec) = %d\n"
                        % (self.ifu_slot_id,datakeep['dit'][ind[i]],datakeep['side'][ind[i]],datakeep['amp'][ind[i]],
                           datakeep['fib'][ind[i]],cwave,i,ind[i],len(ind),len(datakeep['spec']),
                           len(datakeep['specwave'][ind[i]]), len(datakeep['spec'][ind[i]]))
                        , exc_info=True)


        #turn off the errorbar for now
       # try:
       #    # specplot.errorbar(cwave - .8 * ww, mn + ran * (1 + rm) * 0.85,
       #     specplot.errorbar(cwave - .8 * ww, max(F),
       #                       yerr=biweight_midvariance(np.array(datakeep['spec'][:])),
       #                       fmt='o', marker='o', ms=4, mec='k', ecolor=[0.7, 0.7, 0.7],
       #                       mew=1, elinewidth=3, mfc=[0.7, 0.7, 0.7])
       # except:
       #     log.error("Error building spectra plot error bar", exc_info=True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300,transparent=True)

        if G.ZOO_CUTOUTS:
            try:
                e = datakeep['detobj']
                if e.pdf_name is not None:
                    fn = e.pdf_name.rstrip(".pdf")  + "_zoo_1d_sum.png"
                else:
                    fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_1d_sum.png"
                fn = op.join(e.outdir, fn)
                plt.savefig(fn, format="png", dpi=300,transparent=True)
            except:
                log.error("Unable to write zoo_1d_sum image to disk.", exc_info=True)

        plt.close(fig)
        return buf

    # def build_relative_fiber_locs(self, e):
    #     #defunct
    #
    #     fig = plt.figure(figsize=(5, 3))
    #     #plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    #     fibplot = plt.axes()#[0.1, 0.2, 0.8, 0.8])
    #
    #     fibplot.set_title("Relative Fiber Positions")
    #     #fibplot.set_xlabel("arcsecs")
    #     #plt.gca().xaxis.labelpad =
    #
    #     fibplot.plot(0, 0, "r+")
    #
    #     xmin = float('inf')
    #     xmax = float('-inf')
    #     ymin = float('inf')
    #     ymax = float('-inf')
    #
    #     if e.wra:
    #         e_ra = e.wra
    #         e_dec = e.wdec
    #     else:
    #         e_ra = e.ra
    #         e_dec = e.dec
    #
    #     for r, d, c, i, dist, fn in e.fiber_locs:
    #         # fiber absolute position ... need relative position to plot (so fiber - zero pos)
    #         fx = (r - e_ra) * np.cos(np.deg2rad(e_dec)) * 3600.
    #         fy = (d - e_dec) * 3600.
    #
    #         xmin = min(xmin, fx)
    #         xmax = max(xmax, fx)
    #         ymin = min(ymin, fy)
    #         ymax = max(ymax, fy)
    #
    #         fibplot.add_patch(plt.Circle((fx,fy), radius=G.Fiber_Radius, color=c, fill=False,
    #                                        linestyle='solid',zorder=9))
    #         fibplot.text(fx,fy, str(i), ha='center', va='center', fontsize='x-small', color=c)
    #
    #         if fn in G.CCD_EDGE_FIBERS_ALL:
    #             fibplot.add_patch(
    #                 plt.Circle((fx, fy), radius=G.Fiber_Radius + 0.1, color=c, fill=False,
    #                            linestyle='dashed',zorder=9))
    #
    #     # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
    #     ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    #     ext = ext_base + 2*G.Fiber_Radius
    #
    #     rec = plt.Rectangle((-ext,-ext),width=ext*2, height=ext * 2, fill=True, lw=1,
    #                         color='gray', zorder=0, alpha=0.5)
    #     fibplot.add_patch(rec)
    #
    #     fibplot.set_xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
    #     fibplot.set_yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
    #     fibplot.set_aspect('equal')
    #
    #     fig.tight_layout()
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png', dpi=300)
    #
    #     plt.close(fig)
    #     return buf


    #wide (full width) spectrum
    def build_full_width_spectrum (self, datakeep, cwave):

        noise_multiplier = 5.0

        cmap = plt.get_cmap('gray_r')
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))

        if G.SHOW_ALL_1D_SPECTRA:
            num = len(datakeep['xi'])
        else:
            num = 0
        dy = 1.0/(num +5)  #+ 1 skip for legend, + 2 for double height spectra + 2 for double height labels

        if G.SINGLE_PAGE_PER_DETECT:
            if G.SHOW_ALL_1D_SPECTRA:
                figure_sz_y = 2* G.GRID_SZ_Y
            else:
                figure_sz_y = G.GRID_SZ_Y
        else:
            if G.SHOW_ALL_1D_SPECTRA:
                figure_sz_y = 0.6 * 3 * G.GRID_SZ_Y
            else:
                figure_sz_y = 0.25 * 3 * G.GRID_SZ_Y

        #fig = plt.figure(figsize=(5, 6.25), frameon=False)
        fig = plt.figure(figsize=(G.FIGURE_SZ_X, figure_sz_y),frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        ind = list(range(len(datakeep['d'])))

        border_buffer = 0.025 #percent from left and right edges to leave room for the axis labels
        #fits cutouts (fibers)
        for i in range(num):
            #show all these (only skip when summing the spectra later below)
            #skip if below threshold
            #if (datakeep['fiber_sn'][i] is not None) and (datakeep['fiber_sn'][i] < self.min_fiber_sn):
            #    continue

           # borplot = plt.axes([0, i * dy, 1.0, dy*0.75])
           # imgplot = plt.axes([border_buffer, i * dy - 0.125*dy, 1-(2*border_buffer), dy])
            borplot = plt.axes([0, i * dy, 1.0, dy])
            imgplot = plt.axes([border_buffer, i * dy, 1-(2*border_buffer), dy])
            autoAxis = borplot.axis()

            if datakeep['color'][i] is None:
                datakeep['color'][i] = colors[i]#, 0:3]
                datakeep['index'][i] = num - i


            rec = plt.Rectangle((autoAxis[0] , autoAxis[2]),
                                (autoAxis[1] - autoAxis[0]) ,
                                (autoAxis[3] - autoAxis[2]) , fill=False, lw=3,
                                color=datakeep['color'][i], zorder=2)
            borplot.add_patch(rec)
            borplot.set_xticks([])
            borplot.set_yticks([])
            borplot.axis('off')

            ext = list(np.hstack([datakeep['fxl'][ind[i]], datakeep['fxh'][ind[i]],
                                  datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))

            a = datakeep['fw_im'][ind[i]]
            #a = gaussian_filter(datakeep['fw_im'][ind[i]], (2, 1))

            imgplot.imshow(a,
                           origin="lower", cmap=cmap,
                           interpolation="none",
                           vmin=datakeep['vmin2'][ind[i]],
                           vmax=datakeep['vmax2'][ind[i]],
                           extent=ext,zorder=1)

            imgplot.set_xticks([])
            imgplot.set_yticks([])
            imgplot.axis(ext)
            imgplot.axis('off')


            imgplot.text(-0.8*border_buffer, .5, num - i,
                         transform=imgplot.transAxes, fontsize=10, color='k',  # colors[i, 0:3],
                         verticalalignment='bottom', horizontalalignment='left')

        # this is the 1D averaged spectrum
        specplot = plt.axes([border_buffer, float(num + 1.0) * dy, 1.0 - (2 * border_buffer), dy*2])
        rm = 0.2

        #they should all be the same length
        #yes, want round and int ... so we get nearest pixel inside the range)
        left = round(min(datakeep['fw_specwave'][0]))
        right = int(max(datakeep['fw_specwave'][0]))

        bigwave = np.arange(left, right)

        N = len(datakeep['fib'])
        if self.plot_fibers is not None:
            stop = max(N - self.plot_fibers - 1, -1)
        else:
            stop = -1

        y_label = "counts"
      #  min_y = -20
        try:
            j = None
            if len(datakeep['sumspec_wave']) > 0:
                F = np.interp(bigwave, datakeep['sumspec_wave'], datakeep['sumspec_flux'])
                #E = np.interp(bigwave, datakeep['sumspec_wave'], datakeep['sumspec_ferr'])

                try:
                    E = np.interp(bigwave,G.CALFIB_WAVEGRID,datakeep['detobj'].calfib_noise_estimate)
                except:
                    E = np.zeros(len(bigwave))
                y_label = "cgs" #r"cgs [$10^{-17}$]"
 #               min_y = -2
            else:
                F = np.zeros(bigwave.shape)
                #E = np.interp(bigwave, datakeep['sumspec_wave'], datakeep['sumspec_ferr'])
                try:
                    E = np.interp(bigwave, G.CALFIB_WAVEGRID, datakeep['detobj'].calfib_noise_estimate)
                except:
                    E = np.zeros(len(bigwave))
                #new way, per Karl, straight sum
                for j in range(N - 1, stop, -1):
                    # regardless of the number if the sn is below the threshold, skip it
                    if (datakeep['fiber_sn'][j] is not None) and (datakeep['fiber_sn'][j] < self.min_fiber_sn):
                        continue

                    F += (np.interp(bigwave, datakeep['fw_specwave'][ind[j]],
                                    datakeep['fw_spec'][ind[j]] * datakeep['fiber_weight'][ind[j]]) )




            #peak_height = near get the approximate peak height
            mn = np.min(F)
            mx = np.max(F)
            absorber = False
            try:
                absorber = datakeep['detobj'].spec_obj.absorber
            except:
                pass
            try:

                if absorber or (cwave is None or cwave == 0): # G.CONTINUUM_RULES:
                    mn = min(F)
                    mx = max(F)
                else:
                    peak_idx = (np.abs(datakeep['sumspec_wave'] - cwave)).argmin()

                    idx_left = max(0,peak_idx-2)
                    idx_right = min(len(datakeep['sumspec_wave']),peak_idx+3)
                    peak_height = max(datakeep['sumspec_flux'][idx_left:idx_right])

                    mn = max(mn, -0.2 * peak_height)
                    mx = min(mx, 2.0 * peak_height)  # at most go 100% above the peak
            except:
                pass


            if mn > mx: #handle purely negative case
                tmp = mn
                mn = mx
                mx = tmp

            #flux nearest at the cwave position
            #this is redundant to the block immediately above
            # if True:
            #     #this is the (binned) flux at position nearest the central wave peak (cwave)
            #     line_mx = F[(np.abs(bigwave-cwave)).argmin()]
            #     if mx > 3.0*line_mx: #limit mx to a more narrow range)
            #         mx = max(F[(np.abs(bigwave-3500.0)).argmin():(np.abs(bigwave-5500.0)).argmin()])
            #         if mx > 3.0*line_mx:
            #             log.info("Truncating max spectra value...")
            #             mx = 3.0 * line_mx
            #         else:
            #             log.info("Excluding spectra maximum outside 3500 - 5500 AA")

            #override the auto set limits
            if self.ylim is not None:
                log.debug(f"Override the auto y-axis limits {mn},{mx} to {self.ylim[0]},{self.ylim[1]}")
                mn = self.ylim[0]
                mx = self.ylim[1]

            ran = mx - mn

            #plot the Error (fill between) light grey
            #specplot.fill_between(bigwave, 5.0 * E, -5.0 * E, facecolor='gray', alpha=0.4, zorder=4)
            specplot.fill_between(bigwave,noise_multiplier*E,-noise_multiplier*E,facecolor='gray',alpha=0.5,zorder=5)

            #red tips on peaks above 3sigma
            where_red = np.where( (F-noise_multiplier*E) > 0.0)
            mask = np.full(np.shape(bigwave),False)
            mask[where_red]=True

            #specplot.step(bigwave, F, c='b', where='mid', lw=1)
            specplot.plot(bigwave, F, c='b', lw=1,zorder=8)
            if G.SHADE_1D_SPEC_PEAKS:
                specplot.fill_between(bigwave, noise_multiplier * E,F,where=mask,facecolor='r',edgecolor='r', alpha=1.0, zorder=9)
            specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], lw=0.75,ls='dashed', c='k',zorder=9) #[0.3, 0.3, 0.3])


            #add the zero reference line
            specplot.axhline(y=0,color='k',alpha=0.5)

            if G.FIT_FULL_SPEC_IN_WINDOW:
                specplot.axis([left, right,np.min(F),np.max(F)])
            else:
                specplot.axis([left, right, mn - ran / 20, mx + ran / 20])
            # specplot.set_ylabel(y_label) #not honoring it, so just put in the text plot

            specplot.locator_params(axis='y',tight=True,nbins=4)#,y_label='cgs') #y_label deprecated?

            textplot = plt.axes([border_buffer, (float(num)+3) * dy, 1.0 - (2 * border_buffer), dy*2 ])
            textplot.set_xticks([])
            textplot.set_yticks([])
            textplot.axis(specplot.axis())
            textplot.axis('off')

            #if this is flux, not counts, add a ersatz scale label for y axis
            if len( datakeep['sumspec_wave']) > 0:
                textplot.text(3500, textplot.axis()[2], r"e$^{-17}$x2$\AA$", rotation=0, ha='left', va='bottom',
                          fontsize=10, color='k')  # use the e color for this family


            #
            #possibly add the matched line at cwave position
            #

            matched_line_list = [] #use farther down to display line labels otherwise marked as not to be displayed

            the_solution_rest_wave = -1.0
            if not G.ZOO:
                good, scale_score,p_score = datakeep['detobj'].multiline_solution_score()
                if (scale_score > G.MULTILINE_MIN_WEAK_SOLUTION_CONFIDENCE):
                    #a solution
                    sol = datakeep['detobj'].spec_obj.solutions[0]
                    absorber = datakeep['detobj'].spec_obj.solutions[0].emission_line.absorber
                    the_solution_rest_wave = sol.central_rest
                    y_pos = textplot.axis()[2]

                    if good and not absorber and (p_score > 0.7):
                        textplot.text(cwave, y_pos, sol.name + " {", rotation=-90, ha='center', va='bottom',
                                      fontsize=24, color=sol.color)  # use the e color for this family
                    else: #weak solution, use standard font size
                        textplot.text(cwave, y_pos, sol.name + " {", rotation=-90, ha='center', va='bottom',
                                      color=sol.color,fontsize=10)  # use the e color for this family

                    #highlight the matched lines

                    yl, yh = specplot.get_ylim()

                    #adjust if necessary for zero reference line
                    if yl > 0:
                        yl = 0
                    elif yh < 0: #this would be bad
                        yh = 0

                    for f in sol.lines:
                        matched_line_list.append(f.w_rest)
                        try:
                            if f.sigma is not None:
                                hw = 3.0 * f.sigma #highlight half-width
                            else:
                                hw = 6.0 #fixed value
                            # use 'y' rather than sols[0].color ... becomes confusing with black
                            rec = plt.Rectangle((f.w_obs - hw, yl), 2 * hw, yh - yl, fill=True, lw=1,
                                                color=sol.color, alpha=0.5,zorder=1)
                            specplot.add_patch(rec)
                        except:
                            pass

                    #don't color, but still mark rejected lines
                    for f in sol.rejected_lines:
                        matched_line_list.append(f.w_rest)
                #else: #redundant log ... already logged when preparing the upper left text block
                #    log.info("No singular, strong emission line solution.")


                #put dashed line through all possible emission lines (note: possible, not necessarily probable)
                if (datakeep['detobj'].spec_obj.all_found_lines is not None):
                    for f in datakeep['detobj'].spec_obj.all_found_lines: #this is an EmisssionLineInfo object

                        log.info(
                            "eid(%s) emission line at %01.f snr = %0.1f  line_flux = %0.1g  sigma = %0.1f  "
                            "line_score = %0.1f  p(noise) = %g  threshold = %g"
                            % (
                                str(datakeep['detobj'].entry_id), f.fit_x0, f.snr, f.line_flux, f.fit_sigma, f.line_score,
                                f.prob_noise,G.MULTILINE_MAX_PROB_NOISE_TO_PLOT))

                        if f.prob_noise < G.MULTILINE_MAX_PROB_NOISE_TO_PLOT:
                            x_pos = f.raw_x0
                            #y_pos = f.raw_h / 10.0 # per definition of EmissionLineInfo ... these are in 10^-18 cgs
                                                   # and these plots are 10^-17 cgs
                            #specplot.scatter(x_pos, y_pos, facecolors='None', edgecolors='b', zorder=99)
                            specplot.plot([x_pos, x_pos], [mn - ran * rm, mn + ran * (1 + rm)], ls='dashed', c='k',
                                          zorder=1,alpha=0.5)
                            #DEBUG:
                            if G.DEBUG_SHOW_GAUSS_PLOTS:
                                print("Line: %f (%f) " %(f.raw_x0,f.fit_x0))

                # put dashed line through all possible ABSORPTION lines (note: possible, not necessarily probable)
                if (datakeep['detobj'].spec_obj.all_found_absorbs is not None):
                    for f in datakeep['detobj'].spec_obj.all_found_absorbs:
                        log.info(
                            "eid(%s) absorption line at %01.f snr = %0.1f  line_flux = %0.1g  sigma = %0.1f  "
                            "line_score = %0.1f  p(noise) = %g  threshold = %g"
                            % (
                                str(datakeep['detobj'].entry_id), f.fit_x0, f.snr, f.line_flux, f.fit_sigma, f.line_score,
                                f.prob_noise, G.MULTILINE_MAX_PROB_NOISE_TO_PLOT))

                        if f.prob_noise < G.MULTILINE_MAX_PROB_NOISE_TO_PLOT:
                            x_pos = f.raw_x0
                            # y_pos = f.raw_h / 10.0 # per definition of EmissionLineInfo ... these are in 10^-18 cgs
                            # and these plots are 10^-17 cgs
                            # specplot.scatter(x_pos, y_pos, facecolors='None', edgecolors='b', zorder=99)
                            specplot.plot([x_pos, x_pos], [mn - ran * rm, mn + ran * (1 + rm)], ls='dashed', c='k',
                                          zorder=1, alpha=0.5)
                            # DEBUG:
                            if G.DEBUG_SHOW_GAUSS_PLOTS:
                                print("Absorb: %f (%f) " % (f.raw_x0, f.fit_x0))

            #
            #iterate over all (database of) emission lines ... assume the cwave is that line and plot the additional lines
            #
            wavemin = specplot.axis()[0]
            wavemax = specplot.axis()[1]
            legend = []
            name_waves = []
            obs_waves = []

            try:
                #use THIS detection's spectrum object's modified emission_lines list IF there is one,
                #otherwise, use the default list
                emission_line_list = datakeep['detobj'].spec_obj.emission_lines
            except:
                emission_line_list = self.emission_lines


            #found absorbers ... don't print weak emission lines if there is an absorber found there
            absorber_waves = []
            if datakeep['detobj'].spec_obj.all_found_absorbs is not None:
                absorber_waves = [x.fit_x0 for x in datakeep['detobj'].spec_obj.all_found_absorbs]

            for e in emission_line_list:
                if self.known_z is None:
                    if (not e.solution) and (e.w_rest != the_solution_rest_wave): #if not a normal solution BUT it is THE solution, label it
                        continue
                    z = cwave / e.w_rest - 1.0
                else:
                    sol_z = cwave / e.w_rest - 1.0
                    z = self.known_z
                    if abs(sol_z - z) > 0.05:
                        continue

                if (z < 0):
                    if z > G.NEGATIVE_Z_ERROR:
                        z = 0.0
                    else:
                        continue
                count = 0
                for f in emission_line_list:
                    if (f == e) or not (wavemin <= f.redshift(z) <= wavemax) or (abs(f.redshift(z) - cwave) < 5.0):
                        continue
                    # elif f.see_in_absorption: # we are not showing absorbers unless they are found
                    #     continue
                    elif G.DISPLAY_ABSORPTION_LINES and datakeep['detobj'].spec_obj.is_near_absorber(f.w_obs):
                        pass #print this one
                    elif datakeep['detobj'].spec_obj.is_near_a_peak(f.w_obs):
                        pass #print this one
                    elif ((f.display == False) and (not (f.w_rest in matched_line_list))):
                        continue

                    #don't display absorbers unless they are found in the blind line finder scan
                    if f.see_in_absorption and np.any([abs(f.w_obs-x) < 10 for x in absorber_waves]):
                        pass #this is an absorber that matches to a found absorber (print)
                    elif (f.rank > 3) and np.any([abs(f.w_obs-x) < 10 for x in absorber_waves]):
                        continue #this is a weak emitter that is near a found absorber (do not print)

                    count += 1
                    y_pos = textplot.axis()[2]
                    for w in obs_waves:
                        if abs(f.w_obs - w) < 10: # too close, shift one vertically
                            #y_pos = textplot.axis()[2] + mn + ran*0.7
                            y_pos = (textplot.axis()[3] - textplot.axis()[2]) / 2.0 + textplot.axis()[2]
                            break

                    obs_waves.append(f.w_obs)
                    textplot.text(f.w_obs, y_pos, f.name+" {", rotation=-90, ha='center', va='bottom',
                                      fontsize=10, color=e.color)  # use the e color for this family

                if (count > 0) and not (e.name in name_waves):
                    legend.append(mpatches.Patch(color=e.color,label=e.name))
                    name_waves.append(e.name)

            #make a legend ... this won't work as is ... need multiple colors
            skipplot = plt.axes([border_buffer, (float(num)) * dy, 1.0 - (2 * border_buffer), dy])
            skipplot.set_xticks([])
            skipplot.set_yticks([])
            skipplot.axis(specplot.axis())
            skipplot.axis('off')
            skipplot.legend(handles=legend, loc = 'center',ncol=len(legend),frameon=False,
                                       fontsize='small',  borderaxespad=0)

        except:
            if j:
                log.warning("Unable to build full width spec plot. Datakeep info:\n"
                            "IFUSLOTID = %s\n"
                            "Dither = %s\n"
                            "SIDE = %s\n"
                            "AMP = %s\n"
                            "Fiber = %i\n"
                            "Wavelength = %f\n"
                            % (self.ifu_slot_id, datakeep['dit'][ind[j]], datakeep['side'][ind[j]], datakeep['amp'][ind[j]],
                               datakeep['fib'][ind[j]], cwave)
                            , exc_info=True)
            else:
                log.warning("Unable to build full width spec plot.",exc_info=True)

        #draw rectangle around section that is "zoomed"
        yl, yh = specplot.get_ylim()
        rec = plt.Rectangle((cwave - ww, yl), 2 * ww, yh - yl, fill=True, lw=1, color='y', zorder=1)
        specplot.add_patch(rec)

        #add specific ticks
        specplot.set_xticks(np.arange(3500,5600,100))

        if G.LyC and (cwave >= 4860.):
            #draw rectangle (highlight) Lyman Continuum region

            lyc_left = (cwave/G.LyA_rest) * 880.  #i.e. (1+z) * 880  where z == obs/rest -1  ... so obs/rest -1 +1
            lyc_width = (cwave/G.LyA_rest) * 30.

            rec = plt.Rectangle((lyc_left, yl), 2 * lyc_width, yh - yl, fill=True, lw=1, color='red', alpha=0.5, zorder=1)
            specplot.add_patch(rec)

        if G.SHOW_SKYLINES:
            try:
                yl, yh = specplot.get_ylim()
                alpha = 0.75

                central_w = 3545
                half_width = 10
                rec = plt.Rectangle((central_w-half_width, yl), 2 * half_width, yh - yl, fill=True, lw=1,
                                    color='gray',alpha=alpha, zorder=1)
                specplot.add_patch(rec)

                rec = plt.Rectangle((central_w-half_width, yl), 2 * half_width, yh - yl, fill=False, lw=1,
                                    color='k',alpha=alpha, zorder=1,hatch='/')
                specplot.add_patch(rec)

                central_w = 5462
                half_width = 5
                rec = plt.Rectangle((central_w - half_width, yl), 2 * half_width, yh - yl, fill=True, lw=1,
                                    color='gray',alpha=alpha, zorder=1,hatch='/',ec=None)
                specplot.add_patch(rec)

                rec = plt.Rectangle((central_w - half_width, yl), 2 * half_width, yh - yl, fill=False, lw=1,
                                    color='k', alpha=alpha, zorder=1, hatch='/',ec=None)
                specplot.add_patch(rec)
            except:
                pass

        try:
            if datakeep['detobj'].extraction_ffsky:
                x_pos_min,x_pos_max, y_pos_min, _ = textplot.axis()
                textplot.text(x_pos_max + 0.01 *(x_pos_max-x_pos_min), y_pos_min, "ff",
                     fontsize=10, color='k',
                     verticalalignment='bottom', horizontalalignment='right')

        except:
            log.debug("Minor exception in build_full_width_spectrum",exc_info=True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)


        if G.ZOO_CUTOUTS:
            try:
                e = datakeep['detobj']
                if e.pdf_name is not None:
                    fn = e.pdf_name.rstrip(".pdf") +  "_zoo_1d_full.png"
                else:
                    fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_1d_full.png"
                fn = op.join(datakeep['detobj'].outdir, fn)
                plt.savefig(fn, format="png", dpi=300)
            except:
                log.error("Unable to write zoo_1d_full image to disk.", exc_info=True)

        plt.close(fig)
        return buf


    def emis_to_fiber_distance(self,emis,fiber):

        if emis.wra:
            ra = emis.wra
            dec = emis.wdec
        elif emis.ra:
            ra = emis.ra
            dec = emis.dec
        else:
            ra = None
            dec = None

        if (ra is not None) and (fiber.ra is not None):
            try:
                d = np.sqrt((np.cos(np.deg2rad(dec)) * (ra - fiber.ra)) ** 2 + (dec - fiber.dec) ** 2) * 3600
            except:
                log.error("Exception in HETDEX::emis_to_fiber_distance",exc_info=True)
                d = None

        return d

    # 2d spectra cutouts (one per fiber)
    def build_2d_LyC_image(self, datakeep, z):

        log.debug("+++++ LyC 2D Image .....")

        # not dynamic, but if we are going to add a combined 2D spectra cutout at the top, set this to 1
        add_summed_image = 1  # note: adding w/cosmics masked out
        frac_y_separator = 0.2  # add a separator between the colored fibers and the black (summed) fiber at this
        # fraction of the cell height

        cmap = plt.get_cmap('gray_r')

        colors = self.make_fiber_colors(min(4, len(datakeep['ra'])),
                                        len(datakeep['ra']))  # + 2 ) #the +2 is a pad in the call
        num_fibers = len(datakeep['xi'])
        num_to_display = min(MAX_2D_CUTOUTS, num_fibers) + add_summed_image  # for the summed images
        # bordbuff = 0.005
        # borderxl = 0.06
        # borderxr = 0.16
        # borderyb = 0.06
        # borderyt = 0.16

        bordbuff = 0.01
        borderxl = 0.06
        borderxr = 0.16
        borderyb = 0.00
        borderyt = 0.15

        # the +1 for the summed image
        dx = (1. - borderxl - borderxr) / 3.
        dy = (1. - borderyb - borderyt) / (num_to_display)
        dx1 = (1. - borderxl - borderxr) / 3.
        dy1 = (1. - borderyb - borderyt - (num_to_display) * bordbuff) / (num_to_display)
        _xw = int(np.round((1.0 + z) * 8.0))
        Y = (yw / dy) / (_xw / dx) * 5. + frac_y_separator * dy  # + 0.2 as a separator at the top

        Y = max(Y, 0.8)  # set a minimum size

        fig = plt.figure(figsize=(5, Y), frameon=True)
        #fig = plt.figure()#figsize=(5, Y), frameon=False)
        #plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        # previously sorted in order from largest distances to smallest
        ind = list(range(len(datakeep['d'])))

        # assume all the same shape
        #summed_image = np.zeros(datakeep['fw_im'][ind[0]].shape)
        summed_image = None

        # need i to start at zero
        # building from bottom up



        grid_idx = -1
        for i in range(num_fibers + add_summed_image):
            make_display = False
            if i < num_fibers:
                pcolor = colors[
                    i]  # , 0:4] #keep the 4th value (alpha value) ... need that to lower the alpha of the greys
                datakeep['color'][ind[i]] = pcolor
                datakeep['index'][ind[i]] = num_fibers - i

                if i > (num_fibers - num_to_display):  # we are in reverse order (building the bottom cutout first)
                    make_display = True
                    grid_idx += 1
                    is_a_fiber = True
                    # ext = list(np.hstack([datakeep['xl'][ind[i]], datakeep['xh'][ind[i]],
                    #                       datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))

                    xc = datakeep['x_2d_lyc'][ind[i]]
                    yc = datakeep['y_2d_lyc'][ind[i]]

                    if xc < 0 or yc < 0:
                        # we're done
                        log.error("Invalid LyC x (%f) or y (%f) coord" % (xc, yc))

                    xl = xc - _xw  # note: 4.0 == integer of +/- 15 AA from center with 2.0 AA per pixel so 15/2 = 7.5, round to 8
                    xr = xc + _xw
                    yl = yc - yw  # standard height, same as normal 2D cutouts
                    yr = yc + yw

                    ext = list(np.hstack([xl, xr, yl, yr]))


                    # set the hot (cosmic) pixel values to zero then employ guassian_filter
                    a = datakeep['fw_im'][ind[i]][:,xl:xr+1]
                    a = np.ma.masked_where(datakeep['fw_err'][ind[i]][:,xl:xr+1] == -1, a)
                    a = np.ma.filled(a, 0.0)

                    if summed_image is None:
                        summed_image = np.zeros(a.shape)
                    #todo: the shapes are not necessarily the same ... should fix, but this is of limited utility
                    #todo: so for now, if they don't match just skip
                    if summed_image.shape == a.shape:
                        summed_image += a * datakeep['fiber_weight'][ind[i]]

                    # GF = gaussian_filter(datakeep['fw_im'][ind[i]], (2, 1))
                    GF = gaussian_filter(a, (2, 1))

                    gauss_vmin = datakeep['vmin1'][ind[i]]
                    gauss_vmax = datakeep['vmax1'][ind[i]]


                    try:
                        pix_image = datakeep['fw_pix'][ind[i]][:,xl:xr+1]
                    except:
                        log.info("Unable to get pixel flat 2D cutout")
                        pix_image = np.full((yr-yl+1,xr-xl+1),255)
                    image = datakeep['fw_im'][ind[i]][:,xl:xr+1]  # im can be the cosmic removed version, depends on G.PreferCosmicCleaned

                    cmap1 = cmap
                    cmap1.set_bad(color=[0.2, 1.0, 0.23])
                    image = np.ma.masked_where(datakeep['fw_err'][ind[i]][:,xl:xr+1] == -1, image)
                    img_vmin = datakeep['vmin2'][ind[i]]
                    img_vmax = datakeep['vmax2'][ind[i]]

                    # plot_label = str(num_fibers-i)
                    plot_label = str("%0.2f" % datakeep['fiber_weight'][ind[i]]).lstrip(
                        '0')  # save space, kill leading 0

                    # the last one is the top one and is the primary
                    datakeep['primary_idx'] = ind[i]

            else:  # this is the top image (the sum)
                is_a_fiber = False
                make_display = True
                grid_idx += 1
                pcolor = 'k'
                ext = None
                pix_image = None  # blank_pixel_flat() #per Karl, just leave it blank, no pattern
                plot_label = "SUM"

                GF = gaussian_filter(summed_image, (2, 1))
                image = summed_image
                img_vmin, img_vmax = self.get_vrange(summed_image, scale=contrast2)
                gauss_vmin, gauss_vmax = self.get_vrange(summed_image, scale=contrast1)

            if make_display:

                ds9_x = 1. + (xl + xr) / 2.
                ds9_y = 1. + (yl + yr) / 2.

                if is_a_fiber:
                    y_separator = 0.0
                else:
                    y_separator = frac_y_separator * dy

                borplot = plt.axes([borderxl + 0. * dx, borderyb + grid_idx * dy + y_separator, 3 * dx, dy])
                smplot = plt.axes(
                    [borderxl + 2. * dx - bordbuff / 3., borderyb + grid_idx * dy + bordbuff / 2. + y_separator,
                     dx1, dy1])
                pixplot = plt.axes(
                    [borderxl + 1. * dx + 1 * bordbuff / 3., borderyb + grid_idx * dy + bordbuff / 2. + y_separator,
                     dx1, dy1])
                imgplot = plt.axes(
                    [borderxl + 0. * dx + bordbuff / 2., borderyb + grid_idx * dy + bordbuff / 2. + y_separator,
                     dx1, dy1])
                autoAxis = borplot.axis()

                rec = plt.Rectangle((autoAxis[0] + bordbuff / 2., autoAxis[2] + bordbuff / 2.),
                                    (autoAxis[1] - autoAxis[0]) * (1. - bordbuff),
                                    (autoAxis[3] - autoAxis[2]) * (1. - bordbuff), fill=False, lw=3,
                                    color=pcolor, zorder=1)
                rec = borplot.add_patch(rec)
                borplot.set_xticks([])
                borplot.set_yticks([])
                borplot.axis('off')

                smplot.imshow(GF,
                              origin="lower", cmap=cmap,
                              interpolation="none", vmin=gauss_vmin,
                              vmax=gauss_vmax,
                              extent=ext)

                smplot.set_xticks([])
                smplot.set_yticks([])
                # smplot.axis(ext)
                smplot.axis('off')

                if pix_image is not None:
                    vmin_pix = 0.9
                    vmax_pix = 1.1
                    pixplot.imshow(pix_image,
                                   origin="lower", cmap=plt.get_cmap('gray'),
                                   interpolation="none", vmin=vmin_pix, vmax=vmax_pix,
                                   extent=ext)  # vmin=0.9, vmax=1.1

                # still need to always turn off the axis
                pixplot.set_xticks([])
                pixplot.set_yticks([])
                # pixplot.axis(ext)
                pixplot.axis('off')

                imgplot.imshow(image,
                               origin="lower", cmap=cmap1,
                               vmin=img_vmin,
                               vmax=img_vmax,
                               interpolation="none", extent=ext)

                imgplot.set_xticks([])
                imgplot.set_yticks([])
                # imgplot.axis(ext)
                imgplot.axis('off')

                if i < num_fibers:
                    # xi = datakeep['xi'][ind[i]]
                    # yi = datakeep['yi'][ind[i]]
                    # xl = int(np.round(xi - ext[0] - res[0] / 2.))
                    # xh = int(np.round(xi - ext[0] + res[0] / 2.))
                    # yl = int(np.round(yi - ext[2] - res[0] / 2.))
                    # yh = int(np.round(yi - ext[2] + res[0] / 2.))

                    sn = datakeep['fiber_sn'][ind[i]]

                    if sn is None:
                        if self.panacea:
                            sn = -99  # so will fail the check and not print

                        else:  # this only works (relatively well) for Cure
                            print("What??? .... can only be panacea")

                    #fiber weights (irrelevenat here)
                    # borplot.text(-0.2, .5, plot_label,
                    #              transform=imgplot.transAxes, fontsize=10, color='k',  # colors[i, 0:3],
                    #              verticalalignment='bottom', horizontalalignment='left')

                # if self.multiple_observations:
                # add the fiber info to the right of the images
                if not G.ZOO:
                    if is_a_fiber:
                        if self.panacea:

                            borplot.text(1.05, .73, 'D("): %0.2f' % (datakeep['d'][ind[i]]),
                                         transform=smplot.transAxes, fontsize=6, color='k',
                                         verticalalignment='bottom', horizontalalignment='left')

                            borplot.text(1.05, .53,
                                         'x, y: %d, %d' % (ds9_x, ds9_y),
                                         transform=smplot.transAxes, fontsize=6, color='k',
                                         verticalalignment='bottom', horizontalalignment='left')

                            try:
                                l3 = datakeep['date'][ind[i]] + "_" + datakeep['obsid'][ind[i]] + "_" + \
                                     datakeep['expid'][ind[i]]

                                # !!! multi*fits is <specid>_<ifuslot>_<ifuid> !!!
                                # !!! so do NOT change from spec_id
                                # !!! note: having all three identifiers makes the string too long so leave as is
                                l4 = datakeep['spec_id'][ind[i]] + "_" + datakeep['amp'][ind[i]] + "_" + \
                                     datakeep['fib_idx1'][ind[i]]

                                borplot.text(1.05, .33, l3,
                                             transform=smplot.transAxes, fontsize=6, color='k',
                                             verticalalignment='bottom', horizontalalignment='left')
                                borplot.text(1.05, .13, l4,
                                             transform=smplot.transAxes, fontsize=6, color='k',
                                             verticalalignment='bottom', horizontalalignment='left')
                            except:
                                log.error("Exception building extra fiber info.", exc_info=True)

                        else:
                            print("What??? .... can only be panacea")
                    else:
                        borplot.text(1.05, .35, '\nWEIGHTED\nSUM',
                                     transform=smplot.transAxes, fontsize=8, color='k',
                                     verticalalignment='bottom', horizontalalignment='left')

                if grid_idx == (num_to_display - 1):  # (num + add_summed_image - 1):
                    smplot.text(0.5, 1.3, 'Smoothed',
                                transform=smplot.transAxes, fontsize=8, color='k',
                                verticalalignment='top', horizontalalignment='center')
                    pixplot.text(0.5, 1.3, 'Pixel Flat',
                                 transform=pixplot.transAxes, fontsize=8, color='k',
                                 verticalalignment='top', horizontalalignment='center')
                    imgplot.text(0.5, 1.3, '2D Spec',
                                 transform=imgplot.transAxes, fontsize=8, color='k',
                                 verticalalignment='top', horizontalalignment='center')

        buf = io.BytesIO()
        # plt.tight_layout()#pad=0.1, w_pad=0.5, h_pad=1.0)
        plt.savefig(buf,format='png', dpi=300)

        # try:
        #     e = datakeep['detobj']
        #
        #     if e.pdf_name is not None:
        #         fn = e.pdf_name.rstrip(".pdf") + "_2d_lyc_fib.png"
        #     else:
        #         fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_2d_lyc_fib.png"
        #     fn = op.join(e.outdir, fn)
        #     plt.savefig(fn, format="png", dpi=300)
        # except:
        #     log.error("Unable to write 2d_lyc_fib image to disk.", exc_info=True)

        plt.close(fig)
        return buf, Y
        #end build_2D_LyC_image



    def build_full_width_2D_cutouts(self, datakeep, z):
        """
        Sit just under the full 1D-Spectrum
        Full width, summed 2D spectra and summed pixel flat
        Same size as 1D spectra image section

        :param datakeep:
        :param z:
        :return:
        """
        num = 0 #left over code from build_full_width_spectrum (says we are not plotting individual spectra)
        dy = 1.0 / (num + 5)  # + 1 skip for legend, + 2 for double height spectra + 2 for double height labels

        #figure_sz_y = G.GRID_SZ_Y

        fig = plt.figure(figsize=(G.FIGURE_SZ_X,G.GRID_SZ_Y), frameon=False)
        #fig = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        ind = list(range(len(datakeep['d'])))
        ind.reverse() #since last indexed item is the highest weight

        border_buffer = 0.025  # percent from left and right edges to leave room for the axis labels

        # this is the 2D averaged spectrum
        #specplot = plt.axes([border_buffer, float(num + 1.0) * dy, 1.0 - (2 * border_buffer), dy * 2])
        specplot = plt.axes([border_buffer, 0.0, 1.0 - (2 * border_buffer),1.0])

        # this is the pixel flat plots
        #flatplot = plt.axes([border_buffer, float(num + 1.0) * dy, 1.0 - (2 * border_buffer), dy * 2])
        rm = 0.2


        #iterate over and sum, like in above
        num_fibers = len(datakeep['xi'])
        summed_image = np.zeros(datakeep['fw_im'][ind[0]].shape)
        master_bad = np.zeros(summed_image.shape)
        summed_flat  = np.zeros(summed_image.shape)
        separator_img = np.zeros((2,summed_image.shape[1]))

        #todo: question ... all fibers, or just the top 4
        num_fibers = min(4,num_fibers)
        for i in range(num_fibers):
        #for i in range(num_fibers,(num_fibers-5),-1):
            bad = np.where(datakeep['fw_err'][ind[i]] == -1)
            master_bad[bad] = -1

            #a = np.ma.masked_where(datakeep['fw_err'][ind[i]] == -1, a)
            #a = np.ma.filled(a, 0.0)
            summed_image += datakeep['fw_im'][ind[i]] * datakeep['fiber_weight'][ind[i]]

            #todo: question ... weight the pixel flats or not?
            #todo: question ... what about summing of flats taken at different times? would the ranges of pix values
            #      still be compatible (on same scale)?
            summed_flat += datakeep['fw_pix'][ind[i]] # * datakeep['fiber_weight'][ind[i]]





        #plot the two stacked figures
        image_vmin, image_vmax = self.get_vrange(summed_image, scale=contrast1)
        flat_vmin, flat_vmax = self.get_vrange(summed_flat, scale=contrast1)

        #put the bad pixels on sum
        summed_image = np.ma.masked_where(master_bad < 0, summed_image)

        #smooth the 2D image??
        smoothed_image = gaussian_filter(summed_image, (2, 1))
        smoothed_image = np.ma.masked_where(master_bad < 0, smoothed_image)

        #adjust the flat to match vmin, vmax
        #summed_flat = summed_flat*(image_vmin/flat_vmin)

        stacked_img = np.vstack([summed_image,separator_img,summed_flat])

        cmap1 = plt.get_cmap('gray_r')
        cmap1.set_bad(color=[0.2, 1.0, 0.23])

        # ext = list(np.hstack([datakeep['xl'][ind[i]], datakeep['xh'][ind[i]],
        #                       datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))

        #image_vmin, image_vmax = self.get_vrange(smoothed_image, scale=contrast1)
        specplot.imshow(smoothed_image[5:16,:],
                      origin="lower", cmap=cmap1,
                      interpolation="none", vmin=image_vmin,
                      vmax=image_vmax)#,extent=[0,1032,0,28])


        #plt.savefig("/home/dustin/code/python/elixer/lycon/test.png",format="png", dpi=300)
        #end part savefig?
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        # if G.ZOO_CUTOUTS:
        #     try:
        #         e = datakeep['detobj']
        #         if e.pdf_name is not None:
        #             fn = e.pdf_name.rstrip(".pdf") + "_zoo_2d_full.png"
        #         else:
        #             fn = self.output_filename + "_" + str(e.entry_id).zfill(3) + "_zoo_2d_full.png"
        #         fn = op.join(datakeep['detobj'].outdir, fn)
        #         plt.savefig(fn, format="png", dpi=300)
        #     except:
        #         log.error("Unable to write zoo_2d_full image to disk.", exc_info=True)

        plt.close(fig)
        return buf

#end HETDEX class