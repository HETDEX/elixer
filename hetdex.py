
import global_config as G
import matplotlib
matplotlib.use('agg')
import time

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import io
from PIL import Image

from astropy.io import fits as pyfits
from astropy.coordinates import Angle
from astropy.stats import biweight_midvariance
from astropy.modeling.models import Moffat2D, Gaussian2D
from astropy.visualization import ZScaleInterval
from photutils import CircularAperture, aperture_photometry
from scipy.ndimage.filters import gaussian_filter

from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

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

import line_prob

import hetdex_fits
import fiber as voltron_fiber
import ifu as voltron_ifu #only using to locate panacea files (voltron only uses individual fibers, not entire IFUs)
import spectrum as voltron_spectrum

#todo: write a class wrapper for log
#an instance called log that has functions .Info, .Debug, etc
#they all take a string (the message) and the exc_info flag
# inside they prepend a time to the string and then pass along to the regular logger

#log = G.logging.getLogger('hetdex_logger')
#log.setLevel(G.logging.DEBUG)

log = G.Global_Logger('hetdex_logger')
log.setlevel(G.logging.DEBUG)

CONFIG_BASEDIR = G.CONFIG_BASEDIR
VIRUS_CONFIG = G.VIRUS_CONFIG #op.join(CONFIG_BASEDIR,"virus_config")
FPLANE_LOC = G.FPLANE_LOC #op.join(CONFIG_BASEDIR,"virus_config/fplane")
IFUCEN_LOC = G.IFUCEN_LOC #op.join(CONFIG_BASEDIR,"virus_config/IFUcen_files")
DIST_LOC = G.DIST_LOC #op.join(CONFIG_BASEDIR,"virus_config/DeformerDefaults")
PIXFLT_LOC = G.PIXFLT_LOC #op.join(CONFIG_BASEDIR,"virus_config/PixelFlats/20161223")

PLOT_SUMMED_SPECTRA = True #zoom-in plot of top few fibers
MAX_2D_CUTOUTS = 4 #~ 5x3 of 2d cutouts  +1 for the summed cutout

SIDE = voltron_fiber.SIDE
#!!! REMEBER, Y-axis runs 'down':  Python 0,0 is top-left, DS9 is bottom-left
#!!! so in DS9 LU is 'above' LL and RL is 'above' RU
AMP  = voltron_fiber.AMP #["LU","LL","RL","RU"] #in order from bottom to top
AMP_OFFSET = voltron_fiber.AMP_OFFSET# {"LU":1,"LL":113,"RL":225,"RU":337}



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


#todo: may change with Config0 vs Config1 ... so may need to add additional info
def flip_amp(amp=None,buf=None):
    if (amp is None) or (buf is None):
        log.error("Amp designation or buffer is None.")
        return None

    #sanity check buf for correct size
    y,x = np.shape(buf)

    if (x != 1032) and (y != 1032):
        log.error('Amp buffer wrrong shape: (%d,%d). Expecting (1032,1032).' %(x,y))
        return None

    if (amp.upper() == 'LL') or (amp.upper() == 'RU'):
        log.debug('Will not flip amp (%s). Already in correct orientation.' % amp)
        return buf

    if (amp.upper() == 'LU') or (amp.upper() == 'RL'):
        log.debug('Will flip amp (%s). Reverse X and Reverse Y.' % amp)
        return np.fliplr(np.flipud(buf))

    log.warning("Unexpected AMP designation: %s" % amp)
    return None


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

def gaussian(x,x0,sigma,a=1.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None

    return a*np.exp(-np.power((x - x0)/sigma, 2.)/2.)


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

#def fit_gaussian(x,y):
#    yfit = None
#    parm = None
#    pcov = None
#    try:
#        parm, pcov = curve_fit(gaussian, x, y,bounds=((-np.inf,0,-np.inf),(np.inf,np.inf,np.inf)))
#        yfit = gaussian(x,parm[0],parm[1],parm[2])
#    except:
#        log.error("Exception fitting gaussian.",exc_info=True)
#
#    return yfit,parm,pcov


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

    def __init__(self,tokens,emission=True,line_number=None,fcs_base=None,fcsdir=None):
        #fcs_base is a basename of a single fcs directory, fcsdir is the entire FQdirname
        #fcsdir is more specific
        #skip NR (0)
        self.matched_cats = [] #list of catalogs in which this object appears (managed outside this class)
        self.status = 0
        self.plot_dqs_fit = False
        self.dqs = None #scaled score
        self.dqs_raw = None #Detection Quality Score (raw score)
        self.type = 'unk'
        self.entry_id = None #e.g. line number
        self.id = None
        self.x = None
        self.y = None
        self.w = 0.0
        self.la_z = 0.0
        self.dataflux = 0.0
        self.modflux = 0.0
        self.fluxfrac = 1.0
        self.estflux = None
        self.sigma = 0.0 #also doubling as sn (see @property sn farther below)
        self.snr = None
        self.chi2 = 0.0
        self.chi2s = 0.0
        self.chi2w = 0.0
        self.gammq = 0.0
        self.gammq_s = 0.0
        #self.eqw = 0.0
        self.eqw_obs = 0.0
        self.cont = -9999
        self.cont_cgs = -9999
        self.fwhm = -1.0 #in angstroms
        self.panacea = False

        self.ifuslot = None
        self.wra = None
        self.wdec = None

        self.num_hits = 0

        self.fibers = []
        self.fibers_sorted = False
        self.outdir = None

        #flux calibrated data (from Karl's detect and calibration)
        self.fcsdir = None


        self.sumspec_wavelength = []
        self.sumspec_counts = []
        self.sumspec_flux = []
        self.sumspec_flux_unit_scale = 1e-17 #cgs
        self.sumspec_fluxerr = []
        self.sumspec_wavelength_zoom = []
        self.sumspec_counts_zoom = []
        self.sumspec_flux_zoom = []
        self.sumspec_fluxerr_zoom = []
        self.sumspec_2d_zoom = []
        self.spec_obj = voltron_spectrum.Spectrum() #use for classification, etc

        self.p_lae = None #from Andrew Leung
        self.p_oii = None
        self.p_lae_oii_ratio = None

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
                    self.entry_id = int(os.path.basename(fcsdir).split("_")[1]) #assumes 20170322v011_005
                except:
                    pass

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



        # dont do this here ... only call after we know we are going to keep this DetObj
        # as this next step takes a while
        #self.load_fluxcalibrated_spectra()


    @property
    def sn(self):
        if self.snr is not None:
            return self.snr
        else:
            return self.sigma


    def multiline_solution_score(self):
        '''

        :return: bool (True) if the solution is good
                 float best solution score
        '''
        #todo: gradiate score a bit and add other conditions
        #todo: a better way of some kind rather than a by-hand scoring
        if (self.spec_obj is not None) and (len(self.spec_obj.solutions) > 0):
            sols = self.spec_obj.solutions
            # need to tune this
            # score is the sum of the observed eq widths
            if (self.spec_obj.solutions[0].score >= G.MULTILINE_MIN_SOLUTION_SCORE) and \
                    (self.spec_obj.solutions[0].prob_real >= G.MULTILINE_MIN_SOLUTION_CONFIDENCE) and \
                    (self.spec_obj.solutions[0].frac_score > 0.5) and \
                    (len(self.spec_obj.solutions[0].lines) >= G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY):
                if (len(self.spec_obj.solutions) == 1) or \
                    ((len(self.spec_obj.solutions) > 1) and \
                      (self.spec_obj.solutions[0].frac_score / self.spec_obj.solutions[1].frac_score > 2.0)):

                    return True, self.spec_obj.solutions[0].prob_real

            if G.MULTILINE_ALWAYS_SHOW_BEST_GUESS:
                return False, self.spec_obj.solutions[0].prob_real

        return False, 0.0

    #rsp1 (when t5all was provided and we want to load specific fibers for a single detection)
    def load_fluxcalibrated_spectra(self):

        #todo: get the line width as well

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

        #note the 20170615v09 might be v009 ... inconsistent

        if not op.isdir(self.fcsdir):
            log.debug("Cannot find flux calibrated spectra directory: " + self.fcsdir +" . Trying alternate (extra 0)")
            toks = self.fcsdir.split('v')
            toks[-1] = 'v0'+toks[-1]
            self.fcsdir = "".join(toks)

            if not op.isdir(self.fcsdir):
                #still did not find ...
                log.error("Cannot find flux calibrated spectra directory: " + self.fcsdir)
                self.status = -1
                return

        basename = op.basename(self.fcsdir)

        log.info("Loading HETDEX data (flux calibrated spectra, etc) for %s" %self.fcsdir)


        #get basic info
        file = op.join(self.fcsdir,"out2")
        try:
            with open(file,'r') as f:
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
            log.error("Fatal: Cannot read out2 file: %s" % file, exc_info=True)
            self.status = -1
            return

        #get summary information (res file)
        # wavelength   wavelength(best fit) counts sigma  ?? S/N   cont(10^-17)
        #wavelength   wavelength(best fit) flux(10^-17) sigma  ?? S/N   cont(10^-17)
        file = op.join(self.fcsdir, basename + "_2d.res")
        try:
            with open(file, 'r') as f:
                f = ft.skip_comments(f)
                count = 0
                for l in f: #should be exactly one line (but takes the last one if more than one)
                    count += 1
                    if count > 1:
                        log.warning("Unexepcted number of lines in %s" %(file))
                    toks = l.split()
                    #w = float(toks[1])  # sanity check against self.w
                    self.w = float(toks[1])

                    if self.w == 0.0: #seeing some 2d.res files are all zeroes
                        log.error("Fatal: Invalid *_2d.res file. Empty content.: %s " % file)
                        #self.status = -1
                        self.w = -1.0

                        self.estflux = -0.0001
                        self.snr = 0.01
                        self.fwhm = 0.0
                        self.cont_cgs = -0.0001
                        self.eqw_obs = -1.0


                    else:

                        #as is, way too large ... maybe not originally calculated as per angstrom? so divide by wavelength?
                        #or maybe units are not right or a miscalculation?
                        #toks2 is in counts
                        #todo: are we SURE this is -17 and not -18 ??
                        self.estflux = float(toks[2]) * 1e-18 #self.sumspec_flux_unit_scale # * 10 ** (-17)
                        #print("Warning! Using old flux conversion between counts and flux!!!")
                        #self.estflux = float(toks[2]) * flux_conversion(self.w)

                        self.fwhm = 2.35*float(toks[3]) #here sigma is the gaussian sigma, the line width, not significance
                        self.snr = float(toks[5])
                        self.cont_cgs = float(toks[6]) * self.sumspec_flux_unit_scale# * 10 ** (-17)

                        # todo: need a floor for cgs (if negative)
                        # for now only
                        if self.cont_cgs <= 0.:
                            print("Warning! Using predefined floor for continuum")
                            self.cont_cgs = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(self.w)

                        self.eqw_obs = self.estflux / self.cont_cgs
        except:
            log.error("Cannot read *2d.res file: %s" % file, exc_info=True)

        #build the mapping between the tmpxxx files and fibers
        #l1 file
        multi = []
        idstr = []

        file = op.join(self.fcsdir, "l1")
        try:
            out = np.genfromtxt(file, dtype=None,usecols=(4,8))
            multi = np.array(map(lambda x: x.split(".")[0],out[:,0]))
            idstr = out[:,1]
        except:
            log.error("Cannot read l1: %s" % file, exc_info=True)


        #if don't already have fibers, build them
        #l6 file has the paths to the multi*fits file
        #l1 file has the bulk info
        # RA, Dec, X, Y, multi(?ixy), exp, ???, ???, obs_string, obs_date, shot #
        # tmp1xx.files are in order, so first entry is tmp101.dat

        file1 = op.join(self.fcsdir, "l1")
        file6 = op.join(self.fcsdir, "l6")
        if (self.fibers is None) or (len(self.fibers) == 0):
            try:
                out1 = np.genfromtxt(file1, dtype=None)
                out6 = np.genfromtxt(file6, dtype=None)

                if len(out6) != len(out1):
                    log.error("Error. Unmatched file lengths for l1 and l6 under %s" %(self.fcsdir))
                    return

                for i in range(len(out6)):
                    ra = float(out1[i][0])
                    dec = float(out1[i][1])
                    sky_x = float(out1[i][2])
                    sky_y = float(out1[i][3])
                    multi_name = out1[i][4]
                    exp = int(out1[i][5].lstrip("exp"))
                    time_ex = out1[i][8]
                    ymd = out1[i][9]
                    shot_num = out1[i][10]

                    toks = multi_name.split("_")  #example "multi_038_096_014_RL_041.ixy"
                    specid = toks[1] #string
                    ifu_slot = toks[2]
                    ifuid = toks[3]
                    side = toks[4]
                    fib_id = toks[5].split(".")[0]

                    #20171220T094210.0_020_095_004_RU_089
                    idstring = time_ex + "_" + specid + "_" + ifu_slot + "_" + ifuid + "_" + side + "_" + fib_id

                    #just build up from the idstring
                    fiber = voltron_fiber.Fiber(idstring)
                    if fiber is not None:
                        fiber.ra = ra
                        fiber.dec = dec
                        fiber.obsid = int(shot_num)
                        fiber.expid = exp
                        fiber.detect_id = self.id
                        # add the fiber (still needs to load its fits file)
                        # we already know the path to it ... so do that here??
                        fiber.fits_fn = out6[i]

                        fits = hetdex_fits.HetdexFits(fiber.fits_fn, None, None, exp-1, panacea=True)
                        fits.obs_date = fiber.dither_date
                        fits.obs_ymd = fits.obs_date
                        fits.obsid = fiber.obsid
                        fits.expid = fiber.expid
                        fits.amp = fiber.amp
                        fits.side = fiber.amp[0]
                        fiber.fits = fits
                        #self.sci_fits.append(fits)


                        #not here ... this is for the entire detection not just this fiber
                        #self.wra = ra
                        #self.wdec = dec
                        #self.x = sky_x
                        #self.y = sky_y

                        self.fibers.append(fiber)
            except:
                log.error("Cannot read l1,l6 files to build fibers: %s" % self.fcsdir, exc_info=True)

             #build fibers from this directory (and get the multi*fits files into the list)
                #there is a file that links all this together
            #basically, go through the fibers in the directory
            #if the associated multi*fits file is not already in the list, append it
            #construct the fiber and append to the list



        #todo: get the tmpxxx.dat files for the relative through put for each fiber (by wavelength)
        #these are the spectra cutout sections .. so need to weight each wavelength for each fiber
        #in the cutout
        #maybe add another array ... relative throughput


        #get the weights
        file = op.join(self.fcsdir, "list2")
        tmp_base_count = 100
        try:
            tmp = np.genfromtxt(file,dtype=np.str,usecols=0)
            if len(tmp) != len(multi):
                log.error("Cannot match up list2 and l1 files")
            else:

                #tmp = out[:,0]
                out = np.loadtxt(file, dtype=np.float, usecols=(1,2))
                keep = out[:,0]
                w = out[:,1] #weights

                #sum up the weights where keep == 0
                norm = np.sum(w[np.where(keep==0)])
                subset_norm = 0.0
                #max_weight = -999.9

                #todo: if only using a subset of fibers (e.g. from a line file), how do we interpret the weights?
                # as here, as a subset??

                #not optimal for rsp1 organization, but if a line file was provided and the source of the fibers
                # is not from rsp1, this is necessary (and it works either way ... and there are usually just a few
                # fibers, so not that bad)
                for f in self.fibers:
                    f.relative_weight = 0.0

                    for i in range(len(out)):
                        if keep[i] == 0.:
                            #find which fiber, if any, in set this belongs to
                            if (f.multi == multi[i]) and (f.scifits_idstring == idstr[i]):
                                f.relative_weight += w[i] #usually there is only one, so this is 0 + weight = weight
                                subset_norm += w[i]

                                #if w[i] > max_weight:
                                #    max_weight = w[i]

                                #now, get the tmp file for the thoughput
                                tmpfile = op.join(self.fcsdir, "tmp"+str(i+101)+".dat")
                                tmp_out = np.loadtxt(tmpfile, dtype=np.float)
                                #0 = wavelength, 1=counts?, 2=flux? 3,4,5 = throughput 6=cont?

                                f.fluxcal_central_emis_wavelengths = tmp_out[:,0]
                                f.fluxcal_central_emis_counts = tmp_out[:,1]
                                f.fluxcal_central_emis_flux = tmp_out[:,2]
                                f.fluxcal_central_emis_thru = tmp_out[:,3]*tmp_out[:,4]*tmp_out[:,5]
                                f.fluxcal_central_emis_fluxerr =  tmp_out[:,6]
                                #f.fluxcal_emis_cont = tmp_out[:,6]

                        #else:
                        #    # find which fiber, if any, in set this belongs to
                        #    if (f.multi == multi[i]) and (f.scifits_idstring == idstr[i]):
                        #        f.relative_weight += 0.0

                    #f.relative_weight /= norm #note: some we see are zero ... they do not contribute

                for f in self.fibers:
                    f.relative_weight /= subset_norm
                    #f.relative_weight /= max_weight



                #todo: how is Karl using the weights (sum to 100% or something else?)
                #now, remove the zero weighted fibers, then sort
                self.fibers = [x for x in self.fibers if x.relative_weight > 0]
                self.fibers.sort(key=lambda  x: x.relative_weight,reverse=True) #highest weight is index = 0
                self.fibers_sorted = True

        except:
            log.error("Fatal. Cannot read list2: %s" % file, exc_info=True)
            print("Fatal. Cannot read list2: %s" % file)
            self.status = -1
            return

        #avg_ra = 0.0
        #avg_dec = 0.0
        #for f in self.fibers:
        #    avg_ra += f.ra * f.relative_weight
        #    avg_dec += f.dec * f.relative_weight



        #get the full flux calibrated spectra
        file = op.join(self.fcsdir, basename + "specf.dat")
        try:
            size = os.stat(file).st_size
            if size == 0:
                log.error("*specf.res file is zero length: %s" %file)
                self.status = -1
                return
        except:
            pass

        try:
            out = np.loadtxt(file, dtype=None)

            self.sumspec_wavelength = out[:,0]
            self.sumspec_counts = out[:, 1]
            #reminder data scientific notation, so mostly e-17 or e-18
            self.sumspec_flux = out[:,2] * 1e17
            #todo: get flux error (not yet in this file)
            #self.sumspec_fluxerr = out[:,6]  * 1e17

            #self.sumspec_fluxerr = np.full_like(self.sumspec_flux,1.0) #i.e. 0.5x10^-17 cgs
            #np.random.seed(1138)
            #self.sumspec_fluxerr = np.random.random(len(self.sumspec_flux)) # just for test

        except:
            log.error("Fatal. Cannot read *specf.dat file: %s" % file, exc_info=True)
            print("Fatal. Cannot read *specf.dat file: %s" % file)
            self.status = -1
            return

        # get the zoomed in flux calibrated spectra
        file = op.join(self.fcsdir, basename + "spece.dat")
        try:
            out = np.loadtxt(file, dtype=None)

            self.sumspec_wavelength_zoom = out[:, 0]
            self.sumspec_counts_zoom = out[:, 1]
            self.sumspec_flux_zoom = out[:, 2]  * 1e17
            #todo: get flux error (not yet in this file)
            #self.sumspec_fluxerr_zoom = out[:,6]  * 1e17
            #self.sumspec_fluxerr_zoom = np.full_like(self.sumspec_flux_zoom, 0.5)  # i.e. 0.5x10^-17 cgs

        except:
            log.error("Fatal. Cannot read *_spece.res file: %s" % file, exc_info=True)
            print("Fatal. Cannot read *_spece.res file: %s" % file)
            self.status = -1
            return


        #get zoomed 2d cutout
        #fits file
        file = op.join(self.fcsdir, basename + ".fits")
        try:
            f = pyfits.open(file)
            self.sumspec_2d_zoom = f[0].data
            f.close()
        except:
            log.error("could not read file " + file, exc_info=True) #not fatal


        #todo: get info from Karl and load
        #todo: likely look in a known location (possible a relative path from the t5all file?)
        #todo: read in the "catalog" list of fiberfiles (match RA, Dec to emissionline in question)
        #todo:    line count (number) maps to which tmp1xx.dat file to read in for the wavelengths, flux, errors
        #
        #todo: this is a cumulative (summed over some number of fibers) spectrum?
        #todo: need to get to spectrum for each contributing fiber and its weight
        #todo:    (put weight with the fiber? or in hetdex_fiber (along with the fe_data, etc)


        #set_spectra(self, wavelengths, values, errors, central, estflux=None, eqw_obs=None)
        #self.spec_obj.set_spectra(self.sumspec_wavelength,self.sumspec_counts,self.sumspec_fluxerr,self.w)
        self.spec_obj.identifier = "eid(%d)" %self.entry_id
        self.spec_obj.set_spectra(self.sumspec_wavelength, self.sumspec_flux, self.sumspec_fluxerr, self.w,
                                  values_units=-17,estflux=self.estflux,eqw_obs=self.eqw_obs)

        #print("DEBUG ... spectrum peak finder")
        #if G.DEBUG_SHOW_GAUSS_PLOTS:
        #    self.spec_obj.build_full_width_spectrum(show_skylines=True, show_peaks=True, name="testsol")
        #print("DEBUG ... spectrum peak finder DONE")

        self.spec_obj.classify() #solutions can be returned, also stored in spec_obj.solutions


    def dqs_score(self,force_recompute=False):
        #dqs_compute_score eventually calls to dqs_shape() which can trigger a force_recompute
        if G.COMPUTE_QUALITY_SCORE:
            if self.dqs_compute_score(force_recompute=force_recompute):
                return True
            else:
                self.dqs_compute_score(force_recompute=True)

    def dqs_compute_score(self,force_recompute=False): #Detection Quality Score (score)
        if (self.dqs is not None) and not force_recompute:
            return self.dqs

        log.debug("Computing detection quality score for detection #" + str(self.id))
        self.dqs = 0.

        if self.wra is not None:
            ra = self.wra
            dec = self.wdec
        elif self.ra is not None:
            ra = self.ra
            dec = self.dec
        else:
            return self.dqs

        #compute score for each fiber and sum
        score = 0.0
        for f in self.fibers:
            score += f.dqs_score(ra,dec,force_recompute)

        # todo: future possibility ... could be there are additional criteria outside individual fibers that need
        # todo: to be considered. Add them here.

        #CAVEATS: this assumes a point-like emitter with a symmetric (gaussian) distribution of signal
        #if we are on the edge of a resolved object, then there will be preferred direction of increased signal (not
        #symmetric) and that can push down the score, etc

        #these distance and SN penality only makes sense if we have the weighted RA and Dec
        if self.wra is not None:
            gross_noise = 3.0 #e.g. SN = +/- gross_noise
            penalty = 0.0
            bonus_idx = []
            penalty_idx = []
            for i in range(len(self.fibers)):
                if (self.fibers[i].dqs_dist == None) or (self.fibers[i].sn == None) :
                    continue
                for j in range(i+1,len(self.fibers)):
                    if (self.fibers[j].dqs_dist == None) or (self.fibers[j].sn == None):
                        continue

                    if self.fibers[i].dqs_dist < self.fibers[j].dqs_dist:
                        f1 = self.fibers[i]
                        f2 = self.fibers[j]
                    else:
                        f2 = self.fibers[i]
                        f1 = self.fibers[j]

                    #todo: fix for cure (this index to ID is really only true for panacea)
                    f1_id = 1 + i
                    f2_id = 1 + j

                    msg = None
                    sigma = 1.5 #like an average PSF

                    #adding .dqs_dist to sigma is rough approximation of
                    # farther from center == greater sigma (fatter gaussian)
                    g1 = gaussian(f1.dqs_dist, 0.0, sigma+f1.dqs_dist)
                    g2 = gaussian(f2.dqs_dist, 0.0, sigma+f2.dqs_dist)

                    if not (g1 and g2):
                        log.debug("Invalid gaussian for detect ID # %d" % (self.id))
                        continue

                    p1 = f1.sn / g1
                    p2 = f2.sn / g2

                    delta_peak_sn = abs(p1-p2)

                    if (f1.dither_date == f2.dither_date) and (f1.obsid == f2.obsid):
                        penalty_peak_limit = 1.5 * gross_noise
                        bonus_peak_limit = 1.0 * gross_noise
                        if f1.expid == f2.expid:
                            penalty_scale = 0.5
                        else:
                            penalty_scale = 0.4
                    else: #different observations
                        penalty_peak_limit = 2.0 * gross_noise
                        bonus_peak_limit = 1.0 * gross_noise
                        penalty_scale = 0.3


                    if delta_peak_sn > penalty_peak_limit:  # penalty
                        if not (i in penalty_idx): #only penalize for this fiber once
                            penalty_idx.append(i)
                            p = min(1.0,(delta_peak_sn-penalty_peak_limit)/penalty_peak_limit) * penalty_scale * f1.dqs

                            penalty += p
                            msg = "Score Penalty (%g), detect ID# %d, f1:%d f2:%d . Delta_Peak SN = %g , limit = %g" % \
                                  (p, self.id, f1_id, f2_id, p1-p2,penalty_peak_limit)

                    elif (delta_peak_sn < bonus_peak_limit) and not (j in bonus_idx): #what was expected ... Bonus
                        #only get the bonus for this fiber once
                        bonus_idx.append(j)
                        #todo: f2.dqs could be zero ... should there be a minimal bonus and a maximum?
                        p = (bonus_peak_limit - delta_peak_sn ) / bonus_peak_limit
                        penalty -= p
                        msg = "Score bonus (%g), detect ID# %d, f1:%d f2:%d . Delta_SN = %g, limit = %g" % \
                              (p, self.id, f1_id, f2_id, p1-p2, bonus_peak_limit)

                    #else: no bonus, no penalty

                    if msg:
                        log.debug(msg)

            score -= penalty
            if penalty < 0:
                log.info("Detect ID# %d total bonus (%g). New Score = %g" % (self.id, -1* penalty, score))
            else:
                log.info("Detect ID# %d total penalty (%g). New Score = %g" %(self.id, penalty, score))


        self.dqs_raw = score
        if self.dqs_shape():
            self.dqs_calc_scaled_score()
        else:
            return False

        return True

    def dqs_shape(self):
        #todo: defunct?? replace with spectrum::signal_score()
        force_recompute = False
        bad_pix = 0
        fiber_count = 0
        wave_step = 1.0 #AA
        wave_side = 8.0 #AA

        wave_x = np.arange(self.w - wave_side, self.w + wave_side + wave_step, wave_step)
        wave_counts = np.zeros(wave_x.shape)

        #central_wave = np.zeros(PEAK_PIXELS*2 + 1)

        dqs_bad_count = 0
        #total_fiber_weight = 0.0 #if using f.dqs_w the weighting is already scaled

        for f in self.fibers:
            if f.dqs_bad:
                dqs_bad_count += 1
                continue

            if len(f.central_emis_counts) == 0:
                continue #no modification

            # add up all fibers (weighted? or at least filtered by distance?)
            #if (f.dqs != None) and (f.sn > 3.5):
            if f.dqs_w > 0.0:
                fiber_count += 1
                bad_pix += f.central_wave_pixels_bad
                try:
                    #todo: should this be weighted by the score?? or maybe by distance (f.dqs_dist or f.dqs_w)?
                    wave_counts += f.dqs_w * (np.interp(wave_x, f.central_emis_wavelengths,f.central_emis_counts))
                    #total_fiber_weight += f.dqs_w
                except:
                    log.error("Improperly shaped fiber.central_emis_counts: %s" % f.idstring,exc_info=True)
                    return True #return False if want to force a recompute

        if fiber_count == 0:
            #we are done ... no fibers to fit a shape
            log.info("Emission # %d -- no fibers qualify to make spectra" % (self.id))
            return True

        #blunt very negative values
       # wave_counts /= total_fiber_weight  # f.dqs_w already scaled to 1.0 so don't divide  out
        wave_counts = np.clip(wave_counts,0.0,np.inf)

        xfit = np.linspace(wave_x[0], wave_x[-1], 100)

        fit_wave = None
        rms_wave = None
        check_for_bad_pixels = False
        error = None

        wide = True

        #use the wide_fit if we can ... if not, use narrow fit
        # todo: defunct?? replace with spectrum::signal_score()
        try:
            parm, pcov = curve_fit(gaussian, wave_x, wave_counts,p0=(self.w,1.0,0),
                                     bounds=((self.w-wave_side, 0, -np.inf), (self.w+wave_side, np.inf, np.inf)))

            fit_wave = gaussian(xfit, parm[0], parm[1], parm[2])
            rms_wave = gaussian(wave_x, parm[0], parm[1], parm[2])
            error = rms(wave_counts,rms_wave)
        except:
            log.error("Could not wide fit gaussian (will try narrow) -- Detect ID # %d." % self.id)
            wide = False

            #use narrow fit
            try:
                 parm, pcov = curve_fit(gaussian, wave_x, wave_counts, p0=(self.w,1.0,0),
                                        bounds=((self.w-1.0, 0, -np.inf), (self.w+1.0, np.inf, np.inf)))
                 fit_wave = gaussian(xfit, parm[0], parm[1], parm[2])
                 rms_wave = gaussian(wave_x, parm[0], parm[1], parm[2])
                 error = rms(wave_counts, rms_wave)

                 if (fit_wave is not None) and (parm is not None):
                     if parm[1]*1.9 < 1.0:
                         check_for_bad_pixels = True
            except:
                log.error("Detect ID # %d could not narrow fit gaussian. Possible hot/stuck pixel. " % self.id) #, exc_info=True)
                check_for_bad_pixels = True

            #super narrow (or no fit) with high peak
            if check_for_bad_pixels and (dqs_bad_count == 0): #else, already checked
                # reminder: seeing the same fiber (or pixels) repeatedly for a given detection is not unusual
                # (seeing it across many detections would be ... but cannot know that here)
                # just the combination of extremely narrow or no-fit gaussian AND the same pixels is a good indicator
                # of bad pixel(s)

                for f in self.fibers:
                    #sort and grab the top two ... trigger if #1 >> 0 and >> #2 ?
                    sl = sorted(f.central_emis_counts,key=float,reverse=True)
                    if (not f.dqs_bad) and (sl[0] > 50.0) and (sl[0] > 2.0*sl[1]):
                    #could be stuck pixel in one exposure but not another, especially if multiple observations
                        force_recompute = True
                        f.dqs = 0.0
                        f.dqs_bad = True

                        log.info('Detect ID # %d. Possible stuck/hot pixel in fiber %s (# in CCD = %d) '
                             'around wavelength = %0.1f (CCD X,Y = %d,%d) . Peak values = %s ...' %
                            (self.id,f.idstring, f.number_in_ccd,self.w,f.emis_x, f.emis_y, sl[0:3]))

        if force_recompute:
            log.info("Detect ID # %d . Triggering recompuatation of score ..." % self.id)
            return False


        title = ""
        old_score = self.dqs_raw
        new_score = old_score
        sk = -999
        ku = -999
        si = -999
        dx0 = -999
        rh = -999
        mx_norm = max(wave_counts)/100.0

        #fit around designated emis line
        if (fit_wave is not None) and (fiber_count > 0) :
            old_score = self.dqs_raw
            new_score = old_score
            sk = skew(fit_wave)
            ku = kurtosis(fit_wave) # remember, 0 is tail width for Normal Dist. ( ku < 0 == thinner tails)
            si = parm[1] #*1.9 #scale to angstroms
            dx0 = (parm[0]-self.w) #*1.9

            #si and ku are correlated at this scale, for emission lines ... fat si <==> small ku

            height_pix = max(wave_counts)
            height_fit = max(fit_wave)

            if height_pix > 0:
                rh = height_fit/height_pix
            else:
                log.info("Detect ID # %d . Minimum peak height (%f) too small. Score zeroed." % (self.id, height_pix))
                self.dqs_raw = 0.0
                old_score = 0.0
                new_score = 0.0
                rh = 0.0

            #todo: for lower S/N, sigma (width) can be less and still get bonus if fibers have larger separation

            #new_score:
            if (0.75 < rh < 1.25) and (error < 0.2) and (float(bad_pix)/float(fiber_count) < 2.0): # 1 bad pixel in each fiber is okay, but no more

                #central peak position
                if abs(dx0) > 1.9:  #+/- one pixel (in AA)  from center
                    val = (abs(dx0) - 1.9)** 2
                    new_score -= val
                    log.debug("Detect ID # %d. Penalty for excessive error in X0: %f" % (self.id, val))


                #sigma scoring
                if si < 2.0: # and ku < 2.0: #narrow and not huge tails
                    val = mx_norm*np.sqrt(2.0 - si)
                    new_score -= val
                    log.debug("Detect ID # %d. Penalty for low sigma: %f" % (self.id, val))
                    #note: si always > 0.0 and rarely < 1.0
                elif si < 2.5:
                    pass #zero zone
                elif si < 10.0:
                    val = np.sqrt(si-2.5)
                    new_score += val
                    log.debug("Detect ID # %d. Bonus for large sigma: %f" % (self.id, val))
                elif si < 15.0:
                    pass #unexpected, but lets not penalize just yet
                else: #very wrong
                    val = np.sqrt(si-15.0)
                    new_score -= val
                    log.debug("Detect ID # %d. Penalty for excessive sigma: %f" % (self.id, val))


                #only check the skew for smaller sigma
                #skew scoring
                if si < 2.5:
                    if sk < -0.5: #skew wrong directionn
                        val = min(1.0,mx_norm*min(0.5,abs(sk)-0.5))
                        new_score -= val
                        log.debug("Detect ID # %d. Penalty for low sigma and negative skew: %f" % (self.id,val))
                    if (sk > 2.0): #skewed a bit red, a bit peaky, with outlier influence
                        val = min(0.5,sk-2.0)
                        new_score += val
                        log.debug("Detect ID # %d. Bonus for low sigma and positive skew: %f" % (self.id, val))

                self.dqs_raw = new_score

                base_msg = "Emission # %d, Emis Fit dX0 = %g(AA), RH = %0.2f, rms = %0.2f, Sigma = %g(AA), Skew = %g , Kurtosis = %g : Score Change = %g" \
                       % (self.id, dx0, rh, error, si, sk, ku, new_score - old_score)
                log.info(base_msg)
            elif rh > 0.0:
                #todo: based on rh and error give a penalty?? maybe scaled by maximum pixel value? (++val = ++penalty)

                if (error > 0.3) and (0.75 < rh < 1.25): #really bad rms, but we did capture the peak
                    val = mx_norm*(error - 0.3)
                    new_score -= val
                    log.debug("Detect ID # %d. Penalty for excessively bad rms: %f" % (self.id,val))
                elif rh < 0.6: #way under shooting peak (should be a wide sigma) (peak with shoulders?)
                    val = mx_norm * (0.6 - rh)
                    new_score -= val
                    log.debug("Detect ID # %d. Penalty for excessive undershoot peak: %f" % (self.id, val))
                elif rh > 1.4: #way over shooting peak (super peaky ... prob. hot pixel?)
                    val = mx_norm * (rh - 1.4)
                    new_score -= val
                    log.debug("Detect ID # %d. Penalty for excessively overshoot peak: %f" % (self.id, val))

                    self.dqs_raw = new_score
            else:
                log.info("Emission # %d, too many bad pixels or failure to fit peak or overall bad fit. No additional score change (%g)" % (self.id,old_score))

        #todo: temporary ... comment out or mark as if False
        if self.plot_dqs_fit or G.PLOT_GAUSSIAN:
            if wide:
                title += "(Wide) "
            else:
                title += "(Narrow) "

            if error is None:
                error = -1
            title += "ID #%d, Old Score = %0.2f , New Score = %0.2f (%0.1f)\n" \
                     "dX0 = %0.2f, RH = %0.2f, RMS = %f\n"\
                     "Sigma = %0.2f, Skew = %0.2f, Kurtosis = %0.2f\n"\
                      % (self.id, old_score, new_score, self.dqs_calc_scaled_score(new_score),
                         dx0, rh, error, si, sk, ku)

            fig = plt.figure()
            gauss_plot = plt.axes()

            gauss_plot.plot(wave_x,wave_counts,c='k')

            if fit_wave is not None:
                if wide:
                    gauss_plot.plot(xfit,fit_wave,c='r')
                else:
                    gauss_plot.plot(xfit, fit_wave, c='b')
                    gauss_plot.grid(True)

                ymin = min(min(fit_wave),min(wave_counts))
                ymax = max(max(fit_wave),max(wave_counts))
            else:
                ymin = min(wave_counts)
                ymax = max(wave_counts)
            gauss_plot.set_ylabel("Summed Counts")
            gauss_plot.set_xlabel("Wavelength $\AA$ ")

            ymin *= 1.1
            ymax *= 1.1

            if abs(ymin) < 1.0: ymin = -1.0
            if abs(ymax) < 1.0: ymax = 1.0

            gauss_plot.set_ylim((ymin,ymax))
            gauss_plot.set_xlim( (np.floor(wave_x[0]),np.ceil(wave_x[-1])) )
            gauss_plot.set_title(title)
            png = 'gauss_' + str(self.entry_id).zfill(3) + "_d" + str(self.id) + ".png"
            if self.outdir is not None:
                png = op.join(self.outdir,png)
            log.info('Writing: ' + png)
            print('Writing: ' + png)
            fig.tight_layout()
            fig.savefig(png)
            fig.clear()
            plt.close()
            # end plotting

        else:
            log.info("Emission # %d -- unable to fit gaussian" % (self.id))

        if force_recompute:
            return False
        else:
            return True


    def dqs_calc_scaled_score(self,raw=None):
        # 5 point scale
        # A+ = 5.0
        # A  = 4.0
        # B+ = 3.5
        # B  = 3.0
        # C+ = 2.5
        # C  = 2.0
        # D+ = 1.5
        # D  = 1.0
        # F  = 0

        a_p = 14.0
        a__ = 12.5
        a_m = 11.0
        b_p = 8.0
        b__ = 7.0
        c_p = 6.0
        c__ = 5.0
        d_p = 4.0
        d__ = 3.0
        f__ = 2.0

        if raw is None:
            hold = True
            raw = self.dqs_raw
        else:
            hold = False

        if   raw > a_p : score = 5.0  #A+
        elif raw > a__ : score = 4.5 + 0.5*(raw-a__)/(a_p-a__) #A
        elif raw > a_m : score = 4.0 + 0.5*(raw-a_m)/(a__-a_m) #A-
        elif raw > b_p : score = 3.5 + 0.5*(raw-b_p)/(a_m-b_p) #B+ AB
        elif raw > b__ : score = 3.0 + 0.5*(raw-b__)/(b_p-b__) #B
        elif raw > c_p : score = 2.5 + 0.5*(raw-c_p)/(b__-c_p) #C+ BC
        elif raw > c__ : score = 2.0 + 0.5*(raw-c__)/(c_p-c__) #C
        elif raw > d_p : score = 1.5 + 0.5*(raw-d_p)/(c__-d_p) #D+ CD
        elif raw > d__ : score = 1.0 + 0.5*(raw-d__)/(d_p-d__) #D
        elif raw > f__ : score = 0.5 + 0.5*(raw-f__)/(d__-f__) #F
        elif raw > 0.0 : score =  0.5*raw/f__
        else: score = 0.0

        score = round(score,1)

        bad_pix = 0
        for f in self.fibers:
            bad_pix += f.central_wave_pixels_bad

        #allow only a single pixel to fall out of our range
        if float(bad_pix)/len(self.fibers) > 1.0:
            if hold:
                log.info("Detect ID %d maximum score limited by bad (wavelength edge) pixels.")
            score = min(3.5,score)

        if hold:
            log.info("Detect ID # %d, Scale Score = %0.1f (raw = %f)" % (self.id,score,raw))

        if hold:
            self.dqs = score

        return score

    def get_probabilities(self):

        #if we have a flux calibrated, already classified spectra, use that
        #(this is the normal case)
        if self.spec_obj.p_lae_oii_ratio is not None:
            self.p_lae = self.spec_obj.p_lae
            self.p_oii = self.spec_obj.p_oii
            self.p_lae_oii_ratio = self.spec_obj.p_lae_oii_ratio

        #otherwise, build with what we've got (note: does not have additional lines, in this case)
        elif self.w < 10: #really < 3500 is nonsense
            log.info("Warning. Cannot calculate p(LAE) ... no wavelength defined.")
            #can't calculate ... no wavelength
            self.p_lae = -1
            self.p_oii = -1
            self.p_lae_oii_ratio = -1
        else:
            ratio, self.p_lae, self.p_oii = line_prob.prob_LAE(wl_obs=self.w,
                                                               lineFlux=self.estflux,
                                                               ew_obs=(self.eqw_obs),
                                                               c_obs=None, which_color=None,
                                                               addl_fluxes=None, addl_wavelengths=None,
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
                #log.warn("Unexpected fiber id string: %s" % fiber)
            return False

        newfiber = voltron_fiber.Fiber(fiber, detect_id=self.id)

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

        self.fibers.append(voltron_fiber.Fiber(idstring,specid,ifuslot,ifuid,amp,dither_date,dither_time,dither_time_extended,
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

    def __init__(self,args,fcsdir_list=None):
        #fcsdir may be a single dir or a list
        if args is None:
            log.error("Cannot construct HETDEX object. No arguments provided.")
            return None

        if args.score:
            self.plot_dqs_fit = True
        else:
            self.plot_dqs_fit = False

        self.output_filename = args.name

        self.multiple_observations = False #set if multiple observations are used (rather than a single obsdate,obsid)
        self.ymd = None
        self.target_ra = args.ra
        self.target_dec = args.dec
        self.target_err = args.error

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

        if args.cure:
            self.panacea = False
        else:
            self.panacea = True


        self.emission_lines = voltron_spectrum.Spectrum().emission_lines

        # self.emission_lines = [EmissionLine("Ly$\\alpha$ ",1216,'red'),
        #                        EmissionLine("OII ",3727,'green'),
        #                        EmissionLine("OIII",4959,"lime"), EmissionLine("OIII",5007,"lime"), #5007 is the primary
        #                        EmissionLine("CIII", 1909, "purple"),
        #                        EmissionLine("CIV ",1549,"black"),
        #                        EmissionLine("H$\\beta$ ",4861,"blue"),
        #                        EmissionLine("HeII", 1640, "orange"),
        #                        EmissionLine("MgII", 2798, "magenta",solution=False),
        #                        EmissionLine("H$\\gamma$ ", 4341, "royalblue",solution=False),
        #                        EmissionLine("NV ",1240,"teal",solution=False),
        #                        EmissionLine("SiII",1260,"gray",solution=False)]

        #self.panacea_fits_list = [] #list of all panacea multi_*fits files (all dithers) (should be 4amps x 3dithers)

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
        if (args.obsdate is None) and (self.detectline_fn is not None):  # this is optional
            self.read_detectline(force=True)
        elif (self.fcsdir is not None) or (len(self.fcsdir_list) > 0):
            #we have either fcsdir and fcs_base or fcsdir_list
            #consume the rsp1 style directory(ies)
            #DetObj(s) will be built here (as they are built, downstream from self.read_detectline() above
            self.read_fcsdirs()
            build_fits_list = False

        if build_fits_list:
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

        if (args.ra is not None) and (args.dec is not None):
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
                    #only using the ifu package to get the path. voltron does not use entire IFU/observations
                    #just the specific fibers for each detection
                    info = voltron_ifu.find_panacea_path_info(fib.dither_date,
                                                              fib.dither_time_extended,
                                                              fib.dither_time,
                                                              G.PANACEA_RED_BASEDIR)

                    if (len(info) == 0) and (G.PANACEA_RED_BASEDIR != G.PANACEA_RED_BASEDIR_DEFAULT):
                        #try again with the default
                        info = voltron_ifu.find_panacea_path_info(fib.dither_date,
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
                #todo: build up the tokens that DetObj needs?
                toks = None
                e = DetObj(toks, emission=True, fcsdir=d)
                if e is not None:
                    G.UNIQUE_DET_ID_NUM += 1
                    #for consistency with Karl's namine, the entry_id is the _xxx number at the end
                    if e.entry_id is None or e.entry_id == 0:
                        e.entry_id = G.UNIQUE_DET_ID_NUM

                    e.id = G.UNIQUE_DET_ID_NUM
                    e.load_fluxcalibrated_spectra()
                    if e.status >= 0:
                        self.emis_list.append(e)
        elif (self.fcs_base is not None and self.fcsdir is not None): #not the usual case
            toks = None
            e = DetObj(toks, emission=True, fcs_base=self.fcs_base,fcsdir=self.fcsdir)
            if e is not None:
                G.UNIQUE_DET_ID_NUM += 1
                if e.entry_id is None or e.entry_id == 0:
                    e.entry_id = G.UNIQUE_DET_ID_NUM

                e.id = G.UNIQUE_DET_ID_NUM
                e.load_fluxcalibrated_spectra()
                if e.status >= 0:
                    self.emis_list.append(e)


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
                                    self.emis_list.append(e)
                            else: #if is 'none' so they all go here ... must assume same IFU
                                self.emis_list.append(e)
                    else:
                        if (self.ifu_slot_id is not None):
                            if (str(e.ifuslot) == str(self.ifu_slot_id)):
                                self.emis_list.append(e)
                        else:
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

                    e.plot_dqs_fit = self.plot_dqs_fit

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
                                    self.emis_list.append(e)
                            else: #if is 'none' so they all go here ... must assume same IFU
                                self.emis_list.append(e)
                    else:
                        if (e.sigma >= self.sigma) and (e.chi2 <= self.chi2):
                            if (self.ifu_slot_id is not None):
                                if (str(e.ifuslot) == str(self.ifu_slot_id)):
                                    self.emis_list.append(e)
                            else:
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


    def build_hetdex_data_page(self,pages,detectid):

        e = self.get_emission_detect(detectid) #this is a DetObj
        if e is None:
            log.error("Could not identify correct emission to plot. Detect ID = %d" % detectid)
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
            gs = gridspec.GridSpec(2, 40)
        else:
            if G.SHOW_FULL_2D_SPECTRA:
                gs = gridspec.GridSpec(5, 40)#, wspace=0.25, hspace=0.5)
            else:
                gs = gridspec.GridSpec(3, 40)

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

        if self.output_filename is not None:
            title = "%s_%s.pdf\n" % (self.output_filename, str(e.entry_id).zfill(3))
        else:
            title = "" #todo: start with filename

        try:
            title += "Obs: " + e.fibers[0].dither_date + "v" + str(e.fibers[0].obsid).zfill(2) + "_" + \
            str(e.fibers[0].detect_id) + "\n"
        except:
            log.debug("Exception building observation string.",exc_info=True)

        if (e.entry_id is not None) and (e.id is not None):
            title += "Entry# (%d), Detect ID (%d)" %(e.entry_id, e.id)
            if e.line_number is not None:
                title += ", Line# (%d)" % (e.line_number)

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
        e.get_probabilities()

        if e.w > 0:
            la_z = e.w / G.LyA_rest - 1.0
            oii_z = e.w / G.OII_rest - 1.0
        else:
            la_z = 1
            oii_z = 1

        if self.ymd and self.obsid:

            if not G.ZOO:
                title +="\n"\
                    "ObsDate %s  ObsID %s IFU %s  CAM %s\n" \
                    "Science file(s):\n%s"\
                    "RA,Dec (%f,%f) \n"\
                    "Sky X,Y (%f,%f)\n" \
                    "$\lambda$ = %g$\AA$  FWHM = %g$\AA$\n" \
                    "EstFlux = %0.3g"   \
                    % (self.ymd, self.obsid, self.ifu_slot_id,self.specid,sci_files, ra, dec, e.x, e.y,e.w,e.fwhm,
                        e.estflux )

                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux,e.fluxfrac)
                else:
                    title += "\n"
                title +=  "EstCont = %0.3g  EW_r(LyA) = %0.3g$\AA$\n" \
                    %(e.cont_cgs, e.eqw_obs/(1.0+la_z))

            else:  #this if for zooniverse, don't show RA and DEC or Probabilitie
                title += "\n" \
                     "ObsDate %s  ObsID %s IFU %s  CAM %s\n" \
                     "Science file(s):\n%s" \
                     "Sky X,Y (%f,%f)\n" \
                     "$\lambda$ = %g$\AA$  FWHM = %g$\AA$\n" \
                     "EstFlux = %0.3g" \
                             % (self.ymd, self.obsid, self.ifu_slot_id, self.specid, sci_files, e.x, e.y, e.w,e.fwhm,e.estflux)  # note: e.fluxfrac gauranteed to be nonzero
                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux, e.fluxfrac)
                else:
                    title += "\n"

                title += "EstCont = %0.3g  EW_r(LyA) = %0.3g$\AA$\n" % (e.cont_cgs, e.eqw_obs/(1.0+la_z))


        else:
            if not G.ZOO:
                title += "\n" \
                     "Primary IFU Slot %s\n" \
                     "RA,Dec (%f,%f) \n" \
                     "Sky X,Y (%f,%f)\n" \
                     "$\lambda$ = %g$\AA$  FWHM = %g$\AA$\n" \
                     "EstFlux = %0.3g" \
                     % (e.fibers[0].ifuslot, ra, dec, e.x, e.y, e.w,e.fwhm, e.estflux)

                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux,e.fluxfrac)
                else:
                    title += "\n"
                title +=  "EstCont = %0.3g  EW_r(LyA) = %0.3g$\AA$\n" % (e.cont_cgs,e.eqw_obs/(1.0+la_z))

            else: #this if for zooniverse, don't show RA and DEC or probabilities
                title += "\n" \
                     "Primary IFU Slot %s\n" \
                     "Sky X,Y (%f,%f)\n" \
                     "$\lambda$ = %g$\AA$  FWHM = %g$\AA$\n" \
                     "EstFlux = %0.3g " \
                     % ( e.fibers[0].ifuslot, e.x, e.y, e.w,e.fwhm, e.estflux)

                if e.dataflux > 0: # note: e.fluxfrac gauranteed to be nonzero
                    title += "DataFlux = %g/%0.3g\n" % (e.dataflux,e.fluxfrac)
                else:
                    title += "\n"
                title +=  "EstCont = %0.3g  EW_r(LyA) = %0.3g$\AA$\n" % (e.cont_cgs,e.eqw_obs/(1.0+la_z))


        if self.panacea:
            snr = e.sigma
            if (e.snr is not None) and (e.snr != 0.0):
                snr = e.snr
            title += "S/N = %g " % (snr)
        else:
            title += "$\sigma$ = %g " % (e.sigma)

        if (e.chi2 is not None) and (e.chi2 != 666) and (e.chi2 != 0):
            title += " $\chi^2$ = %g" % (e.chi2)


        if e.dqs is None:
            e.dqs_score()
      #  title += "  Score = %0.1f (%0.2f)" % (e.dqs,e.dqs_raw)

        if not G.ZOO:
            if e.p_lae_oii_ratio is not None:
                title += "\nP(LAE)/P(OII) = %0.3g" % (e.p_lae_oii_ratio)
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
            good, p_good = e.multiline_solution_score()
            if ( good ):
                # strong solution
                sol = datakeep['detobj'].spec_obj.solutions[0]
                title += "\n*(%0.3f) %s(%d) z = %0.4f  EW_r = %0.1f$\AA$" %(p_good, sol.name, int(sol.central_rest),sol.z,
                                                                        e.eqw_obs/(1.0+sol.z))
            elif (p_good > 0.0):
                #weak solution ... for display only, not acceptabale as a solution
                #do not set the solution (sol) to be recorded
                sol = datakeep['detobj'].spec_obj.solutions[0]
                title += "\nweak(%s(%d) z = %0.4f  EW_r = %0.1f$\AA$)" % \
                         ( sol.name, int(sol.central_rest), sol.z,e.eqw_obs / (1.0 + sol.z))
            else:
                log.info("No singular, strong emission line solution.")


        #plt.subplot(gs[0:2, 0:3])
        plt.subplot(gs[0:2, 0:10])
        plt.text(0, 0.5, title, ha='left', va='center', fontproperties=font)
        plt.suptitle(time.strftime("%Y-%m-%d %H:%M:%S") +
                     "  Version " + G.__version__ +"  ", fontsize=8,x=1.0,y=0.98,
                     horizontalalignment='right',verticalalignment='top')
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if datakeep is not None:
            img_y = None
            if datakeep['xi']:
                try:
                    plt.subplot(gs[0:2,10:24])
                    plt.gca().axis('off')
                    buf,img_y = self.build_2d_image(datakeep)

                    buf.seek(0)
                    im = Image.open(buf)
                    plt.imshow(im,interpolation='none') #needs to be 'none' else get blurring
                except:
                    log.warning("Failed to 2D cutout image.", exc_info=True)

                # update emission with the ra, dec of all fibers
                # needs to be here, after build_2d_image so the 'index' and 'color' exist for assignment
                try:
                    e.fiber_locs = list(
                        zip(datakeep['ra'], datakeep['dec'], datakeep['color'], datakeep['index'], datakeep['d'],
                            datakeep['fib']))
                except:
                    log.error("Error building fiber_locs", exc_info=True)

                try:
                    plt.subplot(gs[0:2,24:27])
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
                    plt.subplot(gs[0:2,27:30])
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
                    plt.subplot(gs[0:2,30:])
                    plt.gca().axis('off')
                    buf = self.build_spec_image(datakeep,e.w, dwave=1.0)
                    buf.seek(0)
                    im = Image.open(buf)
                    plt.imshow(im,interpolation='none')#needs to be 'none' else get blurring
                except:
                    log.warning("Failed to build spec image.",exc_info = True)


                if G.SINGLE_PAGE_PER_DETECT:
                    #make the first part is own (temporary) page (to be merged later)
                    pages.append(fig)
                    plt.close('all')
                    try:
                        if G.SHOW_FULL_2D_SPECTRA:
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
                        log.warning("Failed to build full width spec/cutout image.", exc_info=True)


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

        #safety check
        # update emission with the ra, dec of all fibers
        if e.fiber_locs is None:
            try:
                e.fiber_locs = list(
                    zip(datakeep['ra'], datakeep['dec'], datakeep['color'], datakeep['index'], datakeep['d'],
                        datakeep['fib']))
            except:
                log.error("Error building fiber_locs", exc_info=True)

        if not G.SINGLE_PAGE_PER_DETECT:
            pages.append(fig)
            plt.close()
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
            log.info("Exception in science_image::get_vrange:",exc_info =True)

        return vmin, vmax

    def clean_data_dict(self,datadict=None):
        if datadict is not None:
            dd = datadict
            for k in dd.keys():
                del dd[k][:]
        else:
            dd = {}
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
            dd['spec_id'] = []
            dd['xi'] = []
            dd['yi'] = []
            dd['xl'] = []
            dd['yl'] = []
            dd['xh'] = []
            dd['yh'] = []
            dd['ds9_x'] = []
            dd['ds9_y'] = []
            dd['sn'] = []
            dd['fiber_sn'] = []
            dd['wscore'] = []
            dd['scatter'] = []
            dd['scatter_sky'] = []
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
            dd['pix'] = []
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
        if self.panacea:
            datakeep = self.build_panacea_hetdex_data_dict(detobj)
        else:
            datakeep = self.build_hetdex_data_dict(detobj)

        detobj.dqs_score() #force_recompute=True)
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
                fiber = voltron_fiber.Fiber(op.basename(sci.filename), str(self.specid), str(self.ifu_slot_id), str(self.ifu_id),
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

                    if e.wra:
                        fiber.dqs_score(e.wra,e.wdec)
                    else:
                        fiber.dqs_score(e.ra,e.dec)
                    datakeep['wscore'].append(fiber.dqs)

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
                buf = flip_amp(amp,pyfits.open(pix_fn)[0].data)
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

                datakeep['fw_spec'].append(sci.fe_data[loc,:])
                datakeep['fw_specwave'].append(wave[:])

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

        #for loc in locations:
        for item in sort_list:
            fits = item.fits

            if not fits:
                #something is very wrong
                log.error("Unexpected None fits in hetdex::build_panacea_hetdex_data_dict")

            dither = fits.dither_index  # 0,1,2 or more
            loc = item.loc
            fiber = item.fiber

            if fiber is None: # did not find it? impossible?
                log.error("Error! Cannot identify fiber in HETDEX:build_panacea_hetdex_data_dict().")
                fiber = voltron_fiber.Fiber(0,0,0,0,'XX',"","","",-1,-1)


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

            d = self.emis_to_fiber_distance(e,fiber)
            if d is not None:
                datakeep['d'].append(d)
            else:
                datakeep['d'].append(item.dist)

            if e.wra:
                fiber.dqs_score(e.wra, e.wdec)
            else:
                fiber.dqs_score(e.ra, e.dec)
            datakeep['wscore'].append(fiber.dqs)

            x_2D = np.interp(e.w,fits.wave_data[loc,:],range(len(fits.wave_data[loc,:])))
            y_2D = np.interp(x_2D,range(len(fits.trace_data[loc,:])),fits.trace_data[loc,:])

            fiber.emis_x = int(x_2D)
            fiber.emis_y = int(y_2D)

            try:
                log.info("Detect # %d, Fiber %s, Cam(%s), ExpID(%d) CCD X,Y = (%d,%d)" %
                         (e.id,fiber.idstring,fiber.specid,fiber.expid,int(x_2D),int(y_2D)))
            except:
                pass

            xl = int(np.round(x_2D - xw))
            xh = int(np.round(x_2D + xw))
            yl = int(np.round(y_2D - yw))
            yh = int(np.round(y_2D + yw))
            datakeep['ds9_x'].append(1. + (xl + xh) / 2.)
            datakeep['ds9_y'].append(1. + (yl + yh) / 2.)

            blank_xl = xl
            blank_xh = xh
            blank_yl = yl
            blank_yh = yh
            blank = np.zeros((yh-yl+1,xh-xl+1))
            scatter_blank = np.zeros((yh - yl + 1 + 10*yw, xh - xl + 1)) #10*yw because +/- 5*yw in height

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


            load_blank = False
            pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid, fits.amp))
            # specid (cam) in filename might not have leading zeroes
            if not op.exists(pix_fn) and (fits.specid[0] == '0'):
                log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid.lstrip("0"), fits.amp))

            if op.exists(pix_fn):
                buf = flip_amp(fits.amp, pyfits.open(pix_fn)[0].data)

                if buf is not None:
                    blank[(yl - blank_yl):(yl - blank_yl) + (yh-yl) + 1, (xl - blank_xl):(xl - blank_xl) + (xh - xl) + 1] = \
                        buf[yl:yh + 1, xl:xh + 1]
                    datakeep['pix'].append(deepcopy(blank))

                else:
                    load_blank = True
            else:
                load_blank = True

            if load_blank:
                # todo: this is really sloppy ... make a better/more efficient pattern
                log.error("Could not find pixel flat: %s . Retry w/o leading 0" % pix_fn)
                datakeep['pix'].append(deepcopy(blank_pixel_flat(xh, xl, yh, yl)))

                #pix_x = xh - xl + 1
                #pix_y = yh - yl + 1
                #pix_blank = np.zeros((pix_y, pix_x))
                #try:
                #    for x in range(pix_x / 2):
                #        for y in range(pix_y / 2):
                #            pix_blank[y * 2, x * 2] = 999
                #except:
                #    pass
                #datakeep['pix'].append(deepcopy(pix_blank))



            #1D spectrum (spec is counts, specwave is the corresponding wavelength)
            wave = fits.wave_data[loc,:]
            # this sometimes produces arrays of different lengths (+/- 1) [due to rounding?]
            # which causes problems later on, so just get the nearst point to the target wavelenght
            # and a fixed number of surrounding pixels

            #Fe_indl = np.searchsorted(wave, e.w - ww, side='left')
            #Fe_indh = np.searchsorted(wave, e.w + ww, side='right')
            #want say, approx +/- 50 angstroms

            center = np.searchsorted(wave, e.w, side='left')
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
                y_2D, x_2D = np.shape(blank)




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

        cmap = plt.get_cmap('gray_r')
#        norm = plt.Normalize()
#        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))

       # colors[0] = (1,0,0,1)
       # colors[1] = (1, 0, 0, 1)
       # colors[2] = (1, 0, 0, 1)

        colors = self.make_fiber_colors(min(4,len(datakeep['ra'])),len(datakeep['ra']))# + 2 ) #the +2 is a pad in the call
        num_fibers = len(datakeep['xi'])
        num_to_display = min(MAX_2D_CUTOUTS,num_fibers) + add_summed_image #for the summed images
        bordbuff = 0.01
        borderxl = 0.05
        borderxr = 0.15
        borderyb = 0.05
        borderyt = 0.15

        #the +1 for the summed image
        dx = (1. - borderxl - borderxr) / 3.
        dy = (1. - borderyb - borderyt) / (num_to_display)
        dx1 = (1. - borderxl - borderxr) / 3.
        dy1 = (1. - borderyb - borderyt - (num_to_display) * bordbuff) / (num_to_display)
        Y = (yw / dy) / (xw / dx) * 5.

        Y = max(Y,0.8) #set a minimum size

        fig = plt.figure(figsize=(5, Y), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        # previously sorted in order from largest distances to smallest
        ind = list(range(len(datakeep['d'])))

        #assume all the same shape
        summed_image = np.zeros(datakeep['im'][ind[0]].shape)

        #need i to start at zero
        #building from bottom up
        grid_idx = -1
        for i in range(num_fibers+add_summed_image):
            make_display = False
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

                    image = datakeep['im'][ind[i]]  # im can be the cosmic removed version, depends on G.PreferCosmicCleaned
                    cmap1 = cmap
                    cmap1.set_bad(color=[0.2, 1.0, 0.23])
                    image = np.ma.masked_where(datakeep['err'][ind[i]] == -1, image)
                    img_vmin = datakeep['vmin2'][ind[i]]
                    img_vmax = datakeep['vmax2'][ind[i]]

                    #plot_label = str(num_fibers-i)
                    plot_label = str("%0.2f" % datakeep['fiber_weight'][ind[i]]).lstrip('0') #save space, kill leading 0

                    #the last one is the top one and is the primary
                    datakeep['primary_idx'] = ind[i]

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
                borplot = plt.axes([borderxl + 0. * dx, borderyb + grid_idx * dy, 3 * dx, dy])
                smplot = plt.axes([borderxl + 2. * dx - bordbuff / 3., borderyb + grid_idx * dy + bordbuff / 2., dx1, dy1])
                pixplot = plt.axes(
                    [borderxl + 1. * dx + 1 * bordbuff / 3., borderyb + grid_idx * dy + bordbuff / 2., dx1, dy1])
                imgplot = plt.axes([borderxl + 0. * dx + bordbuff / 2., borderyb + grid_idx * dy + bordbuff / 2., dx1, dy1])
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
                                   extent=ext) #vmin=0.9, vmax=1.1

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


                    borplot.text(-0.2, .5, plot_label,
                            transform=imgplot.transAxes, fontsize=10, color='k', #colors[i, 0:3],
                            verticalalignment='bottom', horizontalalignment='left')

                #if self.multiple_observations:
                #add the fiber info to the right of the images
                if not G.ZOO:
                    if is_a_fiber:
                        if self.panacea:
                            #dither and fiber position, etc generally meaningless in this case
                            #as there is no good way to immediately go back and find the source image
                            #so just show S/N and distance (and make bigger)
                            if sn > 0:
                                if abs(sn) < 1000:
                                    borplot.text(1.05, .75, 'SN: %0.2f' % (sn),
                                                 transform=smplot.transAxes, fontsize=8, color='r',
                                                 verticalalignment='bottom', horizontalalignment='left')
                                else:
                                    borplot.text(1.05, .75, 'SN: %.1E' % (sn),
                                                 transform=smplot.transAxes, fontsize=8, color='r',
                                                 verticalalignment='bottom', horizontalalignment='left')
                            # distance (in arcsec) of fiber center from object center
                            borplot.text(1.05, .53, 'D("): %0.2f %0.1f' % (datakeep['d'][ind[i]],datakeep['wscore'][ind[i]]),
                                         transform=smplot.transAxes, fontsize=6, color='r',
                                         verticalalignment='bottom', horizontalalignment='left')

                            try:
                                l3 = datakeep['date'][ind[i]] + "_" + datakeep['obsid'][ind[i]] + "_" + datakeep['expid'][ind[i]]
                                l4 = datakeep['spec_id'][ind[i]] + "_" + datakeep['amp'][ind[i]] + "_" + datakeep['fib_idx1'][ind[i]]

                                borplot.text(1.05, .33, l3,
                                             transform=smplot.transAxes, fontsize=6, color='b',
                                             verticalalignment='bottom', horizontalalignment='left')
                                borplot.text(1.05, .13, l4,
                                             transform=smplot.transAxes, fontsize=6, color='b',
                                             verticalalignment='bottom', horizontalalignment='left')
                            except:
                                log.error("Exception building extra fiber info.", exc_info=True)

                        else:
                            if sn > 0:
                                borplot.text(1.05, .75, 'S/N = %0.2f' % (sn),
                                            transform=smplot.transAxes, fontsize=6, color='r',
                                            verticalalignment='bottom', horizontalalignment='left')
                            #distance (in arcsec) of fiber center from object center
                            borplot.text(1.05, .55, 'D(") = %0.2f %0.1f' % (datakeep['d'][ind[i]],datakeep['wscore'][ind[i]]),
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
                        borplot.text(1.05, .35, '\nWEIGHTED\nSUM',
                                     transform=smplot.transAxes, fontsize=8, color='k',
                                     verticalalignment='bottom', horizontalalignment='left')

                if grid_idx == (num_to_display-1): #(num + add_summed_image - 1):
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
        plt.savefig(buf, format='png', dpi=300)

        if G.ZOO_CUTOUTS:
            try:
                fn = self.output_filename + "_" + str(datakeep['detobj'].entry_id).zfill(3) + "_zoo_2d_fib.png"
                fn = op.join( datakeep['detobj'].outdir, fn)
                plt.savefig(fn,format="png",dpi=300)
            except:
                log.error("Unable to write zoo_2d_fib image to disk.",exc_info=True)

        plt.close(fig)
        return buf, Y


    # +/- 3 fiber sizes on CCD (not spacially adjacent fibers)
    def build_scattered_light_image(self, datakeep, img_y = 3, key='scatter'):

            if not key in ['scatter','scatter_sky']:
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
            else: #not possible ... just here for sanity and future expansion
                log.error("Invalid key for build_scattered_light_image: %s" % key)
                return None

            bordbuff = 0.01

            borderxl = 0.05
            borderxr = 0.15

            borderyb = 0.05
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
                plt.title("%s\nx,y: %d,%d"
                          % (title,datakeep['ds9_x'][ind[datakeep_idx]], datakeep['ds9_y'][ind[datakeep_idx]]), fontsize=12)
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
                    zoo_num = "ccd_sub"
                    if key == "scatter_sky":
                        zoo_num = "ccd_sky"
                    fn = self.output_filename + "_" + str(datakeep['detobj'].entry_id).zfill(3) + "_zoo_" + zoo_num \
                         + ".png"
                    fn = op.join(datakeep['detobj'].outdir, fn)
                    plt.savefig(fn, format="png", dpi=300)
                except:
                    log.error("Unable to write zoo_%s image to disk." %(zoo_num), exc_info=True)

            plt.close(fig)
            return buf



    def build_spec_image(self,datakeep,cwave, dwave=1.0):
        #cwave = central wavelength, dwave = delta wave (wavestep in AA)

        fig = plt.figure(figsize=(5, 3))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
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
            min_y = max(min_y, -0.25*mx)#-20) #but not below -20

            log.debug("Spec Plot max count = %f , min count = %f" %(mx,mx))
            specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], ls='--', c=[0.3, 0.3, 0.3])
            specplot.axis([cwave - ww, cwave + ww, min_y, mx + ran / 20])

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
        plt.savefig(buf, format='png', dpi=300)

        if G.ZOO_CUTOUTS:
            try:
                fn = self.output_filename + "_" + str(datakeep['detobj'].entry_id).zfill(3) + "_zoo_1d_sum.png"
                fn = op.join(datakeep['detobj'].outdir, fn)
                plt.savefig(fn, format="png", dpi=300)
            except:
                log.error("Unable to write zoo_1d_sum image to disk.", exc_info=True)

        plt.close(fig)
        return buf

    def build_relative_fiber_locs(self, e):

        fig = plt.figure(figsize=(5, 3))
        #plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        fibplot = plt.axes()#[0.1, 0.2, 0.8, 0.8])

        fibplot.set_title("Relative Fiber Positions")
        #fibplot.set_xlabel("arcsecs")
        #plt.gca().xaxis.labelpad =

        fibplot.plot(0, 0, "r+")

        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')

        if e.wra:
            e_ra = e.wra
            e_dec = e.wdec
        else:
            e_ra = e.ra
            e_dec = e.dec

        for r, d, c, i, dist, fn in e.fiber_locs:
            # fiber absolute position ... need relative position to plot (so fiber - zero pos)
            fx = (r - e_ra) * np.cos(np.deg2rad(e_dec)) * 3600.
            fy = (d - e_dec) * 3600.

            xmin = min(xmin, fx)
            xmax = max(xmax, fx)
            ymin = min(ymin, fy)
            ymax = max(ymax, fy)

            fibplot.add_patch(plt.Circle((fx,fy), radius=G.Fiber_Radius, color=c, fill=False,
                                           linestyle='solid',zorder=9))
            fibplot.text(fx,fy, str(i), ha='center', va='center', fontsize='x-small', color=c)

            if fn in G.CCD_EDGE_FIBERS_ALL:
                fibplot.add_patch(
                    plt.Circle((fx, fy), radius=G.Fiber_Radius + 0.1, color=c, fill=False,
                               linestyle='dashed',zorder=9))

        # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
        ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        ext = ext_base + 2*G.Fiber_Radius

        rec = plt.Rectangle((-ext,-ext),width=ext*2, height=ext * 2, fill=True, lw=1,
                            color='gray', zorder=0, alpha=0.5)
        fibplot.add_patch(rec)

        fibplot.set_xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        fibplot.set_yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        fibplot.set_aspect('equal')

        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf


    #wide (full width) spectrum
    def build_full_width_spectrum (self, datakeep, cwave):

        cmap = plt.get_cmap('gray_r')
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))

        if G.SHOW_FULL_2D_SPECTRA:
            num = len(datakeep['xi'])
        else:
            num = 0
        dy = 1.0/(num +5)  #+ 1 skip for legend, + 2 for double height spectra + 2 for double height labels

        if G.SINGLE_PAGE_PER_DETECT:
            if G.SHOW_FULL_2D_SPECTRA:
                figure_sz_y = 2* G.GRID_SZ_Y
            else:
                figure_sz_y = G.GRID_SZ_Y
        else:
            if G.SHOW_FULL_2D_SPECTRA:
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

        N = len(datakeep['xi'])
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
                y_label = "cgs" #r"cgs [$10^{-17}$]"
 #               min_y = -2
            else:
                F = np.zeros(bigwave.shape)
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

            try:
                peak_idx = (np.abs(datakeep['sumspec_wave'] - cwave)).argmin()
                peak_height = datakeep['sumspec_flux'][peak_idx]

                mn = max(mn,-0.2*peak_height) #at most go -20% of the peak below zero (most likely a bad sky subtraction)
                mx = min(mx, 2.0 * peak_height)  # at most go 100% above the peak
            except:
                pass


            #flux at the cwave position
            #todo: this is wrong F-cwave makes no sense (F is a flux array, cwave is a wavelength)
            if True:
                line_mx = F[(np.abs(bigwave-cwave)).argmin()]
                if mx > 3.0*line_mx: #limit mx to a more narrow range)
                    mx = max(F[(np.abs(bigwave-3500.0)).argmin():(np.abs(bigwave-5500.0)).argmin()])
                    if mx > 3.0*line_mx:
                        log.info("Truncating max spectra value...")
                        mx = 3.0 * line_mx
                    else:
                        log.info("Exclusing spectra maximum outside 3500 - 5500 AA")

            ran = mx - mn
            #specplot.step(bigwave, F, c='b', where='mid', lw=1)
            specplot.plot(bigwave, F, c='b', lw=1)

            specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], ls='dashed', c='k') #[0.3, 0.3, 0.3])
            specplot.axis([left, right, mn - ran / 20, mx + ran / 20])
            # specplot.set_ylabel(y_label) #not honoring it, so just put in the text plot

            specplot.locator_params(axis='y',tight=True,nbins=4,y_label='cgs')

            textplot = plt.axes([border_buffer, (float(num)+3) * dy, 1.0 - (2 * border_buffer), dy*2 ])
            textplot.set_xticks([])
            textplot.set_yticks([])
            textplot.axis(specplot.axis())
            textplot.axis('off')

            #if this is flux, not counts, add a ersatz scale label for y axis
            if len( datakeep['sumspec_wave']) > 0:
                textplot.text(3500, textplot.axis()[2], "e-17", rotation=0, ha='left', va='bottom',
                          fontsize=10, color='k')  # use the e color for this family


            #
            #possibly add the matched line at cwave position
            #
            if not G.ZOO:
                good, p_real = datakeep['detobj'].multiline_solution_score()
                if (p_real > 0.0):
                    #a solution
                    sol = datakeep['detobj'].spec_obj.solutions[0]
                    y_pos = textplot.axis()[2]

                    if good:
                        textplot.text(cwave, y_pos, sol.name + " {", rotation=-90, ha='center', va='bottom',
                                      fontsize=24, color=sol.color)  # use the e color for this family
                    else: #weak solution, use standard font size
                        textplot.text(cwave, y_pos, sol.name + " {", rotation=-90, ha='center', va='bottom',
                                      color=sol.color)  # use the e color for this family

                    #highlight the matched lines

                    yl, yh = specplot.get_ylim()
                    for f in sol.lines:
                        hw = 3.0 * f.sigma #highlight half-width
                        # use 'y' rather than sols[0].color ... becomes confusing with black
                        rec = plt.Rectangle((f.w_obs - hw, yl), 2 * hw, yh - yl, fill=True, lw=1,
                                            color=sol.color, alpha=0.5,zorder=1)
                        specplot.add_patch(rec)
                #else: #redundant log ... already logged when preparing the upper left text block
                #    log.info("No singular, strong emission line solution.")


                #put dashed line through all possible emission lines (note: possible, not necessarily probable)
                if (datakeep['detobj'].spec_obj.all_found_lines is not None):
                    for f in datakeep['detobj'].spec_obj.all_found_lines: #this is an EmisssionLineInfo object
                        if f.prob_noise < 0.1:
                            x_pos = f.raw_x0
                            #y_pos = f.raw_h / 10.0 # per definition of EmissionLineInfo ... these are in 10^-18 cgs
                                                   # and these plots are 10^-17 cgs
                            #specplot.scatter(x_pos, y_pos, facecolors='None', edgecolors='b', zorder=99)
                            specplot.plot([x_pos, x_pos], [mn - ran * rm, mn + ran * (1 + rm)], ls='dashed', c='k',
                                          zorder=1,alpha=0.5)
                            #DEBUG:
                            if G.DEBUG_SHOW_GAUSS_PLOTS:
                                print("Line: %f (%f) " %(f.raw_x0,f.fit_x0))

            #
            #iterate over all emission lines ... assume the cwave is that line and plot the additional lines
            #
            wavemin = specplot.axis()[0]
            wavemax = specplot.axis()[1]
            legend = []
            name_waves = []
            obs_waves = []
            for e in self.emission_lines:
                if not e.solution:
                    continue
                z = cwave / e.w_rest - 1.0
                if (z < 0):
                    if z > -0.01:
                        z = 0.0
                    else:
                        continue
                count = 0
                for f in self.emission_lines:
                    if (f == e) or not (wavemin <= f.redshift(z) <= wavemax ):
                        continue

                    count += 1
                    y_pos = textplot.axis()[2]
                    for w in obs_waves:
                        if abs(f.w_obs - w) < 20: # too close, shift one vertically
                            #y_pos = textplot.axis()[2] + mn + ran*0.7
                            y_pos = (textplot.axis()[3] - textplot.axis()[2]) / 2.0 + textplot.axis()[2]
                            break

                    obs_waves.append(f.w_obs)
                    textplot.text(f.w_obs, y_pos, f.name+" {", rotation=-90, ha='center', va='bottom',
                                      fontsize=12, color=e.color)  # use the e color for this family

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

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)


        if G.ZOO_CUTOUTS:
            try:
                fn = self.output_filename + "_" + str(datakeep['detobj'].entry_id).zfill(3) + "_zoo_1d_full.png"
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


#end HETDEX class