
import global_config as G
import matplotlib
matplotlib.use('agg')

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


import glob
from pyhetdex.cure.distortion import Distortion
import pyhetdex.tools.files.file_tools as ft
from pyhetdex.het.ifu_centers import IFUCenter
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection import TangentPlane as TP
import os.path as op

log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

CONFIG_BASEDIR = G.CONFIG_BASEDIR
VIRUS_CONFIG = op.join(CONFIG_BASEDIR,"virus_config")
FPLANE_LOC = op.join(CONFIG_BASEDIR,"virus_config/fplane")
IFUCEN_LOC = op.join(CONFIG_BASEDIR,"virus_config/IFUcen_files")
DIST_LOC = op.join(CONFIG_BASEDIR,"virus_config/DeformerDefaults")
PIXFLT_LOC = op.join(CONFIG_BASEDIR,"virus_config/PixelFlats/20161223")

PLOT_SUMMED_SPECTRA = False

SIDE = ["L", "R"]
AMP  = ["LU","LL","RL","RU"] #in order from bottom to top
AMP_OFFSET = {"LU":1,"LL":113,"RL":225,"RU":337}

PANACEA_HDU_IDX = { 'image':0,
                    'error':1,
                    'clean_image':2,
                    'flat_image':3,
                    'continuum_sub':4,
                    'residual':5,
                    'ifupos':6,
                    'error_analysis':7,
                    'spectrum':8,
                    'wavelength':9,
                    'trace':10,
                    'fiber_to_fiber':11,
                    'twi_spectrum':12,
                    'sky_subtracted':13,
                    'sky_spectrum':14 }

#lifted from Greg Z.
dist_thresh = 2.  # Fiber Distance (arcsecs)
#todo: change to full width of frame?? (and change xl in build dict to this value, not the difference)
FRAME_WIDTH_X = 1024
xw = 24  # image width in x-dir
yw = 10  # image width in y-dir
contrast1 = 0.9  # convolved image
contrast2 = 0.5  # regular image
res = [3, 9]
ww = xw * 1.9  # wavelength width


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

class Fiber:
    def __init__(self,exp_id,specid,ifuslot,ifuid,amp,number,date,time,time_ex):

        self.exp_id = exp_id
        self.specid = specid
        self.ifuslot = ifuslot
        self.ifuid = ifuid
        self.amp = amp
        self.side = amp[0]
        self.number = int(number) #1 based
        self.dither_date = date
        self.dither_time = time
        self.dither_time_extended = time_ex
        self.sn = None

        self.dither_idx = None
        self.center_x = None
        self.center_y = None

class DetObj:
    '''mostly a container for an emission line or continuum detection from detect_line.dat or detect_cont.dat file'''
    #  0    1   2   3   4   5   6           7       8        9     10    11    12     13      14      15  16
    #  NR  ID   XS  XS  l   z   dataflux    modflux fluxfrac sigma chi2  chi2s chi2w  gammq   gammq_s eqw cont

    #cont
    # 0    1  2     3       4       5  6   7   8     9  10    11   12  13   14   15   16
    # ID  icx icy  <sigma> fwhm_xy  a  b   pa  ir1  ka  kb   xmin xmax ymin ymax zmin zmax

    #0   1  2                   3                 4        5   6      7     8 9    0 1 2 3 4  5   6 7
    #28 28 22.879905466813398 -11.019990953824617 3914.82 666 289.41 289.41 1 7.20 1 1 1 0 0 -300 0 ifu093


    def __init__(self,tokens,emission=True):
        #skip NR (0)
        self.type = 'unk'
        self.id = None
        self.x = None
        self.y = None
        self.w = 0.0
        self.la_z = 0.0
        self.dataflux = 0.0
        self.modflux = 0.0
        self.fluxfrac = 1.0
        self.sigma = 0.0
        self.chi2 = 0.0
        self.chi2s = 0.0
        self.chi2w = 0.0
        self.gammq = 0.0
        self.gammq_s = 0.0
        self.eqw = 0.0
        self.cont = -9999
        self.panacea = False

        self.ifuslot = None
        self.wra = None
        self.wdec = None

        self.fibers = []

        if emission:
            self.type = 'emis'
            self.id = int(tokens[1])
            self.x = float(tokens[2])
            self.y = float(tokens[3])
            self.w = float(tokens[4])
            self.la_z = float(tokens[5])
            self.dataflux = float(tokens[6])
            self.modflux = float(tokens[7])
            self.fluxfrac = float(tokens[8])
            self.estflux = self.dataflux * G.FLUX_CONVERSION/self.fluxfrac #estimated flux in cgs f_lambda
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
            self.eqw = float(tokens[15])
            self.cont = float(tokens[16])


            # 0  1   2 ....
            # 1  1  -16.39  -15.30 4615.18 666   285.7   285.7 1   7.2 1 1 1 0 0  -300 0
            # 17       18         19
            # ifu094  214.64391  52.80016
            # 20                                      21
            # 20170326T105655.6_032_094_028_LU_032   20170326T104328.6_032_094_028_LU_031
            # 20170326T105013.0_032_094_028_LU_032   20170326T105013.0_032_094_028_LU_012
            # 20170326T104328.6_032_094_028_LU_032

            try:
                if len(tokens) > 17: #this is probably an all ifu panacea version
                    self.panacea = True
                    self.ifuslot = str(tokens[17][-3:]) #ifu093 -> 093
                    if len(tokens) > 18:  # has the rest
                        self.wra = float(tokens[18])
                        self.wdec = float(tokens[19])

                        start = 20
                        for i in range(start,len(tokens)): #there are fibers and other stuff to follow
                            if not self.parse_fiber(tokens[i]): #this was not a fiber descriptor
                                break

                        self.cont = float(tokens[i])

                        start = i+1
                        for i in range(start,len(tokens)): #these are in the same order as fibers
                            sn = float(tokens[i])
                            for f in self.fibers:
                                if f.sn is None:
                                    f.sn = sn
                                    break

            except:
                log.info("Error parsing tokens from emission line file.",exc_info=True)

            if (self.eqw == -300) and (self.dataflux != 0) and (self.fluxfrac != 0) \
                    and (self.cont != 666) and (self.cont > 0):  # dummy value
                self.eqw = -1 * self.dataflux / self.fluxfrac / self.cont

        else:
            self.type = 'cont'
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


    def parse_fiber(self,fiber):
        if fiber is None:
            return False
        #20170326T105655.6_032_094_028_LU_032

        toks = fiber.split("_")

        if len(toks) != 6:
            if (len(toks) == 1) and (toks[0] == "666"):
                    return True #this is an "ignore" flag, but still continue as if it were a fiber
            else:
                log.warn("Unexpected fiber id string: %s" % fiber)
            return False

        exp_id = toks[0] #ie. 20170326T105655.6

        #todo: factor out into common call
        dither_date = exp_id[0:8]
        # next should be 'T'
        dither_time = exp_id[9:15]  # not the .# not always there
        if exp_id[15] == ".":
            dither_time_extended = exp_id[9:17]
        else:
            dither_time_extended = None

        specid = toks[1]
        ifuslot = toks[2]
        ifuid = toks[3]
        amp = toks[4]
        fiber_num = toks[5]

        #validate info
        if (ifuslot != self.ifuslot):
            log.error("Mismatched fiber id string. Does not match expected ifuslot id %s vs %s"
                      % (ifuslot,self.ifuslot))
            return True #this was still a fiber, just not one that is valid

        self.fibers.append(Fiber(exp_id,specid,ifuslot,ifuid,amp,fiber_num,dither_date,dither_time,dither_time_extended))

        return True

    def get_dither_number_for_fibers(self):
        #put fiber names in order (alphabetical also == time order)
        #assign dithers by that?
        pass


class HetdexFits:
    '''A single HETDEX fits file ... 2D spectra, expected to be science file'''

    #needs open with basic validation
    #

    def __init__(self,fn,e_fn,fe_fn,dither_index=-1,panacea=False):
        self.okay = True
        self.filename = fn
        self.err_filename = e_fn
        self.fe_filename = fe_fn

        self.panacea = panacea

        self.tel_ra = None
        self.tel_dec = None
        self.parangle = None
        self.ifuid = None # reminder this is the cable
        self.ifuslot = None # reminder this is the position (slot) on the fplane
        self.side = None
        self.amp = None
        self.specid = None
        self.obs_date = None
        self.obs_ymd = None
        self.mjd = None
        self.obsid = None
        self.imagetype = None
        #self.exptime = None #don't need these right now
        #self.dettemp = None #don't need these right now

        self.data = None
        self.err_data = None
        self.fe_data = None
        self.wave_data = None
        self.trace_data = None
        self.pixflat_data = None
        self.fiber_centers = None
        self.fe_crval1 = None
        self.fe_cdelt1 = None

        self.dither_index = dither_index

        #determine if 'cure'-style fits or panacea fits
        #stupid simple just for now
        if "multi_" in self.filename: # example: multi_020_095_004_LU.fits
            self.read_panacea_fits()
        else:
            self.read_fits(use_cosmic_cleaned=G.PreferCosmicCleaned)
            self.read_efits(use_cosmic_cleaned=G.PreferCosmicCleaned)
            self.read_fefits()

    def read_fits(self,use_cosmic_cleaned=False):

        if self.filename is None:
            return None

        try:
            f = pyfits.open(self.filename)
        except:
            log.error("could not open file " + self.filename, exc_info=True)
            return None

        c = None
        try:
            if use_cosmic_cleaned:
                base = op.basename(self.filename)
                if base[0] != 'c':
                    path = op.dirname(self.filename)

                    cosmic = op.join(path,"c"+base)
                    log.info("Attempting to open cosmic cleaned version of file: " + cosmic)
                    c = pyfits.open(cosmic)
        except:
            log.error("could not open file " + cosmic, exc_info=True)
            c = None


        if c is not None:
            self.data = np.array(c[0].data)
        else:
            self.data = np.array(f[0].data)
        #clean up any NaNs
        self.data[np.isnan(self.data)] = 0.0

        try:
            self.tel_ra = float(Angle(f[0].header['TELRA']+"h").degree) #header is in hour::mm:ss.arcsec
            self.tel_dec = float(Angle(f[0].header['TELDEC']+"d").degree) #header is in deg:hh:mm:ss.arcsec
            self.parangle = f[0].header['PARANGLE']
        except:
            log.error("Cannot translate RA and/or Dec from FITS format to degrees in " + self.filename, exc_info=True)

        try:
            self.ifuid = str(f[0].header['IFUID']).zfill(3)
            self.ifuslot = str(f[0].header['IFUSLOT']).zfill(3)
            self.side = f[0].header['CCDPOS']
            self.specid = str(f[0].header['SPECID']).zfill(3)
            self.obs_date = f[0].header['DATE-OBS']

            if '-' in self.obs_date: #expected format is YYYY-MM-DD
                self.obs_ymd = self.obs_date.replace('-','')
            self.mjd = f[0].header['MJD']
            self.obsid = f[0].header['OBSID']
            self.imagetype = f[0].header['IMAGETYP']
            #self.exptime = f[0].header['EXPTIME']
            #self.dettemp = f[0].header['DETTEMP']
        except:
            log.error("Cannot read expected keywords in fits header: " + self.filename,exc_info=True)
            self.okay = False

        try:
            f.close()
        except:
            log.error("could not close file " + self.filename, exc_info=True)

        if c is not None:
            try:
                c.close()
            except:
                log.error("could not close file cosmic file version of " + self.filename, exc_info=True)
        return

    def read_efits(self,use_cosmic_cleaned=False):

        if self.err_filename is None:
            return None

        try:
            f = pyfits.open(self.err_filename)
        except:
            log.error("could not open file " + self.err_filename, exc_info=True)
            return None

        c = None
        try:
            if use_cosmic_cleaned:
                #for simplicity, using self.filename instead of self.err_filename
                #since will assume err_filename = "e." + self.filename
                base = op.basename(self.filename)
                if base[0] != 'c':
                    path = op.dirname(self.err_filename)

                    cosmic = op.join(path, "e.c" + base)
                    log.info("Attempting to open cosmic cleaned version of file: " + cosmic)
                    c = pyfits.open(cosmic)
        except:
            log.error("could not open file " + cosmic, exc_info=True)
            c = None

        if c is not None:
            self.err_data = np.array(c[0].data)
        else:
            self.err_data = np.array(f[0].data)

        # clean up any NaNs
        self.err_data[np.isnan(self.err_data)] = 0.0

        try:
            f.close()
        except:
            log.error("could not close file " + self.err_filename, exc_info=True)

        if c is not None:
            try:
                c.close()
            except:
                log.error("could not close file cosmic file version of " + self.filename, exc_info=True)

        return

    def read_fefits(self):

        if self.fe_filename is None:
            return None

        try:
            f = pyfits.open(self.fe_filename)
        except:
            log.error("could not open file " + self.fe_filename, exc_info=True)
            return None

        try:
            self.fe_data = np.array(f[0].data)
            # clean up any NaNs
            self.fe_data[np.isnan(self.fe_data)] = 0.0
            self.fe_crval1 = f[0].header['CRVAL1']
            self.fe_cdelt1 = f[0].header['CDELT1']
        except:
            log.error("could not read values or data from file " + self.fe_filename, exc_info=True)

        try:
            f.close()
        except:
            log.error("could not close file " + self.fe_filename, exc_info=True)

        return

    def read_panacea_fits(self):
        #this represents one AMP
        #15 hdus, different header keys
        try:
            f = pyfits.open(self.filename)
        except:
            log.error("could not open file " + self.filename, exc_info=True)
            return None

        try:
            #use the cleaned image for display
            self.data = np.array(f[PANACEA_HDU_IDX['clean_image']].data)
            self.data[np.isnan(self.data)] = 0.0 # clean up any NaNs

            #get error equivalent
            self.err_data = np.array(f[PANACEA_HDU_IDX['error']].data)
            self.err_data[np.isnan(self.err_data)] = 0.0

            #get fe equivalent
            self.fe_data = np.array(f[PANACEA_HDU_IDX['sky_subtracted']].data)
            self.fe_data[np.isnan(self.fe_data)] = 0.0

            # get fe equivalent (need also the matching wavelengths)
            self.wave_data = np.array(f[PANACEA_HDU_IDX['wavelength']].data)
            self.wave_data[np.isnan(self.wave_data)] = 0.0

            self.trace_data = np.array(f[PANACEA_HDU_IDX['trace']].data)
            self.trace_data[np.isnan(self.trace_data)] = 0.0

            # get fiber centers
            # the fits representation is backward (with grid x,y: 1,112 and 2,112 (i.e at the top) == fiber 1))
            self.fiber_centers = np.array(f[PANACEA_HDU_IDX['ifupos']].data)

            #self.pixflat_data = np.array(f[PANACEA_HDU_IDX['flat_image']].data)
            #self.pixflat_data[np.isnan(self.pixflat_data)] = 0.0

            self.panacea = True

            #most complete header in the raw image
            idx = PANACEA_HDU_IDX['image']
        except:
            log.error("Cannot read fits header. Missing expected keywords. " + self.filename, exc_info=True)
            self.okay = False
            return

        try:
            self.tel_ra = float(f[idx].header['RA']) * 15.0  # header is in decimal hours
            self.tel_dec = float(f[idx].header['DEC'])  # header is in decimal degs
            self.parangle = f[idx].header['PA']
        except:
            log.error("Non-fatal: Cannot translate RA and/or Dec from FITS format to degrees in " + self.filename, exc_info=True)
            #might be okay, depeding on if the individual emission lines have the weighted RA and Dec Specified

        try:
            self.ifuid = str(f[idx].header['IFUSID']).zfill(3)
            self.ifuslot = str(f[idx].header['IFUSLOT']).zfill(3)
            self.specid = str(f[idx].header['SPECID']).zfill(3)
            self.amp = f[idx].header['AMP']
            self.side = f[idx].header['AMP'][0] #the L or R ... probably don't need this anyway
            #self.exptime = f[idx].header['EXPTIME']
        except:
            log.error("Cannot read fits header. Missing expected keywords. Will attempt to pull from filename." + self.filename, exc_info=True)
            #try to get info from the filename
            self.parse_panacea_fits_name(self.filename)
            return

        try:
            f.close()
        except:
            log.error("could not close file " + self.filename, exc_info=True)

        return

    def parse_panacea_fits_name(self,name):
        if name is not None:
            if "multi_" in name: #multi_037_073_031_LL.fits
                toks = name.split("_")  #multi_fits_basename = "multi_" + self.specid + "_" + self.ifu_slot_id + "_" + self.ifu_id + "_"
                if len(toks) == 5:
                    try:
                        self.specid = toks[1].zfill(3)
                        self.ifuslot = toks[2].zfill(3)
                        self.ifuid = toks[3].zfill(3)
                        self.amp = toks[4][0:2]
                        self.side =toks[4][0]
                    except:
                        log.error("Cannot parse panaces fits filename: %s" %name,exc_info=True)
                        self.okay = False



    def cleanup(self):
        #todo: close fits handles, etc
        pass


class FitsSorter:
#just a container for organization
    def __init__(self,fits=None,dist=0.0,loc=-1,side=None,dither=None,sn=None):
        self.fits = fits
        self.dist = dist
        self.loc = loc
        self.side = side
        self.dither = dither
        self.fiber_sn = sn


class HETDEX:

    #needs dither file, detect_line file, 2D fits files (i.e. 6 for 3 dither positions)
    #needs find fplane file (from observation date in fits files? or should it be passed in?)

    def __init__(self,args):
        if args is None:
            log.error("Cannot construct HETDEX object. No arguments provided.")
            return None

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

        self.ifu_slot_id = None
        self.ifu_id = None
        self.specid = None
        self.obsid = None

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

        if args.cure:
            self.panacea = False
        else:
            self.panacea = True

        self.emission_lines = [EmissionLine("Ly$\\alpha$ ",1216,'red'),
                               EmissionLine("OII ",3727,'green'),
                               EmissionLine("OIII",4959,"lime"), EmissionLine("OIII",5007,"lime"),
                               EmissionLine("CIII", 1909, "purple"),
                               EmissionLine("CIV ",1549,"black"),
                               EmissionLine("H$\\beta$ ",4861,"blue"),
                               EmissionLine("HeII", 1640, "orange"),
                               EmissionLine("MgII", 2798, "magenta",solution=False),
                               EmissionLine("H$\\gamma$ ", 4341, "royalblue",solution=False),
                               EmissionLine("NV ",1240,"teal",solution=False),
                               EmissionLine("SiII",1260,"gray",solution=False)]

        #self.panacea_fits_list = [] #list of all panacea multi_*fits files (all dithers) (should be 4amps x 3dithers)

        #parse the dither file
        #use to build fits list
        if self.dither_fn is not None:
            self.dither = Dither(self.dither_fn)
        else:
            #are we done? must have a dither file?
            log.error("Cannot construct HETDEX object. No dither file provided.")
            return None

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
        if self.fplane_fn is None:
            self.fplane_fn = find_fplane(self.ymd)

        if self.fplane_fn is not None:
            self.fplane = FPlane(self.fplane_fn)
            self.ifuslot_dict, self.cam_ifu_dict, self.cam_ifuslot_dict = build_fplane_dicts(self.fplane_fn)


        #read the detect line file if specified. Build a list of targets based on sigma and chi2 cuts
        if self.detectline_fn is not None: #this is optional
            self.read_detectline()


        #assign dither indices to fibers for each emission object
        for e in self.emis_list:
            for f in e.fibers:
                f.dither_idx = self.dither.get_dither_index(f.dither_date,f.dither_time,f.dither_time_extended)
                #get centers
                for s in self.sci_fits:
                    #dither index should not matter, but if these are combined across much time, it is possible
                    #that the centers could have changed
                    if (s.dither_index == f.dither_idx) and (s.amp == f.amp):
                        f.center_x = s.fiber_centers[f.number - 1,0]
                        f.center_y = s.fiber_centers[f.number - 1,1]
                        break



        #calculate the RA and DEC of each emission line object
        #remember, we are only using a single IFU per call, so all emissions belong to the same IFU

        #if ra and dec were passed in, use them instead of tel_ra and tel_dec

        #note: rot = 360-(90 + 1.8 + PARANGLE) so, PARANGLE = 360 -(90+1.8+rot)
        #the 1.8 constant is under some investigation (have also seen 1.3)

        #if PARANGLE is specified on the command line, use it instead of the FITS PARANGLE
        #360. - (90+1.3+args.rot)) from DetectWebpage
        if args.rot is not None:
            self.rot = float(args.rot)
        elif args.par is not None:
            self.rot = 360. - (90. + 1.3 + args.par)
        else:
            self.rot = 360. - (90. + 1.3 + self.parangle)

        if (args.ra is not None) and (args.dec is not None):
            #if (args.rot is not None):
            #    rot = args.rot
            #else:
            #    rot = 360. - (90. + 1.8 + self.parangle)

            self.tangentplane = TP(args.ra, args.dec, self.rot)
            log.debug("Calculating object RA, DEC from commandline RA=%f , DEC=%f , ROT=%f" \
                      % (args.ra, args.dec, self.rot))
        else:
            self.tangentplane = TP(self.tel_ra, self.tel_dec, self.rot)
            log.debug("Calculating object RA, DEC from: TELRA=%f , TELDEC=%f , PARANGLE=%f , ROT=%f" \
                  % (self.tel_ra, self.tel_dec, self.parangle, self.rot))

        #wants the slot id as a 0 padded string ie. '073' instead of the int (73)
        self.ifux = self.fplane.by_ifuslot(self.ifu_slot_id).x
        self.ifuy = self.fplane.by_ifuslot(self.ifu_slot_id).y

        for e in self.emis_list: #yes this right: x + ifuy, y + ifux
            e.ra, e.dec = self.tangentplane.xy2raDec(e.x + self.ifuy, e.y + self.ifux)
            log.info("Emission Detect ID #%d RA=%f , Dec=%f" % (e.id,e.ra,e.dec))

    #end HETDEX::__init__()


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

        if len(self.dither.basename) < 1: #we just got dx,dy from all.mch, need other info to continue
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
                    log.error("Invalid path to panacea science fits: %s" %path)

            while op.isdir(path):
                for a in AMP:
                    fn = op.join(path, multi_fits_basename + a + ".fits")
                    self.sci_fits.append(HetdexFits(fn, None, None, dit_idx, panacea=True))
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

                    hdf = HetdexFits(fn, None, None, dit_idx)

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
                    self.sci_fits.append(HetdexFits(fn, None, None, dit_idx,panacea=True))

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

                    self.sci_fits.append(HetdexFits(fn,e_fn,fe_fn,dit_idx))
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


    def read_detectline(self):
        #emission line or continuum line
        #todo: determine which (should be more robust)

        if '_cont.dat' in self.detectline_fn:
            self.read_contline()
        else:
            self.read_emisline()


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
                    e = DetObj(toks,emission=False)

                    if e.ifuslot is not None:
                        if e.ifuslot != self.ifu_slot_id:
                            #this emission line does not belong to the IFU we are working on
                            log.info("Continuum detection IFU (%s) does not match current working IFU (%s)" %
                                     (e.ifuslot,self.ifu_slot_id))
                            continue

                    if self.emis_det_id is not None:
                        if str(e.id) in self.emis_det_id:
                            self.emis_list.append(e)
                    else:
                        self.emis_list.append(e)
        except:
            log.error("Cannot read continuum objects.", exc_info=True)

    def read_emisline(self):
        #open and read file, line at a time. Build up list of emission line objects

        if len(self.emis_list) > 0:
            del self.emis_list[:]
        try:
            with open(self.detectline_fn, 'r') as f:
                f = ft.skip_comments(f)
                for l in f:
                    toks = l.split()
                    e = DetObj(toks,emission=True)

                    if e.ifuslot is not None:
                        if e.ifuslot != self.ifu_slot_id:
                            #this emission line does not belong to the IFU we are working on
                            log.info("Emission detection IFU (%s) does not match current working IFU (%s)" %
                                     (e.ifuslot,self.ifu_slot_id))
                            continue

                    if self.emis_det_id is not None:
                        if str(e.id) in self.emis_det_id:
                            self.emis_list.append(e)
                    else:
                        if (e.sigma >= self.sigma) and (e.chi2 <= self.chi2):
                            self.emis_list.append(e)
        except:
            log.error("Cannot read emission line objects.", exc_info=True)

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

        e = self.get_emission_detect(detectid)
        if e is None:
            log.error("Could not identify correct emission to plot. Detect ID = %d" % detectid)
            return None

        print ("Bulding HETDEX header for Detect ID #%d" %detectid)
        #todo: match this up with the catalog sizes

        if G.SINGLE_PAGE_PER_DETECT:
            figure_sz_y = G.GRID_SZ_Y
        else:
            figure_sz_y = 3 * G.GRID_SZ_Y

        fig = plt.figure(figsize=(G.FIGURE_SZ_X, figure_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        #plt.tight_layout()
        plt.gca().axis('off')
        # 2x2 grid (but for sizing, make more fine)
        if G.SINGLE_PAGE_PER_DETECT:
            gs = gridspec.GridSpec(2, 3)
        else:
            if G.SHOW_FULL_2D_SPECTRA:
                gs = gridspec.GridSpec(5, 3)#, wspace=0.25, hspace=0.5)
            else:
                gs = gridspec.GridSpec(3, 3)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        sci_files = ""
        for s in self.dither.basename:
            if not( op.basename(s) in sci_files):
                sci_files += "  " + op.basename(s) + "*.fits\n"

        title = ""
        if e.type == 'emis':
            title += "Emission Line Detect ID #"
        else:
            title += "Continuum Detect ID#"

        if (e.wra is not None) and (e.wdec is not None):  # weighted RA and Dec
            ra = e.wra
            dec = e.wdec
        else:
            ra = e.ra
            dec = e.dec

        title +="%d\n"\
                "ObsDate %s  ObsID %s IFU %s  CAM %s\n" \
                "Science file(s):\n%s"\
                "RA,Dec (%f,%f) \n"\
                "Sky X,Y (%f,%f)\n" \
                "$\lambda$ = %g $\AA$\n" \
                "EstFlux = %0.3g  DataFlux = %g/%0.3g\n" \
                "EstEqw = %g $\AA$ Cont = %g\n" \
                 % (e.id,self.ymd, self.obsid, self.ifu_slot_id,self.specid,sci_files, ra, dec, e.x, e.y,e.w,
                    e.estflux, e.dataflux, e.fluxfrac, e.eqw, e.cont) #note: e.fluxfrac gauranteed to be nonzero

        if self.panacea:
            title += "S/N = %g  Chi2 = %g" % (e.sigma, e.chi2)
        else:
            title += "Sigma = %g  Chi2 = %g" % (e.sigma, e.chi2)

        if e.w > 0:
            la_z = e.w / G.LyA_rest - 1.0
            oii_z = e.w / G.OII_rest - 1.0
            #title = title + "\nLy$\\alpha$ Z = %g" % la_z
            title = title + "\nLyA Z = %g" % la_z
            if (oii_z > 0):
                title = title + "  OII Z = %g" % oii_z
            else:
                title = title + "  OII Z = N/A"

        plt.subplot(gs[0:2, 0])
        plt.text(0, 0.5, title, ha='left', va='center', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        datakeep = None
        if self.panacea:
            datakeep = self.build_panacea_hetdex_data_dict(e)
        else:
            datakeep = self.build_hetdex_data_dict(e)

        if datakeep is not None:
            if datakeep['xi']:
                try:
                    plt.subplot(gs[0:2,1])
                    plt.gca().axis('off')
                    buf = self.build_2d_image(datakeep)

                    buf.seek(0)
                    im = Image.open(buf)
                    plt.imshow(im,interpolation='none') #needs to be 'none' else get blurring
                except:
                    log.warning("Failed to 2D cutout image.", exc_info=True)

                try:
                    plt.subplot(gs[0:2,2:])
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
                        buf = self.build_full_width_2d_image(datakeep, e.w)
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
                        buf = self.build_full_width_2d_image(datakeep, e.w)
                        buf.seek(0)
                        im = Image.open(buf)
                        plt.imshow(im, interpolation='none')  # needs to be 'none' else get blurring
                    except:
                        log.warning("Failed to build full width spec/cutout image.", exc_info=True)


            # update emission with the ra, dec of all fibers
            e.fiber_locs = list(zip(datakeep['ra'], datakeep['dec'],datakeep['color'],datakeep['index'],datakeep['d']))


        if not G.SINGLE_PAGE_PER_DETECT:
            pages.append(fig)
            plt.close()
        #else, the pages were appended invidivually
        return pages

    def get_vrange(self,vals):
        vmin = None
        vmax = None

        try:
            zscale = ZScaleInterval(contrast=0.25,krej=2.5) #nsamples=len(vals)
            vmin,vmax = zscale.get_limits(values=vals )
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
            dd['dit'] = []
            dd['side'] = []
            dd['amp'] = []
            dd['fib'] = []
            dd['xi'] = []
            dd['yi'] = []
            dd['xl'] = []
            dd['yl'] = []
            dd['xh'] = []
            dd['yh'] = []
            dd['sn'] = []
            dd['fiber_sn'] = []
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
            dd['err'] = []
            dd['pix'] = []
            dd['spec'] = []
            dd['specwave'] = []
            dd['fw_spec']  = []
            dd['fw_specwave'] = []
            dd['cos'] = []
            dd['ra'] = []
            dd['dec'] = []
            dd['color'] = []
            dd['index'] = []
        return dd


    def build_hetdex_data_dict(self,e):#e is the emission detection to use
        if e is None:
            return None

        if e.type != 'emis':
            return None

        #basically cloned from Greg Z. make_visualization_detect.py; adjusted a bit for this code base
        datakeep = self.clean_data_dict()

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
            dither = item.dither
            loc = item.loc
            datakeep['d'].append(item.dist)  # distance (in arcsec) of fiber center from object center
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
            datakeep['fib'].append(self.dist[side].map_xy_fibernum(xi, yi))
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



            xl = max(xl,0)
            xh = min(xh,max_x)
            yl = max(yl,0)
            yh = min(yh,max_y)



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

                # this is probably for the 1d spectra
                I = sci.data.ravel()
                s_ind = np.argsort(I)
                len_s = len(s_ind)
                s_rank = np.arange(len_s)
                p = np.polyfit(s_rank - len_s / 2, I[s_ind], 1)

                z1 = I[s_ind[int(len_s / 2)]] + p[0] * (1 - len_s / 2) / contrast1
                z2 = I[s_ind[int(len_s / 2)]] + p[0] * (len_s - len_s / 2) / contrast1

                #z1,z2 = self.get_vrange(sci.data[yl:yh,xl:xh])
                datakeep['vmin1'].append(z1)
                datakeep['vmax1'].append(z2)

                z1 = I[s_ind[int(len_s / 2)]] + p[0] * (1 - len_s / 2) / contrast2
                z2 = I[s_ind[int(len_s / 2)]] + p[0] * (len_s - len_s / 2) / contrast2

                datakeep['vmin2'].append(z1)
                datakeep['vmax2'].append(z2)

                datakeep['err'].append(sci.err_data[yl:yh, xl:xh])

            pix_fn = op.join(PIXFLT_LOC,'pixelflat_cam%s_%s.fits' % (sci.specid, side))
            if op.exists(pix_fn):
                datakeep['pix'].append(pyfits.open(pix_fn)[0].data[yl:yh,xl:xh])

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

                datakeep['fw_spec'].append(sci.fe_data[loc,:])
                datakeep['fw_specwave'].append(wave[:])

        return datakeep


    #build_panacea_hetdex_data_dict
    def build_panacea_hetdex_data_dict(self, e):  # e is the emission detection to use
        if e is None:
            return None

        if e.type != 'emis':
            return None

        # basically cloned from Greg Z. make_visualization_detect.py; adjusted a bit for this code base
        datakeep = self.clean_data_dict()
        sort_list = []

        if len(e.fibers) > 0:
            #use fiber list rather than distance
            for f in e.fibers:
                dither = f.dither_idx
                fits = self.get_sci_fits(dither,f.side,f.amp)

                if fits is None:
                    log.error("Error! Could not find appropriate fits file for fiber: %s %d" % (f.exp_id,f.number))
                    continue
                #look at specific fibers

                dx = e.x - f.center_x - self.dither.dx[dither] #just floats, not arrays like below
                dy = e.y - f.center_y - self.dither.dy[dither]

                d = np.sqrt(dx ** 2 + dy ** 2)
                #turn fiber number into a location. Fiber Number 1 is at the top
                #which is loc (or index) 111
                #so loc = 112 - Fiber Number
                loc = f.number-1

                sort_list.append(FitsSorter(fits,d,loc,sn=f.sn))

            #we want these in the order given, but they print in reverse, so invert the order
            #sort_list.sort(key=lambda x: x.dist, reverse=True)
            sort_list = sort_list[::-1]


        else: #use fibers w/in 2"

            for fits in self.sci_fits:
                dither = fits.dither_index  # 0,1,2

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


        #zero based so fiber is loc + amp_offset

        #for loc in locations:
        for item in sort_list:
            fits = item.fits
            dither = fits.dither_index  # 0,1,2
            loc = item.loc
            datakeep['d'].append(item.dist)
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
            datakeep['fib'].append(len(fits.fe_data) - (loc+1) + AMP_OFFSET[fits.amp])


            xfiber = fits.fiber_centers[loc][0] + self.dither.dx[dither]
            yfiber = fits.fiber_centers[loc][1] + self.dither.dy[dither]
            xfiber += self.ifuy  # yes this is correct xfiber gets ifuy
            yfiber += self.ifux
            #ra and dec of the center of the fiber (loc)
            ra, dec = self.tangentplane.xy2raDec(xfiber, yfiber)
            datakeep['ra'].append(ra)
            datakeep['dec'].append(dec)

            x_2D = np.interp(e.w,fits.wave_data[loc,:],range(len(fits.wave_data[loc,:])))
            y_2D = np.interp(x_2D,range(len(fits.trace_data[loc,:])),fits.trace_data[loc,:])

            xl = int(np.round(x_2D - xw))
            xh = int(np.round(x_2D + xw))
            yl = int(np.round(y_2D - yw))
            yh = int(np.round(y_2D + yw))

            xl = max(xl, 0)
            xh = min(xh, max_x)
            yl = max(yl, 0)
            yh = min(yh, max_y)

            datakeep['xi'].append(x_2D)
            datakeep['yi'].append(y_2D)
            datakeep['xl'].append(xl)
            datakeep['yl'].append(yl)
            datakeep['xh'].append(xh)
            datakeep['yh'].append(yh)
            datakeep['fxl'].append(0)
            datakeep['fxh'].append(FRAME_WIDTH_X - 1)
            #datakeep['d'].append(d[loc])  # distance (in arcsec) of fiber center from object center

            datakeep['sn'].append(e.sigma)

            # also get full x width data
            # would be more memory effecien to just grab full width,
            # then in original func, slice as below

            datakeep['im'].append(fits.data[yl:yh, xl:xh])
            datakeep['fw_im'].append(fits.data[yl:yh, 0:FRAME_WIDTH_X - 1])

            # this is probably for the 1d spectra
            I = fits.data.ravel()
            s_ind = np.argsort(I)
            len_s = len(s_ind)
            s_rank = np.arange(len_s)
            p = np.polyfit(s_rank - len_s / 2, I[s_ind], 1)

            z1 = I[s_ind[int(len_s / 2)]] + p[0] * (1 - len_s / 2) / contrast1
            z2 = I[s_ind[int(len_s / 2)]] + p[0] * (len_s - len_s / 2) / contrast1

            # z1,z2 = self.get_vrange(sci.data[yl:yh,xl:xh])
            datakeep['vmin1'].append(z1)
            datakeep['vmax1'].append(z2)

            z1 = I[s_ind[int(len_s / 2)]] + p[0] * (1 - len_s / 2) / contrast2
            z2 = I[s_ind[int(len_s / 2)]] + p[0] * (len_s - len_s / 2) / contrast2

            datakeep['vmin2'].append(z1)
            datakeep['vmax2'].append(z2)

            datakeep['err'].append(fits.err_data[yl:yh, xl:xh])

            pix_fn = op.join(PIXFLT_LOC, 'pixelflat_cam%s_%s.fits' % (fits.specid, fits.side))
            if op.exists(pix_fn):
                datakeep['pix'].append(pyfits.open(pix_fn)[0].data[yl:yh, xl:xh])
            #datakeep['pix'].append(fits.pixflat_data[yl:yh, xl:xh])

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
            datakeep['spec'].append(fits.fe_data[loc, Fe_indl:(Fe_indh)])
            datakeep['specwave'].append(wave[Fe_indl:(Fe_indh)])

            datakeep['fw_spec'].append(fits.fe_data[loc, :])
            datakeep['fw_specwave'].append(wave[:])

        return datakeep

    #2d spectra cutouts (one per fiber)
    def build_2d_image(self,datakeep):

        cmap = plt.get_cmap('gray_r')
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))
        num = len(datakeep['xi'])
        bordbuff = 0.01
        borderxl = 0.05
        borderxr = 0.15
        borderyb = 0.05
        borderyt = 0.15
        dx = (1. - borderxl - borderxr) / 3.
        dy = (1. - borderyb - borderyt) / num
        dx1 = (1. - borderxl - borderxr) / 3.
        dy1 = (1. - borderyb - borderyt - num * bordbuff) / num
        Y = (yw / dy) / (xw / dx) * 5.

        fig = plt.figure(figsize=(5, Y), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        # previously sorted in order from largest distances to smallest
        ind = list(range(len(datakeep['d'])))

        for i in range(num):
            borplot = plt.axes([borderxl + 0. * dx, borderyb + i * dy, 3 * dx, dy])
            smplot = plt.axes([borderxl + 2. * dx - bordbuff / 3., borderyb + i * dy + bordbuff / 2., dx1, dy1])
            pixplot = plt.axes(
                [borderxl + 1. * dx + 1 * bordbuff / 3., borderyb + i * dy + bordbuff / 2., dx1, dy1])
            imgplot = plt.axes([borderxl + 0. * dx + bordbuff / 2., borderyb + i * dy + bordbuff / 2., dx1, dy1])
            autoAxis = borplot.axis()
            datakeep['color'][i] = colors[i, 0:3]
            datakeep['index'][i] = num -i

            rec = plt.Rectangle((autoAxis[0] + bordbuff / 2., autoAxis[2] + bordbuff / 2.),
                                (autoAxis[1] - autoAxis[0]) * (1. - bordbuff),
                                (autoAxis[3] - autoAxis[2]) * (1. - bordbuff), fill=False, lw=3,
                                color=datakeep['color'][i], zorder=1)
            rec = borplot.add_patch(rec)
            borplot.set_xticks([])
            borplot.set_yticks([])
            borplot.axis('off')
            ext = list(np.hstack([datakeep['xl'][ind[i]], datakeep['xh'][ind[i]],
                                  datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))
            GF = gaussian_filter(datakeep['im'][ind[i]], (2, 1))
            smplot.imshow(GF,
                          origin="lower", cmap=cmap,
                          interpolation="none", vmin=datakeep['vmin1'][ind[i]],
                          vmax=datakeep['vmax1'][ind[i]],
                          extent=ext)
            # plot the center point
            #smplot.scatter(datakeep['xi'][ind[i]], datakeep['yi'][ind[i]],
            #               marker='.', c='g', edgecolor='g', s=10)
            smplot.set_xticks([])
            smplot.set_yticks([])
            smplot.axis(ext)
            smplot.axis('off')


            pixplot.imshow(datakeep['pix'][ind[i]],
                           origin="lower", cmap=plt.get_cmap('gray'),
                           interpolation="none", vmin=0.9, vmax=1.1,
                           extent=ext)
            # plot the center point
            #errplot.scatter(datakeep['xi'][ind[i]], datakeep['yi'][ind[i]],
            #                marker='.', c='r', edgecolor='r', s=10)
            pixplot.set_xticks([])
            pixplot.set_yticks([])
            pixplot.axis(ext)
            pixplot.axis('off')

            #a = datakeep['cos'][ind[i]]

            # ... changed ... it is safe to use the error file all the time ... zeroes only present if cosmic cleaned
            # if (G.PreferCosmicCleaned):
            #    a = np.ma.masked_where(a == -1, a) #should have been -1 not 0
            #    cmap1 = cmap
            #    cmap1.set_bad(color=[0.2, 1.0, 0.23])
            # else:
            #    cmap1 = cmap

            a = datakeep['im'][ind[i]] #im can be the cosmic removed version, depends on G.PreferCosmicCleaned
            cmap1 = cmap
            cmap1.set_bad(color=[0.2, 1.0, 0.23])
            a = np.ma.masked_where( datakeep['err'][ind[i]] == -1, a)

            #test
            #jm,km = datakeep['err'][ind[i]].shape
            #for j in range(jm):
            #    for k in range(km):
            #        if datakeep['err'][ind[i]][j,k] < 0:
            #            msg = "++++++++++ [%d,%d] = %f" % (j,k,datakeep['err'][ind[i]][j,k])
            #            log.critical(msg)

            imgplot.imshow(a,
                           origin="lower", cmap=cmap1,
                           vmin=datakeep['vmin1'][ind[i]],
                           vmax=datakeep['vmax1'][ind[i]],
                           interpolation="none",extent=ext)

            #plot the center point
            #imgplot.scatter(datakeep['xi'][ind[i]], datakeep['yi'][ind[i]],
            #               marker='.', c='b', edgecolor='b', s=10)
            imgplot.set_xticks([])
            imgplot.set_yticks([])
            imgplot.axis(ext)
            imgplot.axis('off')

            xi = datakeep['xi'][ind[i]]
            yi = datakeep['yi'][ind[i]]
            xl = int(np.round(xi - ext[0] - res[0] / 2.))
            xh = int(np.round(xi - ext[0] + res[0] / 2.))
            yl = int(np.round(yi - ext[2] - res[0] / 2.))
            yh = int(np.round(yi - ext[2] + res[0] / 2.))

            sn = datakeep['fiber_sn'][ind[i]]

            if sn is None:
                S = np.where(datakeep['err'][ind[i]][yl:yh, xl:xh] < 0, 0., datakeep['im'][ind[i]][yl:yh, xl:xh]).sum()
                N = np.sqrt(np.where(datakeep['err'][ind[i]][yl:yh, xl:xh] < 0, 0.,
                                     datakeep['err'][ind[i]][yl:yh, xl:xh] ** 2).sum())
                if N != 0:
                    sn = S / N
                else:
                    sn = 0.0

            imgplot.text(-0.2, .5, num - i,
                        transform=imgplot.transAxes, fontsize=6, color='k', #colors[i, 0:3],
                        verticalalignment='bottom', horizontalalignment='left')

            borplot.text(1.05, .75, 'S/N = %0.2f' % (sn),
                        transform=smplot.transAxes, fontsize=6, color='r',
                        verticalalignment='bottom', horizontalalignment='left')
            #distance (in arcsec) of fiber center from object center
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
            if i == (num - 1):
                smplot.text(0.5, 1.3, 'Smoothed',
                            transform=smplot.transAxes, fontsize=8, color='k',
                            verticalalignment='top', horizontalalignment='center')
                pixplot.text(0.5, 1.3, 'Pixel Flat',
                             transform=pixplot.transAxes, fontsize=8, color='k',
                             verticalalignment='top', horizontalalignment='center')
                imgplot.text(0.5, 1.3, 'Image',
                             transform=imgplot.transAxes, fontsize=8, color='k',
                             verticalalignment='top', horizontalalignment='center')

        buf = io.BytesIO()
       # plt.tight_layout()#pad=0.1, w_pad=0.5, h_pad=1.0)
        plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf

    def build_spec_image(self,datakeep,cwave, dwave=1.0):

        fig = plt.figure(figsize=(5, 3))
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))

        rm = 0.2
        r, w = get_w_as_r(1.5, 500, 0.05, 6.)
        specplot = plt.axes([0.1, 0.1, 0.8, 0.8])
        bigwave = np.arange(cwave - ww, cwave + ww + dwave, dwave)
        F = np.zeros(bigwave.shape)
        mn = 100.0
        mx = 0.0
        W = 0.0

        #these are plotted from bottom to top
        #we want, then, the LAST (plot_fibers) to be ploted

        #want ALL ind even if only goint to plot spectra for a subset
        ind = range(len(datakeep['d']))

        #previously sorted in order from largest distances to smallest
        N = len(datakeep['xi'])
        if self.plot_fibers is not None:
            stop = max(N - self.plot_fibers-1,-1)
        else:
            stop = -1

        alpha = 1.0
        linewidth = 2.0

        try:
            for i in range(N-1,stop,-1):
                #regardless of the number if the sn is below the threshold, skip it
                if (datakeep['fiber_sn'][i] is not None) and (datakeep['fiber_sn'][i] < self.min_fiber_sn):
                    continue

                specplot.step(datakeep['specwave'][ind[i]], datakeep['spec'][ind[i]],linestyle="solid",
                              where='mid', color=colors[i, 0:3], alpha=alpha,linewidth=linewidth,zorder=i)
                w1 = np.interp(datakeep['d'][ind[i]], r, w)
                F += (np.interp(bigwave, datakeep['specwave'][ind[i]], datakeep['spec'][ind[i]]) * w1)
                W += w1
                mn = np.min([mn, np.min(datakeep['spec'][ind[i]])])
                mx = np.max([mx, np.max(datakeep['spec'][ind[i]])])
            F /= W
            ran = mx - mn

            if PLOT_SUMMED_SPECTRA:
                specplot.step(bigwave, F, c='k', where='mid', lw=5,linestyle="solid",alpha=0.3,zorder=99)

            min_y = max(mn - ran / 20, -20)

            specplot.axis([cwave - ww, cwave + ww, min_y, mx + ran / 20])
            specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], ls='--', c=[0.3, 0.3, 0.3])

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
            #todo: turn errorbar back on if useful
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

        plt.close(fig)
        return buf

     # 2d spectra cutouts (one per fiber)
    def build_full_width_2d_image(self, datakeep, cwave):

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

            datakeep['color'][i] = colors[i, 0:3]
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
        F = np.zeros(bigwave.shape)

        N = len(datakeep['xi'])
        if self.plot_fibers is not None:
            stop = max(N - self.plot_fibers - 1, -1)
        else:
            stop = -1

        try:
             #new way, per Karl, straight sum
            for j in range(N - 1, stop, -1):
                # regardless of the number if the sn is below the threshold, skip it
                if (datakeep['fiber_sn'][j] is not None) and (datakeep['fiber_sn'][j] < self.min_fiber_sn):
                    continue

                F += (np.interp(bigwave, datakeep['fw_specwave'][ind[j]], datakeep['fw_spec'][ind[j]]) )

            mn = np.min(F)
            mn = max(mn,-20) #negative flux makes no sense (excepting for some noise error)
            mx = np.max(F)
            ran = mx - mn
            specplot.step(bigwave, F, c='b', where='mid', lw=1)

            specplot.plot([cwave, cwave], [mn - ran * rm, mn + ran * (1 + rm)], ls='dashed', c='k') #[0.3, 0.3, 0.3])
            specplot.axis([left, right, mn - ran / 20, mx + ran / 20])

            specplot.locator_params(axis='y',tight=True,nbins=4)

            textplot = plt.axes([border_buffer, (float(num)+3) * dy, 1.0 - (2 * border_buffer), dy*2 ])
            textplot.set_xticks([])
            textplot.set_yticks([])
            textplot.axis(specplot.axis())
            textplot.axis('off')

            #iterate over all emission lines ... assume the cwave is that line and plot the additional lines

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

        #draw rectangle around section that is "zoomed"
        yl, yh = specplot.get_ylim()
        rec = plt.Rectangle((cwave - ww, yl), 2 * ww, yh - yl, fill=True, lw=1, color='y', zorder=1)
        specplot.add_patch(rec)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf

#end HETDEX class