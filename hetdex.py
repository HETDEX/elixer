
import matplotlib
matplotlib.use('agg')

import global_config
import science_image
from astropy.io import fits as pyfits
from astropy.coordinates import Angle
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

import glob
from pyhetdex.cure.distortion import Distortion
import pyhetdex.tools.files.file_tools as ft
from pyhetdex.het.ifu_centers import IFUCenter
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection import TangentPlane as TP
import os.path as op

log = global_config.logging.getLogger('Cat_logger')
log.setLevel(global_config.logging.DEBUG)

CONFIG_BASEDIR = global_config.CONFIG_BASEDIR
VIRUS_CONFIG = op.join(CONFIG_BASEDIR,"virus_config")
FPLANE_LOC = op.join(CONFIG_BASEDIR,"virus_config/fplane")
IFUCEN_LOC = op.join(CONFIG_BASEDIR,"virus_config/IFUcen_files")
DIST_LOC = op.join(CONFIG_BASEDIR,"virus_config/DeformerDefaults")


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


class DetectLine:
    '''Cure detect line file '''
    #needs open and parse
    #needs: find (compute) RA,Dec of entries (used to match up with catalogs)
    def __init__(self):
        pass


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

        self.read_dither(dither_file)

    def read_dither(self, dither_file):
        try:
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
        except:
            log.error("Unable to read dither file: %s :" %dither_file, exc_info=True)



class EmisDet:
    '''mostly a container for an emission line detection from detect_line.dat file'''
    #  0    1   2   3   4   5   6           7       8        9     10    11    12     13      14      15  16
    #  NR  ID   XS  XS  l   z   dataflux    modflux fluxfrac sigma chi2  chi2s chi2w  gammq   gammq_s eqw cont

    def __init__(self,tokens):
        #skip NR (0)
        self.id = int(tokens[1])
        self.x = float(tokens[2])
        self.y = float(tokens[3])
        self.w = float(tokens[4])
        self.la_z = float(tokens[5])
        self.dataflux = float(tokens[6])
        self.modflux = float(tokens[7])
        self.fluxfrac = float(tokens[8])
        self.sigma = float(tokens[9])
        self.chi2 = float(tokens[10])
        self.chi2s = float(tokens[11])
        self.chi2w = float(tokens[12])
        self.gammq = float(tokens[13])
        self.gammq_s = float(tokens[14])
        self.eqw = float(tokens[15])
        self.cont = float(tokens[16])

        self.ra = None  # calculated value
        self.dec = None  # calculated value
        self.nearest_fiber = None

    #todo: calculate the ra and dec from the sky x,y position (and astrometry)
    def calc_ra_dec(self):
        pass


class HetdexFits:
    '''A single HETDEX fits file ... 2D spectra, expected to be science file'''

    #needs open with basic validation
    #

    def __init__(self,fn):
        self.filename = fn

        self.tel_ra = None
        self.tel_dec = None
        self.parangle = None
        self.data = None
        self.ifuslot = None
        self.side = None
        self.specid = None
        self.obs_date = None
        self.obs_ymd = None
        self.mjd = None
        self.obsid = None
        self.imagetype = None
        self.exptime = None
        self.dettemp = None

        self.read_fits()

    def read_fits(self):

        try:
            f = pyfits.open(self.filename)
        except:
            log.error("could not open file " + self.filename, exc_info=True)
            return None

        self.data = np.array(f[0].data)

        try:
            self.tel_ra = float(Angle(f[0].header['TELRA']+"h").degree) #header is in hour::mm:ss.arcsec
            self.tel_dec = float(Angle(f[0].header['TELDEC']+"d").degree) #header is in deg:hh:mm:ss.arcsec
        except:
            log.error("Cannot translate RA and/or Dec from FITS format to degrees in " + self.filename, exc_info=True)

        self.parangle = f[0].header['PARANGLE']
        self.ifuslot = str(f[0].header['IFUSLOT']).zfill(3)
        self.side = f[0].header['CCDPOS']
        self.specid = str(f[0].header['SPECID']).zfill(3)

        self.obs_date = f[0].header['DATE-OBS']

        if '-' in self.obs_date: #expected format is YYYY-MM-DD
            self.obs_ymd = self.obs_date.replace('-','')
        self.mjd = f[0].header['MJD']
        self.obsid = f[0].header['OBSID']
        self.imagetype = f[0].header['IMAGETYP']
        self.exptime = f[0].header['EXPTIME']
        self.dettemp = f[0].header['DETTEMP']

        try:
            f.close()
        except:
            log.error("could not close file " + self.filename, exc_info=True)

        return

    def cleanup(self):
        #todo: close fits handles, etc
        pass



class HetdexIfuObs:
    '''Basically, a container for all the fits, dither, etc that make up one observation for one IFU'''
    def __init__(self):
        self.specid = None
        self.ifuslotid = None
        #self.ifuid = None # I think can omit this ... don't want to confuse with ifuslotid
        #todo: needs  list of fits
        #todo: needs fplane
        #todo: needs dither
        #todo:


#an object (detection) should only be in one IFU, so this is not necessary
#class HetdexFullObs:
#    '''Container for all HetdexIfuObs ... that is, all IFUs data for one observation'''
#    def __init__(self):
#        self.date_ymd = None #date string as yyyymmdd


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

        self.tel_ra = None
        self.tel_dec = None
        self.parangle = None
        self.ifu_slot_id = None
        self.specid = None

        self.dither_fn = args.dither
        self.detectline_fn = args.line
        self.sigma = args.sigma
        self.chi2 = args.chi2
        self.emis_det_id = args.id #might be a list?
        self.dither = None #Dither() obj
        self.fplane_fn = None
        self.fplane = None
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
        if not self.build_fits_list():
            #fatal problem
            self.status = -1
            return

        #get ifu centers
        if args.ifu is not None:
            try:
                self.ifu_ctr = IFUCenter(args.ifu)
            except:
                log.error("Unable to open IFUcen file: %s" %(args.ifu), exc_info=True)
        else:
            ifu_fn  = op.join(IFUCEN_LOC,"IFUcen_VIFU"+self.ifu_slot_id+".txt")
            log.info("No IFUcen file provided. Look for CAM specific file %s" % (ifu_fn))
            try:
                self.ifu_ctr = IFUCenter(ifu_fn)
            except:
                ifu_fn = op.join(IFUCEN_LOC, "IFUcen_HETDEX.txt")
                log.info("Unable to open CAM Specific IFUcen file. Look for generic IFUcen file.")
                try:
                    self.ifu_ctr = IFUCenter(ifu_fn)
                except:
                    log.error("Unable to open IFUcen file.",exc_info=True)

        #get distortion info
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

        #calculate the RA and DEC of each emission line object
        #remember, we are only using a single IFU per call, so all emissions belong to the same IFU
        rot = 360 - (90+1.8+self.parangle)
        tp = TP(self.tel_ra, self.tel_dec, 360. - (90 + 1.3 + rot))
        #wants the slot id as a 0 padded string ie. '073' instead of the int (73)
        ifux = self.fplane.by_ifuslot(self.ifu_slot_id).x
        ifuy = self.fplane.by_ifuslot(self.ifu_slot_id).y

        for e in self.emis_list:
            e.ra, e.dec = tp.xy2raDec(e.x + ifuy, e.y + ifux)


    def build_fits_list(self):
        #read in all fits
        #get the key fits header values

        #only one dither object, but has array of (3) for each value
        for b in self.dither.basename:
            ext = ['_L.fits','_R.fits']
            for e in ext:
                fn = b + e
                if (self.sci_fits_path is not None):
                    fn = op.join(self.sci_fits_path, op.basename(fn))

                self.sci_fits.append(HetdexFits(fn))

        #all should have the same observation date in the headers so just use first
        if len(self.sci_fits) > 0:
            self.ymd = self.sci_fits[0].obs_ymd
            self.tel_ra = self.sci_fits[0].tel_ra
            self.tel_dec = self.sci_fits[0].tel_dec
            self.parangle = self.sci_fits[0].parangle
            self.ifu_slot_id = self.sci_fits[0].ifuslot
            self.specid = self.sci_fits[0].specid

        if (self.tel_dec is None) or (self.tel_ra is None):
            log.error("Fatal. Cannot determine RA and DEC from FITS.", exc_info=True)
            return False
        return True


    def read_detectline(self):
        #open and read file, line at a time. Build up list of emission line objects

        if len(self.emis_list) > 0:
            del self.emis_list[:]
        try:
            with open(self.detectline_fn, 'r') as f:
                f = ft.skip_comments(f)
                for l in f:
                    toks = l.split()
                    e = EmisDet(toks)
                    if self.emis_det_id is not None:
                        if str(e.id) in self.emis_det_id:
                            self.emis_list.append(e)
                    else:
                        if (e.sigma >= self.sigma) and (e.chi2 <= self.chi2):
                            self.emis_list.append(e)
        except:
            log.error("Cannot emission line objects.", exc_info=True)

        return
#end HETDEX class