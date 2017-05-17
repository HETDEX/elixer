
import matplotlib
matplotlib.use('agg')

import global_config
import science_image
from astropy.io import fits as pyfits
from astropy.coordinates import Angle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
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
PIXFLT_LOC = op.join(CONFIG_BASEDIR,"virus_config/PixelFlats/20161223")

SIDE = ["L", "R"]

#lifted from Greg Z.
dist_thresh = 2.  # Fiber Distance (arcsecs)
xw = 24  # image width in x-dir
yw = 10  # image width in y-dir
contrast1 = 0.9  # convolved image
contrast2 = 0.5  # regular image
res = [3, 9]
ww = xw * 1.9  # wavelength width

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

    # todo: expand emission objects with the data needed for cutouts and spectra

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



class HetdexFits:
    '''A single HETDEX fits file ... 2D spectra, expected to be science file'''

    #needs open with basic validation
    #

    def __init__(self,fn,e_fn,fe_fn,dither_index=-1):
        self.filename = fn
        self.err_filename = e_fn
        self.fe_filename = fe_fn

        self.tel_ra = None
        self.tel_dec = None
        self.parangle = None
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

        self.data = None
        self.err_data = None
        self.fe_data = None
        self.fe_crval1 = None
        self.fe_cdelt1 = None

        self.dither_index = dither_index
        self.read_fits()
        self.read_efits()
        self.read_fefits()

    def read_fits(self):

        try:
            f = pyfits.open(self.filename)
        except:
            log.error("could not open file " + self.filename, exc_info=True)
            return None

        self.data = np.array(f[0].data)
        #clean up any NaNs
        self.data[np.isnan(self.data)] = 0.0

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

    def read_efits(self):

        try:
            f = pyfits.open(self.err_filename)
        except:
            log.error("could not open file " + self.err_filename, exc_info=True)
            return None

        self.err_data = np.array(f[0].data)
        # clean up any NaNs
        self.err_data[np.isnan(self.err_data)] = 0.0

        try:
            f.close()
        except:
            log.error("could not close file " + self.err_filename, exc_info=True)

        return

    def read_fefits(self):

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


    def cleanup(self):
        #todo: close fits handles, etc
        pass



#class HetdexIfuObs:
#    '''Basically, a container for all the fits, dither, etc that make up one observation for one IFU'''
#    def __init__(self):
#        self.specid = None
#        self.ifuslotid = None
#        #self.ifuid = None # I think can omit this ... don't want to confuse with ifuslotid
#        #needs  list of fits
#        #needs fplane
#        #needs dither
#        #


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


        self.rot = None
        self.ifux = None
        self.ifuy = None
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

        #calculate the RA and DEC of each emission line object
        #remember, we are only using a single IFU per call, so all emissions belong to the same IFU
        self.rot = 360 - (90+1.8+self.parangle)
        self.tangentplane = TP(self.tel_ra, self.tel_dec, 360. - (90 + 1.3 + self.rot))
        #wants the slot id as a 0 padded string ie. '073' instead of the int (73)
        self.ifux = self.fplane.by_ifuslot(self.ifu_slot_id).x
        self.ifuy = self.fplane.by_ifuslot(self.ifu_slot_id).y

        #todo: expand emission objects with the data needed for cutouts and spectra
        for e in self.emis_list: #yes this right: x + ifuy, y + ifux
            e.ra, e.dec = self.tangentplane.xy2raDec(e.x + self.ifuy, e.y + self.ifux)
            log.info("Emission Detection #%d RA=%g , Dec=%g" % (e.id,e.ra,e.dec))

    #end HETDEX::__init__()


    def get_ifu_centers(self,args):
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


    def build_fits_list(self):
        #read in all fits
        #get the key fits header values

        #only one dither object, but has array of (3) for each value
        #these are in "dither" order
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

                self.sci_fits.append(HetdexFits(fn,e_fn,fe_fn,dit_idx))
            dit_idx += 1

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


    def get_sci_fits(self,dither,side):
        for s in self.sci_fits:
            if ((s.dither_index == dither) and (s.side == side)):
                return s
        return None

    def get_emission_detect(self,detectid):
        for e in self.emis_list:
            if e.id == detectid:
                return e
        return None


    def build_hetdex_data_page(self,pages,detectid):
        datakeep = self.build_hetdex_data_dict(detectid)
        fig = None
        if datakeep is not None:
            if datakeep['xi']:
                fig = self.build_2d_image(datakeep)

        if fig is not None:
            pages.append(fig)

        return pages


    def clean_data_dict(self,datadict=None):
        if datadict is not None:
            dd = datadict
            for k in dd.keys():
                del dd[k][:]
        else:
            dd = {}
            dd['dit'] = []
            dd['side'] = []
            dd['fib'] = []
            dd['xi'] = []
            dd['yi'] = []
            dd['xl'] = []
            dd['yl'] = []
            dd['xh'] = []
            dd['yh'] = []
            dd['sn'] = []
            dd['d'] = []
            dd['dx'] = []
            dd['dy'] = []
            dd['im'] = []
            dd['vmin1'] = []
            dd['vmax1'] = []
            dd['vmin2'] = []
            dd['vmax2'] = []
            dd['err'] = []
            dd['pix'] = []
            dd['spec'] = []
            dd['specwave'] = []
            dd['cos'] = []
            dd['ra'] = []
            dd['dec'] = []
        return dd


    def build_hetdex_data_dict(self,detectid):
        #basically cloned from Greg Z. make_visualization_detect.py; adjusted a bit for this code base
        datakeep = self.clean_data_dict()

        #get the correct emssion
        e = self.get_emission_detect(detectid)

        if e is None:
            log.error("Could not identify correct emission to plot. Detect ID = %d" %detectid)
            return None

        for side in SIDE:  # 'L' and 'R'
            for dither in range(len(self.dither.dx)):  # so, dither is 0,1,2
                dx = e.x - self.ifu_ctr.xifu[side] - self.dither.dx[dither]  # IFU is my self.ifu_ctr
                dy = e.y - self.ifu_ctr.yifu[side] - self.dither.dy[dither]

                d = np.sqrt(dx ** 2 + dy ** 2)

                # all locations (fiber array index) within dist_thresh of the x,y sky coords of the detection
                locations = np.where(d < dist_thresh)[0]

                #this is for one side of one dither of one ifu
                for loc in locations:
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
                    datakeep['xi'].append(xi)
                    datakeep['yi'].append(yi)
                    datakeep['xl'].append(xl)
                    datakeep['yl'].append(yl)
                    datakeep['xh'].append(xh)
                    datakeep['yh'].append(yh)
                    datakeep['d'].append(d[loc])
                    datakeep['sn'].append(e.sigma)

                    sci = self.get_sci_fits(dither,side)
                    if sci is not None:
                        datakeep['im'].append(sci.data[yl:yh,xl:xh])

                        # this is probably for the 1d spectra
                        I = sci.data.ravel()
                        s_ind = np.argsort(I)
                        len_s = len(s_ind)
                        s_rank = np.arange(len_s)
                        p = np.polyfit(s_rank - len_s / 2, I[s_ind], 1)

                        z1 = I[s_ind[int(len_s / 2)]] + p[0] * (1 - len_s / 2) / contrast1
                        z2 = I[s_ind[int(len_s / 2)]] + p[0] * (len_s - len_s / 2) / contrast1

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
                        Fe_indl = np.searchsorted(wave,e.w-ww,side='left')
                        Fe_indh = np.searchsorted(wave,e.w+ww,side='right')
                        datakeep['spec'].append(sci.fe_data[loc,Fe_indl:(Fe_indh+1)])
                        datakeep['specwave'].append(wave[Fe_indl:(Fe_indh+1)])

        return datakeep


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

        ind = sorted(range(len(datakeep['d'])), key=lambda k: datakeep['d'][k],
                     reverse=True)
        for i in range(num):
            borplot = plt.axes([borderxl + 0. * dx, borderyb + i * dy, 3 * dx, dy])
            implot = plt.axes([borderxl + 2. * dx - bordbuff / 3., borderyb + i * dy + bordbuff / 2., dx1, dy1])
            errplot = plt.axes(
                [borderxl + 1. * dx + 1 * bordbuff / 3., borderyb + i * dy + bordbuff / 2., dx1, dy1])
            cosplot = plt.axes([borderxl + 0. * dx + bordbuff / 2., borderyb + i * dy + bordbuff / 2., dx1, dy1])
            autoAxis = borplot.axis()
            rec = plt.Rectangle((autoAxis[0] + bordbuff / 2., autoAxis[2] + bordbuff / 2.),
                                (autoAxis[1] - autoAxis[0]) * (1. - bordbuff),
                                (autoAxis[3] - autoAxis[2]) * (1. - bordbuff), fill=False, lw=3,
                                color=colors[i, 0:3], zorder=1)
            rec = borplot.add_patch(rec)
            borplot.set_xticks([])
            borplot.set_yticks([])
            borplot.axis('off')
            ext = list(np.hstack([datakeep['xl'][ind[i]], datakeep['xh'][ind[i]],
                                  datakeep['yl'][ind[i]], datakeep['yh'][ind[i]]]))
            GF = gaussian_filter(datakeep['im'][ind[i]], (2, 1))
            implot.imshow(GF,
                          origin="lower", cmap=cmap,
                          interpolation="nearest", vmin=datakeep['vmin1'][ind[i]],
                          vmax=datakeep['vmax1'][ind[i]],
                          extent=ext)
            implot.scatter(datakeep['xi'][ind[i]], datakeep['yi'][ind[i]],
                           marker='.', c='r', edgecolor='r', s=10)
            implot.set_xticks([])
            implot.set_yticks([])
            implot.axis(ext)
            implot.axis('off')
            errplot.imshow(datakeep['pix'][ind[i]],
                           origin="lower", cmap=plt.get_cmap('gray'),
                           interpolation="nearest", vmin=0.9, vmax=1.1,
                           extent=ext)
            errplot.scatter(datakeep['xi'][ind[i]], datakeep['yi'][ind[i]],
                            marker='.', c='r', edgecolor='r', s=10)
            errplot.set_xticks([])
            errplot.set_yticks([])
            errplot.axis(ext)
            errplot.axis('off')

            #a = datakeep['cos'][ind[i]]
            a = datakeep['im'][ind[i]] #was the cosmic removed, but will assume that to be the case
            a = np.ma.masked_where(a == 0, a)
            cmap1 = cmap
            cmap1.set_bad(color=[0.2, 1.0, 0.23])
            cosplot.imshow(a,
                           origin="lower", cmap=cmap1,
                           interpolation="nearest", vmin=datakeep['vmin2'][ind[i]],
                           vmax=datakeep['vmax2'][ind[i]],
                           extent=ext)
            cosplot.scatter(datakeep['xi'][ind[i]], datakeep['yi'][ind[i]],
                            marker='.', c='r', edgecolor='r', s=10)
            cosplot.set_xticks([])
            cosplot.set_yticks([])
            cosplot.axis(ext)
            cosplot.axis('off')

            xi = datakeep['xi'][ind[i]]
            yi = datakeep['yi'][ind[i]]
            xl = int(np.round(xi - ext[0] - res[0] / 2.))
            xh = int(np.round(xi - ext[0] + res[0] / 2.))
            yl = int(np.round(yi - ext[2] - res[0] / 2.))
            yh = int(np.round(yi - ext[2] + res[0] / 2.))
            S = np.where(datakeep['err'][ind[i]][yl:yh, xl:xh] < 0, 0., datakeep['im'][ind[i]][yl:yh, xl:xh]).sum()
            N = np.sqrt(np.where(datakeep['err'][ind[i]][yl:yh, xl:xh] < 0, 0.,
                                 datakeep['err'][ind[i]][yl:yh, xl:xh] ** 2).sum())
            sn = S / N

            implot.text(0.9, .75, num - i,
                        transform=implot.transAxes, fontsize=6, color=colors[i, 0:3],
                        verticalalignment='bottom', horizontalalignment='left')

            implot.text(1.10, .75, 'S/N = %0.2f' % (sn),
                        transform=implot.transAxes, fontsize=6, color='r',
                        verticalalignment='bottom', horizontalalignment='left')
            implot.text(1.10, .55, 'D(") = %0.2f' % (datakeep['d'][ind[i]]),
                        transform=implot.transAxes, fontsize=6, color='r',
                        verticalalignment='bottom', horizontalalignment='left')
            implot.text(1.10, .35, 'X,Y = %d,%d' % (datakeep['xi'][ind[i]], datakeep['yi'][ind[i]]),
                        transform=implot.transAxes, fontsize=6, color='b',
                        verticalalignment='bottom', horizontalalignment='left')
            implot.text(1.10, .15, 'D,S,F = %d,%s,%d' % (datakeep['dit'][ind[i]], datakeep['side'][ind[i]],
                                                         datakeep['fib'][ind[i]]),
                        transform=implot.transAxes, fontsize=6, color='b',
                        verticalalignment='bottom', horizontalalignment='left')
            if i == (N - 1):
                implot.text(0.5, .85, 'Image',
                            transform=implot.transAxes, fontsize=8, color='b',
                            verticalalignment='bottom', horizontalalignment='center')
                errplot.text(0.5, .85, 'Error',
                             transform=errplot.transAxes, fontsize=8, color='b',
                             verticalalignment='bottom', horizontalalignment='center')
                cosplot.text(0.5, .85, 'Mask',
                             transform=cosplot.transAxes, fontsize=8, color='b',
                             verticalalignment='bottom', horizontalalignment='center')

        #fig.savefig(outfile, dpi=150)
        #plt.close(fig)
        plt.show()

        return fig

        #end HETDEX class