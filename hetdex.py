
import matplotlib
matplotlib.use('agg')

import global_config
import science_image
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

log = global_config.logging.getLogger('Cat_logger')
log.setLevel(global_config.logging.DEBUG)

VIRUS_CONFIG = "/work/03946/hetdex/maverick/virus_config"
FPLANE_LOC = "/work/03946/hetdex/maverick/virus_config/fplane"


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
                print("Warning! No valid fplane file found for the given date. Will use oldest available.")
                i = 0
            fplane = files[i]
    else:
        print ("Error. No fplane files found.")

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
        print("Error! Cannot build fplane dictionaries. No fplane file.")
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
        self.basename, self.deformer = None, None
        self.dx, self.dy = None,None
        self.seeing, self.norm, self.airmass = None,None,None
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
                    self.basename = _bn
                    self.deformer = _d
                    self.dx = float(_x)
                    self.dy = float(_y)
                    self.seeing = float(_seeing)
                    self.norm = float(_norm)
                    self.airmass = float(_airmass)
        except:
            log.error("Unable to read dither file: %s :" %dither_file, exc_info=True)


class HetdexFits:
    '''A single HETDEX fits file ... 2D spectra, expected to be science file'''

    #needs open with basic validation
    #

    def __init__(self):
        self.filename = None

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

    def __init__(self):
        pass