
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


from pyhetdex.cure.distortion import Distortion
import pyhetdex.tools.files.file_tools as ft
from pyhetdex.het.ifu_centers import IFUCenter
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection import TangentPlane as TP

log = global_config.logging.getLogger('Cat_logger')
log.setLevel(global_config.logging.DEBUG)




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


class HETDEX_Fits:
    '''HETDEX fits file ... 2D spectra, expected to be science file'''

    #needs open with basic validation
    #

    def __init__(self):
        pass

class HETDEX:

    #needs dither file, detect_line file, 2D fits files (i.e. 6 for 3 dither positions)
    #needs find fplane file (from observation date in fits files? or should it be passed in?)

    def __init__(self):
        pass