from __future__ import print_function
import logging
import gc

#catalogs are defined at top of catalogs.py
import socket


if socket.gethostname() == 'z50':
#if False:
    CONFIG_BASEDIR = "/home/dustin/code/python/voltron/data/config/"
    PANACEA_RED_BASEDIR = "/home/dustin/code/python/voltron/data/config/red1/reductions/"

    CANDELS_EGS_Stefanon_2016_BASE_PATH = "/home/dustin/code/python/voltron/data/EGS"

    EGS_GROTH_BASE_PATH = "/home/dustin/code/python/voltron/data/isak"
    EGS_GROTH_CAT_PATH = EGS_GROTH_BASE_PATH #note: there is no catalog

    STACK_COSMOS_BASE_PATH = "/home/dustin/code/python/voltron/data/isak"
    STACK_COSMOS_CAT_PATH = "/home/dustin/code/python/voltron/data/isak"

    SHELA_BASE_PATH = "/home/dustin/code/python/voltron/data/isak/SHELA"
    SHELA_CAT_PATH = "/home/dustin/code/python/voltron/data/isak/SHELA"
else:
    CONFIG_BASEDIR = "/work/03946/hetdex/maverick/"
    PANACEA_RED_BASEDIR = "/work/03946/hetdex/maverick/red1/reductions/"

    CANDELS_EGS_Stefanon_2016_BASE_PATH = "/work/03564/stevenf/maverick/EGS"

    EGS_GROTH_BASE_PATH = "/work/03229/iwold/maverick/groth"
    EGS_GROTH_CAT_PATH = "/work/03229/iwold/maverick/groth" #note: there is no catalog

    STACK_COSMOS_BASE_PATH = "/work/03229/iwold/maverick/stackCOSMOS/nano/"
    STACK_COSMOS_CAT_PATH = "/work/03229/iwold/maverick/stackCOSMOS"

    SHELA_BASE_PATH = "/work/03229/iwold/maverick/fall_field/stack/v2/psf/nano/"
    SHELA_CAT_PATH = SHELA_BASE_PATH

LOG_FILENAME = "voltron.log"
LOG_LEVEL = logging.DEBUG

logging.basicConfig(filename=LOG_FILENAME,level=LOG_LEVEL,filemode='w')
#.debug(), .info(), .warning(), .error(), .critical()

LyA_rest = 1216. #A 1215.668 and 1215.674
OII_rest = 3727.

FLUX_CONVERSION = (1./60)*1e-17


Fiber_Radius = 0.75 #arcsec
PreferCosmicCleaned = True #use cosmic cleaned FITS data if available (note: assumes filename is 'c' + filename)

Figure_DPI = 300
FIGURE_SZ_X = 16 #18
FIGURE_SZ_Y = 10 #12
GRID_SZ_X = 3 # equivalent figure_sz_x for a grid width (e.g. one column)
GRID_SZ_Y = 3 # equivalent figure_sz_y for a grid height (e.g. one row)

SHOW_FULL_2D_SPECTA = True #if true, plot the full width 2D spectra for each hetdex fiber in detection
SINGLE_PAGE_PER_DETECT = False #if true, a single pdf page per emission line detection is made