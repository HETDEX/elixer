from __future__ import print_function
import logging
import os.path as op
from datetime import datetime
import numpy as np

try:
    from hetdex_api.config import HDRconfig
except:
    print("Warning! Cannot find or import HDRconfig from hetdex_api!!")
    print("Defaulting to local ELiXer configuration")




#from guppy import hpy
#HPY = hpy()
#import gc

#catalogs are defined at top of catalogs.py
import socket
hostname = socket.gethostname()

#version
__version__ = '1.8.3a2'

#python version
import sys
PYTHON_MAJOR_VERSION = sys.version_info[0]
PYTHON_VERSION = sys.version_info
if sys.byteorder == 'big':
    BIG_ENDIAN = True
else:
    BIG_ENDIAN = False

LAUNCH_PDF_VIEWER = None

valid_HDR_Versions = [1,2]

HDR_Version = 2

HDR_DATA_BASEPATH = "/data/03946/hetdex"
HDR_WORK_BASEPATH = "/work/03946/hetdex/"
HDR_SCRATCH_BASEPATH = "/scratch/03946/hetdex/"
HDR_BASEPATH = HDR_WORK_BASEPATH

HDF5_DETECT_FN = None
HDF5_CONTINUUM_FN = None
HDF5_SURVEY_FN = None
OBSERVATIONS_BASEDIR = None
BAD_AMP_LIST = None
CONFIG_BASEDIR = None
PANACEA_RED_BASEDIR = None
PANACEA_RED_BASEDIR_DEFAULT = None
PANACEA_HDF5_BASEDIR = None
PIXFLT_LOC = None

CANDELS_EGS_Stefanon_2016_BASE_PATH = None
EGS_CFHTLS_PATH = None
CFHTLS_PHOTOZ_CAT = None

EGS_GROTH_BASE_PATH = None
EGS_GROTH_CAT_PATH = None

GOODS_N_BASE_PATH = None
GOODS_N_CAT_PATH = None

STACK_COSMOS_BASE_PATH = None
STACK_COSMOS_CAT_PATH = None
COSMOS_EXTRA_PATH = None

DECAM_IMAGE_PATH = None
SHELA_BASE_PATH = None

SHELA_CAT_PATH = None
SHELA_PHOTO_Z_COMBINED_PATH = None
SHELA_PHOTO_Z_MASTER_PATH = None

HSC_BASE_PATH = None
HSC_CAT_PATH = None
HSC_IMAGE_PATH = None
#HSC_AUX_IMAGE_PATH = None #not used anymore

KPNO_BASE_PATH = None
KPNO_CAT_PATH = None
KPNO_IMAGE_PATH = None

HETDEX_API_CONFIG = None

if hostname == "z50":  # primary author test box
    HDR_Version = 1
    LAUNCH_PDF_VIEWER = 'qpdfview'
else:
    HDR_Version = 1  # default HDR Version if not specified


#look specifically (and only) for HDR version on call
args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

if "--hdr" in args: #overide default if specified on command line
    try:
        i = args.index("--hdr")
        if i != -1:
            HDR_Version = int(sys.argv[i + 1]) #todo: will this ever not be an integer?
    except:
        pass

    if not (HDR_Version in valid_HDR_Versions) and HDR_Version != 0:
        print("Invalid --hdr specified (%d). Valid choices are: %s" % (HDR_Version, valid_HDR_Versions))
        exit(-1)


def set_hdr_basepath(version=None):
    """
    Sets the globals to be used

    :param version: should be an integer 1 or 2 (as of 2020/02/01) ... higher numbers after
    :return:
    """
    global HDR_DATA_BASEPATH, HDR_SCRATCH_BASEPATH, HDR_WORK_BASEPATH, HDR_BASEPATH, HDR_Version, HETDEX_API_CONFIG


    if version is None:
        version = HDR_Version

    if HETDEX_API_CONFIG is None:
        if version != 0:
            strHDRVersion = f"hdr{version}"
        elif hostname == "z50":
            strHDRVersion = f"hdr{HDR_Version}"
        else: #this is a problem
            print("Invalid HDRversion configuration")
            return

        try:
            HETDEX_API_CONFIG = HDRconfig(survey=strHDRVersion)
        except Exception as e:
            print(e)

    hdr_dir = ""
    # _DATA_, _SCRATCH_ _WORK_ all specific to ELiXer, but the BASEPATH should use HETEDEX_API defined if possible
    if version != 0:
        hdr_dir = f"hdr{version}" #hdr1, hdr2, ....
        HDR_DATA_BASEPATH = op.join(HDR_DATA_BASEPATH,hdr_dir)
        HDR_SCRATCH_BASEPATH = op.join(HDR_SCRATCH_BASEPATH,hdr_dir)
        HDR_WORK_BASEPATH = op.join(HDR_WORK_BASEPATH,hdr_dir)
        HDR_BASEPATH = op.join(HDR_BASEPATH,hdr_dir)
    elif hostname == "z50": #author test box
        hdr_dir = "hdr1"
        print(f"*** using {hdr_dir}")
        HDR_DATA_BASEPATH = op.join(HDR_DATA_BASEPATH, hdr_dir)
        HDR_SCRATCH_BASEPATH = op.join(HDR_SCRATCH_BASEPATH, hdr_dir)
        HDR_WORK_BASEPATH = op.join(HDR_WORK_BASEPATH, hdr_dir)
        HDR_BASEPATH = op.join(HDR_BASEPATH, hdr_dir)

    if HETDEX_API_CONFIG:
        HDR_BASEPATH = HETDEX_API_CONFIG.hdr_dir[hdr_dir]


def select_hdr_version(version):
    """
    Internal function to configure HDR specific paths.

    :param version: which version to set
    :return: bool for success (True)
    """
    #one per line to make it easy to read/find
    global HDR_Version
    global valid_HDR_Versions

    global HDF5_DETECT_FN
    global HDF5_CONTINUUM_FN
    global HDF5_SURVEY_FN
    global OBSERVATIONS_BASEDIR
    global BAD_AMP_LIST
    global CONFIG_BASEDIR
    global PANACEA_RED_BASEDIR
    global PANACEA_RED_BASEDIR_DEFAULT
    global PANACEA_HDF5_BASEDIR
    global PIXFLT_LOC

    global CANDELS_EGS_Stefanon_2016_BASE_PATH
    global EGS_CFHTLS_PATH
    global CFHTLS_PHOTOZ_CAT

    global EGS_GROTH_BASE_PATH
    global EGS_GROTH_CAT_PATH

    global GOODS_N_BASE_PATH
    global GOODS_N_CAT_PATH

    global STACK_COSMOS_BASE_PATH
    global STACK_COSMOS_CAT_PATH
    global COSMOS_EXTRA_PATH

    global DECAM_IMAGE_PATH
    global SHELA_BASE_PATH

    global SHELA_CAT_PATH
    global SHELA_PHOTO_Z_COMBINED_PATH
    global SHELA_PHOTO_Z_MASTER_PATH

    global HSC_BASE_PATH
    global HSC_CAT_PATH
    global HSC_IMAGE_PATH
    global HSC_S15A
    #global HSC_AUX_IMAGE_PATH

    global KPNO_BASE_PATH
    global KPNO_CAT_PATH
    global KPNO_IMAGE_PATH

    global LAUNCH_PDF_VIEWER #for debug machine only

    global HETDEX_API_CONFIG

    #make sure we have a valid version to select
    if not (version in valid_HDR_Versions) and version != 0:
        print("Invalid HDR version specified (%d). Valid choices are: %s" % (version, valid_HDR_Versions))
        return False

    HDR_Version = version
    set_hdr_basepath(version)

    BAD_AMP_LIST = "/work/03261/polonius/maverick/catalogs/bad_amp_list.txt" #not really used anymore

    normal_build = True
    if (hostname == "z50") and (version == 0):  #author test box:
        if False: #for debugging
            normal_build = False
            HDF5_DETECT_FN = "/work/03946/hetdex/hdr1/detect/detect_hdr1.h5"
            HDF5_CONTINUUM_FN = "/work/03946/hetdex/hdr1/detect/continuum_sources.h5"
            HDF5_SURVEY_FN = "/work/03946/hetdex/hdr1/survey/survey_hdr1.h5"

            OBSERVATIONS_BASEDIR = "/work/03946/hetdex/hdr1/reduction/"
            BAD_AMP_LIST = "/home/dustin/code/python/elixer/bad_amp_list.txt"

            CONFIG_BASEDIR = "/work/03946/hetdex/hdr1/software/"
            PANACEA_RED_BASEDIR = "/work/03946/hetdex/hdr1/raw/red1/reductions/"
            PANACEA_RED_BASEDIR_DEFAULT = PANACEA_RED_BASEDIR
            PANACEA_HDF5_BASEDIR = "/work/03946/hetdex/hdr1/reduction/data"

            CANDELS_EGS_Stefanon_2016_BASE_PATH = "/home/dustin/code/python/elixer/data/EGS"
            EGS_CFHTLS_PATH = "/home/dustin/code/python/elixer/data/CFHTLS"
            CFHTLS_PHOTOZ_CAT = "/home/dustin/code/python/elixer/data/CFHTLS/photozCFHTLS-W3_270912.out"
            GOODS_N_BASE_PATH = "/home/dustin/code/python/elixer/data/GOODSN/"
            GOODS_N_CAT_PATH = GOODS_N_BASE_PATH

            EGS_GROTH_BASE_PATH = "/home/dustin/code/python/elixer/data/isak"
            EGS_GROTH_CAT_PATH = EGS_GROTH_BASE_PATH  # note: there is no catalog

            STACK_COSMOS_BASE_PATH = "/home/dustin/code/python/elixer/data/isak"
            STACK_COSMOS_CAT_PATH = "/home/dustin/code/python/elixer/data/isak"
            COSMOS_EXTRA_PATH = "/home/dustin/code/python/elixer/data/"

            SHELA_BASE_PATH = "/media/dustin/dd/hetdex/data/SHELA"  # "/home/dustin/code/python/elixer/data/isak/SHELA"
            DECAM_IMAGE_PATH = SHELA_BASE_PATH  # "/media/dustin/dd/hetdex/data/decam/images"
            SHELA_CAT_PATH = "/media/dustin/dd/hetdex/data/SHELA"  # "/home/dustin/code/python/elixer/data/isak/SHELA"
            SHELA_PHOTO_Z_COMBINED_PATH = "/home/dustin/code/python/elixer/data/isak/SHELA"
            SHELA_PHOTO_Z_MASTER_PATH = "/home/dustin/code/python/elixer/data/isak/SHELA"

            # 2019-08-06 (mshiro base path inaccessible)
            # HSC_BASE_PATH = "/work/04094/mshiro/maverick/HSC/S15A/reduced"
            # HSC_CAT_PATH = "/media/dustin/dd/hetdex/data/HSC/catalog_tracts" #"/work/04094/mshiro/maverick/HSC/S15A/reduced/catalog_tracts"
            # HSC_IMAGE_PATH = "/work/04094/mshiro/maverick/HSC/S15A/reduced/images"

            if op.exists("/work/03946/hetdex/hdr2/imaging/hsc"):
                HSC_BASE_PATH = "/work/03946/hetdex/hdr2/imaging/hsc"
                HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
                HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"
                #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
            else:
                HSC_BASE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced"
                HSC_CAT_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/catalog_tracts"
                HSC_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
                #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"

            DECALS_BASE_PATH = "/media/dustin/dd/hetdex/data/decals"
            DECALS_CAT_PATH = "/media/dustin/dd/hetdex/data/decals"
            DECALS_IMAGE_PATH = "/media/dustin/dd/hetdex/data/decals"

            # KPNO_BASE_PATH = "/work/03261/polonius/hetdex/catalogs/KPNO_Mosaic"
            KPNO_BASE_PATH = "/work/03233/jf5007/maverick/KMImaging/"
            KPNO_CAT_PATH = HSC_BASE_PATH
            KPNO_IMAGE_PATH = HSC_BASE_PATH

    if normal_build:

        if HETDEX_API_CONFIG:
            HDF5_DETECT_FN = HETDEX_API_CONFIG.detecth5
            HDF5_CONTINUUM_FN = HETDEX_API_CONFIG.contsourceh5
            HDF5_SURVEY_FN = HETDEX_API_CONFIG.surveyh5
            OBSERVATIONS_BASEDIR = HETDEX_API_CONFIG.red_dir
            CONFIG_BASEDIR = HETDEX_API_CONFIG.software_dir
            HDF5_RAW_DIR = HETDEX_API_CONFIG.raw_dir #local to this function only
            HDF5_REDUCTION_DIR = HETDEX_API_CONFIG.red_dir #local to this function only

            if HDR_Version == 1:
                PIXFLT_LOC = op.join(CONFIG_BASEDIR, "virus_config/PixelFlats")
            else:
                PIXFLT_LOC = HETDEX_API_CONFIG.pixflat_dir

        else:
            HDF5_DETECT_FN = op.join(HDR_BASEPATH, "detect/detect_hdr1.h5")
            HDF5_CONTINUUM_FN = op.join(HDR_BASEPATH, "detect/continuum_sources.h5")
            HDF5_SURVEY_FN = op.join(HDR_BASEPATH, "survey/survey_hdr1.h5")
            OBSERVATIONS_BASEDIR = op.join(HDR_BASEPATH, "reduction/")
            CONFIG_BASEDIR = op.join(HDR_BASEPATH, "software/")
            HDF5_RAW_DIR = op.join(HDR_BASEPATH, "raw/") #local to this function only
            HDF5_REDUCTION_DIR = op.join(HDR_BASEPATH, "reduction/") #local to this function only

        PANACEA_RED_BASEDIR = op.join(HDF5_RAW_DIR, "red1/reductions/")
        PANACEA_RED_BASEDIR_DEFAULT = PANACEA_RED_BASEDIR
        PANACEA_HDF5_BASEDIR = op.join(HDR_BASEPATH, "reduction/data")

        #
        # Imaging Data Paths
        #

        #see if HDR2 exists yet
        # if op.exists(op.join(HDR_BASEPATH, "imaging/candles_egs/EGS")):
        #     hdr_imaging_basepath = op.join(HDR_BASEPATH,"imaging")
        # else: #temporary code back to HDR1
        #     #print("***** using hdr1 path for imaging *****")
        #     #hdr_imaging_basepath = "/work/03946/hetdex/hdr1/"
        #     print("***** using /data/03261/polonius/hdr2 for imaging *****")
        #     hdr_imaging_basepath = "/data/03261/polonius/hdr2/"

        if HETDEX_API_CONFIG:
            try:
                hdr_imaging_basepath = HETDEX_API_CONFIG.imaging_dir
            except:
                print("***** using /data/03261/polonius/hdr2 for imaging *****")
                hdr_imaging_basepath = "/data/03261/polonius/hdr2/"
        else:
            print("***** using /data/03261/polonius/hdr2 for imaging *****")
            hdr_imaging_basepath = "/data/03261/polonius/hdr2/"

        CANDELS_EGS_Stefanon_2016_BASE_PATH = op.join(hdr_imaging_basepath, "candles_egs/EGS")
        EGS_CFHTLS_PATH = op.join(hdr_imaging_basepath, "candles_egs/CFHTLS")
        CFHTLS_PHOTOZ_CAT = op.join(hdr_imaging_basepath, "candles_egs/CFHTLS/photozCFHTLS-W3_270912.out")

        EGS_GROTH_BASE_PATH = op.join(hdr_imaging_basepath, "candles_egs/groth")
        EGS_GROTH_CAT_PATH = op.join(hdr_imaging_basepath, "candles_egs/groth")  # note: there is no catalog

        #GOODS_N_BASE_PATH = "/work/03564/stevenf/maverick/GOODSN"
        GOODS_N_BASE_PATH = op.join(hdr_imaging_basepath,"goods_north/GOODSN")
        GOODS_N_CAT_PATH = GOODS_N_BASE_PATH

        STACK_COSMOS_BASE_PATH = op.join(hdr_imaging_basepath, "cosmos/stackCOSMOS/nano/")
        STACK_COSMOS_CAT_PATH = op.join(hdr_imaging_basepath, "cosmos/stackCOSMOS")
        COSMOS_EXTRA_PATH = op.join(hdr_imaging_basepath, "cosmos/COSMOS/")

        DECAM_IMAGE_PATH = op.join(hdr_imaging_basepath, "shela/nano/")
        SHELA_BASE_PATH = op.join(hdr_imaging_basepath, "shela/nano/")
        SHELA_CAT_PATH = SHELA_BASE_PATH
        SHELA_PHOTO_Z_COMBINED_PATH = op.join(hdr_imaging_basepath, "shela/SHELA")
        SHELA_PHOTO_Z_MASTER_PATH = op.join(hdr_imaging_basepath, "shela/SHELA")

        HSC_S15A = False
        if op.exists(op.join(hdr_imaging_basepath,"hsc")):
            if HDR_Version == 1:
                if op.exists("/work/03946/hetdex/hdr2/imaging/hsc"):
                    HSC_BASE_PATH = "/work/03946/hetdex/hdr2/imaging/hsc"
                    HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
                    HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"
                else: #us the actual HSC data available at HDR1 time
                    HSC_BASE_PATH = op.join(hdr_imaging_basepath, "imaging/hsc/S15A/reduced")
                    HSC_CAT_PATH = HSC_BASE_PATH + "/catalog_tracts"
                    HSC_IMAGE_PATH = HSC_BASE_PATH + "/images"
                    HSC_S15A = True
            else:
                HSC_BASE_PATH = op.join(hdr_imaging_basepath,"hsc")
                HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
                HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"
            #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
        elif op.exists("/work/03946/hetdex/hdr2/imaging/hsc"):
            HSC_BASE_PATH = "/work/03946/hetdex/hdr2/imaging/hsc"
            HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
            HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"
            #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
        else:
            HSC_BASE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced"
            HSC_CAT_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/catalog_tracts"
            HSC_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
            #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"

        #temporary
        # HSC_BASE_PATH = op.join(HDR_BASEPATH, "imaging/hsc/S15A/reduced")
        # HSC_CAT_PATH = op.join(HDR_BASEPATH, "imaging/hsc/S15A/reduced/catalog_tracts")
        # HSC_IMAGE_PATH = op.join(HDR_BASEPATH, "imaging/hsc/S15A/reduced/images")
        # HSC_AUX_IMAGE_PATH = op.join(HDR_BASEPATH, "imaging/hsc/S15A/reduced/images")

        # KPNO_BASE_PATH = "/work/03261/polonius/hetdex/catalogs/KPNO_Mosaic"
        if op.exists(op.join(hdr_imaging_basepath, "KMImaging")):
            KPNO_BASE_PATH = op.join(hdr_imaging_basepath, "KMImaging")
        else:
            KPNO_BASE_PATH = "/work/03233/jf5007/maverick/KMImaging/"
        KPNO_CAT_PATH = HSC_BASE_PATH
        KPNO_IMAGE_PATH = HSC_BASE_PATH

    return True  # end select_hdr_version


###########################################
# configure the HDR specific paths...
##########################################
select_hdr_version(HDR_Version)


VIRUS_CONFIG = op.join(CONFIG_BASEDIR,"virus_config")
FPLANE_LOC = op.join(CONFIG_BASEDIR,"virus_config/fplane")
IFUCEN_LOC = op.join(CONFIG_BASEDIR,"virus_config/IFUcen_files")
DIST_LOC = op.join(CONFIG_BASEDIR,"virus_config/DeformerDefaults")

if PIXFLT_LOC is None:
    if HDR_Version != 1:
        print("***** temporary hard code pixel flat location *****")
        PIXFLT_LOC = "/data/00115/gebhardt/lib_calib/lib_pflat"
    else:
        PIXFLT_LOC = op.join(CONFIG_BASEDIR, "virus_config/PixelFlats")


REPORT_ELIXER_MCMC_FIT = False

RELATIVE_PATH_UNIVERSE_CONFIG = "line_classifier/universe.cfg"
RELATIVE_PATH_FLUX_LIM_FN = "line_classifier/Line_flux_limit_5_sigma_baseline.dat"

if hostname == "z50":  # primary author test box
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

if "--log" in args: #overide default if specified on command line
    try:
        i = args.index("--log")
        if i != -1:
            log_level = str(sys.argv[i + 1]).lower()

        if log_level == "debug":
            LOG_LEVEL = logging.DEBUG
        elif log_level == "info":
            LOG_LEVEL = logging.INFO
        elif log_level == "error":
            LOG_LEVEL = logging.ERROR
        elif log_level == "critical":
            LOG_LEVEL = logging.CRITICAL
        else:
            pass # leave as is
    except:
        pass



##log initialization moved to elixer.py to incorporate --name into filename
# reminder to self ... this is pointless with SLURM given the bash wraper (which does not know about the
# specific dir name and just builds elixer.run ... so leave this here
LOG_FILENAME = "elixer.log"
#loggin intialization moved to elixer.py in parse_commandline as that is the first place we need to log ...
#   if --help, then the logger is not created
#logging.basicConfig(filename=LOG_FILENAME,level=LOG_LEVEL,filemode='w')
#.debug(), .info(), .warning(), .error(), .critical()



#first time we need to log anything


class Global_Logger:
    FIRST_LOG = True

    def __init__(self,id): #id is a string identifier
        self.logger = logging.getLogger(id)
        self.logger.setLevel(LOG_LEVEL)

        # self.fh = logging.FileHandler(LOG_FILENAME,"w")
        # self.fh.setLevel(LOG_LEVEL)
        # self.logger.addHandler(self.fh)

        # self.ch = logging.StreamHandler(sys.stdout)
        # self.ch.setLevel(LOG_LEVEL)
        # self.logger.addHandler(self.ch)
        #logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, filemode='w')
        #   #don't set the global log level, else imported packages might start logging
        # well that does not quite work ...

        if self.__class__.FIRST_LOG:
            logging.basicConfig(filename=LOG_FILENAME, filemode='w+')
            self.__class__.FIRST_LOG = False
        else:
            logging.basicConfig(filename=LOG_FILENAME, filemode='a')


    def add_time(self,msg):

        #if self.LOGGER_INITIALIZED == False:
        #    logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, filemode='w')
        #    self.LOGGER_INITIALIZED = True

        try:
            d = datetime.now()
            msg = "[%s:%s:%s.%s]  %s" %(str(d.hour).zfill(2),str(d.minute).zfill(2),str(d.second).zfill(2),
                                        str(d.microsecond).zfill(6),msg)
            return msg
        except:
            return msg


    def setlevel(self,level):
        try:
            self.logger.setLevel(level)
        except:
            print("Exception in logger....")

    def debug(self,msg,exc_info=False):
        try:
            msg = self.add_time(msg)
            self.logger.debug(msg,exc_info=exc_info)
        except:
            print("Exception in logger....")

    def info(self,msg,exc_info=False):
        try:
            msg = self.add_time(msg)
            self.logger.info(msg,exc_info=exc_info)
        except:
            print("Exception in logger....")

    def warning(self,msg,exc_info=False):
        try:
            msg = self.add_time(msg)
            self.logger.warning(msg,exc_info=exc_info)
        except:
            print("Exception in logger....")

    def error(self,msg,exc_info=False):
        try:
            msg = self.add_time(msg)
            self.logger.error(msg,exc_info=exc_info)
        except:
            print("Exception in logger....")

    def critical(self, msg, exc_info=False):
        try:
            msg = self.add_time(msg)
            self.logger.critical(msg, exc_info=exc_info)
        except:
            print("Exception in logger....")


def python2():
    if PYTHON_MAJOR_VERSION == 2:
        return True
    else:
        return False

def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx



LyA_rest = 1216. #A 1215.668 and 1215.674
OII_rest = 3727.

#FLUX_CONVERSION = (1./60)*1e-17
HETDEX_FLUX_BASE_CGS = 1e-17
HETDEX_CONTINUUM_MAG_LIMIT = 25.0 #generous, truth is closer to 24.few
# 1.35e-18 ~ 24.0 mag in g-band
# 8.52e-19 ~ 24.5 mag in g-band,
# 5.38e-19 ~ 25.0 mag in g-band
HETDEX_CONTINUUM_FLUX_LIMIT =  5.38e-19

CONTINUUM_FLOOR_COUNTS = 6.5 #5 sigma * 6 counts / sqrt(40 angstroms/1.9 angs per pixel)

Fiber_Radius = 0.75 #arcsec
IFU_Width = 47.26 #arcsec ... includes 2x fiber radius ... more narrow part fiber 1 - 19, else is 49.8
IFU_Height = 49.98 #arcsec
Fiber_1_X = -22.88
Fiber_1_Y = -24.24
Fiber_19_X = 22.88
Fiber_19_Y = -24.24
Fiber_430_X = -22.88
Fiber_430_Y = 24.24
PreferCosmicCleaned = True #use cosmic cleaned FITS data if available (note: assumes filename is 'c' + filename)

Figure_DPI = 300
FIGURE_SZ_X = 18 #18
#FIGURE_SZ_Y = 9 #12
GRID_SZ_X = 3 # equivalent figure_sz_x for a grid width (e.g. one column)
GRID_SZ_Y = 3 # equivalent figure_sz_y for a grid height (e.g. one row)

LyC = False #switch for Lyman Continuum specialized code
PLOT_FULLWIDTH_2D_SPEC = False #if true, show the combined full-width 2D spectra just under the 1D plot

FIT_FULL_SPEC_IN_WINDOW = False #if true, allow y-axis range to fit entire spectrum, not just the emission line
SHOW_ALL_1D_SPECTRA = False #if true, plot the full width 1D spectra for each hetdex fiber in detection
MAX_COMBINE_BID_TARGETS = 3 #if SINGLE_PAGE_PER_DETECT is true, this is the max number of bid targets that can be
                            #merged on a single line. If the number is greater, each bid target gets its own line

#WARNING! As of version 1.1.10 this should ALWAYS be True ... do not change unless you really
#know what you are doing!!!
SINGLE_PAGE_PER_DETECT = True #if true, a single pdf page per emission line detection is made
FORCE_SINGLE_PAGE = True
SHOW_SKYLINES = True

#1 fiber (the edge-most) fiber
CCD_EDGE_FIBERS_BOTTOM = range(1,20)
CCD_EDGE_FIBERS_TOP = range(439,449)
CCD_EDGE_FIBERS_LEFT = [1,20,40,59,79,98,118,137,157,176,196,215,235,254,274,293,313,332,352,371,391,410,430]
CCD_EDGE_FIBERS_RIGHT = [19,39,58,78,97,117,136,156,175,195,214,234,253,273,292,312,331,351,370,390,409,429,448]
#CCD_EDGE_FIBERS_ALL = list(set(CCD_EDGE_FIBERS_BOTTOM+CCD_EDGE_FIBERS_TOP+CCD_EDGE_FIBERS_LEFT+CCD_EDGE_FIBERS_RIGHT))
CCD_EDGE_FIBERS_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 39, 40, 58, 59, 78,
                         79, 97, 98, 117, 118, 136, 137, 156, 157, 175, 176, 195, 196, 214, 215, 234, 235, 253, 254,
                         273, 274, 292, 293, 312, 313, 331, 332, 351, 352, 370, 371, 390, 391, 409, 410, 429,
                         430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448]

#two fibers from edge
CCD_EDGE_FIBERS_BOTTOM_2 = range(1,40)
CCD_EDGE_FIBERS_TOP_2 = range(410,449)
CCD_EDGE_FIBERS_LEFT_2 = [1,2,20,21,40,41,59,60,79,80,98,99,118,119,137,138,157,158,176,177,196,197,215,216,
                        235,236,254,255,274,275,293,294,313,314,332,333,352,353,371,372,391,392,410,411,430,431]
CCD_EDGE_FIBERS_RIGHT_2 = [18,19,38,39,57,58,77,78,96,97,116,117,135,136,155,156,174,175,194,195,213,214,
                         233,234,252,253,272,273,291,292,311,312,330,331,350,351,369,370,389,390,408,409,428,429,447,448]
#CCD_EDGE_FIBERS_ALL = list(set(CCD_EDGE_FIBERS_BOTTOM+CCD_EDGE_FIBERS_TOP+CCD_EDGE_FIBERS_LEFT+CCD_EDGE_FIBERS_RIGHT))
CCD_EDGE_FIBERS_ALL_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 57, 58, 59, 60, 77, 78, 79, 80, 96, 97, 98, 99, 116, 117,
 118, 119, 135, 136, 137, 138, 155, 156, 157, 158, 174, 175, 176, 177, 194, 195, 196, 197, 213, 214, 215, 216, 233, 234,
 235, 236, 252, 253, 254, 255, 272, 273, 274, 275, 291, 292, 293, 294, 311, 312, 313, 314, 330, 331, 332, 333, 350, 351,
 352, 353, 369, 370, 371, 372, 389, 390, 391, 392, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445,
 446, 447, 448]

CALFIB_WAVEGRID = np.arange(3470.,5542.,2.0) #3470 - 5540

#Detection Quality Score Values
FULL_WEIGHT_DISTANCE = Fiber_Radius
ZERO_WEIGHT_DISTANCE = 4.0 * Fiber_Radius
COMPUTE_QUALITY_SCORE = False

#quadratic constants for ax^2 + bx + c for fitted quadratic to weight fall off
#QUAD_A = -1/(Fiber_Radius**2*(FULL_WEIGHT_MULT-ZERO_WEIGHT_MULT)**2)
#QUAD_B = 2*FULL_WEIGHT_MULT/(Fiber_Radius*(FULL_WEIGHT_MULT-ZERO_WEIGHT_MULT)**2)
#QUAD_C = (ZERO_WEIGHT_MULT**2-2*FULL_WEIGHT_MULT*ZERO_WEIGHT_MULT)/((FULL_WEIGHT_MULT-ZERO_WEIGHT_MULT)**2)

QUAD_A = -1/(FULL_WEIGHT_DISTANCE-ZERO_WEIGHT_DISTANCE)**2
QUAD_B = 2*FULL_WEIGHT_DISTANCE/(FULL_WEIGHT_DISTANCE-ZERO_WEIGHT_DISTANCE)**2
QUAD_C = (ZERO_WEIGHT_DISTANCE**2-2*FULL_WEIGHT_DISTANCE*ZERO_WEIGHT_DISTANCE)/((FULL_WEIGHT_DISTANCE-ZERO_WEIGHT_DISTANCE)**2)
PLOT_GAUSSIAN = True

ZOO = False #target output for Zooniverse
ZOO_CUTOUTS = False #produce the small zooniverse cutouts
ZOO_MINI = False

UNIQUE_DET_ID_NUM = 0

FLUX_WAVEBIN_WIDTH = 2.0 # AA
NEGATIVE_Z_ERROR = -0.001 #if compuated z is negative, but greater than this, assume == 0.0

MC_PLAE_SAMPLE_SIZE = 250 #number of random samples to run
MC_PLAE_CONF_INTVL = 0.68 #currently supported 0.68, 0.95, 0.99

CLASSIFY_WITH_OTHER_LINES = True
SPEC_MAX_OFFSET_SPREAD = 2.75 #AA #maximum spread in (velocity) offset (but in AA) across all lines in a solution
MIN_MCMC_SNR = 0.0 #minium SNR from an MCMC fit to accept as a real line (if 0.0, do not MCMC additional lines)
MIN_ADDL_EMIS_LINES_FOR_CLASSIFY = 1

DISPLAY_ABSORPTION_LINES = False
MAX_SCORE_ABSORPTION_LINES = 0.0 #the most an absorption line can contribute to the score (set to 0 to turn off)

MULTILINE_MIN_GOOD_ABOVE_NOISE = 4.0 #below this is not consider a possibly good line
MULTILINE_SCORE_NORM_ABOVE_NOISE = 5.0 #get full 1x score at this level
MULTILINE_SCORE_ABOVE_NOISE_MAX_BONUS = 3.0 #maximum multiplier as max of (peak/noise/NORM, BONUS)
MULTILINE_MIN_SOLUTION_SCORE = 25.0  #remember, this does NOT include the main line's score (about p(noise) = 0.01)
MULTILINE_FULL_SOLUTION_SCORE = 50.0 #scores at or above get full credit for the weight
MULTILINE_MIN_WEAK_SOLUTION_CONFIDENCE = 0.5

MULTILINE_WEIGHT_PROB_REAL = 0.4 #probabilty of real (0-.999) makes up 40% of score
MULTILINE_WEIGHT_SOLUTION_SCORE = 0.5 #related to probability of real, makes 50% of score
MULTILINE_WEIGHT_FRAC_SCORE = 0.1 #fraction of total score for all solutions makes up 10%

MULTILINE_MIN_SOLUTION_CONFIDENCE = 0.95 * MULTILINE_WEIGHT_PROB_REAL + MULTILINE_WEIGHT_SOLUTION_SCORE + 0.5 * MULTILINE_WEIGHT_FRAC_SCORE
MULTILINE_MIN_NEIGHBOR_SCORE_RATIO = 1.5 #if two top scores are possible, the best solution must be this multiplier higher in score

MULTILINE_MAX_PROB_NOISE_TO_PLOT = 0.2 #plot dashed line on spectrum if p(noise) < 0.1
MULTILINE_ALWAYS_SHOW_BEST_GUESS = True #if true, show the best guess even if it does not meet the miniumum requirements
ADDL_LINE_SCORE_BONUS = 5.0 #add for each line at 2+ lines (so 1st line adds nothing)
                            #this is rather "hand-wavy" but gives a nod to having more lines beyond just their score
SHADE_1D_SPEC_PEAKS = False #if true, shade in red the 1D spec peaks above the NORM noise limit (see below)


USE_SOURCE_EXTRACTOR = True #use source extractor to ID objects and get magnitudes ... fall back to circular aperture
                             #if source extractor code fails
DYNAMIC_MAG_APERTURE = True  #allow aperture size to change to fit maximum magnitude
MIN_DYNAMIC_MAG_RADIUS = 1.0 #in arcsec
FIXED_MAG_APERTURE = 1.5 #radius in arcsec (default: each catalog can set its own per image)
MAX_DYNAMIC_MAG_APERTURE = 3.0 #maximum growth in dynamic mag

NUDGE_MAG_APERTURE_MAX_DATE = 20180601 #nudge center only BEFORE this date (using as a proxy for the number of active IFUs)
NUDGE_MAG_APERTURE_CENTER = 1.5  #allow the center of the mag aperture to drift to the 2D Gaussian centroid
                                 #up to this distance in x and y in arcsec (if 0.0 then no drift is allowed)
NUDGE_SEP_MAX_DIST = 1.5 #allow source extractor found objects to be matched to the HETDEX target up to this distances
                          #in arcsec
MAX_SKY_SUBTRACT_MAG = 2.0 #if local sky subtraction results in a magnitude change greater than this value, do not apply it

DEBUG_SHOW_GAUSS_PLOTS = False #set on command line now --gaussplots (but keep here for compatibility with other programs)
MARK_PIXEL_FLAT_CENTER = False

MAX_ANNULUS_RADIUS = 3600.0 #ridiculously large ... need to trim this to a reasonable size
ANNULUS_FIGURE_SZ_X = 12
ANNULUS_FIGURE_SZ_Y = 12

SKY_ANNULUS_MIN_MAG = 15.0 #measure magnitude must be fainter than this to trigger sky subtraction from surrounding annulus

INCLUDE_ALL_AMPS = True #ie. if true, ignore the bad amp list

RECOVERY_RUN = False

ALLOW_EMPTY_IMAGE = False #do not return cutout if it is empty or a simple gradient (essentially, if it has no data)
#note: Pan-STARRS is prioritized over SDSS (since Pan-STARRS is deeper 23.3 vs 22.0)

DECALS_WEB_ALLOW = True #if no other catalogs match, try DECaLS as online query (default if not dispatch mode)
DECALS_WEB_FORCE = False #ignore local catalogs and Force the use of only DECaLS

PANSTARRS_ALLOW = True #if no other catalogs match, try Pan-STARRS as online query (default if not dispatch mode)
PANSTARRS_FORCE = False  #ignore local catalogs and Force the use of only Pan-STARRS

SDSS_ALLOW = True #if no other catalogs match, try SDSS as online query (default if not dispatch mode)
SDSS_FORCE = False  #ignore local catalogs and Force the use of only SDSS

USE_PHOTO_CATS = True  #default normal is True .... use photometry catalogs (if False only generate the top (HETDEX) part)

MAX_NEIGHBORS_IN_MAP = 15

BUILD_HDF5_CATALOG = True

ALLOW_SYSTEM_CALL_PDF_CONVERSION = True #if True, if the Python PDF to PNG fails, attempt a system call to pdftoppm

DISPLAY_PSEUDO_COLOR = False  #display in upper left the pseudo-color from the HETDEX spectrum

COMBINE_PLAE = True # combine (all?) PLAE/POII ratio data into a single estimate
AGGREGATE_PLAE_CLASSIFICATION = True # and combine with other data to make a single p(LAE) estimate (0.0 - 1.0)
ZEROTH_ROW_HEADER = True # and put this in the zeroth row header

#penalties for solutions that fail to account for all lines
MAX_OK_UNMATCHED_LINES = 0
MAX_OK_UNMATCHED_LINES_SCORE = MULTILINE_MIN_SOLUTION_SCORE #25.0