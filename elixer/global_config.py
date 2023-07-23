from __future__ import print_function
import logging
import os
import os.path as op
from os import getenv
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
__version__ = '1.20.0a9'
#Logging
GLOBAL_LOGGING = False #set to True in top elixer calls so we do not normally log from package imports
LOG_TO_STDOUT = False #only kicks in if GLOBAL_LOGGING is False
if not GLOBAL_LOGGING and LOG_TO_STDOUT:
    import traceback

#python version
import sys
PYTHON_MAJOR_VERSION = sys.version_info[0]
PYTHON_VERSION = sys.version_info
if sys.byteorder == 'big':
    BIG_ENDIAN = True
else:
    BIG_ENDIAN = False

ELIXER_CODE_PATH = op.dirname(op.realpath(__file__))

LAUNCH_PDF_VIEWER = None

valid_HDR_Versions = [1,2,2.1,3,3.0,4,4.0]

HDR_Version = "3" #"2.1"
HDR_Version_float = 3.0

HDR_Latest_Str = "hdr4" #latest available, not necessarily the default or what is being used
HDR_Latest_Float = 4.0

WORK_BASEPATH = "/work"
try:
    WORK_BASEPATH = getenv("WORK_BASEPATH")
    if WORK_BASEPATH is None or len(WORK_BASEPATH) == 0:
        WORK_BASEPATH = "/work"
except:
    pass


ELIXER_SPECIAL = 0 #integer, triggers special behavior in code
#HDR_DATA_BASEPATH = "/data/03946/hetdex" #defunct 2020-10-01 #TACC wrangler:/data removed
HDR_WORK_BASEPATH = op.join(WORK_BASEPATH,"03946/hetdex/")
HDR_SCRATCH_BASEPATH = "/scratch/03946/hetdex/"
#HDR_DATA_BASEPATH = HDR_SCRATCH_BASEPATH
HDR_BASEPATH = HDR_WORK_BASEPATH

HDF5_DETECT_FN = None
HDF5_CONTINUUM_FN = None
CONTINUUM_RULES = False #use continuum rules instead of emission line rules
CONTNIUUM_RULES_THRESH = 8.5e-17 #about g=22, #5e-17 #about g=20
HDF5_BROAD_DETECT_FN = None
HDF5_SURVEY_FN = None
OBSERVATIONS_BASEDIR = None
BAD_AMP_LIST = None
BAD_AMP_TABLE = None
CONFIG_BASEDIR = None
PANACEA_RED_BASEDIR = None
PANACEA_RED_BASEDIR_DEFAULT = None
PANACEA_HDF5_BASEDIR = None
PIXFLT_LOC = None
FORCE_MCMC = False
FORCE_MCMC_MIN_SNR = 4.0
LIMIT_GAUSS_FIT_SIGMA_MIN = None #configurable on commandline with --fit_sigma
LIMIT_GAUSS_FIT_SIGMA_MAX = None

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
COSMOS_LAIGLE_BASE_PATH = None
COSMOS_LAIGLE_CAT_PATH = None

GAIA_DEX_BASE_PATH = None
DECAM_IMAGE_PATH = None
SHELA_BASE_PATH = None

SHELA_CAT_PATH = None
SHELA_PHOTO_Z_COMBINED_PATH = None
SHELA_PHOTO_Z_MASTER_PATH = None

HSC_BASE_PATH = None
HSC_CAT_PATH = None
HSC_IMAGE_PATH = None
#HSC_AUX_IMAGE_PATH = None #not used anymore

HSC_SSP_BASE_PATH = None
HSC_SSP_CAT_PATH = None
HSC_SSP_IMAGE_PATH = None
HSC_SSP_PHOTO_Z_PATH = None

KPNO_BASE_PATH = None
KPNO_CAT_PATH = None
KPNO_IMAGE_PATH = None

CFHTLS_BASE_PATH = None

SDSS_CAT_PATH = None

HETDEX_API_CONFIG = None

the_Survey = None #HETDEX API Survey object ... common to be used in elixer
the_DetectionsIndex = None #data release all detections index
the_DetectionsDict = None #dictionary of neighbors detections query objects (separate from HETDEX_API_Detections)
HETDEX_API_Detections = None #per detections query object; bound to a single HDR version and line vs continuum

LOCAL_DEV_HOSTNAMES = ["z50","dg5"]


BUILD_REPORT_BY_FILTER = True #if True, multiple catalogs are used to build the report, with the deepest survey by filter
                           #if False, then the single deepest catalog that overlaps is used with what ever filters it has

LOCAL_DEVBOX = False
if hostname in LOCAL_DEV_HOSTNAMES:  # primary author test box
#if False:
    HDR_Version = "3.0" #"2.1"
    HDR_Version_float = 3.0 #2.1
    LAUNCH_PDF_VIEWER = 'qpdfview'
    LOCAL_DEVBOX = True
else:
    HDR_Version = "3.0" #"2.1"  # default HDR Version if not specified
    HDR_Version_float = 3.0 #2.1
    LOCAL_DEVBOX = False

#look specifically (and only) for HDR version on call
args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

if "--hdr" in args: #overide default if specified on command line
    try:
        i = args.index("--hdr")
        if i != -1:
            HDR_Version = sys.argv[i + 1] #this could be an integer or a float or a string?
#            if HDR_Version.is_integer():
#                HDR_Version = int(HDR_Version)
    except:
        pass

    try:
        toks = HDR_Version.split(".")
        if len(toks) >= 2:
            HDR_Version = ".".join(toks[0:2])
            HDR_Version_float = float(".".join(toks[0:2]))
        elif len(toks) == 1:
            HDR_Version_float = int(toks[0])
        if not (HDR_Version_float in valid_HDR_Versions) and HDR_Version != 0:
            print(f"Invalid --hdr specified ({HDR_Version})")
            exit(-1)
    except:
        print(f"Invalid --hdr specified ({HDR_Version})")
        exit(-1)

def set_hdr_basepath(version=None):
    """
    Sets the globals to be used

    :param version: should be an integer 1 or 2 (as of 2020/02/01) ... higher numbers after
    :return:
    """
    global HDR_SCRATCH_BASEPATH, HDR_WORK_BASEPATH, HDR_BASEPATH, HDR_Version, HDR_Version_float, HETDEX_API_CONFIG


    if version is None:
        version = HDR_Version
        version_float = HDR_Version_float
    else:
        try:
            toks = version.split(".")
            if len(toks) >= 2:
                version_float = float(".".join(toks[0:2]))
            elif len(toks) == 1:
                version_float = int(toks[0])
        except:
            print(f"Invalid version specified ({version})")
            exit(-1)


    if HETDEX_API_CONFIG is None:
        if version_float != 0:
            strHDRVersion = f"hdr{version}"
        elif hostname in LOCAL_DEV_HOSTNAMES:
            strHDRVersion = f"hdr{HDR_Version}"
        else: #this is a problem
            print("Invalid HDRversion configuration")
            return

        try: #might be like "hdr3.0" where "hdr3" is what is expected
            HETDEX_API_CONFIG = HDRconfig(survey=strHDRVersion)
        except KeyError:
            try:
                if strHDRVersion[-2:] == ".0":
                    HETDEX_API_CONFIG = HDRconfig(survey=strHDRVersion[:-2])
                    strHDRVersion = strHDRVersion[:-2]
                    version = version[:-2]
                    HDR_Version = HDR_Version[:-2]
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)

    hdr_dir = ""
    # _DATA_, _SCRATCH_ _WORK_ all specific to ELiXer, but the BASEPATH should use HETEDEX_API defined if possible
    if version_float != 0:
        hdr_dir = f"hdr{version}" #hdr1, hdr2, ....
        #HDR_DATA_BASEPATH = op.join(HDR_DATA_BASEPATH,hdr_dir)
        HDR_SCRATCH_BASEPATH = op.join(HDR_SCRATCH_BASEPATH,hdr_dir)
        HDR_WORK_BASEPATH = op.join(HDR_WORK_BASEPATH,hdr_dir)
        HDR_BASEPATH = op.join(HDR_BASEPATH,hdr_dir)
    elif hostname in LOCAL_DEV_HOSTNAMES: #author test box
        hdr_dir = "hdr1"
        print(f"*** using {hdr_dir}")
        #HDR_DATA_BASEPATH = op.join(HDR_DATA_BASEPATH, hdr_dir)
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
    global HDR_Version_float
    global valid_HDR_Versions

    global HDF5_DETECT_FN
    global HDF5_BROAD_DETECT_FN
    global HDF5_CONTINUUM_FN
    global HDF5_SURVEY_FN
    global OBSERVATIONS_BASEDIR
    global BAD_AMP_LIST
    global BAD_AMP_TABLE
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
    global COSMOS_LAIGLE_BASE_PATH
    global COSMOS_LAIGLE_CAT_PATH
    global GAIA_DEX_BASE_PATH

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

    global HSC_SSP_BASE_PATH
    global HSC_SSP_CAT_PATH
    global HSC_SSP_IMAGE_PATH
    global HSC_SSP_PHOTO_Z_PATH

    global KPNO_BASE_PATH
    global KPNO_CAT_PATH
    global KPNO_IMAGE_PATH

    global CFHTLS_BASE_PATH

    global SDSS_CAT_PATH

    global LAUNCH_PDF_VIEWER #for debug machine only

    global HETDEX_API_CONFIG

    global USE_MASKED_CONTINUUM_FOR_BEST_EW


    try:
        toks = version.split(".")
        if len(toks) >= 2:
            version_float = float(".".join(toks[0:2]))
        elif len(toks) == 1:
            version_float = int(toks[0])
    except:
        print(f"Invalid version specified ({version})")
        exit(-1)

    #make sure we have a valid version to select
    if not (version_float in valid_HDR_Versions) and version_float != 0:
        print(f"Invalid HDR version specified ({version}).")
        return False

    HDR_Version = version
    HDR_Version_float = version_float
    set_hdr_basepath(version)

    BAD_AMP_LIST = op.join(WORK_BASEPATH,"03261/polonius/maverick/catalogs/bad_amp_list.txt") #not really used anymore

    normal_build = True
    # if (hostname == LOCAL_DEV_HOSTNAME) and (version == 0):  #author test box:
    #     if False: #for debugging
    #         normal_build = False
    #         HDF5_DETECT_FN = "/work/03946/hetdex/hdr1/detect/detect_hdr1.h5"
    #         HDF5_CONTINUUM_FN = "/work/03946/hetdex/hdr1/detect/continuum_sources.h5"
    #         HDF5_SURVEY_FN = "/work/03946/hetdex/hdr1/survey/survey_hdr1.h5"
    #
    #         OBSERVATIONS_BASEDIR = "/work/03946/hetdex/hdr1/reduction/"
    #         BAD_AMP_LIST = "/home/dustin/code/python/elixer/bad_amp_list.txt"
    #
    #         CONFIG_BASEDIR = "/work/03946/hetdex/hdr1/software/"
    #         PANACEA_RED_BASEDIR = "/work/03946/hetdex/hdr1/raw/red1/reductions/"
    #         PANACEA_RED_BASEDIR_DEFAULT = PANACEA_RED_BASEDIR
    #         PANACEA_HDF5_BASEDIR = "/work/03946/hetdex/hdr1/reduction/data"
    #
    #         CANDELS_EGS_Stefanon_2016_BASE_PATH = "/home/dustin/code/python/elixer/data/EGS"
    #         EGS_CFHTLS_PATH = "/home/dustin/code/python/elixer/data/CFHTLS"
    #         CFHTLS_PHOTOZ_CAT = "/home/dustin/code/python/elixer/data/CFHTLS/photozCFHTLS-W3_270912.out"
    #         GOODS_N_BASE_PATH = "/home/dustin/code/python/elixer/data/GOODSN/"
    #         GOODS_N_CAT_PATH = GOODS_N_BASE_PATH
    #
    #         EGS_GROTH_BASE_PATH = "/home/dustin/code/python/elixer/data/isak"
    #         EGS_GROTH_CAT_PATH = EGS_GROTH_BASE_PATH  # note: there is no catalog
    #
    #         STACK_COSMOS_BASE_PATH = "/home/dustin/code/python/elixer/data/isak"
    #         STACK_COSMOS_CAT_PATH = "/home/dustin/code/python/elixer/data/isak"
    #         COSMOS_EXTRA_PATH = "/home/dustin/code/python/elixer/data/"
    #
    #         SHELA_BASE_PATH = "/media/dustin/dd/hetdex/data/SHELA"  # "/home/dustin/code/python/elixer/data/isak/SHELA"
    #         DECAM_IMAGE_PATH = SHELA_BASE_PATH  # "/media/dustin/dd/hetdex/data/decam/images"
    #         SHELA_CAT_PATH = "/media/dustin/dd/hetdex/data/SHELA"  # "/home/dustin/code/python/elixer/data/isak/SHELA"
    #         SHELA_PHOTO_Z_COMBINED_PATH = "/home/dustin/code/python/elixer/data/isak/SHELA"
    #         SHELA_PHOTO_Z_MASTER_PATH = "/home/dustin/code/python/elixer/data/isak/SHELA"
    #
    #         # 2019-08-06 (mshiro base path inaccessible)
    #         # HSC_BASE_PATH = "/work/04094/mshiro/maverick/HSC/S15A/reduced"
    #         # HSC_CAT_PATH = "/media/dustin/dd/hetdex/data/HSC/catalog_tracts" #"/work/04094/mshiro/maverick/HSC/S15A/reduced/catalog_tracts"
    #         # HSC_IMAGE_PATH = "/work/04094/mshiro/maverick/HSC/S15A/reduced/images"
    #
    #         if op.exists("/work/03946/hetdex/hdr2/imaging/hsc"):
    #             HSC_BASE_PATH = "/work/03946/hetdex/hdr2/imaging/hsc"
    #             HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
    #             HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"
    #             #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
    #         else:
    #             HSC_BASE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced"
    #             HSC_CAT_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/catalog_tracts"
    #             HSC_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
    #             #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
    #
    #         DECALS_BASE_PATH = "/media/dustin/dd/hetdex/data/decals"
    #         DECALS_CAT_PATH = "/media/dustin/dd/hetdex/data/decals"
    #         DECALS_IMAGE_PATH = "/media/dustin/dd/hetdex/data/decals"
    #
    #         # KPNO_BASE_PATH = "/work/03261/polonius/hetdex/catalogs/KPNO_Mosaic"
    #         KPNO_BASE_PATH = "/work/03233/jf5007/maverick/KMImaging/"
    #         KPNO_CAT_PATH = KPNO_BASE_PATH
    #         KPNO_IMAGE_PATH = KPNO_BASE_PATH

    if normal_build:
        if HETDEX_API_CONFIG:
            HDF5_DETECT_FN = HETDEX_API_CONFIG.detecth5
            try:
                HDF5_BROAD_DETECT_FN = HETDEX_API_CONFIG.detectbroadh5
            except:
                HDF5_BROAD_DETECT_FN = None

            HDF5_CONTINUUM_FN = HETDEX_API_CONFIG.contsourceh5
            HDF5_SURVEY_FN = HETDEX_API_CONFIG.surveyh5
            BAD_AMP_TABLE = HETDEX_API_CONFIG.badamp
            OBSERVATIONS_BASEDIR = HETDEX_API_CONFIG.red_dir
            CONFIG_BASEDIR = HETDEX_API_CONFIG.software_dir
            HDF5_RAW_DIR = HETDEX_API_CONFIG.raw_dir #local to this function only
            HDF5_REDUCTION_DIR = HETDEX_API_CONFIG.red_dir #local to this function only

            try:
                if HDR_Version_float == 1:
                    PIXFLT_LOC = op.join(CONFIG_BASEDIR, "virus_config/PixelFlats")
                #elif HDR_Version_float < 3:
                elif op.exists(HETDEX_API_CONFIG.pixflat_dir):
                    PIXFLT_LOC = HETDEX_API_CONFIG.pixflat_dir
                else:
                    common = op.commonpath([HDF5_CONTINUUM_FN,HDF5_SURVEY_FN,HDF5_REDUCTION_DIR])
                    PIXFLT_LOC = op.join(common,"lib_calib/lib_pflat")
                    if not op.exists(PIXFLT_LOC):
                        #one more try, go up one level
                        PIXFLT_LOC = op.join(common, "../lib_calib/lib_pflat")

                    if not op.exists(PIXFLT_LOC):
                        print("Warning! Cannot find pixel flats. Path(s) do not exist.")
                        log.warning("Warning! Cannot find pixel flats. Path(s) do not exist.")
            except:
                common = op.commonpath([HDF5_CONTINUUM_FN,HDF5_SURVEY_FN,HDF5_REDUCTION_DIR])
                PIXFLT_LOC = op.join(common,"/lib_calib/lib_pflat")

        else: #defunct
            HDF5_DETECT_FN = op.join(HDR_BASEPATH, "detect/detect_hdr1.h5")
            HDF5_BROAD_DETECT_FN = None
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

        if HETDEX_API_CONFIG:
            try:
                hdr_imaging_basepath = HETDEX_API_CONFIG.imaging_dir
            except:
                print("***** using /data/03261/polonius/hdr2 for imaging *****")
                hdr_imaging_basepath = "/data/03261/polonius/hdr2/"
        else:
            print("***** using /data/03261/polonius/hdr2 for imaging *****")
            hdr_imaging_basepath = "/data/03261/polonius/hdr2/"

        remote_imaging_basepath = hdr_imaging_basepath

        try:
            if (hostname in LOCAL_DEV_HOSTNAMES):
                from glob import glob

                drives = glob("/media/dustin/*")
                if len(drives)==1:
                    #there is exactly one drive attached
                    usb_path = op.join(drives[0],"hetdex/hdr2/imaging/")
                else:
                    #assume the hardcoded drive
                    # usb_drive = "Seagate8TB"
                    usb_drive = "easystore"
                    usb_path = op.join(usb_drive, "hetdex/hdr2/imaging/")

                if op.exists(usb_path):
                    print(f"***** using {usb_path} for base imaging *****")
                    hdr_imaging_basepath = usb_path
        except:
            pass #do nothing

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

        COSMOS_LAIGLE_BASE_PATH = op.join(hdr_imaging_basepath, "cosmos/laigle2015")
        COSMOS_LAIGLE_CAT_PATH = op.join(hdr_imaging_basepath, "cosmos/laigle2015")

        try:
            if HETDEX_API_CONFIG:
                GAIA_DEX_BASE_PATH = HETDEX_API_CONFIG.gaiacat
            else:
                if op.exists("/work/03946/hetdex/gaia_hetdex_value_added_catalog/HDR2.1_Gaia_final_table.fits"):
                    GAIA_DEX_BASE_PATH  = "/work/03946/hetdex/gaia_hetdex_value_added_catalog/HDR2.1_Gaia_final_table.fits"
                else:
                    print("WARNING! Cannot find GAIA HETDEX value added catalog.")
        except:
            print("WARNING! Cannot find GAIA HETDEX value added catalog.")

        DECAM_IMAGE_PATH = op.join(hdr_imaging_basepath, "shela/nano/")
        SHELA_BASE_PATH = op.join(hdr_imaging_basepath, "shela/nano/")
        SHELA_CAT_PATH = SHELA_BASE_PATH
        SHELA_PHOTO_Z_COMBINED_PATH = op.join(hdr_imaging_basepath, "shela/SHELA")
        SHELA_PHOTO_Z_MASTER_PATH = op.join(hdr_imaging_basepath, "shela/SHELA")

        HSC_S15A = False
        if HDR_Version_float < 2:
            if op.exists(op.join(hdr_imaging_basepath,"hsc")):
                if HDR_Version_float == 1:
                    if op.exists(op.join(WORK_BASEPATH,"03946/hetdex/hdr2/imaging/hsc")):
                        HSC_BASE_PATH = op.join(WORK_BASEPATH,"03946/hetdex/hdr2/imaging/hsc")
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
            elif op.exists(op.join(WORK_BASEPATH,"03946/hetdex/hdr2/imaging/hsc")):
                HSC_BASE_PATH = op.join(WORK_BASEPATH,"03946/hetdex/hdr2/imaging/hsc")
                HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
                HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"
                #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
            else:
                HSC_BASE_PATH = op.join(WORK_BASEPATH,"03946/hetdex/hdr1/imaging/hsc/S15A/reduced")
                HSC_CAT_PATH = op.join(WORK_BASEPATH,"03946/hetdex/hdr1/imaging/hsc/S15A/reduced/catalog_tracts")
                HSC_IMAGE_PATH = op.join(WORK_BASEPATH,"03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images")
                #HSC_AUX_IMAGE_PATH = "/work/03946/hetdex/hdr1/imaging/hsc/S15A/reduced/images"
        else:
            HSC_BASE_PATH = op.join(hdr_imaging_basepath,"hsc")
            HSC_CAT_PATH = HSC_BASE_PATH + "/cat_tract_patch"
            HSC_IMAGE_PATH = HSC_BASE_PATH + "/image_tract_patch"

            #these are always remote, regardless of the workstation
            HSC_SSP_BASE_PATH = op.join(remote_imaging_basepath,"hsc_ssp")
            HSC_SSP_CAT_PATH = HSC_SSP_BASE_PATH
            HSC_SSP_IMAGE_PATH = HSC_SSP_BASE_PATH #cosmos/g, cosmos/r, w01/g , w01/r, etc ....)
            HSC_SSP_PHOTO_Z_PATH = op.join(HSC_SSP_BASE_PATH,"photz")


        # KPNO_BASE_PATH = "/work/03261/polonius/hetdex/catalogs/KPNO_Mosaic"
        if op.exists(op.join(hdr_imaging_basepath, "KMImaging")):
            KPNO_BASE_PATH = op.join(hdr_imaging_basepath, "KMImaging")
        else:
            KPNO_BASE_PATH = op.join(WORK_BASEPATH,"03233/jf5007/maverick/KMImaging/")
        KPNO_CAT_PATH = KPNO_BASE_PATH
        KPNO_IMAGE_PATH = KPNO_BASE_PATH

        #always on TACC (not stored locally), imaging and catalog tiles are together in same directory
        CFHTLS_BASE_PATH = op.join(HETDEX_API_CONFIG.imaging_dir,"cfhtls")
        # print("!!!!! Temporary CFHTLS BASE PATH !!!!!!!")
        # CFHTLS_BASE_PATH = "/scratch/03261/polonius/hdr2.1.2/imaging/cfhtls"
        if op.exists(op.join(HETDEX_API_CONFIG.imaging_dir,"cfhtls/photozCFHTLS-W3_270912.out")):
            CFHTLS_PHOTOZ_CAT = op.join(HETDEX_API_CONFIG.imaging_dir,"cfhtls/photozCFHTLS-W3_270912.out")
        else:
            CFHTLS_PHOTOZ_CAT = op.join(hdr_imaging_basepath, "candles_egs/CFHTLS/photozCFHTLS-W3_270912.out")


        SDSS_CAT_PATH = op.join(hdr_imaging_basepath,"sdss/specObj-dr16-trim.fits")

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
    if HDR_Version_float != 1:
        print("***** temporary hard code pixel flat location *****")
        PIXFLT_LOC = "/data/00115/gebhardt/lib_calib/lib_pflat"
    else:
        PIXFLT_LOC = op.join(CONFIG_BASEDIR, "virus_config/PixelFlats")


REPORT_ELIXER_MCMC_FIT = False

RELATIVE_PATH_UNIVERSE_CONFIG = "line_classifier/universe.cfg"
RELATIVE_PATH_FLUX_LIM_FN = "line_classifier/Line_flux_limit_5_sigma_baseline.dat"

if hostname in LOCAL_DEV_HOSTNAMES:  # primary author test box
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
if "--merge" in args or "--merge_unique" in args:
    LOG_FILENAME = "elixer_merge.log"
else:
    LOG_FILENAME = "elixer.log"

#loggin intialization moved to elixer.py in parse_commandline as that is the first place we need to log ...
#   if --help, then the logger is not created
#logging.basicConfig(filename=LOG_FILENAME,level=LOG_LEVEL,filemode='w')
#.debug(), .info(), .warning(), .error(), .critical()


#first time we need to log anything

class Global_Logger:
    FIRST_LOG = True
    DO_LOG = True

    def __init__(self,id): #id is a string identifier
        try:
            if not GLOBAL_LOGGING:
                self.__class__.DO_LOG = False
                self.logger = logging.getLogger(None) #make a dummy logger
                self.logger.level = 999
                import traceback
                return

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

            try:
                if self.__class__.FIRST_LOG:
                    logging.basicConfig(filename=LOG_FILENAME, filemode='w+')
                    self.__class__.FIRST_LOG = False
                else:
                    logging.basicConfig(filename=LOG_FILENAME, filemode='a+')
            except: #could be a permissions issue if the file exists and was created by someoneelse
                try:
                    uname = os.getlogin()
                    logname = uname+"_"+LOG_FILENAME
                    if self.__class__.FIRST_LOG:
                        logging.basicConfig(filename=logname, filemode='w+')
                        self.__class__.FIRST_LOG = False
                    else:
                        logging.basicConfig(filename=logname, filemode='a+')

                    print(f"Using {logname} for ELiXer logging ...")
                except:
                    pass

        except:
            self.__class__.DO_LOG = False
            try:
                if 'jovyan' in os.getcwd():
                    print("ELiXer global logging disabled. Possible permissions issue. Not critical.")
                else:
                    print("Warning! ELiXer global logging failure.")
            except:
                pass


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
            if self.__class__.DO_LOG:
                self.logger.setLevel(level)
        except:
            print("Exception in logger (setlevel) ...")

    def debug(self,msg,exc_info=False):
        try:
            if self.__class__.DO_LOG:
                msg = self.add_time(msg)
                self.logger.debug(msg,exc_info=exc_info)
            elif LOG_TO_STDOUT:
                msg = self.add_time(msg)
                print(msg)
                if exc_info:
                    print(traceback.format_exc())
        except:
            print("Exception in logger (debug) ...")

    def info(self,msg,exc_info=False):
        try:
            if self.__class__.DO_LOG:
                msg = self.add_time(msg)
                self.logger.info(msg,exc_info=exc_info)
            elif LOG_TO_STDOUT:
                msg = self.add_time(msg)
                print(msg)
                if exc_info:
                    print(traceback.format_exc())
        except:
            print("Exception in logger (info) ...")

    def warning(self,msg,exc_info=False):
        try:
            if self.__class__.DO_LOG:
                msg = self.add_time(msg)
                self.logger.warning(msg,exc_info=exc_info)
            elif LOG_TO_STDOUT:
                msg = self.add_time(msg)
                print(msg)
                if exc_info:
                    print(traceback.format_exc())
        except:
            print("Exception in logger (warning) ...")

    def error(self,msg,exc_info=False):
        try:
            if self.__class__.DO_LOG:
                msg = self.add_time(msg)
                self.logger.error(msg,exc_info=exc_info)
            elif LOG_TO_STDOUT:
                msg = self.add_time(msg)
                print(msg)
                if exc_info:
                    print(traceback.format_exc())
        except:
            print("Exception in logger (error) ...")

    def critical(self, msg, exc_info=False):
        try:
            if self.__class__.DO_LOG:
                msg = self.add_time(msg)
                self.logger.critical(msg, exc_info=exc_info)
            elif LOG_TO_STDOUT:
                msg = self.add_time(msg)
                print(msg)
                if exc_info:
                    print(traceback.format_exc())
        except:
            print("Exception in logger (critical) ....")


def python2():
    if PYTHON_MAJOR_VERSION == 2:
        return True
    else:
        return False

def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

#convert vacuum (usually for wavelengths < 2000A) to air for consistency
# def vac_to_air(w_vac):
#     return w_vac / (1 + 2.73518e-4 + 131.418 / w_vac ** 2 + 2.76249e8 / w_vac ** 4)

FOV_RADIUS_DEGREE = 0.16 #HETDEX FOV (radius) in degrees (approximately)

AirVacuumThresh = 2000.0 #AA ... below 2000 values are in vacuum, above 2000 in air

LyA_rest = 1215.67 #vacuum 1216. #A 1215.668 and 1215.674
OII_rest = 3727.8 #3726.032 + 3728.815  or  3727.319 + 3729.221 (27 + 14 weights?) = 3727.96846


#all other lines (in air) (from http://astronomy.nmsu.edu/drewski/tableofemissionlines.html)
OIII_4959 = 4958.111
OIII_5007 = 5006.843
OVI_1035 = 1034.7625  #*** vacuum 1031.912 + 1037.613, equal weight

CIV_1549 = 1549.4115 #*** vacuum 1548.187 + 1550.772, 1000 + 900 Weight
CIII_1909 =1908.734 #*** vacuum 1908.734
CII_2326 = 2324.095 # 2323.500 + 2324.690, equal weight

MgII_2799 = 2798.6944 #triplet   2795.528 + 2797.998 + 2802.705, (13 + 10 + 12 weights)

Hbeta_4861 = 4861.333
Hgamma_4340 = 4340.471
Hdelta_4101 = 4101.742
Hepsilon_3970 = 3970.079
Hzeta_3889 = 3889.064 #aka H8
Heta_3835 = 3835.391 #aka H9

HeI_3888 = 3888.647 #not currently used

NV_1241 = 1240.7077 #*** vacuum 1238.821 + 1242.804 (1000 + 900 weights, but close enough to be equal)
SiII_1260 = 1263.40075 #*** vacuum 1260.422 +  1264.730 + 1265.002 (1000 + 2000 + 200 weights)
SiIV_1400 = 1397.7617 #*** vacuum 1393.755 + 1402.770 (15 + 12 weights)

HeII_1640 = 1640.420 #*** vacuum

NeIII_3869 = 3868.760
NeIII_3967 = 3967.470
NeV_3347 = 3345.821
NeVI_3427 = 3425.881

NaI_4980 = 4981.3894 # 4975.3 + 4978.5414 + 4982.8134  as 0 + 1 + 2 weights
NaI_5153 = 5153.4024 #
#NaI_5071 = 5071.2 # might be stronger than others

#absorption
CaII_K_3934 = 3933.6614
CaII_H_3968 = 3968.4673

APPLY_GALACTIC_DUST_CORRECTION = True #if true apply explicit MW dust de-reddening or de-extinction using hetdex_api

LOAD_SPEC_FROM_HETDEX_API = True #if true attempt to load through hetdex_api first and fall back on the h5 file if fail
#FLUX_CONVERSION = (1./60)*1e-17
HETDEX_FLUX_BASE_CGS = 1e-17
# 1.35e-18 ~ 24.0 mag in g-band
# 8.52e-19 ~ 24.5 mag in g-band,
# 5.38e-19 ~ 25.0 mag in g-band
COMPUTE_HETDEX_MAG_LIMIT = True #if true, use the IFU fibers to compute a limit for the detection (otherwise just use
                                #HETDEX_CONTINUUM_MAG_LIMIT
COMPUTE_HETDEX_MAG_LIMIT_FULL_IFU = False #if True, use the full IFU, if False use the amp(s) for the top 4 fibers
HETDEX_CONTINUUM_MAG_LIMIT = 25.0 #24.5 #generous, truth is closer to 24.few
HETDEX_CONTINUUM_FLUX_LIMIT =  5.38e-19 #flux-density based on 25 mag limit (really more like 24.5)

HETDEX_BLUE_SAFE_WAVE = 3600.0 #65; 3600 [idx 6] #use when summing over or fitting to spectrum as whole
HETDEX_RED_SAFE_WAVE = 5400.0 #index 965

SDSS_G_FILTER_BLUE = 3900.0
SDSS_G_FILTER_RED = 5400.0
DEX_G_EFF_LAM = 4726.1 #HSC-2017-g (4843.7) sdss-2010-g (4726.1), other reports SDSS as 4770 or 4640;
DEX_G_EFF_LAM_IDX = 628 #index into CALFIB_WAVEGIRD that is closest to effective lam, that index = 4726AA
#if both below are TRUE and both good, the mean is used; if only one that that one is used
USE_HETDEX_SPEC_GMAG = True #allow calculation of gmag from the HETDEX spectrum directly (spectrum::get_hetdex_gmag())
USE_SDSS_SPEC_GMAG = True #allow calcuation of gmag via SDSS filter (spectrum::get_sdss_gmag()

CONTINUUM_FLOOR_COUNTS = 6.5 #5 sigma * 6 counts / sqrt(40 angstroms/1.9 angs per pixel)

CONTINUUM_THRESHOLD_FOR_ABSORPTION_CHECK = 2.0e-17 # erg/s/cm2/AA (near gmag 21)

USE_MASKED_CONTINUUM_FOR_BEST_EW = False #if true use the emission/aborption masked spectrum, else use as a band-pass

Fiber_Radius = 0.75 #arcsec
IFU_Width = 47.26 #arcsec ... includes 2x fiber radius ... more narrow part fiber 1 - 19, else is 49.8
IFU_Height = 49.98 #arcsec
# Fiber_1_X = -22.88
# Fiber_1_Y = -24.24
# Fiber_19_X = 22.88
# Fiber_19_Y = -24.24
# Fiber_430_X = -22.88
# Fiber_430_Y = 24.24
PreferCosmicCleaned = True #use cosmic cleaned FITS data if available (note: assumes filename is 'c' + filename)

Figure_DPI = 300
FIGURE_SZ_X = 18 #18
#FIGURE_SZ_Y = 9 #12
GRID_SZ_X = 3 # equivalent figure_sz_x for a grid width (e.g. one column)
GRID_SZ_Y = 3 # equivalent figure_sz_y for a grid height (e.g. one row)

LyC = False #switch for Lyman Continuum specialized code
DeblendSpectra = False #if true (automatically true if --LyC), get spectra or mags with flat fnu and PSF deblend neighbors
                        # from the target spectra

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
PENALIZE_FOR_EMISSION_IN_SKYLINE = False #since HDR 2.1.x the skyline removal has been quite good and this is no longer needed

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

WAVEGRID_BLUE_LIMIT = 3470.
WAVEGRID_RED_LIMIT = 5540.
WAVEGRID_NUM_BINS = 1036
SEARCH_DELTA_WAVELENGTH = -1 #if 0 to 100 and combined with --search, is a constraint on searching the catalog
CALFIB_WAVEGRID = np.linspace(WAVEGRID_BLUE_LIMIT,WAVEGRID_RED_LIMIT,WAVEGRID_NUM_BINS) #np.arange(3470.,5542.,2.0) #3470 - 5540
CALFIB_WAVEGRID.flags.writeable = False
CALFIB_WAVEGRID_VAC = None #this will be populated when spectrum_utilities is imported

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
SPEC_MAX_OFFSET_SPREAD = 4.0 #2.75 #AA #maximum spread in (velocity) offset (but in AA) across all lines in a solution
SPEC_MAX_OFFSET_SPREAD_BROAD_THRESHOLD = 15.0 #kick in broad classification (there are effectively no OII with FWHM > 15)
MIN_MCMC_SNR = 0.0 #minium SNR from an MCMC fit to accept as a real line (if 0.0, do not MCMC additional lines)
MIN_ADDL_EMIS_LINES_FOR_CLASSIFY = 1

DISPLAY_ABSORPTION_LINES = False
MAX_SCORE_ABSORPTION_LINES = 0.0 #the most an absorption line can contribute to the score (set to 0 to turn off)

MAXIMUM_LINE_SCORE_CAP = 100.0 #emission and absorption lines are capped to this maximum

MULTILINE_GOOD_LINE_SCORE = 8.0
MULTILINE_USE_ERROR_SPECTRUM_AS_NOISE = False #if False, uses the whole amp to estimate noise, if possible
MULTILINE_MIN_GOOD_ABOVE_NOISE = 3.0 #below this is not consider a possibly good line
MULTILINE_SCORE_NORM_ABOVE_NOISE = 5.0 #get full 1x score at this level
MULTILINE_SCORE_ABOVE_NOISE_MAX_BONUS = 3.0 #maximum multiplier as max of (peak/noise/NORM, BONUS)
MULTILINE_MIN_SOLUTION_SCORE = 25.0  #remember, this does NOT include the main line's score (about p(noise) = 0.01)
MULTILINE_FULL_SOLUTION_SCORE = 50.0 #scores at or above get full credit for the weight
MULTILINE_MIN_WEAK_SOLUTION_CONFIDENCE = 0.6
MULTILINE_USE_CONSISTENCY_CHECKS = True #if True, apply consistency checks (line ratios for AGN, vs low-z, etc)

MULTILINE_WEIGHT_PROB_REAL = 0.4 #probabilty of real (0-.999) makes up 40% of score
MULTILINE_WEIGHT_SOLUTION_SCORE = 0.5 #related to probability of real, makes 50% of score
MULTILINE_WEIGHT_FRAC_SCORE = 0.1 #fraction of total score for all solutions makes up 10%

MULTILINE_MIN_SOLUTION_CONFIDENCE = 0.95 * MULTILINE_WEIGHT_PROB_REAL + MULTILINE_WEIGHT_SOLUTION_SCORE + 0.5 * MULTILINE_WEIGHT_FRAC_SCORE
                                  #about 0.93
MULTILINE_MIN_NEIGHBOR_SCORE_RATIO = 1.5 #if two top scores are possible, the best solution must be this multiplier higher in score

MULTILINE_MAX_PROB_NOISE_TO_PLOT = 0.2 #plot dashed line on spectrum if p(noise) < 0.1
MULTILINE_ALWAYS_SHOW_BEST_GUESS = False #if true, show the best guess even if it does not meet the miniumum requirements
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
NUDGE_MAG_APERTURE_CENTER = 0.0 #0.5  #allow the center of the mag aperture to drift to the 2D Gaussian centroid
                                 #up to this distance in x and y in arcsec (if 0.0 then no drift is allowed)

#We are finding that we could be off as much as 2.0", though about 80% are less than 1.0"  and 90+% less than 1.5"
#There is about 0.35" error in absolute astrometry and then the rest (which is most of the error) is in the finding of
# the flux peak. It is worst for low SNR
NUDGE_SEP_MAX_DIST_EARLY_DATA = 1.75 #1.5 #allow source extractor found objects to be matched to the HETDEX target up to this distances
                          #in arcsec (for early data, 2017 and fist part of 2018 when # of IFUs was low and astrometric
                          #solution was not great

NUDGE_SEP_MAX_DIST_LATER_DATA = 1.25 #1.0 #allow source extractor found objects to be matched to the HETDEX target up to this distances
                          #in arcsec

NUDGE_SEP_MAX_DIST = 1.25 # 1.0 allow source extractor found objects to be matched to the HETDEX target up to this distances
                          #in arcsec. NOTICE. this takes one of the above values (set in elixer.py) based on the observation
                          #date

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
FRAC_UNIQUE_PIXELS_MINIMUM = 0.70 #0.7 bare minumum unique pixels (no other condition included)
FRAC_UNIQUE_PIXELS_NOT_EMPTY = 0.75 #.75 less than --> empty (or bad) (combined with FRAC_DUPLICATE_PIXELS)
FRAC_DUPLICATE_PIXELS = 0.20 #if 0.25 of pixels are all the same (or in the same few) this may be bad
#this only counts up to the top 3 values, so if there are a lot of duplicate pixel values (but only in sets of a few)
#this does not trigger
FRAC_UNIQUE_PIXELS_AUTOCORRELATE = 0.75 #less than --> empty (with an autocorrelation)
FRAC_NONZERO_PIXELS = 0.66 #0.66 #less than --> empty

#note: Pan-STARRS is prioritized over SDSS (since Pan-STARRS is deeper 23.3 vs 22.0)
DECALS_WEB_ALLOW = True #if no other catalogs match, try DECaLS as online query (default if not dispatch mode)
DECALS_WEB_FORCE = False #ignore local catalogs and Force the use of only DECaLS

PANSTARRS_ALLOW = True #if no other catalogs match, try Pan-STARRS as online query (default if not dispatch mode)
PANSTARRS_FORCE = False  #ignore local catalogs and Force the use of only Pan-STARRS

SDSS_ALLOW = True #if no other catalogs match, try SDSS as online query (default if not dispatch mode)
SDSS_FORCE = False  #ignore local catalogs and Force the use of only SDSS
SDSS_SCORE_BOOST = MULTILINE_MIN_SOLUTION_SCORE #specficically for the SDSS z Catalog (other local catalogs are below)
CHECK_SDSS_Z_CATALOG  = True #set to True to check the SDSS z-catalog
# (similar in function to galaxy mask in that if a known z is close and it matches an emission line, associated that z)
CHECK_GAIA_DEX_CATALOG = False

#these are for the non-web catalogs we have, so it excludes SDSS (which is controlled separately just above)
BANDPASS_PREFER_G = True #if true use g band over r if both present, otherwise use r as the primary
CHECK_ALL_CATALOG_BID_Z = True
ALL_CATATLOG_SPEC_Z_BOOST = MULTILINE_FULL_SOLUTION_SCORE * 2.0 #i.e. +100.0 #addititive to the base solution score
ALL_CATATLOG_PHOT_Z_BOOST = 5.0        #ie. +5; some are more reliable than others but this is a broad brush

USE_PHOTO_CATS = True  #default normal is True .... use photometry catalogs (if False only generate the top (HETDEX) part)

MAX_NEIGHBORS_IN_MAP = 15

PROJECT_LINE_IMAGE_TO_COMMON_WCS = True #if True, the lineflux image should be rotated (projected/transformed) to
                                        #the master cutout's WCS (so North direction matches)

BUILD_HDF5_CATALOG = True

ALLOW_SYSTEM_CALL_PDF_CONVERSION = True #if True, if the Python PDF to PNG fails, attempt a system call to pdftoppm

DISPLAY_PSEUDO_COLOR = False  #display in upper left the pseudo-color from the HETDEX spectrum

COMBINE_PLAE = True # combine (all?) PLAE/POII ratio data into a single estimate
AGGREGATE_PLAE_CLASSIFICATION = True # and combine with other data to make a single p(LAE) estimate (0.0 - 1.0)
ZEROTH_ROW_HEADER = True # and put this in the zeroth row header

#penalties for solutions that fail to account for all lines
MAX_OK_UNMATCHED_LINES = 0
MAX_OK_UNMATCHED_LINES_SCORE = MULTILINE_MIN_SOLUTION_SCORE #25.0

MARK_PIXEL_FLAT_DEVIATION = 3.0 #if > 3.0 sigma from mean, then mark as bad well (set to large value to not mark)
MIN_PIXEL_FLAT_CENTER_RATIO = 0.85 #if less than this, the center is bad and may create a false emission line
MAX_NUM_DUPLICATE_CENTRAL_PIXELS = 2 #if more than this, flag as spurious
PIXEL_FLAT_ABSOLUTE_BAD_VALUE = 0.7 #values at or below this in the flat are "bad" and can artificially create emission
                                    #a -1 turns it off (code will ignore this global)
#note: 2 means there is one set of duplicates: ie. [1,2,1,3] would be 2 (1 and 1 are duplicated)

MAX_MAG_FAINT = 28.0 #set as nominal "faint" mag if flux limit reached (if not set by specific catalog ... which, unless
                     # this is an HST catalog, this is pretty good (HST is 29-30)

PLAE_POII_GAUSSIAN_WEIGHT_SIGMA = 5.0 #10.0 s|t by sigma or 1/sigma you get to 80% weight

CHECK_FOR_METEOR = True #if true, check the exposure fiber data for meteor pattern
CHECK_GALAXY_MASK = True #if true, check for detection inclusion in galaxy mask
GALAXY_MASK_D25_SCALE = 3.0 #check withing this x D25 curve to be inside galaxy ellipse
GALAXY_MASK_D25_SCORE_NORM = 2.0 #scoring scale normalization (xxx_D25_scale/score_norm) see hetdex.py DetObj::check_transients_and_flags
GALAXY_MASK_SCORE_BOOST = 100.0 # boost to the solution score if line found to match in a galaxy, mutliplied by
                                # scaled emission line rank and D25 distance, see hetdex.py DetObj::check_transients_and_flags

CLUSTER_POS_SEARCH = 15.0 #not really radius but +/- arcsecs from center position
CLUSTER_WAVE_SEARCH = 2.0 #in AA from line center
CLUSTER_MAG_THRESH = 23.0 #must be brighter than this to be a cluster parent
CLUSTER_SELF_MAG_THRESH = 21.0 #if brighter than this don't bother (unless there are flags)
CLUSTER_SCORE_BOOST = 100.0

ALLOW_BROADLINE_FIT = True
BROADLINE_GMAG_MAX = 23.0 #must be less than (brighter) than this value to auto-trip broad conditions


LINE_FINDER_MEDIAN_SCAN = 13 #SNR scan after applying a median filter to the flux; the value is the #of pixels, must be odd
                            #set to 0 to turn off
                            #13+ seems pretty good at picking up broad AGN
LINE_FINDER_FULL_FIT_SCAN = 0 # -1 = hard no, never run;  0 = soft no, can still run for other conditions, 1 = hard yes
                            #scan at each pixel (each bin in CALFIB_WAVEGRID) and try to fit emission and/or abosrption
                            #NOTE: this will still run if False when no lines are found with other methods
                            #if True (always on) increases line finder run-time by ~ 50%
                            #also seems to slightly encourage weak lines (that are false positives)

LIMIT_GRIDSEARCH_LINE_FINDER = True # turn off the median and full_fit scans (see above two value) on --gridsearch

SUBTRACT_HETDEX_SKY_RESIDUAL = False #if true compute a per-shot sky residual, convolve with per-shot PSF and subtract
# from the HETDEX spectrum (only applies to re-extractions (forced extractions) with ffsky
# requires --aperture xx  --ffsky --sky_residual

GET_SPECTRA_MULTIPROCESS = True #auto sets to False if in SLURM/dispatch mode

R_BAND_UV_BETA_SLOPE = -2.5 #UV slope (beta) for star forming galaxies used to adjust r-band to g-band near the
                            #supposed LyA line;  to turn off the correction, set to flat -2.0
                            #really varies with galaxy SFR, history, z, dust, etc
                            #but seems to run -2.4 to -2.7 or so for star forming galaxies around cosmic noon

LAE_G_MAG_ZERO = 24.2 #@ 4500 somewhat empirical:   looks to swing from about (23.0 @3500) 23.2 @ 3727 to 25.4 @ 5500
LAE_R_MAG_ZERO = 24.0 #somewhat empirical: . .. also using -0.2 or -0.3 mag from g-band per Leung+2017
LAE_MAG_SLOPE = 1.2 # per 1000AA in wave, centered at 4500AA
LAE_MAG_SIGMA = 0.5 #building a Gaussian as probablilty that mag > LAE_X_MAG_ZERO is an LAE
LAE_EW_MAG_TRIGGER_MAX = 25.0 #if the associated EW_rest(LyA) is less than this value, then look at the magnitudes
LAE_EW_MAG_TRIGGER_MIN = 15.0 #if the associated EW_rest(LyA) is greater than this value, then look at the magnitudes

LINEWIDTH_SIGMA_TRANSITION = 4.5  #larger than this, is increasingly more likely to be LyA, below .. could be either
LINEWIDTH_SIGMA_MAX_OII = 6.5 #there just are not any larger than this (FWHM > 16.5)

SEP_FIXED_APERTURE_RADIUS = 1.0 #RADIUS in arcsec ... used at the barycenter position of SEP objects
SEP_FIXED_APERTURE_PSF = False #if true apply the HETDEX seeing PSF
SHOT_SEEING = None #temporay usage for HSC-g comparison

FWHM_TYPE1_AGN_VELOCITY_THRESHOLD = 1500.0 #km/s #FWHM velocity in emission line above this value might be a type 1 AGN


PLYA_VOTE_THRESH = 0.5 # >= vote for LyA, below for not LyA
PLYA_VOTE_THRESH_1 = PLYA_VOTE_THRESH #just for naming consistency
PLYA_VOTE_THRESH_2 = 0.4 #0.6 #limit to just two additional thresholds to check
PLYA_VOTE_THRESH_3 = 0.3 #0.4
#np.unique returns the list sorted and we want to maintain the order
_, idx = np.unique([PLYA_VOTE_THRESH_1,PLYA_VOTE_THRESH_2,PLYA_VOTE_THRESH_3],return_index=True)
idx = sorted(idx)
PLYA_VOTE_THRESH_LIST = np.array([PLYA_VOTE_THRESH_1,PLYA_VOTE_THRESH_2,PLYA_VOTE_THRESH_3])[idx]
#PLYA_VOTE_LO = PLYA_VOTE_THRESH - PLYA_VOTE_THRESH * 0.2 # lower bound for a somewhat "uncertain" region
#PLYA_VOTE_HI = PLYA_VOTE_THRESH + (1 - PLYA_VOTE_THRESH) * 0.2 # upper bound for a somewhat "uncertain" region
PLYA_VOTE_LO = lambda thresh : thresh - thresh * 0.2 # lower bound for a somewhat "uncertain" region
PLYA_VOTE_HI =  lambda thresh : thresh + (1 - thresh) * 0.2 # upper bound for a somewhat "uncertain" region

##################################
#Detection Flags (DF) (32 bit)
##################################

DETFLAG_FOLLOWUP_NEEDED             = 0x00000001  #unspecified reason, catch-all, but human visual inspection recommended
DETFLAG_IMAGING_MAG_INCONSISTENT    = 0x00000002  #large differences in bandpass mags of overlapping imaging (of adequate depth)
DETFLAG_DEX_GMAG_INCONSISTENT       = 0x00000004  #the g-mag from the DEX spectrum is very different from g or r band aperture mag
                                            #where the DEX g-mag is 24.5 or brighter and the imaging is at least as deep

DETFLAG_UNCERTAIN_CLASSIFICATION    = 0x00000008  #contradictory information in classification
                                                  #usually echoed in P(LyA) near 0.5 or Q(z) < 0.5, OR
                                                  #there is a weak to moderate line solution that needs visual inspetion
                                                  #typically this is an OIII-5007 or MgII solution that can be confused
                                                  #with LyA
DETFLAG_BLENDED_SPECTRA             = 0x00000010
                                        #due to extra emission lines, there maybe two or more different objects in the spectrum
                                        #or two or more objects in the central 1.5"radius  region

DETFLAG_COUNTERPART_NOT_FOUND       = 0x00000020
                                        #there is continuum or bright emission in the HETDEX spectrum, but nothing shows
                                        # in imaging; this is partly redundant with DETFLAG_DEX_GMAG_INCONSISTENT
DETFLAG_DISTANT_COUNTERPART         = 0x00000040
                                        #there are SEP ellipses in imaging BUT the nearest SEP ellipse is far away (+0.5")
                                        #may need inspection to see if associated with large object OR is a faint
                                        #detection or even lensed

DETFLAG_COUNTERPART_MAG_MISMATCH    = 0x00000080 #r,g magnitude of catalog counterpart varies significantly from the
                                        #aperture magnitude AND is fainter than 22

DETFLAG_NO_IMAGING                  = 0x00000100 #no overlapping imaging at all
DETFLAG_POOR_IMAGING                = 0x00000200 #poor depth (in g,r) ... like just SDSS or PanSTARRS (worse than 24.5)
                                                 #AND the object is not bright (fainter than 23)

DETFLAG_LARGE_SKY_SUB               = 0x00000400 #possibly excessive sky subtraction in g or r band
                                        #can impact the magnitude calculation (so partly redundant with others)
                                        #NOTE: this is a reserved flag, but there is no code to check it at this time

DETFLAG_EXT_CAT_QUESTIONABLE_Z      = 0x00000800 #best redshift reported is from an external catalog and might be questionable
                                                 #the redshift my by uncertain or it is unclear that it belongs to our object

DETFLAG_Z_FROM_NEIGHBOR             = 0x00001000 #the original redshift was replaced by that of a neighbor
                                                 #as a better redshift

DETFLAG_DEXSPEC_GMAG_INCONSISTENT   = 0x00002000 #the straight gmag from the DEX spectrum and from SDSS filter do not agree

DETFLAG_LARGE_NEIGHBOR              = 0x00004000 #imaging and SEP show/suggest a large, bright neighbor that could be
                                                 #messing up the classification and continuum measure
DETFLAG_POSSIBLE_LOCAL_TRANSIENT    = 0x00008000 #meteor or satellite ... a single bright dither, etc

DETFLAG_BAD_PIXEL_FLAT              = 0x00010000
DETFLAG_DUPLICATE_FIBERS            = 0x00020000
DETFLAG_NEGATIVE_SPECTRUM           = 0x00040000
DETFLAG_POOR_THROUGHPUT             = 0x00080000
DETFLAG_BAD_DITHER_NORM             = 0x00100000
DETFLAG_POOR_SHOT                   = 0x00200000
DETFLAG_QUESTIONABLE_DETECTION      = 0x00400000   #unable to fit a continuum (wide) and cont(n) is fairly negative, or bad emission line fit
DETFLAG_EXCESSIVE_ZERO_PIXELS       = 0x00800000   #too many zero valued pixels at the emission line center in 2D cutouts

DETFLAG_POSSIBLE_PN                 = 0x01000000    #possible planetery nebula hit (usually 5007, without an obvious source)
DETFLAG_NO_DUST_CORRECTION          = 0x02000000    #dust correction was requested but failed (see APPLY_GALACTIC_DUST_CORRECTION)
DETFLAG_BAD_PIXELS                  = 0x04000000    #hot column, maybe bad sky subtraction, etc ... possible false detection
DETFLAG_BAD_EMISSION_LINE           = 0x08000000    #emission line is questionable, could be continuum between absorbers

DETFLAG_NO_ZEROPOINT                = 0x10000000    #may be temporary ... no zeropoint correction could be made


DETFLAG_CORRUPT_DATA                = 0x80000000    #some nontrivial portion of the data may be corrupt, though it may not be significant
#todo: low SNR, weighted position is between fibers (i.e. distances from blue fiber center > 0.74 or 0.75 and SNR < 5.2 or so)


#todo:

#todo: possible flags?
# out of bounds: Seeing FWHM, throughput, or other shot issue?
# out of bounds fiber profile/chi2,etc?
# non-image or empty image or detected corrupt image?
# unusually high sky correction in image
#
#
# this is really quite simple, does it need to be a function? Just sum the flags you want to test against
# and check if flags read in & test flags is non-zero (for any to match)
# def combined_flag_value(flag_list):
#     """
#     given a list of integers (can use the DETFLAG_XXX values), return a single 32-bit integer as the bitwise sum to be
#     used to (sub)select matches (exact or subset) from a list or array of detection flags
#
#     :param flag_list:
#     :return:
#     """
#     out = None
#


##################################
#Voter Flags (32 bit)
# Turn ON each voter
##################################
VoteFeaturesTable = False #if true, include the P(LyA) extra voting features table in the h5 output

VOTER_ACTIVE = 0xFFFFFFFF

#each toggles a single P(LyA) voter
VOTER_ANGULAR_SIZE              = 0x00000001
VOTER_BRIGHT_CONTINUUM          = 0x00000002
VOTE_MULTILINE                  = 0x00000004
VOTE_UNMATCHED_LINES            = 0x00000008
VOTE_PLAE_POII                  = 0x00000010
VOTE_ASYMMETRIC_LINEFLUX        = 0x00000020
VOTE_STRAIGHT_LINE_SIGMA        = 0x00000040
VOTE_STRAIGHT_EW                = 0x00000080
VOTE_PHOTZ                      = 0x00000100
VOTE_DEX_GMAG                   = 0x00000200
VOTE_FLAM_SLOPE                 = 0x00000400
VOTE_EW_PLAE_POII_CORRECTION    = 0x00000800



###################################
# testing sky residuals
###################################

APPLY_SKY_RESIDUAL_TYPE = 0 #0 = No, off, do not use:   1 = per fiber, 2 = per 3.5" aperture, specifically
#Note: there are many issues in ELiXer and post-ELiXer for the aperture version, so it has been removed

#Defunct ... not longer supported as of 2023-06-15
#SKY_RESIDUAL_PER_SHOT = False #if True pull each residusl from the match shot, if False, use the universal model

#!!!! don't forget to produce the __default.fits as the bw of all the individual ones ...
#!!! currently in jupyter-notebook: ff_ll_sky_test.ipynb
#2023-06-13: still an option, but not recommended, replaced by xxx_ALL_xx_MODELS
#2023-06-15: made defunct and commented out code. Will comletely remove in later version.
# SKY_RESIDUAL_FITS_PATH = None #"/scratch/03261/polonius/random_apertures/all_fibers//2018_2022_asym_bw_strict//"
# SKY_RESIDUAL_FITS_PREFIX = None #"fiber_summary_asym_bw_"
# SKY_RESIDUAL_FITS_COL  = None #"ll_stack_050"

# Defunct, replaced by ALL_xx_MODELS
# SKY_RESIDUAL_USE_MODEL = True #if True, use the model of the stack, if False, use the stack directly
# SKY_RESIDUAL_HDR3_LO_FLUXD = None
# #single universal model (not recommended)
# SKY_RESIDUAL_HDR3_LO_FN = op.join(op.dirname(op.realpath(__file__)), "sky_subtraction_residuals/hdr3_local_sky_fiber_residual_model.txt")
# SKY_RESIDUAL_HDR3_FF_FLUXD = None
# #single universal model (not recommended)
# SKY_RESIDUAL_HDR3_FF_FN = op.join(op.dirname(op.realpath(__file__)), "sky_subtraction_residuals/hdr3_ff_sky_fiber_residual_model.txt")


#HDR3 and HDR4 use same correction
SKY_RESIDUAL_ALL_PSF = np.arange(1.2,3.1,0.1)
SKY_FIBER_RESIDUAL_ALL_LL_MODELS = None
SKY_FIBER_RESIDUAL_ALL_FF_MODELS = None
SKY_FIBER_RESIDUAL_HDR3_ALL_LL_MODELS_FN = op.join(op.dirname(op.realpath(__file__)),
                                             "sky_subtraction_residuals/hdr3_local_sky_fiber_residual_models_by_psf.txt")
SKY_FIBER_RESIDUAL_HDR3_ALL_FF_MODELS_FN = op.join(op.dirname(op.realpath(__file__)),
                                             "sky_subtraction_residuals/hdr3_ff_sky_fiber_residual_models_by_psf.txt")
SKY_APERTURE_RESIDUAL_ALL_LL_MODELS = None
SKY_APERTURE_RESIDUAL_ALL_FF_MODELS = None
SKY_APERTURE_RESIDUAL_HDR3_ALL_LL_MODELS_FN = op.join(op.dirname(op.realpath(__file__)),
                                             "sky_subtraction_residuals/hdr3_local_sky_aperture_residual_models_by_psf.txt")
SKY_APERTURE_RESIDUAL_HDR3_ALL_FF_MODELS_FN = op.join(op.dirname(op.realpath(__file__)),
                                             "sky_subtraction_residuals/hdr3_ff_sky_aperture_residual_models_by_psf.txt")

#temporary
#ZEROPOINT_SHIFT_LL = 0 #in e-17 erg/s/cm2/AA an extra flat shift applied to Local SKy subtraction models (additive)
#ZEROPOINT_SHIFT_FF = 0 #in e-17 erg/s/cm2/AA an extra flat shift applied to Local SKy subtraction models (additive)

#2023-06-29 ... just a guess at the moment ... need to calibrate
ZEROPOINT_BASE_LL = 0.2 #default mutiplicative zeropoint correction on flux assuming effective wave of 4726AA
ZEROPOINT_BASE_FF = 0.3
ZEROPOINT_FRAC    = 0.0 #fraction of the above correction that is applied (default is 0.0 = not applied at all)
                        #can be adjusted on command line as another fraction of this value

ZEROFLAT = False #if TRUE, the sky residual is shifted such that the average flux in the flat part (3900-5400AA) is zero
                   #the idea here is that the blue is artificailly enhanced and the flat is the true average background
ZEROFLAT_BLUE = 4000.0 #bluest wavelength to compute the "flat"
ZEROFLAT_RED = 5000.0



fz = False #temporary; if True prefer the compressed fits over the uncompressed fits

DEFAULT_BLUE_END_CORRECTION_MULT = 1./(((CALFIB_WAVEGRID/4726.)**2)*np.array(
    [8.55,8.50,8.44,8.38,8.32,8.26,8.21,8.15,8.10,8.04,7.98,7.93,7.87,7.82,7.77,7.71,7.66,7.61,7.55,7.50,
     7.45,7.40,7.35,7.30,7.25,7.20,7.15,7.10,7.05,7.00,6.95,6.90,6.85,6.81,6.76,6.71,6.66,6.62,6.57,6.53,
     6.48,6.44,6.39,6.35,6.30,6.26,6.21,6.17,6.13,6.09,6.04,6.00,5.96,5.92,5.88,5.83,5.79,5.75,5.71,5.67,
     5.63,5.59,5.56,5.52,5.48,5.44,5.40,5.36,5.33,5.29,5.25,5.22,5.18,5.14,5.11,5.07,5.04,5.00,4.97,4.93,
     4.90,4.86,4.83,4.80,4.76,4.73,4.70,4.66,4.63,4.60,4.57,4.54,4.50,4.47,4.44,4.41,4.38,4.35,4.32,4.29,
     4.26,4.23,4.20,4.17,4.14,4.11,4.09,4.06,4.03,4.00,3.98,3.95,3.92,3.89,3.87,3.84,3.81,3.79,3.76,3.74,
     3.71,3.69,3.66,3.64,3.61,3.59,3.56,3.54,3.51,3.49,3.47,3.44,3.42,3.40,3.37,3.35,3.33,3.31,3.29,3.26,
     3.24,3.22,3.20,3.18,3.16,3.14,3.11,3.09,3.07,3.05,3.03,3.01,2.99,2.97,2.96,2.94,2.92,2.90,2.88,2.86,
     2.84,2.82,2.81,2.79,2.77,2.75,2.73,2.72,2.70,2.68,2.67,2.65,2.63,2.62,2.60,2.58,2.57,2.55,2.54,2.52,
     2.50,2.49,2.47,2.46,2.44,2.43,2.41,2.40,2.38,2.37,2.36,2.34,2.33,2.31,2.30,2.29,2.27,2.26,2.25,2.23,
     2.22,2.21,2.19,2.18,2.17,2.16,2.15,2.13,2.12,2.11,2.10,2.09,2.07,2.06,2.05,2.04,2.03,2.02,2.01,2.00,
     1.98,1.97,1.96,1.95,1.94,1.93,1.92,1.91,1.90,1.89,1.88,1.87,1.86,1.85,1.84,1.83,1.83,1.82,1.81,1.80,
     1.79,1.78,1.77,1.76,1.75,1.75,1.74,1.73,1.72,1.71,1.70,1.70,1.69,1.68,1.67,1.67,1.66,1.65,1.64,1.64,
     1.63,1.62,1.62,1.61,1.60,1.59,1.59,1.58,1.57,1.57,1.56,1.55,1.55,1.54,1.54,1.53,1.52,1.52,1.51,1.51,
     1.50,1.49,1.49,1.48,1.48,1.47,1.47,1.46,1.46,1.45,1.44,1.44,1.43,1.43,1.42,1.42,1.41,1.41,1.41,1.40,
     1.40,1.39,1.39,1.38,1.38,1.37,1.37,1.37,1.36,1.36,1.35,1.35,1.34,1.34,1.34,1.33,1.33,1.32,1.32,1.32,
     1.31,1.31,1.31,1.30,1.30,1.30,1.29,1.29,1.29,1.28,1.28,1.28,1.27,1.27,1.27,1.26,1.26,1.26,1.25,1.25,
     1.25,1.25,1.24,1.24,1.24,1.23,1.23,1.23,1.23,1.22,1.22,1.22,1.22,1.21,1.21,1.21,1.21,1.20,1.20,1.20,
     1.20,1.20,1.19,1.19,1.19,1.19,1.19,1.18,1.18,1.18,1.18,1.18,1.17,1.17,1.17,1.17,1.17,1.16,1.16,1.16,
     1.16,1.16,1.16,1.15,1.15,1.15,1.15,1.15,1.15,1.14,1.14,1.14,1.14,1.14,1.14,1.14,1.13,1.13,1.13,1.13,
     1.13,1.13,1.13,1.13,1.12,1.12,1.12,1.12,1.12,1.12,1.12,1.12,1.11,1.11,1.11,1.11,1.11,1.11,1.11,1.11,
     1.11,1.11,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.10,1.09,1.09,1.09,1.09,1.09,1.09,1.09,
     1.09,1.09,1.09,1.09,1.09,1.09,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,
     1.08,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.07,1.06,1.06,
     1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.05,1.05,
     1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,
     1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,
     1.04,1.04,1.04,1.04,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,
     1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,
     1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.01,1.01,1.01,1.01,1.01,1.01,1.01,
     1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.00,1.00,1.00,
     1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,
     1.00,1.00,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,
     0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,
     0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,
     0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,
     0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,
     0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,
     0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.99,0.99,0.99,
     0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,1.00,1.00,1.00,1.00,
     1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,
     1.01,1.01,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.03,1.03,1.03,1.03,1.03,1.03,1.03,1.03,
     1.03,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.04,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.05,1.06,1.06,1.06,
     1.06,1.06,1.06,1.06,1.07,1.07,1.07,1.07,1.07,1.07,1.08,1.08,1.08,1.08,1.08,1.08,1.09,1.09,1.09,1.09,
     1.09,1.09,1.10,1.10,1.10,1.10,1.10,1.10,1.11,1.11,1.11,1.11,1.11,1.12,1.12,1.12,1.12,1.12,1.12,1.13,
     1.13,1.13,1.13,1.13,1.14,1.14,1.14,1.14,1.15,1.15,1.15,1.15,1.15,1.16,1.16,1.16,1.16,1.17,1.17,1.17,
     1.17,1.17,1.18,1.18,1.18,1.18,1.19,1.19,1.19,1.19,1.19,1.20,1.20,1.20,1.20,1.21,1.21,1.21,1.21,1.22,
     1.22,1.22,1.22,1.23,1.23,1.23,1.23,1.24,1.24,1.24,1.25,1.25,1.25,1.25,1.26,1.26,1.26,1.26,1.27,1.27,
     1.27,1.28,1.28,1.28,1.28,1.29,1.29,1.29,1.30,1.30,1.30,1.30,1.31,1.31,1.31,1.32,1.32,1.32,1.33,1.33,
     1.33,1.33,1.34,1.34,1.34,1.35,1.35,1.35,1.36,1.36,1.36,1.37,1.37,1.37,1.37,1.38,1.38,1.38,1.39,1.39,
     1.39,1.40,1.40,1.40,1.41,1.41,1.41,1.42,1.42,1.42,1.43,1.43,1.43,1.44,1.44,1.44,1.45,1.45,1.45,1.46,
     1.46,1.46,1.47,1.47,1.47,1.48,1.48,1.48,1.49,1.49,1.49,1.50,1.50,1.50,1.51,1.51]))