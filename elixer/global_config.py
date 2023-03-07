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
__version__ = '1.19.0a3'
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


LAUNCH_PDF_VIEWER = None

valid_HDR_Versions = [1,2,2.1,3,3.0]

HDR_Version = "3" #"2.1"
HDR_Version_float = 3.0

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
                    PIXFLT_LOC = op.join(common,"/lib_calib/lib_pflat")
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
            if (hostname in LOCAL_DEV_HOSTNAMES) and op.exists("/media/dustin/Seagate8TB/hetdex/hdr2/imaging/"):
                print("***** using /media/dustin/Seagate8TB/hetdex/hdr2/imaging/ for base imaging *****")
                hdr_imaging_basepath = "/media/dustin/Seagate8TB/hetdex/hdr2/imaging/"
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
VoteFeaturesTable = True #if true, include the P(LyA) extra voting features table
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

NUDGE_SEP_MAX_DIST_EARLY_DATA = 0.75 #1.5 #allow source extractor found objects to be matched to the HETDEX target up to this distances
                          #in arcsec (for early data, 2017 and fist part of 2018 when # of IFUs was low and astrometric
                          #solution was not great

NUDGE_SEP_MAX_DIST_LATER_DATA = 0.5 #1.0 #allow source extractor found objects to be matched to the HETDEX target up to this distances
                          #in arcsec

NUDGE_SEP_MAX_DIST = 0.5 # 1.0 allow source extractor found objects to be matched to the HETDEX target up to this distances
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
LINE_FINDER_FULL_FIT_SCAN = False #scan at each pixel (each bin in CALFIB_WAVEGRID) and try to fit emission and/or abosrption
                            #NOTE: this will still run if False when no lines are found with other methods
                            #if True (always on) increases line finder run-time by ~ 50%
                            #also seems to slightly encourage weak lines (that are false positives)

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

SEP_FIXED_APERTURE_RADIUS = 1.5 #RADIUS in arcsec ... used at the barycenter position of SEP objects

FWHM_TYPE1_AGN_VELOCITY_THRESHOLD = 1500.0 #km/s #FWHM velocity in emission line above this value might be a type 1 AGN


PLYA_VOTE_THRESH = 0.5 # >= vote for LyA, below for not LyA
PLYA_VOTE_THRESH_1 = PLYA_VOTE_THRESH #just for naming consistency
PLYA_VOTE_THRESH_2 = 0.6 #limit to just two additional thresholds to check
PLYA_VOTE_THRESH_3 = 0.4
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
# testin sky residuals
###################################
SKY_RESIDUAL_FITS_PATH = "/scratch/03261/polonius/random_apertures/all_fibers/2018_2022/"
SKY_RESIDUAL_FITS_PREFIX = "fiber_summary_md_"
SKY_RESIDUAL_FITS_COL  = "ll_stack_05"


#LL stack 05
SKY_RESIDUAL_DEFAULT = [
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0008273178829707814, 0.002480848175605206, 0.005292986365314362, 0.00652489770769692, 0.0085578558336117, 0.010033246749198869, 0.010084890116010355,
0.010405753326927659, 0.010617285688792109, 0.011043409708180308, 0.011008822280543938, 0.01045293114803368, 0.010333316255037786, 0.010397792529547818, 0.010795200424150034, 0.011106791250141157, 0.011401679468039508,
0.011076185398366465, 0.01091414307639401, 0.011152456858929351, 0.01235417670204218, 0.013386334719291853, 0.013535758887294708, 0.012050999732014456, 0.01127786139058771, 0.010647481888831203, 0.011348344026623559,
0.01207032550297424, 0.011436401524796563, 0.011169265559266138, 0.010425118279704586, 0.010921797179563703, 0.01095395500431076, 0.010465793424309705, 0.011025379345304561, 0.010688558249544063, 0.010970837810198234,
0.010982665128479336, 0.010323966890606013, 0.010724589669812212, 0.009777483667419078, 0.00990578427352844, 0.009988371332958618, 0.009886457668795286, 0.010060542142580809, 0.009865423287170825, 0.009635115863874123,
0.009597057971274279, 0.009804835478274576, 0.009882016633210658, 0.009904629864947063, 0.009908906848781346, 0.009550160972781123, 0.009666265663377874, 0.009761679182147754, 0.009716745591256326, 0.009681340817080488,
0.009655159635691782, 0.00954670086592269, 0.008456175690642213, 0.008088081296904038, 0.008876340786581135, 0.009334988541442069, 0.00959079170838837, 0.009460033205578902, 0.009353497381731609, 0.009015066063393778,
0.008533667035958134, 0.008586266916797036, 0.008781792803386701, 0.00913779715991611, 0.00932689077874978, 0.009116524967584689, 0.009311421968024237, 0.0091318942947185, 0.009007197280284745, 0.008417220977273012,
0.008443465704636603, 0.008441013473833832, 0.00835280910834925, 0.008306821531868224, 0.007786940135063632, 0.007990524654809254, 0.007944326453520287, 0.008035090589767428, 0.008250132931854962, 0.008104322903916873,
0.007612319848153697, 0.007339973150838834, 0.007487831909670873, 0.007544953802797815, 0.0075165647281568715, 0.007457588291625655, 0.007083975911238495, 0.007032425340135687, 0.007276642557384842, 0.007079867166853135,
0.0068513684031686626, 0.006870409635048406, 0.006927131191364883, 0.006112673692572469, 0.005639908239981609, 0.006225956148783467, 0.006854165132426191, 0.007392988800506682, 0.007402670978551893, 0.007210981045474152,
0.006993660059613037, 0.0067721589432159866, 0.006853125618523546, 0.007037872505991052, 0.007254235357768333, 0.007253871095447895, 0.006869403173122947, 0.006417178701133333, 0.006091032298080454, 0.006257319464776957,
0.006191428513624472, 0.006147906631254485, 0.006754450441155883, 0.006462486809496307, 0.00590168964372757, 0.006207964099683411, 0.007002127235922028, 0.007390891754480382, 0.007297291534132405, 0.0069291334875226695,
0.006411167689819759, 0.006181580502525164, 0.005818523649360914, 0.006509151520378431, 0.006547726517411849, 0.006536999508400731, 0.006391110901630605, 0.006038240514776493, 0.006180385616514497, 0.006091215799268502,
0.005394892378719045, 0.005370039885089385, 0.005718111300022011, 0.005786435863731012, 0.005949721871108333, 0.0059408816816809784, 0.005855795442906592, 0.005607492459217819, 0.0061540821031871844, 0.005729937704178868,
0.005211340925919391, 0.005298928600772593, 0.005534027163586166, 0.005912448372722545, 0.0057560076784332285, 0.00526423021341, 0.005235886625364377, 0.005321471801877938, 0.005452627513001662, 0.005277215765279247,
0.0052005698073099285, 0.004737384796803025, 0.004913682013119282, 0.005299982017334581, 0.005450553429207014, 0.00550445655991615, 0.005235022776657822, 0.005483617895502287, 0.005091115784057589, 0.004414703892576768,
0.00462570630078066, 0.005312998931030142, 0.0055059739294131774, 0.005553678362314838, 0.005576977929256861, 0.005235656278143454, 0.00508934028774996, 0.004905342945059073, 0.005043914487225841, 0.005164215992417002,
0.005062258415895523, 0.005064643722247541, 0.004854867617846202, 0.004928719802539531, 0.0051347120253256665, 0.004817050549305064, 0.004634166876470131, 0.004807145527658869, 0.0047014386395083044, 0.004852075476109807,
0.005073099717973012, 0.004804144815701174, 0.004497091897536576, 0.004673825775309133, 0.004735045574906431, 0.004742971672767351, 0.004710936286935938, 0.004653418268465457, 0.004792084711569545, 0.004641888911285961,
0.004615701801828826, 0.004514275616827676, 0.00447590802761, 0.00444700323847926, 0.00445692758243353, 0.0044451608707002715, 0.004492852994126289, 0.004188867916470794, 0.004119455655838432, 0.004178238645227895,
0.004387587325633404, 0.004311910494355007, 0.004482385025948536, 0.004429927411903035, 0.004517156682152224, 0.004614896055733098, 0.00440532232928396, 0.004386210951980139, 0.004434581072125223, 0.004515851680785075,
0.004487953297841136, 0.0045127409582387925, 0.004556077057162795, 0.004541390915534602, 0.004143678548790319, 0.003877548529318259, 0.003921389331725337, 0.004057922576468547, 0.00405012423323418, 0.004157908610339353,
0.004018186809851491, 0.004142394476651378, 0.00405008045971168, 0.004092956794158616, 0.004034183690140748, 0.0039173929095364796, 0.00411522603429239, 0.004046853122441699, 0.004276486074561432, 0.004405501853508688,
0.004347866105725375, 0.00410173822958632, 0.003770539221296429, 0.0036785553188718196, 0.0037066355941095578, 0.0036353894839205227, 0.0036838795976212105, 0.003837297305937925, 0.00398961218814859, 0.0036490192231511762,
0.003829879264547461, 0.003584706759739013, 0.003959979365507155, 0.003966007386705063, 0.003679032647761363, 0.0037259580522658415, 0.0038212927271463, 0.004012778591516475, 0.003866761718384295, 0.0035974724863403872,
0.0036130172703761113, 0.0034557199067690933, 0.0037627339997802524, 0.003973381945507799, 0.0038167079553886444, 0.0036333757248855677, 0.0035246415104067657, 0.003576638427299172, 0.003695801910458544, 0.003624647621472091,
0.0038142138938821743, 0.0038789596700291016, 0.003921723709404653, 0.0037475957723316967, 0.0038200136071073914, 0.0036789179518503003, 0.0036612658122221527, 0.00378783933101133, 0.0032178774894845957, 0.003424403378453031,
0.0033441538292793872, 0.003639314225043848, 0.003547873848697772, 0.0039222594686284095, 0.003625117387353904, 0.003720906265002735, 0.003576293146582585, 0.003232118848873859, 0.0033716130099208114, 0.003150186300175262,
0.0034276668498263974, 0.003711122506707028, 0.0034286546472229768, 0.0035366308080128847, 0.0034561473649291343, 0.0031989913228875093, 0.003584482980232375, 0.003391301471170495, 0.0034381150165920683, 0.003462124656484391,
0.003388228671408851, 0.0033432462138032813, 0.003336817582321192, 0.00336175469312656, 0.0032513729375057594, 0.0034087644851262034, 0.003502160713871476, 0.00359150814226523, 0.0033999016459009402, 0.003261035329984761,
0.003430289204124542, 0.003446948581339301, 0.0032717338679308706, 0.0033702010634839704, 0.0034306345680478294, 0.0034591950700684935, 0.003348386330687976, 0.0032741845790629823, 0.0031627121370267084, 0.0031548159951975104,
0.003194825681591467, 0.003234752667779216, 0.0031261013512347492, 0.002979503801752022, 0.002901798846591188, 0.0029529939700201372, 0.0033196455538157623, 0.00315170307463071, 0.0032017125076518748, 0.003173428401117598,
0.003281691102691327, 0.0034509734676750744, 0.0032778266161519075, 0.003224798242682535, 0.003362639878469453, 0.0034900652799626443, 0.003421778834257283, 0.003136574079025926, 0.0032004040669603873, 0.0029991480062211266,
0.003008025548449285, 0.0030165032049418003, 0.0032104048931878406, 0.003213353238113132, 0.003183193689505328, 0.0031607022599415622, 0.003168915927625873, 0.003116227991697917, 0.0033861110418914797, 0.0033126771877850575,
0.003281812258146921, 0.003065042906582451, 0.0029570980572538107, 0.003146884017048487, 0.0032311639710342847, 0.0033054361819769224, 0.0034929621445420537, 0.003458858511399731, 0.0032074782841211937, 0.003169096941049875,
0.0031644446004476913, 0.003207792509857125, 0.0032553005298142056, 0.003151708197340363, 0.003264419029575313, 0.00317740465775876, 0.0031857874468830326, 0.0031373193196641246, 0.0034892437926147177, 0.003414998064813454,
0.0033622691259148118, 0.0032212673249977135, 0.003153727842153998, 0.0032304634267276353, 0.0032477029117439467, 0.003300422045433265, 0.0034858235772308958, 0.0033883094795802856, 0.0030011121585099975, 0.003182586035093331,
0.0029982649776092367, 0.003092891865499174, 0.0031882578875025765, 0.0029285737721967014, 0.0030636984240072274, 0.003192723136872187, 0.003152774306114031, 0.00320693672246661, 0.002994328015965633, 0.0030522924199357705,
0.0032116333491181226, 0.003368841359996855, 0.003434404851217326, 0.003252249131566344, 0.0032632581442190375, 0.00323551758886901, 0.003219050320032237, 0.0031447432342679464, 0.003138495602315451, 0.003309119051757837,
0.0033597752019491065, 0.003229780384860278, 0.003138982652551628, 0.0029606841818449556, 0.0030185453729177575, 0.0031667034483064895, 0.003264226716744842, 0.0032950311674874917, 0.0032765064083640407, 0.0031891064549116415,
0.003358215020815052, 0.0032762038115757403, 0.0031496334561683386, 0.003038575073311213, 0.0029093072756544006, 0.0026852029445449677, 0.002923854131171605, 0.0029557972643070697, 0.003079352811586361, 0.003175242864926212,
0.003043597158640704, 0.003083129403853295, 0.003008039267441887, 0.003042694132532766, 0.0032336492615120506, 0.0032929970597601354, 0.0034016014594731385, 0.003433883533679469, 0.003392983803315018, 0.0033263854539803033,
0.0033770709475943288, 0.003476951945721724, 0.0037962577687983066, 0.00376870025308198, 0.003301279866295965, 0.0037456502729405587, 0.00302499781166087, 0.003342615707490519, 0.0034017184392086497, 0.0033939311747868514,
0.003499382851657543, 0.003539831890494393, 0.0035610005437310952, 0.003379433738164511, 0.003190079450967601, 0.0029906485174996805, 0.0031818966478167968, 0.0032800647088210098, 0.0032316138035808806, 0.003327669266001618,
0.003189371905833053, 0.003159763593060122, 0.0032301248274845882, 0.0031996515990125197, 0.003282854189212793, 0.0033103037492226077, 0.003335989824451395, 0.0033884439491688578, 0.0031964915122820858, 0.0031692894013307152,
0.003062581986363139, 0.002905543192198994, 0.0029891113681762196, 0.0028206759742510222, 0.0031694298881663734, 0.002770203163000578, 0.0030561917918025786, 0.0028760347052239166, 0.002894498830034466, 0.002957588047753094,
0.0029025185638655517, 0.00278954221693571, 0.003104453620591732, 0.0031292774598408154, 0.0030074014923589802, 0.0028367357463204754, 0.0028932751592877348, 0.0029547944174115547, 0.0028904532398100194, 0.0029059466638881753,
0.002632956974490025, 0.0027681829958205294, 0.002776855511364558, 0.0028882846467655054, 0.0029919924065579793, 0.0030184837550953324, 0.003089928573393335, 0.003062141519413676, 0.00296575317327182, 0.002875249448383006,
0.00300826517758095, 0.0031592067325521434, 0.0031059159522745755, 0.0029374656725824036, 0.0027959471268434242, 0.003041730924024836, 0.0032883359888090324, 0.0031928867818106677, 0.0029836898909970586, 0.0029775353284193394,
0.0032032536390427142, 0.003266352494930889, 0.0032634615580757614, 0.0030625439697662365, 0.002894172992679499, 0.0027595932128251666, 0.002648072835487816, 0.0025057908009351775, 0.0023396283124556953, 0.0021875943811198372,
0.00205243324274485, 0.0019823715856324125, 0.001875645510427415, 0.001811878987115834, 0.0017858779305936587, 0.0017014974353070584, 0.001601506376710857, 0.001511458159174923, 0.0016157679922220412, 0.0018166794967862557,
0.00202178138178213, 0.001986779249944011, 0.002098001474873341, 0.0021430232976478243, 0.0023436278046592607, 0.00281128328658325, 0.002839812671529316, 0.0030407757909745306, 0.002881551856750701, 0.0029178062951691567,
0.0030810822222216007, 0.0029762471307928444, 0.0030235986532766044, 0.003076500942426753, 0.0030152430230595785, 0.0030488890933929877, 0.0027722184600477767, 0.002907868438495613, 0.0030706701423546252, 0.003046864997209819,
0.002989884499270104, 0.0029986054985271233, 0.0030951945397992, 0.003266015798586627, 0.003137222282243543, 0.003009542812714605, 0.002870994369455211, 0.002848591917816304, 0.0029219181350899678, 0.0028671082633636522,
0.00292053011420565, 0.0028854998660411673, 0.002959506906063438, 0.0030950943031586168, 0.0028641401000553744, 0.0030624972250630047, 0.0029823196723385696, 0.0030780546893372403, 0.003188949084618566, 0.0030782990582909372,
0.002889999547581441, 0.0028465084912668794, 0.0029157490794281885, 0.00301254183817457, 0.00307305964357227, 0.0030619459654922976, 0.003020195301966897, 0.003015250163928782, 0.0029506341718210276, 0.002971771364319582,
0.0030487784363831183, 0.0029545287397226365, 0.002926961351468968, 0.002875974626970524, 0.003020798936573933, 0.0030809291515237415, 0.002918136901082112, 0.0028425659448284694, 0.002928696331719171, 0.0031985384880759,
0.003223272439741672, 0.0031435553322774782, 0.0032939258387772903, 0.003517042964273644, 0.0034309990611049437, 0.0030245588799224323, 0.002674862718516444, 0.002650031874156347, 0.002900038725045179, 0.0032268100232620946,
0.003314431709313086, 0.0033161405471986726, 0.0030496044625826025, 0.0029163187637333493, 0.002961081562604392, 0.003036205154595537, 0.0031344438199226105, 0.003013983397965027, 0.002908443728957997, 0.0029425947557508445,
0.0029698346206058455, 0.003104039372335075, 0.0030109059300186796, 0.00283957534654656, 0.0028710698937187197, 0.0029226251757013175, 0.0031470670092913506, 0.003140962866493982, 0.0031057484942584052, 0.003078914808490658,
0.0029144758408742202, 0.0028388598229381404, 0.002807052609778613, 0.002966259727031812, 0.0028175393329124237, 0.002840066950508851, 0.0028831113424878955, 0.0028764045715875693, 0.00303409539777907, 0.0029320330458908997,
0.0030982238321986587, 0.002982013809387778, 0.0030017055516898183, 0.0030172535115909893, 0.0027448637555405352, 0.0026364619918694807, 0.0023951502807582174, 0.0023099744761607387, 0.002398401544782402, 0.0025984783939492495,
0.0028394728942134234, 0.0028538585722361117, 0.0028459815481463656, 0.002929585659233668, 0.002902327427643123, 0.0028201775448027937, 0.0028043138634430884, 0.0028471129058261313, 0.002969271302780368, 0.0029087520694410316,
0.0029149720952647996, 0.0029083701135085723, 0.0029183801125609666, 0.0029726631803276282, 0.0027565195693801817, 0.002715672094876065, 0.0027717501319598197, 0.0028420215749913663, 0.0028990692883816417, 0.0029954600935256435,
0.003023080814631861, 0.002998346652720042, 0.002895850997737448, 0.0027990796877525115, 0.0028618405355940553, 0.0029900323570594273, 0.0030246528646760344, 0.002944757743174651, 0.002876583992155303, 0.0029187507418628345,
0.003044494657485208, 0.0032314543731309865, 0.0033527215562737704, 0.0032786070408465243, 0.0031532154882112843, 0.002787887533783915, 0.0027794208493665796, 0.002815993902853493, 0.002971005493957401, 0.0029244392349985883,
0.003167583297790671, 0.002982341332941541, 0.0031471413272278082, 0.0030315664902788642, 0.002835621415597355, 0.002693829311173944, 0.0026114656270921047, 0.0027341238639733965, 0.002375536902899832, 0.0025258595844894183,
0.002475074708330747, 0.002412115386307163, 0.0023361999012048345, 0.002394201725921315, 0.003036440067095006, 0.002791166936525087, 0.002816621946009349, 0.002676404035656019, 0.002966680724923154, 0.0027523915573895975,
0.0027135787427199866, 0.0027612631062181986, 0.003059291311937197, 0.0031664601073003213, 0.0030757704920588327, 0.0028980717482261706, 0.002627205212683324, 0.0026375203350991394, 0.002762099461934723, 0.0028595331523536016,
0.0028455340243946467, 0.002910709356369982, 0.0028158533994792536, 0.002713567574995233, 0.0027792742077941595, 0.0028411130177414105, 0.003076799688692785, 0.0028349655593352435, 0.0028028644425116267, 0.002697610981018889,
0.0029798393652364745, 0.0031180034826831095, 0.0028576268566151463, 0.002588790654438095, 0.0027639412115400526, 0.0031019257758707363, 0.003080616093379312, 0.002886649870882566, 0.002734708078402041, 0.002841363544651886,
0.0027369414883031727, 0.002873457854125923, 0.0029467463030988248, 0.0027070758155719574, 0.002831269206969601, 0.0027095221606163547, 0.0027054371935399967, 0.002828574777996824, 0.0029493139580974385, 0.003031426326909466,
0.0030430100316040745, 0.002534289810598051, 0.002461572446075927, 0.0027198480337779817, 0.0030085383237179954, 0.0031301872605859634, 0.0029569784210918253, 0.0029049116274346386, 0.002787037590296817, 0.0029199429553309676,
0.002837868982220595, 0.002795776417618354, 0.002699441061547984, 0.0027425513878501714, 0.0026550405645557247, 0.0025823823808755245, 0.00280328154026537, 0.003053509241349536, 0.0030571732063016353, 0.0029965217534742876,
0.0027749655772177782, 0.002741351755773952, 0.002723360457170831, 0.0027076829119857906, 0.0028396654180534407, 0.002871764099234385, 0.002868257638496142, 0.002857024327225358, 0.002891202174934424, 0.002756597580208359,
0.0027000741209084857, 0.0028626162699109453, 0.0028739745850344775, 0.0029522314476160033, 0.0028019651139404795, 0.0028300320780989866, 0.0028406391905388606, 0.0028753710909092908, 0.003047996353946956, 0.003040577853956011,
0.002877194627424101, 0.0026817605991696723, 0.002825602933699804, 0.0027211185115925845, 0.0028373152872371733, 0.003049779990448807, 0.0030854403858076836, 0.0028880250177480355, 0.002719741363991168, 0.0028423263968774372,
0.0030110551059929153, 0.002977068188422064, 0.0029254091676692937, 0.002839312996712398, 0.0028346593295057857, 0.002842382135781561, 0.002758207739470143, 0.002923873962080359, 0.0030756904956392183, 0.0028266227713981077,
0.0027789587968446903, 0.00294847152074249, 0.0029972798692991945, 0.003005552476367855, 0.00319889969429878, 0.0030210284154812586, 0.0029662861072068357, 0.00264457486685705, 0.002644235716076756, 0.0028368523317404016,
0.0029659049558455526, 0.0029840309992579836, 0.002813769814758347, 0.0028485661470700994, 0.0027739678630005512, 0.0027304080033374112, 0.00279295134146503, 0.0027609006839178505, 0.0028432231589533216, 0.002883500181499696,
0.0028378460584156814, 0.002785610771029617, 0.0027822374746580046, 0.002721606522545881, 0.0027273118675974653, 0.0026967981285283657, 0.0027846898672800433, 0.00287401108340126, 0.0029551308105430114, 0.0030226471585572236,
0.0029712894034000887, 0.002900469950722063, 0.0027117705140653307, 0.002698993905657979, 0.0028611286382040557, 0.002849461073328072, 0.0028534470754364564, 0.0027215348061054323, 0.002800103626770067, 0.002862402074080605,
0.00283544252472392, 0.0028908074613090964, 0.002904827572287142, 0.002874781176482523, 0.0028165101868249604, 0.0027086686804140116, 0.0026471043726534085, 0.002609675213894912, 0.002972086404451, 0.0028321344474476263,
0.0029783823398069303, 0.0031293290171256704, 0.003010764095666103, 0.0027496465833206116, 0.0024971091182430967, 0.002555680607559276, 0.003174052562596035, 0.0034832014828974052, 0.002894028338952457, 0.002424844553959804,
0.002542486108330878, 0.0029044903464774093, 0.003368250999870746, 0.0028954576753616535, 0.002536868160729706, 0.0026085039191826084, 0.0023976819599845958, 0.0029472563117602718, 0.0031265534899175755, 0.0031221257718901316,
0.0028146508377718165, 0.002747684433469431, 0.0030428864466803705, 0.0030968185982430734, 0.0028879948611730648, 0.003003722335101599, 0.0025412045281982413, 0.002486203905265919, 0.002912822527955743, 0.0032570517538694954,
0.0028510533727194956, 0.002724321766859261, 0.003200913476441883, 0.003131943792230629, 0.0021739992566946675, 0.002342464375090068, 0.0027604343174854958, 0.003119534377050513, 0.0029748025532365797, 0.002714759296724153,
0.0027965936119455116, 0.003126099288828305, 0.0026982732572046137, 0.0022264399873105217, 0.002269080819882596, 0.0026545800394304534, 0.0029601471521846627, 0.0030317091985761465, 0.0029974880764451766, 0.0030054174754396076,
0.00296240941988167, 0.0032251599251514997, 0.00272608594402407, 0.0025849369719776233, 0.0027302076508683446, 0.003129623333442299, 0.0034622507885466993, 0.003247736494716076, 0.0030197157303979167, 0.002928005441046783,
0.002959931540239773, 0.002867131873085569, 0.002896107053676881, 0.00308257011730811, 0.0028373977358688105, 0.002995944534433026, 0.003061478707628379, 0.0031787229919236866, 0.0030968605385858963, 0.0032556677779430523,
0.0032227607902993566, 0.002961159219759237, 0.0031735003336262413, 0.003303716213502456, 0.0033733710087053685, 0.0032842515888829196, 0.003242673223169925, 0.0030826291934664355, 0.003536087785180823, 0.0036183089756188823,
0.0032860181953164366, 0.0032559287140852527, 0.003266870056357659, 0.003250341622264412, 0.003303171884725617, 0.0033210605567274417, 0.0032147171163121684, 0.0032955860788394936, 0.003477650044967701, 0.003184361007015971,
0.0032069703237081, 0.003476608397582426, 0.003355352649678026, 0.003204612627200137, 0.003201877989272625, 0.0033451619765131283, 0.003468831526838005, 0.003447067617091097, 0.0032524172323654655, 0.003295756471447722,
0.0034242333483396727, 0.0034553178027338437, 0.0031997049266842647, 0.0032171924891027204, 0.0032229238937382283, 0.0033269095284624616, 0.003345130754648509, 0.0032742124870332443, 0.003223405230515256, 0.0033084398886141404,
0.003126955522563853, 0.0031158784790822143, 0.0032076762198816923, 0.003307206705769715, 0.0032730495298399684, 0.0031140833538767275, 0.00316320679845564, 0.003277227321672893, 0.0031952202020512055, 0.0032178104115357354,
0.0031833207661277636, 0.003208594296589514, 0.0031374632221494807, 0.0030570072518227515, 0.00308229353049906, 0.003072029421229906, 0.0032116295987830905, 0.0031380220132538176, 0.0030981394898110088, 0.0031867571039431047,
0.0031556168299859354, 0.0031627639762997805, 0.003155374509694013, 0.0031086532495126586, 0.002990856068105269, 0.0029449137202726193, 0.003044092117689558, 0.0031088406988657715, 0.0031303693429969473, 0.0029891173151657202,
0.002872437930657031, 0.0030395202237121083, 0.003266404207950486, 0.0035179377644558698, 0.0015693061723454109, 0.0012904823595198886, 0.003702096419676472, 0.004978382391187329, 0.004424739104975545, 0.003460314196870823,
0.0032793497832874067, 0.0032603861262022467, 0.0033602072061221975, 0.003371569600191449, 0.003163754237616246, 0.00260154880711092, 0.002377476694096208, 0.0028550234687729947, 0.0036043665538225828, 0.0034394553217204638,
0.002757690886591104, 0.0024645533149112416, 0.002360949028088311, 0.002103451463833315, 0.0018674273228746895, 0.0018439420230236807, 0.0016773592265485384, 0.0014087701799343263, 0.0010987584389071403, 0.000812979610341494,
0.0005207948796666418, 0.000240969591462874, 3.836540225712406e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0]