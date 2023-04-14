from __future__ import print_function

try:
    from elixer import global_config as G
    G.GLOBAL_LOGGING = True
except:
    import global_config as G
    G.GLOBAL_LOGGING = True

import astropy.extern.ply.yacc
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('default')

try:
    from elixer import hetdex
    from elixer import match_summary
    from elixer import catalogs
    from elixer import utilities as UTIL
    from elixer import science_image
    from elixer import elixer_hdf5
    from elixer import cat_base
    from elixer import spectrum_utilities as SU
    from elixer import clustering
    from elixer import spectrum as elixer_spectrum
except:

    import hetdex
    import match_summary
    import global_config as G
    import catalogs
    import utilities as UTIL
    import science_image
    import elixer_hdf5
    import cat_base
    import spectrum_utilities as SU
    import clustering
    import spectrum as elixer_spectrum

from hetdex_api import survey as hda_survey

plt.style.use('default') #restore to classic if hetdex api changes style
import argparse
import copy

from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as U
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import io
from distutils.version import LooseVersion
try:
    import pyhetdex.tools.files.file_tools as ft
except:
    #however, you will bomb out if you hit its use with old data
    print("Non-fatal warning. Cannot import pyhetdex.")
    pass #pyhetdex is not so important any more

import sys
import glob
import os
import fnmatch
import errno
import time
import numpy as np
#import re
from PIL import Image as PIL_Image
from PIL import ImageFile as PIL_ImageFile
FIG_DPI = 300

use_wand = False
OS_PNG_ONLY = True #can be overriden later in main()
if use_wand:
    from wand.image import Image
else:
    from pdf2image import convert_from_path

import tables

#try:
#    import PyPDF2 as PyPDF
#except ImportError:
#    PyPDF = None

try:
    import elixer.pdfrw as PyPDF
except:
    try:
        import pdfrw as PyPDF
    except ImportError:
        pdfrw = None


VERSION = sys.version.split()[0]
#import random

G_PDF_FILE_NUM = 0

#log = G.logging.getLogger('main_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('main_logger')
log.setlevel(G.LOG_LEVEL)

def get_input(prompt):
    if LooseVersion(VERSION) >= LooseVersion('3.0'):
        i = input(prompt)
    else:
        i = raw_input(prompt)
    return i


def parse_astrometry(file):
    ra = None
    dec = None
    par = None
    try:
        with open(file, 'r') as f:
            f = ft.skip_comments(f)
            for l in f:
                if len(l) > 10: #some reasonable minimum
                    toks = l.split()
                    #todo: some sanity checking??
                    ra = float(toks[0])
                    dec = float(toks[1])
                    par = float(toks[2])
    except:
        log.error("Cannot read astrometry file: %s" % file, exc_info=True)

    return ra,dec,par

def xlat_shotid(raw):
    try:
        if raw is None:
            return None

        if len(np.shape(raw)) == 1: #this is an array or list
            #shot = [float(str(r).lower().replace('v','')) for r in raw]
            shot = [xlat_shotid(s) for s in raw]
            return shot
        else: #single value
            shot = float(str(raw).lower().replace('v',''))
            if shot == 0:
                return None
            elif 3400. < shot < 5700.: #assume to be a HETDEX wavelength, not a shot
                return shot #as a float
            else:
                return int(shot)

    except:
        log.error(f"Exception translating shotid ({raw})",exc_info=True)

class PDF_File():
    def __init__(self,basename,id,pdf_name=None):
        self.basename = '%s' % basename
        self.filename = None
        self.id = int(id)
        self.bid_count = 0 #rough number of bid targets included
        self.status = 0
        if self.id > 0: #i.e. otherwise, just building a single pdf file
            #make the directory
            if not os.path.isdir(self.basename):
                try:
                    os.makedirs(self.basename)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        print ("Fatal. Cannot create pdf output directory: %s" % self.basename)
                        log.critical("Fatal. Cannot create pdf output directory: %s" % self.basename,exc_info=True)
                        exit(-1)
            # have to leave files there ... this is called per internal iteration (not just once0
           # else: #already exists
           #     try: #empty the directory of any previous runs
           #         regex = re.compile(self.filename + '_.*')
           #         [os.remove(os.path.join(self.filename,f)) for f in os.listdir(self.filename) if re.match(regex,f)]
           #     except:
           #         log.error("Unable to clean output directory: " + self.filename,exc_info=True)

            self.filename = None
            if pdf_name is not None:
                #expect the id to be in the pdf_name
                if str(id) in pdf_name:
                    self.filename = os.path.join(self.basename, pdf_name) + ".pdf"

            if self.filename is None:
                if (type(id) == int) or (type(id) == np.int64):
                    if id < 1e9:
                        filename = os.path.basename(self.basename) + "_" + str(id).zfill(3) + ".pdf"
                    else:
                        filename = str(id) + ".pdf"
                else:
                    try:
                        if (type(id) == str) and id.isnumeric() and np.int64(id) > 1e9:
                            filename = str(id) + ".pdf"
                        else:
                            filename = os.path.basename(self.basename) + "_" + str(id) + ".pdf"
                    except:
                        filename = os.path.basename(self.basename) + "_" + str(id) + ".pdf"

                self.filename = os.path.join(self.basename,filename)
        else:
            pass #keep filename as is

        self.pages = None



def make_zeroth_row_header(left_text,show_version=True,redtext=False):
    try:
        # make a .pdf file
        plt.close('all')
        fig = plt.figure(figsize=(G.FIGURE_SZ_X, 0.5))
        gs = gridspec.GridSpec(1, 3)  # one row, 2 columns
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        plt.subplot(gs[0, 0])
        plt.gca().axis('off')
        if redtext or ": -1 (" in left_text: #ie: P(LyA): -1 (negative spectrum)
            plt.text(0, 0.5, left_text, ha='left', va='bottom', fontproperties=font,color='r')
        else:
            plt.text(0, 0.5, left_text, ha='left', va='bottom', fontproperties=font)

        # if G.LyC:
        #     plt.subplot(gs[0, 1])
        #     plt.gca().axis('off')
        #     plt.text(0.5, 0.5, "Lyman Continuum Focus", ha='center', va='bottom', fontproperties=font)

        if show_version:
            plt.subplot(gs[0, 2])
            plt.gca().axis('off')
            title = time.strftime("%Y-%m-%d %H:%M:%S") + "  Version " + G.__version__  # + "  "
            plt.text(1.0, 0.5, title, ha='right', va='bottom', fontproperties=font)
            # plt.suptitle(, fontsize=8, x=1.0, y=0.98,
            #              horizontalalignment='right', verticalalignment='bottom')

    except:
        log.debug("Exception in make_zeroth_row_header final combination", exc_info=True)

def parse_commandline(auto_force=False):
    desc = "(Version %s) Search multiple catalogs for possible object matches.\n\nNote: if (--ra), (--dec), (--par) supplied in " \
           "addition to (--dither),(--line), the supplied RA, Dec, and Parangle will be used instead of the " \
           "TELERA, TELEDEC, and PARANGLE from the science FITS files." % (G.__version__)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--force', help='Do not prompt for confirmation.', required=False,
                        action='store_true', default=False)



    #old CURE options ... no longer supported
    #Some may be re-used for other purposes
    parser.add_argument('-c', '--cure', help='Use Cure processed fits (instead of Panacea).', required=False,
                        action='store_true', default=False)
    parser.add_argument('--sigma', help="Minimum sigma threshold (Cure) to meet in selecting detections", required=False,
                        type=float,default=0.0)
    parser.add_argument('--chi2', help="Maximum chi2 threshold (Cure) to meet in selecting detections", required=False,
                        type=float,default=1e9)


    parser.add_argument('--ra', help='Target RA as decimal degrees or h:m:s.as (end with \'h\')'
                                           'or d:m:s.as (end with \'d\') '
                                           'Examples: --ra 214.963542  or --ra 14:19:51.250h or --ra 214:57:48.7512d'
                                            , required=False)
    parser.add_argument('--dec', help='Target Dec (as decimal degrees or d:m:s.as (end with \'d\'). '
                                      'Negative values must be in quotes with a space before the " -" .'
                                            'Examples: --dec 52.921167  or  --dec " -8:20:55.1" or --dec 52:55:16.20d', required=False)

    #parser.add_argument('--rot',help="Rotation (as decimal degrees). NOT THE PARANGLE.",required=False,type=float)
    parser.add_argument('--par', help="The Parangle in decimal degrees.", required=False, type=float)
    parser.add_argument('--rot', help="The rotation in decimal degrees. (Superceeds use of --par)", required=False, type=float)

    parser.add_argument('--ast', help="Astrometry coordinates file. Use instead of --ra, --dec, --par", required=False)

    parser.add_argument('--obsdate', help="Observation Date. Must be YYYYMMDD.  "
                                          "Must provide for Panacea.", required=False)
    parser.add_argument('--obsid', help="Observation ID (integer). "
                                        "Must provide for Panacea.", required=False, type=int)
    parser.add_argument('--specid', help="SpecID aka CAM (integer) i.e. --specid 13 or --specid 13,14,19  "
                                         "If not specified, all are used. (may be restricted by --ifuid or --ifuslot)", required=False, type=int)
    parser.add_argument('--ifuid', help="IFU ID (integer) *** NOTICE. This is the cable ID.  "
                                        "If not specified, all are used (may be restricted by --specid or --ifuslot)", required=False, type=int)
    parser.add_argument('--ifuslot', help="IFU SLOT ID (integer)  "
                                          "If not specified, all are used (may be restricted by --specid or --ifusid)", required=False, type=int)

    parser.add_argument('-e', '--error', help="Error (+/-) in RA and Dec in arcsecs.", required=False, type=float,
                        default=3.0)

    parser.add_argument('--search', help="Search window (+/-) in RA and Dec in arcsecs (for use only with --ra and --dec).", required=False, type=float)

    parser.add_argument('--fibers', help="Number of fibers to plot in 1D spectra cutout."
                                         "If present, also turns off weighted average.", required=False, type=int)

    parser.add_argument('-n','--name', help="Report filename or directory name (if HETDEX emission lines supplied)",required=False)
    parser.add_argument('--multi', help='*Mandatory. Switch remains only for compatibility. Cannot be turned off.*'
                                        'Produce one PDF file per emission line (in folder from --name).', required=False,
                        action='store_true', default=False)

    parser.add_argument('--dither', help="HETDEX Dither file", required=False)
    parser.add_argument('--path', help="Override path to science fits in dither file", required=False)
    parser.add_argument('--line', help="HETDEX detect line file", required=False)
    parser.add_argument('--fcsdir', help="Flux Calibrated Spectra DIRectory (commonly from rsp1). No wildcards. "
                                         "(see --dets)", required=False)
    parser.add_argument('--dets', help="List of detections (of form '20170314v011_005') or subdirs under fscdir "
                        "(wildcards okay) or file containing a list of detections (one per line)", required=False)

    ##ra dec wave shotid detectid
    parser.add_argument('--coords', help="File containing a list (in order) of RA and Decs (one pair per line)"
                                         " and optionally a shotid and wavelength (use 0 as a placeholder for"
                                         " unspecified shotid or wavelength). Used optionally "
                                         "with --aperture. If --aperture specified, will (re)extract at the exact"
                                         "position. If not specified, will find HETDEX detections within specified"
                                         "--search (first) or --error.",
                        required=False)

    parser.add_argument('--dispatch', help="Dispatched list of directories to process. Auto-created. DO NOT SET MANUALLY",
                        required=False)

    parser.add_argument('--ifu', help="HETDEX IFU (Cure) file", required=False)
    parser.add_argument('--dist', help="HETDEX Distortion (Cure) file base (i.e. do not include trailing _L.dist or _R.dist)",
                        required=False)
    parser.add_argument('--id', help="ID or list of IDs from detect line file for which to search", required=False)

    parser.add_argument('--sn', help="Minimum fiber signal/noise threshold (Panacea) to plot in spectra cutouts",
                        required=False, type=float, default=0.0)
    parser.add_argument('--score', help='Do not build report. Just compute detection scores and output *_fib.txt. '
                                        'Currently incompatible with --cure',
                        required=False, action='store_true', default=False)

    parser.add_argument('-t', '--time', help="Max runtime as hh:mm:ss for in SLURM queue",required=False)
    parser.add_argument('--timex', help="Multiplier on the time for SLURM. i.e. 1.1 would increase the time by 10%%",
                        required=False,type=float,default=1.0)
    parser.add_argument('--email', help="If populated, sends SLURM status to this email address", required=False)

    parser.add_argument('--queue', help="If populated, specifies which TACC queue (vis, gpu) to use.", required=False)

    parser.add_argument('--tasks', help="If populated, specifies how many TACC tasks to use.", required=False)

    parser.add_argument('--ntasks_per_node', help="If populated, specifies the (max) TACC ntasks_per_node to use.", required=False)

    parser.add_argument('--nodes', help="If populated, specifies the maximum TACC nodes to use.", required=False)

    parser.add_argument('--panacea_red',help="Basedir for searching for Panacea reduction files",required=False)

    parser.add_argument('--mini', help='Produce a mini (cellphone friendly) ELiXer image summary', required=False,
                        action='store_true', default=False)
    parser.add_argument('--zoo', help='Produce image cutouts for publication on Zooniverse', required=False,
                        action='store_true', default=False)
    parser.add_argument('--zoox', help='Redact sensitive information AND produce image cutouts for publication on Zooniverse',
                        required=False, action='store_true', default=False)
    parser.add_argument('--jpg', help='Also save report in JPEG format.', required=False,
                        action='store_true', default=False)
    parser.add_argument('--png', help='Also save report in PNG format.', required=False,
                        action='store_true', default=False)

    #1.9.0a3 keeping this around for backward compatibility (if anyone passes it on the call
    #the default behavior now IS blind, so add a not_blind switch to force validation
    parser.add_argument('--blind', help='Do not verify passed in detectIDs. Applies only to HDF5 detectIDs.',
                        required=False, action='store_true', default=True)

    parser.add_argument('--not_blind', help='Explicitly verify passed in detectIDs before processing. Applies only to HDF5 detectIDs.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--allcat', help='Produce individual pages for all catalog matches if there are '
                        'more than 3 matches.', required=False, action='store_true', default=False)

    parser.add_argument('--gaussplots', help='(Debug) output plots of gaussian fits to emission line data',
                        required=False, action='store_true', default=False)
    parser.add_argument('--catcheck', help='Only check to see if there possible matches. Simplified commandline output only',
                        required=False, action='store_true', default=False)

    parser.add_argument('--merge', help='Merge all cat and fib txt files',
                        required=False, action='store_true', default=False)

    parser.add_argument('--remove_duplicates', help='Remove duplicate rows in specified elixer HDF5 catalog file.',
                        required=False)

    parser.add_argument('--merge_unique', help='Merge two ELiXer HDF5 files into a new file keeping the more recent '
                                               'detection. Format: new-file,file1,file2.\nNote: preferred TACC use'
                                               ' with IDEV. Do not use with selixer.', required=False)

    parser.add_argument('--upgrade_hdf5', help='Copy HDF5 file into new format/version. Format old_file,new_file',
                        required=False)

    parser.add_argument('--annulus', help="Inner and outer radii in arcsec (e.g. 10.0,35.2 )", required=False)

    parser.add_argument('--aperture', help="Source extraction aperture (in arcsec) for manual extraction. Must be provided"
                                           " for explicit (re)extraction.", required=False,  type=float)

    parser.add_argument('--ffsky', help='Use Full Field sky subtraction. Default=False.',
                        required=False,action='store_true', default=False)

    parser.add_argument('--wavelength', help="Target wavelength (observed) in angstroms. Used with --annulus or --aperture",
                        required=False, type=float)

    parser.add_argument('--delta_wave',
                        help="Delta wavelength in AA for matching to coordinates. Used with --search",
                        required=False, type=float)

    parser.add_argument('--shotid', help="Integer shotid [YYYYMMDDiii] (optional). Otherwise, searches all.", required=False)

    parser.add_argument('--include_all_amps', help='Override bad amp list and process all amps',
                        required=False, action='store_true', default=False)

    parser.add_argument('--hdf5', help="HDF5 Detections File (see also --dets)", required=False,
                        default=G.HDF5_DETECT_FN)


    parser.add_argument('--recover', help='Recover/continue from previous run. Will append to and NOT overwrite exsiting output.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--no_recover', help='Explicitly override the default (--recover) for dispatch mode.',
                        required=False, action='store_true', default=False)


    parser.add_argument('--prep_recover', help='Clean output directories for recovery run. Executes an interactive script.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--ooops', help='Load Ooops module for SLURM.', required=False,
                        action='store_true', default=False)

    parser.add_argument('--allow_empty_image', help='Allow image cutouts to be empty, otherwise, move to backup catalog.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--sdss', help="SDSS remote query for imaging. Deny (0), Allow (1) if no other catalog available,"
                                       " Force (2) and override any other catalog",
                        required=False, default=1,type=int)  #default to -1 auto shuts off if in dispatch mode

    parser.add_argument('--panstarrs', help="Pan-STARRS remote query for imaging. Deny (0), Allow (1) if no other catalog available,"
                                       " Force (2) and override any other catalog",
                        required=False, default=1,type=int)  #default to -1 auto shuts off if in dispatch mode

    parser.add_argument('--decals', help="DECaLS remote query for imaging. Deny (0), Allow (1) if no other catalog available,"
                                       " Force (2) and override any other catalog",
                        required=False, default=1,type=int) #default to -1 auto shuts off if in dispatch mode

    parser.add_argument('--nophoto', help='Turn OFF the use of archival photometric catalogs.', required=False,
                        action='store_true', default=False)

    parser.add_argument('--fitspec', help='Adjust y-axis range to accomodate entire sepctrum (not just emission line)', required=False,
                        action='store_true', default=False)

    parser.add_argument('--continuum', help='Use HETDEX continuum catalog instead of the standard emission line catalog.'
                                            'Mutually exclusive with --broadline', required=False,
                        action='store_true', default=False)

    parser.add_argument('--broadline', help='Use HETDEX broad emission line catalog instead of the standard emission line catalog.'
                                            'Mutually exclusive with --continuum', required=False,
                        action='store_true', default=False)

    parser.add_argument('--neighborhood', help="Generate report on all HETDEX neighbors within the supplied distance in arcsec",
                        required=False, default=-1.0,type=float)

    parser.add_argument('--neighborhood_only', help="Only generate neighborhood map. Do NOT generate ELiXer report."
                                                    "Value is distance in arcsec for the neighborhood map",
                        required=False, default=-1.0,type=float)

    parser.add_argument('--hdr', help="Override the default HETDEX Data Release version. Specify an integer > 0",
                        required=False, default=0)

    parser.add_argument('--log', help="Logging level. Default (info). Choose: debug, info, error, critical", required=False)


    parser.add_argument('--gridsearch', help='Search a grid around the RA, Dec. 5-tuple.'
                                             'Specify (+/- arcsec, grid size (arcsec), velocity offset (km/s), fwhm (AA), 0=plot, 1=interactive)', required=False)


    parser.add_argument('--fit_sigma', help='Minimum and maximum Gaussian sigma for the line-fitter: 2-tuple.'
                                             'Specify (min,max)', required=False)


    parser.add_argument('--version', help='Print the version to screen.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--lyc', help='Toggle [ON] Lyman Continuum special switch. Do not use unless you know what you are doing.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--deblend', help='Toggle [ON] PSF spectra + flat fnu deblending of neighbors from target spectra.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--dependency', help="For use with SLURM only. Set optional condition and SLURM_ID of job to "
                                             "finish prior to this one starting. e.g: afterok:123456  or afterany:123456",
                        required=False)

    parser.add_argument('--ylim', help='Fixed y-axis limits for full-width 1D plot as (lower,upper)', required=False,type=str)


    parser.add_argument('--known_z', help="Produce plots using this value for redshift", required=False, type=float)

    parser.add_argument('--mcmc', help='Always perform MCMC fit on LSQ pre-fit possible lines.', required=False,
                        action='store_true', default=False)

    # parser.add_argument('--sky_residual', help='Toggle [ON] shot-specific sky residual subtraction for forced-extracions.',
    #                     required=False, action='store_true', default=False)


    parser.add_argument('--viewer', help='Launch the global_config.py set PDF viewer on completion', required=False,
                            action='store_true', default=False)


    parser.add_argument('--check_z', help="Toggle use of catalog counterpart redshift comparisions. "
                                          "Bitmask: 1 = Galaxy Mask, 2 = DEX-GAIA, 4=SDSS, 8=Local Catalogs. "
                                          "Do NOT use unless you specifically know what you are doing. ",
                        required=False, type=int)#default=15) leave the default = None

    parser.add_argument('--special', help="Special purpose modification. The value sets the behavior. Tied to specific code. Do NOT use unless you specifically know what you are doing.",
                        required=False, type=int,default=0)

    parser.add_argument('--cluster', help="Scan all detectids for bright neighbors at high confidence redshift. Specify the elixer h5 file.",
                        required=False)#,default=None)#"elixer_merged_cat.h5")

    parser.add_argument('--require_hetdex', help="If there is no HETDEX data, do not generate a report.",
                        required=False, action='store_true', default=False)#,default=None)#"elixer_merged_cat.h5")

    parser.add_argument('--prefer_g', help="Use g-band instead of r-band for catalog mag comparision if both present.",
                        required=False, action='store_true', default=False)#,default=None)#"elixer_merged_cat.h5")


    parser.add_argument('--voters', help="Toggle use of individual P(LyA) votes. Bitmask. See global_config.py",
                        required=False, type=int)#default=15) leave the default = None

    parser.add_argument('--plya_thresh', help='Specify 1, 2, or 3 (as primary then  1 or 2 secondary) P(LyA) voting thresholds, comma separated floats. Must be stictly 0.0 to 1.0, non-inclusive.',
                        required=False)

    parser.add_argument('--slurm',help="For use with selixer. If (1) automatically queue the SLURM job, if (0) create but"
                                       " do not queue the SLURM job", required=False, type=int, default=1)

    #parser.add_argument('--here',help="Do not create a subdirectory. All output goes in the current working directory.",
    #                    required=False, action='store_true', default=False)

    try:
        args = parser.parse_args()
    except:
        log.critical("Exception! Excpetion parsing command line.",exc_info=True)
        print('Check this common problem. If using --dec with a negative value as d:m:s, the value must be quoted and '
              'there must be a space between the leading quote and the negative sign. e.g. --dec \" -8:20:55.6\"')
        args = None
        return

    try:
        if args.special: #if not None or 0
            G.ELIXER_SPECIAL = args.special
            print(f"***** RUNNING SPECIALIZED CODE X{args.special} *****")
            log.critical(f"***** RUNNING SPECIALIZED CODE X{args.special} *****")
            G.__version__ += f"X{args.special}"
            if args.special == 1:
                G.CHECK_SDSS_Z_CATALOG = False
                G.CHECK_GAIA_DEX_CATALOG = False
                G.CHECK_ALL_CATALOG_BID_Z = False
                G.CHECK_GALAXY_MASK = False

            #residuals
            if args.special >= 1000:
                G.SKY_RESIDUAL_PER_SHOT = True  # if True pull each residusl from the match shot, if False, use the universal model
                G.SKY_RESIDUAL_FITS_PATH = "/scratch/03261/polonius/random_apertures/all_fibers/all/"
                if args.special >= 2000:
                    G.SKY_RESIDUAL_FITS_PREFIX = "fiber_summary_sym_bw_"
                    col = args.special - 2000
                else:
                    G.SKY_RESIDUAL_FITS_PREFIX = "fiber_summary_asym_bw_"
                    col = args.special - 1000

                if args.ffsky:
                    sky_label = "ff"
                else:
                    sky_label = "ll"

                #col is now an integer 0 to 999, though only certain integers have meaning
                G.SKY_RESIDUAL_FITS_COL = f"{sky_label}_stack_{col:03}"

    except:
        pass

    try:
        if args.cluster: #turn off all z-checks (only clustering is used)
            G.CHECK_GALAXY_MASK = False
            G.CHECK_GAIA_DEX_CATALOG = False
            G.CHECK_SDSS_Z_CATALOG = False
            G.CHECK_ALL_CATALOG_BID_Z = False
        elif args.check_z:
            log.info(f"Altering check_z: bitmask={bin(args.check_z)}")
            if args.check_z & 1:
                G.CHECK_GALAXY_MASK = True
            else:
                G.CHECK_GALAXY_MASK = False

            if args.check_z & 2:
                G.CHECK_GAIA_DEX_CATALOG = True
            else:
                G.CHECK_GAIA_DEX_CATALOG = False

            if args.check_z & 4:
                G.CHECK_SDSS_Z_CATALOG = True
            else:
                G.CHECK_SDSS_Z_CATALOG = False

            if args.check_z & 8:
                G.CHECK_ALL_CATALOG_BID_Z = True
            else:
                G.CHECK_ALL_CATALOG_BID_Z = False


    except Exception as e:
        print("Invalid --check_z provided.",e)
        exit(0)

    if args.voters is not None:
        G.VOTER_ACTIVE = args.voters #do nothing ... use the gloabl config value for G.VOTER_ACTIVE

    if G.LAUNCH_PDF_VIEWER is None:
        args.viewer = False

    if args.version:
        print(f"ELiXer version {G.__version__}")
        cl_args = list(map(str.lower, sys.argv))
        if len(cl_args) < 3:
            exit(0)

    if args.lyc:
        G.LyC = True
        G.DeblendSpectra = True

    if args.deblend:
        G.DeblendSpectra = True

    if args.prefer_g:
        G.BANDPASS_PREFER_G = True
    #else leave it as configured in global_config

    if args.mcmc:
        G.FORCE_MCMC = True

    if args.dispatch:
        #if in dispatch mode (SLURM mode) we are already using all/most the cores (as balanced vs memory, etc)
        #so the multiprocess call actually slows down the overall execution
        G.GET_SPECTRA_MULTIPROCESS = False

    #reminder to self ... this is pointless with SLURM given the bash wraper (which does not know about the
    #speccific dir name and just builds elixer.run

    #if args.name is not None:
    #    G.logging.basicConfig(filename="elixer."+args.name+".log", level=G.LOG_LEVEL, filemode='w')
    #else:
    #    print("Missing mandatory paramater --name.")
    #    exit(-1)

    #regardless of setting, --multi must now always be true
    args.multi = True

    if args.recover or ((args.dispatch is not None) and not args.no_recover):
        G.RECOVERY_RUN = True
        args.recover = True

    #first time we need to log anything
    #G.logging.basicConfig(filename=G.LOG_FILENAME, level=G.LOG_LEVEL, filemode='w')


    log.info(args)

    if auto_force:
        args.force = True #forced to be true in dispatch mode

    if args.upgrade_hdf5:
        print("Upgrading catalog file (ignoring all other parameters) ... ")
        return args

    if args.remove_duplicates:
        print("Removing HDF5 duplicate rows (ignoring all other parameters) ... ")
        print("This can take a long time depending on total number of records.")
        return args

    if args.merge or args.merge_unique:
        print("Merging catalogs (ignoring all other parameters) ... ")
        return args

    #don't really use id any more, but pass it into dets
    if args.id is not None:
        #if it is a small number, assume we want to start the local IDs at that number
        #if it is in the HETDEX range, replace --dets with that value (assuming a lookup is being requested)
        try:
            _i = int(args.id)
            if 0 < _i < int(1e9):
                G.UNIQUE_DET_ID_NUM = _i - 1
            else:
                print("Note: old '--id' parameter overriding '--dets'")
                args.dets = args.id
        except:
            print("Note: old '--id' parameter overriding '--dets'")
            args.dets = args.id

    if args.prep_recover:
        print("Attempting to run clean_for_recovery script ... ")
        try:
            import clean_for_recovery
            print("Success\n")
        except Exception as e:
            print("Failed to import ... try again:\n",e)
            try:
                from elixer import clean_for_recovery
                print("Success\n")
            except Exception as e:
                print("Unable to run clean_for_recovery script\n")
                print(e)
        print("prep_recover complete; exiting ...")
        exit(0)

    #if the cutout size (driven by args.error) is smaller than
    #the forced extraction aperture specified, increase the error (window) size
    if args.aperture and args.error:
        if args.aperture > args.error + 0.5:
            log.info(f"Increasing --error ({args.error}) to accomodate --aperture ({args.aperture})")
            args.error = args.aperture

    if not (args.neighborhood_only > 0):
        args.neighborhood_only = False

    if args.shotid is not None:
        try:
            #put into YYYYMMDDxxx as integer format; could be received as YYYYMMDDvXXX
            if 'v' in args.shotid and len(args.shotid)==12:
                args.shotid = int(args.shotid[0:8]+args.shotid[9:])
            else:
                args.shotid = int(args.shotid)

            args.command_line_shotid = args.shotid
        except:
            print("Invalid --shotid. Must be of form (example): 20191129v045 or 20191129045)")
            exit(-1)

    try: #the shotid can be modified in the normal operation
        args.command_line_shotid = args.shotid
    except:
        pass

    if (args.dets is not None) and (args.coords is not None):
        print("Invalid combination of parameters. Cannot specify both --dets and --coords")
        exit(-1)

    if args.not_blind:
        args.blind = False

    if args.allow_empty_image is not None:
        G.ALLOW_EMPTY_IMAGE = args.allow_empty_image
    else:
        G.ALLOW_EMPTY_IMAGE = False

    if args.sdss is not None:
        if args.sdss == 0:
            G.SDSS_ALLOW = False
            G.SDSS_FORCE = False
        elif args.sdss == 1:
            G.SDSS_ALLOW = True
            G.SDSS_FORCE = False
        elif args.sdss == 2:
            G.SDSS_ALLOW = True
            G.SDSS_FORCE = True
        elif args.sdss == -1: #basically, unless explicitly overridden, if we are in dispatch mode, don't use SDSS
                              #since we can easily overwhelm their web interface
            if args.dispatch is not None:
                if (args.nodes is not None) and (int(args.nodes) > 1):
                    G.SDSS_ALLOW = False
                    G.SDSS_FORCE = False
                    print("***notice: --sdss NOT specified. Dispatch is ON. SDSS NOT allowed by default.")
                    log.info("--sdss NOT specified. Dispatch is ON. SDSS NOT allowed by default.")
                else:
                    log.info("--sdss NOT specificed. Dispatch is ON but is only on 1 node. Defaults allowed.")
        else:
            log.warning("Ignoring invalid --sdss value (%d). Using default (Allow == 1)" %args.sdss)
            print("Ignoring invalid --sdss value (%d). Using default (Allow == 1)" %args.sdss)
            G.SDSS_ALLOW = True
            G.SDSS_FORCE = False

    if args.panstarrs is not None:
        if args.panstarrs == 0:
            G.PANSTARRS_ALLOW = False
            G.PANSTARRS_FORCE = False
        elif args.panstarrs == 1:
            G.PANSTARRS_ALLOW = True
            G.PANSTARRS_FORCE = False
        elif args.panstarrs == 2:
            G.PANSTARRS_ALLOW = True
            G.PANSTARRS_FORCE = True
        elif args.panstarrs == -1: #basically, unless explicitly overridden, if we are in dispatch mode, don't use SDSS
                              #since we can easily overwhelm their web interface
#            pass #for now, let the global default rule ... if this is a problem like SDSS, then restrict
            if args.dispatch is not None:
                if (args.nodes is not None) and (int(args.nodes) > 1):
                    G.PANSTARRS_ALLOW = False
                    G.PANSTARRS_FORCE = False
                    print("***notice: --panstarrs NOT specified. Dispatch is ON. PanSTARRS NOT allowed by default.")
                    log.info("--panstarrs NOT specified. Dispatch is ON. PanSTARRS NOT allowed by default.")
                else:
                    log.info("--panstarrs NOT specificed. Dispatch is ON but is only on 1 node. Defaults allowed.")
        else:
            log.warning("Ignoring invalid --panstarrs value (%d). Using default (Allow == 1)" %args.panstarrs)
            print("Ignoring invalid --panstarrs value (%d). Using default (Allow == 1)" %args.panstarrs)
            G.PANSTARRS_ALLOW = True
            G.PANSTARRS_FORCE = False

    if args.decals is not None:
        if args.decals == 0:
            G.DECALS_WEB_ALLOW = False
            G.DECALS_WEB_FORCE = False
        elif args.decals == 1:
            G.DECALS_WEB_ALLOW = True
            G.DECALS_WEB_FORCE = False
        elif args.decals == 2:
            G.DECALS_WEB_ALLOW = True
            G.DECALS_WEB_FORCE = True
        elif args.decals == -1:  # basically, unless explicitly overridden, if we are in dispatch mode, don't use SDSS
            # since we can easily overwhelm their web interface
            #            pass #for now, let the global default rule ... if this is a problem like SDSS, then restrict
            if args.dispatch is not None:
                if (args.nodes is not None) and (int(args.nodes) > 1):
                    G.DECALS_WEB_ALLOW = False
                    G.DECALS_WEB_FORCE = False
                    print("***notice: --decals NOT specified. Dispatch is ON. DECaLS (web) NOT allowed by default.")
                    log.info("--decals NOT specified. Dispatch is ON. DECaLS (web) NOT allowed by default.")
                else:
                    log.info("--decals NOT specificed. Dispatch is ON but is only on 1 node. Defaults allowed.")
        else:
            log.warning("Ignoring invalid --decals value (%d). Using default (Allow == 1)" % args.decals)
            print("Ignoring invalid --decals value (%d). Using default (Allow == 1)" % args.decals)
            G.DECALS_WEB_ALLOW = True
            G.DECALS_WEB_FORCE = False

    #if there is no fall back imaging, we should allow empty imaging
    #2020-02-06 This can create unwanted behavior, even though it is well intentioned; so keep the ALLOW_EMPTY_IMAGE
    #as configured regardless of the settings to allow web calls
    #if (G.DECALS_WEB_ALLOW == False) and (G.PANSTARRS_ALLOW == False) and (G.SDSS_ALLOW == False):
    #    G.ALLOW_EMPTY_IMAGE = True


    if args.nophoto:
        G.USE_PHOTO_CATS = False

    if args.fitspec:
        G.FIT_FULL_SPEC_IN_WINDOW = True

    if args.continuum and args.broadline:
        print("Illegal combination of options. Cannot specify both --continuum and --broadline")
        exit(-1)
    elif args.continuum:
        log.info("Setting CONTINUUM_RULES (args.continuum is set)")
        args.hdf5 = G.HDF5_CONTINUUM_FN
        G.CONTINUUM_RULES = True
        G.MAX_SCORE_ABSORPTION_LINES = 9999.9
        G.DISPLAY_ABSORPTION_LINES = True
    elif args.broadline:
        args.hdf5 = G.HDF5_BROAD_DETECT_FN

    if args.annulus:
        try:
            #args.annulus = tuple(map(float, args.annulus.translate(None, '( )').split(',')))
            args.annulus = tuple(map(float, args.annulus.split(',')))

            if len(args.annulus)==0:
                print("Fatal. Inavlid annulus.")
                log.error("Fatal. Inavlid annulus.")
                exit(-1)
            elif len(args.annulus)==1: #assume the inner radius is zero
                args.annulus = (0,args.annulus[0])
                log.info("Single valued annulus. Assume 0.0 for the inner radius.")
            elif len(args.annulus) > 2:
                print("Fatal. Inavlid annulus.")
                log.error("Fatal. Inavlid annulus.")
                exit(-1)

            if args.annulus[0] > args.annulus[1]:
                print("Fatal. Inavlid annulus. Inner radius larger than outer.")
                log.error("Fatal. Inavlid annulus. Inner radius larger than outer.")
                exit(-1)

            if (args.annulus[0] < 0) or (args.annulus[1] < 0):
                print("Fatal. Inavlid annulus. Negative value.")
                log.error("Fatal. Inavlid annulus. Negative value.")
                exit(-1)

            if (args.annulus[0] > G.MAX_ANNULUS_RADIUS) or (args.annulus[1] > G.MAX_ANNULUS_RADIUS):
                print("Fatal. Inavlid annulus. Excessively large value.")
                log.error("Fatal. Inavlid annulus. Excessively large value.")
                exit(-1)

            if not args.wavelength:
                print("--wavelength required with --annulus")
                log.error("--wavelength required with --annulus")
                exit(-1)

        except: #if annulus provided, this is a fatal exception
            print ("Fatal. Failed to map annulus to tuple.")
            log.error("Fatal. Failed to map annulus to tuple.", exc_info=True)
            exit(-1)

    if args.ylim: #should be a tuple
        try:
            args.ylim = args.ylim.replace(')', '')
            args.ylim = args.ylim.replace('(', '')
        except:
            pass

        try:
            args.ylim = tuple(map(float, args.ylim.split(',')))
            if len(args.ylim) != 2:
                print(f"Non-fatal. Invalid ylim parameters: {args.ylim}. Will ignore.")
                log.error(f"Non-fatal. Invalid ylim parameters: {args.ylim}. Will ignore.")
                args.ylim = None
        except:
            print("Non-fatal. Invalid ylim parameters. Will ignore.")
            log.error("Non-fatal. Invalid ylim parameters. Will ignore.")
            args.ylim = None

    if args.known_z is not None:
        if args.known_z < 0:
            print(f"Non-fatal. Invalid known_z {args.known_z} (must be > 0). Will ignore.")
            log.debug(f"Non-fatal. Invalid known_z {args.known_z} (must be > 0). Will ignore.")
            args.known_z = None
            args.known_z = None

    # if args.sky_residual:
    #     if args.ffsky:
    #         G.SUBTRACT_HETDEX_SKY_RESIDUAL = True
    #     else:
    #         print("Warning! --sky_residual requested but --ffsky missing. Will not substract extra sky residual from spectra.")
    #         G.SUBTRACT_HETDEX_SKY_RESIDUAL = False
    # else:
    #     G.SUBTRACT_HETDEX_SKY_RESIDUAL = False

    if args.gridsearch:

        #first get rid of parenthesis that are not supposed to be there, but are commonly typed in
        try:
            args.gridsearch = args.gridsearch.replace(')','')
            args.gridsearch = args.gridsearch.replace('(','')
        except:
            pass

        try:
            args.gridsearch = tuple(map(float, args.gridsearch.split(',')))
            if len(args.gridsearch) != 5:
                print(f"Fatal. Invalid gridsearch parameters: {args.gridsearch}")
                log.error(f"Fatal. Invalid gridsearch parameters: {args.gridsearch}")
                exit(-1)
            #
            # if len(args.gridsearch) == 3: #old version, does not have the velocity offset max
            #     args.gridsearch = (args.gridsearch[0],args.gridsearch[1],500.0,15.0,args.gridsearch[2])
            #
            # if args.gridsearch[3] == 0:
            #     args.gridsearch = (args.gridsearch[0],args.gridsearch[1],args.gridsearch[2],False)
            # else:
            #     if args.dispatch is None:
            #         args.gridsearch = (args.gridsearch[0], args.gridsearch[1],args.gridsearch[2],True)
            #     else:
            #         log.info("Gridsearch interaction overwritten to False due to dispatch (SLURM) mode.")
            #         print("Gridsearch interaction overwritten to False due to dispatch (SLURM) mode.")
            #         args.gridsearch = (args.gridsearch[0], args.gridsearch[1], args.gridsearch[2], False)
        except:
            # log.info("Exception parsing --gridsearch. Setting to default (3.0,0.4,500.0,15.0,False)",exc_info=True)
            # args.gridsearch = (3.0,0.2,500.0,15.0,False)
            print(f"Fatal. Invalid gridsearch parameters: {args.gridsearch}")
            log.error(f"Fatal. Invalid gridsearch parameters: {args.gridsearch}")
            exit(-1)

    if args.fit_sigma:
        #first get rid of parenthesis that are not supposed to be there, but are commonly typed in
        try:
            args.fit_sigma = args.fit_sigma.replace(')','')
            args.fit_sigma = args.fit_sigma.replace('(','')
        except:
            pass
        try:
            args.fit_sigma = tuple(map(float, args.fit_sigma.split(',')))
            if len(args.fit_sigma) != 2:
                print(f"Fatal. Invalid fit_sigma parameters: {args.fit_sigma}")
                log.error(f"Fatal. Invalid fit_sigma parameters: {args.fit_sigma}")
                exit(-1)

            G.LIMIT_GAUSS_FIT_SIGMA_MIN = args.fit_sigma[0]
            G.LIMIT_GAUSS_FIT_SIGMA_MAX  = args.fit_sigma[1]
            if G.LIMIT_GAUSS_FIT_SIGMA_MIN < 0 or G.LIMIT_GAUSS_FIT_SIGMA_MIN > G.LIMIT_GAUSS_FIT_SIGMA_MAX:
                print(f"Fatal. Invalid fit_sigma parameters: {args.fit_sigma}")
                log.error(f"Fatal. Invalid fit_sigma parameters: {args.fit_sigma}")
                exit(-1)
        except:
            print(f"Fatal. Invalid fit_sigma parameters: {args.fit_sigma}")
            log.error(f"Fatal. Invalid fit_sigma parameters: {args.fit_sigma}")
            exit(-1)


    if args.plya_thresh:

        #first get rid of parenthesis that are not supposed to be there, but are commonly typed in
        try:
            args.plya_thresh = args.plya_thresh.replace(')','')
            args.plya_thresh = args.plya_thresh.replace('(','')
        except:
            pass

        try:
            args.plya_thresh = tuple(map(float, args.plya_thresh.split(',')))
            if len(args.plya_thresh) == 1:
                #one is okay
                t1 = float(args.plya_thresh[0])
                t2 = t1
                t3 = t1
            elif len(args.plya_thresh) == 2:
                #one is okay
                t1 = float(args.plya_thresh[0])
                t2 = float(args.plya_thresh[1])
                t3 = t2
            elif len(args.plya_thresh) == 3:
                #one is okay
                t1 = float(args.plya_thresh[0])
                t2 = float(args.plya_thresh[1])
                t3 = float(args.plya_thresh[2])
            else:
                print(f"Fatal. Invalid plya_thresh parameters: {args.plya_thresh}")
                log.error(f"Fatal. Invalid plya_thresh parameters: {args.plya_thresh}")
                exit(-1)

            #sanity check
            if ((0.0 < t1 < 1.0) and (0.0 < t2 < 1.0) and (0.0 < t2 < 1.0)):
                #all good
                #put in sort order with the base one in the middle ? or assume 1st is primary
                G.PLYA_VOTE_THRESH = t1
                G.PLYA_VOTE_THRESH_1 = t1
                G.PLYA_VOTE_THRESH_2 = t2
                G.PLYA_VOTE_THRESH_3 = t3
                #could be only 1 or two that are unique ... no need to repeat duplicate calls later
                # np.unique returns the list sorted and we want to maintain the order
                _, idx = np.unique([G.PLYA_VOTE_THRESH_1,G.PLYA_VOTE_THRESH_2,G.PLYA_VOTE_THRESH_3], return_index=True)
                idx = sorted(idx)
                G.PLYA_VOTE_THRESH_LIST  = np.array([G.PLYA_VOTE_THRESH_1,G.PLYA_VOTE_THRESH_2,G.PLYA_VOTE_THRESH_3])[idx]
            else:
                print(f"Fatal. Invalid plya_thresh parameters: {args.plya_thresh}")
                log.error(f"Fatal. Invalid plya_thresh parameters: {args.plya_thresh}")
                exit(-1)

        except:
            print(f"Fatal. Invalid plya_thresh parameters: {args.plya_thresh}")
            log.error(f"Fatal. Invalid plya_thresh parameters: {args.plya_thresh}")
            exit(-1)

    if args.wavelength:
        try:
            if not (G.CALFIB_WAVEGRID[0] <= args.wavelength <= G.CALFIB_WAVEGRID[-1]):
                print("Fatal. Invalid target wavelength.")
                log.error("Fatal. Invalid target wavelength.")
                exit(-1)
        except:
            print("Fatal. Invalid target wavelength.")
            log.error("Fatal. Invalid target wavelength.")
            exit(-1)

    if args.delta_wave:
        G.SEARCH_DELTA_WAVELENGTH = args.delta_wave

    if args.gaussplots is not None:
        G.DEBUG_SHOW_GAUSS_PLOTS = args.gaussplots

    if (args.allcat is not None):
        G.FORCE_SINGLE_PAGE = not args.allcat


    if (args.mini is not None) and (args.mini):
        #G.ZOO = False #for now, don't hide, just do the cutouts
        G.ZOO_MINI = True

    if (args.zoo is not None) and (args.zoo):
        #G.ZOO = False #for now, don't hide, just do the cutouts
        G.ZOO_CUTOUTS = True

    if (args.zoox is not None) and (args.zoox):
        G.ZOO = True
        G.ZOO_CUTOUTS = True


    if args.dispatch is not None:
        if args.ra is not None: #then this is from selixer and dispatch needs to be the dets list
            #must run from a list (hence the --dets) ... RA, Dec are single use and must be ignored
            args.ra = None
            args.dec = None

            if args.coords is not None:
                args.coords = args.dispatch
                log.info("Command line: --coords set to --dispatch and ignoring explicit --ra and --dec")
            else:
                args.dets = args.dispatch
                log.info("Command line: --dets set to --dispatch and ignoring --ra and --dec")


    if args.ra is not None:
        if ":" in args.ra:
            try:
                args.ra = float(Angle(args.ra).degree)
            except:
                print("Error. Cannot determine format of RA")
                log.critical("Main exit. Invalid command line parameters.")
                exit(-1)
        elif args.ra[-1].lower() == 'h': #decimal hours
            args.ra = float(args.ra.rstrip('Hh'))  * 15.0
        elif args.ra[-1].lower() == 'd': #decimal (unncessary)
            args.ra = float(args.ra.rstrip('Dd'))
        else:
            args.ra = float(args.ra)

    if args.dec is not None:
        if ":" in args.dec:
            try:
                args.dec = float(Angle(args.dec).degree)
            except:
                print("Error. Cannot determine format of DEC")
                log.critical("Main exit. Invalid command line parameters.")
                exit(-1)
        elif args.dec[-1].lower() == 'd': #decimal (unncessary)
            args.dec = float(args.dec.rstrip('Dd'))
        else:
            args.dec = float(args.dec)

    if args.ast is not None:
        r,d,p = parse_astrometry(args.ast)
        if r is not None:
            args.ra = r
            args.dec = d
            args.par = p


    if not (args.catcheck or args.annulus):
        if args.error < 0:
            print("Invalid --error. Must be non-negative.")
            log.critical("Main exit. Invalid command line parameters.")
            exit(0)

        if not args.force:
            prompt = ""
            if (args.ra is not None) and (args.dec is not None):
                prompt = "Looking for targets +/- %f\" from RA=%f DEC=%f\nProceed (y/n ENTER=YES)?" \
                              % (args.error, args.ra, args.dec)
            elif args.id is not None:
                prompt = "Looking for targets +/- %f\" from ID %s.\nProceed (y/n ENTER=YES)?" \
                         % (args.error,args.id)
            elif (args.dets):
                prompt = "Looking for targets +/- %f\" from detection(s).\nProceed (y/n ENTER=YES)?" \
                            % args.error
            elif (args.coords):
                prompt = "Looking for targets +/- %f\" from detection(s).\nProceed (y/n ENTER=YES)?" \
                            % args.error
            else:
                exit(0)

            i = get_input(prompt)

            if len(i) > 0 and i.upper() !=  "Y":
                print ("Cancelled.")
                log.critical("Main exit. User cancel.")
                exit(0)
            else:
                print()

    if args.include_all_amps:
        G.INCLUDE_ALL_AMPS = True

    if valid_parameters(args):
        return args
    else:
        print("Invalid command line parameters. Cancelled.")
        log.critical("Main exit. Invalid command line parameters.")
        exit(-1)

def valid_parameters(args):

    result = True

    if not (args.catcheck or args.annulus):
        #also check name and error
        if args.name is None:
            print("--name is required")
            result = False

        if args.error is None:
            print("--error is required")
            result = False
    else:
        if args.error is None:
            args.error = 0.0
        if args.name is None:
            args.name = "argscheck"

    if args.search is not None:
        try:
            f = float(args.search)
            if f < 0:
                print("Invalid --search")
                result = False
        except:
            print("Invalid --search")
            result = False

    #must have ra and dec -OR- dither and (ID or (chi2 and sigma))
    if result:
        if (args.ra is None) or (args.dec is None):
            if (args.line is None) and (args.fcsdir is None) and(args.hdf5 is None):
                print("Invalid parameters. Must specify either (--ra and --dec) or detect parameters (--dither, --line, --id, "
                      "--sigma, --chi2, --fcsdir, --hdf5)")
                result =  False
            elif args.cure:
                if (args.ifu is None):
                    print("Warning. IFU file not provided. Report might not contain spectra cutouts. Will search for IFU file "
                          "in the config directory.")
                if (args.dist is None):
                    print("Warning. Distortion file (base) not provided. Report might not contain spectra cutouts. "
                          "Will search for Distortion files in the config directory.")

                #just a warning still return True

    if result and (args.obsdate or args.obsid or args.dither):
        #if you proved obsdata and/or obsid they will be used and you must have all three
        if not (args.obsdate and args.obsid and args.dither):
            msg = "If providing obsdate or obsid you must provide obsdate, obsid, and dither"
            log.error(msg)
            print(msg)
            result = False

    #verify files exist and are not empty (--ast, --dither, --line)
    if result:
        for f in (args.ast,args.dither,args.line):
            if f:
                try:
                    if os.path.exists(f):
                        if os.path.getsize(f) == 0:
                            msg = "Provide file is empty: " + f
                            log.error(msg)
                            print(msg)
                            result = False
                    else:
                        msg  = "Provide file does not exist: " + f
                        log.error(msg)
                        print (msg)
                        result = False
                except:
                    result = False
                    log.error("Exception validating files in commandline.",exc_info=True)
                    print("Exception validating files in commandline. Check log file.")

    if result:
        if args.cure and args.score:
            result = False
            msg = "Incompatible commandline parameters --cure and --score."
            log.error(msg)
            print(msg)


    if result and (args.panacea_red is not None):
        if not os.path.isdir(args.panacea_red):
            result = False
            msg = "Invalid Panacea reduction base directory (--panacea_red ) passed on commandline: "\
                  + args.panacea_red
            log.error(msg)
            print(msg)
        else:
            G.PANACEA_RED_BASEDIR = args.panacea_red

    if args.fcsdir is not None:
        if args.fcsdir[-1] == "/":
            args.fcsdir = args.fcsdir.rstrip("/")
        elif args.fcsdir[-1] == "\\":
            args.fcsdir = args.fcsdir.rstrip("\\")


    return result


def build_hd(args):
    #if (args.dither is not None):
    if (args.line is not None) or (args.fcsdir is not None) or (args.hdf5 is not None):#or (args.id is not None):
            return True

    return False

def build_hetdex_section(pdfname, hetdex, detect_id = 0,pages=None,annulus=False):
    #detection ids are unique (for the single detect_line.dat file we are using)
    if pages is None:
        pages = []
    try:
        if annulus:
            pages = hetdex.build_hetdex_annulus_data_page(pages, detect_id)
        else:
            pages = hetdex.build_hetdex_data_page(pages,detect_id)
    except:
        log.error("Exception calling hetdex.build_hetdex_data_page(): ", exc_info=True)

    if pages is not None:
        if (PyPDF is not None):
            build_report_part(pdfname,pages)
            pages = None

    return pages


def build_pages (pdfname,match,ra,dec,error,cats,pages,num_hits=0,idstring="",base_count = 0,target_w=0,fiber_locs=None,
                 target_flux=None,annulus=None,obs=None,detobj=None):

    _NUDGE_MAG_APERTURE_CENTER_SAVED = G.NUDGE_MAG_APERTURE_CENTER
    if (target_flux is None) or (target_flux < 0):
        target_flux = 0.0

    def update_aperture_rules(restore_nudge=0.0):
        """
        This is a very clumsy way to do this, but it is fast to implement and currently safe since we execute
        only one detection at a time.
        :return:
        """
        return
        try:
            if detobj is not None:
                if detobj.fibers is not None:
                    if int(detobj.fibers[0].dither_date) > G.NUDGE_MAG_APERTURE_MAX_DATE:
                        G.NUDGE_MAG_APERTURE_CENTER = 0.0
                        log.info("Aperture nudge dis-allowed (0.0)")
                    else:
                        log.info("Aperture nudge allowed (%f)" %G.NUDGE_MAG_APERTURE_CENTER)
        except:
            log.debug("Aperture rules NOT updated.",exc_info=True)

    #if a report object is passed in, immediately append to it, otherwise, add to the pages list and return that
    section_title = idstring
    count = 0

    log.info("Building page for %s" %pdfname)

    cat_count = 0
    #extra, non-local catalogs to iterate through IF primary catalog imaging is empty (but we are in the footprint)
    added_decals = False
    added_panstarrs = False
    added_sdss = False
    added_catch_all = False

    if cats is not None:
        log.debug("Checking imaging catalogs (%s)" %(str(cats)))

        if catalogs.cat_decals_web.DECaLS in [type(x) for x in cats]:
            added_decals = True

        if catalogs.cat_panstarrs.PANSTARRS in [type(x) for x in cats]:
            added_panstarrs = True

        if catalogs.cat_sdss.SDSS in [type(x) for x in cats]:
            added_sdss = True

    else:
        log.debug("Imaging catalogs is None")

    num_remaining_cats = len(cats)
    list_of_catalog_cutouts = [] #each catalog that overlaps the RA, Dec returns a list of dictionaries of info

    for c in cats:
        num_remaining_cats -= 1

        if (c is None) or (isinstance(c,list)): #i.e. there are no cats, but it is a list with an empty list
            continue

        if annulus is not None:
            cutout = c.get_stacked_cutout(ra,dec,window=annulus[1])

            #if cutout is not None:
            try:
                r = c.build_annulus_report(obs=obs,cutout=cutout,section_title=section_title)
            #else:
            #    r = None
            except:
                log.error("Exception in elixer::build_pages",exc_info=True)
                r = None

        else:
            try:
                if match is None: #just for down-stream compatibility
                    reset_match = True
                    match = match_summary.Match()
                else:
                    reset_match = False

                update_aperture_rules(_NUDGE_MAG_APERTURE_CENTER_SAVED)
                r = c.build_bid_target_reports(match,ra, dec, error,num_hits=num_hits,section_title=section_title,
                                               base_count=base_count,target_w=target_w,fiber_locs=fiber_locs,
                                               target_flux=target_flux,detobj=detobj)

                if G.BUILD_REPORT_BY_FILTER and r: #here 'r' is a list of dictionaries ("cutouts") from the catalog
                    try:
                        if np.all([x['cutout'] is None for x in r]): #if all the cutous are None, don't add to the list
                            if num_remaining_cats == 0: #if we are out of the original list, see if we can use web calls
                                if isinstance(c,catalogs.cat_decals_web.DECaLS): #this was DECaLS, so fall back ...
                                    if G.PANSTARRS_ALLOW:
                                        cats.append(cat_panstarrs)
                                        num_remaining_cats += 1
                                    elif G.SDSS_ALLOW:
                                        cats.append(cat_sdss)
                                        num_remaining_cats += 1
                                elif isinstance(c,catalogs.cat_panstarrs.PANSTARRS): #this was PanSTARRS so fall back ...
                                    if G.SDSS_ALLOW:
                                        cats.append(cat_sdss)
                                        num_remaining_cats += 1

                                if num_remaining_cats == 0: #out of catalogs to fall back on, keep this r for whatever it is worth
                                    list_of_catalog_cutouts.append(r)
                        else:
                            list_of_catalog_cutouts.append(r)
                    except:
                        list_of_catalog_cutouts.append(r)

                if reset_match:
                    match = None
            except:
                log.error("Exception in elixer::build_pages",exc_info=True)
                r = None

        count = 0
        if G.BUILD_REPORT_BY_FILTER:
            #just iterate to the next catalog and continue to build up the list_of_catalog_cutouts
            if len(list_of_catalog_cutouts) == 0 and num_remaining_cats < 1: # r was None ... no page was created, probably an empty region
                if G.DECALS_WEB_ALLOW and not added_decals: #not FORCE ... that is handled differently
                    cats.append(catalogs.CatalogLibrary().get_decals_web())
                    added_decals = True
                elif G.PANSTARRS_ALLOW and not added_panstarrs: #not FORCE ... that is handled differently
                    cats.append(catalogs.CatalogLibrary().get_panstarrs())
                    added_panstarrs = True
                elif G.SDSS_ALLOW and not added_sdss:
                    cats.append(catalogs.CatalogLibrary().get_sdss())
                    added_sdss = True
                elif not added_catch_all:
                    cats.append(catalogs.CatalogLibrary().get_catch_all())
                    added_catch_all = True
        else:
            if (r is not None) and (len(r) > 1): #always adds figure for "No matching targets"
                cat_count+= 1
                #todo: check that we have imaging? if there is none, go ahead to the next catalog?

                if (cat_count > 1) and G.SINGLE_PAGE_PER_DETECT:
                    msg = "INFO: More than one catalog matched .... taking top catalog only. Skipping PDF for %s" % c.Name
                    print(msg)
                    log.info(msg)
                else:
                    if PyPDF is not None:
                        build_report_part(pdfname,r)
                    else:
                        pages = pages + r
                    count = max(0,len(r)-1) #1st page is the target page
            elif num_remaining_cats < 1: # r was None ... no page was created, probably an empty region
                if G.DECALS_WEB_ALLOW and not added_decals: #not FORCE ... that is handled differently
                    cats.append(catalogs.CatalogLibrary().get_decals_web())
                    added_decals = True
                elif G.PANSTARRS_ALLOW and not added_panstarrs: #not FORCE ... that is handled differently
                    cats.append(catalogs.CatalogLibrary().get_panstarrs())
                    added_panstarrs = True
                elif G.SDSS_ALLOW and not added_sdss:
                    cats.append(catalogs.CatalogLibrary().get_sdss())
                    added_sdss = True
                elif not added_catch_all:
                    cats.append(catalogs.CatalogLibrary().get_catch_all())
                    added_catch_all = True

    #move to AFTER summary section built as we need updates from it (specifically, the catalog matches and ellipses that
    # are "selected"
    # #done going through catalogs
    # if detobj:
    #     detobj.check_spec_solutions_vs_catalog_counterparts()
    #     detobj.check_clustering_redshift()

    if G.BUILD_REPORT_BY_FILTER:
        #we've gone through all the catalogs (including the web catalogs if necessary)
        if len(list_of_catalog_cutouts) > 0:
            #todo: the count is the number of pdf sections to add (should just be one)
            #todo: build up the PDF section (cat_base.py)
            #this is in the base class, so any catalog will do
            try:
                #note that this is as a single section, rather than individual sections for the
                #images and then the catalog match table at the bottom
                r = cats[0].build_cat_summary_pdf_section(list_of_catalog_cutouts, match, ra, dec, error, target_w,
                                          fiber_locs, target_flux,detobj)
                #self.add_bid_entry(entry)

                if PyPDF is not None:
                    build_report_part(pdfname,r)
                else:
                    pages = pages + r
                count = len(r) # max(0,len(r)-1) #1st page is the target page

            except:
                log.error("Unexpected exception calling to cat_base.py build_cat_summary_pdf_section",exc_info=True)
        else: #todo: need to build up a blank image (or ... I think that happens anyway later)
            count = 0

    #done going through catalogs
    if detobj:
        detobj.check_spec_solutions_vs_catalog_counterparts()
        detobj.check_clustering_redshift()

    G.NUDGE_MAG_APERTURE_CENTER = _NUDGE_MAG_APERTURE_CENTER_SAVED
    return pages, count


def open_report(report_name):
    return PdfPages(report_name)

def close_report(report):
    if report is not None:
        report.close()

def add_to_report(pages,report):
    if (pages is None) or (len(pages) == 0):
        return

    print("Adding to report ...")
    rows = len(pages)

    try:
        for r in range(rows):
            report.savefig(pages[r])
    except:
        log.error("Exception in elixer::add_to_report: ", exc_info=True)

    return


def build_report(pages,report_name):
    if (pages is None) or (len(pages) == 0):
        return

    print("Finalizing report ...")

    try:

        pdf = PdfPages(report_name)
        rows = len(pages)

        for r in range(rows):
            pdf.savefig(pages[r])

        pdf.close()
        print("File written: " + report_name)
    except:
        log.error("Exception in elixer::build_report: ", exc_info=True)
        try:
            print("PDF FAILURE: " + report_name)
            pdf.close()
        except:
            pass

    return



def build_report_part(report_name,pages,page_num=None):
    try:
        if (pages is None) or (len(pages) == 0):
            return

        global G_PDF_FILE_NUM

        if page_num is None:
            G_PDF_FILE_NUM += 1
            part_name = report_name + ".part%s" % (str(G_PDF_FILE_NUM).zfill(4))
        else:
            part_name = report_name + ".part%s" % (str(page_num).zfill(4))

        pdf = PdfPages(part_name)
        rows = len(pages)

        for r in range(rows):
            pdf.savefig(pages[r])

        pdf.close()
    except:
        log.error("Exception in elixer::build_report_part: ", exc_info=True)
        try:
            print("PDF PART FAILURE: " + part_name)
            pdf.close()
        except:
            pass

    return


def join_report_parts(report_name, bid_count=0):

    if PyPDF is None:
        return
    print("Finalizing report ...")

    metadata = PyPDF.IndirectPdfDict(
        Title='Emission Line eXplorer Report',
        Author="HETDEX, Univ. of Texas",
        Keywords='ELiXer Version = ' + G.__version__)

    error = False
    if report_name[-1] == '!': #this is a problem report (still generate, but there is an error we want to recover later)
        report_name = report_name.rstrip('!')
        error = True
        #return #for now, don't write anything

    if G.SINGLE_PAGE_PER_DETECT:
        #part0001 is the hetdex part (usually 2 pages)
        #part0002 is the catalog part (at least 2 pages, but if greater than MAX_COMBINED_BID_TARGETS
        #         then 2 pages + 1 page for each target
        if (True):
#        if (bid_count <= G.MAX_COMBINE_BID_TARGETS):
            log.info("Creating single page report for %s. Bid count = %d" % (report_name, bid_count))
            list_pages = []
            extra_pages = []

            first_page = True

            for i in range(G_PDF_FILE_NUM+1): #there may be a zeroth image
                #use this rather than glob since glob sometimes messes up the ordering
                #and this needs to be in the correct order
                #(though this is a bit ineffecient since we iterate over all the parts every time)
                part_name = report_name+".part%s" % str(i).zfill(4)
                if os.path.isfile(part_name):
                    pages = PyPDF.PdfReader(part_name).pages
                    if first_page:
                        first_page = False
                        for p in pages:
                            list_pages.append(p)
                    else: #get the first two, then go single after that
                        for j in range(min(2,len(pages))):
                            list_pages.append(pages[j])

                        # todo: keep all others as individual pages
                        for j in range(2,len(pages)):
                            extra_pages.append(pages[j])

            if len(list_pages) > 0:
                merge_page = PyPDF.PageMerge() + list_pages
            else:
                #there is nothing to merge
                log.info("No pages to merge for " + report_name)
                print("No pages to merge for " + report_name)
                return

            scale = 1.0 #full scale
            y_offset = 0

            #need to count backward ... position 0,0 is the bottom of the page
            #each additional "page" is advanced in y by the y height of the previous "page"
            for i in range(len(merge_page) - 1, -1, -1):
                page = merge_page[i]
                page.scale(scale)
                page.x = 0
                page.y = y_offset
                if (i == 1) and (G.ZEROTH_ROW_HEADER) and (len(merge_page) > 2):
                    y_offset = scale * (page.box[3] - 18.0) #trim excess vertical space from the zeroth row header
                else:
                    y_offset = scale* page.box[3] #box is [x0,y0,x_top, y_top]

            if not report_name.endswith(".pdf"):
                report_name += ".pdf"

            # if error:
            #     report_name = report_name.replace(".pdf", "_FAIL.pdf")
            # if error:
            #     report_name = "FAIL_" + report_name
            writer = PyPDF.PdfWriter(report_name)

            try:
                writer.addPage(merge_page.render())
                writer.trailer.Info = metadata

                # now, add (but don't merge) the other parts
                for p in extra_pages:
                    writer.addPage(p)

                writer.write()
            except:
                log.error("Error writing out pdf: " + report_name, exc_info = True)

        else: #want a single page, but there are just too many sub-pages

            #todo: merge the top 2 pages (the two HETDEX columns and the catalog summary row)

            list_pages = []
            log.info("Single page report not possible for %s. Bid count = %d" %(report_name,bid_count))
            part_num = 0
            list_pages_top2 = []
            list_pages_bottom = []
            for i in range(G_PDF_FILE_NUM):
                # use this rather than glob since glob sometimes messes up the ordering
                # and this needs to be in the correct order
                # (though this is a bit ineffecient since we iterate over all the parts every time)

                part_name = report_name + ".part%s" % str(i + 1).zfill(4)
                if os.path.isfile(part_name):
                    pages = PyPDF.PdfReader(part_name).pages
                    part_num = i + 1
                    for p in pages:
                        list_pages.append(p)
                        if len(list_pages_top2) < 2:
                            list_pages_top2.append(p)
                        else:
                            list_pages_bottom.append(p)
                    break # just merge the first part

            merge_page = PyPDF.PageMerge() + list_pages
            merge_page_top2 = PyPDF.PageMerge() + list_pages_top2

            scale = 1.0  # full scale
            y_offset = 0
            # need to count backward ... position 0,0 is the bottom of the page
            # each additional "page" is advanced in y by the y height of the previous "page"
            for i in range(len(merge_page) - 1, -1, -1):
                page = merge_page[i]
                page.scale(scale)
                page.x = 0
                page.y = y_offset
                y_offset = scale * page.box[3]  # box is [x0,y0,x_top, y_top]

            if not report_name.endswith(".pdf"):
                report_name += ".pdf"

            # if error:
            #     report_name = report_name.replace(".pdf", "_FAIL.pdf")

            writer = PyPDF.PdfWriter(report_name)
            writer.addPage(merge_page.render())

            #now, add (but don't merge) the other parts
            for i in range(part_num,G_PDF_FILE_NUM):
                # use this rather than glob since glob sometimes messes up the ordering
                # and this needs to be in the correct order
                # (though this is a bit ineffecient since we iterate over all the parts every time)
                part_name = report_name + ".part%s" % str(i + 1).zfill(4)
                if os.path.isfile(part_name):
                    writer.addpages(PyPDF.PdfReader(part_name).pages)

            writer.trailer.Info = metadata

            try:
                writer.write()
            except:
                log.error("Error writing out pdf: " + report_name, exc_info=True)

    else:
        log.info("Creating multi-page report for %s. Bid count = %d" % (report_name, bid_count))
        writer = PyPDF.PdfWriter()
        #for --multi the file part numbers are unique. Only the first file starts with 001. The second starts with
        #where the first left off
        for i in range(G_PDF_FILE_NUM):
            #use this rather than glob since glob sometimes messes up the ordering
            #and this needs to be in the correct order
            #(though this is a bit ineffecient since we iterate over all the parts every time)
            part_name = report_name+".part%s" % str(i+1).zfill(4)
            if os.path.isfile(part_name):
                writer.addpages(PyPDF.PdfReader(part_name).pages)

        writer.trailer.Info = metadata

        # if error:
        #     report_name = report_name.replace(".pdf", "_FAIL.pdf")

        writer.write(report_name)

    print("File written: " + report_name)


def delete_report_parts(report_name):
    if report_name[-1] == "!": #remove error marker, if present
        report_name = report_name.rstrip("!")
    for f in glob.glob(report_name+".part*"):
        os.remove(f)

def confirm(hits,force):

    if not force:

        if hits < 0:
            msg = "\n%d total possible matches found (no overlapping catalogs).\nProceed anyway (y/n ENTER=YES)?" % hits
        else:
            msg = "\n%d total possible matches found.\nProceed (y/n ENTER=YES)?" % hits

        i = get_input(msg)

        if len(i) > 0 and i.upper() != "Y":
            print("Cancelled.")
            return False
        else:
            print()
    else:
        print("%d possible matches found. Building report..." % hits)

    return True


def ifulist_from_detect_file(args):
    ifu_list = []
    if args.line is not None:
        try:
            with open(args.line, 'r') as f:
                f = ft.skip_comments(f)
                for l in f:
                    toks = l.split()
                    if len(toks) > 17: #this may be an aggregate line file (last token = ifuxxx)
                        if "ifu" in toks[17]:
                            ifu = str(toks[17][-3:])  # ifu093 -> 093
                            if ifu in ifu_list:
                                continue
                            else:
                                ifu_list.append(ifu)
        except:
            log.info("Exception checking detection file for ifu list", exc_info=True)
    return ifu_list


def write_fibers_file(filename,hd_list):
    if not filename or not hd_list or (len(hd_list) == 0):
        return None

    sep = "\t"

    write_header = True
    if G.RECOVERY_RUN:
        try:
            if os.path.isfile(filename):
                write_header = False
        except:
            log.info("Unexpected exception (not fatal) in elixer::write_fibers_file",exc_info=True)

        try:
            f = open(filename, 'a+') #open for append, but create if it does not exist
        except:
            log.error("Exception create match summary file: %s" % filename, exc_info=True)
            return None
    else: #not a recovery run ... overwrite what is there
        try:
            f = open(filename, 'w')
        except:
            log.error("Exception create match summary file: %s" % filename, exc_info=True)
            return None

    if write_header:
        #write header info
        headers = [
            "fullname / PDF name",
            "input (entry) ID",
            "detect ID",
            #"detection quality score",
            "emission line RA (decimal degrees)",
            "emission line Dec (decimal degrees)",
            "emission line wavelength (AA)",
            #"emission line sky X",
            #"emission line sky Y",
            "emission line sigma (significance) for cure or S/N for panacea",
            "emission line chi2 (point source fit) (cure)",
            "emission line estimated fraction of recovered flux",
            "emission line flux (electron counts)",
            "emission line flux (cgs)",
            "emission line continuum flux (electron counts)",
            "emission line continuum flux (cgs)",
            "emission line equivalent width (observed) [estimated]",
            "P(LAE)/P(OII)",
            "number of fiber records to follow (each consists of the following columns)",
            "  fiber_id string (panacea) or reduced science fits filename (cure)",
            "  observation date YYYYMMDD",
            "  observation ID (for that date)",
            "  exposure ID",
            "  fiber number on full CCD (1-448)",
            "  RA of fiber center",
            "  Dec of fiber center",
            "  X of the fiber center in the IFU",
            "  Y of the fiber center in the IFU",
            "  S/N of emission line in this fiber",
            #"  weighted quality score",
            "  X coord on the CCD for the amp of this emission line in this fiber (as shown in ds9)",
            "  Y coord on the CCD for the amp of this emission line in this fiber (as shown in ds9)",
            "  the next fiber_id string and so on ..."
        ]


        # write help (header) part
        f.write("# version " + str(G.__version__) + "\n")
        f.write("# date time " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n" )
        f.write("# each row contains one emission line with accompanying fiber information\n")
        col_num = 0
        for h in headers:
            col_num += 1
            f.write("# %d %s\n" % (col_num, h))

    #entry_num = 0
    for hd in hd_list:
        for emis in hd.emis_list:
            #entry_num += 1
            #f.write(str(entry_num))
            f.write(str(emis.pdf_name))
            f.write(sep + str(emis.entry_id))
            f.write(sep + str(emis.id))
            #if emis.dqs is None: #as of 1.4.0a11+ not using dqs
            #    emis.dqs_score()
            #f.write(sep + str(emis.dqs))
            if emis.wra:
                f.write(sep + str(emis.wra))
                f.write(sep + str(emis.wdec))
            else:
                f.write(sep + str(emis.ra))
                f.write(sep + str(emis.dec))
            f.write(sep + str(emis.w))

        #dead ... no longer being calculated
           # f.write(sep + str(emis.x))
           # f.write(sep + str(emis.y))

            if emis.snr is not None and emis.snr != 0:
                f.write(sep + str(emis.snr))
            elif emis.sn is not None and emis.sn != 0:
                f.write(sep + str(emis.sn))
            else:
                f.write(sep + str(emis.sigma))

            f.write(sep + str(emis.chi2))
            f.write(sep + str(emis.fluxfrac))
            f.write(sep + str(emis.dataflux))
            f.write(sep + str(emis.estflux))
            f.write(sep + str(emis.cont))
            f.write(sep + str(emis.cont_cgs))
            f.write(sep + str(emis.eqw_obs))
            f.write(sep + str(emis.p_lae_oii_ratio))

            f.write(sep + str(len(emis.fibers)))

            for fib in emis.fibers:
                f.write(sep + str(fib.idstring))
                f.write(sep + str(fib.dither_date))
                f.write(sep + str(fib.obsid))
                f.write(sep + str(fib.expid))
                f.write(sep + str(fib.number_in_ccd))
                f.write(sep + str(fib.ra)) #of fiber center
                f.write(sep + str(fib.dec))
                f.write(sep + str(fib.sky_x)) #of fiber center
                f.write(sep + str(fib.sky_y))
                f.write(sep + str(fib.sn))
                #f.write(sep + str(fib.dqs)) #as of 1.4.0a11 not using dqs
                f.write(sep + str(fib.ds9_x))
                f.write(sep + str(fib.ds9_y))

            f.write("\n")

    msg = "File written: " + filename
    log.info(msg)
    print(msg)


def which(file):
    try:
        path = os.getenv('PATH')
        for p in path.split(os.path.pathsep):
            p = os.path.join(p, file)
            if os.path.exists(p) and os.access(p, os.X_OK):
                return p
    except:
        log.info("Exception in which",exc_info=True)
        return None



def convert_pdf(filename, resolution=150, jpeg=False, png=True):
    """
    The call from outside ... now basically calls the old internals as run_convert_pdf
    and can retry if there is a detected problem

    :param filename:
    :param resolution:
    :param jpeg:
    :param png:
    :return:
    """

    if filename is None:
        return -1

    if filename[-1] == "!":
        filename = filename.rstrip("!")

    try:
        ext = filename[-4:]
        if ext.lower() != ".pdf":
            try:
                log.debug("Invalid filename passed to elixer::convert_pdf(%s)" % filename)
            except:
                return -1
    except:
        try:
            log.debug("Invalid filename passed to elixer::convert_pdf(%s)" %filename)
        except:
            return -1

    try:
        #check that the file exists (can be a timing issue on tacc)
        if not os.path.isfile(filename):
            log.info("Error converting (%s) to image type. File not found (may be filesystem lag. Will sleep and retry."
                     %(filename) )
            time.sleep(5.0) #5 sec should be plenty

            if not os.path.isfile(filename):
                log.info(
                    "Error converting (%s) to image type. File still not found. Aborting conversion."
                    % (filename))
                return -1
    except:
        pass

    try:
        max_retries = 3
        retry_ct = 0
        systemcalls= ["pdftoppm","convert"]  #convert
        while retry_ct < max_retries:
            retry_ct += 1
            if (run_convert_pdf(filename, resolution=resolution, jpeg=jpeg, png=png,
                                systemcall=systemcalls[(retry_ct-1)%len(systemcalls)]) < 0):
                retry = 99
                break
            else: #check the result
                #BOTH png and jpeg could be generated, but we are only going to check one (the png preferred)
                if png:
                    image_name = filename.rstrip(".pdf") + ".png"
                elif jpeg:
                    image_name = filename.rstrip(".pdf") + ".jpg"

                try:
                    size = os.path.getsize(image_name)
                    if OS_PNG_ONLY:
                        log.debug(f"Conversion assumed good at ({size}) for {image_name}. Only OS conversions allowed.")
                        retry = 99
                        break
                    elif size > 430000: #some are legit conversions though
                        log.debug(f"Conversion filesize ({size}) good for {image_name}.")
                        retry = 99
                        break
                    elif (retry_ct < max_retries):
                        img_dim = check_imagefile_dimensions(image_name)
                        #check for none or height
                        if img_dim is None or img_dim[1] > 1600 :
                            #just the cutouts is a bit over 900
                            #the normal full size is ~ 1838 pix
                            # could still be okay, but we will retry anyway .. if retries are exhausted, it will stick
                            log.info(f"Small filesize ({size}) for {image_name}. Will assume missing data and retry.")
                            os.remove(image_name)
                            time.sleep(5.0 * retry_ct)  #sleep in increasing chunks of 5 seconds to let memory clear
                        else:
                            log.debug(f"Conversion filesize ({size}) good for {image_name}. Incomplete report, reduced size {img_dim}).")
                            retry = 99
                            break
                    else:
                        log.info(f"Small filesize ({size}) for {image_name}. Out of retries.")
                except:
                    log.info(f"Could not get file size for {image_name}. Aborting retries.")
                    retry = 99
                    break
    except:
        log.error(f"Exception converting PDF {filename} to image type.", exc_info=True)


def check_imagefile_dimensions(fn):
    """
    Try to get the image dimensions for a given image file
    :param fn:
    :return:
    """
    img_dim = None
    try:
        with open(fn, "rb") as f:
            ImPar=PIL_ImageFile.Parser()
            chunk = f.read(2048)
            count=2048
            while chunk != "":
                ImPar.feed(chunk)
                if ImPar.image:
                    break
                chunk = f.read(2048)
                count+=2048
            img_dim = ImPar.image.size
    except:
        log.debug("Unable to check image size from file.")
    return img_dim

def run_convert_pdf(filename, resolution=150, jpeg=False, png=True,systemcall="pdftoppm"):
    """

    :param filename:
    :param resolution:
    :param jpeg:
    :param png:
    :return: an integer: -1 is a fatal error (do not conitnue), 0 is all good, 1 is retry error
    """

    #first two checks are redundant with convert_pdf(), but kept for sanity purposes or if this is called directly

    #file might not exist, but this will just trap an execption
    if filename is None:
        return -1

    if filename[-1] == "!":
        filename = filename.rstrip("!")

    try:
        ext = filename[-4:]
        if ext.lower() != ".pdf":
            try:
                log.debug("Invalid filename passed to elixer::convert_pdf(%s)" % filename)
            except:
                return -1
    except:
        try:
            log.debug("Invalid filename passed to elixer::convert_pdf(%s)" %filename)
        except:
            return -1


    try:
        #check that the file exists (can be a timing issue on tacc)
        if not os.path.isfile(filename):
            log.info("Error converting (%s) to image type. File not found (may be filesystem lag. Will sleep and retry."
                     %(filename) )
            time.sleep(5.0) #5 sec should be plenty

            if not os.path.isfile(filename):
                log.info(
                    "Error converting (%s) to image type. File still not found. Aborting conversion."
                    % (filename))
                return -1


        # wand.exceptions.PolicyError: not authorized  ....
        #
        # A workaround for now is to simply change the policy in
        #           /etc/ImageMagick-6/policy.xml for PDFs to read :
        # <policy domain="coder" rights="read" pattern="PDF" />

        if use_wand:
            pages = Image(filename=filename, resolution=resolution)
            for i, page in enumerate(pages.sequence):
                with Image(page) as img:
                    img.colorspace = 'rgb'

                    if jpeg:
                        img.format = 'jpg'
                        image_name = filename.rstrip(".pdf") + ".jpg"
                        img.save(filename=image_name)
                        print("File written: " + image_name)

                    if png:
                        img.format = 'png'
                        image_name = filename.rstrip(".pdf") + ".png"
                        img.save(filename=image_name)
                        print("File written: " + image_name)

        elif OS_PNG_ONLY:
            try:
                if G.ALLOW_SYSTEM_CALL_PDF_CONVERSION:
                    #try pdftoppm or convert
                    if (systemcall == "pdftoppm") and (which("pdftoppm") is not None):
                        try:
                            log.info("Attempting blind system call to pdftoppm to convert ... ")
                            os.system("pdftoppm %s %s -png -singlefile" % (filename, filename.rstrip(".pdf")))
                            log.info("No immediate error reported on pdftoppm call ... ")
                        except Exception as e:
                            if type(e) is pdf2image.exceptions.PDFInfoNotInstalledError:
                                log.error("System call conversion failed (PDFInfoNotInstalledError).", exc_info=False)
                            else:
                                log.error("System call conversion failed.",exc_info=True)
                    elif which ("convert") is not None:
                        try:
                            log.info("Attempting blind system call to convert ... ")
                            # alternate call for wrangler
                            # base is really 150 dpi but have to set convert to 200 to mimic resolution
                            os.system("convert -density 200 %s %s" % (filename, filename.rstrip(".pdf") + ".png"))
                            log.info("No immediate error reported on convert call ... ")
                        except Exception as e:
                            log.error("System call conversion failed.", exc_info=True)
                    else:
                        log.error("No viable system call available to convert PDF to PNG")
            except:
                log.error("System call (pdftoppm) conversion failed.", exc_info=True)
        else:
            #this does not currently work on the hub
            pages = convert_from_path(filename,resolution)
            if png:
                for i in range(len(pages)):
                    if i > 0:
                        image_name = filename.rstrip(".pdf") + "_p%02d.png" %i
                    else:
                        image_name = filename.rstrip(".pdf") + ".png"
                    pages[i].save(image_name,"PNG")
                    print("File written: " + image_name)

            if jpeg:
                for i in range(len(pages)):
                    if i > 0:
                        image_name = filename.rstrip(".pdf") + "_p%02d.jpg" %i
                    else:
                        image_name = filename.rstrip(".pdf") + ".jpg"
                    pages[i].save(image_name,"JPEG")
                    print("File written: " + image_name)

    except Exception as e:
        if type(e) is OSError:
            log.error("Error (1) converting pdf to image type: (OSError: probably memory)" + filename, exc_info=False)
        elif type(e) is FileNotFoundError:
            log.error("Error (1) converting pdf to image type: (FileNotFoundError)" + filename, exc_info=False)
        # elif type(e) is PDFInfoNotInstalledError:
        #     log.error("Error (1) converting pdf to image type: (PDFInfoNotInstalledError)" + filename, exc_info=False)
        elif ('poppler' in str(e)) or ("PDFInfoNotInstalledError" in str(type(e))):
            log.error("Error (1) converting pdf to image type: (pdfinfo cannot find poppler)" + filename, exc_info=True)
        else:
            log.error("Error (1) converting pdf to image type: " + filename + "  Exception type: " + str(type(e)), exc_info=True)

        try:
            if G.ALLOW_SYSTEM_CALL_PDF_CONVERSION:
                #try pdftoppm or convert
                if (systemcall == "pdftoppm") and (which("pdftoppm") is not None):
                    try:
                        log.info("Attempting blind system call to pdftoppm to convert ... ")
                        os.system("pdftoppm %s %s -png -singlefile" % (filename, filename.rstrip(".pdf")))
                        #alternate call for wrangler
                        #base is really 150 dpi but have to set convert to 200 to mimic resolution
                        os.system("convert -density 200 %s %s" % (filename, filename.rstrip(".pdf")+".png"))
                        log.info("No immediate error reported on pdftoppm call ... ")
                    except Exception as e:
                        if type(e) is pdf2image.exceptions.PDFInfoNotInstalledError:
                            log.error("System call conversion failed (PDFInfoNotInstalledError).", exc_info=False)
                        else:
                            log.error("System call conversion failed.",exc_info=True)
                elif which ("convert") is not None:
                    try:
                        log.info("Attempting blind system call to convert ... ")
                        # alternate call for wrangler
                        # base is really 150 dpi but have to set convert to 200 to mimic resolution
                        os.system("convert -density 200 %s %s" % (filename, filename.rstrip(".pdf") + ".png"))
                        log.info("No immediate error reported on convert call ... ")
                    except Exception as e:
                        log.error("System call conversion failed.", exc_info=True)
                else:
                    log.error("No viable system call available to convert PDF to PNG")
        except:
            log.error("System call (pdftoppm) conversion failed.", exc_info=True)
        return 0

    return 0



def get_hdf5_detectids_by_coord(hdf5,ra,dec,error,shotid=None,wave=None,sort=False):
    """
    Find all detections within +/- error from given ra and dec

    :param ra: decimal degrees
    :param dec:  decimal degress
    :param error:  decimal degrees
    :return:
    """

    detectids = []
    ras = []
    decs = []
    dists = []
    try:
        if (hdf5 is None) or (not os.path.exists(hdf5)):
            log.error(f"File not found: {hdf5}")
            if sort:
                return detectids, ras, decs, dists
            else:
                return detectids

        log.info("Searching for records by RA, Dec + error (this may take a while) ... ")
        dec_correction = np.cos(np.deg2rad(dec))

        #get the shots first
        #shot FOV diameter 18' (telecope is 22') ... just to be safe use the larger
        if shotid is None or ((isinstance(shotid,list) or isinstance(shotid,np.ndarray)) and len(shotid)==0) or shotid ==0:
            with tables.open_file(G.HDF5_SURVEY_FN, mode="r") as h5: #survey
                stb = h5.root.Survey
                fov = 0.1834 #22arcmin diameter to 11' radius x 60 arcsec / 3600 to get degree
                ra1 = ra - fov/dec_correction
                ra2 = ra + fov/dec_correction
                dec1 = dec - fov
                dec2 = dec + fov
                log.debug("Reading for shotids ...")
                shotlist = stb.read_where("(ra > ra1) & (ra < ra2) & (dec > dec1) & (dec < dec2)",field="shotid")
                if shotlist is not None:
                    log.info(f"Shots found ({len(shotlist)}): {shotlist}")
                else:
                    log.info(f"Shots found: 0")
        else:
            if isinstance(shotid,list) or isinstance(shotid,np.ndarray):
                shotlist = shotid
            else:
                shotlist = [shotid]

        with tables.open_file(hdf5, mode="r") as h5:
            dtb = h5.root.Detections
            ra1 = ra - error/dec_correction
            ra2 = ra + error/dec_correction
            dec1 = dec - error
            dec2 = dec + error

            rows = None
            for q_shot in shotlist: #probably not indexed by shot
                if wave is not None and wave != 0 and G.SEARCH_DELTA_WAVELENGTH > 0:
                    q_wave1 = wave - G.SEARCH_DELTA_WAVELENGTH
                    q_wave2 = wave + G.SEARCH_DELTA_WAVELENGTH
                    q_rows = dtb.read_where("(shotid == q_shot) & (ra > ra1) & (ra < ra2) & (dec > dec1) & (dec < dec2) & (wave >= q_wave1) & (wave <= q_wave2)")
                else:
                    q_rows = dtb.read_where("(shotid == q_shot) & (ra > ra1) & (ra < ra2) & (dec > dec1) & (dec < dec2)")
                if q_rows is not None and len(q_rows) > 0:
                    if rows is None:
                        rows = q_rows
                    else:
                        rows = np.concatenate((rows,q_rows))


            if (rows is not None) and (len(rows) > 0):
                detectids = rows['detectid']

                #less important, sort by distance
                if sort:
                    try:
                        ras = rows['ra']
                        decs = rows['dec']
                        dist = [UTIL.angular_distance(ra,dec,r,d) for r,d in zip(ras,decs)]

                        all = sorted(zip(dist,detectids,ras,decs))
                        dists = np.array([x for x,_,_,_ in all])
                        detectids = np.array([x for _,x,_,_ in all])
                        ras = np.array([x for _,_,x,_ in all])
                        decs = np.array([x for _,_,_,x in all])

                        #trim those at corners that are actually out of range
                        sel = np.where(dists <= (error*3600.0))
                        dists=dists[sel]
                        detectids=detectids[sel]
                        ras=ras[sel]
                        decs=decs[sel]

                        # detectids = [d for _,d in sorted(zip(dist, detectids))]
                        # ras = [d for _,d in sorted(zip(dist, ras))]
                        # decs = [d for _,d in sorted(zip(dist, decs))]
                    except:
                        log.debug("Unable to sort by distance",exc_info=True)

                msg = "%d detection records found +/- %g\" from %f, %f (%s)" % (len(detectids), error * 3600., ra, dec, hdf5)
                log.info(msg)
                print(msg)
                log.info(f"DetectIDs: {detectids}")
            else:
                msg = "0 detection records found +/- %g\" from %f, %f (%s)" % (error * 3600., ra, dec,hdf5)
                log.info(msg)
                print(msg)


    except:
        log.error(f"Exception in elixer.py get_hdf5_detectids_by_coord",exc_info=True)

    if sort:
        return detectids,ras,decs,dists
    else:
        return detectids




def get_hdf5_detectids_to_process(args,as_rows=False):
    """
    returns a list of detectids (Int64) to process

    This is built from the supplied HDF5 detection file and dets list, translated to detectIDs

    :param args:
    :return:
    """

    if args.hdf5 is None:
        return []

    #check that the file exists
    try:
        if not os.path.isfile(args.hdf5):
            msg = "Fatal. Supplied HDF5 file does not exist (%s)" % (args.hdf5)
            print(msg)
            log.error(msg)
            exit(-1)
    except:
        msg = "Fatal. Supplied HDF5 file does not exist (%s)" % (args.hdf5)
        print(msg)
        log.error(msg,exc_info=True)
        exit(-1)

    detectids = []
    detlist = None
    check_for_numeric = False

    #special case, dispatch mode where dispatch has coords BUT this is not a re-extraction
    #so, we want to assign the dispatch file to the coords file and run as a search
    try:
        if (args.dispatch is not None): #applies ONLY to DISPATCH mode
            if (args.aperture is None) and (args.coords is not None):
                #this is a coordinate search even if --search and --delta_wave are not provided
                args.coords = args.dispatch
                coord_search = True
            else:
                coord_search = False
        else:
            coord_search = None
    except:
        msg = "Fatal. Cannot execute dispatch mode coordinate search "
        print(msg)
        log.error(msg,exc_info=True)
        exit(-1)

    try:
        if args.dispatch is not None and coord_search is False:  # from multi-task SLURM only
            try:
                # is this a list or a file
                if os.path.isfile(args.dispatch):
                    if args.aperture:  # this is an extraction
                        detlist = [] #will be a list of lists
                        with open(args.dispatch) as f:
                            for line in f:  #these COULD be ra dec shot wave OR just a detectid
                                try:
                                    toks = line.split()
                                    if toks is None or len(toks) == 0:
                                        log.error("Invalid dispatch line.")
                                        continue
                                    elif len(toks)==1: #this is probably just a detectid
                                        try:
                                            did = np.int64(toks[0])
                                            detlist.append(did)
                                        except:
                                            log.error(f"Invalid --coords / --dets file line format: {line}")
                                    else: #this is a set of coords
                                        local_ra, local_dec = UTIL.coord2deg(str(toks[0]) + " " + str(toks[1]))
                                        if local_ra is not None and local_dec is not None:
                                            row = [local_ra, local_dec]
                                        else:
                                            row = [float(toks[0]),float(toks[1])]
                                        if len(toks) >= 3: #shotid or wave and might have string in it
                                            # shot = toks[2].lower()
                                            # row.append(int(float(shot.replace('v',''))))
                                            row.append(xlat_shotid(toks[2]))
                                            if len(toks) >= 4: #wavelength (in AA) or shotid
                                                #row.append(float(toks[3]))
                                                row.append(xlat_shotid(toks[3]))

                                            if len(toks) >= 5: #name/detectid (as integer)
                                                row.append(int(toks[4]))

                                            #if shotid and wavelength are flipped, it should be caught later?
                                        else:
                                            row.append(None)

                                        detlist.append(row)
                                except:
                                    log.error(f"Invalid --coords file line format: {line}")

                        return detlist
                    elif G.python2():
                        detlist = np.genfromtxt(args.dispatch, dtype=None, comments='#', usecols=(0,))
                    else:
                        detlist = np.genfromtxt(args.dispatch, dtype=None, comments='#', usecols=(0,),encoding=None)

                    log.debug("[dispatch] Loaded as file")
                else:
                    detlist = args.dispatch.replace(', ', ',').split(',')  # allow comma or comma-space separation
                    log.debug("[dispatch] Loaded --dets as list")
            except:
                log.error("Exception processing detections (--dispatch) detlist. FATAL. ", exc_info=True)
                print("Exception processing detections (--dispatch) detlist. FATAL.")
                exit(-1)

        elif args.dets is None:
            # maybe an ra and dec  OR a list of RA and Decs
            if args.coords is not None:
                if args.aperture is None and not as_rows: #then this is a list of RA, Dec to find nearest HETDEX detections
                    #need to iterate over coords
                    if args.search is not None:
                        error = args.search
                    else:
                        error = args.error

                    try:
                        ras,decs,shotids,waves,*_ = read_coords_file(args.coords,args,as_rows)
                        if shotids is None or ((isinstance(shotids, list) or isinstance(shotids, np.ndarray)) and len(shotids) == 0):
                            shotids = np.zeros(len(ras))
                        for r,d,s,w in zip (ras,decs,shotids,waves):
                            dlist = get_hdf5_detectids_by_coord(args.hdf5,r,d,error/3600.,s,w)
                            if len(dlist) > 0:
                                detectids.extend(dlist)
                    except:
                        log.error("Unable to read in --coords specified file.",exc_info=True)
                        return []

                    return detectids
                else: #these are re-extractions
                    try:
                        rows = read_coords_file(args.coords,args,as_rows=True) #shotids don't matter here
                    except:
                        log.error("Unable to read in --coords specified file.",exc_info=True)
                        return []

                    return rows

            elif (args.ra is not None) and (args.dec is not None) and ((args.error is not None) or (args.search is not None)):

                if args.aperture: #this is a re-extraction request
                    if args.shotid:
                        shot = xlat_shotid(args.shotid)
                    else:
                        shot = "0"

                    if args.wavelength:
                        wave = args.wavelength
                    else:
                        wave = "0"

                    line = str(args.ra) + " " + str(args.dec) + " " + str(shot) + " " + str(wave)
                    return [line]
                else:
                    # args.ra and dec are now guaranteed to be decimal degrees. args.error is in arcsecs
                    if args.search is not None:
                        error = args.search
                    else:
                        error = args.error

                    return get_hdf5_detectids_by_coord(args.hdf5, args.ra, args.dec, error / 3600.,args.shotid,args.wavelength)
            else:
                return []

        #dets might be a single value or a list
        if detlist is None:
            try:
                #is this a list or a file
                if os.path.isfile(args.dets):
                    if G.python2():
                        detlist = np.genfromtxt(args.dets, dtype=None,comments='#',usecols=(0,))
                    else:
                        detlist = np.genfromtxt(args.dets, dtype=None, comments='#', usecols=(0,),
                                                encoding=None)
                    detlist_is_file = True
                    check_for_numeric = True #since this is a file and these are HDF5 detectids they must be integers
                    log.debug("Loaded --dets as file")
                elif os.path.isfile(os.path.join("..",args.dets)):
                    if G.python2():
                        detlist = np.genfromtxt(os.path.join("..",args.dets), dtype=None, comments='#', usecols=(0,))
                    else:
                        detlist = np.genfromtxt(os.path.join("..", args.dets), dtype=None, comments='#',
                                                usecols=(0,),encoding=None)
                    detlist_is_file = True
                    check_for_numeric = True
                    log.debug("Loaded --dets as ../<file> ")
                else:
                    detlist = args.dets.replace(', ',',').split(',') #allow comma or comma-space separation
                    log.debug("Loaded --dets as list")
                    check_for_numeric = True

            except:
                log.error("Exception processing detections (--dets) detlist. FATAL. ", exc_info=True)
                print("Exception processing detections (--dets) detlist. FATAL.")
                h5.close()
                exit(-1)

        len_detlist = 0
        try:
            len_detlist = len(detlist)  # ok
        except:
            try:
                len_detlist = detlist.size
            except:
                pass

        if len_detlist == 1 and not isinstance(detlist,list):
            # so we get a list of strings, not just a single string and not a list of characters
            detlist = [str(detlist)]

        if check_for_numeric and len_detlist > 0:
            try:
                detlist = [np.int64(x) for x in detlist]
                # numeric = [x.isnumeric() for x in detlist]
                # if np.all(numeric):
                #    detlist = [np.int64(x) for x in detlist]
            except:
                pass


        if args.blind:
            log.info("Blindly accepting detections list")
            return detlist

        #else ... verify the entries are valid detectIDs
        h5 = tables.open_file(args.hdf5, mode="r")

        dtb = h5.root.Detections

        #iterate through --dets (might be detectID or some other format ... allow mixed)
        for d in detlist:

            #is it an int?
            #could be a string as an int
            try:
                if str(d).isdigit():
                    #this is a detectid"
                    id = int(d)
                    rows = dtb.read_where("detectid==id")
                    num = rows.size

                    if num == 0:
                        log.info("%d not found in HDF5 detections" %(id))
                        continue
                    elif num == 1:
                        log.info("%d added to detection list" %(id))
                        detectids.append(id)
                        continue
                    else:
                        log.info("%d detectid is not unique" %(id))
                        #might be something else, like 20180123 ??
                        #todo: for now, just skip it ... might consider checking something els
                        continue
            except:
                pass

            #todo: ould be a range: like 123-130 or 123:130 as slices or ranges?

            #is it an old style id ... like "20180123v009_5"
            try:
                if "v" in str(d).lower():

                    if "_" in str(d): #this is a specific input

                        #todo: this could represent a name .... could map to multiple detectIDs
                        #currently called an "inputid"
                        id  = str(d)
                        rows = dtb.read_where("inputid==id")
                        num = rows.size

                        if num == 0:
                            log.info("%s not found in HDF5 detections" % (id))
                            continue
                        elif num == 1:
                            d_id = rows['detectid'][0]
                            log.info("%s added to detection list as %d" % (id,d_id))
                            detectids.append(d_id)
                            continue
                        else:
                            log.info("%s inputid is not unique" % (id))
                            # might be something else, like 20180123 ??
                            # todo: for now, just skip it ... might consider checking something els
                            continue
                    else: #this is just a datevshot, get the entire shot
                        #assumes a 20180123v009 format
                        try:
                            toks = str(d).lower().split('v')
                            q_date = int(toks[0])
                            q_shot = int(toks[0]+toks[1])

                            rows = dtb.read_where("(date==q_date) & (shotid==q_shot)")

                            if rows is not None:
                                for row in rows:
                                    detectids.append(row['detectid'])
                        except:
                            log.error("Invalid detection identifier: %s" %d)

            except:
                pass

        h5.close()

    except:
        msg = "Fatal. Could not consume HDF5 file."
        print(msg)
        log.error(msg,exc_info=True)
        try:
            h5.close()
        except:
            pass
        exit(-1)

    return detectids


def read_coords_file(filename,args=None,as_rows=False):
    """
    try to read in a coords file (of form  ra dec shotid
    for future use, may allow shotid to be '0'
    (for nearest HETDEX match, shotid is ignored)
    """

    #could be at this cwd or up one level (if this is from a SLURM call)

    if os.path.isfile(filename):
        pass
    elif os.path.isfile(os.path.join("..", filename)):
        filename = os.path.join("..", filename)
    else:
        #filename does not exist (where expected)
        print(f"Unable to find {filename} .")
        log.error(f"Unable to find {filename} .")
        if as_rows:
            return []
        else:
            return [], [], [], [], []

    ras, decs, shots, waves, rows, names = [],[],[],[],[],[]
    try:
        if (args is not None) and (args.aperture is None) and (args.search is None):
            if as_rows:
                rows = np.loadtxt(filename, unpack=False, usecols=(0, 1))  # ignore shotids
            else:
                ras, decs = np.loadtxt(filename, unpack=True,usecols=(0,1)) #ignore shotids
                shots = np.zeros(len(ras))
                waves = np.zeros(len(ras))
                names = np.zeros(len(ras))
        else:

            #     #rows = open(filename, "r").read().splitlines()
            if as_rows:
                rows = np.loadtxt(filename, unpack=False,dtype=str)
            #     try:
            #         if np.shape(rows)[1] > 2:
            #             rows[:,2] = xlat_shotid(rows[:,2])
            #     except:
            #         pass
            #
            else:

                try: #5 values
                    ras, decs, shots, waves,names = np.loadtxt(filename, unpack=True)
                    shots = xlat_shotid(shots)
                except:
                    try: #4 values
                        ras, decs, shots, waves = np.loadtxt(filename, unpack=True)
                        shots = xlat_shotid(shots)
                    except:
                        try: #3 values
                            ras, decs, shots = np.loadtxt(filename, unpack=True)
                            shots = xlat_shotid(shots)
                            waves = np.zeros(len(ras))
                        except: #2 values
                            ras, decs = np.loadtxt(filename, unpack=True)
                            shots = np.zeros(len(ras))
                            waves = np.zeros(len(ras))
            # if as_rows: #this doest NOT work the way I want ... produces a string that has [,,,,]
            #     #yes this is a weird way to do this
            #     #but numpy forces its arrays to all be of the same type and we prefer the shots to be integers
            #     rows = [str(list(a)) for a in zip(ras,decs,shots,waves)]
    except:
        log.error("Unable to read in --coords specified file.", exc_info=True)
        if as_rows:
            return []
        else:
            return [],[],[],[],[]

    if as_rows:
        #todo: if not 2d array, wrap as 2d array
        try:
            if len(np.shape(rows)) == 1:
                rows = [rows]
        except:
            pass
        return rows
    else:
        return ras, decs, shots, waves, names



def get_fcsdir_subdirs_to_process(args):
#return list of rsp1 style directories to process
    if (args.fcsdir is None) or (args.line is not None): #if a line file was provided, we will use it instead of this
        return []

    fcsdir = args.fcsdir
    detlist_is_file = False
    detlist = [] #list of detections in 20170322v011_005 format
    subdirs = [] #list of specific rsp1 subdirectories to process (output)

    #print(os.getcwd())

    if args.dispatch is not None: #from multi-task SLURM only
        try:
            # is this a list or a file
            if os.path.isfile(args.dispatch):
                if G.python2():
                    detlist = np.genfromtxt(args.dispatch, dtype=None, comments='#', usecols=(0,))
                else:
                    detlist = np.genfromtxt(args.dispatch, dtype=None, comments='#', usecols=(0,),encoding=None)
                log.debug("[dispatch] Loaded --dets as file")
            else:
                detlist = args.dispatch.replace(', ', ',').split(',')  # allow comma or comma-space separation
                log.debug("[dispatch] Loaded --dets as list")
        except:
            log.error("Exception processing detections (--dispatch) detlist. FATAL. ", exc_info=True)
            print("Exception processing detections (--dispatch) detlist. FATAL.")
            exit(-1)

    elif args.dets is not None:
        try:
            #is this a list or a file
            if os.path.isfile(args.dets):
                if G.python2():
                    detlist = np.genfromtxt(args.dets, dtype=None,comments='#',usecols=(0,))
                else:
                    detlist = np.genfromtxt(args.dets, dtype=None, comments='#', usecols=(0,),encoding=None)
                detlist_is_file = True
                log.debug("Loaded --dets as file")
            elif os.path.isfile(os.path.join("..",args.dets)):
                if G.python2():
                    detlist = np.genfromtxt(os.path.join("..",args.dets), dtype=None, comments='#', usecols=(0,))
                else:
                    detlist = np.genfromtxt(os.path.join("..", args.dets), dtype=None, comments='#', usecols=(0,),encoding=None)
                detlist_is_file = True
                log.debug("Loaded --dets as ../<file> ")
            else:
                detlist = args.dets.replace(', ',',').split(',') #allow comma or comma-space separation
                log.debug("Loaded --dets as list")
        except:
            log.error("Exception processing detections (--dets) detlist. FATAL. ", exc_info=True)
            print("Exception processing detections (--dets) detlist. FATAL.")
            exit(-1)

    try:
        if not os.path.isdir(fcsdir):
            log.error("Cannot process flux calibrated spectra directory: " + str(fcsdir))

        #scan subdirs for detections ... assume an rsp1 style format
        #look for "list2" file or "*spec.dat" files

        #pattern = "*spec.dat" #using one of the expected files in subdir not likely to be found elsewhere
        #assumes no naming convention for the directory names or intermediate structure in the tree

        len_detlist = 0
        try:
            len_detlist = len(detlist) #ok
        except:
            try:
                len_detlist = detlist.size
            except:
                pass

        if len_detlist == 1 and not isinstance(detlist,list): #so we get a list of strings, not just a single string and not a list of characters
            detlist = [str(detlist)]

        #if (detlist is None) or (len(detlist) == 0):
        if len_detlist == 0:
            for root, dirs, files in os.walk(fcsdir):
                pattern = os.path.basename(root)+"specf.dat" #ie. 20170322v011_005spec.dat
                for name in files:
                    if name == pattern:
                    #if fnmatch.fnmatch(name, pattern):
                        subdirs.append(root)
                        print("Adding: %s" % root)
                        log.debug("Adding %s" % root)
                        break #stop looking at names in THIS dir and go to next
        else:
            #second check, first version of Karl's headers had no comment
            if detlist[0] == 'name':
                detlist = detlist[1:]

            use_search = False
            log.debug("Attempting FAST search for %d directories ..." %len_detlist)
            #try fast way first (assume detlist is immediate subdirectory name, if that does not work, use slow method
            for d in detlist:

                try:
                    if d[-1] == "/":
                        d = d.rstrip("/")
                except:
                    pass

                if args.annulus:
                    pattern = d + ".spsum"
                else:
                    pattern = d + "specf.dat"
                if os.path.isfile(os.path.join(fcsdir,d,pattern)) or \
                    os.path.isfile(os.path.join(fcsdir, d, os.path.basename(d) + "specf.dat")):
                    subdirs.append(os.path.join(fcsdir,d))
                    log.debug("Adding %s" % os.path.join(fcsdir,d))
                elif os.path.isfile(os.path.join(d,os.path.basename(d)+"specf.dat")):
                    subdirs.append(d)
                    log.debug("Adding %s" % d)
                else:
                    if ("*" in d) or ("?" in d):
                        #this is a wildcard
                        log.debug("Proceeding with wildcard (--fcsdir) search ...")
                    else:
                        log.debug("FAILED to find %s. Will proceed with full search ...." % d)
                        print("*** FAILED: %s" % d) #but keep going

                    #fail, fast method will not work
                    #if any fail, all fail?
                    # log.debug("FAST search fail ... ")
                    # del subdirs[:]
                    # use_search = True
                    # break

            if len(subdirs) == 0:
                log.debug("Fast search method failed. Detlist does not match to subdirectories. Attempting full search...")

                if type(detlist) is str:
                    if '*' in detlist or "?" in detlist:
                        use_search = True
                elif type(detlist) is list:
                    for d in detlist:
                        if '*' in d or "?" in d:
                            use_search = True
                            break

                if use_search:
                    log.debug("Searching fcsdir for matching subdirs ...")
                    if detlist_is_file: #this was a file provided with a list of directories but totally failed, so
                                        #treat like there was no detlist provided at all
                        for root, dirs, files in os.walk(fcsdir):
                            if args.annulus:
                                pattern = os.path.basename(root) + ".spsum"  # ie. 20170322v011_005spec.dat
                            else:
                                pattern = os.path.basename(root) + "specf.dat"  # ie. 20170322v011_005spec.dat
                            for name in files:
                                if name == pattern:
                                    # if fnmatch.fnmatch(name, pattern):
                                    subdirs.append(root)
                                    print("Adding: %s" % root)
                                    log.debug("Adding %s" % root)
                                    break  # stop looking at names in THIS dir and go to next
                    else: #here, detlist was a short command line list or something with wildcards
                        for root, dirs, files in os.walk(fcsdir):
                            #this is a little different in that detlist might be populated with a wildcard pattern
                            if args.annulus:
                                patterns = [x + ".spsum" for x in detlist]  # ie. 20170322v011_005spec.dat
                            else:
                                patterns = [x + "specf.dat" for x in detlist] #ie. 20170322v011_005spec.dat
                            for name in files:
                                #if name in patterns:
                                for p in patterns:
                                    if fnmatch.fnmatch(name, p): #could have wild card
                                        subdirs.append(root)
                                        print("Adding: %s" % root)
                                        log.debug("Adding %s" %root)
                                        break #stop looking at names in THIS dir and go to next
    except:
        log.error("Exception attempting to process --fcsdir. FATAL.",exc_info=True)
        print("Exception attempting to process --fcsdir. FATAL.")
        exit(-1)

    return subdirs

def upgrade_hdf5(args=None):

    try:
        #tokenize
        toks = args.upgrade_hdf5.split(',')
        if len(toks) != 2:
            print("Invalid parameters for upgrade_hdf5 (%s)"%args.upgrade_hdf5)
            log.error("Invalid parameters for upgrade_hdf5 (%s)"%args.upgrade_hdf5)
            return

        result = elixer_hdf5.upgrade_hdf5(toks[0],toks[1])

        if result:
            print("Success. File = %s" %toks[1])
        else:
            print("FAIL.")
    except:
        log.error("Exception! upgrading HDF5 file in upgrade_hdf5",exc_info=True)


def remove_h5_duplicate_rows(args=None):
    try:
        result = elixer_hdf5.remove_duplicates(args.remove_duplicates)
    except:
        log.error("Exception! merging HDF5 files in merge_unique", exc_info=True)


def merge_unique(args=None):
    """
    note: elixer_hdf5 will chunk the h5 files into small blocks and merge the blocks (much more efficient)
    :param args:
    :return:
    """

    try:
        #tokenize
        toks = args.merge_unique.split(',')
        if len(toks) != 3:
            print("Invalid parameters for merge_unique (%s)"%args.merge_unique)
            log.error("Invalid parameters for merge_unique (%s)"%args.merge_unique)
            return

        result = elixer_hdf5.merge_unique(toks[0],toks[1],toks[2])

        if result:
            print("Success. File = %s" %toks[0])
        else:
            print("FAIL.")
    except:
        log.error("Exception! merging HDF5 files in merge_unique",exc_info=True)


def merge_hdf5(args=None):
    """
    Similar to merge ... replaces merge ... joins ELiXer HDF5 catlogs.
    Does not check for duplicate entries.
    :param args:
    :return:
    """

    try:
        merge_fn = "elixer_merged_cat.h5"
        fn_list = sorted(glob.glob("*_cat.h5"))
        fn_list += sorted(glob.glob("*_cat_*.h5"))
        fn_list += sorted(glob.glob("dispatch_*/*/*_cat.h5"))

        #fn_list.append(sorted(glob.glob("dispatch_*/*/*_cat.h5")))
        if len(fn_list) != 0:
            merge_fn = elixer_hdf5.merge_elixer_hdf5_files(merge_fn,fn_list)
            if merge_fn is not None:
                print("Done: " + merge_fn)
            else:
                print("Failed to write HDF5 catalog.")
        else:
            print("No HDF5 catalog files found. Are you in the directory with the dispatch_* subdirs?")
    except Exception as e:
        print(e)
        log.error("Exception! merging HDF5 files.",exc_info=True)

def merge(args=None):

    merge_hdf5(args)
    #todo: discontinue use of _cat.txt and _fib.txt ?

    #must be in directory with all the dispatch_xxx folders
    #for each folder, read in and join the  dispatch_xxx

    if False: #stop merging fib and cat.txt files
        #cat file first
        try:
            first = True
            merge_fn = None
            merge = None
            for fn in sorted(glob.glob("dispatch_*/*/*_cat.txt")):
                if first:
                    first = False
                    merge_fn = os.path.basename(fn)
                    merge = open(merge_fn, 'w')

                    with open(fn,'r') as f:
                        merge.writelines(l for l in f)
                else:
                    with open(fn,'r') as f:
                        merge.writelines(l for l in f if l[0] != '#')

            if merge is not None:
                merge.close()
                print("Done: " + merge_fn)
            else:
                print("No catalog files found. Are you in the directory with the dispatch_* subdirs?")

        except:
            pass

        #fib file
        try:
            first = True
            merge_fn = None
            merge = None
            for fn in sorted(glob.glob("dispatch_*/*/*_fib.txt")):
                if first:
                    first = False
                    merge_fn = os.path.basename(fn)
                    merge = open(merge_fn, 'w')

                    with open(fn,'r') as f:
                        merge.writelines(l for l in f)
                else:
                    with open(fn,'r') as f:
                        merge.writelines(l for l in f if l[0] != '#')

            if merge is not None:
                merge.close()
                print("Done: " + merge_fn)
            else:
                print("No fiber files found. Are you in the directory with the dispatch_* subdirs?")

        except:
            pass


def prune_detection_list(args,fcsdir_list=None,hdf5_detectid_list=None):
    #find the pdf reports that should correspond to each detction
    #if found that detection is already done, so remove it from the list

    #!! note: the h5 is written before the report files, so if we got far enough
    # to write out a report file, then the h5 is complete and no need to re-check it.

    #implement as build a new list ... just easier that way
    if ((fcsdir_list is None) or (len(fcsdir_list) == 0)) and \
            ((hdf5_detectid_list is None) or (len(hdf5_detectid_list) == 0)):
        msg = "Unexpected empty lists. Cannot run in recovery mode."
        log.error(msg)
        print(msg)
        return None

    newlist = []
    if args.neighborhood_only:
        extension = "nei.png"
    else:
        extension = ".pdf"
    if (hdf5_detectid_list is not None) and (len(hdf5_detectid_list) > 0):

        for d in hdf5_detectid_list:
            try:
                #does the file exist
                #todo: this should be made common code, so naming is consistent

                #this might be an ra,dec
                try:
                    # d is a list or a single value here
                    _ = d[0]  # if d is an array, set any None's to '0'
                    #might have a shot ID
                    try:
                        #shotid could be d[2] or d[3]
                        manual_id = None
                        if len(d) == 3:
                            shotid = xlat_shotid(d[2])
                            if (shotid is not None) and (3400. < shotid < 5700.):  # assume this is really a wavelength
                                #this is a wavelength, so there is no shot
                                shotid = None
                        elif len(d) == 4:
                            shotid = xlat_shotid(d[2])
                            if (shotid is not None) and (3400. < shotid < 5700.):  # assume this is really a wavelength
                                # this is a wavelength, so there is no shot
                                shotid = xlat_shotid(d[3])
                        elif len(d) == 5: #the last field is the ID to use
                            manual_id = d[4]
                        else:
                            shotid = None

                        if manual_id is None:
                            filename = str(UTIL.id_from_coord(d[0],d[1]),shotid) #need to keep the original d aroound
                            if shotid is None:
                                filename = filename[0:-5] + "?????" #strip the yy + day + shot part and replace with wildcard
                        else:
                            filename = str(manual_id)
                    except:
                        #no shotid
                        filename = str(UTIL.id_from_coord(d[0],d[1])) #need to keep the original d around
                        filename = filename[0:-5] + "?????"  # strip the yy + day + shot part and replace with wildcard
                except:
                    # d is not an array, but need to make it one, this is probably a detectID
                    filename = str(d) #assume a normal hetdex ID

                okay_to_skip = False
                if os.path.isfile(os.path.join(args.name, args.name + "_" + filename + extension)) or \
                    os.path.isfile(os.path.join(args.name, filename + extension)):

                    okay_to_skip = True
                    if not args.neighborhood_only:
                        #do we need the png or the neighborhood?
                        if args.png:
                            #if not os.path.isfile(os.path.join(args.name, filename + ".png")):
                            if len(glob.glob(os.path.join(args.name, filename + ".png"))) == 0:
                                okay_to_skip = False

                        if okay_to_skip and args.neighborhood:
                            #if not os.path.isfile(os.path.join(args.name, filename + "_nei.png")):
                            if len(glob.glob(os.path.join(args.name, filename + "_nei.png"))) == 0:
                                okay_to_skip = False

                        if okay_to_skip and args.mini:
                           # if not os.path.isfile(os.path.join(args.name, filename + "_mini.png")):
                            if len(glob.glob(os.path.join(args.name, filename + "_mini.png"))) == 0:
                                okay_to_skip = False

                    if okay_to_skip:
                        log.info("Already processed %s. Will skip recovery." %(filename))
                    else:
                        log.info("Not all components found (%s). Will process ..." %(filename))
                        newlist.append(d)
                else:
                    log.info("Not found (%s). Will process ..." %(filename))
                    newlist.append(d)
            except:
                log.warning(f"Exception in prune_detection_list. Will re-run {d}",exc_info=True)
                newlist.append(d)

    else: #this is the fcsdir list
        for d in fcsdir_list:
            try:
                filename = os.path.basename(str(d))
                if os.path.isfile(os.path.join(args.name, args.name + "_" + filename + extension)) or \
                    os.path.isfile(os.path.join(args.name, filename + extension)):

                    okay_to_skip = True
                    if not args.neighborhood_only:
                        #do we need the png or the neighborhood?
                        if args.png:
                            if not os.path.isfile(os.path.join(args.name, filename + ".png")):
                                okay_to_skip = False

                        if okay_to_skip and args.neighborhood:
                            if not os.path.isfile(os.path.join(args.name, filename + "_nei.png")):
                                okay_to_skip = False

                        if okay_to_skip and args.mini:
                            if not os.path.isfile(os.path.join(args.name, filename + "_mini.png")):
                                okay_to_skip = False

                    if okay_to_skip:
                        log.info("Already processed %s. Will skip recovery." %(filename))
                    else:
                        log.info("Not all components found (%s). Will process ..." %(filename))
                        newlist.append(d)
                else:
                    log.info("Not found (%s). Will process ..." %(filename))
                    newlist.append(d)
            except:
                log.warning(f"Exception in prune_detection_list. Will re-run {d}",exc_info=True)
                newlist.append(d)

    return newlist



def build_3panel_zoo_image(fname, image_2d_fiber, image_1d_fit, image_cutout_fiber_pos,
                           image_cutout_neighborhood,
                           image_cutout_fiber_pos_size=None,
                           image_cutout_neighborhood_size=None,
                           line_image_cutout=None):
    """
    Note: needs matplotlib > 3.1.x to work properly

    :param fname:
    :param image_2d_fiber:
    :param image_1d_fit:
    :param image_cutout_fiber_pos:
    :return:
    """

    if (fname is None) or (image_2d_fiber is None) or (image_1d_fit is None) or (image_cutout_fiber_pos is None):
        msg = "Missing required data in elixer::build_3panel_zoo_image() "
        if fname is None:
            msg += "fname "
        if image_2d_fiber is None:
            msg += " image_2d_fiber "
        if image_1d_fit is None:
            msg += " image_1d_fit "
        if image_cutout_fiber_pos is None:
            msg += " image_cutout_fiber_pos "
        if line_image_cutout is None:
            msg += " line_image_cutout "
        log.error(msg)
        return
    try:
        #note: 0,0 (for grid spec) is upper left
        # first slice is the vertical positioning (#rows), second is the horizontal (#cols)

        #scaling for the fiber POS cutout and neighborhood cutout
        box_ratio = 0.0
        if (image_cutout_fiber_pos_size is not None) and (image_cutout_neighborhood_size is not None) and \
           (image_cutout_fiber_pos_size > 0) and (image_cutout_neighborhood_size > 0):
            box_ratio = float(image_cutout_fiber_pos_size) / float(image_cutout_neighborhood_size)
            #todo: make a box at the center box_ratio * size of neighborhood image



        #update pos cutout with outline
        #this just does not seem to work right
        # image_cutout_fiber_pos_frame = None
        # try:
        #     plt.close('all')
        #     plt.figure()
        #     plt.gca().set_axis_off()
        #     image_cutout_fiber_pos.seek(0)
        #     plt.imshow(PIL_Image.open(image_cutout_fiber_pos))
        #
        #     #add window outline
        #     scale_ratio = 0.75
        #     xl,xr = plt.gca().get_xlim()
        #     yb,yt = plt.gca().get_ylim()
        #     zero_x = (xl + xr) / 2.
        #     zero_y = (yb + yt) / 2.
        #     rx = (xr - xl) * scale_ratio / 2.0
        #     ry = (yt - yb) * scale_ratio / 2.0
        #
        #     plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2, height=ry * 2,
        #                                       angle=0, color='red', fill=False))
        #     image_cutout_fiber_pos_frame = io.BytesIO()
        #     plt.savefig(image_cutout_fiber_pos_frame, format='png', dpi=FIG_DPI, transparent=True)
        #     plt.close('all')
        # except:
        #     log.debug("Exception! adding outline box to fiber POS plot in for mini report.",exc_info=True)
        #



        #plot with gridspec
        #background color requested
        #HEX: 1C1C1E
        #RGB: 28, 28, 30
        #RGBA: 26, 28, 30, 1
        fig = plt.figure(facecolor='#1c1c1e',constrained_layout=False)#,figsize=(2,3)) #x,y or cols, rows
        plt.subplots_adjust(wspace=0, hspace=0)
        #gs1 = fig.add_gridspec(ncols=1,nrows=1,figure=fig,left=0.02, right=0.30, top=0.90, bottom=0.30, wspace=0.01)
        #plt.subplots_adjust(left=0.00, right=0.95, top=0.95, bottom=0.0)

        gs = gridspec.GridSpec(nrows=61,ncols=21,figure=fig,wspace=0.0,hspace=0.0)  # rows, columns or y,x



        # is hidden
        # 1st column 2d fiber cutouts
        ax1 = fig.add_subplot(gs[2:45,0:10]) #was 2
        ax1.set_axis_off()
        # fig = plt.subplot(gs[0:-1,0:30])#,gridspec_kw = {'wspace':0, 'hspace':0})
        # plt.subplots_adjust(wspace=0, hspace=0)

        # ax1.axis('off')
        # ax1.axes.get_xaxis().set_visible(False)
        # ax1.axes.get_yaxis().set_visible(False)

        image_2d_fiber.seek(0)
        im1 = PIL_Image.open(image_2d_fiber)
        ax1.imshow(im1)

        #gs2 = fig.add_gridspec(ncols=1, nrows=2, figure=fig, left=0.35, right=0.98,top=0.90, bottom=0.30, wspace=0.01)

        #2d fiber position / master cutout
        #ax2 = fig.add_subplot(gs[0:26,0:])
        ax2 = fig.add_subplot(gs[0:23,0:])
        ax2.set_axis_off()
        #fig = plt.subplot(gs[0:50,30:-1])#,gridspec_kw = {'wspace':0, 'hspace':0})
        #plt.subplots_adjust(wspace=0, hspace=0)

        #ax2.axis('off')
        #ax2.axes.get_xaxis().set_visible(False)
        #ax2.axes.get_yaxis().set_visible(False)

        # if image_cutout_fiber_pos_frame is not None:
        #     image_cutout_fiber_pos_frame.seek(0)
        #     im2 = PIL_Image.open(image_cutout_fiber_pos_frame)
        #     ax2.imshow(im2)
        # else:
        #     image_cutout_fiber_pos.seek(0)
        #     im2 = PIL_Image.open(image_cutout_fiber_pos)
        #     ax2.imshow(im2)

        image_cutout_fiber_pos.seek(0)
        im2 = PIL_Image.open(image_cutout_fiber_pos)
        ax2.imshow(im2)

        # 2D wide (neighborhood) master cutout
        if image_cutout_neighborhood is not None:

            ax4 = fig.add_subplot(gs[19:42,0:]) #was 22:48
            ax4.set_axis_off()
            # fig = plt.subplot(gs[0:50,30:-1])#,gridspec_kw = {'wspace':0, 'hspace':0})
            # plt.subplots_adjust(wspace=0, hspace=0)

            # ax2.axis('off')
            # ax2.axes.get_xaxis().set_visible(False)
            # ax2.axes.get_yaxis().set_visible(False)

            image_cutout_neighborhood.seek(0)
            im4 = PIL_Image.open(image_cutout_neighborhood)
            ax4.imshow(im4)

            if False:
                zero_x = (ax4.get_xlim()[0] + ax4.get_xlim()[1])/2.
                zero_y = (ax4.get_ylim()[0] + ax4.get_ylim()[1]) / 2.

                rx = (zero_x * box_ratio) / 2.0
                ry = (zero_y * box_ratio) / 2.0
                half_side_x = rx
                half_side_y = ry

                ax4.add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=half_side_x * 2, height=half_side_y * 2,
                                                   angle=0, color='blue', fill=False))

                #ax4.add_patch(plt.Rectangle((zero_x - rx , zero_y + ry), width=100.0, height=100.0,
                #                        angle=0, color='red', fill=False))

        else:
            log.info("Warning! Unable to fully populate mini report. Neighborhood cutout is None.")

        #new line image
        try:
            if line_image_cutout is not None:
                ax5 = fig.add_subplot(gs[38:,0:])#,gridspec_kw = {'wspace':0, 'hspace':0})
                ax5.set_axis_off()
                line_image_cutout.seek(0)

                #leaving here just for a reference ...
                #failed attempt to grab a larger image (by root(2)), apply the WCS transform, then here
                #trim off 1/2 of each of the extra from each side and plot
                # (this does not work) ... scaling gets very messed up
                # x,y = PIL_Image.open(line_image_cutout).size
                # trim = (1.0 - 0.5 * np.sqrt(2.))/2.0
                # lx = trim * x
                # rx = trim * x
                # ty = trim * y #was 0.1
                # by = trim * y #was 0
                # im5 = PIL_Image.open(line_image_cutout).crop((lx,ty,x-rx,y-(ty+by)))

                im5 = PIL_Image.open(line_image_cutout)
                ax5.imshow(im5)
        except:
            pass #sometimes the line_image call fails and there is nothing in it



        #1d Gaussian fit
        #ax3 = fig.add_subplot(gs[46:,1:15])#,gridspec_kw = {'wspace':0, 'hspace':0})
        ax3 = fig.add_subplot(gs[46:60,1:9])#,gridspec_kw = {'wspace':0, 'hspace':0})
        ax3.set_axis_off()
        #plt.subplots_adjust(wspace=0, hspace=0)

        #ax3.axis('off')
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        image_1d_fit.seek(0)

        x,y = PIL_Image.open(image_1d_fit).size
        # lx = 0.10 * x
        # rx = 0.03 * x
        # ty = 0.10 * y #was 0.1
        # by = 0.00 * y #was 0
        lx = 0.18 * x
        rx = 0.16 * x
        ty = 0.10 * y #was 0.1
        by = 0.00 * y #was 0


        #crop is (upper left corner ....
        #( x, y, x + width , y + height )

        im3 = PIL_Image.open(image_1d_fit).crop((lx,ty,x-rx,y-(ty+by)))
        ax3.imshow(im3)


        if True: #make True if want to build image then crop it, otherwise leave as False and just save the plt
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=FIG_DPI, bbox_inches='tight', transparent=True,facecolor=fig.get_facecolor())
            plt.close()

            plt.figure()
            buf.seek(0)
            #im = PIL_Image.open(buf)

            x, y = PIL_Image.open(buf).size
            lx = 0.03 * x
            rx = 0.12 * x
            ty = 0.05 * y
            by = 0.00 * y

            im = plt.imshow(PIL_Image.open(buf).crop((lx,ty,x-(lx+rx),y-(ty+by))))

            #!!! NOTICE: if the resolution changes, or the pixel size changes, or the crop, etc, these coords
            #  will need to be re-done
            #fill in a bothersome region that refuses to be covered by the red box
            # (right hand side of top right cutout)
            #plt.plot([741, 741], [398, 30], color='r', lw=0.5)
            plt.plot([738, 738], [341, 20], color='r', lw=0.5)

            #and a horizontal masking line for the bottom right image
            #plt.plot([372, 740], [805,805], color='#1c1c1e', lw=1) ##1c1c1e

            #and a horizontal masking line for the bottom left image
           # plt.plot([0, 410], [1050,1050], color='r', lw=5) ##1c1c1e
            plt.plot([0, 400], [1050,1050], color='#1c1c1e', lw=5) ##1c1c1e


            #zoom lines ... but they depend on the distance
            if image_cutout_neighborhood_size == 10:
                #plt.scatter(0,0,s=20,color='r')
                #don't forget there is an extra border (about 25pix) around it, 0,0 is top left
                plt.plot([528, 415], [576, 340], color='r', lw=0.5,ls="--") #lower left (bottom) to lower left (top) [x,x],[y,y]
                plt.plot([625, 735], [576, 340], color='r', lw=0.5,ls="--") #lower right (bottom) to lower right (top)


            # plt.plot([510,387],[622,112],color='r',lw=1) # upper left to upper left
            # plt.plot([615,736],[622,112],color='r',lw=1) # upper right to upper right

            #plt.plot([505, 465], [622, 456], color='r', lw=0.5,ls="--") #upper left stop at top-image bottom edge
            #plt.plot([616, 659], [622, 456], color='r', lw=0.5,ls="--") #upper right "ditto"

            #patch =
            #plt.tight_layout(rect=(0.1,0.1,0.8,0.8))

            # try to trim
            # patch = patches.Rectangle((0, 0), width=1200, height=1050, angle=0.0,transform=plt.gca().transData)
            # im.set_clip_path(patch)
            plt.gca().axis('off')

            #fig.tight_layout()
            plt.savefig(fname, format='png', dpi=FIG_DPI,bbox_inches='tight',transparent=False,facecolor=fig.get_facecolor())
            log.debug("File written: %s" % (fname))
            plt.close()
        else:
            plt.savefig(fname, format='png', dpi=FIG_DPI, bbox_inches='tight', transparent=False,facecolor=fig.get_facecolor())
            log.debug("File written: %s" % (fname))
            plt.close()

        #alternate
        #cvs = PIL_Image.new('RGB',(2200,2400))


    except:
        log.error("Exception! in elixer::build_3panel_zoo_image",exc_info=True)
        #temporary
        #print("*** mini exception ***")



def build_neighborhood_map(hdf5=None,cont_hdf5=None,detectid=None,ra=None, dec=None, distance=None, cwave=None,
                           fname=None,original_distance=None,this_detection=None,broad_hdf5=None,primary_shotid=None,wave_range=None):
    """

    :param hdf5:
    :param ra:
    :param dec:
    :param distance:
    :param cwave:
    :param fname:
    :param original_distance: e.g. args.error ... the normal cutout window size
    :return: PNG of the figure
    """

    just_mini_cutout = False
    nei_buf = None
    line_buf = None
    line_buf_tight = None

    if G.ZOO_MINI:
        if ((detectid is None) and ((ra is None) or (dec is None))):
            log.info("Invalid data passed to build_neighborhood_map")
            return None, None, None

        if (distance is None) or (distance <= 0.0):
            distance = 10.0
            just_mini_cutout = True
    else:
        if (distance is None) or (distance < 0.0) or ((detectid is None) and ((ra is None) or (dec is None))):
            log.info("Invalid data passed to build_neighborhood_map")
            return None, None, None

    #get all the detectids
    error = distance/3600.0

    #use as a backup, but as of 20200718, always attempting to check ALL HETDEX detection catalogs, not
    #just what was passed in
    param_hdf5 = hdf5
    param_broad_hdf5 = broad_hdf5
    param_cont_hdf5 = cont_hdf5

    if G.HETDEX_API_CONFIG:
        try:
            hdf5 = G.HETDEX_API_CONFIG.detecth5
        except:
            hdf5 = param_hdf5
        try:
            broad_hdf5 = G.HETDEX_API_CONFIG.detectbroadh5
        except:
            broad_hdf5 = param_broad_hdf5

        try:
            cont_hdf5 = G.HETDEX_API_CONFIG.contsourceh5
        except:
            cont_hdf5 = param_cont_hdf5

    if hdf5 is None:
        hdf5 = G.HDF5_DETECT_FN

    if broad_hdf5 is None:
        broad_hdf5 = G.HDF5_BROAD_DETECT_FN

    if cont_hdf5 is None:
        cont_hdf5 = G.HDF5_CONTINUUM_FN

    #sanity check and prevent duplication
    try:
        if broad_hdf5 == hdf5:
            broad_hdf5 = None
    except:
        pass

    try:
        if cont_hdf5 == hdf5:
            cont_hdf5 = None
    except:
        pass

    if (detectid is not None) and (ra is None):
        try:
            with tables.open_file(hdf5, mode="r") as h5_detect:
                id = detectid
                detection_table = h5_detect.root.Detections
                rows = detection_table.read_where("detectid==id")
                if (rows is None) or (rows.size != 1):
                    log.info("Invalid data passed to build_neighborhood_map")
                    return None, None, None
                ra = rows['ra'][0]
                dec = rows['dec'][0]

                if cwave is None:
                    cwave = rows['wave'][0]
        except:
            log.info("Exception. Unable to lookup detectid coordinates.",exc_info=True)
            return None, None, None

    total_detectids = 0
    if not just_mini_cutout:
        neighbor_color = "red"
        detectids, ras, decs, dists = get_hdf5_detectids_by_coord(hdf5, ra=ra, dec=dec, error=error, shotid=None,
                                                                  wave=None,sort=True)

        all_ras = ras[:]
        all_decs = decs[:]

        if detectids is not None:
            total_detectids += len(detectids)

        ###########################
        #Broadline sources
        ###########################
        broad_detectids = []
        broad_ras = []
        broad_decs = []
        broad_dists = []
        if broad_hdf5 is not None:
            broad_detectids, broad_ras, broad_decs, broad_dists = get_hdf5_detectids_by_coord(broad_hdf5,ra=ra, dec=dec,
                                                                                          error=error, shotid=None,
                                                                                          wave=None,sort=True)

            if (broad_ras is not None) and (broad_decs is not None):
                np.concatenate((all_ras,broad_ras))
                np.concatenate((all_decs,broad_decs))

            if broad_detectids is not None:
                total_detectids += len(broad_detectids)

        ##########################
        # Continuum sources
        ##########################
        cont_detectids = []
        cont_ras = []
        cont_decs = []
        cont_dists = []
        if cont_hdf5 is not None:
            cont_detectids, cont_ras, cont_decs, cont_dists = get_hdf5_detectids_by_coord(cont_hdf5, ra=ra, dec=dec,
                                                                                          error=error,shotid=None,
                                                                                          wave=None,sort=True)

            if (cont_ras is not None) and (cont_decs is not None):
                np.concatenate((all_ras,cont_ras))
                np.concatenate((all_decs,cont_decs))

            if cont_detectids is not None:
                total_detectids += len(cont_detectids)



        ###############################
        #check for any sources to plot
        ###############################
        if (len(detectids) == 0) and (len(broad_detectids)== 0) and (len(cont_detectids)== 0) and (this_detection is None):
            #nothing to do
            log.info("No HETDEX detections found: (%f,%f) +/- %d\"" %(ra,dec,distance))
            return None, None, None

        # try:
        #     print("!!!!!!!!!!!!!!!!!!!!!!! REMOVE ME !!!!!!!!!!!!!!!!!!!")
        #     np.savetxt(f"{detectids[0]}.clu",detectids,fmt="%d")
        # except:
        #     pass

        if len(detectids) > G.MAX_NEIGHBORS_IN_MAP:
            msg = "Maximum number of reportable (emission line) neighbors exceeded (%d). Will truncate to nearest %d." % (len(detectids),
                                                                                                G.MAX_NEIGHBORS_IN_MAP)
            log.info(msg)
            print(msg)

            #temporary
            # for d in detectids:
            #     print(d)

            detectids = detectids[:G.MAX_NEIGHBORS_IN_MAP]
            ras = ras[:G.MAX_NEIGHBORS_IN_MAP]
            decs = decs[:G.MAX_NEIGHBORS_IN_MAP]
            dists = dists[:G.MAX_NEIGHBORS_IN_MAP]

        if len(cont_detectids) > G.MAX_NEIGHBORS_IN_MAP:
            msg = "Maximum number of reportable (continuum) neighbors exceeded (%d). Will truncate to nearest %d." % (len(detectids),
                                                                                                G.MAX_NEIGHBORS_IN_MAP)
            log.info(msg)
            print(msg)

            cont_detectids = cont_detectids[:G.MAX_NEIGHBORS_IN_MAP]
            cont_ras = cont_ras[:G.MAX_NEIGHBORS_IN_MAP]
            cont_decs = cont_decs[:G.MAX_NEIGHBORS_IN_MAP]
            cont_dists = cont_dists[:G.MAX_NEIGHBORS_IN_MAP]

        if len(broad_detectids) > G.MAX_NEIGHBORS_IN_MAP:
            msg = "Maximum number of reportable (broad) neighbors exceeded (%d). Will truncate to nearest %d." % (len(detectids),
                                                                                                G.MAX_NEIGHBORS_IN_MAP)
            log.info(msg)
            print(msg)

            broad_detectids = broad_detectids[:G.MAX_NEIGHBORS_IN_MAP]
            broad_ras = broad_ras[:G.MAX_NEIGHBORS_IN_MAP]
            broad_decs = broad_decs[:G.MAX_NEIGHBORS_IN_MAP]
            broad_dists = broad_dists[:G.MAX_NEIGHBORS_IN_MAP]


    #get the single master cutout (need to stack? or select best image (best == most pixels)?)
    cat_library = catalogs.CatalogLibrary()
    ext = distance * 1.5  # extent is from the 0,0 position AND we want to grab a bit more than the radius (so can see around it)
    cutouts = cat_library.get_cutouts(position=SkyCoord(ra, dec, unit='deg'),allow_bad_image=False,radius=ext,allow_web=True)

    if cutouts is not None:
        log.debug("Neighborhood cutouts = %d" %(len(cutouts)))
    else:
        log.info("No neighborhood cutouts returned.")

    #find the best cutout (most pixels)
    def sqpix(_cutout):
        try:
            sq = _cutout.wcs.pixel_shape[0]*_cutout.wcs.pixel_shape[1]
        except:
            sq = -1
        return sq


    sci = science_image.science_image()  # empty ... just want for functions

    def make_master(_cutouts):
        mc = None
        try:
            if len(_cutouts) > 0:
                #start with the first cutout that has the most pixels (there may be multiples with the same number of pixels)
                pix2 = np.array([sqpix(c['cutout']) for c in _cutouts])
                best = np.argmax(pix2)
                mc = copy.copy(_cutouts[best]['cutout'])

                if mc is None:
                    log.warning("No cutout (None) available for neighborhood.")
                    log.debug("Number of cutouts (%d)" %(len(_cutouts)))
                    log.debug("Square pixel counts (%s)" %(str(pix2)))
                    return None

                #iterate through all those with the same number of (best) pixels
                sel = np.where((pix2 == pix2[best]))[0]
                if len(sel) > 1:
                    # need to sum up
                    time = np.full(len(pix2),np.nan) #np.zeros(len(pix2))
                    for i in sel:
                        try:
                            try:
                                if _cutouts[i]['filter'].lower() in ['g','r','f606w']: #WHY am I not checking the unique_fration?
                                    check_unique_fraction = False
                                else:
                                    check_unique_fraction = True
                            except:
                                check_unique_fraction = False

                            if science_image.is_cutout_empty(_cutouts[i]['cutout'],check_unique_fraction=check_unique_fraction):
                                time[i] = np.nan
                                continue #skip it and move on to the next
                            time[i] = _cutouts[i]['hdu'][0]['EXPTIME']
                        except:
                            time[i] = 0.0

                    total_time = max(np.nansum(time),1.0)
                    mc.data *= 0.0

                    if total_time == 1.0:
                        for i in sel:
                            if not np.isnan(time[i]):
                                mc.data += _cutouts[i]['cutout'].data
                    else:
                        for i in sel:
                            if not np.isnan(time[i]) and time[i] > 0:
                                mc.data += _cutouts[i]['cutout'].data*time[i]/total_time
                        #time is not the best metric, but not too bad ... assumes all filters have roughly equivalent
                        #throughputs ... should not use for any measurements, but just for making a picture it is okay

        except:
            log.debug("Exception in build_neighborhood_map:",exc_info=True)

        return mc

    master_cutout = make_master(cutouts)

    if science_image.is_cutout_empty(master_cutout):
        log.info("build_neighborhood_map master_cutout is empty. Will try web calls for DECaLS, PanSTARRS, and/or SDSS.")

        #todo: maybe limit to G or R band request?
        #next code bit is sloppy but I am in a hurry and copy-paste is easy
        if G.DECALS_WEB_ALLOW:
            log.info("Calling DECaLS (web) ...")
            ps_cutouts = catalogs.cat_decals_web.DECaLS().get_cutouts(ra,dec,distance,aperture=None,filter=['g','r'],first=True)
            #note, different than cutouts above?
            mc = make_master(ps_cutouts)
            if mc is not None:
                master_cutout = mc
            elif G.PANSTARRS_ALLOW:
                log.info("Calling PanSTARRs ...")
                ps_cutouts = catalogs.cat_panstarrs.PANSTARRS().get_cutouts(ra,dec,distance,aperture=None)
                #note, different than cutouts above?
                mc = make_master(ps_cutouts)
                if mc is not None:
                    master_cutout = mc
                elif G.SDSS_ALLOW:
                    log.info("Calling SDSS ...")
                    sdss_cutouts = catalogs.cat_sdss.SDSS().get_cutouts(ra,dec,distance,aperture=None)
                    #note, different than cutouts above?
                    mc = make_master(sdss_cutouts)
                    if mc is not None:
                        master_cutout = mc
            elif G.SDSS_ALLOW:
                log.info("Calling SDSS ...")
                sdss_cutouts = catalogs.cat_sdss.SDSS().get_cutouts(ra,dec,distance,aperture=None)
                #note, different than cutouts above?
                mc = make_master(sdss_cutouts)
                if mc is not None:
                    master_cutout = mc

        elif G.PANSTARRS_ALLOW:
            log.info("Calling PanSTARRs ...")
            ps_cutouts = catalogs.cat_panstarrs.PANSTARRS().get_cutouts(ra,dec,distance,aperture=None)
            #note, different than cutouts above?
            mc = make_master(ps_cutouts)
            if mc is not None:
                master_cutout = mc
            elif G.SDSS_ALLOW:
                log.info("Calling SDSS ...")
                sdss_cutouts = catalogs.cat_sdss.SDSS().get_cutouts(ra,dec,distance,aperture=None)
                #note, different than cutouts above?
                mc = make_master(sdss_cutouts)
                if mc is not None:
                    master_cutout = mc

        elif G.SDSS_ALLOW:
            log.info("Calling SDSS ...")
            sdss_cutouts = catalogs.cat_sdss.SDSS().get_cutouts(ra,dec,distance,aperture=None)
            #note, different than cutouts above?
            mc = make_master(sdss_cutouts)
            if mc is not None:
                master_cutout = mc
        else:
            log.info("DECaLS, PanSTARRS, and SDSS not allowed.")


    if master_cutout is None:
        log.warning("Unable to make a master_cutout for neighborhood.")
        x = ext / 2.0
        y = ext / 2.0
        vmin = None
        vmax = None
    else:
        try:
            #update extent ... the requested size might not match, if extent is stretched differently than the actual
            #image, the positions marked will be wrong
            #DECaLs is a problem as the shape_input reported is the entire size of the original image,
            #so this stretch can be VERY wrong
            ext_rescale = master_cutout.shape[0] / master_cutout.shape_input[0]
            if ext_rescale < 0.5: #should only be a small change, so if very different (like in DECaLS) use the xmaax
                ext_rescale =  master_cutout.xmax_cutout / master_cutout.xmax_original
            ext = ext * ext_rescale
            log.debug(f"Neighborhood map master_cuout extent rescale: {ext_rescale}")

            #use the interior 30% to set the vmin, vmax to aid in matching to the smaller cutouts
            #at args.error = 3.0 and args.neighors = 10.0, this would be essentially the same pixel extents that
            #set the contstrast stretch for the main ELiXer imaging thumbnails
            # 20201021 -DD - this works okay much of the time to preserve the contrast of the interior bit, but
            #                it does often create washed out look and objects can get lost
            #                (good example: 2102183346)
            # idx0,idx1 = master_cutout.data.shape
            # frac = 0.3  #ie. at interior 20%, the left index becomes 0.5 - (0.2/2) = 0.4 * width
            # left0 = int(idx0 * 0.5 *(1.-frac))
            # left1 = int(idx1 * 0.5 *(1.-frac))
            # rght0 = int(left0 + frac * idx0)
            # rght1 = int(left1 + frac * idx1)
            #
            # if (rght0-left0) * (rght1-left1) > 500: #if there are enough pixels in the interior 30%
            #     vmin, vmax = UTIL.get_vrange(master_cutout.data[left0:rght0,left1:rght1])  # ,contrast=0.25)
            # else: #otherwise, just use the whole image
            #     vmin, vmax = UTIL.get_vrange(master_cutout.data)  # ,contrast=0.25)
            #
            vmin, vmax = UTIL.get_vrange(master_cutout.data)  # ,contrast=0.25)
            x, y = sci.get_position(ra, dec, master_cutout)  # x,y of the center
        except:
            log.debug("Exception! elixer::build_neighborhood_map, vmin,vmax,x,y set to defaults.", exec_info=True)
            x = ext / 2.0
            y = ext / 2.0
            vmin = None
            vmax = None

        try:
            fig = plt.figure()
            plt.imshow(master_cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
            plt.axis('off')

            if original_distance is not None and original_distance > 0:

            #add zoom window
                try:

                    box_ratio = original_distance / distance

                    xl,xr = plt.gca().get_xlim()
                    yb,yt = plt.gca().get_ylim()

                    zero_x = (xl+xr) / 2.
                    zero_y = (yb+yt) / 2.

                    rx = ((xr-xl) * box_ratio) / 2.0
                    ry = ((yt-yb) * box_ratio) / 2.0

                    plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2, height=ry * 2,
                                                       angle=0, color='red', fill=False))
                except:
                    log.debug("Exception! adding zoom box to mini-cutout.",exc_info=True)



            nei_buf = io.BytesIO()
            plt.savefig(nei_buf, format='png', dpi=FIG_DPI, transparent=True)
            #plt.savefig(nei_buf, format='jpg', dpi=150, quality=90,optimize=True,transparent=True)
        except:
            log.info("Exception! Unable to make mini-cutout from Neighborhood master.",exc_info=True)
            nei_buf = None


    if just_mini_cutout: #stop here
        #need to still get the line image

        mini_line_image = None
        try:
            if primary_shotid is not None and wave_range is not None:

                try:
                    if master_cutout is not None:
                        pixscale = sci.calc_pixel_size(master_cutout.wcs)
                        max_rotation_resize = np.sqrt(2.)
                    else:
                        pixscale = 0.25 #make the line image on the same scale as the master_cutout for easier mapping
                        max_rotation_resize = 1.0
                except:
                    pixscale = 0.25 #make the line image on the same scale as the master_cutout for easier mapping
                    max_rotation_resize = 1.0

                pixscale = 0.25 #make the line image on the same scale as the master_cutout for easier mapping
                mini_line_image = science_image.get_line_image(plt,friendid=None,detectid=None,
                                                          coords=SkyCoord(ra=ra,dec=dec,frame='icrs',unit='deg'),
                                                          shotid=primary_shotid, subcont=True, convolve_image=False,
                                                          pixscale=pixscale, imsize=3*distance*max_rotation_resize,
                                                          wave_range=wave_range,
                                                          sigma=None,
                                                          return_coords=False)
        except:
            log.warning("Exception building line image",exc_info=True)
            mini_line_image = None

        plt.close('all')

        try:
            plt.close('all')
            fig = plt.figure()
            line_buf = io.BytesIO()

            if master_cutout is not None:
                if ext is None or ext == 0: #use the existing extent first
                    ext = mini_line_image.shape[0] * pixscale / 2.

                im_ax = fig.add_subplot(111,projection=master_cutout.wcs) #plt.subplot(111,projection=master_cutout.wcs)

                plt.imshow(mini_line_image.data, origin='lower', interpolation='None', #extent=[-ext, ext, -ext, ext],
                           vmin=mini_line_image.vmin,vmax=mini_line_image.vmax,#cmap=plt.get_cmap('gray_r'),
                           transform=im_ax.get_transform(mini_line_image.wcs))
                im_ax.set_axis_off()

                #on the same pixel scale by design (uses the master_cutout pixel scale) ... off by, at most 1/2 pixel and are square
                #so use the master_cutout shape to "trim" off the excess collected to deal with potential rotations to align WCS
                im_ax.set_xlim(-0.5, master_cutout.data.shape[1] - 0.5)
                im_ax.set_ylim(-0.5, master_cutout.data.shape[0] - 0.5)

                if original_distance is not None and original_distance > 0:
                    #add zoom window
                    try:
                        box_ratio = original_distance / distance

                        xl,xr = plt.gca().get_xlim()
                        yb,yt = plt.gca().get_ylim()

                        zero_x = (xl+xr) / 2.
                        zero_y = (yb+yt) / 2.

                        rx = ((xr-xl) * box_ratio) / 2.0
                        ry = ((yt-yb) * box_ratio) / 2.0

                        plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2, height=ry * 2,
                                                          angle=0, color='red', fill=False))
                    except:
                        log.debug("Exception! adding zoom box to mini-cutout.",exc_info=True)

                plt.savefig(line_buf, format='png', dpi=FIG_DPI, transparent=True)
            else:
                im_ext = mini_line_image.shape[0] * pixscale / 2.
                plt.imshow(mini_line_image.data, origin='lower', interpolation='none', #extent=[-im_ext, im_ext, -im_ext, im_ext],
                           vmin=mini_line_image.vmin,vmax=mini_line_image.vmax)#,cmap=plt.get_cmap('gray_r'))
                plt.gca().set_axis_off()
                plt.savefig(line_buf, format='png', dpi=FIG_DPI, transparent=True)
        except:
            log.debug("Exception! Exception saving line_buf.")

        return None, nei_buf, line_buf


    #get the PSF weighted full 1D spectrum for each detectid
    spec = []
    wave = []
    emis = []
    shot = []
    try:
        with tables.open_file(hdf5, mode="r") as h5_detect:
            stb = h5_detect.root.Spectra
            dtb = h5_detect.root.Detections
            for d in detectids:
                rows = stb.read_where("detectid==d")

                if rows.size == 1:
                    spec.append(rows['spec1d'][0])
                    wave.append(rows['wave1d'][0])

                    drows = dtb.read_where("detectid==d")
                    if drows.size == 1:
                        emis.append(drows['wave'][0])
                        shot.append(drows['shotid'][0])
                    else:
                        emis.append(-1.0)
                        shot.append(0)

                else:
                    #there's a problem
                    spec.append(np.zeros(len(G.CALFIB_WAVEGRID)))
                    wave.append(G.CALFIB_WAVEGRID)
                    emis.append(-1.0)
                    shot.append(0)
    except Exception as e: #file might not exist
        if 'does not exist' in str(e):
            log.debug(f"Detection h5 does not exist: {hdf5} ")



    #now add the continuum sources if any
    if len(cont_detectids) > 0:
        try:
            with tables.open_file(cont_hdf5, mode="r") as h5_detect:
                stb = h5_detect.root.Spectra
                dtb = h5_detect.root.Detections
                for d in cont_detectids:
                    rows = stb.read_where("detectid==d")
                    if rows.size == 1:
                        spec.append(rows['spec1d'][0])
                        wave.append(rows['wave1d'][0])
                        emis.append(-1.0)
                    else:
                        #there's a problem
                        spec.append(np.zeros(len(G.CALFIB_WAVEGRID)))
                        wave.append(G.CALFIB_WAVEGRID)
                        emis.append(-1.0)

                    drows = dtb.read_where("detectid==d")
                    if drows.size == 1:
                        shot.append(drows['shotid'][0])
                    else:
                        shot.append(0)
        except:
            pass


    num_rows = len(detectids) + len(cont_detectids)
    #need to join continuum rows to emission line detectrows now
    # detectids += cont_detectids
    # ras += cont_ras
    # decs += cont_decs
    # dists += cont_dists
    detectids = np.concatenate((detectids,cont_detectids))
    ras = np.concatenate((ras,cont_ras))
    decs = np.concatenate((decs,cont_decs))
    dists = np.concatenate((dists,cont_dists))



    #now add the BROAD LINE sources if any
    if len(broad_detectids) > 0:
        try:
            with tables.open_file(broad_hdf5, mode="r") as h5_detect:
                stb = h5_detect.root.Spectra
                dtb = h5_detect.root.Detections
                for d in broad_detectids:
                    rows = stb.read_where("detectid==d")
                    if rows.size == 1:
                        spec.append(rows['spec1d'][0])
                        wave.append(rows['wave1d'][0])
                        emis.append(-1.0)
                    else:
                        #there's a problem
                        spec.append(np.zeros(len(G.CALFIB_WAVEGRID)))
                        wave.append(G.CALFIB_WAVEGRID)
                        emis.append(-1.0)

                    drows = dtb.read_where("detectid==d")
                    if drows.size == 1:
                        shot.append(drows['shotid'][0])
                    else:
                        shot.append(0)
        except:
            pass

    num_rows = len(detectids) + len(broad_detectids)
    #need to join continuum rows to emission line detectrows now
    # detectids += broad_detectids
    # ras += broad_ras
    # decs += broad_decs
    # dists += broad_dists
    detectids = np.concatenate((detectids,broad_detectids))
    ras = np.concatenate((ras,broad_ras))
    decs = np.concatenate((decs,broad_decs))
    dists = np.concatenate((dists,broad_dists))

    #todo: here add THIS detection IF this is a re-extraction
    if this_detection is not None:
        #prepend THIS detection to:
        num_rows += 1
        ras = np.insert(ras,0,this_detection.ra)
        decs = np.insert(decs,0,this_detection.dec)
        dists = np.insert(dists,0,0.0)


        wave = np.insert(wave,0,G.CALFIB_WAVEGRID,axis=0)
        shot = np.insert(shot,0,this_detection.survey_shotid)

        try:
            if (this_detection.sumspec_flux is not None) and \
                len(this_detection.sumspec_flux) == len(G.CALFIB_WAVEGRID):
                spec=np.insert(spec,0, this_detection.sumspec_flux / 2.0,axis=0)  # assume HETDEX 2AA and put into /1AA
                emis=np.insert(emis,0, this_detection.target_wavelength)
                detectids=np.insert(detectids,0, this_detection.entry_id)
            else: #probably this was a neighborhood_only call
                spec=np.insert(spec,0, np.zeros(len(G.CALFIB_WAVEGRID)),axis=0)
                emis=np.insert(emis,0, 0)
                detectids=np.insert(detectids,0, this_detection.entry_id)
        except:
            log.info("No sumspec_flux to add for 'this_detection'.",exc_info=True)
            spec=np.insert(spec,0,np.zeros(len(G.CALFIB_WAVEGRID)),axis=0)
            emis=np.insert(emis,0, 0)
            detectids=np.insert(detectids,0, this_detection.entry_id)



    #make the line cutout
    mini_line_image = None #"mini" in that it is for the *_mini.png, but it is acutally a larger cutout size
    line_image = None
    try:
        if primary_shotid is not None and wave_range is not None:

            try:
                if master_cutout is not None:
                    pixscale = sci.calc_pixel_size(master_cutout.wcs)
                    max_rotation_resize = np.sqrt(2.)
                else:
                    pixscale = 0.25 #make the line image on the same scale as the master_cutout for easier mapping
                    max_rotation_resize = 1.0
            except:
                pixscale = 0.25 #make the line image on the same scale as the master_cutout for easier mapping
                max_rotation_resize = 1.0

            mini_line_image = science_image.get_line_image(plt,friendid=None,detectid=None,
                                                      coords=SkyCoord(ra=ra,dec=dec,frame='icrs',unit='deg'),
                                                      shotid=primary_shotid, subcont=True, convolve_image=False,
                                                      pixscale=pixscale, imsize=3*distance*max_rotation_resize,
                                                      wave_range=wave_range,
                                                      sigma=None,
                                                      return_coords=False)

            #for convenience at this point, make a smaller version that will not be rotated and is part of the neighborhoo
            if G.PROJECT_LINE_IMAGE_TO_COMMON_WCS:
                line_image = mini_line_image
            elif master_cutout is not None and mini_line_image is not None:
                line_image = copy.deepcopy(mini_line_image)
                m0,m1 = master_cutout.data.shape
                l0,l1 = line_image.data.shape
                c0 = int(0.5*(l0-m0))
                c1 = int(0.5*(l1-m1)) #these should be square, so c0 should == c1
                line_image.data = line_image.data[c0:l0-c0,c1:l1-c1]
            else:
                line_image = mini_line_image
    except:
        log.warning("Exception building line image",exc_info=True)
        mini_line_image = None
        line_image = None

    try:
        plt.close('all')
        fig = plt.figure()
        line_buf = io.BytesIO()
        line_buf_tight = io.BytesIO()

        if master_cutout is not None:

            my,mx = np.shape(master_cutout.data)

            if ext is None or ext == 0: #use the existing extent first
                ext = mini_line_image.shape[0] * pixscale / 2.

            im_ax = fig.add_subplot(111,projection=master_cutout.wcs) #plt.subplot(111,projection=master_cutout.wcs)

            if mini_line_image is not None:
                plt.imshow(mini_line_image.data, origin='lower', interpolation='None', #extent=[-ext, ext, -ext, ext],
                       vmin=mini_line_image.vmin,vmax=mini_line_image.vmax,#cmap=plt.get_cmap('gray_r'),
                       transform=im_ax.get_transform(mini_line_image.wcs))
            im_ax.set_axis_off()



            #on the same pixel scale by design (uses the master_cutout pixel scale) ... off by, at most 1/2 pixel and are square
            #so use the master_cutout shape to "trim" off the excess collected to deal with potential rotations to align WCS
            im_ax.set_xlim(-0.5, master_cutout.data.shape[1] - 0.5)
            im_ax.set_ylim(-0.5, master_cutout.data.shape[0] - 0.5)

            #this buffer is for display later in this neighborhood function, the other is for the Zooniverse (mini)
            plt.savefig(line_buf_tight, format='png', dpi="figure", transparent=True, bbox_inches='tight')

            if original_distance is not None and original_distance > 0:
                #add zoom window
                try:
                    box_ratio = original_distance / distance

                    xl,xr = plt.gca().get_xlim()
                    yb,yt = plt.gca().get_ylim()

                    zero_x = (xl+xr) / 2.
                    zero_y = (yb+yt) / 2.

                    rx = ((xr-xl) * box_ratio) / 2.0
                    ry = ((yt-yb) * box_ratio) / 2.0

                    plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2, height=ry * 2,
                                                      angle=0, color='red', fill=False))
                except:
                    log.debug("Exception! adding zoom box to mini-cutout.",exc_info=True)

            plt.savefig(line_buf, format='png', dpi=FIG_DPI, transparent=True)#,bbox_inches='tight')
        else:
            im_ext = mini_line_image.shape[0] * pixscale / 2.
            plt.imshow(mini_line_image.data, origin='lower', interpolation='none',#extent=[-im_ext, im_ext, -im_ext, im_ext],
                           vmin=mini_line_image.vmin,vmax=mini_line_image.vmax)#,cmap=plt.get_cmap('gray_r'))
            plt.gca().set_axis_off()
            plt.savefig(line_buf, format='png', dpi=FIG_DPI, transparent=True)
            plt.savefig(line_buf_tight, format='png', dpi="figure", transparent=True, bbox_inches='tight')

    except:
        log.debug("Exception! Exception saving line_buf.",exc_info=True)

    row_step = 10 #allow space in between
    plt.close('all')

    if line_image is not None:
        fig = plt.figure(figsize=(G.FIGURE_SZ_X, G.GRID_SZ_Y * (num_rows+1)))
    else:
        fig = plt.figure(figsize=(G.FIGURE_SZ_X, G.GRID_SZ_Y * num_rows))

    plt.subplots_adjust(left=0.00, right=0.95, top=0.95, bottom=0.0)
    if num_rows > G.MAX_NEIGHBORS_IN_MAP:
        plt.suptitle(f"{total_detectids} HETDEX detections found nearby. Showing nearest {len(detectids)}.", fontsize=32)

    font = FontProperties()
    font.set_family('monospace')
    font.set_size(12)

    target_box_side = 3.0  # distance / 4.0

    # if master_cutout is not None:
    #     vmin,vmax = UTIL.get_vrange(master_cutout.data)#,contrast=0.25)
    #     x, y = sci.get_position(ra, dec, master_cutout)  # x,y of the center
    # else:
    #     x = ext/2.0
    #     y = ext/2.0
    #     vmin = None
    #     vmax = None

    #generic catalog for the utilities (use in a few place below)
    cat = cat_base.Catalog()

    if line_image is not None:
        gs = gridspec.GridSpec(row_step*(num_rows+1),10) #rows, columns #one extra row for the line_image
        # but DON'T CHANGE the num_rows as it is indexed to the ra, dec, wave, shot arrays
    else:
        gs = gridspec.GridSpec(row_step*num_rows,10) #rows, columns

    gs_idx = -1 #so will start at zero

    top_axes = None
    for i in range(num_rows):
        #first the cutout
        gs_idx += 1
        plt.subplot(gs[gs_idx*row_step+1:(gs_idx+1)*row_step-1,0:3])

        if True:

            #*** remember the positioning in pixel space is based on the RA, Dec, the pixel scale and the extent scaling
            # (everything needs to be matched up for this to work correctly)
            if master_cutout is not None:
                try:
                    plt.imshow(master_cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                               vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

                    plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

                    plt.plot(0, 0, "r+")

                    top_axes = plt.gca()


                    # add all locations
                    if i == 0: #only for the first box
                       for _ra, _dec in zip(all_ras,all_decs):
                            fx, fy = sci.get_position(_ra, _dec, master_cutout)
                            plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                          width=target_box_side, height=target_box_side,
                                                          angle=0.0, color='white', alpha=0.75,fill=False, linewidth=1.0, zorder=2))

                    # add THE neighbor box for this row on top
                    fx, fy = sci.get_position(ras[i], decs[i], master_cutout)
                    plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=neighbor_color, fill=False, linewidth=1.0, zorder=2))


                    cat.add_north_box(plt, sci, master_cutout, distance, 0, 0, theta=None)

                except:
                    log.warning("Exception.", exc_info=True)

            else:

                try:
                    fx = (ra - ras[i]) * np.cos(np.deg2rad(dec)) * 3600. #need to flip since negative RA is to the right
                    fy = (decs[i] - dec) * 3600.

                    plt.imshow(np.zeros((int(ext),int(ext))), interpolation='none', cmap=plt.get_cmap('gray_r'),
                               vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])


                    # add all locations
                    if i == 0:
                        for _ra, _dec in zip(all_ras,all_decs):
                            _fx, _fy = sci.get_position(_ra, _dec, master_cutout)

                            if (_fx is not None) and (_fy is not None):
                                plt.gca().add_patch(plt.Rectangle(((_fx - x) - target_box_side / 2.0, (_fy - y) - target_box_side / 2.0),
                                                          width=target_box_side, height=target_box_side,
                                                          angle=0.0, color='white', alpha=0.75,fill=False, linewidth=1.0, zorder=2))

                    #add (overwrite) the highlighted location
                    top_axes = plt.gca()
                    plt.gca().add_patch(plt.Rectangle((fx - target_box_side / 2.0, fy - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=neighbor_color, fill=False, linewidth=1.0, zorder=2))

                    # plt.gca().add_patch(
                    #     plt.Circle((fx, fy), radius=target_box_side, color='b', fill=False,zorder=9))

                    plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

                    plt.plot(0, 0, "r+")
                except:
                    log.warning("Exception.",exc_info=True)

            #the 1D spectrum
            plt.subplot(gs[gs_idx*row_step+1:(gs_idx+1)*row_step-1,3:])
            plt.title(r'Dist: %0.1f"  RA,Dec: (%0.5f,%0.5f)   $\lambda$: %0.2f   DetectID: %s  Shot: %s'
                      %(dists[i],ras[i],decs[i],emis[i],str(int(detectids[i])), str(shot[i])))
            plt.plot(wave[i],spec[i],zorder=9,color='b')
            plt.axhline(0,color='k',lw=1,zorder=0)
            if cwave is not None:
                plt.axvline(x=cwave,linestyle="--",zorder=1,color='k',linewidth=1.0,alpha=0.5)
            if emis[i] != -1.0:
                plt.axvline(x= emis[i],linestyle="--",zorder=1,color=neighbor_color,linewidth=1.0,alpha=0.5)
            plt.xlim((G.CALFIB_WAVEGRID[0],G.CALFIB_WAVEGRID[-1]))

            try:
                if (cwave is not None) and (3550.0 < cwave < 5450) and (3550.0 < emis[i] < 5450):
                    ymx = np.max(spec[i][40:991])
                    ymn = np.min(spec[i][40:991])
                    rn = ymx - ymn
                    plt.ylim(ymx-rn*1.1, ymn+rn*1.1)
            except:
                log.warning("Unable to set ylim for 1D spectrum in build_neighborhood_map().")

            #todo: check the specta lines (each detection's main line ... if 3600 to 5400, do not include
            #todo: ends in the y-limit calculation (to avoid run-away range values)

            #special case for line image (new way ... use image projected onto master cutout WCS
            if G.PROJECT_LINE_IMAGE_TO_COMMON_WCS and i == 0 and line_buf_tight is not None and line_image is not None: #second position
                gs_idx += 1

                #lets try displaying the figure buffer
                #line_buf line_buf_tight = io.BytesIO()

                line_buf_tight.seek(0)
                im_line = PIL_Image.open(line_buf_tight)
                i_ext = ext

                #don't set origin lower?
                em = 1.06 #needs a little nudge
                im_ax = plt.subplot(gs[gs_idx * row_step + 1:(gs_idx + 1) * row_step - 1, 0:3])
                #im_ax.set_axis_off()
                #yes, need to keep the extent
                im_ax.imshow(im_line,extent=[-i_ext*em, i_ext*em, -i_ext*em, i_ext*em])
                # do NOT use aspect
                #, aspect='auto',interpolation='nearest')#
                #, aspect='none',
                             #origin='upper',extent=[-i_ext*em, i_ext*em, -i_ext*em, i_ext*em],
                             #vmin=line_image.vmin, vmax=line_image.vmax)

                #plt.axis('tight')
                im_ax.set_xticks([int(i_ext), int(i_ext / 2.), 0, int(-i_ext / 2.), int(-i_ext)])
                im_ax.set_yticks([int(i_ext), int(i_ext / 2.), 0, int(-i_ext / 2.), int(-i_ext)])
                im_ax.set_xlim(-i_ext,i_ext)
                im_ax.set_ylim(-i_ext,i_ext)

                # zero position box (the detection)
                im_ax.add_patch(plt.Rectangle(( 0 - target_box_side / 2.0,  0 - target_box_side / 2.0),
                                              width=target_box_side, height=target_box_side,
                                              angle=0.0, color=neighbor_color, fill=False, linewidth=1.0, zorder=2))
                im_ax.plot(0, 0, "r+")

                #use master_cutout instead of line_image since we are projecting on to the master wcs
                cat.add_north_box(plt, sci, master_cutout, distance, 0, 0, theta=None)#np.pi/2.0)

                #add other detections
                # cx, cy = sci.get_position(ra, dec, line_image)
                #
                # for _ra, _dec in zip(all_ras,all_decs):
                #     fx, fy = sci.get_position(_ra, _dec, line_image)
                #     plt.gca().add_patch(plt.Rectangle(((fx - cx)  - target_box_side / 2.0, (fy - cy)  - target_box_side / 2.0),
                #                                       width=target_box_side, height=target_box_side,
                #                                       angle=0.0, color='white', alpha=0.75,fill=False, linewidth=1.0, zorder=2))

                #the 1D spectrum for the line image
                plt.subplot(gs[gs_idx*row_step+1:(gs_idx+1)*row_step-1,3:])
                plt.title(r'Line Image: $\lambda$: %0.2f +/- %0.2f'  %(emis[i],line_image.d_wave))
                plt.plot(wave[i],spec[i],zorder=9,color='b')
                plt.axhline(0,color='k',lw=1,zorder=0)

                if line_image.wave is not None:
                    yl, yh = plt.gca().get_ylim()

                    plt.fill_between([line_image.wave-line_image.d_wave,line_image.wave+line_image.d_wave],
                                     yl,yh,facecolor='gold',alpha=0.5,zorder=5)



                # if cwave is not None:
                #     plt.axvline(x=cwave,linestyle="--",zorder=1,color='k',linewidth=1.0,alpha=0.5)
                # if emis[i] != -1.0:
                #     plt.axvline(x= emis[i],linestyle="--",zorder=1,color=neighbor_color,linewidth=1.0,alpha=0.5)
                plt.xlim((G.CALFIB_WAVEGRID[0],G.CALFIB_WAVEGRID[-1]))

                # if (cwave is not None) and (3550.0 < cwave < 5450) and (3550.0 < emis[i] < 5450):
                #     ymx = np.max(spec[i][40:991])
                #     ymn = np.min(spec[i][40:991])
                #     rn = ymx - ymn
                #     plt.ylim(ymx-rn*1.1, ymn+rn*1.1)

            #special case for line image (original) ... line image is in its own orientation
            elif i == 0 and line_image is not None: #second position
                gs_idx += 1
                plt.subplot(gs[gs_idx*row_step+1:(gs_idx+1)*row_step-1,0:3]) #,projection=master_cutout.wcs)

                i_ext = ext #distance * 1.5 #ext for the line image can be on a different scale than master_cutout

                # try:
                #     diff_pix = sci.calc_pixel_size(line_image.wcs) / sci.calc_pixel_size(master_cutout.wcs)
                # except:
                #     diff_pix = 1.0
                #
                # log.info(f"Neighborhood map line_image pixelscale/master_cutout {diff_pix}")

                #if line_buf is not None:
                if False:
                   line_buf.seek(0)
                   rot_line_image = PIL_Image.open(line_buf)
                   plt.imshow(rot_line_image,# origin='lower', interpolation='none',#cmap=plt.get_cmap('gray_r'),
                              vmin=line_image.vmin, vmax=line_image.vmax)#, extent=[-i_ext, i_ext, -i_ext, i_ext])
                else: #here_line_buff

                    #original
                    plt.imshow(line_image.data, origin='lower', interpolation='none',#cmap=plt.get_cmap('gray_r'),
                              vmin=line_image.vmin, vmax=line_image.vmax,extent=[-i_ext, i_ext, -i_ext, i_ext])

                    #plt.colorbar()#,fraction=0.07)#,anchor=(0.3,0.0))

                    plt.xticks([int(i_ext), int(i_ext / 2.), 0, int(-i_ext / 2.), int(-i_ext)])
                    plt.yticks([int(i_ext), int(i_ext / 2.), 0, int(-i_ext / 2.), int(-i_ext)])

                    #zero position box (the detection)
                    plt.gca().add_patch(plt.Rectangle(( 0 - target_box_side / 2.0,  0 - target_box_side / 2.0),
                                                  width=target_box_side, height=target_box_side,
                                                  angle=0.0, color=neighbor_color, fill=False, linewidth=1.0, zorder=2))


                    cat.add_north_box(plt, sci, line_image, distance, 0, 0, theta=None)#np.pi/2.0)

                #add other detections
                # cx, cy = sci.get_position(ra, dec, line_image)
                #
                # for _ra, _dec in zip(all_ras,all_decs):
                #     fx, fy = sci.get_position(_ra, _dec, line_image)
                #     plt.gca().add_patch(plt.Rectangle(((fx - cx)  - target_box_side / 2.0, (fy - cy)  - target_box_side / 2.0),
                #                                       width=target_box_side, height=target_box_side,
                #                                       angle=0.0, color='white', alpha=0.75,fill=False, linewidth=1.0, zorder=2))

                #the 1D spectrum for the line image
                plt.subplot(gs[gs_idx*row_step+1:(gs_idx+1)*row_step-1,3:])
                plt.title(r'Line Image: $\lambda$: %0.2f +/- %0.2f'  %(emis[i],line_image.d_wave))
                plt.plot(wave[i],spec[i],zorder=9,color='b')
                plt.axhline(0,color='k',lw=1,zorder=0)

                if line_image.wave is not None:
                    yl, yh = plt.gca().get_ylim()

                    plt.fill_between([line_image.wave-line_image.d_wave,line_image.wave+line_image.d_wave],
                                     yl,yh,facecolor='gold',alpha=0.5,zorder=5)



                # if cwave is not None:
                #     plt.axvline(x=cwave,linestyle="--",zorder=1,color='k',linewidth=1.0,alpha=0.5)
                # if emis[i] != -1.0:
                #     plt.axvline(x= emis[i],linestyle="--",zorder=1,color=neighbor_color,linewidth=1.0,alpha=0.5)
                plt.xlim((G.CALFIB_WAVEGRID[0],G.CALFIB_WAVEGRID[-1]))

                # if (cwave is not None) and (3550.0 < cwave < 5450) and (3550.0 < emis[i] < 5450):
                #     ymx = np.max(spec[i][40:991])
                #     ymn = np.min(spec[i][40:991])
                #     rn = ymx - ymn
                #     plt.ylim(ymx-rn*1.1, ymn+rn*1.1)



    if fname is not None:
        try:
            plt.savefig(fname,format='png', dpi=75)
            log.debug("File written: %s" %(fname))

            # if False:
            #     import astropy.io.fits as fits
            #
            #     for i in range(len(cutouts)):
            #         co = cutouts[i]['cutout']
            #         hdu = fits.PrimaryHDU(co.data)  # essentially empty header
            #         hdu.header.update(co.wcs.to_header())  # insert the cutout's WCS
            #         hdu.writeto('/home/dustin/code/python/elixer/cutouts/test_cutout_%d.fits' % i, overwrite=True)

        except:
            log.info("Exception attempting to save neighborhood map png.",exc_info=True)

    buf = io.BytesIO()
    #plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)#,bbox_inches = 'tight', pad_inches = 0)

    return buf, nei_buf, line_buf


def check_package_versions():
    """
    very basic, check the common packages are at minimum levels
    """
    pass

def main():

    global G_PDF_FILE_NUM, OS_PNG_ONLY
    
    already_launched_viewer = False #skip the PDF viewer laun

    log.critical(f"log level {G.LOG_LEVEL}")
    #G.gc.enable()
    #G.gc.set_debug(G.gc.DEBUG_LEAK)
    try:
        args = parse_commandline()
        if args is None:
            print("Unable to parse command line. Exiting...")
            exit(0)
    except:
        log.critical("Exception in command line.",exc_info=True)
        exit(0)

    elixer_spectrum.update_with_globals()
    try: #may be used for refrerence later
         #several arguments can be modified during runtime and the original may need to be references
        original_command_line_args = copy.deepcopy(args)
    except:
        pass

    if args.upgrade_hdf5:
        upgrade_hdf5(args)
        exit(0)

    if args.remove_duplicates:
        remove_h5_duplicate_rows(args)
        exit(0)

    if args.merge_unique:
        merge_unique(args)
        exit(0)

    if args.merge:
        merge(args)
        exit(0)

    #later, below most of the processing will be skipped and only the neighborhood map is generated
    if args.neighborhood_only:
        args.neighborhood = args.neighborhood_only

    try:
        if (args.dispatch is None) or ('jupyter' in G.hostname.lower()):
            OS_PNG_ONLY = False
        else: #if in dispatch mode, likely this is on stampede2 or other HPC and we should use the system call
            OS_PNG_ONLY = True
    except:
        OS_PNG_ONLY = False

    #always build these ... the library handles the USE_PHOTO_CATS (--nophoto) global
    cat_library = catalogs.CatalogLibrary()
    cats = cat_library.get_full_catalog_list()
    catch_all_cat = cat_library.get_catch_all()

    #these web catalogs are handled individually with --sdss, --panstars, --decals
    cat_sdss = cat_library.get_sdss()
    cat_panstarrs = cat_library.get_panstarrs()
    cat_decals_web = cat_library.get_decals_web()

    #
    # build_neighborhood_map(args.hdf5, 1000525650,None, None, args.neighborhood,cwave=None,fname='lycon/test_nei.png')
    # exit()


    log.critical(f"***** ELiXer version {G.__version__} *****")
    log.critical(f"***** HETDEX DATA RELEASE {G.HDR_Version} *****")

    pages = []

    viewer_file_list = []

    #if a --line file was provided ... old way (pre-April 2018)
    #always build ifu_list
    ifu_list = ifulist_from_detect_file(args)

    hd_list = [] #if --line provided, one entry for each amp (or side) and dither
    #for rsp1 variant, hd_list should be one per detection (one per rsp subdirectory) or just one hd object?
    file_list = [] #output pdfs (parts)

    fcsdir_list = []
    hdf5_detectid_list = []
    if args.aperture:
        explicit_extraction = True
    else:
        explicit_extraction = False
    #is this an explicit extraction?
    if args.aperture and args.ra and args.dec:
        #args.wavelength, args.shotid are optional
        print("Explicit extraction ...") #single explicit extraction
        explicit_extraction = True
    elif args.fcsdir is not None:
        fcsdir_list = get_fcsdir_subdirs_to_process(args) #list of rsp1 style directories to process (each represents one detection)
        if fcsdir_list is not None:
            log.info("Processing %d entries in FCSDIR" %(len(fcsdir_list)))
            print("Processing %d entries in FCSDIR" %(len(fcsdir_list)))

    elif not args.neighborhood_only:
        if args.aperture: #still
            explicit_extraction = True
            print("Explicit extraction ...") #list of explicit extractions

        hdf5_detectid_list = get_hdf5_detectids_to_process(args)
        if hdf5_detectid_list is not None:
            log.info("Processing %d entries in HDF5" %(len(hdf5_detectid_list)))
            print("Processing %d entries in HDF5" %(len(hdf5_detectid_list)))
    else: #still even if neighborhood_only, may want neighborhood around detection
        hdf5_detectid_list = get_hdf5_detectids_to_process(args)
    #add as a payload to args so can easily check later
    args.explicit_extraction = explicit_extraction

    PDF_File(args.name, 1) #use to pre-create the output dir (just toss the returned pdf container)

    #clustering ID check
    cluster_list = None
    if args.cluster:
        try:
            #open h5 file (make sure it is okay)
            #the full path might not have been provided, so if this fails assume we are in a dispatch_xxxx location
            #and try again higher up
            if os.path.exists(args.cluster):
                cluster_h5 = tables.open_file(args.cluster)
            elif os.path.exists(os.path.join("../../",args.cluster)):
                cluster_h5 = tables.open_file(os.path.join("../../",args.cluster))
            else:
                print(f"Fatal. Unable to locate/open clustering input h5 file {args.cluster}")
                exit(-1)

            # then run clustering on multiple IDs
            cluster_list = clustering.cluster_multiple_detectids(hdf5_detectid_list,cluster_h5,outfile=False)

            # then replace the hdf5_detectid_list with just those returned
            if cluster_list is not None and len(cluster_list) > 0:
                old_len = len(hdf5_detectid_list)
                hdf5_detectid_list = [c['detectid'] for c in cluster_list]
                log.info(f"Reduced initial detectionIDs to re-run for clustering. New count = {len(hdf5_detectid_list)}, old count = {old_len}.")
                print(f"Reduced initial detectionIDs to re-run for clustering. New count = {len(hdf5_detectid_list)}, old count = {old_len}.")
            else:
                print("No clustering identified. Exiting.")
                cluster_h5.close()
                exit(0)

            #close the h5 file
            cluster_h5.close()

            #turn off neighborhood request
            if args.neighborhood is not None and args.neighborhood != 0:
                args.neighborhood = 0
                args.neighborhood_only = 0
                log.info("CLUSTERING ... turn off neighborhood map.")
        except Exception as e:
            print("Fatal exception.",e)
            exit(-1)


    #if this is a re-run (recovery run) remove any detections that have already been processed
    master_loop_length = 1
    master_fcsdir_list = []
    master_hdf5_detectid_list = []

    if G.RECOVERY_RUN:
        #as of 1.18.0a2, will use the ra,dec generated ID to check for completion
        # if args.aperture is not None:
        #     #G.RECOVERY_RUN = False
        #     log.info("Forced extraction does not fully support RECOVERY MODE (behavior may be unexpected).")
        #     print("Forced extraction does not fully support RECOVERY MODE (behavior may be unexpected).")
        #     #still want the master loop, though
        #     master_loop_length = len(hdf5_detectid_list)
        #     master_hdf5_detectid_list = hdf5_detectid_list
        if len(hdf5_detectid_list) > 0:
            hdf5_detectid_list = prune_detection_list(args,None,hdf5_detectid_list)
            if len(hdf5_detectid_list) == 0:
                print("[RECOVERY MODE] All detections already processed. Exiting...")
                log.info("[RECOVERY MODE] All detections already processed. Exiting...")
                log.critical("Main complete.")
                exit(0)
            else:
                master_loop_length = len(hdf5_detectid_list)
                master_hdf5_detectid_list = hdf5_detectid_list
                log.info("[RECOVERY] Processing %d entries in HDF5" % (len(hdf5_detectid_list)))
                print("[RECOVERY] Processing %d entries in HDF5" % (len(hdf5_detectid_list)))
        elif len(fcsdir_list):
            fcsdir_list = prune_detection_list(args,fcsdir_list,None)
            if len(fcsdir_list) == 0:
                print("[RECOVERY MODE] All detections already processed. Exiting...")
                log.info("[RECOVERY MODE] All detections already processed. Exiting...")
                log.critical("Main complete.")
                exit(0)
            else:
                master_loop_length = len(fcsdir_list)
                master_fcsdir_list = fcsdir_list
                log.info("[RECOVERY] Processing %d entries in FCSDIR" % (len(fcsdir_list)))
                print("[RECOVERY] Processing %d entries in FCSDIR" % (len(fcsdir_list)))
        else:
            G.RECOVERY_RUN = False
            log.debug("No list (hdf5 or fcsdir) of detections, so RECOVERY MODE turned off.")

    #now, we have only the detections that have NOT been processed

    for master_loop_idx in range(master_loop_length):

        #stupid, but works until I can properly reorganize
        #on each run through this loop, set the list to one element
        if len(master_hdf5_detectid_list) > 0: #this is an hdf5 run
            hdf5_detectid_list = [master_hdf5_detectid_list[master_loop_idx]]
        elif len(master_fcsdir_list) > 0:
            fcsdir_list = [master_fcsdir_list[master_loop_idx]]

        #reset the other lists
        del hd_list[:]
        del file_list[:]

        hd_list = []
        file_list = []
        match_list = match_summary.MatchSet()

        # first, if hetdex info provided, build the hetdex part of the report
        # hetedex part
        if build_hd(args):
            #hd_list by IFU (one hd object per IFU, pre-April 2018)
            #each hd object can have multiple DetObjs (assigned s|t the first fiber's IFU matches the hd IFU)

            if args.neighborhood_only:
                basic_only=True
            else:
                basic_only=False

            if (len(ifu_list) > 0) and ((args.ifuslot is None) and (args.ifuid is None) and (args.specid is None)):

                #sort so easier to find
                ifu_list.sort()

                for ifu in ifu_list:
                    args.ifuslot = int(ifu)
                    hd = hetdex.HETDEX(args,basic_only=basic_only,cluster_list=cluster_list)
                    if (hd is not None) and (hd.status != -1):
                        hd_list.append(hd)
            elif len(fcsdir_list) > 0: #rsp style
                #build one hd object for each ?
                #assume of form 20170322v011_xxx
                obs_dict = {} #key=20170322v011  values=list of dirs

                for d in fcsdir_list:
                    key = os.path.basename(d).split("_")[0]
                    if key in obs_dict:
                        obs_dict[key].append(d)
                    else:
                        obs_dict[key] = [d]

                for key in obs_dict.keys():
                    plt.close('all')
                    hd = hetdex.HETDEX(args,fcsdir_list=obs_dict[key],basic_only=basic_only,cluster_list=cluster_list) #builds out the hd object (with fibers, DetObj, etc)
                    #todo: maybe join all hd objects that have the same observation
                    # could save in loading catalogs (assuming all from the same observation)?
                    # each with then multiple detections (DetObjs)
                    if hd.status == 0:
                        hd_list.append(hd)
            elif len(hdf5_detectid_list) > 0: #HDF5 (DataRelease style)
                if explicit_extraction:      #only one detection per hetdex object
                    #survey = None
                    for d in hdf5_detectid_list:
                        plt.close('all')

                        #for safety
                        try:
                            #d is a list or a single value here
                            _ = d[0] #if d is an array, set any None's to '0'
                            d = np.array(d)
                            d[d == None] = '0'
                        except:
                            #d is not an array, but need to make it one, this is probably a detectID
                            d = np.array([d])
                            d[d == None] = '0'
                        #    log.warning("Exception checking hdf5_detectid_list for None-types",exc_info=True)

                        if isinstance(d,np.int64): #this is a detetid, not list of values RA, Dec, ...
                            hd = hetdex.HETDEX(args, fcsdir_list=None, hdf5_detectid_list=[d], basic_only=basic_only,cluster_list=cluster_list)
                            if hd.status == 0:
                                hd_list.append(hd)
                            continue

                        #otherwise this a a list of RA, Dec, ...
                        #update the args with the ra dec and shot to build an appropriate hetdex object for extraction

                        #d might just be a detectID
                        try:
                            local_ra, local_dec = UTIL.coord2deg(str(d[0]) + " " + str(d[1]))
                            if local_ra is not None and local_dec is not None:
                                d[0] = local_ra
                                d[1] = local_dec
                        except:
                            pass

                        try:
                            if len(d) == 0:
                                log.error(f"Invalid detectID list entry: {d} ")
                                continue

                            elif len(d) == 1: #should be a detectid
                                plt.close('all')
                                hd = hetdex.HETDEX(args, fcsdir_list=None, hdf5_detectid_list=d,
                                                       basic_only=basic_only, cluster_list=cluster_list)

                                if hd.status == 0:
                                    hd_list.append(hd)
                                continue #either way, this entry is handled, we don't want to add it again below

                            elif len(d) == 2:
                                args.ra = float(d[0])
                                args.dec = float(d[1])
                                args.shotid = None
                            elif len(d) == 3:
                                args.ra = float(d[0])
                                args.dec = float(d[1])
                                args.shotid = xlat_shotid(d[2])
                                if (args.shotid is not None) and (3400. < args.shotid < 5700.): #assume this is really a wavelength
                                    args.wavelength = args.shotid
                                    args.shotid = None
                            elif len(d) == 4:
                                args.ra = float(d[0])
                                args.dec = float(d[1])
                                args.shotid = xlat_shotid(d[2])
                                if args.shotid is None:
                                    #could be a zero
                                    args.wavelength = None
                                    args.shotid = xlat_shotid(d[3])
                                elif (3400. < args.shotid < 5700.):  # assume this is really a wavelength and shotid is flipped with wavelength
                                    args.wavelength = args.shotid
                                    args.shotid = xlat_shotid(d[3])
                                else: #shotid has a value, assume the next is the wavelength
                                    args.wavelength = float(d[3])
                                if args.wavelength == 0:
                                    args.wavelength = None
                            elif len(d) == 5:
                                args.ra = float(d[0])
                                args.dec = float(d[1])
                                args.shotid = xlat_shotid(d[2])
                                if args.shotid is None:
                                    #could be a zero
                                    args.wavelength = None
                                    args.shotid = xlat_shotid(d[3])
                                elif (3400. < args.shotid < 5700.):  # assume this is really a wavelength and shotid is flipped with wavelength
                                    args.wavelength = args.shotid
                                    args.shotid = xlat_shotid(d[3])
                                else: #shotid has a value assume the next is the wavelength
                                    args.wavelength = float(d[3])
                                if args.wavelength == 0:
                                    args.wavelength = None
                                args.manual_name = int(d[4]) #needs to be an int
                            else:
                                #something wrong
                                log.error(f"Unable to build hetdex object for entry d: ({d})")
                                print(f"Unable to build hetdex object for entry d: ({d})")

                            if not args.shotid: #must fill this in, so look for all shots and add a hetdex obj for each
                                #todo: fill in shots here
                                #print("Todo: fill in unspecified shotids")
                                if G.the_Survey is None:
                                    G.the_Survey = hda_survey.Survey(survey=f"hdr{G.HDR_Version}")
                                    if not G.the_Survey:
                                        log.error(f"Cannot build hetdex_api survey object to determine shotid for {d}")
                                        continue

                                #this is only looking at the pointing, not checking individual fibers, so
                                #need to give it a big radius to search that covers the focal plane
                                #if centered (and it should be) no ifu edge is more than 12 acrmin away
                                shotlist = G.the_Survey.get_shotlist(SkyCoord(args.ra, args.dec, unit='deg',frame='icrs'),
                                                               radius=G.FOV_RADIUS_DEGREE*U.deg)
                                try:
                                    base_name = args.manual_name #need to save it off, since we are going to modify args.manual_name
                                except:
                                    base_name = None

                                if len(shotlist) > 0:
                                    for i,s in enumerate(shotlist):
                                        args.shotid = s
                                        if base_name is not None: #leave three spaces for extra shots, very rare to have
                                            #more than a few, but this separates it nicely
                                            try:
                                                args.manual_name = base_name * 1000 + i
                                                if args.manual_name < 9e9:
                                                    args.manual_name = args.manual_name + int(9e10) #add leading 9
                                            except:
                                                pass
                                        hd = hetdex.HETDEX(args, basic_only=basic_only,cluster_list=cluster_list)
                                        if hd.status == 0:
                                            hd_list.append(hd)
                                    try:
                                        if base_name is not None:
                                            args.manual_name = base_name #put it back
                                    except:
                                        pass
                                else:
                                    try:
                                        log.info(f"[{base_name}] RA,Dec ({args.ra},{args.dec}): No matching shots found. --require_hetdex = {args.require_hetdex}")
                                    except:
                                        log.info("No matching shots found.")
                            else:
                                hd = hetdex.HETDEX(args,basic_only=basic_only,cluster_list=cluster_list)
                                if hd.status == 0:
                                    hd_list.append(hd)
                        except:
                            log.error(f"Unable to build hetdex object for ra ({args.ra}), dec ({args.dec}), shot ({args.shotid})",exc_info=True)
                            print(f"Unable to build hetdex object for ra ({args.ra}), dec ({args.dec}), shot ({args.shotid})")
                            args.ra = None
                            args.dec = None
                            args.shotid = None
                else:
                    #only one detection per hetdex object
                    for d in hdf5_detectid_list:
                        plt.close('all')
                        hd = hetdex.HETDEX(args,fcsdir_list=None,hdf5_detectid_list=[d],basic_only=basic_only,cluster_list=cluster_list)

                        if hd.status == 0:
                            hd_list.append(hd)

            elif explicit_extraction:

                if args.shotid:
                    hd = hetdex.HETDEX(args,basic_only=basic_only,cluster_list=cluster_list) #builds out the hd object (with fibers, DetObj, etc)
                    if hd is not None:
                        if hd.status == 0:
                            hd_list.append(hd)
                elif args.neighborhood_only:
                    args.shotid = "00000000000"
                    hd = hetdex.HETDEX(args, basic_only=basic_only,cluster_list=cluster_list)
                    if hd.status == 0:
                        hd_list.append(hd)
                else:
                    if G.the_Survey is None:
                        G.the_Survey = hda_survey.Survey(survey=f"hdr{G.HDR_Version}")
                    if not G.the_Survey:
                        log.error(f"Cannot build hetdex_api survey object to determine shotid for {d}")
                        print(f"Cannot build hetdex_api survey object to determine shotid for {d}")
                        exit(-1)

                    # this is only looking at the pointing, not checking individual fibers, so
                    # need to give it a big radius to search that covers the focal plane
                    # if centered (and it should be) no ifu edge is more than 12 acrmin away
                    shotlist = G.the_Survey.get_shotlist(SkyCoord(args.ra, args.dec, unit='deg', frame='icrs'),
                                                   radius=G.FOV_RADIUS_DEGREE * U.deg)
                    for s in shotlist:
                        args.shotid = s
                        hd = hetdex.HETDEX(args, basic_only=basic_only,cluster_list=cluster_list)
                        if hd.status == 0:
                            hd_list.append(hd)

                    #keep at least one even if all detections are bad (status < 0), but toss any bad ones otherwise
                    #NOTICE: hd_list in this case is comprised ONLY of targets at ONE coordinate set
                    #so this logic is okay ... IF hd_list is cumulative over multiple coordiantes, this is WRONG !!!
                    try:
                        status_list = [y.status for sub in [x.emis_list for x in hd_list] for y in sub]
                        if len(status_list) > 1 and min(status_list) < 0:
                            if max(status_list) >=0: #at least one is good, remove all the bad ones
                                #clean out any bad status detects as we already have the position covered
                                for h in reversed(hd_list):
                                    for e in reversed(h.emis_list):
                                        if e.status < 0:
                                            h.emis_list.remove(e)
                                    if len(h.emis_list) == 0:
                                        hd_list.remove(h)
                            else: #all are bad, so just keep one
                                hd_list = [hd_list[0]]
                                hd_list[0].emis_list =  [hd_list[0].emis_list[0]]
                    except:
                        log.debug("(minor) Unable to clean hd list.",exc_info=True)


            else: #this should not happen
                hd = hetdex.HETDEX(args,cluster_list=cluster_list) #builds out the hd object (with fibers, DetObj, etc)
                if hd is not None:
                    if hd.status == 0:
                        hd_list.append(hd)

        if not args.neighborhood_only:

            if args.score:
                #todo: future possibility that additional analysis needs to be done (beyond basic fiber info). Move as needed
                if len(hd_list) > 0:
                    if not os.path.isdir(args.name):
                        try:
                            os.makedirs(args.name)
                        except OSError as exception:
                            if exception.errno != errno.EEXIST:
                                print("Fatal. Cannot create pdf output directory: %s" % args.name)
                                log.critical("Fatal. Cannot create pdf output directory: %s" % args.name, exc_info=True)
                                exit(-1)
                    #esp. for older panacea and for cure, need data built in the data_dict to compute score
                    for hd in hd_list:
                        for emis in hd.emis_list:
                            emis.outdir = args.name
                            hd.build_data_dict(emis)

                    write_fibers_file(os.path.join(args.name, args.name + "_fib.txt"), hd_list)
                log.critical("Main complete.")
                exit(0)

            if len(hd_list) > 0:
                total_emis = 0
                for hd in hd_list:
                    if hd.status != 0:
                        if len(hd_list) > 1:
                            continue
                        else:
                            # fatal
                            print("Fatal error. Cannot build HETDEX working object.")
                            log.critical("Fatal error. Cannot build HETDEX working object.")
                            log.critical("Main exit. Fatal error.")
                            exit (-1)

                    #iterate over all emission line detections
                    #this is not really the best solution
                    # if len(hd.emis_list) == 0: #none were found in HETDEX, but still make imaging cutouts
                    #    try:
                    #        #make a dummy emission object just to get the catalog position cutouts
                    #        e = hetdex.DetObj(None, emission=True, basic_only=True)
                    #        if e is not None:
                    #            G.UNIQUE_DET_ID_NUM += 1
                    #            e.id = G.UNIQUE_DET_ID_NUM
                    #            e.ra = hd.target_ra
                    #            e.dec = hd.target_dec
                    #            hd.emis_list.append(e)
                    #    except:
                    #        pass

                    if len(hd.emis_list) > 0:
                        total_emis += len(hd.emis_list)
                        print()
                        #first see if there are any possible matches anywhere
                        #matched_cats = []  ... here matched_cats needs to be per each emission line (DetObj)
                        num_hits = 0

                        for e in hd.emis_list:
                            plt.close('all')
                            log.info("Processing catalogs for eid(%s) ... " %str(e.entry_id))

                            if G.DECALS_WEB_FORCE: #prioritize DECaLS over PANSTARRS
                                e.matched_cats.append(cat_decals_web)
                            elif G.PANSTARRS_FORCE: #prioritize PANSTARRS over SDSS
                                e.matched_cats.append(cat_panstarrs)
                            elif G.SDSS_FORCE:
                                e.matched_cats.append(cat_sdss)
                            else:
                                for c in cats:
                                    if (e.wra is not None) and (e.wdec is not None):  # weighted RA and Dec
                                        ra = e.wra
                                        dec = e.wdec
                                    else:
                                        ra = e.ra
                                        dec = e.dec
                                    if c.position_in_cat(ra=ra, dec=dec, error=args.error): #
                                        in_cat = True
                                        hits, _, _ = c.build_list_of_bid_targets(ra=ra, dec=dec, error=args.error)
                                        if hits < 0:
                                            # detailed examination found that the position cannot be found in the catalogs
                                            # this is for a tiled catalog (like SHELA or HSC) where the range is there, but
                                            # there is no tile that covers that specific position

                                            hits = 0
                                            in_cat = False

                                        num_hits += hits
                                        e.num_hits = hits #yes, for printing ... just the hits in this ONE catalog

                                        if in_cat and (c not in e.matched_cats):
                                            e.matched_cats.append(c)

                                        print("%d hits in %s for Detect ID #%d" % (hits, c.name, e.id))
                                        log.info("%d hits in %s for Detect ID #%d" % (hits, c.name, e.id))
                                    else: #todo: don't bother printing the negative case
                                        print("Coordinates not in range of %s for Detect ID #%d" % (c.name,e.id))
                                        log.info("Coordinates not in range of %s for Detect ID #%d" % (c.name, e.id))

                                if len(e.matched_cats) == 0:

                                    log.info("No catalog overlap. DECALS_WEB_ALLOW (%s), PANSTARRS_ALLOW (%s), SDSS_ALLOW (%s)"
                                             % (str(G.DECALS_WEB_ALLOW), str(G.PANSTARRS_ALLOW), str(G.SDSS_ALLOW) ))

                                    if G.DECALS_WEB_ALLOW:  # prioritize DECaLS over PANSTARRS
                                        e.matched_cats.append(cat_decals_web)
                                    elif G.PANSTARRS_ALLOW:  # prioritize PANSTARRS over SDSS
                                        e.matched_cats.append(cat_panstarrs)
                                    elif G.SDSS_ALLOW:
                                        e.matched_cats.append(cat_sdss)
                                    else:
                                        e.matched_cats.append(catch_all_cat)

                        if (args.annulus is None) and (not confirm(num_hits,args.force)):
                            log.critical("Main exit. User cancel.")
                            exit(0)

                        #now build the report for each emission detection
                        for e in hd.emis_list:

                            if e.survey_shotid is not None and e.survey_shotid < 20180601000:
                                G.NUDGE_SEP_MAX_DIST = G.NUDGE_SEP_MAX_DIST_EARLY_DATA
                            else:
                                G.NUDGE_SEP_MAX_DIST = G.NUDGE_SEP_MAX_DIST_LATER_DATA

                            pdf = PDF_File(args.name, e.entry_id, e.pdf_name)
                            e.outdir = pdf.basename
                            #update pdf_name to match
                            try:
                                e.pdf_name = os.path.basename(pdf.filename)
                            except:
                                pass #not important if this fails

                            id = "Detect ID #" + str(e.id)
                            if (e.wra is not None) and (e.wdec is not None): #weighted RA and Dec
                                ra = e.wra
                                dec = e.wdec
                            else:
                                ra = e.ra
                                dec = e.dec


                            #todo: ANNULUS STUFF HERE
                            #todo: still use hetdex objects, but want a different hetdex section
                            #todo: and we won't be catalog matching (though will grab images from them)

                            if args.annulus is None:
                                pdf.pages = build_hetdex_section(pdf.filename,hd,e.id,pdf.pages) #this is the fiber, spectra cutouts for this detect

                                # if e.status < 0:
                                #     log.error(f"{e.id} Unable to build HETDEX section.")
                                #     continue

                                match = match_summary.Match(e)

                                pdf.pages,pdf.bid_count = build_pages(pdf.filename, match, ra, dec, args.error, e.matched_cats, pdf.pages,
                                                              num_hits=e.num_hits, idstring=id,base_count=0,target_w=e.w,
                                                              fiber_locs=e.fiber_locs,target_flux=e.estflux,detobj=e)

                                #add in lines and classification info
                                match_list.add(match) #always add even if bids are none
                                if e.status < 0:
                                    pdf.status = -1
                                    pdf.filename += "!"
                                file_list.append(pdf)
                            else: #todo: this is an annulus examination (fiber stacking)
                                log.info("***** ANNULUS ***** ")
                                pdf.pages = build_hetdex_section(pdf.filename, hd, e.id, pdf.pages, annulus=True)

                                # if e.status < 0:
                                #     log.error(f"{e.id} Unable to build HETDEX section.")
                                #     continue

                                pdf.pages, pdf.bid_count = build_pages(pdf.filename, None, ra, dec, args.error, e.matched_cats,
                                                                       pdf.pages, num_hits=0, idstring=id, base_count=0,
                                                                       target_w=e.w, fiber_locs=e.fiber_locs,
                                                                       target_flux=e.estflux, annulus=args.annulus,obs=e.syn_obs)
                                if e.status < 0:
                                    pdf.filename += "!"
                                    pdf.status = -1

                                file_list.append(pdf)

                # else: #for multi calls (which are common now) this is of no use
                   #     print("\nNo emission detections meet minimum criteria for specified IFU. Exiting.\n"
                   #     log.warning("No emission detections meet minimum criteria for specified IFU. Exiting.")

                if total_emis < 1:
                    log.info("No detections match input parameters.")
                    print("No detections match input parameters.")

            elif (args.ra is not None) and (args.dec is not None):
                if (args.require_hetdex is False):
                    num_hits = 0
                    num_cats = 0
                    catlist_str = ""
                    matched_cats = [] #there were no detection objects (just an RA, Dec) so use a generic, global matched_cats

                    if G.DECALS_WEB_FORCE:
                        matched_cats.append(cat_decals_web)
                        catlist_str += cat_decals_web.name + ", "
                    elif G.PANSTARRS_FORCE:
                        matched_cats.append(cat_panstarrs)
                        catlist_str += cat_panstarrs.name + ", "  # still need this for autoremoval of trailing ", " later
                    elif G.SDSS_FORCE:
                        matched_cats.append(cat_sdss)
                        catlist_str += cat_sdss.name + ", " #still need this for autoremoval of trailing ", " later
                    else:
                        for c in cats:
                            if c.position_in_cat(ra=args.ra,dec=args.dec,error=args.error):
                                if args.error > 0:
                                    hits,_,_ = c.build_list_of_bid_targets(ra=args.ra,dec=args.dec,error=args.error)

                                    if hits < 0:
                                        #there was a problem ... usually we are in the footprint
                                        #but in a gap or missing area
                                        hits = 0
                                    else:
                                        num_hits += hits
                                        num_cats += 1
                                        if c not in matched_cats:
                                            matched_cats.append(c)
                                            catlist_str += c.name + ", "

                                    if hits > 0:
                                        print ("%d hits in %s" %(hits,c.name))
                                    elif args.catcheck:
                                        print("%d hits in %s (*only checks closest tile)" %(hits,c.name))
                                else:
                                    num_cats += 1
                                    if c not in matched_cats:
                                        matched_cats.append(c)
                                        catlist_str += c.name + ", "

                        if (len(matched_cats) == 0):
                            if G.DECALS_WEB_ALLOW:
                                matched_cats.append(cat_decals_web)
                                catlist_str += cat_decals_web.name + ", "
                            elif G.PANSTARRS_ALLOW:
                                #todo: should peek and see if panstarrs has a hit? if not then fall to SDSS?
                                matched_cats.append(cat_panstarrs)
                                catlist_str += cat_panstarrs.name + ", " #still need this for autoremoval of trailing ", " later
                            elif G.SDSS_ALLOW:
                                matched_cats.append(cat_sdss)
                                catlist_str += cat_sdss.name + ", " #still need this for autoremoval of trailing ", " later


                    if args.catcheck:
                        catlist_str = catlist_str[:-2]
                        print("%d overlapping catalogs (%f,%f). %s" %(num_cats,args.ra, args.dec, catlist_str))
                        exit(0)
                        #if num_cats == 0:
                        #    num_hits = -1 #will show -1 if no catalogs vs 0 if there are matching catalogs, just no matching targets
                            #print("-1 hits. No overlapping imaging catalogs.")
                    else:
                        if not confirm(num_hits,args.force):
                            log.critical("Main exit. User cancel.")
                            exit(0)

                        pages,_ = build_pages(args.name,None,args.ra, args.dec, args.error, matched_cats, pages, idstring="# 1 of 1")
                else:

                    try:
                        if args.coords is not None or (hdf5_detectid_list is not None and len(hdf5_detectid_list) > 1):
                            pass #there was a whole list and it has already been logged
                        else:
                            log.info(f"[{args.manual_name}], RA,Dec: ({args.ra},{args.dec}). No matching HETDEX information and --require_hetdex is specified.")
                    except:
                        log.info("No matching HETDEX information and --require_hetdex is specified.")
            else:
                print("Invalid command line call. Insufficient information to execute or No detections meet minimum criteria.")
                exit(-1)

            #need to combine PLAE/POII and other classification data before joining report parts
            if G.COMBINE_PLAE: # and not G.CONTINUUM_RULES:
                for h in hd_list:
                    for e in h.emis_list:
                        if e.status >= 0:
                            plae, plae_sd, size_in_psf, diam_in_arcsec = e.combine_all_plae(use_continuum=True)

                            #scale_plae_list = []
                            #reason_list = []
                            if G.AGGREGATE_PLAE_CLASSIFICATION:
                                    #keep the first
                                #for plya_thresh in G.PLYA_VOTE_THRESH_LIST:  # unique list
                                #no ... just call once .... the threshold does not truly matter here
                                # and the unsure extra weight SHOULD ALWAYS be 0.5, regardless
                                scale_plae, reason = e.aggregate_classification()

                                if (scale_plae is None) or np.isnan(scale_plae):
                                    scale_plae = -99.0
                                    # scale_plae_list.append(scale_plae)
                                    # reason_list.append(reason) #really only using the 1st one

                            if G.ZEROTH_ROW_HEADER:
                                header_text = ""
                                #if len(scale_plae_list) >= 1: #(scale_plae is not None) and (not np.isnan(scale_plae)):
                                if (scale_plae is not None) and (not np.isnan(scale_plae)):
                                    try:

                                        try:
                                            plae_high = min(1000.0,e.classification_dict['plae_hat_hi'])
                                        except:
                                            plae_high = -1

                                        try:
                                            plae_low = max(0.001,e.classification_dict['plae_hat_lo'])
                                        except:
                                            plae_low = -1

                                        #if scale_plae_list[0] < 0:
                                        if scale_plae < 0:
                                                header_text = r"Combined P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  P(Ly$\alpha$): %d %s" \
                                                              % (round(plae, 3), round(plae_high, 3), round(plae_low, 3),
                                                                 int(scale_plae), reason)
                                        else:
                                            try: #will want to check some flags in best_redshift() to impact p_of_z
                                                if not e.full_flag_check_performed:
                                                    e.flag_check()
                                                if e.flags != 0:
                                                    header_text += f"    Flags:0x{e.flags:08x}"
                                            except:
                                                pass

                                            best_z_list = []
                                            p_of_z_list = []
                                            #for scale_plae in scale_plae_list:
                                            for plya_thresh in G.PLYA_VOTE_THRESH_LIST:
                                                best_z, p_of_z = e.best_redshift(plya_thresh)
                                                best_z_list.append(best_z)
                                                p_of_z_list.append(p_of_z)

                                            if e.flags & G.DETFLAG_UNCERTAIN_CLASSIFICATION:
                                                e.flags |= G.DETFLAG_FOLLOWUP_NEEDED

                                            if (e.flags & G.DETFLAG_FOLLOWUP_NEEDED) or \
                                               (e.flags & G.DETFLAG_LARGE_NEIGHBOR and e.flags & G.DETFLAG_COUNTERPART_NOT_FOUND):
                                                e.needs_review = 1

                                            try:
                                                combined_ew = e.classification_dict['combined_eqw_rest_lya']
                                                combined_ew_err = e.classification_dict['combined_eqw_rest_lya_err']
                                            except:
                                                combined_ew = 0
                                                combined_ew_err = 0

                                            if len(p_of_z_list) > 0 and p_of_z_list[0] > 0:
                                                if e.cluster_z == best_z_list[0]:
                                                    #scale_plae = scale_plae_list[0]
                                                    p_of_z = p_of_z_list[0]
                                                    best_z = best_z_list[0]
                                                    e.flags |= G.DETFLAG_Z_FROM_NEIGHBOR
                                                    header_text = r"EW: %0.1f$\pm$%0.1f$\AA$  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  " \
                                                                  r"P(Ly$\alpha$): %0.3f  Q(z): %0.2f  z: %0.4f*" \
                                                                  % (max(-9999,min(combined_ew,9999)),max(-9999,min(combined_ew_err,9999)),
                                                                      round(plae, 3),round(plae_high, 3),round(plae_low, 3),scale_plae,p_of_z,best_z)
                                                else:
                                                    #what line is best_z?
                                                    try:
                                                        if G.CONTINUUM_RULES and e.spec_obj.solutions is not None and \
                                                            len(e.spec_obj.solutions) > 0 and e.spec_obj.solutions[0].emission_line.absorber:
                                                            p_of_z = p_of_z_list[0]
                                                            best_z = best_z_list[0]
                                                            if e.spec_obj.central_eli.absorber:
                                                                line_label = e.spec_obj.match_line(e.w,best_z_list[0],aa_error=6.0,
                                                                                                   allow_absorption=True,
                                                                                                   allow_emission=True).name
                                                                #need to allow emission since most lines are kept that way
                                                                #but can be both (like hydrogen series in a WD)
                                                            else:
                                                                line_label = e.spec_obj.match_line(e.w, best_z_list[0],
                                                                                                   aa_error=6.0,
                                                                                                   allow_absorption=False,
                                                                                                   allow_emission=True).name
                                                        else:
                                                            line_label = e.spec_obj.match_line(e.w,best_z_list[0],aa_error=6.0).name
                                                    except:
                                                        line_label = ""


                                                    if len(best_z_list) == 1:
                                                        header_text = r"EW: %0.1f$\pm$%0.1f$\AA$  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  " \
                                                                  r"P(Ly$\alpha$): %0.3f  Q(z): %0.2f  z: %0.4f %s" \
                                                              % (max(-9999,min(combined_ew,9999)),max(-9999,min(combined_ew_err,9999)),round(plae, 3),round(plae_high, 3),round(plae_low, 3),
                                                                 scale_plae,p_of_z_list[0],best_z_list[0],line_label)
                                                    elif len(best_z_list) == 2:
                                                        header_text = r"EW: %0.1f$\pm$%0.1f$\AA$  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  " \
                                                                  r"P(Ly$\alpha$): %0.3f  Q(z): $%0.2f\ ^{%0.2f}$  z: $%0.4f\ ^{%0.4f}$ %s" \
                                                              % (max(-9999,min(combined_ew,9999)),max(-9999,min(combined_ew_err,9999)),round(plae, 3),round(plae_high, 3),round(plae_low, 3),
                                                                 scale_plae,
                                                                 p_of_z_list[0],p_of_z_list[1],
                                                                 best_z_list[0],best_z_list[1],line_label)
                                                    elif len(best_z_list) == 3:
                                                        header_text = r"EW: %0.1f$\pm$%0.1f$\AA$  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  " \
                                                                      r"P(Ly$\alpha$): %0.3f  Q(z): $%0.2f\ ^{%0.2f}_{%0.2f}$  z: $%0.4f\ ^{%0.4f}_{%0.4f}$ %s" \
                                                                      % (max(-9999, min(combined_ew, 9999)),
                                                                         max(-9999, min(combined_ew_err, 9999)),
                                                                         round(plae, 3), round(plae_high, 3),
                                                                         round(plae_low, 3),
                                                                         scale_plae,
                                                                         p_of_z_list[0], p_of_z_list[1], p_of_z_list[2],
                                                                         best_z_list[0], best_z_list[1],best_z_list[2],line_label)
                                                    else:
                                                        header_text = r"EW: %0.1f$\pm$%0.1f$\AA$  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  " \
                                                                  r"P(Ly$\alpha$): %0.3f  Q(z): %0.2f  z: %0.4f %s" \
                                                              % (max(-9999,min(combined_ew,9999)),max(-9999,min(combined_ew_err,9999)),round(plae, 3),round(plae_high, 3),round(plae_low, 3),
                                                                 scale_plae,p_of_z_list[0],best_z_list[0],line_label)
                                                        log.error(f"ERROR! Unexpected lenght of best_z_list: {len(best_z_list)}")
                                            else:
                                                header_text = r"EW: %0.1f$\pm$%0.1f$\AA$  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$  P(Ly$\alpha$): %0.3f" \
                                                  % (max(-9999,min(combined_ew,9999)),max(-9999,min(combined_ew_err,9999)),round(plae, 3),round(plae_high, 3),round(plae_low, 3),scale_plae)

                                        try:
                                            if len(e.spec_obj.classification_label) > 0:
                                                header_text += "  " + e.spec_obj.classification_label.rstrip(",")
                                        except:
                                            pass
                                    except:# Exception as Ex:
                                        pass
                                        #print(Ex)


                                try:
                                    if not e.full_flag_check_performed:
                                        e.flag_check()
                                    if e.flags != 0:
                                        header_text += f"  Flags:0x{e.flags:08x}"
                                    if reason is not None and len(reason) > 3 and scale_plae >= 0:
                                        header_text += f"  {reason}"
                                except:
                                    pass


                                if G.LyC:
                                    header_text += f" (Lyman Continuum Focus)"


                                try:
                                    build_report_part(os.path.join(e.outdir, e.pdf_name),[make_zeroth_row_header(header_text,redtext=e.red_header)],0)
                                except:
                                    log.debug("Exception calling build_report_part",exc_info=True)
                        # else: #try both
                        #
                        #      # method 1 (PLAE direct)
                        #     plae, plae_sd, size_in_psf = e.combine_all_plae(use_continuum=False)
                        #     scale_plae1 = e.aggregate_classification()
                        #     if (scale_plae1 is None) or np.isnan(scale_plae1):
                        #          scale_plae1 = -99.0
                        #     # method 2 (re-do PLAE with continuum)
                        #     plae, plae_sd, size_in_psf = e.combine_all_plae(use_continuum=True)
                        #     scale_plae2 = e.aggregate_classification()
                        #     if (scale_plae2 is None) or np.isnan(scale_plae2):
                        #         scale_plae2 = -99.0
                        #
                        #     header_text = f"Classification P(LAE): Method(1) {scale_plae1 * 100.:0.2f}%  Method(2) {scale_plae2 * 100.:0.2f}%"
                        #     try:
                        #         build_report_part(os.path.join(e.outdir, e.pdf_name), [make_zeroth_row_header(header_text)], 0)
                        #     except:
                        #         log.debug("Exception calling build_report_part", exc_info=True)
            elif G.CONTINUUM_RULES:
                header_text = "Continuum Source"
                try:
                    build_report_part(os.path.join(e.outdir, e.pdf_name), [make_zeroth_row_header(header_text,redtext=e.red_header)], 0)
                except:
                    log.debug("Exception calling build_report_part", exc_info=True)

            if len(file_list) > 0:
                for f in file_list:
                    build_report(f.pages,f.filename)
            else:
                build_report(pages,args.name)

            if PyPDF is not None:
                if len(file_list) > 0:
                    try:
                        for f in file_list:
                            join_report_parts(f.filename,f.bid_count)
                            delete_report_parts(f.filename)
                    except:
                        log.error("Joining PDF parts failed for %s" %f.filename,exc_info=True)
                else:
                    join_report_parts(args.name)
                    delete_report_parts(args.name)



            if G.LyC or G.DeblendSpectra:
                #top level call to fetch the neighbor spectra for each of the detection objects
                try:
                    for hd in hd_list:
                        for e in hd.emis_list:
                            if e.status >=0:
                                #there can be multiple filters and multiple catalog/instruments
                                #we only want ONE instance of each neighbor
                                e.unique_sep_neighbors() #returnd list of sep objects (list of dictionaries)
                                #this is also a property of the DetObj (e.neighbors_sep_list)
                                #so can work on that list
                                #so go ahead and fetch the spectra for each entry
                                if (e.neighbors_sep is not None) and (e.neighbors_sep['sep_objects'] is not None):
                                    log.info(f"Forced extraction of {len(e.neighbors_sep['sep_objects'])} neighbor positions...")
                                    for n in e.neighbors_sep['sep_objects']:
                                        e.neighbor_forced_extraction(n,filter=e.neighbors_sep['filter_name'],
                                                                     catalog_name=e.neighbors_sep['catalog_name']) #populates the spectrum

                except:
                    log.error("Exception! Exception building LyC project Neighbor spectra. Top level.",exc_info=True)


                ##########################
                # PSF Spectra Deblending
                ##########################
                try:
                    log.info("Begninning spectra PSF deblending.")
                    try:
                        if args.aperture is not None:
                            aperture = args.aperture
                        else:
                            aperture = 3.5
                    except:
                        aperture = 3.5

                    #roughly
                    # 1. select the neighbors we want to deblend (take care to select out to 2x aperture and don't double
                    #    count the HETDEX candidate)
                    # 2. and build out the separation matrix (includes our target, but don't double count it)
                    # 3. correct their spectra (fix any "holes" and populate those without spectra from flat fnu from magnitude and band)
                    #
                    for hd in hd_list:
                        for e in hd.emis_list:
                            if e.status < 0:
                                continue #go to the next one

                            #pre-scan for large objects that would preclude the deblending???
                            #conduct anyway, but flag (maybe in the deblend table)
                            #this would be any 0 < dist_curve < 2*aperture with 'major' > seeing fwhm? or maybe a fixed 1.5" or 1.7" or 2.0" ???
                            if e.neighbors_sep is None or len(e.neighbors_sep)==0 or \
                                    e.neighbors_sep['sep_objects'] is None or len(e.neighbors_sep['sep_objects'])==0:
                                log.info(f"({e.entry_id}) No neighbors to trigger deblending. Will report spectrum as is.")
                                e.deblended_flux = e.sumspec_flux
                                e.deblended_fluxerr = e.sumspec_fluxerr
                                e.deblended_gmag = e.best_gmag
                                e.deblended_gmag_unc = e.best_gmag_unc
                                e.deblended_gmag_cont = e.best_gmag_cgs_cont
                                e.deblended_gmag_cont_unc = e.best_gmag_cgs_cont_unc
                                continue


#
# HERE !!! issue with sel ... the lengths can be different later .... might not want sel applied to ap_mag, etc right
# here as they can then become shorted than 'sep_objects' .... alternately make a copy of e.neighbors_sep['sep_objects']
# and cut it down by [sel] and use the copy going forward and not the original
#
                            try:
                                dcurve = np.array([x['dist_curve'] for x in e.neighbors_sep['sep_objects']])
                                dist_sel = dcurve <= 2*aperture
                                if np.count_nonzero(dist_sel) == 0:
                                    continue

                                sep_objects = np.array(e.neighbors_sep['sep_objects'])[dist_sel]
                                major = np.array([x['a'] for x in sep_objects])
                                ap_mag = np.array([x['mag'] for x in sep_objects])
                                ap_mag_err = np.array([x['mag_err'] for x in sep_objects])
                                #ap_mag_limit = get the single maglimit from the e.neighbors_sep ... not from the h5 table
                                dex_mag =  np.array([x['dex_g_mag'] if x['dex_g_mag'] < G.HETDEX_CONTINUUM_MAG_LIMIT
                                                     else G.HETDEX_CONTINUUM_MAG_LIMIT for x in sep_objects])
                                dex_mag_err =  np.array([x['dex_g_mag'] for x in sep_objects])
                                #check vs the corresponding survey seeing or the hetdex seeing?
                                #... since we are re-extracting, point-source like for HETDEX should be on the HETDEX seeing
                                # (e.survey_fwhm IS the HETDEX seeing)
                                if np.count_nonzero(major > 2.0 * e.survey_fwhm) > 0:
                                    e.deblended_flags |= G.DETFLAG_LARGE_NEIGHBOR

                                #get selection for the target object
                                target_sel = np.array([x['selected'] for x in sep_objects])

                                if np.count_nonzero(target_sel) != 1:
                                    target_mag = e.best_gmag
                                    target_mag_err = e.best_gmag_unc
                                    target_mag_filter = 'g'
                                else:
                                    idx = np.where(target_sel)[0][0]
                                    target_mag = sep_objects[idx]['mag']
                                    target_mag_err = sep_objects[idx]['mag_err']
                                    target_mag_filter = e.neighbors_sep['filter_name']


                                #should be 1 or none where selected is True (so only a single -1 at most)
                                #just apply the logic to the ra and select on it
                                sm_ra = np.array([-1 if (x['selected'] or x['dist_baryctr'] > 2*aperture) else x['ra']
                                                                                        for x in sep_objects])
                                sm_dec = np.array([x['dec'] for x in sep_objects])
                                neighbor_sel = sm_ra != -1
                                sm_ra = sm_ra[neighbor_sel]
                                sm_dec = sm_dec[neighbor_sel]

                                log.info(f"Building {len(sm_ra)+1}x{len(sm_ra)+1} separation matrix from {len(dcurve)} recoreded neighbors ...")
                                separation_matrix, rc = SU.build_separation_matrix([e.ra] + list(sm_ra), [e.dec]+ list(sm_dec)) #only out to 2x aperture since that is as far as we set PSF weights
                                if separation_matrix is None: #this is a problem and we are done
                                    if rc == 0:
                                        log.info(f"({e.entry_id}) No neighbors to trigger deblending. Will report spectrum as is.")
                                        e.deblended_flux = e.sumspec_flux
                                        e.deblended_fluxerr = e.sumspec_fluxerr

                                        e.deblended_gmag = e.best_gmag
                                        e.deblended_gmag_unc = e.best_gmag_unc
                                        e.deblended_gmag_cont = e.best_gmag_cgs_cont
                                        e.deblended_gmag_cont_unc = e.best_gmag_cgs_cont_unc

                                    else:
                                        log.error(f"({e.entry_id}) Separation Matrix failure {rc}. Cannot conduct deblending. Setting all zeroes.")
                                    #get all zeroe?
                                else:

                                    #these are wavelength dependent, (Kolmogorov relation: Seeing FWHM propto lambda^-0.2)
                                    #Karl's HETDEX paper sec 6.15 and (Roddier 1981)
                                    # So, we will get an array of key wavelength PSFs and at the end
                                    # linearly interpolate between them for the solution
                                    # The default seeing_fwhm is figures at 4540AA
                                    #Note: for the interpolations, we do have to anchor at the ends. Extrapolation is not allowed.
                                    #key_waves = [3470.,3600.,3800.,4000.,4200.,4400.,4540.,4600.,4800.,5000.,5200.,5400.,5540.]
                                    key_waves = [3470., 4000., 4540., 5000., 5540.]
                                    anchor_wave = 4540.

                                    #compute the PSF and the overlap_matrix
                                    log.info(f"Building PSF overlap matrices ...")

                                    #psf_list = []
                                    overlap_matrix_list = []

                                    for wave in key_waves:
                                        log.debug(f"Building PSF overlap matrix at {wave:0.1f}AA ...")
                                        psf = SU.get_psf(e.survey_fwhm * (wave/anchor_wave)**(-0.2), ap_radius=aperture,
                                                         max_sep=3*aperture, scale=0.25,normalize=True)
                                        overlap_matrix = SU.psf_overlap(psf, separation_matrix)
                                        overlap_matrix_list.append(overlap_matrix)

                                        #sanity check: only valid check if the PSF volume is normalized
                                        if np.count_nonzero(overlap_matrix > 1.0):
                                            log.warning(f"Warning. Deblending overlap matrix for {wave:0.1f}AA is invalid. Has fractions > 1")
                                            e.deblended_flags |= G.DETFLAG_BLENDED_SPECTRA

                                    overlap_matrix_list = np.array(overlap_matrix_list) #key_waves x N x N
                                    #now, build a single overlap_tensor from the list by iterpolating between the key waves
                                    # THIS overlap_tensor is no longer an NxN matrix adds a 3rd dimension that is the per wavelength overlap
                                    # iterate over all pairs (it is symmetric) of objects (i,j) and build a 1036 long array of floats
                                    # for the fractional overlap of that pair per wavelength by using the 13 key_waves
                                    # and interpolating between them
                                    N = np.shape(separation_matrix)[0] #NxN
                                    #it is symmetric and the diagnonal is all 1's, so this is not necessary
                                    #but for the moment, I am coding it out just to keep it all straight
                                    #overlap_tensor = np.full((N,N,len(G.CALFIB_WAVEGRID)),0.0)
                                    overlap_tensor = np.full((N,N),None)
                                    #full_overlap = np.full(len(G.CALFIB_WAVEGRID),1.0)
                                    #for i in range(N):
                                    #    overlap_tensor[i,i] = full_overlap
                                    for i in np.arange(1,N,1): # don't need the i = j cells as they are all 1 (so no 0,0, 1,1, etc)
                                        for j in np.arange(0,i):
                                            #now have to get all the key wavelenghts
                                            key_fracs = overlap_matrix_list[:,i,j]
                                            #overlap_tensor[i,j] = np.interp(G.CALFIB_WAVEGRID,key_waves,key_fracs)
                                            overlap_tensor[i][j] = np.interp(G.CALFIB_WAVEGRID,key_waves,key_fracs)
                                            #overlap_tensor[j,i] = overlap_tensor[i,j] #again, symmetric so this is not necessary

                                            # #testing
                                            # plt.close('all')
                                            # plt.plot(G.CALFIB_WAVEGRID,overlap_tensor[i,j])
                                            # plt.savefig(f"overlap_tensor_{i}_{j}.png")


                                    #originally in LyCon project this was part of the deblending, but here we have pulled it out
                                    #as a separate step

                                    #make the lists of spectra (fluxex) and errors
                                    measured_fluxes = np.array([x['flux'] for x in np.array(sep_objects)[neighbor_sel]])
                                    measured_flux_errs = np.array([x['flux_err'] for x in np.array(sep_objects)[neighbor_sel]])
                                    measured_mags = np.array([x['mag'] for x in np.array(sep_objects)[neighbor_sel]])
                                    measured_mag_errs = np.array([x['mag_err'] for x in np.array(sep_objects)[neighbor_sel]])
                                    #measured_mag_filters = np.full(len(measured_fluxes),e.neighbors_sep['filter_name'])
                                    #measured_mag_filters[0] = 'g' #for the HETDEX gmag of the target

                                    #should we reset those neighbor fluxes where the dex_g is fainter than the figured limit (or fixed at 24.5 or 25.0)
                                    #to be all zero s|t they will be replaced with the mag based flat_fnu spectra ?
                                    #otherwise are we just subtracting noise? .... is that okay? since it is based on the error as well ?
                                    #and / or if there is no detection above our limit, but we have spectra, wouldn't that just then be
                                    #part of the normal background and would have been (1) mostly handled as part of ffsky and (2)
                                    #the rest handled with a residual subtraction?

                                    # zero out the spectra of those that are fainter than the mag limit (using g or r as sufficient to compare
                                    # to dex-g
                                    s2 = ap_mag[neighbor_sel] > dex_mag[neighbor_sel]
                                    log.info(f"Replacing {np.count_nonzero(s2)} spectra with flat fnu as too faint for HETDEX.")
                                    measured_fluxes[s2] =  np.zeros(len(G.CALFIB_WAVEGRID))
                                    measured_flux_errs[s2] =  np.zeros(len(G.CALFIB_WAVEGRID))

                                    ##############
                                    #debug
                                    #############
                                    #np.savetxt("preflux.txt",measured_fluxes)
                                    #np.savetxt("preflux_err.txt",measured_fluxes)

                                    log.info(f"Filling in spectra ...")
                                    for i in range(len(measured_fluxes)): #todo: should we also check (in patch_holes) that the spectra is not grossly deformed?
                                        measured_fluxes[i], measured_flux_errs[i] = SU.patch_holes_in_hetdex_spectrum(G.CALFIB_WAVEGRID,
                                                                                                            measured_fluxes[i],
                                                                                                            measured_flux_errs[i],
                                                                                                            measured_mags[i],
                                                                                                            measured_mag_errs[i],
                                                                                                            e.neighbors_sep['filter_name'])

                                    #and the target candidate
                                    target_flux, target_flux_err = SU.patch_holes_in_hetdex_spectrum(G.CALFIB_WAVEGRID,
                                                                                                                  e.sumspec_flux,
                                                                                                                  e.sumspec_fluxerr,
                                                                                                                  target_mag,
                                                                                                                  target_mag_err,
                                                                                                                  target_mag_filter)
                                    ##############
                                    #debug
                                    #############
                                    #np.savetxt("postflux.txt",measured_fluxes)
                                    #np.savetxt("postflux_err.txt",measured_fluxes)
                                    #np.savetxt("overlap_matrix.txt",overlap_matrix)
                                    #np.savetxt("separation_matrix.txt",separation_matrix)

                                    num_mc = 1000
                                    true_flux_matrix_list = []
                                    log.info(f"Running {num_mc} deblending samples ...")

                                    #add back in the target spectrum
                                    measured_fluxes = np.vstack((target_flux,measured_fluxes))
                                    measured_flux_errs = np.vstack((target_flux_err,measured_flux_errs))

                                    #zero_check_matrix = np.zeros(np.shape(measured_fluxes))
                                    #set first row  (the target fluxes) to not be checked
                                    #zero_check_matrix[0] = np.full(len(G.CALFIB_WAVEGRID),-np.inf)

                                    for i in range(num_mc):
                                        iter_measured_flux = np.random.normal(measured_fluxes, measured_flux_errs)
                                        #just a test ... maybe should only zero those that are not row 0 (the target flux)?
                                        #iter_measured_flux = np.clip(iter_measured_flux[1:], a_min = 0, a_max = None)
                                        #zero_sel = iter_measured_flux < zero_check_matrix
                                        #iter_measured_flux[zero_sel] = 0

                                        true_flux_matrix = SU.spectra_deblend(iter_measured_flux, overlap_tensor)
                                        true_flux_matrix_list.append(true_flux_matrix)

                                    true_flux_matrix_list = np.array(true_flux_matrix_list)
                                    # average over the list
                                    # really only need the zeroth entry as the LAE candidate
                                    deblended_matrix = true_flux_matrix_list[:, 0, :] #indicies are 1st: each run,  2nd: which target, 3rd: flux bins
                                    e.deblended_flux = np.mean(deblended_matrix, axis=0)
                                    e.deblended_fluxerr = np.std(deblended_matrix, axis=0)

                                    # #testing
                                    # plt.close('all')
                                    # plt.plot(G.CALFIB_WAVEGRID,e.deblended_flux,label="Deblended")
                                    # plt.plot(G.CALFIB_WAVEGRID, e.sumspec_flux, label="Original")
                                    # plt.legend()
                                    # plt.savefig(f"deblend_test.png")

                                    e.deblended_gmag, e.deblended_gmag_unc, e.deblended_gmag_cont, e.deblended_gmag_cont_unc = \
                                                    elixer_spectrum.get_best_gmag(e.deblended_flux/G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS,
                                                                    e.deblended_fluxerr/G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS,
                                                                    G.CALFIB_WAVEGRID)
                                    log.info("Deblending complete.")

                                    #SU.spectra_deblend  ... maybe ~ 1000 iterations, sampling over errors and then take biweight "Avg"?
                                    #                    ... that should handle the error propogation issues (use std as error)
                                    #                    ... maybe as an option in the global_config, set iterations = 1 if just once?#
                                    #!!!NOTE!!! this will NOT have the shot average background residual subtracted
                                    #
                                    # maybe modify the dex mag limit check to get entire IFU array for shot and figure the residual to subtract?
                                    # (as a new function, similar to the dex mag limit check)
                            except:
                                log.error("Exception (inner) while deblending spectrum.", exc_info=True)

                except:
                    log.error("Exception (outer) while deblending spectrum.",exc_info=True)


            if G.BUILD_HDF5_CATALOG: #change to HDF5 catalog
                try:
                    #check flags ... only get recorded in HDF5, so only do that here
                    for hd in hd_list:
                        for e in hd.emis_list:
                            if not e.full_flag_check_performed:
                                e.flag_check()

                    h5name = os.path.join(args.name, args.name + "_cat.h5")
                    elixer_hdf5.extend_elixer_hdf5(h5name,hd_list,overwrite=True)
                    for hd in hd_list:
                        for e in hd.emis_list:
                            if e.status >=0:
                                d_id = e.hdf5_detectid
                                entry_ct = elixer_hdf5.detectid_in_file(h5name,d_id)
                                if entry_ct != 1:
                                    log.warning(f"Unexpected number of entries ({entry_ct}) in h5 file for detectid {d_id}, file {h5name}")
                                    if entry_ct == 0:
                                        log.warning(f"Retry insertion into h5 file. detectid = {d_id}")
                                        elixer_hdf5.extend_elixer_hdf5(h5name, [hd], overwrite=True)
                                        entry_ct = elixer_hdf5.detectid_in_file(h5name, d_id)
                                        if entry_ct != 1:
                                            log.warning(f"No retry: Unexpected number of entries ({entry_ct}) in h5 file for detectid {d_id}, file {h5name}")
                except:
                    log.error("Exception building HDF5 catalog",exc_info=True)

            if False: #turn off fib and cat.txt files
                if match_list.size > 0:
                    match_list.write_file(os.path.join(args.name,args.name+"_cat.txt"))

                write_fibers_file(os.path.join(args.name, args.name + "_fib.txt"),hd_list)


            #todo: iterate over detections and make a clean sample for LyC
            #conditions ...
            #   exactly 1 catalog detection
            #   all 3 P(LAE)/P(OII) of similar value (that is all > say, 10)
            #todo: OR - record all in HDF5 and subselect later
            #not an optimal search (lots of redundant hits, but there will only be a few and this is simple to code
            if False:
                for h in hd_list: #iterate over all hetdex detections
                    for e in h.emis_list:
                        for c in match_list.match_set: #iterate over all match_list detections
                            if e.id == c.detobj.id: #internal ID (guaranteed unique)

                                print("Detect ID",c.detobj.entry_id)
                                for b in c.bid_targets:
                                    print("         ",b.p_lae_oii_ratio,b.distance)
                                print("\n")

                                if False:
                                    if len(c.bid_targets) == 2: #the HETDEX + imaging, and 1 catalog match
                                        #c.bid_targets[0] is the hetdex one
                                        #check all plae_poii
                                        if (c.detobj.p_lae_oii_ratio > 10)          and \
                                           (c.bid_targets[0].p_lae_oii_ratio > 10) and \
                                           (c.bid_targets[1].p_lae_oii_ratio > 10):
                                            #meets criteria, so log
                                            if c.bid_targets[1].distance < 1.0:
                                                print(c.detobj.id,c.detobj.p_lae_oii_ratio,c.bid_targets[0].p_lae_oii_ratio,c.bid_targets[1].p_lae_oii_ratio )
                                break

            if args.line:
                try:
                    import shutil
                    shutil.copy(args.line,os.path.join(args.name,os.path.basename(args.line)))
                except:
                    log.error("Exception copying line file: ", exc_info=True)

            if (args.jpg or args.png) and (PyPDF is not None):
                if len(file_list) > 0:
                    for f in file_list:
                        if (G.LAUNCH_PDF_VIEWER is not None) and args.viewer:
                            viewer_file_list.append(f.filename)

                        try:
                            convert_pdf(f.filename,jpeg=args.jpg, png=args.png)
                        except:
                            log.error("Error (2) converting pdf to image type: " + f.filename, exc_info=True)
                else:
                    if (G.LAUNCH_PDF_VIEWER is not None) and args.viewer:
                        viewer_file_list.append(args.name + ".pdf")

                    try:
                        convert_pdf(args.name + ".pdf",jpeg=args.jpg, png=args.png)
                    except:
                        log.error("Error (3) converting pdf to image type: " + f.filename, exc_info=True)
            else: #no conversion, but might still want to launch the viewer
                if len(file_list) > 0:
                    for f in file_list:
                        if (G.LAUNCH_PDF_VIEWER is not None) and args.viewer:
                            viewer_file_list.append(f.filename)
                else:
                    if (G.LAUNCH_PDF_VIEWER is not None) and args.viewer:
                        viewer_file_list.append(args.name + ".pdf")

        #do neighborhood 1st so can use broad cutout for --mini
        nei_mini_buf = None
        if G.ZOO_MINI or ((args.neighborhood is not None) and (args.neighborhood > 0.0)):
            if ((args.neighborhood is not None) and (args.neighborhood > 0.0)):
                msg = "Building neighborhood at (%g\") for all detections ...." % (args.neighborhood)
                log.info(msg)
                print(msg)
            for h in hd_list:  # iterate over all hetdex detections
                for e in h.emis_list:
                    if args.neighborhood_only:
                        pdf = PDF_File(args.name, e.entry_id, e.pdf_name)
                        e.outdir = pdf.basename
                        # update pdf_name to match
                        try:
                            e.pdf_name = os.path.basename(pdf.filename)
                        except:
                            pass  # not important if this fails

                    if e.wra is not None:
                        ra = e.wra
                        dec = e.wdec
                    else:
                        ra = e.ra
                        dec = e.dec

                    try:
                        if e.entry_id >= 1e9:
                            nei_name = os.path.join(pdf.basename, str(e.entry_id) + "_nei.png")
                        else:
                            nei_name = os.path.join(pdf.basename, e.pdf_name.rstrip(".pdf") + "_nei.png")

                        if e.fwhm < 0:
                            wave_range = [e.w-12.5,e.w+12.5]
                        else:
                            wave_range = [e.w-e.fwhm*3/2.355,e.w+e.fwhm*3/2.355]

                        _, nei_mini_buf, line_mini_buf = build_neighborhood_map(hdf5=args.hdf5, cont_hdf5=G.HDF5_CONTINUUM_FN,
                                           detectid=None, ra=ra, dec=dec, distance=args.neighborhood, cwave=e.w,
                                           fname=nei_name, original_distance=args.error,
                                           this_detection=e if explicit_extraction else None,
                                           broad_hdf5=G.HDF5_BROAD_DETECT_FN,
                                           primary_shotid=e.survey_shotid,
                                           wave_range=wave_range)

                        e.nei_mini_buf = nei_mini_buf
                        e.line_mini_buf = line_mini_buf
                    except:
                        log.warning("Exception calling build_neighborhood_map.",exc_info=True)

            if len(hd_list) == 0: #there were not any hetdex detections to anchor, just use RA, Dec?
                if (args.ra is not None) and (args.dec is not None):
                    try:
                        _, nei_mini_buf, line_mini_buf = build_neighborhood_map(hdf5=args.hdf5, cont_hdf5=G.HDF5_CONTINUUM_FN,
                                           detectid=None, ra=args.ra, dec=args.dec, distance=args.neighborhood,
                                           cwave=None,
                                           fname=os.path.join(args.name, args.name + "_nei.png"),
                                           original_distance=args.error,
                                           this_detection=None,
                                           broad_hdf5=G.HDF5_BROAD_DETECT_FN)
                    except:
                        log.warning("Exception calling build_neighborhood_map.",exc_info=True)



        if G.ZOO_MINI and not args.neighborhood_only:
            msg = "Building ELiXer-lite summary images for all detections ...."
            log.info(msg)
            print(msg)
            for h in hd_list:  # iterate over all hetdex detections
                for e in h.emis_list:
                    if e.entry_id >= 1e9:
                        mini_name = os.path.join(pdf.basename, str(e.entry_id) + "_mini.png")
                    else:
                        mini_name = os.path.join(pdf.basename, e.pdf_name.rstrip(".pdf") + "_mini.png")

                    build_3panel_zoo_image(fname=mini_name,
                                           image_2d_fiber=e.image_2d_fibers_1st_col,
                                           image_1d_fit=e.image_1d_emission_fit,
                                           image_cutout_fiber_pos=e.image_cutout_fiber_pos,
                                           image_cutout_neighborhood=e.nei_mini_buf,
                                           image_cutout_fiber_pos_size=args.error,
                                           image_cutout_neighborhood_size=args.neighborhood,
                                           line_image_cutout=e.line_mini_buf)


        # really has to be here (hd_list is reset on each loop and the "recover" will not work otherwise)
        # (if we run all of them at the end, since "--recover" looks at the .pdf results, these will not run
        # if the pdf was complete, but these plots were not)
        # even so (and this is true of the neighborhood and mini as well, but to a lesser extent as they are faster)
        # for the last PDF to be completed, if SLURM times out, this would be missed and not re-run
        if args.gridsearch:
            if len(args.gridsearch) != 5:
                log.warning(f"Invalid gridsearch parameter ({args.gridsearch})")
            else:
                log.debug("Preparing for gridsearch ...")

                # work around for local launcher when using gridsearch
                # normally this executes for all at the end, but assuming that we are only running one or a few,
                # we the auto-viewer to pop up before the gridsearch plots so we have a reference
                if (G.LAUNCH_PDF_VIEWER is not None) and args.viewer and (
                        len(viewer_file_list) > 0) and not args.neighborhood_only:
                    import subprocess
                    already_launched_viewer = True
                    cwd = os.getcwd()
                    cmdlist = [G.LAUNCH_PDF_VIEWER]
                    launch = False
                    for f in viewer_file_list:
                        fpath = os.path.join(cwd, f)
                        if os.path.exists(fpath):
                            cmdlist.append(fpath)
                            launch = True

                    if launch:
                        subprocess.Popen(cmdlist)


                for h in hd_list:  # iterate over all hetdex detections
                    for e in h.emis_list:
                        try:
                            if e.wra:
                                ra = e.wra
                                dec = e.wdec
                            else:
                                ra = e.ra
                                dec = e.dec

                            if args.wavelength:
                                cw = args.wavelength
                            else:
                                cw = e.w

                            if e.pdf_name:
                                savefn = os.path.join(e.outdir, e.pdf_name.rstrip(".pdf"))
                            else:
                                savefn = os.path.join(e.outdir, str(e.entry_id))

                            if e.survey_shotid: #use the shot from the DetObj
                                shotlist = [e.survey_shotid]
                            elif args.command_line_shotid:
                                #this could have been modified upstream and may not be the original
                                #(note: this is not exactly as it appears on the command line, but is immediatelu
                                #after it has been transformed into the common, integer format
                                shotlist = [args.command_line_shotid]
                            else:
                                shotlist = SU.get_shotids(ra, dec)

                            ra_meshgrid, dec_meshgrid = SU.make_raster_grid(ra, dec, args.gridsearch[0],
                                                                            args.gridsearch[1])

                            x, y = np.shape(ra_meshgrid)
                            log.info(f"{e.entry_id} gridsearch ({ra},{dec},{cw}) at {x}x{y}x{len(shotlist)}")

                            edict = SU.raster_search(ra_meshgrid, dec_meshgrid, shotlist, cw,
                                                     max_velocity=args.gridsearch[2],max_fwhm=args.gridsearch[3],aperture=3.0,
                                                     ffsky=args.ffsky)
                            #show most common (others are available via direct call to the saved py file)
                            z = SU.make_raster_plots(edict, ra_meshgrid, dec_meshgrid, cw,"meanflux_density",
                                                          save=savefn,savepy=savefn,show=args.gridsearch[4])
                            #don't know how meaningful this really is, given that this is a single source (not a stack)
                            #and the f900 would be at the 0.01 S/N level
                            # z = SU.make_raster_plots(edict, ra_meshgrid, dec_meshgrid, cw,
                            #                          "f900", show=args.gridsearch[3], save=savefn)
                            # z = SU.make_raster_plots(edict, ra_meshgrid, dec_meshgrid, cw,
                            #                          "velocity_offset", show=args.gridsearch[3], save=None,savepy=savefn)
                            # z = SU.make_raster_plots(edict, ra_meshgrid, dec_meshgrid, cw,
                            #                          "continuum_level", show=args.gridsearch[3], save=None,savepy=savefn)
                        except:
                            log.info(f"Exception grid search {e.entry_id}", exc_info=True)

    #end for master_loop_idx in range(master_loop_length):


    if (G.LAUNCH_PDF_VIEWER is not None) and args.viewer and (len(viewer_file_list) > 0) \
            and not args.neighborhood_only and not already_launched_viewer:
        import subprocess
        cwd = os.getcwd()
        cmdlist = [G.LAUNCH_PDF_VIEWER]
        launch = False
        for f in viewer_file_list:
            fpath = os.path.join(cwd,f)
            if os.path.exists(fpath):
                cmdlist.append(fpath)
                launch = True

        if launch:
            subprocess.Popen(cmdlist)





    # if (args.neighborhood is not None) and (args.neighborhood > 0.0):
    #     msg = "Building neighborhood at (%g\") for all detections ...." %(args.neighborhood)
    #     log.info(msg)
    #     print(msg)
    #     for h in hd_list:  # iterate over all hetdex detections
    #         for e in h.emis_list:
    #             if e.wra is not None:
    #                 ra = e.wra
    #                 dec = e.wdec
    #             else:
    #                 ra = e.ra
    #                 dec = e.dec
    #
    #             build_neighborhood_map(hdf5=args.hdf5,cont_hdf5=G.HDF5_CONTINUUM_FN,
    #                                    detectid=None,ra=ra,dec=dec,distance=args.neighborhood,cwave=e.w,
    #                                    fname=os.path.join(pdf.basename,str(e.entry_id)+"nei.png"))
    #


    log.critical("Main complete.")

    exit(0)

# end main


if __name__ == '__main__':
    main()