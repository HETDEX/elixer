"""
simple script to collect the "empty" fibers for a single shot (all dithers) validating each against acceptance critieria,
and save to an astropy table

intended for use with SLURM

caller specifies only the shotid

loosely based on hetdex_all_shots_random_empty_apertures notebooks on /work/03261/polonius/notebooks

There is a basic check for continuum and emission lines at an absolute level and then variations on cuts/clips
overall, per wavelength bin, and/or per wavelength range

Then under those cuts, the fibers are stacked into a single average and recorded for the shot

"""
import sys
import os
import os.path as op
import datetime

survey_name = "hdr4" #"hdr4" #"hdr2.1"
TEST = False

##################################
# should we even run this one?
##################################

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#form is integer dateobs (no 'v')
if "--shot" in args:
    i = args.index("--shot")
    try:
        shotid = int(sys.argv[i + 1])
    except:
        print("bad shotid specified")
        exit(-1)
else:
    print("no shotid specified")
    exit(-1)
shot = shotid

if "--ffsky" in args:
    print("using ff sky subtraction")
    ffsky = True
else:
    print("using local sky subtraction")
    ffsky = False

if "--bothsky" in args:
    print("Using local sky subtraction selection. Recording both local and ffsky extractions.")
    print("Overriding --ffsky")
    bothsky = True
    ffsky = False
else:
    print("using local sky subtraction")
    bothsky = False

if "--fiber_corr" in args:
    print("Apply per fiber residual correction")
    per_fiber_corr = True
else:
    print("Do Not apply per fiber residual correction")
    per_fiber_corr = False


#dust is a bit dodgy here since it is RA, Dec and thus per fiber specific and IT DOES VARY from fiber to fiber
#albeit rather slowly
# if "--dust" in args:
#     print("recording dust correction")
#     dust = True
# else:
#     print("NO dust correction")
#     dust = False


print(f"{shotid} starting: {datetime.datetime.now()}")

table_outname = "empty_fibers_"+str(shotid) + "_ll.fits"
table_outname2 = "empty_fibers_"+str(shotid) + "_ff.fits"


#maybe this one was already done?
# which files to check really depends on --ffsky and --bothsky
    #did they all copy or did it timeout during a copy?

if bothsky:
    if op.exists(table_outname) and op.exists(table_outname2):
        #all good
        print(f"{shotid} --bothsky outputs already exist. Exiting. {datetime.datetime.now()}")
        exit(0)
elif ffsky:
    if op.exists(table_outname2):
        #all good
        print(f"{shotid} --ffsky output already exists. Exiting. {datetime.datetime.now()}")
        exit(0)
else:
    if op.exists(table_outname):
        #all good
        print(f"{shotid} output already exist. Exiting. {datetime.datetime.now()}")
        exit(0)

if op.exists(table_outname) or op.exists(table_outname2):
    print(f"{shotid} incomplete. Reset and retry. {datetime.datetime.now()}")

    try:
        if not ffsky or bothsky:
            os.remove(table_outname)
    except:
        pass

    try:
        if ffsky or bothsky:
            os.remove(table_outname2)
    except:
        pass



#####################################
# OK. This one needs to be run, so
# continue loading packages, etc
#####################################

print(f"{shotid} load ....  {datetime.datetime.now()}")

import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
from astropy.stats import sigma_clip
import astropy.stats.biweight as biweight
import astropy.units as u
import copy
import glob
import shutil

from hetdex_api.config import HDRconfig
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import Survey,FiberIndex
#from hetdex_tools.get_spec import get_spectra
from hetdex_api.extinction import *  #includes deredden_spectra
from hetdex_api.extract import Extract

from elixer import spectrum as elixer_spectrum
from elixer import spectrum_utilities as SU
from elixer import global_config as G
from elixer import catalogs

survey_name = "hdr4" #"hdr4" #"hdr2.1"

#tmppath = "/tmp/hx/"
#tmppath = "/home/dustin/temp/random_apertures/hx/"

# if not os.path.exists(tmppath):
#     try:
#         os.makedirs(tmppath, mode=0o755)
#         if not os.access(tmppath, os.W_OK):
#             print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
#             exit(-1)
#     except: #can catch timing on creation if multiples are trying to make it
#         if not os.access(tmppath, os.W_OK):
#             print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
#             exit(-1)

#########################################
# Helper Funcs
#########################################

def linear_g(seeing): #estimate of depth from the seeing (based on HDR 3.0.3 detections)
    # return seeing * (-31./70.) + 25.76  # middle of the y_err
    #return seeing * (-31. / 70.) + 25.56  # bottom of the y_err
    return SU.estimated_depth(seeing)


def split_spectra_into_bins(fluxd_2d, fluxd_err_2d, sort=True, trim=0.9):
    """
    given 2d array of spectra (G.CALFIB_WAVEGRID wavelengths assumed) in flux_density,
    split and retun as a dictionary of 5 bins (10 entries ... 5bins x 2 for fluxd and fluxd_error)
    :param fluxd_2d:
    :param fluxd_err_2d:
    :param sort: if True, sort under each bin (independently)
    :param trim: if 0 to 1.0, keep the bottom "trim" fraction of the sorted sample (only if sort is True)
    :return:
    """

    rd = {}  # return dict

    bin_idx_ranges = [(15, 195), (195, 400), (400, 605), (605, 810), (810, 1015)]  # (inclusive-exclusive)
    full_idx_ranges = [(0, 195), (195, 400), (400, 605), (605, 810),
                       (810, 1036)]  # 1st and last different to cover full 1036

    # todo: when appending, though need the full width, so 1st bin needs to be (0,195) and the last (810,1036)

    for i, ((left, right), (full_left, full_right)) in enumerate(zip(bin_idx_ranges, full_idx_ranges)):
        f = fluxd_2d[:, left:right]
        e = fluxd_err_2d[:, left:right]
        r = (full_left, full_right)
        ff = fluxd_2d[:, full_left:full_right]
        ef = fluxd_err_2d[:, full_left:full_right]

        sel_e = np.isnan(e)
        sel_ef = np.isnan(ef)
        f[sel_e] = np.nan
        ff[sel_ef] = np.nan

        if sort:
            md = np.nanmedian(fluxd_2d, axis=1)
            idx = np.argsort(md)  # ascending order, smallest average flux to largest

            if trim is not None and (0 < trim < 1.0):
                max_idx = int(len(md) * trim)
            else:
                max_idx = len(md)
            f = f[idx[0:max_idx]]
            e = e[idx[0:max_idx]]
            ff = ff[idx[0:max_idx]]
            ef = ef[idx[0:max_idx]]

        rd[f"f{i + 1}"] = copy.copy(ff)  # flux density in bin range
        rd[f"e{i + 1}"] = copy.copy(ef)  # flux density error in bin range
        rd[f"r{i + 1}"] = copy.copy(r)  # bin range indicies (inclusive index, exclusive index)

    return rd


def stack_by_wavebin_bw(fluxd_2d, fluxd_err_2d, trim=1.00, sc=None, ir=None):
    """
    stack down each wavelength bin, trimming under each

    sc = sigma_clip (symmetric)
    ir = internal ratio (e.g. interior 2/3)

    """

    stack = np.zeros(len(G.CALFIB_WAVEGRID))
    stack_err = np.zeros(len(G.CALFIB_WAVEGRID))
    contrib = np.zeros(len(G.CALFIB_WAVEGRID))
    N = len(fluxd_2d)

    for i in range(1036):
        # only consider those that have an associated error
        # errors of < 0 are flagged as bad pixels, cosmic strikes, etc
        sel = np.array(fluxd_err_2d[:, i] > 0) & np.array(~np.isnan(fluxd_err_2d[:, i])) & np.array(
            ~np.isnan(fluxd_2d[:, i]))
        if np.count_nonzero(sel) == 0:
            stack[i] = 0
            stack_err[i] = 0
            continue

        if trim < 1.0:
            idx = np.argsort(fluxd_2d[:, i][sel])
            max_idx = int(len(idx) * trim)
            column = fluxd_2d[:, i][sel][idx][0:max_idx]
            column_err = fluxd_err_2d[:, i][sel][idx][0:max_idx]
        elif sc is not None:
            column = fluxd_2d[:, i][sel]
            column_err = fluxd_err_2d[:, i][sel]
            mask = sigma_clip(column, sigma=sc)
            column = column[~mask.mask]
            column_err = column_err[~mask.mask]
        elif ir is not None:
            if ir >= 1.0:
                column = fluxd_2d[:, i][sel]
                column_err = fluxd_err_2d[:, i][sel]
            else:
                er = (1.0 - ir) / 2.0  # exterior ratio, e.g. if ir = 2/3 then er = 1/6 ... trim away the upper and lower 1/6
                idx = np.argsort(fluxd_2d[:, i][sel])
                low_idx = int(len(idx) * er)
                high_idx = int(len(idx) * (ir + er))
                column = fluxd_2d[:, i][sel][idx][low_idx:high_idx]
                column_err = fluxd_err_2d[:, i][sel][idx][low_idx:high_idx]
        # print(ir,low_idx,high_idx)

        elif trim == 1.0:
            column = fluxd_2d[:, i][sel]
            column_err = fluxd_err_2d[:, i][sel]
        else:
            return None, None, None

        try:
            stack[i] = biweight.biweight_location(column)
            stack_err[i] = biweight.biweight_scale(column)/np.sqrt(len(column))
            contrib[i] = len(column)
        except Exception as E:
            print(E)
            stack[i] = 0
            stack_err[i] = 0

    return stack, stack_err, contrib



def stack_by_waverange_bw(fluxd_2d, fluxd_err_2d, waverange = (3500,5500), trim=1.00, sc=None, ir=None):
    """

    stack by selection over the specified waverange
    unlike stack_by_wavebin_bw, here we cut out entire fibers and stack only complete fibers with whatever is left


    sc = sigma_clip (symmetric)
    ir = internal ratio (e.g. interior 2/3)

    """

    stack = np.zeros(len(G.CALFIB_WAVEGRID))
    stack_err = np.zeros(len(G.CALFIB_WAVEGRID))
    contrib = np.zeros(len(G.CALFIB_WAVEGRID))
    N = len(fluxd_2d)
    sel = None
    waveidx = np.array([0,0])
    waveidx[0],*_ = SU.getnearpos(G.CALFIB_WAVEGRID,waverange[0])
    waveidx[1], *_ = SU.getnearpos(G.CALFIB_WAVEGRID, waverange[1])

    #make an array against which to select
    #since these are all top cuts or sigma clips or interior ratios, it does not matter if these are sums or averages
    flux_sums = np.nansum(fluxd_2d[:,waveidx[0]:waveidx[1]+1],axis=1)

    if trim < 1.0:
        idx = np.argsort(flux_sums)
        max_idx = int(N * trim)
        sel = flux_sums < flux_sums[max_idx]
    elif sc is not None:
        mask = sigma_clip(flux_sums, sigma=sc)
        sel = ~mask.mask
    elif ir is not None:
        er = (1.0 - ir) / 2.0  # exterior ratio, e.g. if ir = 2/3 then er = 1/6 ... trim away the upper and lower 1/6
        idx = np.argsort(flux_sums)
        low_idx = int(N * er)
        high_idx = int(N * (ir + er))

        sel = np.array(flux_sums < flux_sums[high_idx])
        sel = sel & np.array(flux_sums < flux_sums[low_idx])
    elif trim == 1.0:
        sel = np.full(N,True)
    else:
        return None, None, None

    try:

        stack, stack_err, _, contrib = SU.stack_spectra(fluxd_2d[sel],fluxd_err_2d[sel],
                                                        np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(sel),1)),
                                                        grid=G.CALFIB_WAVEGRID, avg_type="biweight",
                                                        straight_error=False, std=False)
    except Exception as E:
        print(E)
        stack[i] = 0
        stack_err[i] = 0

    return stack, stack_err, contrib

#########################
# Main Logic
#########################

if not TEST:
    hetdex_api_config = HDRconfig(survey_name)
    survey = Survey(survey_name)
    print("Getting FiberIndex ...")
    FibIndex = FiberIndex(survey_name)

    survey_table=survey.return_astropy_table()
    sel = np.array(survey_table['shotid'] == shotid)
    seeing = float(survey_table['fwhm_virus'][sel])
    response = float(survey_table['response_4540'][sel])
else:
    seeing = 0.0
    response = 0.0

#############################################
# out put tables
#############################################

#local sky, ALL stacking statistics are biweight
D = {} #dummy dict for Table T
T = Table(dtype=[('shotid', int),('seeing',float),('response',float),

                 ('raw_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flagged fibers removed
                 ('raw_fluxd_err', (float, len(G.CALFIB_WAVEGRID))), #NO OTHER TRIMMING
                 ('raw_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### fixed continuum trim (applies to ALL below) ###
                 #this might be the biggest differentiator ... what do we define as continuum level?
                 #and should this be shot variable?

                 ('trim_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #keep all AFTER flag trim and absolute flux cut
                 ('trim_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('trim_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### top percent trim (applies only to this section) ###
                 ('t01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 1% per wavelength bin
                 ('t01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 2% per wavelength bin
                 ('t02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 3% per wavelength bin
                 ('t03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 4% per wavelength bin
                 ('t04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 5% per wavelength bin
                 ('t05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### sigma clips (applies only to this section) ###
                 #sigma clip of 1 is WAAAY too agreesive, 2 is maybe okay, but we'll just use 3,4,5

                 ('sc3_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 3-sigm clip per wavelength bin
                 ('sc3_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('sc3_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('sc5_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 5-sigm clip per wavelength bin
                 ('sc5_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('sc5_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### interior fraction (applies only to this section) ###

                 #these are all extememly similar; after trimming for continuum, the distributions
                 #are extemely Gaussian, so this particular variation makes almost no difference

                 # ('ir50_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 50% (trim off top and bottom 25%)
                 # ('ir50_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 # ('ir50_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 #roughly interior +/- 1 std (assuming a normal distro)
                 ('ir67_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 2/3 (trim off top and bottom 1/6)
                 ('ir67_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ir67_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 #roughly interior +/- 2 std (assuming a normal distro)
                 ('ir95_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 95% (trim off top and bottom 2.5%)
                 ('ir95_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ir95_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 # roughly interior +/- 3 std (assuming a normal distro)
                 ('ir99_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 99% (trim off top and bottom 0.5%)
                 ('ir99_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ir99_contrib', (float, len(G.CALFIB_WAVEGRID))),


                 #
                 # whole fiber selection
                 #

                 ### top percent trim (applies only to this section) ###
                 ('ft01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 1% per wavelength bin
                 ('ft01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ft01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('ft02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 2% per wavelength bin
                 ('ft02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ft02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('ft03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 3% per wavelength bin
                 ('ft03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ft03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('ft04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 4% per wavelength bin
                 ('ft04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ft04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('ft05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 5% per wavelength bin
                 ('ft05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ft05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### sigma clips (applies only to this section) ###
                 #sigma clip of 1 is WAAAY too agreesive, 2 is maybe okay, but we'll just use 3,4,5

                 ('fsc3_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 3-sigm clip per wavelength bin
                 ('fsc3_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fsc3_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('fsc5_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 5-sigm clip per wavelength bin
                 ('fsc5_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fsc5_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### interior fraction (applies only to this section) ###

                 #roughly interior +/- 1 std (assuming a normal distro)
                 ('fir67_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 2/3 (trim off top and bottom 1/6)
                 ('fir67_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fir67_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 #roughly interior +/- 2 std (assuming a normal distro)
                 ('fir95_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 95% (trim off top and bottom 2.5%)
                 ('fir95_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fir95_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 # roughly interior +/- 3 std (assuming a normal distro)
                 ('fir99_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 99% (trim off top and bottom 0.5%)
                 ('fir99_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fir99_contrib', (float, len(G.CALFIB_WAVEGRID))),


                 ])

#ffsky (if both)
#T2 = copy.deepcopy(T)
D2 = {} #dummy dict for Table T2
T2 = Table(dtype=[('shotid', int), ('seeing',float), ('response',float),

                  ('raw_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flagged fibers removed
                  ('raw_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),  #NO OTHER TRIMMING
                  ('raw_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### fixed continuum trim (applies to ALL below) ###
                  #this might be the biggest differentiator ... what do we define as continuum level?
                  #and should this be shot variable?

                  ('trim_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #keep all AFTER flag trim and absolute flux cut
                  ('trim_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('trim_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### top percent trim (applies only to this section) ###

                  ('t01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 1% per wavelength bin
                  ('t01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 2% per wavelength bin
                  ('t02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 3% per wavelength bin
                  ('t03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 4% per wavelength bin
                  ('t04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 5% per wavelength bin
                  ('t05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### sigma clips (applies only to this section) ###
                  #sigma clip of 1 is WAAAY too agreesive, 2 is maybe okay, but we'll just use 3,4,5

                  ('sc3_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 3-sigm clip per wavelength bin
                  ('sc3_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('sc3_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('sc5_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 5-sigm clip per wavelength bin
                  ('sc5_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('sc5_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### interior fraction (applies only to this section) ###

                  #these are all extememly similar; after trimming for continuum, the distributions
                  #are extemely Gaussian, so this particular variation makes almost no difference

                  # ('ir50_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 50% (trim off top and bottom 25%)
                  # ('ir50_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  # ('ir50_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  #roughly interior +/- 1 std (assuming a normal distro)
                  ('ir67_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 2/3 (trim off top and bottom 1/6)
                  ('ir67_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ir67_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  #roughly interior +/- 2 std (assuming a normal distro)
                  ('ir95_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 95% (trim off top and bottom 2.5%)
                  ('ir95_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ir95_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  # roughly interior +/- 3 std (assuming a normal distro)
                  ('ir99_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 99% (trim off top and bottom 0.5%)
                  ('ir99_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ir99_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  #
                  # whole fiber selection
                  #

                  ### top percent trim (applies only to this section) ###
                  ('ft01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% per wavelength bin
                  ('ft01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ft01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('ft02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2% per wavelength bin
                  ('ft02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ft02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('ft03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3% per wavelength bin
                  ('ft03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ft03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('ft04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4% per wavelength bin
                  ('ft04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ft04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('ft05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 5% per wavelength bin
                  ('ft05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('ft05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### sigma clips (applies only to this section) ###
                  # sigma clip of 1 is WAAAY too agreesive, 2 is maybe okay, but we'll just use 3,4,5

                  ('fsc3_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 3-sigm clip per wavelength bin
                  ('fsc3_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('fsc3_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('fsc5_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 5-sigm clip per wavelength bin
                  ('fsc5_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('fsc5_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### interior fraction (applies only to this section) ###

                  # roughly interior +/- 1 std (assuming a normal distro)
                  ('fir67_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 2/3 (trim off top and bottom 1/6)
                  ('fir67_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('fir67_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  # roughly interior +/- 2 std (assuming a normal distro)
                  ('fir95_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 95% (trim off top and bottom 2.5%)
                  ('fir95_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('fir95_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  # roughly interior +/- 3 std (assuming a normal distro)
                  ('fir99_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 99% (trim off top and bottom 0.5%)
                  ('fir99_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('fir99_contrib', (float, len(G.CALFIB_WAVEGRID))),
                  ])



if not TEST:
#if True:
    print(f"{shotid} get_fibers_table() ....  {datetime.datetime.now()}")
    fibers_table = get_fibers_table(shot)
    print(f"{shotid} [DONE] get_fibers_table() ....  {datetime.datetime.now()}, # rows = {len(fibers_table)}")
    #drop the columns we don't care about to save memory
    fibers_table.keep_columns(['fiber_id','calfib','calfib_ffsky','calfibe','fiber_to_fiber','trace','chi2']) #have to keep the fiber_id for the moment

    start = f'{shot}_0'
    stop = f'{shot}_9'
    print(f"{shotid} mask table ....  {datetime.datetime.now()}")
    mask_table = Table(FibIndex.fibermaskh5.root.Flags.read_where("(fiber_id > start) & (fiber_id < stop)"))
    print(f"{shotid} [Done] mask table ....  {datetime.datetime.now()}, # rows = {len(mask_table)}")
    if len(mask_table) > 0:
        try:
            print(f"{shotid} join table ....  {datetime.datetime.now()}")
            super_tab = join(fibers_table, mask_table, "fiber_id")
            print(f"{shotid} [Done] join table ....  {datetime.datetime.now()}")
        except:
            super_tab = fibers_table
    else:
        super_tab = fibers_table
    del fibers_table #don't need it anymore
    del mask_table
    try:
        super_tab.remove_columns(['fiber_id']) #don't need it anymore
    except:
        pass


    #######################################################################
    # first, always cut all fibers (entire fibers) that are flagged
    # remove any that are bad (flag = True is actually good)
    #######################################################################

    print(f"{shotid} removing flagged fibers ....  {datetime.datetime.now()}")
    try: #if the mask_table was empty, then these columns do not exist and there is no flagging to select
        if 'flag' in super_tab.columns:
            sel = super_tab['flag'] & super_tab['amp_flag'] & super_tab['meteor_flag'] & super_tab['gal_flag'] & super_tab['shot_flag'] & super_tab['throughput_flag']
            super_tab = super_tab[sel]

            print(f"{shotid} removed {len(super_tab) - np.count_nonzero(sel)} flagged fibers ....")
            #and we do not need the flag columns anymore
            super_tab.remove_columns(['flag','amp_flag','meteor_flag','gal_flag','shot_flag','throughput_flag'])
    except:
        pass

##############################
# for testing only
##############################
#print("!!!!! REMOVE ME !!!!!")
#super_tab.write("super_tab_test.fits", format="fits", overwrite=False)
super_tab = Table.read("super_tab_test.fits", format="fits")


#######################################################################
# next, always cut all fibers (entire fibers) that have excessive NaNs
# (in data or error) or excessive zero errors
#######################################################################

print(f"{shotid} removing excessive nans or zeros fibers ....  {datetime.datetime.now()}")
try:
    fd_bad = np.count_nonzero(np.isnan(super_tab["calfib"]),axis=1)
    fe_bad = np.count_nonzero(np.isnan(super_tab["calfibe"]),axis=1)
    ff_bad = np.count_nonzero(np.isnan(super_tab["calfib_ffsky"]),axis=1)

    #want to KEEP these so, select low numbers of NaNs
    sel = np.array(fd_bad < 100) & np.array(fe_bad < 100) & np.array(ff_bad < 100)

    print(f"{shotid} removed {len(super_tab) - np.count_nonzero(sel)} excessive NaN fibers ....")
    super_tab = super_tab[sel]

    #now remove excessive zeroe
    #NOTE: a zero in the calfib or calfib_ffsky is fine. There can be exactly zero flux (though rare, even
    # with rounding. A zero in the error (calfibe) is NOT OKAY. That is a flag that there is something wrong.
    # That said, there are often zero calfibe values at the very beginning and end of the rectified arrays where
    # there really are no values to put in AND if there are a lot of exactly zero flux values, we also assume that is
    # a problem.
    sz = len(G.CALFIB_WAVEGRID)
    fd_bad = sz - np.count_nonzero(super_tab["calfib"],axis=1)
    fe_bad = sz - np.count_nonzero(super_tab["calfibe"],axis=1)
    ff_bad = sz - np.count_nonzero(super_tab["calfib_ffsky"],axis=1)

    #want to KEEP these so, select low numbers of NaNs
    sel = np.array(fd_bad < 100) & np.array(fe_bad < 100) & np.array(ff_bad < 100)

    print(f"{shotid} removed {len(super_tab) - np.count_nonzero(sel)} excessive number of zero valued fibers ....")
    super_tab = super_tab[sel]


except:
    pass




flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

D['raw_fluxd'] = flux_stack
D['raw_fluxd_err'] = fluxe_stack
D['raw_contrib'] = contrib


flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib_ffsky"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

D2['raw_fluxd'] = flux_stack
D2['raw_fluxd_err'] = fluxe_stack
D2['raw_contrib'] = contrib


############################################################################
# next, cut all fibers with obvious continuum or deeply negative problems
############################################################################
print(f"{shotid} removing continuum fibers ....  {datetime.datetime.now()}")
rd = split_spectra_into_bins(super_tab['calfib'],super_tab['calfibe'],sort=False,trim=None)

super_tab['avg1'] = np.nanmedian(rd['f1'],axis=1) #3500-3860
super_tab['avg2'] = np.nanmedian(rd['f2'],axis=1) #3860-4270
super_tab['avg3'] = np.nanmedian(rd['f3'],axis=1) #4270-4860
super_tab['avg4'] = np.nanmedian(rd['f4'],axis=1) #4860-5090
super_tab['avg5'] = np.nanmedian(rd['f5'],axis=1) #5090-5500


norm_min = -0.05
norm_max = 0.05
#first bin, at exteme blue is different
sel =       np.array(super_tab['avg1'] > norm_min) & np.array(super_tab['avg1'] < 0.25) #3500-3860

sel = sel & np.array(super_tab['avg2'] > norm_min) & np.array(super_tab['avg2'] < norm_max) #3860-4270
sel = sel & np.array(super_tab['avg3'] > norm_min) & np.array(super_tab['avg3'] < norm_max) #4270-4860
sel = sel & np.array(super_tab['avg4'] > norm_min) & np.array(super_tab['avg4'] < norm_max) #4860-5090
sel = sel & np.array(super_tab['avg5'] > norm_min) & np.array(super_tab['avg5'] < norm_max) #5090-5500
print(f"{shotid} removed {len(super_tab) - np.count_nonzero(sel)} continuum fibers ....")
super_tab = super_tab[sel] #base for both local and ffsky

###########################################################################################
# next, cut all fibers with emission lines??? (this could be trixy)
# maybe a median filter that is 3 or 5 wide and if there are any peaks above
# some level, then cut the whole fiber?
# any faint emission line will not repeat over more than a handfull of fibers
#   and when averaged into 60-70K fibers, will be meaningless
###########################################################################################


# todo: ???? skip the emission line stuff


#############################################################################
# make stack of ALL fibers that survived the flagging, Nans, zeros trims
#############################################################################

flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

D['trim_fluxd'] = flux_stack
D['trim_fluxd_err'] = fluxe_stack
D['trim_contrib'] = contrib

flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib_ffsky"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

D2['trim_fluxd'] = flux_stack
D2['trim_fluxd_err'] = fluxe_stack
D2['trim_contrib'] = contrib

#########################################
# NOW apply the variable cuts:
#########################################

#########################################
# trim top 1,2,3,4, & 5%
#########################################

print(f"{shotid} top xx% trim ....  {datetime.datetime.now()}")

flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.99, sc=None, ir=None)
D['t01_fluxd'] = flux_stack
D['t01_fluxd_err'] = fluxe_stack
D['t01_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.98, sc=None, ir=None)
D['t02_fluxd'] = flux_stack
D['t02_fluxd_err'] = fluxe_stack
D['t02_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.97, sc=None, ir=None)
D['t03_fluxd'] = flux_stack
D['t03_fluxd_err'] = fluxe_stack
D['t03_contrib'] = contrib

flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.96, sc=None, ir=None)
D['t04_fluxd'] = flux_stack
D['t04_fluxd_err'] = fluxe_stack
D['t04_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.95, sc=None, ir=None)
D['t05_fluxd'] = flux_stack
D['t05_fluxd_err'] = fluxe_stack
D['t05_contrib'] = contrib


#now for ffsky
flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.99, sc=None, ir=None)
D2['t01_fluxd'] = flux_stack
D2['t01_fluxd_err'] = fluxe_stack
D2['t01_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.98, sc=None, ir=None)
D2['t02_fluxd'] = flux_stack
D2['t02_fluxd_err'] = fluxe_stack
D2['t02_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.97, sc=None, ir=None)
D2['t03_fluxd'] = flux_stack
D2['t03_fluxd_err'] = fluxe_stack
D2['t03_contrib'] = contrib

flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.96, sc=None, ir=None)
D2['t04_fluxd'] = flux_stack
D2['t04_fluxd_err'] = fluxe_stack
D2['t04_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.95, sc=None, ir=None)
D2['t05_fluxd'] = flux_stack
D2['t05_fluxd_err'] = fluxe_stack
D2['t05_contrib'] = contrib



###################################
# sigma clip 3, 5 sigma
###################################
print(f"{shotid} sigma clip trim ....  {datetime.datetime.now()}")

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=3.0,ir=None)
D['sc3_fluxd'] = flux_stack
D['sc3_fluxd_err'] = fluxe_stack
D['sc3_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=5.0,ir=None)
D['sc5_fluxd'] = flux_stack
D['sc5_fluxd_err'] = fluxe_stack
D['sc5_contrib'] = contrib

#ffsky
flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=3.0,ir=None)
D2['sc3_fluxd'] = flux_stack
D2['sc3_fluxd_err'] = fluxe_stack
D2['sc3_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=5.0,ir=None)
D2['sc5_fluxd'] = flux_stack
D2['sc5_fluxd_err'] = fluxe_stack
D2['sc5_contrib'] = contrib

###################################
# inter keep roughly 1,2,3 sigma
# assuming Normal Distro (so 67%, 95%, 99%)
###################################
print(f"{shotid} internal fraction trim ....  {datetime.datetime.now()}")

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.67)
D['ir67_fluxd'] = flux_stack
D['ir67_fluxd_err'] = fluxe_stack
D['ir67_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.95)
D['ir95_fluxd'] = flux_stack
D['ir95_fluxd_err'] = fluxe_stack
D['ir95_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.99)
D['ir99_fluxd'] = flux_stack
D['ir99_fluxd_err'] = fluxe_stack
D['ir99_contrib'] = contrib

#ffsky
flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.67)
D2['ir67_fluxd'] = flux_stack
D2['ir67_fluxd_err'] = fluxe_stack
D2['ir67_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.95)
D2['ir95_fluxd'] = flux_stack
D2['ir95_fluxd_err'] = fluxe_stack
D2['ir95_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.99)
D2['ir99_fluxd'] = flux_stack
D2['ir99_fluxd_err'] = fluxe_stack
D2['ir99_contrib'] = contrib

#################################################################
#
# now repeat for ENTIRE FIBER (rather than per wavelength)
#
#################################################################


#########################################
# trim top 1,2,3,4, & 5%
#########################################

#waverange = (3500,5500)

print(f"{shotid} fiber top xx% trim ....  {datetime.datetime.now()}")

flux_stack, fluxe_stack, contrib= stack_by_waverange_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.99, sc=None, ir=None)
D['ft01_fluxd'] = flux_stack
D['ft01_fluxd_err'] = fluxe_stack
D['ft01_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.98, sc=None, ir=None)
D['ft02_fluxd'] = flux_stack
D['ft02_fluxd_err'] = fluxe_stack
D['ft02_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.97, sc=None, ir=None)
D['ft03_fluxd'] = flux_stack
D['ft03_fluxd_err'] = fluxe_stack
D['ft03_contrib'] = contrib

flux_stack, fluxe_stack, contrib= stack_by_waverange_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.96, sc=None, ir=None)
D['ft04_fluxd'] = flux_stack
D['ft04_fluxd_err'] = fluxe_stack
D['ft04_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.95, sc=None, ir=None)
D['ft05_fluxd'] = flux_stack
D['ft05_fluxd_err'] = fluxe_stack
D['ft05_contrib'] = contrib


#now for ffsky
flux_stack, fluxe_stack, contrib= stack_by_waverange_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.99, sc=None, ir=None)
D2['ft01_fluxd'] = flux_stack
D2['ft01_fluxd_err'] = fluxe_stack
D2['ft01_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.98, sc=None, ir=None)
D2['ft02_fluxd'] = flux_stack
D2['ft02_fluxd_err'] = fluxe_stack
D2['ft02_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.97, sc=None, ir=None)
D2['ft03_fluxd'] = flux_stack
D2['ft03_fluxd_err'] = fluxe_stack
D2['ft03_contrib'] = contrib

flux_stack, fluxe_stack, contrib= stack_by_waverange_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.96, sc=None, ir=None)
D2['ft04_fluxd'] = flux_stack
D2['ft04_fluxd_err'] = fluxe_stack
D2['ft04_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.95, sc=None, ir=None)
D2['ft05_fluxd'] = flux_stack
D2['ft05_fluxd_err'] = fluxe_stack
D2['ft05_contrib'] = contrib



###################################
# sigma clip 3, 5 sigma
###################################
print(f"{shotid} fiber sigma clip trim ....  {datetime.datetime.now()}")

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=3.0,ir=None)
D['fsc3_fluxd'] = flux_stack
D['fsc3_fluxd_err'] = fluxe_stack
D['fsc3_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=5.0,ir=None)
D['fsc5_fluxd'] = flux_stack
D['fsc5_fluxd_err'] = fluxe_stack
D['fsc5_contrib'] = contrib

#ffsky
flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=3.0,ir=None)
D2['fsc3_fluxd'] = flux_stack
D2['fsc3_fluxd_err'] = fluxe_stack
D2['fsc3_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=5.0,ir=None)
D2['fsc5_fluxd'] = flux_stack
D2['fsc5_fluxd_err'] = fluxe_stack
D2['fsc5_contrib'] = contrib

###################################
# inter keep roughly 1,2,3 sigma
# assuming Normal Distro (so 67%, 95%, 99%)
###################################
print(f"{shotid} fiber internal fraction trim ....  {datetime.datetime.now()}")

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.67)
D['fir67_fluxd'] = flux_stack
D['fir67_fluxd_err'] = fluxe_stack
D['fir67_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.95)
D['fir95_fluxd'] = flux_stack
D['fir95_fluxd_err'] = fluxe_stack
D['fir95_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.99)
D['fir99_fluxd'] = flux_stack
D['fir99_fluxd_err'] = fluxe_stack
D['fir99_contrib'] = contrib

#ffsky
flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.67)
D2['fir67_fluxd'] = flux_stack
D2['fir67_fluxd_err'] = fluxe_stack
D2['fir67_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.95)
D2['fir95_fluxd'] = flux_stack
D2['fir95_fluxd_err'] = fluxe_stack
D2['fir95_contrib'] = contrib

flux_stack, fluxe_stack, contrib = stack_by_waverange_bw(super_tab['calfib_ffsky'], super_tab['calfibe'],trim=1.0,sc=None,ir=0.99)
D2['fir99_fluxd'] = flux_stack
D2['fir99_fluxd_err'] = fluxe_stack
D2['fir99_contrib'] = contrib




print(f"{shotid} writing output tables ....  {datetime.datetime.now()}")


T.add_row([ shotid,seeing,response,
            D['raw_fluxd'],D['raw_fluxd_err'],D['raw_contrib'],
            D['trim_fluxd'],D['trim_fluxd_err'],D['trim_contrib'],
            D['t01_fluxd'],D['t01_fluxd_err'],D['t01_contrib'],
            D['t02_fluxd'],D['t02_fluxd_err'],D['t02_contrib'],
            D['t03_fluxd'],D['t03_fluxd_err'],D['t03_contrib'],
            D['t04_fluxd'],D['t04_fluxd_err'],D['t04_contrib'],
            D['t05_fluxd'],D['t05_fluxd_err'],D['t05_contrib'],
            D['sc3_fluxd'],D['sc3_fluxd_err'],D['sc3_contrib'],
            D['sc5_fluxd'],D['sc5_fluxd_err'],D['sc5_contrib'],
            D['ir67_fluxd'],D['ir67_fluxd_err'],D['ir67_contrib'],
            D['ir95_fluxd'],D['ir95_fluxd_err'],D['ir95_contrib'],
            D['ir99_fluxd'],D['ir99_fluxd_err'],D['ir99_contrib'],
            D['ft01_fluxd'], D['ft01_fluxd_err'], D['ft01_contrib'],
            D['ft02_fluxd'], D['ft02_fluxd_err'], D['ft02_contrib'],
            D['ft03_fluxd'], D['ft03_fluxd_err'], D['ft03_contrib'],
            D['ft04_fluxd'], D['ft04_fluxd_err'], D['ft04_contrib'],
            D['ft05_fluxd'], D['ft05_fluxd_err'], D['ft05_contrib'],
            D['fsc3_fluxd'], D['fsc3_fluxd_err'], D['fsc3_contrib'],
            D['fsc5_fluxd'], D['fsc5_fluxd_err'], D['fsc5_contrib'],
            D['fir67_fluxd'], D['fir67_fluxd_err'], D['fir67_contrib'],
            D['fir95_fluxd'], D['fir95_fluxd_err'], D['fir95_contrib'],
            D['fir99_fluxd'], D['fir99_fluxd_err'], D['fir99_contrib'],
])

T2.add_row([shotid,seeing,response,
            D2['raw_fluxd'],D2['raw_fluxd_err'],D2['raw_contrib'],
            D2['trim_fluxd'],D2['trim_fluxd_err'],D2['trim_contrib'],
            D2['t01_fluxd'],D2['t01_fluxd_err'],D2['t01_contrib'],
            D2['t02_fluxd'],D2['t02_fluxd_err'],D2['t02_contrib'],
            D2['t03_fluxd'],D2['t03_fluxd_err'],D2['t03_contrib'],
            D2['t04_fluxd'],D2['t04_fluxd_err'],D2['t04_contrib'],
            D2['t05_fluxd'],D2['t05_fluxd_err'],D2['t05_contrib'],
            D2['sc3_fluxd'],D2['sc3_fluxd_err'],D2['sc3_contrib'],
            D2['sc5_fluxd'],D2['sc5_fluxd_err'],D2['sc5_contrib'],
            D2['ir67_fluxd'],D2['ir67_fluxd_err'],D2['ir67_contrib'],
            D2['ir95_fluxd'],D2['ir95_fluxd_err'],D2['ir95_contrib'],
            D2['ir99_fluxd'],D2['ir99_fluxd_err'],D2['ir99_contrib'],
            D2['ft01_fluxd'], D2['ft01_fluxd_err'], D2['ft01_contrib'],
            D2['ft02_fluxd'], D2['ft02_fluxd_err'], D2['ft02_contrib'],
            D2['ft03_fluxd'], D2['ft03_fluxd_err'], D2['ft03_contrib'],
            D2['ft04_fluxd'], D2['ft04_fluxd_err'], D2['ft04_contrib'],
            D2['ft05_fluxd'], D2['ft05_fluxd_err'], D2['ft05_contrib'],
            D2['fsc3_fluxd'], D2['fsc3_fluxd_err'], D2['fsc3_contrib'],
            D2['fsc5_fluxd'], D2['fsc5_fluxd_err'], D2['fsc5_contrib'],
            D2['fir67_fluxd'], D2['fir67_fluxd_err'], D2['fir67_contrib'],
            D2['fir95_fluxd'], D2['fir95_fluxd_err'], D2['fir95_contrib'],
            D2['fir99_fluxd'], D2['fir99_fluxd_err'], D2['fir99_contrib'],
])

T.write(table_outname, format='fits', overwrite=True)
T2.write(table_outname2, format='fits', overwrite=True)

print(f"{shotid} Done!  {datetime.datetime.now()}")