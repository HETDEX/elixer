"""
simple script to collect the "average empty" fibers for a single shot (all dithers) validating each against
acceptance critieria, and save to an astropy table

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
remove_detection_fibers = True #remove the fibers (usually in 3.5" apertures) that are nominally included in
                               #existing HETDEX detections (emission line or continuum)
TEST = False

PER_FIBER_PER_WAVE_MASKING = True

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


#new options --local --ffsky --rescor
if ("--local" in args) or ("--losky" in args):
    print("including local sky subtraction")
    losky = True
else:
    losky = False

if "--ffsky" in args:
    print("including ffsky subtraction")
    ffsky = True
else:
    #print("using local sky subtraction")
    ffsky = False

if "--rescor" in args:
    print("including ffsky+rescor subtraction")
    rescor = True
else:
    #print("using local sky subtraction")
    rescor = False

# if "--bothsky" in args:
#     print("Using local sky subtraction selection. Recording both local and ffsky extractions.")
#     print("Overriding --ffsky")
#     bothsky = True
#     ffsky = False
# else:
#     #print("using local sky subtraction")
#     bothsky = False

#always run rescor if running ffsky
# if "--add_rescor" in args:
#     print("Apply extra (Maja) ffsky residual correction")
#     add_rescor = True
# else:
#     print("Do Not apply extra (Maja) ffsky residual correction")
#     add_rescor = False

if "--fiber_corr" in args:
    print("Apply per fiber residual correction")
    per_fiber_corr = True
else:
    print("Do Not apply per fiber residual correction")
    per_fiber_corr = False


#sanity check
if not (losky or ffsky or rescor):
    print("No sky subtraction specified. Exiting().")
    exit(-1)


if PER_FIBER_PER_WAVE_MASKING:
    print("Using per-fiber per-wavelength flagging")
else:
    print("Not using additional per-fiber per-wavelength flagging")

#dust is a bit dodgy here since it is RA, Dec and thus per fiber specific and IT DOES VARY from fiber to fiber
#albeit rather slowly
# if "--dust" in args:
#     print("recording dust correction")
#     dust = True
# else:
#     print("NO dust correction")
#     dust = False


print(f"{shotid} starting: {datetime.datetime.now()}")

table_outname1 = "empty_fibers_"+str(shotid) + "_ll.fits"
table_outname2 = "empty_fibers_" + str(shotid) + "_ff.fits"
table_outname3 = "empty_fibers_"+str(shotid) + "_ffrc.fits"

#if any requested table alread exists, exit.
#the user must clear all tables manually

if losky and op.exists(table_outname1):
    #all good
    print(f"{shotid} --losky output already exists, {table_outname1}. Exiting. {datetime.datetime.now()}")
    exit(0)
elif ffsky and op.exists(table_outname2):
    #all good
    print(f"{shotid} --ffsky output already exists, {table_outname2}. Exiting. {datetime.datetime.now()}")
    exit(0)
elif rescor and op.exists(table_outname3):
    #all good
    print(f"{shotid} --ffsky+rescor output already exists, {table_outname3}. Exiting. {datetime.datetime.now()}")
    exit(0)

#maybe this one was already done?
# which files to check really depends on --ffsky and --bothsky
    #did they all copy or did it timeout during a copy?

# if bothsky:
#     if op.exists(table_outname) and op.exists(table_outname2):
#         #all good
#         print(f"{shotid} --bothsky outputs already exist. Exiting. {datetime.datetime.now()}")
#         exit(0)
# elif ffsky:
#     if op.exists(table_outname2):
#         #all good
#         print(f"{shotid} --ffsky output already exists. Exiting. {datetime.datetime.now()}")
#         exit(0)
# else:
#     if op.exists(table_outname):
#         #all good
#         print(f"{shotid} output already exist. Exiting. {datetime.datetime.now()}")
#         exit(0)

# if op.exists(table_outname) or op.exists(table_outname2):
#     print(f"{shotid} incomplete. Reset and retry. {datetime.datetime.now()}")
#
#     try:
#         if not ffsky or bothsky:
#             os.remove(table_outname)
#     except:
#         pass
#
#     try:
#         if ffsky or bothsky:
#             os.remove(table_outname2)
#     except:
#         pass



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
from hetdex_api.detections import Detections

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

    # when appending, though need the full width, so 1st bin needs to be (0,195) and the last (810,1036)

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

    trim can be an array

    sc = sigma_clip (symmetric)
    ir = internal ratio (e.g. interior 2/3)

    """

    lw = len(G.CALFIB_WAVEGRID)
    stack = np.zeros(lw)
    stack_err = np.zeros(lw)
    contrib = np.zeros(lw)
    N = len(fluxd_2d)


    try:
        lt = len(trim)
        if lt == lw:
            trim_array = True
            if np.any(trim!=1.0): #there are any non-1's
                trim_one = False
        else:
            print(f"Invalid trim array specified. Length ({lt}) does not match wavelength bins ({lw})")
            return None, None, None
    except:
        trim_array = False

    for i in range(1036):
        # only consider those that have an associated error (uncertainty)
        # errors of < 0 are flagged as bad pixels, cosmic strikes, etc
        sel = np.array(fluxd_err_2d[:, i] > 0) & np.array(~np.isnan(fluxd_err_2d[:, i])) & np.array(
            ~np.isnan(fluxd_2d[:, i]))
        if np.count_nonzero(sel) == 0:
            stack[i] = 0
            stack_err[i] = 0
            continue

        if trim_array and not trim_one:
            idx = np.argsort(fluxd_2d[:, i][sel])
            max_idx = int(len(idx) * trim[i])
            column = fluxd_2d[:, i][sel][idx][0:max_idx]
            column_err = fluxd_err_2d[:, i][sel][idx][0:max_idx]
        elif not trim_array and trim < 1.0:
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

        elif (trim_array and trim_one) or (not trim_array and trim == 1.0):
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


def fiber_flux_sort_statistic(fluxd_2d, fluxd_err_2d, waverange = (3600,5400),statistic="sum"):
    """

    sums, means, medians

    :param fluxd_2d:
    :param fluxd_err_2d:
    :param waverange:
    :param statistic: choose "sum", "mean", "median"
    :return:
    """

    N = len(fluxd_2d)
    sel = None
    waveidx = np.array([0,0])
    waveidx[0],*_ = SU.getnearpos(G.CALFIB_WAVEGRID,waverange[0])
    waveidx[1], *_ = SU.getnearpos(G.CALFIB_WAVEGRID, waverange[1])

    if statistic=="sum":
        flux_stat = np.nansum(fluxd_2d[:,waveidx[0]:waveidx[1]+1],axis=1)
    elif statistic == "mean":
        flux_stat = np.nanmean(fluxd_2d[:,waveidx[0]:waveidx[1]+1],axis=1)
    elif statistic == "median":
        flux_stat = np.nanmedian(fluxd_2d[:, waveidx[0]:waveidx[1] + 1], axis=1) #this takes a long time

    return flux_stat

def stack_by_waverange_bw(fluxd_2d, fluxd_err_2d, flux_sort = None, waverange = (3600,5400), trim=1.00, sc=None, ir=None):
    """

    stack by selection over the specified waverange
    unlike stack_by_wavebin_bw, here we cut out entire fibers and stack only complete fibers with whatever is left


    sc = sigma_clip (symmetric)
    ir = internal ratio (e.g. interior 2/3)
    flux_sort ... can be passed in instead of computing flux_sums each time (see flux_sorts() function

    """


    try:
        N = len(fluxd_2d)

        if flux_sort is None or len(flux_sort) != N:
            waveidx = np.array([0,0])
            waveidx[0],*_ = SU.getnearpos(G.CALFIB_WAVEGRID,waverange[0])
            waveidx[1], *_ = SU.getnearpos(G.CALFIB_WAVEGRID, waverange[1])

            # make an array against which to select
            # since these are all top cuts or sigma clips or interior ratios, it does not matter if these are sums or averages
            flux_sums = np.nansum(fluxd_2d[:, waveidx[0]:waveidx[1] + 1], axis=1)
        else:
            flux_sums = flux_sort #might actually be means or medians, etc, but regardless can be sorted monotonically
                                  #such that the highest values correspond to the largest fluxes

        if trim < 1.0:
            idx = np.argsort(flux_sums)
            max_idx = min(int(N * trim),len(idx))
            sel = np.array(flux_sums < flux_sums[idx[max_idx]])
        elif sc is not None:
            mask = sigma_clip(flux_sums, sigma=sc)
            sel = ~mask.mask
        elif ir is not None:
            er = (1.0 - ir) / 2.0  # exterior ratio, e.g. if ir = 2/3 then er = 1/6 ... trim away the upper and lower 1/6
            idx = np.argsort(flux_sums) #sorts low to high
            low_idx = int(N * er)
            high_idx = int(N * (ir + er))

            sel = np.array(flux_sums < flux_sums[idx[high_idx]])
            sel = sel & np.array(flux_sums > flux_sums[idx[low_idx]])
        elif trim == 1.0:
            sel = np.full(N,True)
        else:
            return None, None, 0
    except Exception as E:
        print(E)
        return None, None, 0

    try:

        stack, stack_err, _, contrib = SU.stack_spectra(fluxd_2d[sel],fluxd_err_2d[sel],
                                                        np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(sel),1)),
                                                        grid=G.CALFIB_WAVEGRID, avg_type="biweight",
                                                        straight_error=False, std=False)
    except Exception as E:
        print(E)
        stack = None
        stack_err = None
        contrib = 0

    return stack, stack_err, contrib


def threshold_value(values, percent):
    idx = np.argsort(values)
    max_idx = int(len(idx) * percent)  # the index in idx at the keep % value
    max_val = values[idx[max_idx]]  # the value of idx[max_idx]
    return max_val

#########################
# Main Logic
#########################

flags = 0 #flags for this shot  (see global_config for list of flags)






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
D1 = {} #dummy dict for Table T
T1 = Table(dtype=[('shotid', int), ('seeing',float), ('response',float),
                  ('flags', int),

                  ('raw_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flagged fibers removed
                  ('raw_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),  #NO OTHER TRIMMING
                  ('raw_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### fixed continuum trim (applies to ALL below) ###
                  #this might be the biggest differentiator ... what do we define as continuum level?
                  #and should this be shot variable?

                  ('trim_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #keep all AFTER flag trim and absolute flux cut
                  ('trim_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('trim_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  #variable trim

                  ('vt13_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% in blue to 3% in red
                  ('vt13_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('vt13_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('vt15_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% in blue to 5% in red
                  ('vt15_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('vt15_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### top percent trim (applies only to this section) ###
                  ('t01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 1% per wavelength bin
                  ('t01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t012_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.2% per wavelength bin
                  ('t012_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t012_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t014_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.4% per wavelength bin
                  ('t014_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t014_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t016_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.6% per wavelength bin
                  ('t016_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t016_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t018_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.8% per wavelength bin
                  ('t018_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t018_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 2% per wavelength bin
                  ('t02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t022_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.2% per wavelength bin
                  ('t022_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t022_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t024_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.4% per wavelength bin
                  ('t024_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t024_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t026_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.6% per wavelength bin
                  ('t026_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t026_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t028_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.8% per wavelength bin
                  ('t028_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t028_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3% per wavelength bin
                  ('t03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t032_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.2% per wavelength bin
                  ('t032_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t032_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t034_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.4% per wavelength bin
                  ('t034_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t034_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t036_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.6% per wavelength bin
                  ('t036_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t036_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t038_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.8% per wavelength bin
                  ('t038_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t038_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4% per wavelength bin
                  ('t04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t042_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.2% per wavelength bin
                  ('t042_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t042_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t044_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.4% per wavelength bin
                  ('t044_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t044_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t046_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.6% per wavelength bin
                  ('t046_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t046_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t048_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.8% per wavelength bin
                  ('t048_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t048_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 5% per wavelength bin
                  ('t05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ])

#ffsky (if both)
#T2 = copy.deepcopy(T)
D2 = {} #dummy dict for Table T2
T2 = Table(dtype=[('shotid', int), ('seeing',float), ('response',float),
                  ('flags', int),
                  ('raw_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flagged fibers removed
                  ('raw_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),  #NO OTHER TRIMMING
                  ('raw_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### fixed continuum trim (applies to ALL below) ###
                  #this might be the biggest differentiator ... what do we define as continuum level?
                  #and should this be shot variable?

                  ('trim_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #keep all AFTER flag trim and absolute flux cut
                  ('trim_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('trim_contrib', (float, len(G.CALFIB_WAVEGRID))),


                  ### variable trim

                  ('vt13_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% in blue to 3% in red
                  ('vt13_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('vt13_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('vt15_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% in blue to 5% in red
                  ('vt15_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('vt15_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### top percent trim (applies only to this section) ###
                  ('t01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 1% per wavelength bin
                  ('t01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t012_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.2% per wavelength bin
                  ('t012_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t012_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t014_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.4% per wavelength bin
                  ('t014_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t014_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t016_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.6% per wavelength bin
                  ('t016_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t016_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t018_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.8% per wavelength bin
                  ('t018_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t018_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2% per wavelength bin
                  ('t02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t022_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.2% per wavelength bin
                  ('t022_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t022_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t024_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.4% per wavelength bin
                  ('t024_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t024_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t026_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.6% per wavelength bin
                  ('t026_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t026_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t028_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.8% per wavelength bin
                  ('t028_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t028_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3% per wavelength bin
                  ('t03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t032_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.2% per wavelength bin
                  ('t032_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t032_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t034_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.4% per wavelength bin
                  ('t034_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t034_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t036_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.6% per wavelength bin
                  ('t036_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t036_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t038_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.8% per wavelength bin
                  ('t038_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t038_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4% per wavelength bin
                  ('t04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t042_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.2% per wavelength bin
                  ('t042_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t042_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t044_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.4% per wavelength bin
                  ('t044_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t044_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t046_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.6% per wavelength bin
                  ('t046_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t046_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t048_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.8% per wavelength bin
                  ('t048_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t048_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 5% per wavelength bin
                  ('t05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ])

#FFsky + rescor
D3 = {}  # dummy dict for Table T3
T3 = Table(dtype=[('shotid', int), ('seeing', float), ('response', float),
                  ('flags', int),
                  ('raw_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flagged fibers removed
                  ('raw_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),  # NO OTHER TRIMMING
                  ('raw_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### fixed continuum trim (applies to ALL below) ###
                  # this might be the biggest differentiator ... what do we define as continuum level?
                  # and should this be shot variable?

                  ('trim_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flag trim and absolute flux cut
                  ('trim_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('trim_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### variable trim

                  ('vt13_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% in blue to 3% in red
                  ('vt13_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('vt13_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('vt15_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% in blue to 5% in red
                  ('vt15_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('vt15_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ### top percent trim (applies only to this section) ###
                  ('t01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1% per wavelength bin
                  ('t01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t012_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.2% per wavelength bin
                  ('t012_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t012_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t014_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.4% per wavelength bin
                  ('t014_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t014_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t016_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.6% per wavelength bin
                  ('t016_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t016_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t018_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 1.8% per wavelength bin
                  ('t018_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t018_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2% per wavelength bin
                  ('t02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t022_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.2% per wavelength bin
                  ('t022_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t022_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t024_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.4% per wavelength bin
                  ('t024_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t024_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t026_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.6% per wavelength bin
                  ('t026_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t026_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t028_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 2.8% per wavelength bin
                  ('t028_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t028_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3% per wavelength bin
                  ('t03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t032_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.2% per wavelength bin
                  ('t032_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t032_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t034_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.4% per wavelength bin
                  ('t034_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t034_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t036_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.6% per wavelength bin
                  ('t036_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t036_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t038_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 3.8% per wavelength bin
                  ('t038_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t038_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4% per wavelength bin
                  ('t04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t042_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.2% per wavelength bin
                  ('t042_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t042_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t044_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.4% per wavelength bin
                  ('t044_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t044_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t046_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.6% per wavelength bin
                  ('t046_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t046_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t048_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 4.8% per wavelength bin
                  ('t048_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t048_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ('t05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # trim off top 5% per wavelength bin
                  ('t05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                  ('t05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                  ])

#if not TEST:
if True:
    print(f"{shotid} get_fibers_table() ....  {datetime.datetime.now()}")

    #note: for get_fibers_table()
    # mask_value by default is np.NaN
    # mask_options as None masks everything
    # might no actually need the mask column itself if we are masking in place
    # with the NaN's in place, entire fibers can be cut out if many NaNs (per normal logic) ... this is correct behavior
    #                          if fiber survives, individual wavebin NaNs will be exluded from the averaging

    if rescor: #adds just one more column for califb_ffsky_rescor
        fibers_table = get_fibers_table(shot,add_rescor=True,add_mask=PER_FIBER_PER_WAVE_MASKING,
                                        mask_in_place=PER_FIBER_PER_WAVE_MASKING,mask_options=None)
        print(f"{shotid} [DONE] get_fibers_table() + rescor ....  {datetime.datetime.now()}, # rows = {len(fibers_table)}")
    else: #local or ffsky must already be true
        fibers_table = get_fibers_table(shot,add_rescor=False,add_mask=PER_FIBER_PER_WAVE_MASKING,
                                        mask_in_place=PER_FIBER_PER_WAVE_MASKING,mask_options=None)
        print(f"{shotid} [DONE] get_fibers_table()  ....  {datetime.datetime.now()}, # rows = {len(fibers_table)}")



    ###########################################################################################################
    # todo: cut out the fibers that are within 3.5" of an existing HETDEX continuum or emission line detection
    ###########################################################################################################

    #need to load the detections for this shot (emission and continuum ... though, oustensibly continuum detections
    #would also get removed just with our continuum cuts in this script

    #then either need the fibers associated with each (directly) or use their PSF weighted RA, Dec centers and
    #select fibers with centers within 3.5"

    if remove_detection_fibers:
        ###################################################
        # remove fibers included in HETDEX detections
        ###################################################

        # load the detetctions for THIS shot (all detections, continuum and line)
        print(f"Loading detections catalog index ... {datetime.datetime.now()}")
        DI = Detections(catalog_type='index', searchable=True)
        q_shotid = shotid

        # sn 4.8 as lower limit for lines, sn==0.0 are for the continuum detections
        #print(f"Fetching detections for shotid {q_shotid} ...")
        shot_dets = DI.hdfile.root.DetectIndex.read_where("(shotid==q_shotid) & ((sn > 4.8) | (sn==0.0))",
                                                          field="detectid")

        #print(f"Num dets {len(shot_dets)}")

        fibers = []
        print(f"Fetching fiber_ids ... {datetime.datetime.now()}")

        shot_dets = np.array(shot_dets)
        # HDR3 Lines
        sel_det = np.array(shot_dets < 3090000000)
        if np.count_nonzero(sel_det) > 0:
            #print(np.count_nonzero(sel_det))
            dcat = Detections("hdr3", catalog_type="lines")
            for d in shot_dets[sel_det]:
                try:
                    fibers += [x[4] for x in dcat.get_fiber_info(d)]
                except:
                    pass

        # HDR3 Continuum
        sel_det = np.array(shot_dets >= 3090000000) & np.array(shot_dets < 4000000000)
        if np.count_nonzero(sel_det) > 0:
            #print(np.count_nonzero(sel_det))
            dcat = Detections("hdr3", catalog_type="continuum")
            for d in shot_dets[sel_det]:
                try:
                    fibers += [x[4] for x in dcat.get_fiber_info(d)]
                except:
                    pass

        # HDR4 Lines
        sel_det = np.array(shot_dets >= 4000000000) & np.array(shot_dets < 4090000000)
        if np.count_nonzero(sel_det) > 0:
            #print(np.count_nonzero(sel_det))
            dcat = Detections("hdr4", catalog_type="lines")
            for d in shot_dets[sel_det]:
                try:
                    fibers += [x[4] for x in dcat.get_fiber_info(d)]
                except:
                    pass

        # HDR4 Continuum
        sel_det = np.array(shot_dets >= 4090000000)
        if np.count_nonzero(sel_det) > 0:
            #print(np.count_nonzero(sel_det))
            dcat = Detections("hdr4", catalog_type="continuum")
            for d in shot_dets[sel_det]:
                try:
                    fibers += [x[4] for x in dcat.get_fiber_info(d)]
                except:
                    pass

        #todo: HDR5 Lines


        #todo: HDR5 Continuum

        ###################################
        # now, remove the fibers
        ###################################

        # remove fiber ids included in detections
        # exactly the same regardless of local, ffsky, or rescor
        print(f"Checking for matching fiber_ids  ... {datetime.datetime.now()}")
        u_fibers = np.unique(fibers).astype(str)
        sel_fibid = [x in u_fibers for x in fibers_table['fiber_id']]

        fibers_to_remove = np.count_nonzero(sel_fibid)

        #
        # if there are a huge fraction to be removed, say, more than 1/2 or 2/3? flag this as a problem and don't do it?
        #

        if fibers_to_remove/len(fibers_table) > 0.66:
            print(f"Warning! Unexpectedly large number of detection fibers: {fibers_to_remove/len(fibers_table)*100.:0.2}%. "
                  f"Will flag and NOT remove those fibers.")
            flags = flags | G.EFR_FLAG_TOO_MANY_DETECTION_FIBERS
        else:
            print(f"Removing {fibers_to_remove} fibers ...")

            print(f"Table size before: {len(fibers_table)}")
            fibers_table = fibers_table[~np.array(sel_fibid)]
            print(f"Table size after: {len(fibers_table)}")



    #drop the columns we don't care about to save memory
    cols_to_keep = ['fiber_id','calfibe','fiber_to_fiber','trace','chi2'] #have to keep the fiber_id for the moment

    if losky:
        cols_to_keep.append('calfib')
    if ffsky:
        cols_to_keep.append('calfib_ffsky')
    if rescor:
        cols_to_keep.append('calfib_ffsky_rescor')

    fibers_table.keep_columns(cols_to_keep)


    start = f'{shot}_0'
    stop = f'{shot}_9'
    print(f"{shotid} mask table ....  {datetime.datetime.now()}")
    mask_table = Table(FibIndex.fibermaskh5.root.Flags.read_where("(fiber_id > start) & (fiber_id < stop)"))
    print(f"{shotid} [Done] mask table ....  {datetime.datetime.now()}, # rows = {len(mask_table)}")
    if len(mask_table) > 0:
        try:
            print(f"{shotid} join tables ....  {datetime.datetime.now()}")
            super_tab = join(fibers_table, mask_table, "fiber_id")
            print(f"{shotid} [Done] join tables ....  {datetime.datetime.now()}")
        except:
            super_tab = fibers_table
    else:
        super_tab = fibers_table

    del fibers_table
    del mask_table


    #  DD:20240206 no ...  need to keep the fiber_id column now as we will use it to remove
    #  fibers that are part of existing HETDEX detections
    # try:
    #     super_tab.remove_columns(['fiber_id']) #don't need it anymore
    # except:
    #     pass


    #######################################################################
    # first, always cut all fibers (entire fibers) that are flagged
    # remove any that are bad (flag = True is actually good)
    #######################################################################

    print(f"{shotid} removing flagged fibers ....  {datetime.datetime.now()}")
    try: #if the mask_table was empty, then these columns do not exist and there is no flagging to select
        if 'flag' in super_tab.columns:
            sel = super_tab['flag'] & super_tab['amp_flag'] & super_tab['meteor_flag'] & super_tab['gal_flag'] & \
                      super_tab['shot_flag'] & super_tab['throughput_flag']
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
#super_tab = Table.read("super_tab_test.fits", format="fits")


#######################################################################
# next, always cut all fibers (entire fibers) that have excessive NaNs
# (in data or error) or excessive zero errors
#######################################################################

print(f"{shotid} removing excessive nans or zeros fibers ....  {datetime.datetime.now()}")

#todo: should consider just basing this zero/NaN value removal on local sky only? (though we removed it above)
# so would have to be use local sky only if it exists, then ffsky, then rescor?
# the idea is that local sky might be the best check here ... more stable than ffsky and does not have an additional
# rescor operation placed on top of it
# that said, we don't want ffsky or rescor getting bunch of 0 or NaNs either if they are different and all are requested
# ... maybe just leave as is ... not likely that there will be much difference (e.g. local by itself vs all 3 requested
# might generate some extra fiber removals, but it is not very many)

try:

    fe_bad = np.count_nonzero(np.isnan(super_tab["calfibe"]),axis=1)

    if losky:
        fd_bad = np.count_nonzero(np.isnan(super_tab["calfib"]),axis=1)
    else:
        fd_bad = np.zeros(len(super_tab))

    if ffsky:
        ff_bad = np.count_nonzero(np.isnan(super_tab["calfib_ffsky"]),axis=1)
    else:
        ff_bad = np.zeros(len(super_tab))

    if rescor:
        fr_bad = np.count_nonzero(np.isnan(super_tab["calfib_ffsky_rescor"]),axis=1)
    else:
        fr_bad = np.zeros(len(super_tab))

    #want to KEEP these so, select low numbers of NaNs
    sel = np.array(fd_bad < 100) & np.array(fe_bad < 100) & np.array(ff_bad < 100) & np.array(fr_bad < 100)

    print(f"{shotid} removed {len(super_tab) - np.count_nonzero(sel)} excessive NaN fibers ....")
    super_tab = super_tab[sel]

    #now remove excessive zeroe
    #NOTE: a zero in the calfib or calfib_ffsky is fine. There can be exactly zero flux (though rare, even
    # with rounding. A zero in the error (calfibe) is NOT OKAY. That is a flag that there is something wrong.
    # That said, there are often zero calfibe values at the very beginning and end of the rectified arrays where
    # there really are no values to put in AND if there are a lot of exactly zero flux values, we also assume that is
    # a problem.
    sz = len(G.CALFIB_WAVEGRID)

    fe_bad = sz - np.count_nonzero(super_tab["calfibe"], axis=1)

    if losky:
        fd_bad = sz - np.count_nonzero(super_tab["calfib"],axis=1)
    else:
        fd_bad = np.zeros(len(super_tab))

    if ffsky:
        ff_bad = sz - np.count_nonzero(super_tab["calfib_ffsky"],axis=1)
    else:
        ff_bad = np.zeros(len(super_tab))

    if rescor:
        fr_bad = sz - np.count_nonzero(super_tab["calfib_ffsky_rescor"], axis=1)
    else:
        fr_bad = np.zeros(len(super_tab))

    #want to KEEP these so, select low numbers of NaNs
    sel = np.array(fd_bad < 100) & np.array(fe_bad < 100) & np.array(ff_bad < 100)

    print(f"{shotid} removed {len(super_tab) - np.count_nonzero(sel)} excessive number of zero valued fibers ....")
    super_tab = super_tab[sel]


except:
    pass


if losky:
    flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

    D1['raw_fluxd'] = flux_stack
    D1['raw_fluxd_err'] = fluxe_stack
    D1['raw_contrib'] = contrib

if ffsky:
    flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib_ffsky"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

    D2['raw_fluxd'] = flux_stack
    D2['raw_fluxd_err'] = fluxe_stack
    D2['raw_contrib'] = contrib

if rescor:
    flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib_ffsky_rescor"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

    D3['raw_fluxd'] = flux_stack
    D3['raw_fluxd_err'] = fluxe_stack
    D3['raw_contrib'] = contrib


#!!! HERE !!!

############################################################################
# next, cut all fibers with obvious continuum or deeply negative problems
############################################################################
print(f"{shotid} removing continuum fibers ....  {datetime.datetime.now()}")

if losky:
    flux_col = 'calfib'
elif ffsky:
    flux_col = 'calfib_ffsky'
elif rescor:
    flux_col = "calfib_ffsky_rescor"
else: #safety
    print(f"Problem. Unknown sky model for continuum cut.")
    exit(-1)

print(f"Using {flux_col} for continuum cut")

rd = split_spectra_into_bins(super_tab[flux_col],super_tab['calfibe'],sort=False,trim=None)

super_tab['avg1'] = np.nanmedian(rd['f1'],axis=1) #3500-3860
super_tab['avg2'] = np.nanmedian(rd['f2'],axis=1) #3860-4270
super_tab['avg3'] = np.nanmedian(rd['f3'],axis=1) #4270-4860
super_tab['avg4'] = np.nanmedian(rd['f4'],axis=1) #4860-5090
super_tab['avg5'] = np.nanmedian(rd['f5'],axis=1) #5090-5500

#2024-07-24 ... all these are CURRENTLY the same, but may want to tune to each type
# needs investigation ... for example, rescor tends to slighly oversubtract, so the thresholds may not be appropriate
#                     .... and ffsky shape is different than llsky, esp i nthe blue so that first block (3500-3860)
#                               may need different threshold
if flux_col == "calfib":
    norm_min = -0.05
    norm_max = 0.05   #about g ~ 25
    #first bin, at exteme blue is different
    sel =       np.array(super_tab['avg1'] > norm_min) & np.array(super_tab['avg1'] < 0.25) #3500-3860
    sel = sel & np.array(super_tab['avg2'] > norm_min) & np.array(super_tab['avg2'] < norm_max) #3860-4270
    sel = sel & np.array(super_tab['avg3'] > norm_min) & np.array(super_tab['avg3'] < norm_max) #4270-4860
    sel = sel & np.array(super_tab['avg4'] > norm_min) & np.array(super_tab['avg4'] < norm_max) #4860-5090
    sel = sel & np.array(super_tab['avg5'] > norm_min) & np.array(super_tab['avg5'] < norm_max) #5090-5500

elif flux_col == "calfib_ffsky":
    norm_min = -0.05
    norm_max = 0.05 #about g ~ 25
    #first bin, at exteme blue is different
    sel =       np.array(super_tab['avg1'] > norm_min) & np.array(super_tab['avg1'] < 0.25) #3500-3860
    sel = sel & np.array(super_tab['avg2'] > norm_min) & np.array(super_tab['avg2'] < norm_max) #3860-4270
    sel = sel & np.array(super_tab['avg3'] > norm_min) & np.array(super_tab['avg3'] < norm_max) #4270-4860
    sel = sel & np.array(super_tab['avg4'] > norm_min) & np.array(super_tab['avg4'] < norm_max) #4860-5090
    sel = sel & np.array(super_tab['avg5'] > norm_min) & np.array(super_tab['avg5'] < norm_max) #5090-5500
elif flux_col == "calfib_ffsky_rescor":
    norm_min = -0.05
    norm_max = 0.05 #about g ~ 25
    #unlike the other two, the exteme blue for rescor is about the same ... very flat comparitively
    sel =       np.array(super_tab['avg1'] > norm_min) & np.array(super_tab['avg1'] < norm_max) #3500-3860
    sel = sel & np.array(super_tab['avg2'] > norm_min) & np.array(super_tab['avg2'] < norm_max) #3860-4270
    sel = sel & np.array(super_tab['avg3'] > norm_min) & np.array(super_tab['avg3'] < norm_max) #4270-4860
    sel = sel & np.array(super_tab['avg4'] > norm_min) & np.array(super_tab['avg4'] < norm_max) #4860-5090
    sel = sel & np.array(super_tab['avg5'] > norm_min) & np.array(super_tab['avg5'] < norm_max) #5090-5500
else: #safety
    print(f"Problem. Unknown column ({col}) for continuum cut.")
    exit(-1)

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
# I think this is mostly okay since an emission line would affect only a few wavelengths and a few fibers
# (generally) and will have little impact SINCE we are trying to additionally screen out almost detectable
# continuum in the next step



###############################################################################################
# Next cut "almost" detected continuum sources, under the assumption that fibers with
# the most high flux values compared to the other fibers probably have faint objects in them
# (here the ~ 1% of fibers that have 20% or more of bins in the top 10% of fluxes)
###############################################################################################

if True:
    # MARK fiber+wavelength bin to be removed based on a fractional cut, but do not remove
    # after iterating over all wavelength bins, remove entire fibers that have large numbers of marks

    # REMOVE FROM THE TABLE!, so this will produce a selection to apply to the table
    # So, we must maintain the fiber ordering as it is in the table

    #prefer basing this one off of calfib_ffsky since it is over the entire array
    if rescor:
        flux_col = "calfib_ffsky_rescor"
    elif ffsky:
        flux_col = "calfib_ffsky" #"calfib"
    else: #local as last resort
        flux_col = "calfib"  # "calfib"

    #temporary storage as 2D array fibers x wavelenths, if True then we would keep that fiber[wavelength] based on
    #whatever criteria we adopt
    keep_matrix = np.full(super_tab[flux_col].shape, False)
    fixed_percent_keep = 0.90 # or mark those in the top 10% of flux for each wavelength bin
                              #or, as coded, we are keeping the wavelength flux bins in the bottom 90%

    for i in range(1036): #len(G.CALFIB_WAVEGRID)
        # only consider those that have an associated error (uncertainty)
        # errors of < 0 are flagged as bad pixels, cosmic strikes, etc

        # treat NaNs or fluxd_err == 0 as BAD and flag them ... excessive numbers of these should cause
        # a fiber to removed just as well
        # However, "excessive" zero and nans have ALREADY been removed, so just igore these

        # So ... find the max allowed flux value (based on a % cut or clip)
        # then flag all values that are above the max

        sel = np.array(super_tab["calfibe"][:, i] > 0) & \
              np.array(~np.isnan(super_tab["calfibe"][:, i])) & \
              np.array(~np.isnan(super_tab[flux_col][:, i]))

        if np.count_nonzero(sel) == 0:
            keep_matrix[:, i] = True
            continue

        idx = np.argsort(super_tab[flux_col][:, i][sel])
        max_idx = int(len(idx) * fixed_percent_keep)  # the index in idx at the keep % value

        max_val = super_tab[flux_col][:, i][sel][idx[max_idx]]  # the value of idx[max_idx]

        # flag to keep ALL flux values in the column (not just the 'sel' subselection)
        # that are less than the max_val
        keep_matrix[:, i][super_tab[flux_col][:, i] < max_val] = True

    keep_fibers = np.count_nonzero(keep_matrix, axis=1)

    del keep_matrix

    #keep the fibers where 80% or more of their wavelength bins are NOT in the most extreme 10%
    min_keep_ct = int(0.80*len(G.CALFIB_WAVEGRID))
    sel = keep_fibers >= min_keep_ct

    print(f"Table size before fiber cut: {len(super_tab)}")
    print(f"Table size AFTER fiber cut: {np.count_nonzero(sel)} fibers.  ({len(super_tab)-np.count_nonzero(sel) })")

    super_tab = super_tab[sel]
   # print(f"Table size after fiber cut: {len(super_tab)}")

    #it is extremely unlikely that a fiber has 20% or more of its bins in the top 10%
    # (binomial distro, assuming Gaussian random, is has a prob in the e-22 range)
    # can use other cuts ... like fibers with 1/8 or more of bins in the top 5% is similar, etc.
    # I like having a larger fraction of a fibers bins, so I like 20% or more in the top 10%
    # This removes around 1% of fibers (at this point ... after the other selections)


#############################################################################
# make stack of ALL fibers that survived the flagging, Nans, zeros trims
#############################################################################


if losky:
    flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

    D1['trim_fluxd'] = flux_stack
    D1['trim_fluxd_err'] = fluxe_stack
    D1['trim_contrib'] = contrib

if ffsky:
    flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib_ffsky"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

    D2['trim_fluxd'] = flux_stack
    D2['trim_fluxd_err'] = fluxe_stack
    D2['trim_contrib'] = contrib

if rescor:
    flux_stack, fluxe_stack,contrib = stack_by_wavebin_bw(super_tab["calfib_ffsky_rescor"], super_tab["calfibe"], trim=1.00, sc=None, ir=None)

    D3['trim_fluxd'] = flux_stack
    D3['trim_fluxd_err'] = fluxe_stack
    D3['trim_contrib'] = contrib


#########################################
# NOW apply the variable cuts:
#########################################


##########################################
# variable trim
##########################################

print(f"{shotid} variable xx% trim ....  {datetime.datetime.now()}")

#1% to 3%
rwave = 4000.0
rtrim = 0.97
ridx,*_ = SU.getnearpos(G.CALFIB_WAVEGRID,rwave)

bwave = 3750.0
btrim = 0.99
bidx,*_ = SU.getnearpos(G.CALFIB_WAVEGRID,bwave)
ltrim = ridx-bidx+1

#base amount, 4000AA and redward
trim_array = np.full(len(G.CALFIB_WAVEGRID),rtrim)
#rapidly decrease cut blueward to 3750

trim_array[0:bidx] = btrim
trim_array[bidx:ridx+1] = np.linspace(btrim,rtrim,ltrim)

if losky:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=trim_array, sc=None, ir=None)
    D1['vt13_fluxd'] = flux_stack
    D1['vt13_fluxd_err'] = fluxe_stack
    D1['vt13_contrib'] = contrib


if ffsky:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=trim_array, sc=None, ir=None)
    D2['vt13_fluxd'] = flux_stack
    D2['vt13_fluxd_err'] = fluxe_stack
    D2['vt13_contrib'] = contrib

if rescor:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'],super_tab['calfibe'],trim=trim_array, sc=None, ir=None)
    D3['vt13_fluxd'] = flux_stack
    D3['vt13_fluxd_err'] = fluxe_stack
    D3['vt13_contrib'] = contrib


# 1% to 5%
rwave = 4000.0
rtrim = 0.95
ridx,*_ = SU.getnearpos(G.CALFIB_WAVEGRID,rwave)

bwave = 3750.0
btrim = 0.99
bidx,*_ = SU.getnearpos(G.CALFIB_WAVEGRID,bwave)
ltrim = ridx-bidx+1

#base amount, 4000AA and redward
trim_array = np.full(len(G.CALFIB_WAVEGRID),rtrim)
#rapidly decrease cut blueward to 3750

trim_array[0:bidx] = btrim
trim_array[bidx:ridx+1] = np.linspace(btrim,rtrim,ltrim)

if losky:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=trim_array, sc=None, ir=None)
    D1['vt15_fluxd'] = flux_stack
    D1['vt15_fluxd_err'] = fluxe_stack
    D1['vt15_contrib'] = contrib


if ffsky:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=trim_array, sc=None, ir=None)
    D2['vt15_fluxd'] = flux_stack
    D2['vt15_fluxd_err'] = fluxe_stack
    D2['vt15_contrib'] = contrib

if rescor:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'],super_tab['calfibe'],trim=trim_array, sc=None, ir=None)
    D3['vt15_fluxd'] = flux_stack
    D3['vt15_fluxd_err'] = fluxe_stack
    D3['vt15_contrib'] = contrib

#########################################
# fixed trim top 1,2,3,4, & 5%
#########################################

print(f"{shotid} top xx% trim ....  {datetime.datetime.now()}")

if losky:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.99, sc=None, ir=None)
    D1['t01_fluxd'] = flux_stack
    D1['t01_fluxd_err'] = fluxe_stack
    D1['t01_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.988, sc=None, ir=None)
    D1['t012_fluxd'] = flux_stack
    D1['t012_fluxd_err'] = fluxe_stack
    D1['t012_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.986, sc=None, ir=None)
    D1['t014_fluxd'] = flux_stack
    D1['t014_fluxd_err'] = fluxe_stack
    D1['t014_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.984, sc=None, ir=None)
    D1['t016_fluxd'] = flux_stack
    D1['t016_fluxd_err'] = fluxe_stack
    D1['t016_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.982, sc=None, ir=None)
    D1['t018_fluxd'] = flux_stack
    D1['t018_fluxd_err'] = fluxe_stack
    D1['t018_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.98, sc=None, ir=None)
    D1['t02_fluxd'] = flux_stack
    D1['t02_fluxd_err'] = fluxe_stack
    D1['t02_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.978, sc=None, ir=None)
    D1['t022_fluxd'] = flux_stack
    D1['t022_fluxd_err'] = fluxe_stack
    D1['t022_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.976, sc=None, ir=None)
    D1['t024_fluxd'] = flux_stack
    D1['t024_fluxd_err'] = fluxe_stack
    D1['t024_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.974, sc=None, ir=None)
    D1['t026_fluxd'] = flux_stack
    D1['t026_fluxd_err'] = fluxe_stack
    D1['t026_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.972, sc=None, ir=None)
    D1['t028_fluxd'] = flux_stack
    D1['t028_fluxd_err'] = fluxe_stack
    D1['t028_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.97, sc=None, ir=None)
    D1['t03_fluxd'] = flux_stack
    D1['t03_fluxd_err'] = fluxe_stack
    D1['t03_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.968, sc=None, ir=None)
    D1['t032_fluxd'] = flux_stack
    D1['t032_fluxd_err'] = fluxe_stack
    D1['t032_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.966, sc=None, ir=None)
    D1['t034_fluxd'] = flux_stack
    D1['t034_fluxd_err'] = fluxe_stack
    D1['t034_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.964, sc=None, ir=None)
    D1['t036_fluxd'] = flux_stack
    D1['t036_fluxd_err'] = fluxe_stack
    D1['t036_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.962, sc=None, ir=None)
    D1['t038_fluxd'] = flux_stack
    D1['t038_fluxd_err'] = fluxe_stack
    D1['t038_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.96, sc=None, ir=None)
    D1['t04_fluxd'] = flux_stack
    D1['t04_fluxd_err'] = fluxe_stack
    D1['t04_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.958, sc=None, ir=None)
    D1['t042_fluxd'] = flux_stack
    D1['t042_fluxd_err'] = fluxe_stack
    D1['t042_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.956, sc=None, ir=None)
    D1['t044_fluxd'] = flux_stack
    D1['t044_fluxd_err'] = fluxe_stack
    D1['t044_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.954, sc=None, ir=None)
    D1['t046_fluxd'] = flux_stack
    D1['t046_fluxd_err'] = fluxe_stack
    D1['t046_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.952, sc=None, ir=None)
    D1['t048_fluxd'] = flux_stack
    D1['t048_fluxd_err'] = fluxe_stack
    D1['t048_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib'],super_tab['calfibe'],trim=0.95, sc=None, ir=None)
    D1['t05_fluxd'] = flux_stack
    D1['t05_fluxd_err'] = fluxe_stack
    D1['t05_contrib'] = contrib



if ffsky:
    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.99, sc=None, ir=None)
    D2['t01_fluxd'] = flux_stack
    D2['t01_fluxd_err'] = fluxe_stack
    D2['t01_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.988, sc=None, ir=None)
    D2['t012_fluxd'] = flux_stack
    D2['t012_fluxd_err'] = fluxe_stack
    D2['t012_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.986, sc=None, ir=None)
    D2['t014_fluxd'] = flux_stack
    D2['t014_fluxd_err'] = fluxe_stack
    D2['t014_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.984, sc=None, ir=None)
    D2['t016_fluxd'] = flux_stack
    D2['t016_fluxd_err'] = fluxe_stack
    D2['t016_contrib'] = contrib

    flux_stack, fluxe_stack, contrib= stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.982, sc=None, ir=None)
    D2['t018_fluxd'] = flux_stack
    D2['t018_fluxd_err'] = fluxe_stack
    D2['t018_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'],super_tab['calfibe'],trim=0.98, sc=None, ir=None)
    D2['t02_fluxd'] = flux_stack
    D2['t02_fluxd_err'] = fluxe_stack
    D2['t02_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.978,
                                                           sc=None, ir=None)
    D2['t022_fluxd'] = flux_stack
    D2['t022_fluxd_err'] = fluxe_stack
    D2['t022_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.976,
                                                           sc=None, ir=None)
    D2['t024_fluxd'] = flux_stack
    D2['t024_fluxd_err'] = fluxe_stack
    D2['t024_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.974,
                                                           sc=None, ir=None)
    D2['t026_fluxd'] = flux_stack
    D2['t026_fluxd_err'] = fluxe_stack
    D2['t026_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.972,
                                                           sc=None, ir=None)
    D2['t028_fluxd'] = flux_stack
    D2['t028_fluxd_err'] = fluxe_stack
    D2['t028_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.97,
                                                           sc=None, ir=None)
    D2['t03_fluxd'] = flux_stack
    D2['t03_fluxd_err'] = fluxe_stack
    D2['t03_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.968,
                                                           sc=None, ir=None)
    D2['t032_fluxd'] = flux_stack
    D2['t032_fluxd_err'] = fluxe_stack
    D2['t032_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.966,
                                                           sc=None, ir=None)
    D2['t034_fluxd'] = flux_stack
    D2['t034_fluxd_err'] = fluxe_stack
    D2['t034_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.964,
                                                           sc=None, ir=None)
    D2['t036_fluxd'] = flux_stack
    D2['t036_fluxd_err'] = fluxe_stack
    D2['t036_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.962,
                                                           sc=None, ir=None)
    D2['t038_fluxd'] = flux_stack
    D2['t038_fluxd_err'] = fluxe_stack
    D2['t038_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.96,
                                                           sc=None, ir=None)
    D2['t04_fluxd'] = flux_stack
    D2['t04_fluxd_err'] = fluxe_stack
    D2['t04_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.958,
                                                           sc=None, ir=None)
    D2['t042_fluxd'] = flux_stack
    D2['t042_fluxd_err'] = fluxe_stack
    D2['t042_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.956,
                                                           sc=None, ir=None)
    D2['t044_fluxd'] = flux_stack
    D2['t044_fluxd_err'] = fluxe_stack
    D2['t044_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.954,
                                                           sc=None, ir=None)
    D2['t046_fluxd'] = flux_stack
    D2['t046_fluxd_err'] = fluxe_stack
    D2['t046_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.952,
                                                           sc=None, ir=None)
    D2['t048_fluxd'] = flux_stack
    D2['t048_fluxd_err'] = fluxe_stack
    D2['t048_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky'], super_tab['calfibe'], trim=0.95,
                                                           sc=None, ir=None)
    D2['t05_fluxd'] = flux_stack
    D2['t05_fluxd_err'] = fluxe_stack
    D2['t05_contrib'] = contrib

if rescor:
    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.99,
                                                           sc=None, ir=None)
    D3['t01_fluxd'] = flux_stack
    D3['t01_fluxd_err'] = fluxe_stack
    D3['t01_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.988,
                                                           sc=None, ir=None)
    D3['t012_fluxd'] = flux_stack
    D3['t012_fluxd_err'] = fluxe_stack
    D3['t012_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.986,
                                                           sc=None, ir=None)
    D3['t014_fluxd'] = flux_stack
    D3['t014_fluxd_err'] = fluxe_stack
    D3['t014_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.984,
                                                           sc=None, ir=None)
    D3['t016_fluxd'] = flux_stack
    D3['t016_fluxd_err'] = fluxe_stack
    D3['t016_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.982,
                                                           sc=None, ir=None)
    D3['t018_fluxd'] = flux_stack
    D3['t018_fluxd_err'] = fluxe_stack
    D3['t018_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.98,
                                                           sc=None, ir=None)
    D3['t02_fluxd'] = flux_stack
    D3['t02_fluxd_err'] = fluxe_stack
    D3['t02_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.978,
                                                           sc=None, ir=None)
    D3['t022_fluxd'] = flux_stack
    D3['t022_fluxd_err'] = fluxe_stack
    D3['t022_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.976,
                                                           sc=None, ir=None)
    D3['t024_fluxd'] = flux_stack
    D3['t024_fluxd_err'] = fluxe_stack
    D3['t024_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.974,
                                                           sc=None, ir=None)
    D3['t026_fluxd'] = flux_stack
    D3['t026_fluxd_err'] = fluxe_stack
    D3['t026_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.972,
                                                           sc=None, ir=None)
    D3['t028_fluxd'] = flux_stack
    D3['t028_fluxd_err'] = fluxe_stack
    D3['t028_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.97,
                                                           sc=None, ir=None)
    D3['t03_fluxd'] = flux_stack
    D3['t03_fluxd_err'] = fluxe_stack
    D3['t03_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.968,
                                                           sc=None, ir=None)
    D3['t032_fluxd'] = flux_stack
    D3['t032_fluxd_err'] = fluxe_stack
    D3['t032_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.966,
                                                           sc=None, ir=None)
    D3['t034_fluxd'] = flux_stack
    D3['t034_fluxd_err'] = fluxe_stack
    D3['t034_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.964,
                                                           sc=None, ir=None)
    D3['t036_fluxd'] = flux_stack
    D3['t036_fluxd_err'] = fluxe_stack
    D3['t036_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.962,
                                                           sc=None, ir=None)
    D3['t038_fluxd'] = flux_stack
    D3['t038_fluxd_err'] = fluxe_stack
    D3['t038_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.96,
                                                           sc=None, ir=None)
    D3['t04_fluxd'] = flux_stack
    D3['t04_fluxd_err'] = fluxe_stack
    D3['t04_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.958,
                                                           sc=None, ir=None)
    D3['t042_fluxd'] = flux_stack
    D3['t042_fluxd_err'] = fluxe_stack
    D3['t042_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.956,
                                                           sc=None, ir=None)
    D3['t044_fluxd'] = flux_stack
    D3['t044_fluxd_err'] = fluxe_stack
    D3['t044_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.954,
                                                           sc=None, ir=None)
    D3['t046_fluxd'] = flux_stack
    D3['t046_fluxd_err'] = fluxe_stack
    D3['t046_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.952,
                                                           sc=None, ir=None)
    D3['t048_fluxd'] = flux_stack
    D3['t048_fluxd_err'] = fluxe_stack
    D3['t048_contrib'] = contrib

    flux_stack, fluxe_stack, contrib = stack_by_wavebin_bw(super_tab['calfib_ffsky_rescor'], super_tab['calfibe'], trim=0.95,
                                                           sc=None, ir=None)
    D3['t05_fluxd'] = flux_stack
    D3['t05_fluxd_err'] = fluxe_stack
    D3['t05_contrib'] = contrib






print(f"{shotid} writing output tables ....  {datetime.datetime.now()}")

if losky:
    Dx = D1
    T1.add_row([ shotid,seeing,response,flags,
                Dx['raw_fluxd'], Dx['raw_fluxd_err'], Dx['raw_contrib'],
                Dx['trim_fluxd'],Dx['trim_fluxd_err'],Dx['trim_contrib'],
                Dx['vt13_fluxd'],Dx['vt13_fluxd_err'],Dx['vt13_contrib'],
                Dx['vt15_fluxd'],Dx['vt15_fluxd_err'],Dx['vt15_contrib'],

                Dx['t01_fluxd'], Dx['t01_fluxd_err'], Dx['t01_contrib'],
                Dx['t012_fluxd'],Dx['t012_fluxd_err'],Dx['t012_contrib'],
                Dx['t014_fluxd'],Dx['t014_fluxd_err'],Dx['t014_contrib'],
                Dx['t016_fluxd'],Dx['t016_fluxd_err'],Dx['t016_contrib'],
                Dx['t018_fluxd'],Dx['t018_fluxd_err'],Dx['t018_contrib'],

                Dx['t02_fluxd'], Dx['t02_fluxd_err'], Dx['t02_contrib'],
                Dx['t022_fluxd'], Dx['t022_fluxd_err'], Dx['t022_contrib'],
                Dx['t024_fluxd'], Dx['t024_fluxd_err'], Dx['t024_contrib'],
                Dx['t026_fluxd'], Dx['t026_fluxd_err'], Dx['t026_contrib'],
                Dx['t028_fluxd'], Dx['t028_fluxd_err'], Dx['t028_contrib'],

                Dx['t03_fluxd'],  Dx['t03_fluxd_err'],  Dx['t03_contrib'],
                Dx['t032_fluxd'], Dx['t032_fluxd_err'], Dx['t032_contrib'],
                Dx['t034_fluxd'], Dx['t034_fluxd_err'], Dx['t034_contrib'],
                Dx['t036_fluxd'], Dx['t036_fluxd_err'], Dx['t036_contrib'],
                Dx['t038_fluxd'], Dx['t038_fluxd_err'], Dx['t038_contrib'],

                Dx['t04_fluxd'],  Dx['t04_fluxd_err'],  Dx['t04_contrib'],
                Dx['t042_fluxd'], Dx['t042_fluxd_err'], Dx['t042_contrib'],
                Dx['t044_fluxd'], Dx['t044_fluxd_err'], Dx['t044_contrib'],
                Dx['t046_fluxd'], Dx['t046_fluxd_err'], Dx['t046_contrib'],
                Dx['t048_fluxd'], Dx['t048_fluxd_err'], Dx['t048_contrib'],
                Dx['t05_fluxd'],  Dx['t05_fluxd_err'],  Dx['t05_contrib'],
    ])

    T1.write(table_outname1, format='fits', overwrite=True)

if ffsky:
    Dx = D2
    T2.add_row([ shotid,seeing,response,flags,
                 Dx['raw_fluxd'], Dx['raw_fluxd_err'], Dx['raw_contrib'],
                 Dx['trim_fluxd'], Dx['trim_fluxd_err'], Dx['trim_contrib'],
                 Dx['vt13_fluxd'], Dx['vt13_fluxd_err'], Dx['vt13_contrib'],
                 Dx['vt15_fluxd'], Dx['vt15_fluxd_err'], Dx['vt15_contrib'],

                 Dx['t01_fluxd'], Dx['t01_fluxd_err'], Dx['t01_contrib'],
                 Dx['t012_fluxd'], Dx['t012_fluxd_err'], Dx['t012_contrib'],
                 Dx['t014_fluxd'], Dx['t014_fluxd_err'], Dx['t014_contrib'],
                 Dx['t016_fluxd'], Dx['t016_fluxd_err'], Dx['t016_contrib'],
                 Dx['t018_fluxd'], Dx['t018_fluxd_err'], Dx['t018_contrib'],

                 Dx['t02_fluxd'], Dx['t02_fluxd_err'], Dx['t02_contrib'],
                 Dx['t022_fluxd'], Dx['t022_fluxd_err'], Dx['t022_contrib'],
                 Dx['t024_fluxd'], Dx['t024_fluxd_err'], Dx['t024_contrib'],
                 Dx['t026_fluxd'], Dx['t026_fluxd_err'], Dx['t026_contrib'],
                 Dx['t028_fluxd'], Dx['t028_fluxd_err'], Dx['t028_contrib'],

                 Dx['t03_fluxd'], Dx['t03_fluxd_err'], Dx['t03_contrib'],
                 Dx['t032_fluxd'], Dx['t032_fluxd_err'], Dx['t032_contrib'],
                 Dx['t034_fluxd'], Dx['t034_fluxd_err'], Dx['t034_contrib'],
                 Dx['t036_fluxd'], Dx['t036_fluxd_err'], Dx['t036_contrib'],
                 Dx['t038_fluxd'], Dx['t038_fluxd_err'], Dx['t038_contrib'],

                 Dx['t04_fluxd'], Dx['t04_fluxd_err'], Dx['t04_contrib'],
                 Dx['t042_fluxd'], Dx['t042_fluxd_err'], Dx['t042_contrib'],
                 Dx['t044_fluxd'], Dx['t044_fluxd_err'], Dx['t044_contrib'],
                 Dx['t046_fluxd'], Dx['t046_fluxd_err'], Dx['t046_contrib'],
                 Dx['t048_fluxd'], Dx['t048_fluxd_err'], Dx['t048_contrib'],
                 Dx['t05_fluxd'], Dx['t05_fluxd_err'], Dx['t05_contrib'],

    ])

    T2.write(table_outname2, format='fits', overwrite=True)

if rescor:
    Dx = D3
    T3.add_row([ shotid,seeing,response,flags,
                 Dx['raw_fluxd'], Dx['raw_fluxd_err'], Dx['raw_contrib'],
                 Dx['trim_fluxd'], Dx['trim_fluxd_err'], Dx['trim_contrib'],
                 Dx['vt13_fluxd'], Dx['vt13_fluxd_err'], Dx['vt13_contrib'],
                 Dx['vt15_fluxd'], Dx['vt15_fluxd_err'], Dx['vt15_contrib'],

                 Dx['t01_fluxd'], Dx['t01_fluxd_err'], Dx['t01_contrib'],
                 Dx['t012_fluxd'], Dx['t012_fluxd_err'], Dx['t012_contrib'],
                 Dx['t014_fluxd'], Dx['t014_fluxd_err'], Dx['t014_contrib'],
                 Dx['t016_fluxd'], Dx['t016_fluxd_err'], Dx['t016_contrib'],
                 Dx['t018_fluxd'], Dx['t018_fluxd_err'], Dx['t018_contrib'],

                 Dx['t02_fluxd'], Dx['t02_fluxd_err'], Dx['t02_contrib'],
                 Dx['t022_fluxd'], Dx['t022_fluxd_err'], Dx['t022_contrib'],
                 Dx['t024_fluxd'], Dx['t024_fluxd_err'], Dx['t024_contrib'],
                 Dx['t026_fluxd'], Dx['t026_fluxd_err'], Dx['t026_contrib'],
                 Dx['t028_fluxd'], Dx['t028_fluxd_err'], Dx['t028_contrib'],

                 Dx['t03_fluxd'], Dx['t03_fluxd_err'], Dx['t03_contrib'],
                 Dx['t032_fluxd'], Dx['t032_fluxd_err'], Dx['t032_contrib'],
                 Dx['t034_fluxd'], Dx['t034_fluxd_err'], Dx['t034_contrib'],
                 Dx['t036_fluxd'], Dx['t036_fluxd_err'], Dx['t036_contrib'],
                 Dx['t038_fluxd'], Dx['t038_fluxd_err'], Dx['t038_contrib'],

                 Dx['t04_fluxd'], Dx['t04_fluxd_err'], Dx['t04_contrib'],
                 Dx['t042_fluxd'], Dx['t042_fluxd_err'], Dx['t042_contrib'],
                 Dx['t044_fluxd'], Dx['t044_fluxd_err'], Dx['t044_contrib'],
                 Dx['t046_fluxd'], Dx['t046_fluxd_err'], Dx['t046_contrib'],
                 Dx['t048_fluxd'], Dx['t048_fluxd_err'], Dx['t048_contrib'],
                 Dx['t05_fluxd'], Dx['t05_fluxd_err'], Dx['t05_contrib'],

    ])

    T3.write(table_outname3, format='fits', overwrite=True)


print(f"{shotid} Done!  {datetime.datetime.now()}")