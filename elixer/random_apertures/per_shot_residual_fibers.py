"""

This is an alternate way to compute the background or sky residual for a shot. It is a similar to
reandom_apertures in purpose, but collects all "empty" fibers for a single shot without the use of random apertures.

All fibers for the shot are collected and evaluated individually by rules to determine if they are "empty"



See Notebook version empty_fibers_residuals_for_shot.py (on /work/03261/polonius/hub/)

"""


import sys
import os
import os.path as op
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
import astropy.units as u
import copy
import glob
import shutil

from hetdex_api.config import HDRconfig
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import Survey,FiberIndex
from hetdex_tools.get_spec import get_spectra
from hetdex_api.extinction import *  #includes deredden_spectra
from hetdex_api.extract import Extract

from elixer import spectrum as elixer_spectrum
from elixer import spectrum_utilities as SU
from elixer import global_config as G

survey_name = "hdr4" #"hdr4" #"hdr2.1"
average = "weighted_biweight"
tmppath = "/tmp/hx/"
#tmppath = "/home/dustin/temp/random_apertures/hx/"

if not os.path.exists(tmppath):
    try:
        os.makedirs(tmppath, mode=0o755)
        if not os.access(tmppath, os.W_OK):
            print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
            exit(-1)
    except: #can catch timing on creation if multiples are trying to make it
        if not os.access(tmppath, os.W_OK):
            print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
            exit(-1)

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#as an integer, aka datevobs w/o the 'v'
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
    G.APPLY_SKY_RESIDUAL_TYPE = 1
else:
    print("Do Not apply per fiber residual correction")
    per_fiber_corr = False

if "--dust" in args:
    print("recording dust correction")
    dust = True
else:
    print("NO dust correction")
    dust = False

applydust = False


#keeping as 2 and 4 to mirror random_apertures
table_outname2 = "residual_fibers_"+str(shotid) + "_fibers.fits"
table_outname4 = "residual_fibers_"+str(shotid) + "_ff_fibers.fits"

# maybe this one was already done?
# which files to check really depends on --ffsky and --bothsky
# did they all copy or did it timeout during a copy?

if bothsky:
    if op.exists(table_outname2) and op.exists(table_outname4):
        # all good
        print(f"{shotid} --bothsky outputs already exists. Exiting.")
        exit(0)
elif ffsky:
    if op.exists(table_outname4):
        # all good
        print(f"{shotid} --ffsky outputs already exists. Exiting.")
        exit(0)
else:
    if op.exists(table_outname2):
        # all good
        print(f"{shotid} outputs already exists. Exiting.")
        exit(0)

if  op.exists(table_outname2) or op.exists(table_outname4):
    print(f"{shotid} incomplete. Reset and retry")

    try:
        if not ffsky or bothsky:
            os.remove(table_outname2)
    except:
        pass

    try:
        if ffsky or bothsky:
            os.remove(table_outname4)
    except:
        pass




#todo: make per fiber? still mostly noise dominant so, this is probably okay
wave_continuum_chunks_idx = [np.arange(15, 166, 1),  # 3500-3800
                             np.arange(215, 616, 1),  # 3900-4700
                             np.arange(615, 966, 1),  # 4700-5400
                             ]
# array of tuples that set the minimum and maximum average flux density value (e-17) that are loosely okay.
# these are not based on seeing FWHM.
# aligns with wave_continuum_chunks_idx
#aperture values: note 0.1e-17 is about 24.2 at 4700AA
acceptable_fluxd = [(-0.1, 0.5), #swing in the blue ... really should never be negative, roughly 5-10x is maybe okay
                    (-0.1, 0.1),
                    (-0.1, 0.1),]



#individual fibers
T2 = Table(dtype=[('ra', float), ('dec', float), ('fiber_ra', float), ('fiber_dec', float),
                 ('shotid', int),
                 ('seeing',float),
                 ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('dust_corr', (float, len(G.CALFIB_WAVEGRID))),
                 ])

#individual fibers (ffsky) if --bothsky
T4 = Table(dtype=[('ra', float), ('dec', float), ('fiber_ra', float), ('fiber_dec', float),
                 ('shotid', int),
                 ('seeing',float),
                 ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('dust_corr', (float, len(G.CALFIB_WAVEGRID))),
                 ])


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


def stack_by_trimmed_bins(ad, average="biweight"):
    """
    ad is the dictionary with the data
    """
    # stack apertures by trimmed data
    i = 0
    while f"f{i + 1}" in ad.keys():
        fkey = f"f{i + 1}"
        ekey = f"e{i + 1}"
        rkey = f"r{i + 1}"
        try:
            aper_flux_stack, aper_fluxe_stack, aper_grid, aper_contributions = SU.stack_spectra(
                ad[fkey],
                ad[ekey],
                np.tile(G.CALFIB_WAVEGRID[ad[rkey][0]:ad[rkey][1]], (len(ad[fkey]), 1)),
                grid=G.CALFIB_WAVEGRID[ad[rkey][0]:ad[rkey][1]],
                avg_type=average,
                straight_error=False)
            aper_contributions = int(np.nanmean(aper_contributions))  # should actually be a constant

            ad[f"fs{i + 1}"] = aper_flux_stack
            ad[f"es{i + 1}"] = aper_fluxe_stack

        except Exception as e:
            print(e, flush=True)
            ad[f"fs{i + 1}"] = None
            ad[f"es{i + 1}"] = None

        i += 1

    # now stitch them all together into a single spectrum and error
    ad["fluxd_stack"] = np.concatenate([ad[x] for x in ad.keys() if x[0:2] == "fs"])
    ad["fluxe_stack"] = np.concatenate([ad[x] for x in ad.keys() if x[0:2] == "es"])

    return ad


def mask_flux_where_bad(fluxd_2d, fluxd_err_2d):
    """
    where the error <=0 or nan, set flux to nan
    """

    sel = np.isnan(fluxd_err_2d) | np.array(fluxd_err_2d <= 0)
    fluxd_2d[sel] = np.nan
    return fluxd_2d


##################################
# run stuff
#################################

hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
print("Reading survey table ...")
survey_table=survey.return_astropy_table()
print("Getting FiberIndex ...")
FibIndex = FiberIndex(survey_name)
print("Reading amp flags table ...")
ampflag_table = Table.read(hetdex_api_config.badamp)



##
## grab all the fibers for this shot and randomly select random_fibers_per_shot
## go ahead and read the badamps for this shot so can more quickly compare
##
sel = (ampflag_table['shotid'] == shotid) & (ampflag_table['flag'] == 0)  # here, 0 is bad , 1 is good
bad_amp_list = ampflag_table['multiframe'][sel] #using the fiber index values which includes the bad amp flag

print("Building super table of fiberIDs ...")
idx = FibIndex.hdfile.root.FiberIndex.get_where_list("shotid==shot")
#fibers_table = Table(FibIndex.hdfile.root.FiberIndex.read_coordinates(idx)) #read all fibers
mask_table = Table(FibIndex.fibermaskh5.root.Flags.read_coordinates(idx))
#super_tab = join(fibers_table, mask_table, "fiber_id")


sel = np.array(survey_table['shotid'] == shotid)
seeing = float(survey_table['fwhm_virus'][sel])
response = float(survey_table['response_4540'][sel])

del survey_table



print("Fetching all fibers for shot ...")
#get all the fibers
fT = get_fibers_table(
    shot=shotid,
    coords=None,
    ifuslot=None,
    multiframe=None,
    expnum=None,
    radius=None,
    survey=survey_name,
    astropy=True,
    verbose=False,
    rawh5=False,
    F=None,
    fiber_flux_offset=None,
)

#jettison the columns we don't need to save memory and speed up actions
#keep chi2 ... might be useful cut at somepoint
#maybe think about keeping: chi2, calfib_counts, calfibe_counts (use istead of flux?)
#don't care any more about amp or fiber_id or ra, dec
fT.remove_columns(['ra','dec','amp','calfib_counts','calfibe_counts','chi2','contid','error1D','expnum',
                    'fiber_to_fiber','fibidx','fibnum','fpx','fpy','ifuid','ifuslot','ifux','ifuy','obsind','rms',
                    'sky_spectrum','sky_subtracted','specid','spectrum','trace','wavelength'])

#kept: multiframe (need below), calfib, calfib_ffsky, calfibe as needed for selection
fT = join(fT, mask_table, "fiber_id")


#select to those that are not "bad"
#kill the bad amps and fibers (that is, keep the not bad ones) (might be redundant)
# the sel_good_fibers along may be sufficient
sel_good_amps = np.array([mf not in bad_amp_list for mf in fT['multiframe']])
sel_good_fibers = np.array(fT['flag']) #flag ++ True is good
fT = fT[sel_good_amps & sel_good_fibers]

#now dump the multiframe and flags as their selection is done
fT.remove_columns(['fiber_id','multiframe','flag','amp_flag','gal_flag','meteor_flag'])




#let's keep the original order and all
#the trimming cut will follow later
calfib_bins = split_spectra_into_bins(fT['calfib'], fT['calfibe'], sort=False, trim=None)
#calfibe repeat is redundant, but keesp code simple and consistent
calfib_ffsky_bins = split_spectra_into_bins(fT['calfib_ffsky'], fT['calfibe'], sort=False, trim=None)


#add as columns (fluxes)
fT['calfib_f1'] = calfib_bins['f1']
fT['calfib_f2'] = calfib_bins['f2']
fT['calfib_f3'] = calfib_bins['f3']
fT['calfib_f4'] = calfib_bins['f4']
fT['calfib_f5'] = calfib_bins['f5']
fT['calfibe_f1'] = calfib_bins['e1']
fT['calfibe_f2'] = calfib_bins['e2']
fT['calfibe_f3'] = calfib_bins['e3']
fT['calfibe_f4'] = calfib_bins['e4']
fT['calfibe_f5'] = calfib_bins['e5']
fT['calfib_ffsky_f1'] = calfib_ffsky_bins['f1']
fT['calfib_ffsky_f2'] = calfib_ffsky_bins['f2']
fT['calfib_ffsky_f3'] = calfib_ffsky_bins['f3']
fT['calfib_ffsky_f4'] = calfib_ffsky_bins['f4']
fT['calfib_ffsky_f5'] = calfib_ffsky_bins['f5']

#add medians
# ! do not use where calfibe is 0 or None
fT['calfib_f1_md'] = np.nanmedian(mask_flux_where_bad(calfib_bins['f1'],calfib_bins['e1']),axis=1)
fT['calfib_f2_md'] = np.nanmedian(mask_flux_where_bad(calfib_bins['f2'],calfib_bins['e2']),axis=1)
fT['calfib_f3_md'] = np.nanmedian(mask_flux_where_bad(calfib_bins['f3'],calfib_bins['e3']),axis=1)
fT['calfib_f4_md'] = np.nanmedian(mask_flux_where_bad(calfib_bins['f4'],calfib_bins['e4']),axis=1)
fT['calfib_f5_md'] = np.nanmedian(mask_flux_where_bad(calfib_bins['f5'],calfib_bins['e5']),axis=1)

fT['calfib_ffsky_f1_md'] = np.nanmedian(mask_flux_where_bad(calfib_ffsky_bins['f1'],calfib_bins['e1']),axis=1)
fT['calfib_ffsky_f2_md'] = np.nanmedian(mask_flux_where_bad(calfib_ffsky_bins['f2'],calfib_bins['e2']),axis=1)
fT['calfib_ffsky_f3_md'] = np.nanmedian(mask_flux_where_bad(calfib_ffsky_bins['f3'],calfib_bins['e3']),axis=1)
fT['calfib_ffsky_f4_md'] = np.nanmedian(mask_flux_where_bad(calfib_ffsky_bins['f4'],calfib_bins['e4']),axis=1)
fT['calfib_ffsky_f5_md'] = np.nanmedian(mask_flux_where_bad(calfib_ffsky_bins['f5'],calfib_bins['e5']),axis=1)

#now eliminate fibers based on continua
#base this in LOCAL SKY and oringal LyCon paper ... those ranges were based on 2AA bins, so cut in half here
sel = np.array(fT['calfib_f1_md'] > -0.25) & np.array(fT['calfib_f1_md'] < 0.25)
sel = sel & np.array(fT['calfib_f2_md'] > -0.2) & np.array(fT['calfib_f2_md'] < 0.2)
sel = sel & np.array(fT['calfib_f3_md'] > -0.2) & np.array(fT['calfib_f3_md'] < 0.2)
sel = sel & np.array(fT['calfib_f4_md'] > -0.2) & np.array(fT['calfib_f4_md'] < 0.2)
sel = sel & np.array(fT['calfib_f5_md'] > -0.2) & np.array(fT['calfib_f5_md'] < 0.2)

fT = fT[sel]


# Now, trim off the top xx% wavelength bins by fluxes
# here we use chunks in 5 wavelength bin ranges
trim = 0.9
trim_calfib_bins = split_spectra_into_bins(fT['calfib'], fT['calfibe'], sort=True, trim=trim)
#calfibe repeat is redundant, but keesp code simple and consistent
trim_calfib_ffsky_bins = split_spectra_into_bins(fT['calfib_ffsky'], fT['calfibe'], sort=True, trim=trim)

final_trimmed_num_fibers = len(trim_calfib_bins['f1'])

#
# now stack as synthetic spectrum from the 5 wavelength bins
#
trim_calfib_bins = stack_by_trimmed_bins(trim_calfib_bins,average)
trim_calfib_ffsky_bins = stack_by_trimmed_bins(trim_calfib_ffsky_bins,average)

#want the "fluxd_stack" and "fluxe_stack" in each

#this is the nominal end product for this shot, though we have NOT checked for emission lines
# or other weird artificat/features

#todo: turn this into a model? like a Poly6?

#todo: save this off in a table?

#
# HERE!!!
#

#todo: need to get rid of emission lines (example 20220703017)




#
# #keep between 2/3 and 4/5 (0.667 and 0.8) ... keep the most that yields zero but not more than a 2/3 cut
# #try 0.8, 0.75. 0.7 and 0.67, but once done us the uniform smallest value
# md_cols_to_check = ['calfib_f1_md','calfib_f2_md','calfib_f3_md','calfib_f4_md','calfib_f5_md']
# ll_cols_to_stack = ['calfib_f1','calfib_f2','calfib_f3','calfib_f4','calfib_f5']
# ff_cols_to_stack = ['calfib_ffsky_f1','calfib_ffsky_f2','calfib_ffsky_f3','calfib_ffsky_f4','calfib_ffsky_f5']
# ee_cols_to_stack = ['calfibe_f1','calfibe_f2','calfibe_f3','calfibe_f4','calfibe_f5']
# range_to_stack =   [(0, 195), (195, 400), (400, 605), (605, 810), (810, 1036)]
#
# md_trims_to_check = [0.80,0.75,0.70,0.67]
# max_best_trim = []
# for col in md_cols_to_check:
#     avg = []
#     for trim in md_trims_to_check:
#         idx = np.argsort(fT[col])
#         max_idx = int(len(fT) * trim)
#         avg.append(np.nanmedian(fT[col][idx[0:max_idx]]))
#
#     #get value closest to zero (could be multiples, will yield the largest (first))
#     min_idx = np.argmin(abs(np.array(avg)))
#     max_best_trim.append(md_trims_to_check[min_idx])
#
# #now choose the smallest trim (keep the fewest fibers)
# trim = np.min(max_best_trim)
#
# #and perform the trim in each BIN
# by_bin_ll_stack = np.array([])
# by_bin_ll_err_stack = np.array([])
# by_bin_ff_stack = np.array([])
# by_bin_ff_err_stack = np.array([])
# by_bin_trim = trim
# for md_col,ll_col,ff_col,ee_col,rr in zip(md_cols_to_check,ll_cols_to_stack,ff_cols_to_stack,ee_cols_to_stack,range_to_stack):
#     idx = np.argsort(fT[md_col])
#     max_idx = int(len(fT) * trim)
#
#     flux_stack, fluxe_stack, grid, contributions = SU.stack_spectra(
#                                                     fT[ll_col][idx[0:max_idx]],
#                                                     fT[ee_col][idx[0:max_idx]],
#                                                     np.tile(G.CALFIB_WAVEGRID[rr[0]:rr[1]], (max_idx, 1)),
#                                                     grid=G.CALFIB_WAVEGRID[rr[0]:rr[1]],
#                                                     avg_type=average,
#                                                     straight_error=False)
#
#     by_bin_ll_stack = np.concatenate((by_bin_ll_stack, flux_stack))
#     by_bin_ll_err_stack = np.concatenate((by_bin_ll_err_stack, fluxe_stack))
#
#     flux_stack, fluxe_stack, grid, contributions = SU.stack_spectra(
#                                                     fT[ff_col][idx[0:max_idx]],
#                                                     fT[ee_col][idx[0:max_idx]],
#                                                     np.tile(G.CALFIB_WAVEGRID[rr[0]:rr[1]], (max_idx, 1)),
#                                                     grid=G.CALFIB_WAVEGRID[rr[0]:rr[1]],
#                                                     avg_type=average,
#                                                     straight_error=False)
#
#     by_bin_ff_stack = np.concatenate((by_bin_ff_stack, flux_stack))
#     by_bin_ff_err_stack = np.concatenate((by_bin_ff_err_stack, fluxe_stack))
#


#and overall selection (entire fibers)





# total flux


# per bin flux

len(fT)