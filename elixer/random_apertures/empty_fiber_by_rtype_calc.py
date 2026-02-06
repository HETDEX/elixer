"""
This is part of the empty fiber calculations and is taken from the hsc_stacking_empty_fiber_calibrations.ipynb

In the notebook, after all the setup, there is cell that iterates over all the rtypes [None, raw, trim, t01,t012 ....]
and for each of those, gets all the detections that fall within each magnitude bin [23.1, 23.3, 23.5 .... 27.1],
and for each detection, recreates its PSF weighted spectra, applying the particular rtype.

These spectra are then stacked within each rtype+mag_bin and plotted. The "best" rtype is chosen as the default calibration.
("best" being overall closest to the 1:1 center line of the mag bin)

!!! Note: an additional correction should be made that could alter the "best" rtype selection. Based on Mahan's investigation,
the HSC imaging is a bit over(sky) subtracted and as the object magnitudes get faint and are stacked, this may become
a meaningful difference.


This needs to be exectued in the directory above the ./tables

"""
import os.path
import os.path as op
import sys
import glob
import numpy as np

import copy
import pickle

import tables

from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
from astropy.stats import sigma_clip
import astropy.stats.biweight as biweight
import astropy.units as u

from hetdex_api.config import HDRconfig
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import Survey,FiberIndex
from hetdex_tools.get_spec import get_spectra
from hetdex_api.extinction import *  #includes deredden_spectra

from elixer import spectrum as elixer_spectrum
from elixer import spectrum_utilities as SU
from elixer import global_config as G
from elixer import weighted_biweight as weighted_biweight
from elixer import mcmc_gauss

#from tqdm import tqdm #, trange
from itertools import combinations, product
from scipy.ndimage import gaussian_filter1d

#import time
from datetime import datetime, timedelta
import requests

time_outer_start = datetime.now()

HDR_VERSION = "5"
DEREDDEN = True
RES_CORRECTION = 1 # 1 = per fiber, 2 = per aperture
RE_PSF = True #remake the PSF weighted extraction ... MUST BE TRUE FOR THIS EXERCISE

txx_list = [ None,'raw','trim',
            't01','t012','t014','t016','t018',
            't02','t022','t024','t026','t028',
            't03','t032','t034','t036','t038',
            't04','t042','t044','t046','t048',
            't05'
            ]

args = list(sys.argv)
del args[0] #args.pop(0) #remove THIS file
args = [x.replace("--","-") for x in args]


if "-i" in args:
    i = args.index("-i")
    try:
        which_txx = int(args[i+1])
        if not 0 <= which_txx < len(txx_list):   #negative values are the same, but force just the clean operation (e.g. to be used on an old run)
            print(f"Invalid -i specified ({which_txx}). Must be 0 <= i < {len(txx_list)}.")
            exit(-1)
    except:
        print(f"Invalid -i specified")
        exit(-1)

    del args[i+1]  # args.pop(0) #remove THIS file
    args.remove("-i")
else:
    print(f"Invalid:   -i  not specified.")
    exit(-1)


if "-sky" in args:
    i = args.index("-sky")
    try:
        sky = str(args[i+1])
    except:
        print(f"Invalid -sky specified")
        exit(-1)

    del args[i+1]  # args.pop(0) #remove THIS file
    args.remove("-sky")
else:
    print(f"Invalid:  --sky  not specified.")
    exit(-1)

if sky == "ll":
    FFSKY = False
    RESCOR = False
elif sky == "ff":
    FFSKY = True
    RESCOR = False
elif sky == "rc":
    FFSKY = True
    RESCOR = True
else:
    print(f"Problem. Inconsistent sky: {sky}")
    exit(-1)

txx = txx_list[which_txx]
#get all the detections
fns = sorted(glob.glob("mag_bin_dets_*.txt"))
detection_lists = []
counts = 0
for fn in fns:
    dets = np.loadtxt(fn,dtype=int)
    detection_lists.append(dets)
    counts += len(dets)
    print(txx, fn,len(dets),flush=True)

print(txx,"total",counts,flush=True)

fiber_table_basedir = "./tables"
basename = "hsc_sep_rex_flags_nei1p5"
dTh5 = tables.open_file(op.join(fiber_table_basedir,f"{basename}_dets_{sky}_all.h5"))

step = 0.2
delta = step/2.0 #+ step/4.0
mag_bins = np.arange(23.0,27.2, step)
#bin_mids = 0.5 * (bins[0:-1]+bins[1:])
mag_bin_mids = mag_bins + step/2.0

#convert to fluxes
flux_bins = [SU.mag2cgs(x,G.DEX_G_EFF_LAM) for x in mag_bins[::-1]]
flux_bin_mids = [SU.mag2cgs(x,G.DEX_G_EFF_LAM) for x in mag_bin_mids[::-1]]


###################################
# funcs declarations
###################################

G.BGR_RES_FIBER_H5_LL_FN = "./tables/BGR_RES_FIBER_H5_LL.h5"
G.BGR_RES_FIBER_H5_FF_FN = "./tables/BGR_RES_FIBER_H5_FF.h5"
#G.BGR_RES_FIBER_H5_FF_FN = "./tables/empty_fibers_ff_all_20250120.h5"
G.BGR_RES_FIBER_H5_FFRC_FN = "./tables/BGR_RES_FIBER_H5_FFRC.h5"

# G.BGR_RES_FIBER_H5_LL_FN = "./empty_fibers_ll__all.h5"
# G.BGR_RES_FIBER_H5_FF_FN = "./empty_fibers_ff__all.h5"
# G.BGR_RES_FIBER_H5_FFRC_FN = "./empty_fibers_ffrc__all.h5"

G.LOG_TO_STDOUT = True
G.GLOBAL_LOGGING = True

log = False


# !!! note ... the fluxes are about spot on, with differences, at most 15 orders of magnitude less than 1
# !!! but errors differences are running a few % to 20-30%

def make_psf_weighted_spectra(fiber_fluxd, fiber_fluxd_err, fiber_wave_weights, fiber_wave_masks):
    """
    fiber_fluxd:         2D array, 1036 per fiber (normally in fluxd units), usually from fT table['clean_fluxd'] !!!
    fiber_fluxd_err:     ditto but error arrays, usually from fT table['clean_fluxd_err'] !!!
    fiber_wave_weights:  usually from fT table['wave_weights']
    fiber_wave_masks:  usually from fT table['wave_masks']

    !!! warning !!! be sure to use the "clean_fluxd" and "clean_fluxd_err" UNLESS you REALLY know what you are doing !!!

    """

    try:

        norms = np.nansum([x[0] * x[1] ** 2 for x in zip(fiber_wave_masks, fiber_wave_weights)], axis=0)

        psf_sum = np.nansum([x[0] * x[1] * x[2] for x in zip(fiber_fluxd,
                                                             fiber_wave_masks,
                                                             fiber_wave_weights)], axis=0) / norms

        psf_sum_err = np.sqrt(np.nansum([(x[0] ** 2) * x[1] * x[2] for x in zip(fiber_fluxd_err,
                                                                                fiber_wave_masks,
                                                                                fiber_wave_weights)], axis=0) / norms)

        return psf_sum, psf_sum_err
    except Exception as E:
        print(E,flush=True)

        return None, None


def subselect_dets_on_density(de0, de1, detectids, shotids_for_detectids):
    """
    de0,de1 = left and right edges (density_edges)
    """
    # wasteful to keeep re-reading, but simple to keep it here and not often hit
    bin_shots, bin_dets, bin_faint_dets, bin_binid = np.loadtxt("hsc_matched_with_shots_binned.txt", unpack=True,
                                                                usecols=(0, 1, 2, 4), dtype=int)
    bin_norm = np.loadtxt("hsc_matched_with_shots_binned.txt", unpack=True, usecols=(3), dtype=float)

    sel = np.array(bin_norm > de0) & np.array(bin_norm <= de1)

    bin_shots[sel]

    sel_shots = [s in bin_shots[sel] for s in shotids_for_detectids]
    return detectids[sel_shots], sel_shots

# print(datetime.now().strftime("%H:%M:%S"))
def stack_list_observed(detectids, rtype):
    """
    use the globals to control the FFSKY, RES_CORRECTION, RTYPE, etc

    the passed in detectids are PRE-SELECTED for whatever you want (e.g. for a mag range)
    """
    # left,*_ =  SU.getnearpos(G.CALFIB_WAVEGRID,3525)
    # right,*_ =  SU.getnearpos(G.CALFIB_WAVEGRID,5475) #5502
    #     left = 25
    #     right = -24 #len(G.CALFIB_WAVEGRID)+1 # -24
    left = 0
    right = len(G.CALFIB_WAVEGRID) + 1  # -24
    flux_bin_width = 1.0  # these are fd
    flux_scale = 1e-17
    waves = G.CALFIB_WAVEGRID[left:right]

    # rest_lumd = []
    # rest_lumde = []
    # rest_waves = []

    obs_fluxd = []
    obs_fluxde = []
    # obs_waves = [] #not needed ... always G.CALFIB_WAVEGRID

    # ct_no_correction = 0

    fTh5 = None
    # operating on dT as an astropy table
    try:
        last_shot = None
        #print(f"Stacking {len(detectids)} ... ")
        for q_detectid in detectids: #, desc=f"{rtype}"):
            if log:
                print(f"looking up in dTh5: {datetime.now().strftime('%H:%M:%S')}",flush=True)

            # dTrows = dTh5.root.Table.read_where("detectid==q_detectid")
            dTrows = dTh5.root.Detections.read_where("detectid==q_detectid")
            if log:
                print(f"looking up in dTh5 .. done : {datetime.now().strftime('%H:%M:%S')}",flush=True)

            if len(dTrows) != 1:
                print(f"Error! {q_detectid} bad lookup in dTh5.",flush=True)
                continue

            q_shotid = dTrows[0]['shotid']
            q_ra = dTrows[0]['ra']
            q_dec = dTrows[0]['dec']
            q_seeing = dTrows[0]['seeing']
            q_response = dTrows[0]['response']

            if True:  # MUST USE RE_PSF RE_PSF:
                if log:
                    print(f"looking up in fTh5: {datetime.now().strftime('%H:%M:%S')}",flush=True)

                # expect multiple rows (one for each fiber in the original extraction)
                # load the fTh5 shot specific table
                try:
                    if last_shot != q_shotid:  # otherwise, we should already have the table (fTh5) open
                        last_shot = q_shotid
                        if fTh5 is not None:
                            fTh5.close()
                        fTh5 = tables.open_file(op.join(fiber_table_basedir, f"{q_shotid}_{basename}_{sky}.h5"),flush=True)
                except Exception as e:
                    print("Exception! (stack_list_observed)", e,flush=True)
                    break

                # frows = fTh5.root.Table.read_where('detectid==q_detectid')
                frows = fTh5.root.Fibers.read_where('detectid==q_detectid')

                if log:
                    print(f"looking up in fTh5 .. done : {datetime.now().strftime('%H:%M:%S')}",flush=True)

                if RES_CORRECTION == 1 and rtype is not None and rtype != "None":
                    # print("Reading rows ...")
                    if log:
                        print(f"fetching empty fiber ... : {datetime.now().strftime('%H:%M:%S')}",flush=True)
                    empty_fiber, empty_fiber_err, contrib, status = SU.get_empty_fiber_residual_h5(
                        hdr=HDR_VERSION,
                        rtype=rtype,
                        shotid=q_shotid,
                        seeing=q_seeing,
                        response=q_response,
                        ffsky=FFSKY,
                        add_rescor=RESCOR,
                        persist=True)
                    if log:
                        print(f"fetching empty fiber ... Done: {datetime.now().strftime('%H:%M:%S')}",flush=True)

                    if status > 0:
                        print(f"{q_detectid} : 0x{status:08x}",flush=True)
                        continue
                else:
                    # print(f"Unexpected: {RES_CORRECTION} {FFSKY} {RESCOR} {rtype}")
                    empty_fiber = np.full(len(frows['clean_fluxd'][0]), 0)
                    empty_fiber_err = np.full(len(frows['clean_fluxd'][0]), 0)
                    # ct_no_correction += 1

                if log:
                    print(f"make_psf_weighted_spectra  ... : {datetime.now().strftime('%H:%M:%S')}",flush=True)

                fluxd, fluxde = make_psf_weighted_spectra(frows['clean_fluxd'] - empty_fiber,
                                                          frows['clean_fluxd_err'] - empty_fiber_err,
                                                          frows['wave_weights'],
                                                          frows['wave_masks'])

                if log:
                    print(f"make_psf_weighted_spectra  ... Done : {datetime.now().strftime('%H:%M:%S')}",flush=True)

            #             else: #using the original PSF weighted extraction
            #                 fluxd = drow['fluxd']
            #                 fluxde = drow['fluxd_err']

            if RES_CORRECTION == 2:  # per aperture
                pass  # todo
            # elif RES_CORRECTION == 1: #per fiber
            #    pass #already done in RE_PSF ... or alreay done if that is part of the dT table
            # else: no RES_CORRECTION
            #

            if DEREDDEN:
                if log:
                    print(f"DEREDDEN  ...: {datetime.now().strftime('%H:%M:%S')}",flush=True)
                # coord = SkyCoord(ra=q_ra * u.deg, dec=q_dec * u.deg)
                # dust_corr = deredden_spectra(G.CALFIB_WAVEGRID,coord)
                dust_corr = dTrows[0]['dust_corr']
                fluxd *= dust_corr
                fluxde *= dust_corr

                if log:
                    print(f"DEREDDEN  ... Done : {datetime.now().strftime('%H:%M:%S')}",flush=True)

            obs_fluxd.append(fluxd[left:right])
            obs_fluxde.append(fluxde[left:right])

            ##################################################################
            #   NO redshift ... these are OBSERVED FRAME STACKS ONLY !!!
            ##################################################################

        # print("# no corrections", ct_no_correction)
    except Exception as E:
        print(d, E,flush=True)

        # list of fluxes now down, so stack

    # print(f"stacking ... : {datetime.now().strftime('%H:%M:%S')}")

    # print(f"obs_fluxd: {np.shape(obs_fluxd)}")
    # print(f"obs_fluxde: {np.shape(obs_fluxde)}")
    # print(f"waves: {np.shape(waves)}")
    # print(f"tile: {np.shape(np.tile(waves,(len(obs_fluxd),1)))}")

    fluxd_stack, fluxde_stack, grid, contributions, fluxd_std = SU.stack_spectra(
        obs_fluxd,
        obs_fluxde,
        np.tile(waves, (len(obs_fluxd), 1)),
        grid=waves,
        avg_type="biweight",
        straight_error=False,
        std=True)

    # print(f"stacking ... done : {datetime.now().strftime('%H:%M:%S')}")

    del obs_fluxd
    del obs_fluxde

    return fluxd_stack, fluxde_stack, grid, contributions, fluxd_std


#main part

RTYPE = txx  # or raw, trim, vt13, vt15, t01 ... t05
pickle_fn = f"dict_{sky}_{RTYPE}_dered.pickle"


#if the pickle file is already there, read it in and pickup where it left off
# ... this assumes we are resuming here and the indices, etc are all still valid
if os.path.exists(pickle_fn):
    with open(pickle_fn, 'rb') as f:
        sd = pickle.load(f)

    #where did we leave off ?
    last_idx = len(sd['rtype'])

else:
    last_idx = 0
    sd = {'detectids': [],
          'count': [],
          'rtype': [],
          'mag_bin_mids': [],
          'fluxd_stacks': [],
          'fluxde_stacks': [],
          'grids': [],
          'contributions': [],
          'fluxd_stds': []}

for i in range(last_idx,len(detection_lists)): # desc=f"mags"):
    # print(f"magbin: {mag_bin_mids[i]:0.1f} +/- 0.1 ...")
    #print(f"DUMMY: {txx} : g ~ {mag_bin_mids[i]:0.1f} : Stacking {len(detection_lists[i])} detections ...", flush=True)
    if len(detection_lists[i]) > 0:
        print(f"{txx} : g ~ {mag_bin_mids[i]:0.1f} : Stacking {len(detection_lists[i])} detections ...",flush=True)
        time_start = datetime.now()
        flx, flxe, gr, con, std = stack_list_observed(detection_lists[i], txx)

        sd['detectids'].append(detection_lists[i])
        sd['count'].append(len(detection_lists[i]))
        sd['rtype'].append(txx)
        sd['mag_bin_mids'].append(round(mag_bin_mids[i],1))
        sd['fluxd_stacks'].append(flx)
        sd['fluxde_stacks'].append(flxe)
        sd['grids'].append(gr)
        sd['contributions'].append(con)
        sd['fluxd_stds'].append(std)

        time_stop = datetime.now()
        dtime = time_stop - time_start
        print(f"{txx} : g ~ {mag_bin_mids[i]:0.1f} done. Iter Elapsed {dtime} , Total Elapsed {time_stop - time_outer_start}\n"
              f"        Iter {dtime.total_seconds()/len(detection_lists[i]):0.2f} s/det",flush=True)


        with open(pickle_fn,"wb") as f:
            pickle.dump(sd, f)

dtime = datetime.now() - time_outer_start
print(f"Done. Elapsed {dtime}.",flush=True)