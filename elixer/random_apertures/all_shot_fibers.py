"""
grab all fibers for a shot
subselect the not "bad" ones

stack calfib, calfib_ffsky
correct those two stack with MW dust correction

subselect on 4000-5000 AA; 3 sigma at 5 repeat

save the two stacks

save the sum of 4000 to 5000 AA for each

record the shot info (PSF, exposure time, etc)

no pre-selection trimming

"""
import sys
import os.path as op
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
from astropy.stats import sigma_clip
import astropy.units as u
import copy
import glob

from hetdex_api.config import HDRconfig
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import Survey,FiberIndex
from hetdex_tools.get_spec import get_spectra
from hetdex_api.extinction import *  #includes deredden_spectra

from elixer import spectrum as elixer_spectrum
from elixer import spectrum_utilities as SU
from elixer import global_config as G
from elixer import weighted_biweight as weighted_biweight
import astropy.stats.biweight as biweight


#trim to exclude values outside this range
zone1_fluxd_range = (-2.0,10.0) #3500-4000AA
zone2_fluxd_range = (-2.0,2.0) #4000-5000AA
zone3_fluxd_range = (-2.0,2.0) #5000-5500


def whole_shot_by_pct(fiber_table,trim_pct, ffsky=False, avg_type = 'biweight', enforce_fluxd_range=False):#, use_counts=False):
    FT = fiber_table
    sel = np.full(len(FT), True)
    calfibe = FT['calfibe'][sel]
    calfibe_ct = FT['calfibe_counts'][sel]
    if ffsky:
        calfib = FT['calfib_ffsky'][sel]
        calfib_ct = FT['calfib_counts_ffsky'][sel]
    else:
        calfib = FT['calfib'][sel]
        calfib_ct = FT['calfib_counts'][sel]

    # calfib = FT['calfib_counts'][sel]
    # calfibe = FT['calfibe_counts'][sel]


    #compute fluxd by zone
    rt, *_ = SU.getnearpos(G.CALFIB_WAVEGRID, 4000)
    zone1_calfib = np.nanmean(calfib[:, 0:rt], axis=1)

    lt = rt
    rt, *_ = SU.getnearpos(G.CALFIB_WAVEGRID, 5000)
    zone2_calfib = np.nanmean(calfib[:, lt:rt], axis=1)

    lt = rt
    # rt,*_ = SU.getnearpos(G.CALFIB_WAVEGRID,5000)
    zone3_calfib = np.nanmean(calfib[:, lt:], axis=1)


    if enforce_fluxd_range:
        trim_sel = np.array(zone1_fluxd_range[0] <= zone1_calfib) & np.array(zone1_calfib <= zone1_fluxd_range[1]) & \
                   np.array(zone2_fluxd_range[0] <= zone2_calfib) & np.array(zone2_calfib <= zone2_fluxd_range[1]) & \
                   np.array(zone3_fluxd_range[0] <= zone3_calfib) & np.array(zone3_calfib <= zone3_fluxd_range[1])

        calfib = calfib[trim_sel]
        calfibe = calfibe[trim_sel]

        calfib_ct = calfib[trim_sel]
        calfibe_ct = calfibe[trim_sel]

        zone1_calfib = zone1_calfib[trim_sel]
        zone2_calfib = zone2_calfib[trim_sel]
        zone3_calfib = zone3_calfib[trim_sel]

    # now apply pct cuts
    # zone_calfib = np.sort(zone1_calfib)
    zone1_top_cut_value = np.sort(zone1_calfib)[int(len(zone1_calfib) * (1 - trim_pct))]
    # del zone_calfib

    # zone_calfib = np.sort(zone2_calfib)
    zone2_top_cut_value = np.sort(zone2_calfib)[int(len(zone2_calfib) * (1 - trim_pct))]
    # del zone_calfib

    # zone_calfib = np.sort(zone3_calfib)
    zone3_top_cut_value = np.sort(zone3_calfib)[int(len(zone3_calfib) * (1 - trim_pct))]
    # del zone_calfib

    trim_sel = np.array(zone1_calfib < zone1_top_cut_value) & \
               np.array(zone2_calfib < zone2_top_cut_value) & \
               np.array(zone3_calfib < zone3_top_cut_value)


    stack, stacke, _, _ = SU.stack_spectra(calfib[trim_sel],
                                                 calfibe[trim_sel],
                                                 np.tile(G.CALFIB_WAVEGRID, (np.count_nonzero(trim_sel), 1)),
                                                 grid=G.CALFIB_WAVEGRID,
                                                 avg_type=avg_type,  # "weighted_biweight",
                                                 straight_error=False)

    stack_ct, stacke_ct, _, _ = SU.stack_spectra(calfib_ct[trim_sel],
                                                 calfibe_ct[trim_sel],
                                                 np.tile(G.CALFIB_WAVEGRID, (np.count_nonzero(trim_sel), 1)),
                                                 grid=G.CALFIB_WAVEGRID,
                                                 avg_type=avg_type,  # "weighted_biweight",
                                                 straight_error=False)

    return stack, stacke, np.count_nonzero(trim_sel), stack_ct, stacke_ct




#todo: loop over by exposure x amp x IFU (for LL sky) and by exposure for FF sky
#todo: treat each wavelength bin independently
#def independent_wavebins (fiber_table,trim_pct,ffsky=False, avg_type = 'biweight'):





#get the shot from the command line
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


# if "--dust" in args:
#     apply_dust_crrection = True
# else:
#     apply_dust_crrection = False
apply_dust_crrection = False


# if "--counts" in args:
#     use_counts = True
# else:
#     use_counts = False

avg_type = "median" #"biweight",#"weighted_biweight",
avg_xlat = {"mean":"mn","median":"md","biweight":"bw","weighted_biweight":"wbw"}
table_outname = f"fiber_summary_{avg_xlat[avg_type]}_"
if apply_dust_crrection:
    table_outname += "dust_"

# if use_counts:
#     table_outname += "counts_"

table_outname += f"{shotid}.fits"


#maybe this one was already done?
if op.exists(table_outname ):
    print(f"{table_outname} already created. Skipping.")
    exit(0)

print(f"Initializing {table_outname} ...")

survey_name = "hdr3" #"hdr2.1"
hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
survey_table=survey.return_astropy_table()
FibIndex = FiberIndex(survey_name)
ampflag_table = Table.read(hetdex_api_config.badamp)

sel = np.array(survey_table['shotid'] == shotid)
seeing = float(survey_table['fwhm_virus'][sel])
response = float(survey_table['response_4540'][sel])
ra = float(survey_table['ra'][sel])
dec = float(survey_table['dec'][sel])

del survey_table


print("Loading fibers ...")

idx = FibIndex.hdfile.root.FiberIndex.get_where_list("shotid==shot")
fibers_table = Table(FibIndex.hdfile.root.FiberIndex.read_coordinates(idx))
mask_table = Table(FibIndex.fibermaskh5.root.Flags.read_coordinates(idx))
super_tab = join(fibers_table, mask_table, "fiber_id")
#want to keep all those with ['flag'] == True #actually means it is good
good_flag = np.array(super_tab['flag'])

#get the full fiber table for the shot
FT = get_fibers_table(shot=shot)
FT_total_ct = len(FT) #total number of fibers per in the shot

print(f"Cleaning from {len(FT)}...")

#remove the bad amps
sel = (ampflag_table['shotid'] == shotid) & (ampflag_table['flag'] == 0)  # here, 0 is bad , 1 is good
#bad_amp_list = ampflag_table['multiframe'][sel] #using the fiber index values which includes the bad amp flag
#badamp = np.array(ampflag_table['multiframe'][sel]) #this makes b'str' values, so don't jump to the np.array
bad = np.array([mf in ampflag_table['multiframe'][sel] for mf in FT['multiframe']])

del ampflag_table
del fibers_table
del mask_table
del super_tab

#remove all the flagged fibers
keep = good_flag & ~bad
FT = FT[keep]

li = 265 #index for 4000AA
ri = 765 #index for 5000AA (actually 5002, but the last bin is not included in the slice)
wn = ri-li - (540-512) #number of bins (minus the chip gap) again don't need the +1 since the last bin is not included in the slice

# moved to the individual stacking sections as not all stacking options want these masked
# #mask the chip gap with np.nan
# FT['calfib'][:,512:540] = np.nan
# FT['calfibe'][:,512:540] = np.nan
# FT['calfib_ffsky'][:,512:540] = np.nan
# #chip_gap = #4492-4548 mask out: idx 512 (4494) 540 (4540)  so would drop 4494-4548
# #here ... now FT is "clean"
# #sum over 4000-5000, sigma clip and trim  BOTH local and FFsky


# #scan for all (or way too many zeros in any of the three arrays
# #               all waves - chip gap - nonzeros (nan counts as a nonzero)(
FT['ll_zeros'] = 1036 - np.count_nonzero(FT['calfib'],axis=1)
FT['ff_zeros'] = 1036 - np.count_nonzero(FT['calfib_ffsky'],axis=1)
FT['er_zeros'] = 1036 - np.count_nonzero(FT['calfibe'],axis=1)

#most have at least 20-30 zeros at the edges
zero_sel = np.array(FT['ll_zeros'] <= 100) & np.array(FT['ff_zeros'] <= 100)  & np.array(FT['er_zeros'] <= 100)
FT = FT[zero_sel]
FT_cleaned_ct = len(FT) #total number of fibers per in the shot
print(f"Cleaned to {len(FT)}")

#if apply dust #since this is applied later by HETDEX_API, etc, probably should NOT apply here
if apply_dust_crrection:
    print("Applying Dust Correction ...")

    dust_corr = deredden_spectra(G.CALFIB_WAVEGRID, SkyCoord(ra, dec, unit="deg"))
    FT['calfib'] *= dust_corr
    FT['calfib_ffsky'] *= dust_corr
    FT['calfibe'] *= dust_corr


#for pct in [0.05,0.10,0.15]:
ll_stack_05, ll_stacke_05, ll_ct_05, ll_stack_ct_05, ll_stacke_ct_05, = whole_shot_by_pct(fiber_table=FT,trim_pct=0.05, ffsky=False,
                                        avg_type = 'biweight', enforce_fluxd_range=True)

ll_stack_10, ll_stacke_10, ll_ct_10, ll_stack_ct_10, ll_stacke_ct_10 = whole_shot_by_pct(fiber_table=FT,trim_pct=0.10, ffsky=False,
                                        avg_type = 'biweight', enforce_fluxd_range=True)

ll_stack_15, ll_stacke_15, ll_ct_15, ll_stack_ct_15, ll_stacke_ct_15 = whole_shot_by_pct(fiber_table=FT,trim_pct=0.15, ffsky=False,
                                        avg_type = 'biweight', enforce_fluxd_range=True)


ff_stack_05, ff_stacke_05, ff_ct_05, ff_stack_ct_05, ff_stacke_ct_05 = whole_shot_by_pct(fiber_table=FT,trim_pct=0.05, ffsky=True,
                                        avg_type = 'biweight', enforce_fluxd_range=True)

ff_stack_10, ff_stacke_10, ff_ct_10, ff_stack_ct_10, ff_stacke_ct_10 = whole_shot_by_pct(fiber_table=FT,trim_pct=0.10, ffsky=True,
                                        avg_type = 'biweight', enforce_fluxd_range=True)

ff_stack_15, ff_stacke_15, ff_ct_15, ff_stack_ct_15, ff_stacke_ct_15 = whole_shot_by_pct(fiber_table=FT,trim_pct=0.15, ffsky=True,
                                        avg_type = 'biweight', enforce_fluxd_range=True)

#basically just one row
T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int),
                 ('seeing',float),('response',float),
                 ('fiber_total_ct',float),('fiber_cleaned_ct',float),
                 ('ll_ct_05',float), ('ff_ct_05',float),('ll_ct_10', float),('ff_ct_10', float),('ll_ct_15', float),('ff_ct_15', float),

                 ('ll_stack_05', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke_05', (float, len(G.CALFIB_WAVEGRID))),

                 ('ff_stack_05', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke_05', (float, len(G.CALFIB_WAVEGRID))),


                 ('ll_stack_10', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke_10', (float, len(G.CALFIB_WAVEGRID))),

                 ('ff_stack_10', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke_10', (float, len(G.CALFIB_WAVEGRID))),


                 ('ll_stack_15', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke_15', (float, len(G.CALFIB_WAVEGRID))),

                 ('ff_stack_15', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke_15', (float, len(G.CALFIB_WAVEGRID)))


                 ('ll_stack_ct_05', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke_ct_05', (float, len(G.CALFIB_WAVEGRID))),

                 ('ff_stack_ct_05', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke_ct_05', (float, len(G.CALFIB_WAVEGRID))),

                 ('ll_stack_ct_10', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke_ct_10', (float, len(G.CALFIB_WAVEGRID))),

                 ('ff_stack_ct_10', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke_ct_10', (float, len(G.CALFIB_WAVEGRID))),

                 ('ll_stack_ct_15', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke_ct_15', (float, len(G.CALFIB_WAVEGRID))),

                 ('ff_stack_ct_15', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke_ct_15', (float, len(G.CALFIB_WAVEGRID)))
                 ])

T.add_row([ra, dec, shotid, seeing, response,
           FT_total_ct, FT_cleaned_ct,
           ll_ct_05,ff_ct_05,ll_ct_10,ff_ct_10,ll_ct_15,ff_ct_15,

           ll_stack_05, ll_stacke_05,  ff_stack_05,ff_stacke_05,
           ll_stack_10, ll_stacke_10,  ff_stack_10,ff_stacke_10,
           ll_stack_15, ll_stacke_15,  ff_stack_15,ff_stacke_15,

           ll_stack_ct_05, ll_stacke_ct_05, ff_stack_ct_05, ff_stacke_ct_05,
           ll_stack_ct_10, ll_stacke_ct_10, ff_stack_ct_10, ff_stacke_ct_10,
           ll_stack_ct_15, ll_stacke_ct_15, ff_stack_ct_15, ff_stacke_ct_15,
           ])

T.write(table_outname, format='fits', overwrite=True)

#
# ###
# ## todo: three options
# ## 1) clipping
# ## 2) keep bottom 2/3 of 4000-5000AA sum
# ## 3) keep bottom 2/3 in EACH BIN (ignoring chip gap)
# ##
#
# if False: #clipping version
#
#     # mask the chip gap with np.nan
#     FT['calfib'][:, 512:540] = np.nan
#     FT['calfibe'][:, 512:540] = np.nan
#     FT['calfib_ffsky'][:, 512:540] = np.nan
#     # chip_gap = #4492-4548 mask out: idx 512 (4494) 540 (4540)  so would drop 4494-4548
#     # here ... now FT is "clean"
#     # sum over 4000-5000, sigma clip and trim  BOTH local and FFsky
#
#
#     print("Summing and Clipping ...")
#
#     FT['ll_sum'] = np.nansum(FT['calfib'][:,li:ri],axis=1) #technically sum of flux densitites so watch the factor of 2
#     FT['ff_sum'] = np.nansum(FT['calfib_ffsky'][:,li:ri],axis=1)
#     #FT['er_sum'] = np.sqrt(np.nansum(FT['calfibe'][:,li:ri]**2,axis=1)) #don't think I need this ? could make kick out any with large error?
#
#
#     #FT['ll_med'] = np.nanmedian(FT['calfib'][:,li:ri],axis=1) #this is the median flux denisity
#     #FT['ff_med'] = np.nanmedian(FT['calfib_ffsky'][:,li:ri],axis=1)
#
#     #FT['ll_std'] = np.nanstd(FT['calfib'][:,li:ri],axis=1)
#     #FT['ff_std'] = np.nanstd(FT['calfib_ffsky'][:,li:ri],axis=1)
#
#
#     ll_flux_ma = sigma_clip(FT['ll_sum'],sigma=3,maxiters=5,masked=True)
#     ff_flux_ma = sigma_clip(FT['ff_sum'],sigma=3,maxiters=5,masked=True)
#
#     #mask == reject (i.e. masked out)
#     FT['ll_mask'] = ll_flux_ma.mask
#     FT['ff_mask'] = ff_flux_ma.mask
#
#     #so keep is the inverse of mask
#     FT['ll_keep'] = ~ll_flux_ma.mask
#     FT['ff_keep'] = ~ff_flux_ma.mask
#     ll_N = np.count_nonzero(FT['ll_keep'])
#     ff_N = np.count_nonzero(FT['ff_keep'])
#
#     #then stack the remaining and figure the offset to zero for 4000-5000AA
#
#     print(f"Stacking  ({np.count_nonzero(FT['ll_keep'])},{np.count_nonzero(FT['ff_keep'])})...")
#
#     ll_stack, ll_stacke, _, _ = SU.stack_spectra(
#                                                     FT['calfib'][FT['ll_keep']],
#                                                     FT['calfibe'][FT['ll_keep']],
#                                                     np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(FT['ll_keep']),1)),
#                                                     grid=G.CALFIB_WAVEGRID,
#                                                     avg_type=avg_type,
#                                                     straight_error=False)
#
#
#     ff_stack, ff_stacke, _, _ = SU.stack_spectra(
#                                                     FT['calfib_ffsky'][FT['ff_keep']],
#                                                     FT['calfibe'][FT['ff_keep']],
#                                                     np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(FT['ff_keep']),1)),
#                                                     grid=G.CALFIB_WAVEGRID,
#                                                     avg_type=avg_type,#"biweight",#"weighted_biweight",
#                                                     straight_error=False)
#
#     #already masked, but needed for naming consistency later
#     ll_stack_mask = ll_stack
#     ll_stacke_mask = ll_stacke
#     ff_stack_mask = ff_stack
#     ff_stacke_mask = ff_stacke
#
#     FT['ll_sum'] = np.nansum(FT['calfib'][:, li:ri], axis=1)  # technically sum of flux densitites so watch the factor of 2
#     FT['ff_sum'] = np.nansum(FT['calfib_ffsky'][:, li:ri], axis=1)
#
#     ll_median_offset = np.nanmedian(
#         FT['ll_sum'][FT['ll_keep']]) / wn  # median of the sums ... this is y-shift for the characteristic curve
#     ll_stderr_offset = np.nanstd(FT['ll_sum'][FT['ll_keep']] / wn) / ll_N  # this is the error on that y-shift
#     # idea here is to use ll_stderr_offset as a new source of uncertainty ... seems to run about 1%
#     ff_median_offset = np.nanmedian(
#         FT['ff_sum'][FT['ff_keep']]) / wn  # median of the sums ... this is y-shift for the characteristic curve
#     ff_stderr_offset = np.nanstd(FT['ff_sum'][FT['ff_keep']] / wn) / ff_N  # this is the error on that y-shift
#
#
#
#
# elif False: #bottom 2/3 of 4000-5000AA
#
#     # mask the chip gap with np.nan
#     FT['calfib'][:, 512:540] = np.nan
#     FT['calfibe'][:, 512:540] = np.nan
#     FT['calfib_ffsky'][:, 512:540] = np.nan
#     # chip_gap = #4492-4548 mask out: idx 512 (4494) 540 (4540)  so would drop 4494-4548
#     # here ... now FT is "clean"
#     # sum over 4000-5000, sigma clip and trim  BOTH local and FFsky
#
#
#     print("Summing and Clipping ...")
#
#     FT['ll_sum'] = np.nansum(FT['calfib'][:,li:ri],axis=1) #technically sum of flux densitites so watch the factor of 2
#     FT['ff_sum'] = np.nansum(FT['calfib_ffsky'][:,li:ri],axis=1)
#     #FT['er_sum'] = np.sqrt(np.nansum(FT['calfibe'][:,li:ri]**2,axis=1)) #don't think I need this ? could make kick out any with large error?
#
#     # keep the lower 2/3  off the sums?
#     sort = np.argsort(FT['ll_sum'])
#     fd = np.array(FT['calfib'])[sort]
#     # error needs to follow fd
#     fde = np.array(FT['calfibe'])[sort]
#
#     sel = np.full(len(fd), True)
#     # kill any negative by 5 or 6 sigma
#     sig = np.nanstd(FT['ll_sum'])
#     mn = np.nanmean(FT['ll_sum'])
#     sel = sel & np.array(FT['ll_sum'] > (mn - 5 * sig))
#
#     # remove top xx
#     sel[-int(len(fd) * 0.33):] = False
#     ll_N = np.count_nonzero(sel)
#
#
#     #then stack the remaining and figure the offset to zero for 4000-5000AA
#
#     #print(f"Stacking  ({np.count_nonzero(FT['ll_keep'])},{np.count_nonzero(FT['ff_keep'])})...")
#
#     ll_stack, ll_stacke, _, _ = SU.stack_spectra(
#                                                     FT['calfib'][sel],
#                                                     FT['calfibe'][sel],
#                                                     np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(sel),1)),
#                                                     grid=G.CALFIB_WAVEGRID,
#                                                     avg_type=avg_type,
#                                                     straight_error=False)
#
#     ll_median_offset = np.nanmedian(
#         FT['ll_sum'][sel]) / wn  # median of the sums ... this is y-shift for the characteristic curve
#     ll_stderr_offset = np.nanstd(FT['ll_sum'][sel] / wn) / ll_N  # this is the error on that y-shift
#
#
#     # keep the lower 2/3  off the sums?
#     sort = np.argsort(FT['ff_sum'])
#     fd = np.array(FT['calfib_ffsky'])[sort]
#     # error needs to follow fd
#     fde = np.array(FT['calfibe'])[sort]
#
#     # kill any negative by 5 or 6 sigma
#     sel = np.full(len(fd), True)
#     sig = np.nanstd(FT['ff_sum'])
#     mn = np.nanmean(FT['ff_sum'])
#     sel = sel & np.array(FT['ff_sum'] > (mn - 5 * sig))
#
#     # remove top xx
#     sel[-int(len(fd) * 0.33):] = False
#     ff_N = np.count_nonzero(sel)
#
#     ff_stack, ff_stacke, _, _ = SU.stack_spectra(
#                                                     FT['calfib_ffsky'][FT['ff_keep']],
#                                                     FT['calfibe'][FT['ff_keep']],
#                                                     np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(FT['ff_keep']),1)),
#                                                     grid=G.CALFIB_WAVEGRID,
#                                                     avg_type=avg_type,#"biweight",#"weighted_biweight",
#                                                     straight_error=False)
#
#     #already masked, but needed for naming consistency later
#     ll_stack_mask = ll_stack
#     ll_stacke_mask = ll_stacke
#     ff_stack_mask = ff_stack
#     ff_stacke_mask = ff_stacke
#
#     FT['ll_sum'] = np.nansum(FT['calfib'][:, li:ri], axis=1)  # technically sum of flux densitites so watch the factor of 2
#     FT['ff_sum'] = np.nansum(FT['calfib_ffsky'][:, li:ri], axis=1)
#
#
#     # idea here is to use ll_stderr_offset as a new source of uncertainty ... seems to run about 1%
#     ff_median_offset = np.nanmedian(
#         FT['ff_sum'][sel]) / wn  # median of the sums ... this is y-shift for the characteristic curve
#     ff_stderr_offset = np.nanstd(FT['ff_sum'][sel] / wn) / ff_N  # this is the error on that y-shift
#
#
#
# elif True: #bottom 2/3 of each wavelength bin
#     #stupid simple loop
#     ll_stack = np.zeros(len(G.CALFIB_WAVEGRID))
#     ll_stacke = np.zeros(len(G.CALFIB_WAVEGRID))
#     ff_stack = np.zeros(len(G.CALFIB_WAVEGRID))
#     ff_stacke = np.zeros(len(G.CALFIB_WAVEGRID))
#     for i in range(len(G.CALFIB_WAVEGRID)):
#         try:
#             #FF sky
#             sort = np.argsort(FT['calfib_ffsky'][:,i])
#             fd = np.array(FT['calfib_ffsky'][:, i])[sort]
#             # error needs to follow fd
#             fde = np.array(FT['calfibe'][:,i])[sort]
#             sel = np.full(len(fd), True)
#
#             #kill any negative by 5 or 6 sigma
#             sig = np.nanstd(fd)
#             mn = np.nanmean(fd)
#             sel = sel & np.array(fd > (mn-5*sig) )
#
#             #remove top xx
#             sel[-int(len(fd)*0.33):] = False
#
#             flux_array = fd[sel]
#             fluxe_array = fde[sel]
#
#             #get the average
#             f = biweight.biweight_location(flux_array)
#             fe = biweight.biweight_scale(flux_array/np.sqrt(len(flux_array)))
#
#             ff_stack[i] = f
#             ff_stacke[i] = fe
#         except Exception as e:
#             print("Exception",e)
#
#         #Local Sky
#         try:
#             sort = np.argsort(FT['calfib'][:, i])
#             fd = np.array(FT['calfib'][:, i])[sort]
#             # error needs to follow fd
#             fde = np.array(FT['calfibe'][:, i])[sort]
#
#             # kill any negative by 5 or 6 sigma
#             sig = np.nanstd(fd)
#             mn = np.nanmean(fd)
#             sel = sel & np.array(fd > (mn - 5 * sig))
#
#             # remove top xx
#             sel[-int(len(fd) * 0.33):] = False
#
#             flux_array = fd[sel]
#             fluxe_array = fde[sel]
#
#             # get the average
#             f = biweight.biweight_location(flux_array)
#             fe = biweight.biweight_scale(flux_array / np.sqrt(len(flux_array)))
#
#             ll_stack[i] = f
#             ll_stacke[i] = fe
#         except Exception as e:
#             print("Exception",e)
#
#     #now, need to mask for the polyfit
#     ll_stack_mask = copy.copy(ll_stack)
#     ll_stacke_mask = copy.copy(ll_stacke)
#     ff_stack_mask = copy.copy(ff_stack)
#     ff_stacke_mask = copy.copy(ff_stacke)
#
#     # mask the chip gap with np.nan
#     ll_stack_mask[512:540] = np.nan
#     ll_stacke_mask[512:540] = np.nan
#     ff_stack_mask[512:540] = np.nan
#     ff_stacke_mask[512:540] = np.nan
#
#     # get the median offsets
#     ll_median_offset = np.nanmedian(ll_stack_mask[li:ri])
#     ll_stderr_offset = 0 #np.nanstd(FT['ll_sum'][FT['ll_keep']] / wn) / ll_N  # this is the error on that y-shift
#     # idea here is to use ll_stderr_offset as a new source of uncertainty ... seems to run about 1%
#     ff_median_offset = np.nanmedian(ff_stack_mask[li:ri])
#     ff_stderr_offset = 0 #np.nanstd(FT['ff_sum'][FT['ff_keep']] / wn) / ff_N  # this is the error on that y-shift
#
#
#
# #always do these steps (depending on the stacking above, the nans might already be assigned, but it does not matter
# # to re-assign them to same
# # mask the chip gap with np.nan
# FT['calfib'][:, 512:540] = np.nan
# FT['calfibe'][:, 512:540] = np.nan
# FT['calfib_ffsky'][:, 512:540] = np.nan
# # chip_gap = #4492-4548 mask out: idx 512 (4494) 540 (4540)  so would drop 4494-4548
# # here ... now FT is "clean"
# # sum over 4000-5000, sigma clip and trim  BOTH local and FFsky
#
# FT['ll_sum'] = np.nansum(FT['calfib'][:,li:ri],axis=1) #technically sum of flux densitites so watch the factor of 2
# FT['ff_sum'] = np.nansum(FT['calfib_ffsky'][:,li:ri],axis=1)
#
#
#
#
#
# deg = 7
# try:
#     psel = np.full(len(G.CALFIB_WAVEGRID), False)
#     psel[30:1005] = True
#     psel = psel & ~np.isnan(ll_stack_mask)
#     ll_poly = np.polyfit(G.CALFIB_WAVEGRID[psel],ll_stack_mask[psel],deg=deg)
# except:
#     ll_poly = np.zeros(deg)
#
# try:
#     psel = np.full(len(G.CALFIB_WAVEGRID), False)
#     psel[30:1005] = True
#     psel = psel & ~np.isnan(ff_stack_mask)
#     ff_poly = np.polyfit(G.CALFIB_WAVEGRID[psel],ff_sff_stack_masktack[psel],deg=deg)
# except:
#     ff_poly = np.zeros(deg)

# #basically just one row
# T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int),
#                  ('seeing',float),('response',float),
#                  ('fiber_total_ct',float),('fiber_cleaned_ct',float),
#                  ('ll_ct',float),
#                  ('ll_stack', (float, len(G.CALFIB_WAVEGRID))),
#                  ('ll_stacke', (float, len(G.CALFIB_WAVEGRID))),
#                  ('ff_ct',float),
#                  ('ff_stack', (float, len(G.CALFIB_WAVEGRID))),
#                  ('ff_stacke', (float, len(G.CALFIB_WAVEGRID))),
#                  ])
#
# T.add_row([ra, dec, shotid, seeing, response,
#            FT_total_ct, FT_cleaned_ct,ll_ct,
#            ll_stack, ll_stacke, ff_ct , ff_stack,ff_stacke])
#
# T.write(table_outname, format='fits', overwrite=True)

print("Done.",table_outname)
