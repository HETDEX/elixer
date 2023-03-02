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


if "--dust" in args:
    apply_dust_crrection = True
else:
    apply_dust_crrection = False

avg_type = "median" #"biweight",#"weighted_biweight",
avg_xlat = {"mean":"mn","median":"md","biweight":"bw","weighted_biweight":"wbw"}
table_outname = f"fiber_summary_{avg_xlat[avg_type]}_"
if apply_dust_crrection:
    table_outname += "dust_"
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

print(f"Cleaning from {len(FT)}...")

#remove the bad amps
sel = (ampflag_table['shotid'] == shotid) & (ampflag_table['flag'] == 0)  # here, 0 is bad , 1 is good
#bad_amp_list = ampflag_table['multiframe'][sel] #using the fiber index values which includes the bad amp flag
#badamp = np.array(ampflag_table['multiframe'][sel]) #this makes b'str' values, so don't jump to the np.array
bad = np.array([mf in ampflag_table['multiframe'][sel] for mf in FT['multiframe']])

#remove all the flagged fibers
keep = good_flag & ~bad
FT = FT[keep]

li = 265 #index for 4000AA
ri = 765 #index for 5000AA (actually 5002, but the last bin is not included in the slice)
wn = ri-li - (540-512) #number of bins (minus the chip gap) again don't need the +1 since the last bin is not included in the slice

#mask the chip gap with np.nan
FT['calfib'][:,512:540] = np.nan
FT['calfibe'][:,512:540] = np.nan
FT['calfib_ffsky'][:,512:540] = np.nan
#chip_gap = #4492-4548 mask out: idx 512 (4494) 540 (4540)  so would drop 4494-4548
#here ... now FT is "clean"
#sum over 4000-5000, sigma clip and trim  BOTH local and FFsky

#scan for all (or way too many zeros in any of the three arrays
#               all waves - chip gap - nonzeros (nan counts as a nonzero)(
FT['ll_zeros'] = 1036 - np.count_nonzero(FT['calfib'],axis=1)
FT['ff_zeros'] = 1036 - np.count_nonzero(FT['calfib_ffsky'],axis=1)
FT['er_zeros'] = 1036 - np.count_nonzero(FT['calfibe'],axis=1)

#most have at least 20-30 zeros at the edges
zero_sel = np.array(FT['ll_zeros'] <= 100) & np.array(FT['ff_zeros'] <= 100)  & np.array(FT['er_zeros'] <= 100)
FT = FT[zero_sel]
print(f"Cleaned to {len(FT)}")

#if apply dust
if apply_dust_crrection:
    print("Applying Dust Correction ...")

    dust_corr = deredden_spectra(G.CALFIB_WAVEGRID, SkyCoord(ra, dec, unit="deg"))
    FT['calfib'] *= dust_corr
    FT['calfib_ffsky'] *= dust_corr
    FT['calfibe'] *= dust_corr


print("Summing and Clipping ...")

FT['ll_sum'] = np.nansum(FT['calfib'][:,li:ri],axis=1) #technically sum of flux densitites so watch the factor of 2
FT['ff_sum'] = np.nansum(FT['calfib_ffsky'][:,li:ri],axis=1)
#FT['er_sum'] = np.sqrt(np.nansum(FT['calfibe'][:,li:ri]**2,axis=1)) #don't think I need this ? could make kick out any with large error?


#FT['ll_med'] = np.nanmedian(FT['calfib'][:,li:ri],axis=1) #this is the median flux denisity
#FT['ff_med'] = np.nanmedian(FT['calfib_ffsky'][:,li:ri],axis=1)

#FT['ll_std'] = np.nanstd(FT['calfib'][:,li:ri],axis=1)
#FT['ff_std'] = np.nanstd(FT['calfib_ffsky'][:,li:ri],axis=1)


ll_flux_ma = sigma_clip(FT['ll_sum'],sigma=3,maxiters=5,masked=True)
ff_flux_ma = sigma_clip(FT['ff_sum'],sigma=3,maxiters=5,masked=True)

#mask == reject (i.e. masked out)
FT['ll_mask'] = ll_flux_ma.mask
FT['ff_mask'] = ff_flux_ma.mask

#so keep is the inverse of mask
FT['ll_keep'] = ~ll_flux_ma.mask
FT['ff_keep'] = ~ff_flux_ma.mask
ll_N = np.count_nonzero(FT['ll_keep'])
ff_N = np.count_nonzero(FT['ff_keep'])

#then stack the remaining and figure the offset to zero for 4000-5000AA

print(f"Stacking  ({np.count_nonzero(FT['ll_keep'])},{np.count_nonzero(FT['ff_keep'])})...")

ll_stack, ll_stacke, _, _ = SU.stack_spectra(
                                                FT['calfib'][FT['ll_keep']],
                                                FT['calfibe'][FT['ll_keep']],
                                                np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(FT['ll_keep']),1)),
                                                grid=G.CALFIB_WAVEGRID,
                                                avg_type=avg_type,
                                                straight_error=False)


ff_stack, ff_stacke, _, _ = SU.stack_spectra(
                                                FT['calfib_ffsky'][FT['ff_keep']],
                                                FT['calfibe'][FT['ff_keep']],
                                                np.tile(G.CALFIB_WAVEGRID,(np.count_nonzero(FT['ff_keep']),1)),
                                                grid=G.CALFIB_WAVEGRID,
                                                avg_type=avg_type,#"biweight",#"weighted_biweight",
                                                straight_error=False)


#get the median offsets
ll_median_offset = np.nanmedian(FT['ll_sum'][FT['ll_keep']])/wn #median of the sums ... this is y-shift for the characteristic curve
ll_stderr_offset = np.nanstd(FT['ll_sum'][FT['ll_keep']]/wn)/ll_N #this is the error on that y-shift
#idea here is to use ll_stderr_offset as a new source of uncertainty ... seems to run about 1%
ff_median_offset = np.nanmedian(FT['ff_sum'][FT['ff_keep']])/wn #median of the sums ... this is y-shift for the characteristic curve
ff_stderr_offset = np.nanstd(FT['ff_sum'][FT['ff_keep']]/wn)/ff_N #this is the error on that y-shift


#also want to kill the ends

deg = 7
try:
    psel = np.full(len(G.CALFIB_WAVEGRID), False)
    psel[30:1005] = True
    psel = psel & ~np.isnan(ll_stack)
    ll_poly = np.polyfit(G.CALFIB_WAVEGRID[psel],ll_stack[psel],deg=deg)
except:
    ll_poly = np.zeros(deg)

try:
    psel = np.full(len(G.CALFIB_WAVEGRID), False)
    psel[30:1005] = True
    psel = psel & ~np.isnan(ff_stack)
    ff_poly = np.polyfit(G.CALFIB_WAVEGRID[psel],ff_stack[psel],deg=deg)
except:
    ff_poly = np.zeros(deg)

#basically just one row
T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int),
                 ('seeing',float),('response',float),
                 ('ll_median_offset',float),('ll_stderr_offset',float),
                 ('ff_median_offset',float),('ff_stderr_offset',float),
                 ('ll_stack', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_stacke', (float, len(G.CALFIB_WAVEGRID))),
                 ('ll_poly',(float,len(ll_poly))),
                 ('ff_stack', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_stacke', (float, len(G.CALFIB_WAVEGRID))),
                 ('ff_poly', (float, len(ff_poly))),
                 ]
               )

T.add_row([ra, dec, shotid, seeing, response,
           ll_median_offset, ll_stderr_offset, ff_median_offset, ff_stderr_offset,
           ll_stack, ll_stacke,ll_poly, ff_stacke,ff_stacke,ff_poly])

T.write(table_outname, format='fits', overwrite=True)

print("Done.",table_outname)
