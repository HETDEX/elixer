"""
simple file to pull a code specified number of random apertures, validating each against acceptance critieria,
and save to an astropy table on a per shot basis

intended for use with SLURM

caller specifies only the shotid

this code is modified to specify aperture size, number of apertures wanted, maximum number of apertures to attempt,
and any selection criteria, etc

based on hetdex_all_shots_random_empty_apertures notebooks on /work/03261/polonius/notebooks

"""
import sys
import os.path as op
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
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

if "--ffsky" in args:
    print("using ff sky subtraction")
    ffsky = True
else:
    print("using local sky subtraction")
    ffsky = False


if "--aper" in args:
    i = args.index("--aper")
    try:
        aper = float(sys.argv[i + 1])
        print(f"using specified  {aper}\" aperture")
    except:
        print("bad --aper specified")
        exit(-1)
else:
    print("using default 3.5\" aperture")
    aper = 3.5  # 3.5" aperture


if "--minmag" in args:
    i = args.index("--minmag")
    try:
        min_gmag = float(sys.argv[i + 1])
        print(f"using specified {min_gmag} bright limit mag.")
    except:
        print("bad --minmag specified")
        exit(-1)
else:
    print("using default 24.5 bright limit on mag.")
    min_gmag = 24.5  # 3.5" aperture


outfile = open("random_apertures_"+str(shotid)+".coord", "w+")  # 3500-3800 avg +/- 0.04
table_outname = "random_apertures_"+str(shotid) + ".fits"


#maybe this one was already done?
if op.exists(table_outname ):
    exit(0)

survey_name = "hdr3" #"hdr2.1"
hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
survey_table=survey.return_astropy_table()
FibIndex = FiberIndex(survey_name)
ampflag_table = Table.read(hetdex_api_config.badamp)

########################
# Main Loop
########################



random_fibers_per_shot = 500  # number of anchor fibers to select per shot (also then the maximum number of aperture attempts)
empty_apertures_per_shot = 200  # stop once we hit this number of "successful" apertures in a shot
ra_nudge = 0.75  # random nudge between 0 and this value in fiber center RA
dec_nudge = 0.75  # ditto for Dec
min_fibers = 15  # min number of fibers in an extraction
negative_flux_limit = -0.4e-17  # erg/s/cm2/AA ... more negative than this, assume something is wrong
hetdex_nearest_detection = 2.0 #in arcsec

wave_continuum_chunks_idx = [np.arange(15, 166, 1),  # 3500-3800
                             ]
# array of tuples that set the minimum and maximum average flux density value (e-17) that are okay.
# aligns with wave_continuum_chunks_idx
acceptable_fluxd = [(-0.04, 0.04),
                    ]



##
## grab all the fibers for this shot and randomly select random_fibers_per_shot
## go ahead and read the badamps for this shot so can more quickly compare
##
sel = (ampflag_table['shotid'] == shotid) & (ampflag_table['flag'] == 0)  # here, 0 is bad , 1 is good
#bad_amp_list = ampflag_table['multiframe'][sel] #using the fiber index values which includes the bad amp flag

idx = FibIndex.hdfile.root.FiberIndex.get_where_list("shotid==shot")
if idx is None or len(idx) < 20000:
    fi_shotids = FibIndex.hdfile.root.FiberIndex.read(field="shotid")
    idx = np.where(fi_shotids==shotid)[0]

rand_idx = np.random.choice(idx, size=random_fibers_per_shot, replace=False)
fibers_table = Table(FibIndex.hdfile.root.FiberIndex.read_coordinates(rand_idx)) #randomize the fiber ordering
mask_table = Table(FibIndex.fibermaskh5.root.Flags.read_coordinates(rand_idx))
super_tab = join(fibers_table, mask_table, "fiber_id")

#
# todo: if min_mag is 0 (say, not specified) attempt to make an overall depth estimate for the SHOT
# todo: based on the elixer method in spectrum_utilities::calc_dex_g_limit()
# todo: maybe just grab a bunch of random fibers, say up to 500?
# todo: toss out any with high flux counts or high noise (like in elixer)
# todo: then send the rest into SU.calc_dex_g_limit() along with the seeing FWHM etc
# todo: and use the resulting value as the mag limit? OR just send in ALL the fibers?





##
## iterate over the random fibers, verify is NOT on a bad amp, nudge the coordinate and extract
##
aper_ct = 0
write_every = 100

T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int),
                 ('seeing',float),('response',float),('apcor',float),
                 ('f50_3800',float),('f50_5000',float),
                 ('dex_g',float),('dex_g_err',float),('dex_cont',float),('dex_cont_err',float),
                 ('fluxd_sum',float),('fluxd_sum_wide',float),('fluxd_median',float),('fluxd_median_wide',float),
                 ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_err', (float, len(G.CALFIB_WAVEGRID)))])

sel = np.array(survey_table['shotid'] == shotid)
seeing = float(survey_table['fwhm_virus'][sel])
response = float(survey_table['response_4540'][sel])

del survey_table

elix_h5  = tables.open_file("/scratch/03946/hetdex/hdr3/detect/elixer.h5")
dex_ra = elix_h5.root.Detections.read(field='ra')
dex_dec = elix_h5.root.Detections.read(field='dec')
elix_h5.close()


for f in super_tab: #these fibers are in a random order so just iterating over them is random
                    #though it is possible to get two apertures at the same location (adjacent fibers and random
                    #ra, dec shifts could line them up ... would be exceptionally unlikely, so we will ignore)
                    #It is more likely to pick up one or a few that have overlapping apertures, but since this is
                    #PSF weighted, the effect of the overlap is limited and quickly falls off over 1-2"

    if aper_ct > empty_apertures_per_shot:
        # we have enough, move on to the next shot
        break

    if not (f['amp_flag'] and f['meteor_flag'] and f['gal_flag']):
        # if any of the three are false, skip this fiber and move on
        pass

    ra = np.random.uniform(-ra_nudge, ra_nudge) / 3600. + f['ra']
    dec = np.random.uniform(-dec_nudge, dec_nudge) / 3600. + f['dec']

    coord = SkyCoord(ra, dec, unit="deg")
    try:

        apt = get_spectra(coord, survey=survey_name, shotid=shot,
                          ffsky=ffsky, multiprocess=True, rad=aper,
                          tpmin=0.0, fiberweights=True, loglevel="ERROR")  # don't need the fiber weights
        try:
            if apt['fiber_weights'][0].shape[0] < min_fibers:
                continue  # too few fibers, probably on the edge or has dead fibers
        except Exception as e:
            continue

        # fluxd = np.nan_to_num(apt['spec'][0]) * 1e-17
        # fluxd_err = np.nan_to_num(apt['spec_err'][0]) * 1e-17
        # lets leave the nans in place
        fluxd = np.array(apt['spec'][0]) * 1e-17
        fluxd_err = np.array(apt['spec_err'][0]) * 1e-17
        wavelength = np.array(apt['wavelength'][0])

        dust_corr = deredden_spectra(wavelength, coord)
        fluxd *= dust_corr
        fluxd_err *= dust_corr

        ##
        ## check the extracted spectrum for evidence of continuum or emission lines
        ##


        g, c, ge, ce = elixer_spectrum.get_hetdex_gmag(fluxd, wavelength, fluxd_err)
        # g,ge,_,_ = elixer_spectrum.get_best_gmag(fluxd,fluxd_err,wavelength) #weird astropy issue, just use hetdex_gmag
        # if g is None:
        #    print(f"bad g {ra},{dec}")
        #    continue #point source mag is too bright, likely has something in it
        # g == None for negative continuum or below limit, so None itself is okay, need to check continuum
        if g is not None and g < min_gmag:
            continue  # point source mag is too bright, likely has something in it
        elif c < negative_flux_limit: #g might be None, if continuum is negative, then we don't get a magnitude
            continue  # point source mag is too bright, likely has something in it

        dex_cont = c
        dex_cont_err = ce

        if g is None or np.isnan(g):
            dex_g = 99.0
            deg_g_err = 0
        else:
            dex_g = g
            dex_g_err = ge
        ##
        ## check for excess in chunks of wavelength range, particularly in the blue, say 3500-3800
        ##
        fail = False
        for chunk, ok_fluxd in zip(wave_continuum_chunks_idx, acceptable_fluxd):
            # can't use the normal get_hetdex_gmag as it has safeties on the width of the spectrum and the covered wavelengths
            if ok_fluxd[0] < np.nanmedian(fluxd[chunk]) < ok_fluxd[1]:
                # acceptable
                pass
            else:
                fail = True
                break

        if fail:
            continue

        # check for any emission lines ... simple scan? or should we user elixer's method?
        #2022-12-09 should re-run and use this with fluxd*2.0 since that is how it is calibrated
        pos, status = elixer_spectrum.sn_peakdet_no_fit(wavelength, fluxd*2.0, fluxd_err*2.0, dx=3, dv=3, dvmx=4.0,
                                                        absorber=False,
                                                        spec_obj=None, return_status=True)
        if status != 0:
            continue  # some failure or 1 or more possible lines


        #todo: check for HETDEX detections? could check imaging, but would want to limit to anything brigher than 25.0 or 25.5
        fail = False
        sel_coord = np.array(abs(dex_ra - ra) < 5.0/3600) & np.array(abs(dex_dec-dec) < 5.0/3600)
        if np.count_nonzero(sel_coord) > 0:
            c1 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            carray = SkyCoord(ra=dex_ra[sel_coord] * u.deg, dec=dex_dec[sel_coord] * u.deg)
            dist = c1.separation(carray).value * 3600.
            if np.any(dist <= hetdex_nearest_detection):
                fail = True

        # elif len(pos) > 1:
        #    continue #possible emission lines

        if fail:
            continue

        f50, apcor = SU.get_fluxlimits(ra, dec, [3800.0, 5000.0], shot)

        # FUTURE/OPTIONAL: Should we check the imaging?? run a cutout and SEP check and only accept if nothing found?

        #
        # will run elixer on all these positions as well to make sure there are not other issues
        #
        # this is acceptable coords, so save:

        fluxd_sum = np.nansum(fluxd[215:966])
        fluxd_sum_wide = np.nansum(fluxd[65:966])
        fluxd_median = np.nanmedian(fluxd[215:966])
        fluxd_median_wide = np.nanmedian(fluxd[65:966])

        outfile.write(f"{ra}  {dec}  {shot}\n")
        outfile.flush()

        T.add_row([ra, dec, shotid, seeing, response, apcor[1], f50[0], f50[1],
                   dex_g, dex_g_err, dex_cont, dex_cont_err,
                   fluxd_sum,fluxd_sum_wide,fluxd_median,fluxd_median_wide,
                   fluxd, fluxd_err])

        aper_ct += 1

    except Exception as e:
        print("Exception (2) !", e)
        continue

T.write(table_outname, format='fits', overwrite=True)


