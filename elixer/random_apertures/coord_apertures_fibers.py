"""
based on random_apertures.py, but uses previous determined coordinates rather than seeking new random apertures
... saves off the FIBER spectra in the aperture, not the PSF weighted spectrum
"""
COORD_ID = None #"ll_model" #"ll_1050"
SKY_RESIDUAL_FITS_PATH = None #"/scratch/03261/polonius/random_apertures/all_fibers/all/"
#SKY_RESIDUAL_FITS_PREFIX = "fiber_summary_sym_bw_"
SKY_RESIDUAL_FITS_PREFIX = None #"fiber_summary_asym_bw_"
SKY_RESIDUAL_FITS_COL = None#"ll_stack_050"
NUM_TOP_FIBERS = 1 #for 2020 with 3 fibers, ends up being about 700k spectra to stack ... barely enough memory
                   #on the laptop to do that, if close everything else

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

coord_fn = None
if "--coord" in args:
    i = args.index("--coord")
    try:
        coord_fn = sys.argv[i + 1]
    except:
        print("bad coordinate filename specified")
        exit(-1)
elif "--shot" in args:
    i = args.index("--shot")
    try:
        coord_fn = "random_apertures_" + sys.argv[i + 1] +".coord"
    except:
        print("bad coordinate filename specified")
        exit(-1)

else:
    print("no coordinate filename (--coord) specified")
    exit(-1)


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

coord_ra, coord_dec, coord_shot = np.loadtxt(coord_fn,unpack=True)
coord_shot =coord_shot.astype(np.int64)
shotid = coord_shot[0]
shot = coord_shot[0]


#shot and shotid from the --coord file

if "--ffsky" in args:
    print("using ff sky subtraction")
    ffsky = True
else:
    print("using local sky subtraction")
    ffsky = False

if "--dust" in args:
    print("applying dust correction")
    dust = True
else:
    print("NO dust correction")
    dust = False

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

if COORD_ID is not None:
    table_outname = f"coord_fibers_{COORD_ID}_" + str(shotid) + ".fits"
else:
    table_outname = f"coord_fibers_" + str(shotid) + ".fits"

#maybe this one was already done?
if op.exists(table_outname):
    exit(0)

#get the single fiber residual for THIS shot


#            #residuals
#             if args.special >= 1000:
#                 G.SKY_RESIDUAL_PER_SHOT = True  # if True pull each residusl from the match shot, if False, use the universal model
#                 G.SKY_RESIDUAL_FITS_PATH = "/scratch/03261/polonius/random_apertures/all_fibers/all/"
#                 if args.special >= 2000:
#                     G.SKY_RESIDUAL_FITS_PREFIX = "fiber_summary_sym_bw_"
#                     col = args.special - 2000
#                 else:
#                     G.SKY_RESIDUAL_FITS_PREFIX = "fiber_summary_asym_bw_"
#                     col = args.special - 1000
#
#                 if args.ffsky:
#                     sky_label = "ff"
#                 else:
#                     sky_label = "ll"
#
#                 #col is now an integer 0 to 999, though only certain integers have meaning
#                 G.SKY_RESIDUAL_FITS_COL = f"{sky_label}_stack_{col:03}"
# if False:
#     if SKY_RESIDUAL_FITS_PATH is not None:
#         shot_sky_subtraction_residual = SU.fetch_per_shot_single_fiber_sky_subtraction_residual(SKY_RESIDUAL_FITS_PATH,
#                                                                                         shotid,
#                                                                                         SKY_RESIDUAL_FITS_COL,
#                                                                                         SKY_RESIDUAL_FITS_PREFIX)
#     else: #use the model
#         shot_sky_subtraction_residual = SU.fetch_universal_single_fiber_sky_subtraction_residual(
#                                                                                             ffsky=ffsky,
#                                                                                             hdr=G.HDR_Version)

    if shot_sky_subtraction_residual is None:
        print("FAIL!!! No single fiber shot residual retrieved.")

    fiber_flux_offset = -1 * shot_sky_subtraction_residual
else:
    fiber_flux_offset = None

survey_name = "hdr3" #"hdr2.1"
hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
survey_table=survey.return_astropy_table()
# FibIndex = FiberIndex(survey_name)
# ampflag_table = Table.read(hetdex_api_config.badamp)

########################
# Main Loop
########################

##
## iterate over the random fibers, verify is NOT on a bad amp, nudge the coordinate and extract
##
#aper_ct = 0
write_every = 100

T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int), ('ffsky',bool),
                 ('seeing',float),('response',float),('apcor',float),
                 ('f50_3800',float),('f50_5000',float),
                 ('dex_g',float),('dex_g_err',float),('dex_cont',float),('dex_cont_err',float),
                 ('fluxd_sum',float),('fluxd_sum_wide',float),('fluxd_median',float),('fluxd_median_wide',float),
                 ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_err', (float, len(G.CALFIB_WAVEGRID)))])
                 #('flux_mul', (float, len(G.CALFIB_WAVEGRID)))])

sel = np.array(survey_table['shotid'] == shotid)
seeing = float(survey_table['fwhm_virus'][sel])
response = float(survey_table['response_4540'][sel])

del survey_table

# elix_h5  = tables.open_file("/scratch/03946/hetdex/hdr3/detect/elixer.h5")
# dex_ra = elix_h5.root.Detections.read(field='ra')
# dex_dec = elix_h5.root.Detections.read(field='dec')
# elix_h5.close()


for ra,dec,shotid in zip(coord_ra,coord_dec,coord_shot):
    try:
        coord = SkyCoord(ra, dec, unit="deg")
        shot = shotid

        apt = get_spectra(coord, survey=survey_name, shotid=shot,
                          ffsky=ffsky, multiprocess=True, rad=aper,
                          tpmin=0.0, fiberweights=True, loglevel="ERROR",
                          fiber_flux_offset = fiber_flux_offset)  # DO want the weights here


        # fluxd = np.nan_to_num(apt['spec'][0]) * 1e-17
        # fluxd_err = np.nan_to_num(apt['spec_err'][0]) * 1e-17
        # lets leave the nans in place
        psf_fluxd = np.array(apt['spec'][0]) #* 1e-17
        psf_fluxd_err = np.array(apt['spec_err'][0]) #* 1e-17
        wavelength = np.array(apt['wavelength'][0])

        #the per fiber correction, since a common spectrum is wanted, cannot be made with dust correction that
        #depends on the RA, Dec
        if dust:
            dust_corr = deredden_spectra(wavelength, coord)
            psf_fluxd *= dust_corr
            psf_fluxd_err *= dust_corr


        #want the top xxx fibers by weight
        fiber_weights = apt['fiber_weights'][0]  # as array of ra,dec,weight


        ftb = get_fibers_table(shot, coord,
                                   radius=aper,
                                   # radius=60.0 * U.arcsec,
                                   survey=survey_name)


        #match up with the weights
        ftb_weights = [] #has the weights matched to the order of fibers in ftb
        for row in ftb:

            fw_idx = np.where((abs(fiber_weights[:, 0] - row['ra']) < 0.00003) &
                              (abs(fiber_weights[:, 1] - row['dec']) < 0.00003))[0]

            if len(fw_idx) == 1:
                ftb_weights.append(fiber_weights[fw_idx, 2][0])

        w_idx = np.argsort(ftb_weights)[::-1] #want decending order, so 1st is the highest weight from ftb



        ##
        ## check the extracted spectrum for evidence of continuum or emission lines
        ##

        # g, c, ge, ce = elixer_spectrum.get_hetdex_gmag(fluxd, wavelength, fluxd_err)
        dex_cont = -999
        dex_cont_err = 0

        #just for consistencey with other tables
        dex_g = 99.0
        dex_g_err = 0

        #what is the multiplicative uplift from fiber to PSF weighted aperture
        # ... this is just too noisy on a per extration basis
        if False:
            deg = 6
            psel = np.full(len(G.CALFIB_WAVEGRID), False)
            psel[30:1005] = True
            psel = psel & ~np.isnan(psf_fluxd)
            psel = psel & ~np.isinf(psf_fluxd)

            psf_poly = np.polyfit(G.CALFIB_WAVEGRID[psel], psf_fluxd[psel], deg=deg)
            psf_model = np.polyval(psf_poly, G.CALFIB_WAVEGRID)

            if ffsky:
                avg_fib_fluxd = np.nanmedian(ftb['calfib_ffsky'],axis=0)
            else:
                avg_fib_fluxd = np.nanmedian(ftb['calfib'],axis=0)

            psel = np.full(len(G.CALFIB_WAVEGRID), False)
            psel[30:1005] = True
            psel = psel & ~np.isnan(avg_fib_fluxd)
            psel = psel & ~np.isinf(avg_fib_fluxd)

            fib_poly = np.polyfit(G.CALFIB_WAVEGRID[psel], avg_fib_fluxd[psel], deg=deg)
            fib_model = np.polyval(fib_poly, G.CALFIB_WAVEGRID)

            flux_mul = psf_model / fib_model

        # check for any emission lines ... simple scan? or should we user elixer's method?
        #2022-12-09 should re-run and use this with fluxd*2.0 since that is how it is calibrated
        # pos, status = elixer_spectrum.sn_peakdet_no_fit(wavelength, fluxd*2.0, fluxd_err*2.0, dx=3, dv=3, dvmx=4.0,
        #                                                 absorber=False,
        #                                                 spec_obj=None, return_status=True)
        # if status != 0:
        #     continue  # some failure or 1 or more possible lines

        #f50, apcor = SU.get_fluxlimits(ra, dec, [3800.0, 5000.0], shot)

        f50 = [np.nan,np.nan]
        apcor = [np.nan,np.nan]

        #get the top few:
        for i in w_idx[0:NUM_TOP_FIBERS]:
            ra = ftb[i]['ra']
            dec = ftb[i]['dec']
            if ffsky:
                fluxd = ftb[i]['calfib_ffsky']
            else:
                fluxd = ftb[i]['calfib']
            fluxd_err = ftb[i]['calfibe']

            fluxd_sum = np.nansum(fluxd[215:966])
            fluxd_sum_wide = np.nansum(fluxd[65:966])
            fluxd_median = np.nanmedian(fluxd[215:966])
            fluxd_median_wide = np.nanmedian(fluxd[65:966])

            T.add_row([ra, dec, shotid, ffsky, seeing, response, apcor[1], f50[0], f50[1],
                       dex_g, dex_g_err, dex_cont, dex_cont_err,
                       fluxd_sum,fluxd_sum_wide,fluxd_median,fluxd_median_wide,
                       fluxd, fluxd_err])

        #aper_ct += 1

    except Exception as e:
        print("Exception (2) !", e)
        continue

T.write(table_outname, format='fits', overwrite=True)


