"""
based on random_apertures.py, but uses previous determined coordinates rather than seeking new random apertures
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

coord_fn = None
if "--coord" in args:
    i = args.index("--coord")
    try:
        coord_fn = sys.argv[i + 1]
    except:
        print("bad coordinate filename specified")
        exit(-1)
else:
    print("no coordinate filename (--coord) specified")
    exit(-1)

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


table_outname = "coord_apertures_" + str(shotid) + ".fits"



#maybe this one was already done?
if op.exists(table_outname):
    exit(0)

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
aper_ct = 0
write_every = 100

T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int), ('ffsky',bool),
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


for ra,dec,shotid in zip(coord_ra,coord_dec,coord_shot):
    try:
        coord = SkyCoord(ra, dec, unit="deg")
        shot = shotid

        apt = get_spectra(coord, survey=survey_name, shotid=shot,
                          ffsky=ffsky, multiprocess=True, rad=aper,
                          tpmin=0.0, fiberweights=True, loglevel="ERROR")  # don't need the fiber weights


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
        dex_cont = c
        dex_cont_err = ce

        if g is None or np.isnan(g):
            dex_g = 99.0
            deg_g_err = 0
        else:
            dex_g = g
            dex_g_err = ge

        # check for any emission lines ... simple scan? or should we user elixer's method?
        #2022-12-09 should re-run and use this with fluxd*2.0 since that is how it is calibrated
        pos, status = elixer_spectrum.sn_peakdet_no_fit(wavelength, fluxd*2.0, fluxd_err*2.0, dx=3, dv=3, dvmx=4.0,
                                                        absorber=False,
                                                        spec_obj=None, return_status=True)
        if status != 0:
            continue  # some failure or 1 or more possible lines

        f50, apcor = SU.get_fluxlimits(ra, dec, [3800.0, 5000.0], shot)

        fluxd_sum = np.nansum(fluxd[215:966])
        fluxd_sum_wide = np.nansum(fluxd[65:966])
        fluxd_median = np.nanmedian(fluxd[215:966])
        fluxd_median_wide = np.nanmedian(fluxd[65:966])

        T.add_row([ra, dec, shotid, ffsky, seeing, response, apcor[1], f50[0], f50[1],
                   dex_g, dex_g_err, dex_cont, dex_cont_err,
                   fluxd_sum,fluxd_sum_wide,fluxd_median,fluxd_median_wide,
                   fluxd, fluxd_err])

        aper_ct += 1

    except Exception as e:
        print("Exception (2) !", e)
        continue

T.write(table_outname, format='fits', overwrite=True)

