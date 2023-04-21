"""
Set of simple functions that act on numpy arrays mostly
Keep this simple ... no complex stucture, not a lot of error control
"""

from __future__ import print_function


try:
    from hetdex_tools.get_spec import get_spectra as hda_get_spectra
except Exception as e:
    print("WARNING!!!! CANNOT IMPORT hetdex_api tools get_spectra: ",e)

try:
    from hetdex_api import survey as hda_survey
except Exception as e:
    print("WARNING!!!! CANNOT IMPORT hetdex_api survey: ",e)

try:
    from hetdex_api.flux_limits.shot_sensitivity import ShotSensitivity
except Exception as e:
    print("WARNING!!!! CANNOT IMPORT hetdex_api.flux_limits.shot_sensitivity ShotSensitivity: ",e)

try:
    from hetdex_api.extract import Extract
    # from hetdex_api.shot import get_fibers_table as hda_get_fibers_table
except Exception as e:
    print("WARNING!!!! CANNOT IMPORT hetdex_api extract: ", e)

try:
    from elixer import global_config as G
    from elixer import weighted_biweight as weighted_biweight
    from elixer import utilities as utils
except:
    import global_config as G
    import weighted_biweight as weighted_biweight
    import utilities as utils

import numpy as np
import pickle
import os.path as op
from astropy.table import Table
import astropy.constants
import astropy.units as U
from astropy.coordinates import SkyCoord
import astropy.cosmology as Cosmo
import astropy.stats.biweight as biweight
from photutils import CircularAperture #pixel coords
from photutils import aperture_photometry

from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.special import factorial

import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter,MaxNLocator

#SU = Simple Universe (concordance)
SU_H0 = 70. * U.km / U.s / U.Mpc
SU_Omega_m0 = 0.3
SU_T_CMB = 2.73


#
# #these are for the older peak finder (based on direction change)
# MIN_FWHM = 2.0 #AA (must xlat to pixels) (really too small to be realistic, but is a floor)
# MAX_FWHM = 40.0 #booming LyA are around 15-16; real lines can be larger, but tend to be not what we are looking for
#                 # and these are more likly continuum between two abosrpotion features that is mistaken for a line
#                 #AGN seen with almost 25AA with CIV and NV around 35AA
# MAX_NORMAL_FWHM = 20.0 #above this, need some extra info to accept
# MIN_HUGE_FWHM_SNR = 25.0 #if the FWHM is above the MAX_NORMAL_FWHM, then then SNR needs to be above this value
# MIN_ELI_SNR = 3.0 #bare minium SNR to even remotely consider a signal as real
# MIN_ELI_SIGMA = 1.0 #bare minium (expect this to be more like 2+)
# MIN_HEIGHT = 10
# MIN_DELTA_HEIGHT = 2 #to be a peak, must be at least this high above next adjacent point to the left
# DEFAULT_BACKGROUND = 6.0
# DEFAULT_BACKGROUND_WIDTH = 100.0 #pixels
# DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND = 10.0 #pixels
#
# GAUSS_FIT_MAX_SIGMA = 17.0 #maximum width (pixels) for fit gaussian to signal (greater than this, is really not a fit)
# GAUSS_FIT_MIN_SIGMA = 1.0 #roughly 1/2 pixel where pixel = 1.9AA (#note: "GOOD_MIN_SIGMA" below provides post
#                           # check and is more strict) ... allowed to fit a more narrow sigma, but will be rejected later
#                           # as not a good signal .... should these actually be the same??
# GAUSS_FIT_AA_RANGE = 40.0 #AA to either side of the line center to include in the gaussian fit attempt
#                           #a bit of an art here; too wide and the general noise and continuum kills the fit (too wide)
#                           #too narrow and fit hard to find if the line center is off by more than about 2 AA
#                           #40 seems (based on testing) to be about right (50 leaves too many clear, but weak signals behind)
# GAUSS_FIT_PIX_ERROR = 4.0 #error (freedom) in pixels (usually  wavebins): have to allow at least 2 pixels of error
#                           # (if aa/pix is small, _AA_ERROR takes over)
# GAUSS_FIT_AA_ERROR = 1.0 #error (freedom) in line center in AA, still considered to be okay
# GAUSS_SNR_SIGMA = 5.0 #check at least these pixels (pix*sigma) to either side of the fit line for SNR
#                       # (larger of this or GAUSS_SNR_NUM_AA) *note: also capped to a max of 40AA or so (the size of the
#                       # 'cutout' of the signal (i.e. GAUSS_FIT_AA_RANGE)
# GAUSS_SNR_NUM_AA = 5.0 #check at least this num of AA to either side (2x +1 total) of the fit line for SNR in gaussian fit
#                        # (larger of this or GAUSS_SNR_SIGMA

log = G.Global_Logger('spectrum_utils')
log.setlevel(G.LOG_LEVEL)


filter_iso_dict = {'u': 3650.0,
                   'b': 4450.0,
                   'g': 4640.0,
                   'v': 5510.0,
                   'r': 6580.0,
                   'i': 8060.0,
                   'z': 9000.0,
                   'y': 10200.0,
                   'f606w': 6000.0,
                   'acs_f606w_flux':6000.0, #name sometimes set this way from catalog
                   'f435w': 4310.0,
                   'acs_f606w_flux':4310.0,
                   'f775w': 7693.,
                   'f814w': 8045.,
                   'f105w': 10550.,
                   'f125w': 12486.,
                   'f140w': 13923.,
                   'f160w': 15370.
                   }


#MULTILINE_CONFIDENCE = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.25, 0.45, 0.70, 0.85, 0.90, 0.98]
MULTILINE_CONFIDENCE = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.15, 0.45, 0.60, 0.70, 0.85, 0.90, 0.98]
MULTILINE_SCORE      = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 1.00]
INTERP_MULTILINE_SCORE = np.linspace(0.0,1.0,100)
INTERP_MULTILINE_CONFIDENCE = np.interp(INTERP_MULTILINE_SCORE,MULTILINE_SCORE,MULTILINE_CONFIDENCE)

#from hetdex_api.survey import Survey
from astropy import time, coordinates as Coord
from astropy import constants as Const

McDonald_Coord = Coord.EarthLocation.of_site('mcdonald')

#simple dict format to use for zPDFs
zPDF_dict = {"PDF": [], "z": [], "path": None, "desc": None}

def sum_zPDF(target_z,pdf,zarray,delta_z=0.25):
    """
    Sum up the zPDF probabilties over a range of redshifts
    Does not assume the zPDF is normalized to 1.0 and forces a normalization in the sum

    :param target_z: the target redshift to check
    :param pdf: the zPDF probabilities P(z)
    :param zarray: array of redshifts that match up with the P(z) (e.g. the z in the P(z)
    :param delta_z: the redshift range over which to sum (e.g. target_z +/- delta_z)
    :return: the summed P(z)
    """
    try:
        left,_,_ = getnearpos(np.array(zarray).astype(float),max(0,target_z-delta_z))
        right,_,_ = getnearpos(np.array(zarray).astype(float),target_z+delta_z)
        p_z = np.sum(pdf[left:right+1]) / np.sum(pdf)
    except:
        p_z = 0
        log.info("Exception! Exception checking P(z) in spectrum_utilities sum_zPDF.",exc_info=True)

    return p_z

def is_edge_fiber(absolute_fiber_num, ifux=None, ifuy=None):
    """
    fiber_num is the ABSOLUTE fiber number 1-448
    NOT the per amp number (1-112)

    or use fiber center IFUx and IFUy as the fiber may be in a non-standard location
    but the IFUx and IFUy should be correct

    # -22.88 < x < 22.88 ,  -24.24 < y < 24.24

    :param fiber_num:
    :return:
    """

    if ifux is None or ifuy is None:
        return absolute_fiber_num in G.CCD_EDGE_FIBERS_ALL
    else:
        # back off just a bit for some slop
        if (-22.5 < ifux < 22.5) and (-24.0 < ifuy < 24.0):
            return False
        else:
            return True


def get_fluxlimits(ra,dec,wave,datevobs,sncut=4.8,flim_model=None,ffsky=False,rad=3.5):
    """
    wrapper to call into HETDEX API

    This would be the 50% flux limit (i.e. we detect 50% of emission lines as this location (ra,dec,wave,shot) at that
    flux level (with an assummed linewidth)

    :param datevobs:  string
    :param flim_model: string (current is "v4"?) None gives most current
    :param snrcut:
    :param ra:
    :param dec:
    :param wave:
    :return: array of flux limits (integrated line fluxes, by default over 7 wavebins) and apcor
    """

    try:
        log.info(f"Retreiving flux limits and apcor using flim_mode {flim_model} ...")

        try: #if wave is an array of wavelenghts, then ra, dec need to be arrays of equal length
            if wave is None:
                wave = G.CALFIB_WAVEGRID

            if np.shape(ra) != np.shape(wave):
                if np.shape(ra) == ():
                    ra = np.full(len(wave),ra)
                    dec = np.full(len(wave), dec)
                else: #they have shapes but don't match
                    log.error("spectrum_utilitiess::get_fluxlimits() bad input. RA, Dec shape does not match wave shape.")
                    return None, None
        except:
            log.error("spectrum_utilitiess::get_fluxlimits() bad input. RA, Dec shape does not match wave shape.")
            return None, None

        try:
            _ = datevobs / 1 #simple
            #probably a shotid
            datevobs = str(datevobs)[:-3] + 'v' + str(datevobs)[-3:]
        except:
            #it is already a string
            pass
        #(self, datevshot, release=None, flim_model=None, rad=3.5,
        #ffsky=False, wavenpix=3, d25scale=3.0, verbose=False,
        #sclean_bad = True, log_level="WARNING")
        shot_sens = ShotSensitivity(datevobs, release=f"hdr{G.HDR_Version}", flim_model=flim_model, rad=rad, ffsky=ffsky,
                                     verbose=False, log_level="CRITICAL") #wavenpix=3, d25scale=3.0,sclean_bad=True,

        # sncut : float
        #     cut in detection significance
        #     that defines this catalogue
        # direct_sigmas : bool
        #     return the noise values directly
        #     without passing them through
        #     the noise to 50% completeness
        #     flux
        # linewidth : array
        #     optionally pass the linewidth of
        #     the source (in AA) to activate the linewidth
        #     dependent part of the completeness
        #     model (default = None).
        #f50, apcor = shot_sens.get_f50(ra, dec, wave, sncut, direct_sigmas =True, linewidth = 2.0)
        f50, apcor = shot_sens.get_f50(ra, dec, wave, sncut)

        return f50, apcor
    except Exception as e:
        log.error(f"Exception attempting to get flux limits and apcor.",exc_info=True)
        print(e)
        return None, None


def calc_dex_g_limit(calfib,calfibe=None,fwhm=1.7,flux_limit=4.0,wavelength=4640.,aper=3.5,ifu_fibid = None,
                     central_fiber=None, detectid=None):
    """
    calcuate an approximage gband mag limit for THIS set of calfibs (e.g. typically one IFU for one shot)

    losely based on what was done for LyCon and selecting "empty" fibers for a background

    :param calfib: (or ffsky) assumed in flux units over 2AA x10^-17 erg/s/cm2
    :param calfibe:
    :param fwhm: seeing fwhm
    :param flux_limit: default assumed limit
    :param wavelength: f_iso wavelength
    :param aper: extraction aperture in arcsec
    :param ifu_fibid: fiber ID 1 - 448 (really only used in debugging)
    :param central_fiber: the "blue" central fiber
    :param detectid:
    :return:
    """

    debug_printing = False #turn on extra debug print

    try:
        limit = G.HETDEX_CONTINUUM_MAG_LIMIT
        min_num_final_fibers = max(100, int(len(calfib)/4) )  # 1/4 of the standard, but at least 100
        #so if 1344 passed in, would need at leat 336 ... if 336 passed in (112x3) would need 100
        # min_std_of_fiber_means = 0.003 #e-17 ... leads to mag limits 26 and fainter; see these with large objects in the IFU
        # can still push down the error ... I think it squelces variation in the IFU
        # maybe an issue with the calibration?
        min_mean_calfibe = 0.055  # e-17 like above, large object squelces the error? ... this is a bit of a guess not sure it makes sense

        edge = False
        try:
            if central_fiber is not None and is_edge_fiber(central_fiber.number_in_ccd):
                edge = True
        except:
            log.debug("Exception checking for edge fiber in calc_dex_g_limit",exc_info=True)

        # first trim off the ends that are not as well calibrated and/or subject to extemes
        #all_calfib = calfib[:, 100:-100] #was 100,-100, 215:-70 is 3900-5400
        #all_calfibe = calfibe[:,100:-100]
        all_calfib = calfib[:, 215:-70] #was 100,-100, 215:-70 is 3900-5400 (closer to SDSS-g, but maybe does not matter much)
        all_calfibe = calfibe[:,215:-70]
        if ifu_fibid is None:
            ifu_fibid = np.full(np.shape(all_calfib)[0],-1)

        base_edge = np.count_nonzero([is_edge_fiber(x) for x in ifu_fibid]) / len(ifu_fibid)

        #could be something very wrong with the error
        #get rid of any with obvious emission lines
        #make each element the mean of itself and its two neighbors and compare to the flux limit
        #this is roughly equivalent to the LyCon paper looking for any 3 consecutive wavebins with 4.0, 5.0, 4.0 flux or greater
        mf = (calfib[:, :-2] + calfib[:, 1:-1] + calfib[:, 2:]) / 3.0
        #mf = mf[:,99:-99] #to match 100:-100 having shrunk by one (though it does not really matter)
        mf = mf[:, 214:-69]  # to match 215:-70 having shrunk by one (though it does not really matter)
        sel = np.max(mf, axis=1) < flux_limit
        all_calfib = all_calfib[sel]
        all_calfibe = all_calfibe[sel]
        ifu_fibid = ifu_fibid[sel]


        #get rid of any with calfibe issues
        #lots of zeros or -1 values (make the califb value == nan??
        bad_e = np.isnan(all_calfibe) | np.array(all_calfibe <=0)
        all_calfib[bad_e] = np.nan
        all_calfibe[bad_e] = np.nan #will want the nan later; do not want any 0 values for this purpose
        all_calfib[all_calfib==0] = np.nan #some could legit be exactly zero, but that would be very rare

        #remove any fibers with more than 20 nans ?
        sel = np.array([np.count_nonzero(np.isnan(x)) for x in all_calfib[:]]) < 20
        all_calfib = all_calfib[sel]
        all_calfibe = all_calfibe[sel]
        ifu_fibid = ifu_fibid[sel]

        #get rid of continuum (and negative continuum)
        #for better accuracy this really should be done in 500AA chunks or so, rather that over (almost) the whole
        #spectral width at once
        cont_calfib = np.nanmedian(all_calfib, axis=1) /2.0 # mean flux denisties
        sel = np.array(cont_calfib < 0.02) & np.array(cont_calfib > -0.02)
        # so average above 2e-19 erg/s/cm2/AA or aboout g 23.8 at 5 sigma under a 3.5" aperture and typical seeing
        # assuming this as a scatter (so 1.5 (seeing correction) * 5.0 (sigma) * 0.02e-17 at 4640AA => 23.8)
        # (should always be better than this)
        # (note: as measured flux in one single fiber, this is only about g=26)
        # Typically these keeps about 75% of all fibers
        all_calfib = all_calfib[sel]
        all_calfibe = all_calfibe[sel]
        cont_calfib = cont_calfib[sel]
        ifu_fibid = ifu_fibid[sel]

        calfibe_means = np.nanmean(all_calfibe, axis=1)
        califbe_mu = np.nanmean(calfibe_means)
        calfibe_std = np.nanstd(calfibe_means)

        sclip = 1.0
        sel = np.array( (calfibe_means - califbe_mu) < sclip * calfibe_std)  #one side only (remove largest errors)
        all_calfib = all_calfib[sel]
        all_calfibe = all_calfibe[sel]
        ifu_fibid = ifu_fibid[sel]

        # check here ... which fibers are trimmed off
        if len(ifu_fibid) == 0: #there is definitely a problem
            log.info(f"({detectid}) HETDEX g-limit: Too few fibers ({len(all_calfib)}) to reliably compute mag limit. "
                     f"Using default {G.HETDEX_CONTINUUM_MAG_LIMIT}.")
            return G.HETDEX_CONTINUUM_MAG_LIMIT
        else:
            all_fibers = np.count_nonzero([is_edge_fiber(x) for x in ifu_fibid]) / len(ifu_fibid)
            remaining_fibers = np.count_nonzero([is_edge_fiber(x) for x in ifu_fibid]) / len(ifu_fibid)

        #these are effectively empty fibers now and a measure of the noise
        #mean = np.nanmean(all_calfib)/2.0 #full mean over all remaining fibers and wavebins as  flux denisty
        #std = np.std(all_calfib)/2.0
        #std_of_mean = np.std(np.mean(all_calfib/2.0,axis=0)) #std of the means treating each fiber individually
        #mean_of_means = np.mean(np.mean(all_calfib,axis=0)) #should be the same as the full mean

        #want the mean of the means of the fibers (each fiber gets its own mean and then we want the mean and the std of those)
        fiber_means = np.nanmean(all_calfib,axis=1) / 2.0 #!! don't forget the 2.0 !! these are fluxes in 2AA bins, need flux densities
        #fiber_mean_errors = np.nanmean(np.sqrt(all_calfibe * all_calfibe),axis=1)/2.0
        mean_of_fiber_means = np.nanmean(fiber_means)
        std_of_fiber_means = np.nanstd(fiber_means)
        #mean_of_fiber_errors = np.nanmean(fiber_mean_errors)
        #std_of_fiber_errors = np.nanstd(fiber_mean_errors)

        if abs(mean_of_fiber_means > 0.1): # 0.05 ~25.01 g, 0.075 ~ 24.57, 0.08 ~ 24.50g, 0.10 ~24.26
            # #would be same as straight mean of all wavelength bin fluxes
            #something is wrong
            log.info(f"({detectid}) HETDEX g-limit: bad calculation. mean of fiber means {mean_of_fiber_means}. "
                     f"Setting to default {G.HETDEX_CONTINUUM_MAG_LIMIT}.")
            return G.HETDEX_CONTINUUM_MAG_LIMIT


        if len(all_calfib) < min_num_final_fibers:
            log.info(f"({detectid}) HETDEX g-limit: Too few fibers ({len(all_calfib)}) to reliably compute mag limit. "
                     f"Using default {G.HETDEX_CONTINUUM_MAG_LIMIT}.")
            return G.HETDEX_CONTINUUM_MAG_LIMIT

        #what about other checks ... something weird with the calfibe? negative mean_of_fiber_means?
        #negative is okay ... give the cuts above it could be very close to zero, just barely negative
        # if np.nanmedian(fiber_means) < 0:
        #     log.info(f"HETDEX g-limit: Negative fiber median. Using default {G.HETDEX_CONTINUUM_MAG_LIMIT}.")
        #     return G.HETDEX_CONTINUUM_MAG_LIMIT

        if abs((np.nanmean(all_calfibe)/2.0) / mean_of_fiber_means) < 10: #calfibe cannot be negative
            log.info(f"HETDEX g-limit: Fiber errors abnormally small. Using default {G.HETDEX_CONTINUUM_MAG_LIMIT}.")
            return G.HETDEX_CONTINUUM_MAG_LIMIT

        #this is what the limit is built on ... sometimes you get a bright object (like a large-ish galaxy)
        #in the IFU and it drives down the error
        # if (np.nanmean(all_calfibe)/2) < min_mean_calfibe:
        #     log.info(f"HETDEX g-limit: Fiber flux errors abnormally small. Using default {G.HETDEX_CONTINUUM_MAG_LIMIT}.")
        #     return G.HETDEX_CONTINUUM_MAG_LIMIT

        #since this a background of "empty" fibers the PSF does not matter
        #as we assume this to be uniform, so any PSF would give the same flux.
        #BUT, the aperture does matter, so how much flux would we expect in a 3.5" diam aperture?
        #well, this is for one 1.5" diam fiber, so a 3.5 diam aperture vs 1.5" diam aperture so (3.5 / 1.5)**2
        if aper is None or aper <= 0:
            aper = 3.5

        # approx area using just the heights (since the base is on a uniform grid, that will divide out)
        #this is really an area correction, but since the 1D spectrum for HETDEX is PSF weighted,
        #this is based on the (approximate) fraction of the whole of that weight provided by the central most fiber
        #IF we are centered on it. Better seeing (smaller fwhm) is a smaller correction since more flux is from that
        #center fiber (narrower PSF).

        #BUT can we argue that for point sources, we capture (approximately) all the flux in aperture regardless
        #of the PSF (to a point, but with a 3.5" radius aperuture and PSF below 3", typically ~1.7")
        step = 0.01
        sigma = fwhm/2.355
        inner = np.sum(gaussian(np.arange(0,0.75+step,step),0, sigma, a=1.0, y=0.0))
        outer_bins = np.arange(0, aper + step, step)
        whole_w = gaussian(outer_bins, 0, sigma, a=1.0, y=0.0)
        whole = np.sum(whole_w)

        #if True: #use the FWHM to set effective radius
        effective_radius = min(5*sigma, aper)  #out to 5 sigma; caps out at 3.5" by FWHM ~ 1.7
        # else: #use where the weight gets close to zero
        #     rad_idx = np.where(whole_w < 1e-6)[0] # caps at 3.5" by FWHM ~ 1.6
        #     if len(rad_idx) == 0:
        #         effective_radius = aper
        #     else:
        #         effective_radius = outer_bins[rad_idx[0]]

        gaps_correction = 0.9487  # assume xx% coverage (roughly  1 - (root(3)-pi/2))/pi) #the area outside of fiber radius
                                  # not covered by fiber
        radius_rat = effective_radius / 0.75 #single fiber radius
        area_rat = whole / inner

        psf_corr = area_rat * radius_rat * gaps_correction

        #this is sort of a best case ... if the object is faint and not near the center of a fiber, it can be more
        #difficult to detect
        limit = cgs2mag(psf_corr * std_of_fiber_means * 1e-17, wavelength)

        if limit is None or np.isnan(limit):
            limit = G.HETDEX_CONTINUUM_MAG_LIMIT
        else: #round up to 0.1f
            limit = np.ceil(limit*10.)/10.


        #sanity check ... what are wholly unreasonable values??
        #normally  25.5+ would seem improbable, but what about great seeing and throughput with a long exposeure??

        if debug_printing:
            print(f"base_edge: {base_edge:0.4f} pre-cut: {all_fibers:0.4f} final_edge: {remaining_fibers:0.4f} "
              f"limit: {limit:0.4f}  mean_fluxd: {mean_of_fiber_means:0.4f}  std_fluxd {std_of_fiber_means:0.4f}  "
              f"mean_fluxd_err: {np.nanmean(all_calfibe)/2.0:0.4f}  seeing: {fwhm:0.2f}  psf_cor:  {psf_corr:0.2f}  "
              f"num_fibers: {len(fiber_means)}  edge: {edge}  detectid: {detectid}")

        log.info(f"({detectid}) HETDEX g-limit: limit {limit:0.4f},  mean_fluxd: {mean_of_fiber_means:0.4f}  "
              f"std_fluxd {std_of_fiber_means:0.4f}  "
              f"mean_fluxd_err: {np.nanmean(all_calfibe)/2.0:0.4f}  seeing: {fwhm:0.2f}  psf_cor:  {psf_corr:0.2f}  "
              f"num_fibers: {len(fiber_means)}  edge: {edge} ")

    except:
        log.warning("Exception in calc_dex_g_limit",exc_info=True)

    return limit

#cloned with minor changes from Erin Cooper's HETDEX_API
def get_bary_corr(shotid, units='km/s'):
    """
    For a given HETDEX observation, return the barycentric radial velocity
    correction.

    Parameters
    ----------
    shotid
        interger observation ID
    units
        string indicating units to be used. Must be readable to astropy units.
        Defaults to km/s

    Return
    ------
    vcor
        radial velocity correction in units
    """

    #global the_Survey
    try:
        if G.the_Survey is None:
            G.the_Survey = hda_survey.Survey(G.HETDEX_API_CONFIG.LATEST_HDR_NAME) #make G.survey

        sel_shot = np.array(G.the_Survey.shotid == shotid)
        if np.count_nonzero(sel_shot) > 0:
            coords = G.the_Survey.coords[sel_shot]
            mjds = G.the_Survey.mjd[sel_shot]
            t = time.Time(np.average(mjds), format='mjd')

            vcor = coords.radial_velocity_correction(kind='barycentric', obstime=t, location=McDonald_Coord).to(units)

            return vcor.value
        else:
            return None
    except:
        log.info("Exception! Exception in spectrum_utilities::get_bary_corr()")

# def vac_to_air(w_vac):
#     #http: // www.sdss3.org / dr9 / spectro / spectro_basics.php
#     return w_vac / (1 + 2.73518e-4 + 131.418 / w_vac ** 2 + 2.76249e8 / w_vac ** 4)

#directly lifted from specutils with some modifications
def air_to_vac(wavelength):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006

    !!! warning ... does not check for 2000AA normal cutoff ....

    """

    try:

        if not hasattr(wavelength,"value"):
            wavelength = wavelength * U.angstrom
            strip_unit = True
        else:
            strip_unit = False

        wlum = wavelength.to(U.um).value
        val = (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wavelength

        if strip_unit:
            val = val.value
        return val
    except:
        log.warning("Exception in spectrum_utilities air_to_vac().",exc_info=True)
        return None

def vac_to_air(wavelength):
    """
    Griesen 2006 reports that the error in naively inverting Eqn 65 is less
    than 10^-9 and therefore acceptable.  This is therefore eqn 67

     !!! warning ... does not check for 2000AA normal cutoff ....
    """
    try:
        if not hasattr(wavelength,"value"):
            wavelength = wavelength * U.angstrom
            strip_unit = True
        else:
            strip_unit = False

        wlum = wavelength.to(U.um).value
        nl = (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4))
        val = wavelength/nl
        if strip_unit:
            val = val.value

        return val
    except:
        log.warning("Exception in spectrum_utilities vac_to_air().", exc_info=True)
        return None


def z_correction(z,w_obs,vcor=None,shotid=None):#,*args):
    """
    Correct the computed redshift, which uses air wavelengths, etc to vacuum and make other corrections for:
        #todo: list applied corrections like There are 3-4 other corrections to the wavelength solution that we are not considering.
        These are, in order of importance:
           * heliocentric correction. This is the most important of all. Early on I was doing this, but then stopped for some reason. In any table, we need to give the information to calculate this. Thus we need UTC, RA, DEC. As long as we have UTC in the table, then we are good to state there is no helio correction
           * transform to vacuum (as discussed above). I’m in the camp that we report air wavelengths, and then give the correction to vacuum.
           * calibrate twilight using solar spectrum. This can have a small effect that I haven’t measured
           * use sun-disk integrated spectrum compared to kpno. smaller effect
           * correction due to GR redshift of earth-sun: 0.6 km/s



    This needs to adjust the z which used AIR observed and AIR rest wavelengths. So we need to convert the observered wavelength
    to vacuum AND the rest wavelegth to vacuum

    # so here is the correction for the observed_corrected_wavelength to use for the redshift estimate:
    wave_air+1.+(wave_air-3500)/4000.+vcor/(3e5/wave_air)
    where wave_air is what we measure (3500-5500) and vcor is the value that Erin just calculated. Double check I did this correctly.

    :param z:
    :param w_obs:
    :param vcor: velocity correction for earth orbit
    :param args:
    :return:
    """
    try:

        if vcor is None:
            if shotid is not None:
                vcor = get_bary_corr(shotid)
                # can come back as an array
                try:
                    vcor = vcor[0]
                except:
                    pass
            else:
                vcor = 0

        w_vac = air_to_vac(w_obs)
        w_vel = vcor/(3e5/w_obs) #observed wavelength corrected from air to vacuum and corrected for Earth's velocity
        w_rest = w_obs / (z + 1.0) #the line's rest wavelength as used is encoded in the w_obs (uncorrected) and the uncorrected redshift
        if w_rest > G.AirVacuumThresh: #else, already is in vacuum
            w_rest = air_to_vac(w_rest) #rest-frame wavelength corrected from air to vacuum (don't want vcor here)


        #log updates
        log.debug(f"Redshift corrections: w_obs {w_obs:0.2f} to {w_vac + w_vel:0.2f}: vac corr {w_vac - w_obs:0.2f} + "
                  f"Earth velocity corr ({vcor:0.2f}km/s,{w_vel:0.2f}AA); rest {w_rest:0.2f}")

        #combine updates to new observed wavelength (now in vacuum and with velocity correction)
        w_vac = w_vac + w_vel

        z_cor = w_vac / w_rest - 1.0 #now both are in vacuum and the observed also corrected for Earth's velocity

        log.debug(f"Redshift corrections: old z {z:0.4f} to new z {z_cor:0.4f}")

        return z_cor
    except:
        log.warning("Exception in spectrum_utilities z_correction().",exc_info=True)
        return None

def map_multiline_score_to_confidence(score):
    """
    Right now, this is opinion driven
    interpolate between bins
    todo: establish via comparing multiline scale score vs "confirmed" redshift

    :param score: the multiline scaled score (0-1)
    :return:
    """
    try:
        i,*_ = getnearpos(INTERP_MULTILINE_SCORE,score)
        return INTERP_MULTILINE_CONFIDENCE[i]
    except:
        return 0


def filter_iso(filtername, lam):
    """
    Approximate iso wavelength lookup by filter. If not found, just returns the wavelength passed in.
    :param filtername:
    :param lam:
    :return:
    """
    global filter_iso_dict
    try:
        if (filtername is not None) and filtername.lower() in filter_iso_dict.keys():
            return filter_iso_dict[filtername.lower()]
        else:
            log.info(f"Unable to match filter {filtername} to iso wavelength")
            return lam
    except:
        log.debug("Exception is spectrum_utilities::filter_iso",exc_info=True)
        return lam


def mag2cgs(mag,lam):
    """
    :param mag:   AB mag
    :param lam: central wavelength or (better) f_lam_iso
    :return: flam (erg/s/cm2/AA)
    """

    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        return 3631. * 1e-23 * 10**(-0.4 * mag) * c / (lam * lam)
    except:
        return 0

def mag2flam(mag,lam):
    return mag2cgs(mag,flam)

def mag2fnu(mag):
    """
    :param mag:   AB mag
    :return: fnu (erg/s/cm2/Hz)
    """

    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        return 3631. * 1e-23 * 10**(-0.4 * mag)
    except:
        return 0


def fnu2mag(fnu): #erg/s/cm2/Hz to mag
    try:#fnu to Jansky then / reference
        return -2.5 * np.log10(fnu * 1e23 / 3631.)
    except:
        log.info("Exception! in fnu2mag.",exc_info=True)
        return None

def flam2fnu(flam,waves):
    """

    :param flam: erg/s/sm2/AA
    :param waves: AA
    :return:
    """
    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        return flam * waves * waves / c
    except:
        return None


def fnu2flam(fnu, waves):
    """

    :param fnu: erg/s/cm2/Hz
    :param waves: AA
    :return:
    """
    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        return c * fnu / (waves * waves )
    except:
        return None

def ujy2cgs(ujy,lam): #micro-jansky to erg/s/cm2/AA
    conv = None
    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        conv =  ujy * 1e-29 * c / (lam*lam)
    except:
        log.info("Exception! in ujy2cgs.",exc_info=True)
    return conv

def cgs2ujy(cgs,lam): #erg/s/cm2/AA to micro-jansky
    conv = None
    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        conv = cgs * 1e29 / c * lam*lam
    except:
        log.info("Exception! in cgs2ujy.", exc_info=True)
    return conv


def ujy2mag(ujy): #micro-jansky to erg/s/cm2/AA
    conv = None
    try:
        conv = -2.5 * np.log10(ujy / 1e6 / 3631.)
    except:
        log.info("Exception! in ujy2mag.",exc_info=True)
    return conv


def cgs2mag(cgs,lam):
    """
    :param cgs:   erg/s/cm2/AA
    :return:
    """
    try:
       return ujy2mag(cgs2ujy(cgs,lam))
    except:
        return 0




def continuum_band_adjustment(obs_wave,band):
    """
    Adjustment, assuming UV Beta slope for star forming galaxies, for the given band pass to the obs_wave (assuming
    it is LyA)
    Using f_lam propto lambda^beta
    :param obs_wave:
    :param band:
    :return:
    """
    try:
        #if want a fixed +0.3 mag like in Leung+2017, then return x 1.318 ~= 10**(0.12) , which is 10**(-.4*0.3)
        #return 1.318

        #this is for restframe ... so need to compress assuming z_LyA
        #adjust as beta exponent over the relative ratio of the observed wavlength and the iso-wavelength of the filter
        return (obs_wave/filter_iso(band,obs_wave))**(G.R_BAND_UV_BETA_SLOPE+2)
    except:
        return 1.0

def ew_obs(lineflux,lineflux_err, obs_wave, band, filter_flux, filter_flux_err):
    """
    Compute EW observed from a photometric bandpass (g or r only supported)
    Handles a small correction to r-band when using it for continuum per Leung+2017 (0.3 mag brighter as 'g' from 'r')

    :param lineflux: erg/s/cm2
    :param mag:  AB mag
    :param band: 'g','r','f606w'
    :param obs_wave: in AA
    :return: rest EW
    """

    ew_obs = None
    ew_obs_err = 0.0
    #beta = G.R_BAND_UV_BETA_SLOPE
    flux_adjust = 1.0

    if filter_flux_err is None:
        filter_flux_err = 0

    if lineflux_err is None:
        lineflux_err = 0

    if band is None:
        band = 'x'

    #mag to continuum is more like f_nu than f_lambda)
    try:
        if band.lower() in ['r','f606w']:
            #mag -= 0.3 #0.3 approximate adjust from Leung 2017 but really should vary a bit by wavelength
            flux_adjust = continuum_band_adjustment(obs_wave, band)
            #probably too much of an adjustment: roughly 3.5x at 3500AA, 2x at 4500 and 1.4x at 5500 for beta = -2
            #probably too much of an adjustment: roughly 2.8x at 3500AA, 1.9x at 4500 and 1.3x at 5500 for beta = -1.7
        elif band.lower() in ['g']:
            pass
        else:
            log.warning (f"Invalid photometric bandpass {band} in spectrum_utilities::lya_ewr()")

        #continuum = mag2cgs(mag,filter_iso(band,obs_wave))*flux_adjust
        log.debug(f"Continuum estimate adjustment for {band}: x{flux_adjust:0.2f}")
        continuum = filter_flux*flux_adjust
        if continuum !=0:
            ew_obs = lineflux/continuum
        else:
            return 0, 0

        try:
            ew_obs_err = abs(ew_obs * np.sqrt(  (lineflux_err / lineflux) ** 2 + (filter_flux_err / continuum) ** 2))
        except:
            ew_obs_err = 0.0

    except:
        log.error(f"Exception in spectrum_utilities::lya_ewr",exc_info=True)

    return ew_obs, ew_obs_err

def lya_ewr(lineflux,lineflux_err, obs_wave, band, filter_flux, filter_flux_err):
    """
    Compute the LyA rest EW from a photometric bandpass (g or r only supported)
    Handles a small correction to r-band when using it for continuum per Leung+2017 (0.3 mag brighter as 'g' from 'r')

    :param lineflux: erg/s/cm2
    :param mag:  AB mag
    :param band: 'g','r','f606w'
    :param obs_wave: in AA
    :return: rest EW and error
    """
    try:
        if (lineflux is None) or (lineflux == 0) or (obs_wave is None) or (obs_wave == 0) or (filter_flux is None) or (filter_flux == 0):
            return np.nan, np.nan

        return np.array(ew_obs(lineflux,lineflux_err, obs_wave, band, filter_flux, filter_flux_err))/(obs_wave/G.LyA_rest)
    except:
        log.error(f"Exception in spectrum_utilities::lya_ewr",exc_info=True)
        return np.nan,np.nan

def getnearpos(array,value):
    """
    Nearest, but works best (with less than and greater than) if monotonically increasing. Otherwise,
    lt and gt are (almost) meaningless

    :param array:
    :param value:
    :return: nearest index, nearest index less than the value, nearest index greater than the value
            None if there is no less than or greater than
    """
    if type(array) == list:
        array = np.array(array)

    idx = (np.abs(array-value)).argmin()

    if array[idx] == value:
        lt = idx
        gt = idx
    elif array[idx] < value:
        lt = idx
        gt = idx + 1
    else:
        lt = idx - 1
        gt = idx

    if lt < 0:
        lt = None

    if gt > len(array) -1:
        gt = None


    return idx, lt, gt


def g2r(gmag):
    """
    Per Leung+2017 0.3 mag difference between g and r (r is brighter)
    :param gmag:
    :return:
    """

    try:
        return gmag
        #return gmag - 0.3
    except:
        log.error("Exception in g2r",exc_info=True)


def r2g(rmag):
    """
    Per Leung+2017 0.3 mag difference between g and r (r is brighter)
    :param gmag:
    :return:
    """

    try:
        return rmag
        #return rmag + 0.3
    except:
        log.error("Exception in r2g",exc_info=True)


def make_fnu_flat_spectrum(mag,filter='g',waves=None):
    """
    flat in fnu but returns flam

    :param mag:
    :param filter: should be 'g' or 'r'
    :param waves: if None assumes HETDEX (needs to be in AA if supplied with unit attached)
    :return:  f_lam (flux density in each wavelength bin)
    """

    try:
        if type(filter) == bytes:
            _filter = filter.decode()
        else:
            _filter = filter

        if _filter.lower() not in  ['g','r','f606w']:
            log.info(f"Invalid filter {filter} passed to make_fnu_flat_spectrum")
            return None

        if _filter.lower() != 'g': #then
            mag = r2g(mag)

        if waves is None or len(waves)==0:
            waves = G.CALFIB_WAVEGRID#np.arange(3470.,5542.,2.0) #*U.angstrom  #3470 - 5540

        #iso = filter_iso(filter,-1) #return get the f iso wavelegnth, otherwise assume
        # try:
        #     if waves.unit:
        #         pass
        # except:
        #     waves = waves * U.angstrom
        #iso = iso * U.angstrom

        fnu = mag2fnu(mag) #* U.erg / U.s / (U.cm * U.cm) * U.s #use * U.s instead of / U.Hz so the seconds cancel in the
        #final product (else says s^2 * Hz
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value

        flam = fnu*c/(waves*waves)

        return flam
    except:
        log.error("Exception! Exception in spectrum_utilites.make_fnu_flat_spectrum()",exc_info=True)

    return None

def make_flux_flat_spectrum(mag,filter='g',waves=None):
    """

    :param mag:
    :param filter: should be 'g' or 'r'
    :param waves: if None assumes HETDEX (needs to be in AA if supplied with unit attached)
    :return:  f_lam
    """
    try:
        if type(filter) == bytes:
            _filter = filter.decode()
        else:
            _filter = filter
        if _filter not in  ['g','r']:
            log.info(f"Invalid filter {filter} passed to make_flux_flat_spectrum")
            return None

        if waves is None or len(waves)==0:
            waves = G.CALFIB_WAVEGRID #np.arange(3470.,5542.,2.0)#*U.angstrom  #3470 - 5540

        iso = filter_iso(_filter,-1) #return get the f iso wavelegnth, otherwise assume

        # try:
        #     if waves.unit:
        #         pass
        # except:
        #     waves = waves * U.angstrom
        #iso = iso * U.angstrom

        flam_iso = mag2cgs(mag,iso) #* U.erg / U.s / (U.cm * U.cm) / U.AA #use * u.s instead of / u.Hz so the seconds cancel in the
        flux = np.full(len(waves),flam_iso*(waves[1]-waves[0])) #*u.erg / u.s / (u.cm * u.cm)

        return flux
    except:
        log.error("Exception! Exception in spectrum_utilites.make_flux_flat_spectrum()",exc_info=True)

    return None

def snr(flux,noise,flux_err=None,wave=None,center=None,delta=None):
    """
    Calculate the signal to noise as the sum of the flux over a region divided by the quadrature sum of the error over
    the same region.

    If wave, center, and sigma are provided, will compute over the region centered on "center" +/- "delta" where
    "delta" would be something like 2x or 3x sigma (for an emission line).

    :param flux: data flux (must be in FLUX units, not flux density) so can be summed
                the flux could also come from the model values so long as it has the same shape and aligns 1:1 with noise
    :param noise: noise or uncertainty on each flux measure (same units)
    :param flux_err: error on the flux value (specifically when flux is from a model). If flux is just the original data
                     then "noise" is the error on that flux
    :param wave:  1:1 wavelength bins corresponding to flux and noise
    :param center: center position in wave unit (AA)
    :param delta:  distance (in wave units, AA) from the center to move in either direction
    :return: snr, snr_error
    """

    try:
        if (flux is None) or (noise is None):
            log.warning("Invalid parameters passed to spectrum_utilities::snr(): 1")
            return None, None

        if hasattr(flux,'__len__'):#this is a list or array
            if (len(noise)!=len(flux)):
                log.warning("Invalid parameters passed to spectrum_utilities::snr(): 2")
                return None, None
            if (flux_err is not None) and (len(flux)!= len(flux_err)):
                log.warning("Invalid parameters passed to spectrum_utilities::snr(): 3")
                return None, None

            flux_is_number = False
        else:
            flux_is_number = True


        if (wave is not None) and (center is not None) and (delta is not None):
            #center_idx,_,_ = getnearpos(wave,center)
            left_idx,_,_ = getnearpos(wave,center-delta)
            right_idx,_,_ = getnearpos(wave,center+delta)

            if not ((right_idx - left_idx) > 0):
                log.warning("Invalid parameters passed to spectrum_utilities::snr(). Unable to create valid bounds.")
                return None, None
        else:
            left_idx = 0
            right_idx = len(flux)

        signal_error = 0
        if flux_is_number:
            signal = flux
            if flux_err is not None:
                signal_error = flux_err
        else:
            signal = np.sum(flux[left_idx:right_idx+1])
            if (flux_err is not None) and (len(flux_err)==len(flux)):
                signal_error = np.sqrt(np.sum(flux_err[left_idx:right_idx+1]*flux_err[left_idx:right_idx+1]))

        noise = np.sqrt(np.sum(noise[left_idx:right_idx+1]*noise[left_idx:right_idx+1]))

        if noise > 0:
            #notice: there is no error on the noise, so using error propogation on division is pointless
            #and you end up with exactly signal_error/noise
            # snr_error = signal/noise * np.sqrt((signal_error/signal)**2 + (0/noise)**2)

            return signal/noise, signal_error/noise #ONLY SUCCESS PATH
        else:
            log.warning("Invalid parameters passed to spectrum_utitlities::snr(). Invalid noise.")
            return None, None
    except:
        log.error("Exception in spectrum_utilites::snr()",exc_info=True)

    return None, None


def chi_sqr(obs, exp, error=None, c=None, dof=2):#, reduced=True):
    """

    :param obs: (data)
    :param exp: (model)
    :param error: (error on the data)
    :param c: can pass in a fixed c (in most cases, should just be 1.0)
    :param dof: aka number of parameters to fit (for our Gaussian, that's 3 (sigma, mu, y))
    :return: chi2 and c (best level)
    """

    obs = np.array(obs) #aka data
    exp = np.array(exp) #aka model

    x = len(obs)

    if error is not None:
        error = np.array(copy.copy(error)) #copy, since we are going to possible modify it

    if (error is not None) and (c is None):
        c = np.sum((obs*exp)/(error*error)) / np.sum((exp*exp)/(error*error))
    elif c is None:
        c = 1.0

    chisqr = 0
    if error is None:
        error=np.zeros(np.shape(obs))
        error += 1.0

    error[error==0] = 1.0

    #test
    #dof = None
    # if dof is not None:
    #     dof -= 1

    try:
        if dof is not None and (len(obs)-dof) > 0: #using dof to also imply this is a reduced chi2
            chisqr =  1./(len(obs)-dof) * np.sum(((obs - c * exp) / error) ** 2)
        else:
            chisqr = np.sum( ((obs - c*exp)/error)**2 )
    except:
        log.warning("Exception! Exception computing chi2.",exc_info=True)
    #chisqr = np.sum( ((obs - c*exp)**2)/(error**2))

    # for i in range(x):
    #         #chisqr = chisqr + ((obs[i]-c*exp[i])**2)/(error[i]**2)
    #         chisqr = chisqr + ((obs[i] - c * exp[i]) ** 2) / (exp[i])

    return chisqr,c



def check_oiii(z,flux,flux_err,wave,delta=0,cont=0,cont_err=0):
    """
    Explicitly check if there flux at 5007 rest is 3x the flux at 4959 rest
    allowing for some error

    NOTICE!!! These are recorded fluxes. The continuum HAS NOT BEEN subtracted, so technically this is wrong, but,
    these are faint cases where we have no continuum detection so it is essentially measured at zero (not really at
    zero, but it is 10x (or more) smaller than the peak) and makes almost no difference for these cases.

    There is a minimum adjustment to subtract off continuum if it is passed in (same scale as flux, but as flux density)

    :param z:
    :param flux: flux units of some kind (not flux density)
    :param flux_err: ditto
    :param wave:  AA
    :param delta:  integer for +/- wavelength bins from the centerline to add up flux
    :param cont:  flux density but same scale as flux (i.e e-17)
    :param cont_err:
    :return: -1 error, 0 no, 1 yes
    """

    try:
        if not(-0.01 < z < 0.106): #out of range
            return 0

        if (delta is None) or (delta < 0):
            delta = 1
        else:
            delta = int(delta) #has to be an integer

        if (cont is not None) and (cont > 0): #assume in similar units but as flux density so will x2 for HETDEX bin width
            cont = 2.* cont
            if cont_err is not None:
                cont_err = 2.*cont_err
            else:
                cont_err = 0
        else:
            cont = 0 #to subtract
            cont_err = 0

        i4959,*_ = getnearpos(wave,(1+z)*G.OIII_4959)
        f4959 = np.sum(flux[i4959-delta:i4959+delta+1]) - cont*(1+2*delta)
        e4959 = np.sqrt(np.sum(flux_err[i4959-delta:i4959+delta+1]**2) + (cont_err*(1+2*delta))**2 )

        i5007,*_ = getnearpos(wave,(1+z)*G.OIII_5007)
        f5007 = np.sum(flux[i5007-delta:i5007+delta+1]) - cont*(1+2*delta)
        e5007 = np.sqrt(np.sum(flux_err[i5007-delta:i5007+delta+1]**2  + (cont_err*(1+2*delta))**2))

        ratio = f5007/f4959
        err = ratio * np.sqrt((e4959/f4959)**2 + (e5007/f5007)**2)

        imax4959 = i4959 - delta + np.argmax(flux[i4959-delta:i4959+delta+1])#getting an index of 0,1 or 2 need to add to the base index
        fmax4959 = flux[imax4959] - cont
        emax4959 = flux_err[imax4959] + cont_err

        imax5007 = i5007 - delta + np.argmax(flux[i5007-delta:i5007+delta+1]) #getting an index of 0,1 or 2 need to add to the base index
        fmax5007 = flux[imax5007] - cont
        emax5007 = flux_err[imax5007] + cont_err

        max_ratio = fmax5007/fmax4959
        err_max_ratio = max_ratio * np.sqrt((emax4959/fmax4959)**2 + (emax5007/fmax5007)**2)

        log.info(f"OIII flux 5007/4959 ratio check. sum = {ratio:0.2f} +/- {err:0.3f}, max = {max_ratio:0.2f} +/- {err_max_ratio:0.2f} ")

        if ((3.0 - err) < ratio < (3.0 + err) or (3.0 - err_max_ratio) < max_ratio < (3.0 + err_max_ratio)):
            return 1
        else:
            return 0
    except:
        return -1

def build_cosmology(H0 = SU_H0, Omega_m0 = SU_Omega_m0, T_CMB = SU_T_CMB):
    cosmo = Cosmo.FlatLambdaCDM(H0=H0,Om0=Omega_m0,Tcmb0=T_CMB)
    return cosmo


def luminosity_distance(z,cosmology=None):
    """
    :param cosmology:
    :param z:
    :return:  lumiosity distance as astropy units
    """

    if cosmology is None:
        cosmology = build_cosmology() #use the defaults

    return cosmology.luminosity_distance(z)

#def r2Muv(r,z,correction=1,cosmology=None)
def absolute_mag(mag,z,correction=1,cosmology=None):
    """
    Just the distance modulus with K-correction

    by definition:   m = M + DM + K     ==>     M = m - K - DM
        where
        m is the apparent band-pass magnitude
        M is the Absolute Magnitude (could be in a different bandpass)
        DM is the distance modulus = 5 * log10(DL/10pc)  where DL = luminosity distance
        K is the K correction = -2.5*log10((1+z)*L_ve/L_v)
            where
                L_ve is the Luminosity emitted (so at frequency (v) * (1+z))
                L_v is the Luminosity obsererd (at the observed frequency)
                this handles the change in band-pass and would be a reference or figured from photometry


    :param mag: band-pass mag (NOT bolometric ... if bolometric then the correction should be zero)
    :param z:
    :param correction: (K-correction) (integral form: see https://ned.ipac.caltech.edu/level5/Sept02/Hogg/Hogg2.html)
    see https://ned.ipac.caltech.edu/level5/Hogg/Hogg7.html ... the ratio of the Luminosity emitted / observed
    note: for z in 3 - 3.5, the r-band probes the rest-frame UV, so there is no K correction, BUT here that means that
    the correction = 1   (the ratio of the obs_r * emitted_uv / (standard referecne obs_r * standard ref emitted_uv)
    is one

    Another way to look at

    :param cosmology:
    :return:
    """

    #K = 2.5 * np.log10((1+z)*Le/L) where Le is the Luminosity in the emitted frame (over the emitted bandpass) and L is
    #the Luminosity in the observed frame in the observed band pass each as functions of frequency (like in Jy)
    #it flips to 1/(1+z) if functions of wavelength ... again see https://ned.ipac.caltech.edu/level5/Hogg/Hogg7.html

    if correction > 0:
        k_corr = -2.5 * np.log10((1+z)*correction)
    else:
        k_corr = 0

    return mag - k_corr - 5. * np.log10(luminosity_distance(z,cosmology).to(U.parsec)/(10.0 * U.parsec)).value



# return mag - 5. * np.log10(luminosity_distance(z,cosmology).to(U.parsec).value) + 5.0  - correction


def physical_diameter(z,a,cosmology=None):
    """
    :param cosmology:
    :param z:
    :param a: size (diameter) in arcsec
    :return:  lumiosity distance as astropy units
    """

    if cosmology is None:
        cosmology = build_cosmology() #use the defaults

    apk = cosmology.arcsec_per_kpc_proper(z)

    if apk > 0:
        return  (a / apk).value #strip units?
    else:
        return None

def shift_to_restframe(z, flux, wave, ez=0.0, eflux=None, ewave=None, apply_air_to_vac=False):
    """

    !!! This is a STUPID shift ... just compresses the wavelengths and (1+z)**3 on the flux !!!
    !!! depending on your purpose, you should use shift_to_rest_luminosity() defined later !!!

    We are assuming no (or insignificant) error in wavelength and z ...

    All three numpy arrays must be of the same length and same indexing
    :param z:
    :param flux: numpy array of fluxes ... units don't matter (but assumed to be a flux NOT flux denisty)
    :param wave: numpy array of wavelenths ... units don't matter
    :param ez:
    :param eflux:  numpy array of flux errors ... again, units don't matter, but same as the flux
                   (note: the is NOT the wavelengths of the flux error)
    :param ewave: numpy array of wavelenth errors ... units don't matter
    :param apply_air_to_vac: if true, first apply observed (air) to vacuum correction to the (observed) spectrum
    :return:
    """

    #todo: how are the wavelength bins defined? edges? center of bin?
    #that is; 4500.0 AA bin ... is that 4490.0 - 4451.0  or 4500.00 - 4502.0 .....
    if apply_air_to_vac:
        wave = air_to_vac(wave)

    #rescale the wavelengths
    wave /= (1.+z) #shift to lower z and squeeze bins

    flux *= (1.+z)**3 #shift up flux (shorter wavelength = higher energy)


    #todo: deal with wavelength bin compression (change in counting error?)


    #todo: deal with the various errors (that might not have been passed in)
    # if eflux is None or len(eflux) == 0:
    #     eflux = np.zeros(np.shape(flux))
    #
    # if ewave is None or len(ewave) == 0:
    #     ewave = np.zeros(np.shape(flux))

    #todo: deal with luminosity distance (undo dimming?) and turn into luminosity??
    #
    #lum = flux*4.*np.pi*(luminosity_distance(z))**2
    #
    # or just boost the flux to zero distance point source (don't include the 4*pi)
    #
    #flux = flux * (luminosity_distance(z))**2




    return flux, wave, eflux, ewave


def make_grid(all_waves):
    """
    Takes in all the wavelength arrays to figure the best grid so that
    the interpolation only happens on a single grid

    The grid is on the step size of the spectrum with the smallest steps and the range is
    between the shortest and longest wavelengths that are in all spectra.

    :param all_waves: 2d array of all wavelengths
    :return: grid (1d array) of wavelengths

    """

    all_waves = np.array(all_waves)
    #set the range to the maximum(minimum) to the minimum(maximum)
    mn = np.max(np.amin(all_waves,axis=1)) #maximum of the minimums of each row
    mx = np.min(np.amax(all_waves,axis=1)) #minimum of the maximums of each row

    # the step is the smallest of the step sizes
    # assume (within each wavelength array) the stepsize is uniform
    step = np.min(all_waves[:,1] - all_waves[:,0])

    #return the grid
    return np.arange(mn, mx + step, step)


def interpolate(flux,wave,grid,eflux=None,ewave=None):
    """

    :param flux:
    :param wave:
    :param grid:
    :param eflux:
    :param ewave:
    :return: interpolated flux and interpolated flux error
            note: does not return the wavelengths as that is the grid that was passed in
    """

    #todo: how do we really want to handle interpolating the noise (and how much new noise does
    #interpolation add)?

    interp_flux = np.interp(grid, wave, flux)
    if (eflux is not None) and (len(eflux)==len(wave)):
        interp_eflux = np.interp(grid, wave, eflux)
    else:
        interp_eflux = np.zeros(np.shape(interp_flux))

    #todo: dividing by constant with an error, so adjust the error here too
    interp_ewave = ewave

    return interp_flux, interp_eflux, interp_ewave

def add_spectra(flux1,flux2,wave1,wave2,grid=None,eflux1=None,eflux2=None,ewave1=None,ewave2=None):

    """
    Add two spectra, assumed to be already in the same (e.g. rest) frame
    Does not assume they are on the same grid, but the grid is supplied
    Assumes each spectra's grid is fixed width and is strictly increasing to higher indices

    :param flux1:
    :param flux2:
    :param wave1:
    :param wave2:
    :param grid:
    :param eflux1:
    :param eflux2:
    :param ewave1: error in wavelength measures for 1st spectra
                    (note: NOT the wavelengths of the flux error)
    :param ewave2: error in wavelength measures for 2nd spectra
                    (note: NOT the wavelengths of the flux error)
    :return:
    """

    #todo: deal with errers in flux and in wavelength

    #get the maximum overlap between the two arrays
    #not going to add extrapolated values (i.e. outside this minimum overlap)
    w_min = max(wave1[0],wave2[0])
    w_max = min(wave1[-1],wave2[-1])

    #how many datapoints?

    #if grid is not supplied, find which is smaller and add on its grid
    if (grid is None) or (len(grid) == 0):
        step =  min(wave1[1]-wave1[0],wave2[1]-wave2[0])
        #different min, max ... want the grid to cover the maximums
        #when summing, though, the extrapolated values, outside the original range
        #   will not be added
        g_min = min(wave1[0],wave2[0])
        g_max = max(wave1[-1],wave2[-1])
        grid = np.arange(g_min,g_max+step,step)

    #(linear) interpolate each onto the same grid
    #todo: not sure linear interpolation of error is the best .. maybe should take larger of two nearest?
    # ... what does this do to the noise?
    interp_flux1 = np.interp(grid, wave1, flux1)
    if (eflux1 is not None) and (len(eflux1)==len(wave1)):
        interp_eflux1 = np.interp(grid, wave1, eflux1)
    else:
        interp_eflux1 = np.zeros(np.shape(interp_flux1))

    interp_flux2 = np.interp(grid, wave2, flux2)
    if (eflux2 is not None) and (len(eflux2) == len(wave2)):
        interp_eflux2 = np.interp(grid, wave2, eflux2)
    else:
        interp_eflux2 = np.zeros(np.shape(interp_flux2))

    #then add
    flux = interp_flux1 + interp_flux2
    eflux = interp_eflux1 + interp_eflux2

    #todo: how to deal with any errors in the wavelength determination (bin)?
    #z is determined from the wavelength, which has error and then the shift is determined from z

    #then crop to the overlap region
    i, mn, _ = getnearpos(grid,w_min) #want the nearest position greater than the minimum overlap position

    if mn is None:
        mn = i

    i, _,mx = getnearpos(grid, w_max) #want the nearest position less than the maximum overlap position

    if mx is None:
        mx = i

    flux = flux[mn:mx+1]
    eflux = eflux[mn:mx+1]
    wave = grid[mn:mx+1]
    ewave = np.zeros(np.shape(wave))  #for now, no error


    return flux,wave,eflux,ewave


def red_vs_blue(cwave,wave,flux,flux_err,fwhm=None):
    """
    Stack the spectra on the red side and blue side of the emission line (cwave) and check for consistency vs zero continuum.

    LAE should be very red color by this (little contiuum to blue , more to the red side of LyA)
    but OII should be red also (less to blue, a bit more to red, approaching the Balmer break),
    so this may be a matter of degree OR detection to red but not to blue?)

    :param cwave:
    :param flux: in CGS as flux density (erg/s/cm2/AA)
    :param waves: in AA
    :param flux_err:
    :return: dictionary
    """
    min_bins = 20 #have to have at least min_bins on each size to compute
    min_detect_sigma = 1.0 #what multiple above err must the detection be to be used?

    try:
        #assume waves of same step size (2AA for HETDEX)
        if (cwave is None) or (flux is None) or (wave is None) or (flux_err is None) \
            or (len(flux) != len(wave)) or (len(flux) != len(flux_err)) or (len(flux) < 100):
            log.debug("RvB Invalid parameters")
            return None

        step = wave[1]-wave[0]
        length = len(wave)

        #split the spectrum into a red side (cwave + 20 AA to the end) and a blue side (end to cwave-20AA)
        idx,lt,rt = getnearpos(wave,cwave)

        if fwhm is not None and fwhm > 0.0:
            blue_idx,_,_ = getnearpos(wave,cwave-2*fwhm)
            red_idx,_,_ = getnearpos(wave,cwave+2*fwhm)
        else: #if no fwhm go +/- 20AA from central wave
            blue_idx,_,_ = getnearpos(wave,cwave-20.0)
            red_idx,_,_ = getnearpos(wave,cwave+20.0)

        if (blue_idx <= 0) or (red_idx >= length):
            log.debug("RvB color, unusable index (b: %d), (r: %d)" %(blue_idx,red_idx))
            return None

        if ((blue_idx+1) < min_bins) or ((len(wave)-red_idx) < min_bins):
            log.debug("RvB color, insufficient bins. Indicies: (b: %d), (r: %d)" % (blue_idx, red_idx))
            return None

        #sum up flux
        #sum up flux_err in quadrature

        blue_bins = blue_idx + 1
        blue_width = blue_bins * step
        blue_mid_wave = wave[int(blue_idx/2)]
        blue_side = np.array(flux[0:blue_idx+1])
        blue_flux = np.sum(blue_side)
        blue_err = np.sqrt(np.sum(np.array(flux_err[0:blue_idx+1])**2))

        # #as mean (with error)
        # blue_flux_density = blue_flux / blue_bins
        # blue_flux_density_err = blue_err / blue_bins
        #
        # #or as median
        # blue_flux_density = np.median(blue_side)
        # blue_flux_density_err = np.std(blue_side)/np.sqrt(blue_bins)
        #
        # #or as biweight
        # blue_flux_density = biweight.biweight_location(blue_side)
        # blue_flux_density_err = biweight.biweight_scale(blue_side)/np.sqrt(blue_bins)

        #todo: probably won't matter much, but should we force blue and red to both be weighted or un-weighted?
        #should be a very rare (if ever) case
        #or as weighted biweight
        try:
            blue_flux_density = weighted_biweight.biweight_location_errors(blue_side,errors=flux_err[0:blue_idx+1])
            blue_flux_density_err = biweight.biweight_scale(blue_side)/np.sqrt(blue_bins)
        except:
            log.info("Weighted_biweight failed. Switching to normal biweight")
            blue_flux_density = biweight.biweight_location(blue_side)
            blue_flux_density_err = biweight.biweight_scale(blue_side)/np.sqrt(blue_bins)


        #now to jansky
        blue_flux_density_ujy = cgs2ujy(blue_flux_density,blue_mid_wave)
        blue_flux_density_err_ujy = cgs2ujy(blue_flux_density_err,blue_mid_wave)


        red_bins = (len(wave) - red_idx)
        red_width = red_bins * step
        red_mid_wave = wave[int((len(wave)+red_idx) / 2)]
        red_side = np.array(flux[red_idx:])
        red_flux = np.sum(red_side)
        red_err = np.sqrt(np.sum(np.array(flux_err[red_idx:]) ** 2))

        # #as mean with error
        # red_flux_density = red_flux / red_bins
        # red_flux_density_err = red_err / red_bins
        #
        # #or as median
        # red_flux_density = np.median(red_side)
        # red_flux_density_err = np.std(red_side)/np.sqrt(red_bins)
        #
        # # or as biweight
        # red_flux_density = biweight.biweight_location(red_side)
        # red_flux_density_err = biweight.biweight_scale(red_side) / np.sqrt(red_bins)

        # or as weighted biweight
        try:
            red_flux_density = weighted_biweight.biweight_location_errors(red_side,errors=flux_err[red_idx:])
            red_flux_density_err = biweight.biweight_scale(red_side) / np.sqrt(red_bins)
        except:
            log.info("Weighted_biweight failed. Switching to normal biweight")
            red_flux_density = biweight.biweight_location(red_side)
            red_flux_density_err = biweight.biweight_scale(red_side) / np.sqrt(red_bins)

        # now to jansky
        red_flux_density_ujy = cgs2ujy(red_flux_density, red_mid_wave)
        red_flux_density_err_ujy = cgs2ujy(red_flux_density_err, red_mid_wave)

        rvb = {}
        rvb['blue_flux_density'] = blue_flux_density
        rvb['blue_flux_density_err'] = blue_flux_density_err
        rvb['blue_width'] = blue_width
        rvb['blue_flux_density_ujy'] = blue_flux_density_ujy
        rvb['blue_flux_density_err_ujy'] = blue_flux_density_err_ujy


        rvb['red_flux_density'] = red_flux_density
        rvb['red_flux_density_err'] = red_flux_density_err
        rvb['red_width'] = red_width
        rvb['red_flux_density_ujy'] = red_flux_density_ujy
        rvb['red_flux_density_err_ujy'] = red_flux_density_err_ujy

        # if False: #as CGS ... remember though, these are per AA
        # ...so would need to then have (x_flux_denisty * x_wave / (y_flux_density * y_wave)) * (x_wave/y_wave)
        # or the ratio * (x_wave/y_wave)**2   (remember per lambda is inverse to energy where frequency is not)
        #     if (red_flux_density != 0) and (blue_flux_density != 0):
        #         ratio = blue_flux_density / red_flux_density
        #         ratio_err = abs(ratio) * np.sqrt((blue_flux_density_err/blue_flux_density)**2 +
        #                                     (red_flux_density_err/red_flux_density)**2)
        #     else:
        #         ratio = 1.0
        #         ratio_err = 0.0
        # else: #as uJy


        #can not have nan's (all log arguments here are stictly positive?)
        rvb['blue'] = -2.5*np.log10(blue_flux_density_ujy)
        rvb['blue_err'] = -2.5*np.log10(blue_flux_density_err_ujy)
        rvb['red'] = -2.5*np.log10(red_flux_density_ujy)
        rvb['red_err'] = -2.5*np.log10(red_flux_density_err_ujy)

        #check for detection ... if signal is below the error (min_detect_sigma), replace
        #the signal with the 1-sigma error (max value) and treat as a limit on the color

        log.debug("Spectrum (pseudo) blue = (%f,+/-%f) uJy" % (blue_flux_density_ujy,blue_flux_density_err_ujy) )
        log.debug("Spectrum (pseudo)  red = (%f,+/-%f) uJy" % (red_flux_density_ujy, red_flux_density_err_ujy))

        #blue side detection
        flag = 0 #bit-wise:   01 = blue limit, 10 = red limit, 11 = both (meaningless then?)
        if blue_flux_density_ujy / blue_flux_density_err_ujy < min_detect_sigma:
            blue_flux_density_ujy = blue_flux_density_err_ujy #1-sigma limit
            flag = 1 #blue-side sigma limit

        #red side detection
        if red_flux_density_ujy / red_flux_density_err_ujy < min_detect_sigma:
            red_flux_density_ujy = red_flux_density_err_ujy #1-sigma limit
            flag += 2 #red-side sigma limit


        rvb['flag'] = flag
        if flag == 0:
            rvb['flag_str'] = "good"
        elif flag == 1:
            rvb['flag_str'] = "lower limit"
        elif flag == 2:
            rvb['flag_str'] = "upper limit"
        else: #flag == 3
            rvb['flag_str'] = "non-detect"


        if (red_flux_density_ujy != 0) and (blue_flux_density_ujy != 0):
            ratio = red_flux_density_ujy / blue_flux_density_ujy
            ratio_err = abs(ratio) * np.sqrt((blue_flux_density_err_ujy/blue_flux_density_ujy)**2 +
                                        (red_flux_density_err_ujy/red_flux_density_ujy)**2)
        else:
            ratio = 1.0
            ratio_err = 0.0


        rvb['ratio'] = ratio
        rvb['ratio_err'] = ratio_err

        #todo: non detections ... enforce at least 30 datapoints to either side
        #todo: if one side is detected, use the error of the other as a limit
        #i.e. blue = 0 +/- 0.1  red = 10 +/-1  use red/blue as 10/0.1 as 1 sigma limit
        #todo: color 2.5 log (r/b)   (or -2.5 log(b/r)

        rvb['color'] = 2.5 *np.log10(ratio)  #did this as red/blue or red - blue (instead of the usual
                                            #blue filter - red filter, so using 2.5* instead of -2.5* so the  sign
                                            #is correct .... negative = more blue
        #error is +/- ... so add 1st index (gets more red, i.e the reported color is a lower limit)
        #                    add 2nd index (gets more blue, i.e. the reported color is an upper limit)

        if flag: #any problems
            if flag == 1:
                max_red  = 999
                max_blue = rvb['color']
            elif flag == 2:
                max_red = rvb['color']
                max_blue = -999
            else:
                max_red = 999
                max_blue = -999
        else: #all good
            max_red  = np.nan_to_num(2.5 * np.log10(ratio - ratio_err))
            max_blue = np.nan_to_num(2.5 * np.log10(ratio + ratio_err))


        rvb['color_err'] = (max_red-rvb['color'],max_blue-rvb['color']) #or (redder, bluer) or (larger number, smaller number)
        rvb['color_range'] = (max_blue, max_red)

        log.debug("Spectrum (pseudo) color = %0.03g (%0.3g,%0.3g) [%0.3g,%0.3g] flag: %d" \
                             %(rvb['color'],rvb['color_err'][0],rvb['color_err'][1],
                               rvb['color_range'][0],rvb['color_range'][1],flag))

        # rvb['color_err_b'] = -2.5 *np.log10(ratio + ratio_err) #more on red or less on blue ... color gets more RED
        # rvb['color_err_r'] = -2.5 * np.log10(ratio - ratio_err)#less on red or more on blue ... color gets more BLUE

    except:
        log.info("Exception! in spectrum_utilities red_vs_blue.",exc_info=True)
        return None

    return rvb


def velocity_offset(wave1,wave2):
    """
    Return the velocity offset from wave1 to wave2
    (negative values = blue shift = wave2 is blue of wave1)
    :param wave1: wavelength 1 (unitless or units must match)
    :param wave2: wavelength 2 (unitless or units must match)
    :return: velocity offset as quantity
    """

    try:
        return astropy.constants.c.to(U.km/U.s) * (1 - wave1/wave2)
    except:
        return None

def wavelength_offset(wave,velocity):
    """
    Return change in wavelength (observered) for a given base wavelength and velocity offset
    (reminder, negative velocity is toward or blue shifted and the return will be negative)
    :param wave: wavelength 1 (unitless or units must match)
    :param velocity: in km/s
    :return: wavelength (same units as wave)
    """
    try:
        if isinstance(velocity, U.Quantity):
            return wave *  velocity/astropy.constants.c.to(U.km/U.s)
        else:
            return wave *  velocity / astropy.constants.c.to(U.km / U.s).value
    except:
        return None



def extract_at_position(ra,dec,aperture,shotid,ffsky=False,multiproc=G.GET_SPECTRA_MULTIPROCESS):
    """

    :param ra:
    :param dec:
    :param aperture:
    :param shotid:
    :param ffsky:
    :return: in flux e-17 (just like HETDEX standard) (NOT flux density)
    """

    return_dict = {}
    return_dict['flux'] = None
    return_dict['fluxerr'] = None
    return_dict['wave'] = None
    return_dict['ra'] = ra
    return_dict['dec'] = dec
    return_dict['shot'] = shotid
    return_dict['aperture'] = aperture
    return_dict['ffsky'] = ffsky

    # if G.LOG_LEVEL <= 10:  # 10 = DEBUG
    #     get_spectra_loglevel = "INFO"
    # else:
    #     get_spectra_loglevel = "ERROR"

    try:
        coord = SkyCoord(ra=ra * U.deg, dec=dec * U.deg)
        apt = hda_get_spectra(coord, survey=f"hdr{G.HDR_Version}", shotid=shotid,ffsky=ffsky,
                          multiprocess=multiproc, rad=aperture,tpmin=0.0,fiberweights=False,
                              loglevel = "ERROR")

        if len(apt) == 0:
            #print(f"No spectra for ra ({self.ra}) dec ({self.dec})")
            log.info(f"No spectra for ra ({ra}) dec ({dec})")
            return return_dict

        # returned from get_spectra as flux density (per AA), so multiply by wavebin width to match the HDF5 reads
        return_dict['flux'] = np.nan_to_num(apt['spec'][0]) * G.FLUX_WAVEBIN_WIDTH   #in 1e-17 units (like HDF5 read)
        return_dict['fluxerr'] = np.nan_to_num(apt['spec_err'][0]) * G.FLUX_WAVEBIN_WIDTH
        return_dict['wave'] = np.array(apt['wavelength'][0])
        return_dict['apcor'] =  np.array(apt['apcor'][0])
        return_dict['ra'] = ra
        return_dict['dec'] = dec
    except Exception as E:
        print(f"Exception in Elixer::spectrum_utilities::extract_at_position",E)
        log.info(f"Exception in Elixer::spectrum_utilities::extract_at_position",exc_info=True)

    return return_dict



def extract_at_multiple_positions(ra,dec,aperture,shotid,ffsky=False,multiproc=G.GET_SPECTRA_MULTIPROCESS):
    """

    :param ra: list
    :param dec: list
    :param aperture: single value
    :param shotid: single value
    :param ffsky: single value
    :return: in flux e-17 (just like HETDEX standard) (NOT flux density)
    """



    return_list = list(np.full(len(ra),None)) #list of return_dicts
    dummy_dict = {}
    dummy_dict['flux'] = np.full(len(G.CALFIB_WAVEGRID),np.nan)
    dummy_dict['fluxerr'] = np.full(len(G.CALFIB_WAVEGRID),0.0)
    dummy_dict['wave'] = G.CALFIB_WAVEGRID
    dummy_dict['ra'] = None
    dummy_dict['dec'] = None
    dummy_dict['shot'] = shotid
    dummy_dict['aperture'] = aperture
    dummy_dict['ffsky'] = ffsky

    try:
        if G.LOG_LEVEL <= 10:  # 10 = DEBUG
            get_spectra_loglevel = "INFO"
        else:
            get_spectra_loglevel = "ERROR"

        coords = SkyCoord(ra=ra * U.deg, dec=dec * U.deg) #list of coords, ra, dec are lists
        apts = hda_get_spectra(coords, ID=np.arange(len(coords)),survey=f"hdr{G.HDR_Version}", shotid=shotid,ffsky=ffsky,
                          multiprocess=multiproc, rad=aperture,tpmin=0.0,fiberweights=False,
                          loglevel = get_spectra_loglevel)

        if len(apts) == 0:
            #print(f"No spectra for ra ({self.ra}) dec ({self.dec})")
            log.info(f"No spectra for ra ({ra}) dec ({dec})")
            return []
        # if len(apts) != len(ra):
        #     log.info(f"Mismatch in returned spectra {len(apts)} vs requested {len(ra)}")
        #     return []

        # returned from get_spectra as flux density (per AA), so multiply by wavebin width to match the HDF5 reads
        #should be in the same order
        for apt in apts:
            return_dict = {}

            return_dict['flux'] = np.nan_to_num(apt['spec']) * G.FLUX_WAVEBIN_WIDTH   #in 1e-17 units (like HDF5 read)
            return_dict['fluxerr'] = np.nan_to_num(apt['spec_err']) * G.FLUX_WAVEBIN_WIDTH
            return_dict['wave'] = np.array(apt['wavelength'])
            return_dict['apcor'] =  np.array(apt['apcor'])
            return_dict['ra'] = coords[apt['ID']].ra.value
            return_dict['dec'] =  coords[apt['ID']].dec.value
            return_dict['shot'] = shotid
            return_dict['aperture'] = aperture
            return_dict['ffsky'] = ffsky

            return_list[apt['ID']] = return_dict


        sel = np.array([r is None for r in return_list])
        #fill in the Nones
        for i in np.arange(len(ra))[sel]:
            dummy_dict['ra'] = ra[i]
            dummy_dict['dec'] = dec[i]
            return_list[i] =  dummy_dict
    except Exception as E:
        print(f"Exception in Elixer::spectrum_utilities::extract_at_position",E)
        log.info(f"Exception in Elixer::spectrum_utilities::extract_at_position",exc_info=True)

    return return_list

def rms(data, fit,cw_pix=None,hw_pix=None,norm=True):
    """

    :param data: (raw) data
    :param fit:  fitted data (on the same scale)
    :param cw_pix: (nearest) pixel (index) of the central peak (could be +/- 1 pix (bin)
    :param hw_pix: half-width (in pixels from the cw_pix) overwhich to calculate rmse (i.e. cw_pix +/- hw_pix)
    :param norm: T/F whether or not to divide by the peak of the raw data
    :return:
    """
    #sanity check
    min_pix = 5  # want at least 5 pix (bins) to left and right
    try:
        if cw_pix is None or hw_pix is None:
            cw_pix = len(data)//2
            hw_pix = cw_pix-1

        if (data is None):
            log.warning("Invalid data (None) supplied for rms.")
            return -999
        elif (fit is None):
            log.warning("Invalid data (fit=None) supplied for rms.")
            return -999
        elif (len(data) != len(fit)):
            log.warning("Invalid data supplied for rms, length of fit <> data.")
            return -999
        elif any(np.isnan(data)):
            log.warning("Invalid data supplied for rms, NaNs in data.")
            return -999
        elif any(np.isnan(fit)):
            log.warning("Invalid data supplied for rms, NaNs in fit.")
            return -999
        elif not (min_pix < cw_pix < (len(data) - min_pix)):
            # could be highly skewed (esp think of large asym in LyA, with big velocity disp. (booming AGN)
            log.warning("Invalid data supplied for rms. Minimum distance from array edge not met.")
            return -999

        if norm:
            mx = max(data)
            if mx < 0:
                log.warning("Invalid data supplied for rms. max data < 0")
                return -999
        else:
            mx = 1.0

        d = np.array(data)/mx
        f = np.array(fit)/mx

        if ((cw_pix is not None) and (hw_pix is not None)):
            left = max(cw_pix - hw_pix,0)
            right = min(cw_pix + hw_pix,len(data))

            #due to rounding of pixels (bins) from the caller (the central index +/- 2 and the half-width to either side +/- 2)
            # either left or right can be off by a max total of 4 pix
            # rounding_error = 4
            # if -1*rounding_error <= left < 0:
            #     left = 0
            #
            # if len(data) < right <= (len(data) +rounding_error):
            #     right = len(data)


            if (left < 0) or (right > len(data)):
                log.warning("Invalid range supplied for rms. Data len = %d. Central Idx = %d , Half-width= %d"
                            % (len(data),cw_pix,hw_pix))
                return -999

            d = d[left:right+1]
            f = f[left:right+1]

        return np.sqrt(((f - d) ** 2).mean())
    except:
        return -1 #non-sense value for snr

def gaussian(x, x0, sigma, a=1.0, y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None

    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y


def dbl_gaussian(x, u1, s1, A1, u2, s2, A2, y):
    return A1 * (np.exp(-np.power((x - u1) / s1, 2.) / 2.) / np.sqrt(2 * np.pi * s1 ** 2)) + \
           A2 * (np.exp(-np.power((x - u2) / s2, 2.) / 2.) / np.sqrt(2 * np.pi * s2 ** 2)) + y

def simple_fit_line (wavelengths, values, errors=None,trim=True,lines=None):
    """
    Just a least squares fit, no MCMC
    Trim off the blue and red-most regions that are a bit dodgy

    :param wavelengths: unitless floats (but generally in AA)
    :param values: unitless floats (but generally in erg/s/cm2 over 2AA e-17) (HETDEX standard)
                   the caller must set to flux, flam, or fnu as needed
    :param errors: unitless floats (same as values)
    :param trim: Trim off the blue and red-most regions that are a bit dodgy
    :param lines: list of emission lines (and/or absorption) to mask out
    :return: [intercept, slope] , [error on intercept, error on slope]
    """

    if (wavelengths is None) or (values is None) or (len(wavelengths)==0) or (len(values)==0) \
            or (len(wavelengths) != len(values)):
        log.warning("Zero length (or None) spectrum passed to simple_fit_line().")
        return None, None

    try:

        #mask first
        mask = np.full(len(wavelengths),True) #we will keep the True values
        if lines is not None:
            for l in lines:
                idx,*_ = getnearpos(wavelengths,l.fit_x0)
                width = int(l.fit_sigma * 3.0)
                left = max(0,idx-width)
                right = min(len(wavelengths),idx+width+1)
                mask[left:right]=False

        if trim  and (len(wavelengths) == 1036): #assumes HETDEX standard rectified 2AA wide bins 3470-5540
            mask[0:66] = False
            mask[966:] = False

        #local copy to protect values
        _values = copy.copy(np.array(values))

        #check the errors
        try:
            if errors is not None and len(errors)==len(values):
                weights = 1./np.array(errors)

                #any weights or weights that correspond to values that are nan or zero get a 0 weight
                weights[np.isnan(_values)] = 0.0 #zero out their weights
                weights[np.isinf(_values)] = 0.0 #zero out their weights
                weights[np.isnan(weights)] = 0.0
                weights[np.isinf(weights)] = 0.0
                weights = weights[mask]
            else:
                weights = None
        except:
            weights = None

        #set to innocuous values
        _values[np.isnan(_values)] = 0.0
        _values[np.isneginf(_values)] = np.min(_values[np.invert(np.isinf(_values))])
        _values[np.isposinf(_values)] = np.max(_values[np.invert(np.isinf(_values))])

        coeff, cov  = np.polyfit(np.array(wavelengths)[mask], _values[mask],
                                 w=weights,cov=True,deg=1)


        #flip the array so [0] = 0th, [1] = 1st ...
        errors = np.flip(np.sqrt(np.diag(cov)),0)
        coeff = np.flip(coeff,0)
    except:
        log.debug("Exception in simple_fit_line() ", exc_info=True)
        coeff = None
        errors = None

    return coeff, errors



def eval_line_at_point(w,bm,er=None):

    """
    Basically evaluate a point on a line given the line parameters and (optionally) their variances
    The units depend on the units of the parameters and is up to the caller to know them.
    For HETDEX/elixer they are usually flux over 2AA in erg/s/cm2 x10^-17

    :param w: the wavelength at which to evaluate
    :param bm:  as in y = mx + b  ... a two value array as [b,m] or [x^0, x^1]
    :param er:  the error on b and m as a two value array  (as sigma or sqrt(variance))
    :return: continuum and error
    """
    try:
        y = bm[1] * w + bm[0]
        ye = abs(y) * np.sqrt((er[1]/bm[1])**2 + er[0]**2)
    except:
        y = None
        ye = None

    return y, ye

def est_linear_continuum(w,bm,er=None):
    return eval_line_at_point(w,bm,er)

def est_linear_B_minus_V(bm,er=None):
    """

    :param bm:  as in y = mx + b  ... a two value array as [b,m] or [x^0, x^1]
    :param er:  the error on b and m as a two value array  (as sigma or sqrt(variance))
    :return: B-V and error
    """
    try:

        yb, ybe = est_linear_continuum(4361.,bm,er) #standard f_iso,lam for B
        yv, yve = est_linear_continuum(5448.,bm,er) #standard f_iso,lam for V

        #need to convert from flux over 2AA to a f_nu like units
        #since B-V is a ratio, the /2.0AA and x1e-17 factors don't matter

        yb = cgs2ujy(yb,4361.)
        ybe = cgs2ujy(ybe,4361.)

        yv = cgs2ujy(yv,5448.)
        yve = cgs2ujy(yve,5448.)

        fac = 2.5/np.log(10.) #this is the common factor in the partial derivative of the 2.5 log10 (v/b)

        b_v = 2.5 * np.log(yv/yb)
        b_ve = abs(b_v)*np.sqrt((ybe*fac/yv)**2 + (yve*fac/yb)**2)

    except:
        b_v = None
        b_ve = None

    return b_v, b_ve


    """
    default min_sigma is from HETDEX spec, at 1.7
    #HETDEX uses +/- 50AA in steps of 4AA
    """

def quick_fit(waves, flux, flux_err, w, delta_w=4.0, width=50, min_sigma=1.7, max_sigma=20,absorber=False):
    """

    :param waves: in AA
    :param flux:  in arbitraty unit
    :param flux_err:
    :param w: central wavelength to fit a Gaussian
    :param delta_w: allowed range to either side of w where the x0 can be fitted (in AA)
    :param width:  width (in AA) to either side of w in which to sample and fit
    :param min_sigma:
    :param max_sigma:
    :return:  snr, chi2, ew, parm, pcov, model_fit_full
    """

    center_pix, *_ = getnearpos(waves, w)

    # width is to either side and is in AA here, so need to turn to pix
    width = int(width / G.FLUX_WAVEBIN_WIDTH)  # to turn from AA to pix, where eachpix is 2AA wide

    left = max(0, center_pix - width)
    right = min(len(waves) - 1, center_pix + width)

    narrow_wave_x = waves[left:right + 1]
    narrow_wave_y = flux[left:right + 1]

    errors = flux_err[left:right + 1]
    errors[errors == 0] = np.nan
    wave_err_sigma = 1. / (errors * errors)  # double checked and this is correct (assuming errors is +/- as expected)

    # gaussian(x, x0, sigma, a=1.0, y=0.0):
    try:
        if absorber: #area goes negative
            parm, pcov = curve_fit(f=gaussian,
                               xdata=np.float64(narrow_wave_x),
                               ydata=np.float64(narrow_wave_y),
                               p0=(w, min_sigma, -1.0, 0.0),
                               absolute_sigma=False,
                               # bounds limit to fitting at THIS wavelength or either adjacent bin
                               bounds=((w - delta_w, min_sigma,
                                        -1 * max(narrow_wave_y) * len(narrow_wave_y),
                                        min(flux) - max(flux)),
                                       (w + delta_w, max_sigma,
                                        0.0,
                                        max(flux) * 1.5)),

                               # sigma=1./(narrow_wave_errors*narrow_wave_errors)
                               sigma=wave_err_sigma, # , #handles the 1./(err*err)
                               # note: if sigma == None, then curve_fit uses array of all 1.0
                               #method='trf',
                               )
        else:
            parm, pcov = curve_fit(f=gaussian,
                               xdata=np.float64(narrow_wave_x),
                               ydata=np.float64(narrow_wave_y),
                               p0=(w, min_sigma, 1.0, np.nanmedian(narrow_wave_y)), #0.0 #maybe use the median value?
                               absolute_sigma=False,
                               # bounds limit to fitting at THIS wavelength or either adjacent bin
                               # bounds=((w - delta_w, min_sigma, 0.0,
                               #          min(flux) - max(flux)),
                               #         (w + delta_w, max_sigma, max(narrow_wave_y) * len(narrow_wave_y),
                               #          max(flux) * 1.5)),
                               bounds=((w - delta_w, min_sigma, 0.0,
                                            min(narrow_wave_y)-errors[np.argmin(narrow_wave_y)]),
                                           (w + delta_w, max_sigma, max(narrow_wave_y) * len(narrow_wave_y),
                                            max(narrow_wave_y) + errors[np.argmax(narrow_wave_y)])),

                               # sigma=1./(narrow_wave_errors*narrow_wave_errors)
                               sigma=wave_err_sigma,  # , #handles the 1./(err*err)
                               # note: if sigma == None, then curve_fit uses array of all 1.0
                               #method='trf'
                               )
    except Exception as ex:
        try:
            if ex.args[0].find("Optimal parameters not found") > -1:
                log.debug("Could not fit gaussian (1a) near %f" % w, exc_info=False)
            elif ex.args[0].find("is infeasible") > -1:
                log.debug("Could not fit gaussian (2a) near %f" % w, exc_info=False)
            elif ex.args[0].find("Each lower bound must be strictly less than each upper bound") > -1:
                log.debug("Could not fit gaussian (2.1a) near %f" % w, exc_info=False)
            else:
                log.error("Could not fit gaussian (3a) near %f" % w, exc_info=True)
        except:
            log.error("Could not fit gaussian (4a) near %f" % w, exc_info=True)

        return 0, 999, 0, [0, 0, 0, 0], np.zeros((4, 4)), np.zeros(len(waves))

    perr = np.sqrt(np.diag(pcov))
    if not np.any(pcov):
        print("problem. pcov all zero")

    xfit = np.linspace(waves[left], waves[right], len(narrow_wave_x))  # or make higher res at say 1000 ?
    # if higher res, does that screw up the chi2??

    model_fit = gaussian(waves[left:right + 1], parm[0], parm[1], parm[2], parm[3])
    model_fit_full = gaussian(waves, parm[0], parm[1], parm[2], parm[3])

    # need to decrease the snr to 2.5 sigma?
    delta_wave = max(parm[1] * 2.0, 2.1195)  # must be at least +/- 2.1195AA
    # new left and right
    snr_left, *_ = getnearpos(waves, parm[0] - delta_wave)
    snr_right, *_ = getnearpos(waves, parm[0] + delta_wave)

    if waves[left] - (parm[0] - delta_wave) < 0:
        left += 1  # less than 50% overlap in the left bin, so move one bin to the red
    if waves[right] - (parm[0] + delta_wave) > 0:
        right -= 1  # less than 50% overlap in the right bin, so move one bin to the blue

    snr = abs(np.sum(model_fit_full[snr_left:snr_right + 1] - parm[3])) / np.sqrt(
        np.sum(flux_err[snr_left:snr_right + 1] ** 2))  # or error as flux_err[left:right+1]**2

    # chi2 should be checked over the full width of the fit, not the +/- 2 sigma of the emission
    # that left/right is set at the top of this func
    chi2, _ = chi_sqr(flux[left:right + 1], model_fit_full[left:right + 1], error=flux_err[left:right + 1],
                             c=1.0, dof=2)

    try:
        ew = parm[2] / parm[3]  # "area" / y (observed)
    except:
        ew = 0

    return snr, chi2, ew, parm, pcov, model_fit_full


def quick_linescore(snr, chi2, sigma, ew, data_side_aa=40.0):
    """
    something like the line score, but simpler ... just for fast comparision
    """
    try:
        #HETDEX specific, normally 40-50 AA on a side is used as best,
        #narrower allowances can give a slightly higher quick score
        if data_side_aa < 40.0:
            w_mult = 0.9
        else:
            w_mult = 1.0

        if sigma < 2.0 or sigma > 8.5:
            s_mult = 0.9
        else:
            s_mult = 1.0

        # return 5.0 * snr / np.sqrt(2 * sigma) \
        #        - 1.0 * max(0.1, chi2) \
        #        + 1.0 * abs(min(200, ew) / np.sqrt(2 * sigma) / 100.0)

        return w_mult * s_mult * ( \
             (5.0 * snr)  \
           - (1.0 * max(0.1, chi2)) \
           + (1.0 * abs(min(200, ew) / 100.0)) \
              )
    except:
        return 0



def quick_line_finder(waves,flux,flux_err,delta_w=2.0,width=50,min_sigma=1.5,max_sigma=8.5, absorber=False):

    """
    HETDEX is +/- 50 in steps of 4AA (or 8AA, depending on search)
    """

    try:
        max_idx = None
        all_snr = []
        all_chi2 = []
        all_sigma = []
        all_area = []
        all_cw =[]
        all_ew = []


        #Karl's search is in steps of 8AA ... the step of 2AA here might be too narrow
        waves = np.arange(3470,5542,2)
        for w in tqdm(waves):
            snr, chi2, ew, parm, pcov, model = fit(waves,flux,flux_err,w=w,delta_w=delta_w,width=width,
                                                   min_sigma=min_sigma,max_sigma=max_sigma,absorber=absorber)

            #reject any that are at maximum sigma
            if np.isclose(parm[1],max_sigma,atol=1e-2) or parm[1] > max_sigma:
                pass
            else:
                all_snr.append(snr)
                all_chi2.append(chi2)
                all_ew.append(ew)
                all_sigma.append(parm[1])
                all_area.append(parm[2])
                all_cw.append(parm[0])


        waves = np.array(waves)
        all_cw = np.array(all_cw)
        all_chi2 = np.array(all_chi2)
        all_snr = np.array(all_snr)
        all_sigma = np.array(all_sigma)
        all_area = np.array(all_area)
        all_ew = np.array(all_ew)
        all_score = np.zeros(len(all_ew))

        #condense ... keep a "best" from sets within some distance
        if True:
            csel = np.full(len(all_cw),False)
            ci = 0 #current index
            #wave_limit = 6.0 #AA #maybe dynamic? like at least 2.0 but up to 2*sigma?
            csel[ci] = True
            all_score[ci] = simple_score(all_snr[ci],all_chi2[ci],all_sigma[ci],all_ew[ci])
            for i in range(1,len(all_cw)):

                wave_limit = np.clip(3*all_sigma[ci],6.0,20.0)

                if abs(all_cw[i] - all_cw[ci]) < wave_limit:
                    all_score[i] = simple_score(all_snr[i],all_chi2[i],all_sigma[i],all_ew[i])
                    if all_score[i] > all_score[ci]: #replace the current index with the new index
                        csel[ci] = False
                        ci = i
                        csel[i] = True
                else:
                    ci = i
                    csel[ci] = True
                    all_score[ci] = simple_score(all_snr[ci],all_chi2[ci],all_sigma[ci],all_ew[ci])


        #choose the "Best" to keep
        #first, must have a reasonable chi2 , but don't be too strict
        #best_sel = csel & np.array(all_chi2/np.sqrt(2*all_sigma) < 1.1)
        best_sel = csel & np.array(all_chi2 < 5.0)
        #and a minimum sigma
        best_sel = best_sel & np.array(all_sigma > 1.5)
        #and a minimum SNR
        best_sel = best_sel & np.array(all_snr > 3.0)
        best_sel = best_sel & np.array(all_score > 5.0)
        #more restrictive
        best_sel = best_sel & ( np.array(all_snr > 4.5) | np.array(all_score > 7.0) )


        if True: #debugging
            try:
                max_idx = np.argmax(np.array(all_score)[best_sel])
                print(f"Candidates = {np.count_nonzero(csel)}, {np.count_nonzero(best_sel)}")
                #print(f"wavebin = {waves[best_sel][max_idx]}, wave = {all_cw[best_sel][max_idx]:0.2f}, \
    #             print(f"wave = {all_cw[best_sel][max_idx]:0.2f}, \
    #             snr = {all_snr[best_sel][max_idx]:0.2f}, chi2 = {all_chi2[best_sel][max_idx]:0.2f}, \
    #             sigma = {all_sigma[best_sel][max_idx]:0.2f}, area = {all_area[best_sel][max_idx]:0.2f}, \
    #             flux = {all_area[best_sel][max_idx]/20.0:0.2f}, ew = {all_ew[best_sel][max_idx]:0.2f}, \
    #             score = {all_score[best_sel][max_idx]:0.2f}")

                print("wave",[float(f"{x:0.2f}") for x in all_cw[best_sel]])
                print(" snr",[float(f"{x:0.2f}") for x in all_snr[best_sel]])
                print("chi2",[float(f"{x:0.2f}") for x in all_chi2[best_sel]])
                print("sigm",[float(f"{x:0.2f}") for x in all_sigma[best_sel]])
                print("scre",[float(f"{x:0.2f}") for x in all_score[best_sel]])
            except:
                pass

        if True:
            #return only the "best"
            all_cw = all_cw[best_sel]
            all_chi2 = all_chi2[best_sel]
            all_snr = all_snr[best_sel]
            all_sigma = all_sigma[best_sel]
            all_area = all_area[best_sel]
            all_ew = all_ew[best_sel]
            all_score = all_score[best_sel]
            try:
                max_idx = np.argmax(all_score)
                #print("****",max_idx)
            except Exception as e:
                print(e)
                max_idx = None

        return max_idx, all_cw, all_snr, all_score, all_ew, all_chi2, all_sigma, all_area

    except Exception as e:
        print(e)
        return None, [], [], [], [], [], [], []


def simple_fit_wave(values,errors,wavelengths,central,wave_slop_kms=500.0,max_fwhm=15.0):
    """
    Simple curve_fit to gaussian; lsq "best"

    :param values:
    :param errors:
    :param wavelengths:
    :param central:
    :param wave_slop_kms: +/- from central wavelength in km/s (default of 1000 km/s ~ +/-13AA at 4500AA)
    :param max_fwhm: in AA
    :return:
    """

    return_dict = {}
    return_dict['x0'] = None
    return_dict['fitflux'] = 0.0
    return_dict['continuum_level'] = 0.0
    return_dict['velocity_offset'] = 0.0
    return_dict['sigma'] = 0.0
    return_dict['rmse'] = 0.0
    return_dict['snr'] = 0.0
    return_dict['chi2'] = 9999.9
    return_dict['meanflux_density'] = 0.0
    return_dict['velocity_offset_limit'] = wave_slop_kms  # not memory efficient to store the same value
    # repeatedly, but, thinking ahead to maybe allowing the max velocity limit to vary by coordinate
    #return_dict['score'] = 0.0

    if (values is None) or (len(values) == 0):
        return return_dict

    try:

        #change if the user specified something specific
        if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None:
            min_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MIN
            max_sigma = min(G.LIMIT_GAUSS_FIT_SIGMA_MAX, max_fwhm/2.355)
            if min_sigma > max_sigma:
                log.error(f"Error! Error in Gridsearch. Specified minimum sigma is greater than the max. min: {min_sigma}, max: {max_sigma}")
                return return_dict
        else:
            min_sigma = 1.5  # FWHM  ~ 3.5AA (w/o error, best measure would be about 5AA)
            max_sigma = max_fwhm / 2.355

        wave_slop = wavelength_offset(central,wave_slop_kms) #for HETDEX, in AA

        #wave_side = int(np.max([3*wave_slop,3*max_sigma,10.0]) / G.FLUX_WAVEBIN_WIDTH) +1
        #wave_side = int(round(max(40,2*wave_slop) / G.FLUX_WAVEBIN_WIDTH)) #at least 40AA to either side or twice the slop
        wave_side = int(40/G.FLUX_WAVEBIN_WIDTH) #wavebins, not AA
        idx,_,_ = getnearpos(wavelengths,central)
        min_idx = max(0,idx-wave_side)
        max_idx = min(len(values),idx+wave_side)

        narrow_wave_x = wavelengths[min_idx:max_idx+1]
        narrow_wave_counts = values[min_idx:max_idx+1]

        #what about fractionals?
        _, left, _ = getnearpos(wavelengths,central-wave_slop)
        _,_,right = getnearpos(wavelengths,central+wave_slop)
        return_dict['meanflux_density'] = np.nansum(values[left:right+1])/\
                                         (wavelengths[right+1]-wavelengths[left])
        #reminder, max_idx+1 since we want to include the WIDTH of the last bin in the sum
        #also, assumes input is in flux units and the wavebins are the same width

        if (errors is not None) and (len(errors) == len(wavelengths)):
            narrow_wave_errors = errors[min_idx:max_idx+1]
            #replace any 0 with 1
            narrow_wave_errors[np.where(narrow_wave_errors == 0)] = 1
            narrow_wave_err_sigma = 1. / (narrow_wave_errors * narrow_wave_errors)
        else:
            narrow_wave_errors = None
            narrow_wave_err_sigma = None

        mx_ct = np.nanmax(narrow_wave_counts)
        mn_ct = np.nanmin(narrow_wave_counts)
        try:

            parm, pcov = curve_fit(gaussian, np.float64(narrow_wave_x), np.float64(narrow_wave_counts),
                                   p0=(central, 0.5*(min_sigma+max_sigma), 1.0, mn_ct),
                                   bounds=((central - wave_slop, min_sigma, 0.0, mn_ct-0.5*(mx_ct-mn_ct)),
                                           (central + wave_slop, max_sigma, 2*wave_slop*mx_ct,mx_ct)),
                                   # sigma=1./(narrow_wave_errors*narrow_wave_errors)
                                   sigma=narrow_wave_err_sigma,  # , #handles the 1./(err*err)
                                   # note: if sigma == None, then curve_fit uses array of all 1.0
                                   #method='trf'
                                   )

            try:
                if not np.any(pcov):  # all zeros ... something wrong
                    log.info("Something very wrong with curve_fit")
            except Exception as E:
                print(E)
                return return_dict

            perr = np.sqrt(np.diag(pcov))

            x0 = parm[0]
            sigma = parm[1]
            A = parm[2]
            Y = parm[3]
            fitflux = A / G.FLUX_WAVEBIN_WIDTH
            continuum_level = Y
            vel_offset = velocity_offset(central, x0).value

            #sn_pix to either side of the peak
            num_sn_pix = int(round(min(20, len(narrow_wave_counts)// 2 - 1)))  # don't go larger than the actual array

            rms_wave = gaussian(narrow_wave_x, parm[0], parm[1], parm[2], parm[3])
            if (2 * sigma * 2.355) > (len(narrow_wave_counts)):
                #could be very skewed and broad, so don't center on the emission peak, but center on the array
                fit_rmse = rms(narrow_wave_counts, rms_wave, cw_pix=len(narrow_wave_counts) // 2, hw_pix=num_sn_pix,
                           norm=False)
            else:
                fit_rmse = rms(narrow_wave_counts, rms_wave, cw_pix=getnearpos(narrow_wave_x, x0 )[0], hw_pix=num_sn_pix,
                                 norm=False)

            num_sn_pix = num_sn_pix * 2 + 1 #update to full counting of sn_pix (each side + 1 for the peak bin)
            #old computation
            #snr = A / (np.sqrt(num_sn_pix) * fit_rmse)


            #updated SNR and Chi2 to match mcmc and signalscore
            sigma_width = 2.0
            delta_wave = max(sigma * sigma_width, 2.1195)  # must be at least +/- 2.1195AA
            left, *_ = getnearpos(wavelengths, x0 - delta_wave)
            right, *_ = getnearpos(wavelengths, x0 + delta_wave)
            if wavelengths[left] - (x0 - delta_wave) < 0:
                left += 1  # less than 50% overlap in the left bin, so move one bin to the red
            if wavelengths[right] - (x0 + delta_wave) > 0:
                right -= 1  # less than 50% overlap in the right bin, so move one bin to the blue

            right += 1  # since the right index is not included in slice

            model_fit = gaussian(wavelengths[left:right], x0, sigma, A,Y)
            data_err = copy.copy(errors[left:right])
            data_err[data_err <= 0] = np.nan
            data_flux = values[left:right]
            snr = abs(np.sum(model_fit - Y)) / np.sqrt(np.nansum(data_err ** 2))
            #elsewhere we are using the full width of the spectrum, so do that here too?
            delta_wave = max(5*sigma,6.0 ) #use 40AA
            _,left, _ = getnearpos(wavelengths, x0 - delta_wave)
            _,_,right = getnearpos(wavelengths, x0 + delta_wave)
            if wavelengths[left] - (x0 - delta_wave) < 0:
                left += 1  # less than 50% overlap in the left bin, so move one bin to the red
            if wavelengths[right] - (x0 + delta_wave) > 0:
                right -= 1  # less than 50% overlap in the right bin, so move one bin to the blue

            right += 1  # since the right index is not included in slice


            chi2_model_fit = gaussian(wavelengths[left:right], x0, sigma,A,Y)
            chi2, _ = chi_sqr(values[left:right], chi2_model_fit, error=errors[left:right], c=1.0 ,dof=2)


            # #################
            # # test
            # #################
            # print("!!!!!!! REMOVE ME !!!!!!!")
            # plt.close('all')
            # plot_fit = gaussian(narrow_wave_x,x0, sigma, A,Y)
            # plt.title(f"S/N {snr:0.1f} Chi2 {chi2:0.2f} sig {sigma:0.1f} flux {fitflux:0.2f} cont {continuum_level:0.1f}")
            # plt.errorbar(narrow_wave_x,narrow_wave_counts,yerr=narrow_wave_errors)
            # plt.plot(narrow_wave_x,plot_fit)
            # plt.savefig("chi2_test.png")

            return_dict['x0'] = x0
            return_dict['fitflux'] = fitflux
            return_dict['continuum_level'] = continuum_level
            return_dict['velocity_offset'] = vel_offset
            return_dict['sigma'] = sigma
            return_dict['rmse'] = fit_rmse
            return_dict['chi2'] = chi2
            return_dict['snr'] = snr
            #return_dict['score'] = line_score


            #assume fit line is LyA and get the sum around rest 880-910
            try:
                lyc_obs = (880.0 * (1.+x0/G.LyA_rest), 910.0 * (1.+x0/G.LyA_rest))
                lyc_idx = blue,_,_ = getnearpos(wavelengths,lyc_obs[0])
                lyc_idx = red, _, _ = getnearpos(wavelengths, lyc_obs[1])
                return_dict['f900'] = cgs2ujy(np.sum(values[blue,red+1])/(lyc_obs[1]-lyc_obs[0]),900.0)
                return_dict['f900e'] = cgs2ujy(np.sum(errors[blue,red+1])/(lyc_obs[1]-lyc_obs[0]),900.0)
            except:
                return_dict['f900'] = -999
                return_dict['f900e'] = -999

        except Exception as E:
            print(E)

    except Exception as E:
        print(E)

    return return_dict


def combo_fit_wave(peak_func,values,errors,wavelengths,central,wave_slop_kms=500.0,max_fwhm=15.0):
    """
    try scanning for lines and if that fails call simple_fit_wave
    :param values:
    :param errors:
    :param wavelengths:
    :param central:
    :param wave_slop_kms:
    :param max_fwhm:
    :return:
    """
    return_dict = {}
    return_dict['x0'] = None
    return_dict['fitflux'] = 0.0
    return_dict['continuum_level'] = 0.0
    return_dict['velocity_offset'] = 0.0
    return_dict['sigma'] = 0.0
    return_dict['rmse'] = 0.0
    return_dict['snr'] = 0.0
    return_dict['chi2'] = 9999.9
    return_dict['meanflux_density'] = 0.0
    return_dict['velocity_offset_limit'] = wave_slop_kms

    try_simple_fit = False
    try:

        wave_slop = wavelength_offset(central, wave_slop_kms)
        #extra +/-5 just for safety in being able to fit a line
        _,min_w,_ = getnearpos(wavelengths,central-wave_slop-max_fwhm/2.0)
        _,_,max_w = getnearpos(wavelengths,central+wave_slop+max_fwhm/2.0)

        #extraction puts these in HETDEX e-17 units (as integrated flux, so over x2.0AA bins)
        all_found_lines = peak_func(wavelengths[min_w:max_w+1], values[min_w:max_w+1], errors[min_w:max_w+1],
                                     values_units=-17)

        if all_found_lines is not None and len(all_found_lines) > 0:
            #there a line in the range
            if len(all_found_lines) == 1:
                idx = 0
            else: #get nearest
                all_found_lines.sort(key=lambda x: x.fit_x0, reverse=False)
                idx,_,_ = getnearpos([x.fit_x0 for x in all_found_lines])

            #If we are always going to call simple_fit_wave, then flip the True/False logic here
            if True:
                central = all_found_lines[idx].fit_x0
                max_fwhm = all_found_lines[idx].fit_sigma*2.355*1.1 #give an exta 10% for slop
                wave_slop_kms = 0.5/central * astropy.constants.c.to(U.km/U.s).value #keep it small now that we have an identified wavelength
                try_simple_fit = True
            else:
                return_dict['x0'] = all_found_lines[idx].fit_x0
                return_dict['fitflux'] = all_found_lines[idx].line_flux * 1e17
                return_dict['continuum_level'] = all_found_lines[idx].cont * 1e17
                return_dict['velocity_offset'] = velocity_offset(central, all_found_lines[idx].fit_x0).value
                return_dict['sigma'] = all_found_lines[idx].fit_sigma
                return_dict['rmse'] = all_found_lines[idx].fit_rmse
                return_dict['chi2'] = all_found_lines[idx].fit_chi2
                return_dict['snr'] = all_found_lines[idx].snr

                #now w/o the extra padding
                _, min_w, _ = getnearpos(wavelengths, central - wave_slop)
                _, _, max_w = getnearpos(wavelengths, central + wave_slop)
                return_dict['meanflux_density'] = np.nansum(values[min_w:max_w + 1]) / \
                                                  (wavelengths[max_w + 1] - wavelengths[min_w])

                #assume fit line is LyA and get the sum around rest 880-910
                try:
                    lyc_obs = (880.0 * (1.+x0/G.LyA_rest), 910.0 * (1.+all_found_lines[idx].fit_x0/G.LyA_rest))
                    lyc_idx = blue,_,_ = getnearpos(wavelengths,lyc_obs[0])
                    lyc_idx = red, _, _ = getnearpos(wavelengths, lyc_obs[1])
                    return_dict['f900'] = cgs2ujy(np.sum(values[blue,red+1])/(lyc_obs[1]-lyc_obs[0]),900.0)
                    return_dict['f900e'] = cgs2ujy(np.sum(errors[blue,red+1])/(lyc_obs[1]-lyc_obs[0]),900.0)
                except:
                    return_dict['f900'] = -999
                    return_dict['f900e'] = -999

                return return_dict
        else:
            try_simple_fit = True

    except:
        log.info("Exception in spectrum_utilites::combo_fit_wave",exc_info=True)
        try_simple_fit = True

    #now run simple fit
    if try_simple_fit:
        try:
            return simple_fit_wave(values, errors, wavelengths, central,
                            wave_slop_kms=wave_slop_kms, max_fwhm=max_fwhm)

        except:
            log.info("Exception in spectrum_utilites::combo_fit_wave", exc_info=True)

    return return_dict



def fetch_per_shot_single_fiber_sky_subtraction_residual(path,shotid,column,prefix=None):
    """
    in this version (for testing) all the residual fits files are in one place, with each holding one row for the shot
    only returns the residual ... not the error

    This is applied with the call to HETDEX_API get_spectra() and, as such, this needs to be in:
        erg/s/cm2/AA e-17 (not 2AA)
    This should NOT be de-reddened or have any other such corrections as those are applied as part of get_spectra()

    :param path:
    :param shotid:
    :param column:
    :return: residual
    """

    try:
        if G.APPLY_SKY_RESIDUAL_TYPE != 1:
            return None

        if prefix is None:
            if G.SKY_RESIDUAL_FITS_PREFIX is None:
                #we are not doing this
                log.info("***** No sky subtraction residual configured.")
                return None
            else:
                prefix = G.SKY_RESIDUAL_FITS_PREFIX

        T = None
        file = op.join(path,f"{prefix}{shotid}.fits")
        if op.exists(file):
            T = Table.read(file)
        else:
            #yes, there ends up being two underscores before default
            file = op.join(path,f"{prefix}_default.fits")
            if op.exists(file):
                T = Table.read(file)

        if T is not None:
            log.info(f"***** Returning sky subtraction residual: {file} [{column}.")
            return T[column][0]
        else:
            log.info(f"***** Unable to find sky subtraction residual: {file} {column}.")
    except:
        log.error(f"Exception! Exception loading sky residual for {shotid} + {column}.",exc_info=True)
        return None
    return None

def fetch_universal_single_fiber_sky_subtraction_residual(ffsky=False,hdr=G.HDR_Version):
    """

        This is applied with the call to HETDEX_API get_spectra() and, as such, this needs to be in:
        erg/s/cm2/AA e-17 (not 2AA)
    This should NOT be de-reddened or have any other such corrections as those are applied as part of get_spectra()


    :param ffsky:
    :param hdr:
    :return:
    """

    try:
        if G.APPLY_SKY_RESIDUAL_TYPE != 1:
            return None

        log.debug("Loading universal sky residual model...")
        if G.SKY_RESIDUAL_USE_MODEL:
            which_col = 1
        else:
            which_col = 2
        if hdr == "3":
            if ffsky:
                if G.SKY_RESIDUAL_HDR3_FF_FLUXD is None:
                    # try to load it
                    G.SKY_RESIDUAL_HDR3_FF_FLUXD = np.loadtxt(G.SKY_RESIDUAL_HDR3_FF_FN, usecols=(which_col))

                log.info(f"***** Returning ff sky subtraction residual ({which_col})")
                return G.SKY_RESIDUAL_HDR3_FF_FLUXD
            else: #local sky
                if G.SKY_RESIDUAL_HDR3_LO_FLUXD is None:
                    # try to load it
                    G.SKY_RESIDUAL_HDR3_LO_FLUXD = np.loadtxt(G.SKY_RESIDUAL_HDR3_LO_FN, usecols=(which_col))

                log.info(f"***** Returning local sky subtraction residual ({which_col})")
                return G.SKY_RESIDUAL_HDR3_LO_FLUXD
        else:
            log.warning(f"Unknown HDR version ({hdr}). No sky residual available.")
    except:
        log.error(f"Exception! Exception loading universal sky residual.", exc_info=True)
        return None
    log.error(f"No universal sky residual found.", exc_info=True)
    return None

# ###############################################################
#  There are too many issues with deblending, de-reddening, etc
# (deblinding should happen first to completion, perior to this, etc)
# so this per aperture should be done POST ELIXER and PRE STACKING
# ##############################################################
#
# def fetch_per_shot_aperture_sky_subtraction_residual(path,shotid,column,prefix=None,aperture=3.5):
#     """
#     similar to the fetch_per_shot_single_fiber_sky_subtraction_residual but this is for an entire aperture, not
#     just a single fiber.
#
#     For HETDEX the default aperture is 3.5" radius
#     Any other aperture size is invalid unless specifically adapted.
#     This can be applied to a HETDEX lookup spectrum (i.e. from a data release run) or AFTER HETDEX_API get_spectra()
#
#     !!! this is a PSF weighted, DAR corrected, etc spectrum !!!
#
#     :param path:
#     :param shotid:
#     :param column:
#     :return: residual
#     """
#
#     try:
#         if G.APPLY_SKY_RESIDUAL_TYPE != 2:
#             return None
#
#         if aperture != 3.5:
#             log.error(f"ERROR! Invalid aperture size ({aperture}) requested for fetch_per_shot_aperture_sky_subtraction_residual.")
#             return None
#
#         if prefix is None:
#             if G.APERTURE_SKY_RESIDUAL_FITS_PREFIX is None:
#                 #we are not doing this
#                 log.info("***** No aperture sky subtraction residual configured.")
#                 return None
#             else:
#                 prefix = G.APERTURE_SKY_RESIDUAL_FITS_PREFIX
#
#         T = None
#         file = op.join(path,f"{prefix}{shotid}.fits")
#         if op.exists(file):
#             T = Table.read(file)
#         else:
#             #yes, there ends up being two underscores before default
#             file = op.join(path,f"{prefix}_default.fits")
#             if op.exists(file):
#                 T = Table.read(file)
#
#         if T is not None:
#             log.info(f"***** Returning aperture sky subtraction residual: {file} [{column}.")
#             return T[column][0]
#         else:
#             log.info(f"***** Unable to find aperture sky subtraction residual: {file} {column}.")
#     except:
#         log.error(f"Exception! Exception loading aperture sky residual for {shotid} + {column}.",exc_info=True)
#         return None
#     return None
#
# def fetch_universal_aperture_sky_subtraction_residual(ffsky=False,hdr=G.HDR_Version,aperture=3.5,):
#     """
#     similar to the single fiber verision, this, instead is for an empty "aperture" at the
#     matched aperture size.
#     For HETDEX this is 3.5"
#     Any other aperture size is invalid unless specifically adapted.
#     This can be applied to a HETDEX lookup spectrum (i.e. from a data release run) or AFTER HETDEX_API get_spectra()
#
#     !!! this is a PSF weighted, DAR corrected, etc spectrum !!!
#
#     :param ffsky:
#     :param hdr:
#     :param aperture: the apeture to lookup
#     :return:
#     """
#
#     try:
#         if G.APPLY_SKY_RESIDUAL_TYPE != 2:
#             return None
#
#         if aperture != 3.5:
#             log.error(f"ERROR! Invalid aperture size ({aperture}) requested for fetch_universal_aperture_sky_subtraction_residual.")
#             return None
#
#
#         log.debug("Loading universal sky residual model...")
#         if G.APERTURE_SKY_RESIDUAL_USE_MODEL:
#             which_col = 1
#         else:
#             which_col = 2
#         if hdr == "3":
#             if ffsky:
#                 if G.APERTURE_SKY_RESIDUAL_HDR3_FF_FLUXD is None:
#                     # try to load it
#                     G.APERTURE_SKY_RESIDUAL_HDR3_FF_FLUXD = np.loadtxt(G.APERTURE_SKY_RESIDUAL_HDR3_FF_FN, usecols=(which_col))
#
#                 log.info(f"***** Returning aperture ff sky subtraction residual ({which_col})")
#                 return G.APERTURE_SKY_RESIDUAL_HDR3_FF_FLUXD
#             else: #local sky
#                 if G.APERTURE_SKY_RESIDUAL_HDR3_LO_FLUXD is None:
#                     # try to load it
#                     G.APERTURE_SKY_RESIDUAL_HDR3_LO_FLUXD = np.loadtxt(G.APERTURE_SKY_RESIDUAL_HDR3_LO_FN, usecols=(which_col))
#
#                 log.info(f"***** Returning aperture local sky subtraction residual ({which_col})")
#                 return G.APERTURE_SKY_RESIDUAL_HDR3_LO_FLUXD
#         else:
#             log.warning(f"Unknown HDR version ({hdr}). No aperture sky residual available.")
#     except:
#         log.error(f"Exception! Exception loading universal aperture sky residual.", exc_info=True)
#         return None
#     log.error(f"No universal aperture sky residual found.", exc_info=True)
#     return None
#
#
#
#
#



def check_overlapping_psf(source_mag,neighbor_mag,psf,dist_baryctr,dist_ellipse=None,effective_radius=None,aperture=1.5):
    """
    Since assuming a point-source, should use the barycenter, but size may come into play for extended objects

    :param source_mag: (assumed to be g, r, or equivalent)
    :param neighbor_mag: (assumed to be g, r, or equivalent)
    :param psf: the PSF (grid) for the shot (see get_psf) [0] is the weight, [1] and [2] are the x and y grids
    :param dist_baryctr: in arcsec
    :param dist_ellipse: in arcsec
    :param effective_radius: in arcsec (rough measure of size ... assumes approximately circular)
    :param aperture: radius in arcsec (over which to integrate the overlapped flux (density))
    :return: fraction of source, overlap as uJy (flux density), and as flat flux
    """

    #we are keeping a single moffat psf, so think of this as being centered on each neighbot (in succession)
    #with the hetdex position (aperture) moving relative to the PSF

    #todo: if dist_baryctr is unknown or if effective radius is large enough to not be a point source ....

    if dist_baryctr and dist_baryctr > 0:
        ctrx = dist_baryctr  #consider the aperture moving to the "right"
    elif dist_ellipse and effective_radius and dist_ellipse > 0 and effective_radius > 0:
        ctrx = dist_ellipse + effective_radius
    else: #not enough info to use
        return -1, -1, -1

    ctry = 0  # just shifting the aperture in x, not up or down

    # build up a distance from the center of the aperture to each poistion in the grid
    dmask = np.sqrt(((psf[1] - ctrx)) ** 2 + ((psf[2] - ctry)) ** 2)
    # then make a mask of same size where True is inside the aperture
    apmask = dmask <= aperture

    nei_ujy = 3631.0 * 1e6 * 10**(-0.4*neighbor_mag) #flux (uJy)
    src_ujy = 3631.0 * 1e6 * 10**(-0.4*source_mag) #flux (uJy)

    # place it in the middle of the hetdex spectrum even though the mag source may well be in an r-band
    # 2.0 * ... so it is in same flux units as HETDEX (and REMEMBER, we are treating this as a flat spectum)
    #nei_flx = 2.0 * ujy2cgs(nei_ujy,G.DEX_G_EFF_LAM)
    #src_flx = 2.0 * ujy2cgs(src_ujy,G.DEX_G_EFF_LAM)

    overlap = np.sum(apmask*psf[0]*nei_ujy)
    fraction_overlap = overlap / src_ujy
    flat_flux = 2.0 * ujy2cgs(overlap,G.DEX_G_EFF_LAM)

    return fraction_overlap, overlap, flat_flux


#todo: is it just the PSF grid or is it the 3xNxN version?
#in which case we want to multiply psf[0]
def psf_by_wave(psf,wave_0,wave):
    """

    !!!NO!!!! it is the FWHM not the PSF function,
    So, technically need a new PSF grid for each wavelength where the FWHM sent in to get_psf is x wave**-0.2

    use the Kolmogorov relation (Roddier 1981 and Karl's paper sec 6.15)
    to adjust the PSF to the wavelength being checked

    :param psf: the baseline PSF
    :param wave_0:  the wavelength at which the baseline PSF was computed
    :param wave: the new wavelength
    :return: updated PSF
    """
    try:
        return psf * (wave/wave_0)**-0.2
    except:
        log.error("Exception! Exception in spectrum_utilities::psf_by_wave().",exc_info=True)
        return None

def get_psf(shot_fwhm,ap_radius,max_sep,scale=0.25,normalize=True):
    """
    Build and return a moffat profile grid
    (unless replaced in the future, this is just a wrappered call to hetdex_api)

    :param shot_fwhm: FWHM (seeing) for the shot
    :param ap_radius: radius of the aperture in which to sum up flux (the central object's aperture)
    :param max_sep: maximum distance (in arcsec) out to which we want to check the PSF overlap
    :param scale: arsec per pixel (sets number of pixels to use in the grid)
    :param normalize: if True, divide the weights by the sum s|t the volume under the PSF is 1.0
    :return: [0] is the weight, [1] and [2] are the x and y grids
    """
    side = 2 * (max_sep + ap_radius)
    psf = Extract().moffat_psf(shot_fwhm,side,scale)

    if normalize:
        psf[0] = psf[0] / np.sum(psf[0])

    return psf

def get_psf_fixed_side(shot_fwhm,ap_radius,side,scale=0.25,normalize=True):
    """
    Build and return a moffat profile grid
    (unless replaced in the future, this is just a wrappered call to hetdex_api)

    :param shot_fwhm: FWHM (seeing) for the shot
    :param ap_radius: radius of the aperture in which to sum up flux (the central object's aperture)
    :param max_sep: maximum distance (in arcsec) out to which we want to check the PSF overlap
    :param scale: arsec per pixel (sets number of pixels to use in the grid)
    :param normalize: if True, divide the weights by the sum s|t the volume under the PSF is 1.0
    :return: [0] is the weight, [1] and [2] are the x and y grids
    """

    psf = Extract().moffat_psf(shot_fwhm,side,scale)

    if normalize:
        psf[0] = psf[0] / np.sum(psf[0])

    return psf

def apply_psf(spec,err,coord,shotid,fwhm,radius=3.0,ffsky=True,hdrversion="hdr2.1"):
    """
    Apply the shot specific PSF to the provided spec and error.

    This is for use in applying the same PSF, weights and mask that was used for a
    matching detection to the sky residuals defined for the shot

    (spec and err assumed to be on 1036 (Calfib) grid)

    :param spec: spectrum (flux or flux_density) values
    :param err: error on the spectrum
    :param fwhm: seeing for the shot
    :return:

    *** RETURN is in the SAME UNITS as passed in  ... so for LyCon, normally passed in as flux over 2AA bins
    *** so it stays that way (erg/s/cm2  x2AA) and is NOT per AA (as would normally be returned from hetdex_api)
    """

    spectrum_conv_psf = None
    error_conv_psf = None
    try:
        E = Extract()

        if radius == None or radius <= 0:
            radius = 3.0

        E.load_shot(shotid, fibers=True, survey=hdrversion)

        if fwhm is None or fwhm <= 0:
            q_shotid = shotid
            fwhm = E.shoth5.root.Shot.read_where("shotid==q_shotid", field="fwhm_virus")[0]


        info_result = E.get_fiberinfo_for_coord(coord, radius=radius, ffsky=ffsky)

        if info_result is not None:
            ifux, ifuy, xc, yc, ra, dec, fiber_data, fiber_error, mask = info_result

            #replace the fiber data and error with arrays of (sky residual) spec and err
            #but use the same mask
            data = np.full(fiber_data.shape,spec)
            error = np.full(fiber_data.shape,err)

            moffat = E.moffat_psf(fwhm, 10.5, 0.25)
            weights = E.build_weights(xc, yc, ifux, ifuy, moffat)
            result = E.get_spectrum(data, error, mask, weights)

            spectrum_conv_psf, error_conv_psf = [res for res in result]

        E.shoth5.close()

    except Exception as e:
        print(e)

    return spectrum_conv_psf, error_conv_psf


def make_raster_grid(ra,dec,width,resolution):
    """
    Make the RA and Dec meshgrid for rastering. Centered on provided ra, dec.
    :param ra: decimal degrees
    :param dec:  decimal degrees
    :param width: arcsec
    :param resolution:arcsec
    :return: ra_meshgrid  (RAs for each Dec) and dec_meshgrid (Decs for each RA)
    """
    try:
        width /= 3600.
        resolution /= 3600.

        # todo note: generally expected to be small enough at low enough DEC that curvature is ignored?
        #  or should I adjust the RA accordingly?

        ra_grid = np.concatenate((np.flip(np.arange(ra-resolution,ra -width-resolution,-1*resolution)),
                                  np.arange(ra, ra + width + resolution, resolution)))

        dec_grid = np.concatenate((np.flip(np.arange(dec-resolution,dec-width-resolution, -1*resolution)),
                                   np.arange(dec, dec + width + resolution, resolution)))
        ra_meshgrid, dec_meshgrid = np.meshgrid(ra_grid, dec_grid)
        return ra_meshgrid, dec_meshgrid
    except:
        log.error("Exception in spectrum_utilities::make_raster_grid",exc_info=True)
        return None, None


def raster_search(ra_meshgrid,dec_meshgrid,shotlist,cw,aperture=3.0,max_velocity=500.0,max_fwhm=15.0,ffsky=False):
    """
    Iterate over the supplied meshgrids and force extract at each point
    :param ra_meshgrid: (see make_raster_grid)
    :param dec_meshgrid:
    :param shotlist: (must be a list, so if just one shot, pass as [shot]
    :param cw: central wavelength (search/extract around this wavelength
    :param aperture: in arcsec
    :param max_velocity: in km/s (max allowance to either side of the specified <cw> to fit)
    :return: meshgrid of (extraction) dictionaries with location and extraction info
    """

    try:
        from elixer import spectrum as SP
    except:
        import spectrum as SP

    try:
        edict = None
        ct = 0
        wct = 0

        #for s in [shotlist[1]]:
        # bw_events = 0
        # max_bw_ct = 0
        # min_bw_ct = 9999
        # wbw_events = 0

        msg = f"Raster extracting emission. RA x Dec x Shots " \
                 f"({np.shape(ra_meshgrid)[1]}x{np.shape(ra_meshgrid)[0]}x{len(shotlist)}) = " \
                 f"{np.shape(ra_meshgrid)[1] * np.shape(ra_meshgrid)[0] * len(shotlist)} extractions."
        log.info(msg)
        print(msg)


        if True: #new way (effectively a single call to hetdex_api get_spectra() with an array of coords)
            #get all extractions for each shot
            all_ra = ra_meshgrid.flatten('F') #to keep the same ordering as before and eliminate the extra transpose
            all_dec = dec_meshgrid.flatten('F')
            #edict = np.zeros_like(ra_meshgrid, dtype=dict)
            #sanity check ...
            # all_ra = []
            # all_dec = []
            # for r in range(np.shape(ra_meshgrid)[1]): #columns (x or RA values)
            #     for d in range(np.shape(ra_meshgrid)[0]): #rows (y or Dec values)
            #         all_ra.append(ra_meshgrid[d,r])
            #         all_dec.append(dec_meshgrid[d,r])
            # all_dec = np.array(all_dec)
            # all_ra = np.array(all_ra)

            all_ex = []
            #grab all RA,Dec for each shot (full grid, but one shot at a time)
            #since we are going to average over all the shots
            for s in shotlist:
                exlist = extract_at_multiple_positions(all_ra,all_dec, aperture, shotid=s,
                                     ffsky=ffsky, multiproc=False)

                all_ex.append(exlist)

            all_ex = np.array(all_ex)

            if len(all_ex) == 0: #failed
                log.error("No extractions.")
                return edict

            #otherwise we iterate over all the shots (all should have the same number of grids)
            #all shots is effectively 3D: shot x RA x Dec with each being an extraction dictionary object
            #though at the moment, since the exlist is flattened, it is just 2D: shot x (RA,DEC)
            #need to average the flux and flux errors (and apcor?) along the shot axis
            elif len(all_ex) > 1:
                exlist = all_ex[0] #choose a baseline
                for i in range(len(exlist)):
                    # now weighted biweight
                    exlist[i]['flux'], exlist[i]['fluxerr'], *_ = stack_spectra( [e['flux'] for e in all_ex[:,i]],
                                                                   [e['fluxerr'] for e in all_ex[:,i]],
                                                                   np.tile(G.CALFIB_WAVEGRID,
                                                                           (len(all_ex), 1)),
                                                                   grid=G.CALFIB_WAVEGRID,
                                                                   avg_type="weighted_biweight",
                                                                   straight_error=True)
                    #not really using this, but shoudl still be weighed biweight ...
                    #use the weights from the flux stack in the same fraction

                    try:
                        frac_errs = np.array([e['fluxerr'] for e in all_ex[:,i]])/np.array([e['flux'] for e in all_ex[:,i]])
                        exlist[i]['apcor'] = weighted_biweight.biweight_location_errors([e['apcor'] for e in all_ex[:,i]],
                                                                                    frac_errs)
                    except:
                        exlist[i]['apcor'] = np.nanmean([e['apcor'] for e in all_ex[:,i]])


            else:
                exlist = all_ex[0]

            #now operate on the collapsed/stacked exlist

            #Turn off extra line scanning and just hit the position specified
            if G.LIMIT_GRIDSEARCH_LINE_FINDER:
                saved_LINE_FINDER_FULL_FIT_SCAN = G.LINE_FINDER_FULL_FIT_SCAN
                saved_LINE_FINDER_MEDIAN_SCAN  = G.LINE_FINDER_MEDIAN_SCAN

                G.LINE_FINDER_FULL_FIT_SCAN = -1
                G.LINE_FINDER_MEDIAN_SCAN = 0

            for idx,ex in enumerate(exlist):
                log.debug(f"gridsearch scanning spectrum #{idx+1} of {len(exlist)} ...")
                #print(f"gridsearch scanning spectrum #{idx + 1} of {len(exlist)} ...")
                try:
                    # now, fit to Gaussian
                    ex['fit'] = combo_fit_wave(SP.peakdet, ex['flux'],
                                               ex['fluxerr'],
                                               ex['wave'],
                                               cw,
                                               wave_slop_kms=max_velocity,
                                               max_fwhm=max_fwhm)

                    # kill off bad fits based on snr, rmse, sigma, continuum
                    # overly? generous sigma ... maybe make similar to original?
                    test_good = False
                    # if (1.0 < ex['fit']['sigma'] < 20.0) and \
                    #         (3.0 < ex['fit']['snr'] < 1000.0):
                    if (0.0 < ex['fit']['sigma'] < 50.0) and \
                            (0.0 < ex['fit']['snr']):
                        if ex['fit']['fitflux'] > 0:
                            # print("Winner")
                            wct += 1
                            test_good = True
                    else:  # is bad, wipe out
                        # print("bad fit")
                        ex['bad_fit'] = copy.copy(ex['fit'])
                        ex['fit']['snr'] = 0
                        ex['fit']['fitflux'] = 0
                        ex['fit']['continuum_level'] = 0
                        ex['fit']['velocity_offset'] = 0
                        ex['fit']['rmse'] = 0
                        ex['fit']['x0'] = 0
                        ex['fit']['sigma'] = 0
                except:
                    log.debug("Exceotion in raster_search.",exc_info=True)
                    if ex['fit'] is None:
                        ex['fit'] = {}
                        ex['fit']['x0'] = None
                        ex['fit']['fitflux'] = 0.0
                        ex['fit']['continuum_level'] = 0.0
                        ex['fit']['velocity_offset'] = 0.0
                        ex['fit']['sigma'] = 0.0
                        ex['fit']['rmse'] = 0.0
                        ex['fit']['snr'] = 0.0
                        ex['fit']['meanflux_density'] = 0.0
                        ex['fit']['velocity_offset_limit'] = max_velocity

            #restore original extra line scanning
            if G.LIMIT_GRIDSEARCH_LINE_FINDER:
                G.LINE_FINDER_FULL_FIT_SCAN = saved_LINE_FINDER_FULL_FIT_SCAN
                G.LINE_FINDER_MEDIAN_SCAN = saved_LINE_FINDER_MEDIAN_SCAN

            #and reshape into the 2D grid
            edict = np.array(exlist).reshape(np.shape(ra_meshgrid),order='F')#.T #transpose looks right but index is off

            #test
            # print(f"***** RA {np.shape(ra_meshgrid)}, Dec {np.shape(dec_meshgrid)}, edict {np.shape(edict)}")
            # for r2,e2 in zip(ra_meshgrid,edict):
            #     for r,e in zip (r2,e2):
            #         if r != e['ra']:
            #             print("!!!! FAIL RA != Edict RA")
            #
            # for d2,e2 in zip(dec_meshgrid,edict):
            #     for d,e in zip (d2,e2):
            #         if d != e['dec']:
            #             print("!!!! FAIL RA != Edict RA")

        else: #old way
            edict = np.transpose(np.zeros_like(ra_meshgrid, dtype=dict))  # need to transpose since addressing as dec,ra
            #Turn off extra line scanning and just hit the position specified
            if G.LIMIT_GRIDSEARCH_LINE_FINDER:
                saved_LINE_FINDER_FULL_FIT_SCAN = G.LINE_FINDER_FULL_FIT_SCAN
                saved_LINE_FINDER_MEDIAN_SCAN  = G.LINE_FINDER_MEDIAN_SCAN

                G.LINE_FINDER_FULL_FIT_SCAN = -1
                G.LINE_FINDER_MEDIAN_SCAN = 0

            tot = np.shape(ra_meshgrid)[1] * np.shape(ra_meshgrid)[0]
            for r in range(np.shape(ra_meshgrid)[1]): #columns (x or RA values)
                for d in range(np.shape(ra_meshgrid)[0]): #rows (y or Dec values)
                    exlist = []
                    for s in shotlist:
                        #print("Shot = ", s)
                        ct += 1
                        log.debug(f"Extracting {ct} of {tot}")
                        #print(ct) ra,dec,aperture,shotid,ffsky=False,multiproc=G.GET_SPECTRA_MULTIPROCESS
                        ex = extract_at_position(ra_meshgrid[d,r], dec_meshgrid[d,r], aperture,shotid=s,
                                                 ffsky=ffsky,multiproc=False)

                        if ex['flux'] is None:
                            continue

                        exlist.append(ex)

                    if len(exlist) == 0:
                        continue
                    elif len(exlist) == 1:
                        avg_f = exlist[0]['flux']
                        avg_fe = exlist[0]['fluxerr']

                    elif len(exlist) > 1:
                        #make a new ex dictionary with the weighted biweight average of the data and run simple fit
                        flux = []
                        fluxe = []
                        #waves are all the same
                        for ex in exlist:
                            flux.append(ex['flux'])
                            fluxe.append(ex['fluxerr'])

                        flux = np.array(flux)
                        fluxe = np.array(fluxe)

                        #now weighted biweight
                        avg_f  = np.zeros(len(flux[0]))
                        avg_fe = np.zeros(len(flux[0]))


                        for i in range(len(flux[0])):

                            wslice_err = np.array([m[i] for m in fluxe])  # all rows, ith index
                            wslice = np.array([m[i] for m in flux])  # all rows, ith index

                            wslice_err = wslice_err[np.where(wslice != -999)]  # so same as wslice, since these need to line up
                            wslice = wslice[np.where(wslice != -999)]  # git rid of the out of range interp values

                            # git rid of any nans
                            wslice_err = wslice_err[~np.isnan(wslice)]  # err first so the wslice is not modified
                            wslice = wslice[~np.isnan(wslice)]

                            try:
                                f = weighted_biweight.biweight_location_errors(wslice, errors=wslice_err)
                                #not really using the errors, but this is good enough?
                                fe = weighted_biweight.biweight_scale(wslice) / np.sqrt(len(wslice))
                                #wbw_events += 1
                            except:
                                #log.info("Weighted_biweight failed. Switching to normal biweight")
                                f = weighted_biweight.biweight_location(wslice)
                                fe = weighted_biweight.biweight_scale(wslice) /np.sqrt(len(wslice))
                                #bw_events += 1

                            if np.isnan(f):
                                avg_f[i] = 0
                                avg_fe[i] = 0
                            elif np.isnan(fe):
                                avg_f[i] = f
                                avg_fe[i] = 0
                            else:
                                avg_f[i] = f
                                avg_fe[i] = fe
                    else:
                        #something very wrong
                        #print("********** unexpectedly zero spectra???????")
                        continue
                    #end if

                    #now, fit to Gaussian
                    ex['fit'] = combo_fit_wave(SP.peakdet,avg_f,
                                                   avg_fe,
                                                   exlist[0]['wave'],
                                                   cw,
                                                   wave_slop_kms=max_velocity,
                                                   max_fwhm=max_fwhm)

                    # kill off bad fits based on snr, rmse, sigma, continuum
                    # overly? generous sigma ... maybe make similar to original?
                    test_good = False
                    # if (1.0 < ex['fit']['sigma'] < 20.0) and \
                    #         (3.0 < ex['fit']['snr'] < 1000.0):
                    #     if ex['fit']['fitflux'] > 0:
                    if (0.0 < ex['fit']['sigma'] < 50.0) and \
                                    (0.0 < ex['fit']['snr']):
                        if ex['fit']['fitflux'] > 0:
                            #print("Winner")
                            wct += 1
                            test_good = True
                    else:  # is bad, wipe out
                        #print("bad fit")
                        ex['bad_fit'] = copy.copy(ex['fit'])
                        ex['fit']['snr'] = 0
                        ex['fit']['fitflux'] = 0
                        ex['fit']['continuum_level'] = 0
                        ex['fit']['velocity_offset'] = 0
                        ex['fit']['rmse'] = 0
                        ex['fit']['x0'] = 0
                        ex['fit']['sigma'] = 0

                    edict[r, d] = ex

                    #todo: REMOVE ME
                    if False: #extra debugging
                        print("**********Remove me**********")
                        plt.close('all')
                        plt.figure(figsize=(6,3))

                        plt.axvline(cw,color='k',alpha=0.5)
                        plt.axvline(cw-cw*max_velocity/3e5,ls="dashed",color='k',alpha=0.5)
                        plt.axvline(cw+cw*max_velocity/3e5,ls="dashed",color='k',alpha=0.5)
                        try:
                            plt.axvline(ex['fit']['x0'],color="r")
                        except:
                            pass

                        plt.plot(G.CALFIB_WAVEGRID, avg_f)
                        plt.xlim(cw-2*cw*max_velocity/3e5,cw+2*cw*max_velocity/3e5)

                        plt.title(f"Good {test_good}: r({r}) d({d}) RA: {ra_meshgrid[d,r]:0.6f}  Dec: {dec_meshgrid[d,r]:0.6f}\n"
                                  f"fwhm = {ex['fit']['sigma']*2.355:0.2f}, flux = {ex['fit']['fitflux']:0.2f}, snr = {ex['fit']['snr']:0.2f}")
                        plt.tight_layout()
                        plt.savefig(f"ip_r{str(r).zfill(3)}_d{str(d).zfill(3)}.png")


            log.info(f"Raster 'good' emission fits ({wct}) / ({ct})")

            #restore original extra line scanning
            if G.LIMIT_GRIDSEARCH_LINE_FINDER:
                G.LINE_FINDER_FULL_FIT_SCAN = saved_LINE_FINDER_FULL_FIT_SCAN
                G.LINE_FINDER_MEDIAN_SCAN = saved_LINE_FINDER_MEDIAN_SCAN

        #print("Good fits:", wct)
        #print(f"Biweights: max_ct ({max_bw_ct}), min_ct ({min_bw_ct}), wbw ({wbw_events}), bw ({bw_events})")

        return edict
    except:
        log.error("Exception in spectrum_utilities::raster_search", exc_info=True)
        return None



def make_raster_plots(dict_meshgrid,ra_meshgrid,dec_meshgrid,cw,key,colormap=cm.coolwarm,save=None,savepy=None,show=False):
    """
    Make 3 plots (an interactive (if show is true) 3D plot, a contour plot and a color meshplot (like a heatmap)
    :param dict_meshgrid: meshgrid of dictionaries for the RA, Dec
    :param ra_meshgrid:
    :param dec_meshgrid:
    :param cw: central wavelength
    :param key: one of "fitflux" (integrated line flux), "f900", "velocity_offset", "continuum_level"
    :param colormap: matplotlib color map to use for scale
    ##:param lyc: if True, assume the cw is LyA and produce an extra fitflux plot around rest-900AA
    :param save: filename base to save the plots (the files as saved as <save>_3d.png, <save>_contour.png, <save>_mesh.png
    :param savepy: save a pickle and a python file to load it and launche interactive plots
    :param show: if True, display the plots interactively
    :return: returns a mesh grid of the values selected by <key>
    """

    #3D mesh apperently does not support masked arrays correctly, so cannot mask bad values
    # colomap.set_bad(color=[0.2, 1.0, 0.23]) #green

    try:
        #set RA, DEC as central position of the meshgrids
        r,c = np.shape(ra_meshgrid)
        RA = ra_meshgrid[0,(c-1)//2]

        r, c = np.shape(dec_meshgrid)
        DEC = dec_meshgrid[(r-1)//2,0]

        plt.close('all')
        old_backend = None
        if show:
            try:
                old_backend = plt.get_backend()
                plt.switch_backend('TkAgg')
            except:
                log.error("Cannot run interactive plot ...", exc_info=True)
                try:
                    if old_backend:
                        plt.switch_backend(old_backend)
                except:
                    pass
        ########################
        #as a contour
        ########################

        #z for integrated flux
        #colomap.set_bad(color=[0.2, 1.0, 0.23]) #green
        colormap.set_bad(color=[1.0, 1.0, 1.0]) #white
        bad_value = -1e9
        z = np.full(ra_meshgrid.shape,bad_value,dtype=float)
        #does not have to be loops, but this is easy to read
        #could instead be:
        # z = np.array([ [xy['fit'][key] for xy in x] for x in dict_meshgrid]).reshape(np.shape(dict_meshgrid))
        # but then also need to find the "bad fit" locations and mask those or set to -1
        # since there are rarely more than a few hundred (at most) grid points, this is plenty fast,
        # and rarely used, so just keep it easy to read

        for r in range(np.shape(ra_meshgrid)[1]):  # columns (x or RA values)
            for d in range(np.shape(ra_meshgrid)[0]):  # rows (y or Dec values)
                try:
                    if key != 'fitflux' and 'bad_fit' in dict_meshgrid[d, r].keys():
                        pass # no fit, assume same as "no-data"
                    else:
                        z[d,r] = dict_meshgrid[d, r]['fit'][key]
                except:
                    pass  # nothing there, so leave the 'bad value' in place

        z = np.ma.masked_where(z == bad_value, z)

        try:
            max_aa_offset = dict_meshgrid[0][0]['fit']['velocity_offset_limit'] #in km/s
            max_aa_offset = cw * max_aa_offset / (astropy.constants.c.to('km/s')).value
        except:
           max_aa_offset = None
           # max_aa_offset = cw * 1500.00 / (astropy.constants.c.to('km/s')).value

        fig = plt.figure()
        ax = plt.gca()
        num_levels = np.max(ra_meshgrid.shape)
        levels = MaxNLocator(nbins=num_levels).tick_values(z.min(), z.max())
        #surf = ax.contourf((dec_meshgrid-DEC)*3600.0,(ra_meshgrid-RA)*3600.0, z, cmap=colormap, levels=levels)
        surf = ax.contourf((ra_meshgrid-RA)*3600.0,(dec_meshgrid-DEC)*3600.0, z, cmap=colormap, levels=levels)
        #surf = ax.contourf(ra_meshgrid, dec_meshgrid, z, cmap=colormap, levels=levels)
        ax.invert_xaxis()

        #get the x-range, then reverse so East is to the left
        #xlim = ax.get_xlim()
        #ax.set_xlim(xlim[1],xlim[0])

        if max_aa_offset is not None:
            info_title = fr"$\alpha$ ({RA:0.5f}) $\delta$ ({DEC:0.5f}) $\lambda$ ({cw:0.2f} +/- {max_aa_offset:0.2f})"
        else:
            info_title = fr"$\alpha$ ({RA:0.5f}) $\delta$ ({DEC:0.5f}) $\lambda$ ({cw:0.2f})"

        ax.set_title(info_title)
        ax.set_xlabel(r'$\Delta$RA"')
        ax.set_ylabel(r'$\Delta$Dec"')
        fig.colorbar(surf, shrink=0.5, aspect=5,label=key)

        if save:
            plt.savefig(save +  "_" + key + "_contour.png")

        if show:
            fig.show()


        ######################
        # as a color mesh
        ######################
        fig = plt.figure()
        ax = plt.gca()
        #notice here, ra & dec are in the other order vs 3D plot
       #surf = ax.pcolormesh((dec_meshgrid-DEC)*3600.0, (ra_meshgrid-RA)*3600.0, z, cmap=colormap)
        surf = ax.pcolormesh((ra_meshgrid-RA)*3600.0, (dec_meshgrid-DEC)*3600.0, z, cmap=colormap)
        #get the x-range, then reverse so East is to the left
        # xlim = ax.get_xlim()
        # ax.set_xlim(xlim[1],xlim[0])
        ax.invert_xaxis()

        ax.set_title(info_title)
        ax.set_xlabel(r'$\Delta$RA"')
        ax.set_ylabel(r'$\Delta$Dec"')
        fig.colorbar(surf, shrink=0.5, aspect=5,label=key)

        if save:
            plt.savefig(save +  "_" + key + "_mesh.png")

        if show:
            fig.show()

        ##########################
        # as interactive 3D
        # do this one last so can interact and see the other two
        ##########################
        bad_value = 0
        z = np.full(ra_meshgrid.shape, bad_value, dtype=float)
        for r in range(np.shape(ra_meshgrid)[1]):  # columns (x or RA values)
            for d in range(np.shape(ra_meshgrid)[0]):  # rows (y or Dec values)
                try:
                    z[d, r] = dict_meshgrid[d, r]['fit'][key]
                except:
                    z[d, r] = 0  # 3D plot does not handle masked arrays #nothing there, so leave the 'bad value' in place

        # 3D Plot (interactive)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # because this plots as X,Y,Z, and we're using a meshgrid, the x-coords (the RA's are in the dec_meshgrid,
        # that is, all the RAs  (X's) for each given Dec)
        # in otherwords, though, the RAs are values of X coords, but first parameter are the ROWS (row-major)
        # which are along the vertical (each row contains the list of all RAs (x's) for that row
        #surf = ax.plot_surface((dec_meshgrid - DEC) * 3600.0, (ra_meshgrid - RA) * 3600.0, z, cmap=colormap,
        #                       linewidth=0, antialiased=False)
        surf = ax.plot_surface((ra_meshgrid - RA) * 3600.0, (dec_meshgrid - DEC) * 3600.0, z, cmap=colormap,
                               linewidth=0, antialiased=False)

        # get the x-range, then reverse so East is to the left
        # xlim = ax.get_xlim()
        # ax.set_xlim(xlim[1], xlim[0])
        ax.invert_xaxis()

        ax.set_title(info_title)
        ax.set_xlabel(r'$\Delta$RA"')
        ax.set_ylabel(r'$\Delta$Dec"')
        ax.set_zlabel(key)
        fig.colorbar(surf, shrink=0.5, aspect=5, label=key)

        if save is not None:  # interactive does not work
            # pickle.dump(plt.gcf(),open(save+".fig.pickle","wb"))
            plt.savefig(save + "_" + key + "_3d.png")

        if show:
            plt.show()

        #restore the original backend
        if old_backend:
            plt.switch_backend(old_backend)


        if savepy is not None:
            try:
                #save a pickle of the parameters and a python file to run?
                parms_dict = {}

                #make these small?? (drop the details?)
                # cp_dict = copy.copy(dict_meshgrid)
                # for c in np.ndarray.flatten(cp_dict):
                #     try:
                #         c['flux'] =[]
                #         c['fluxerr'] = []
                #         c['wave'] = []
                #     except: #entry might be empty (0) and not hold a dictionary
                #         pass
                # parms_dict['dict_meshgrid'] = cp_dict

                #or dump the whole data (keeping the spectra at each point?) could be usefull?
                parms_dict['dict_meshgrid'] = dict_meshgrid

                parms_dict['ra_meshgrid'] = ra_meshgrid
                parms_dict['dec_meshgrid'] = dec_meshgrid
                parms_dict['cw'] = cw
                parms_dict['colormap'] = colormap

                fn = savepy + "_grid"
                #subsequent calls to other plots will overwrite the pickle, but it is the same data
                pickle.dump(parms_dict,open(fn+".pickle","wb"))

                pyfilestr = "from elixer import spectrum_utilities as SU\n"
                pyfilestr += "import pickle\n"
                pyfilestr += "import sys\n"
                pyfilestr += "import numpy as np\n"
                pyfilestr += "args = list(map(str.lower, sys.argv))\n"
                pyfilestr += f"parms_dict = pickle.load(open('{op.basename(fn)+'.pickle'}','rb'))\n"
                pyfilestr += "data = np.ndarray.flatten(parms_dict['dict_meshgrid'])\n"
                pyfilestr += "idx = np.where(data != 0)[0]\n"
                pyfilestr += "if (idx is None) or len(idx)==0:\n"
                pyfilestr += "  print('No emission line was able to be fit with provided parameters.')\n"
                pyfilestr += "  exit(0)\n"
                pyfilestr += "try:\n"
                pyfilestr += "    allkeys = data[idx[0]]['fit'].keys()\n"
                pyfilestr += "except:\n"
                pyfilestr += "    allkeys = data[idx[0]]['bad_fit'].keys()\n"
                pyfilestr += "if (len(args) > 1):\n"
                pyfilestr += "    if args[1] in allkeys:\n"
                pyfilestr += "        key = args[1] \n"
                pyfilestr += "    else:\n"
                pyfilestr += "        print(f'Available keys to plot: {allkeys}')\n"
                pyfilestr += "        exit(0)\n"
                pyfilestr += "else:\n"
                pyfilestr += "    print('Assuming meanflux_density plot')\n"
                pyfilestr += "    print(f'Available keys to plot: {allkeys}')\n"
                pyfilestr += "    key = 'meanflux_density'\n"
                pyfilestr += "SU.make_raster_plots(parms_dict['dict_meshgrid']," \
                             "parms_dict['ra_meshgrid']," \
                             "parms_dict['dec_meshgrid']," \
                             "parms_dict['cw']," \
                             "key," \
                             "parms_dict['colormap']," \
                             "show=True,save=None,savepy=None)\n"


                with open(fn+".py","w") as f:
                    f.write(pyfilestr)
            except Exception as e:
                print("Unable to save interactive py plots",e)

        return z
    except:
        log.error("Exception in spectrum_utilities::make_raster_plots",exc_info=True)
        return None

def get_shotids(ra,dec,hdr_version=G.HDR_Version):
    """

    :param ra: decimal degrees
    :param dec: decimal degrees
    :param hdr_version: 1,2, etc. default is the current ELiXer data release default
    :return: list of integer shotids that overlap the provided ra and dec
    """


    try:
        if G.the_Survey is None:
            G.the_Survey = hda_survey.Survey(survey=f"hdr{hdr_version}")

        if not G.the_Survey:
            log.info(f"Cannot build hetdex_api survey object to determine shotid")
            return []

        # this is only looking at the pointing, not checking individual fibers, so
        # need to give it a big radius to search that covers the focal plane
        # if centered (and it should be) no ifu edge is more than 12 acrmin away
        shotlist = G.the_Survey.get_shotlist(SkyCoord(ra, dec, unit='deg', frame='icrs'),
                                           radius=G.FOV_RADIUS_DEGREE * U.deg)

        log.debug(f"{len(shotlist)} shots near ({ra},{dec})")

        return shotlist
    except:
        log.error("Exception in spectrum_utilities::get_shotids", exc_info=True)
        return []

####################################
# example of using raster search
###################################
# from elixer import spectrum_utilities as SU
# RA = 214.629608
# DEC  =  52.544327
# cw = 3819.0
# shotlist = SU.get_shotids(RA,DEC)
# ra_meshgrid, dec_meshgrid = SU.make_raster_grid(RA,DEC,3.0,0.4)
# edict = SU.raster_search(ra_meshgrid,dec_meshgrid,shotlist,cw,max_velocity=2000,aperture=3.0)
# z = SU.make_raster_plots(edict,ra_meshgrid,dec_meshgrid,cw,"fitflux",show=True)
# notice:  in the above call, to save static images, specify a filename base in the "save" parameter
#          or for interactive images, in the "savepy" parameter



def patch_holes_in_hetdex_spectrum(wavelengths,flux,flux_err,mag,mag_err=0,filter='g',nan_okay=200):
    """

    note: for stacking, we want to keep NaNs so they are skipped

    :param wavelengths:
    :param flux:
    :param flux_err:
    :param mag:
    :param mag_err:
    :param filter:
    :param nan_okay: if up to this number of NaN's are okay, then return with them, otherwise fill in with an interpolation
    :return:
    """


    try:
        #(mag,filter='g',waves=None)
        if type(filter) == bytes:
            _filter = filter.decode()
        else:
            _filter = filter
        patched_flux = copy.copy(flux)
        patched_flux_err = copy.copy(flux_err)

        #want positions where flux is NaN or None OR flux_err is NaN or None or zero
        patched_flux[patched_flux==None] = np.nan #so patched flux can have NaNs but no None
        patched_flux_err[patched_flux_err==None] = 0#np.nan
        patched_flux_err[patched_flux_err==0] = np.nan

        sel1 = np.isnan(patched_flux_err)
        sel2 = np.isnan(patched_flux)
        sel = sel1 | sel2

        if np.sum(sel)<= nan_okay:
            return patched_flux,patched_flux_err

        #otherwise need to substitute
        flat_flux = make_fnu_flat_spectrum(mag,_filter,wavelengths) * G.FLUX_WAVEBIN_WIDTH / G.HETDEX_FLUX_BASE_CGS
        if mag_err is not None and mag_err != 0:
            flat_flux_hi = make_fnu_flat_spectrum(mag-mag_err,_filter,wavelengths) * G.FLUX_WAVEBIN_WIDTH / G.HETDEX_FLUX_BASE_CGS
            flat_flux_lo = make_fnu_flat_spectrum(mag+mag_err,_filter,wavelengths) * G.FLUX_WAVEBIN_WIDTH / G.HETDEX_FLUX_BASE_CGS
            flat_flux_err = 0.5*(flat_flux_hi-flat_flux_lo)
        else:
            #flat_flux_err = np.zeros(len(wavelengths)).astype(float)
            #assume 25%
            flat_flux_err = flat_flux*0.25

        patched_flux[sel] = flat_flux[sel]
        patched_flux_err[sel] = flat_flux_err[sel]

        return patched_flux,patched_flux_err
    except:
        log.error("Exception! Exception in patch_holes_in_hetdex_spectrum.",exc_info=True)



# this is probably unnecessary. Given the MC sampling on each individual deblend and then the later stacking of
# many thousands of spectra, any weird, deeply negative neighbor spectra than increases the HETDEX spectrum for a
# wavelength bin will be averaged out by the stacking
#
# def clear_negative_fluxes_in_spectrum(wavelengths,flux,flux_err,flux_limits=None):
#     """
#     If flux_limits is specified, set any wavelength bin flux below the flux_limit to zero,
#     otherwise if it is less than zero, set to zero.
#
#     NOTICE: the errors are not directly included since the callers do not directly use the errors on an individual deblend call,
#     instead, the errors are used in an MC .... the flux is nudged by sampling over the reported errors and the deblend
#     is called many times and an average taken
#
#     Instead we use the errors to inform on the zero set. If the flux is less than zero AND
#
#     SO... this func should be called AFTER each sampling and BEFORE the deblend call
#
#     :param wavelengths:
#     :param flux:
#     :param flux_err:
#     :param flux_limits:
#     :return: flux
#     """
#
#     try:
#         if flux_limits is None or len(flux_limits) == 0:
#             flux_limits = np.zeros(len(wavelengths))
#
#         sel = np.array(flux) < np.array(flux_limits)
#         #sel2 = np.array(flux) < 2.0*np.array(flux_err)
#         #sel = sel & sel2
#         flux[sel] = 0.0
#     except:
#         log.error("Exception! Exception in spectrum_utilities::clear_negative_fluxes_in_spectrum().",exc_info=True)
#     return flux

def norm_overlapping_psf(psf,dist_baryctr,dist_ellipse=None,effective_radius=None):
    """

    Return the fraction of PSF weight "volume" overlapping between to sources separated by dist_baryctr with the
    same PSF (same shot).

    THIS IS NOT the actual contribution of FLUX from a contaminant. THIS IS JUST the weight volume in the region
    of intersection between two identical, spatially shifted PSF models.

    Since assuming a point-source, should use the barycenter, but size may come into play for extended objects

    :param psf: the PSF (grid) for the shot (see get_psf) [0] is the weight, [1] and [2] are the x and y grids
    :param dist_baryctr: in arcsec
    :param dist_ellipse: in arcsec
    :param effective_radius: in arcsec (rough measure of size ... assumes approximately circular)
    :return: weight volume (or fraction) in overlap region
    """

    #we are keeping a single moffat psf, so think of this as being centered on each neighbot (in succession)
    #with the hetdex position (aperture) moving relative to the PSF

    #??: if dist_baryctr is unknown or if effective radius is large enough to not be a point source ....

    if dist_baryctr and dist_baryctr > 0:
        ctrx = dist_baryctr  #consider the aperture moving to the "right"
    elif dist_ellipse and effective_radius and dist_ellipse > 0 and effective_radius > 0:
        ctrx = dist_ellipse + effective_radius
    elif dist_baryctr is None or dist_baryctr == 0:
        ctrx = 0
    else: #not enough info to use
        return -1.0

    #think of this as two identical PSF grids, with one shifted to the right by ctrx
    #where the vacated grid cells from the right shfited ctrx get a zero weight

    #this is the distance between PSF centers (ie. between the two object centers) in grid units
    #again, since the PSF are symmetric about their center, it does not matter on which axis(es) the
    #distance is applied, so to keep it simple, just visualize as a shift along the x-direction to +x
    x_shift = int(ctrx/(psf[1][1][1]-psf[1][1][0]))

    # psf2 = copy.copy(psf[0])
    # psf2 = np.roll(psf2,x_shift,1)
    # psf2[:,0:x_shift+1] = 0 #zero out the cells on the right that rolled around
    #
    # overlap = np.sum(np.minimum(psf2,psf[0])) #this would be the overlap region
    #
    # #overlap = np.sum(psf2*psf[0]) #this would be the sum of the fractions of the psf weights from one object
    #                               #included in the other by its own weights
    #                               #i.e. if in the 0.02 weight section of the target we have the 0.03 weight section
    #                               # of the neighbor, then that overlap section contrinbutes 0.02 * 0.03 = 0.0006 weight
    #                               # to the target from the neighbor (that space or fiber has 2% of the target's flux
    #                               # and 3% of the neighbors flux, so we subtract 0.06% of the True neighhor flux for that
    #                               # section). Sum up all the sections (here, pixels of the moffat) to get the total
    #                               # contribution of the neighbor to the target).
    #
    vol = np.sum(psf[0])

    psf2 = copy.copy(psf[0]/vol) #need to normalize the psf weights since we want the fractional overalp
    psf1 = copy.copy(psf2)
    psf2 = np.roll(psf2,x_shift,1)
    psf2[:,0:x_shift+1] = 0 #zero out the cells on the right that rolled around

    overlap = np.sum(np.minimum(psf2,psf1))

    return overlap


def build_separation_matrix(ra,dec):
    """
    Takes a list of RA and Dec (in the same order as the spectra matrix, etc) with the LAE candidate at index 0
    :param ra:
    :param dec:
    :return: square separation matrix and a code
    """
    try:
        row = len(ra)
        col = row
        if (row<2):
            log.error(f"Error in build_separation_matrix. Too few objects ({row},{col})")
            return None, 0

        #not ideal since looping, but norm_overlapping_psf is not written with matrices
        sep_matrix = np.zeros((row,col))
        #just need the lower triangle
        for c in np.arange(0,col):
            for r in np.arange(c+1,row):
                s = utils.angular_distance(ra[r],dec[r],ra[c],dec[c])
                sep_matrix[r,c] = s
                sep_matrix[c,r] = s #redundant, but just in case

        return sep_matrix, 0
    except:
        log.error("Exception! Exception in spectrum_utilities.build_separation_matrix().",exc_info=True)

    return None, -1

def psf_overlap(psf,separation_matrix):
    """

    :param psf: the PSF model for this shot i.e. from get_psf()
    :param separation_matrix: pair wise separations between sources in arcsec
    :return:  pair-wise fractional overalp of PSF
    """

    try:
        row,col = np.shape(separation_matrix) #has to be square
        if row != col or (row<2):
            log.error(f"Error in psf_overlap. separation_matrix not square ({row},{col})")
            return None

        #not ideal since looping, but norm_overlapping_psf is not written with matrices
        overlap_matrix = np.ones((row,col))
        #just need the lower triangle
        for c in np.arange(0,col):
            for r in np.arange(c+1,row):
                f = norm_overlapping_psf(psf,separation_matrix[r,c])
                #need f**2 since the overlap is symmetric and weighted by each others PSF overlap
                #(note at full overlap f==1 --> 1**2 == 1)
                overlap_matrix[r,c] = f #*f
                overlap_matrix[c,r] = overlap_matrix[r,c] #redundant, but just in case

        return overlap_matrix
    except:
        log.error("Exception! Exception in spectrum_utilities.psf_overlap().",exc_info=True)

    return None

def integrate_fiber_over_psf(fwhm,fiber_flux,fiber_flux_err):
    """
    Take the spectrum from a single fiber as representative of a uniform background and integrate that background over
    the shot seeing PSF

    No extra DAR, etc applied as this represents as uniform background.

    This uses the same fiber weighting as a normal 3-dither extraction, though there is a very small variation since
    we are not using the exact same fiber positions as whatever produced the fiber_flux and fiber_flux_err. If the
    centroid is in the exact center of the "blue" fiber which is at the exact center of the moffat PSF model, (and the
    same psf_width and step are used) then that small variation should vanish.

    :param fwhm: shot seeing fwhm
    :param fiber_flux:
    :param fiber_flux_err:
    :return: "PSF" weighted background spectrum (flux and flux error)
    """

    try:
        aperture_method='exact' #'exact' 'center' 'subpixel'
        fiber_radius = 0.75 #in arcsecs
        psf_width = 10.5 #in arcsec ... only compute out to here. Never truly goes to zero, but rapidly asymptotes
        step = 0.25 #in arcsec to generate moffat PSF

        E = Extract()
        moffat = E.moffat_psf(fwhm, psf_width, step)
        w,x,y = np.shape(moffat)
        x -= 1 #size to max index
        y -= 1
        # (0,0) is in lower left corner (or upper left ... does not matter) x/2, y/2 is the center
        pix_aperture = CircularAperture((x/2.0, y/2.0), r=fiber_radius/step) #in pixel space
        phot_table = aperture_photometry(moffat[0], pix_aperture, method=aperture_method)
        counts = phot_table['aperture_sum'][0]

        #treat the weights of the moffat like a height, so the volume is the area of the fiber x "height"
        #since the moffat is normalized to a "volume" of 1, this is vol is the fractional volume covered by the
        #fiber footprint
        vol = counts*np.pi*fiber_radius**2

        #weights assuming fibers starting at the center of the moffat
        weights = E.build_weights(0, 0, [0], [0], moffat)[0]
        scale = np.max(weights) #i.e. the peak of the moffat (or norm s|t the moffat would integrate to 1)

        #since we need to virtually "fill" the moffat with fibers we need a correction for the missing area between 3 dithers
        #(ie. area between 3 fiber circles: the equalateral triange area - 3*sigle sector area)
        #or 1/2 b*h - 3* 1/6 * pi* r**2  where the 60deg angle of the sector is 1/6 of the 360 deg circle
        # or 1/2 (2r * root(3)*r) - 1/2 pi r**2) == (r**2) * (root(3)-pi/2)
        missing = (fiber_radius**2)*(np.sqrt(3.0)-np.pi/2.0)

        #now "integrate" over the moffat by fitting the weight volume occupied by the fiber footprint at the center
        #taking out the correction for the missing area footprints between the fibers in a perfect dither sampling
        int_flux = fiber_flux/scale/vol*(1.0-missing)
        int_flux_err = fiber_flux_err/scale/vol*(1.0-missing)

        return int_flux, int_flux_err
    except:
        log.error("Exception! Exception in spectrum_utilities.psf_overlap().",exc_info=True)

    return None, None

def spectra_deblend(measured_flux_matrix, overlap_tensor):
    """
    Single iteration call. Expected that the caller will resample measured_flux_matrix from the uncertainties and call
    this function repeatedly.

    NEW: 2023-04-13 overlap_matrix is now overlap_tendor where each pair of objects has a full length "spectrum" of
         overlap fractions

    :param measured_flux_matrix: each row is a mesured spectrum (or from flat in fnu) (columns are the wavebins)
    #:param overlap_matrix: pair-wise PSF overlaps; row or column corresponses to the row index in measured_flux_matrix
    :param overlap_tensor: pair-wise PSF overlaps; row or column corresponses to the row index in measured_flux_matrix
                          and the 3rd dimension is the "spectrum" of overlap fractions
    :return: true (deblended) flux matrix (same shape as measured_flux_matrix)
    """

    try:
        row,col = np.shape(measured_flux_matrix) #rows are the sources, columns are the measured fluxes
        true_flux_matrix = np.full((row,col),np.nan) #not strictly "True" flux, just estimated deblend, but keeps the paper naming convention


        #N = np.shape(overlap_tensor)[0]
        #overlap_matrix = np.ones((N,N))
        #each c is wavelength
        for c in range(col):
        #    for i in np.arange(1,N,1):
        #        for j in np.arange(0,i):
        #            overlap_matrix[i,j] = overlap_tensor[i][j][c]
        #            overlap_matrix[j,i] = overlap_matrix[i,j]

            true_flux_matrix[:,c] = np.linalg.solve(overlap_tensor[:,:,c],measured_flux_matrix[:,c])
           # true_flux_matrix[:,c] = np.linalg.solve(overlap_matrix,measured_flux_matrix[:,c])

            #debug
            # flux_vector =  measured_flux_matrix[:,c] #a slice down the flux matrix, all rows, column = c
            # solve = np.linalg.solve(overlap_matrix,flux_vector)
            #
            # #print(solve)
            # true_flux_matrix[:,c] = solve

        return true_flux_matrix
    except:
        log.error("Exception! Exception in spectrum_utilities.spectra_deblend().",exc_info=True)

    return None


##################################
# Cloned from LyCon project
##################################


def shift_flam_to_rest_luminosity_per_aa(z,flux_density,wave,eflux=None,apply_air_to_vac=False):
    """
    !!! ??? this is REALLY a luminosity "density" analogous to flux density, so Lum/AA not Lum/volume ??? !!!

    Assume z, wavelengths, and luminosity distances are without error

    :param z:
    :param flux_density:
    :param wave:
    :param eflux:
    :param apply_air_to_vac: if true, apply the air to vacuum correction on the observed spectrum before redshifting
    :param per_aa: if true (default)
    :return: luminosity/AA , rest wavelengths , luminosity_error/AA
    """

    if z < -0.001:
        log.error(f"What? invalid z: {z}")
        return None, None, None
    elif z < 0: #close enough to zero that we will call it zero
        z = 0

    if ( (flux_density is None) or (wave is None)) or (len(flux_density) != len(wave)) or (len(flux_density) == 0):
        log.error("Invalid flux_density and wavelengths")
        return None, None, None

    try:

        if apply_air_to_vac:
            wave = air_to_vac(wave)

        wave = wave / (1.0 + z)
        ld = luminosity_distance(z).to(U.cm) #this is in Mpc

        try:
            _ = flux_density.value
        except:
            #flux density does not have a units, so strip units from ld
            ld = ld.value

        conv = 4.0 * np.pi * ld * ld * (1.0 + z) # extra (1+z) is to deal with the input being a flux density
        #e.g. for a flux density erg/s/cm2/AA need 3 factors of (1+z). for erg,s, and AA.
        #The ld*ld takes care of the cm2
        #The ld is co-moving * (1+z) so there are 2 or the 3 factors and we need that one more.
        #What you get back, then is erg/s/AA or a kind of luminosity density !!!*** (but usually that term
        #   means luminosity per volume, like Lum/Mpc3).
        #Passing in erg/s/cm2/AA in 2AA bins for z = 3 gives back erg/s/AA restframe in 0.5AA bins
        #So the Luminosity in a bin then is that Lum/AA * 0.5AA
        #BUT when fitting a Gaussin, we are integrating over the Lum/AA s|t it is essentially Lum/AA * width (in AA) = integrated Lum
        lum = flux_density * conv
        if (eflux is not None) and (len(eflux) == len(wave)):
            lum_err = eflux * conv
        else:
            lum_err =None

        return lum, wave , lum_err

    except Exception as e:
        print(e)

    return None, None, None


def shift_flam_to_rest_luminosity(z,flux_density,wave,eflux=None,apply_air_to_vac=False):
    """
    Takes flux density (flam) and returns luminosity (NOT luminosity per AA)

    Assume z, wavelengths, and luminosity distances are without error

    :param z:
    :param flux_density:
    :param wave:
    :param eflux:
    :param apply_air_to_vac: if true, apply the air to vacuum correction on the observed spectrum before redshifting
    :param per_aa: if true (default)
    :return: luminosity/AA , rest wavelengths , luminosity_error/AA
    """

    if z < -0.001:
        log.error(f"What? invalid z: {z}")
        return None, None, None
    elif z < 0: #close enough to zero that we will call it zero
        z = 0

    if ( (flux_density is None) or (wave is None)) or (len(flux_density) != len(wave)) or (len(flux_density) == 0):
        log.error("Invalid flux_density and wavelengths")
        return None, None, None

    try:

        if apply_air_to_vac:
            wave = air_to_vac(wave)

        wave = wave / (1.0 + z)
        ld = luminosity_distance(z).to(U.cm) #this is in Mpc

        try:
            _ = flux_density.value
        except:
            #flux density does not have a units, so strip units from ld
            ld = ld.value

        conv = 4.0 * np.pi * ld * ld * (1.0 + z) # extra (1+z) is to deal with the input being a flux density
        #e.g. for a flux density erg/s/cm2/AA need 3 factors of (1+z). for erg,s, and AA.
        #The ld*ld takes care of the cm2
        #The ld is co-moving * (1+z) so there are 2 or the 3 factors and we need that one more.
        #What you get back, then is erg/s/AA or a kind of luminosity density !!!*** (but usually that term
        #   means luminosity per volume, like Lum/Mpc3).
        #Passing in erg/s/cm2/AA in 2AA bins for z = 3 gives back erg/s/AA restframe in 0.5AA bins
        #So the Luminosity in a bin then is that Lum/AA * 0.5AA
        #BUT when fitting a Gaussin, we are integrating over the Lum/AA s|t it is essentially Lum/AA * width (in AA) = integrated Lum
        lum = flux_density * conv * (wave[1]-wave[0])
        if (eflux is not None) and (len(eflux) == len(wave)):
            lum_err = eflux * conv * (wave[1]-wave[0])
        else:
            lum_err =None

        return lum, wave , lum_err

    except Exception as e:
        print(e)

    return None, None, None


def shift_flux_to_rest_luminosity(z,flux,wave,eflux=None,apply_air_to_vac=False):
    """
    Assume z, wavelengths, and luminosity distances are without error

    :param z:
    :param flux: FLUX in the bin NOT flux density
    :param wave:
    :param eflux:
    :param apply_air_to_vac: if true, apply the air to vacuum correction on the observed spectrum before redshifting
    :param per_aa: if true (default)
    :return: luminosity/AA , rest wavelengths , luminosity_error/AA
    """

    if z < -0.001:
        log.error(f"What? invalid z: {z}")
        return None, None, None
    elif z < 0: #close enough to zero that we will call it zero
        z = 0

    if ( (flux is None) or (wave is None)) or (len(flux) != len(wave)) or (len(flux) == 0):
        log.error("Invalid flux and wavelengths")
        return None, None, None

    try:

        if apply_air_to_vac:
            wave = air_to_vac(wave)

        wave = wave / (1.0 + z)
        ld = luminosity_distance(z).to(U.cm) #this is in Mpc

        try:
            _ = flux.value
        except:
            #flux does not have a units, so strip units from ld
            ld = ld.value

        conv = 4.0 * np.pi * ld * ld #* (1.0 + z) # NO extra (1+z) here since this is a FLUX not a flux density
        lum = flux * conv
        if (eflux is not None) and (len(eflux) == len(wave)):
            lum_err = eflux * conv
        else:
            lum_err =None

        return lum, wave , lum_err

    except Exception as e:
        print(e)

    return None, None, None


def rest_line_luminosity(z,line_flux,line_flux_err=None):
    """

    Assume z, wavelengths, and luminosity distances are without error

    :param z:
    :param line_flux: in erg/s/cm2
    :param line_flux_err: in erg/s/cm2
    :return:
    """

    # if z < 0.0:
    #     print("What? invalid z: ", z)
    #     return None, None
    #
    # if (line_flux is None) :
    #     print("Invalid line_flux")
    #     return None, None

    lum = np.nan
    lum_err = np.nan
    try:
        ld = luminosity_distance(z)
        factor = (4.0 * np.pi * ld * ld ).to(U.cm**2) #no DO NOT have (1+z) in there ... this is line flux, not a density
        #and the 1+z factors are handled in the Luminosity Distance

        units = None
        try:
            units = lum.unit
            if units != (U.erg / (U.s * U.cm**2)):
                if units != U.dimensionless_unscaled:
                    log.error(f"Invalid line flux units {units}")
                else:
                    units = None
        except:
            pass

        if units is None:
            lum = line_flux * factor.value
        else:
            lum = line_flux * factor

        if line_flux_err is not None:
            if units is None:
                lum_err = line_flux_err * factor.value
            else:
                lum_err = line_flux_err * factor

    except:
        pass

    return lum, lum_err



def shift_to_rest_flam(z, flux_density, wave, eflux=None, block_skylines=True, apply_air_to_vac=False):
    ### WARNING !!! REMINDER !!! np.arrays are mutable so what are passed in here get modified (flux_density, etc)

    # !!! This is semi-stupid ... only applies (1+z)**3 to flux!!! You may want to use shift_to_rest_luminosity

    """
    Really, the "observed" flux density in the reference frame of the rest-frame

    We are assuming no (or insignificant) error in wavelength and z ...

    All three numpy arrays must be of the same length and same indexing
    :param z:
    :param flux_density: numpy array of flux densities ...  erg s^-1 cm^-2 AA^-1
    :param wave: numpy array of wavelenths ... units don't matter
    :param eflux:  numpy array of flux densities errors ... again, units don't matter, but same as the flux
                   (note: the is NOT the wavelengths of the flux error)
    :param apply_air_to_vac: if true, apply the air to vacuum correction on the observed spectrum before redshifting
    :return:
    """

    #todo: how are the wavelength bins defined? edges? center of bin?
    #that is; 4500.0 AA bin ... is that 4490.0 - 4451.0  or 4500.00 - 4502.0 .....

    if z < 0.0:
        log.error(f"What? invalid z: {z}")
        flux_density *= 0.0 #zero out the flux, something seriously wrong
        return flux_density, wave, eflux

    # if block_skylines:
    #     flux_density[np.where((3535 < wave)&(wave < 3555))] = float("nan")

    try:
        if apply_air_to_vac:
            wave = air_to_vac(wave)
        #rescale the wavelengths
        wave /= (1.+z) #shift to lower z and squeeze bins

        flux_density *= ((1.+z) * (1.+z) * (1.+z)) #shift up flux (shorter wavelength = higher energy)
        #sanity check .... "density" goes up by (1+z) because of smaller wavebin, energy goes up by (1+z) because of shorter wavelength
    except:
        pass

    # if block_skylines:
    #     flux_density[np.where((3535 < wave)&(wave < 3555))] = float("nan")


    #todo: deal with wavelength bin compression (change in counting error?)

    #todo: deal with the various errors (that might not have been passed in)
    # if eflux is None or len(eflux) == 0:
    #     eflux = np.zeros(np.shape(flux))
    #
    # if ewave is None or len(ewave) == 0:
    #     ewave = np.zeros(np.shape(flux))

    #todo: deal with luminosity distance (undo dimming?) and turn into luminosity??
    #
    #lum = flux*4.*np.pi*(luminosity_distance(z))**2
    #
    # or just boost the flux to zero distance point source (don't include the 4*pi)
    #
    #flux = flux * (luminosity_distance(z))**2

    return flux_density, wave, eflux


def make_grid(all_waves):
    """
    Takes in all the wavelength arrays to figure the best grid so that
    the interpolation only happens on a single grid

    The grid is on the step size of the spectrum with the smallest steps and the range is
    between the shortest and longest wavelengths that are in all spectra.

    :param all_waves: 2d array of all wavelengths
    :return: grid (1d array) of wavelengths

    """

    all_waves = np.array(all_waves)
    #set the range to the maximum(minimum) to the minimum(maximum)
    mn = np.max(np.amin(all_waves,axis=1)) #maximum of the minimums of each row
    mx = np.min(np.amax(all_waves,axis=1)) #minimum of the maximums of each row

    # the step is the smallest of the step sizes
    # assume (within each wavelength array) the stepsize is uniform
    step = np.min(all_waves[:,1] - all_waves[:,0])

    #return the grid
    return np.arange(mn, mx + step, step)


def make_grid_max_length(all_waves):
    """
    Takes in all the wavelength arrays to figure the best grid so that
    the interpolation only happens on a single grid

    The grid is on the step size of the spectrum with the smallest steps and the range is
    the maximum of ALL spectral ranges (NOT THE MAX OVERLAP, but the absolute max)

    :param all_waves: 2d array of all wavelengths
    :return: grid (1d array) of wavelengths

    """

    #all_waves = np.array(all_waves)
    # matrix_waves = np.vstack(all_waves) #can only vstack if lengths are the same
    # #set the range to the maximum(minimum) to the minimum(maximum)
    # mn = np.min(np.amin(matrix_waves,axis=1)) #minimum of the minimums of each row
    # mx = np.max(np.amax(matrix_waves,axis=1)) #maximums of the maximums of each row
    # # the step is the smallest of the step sizes
    # # assume (within each wavelength array) the stepsize is uniform
    # step = np.min(matrix_waves[:,1] - matrix_waves[:,0])

    mn = min(x[0] for x in all_waves)
    mx = max(x[-1] for x in all_waves)
    step = min(x[1]-x[0] for x in all_waves)

    #return the grid
    return np.arange(mn, mx + step, step)

def stack_spectra(fluxes,flux_errs,waves, grid=None, avg_type="weighted_biweight",straight_error=False):
    """
        Assumes all spectra are in the same frame (ie. all in restframe) and ignores the flux type, but
        assumes all in the same units and scale (for flux, flux err and wave).

    flux, flux_err, and wave are 2D matrices (array of arrays)

    :param fluxes: should be unitless values
    :param flux_errs:  should be unitless values
    :param waves:  should be unitless values
    :param avg_type:
    :param straight_error:
    :return:
    """

    data_shape = np.shape(fluxes)

    if (np.shape(flux_errs) != np.shape(waves)) or (np.shape(flux_errs) != data_shape):
        log.error("Inconsistent data shape for fluxes, flux_errs, and waves")
        return None,None,None,None

    if grid is None or len(grid) == 0:
        grid = make_grid_max_length(waves) #full width grid of absolute maximum spread in wavelengths

    resampled_fluxes = []
    resampled_flux_errs = []
    contrib_count = np.zeros(len(grid)) #number of spectra contributing to this wavelength bin




    stack_flux = np.full(len(grid),np.nan)
    stack_flux_err = np.full(len(grid),np.nan)
    #stack_wave # this is just the grid

    #resample all input fluxes onto the same grid
    for i in range(data_shape[0]):
        res_flux, res_err, res_wave = interpolate(fluxes[i],waves[i],grid,eflux=flux_errs[i])

        min_wave = waves[i][0]
        max_wave = waves[i][-1]

        res_flux[np.where(grid < min_wave)] = np.nan
        res_flux[np.where(grid > max_wave)] = np.nan


        resampled_fluxes.append(res_flux) #aka interp_flux_density_matrix
        resampled_flux_errs.append(res_err) #aka  interp_flux_density_err_matrix
        #all on the same wave grid, so don't need that back

    #########################################################################
    # average down each wavebin THEN across the (single) averaged spectrum
    #########################################################################

    #save the units

    for i in range(len(grid)):
        #build error first since wslice is modified and re-assigned
        #build the error on the flux (wslice) so they stay lined up

        #have to remove the quantity here, just use the float values
        wslice_err = np.array([m[i] for m in resampled_flux_errs])  # interp_flux_matrix[:, i]  # all rows, ith index
        wslice = np.array([m[i] for m in resampled_fluxes]) #interp_flux_matrix[:, i]  # all rows, ith index

        #since using np.nan and np.nan != np.nan, so where wslice == itself it is not nan
        wslice_err = wslice_err[np.where(wslice==wslice)]  # so same as wslice, since these need to line up
        wslice = wslice[np.where(wslice==wslice)]  # git rid of the out of range interp values

        # git rid of any nans
        wslice_err = wslice_err[~np.isnan(wslice)] #err first so the wslice is not modified
        wslice = wslice[~np.isnan(wslice)]

        contrib_count[i] = len(wslice)

        if len(wslice) == 0:
            stack_flux[i] = np.nan
            stack_flux_err[i] = np.nan
        elif len(wslice) == 1:
            try:
                stack_flux[i] = wslice[0]
                stack_flux_err[i] = 0
            except Exception as e:
                print(e)
                stack_flux[i] = 0
                stack_flux_err[i] = 0
        elif len(wslice) == 2:
            try:
                stack_flux[i] = np.nanmean(wslice)
                stack_flux_err[i] = np.nanstd(wslice)
            except Exception as e:
                print(e)
                stack_flux[i] = 0
                stack_flux_err[i] = 0
        else: #3 or more
            if straight_error or (avg_type == 'mean_95'):
                try:
                    mean_cntr, var_cntr, std_cntr = bayes_mvs(wslice, alpha=0.95)
                    if np.isnan(mean_cntr[0]):
                        raise( Exception('mean_ctr is nan'))
                    stack_flux[i] = mean_cntr[0]
                    #an average error
                    stack_flux_err[i] = 0.5 * (abs(mean_cntr[0]-mean_cntr[1][0]) + abs(mean_cntr[0]-mean_cntr[1][1]))
                except Exception as e:
                    log.error("Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..." %(i,grid[i]),e)
                    try:
                        stack_flux[i] = biweight.biweight_location(wslice)
                        stack_flux_err[i] = biweight.biweight_scale(wslice) #* 2. #2 sigma ~ 95%
                    except Exception as e:
                        log.error(e)
            elif straight_error or (avg_type == 'mean_68'):
                try:
                    mean_cntr, var_cntr, std_cntr = bayes_mvs(wslice, alpha=0.68)
                    if np.isnan(mean_cntr[0]):
                        raise (Exception('mean_ctr is nan'))
                    stack_flux[i] = mean_cntr[0]
                    # an average error
                    stack_flux_err[i] = 0.5 * (
                            abs(mean_cntr[0] - mean_cntr[1][0]) + abs(mean_cntr[0] - mean_cntr[1][1]))
                except Exception as e:
                    log.error("Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..." % (
                        i, grid[i]), e)

                    try:
                        stack_flux[i] = biweight.biweight_location(wslice)
                        stack_flux_err[i] = biweight.biweight_scale(wslice)   # * 2. #2 sigma ~ 95%
                    except Exception as e:
                        log.error(e)
            elif (avg_type == 'mean_std'):
                try:
                    stack_flux[i] = np.nanmean(wslice)
                    # an average error
                    stack_flux_err[i] = np.nanstd(wslice) / np.sqrt(len(wslice))
                except Exception as e:
                    log.error("Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..." %(i,grid[i]),e)
                    try:
                        stack_flux[i] = biweight.biweight_location(wslice)
                        stack_flux_err[i] = biweight.biweight_scale(wslice) #* 2. #2 sigma ~ 95%
                    except Exception as e:
                        log.error(e)

            elif (avg_type == 'median'):
                try:
                    stack_flux[i] = np.nanmedian(wslice)
                    #an average error
                    stack_flux_err[i] =np.nanstd(wslice)/np.sqrt(len(wslice))
                except Exception as e:
                    log.error("Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..." %(i,grid[i]),e)

                    try:
                        stack_flux[i] = biweight.biweight_location(wslice)
                        stack_flux_err[i] = biweight.biweight_scale(wslice)#* 2. #2 sigma ~ 95%
                    except Exception as e:
                        log.error(e)

            elif (avg_type == 'biweight'):
                try:
                    stack_flux[i] = biweight.biweight_location(wslice)
                    stack_flux_err[i] = biweight.biweight_scale(wslice) / np.sqrt(len(wslice))
                except Exception as e:
                    log.error("Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..." %(i,grid[i]),e)
                    try:
                        stack_flux[i] = biweight.biweight_location(wslice)
                        stack_flux_err[i] = biweight/biweight_scale(wslice) #* 2. #2 sigma ~ 95%
                    except Exception as e:
                        log.error(e)
            else: #weighted_biweight
                #definitely keep the scale defaults (c=6,c=9) per Karl, etc .. these best wieght for Gaussian limits

                try:
                    stack_flux[i] = weighted_biweight.biweight_location_errors(wslice, errors=wslice_err)
                    stack_flux_err[i] = weighted_biweight.biweight_scale(wslice) / np.sqrt(len(wslice))
                except Exception as e:
                    log.error(e)
                    stack_flux[i] = biweight.biweight_location(wslice)
                    stack_flux_err[i] = biweight.biweight_scale(wslice) / np.sqrt(len(wslice))

        #end loop

    return stack_flux,stack_flux_err,grid,contrib_count


#
# taken with slight modification from CIGALE
# https://github.com/JohannesBuchner/cigale

def igm_transmission(wavelength, redshift):
    """Intergalactic transmission (Meiksin, 2006)

    Compute the intergalactic transmission as described in Meiksin, 2006.

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength(s) in AA. **** OBSERVED FRAME ****
    redshift: float
        The redshift. Must be strictly positive.

    Returns
    -------
    igm_transmission: numpy array of floats
        The intergalactic transmission at each input wavelength.

    """
    wavelength = np.array(copy.copy(wavelength)) / 10.0 #convert to nm

    n_transitions_low = 10
    n_transitions_max = 31
    gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    lambda_limit = 91.2  # Lyman limit in nm

    lambda_n = np.empty(n_transitions_max)
    z_n = np.empty((n_transitions_max, len(wavelength)))
    for n in range(2, n_transitions_max):
        lambda_n[n] = lambda_limit / (1. - 1. / float(n*n))
        z_n[n, :] = (wavelength / lambda_n[n]) - 1.

    # From Table 1 in Meiksin (2006), only n >= 3 are relevant.
    # fact has a length equal to n_transitions_low.
    fact = np.array([1., 1., 1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373,
                     0.0283])

    # First, tau_alpha is the mean Lyman alpha transmitted flux,
    # Here n = 2 => tau_2 = tau_alpha
    tau_n = np.zeros((n_transitions_max, len(wavelength)))
    if redshift <= 4:
        tau_a = 0.00211 * np.power(1. + redshift,  3.7)
        tau_n[2, :] = 0.00211 * np.power(1. + z_n[2, :], 3.7)
    elif redshift > 4:
        tau_a = 0.00058 * np.power(1. + redshift,  4.5)
        tau_n[2, :] = 0.00058 * np.power(1. + z_n[2, :], 4.5)

    # Then, tau_n is the mean optical depth value for transitions
    # n = 3 - 9 -> 1
    for n in range(3, n_transitions_max):
        if n <= 5:
            w = np.where(z_n[n, :] < 3)
            tau_n[n, w] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, w]), (1. / 3.)))
            w = np.where(z_n[n, :] >= 3)
            tau_n[n, w] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, w]), (1. / 6.)))
        elif 5 < n <= 9:
            tau_n[n, :] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, :]), (1. / 3.)))
        else:
            tau_n[n, :] = (tau_n[9, :] * 720. /
                           (float(n) * (float(n*n - 1.))))

    for n in range(2, n_transitions_max):
        w = np.where(z_n[n, :] >= redshift)
        tau_n[n, w] = 0.

    z_l = wavelength / lambda_limit - 1.
    w = np.where(z_l < redshift)

    tau_l_igm = np.zeros_like(wavelength)
    tau_l_igm[w] = (0.805 * np.power(1. + z_l[w], 3) *
                    (1. / (1. + z_l[w]) - 1. / (1. + redshift)))

    term1 = gamma - np.exp(-1.)

    n = np.arange(n_transitions_low - 1)
    term2 = np.sum(np.power(-1., n) / (factorial(n) * (2*n - 1)))

    term3 = ((1.+redshift) * np.power(wavelength[w]/lambda_limit, 1.5) -
             np.power(wavelength[w]/lambda_limit, 2.5))

    term4 = np.sum(np.array(
        [((2.*np.power(-1., n) / (factorial(n) * ((6*n - 5)*(2*n - 1)))) *
          ((1.+redshift) ** (2.5-(3 * n)) *
           (wavelength[w]/lambda_limit) ** (3*n) -
           (wavelength[w]/lambda_limit) ** 2.5))
         for n in np.arange(1, n_transitions_low)]), axis=0)

    tau_l_lls = np.zeros_like(wavelength)
    tau_l_lls[w] = n0 * ((term1 - term2) * term3 - term4)

    tau_taun = np.sum(tau_n[2:n_transitions_max, :], axis=0)

    lambda_min_igm = (1+redshift)*70.
    w = np.where(wavelength < lambda_min_igm)

    weight = np.ones_like(wavelength)
    weight[w] = np.power(wavelength[w]/lambda_min_igm, 2.)
    # Another weight using erf function can be used.
    # However, you would need to add: from scipy.special import erf
    # weight[w] = 0.5*(1.+erf(0.05*(wavelength[w]-lambda_min_igm)))

    return np.exp(-tau_taun-tau_l_igm-tau_l_lls) * weight


###############################################
# cloned from original LyCon paper / project
###############################################



def standardize_data_type(mdata):
    """

    :param mdata:
    :return: a quantity if possible, or Quantity array (rather than list or array of quantities)
    """

    if mdata is None:
        return None

    data_len = None
    data_unit = None
    data_type = type(mdata)

    if data_type == u.quantity.Quantity: #already a quantity (so not a list), could be a Quantity array
        return mdata
    #else could be a simple type, or a list or array (multi-dimensional) of quantities, etc

    try:
        shape = dshape(mdata)
    except:
        shape = None

    if (shape is not None) and (len(shape) > 1):
        data_list = []
        #take 1st (remaning) dimension and iterate over it
        for i in range(shape[0]):
            data_list.append(standardize_data_type(mdata[i]))
        mdata = np.array(data_list)
    else:

        try:
            data_len = len(mdata) #then this is a list or an array of some kind
            data_unit = mdata[0].unit #this could also fail, if, say a list of floats, since float has no .unit
        except:
            if data_len is None: #if this was a non array type
                try:
                    data_unit = mdata.unit #again, could fail if non quantity type
                except:
                    pass

        if data_unit is None:
            if data_len is None:
                return mdata #we're done, nothing to do
            else:
                return np.array(mdata) #return as an array
        else:
            if data_len is None: #single quantity, so, we're done
                return mdata
            else: #list or array of individual quantities
                if data_type == u.quantity.Quantity: #already a Quantity array
                    return mdata
                else:
                    mdata = np.array([d.value for d in mdata])
                    mdata *= data_unit
                    return mdata

    return mdata


def dvalue(data):
    """
    Strip any units (if present) and return the value
    :param data:
    :return:
    """
    data = standardize_data_type(data)

    if type(data) == u.quantity.Quantity:
        return data.value
    else:
        return data


def bin_fixed_width(flux,wave,num_wavebins,err=None,median_filter=False,gaussian=0,match_idx=0): #assumes a regular input wavegrid
    """
    Implicit assumption that the bins are all the same width (i.e. all 2AA or all 0.44 AA, etc)

    Takes FLUXES but returns FLUX DENSITIES

    :param flux: ***flux*** of single 1D spectra
    :param wave: wavelengths associated with flux
    :param num_wavebins: width in number (integer only!!!) of wavebins overwhich to bin

    :return: binned_flux, binned_waves (center of bin), binned_widths (in wavelengths)
    """

    #width = how many adjacent wavebins to combine

    if median_filter:
        if (num_wavebins > 1) and (num_wavebins % 2):
            binned_flux = medfilt(flux,num_wavebins)*flux[0].unit
            binned_wave = wave #no change, since this is a median filter
            binned_widths = np.full(len(flux),num_wavebins)
            if err:
                binned_err = medfilt(err,num_wavebins) *err[0].unit
            else:
                binned_err = np.zeros(len(flux))*flux[0].unit

            return standardize_data_type(binned_flux), standardize_data_type(binned_wave), standardize_data_type(binned_widths),\
               standardize_data_type(binned_err)

    gaussian_binned_flux = None
    if gaussian:
        if (num_wavebins > 1) and (num_wavebins % 2):
            gaussian_binned_flux = gaussian_filter1d(flux,gaussian)*flux[0].unit
            binned_wave = wave #no change, since this is a median filter
            binned_widths = np.full(len(flux),num_wavebins)
            if err:
                binned_err = gaussian_filter1d(err,gaussian) / (gaussian) * err[0].unit # np.sqrt(2.355*gaussian)  *err[0].unit
            else:
                binned_err = np.zeros(len(flux))*flux[0].unit

            return standardize_data_type(gaussian_binned_flux), \
                   standardize_data_type(binned_wave), \
                   standardize_data_type(binned_widths),\
                   standardize_data_type(binned_err)

    #print(f"***** bin_fixed_witdh() units: flux {flux.unit}  wave {wave.unit}")

    binned_flux = []
    binned_wave = []  #in AA
    binned_widths = [] #in delta_AA
    binned_err = []

    #certainly a more pythonic way to do this, but I'm tired and there are not that many to loop over
    #base = (match_idx+1) % num_wavebins
    base = (match_idx+num_wavebins//2+1) % num_wavebins
    while (base + num_wavebins) < len(wave):
        binned_widths.append(wave[base + num_wavebins] - wave[base])  # technically should take all differences and sum, but okay for fixed
        #print(f"***** binned_widths[-1].units {binned_widths[-1].unit}")
        binned_flux.append(np.sum(flux[base:base+num_wavebins]) / binned_widths[-1])
        #print(f"***** binned_flux[-1].units {binned_flux[-1].unit}")
        if err is not None: #remember errors of sum, sum in quadrature
            binned_err.append(np.sqrt(np.sum(err[base:base + num_wavebins]**2)) / binned_widths[-1])
        binned_wave.append(0.5*(wave[base]+wave[base+num_wavebins-1])) #assign the middle wavelength as the wavelength value


        base += num_wavebins

    #add the last one (may be short)
    if base < len(wave):
        binned_widths.append((len(wave[base:]))*(wave[1]-wave[0])) #can't grab the next wavebin off the end,
        #so need to add 1 and multiply by the fixed wavegrid passed in

        binned_flux.append(np.sum(flux[base:])/binned_widths[-1])
        if err is not None:  #remember errors of sum, sum in quadrature
            binned_err.append(np.sqrt(np.sum(err[base:]**2))/binned_widths[-1])
        binned_wave.append(0.5 * (wave[base] + wave[-1]))

    #print(f"***** binned_flux unit {binned_flux[0].unit}")
    if gaussian_binned_flux is not None:
        unit = binned_flux[0].unit
        return gaussian_binned_flux*unit, \
               standardize_data_type(binned_wave), \
               standardize_data_type(binned_widths),\
               standardize_data_type(binned_err)
    else:

        return standardize_data_type(binned_flux), \
               standardize_data_type(binned_wave), \
               standardize_data_type(binned_widths),\
               standardize_data_type(binned_err)


def bin_num_bins(flux,wave,num_bins): #assumes a regular input wavegrid
    """

    :param flux: flux (NOT flux density) of single 1D spectra
    :param wave: wavelengths associated with flux
    :param num_bins: how many bins (of equal width)
    :return: binned_flux, binned_waves (center of bin), binned_widths (in wavelengths)
    """
    #send in flux NOT a flux density
    #width = how many adjacent wavebins to combine


    binned_flux = []
    binned_wave = []
    binned_widths = []


    #certainly a more pythonic way to do this, but I'm tired and there are not that many to loop over
    binned_flux, wave_edges, bin_idx = scpbin(wave,flux,statistic='sum',bins=num_bins)
    binned_wave, wave_edges, bin_idx = scpbin(wave,wave,statistic='mean',bins=num_bins)

    #binned_wave = [0.5 * (wave_edges[i] + wave_edges[i+1]) for i in range(len(binned_flux))]
    binned_widths = [(wave_edges[i+1] - wave_edges[i]) for i in range(len(binned_flux))]

    return np.array(binned_flux), np.array(binned_wave), np.array(binned_widths)


#from __future__ import print_function, division, absolute_import

####################################################################################
# taken frome SpectRes
# https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
# Adam Carnall
# BUT this is effectively no different than what I am already doing with interpolation
#
###################################################################################

def make_bins(wavs):
    """ Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins. """

    try:
        edges = np.zeros(wavs.shape[0]+1)
        widths = np.zeros(wavs.shape[0])
        edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
        widths[-1] = (wavs[-1] - wavs[-2])
        edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
        edges[1:-1] = (wavs[1:] + wavs[:-1])/2
        widths[:-1] = edges[1:-1] - edges[:-2]

        return edges, widths
    except:
        log.warning("Exception! Exception in [spectres] spectrum_utilities::make_bins()",exc_info=True)
        return [],[]


def spect_rebin(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None,
             verbose=False):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.
    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.
    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.
    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.

    try:
        old_wavs = spec_wavs
        old_fluxes = spec_fluxes
        old_errs = spec_errs

        # Make arrays of edge positions and widths for the old and new bins

        old_edges, old_widths = make_bins(old_wavs)
        new_edges, new_widths = make_bins(new_wavs)

        # Generate output arrays to be populated
        new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

        if old_errs is not None:
            if old_errs.shape != old_fluxes.shape:
                raise ValueError("If specified, spec_errs must be the same shape "
                                 "as spec_fluxes.")
            else:
                new_errs = np.copy(new_fluxes)

        start = 0
        stop = 0
        warned = False

        # Calculate new flux and uncertainty values, looping over new bins
        for j in range(new_wavs.shape[0]):

            # Add filler values if new_wavs extends outside of spec_wavs
            if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
                new_fluxes[..., j] = fill

                if spec_errs is not None:
                    new_errs[..., j] = fill

                if (j == 0 or j == new_wavs.shape[0]-1) and verbose and not warned:
                    warned = True
                    print("\nSpectres: new_wavs contains values outside the range "
                          "in spec_wavs, new_fluxes and new_errs will be filled "
                          "with the value set in the 'fill' keyword argument. \n")
                continue

            # Find first old bin which is partially covered by the new bin
            while old_edges[start+1] <= new_edges[j]:
                start += 1

            # Find last old bin which is partially covered by the new bin
            while old_edges[stop+1] < new_edges[j+1]:
                stop += 1

            # If new bin is fully inside an old bin start and stop are equal
            if stop == start:
                new_fluxes[..., j] = old_fluxes[..., start]
                if old_errs is not None:
                    new_errs[..., j] = old_errs[..., start]

            # Otherwise multiply the first and last old bin widths by P_ij
            else:
                start_factor = ((old_edges[start+1] - new_edges[j])
                                / (old_edges[start+1] - old_edges[start]))

                end_factor = ((new_edges[j+1] - old_edges[stop])
                              / (old_edges[stop+1] - old_edges[stop]))

                old_widths[start] *= start_factor
                old_widths[stop] *= end_factor

                # Populate new_fluxes spectrum and uncertainty arrays
                f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
                new_fluxes[..., j] = np.sum(f_widths, axis=-1)
                new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

                if old_errs is not None:
                    e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                    new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                    new_errs[..., j] /= np.sum(old_widths[start:stop+1])

                # Put back the old bin widths to their initial values
                old_widths[start] /= start_factor
                old_widths[stop] /= end_factor

        # If errors were supplied return both new_fluxes and new_errs.
        if old_errs is not None:
            return new_fluxes, new_errs

        # Otherwise just return the new_fluxes spectrum array
        else:
            return new_fluxes
    except:
        log.warning("Exception! Exception in [spectres] spectrum_utilities::make_bins()",exc_info=True)
        if spec_errs is not None:
            return [],[]
        else:
            return []


#############################################
# end spectres
#############################################

try:
    if G.CALFIB_WAVEGRID_VAC is None:
        G.CALFIB_WAVEGRID_VAC = air_to_vac(G.CALFIB_WAVEGRID)
except:
    log.error("Unable to set G.CALFIB_WAVEGRID_VAC")