try:
    from elixer import global_config as G
    from elixer import line_prob
    from elixer import mcmc_gauss
    from elixer import mcmc_double_gauss
    from elixer import spectrum_utilities as SU
    from elixer import utilities as utils
    from elixer import weighted_biweight as weighted_biweight
except:
    import global_config as G
    import line_prob
    import mcmc_gauss
    import mcmc_double_gauss
    import spectrum_utilities as SU
    import utilities as utils
    import weighted_biweight as weighted_biweight

import matplotlib
#matplotlib.use('agg')

import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
#import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import io
#from scipy.stats import gmean
#from scipy import signal
#from scipy.integrate import simps
from scipy.stats import skew, kurtosis,chisquare
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import astropy.stats.biweight as biweight
import copy

import os.path as op
from speclite import filters as speclite_filters
from astropy import units as units

#log = G.logging.getLogger('spectrum_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('spectrum_logger')
log.setlevel(G.LOG_LEVEL)

#these are for the older peak finder (based on direction change)
MIN_FWHM = 2.0 #AA (must xlat to pixels) (really too small to be realistic, but is a floor)
MAX_FWHM = 40.0 #big LyA are around 15-16; booming can get into the 20s, real lines can be larger,
                # but tend to be not what we are looking for
                # and these are more likly continuum between two abosrpotion features that is mistaken for a line
                #AGN seen with almost 25AA with CIV and NV around 35AA
MAX_NORMAL_FWHM = 20.0 #above this, need some extra info to accept
MIN_HUGE_FWHM_SNR = 19.0 #if the FWHM is above the MAX_NORMAL_FWHM, then then SNR needs to be above this value
                        #would say "20.0 AA" but want some room for error
MIN_ELI_SNR = 3.0 #bare minium SNR to even remotely consider a signal as real
MIN_ELI_SIGMA = 1.0 #bare minium (expect this to be more like 2+)
MIN_HEIGHT = 10
MIN_DELTA_HEIGHT = 2 #to be a peak, must be at least this high above next adjacent point to the left
DEFAULT_BACKGROUND = 6.0
DEFAULT_BACKGROUND_WIDTH = 100.0 #pixels
DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND = 10.0 #pixels


#note: these are not the same usage as the global_config LIMIT_GAUSS_FIT_SIGMA_MIN and LIMIT_GAUSS_FIT_SIGMA_MAX
#which can override these later
GAUSS_FIT_MAX_SIGMA = 17.0 #maximum width (pixels) for fit gaussian to signal (greater than this, is really not a fit)
GAUSS_FIT_MIN_SIGMA = 1.0 #roughly 1/2 pixel where pixel = 1.9AA (#note: "GOOD_MIN_SIGMA" below provides post
                          # check and is more strict) ... allowed to fit a more narrow sigma, but will be rejected later
                          # as not a good signal .... should these actually be the same??

#depending on when this is included, the command line may not yet have been parsed, so
#you need to be aware and may need to call update_from_globals() later, after the command line is parsed
if G.LIMIT_GAUSS_FIT_SIGMA_MAX is not None:
    GAUSS_FIT_MAX_SIGMA = G.LIMIT_GAUSS_FIT_SIGMA_MAX

if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None:
    GAUSS_FIT_MIN_SIGMA = G.LIMIT_GAUSS_FIT_SIGMA_MIN

GAUSS_FIT_AA_RANGE = 50.0 #AA to either side of the line center to include in the gaussian fit attempt
                          #a bit of an art here; too wide and the general noise and continuum kills the fit (too wide)
                          #too narrow and fit hard to find if the line center is off by more than about 2 AA
                          #20 works pretty well for faint lines
                          #40 seems (based on testing) to be about right (50 leaves too many clear, but weak signals behind)
                          #50 is the HETDEX default, though
GAUSS_FIT_PIX_ERROR = 4.0 #error (freedom) in pixels (usually  wavebins): have to allow at least 2 pixels of error
                          # (if aa/pix is small, _AA_ERROR takes over)
GAUSS_FIT_AA_ERROR = 1.0 #error (freedom) in line center in AA, still considered to be okay
GAUSS_SNR_SIGMA = 4.0 #check at least these pixels (pix*sigma) to either side of the fit line for SNR
                      # (larger of this or GAUSS_SNR_NUM_AA) *note: also capped to a max of 40AA or so (the size of the
                      # 'cutout' of the signal (i.e. GAUSS_FIT_AA_RANGE)
GAUSS_SNR_NUM_AA = 5.0 #check at least this num of AA to either side (2x +1 total) of the fit line for SNR in gaussian fit
                       # (larger of this or GAUSS_SNR_SIGMA


SKY_LINES_DICT = {3545:10.0,5462:5.0} #dictionary key = skyline peak, value = half-width, so 3545 +/- 10.0 AA

#copied from manual run of 100,000 noise spectra (see exp_prob.py)
#if change the noise model or SNR or line_flux algorithm or "EmissionLineInfo::is_good", need to recompute these
#as MDF
PROB_NOISE_LINE_SCORE = \
[  2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,   9.5,  10.5,
        11.5,  12.5,  13.5,  14.5,  15.5]

#PROB_NOISE_GIVEN_SCORE = np.zeros(len(PROB_NOISE_LINE_SCORE))#\
PROB_NOISE_GIVEN_SCORE =  \
[        3.77800000e-01,   2.69600000e-01,   1.66400000e-01,
         9.05000000e-02,   4.69000000e-02,   2.28000000e-02,
         1.07000000e-02,   7.30000000e-03,   3.30000000e-03,
         2.30000000e-03,   1.00000000e-03,   5.00000000e-04,
         5.00000000e-04,   2.00000000e-04]

PROB_NOISE_TRUNCATED = 0.0002 #all bins after the end of the list get this value
PROB_NOISE_MIN_SCORE = 2.0 #min score that makes it to the bin list

MAX_LINESCORE_SNR = 20.0 #limit the contribution to the line score of the computed line SNR
                         #all SNR values are capped at this value in the line score calculation

#beyond an okay fit (see GAUSS_FIT_xxx above) is this a "good" signal
GOOD_BROADLINE_SIGMA = 6.0 #getting broad
LIMIT_BROAD_SIGMA = 8.49 #not 8.5 ... tend to get 8.49999, #7.0 #above this the emission line must specifically allow "broad"

GOOD_BROADLINE_SNR = 11.0 # upshot ... protect against neighboring "noise" that fits a broad line ...
                          # if big sigma, better have big SNR
GOOD_BROADLINE_RAW_SNR = 4.0 # litteraly signal/noise (flux values / error values +/- 3 sigma from peak)
GOOD_MIN_LINE_SNR = 4.0
GOOD_MIN_LINE_SCORE = 3.0 #lines are added to solution only if 'GOOD' (meaning, minimally more likely real than noise)
#does not have to be the same as PROB_NOISE_MIN_SCORE, but that generally makes sense
GOOD_BROADLINE_MIN_LINE_SCORE = 20.0 #if we have a special broad-line match, the score MUST be higher to accept
GOOD_FULL_SNR = 9.0 #ignore SBR is SNR is above this
#GOOD_MIN_SNR = 5.0 #bare-minimum; if you change the SNR ranges just above, this will also need to change
GOOD_MIN_SBR = 6.0 #signal to "background" noise (looks at peak height vs surrounding peaks) (only for "weak" signals0
GOOD_MIN_SIGMA = 1.35 #very narrow, but due to measurement error, could be possible (should be around 2 at a minimum, but with error ...)
#1.8 #in AA or FWHM ~ 4.2 (really too narrow, but allowing for some error)
#GOOD_MIN_EW_OBS = 1.5 #not sure this is a good choice ... really should depend on the physics of the line and
                      # not be absolute
#GOOD_MIN_EW_REST = 1.0 #ditto here

#GOOD_MIN_LINE_FLUX = 5.0e-18 #todo: this should be the HETDEX flux limit (but that depends on exposure time and wavelength)
#combined in the score

#GOOD_MAX_DX0_MUTL .... ALL in ANGSTROMS
MAX_LYA_VEL_OFFSET = 500.0 #km/s
NOMINAL_MAX_OFFSET_AA =8.0 # 2.75  #using 2.75AA as 1/2 resolution for HETDEX at 5.5AA
                           # ... maybe 3.0 to give a little wiggle room for fit?
GOOD_MAX_DX0_MULT_LYA = [-8.0,NOMINAL_MAX_OFFSET_AA] #can have a sizeable velocity offset for LyA (actaully depends on z, so this is a default)
#assumes a max velocity offset of 500km/s at 4500AA ==> 4500 * 500/c = 7.5AA
GOOD_MAX_DX0_MULT_OTHER = [-1.*NOMINAL_MAX_OFFSET_AA,NOMINAL_MAX_OFFSET_AA] #all others are symmetric and smaller

GOOD_MAX_DX0_MULT = GOOD_MAX_DX0_MULT_OTHER#[-1.75,1.75] #3.8 (AA)
                    # #maximum error (domain freedom) in fitting to line center in AA
                    #since this is based on the fit of the extra line AND the line center error of the central line
                    #this is a compound error (assume +/- 2 AA since ~ 1.9AA/pix for each so at most 4 AA here)?
#GOOD_MIN_H_CONT_RATIO = 1.33 #height of the peak must be at least 33% above the continuum fit level


#todo: impose line ratios?
#todo:  that is, if line_x is assumed and line_y is assumed, can only be valid if line_x/line_y ~ ratio??
#todo:  i.e. [OIII(5007)] / [OIII(4959)] ~ 3.0 (2.993 +/- 0.014 ... at least in AGN)

ABSORPTION_LINE_SCORE_SCALE_FACTOR = 0.5 #treat absorption lines as 50% of the equivalent emission line score


#FLUX conversion are pretty much defunct, but kept here as a last ditch conversion if all else fails
FLUX_CONVERSION_measured_w = [3000., 3500., 3540., 3640., 3740., 3840., 3940., 4040., 4140., 4240., 4340., 4440., 4540., 4640., 4740., 4840.,
     4940., 5040., 5140.,
     5240., 5340., 5440., 5500., 6000.]
FLUX_CONVERSION_measured_f = [1.12687e-18, 1.12687e-18, 9.05871e-19, 6.06978e-19, 4.78406e-19, 4.14478e-19, 3.461e-19, 2.77439e-19, 2.50407e-19,
     2.41462e-19, 2.24238e-19, 2.0274e-19, 1.93557e-19, 1.82048e-19, 1.81218e-19, 1.8103e-19, 1.81251e-19,
     1.80744e-19, 1.85613e-19, 1.78978e-19, 1.82547e-19, 1.85056e-19, 2.00788e-19, 2.00788e-19]

FLUX_CONVERSION_w_grid = np.arange(3000.0, 6000.0, 1.0)
FLUX_CONVERSION_f_grid = np.interp(FLUX_CONVERSION_w_grid, FLUX_CONVERSION_measured_w, FLUX_CONVERSION_measured_f)

FLUX_CONVERSION_DICT = dict(zip(FLUX_CONVERSION_w_grid,FLUX_CONVERSION_f_grid))

#this actually varies with seeing FWHM and throughput and really should be computed per shot
#but this is a pretty good average for use with stacking
COUNTS_TO_FLUX_BASE_NORMALIZATION = np.array(FLUX_CONVERSION_measured_f) / FLUX_CONVERSION_measured_f[14] #idx 14 (4740AA) is closest to 4726AA
COUNTS_TO_FLUX_BASE_NORMALIZATION_INTERP = np.interp(G.CALFIB_WAVEGRID, FLUX_CONVERSION_measured_w, COUNTS_TO_FLUX_BASE_NORMALIZATION)


def update_with_globals():
    global GAUSS_FIT_MAX_SIGMA,GAUSS_FIT_MIN_SIGMA

    if G.LIMIT_GAUSS_FIT_SIGMA_MAX is not None:
        GAUSS_FIT_MAX_SIGMA = G.LIMIT_GAUSS_FIT_SIGMA_MAX

    if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None:
        GAUSS_FIT_MIN_SIGMA = G.LIMIT_GAUSS_FIT_SIGMA_MIN


def in_same_family(a,e):#are the emission lines different species of the same element
    """

    :param a: an emission line object
    :param e: an emission line object
    :return:
    """
    same = False
    try:
        if (e.name[0] == 'O') and (a.name[0] == 'O'):
            same = True
        elif (e.name[0:2] == 'CI') and (a.name[0:2] == 'CI'):
            same = True
        elif (e.name[0:2] == 'H$') and (a.name[0:2] == 'H$'):
            same = True
    except:
        pass

    return same

def conf_interval(num_samples,sd,conf=0.95):
    return SU.conf_interval(num_samples,sd,conf)
# moved to spectrum utilities
# def conf_interval(num_samples,sd,conf=0.95):
#     """
#     mean +/- error  ... this is the +/- error part as 95% (or other) confidence interval (assuming normal distro)
#
#     :param num_samples:
#     :param sd: standard deviation
#     :param conf:
#     :return:
#     """
#
#     if num_samples < 30:
#         return None
#
#     #todo: put in other values
#     if conf == 0.68:
#         t = 1.0
#     elif conf == 0.95:
#         t = 1.96
#     elif conf == 0.99:
#         t = 2.576
#     else:
#         log.debug("todo: need to handle other confidence intervals: ", conf)
#         return None
#
#     return t * sd / np.sqrt(num_samples)


def get_sdss_gmag(_flux_density, wave, flux_err=None, num_mc=G.MC_PLAE_SAMPLE_SIZE, confidence=G.MC_PLAE_CONF_INTVL,
                  ignore_global=False):
    return SU.get_sdss_gmag(_flux_density, wave, flux_err, num_mc, confidence, ignore_global)

# moved to spectrum utilities
# def get_sdss_gmag(_flux_density, wave, flux_err=None, num_mc=G.MC_PLAE_SAMPLE_SIZE, confidence=G.MC_PLAE_CONF_INTVL, ignore_global=False):
#     """
#
#     :param flux_density: erg/s/cm2/AA  (*** reminder, HETDEX sumspec usually a flux erg/s/cm2 NOT flux denisty)
#     :param wave: in AA
#     :param flux_err: error array for flux_density (if None, then no error is computed)
#     :param num_mc: number of MC samples to run
#     :param confidence:  confidence interval to report
#     :return: AB mag in g-band and continuum estimate (erg/s/cm2/AA)
#             if flux_err is specified then also returns error on mag and error on the flux (continuum)
#     """
#
#     #log.debug("++++  #todo: in caller or here, enfore a limit based on the 1-sigma flux limits at the CCD position and the seeing FWHM")
#
#     if not ignore_global:
#         if not G.USE_SDSS_SPEC_GMAG:
#             if flux_err is not None:
#                 return None, None, None, None
#             else:
#                 return None, None
#
#     try:
#         mag = None
#         cont = None
#         mag_err = None
#         cont_err = None
#         no_error = False
#         if flux_err is None or len(flux_err) == 0 or not np.any(flux_err):
#             no_error = True
#
#
#         # num_mc = G.MC_PLAE_SAMPLE_SIZE #good enough for this (just use the same as the MC for the PLAE/POII
#         # confidence = G.MC_PLAE_CONF_INTVL
#
#         filter_name = 'sdss2010-g'
#         #filter_name = 'hsc2017-g'
#         sdss_filter = speclite_filters.load_filters(filter_name)
#         # not quite correct ... but can't find the actual f_iso freq. and f_iso lambda != f_iso freq, but
#         # we should not be terribly far off (and there are larger sources of error here anyway since this is
#         # padded HETDEX data passed through an SDSS-g filter (approximately)
#         #iso_f = 3e18 / sdss_filter.effective_wavelengths[0].value
#         iso_lam = sdss_filter.effective_wavelengths[0].value
#         #HSC-2017-g (4843.7) sdss-2010-g (4726.1), other reports SDSS as 4770 or 4640;
#
#
#         #sanity check flux_density
#         flux_density = copy.copy(_flux_density)
#         sel = np.where(abs(flux_density) > 1e-5) #remember, these are e-17, so that is enormous
#         if np.any(sel):
#             msg = "Warning! Absurd flux density values: [%f,%f] (normal expected values e-15 to e-19 range)" %(min(flux_density[sel]),max(flux_density[sel]))
#             print(msg)
#             log.warning(msg)
#             flux_density[sel] = 0.0
#
#         #if flux_err is specified, assume it is Gaussian and sample, repeatedly building up spectra
#         if flux_err is not None and not no_error:
#             try:
#                 flux_err = np.array(flux_err)
#                 flux_density = np.array(flux_density)
#
#                 mag_list = []
#                 cont_list = []
#                 sel = ~np.isnan(flux_err) & np.array(flux_err!=0)
#
#                 #trim off the ends (only use 3600-5400)
#                 #SDSS g drops shaprly below 3900 or 4000 AA, but this call should already handle it
#                 #but will use our defined values for consistency
#                 sel = sel & np.array(wave > G.SDSS_G_FILTER_BLUE) & np.array(wave < G.SDSS_G_FILTER_RED)
#
#                 if not np.any(sel):
#                     log.info("Invalid spectrum or error in get_sdds_gma.")
#                     if flux_err is not None: #even if this failed, the caller expects the extra two returns
#                         return mag, cont, mag_err, cont_err
#                     else:
#                         return mag, cont
#
#                 for i in range(num_mc):
#                     flux_sample = np.random.normal(flux_density[sel], flux_err[sel])
#
#                     flux, wlen = sdss_filter.pad_spectrum(
#                         flux_sample * (units.erg / units.s / units.cm ** 2 / units.Angstrom), wave[sel] * units.Angstrom)
#                     mag = sdss_filter.get_ab_magnitudes(flux, wlen)[0][0]
#                     #cont = 3631.0 * 10 ** (-0.4 * mag) * 1e-23 * iso_f / (wlen[-1] - wlen[0]).value  # (5549.26 - 3782.54) #that is the approximate bandpass
#
#                     cont = 3631.0 * 10 ** (-0.4 * mag) * 1e-23 * 3e18 / (iso_lam * iso_lam)
#
#                     mag_list.append(mag)
#                     cont_list.append(cont)
#
#                 mag_list = np.array(mag_list)
#                 cont_list = np.array(cont_list)
#
#                 #clean the nans
#                 mag_list  = mag_list[~np.isnan(mag_list)]
#                 cont_list = cont_list[~np.isnan(cont_list)]
#
#                 loc = biweight.biweight_location(mag_list)  # the "average"
#                 scale = biweight.biweight_scale(mag_list)
#                 ci = conf_interval(len(mag_list), scale * np.sqrt(num_mc), conf=confidence)
#                 mag = loc
#                 mag_err = ci
#
#                 loc = biweight.biweight_location(cont_list)  # the "average"
#                 scale = biweight.biweight_scale(cont_list)
#                 ci = conf_interval(len(cont_list), scale * np.sqrt(num_mc), conf=confidence)
#                 cont = loc
#                 cont_err = ci
#
#                 no_error = False
#             except:
#                 log.info("Exception in spectrum::get_sdss_gmag()",exc_info=True)
#                 no_error = True
#
#         if no_error: #if we cannot compute the error, the just call once (no MC sampling)
#             sel = np.array(wave > G.SDSS_G_FILTER_BLUE) & np.array(wave < G.SDSS_G_FILTER_RED)
#             flux, wlen = sdss_filter.pad_spectrum(flux_density[sel]* (units.erg / units.s /units.cm**2/units.Angstrom),wave[sel]* units.Angstrom)
#             mag = sdss_filter.get_ab_magnitudes(flux , wlen )[0][0]
#             #cont = 3631.0 * 10**(-0.4*mag) * 1e-23 * iso_f / (wlen[-1] - wlen[0]).value
#             cont = 3631.0 * 10 ** (-0.4 * mag) * 1e-23 * 3e18 / (iso_lam * iso_lam)#(5549.26 - 3782.54) #that is the approximate bandpass
#             mag_err = None
#             cont_err = None
#     except:
#         log.warning("Exception! in spectrum::get_sdss_gmag.",exc_info=True)
#
#     if flux_err is not None: #even if this failed, the caller expects the extra two returns
#         return mag, cont, mag_err, cont_err
#     else:
#         return mag, cont
#
#

def get_hetdex_gmag(_flux_density, wave, _flux_density_err=None, ignore_global=False, log_iso_detid=None):
    return SU.get_hetdex_gmag(_flux_density, wave, _flux_density_err, ignore_global, log_iso_detid)

# moved to spectrum utilities
# def get_hetdex_gmag(_flux_density, wave, _flux_density_err=None, ignore_global=False, log_iso_detid=None):
#     """
#     Similar to get_sdss_gmag, but this uses ONLY the HETDEX spectrum and its errors
#
#     Simple mean over spectrum ... should use something else? Median or Biweight?
#
#     UPDATE: this version (vs the _old version) converts to fnu first THEN takes the aveage and feeds that to the magnitude
#     as opposed to averaging the flam flux and then converting to fnu
#
#     :param flux_density: erg/s/cm2/AA  (*** reminder, HETDEX sumspec usually a flux erg/s/cm2 NOT flux denisty)
#     :param wave: in AA
#     :param flux_err: error array for flux_density (if None, then no error is computed)
#     :return: AB mag in g-band and continuum estimate (erg/s/cm2/AA)
#             if flux_err is specified then also returns error on mag and error on the flux (continuum)
#     """
#
#     #in caller or here, enfore a limit based on the 1-sigma flux limits at the CCD position and the seeing FWHM
#     #log.debug("++++  #todo: in caller or here, enfore a limit based on the 1-sigma flux limits at the CCD position and the seeing FWHM")
#
#     if not ignore_global:
#         if not G.USE_HETDEX_SPEC_GMAG:
#             if _flux_density_err is not None:
#                 return None, None, None, None
#             else:
#                 return None, None
#
#
#     method = 2 #1 = mean, 2 = weighted mean, 3 = weighted biweight
#     try:
#         #use the SDSS-g wavelength if can as would be used by the get_sdss_gmag() above
#         filter_name = 'sdss2010-g'
#         #filter_name = 'hsc2017-g'
#         f_lam_iso = speclite_filters.load_filters(filter_name).effective_wavelengths[0].value
#         # HSC-2017-g (4843.7) sdss-2010-g (4726.1), other reports SDSS as 4770 or 4640;
#     except:
#         #f_lam_iso = 4726.1 #should be about this value anyway
#         f_lam_iso = G.DEX_G_EFF_LAM
#
#     try:
#         #f_lam_iso = 4500.0  # middle of the range #not really the "true" f_lam_iso, but prob. intrudces small error compared to others
#         mag = None
#         cont = None
#         mag_err = None
#         cont_err = None
#         no_errors = False
#         if (_flux_density_err is None) or (len(_flux_density_err) == 0):
#             flux_density_err = np.zeros(len(wave))
#             no_errors = True
#         elif not np.any(_flux_density_err):
#             no_errors = True
#             flux_density_err = np.zeros(len(_flux_density))
#         else:
#             flux_density_err = copy.copy(_flux_density_err)
#
#
#         #sanity check flux_density
#         flux_density = copy.copy(_flux_density)
#         sel = np.where(abs(flux_density) > 1e-5) #remember, these are e-17, so that is enormous
#         if np.any(sel):
#             msg = "Warning! Absurd flux density values: [%f,%f] (normal expected values e-15 to e-19 range)" %(min(flux_density[sel]),max(flux_density[sel]))
#             print(msg)
#             log.warning(msg)
#             flux_density[sel] = 0.0
#
#
#         #trim off the ends (only use 3600-5400)
#         #idx_3600,*_ = SU.getnearpos(wave,3600.)
#         #idx_5400,*_ = SU.getnearpos(wave,5400.)
#
#         # fluxbins = np.array(flux_density[idx_3600:idx_5400+1]) * G.FLUX_WAVEBIN_WIDTH
#         # fluxerrs = np.array(flux_density_err[idx_3600:idx_5400+1]) * G.FLUX_WAVEBIN_WIDTH
#
#         sel = np.array(wave > G.SDSS_G_FILTER_BLUE) & np.array(wave < G.SDSS_G_FILTER_RED)
#         if not no_errors and flux_density_err is not None and np.any(flux_density_err):
#             sel = sel & ~np.isnan(flux_density_err) & np.array(flux_density_err != 0)
#
#         if np.count_nonzero(sel) < 100:
#             log.info("Invalid spectrum or error in get_hetdex_gmag.")
#             if flux_density_err is not None: #even if this failed, the caller expects the extra two returns
#                 return mag, cont, mag_err, cont_err
#             else:
#                 return mag, cont
#
#         fnu = SU.flam2fnu(flux_density[sel],wave[sel])*1e30
#         fnu_errs = SU.flam2fnu(flux_density_err[sel],wave[sel])*1e30
#
#         if method == 1:
#             band_avg_fnu = np.nanmean(fnu)
#             if no_errors:
#                 band_avg_err = 0
#             else:
#                 band_avg_err = np.sqrt(np.sum(fnu_errs*fnu_errs))/np.count_nonzero(sel)
#         elif method == 2: #makes a small improvement in matching to HSC-g faint mags
#             if no_errors:
#                 band_avg_fnu = np.nanmean(fnu)
#                 band_avg_err = 0
#             else:
#                 band_avg_fnu = utils.weighted_mean(fnu,1.0/(fnu_errs**2))
#                 band_avg_err = np.sqrt(np.sum(fnu_errs*fnu_errs))/np.count_nonzero(sel)
#         elif method == 3: #makes no real difference, some small scatter
#             if no_errors:
#                 band_avg_fnu = biweight.biweight_location(fnu)#,errors=fnu_errs)
#                 band_avg_err = biweight.biweight_scale(fnu)/np.sqrt(np.count_nonzero(sel))
#             else:
#                 band_avg_fnu = weighted_biweight.biweight_location_errors(fnu,errors=fnu_errs)
#                 band_avg_err = biweight.biweight_scale(fnu)/np.sqrt(np.count_nonzero(sel))
#
#         band_avg_fnu  /= 1e30
#         band_avg_err /= 1e30
#
#         #the old way, but also for the continuum as flux density erg/s/cm2/AA ... un-weighted
#         fluxbins = np.array(flux_density)[sel] #* G.FLUX_WAVEBIN_WIDTH
#         fluxerrs = np.array(flux_density_err)[sel] #* G.FLUX_WAVEBIN_WIDTH
#
#         if method == 1:
#             band_flux_density = np.nanmean(fluxbins)
#             if no_errors:
#                 banband_flux_density_errd_avg_err = 0
#             else:
#                 banband_flux_density_errd_avg_err = np.sqrt(np.sum(fluxerrs*fluxerrs))/np.count_nonzero(sel)
#         elif method == 2: #makes a small improvement in matching to HSC-g faint mags
#             if no_errors:
#                 band_flux_density = np.nanmean(fluxbins) #/ (fluxerrs ** 2))
#                 band_flux_density_err = 0
#             else:
#                 band_flux_density = utils.weighted_mean(fluxbins,1.0/(fluxerrs**2))
#                 band_flux_density_err = np.sqrt(np.sum(fluxerrs*fluxerrs))/np.count_nonzero(sel)
#         elif method == 3: #makes no real difference, some small scatter
#             if no_errors:
#                 band_flux_density = biweight.biweight_location(fluxbins)#, errors=fluxerrs)
#                 band_flux_density_err = biweight.biweight_scale(fnu) / np.sqrt(np.count_nonzero(sel))
#             else:
#                 band_flux_density = weighted_biweight.biweight_location_errors(fluxbins,errors=fluxerrs)
#                 band_flux_density_err = biweight.biweight_scale(fnu)/np.sqrt(np.count_nonzero(sel))
#
#         #integrated_flux = np.sum(fluxbins)
#         #integrated_errs = np.sqrt(np.sum(fluxerrs*fluxerrs))
#         #band_flux_density = integrated_flux/(np.count_nonzero(sel)) #*G.FLUX_WAVEBIN_WIDTH)
#         #band_flux_density_err = integrated_errs/(np.count_nonzero(sel)) #*G.FLUX_WAVEBIN_WIDTH)
#
#         if band_flux_density > 0:
#             mag_flam = SU.cgs2mag(band_flux_density, f_lam_iso)
#             mag_flam_bright = SU.cgs2mag(band_flux_density + band_flux_density_err, f_lam_iso)
#             mag_flam_faint = SU.cgs2mag(band_flux_density - band_flux_density_err, f_lam_iso)
#             if np.isnan(mag_flam_faint):
#                 log.debug("Warning. HETDEX full spectrum mag estimate is invalid on the faint end.")
#                 mag_flam_err = mag_flam - mag_flam_bright
#             else:
#                 mag_flam_err = 0.5 * (mag_flam_faint - mag_flam_bright)  # not symmetric, but this is roughly close enough
#         else:
#             mag_flam = None
#             mag_flam_err = None
#
#             # log.info(
#             #     f"HETDEX full width gmag, continuum estimate ({band_flux_density:0.3g}) below flux limit. Setting mag to None.")
#             # if flux_density_err is not None:
#             #     return None, band_flux_density, None, band_flux_density_err
#             # else:
#             #     return None, band_flux_density
#
#         if band_avg_fnu > 0:
#             mag_fnu = SU.fnu2mag(band_avg_fnu)
#             mag_fnu_bright = SU.fnu2mag((band_avg_fnu+band_avg_err))
#             mag_fnu_faint = SU.fnu2mag((band_avg_fnu-band_avg_err))
#             if np.isnan(mag_fnu_faint):
#                 log.debug("Warning. HETDEX full spectrum mag estimate is invalid on the faint end.")
#                 mag_fnu_err = mag_fnu - mag_fnu_bright
#             else:
#                 mag_fnu_err = 0.5 * (mag_fnu_faint-mag_fnu_bright) #not symmetric, but this is roughly close enough
#         else:
#             mag_fnu = None
#             mag_fnu_err = None
#
#             # log.info(f"HETDEX full width gmag, continuum estimate ({band_flux_density:0.3g}) below flux limit. Setting mag to None.")
#             # if flux_density_err is not None:
#             #     return None, band_flux_density, None, band_flux_density_err
#             # else:
#             #     return None, band_flux_density
#
#
#         if True: #prefer mag_flam over mag_fnu
#             if mag_flam is not None:
#                 mag = mag_flam
#                 mag_err = mag_flam_err
#             else:
#                 mag = mag_fnu
#                 mag_err = mag_fnu_err
#         else:
#             if mag_fnu is not None:
#                 mag = mag_fnu
#                 mag_err = mag_fnu_err
#             else:
#                 mag = mag_flam
#                 mag_err = mag_flam_err
#
#         # what is the HETDEX iso wavelength ???
#         if log_iso_detid is not None:
#             log.info(
#                 f"{log_iso_detid} COMPUTED ISO WAVELENGTH: {np.sqrt(band_avg_fnu / band_flux_density * 2.99792458e+18)} "
#                 f"Defined ISO wavelength: {f_lam_iso} mag flam: {mag_flam} +/- {mag_flam_err} mag fnu: {mag_fnu} +/- {mag_fnu_err}")
#
#         #todo: technically, should remove the emission lines to better fit actual contiuum, rather than just use band_flux_density
#         # but I think this is okay and appropriate and matches the other uses as the "band-pass" continuum
#         if flux_density_err is not None:
#             return mag, band_flux_density, mag_err, band_flux_density_err
#         else:
#             return mag, band_flux_density
#
#     except:
#         log.warning("Exception! in spectrum::get_hetdex_gmag.",exc_info=True)
#         if flux_density_err is not None:
#             return None, None, None, None
#         else:
#             return None, None
#
#
# def get_hetdex_gmag_old(flux_density, wave, flux_density_err=None, ignore_global=False):
#     """
#     Similar to get_sdss_gmag, but this uses ONLY the HETDEX spectrum and its errors
#
#     Simple mean over spectrum ... should use something else? Median or Biweight?
#
#     :param flux_density: erg/s/cm2/AA  (*** reminder, HETDEX sumspec usually a flux erg/s/cm2 NOT flux denisty)
#     :param wave: in AA
#     :param flux_err: error array for flux_density (if None, then no error is computed)
#     :return: AB mag in g-band and continuum estimate (erg/s/cm2/AA)
#             if flux_err is specified then also returns error on mag and error on the flux (continuum)
#     """
#
#     #in caller or here, enfore a limit based on the 1-sigma flux limits at the CCD position and the seeing FWHM
#     #log.debug("++++  #todo: in caller or here, enfore a limit based on the 1-sigma flux limits at the CCD position and the seeing FWHM")
#
#     if not ignore_global:
#         if not G.USE_HETDEX_SPEC_GMAG:
#             if flux_density_err is not None:
#                 return None, None, None, None
#             else:
#                 return None, None
#
#     try:
#         #use the SDSS-g wavelength if can as would be used by the get_sdss_gmag() above
#         filter_name = 'sdss2010-g'
#         #filter_name = 'hsc2017-g'
#         f_lam_iso = speclite_filters.load_filters(filter_name).effective_wavelengths[0].value
#         # HSC-2017-g (4843.7) sdss-2010-g (4726.1), other reports SDSS as 4770 or 4640;
#     except:
#         #f_lam_iso = 4726.1 #should be about this value anyway
#         f_lam_iso = G.DEX_G_EFF_LAM
#
#     try:
#         #f_lam_iso = 4500.0  # middle of the range #not really the "true" f_lam_iso, but prob. intrudces small error compared to others
#         mag = None
#         cont = None
#         mag_err = None
#         cont_err = None
#         if (flux_density_err is None) or (len(flux_density_err) == 0):
#             flux_density_err = np.zeros(len(wave))
#
#
#         #sanity check flux_density
#         sel = np.where(abs(flux_density) > 1e-5) #remember, these are e-17, so that is enormous
#         if np.any(sel):
#             msg = "Warning! Absurd flux density values: [%f,%f] (normal expected values e-15 to e-19 range)" %(min(flux_density[sel]),max(flux_density[sel]))
#             print(msg)
#             log.warning(msg)
#             flux_density[sel] = 0.0
#
#
#         #trim off the ends (only use 3600-5400)
#         #idx_3600,*_ = SU.getnearpos(wave,3600.)
#         #idx_5400,*_ = SU.getnearpos(wave,5400.)
#
#         # fluxbins = np.array(flux_density[idx_3600:idx_5400+1]) * G.FLUX_WAVEBIN_WIDTH
#         # fluxerrs = np.array(flux_density_err[idx_3600:idx_5400+1]) * G.FLUX_WAVEBIN_WIDTH
#
#         sel = np.array(wave > G.SDSS_G_FILTER_BLUE) & np.array(wave < G.SDSS_G_FILTER_RED)
#         fluxbins = np.array(flux_density)[sel] * G.FLUX_WAVEBIN_WIDTH
#         fluxerrs = np.array(flux_density_err)[sel] * G.FLUX_WAVEBIN_WIDTH
#
#         sel = ~np.isnan(fluxerrs) & np.array(fluxerrs!=0)
#
#         if not np.any(sel):
#             log.info("Invalid spectrum or error in get_hetdex_gmag.")
#             if flux_density_err is not None: #even if this failed, the caller expects the extra two returns
#                 return mag, cont, mag_err, cont_err
#             else:
#                 return mag, cont
#
#
#         integrated_flux = np.sum(fluxbins[sel])
#         integrated_errs = np.sqrt(np.sum(fluxerrs[sel]*fluxerrs[sel]))
#
#         #This already been thoughput adjusted? (Yes? I think)
#         #so there is no need to adjust for transmission
#         # remeber to add one more bin (bin 2 - bin 1 != 1 bin it is 2 bins, not 1 as both bins are included)
#         band_flux_density = integrated_flux/(np.sum(sel) *G.FLUX_WAVEBIN_WIDTH)
#         band_flux_density_err = integrated_errs/(np.sum(sel) *G.FLUX_WAVEBIN_WIDTH)
#
#
#         if band_flux_density > 0:
#             mag = SU.cgs2mag(band_flux_density,f_lam_iso)
#             mag_bright = SU.cgs2mag(band_flux_density+band_flux_density_err, f_lam_iso)
#             mag_faint = SU.cgs2mag(band_flux_density-band_flux_density_err, f_lam_iso)
#             if np.isnan(mag_faint):
#                 log.debug("Warning. HETDEX full spectrum mag estimate is invalid on the faint end.")
#                 mag_err = mag - mag_bright
#             else:
#                 mag_err = 0.5 * (mag_faint-mag_bright) #not symmetric, but this is roughly close enough
#         else:
#             log.info(f"HETDEX full width gmag, continuum estimate ({band_flux_density:0.3g}) below flux limit. Setting mag to None.")
#             if flux_density_err is not None:
#                 return None, band_flux_density, None, band_flux_density_err
#             else:
#                 return None, band_flux_density
#
#
#         #todo: technically, should remove the emission lines to better fit actual contiuum, rather than just use band_flux_density
#         # but I think this is okay and appropriate and matches the other uses as the "band-pass" continuum
#         if flux_density_err is not None:
#             return mag, band_flux_density, mag_err, band_flux_density_err
#         else:
#             return mag, band_flux_density
#
#     except:
#         log.warning("Exception! in spectrum::get_hetdex_gmag.",exc_info=True)
#         if flux_density_err is not None:
#             return None, None, None, None
#         else:
#             return None, None
#
#

def get_best_gmag(flux_density, flux_density_err, wavelengths):
    return SU.get_best_gmag(flux_density, flux_density_err, wavelengths)

# #moved to spectrum utilities
# def get_best_gmag(flux_density, flux_density_err, wavelengths):
#     """
#
#     :param flux_density: as erg/s/cm2/AA
#     :param flux_density_err: ditto
#     :param wavelengths: in AA
#     :return:  best_gmag, best_gmag_unc, best_gmag_cgs_cont, best_gmag_cgs_cont_unc
#     """
#     sdss_okay = 0
#     hetdex_okay = 0
#     best_gmag = None
#     best_gmag_unc = None
#     best_gmag_cgs_cont = None
#     best_gmag_cgs_cont_unc = None
#
#     # sum over entire HETDEX spectrum to estimate g-band magnitude
#     try:
#         hetdex_gmag, hetdex_gmag_cgs_cont, hetdex_gmag_unc, hetdex_gmag_cgs_cont_unc = get_hetdex_gmag(flux_density,
#                                                                                         wavelengths,
#                                                                                         flux_density_err)
#
#         log.debug(f"HETDEX spectrum gmag {hetdex_gmag} +/- {hetdex_gmag_unc}")
#         log.debug(f"HETDEX spectrum cont {hetdex_gmag_cgs_cont} +/- {hetdex_gmag_cgs_cont_unc}")
#
#         if (hetdex_gmag_cgs_cont is not None) and (hetdex_gmag_cgs_cont != 0) and not np.isnan(hetdex_gmag_cgs_cont):
#             if (hetdex_gmag_cgs_cont_unc is None) or np.isnan(hetdex_gmag_cgs_cont_unc):
#                 hetdex_gmag_cgs_cont_unc = 0.0
#                 hetdex_okay = 1
#             else:
#                 hetdex_okay = 2
#         if (hetdex_gmag is None) or np.isnan(hetdex_gmag):
#             hetdex_okay = 0
#     except:
#         hetdex_okay = 0
#         log.error("Exception computing HETDEX spectrum gmag", exc_info=True)
#
#     # feed HETDEX spectrum through SDSS gband filter
#     try:
#         # reminder needs erg/s/cm2/AA and sumspec_flux in ergs/s/cm2 so divied by 2AA bin width
#         #                self.sdss_gmag, self.cont_cgs = elixer_spectrum.get_sdss_gmag(self.sumspec_flux/2.0*1e-17,self.sumspec_wavelength)
#         sdss_gmag, sdss_cgs_cont, sdss_gmag_unc, sdss_cgs_cont_unc = get_sdss_gmag(flux_density,
#                                                                                    wavelengths,
#                                                                                    flux_density_err)
#
#         log.debug(f"SDSS spectrum gmag {sdss_gmag} +/- {sdss_gmag_unc}")
#         log.debug(f"SDSS spectrum cont {sdss_cgs_cont} +/- {sdss_cgs_cont_unc}")
#
#         if (sdss_cgs_cont is not None) and (sdss_cgs_cont != 0) and not np.isnan(sdss_cgs_cont):
#             if (sdss_cgs_cont_unc is None) or np.isnan(sdss_cgs_cont_unc):
#                 sdss_cgs_cont_unc = 0.0
#                 sdss_okay = 1
#             else:
#                 sdss_okay = 2
#
#         if (sdss_gmag is None) or np.isnan(sdss_gmag):
#             sdss_okay = 0
#
#     except:
#         sdss_okay = 0
#         log.error("Exception computing SDSS g-mag", exc_info=True)
#
#     # choose the best
#     # even IF okay == 0, still record the probably bogus value (when
#     # actually using the values elsewhere they are compared to a limit and the limit is used if needed
#
#     try:
#         if (hetdex_okay == sdss_okay) and (hetdex_gmag is not None) and (sdss_gmag is not None) and \
#                 abs(hetdex_gmag - sdss_gmag) < 1.0:  # use both as an average? what if they are very different?
#             # make the average
#             avg_cont = 0.5 * (hetdex_gmag_cgs_cont + sdss_cgs_cont)
#             avg_cont_unc = np.sqrt((0.5 * hetdex_gmag_cgs_cont_unc)**2 + (0.5 * sdss_cgs_cont_unc) ** 2)
#
#             #best_gmag_selected = 'mean'
#             # HSC-2017-g (4843.7) sdss-2010-g (4726.1), other reports SDSS as 4770 or 4640;
#             filter_name = 'sdss2010-g'
#             # filter_name = 'hsc2017-g'
#             try:
#                 f_lam_iso = speclite_filters.load_filters(filter_name).effective_wavelengths[0].value
#             except:
#                 f_lam_iso = G.DEX_G_EFF_LAM
#
#             best_gmag = -2.5 * np.log10(SU.cgs2ujy(avg_cont, f_lam_iso) / 1e6 / 3631.)
#             mag_faint = -2.5 * np.log10(SU.cgs2ujy(avg_cont - avg_cont_unc, f_lam_iso) / 1e6 / 3631.)
#             if np.isnan(mag_faint):
#                 msg_faint = best_gmag + 0.75  # about 50% error to the faint
#             mag_bright = -2.5 * np.log10(SU.cgs2ujy(avg_cont + avg_cont_unc, f_lam_iso) / 1e6 / 3631.)
#             best_gmag_unc = 0.5 * (mag_faint - mag_bright)
#
#             best_gmag_cgs_cont = avg_cont
#             best_gmag_cgs_cont_unc = avg_cont_unc
#
#             log.debug(f"Mean spectrum gmag {best_gmag:0.2f} +/- {best_gmag_unc:0.3f}; cont {best_gmag_cgs_cont} +/- {best_gmag_cgs_cont_unc}")
#
#         elif (hetdex_okay >= sdss_okay) and (hetdex_okay > 0) and not np.isnan(hetdex_gmag_cgs_cont) and (hetdex_gmag_cgs_cont is not None):
#             #best_gmag_selected = 'hetdex'
#             best_gmag = hetdex_gmag
#             best_gmag_unc = hetdex_gmag_unc
#             best_gmag_cgs_cont = hetdex_gmag_cgs_cont
#             best_gmag_cgs_cont_unc = hetdex_gmag_cgs_cont_unc
#
#             log.debug("Using HETDEX full width gmag over SDSS gmag.")
#         elif sdss_okay > 0 and not np.isnan(sdss_cgs_cont) and (sdss_cgs_cont is not None):
#             #best_gmag_selected = 'sdss'
#             best_gmag = sdss_gmag
#             best_gmag_unc = sdss_gmag_unc
#             best_gmag_cgs_cont = sdss_cgs_cont
#             best_gmag_cgs_cont_unc = sdss_cgs_cont_unc
#
#             log.debug("Using SDSS gmag over HETDEX full width gmag")
#         else:  # something catastrophically bad
#             log.debug("No full width spectrum g-mag estimate is valid.")
#             #best_gmag_selected = 'limit'
#             best_gmag = -999  # G.HETDEX_CONTINUUM_MAG_LIMIT
#             best_gmag_unc = 0
#             #hetdex gmag might still have a flux value, though it is probably negative
#             if hetdex_gmag_cgs_cont is not None and not np.isnan(hetdex_gmag_cgs_cont):
#                 best_gmag_cgs_cont = hetdex_gmag_cgs_cont
#                 best_gmag_cgs_cont_unc = hetdex_gmag_cgs_cont_unc
#             else:
#                 best_gmag_cgs_cont = -999
#                 hetdex_gmag_cgs_cont_unc = 0
#     except:
#         best_gmag = -999
#         best_gmag_unc = 0
#         best_gmag_cgs_cont = -999
#         best_gmag_cgs_cont_unc = 0
#         log.error("Exception selecting best g-mag from spectrum", exc_info=True)
#
#     return best_gmag, best_gmag_unc, best_gmag_cgs_cont, best_gmag_cgs_cont_unc
#

#moved to spectrum utilities
# def fit_line(wavelengths,values,errors=None,trim = False, emission_lines=None, absorption_lines=None):
#     """
#     #super simple line fit ... very basic
#     #rescale x so that we start at x = 0
#
#     :param wavelengths:
#     :param values:
#     :param errors:
#     :param trim: if true, trim off blue of 3600 and red of 5400
#     :param emission_lines: avoid as +/- 3 sigma found emission and absorption lines
#     :param absorption_lines:
#     :return: [intercept, slope] , [error on intercept, error on slope]
#     """
#
#     try:
#         sel = np.full(len(wavelengths),True)
#
#         if trim:
#             blue = G.HETDEX_BLUE_SAFE_WAVE
#             red = G.HETDEX_RED_SAFE_WAVE
#             sel = sel & np.array(wavelengths > blue) & np.array(wavelengths < red)
#
#         if emission_lines is not None:
#             for l in emission_lines:
#                 #mask out lines ... build up a sel to use
#                 try:
#                     blue = l.fit_x0 - 3*l.fit_sigma
#                     red = l.fit_x0 + 3*l.fit_sigma
#                     sel = sel & ~(np.array(wavelengths > blue) & np.array(wavelengths < red))
#                 except:
#                     pass
#
#         if absorption_lines is not None:
#             for l in absorption_lines:
#                 #mask out lines ... build up a sel to use
#                 try:
#                     blue = l.fit_x0 - 3*l.fit_sigma
#                     red = l.fit_x0 + 3*l.fit_sigma
#                     sel = sel & ~(np.array(wavelengths > blue) & np.array(wavelengths < red))
#                 except:
#                     pass
#
#         #optionally, trim off the bluest and reddest values ... if not overlapping with emission/absorption
#
#         coeff,cov = np.polyfit(wavelengths[sel],values[sel],deg=1,cov=True)
#
#         #abs() should be entirely redundane ... the diagnals should always be positive
#         errors = np.flip(np.sqrt(np.diag(cov)),0)
#         #errors = [np.sqrt(abs(cov[1,1])),np.sqrt(abs(cov[0,0]))]
#         #flip the array so [0] = 0th, [1] = 1st ...
#         coeff = np.flip(coeff,0)
#
#     except:
#         coeff = [None,None]
#
#     if False: #just for debug
#         fig = plt.figure(figsize=(8, 2), frameon=False)
#         line_plot = plt.axes()
#         line_plot.plot(wavelengths, values, c='b')
#
#         x_vals = np.array(line_plot.get_xlim())
#         y_vals = coeff[0] + coeff[1] * x_vals
#         line_plot.plot(x_vals, y_vals, '--',c='r')
#
#         fig.tight_layout()
#         fig.savefig("line.png")
#         fig.clear()
#         plt.close()
#         # end plotting
#
#     return coeff, errors
#
# def est_linear_continuum(w,bm,er=None):
#     """
#     Basically evaluate a point on a line given the line parameters and (optionally) their variances
#     The units depend on the units of the parameters and is up to the caller to know them.
#     For HETDEX/elixer they are usually flux over 2AA in erg/s/cm2 x10^-17
#
#     :param w: the wavelength at which to evaluate
#     :param bm:  as in y = mx + b  ... a two value array as [b,m] or [x^0, x^1]
#     :param er:  the error on b and m as a two value array  (as sigma or sqrt(variance))
#     :return: continuum and error
#     """
#     try:
#         y = bm[1] * w + bm[0]
#         ye = abs(y) * np.sqrt((er[1]/bm[1])**2 + er[0]**2)
#     except:
#         y = None
#         ye = None
#
#     return y, ye
#
#
#
# def est_linear_B_minus_V(bm,er=None):
#     """
#
#     :param bm:  as in y = mx + b  ... a two value array as [b,m] or [x^0, x^1]
#     :param er:  the error on b and m as a two value array  (as sigma or sqrt(variance))
#     :return: B-V and error
#     """
#     try:
#
#         yb, ybe = est_linear_continuum(4361.,bm,er) #standard f_iso,lam for B
#         yv, yve = est_linear_continuum(5448.,bm,er) #standard f_iso,lam for V
#
#         #need to convert from flux over 2AA to a f_nu like units
#         #since B-V is a ratio, the /2.0AA and x1e-17 factors don't matter
#
#         yb = SU.cgs2ujy(yb,4361.)
#         ybe = SU.cgs2ujy(ybe,4361.)
#
#         yv = SU.cgs2ujy(yv,5448.)
#         yve = SU.cgs2ujy(yve,5448.)
#
#         fac = 2.5/np.log(10.) #this is the common factor in the partial derivative of the 2.5 log10 (v/b)
#
#         b_v = 2.5 * np.log(yv/yb)
#         b_ve = abs(b_v)*np.sqrt((ybe*fac/yv)**2 + (yve*fac/yb)**2)
#
#     except:
#         b_v = None
#         b_ve = None
#
#     return b_v, b_ve
#

def invert_spectrum(wavelengths,values):
    # subtracting from the maximum value inverts the slope also, and keeps the overall shape intact
    # subtracting from the line fit slope flattens out the slope (leveling out the continuum) and changes the overall shape
    #
    #coeff = fit_line(wavelengths,values)
    #inverted = coeff[1]*wavelengths+coeff[0] - values

    mx = np.max(values)
    inverted = mx - values

    # if False: #for debugging
    #     if not 'coeff' in locals():
    #         coeff = [mx, 0]
    #
    #     fig = plt.figure(figsize=(8, 2), frameon=False)
    #     line_plot = plt.axes()
    #     line_plot.plot(wavelengths, values, c='g',alpha=0.5)
    #     x_vals = np.array(line_plot.get_xlim())
    #     y_vals = coeff[0] + coeff[1] * x_vals
    #     line_plot.plot(x_vals, y_vals, '--', c='b')
    #
    #     line_plot.plot(wavelengths, inverted, c='r' ,lw=0.5)
    #     fig.tight_layout()
    #     fig.savefig("inverted.png")
    #     fig.clear()
    #     plt.close()


    return inverted


def norm_values(values,values_units):
    '''
    Basically, make spectra values either counts or cgs x10^-18 (whose magnitdues are pretty close to counts) and the
    old logic and parameters can stay the same
    :param values:
    :param values_units:
    :return:
    '''

    #return values, values_units
    if values is not None:
        values = np.array(values)

    if values_units == 0: #counts
        return values, values_units
    elif values_units == 1:
        return values * 1e18, -18
    elif values_units == -17:
        return values * 10.0, -18
    elif values_units == -18:
        return values, values_units
    else:
        log.warning("!!! Problem. Unexpected values_units = %s" % str(values_units))
        return values, values_units


def flux_conversion(w): #electrons to ergs at wavelenght w
    if w is None:
        return 0.0
    w = round(w)

    if w in FLUX_CONVERSION_DICT.keys():
        return FLUX_CONVERSION_DICT[w]
    else:
        log.error("ERROR! Unable to find FLUX CONVERSION entry for %f" %w)
        return 0.0


def pix_to_aa(pix):
    #constant for now since interpolating to 1 AA per pix
    #e.g. pix * 1.0
    return float(pix)

def getnearpos(array,value):
    try:
        idx = (np.abs(array-value)).argmin()
    except:
        return None
    return idx


def gaussian(x,x0,sigma,a=1.0,y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None
    #return a * np.exp(-np.power((x - x0) / sigma, 2.) / 2.)
    #return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.))  + y

    #have the / np.sqrt(...) part so the basic shape is normalized to 1 ... that way the 'a' becomes the area
    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y
    #return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.)) + y




def gaussian_unc(x, mu, mu_u, sigma, sigma_u, A, A_u, y, y_u ):

    def df_dmu(x,mu,sigma,A):
        return A * (x - mu)/(np.sqrt(2.*np.pi)*sigma**3)*np.exp(-np.power((x - mu) / sigma, 2.) / 2.)

    def df_dsigma(x,mu,sigma,A):
        return A / (np.sqrt(2.*np.pi)*sigma**2) * (((x-mu)/sigma)**2 -1) * np.exp(-np.power((x - mu) / sigma, 2.) / 2.)

    def df_dA(x,mu,sigma):
        return 1./ (np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu) / sigma, 2.) / 2.)

    def df_dy():
        return 1

    try:
        f = gaussian(x,mu,sigma,A,y)

        variance = (mu_u**2) * (df_dmu(x,mu,sigma,A)**2) + (sigma_u**2) * (df_dsigma(x,mu,sigma,A)**2) + \
                   (A_u**2) * (df_dA(x,mu,sigma)**2) + (y_u**2) * (df_dy()**2)
    except:
        log.warning("Exception in spectrum::gaussian_unc: ", exc_info=True)
        f = None
        variance = 0


    return f, np.sqrt(variance)
#
#
#
# DEFUNCT: moved to spectrum_utilities.py
# def rms(data, fit,cw_pix=None,hw_pix=None,norm=True):
#     """
#
#     :param data: (raw) data
#     :param fit:  fitted data (on the same scale)
#     :param cw_pix: (nearest) pixel (index) of the central peak (could be +/- 1 pix (bin)
#     :param hw_pix: half-width (in pixels from the cw_pix) overwhich to calculate rmse (i.e. cw_pix +/- hw_pix)
#     :param norm: T/F whether or not to divide by the peak of the raw data
#     :return:
#     """
#     #sanity check
#     if (data is None) or (fit is None) or (len(data) != len(fit)) or any(np.isnan(data)) or any(np.isnan(fit)):
#         return -999
#
#     if norm:
#         mx = max(data)
#         if mx < 0:
#             return -999
#     else:
#         mx = 1.0
#
#     d = np.array(data)/mx
#     f = np.array(fit)/mx
#
#     if ((cw_pix is not None) and (hw_pix is not None)):
#         left = cw_pix - hw_pix
#         right = cw_pix + hw_pix
#
#         #due to rounding of pixels (bins) from the caller (the central index +/- 2 and the half-width to either side +/- 2)
#         # either left or right can be off by a max total of 4 pix
#         rounding_error = 4
#         if -1*rounding_error <= left < 0:
#             left = 0
#
#         if len(data) < right <= (len(data) +rounding_error):
#             right = len(data)
#
#         if (left < 0) or (right > len(data)):
#             log.warning("Invalid range supplied for rms. Data len = %d. Central Idx = %d , Half-width= %d"
#                       % (len(data),cw_pix,hw_pix))
#             return -999
#
#         d = d[left:right+1]
#         f = f[left:right+1]
#
#     return np.sqrt(((f - d) ** 2).mean())
#

#def fit_gaussian(x,y):
#    yfit = None
#    parm = None
#    pcov = None
#    try:
#        parm, pcov = curve_fit(gaussian, x, y,bounds=((-np.inf,0,-np.inf),(np.inf,np.inf,np.inf)))
#        yfit = gaussian(x,parm[0],parm[1],parm[2])
#    except:
#        log.error("Exception fitting gaussian.",exc_info=True)
#
#    return yfit,parm,pcov


class EmissionLineInfo:
    """
    mostly a container, could have pretty well just used a dictionary
    """
    def __init__(self):

        #unless noted, these are without units
        self.fit_a = None #expected in counts or in x10^-18 cgs [notice!!! -18 not -17]
        self.fit_a_err = 0.
        self.fit_x0 = None #central peak (x) position in AA
        self.fit_x0_err = 0.
        self.fit_dx0 = None #difference in fit_x0 and the target wavelength in AA, like bias: target-fit
        self.fit_sigma = 0.0
        self.fit_sigma_err = 0.0
        self.fit_y = None #y offset for the fit (essentially, the continuum estimate)
        self.fit_y_err = 0.0
        self.fit_h = None #max of the fit (the peak) #relative height
        self.fit_rh = None #fraction of fit height / raw peak height
        self.fit_rmse = -999
        self.fit_chi2 = None
        self.fit_norm_rmse = -999
        self.fit_bin_dx = 1.0 #default to 1.0 for no effect (bin-width of flux bins if flux instead of flux/dx)

        self.y_unc = None
        self.a_unc = None

        self.fit_line_flux = None #has units applied
        self.fit_line_flux_err = 0.0 #has units applied
        self.fit_continuum = None #has units applied (in whatever units the fit is made, so CAN BE x2AA, for example)
        self.fit_continuum_err = 0.0 #has units applied

        self.fit_wave = []
        self.fit_vals = []

        self.pix_size = None
        self.sn_pix = 0 #total number of pixels used to calcualte the SN (kind of like a width in pixels)

        #!! Important: raw_wave, etc is NOT of the same scale or length of fit_wave, etc
        self.raw_wave = []
        self.raw_vals = []
        self.raw_errs = []
        self.raw_h =  None
        self.raw_x0 = None

        self.line_flux = -999. #the line flux
        self.line_flux_err = 0. #the line flux
        self.cont = -999.
        self.cont_err = 0.

        self.snr = 0.0
        self.snr_err = 0.0
        self.sbr = 0.0
        self.eqw_obs = -999
        self.eqw_obs_err = 0
        self.fwhm = -999
        self.score = None
        self.raw_score = None
        self.raw_line_score = None
        self.line_score = None
        self.prob_noise = 1.0

        #MCMC errors and info
        # 3-tuples [0] = fit, [1] = fit +16%,  [2] = fit - 16% (i.e. ~ +/- 1 sd ... the interior 66%)
        self.mcmc_x0 = None #aka mu
        self.mcmc_sigma = None
        self.mcmc_a = None #area
        self.mcmc_y = None
        self.mcmc_ew_obs = None #calcuated value (using error propogation from mcmc_a and mcmc_y)
        self.mcmc_ew_obs_err = 0
        self.mcmc_snr = -1
        self.mcmc_snr_err = 0.0
        self.mcmc_dx = 1.0 #default to 1.0 so mult or div have no effect
        self.mcmc_line_flux = None #actual line_flux not amplitude (not the same if y data is flux instead of flux/dx)
        self.mcmc_continuum = None #ditto for continuum
        self.mcmc_line_flux_tuple = None #3-tuple version of mcmc_a / mcmc_dx
        self.mcmc_continuum_tuple = None #3-tuple version of mcmc_y / mcmc_dx
        self.mcmc_chi2 = None


        self.broadfit = False #set to TRUE if a broadfit conditions were applied to the fit
        self.absorber = False #set to True if this is an absorption line

        self.mcmc_plot_buffer = None
        self.gauss_plot_buffer = None

        self.noise_estimate = None
        self.noise_estimate_wave = None
        self.unique = None #is this peak unique, alone in its immediate vacinity
        self.suggested_doublet = None # if non-zero, is the rest-frame wavelength matching to an EmissionLine object
                                      # that may have been matched as a doublet (like MgII, NV, CIV, OVI)

        self.gmag = None
        self.gmag_err = None


    @property
    def w_obs(self):
        return self.fit_x0

    def unc_str(self,tuple):
        s = ""
        try:
            flux = ("%0.2g" % tuple[0]).split('e')
            unc = ("%0.2g" % (0.5 * (abs(tuple[1]) + abs(tuple[2])))).split('e')

            if len(flux) == 2:
                fcoef = float(flux[0])
                fexp = float(flux[1])
            else:
                fcoef = flux
                fexp = 0

            if len(unc) == 2:
                ucoef = float(unc[0])
                uexp = float(unc[1])
            else:
                ucoef = unc
                uexp = 0

            s = '%0.2f($\pm$%0.2f)e%d' % (fcoef, ucoef * 10 ** (uexp - fexp), fexp)
        except:
            log.warning("Exception in EmissionLineInfo::unc_str()", exc_info=False)

        return s

    def mcmc_to_fit(self,mcmc,values_units,values_dx):
        """
        translate mcmc found parms to fit_xxx parms
        :param mcmc = the MCMC object from the mcmc run
        :param values_units = 0, None, 1,  -17, -18 ... per usual
        :param values_dx = wavebin size (usuall 2AA per bin)
        :return:
        """
        try:
            if mcmc is None or mcmc.mcmc_mu is None:
                log.info("Invalid (None) mcmc passed to EmissionLineInfo::mcmc_to_fit")
                return

            # 3-tuple [0] = fit, [1] = fit +16%,  [2] = fit - 16%
            self.mcmc_x0 = mcmc.mcmc_mu
            self.mcmc_sigma = mcmc.mcmc_sigma
            self.mcmc_snr = mcmc.mcmc_snr
            self.mcmc_snr_err = mcmc.mcmc_snr_err
            
            if values_units is None:
                values_units = 1.0

            if mcmc.mcmc_A is not None:
                self.mcmc_a = np.array(mcmc.mcmc_A)
                if (values_dx is not None) and (values_dx > 0):
                    self.mcmc_dx = values_dx
                    self.mcmc_line_flux = self.mcmc_a[0]/values_dx
                    self.mcmc_line_flux_tuple =  np.array(mcmc.mcmc_A)/values_dx
            else:
                self.mcmc_a = np.array((0.,0.,0.))
                self.mcmc_line_flux = self.mcmc_a[0]
                self.mcmc_line_flux_tuple = np.array((0.,0.,0.))

            if mcmc.mcmc_y is not None:
                self.mcmc_y = np.array(mcmc.mcmc_y)
                if (values_dx is not None) and (values_dx > 0):
                    self.mcmc_dx = values_dx
                    self.mcmc_continuum = self.mcmc_y[0] / self.mcmc_dx
                    self.mcmc_continuum_tuple = np.array(mcmc.mcmc_y) / self.mcmc_dx
            else:
                self.mcmc_y = np.array((0.,0.,0.))
                self.mcmc_continuum = self.mcmc_y[0]  #/ self.mcmc_dx  just a 0 / anything
                self.mcmc_continuum_tuple = np.array(self.mcmc_y) #/ self.mcmc_dx

            if values_units < 0:
                self.mcmc_a *= 10**values_units
                self.mcmc_y *= 10**values_units
                self.mcmc_continuum  *= 10**values_units
                self.mcmc_line_flux *= 10**values_units
                try:
                    self.mcmc_line_flux_tuple *= 10 ** values_units
                    self.mcmc_continuum_tuple *= 10 ** values_units
                except:
                    log.error("*** Exception!", exc_info=True)

            # calc EW and error with approximate symmetric error on area and continuum
            if self.mcmc_y[0] != 0 and self.mcmc_a[0] != 0:
                ew = self.mcmc_a[0] / self.mcmc_y[0]
                ew_err = ew * np.sqrt((mcmc.approx_symmetric_error(self.mcmc_a) / self.mcmc_a[0]) ** 2 +
                                      (mcmc.approx_symmetric_error(self.mcmc_y) / self.mcmc_y[0]) ** 2)
            else:
                ew = self.mcmc_a[0]
                ew_err = mcmc.approx_symmetric_error(self.mcmc_a)


            #todo: could add an uncertainty on this since we have MCMC uncertainties on the parameters (mu, sigma, y)

            #def gaussian(x,x0,sigma,a=1.0,y=0.0):
            #could do this by say +/- 3 sigma (Karl uses +/- 50 AA)
            self.fit_chi2 = mcmc.mcmc_chi2

            #if recommend_mcmc: #this was a marginal LSQ fit, so replace key the "fit_xxx" with the mcmc values
            if self.mcmc_x0 is not None:
                fit_scale = 10.0**(-1*values_units)
                true_flux_conv = 1.0 #DD 2024-08-26 ... this should already be correct now so the adjustment code below is
                                     #no longer necessary?
                # true_flux_conv = G.HETDEX_FLUX_BASE_CGS * fit_scale #often there is an extra 10x to undo
                # if true_flux_conv != 0:
                #     true_flux_conv = 1.0/true_flux_conv
                # else:
                #     true_flux_conv = 1.0 #do nothing

                self.fit_x0 = self.mcmc_x0[0]
                self.fit_x0_err = 0.5*(self.mcmc_x0[1]+self.mcmc_x0[2])

                self.fit_sigma = self.mcmc_sigma[0]
                self.fit_sigma_err = 0.5*(self.mcmc_sigma[1]+self.mcmc_sigma[2])

                self.fit_a = self.mcmc_a[0] * fit_scale * true_flux_conv #net result should be in e-17
                self.fit_a_err = 0.5*(self.mcmc_a[1]+self.mcmc_a[2]) * fit_scale * true_flux_conv

                self.fit_y = self.mcmc_y[0] * fit_scale * true_flux_conv
                self.fit_y_err = 0.5*(self.mcmc_y[1]+self.mcmc_y[2])* fit_scale * true_flux_conv


                #MCMC is preferred to update key values
                self.line_flux = self.mcmc_line_flux * true_flux_conv
                self.line_flux_err = 0.5*(self.mcmc_line_flux_tuple[1]+self.mcmc_line_flux_tuple[2]) * true_flux_conv


                if self.mcmc_snr is not None and self.mcmc_snr > 0:
                    log.debug(f"MCMC SNR update: old {self.snr}+/-{self.snr_err}, new {self.mcmc_snr}+/-{self.mcmc_snr_err}")
                    self.snr = self.mcmc_snr
                    self.snr_err = self.mcmc_snr_err

                self.fwhm = self.mcmc_sigma[0]*2.355

                self.mcmc_ew_obs = [ew, ew_err, ew_err]
                #log.debug("MCMC Peak height = %f" % ())
                log.debug("MCMC calculated EW_obs for main line = %0.3g +/- %0.3g" % (ew, ew_err))
                log.debug(f"MCMC line flux = {self.mcmc_line_flux}")
                log.debug(f"MCMC line SNR = {self.mcmc_snr}")

                #log.debug(f"Calc SNR line_flux/data err: {self.line_flux/np.sqrt(np.sum(narrow_wave_errors))}")
        except:
            log.error("Exception! Exception mapping MCMC fit parms to fit_xxx parms.",exc_info=True)


    def hk_mcmc_to_fit(self,mcmc,values_units,values_dx,use_2=True):
        """

        based on mcmc_to_fit but special cased to H&K which as two mcmcs in it

        translate mcmc found parms to fit_xxx parms
        :param mcmc = the MCMC object from the mcmc run
        :param values_units = 0, None, 1,  -17, -18 ... per usual
        :param values_dx = wavebin size (usuall 2AA per bin)
        :param use_2 = use the second mcmc fit
        :return:
        """
        try:
            if mcmc is None or mcmc.mcmc_mu is None:
                log.info("Invalid (None) mcmc passed to EmissionLineInfo::mcmc_to_fit")
                return

            # 3-tuple [0] = fit, [1] = fit +16%,  [2] = fit - 16%

            #common to both 1st and 2nd mcmc
            self.mcmc_snr = mcmc.mcmc_snr
            self.mcmc_snr_err = mcmc.mcmc_snr_err

            if mcmc.mcmc_y is not None:
                self.mcmc_y = np.array(mcmc.mcmc_y)
                if (values_dx is not None) and (values_dx > 0):
                    self.mcmc_dx = values_dx
                    self.mcmc_continuum = self.mcmc_y[0] / self.mcmc_dx
                    self.mcmc_continuum_tuple = np.array(mcmc.mcmc_y) / self.mcmc_dx
            else:
                self.mcmc_y = np.array((0., 0., 0.))
                self.mcmc_continuum = self.mcmc_y[0]  # / self.mcmc_dx  just a 0 / anything
                self.mcmc_continuum_tuple = np.array(self.mcmc_y)  # / self.mcmc_dx

            if use_2:
                self.mcmc_x0 = mcmc.mcmc_mu_2
                self.mcmc_sigma = mcmc.mcmc_sigma_2


                if mcmc.mcmc_A_2 is not None:
                    self.mcmc_a = np.array(mcmc.mcmc_A_2)
                    if (values_dx is not None) and (values_dx > 0):
                        self.mcmc_dx = values_dx
                        self.mcmc_line_flux = self.mcmc_a[0] / values_dx
                        self.mcmc_line_flux_tuple = np.array(mcmc.mcmc_A_2) / values_dx
                else:
                    self.mcmc_a = np.array((0., 0., 0.))
                    self.mcmc_line_flux = self.mcmc_a[0]
                    self.mcmc_line_flux_tuple = np.array((0., 0., 0.))


                self.line_flux = mcmc.mcmc_A_2[0]/self.mcmc_dx * 10**values_units
                # NOTE 0.25* since 0.5 * for the average and /2.0 for the 2AA
                self.line_flux_err = 0.25*(mcmc.mcmc_A_2[1]+mcmc.mcmc_A_2[2])* 10**values_units

                self.fit_x0 = mcmc.mcmc_mu_2[0]
                #self.w_obs = mcmc.mcmc_mu_2[0] #not using w_obs in EmissionLineInfo object (just in EmissionLine object)


            else: #use 1st mcmc
                self.mcmc_x0 = mcmc.mcmc_mu
                self.mcmc_sigma = mcmc.mcmc_sigma

                if mcmc.mcmc_A is not None:
                    self.mcmc_a = np.array(mcmc.mcmc_A)
                    if (values_dx is not None) and (values_dx > 0):
                        self.mcmc_dx = values_dx
                        self.mcmc_line_flux = self.mcmc_a[0]/values_dx
                        self.mcmc_line_flux_tuple =  np.array(mcmc.mcmc_A)/values_dx
                else:
                    self.mcmc_a = np.array((0.,0.,0.))
                    self.mcmc_line_flux = self.mcmc_a[0]
                    self.mcmc_line_flux_tuple = np.array((0.,0.,0.))

                self.line_flux = mcmc.mcmc_A[0]/self.mcmc_dx * 10**values_units
                # NOTE 0.25* since 0.5 * for the average and /2.0 for the 2AA
                self.line_flux_err = 0.25*(mcmc.mcmc_A[1]+mcmc.mcmc_A[2])* 10**values_units

                self.fit_x0 = mcmc.mcmc_mu[0]
                #self.w_obs = hk_mcmc.mcmc_mu[0] #not using w_obs in EmissionLineInfo object (just in EmissionLine object)



            if values_units < 0:
                self.mcmc_a *= 10**values_units
                self.mcmc_y *= 10**values_units
                self.mcmc_continuum  *= 10**values_units
                self.mcmc_line_flux *= 10**values_units
                try:
                    self.mcmc_line_flux_tuple *= 10 ** values_units
                    self.mcmc_continuum_tuple *= 10 ** values_units
                except:
                    log.error("*** Exception!", exc_info=True)

            # calc EW and error with approximate symmetric error on area and continuum
            if self.mcmc_y[0] != 0 and self.mcmc_a[0] != 0:
                ew = self.mcmc_a[0] / self.mcmc_y[0]
                ew_err = ew * np.sqrt((mcmc.approx_symmetric_error(self.mcmc_a) / self.mcmc_a[0]) ** 2 +
                                      (mcmc.approx_symmetric_error(self.mcmc_y) / self.mcmc_y[0]) ** 2)
            else:
                ew = self.mcmc_a[0]
                ew_err = mcmc.approx_symmetric_error(self.mcmc_a)


            #todo: could add an uncertainty on this since we have MCMC uncertainties on the parameters (mu, sigma, y)

            #def gaussian(x,x0,sigma,a=1.0,y=0.0):
            #could do this by say +/- 3 sigma (Karl uses +/- 50 AA)
            self.fit_chi2 = mcmc.mcmc_chi2

            #if recommend_mcmc: #this was a marginal LSQ fit, so replace key the "fit_xxx" with the mcmc values
            if self.mcmc_x0 is not None:
                fit_scale = 10.0**(-1*values_units)
                true_flux_conv = G.HETDEX_FLUX_BASE_CGS * fit_scale  # often there is an extra 10x to undo
                if true_flux_conv != 0:
                    true_flux_conv = 1.0 / true_flux_conv
                else:
                    true_flux_conv = 1.0  # do nothing

                self.fit_x0 = self.mcmc_x0[0]
                self.fit_x0_err = 0.5*(self.mcmc_x0[1]+self.mcmc_x0[2])

                self.fit_sigma = self.mcmc_sigma[0]
                self.fit_sigma_err = 0.5*(self.mcmc_sigma[1]+self.mcmc_sigma[2])

                self.fit_a = self.mcmc_a[0] * fit_scale * true_flux_conv  #net result should be in e-17
                self.fit_a_err = 0.5*(self.mcmc_a[1]+self.mcmc_a[2]) * fit_scale * true_flux_conv

                #this needs to be flux level for plotting (not flux density) and in e-17 units
                self.fit_y = self.mcmc_y[0] * fit_scale * true_flux_conv
                self.fit_y_err = 0.5*(self.mcmc_y[1]+self.mcmc_y[2])* fit_scale * true_flux_conv


                #MCMC is preferred to update key values
                self.line_flux = self.mcmc_line_flux * true_flux_conv
                self.line_flux_err = 0.5*(self.mcmc_line_flux_tuple[1]+self.mcmc_line_flux_tuple[2])


                if self.mcmc_snr is not None and self.mcmc_snr > 0:
                    log.debug(f"MCMC SNR update: old {self.snr}+/-{self.snr_err}, new {self.mcmc_snr}+/-{self.mcmc_snr_err}")
                    self.snr = self.mcmc_snr
                    self.snr_err = self.mcmc_snr_err

                self.fwhm = self.mcmc_sigma[0]*2.355

                self.mcmc_ew_obs = [ew, ew_err, ew_err]
                #log.debug("MCMC Peak height = %f" % ())
                log.debug("MCMC calculated EW_obs for main line = %0.3g +/- %0.3g" % (ew, ew_err))
                log.debug(f"MCMC line flux = {self.mcmc_line_flux}")
                log.debug(f"MCMC line SNR = {self.mcmc_snr}")

                #log.debug(f"Calc SNR line_flux/data err: {self.line_flux/np.sqrt(np.sum(narrow_wave_errors))}")
        except:
            log.error("Exception! Exception mapping MCMC fit parms to fit_xxx parms.",exc_info=True)

    @property
    def flux_unc(self):
        #return a string with flux uncertainties in place
        return self.unc_str(self.mcmc_line_flux_tuple)

    @property
    def cont_unc(self):
        #return a string with flux uncertainties in place
        return self.unc_str(self.mcmc_continuum_tuple)


    @property
    def eqw_lya_unc(self):
        #return a string with flux uncertainties in place
        s = ""
        try:
            ew = self.mcmc_line_flux / self.mcmc_continuum / (self.fit_x0 / G.LyA_rest)
            a_unc = 0.5 * (abs(self.mcmc_line_flux_tuple[1])+abs(self.mcmc_line_flux_tuple[2]))
            y_unc = 0.5 * (abs(self.mcmc_continuum_tuple[1])+abs(self.mcmc_continuum_tuple[2]))

            ew_unc = abs(ew) * np.sqrt((a_unc / self.mcmc_line_flux) ** 2 + (y_unc / self.mcmc_continuum) ** 2)
            s = "%0.2g($\pm$%0.2g)" % (ew, ew_unc)
        except:
            try:
                ew = self.line_flux / self.fit_y / (self.fit_x0 / G.LyA_rest)
                a_unc = abs(self.line_flux_err)
                y_unc = abs(self.fit_y_err)

                ew_unc = abs(ew) * np.sqrt((a_unc / self.line_flux) ** 2 + (y_unc / self.fit_y) ** 2)
                s = "%0.2g($\pm$%0.2g)" % (ew, ew_unc)
            except:
                log.warning("Exception in eqw_lya_unc",exc_info=True)

        return s

    def raw_snr(self):
        """
        return the SNR (litterly as the flux values / noise values) over the 3sigma with of the line
        :return:
        """
        snr = 0.0
        try:
            idx = getnearpos(self.raw_wave,self.fit_x0) #single value version of getnearpos
            if idx is None:
                return 0.0

            width = int(self.fit_sigma * 4.0)
            left = max(0,idx-width)
            right = min(len(self.raw_wave),idx+width+1)

            signal = np.nansum(self.raw_vals[left:right])
            error = np.sqrt(np.nansum(self.raw_errs[left:right]*self.raw_errs[left:right]))

            snr = signal/error
        except:
            log.info("Exception in EmissionLineInfo::raw_snr",exc_info=True)

        return snr

    #
    # def peak_vs_continuum(self,flux,flux_err,waves,center,sigma):
    #     """
    #
    #     :param flux:
    #     :param flux_err:
    #     :param waves:
    #     :param center:
    #     :param sigma:
    #     :return:
    #     """
    #
    #     rat = 1
    #     try:
    #         i = getnearpos(waves,center)
    #
    #         left = flux[0:i-int(2*sigma)]
    #         mid = flux[i-int(2*sigma):i+int(2*sigma)]
    #         right = flux[i+int(2*sigma):]
    #
    #         cont = (np.sum(left) + np.sum(right)) / (len(left)+len(right)) #whether 2AA or 1AA does not matter since will be a ratio
    #         emis = np.sum(mid) / len(mid)
    #
    #         #for now, we won't worry about uncertainties
    #
    #         delta = abs(emis-cont)
    #         rat = delta/emis
    #
    #     except:
    #         log.debug("Exception",exc_info=True)
    #
    #     return rat


    def build(self,values_units=0,allow_broad=False, broadfit=1, override_continuum_rules=False):
        """

        :param values_units:
        :param allow_broad:  can be broad (really big sigma)
        :param broadfit:  was fit using the broad adjustment (median filter)
        :return:
        """
        if self.snr > MIN_ELI_SNR and self.fit_sigma > MIN_ELI_SIGMA:
            if self.fit_sigma is not None:
                self.fwhm = 2.355 * self.fit_sigma  # e.g. 2*sqrt(2*ln(2))* sigma

            unit = 1.0
            if self.fit_x0 is not None:

                if values_units != 0:
                    if values_units == 1:
                        unit = 1.0
                    elif values_units == -17:
                        unit = 1.0e-17
                    elif values_units == -18:
                        unit = 1.0e-18
                    else:
                        unit = 1.0
                        log.warning(("!!! Problem. Unexpected values units in EmissionLineInfo::build(): %s") % str(values_units))

                    #need these unadjusted since they inform the MCMC fit
                    # self.fit_a *= unit
                    # self.fit_y *= unit
                    # self.fit_h *= unit

                    self.line_flux = self.fit_a / self.fit_bin_dx * unit
                    self.line_flux_err = self.fit_a_err / self.fit_bin_dx * unit
                    self.fit_line_flux = self.line_flux
                    self.fit_line_flux_err = self.line_flux_err
                    self.cont = self.fit_y / self.fit_bin_dx * unit
                    self.cont_err = self.fit_y_err / self.fit_bin_dx * unit
                    self.fit_continuum = self.cont
                    self.fit_continuum_err = self.cont_err
                    #fix fit_h
                    # if (self.fit_h > 1.0) and (values_units < 0):
                    #     self.fit_h *= unit

                else: #very old deals with counts instead of flux
                    if (self.fit_a is not None):
                        #todo: HERE ... do not attempt to convert if this is already FLUX !!!
                        #todo: AND ... need to know units of flux (if passed from signal_score are in x10^-18 not -17
                        self.line_flux = self.fit_a / self.fit_bin_dx * flux_conversion(self.fit_x0)  # cgs units
                        self.fit_line_flux = self.line_flux
                        self.line_flux_err = self.fit_a_err / self.fit_bin_dx * flux_conversion(self.fit_x0)

                    if (self.fit_y is not None) and (self.fit_y > G.CONTINUUM_FLOOR_COUNTS):
                        self.cont = self.fit_y * flux_conversion(self.fit_x0)
                        self.cont_err = self.fit_y_err * flux_conversion(self.fit_x0)
                    else:
                        self.cont = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(self.fit_x0)
                    self.fit_continuum = self.cont


            if self.line_flux and self.cont:
                self.eqw_obs = self.line_flux / self.cont
                try:
                    self.eqw_obs_err = self.eqw_obs * np.sqrt( (self.line_flux_err/self.line_flux)**2 + (self.cont_err/self.cont)**2)
                except:
                    self.eqw_obs_err = 0

            #line_flux is now in erg/s/... the 1e17 scales up to reasonable numbers (easier to deal with)
            #we are technically penalizing a for large variance, but really this is only to weed out
            #low and wide fits that are probably not reliable

            #and penalize for large deviation from the highest point and the line (gauss) mean (center) (1.9 ~ pixel size)
            #self.line_score = self.snr * self.line_flux * 1e17 / (2.5 * self.fit_sigma * (1. + abs(self.fit_dx0/1.9)))

            #penalize for few pixels (translated to angstroms ... anything less than 21 AA total)
            #penalize for too narrow sigma (anything less than 1 pixel

            #the 10.0 is just to rescale ... could make 1e17 -> 1e16, but I prefer to read it this way

            if self.absorber:
                above_noise = 1.0
            else:
                above_noise = self.peak_sigma_above_noise()
                #this can fail to behave as expected for large galaxies (where much/all of IFU is covered)
                #since all the lines are represented in many fibers, that appears to be "noise"
                if above_noise is None:
                    above_noise = 1.0
                else:
                    above_noise = min(above_noise / G.MULTILINE_SCORE_NORM_ABOVE_NOISE, G.MULTILINE_SCORE_ABOVE_NOISE_MAX_BONUS)
                    # cap at 3x (so above 9x noise, we no longer graduate)
                    # that way, some hot pixel that spikes at 100x noise does not automatically get "real"
                    # but will still be throttled down due to failures with other criteria

            unique_mul = 1.0 #normal
            if (self.unique == False):# and (self.fwhm < 6.5):
                if G.CONTINUUM_RULES:
                    unique_mul = 0.25
                else:
                    #resolution is around 5.5, so if this is less than about 7AA, it could be noise?
                    unique_mul = 0.5 #knock it down (it is mixed in with others)
                    #else it is broad enough that we don't care about possible nearby lines as noise

            #def unique_peak(spec, wave, cwave, fwhm, width=10.0, frac=0.9):
            if GOOD_MAX_DX0_MULT[0] < self.fit_dx0 < GOOD_MAX_DX0_MULT[1]:
                adjusted_dx0_error = 0.0
            else:
                adjusted_dx0_error = self.fit_dx0

            if allow_broad:
                if self.gmag is not None and self.gmag < G.BROADLINE_GMAG_MAX:
                    max_fwhm = 130.0 #e.g. 55.0 sigma x 2.355 to match the 55.0 sigma limit on the xlrg type for signal score
                else:
                    max_fwhm = MAX_FWHM * 1.5
            else:
                max_fwhm = MAX_FWHM

            #pvc = self.peak_vs_continuum(self.raw_vals,self.raw_errs,self.raw_wave,self.fit_x0,self.fit_sigma)
            # if pvc < 0.5:
            #     #this is no good
            #     log.info(f"Poor distinction from local summed continuum: {pvc:0.2f}. Scoring to zero.")
            #     self.line_score = 0
            # if self.eqw_obs + self.eqw_obs_err < 1.0:
            #     log.info(f"Poor EW_obs from fit: {self.eqw_obs:0.2f} +/- {self.eqw_obs_err}. Scoring to zero.")
            #     self.line_score = 0
            try:
                if (self.fwhm is None) or (self.fwhm < max_fwhm):
                    #this MIN_HUGE_FWHM_SNR is based on the usual 2AA bins ... if this is broadfit, need to return to the
                    # usual SNR definition

                    snr_limit_score = min(self.snr,MAX_LINESCORE_SNR)

                    if (self.fwhm > MAX_NORMAL_FWHM) and \
                            ((self.snr *np.sqrt(broadfit) < MIN_HUGE_FWHM_SNR) and (self.raw_snr() < GOOD_BROADLINE_RAW_SNR)) \
                            and (self.fit_chi2 > 1.5):
                        log.debug(f"Huge fwhm {self.fwhm} with relatively poor SNR {self.snr} < required SNR {MIN_HUGE_FWHM_SNR} and "
                                  f"{self.raw_snr()} < {GOOD_BROADLINE_RAW_SNR}. "
                                  "Probably bad fit or merged lines. Rejecting score.")
                        self.line_score = 0
                        self.raw_line_score = 0
                    else:
                        #if we have a good continuum measureuse the EW
                        # if ((self.cont is not None) and (self.cont > 0)) and \
                        #     ((self.cont_err is not None) and (self.cont_err > 0)) and \
                        #         (self.cont_err/self.cont < 0.5):

                        #Why not use this? Getting a continuum fit near the emission line is hard and not as reliable as
                        # just the line flux. Causes larger swings in the line score.
                        # todo: maybe can use the wide-fit continuum estimate? or maybe only when G.CONTINUUM_RULES?
                        # if False:
                        #     log.debug("Using EW scoring ...")
                        #     self.line_score = snr_limit_score * above_noise * unique_mul * self.eqw_obs * \
                        #                       min(self.fit_sigma/self.pix_size,1.0) * \
                        #                       min((self.pix_size * self.sn_pix)/21.0,1.0) / \
                        #                       (1. + abs(adjusted_dx0_error / self.pix_size) )
                        # else:
                        #     log.debug("Using line flux scoring ...")
                        #     self.line_score = snr_limit_score * above_noise * unique_mul * self.line_flux * 1e16 * \
                        #           min(self.fit_sigma/self.pix_size,1.0) * \
                        #           min((self.pix_size * self.sn_pix)/21.0,1.0) / \
                        #            (1. + abs(adjusted_dx0_error / self.pix_size))

                        #NOTE to self: used to be * 1e17 and with a /10.0 at the bottom, which is redundant
                        # so just change to 1e16
                        self.line_score = snr_limit_score * above_noise * unique_mul * self.line_flux * 1e16 * \
                                          min(self.fit_sigma/self.pix_size,1.0) * \
                                          min((self.pix_size * self.sn_pix)/21.0,1.0) / \
                                          (1. + abs(adjusted_dx0_error / self.pix_size))

                        #self.line_score = min(G.MAXIMUM_LINE_SCORE_CAP,self.line_score) #cap at the end??

                        try:
                            if (self.snr < 8.0 and self.fit_chi2 > 3.0) or \
                               ((self.snr > 8.0) and (self.fit_chi2 > 3.0) and (self.snr/self.fit_chi2 < 3)):
                                #penalize the line score
                                # if self.snr > self.fit_chi2: #moderate a little
                                #     self.line_score = self.line_score / (self.fit_chi2 - 1.0) / 0.75
                                # else:
                                #     self.line_score /= (self.fit_chi2 - 1.0)
                                self.line_score /= (self.fit_chi2 - 1.0)
                        except:
                            self.line_score = 0
                            self.raw_line_score = 0

                        try:
                            if self.snr < 6.0 and self.fwhm > MAX_NORMAL_FWHM: #really broad and low SNR
                                rescale = self.snr / 6.0 * (MAX_NORMAL_FWHM / self.fwhm )
                                self.line_score *= rescale
                                log.info(f"Rescoring line. Very broad fwhm ({self.fwhm:0.2f}/{MAX_NORMAL_FWHM}) "
                                         f"and low SNR ({self.snr:0.2f}/6.0). Factor {rescale:0.2f}. New score = {self.line_score}")
                        except:
                            self.line_score = 0
                            self.raw_line_score = 0

                        if self.absorber:
                            self.line_score *= -1

                        #check for line in the nasty sky-lines 3545
                        if G.PENALIZE_FOR_EMISSION_IN_SKYLINE:
                            for k in SKY_LINES_DICT.keys():
                                dp = abs(self.fit_x0 - k)
                                if dp < SKY_LINES_DICT[k] and (self.fit_sigma * 1.1775 < (SKY_LINES_DICT[k]-dp)):
                                    #the emission line is inside the Sky Line window and its fit half-width (1/2 FWHM) is
                                    #less than the distance from the peak to the windows edge
                                    #then we don't trust that this is not the sky line and cut the score by half?
                                    log.info(f"Line score. Emission line in sky line window. "
                                             f"Rescoring: {self.line_score} to {self.line_score/2.0}")
                                    self.line_score /= 2.0

                        self.raw_line_score = self.line_score
                        self.line_score = min(G.MAXIMUM_LINE_SCORE_CAP, self.line_score)
                else:
                    log.debug(f"Huge fwhm {self.fwhm}, Probably bad fit or merged lines. Rejecting score.")
                    self.line_score = 0
                    self.raw_line_score = 0

            except:
                log.debug(f"Exception computing score. Rejecting score.")
                self.line_score = 0
                self.raw_line_score = 0

            if self.absorber:
                if G.MAX_SCORE_ABSORPTION_LINES or override_continuum_rules: #if not scoring absorption, should never actually get here ... this is a safety
                    # as hand-wavy correction, reduce the score as an absorber
                    # to do this correctly, need to NOT invert the values and treat as a proper absorption line
                    #   and calucate a true flux and width down from continuum
                    if ((self.cont - self.cont_err) < 2e-17) or (self.snr < 5.0): #if this has significant continuum, keep the score as is
                        if override_continuum_rules:
                            new_score = self.line_score * ABSORPTION_LINE_SCORE_SCALE_FACTOR
                        else:
                            new_score = min(G.MAX_SCORE_ABSORPTION_LINES, self.line_score * ABSORPTION_LINE_SCORE_SCALE_FACTOR)
                        log.info(f"Rescalling line_score for absorption line: {self.line_score} to {new_score} @ {self.fit_x0:0.2f}AA")
                        self.line_score = new_score
                        self.raw_line_score = self.line_score
                        self.line_score = min(G.MAXIMUM_LINE_SCORE_CAP,self.line_score)
                else:
                    log.info(f"Zeroing line_score for absorption line. {self.fit_x0:0.2f}AA")
                    self.line_score = 0.0
                    self.raw_line_score = 0
            #
            # !!! if you change this calculation, you need to re-calibrate the prob(Noise) (re-run exp_prob.py)
            # !!! and update the Solution cut criteria in global_config.py (MULTILINE_MIN_SOLUTION_SCORE, etc) and
            # !!!    in hetdex.py (DetObj::multiline_solution_score)
            # !!! and update GOOD_MIN_LINE_SCORE and PROB_NOISE_MIN_SCORE
            # !!! It is a little ciruclar as MIN scores are dependent on the results of the exp_prob.py run
            #

            self.prob_noise = self.get_prob_noise()
        else:
            self.fwhm = -999
            self.cont = -999
            self.line_flux = -999
            self.line_score = 0
            self.raw_line_score = 0


    # def calc_line_score(self):
    #
    #     return   self.snr * self.line_flux * 1e17 * \
    #              min(self.fit_sigma / self.pix_size, 1.0) * \
    #              min((self.pix_size * self.sn_pix) / 21.0, 1) / \
    #              (10.0 * (1. + abs(self.fit_dx0 / self.pix_size)))

    def get_prob_noise(self):
        MDF = False

        try:
            if (self.line_score is None) or (self.line_score < PROB_NOISE_MIN_SCORE):
                return 0.98 # really not, but we will cap it
            #if we are off the end of the scores, set to a fixed probability
            elif self.line_score > max(PROB_NOISE_LINE_SCORE) + (PROB_NOISE_LINE_SCORE[1]-PROB_NOISE_LINE_SCORE[0]):
                return PROB_NOISE_TRUNCATED #set this as the minium
            else:

                if MDF:
                    prob = 0.0
                    assumed_error_frac = 0.5
                    score_bin_width = PROB_NOISE_LINE_SCORE[1] - PROB_NOISE_LINE_SCORE[0]
                    #treat the arrays as MDF and use an error in LineFlux as a range over which to sum
                    min_score_bin = round(float(max(0, self.line_score*(1.0 - assumed_error_frac))) / score_bin_width)\
                                    * score_bin_width
                    max_score_bin = round(float( self.line_score*(1.0+assumed_error_frac)) / score_bin_width)\
                                    * score_bin_width

                    min_score_idx = np.where(PROB_NOISE_LINE_SCORE == min_score_bin)[0][0]
                    max_score_idx = np.where(PROB_NOISE_LINE_SCORE == max_score_bin)[0][0]

                    for i in range(min_score_idx, max_score_idx + 1):
                        prob += PROB_NOISE_GIVEN_SCORE[i]

                    return prob
                else:
                    return PROB_NOISE_GIVEN_SCORE[getnearpos(PROB_NOISE_LINE_SCORE,self.line_score)]
        except:
            return 1.0


    def peak_sigma_above_noise(self):
        s = None

        if (self.noise_estimate is not None) and (len(self.noise_estimate) > 0):
            try:
                noise_idx = getnearpos(self.noise_estimate_wave, self.fit_x0)
                raw_idx = getnearpos(self.raw_wave, self.fit_x0)
                s = self.raw_vals[raw_idx] / self.noise_estimate[noise_idx]
            except:
                pass

        return s

    def is_good(self,z=0.0,allow_broad=False,supporting_line=False):
        #(self.score > 0) and  #until score can be recalibrated, don't use it here
        #(self.sbr > 1.0) #not working the way I want. don't use it
        result = False

        def ratty(snr,sigma): #a bit redundant vs similar check under self.build()
            if (sigma > GOOD_BROADLINE_SIGMA) and ((snr < GOOD_BROADLINE_SNR) and (self.raw_snr() < GOOD_BROADLINE_RAW_SNR)):
                log.debug("Ratty fit on emission line.")
                return True
            else:
                return False

        if G.CONTINUUM_RULES:
            line_score_multiplier = 2.0
        else:
            line_score_multiplier = 1.0

        if supporting_line:
            good_min_line_score = GOOD_MIN_LINE_SCORE * 0.8 #supporting (or targetted fit) line is expected at this position
        else:                                               #so give it a little extra leeway
            good_min_line_score = GOOD_MIN_LINE_SCORE

        if not self.absorber and not (allow_broad or self.broadfit) and (self.fit_sigma >= LIMIT_BROAD_SIGMA) and \
                not ((self.fwhm < MAX_FWHM) and (self.snr > MIN_HUGE_FWHM_SNR)):
            log.debug(f"Line sigma {self.fit_sigma} in broad range {LIMIT_BROAD_SIGMA} and broad line not allowed.")
            return False
        elif not self.absorber and not G.CONTINUUM_RULES and (self.fit_sigma > GOOD_BROADLINE_SIGMA) and \
                (self.line_score < (1.5 * good_min_line_score * self.fit_sigma/GOOD_BROADLINE_SIGMA)):
            #and  ((self.eqw_obs - self.eqw_obs_err) < 5.0) and (abs(self.fit_dx0) > 5.0):
            log.debug(f"Line sigma {self.fit_sigma} in broad range {GOOD_BROADLINE_SIGMA} but "
                      f"line_score {self.line_score} below minumum {1.5 * good_min_line_score * self.fit_sigma/GOOD_BROADLINE_SIGMA}.")
            result = False
        # minimum to be possibly good
        elif (self.line_score >= line_score_multiplier * good_min_line_score) and (self.fit_sigma >= GOOD_MIN_SIGMA):
        #if(self.snr >= GOOD_MIN_LINE_SNR) and (self.fit_sigma >= GOOD_MIN_SIGMA):
            if not ratty(self.snr,self.fit_sigma):
                s = self.peak_sigma_above_noise()
                if (s is None) or (s > G.MULTILINE_MIN_GOOD_ABOVE_NOISE):
                    result = True
                else:
                    if (self.snr > GOOD_FULL_SNR) or (self.sbr > GOOD_MIN_SBR):
                        result = True

            #note: GOOD_MAX_DX0_MULT enforced in signal_score

        # if ((self.snr > GOOD_FULL_SNR) or ((self.snr > GOOD_MIN_SNR) and (self.sbr > GOOD_MIN_SBR))) and \
        #    (self.fit_sigma > GOOD_MIN_SIGMA) and \
        #    (self.line_flux > GOOD_MIN_LINE_FLUX) and \
        #    (self.fit_h/self.cont > GOOD_MIN_H_CONT_RATIO) and \
        #    (abs(self.fit_dx0) < GOOD_MAX_DX0):
        #         result = True

            #if self.eqw_obs/(1.+z) > GOOD_MIN_EW_REST:
            #    result =  True

        return result
 #end EmissionLineInfo Class



def local_continuum(wavelengths, values, errors, central, amplitude, sigma, cont_width=50,start=6.0 ):
    """
    return the "average" continuum near the emission/absorption line
    :param wavelengths: full width of all wavelenghts
    :param values: same scale ase used to fit the amplitdued and sigma
    :param errors: ditto
    :param central: central (mu) of the line
    :param amplitude: amplitude or max height at the peak (same units as values)
    :param sigma: fit sigma
    :param cont_width:  width in AA to sample spectrum for continuum
    :param start:  number of sigma away from the central to start the cont_width
    :return:
    """

    avg = None #average
    std = None #sigma
    try:
        #just assume 2.0AA pixel scale
        numpix = int(cont_width/2.0)
        idx1 = getnearpos(wavelengths,central-start*sigma) #right end of the blueward section
        idx2 = getnearpos(wavelengths,central+start*sigma) #left start of the redward section

        idx0 = max(0,idx1-numpix) #right start of the blueward section
        idx3 = min(len(wavelengths)-1,idx2+numpix) #left end of the redward section

        #not worry about the errors for now
        avg = np.nanmean(np.concatenate( (values[idx0:idx1+1], values[idx2:idx3+1])))
        std = np.nanstd(np.concatenate( (values[idx0:idx1+1], values[idx2:idx3+1])))
    except:
        log.warning(f"Exception! Exception in local_continuum().",exc_info=True)

    return avg,std

#really should change this to use kwargs
def signal_score(wavelengths,values,errors,central,central_z = 0.0, spectrum=None,values_units=0, sbr=None,
                 min_sigma=None,show_plot=False,plot_id=None,plot_path=None,do_mcmc=False,absorber=False,
                 force_score=False,values_dx=G.FLUX_WAVEBIN_WIDTH,allow_broad=False,broadfit=1,relax_fit=False,
                 min_fit_sigma=1.0,test_solution=None,wave_fit_side_aa=GAUSS_FIT_AA_RANGE, fit_range_AA=None,
                 quick_fit=False,spec_obj=None,targetted_fit=False,max_sigma=None):
    """

    :param wavelengths:
    :param values:
    :param errors:
    :param central:
    :param central_z:
    :param spectrum:
    :param values_units:
    :param sbr:
    :param min_sigma: #minimum allowed to be "real"
    :param min_fit_sigma: "minimum end past to fitting ... could be larger or smaller than just min_sigma
    :param show_plot:
    :param plot_id:
    :param plot_path:
    :param do_mcmc:
    :param absorber:
    :param force_score:
    :param values_dx:
    :param allow_broad:
    :param broadfit: (median filter size used to smooth for a broadfit) 1 = no filter (or a bin of 1 which is no filter)
    :return:
    """

    #values_dx is the bin width for the values if multiplied out (s|t) values are flux and not flux/dx
    #   by default, Karl's data is on a 2.0 AA bin width

    #error on the wavelength of the possible line depends on the redshift and its error and the wavelength itself
    #i.e. wavelength error = wavelength / (1+z + error)  - wavelength / (1+z - error)
    # want that by 1/2 (so +/- error from center ... note: not quite symmetric, but close enough over small delta)
    # and want in pixels so divide by pix_size
    #todo: note error itself should be a function of z ... otherwise, error is constant and as z increases, the
    #todo:   pix_error then decreases
    #however, this error is ON TOP of the wavelength measurement error, assumed to be +/- 1 pixel?
    def pix_error(z,wavelength,error=0.001, pix_size= 2.0):
        """
        NOTICE ALWAYS RETURNING ZERO RIGHT NOW
        :param z:
        :param wavelength:
        :param error:
        :param pix_size:
        :return:  error in measurement IN PIXELS (not AA)
        """
        return 0.0

        try:
            e =  0.5 * wavelength * (2. * error / ((1.+z)**2 - error**2)) / pix_size
        except:
            e = 0.0
            log.warning("Invalid pix_error in spectrum::signal_score",exc_info=True)

        return e


    if (wavelengths is None) or (values is None) or (len(wavelengths)==0) or (len(values)==0):
        log.warning("Zero length (or None) spectrum passed to spectrum::signal_score().")
        return None

    if min_sigma is None:
        if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None:  #explicitly defined by user on command line
            min_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MIN
        else: #otherwise, use the elixer default
            min_sigma = GAUSS_FIT_MIN_SIGMA

    if max_sigma is None:
        if G.LIMIT_GAUSS_FIT_SIGMA_MAX is not None: #explicitly defined by user on command line
            max_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MAX
        else: #otherwise, use the elixer default
            max_sigma = 999.0

    recommend_mcmc = False #internal trigger ... may want MCMC even if do_mcmc is false
    accept_fit = False
    if do_mcmc:
        forced_mcmc = True #do_mcmc is reused a little later for another case, so this is to disambiguate
    else:
        forced_mcmc = False
    #if values_are_flux:
    #    # assumed then to be in cgs units of x10^-17 as per typical HETDEX values
    #    # !!!! reminder, do NOT do values *= 10.0  ... that is an in place operation and overwrites the original
    #    values = values * 10.0  # put in units of 10^-18 so they pretty much match up with counts

    err_units = values_units #assumed to be in the same units
    values, values_units = norm_values(values,values_units)
    if errors is not None and (len(errors) == len(values)):
        errors, err_units = norm_values(errors,err_units)
       # errors /= 10.0 #todo: something weird here with the latest update ... seem to be off by 10.0

    #sbr signal to background ratio
    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix

    #if near a peak we already found, nudge to align
    if isinstance(spectrum,Spectrum):
        w = spectrum.is_near_a_peak(central,pix_size)
        if w:
            central = w


    peak_pos = getnearpos(wavelengths, central)

    #####################
    #these will be recomputed  AFTER the quick fits selected the "best", for now default values are used
    ####################

    # want +/- 20 angstroms in pixel units (maybe should be 50 AA? ... HETDEX uses 50AA)
    wave_side_pix = int(round(wave_fit_side_aa / pix_size))  # pixels
    #1.5 seems to be good multiplier ... 2.0 is a bit too much;
    # 1.0 is not bad, but occasionally miss something by just a bit

    if fit_range_AA is None:
        fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
        #fit_range_AA = GAUSS_FIT_PIX_ERROR * pix_size #1.0  # peak must fit to within +/- fit_range AA
                                  # this allows room to fit, but will enforce +/- pix_size after
    #num_of_sigma = 3.0  # number of sigma to include on either side of the central peak to estimate noise

    len_array = len(wavelengths)
    idx = getnearpos(wavelengths,central)
    min_idx = max(0,idx-wave_side_pix)
    max_idx = min(len_array,idx+wave_side_pix)
    wave_x = wavelengths[min_idx:max_idx+1]
    wave_counts = values[min_idx:max_idx+1]
    if (errors is not None) and (len(errors) == len(wavelengths)):
        wave_errors = errors[min_idx:max_idx+1]
        #replace any 0 with 1
        wave_errors[np.where(wave_errors == 0)] = 1
        wave_err_sigma = 1. / (wave_errors * wave_errors) #double checked and this is correct (assuming errors is +/- as expected)
        #as a reminder, if the errors are all the same, then it does not matter what they are, it reduces to the standard
        #arithmetic mean :  Sum 1 to N (x_n**2, sigma_n**2) / (Sum 1 to N (1/sigma_n**2) ==> 1/N * Sum(x_n**2)
        # since sigma_n (as a constant) divides out
    else:
        wave_errors = None
        wave_err_sigma = None

    if False: #do I want to use a more narrow range for the gaussian fit? still uses the wider range for RMSE
        min_idx = max(0, idx - wave_side_pix/2)
        max_idx = min(len_array, idx + wave_side_pix/2)
        narrow_wave_x = wavelengths[min_idx:max_idx+1]
        narrow_wave_counts = values[min_idx:max_idx + 1]
        if (wave_errors is not None):
            narrow_wave_errors = wave_errors[min_idx:max_idx + 1]
            narrow_wave_err_sigma =  wave_err_sigma[min_idx:max_idx + 1]
        else:
            narrow_wave_err_sigma = None
            narrow_wave_errors = None
    else:
        narrow_wave_x = wave_x
        narrow_wave_counts = wave_counts
        narrow_wave_errors = wave_errors
        narrow_wave_err_sigma = wave_err_sigma

    #blunt very negative values
    #wave_counts = np.clip(wave_counts,0.0,np.inf)

    xfit = np.linspace(wave_x[0], wave_x[-1], 1000) #range over which to plot the gaussian equation

    try:
        # find the highest point in the raw data inside the range we are allowing for the line center fit
        dpix = int(round(fit_range_AA / pix_size))
        left = max(0,peak_pos-dpix)
        right = min(len(values),peak_pos+dpix)
        if absorber:
            raw_peak = min(values[left:right+1])
            #can be negative (should not be, except if saturated and with error pushes negative)
        else:
            raw_peak = max(values[left:right+1])
            if raw_peak <= 0:
                if not quick_fit:
                    log.warning("Spectrum::signal_score invalid raw peak %f" %raw_peak)
                return None
    except:
        #this can fail if on very edge, but if so, we would not use it anyway
        log.debug("Raw Peak value failure (1) for wavelength (%f) at index (%d). Cannot fit to gaussian. " %(central,peak_pos))
        return None

    fit_peak = None

    eli = EmissionLineInfo()
    eli.absorber = absorber
    eli.pix_size = pix_size
    eli.fit_bin_dx = values_dx
    num_sn_pix = 0

    bad_curve_fit = False
    if allow_broad:
        max_fit_sigma = GAUSS_FIT_MAX_SIGMA *1.5 + 1.0 # allow a model fit bigger than what is actually acceptable
    else:                                              # so can throw out reall poor broad fits
        max_fit_sigma = GAUSS_FIT_MAX_SIGMA + 1.0


    #this is lazy and should be refactored since this is already known else where, but want some indications of mag
    gmag = None
    gmag_err = None
    if spec_obj is not None:
        try:
            gmag = spec_obj.gmag
            gmag_err = spec_obj.gmag_unc
            if gmag <= 0:
                gmag = None
        except:
            pass

    try:

        # parm[0] = central point (x in the call), parm[1] = sigma, parm[2] = 'a' multiplier (happens to also be area)
        # parm[3] = y offset (e.g. the "continuum" level)
        #get the gaussian for the more central part, but use that to get noise from wider range
        #sigma lower limit at 0.5 (could make more like pixel_size / 4.0 or so, but probabaly should not depend on that
        # the minimum size is in angstroms anyway, not pixels, and < 0.5 is awfully narrow to be real)
        # instrument resolution ~ 1.9AA/pix (dispersion around 2.2?)

        #if narrow_wave_err_sigma is None:
        #    print("**** NO UNCERTAINTIES ****")
        #   log.warning("**** NO UNCERTAINTIES ****")


        #old way
        # if absorber: #area is to be negative
        #     parm, pcov = curve_fit(gaussian, np.float64(narrow_wave_x), np.float64(narrow_wave_counts),
        #                            p0=(central,max(min_fit_sigma,1.5),-1.0,0.0),
        #                            bounds=((central-fit_range_AA, min_fit_sigma, -1e5, -100.0),
        #                                    (central+fit_range_AA, max_fit_sigma, 0.0, 1e4)),
        #                            #sigma=1./(narrow_wave_errors*narrow_wave_errors)
        #                            sigma=narrow_wave_err_sigma#, #handles the 1./(err*err)
        #                            #note: if sigma == None, then curve_fit uses array of all 1.0
        #                            #method='trf'
        #                            )
        #     snr, chi2, ew , parm, pcov, model =  SU.quick_fit(wavelengths,values,errors,central,
        #                                                      fit_range_AA,wave_fit_side_aa,min_fit_sigma,max_fit_sigma,
        #                                                       absorber=True)
        # else:
        #     # parm, pcov = curve_fit(gaussian, np.float64(narrow_wave_x), np.float64(narrow_wave_counts),
        #     #                     p0=(central,max(min_fit_sigma,1.5),1.0,0.0),
        #     #                     bounds=((central-fit_range_AA, min_fit_sigma, 0.0, -100.0),
        #     #                             (central+fit_range_AA, max_fit_sigma, 1e5, 1e4)),
        #     #                     #sigma=1./(narrow_wave_errors*narrow_wave_errors)
        #     #                     sigma=narrow_wave_err_sigma#, #handles the 1./(err*err)
        #     #                    #note: if sigma == None, then curve_fit uses array of all 1.0
        #     #                    #method='trf'
        #     #                    )
        #
        #     #alternate
        #     # parm, pcov = curve_fit(gaussian, np.float64(narrow_wave_x), np.float64(narrow_wave_counts),
        #     #                     p0=(central,max(min_fit_sigma,1.5),1.0,0.0),
        #     #                     bounds=((central-fit_range_AA, min_fit_sigma, 0.0,
        #     #                                             min(narrow_wave_counts)-max(narrow_wave_counts)),
        #     #                             (central+fit_range_AA, max_fit_sigma,
        #     #                                     max(narrow_wave_counts)*len(narrow_wave_counts),
        #     #                                     max(narrow_wave_counts)*1.5)),
        #     #                     #sigma=1./(narrow_wave_errors*narrow_wave_errors)
        #     #                     sigma=narrow_wave_err_sigma#, #handles the 1./(err*err)
        #     #                    #note: if sigma == None, then curve_fit uses array of all 1.0
        #     #                    #method='trf'
        #     #                    )
        #

        #
        # newer way 0,999,0,[0,0,0,0],np.zeros((4,4)),np.zeros(len(waves))
        # loop over a few ranges and choose the "best"

        if absorber:
            fit_dict_array = [#much larger than 20AA side can miss H&k
                               {"type": "small", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 20.0,
                               "min_fit_sigma": 1.5, "max_fit_sigma": 5.5, "min_snr":  6.0,
                               "max_gmag": 99, "min_gmag": 0,
                               "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None, "score": 0},

                              {"type": "medium", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 40.0,
                               "min_fit_sigma": 2.0, "max_fit_sigma": 8.5, "min_snr": 6.0, #higher SNR requireed for absorption
                               "max_gmag": 23.0, "min_gmag": 0.0,
                               "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None, "score": 0},

                              # {"type": "HDstd", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 50.0,
                              #  "min_fit_sigma": 1.7, "max_fit_sigma": 8.5, "min_snr":3.5 if targetted_fit else 4.0,
                              #  "max_gmag":99,"min_gmag": 0.0,
                              #  "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None, "score": 0},
                            ]

        else:
            if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None: #the user specified something
                #force to the users sigmas, based on the "medium" settings
                fit_dict_array = [{"type": "user", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 40.0,
                                       "min_fit_sigma": min_sigma, "max_fit_sigma": max_sigma,
                                       "min_snr": 3.0 if targetted_fit else 4.0,
                                       "max_gmag": G.BROADLINE_GMAG_MAX, "min_gmag": 0.0,
                                       "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None,
                                       "score": 0}]
            else:
                fit_dict_array = [ {"type": "small", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 20.0,
                            "min_fit_sigma": 1.5, "max_fit_sigma": 5.5, "min_snr":3.5 if targetted_fit else 4.0,
                            "max_gmag":99, "min_gmag": G.BROADLINE_GMAG_MAX,
                            "snr":0, "chi2":999, "ew":0, "parm":[], "pcov":[], "model":None, "score":0},

                           {"type": "medium", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 40.0,
                            "min_fit_sigma": 2.0, "max_fit_sigma": 8.5, "min_snr":3.8 if targetted_fit else 4.3,
                            "max_gmag":99, "min_gmag": 0.0,
                            "snr":0, "chi2":999, "ew":0, "parm":[], "pcov":[], "model":None, "score":0},

                           # {"type": "HDstd", "fit_range_AA": fit_range_AA, "wave_fit_side_aa": 50.0,
                           #  "min_fit_sigma": 1.7, "max_fit_sigma": 8.5, "min_snr":3.5 if targetted_fit else 4.0,
                           #  "max_gmag":99,"min_gmag": 0.0,
                           #  "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None, "score": 0},

                           {"type": "large", "fit_range_AA": fit_range_AA*2.0, "wave_fit_side_aa": 80.0,
                            "min_fit_sigma": 4.0, "max_fit_sigma": 25.0, "min_snr":9.0 if targetted_fit else 10.0,
                            "max_gmag":99,"min_gmag": 0.0,
                            "snr":0, "chi2":999, "ew":0, "parm":[], "pcov":[], "model":None, "score":0},

                           {"type": "xlrg", "fit_range_AA": fit_range_AA*4.0, "wave_fit_side_aa": 150.0,
                            "min_fit_sigma": 15.0, "max_fit_sigma": 55.0, "min_snr":13.0 if targetted_fit else 15.0,
                            "max_gmag":G.BROADLINE_GMAG_MAX, "min_gmag": 0.0,
                            "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None, "score": 0},


                        ]
                if 55.0 < max_sigma < 80.0:
                    fit_dict_array.append(
                        {"type": "xxlrg", "fit_range_AA": fit_range_AA * 10.0, "wave_fit_side_aa": max_sigma*3.3,
                         "min_fit_sigma": max_sigma/3.0, "max_fit_sigma": max_sigma,
                         "min_snr": 20.0 if targetted_fit else 30.0,
                         "max_gmag": G.BROADLINE_GMAG_MAX, "min_gmag": 0.0,
                         "snr": 0, "chi2": 999, "ew": 0, "parm": [], "pcov": [], "model": None, "score": 0},

                    )

        #track and select the "best" to process below
        max_tested_sigma = GAUSS_FIT_MAX_SIGMA #if xlarge is allowed, this can be exceeded

        for i,fd in enumerate(fit_dict_array):
            #if the sigma limit or the gmag is out of range for the particular "type" then skip it
            # yes, I want max vs max, not min vs max
            if fd["type"] != "user" and ((fd["max_fit_sigma"] <= min_sigma) or (fd["max_fit_sigma"] > max_sigma) or \
                    (gmag is not None and ( gmag > fd["max_gmag"] or gmag < fd["min_gmag"]))):
                fd["parm"] = [central,0,0,0]
                fd["pcov"] = np.zeros((4, 4))
                fd["score"] = -1
                #if this is explicitly a broadfit, don't bother with the narrowest limits
            else:

                max_tested_sigma = max (max_tested_sigma,fd["max_fit_sigma"])
                fd["snr"], fd["chi2"], fd["ew"], fd["parm"], fd["pcov"], fd["model"] = SU.quick_fit(
                                                            wavelengths, values, errors, central,
                                                            fd["fit_range_AA"], fd["wave_fit_side_aa"],
                                                            fd["min_fit_sigma"], fd["max_fit_sigma"],
                                                            absorber)

                #enforce a minimum snr UNLESS we are going to force an MCMC
                if not forced_mcmc and fd["snr"] < fd["min_snr"]:
                    fd["score"] = -1
                    continue

                fd["score"] = SU.quick_linescore(fd["snr"],fd["chi2"],fd["parm"][1],fd["ew"],fd["wave_fit_side_aa"],
                                                 min_sigma = fd["min_fit_sigma"], max_sigma=fd["max_fit_sigma"],
                                                 absorber=absorber)

                #may reject if at the maximum sigma
                #does NOT apply to absorbers
                if not absorber:
                    if np.isclose(fd["parm"][1],fd["max_fit_sigma"],atol=1e-2) or (fd["parm"][1] > fd["max_fit_sigma"]):
                        if fd["chi2"] > 3.5:
                            fd["score"] = 0
                        #if fd["snr"] < 10.0: #todo: maybe also check score?
                        #    fd["score"] = 0
                        else:
                            try:  # add a new key for use later when filtering out emission that is just continuum between absorbers
                                fd["next_max_sigma"] = fit_dict_array[i + 1]["max_fit_sigma"]
                            except:
                                pass

        fd_idx = None

        #get the largest line score of lines with chi2 < 2.5
        try:
            fit_dict_array = np.array(fit_dict_array)
            max_chi2 = 3.5 if absorber else 2.5
            sel_ls = [ (fd["chi2"] <= max_chi2) and (fd["fit_sigma"]) for fd in fit_dict_array]
            if np.count_nonzero(sel_ls) > 0:
                fd_idx = np.nanargmax([fd["score"] for fd in fit_dict_array[sel_ls]])
                fd_idx = np.argwhere([fd['type'] == fit_dict_array[sel_ls][fd_idx]['type'] for fd in fit_dict_array])[0][0]
        except:
            fd_idx = None

        #if that did not yeild a result, then select line scores above 10 or 15 and then the lowest chi2 in that set ... if there are none, then
        #choose the largest line score
        if fd_idx is None or fit_dict_array[fd_idx]['score'] < 10:
            try:
                fit_dict_array = np.array(fit_dict_array)
                sel_ls = [fd["score"] >= 10 for fd in fit_dict_array]
                if np.count_nonzero(sel_ls) > 0:
                    fd_idx = np.nanargmin([fd["chi2"] for fd in fit_dict_array[sel_ls]])
                    fd_idx = np.argwhere([fd['type'] == fit_dict_array[sel_ls][fd_idx]['type'] for fd in fit_dict_array])[0][0]
            except:
                fd_idx = None

        if fd_idx is None:
            try:
                fd_idx = np.nanargmax([fd["score"] for fd in fit_dict_array])
            except:
                pass

        if fit_dict_array[fd_idx]['score'] <=0 and not (targetted_fit and forced_mcmc):
            #nothing at all here
            return None

        #update inputs
        fit_range_AA = fit_dict_array[fd_idx]["fit_range_AA"]

        #however, we DO want a small preference for medium over small, so if the scores are close, then
        #bump medium?

        log.debug(f"Best fit ({central:0.1f}): ({fit_dict_array[fd_idx]['parm'][0]:0.1f}) "
                  f"{fit_dict_array[fd_idx]['type']}, quick score {fit_dict_array[fd_idx]['score']:0.1f}, "
                  f"snr {fit_dict_array[fd_idx]['snr']:0.1f}, chi2 {fit_dict_array[fd_idx]['chi2']:0.1f}, "
                  f"sigma {fit_dict_array[fd_idx]['parm'][1]:0.1f}, area {fit_dict_array[fd_idx]['parm'][2]:0.1f}, "
                  f"ew {fit_dict_array[fd_idx]['ew']:0.1f}, cont {fit_dict_array[fd_idx]['parm'][3]:0.1f} ")

        #EXTRA logging for debugging
        # for idx in range(len(fit_dict_array)):
        #     try:
        #         log.debug(f"*** All fit:  ({fit_dict_array[fd_idx]['parm'][0]:0.1f}) "
        #                   f"{fit_dict_array[idx]['type']}, quick score {fit_dict_array[idx]['score']:0.2f}, "
        #                   f"snr {fit_dict_array[idx]['snr']:0.2f}, chi2 {fit_dict_array[idx]['chi2']:0.2f}, "
        #                   f"sigma {fit_dict_array[idx]['parm'][1]:0.2f}, area = {fit_dict_array[idx]['parm'][2]:0.2f}, "
        #                   f"ew {fit_dict_array[idx]['ew']:0.2f}")
        #     except:
        #         log.debug(f"***** idx {idx} *****")


        #print(f" *** Selected: {fit_dict_array[fd_idx]['type']}")
        parm = fit_dict_array[fd_idx]["parm"]
        pcov = fit_dict_array[fd_idx]["pcov"]
        wave_fit_side_aa = fit_dict_array[fd_idx]["wave_fit_side_aa"]
        min_fit_sigma =  fit_dict_array[fd_idx]["min_fit_sigma"]
        max_fit_sigma =  fit_dict_array[fd_idx]["max_fit_sigma"]

        perr = np.sqrt(np.diag(pcov)) #1-sigma level errors on the fitted parameters
        #e.g. flux = a = parm[2]   +/- perr[2]*num_of_sigma_confidence
        #where num_of_sigma_confidence ~ at a 5 sigma confidence, then *5 ... at 3 sigma, *3

        try:# add a new parameter for use later when filtering out emission that is just continuum between absorbers
            if "next_max_sigma" in fit_dict_array[fd_idx].keys():
                eli.next_max_sigma = fit_dict_array[fd_idx]["next_max_sigma"]
        except:
            pass

        if quick_fit:
            eli.fit_x0 = parm[0]
            eli.fit_x0_err = perr[0]
            eli.fit_dx0 = central -  parm[0]
            eli.fit_sigma = parm[1]  # units of AA not pixels
            eli.fit_sigma_err = perr[1]
            eli.fit_a = parm[2]  # this is an AREA so there are 2 powers of 10.0 in it (hx10 * wx10) if in e-18 units
            eli.fit_a_err = perr[2]
            eli.fit_y = parm[3]
            eli.fit_y_err = perr[3]
            eli.line_score = fit_dict_array[fd_idx]["score"] #just for now, will get replaced with actual line_score
            return eli

        ######################
        #refigure top values
        ######################

        # want +/- 20 angstroms in pixel units (maybe should be 50 AA? ... HETDEX uses 50AA)
        wave_side_pix = int(round(wave_fit_side_aa / pix_size))  # pixels
        # 1.5 seems to be good multiplier ... 2.0 is a bit too much;
        # 1.0 is not bad, but occasionally miss something by just a bit

        #changed: this is set above where the best fit is made, based on the small, medium, large fit
        #fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
        # fit_range_AA = GAUSS_FIT_PIX_ERROR * pix_size #1.0  # peak must fit to within +/- fit_range AA
        # this allows room to fit, but will enforce +/- pix_size after
        # num_of_sigma = 3.0  # number of sigma to include on either side of the central peak to estimate noise

        len_array = len(wavelengths)
        idx = getnearpos(wavelengths, central)
        min_idx = max(0, idx - wave_side_pix)
        max_idx = min(len_array, idx + wave_side_pix)
        wave_x = wavelengths[min_idx:max_idx + 1]
        wave_counts = values[min_idx:max_idx + 1]
        if (errors is not None) and (len(errors) == len(wavelengths)):
            wave_errors = errors[min_idx:max_idx + 1]
            # replace any 0 with 1
            wave_errors[np.where(wave_errors == 0)] = 1
            wave_err_sigma = 1. / (
                        wave_errors * wave_errors)  # double checked and this is correct (assuming errors is +/- as expected)
            # as a reminder, if the errors are all the same, then it does not matter what they are, it reduces to the standard
            # arithmetic mean :  Sum 1 to N (x_n**2, sigma_n**2) / (Sum 1 to N (1/sigma_n**2) ==> 1/N * Sum(x_n**2)
            # since sigma_n (as a constant) divides out
        else:
            wave_errors = None
            wave_err_sigma = None

        narrow_wave_x = wave_x
        narrow_wave_counts = wave_counts
        narrow_wave_errors = wave_errors
        narrow_wave_err_sigma = wave_err_sigma

        # blunt very negative values
        # wave_counts = np.clip(wave_counts,0.0,np.inf)

        xfit = np.linspace(wave_x[0], wave_x[-1], 1000)  # range over which to plot the gaussian equation

        try:
            # find the highest point in the raw data inside the range we are allowing for the line center fit
            dpix = int(round(fit_range_AA / pix_size))
            if absorber:
                raw_peak = min(values[peak_pos - dpix:peak_pos + dpix + 1])
                # can be negative (should not be, except if saturated and with error pushes negative)
            else:
                raw_peak = max(values[peak_pos - dpix:peak_pos + dpix + 1])
                if raw_peak <= 0:
                    log.warning("Spectrum::signal_score invalid raw peak %f" % raw_peak)
                    return None
        except:
            # this can fail if on very edge, but if so, we would not use it anyway
            log.debug("Raw Peak value failure (2) for wavelength (%f) at index (%d). Cannot fit to gaussian. " % (
            central, peak_pos))
            return None




        try:
            if not np.any(pcov): #all zeros ... something wrong
                log.info("Something very wrong with curve_fit")
                bad_curve_fit = True
                do_mcmc = True
        except:
            pass

        #gaussian(x, x0, sigma, a=1.0, y=0.0):
        eli.fit_vals = gaussian(xfit, parm[0], parm[1], parm[2], parm[3])
        eli.fit_wave = xfit.copy()
        eli.raw_vals = wave_counts[:]
        eli.raw_wave = wave_x[:]
        if wave_errors is not None:
            eli.raw_errs = wave_errors[:]

        if spectrum is not None:
            try:
                #noise estimate from Spetrcum is from HDF5 and is in 10^-17 units
                if spectrum.noise_estimate is not None:
                    if values_units == -18:
                        m = 10.0
                    else:
                        m = 1.0

                    eli.noise_estimate = spectrum.noise_estimate[:] * m
                if spectrum.noise_estimate_wave is not None:
                    eli.noise_estimate_wave = spectrum.noise_estimate_wave[:]
            except:
                pass

        #matches up with the raw data scale so can do RMSE
        rms_wave = gaussian(wave_x, parm[0], parm[1], parm[2], parm[3])

        eli.fit_x0 = parm[0]
        eli.fit_x0_err = perr[0]
        eli.fit_dx0 = central - eli.fit_x0
        if absorber:
            scaled_fit_h = min(eli.fit_vals)
        else:
            scaled_fit_h = max(eli.fit_vals)
        eli.fit_h = scaled_fit_h
        eli.fit_rh = eli.fit_h / raw_peak
        eli.fit_sigma = parm[1] #units of AA not pixels
        eli.fit_sigma_err = perr[1]
        eli.fit_a = parm[2] #this is an AREA so there are 2 powers of 10.0 in it (hx10 * wx10) if in e-18 units
        eli.fit_a_err = perr[2]
        eli.fit_y = parm[3]
        eli.fit_y_err = perr[3]

        if (values_dx is not None) and (values_dx > 0):
            eli.fit_bin_dx = values_dx
            eli.fit_line_flux = eli.fit_a / eli.fit_bin_dx
            eli.fit_line_flux_err = eli.fit_a_err/eli.fit_bin_dx #assumes no error in fit_bin_dx (and that is true)
            #what about eli.cont
            eli.fit_continuum = eli.fit_y / eli.fit_bin_dx
            eli.fit_continuum_err = eli.fit_y_err / eli.fit_bin_dx
        else:
            eli.fit_line_flux = eli.fit_a
            eli.fit_line_flux_err = eli.fit_a_err
            eli.fit_continuum = eli.fit_y
            eli.fit_continuum_err = eli.fit_y_err

        if gmag is not None:
            eli.gmag = gmag
            eli.gmag_err = gmag_err

        raw_idx = getnearpos(eli.raw_wave, eli.fit_x0)
        if raw_idx < 3:
            raw_idx = 3

        if raw_idx > len(eli.raw_vals)-4:
            raw_idx = len(eli.raw_vals)-4
        #if still out of range, will throw a trapped exception ... we can't use this data anyway
        if absorber:
            eli.raw_h = min(eli.raw_vals[raw_idx - 3:raw_idx + 4])
        else:
            eli.raw_h = max(eli.raw_vals[raw_idx - 3:raw_idx + 4])
        eli.raw_x0 = eli.raw_wave[getnearpos(eli.raw_vals, eli.raw_h)]

        if absorber:
            fit_peak = min(eli.fit_vals)
        else:
            fit_peak = max(eli.fit_vals)

        if absorber:
            peak_fit_mult = 0.33
        else:
            peak_fit_mult = 0.25

        if relax_fit:
            peak_fit_mult = 0.5

        captured_peak = True
        # if absorber:
        #     if( abs(fit_peak - raw_peak) < (raw_peak * peak_fit_mult) ):
        #         captured_peak = False
        #         log.debug("Failed to capture peak: raw = %f , fit = %f, frac = %0.2f" % (raw_peak, fit_peak,
        #                                                                                  abs(raw_peak - fit_peak) / raw_peak))
        # else:

        if not do_mcmc: #only bother to check peak caputre if we are not explicitly directed to run mcmc
            peak_fit = abs(fit_peak - raw_peak)
            if ( peak_fit > (raw_peak * peak_fit_mult) ):
                if peak_fit > (raw_peak * 0.5) and not \
                        (absorber and ((3930 < central < 3940) or (3962 < central < 3972))): #possible H&K low-z (star)
                    captured_peak = False
                    #if (abs(raw_peak - fit_peak) / raw_peak > 0.2):  # didn't capture the peak ... bad, don't calculate anything else
                        #log.warning("Failed to capture peak")
                    log.debug("Failed to capture peak: raw = %f , fit = %f, frac = %0.2f" % (raw_peak, fit_peak,
                                                                                             peak_fit / raw_peak))
                else: #missed the tighter constratint, but passed relax. Need to run MCMC
                    log.debug("Poor capture peak. MCMC recommended: raw = %f , fit = %f, frac = %0.2f" % (raw_peak, fit_peak,
                                                                                        peak_fit / raw_peak))

                    #print("***** TURNED OFF recommend_mcmc ******")
                    recommend_mcmc = True
                    bad_curve_fit  = True

        if captured_peak:
            #check the dx0

            p_err = pix_error(central_z,eli.fit_x0,pix_size=pix_size)

            #old ... here GOOD_MAX_DXO_MULT was in pixel multiples
            #if (abs(eli.fit_dx0) > (GOOD_MAX_DX0_MULT * pix_size + p_err)):
            #    log.debug("Failed to capture peak: dx0 = %f, pix_size = %f, wavelength = %f, pix_z_err = %f"
            #              % (eli.fit_dx0,pix_size, eli.fit_x0,p_err))


            #GOOD_MAX_DX0_MULT[0] is the negative direction, [1] is the positive direction
            #but fit_dx0 is defined as the expected position - fit position, so a positive fit_dx0 has the fit position
            # short (left) of the expected position, which corresponds to the negative offset allowance
            # if (eli.fit_dx0 > (GOOD_MAX_DX0_MULT[1] + p_err * pix_size)) or \
            #    (eli.fit_dx0 < (GOOD_MAX_DX0_MULT[0] + p_err * pix_size)):
            #     log.debug("Failed to capture peak: dx0 = %f, pix_size = %f, wavelength = %f, pix_z_err = %f"
            #               % (eli.fit_dx0,pix_size, eli.fit_x0,p_err))
            if (eli.fit_dx0 > (fit_range_AA + p_err * pix_size)) or \
               (eli.fit_dx0 < (-1 * fit_range_AA + p_err * pix_size)):

                log.debug("Failed to capture peak: dx0 = %f, pix_size = %f, wavelength = %f, pix_z_err = %f"
                          % (eli.fit_dx0,pix_size, eli.fit_x0,p_err))


            else:
                accept_fit = True
                log.debug("Success: captured peak: raw = %f , fit = %f, frac = %0.2f"
                          % (raw_peak, fit_peak, abs(raw_peak - fit_peak) / raw_peak))

                #num_sn_pix = int(round(max(GAUSS_SNR_SIGMA * eli.fit_sigma, GAUSS_SNR_NUM_AA)/pix_size)) #half-width in AA
                num_sn_pix = int(round(max(GAUSS_SNR_SIGMA * eli.fit_sigma, GAUSS_SNR_NUM_AA))) #don't divi by pix_size
                    #at this point, the pixel units or width don't matter ... everything is per pixel
                num_sn_pix = int(round(min(num_sn_pix,len(wave_counts)/2 - 1))) #don't go larger than the actual array

                # if (2 * sigma * 2.355) > (len(narrow_wave_counts)):
                #     # could be very skewed and broad, so don't center on the emission peak, but center on the array
                #     cw_idx = len(narrow_wave_counts) // 2
                # else:
                cw_idx = getnearpos(wave_x, eli.fit_x0 )

                #?rms just under the part of the plot with signal (not the entire fit part) so, maybe just a few AA or pix
                eli.fit_norm_rmse = SU.rms(wave_counts, rms_wave, cw_pix=cw_idx, hw_pix=num_sn_pix,
                             norm=True)
                eli.fit_rmse = SU.rms(wave_counts, rms_wave, cw_pix=cw_idx, hw_pix=num_sn_pix,
                             norm=False)

                #*2 +1 because the rmse is figures as +/- the "num_sn_pix" from the center pixel (so total width is *2 + 1)
                num_sn_pix = num_sn_pix * 2 + 1 #need full width later (still an integer)
                eli.sn_pix = num_sn_pix

                #test (though, essentially the curve_fit is a least-squarest fit (which is
                # really just a chi2 fit, (technically IF the model data is Gaussian)), so
                # since we got here from a that fit, this chi2 would have to be small (otherwise
                # the fit would have failed .. and this is the "best" of those fits)
                #dof = 3 for the 3 parameters we are fitting (mu, sigma, y)
                try:
                    chi2_half_width = wave_fit_side_aa  # 50.0 #eli.fit_sigma * GAUSS_SNR_SIGMA #in AA
                    left,*_ = SU.getnearpos(wavelengths,eli.fit_x0 - chi2_half_width)
                    right,*_ = SU.getnearpos(wavelengths,eli.fit_x0 + chi2_half_width)
                    right = min(right+1,len(wavelengths))

                    data_waves = wavelengths[left:right]
                    data_flux = values[left:right]
                    if errors is not None and len(errors)==len(wavelengths):
                        data_err = errors[left:right]
                    else:
                        data_err = np.zeros(len(data_flux))

                    #this is correct (full fitted width)
                    eli.fit_chi2, _ = SU.chi_sqr(data_flux,
                                                 gaussian(data_waves, eli.fit_x0, eli.fit_sigma, eli.fit_a, eli.fit_y),
                                                 error=data_err,c=1.0)#, dof=3)
                except:
                    log.debug("*** Chi2 error.",exc_info=True)
                #scipy_chi2,scipy_pval = chisquare(wave_counts,rms_wave)

    except Exception as ex:
        try: #bug? in Python3 ... after 3.4 message attribute is lost?
            if ex.message.find("Optimal parameters not found") > -1:
                log.debug("Could not fit gaussian (1) near %f" % central,exc_info=False)
            else:
                log.error("Could not fit gaussian (2) near %f" % central, exc_info=True)
        except:
            try:
                if ex.args[0].find("Optimal parameters not found") > -1:
                    log.debug("Could not fit gaussian (3) near %f" % central, exc_info=False)
                else:
                    log.error("Exception in signal_score()",exc_info=True)
            except:
                log.error("Could not fit gaussian near (4) %f" % central, exc_info=True)
        return None

    #if there is a large anchor sigma (the sigma of the "main" line), then the max_sigma can be allowed to go higher
    if allow_broad:
        # if gmag is not None and gmag < G.BROADLINE_GMAG_MAX:
        #     max_sigma = max_tested_sigma # to match the xlrg type in the quick fit above
        # else:
        #     max_sigma = GAUSS_FIT_MAX_SIGMA * 1.5
        max_sigma = max_tested_sigma
    else:
        max_sigma = GAUSS_FIT_MAX_SIGMA

    if (eli.fit_rmse > 0) and (eli.fit_sigma >= min_sigma) and ( 0 < (eli.fit_sigma-eli.fit_sigma_err) <= max_sigma) and \
        (eli.fit_a_err < abs(eli.fit_a) ):

        #this snr makes sense IF we assume the noise is distributed as a gaussian (which is reasonable)
        #then we'd be looking at something like 1/N * Sum (sigma_i **2) ... BUT , there are so few pixels
        #  typically around 10 and there really should be at least 30  to approximate the gaussian shape

        try: #data_err is the data errors for the pixels selected in the RMSE fit range
            #eli.snr = eli.fit_a/(np.sum(np.sqrt(data_err)))
            delta_wave = max( eli.fit_sigma*2.0,2.1195)  #must be at least +/- 2.1195AA
            left,*_ = SU.getnearpos(wavelengths,eli.fit_x0 - delta_wave)
            right,*_ = SU.getnearpos(wavelengths,eli.fit_x0 + delta_wave)

            if wavelengths[left] - (eli.fit_x0-delta_wave) < 0:
                left += 1 #less than 50% overlap in the left bin, so move one bin to the red
            if wavelengths[right] - (eli.fit_x0+delta_wave) > 0:
                right -=1 #less than 50% overlap in the right bin, so move one bin to the blue

            #lastly ... could combine, but this is easier to read
            right += 1 #since the right index is not included in slice

            model_fit = gaussian(wavelengths[left:right], eli.fit_x0, eli.fit_sigma, eli.fit_a, eli.fit_y)
            #apcor = np.ones(len(model_fit))#need the actual aperture correction (may already be applied in the future)

            #the fit_a is in 2AA bins so is a factor of 2 high
            #0.955 is since this is +/- 2 sigma (3 sigma would be 0.996, 1 sigma would be 0.682)
            #eli.snr = 0.955*abs(eli.fit_a)/np.sqrt(np.sum(errors[left:right]**2))
            eli.snr = abs(np.sum(model_fit-eli.fit_y))/np.sqrt(np.sum(errors[left:right]**2))

            chi2_model_fit = gaussian(narrow_wave_x, eli.fit_x0, eli.fit_sigma, eli.fit_a, eli.fit_y)
            eli.fit_chi2, _ = SU.chi_sqr(narrow_wave_counts, chi2_model_fit, error=narrow_wave_errors, c=1.0)  # ,dof=3)
            log.debug(f"curve_fit SNR: {eli.snr}; chi2: {eli.fit_chi2}; Area={eli.fit_a} RMSE={eli.fit_rmse} Pix={num_sn_pix}")
        except: #try alternate SNR
            log.info("signal_score() SNR fail. Falling back to RMSE based.")
            eli.snr = eli.fit_a/(np.sqrt(num_sn_pix)*eli.fit_rmse)/np.sqrt(broadfit)
            log.debug(f"curve_fit SNR: {eli.snr}; Area={eli.fit_a} RMSE={eli.fit_rmse} Pix={num_sn_pix}")

        eli.unique = unique_peak(values,wavelengths,eli.fit_x0,eli.fit_sigma*2.355,absorber=absorber)

        if absorber:
            rough_continuum = eli.fit_y + eli.fit_y_err #best case lowest continuum estimate
        else:
            rough_continuum = eli.fit_y - eli.fit_y_err #best case lowest continuum estimate
        rough_height = abs(eli.fit_h - rough_continuum)
        rough_fwhm = eli.fit_sigma * 2.355
        max_allowed_peak_err_fraction = 0.34
        min_height_to_fwhm = 1.66

        bail = False
        if recommend_mcmc: #if choose not to conintue, this would have been a rejected line, so zero out and bail
            if ((eli.snr > 15.0) or ((5.0 < eli.snr < 15.0) and (eli.fit_chi2 < 2.0))) and (eli.unique):
                do_mcmc = True
            else:
                log.debug("Failed to meet minimum condition to continue with MCMC. Enforcing prior rejection.")
                recommend_mcmc = False
                captured_peak = False
                bail = True

        if bail:
            accept_fit = False
            snr = 0.0
            eli.snr = 0.0
            eli.line_score = 0.0
            eli.raw_line_score = 0.0
            eli.line_flux = 0.0
        elif not eli.unique and ((eli.fit_a_err / abs(eli.fit_a)) > max_allowed_peak_err_fraction) and (eli.fit_sigma > GAUSS_FIT_MAX_SIGMA):
            accept_fit = False
            snr = 0.0
            eli.snr = 0.0
            eli.line_score = 0.0
            eli.raw_line_score = 0.0
            eli.line_flux = 0.0
        # elif (eli.fit_a_err / eli.fit_a > 0.5):
        #     #error on the area is just to great to trust, regardless of the fit height
        #     accept_fit = False
        #     snr = 0.0
        #     eli.snr = 0.0
        #     eli.line_score = 0.0
        #     eli.line_flux = 0.0
        #     log.debug(f"Fit rejected: fit_a_err/fit_a {eli.fit_a_err / eli.fit_a} > 0.5")
        elif ((not allow_broad and rough_height < rough_fwhm) or (allow_broad and (rough_height * 2) < rough_fwhm)) \
                and not recommend_mcmc and gmag is not None and gmag > G.BROADLINE_GMAG_MAX: #widder than tall; skip if recommend mcmc triggered
            accept_fit = False
            snr = 0.0
            eli.snr = 0.0
            eli.line_score = 0.0
            eli.raw_line_score = 0.0
            eli.line_flux = 0.0
            log.debug(
                f"Fit rejected: widder than tall. Height above y = {rough_height}, FWHM = {rough_fwhm}")
        elif (eli.fit_a_err / abs(eli.fit_a) > max_allowed_peak_err_fraction) and (rough_height/rough_fwhm < min_height_to_fwhm):
            #error on the area is just too great to trust along with very low peak height (and these are already broad)
            accept_fit = False
            snr = 0.0
            eli.snr = 0.0
            eli.line_score = 0.0
            eli.raw_line_score = 0.0
            eli.line_flux = 0.0
            log.debug(f"Fit rejected: fit_a_err/fit_a {eli.fit_a_err / abs(eli.fit_a)} > {max_allowed_peak_err_fraction}"
                      f" and rough_height/rough_continuum {rough_height/rough_fwhm} < f{min_height_to_fwhm}")
        else:
            try:
                if eli.absorber and (eli.cont >= G.CONTINUUM_THRESHOLD_FOR_ABSORPTION_CHECK or gmag <= G.DEX_GMAG_THRESHOLD_FOR_ABSORPTION_CHECK):
                    eli.build(values_units=values_units,allow_broad=allow_broad,broadfit=broadfit,override_continuum_rules=True)
                else:
                    eli.build(values_units=values_units, allow_broad=allow_broad, broadfit=broadfit)
            except:
                eli.build(values_units=values_units, allow_broad=allow_broad, broadfit=broadfit)
            #eli.snr = max(eli.fit_vals) / (np.sqrt(num_sn_pix) * eli.fit_rmse)
            snr = eli.snr
    else:
        #    if (eli.fit_rmse > 0) and (eli.fit_sigma >= min_sigma) and ( 0 < (eli.fit_sigma-eli.fit_sigma_err) <= max_sigma) and \
        #(eli.fit_a_err < abs(eli.fit_a) ):
        log.debug(f"Fit rejected ({eli.fit_x0:0.2f}) : fit_rmse {eli.fit_rmse:0.2f}, "
                  f"sigma {eli.fit_sigma:0.2f} +/- {eli.fit_sigma_err:0.2f}, "
                  f"fit_a {eli.fit_a:0.2f} +/- {eli.fit_a_err:0.2f}. Allow Broad ({allow_broad})")
        accept_fit = False
        snr = 0.0
        eli.snr = 0.0
        eli.line_score = 0.0
        eli.raw_line_score = 0.0
        eli.line_flux = 0.0

    log.debug("SNR at %0.2f (fiducial = %0.2f) = %0.2f"%(eli.fit_x0,central,snr))
    #log.debug("SNR (vs fibers) at %0.2f (fiducial = %0.2f) = %0.2f"%(eli.fit_x0,central,eli.peak_sigma_above_noise()))

    title = ""

    #todo: re-calibrate to use SNR instead of SBR ??
    sbr = snr
    if sbr is None:
        sbr = est_peak_strength(wavelengths,values,central,values_units,absorber=absorber)
        if sbr is None:
            #done, no reason to continue
            log.warning("Could not determine SBR at wavelength = %f. Will use SNR." %central)
            sbr = snr

    score = eli.line_score
    eli.sbr = sbr

    fit_wave = eli.fit_vals
    error = eli.fit_norm_rmse

    #fit around designated emis line
    #print("*!*!*!*!*!*! TRUNED OFF original scoring adjustments to line fit *!*!*!*!*!*!")
    if False:

        score = sbr
        sk = -999
        ku = -999
        si = -999
        dx0 = -999  # in AA
        rh = -999
        if absorber:
            mx_norm = min(wave_counts) / 100.0
        else:
            mx_norm = max(wave_counts) / 100.0

        if (fit_wave is not None):
            sk = skew(fit_wave)
            ku = kurtosis(fit_wave) # remember, 0 is tail width for Normal Dist. ( ku < 0 == thinner tails)
            si = eli.fit_sigma  #*1.9 #scale to angstroms
            dx0 = eli.fit_dx0 #*1.9

            #si and ku are correlated at this scale, for emission lines ... fat si <==> small ku

            height_pix = raw_peak
            height_fit = scaled_fit_h

            if height_pix > 0:
                rh = height_fit/height_pix
            else:
                log.debug("Minimum peak height (%f) too small. Score zeroed." % (height_pix))
                dqs_raw = 0.0
                score = 0.0
                rh = 0.0

            #todo: for lower S/N, sigma (width) can be less and still get bonus if fibers have larger separation

            #new_score:
            if (0.75 < rh < 1.25) and (error < 0.2): # 1 bad pixel in each fiber is okay, but no more

                #central peak position
                #2020-03-09 turn off ... being off in dx0 is handled elsewhere and there are valid physical reasons this can be so
                # if abs(dx0) > pix_size:# 1.9:  #+/- one pixel (in AA)  from center
                #     val = (abs(dx0) - pix_size)** 2
                #     score -= val
                #     log.debug("Penalty for excessive error in X0: %f" % (val))
                #

                #sigma scoring
                if si < 2.0: # and ku < 2.0: #narrow and not huge tails
                    val = mx_norm*np.sqrt(2.0 - si)
                    score -= val
                    log.debug("Penalty for low sigma: %f" % (val))
                    #note: si always > 0.0 and rarely < 1.0
                elif si < 2.5:
                    pass #zero zone
                elif si < 10.0:
                    val = np.sqrt(si-2.5)
                    score += val
                    log.debug("Bonus for large sigma: %f" % (val))
                elif si < 15.0:
                    pass #unexpected, but lets not penalize just yet
                elif not allow_broad: #very wrong (could be a broadline hit)
                    if si > (5*min_sigma): #if a large min_sigma is passed in, this can be allowed to be larger w/o penalty
                        val = np.sqrt(si-15.0)
                        score -= val
                        log.debug("Penalty for excessive sigma: %f" % (val))


                #only check the skew for smaller sigma
                #skew scoring

                #2020-03-09 turn off ... noise can be high enough that this is not a valid test
                #plus this gets applied to ALL lines, not just LyA, so this is not a valid check in most cases
                # if si < 2.5:
                #     if sk < -0.5: #skew wrong directionn
                #         val = min(1.0,mx_norm*min(0.5,abs(sk)-0.5))
                #         score -= val
                #         log.debug("Penalty for low sigma and negative skew: %f" % (val))
                #     if (sk > 2.0): #skewed a bit red, a bit peaky, with outlier influence
                #         val = min(0.5,sk-2.0)
                #         score += val
                #         log.debug("Bonus for low sigma and positive skew: %f" % (val))

                base_msg = "Fit dX0 = %g(AA), RH = %0.2f, rms = %0.2f, Sigma = %g(AA), Skew = %g , Kurtosis = %g "\
                       % (dx0, rh, error, si, sk, ku)
                log.debug(base_msg)
            elif rh > 0.0:
                #todo: based on rh and error give a penalty?? maybe scaled by maximum pixel value? (++val = ++penalty)

                if (error > 0.3) and (0.75 < rh < 1.25): #really bad rms, but we did capture the peak
                    val = mx_norm*(error - 0.3)
                    score -= val
                    log.debug("Penalty for excessively bad rms: %f" % (val))
                elif rh < 0.6: #way under shooting peak (should be a wide sigma) (peak with shoulders?)
                    val = mx_norm * (0.6 - rh)
                    score -= val
                    log.debug("Penalty for excessive undershoot peak: %f" % (val))
                elif rh > 1.4: #way over shooting peak (super peaky ... prob. hot pixel?)
                    val = mx_norm * (rh - 1.4)
                    score -= val
                    log.debug("Penalty for excessively overshoot peak: %f" % (val))
            else:
                log.debug("Too many bad pixels or failure to fit peak or overall bad fit. ")
                score = 0.0
        else:
            log.debug("Unable to fit gaussian. ")
            score = 0.0

    mcmc = None

    #print(" ******************** UNDO !!!! turned off MCMC ********************")
    if do_mcmc or (G.FORCE_MCMC and accept_fit and (eli is not None) and (eli.snr > G.FORCE_MCMC_MIN_SNR)):
        # print("*****TESTING DOUBLE GAUSS******")
        # print("***** check_for_doublet *****")
        # eli2 = check_for_doublet(eli,wavelengths,values,errors,central,values_units)

        # if spec_obj is not None and do_mcmc and targetted_fit: #only want this on the MAIN line?
        #     #print("*****TESTING DOUBLE GAUSS******")
        #     print("***** check_for_lya_blue *****")
        #     #note that a lot of OII can also appear to have a blue peak (and it is a doublet) so be careful
        #     spec_obj.lya_mcmc_dict = check_for_lya_blue(eli, wavelengths, values, errors, central, values_units)
        #     if spec_obj.lya_mcmc_dict is not None:
        #         print(f"Possible LyA blue peak observed: {spec_obj.lya_mcmc_dict['blue_mu']:0.2f},{spec_obj.lya_mcmc_dict['red_mu']:0.2f}, "
        #                  f"{spec_obj.lya_mcmc_dict['kms_sep']:0.1f} km/s, S/N = {spec_obj.lya_mcmc_dict['mcmc'].mcmc_snr},"
        #                  f"blue chi2 = {spec_obj.lya_mcmc_dict['blue_chi2']:0.2f}")
        #     else:
        #         print("No blue peak identified.")


        mcmc = mcmc_gauss.MCMC_Gauss()

        if bad_curve_fit and (eli.snr is None or eli.fit_chi2 is None or eli.snr < 5.0 or eli.fit_chi2 > 3.0):
            mcmc.initial_mu = central
            mcmc.initial_sigma = 1.7
            mcmc.initial_A = raw_peak * 2.355 * mcmc.initial_sigma  # / adjust
            if absorber:
                mcmc.initial_A *= -1
        else:
            mcmc.initial_mu = eli.fit_x0
            mcmc.initial_sigma = eli.fit_sigma
            mcmc.initial_A = eli.fit_a  # / adjust

        mcmc.initial_y = eli.fit_y  # / adjust
        mcmc.initial_peak = raw_peak  # / adjust
        mcmc.data_x = narrow_wave_x
        mcmc.data_y = narrow_wave_counts # / adjust
        mcmc.err_y = narrow_wave_errors  # not the 1./err*err .... that is done in the mcmc likelihood function

        if absorber:
            mcmc.min_y = np.min(narrow_wave_counts)
            mcmc.max_y = np.max(narrow_wave_counts) * 1.5

        if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None:
            mcmc.min_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MIN
            mcmc.max_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MAX
        else:
            mcmc.min_sigma = min_sigma
            if max_sigma >= 999:
                mcmc.max_sigma = GAUSS_FIT_MAX_SIGMA
            else:
                mcmc.max_sigma = max_sigma

        # if using the scipy::curve_fit, 50-100 burn-in and ~1000 main run is plenty
        # if other input (like Karl's) ... the method is different and we are farther off ... takes longer to converge
        #   but still converges close to the scipy::curve_fit
        if recommend_mcmc: #this is a low-stakes follow on to LSQ, so a lesser run is okay
            mcmc.burn_in = 250
            mcmc.main_run = 750
        else:
            mcmc.burn_in = 250
            mcmc.main_run = 1200

        #print(f"**** MCMC: run ({mcmc.main_run}), mu ({mcmc.initial_mu}), data_x ({len(mcmc.data_x)}), "
        #      f"sigma ({mcmc.initial_sigma}), A ({mcmc.initial_A}), y ({mcmc.initial_y }) ")

        # mcmc.burn_in = 250
        # mcmc.main_run = 1200
        mcmc.run_mcmc()


        if True:
            eli.mcmc_to_fit(mcmc,values_units,values_dx)
            #and rescore from MCMC
            old_score = eli.line_score
            try:
                if eli.absorber and (eli.cont >= G.CONTINUUM_THRESHOLD_FOR_ABSORPTION_CHECK
                                     or gmag <= G.DEX_GMAG_THRESHOLD_FOR_ABSORPTION_CHECK):
                    eli.build(values_units=values_units,allow_broad=allow_broad,broadfit=broadfit,
                              override_continuum_rules=True)
                else:
                    eli.build(values_units=values_units,allow_broad=allow_broad,broadfit=broadfit)
            except:
                eli.build(values_units=values_units, allow_broad=allow_broad, broadfit=broadfit)
            log.info(f"Rescore from MCMC: old {old_score}, new {eli.line_score}")

        else:

            #correct SNR for PSF? (seeing FWHM / aperture)**2
            #psf_correction = ### does not currently have that info here
            #mcmc.mcmc_snr *= psf_correction
            #mcmc.mcms_snr_err /= psf_correction
            #divide one and multiply the other?? or does the snr_err even need a correction?

            #TEST


            # if True:
            #     png = "mcmc" + plot_id + str(central) + "_" + ".png"
            #     if plot_path is not None:
            #         png = op.join(plot_path, png)
            #     mcmc.visualize(png)

            # 3-tuple [0] = fit, [1] = fit +16%,  [2] = fit - 16%
            eli.mcmc_x0 = mcmc.mcmc_mu
            eli.mcmc_sigma = mcmc.mcmc_sigma
            eli.mcmc_snr = mcmc.mcmc_snr
            eli.mcmc_snr_err = mcmc.mcmc_snr_err

            #updated a bit later
            # eli.snr =   eli.mcmc_snr
            # eli.snr_err = eli.mcmc_snr_err

            if mcmc.mcmc_A is not None:
                eli.mcmc_a = np.array(mcmc.mcmc_A)
                if (values_dx is not None) and (values_dx > 0):
                    #have to divide out the width of the bins since that was not already accounted for
                    #e.g. we pass in fluxes in a 2AA wide bin and need to "normalize" to the bin width
                    eli.mcmc_dx = values_dx
                    eli.mcmc_line_flux = eli.mcmc_a[0]/values_dx
                    eli.mcmc_line_flux_tuple =  np.array(mcmc.mcmc_A)/values_dx
            else:
                eli.mcmc_a = np.array((0.,0.,0.))
                eli.mcmc_line_flux = eli.mcmc_a[0]
                eli.mcmc_line_flux_tuple = np.array((0.,0.,0.))

            if mcmc.mcmc_y is not None:
                eli.mcmc_y = np.array(mcmc.mcmc_y)
                if (values_dx is not None) and (values_dx > 0):
                    eli.mcmc_dx = values_dx
                    eli.mcmc_continuum = eli.mcmc_y[0]  / values_dx
                    eli.mcmc_continuum_tuple = np.array(mcmc.mcmc_y)  / values_dx
            else:
                eli.mcmc_y = np.array((0.,0.,0.))
                eli.mcmc_continuum = eli.mcmc_y[0]  #/ values_dx #just a 0 divide by anything
                eli.mcmc_continuum_tuple = np.array(eli.mcmc_y)  #/ values_dx

            if values_units < 0:
                eli.mcmc_a *= 10**values_units
                eli.mcmc_y *= 10**values_units
                eli.mcmc_continuum  *= 10**values_units
                eli.mcmc_line_flux *= 10**values_units
                try:
                    eli.mcmc_line_flux_tuple *= 10 ** values_units
                    eli.mcmc_continuum_tuple *= 10 ** values_units
                except:
                    log.error("*** Exception!", exc_info=True)

                #no ... this is wrong ... its all good now
            # if values_units == -18:  # converted from e-17, but this is an area so there are 2 factors
            #     eli.mcmc_a = tuple(np.array(eli.mcmc_a ) / [10., 1., 1.])

            # calc EW and error with approximate symmetric error on area and continuum
            if eli.mcmc_y[0] != 0 and eli.mcmc_a[0] != 0:
                ew = eli.mcmc_a[0] / eli.mcmc_y[0]
                ew_err = ew * np.sqrt((mcmc.approx_symmetric_error(eli.mcmc_a) / eli.mcmc_a[0]) ** 2 +
                                      (mcmc.approx_symmetric_error(eli.mcmc_y) / eli.mcmc_y[0]) ** 2)
            else:
                ew = eli.mcmc_a[0]
                ew_err = mcmc.approx_symmetric_error(eli.mcmc_a)


            #todo: could add an uncertainty on this since we have MCMC uncertainties on the parameters (mu, sigma, y)

            #def gaussian(x,x0,sigma,a=1.0,y=0.0):
            #could do this by say +/- 3 sigma (Karl uses +/- 50 AA)
            try:
                chi2_half_width = wave_fit_side_aa #GAUSS_FIT_AA_RANGE #50.0 #mcmc.mcmc_sigma[0] * 2.5  #in AA
                left, *_ = SU.getnearpos(wavelengths,mcmc.mcmc_mu[0] - chi2_half_width)
                right,*_ = SU.getnearpos(wavelengths,mcmc.mcmc_mu[0] + chi2_half_width)
                right = min(right+1,len(wavelengths))

                data_waves = wavelengths[left:right]
                data_flux = values[left:right]
                if errors is not None and len(errors)==len(wavelengths):
                    data_err = errors[left:right]
                else:
                    data_err = np.zeros(len(data_flux))
                mcmc_flux = gaussian(data_waves, mcmc.mcmc_mu[0], mcmc.mcmc_sigma[0], mcmc.mcmc_A[0], mcmc.mcmc_y[0])

                #this is correct
                eli.mcmc_chi2, _ = SU.chi_sqr(data_flux,mcmc_flux,error=data_err,c=1.0)#,dof=3)

                #wave_x is ~ 40AA around the center[total of 41 bins in length, usually]
                # mcmc_flux = gaussian(wave_x, mcmc.mcmc_mu[0], mcmc.mcmc_sigma[0], mcmc.mcmc_A[0], mcmc.mcmc_y[0])
                # eli.mcmc_chi2, _ = SU.chi_sqr(wave_counts,mcmc_flux,error=wave_errors,c=1.0,dof=3)
            except:
                log.debug("*** Chi2 error.", exc_info=True)

            #if recommend_mcmc: #this was a marginal LSQ fit, so replace key the "fit_xxx" with the mcmc values
            if eli.mcmc_x0 is not None:
                fit_scale = 1/(10 ** values_units)

                eli.fit_x0 = eli.mcmc_x0[0]
                eli.fit_x0_err = 0.5*(eli.mcmc_x0[1]+eli.mcmc_x0[2])

                eli.fit_sigma = eli.mcmc_sigma[0]
                eli.fit_sigma_err = 0.5*(eli.mcmc_sigma[1]+eli.mcmc_sigma[2])

                if eli.mcmc_h is not None:
                    eli.fit_h = eli.mcmc_h

                eli.fit_a = eli.mcmc_a[0] * fit_scale
                eli.fit_a_err = 0.5*(eli.mcmc_a[1]+eli.mcmc_a[2]) * fit_scale

                eli.fit_y = eli.mcmc_y[0] * fit_scale
                eli.fit_y_err = 0.5*(eli.mcmc_y[1]+eli.mcmc_y[2])* fit_scale


                #MCMC is preferred to update key values
                eli.line_flux = eli.mcmc_line_flux
                eli.line_flux_err = 0.5*(eli.mcmc_line_flux_tuple[1]+eli.mcmc_line_flux_tuple[2])

                #explicit SNR
                # left,*_ = SU.getnearpos(wavelengths,eli.fit_x0-eli.fit_sigma*4)
                # right,*_ = SU.getnearpos(wavelengths,eli.fit_x0+eli.fit_sigma*4)
                # noise = np.sum(errors[left:right]*(10 ** values_units))


                if eli.mcmc_snr is not None and eli.mcmc_snr > 0:
                    log.debug(f"MCMC SNR update: old {eli.snr}+/-{eli.snr_err}, new {eli.mcmc_snr}+/-{eli.mcmc_snr_err}")
                    eli.snr = eli.mcmc_snr
                    eli.snr_err = eli.mcmc_snr_err

                eli.fwhm = eli.mcmc_sigma[0]*2.355

                eli.mcmc_ew_obs = [ew, ew_err, ew_err]
                log.debug("MCMC Peak height = %f" % (max(narrow_wave_counts)))
                log.debug("MCMC calculated EW_obs for main line = %0.3g +/- %0.3g" % (ew, ew_err))
                log.debug(f"MCMC line flux = {eli.mcmc_line_flux}")


                log.debug(f"Calc SNR line_flux/data err: {eli.line_flux/np.sqrt(np.sum(narrow_wave_errors))}")

                #and rescore from MCMC
                old_score = eli.line_score
                eli.build(values_units=values_units,allow_broad=allow_broad,broadfit=broadfit)
                log.info(f"Rescore from MCMC: old {old_score}, new {eli.line_score}")




        #testing alternate SNR calculation
        # log.debug(f"Line flux {eli.mcmc_line_flux} ")
        # mcmc_snr = SU.snr(eli.mcmc_line_flux,narrow_wave_errors*10**values_units,flux_err=mcmc.approx_symmetric_error(eli.mcmc_line_flux_tuple),
        #                   wave=narrow_wave_x,center=mcmc.mcmc_mu[0],delta=mcmc.mcmc_sigma[0]*2.0)
        # data_snr = SU.snr(narrow_wave_counts,narrow_wave_errors,flux_err=None,wave=narrow_wave_x,center=eli.fit_x0,delta=eli.fit_sigma*2.)
        # log.debug(f"MCMC area SNR: {mcmc_snr}")
        # model_flux = SU.gaussian(narrow_wave_x,mcmc.mcmc_mu[0],mcmc.mcmc_sigma[0],a=mcmc.mcmc_A[0],y=mcmc.mcmc_y[0])
        # mcmc_snr = SU.snr(model_flux,narrow_wave_errors,wave=narrow_wave_x,center=mcmc.mcmc_mu[0],delta=mcmc.mcmc_sigma[0]*2.0)
        # log.debug(f"MCMC model SNR: {mcmc_snr}")
        # log.debug(f"MCMC data SNR: {data_snr}")


    if show_plot or G.DEBUG_SHOW_GAUSS_PLOTS:# or eli.snr > 40.0:
        if error is None:
            error = -1

        g = eli.is_good(z=central_z,allow_broad=allow_broad)
        a = accept_fit

        # title += "%0.2f z_guess=%0.4f A(%d) G(%d)\n" \
        #          "Score = %0.2f (%0.1f), SBR = %0.2f (%0.1f), SNR = %0.2f (%0.1f) wpix = %d\n" \
        #          "Peak = %0.2g, Line(A) = %0.2g, Cont = %0.2g, EqW_Obs=%0.2f\n"\
        #          "dX0 = %0.2f, RH = %0.2f, RMS = %0.2f (%0.2f) \n"\
        #          "Sigma = %0.2f, Skew = %0.2f, Kurtosis = %0.2f"\
        #           % (eli.fit_x0,central_z,a,g,score, signal_calc_scaled_score(score),sbr,
        #              signal_calc_scaled_score(sbr),snr,signal_calc_scaled_score(snr),num_sn_pix,
        #              eli.fit_h,eli.line_flux, eli.cont,eli.eqw_obs,
        #              dx0, rh, error,eli.fit_rmse, si, sk, ku)

        if eli.absorber:
            line_type = "[Absorption]"
        else:
            line_type = ""

        title += "%0.2f z_guess=%0.4f A(%d) G(%d) %s\n" \
                 "Line Score = %0.2f , SNR = %0.2f (%0.1f) , wpix = %d, LineFlux = %0.2g\n" \
                 "Peak = %0.2g, Area = %0.2g, Y = %0.2g, EqW_Obs=%0.2f\n"\
                 "dX0 = %0.2f, RH = %0.2f, RMS = %0.2f (%0.2f) \n"\
                 "Sigma = %0.2f, Skew = %0.2f, Kurtosis = %0.2f"\
                  % (eli.fit_x0,central_z,a,g,line_type,eli.line_score,
                     snr,signal_calc_scaled_score(snr),num_sn_pix,eli.fit_line_flux,
                     eli.fit_h,eli.fit_a, eli.fit_y,eli.eqw_obs,
                     dx0, rh, error,eli.fit_rmse, si, sk, ku)

        fig = plt.figure()
        gauss_plot = plt.axes()

        gauss_plot.plot(wave_x,wave_counts,c='k')

        try:
            gauss_plot.axvline(x=central,c='k',linestyle="--")
            gauss_plot.axvline(x=central+fit_range_AA, c='r', linestyle="--")
            gauss_plot.axvline(x=central-fit_range_AA, c='r', linestyle="--")
            if num_sn_pix > 0:
                half_sn = (num_sn_pix - 1) / 2. * pix_size

                gauss_plot.axvline(x=central + half_sn, c='g')
                gauss_plot.axvline(x=central - half_sn, c='g')
        except:
            log.debug("Cannot plot central line fit boundaries.",exc_info=True)

        if fit_wave is not None:
            gauss_plot.plot(xfit, fit_wave, c='b',zorder=99,lw=1)
            gauss_plot.grid(True)
        #     ymin = min(min(fit_wave),min(wave_counts))
        #     ymax = max(max(fit_wave),max(wave_counts))
        # else:
        #     ymin = min(wave_counts)
        #     ymax = max(wave_counts)

        if mcmc is not None:
            try:

              #  y,y_unc = gaussian_unc(narrow_wave_x, mcmc.mcmc_mu[0], mcmc.approx_symmetric_error(mcmc.mcmc_mu),
              #                         mcmc.mcmc_sigma[0], mcmc.approx_symmetric_error(mcmc.mcmc_sigma),
              #                         mcmc.mcmc_A[0], mcmc.approx_symmetric_error(mcmc.mcmc_A),
              #                         mcmc.mcmc_y[0], mcmc.approx_symmetric_error(mcmc.mcmc_y))
              #
              #  gauss_plot.errorbar(narrow_wave_x,y,yerr=[y_unc,y_unc],fmt="--o",alpha=0.5,color='green')

                y, y_unc = gaussian_unc(xfit, mcmc.mcmc_mu[0], mcmc.approx_symmetric_error(mcmc.mcmc_mu),
                                        mcmc.mcmc_sigma[0], mcmc.approx_symmetric_error(mcmc.mcmc_sigma),
                                        mcmc.mcmc_A[0], mcmc.approx_symmetric_error(mcmc.mcmc_A),
                                        mcmc.mcmc_y[0], mcmc.approx_symmetric_error(mcmc.mcmc_y))


                gauss_plot.fill_between(xfit,y+y_unc,y-y_unc,alpha=0.4,color='g')
                gauss_plot.plot(xfit, y, c='g', lw=1,alpha=1)#,zorder=1)

                #gauss_plot.plot(xfit,gaussian(xfit,mcmc.mcmc_mu[0], mcmc.mcmc_sigma[0],mcmc.mcmc_A[0],mcmc.mcmc_y[0]),
                #            c='b', lw=10,alpha=0.2,zorder=1)
            except:
                log.warning("Exception in spectrum::signal_score() trying to plot mcmc output." ,exc_info=True)


        gauss_plot.set_ylabel("Flux [unsp] ")
        gauss_plot.set_xlabel("Wavelength [$\AA$] ")

        # ymin, ymax = gauss_plot.get_ylim()
        #
        # ymin *= 1.1
        # ymax *= 1.1
        #
        # if abs(ymin) < 1.0: ymin = -1.0
        # if abs(ymax) < 1.0: ymax = 1.0

       # gauss_plot.set_ylim((ymin,ymax))
        gauss_plot.set_xlim( (np.floor(wave_x[0]),np.ceil(wave_x[-1])) )
        gauss_plot.set_title(title)
        stat = ""
        if a:
            stat += "a"
        if g:
            stat += "g"

        if plot_id is not None:
            plot_id = "_" + str(plot_id) + "_"
        else:
            plot_id = "_"
        png = "gauss" + plot_id + str(central)+ "_" + stat + ".png"

        if plot_path is not None:
            png = op.join(plot_path,png)

        log.info('Writing: ' + png)
        #print('Writing: ' + png)
        fig.tight_layout()
        fig.savefig(png)

        if eli is not None:
            eli.gauss_plot_buffer = io.BytesIO()
            plt.savefig(eli.gauss_plot_buffer, format='png', dpi=300)

        fig.clear()
        plt.close()

        if mcmc is not None:
            png = "mcmc" + plot_id + str(central) + "_" + stat + ".png"
            if plot_path is not None:
                png = op.join(plot_path, png)
            buf = mcmc.visualize(png)

            if eli is not None:
                eli.mcmc_plot_buffer = buf

        # end plotting

    if accept_fit or forced_mcmc:
        #last check
        if G.CONTINUUM_RULES and not absorber:
            #def local_continuum(wavelengths, values, errors, central, amplitude, sigma, cont_width=50,start=6.0 ):
            cont_avg, cont_std = local_continuum(wavelengths,values,errors,eli.fit_x0,eli.fit_h,eli.fit_sigma)
            if cont_avg is not None and cont_std is not None:
                if eli.fit_h < (cont_avg + cont_std):
                    #not a real emssion line, probably a return to continuum between two aborbers
                    log.info(f"Fit rejected. Emission probably continuum between two absorbers. @ {eli.fit_x0:0.2f}, "
                             f"peak = {eli.fit_h:0.2f}, cont = {cont_avg:0.2f} +/- {cont_std:0.2f}")
                    eli.raw_score = 0
                    eli.score = 0
                    eli.snr = 0.0
                    eli.line_score = 0.0
                    eli.raw_line_score = 0.0
                    eli.line_flux = 0.0
                    return eli

        eli.raw_score = score
        eli.score = signal_calc_scaled_score(score)
        log.debug(f"Fit not rejected. eli score: {eli.score} line score: {eli.line_score}")
        return eli
    else:
        log.debug("Fit rejected")
        return None



def run_mcmc(eli,wavelengths,values,errors,central,values_units,values_dx=G.FLUX_WAVEBIN_WIDTH,wave_fit_side_aa=GAUSS_FIT_AA_RANGE):

    #values_dx is the bin width for the values if multiplied out (s|t) values are flux and not flux/dx
    #   by default, Karl's data is on a 2.0 AA bin width

    err_units = values_units  # assumed to be in the same units
    values, values_units = norm_values(values, values_units)
    if errors is not None and (len(errors) == len(values)):
        errors, err_units = norm_values(errors, err_units)

    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix
    wave_side_pix = int(round(wave_fit_side_aa / pix_size))  # pixels
    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)

    len_array = len(wavelengths)
    idx = getnearpos(wavelengths, central)
    min_idx = max(0, idx - wave_side_pix)
    max_idx = min(len_array, idx + wave_side_pix)
    wave_x = wavelengths[min_idx:max_idx + 1]
    wave_counts = values[min_idx:max_idx + 1]
    if (errors is not None) and (len(errors) == len(wavelengths)):
        wave_errors = errors[min_idx:max_idx + 1]
        # replace any 0 with 1
        wave_errors[np.where(wave_errors == 0)] = 1
    else:
        wave_errors = None

    narrow_wave_x = wave_x
    narrow_wave_counts = wave_counts
    narrow_wave_errors = wave_errors

    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
    peak_pos = getnearpos(wavelengths, central)

    try:
        # find the highest point in the raw data inside the range we are allowing for the line center fit
        dpix = int(round(fit_range_AA / pix_size))
        raw_peak = max(values[peak_pos - dpix:peak_pos + dpix + 1])
        if raw_peak <= 0:
            log.warning("Spectrum::run_mcmc invalid raw peak %f" % raw_peak)
            return eli
    except:
        # this can fail if on very edge, but if so, we would not use it anyway
        log.debug(
            "Raw Peak value failure (3) for wavelength (%f) at index (%d). Cannot fit to gaussian. " % (central, peak_pos))
        return eli


    mcmc = mcmc_gauss.MCMC_Gauss()
    mcmc.initial_mu = eli.fit_x0
    mcmc.initial_sigma = eli.fit_sigma
    mcmc.initial_A = eli.fit_a  # / adjust
    mcmc.initial_y = eli.fit_y  # / adjust
    mcmc.initial_peak = raw_peak  # / adjust
    mcmc.data_x = narrow_wave_x
    mcmc.data_y = narrow_wave_counts  # / adjust
    mcmc.err_y = narrow_wave_errors  # not the 1./err*err .... that is done in the mcmc likelihood function

    # if using the scipy::curve_fit, 50-100 burn-in and ~1000 main run is plenty
    # if other input (like Karl's) ... the method is different and we are farther off ... takes longer to converge
    #   but still converges close to the scipy::curve_fit
    mcmc.burn_in = 250
    mcmc.main_run = 1000

    try:
        mcmc.run_mcmc()
    except:
        log.warning("Exception in spectrum.py calling mcmc.run_mcmc()", exc_info=True)
        return eli

    # 3-tuple [0] = fit, [1] = fit +16%,  [2] = fit - 16%
    eli.mcmc_x0 = mcmc.mcmc_mu
    eli.mcmc_sigma = mcmc.mcmc_sigma
    eli.mcmc_snr = mcmc.mcmc_snr

    if mcmc.mcmc_A is not None:
        eli.mcmc_a = np.array(mcmc.mcmc_A)
        if (values_dx is not None) and (values_dx > 0):
            eli.mcmc_dx = values_dx
            eli.mcmc_line_flux = eli.mcmc_a[0] / eli.mcmc_dx
            eli.mcmc_line_flux_tuple =  np.array(mcmc.mcmc_A)/values_dx
    else:
        eli.mcmc_a = np.array((0., 0., 0.))
        eli.mcmc_line_flux = eli.mcmc_a[0]

    if mcmc.mcmc_y is not None:
        eli.mcmc_y = np.array(mcmc.mcmc_y)
        if (values_dx is not None) and (values_dx > 0):
            eli.mcmc_dx = values_dx
            eli.mcmc_continuum = eli.mcmc_y[0] / values_dx
            eli.mcmc_continumm_tuple =  np.array(mcmc.mcmc_y) / values_dx
    else:
        eli.mcmc_y = np.array((0., 0., 0.))
        eli.mcmc_continuum = eli.mcmc_y[0]
        eli.mcmc_continumm_tuple = eli.mcmc_y

    if values_units < 0:
        eli.mcmc_a *= 10 ** values_units
        eli.mcmc_y *= 10 ** values_units
        eli.mcmc_continuum *= 10 ** values_units
        eli.mcmc_line_flux *= 10 ** values_units
        try:
            eli.mcmc_line_flux_tuple *= 10 ** values_units
            eli.mcmc_continuum_tuple *= 10 ** values_units
        except:
            log.error("*** Exception!",exc_info=True)

    # calc EW and error with approximate symmetric error on area and continuum
    if eli.mcmc_y[0] != 0 and eli.mcmc_a[0] != 0:
        ew = eli.mcmc_a[0] / eli.mcmc_y[0]
        ew_err = ew * np.sqrt((mcmc.approx_symmetric_error(eli.mcmc_a) / eli.mcmc_a[0]) ** 2 +
                              (mcmc.approx_symmetric_error(eli.mcmc_y) / eli.mcmc_y[0]) ** 2)
    else:
        ew = eli.mcmc_a[0]
        ew_err = mcmc.approx_symmetric_error(eli.mcmc_a)

    eli.mcmc_ew_obs = [ew, ew_err, ew_err]
    log.debug("MCMC Peak height = %f" % (max(narrow_wave_counts)))
    log.debug("MCMC calculated EW_obs for main line = %0.3g +/- %0.3g" % (ew, ew_err))

    return eli




# This seems to be de-volving to pretty much "is the shape broad enough and flat enough at the peak" to be
# a HETDEX resolvable doublet (so ~ 6AA in observed frame)? MCMC fits seem moderatly accurate, but there is
# usually so much error on the peak positions to make this useless
def check_for_doublet(eli,wavelengths,values,errors,central,values_units,values_dx=G.FLUX_WAVEBIN_WIDTH,
                      wave_fit_side_aa=GAUSS_FIT_AA_RANGE):
    """
    like run_mcmc but allow a double Gaussian

    Needs to start with a wide separation of the two mu's (rather than right on top of each other), but will move
    in to the best position. A true single Gaussian can still return 2 mu's but they are symmetric about the true
    mu and the two Gaussians are the same size, etc

    So, to evaluate:
    1) the separation must be larger than 5.5AA observed (HETDEX best limit)
    2) the separation must be close to what is expected by the rest spearation * (1+z)
    3) the two areas should not be equal?
    4) the two sigmas should not be equal?

    :param eli:
    :param wavelengths:
    :param values:
    :param errors:
    :param central:
    :param values_units:
    :param values_dx:
    # :param rest_dw = wavelength separation in the peaks in the rest frame (i.e if NV, 1238.8 and 1242.8 --> == 4.0)
    # :param rest_w = rest wavelength (as if a single line), for NV == 1240.
    :return:
    """

    #values_dx is the bin width for the values if multiplied out (s|t) values are flux and not flux/dx
    #   by default, Karl's data is on a 2.0 AA bin width

    err_units = values_units  # assumed to be in the same units
    values, values_units = norm_values(values, values_units)
    if errors is not None and (len(errors) == len(values)):
        errors, err_units = norm_values(errors, err_units)

    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix
    wave_side_pix = int(round(wave_fit_side_aa / pix_size))  # pixels
    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)

    len_array = len(wavelengths)
    idx = getnearpos(wavelengths, central)
    min_idx = max(0, idx - wave_side_pix)
    max_idx = min(len_array, idx + wave_side_pix)
    wave_x = wavelengths[min_idx:max_idx + 1]
    wave_counts = values[min_idx:max_idx + 1]
    if (errors is not None) and (len(errors) == len(wavelengths)):
        wave_errors = errors[min_idx:max_idx + 1]
        # replace any 0 with 1
        wave_errors[np.where(wave_errors == 0)] = 1
    else:
        wave_errors = None

    narrow_wave_x = wave_x
    narrow_wave_counts = wave_counts
    narrow_wave_errors = wave_errors

    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
    peak_pos = getnearpos(wavelengths, central)
    dw = 15.0 #separation between the two peaks (observed) as a starting point

    mcmc = mcmc_double_gauss.MCMC_Double_Gauss()
    mcmc.initial_A = eli.fit_a/2.0
    mcmc.initial_y = eli.fit_y
    mcmc.initial_sigma = eli.fit_sigma/2.0
    mcmc.initial_mu = eli.fit_x0-dw/2.0
    #mcmc.initial_peak = raw_peak1
    mcmc.initial_A_2 = eli.fit_a/2.0
    mcmc.initial_mu_2 = eli.fit_x0+dw/2.0
    mcmc.initial_sigma_2 = eli.fit_sigma/2.0
    #mcmc.initial_peak_2 = raw_peak2
    mcmc.data_x = narrow_wave_x
    mcmc.data_y = narrow_wave_counts
    mcmc.err_y = narrow_wave_errors#np.zeros(len(mcmc.data_y)) #narrow_wave_errors
    mcmc.err_x = np.zeros(len(mcmc.err_y))

    # if using the scipy::curve_fit, 50-100 burn-in and ~1000 main run is plenty
    # if other input (like Karl's) ... the method is different and we are farther off ... takes longer to converge
    #   but still converges close to the scipy::curve_fit
    mcmc.burn_in = 500
    mcmc.main_run = 2000

    try:
        mcmc.run_mcmc()
    except:
        log.warning("Exception in spectrum.py check_for_doublet() calling mcmc.run_mcmc()", exc_info=True)
        return eli

    try:
        #HETDEX spectral res is really about 5.5AA, but will say 4.0AA here to allow for some error
        #say 40.0AA for now for an upper limit ... too far to be called a doublet
        #assuming about 8AA rest for the doublet, shifted to z = 3.5 -> 36AA

       # print("*****turn off double_guass_fit.png *****")
      #  mcmc.show_fit(filename="double_gauss_fit.png")
      #  mcmc.visualize(filename="double_gauss_vis.png")

        #these are all triples (mean value, + error, -error)
        #could this be a fit?
        #notes: mcmc_A_2 is allowed to go to zero in the fit (mcmc_A is not), so if A_2 is zero this is a single Gaussian fit
        #just for sanity, check both A and A_2
        if (mcmc.mcmc_snr > 4.0) and (4.0 < abs(mcmc.mcmc_mu[0] - mcmc.mcmc_mu_2[0]) < 40) and \
           (mcmc.mcmc_A_2[0] > 0) and (mcmc.mcmc_A[0] > 0):
            #todo: look at the errors on mu ... should be small-ish (like 1-2 AA??)

            log.info(f"Possible emission line doublet observed (x,+,-): {mcmc.mcmc_mu },{mcmc.mcmc_mu_2}")

            #and (0.8 < (mcmc.mcmc_A/mcmc.mcmc_A_2) < 1.25): not checking to see if similar ... they might not be
            #think about possibly using for LyA and the blue peak
            #todo: check the relative peak heights and areas
            #is this consistent with a known doublet?
            list_doublets = [] #list of rest waves to match the list of emission lines as single gaussians in
                               # Spectrum class emission_lines

            if mcmc.mcmc_mu[0] < mcmc.mcmc_mu_2[0]:
                blue_mu = mcmc.mcmc_mu[0]
                blue_sigma = mcmc.mcmc_sigma[0]
                blue_A = mcmc.mcmc_A[0]
                blue_y = mcmc.mcmc_y[0]
                red_mu = mcmc.mcmc_mu_2[0]
                red_sigma = mcmc.mcmc_sigma_2[0]
                red_A = mcmc.mcmc_A_2[0]
                red_y = mcmc.mcmc_y[0] #there is only 1 y fit
                red_fit_peak = np.max(SU.gaussian(mcmc.data_x,red_mu,red_sigma,red_A,red_y))
                blue_fit_peak = np.max(SU.gaussian(mcmc.data_x,blue_mu,blue_sigma,blue_A,blue_y))
            else:
                red_mu = mcmc.mcmc_mu[0]
                red_sigma = mcmc.mcmc_sigma[0]
                red_A = mcmc.mcmc_A[0]
                red_y = mcmc.mcmc_y[0]
                blue_mu = mcmc.mcmc_mu_2[0]
                blue_sigma = mcmc.mcmc_sigma_2[0]
                blue_A = mcmc.mcmc_A_2[0]
                blue_y = mcmc.mcmc_y[0] #there is only 1 y fit

                red_fit_peak = np.max(SU.gaussian(mcmc.data_x,red_mu,red_sigma,red_A,red_y))
                blue_fit_peak = np.max(SU.gaussian(mcmc.data_x,blue_mu,blue_sigma,blue_A,blue_y))

            ratio_A = blue_A/red_A
            ratio_peak = blue_fit_peak/red_fit_peak
            obs_miss = 6.0 #allowd to miss by 4.0 AA observed

            #NV (dont'really expect to ever see this one by itself)
            zp1 = central/G.NV_1241
            miss = obs_miss/zp1
            p1 = 1238.8
            p2 = 1242.8
            rest_dw = (red_mu - blue_mu)/zp1

            if  (abs(blue_mu/zp1 - p1) < miss) and \
                (abs(red_mu/zp1 - p2) < miss) and \
                (abs((red_mu-blue_mu)/zp1 - (p2-p1)) < miss):
                    #todo: any other conditions like blue area > red area etc
                    list_doublets.append(G.NV_1241)
                    log.info("Possible fit to NV as doublet.")


            #MgII (can be by itself)
            zp1 = central/G.MgII_2799
            miss = obs_miss/zp1
            p1 = 2795.5
            p2 = 2802.7
            rest_dw = (red_mu - blue_mu)/zp1

            if  (abs(blue_mu/zp1 - p1) < miss) and \
                (abs(red_mu/zp1 - p2) < miss) and \
                (abs((red_mu-blue_mu)/zp1 - (p2-p1)) < miss) and\
                (ratio_peak > 1.0):
                #MgII blue peak supposedly larger than red
                list_doublets.append(G.MgII_2799)
                log.info("Possible fit to MgII as doublet.")

            #CIV
            zp1 = central/G.CIV_1549
            miss = obs_miss/zp1
            p1 = 1548.2
            p2 = 1550.8
            rest_dw = (red_mu - blue_mu)/zp1

            if  (abs(blue_mu/zp1 - p1) < miss) and \
                (abs(red_mu/zp1 - p2) < miss) and \
                (abs((red_mu-blue_mu)/zp1 - (p2-p1)) < miss): # and \
                #(ratio_A > 1.0):
                list_doublets.append(G.CIV_1549)
                log.info("Possible fit to CIV as doublet.")

            #OVI
            zp1 = central/G.OVI_1035
            miss = obs_miss/zp1
            p1 = 1031.9
            p2 = 1037.6
            rest_dw = (red_mu - blue_mu)/zp1

            if  (abs(blue_mu/zp1 - p1) < miss) and \
                (abs(red_mu/zp1 - p2) < miss) and \
                (abs((red_mu-blue_mu)/zp1 - (p2-p1)) < miss):

                list_doublets.append(G.OVI_1035)
                log.info("Possible fit to OVI as doublet.")


            if len(list_doublets) == 0: #no matches
                log.info("No specific rest-frame emission line doublet match found.")
            elif len(list_doublets) == 1: #exatly one match
                log.info(f"Exactly one rest-frame emission line doublet found: {list_doublets[0]}")
                eli.suggested_doublet = list_doublets[0]
            else:
                log.info("Too many possible rest-frame emission line doublets")

            return eli
        else:
            log.info(f"No indication of doublet. Double Gaussian fit is poor: mu {mcmc.mcmc_mu },{mcmc.mcmc_mu_2}")
            return eli
    except:
        log.warning("Exception in spectrum.py check_for_doublet()", exc_info=True)
        return eli

    return eli







def check_for_lya_blue(eli,wavelengths,values,errors,central,values_units,values_dx=G.FLUX_WAVEBIN_WIDTH,
                      wave_fit_side_aa=GAUSS_FIT_AA_RANGE):
    """
    like run_mcmc but allow a double Gaussian

    Modified from check_for_doublet, but allows a greater range and is intended to identify possible blue LyA peak

    !!! warning !!! OII can ALSO look like this, and it is a doublet (though not really resolveable at HETDEX R

    :param eli:
    :param wavelengths:
    :param values:
    :param errors:
    :param central:
    :param values_units:
    :param values_dx:
    # :param rest_dw = wavelength separation in the peaks in the rest frame (i.e if NV, 1238.8 and 1242.8 --> == 4.0)
    # :param rest_w = rest wavelength (as if a single line), for NV == 1240.
    :return:
    """

    #values_dx is the bin width for the values if multiplied out (s|t) values are flux and not flux/dx
    #   by default, Karl's data is on a 2.0 AA bin width

    err_units = values_units  # assumed to be in the same units
    values, values_units = norm_values(values, values_units)
    if errors is not None and (len(errors) == len(values)):
        errors, err_units = norm_values(errors, err_units)

    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix
    wave_side_pix = int(round(wave_fit_side_aa / pix_size))  # pixels
    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)

    len_array = len(wavelengths)
    idx = getnearpos(wavelengths, central)
    min_idx = max(0, idx - wave_side_pix)
    max_idx = min(len_array, idx + wave_side_pix)
    wave_x = wavelengths[min_idx:max_idx + 1]
    wave_counts = values[min_idx:max_idx + 1]
    if (errors is not None) and (len(errors) == len(wavelengths)):
        wave_errors = errors[min_idx:max_idx + 1]
        # replace any 0 with 1
        wave_errors[np.where(wave_errors == 0)] = 1
    else:
        wave_errors = None

    narrow_wave_x = wave_x
    narrow_wave_counts = wave_counts
    narrow_wave_errors = wave_errors

    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
    peak_pos = getnearpos(wavelengths, central)
    #dw = 15.0 #separation between the two peaks (observed) as a starting point

    mcmc = mcmc_double_gauss.MCMC_Double_Gauss()
    #blue peak
    mcmc.initial_A = eli.fit_a/2.0
    mcmc.initial_y = eli.fit_y
    mcmc.initial_sigma = eli.fit_sigma/2.0
    mcmc.initial_mu = eli.fit_x0 - 1500.0/3e5*eli.fit_x0 #up to 1500 km/s to the blue
    #mcmc.initial_peak = raw_peak1
    #red peak ... assumes we got a single Gaussian that is a good fit to just the red
    mcmc.initial_A_2 = eli.fit_a
    mcmc.initial_mu_2 = eli.fit_x0
    mcmc.initial_sigma_2 = eli.fit_sigma
    #mcmc.initial_peak_2 = raw_peak2
    mcmc.data_x = narrow_wave_x
    mcmc.data_y = narrow_wave_counts
    mcmc.err_y = narrow_wave_errors#np.zeros(len(mcmc.data_y)) #narrow_wave_errors
    mcmc.err_x = np.zeros(len(mcmc.err_y))

    # if using the scipy::curve_fit, 50-100 burn-in and ~1000 main run is plenty
    # if other input (like Karl's) ... the method is different and we are farther off ... takes longer to converge
    #   but still converges close to the scipy::curve_fit
    mcmc.burn_in = 500
    mcmc.main_run = 2000

    try:
        mcmc.run_mcmc()
    except:
        log.warning("Exception in spectrum.py check_for_doublet() calling mcmc.run_mcmc()", exc_info=True)
        return None

    try:
        #HETDEX spectral res is really about 5.5AA, but will say 4.0AA here to allow for some error
        #say 40.0AA for now for an upper limit ... too far to be called a doublet
        #assuming about 8AA rest for the doublet, shifted to z = 3.5 -> 36AA

        print("*****turn off double_guass_fit.png *****")
        mcmc.show_fit(filename="double_gauss_fit.png")
        mcmc.visualize(filename="double_gauss_vis.png")

        #these are all triples (mean value, + error, -error)
        #could this be a fit?
        #notes: mcmc_A_2 is allowed to go to zero in the fit (mcmc_A is not), so if A_2 is zero this is a single Gaussian fit
        #just for sanity, check both A and A_2
        if (mcmc.mcmc_snr > eli.snr*0.5) and (mcmc.mcmc_snr > 5.0) and \
           (150.0/3e5*mcmc.mcmc_mu_2[0] < abs(mcmc.mcmc_mu[0] - mcmc.mcmc_mu_2[0]) < 2000.0/3e5*mcmc.mcmc_mu_2[0]) and \
           (mcmc.mcmc_A_2[0] > 0) and (mcmc.mcmc_A[0] > 0):

            lya_dict = {}
            #should not happen, but could flip red and blue peaks
            if mcmc.mcmc_mu[0] < mcmc.mcmc_mu_2[0]:
                lya_dict['blue_mu'] = mcmc.mcmc_mu[0]
                lya_dict['blue_sigma'] = mcmc.mcmc_sigma[0]
                lya_dict['blue_A'] = mcmc.mcmc_A[0]
                lya_dict['blue_y'] = mcmc.mcmc_y[0]
                lya_dict['red_mu'] = mcmc.mcmc_mu_2[0]
                lya_dict['red_sigma'] = mcmc.mcmc_sigma_2[0]
                lya_dict['red_A'] = mcmc.mcmc_A_2[0]
                lya_dict['red_y'] = mcmc.mcmc_y[0] #there is only 1 y fit
                lya_dict['red_fit_peak'] = np.max(SU.gaussian(mcmc.data_x,lya_dict['red_mu'],lya_dict['red_sigma'] ,
                                                              lya_dict['red_A'] ,lya_dict['red_y']))
                lya_dict['blue_fit_peak'] = np.max(SU.gaussian(mcmc.data_x,lya_dict['blue_mu'],lya_dict['blue_sigma'],
                                                               lya_dict['blue_A'] ,lya_dict['blue_y']))
            else:
                lya_dict['red_mu'] = mcmc.mcmc_mu[0]
                lya_dict['red_sigma'] = mcmc.mcmc_sigma[0]
                lya_dict['red_A'] = mcmc.mcmc_A[0]
                lya_dict['red_y'] = mcmc.mcmc_y[0]
                lya_dict['blue_mu'] = mcmc.mcmc_mu_2[0]
                lya_dict['blue_sigma'] = mcmc.mcmc_sigma_2[0]
                lya_dict['blue_A'] = mcmc.mcmc_A_2[0]
                lya_dict['blue_y'] = mcmc.mcmc_y[0] #there is only 1 y fit

                lya_dict['red_fit_peak'] = np.max(SU.gaussian(mcmc.data_x,lya_dict['red_mu'],lya_dict['red_sigma'],
                                                              lya_dict['red_A'] ,lya_dict['red_y']))
                lya_dict['blue_fit_peak'] = np.max(SU.gaussian(mcmc.data_x,lya_dict['blue_mu'],lya_dict['blue_sigma'],
                                                               lya_dict['blue_A'] ,lya_dict['blue_y']))


            lya_dict['kms_sep'] = (lya_dict['red_mu']-lya_dict['blue_mu'])/lya_dict['red_mu']*3e5
            lya_dict['snr'] = mcmc.mcmc_snr
            lya_dict['mcmc'] =  mcmc

            #peaks must be distinct and not overlapping (by at least 1 sigma on each peak)
            if (lya_dict['blue_mu'] + lya_dict['blue_sigma']) < (lya_dict['red_mu'] - lya_dict['red_sigma']):

                #should check the blue side SNR and Chi2 (just the blue side region, but as the sum of the Gaussians)
                left, *_ = SU.getnearpos(mcmc.data_x, lya_dict['blue_mu']-2.0*lya_dict['blue_sigma'])
                right,*_ = SU.getnearpos(mcmc.data_x, lya_dict['blue_mu']+2.0*lya_dict['blue_sigma'])
                data_x = mcmc.data_x[left:right+1]
                data_y = mcmc.data_y[left:right+1]
                err_y = mcmc.err_y[left:right+1]
                model_y = SU.dbl_gaussian(data_x, lya_dict['blue_mu'], lya_dict['blue_sigma'], lya_dict['blue_A'],
                                          lya_dict['red_mu'] , lya_dict['red_sigma'] , lya_dict['red_A'] , lya_dict['red_y'])
                blue_chi2,*_ = SU.chi_sqr(data_y, model_y ,err_y,c=1.0,dof=2)

                #blue side SNR (too few points to be really meaningful ... Area might be better)
                #blue_snr = abs(np.sum(model_y - err_y)) / np.sqrt(np.nansum(err_y ** 2))

                #if blue_snr > 5.0 and blue_chi2 < 1.5:
                if blue_chi2 < 1.5 and lya_dict['blue_A']/lya_dict['red_A'] > 0.1:
                    lya_dict['blue_chi2'] = blue_chi2
                    #lya_dict['blue_snr'] = blue_snr

                    log.info(f"Possible LyA blue peak observed: {lya_dict['blue_mu']:0.2f},{lya_dict['red_mu']:0.2f}, "
                             f"{lya_dict['kms_sep']:0.1f} km/s, S/N = {mcmc.mcmc_snr}, "
                             f"blue chi2 = {lya_dict['blue_chi2']:0.2f}")
                else:
                    del lya_dict
                    lya_dict = None
            else:
                del lya_dict
                lya_dict = None
            # ratio_A = blue_A/red_A
            # ratio_peak = blue_fit_peak/red_fit_peak

            #is there any circumstance where blue peak would be larger than red? Maybe in an AGN, but that has its
            #own probelms.

            return lya_dict
        else:
            log.info(f"No indication of LyA blue peak. Double Gaussian fit is poor: mu {mcmc.mcmc_mu },{mcmc.mcmc_mu_2}")
            return None
    except:
        log.warning("Exception in spectrum.py check_for_doublet()", exc_info=True)
        return None

    return None


def fit_for_h_and_k(k_eli,h_eli,wavelengths,values,errors,values_units,values_dx=G.FLUX_WAVEBIN_WIDTH,
                    wave_fit_side_aa=GAUSS_FIT_AA_RANGE):
    """
    like run_mcmc but allow a double Gaussian

    Needs to start with a wide separation of the two mu's (rather than right on top of each other), but will move
    in to the best position. A true single Gaussian can still return 2 mu's but they are symmetric about the true
    mu and the two Gaussians are the same size, etc

    So, to evaluate:
    1) the separation must be larger than 5.5AA observed (HETDEX best limit)
    2) the separation must be close to what is expected by the rest spearation * (1+z)
    3) the two areas should not be equal?
    4) the two sigmas should not be equal?


    :param wavelengths:
    :param values:
    :param errors:
    :param central:
    :param values_units:
    :param values_dx: ? not using here?
    # :param rest_dw = wavelength separation in the peaks in the rest frame (i.e if NV, 1238.8 and 1242.8 --> == 4.0)
    # :param rest_w = rest wavelength (as if a single line), for NV == 1240.
    :return:
    """

    #values_dx is the bin width for the values if multiplied out (s|t) values are flux and not flux/dx
    #   by default, Karl's data is on a 2.0 AA bin width

    try:
        log.debug(f"Fitting for H&K near {h_eli.fit_x0:0.2f} & {k_eli.fit_x0:0.2f}")

        err_units = values_units  # assumed to be in the same units
        values, values_units = norm_values(values, values_units)
        if errors is not None and (len(errors) == len(values)):
            errors, err_units = norm_values(errors, err_units)

        pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix
        #may need to be wider to accomodate enough spectrum for both lines
        wave_side_pix = int(round(max(wave_fit_side_aa,50) / pix_size))  # pixels
        fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)

        len_array = len(wavelengths)
        central = 0.5 * (h_eli.fit_x0 + k_eli.fit_x0)
        idx = getnearpos(wavelengths, central)
        min_idx = max(0, idx - wave_side_pix)
        max_idx = min(len_array, idx + wave_side_pix)
        wave_x = wavelengths[min_idx:max_idx + 1]
        wave_counts = values[min_idx:max_idx + 1]
        if (errors is not None) and (len(errors) == len(wavelengths)):
            wave_errors = errors[min_idx:max_idx + 1]
            # replace any 0 with 1
            wave_errors[np.where(wave_errors == 0)] = 1
        else:
            wave_errors = None

        narrow_wave_x = wave_x
        narrow_wave_counts = wave_counts
        narrow_wave_errors = wave_errors

        fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
        peak_pos = getnearpos(wavelengths, central)

        mcmc = mcmc_double_gauss.MCMC_Double_Gauss()
        mcmc.values_units = values_units
        mcmc.range_mu = 5.0
        mcmc.initial_A = min(k_eli.fit_a,h_eli.fit_a)/2.0 #these are negative so, min
        mcmc.initial_y = np.max(narrow_wave_counts)
        mcmc.initial_sigma = max(k_eli.fit_sigma,2.0)/2.0
        mcmc.initial_mu = k_eli.fit_x0
        #mcmc.initial_peak = raw_peak1

        mcmc.initial_A_2 = min(k_eli.fit_a,h_eli.fit_a)/2.0 #these are negative so, min
        mcmc.initial_mu_2 = h_eli.fit_x0
        mcmc.initial_sigma_2 = max(h_eli.fit_sigma,2.0)/2.0
        #mcmc.initial_peak_2 = raw_peak2

        mcmc.data_x = narrow_wave_x
        mcmc.data_y = narrow_wave_counts
        mcmc.err_y = narrow_wave_errors#np.zeros(len(mcmc.data_y)) #narrow_wave_errors
        mcmc.err_x = np.zeros(len(mcmc.err_y))

        # if using the scipy::curve_fit, 50-100 burn-in and ~1000 main run is plenty
        # if other input (like Karl's) ... the method is different and we are farther off ... takes longer to converge
        #   but still converges close to the scipy::curve_fit
        mcmc.burn_in = 250
        mcmc.main_run = 1200

        try:
            mcmc.run_mcmc()
        except:
            log.warning("Exception in spectrum.py fit_for_h_and_k() calling mcmc.run_mcmc()", exc_info=True)
            return None, mcmc

        try:
            #HETDEX spectral res is really about 5.5AA, but will say 4.0AA here to allow for some error
            #say 40.0AA for now for an upper limit ... too far to be called a doublet
            #assuming about 8AA rest for the doublet, shifted to z = 3.5 -> 36AA

            # print("*****turn off double_guass_fit.png *****")
            # mcmc.show_fit(filename="fit_for_h_and_k.png")
            #  mcmc.visualize(filename="double_gauss_vis.png")
            if (mcmc.mcmc_snr > 5.0) and (abs(mcmc.mcmc_mu[0]/k_eli.fit_x0 - mcmc.mcmc_mu_2[0]/h_eli.fit_x0) < 0.005) and \
                    (abs( (mcmc.mcmc_A_2[0] - mcmc.mcmc_A[0]) / (0.5* (mcmc.mcmc_A_2[0] + mcmc.mcmc_A[0]))) < 0.5):

                log.info(f"Possible H & K observed (x,+,-): {mcmc.mcmc_mu_2 },{mcmc.mcmc_mu}")

                return True,mcmc
            else:
                log.info(f"No indication of H&K. Fit failed or basic checks failed. mu {mcmc.mcmc_mu },{mcmc.mcmc_mu_2}")
                return False,mcmc
        except:
            log.warning("Exception in spectrum.py fit_for_h_and_k()", exc_info=True)
            return None,mcmc
    except:
        return None, None

    return None,mcmc



def signal_calc_scaled_score(raw):
    # 5 point scale
    # A+ = 5.0
    # A  = 4.0
    # B+ = 3.5
    # B  = 3.0
    # C+ = 2.5
    # C  = 2.0
    # D+ = 1.5
    # D  = 1.0
    # F  = 0

    a_p = 14.0
    a__ = 12.5
    a_m = 11.0
    b_p = 8.0
    b__ = 7.0
    c_p = 6.0
    c__ = 5.0
    d_p = 4.0
    d__ = 3.0
    f__ = 2.0

    if raw is None:
        return 0.0
    else:
        hold = False

    if   raw > a_p : score = 5.0  #A+
    elif raw > a__ : score = 4.5 + 0.5*(raw-a__)/(a_p-a__) #A
    elif raw > a_m : score = 4.0 + 0.5*(raw-a_m)/(a__-a_m) #A-
    elif raw > b_p : score = 3.5 + 0.5*(raw-b_p)/(a_m-b_p) #B+ AB
    elif raw > b__ : score = 3.0 + 0.5*(raw-b__)/(b_p-b__) #B
    elif raw > c_p : score = 2.5 + 0.5*(raw-c_p)/(b__-c_p) #C+ BC
    elif raw > c__ : score = 2.0 + 0.5*(raw-c__)/(c_p-c__) #C
    elif raw > d_p : score = 1.5 + 0.5*(raw-d_p)/(c__-d_p) #D+ CD
    elif raw > d__ : score = 1.0 + 0.5*(raw-d__)/(d_p-d__) #D
    elif raw > f__ : score = 0.5 + 0.5*(raw-f__)/(d__-f__) #F
    elif raw > 0.0 : score =  0.5*raw/f__
    else: score = 0.0

    score = round(score,1)

    return score


# def est_ew_obs(fwhm=None,peak=None, wavelengths=None, values=None, central=None,values_units=0):
#
#     try:
#         if (wavelengths is not None) and (values is not None) and (central is not None):
#             fwhm =  est_fwhm(wavelengths,values,central,values_units)
#             if peak is None:
#                 peak = values[getnearpos(wavelengths, central)]
#
#         if (fwhm is not None) and (peak is not None):
#             return pix_to_aa(fwhm)*peak
#         else:
#             return None
#     except:
#         log.error("Error in spectrum::est_ew",exc_info=True)
#         return None
#
# def est_ew_rest():
#     #need to know z
#     pass
#
#


def est_fwhm(wavelengths,values,central,values_units=0):

    num_pix = len(wavelengths)
    idx = getnearpos(wavelengths, central)

    values, values_units = norm_values(values,values_units)

    background,zero = est_background(wavelengths,values,central,values_units)

    if zero is None:
        zero = 0.0

    hm = (values[idx] - zero) / 2.0

    #hm = float((pv - zero) / 2.0)
    pix_width = 0

    # for centroid (though only down to fwhm)
    sum_pos_val = wavelengths[idx] * values[idx]
    sum_pos = wavelengths[idx]
    sum_val = values[idx]

    # check left
    pix_idx = idx - 1

    try:
        while (pix_idx >= 0) and ((values[pix_idx] - zero) >= hm) \
                and ((values[pix_idx] -zero) < values[idx]):
            sum_pos += wavelengths[pix_idx]
            sum_pos_val += wavelengths[pix_idx] * values[pix_idx]
            sum_val += values[pix_idx]
            pix_width += 1
            pix_idx -= 1

    except:
        pass

    # check right
    pix_idx = idx + 1

    try:
        while (pix_idx < num_pix) and ((values[pix_idx]-zero) >= hm) \
                and ((values[pix_idx] - zero) < values[idx]):
            sum_pos += wavelengths[pix_idx]
            sum_pos_val += wavelengths[pix_idx] * values[pix_idx]
            sum_val += values[pix_idx]
            pix_width += 1
            pix_idx += 1
    except:
        pass

    #print("FWHM = %f at %f" %(pix_width, central))

    return pix_width

def est_background(wavelengths,values,central,values_units = 0,dw=DEFAULT_BACKGROUND_WIDTH,xw=10.0,peaks=None,valleys=None):
    """
    mean of surrounding (simple) peaks, excluding any obvious lines (above 3x std) - the zero

    :param wavelengths: [array] position (wavelength) coordinates of spectra
    :param values: [array] values of the spectra
    :param central: central wavelength aboout which to estimate noise
    :param dw: width about the central wavelength over which to estimate noise
    :param xw: width from the central wavelength to begin the dw window
               that is, average over all peaks between (c-xw-dw) and (c-xw) AND (c+xw) and (c+xw+dw)
               like a 1d annulus
    :param px: optional peak coordinates (wavelengths)
    :param pv: optional peak values (counts)
    :return: background, zero
    """

    values, values_units = norm_values(values, values_units)

    xw = max(DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND,xw)

    outlier_x = 3.0
    background = DEFAULT_BACKGROUND
    wavelengths = np.array(wavelengths)
    values = np.array(values)
    zero = None

    if dw > len(wavelengths)/2.0:
        return None, None

    try:
        # peaks, vallyes are 3D arrays = [index in original array, wavelength, value]
        if peaks is None or valleys is None:
            peaks, valleys = simple_peaks(wavelengths,values,values_units=values_units)

        if peaks is None or len(peaks) < 1:
            log.debug("No peaks returned. spectrum::est_background(...). Values range (%f,%f)" %(min(values),max(values)))
            return background, zero

        #get all the peak values that are in our background sample range
        peak_v = peaks[:,2]
        peak_w = peaks[:,1]

        peak_v = peak_v[((peak_w >= (central - xw - dw)) & (peak_w <= (central - xw))) |
                   ((peak_w >= (central + xw)) & (peak_w <= (central + xw + dw)))]

        # get all the valley values that are in our background sample range
        valley_v = valleys[:, 2]
        valley_w = valleys[:, 1]

        valley_v = valley_v[((valley_w >= (central - xw - dw)) & (valley_w <= (central - xw))) |
                        ((valley_w >= (central + xw)) & (valley_w <= (central + xw + dw)))]

        #remove outliers (under assumption that extreme outliers are signals or errors)
        if (len(peak_v) > 3) and (len(valley_v) > 3):
            peak_v = peak_v[abs(peak_v - np.mean(peak_v)) < abs(outlier_x * np.std(peak_v))]
            valley_v = valley_v[abs(valley_v-np.mean(valley_v)) < abs(outlier_x * np.std(valley_v))]
        else:
            background, zero = est_background(wavelengths, values, central, values_units,
                                              dw * 2, xw, peaks=None, valleys=None)
            return background, zero

        #zero point is the total average
        zero = np.mean(np.append(peak_v,valley_v))

        if len(peak_v) > 2:
            peak_background = np.mean(peak_v) - zero
            #peak_background = np.std(peak_v)**2
        else:
            background, zero = est_background(wavelengths,values,central,values_units,dw*2,xw,peaks=None,valleys=None)
            return background, zero

       # since looking for emission, not absorption, don't care about the valley background
       # if len(valley_v) > 2: #expected to be negavive
       #     valley_background = np.mean(valley_v) - zero
       #     #valley_background = np.std(valley_v) ** 2
       # else:
       #     valley_background = DEFAULT_BACKGROUND

        background = peak_background

    except:
        log.error("Exception estimating background: ", exc_info=True)

    return background, zero


#todo: detect and estimate contiuum (? as SNR or mean value? over some range(s) of wavelength?)
# ie. might have contiuum over just part of the spectra
def est_continuum(wavengths,values,central):
    pass

#todo: actual signal
def est_signal(wavelengths,values,central,xw=None,zero=0.0):
    pass

#todo: actual noise, not just the local background
def est_noise():
    pass


def unique_peak(spec,wave,cwave,fwhm,width=10.0,frac=0.9,absorber=False):
    """
    Is the peak at cwave relatively unique (is it the highest within some range
    :param spec:
    :param wave:
    :param cwave:
    :param fwhm:
    :param width: number of angstroms to look to either side of the peak (and sd)
    :param frac: fraction of peak value to compare
    :return:
    """

    try:
        v = copy.copy(spec)
        if absorber:
            v = invert_spectrum(wave,spec)

        peak_val = max(v[list(SU.getnearpos(wave,cwave))]) #could be +/-1 to either side (depending on binning), so use all returns
        blue_stop, *_ = SU.getnearpos(wave,cwave-fwhm)
        red_start, *_ = SU.getnearpos(wave,cwave+fwhm)

        blue_start,*_ = SU.getnearpos(wave,cwave-fwhm - width)
        red_stop, *_ = SU.getnearpos(wave,cwave+fwhm + width)

        region = np.concatenate((v[blue_start:blue_stop+1],v[red_start:red_stop+1]))
        hits = np.where(region > (frac * peak_val))[0]

        if len(hits) < 3: #1 or 2 hits could be a barely resolved doublet (or at least adjacent lines)
            return True
        else:
            log.debug(f"Peak {cwave} appears to be in noise.")
            return False

    except:
        log.debug("Exception in spectrum::unique_peak.",exc_info=True)
        return False



def est_peak_strength(wavelengths,vals,central,values_units=0,dw=DEFAULT_BACKGROUND_WIDTH,peaks=None,valleys=None,absorber=False):
    """

    :param wavelengths:
    :param values:
    :param central:
    :param dw:
    :param xw:
    :param px:
    :param pv:
    :return:
    """

    values = copy.copy(vals)

    values, values_units = norm_values(values, values_units)

    if absorber:
        values = invert_spectrum(wavelengths,values)

    sbr = None #Signal to Background Ratio  (similar to SNR)
    xw = est_fwhm(wavelengths,values,central,values_units)

    background,zero = est_background(wavelengths,values,central,values_units,dw,xw,peaks,valleys)

    if background is not None:
        # signal = nearest values (pv) to central ?? or average of a few near the central wavelength
        #signal = est_signal(wavelengths,values,central,xw,zero)

        peak_pos = getnearpos(wavelengths, central)
        try:
            peak_str = max(values[peak_pos - 1:peak_pos + 2]) - zero
        except:
            # this can fail if on very edge, but if so, we would not use it anyway
            log.debug("Raw Peak value failure for wavelength (%f) at index (%d). Cannot calculate SBR. "
                     % (central, peak_pos))
            return 0

        #signal = ((np.sqrt(signal)-zero)/2.0)**2

        if peak_str is not None:
           # sbr = (signal-background)/(background)
           sbr = peak_str/background

    return sbr


#todo: update to deal with flux instead of counts
#def simple_peaks(x,v,h=MIN_HEIGHT,delta_v=2.0,values_units=0):
def simple_peaks(x, vals, h=None, delta_v=None, values_units=0,absorber=False):
    """

    :param x:
    :param v:
    :return:  #3 arrays: index of peaks, coordinate (wavelength) of peaks, values of peaks
              2 3D arrays: index, wavelength, value for (1) peaks and (2) valleys
    """

    maxtab = []
    mintab = []

    if h is None:
        h = np.mean(v)*0.8 #assume the mean to be roughly like the continuum level ... make min height with some slop

    if delta_v is None:
        delta_v = 0.2*h

    if x is None:
        x = np.arange(len(v))

    v = copy.copy(vals)
    v, values_units = norm_values(v, values_units)

    v = np.asarray(v)
    if absorber:
        v = invert_spectrum(x,v)
    num_pix = len(v)

    if num_pix != len(x):
        log.warning('simple_peaks: Input vectors v and x must have same length')
        return None,None

    minv, maxv = np.Inf, -np.Inf
    minpos, maxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        thisv = v[i]
        if thisv > maxv:
            maxv = thisv
            maxpos = x[i]
            maxidx = i
        if thisv < minv:
            minv = thisv
            minpos = x[i]
            minidx = i
        if lookformax:
            if (thisv >= h) and (thisv < maxv - delta_v):
                #i-1 since we are now on the right side of the peak and want the index associated with max
                maxtab.append((maxidx,maxpos, maxv))
                minv = thisv
                minpos = x[i]
                lookformax = False
        else:
            if thisv > minv + delta_v:
                mintab.append((minidx,minpos, minv))
                maxv = thisv
                maxpos = x[i]
                lookformax = True

    #return np.array(maxtab)[:, 0], np.array(maxtab)[:, 1], np.array(maxtab)[:, 2]
    return np.array(maxtab), np.array(mintab)



def filter_emission_line_as_continuum(emission_list,absorption_list,central=None):
    """
    check for emission lines sandwiched between absorption lines, s|t the emission might
    actually just be continuum between two absorption lines
    :param emission_list:
    :param absorption_list:
    :return: updated emission_list, bool if central is "remvoed"
    """
    try:
        if absorption_list is None or len(absorption_list) < 2: #have to have 2 or more
            return emission_list, False

        wave_order_absorption = sorted(absorption_list, key=lambda  x: x.fit_x0)
        absorb_waves = [x.fit_x0 for x in wave_order_absorption]
        removed_central = False

        #is there an absorption line to the blue and to the red within 2x FWHM of emis?
        #or do emssion + 3 sigma overlap with absorption + its 3 sigma (BUT absorption sigma is restricted/limited)
        #what about at the extremes (blue and red end)?

        #getnearpos to blue and to red and if both within 2xFWHM of emission AND each abs(peak height -y) is similar (all 3)
        #then delete the emission line
        keep = np.full(len(emission_list),True)
        for i,emis in enumerate(emission_list):
            try:
                _,blue,_ = SU.getnearpos(absorb_waves,emis.fit_x0)
                _,_,red = SU.getnearpos(absorb_waves,emis.fit_x0)

                if blue is None or red is None:
                    continue

                #could be maxed out sigma, then this could be too small
                #these are the current max sigmas for small, medium, large, xlrg ... if we
                #are maxed out, then bump up to the next?
                #alternately consider changing the fit_sigma multiplier to x3 or x4 ?

                if hasattr(emis,"next_max_sigma"):
                    sigma = emis.next_max_sigma
                else:
                    sigma = emis.fit_sigma

                # if np.isclose(emis.fit_sigma,5.5,1e-2):
                #     sigma = 8.5
                # elif np.isclose(emis.fit_sigma,8.5,1e-2):
                #     sigma = 25.0
                # elif np.isclose(emis.fit_sigma,25,1e-2):
                #     sigma = 55.0
                # else:
                #     simga = emis.fit_sigma

                #delta_w = 2 * emis.fit_sigma * 2.355
                delta_w = 2 * sigma * 2.355

                if absorb_waves[red] - delta_w < emis.fit_x0 < absorb_waves[blue] + delta_w:
                    # check the fit_h?
                    if np.mean([absorption_list[blue].fit_y - absorption_list[blue].fit_h,
                                 absorption_list[red].fit_y-absorption_list[red].fit_h]) > 0.5 * (emis.fit_h-emis.fit_y):
                        #mark to remove emis
                        if central is not None and abs(emis.fit_x0 - central) < 6.0:
                            log.info(f"Removing (anchor) emission line at ({emis.fit_x0}:0.1f) as likely continuum "
                                     f"between absorbers.")
                            removed_central = True
                        else:
                            log.info(f"Removing emission line at ({emis.fit_x0}:0.1f) as likely continuum "
                                     f"between absorbers.")
                        keep[i] = False
            except:
                log.debug("Exception (interior) in flter_emission_line_as_continuum",exc_info=True)


        return list(np.array(emission_list)[keep]), removed_central
    except:
        log.warning("Exception in filter_emission_line_as_continuum",exc_info=True)

    return emission_list, removed_central


def sn_peakdet_no_fit(wave,spec,spec_err,dx=3,rx=2,dv=2.0,dvmx=3.0,absorber=False,spec_obj=None,return_status=False):
    """

    :param wave: x-values (wavelength)
    :param spec: v-values (spectrum values)
    :param spec_err: error on v (treat as 'noise')
    :param dx: minimum number of x-bins to trigger a possible line detection
    :param rx: like dx but just for rise and fall
    :param dv:  minimum height in value (in s/n, not native values) to trigger counting of bins
    :param dvmx: at least one point though must be >= to this in S/N
    :return:
    """

    try:
        if not (len(wave) == len(spec) == len(spec_err)):
            log.debug("Bad call to sn_peakdet(). Lengths of arrays do not match")
            if return_status:
                return [], -1
            else:
                return []

        x = np.array(wave)
        v = np.array(copy.copy(spec))
        if absorber:
            v = invert_spectrum(x,v)
        e = np.array(spec_err)
        sn = v/e
        hvi = np.where(sn > dv)[0] #hvi high v indicies (where > dv)

        if len(hvi) < 1:
            log.debug(f"sn_peak - no bins above minimum snr {dv}")
            if return_status:
                return [], 0
            else:
                return []

        pos = [] #positions to search (indicies into original wave array)
        run = [hvi[0],]
        rise = [hvi[0],] #assume start with a rise
        fall = []

        #two ways to trigger a peak:
        #several bins in a row above the SNR cut, then one below
        #or many bins in a row, that rise then fall with lengths of rise and fall above the dx length
        for h in hvi:
            if (h-1) == run[-1]: #the are adjacent in the original arrays
                #what about sharp drops in value? like multiple peaks above continuum?
                if v[h] >= v[run[-1]]: #rising
                    rise.append(h)
                    if len(rise) >= rx:
                        rise_trigger = True
                        fall = []
                else: #falling
                    fall.append(h)
                    if len(fall) >= rx: #assume the end of a line and trigger a new run
                        fall_trigger = True
                        rise = []
                if rise_trigger and fall_trigger: #call this a peak, start a new run
                    if len(run) >= dx and np.any(sn[run] >= dvmx):
                        try:
                            mx = np.nanargmax(v[run])  # find largest value in the original arrays from these indicies
                            pos.append(mx + run[0])  # append that position to pos
                        except:
                            pass
                    run = [h]  # start a new run
                    rise = [h]
                    fall = []
                    fall_trigger = False
                    rise_trigger = False
                else:
                    run.append(h)

            else: #not adjacent, are there enough in run to append?
                if len(run) >= dx and np.any(sn[run] >= dvmx):
                    try:
                        mx = np.nanargmax(v[run]) #find largest value in the original arrays from these indicies
                        pos.append(mx+run[0]) #append that position to pos
                    except:
                        pass
                run = [h] #start a new run
                rise = [h]
                fall = []
                fall_trigger = False
                rise_trigger = False
    except:
        log.error("Exception in sn_peakdet",exc_info=True)
        if return_status:
            return [], -1
        else:
            return []

    if return_status:
        return pos, len(pos)
    else:
        return pos


def sn_peakdet(wave,spec,spec_err,dx=3,rx=2,dv=2.0,dvmx=3.0,values_units=0,
            enforce_good=True,min_sigma=None,absorber=False,do_mcmc=False,allow_broad=False,spec_obj=None):
    """

    :param wave: x-values (wavelength)
    :param spec: v-values (spectrum values)
    :param spec_err: error on v (treat as 'noise')
    :param dx: minimum number of x-bins to trigger a possible line detection
    :param rx: like dx but just for rise and fall
    :param dv:  minimum height in value (in s/n, not native values) to trigger counting of bins
    :param dvmx: at least one point though must be >= to this in S/N
    :param values_units:
    :param enforce_good:
    :param min_sigma:
    :param absorber:
    :return:
    """

    eli_list = []

    try:
        if not (len(wave) == len(spec) == len(spec_err)):
            log.debug("Bad call to sn_peakdet(). Lengths of arrays do not match")
            return []

        if min_sigma is None: #GAUSS_FIT_MIN_SIGMA can change after construction, so cannot set as a parameter default
            min_sigma = GAUSS_FIT_MIN_SIGMA

        x = np.array(wave)
        v = np.array(copy.copy(spec))
        if absorber:
            v = invert_spectrum(x,v)
        e = np.array(spec_err)
        sn = v/e
        hvi = np.where(sn > dv)[0] #hvi high v indicies (where > dv)

        if len(hvi) < 1:
            log.debug(f"sn_peak - no bins above minimum snr {dv}")
            return []

        pos = [] #positions to search (indicies into original wave array)
        run = [hvi[0],]
        rise = [hvi[0],] #assume start with a rise
        fall = []

        #two ways to trigger a peak:
        #several bins in a row above the SNR cut, then one below
        #or many bins in a row, that rise then fall with lengths of rise and fall above the dx length
        for h in hvi:
            if (h-1) == run[-1]: #the are adjacent in the original arrays
                #what about sharp drops in value? like multiple peaks above continuum?
                if v[h] >= v[run[-1]]: #rising
                    rise.append(h)
                    if len(rise) >= rx:
                        rise_trigger = True
                        fall = []
                else: #falling
                    fall.append(h)
                    if len(fall) >= rx: #assume the end of a line and trigger a new run
                        fall_trigger = True
                        rise = []
                if rise_trigger and fall_trigger: #call this a peak, start a new run
                    if len(run) >= dx and np.any(sn[run] >= dvmx):
                        try:
                            mx = np.nanargmax(v[run])  # find largest value in the original arrays from these indicies
                            pos.append(mx + run[0])  # append that position to pos
                        except:
                            pass
                    run = [h]  # start a new run
                    rise = [h]
                    fall = []
                    fall_trigger = False
                    rise_trigger = False
                else:
                    run.append(h)

            else: #not adjacent, are there enough in run to append?
                if len(run) >= dx and np.any(sn[run] >= dvmx):
                    try:
                        mx = np.nanargmax(v[run]) #find largest value in the original arrays from these indicies
                        pos.append(mx+run[0]) #append that position to pos
                    except:
                        pass
                run = [h] #start a new run
                rise = [h]
                fall = []
                fall_trigger = False
                rise_trigger = False

        #now pos has the indicies in the original arrays of the highest values in runs of high S/N bins
        for p in pos:
            #log.debug(f"*** sn_peakdet {wave[p]}")
            try:
                eli = signal_score(wave, spec, spec_err, wave[p], values_units=values_units, min_sigma=min_sigma,
                               absorber=absorber,do_mcmc=do_mcmc,allow_broad=allow_broad,spec_obj=spec_obj)

                # if (eli is not None) and (eli.score > 0) and (eli.snr > 7.0) and (eli.fit_sigma > 1.6) and (eli.eqw_obs > 5.0):
                if (eli is not None) and ((not enforce_good) or eli.is_good(allow_broad=True)):
                    #extra check for broadlines the score must be higher than usual
                    #if (min_sigma < 4.0) or ((min_sigma >= 4.0) and (eli.line_score > 25.0)):
                    eli_list.append(eli)
            except:
                log.error("Exception calling signal_score in sn_peakdet",exc_info=True)

    except:
        log.error("Exception in sn_peakdet",exc_info=True)
        return []

    return combine_lines(eli_list)



def peakdet(x,vals,err=None,dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0,values_units=0,
            enforce_good=True,min_sigma=None,absorber=False,spec_obj=None):
    """
    Keeping same interface as old peakdet for convenience

    :param x:
    :param vals:
    :param err:
    :param dw:
    :param h:
    :param dh:
    :param zero:
    :param values_units:
    :param enforce_good:
    :param min_sigma:
    :param absorber:
    :return:
    """

    if (vals is None) or (len(vals) < 3):
        return [] #cannot execute

    if min_sigma is None: #GAUSS_FIT_MIN_SIGMA can change after construction, so cannot set as a default for the parameter
        min_sigma = GAUSS_FIT_MIN_SIGMA

    v = copy.copy(vals)
    eli_list = []
    eli_poor_list = []

    #always run the SNR scan
    log.info(f"Running line finder, SN ({'absorption' if absorber else 'emission'}) ...")
    eli_list = sn_peakdet(x,v,err,values_units=values_units,enforce_good=enforce_good,min_sigma=min_sigma,
                          absorber=absorber,spec_obj=spec_obj)

    #median filter DOES help for broad lines (LINE_FINDER_MEDIAN_SCAN must be odd, so %2 needs to be not zero)
    if G.LINE_FINDER_MEDIAN_SCAN > 0 and (G.LINE_FINDER_MEDIAN_SCAN % 2):
        try:
            log.info(f"Running line finder, SN with median ({G.LINE_FINDER_MEDIAN_SCAN}) filter ({'absorption' if absorber else 'emission'}) ...")
            #repeat with median filter and kick up the minimum sigma for a broadfit
            medfilter_eli_list = sn_peakdet(x,medfilt(v,G.LINE_FINDER_MEDIAN_SCAN),medfilt(err,G.LINE_FINDER_MEDIAN_SCAN),
                                           values_units=values_units,
                                           enforce_good=enforce_good,min_sigma=GOOD_BROADLINE_SIGMA,absorber=absorber,
                                           allow_broad=True,spec_obj=spec_obj)

            # medfilter_eli_list = sn_peakdet(x,gaussian_filter1d(v,5),gaussian_filter1d(err,5),values_units=values_units,
            #                                 enforce_good=enforce_good,min_sigma=GOOD_BROADLINE_SIGMA,absorber=absorber,
            #                                 allow_broad=True)

            for m in medfilter_eli_list:
                m.broadfit = True

            if medfilter_eli_list and len(medfilter_eli_list) > 0:
                eli_list += medfilter_eli_list
        except:
            log.debug("Exception in peakdet with median filter",exc_info=True)


    #just iterate over signal_score
    #G.LINE_FINDER_FULL_FIT_SCAN: -1 = hard no, do not run
    #G.LINE_FINDER_FULL_FIT_SCAN: 0 = soft run ... run if eli_list == 0
    #G.LINE_FINDER_FULL_FIT_SCAN: 1 = yes
    if (G.LINE_FINDER_FULL_FIT_SCAN > -1) and (G.LINE_FINDER_FULL_FIT_SCAN > 0 or len(eli_list) == 0) and not absorber: #this can be slow ... only run if set OR if no lines found??
        log.info(f"Running line finder, full fit ({'absorption' if absorber else 'emission'})  ...")
        quick_eli_list = []

        if G.GAUSS_FIT_DELTA_WAVE > 0:
            step = x[1]-x[0]#4.0 #AA
            fit_range = step * 0.55
            quick_fit = False
            relax_fit = True
            do_mcmc = False
            targetted_fit = True
        else:
            step = G.FLUX_WAVEBIN_WIDTH * 2
            fit_range = step * 0.55
            quick_fit = True
            relax_fit = False
            do_mcmc = False
            targetted_fit = False


        #min_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MIN if G.LIMIT_GAUSS_FIT_SIGMA_MIN is not None else 10.0
        #actually for min_sigma, None is okay
        max_sigma = G.LIMIT_GAUSS_FIT_SIGMA_MAX if G.LIMIT_GAUSS_FIT_SIGMA_MAX is not None else 10.0
        # this is very slow, but maybe use broader steps?
        #for central_wave in np.arange(G.CALFIB_WAVEGRID[0],G.CALFIB_WAVEGRID[-1]+step,step):
        # should use the actual data we have (which might be less than the full width, and skip the leading and trailing bin since nothing to either side to fit)
        for central_wave in np.arange(x[1], x[-2] + step, step):

            quick_eli = signal_score(x, v, err, central_wave, values_units=values_units, absorber=absorber,
                                     fit_range_AA=fit_range,targetted_fit=targetted_fit,do_mcmc=do_mcmc,
                                     quick_fit=quick_fit,relax_fit=relax_fit,allow_broad=False,spec_obj=spec_obj,
                                     max_sigma=max_sigma,min_sigma=G.LIMIT_GAUSS_FIT_SIGMA_MIN) #max_sigma at 10. allows small and medium (and hetdex std) fits
            #fit_range_AA is to either side, so only want some overlap so step/2 * 1.5 or step * 0.75 (may be too much of an overlap, just need a little)
            #anything greater than 0.5

            if (quick_eli is not None):
                quick_eli_list.append(quick_eli)

        refined_eli_list = combine_lines(quick_eli_list)

        for e in refined_eli_list:
            #now a full fit
            eli = signal_score(x, v, err, e.fit_x0,values_units=values_units,
                               min_sigma=min_sigma, absorber=absorber,spec_obj=spec_obj,
                               fit_range_AA=fit_range, targetted_fit=targetted_fit, do_mcmc=do_mcmc,
                               quick_fit=quick_fit, relax_fit=relax_fit, allow_broad=False,
                               max_sigma=max_sigma)

            if (eli is not None):
                if ((not enforce_good) or eli.is_good()):
                    eli_list.append(eli)
                else:
                    eli_poor_list.append(eli)

    #condense the list
    log.debug(f"Peakdet results (pre-condense): {len(eli_list)} possible {'absorption' if absorber else 'emission'} "
              f"lines at: {[e.fit_x0 for e in eli_list]}")

    if len(eli_list) == 0:
        log.debug(
            f"Peakdet results (pre-condense) poor list: {len(eli_poor_list)} possible {'absorption' if absorber else 'emission'} "
            f"lines at: {[e.fit_x0 for e in eli_poor_list]}")
        combined_eli = combine_lines(eli_poor_list, sep=6.0)
        if len(combined_eli) > 1:
            try:
                idx = np.nanargmax([e.raw_line_score for e in combined_eli])
                combined_eli = [combined_eli[idx]]
            except:
                log.warning("Exception logging peakdet list", exc_info=True)
    else:
        combined_eli = combine_lines(eli_list,sep=6.0)

    try:
        log.info(f"Peakdet results: {len(combined_eli)} possible {'absorption' if absorber else 'emission'} "
                  f"lines at: {[e.fit_x0 for e in combined_eli]}")
        #print(f"*****Peakdet, {len(combined_eli)} possible {lt} lines at:\n{[e.fit_x0 for e in combined_eli]} ")
    except:
        log.warning("Exception logging peakdet list",exc_info=True)
    return combined_eli
#end (new) peakdet



def combine_lines(eli_list,sep=4.0):
    """

    :param eli_list:
    :param sep: max peak separation in AA (for peakdet values, true duplicates are very close, sub AA close)
    :return:
    """

    def is_dup(wave1,wave2,sep):
        if abs(wave1-wave2) < sep:
            return True
        else:
            return False
    #todo: consider the sigma of the fit (broad fit of higher score should consume narrow fit nearby if overlap)

    keep_list = []
    for e in eli_list:
        add = True
        for i in range(len(keep_list)):

            try:
                match_sep = max(2.0 * max(e.fit_sigma,keep_list[i].fit_sigma), sep)
            except:
                match_sep = sep

            if abs(e.fit_x0 - keep_list[i].fit_x0) < match_sep:
                add = False
                #keep the larger score
                try:
                    #if this was a result of a "quick fit", there is no raw_line_score,
                    #so we fall back to the line_score assigned by the quick fit (which is not the same as a proper
                    #emission line score from the build() call
                    if e.raw_line_score is not None and keep_list[i].raw_line_score is not None:
                        if e.raw_line_score > keep_list[i].raw_line_score:
                            keep_list[i] = copy.deepcopy(e)
                    elif e.line_score is not None and keep_list[i].line_score is not None:
                        if e.line_score > keep_list[i].line_score:
                            keep_list[i] = copy.deepcopy(e)
                    elif e.snr is not None and keep_list[i].snr is not None:
                        if e.snr > keep_list[i].snr:
                            keep_list[i] = copy.deepcopy(e)
                except:
                    log.debug("Exception in combine_lines",exc_info=True)


        if add:
            keep_list.append(copy.deepcopy(e))

    return keep_list



class EmissionLine:
    def __init__(self,name,w_rest,plot_color,solution=True,display=True,z=0,score=0.0,rank=0,
                 min_fwhm=999.0,min_obs_wave=9999.0,max_obs_wave=9999.0,broad=False,score_multiplier=1.0,
                 absorber=False, rest_ew = None):

        ###################################################################
        # for the "base" definition of lines
        ###################################################################

        self.name = name
        self.w_rest = w_rest
        self.w_obs = w_rest * (1.0 + z)
        self.z = z
        self.color = plot_color
        self.solution = solution #True = can consider this as the target lines (as a single line solution
        self.display = display #True = plot label on full 1D plot
        self.rank = rank #indicator of the rank in solutions (from 1 to x, with 1 being high, like LyA)
                       #roughly corresponds to expected line strength (1= high, 4= low)
                       #the base idea is that if the emission line center is a "low" rank, but there are high ranks
                       #in the wavelength range that are not found, the line may not be real

        self.min_fwhm = min_fwhm #FWHM in AA, if solution is FALSE but measured FWHM is above this value, can still be considered
                          #as a single line solution (ie. CIII, CIV, MgII ... really broad in AGN and could be alone)
        self.min_obs_wave = min_obs_wave    #if solution is FALSE, but observed wave between these values, can still be a solution
        self.max_obs_wave = max_obs_wave
        self.score_multiplier = score_multiplier #used to tune value based on the line identification
            #for example, LyA+NV commonly similar to OII+something at 3800AA, so the fit to line of NV may
            #score well, but should down-scale it because of the confusion

        #both part of the definition and in individual instances
        self.absorber = absorber #true if an abosrption line
        self.rest_ew_range =  rest_ew #this should be a good bit wider than what is expected as it is used as a hard cutoff
                                      # so, if you expect say, 2AA to 30AA, maybe set to 0 to 35 or 40AA

        ####################################################################
        #  for specific instances of emission (or absorption) lines
        #  i.e. in an observed object
        ####################################################################

        #can be filled in later if a specific instance is created and a model fit to it
        self.score = score
        self.snr = None
        self.sbr = None
        self.chi2 = 0.0
        self.flux = None
        self.flux_err = 0.0
        self.eqw_obs = None
        self.eqw_rest = None
        self.sigma = None #gaussian fit sigma
        self.sigma_err = 0.0

        #a bit redundant with EmissionLineInfo
        self.line_score = 0.0
        self.raw_line_score = 0.0
        self.prob_noise = 1.0
        self.fit_dx0 = None #from EmissionLineInfo, but we want this for a later comparison of all other lines in the solution


        if G.ALLOW_BROADLINE_FIT:
            self.broad = broad #this can be a potentially very broad line (i.e. may be with an AGN)
        else:
            self.broad = False

        self.eli = None #optional may obtain EmissionLineInfo object

    def redshift(self,z):
        """
        Notice! This sets the redshift and the observed wavelength
        :param z:
        :return:
        """
        self.z = z
        self.w_obs = self.w_rest * (1.0 + z)
        return self.w_obs

    @property
    def see_in_emission(self):
        return not self.absorber

    @property
    def see_in_absorption(self):
        return self.absorber



class Classifier_Solution:
    def __init__(self,w=0.0):
        self.score = 0.0
        self.frac_score = 0.0
        self.scale_score = -1.0 #right now, not computed until the end, in hetdex.py multiline_solution_score()
        self.z = 0.0
        self.central_rest = w
        self.name = ""
        self.color = None
        self.emission_line = None
        self.possible_pn = 0 #the higher the number, the more "possible"

        self.prob_noise = 1.0
        self.lines = [] #list of EmissionLine
        self.rejected_lines = [] #list of EmissionLines that were scored as okay, but rejected for other reasons

        #self.unmatched_lines = [] #not bothering to track the specific lines after computing score & count
        self.unmatched_lines_score = 0
        self.unmatched_lines_count = 0

        #
        self.galaxy_mask_d25 = None #mark if in galaxy mask
        self.separation = 0 #set to angular separation if this solution comes from an external catalog (like SDSS)

        self.external_catalog_solution = False #set to true if this came from an external catalog
        self.photz_zPDF_boosted = 0 #score has been boosted by one (or more) photz PDFs (count of the # of boosts)
        self.specz_boosted = 0 #ditto but for specz
        self.photz_single_boosted = 0 #ditto but for a fixed, single value photz w/o a zPDF

    @property
    def prob_real(self):
        #return min(1-self.prob_noise,0.999) * min(1.0, max(0.67,self.score/G.MULTILINE_MIN_SOLUTION_SCORE))
        try:
            if self.prob_noise == 1.0 and self.score > 10: #this can't be right
                self.prob_noise = np.prod([x.prob_noise for x in self.lines])
        except:
            log.error("Error computing prob_real for solution.")

        return min(1 - self.prob_noise, 0.999) * min(1.0, float(len(self.lines))/(G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY))# + 1.0))


    def calc_score(self):
        self.score = 0.0
        self.prob_noise = 1.0

        for l in self.lines:
            self.score += l.line_score  # score for this solution
            self.prob_noise *= l.prob_noise

        n = len(np.where([l.absorber == False for l in self.lines])[0])

        if n >= G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY:
            bonus = 0.5 * (n ** 2 - n) * G.ADDL_LINE_SCORE_BONUS  # could be negative
            self.score += bonus

        return self.score


class PanaceaSpectrum:
    """
    identification, etc from Panacea
    the operable unit
    now from HDF5
    """

    def __init__(self):
        #address
        self.amp = None
        self.ifuid = None
        self.ifuslot = None
        self.specid = None
        self.fibnum = None

        #when
        self.expnum = None
        self.obsind = None


        #coords
        self.ra = None
        self.dec = None
        self.fpx = None
        self.fpy = None
        self.ifux = None
        self.ifuy = None

        #data
        self.spectrum = None
        self.error1Dfib = None
        self.wavelength = None
        self.sky_subtracted = None

        #calibration
        self.fiber_to_fiber = None
        self.trace = None
        self.twi_spectrum = None

#end class PanaceaSpectrum


class Spectrum:
    """
    helper functions for spectra
    actual spectra data is kept in fiber.py
    """

    def __init__(self):
        #reminder ... colors don't really matter (are not used) if solution is not True)
        #try to keep the name in 4 characters
        w = 4
        self.max_rank = 5 #largest rank in line ranking below
        self.lya_mcmc_dict = None #used if check for Blue peak and one is possibly found

        #2021-06-14 ... now emission AND absorption lines
        # *** for a line that can be either, needs two entries ...

        self.emission_lines = [
            #extras for HW
            # EmissionLine("H$\\alpha$".ljust(w), 6562.8, "blue"),
            # EmissionLine("NaII".ljust(w),6549.0,"lightcoral",solution=True, display=True),
            # EmissionLine("NaII".ljust(w),6583.0,"lightcoral",solution=True, display=True),
            # EmissionLine("Pa$\\beta$".ljust(w),12818.0,"lightcoral",solution=True, display=True),
            # EmissionLine("Pa$\\alpha$".ljust(w),18751.0,"lightcoral",solution=True, display=True),

            #solution == can be a single line solution .... if False, only counts as a possible solution if
            # there is at least one corroborating line
            # see (among others) https://ned.ipac.caltech.edu/level5/Netzer/Netzer2_1.html

            ###################
            # EMITTERS
            ##################

            EmissionLine("Ly$\\alpha$".ljust(w), G.LyA_rest, 'red',rank=1,broad=True),

            EmissionLine("OII".ljust(w), G.OII_rest, 'green',rank=1,broad=True), #not as broad, but can be > 20AA
            #EmissionLine("OIII".ljust(w), 3132, "lime",rank=3),
            EmissionLine("OIII".ljust(w), G.OIII_4959, "lime",rank=3),#4960.295 (vacuum) 4958.911 (air)
            EmissionLine("OIII".ljust(w), G.OIII_5007, "lime",rank=1), #5008.240 (vacuum) 5006.843 (air)
            #EmissionLine("OIV".ljust(w), 1400, "lime", solution=False, display=True, rank=4),  # or 1393-1403 also OIV]
            # (alone after LyA falls off red end, no max wave)
            #OVI doublet 1032 and 1037
            EmissionLine("OVI".ljust(w), G.OVI_1035, "lime",solution=False,display=True,rank=3,
                         min_fwhm=12.0,min_obs_wave=G.Hbeta_4861-20.,max_obs_wave=5540.0+20.),

            # big in AGN (never alone in our range)
            # EWrest for CIV can get pretty big (100-150AA confirmed),
            EmissionLine("CIV".ljust(w), G.CIV_1549, "blueviolet",solution=True,display=True,rank=3,broad=True),
            # big in AGN (alone before CIV enters from blue and after MgII exits to red) [HeII too unreliable to set max_obs_wave]
            # note: some docs have CIII-1909 eqw up to 27AA or so, so give it some room; most lower (few AA to 10-15 ish)
            # high LyC leakers might have it at that high end of 20-25AA .. given our uncertainties on lineflux and continuum
            # figure we can be off by 40-50% absolute worst case in EW ... though statsitcally more like 30-35%
            EmissionLine("CIII".ljust(w), G.CIII_1909, "purple",solution=True,display=True,rank=3,broad=True,
                         min_fwhm=12.0,min_obs_wave=3751.0-20.0,max_obs_wave=4313.0+20.0, rest_ew=(0,40.0)),
            #big in AGN (too weak to be alone)
            EmissionLine("CII".ljust(w), G.CII_2326, "purple",solution=False,display=True,rank=4,broad=True),  # in AGN

            #big in AGN (alone before CIII enters from the blue )  this MgII is a doublet, 2795, 2802 ... can sometimes
            #  see the doublet in the HETDEX spectrum
            # What about when combined with OII 3277 (MgII maybe broad, but OII is not?)
            EmissionLine("MgII".ljust(w), G.MgII_2799, "magenta",solution=True,display=True,rank=2,broad=True,
                         min_fwhm=12.0,min_obs_wave=3500.0-20.0, max_obs_wave=5131.0+20.0),

            #thse H_x lines are never alone (OIII or OII are always present)
            EmissionLine("H$\\beta$".ljust(w), G.Hbeta_4861, "blue",solution=True,rank=3), #4862.68 (vacuum) 4861.363 (air)
            EmissionLine("H$\\gamma$".ljust(w), G.Hgamma_4340, "royalblue",solution=True,rank=4),

            EmissionLine("H$\\delta$".ljust(w), G.Hdelta_4101, "royalblue", solution=False,display=False,rank=4),
            EmissionLine("H$\\epsilon$".ljust(w), G.Hepsilon_3970, "royalblue", solution=False,display=False,rank=5), #very close to CaII(3968)
            EmissionLine("H$\\zeta$".ljust(w), G.Hzeta_3889, "royalblue", solution=False,display=False,rank=5),
            EmissionLine("H$\\eta$".ljust(w), G.Heta_3835, "royalblue", solution=False,display=False,rank=5),

            # big in AGN, but never alone in our range
            EmissionLine("NV".ljust(w), G.NV_1241, "teal", solution=False,display=True,rank=3,broad=True,score_multiplier=0.25),

            #XX is a similar odd complex in OII galaxies we commonly see (don't know what lines)
            #EmissionLine("XX".ljust(w), 3800, "green", solution=False,display=True,rank=3,broad=True),

            EmissionLine("SiII".ljust(w), G.SiII_1260, "gray", solution=False,display=True,rank=4),
            EmissionLine("SiIV".ljust(w), G.SiIV_1400, "gray", solution=False, display=True, rank=5,broad=True), #or 1393-1403 also OIV]

            #big in AGN, but never alone in our range
            EmissionLine("HeII".ljust(w), G.HeII_1640, "orange", solution=True,display=True,rank=3),
            #EmissionLine("HeI".ljust(w), G.HeI_4471, "orange", solution=False, display=False, rank=5),
            ## maybe add HeII 2733 as well??? don't know how strong it is

            EmissionLine("NeIII".ljust(w), G.NeIII_3869, "deeppink", solution=False,display=False,rank=4),
            EmissionLine("NeIII".ljust(w), G.NeIII_3967, "deeppink", solution=False,display=False,rank=4),  #very close to CaII(3970)
            EmissionLine("NeV".ljust(w), G.NeV_3347, "deeppink", solution=False,display=False,rank=5),
            EmissionLine("NeVI".ljust(w), G.NeVI_3427, "deeppink", solution=False, display=False,rank=5),

            EmissionLine("NaI".ljust(w),G.NaI_4980,"lightcoral",solution=False, display=False,rank=4),  #4978.5 + 4982.8
            EmissionLine("NaI".ljust(w),G.NaI_5153,"lightcoral",solution=False, display=False,rank=4),  #5148.8 + 5153.4

            #########################
            #ABSORBERS
            ########################
            #stars, red galaxies
            EmissionLine("(K)CaII".ljust(w), G.CaII_K_3934, "skyblue", solution=True, display=False,rank=1,absorber=True),
            EmissionLine("(H)CaII".ljust(w), G.CaII_H_3968, "skyblue", solution=True, display=False,rank=1,absorber=True),

            #add as absorber
            # EmissionLine("SiII".ljust(w), 1260, "gray", solution=False,display=True,rank=4,absorber=True),
            # EmissionLine("SiIV".ljust(w), 1400, "gray", solution=False, display=True, rank=4,absorber=True), #or 1393-1403 also OIV]


            #merged CaII(3970) with H\$epsilon$(3970)
            #EmissionLine("CaII".ljust(w), 3970, "skyblue", solution=False, display=False)  #very close to NeIII(3967)
           ]

        self.h_and_k_waves = [G.CaII_K_3934,G.CaII_H_3968]
        self.h_and_k_mcmc = None

        self.wavelengths = []
        self.values = [] #could be fluxes or counts or something else ... right now needs to be counts
        self.errors = []
        self.values_units = 0

        self.gband_masked_correction_factor = -1

        self.noise_estimate = None
        self.noise_estimate_wave = None

        # very basic info, fit line to entire spectrum to see if there is a general slope
        #useful in identifying stars (kind of like a color U-V (ish))
        #these are all as f_lam (erg/s/cm2/AA) not flux x2AA
        self.spectrum_linear_coeff = None #index = power so [0] = onstant, [1] = 1st .... e.g. mx+b where m=[1], b=[0]
        self.spectrum_linear_coeff_err = None
        self.spectrum_linear_continuum = None #estimate from a line fit to the spectrum at the central emission
        self.spectrum_linear_continuum_err = None #error on that estimtate
        self.spectrum_slope = None
        self.spectrum_slope_err = None

        self.central = None
        self.absorber = False #set to True if the central wavelength corresponds to an absorption line
        self.estflux = None
        self.estflux_unc = None
        self.estcont = None
        self.estcont_unc = None
        self.est_g_cont = None #from the HETDEX spectrum
        self.est_g_cont_unc = None
        self.gmag = None #from the HETDEX spectrum
        self.gmag_unc = None
        self.eqw_obs = None
        self.eqw_obs_unc = None
        self.fwhm = None
        self.fwhm_unc = None


        self.central_eli = None

        self.solutions = []
        self.unmatched_solution_count = 0
        self.unmatched_solution_score = 0
        self.all_found_lines = None #EmissionLineInfo objs (want None here ... if no lines, then peakdet returns [])
        self.all_found_absorbs = None
        self.classification_label = "" #string of possible classification applied (i.e. "AGN", "low-z","star", "meteor", etc)
        self.meteor_strength = 0 #qualitative strength of meteor classification

        self.addl_fluxes = []
        self.addl_wavelengths = []
        self.addl_fluxerrs = []
        self.p_lae = None
        self.p_oii = None
        self.p_lae_oii_ratio = None
        self.p_lae_oii_ratio_range = None #[ratio, max ratio, min ratio]

        self.identifier = None #optional string to help identify in the log
        self.plot_dir = None

        self.wd_checked = 0 #0 = unchecked, -1 = NOT a WD, 1 = is a WD
        self.wd_z = -999

        #from HDF5


    def gband_masked_continuum(self, recompute=False):
        """
        Sum over observed EW for all emission and (subtract) all absorption
        and use as a correction for any gband continuum estimate
        :return:  continuum from masked spectra, error on continuum, correction to multiply into gband
        """

        gmag_cgs_cont2 = None
        gmag_cgs_cont_unc2 = None
        try:
            _errors = copy.copy(self.errors)
            if self.central is not None and self.central > 0 and self.fwhm is not None and self.fwhm > 0:
                all_line_centers = np.array([self.central])
                all_line_sigmas = np.array([round(2*self.fwhm/2.355)]) #integer rounded number of wavelength bins for a 2*sigma w
            else:
                all_line_centers = np.array([])
                all_line_sigmas = np.array([]) #integer rounded number of wavelength bins for a 2*sigma width

            if self.all_found_lines is not None and len(self.all_found_lines) > 0:
                all_line_centers = np.concatenate((all_line_centers,np.array([x.fit_x0 for x in self.all_found_lines])))
                all_line_sigmas = np.concatenate((all_line_sigmas,np.array([round(x.fit_sigma*2/G.FLUX_WAVEBIN_WIDTH)
                                                                           for x in self.all_found_lines])))

            if self.all_found_absorbs is not None and len(self.all_found_absorbs) > 0:
                all_line_centers = np.concatenate((all_line_centers,np.array([x.fit_x0 for x in self.all_found_absorbs])))
                all_line_sigmas = np.concatenate((all_line_sigmas,np.array([round(x.fit_sigma*2/G.FLUX_WAVEBIN_WIDTH)
                                                                           for x in self.all_found_absorbs])))

            if len(all_line_centers) > 0:
                max_idx = len(self.wavelengths)
                for c,s in zip(all_line_centers,all_line_sigmas):
                    idx = getnearpos(self.wavelengths,c)
                    left = max(0,idx-int(s)+1)
                    right = min(max_idx,idx+int(s)+1)
                    _errors[left:right] = 0 #setting to zero will filter it out of the get_hetdex_gmag logic


                #w/o the wavelengths masked
                gmag1, gmag_cgs_cont1, gmag_unc1, gmag_cgs_cont_unc1 = get_hetdex_gmag(
                                                                   self.values / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                                   self.wavelengths,
                                                                   self.errors / 2.0 * G.HETDEX_FLUX_BASE_CGS)

                if gmag_cgs_cont1 is not None and gmag_cgs_cont1 >= G.HETDEX_CONTINUUM_FLUX_LIMIT: #if the unmasked continuum estimate is already below our limit,
                                                                    # don't bother with the correction as it is meaningless
                    #with the wavelengths masked
                    gmag2, gmag_cgs_cont2, gmag_unc2, gmag_cgs_cont_unc2 = get_hetdex_gmag(
                                                                           self.values / 2.0 * G.HETDEX_FLUX_BASE_CGS,
                                                                           self.wavelengths,
                                                                           _errors / 2.0 * G.HETDEX_FLUX_BASE_CGS)

                    if gmag_cgs_cont2 is not None:
                        gmag_cgs_cont2 = max(G.HETDEX_CONTINUUM_FLUX_LIMIT,gmag_cgs_cont2)

                        if gmag_cgs_cont1 > 0 and gmag_cgs_cont2 > 0:
                            self.gband_masked_correction_factor = gmag_cgs_cont2/gmag_cgs_cont1
                            log.info(f"g-band correction factor = {self.gband_masked_correction_factor}")
                        else:
                            self.gband_masked_correction_factor = 1.0
                    else:
                        self.gband_masked_correction_factor = 1.0
                else:
                    self.gband_masked_correction_factor = 1.0
            else:
                self.gband_masked_correction_factor = 1.0
        except:
            log.info("Exception in gband_masked_correction.",exc_info=True)

        # try:
        #     if use_ew: #use the EW logic
        #         #yes, ignoring uncertainties on the observed EW
        #         if self.all_found_lines:
        #             emis = np.sum([l.eqw_obs if (l.snr > GOOD_MIN_LINE_SNR and l.line_score > GOOD_MIN_LINE_SCORE)
        #                            else 0 for l in self.all_found_lines ])
        #         else:
        #             emis = 0
        #
        #         if self.all_found_absorbs:
        #             absorb = np.sum([abs(l.eqw_obs) if (l.snr > GOOD_MIN_LINE_SNR and l.line_score > GOOD_MIN_LINE_SCORE)
        #                              else 0 for l in self.all_found_absorbs ])
        #         else:
        #             absorb = 0
        #
        #         self.gband_masked_correction_factor = 1.0 - (emis-absorb)/(G.HETDEX_RED_SAFE_WAVE-G.HETDEX_BLUE_SAFE_WAVE+1)
        #         log.info(f"g-band correction factor = {self.gband_masked_correction_factor}")
        #
        # except:
        #     log.info("Exception in gband_masked_correction.",exc_info=True)
        #     return 1.0

        if not (0.0 <= self.gband_masked_correction_factor <= 1.0):
            log.warning(f"Nonsense gband continuum correction factor {self.gband_masked_correction_factor}. Resetting.")
            self.gband_masked_correction_factor = 1.0

        return gmag_cgs_cont2, gmag_cgs_cont_unc2, self.gband_masked_correction_factor


    def gband_masked_correction(self, recompute=False):
        """

        :param recompute:
        :return:   correction to multiply into gband
        """

        try:
            if not recompute and (0.0 <= self.gband_masked_correction_factor <= 1.0):
                return self.gband_masked_correction_factor

            #don't care about the actual continuum returns, just set the gband_masked_correction_factor
            self.gband_masked_continuum()

        except:
            log.info("Exception in gband_masked_correction.",exc_info=True)

        if not (0.0 <= self.gband_masked_correction_factor <= 1.0):
            log.warning(f"Nonsense gband continuum correction factor {self.gband_masked_correction_factor}. Resetting.")
            self.gband_masked_correction_factor = 1.0

        return self.gband_masked_correction_factor



    def rescore(self,sum_score=None,undo_absorb_penalty=False):
        """
        Rescore solutons based on changes to individual solution scores
        :param sum_score:
        :return:
        """
        try:
            if not sum_score:
                sum_score = np.sum(s.score for s in self.solutions)

            for s in self.solutions:
                if sum_score != 0:
                    try:
                        if undo_absorb_penalty and G.CONTINUUM_RULES and s.emission_line.absorber and \
                                s.central_rest in self.h_and_k_waves and ABSORPTION_LINE_SCORE_SCALE_FACTOR != 0: #h&k
                            s.score /= ABSORPTION_LINE_SCORE_SCALE_FACTOR
                            log.info(f"Undo absorption line down-scoring for H&K solution.")
                    except:
                        log.debug("Exception.",exc_info=True)


                    if s.external_catalog_solution:
                        #this came from an external catalog only, it is not an elixer solution with a catalog boost
                        #spec z keeps the most
                        #photz_zPDF_boosted keeps the next most
                        #normal photz keeps the least
                        if s.specz_boosted:
                            pass #leave as is
                        elif s.photz_zPDF_boosted:
                            old_score = s.score
                            s.score /= 2.0
                            log.debug(f"Reducing score of external catalog photz with zPDF. {s.name}, z={s.z:0.4f} Old score ({old_score:0.4f}), new score ({s.score:0.4f})")
                        else:
                            old_score = s.score
                            s.score /= 4.0
                            log.debug(f"Reducing score of external catalog photz without zPDF. {s.name}, z={s.z:0.4f} Old score ({old_score:0.4f}), new score ({s.score:0.4f})")



                    s.frac_score = s.score/sum_score
                    s.scale_score = s.prob_real * G.MULTILINE_WEIGHT_PROB_REAL + \
                                    min(1.0, s.score / G.MULTILINE_FULL_SOLUTION_SCORE) *  G.MULTILINE_WEIGHT_SOLUTION_SCORE + \
                                    s.frac_score * G.MULTILINE_WEIGHT_FRAC_SCORE
                else:
                    s.frac_score = 0
                    s.scale_score = 0

            #sort by score
            self.solutions.sort(key=lambda x: x.scale_score, reverse=True)

            for s in self.solutions:
                ll =""
                for l in s.lines:
                    ll += " %s(%0.1f at %0.1f)," %(l.name,l.w_rest,l.w_obs)
                msg = "Rescored Possible Solution %s (%0.3f): %s (%0.1f at %0.1f), Frac = %0.2f, Score = %0.1f (%0.3f), z = %0.5f, +lines=%d %s" \
                      % (self.identifier, s.prob_real,s.emission_line.name,s.central_rest,s.central_rest*(1.0+s.z), s.frac_score,
                         s.score,s.scale_score,s.z, len(s.lines),ll )
                log.info(msg)

        except:
            log.warning("Exception rescoring solutions.",exc_info=True)

    def match_line(self,obs_w,z,z_error=None,z_frac_err=None,aa_error=8.0,allow_emission=True,allow_absorption=False,max_rank=5,
                   line_sigma=None,continuum=False):
        """
         Given an input obsevered wavelength and a target redshift, return the matching emission line if found with the
            +/- aa_error in angstroms of the main (central) emission line


        :param obs_w:
        :param z:
        :param z_error: translate this to an AA error to allow for the match
                        for spec_z should be 0.05 in z (or smaller). For phot_z, maybe up to 0.5?
        :param z_frac_err: fractional (relative) error in z rather than the absolute error of z_error
        :param aa_error: absolute error in wavelength (in AA)
        :param allow_emission:
        :param allow_absorption:
        :param max_rank:
        :param line_sigma:
        :param continuum:
        :return:
        """

        try:
            all_match = self.match_lines(obs_w,z,z_error,z_frac_err,aa_error,allow_emission,allow_absorption,max_rank,
                                         line_sigma=line_sigma,continuum=continuum)

            if (all_match is None) or (len(all_match) == 0):
                return None
            elif len(all_match) == 1:
                return all_match[0]
            else:
                #all similar wavelength (similar z) given the match is on z, so get lowest rank
                #choose "best" ... lowest rank
                #todo: this can be an issue ... can get, say LyA and NV and CIII to all
                # be possible if the z_error is even moderalely big (~ 0.3 or so)

                #select the highest rank (if there is exactly one)
                ranks = np.array([e.rank for e in all_match])
                sel = np.array(ranks == min(ranks))
                if np.count_nonzero(sel) == 1:
                    return np.array(all_match)[sel][0]

                #still more than one
                all_match = np.array(all_match)[sel]
                #
                # #cetain pairs show up a lot, NeIII + H_delta for example, check those
                # if len(all_match) == 2:
                #     w_rest = np.array([e.w_rest for e in all_match])
                #     if G.Hdelta_4101 in w_rest and allow_absorption:
                #         sel = np.array([w_rest == G.Hdelta_4101])
                #         return all_match[sel][0]

                #lastly just choose the one that is nearest the obs_w, even though any remaining are in the acceptable range
                #least sepearation between obs_w and rest at the given z
                idx = np.nanargmin([e.w_rest * (1.0 + z) for e in all_match])
                #idx = np.argmin([e.rank for e in all_match])
                return all_match[idx]
        except:
            log.warning("Exception in Spectrum::match_line()",exc_info=True)

        return None


    def match_lines(self,obs_w,z,z_error=None,z_frac_err=None,aa_error=None,allow_emission=True,allow_absorption=False,max_rank=5,
                    line_sigma=None,continuum=False):
        """
        Like match_line, but plural. Can return multiple lines

        Given an input obsevered wavelength and a target redshift, return the matching emission line if found with the
            +/- aa_error in angstroms of the main (central) emission line

        :param obs_w:
        :param z:
        :param z_error: translate this to an AA error to allow for the match
                        for spec_z should be 0.05 in z (or smaller). For phot_z, maybe up to 0.5?
        :param z_frac_err: fractional (relative) error in z rather than the absolute error of z_error
        :param aa_error: absolute error in wavelength (in AA)
        :param allow_emission:
        :param allow_absorption:
        :param max_rank:  maximum allowed rank to match; if matched line is greater than rank, do not accept match
        :param line_sigma:
        :param continuum:
        :return:
        """

        try:
            all_match = []

            if continuum: #then normal  emission (like hydrogen series) can also be in absorption
                allow_absorption = True

            # continuum over rides broad and not abosorption (if there is continuum then we can allow any to be "broad"
            # and be in absorption

            if z_error is not None or z_frac_err is not None:
                #lines are far enough apart that we don't need to worry about multiple matches
                for e in self.emission_lines:
                    if (e.rank <= max_rank) and ((e.see_in_emission and allow_emission) or (e.see_in_absorption and allow_absorption)):

                        if z_frac_err is not None:
                            bid_z = (obs_w/e.w_rest) - 1.0
                            z_error = utils.fracdiff(bid_z,z,0.0,0.0)[0] #assuming no error in z (it ends up being rather small anyway)
                            ok_condition = z_error <= z_frac_err
                        else:
                            ok_condition = (e.w_rest * (1.0 + max(z-z_error,0))) <=  obs_w  <= (e.w_rest * (1.0 + z+z_error))

                        if ok_condition:
                            if not continuum and (line_sigma is not None) and (line_sigma > 6.0):
                                if e.broad:
                                    all_match.append(e)
                                else:
                                    pass
                            else:
                                #could match to a faint line by happenstance and not really be that line
                                #but this is position only
                                all_match.append(e)
            elif aa_error is not None:
                #lines are far enough apart that we don't need to worry about multiple matches
                for e in self.emission_lines:
                    if (e.rank <= max_rank) and ((e.see_in_emission and allow_emission) or (e.see_in_absorption and allow_absorption)):
                        if abs(e.w_rest * (1.0 + z) - obs_w) < aa_error:
                            if not continuum and (line_sigma is not None) and (line_sigma > 6.0):
                                if e.broad:
                                    all_match.append(e)
                                else:
                                    pass
                            else:
                                # could match to a faint line by happenstance and not really be that line
                                # but this is position only
                                all_match.append(e)
            else:
                log.debug("Invalid parameters passed to Spectrum::match_lines()")


            return all_match

        except:
            log.warning("Exception in Spectrum::match_lines()",exc_info=True)

        return None



    def match_found_lines(self,z,z_error=None,z_frac_err=None,aa_error=None,allow_emission=True,allow_absorption=False,
                    line_sigma=None,continuum=False,max_rank=5):
        """

        Like match_lines, but checks to see if any of the found lines are consistent with a line ID given the redshift

        Given an input obsevered wavelength and a target redshift, return the matching emission line if found with the
            +/- aa_error in angstroms of the main (central) emission line

        :param obs_w:
        :param z:
        :param z_error: translate this to an AA error to allow for the match
                        for spec_z should be 0.05 in z (or smaller). For phot_z, maybe up to 0.5?
        :param z_frac_err: fractional (relative) error on z

        :param aa_error: in angstroms
        :param max_rank: maximum allowed rank to match; if matched line is greater than rank, do not accept match
        :return:
        """
        try:

            if self.all_found_lines is None:
                return []

            all_match = []

            if continuum: #then normal  emission (like hydrogen series) can also be in absorption
                allow_absorption = True

            # continuum over rides broad and not abosorption (if there is continuum then we can allow any to be "broad"
            # and be in absorption

            for e in self.all_found_lines:
                match = self.match_line(e.fit_x0,z,z_error=z_error,z_frac_err=z_frac_err,aa_error=aa_error, allow_absorption=allow_absorption,
                                max_rank=max_rank)
                if match is not None:
                    all_match.append(match)


            if all_match is not None and len(all_match) > 1:
                _,unique_found_lines_sel=np.unique([x.w_rest for x in all_match],return_index=True)
                all_match = np.array(all_match)[unique_found_lines_sel]

            return all_match

        except:
            log.warning("Exception in Spectrum::match_found_lines()",exc_info=True)

        return []



    def add_classification_label(self,label="",prepend=False,replace=False):
        """

        :param label: text label
        :param prepend: place the label at the front of the list of labels
        :param replace: replace the list of labels with this single label
        :return:
        """
        try:
            if replace or self.classification_label is None:
                self.classification_label = label
            else:
                toks = self.classification_label.split(",")
                if label not in toks:
                    if prepend:
                        self.classification_label = label + "," + self.classification_label
                    else:
                        self.classification_label += label + ","
            log.debug(f"Added classification label: {label}")
        except:
            log.warning("Unexpected exception",exc_info=True)

    def scale_consistency_score_to_solution_score_factor(self,consistency_score):
        """
        take the score from the various solution_consistent_with_xxx and
        scale it to appropriate use for the solution scoring

        :param consistency_score:
        :return:
        """
        #todo: figure out what this should be;
        #todo: set a true upper limit (2-3x or so)

        upper_limit = 3.0
        lower_limit = 0.2
        if consistency_score < -4:
            #consistency_score = 0.0
            log.info(f"Consistency score penalty limited to {lower_limit} with score {consistency_score}")
            consistency_score = lower_limit #limit to 0.2
        elif consistency_score < 0:
            #already negative so, a -1 --> 1/2, -2 --> 1/3 and so on
            consistency_score = -1./(consistency_score-1.)
        else:
            consistency_score += 1.0

        return min(max(consistency_score, lower_limit), upper_limit) #don't let it be zero

    # # actually impemented in hetdex.py DetObj.check_for_meteor() as it is more convenient to do so there
    # # were the individual exposures and fibers are readily available
    # def solution_consistent_with_meteor(self, solution):
    #     """
    #
    #     if there is (positive) consistency (lines match and ratios match) you get a boost
    #     if there is no consistency (that is, the lines don't match up) you get no change
    #     if there is anti-consistency (the lines match up but are inconsistent by ratio, you can get a score decrease)
    #
    #
    #     :param solution:
    #     :return: +1 point for each pair of lines that are consistent (or -1 for ones that are anti-consistent)
    #     """
    #
    #     # check the lines, are they consistent with low z OII galaxy?
    #     try:
    #         pass
    #     except:
    #         log.info("Exception in Spectrum::solution_consistent_with_meteor", exc_info=True)
    #         return 0
    #
    #     return 0

    #todo:
    def solution_consistent_with_star(self,solution):
        """

        if there is (positive) consistency (lines match and ratios match) you get a boost
        if there is no consistency (that is, the lines don't match up) you get no change
        if there is anti-consistency (the lines match up but are inconsistent by ratio, you can get a score decrease)


        :param solution:
        :return: +1 point for each pair of lines that are consistent (or -1 for ones that are anti-consistent)
        """

        # check the lines, are they consistent with low z OII galaxy?
        try:
            pass
        except:
            log.info("Exception in Spectrum::solution_consistent_with_star", exc_info=True)
            return 0

        return 0

    def detection_consistent_with_wd(self):
        """
        This is a little different in that this applies to the detction all absorbers, not a specific solution, since
        the solution logic does not attempt to fit most of these lines

        basically, do we see the hydrogen series in absorption (or helium, oxygen, carbon)
        probably will not handle faint, cool WDs
        Seems to do well with DAs, not sure about the others

        only positive ... do not return a penality if there is no solid match

        In truth, these could also be A-stars, esp if the Hydrogen lines are narrow and if H&K are found (though
         they are sometimes found in WD too)

        :return: 0/1 if not consistent /constent, redshift
        """

        #
        try:
            if self.spectrum_slope is not None and self.spectrum_slope_err is not None:
                slope = self.spectrum_slope + self.spectrum_slope_err
            else:
                slope = 0 #unknown

            min_wd_sigma = 8.0

            if self.all_found_absorbs is None or (len(self.all_found_absorbs) < 4) or slope > 0: #really expect all these lines, so if too few, just bail
                                                                        #or if it is red (positive slope) it is not a WD
                return 0,-999, ""

            def match_lines(obs_waves, obs_sigmas, target_waves, target_points,threshold_points):
                #be careful, this is a local definition of match_lines() and is NOT THE SAME
                #as the spectrum::match_lines()
                try:
                    z_list = [] #if there is a matched line, add the redshift to the z_list and expect there to be 4+ and all similar and near zero
                    z_points = 0
                    sigma_list = []

                    for w,lw in zip(obs_waves,obs_sigmas):
                        w1 = w - max(4.0,lw/2.0)
                        w2 = w + max(4.0,lw/2.0)
                        s = np.array(target_waves > w1 ) & np.array(target_waves < w2)
                        if np.count_nonzero(s) == 1: #1 match
                            z_list.append(w/target_waves[s][0] - 1.0)
                            z_points += target_points[s][0]
                            sigma_list.append(lw)


                    if z_points > threshold_points:
                        z_mean = np.mean(z_list)
                        z_std = np.std(z_list)

                        if (-0.05 < z_mean < 0.05) and (z_std < 0.01):
                            return 1, z_mean, np.median(sigma_list) + np.std(sigma_list)
                except:
                    pass

                return 0, -999, -999 #mean sigma of matches, use as another trigger

            line_waves = np.array([l.fit_x0 for l in self.all_found_absorbs])
            line_sigmas = np.array([l.fit_sigma for l in self.all_found_absorbs])

            #
            # This should cover the DA and DAB types
            #
            hydrogen_series = np.array([3750.158, 3770.637, 3797.904, G.Heta_3835, G.Hzeta_3889, G.Hepsilon_3970,
                                        G.Hdelta_4101, G.Hgamma_4340, G.Hbeta_4861])
            #                         12         11      10        9       He_I + H_8    epsilon,  delta,     gamma,   beta
            hydrogen_points = np.array([1,1,1,1,2,2,2,3,3])
            #others = [3888.647,3933.66,3967.470,3968.47]
            #         He_I     CaII(K), NeIII, CaII(H)

            #
            # This should cover the DB type (just Helium)
            #
            helium_series = np.array([3888.647, 4026.190, 4143.761, 4471.479, 4685.710])
            helium_points = np.array([1,1,1,1,1])


            #
            # DQ types (mostly Carbon and Oxygen)
            #
            oxygen_series = np.array([G.OII_rest, 4317.139, 4363.210, 4414.899, 4650.0, 4958.911, 5006.843])
            #                                                               4650 = CIII
            oxygen_points = np.array([1,1,1,1,1,1,1])


            #check DA, DAB
            res, z, sigma = match_lines(line_waves,line_sigmas,hydrogen_series,hydrogen_points,11)

            if res > 0:
                if sigma > min_wd_sigma:
                    log.info(f"Matched WD to DA, DAB type (hydrogen series). line sigma {sigma:0.1f}")
                    return res, z, "WDa"
                else:
                    log.info(f"Matched A-type star (hydrogen series). line sigma {sigma:0.1f}")
                    return res, z, "star"

            #check DB
            res, z, sigma = match_lines(line_waves,line_sigmas,helium_series,helium_points,4)
            if res > 0:
                if sigma > min_wd_sigma:
                    log.info(f"Matched WD to DB type (helium series). line sigma {sigma:0.1f}")
                    return res, z, "WDb"
                else:
                    log.info(f"Matched (helium series). line sigma {sigma:0.1f}")
                    return res, z, "star"

            #check DQ
            res, z, sigma = match_lines(line_waves,line_sigmas,oxygen_series,oxygen_points,5)
            if res > 0:
                if sigma > min_wd_sigma:
                    log.info(f"Matched WD to DQ type (carbon + oxygen series). line sigma {sigma:0.1f}")
                    return res, z, "WDq"
                else:
                    log.info(f"Matched star (carbon + oxygen series). line sigma {sigma:0.1f}")
                    return res, z, "star"

        except:
            log.info("Exception in Spectrum::detection_consistent_with_wd", exc_info=True)
            return 0, -999, ""

        return 0, -999, ""

    def solution_consistent_with_H_and_K(self,solution):
        """
        For continuum sources at low-z, we often find the H & K lines (CaII 3968 [H] and 3934 [K])
        To be consistent we need positive continuum and both lines (but not necessarirly a red slope ... could be a G
        or hotter star and still see H&K and would be blue ove our range)

        :param soultion:
        :return:
        """

        score = 0
        try:
            #here we want absorption lines
            sel = np.where(np.array([l.absorber for l in solution.lines]) == True)[0]
            sol_lines = np.array(solution.lines)[sel]
            line_waves = np.array([solution.central_rest] + [l.w_rest for l in sol_lines])
            #line_ew = [self.eqw_obs / (1 + solution.z)] + [l.eqw_rest for l in sol_lines]
            #line_flux is maybe more reliable ... the continuum estimates for line_ew can go wrong and give horrible results
            line_flux = [self.estflux] + [l.flux for l in sol_lines]
            line_flux_err = [self.estflux_unc] + [l.flux_err for l in sol_lines]
            line_fwhm = [self.fwhm] + [l.sigma * 2.355 for l in sol_lines]
            line_fwhm_err = [self.fwhm_unc] + [l.sigma_err * 2.355 for l in sol_lines]


            try:
                k = np.where(line_waves==G.CaII_K_3934)[0][0]
                h = np.where(line_waves==G.CaII_H_3968)[0][0]
            except:
                #did not find them
                return score

            #should have similar flux and fwhm and depth
            # NOTE: there might be some useful stellar population physics encoded in the ratio but for now, will
            # settle for similar (say/ withint 30%?)

            #sanity
            if np.sign(line_flux[k]) != np.sign(line_flux[h]):
                return 0

            #don't bother with the uncertainty since we are letting this be pretty lax
            if abs((line_flux[h]-line_flux[k])/(0.5*(line_flux[h]+line_flux[k]))) < 0.5:
                if abs((line_fwhm[h]-line_fwhm[k])/(0.5*(line_fwhm[h]+line_fwhm[k]))) < 0.5:
                    return 1.0

        except:
            log.info("Exception in Spectrum::solution_consistent_with_H_and_K", exc_info=True)
            return 0

        return score


    def solution_consistent_with_low_z(self,solution):
        """

        if there is (positive) consistency (lines match and ratios match) you get a boost
        if there is no consistency (that is, the lines don't match up) you get no change
        if there is anti-consistency (the lines match up but are inconsistent by ratio, you can get a score decrease)


        :param solution:
        :return: +1 point for each pair of lines that are consistent (or -1 for ones that are anti-consistent)
        """

        # check the lines, are they consistent with low z OII galaxy?
        try:
            #compared to OII (3272) EW: so line ew / OII EW; a value of 0 means no info
            #               OII      NeV  NeIV H_eta   -NeIII-   H_zeta  CaII  H_eps  H_del  H_gam H_beta   -NaI-     -OIII-
            #rest_waves = [G.OII_rest,3347,3427,3835,  3869,3967, 3889,   3935, 3970,  4101,  4340, 4861,  4980,5153, 4959,5007]
            #                       OII       H_eta  H_zeta H_eps H_del  H_gam H_beta   -OIII-
            rest_waves = np.array([G.OII_rest, G.Heta_3835,  G.Hzeta_3889,  G.Hepsilon_3970, G.Hdelta_4101,
                                   G.Hgamma_4340, G.Hbeta_4861,  G.OIII_4959, G.OIII_5007])
            #                         0         1      2      3     4     5     6      7     8
            obs_waves  = rest_waves * (1. + solution.z)

            #OIII 4960 / OIII 5007 ~ 1/3
            #using as rough reference https://watermark.silverchair.com/stt151.pdf (MNRAS 430, 35103536 (2013))

            # note that H_beta and OIII can be more intense than OII
            # pretty loose since the Hx lines can really be large compared to OII in very metal poor objects
            #                                      CaII
            #             OII   H_eta      H_zeta  H_eps  H_del H_gam  H_beta  -OIII-
            min_ratios = [1,     0.01,      0.05,  0.05,  0.1, 0.15,   0.4,    0.1, 0.3]
            max_ratios = [1,     0.06,      0.20,  1.20,  0.5, 1.50,   3.3,    6.5, 20.0]

            #required match matrix ... if line at row (x) is found and line at column (y) is in range, it MUST be found too
            #this is in order of the lines in rest_waves
            # 1 for youself along the diagnonal
            # 1 also for a line that probably should be found, but is not catastrophic if not found
            # > 1 for lines that really must be found
            # !!! note: not strictly true that we must have OII-3727 if we see the Balmer lines,
            #     but for HETDEX data and the majority of late (low-z) galaxies this is so
            #     (not likely to see some pristine, supper young galaxy without detectable OII ....
            #      though there have been a few weird exampele where HETDEX picks up OIII-5007 but strangely not OII-3727)
            match_matrix =[[1,0,0,0,0,0,0,0,0],  #0 [OII]
                           [1,1,1,1,2,3,5,0,0],  #1 H_eta
                           [1,0,1,1,2,3,5,0,0],  #2 H_zeta
                           [1,0,0,1,1,2,5,0,0],  #3 H_epsilon
                           [1,0,0,0,1,1,5,0,0],  #4 H_delta
                           [1,0,0,0,0,1,5,0,0],  #5 H_gamma
                           [1,0,0,0,0,0,1,0,0],  #6 H_beta
                           [1,0,0,0,0,0,0,1,2],  #7 OIII 4959
                           [1,0,0,0,0,0,0,0,1]]  #8 OIII 5007
            match_matrix = np.array(match_matrix)

            #row/column (is mininum, where lines are smallest compared to OII)
            # the inverse is still the minimum just the inverted ratio)

            # (5007 to 4959  at 2.98 ~ 3.00  Storey  &  Zeippen  2000)

            min_ratio_matrix = \
            [ [1.00, None, None, None, None, None, 10.0, None, None],  #OII
              [None, 1.00, None, None, None, None, None, None, None],  #H_eta
              [None, None, 1.00, None, None, None, None, None, None],  #H_zeta
              [None, None, None, 1.00, None, None, None, None, None],  #H_eps
              [None, None, None, None, 1.00, None, None, None, None],  #H_del
              [None, None, None, None, None, 1.00, None, None, None],  #H_gamma
              [0.10, None, None, None, None, None, 1.00, None, None],  #H_beta
              [None, None, None, None, None, None, None, 1.00, 0.33],  #OIII 4959
              [None, None, None, None, None, None, None, 3.00, 1.00]]  #OIII 5007
             # OII   H_eta H_zet H_eps H_del H_gam H_bet  OIII OIII

            max_ratio_matrix = \
            [ [1.00, None, None, None, None, None, 1.25, None, None],  #OII
              [None, 1.00, None, None, None, None, None, None, None],  #H_eta
              [None, None, 1.00, None, None, None, None, None, None],  #H_zeta
              [None, None, None, 1.00, None, None, None, None, None],  #H_eps
              [None, None, None, None, 1.00, None, None, None, None],  #H_del
              [None, None, None, None, None, 1.00, None, None, None],  #H_gamma
              [0.80, None, None, None, None, None, 1.00, None, None],  #H_beta
              [None, None, None, None, None, None, None, 1.00, 0.33],  #OIII 4959
              [None, None, None, None, None, None, None, 3.00, 1.00]]  #OIII 5007
             # OII   H_eta H_zet H_eps H_del H_gam H_bet  OIII OIII

            sel = np.where(np.array([l.absorber for l in solution.lines]) == False)[0]
            sol_lines = np.array(solution.lines)[sel]
            line_waves = [solution.central_rest] + [l.w_rest for l in sol_lines]
            if self.central_eli is not None:
                line_snr = [self.central_eli.snr] + [l.snr for l in sol_lines]
            else:
                line_snr = [0] + [l.snr for l in sol_lines]
            #line_ew = [self.eqw_obs / (1 + solution.z)] + [l.eqw_rest for l in sol_lines]
            #line_flux is maybe more reliable ... the continuum estimates for line_ew can go wrong and give horrible results
            line_flux = [self.estflux] + [l.flux for l in sol_lines]
            line_flux_err = [self.estflux_unc] + [l.flux_err for l in sol_lines]
            line_fwhm = [self.fwhm] + [l.sigma * 2.355 for l in sol_lines]
            line_fwhm_err = [self.fwhm_unc] + [l.sigma_err * 2.355 for l in sol_lines]
            #line_rank = [solution.emission_line.rank] + [l.rank for l in sol_lines]

            overlap, rest_idx, line_idx = np.intersect1d(rest_waves, line_waves, return_indices=True)


            central_fwhm =  self.fwhm
            central_fwhm_err = self.fwhm_unc
            #todo: get samples and see if there is a correlation with slope
            #slope = self.spectrum_slope
            #slope_err = self.spectrum_slope_err

            if len(overlap) < 1:
                #todo: any fwhm that would imply low-z? more narrow?
                return 0

            #check the match_matrix
            missing = []
            missing_weight = 0
            in_range = np.where((obs_waves > G.HETDEX_BLUE_SAFE_WAVE) & (obs_waves < 5500.))[0]
            for i in range(len(overlap)):
                if np.sum(match_matrix[rest_idx[i]]) > 1:
                    #at least one other line must be found (IF the obs_wave is in the HETDEX range)
                    sel = np.intersect1d(in_range,np.where(match_matrix[rest_idx[i]])[0])
                    missing_idx = np.setdiff1d(sel,rest_idx) #for THIS row
                    missing = np.union1d(missing,missing_idx).astype(int)
                    #current
                    missing_weight += np.sum(match_matrix[rest_idx[i]][missing_idx]) # -1 #the -1 is so we don't count the line we assume we are
                    #OLD way
                    #missing_weight += np.sum(match_matrix[rest_idx[i]][sel])  -1 #the -1 is so we don't count the line we assume we are

            score = -1 * max(missing_weight,len(missing))

            #add special case: if OIII-5007 and BRIGHT then MUST find OIII-4959
            try:
                idx_5007 = np.where(np.array(rest_idx)==8)[0][0]
                if line_flux[line_idx[idx_5007]] > 1.e-16 and line_snr[line_idx[idx_5007]] > 8.0:
                    try:
                        idx_4959 = np.where(np.array(rest_idx)==7)[0][0]
                        #we did find it
                        if 5.0 < line_flux[line_idx[idx_5007]]/line_flux[line_idx[idx_4959]] < 1.5: #out of bounds even with reasonable error
                            log.info(f"OIII-5007 is bright but ratio to OIII-4959 is out of bounds "
                                     f"{line_flux[line_idx[idx_5007]]/line_flux[line_idx[idx_4959]]:0.4g}. Penalizing score.")
                            score -= 2.0
                    except:
                        #did not find 4959 and MUST find it in this case
                        log.info("OIII-5007 is bright but OIII-4959 not found. Penalizing score.")
                        score -= 2.0
            except:
                pass


            if score < 0:
                log.info(f"LzG consistency failure. Initial Score = {score}. "
                         f"Missing expected lines [(rest,obs)] {[z for z in zip(rest_waves[missing],obs_waves[missing])]}. ")
            # compare all pairs of lines

            if len(overlap) < 2:  # done (0 or 1) if only 1 line, can't go any farther with comparison
                #todo: any fwhm that would imply low-z? more narrow?
                return score


            for i in range(len(overlap)):
                for j in range(i+1,len(overlap)):
                    if (line_flux[line_idx[i]] != 0):
                        if (min_ratios[rest_idx[i]] != 0) and (min_ratios[rest_idx[j]] != 0) and \
                           (max_ratios[rest_idx[i]] != 0) and (max_ratios[rest_idx[j]] != 0):

                            ratio = line_flux[line_idx[j]] / line_flux[line_idx[i]]
                            try:
                                ratio_err = abs(ratio) * np.sqrt((line_flux_err[line_idx[j]] /line_flux[line_idx[j]]) ** 2 +
                                                                 (line_flux_err[line_idx[i]] / line_flux[line_idx[i]]) ** 2)
                            except:
                                ratio_err = 0 #line_flux for j might be zero, this will fail then

                            # try the matrices first (if they are zero, they are not populated yet
                            # so fall back to the list)
                            min_ratio = min_ratio_matrix[rest_idx[j]][rest_idx[i]]
                            max_ratio = max_ratio_matrix[rest_idx[j]][rest_idx[i]]

                            if (min_ratio is None) or (max_ratio is None):
                                min_ratio = min_ratios[rest_idx[j]] / min_ratios[rest_idx[i]]
                                max_ratio = max_ratios[rest_idx[j]] / max_ratios[rest_idx[i]]

                            if min_ratio > max_ratio: #order is backward, so flip
                                min_ratio, max_ratio = max_ratio, min_ratio

                            if min_ratio <= (ratio+ratio_err) and (ratio-ratio_err) <= max_ratio:
                                #now check fwhm is compatible
                                fwhm_i = line_fwhm[line_idx[i]]
                                fwhm_j = line_fwhm[line_idx[j]]

                                #none of the low-z lines can be super broad
                                if (fwhm_i > (LIMIT_BROAD_SIGMA * 2.355)) or (fwhm_j > (LIMIT_BROAD_SIGMA * 2.355)):
                                    score -=1
                                    log.debug(f"Ratio mis-match (-1) for solution = {solution.central_rest}: "
                                              f"FWHM {fwhm_j:0.2f}, {fwhm_i:0.2f} exceed max allowed {2.355 *LIMIT_BROAD_SIGMA} ")
                                    continue

                                fwhm_err_i = line_fwhm_err[line_idx[i]]
                                fwhm_err_j = line_fwhm_err[line_idx[j]]
                                avg_fwhm = 0.5* (fwhm_i + fwhm_j)
                                diff_fwhm = max(0,abs(fwhm_i - fwhm_j) - (fwhm_err_i+fwhm_err_j))

                                if avg_fwhm > 0 and diff_fwhm/avg_fwhm < 0.5:
                                    # bump = min(1, 2.0 / max(line_rank[i],line_rank[j]))
                                    # score += bump
                                    # log.debug(f"Ratio match (+{bump:0.2f}) for solution = {solution.central_rest}: "
                                    #           f"rest {overlap[j]} to {overlap[i]}: "
                                    #           f"{min_ratio:0.2f} < {ratio:0.2f} +/- {ratio_err:0.2f} < {max_ratio:0.2f} "
                                    #           f"FWHM {fwhm_j}, {fwhm_i}")

                                    score += 1
                                    log.debug(f"Ratio match (+1) for solution = {solution.central_rest}: "
                                              f"rest {overlap[j]} to {overlap[i]}: "
                                              f"{min_ratio:0.2f} < {ratio:0.2f} +/- {ratio_err:0.2f} < {max_ratio:0.2f} "
                                              f"FWHM {fwhm_j}, {fwhm_i}")

                                    if rest_waves[rest_idx[j]] == G.OII_rest and rest_waves[rest_idx[i]] == G.OIII_5007:
                                        if 1/ratio > 5.0:
                                            self.add_classification_label("o32")
                                    elif rest_waves[rest_idx[j]] == G.OIII_5007 and rest_waves[rest_idx[i]] == G.OII_rest:
                                        if ratio > 5.0:
                                            self.add_classification_label("o32")

                                else:
                                    log.debug(f"FWHM no match (0) for solution = {solution.central_rest}: "
                                              f"rest {overlap[j]} to {overlap[i]}: FWHM {fwhm_j}, {fwhm_i}, "
                                              f"ratios {min_ratio:0.2f} < {ratio:0.2f} +/- {ratio_err:0.2f} < {max_ratio:0.2f}")
                            else:
                                if ratio < min_ratio:
                                    frac = (min_ratio - ratio) / min_ratio
                                else:
                                    frac = (ratio - max_ratio) / max_ratio

                                if 0.5 < frac < 250.0: #if more than 250 more likely there is something wrong
                                    score -= 1
                                    log.debug(
                                        f"Ratio mismatch (-1) for solution = {solution.central_rest}: "
                                        f"rest {overlap[j]} to {overlap[i]}: "
                                        f"{min_ratio:0.2f} !< {ratio:0.2f} +/- {ratio_err:0.2f} !< {max_ratio:0.2f} ")
                                # else:
                                #     log.debug(
                                #         f"Ratio no match (0) for solution = {solution.central_rest}: "
                                #         f"rest {overlap[j]} to {overlap[i]}:  {min_ratio} !< {ratio} !< {max_ratio}")

            # todo: sther stuff??
            # if score > 0:
            #     self.add_classification_label("LzG") #Low-z Galaxy
            return score

        except:
            log.info("Exception in Spectrum::solution_consistent_with_low_z", exc_info=True)
            return 0

        return 0


    def single_broad_line_redshift(self,w,fwhm,fwhm_err=0): #,solution=None):
        """ Single broad line, possibly LyA, CIII, MgII in certain ranges
            Others would expect to see another line


            Assume only MgII or CIII if nothing else
            MgII: 3470-3760, 4421-5120
            CIII or MgII: 3760-4314
            CIV or MgII : 4421-4520 (LyA just visible at extreme blue if CIV, but due to noise myabe allow this?)
            should be no single lines > 5120 (expect LyA to have OVI as well, but maybe not)
            LyA: > 5120

            #check for shape? double peak MgII ... if no, then assume CIII (low confidence)??
            
            :param w = wavelength, observed frame in AA
            :param fwhm =  single Gaassian fitted FWHM in AA
            :param fwhm_err = uncertainty on fwhm in AA
        """
        try:

            v = 3e5 * fwhm/w
            #reduce the threshold by the fractional error on the fwhm
            #default is at 1200. km/s
            velocity_thresh = max(800.0, G.BROAD_FWHM_KMS * (1. - fwhm_err / fwhm))

            if   3470 <= w < 3760:
                return w/G.MgII_2799 - 1.0 , "MgII" #MgII
            elif 3760 <= w < 4314:
                #MgII or CIII?
                #semi arbitrary
                if v > velocity_thresh:
                    return w/G.CIII_1909 -1, "CIII" #CIII
                else:
                    return w/G.MgII_2799 - 1.0, "MgII" #MgII
            elif 4314 <= w < 4421:
                #MgII or LyA??
                if v > velocity_thresh:
                    return w/G.LyA_rest -1, "LyA" #LyA
                else:
                    return w/G.MgII_2799 - 1.0, "MgII" #MgII
            # elif 4421 <= w < 4520:
            #     #Lya or MgII
                 #after CIV falls off
            elif 4421 < w < 5120:
                #  LyA or MgII (for MgII at 5120, CIII shows up)
                if v > velocity_thresh:
                    return w/G.LyA_rest - 1.0, "LyA"
                else:
                    return w/G.MgII_2799 - 1.0, "MgII" #MgII
            elif w >= 5120:
                return w/G.LyA_rest - 1.0, "LyA"

            return None, None

        except:
            return None, None


    def single_emission_line_redshift(self,line,w):
        """
            Can this single emission line at abserved wavelength w be allowed?
            :param line = an emission line object
        """
        try:

            if line.w_rest == G.LyA_rest or line.w_rest == G.OII_rest or np.isclose(line.w_rest,G.OIII_5007,atol=2):
                #always allowed to be a single line (even OIII-5007 ... extreme O32 objects)
                return True
            elif np.isclose(line.w_rest,G.MgII_2799,atol=2):
                #MgII
                if (3470 <= w < 5120):
                    return True
            elif np.isclose(line.w_rest,G.CIII_1909,atol=2):
                #CIII
                if 3760 <= w < 4314:
                    return True
            elif np.isclose(line.w_rest,G.CIV_1549,atol=2):
                #CIV (technially never alone but in this narrow range, CIII can be at extreme red or LyA at extreme blue
                # and we don't do a good job of picking it up
                if  4460 <= w < 4492:
                    return True

            return False
        except:
            return False


    def solution_consistent_with_agn(self,solution):
        """
        REALLY AGN or higher z (should not actually include MgII)

        if there is (positive) consistency (lines match and ratios match) you get a boost
        if there is no consistency (that is, the lines don't match up) you get no change
        if there is anti-consistency (the lines match up but are inconsistent by ratio, you can get a score decrease)


        :param solution:
        :return: +1 point for each pair of lines that are consistent (or -1 for ones that are anti-consistent)
        """

        #
        # note: MgII alone (as broad line) not necessarily AGN ... can be from outflows?
        #

        #check the lines, are they consistent with AGN?
        try: #todo: for MgII, OII can also be present
            rest_waves = np.array([G.LyA_rest,G.CIV_1549,G.CIII_1909,G.CII_2326,G.MgII_2799,G.NV_1241,G.SiII_1260,G.SiIV_1400,G.HeII_1640,G.OVI_1035, G.OII_rest])
            #aka                     LyA,      CIV, CIII, CII,   MgII,  NV,  SiII, SiIV, HeII, OVI,  OII
            obs_waves = rest_waves * (1. + solution.z)

            # compared to LyA EW: so line ew/ LyA EW; a value of 0 means no info
            #todo: need info/citation on this
            #todo: might have to make a matrix if can't reliably compare to LyA
            #todo:   e.g. if can only say like NV < MgII or CIV > CIII, etc

            # using as a very rough guide: https://ned.ipac.caltech.edu/level5/Netzer/Netzer2_1.html
            # and manual HETDEX spectra
            #            #LyA, CIV,  CIII, CII,  MgII,  NV,     SiII, SiIV,  HeII,  OVI   OII
            min_ratios = [1.0, 0.07, 0.02, 0.01,  0.05, 0.05,  0.00, 0.03,   0.01,  0.03, 0.01]
            max_ratios = [1.0, 0.70, 0.30, 0.10,  0.40, 0.40,  0.00, 0.20,   0.20,  0.40, 9.99]
            #            *** apparently NV can be huge .. bigger than CIV even see 2101164104
            #            *** OII entries just to pass logic below (only appears on our range for some MgII)


            #required match matrix ... if line at row (x) is found and line at column (y) is in range, it MUST be found too
            #this is in order of the lines in rest_waves
            #in ALL cases, LyA better be found IF it is in range (so making it a 2 ... need 2 other matched lines to overcome missing LyA)
            match_matrix =[[1,0,0,0,0,0,0,0,0,0,0],  #0 LyA
                           [1,1,0,0,0,0,0,0,0,0,0],  #1 CIV #CIII, CII should be there, but in AGN they can be hard to see in EliXer
                           [1,1,1,0,0,0,0,0,0,0,0],  #2 CIII #but if we see CIII, CIV should REALLY be obvious (CIII is semi forbidden)
                           [1,1,0,1,0,0,0,0,0,0,0],  #3 CII #not sure about CII [semi forbidden]
                           [1,0,0,0,1,0,0,0,0,0,1],  #4 MgII
                           [1,0,0,0,0,1,0,0,0,0,0],  #5 NV
                           [1,1,1,0,0,1,1,0,0,0,0],  #6 SiII
                           [1,1,1,0,0,0,1,1,0,0,0],  #7 SiIV
                           [1,0,0,0,0,0,0,0,1,0,0],  #8 HeII
                           [1,0,0,0,0,0,0,0,0,1,0],  #9 OVI
                           [0,0,0,0,1,0,0,0,0,0,1] ] #10 OII (just with MgII)
                         #  0 1 2 3 4 5 6 7 8 9 10

            match_matrix = np.array(match_matrix)

            match_matrix_weights = np.array([3,1,1,0.5,1,1,0.5,0.5,1,1,2])

            #todo: 2 matrices (min and max ratios) so can put each line vs other line
            # like the match_matrix in the low-z galaxy check (but with floats) as row/column
            # INCOMPLETE ... not in USE YET

            #row/column (is mininum, where lines are smallest compared to LyA)
            # the inverse is still the minimum just the inverted ratio)
            min_ratio_matrix = \
            [ [1.00, None, None, None, None, None, None, None, None, None, None],  #LyA
              [None, 1.00, None, None, None, 9.99, 4.00, None, 6.66, None, None],  #CIV
              [None, None, 1.00, None, None, None, None, None, None, None, None],  #CIII
              [None, None, None, 1.00, None, None, None, None, None, None, None],  #CII
              [None, None, None, None, 1.00, None, None, None, None, None, 20.0],  #MgII
              [None, 0.10, None, None, None, 1.00, None, None, None, None, None],  #NV
              [None, 0.25, None, None, None, None, 1.00, None, None, None, None],  #SiII
              [None, None, None, None, None, None, None, 1.00, None, None, None],  #SiIV
              [None, 0.15, None, None, None, None, None, None, 1.00, None, None],  #HeII
              [None, None, None, None, None, None, None, None, None, 1.00, None],  #OVI
              [None, None, None, None, 0.05, None, None, None, None, None, 1.00 ]] #OII
             # LyA   CIV   CIII  CII   MgII   NV   SiII  SiVI  HeII   OVI  OII

            #row/column (is maximum ... where lines are the largest compared to LyA)
            max_ratio_matrix = \
            [ [1.00, None, None, None, None, None, None, None, None, None, None],  #LyA
              [None, 1.00, None, None, None, 0.33, 0.10, None, 1.43, None, None],  #CIV
              [None, None, 1.00, None, None, None, None, None, None, None, None],  #CIII
              [None, None, None, 1.00, None, None, None, None, None, None, None],  #CII
              [None, None, None, None, 1.00, None, None, None, None, None, 2.00],  #MgII
              [None, 3.00, None, None, None, 1.00, None, None, None, None, None],  #NV
              [None, 10.0, None, None, None, None, 1.00, None, None, None, None],  #SiII
              [None, None, None, None, None, None, None, 1.00, None, None, None],  #SiIV
              [None, 0.70, None, None, None, None, None, None, 1.00, None, None],  #HeII
              [None, None, None, None, None, None, None, None, None, 1.00, None],  #OVI
              [None, None, None, None, 0.50, None, None, None, None, None, 1.00] ] #OII
             # LyA    CIV  CIII   CII  MgII   NV   SiII  SiVI  HeII   OVI  OII

            sel = np.where(np.array([l.absorber for l in solution.lines])==False)[0]
            sol_lines = np.array(solution.lines)[sel]
            line_waves = [solution.central_rest] + [l.w_rest for l in sol_lines]
            try:
                line_raw_line_score = [self.central_eli.raw_line_score] + [l.raw_line_score for l in sol_lines]
            except:
                line_raw_line_score = [-1] + [l.raw_line_score for l in sol_lines]
            #line_ew = [self.eqw_obs/(1+solution.z)] + [l.eqw_rest for l in sol_lines]
            # line_flux is maybe more reliable ... the continuum estimates for line_ew can go wrong and give horrible results
            line_flux = [self.estflux] + [l.flux for l in sol_lines]
            line_flux_err = [self.estflux_unc] + [l.flux_err for l in sol_lines]
            line_fwhm = [self.fwhm] + [l.sigma * 2.355 for l in sol_lines]
            line_fwhm_err = [self.fwhm_unc] + [l.sigma_err * 2.355 for l in sol_lines]
            line_broad = [solution.emission_line.broad] + [l.broad for l in sol_lines] #can they be broad
            line_rank = [solution.emission_line.rank] + [l.rank for l in sol_lines]

            overlap, rest_idx, line_idx = np.intersect1d(rest_waves,line_waves,return_indices=True)

            central_fwhm =  self.fwhm
            central_fwhm_err = self.fwhm_unc
            #todo: get samples and see if there is a correlation with slope
            #slope = self.spectrum_slope
            #slope_err = self.spectrum_slope_err

            score = 0

            # check the match_matrix
            missing = []
            #missing_weight = 0 #handled by the match_matrix_weights

            #can be really hard in the blue, so start a bit red of blue most
            in_range = np.where((obs_waves > G.HETDEX_BLUE_SAFE_WAVE) & (obs_waves < 5500.))[0]
            for i in range(len(overlap)):
                if np.sum(match_matrix[rest_idx[i]]) > 1:
                    # at least one other line must be found (IF the obs_wave is in the HETDEX range)
                    sel = np.intersect1d(in_range, np.where(match_matrix[rest_idx[i]])[0])
                    missing = np.union1d(missing, np.setdiff1d(sel, rest_idx)).astype(int)
                    #missing_weight += np.sum(match_matrix[rest_idx[i]][sel]) - 1  # the -1 is so we don't count the line we assume we are

            #LyA COULD be there even if not found
            if 0 in missing:
                #LyA is NOT found in the list of lines, but this is supposedly in the 1.88 < z < 3.52 range
                #now, this *could* be an LBG or somehow LyA is very weak, but that is very unlikely for HETDEX
                #and HETDEX wants LAEs, so mark this as unlikely

                lya_missing = True
                #at the blue end in particular, the LyA line can be very hard to find, esp if broad, so
                #we need to make some allowances here
                #if our anchor line FWHM is large and the LyA should be blue of 3700, check to see if there is
                #big flux where lya should be, +/- anchor line's 2*sigma (just 2 sigma since can be ragged)
                #e.g. if it is similar to anchor line flux, then assume we just failed to fit it
                try:
                    if central_fwhm > 14.0:# and solution.z < 2.05: #z < 2.05 is about wave < 3700AA
                        center,*_ = SU.getnearpos(self.wavelengths,G.LyA_rest*(1+solution.z))
                        left = max(0,center-int(central_fwhm/2.355*2.0))
                        right = center+int(central_fwhm/2.355*2.0) #due to early checks, cannot be out of range
                        lya_int_flux = np.sum(self.values[left:right])
                        lya_int_flux_err = np.sqrt(np.sum(self.errors[left:right]**2))

                        #the outer 2sigma wings (or excluding the interior +/- 1 sigma)
                        l_left = max(0,center-int(central_fwhm/2.355*2.0))
                        l_right = max(0,center-int(central_fwhm/2.355*1.0))
                        r_left = center+int(central_fwhm/2.355*1.0) #due to early checks, cannot be out of range
                        r_right = center+int(central_fwhm/2.355*2.0) #due to early checks, cannot be out of range
                        lya_wing_flux =np.sum(self.values[l_left:l_right]) + np.sum(self.values[r_left:r_right])

                        #compare to anchor flux
                        center,*_ = SU.getnearpos(self.wavelengths,self.central)
                        left = max(0,center-int(central_fwhm/2.355*2.0))
                        right = center+int(central_fwhm/2.355*2.0) #due to early checks, cannot be out of range
                        anchor_int_flux = np.sum(self.values[left:right])
                        anchor_int_flux_err = np.sqrt(np.sum(self.errors[left:right]**2))

                        #LyA (best case) has to be larger than the non-LyA line
                        if ((lya_int_flux + lya_int_flux_err) > (anchor_int_flux-anchor_int_flux_err)) and \
                                ((lya_int_flux_err/lya_int_flux) < 0.5) and ((anchor_int_flux_err/anchor_int_flux) < 0.5) and \
                                (lya_wing_flux/lya_int_flux < 0.33): #should be < 0.28, but leaving slop for error
                            lya_missing = False
                            missing = missing[missing != 0]

                except:
                    log.warning("Exception in solution_consistent_wtih_lae attemping to check non fitted flux @ LyA.",exc_info=True)



            # score = -1 * len(missing)
            score = -1 * np.sum(match_matrix_weights[missing])

            if score < 0:
                log.info(f"AGN consistency failure. Initial Score = {score}. "
                         f"Missing expected lines {[z for z in zip(rest_waves[missing], obs_waves[missing])]}. ")

            if len(overlap) < 2: #done (0 or 1) if only 1 line, can't go any farther with comparision
                #for FWHM nudge to AGN, the one line MUST at least be on the list of AGN lines
                if (score > 0) and (len(overlap) > 0) and (central_fwhm - central_fwhm_err > 12.0) and \
                    ((self.spectrum_slope + self.spectrum_slope_err) < 0.02 )      and \
                    ((self.spectrum_slope - self.spectrum_slope_err) > -0.02 ):
                    #the slope is just a guess ... trying to separate out most stars
                    #self.add_classification_label("AGN")
                    try:
                        if (np.mean([line_fwhm[i] for i in line_idx]) > 12.0) and\
                                len(np.intersect1d(overlap,np.array([G.LyA_rest,G.CIV_1549,G.CIII_1909,G.MgII_2799])) > 0):
                            #allowed singles: LyA, MgII, CIV, CIII (and really not CIV by itself in our range)
                            return 0.25  # still give a little boost to AGN classification?
                    except:
                        pass

                    return 0
                else:
                    return 0


            #compare all pairs of lines
            #
            # REMINDER: line_idx[i] indexes based on the overlap (should see only with line_xxx[] lists)
            #           rest_idx[i] indexes based on the fixed list of rest_wavelengths (maps overlap to rest)
            for i in range(len(overlap)):
                # add 0.5 for each of LyA, OVI, CIV, CIII in list if they are broad (> 1200 km/s; 18AA at 4500 is roughly match)
                if overlap[i] in [G.LyA_rest, G.OVI_1035, G.NV_1241,G.CIV_1549, G.HeII_1640, G.CIII_1909, G.CII_2326]:
                    if line_fwhm[i] > 18.0 and line_broad[i] and line_raw_line_score[i] > 50.0:
                        score += 0.5

                for j in range(i+1,len(overlap)):
                    if (line_flux[line_idx[i]] != 0):
                        if (min_ratios[rest_idx[i]] != 0) and (min_ratios[rest_idx[j]] != 0) and \
                           (max_ratios[rest_idx[i]] != 0) and (max_ratios[rest_idx[j]] != 0):

                            ratio = line_flux[line_idx[j]] / line_flux[line_idx[i]]
                            try:
                                ratio_err = abs(ratio) * np.sqrt((line_flux_err[line_idx[j]] /line_flux[line_idx[j]]) ** 2 +
                                                             (line_flux_err[line_idx[i]] / line_flux[line_idx[i]]) ** 2)
                            except:
                                ratio_err = 0 #line_flux for j could be zero, so this will fail

                            #try the matrices first (if they are zero, they are not populated yet
                            # so fall back to the list)
                            min_ratio = min_ratio_matrix[rest_idx[j]][rest_idx[i]]
                            max_ratio = max_ratio_matrix[rest_idx[j]][rest_idx[i]]

                            if (min_ratio is None) or (max_ratio is None):
                                min_ratio = min_ratios[rest_idx[j]]/min_ratios[rest_idx[i]]
                                max_ratio = max_ratios[rest_idx[j]] / max_ratios[rest_idx[i]]

                            if min_ratio > max_ratio: #order is backward, so flip
                                min_ratio, max_ratio = max_ratio, min_ratio

                            if min_ratio <= (ratio+ratio_err) and (ratio-ratio_err) <= max_ratio:
                                #now check fwhm is compatible
                                #todo: consider using the fwhm error (is the difference consistent with zero? or
                                # maybe is the ratio consistent with less than 50% difference? or
                                # maybe if both are greater than 12 or 14AA, just call them equivalent
                                fwhm_i = line_fwhm[line_idx[i]]
                                fwhm_j = line_fwhm[line_idx[j]]
                                fwhm_err_i = line_fwhm_err[line_idx[i]]
                                fwhm_err_j = line_fwhm_err[line_idx[j]]
                                avg_fwhm = 0.5* (fwhm_i + fwhm_j)
                                diff_fwhm = max(0,abs(fwhm_i - fwhm_j) - (fwhm_err_i+fwhm_err_j))

                                if (line_broad[line_idx[i]] == line_broad[line_idx[j]]):
                                    adjust = 1.0  # they should be similar (both broad or narrow)
                                elif (line_broad[line_idx[j]]):
                                    adjust = 2.0  #
                                elif (line_broad[line_idx[i]]):
                                    adjust = 0.5  #

                                if avg_fwhm > 0 and adjust*diff_fwhm/avg_fwhm < 0.5:
                                    #full 1pt for rank 1,2 lines, 2/3 for rank 3, 0.5 for rank 4 ...
                                    #want the max line rank (the lower "reliable" or "quality" line)
                                    bump = min(1, 2.0 / max(line_rank[i],line_rank[j]))
                                    score += bump
                                    log.debug(f"Ratio match (+{bump:0.2f}) for solution = {solution.central_rest}: "
                                              f"rest {overlap[j]} to {overlap[i]}: "
                                              f"{min_ratio:0.2f} < {ratio:0.2f} +/- {ratio_err:0.2f} < {max_ratio:0.2f} "
                                              f"FWHM {fwhm_j}, {fwhm_i}")
                                else:
                                    log.debug(f"FWHM no match (0) for solution = {solution.central_rest}: "
                                              f"rest {overlap[j]} to {overlap[i]}: FWHM {fwhm_j}, {fwhm_i}: "
                                              f"ratios: {min_ratio:0.2f} < {ratio:0.2f} +/- {ratio_err:0.2f} < {max_ratio:0.2f}")

                            else:
                                if ratio < min_ratio:
                                    frac = (min_ratio-ratio)/min_ratio
                                else:
                                    frac = (ratio - max_ratio)/max_ratio

                                if frac > 0.5:
                                    score -= 1
                                    log.debug(f"Ratio mismatch (-1) for solution = {solution.central_rest}: "
                                              f"rest {overlap[j]} to {overlap[i]}: "
                                              f"{min_ratio:0.2f} !< {ratio:0.2f} +/- {ratio_err:0.2f} !< {max_ratio:0.2f} ")


            #todo: sther stuff
            # like spectral slope?
            #if score > 0:
            #    self.add_classification_label("AGN")

            return score

        except:
            log.info("Exception in Spectrum::solution_consistent_with_agn",exc_info=True)
            return 0


    def solution_consistent_with_lae(self,solution,central_eli=None,continuum=None,continuum_err=None):
        """
       Similar to consisten with AGN, but really only focus on a solution that includes LyA as a line, even if it is
       not the main line. In that case, in addition to the line compatibility, the EW of the LyA line must be > 20AA

       As such you MUST have the continuum estimate to use (should ideally be the aggregate at the end)

        if there is (positive) consistency (lines match and ratios match) you get a boost
        if there is no consistency (that is, the lines don't match up) you get no change
        if there is anti-consistency (the lines match up but are inconsistent by ratio, you can get a score decrease)


        :param solution:
        :return: +1 point for each pair of lines that are consistent (or -1 for ones that are anti-consistent)
        """

        #
        # note: MgII alone (as broad line) not necessarily AGN ... can be from outflows?
        #

        #check the lines, are they consistent with AGN?
        try: #todo: for MgII, OII can also be present
            rest_waves = np.array([G.LyA_rest,G.CIV_1549,G.CIII_1909,G.CII_2326,G.MgII_2799,G.NV_1241,G.SiII_1260,G.SiIV_1400,G.HeII_1640,G.OVI_1035, G.OII_rest])
            #aka                     LyA,      CIV, CIII, CII,   MgII,  NV,  SiII, SiIV, HeII, OVI,  OII
            obs_waves = rest_waves * (1. + solution.z)

            # compared to LyA EW: so line ew/ LyA EW; a value of 0 means no info
            #todo: need info/citation on this
            #todo: might have to make a matrix if can't reliably compare to LyA
            #todo:   e.g. if can only say like NV < MgII or CIV > CIII, etc

            # using as a very rough guide: https://ned.ipac.caltech.edu/level5/Netzer/Netzer2_1.html
            # and manual HETDEX spectra
            #            #LyA, CIV,  CIII, CII,  MgII,  NV,     SiII, SiIV,  HeII,  OVI   OII
            min_ratios = [1.0, 0.07, 0.02, 0.01,  0.05, 0.05,  0.00, 0.03,   0.01,  0.03, 0.01]
            max_ratios = [1.0, 0.70, 0.30, 0.10,  0.40, 0.40,  0.00, 0.20,   0.20,  0.40, 9.99]
            #            *** apparently NV can be huge .. bigger than CIV even see 2101164104
            #            *** OII entries just to pass logic below (only appears on our range for some MgII)


            #required match matrix ... if line at row (x) is found and line at column (y) is in range, it MUST be found too
            #this is in order of the lines in rest_waves
            #in ALL cases, LyA better be found IF it is in range (so making it a 2 ... need 2 other matched lines to overcome missing LyA)
            match_matrix =[[1,0,0,0,0,0,0,0,0,0,0],  #0 LyA
                           [1,1,1,1,0,0,0,0,0,0,0],  #1 CIV
                           [1,0,1,0,0,0,0,0,0,0,0],  #2 CIII
                           [1,0,0,1,0,0,0,0,0,0,0],  #3 CII
                           [1,0,0,0,1,0,0,0,0,0,1],  #4 MgII
                           [1,0,0,0,0,1,0,0,0,0,0],  #5 NV
                           [1,1,1,0,0,1,1,0,0,0,0],  #6 SiII
                           [1,1,1,0,0,0,1,1,0,0,0],  #7 SiIV
                           [1,0,0,0,0,0,0,0,1,0,0],  #8 HeII
                           [1,0,0,0,0,0,0,0,0,1,0],  #9 OVI
                           [0,0,0,0,1,0,0,0,0,0,1] ] #10 OII (just with MgII)
                         #  0 1 2 3 4 5 6 7 8 9 10



            match_matrix = np.array(match_matrix)
                                          #   0   1  2   3   4   5   6    7  8   9   10
            match_matrix_weights = np.array([2.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])

            #todo: 2 matrices (min and max ratios) so can put each line vs other line
            # like the match_matrix in the low-z galaxy check (but with floats) as row/column
            # INCOMPLETE ... not in USE YET

            #row/column (is mininum, where lines are smallest compared to LyA)
            # the inverse is still the minimum just the inverted ratio)
            min_ratio_matrix = \
                [ [1.00, None, None, None, None, None, None, None, None, None, None],  #LyA
                  [None, 1.00, None, None, None, None, 4.00, None, 6.66, None, None],  #CIV
                  [None, None, 1.00, None, None, None, None, None, None, None, None],  #CIII
                  [None, None, None, 1.00, None, None, None, None, None, None, None],  #CII
                  [None, None, None, None, 1.00, None, None, None, None, None, 20.0],  #MgII
                  [None, None, None, None, None, 1.00, None, None, None, None, None],  #NV
                  [None, 0.25, None, None, None, None, 1.00, None, None, None, None],  #SiII
                  [None, None, None, None, None, None, None, 1.00, None, None, None],  #SiIV
                  [0.01, 0.15, None, None, None, None, None, None, 1.00, None, None],  #HeII
                  [None, None, None, None, None, None, None, None, None, 1.00, None],  #OVI
                  [None, None, None, None, 0.05, None, None, None, None, None, 1.00 ]] #OII
                 # LyA   CIV   CIII  CII   MgII   NV   SiII  SiVI  HeII   OVI  OII

            #row/column (is maximum ... where lines are the largest compared to LyA)
            max_ratio_matrix = \
                [ [1.00, None, None, None, None, None, None, None, None, None, None],  #LyA
                  [None, 1.00, None, None, None, None, 0.10, None, 1.43, None, None],  #CIV
                  [None, None, 1.00, None, None, None, None, None, None, None, None],  #CIII
                  [None, None, None, 1.00, None, None, None, None, None, None, None],  #CII
                  [None, None, None, None, 1.00, None, None, None, None, None, 2.00],  #MgII
                  [None, None, None, None, None, 1.00, None, None, None, None, None],  #NV
                  [None, 10.0, None, None, None, None, 1.00, None, None, None, None],  #SiII
                  [None, None, None, None, None, None, None, 1.00, None, None, None],  #SiIV
                  [0.50, 0.70, None, None, None, None, None, None, 1.00, None, None],  #HeII
                  [None, None, None, None, None, None, None, None, None, 1.00, None],  #OVI
                  [None, None, None, None, 0.50, None, None, None, None, None, 1.00] ] #OII
                 # LyA    CIV  CIII   CII  MgII   NV   SiII  SiVI  HeII   OVI  OII

            sel = np.where(np.array([l.absorber for l in solution.lines])==False)[0]
            sol_lines = np.array(solution.lines)[sel]
            line_waves = [solution.central_rest] + [l.w_rest for l in sol_lines]
            line_obs_waves = [solution.central_rest*(1+solution.z)] + [l.w_obs for l in sol_lines]
            line_eli = [None] + [l.eli for l in sol_lines]
            #line_ew = [self.eqw_obs/(1+solution.z)] + [l.eqw_rest for l in sol_lines]
            # line_flux is maybe more reliable ... the continuum estimates for line_ew can go wrong and give horrible results
            line_flux = [self.estflux] + [l.flux for l in sol_lines]
            line_flux_err = [self.estflux_unc] + [l.flux_err for l in sol_lines]
            line_fwhm = [self.fwhm] + [l.sigma * 2.355 for l in sol_lines]
            line_fwhm_err = [self.fwhm_unc] + [l.sigma_err * 2.355 for l in sol_lines]
            line_broad = [solution.emission_line.broad] + [l.broad for l in sol_lines] #can they be broad
            #line_rank = [solution.emission_line.rank] + [l.rank for l in sol_lines]

            overlap, rest_idx, line_idx = np.intersect1d(rest_waves,line_waves,return_indices=True)

            central_fwhm =  self.fwhm
            central_fwhm_err = self.fwhm_unc
            #todo: get samples and see if there is a correlation with slope
            #slope = self.spectrum_slope
            #slope_err = self.spectrum_slope_err

            score = 0

            #is LyA in the list of lines?? if so, check its EW
            sel_lya = [np.isclose(line_waves, G.LyA_rest,atol=2.0)]
            if np.sum(sel_lya) == 1: #LyA is in the list
                #get the lya_idx specifically
                lya_idx = list(sel_lya[0]).index(True)

                #and scale by SNR
                if hasattr(line_eli[lya_idx],"snr") and line_eli[lya_idx].snr > 0:
                    snr = line_eli[lya_idx].snr
                elif lya_idx == 0 and central_eli is not None:# it is the main line
                    snr = central_eli.snr
                else:
                    log.info("In solution_consistent_with_lae(). No SNR info. Cannot continue.")
                    return 0

                #if the snr of LyA is less than SNR of main line, then really don't trust if
                if central_eli is None:
                    score = max(0.0,score)
                elif lya_idx > 0: #basically, this LyA non-central line has to be SNR > 6.0 or at least 90% of the other line
                    #unless it is at the very extreme blue
                    if ( (line_obs_waves[lya_idx] <= 3650) and (snr < 3.5)) or \
                        ((line_obs_waves[lya_idx] > 3650) and (snr < 6.0 and ((snr / central_eli.snr) < 0.9 ))):
                    #if snr < 6.0 and ((snr / central_eli.snr) < 0.9 ):
                        # if snr < 5.0:
                        #     score = -2.0
                        # elif snr < 5.5:
                        #     score = -1.5
                        # else:
                        #     score = -1.0
                        score = -1.0
                        log.info(f"In solution_consistent_with_lae(), SNR of non-central LyA too low: {line_eli[lya_idx].snr}")
                        #return score


                    #check the flux ratios vs the central line and non-central LyA
                    try:
                        central_rest_idx = np.where(overlap==line_waves[0])[0][0]
                        lya_rest_idx = np.where(overlap == line_waves[lya_idx])[0][0]
                        min_ratio = min_ratio_matrix[rest_idx[central_rest_idx]][rest_idx[lya_rest_idx]]
                        max_ratio = max_ratio_matrix[rest_idx[central_rest_idx]][rest_idx[lya_rest_idx]]

                        if (min_ratio is not None) and (max_ratio is not None):
                            if min_ratio > max_ratio:  # order is backward, so flip
                                min_ratio, max_ratio = max_ratio, min_ratio

                            flux_ratio = line_flux[0] / line_flux[lya_idx]
                            flux_ratio_err = flux_ratio * np.sqrt((line_flux_err[0]/line_flux[0])**2 +
                                                                  (line_flux_err[lya_idx]/line_flux[lya_idx])**2 )

                            if (flux_ratio + flux_ratio_err < min_ratio)  or  (flux_ratio - flux_ratio_err > max_ratio):
                            #this is inconsistent with assumed ratios
                                score -= 1.0 #subtract one

                                log.info(f"In solution_consistent_with_lae(), incompatible flux ratio: "
                                         f"{flux_ratio:0.3f}+/-{flux_ratio_err:0.4f} vs [{min_ratio:0.3f},{max_ratio:0.3f}]")

                    except:
                        pass

                    return score

                else: #main line is LyA ... continue on
                    pass


                #todo: IF the LyA line has continuum above the DEX limit, then use that continuum even if
                #todo: an aggregate continuum was passed in?

                #check the continuum
                if continuum is None or continuum <= 0:
                    try:
                        if hasattr(line_eli[lya_idx],"cont") and line_eli[lya_idx].cont is not None and line_eli[lya_idx].cont > 0:
                                continuum = line_eli[lya_idx].cont
                                continuum_err = line_eli[lya_idx].cont_err
                        elif lya_idx == 0:
                            continuum = central_eli.cont
                            continuum_err = central_eli.cont_err
                        else:
                            log.info(f"In solution_consistent_with_lae(): Invalid continuum. Cannot continue")
                            return 0
                    except:
                        log.info(f"In solution_consistent_with_lae(): Invalid continuum. Cannot continue")
                        return 0

                ew = line_flux[lya_idx] / continuum
                ew_err = 0
                try:
                    if continuum_err is not None and continuum_err > 0:
                        ew_err = ew * np.sqrt( (line_flux_err[lya_idx]/line_flux[lya_idx])**2 + (continuum_err/continuum)**2)
                except:
                    ew_err = 0

                #have to convert to LyA rest EW
                ew = ew / (solution.z + 1)
                ew_err = ew_err / (solution.z + 1)
                log.info(f"In solution_consitent_with_lae(): EW = {ew:0.2f} +/- {ew_err:0.3f}, SNR = {snr}")
                if (ew > 20.0) or (((ew+ew_err) > 20.0) and ((ew-ew_err) > 15.0)):
                    if (ew-ew_err) > 40:
                        score = 1.0 * snr/8.0
                    elif (ew-ew_err) > 25.0:
                        score = 0.5 * snr/8.0
                    else:
                        score = 0.25 * snr/8.0
                #this could be an AGN and then the continuum measure from the band pass can be way too high, and then
                #the EW too low, so give it some room, if the continuum is a little high and the line is at least a little broad
                elif (ew > 15.0) or (ew+ew_err) > 20.0 or (self.fwhm is not None and self.fwhm > 16.0):
                    if  (line_flux[lya_idx] > 5e-17) and \
                        (lya_idx == np.argmax(line_flux)) and \
                        (SU.cgs2mag(continuum,line_obs_waves[lya_idx]) < 24.0) and \
                        (line_fwhm[lya_idx]/line_obs_waves[lya_idx]*3e5 > 600.0):
                        score = 0.25 * snr/8.0
                    else:
                        score = 0 #probably not an LAE, by definition, but might be an LBG or could be something else, but let's no penalize

               # elif ew > 15.0 and ew_err < 5.0:
               #     score = 0 #probably not an LAE, by definition, but might be an LBG or could be something else, but let's no penalize
                else:
                    score = -1.0

                if snr < 5.5:
                    score = min(score,0.0) #don't trust it, but I want to log record what we have


                # p_lae_oii_ratio, p_lae, p_oii, plae_errors = line_prob.mc_prob_LAE(wl_obs=G.LyA_rest*(1+solution.z),
                #                                                                   lineFlux=line_flux[lya_idx],
                #                                                                   lineFlux_err=line_flux_err[lya_idx],
                #                                                                   continuum=continuum,
                #                                                                   continuum_err=continuum_err,
                #                                                                   c_obs=None, which_color=None,
                #                                                                   addl_wavelengths=[],
                #                                                                   addl_fluxes=[],
                #                                                                   addl_errors=[],
                #                                                                   sky_area=None,
                #                                                                   cosmo=None, lae_priors=None,
                #                                                                   ew_case=None, W_0=None,
                #                                                                   z_OII=None, sigma=None)
                #
                # try:
                #     if plae_errors:
                #         p_lae_oii_ratio_range = plae_errors['ratio']
                # except:
                #     pass

                #last sanity check ... however, this CAN be an AGN and that throws this off, esp since this has another line
                if self.fwhm < 16.0 and score < 0.5 and (SU.cgs2mag(continuum-continuum_err,G.LyA_rest*(1+solution.z)) < (G.LAE_G_MAG_ZERO-0.5)):
                    score = min(-1.0,score) #very inconsistent with LyA, so at least -1.0 or lower

                return score
            else: #LyA is NOT found in the list of lines, but this is supposedly in the 1.88 < z < 3.52 range
                #now, this *could* be an LBG or somehow LyA is very weak, but that is very unlikely for HETDEX
                #and HETDEX wants LAEs, so mark this as unlikely

                lya_missing = True
                #at the blue end in particular, the LyA line can be very hard to find, esp if broad, so
                #we need to make some allowances here
                #if our anchor line FWHM is large and the LyA should be blue of 3700, check to see if there is
                #big flux where lya should be, +/- anchor line's 2*sigma (just 2 sigma since can be ragged)
                #e.g. if it is similar to anchor line flux, then assume we just failed to fit it
                try:
                    if central_fwhm > 14.0:# and solution.z < 2.05: #z < 2.05 is about wave < 3700AA
                        center,*_ = SU.getnearpos(self.wavelengths,G.LyA_rest*(1+solution.z))
                        left = max(0,center-int(central_fwhm/2.355*2.0))
                        right = center+int(central_fwhm/2.355*2.0) #due to early checks, cannot be out of range
                        lya_int_flux = np.sum(self.values[left:right])
                        lya_int_flux_err = np.sqrt(np.sum(self.errors[left:right]**2))

                        #the outer 2sigma wings (or excluding the interior +/- 1 sigma)
                        l_left = max(0,center-int(central_fwhm/2.355*2.0))
                        l_right = max(0,center-int(central_fwhm/2.355*1.0))
                        r_left = center+int(central_fwhm/2.355*1.0) #due to early checks, cannot be out of range
                        r_right = center+int(central_fwhm/2.355*2.0) #due to early checks, cannot be out of range
                        lya_wing_flux =np.sum(self.values[l_left:l_right]) + np.sum(self.values[r_left:r_right])

                        #compare to anchor flux
                        center,*_ = SU.getnearpos(self.wavelengths,self.central)
                        left = max(0,center-int(central_fwhm/2.355*2.0))
                        right = center+int(central_fwhm/2.355*2.0) #due to early checks, cannot be out of range
                        anchor_int_flux = np.sum(self.values[left:right])
                        anchor_int_flux_err = np.sqrt(np.sum(self.errors[left:right]**2))

                        #LyA (best case) has to be larger than the non-LyA line
                        if ((lya_int_flux + lya_int_flux_err) > (anchor_int_flux-anchor_int_flux_err)) and \
                            ((lya_int_flux_err/lya_int_flux) < 0.5) and ((anchor_int_flux_err/anchor_int_flux) < 0.5) and \
                                (lya_wing_flux/lya_int_flux < 0.33): #should be < 0.28, but leaving slop for error
                            lya_missing = False #no boost, but no penalty either in this case

                except:
                    log.warning("Exception in solution_consistent_wtih_lae attemping to check non fitted flux @ LyA.",exc_info=True)


                missing = []
                in_range = np.where((obs_waves > G.HETDEX_BLUE_SAFE_WAVE) & (obs_waves < 5500.))[0]
                for i in range(len(overlap)):
                    if np.sum(match_matrix[rest_idx[i]]) > 1:
                        # at least one other line must be found (IF the obs_wave is in the HETDEX range)
                        sel = np.intersect1d(in_range, np.where(match_matrix[rest_idx[i]])[0])
                        missing = np.union1d(missing, np.setdiff1d(sel, rest_idx)).astype(int)

                if 0 in missing and lya_missing is False: #LyA is idx 0
                    missing = missing[missing != 0]
                # score = -1 * len(missing)
                score = -1 * np.sum(match_matrix_weights[missing])

                if score == 0 and not lya_missing:
                    #we want a small boost since nothing else was missing and we want to force
                    #a check against AGN
                    score = 0.1

                if score > 0:
                    log.info(f"In solution_consitent_with_lae():  Possible LyA. Score = {score}")
                elif score < 0:
                    log.info(f"In solution_consitent_with_lae():  Missing {len(missing)} required lines. Score = {score}")

                #if score == 0, don't bother logging

                #todo: right now only checking the presence of the line, not the ratios

                return score
        except:
            log.info("Exception in Spectrum::solution_consistent_with_lae",exc_info=True)
            return 0
        #end consistent with LAE

    def top_hat_filter(self,w_rest,w_obs, wx, hat_width=None, negative=False):
        #optimal seems to be around 1 to < 2 resolutions (e.g. HETDEX ~ 6AA) ... 6 is good, 12 is a bit
        #unstable ... or as rougly 3x pixel pix_size


        #build up an array with tophat filters at emission line positions
        #based on the rest and observed (shifted and streched based on the redshift)
        # wx is the array of wavelengths (e.g the x coords)
        # hat width in angstroms
        try:
            w_rest = float(w_rest)
            w_obs = float(w_obs)
            num_hats = 0

            if negative:
                filter = np.full(np.shape(wx), -1.0)
            else:
                filter = np.zeros(np.shape(wx))

            pix_size = float(wx[1]-wx[0]) #assume to be evenly spaced


            if hat_width is None:
                hat_width = 3.0*pix_size

            half_hat = int(np.ceil(hat_width/pix_size)/2.0) #hat is split evenly on either side of center pix
            z = w_obs/w_rest -1.0

            #for each line in self.emission_lines that is in range, add a top_hat filter to filter
            for e in self.emission_lines:
                w = e.redshift(z)

                #set center pixel and half-hat on either side to 1.0
                if (w > wx[0]) and (w < wx[-1]):
                    num_hats += 1
                    idx = getnearpos(wx,w)
                    filter[idx-half_hat:idx+half_hat+1] = 1.0
        except:
            log.warning("Unable to build top hat filter.", exc_info=True)
            return None

        return filter, num_hats


    def set_spectra(self,wavelengths, values, errors, central, values_units = 0, estflux=None, estflux_unc=None,
                    eqw_obs=None, eqw_obs_unc=None, fit_min_sigma=None,estcont=None,estcont_unc=None,
                    fwhm=None,fwhm_unc=None,continuum_g=None,continuum_g_unc=None,reset=False,gmag=None, gmag_err=None,
                    detobj=None):
        self.wavelengths = []
        self.values = []
        self.errors = []

        if fit_min_sigma is None: #GAUSS_FIT_MIN_SIGMA can change after construction, so cannot set as a parameter default
            fit_min_sigma = GAUSS_FIT_MIN_SIGMA

        #these could have already been done if we searched for a central wavelength
        if reset:
            self.all_found_lines = None
            self.all_found_absorbs = None

        self.solutions = []
        #self.central_eli = None

        if self.noise_estimate is None or len(self.noise_estimate) == 0:
            self.noise_estimate = errors[:]
            self.noise_estimate_wave = wavelengths[:]

        if central is None:
            self.wavelengths = wavelengths
            self.values = values
            self.errors = errors
            self.values_units = values_units
            self.central = central
            return

        if fwhm is not None:
            self.fwhm = fwhm
            if fwhm_unc is not None:
                self.fwhm_unc = fwhm_unc #might be None
            else:
                self.fwhm_unc = 0

        if continuum_g is not None:
            self.estcont = continuum_g
            self.estcont_unc = continuum_g_unc

        if gmag is not None:
            self.gmag = gmag
            self.gmag_unc = gmag_err

        #scan for lines
        try:
            if self.all_found_lines is None:
                self.all_found_lines = peakdet(wavelengths, values, errors, values_units=values_units, enforce_good=True,
                                               spec_obj=self)
        except:
            log.warning("Exception in spectum::set_spectra()",exc_info=True)

        try:
            if self.all_found_absorbs is None and continuum_g is not None and continuum_g > G.CONTINUUM_THRESHOLD_FOR_ABSORPTION_CHECK:
                self.all_found_absorbs = peakdet(wavelengths, values, errors, values_units=values_units,
                                                 enforce_good=True,absorber=True,spec_obj=self)
        except:
            log.warning("Exception in spectum::set_spectra()",exc_info=True)

        if self.all_found_lines is not None and len(self.all_found_lines) > 1:
            self.all_found_lines, removed_central = filter_emission_line_as_continuum(self.all_found_lines, self.all_found_absorbs, self.central)

            if removed_central and detobj is not None:
                detobj.flags |= G.DETFLAG_FOLLOWUP_NEEDED
                detobj.flags |= G.DETFLAG_BAD_EMISSION_LINE
                detobj.flags |= G.DETFLAG_UNCERTAIN_CLASSIFICATION

        #run MCMC on this one ... the main line
        try:

            if self.identifier is None and self.plot_dir is None:
                show_plot = False #intermediate call, not the final
            else:
                show_plot = G.DEBUG_SHOW_GAUSS_PLOTS

            try:
                allow_broad = (self.fwhm + self.fwhm_unc) > (GOOD_BROADLINE_SIGMA * 2.355)
            except:
                allow_broad = False

            if self.central_eli is None: #could already be set (e.g. if H&K line fit for continuum detection)
                eli = signal_score(wavelengths=wavelengths, values=values, errors=errors,central=central,spectrum=self,
                                   values_units=values_units, sbr=None, min_sigma=fit_min_sigma,
                                   show_plot=show_plot,plot_id=self.identifier,
                                   plot_path=self.plot_dir,do_mcmc=True,allow_broad=allow_broad,absorber=self.absorber,
                                   spec_obj=self,targetted_fit=True)
            else:
                log.info("Skipping central line signal_score() as already computed.")
                eli = self.central_eli

        except:
            log.error("Exception in spectrum::set_spectra calling signal_score().",exc_info=True)
            eli = None

        if eli:
            if (estflux is None) or (eqw_obs is None) or (estflux == -1) or (eqw_obs <= 0.0):
                #basically ... if I did not get this from Karl, use my own measure
                if (eli.mcmc_a is not None) and (eli.mcmc_y is not None):
                    a_unc = 0.5 * (abs(eli.mcmc_a[1]) + abs(eli.mcmc_a[2])) / eli.mcmc_dx
                    y_unc = 0.5 * (abs(eli.mcmc_y[1]) + abs(eli.mcmc_y[2])) / eli.mcmc_dx

                    estflux = eli.mcmc_line_flux
                    estflux_unc = a_unc

                    estcont = eli.mcmc_continuum
                    estcont_unc = y_unc

                    eqw_obs = abs(estflux / eli.mcmc_continuum)
                    eqw_obs_unc = abs(eqw_obs) * np.sqrt((a_unc / estflux) ** 2 + (y_unc / eli.mcmc_continuum) ** 2)
                else: #not from mcmc, so we have no error
                    estflux = eli.line_flux
                    estflux_unc = 0.0
                    eqw_obs = eli.eqw_obs
                    eqw_obs_unc = 0.0

            if (self.fwhm is None) or (self.fwhm  <= 0):
                if eli.mcmc_sigma:
                    try:
                        self.fwhm = eli.mcmc_sigma[0]*2.355
                        self.fwhm_unc = 0.5*(eli.mcmc_sigma[1]+eli.mcmc_sigma[2])*2.355
                    except:
                        pass

            if (self.fwhm is None) or (self.fwhm  <= 0):
                try:
                    self.fwhm = eli.fit_sigma*2.355
                    self.fwhm_unc = eli.fit_sigma_err*2.355 #right now, not actually calc the uncertainty
                except:
                    pass

            #if (self.snr is None) or (self.snr == 0):
            #    self.snr = eli.snr

            if self.central_eli != eli:
                self.central_eli = copy.deepcopy(eli)

        else:
            log.warning("Warning! Did not successfully compute signal_score on main emission line.")

        self.wavelengths = wavelengths
        self.values = values
        self.errors = errors
        self.values_units = values_units
        self.central = central
        #the rest are below

        try:
            # get very basic info (line fit)
            self.gband_masked_correction()

            #also get the overall slope
            lines_to_mask = self.all_found_lines[:] if self.all_found_lines != None else []
            lines_to_mask += self.all_found_absorbs[:] if self.all_found_absorbs != None else []

            #in flam not flux
            self.spectrum_linear_coeff, self.spectrum_linear_coeff_err = SU.simple_fit_line(wavelengths,
                                                                                            values/G.FLUX_WAVEBIN_WIDTH,
                                                                                             errors, trim=True,
                                                                                             lines=lines_to_mask)

            #e.g. flam from the line fit to the spectrum at the emission line center
            self.spectrum_linear_continuum,self.spectrum_linear_continuum_err = SU.eval_line_at_point(central,
                                                                                                     self.spectrum_linear_coeff,
                                                                                                     self.spectrum_linear_coeff_err)


            self.spectrum_linear_continuum *= np.power(10.,values_units)
            self.spectrum_linear_continuum_err *= np.power(10.,values_units)

            #would we want these would be in f_nu units ?
            # fnu_values = SU.cgs2ujy(values/G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS, wavelengths) #* 1e-29
            # fnu_errors = SU.cgs2ujy(errors/G.FLUX_WAVEBIN_WIDTH * G.HETDEX_FLUX_BASE_CGS, wavelengths) #* 1e-29
            # fnu_linear_coeff, fnu_linear_coeff_err = fit_line(wavelengths,fnu_values,fnu_errors,
            #                                                                       True,self.all_found_lines,
            #                                                                       self.all_found_absorbs)
            # self.spectrum_slope = fnu_linear_coeff[1] #* np.power(10.,values_units)
            # self.spectrum_slope_err = fnu_linear_coeff_err[1]# * np.power(10.,values_units)

            #or just leave in erg/s/cm2 x2AA
            self.spectrum_slope = self.spectrum_linear_coeff[1] * np.power(10.,values_units)
            self.spectrum_slope_err = self.spectrum_linear_coeff_err[1] * np.power(10.,values_units)

            #this one does take as a line in integrated flux (f_lam) units, but computes B-V in f_nu
            self.est_obs_b_minus_v, self.est_obs_b_minus_v_err = SU.est_linear_B_minus_V(self.spectrum_linear_coeff,
                                                                                      self.spectrum_linear_coeff_err)

            log.info(f"{self.identifier}: Spectrum basic info: slope = {self.spectrum_linear_coeff[1]:0.3g} +/- "
                     f"{self.spectrum_linear_coeff_err[1]:0.4g} erg/s/cm2/AA e-17")

            log.info(f"{self.identifier}: Spectrum basic info: intercept = {self.spectrum_linear_coeff[0]:0.3g} +/- "
                     f"{self.spectrum_linear_coeff_err[0]:0.4g} erg/s/cm2/AA e-17")

            log.info(f"{self.identifier}: Spectrum basic info: linear continuum @ {central:0.2f} = {self.spectrum_linear_continuum:0.3g} +/- "
                     f"{self.spectrum_linear_continuum_err:0.3g} erg/s/cm2/AA")

            log.info(f"{self.identifier}: Spectrum basic info: linear B-V (observed) = {self.est_obs_b_minus_v:0.3g} +/- "
                     f"{self.est_obs_b_minus_v_err:0.3g}")


            # #also get the overall slope
            # lines_to_mask = self.all_found_lines[:] if self.all_found_lines != None else []
            # lines_to_mask += self.all_found_absorbs[:] if self.all_found_absorbs != None else []
            # try:
            #     self.spectrum_slope, self.spectrum_slope_err = SU.simple_fit_slope(wavelengths, values, errors,lines=lines_to_mask)
            #     log.info("%s Spectrum basic slope: %g +/- %g" %(self.identifier,self.spectrum_slope,self.spectrum_slope_err))
            # except:
            #     pass
            #if self.snr is None:
            #    self.snr = 0

            # # also get the overall slope
            # self.spectrum_slope, self.spectrum_slope_err = SU.simple_fit_slope(wavelengths, values, errors)
            #
            # log.info("%s Spectrum basic slope: %g +/- %g"
            #          %(self.identifier,self.spectrum_slope,self.spectrum_slope_err))
            #todo: maybe also a basic parabola? (if we capture an overall peak? like for a star black body peak?
        except:
            pass


        # self.wavelengths = wavelengths
        # self.values = values
        # self.errors = errors
        # self.values_units = values_units
        # self.central = central
        self.estflux = estflux
        self.estflux_unc = estflux_unc
        self.eqw_obs = eqw_obs
        self.eqw_obs_unc = eqw_obs_unc
        self.estcont = estcont
        self.estcont_unc = estcont_unc





    def find_central_wavelength(self,wavelengths = None,values = None, errors=None, values_units=0, return_list=False):
        """

        :param wavelengths:
        :param values:
        :param errors:
        :param values_units:
        :return:
        """
        #
        # def best_index(lines):
        #     """
        #     return the index of the best lines by SNR, Chi2, line_score
        #     :param lines:
        #     :return:
        #     """
        #     idx = -1
        #     try:
        #         if lines is None or len(lines) == 0:
        #             return -1
        #         elif len(lines)==1:
        #             return 0
        #
        #         sel = np.full(len(lines),True)
        #         #first, SNR ....
        #
        #     except:
        #         pass
        #
        #     return idx

        central = 0.0
        update_self = False
        absorber = False
        if (wavelengths is None) or (values is None):
            wavelengths = self.wavelengths
            values = self.values
            values_units = self.values_units
            update_self = True

        if self.central is None:
            update_self = True

        found_absorbers = []
        found_lines = []

        if self.all_found_lines is None or len(self.all_found_lines) == 0:
            try:
                found_lines = peakdet(wavelengths,values,errors,values_units=values_units,spec_obj=self)
            except:
                found_lines = []

            self.all_found_lines = found_lines

            try:

                est_g_cont = self.est_g_cont #just a very basic check on whether we are bright enough to trigger continuum rules
                if est_g_cont is None or est_g_cont == 0:
                    if values_units != 0:
                        est_g_cont = np.median(values)*10**values_units
                    else:
                        est_g_cont = np.median(values)

                if (G.CONTINUUM_RULES or ((est_g_cont is not None) and (est_g_cont > G.CONTNIUUM_RULES_THRESH))) and \
                        (self.all_found_absorbs is None or len(self.all_found_absorbs) == 0):
                    try:
                        #this is a dumb way to do this ... need to go refactor and change the CONTINUUM_RULES from a global
                        #condition to just a per-detection condition
                        orig_CONTINUUM_RULES = G.CONTINUUM_RULES
                        orig_MAX_SCORE_ABSORPTION_LINES = G.MAX_SCORE_ABSORPTION_LINES
                        orig_DISPLAY_ABSORPTION_LINES = G.DISPLAY_ABSORPTION_LINES

                        G.CONTINUUM_RULES = True
                        G.MAX_SCORE_ABSORPTION_LINES = 9999.9
                        G.DISPLAY_ABSORPTION_LINES = True

                        found_absorbers = peakdet(wavelengths,values,errors,values_units=values_units,absorber=True,spec_obj=self)

                        G.CONTINUUM_RULES = orig_CONTINUUM_RULES
                        G.MAX_SCORE_ABSORPTION_LINES = orig_MAX_SCORE_ABSORPTION_LINES
                        G.DISPLAY_ABSORPTION_LINES = orig_DISPLAY_ABSORPTION_LINES
                    except:
                        found_absorbers = []
                        G.CONTINUUM_RULES = orig_CONTINUUM_RULES
                        G.MAX_SCORE_ABSORPTION_LINES = orig_MAX_SCORE_ABSORPTION_LINES
                        G.DISPLAY_ABSORPTION_LINES = orig_DISPLAY_ABSORPTION_LINES
            except:
                # log.error("Exception! <Traceback>",exc_info=True)
                # log.error(f"{G.CONTINUUM_RULES}, {self.est_g_cont}, {self.all_found_absorbs} ")
                found_absorbers = []

            self.all_found_absorbs = found_absorbers

        if (found_lines and len(found_lines) > 0) or (found_absorbers and len(found_absorbers)>0):

            #check for H&K
            x0 = np.array([k.fit_x0 for k in found_absorbers])
            x0e = np.array([k.fit_x0_err for k in found_absorbers])

            #check pairs of lines ... must be the right separation and similar area, depth, sigma, snr
            #start from 3933 (some slop from 3934) at z=0 and work out

            k_reference = self.emission_lines[np.where([k.name == "(K)CaII" for k in self.emission_lines])[0][0]]
            h_reference = self.emission_lines[np.where([k.name == "(H)CaII" for k in self.emission_lines])[0][0]]

            for k in found_absorbers:
                if (k.fit_x0 + k.fit_x0_err)  > self.h_and_k_waves[0]: #could be k
                    zp1 = k.fit_x0/self.h_and_k_waves[0]
                    zp1e = k.fit_x0_err/self.h_and_k_waves[0]

                    hwave = zp1*self.h_and_k_waves[1]
                    hwave_err = zp1e*self.h_and_k_waves[1]

                    sel1 = (x0-x0e) > (hwave-hwave_err)
                    sel2 = (x0+x0e) < (hwave+hwave_err)
                    sel = sel1 & sel2

                    #there is one pair that fits h & k positions
                    # could be one or more? should only expect one match though
                    for h in np.array(found_absorbers)[sel]:
                        good,hk_mcmc = fit_for_h_and_k(k,h,wavelengths,values,errors,values_units,values_dx=G.FLUX_WAVEBIN_WIDTH)

                        if (good is not None) and good:
                            # !!! WARNING ... must use the hk_mcmc.values_units NOT this functions values_units
                            # as they can be different

                            h.mcmc_to_fit(hk_mcmc,hk_mcmc.values_units,2.0)

                            #update the EmissionLineInfo in h
                            h.pix_size = 2.0
                            h.fit_bin_dx = 2.0
                            h.sn_pix = hk_mcmc.mcmc_snr_pix
                            h.absorber = True
                            h.fit_sigma = hk_mcmc.mcmc_sigma_2[0]
                            h.fit_sigma_err = (hk_mcmc.mcmc_sigma_2[1] + hk_mcmc.mcmc_sigma_2[1])/2.0
                            h.fwhm = h.fit_sigma*2.355
                            h.fit_a = hk_mcmc.mcmc_A_2[0]
                            h.fit_a_err = 0.5 * (hk_mcmc.mcmc_A_2[1]+hk_mcmc.mcmc_A_2[2])
                            h.fit_y = hk_mcmc.mcmc_y[0]
                            h.fit_y_err = 0.5 * (hk_mcmc.mcmc_y[1]+hk_mcmc.mcmc_y[2])

                            #eli.line_flux = hk_mcmc.mcmc_A_2[0]/2.0 * 10**hk_mcmc.values_units
                            #eli.line_flux_err = 0.25*(hk_mcmc.mcmc_A_2[1]+hk_mcmc.mcmc_A_2[2])* 10**hk_mcmc.values_units
                            #NOTE 0.25* since 0.5 * for the average and /2.0 for the 2AA
                            h.fit_x0 = hk_mcmc.mcmc_mu_2[0]
                            h.fit_dx0 = self.h_and_k_waves[1] - hk_mcmc.mcmc_mu_2[0]/(hk_mcmc.mcmc_mu[0]/self.h_and_k_waves[0])
                            #h.w_obs = hk_mcmc.mcmc_mu_2[0] #not using w_obs in EmissionLineInfo object (just in EmissionLine object)
                            h.snr = hk_mcmc.mcmc_snr #technically the SNR for the combined lines, but since
                            #these are fit as two, keep this value and maybe boost the score


                            h.mcmc_x0 = hk_mcmc.mcmc_mu_2
                            h.mcmc_sigma = hk_mcmc.mcmc_sigma_2
                            h.mcmc_snr = hk_mcmc.mcmc_snr
                            h.mcmc_snr_err = hk_mcmc.mcmc_snr_err
                            h.mcmc_a = np.array(hk_mcmc.mcmc_A_2) #yes, a 3-tuple
                            h.mcmc_y = np.array(hk_mcmc.mcmc_y) #this is probably in the e-18 units but as a flux density
                            h.mcmc_dx = h.fit_dx0
                            h.mcmc_line_flux = h.fit_a/h.fit_bin_dx*(10**hk_mcmc.values_units)
                            lineflux_err = 0.5*(h.mcmc_a[1]+h.mcmc_a[2])/h.fit_bin_dx*10**hk_mcmc.values_units

                            h.fit_line_flux = h.mcmc_line_flux
                            h.fit_line_flux_err = lineflux_err
                            h.line_flux = h.mcmc_line_flux
                            h.line_flux_err = lineflux_err

                            h.mcmc_line_flux_tuple = [h.mcmc_line_flux,h.mcmc_line_flux+lineflux_err,h.mcmc_line_flux-lineflux_err]
                            h.mcmc_continuum = hk_mcmc.mcmc_y[0]*10**hk_mcmc.values_units / G.FLUX_WAVEBIN_WIDTH
                            h.mcmc_continuum_tuple = np.array(hk_mcmc.mcmc_y)*10**hk_mcmc.values_units / G.FLUX_WAVEBIN_WIDTH
                            h.fit_continuum = h.mcmc_continuum
                            h.fit_continuum_err = 0.5*(hk_mcmc.mcmc_y[1]+hk_mcmc.mcmc_y[2])/h.fit_bin_dx*10**hk_mcmc.values_units

                #             if (eli.mcmc_a is not None) and (eli.mcmc_y is not None):
                # a_unc = 0.5 * (abs(eli.mcmc_a[1]) + abs(eli.mcmc_a[2])) / eli.mcmc_dx
                # y_unc = 0.5 * (abs(eli.mcmc_y[1]) + abs(eli.mcmc_y[2])) / eli.mcmc_dx
                #
                # estflux = eli.mcmc_line_flux
                # estflux_unc = a_unc
                #
                # estcont = eli.mcmc_continuum
                # estcont_unc = y_unc
                #
                # eqw_obs = abs(estflux / eli.mcmc_continuum)
                # eqw_obs_unc = abs(eqw_obs) * np.sqrt((a_unc / estflux) ** 2 + (y_unc / eli.mcmc_continuum) ** 2)
                #
                #
                #



                            #todo: need to update other fields mcmc_ew_obs, etc
                            if hk_mcmc.mcmc_y[0] != 0 and hk_mcmc.mcmc_A_2[0] != 0:
                                ew = hk_mcmc.mcmc_A_2[0] / hk_mcmc.mcmc_y[0]
                                ew_err = ew * np.sqrt((hk_mcmc.approx_symmetric_error(hk_mcmc.mcmc_A_2) / hk_mcmc.mcmc_A_2[0]) ** 2 +
                                                      (hk_mcmc.approx_symmetric_error(hk_mcmc.mcmc_y) / hk_mcmc.mcmc_y[0]) ** 2)
                            else:
                                ew = hk_mcmc.mcmc_A_2[0]
                                ew_err = hk_mcmc.approx_symmetric_error(hk_mcmc.mcmc_A_2)

                            h.mcmc_ew_obs = [ew, ew_err, ew_err]

                            h.build(values_units=values_units,allow_broad=False,broadfit=False,override_continuum_rules=True)
                            h.raw_score = h.line_score
                            h.score = signal_calc_scaled_score(h.line_score)

                            self.central_eli = h
                            self.h_and_k_mcmc = hk_mcmc

                            #todo: go ahead and add as a solution???

            #choose the highest line_score
            i = -1
            j = -1
            if found_lines:
                try:
                    i = np.nanargmax([x.raw_line_score for x in found_lines])
                except:
                    pass

            if found_absorbers:
                try:
                    j = np.nanargmax([x.raw_line_score for x in found_absorbers])
                except:
                    pass


            pending_central_eli = None
            if (i > -1) and (j > -1):

                # if found_lines[i].line_score > G.MULTILINE_GOOD_LINE_SCORE:
                #     central = found_lines[i].fit_x0
                #now use the line_score if both absorber and emitter found
                if (found_lines[i].snr > found_absorbers[j].snr) or \
                    (found_lines[i].snr > 4.5 and found_lines[i].snr/found_absorbers[j].snr > 0.7):
                    central = found_lines[i].fit_x0
                    pending_central_eli = found_lines[i]
                elif (found_absorbers[j].raw_line_score > (1.5 * found_lines[i].raw_line_score) ) or \
                   (found_absorbers[j].raw_line_score > found_lines[i].raw_line_score and
                    self.gmag is not None and self.gmag < G.BROADLINE_GMAG_MAX):
                    central = found_absorbers[j].fit_x0
                    pending_central_eli = found_absorbers[j]
                    absorber = True
                else:
                    central = found_lines[i].fit_x0
                    pending_central_eli = found_lines[i]
            elif (i > -1):
                central = found_lines[i].fit_x0
                pending_central_eli = found_lines[i]
            else:
                central = found_absorbers[j].fit_x0
                pending_central_eli = found_absorbers[j]
                absorber = True

            if update_self:
                self.central = central#found_lines[i].fit_x0 #max scored line
                self.absorber = absorber
                # if self.central_eli is None:
                #    self.central_eli = pending_central_eli
                #    log.debug(f"Assigning central_eli to found central wavelength: {central}")
                # elif self.central_eli != pending_central_eli:
                #    log.debug(f"Replacing prior central_eli ({self.central_eli.fit_x0}) with {central}")
                #    self.central_eli = pending_central_eli

                #to be the central_eli, you MUST have the mcmc run
                try:
                    if self.central_eli is not None and not np.isclose(self.central_eli.fit_x0, pending_central_eli.fit_x0, atol=4.0):
                        del self.central_eli
                        if pending_central_eli.mcmc_x0 is not None:
                            self.central_eli = pending_central_eli
                        else:
                            self.central_eli = None
                except:
                    pass

            log.info(f"Find central wavelength: {central}")

            if return_list:
                return central, found_lines, found_absorbers
            else:
                return central

        #otherwise use this simpler method to find something

        #find the peaks and use the largest
        #for now, largest means highest value

        # if values_are_flux:
        #     #!!!!! do not do values *= 10.0 (will overwrite)
        #     # assumes fluxes in e-17 .... making e-18 ~= counts so logic can stay the same
        #     values = values * 10.0

        values,values_units = norm_values(values,values_units)

        #does not need errors for this purpose
        peaks = peakdet(wavelengths,values,errors,values_units=values_units,enforce_good=False,spec_obj=self)
        #as of 2018-06-11 these are EmissionLineInfo objects
        max_score = -np.inf
        if peaks is None:
            log.info("Find central wavelength: No viable emission lines found.")
            if return_list:
                return 0.,[],[]
            else:
                return 0.0

        #find the largest flux
        for p in peaks:
            #  0   1   2   3          4
            # pi, px, pv, pix_width, centroid_pos
            #if p[2] > max_v:
            #    max_v = p[2]
            #    central = p[4]
            if p.raw_line_score > max_score:
                max_score = p.raw_line_score
                central = p.fit_x0

        if update_self:
            self.central = central

        log.info("Find central wavelength = %f" %central)

        if return_list:
            return central,[],[]
        else:
            return central

    def classify(self,wavelengths = None,values = None, errors=None, central = None, values_units=0,known_z=None,
                 continuum_limit=None,continuum_limit_err=None,detobj=None):
        #for now, just with additional lines
        #todo: later add in continuum
        #todo: later add in bayseian stuff
        if not G.CLASSIFY_WITH_OTHER_LINES:
            return []

        self.solutions = []
        if (wavelengths is not None) and (values is not None) and (central is not None):
            self.set_spectra(wavelengths,values,errors,central,values_units=values_units,detobj=detobj)
        else:
            wavelengths = self.wavelengths
            values = self.values
            central = self.central
            errors=self.errors
            values_units = self.values_units

        #if central wavelength not provided, find the peaks and use the largest
        #for now, largest means highest value
        if (central is None) or (central == 0.0):
            try:
                if G.CONTINUUM_RULES:
                    central = wavelengths[np.argmax(values)]
                    self.central = central
                else:
                    central = self.find_central_wavelength(wavelengths,values,errors,values_units=values_units)
            except:
                pass

        if (central is None) or (central == 0.0):
            log.warning("Cannot classify. No central wavelength specified or found.")
            return []

        solutions = self.classify_with_additional_lines(wavelengths,values,errors,central,values_units,known_z=known_z,
                                                        continuum_limit=continuum_limit,continuum_limit_err=continuum_limit_err)

        self.solutions = solutions

        #set the unmatched solution (i.e. the solution score IF all the extra lines were unmatched,
        #!!! THIS IS NOT the unmatched score for the best solution) #instead, find the LyA solution and check it specifically
        try:
            self.unmatched_solution_count, self.unmatched_solution_score = self.compute_unmatched_lines_score(Classifier_Solution(self.central))
            log.debug(f"Unmatched solution line count {self.unmatched_solution_count} and score {self.unmatched_solution_score}")
        except:
            log.debug("Exception computing unmatched solution count and score",exc_info=True)

        # if there are no solutions returned, or only weak solutions AND there are multiple strong lines
        # try again, but allow ELiXer to choose the anchor line
        # It may well be that the line passed in is not in the ELiXer list and/or is not a real emission line

        try:
            if False and solutions is None or len(solutions) == 0 or solutions[0].score < G.MULTILINE_WEIGHT_SOLUTION_SCORE:
                #todo: could be more selective and alsu use SNR and chi2
                num_strong_lines = np.count_nonzero([x.raw_score > 10.0 for x in self.all_found_lines])
                if num_strong_lines > 2: #3 or more
                    #save off the original central, central eli, etc
                    #todo: there is MORE to save off and restore
                    saved_central = self.central
                    saved_central_eli = None
                    if self.central_eli is not None:
                        saved_central_eli = copy.deepcopy(self.central_eli)
                        del self.central_eli
                        self.central_eli = None
                    saved_solutions = None
                    if self.solutions is not None:
                        saved_solutions = copy.deepcopy(self.solutions)
                        del self.solutions
                        self.solutions = []

                    try:
                        self.central = self.all_found_lines[np.nanargmax([x.raw_line_score for x in self.all_found_lines])].fit_x0
                    except:
                        log.info("Cannot set self.central in spectrum::classify()", exc_info=True)
                    #self.central = self.find_central_wavelength(wavelengths, values, errors, values_units=values_units)

                    alt_solutions =  self.classify_with_additional_lines(wavelengths,values,errors,
                                                                         self.central,values_units,known_z=known_z,
                                                                         continuum_limit=continuum_limit,
                                                                         continuum_limit_err=continuum_limit_err)

                    if alt_solutions is not None and len(alt_solutions) > 0:
                        print("todo here, choose the best solution and line for the redshift?")

                    #restore originals
                    self.central = saved_central
                    self.central_eli = saved_central_eli
                    self.solutions = saved_solutions

        except:
            log.debug("Exception re-running classify_with_additional_lines() with free anchor line.", exc_info=True)


            #if this has strong continuum AND there is no H&K solution already,
        # explicitly check for an H&K solution (which is not subject to
        #unmatched lines if z ~ 0 ... often a star and there are lots of unmatches lines


        def h_and_k_to_solution(hk_mcmc,h_reference):
            #build out a solution
            sol = Classifier_Solution()
            sol.z = max(0,hk_mcmc.mcmc_mu_2[0]/self.h_and_k_waves[1] - 1.0) #could be slightly negative just due to fitting and rounding
            sol.central_rest = h_reference.w_rest
            sol.name = h_reference.name
            sol.color = h_reference.color
            sol.emission_line = copy.deepcopy(h_reference)
            sol.emission_line.w_obs = hk_mcmc.mcmc_mu_2[1]

            sol.emission_line.w_obs = sol.emission_line.w_rest*(1.0 + sol.z)
            sol.emission_line.solution = True
            sol.prob_noise = 1.0

            hline = copy.deepcopy(h_reference)
            eli = EmissionLineInfo()
            eli.pix_size = 2.0
            eli.fit_bin_dx = 2.0
            eli.sn_pix = hk_mcmc.mcmc_snr_pix
            eli.absorber = True


            eli.hk_mcmc_to_fit(hk_mcmc,hk_mcmc.values_units,G.FLUX_WAVEBIN_WIDTH,use_2=True)
            #eli.mcmc_to_fit(hk_mcmc,hk_mcmc.values_units,2.0)

            eli.fit_sigma = hk_mcmc.mcmc_sigma_2[0]
            eli.fit_sigma_err = (hk_mcmc.mcmc_sigma_2[1] + hk_mcmc.mcmc_sigma_2[1])/2.0
            eli.fit_a = hk_mcmc.mcmc_A_2[0]
            eli.fit_a_err = 0.5 * (hk_mcmc.mcmc_A_2[1]+hk_mcmc.mcmc_A_2[2])
            eli.fit_y = hk_mcmc.mcmc_y[0]
            eli.fit_y_err = 0.5 * (hk_mcmc.mcmc_y[1]+hk_mcmc.mcmc_y[2])

            #eli.line_flux = hk_mcmc.mcmc_A_2[0]/2.0 * 10**hk_mcmc.values_units
            #eli.line_flux_err = 0.25*(hk_mcmc.mcmc_A_2[1]+hk_mcmc.mcmc_A_2[2])* 10**values_units
            #NOTE 0.25* since 0.5 * for the average and /2.0 for the 2AA
            eli.fit_x0 = hk_mcmc.mcmc_mu_2[0]
            eli.fit_dx0 = self.h_and_k_waves[1] - hk_mcmc.mcmc_mu_2[0]/(sol.z+1)
            #eli.w_obs = hk_mcmc.mcmc_mu_2[0] #not using w_obs in EmissionLineInfo object (just in EmissionLine object)
            eli.snr = hk_mcmc.mcmc_snr #technically the SNR for the combined lines, but since
            #these are fit as two, keep this value and maybe boost the score
            eli.build(values_units=hk_mcmc.values_units,allow_broad=False,broadfit=False,override_continuum_rules=True)
            eli.raw_score = eli.line_score
            eli.score = signal_calc_scaled_score(eli.line_score)

            #hit just the necessary fields

            hline = copy.deepcopy(h_reference)
            hline.absorber = True
            hline.fit_sigma = hk_mcmc.mcmc_sigma_2[0]
            hline.fit_sigma_err = (hk_mcmc.mcmc_sigma_2[1] + hk_mcmc.mcmc_sigma_2[1])/2.0
            hline.line_flux = hk_mcmc.mcmc_A_2[0]/2.0 * 10**hk_mcmc.values_units
            hline.line_flux_err = 0.25*(hk_mcmc.mcmc_A_2[1]+hk_mcmc.mcmc_A_2[2])* 10**hk_mcmc.values_units
            #NOTE 0.25* since 0.5 * for the average and /2.0 for the 2AA
            hline.fit_x0 = hk_mcmc.mcmc_mu_2[0]
            hline.fit_dx0 = self.h_and_k_waves[1] - hk_mcmc.mcmc_mu_2[0]/(sol.z+1)
            hline.w_obs = hk_mcmc.mcmc_mu_2[0]
            hline.snr = hk_mcmc.mcmc_snr #technically the SNR for the combined lines, but since
            hline.chi2 = hk_mcmc.mcmc_chi2
            #these are fit as two, keep this value and maybe boost the score
            hline.flux = hline.line_flux
            hline.flux_err = hline.line_flux_err

            hline.fit_line_flux = hline.line_flux
            hline.fit_line_flux_err = hline.line_flux_err

            hline.mcmc_line_flux_tuple = [hline.line_flux,hline.line_flux+hline.line_flux_err,hline.line_flux-hline.line_flux_err]
            hline.mcmc_continuum = hk_mcmc.mcmc_y[0]*10**hk_mcmc.values_units
            hline.mcmc_continuum_tuple = np.array(hk_mcmc.mcmc_y)*10**hk_mcmc.values_units
            hline.fit_continuum = hline.mcmc_continuum
            hline.fit_continuum_err = 0.5*(hk_mcmc.mcmc_y[1]+hk_mcmc.mcmc_y[2])/eli.fit_bin_dx*10**hk_mcmc.values_units

         #   hline.mcmc_line_flux = hline.fit_a_err[0]/hline.fit_bin_dx*10**hk_mcmc.values_units
         #   lineflux_err = hline.mcmc_a[0]/hline.fit_bin_dx*10**hk_mcmc.values_units
         #   hline.mcmc_line_flux_tuple = [hline.mcmc_line_flux,hline.mcmc_line_flux+lineflux_err,hline.mcmc_line_flux-lineflux_err]

            sol.score = eli.line_score
            sol.prob_noise *= eli.prob_noise
            sol.lines.append(hline)
            sol.eli = eli

            return sol


        h_and_k_found = self.h_and_k_mcmc is not None
        if h_and_k_found:
            hk_mcmc = self.h_and_k_mcmc
            h_reference = self.emission_lines[np.where([k.name == "(H)CaII" for k in self.emission_lines])[0][0]]
            sol =  h_and_k_to_solution(self.h_and_k_mcmc,h_reference)
            if sol is not None:
                self.solutions.append(sol)
        else:  #moved to where we find the central wavelength
            hk_mcmc = None
            if (G.CONTINUUM_RULES or (self.estcont > G.CONTNIUUM_RULES_THRESH)) and (self.all_found_absorbs and len(self.all_found_absorbs) > 0) and \
                    not np.any([s.emission_line.w_rest in [self.h_and_k_waves] for s in solutions]):

                x0 = np.array([k.fit_x0 for k in self.all_found_absorbs])
                x0e = np.array([k.fit_x0_err for k in self.all_found_absorbs])
                # sel = x0+x0e > self.h_and_k_waves[0]
                #
                # x0 = x0[sel]
                # x0e = x0e[sel]


                #future: cannot just check for already found absorption lines with the correct separations
                #since the fits can fail for a variety of reasons ... should scan again as double aborption Gaussian?
                #BUT, usually they should be found (mostly the problem is with stars and extra lines, and there
                #is a work around implemented earlier for that

                #check pairs of lines ... must be the right separation and similar area, depth, sigma, snr
                #start from 3933 (some slop from 3934) at z=0 and work out

                k_reference = self.emission_lines[np.where([k.name == "(K)CaII" for k in self.emission_lines])[0][0]]
                h_reference = self.emission_lines[np.where([k.name == "(H)CaII" for k in self.emission_lines])[0][0]]

                wsel = np.array(x0 > 3932.0) #allow a little blue of 3934 for slop

               # log.debug(f"Checking for H&K in possible abosorption lines near: {sorted([x.raw_x0 for x in np.array(self.all_found_absorbs)[wsel]])}")

                for k in np.array(self.all_found_absorbs)[wsel]:
                    # #if (k.fit_x0 + k.fit_x0_err)  > self.h_and_k_waves[0]: #could be k
                    if True:
                        zp1 = k.fit_x0/self.h_and_k_waves[0]
                        zp1e = k.fit_x0_err/self.h_and_k_waves[0]
                        #
                        # hwave = zp1*self.h_and_k_waves[1]
                        # hwave_err = zp1e*self.h_and_k_waves[1]

                        # sel1 = (x0-x0e) > (hwave-hwave_err)
                        # sel2 = (x0+x0e) < (hwave+hwave_err)
                        # sel = sel1 & sel2

                        sel = np.isclose(self.h_and_k_waves[1]*zp1,[x.fit_x0 for x in self.all_found_absorbs],atol=2.5)

                        #there is one pair that fits h & k positions
                        # could be one or more? should only expect one match though
                        for h in np.array(self.all_found_absorbs)[sel]:

                            good,hk_mcmc = fit_for_h_and_k(k,h,wavelengths,values,errors,values_units,values_dx=G.FLUX_WAVEBIN_WIDTH)

                            if (good is not None) and good:
                               sol =  h_and_k_to_solution(hk_mcmc,h_reference)
                               if self.h_and_k_mcmc is None:
                                   self.h_and_k_mcmc = copy.deepcopy(hk_mcmc)
                               else:
                                   if self.h_and_k_mcmc.mcmc_snr < self.h_and_k_mcmc.mcmc_snr:
                                       del self.h_and_k_mcmc
                                       self.h_and_k_mcmc = copy.deepcopy(hk_mcmc)
                               if sol is not None:
                                   self.solutions.append(sol)
                                   h_and_k_found = True

                #H&K are so common, if we did not find them in the noted absorption featuers,
                #try, eplicitly near H&K at z = 0
                # if not h_and_k_found:
                #     if not (np.any(np.isclose(self.h_and_k_waves[0],[x.fit_x0 for x in self.all_found_absorbs],atol=6.0)) and
                #             np.any(np.isclose(self.h_and_k_waves[1],[x.fit_x0 for x in self.all_found_absorbs], atol=6.0))):
                #         #one or both were NOT in the original list
                #         good, hk_mcmc = fit_for_h_and_k(self.h_and_k_waves[0], self.h_and_k_waves[1],
                #                                         wavelengths, values, errors, values_units,
                #                                         values_dx=G.FLUX_WAVEBIN_WIDTH)
                #
                #         if (good is not None) and good:
                #             sol = h_and_k_to_solution(hk_mcmc, h_reference)
                #             if self.h_and_k_mcmc is None:
                #                 self.h_and_k_mcmc = copy.deepcopy(hk_mcmc)
                #             else:
                #                 if self.h_and_k_mcmc.mcmc_snr < self.h_and_k_mcmc.mcmc_snr:
                #                     del self.h_and_k_mcmc
                #                     self.h_and_k_mcmc = copy.deepcopy(hk_mcmc)
                #             if sol is not None:
                #                 self.solutions.append(sol)
                #                 h_and_k_found = True


        if h_and_k_found:
            try:
                self.rescore()#undo_absorb_penalty=True)
                #todo: if the h or k line is now the top score, set the central wavelength to that solution??
                if self.solutions[0].central_rest in self.h_and_k_waves: #can't that is in hetdex object?

                    if self.h_and_k_mcmc is not None:
                        hk_mcmc = self.h_and_k_mcmc


                    if self.solutions[0].central_rest == self.h_and_k_waves[1]:
                        self.central = hk_mcmc.mcmc_mu_2[0]
                        is_3968 = True #H line
                    else:
                        self.central = hk_mcmc.mcmc_mu[0]
                        is_3968 = False #K line

                    try:
                        eli = self.solutions[0].eli
                    except:
                        eli = None

                    if eli is None:

                        eli = EmissionLineInfo() #this will become the new central_eli so needs to be fully populated

                        eli.hk_mcmc_to_fit(hk_mcmc, hk_mcmc.values_units, values_dx=G.FLUX_WAVEBIN_WIDTH,use_2=is_3968)

                        eli.pix_size = 2.0
                        eli.fit_bin_dx = 2.0
                        eli.sn_pix = hk_mcmc.mcmc_snr_pix
                        eli.absorber = True
                        #eli.fit_sigma = hk_mcmc.mcmc_sigma_2[0]
                        #eli.fit_sigma_err = (hk_mcmc.mcmc_sigma_2[1] + hk_mcmc.mcmc_sigma_2[1])/2.0
                        #eli.fit_a = hk_mcmc.mcmc_A_2[0]
                        #eli.fit_a_err = 0.5 * (hk_mcmc.mcmc_A_2[1] + hk_mcmc.mcmc_A_2[2])
                        #eli.fit_y = hk_mcmc.mcmc_y[0]
                        #eli.fit_y_err = 0.5 * (hk_mcmc.mcmc_y[1]+hk_mcmc.mcmc_y[2])


                        #eli.fit_chi2 = hk_mcmc.mcmc_chi2

                        #eli.line_flux = hk_mcmc.mcmc_A_2[0]/2.0 * 10**values_units
                        #eli.line_flux_err = 0.25*(hk_mcmc.mcmc_A_2[1]+hk_mcmc.mcmc_A_2[2])* 10**values_units
                        #NOTE 0.25* since 0.5 * for the average and /2.0 for the 2AA
                        #eli.fit_x0 = hk_mcmc.mcmc_mu_2[0]
                        #still need this one ... how far off is our fitted x0?
                        eli.fit_dx0 = self.h_and_k_waves[1] - hk_mcmc.mcmc_mu_2[0]/(hk_mcmc.mcmc_mu[0]/self.h_and_k_waves[0])
                        #eli.w_obs = hk_mcmc.mcmc_mu_2[0]
                        #eli.snr = hk_mcmc.mcmc_snr #technically the SNR for the combined lines, but since
                        #these are fit as two, keep this value and maybe boost the score



                        eli.build(values_units=values_units,allow_broad=False,broadfit=False,override_continuum_rules=True)
                        eli.raw_score = eli.line_score
                        eli.score = signal_calc_scaled_score(eli.line_score)

                    self.central_eli = eli
            except:
                log.warning("Exception! in Spectrum.classify() switching to H&K Solution.",exc_info=True)

        #get the LAE and OII solutions and send to Bayesian to check p_LAE/p_OII
        self.addl_fluxes = []
        self.addl_wavelengths = []
        self.addl_fluxerrs = []
        for s in solutions:
            if (abs(s.central_rest - G.LyA_rest) < 2.0) or \
               (abs(s.central_rest - G.OII_rest) < 2.0): #LAE or OII

                for l in s.lines:
                    if l.flux > 0:
                        self.addl_fluxes.append(l.flux)
                        self.addl_wavelengths.append((l.w_obs))
                        #todo: get real error (don't have that unless I run mcmc and right now, only running on main line)
                        self.addl_fluxerrs.append(0.0)
                       # self.addl_fluxerrs.append(l.flux*.3)

        #if len(addl_fluxes) > 0:
        self.get_bayes_probabilities(addl_wavelengths=self.addl_wavelengths,addl_fluxes=self.addl_fluxes,
                                     addl_errors=self.addl_fluxerrs)
        #self.get_bayes_probabilities(addl_fluxes=None, addl_wavelengths=None)

        return solutions


    def is_near_a_peak(self,w,aa=4.0,return_score=False): #is the provided wavelength near one of the found peaks (+/- few AA or pixels)

        wavelength = 0.0
        score = None
        if (self.all_found_lines is None):
            self.all_found_lines = peakdet(self.wavelengths, self.values, self.errors,values_units=self.values_units,spec_obj=self)

        if self.all_found_lines is None:
            if return_score:
                return 0.0, None
            else:
                return 0.0

        for f in self.all_found_lines:
            if abs(f.fit_x0 - w) < aa:
                wavelength = f.fit_x0
                score = f.line_score
                break
        if return_score:
            return wavelength,score
        else:
            return wavelength

    def compute_unmatched_lines_score(self,solution,aa=4.0):
        """
        Return the lines and summed line scores for unmatched lines associated with a solution
        :param solutions:
        :param aa:
        :return:
        """

        try:
            if G.CONTINUUM_RULES and self.central_eli.gmag < 20:
                log.info(f"Continuum rules and object too bright (g = {self.central_eli.gmag:0.1f}). Skipping unmatched line scoring.")
                return 0,0
        except:
            pass

        def rescale(wave):
            #full scale at 3700 down to 1/2 scale at 3550 and below (noisiest section and on nasty sky-line), linear
            try:
                if wave < 3700:
                    return  max(0.5,1./300. * wave - 34./3.)
                else:
                    return 1.0
            except:
                return 1.0

        try:

            try:
                if solution.emission_line and solution.emission_line.absorber:
                    aa = max(aa,20.0)
            except:
                pass

            if (self.all_found_lines is None):
                self.all_found_lines = peakdet(self.wavelengths, self.values, self.errors,values_units=self.values_units,spec_obj=self)

            if self.all_found_lines is None or len(self.all_found_lines)==0:
                return 0,0

            if G.CONTINUUM_RULES:
                line_score_threshold = 2.0 * GOOD_MIN_LINE_SCORE
            else:
                line_score_threshold = GOOD_MIN_LINE_SCORE

            #tweak down the score for lines < 3700 (near, but blue of OII and well into the noisiest part)
            unmatched_score_list = np.array([x.line_score * rescale(x.fit_x0)
                                             for x in self.all_found_lines if (3550.0 < x.fit_x0 < 5500.0) and (x.line_score > line_score_threshold) ])
            unmatched_raw_score_list = np.array([x.raw_line_score * rescale(x.fit_x0)
                                             for x in self.all_found_lines if
                                             (3550.0 < x.fit_x0 < 5500.0) and (x.line_score > line_score_threshold)])
            unmatched_wave_list = np.array([x.fit_x0 for x in self.all_found_lines if (3550.0 < x.fit_x0 < 5500.0) and (x.line_score > line_score_threshold)])
            unmatched_wave_sigma = np.array([x.fit_sigma for x in self.all_found_lines if (3550.0 < x.fit_x0 < 5500.0) and (x.line_score > line_score_threshold)])

            solution_wave_list = np.array([solution.central_rest * (1.+solution.z)] + [x.w_obs for x in solution.lines])
            solution_score_list = np.array([0.0] + [x.line_score for x in solution.lines ])
            solution_sigma = np.array([0.0] + [x.sigma for x in solution.lines])

                #0 for the central line, as it does not count toward the multiline score

            #there can be LOTS of lines, esp for bright objects. Many will be false lines or junk and their scores
            # will likely be low, BUT, because we cap the maximum score, lots of junk lines can add up and kick out
            # an otherwise good solution, SO, limit the number of lines
            top_x = 5
            if len(unmatched_raw_score_list) > top_x:
                sorted_scores = sorted(unmatched_raw_score_list)[::-1]
                top_thresh = sorted_scores[top_x]
                top_sel = unmatched_raw_score_list > top_thresh
                unmatched_score_list  = unmatched_score_list[top_sel]
                unmatched_wave_list  = unmatched_wave_list[top_sel]
                unmatched_wave_sigma = unmatched_wave_sigma[top_sel]

            #decrease score to deal with huge FWHM or sigma (note 8.5+ sigma is generally pretty braod)
            sigma_sel = unmatched_wave_sigma > 10.0
            if np.count_nonzero(sigma_sel) > 0:
                unmatched_score_list[sigma_sel] = unmatched_score_list[sigma_sel] * (10.0/unmatched_wave_sigma[sigma_sel])

            #do the same for the solution
            sigma_sel = solution_sigma > 10.0
            if np.count_nonzero(sigma_sel) > 0:
                solution_score_list[sigma_sel] = solution_score_list[sigma_sel] * (10.0/solution_sigma[sigma_sel])


            for line in solution_wave_list:
                idx = np.where(abs(unmatched_wave_list-line) <= aa)[0]
                if idx is not None and len(idx) > 0: #should usually be just 0 or 1
                    #remove from unmatched_list as these are now matched
                    idx = idx[::-1]
                    for i in idx:
                        unmatched_wave_list = np.delete(unmatched_wave_list,i)
                        unmatched_score_list = np.delete(unmatched_score_list,i)

            #now check based on line FWHM (broad lines found differently could be off in peak position, but overlap)
            for i in range(len(unmatched_wave_list)-1,-1,-1):
                for line in solution.lines:
                    if abs(line.w_obs - unmatched_wave_list[i]) < (3.0*line.sigma):
                        unmatched_wave_list = np.delete(unmatched_wave_list,i)
                        unmatched_score_list = np.delete(unmatched_score_list,i)
                        break


            #also have to check anchor line (as fit)
            try:
                if self.central_eli is not None:
                    for i in range(len(unmatched_wave_list)-1,-1,-1):
                        line = self.central_eli
                        if abs(line.fit_x0 - unmatched_wave_list[i]) < (3.0*line.fit_sigma):
                            unmatched_wave_list = np.delete(unmatched_wave_list,i)
                            unmatched_score_list = np.delete(unmatched_score_list,i)


                elif self.fwhm is not None: #no central eli (might not have a fitted fwhm)
                    for i in range(len(unmatched_wave_list)-1,-1,-1):
                        if abs(self.central- unmatched_wave_list[i]) < (3.0*self.fwhm/2.355):
                            unmatched_wave_list = np.delete(unmatched_wave_list,i)
                            unmatched_score_list = np.delete(unmatched_score_list,i)

            except:
                log.info("Warning. Exception checking anchor line vs unmatched lines.",exc_info=True)


            #what is left over
            if len(unmatched_score_list) > 0:
                if G.CONTINUUM_RULES:
                    unmatched_score_list /= 2.0 #for continuum sources, there are many spurious lines; to help counter that, reduce the score
                    reduced = "*continuum reduced*"
                else:
                    reduced = ""

                unmatched_sum =  np.nansum(unmatched_score_list)
                solution_sum =  np.nansum(solution_score_list)

                if len(solution_wave_list) >= 3 and  len(solution_wave_list) > len(unmatched_wave_list) and solution_sum > unmatched_sum:
                    return 0,0 #we will let it go
                else:
                    log.debug("Unmatched lines: (wave,score): " + reduced + str([(w,s) for w,s in zip(unmatched_wave_list,unmatched_score_list)]))
                    return len(unmatched_score_list), np.nansum(unmatched_score_list)
            return 0,0
        except:
            log.debug("Exception in spectrum::compute_unmatched_lines_score",exc_info=True)
            return 0,0

    def is_near_absorber(self,w,aa=4.0):#pix_size=1.9): #is the provided wavelength near one of the found peaks (+/- few AA or pixels)

        if not (G.DISPLAY_ABSORPTION_LINES or G.MAX_SCORE_ABSORPTION_LINES):
            return 0.0

        wavelength = 0.0
        if (self.all_found_absorbs is None):
            self.all_found_absorbs = peakdet(self.wavelengths, self.values,
                                             self.errors, values_units=self.values_units,absorber=True,spec_obj=self)
            self.clean_absorbers()

        if self.all_found_absorbs is None:
            return 0.0

        for f in self.all_found_absorbs:
            if abs(f.fit_x0 - w) < aa:
                wavelength = f.fit_x0
                break

        return wavelength


    def clean_absorbers(self):
        #the intent is to not mark a "partial trough next to a peak as an absorption feature
        #but this does not really do the job
        #really should properly fit an absorption profile and not use this cheap, flip the spectra approach
        return
        if self.all_found_absorbs is not None:
            for i in range(len(self.all_found_absorbs)-1,-1,-1):
                if self.is_near_a_peak(self.all_found_absorbs[i].fit_x0,aa=10.0):
                    del self.all_found_absorbs[i]



    def classify_with_additional_lines(self,wavelengths = None,values = None,errors=None,central = None,
                                       values_units=0,known_z=None,continuum_limit=None, continuum_limit_err=None):
        """
        using the main line
        for each possible classification of the main line
            for each possible additional line
                if in the range of the spectrum
                    fit a line (?gaussian ... like the score?) to the exact spot of the additional line
                        (allow the position to shift a little)
                    get the score and S/N (? how best to get S/N? ... look only nearby?)
                    if score is okay and S/N is at least some minium (say, 2)
                        add to weighted solution (say, score*S/N)
                    (if score or S/N not okay, skip and move to the next one ... no penalties)

        best weighted solution wins
        ?what about close solutions? maybe something where we return the best weight / (sum of all weights)?
        or (best weight - 2nd best) / best ?

        what to return?
            with a clear winner:
                redshift of primary line (z)
                rest wavelength of primary line (e.g. effectively, the line identification) [though with z is redundant]
                list of additional lines found (rest wavelengths?)
                    and their scores or strengths?

        should return all scores? all possible solutions? maybe a class called classification_solution?

        """

        if (values is None) or (wavelengths is None) or (central is None):
            values = self.values
            wavelengths = self.wavelengths
            errors = self.errors
            central = self.central
            values_units = self.values_units

        if (self.all_found_lines is None):
            self.all_found_lines = peakdet(wavelengths,values,errors, values_units=values_units,spec_obj=self)

        if (G.CONTINUUM_RULES or G.DISPLAY_ABSORPTION_LINES or G.MAX_SCORE_ABSORPTION_LINES) and (self.all_found_absorbs is None):
            self.all_found_absorbs = peakdet(wavelengths, values,errors,
                                            values_units=values_units,absorber=True,spec_obj=self)
            self.clean_absorbers()


        solutions = []

        #if G.CONTINUUM_RULES:
        #    return solutions

        per_line_total_score = 0.0 #sum of all scores (use to represent each solution as fraction of total score)


        #for each self.emission_line
        #   run down the list of remianing self.emission_lines and calculate score for each
        #   make a copy of each emission_line, set the score, save to the self.lines list []
        #
        #sort solutions by score

        max_w = max(wavelengths)
        min_w = min(wavelengths)

        #if the central line is in absorption, only test the lines that are absorbers
        for e in [ee for ee in self.emission_lines if ee.absorber == self.absorber]:
            #!!! consider e.solution to mean it cannot be a lone solution (that is, the line without other lines)
            #if not e.solution: #changed!!! this line cannot be the ONLY line, but can be the main line if there are others
            #    continue

            if not G.CONTINUUM_RULES and e.see_in_absorption:
                continue #we are looking for emision only and this line is not avaible to match in emission
            #else: central can be either emission or absorption

            anchor_ew_score_penalty_factor = 1.0 #apply if the bin line is outside the accepted rest eqwidth range

            central_z = central/e.w_rest - 1.0
            if (central_z) < 0.0:
                if central_z > G.NEGATIVE_Z_ERROR: #assume really z=0 and this is wavelength error
                    central_z = 0.0
                else:
                    continue #impossible, can't have a negative z

            #check the EW range
            if e.rest_ew_range is not None:
                try:
                    ew_rest = self.eqw_obs / (central_z + 1)
                    if (e.rest_ew_range[0] is not None and ew_rest < e.rest_ew_range[0]) or \
                       (e.rest_ew_range[1] is not None and ew_rest > e.rest_ew_range[1]):
                        log.info(f"Penalizing bid anchor line {e.name}, {e.w_rest:0.1f} as EWr {ew_rest:0.1f} is out of range ({e.rest_ew_range}). ")
                        anchor_ew_score_penalty_factor *= 0.5

                except:
                    pass

            if known_z is not None:
                if abs(central_z-known_z) > 0.05:
                    log.info(f"Known z {known_z:0.2f} invalidates solution for {e.name} at z = {central_z:0.2f}")
                    continue
            elif (self.fwhm) and (self.fwhm_unc) and (((self.fwhm-self.fwhm_unc)/2.355 > LIMIT_BROAD_SIGMA) and not e.broad):
                log.info(f"FWHM ({self.fwhm},+/- {self.fwhm_unc}) too broad for {e.name}. Solution disallowed.")
                continue
            else:
                #normal rules apply only allow major lines or lines marked as allowing a solution

                #2020-09-08 DD take out the check for rank 4; extra comparisons later make this no longer necessary
                #to filter out noisy matches
                # if e.rank > 4:  # above rank 4, don't even consider as a main line (but it can still be a supporting line)
                #    continue

                try:
                    if not (e.solution) and (e.min_obs_wave < central < e.max_obs_wave) and (self.fwhm >= e.min_fwhm):
                        e.solution = True  # this change applies only to THIS instance of a spectrum, so it is safe
                except:  # could be a weird issue
                    if self.fwhm is not None:
                        log.debug("Unexpected exception in specturm::classify_with_additional_lines", exc_info=True)
                    # else: #likely because we could not get a fit at that position

            if e.w_rest == G.LyA_rest:
                #assuming a maximum expected velocity offset, we can allow the additional lines to be
                #asymmetrically less redshifted than LyA
                GOOD_MAX_DX0_MULT = [MAX_LYA_VEL_OFFSET / 3.0e5 * central  ,GOOD_MAX_DX0_MULT_LYA[1]]
            else:
                GOOD_MAX_DX0_MULT = GOOD_MAX_DX0_MULT_OTHER

            sol = Classifier_Solution()
            sol.z = central_z
            sol.central_rest = e.w_rest
            sol.name = e.name
            sol.color = e.color
            sol.emission_line = copy.deepcopy(e)
            sol.emission_line.w_obs = sol.emission_line.w_rest*(1.0 + sol.z)
            sol.emission_line.solution = True
            sol.emission_line.z = central_z
            sol.prob_noise = 1.0

            for a in self.emission_lines: #2021-06-14 ... now emission and absorption
                if e == a:
                    continue

                if not G.CONTINUUM_RULES and a.see_in_absorption:
                    continue #this is an absorption line, but we are not checking for absorption

                a_central = a.w_rest*(sol.z+1.0)
                if (a_central > max_w) or (a_central < min_w) or (abs(a_central-central) < 5.0):
                    continue

                supporting_ew_score_penalty_factor = 1.0

                log.debug("Testing line solution. Anchor line (%s, %0.1f) at %0.1f, target line (%s, %0.1f) at %0.1f."
                          %(e.name,e.w_rest,e.w_rest*(1.+central_z),a.name,a.w_rest,a_central))



                try:
                    if a.rank < 4:
                        if a.rank <= e.rank and self.central_eli is not None:
                            min_sigma = max(self.central_eli.fit_sigma/2.0,2.0)
                        else:
                            min_sigma = 2.0
                    else:
                        min_sigma = GAUSS_FIT_MIN_SIGMA
                except:
                    min_sigma = GAUSS_FIT_MIN_SIGMA

                try:
                    max_sigma = self.central_eli.fit_sigma
                    if self.central_eli.fit_sigma_err is not None:
                        max_sigma += self.central_eli.fit_sigma_err

                    max_sigma = max(20.0,max_sigma * 2)

                except:
                    max_sigma = None

                eli = signal_score(wavelengths=wavelengths, values=values, errors=errors, central=a_central,
                                   central_z = central_z, values_units=values_units, spectrum=self,
                                   show_plot=False, do_mcmc=False,min_fit_sigma=min_sigma,
                                   allow_broad= (a.broad and e.broad),
                                   relax_fit=(e.w_rest==G.OIII_5007)and(a.w_rest==G.OIII_4959),absorber=a.see_in_absorption,
                                   test_solution=sol,spec_obj=self,targetted_fit=True,max_sigma=max_sigma)

                if eli and a.broad and e.broad and (eli.fit_sigma < eli.fit_sigma_err) and \
                    ((eli.fit_sigma + eli.fit_sigma_err) > GOOD_BROADLINE_SIGMA):
                        #try again with medfilter fit
                        eli = signal_score(wavelengths=wavelengths, values=medfilt(values, 5), errors=medfilt(errors, 5),
                            central=a_central, central_z = central_z, values_units=values_units, spectrum=self,
                            show_plot=False, do_mcmc=False, allow_broad= (a.broad and e.broad), absorber=a.see_in_absorption,
                                           spec_obj=self,targetted_fit=True,max_sigma=max_sigma)
                elif eli is None and a.broad and e.broad:
                    #are they in the same family? OII, OIII, OIV :  CIV, CIII, CII : H_beta, ....
                    samefamily = in_same_family(a,e)

                    if not samefamily or (samefamily and (self.central_eli and self.central_eli.fit_sigma and self.central_eli.fit_sigma > 5.0)):
                        eli = signal_score(wavelengths=wavelengths, values=medfilt(values, 5), errors=medfilt(errors, 5),
                                       central=a_central, central_z=central_z, values_units=values_units, spectrum=self,
                                       show_plot=False, do_mcmc=False, allow_broad=(a.broad and e.broad),broadfit=5,
                                           absorber=a.see_in_absorption,
                                           spec_obj=self,targetted_fit=True)
                # elif a.broad: #the supporting line could be broad
                #     common_combo = False
                #
                #     #MgII and OII
                #     if 2794 < a.w_rest < 2802: #MgII (normally set to 2977, but is a doublet)
                #         common_combo = True
                #
                #     if common_combo:
                #         eli_broad = signal_score(wavelengths=wavelengths, values=medfilt(values, 5),
                #                            errors=medfilt(errors, 5),
                #                            central=a_central, central_z=central_z, values_units=values_units,
                #                            spectrum=self,
                #                            show_plot=False, do_mcmc=False, allow_broad=True,
                #                            broadfit=5)
                #
                #         #could score better as a broad line
                #         if eli:
                #             if eli_broad and eli_broad.line_score > eli.line_score:
                #                 eli = eli_broad
                #         else:
                #             eli = eli_broad

                #try as absorber


                # #now handled above
                # if G.MAX_SCORE_ABSORPTION_LINES and eli is None and self.is_near_absorber(a_central):
                #     # eli = signal_score(wavelengths=wavelengths, values=invert_spectrum(wavelengths,values), errors=errors, central=a_central,
                #     #                    central_z=central_z, values_units=values_units, spectrum=self,
                #     #                    show_plot=False, do_mcmc=False,absorber=True)
                #     #we no longer want to invert (we will fit to absorption)
                #     eli = signal_score(wavelengths=wavelengths, values=values, errors=errors, central=a_central,
                #                        central_z=central_z, values_units=values_units, spectrum=self,
                #                        show_plot=False, do_mcmc=False,absorber=True)
                #

                good = False

                allow_broad_check = e.broad and a.broad
                if not allow_broad_check and a.broad:
                    #certain combinations are okay though
                    #MgII (i.e. with OII, OII is narrow-ish but MgII can be broad)
                    if 2794 < a.w_rest < 2802: #MgII
                        allow_broad_check = True
                    elif 1548 < a.w_rest < 1551: #CIV usually 1549
                        allow_broad_check = True
                    elif a.w_rest == G.LyA_rest:
                        allow_broad_check = True
                    elif a.w_rest == G.OII_rest: #OII is sometimes broad
                        allow_broad_check = True

                if (eli is not None) and eli.is_good(z=sol.z,allow_broad=allow_broad_check,supporting_line=True):
                    good = True

                if a.rest_ew_range is not None:
                    try:
                        ew_rest =  eli.eqw_obs / (central_z + 1)
                        if (a.rest_ew_range[0] is not None and ew_rest < a.rest_ew_range[0]) or \
                           (a.rest_ew_range[1] is not None and ew_rest > a.rest_ew_range[1]):
                            log.info(
                                f"Penalizing bid supporting line {a.name}, {a.w_rest:0.1f} as EWr {ew_rest:0.1f} is out of range ({a.rest_ew_range}). ")
                            supporting_ew_score_penalty_factor *= 0.5

                    except:
                        pass

                #This has to happen AFTER is good check
                #apply any score modifications based on the line identification (made in the definitions of the EmissionLine objects)
                #i.e. for LyA+NV vs OII+complex at 3800AA-rest
                if eli and ((a.score_multiplier != 1.0) or (supporting_ew_score_penalty_factor != 1.0)):
                    try:
                        old_score = eli.line_score
                        eli.line_score = eli.line_score * a.score_multiplier * supporting_ew_score_penalty_factor
                        log.debug(f"Down-scoring {a.name},{a.w_rest} per definition. Old score {old_score}, new score {eli.line_score}")
                    except:
                        log.warning("Exception. Unexpected execption down-scoring NV in Spectrum::classify_with_additional_lines()")


                #specifically check for 5007 and 4959 as nasty LAE contaminatant
                if eli and not good:
                    try:
                        if (np.isclose(a.w_rest,G.OIII_4959,atol=1.0) and np.isclose(e.w_rest,G.OIII_5007,atol=1.0)):
                            ratio = self.central_eli.fit_a / eli.fit_a
                            ratio_err = abs(ratio) * np.sqrt( (eli.fit_a_err / eli.fit_a) ** 2 +
                                                    (self.central_eli.fit_a_err / self.central_eli.fit_a) ** 2)

                            if (ratio - ratio_err) < 3 < (ratio + ratio_err):
                                good = True

                        elif (np.isclose(a.w_rest,G.OIII_5007,atol=1.0) and np.isclose(e.w_rest,G.OIII_4959,atol=1.0)):
                            ratio = eli.fit_a / self.central_eli.fit_a
                            ratio_err = abs(ratio) * np.sqrt( (eli.fit_a_err / eli.fit_a) ** 2 +
                                                    (self.central_eli.fit_a_err / self.central_eli.fit_a) ** 2)

                            if (ratio - ratio_err) < 3 < (ratio + ratio_err):
                                good = True
                        elif G.CONTINUUM_RULES or continuum_limit > G.CONTNIUUM_RULES_THRESH: #bright, OIII 4959 can get kist
                            #the anchor might be a Balmer series or OII-3272 (note OIII-5007 already checked above)
                            if np.isclose(a.w_rest, G.OIII_4959, atol=1.0):
                                if eli.fit_sigma < 10 and eli.snr > 5.5 and eli.raw_line_score > 25.0 and eli.fit_chi2 < 3.0:
                                    good = True
                    except:
                        pass

                if good:
                    #if this line is too close to another, keep the one with the better score
                    add_to_sol = True

                    # BASIC FWHM check (by rank)
                    # higher ranks are stronger lines and must have similar or greater fwhm (or sigma)
                    #rank 1 is highest, 4 lowest; a is the line being tested, e is the solution anchor line
                    if (a.rank < e.rank) or ((a.rank == e.rank) and in_same_family(a,e)):
                        try: #todo: something similar in the specific consistency checks? (not sure here anyway since fwhm is related to lineflux)
                            #maybe fit_h is a better, more independent factor?
                            #but needs to be height above continuum, so now we are looking at EqW
                            #and we're just going in circles. Emprically, line_flux seems to work better than the others
                            # adjust = eli.line_flux / self.central_eli.line_flux
                            # #adjust = (eli.fit_h-eli.fit_y) / (self.central_eli.fit_h - self.central_eli.fit_y)
                            # adjust = min(adjust,1.0/adjust)

                            if (a.broad == e.broad):
                                adjust = 1.0 #they should be similar (both broad or narrow)
                            elif (a.broad):
                                adjust = 0.33 #3.0 #the central line can be more narrow
                            elif (e.broad):
                                adjust = 3.0 #0.33 #the central line can be more broad

                            #2.0 x (s1 - s2)/(s1+s2) is the difference divided by the mean
                            fwhm_comp = adjust * 2.0 * (eli.fit_sigma - self.central_eli.fit_sigma)  / \
                                        (eli.fit_sigma + self.central_eli.fit_sigma)

                            #if -0.5 < fwhm_comp  < 0.5: #too strict ... specifically need to worry about
                            #combo's of lines where one of the combo (like MgII or even CIV is a doublet) and the
                            #other is not
                            if -1.0 < fwhm_comp  < 1.0:
                                    # delta sigma is okay, the higher rank is larger sigma (negative result) or within 50%
                                pass
                            else:
                                if a.snr < 5.7:
                                    log.debug(f"Sigma sanity check failed {self.identifier}. Disallowing {a.name} at sigma {eli.fit_sigma:0.2f} "
                                              f" vs anchor sigma {self.central_eli.fit_sigma:0.2f}")
                                    add_to_sol = False
                                # this line should not be part of the solution
                        except:
                            pass


                    #check the main line first
                    if abs(central - eli.fit_x0) < 5.0:
                        # the new line is not as good so just skip it
                        log.debug("Emission line (%s) at (%f) close to or overlapping primary line (%f). Rejecting."
                                 % (self.identifier, eli.fit_x0,central))
                        add_to_sol = False

                    else:
                        for i in range(len(sol.lines)):
                            if abs(sol.lines[i].w_obs - eli.fit_x0) < 10.0:

                                #keep the emission line over the absorption line, regardless of score, if that is the case
                                if sol.lines[i].absorber != eli.absorber:
                                    if eli.absorber:
                                        log.debug("Emission line too close to absorption line (%s). Removing %s(%01.f) "
                                                 "from solution in favor of %s(%0.1f)"
                                            % (self.identifier, a.name, a.w_rest, sol.lines[i].name, sol.lines[i].w_rest))

                                        add_to_sol = False
                                    else:
                                        log.debug("Emission line too close to absorption line (%s). Removing %s(%01.f) "
                                                 "from solution in favor of %s(%0.1f)"
                                            % (self.identifier, sol.lines[i].name, sol.lines[i].w_rest, a.name, a.w_rest))
                                        # remove this solution
                                        per_line_total_score -= sol.lines[i].line_score
                                        sol.score -= sol.lines[i].line_score
                                        sol.prob_noise /= sol.lines[i].prob_noise
                                        del sol.lines[i]
                                else: #they are are of the same type, so keep the better score
                                    if sol.lines[i].line_score < eli.line_score:
                                        log.debug("Lines too close (%s). Removing %s(%01.f) from solution in favor of %s(%0.1f)"
                                                 % (self.identifier,sol.lines[i].name, sol.lines[i].w_rest,a.name, a.w_rest))
                                        #remove this solution
                                        per_line_total_score -= sol.lines[i].line_score
                                        sol.score -= sol.lines[i].line_score
                                        sol.prob_noise /= sol.lines[i].prob_noise
                                        del sol.lines[i]
                                        break
                                    else:
                                        #the new line is not as good so just skip it
                                        log.debug("Lines too close (%s). Removing %s(%01.f) from solution in favor of %s(%0.1f)"
                                                 % (self.identifier,a.name, a.w_rest,sol.lines[i].name, sol.lines[i].w_rest))
                                        add_to_sol = False
                                        break

                    #now, before we add, if we have not run MCMC on the feature, do so now
                    if G.MIN_MCMC_SNR > 0:
                        if add_to_sol:
                            if eli.mcmc_x0 is None:
                                eli = run_mcmc(eli,wavelengths,values,errors,a_central,values_units)

                            #and now validate the MCMC SNR (reminder:  MCMC SNR is line flux (e.g. Area) / (1sigma uncertainty)
                            if eli.mcmc_snr is None:
                                add_to_sol = False
                                log.debug("Line (at %f) rejected due to missing MCMC SNR" %(a_central))
                            elif eli.mcmc_snr < G.MIN_MCMC_SNR:
                                add_to_sol = False
                                log.debug("Line (at %f) rejected due to poor MCMC SNR (%f)" % (a_central,eli.mcmc_snr))
                            #todo: should we recalculate the score with the MCMC data (flux, SNR, etc)??
                            #todo: or, at this point, is this a binary condition ... the line is there, or not
                            #todo: .... still with multiple solutions possible, we must meet the minimum and then the best
                            #todo:      score (clear winner) wins


                    if add_to_sol:
                        l = copy.deepcopy(a)
                        l.w_obs = l.w_rest * (1.0 + sol.z)
                        l.z = sol.z
                        l.score = eli.score
                        l.snr = eli.snr
                        l.sbr = eli.sbr
                        l.chi2 = eli.fit_chi2
                        l.eqw_obs = eli.eqw_obs
                        l.eqw_rest = l.eqw_obs / (1.0 + l.z)
                        l.flux = eli.line_flux
                        l.flux_err = eli.line_flux_err
                        l.sigma = eli.fit_sigma
                        l.sigma_err = eli.fit_sigma_err
                        l.line_score = eli.line_score
                        l.raw_line_score = eli.raw_line_score
                        l.prob_noise = eli.prob_noise
                        l.absorber = eli.absorber
                        l.fit_dx0 = eli.fit_dx0
                        l.eli = copy.deepcopy(eli) #get all the emission line info fit data (only a few places use it)

                        per_line_total_score += eli.line_score  # cumulative score for ALL solutions
                        sol.score += eli.line_score  # score for this solution
                        sol.prob_noise *= eli.prob_noise

                        sol.lines.append(l)
                        if l.absorber:
                            line_type = "absorption"
                        else:
                            line_type = "emission"
                        log.info("Accepting %s line (%s): %s(%0.1f at %01.f) snr = %0.1f  MCMC_snr = %0.1f  "
                                 "line_flux = %0.1g  sigma = %0.1f  line_score = %0.1f  p(noise) = %g"
                                 %(line_type, self.identifier,l.name,l.w_rest,l.w_obs,l.snr, eli.mcmc_snr, l.flux,
                                  l.sigma, l.line_score,l.prob_noise))
                else: #is not good
                    if eli is None:
                        log.debug("Line rejected (failed to fit).")
                    else:
                        log.debug("Line rejected (failed is_good).")

                    #however ... the median filter fit might have something, esp. true for broad LyA with strong blue
                    #is there a line found where expected??
                    try:
                        if self.all_found_lines is not None:
                            for fl in self.all_found_lines:
                                if abs(fl.fit_x0-a_central) < NOMINAL_MAX_OFFSET_AA: #and \
                                        #fl.fit_sigma/self.central_eli.fit_sigma < 3:

                                    lineinfo = self.match_line(fl.fit_x0,sol.z,max_rank=3)
                                    if lineinfo is not None:
                                        l = copy.deepcopy(fl)
                                        l.w_rest = lineinfo.w_rest
                                        l.z = sol.z
                                        l.eqw_rest = fl.eqw_obs / (1.0 + sol.z)
                                        l.name = lineinfo.name
                                        l.sigma = l.fit_sigma
                                        l.sigma_err = l.fit_sigma_err
                                        l.w_obs = l.fit_x0
                                        l.flux = l.fit_line_flux
                                        l.flux_err = l.line_flux_err
                                        l.eli = l #self referencing ... has same fields
                                        l.broad = l.broadfit
                                        l.tentative = True #extra property
                                        l.rank = lineinfo.rank

                                        sol.score += l.line_score
                                        sol.lines.append(l)

                                        log.info("Tentative (found line) %s line (%s): %s(%0.1f at %01.f) snr = %0.1f  MCMC_snr = %0.1f  "
                                                 "line_flux = %0.1g  sigma = %0.1f  line_score = %0.1f  p(noise) = %g"
                                                 %("emission", self.identifier,sol.lines[-1].name,sol.lines[-1].w_rest,
                                                   sol.lines[-1].w_obs,sol.lines[-1].snr, -1, sol.lines[-1].flux,
                                                   sol.lines[-1].sigma, sol.lines[-1].line_score,sol.lines[-1].prob_noise))


                    except:
                        pass

            #now apply penalty for unmatched lines?
            try:
                sol.unmatched_lines_count, sol.unmatched_lines_score = self.compute_unmatched_lines_score(sol)

                if sol.unmatched_lines_count > G.MAX_OK_UNMATCHED_LINES and sol.unmatched_lines_score > G.MAX_OK_UNMATCHED_LINES_SCORE:
                    log.info(f"Solution ({sol.name} {sol.central_rest:0.2f} at {sol.central_rest * (1+sol.z)}) penalized for excessive unmatched lines. Old score: {sol.score:0.2f}, "
                             f"Penalty {sol.unmatched_lines_score:0.2f} on {sol.unmatched_lines_count} lines")
                    sol.score -= sol.unmatched_lines_score
            except:
                log.info("Exception adjusting solution score for unmatched lines",exc_info=True)

            if sol.score > 0.0:
                # check if not a solution, has at least one other line

                allow_solution = False

                if (sol.lines is not None) and (len(sol.lines) > 1): #anything with 2 or more lines is "allowed"
                    allow_solution = True
                elif (e.solution): #only 1 line
                    allow_solution = True
                #need fwhm and obs line range
                elif (self.fwhm is not None) and (self.fwhm >= e.min_fwhm): #can be None if no good fit
                    allow_solution = True
                #technically, this condition should not trigger as if we are in the single line range, there IS NO SECOND LINE
                elif (e.min_obs_wave < central < e.max_obs_wave) :
                    #only 1 line and not a 1 line solution, except in certain configurations
                    allow_solution = True
                else:
                    allow_solution = False

                if allow_solution or (known_z is not None):
                    # log.info("Solution p(noise) (%f) from %d additional lines" % (sol.prob_noise, len(sol.lines) - 1))
                    # bonus for each extra line over the minimum

                    # sol.lines does NOT include the main line (just the extra lines)
                    n = len(np.where([l.absorber == False for l in sol.lines])[0])
                    #          n = len(sol.lines) + 1 #+1 for the main line
                    if n >= G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY:
                        bonus = 0.5 * (n ** 2 - n) * G.ADDL_LINE_SCORE_BONUS  # could be negative
                        # print("+++++ %s n(%d) bonus(%g)"  %(self.identifier,n,bonus))
                        sol.score += bonus
                        per_line_total_score += bonus

                    sol.score *= anchor_ew_score_penalty_factor
                    solutions.append(sol)
                else:
                    log.debug("Line (%s, %0.1f) not allowed as single line solution." % (sol.name, sol.central_rest))


        #end for e in emission lines

        #clean up invalid solutions (multiple lines with very different systematic velocity offsets)
        if True:
            for s in solutions:
                if s.emission_line.w_rest == G.LyA_rest: #does NOT apply to LyA, which can have a large velocity offset
                    continue
                all_dx0 = [l.fit_dx0 for l in s.lines]
                all_score = [l.line_score for l in s.lines]
                rescore = False
                #enforce similar all_dx0
                if self.fwhm > G.SPEC_MAX_OFFSET_SPREAD_BROAD_THRESHOLD: #allow extra slop ... the broad lines are hard to center and may have absorption in them
                    max_offset_spread = self.fwhm/2.355 #i.e. about one sigma of the anchor line sigma
                else:
                    max_offset_spread = G.SPEC_MAX_OFFSET_SPREAD
                try:
                    while len(all_dx0) > 1:
                        #todo: no ... throw out the line farthest from the average and recompute ....
                        if (max(all_dx0) - min(all_dx0)) > max_offset_spread: #differ by more than 2 AA
                            #throw out lowest score? or greatest dx0?
                            avg_dx0 = np.median(all_dx0)
                            delta_dx0 = np.array(abs(all_dx0-avg_dx0))
                            all_i = np.argwhere(delta_dx0 == np.max(delta_dx0)).flatten()
                            if len(all_i) > 1:
                                #find lowest score if there are multple lowest scores, it does not matter which
                                i = np.nanargmin(np.array(all_score)[all_i])
                            else:
                                i = all_i[0]

                            #i = np.argmax(abs(all_dx0-avg_dx0))
                            #if s.lines[i].score >
                            log.info("Removing largest offset line from solution %s (%s at %0.1f) due to extremes in fit_dx0 (%f,%f)."
                                     " Line (%s) Score (%f) Offset (%f)"
                                     %(self.identifier,s.emission_line.name,s.central_rest,min(all_dx0),max(all_dx0),
                                       s.lines[i].name, s.lines[i].score, all_dx0[i]))

                            s.rejected_lines.append(copy.deepcopy(s.lines[i]))
                            del all_dx0[i]
                            del all_score[i]
                            del s.lines[i]
                            rescore = True
                        else:
                            break
                except:
                    log.warning("Exception! Unexpected exception: ",exc_info=True)

                if rescore:
                    #remove this solution?
                    old_score = s.score
                    per_line_total_score -= s.score

                    s.calc_score()
                    per_line_total_score += s.score

                    # HAVE to REAPPLY
                    # now apply penalty for unmatched lines?
                    try:
                        s.unmatched_lines_count, s.unmatched_lines_score = self.compute_unmatched_lines_score(s)

                        if s.unmatched_lines_count > G.MAX_OK_UNMATCHED_LINES and s.unmatched_lines_score > G.MAX_OK_UNMATCHED_LINES_SCORE:
                            log.info("Re-apply unmatched lines after rescoring....")
                            log.info(
                                f"Solution ({s.name} {s.central_rest:0.2f} at {s.central_rest * (1 + s.z)}) penalized for excessive unmatched lines. Old score: {s.score:0.2f}, "
                                f"Penalty {s.unmatched_lines_score:0.2f} on {s.unmatched_lines_count} lines")
                            s.score -= s.unmatched_lines_score
                    except:
                        log.info("Exception adjusting solution score for unmatched lines", exc_info=True)


                    log.info("Solution:  %s (%s at %0.1f) rescored due to extremes in fit_dx0. Old score (%f) New Score (%f)"
                             %(self.identifier,s.emission_line.name,s.central_rest,old_score,s.score))

                #check for solution lines that agree with elixer found lines
                for l in s.lines:
                    elixer_wave, elixer_score = self.is_near_a_peak(l.w_obs,aa=2.0,return_score=True)
                    if elixer_wave > 0:
                        s.score += GOOD_MIN_LINE_SCORE #min(elixer_score,10.0) #the score is at least +3.0 (GOOD_MIN_LINE_SCORE)

        #todo: check line ratios, AGN consistency, etc here?
        #at least 3 func calls
        #consistent_with_AGN (returns some scoring that is only used to boost AGN consistent solutions)
        #consistent_with_oii_galaxy()
        #consistent_with_star
        #??consistent with meteor #this may need to be elsewhere and probably involves check individual exposures
        #                          #since it would only show up in one exposure

        if G.MULTILINE_USE_CONSISTENCY_CHECKS:# and (self.central_eli is not None):
            if self.central_eli is None:
                #could  not fit the central line, so no solution is valid if it relies on the central
                #(can still be valid if there are mulitple other lines)
                central_eli = EmissionLineInfo()
                central_eli.line_score = 0.0
                central_eli.raw_line_score = 0.0
            else:
                central_eli = self.central_eli

            for s in solutions:
                if (central_eli.line_score == 0):
                    if (s is not None) and (s.lines is not None) and (len(s.lines) > 1):
                        pass #still okay there are 2+ other lines
                    else:
                        log.info(f"Solution {s.name} rejected. No central fit and few lines. Zeroing score.")
                        s.score = 0.0
                        s.scale_score = 0.0
                        s.frac_score = 0.0
                        continue

                # the pair of lines being checked are 4959 and 5007 (or the solution contains those pair of lines)
                # or with OII or H_beta
                try:
                    if (np.isclose(s.central_rest,G.OIII_4959,atol=1.0) or np.isclose(s.central_rest,G.OIII_5007,atol=1.0)) and \
                        ( np.any( [(np.isclose(x.w_rest,G.OIII_4959,atol=1.0) or np.isclose(x.w_rest,G.OIII_5007,atol=1.0)  or
                                  np.isclose(x.w_rest,G.OII_rest,atol=1.0) or np.isclose(x.w_rest,G.Hbeta_4861,atol=1.0))
                                  and abs(x.fit_dx0) < 2.0 for x in s.lines])  ):
                        #we've got 5007 or 4959 with the other mate OR H_beta or OII
                        #but still enforce flux 5007/4959 of 3
                        if SU.check_oiii(s.z,values,errors,wavelengths,delta=1,cont=self.estcont,cont_err=self.estcont_unc) == 1:
                            oiii_lines = True
                        else:
                            #note: IF the line positions still line up, this can still get classified as OIII
                            #by the rest of the multiline solution (including any score augmentation by
                            #consistency check with lzg)
                            oiii_lines = False
                    else:
                        oiii_lines = False
                except:
                    oiii_lines = False

                # even if weak, go ahead and check for inconsistency (ie. if 4959 present but 5007 is not, then that
                # solution does not make sense), but only allow a positive boost to the scoring IF the base solution
                # is not weak or IF this is a possible OIII 4958+5007 combination (which is well constrained)
                if s.score < G.MULTILINE_MIN_SOLUTION_SCORE and not oiii_lines:
                    no_plus_boost = True
                else:
                    no_plus_boost = False

                #todo: iterate over all types of objects
                #if there is no consistency (that is, the lines don't match up) you get no change
                #if there is anti-consistency (the lines match up but are inconsistent by ratio, you can get a score decrease)


                #LAE (general)
                lae_boost = 1.0 #want to keep it for below and AGN
                if 1.88 < s.z < 3.53: # and s.emission_line.w_rest != G.LyA_rest:
                    lae_boost = self.scale_consistency_score_to_solution_score_factor(
                                    self.solution_consistent_with_lae(s,central_eli,continuum_limit,continuum_limit_err))

                    if lae_boost != 1.0:
                        log.info(f"Solution: {s.name} score {s.score} to be modified by x{lae_boost} for consistency with LAE")

                        per_line_total_score -= s.score
                        s.score = lae_boost * s.score
                        per_line_total_score += s.score


                #to be consistent with AGN, if  1.9 < z < 3.5 it MUST be also consistent with LAE
                #AGN
                if 1.88 < s.z < 3.53: # and s.emission_line.w_rest != G.LyA_rest:
                    if (lae_boost > 1.0):
                        boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_agn(s))
                    else:
                        boost = 1.0 #don't bother checking for AGN since not positively consistent with LAE and in our z range
                else: #not in s range or is already LyA, so do normal check
                    boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_agn(s))

                if boost != 1.0:
                    log.info(f"Solution: {s.name} score {s.score} to be modified by x{boost} for consistency with AGN")

                    #for the labeling, need to check vs the TOTAL score (so include the primary line)
                    if ( (s.score + central_eli.line_score) > G.MULTILINE_FULL_SOLUTION_SCORE) and (boost > 1.0) and \
                            (no_plus_boost is False):
                        #check BEFORE the boost
                        #however, only apply the label if at least one line is broad
                        line_fwhm = np.array([central_eli.fit_sigma*2.355] + [l.sigma * 2.355 for l in s.lines])
                        line_fwhm_err = np.array([central_eli.fit_sigma_err*2.355] + [l.sigma_err * 2.355 for l in s.lines])
                        if max(line_fwhm+line_fwhm_err) > 14.0 and s.emission_line.w_rest != G.MgII_2799:
                            self.add_classification_label("agn")
                        else:
                            log.info(f"Solution: {s.name} 'agn' label omitted, but boost applied.")

                    per_line_total_score -= s.score
                    s.score = boost * s.score
                    per_line_total_score += s.score


                # low-z galaxy
                boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_low_z(s))

                if boost != 1.0:
                    log.info(f"Solution: {s.name} score {s.score} to be modified by x{boost} for consistency with low-z galaxy")

                    if ((s.score + central_eli.line_score) > G.MULTILINE_FULL_SOLUTION_SCORE) and \
                            (boost > 1.0) and (no_plus_boost is False):  # check BEFORE the boost
                        self.add_classification_label("lzg") #Low-z Galaxy

                    per_line_total_score -= s.score
                    s.score =  boost * s.score
                    #!!! do not impose a boost minium limit on the check for oiii_lines ... you CAN get OIII 5007 and
                    #not a significant OII line, or HBeta, so the boost criteria may be << 1 and that is okay
                    if s.score < G.MULTILINE_MIN_SOLUTION_SCORE and oiii_lines:# and boost > 0.2:

                        if s.rejected_lines is not None and \
                            np.any(np.intersect1d([G.OIII_4959,G.OIII_5007], [rl.w_rest for rl in s.rejected_lines])):
                            #predicated on both OIII lines being present, but at least one was rejected
                            pass #no boost
                        else:
                            log.info(f"Solution: {s.name} score {s.score} raised to minimum {G.MULTILINE_MIN_SOLUTION_SCORE} for 4959+5007")
                            s.score = G.MULTILINE_MIN_SOLUTION_SCORE
                            if boost > 0:
                                s.prob_noise = min(s.prob_noise,0.5/boost)

                    per_line_total_score += s.score
                elif ((s.central_rest == G.OIII_5007) or (s.central_rest == G.OIII_4959)) and oiii_lines:

                    if s.score < G.MULTILINE_MIN_SOLUTION_SCORE:# and boost > 0.2:
                        if s.rejected_lines is not None and \
                                np.any(np.intersect1d([G.OIII_4959,G.OIII_5007], [rl.w_rest for rl in s.rejected_lines])):
                            #predicated on both OIII lines being present, but at least one was rejected
                            s.possible_pn += 1
                            pass #no boost
                        else:
                            log.info(f"Solution: {s.name} score {s.score} raised to minimum {G.MULTILINE_MIN_SOLUTION_SCORE} for including possible OIII")
                            s.score = G.MULTILINE_MIN_SOLUTION_SCORE
                            s.possible_pn += 2 #possible planetary nebula .. still need more info (like lack of a discrete SEP source)
                            if boost > 0:
                                s.prob_noise = min(s.prob_noise,0.5)
                    else:
                        s.possible_pn += 3 #possible planetary nebula .. still need more info (like lack of a discrete SEP source)

                    per_line_total_score += s.score

                #H&K a low-z galaxy or star
                if G.CONTINUUM_RULES:
                    boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_H_and_K(s))
                    if boost != 1.0:
                        log.info(f"Solution: {s.name} score {s.score} to be modified by x{boost} for consistency with H & K lines")

                        # if ((s.score + central_eli.line_score) > G.MULTILINE_FULL_SOLUTION_SCORE) and \
                        #         (boost > 1.0) and (no_plus_boost is False):  # check BEFORE the boost
                        #     self.add_classification_label("lzg") #Low-z Galaxy

                        per_line_total_score -= s.score
                        s.score =  boost * s.score
                        per_line_total_score += s.score


                    #note: this applies to the detection only, so really only need this once
                    if not self.wd_checked:
                        check, wd_z, suggested_label = self.detection_consistent_with_wd()
                        #boost = self.scale_consistency_score_to_solution_score_factor(self.detection_consistent_with_wd())
                        if check > 0.0:
                            self.wd_checked = 1
                            self.wd_z = wd_z
                            self.add_classification_label(suggested_label) #("WD")
                        else:
                            self.wd_checked = -1

                    if self.wd_checked > 0:
                        if abs(s.z - self.wd_z) < 0.001: #if this solution happens to be consistent with the z, then boost it
                            log.info(f"Solution: {s.name} score {s.score} to be modified by x{boost} for consistency with WD")

                            per_line_total_score -= s.score
                            s.score =  boost * s.score
                            per_line_total_score += s.score

        else: #still check for invalid solutions (no valid central emission line)
            if self.central_eli is None:
                for s in solutions:
                    if (s is not None) and (s.lines is not None) and (len(s.lines) > 1):
                        pass #still okay there are 2+ other lines
                    else:
                        log.info(f"Solution {s.name} rejected. No central fit and few lines. Zeroing score.")
                        s.score = 0.0
                        s.scale_score = 0.0
                        s.frac_score = 0.0
                        continue
        #remove and zeroed scores
        try:
            if solutions is not None and len(solutions)>0:
                for i in range(len(solutions)-1,-1,-1):
                    if solutions[i].score <= 0:
                        del solutions[i]
                    elif solutions[i].emission_line.rank > 3 and min([x.rank for x in solutions[i].lines]) > 3:
                        del solutions[i]
        except:
            log.debug("Exception clearing solutions",exc_info=True)


        #also check here, in case there were no other solutions
        if not self.wd_checked:
            check, wd_z, suggested_label = self.detection_consistent_with_wd()
            #boost = self.scale_consistency_score_to_solution_score_factor(self.detection_consistent_with_wd())
            if check > 0.0:
                self.wd_checked = 1
                self.wd_z = wd_z
                self.add_classification_label(suggested_label)#("WD")

                #add a z=0 solution
                try:
                    sol = Classifier_Solution()
                    sol.z = 0
                    sol.central_rest = self.central_eli.fit_x0

                    #find the line (fl.fit_x0,sol.z,max_rank=3)
                    # lineinfo = self.match_line(fl.fit_x0,sol.z,max_rank=3)
                     #               if lineinfo is not None:
                    lineinfo = self.match_line(sol.central_rest,0,max_rank=99,allow_absorption=True)
                    if lineinfo is not None:
                        #special case NeIII pops up as a higher rank than H_epsilon as that is based on emission
                        #but if this is a WD, then NeIII 3967 is really H_epsilon 3970AA
                        if lineinfo.w_rest == G.NeIII_3967 or lineinfo.w_rest == G.CaII_H_3968:
                            try:
                                h_ep_sel = np.array([x.w_rest == G.Hepsilon_3970 for x in self.emission_lines])
                                lineinfo = np.array(self.emission_lines)[h_ep_sel][0]
                                #if this fails, just let it go and revert to the earlier match
                            except:
                                log.debug("Minor exception.",exc_info=True)
                        sol.name = lineinfo.name
                        sol.color = lineinfo.color
                        sol.emission_line = copy.deepcopy(lineinfo)

                    sol.emission_line.w_obs = self.central_eli.fit_x0
                    sol.emission_line.solution = True
                    sol.emission_line.z = 0.0
                    sol.prob_noise = 0.0
                    sol.emission_line.absorber = self.central_eli.absorber
                    #need to add lines
                    for l in self.all_found_absorbs:
                        lineinfo = self.match_line(l.fit_x0, 0)
                        if lineinfo is not None:
                            sol.lines.append(lineinfo) #just to have them

                    sol.score = G.MULTILINE_FULL_SOLUTION_SCORE + 0.5 * len(sol.lines) * G.MULTILINE_FULL_SOLUTION_SCORE
                    solutions.append(sol)
                except:
                    pass

            else:
                self.wd_checked = -1


        per_solution_total_score = np.nansum([s.score for s in solutions])

        for s in solutions:
            s.frac_score = s.score/per_solution_total_score
            s.scale_score = s.prob_real * G.MULTILINE_WEIGHT_PROB_REAL + \
                          min(1.0, s.score / G.MULTILINE_FULL_SOLUTION_SCORE) *  G.MULTILINE_WEIGHT_SOLUTION_SCORE + \
                          s.frac_score * G.MULTILINE_WEIGHT_FRAC_SCORE

        #sort by score
        solutions.sort(key=lambda x: x.scale_score, reverse=True)

        #check for display vs non-display (aka primary emission line solution)
        if len(solutions) > 1:
            if (solutions[0].frac_score / solutions[1].frac_score) < 2.0:
                if (solutions[0].emission_line.display is False) and (solutions[1].emission_line.display is True):
                    #flip them
                    log.debug("Flipping top solutions to favor display line over non-display line")
                    temp_sol = solutions[0]
                    solutions[0] = solutions[1]
                    solutions[1] = temp_sol

        for s in solutions:
            ll =""
            for l in s.lines:
                ll += " %s(%0.1f at %0.1f)," %(l.name,l.w_rest,l.w_obs)
            msg = "Possible Solution %s (%0.3f): %s (%0.1f at %0.1f), Frac = %0.2f, Score = %0.1f (%0.3f), z = %0.5f, +lines=%d %s"\
                    % (self.identifier, s.prob_real,s.emission_line.name,s.central_rest,s.central_rest*(1.0+s.z), s.frac_score,
                       s.score,s.scale_score,s.z, len(s.lines),ll )
            log.info(msg)
            #
            if G.DEBUG_SHOW_GAUSS_PLOTS:
                print(msg)

        return solutions


    def consistency_checks(self,solution):
        """
        Exectute the consistency checks, but do not alter the score.
        Essentially this is a boolean conditionas to whether the proposed solution (generally from an outside catalog)
         FAILS our own internal checks. If it does not FAIL, then it can be accepted.

        This is largely intended to be a redshift consistentcy check (so will check vs low-z galaxy, for example,
        where line ratios, etc come into play)
        :param solution:
        :return:
        """

        try:
            if not G.MULTILINE_USE_CONSISTENCY_CHECKS:
                return True

            s = solution #just for consistency
            z = s.z
            consistent = True
            lowz_boost = 1.0
            hk_boost  = 1.0
            oiii_boost = 1.0
            lae_boost = 1.0

            boost_list = []



            #not check vs AGN ... this is meant to be a redshift consistency check
            #AGN
            # boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_agn(s))
            #
            # #using the boost to decide if it is consistent (boost > 1) or inconsistent (boost < 1)
            # # if boost == 1, no decision was made and nothing changes
            # if boost < 1.0:
            #     consistent = False
            # elif boost > 1.0:
            #     consistent = True
            #
            # low-z galaxy
            if z < 1.0:
                try:
                    if (np.isclose(s.central_rest,G.OIII_4959,atol=1.0) or np.isclose(s.central_rest,G.OIII_5007,atol=1.0)) and \
                            ( np.any( [(np.isclose(x.fit_x0/(1+z),G.OIII_4959,atol=1.0) or np.isclose(x.fit_x0/(1+z),G.OIII_5007,atol=1.0)  or
                                        np.isclose(x.fit_x0/(1+z),G.OII_rest,atol=1.0) or np.isclose(x.fit_x0/(1+z),G.Hbeta_4861,atol=1.0))
                                       and abs(x.fit_dx0) < 2.0 for x in self.all_found_lines])  ):
                        #we've got 5007 or 4959 with the other mate OR H_beta or OII
                        #but still enforce flux 5007/4959 of 3
                        if SU.check_oiii(s.z,self.values,self.errors,self.wavelengths,delta=1,cont=self.estcont,cont_err=self.estcont_unc) == 1:
                            oiii_boost = 2.0 #i.e. possibly consistent with oiii
                            boost_list.append(oiii_boost)
                except:
                    oiii_lines = 1.0

                lowz_boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_low_z(s))
                boost_list.append(lowz_boost)

                #H&K a low-z galaxy or star
                if G.CONTINUUM_RULES or solution.emission_line.absorber:
                    hk_boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_H_and_K(s))
                    if (solution.emission_line.w_rest in [G.CaII_K_3934,G.CaII_H_3968]) and (hk_boost <= 1.0) : #explicitly H or K
                        #the line would have to be H or K, but we cannot fit H&k to it
                        boost_list.append(0)
                    else:
                        boost_list.append(hk_boost)

            elif 1.89 < z < 3.52: #check for LAE
                lae_boost = self.scale_consistency_score_to_solution_score_factor(self.solution_consistent_with_lae(s))
                boost_list.append(lae_boost)

            #combine the "boost" values
            #basically, if any are less than 1.0, we reject UNLESS at least one is > 1.0
            if np.any([b > 1.0 for b in boost_list]):
                return True
            elif np.any([b < 1.0 for b in boost_list]):
                return False
            else:
                return True


        except:
            log.warning("Exception! Spectrum::consistenty_checks() fail.",exc_info=True)
            return True # since no determination can be made, given the intent of this function, return a True
                        #so the solution can be accepted



    def get_bayes_probabilities(self,addl_wavelengths=None,addl_fluxes=None,addl_errors=None):
        # todo: feed in addl_fluxes from the additonal line solutions (or build specifically)?
        # todo: make use of errors

        #care only about the LAE and OII solutions:
        #todo: find the LyA and OII options in the solution list and use to fill in addl_fluxes?

        # self.p_lae_oii_ratio, self.p_lae, self.p_oii, plae_errors = line_prob.prob_LAE(wl_obs=self.central,
        #                                                    lineFlux=self.estflux,
        #                                                    lineFlux_err=self.estflux_unc,
        #                                                    ew_obs=self.eqw_obs,
        #                                                    ew_obs_err=self.eqw_obs_unc,
        #                                                    c_obs=None, which_color=None,
        #                                                    addl_wavelengths=addl_wavelengths,
        #                                                    addl_fluxes=addl_fluxes,
        #                                                    addl_errors=addl_errors,
        #                                                    sky_area=None,
        #                                                    cosmo=None, lae_priors=None,
        #                                                    ew_case=None, W_0=None,
        #                                                    z_OII=None, sigma=None, estimate_error=True)


        self.p_lae_oii_ratio, self.p_lae, self.p_oii, plae_errors = line_prob.mc_prob_LAE(wl_obs=self.central,
                                                           lineFlux=self.estflux,
                                                           lineFlux_err=self.estflux_unc,
                                                           continuum=self.estcont,
                                                           continuum_err=self.estcont_unc,
                                                           c_obs=None, which_color=None,
                                                           addl_wavelengths=addl_wavelengths,
                                                           addl_fluxes=addl_fluxes,
                                                           addl_errors=addl_errors,
                                                           sky_area=None,
                                                           cosmo=None, lae_priors=None,
                                                           ew_case=None, W_0=None,
                                                           z_OII=None, sigma=None)

        try:
            if plae_errors:
                self.p_lae_oii_ratio_range = plae_errors['ratio']
        except:
            pass
        # if (self.p_lae is not None) and (self.p_lae > 0.0):
        #     if (self.p_oii is not None) and (self.p_oii > 0.0):
        #         self.p_lae_oii_ratio = self.p_lae /self.p_oii
        #     else:
        #         self.p_lae_oii_ratio = float('inf')
        # else:
        #     self.p_lae_oii_ratio = 0.0
        #
        # self.p_lae_oii_ratio = min(line_prob.MAX_PLAE_POII,self.p_lae_oii_ratio) #cap to MAX

    def build_full_width_spectrum(self,wavelengths = None,  counts = None, errors = None, central_wavelength = None,
                                  values_units = 0, show_skylines=True, show_peaks = True, name=None,
                                  dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0,peaks=None,annotate=True,
                                  figure=None,show_line_names=True):


        use_internal = False
        if (counts is None) or (wavelengths is None) or (central_wavelength is None):
            counts = self.values
            #if self.values_are_flux: #!!! need a copy here ... DO NOT counts *= 10.0
            #    counts = counts * 10.0 #flux assumed to be cgs x10^-17 ... by 10x to x10^-18 become very similar to counts in scale

            counts, values_units = norm_values(counts,self.values_units)
            wavelengths = self.wavelengths
            central_wavelength = self.central
            use_internal = True

        if len(counts)==0:
            #not empty but still wrong
            log.error("Spectrum::build_full_width_spectrum. No spectrum to plot.")
            return None

        # fig = plt.figure(figsize=(5, 6.25), frameon=False)
        if figure is None:
            fig = plt.figure(figsize=(G.ANNULUS_FIGURE_SZ_X, 2), frameon=False)
            plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
        else:
            fig = figure


        if show_line_names:
            dy = 1.0 / 5.0  # + 1 skip for legend, + 2 for double height spectra + 2 for double height labels
            specplot = plt.axes([0.05, 0.20, 0.90, 0.40])
        else:
            dy = 0.0
            specplot = plt.axes([0.05, 0.1, 0.9, 0.8]) #left,bottom,width, height in fraction of 1.0

        # this is the 1D averaged spectrum
        #textplot = plt.axes([0.025, .6, 0.95, dy * 2])
        #specplot = plt.axes([0.05, 0.20, 0.90, 0.40])
        #specplot = plt.axes([0.025, 0.20, 0.95, 0.40])

        # they should all be the same length
        # yes, want round and int ... so we get nearest pixel inside the range)
        left = wavelengths[0]
        right = wavelengths[-1]

        try:
            mn = np.min(counts)
            mn = max(mn, -20)  # negative flux makes no sense (excepting for some noise error)
            mx = np.max(counts)
            ran = mx - mn
            specplot.plot(wavelengths, counts,lw=0.5,c='b')

            specplot.axis([left, right, mn - ran / 20, mx + ran / 20])
            yl, yh = specplot.get_ylim()

            specplot.locator_params(axis='y', tight=True, nbins=4)


            if show_peaks:
                #emistab.append((pi, px, pv,pix_width,centroid))
                if peaks is None:
                    if (self.all_found_lines is not None):
                        peaks = self.all_found_lines
                    else:
                        peaks = peakdet(wavelengths,counts,errors, dw,h,dh,zero,values_units=values_units,
                                        spec_obj=self) #as of 2018-06-11 these are EmissionLineInfo objects
                        self.all_found_lines = peaks
                        if G.DISPLAY_ABSORPTION_LINES or G.MAX_SCORE_ABSORPTION_LINES:
                            self.all_found_absorbs = peakdet(wavelengths, counts, errors,
                                                             values_units=values_units,absorber=True, spec_obj=self)
                            self.clean_absorbers()


                #scores = []
                #for p in peaks:
                #    scores.append(signal_score(wavelengths, counts, p[1]))

                #for i in range(len(scores)):
                #    print(peaks[i][0],peaks[i][1], peaks[i][2], peaks[i][3], peaks[i][4], scores[i])

                if (peaks is not None) and (len(peaks) > 0):
                    # specplot.scatter(np.array(peaks)[:, 1], np.array(peaks)[:, 2], facecolors='none', edgecolors='r',
                    #                  zorder=99)
                    #
                    # for i in range(len(peaks)):
                    #     h = peaks[i][2]
                    #     specplot.annotate("%0.1f"%peaks[i][5],xy=(peaks[i][1],h),xytext=(peaks[i][1],h),fontsize=6,zorder=99)
                    #
                    #     log.debug("Peak at: %g , Score = %g , SNR = %g" %(peaks[i][1],peaks[i][5], peaks[i][6]))

                    #                 0   1   2  3           4            5          6
                    #emistab.append((pi, px, pv, pix_width, centroid_pos,eli.eqw_obs,eli.snr))

                    x = [p.raw_x0 for p in peaks]
                    y = [p.raw_h for p in peaks] #np.array(peaks)[:, 2]

                    specplot.scatter(x, y, facecolors='none', edgecolors='r',zorder=99)

                    if annotate:
                        for i in range(len(peaks)):
                            h = peaks[i].raw_h
                            specplot.annotate("%0.1f"%peaks[i].eqw_obs,xy=(peaks[i].fit_x0,h),xytext=(peaks[i].fit_x0,h),
                                              fontsize=6,zorder=99)

                            log.debug("Peak at %g , Score = %g , SNR = %g" %(peaks[i].fit_x0,peaks[i].eqw_obs, peaks[i].snr))


            #textplot = plt.axes([0.025, .6, 0.95, dy * 2])
            textplot = plt.axes([0.05, .6, 0.90, dy * 2])
            textplot.set_xticks([])
            textplot.set_yticks([])
            textplot.axis(specplot.axis())
            textplot.axis('off')

            if central_wavelength > 0:
                wavemin = specplot.axis()[0]
                wavemax = specplot.axis()[1]
                legend = []
                name_waves = []
                obs_waves = []

                rec = plt.Rectangle((central_wavelength - 20.0, yl), 2 * 20.0, yh - yl, fill=True, lw=0.5, color='y', zorder=1)
                specplot.add_patch(rec)

                if show_line_names:

                    if use_internal and (len(self.solutions) > 0):

                        e = self.solutions[0].emission_line

                        z = self.solutions[0].z

                        #plot the central (main) line
                        y_pos = textplot.axis()[2]
                        textplot.text(e.w_obs, y_pos, e.name + " {", rotation=-90, ha='center', va='bottom',
                                      fontsize=12, color=e.color)  # use the e color for this family

                        #plot the additional lines
                        for f in self.solutions[0].lines:
                            if f.score > 0:
                                y_pos = textplot.axis()[2]
                                textplot.text(f.w_obs, y_pos, f.name + " {", rotation=-90, ha='center', va='bottom',
                                              fontsize=12, color=e.color)  # use the e color for this family


                        #todo: show the fractional score?
                        #todo: show the next highest possibility?
                        legend.append(mpatches.Patch(color=e.color,
                                                     label="%s, z=%0.5f, Score = %0.1f (%0.2f)" %(e.name,self.solutions[0].z,
                                                                                self.solutions[0].score,
                                                                                self.solutions[0].frac_score)))
                        name_waves.append(e.name)


                    else:
                        for e in self.emission_lines:
                            if not e.solution:
                                continue

                            z = central_wavelength / e.w_rest - 1.0

                            if (z < 0):
                                continue

                            count = 0
                            for f in self.emission_lines:
                                if (f == e) or not (wavemin <= f.redshift(z) <= wavemax):
                                    continue

                                count += 1
                                y_pos = textplot.axis()[2]
                                for w in obs_waves:
                                    if abs(f.w_obs - w) < 20:  # too close, shift one vertically
                                        y_pos = (textplot.axis()[3] - textplot.axis()[2]) / 2.0 + textplot.axis()[2]
                                        break

                                obs_waves.append(f.w_obs)
                                textplot.text(f.w_obs, y_pos, f.name + " {", rotation=-90, ha='center', va='bottom',
                                              fontsize=12, color=e.color)  # use the e color for this family

                            if (count > 0) and not (e.name in name_waves):
                                legend.append(mpatches.Patch(color=e.color, label=e.name))
                                name_waves.append(e.name)

                    # make a legend ... this won't work as is ... need multiple colors
                    skipplot = plt.axes([.025,0.0, 0.95, dy])
                    skipplot.set_xticks([])
                    skipplot.set_yticks([])
                    skipplot.axis(specplot.axis())
                    skipplot.axis('off')
                    skipplot.legend(handles=legend, loc='center', ncol=len(legend), frameon=False,
                                    fontsize='small', borderaxespad=0)

        except:
            log.warning("Unable to build full width spec plot.", exc_info=True)

        if show_skylines:
            try:
                yl, yh = specplot.get_ylim()
                # see https://www.eso.org/observing/dfo/quality/UVES/txt/sky/gident_346.dat
                central_w = 3545
                half_width = 10
                rec = plt.Rectangle((central_w - half_width, yl), 2 * half_width, yh - yl, fill=True, lw=1,
                                    color='gray', alpha=0.5, zorder=1)
                specplot.add_patch(rec)

                central_w = 5462
                half_width = 5
                rec = plt.Rectangle((central_w - half_width, yl), 2 * half_width, yh - yl, fill=True, lw=1,
                                    color='gray', alpha=0.5, zorder=1)
                specplot.add_patch(rec)
            except:
                log.warning("Unable add skylines.", exc_info=True)

        if name is not None:
            try:
                #plt.tight_layout(w_pad=1.1)
                plt.savefig(name+".png", format='png', dpi=300)
            except:
                log.warning("Unable save plot to file.", exc_info=True)


        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf


