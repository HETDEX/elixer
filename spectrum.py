import global_config as G
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
#import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import io
#from scipy.stats import gmean
#from scipy import signal
#from scipy.integrate import simps
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
import copy
import line_prob


#log = G.logging.getLogger('spectrum_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('spectrum_logger')
log.setlevel(G.logging.DEBUG)

#these are for the older peak finder (based on direction change)
MIN_FWHM = 2 #AA (must xlat to pixels)
MIN_ELI_SNR = 2.0 #bare minium SNR to even remotely consider a signal as real
MIN_HEIGHT = 10
MIN_DELTA_HEIGHT = 2 #to be a peak, must be at least this high above next adjacent point to the left
DEFAULT_BACKGROUND = 6.0
DEFAULT_BACKGROUND_WIDTH = 100.0 #pixels
DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND = 10.0 #pixels


GAUSS_FIT_MAX_SIGMA = 10.0 #maximum width (pixels) for fit gaussian to signal (greater than this, is really not a fit)
GAUSS_FIT_MIN_SIGMA = 1.0 #roughly 1/2 pixel where pixel = 1.9AA (#note: "GOOD_MIN_SIGMA" below provides post
                          # check and is more strict) ... allowed to fit a more narrow sigma, but will be rejected later
                          # as not a good signal .... should these actually be the same??
GAUSS_FIT_AA_RANGE = 40.0 #AA to either side of the line center to include in the gaussian fit attempt
                          #a bit of an art here; too wide and the general noise and continuum kills the fit (too wide)
                          #too narrow and fit hard to find if the line center is off by more than about 2 AA
                          #40 seems (based on testing) to be about right (50 leaves too many clear, but weak signals behind)
GAUSS_FIT_PIX_ERROR = 2.0 #error (freedom) in pixels: have to allow at least 2 pixels of error
                          # (if aa/pix is small, _AA_ERROR takes over)
GAUSS_FIT_AA_ERROR = 1.0 #error (freedom) in line center in AA, still considered to be okay
GAUSS_SNR_SIGMA = 3.0 #check at least these pixels (pix*sigma) to either side of the fit line for SNR
                      # (larger of this or GAUSS_SNR_NUM_AA)
GAUSS_SNR_NUM_AA = 5.0 #check at least 4 AA to either side (9 total) of the fit line for SNR in gaussian fit
                       # (larger of this or GAUSS_SNR_SIGMA

#beyond an okay fit (see GAUSS_FIT_xxx above) is this a "good" signal
GOOD_FULL_SNR = 9.0 #ignore SBR is SNR is above this
GOOD_MIN_SNR = 5.0 #bare-minimum; if you change the SNR ranges just above, this will also need to change
GOOD_MIN_SBR = 3.0 #signal to "background" noise (looks at peak height vs surrounding peaks) (only for "weak" signals0
GOOD_MIN_SIGMA = 1.4
GOOD_MIN_EW_OBS = 1.5 #not sure this is a good choice ... really should depend on the physics of the line and
                      # not be absolute
GOOD_MIN_EW_REST = 1.0 #ditto here
GOOD_MIN_LINE_FLUX = 5.0e-18 #todo: this should be the HETDEX flux limit (but that depends on exposure time and wavelength)
GOOD_MAX_DX0 = 3.8 #maximum error (domain freedom) in fitting to line center in AA
                    #since this is based on the fit of the extra line AND the line center error of the central line
                    #this is a compound error (assume +/- 2 AA since ~ 1.9AA/pix for each so at most 4 AA here)?
GOOD_MIN_H_CONT_RATIO = 1.33 #height of the peak must be at least 33% above the continuum fit level
#todo: impose line ratios?
#todo:  that is, if line_x is assumed and line_y is assumed, can only be valid if line_x/line_y ~ ratio??
#todo:  i.e. [OIII(5007)] / [OIII(4959)] ~ 3.0 (2.993 +/- 0.014 ... at least in AGN)


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

#!!!!!!!!!! Note. all widths (like dw, xw, etc are in pixel space, so if we are not using
#!!!!!!!!!!       1 pixel = 1 Angstrom, be sure to adjust



def norm_values(values,values_units):
    '''
    Basically, make spectra values either counts or cgs x10^-18 (whose magnitdues are pretty close to counts) and the
    old logic and parameters can stay the same
    :param values:
    :param values_units:
    :return:
    '''
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
    idx = (np.abs(array-value)).argmin()
    return idx


def gaussian(x,x0,sigma,a=1.0,y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None
    #return a * np.exp(-np.power((x - x0) / sigma, 2.) / 2.)
    #return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.))  + y

    #have the / np.sqrt(...) part so the basic shape is normalized to 1 ... that way the 'a' becomes the area
    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y



def rms(data, fit,cw_pix=None,hw_pix=None,norm=True):
    """

    :param data: (raw) data
    :param fit:  fitted data (on the same scale)
    :param cw_pix: (nearest) pixel (index) of the central peak
    :param hw_pix: half-width (in pixels from the cw_pix) overwhich to calculate rmse (i.e. cw_pix +/- hw_pix)
    :param norm: T/F whether or not to divide by the peak of the raw data
    :return:
    """
    #sanity check
    if (data is None) or (fit is None) or (len(data) != len(fit)) or any(np.isnan(data)) or any(np.isnan(fit)):
        return -999

    if norm:
        mx = max(data)
        if mx < 0:
            return -999
    else:
        mx = 1.0

    d = np.array(data)/mx
    f = np.array(fit)/mx

    if ((cw_pix is not None) and (hw_pix is not None)):
        left = cw_pix - hw_pix
        right = cw_pix + hw_pix

        if (left < 0) or (right > len(data)):
            log.error("Invalid range supplied for rms. Data len = %d. Central Idx = %d , Half-width= %d"
                      % (len(data),cw_pix,hw_pix))
            return -999

        d = d[cw_pix-hw_pix:cw_pix+hw_pix+1]
        f = f[cw_pix-hw_pix:cw_pix+hw_pix+1]

    return np.sqrt(((f - d) ** 2).mean())


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

        self.fit_a = None #expected in counts or in x10^-18 cgs [notice!!! -18 not -17]
        self.fit_x0 = None #central peak (x) position in AA
        self.fit_dx0 = None #difference in fit_x0 and the target wavelength in AA, like bias: target-fit
        self.fit_sigma = 0.0
        self.fit_y = None #y offset for the fit (essentially, the continuum estimate)
        self.fit_h = None #max of the fit (the peak) #relative height
        self.fit_rh = None #fraction of fit height / raw peak height
        self.fit_rmse = -999
        self.fit_norm_rmse = -999

        self.fit_wave = []
        self.fit_vals = []

        self.pix_size = None

        #!! Important: raw_wave, etc is NOT of the same scale or length of fit_wave, etc
        self.raw_wave = []
        self.raw_vals = []
        self.raw_h =  None
        self.raw_x0 = None

        self.line_flux = -999 #the line fluux
        self.cont = -999

        self.snr = 0.0
        self.sbr = 0.0
        self.eqw_obs = -999
        self.cont = -999
        self.fwhm = -999
        self.score = None
        self.raw_score = None

    def build(self,values_units=0):
        if self.snr > MIN_ELI_SNR:
            if self.fit_sigma is not None:
                self.fwhm = 2.355 * self.fit_sigma  # e.g. 2*sqrt(2*ln(2))* sigma

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

                    #if self.fit_a > 1.0: #assume x0^-18 units (since that is what is fit ... similar to counts)
                    #    unit = 1e-18
                    #else:
                    #    unit = 1.0

                    self.line_flux = self.fit_a * unit
                    # !! remember, fit_a is an area and thus has two of 10.0 in it (h*w ... each with x10)
                    # so if in e-17 or e-18 units, need to remove the other 10.0
                    if unit == 1.0e-18:
                        self.line_flux /= 10.0
                    self.cont = self.fit_y * unit

                    #fix fit_h
                    if (self.fit_h > 1.0) and (values_units < 0):
                        self.fit_h *= unit

                else:
                    if (self.fit_a is not None):
                        #todo: HERE ... do not attempt to convert if this is already FLUX !!!
                        #todo: AND ... need to know units of flux (if passed from signal_score are in x10^-18 not -17
                        self.line_flux = self.fit_a * flux_conversion(self.fit_x0)  # cgs units

                    if (self.fit_y is not None) and (self.fit_y > G.CONTINUUM_FLOOR_COUNTS):
                        self.cont = self.fit_y * flux_conversion(self.fit_x0)
                    else:
                        self.cont = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(self.fit_x0)

            if self.line_flux and self.cont:
                self.eqw_obs = self.line_flux / self.cont
        else:
            self.fwhm = -999
            self.cont = -999
            self.line_flux = -999

    def is_good(self,z=0.0):
        #(self.score > 0) and  #until score can be recalibrated, don't use it here
        #(self.sbr > 1.0) #not working the way I want. don't use it
        result = False
        if ((self.snr > GOOD_FULL_SNR) or ((self.snr > GOOD_MIN_SNR) and (self.sbr > GOOD_MIN_SBR))) and \
           (self.fit_sigma > GOOD_MIN_SIGMA) and \
           (self.line_flux > GOOD_MIN_LINE_FLUX) and \
           (self.fit_h/self.cont > GOOD_MIN_H_CONT_RATIO) and \
           (abs(self.fit_dx0) < GOOD_MAX_DX0):
                result = True

            #if self.eqw_obs/(1.+z) > GOOD_MIN_EW_REST:
            #    result =  True

        return result


def signal_score(wavelengths,values,errors,central,central_z = 0.0, spectrum=None,values_units=0, sbr=None, show_plot=False):

    #error on the wavelength of the possible line depends on the redshift and its error and the wavelength itself
    #i.e. wavelength error = wavelength / (1+z + error)  - wavelength / (1+z - error)
    # want that by 1/2 (so +/- error from center ... note: not quite symmetric, but close enough over small delta)
    # and want in pixels so divide by pix_size
    #todo: note error itself should be a function of z ... otherwise, error is constant and as z increases, the
    #todo:   pix_error then decreases
    #however, this error is ON TOP of the wavelength measurement error, assumed to be +/- 1 pixel?
    def pix_error(z,wavelength,error=0.001, pix_size= 2.0):
        return 0.0

        try:
            e =  0.5 * wavelength * (2. * error / ((1.+z)**2 - error**2)) / pix_size
        except:
            e = 0.0
            log.warning("Invalid pix_error in spectrum::signal_score",exc_info=True)

        return e




    accept_fit = False
    #if values_are_flux:
    #    # assumed then to be in cgs units of x10^-17 as per typical HETDEX values
    #    # !!!! reminder, do NOT do values *= 10.0  ... that is an in place operation and overwrites the original
    #    values = values * 10.0  # put in units of 10^-18 so they pretty much match up with counts

    values, values_units = norm_values(values,values_units)

    #sbr signal to background ratio
    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix

    #if near a peak we already found, nudge to align
    if isinstance(spectrum,Spectrum):
        w = spectrum.is_near_a_peak(central,pix_size)
        if w:
            central = w

    # want +/- 20 angstroms in pixel units (maybe should be 50 AA?
    wave_side = int(round(GAUSS_FIT_AA_RANGE / pix_size))  # pixels
    #1.5 seems to be good multiplier ... 2.0 is a bit too much;
    # 1.0 is not bad, but occasionally miss something by just a bit

    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)
    #fit_range_AA = GAUSS_FIT_PIX_ERROR * pix_size #1.0  # peak must fit to within +/- fit_range AA
                                  # this allows room to fit, but will enforce +/- pix_size after
    #num_of_sigma = 3.0  # number of sigma to include on either side of the central peak to estimate noise

    len_array = len(wavelengths)
    idx = getnearpos(wavelengths,central)
    min_idx = max(0,idx-wave_side)
    max_idx = min(len_array,idx+wave_side)
    wave_x = wavelengths[min_idx:max_idx+1]
    wave_counts = values[min_idx:max_idx+1]
    if (errors is not None) and (len(errors) == len(wavelengths)):
        wave_errors = errors[min_idx:max_idx+1]
    else:
        wave_errors = None

    if False: #do I want to use a more narrow range for the gaussian fit? still uses the wider range for RMSE
        min_idx = max(0, idx - wave_side/2)
        max_idx = min(len_array, idx + wave_side/2)
        narrow_wave_x = wavelengths[min_idx:max_idx+1]
        narrow_wave_counts = values[min_idx:max_idx + 1]
        if (errors is not None) and (len(errors) == len(wavelengths)):
            narrow_wave_errors = errors[min_idx:max_idx + 1]
        else:
            narrow_wave_errors = None
    else:
        narrow_wave_x = wave_x
        narrow_wave_counts = wave_counts
        narrow_wave_errors = wave_errors

    #blunt very negative values
    #wave_counts = np.clip(wave_counts,0.0,np.inf)

    xfit = np.linspace(wave_x[0], wave_x[-1], 1000) #range over which to plot the gaussian equation
    peak_pos = getnearpos(wavelengths, central)

    try:
        # find the highest point in the raw data inside the range we are allowing for the line center fit
        dpix = int(round(fit_range_AA / pix_size))
        raw_peak = max(values[peak_pos-dpix:peak_pos+dpix+1])
    except:
        #this can fail if on very edge, but if so, we would not use it anyway
        log.info("Raw Peak value failure for wavelength (%f) at index (%d). Cannot fit to gaussian. " %(central,peak_pos))
        return None

    fit_peak = None

    eli = EmissionLineInfo()
    eli.pix_size = pix_size
    num_sn_pix = 0

    #use ONLY narrow fit
    try:

        # parm[0] = central point (x in the call), parm[1] = sigma, parm[2] = 'a' multiplier (happens to also be area)
        # parm[3] = y offset (e.g. the "continuum" level)
        #get the gaussian for the more central part, but use that to get noise from wider range
        #sigma lower limit at 0.5 (could make more like pixel_size / 4.0 or so, but probabaly should not depend on that
        # the minimum size is in angstroms anyway, not pixels, and < 0.5 is awfully narrow to be real)
        # instrument resolution ~ 1.9AA/pix (dispersion around 2.2?)
        #todo: add sigma=error
        parm, pcov = curve_fit(gaussian, narrow_wave_x, narrow_wave_counts,
                                p0=(central,1.5,1.0,0.0),
                                bounds=((central-fit_range_AA, 1.0, 0.0, -100.0),
                                        (central+fit_range_AA, np.inf, np.inf, np.inf)),
                                #sigma=1./(narrow_wave_errors*narrow_wave_errors)
                                sigma=narrow_wave_errors
                               #method='trf'
                               )

        perr = np.sqrt(np.diag(pcov)) #1-sigma level errors on the fitted parameters
        #e.g. flux = a = parm[2]   +/- perr[2]*num_of_sigma_confidence
        #where num_of_sigma_confidence ~ at a 5 sigma confidence, then *5 ... at 3 sigma, *3

        eli.fit_vals = gaussian(xfit, parm[0], parm[1], parm[2], parm[3])
        eli.fit_wave = xfit.copy()
        eli.raw_vals = wave_counts[:]
        eli.raw_wave = wave_x[:]

        #matches up with the raw data scale so can do RMSE
        rms_wave = gaussian(wave_x, parm[0], parm[1], parm[2], parm[3])

        eli.fit_x0 = parm[0]
        eli.fit_dx0 = central - eli.fit_x0
        scaled_fit_h = max(eli.fit_vals)
        eli.fit_h = scaled_fit_h
        eli.fit_rh = eli.fit_h / raw_peak
        eli.fit_sigma = parm[1] #units of AA not pixels
        eli.fit_a = parm[2] #this is an AREA so there are 2 powers of 10.0 in it (hx10 * wx10) if in e-18 units
        eli.fit_y = parm[3]

        raw_idx = getnearpos(eli.raw_wave, eli.fit_x0)
        eli.raw_h = max(eli.raw_vals[raw_idx - 3:raw_idx + 4])
        eli.raw_x0 = eli.raw_wave[getnearpos(eli.raw_vals, eli.raw_h)]

        fit_peak = max(eli.fit_vals)

        if (abs(raw_peak - fit_peak) / raw_peak > 0.2):  # didn't capture the peak ... bad, don't calculate anything else
            #log.warning("Failed to capture peak")
            log.debug("Failed to capture peak: raw = %f , fit = %f, frac = %0.2f" % (raw_peak, fit_peak,
                                                                                 abs(raw_peak - fit_peak) / raw_peak))
        else:
            #check the dx0

            p_err = pix_error(central_z,eli.fit_x0,pix_size=pix_size)
            if (abs(eli.fit_dx0) > (1.75 * pix_size + p_err)):
                log.debug("Failed to capture peak: dx0 = %f, pix_size = %f, wavelength = %f, pix_z_err = %f"
                          % (eli.fit_dx0,pix_size, eli.fit_x0,p_err))
            else:
                accept_fit = True
                log.debug("Success: captured peak: raw = %f , fit = %f, frac = %0.2f"
                          % (raw_peak, fit_peak, abs(raw_peak - fit_peak) / raw_peak))

                num_sn_pix = int(round(max(GAUSS_SNR_SIGMA * eli.fit_sigma, GAUSS_SNR_NUM_AA)/pix_size)) #half-width in AA

                #?rms just under the part of the plot with signal (not the entire fit part) so, maybe just a few AA or pix
                eli.fit_norm_rmse = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, eli.fit_x0 ), hw_pix=num_sn_pix,
                             norm=True)
                eli.fit_rmse = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, eli.fit_x0 ), hw_pix=num_sn_pix,
                             norm=False)

                num_sn_pix = num_sn_pix * 2 + 1 #need full width later (still an integer)
    except Exception as ex:
        if ex.message.find("Optimal parameters not found") > -1:
            log.info("Could not fit gaussian near %f" % central,exc_info=False)
        else:
            log.error("Could not fit gaussian near %f" % central, exc_info=True)
        return None

    if (eli.fit_rmse > 0) and (eli.fit_sigma <= GAUSS_FIT_MAX_SIGMA) and (eli.fit_sigma >= GAUSS_FIT_MIN_SIGMA):
        eli.snr = eli.fit_a/(np.sqrt(num_sn_pix)*eli.fit_rmse)
        eli.build(values_units=values_units)
        #eli.snr = max(eli.fit_vals) / (np.sqrt(num_sn_pix) * eli.fit_rmse)
        snr = eli.snr
    else:
        accept_fit = False
        snr = 0.0
        eli.line_flux = 0.0

    log.debug("SNR at %0.2f = %0.2f"%(central,snr))

    title = ""

    #todo: re-calibrate to use SNR instead of SBR ??
    if sbr is None:
        sbr = est_peak_strength(wavelengths,values,central,values_units)
        if sbr is None:
            #done, no reason to continue
            log.warning("Could not determine SBR at wavelength = %f" %central)
            return None

    score = sbr
    eli.sbr = sbr
    sk = -999
    ku = -999
    si = -999
    dx0 = -999 #in AA
    rh = -999
    mx_norm = max(wave_counts)/100.0

    fit_wave = eli.fit_vals
    error = eli.fit_norm_rmse

    #fit around designated emis line
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
            log.info("Minimum peak height (%f) too small. Score zeroed." % (height_pix))
            dqs_raw = 0.0
            score = 0.0
            rh = 0.0

        #todo: for lower S/N, sigma (width) can be less and still get bonus if fibers have larger separation

        #new_score:
        if (0.75 < rh < 1.25) and (error < 0.2): # 1 bad pixel in each fiber is okay, but no more

            #central peak position
            if abs(dx0) > pix_size:# 1.9:  #+/- one pixel (in AA)  from center
                val = (abs(dx0) - pix_size)** 2
                score -= val
                log.debug("Penalty for excessive error in X0: %f" % (val))


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
            else: #very wrong
                val = np.sqrt(si-15.0)
                score -= val
                log.debug("Penalty for excessive sigma: %f" % (val))


            #only check the skew for smaller sigma
            #skew scoring
            if si < 2.5:
                if sk < -0.5: #skew wrong directionn
                    val = min(1.0,mx_norm*min(0.5,abs(sk)-0.5))
                    score -= val
                    log.debug("Penalty for low sigma and negative skew: %f" % (val))
                if (sk > 2.0): #skewed a bit red, a bit peaky, with outlier influence
                    val = min(0.5,sk-2.0)
                    score += val
                    log.debug("Bonus for low sigma and positive skew: %f" % (val))

            base_msg = "Fit dX0 = %g(AA), RH = %0.2f, rms = %0.2f, Sigma = %g(AA), Skew = %g , Kurtosis = %g "\
                   % (dx0, rh, error, si, sk, ku)
            log.info(base_msg)
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
            log.info("Too many bad pixels or failure to fit peak or overall bad fit. ")
            score = 0.0
    else:
        log.info("Unable to fit gaussian. ")
        score = 0.0

    if show_plot or G.DEBUG_SHOW_GAUSS_PLOTS:
        if error is None:
            error = -1

        g = eli.is_good(z=central_z)
        a = accept_fit

        title += "%0.2f z_guess=%0.4f A(%d) G(%d)\n" \
                 "Score = %0.2f (%0.1f), SBR = %0.2f (%0.1f), SNR = %0.2f (%0.1f) wpix = %d\n" \
                 "Peak = %0.2g, Line(A) = %0.2g, Cont = %0.2g, EqW_Obs=%0.2f\n"\
                 "dX0 = %0.2f, RH = %0.2f, RMS = %0.2f (%0.2f) \n"\
                 "Sigma = %0.2f, Skew = %0.2f, Kurtosis = %0.2f"\
                  % (eli.fit_x0,central_z,a,g,score, signal_calc_scaled_score(score),sbr,
                     signal_calc_scaled_score(sbr),snr,signal_calc_scaled_score(snr),num_sn_pix,
                     eli.fit_h,eli.line_flux, eli.cont,eli.eqw_obs,
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
            gauss_plot.plot(xfit, fit_wave, c='b')
            gauss_plot.grid(True)

            ymin = min(min(fit_wave),min(wave_counts))
            ymax = max(max(fit_wave),max(wave_counts))
        else:
            ymin = min(wave_counts)
            ymax = max(wave_counts)
        gauss_plot.set_ylabel("Summed Counts")
        gauss_plot.set_xlabel("Wavelength $\AA$ ")

        ymin *= 1.1
        ymax *= 1.1

        if abs(ymin) < 1.0: ymin = -1.0
        if abs(ymax) < 1.0: ymax = 1.0

        gauss_plot.set_ylim((ymin,ymax))
        gauss_plot.set_xlim( (np.floor(wave_x[0]),np.ceil(wave_x[-1])) )
        gauss_plot.set_title(title)
        stat = ""
        if a:
            stat += "a"
        if g:
            stat += "g"

        png = "gauss_" + str(central)+ "_" + stat + ".png"

        log.info('Writing: ' + png)
        #print('Writing: ' + png)
        fig.tight_layout()
        fig.savefig(png)
        fig.clear()
        plt.close()
        # end plotting

    if accept_fit:
        eli.raw_score = score
        eli.score = signal_calc_scaled_score(score)
        return eli
    else:
        log.info("Fit rejected")
        return None


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

def est_peak_strength(wavelengths,values,central,values_units=0,dw=DEFAULT_BACKGROUND_WIDTH,peaks=None,valleys=None):
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
    values, values_units = norm_values(values, values_units)

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
            log.info("Raw Peak value failure for wavelength (%f) at index (%d). Cannot calculate SBR. "
                     % (central, peak_pos))
            return 0

        #signal = ((np.sqrt(signal)-zero)/2.0)**2

        if peak_str is not None:
           # sbr = (signal-background)/(background)
           sbr = peak_str/background

    return sbr


#todo: update to deal with flux instead of counts
def simple_peaks(x,v,h=MIN_HEIGHT,delta_v=2.0,values_units=0):
    """

    :param x:
    :param v:
    :return:  #3 arrays: index of peaks, coordinate (wavelength) of peaks, values of peaks
              2 3D arrays: index, wavelength, value for (1) peaks and (2) valleys
    """

    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v, values_units = norm_values(v, values_units)

    v = np.asarray(v)
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


def peakdet(x,v,dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0,values_units=0):

    """

    :param x:
    :param v:
    :param dw:
    :param h:
    :param dh:
    :param zero:
    :return: array of [ pi, px, pv, pix_width, centroid_pos, eli.score, eli.snr]
    """

    #peakind = signal.find_peaks_cwt(v, [2,3,4,5],min_snr=4.0) #indexes of peaks

    #emis = zip(peakind,x[peakind],v[peakind])
    #emistab.append((pi, px, pv, pix_width, centroid))
    #return emis



    #dh (formerly, delta)
    #dw (minimum width (as a fwhm) for a peak, else is noise and is ignored) IN PIXELS
    # todo: think about jagged peaks (e.g. a wide peak with many subpeaks)
    #zero is the count level zero (nominally zero, but with noise might raise or lower)
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html


    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []
    emistab = []
    eli_list = []
    delta = dh

    pix_size = abs(x[1] - x[0])  # aa per pix
    # want +/- 20 angstroms
    wave_side = int(round(20.0 / pix_size))  # pixels

    dw = int(dw / pix_size) #want round down (i.e. 2.9 -> 2) so this is fine

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)
    num_pix = len(v)

    if num_pix != len(x):
        log.warning('peakdet: Input vectors v and x must have same length')
        return []

    if not np.isscalar(dh):
        log.warning('peakdet: Input argument delta must be a scalar')
        return []

    if dh <= 0:
        log.warning('peakdet: Input argument delta must be positive')
        return []


    v_0 = v[:]
    x_0 = x[:]
    values_units_0 = values_units

    #if values_are_flux:
    #    v = v * 10.0

    v,values_units = norm_values(v,values_units)

    #smooth v and rescale x,
    #the peak positions are unchanged but some of the jitter is smoothed out
    #v = v[:-2] + v[1:-1] + v[2:]
    v = v[:-4] + v[1:-3] + v[2:-2] + v[3:-1] + v[4:]
    #v = v[:-6] + v[1:-5] + v[2:-4] + v[3:-3] + v[4:-2] + v[5:-1] + v[6:]
    v /= 5.0
    x = x[2:-2]

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
            if (thisv >= h) and (thisv < maxv - delta):
                #i-1 since we are now on the right side of the peak and want the index associated with max
                maxtab.append((maxidx,maxpos, maxv))
                minv = thisv
                minpos = x[i]
                lookformax = False
        else:
            if thisv > minv + delta:
                mintab.append((minidx,minpos, minv))
                maxv = thisv
                maxpos = x[i]
                lookformax = True


    if len(maxtab) < 1:
        log.warning("No peaks found with given conditions: mininum:  fwhm = %f, height = %f, delta height = %f" \
                %(dw,h,dh))
        return []

    #make an array, slice out the 3rd column
    #gm = gmean(np.array(maxtab)[:,2])
    peaks = np.array(maxtab)[:, 2]
    gm = np.mean(peaks)
    std = np.std(peaks)

    #now, throw out anything waaaaay above the mean (toss out the outliers and recompute mean)
    if False:
        sub = peaks[np.where(abs(peaks - gm) < (3.0*std))[0]]
        if len(sub) < 3:
            sub = peaks
        gm = np.mean(sub)

    for pi,px,pv in maxtab:
        #check fwhm (assume 0 is the continuum level)

        #minium height above the mean of the peaks (w/o outliers)
        if False:
            if (pv < 1.333 * gm):
                continue

        hm = float((pv - zero) / 2.0)
        pix_width = 0

        #for centroid (though only down to fwhm)
        sum_pos_val = x[pi] * v[pi]
        sum_pos = x[pi]
        sum_val = v[pi]

        #check left
        pix_idx = pi -1

        try:
            while (pix_idx >=0) and (v[pix_idx] >= hm):
                sum_pos += x[pix_idx]
                sum_pos_val += x[pix_idx] * v[pix_idx]
                sum_val += v[pix_idx]
                pix_width += 1
                pix_idx -= 1

        except:
            pass

        #check right
        pix_idx = pi + 1

        try:
            while (pix_idx < num_pix) and (v[pix_idx] >= hm):
                sum_pos += x[pix_idx]
                sum_pos_val += x[pix_idx] * v[pix_idx]
                sum_val += v[pix_idx]
                pix_width += 1
                pix_idx += 1
        except:
            pass

        #check local region around centroid
        centroid_pos = sum_pos_val / sum_val #centroid is an index

        #what is the average value in the vacinity of the peak (exlcuding the area under the peak)
        #should be 20 AA not 20 pix
        side_pix = max(wave_side,pix_width)
        left = max(0,(pi - pix_width)-side_pix)
        sub_left = v[left:(pi - pix_width)]
   #     gm_left = np.mean(v[left:(pi - pix_width)])

        right = min(num_pix,pi+pix_width+side_pix+1)
        sub_right = v[(pi + pix_width):right]
   #     gm_right = np.mean(v[(pi + pix_width):right])

        #minimum height above the local gm_average
        #note: can be a problem for adjacent peaks?
        if False:
            if pv < (2.0 * np.mean(np.concatenate((sub_left,sub_right)))):
                continue

        #check vs minimum width
        if not (pix_width < dw):
            #see if too close to prior peak (these are in increasing wavelength order)

            #this is dumb but necessary for right now since signal_score will multiply by 10
            eli = signal_score(x_0, v_0, None, px,values_units=values_units_0)

            #if (eli is not None) and (eli.score > 0) and (eli.snr > 7.0) and (eli.fit_sigma > 1.6) and (eli.eqw_obs > 5.0):
            if (eli is not None) and eli.is_good():
                eli_list.append(eli)
                if len(emistab) > 0:
                    if (px - emistab[-1][1]) > 6.0:
                        emistab.append((pi, px, pv,pix_width,centroid_pos,eli.eqw_obs,eli.snr))
                    else: #too close ... keep the higher peak
                        if pv > emistab[-1][2]:
                            emistab.pop()
                            emistab.append((pi, px, pv, pix_width, centroid_pos,eli.eqw_obs,eli.snr))
                else:
                    emistab.append((pi, px, pv, pix_width, centroid_pos,eli.eqw_obs,eli.snr))


    #return np.array(maxtab), np.array(mintab)
    #print("DEBUG ... peak count = %d" %(len(emistab)))
    #for i in range(len(emistab)):
    #    print(emistab[i][1],emistab[i][2], emistab[i][5])
    #return emistab
    return eli_list


class EmissionLine():
    def __init__(self,name,w_rest,plot_color,solution=True,z=0,score=0.0):
        self.name = name
        self.w_rest = w_rest
        self.w_obs = w_rest * (1.0 + z)
        self.z = z
        self.color = plot_color
        self.solution = solution #True = can consider this as the target line

        #can be filled in later if a specific instance is created and a model fit to it
        self.score = score
        self.snr = None
        self.sbr = None
        self.flux = None
        self.eqw_obs = None
        self.eqw_rest = None
        self.sigma = None #gaussian fit sigma

    def redshift(self,z):
        self.z = z
        self.w_obs = self.w_rest * (1.0 + z)
        return self.w_obs




class Classifier_Solution:
    def __init__(self):
        self.score = 0.0
        self.frac_score = 0.0
        self.z = 0.0
        self.central_rest = 0.0
        self.name = ""
        self.color = None
        self.emission_line = None

        self.lines = [] #list of EmissionLine


class Spectrum:
    """
    helper functions for spectra
    actual spectra data is kept in fiber.py
    """

    def __init__(self):
        #reminder ... colors don't really matter (are not used) if solution is not True)
        self.emission_lines = [EmissionLine("Ly$\\alpha$ ", G.LyA_rest, 'red'),
                               EmissionLine("OII ", G.OII_rest, 'green'),
                               EmissionLine("OIII", 4960.295, "lime"),
                               EmissionLine("OIII", 5008.240, "lime"),

                               EmissionLine("CIV ", 1549.48, "blueviolet"),  # big in AGN
                               EmissionLine("CIII", 1908.734, "purple"),  #big in AGN
                               EmissionLine("MgII", 2799.117, "magenta"),  #big in AGN

                               EmissionLine("H$\\beta$ ", 4862.68, "blue"),
                               EmissionLine("H$\\gamma$ ", 4341.68, "royalblue"),
                               #EmissionLine("H$\\delta ", 4102, "royalblue", solution=False),
                               #EmissionLine("H$\\epsilon ", 3970, "royalblue", solution=False),
                               #EmissionLine("H$\\zeta ", 3889, "royalblue", solution=False),
                               #EmissionLine("H$\\eta ", 3835, "royalblue", solution=False),

                               EmissionLine("NV ", 1240.81, "teal", solution=False),
                               EmissionLine("SiII", 1260, "gray", solution=False),
                               EmissionLine("HeII", 1640.4, "orange", solution=False),
                               EmissionLine("NeIII", 3869, "pink", solution=False),
                               EmissionLine("NeIII", 3967, "pink", solution=False),
                               EmissionLine("NeV", 3346.79, "pink", solution=False),
                               EmissionLine("NeVI", 3426.85, "pink", solution=False),
                               EmissionLine("NaI", 4980, "lightcoral", solution=False),  #4978.5 + 4982.8
                               EmissionLine("NaI",5153,"lightcoral",solution=False)  #5148.8 + 5153.4
                               ]

        self.wavelengths = []
        self.values = [] #could be fluxes or counts or something else ... right now needs to be counts
        self.errors = []
        self.values_units = 0

        self.central = None
        self.estflux = None
        self.eqw_obs = None

        self.central_eli = None

        self.solutions = []
        self.all_found_lines = None #EmissionLineInfo objs (want None here ... if no lines, then peakdet returns [])

        self.addl_fluxes = []
        self.addl_wavelengths = []
        self.addl_fluxerrs = []
        self.p_lae = None
        self.p_oii = None
        self.p_lae_oii_ratio = None

    def top_hat_filter(self,w_rest,w_obs, wx, hat_width=None, negative=False):
        #optimal seems to be around 1 to < 2 resolutions (e.g. HETDEX ~ 6AA) ... 6 is good, 12 is a bit
        #unstable ... or as rougly 3x pixel pix_size


        #build up an array with tophat filters at emission line positions
        #based on the rest and observed (shifted and streched based on the redshift)
        # wx is the array of wavelengths (e.g the x coords)
        # hat width in angstroms
        try:
            w_rest = np.float(w_rest)
            w_obs = np.float(w_obs)
            num_hats = 0

            if negative:
                filter = np.full(np.shape(wx), -1.0)
            else:
                filter = np.zeros(np.shape(wx))

            pix_size = np.float(wx[1]-wx[0]) #assume to be evenly spaced


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


    def set_spectra(self,wavelengths, values, errors, central, values_units = 0, estflux=None,eqw_obs=None):
        del self.wavelengths[:]
        del self.values[:]
        del self.errors[:]
        if self.all_found_lines is not None:
            del self.all_found_lines[:]
        if self.solutions is not None:
            del self.solutions[:]

        if (estflux is None) or (eqw_obs is None):
            eli = signal_score(wavelengths=wavelengths, values=values, errors=errors,central=central,
                               values_units=values_units,sbr=None, show_plot=False)
            if eli:
                estflux = eli.line_flux
                eqw_obs = eli.eqw_obs

                self.central_eli = copy.deepcopy(eli)

        self.wavelengths = wavelengths
        self.values = values
        self.errors = errors
        self.values_units = values_units
        self.central = central
        self.estflux = estflux
        self.eqw_obs = eqw_obs


    def find_central_wavelength(self,wavelengths = None,values = None, values_units=0):
        central = 0.0
        update_self = False
        if (wavelengths is None) or (values is None):
            wavelengths = self.wavelengths
            values = self.values
            update_self = True

        #find the peaks and use the largest
        #for now, largest means highest value

        # if values_are_flux:
        #     #!!!!! do not do values *= 10.0 (will overwrite)
        #     # assumes fluxes in e-17 .... making e-18 ~= counts so logic can stay the same
        #     values = values * 10.0

        values,values_units = norm_values(values,values_units)

        peaks = peakdet(wavelengths,values) #as of 2018-06-11 these are EmissionLineInfo objects
        max_v = -np.inf
        #find the largest flux
        for p in peaks:
            #  0   1   2   3          4
            # pi, px, pv, pix_width, centroid_pos
            #if p[2] > max_v:
            #    max_v = p[2]
            #    central = p[4]
            if p.line_flux > max_v:
                max_v = p.line_flux
                central = p.fit_x0

        if update_self:
            self.central = central

        log.info("Central wavelength = %f" %central)

        return central

    def classify(self,wavelengths = None,values = None, errors=None, central = None, values_units=0):
        #for now, just with additional lines
        #todo: later add in continuum
        #todo: later add in bayseian stuff
        if not G.CLASSIFY_WITH_OTHER_LINES:
            return []

        del self.solutions[:]
        if (wavelengths is not None) and (values is not None) and (central is not None):
            self.set_spectra(wavelengths,values,errors,central,values_units=values_units)
        else:
            wavelengths = self.wavelengths
            values = self.values
            central = self.central
            errors=self.errors
            values_units = self.values_units

        #if central wavelength not provided, find the peaks and use the largest
        #for now, largest means highest value
        if (central is None) or (central == 0.0):
            central = self.find_central_wavelength(wavelengths,values,values_units=values_units)

        if (central is None) or (central == 0.0):
            log.warning("Cannot classify. No central wavelength specified or found.")
            return []

        solutions = self.classify_with_additional_lines(wavelengths,values,errors,central,values_units)
        self.solutions = solutions

        #get the LAE and OII solutions and send to Bayesian to check p_LAE/p_OII
        del self.addl_fluxes[:]
        del self.addl_wavelengths[:]
        del self.addl_fluxerrs[:]

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
                        #todo: get real error
                        self.addl_fluxerrs.append(0.0)

        #if len(addl_fluxes) > 0:
        self.get_bayes_probabilities(addl_wavelengths=self.addl_wavelengths,addl_fluxes=self.addl_fluxes,
                                     addl_errors=self.addl_fluxerrs)
        #self.get_bayes_probabilities(addl_fluxes=None, addl_wavelengths=None)

        return solutions


    def is_near_a_peak(self,w,pix_size=1.9): #is the provided wavelength near one of the found peaks (+/- few AA or pixels)

        wavelength = 0.0
        if (self.all_found_lines is None):
            self.all_found_lines = peakdet(self.wavelengths, self.values, values_units=self.values_units)

        if self.all_found_lines is None:
            return 0.0

        GAUSS_FIT_PIX_ERROR = 3.0  # error (freedom) in pixels: have to allow at least 2 pixels of error
        # (if aa/pix is small, _AA_ERROR takes over)
        GAUSS_FIT_AA_ERROR = 1.0  # error (freedom) in line center in AA, still considered to be okay

        max_dx = max(GAUSS_FIT_AA_ERROR, GAUSS_FIT_PIX_ERROR*pix_size)
        for f in self.all_found_lines:
            dx = abs(f.fit_x0 - w)
            if dx < max_dx:
                wavelength = f.fit_x0
                break

        return wavelength




    def classify_with_additional_lines(self,wavelengths = None,values = None,errors=None,central = None,
                                       values_units=0):
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
            self.all_found_lines = peakdet(wavelengths,values,values_units=values_units)

        solutions = []
        total_score = 0.0 #sum of all scores (use to represent each solution as fraction of total score)


        #for each self.emission_line
        #   run down the list of remianing self.emission_lines and calculate score for each
        #   make a copy of each emission_line, set the score, save to the self.lines list []
        #
        #sort solutions by score

        max_w = max(wavelengths)
        min_w = min(wavelengths)

        for e in self.emission_lines:
            if not e.solution: #if this line is not allowed to be the main line
                continue

            if (central/e.w_rest - 1.0) < 0.0:
                continue #impossible, can't have a negative z

            sol = Classifier_Solution()
            sol.z = central/e.w_rest - 1.0
            sol.central_rest = e.w_rest
            sol.name = e.name
            sol.color = e.color
            sol.emission_line = copy.deepcopy(e)
            sol.emission_line.w_obs = sol.emission_line.w_rest*(1.0 + sol.z)
            sol.emission_line.solution = True

            for a in self.emission_lines:
                if e == a:
                    continue

                a_central = a.w_rest*(sol.z+1.0)
                if (a_central > max_w) or (a_central < min_w):
                    continue

                if central is not None:
                    central_z = central/e.w_rest - 1.0
                else:
                    central_z = 0.0

                eli = signal_score(wavelengths, values, errors, a_central,
                                   central_z = central_z, values_units=values_units, spectrum=self)

                #if (eli is not None):# and (eli.score > 0.0):
                #    total_score += eli.score
                #    sol.score += eli.score

                #if (eli is not None) and (eli.sbr > 1.0):
                if (eli is not None) and eli.is_good(z=sol.z):
                    #is there a corresponding peak?
                    #this helps enforce a shape (has to correspond to one of the peaks found by shape)
                    if self.is_near_a_peak(eli.fit_x0,eli.pix_size): #todo: redundant now that is in signal_score
                        total_score += eli.snr * (eli.line_flux * 1.0e17) #eli.eqw_obs
                        sol.score += eli.snr * (eli.line_flux * 1.0e17) #eli.eqw_obs
                        l = copy.deepcopy(a)
                        l.w_obs = l.w_rest * (1.0 + sol.z)
                        l.z = sol.z
                        l.score = eli.score
                        l.snr = eli.snr
                        l.sbr = eli.sbr
                        l.eqw_obs = eli.eqw_obs
                        l.eqw_rest = l.eqw_obs / (1.0 + l.z)
                        l.flux = eli.line_flux
                        l.sigma = eli.fit_sigma

                        sol.lines.append(l)

            if sol.score > 0.0:
                solutions.append(sol)

        for s in solutions:
            s.frac_score = s.score/total_score

        #sort by score
        solutions.sort(key=lambda x: x.score, reverse=True)

        for s in solutions:
            ll =""
            for l in s.lines:
                ll += " %s(%0.1f at %0.1f)," %(l.name,l.w_rest,l.w_obs)
            msg = "Possible Solution: %s (%0.1f at %0.1f), Frac = %0.2f, Score = %0.1f, z = %0.5f, +lines=%d %s" \
                    % (s.emission_line.name,s.central_rest,s.central_rest*(1.0+s.z), s.frac_score, s.score,s.z,
                       len(s.lines),ll )
            log.info(msg)
            # todo: DEBUG  remove ... temporary
            print(msg)

        return solutions


    def get_bayes_probabilities(self,addl_wavelengths=None,addl_fluxes=None,addl_errors=None):
        # todo: feed in addl_fluxes from the additonal line solutions (or build specifically)?
        # todo: make use of errors

        #care only about the LAE and OII solutions:
        #todo: find the LyA and OII options in the solution list and use to fill in addl_fluxes?

        ratio, self.p_lae, self.p_oii = line_prob.prob_LAE(wl_obs=self.central,
                                                           lineFlux=self.estflux,
                                                           ew_obs=(self.eqw_obs),
                                                           c_obs=None, which_color=None,
                                                           addl_fluxes=addl_fluxes, addl_wavelengths=addl_wavelengths,
                                                           sky_area=None,
                                                           cosmo=None, lae_priors=None,
                                                           ew_case=None, W_0=None,
                                                           z_OII=None, sigma=None)
        if (self.p_lae is not None) and (self.p_lae > 0.0):
            if (self.p_oii is not None) and (self.p_oii > 0.0):
                self.p_lae_oii_ratio = self.p_lae /self.p_oii
            else:
                self.p_lae_oii_ratio = float('inf')
        else:
            self.p_lae_oii_ratio = 0.0

        self.p_lae_oii_ratio = min(line_prob.MAX_PLAE_POII,self.p_lae_oii_ratio) #cap to MAX

    def build_full_width_spectrum(self, counts = None, wavelengths = None, central_wavelength = None,
                                  show_skylines=True, show_peaks = True, name=None,
                                  dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0):


        use_internal = False
        if (counts is None) or (wavelengths is None) or (central_wavelength is None):
            counts = self.values
            #if self.values_are_flux: #!!! need a copy here ... DO NOT counts *= 10.0
            #    counts = counts * 10.0 #flux assumed to be cgs x10^-17 ... by 10x to x10^-18 become very similar to counts in scale

            counts, values_units = norm_values(counts,self.values_units)
            wavelengths = self.wavelengths
            central_wavelength = self.central
            use_internal = True


        # fig = plt.figure(figsize=(5, 6.25), frameon=False)
        fig = plt.figure(figsize=(8, 2), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        dy = 1.0 / 5.0  # + 1 skip for legend, + 2 for double height spectra + 2 for double height labels

        # this is the 1D averaged spectrum
        #textplot = plt.axes([0.025, .6, 0.95, dy * 2])
        specplot = plt.axes([0.05, 0.20, 0.90, 0.40])
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
                if (self.all_found_lines is not None):
                    peaks = self.all_found_lines
                else:
                    peaks = peakdet(wavelengths,counts,dw,h,dh,zero,values_units=values_units) #as of 2018-06-11 these are EmissionLineInfo objects

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


