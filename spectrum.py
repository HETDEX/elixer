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


log = G.logging.getLogger('spectrum_logger')
log.setLevel(G.logging.DEBUG)

MIN_FWHM = 5 #AA (must xlat to pixels)
MIN_HEIGHT = 10
MIN_DELTA_HEIGHT = 2 #to be a peak, must be at least this high above next adjacent point to the left
DEFAULT_BACKGROUND = 6.0
DEFAULT_BACKGROUND_WIDTH = 100.0 #pixels
DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND = 10.0 #pixels
MAX_SIGMA = 10.0 #maximum width (pixels) for fit gaussian to signal (greater than this, is really not a fit)
DEBUG_SHOW_PLOTS = True


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


def fit_gaussian(x,y):
    yfit = None
    parm = None
    pcov = None
    try:
        parm, pcov = curve_fit(gaussian, x, y,bounds=((-np.inf,0,-np.inf),(np.inf,np.inf,np.inf)))
        yfit = gaussian(x,parm[0],parm[1],parm[2])
    except:
        log.error("Exception fitting gaussian.",exc_info=True)

    return yfit,parm,pcov

#todo: !!!!! definitely add a penality for not catching the peak (maybe within 10% or 20% at most?)
#maybe even zero it out if miss more than 25%?
def est_snr(wavelengths,values,central): #,rms=None,fwhm=None,peak=None):
    #todo: what about a second, nearby peak? Will that mess up the +/- 20 AA
    #assume fixed pixel size
    pix_size = abs(wavelengths[1]-wavelengths[0]) #aa per pix
    #want +/- 20 angstroms in pixel units
    wave_side = int(round(20.0 / pix_size)) # pixels
    num_of_sigma = 3 #number of sigma to include on either side of the central peak to estimate noise
    fit_range_AA = 1.0  # peak must fit to within +/- fit_range in angstroms (not PIX)

    len_array = len(wavelengths)

    idx = getnearpos(wavelengths, central)
    min_idx = max(0, idx - wave_side)
    max_idx = min(len_array, idx + wave_side)
    wave_x = wavelengths[min_idx:max_idx + 1]
    wave_counts = values[min_idx:max_idx + 1]

    min_idx = max(0, idx - wave_side/2)
    max_idx = min(len_array, idx + wave_side/2)
    narrow_wave_x = wavelengths[min_idx:max_idx+1]
    narrow_wave_counts = values[min_idx:max_idx + 1]



    # blunt very negative values
    # wave_counts = np.clip(wave_counts,0.0,np.inf)

    xfit = np.linspace(wave_x[0], wave_x[-1], 100)
    #xfit = wave_x

    fit_wave = None

    num_pix = None
    area = 0.0
    sigma = 0.0
    error = None
    raw_peak = values[getnearpos(wavelengths,central)]
    fit_peak = 0.0

    #fwhm = est_fwhm(wavelengths, values, central)

    # use ONLY narrow fit
    try:
        #parm[0] = central point, parm[1] = sigma, parm[2] = area
        parm, pcov = curve_fit(gaussian, narrow_wave_x, narrow_wave_counts, p0=(central, 1.0, 0),
                               bounds=((central - fit_range_AA, 0, -np.inf), (central + fit_range_AA, np.inf, np.inf)))
        fit_wave = gaussian(xfit, parm[0], parm[1], parm[2])

        # putting the gaussian fit on the same scale (wave step size) as the data
        rms_wave = gaussian(wave_x, parm[0], parm[1], parm[2])
        sigma = parm[1]  #this is in AA
        num_pix = int(round(num_of_sigma * sigma / pix_size)) * 2 + 1

        # rms just under the part of the plot with signal (not the entire fit part) so, maybe just a few AA or pix
        error = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, central), hw_pix=(num_pix - 1) / 2,
                    norm=False)
        area = parm[2]

        fit_peak = max(fit_wave)

        if (abs(raw_peak-fit_peak)/raw_peak > 0.2):#didn;t capture the peak ... bad
            log.warning("Failed to capture peak")
            error = None #so we skip out ... call this a non-fit, so there is no signal here
            log.debug("Failed to capture peak: raw = %f , fit = %f, frac = %0.2f" %(raw_peak,fit_peak,
                                                                                abs(raw_peak-fit_peak)/raw_peak ))
        else:
            log.debug("Success: captured peak: raw = %f , fit = %f, frac = %0.2f"
                      % (raw_peak, fit_peak, abs(raw_peak - fit_peak) / raw_peak ))


        #should this be just from the brightest fiber???
        total_flux = area * flux_conversion(central)  # cgs units
        fit_cont_est = 0.5 * (fit_wave[getnearpos(xfit,central-3*sigma)] + fit_wave[getnearpos(xfit,central+3*sigma)]) \
                        * flux_conversion(central)
        raw_cont_est = 0.5 * (values[getnearpos(wavelengths,central-3*sigma)] + \
                              values[getnearpos(wavelengths,central+3*sigma)]) * \
                              flux_conversion(central)

    except:
        log.error("Could not fit gaussian.")
        return 0.0

    if (error is not None) and (sigma < MAX_SIGMA):
        snr = area/(np.sqrt(num_pix)*error)
    else:
        snr = 0.0

    return snr



class EmissionLineInfo:
    """
    mostly a container, could have pretty well just used a dictionary
    """
    def __init__(self):

        self.fit_a = None
        self.fit_x0 = None
        self.fit_sigma = 0.0
        self.fit_y = None
        self.fit_rmse = -999
        self.fit_norm_rmse = -999

        self.fit_wave = []
        self.fit_vals = []

        self.pix_size = None
        self.raw_wave = []
        self.raw_vals = []

        self.total_flux = -999
        self.cont = -999

        self.snr = 0.0
        self.eqw = -999
        self.cont = None
        self.fwhm = None
        self.score = None
        self.raw_score = None

    def build(self):
        if self.fit_sigma is not None:
            self.fwhm = 2.355 * self.fit_sigma  # e.g. 2*sqrt(2*ln(2))* sigma

        if self.fit_x0 is not None:
            if self.fit_a is not None:
                self.total_flux = self.fit_a * flux_conversion(self.fit_x0)  # cgs units
            if (self.fit_y is not None) and (self.fit_y > G.CONTINUUM_FLOOR_COUNTS):
                self.cont = self.fit_y * flux_conversion(self.fit_x0)
            else:
                self.cont = G.CONTINUUM_FLOOR_COUNTS * flux_conversion(self.fit_x0)


        if self.total_flux and self.cont:
            self.eqw = self.total_flux / self.cont




#todo: !!!!!!!!!!!!!! need to update for sigma in AA not pixels and various calculations below
def signal_score(wavelengths,values,central,sbr=None, show_plot=False):

    #sbr signal to background ratio
    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix
    # want +/- 20 angstroms in pixel units
    wave_side = int(round(20.0 / pix_size))  # pixels
    fit_range_AA = pix_size #1.0  # peak must fit to within +/- fit_range AA
    num_of_sigma = 3  # number of sigma to include on either side of the central peak to estimate noise

    len_array = len(wavelengths)

    idx = getnearpos(wavelengths,central)
    min_idx = max(0,idx-wave_side)
    max_idx = min(len_array,idx+wave_side)
    wave_x = wavelengths[min_idx:max_idx+1]
    wave_counts = values[min_idx:max_idx+1]

    if False: #do I want to use a more narrow range for the gaussian fit? still uses the wider range for RMSE
        min_idx = max(0, idx - wave_side/2)
        max_idx = min(len_array, idx + wave_side/2)
        narrow_wave_x = wavelengths[min_idx:max_idx+1]
        narrow_wave_counts = values[min_idx:max_idx + 1]
    else:
        narrow_wave_x = wave_x
        narrow_wave_counts = wave_counts

    #blunt very negative values
    #wave_counts = np.clip(wave_counts,0.0,np.inf)

    xfit = np.linspace(wave_x[0], wave_x[-1], 1000) #range over which to plot the gaussian equation

    peak_pos = getnearpos(wavelengths, central)

    #nearest pixel +/- 1 pixel (yes, pixel, not angstrom)
    try:
        raw_peak = max(values[peak_pos-1:peak_pos+2])
    except:
        #this can fail if on very edge, but if so, we would not use it anyway
        log.info("Raw Peak value failure for wavelength (%f) at index (%d). Cannot fit to gaussian. " %(central,peak_pos))
        return None

    fit_peak = None

    eli = EmissionLineInfo()
    eli.pix_size = pix_size

    #use ONLY narrow fit
    try:

        # parm[0] = central point (x in the call), parm[1] = sigma, parm[2] = 'a' multiplier (happens to also be area)
        # parm[3] = y offset (e.g. the "continuum" level)
        #get the gaussian for the more central part, but use that to get noise from wider range
        parm, pcov = curve_fit(gaussian, narrow_wave_x, narrow_wave_counts,
                                p0=(central,1.0,1.0,0.0),
                                bounds=( (central-fit_range_AA, 0.5, 1.0, -100.0),
                                         (central+fit_range_AA, np.inf, np.inf, np.mean(narrow_wave_counts))))

        eli.fit_vals = gaussian(xfit, parm[0], parm[1], parm[2], parm[3])
        eli.fit_wave = xfit.copy()

        #matches up with the raw data scale so can do RMSE
        rms_wave = gaussian(wave_x, parm[0], parm[1], parm[2], parm[3])

        eli.fit_x0 = parm[0]
        eli.fit_sigma = parm[1] #units of AA not pixels
        eli.fit_a = parm[2]
        eli.fit_y = parm[3]

        eli.build()

        fit_peak = max(eli.fit_vals)

        if (abs(raw_peak - fit_peak) / raw_peak > 0.2):  # didn't capture the peak ... bad, don't calculate anything else
            log.warning("Failed to capture peak")
            log.debug("Failed to capture peak: raw = %f , fit = %f, frac = %0.2f" % (raw_peak, fit_peak,
                                                                                 abs(raw_peak - fit_peak) / raw_peak))
        else:
            log.debug("Success: captured peak: raw = %f , fit = %f, frac = %0.2f"
                      % (raw_peak, fit_peak, abs(raw_peak - fit_peak) / raw_peak))

            num_pix = int(round(num_of_sigma * eli.fit_sigma / pix_size)) * 2 + 1

             # rms just under the part of the plot with signal (not the entire fit part) so, maybe just a few AA or pix
            eli.fit_norm_rmse = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, central), hw_pix=(num_pix-1)/2,
                         norm=True)
            eli.fit_rmse = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, central), hw_pix=(num_pix-1)/2,
                         norm=False)

    except:
        log.error("Could not fit gaussian.",exc_info=True)
        return None

    if (eli.fit_rmse > 0) and (eli.fit_sigma < MAX_SIGMA):
        eli.snr = eli.fit_a/(np.sqrt(num_pix)*eli.fit_rmse)
        #eli.snr = max(eli.fit_vals) / (np.sqrt(num_pix) * eli.fit_rmse)
        snr = eli.snr
    else:
        snr = 0.0

    log.debug("SNR at %0.2f = %0.2f"%(central,snr))

    title = ""

    #todo: re-calibrate to use SNR instead of SBR ??
    if sbr is None:
        sbr = est_peak_strength(wavelengths,values,central)
        if sbr is None:
            #done, no reason to continue
            log.warning("Could not determine SBR at wavelength = %f" %central)
            return 0.0

    score = sbr
    sk = -999
    ku = -999
    si = -999
    dx0 = -999
    rh = -999
    mx_norm = max(wave_counts)/100.0

    fit_wave = eli.fit_vals
    error = eli.fit_norm_rmse

    #fit around designated emis line
    if (fit_wave is not None):
        sk = skew(fit_wave)
        ku = kurtosis(fit_wave) # remember, 0 is tail width for Normal Dist. ( ku < 0 == thinner tails)
        si = parm[1] #*1.9 #scale to angstroms
        dx0 = (parm[0]-central) #*1.9

        #si and ku are correlated at this scale, for emission lines ... fat si <==> small ku

        height_pix = max(wave_counts)
        height_fit = max(fit_wave)

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
            if abs(dx0) > 1.9:  #+/- one pixel (in AA)  from center
                val = (abs(dx0) - 1.9)** 2
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

    if show_plot or DEBUG_SHOW_PLOTS:
        if error is None:
            error = -1
        title += "Score = %0.2f (%0.1f), SBR = %0.2f (%0.1f), SNR = %0.2f (%0.1f)\n" \
                 "Flux = %0.2g, Cont = %0.2g, EqW=%0.2f\n"\
                 "dX0 = %0.2f, RH = %0.2f, RMS = %0.2f (%0.2f) \n"\
                 "Sigma = %0.2f, Skew = %0.2f, Kurtosis = %0.2f"\
                  % (score, signal_calc_scaled_score(score),sbr,
                     signal_calc_scaled_score(sbr),snr,signal_calc_scaled_score(snr),
                     eli.total_flux, eli.cont,eli.eqw,
                     dx0, rh, error,eli.fit_rmse, si, sk, ku)

        fig = plt.figure()
        gauss_plot = plt.axes()

        gauss_plot.plot(wave_x,wave_counts,c='k')

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
        png = 'gauss_' + str(central)+ ".png"

        log.info('Writing: ' + png)
        #print('Writing: ' + png)
        fig.tight_layout()
        fig.savefig(png)
        fig.clear()
        plt.close()
        # end plotting

    eli.raw_score = score
    eli.score = signal_calc_scaled_score(score)
    return eli


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


def est_ew_obs(fwhm=None,peak=None, wavelengths=None, values=None, central=None):

    try:
        if (wavelengths is not None) and (values is not None) and (central is not None):
            fwhm =  est_fwhm(wavelengths,values,central)
            if peak is None:
                peak = values[getnearpos(wavelengths, central)]

        if (fwhm is not None) and (peak is not None):
            return pix_to_aa(fwhm)*peak
        else:
            return None
    except:
        log.error("Error in spectrum::est_ew",exc_info=True)
        return None

def est_ew_rest():
    #need to know z
    pass




def est_fwhm(wavelengths,values,central):

    num_pix = len(wavelengths)
    idx = getnearpos(wavelengths, central)


    background,zero = est_background(wavelengths,values,central)

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

def est_background(wavelengths,values,central,dw=DEFAULT_BACKGROUND_WIDTH,xw=20.0,peaks=None,valleys=None):
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

    xw = max(DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND,xw)

    outlier_x = 3.0
    background = DEFAULT_BACKGROUND
    wavelengths = np.array(wavelengths)
    values = np.array(values)

    if dw > len(wavelengths)/2.0:
        return None, None

    try:
        # peaks, vallyes are 3D arrays = [index in original array, wavelength, value]
        if peaks is None or valleys is None:
            peaks, valleys = simple_peaks(wavelengths,values)

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
            background, zero = est_background(wavelengths, values, central, dw * 2, xw, peaks=None, valleys=None)
            return background, zero

        #zero point is the total average
        zero = np.mean(np.append(peak_v,valley_v))

        if len(peak_v) > 2:
            peak_background = np.mean(peak_v) - zero
            #peak_background = np.std(peak_v)**2
        else:
            background, zero = est_background(wavelengths,values,central,dw*2,xw,peaks=None,valleys=None)
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

def est_peak_strength(wavelengths,values,central,dw=DEFAULT_BACKGROUND_WIDTH,peaks=None,valleys=None):
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
    sbr = None #Signal to Background Ratio  (similar to SNR)
    xw = est_fwhm(wavelengths,values,central)

    background,zero = est_background(wavelengths,values,central,dw,xw,peaks,valleys)

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


def simple_peaks(x,v,h=MIN_HEIGHT,delta_v=2.0):
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

    v = np.asarray(v)
    num_pix = len(v)

    if num_pix != len(x):
        log.warning('peakdet: Input vectors v and x must have same length')
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


def peakdet(x,v,dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0):

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

    Returns two arrays

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
    sub = peaks[np.where(abs(peaks - gm) < (3.0*std))[0]]
    if len(sub) < 3:
        sub = peaks
    gm = np.mean(sub)

    for pi,px,pv in maxtab:
        #check fwhm (assume 0 is the continuum level)

        #minium height above the mean of the peaks (w/o outliers)
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
        if pv < (2.0 * np.mean(np.concatenate((sub_left,sub_right)))):
            continue

        #check vs minimum width
        if not (pix_width < dw):
            #see if too close to prior peak (these are in increasing wavelength order)
            eli = signal_score(x, v, px)

            if (eli is not None) and (eli.score > 0):
                if len(emistab) > 0:
                    if (px - emistab[-1][1]) > 6.0:
                        emistab.append((pi, px, pv,pix_width,centroid_pos,eli.score))
                    else: #too close ... keep the higher peak
                        if pv > emistab[-1][2]:
                            emistab.pop()
                            emistab.append((pi, px, pv, pix_width, centroid_pos,eli.score))
                else:
                    emistab.append((pi, px, pv, pix_width, centroid_pos,eli.score))


    #return np.array(maxtab), np.array(mintab)
    return emistab


class EmissionLine():
    def __init__(self,name,w_rest,plot_color,solution=True,z=0,score=0.0):
        self.name = name
        self.w_rest = w_rest
        self.w_obs = w_rest * (1.0 + z)
        self.z = z
        self.color = plot_color
        self.solution = solution #True = can consider this as the target line
        self.score = score

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

        self.emission_line = None

        self.lines = [] #list of EmissionLine


class Spectrum:
    """
    helper functions for spectra
    actual spectra data is kept in fiber.py
    """

    def __init__(self):

        self.emission_lines = [EmissionLine("Ly$\\alpha$ ", 1216, 'red'),
                               EmissionLine("OII ", 3727, 'green'),
                               EmissionLine("OIII", 4959, "lime"), EmissionLine("OIII", 5007, "lime"),
                               EmissionLine("CIII", 1909, "purple"),
                               EmissionLine("CIV ", 1549, "black"),
                               EmissionLine("H$\\beta$ ", 4861, "blue"),
                               EmissionLine("HeII", 1640, "orange"),
                               EmissionLine("MgII", 2798, "magenta", solution=False),
                               EmissionLine("H$\\gamma$ ", 4341, "royalblue", solution=False),
                               EmissionLine("NV ", 1240, "teal", solution=False),
                               EmissionLine("SiII", 1260, "gray", solution=False)]

        self.wavelengths = []
        self.values = [] #could be fluxes or counts or something else
        self.central = None

        self.solutions = []

    def set_spectra(self,wavelengths, values, central):
        del self.wavelengths[:]
        del self.values[:]

        self.wavelengths = wavelengths
        self.values = values
        self.central = central


    def find_central_wavelength(self,wavelengths = None,values = None):
        central = 0.0
        update_self = False
        if (wavelengths is None) or (values is None):
            wavelengths = self.wavelengths
            values = self.values
            update_self = True

        #find the peaks and use the largest
        #for now, largest means highest value

        peaks = peakdet(wavelengths,values)
        max_v = -np.inf
        for p in peaks:
            #  0   1   2   3          4
            # pi, px, pv, pix_width, centroid_pos
            if p[2] > max_v:
                max_v = p[2]
                central = p[4]

        if update_self:
            self.central = central

        log.info("Central wavelength = %f" %central)

        return central

    def classify(self,wavelengths = None,values = None,central = None):
        #for now, just with additional lines
        #todo: later add in continuum
        #todo: later add in bayseian stuff
        del self.solutions[:]
        if (wavelengths is not None) and (values is not None) and (central is not None):
            self.set_spectra(wavelengths,values,central)
        else:
            wavelengths = self.wavelengths
            values = self.values
            central = self.central

        #if central wavelength not provided, find the peaks and use the largest
        #for now, largest means highest value
        if (central is None) or (central == 0.0):
            central = self.find_central_wavelength(wavelengths,values)

        if (central is None) or (central == 0.0):
            log.warning("Cannot classify. No central wavelength specified or found.")
            return []

        solutions = self.classify_with_additional_lines(wavelengths,values,central)
        self.solutions = solutions

        return solutions

    def classify_with_additional_lines(self,wavelengths = None,values = None,central = None):
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
            central = self.central

        solutions = []
        total_score = 0.0 #sum of all scores (use to represent each solution as fraction of total score)


        #todo:
        #for each self.emission_line
        #   run down the list of remianing self.emission_lines and calculate score for each
        #   make a copy of each emission_line, set the score, save to the self.lines list []
        #
        #sort solutions by score

        max_w = max(wavelengths)
        min_w = min(wavelengths)

        for e in self.emission_lines:

            if (central/e.w_rest - 1.0) < 0.0:
                continue #impossible, can't have a negative z

            sol = Classifier_Solution()
            sol.z = central/e.w_rest - 1.0
            sol.central_rest = e.w_rest
            sol.emission_line = copy.deepcopy(e)
            sol.emission_line.w_obs = sol.emission_line.w_rest*(1.0 + sol.z)
            sol.emission_line.solution = True

            for a in self.emission_lines:
                if e == a:
                    continue

                a_central = a.w_rest*(sol.z+1.0)
                if (a_central > max_w) or (a_central < min_w):
                    continue

                eli = signal_score(wavelengths, values,a_central )

                if (eli is not None) and (eli.score > 0.0):
                    total_score += eli.score
                    sol.score += eli.score
                    l = copy.deepcopy(a)
                    l.w_obs = l.w_rest * (1.0 + sol.z)
                    l.score = eli.score
                    sol.lines.append(l)

            if sol.score > 0.0:
                solutions.append(sol)

        for s in solutions:
            s.frac_score = s.score/total_score

        #sort by score
        solutions.sort(key=lambda x: x.score, reverse=True)


        for s in solutions:
            msg = "%s, Frac = %0.2f, Score = %0.1f, z = %0.5f, obs_w = %0.1f, rest_w = %0.1f" \
                    % (s.emission_line.name,s.frac_score, s.score,s.z, s.central_rest*(1.0+s.z),s.central_rest )
            log.info(msg)
            # todo: remove ... temporary
            print(msg)



        return solutions



    def build_full_width_spectrum(self, counts = None, wavelengths = None, central_wavelength = None,
                                  show_skylines=True, show_peaks = True, name=None,
                                  dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0):


        use_internal = False
        if (counts is None) or (wavelengths is None) or (central_wavelength is None):
            counts = self.values
            wavelengths = self.wavelengths
            central_wavelength = self.central
            use_internal = True


        # fig = plt.figure(figsize=(5, 6.25), frameon=False)
        fig = plt.figure(figsize=(8, 3), frameon=False)
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
            specplot.step(wavelengths, counts,  where='mid', lw=1)

            specplot.axis([left, right, mn - ran / 20, mx + ran / 20])
            yl, yh = specplot.get_ylim()

            specplot.locator_params(axis='y', tight=True, nbins=4)


            if show_peaks:
                #emistab.append((pi, px, pv,pix_width,centroid))
                peaks = peakdet(wavelengths, counts,dw,h,dh,zero)

                #scores = []
                #for p in peaks:
                #    scores.append(signal_score(wavelengths, counts, p[1]))

                #for i in range(len(scores)):
                #    print(peaks[i][0],peaks[i][1], peaks[i][2], peaks[i][3], peaks[i][4], scores[i])

                if (peaks is not None) and (len(peaks) > 0):
                    specplot.scatter(np.array(peaks)[:, 1], np.array(peaks)[:, 2], facecolors='none', edgecolors='r')

                    for i in range(len(peaks)):
                        h = peaks[i][2]
                        specplot.annotate(str(peaks[i][5]),xy=(peaks[i][1],h),xytext=(peaks[i][1],h),fontsize=6)

                        log.debug("Peak at: %f , Score = %f" %(peaks[i][1],peaks[i][5]))



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

                rec = plt.Rectangle((central_wavelength - 20.0, yl), 2 * 20.0, yh - yl, fill=True, lw=1, color='y', zorder=1)
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
                plt.savefig(name+".png", format='png', dpi=300)
            except:
                log.warning("Unable save plot to file.", exc_info=True)


        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf


