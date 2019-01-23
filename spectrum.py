import global_config as G
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
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
import copy
import line_prob
import mcmc_gauss
import os.path as op


#log = G.logging.getLogger('spectrum_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('spectrum_logger')
log.setlevel(G.logging.DEBUG)

#these are for the older peak finder (based on direction change)
MIN_FWHM = 2 #AA (must xlat to pixels)
MIN_ELI_SNR = 3.0 #bare minium SNR to even remotely consider a signal as real
MIN_ELI_SIGMA = 1.0 #bare minium (expect this to be more like 2+)
MIN_HEIGHT = 10
MIN_DELTA_HEIGHT = 2 #to be a peak, must be at least this high above next adjacent point to the left
DEFAULT_BACKGROUND = 6.0
DEFAULT_BACKGROUND_WIDTH = 100.0 #pixels
DEFAULT_MIN_WIDTH_FROM_CENTER_FOR_BACKGROUND = 10.0 #pixels


GAUSS_FIT_MAX_SIGMA = 17.0 #maximum width (pixels) for fit gaussian to signal (greater than this, is really not a fit)
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
GAUSS_SNR_SIGMA = 5.0 #check at least these pixels (pix*sigma) to either side of the fit line for SNR
                      # (larger of this or GAUSS_SNR_NUM_AA) *note: also capped to a max of 40AA or so (the size of the
                      # 'cutout' of the signal (i.e. GAUSS_FIT_AA_RANGE)
GAUSS_SNR_NUM_AA = 5.0 #check at least this num of AA to either side (2x +1 total) of the fit line for SNR in gaussian fit
                       # (larger of this or GAUSS_SNR_SIGMA


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


#beyond an okay fit (see GAUSS_FIT_xxx above) is this a "good" signal
GOOD_MIN_LINE_SCORE = 2.0 #lines are added to solution only if 'GOOD' (meaning, minimally more likely real than noise)
#does not have to be the same as PROB_NOISE_MIN_SCORE, but that generally makes sense
#GOOD_FULL_SNR = 9.0 #ignore SBR is SNR is above this
#GOOD_MIN_SNR = 5.0 #bare-minimum; if you change the SNR ranges just above, this will also need to change
#GOOD_MIN_SBR = 3.0 #signal to "background" noise (looks at peak height vs surrounding peaks) (only for "weak" signals0
GOOD_MIN_SIGMA = 1.425 #in AA, roughly # 0.75 * pixel_size
#GOOD_MIN_EW_OBS = 1.5 #not sure this is a good choice ... really should depend on the physics of the line and
                      # not be absolute
#GOOD_MIN_EW_REST = 1.0 #ditto here

#GOOD_MIN_LINE_FLUX = 5.0e-18 #todo: this should be the HETDEX flux limit (but that depends on exposure time and wavelength)
#combined in the score
GOOD_MAX_DX0_MULT = 1.75 #3.8 (AA) ... now 1.75 means 1.75 * pixel_size
                    # #maximum error (domain freedom) in fitting to line center in AA
                    #since this is based on the fit of the extra line AND the line center error of the central line
                    #this is a compound error (assume +/- 2 AA since ~ 1.9AA/pix for each so at most 4 AA here)?
#GOOD_MIN_H_CONT_RATIO = 1.33 #height of the peak must be at least 33% above the continuum fit level
ADDL_LINE_SCORE_BONUS = 5.0 #add for each line at 2+ lines (so 1st line adds nothing)
                            #this is rather "hand-wavy" but gives a nod to having more lines beyond just their score
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




def fit_line(wavelengths,values,errors=None):
#super simple line fit ... very basic
#rescale x so that we start at x = 0
    coeff = np.polyfit(wavelengths,values,deg=1)

    #flip the array so [0] = 0th, [1] = 1st ...
    coeff = np.flip(coeff,0)

    if False: #just for debug
        fig = plt.figure(figsize=(8, 2), frameon=False)
        line_plot = plt.axes()
        line_plot.plot(wavelengths, values, c='b')

        x_vals = np.array(line_plot.get_xlim())
        y_vals = coeff[0] + coeff[1] * x_vals
        line_plot.plot(x_vals, y_vals, '--',c='r')

        fig.tight_layout()
        fig.savefig("line.png")
        fig.clear()
        plt.close()
        # end plotting
    return coeff


def invert_spectrum(wavelengths,values):
    # subtracting from the maximum value inverts the slope also, and keeps the overall shape intact
    # subtracting from the line fit slope flattens out the slope (leveling out the continuum) and changes the overall shape
    #
    #coeff = fit_line(wavelengths,values)
    #inverted = coeff[1]*wavelengths+coeff[0] - values

    mx = np.max(values)
    inverted = mx - values

    if False: #for debugging
        if not 'coeff' in locals():
            coeff = [mx, 0]

        fig = plt.figure(figsize=(8, 2), frameon=False)
        line_plot = plt.axes()
        line_plot.plot(wavelengths, values, c='g',alpha=0.5)
        x_vals = np.array(line_plot.get_xlim())
        y_vals = coeff[0] + coeff[1] * x_vals
        line_plot.plot(x_vals, y_vals, '--', c='b')

        line_plot.plot(wavelengths, inverted, c='r' ,lw=0.5)
        fig.tight_layout()
        fig.savefig("inverted.png")
        fig.clear()
        plt.close()


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
    idx = (np.abs(array-value)).argmin()
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
        self.sn_pix = 0 #total number of pixels used to calcualte the SN (kind of like a width in pixels)

        #!! Important: raw_wave, etc is NOT of the same scale or length of fit_wave, etc
        self.raw_wave = []
        self.raw_vals = []
        self.raw_h =  None
        self.raw_x0 = None

        self.line_flux = -999 #the line flux
        self.cont = -999

        self.snr = 0.0
        self.sbr = 0.0
        self.eqw_obs = -999
        self.fwhm = -999
        self.score = None
        self.raw_score = None

        self.line_score = None
        self.prob_noise = 1.0

        #MCMC errors and info
        # 3-tuples [0] = fit, [1] = fit +16%,  [2] = fit - 16% (i.e. ~ +/- 1 sd ... the interior 66%)
        self.mcmc_x0 = None #aka mu
        self.mcmc_sigma = None
        self.mcmc_a = None #area
        self.mcmc_y = None
        self.mcmc_ew_obs = None #calcuated value (using error propogation from mcmc_a and mcmc_y)
        self.mcmc_snr = None


        self.absorber = False #set to True if this is an absorption line

        self.mcmc_plot_buffer = None
        self.gauss_plot_buffer = None



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
            log.warning("Exception in EmissionLineInfo::flux_unc()", exc_info=True)

        return s

    @property
    def flux_unc(self):
        #return a string with flux uncertainties in place
        return self.unc_str(self.mcmc_a)

    @property
    def cont_unc(self):
        #return a string with flux uncertainties in place
        return self.unc_str(self.mcmc_y)


    @property
    def eqw_lya_unc(self):
        #return a string with flux uncertainties in place
        s = ""
        try:
           # ew = np.array(self.mcmc_ew_obs)/(self.fit_x0 / G.LyA_rest) #reminder this is 1+z
           # s  =  "%0.2g($\pm$%0.2g)" %(ew[0],(0.5 * (abs(ew[1]) + abs(ew[2]))))

            #more traditional way
            ew = self.mcmc_a[0] / self.mcmc_y[0] /(self.fit_x0 / G.LyA_rest)
            a_unc = 0.5 * (abs(self.mcmc_a[1])+abs(self.mcmc_a[2]))
            y_unc = 0.5 * (abs(self.mcmc_y[1])+abs(self.mcmc_y[2]))

            #wrong!! missing the abs(ew) and the ratios inside are flipped
            #ew_unc = np.sqrt((self.mcmc_a[0]/a_unc)**2 + (self.mcmc_y[0]/y_unc)**2)

            ew_unc = abs(ew) * np.sqrt((a_unc/self.mcmc_a[0])**2 + (y_unc/self.mcmc_y[0])**2)

            s = "%0.2g($\pm$%0.2g)" % (ew, ew_unc)


        except:
            log.warning("Exception in eqw_lya_unc",exc_info=True)

        return s

    def build(self,values_units=0):
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

                    self.line_flux = self.fit_a * unit
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

            #line_flux is now in erg/s/... the 1e17 scales up to reasonable numbers (easier to deal with)
            #we are technically penalizing a for large variance, but really this is only to weed out
            #low and wide fits that are probably not reliable

            #and penalize for large deviation from the highest point and the line (gauss) mean (center) (1.9 ~ pixel size)
            #self.line_score = self.snr * self.line_flux * 1e17 / (2.5 * self.fit_sigma * (1. + abs(self.fit_dx0/1.9)))

            #penalize for few pixels (translated to angstroms ... anything less than 21 AA total)
            #penalize for too narrow sigma (anything less than 1 pixel

            #the 10.0 is just to rescale ... could make 1e17 -> 1e16, but I prefer to read it this way

            self.line_score = self.snr * self.line_flux * 1e17 * \
                              min(self.fit_sigma/self.pix_size,1.0) * \
                              min((self.pix_size * self.sn_pix)/21.0,1) / \
                              (10.0 * (1. + abs(self.fit_dx0 / self.pix_size)) )

            if self.absorber:
                if G.MAX_SCORE_ABSORPTION_LINES: #if not scoring absorption, should never actually get here ... this is a safety
                    # as hand-wavy correction, reduce the score as an absorber
                    # to do this correctly, need to NOT invert the values and treat as a proper absorption line
                    #   and calucate a true flux and width down from continuum
                    new_score = min(G.MAX_SCORE_ABSORPTION_LINES, self.line_score * ABSORPTION_LINE_SCORE_SCALE_FACTOR)
                    log.info("Rescalling line_score for absorption line: %f to %f" %(self.line_score,new_score))
                    self.line_score = new_score
                else:
                    log.info("Zeroing line_score for absorption line.")
                    self.line_score = 0.0
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


    def is_good(self,z=0.0):
        #(self.score > 0) and  #until score can be recalibrated, don't use it here
        #(self.sbr > 1.0) #not working the way I want. don't use it
        result = False

        # minimum to be possibly good
        if (self.line_score >= GOOD_MIN_LINE_SCORE) and (self.fit_sigma >= GOOD_MIN_SIGMA):
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



#really should change this to use kwargs
def signal_score(wavelengths,values,errors,central,central_z = 0.0, spectrum=None,values_units=0, sbr=None,
                 min_sigma=GAUSS_FIT_MIN_SIGMA,show_plot=False,plot_id=None,plot_path=None,do_mcmc=False,absorber=False,
                 force_score=False):

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


    if (wavelengths is None) or (values is None) or (len(wavelengths)==0) or (len(values)==0):
        log.warning("Zero length (or None) spectrum passed to spectrum::signal_score().")
        return None


    accept_fit = False
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
        min_idx = max(0, idx - wave_side/2)
        max_idx = min(len_array, idx + wave_side/2)
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
    peak_pos = getnearpos(wavelengths, central)

    try:
        # find the highest point in the raw data inside the range we are allowing for the line center fit
        dpix = int(round(fit_range_AA / pix_size))
        raw_peak = max(values[peak_pos-dpix:peak_pos+dpix+1])
        if raw_peak <= 0:
            log.warning("Spectrum::signal_score invalid raw peak %f" %raw_peak)
            return None
    except:
        #this can fail if on very edge, but if so, we would not use it anyway
        log.info("Raw Peak value failure for wavelength (%f) at index (%d). Cannot fit to gaussian. " %(central,peak_pos))
        return None

    fit_peak = None

    eli = EmissionLineInfo()
    eli.absorber = absorber
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

        #if narrow_wave_err_sigma is None:
        #    print("**** NO UNCERTAINTIES ****")
        #   log.warning("**** NO UNCERTAINTIES ****")

        parm, pcov = curve_fit(gaussian, narrow_wave_x, narrow_wave_counts,
                                p0=(central,1.5,1.0,0.0),
                                bounds=((central-fit_range_AA, min_sigma, 0.0, -100.0),
                                        (central+fit_range_AA, np.inf, np.inf, np.inf)),
                                #sigma=1./(narrow_wave_errors*narrow_wave_errors)
                                sigma=narrow_wave_err_sigma #handles the 1./(err*err)
                               #note: if sigma == None, then curve_fit uses array of all 1.0
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
        if raw_idx < 3:
            raw_idx = 3

        if raw_idx > len(eli.raw_vals)-4:
            raw_idx = len(eli.raw_vals)-4
        #if still out of range, will throw a trapped exception ... we can't use this data anyway
        eli.raw_h = max(eli.raw_vals[raw_idx - 3:raw_idx + 4])
        eli.raw_x0 = eli.raw_wave[getnearpos(eli.raw_vals, eli.raw_h)]

        fit_peak = max(eli.fit_vals)

        if ( abs(fit_peak - raw_peak) > (raw_peak * 0.25) ):
        #if (abs(raw_peak - fit_peak) / raw_peak > 0.2):  # didn't capture the peak ... bad, don't calculate anything else
            #log.warning("Failed to capture peak")
            log.debug("Failed to capture peak: raw = %f , fit = %f, frac = %0.2f" % (raw_peak, fit_peak,
                                                                                 abs(raw_peak - fit_peak) / raw_peak))
        else:
            #check the dx0

            p_err = pix_error(central_z,eli.fit_x0,pix_size=pix_size)
            if (abs(eli.fit_dx0) > (GOOD_MAX_DX0_MULT * pix_size + p_err)):
                log.debug("Failed to capture peak: dx0 = %f, pix_size = %f, wavelength = %f, pix_z_err = %f"
                          % (eli.fit_dx0,pix_size, eli.fit_x0,p_err))
            else:
                accept_fit = True
                log.debug("Success: captured peak: raw = %f , fit = %f, frac = %0.2f"
                          % (raw_peak, fit_peak, abs(raw_peak - fit_peak) / raw_peak))

                num_sn_pix = int(round(max(GAUSS_SNR_SIGMA * eli.fit_sigma, GAUSS_SNR_NUM_AA)/pix_size)) #half-width in AA
                num_sn_pix = int(round(min(num_sn_pix,len(wave_counts)/2 - 1))) #don't go larger than the actual array

                #?rms just under the part of the plot with signal (not the entire fit part) so, maybe just a few AA or pix
                eli.fit_norm_rmse = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, eli.fit_x0 ), hw_pix=num_sn_pix,
                             norm=True)
                eli.fit_rmse = rms(wave_counts, rms_wave, cw_pix=getnearpos(wave_x, eli.fit_x0 ), hw_pix=num_sn_pix,
                             norm=False)

                num_sn_pix = num_sn_pix * 2 + 1 #need full width later (still an integer)

                eli.sn_pix = num_sn_pix
    except Exception as ex:
        if ex.message.find("Optimal parameters not found") > -1:
            log.info("Could not fit gaussian near %f" % central,exc_info=False)
        else:
            log.error("Could not fit gaussian near %f" % central, exc_info=True)
        return None

    if (eli.fit_rmse > 0) and (eli.fit_sigma <= GAUSS_FIT_MAX_SIGMA) and (eli.fit_sigma >= min_sigma):

        #this snr makes sense IF we assume the noise is distributed as a gaussian (which is reasonable)
        #then we'd be looking at something like 1/N * Sum (sigma_i **2) ... BUT , there are so few pixels
        #  typically around 10 and there really should be at least 30  to approximate the gaussian shape
        eli.snr = eli.fit_a/(np.sqrt(num_sn_pix)*eli.fit_rmse)
        eli.build(values_units=values_units)
        #eli.snr = max(eli.fit_vals) / (np.sqrt(num_sn_pix) * eli.fit_rmse)
        snr = eli.snr
    else:
        accept_fit = False
        snr = 0.0
        eli.snr = 0.0
        eli.line_score = 0.0
        eli.line_flux = 0.0

    log.debug("SNR at %0.2f = %0.2f"%(central,snr))

    title = ""

    #todo: re-calibrate to use SNR instead of SBR ??
    sbr = snr
    if sbr is None:
        sbr = est_peak_strength(wavelengths,values,central,values_units)
        if sbr is None:
            #done, no reason to continue
            log.warning("Could not determine SBR at wavelength = %f. Will use SNR." %central)
            sbr = snr

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

    mcmc = None
    if do_mcmc:
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
        mcmc.run_mcmc()

        # 3-tuple [0] = fit, [1] = fit +16%,  [2] = fit - 16%
        eli.mcmc_x0 = mcmc.mcmc_mu
        eli.mcmc_sigma = mcmc.mcmc_sigma
        eli.mcmc_snr = mcmc.mcmc_snr

        if mcmc.mcmc_A is not None:
            eli.mcmc_a = np.array(mcmc.mcmc_A)
        else:
            eli.mcmc_a = np.array((0.,0.,0.))

        if mcmc.mcmc_y is not None:
            eli.mcmc_y = np.array(mcmc.mcmc_y)
        else:
            eli.mcmc_y = np.array((0.,0.,0.))

        if values_units < 0:
            eli.mcmc_a *= 10**values_units
            eli.mcmc_y *= 10**values_units

        #no ... this is wrong ... its all good now
        # if values_units == -18:  # converted from e-17, but this is an area so there are 2 factors
        #     eli.mcmc_a = tuple(np.array(eli.mcmc_a ) / [10., 1., 1.])

        # calc EW and error with approximate symmetric error on area and continuum
        if eli.mcmc_y[0] != 0 and eli.mcmc_a[0] != 0:
            ew = abs(eli.mcmc_a[0] / eli.mcmc_y[0])
            ew_err = ew * np.sqrt((mcmc.approx_symmetric_error(eli.mcmc_a) / eli.mcmc_a[0]) ** 2 +
                                  (mcmc.approx_symmetric_error(eli.mcmc_y) / eli.mcmc_y[0]) ** 2)
        else:
            ew = eli.mcmc_a[0]
            ew_err = mcmc.approx_symmetric_error(eli.mcmc_a)


        eli.mcmc_ew_obs = [ew, ew_err, ew_err]
        log.info("MCMC Peak height = %f" % (max(narrow_wave_counts)))
        log.info("MCMC calculated EW_obs for main line = %0.3g +/- %0.3g" % (ew, ew_err))


    if show_plot or G.DEBUG_SHOW_GAUSS_PLOTS:# or eli.snr > 40.0:
        if error is None:
            error = -1

        g = eli.is_good(z=central_z)
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
                 "Line Score = %0.2f , SNR = %0.2f (%0.1f) , wpix = %d\n" \
                 "Peak = %0.2g, Line(A) = %0.2g, Cont = %0.2g, EqW_Obs=%0.2f\n"\
                 "dX0 = %0.2f, RH = %0.2f, RMS = %0.2f (%0.2f) \n"\
                 "Sigma = %0.2f, Skew = %0.2f, Kurtosis = %0.2f"\
                  % (eli.fit_x0,central_z,a,g,line_type,eli.line_score,
                     snr,signal_calc_scaled_score(snr),num_sn_pix,
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

    if accept_fit:
        eli.raw_score = score
        eli.score = signal_calc_scaled_score(score)
        return eli
    else:
        log.info("Fit rejected")
        return None




def run_mcmc(eli,wavelengths,values,errors,central,values_units):

    err_units = values_units  # assumed to be in the same units
    values, values_units = norm_values(values, values_units)
    if errors is not None and (len(errors) == len(values)):
        errors, err_units = norm_values(errors, err_units)

    pix_size = abs(wavelengths[1] - wavelengths[0])  # aa per pix
    wave_side = int(round(GAUSS_FIT_AA_RANGE / pix_size))  # pixels
    fit_range_AA = max(GAUSS_FIT_PIX_ERROR * pix_size, GAUSS_FIT_AA_ERROR)

    len_array = len(wavelengths)
    idx = getnearpos(wavelengths, central)
    min_idx = max(0, idx - wave_side)
    max_idx = min(len_array, idx + wave_side)
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
        log.info(
            "Raw Peak value failure for wavelength (%f) at index (%d). Cannot fit to gaussian. " % (central, peak_pos))
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
    else:
        eli.mcmc_a = np.array((0., 0., 0.))

    if mcmc.mcmc_y is not None:
        eli.mcmc_y = np.array(mcmc.mcmc_y)
    else:
        eli.mcmc_y = np.array((0., 0., 0.))

    if values_units < 0:
        eli.mcmc_a *= 10 ** values_units
        eli.mcmc_y *= 10 ** values_units

    # calc EW and error with approximate symmetric error on area and continuum
    if eli.mcmc_y[0] != 0 and eli.mcmc_a[0] != 0:
        ew = abs(eli.mcmc_a[0] / eli.mcmc_y[0])
        ew_err = ew * np.sqrt((mcmc.approx_symmetric_error(eli.mcmc_a) / eli.mcmc_a[0]) ** 2 +
                              (mcmc.approx_symmetric_error(eli.mcmc_y) / eli.mcmc_y[0]) ** 2)
    else:
        ew = eli.mcmc_a[0]
        ew_err = mcmc.approx_symmetric_error(eli.mcmc_a)

    eli.mcmc_ew_obs = [ew, ew_err, ew_err]
    log.info("MCMC Peak height = %f" % (max(narrow_wave_counts)))
    log.info("MCMC calculated EW_obs for main line = %0.3g +/- %0.3g" % (ew, ew_err))



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
#def simple_peaks(x,v,h=MIN_HEIGHT,delta_v=2.0,values_units=0):
def simple_peaks(x, v, h=None, delta_v=None, values_units=0):
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


def peakdet(x,v,err=None,dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0,values_units=0,
            enforce_good=True,min_sigma=GAUSS_FIT_MIN_SIGMA,absorber=False):

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

    if (v is None) or (len(v) < 3):
        return []

    maxtab = []
    mintab = []
    emistab = []
    eli_list = []
    delta = dh

    if x is None:
        x = np.arange(len(v))

    pix_size = abs(x[1] - x[0])  # aa per pix
    if pix_size == 0:
        log.error("Unexpected pixel_size in spectrum::peakdet(). Wavelength step is zero.")
        return []
    # want +/- 20 angstroms
    wave_side = int(round(20.0 / pix_size))  # pixels

    dw = int(dw / pix_size) #want round down (i.e. 2.9 -> 2) so this is fine

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

    #don't need to normalize errors for peakdet ... will be handled in signal_score
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


    ################
    #DEBUG
    ################

    if False:
        so = Spectrum()
        eli = []
        for p in maxtab:
            e = EmissionLineInfo()
            e.raw_x0 = p[1] #xposition p[0] is the index
            e.raw_h = v_0[p[0]+2] #v_0[getnearpos(x_0,p[1])]
            eli.append(e)

        so.build_full_width_spectrum(wavelengths=x_0, counts=v_0, errors=None, central_wavelength=0,
                                      show_skylines=False, show_peaks=True, name="peaks",
                                      dw=MIN_FWHM, h=MIN_HEIGHT, dh=MIN_DELTA_HEIGHT, zero=0.0,peaks=eli,annotate=False)



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
            eli = signal_score(x_0, v_0, err, px,values_units=values_units_0,min_sigma=min_sigma,absorber=absorber)

            #if (eli is not None) and (eli.score > 0) and (eli.snr > 7.0) and (eli.fit_sigma > 1.6) and (eli.eqw_obs > 5.0):
            if (eli is not None) and ((not enforce_good) or eli.is_good()):
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

    ################
    #DEBUG
    ################
    if False:
        so = Spectrum()
        eli = []
        for p in eli_list:
            e = EmissionLineInfo()
            e.raw_x0 = p.raw_x0
            e.raw_h = p.raw_h / 10.0
            eli.append(e)
        so.build_full_width_spectrum(wavelengths=x_0, counts=v_0, errors=None, central_wavelength=0,
                                     show_skylines=False, show_peaks=True, name="peaks_trimmed",
                                     dw=MIN_FWHM, h=MIN_HEIGHT, dh=MIN_DELTA_HEIGHT, zero=0.0, peaks=eli,
                                     annotate=False)

    return eli_list


class EmissionLine():
    def __init__(self,name,w_rest,plot_color,solution=True,display=True,z=0,score=0.0):
        self.name = name
        self.w_rest = w_rest
        self.w_obs = w_rest * (1.0 + z)
        self.z = z
        self.color = plot_color
        self.solution = solution #True = can consider this as the target lines
        self.display = display #True = plot label on full 1D plot

        #can be filled in later if a specific instance is created and a model fit to it
        self.score = score
        self.snr = None
        self.sbr = None
        self.flux = None
        self.eqw_obs = None
        self.eqw_rest = None
        self.sigma = None #gaussian fit sigma

        #a bit redundant with EmissionLineInfo
        self.line_score = 0.0
        self.prob_noise = 1.0

        self.absorber = False #true if an abosrption line

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

        self.prob_noise = 1.0
        self.lines = [] #list of EmissionLine

    @property
    def prob_real(self):
        #return min(1-self.prob_noise,0.999) * min(1.0, max(0.67,self.score/G.MULTILINE_MIN_SOLUTION_SCORE))
        return min(1 - self.prob_noise, 0.999) * min(1.0, float(len(self.lines))/(G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY + 1.0))


class Spectrum:
    """
    helper functions for spectra
    actual spectra data is kept in fiber.py
    """

    def __init__(self):
        #reminder ... colors don't really matter (are not used) if solution is not True)
        #try to keep the name in 4 characters
        w = 4

        self.emission_lines = [
            #extras for HW
            # EmissionLine("H$\\alpha$".ljust(w), 6562.8, "blue"),
            # EmissionLine("NaII".ljust(w),6549.0,"lightcoral",solution=True, display=True),
            # EmissionLine("NaII".ljust(w),6583.0,"lightcoral",solution=True, display=True),
            # EmissionLine("Pa$\\beta$".ljust(w),12818.0,"lightcoral",solution=True, display=True),
            # EmissionLine("Pa$\\alpha$".ljust(w),18751.0,"lightcoral",solution=True, display=True),


            EmissionLine("Ly$\\alpha$".ljust(w), G.LyA_rest, 'red'),

            EmissionLine("OII".ljust(w), G.OII_rest, 'green'),
            EmissionLine("OIII".ljust(w), 4960.295, "lime"),
            EmissionLine("OIII".ljust(w), 5008.240, "lime"),

            EmissionLine("CIV".ljust(w), 1549.48, "blueviolet",display=False),  # big in AGN
            EmissionLine("CIII".ljust(w), 1908.734, "purple",display=False),  #big in AGN
            EmissionLine("CII".ljust(w),  2326.0, "purple",solution=False,display=False),  # in AGN

            EmissionLine("MgII".ljust(w), 2799.117, "magenta",display=False),  #big in AGN


            EmissionLine("H$\\beta$".ljust(w), 4862.68, "blue"),
            EmissionLine("H$\\gamma$".ljust(w), 4341.68, "royalblue"),
            EmissionLine("H$\\delta$".ljust(w), 4102, "royalblue", solution=False,display=False),
            EmissionLine("H$\\epsilon$/CaII".ljust(w), 3970, "royalblue", solution=False,display=False), #very close to CaII(3970)
            EmissionLine("H$\\zeta$".ljust(w), 3889, "royalblue", solution=False,display=False),
            EmissionLine("H$\\eta$".ljust(w), 3835, "royalblue", solution=False,display=False),

            EmissionLine("NV".ljust(w), 1240.81, "teal", solution=False,display=False),

            EmissionLine("SiII".ljust(w), 1260, "gray", solution=False,display=False),

            EmissionLine("HeII".ljust(w), 1640.4, "orange", solution=False,display=False),

            EmissionLine("NeIII".ljust(w), 3869, "pink", solution=False,display=False),
            EmissionLine("NeIII".ljust(w), 3967, "pink", solution=False,display=False),  #very close to CaII(3970)
            EmissionLine("NeV".ljust(w), 3346.79, "pink", solution=False,display=False),
            EmissionLine("NeVI".ljust(w), 3426.85, "pink", solution=False, display=False),

            EmissionLine("NaI".ljust(w),4980,"lightcoral",solution=False, display=False),  #4978.5 + 4982.8
            EmissionLine("NaI".ljust(w),5153,"lightcoral",solution=False, display=False),  #5148.8 + 5153.4

            #stars
            EmissionLine("CaII".ljust(w), 3935, "skyblue", solution=False, display=False)

            #merged CaII(3970) with H\$epsilon$(3970)
            #EmissionLine("CaII".ljust(w), 3970, "skyblue", solution=False, display=False)  #very close to NeIII(3967)
           ]

        self.wavelengths = []
        self.values = [] #could be fluxes or counts or something else ... right now needs to be counts
        self.errors = []
        self.values_units = 0

        # very basic info, fit line to entire spectrum to see if there is a general slope
        #useful in identifying stars (kind of like a color U-V (ish))
        self.spectrum_linear_coeff = None #index = power so [0] = onstant, [1] = 1st .... e.g. mx+b where m=[1], b=[0]

        self.central = None
        self.estflux = None
        self.estflux_unc = None
        self.eqw_obs = None
        self.eqw_obs_unc = None

        self.central_eli = None

        self.solutions = []
        self.all_found_lines = None #EmissionLineInfo objs (want None here ... if no lines, then peakdet returns [])
        self.all_found_absorbs = None


        self.addl_fluxes = []
        self.addl_wavelengths = []
        self.addl_fluxerrs = []
        self.p_lae = None
        self.p_oii = None
        self.p_lae_oii_ratio = None

        self.identifier = None #optional string to help identify in the log
        self.plot_dir = None

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


    def set_spectra(self,wavelengths, values, errors, central, values_units = 0, estflux=None, estflux_unc=None,
                    eqw_obs=None, eqw_obs_unc=None, fit_min_sigma=GAUSS_FIT_MIN_SIGMA):
        self.wavelengths = []
        self.values = []
        self.errors = []
        self.all_found_lines = None
        self.all_found_absorbs = None
        self.solutions = None
        self.central_eli = None

        if central is None:
            self.wavelengths = wavelengths
            self.values = values
            self.errors = errors
            self.values_units = values_units
            self.central = central
            return

        #run MCMC on this one ... the main line
        try:

            if self.identifier is None and self.plot_dir is None:
                show_plot = False #intermediate call, not the final
            else:
                show_plot = True

            eli = signal_score(wavelengths=wavelengths, values=values, errors=errors,central=central,
                               values_units=values_units, sbr=None, min_sigma=fit_min_sigma,
                               show_plot=show_plot,plot_id=self.identifier,
                               plot_path=self.plot_dir,do_mcmc=True)
        except:
            log.error("Exception in spectrum::set_spectra calling signal_score().",exc_info=True)
            eli = None

        if eli:
            if (estflux is None) or (eqw_obs is None):
                #basically ... if I did not get this from Karl, use my own measure
                if (eli.mcmc_a is not None) and (eli.mcmc_y is not None):
                    a_unc = 0.5 * (abs(eli.mcmc_a[1]) + abs(eli.mcmc_a[2]))
                    y_unc = 0.5 * (abs(eli.mcmc_y[1]) + abs(eli.mcmc_y[2]))

                    estflux = eli.mcmc_a[0]
                    estflux_unc = a_unc

                    eqw_obs = abs(eli.mcmc_a[0] / eli.mcmc_y[0])
                    eqw_obs_unc = abs(eqw_obs) * np.sqrt((a_unc / eli.mcmc_a[0]) ** 2 + (y_unc / eli.mcmc_y[0]) ** 2)
                else: #not from mcmc, so we have no error
                    estflux = eli.line_flux
                    estflux_unc = 0.0
                    eqw_obs = eli.eqw_obs
                    eqw_obs_unc = 0.0



            #if (self.snr is None) or (self.snr == 0):
            #    self.snr = eli.snr

            self.central_eli = copy.deepcopy(eli)

            # get very basic info (line fit)
            coeff = fit_line(wavelengths, values, errors)  # flipped order ... coeff[0] = 0th, coeff[1]=1st
            self.spectrum_linear_coeff = coeff
            log.info("%s Spectrum basic info (mx+b): %f(x) + %f" %(self.identifier,coeff[1],coeff[0]))
            #todo: maybe also a basic parabola? (if we capture an overall peak? like for a star black body peak?
        else:
            log.warning("Warning! Did not successfully compute signal_score on main emission line.")

        self.wavelengths = wavelengths
        self.values = values
        self.errors = errors
        self.values_units = values_units
        self.central = central
        self.estflux = estflux
        self.estflux_unc = estflux_unc
        self.eqw_obs = eqw_obs
        self.eqw_obs_unc = eqw_obs_unc
        #if self.snr is None:
        #    self.snr = 0


    def find_central_wavelength(self,wavelengths = None,values = None, errors=None,values_units=0):
        central = 0.0
        update_self = False
        if (wavelengths is None) or (values is None):
            wavelengths = self.wavelengths
            values = self.values
            values_units = self.values_units
            update_self = True

        #find the peaks and use the largest
        #for now, largest means highest value

        # if values_are_flux:
        #     #!!!!! do not do values *= 10.0 (will overwrite)
        #     # assumes fluxes in e-17 .... making e-18 ~= counts so logic can stay the same
        #     values = values * 10.0

        values,values_units = norm_values(values,values_units)

        #does not need errors for this purpose
        peaks = peakdet(wavelengths,values,errors,values_units=values_units) #as of 2018-06-11 these are EmissionLineInfo objects
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

        self.solutions = []
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
            central = self.find_central_wavelength(wavelengths,values,errors,values_units=values_units)

        if (central is None) or (central == 0.0):
            log.warning("Cannot classify. No central wavelength specified or found.")
            return []

        solutions = self.classify_with_additional_lines(wavelengths,values,errors,central,values_units)
        self.solutions = solutions

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

        #if len(addl_fluxes) > 0:
        self.get_bayes_probabilities(addl_wavelengths=self.addl_wavelengths,addl_fluxes=self.addl_fluxes,
                                     addl_errors=self.addl_fluxerrs)
        #self.get_bayes_probabilities(addl_fluxes=None, addl_wavelengths=None)

        return solutions


    def is_near_a_peak(self,w,aa=4.0): #is the provided wavelength near one of the found peaks (+/- few AA or pixels)

        wavelength = 0.0
        if (self.all_found_lines is None):
            self.all_found_lines = peakdet(self.wavelengths, self.values, self.errors,values_units=self.values_units)

        if self.all_found_lines is None:
            return 0.0

        for f in self.all_found_lines:
            if abs(f.fit_x0 - w) < aa:
                wavelength = f.fit_x0
                break

        return wavelength

    def is_near_absorber(self,w,aa=4.0):#pix_size=1.9): #is the provided wavelength near one of the found peaks (+/- few AA or pixels)

        if not (G.DISPLAY_ABSORPTION_LINES or G.MAX_SCORE_ABSORPTION_LINES):
            return 0

        wavelength = 0.0
        if (self.all_found_absorbs is None):
            self.all_found_absorbs = peakdet(self.wavelengths, invert_spectrum(self.wavelengths,self.values),
                                             self.errors, values_units=self.values_units,absorber=True)
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
            self.all_found_lines = peakdet(wavelengths,values,errors, values_units=values_units)
            if G.DISPLAY_ABSORPTION_LINES or G.MAX_SCORE_ABSORPTION_LINES:
                self.all_found_absorbs = peakdet(wavelengths, invert_spectrum(wavelengths,values),errors,
                                                 values_units=values_units,absorber=True)
                self.clean_absorbers()



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

            central_z = central/e.w_rest - 1.0
            if (central_z) < 0.0:
                if central_z > -0.01:
                    central_z = 0.0
                else:
                    continue #impossible, can't have a negative z

            sol = Classifier_Solution()
            sol.z = central_z
            sol.central_rest = e.w_rest
            sol.name = e.name
            sol.color = e.color
            sol.emission_line = copy.deepcopy(e)
            sol.emission_line.w_obs = sol.emission_line.w_rest*(1.0 + sol.z)
            sol.emission_line.solution = True
            sol.prob_noise = 1.0

            for a in self.emission_lines:
                if e == a:
                    continue

                a_central = a.w_rest*(sol.z+1.0)
                if (a_central > max_w) or (a_central < min_w):
                    continue

                eli = signal_score(wavelengths=wavelengths, values=values, errors=errors, central=a_central,
                                   central_z = central_z, values_units=values_units, spectrum=self,
                                   show_plot=False, do_mcmc=False)


                #try as absorber
                if G.MAX_SCORE_ABSORPTION_LINES and eli is None and self.is_near_absorber(a_central):
                    eli = signal_score(wavelengths=wavelengths, values=invert_spectrum(wavelengths,values), errors=errors, central=a_central,
                                       central_z=central_z, values_units=values_units, spectrum=self,
                                       show_plot=False, do_mcmc=False,absorber=True)

                if (eli is not None) and eli.is_good(z=sol.z):
                    #if this line is too close to another, keep the one with the better score
                    add_to_sol = True
                    for i in range(len(sol.lines)):
                        if abs(sol.lines[i].w_obs - eli.fit_x0) < 10.0:

                            #keep the emission line over the absorption line, regardless of score, if that is the case
                            if sol.lines[i].absorber != eli.absorber:
                                if eli.absorber:
                                    log.info("Emission line too close to absorption line (%s). Removing %s(%01.f) "
                                             "from solution in favor of %s(%0.1f)"
                                        % (self.identifier, a.name, a.w_rest, sol.lines[i].name, sol.lines[i].w_rest))

                                    add_to_sol = False
                                else:
                                    log.info("Emission line too close to absorption line (%s). Removing %s(%01.f) "
                                             "from solution in favor of %s(%0.1f)"
                                        % (self.identifier, sol.lines[i].name, sol.lines[i].w_rest, a.name, a.w_rest))
                                    # remove this solution
                                    total_score -= sol.lines[i].line_score
                                    sol.score -= sol.lines[i].line_score
                                    sol.prob_noise /= sol.lines[i].prob_noise
                                    del sol.lines[i]
                            else: #they are are of the same type, so keep the better score
                                if sol.lines[i].line_score < eli.line_score:
                                    log.info("Lines too close (%s). Removing %s(%01.f) from solution in favor of %s(%0.1f)"
                                             % (self.identifier,sol.lines[i].name, sol.lines[i].w_rest,a.name, a.w_rest))
                                    #remove this solution
                                    total_score -= sol.lines[i].line_score
                                    sol.score -= sol.lines[i].line_score
                                    sol.prob_noise /= sol.lines[i].prob_noise
                                    del sol.lines[i]
                                    break
                                else:
                                    #the new line is not as good so just skip it
                                    log.info("Lines too close (%s). Removing %s(%01.f) from solution in favor of %s(%0.1f)"
                                             % (self.identifier,a.name, a.w_rest,sol.lines[i].name, sol.lines[i].w_rest))
                                    add_to_sol = False
                                    break


                    #now, before we add, if we have not run MCMC on the feature, do so now
                    if add_to_sol:
                        if eli.mcmc_x0 is None:
                            eli = run_mcmc(eli,wavelengths,values,errors,a_central,values_units)

                        #and now validate the MCMC SNR (reminder:  MCMC SNR is line flux (e.g. Area) / (1sigma uncertainty)
                        if eli.mcmc_snr is None:
                            add_to_sol = False
                            log.info("Line (at %f) rejected due to missing MCMC SNR" %(a_central))
                        elif eli.mcmc_snr < G.MIN_MCMC_SNR:
                            add_to_sol = False
                            log.info("Line (at %f) rejected due to poor MCMC SNR (%f)" % (a_central,eli.mcmc_snr))
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
                        l.eqw_obs = eli.eqw_obs
                        l.eqw_rest = l.eqw_obs / (1.0 + l.z)
                        l.flux = eli.line_flux
                        l.sigma = eli.fit_sigma
                        l.line_score = eli.line_score
                        l.prob_noise = eli.prob_noise
                        l.absorber = eli.absorber

                        total_score += eli.line_score  # cumulative score for ALL solutions
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

            if sol.score > 0.0:
                #log.info("Solution p(noise) (%f) from %d additional lines" % (sol.prob_noise, len(sol.lines) - 1))
                #bonus for each extra line over the minimum

                #sol.lines does NOT include the main line (just the extra lines)
                n = len(np.where([l.absorber == False for l in sol.lines])[0])
      #          n = len(sol.lines) + 1 #+1 for the main line
                if  n >= G.MIN_ADDL_EMIS_LINES_FOR_CLASSIFY:
                    bonus =0.5*(n**2 - n)*ADDL_LINE_SCORE_BONUS #could be negative
                    #print("+++++ %s n(%d) bonus(%g)"  %(self.identifier,n,bonus))
                    sol.score += bonus
                    total_score += bonus
                solutions.append(sol)
        #end for e in emission lines

        for s in solutions:
            s.frac_score = s.score/total_score

        #sort by score
        solutions.sort(key=lambda x: x.score, reverse=True)

        for s in solutions:
            ll =""
            for l in s.lines:
                ll += " %s(%0.1f at %0.1f)," %(l.name,l.w_rest,l.w_obs)
            msg = "Possible Solution %s (%0.3f): %s (%0.1f at %0.1f), Frac = %0.2f, Score = %0.1f, z = %0.5f, +lines=%d %s"\
                    % (self.identifier, s.prob_real,s.emission_line.name,s.central_rest,s.central_rest*(1.0+s.z), s.frac_score,
                       s.score,s.z, len(s.lines),ll )
            log.info(msg)
            #
            if G.DEBUG_SHOW_GAUSS_PLOTS:
                print(msg)

        return solutions


    def get_bayes_probabilities(self,addl_wavelengths=None,addl_fluxes=None,addl_errors=None):
        # todo: feed in addl_fluxes from the additonal line solutions (or build specifically)?
        # todo: make use of errors

        #care only about the LAE and OII solutions:
        #todo: find the LyA and OII options in the solution list and use to fill in addl_fluxes?

        ratio, self.p_lae, self.p_oii = line_prob.prob_LAE(wl_obs=self.central,
                                                           lineFlux=self.estflux,
                                                           lineFlux_err=self.estflux_unc,
                                                           ew_obs=self.eqw_obs,
                                                           ew_obs_err=self.eqw_obs_unc,
                                                           c_obs=None, which_color=None,
                                                           addl_wavelengths=addl_wavelengths,
                                                           addl_fluxes=addl_fluxes,
                                                           addl_errors=addl_errors,
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
                        peaks = peakdet(wavelengths,counts,errors, dw,h,dh,zero,values_units=values_units) #as of 2018-06-11 these are EmissionLineInfo objects
                        self.all_found_lines = peaks
                        if G.DISPLAY_ABSORPTION_LINES or G.MAX_SCORE_ABSORPTION_LINES:
                            self.all_found_absorbs = peakdet(wavelengths, invert_spectrum(wavelengths,counts), errors,
                                                             values_units=values_units,absorber=True)
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


