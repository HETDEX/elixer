from __future__ import print_function

try:
    from elixer import global_config as G
    from elixer import corner
    from elixer import utilities
except:
    import global_config as G
    import corner
    import utilities

import numpy as np
import io
import matplotlib.pyplot as plt
import emcee

import warnings

log = G.Global_Logger('mcmc_logger')
log.setlevel(G.LOG_LEVEL)


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

        if cw_pix is None and hw_pix is None:
            pass
        else:
            if cw_pix is None or hw_pix is None:
                cw_pix = len(data)//2
                hw_pix = cw_pix-1

            if not (min_pix < cw_pix < (len(data) - min_pix)):
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

        # np.sqrt(np.sum((f-d)*(f-d))/len(f)) #same thing
        return np.sqrt(((f - d) ** 2).mean())
    except:
        return -1 #non-sense value for snr

class MCMC_Gauss:

    UncertaintyRange = [16,50,84]

    def __init__(self):
        #intial values are meant to be near the truth
        #and are expected to come from, say, some "best" fit
        self.initial_mu = None
        self.initial_sigma = None
        self.initial_A = None #set to a negative value if this is an absorption line
        self.initial_y = None
        self.initial_peak = None

        self.max_sigma = 20.0
        self.range_mu = 5.0
        self.max_A_mult = 2.0
        self.max_y_mult = 2.0
        self.min_y = -10.0 #-100.0 #should this be zero? or some above zero but low limit

        self.data_x = None
        self.data_y = None
        self.err_x = None
        self.err_y = None

        #just for reference ... MCMC itself does not need to know about this
        #the caller DOES though and needs to adjust the line_flux accordingly
        #self.dx = None #original bin width IF NOT part of the data_y already

        #this is mostly a guess ... no great way to automate, but this is pretty quick
        #and since the initials are from a scipy curve fit, we stabablize pretty fast
        self.burn_in = 100
        self.main_run = 1000
        self.walkers = 100

        self.sampler = None #mcmc sampler
        self.samples = None #resulting samples

        self.mcmc_mu = None  #3-tuples [0] = fit, [1] = fit +16%,  [2] = fit - 16%
        self.mcmc_sigma = None
        self.mcmc_A = None
        self.mcmc_y = None

        #not tuple, just single floats
        self.mcmc_snr = None
        self.mcmc_snr_err = 0
        # just for reference ... MCMC itself does not need to know about this
        # the caller DOES though and needs to adjust the line_flux accordingly
        #self.mcmc_line_flux = None #the actual line flux (erg s^-1 cm^-2);
                                   # deals with bin width and input as flux instead of flux/dx

    def approx_symmetric_error(self,parm): #parm is assumed to be a 3 vector as [0] = mean, [1] = +error, [2] = -error

        try:
            if parm is None or (len(parm)!= 3) :
                return None

            p1 = abs(parm[1])
            p2 = abs(parm[2])
            avg = 0.5*(p1+p2)

            if avg == 0:
                return 0

            similarity = abs(p1-p2)/avg

            if similarity > 0.1:
                log.warning("Warning! Asymmetric uncertainty similarity = %0.3g (%0.3g, %0.3g)" %(similarity,p1,p2))

            #for now, do it anyway
            return avg
            #return max(p1,p2)

        except:
            return None


    def noise_model(self):
        #todo: fill in model for the noise
        #some distribution
        #or can this also be solved (fitted)?
        return 0.0

    def compute_model(self,x,mu, sigma, A, y):
        try:
            return A * (np.exp(-np.power((x - mu) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y
        except:
            return np.nan

    def model(self,x,theta):
        mu, sigma, A, y, ln_f = theta #note: not using ln_f here
        if (x is None) or (mu is None) or (sigma is None):
            return None
        try:
            value = self.compute_model(x,mu, sigma, A, y)
            # note: noise is separate and included in the lnlike() function
            #value = A * (np.exp(-np.power((x - mu) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y
        except:
            value = np.nan
        return value

    def lnlike(self, theta, x, y, yerr):
        ln_f = theta[-1] #last parameter in theta
        model = self.model(x, theta)
        noise = self.noise_model()
        diff = y - (model + noise) #or ... do we roll noise in with y_err??
        #sigma2 = (self.err_y ** 2) +  np.exp(2*ln_f) * model**2
            #assumes some additional uncertainties in y based on an underestimation in the model by some factor f
            #err_y are the reported error bars in the y-direction (e.g. the flux)
            #  assuming these are essentially the variance in the generative model
            #  e.g. the model itself is a gaussian that predicts a y value from the x value (and the other parameters)
            #           each y value that is predicted (generated) has a mean and a variance (or standar deviation)
            #          and the error bars can be thought of as due to the variace in the gaussian distribution of y
            # (note: do not confuse the gaussians .... the emission line is also more or less a gaussian, but here
            # I am talking about a gaussian in y (kind of like a little gaussian at each y running along the y-axis so
            # each y "point" is the mean of the gaussian and the vertical error bars are like the 1-sigma width of the
            # y gaussian)
            # notice ... the assumed y_err of 1 ==> sigma == 1 (or rather, sigma**2 == 1) ... eg. a standard normal distro)

        # assume that the (distribution of) errors in y are known and indepentent
        sigma2 = (self.err_y ** 2)
        return -0.5 * (np.sum((diff ** 2) / sigma2 + np.log(sigma2)))

    # if any are zero, the whole prior is zero
    # all priors here are uniformitive ... i.e they are all flat ... either zero or one
    def lnprior(self, theta):  # theta is a n-tuple (_,_,_ ... )
        mu, sigma, A, y, ln_f = theta
        # note: could take some other dynamic maximum for y (like compute the peak ... y can't be greater than that

        if self.initial_A < 0 : #same as emission, but "A" is negative (flip sign) and y is between a max and zero
            if (-self.range_mu < mu - self.initial_mu < self.range_mu) and \
                    (0.0 < sigma < self.max_sigma) and \
                    (self.max_A_mult * self.initial_A < A < 0.0) and \
                    (self.min_y < y < self.max_y_mult * self.initial_peak):
                return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)
        else:
            if (-self.range_mu < mu - self.initial_mu < self.range_mu) and \
                (0.0 < sigma < self.max_sigma) and \
                (0.0 < A < self.max_A_mult * self.initial_A) and \
                (self.min_y < y < self.max_y_mult * self.initial_peak):
                return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)
        return -np.inf  # -999999999 #-np.inf #roughly ln(0) == -inf

    def lnprob(self, theta, x, y, yerr):
        """
        ln(probability)

        :param theta: parameters to check
        :param x:  THE data (x axis or wavelengths, in this case)
        :param y: THE data (y axis or flux counts, in this case)
        :param yerr:  The error on the y axis data flux counts
        :return:
        """
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr)  # again, since logs, this is a sum ln(likelihood) + ln(prior)


    def sanity_check_init(self):
        try:
            #if the error on y is None or if it is all zeros, set to all ones
            if self.err_y is None:
                self.err_y = np.ones(np.shape(self.data_y))
            elif not np.any(self.err_y):
                self.err_y = np.ones(np.shape(self.data_y))

            if self.err_x is None:
                self.err_x = np.ones(np.shape(self.data_x))

            if (self.data_x is None) or (self.data_y is None) or (self.err_y is None):
                return False

            if len(self.data_x) == len(self.data_y) == len(self.err_y): #leave off self.err_x as could be None
                if (self.err_x is not None):
                    if len(self.data_x) != len(self.err_x):
                        return False

                if self.initial_sigma < 0.0: #self.initial_A < 0.0  ... actually, leave A alone .. might allow absorportion later
                    return False

                if self.initial_y > self.initial_peak:
                    return False
            else:
                return False
            return True
        except:
            log.warning("Exception in mcmc_gauss::sanity_check",exc_info=True)
            return False

    def run_mcmc(self):

        if not self.sanity_check_init():
            log.info("Sanity check failed. Cannot conduct MCMC.")
            return False

        result = True

        #here for initial positions of the walkers, sample from narrow gaussian (hence the randn or randNormal)
        #centered on each of the maximum likelihood selected parameter values
        #mu, sigma, A, y, ln_f = theta #note f or ln(f) is another uncertainty ...an underestimation of the variance
        #                               by some factor (f) .... e.g. variance = variance + f * model
        initial_pos = [self.initial_mu,self.initial_sigma,self.initial_A,self.initial_y,0.0]

        #even with the random nudging the pos values must be greater than (or less than for absorption) these values

        #mostly for the A (area)
        if self.initial_A < 0: #absorber
            max_pos = [np.inf, np.inf,     0.0, max(self.data_y),  np.inf]
            min_pos = [   0.0,   0.01, -np.inf,          -np.inf, -np.inf]
        else:
            #here, because of the max check, none mu, sigma, or A will be negative
            max_pos = [np.inf, np.inf,np.inf,max(self.data_y), np.inf] #must be less than this
            min_pos = [   0.0,  0.01,   0.01,         -np.inf,-np.inf] #must be greater than this

        ndim = len(initial_pos)
        scale = np.array([1.,1.,1.,1.,-100.5]) #don't nudge ln_f ...note ln_f = -4.5 --> f ~ 0.01

        #nudge the initial positions around a bit

        try:
            # if self.initial_A < 0: #absorber
            #     pos = [np.minimum(initial_pos + scale * np.random.randn(ndim),limit_pos) for i in range(self.walkers)]
            # else:

            pos = [np.minimum(np.maximum(initial_pos + scale * np.random.randn(ndim),min_pos),max_pos) for i in range(self.walkers)]

            #pos = [initial_pos + scale * np.random.randn(ndim) for i in range(self.walkers)]

            #build the sampler
            #todo: incorporate self.err_x ? (realistically, do we have significant uncertainty in x?)
            self.sampler = emcee.EnsembleSampler(self.walkers, ndim, self.lnprob,
                                            args=(self.data_x,self.data_y, self.err_y))
            #args are the positional args AFTER theta for self.lnprob function

            with warnings.catch_warnings(): #ignore the occassional warnings from the walkers (NaNs, etc that reject step)
                warnings.simplefilter("ignore")
                log.debug("MCMC burn in (%d) ...." %self.burn_in)
                pos, prob, state = self.sampler.run_mcmc(pos, self.burn_in)  # burn in
                log.debug("MCMC main run (%d) ..." %self.main_run)
                pos, prob, state = self.sampler.run_mcmc(pos, self.main_run, rstate0=state)  # start from end position of burn-in

            self.samples = self.sampler.flatchain  # collapse the walkers and interations (aka steps or epochs)

            log.debug("MCMC mean acceptance fraction: %0.3f" %(np.mean(self.sampler.acceptance_fraction)))

            #for each, in order
            #v[0] is the 16 percentile (~ - 1sigma)
            #v[1] is the 50 percentile (so the "average")
            #v[2] is the 84 percentile (~ +1sigma)
            #the tuple reports then as ["average", "84th - average", "average - 16th"]
            #should always be positive (assuming a positive value) BUT when printed to the log, the 3rd value is made
            #to be negative showing that you would subtract it from the average to get to the 16th percentile

            sigma_width = 4

            #using 68% interval
            self.mcmc_mu, self.mcmc_sigma, self.mcmc_A, self.mcmc_y, mcmc_f = \
                map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(self.samples, self.UncertaintyRange,axis=0)))

            # mcmc_mu_95, mcmc_sigma_95, mcmc_A_95, mcmc_y_95, mcmc_f = \
            #     map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(self.samples, [5,50,95],axis=0)))

            try: #basic info used by multiple SNR calculations
                bin_width = self.data_x[1] - self.data_x[0]
                model_fit = self.compute_model(self.data_x,self.mcmc_mu[0],self.mcmc_sigma[0],self.mcmc_A[0],self.mcmc_y[0])
                left,*_ = utilities.getnearpos(self.data_x,self.mcmc_mu[0]-self.mcmc_sigma[0]*sigma_width)
                right,*_ = utilities.getnearpos(self.data_x,self.mcmc_mu[0]+self.mcmc_sigma[0]*sigma_width)
                if left > 0:
                    left -= 1
                if (right+1) < len(self.data_x):
                    right += 1
                rms_err = rms(self.data_y[left:right],model_fit[left:right],None,None,False)

                #signal is the sum of the data between +/- 2.5 sigma (minus the y-offset)
                # noise is the sqrt of the nummber of pixels (wavebins)*the RMSerror)
            except:
                log.warning("Exception calculating MCMC SNR: ", exc_info=True)


            #
            # try:
            #     ###############################
            #     # Different SNR measures
            #     ###############################
            #
            #     #
            #     # 1. sum over the model (subtracting off the y-offset) and divide by sqrt(pixels)*rmse
            #     #
            #     self.mcmc_snr = (np.sum(model_fit[left:right])-(right-left+1)*self.mcmc_y[0])/(np.sqrt(right-left+1)*rms_err)
            #     log.info(f"MCMC SNR model w/RMS: {self.mcmc_snr}")
            # except:
            #     log.warning("Exception calculating MCMC SNR: ", exc_info=True)
            #
            # try:
            #     #
            #     # 2. sum over the model (subtracting off the y-offset) and divide by sum of the sqrt of data errors
            #     #
            #     if self.err_y is not None and len(self.err_y) == len(self.data_y):
            #         self.mcmc_snr = (np.sum(model_fit[left:right])-(right-left+1)*self.mcmc_y[0])/(np.sum(np.sqrt(self.err_y[left:right])))
            #         log.info(f"MCMC SNR model w/data error: {self.mcmc_snr}")
            #
            #     #
            #     # 3. sum over the data values (subtracting off the y-offset) and divide by the sum of the sqrt of the data errors
            #     #
            #         self.mcmc_snr = (np.sum(self.data_y[left:right])-(right-left+1)*self.mcmc_y[0])/(np.sum(np.sqrt(self.err_y[left:right])))
            #         log.info(f"MCMC SNR data w/data error: {self.mcmc_snr}")
            # except:
            #     log.warning("Exception calculating MCMC SNR: ", exc_info=True)
            #
            #
            # try:
            #     #
            #     # 4. sum of the model fit (subtractig the y-offset) divide by the sum of the sqrt of the propagated uncertainties in the fit
            #     #
            #
            #     unc_array = utilities.gaussian_uncertainty(None,self.data_x,
            #                                                self.mcmc_mu[0],0.5*(self.mcmc_mu[1]+self.mcmc_mu[2]),
            #                                                self.mcmc_sigma[0],0.5*(self.mcmc_sigma[1]+self.mcmc_sigma[2]),
            #                                                self.mcmc_A[0],0.5*(self.mcmc_A[1]+self.mcmc_A[2]),
            #                                                self.mcmc_y[0],0.5*(self.mcmc_y[1]+self.mcmc_y[2]))
            #
            #     self.mcmc_snr = (np.sum(model_fit[left:right]) - (right-left+1)*self.mcmc_y[0])/np.sum(np.sqrt(unc_array[left:right]))
            #     log.info(f"MCMC SNR model w/error prop: {self.mcmc_snr}")
            #
            # except:
            #     log.warning("Exception calculating MCMC SNR: ", exc_info=True)
            #
            # try:
            #     #
            #     # 5. Area under the model curve divided by the # pixels * the rmse
            #     #
            #
            #     #todo: need to adjust the AREA down by the fraction that is covered by the sigma_width set at the top
            #     self.mcmc_snr = self.mcmc_A[0]/((right-left+1)*rms_err)
            #     log.info(f"MCMC SNR Area model w/RMS: {self.mcmc_snr} RMSE={rms_err} Pix={right-left+1}")
            #
            # except:
            #     log.warning("Exception calculating MCMC SNR: ", exc_info=True)
            #
            # try:
            #     #
            #     # 6. Area under the model fit divided by the mean of the 68% wings from the MCMC samples
            #     #
            #
            #     self.mcmc_snr = self.mcmc_A[0]/(0.5*(self.mcmc_A[1]+self.mcmc_A[2]))
            #     log.info(f"MCMC SNR model Area with uncertainty: {self.mcmc_snr}")
            #
            #     #self.mcmc_snr = np.sum(model_fit)/(np.sqrt(len(self.data_x))*rms_err)
            #     #these are fluxes, so just sum over the model to get approximate total flux (aread under the curve) = signal
            #     #and divide by the sqrt of N (number of pixels) * the rmse as the error
            #
            # except:
            #     log.warning("Exception calculating MCMC SNR: ", exc_info=True)

            try:
                #
                # 7. Area under the model fit divided by the mean of the 68% wings from the MCMC samples
                # As of 2021-06-08 this is what Karl is using
                # but need to check the left and right and the err_y is what I expect (no 2AA correction)

                #self.mcmc_snr = self.mcmc_A[0]/ (np.sum(np.sqrt(self.err_y[left:right])))
                #note: sqrt(bin_width) is for the 2AA binning (the area does not know about binning)
                self.mcmc_snr = self.mcmc_A[0] / np.sqrt(np.sum(self.err_y[left:right]*self.err_y[left:right])) / np.sqrt(bin_width)
                self.mcmc_snr_err = 0.5*(self.mcmc_A[1]+self.mcmc_A[2])/self.mcmc_A[0] * self.mcmc_snr
                log.info(f"MCMC SNR model Area with data error: {self.mcmc_snr} +/- {self.mcmc_snr_err}")

                #self.mcmc_snr = np.sum(model_fit)/(np.sqrt(len(self.data_x))*rms_err)
                #these are fluxes, so just sum over the model to get approximate total flux (aread under the curve) = signal
                #and divide by the sqrt of N (number of pixels) * the rmse as the error

            except:
                log.warning("Exception calculating MCMC SNR: ", exc_info=True)

            if self.mcmc_snr is None:
                self.mcmc_snr = -1

            #note: these are logged as ["avg", +err, -err] so the last value becomes the negative
            log.info("MCMC mu: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_mu, self.mcmc_mu[0],self.mcmc_mu[1],self.mcmc_mu[2]))
            log.info("MCMC sigma: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_sigma, self.mcmc_sigma[0],self.mcmc_sigma[1],self.mcmc_sigma[2]))
            log.info("MCMC A: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g) *usually over 2AA bins" %
                     (self.initial_A, self.mcmc_A[0],self.mcmc_A[1],self.mcmc_A[2] ))
            log.info("MCMC y: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)"%
                     (self.initial_y, self.mcmc_y[0],self.mcmc_y[1],self.mcmc_y[2]))
            log.info("MCMC SNR: %0.5g" % self.mcmc_snr)
            log.info("MCMC f: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (0.0, mcmc_f[0], mcmc_f[1], mcmc_f[2]))
        except:
            log.error("Exception in mcmc_gauss::run_mcmc",exc_info=True)
            result = False

        return result



    def visualize(self,filename=None):
        try:
            if self.samples is not None:
                warnings.simplefilter(action='ignore', category=FutureWarning)

                fig = corner.corner(self.samples, labels=["$mu$", "$sigma$", "$A$", "$y$","f"],
                                    truths=[self.initial_mu, self.initial_sigma, self.initial_A, self.initial_y,None])
                #fifth = None is for the 'f' parameter ... there is no initial for it
                if filename is not None:
                    log.info('Writing: ' + filename)
                    fig.savefig(filename)
                else:
                    plt.show()

                buf = None
                try:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300)
                except:
                    log.warning("Exception in mcmc_gauss::visualize",exc_info=True)
                return buf
        except:
            log.warning("Exception in mcmc_gauss::visualize",exc_info=True)
            return None
