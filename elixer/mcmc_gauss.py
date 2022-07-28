from __future__ import print_function

try:
    from elixer import global_config as G
    from elixer import corner
    from elixer import utilities
    from elixer import emcee
    from elixer import spectrum_utilities as SU
except:
    import global_config as G
    import corner
    import utilities
    import emcee
    import spectrum_utilities as SU

import numpy as np
import io
import matplotlib.pyplot as plt
import copy
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
        self.mcmc_h = None #maximum height (or minium for absorber)

        #not tuple, just single floats
        self.mcmc_snr = None
        self.mcmc_snr_err = 0
        self.mcmc_snr_pix = 0
        self.mcmc_chi2 = 0
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
            if (-self.range_mu < (mu - self.initial_mu) < self.range_mu) and \
                    (0.0 < sigma < self.max_sigma) and \
                    (self.max_A_mult * self.initial_A < A < 0.0) and \
                    (self.min_y < y < self.max_y_mult * self.initial_peak):
                return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)
        else:
            if (-self.range_mu < (mu - self.initial_mu) < self.range_mu) and \
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


                if (self.initial_sigma is None) or (self.initial_mu is None) or (self.initial_A is None) or (self.initial_y is None):
                    return False

                if self.initial_sigma < 0.0: #self.initial_A < 0.0  ... actually, leave A alone .. might allow absorportion later
                    return False


                if self.initial_peak is None:
                    left,*_ = utilities.getnearpos(self.data_x,self.initial_mu-self.initial_sigma*4)
                    right,*_ = utilities.getnearpos(self.data_x,self.initial_mu+self.initial_sigma*4)

                    self.initial_peak = max(self.data_y[left:right])

                if ((self.initial_A > 0) and (self.initial_y > self.initial_peak) or \
                    (self.initial_A < 0) and (self.initial_y < self.initial_peak) ):
                    #i.e. if an emission (A > 0) then y must be less than the peak
                    # else if an absorption line (A < 0) then y must be greater than the peak
                    return False
            else:
                return False
            return True
        except:
            log.warning("Exception in mcmc_gauss::sanity_check",exc_info=True)
            return False

    def run_mcmc(self):


        #cannot have nans
        #note: assumes data_x (the spectral axis) and err_x have none since they are on a known grid
        data_nans = np.isnan(self.data_y)
        err_nans = np.isnan(self.err_y)

        if (np.sum(data_nans) > 0) or (np.sum(err_nans) > 0):
            self.data_y = copy.copy(self.data_y)[~data_nans]
            self.err_y = copy.copy(self.err_y)[~data_nans]
            self.data_x = copy.copy(self.data_x)[~data_nans]
            self.err_x = copy.copy(self.err_x)[~data_nans]
            #and clean up any other nan's in the error array for y
            err_nans = np.isnan(self.err_y)
            self.err_y[err_nans] = np.nanmax(self.err_y*10)

        if not self.sanity_check_init():
            log.info("Sanity check failed. Cannot conduct MCMC.")
            return False

        result = True

        #here for initial positions of the walkers, sample from narrow gaussian (hence the randn or randNormal)
        #centered on each of the maximum likelihood selected parameter values
        #mu, sigma, A, y, ln_f = theta #note f or ln(f) is another uncertainty ...an underestimation of the variance
        #                               by some factor (f) .... e.g. variance = variance + f * model
        initial_pos = [self.initial_mu,self.initial_sigma,self.initial_A,self.initial_y,0.0]
        ndim = len(initial_pos)
        #even with the random nudging the pos values must be greater than (or less than for absorption) these values

        try:
            ####################################################################################################
            # This is an alternate way to control the jitter in the initial positions,
            # Set the boolean below to True to use this vs. the original method (smaller, normal distro jitter)
            ####################################################################################################
            if True:
                ip_mu = initial_pos[0] + np.random.uniform(-1.0*(self.data_x[1]-self.data_x[0]),1.0*(self.data_x[1]-self.data_x[0]),self.walkers)
                #sigma cannot go zero or below
                ip_sigma = initial_pos[1] + np.random.uniform(-0.5*self.initial_sigma,0.5*self.initial_sigma,self.walkers)
                #area cannot flip signs
                ip_A = initial_pos[2] +  np.random.uniform(0,1.0*self.initial_A,self.walkers)
                #y should not exceed min/max data value, but won't cause and error if it does
                # ... should technically never be negative regardless of absorption or emission
                ip_y = np.random.uniform(0,max(self.data_y),self.walkers)
                ip_lnf = np.random.uniform(0.005,0.015,self.walkers) #np.zeros(self.walkers) #np.random.uniform(0.005,0.015,self.walkers) #np.zeros(self.walkers)

                pos = np.column_stack((ip_mu,ip_sigma,ip_A,ip_y,ip_lnf))

                # #for p in pos: #just a debug check
                # #  print(f"{p[0]:0.4g},{p[1]:0.4g},{p[2]:0.4g},{p[3]:0.4g},{p[4]:0.4g}")
            else:
                ##############################################################################################
                # OTHERWISE, keep the block below
                #############################################################################################

                #mostly for the A (area)
                if self.initial_A < 0: #absorber
                    max_pos = [np.inf, np.inf,     0.0, max(self.data_y),  np.inf]
                    min_pos = [   0.0,   0.01, -np.inf,          -np.inf, -np.inf]
                else:
                    #here, because of the max check, none mu, sigma, or A will be negative
                    max_pos = [np.inf, np.inf,np.inf,max(self.data_y), np.inf] #must be less than this
                    min_pos = [   0.0,  0.01,   0.01,         -np.inf,-np.inf] #must be greater than this

                scale = np.array([10.,5.,2.0*self.initial_A,5.0*self.initial_y,0.0]) #don't nudge ln_f ...note ln_f = -4.5 --> f ~ 0.01
                pos = [np.minimum(np.maximum(initial_pos + scale * np.random.uniform(-1,1,ndim),min_pos),max_pos) for i in range(self.walkers)]

            #build the sampler
            self.sampler = emcee.EnsembleSampler(self.walkers, ndim, self.lnprob,
                                                 args=(self.data_x,self.data_y, self.err_y))

            #args are the positional args AFTER theta for self.lnprob function

            with warnings.catch_warnings(): #ignore the occassional warnings from the walkers (NaNs, etc that reject step)
                warnings.simplefilter("ignore")
                log.debug("MCMC burn in (%d) ...." %self.burn_in)
                pos, prob, state = self.sampler.run_mcmc(pos, self.burn_in,skip_initial_state_check=True)  # burn in
                log.debug("MCMC main run (%d) ..." %self.main_run)
                log.debug("MCMC (w:%0.2f) main run (%d) ..." %(self.initial_mu,self.main_run))
                pos, prob, state = self.sampler.run_mcmc(pos, self.main_run, rstate0=state,skip_initial_state_check=True)  # start from end position of burn-in

            self.samples = self.sampler.flatchain  # collapse the walkers and interations (aka steps or epochs)

            log.debug("MCMC mean acceptance fraction: %0.3f" %(np.mean(self.sampler.acceptance_fraction)))

            #for each, in order
            #v[0] is the 16 percentile (~ - 1sigma)
            #v[1] is the 50 percentile (so the "average")
            #v[2] is the 84 percentile (~ +1sigma)
            #the tuple reports then as ["average", "84th - average", "average - 16th"]
            #should always be positive (assuming a positive value) BUT when printed to the log, the 3rd value is made
            #to be negative showing that you would subtract it from the average to get to the 16th percentile

            sigma_width = 2
            flux_frac = 0.955 # 0.955 # i.e. depends on sigma: 1 = 0.682, 2 = 0.955,  3 = 0.996, 4+ just use 1.0

            #using 68% interval
            self.mcmc_mu, self.mcmc_sigma, self.mcmc_A, self.mcmc_y, mcmc_f = \
                map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(self.samples, self.UncertaintyRange,axis=0)))

            # mcmc_mu_95, mcmc_sigma_95, mcmc_A_95, mcmc_y_95, mcmc_f = \
            #     map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(self.samples, [5,50,95],axis=0)))


            try: #basic info used by multiple SNR calculations

                #2021-07-06
                #updated rules; to match with Karl only want to include the wavelength bin if 50+% of the bin
                #is within the limit (e.g. if the mu-2*sigma is 3999.001, then the 3998 bin IS NOT included, but the
                # 4000AA bin IS includede. .... if the mu-2*sigma is 3998.999 then the 3998 bin IS included
                #ALSO, sum over the same model wavelength bins rather than use the integrated flux * gaussian adjustment
                #ALSO, require a minium of 3 bins (with an exception if 1 of the

                #bin_width = self.data_x[1] - self.data_x[0]
                #2.1195 = 1.8AA * 2.355 / 2.0  ... the instrumental DEX (1.8AA)
                delta_wave = max(self.mcmc_sigma[0]*sigma_width,2.1195)  #must be at least +/- 2.1195AA
                #!!! Notice: if we ever change this to +/- 1sigma, need to switch out to sum over the data
                #instead of the model !!!
                left,*_ = utilities.getnearpos(self.data_x,self.mcmc_mu[0]-delta_wave)
                right,*_ = utilities.getnearpos(self.data_x,self.mcmc_mu[0]+delta_wave)

                if self.data_x[left] - (self.mcmc_mu[0]-delta_wave) < 0:
                    left += 1 #less than 50% overlap in the left bin, so move one bin to the red
                if self.data_x[right] - (self.mcmc_mu[0]+delta_wave) > 0:
                    right -=1 #less than 50% overlap in the right bin, so move one bin to the blue

                #lastly ... could combine, but this is easier to read
                right += 1 #since the right index is not included in slice

                #we want the next wavebin to either side
                # left = max(0,left-1)
                #right = min(right+1,len(self.data_x)) #+2 insted of +1 since the slice does not include the end
                #at 4 sigma the mcmc_A[0] is almost identical to the model_fit (as you would expect)
                #note: if choose to sum over model fit, remember that this is usually over 2AA wide bins, so to
                #compare to the error data, need to multiply the model_sum by the bin width (2AA)
                #(or noting that the Area == integrated flux x binwidth)
                model_fit = self.compute_model(self.data_x[left:right],self.mcmc_mu[0],self.mcmc_sigma[0],self.mcmc_A[0],self.mcmc_y[0])
                #apcor = np.ones(len(model_fit))
                #subtract off the y continuum since we want flux in the model
                if self.initial_A < 0:
                    self.mcmc_h = np.nanmin(model_fit)
                else:
                    self.mcmc_h = np.nanmax(model_fit)
                data_err = copy.copy(self.err_y[left:right])
                data_err[data_err<=0] = np.nan #Karl has 0 value meaning it is flagged and should be skipped

                data_flux = self.data_y[left:right]
                #rms_err = rms(self.data_y[left:right],model_fit[left:right],None,None,False)

                #signal is the sum of the data between +/- 2.5 sigma (minus the y-offset)
                # noise is the sqrt of the nummber of pixels (wavebins)*the RMSerror)
            except:
                log.warning("Exception calculating MCMC SNR: ", exc_info=True)


            try:
                #
                # Area under the model fit divided by the mean of the 68% wings from the MCMC samples
                # As of 2021-06-08 this is what Karl is using
                # but need to check the left and right and the err_y is what I expect (no 2AA correction)

                #signal = area under the curve or sum(model_fit)*bin_width .... line_flux = area / bin_width * units * scale
                #noise = error on data (i.e. err_y) ... summed in quadrature over center +/- 4 sigma

                # self.mcmc_snr = abs(self.mcmc_A[0]) / np.sqrt(np.sum(self.err_y[left:right]*self.err_y[left:right])) #/ np.sqrt(bin_width)
                # self.mcmc_snr_err = abs(0.5*(self.mcmc_A[1]+self.mcmc_A[2])/self.mcmc_A[0] * self.mcmc_snr)
                # log.info(f"MCMC SNR model Area with data error: {self.mcmc_snr} +/- {self.mcmc_snr_err}")

                #self.mcmc_snr = flux_frac*abs(self.mcmc_A[0]/2.0) / np.sqrt(np.nansum(data_err**2))
                self.mcmc_snr = abs(np.sum(model_fit-self.mcmc_y[0])) / np.sqrt(np.nansum(data_err**2))
                self.mcmc_snr_err = abs(0.5*(self.mcmc_A[1]+self.mcmc_A[2])/self.mcmc_A[0] * self.mcmc_snr)
                self.mcmc_snr_pix = len(model_fit)
                log.info(f"MCMC SNR model Area with data error: {self.mcmc_snr} +/- {self.mcmc_snr_err}")


                #the chi2 should be over the full fit width , not +/- 2 sigma used for the signal
                chi2_model_fit = self.compute_model(self.data_x, self.mcmc_mu[0], self.mcmc_sigma[0],
                                                self.mcmc_A[0], self.mcmc_y[0])

                self.mcmc_chi2, _ = SU.chi_sqr(self.data_y,chi2_model_fit,error=self.err_y,c=1.0)#,dof=3)
                log.info(f"MCMC chi2: {self.mcmc_chi2}")

               # print(f"***** TEST MCMC SNR: {self.mcmc_snr}")
                #self.mcmc_snr = abs(np.sum(model_fit)) / np.sqrt(np.sum(self.err_y[left:right]*self.err_y[left:right])) #/ np

                #log.info(f"*** { abs(self.mcmc_A[0]/1.0) / np.sqrt(np.sum(self.err_y[left:right]/1.0))}")

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
