"""
cloned from mcmc_gauss
but uses a double Gaussian (trying to fit wide doublet lines ... not single lines with a broad + narrow component)

Mostly the same as mcmc_gauss, but the model is different
(could later just make these generic where you specifiy the number of Gaussians to use, but I do not anticipate
 using more than two)

 NOTE: these are not two fully independent Gaussians ... mu, sigma and amplitude can vary independently but I am
 keeping the same y (continuum) level

"""

from __future__ import print_function

try:
    from elixer import global_config as G
    from elixer import corner
    from elixer import emcee
except:
    import global_config as G
    import corner
    import emcee

import numpy as np
import io
import matplotlib.pyplot as plt

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

        return np.sqrt(((f - d) ** 2).mean())
    except:
        return -1 #non-sense value for snr


class MCMC_Double_Gauss:

    UncertaintyRange = [16,50,84]

    def __init__(self):
        #intial values are meant to be near the truth
        #and are expected to come from, say, some "best" fit
        self.initial_mu = None
        self.initial_sigma = None
        self.initial_A = None
        self.initial_y = None
        #self.initial_peak = None


        self.initial_mu_2 = None
        self.initial_sigma_2 = None
        self.initial_A_2 = None
        #NOTICE: we are keeping the y value the same
        #self.initial_peak_2 = None

        self.max_sigma = 30.0
        self.range_mu = 5.0
        self.max_A_mult = 2.0
        self.max_y_mult = 2.0
        #self.min_y = #-10.0 #-100.0 #should this be zero? or some above zero but low limit
        self.delta_y = 0#for the double Gaussian, we assume the y-value is pretty good, so don't let it vary much

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
        self.mcmc_snr = None #snr as flux_area/1-sigma uncertainty

        self.mcmc_mu_2 = None  #3-tuples [0] = fit, [1] = fit +16%,  [2] = fit - 16%
        self.mcmc_sigma_2 = None
        self.mcmc_A_2 = None

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


    def compute_model(self,x,mu,sigma,A,y,mu2,sigma2,A2):
        try:
            return A * (np.exp(-np.power((x - mu) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + \
                A2 * (np.exp(-np.power((x - mu2) / sigma2, 2.) / 2.) / np.sqrt(2 * np.pi * sigma2 ** 2)) + y
        except:
            return np.nan

    def model(self,x,theta):
        """

        :param x:
        :param theta:
        :return:
        """
        mu, sigma, A, y, mu2, sigma2, A2, ln_f = theta #note: not using ln_f here
        if (x is None) or (mu is None) or (sigma is None) or (mu2 is None) or (sigma2 is None):
            return None
        try:
            value = self.compute_model(x,mu,sigma,A,y,mu2,sigma2,A2)
            # note: noise is separate and included in the lnlike() function
            # value = A * (np.exp(-np.power((x - mu) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + \
            #         A2 * (np.exp(-np.power((x - mu2) / sigma2, 2.) / 2.) / np.sqrt(2 * np.pi * sigma2 ** 2)) + y
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

        #this is wrong ????
        return -0.5 * (np.sum((diff ** 2) / sigma2 + np.log(sigma2)))

    # if any are zero, the whole prior is zero
    # all priors here are uniformitive ... i.e they are all flat ... either zero or one
    def lnprior(self, theta):  # theta is a n-tuple (_,_,_ ... )
        mu, sigma, A, y, mu2, sigma2, A2, ln_f = theta
        # note: could take some other dynamic maximum for y (like compute the peak ... y can't be greater than that
        #let  A_2 be able to be zero so this becomes a single gaussian
        if ( abs(mu - self.initial_mu) < self.range_mu) and \
                (0.1 < sigma < self.max_sigma) and \
                (0.0 < A < self.max_A_mult * self.initial_A) and \
            (abs(mu2 - self.initial_mu_2) < self.range_mu) and \
                (0.1 <= sigma2 < self.max_sigma) and \
                (0.0 <= A2 < self.max_A_mult * self.initial_A_2) and \
            ((y-self.delta_y) < y < (y+self.delta_y)):
            return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)




        if self.initial_A < 0 : #same as emission, but "A" is negative (flip sign) and y is between a max and zero

            if ( abs(mu - self.initial_mu) < self.range_mu) and \
                    (0.1 < sigma < self.max_sigma) and \
                    (self.max_A_mult * self.initial_A <= A < 0.0) and \
                    (abs(mu2 - self.initial_mu_2) < self.range_mu) and \
                    (0.1 <= sigma2 < self.max_sigma) and \
                    (self.max_A_mult * self.initial_A_2 <= A2 < 0.0) and \
                    ((y-self.delta_y) < y < (y+self.delta_y)):
                return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)
        else:
            if ( abs(mu - self.initial_mu) < self.range_mu) and \
                    (0.1 < sigma < self.max_sigma) and \
                    (0.0 < A < self.max_A_mult * self.initial_A) and \
                    (abs(mu2 - self.initial_mu_2) < self.range_mu) and \
                    (0.1 <= sigma2 < self.max_sigma) and \
                    (0.0 <= A2 < self.max_A_mult * self.initial_A_2) and \
                    ((y-self.delta_y) < y < (y+self.delta_y)):
                return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)
        return -np.inf  # -999999999 #-np.inf #roughly ln(0) == -inf




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
                if (self.err_x is not None) and len(self.data_x) != len(self.err_x):
                    return False

                if self.initial_sigma < 0.0: #self.initial_A < 0.0  ... actually, leave A alone .. might allow absorportion later
                    return False

                # if self.initial_y > self.initial_peak:
                #     return False
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

        #set the allowed y-variation
        self.delta_y = abs(self.initial_y)*0.2#,max(self.initial_peak,self.initial_peak_2)*0.1)

        self.max_sigma = self.initial_sigma + self.initial_sigma_2
        self.range_mu = max(5.0,abs(self.initial_mu - self.initial_mu_2))  #should be allowed to line up (which would be right at /2.0
        #so using 1.5 for some slop

        #here for initial positions of the walkers, sample from narrow gaussian (hence the randn or randNormal)
        #centered on each of the maximum likelihood selected parameter values
        #mu, sigma, A, y, ln_f = theta #note f or ln(f) is another uncertainty ...an underestimation of the variance
        #                               by some factor (f) .... e.g. variance = variance + f * model
        initial_pos = [self.initial_mu, self.initial_sigma,  self.initial_A,self.initial_y,
                       self.initial_mu_2,self.initial_sigma_2,self.initial_A_2,0.0]

        #mostly for the A (area)
        if self.initial_A < 0: #absorber
            max_pos = [np.inf, np.inf,     0.0, max(self.data_y),  np.inf, np.inf,     0.0,   np.inf]
            min_pos = [   0.0,   0.01, -np.inf,          -np.inf,     0.0,   0.01, -np.inf,  -np.inf]
        else:
            #here, because of the max check, none mu, sigma, or A will be negative
            max_pos = [np.inf, np.inf,np.inf,max(self.data_y), np.inf, np.inf, np.inf,   np.inf] #must be less than this
            min_pos = [   0.0,  0.01,   0.01,         -np.inf,    0.0,   0.01,   0.01,  -np.inf] #must be greater than this


        #reminder, there is no y2 so that position is absent from the second line
        ndim = len(initial_pos)
        scale = np.array([1.,1.,1.,0.2,   1.,1.,1.,  -100.5]) #don't nudge ln_f ...note ln_f = -4.5 --> f ~ 0.01
        #nudge the initial positions around a bit

        try:
#            pos = [initial_pos +  scale * np.random.randn(ndim) for i in range(self.walkers)]
            pos = [np.minimum(np.maximum(initial_pos + scale * np.random.randn(ndim),min_pos),max_pos) for i in range(self.walkers)]


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

            ##HERE
            self.mcmc_mu, self.mcmc_sigma, self.mcmc_A, self.mcmc_y, self.mcmc_mu_2, self.mcmc_sigma_2, self.mcmc_A_2, mcmc_f = \
                map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(self.samples, self.UncertaintyRange,axis=0)))

            try:
                #self.mcmc_snr = self.mcmc_A[0] / (0.5 * (abs(self.mcmc_A[1]) + abs(self.mcmc_A[2])))
                rms_model = self.compute_model(self.data_x,self.mcmc_mu[0],self.mcmc_sigma[0],self.mcmc_A[0],self.mcmc_y[0],
                                               self.mcmc_mu_2[0],self.mcmc_sigma_2[0],self.mcmc_A_2[0])
                err = rms(self.data_y,rms_model,None,None,False)
                self.mcmc_snr = (np.sum(rms_model)-len(rms_model)*self.mcmc_y[0])/(np.sqrt(len(self.data_x))*err)
                #these are fluxes, so just sum over the model to get approximate total flux (aread under the curve) = signal
                #and divide by the sqrt of N (number of pixels) * the rmse as the error
                #self.mcmc_snr = rms(self.data_y,rms_model,None,None,False)

            except:
                self.mcmc_snr = -1
                log.warning("Exception calculating MCMC SNR: ", exc_info=True)



            log.info("MCMC mu: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_mu, self.mcmc_mu[0],self.mcmc_mu[1],self.mcmc_mu[2]))
            log.info("MCMC mu2: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_mu_2, self.mcmc_mu_2[0],self.mcmc_mu_2[1],self.mcmc_mu_2[2]))

            log.info("MCMC sigma: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_sigma, self.mcmc_sigma[0],self.mcmc_sigma[1],self.mcmc_sigma[2]))

            log.info("MCMC sigma2: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_sigma_2, self.mcmc_sigma_2[0],self.mcmc_sigma_2[1],self.mcmc_sigma_2[2]))

            log.info("MCMC A: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g) *usually over 2AA bins" %
                     (self.initial_A, self.mcmc_A[0],self.mcmc_A[1],self.mcmc_A[2] ))

            log.info("MCMC A2: initial[%0.5g] mcmc(%0.5g, +%0.5g, -%0.5g) *usually over 2AA bins" %
                     (self.initial_A_2, self.mcmc_A_2[0],self.mcmc_A_2[1],self.mcmc_A_2[2] ))

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

                fig = corner.corner(self.samples, labels=["$mu$", "$sigma$", "$A$", "$y$","$mu2$","$sigma2$","$A2$","f"],
                                    truths=[self.initial_mu, self.initial_sigma, self.initial_A, self.initial_y,
                                            self.initial_mu_2,self.initial_sigma_2,self.initial_A_2,None])
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

    def show_fit(self,filename=None):
        def dbl_gaussian(x,u1,s1,A1,u2,s2,A2,y):
            return A1 * (np.exp(-np.power((x - u1) / s1, 2.) / 2.) / np.sqrt(2 * np.pi * s1 ** 2)) + \
                   A2 * (np.exp(-np.power((x - u2) / s2, 2.) / 2.) / np.sqrt(2 * np.pi * s2 ** 2)) + y

        def gaussian(x,u1,s1,A1=1.0,y=0.0):
            return A1 * (np.exp(-np.power((x - u1) / s1, 2.) / 2.) / np.sqrt(2 * np.pi * s1 ** 2)) + y

        try:
           plt.figure()

           plt.errorbar(self.data_x,self.data_y,yerr=self.err_y,xerr=self.err_x,fmt=".")
           x = np.linspace(self.data_x[0],self.data_x[-1],100)
           y = dbl_gaussian(x,self.mcmc_mu[0],self.mcmc_sigma[0],self.mcmc_A[0],
                            self.mcmc_mu_2[0],self.mcmc_sigma_2[0],self.mcmc_A_2[0],self.mcmc_y[0])
           plt.plot(x,y,color='g')

           y = gaussian(x,self.mcmc_mu[0],self.mcmc_sigma[0],self.mcmc_A[0],self.mcmc_y[0])
           plt.plot(x,y,color='b')

           y = gaussian(x,self.mcmc_mu_2[0],self.mcmc_sigma_2[0],self.mcmc_A_2[0],self.mcmc_y[0])
           plt.plot(x,y,color='r')

           if filename:
               plt.savefig(filename)

        except:
            log.warning("Exception in mcmc_gauss::show_fit",exc_info=True)
            return None
