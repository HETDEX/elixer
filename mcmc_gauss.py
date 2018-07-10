from __future__ import print_function

import global_config as G
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import warnings

log = G.Global_Logger('mcmc_logger')
log.setlevel(G.logging.DEBUG)

#todo: incorporate error in x coord

class MCMC_Gauss:

    def __init__(self):
        #intial values are meant to be near the truth
        #and are expected to come from, say, some "best" fit
        self.initial_mu = None
        self.initial_sigma = None
        self.initial_A = None
        self.initial_y = None
        self.initial_peak = None

        self.max_sigma = 20.0
        self.range_mu = 5.0
        self.max_A_mult = 2.0
        self.max_y_mult = 2.0
        self.min_y = -100.0

        self.data_x = None
        self.data_y = None
        self.err_x = None
        self.err_y = None

        #this is mostly a guess ... no great way to automate, but this is pretty quick
        #and since the initials are from a scipy curve fit, we stabablize pretty fast
        self.burn_in = 100
        self.main_run = 1000
        self.walkers = 100

        self.sampler = None #mcmc sampler
        self.samples = None #resulting samples

        self.mcmc_mu = None
        self.mcmc_sigma = None
        self.mcmc_A = None
        self.mcmc_y = None


    def model(self,x,theta):
        mu, sigma, A, y = theta
        if (x is None) or (mu is None) or (sigma is None):
            return None
        try:
            value = A * (np.exp(-np.power((x - mu) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y
        except:
            value = np.nan
        return value

    def lnlike(self, theta, x, y, yerr):
        diff = y - self.model(x, theta)
        inv_sigma2 = 1.0 / (self.err_y ** 2)  # + model**2)
        return -0.5 * (np.sum(diff ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    # if any are zero, the whole prior is zero
    # all priors here are uniformitive ... i.e they are all flat ... either zero or one
    def lnprior(self, theta):  # theta is a n-tuple (_,_,_ ... )
        mu, sigma, A, y = theta
        # note: could take some other dynamic maximum for y (like compute the peak ... y can't be greater than that
        if (-self.range_mu < mu - self.initial_mu < self.range_mu) and \
                (0.0 < sigma < self.max_sigma) and \
                (0.0 < A < self.max_A_mult * self.initial_A) and \
                (self.min_y < y < self.max_y_mult * self.initial_peak):
            return 0.0  # remember this is ln(prior) so a return of 0.0 == 1  (since ln(1) == 0.0)
        return -np.inf  # -999999999 #-np.inf #roughly ln(0) == -inf

    def lnprob(self, theta, x, y, yerr):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, x, y, yerr)  # again, since logs, this is a sum ln(likelihood) + ln(prior)


    def sanity_check_init(self):
        try:
            if self.err_y is None:
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
        ndim = 4  # 4 dims since 4 parms mu, sigma, A, y = theta
        #here for initial positions of the walkers, sample from narrow gaussian (hence the randn or randNormal)
        #centered on each of the maximum likelihood selected parameter values
        #mu, sigma, A, y, lnf = theta
        initial_pos = [self.initial_mu,self.initial_sigma,self.initial_A,self.initial_y]
        scale = np.array([1.,1.,1.,1.])
        #nudge the initial positions around a bit

        try:
            pos = [initial_pos +  scale * np.random.randn(ndim) for i in range(self.walkers)]

            #build the sampler
            #todo: incorporate self.err_x
            self.sampler = emcee.EnsembleSampler(self.walkers, ndim, self.lnprob,
                                            args=(self.data_x,self.data_y, self.err_y))

            with warnings.catch_warnings(): #ignore the occassional warnings from the walkers (NaNs, etc that reject step)
                warnings.simplefilter("ignore")
                log.debug("Burn in ....")
                pos, prob, state = self.sampler.run_mcmc(pos, self.burn_in)  # burn in
                log.debug("Main run ...")
                pos, prob, state = self.sampler.run_mcmc(pos, self.main_run, rstate0=state)  # start from end position of burn-in

            self.samples = self.sampler.flatchain  # collapse the walkers and interations (aka steps or epochs)

            log.info("MCMC mean acceptance fraction: %0.3f" %(np.mean(self.sampler.acceptance_fraction)))

            self.mcmc_mu, self.mcmc_sigma, self.mcmc_A, self.mcmc_y = \
                map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),zip(*np.percentile(self.samples, [16, 50, 84],axis=0)))

            log.info("MCMC mu: [%0.5g] (%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_mu, self.mcmc_mu[0],self.mcmc_mu[1],self.mcmc_mu[2]))
            log.info("MCMC sigma: [%0.5g] (%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_sigma, self.mcmc_sigma[0],self.mcmc_sigma[1],self.mcmc_sigma[2]))
            log.info("MCMC A: [%0.5g] (%0.5g, +%0.5g, -%0.5g)" %
                     (self.initial_A, self.mcmc_A[0],self.mcmc_A[1],self.mcmc_A[2] ))
            log.info("MCMC y: [%0.5g] (%0.5g, +%0.5g, -%0.5g)"%
                     (self.initial_y, self.mcmc_y[0],self.mcmc_y[1],self.mcmc_y[2]))
        except:
            log.error("Exception in mcmc_gauss::run_mcmc",exc_info=True)
            result = False

        return result



    def visualize(self,filename=None):
        try:
            if self.samples is not None:
                fig = corner.corner(self.samples, labels=["$mu$", "$sigma$", "$A$", "$y$"],
                                    truths=[self.initial_mu, self.initial_sigma, self.initial_A, self.initial_y])
                if filename is not None:
                    fig.savefig(filename)
                else:
                    plt.show()
        except:
            log.warning("Exception in mcmc_gauss::visualize",exc_info=True)
