#import Bayes.bayesian
#import Bayes.nb
import math
try:
    from ConfigParser import RawConfigParser
except:
    from configparser import RawConfigParser
import os
import sys

import astropy.stats.biweight as biweight
import matplotlib.pyplot as plt
from scipy.stats import bayes_mvs


try:
    from elixer import global_config as G
    from elixer import weighted_biweight as elixer_biweight
    import elixer.line_classifier.probs.classification_prob as LineClassifierPro
#    import elixer.line_classifier.probs.classification_prob_leung as LineClassifierPro_Leung
except:
    import global_config as G
    import line_classifier.probs.classification_prob as LineClassifierPro
    import weighted_biweight as elixer_biweight
#    import line_classifier.probs.classification_prob_leung as LineClassifierPro_Leung

import numpy as np
#numpy import array
#don't need to do this ... is performed in source_prob call
#from line_classifier.misc.tools import generate_cosmology_from_config, read_flim_file


MAX_PLAE_POII = 1000.
MIN_PLAE_POII = 0.001
UNIVERSE_CONFIG = None
FLUX_LIMIT_FN = None
COSMOLOGY = None

#log = G.logging.getLogger('line_prob_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('line_prob_logger')
log.setlevel(G.LOG_LEVEL)

def conf_interval_asym(data,avg,conf=0.68):
    """

    :param data:
    :param avg:
    :param conf:
    :return:
    """

    high, low = None, None
    try:
        size = len(data)
        if size < 10:
            log.info(f"conf_interval_asym, data size too small {size}")
            return None, None

        step = int(round(conf/2. * size))
        s = np.array(sorted(data))

        idx = (np.abs(s - avg)).argmin()
        if (idx == 0) or (idx == size-1):
            #there is a problem or the list is all essentially identical
            if np.std(s) < 1e-5: #effectively zero, the average has no error and the ci is avg +/- 0
                return s[idx], s[idx]

        #what if many of the same value, want to put our position in the center of that run
        same_idx = np.where(s==s[idx])[0]
        if len(same_idx) > 1:
            log.debug(f"conf_interval_asym, multiple matches ({len(same_idx)}) to avg {avg}")
            idx = int(np.nanmedian(same_idx))

        low = s[max(0,idx-step)]
        high = s[min(size-1,idx+step)]

        log.debug(f"Asym Confidence interval: high: {high}, low: {low}, ci: {conf}, len: {size}")

    except:
        log.debug("Exception in conf_interval_asym",exc_info=True)

    return high, low


def conf_interval(num_samples,sd,conf=0.95):
    """
    mean +/- error  ... this is the +/- error part as 95% (or other) confidence interval (assuming normal distro)

    :param num_samples:
    :param sd: standard deviation
    :param conf:
    :return:
    """

    if num_samples < 30:
        return None

    #todo: put in other values
    if conf == 0.68:
        t = 1.0
    elif conf == 0.95:
        t = 1.96
    elif conf == 0.99:
        t = 2.576
    else:
        log.debug("todo: need to handle other confidence intervals: ", conf)
        return None

    return t * sd / np.sqrt(num_samples)



# def example_prob_LAE():
#     Cosmology = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 'h': 0.70, 'omega_k_0': 0}
#     LAE_Priors = 1  # case 1
#     alpha_LAE = -1.65  # but can have other values based on LAE_Priors case adopted (-1.36, -1.3, etc?)
#
#     # def main():
#
#     # initialization taken directly from Andrew's code, but I don't know what they mean
#     Bayes.nb.init(_alpha_LAE=-1.65, _mult_LStar_LAE=1, _mult_PhiStar_LAE=1, _mult_w0_LAE=1,
#                   _alpha_OII=-1.2, _mult_LStar_OII=1, _mult_PhiStar_OII=1, _mult_w0_OII=1)
#
#     ratio_LAE, plgd, pogd = Bayes.bayesian.prob_ratio(wl_obs=3900.0, lineFlux=1.6e-17, ew_obs=-20.0,
#                                                       c_obs=None, which_color=None, addl_fluxes=None,
#                                                       sky_area=5.5e-07, cosmo=Cosmology, LAE_priors=LAE_Priors,
#                                                       EW_case=1, W_0=None, z_OII=None, sigma=None)
#
#     print(ratio_LAE, plgd, pogd)


def fiber_area_in_sqdeg(num_fibers=1):
    #assumes no overlap
    return num_fibers*(math.pi*G.Fiber_Radius**2)/(3600.)**2
#
#
# #wrapper for Andrew Leung's base code
# def prob_LAE_old(wl_obs,lineFlux,ew_obs,c_obs, which_color=None, addl_wavelengths=None, addl_fluxes=None,addl_errors=None,
#             sky_area=None, cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None, sigma=None):
#     '''
#
#     :param wl_obs:
#     :param lineFlux:
#     :param ew_obs:
#     :param c_obs:
#     :param which_color:
#     :param addl_fluxes: cgs flux
#     :param addl_wavelengths: wavelength(observed) for each addl_flux at same index
#     :param sky_area:
#     :param cosmo:
#     :param lae_priors:
#     :param ew_case:
#     :param W_0:
#     :param z_OII:
#     :param sigma:
#     :return:
#     '''
#
#     #sanity check
#     if (ew_obs is None) or (ew_obs == -300) or (ew_obs == 666) or (lineFlux < 0):
#         #bsically, sentinel (null) values from Karl's input file
#         return 0.0,0.0,0.0
#
#     #what about different equivalent width calculations for LAE vs OII (different rest wavelength ... or is the ew_obs
#     #not rest_frame --- the more likely ... in which case, don't divide by the ratio of observed wavelength/rest)?
#
#     #addl_fluxes should be a dictionary: {wavlength,flux} (need to update downstream code though)
#         #to be consistent with Andrew's code, made addl_fluxex and addl_wavelengths, index matched arrays
#
#     #sky_area in square degrees (should this be the overlapped sum of all fibers? each at 0.75" radius?
#                 #makes very little difference in the output
#
#     #LAE call needs LAE_Priors, but OII does not
#     #OII calls need the last four  parameters (EW_case, z_OII, W_0, sigma) but the LAE calls do not
#
#     Cosmology_default = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 'h': 0.70, 'omega_k_0': 0}
#     LAE_Priors_default = 1  # case 1
#     EW_case_default = 1
#     alpha_LAE_default = -1.65  # but can have other values based on LAE_Priors case adopted (-1.36, -1.3, etc?)
#     sky_area_default = 1.36e-7 # one fiber in square degrees (has very little effect on probabilities anyway and
#                                # really is only used in the simulation
#     #looks like z_OII is meant to be a list that is used by the simulation, so not needed here
#     #the actual z_OII that is used is calculated as you would expect from the wavelength, assuming it is OII
#
#     # def main():
#     if cosmo is None:
#         cosmo = Cosmology_default
#
#     if lae_priors is None:
#         lae_priors = LAE_Priors_default
#
#     if ew_case is None:
#         ew_case = EW_case_default
#
#     if sky_area is None:
#         sky_area = sky_area_default
#
#
#     #looks like lineFlux should be in cgs units (despite the paper showing luminosity func in Jy)
#     #convert lineFlux from cgs to Jansky
#     #lineFlux = lineFlux / (1e-23) * (wl_obs**2)/(3e18)
#
#     #suppress sign of EW (always wants positive)
#     ew_obs = abs(ew_obs)
#
#     #todo: addl_errors=None
#     try:
#     # initialization taken directly from Andrew's code, but I don't know what they mean
#         Bayes.nb.init(_alpha_LAE=alpha_LAE_default, _mult_LStar_LAE=1, _mult_PhiStar_LAE=1, _mult_w0_LAE=1,
#                       _alpha_OII=-1.2, _mult_LStar_OII=1, _mult_PhiStar_OII=1, _mult_w0_OII=1)
#
#         #plgd == Probability of lae given the data?
#         #pogd = Probability of OII given the data ?
#         ratio_LAE, plgd, pogd = Bayes.bayesian.prob_ratio(wl_obs=wl_obs, lineFlux=lineFlux, ew_obs=ew_obs,
#                                                           c_obs=c_obs, which_color=which_color,
#                                                           addl_wavelengths=addl_wavelengths,
#                                                           addl_fluxes=addl_fluxes,
#                                                           sky_area=sky_area, cosmo=cosmo, LAE_priors=lae_priors,
#                                                           EW_case=ew_case, W_0=W_0, z_OII=z_OII,  sigma=sigma)
#
#
#         #ratio_LAE is plgd/pogd
#         #slightly different representation of ratio_LAE (so recomputed for elixer use)
#         if (plgd is not None) and (plgd > 0.0):
#             if (pogd is not None) and (pogd > 0.0):
#                 ratio_LAE = float(plgd) / pogd
#             else:
#                 ratio_LAE = float('inf')
#         else:
#             ratio_LAE = 0.0
#
#         ratio_LAE = min(MAX_PLAE_POII,ratio_LAE)
#     except:
#         ratio_LAE = 0
#         plgd = 0
#         pogd = 0
#         log.error("Exception calling into Bayes: ",  exc_info=True)
#
#
#     return ratio_LAE, plgd, pogd
#




#def source_prob(config, ra, dec, zs, fluxes, flux_errs, ews_obs, ew_err, c_obs, which_color, addl_fluxes,
#                addl_fluxes_error, addl_line_names, flim_file, extended_output=False):
#     """
#     Return P(LAE) = P(LAE|DATA)/P(DATA) and evidence ratio P(DATA|LAE)P(LAE)/(P(DATA|OII)P(OII))
#     given input information about the sources
#
#     Parameters
#     ----------
#     config : ConfigParser
#         configuration object
#     ra, dec, zs, fluxes, flux_errs, ews_obs, ew_err, c_obs : array
#         positions, redshifts (assuming LAE), line fluxes, errors on
#         line fluxes, equivalent widths, errors on EW and
#         colours of the sources (latter two not used!)
#     which_color : str
#         which colour is given
#     addl_fluxes, addl_fluxes_error : 2D array
#         each i of addl_fluxes[i, :] should correspond,
#         in order, to the fluxes measured for each source
#         for the emission line named at position i in
#         addl_line_names. To not use set to None (not an
#         array of None!)
#     addl_line_names : array
#         names of emission lines stored in addl_fluxes, in the
#         correct order (see above). To not use set to None (not an array of None!)
#     flim_file : str
#         file containing the flux limits
#     h : float
#         Hubbles constant/100
#     extended_output : bool
#         Return extended output
#
#     Returns
#     -------
#     posterior_odds, prob_lae_given_data : float arrays
#         posterior_odds = P(DATA|LAE)P(LAE)/(P(DATA|OII)P(OII))
#         P(LAE|DATA) = P(DATA|LAE)*P(LAE)/(P(DATA|LAE)*P(LAE) + P(DATA|OII)*P(OII))
#
#     """

#
# def xxx_bootstrap_prob_LAE(wl_obs,lineFlux,lineFlux_err=None, continuum=None, continuum_err=None, c_obs=None, which_color=None,
#             addl_wavelengths=None, addl_fluxes=None,addl_errors=None, sky_area=None, cosmo=None, lae_priors=None, ew_case=None, W_0=None,
#              z_OII=None, sigma=None, num_bootstraps=10000, confidence=0.68):
#     """
#
#     :param wl_obs:
#     :param lineFlux:
#     :param lineFlux_err:
#     :param continuum:
#     :param continuum_err:
#     :param c_obs:
#     :param which_color:
#     :param addl_wavelengths:
#     :param addl_fluxes:
#     :param addl_errors:
#     :param sky_area:
#     :param cosmo:
#     :param lae_priors:
#     :param ew_case:
#     :param W_0:
#     :param z_OII:
#     :param sigma:
#     :param num_bootstraps:
#     :param confidence: confidence interval ... commonly 0.68 or 0.95 or 0.99, etc
#     :return:
#     """
#     #sanity check
#     if confidence < 0 or confidence > 1.0:
#         log.debug("Nonsense confidence (%f) in bootstrap_prob_LAE" %(confidence))
#         return None, None, None, None
#
#     lineflux_array = np.random.normal(lineFlux,lineFlux_err,num_bootstraps)
#     continuum_array = np.random.normal(continuum,continuum_err,num_bootstraps)
#     ew_obs_array = lineflux_array / continuum_array
#
#     lae_oii_ratio_list = []
#     p_lae_list = []
#     p_oii_list = []
#
#     for lf,ew in zip(lineflux_array,ew_obs_array):
#         try:
#             lae_oii_ratio, p_lae, p_oii  = prob_LAE(wl_obs=wl_obs,
#                                    lineFlux=lf,
#                                    ew_obs=ew,
#                                    lineFlux_err=0,
#                                    ew_obs_err=0,
#                                    c_obs=None, which_color=None, addl_wavelengths=addl_wavelengths,
#                                    addl_fluxes=addl_fluxes, addl_errors=addl_errors, sky_area=None,
#                                    cosmo=None, lae_priors=None,
#                                    ew_case=None, W_0=None,
#                                    z_OII=None, sigma=None, estimate_error=False)
#
#             lae_oii_ratio_list.append(lae_oii_ratio)
#             p_lae_list.append(p_lae)
#             p_oii_list.append(p_oii)
#
#         except:
#             log.debug("Exception calling prob_LAE in bootstrap_prob_LAE",exc_info=True)
#
#         #now the "original" single call at the "exact" values for the flux and ew
#     try:
#         lae_oii_ratio, p_lae, p_oii = prob_LAE(wl_obs=wl_obs,
#                                                lineFlux=lineFlux,
#                                                ew_obs=lineFlux/continuum,
#                                                lineFlux_err=0,
#                                                ew_obs_err=0,
#                                                c_obs=None, which_color=None, addl_wavelengths=addl_wavelengths,
#                                                addl_fluxes=addl_fluxes, addl_errors=addl_errors, sky_area=None,
#                                                cosmo=None, lae_priors=None,
#                                                ew_case=None, W_0=None,
#                                                z_OII=None, sigma=None, estimate_error=False)
#     except:
#         log.debug("Exception calling standard prob_LAE in bootstrap_prob_LAE", exc_info=True)
#         return None, None, None, None
#
#     try:
#         #using biweight
#         loc = biweight.biweight_location(lae_oii_ratio_list) #the "average"
#         scale = biweight.biweight_scale(lae_oii_ratio_list)
#         ci = conf_interval(len(lae_oii_ratio_list),scale,conf=0.68)
#         ratio_LAE_list = [loc,loc-ci,loc+ci]
#         # temp:
#         #import matplotlib.pyplot as plt
#         plt.close('all')
#         plt.figure()
#         vals, bins, _ = plt.hist(lae_oii_ratio_list, bins="auto")
#         plt.title("%0.3g (%0.3g, %0.3g) bins (%d)\n min (%0.3g) max (%0.3g)"
#                   % (ratio_LAE_list[0], ratio_LAE_list[1], ratio_LAE_list[2], len(bins) - 1, min(vals), max(vals)))
#         plt.savefig("plae_bw_hist_%d.png" %num_bootstraps)
#     except:
#         log.debug("Exception calling biweight or conf_interval in bootstap_prob_LAE", exc_info=True)
#         return lae_oii_ratio, p_lae, p_oii, None
#
#
#
#     try:
#         if True:
#             mean_cntr, var_cntr, std_cntr = bayes_mvs(lae_oii_ratio_list, alpha=0.68)
#             ratio_LAE_list = [lae_oii_ratio, None, None]
#             if not np.isnan(mean_cntr[0]):
#                 ratio_LAE_list[0] = mean_cntr[0]
#                 ratio_LAE_list[1] = mean_cntr[1][0]
#                 ratio_LAE_list[2] = mean_cntr[1][1]
#                 #temp:
#                 #import matplotlib.pyplot as plt
#                 plt.close('all')
#                 plt.figure()
#                 vals, bins, _ = plt.hist(lae_oii_ratio_list,bins="auto")
#                 plt.title("%0.3g (%0.3g, %0.3g) bins (%d)\n min (%0.3g) max (%0.3g)"
#                           % (ratio_LAE_list[0], ratio_LAE_list[1], ratio_LAE_list[2],len(bins)-1,min(vals),max(vals)))
#                 plt.savefig("plae_mean_hist_%d.png" %num_bootstraps)
#             else:
#                 ratio_LAE_list[1] = lae_oii_ratio
#                 ratio_LAE_list[2] = lae_oii_ratio
#
#         mean_cntr, var_cntr, std_cntr = bayes_mvs(p_lae_list, alpha=0.68)
#         plgd_list = [p_lae,None,None]
#         if not np.isnan(mean_cntr[0]):
#             plgd_list[0] = mean_cntr[0]
#             plgd_list[1] = mean_cntr[1][0]
#             plgd_list[2] = mean_cntr[1][1]
#         else:
#             plgd_list[1] = p_lae
#             plgd_list[2] = p_lae
#
#         mean_cntr, var_cntr, std_cntr = bayes_mvs(p_oii_list, alpha=0.68)
#         pogd_list = [p_oii,None,None]
#         if not np.isnan(mean_cntr[0]):
#             pogd_list[0] = mean_cntr[0]
#             pogd_list[1] = mean_cntr[1][0]
#             pogd_list[2] = mean_cntr[1][1]
#         else:
#             pogd_list[1] = p_oii
#             pogd_list[2] = p_oii
#     except:
#         log.debug("Exception calling bayes_mvs in bootstrap_prob_LAE", exc_info=True)
#         return lae_oii_ratio, p_lae, p_oii, None
#
#     log.info("Bootstrap PLAE: %0.4g (%0.4g, %0.4g) " %(ratio_LAE_list[0],ratio_LAE_list[1],ratio_LAE_list[2]))
#
#     return lae_oii_ratio, p_lae, p_oii, {'ratio':ratio_LAE_list,'plgd':plgd_list,'pogd':pogd_list}
#
#

def prob_LAE(wl_obs,lineFlux,lineFlux_err=None, ew_obs=None, ew_obs_err=None, c_obs=None, which_color=None, addl_wavelengths=None,
             addl_fluxes=None,addl_errors=None, sky_area=None, cosmo=None, lae_priors=None, ew_case=None, W_0=None,
             z_OII=None, sigma=None,estimate_error=False):


#temporarary ... call both and compare
    # old_ratio_LAE, old_plgd, old_pogd = prob_LAE_old(wl_obs,lineFlux,ew_obs,c_obs, which_color=which_color,
    #                                     addl_wavelengths=addl_wavelengths, addl_fluxes=addl_fluxes,addl_errors=addl_errors,
    #         sky_area=sky_area, cosmo=cosmo, lae_priors=lae_priors, ew_case=ew_case, W_0=W_0, z_OII=z_OII, sigma=sigma)
    #


    global UNIVERSE_CONFIG, FLUX_LIMIT_FN
    if UNIVERSE_CONFIG is None:
        try:
            config_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), G.RELATIVE_PATH_UNIVERSE_CONFIG)
            UNIVERSE_CONFIG = RawConfigParser()
            UNIVERSE_CONFIG.read(config_fn)
            log.debug("Load universe config for LAE/OII discriminator: %s" %config_fn)

            FLUX_LIMIT_FN = os.path.join(os.path.dirname(os.path.realpath(__file__)),G.RELATIVE_PATH_FLUX_LIM_FN)
            log.debug("Load flux limit filename for LAE/OII discriminator: %s" % FLUX_LIMIT_FN)

            # don't need to do this ... is performed in source_prob call
            #COSMOLOGY = generate_cosmology_from_config(UNIVERSE_CONFIG)
        except:
            log.warning("Exception loading LAE/OII discriminator config",exc_info=True)
            print("Exception loading LAE/OII discriminator config")

    # posterior_odds = 0.0
    # prob_lae_given_data = 0.0

    #build up parameters (need to be numpy arrays for the call)
    ra = None #have no meaning in this case? could set to [100.0] and [0.0] per example?
    dec = None
    z_LyA = wl_obs / G.LyA_rest - 1.0
    z_OII = wl_obs / G.OII_rest - 1.0

    #suppress sign of EW (always wants positive)
    # (and, note this is the OBSERVERED EqW, not EW/(1+z_LAE) ... that calc occurs inside the calls)
    ew_obs = abs(ew_obs)

    if lineFlux_err is None:
        lineFlux_err = 0.0

    if ew_obs_err is None:
        ew_obs_err = 0.0


    #convert additional wavelengths into names for the call
    #from the UNIVERSE_CONFIG file
    known_lines = UNIVERSE_CONFIG.items("wavelengths") #array of tuples (name,wavelength)

    extra_fluxes = []
    extra_fluxes_err = []
    extra_fluxes_name = []

    #all the extra lines used by the Bayes code are visible in our range only if OII is the primary
    #so assume OII and shift to rest frame

    # LAE = 1215.668
    # OII = 3727.45
    # NeIII = 3869.00
    # H_beta = 4861.32
    # OIII4959 = 4958.91
    # OIII5007 = 5006.84

    #iterate over all in addl_wavelengths, if +/- (1? 2? AA ... what are we using elsewhere?) assign name
    #if no match, toss the additional flux, etc

    wl_unc = 2.0 #AA

    if (addl_wavelengths is None) or (addl_fluxes is None) or (addl_errors is None):
        addl_wavelengths = []
        addl_fluxes = []
        addl_errors = []


    try:
        for n, w in known_lines:
            w_oii = float(w) * (z_OII + 1.)
            for i in range(len(addl_fluxes)):
                if abs(w_oii-addl_wavelengths[i]) < wl_unc:
                    extra_fluxes.append(addl_fluxes[i])
                    extra_fluxes_name.append(n)
                    try:
                        extra_fluxes_err.append(addl_errors[i])
                    except:
                        extra_fluxes_err.append(0.0)
                        log.warning("Exception (non-fatal) building extra line fluxes in line_prob.py. " + \
                                    "Unable to set flux uncertainty.", exc_info=True)

                    break
    except:
        log.error("Exception building extra line fluxes in line_prob.py.", exc_info=True)
        if estimate_error:
            return 0,0,0,{}
        else:
            return 0,0,0

    plae_errors = {} #call classifier multiple times and get an error estimate on the PLAE/POII ratio

    #at least for now, just call for EqW ... that is the biggest error source
    #flux_array_range = [lineFlux]

    if estimate_error and ew_obs_err:
        ew_list = [ew_obs,ew_obs-ew_obs_err,ew_obs+ew_obs_err] # so: value, -error, +error
    else:
        ew_list = [ew_obs]

    posterior_odds_list = []
    prob_lae_given_data_list = []

    for e in ew_list:
        try:
            posterior_odds, prob_lae_given_data = LineClassifierPro.source_prob(UNIVERSE_CONFIG,
                                                          np.array([ra]), np.array([dec]), np.array([z_LyA]),
                                                          np.array([lineFlux]), np.array([lineFlux_err]),
                                                          np.array([e]), np.array([ew_obs_err]),
                                                          c_obs=None, which_color=None,
                                                          addl_fluxes=np.array(extra_fluxes),
                                                          addl_fluxes_error=np.array(extra_fluxes_err),
                                                          addl_line_names=np.array(extra_fluxes_name),
                                                          flim_file=FLUX_LIMIT_FN,extended_output=False)


            if isinstance(posterior_odds,list) or isinstance(posterior_odds,np.ndarray):
                if len(posterior_odds) == 1:
                    posterior_odds = posterior_odds[0]
                else:
                    log.info("Weird. posterior_odds %s" %(posterior_odds))

            if isinstance(prob_lae_given_data,list) or isinstance(prob_lae_given_data,np.ndarray):
                if len(prob_lae_given_data) == 1:
                    prob_lae_given_data = prob_lae_given_data[0]
                else:
                    log.info("Weird. prob_lae_given_data %s" %(prob_lae_given_data))

            posterior_odds_list.append(posterior_odds)
            prob_lae_given_data_list.append(prob_lae_given_data)

        except:
            log.error("Exception calling LineClassifierPro::source_prob()", exc_info=True)

    ###############################
    #just testing ....
    ###############################
    if False:

        # @pytest.mark.parametrize("z, fluxes, ew_obs, addl_fluxes, addl_names, e_ratio, e_prob_lae",
        #                          [
        #                              (1.9, 9e-17, 40, None, None, 1e+32, 1.0),
        #                              (2.48, 9e-17, 40, None, None, 9.0769011810393501, 0.90076314314944406),
        #                              (3.18, 9e-17, 40, None, None, 0.17790889751426178, 0.151037909544365),

        #                              (2.08, 9e-17, 40, [[5e-17]], ["NeIII"], 10.917948575339162, 0.91609294219734949),

        #                              (2.12, 9e-17, 40, [[6e-17]], ["H_beta"], 2.2721726484396545e-09,
        #                               2.2721726536024229e-09),
        #                              (2.08, 9e-17, 40, [[7e-17], [9e-17 * 4.752 / 1.791]], ["OIII4959", "OIII5007"], 0.0,
        #                               0.0)
        #                          ])

        #flim_file = '/home/dustin/code/python/elixer/line_classifier_install/tests/data/Line_flux_limit_5_sigma_baseline.dat'

        #posterior_odds, prob_lae_given_data = LineClassifierPro.source_prob(UNIVERSE_CONFIG, [100.0], [0.0],
        #                                                                [2.08], array([9e-17]),
        #                                                array([0.0]), [40.], [0.0], None, None, None, None, None,
        #                                                FLUX_LIMIT_FN)

        try:
            from elixer.line_classifier.misc.tools import read_flim_file
        except:
            from line_classifier.misc.tools import read_flim_file
        flims = read_flim_file(FLUX_LIMIT_FN)
        errors = []
        for x in ["NeIII"]:
            zoii = (1.0 + 2.08) * UNIVERSE_CONFIG.getfloat("wavelengths", "LAE") / UNIVERSE_CONFIG.getfloat("wavelengths", "OII") - 1.0
            lambda_ = UNIVERSE_CONFIG.getfloat("wavelengths", x) * (1.0 + zoii)
            errors.append(0.2 * flims(lambda_ / UNIVERSE_CONFIG.getfloat("wavelengths", "LAE") - 1.0))
        errors = np.array(errors)


        posterior_odds, prob_lae_given_data = LineClassifierPro.source_prob(UNIVERSE_CONFIG, ra,dec,
                                                          np.array([2.08]), np.array([9e-17]), np.array([0.0]),
                                                          [40.0], [0.0], None, None,
                                                          np.array([5e-17]), errors, np.array(["NeIII"]), FLUX_LIMIT_FN)
            #LineClassifierPro.source_prob(extended_output=False)

    ################################
    # end testing
    ###############################

    pogd_list = []
    plgd_list = []
    ratio_LAE_list = []

    for posterior_odds in posterior_odds_list:

        if (posterior_odds is not None) and (posterior_odds != 0):
            pogd = np.float(prob_lae_given_data) / posterior_odds
        else:
            pogd = 0.

        plgd = np.float(prob_lae_given_data)
        ratio_LAE = np.float(min(MAX_PLAE_POII, posterior_odds))
        ratio_LAE = np.float(max(ratio_LAE,MIN_PLAE_POII))

        ratio_LAE_list.append(ratio_LAE)
        plgd_list.append(plgd)
        if type(pogd) != float:
            pogd_list.append(pogd.value)
        else:
            pogd_list.append(pogd)

    #temporary -- compare results and note if the new method disagrees with the old
    # if old_ratio_LAE + ratio_LAE > 0.2: #if they are both small, don't bother
    #     if abs((old_ratio_LAE - ratio_LAE)/(0.5*(old_ratio_LAE+ratio_LAE))) > 0.01:
    #         msg = "Warning! Difference in P(LAE)/P(OII).\n  Original: P(LAE|data) = %f, P(OII|data) = %f, ratio = %f" \
    #               "\n  New: P(LAE|data) = %f, P(OII|data) = %f, ratio = %f" \
    #               %(old_plgd,old_pogd,old_ratio_LAE,plgd,pogd,ratio_LAE)
    #
    #         log.warning("***" + msg)
    #         #print(msg)

    if estimate_error:
        return ratio_LAE_list[0], plgd_list[0], pogd_list[0], {'ratio':ratio_LAE_list,'plgd':plgd_list,'pogd':pogd_list}
    else:
        return ratio_LAE_list[0], plgd_list[0], pogd_list[0]


def mc_prob_LAE(wl_obs,lineFlux,lineFlux_err=None, continuum=None, continuum_err=None, ew_obs=None, ew_obs_err=None,
                c_obs=None, which_color=None, addl_wavelengths=None, addl_fluxes=None,addl_errors=None, sky_area=None,
                cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None, sigma=None,
                num_mc=G.MC_PLAE_SAMPLE_SIZE, confidence=G.MC_PLAE_CONF_INTVL):
    """

    :param wl_obs:
    :param lineFlux:
    :param lineFlux_err:
    :param continuum:
    :param continuum_err:
    :param ew_obs: reconstruct continuum if not provided
    :param ew_obs_err: reconstruct continuum_err if not provided
    :param c_obs:
    :param which_color:
    :param addl_wavelengths:
    :param addl_fluxes:
    :param addl_errors:
    :param sky_area:
    :param cosmo:
    :param lae_priors:
    :param ew_case:
    :param W_0:
    :param z_OII:
    :param sigma:
    :param num_mc:
    :param confidence: confidence interval ... commonly 0.68 or 0.95 or 0.99, etc
    :return:
    """

    try:

        #sanity check
        if confidence < 0 or confidence > 1.0:
            log.debug("Nonsense confidence (%f) in mc_prob_LAE" %(confidence))
            return None, None, None, None

        if (continuum is None):
            if (ew_obs is None):
                log.debug("Insufficient info for mc_prob_LAE, continuum and ew_obs not provided")
                return 0, 0, 0, {'ratio':[0,0,0],'plgd':[0],'pogd':[0]}
            else: #build continuum and continuum error from ew
                if ew_obs > 0:
                    continuum = lineFlux / ew_obs
                    if (ew_obs_err is not None) and (ew_obs_err != 0):
                        #this can be negative if the ew error is unreasonably small compared to the lineflux error
                        #in which case the sqrt fails, but set it to 0.0 and the error will be dominated by the lineflux anyway
                        continuum_err = continuum * np.sqrt(max(0.0,(ew_obs_err/ew_obs)**2 - (lineFlux_err/lineFlux)**2))
                    else:
                        continuum_err = 0.0
                else:
                    log.debug("Invalid lineflux or continuum or ew_obs")
                    return 0, 0, 0, {'ratio':[0,0,0],'plgd':[0],'pogd':[0]}

        if (lineFlux <= 0) or (continuum <= 0):
            log.debug("Invalid lineflux or continuum")
            return 0, 0, 0, {'ratio':[0,0,0],'plgd':[0],'pogd':[0]}

        if (lineFlux_err is None):
            log.debug("LineFlux error is None")
            lineFlux_err = 0

        if (lineFlux_err < 0):
            log.debug("LineFlux error < 0")
            lineFlux_err = 0

        if (continuum_err is None):
            log.debug("Continuum error is None")
            continuum_err = 0

        if (continuum_err < 0):
            log.debug("Continuum error < 0")
            continuum_err = 0

        if continuum_err == lineFlux_err == 0:
            log.debug("Continuum error and Lineflux error set to zero. Single run only (no mc).")
            num_mc = 1

        _max_sample_retry = 10 #number of attemps to get a valid lineflux and continuum (both must be positive)
        log.debug("Sampling (%d) PLAE/POII ... " %(num_mc))
        # lineflux_array = np.random.normal(lineFlux,lineFlux_err,num_mc)
        # continuum_array = np.random.normal(continuum,continuum_err,num_mc)
        # ew_obs_array = lineflux_array / continuum_array

        lae_oii_ratio_list = []
        p_lae_list = []
        p_oii_list = []

        global UNIVERSE_CONFIG, FLUX_LIMIT_FN
        if UNIVERSE_CONFIG is None:
            try:
                config_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), G.RELATIVE_PATH_UNIVERSE_CONFIG)
                UNIVERSE_CONFIG = RawConfigParser()
                UNIVERSE_CONFIG.read(config_fn)
                log.debug("Load universe config for LAE/OII discriminator: %s" %config_fn)

                FLUX_LIMIT_FN = os.path.join(os.path.dirname(os.path.realpath(__file__)),G.RELATIVE_PATH_FLUX_LIM_FN)
                log.debug("Load flux limit filename for LAE/OII discriminator: %s" % FLUX_LIMIT_FN)

                # don't need to do this ... is performed in source_prob call
                #COSMOLOGY = generate_cosmology_from_config(UNIVERSE_CONFIG)
            except:
                log.warning("Exception loading LAE/OII discriminator config",exc_info=True)
                print("Exception loading LAE/OII discriminator config")

        # posterior_odds = 0.0
        # prob_lae_given_data = 0.0

        #build up parameters (need to be numpy arrays for the call)
        ra = None #have no meaning in this case? could set to [100.0] and [0.0] per example?
        dec = None
        z_LyA = wl_obs / G.LyA_rest - 1.0
        z_OII = wl_obs / G.OII_rest - 1.0

        #convert additional wavelengths into names for the call
        #from the UNIVERSE_CONFIG file
        known_lines = UNIVERSE_CONFIG.items("wavelengths") #array of tuples (name,wavelength)

        extra_fluxes = []
        extra_fluxes_err = []
        extra_fluxes_name = []

        #all the extra lines used by the Bayes code are visible in our range only if OII is the primary
        #so assume OII and shift to rest frame

        # LAE = 1215.668
        # OII = 3727.45
        # NeIII = 3869.00
        # H_beta = 4861.32
        # OIII4959 = 4958.91
        # OIII5007 = 5006.84

        #iterate over all in addl_wavelengths, if +/- (1? 2? AA ... what are we using elsewhere?) assign name
        #if no match, toss the additional flux, etc

        wl_unc = 2.0 #AA

        if (addl_wavelengths is None) or (addl_fluxes is None) or (addl_errors is None):
            addl_wavelengths = []
            addl_fluxes = []
            addl_errors = []


        try:
            for n, w in known_lines:
                w_oii = float(w) * (z_OII + 1.)
                for i in range(len(addl_fluxes)):
                    if abs(w_oii-addl_wavelengths[i]) < wl_unc:
                        extra_fluxes.append(addl_fluxes[i])
                        extra_fluxes_name.append(n)
                        try:
                            extra_fluxes_err.append(addl_errors[i])
                        except:
                            extra_fluxes_err.append(0.0)
                            log.warning("Exception (non-fatal) building extra line fluxes in line_prob.py. " + \
                                        "Unable to set flux uncertainty.", exc_info=True)

                        break
        except:
            log.error("Exception building extra line fluxes in line_prob.py.", exc_info=True)
            if estimate_error:
                return 0, 0, 0, {'ratio':[0,0,0],'plgd':[0],'pogd':[0]}
            else:
                return 0,0,0

        plae_errors = {} #call classifier multiple times and get an error estimate on the PLAE/POII ratio

        #at least for now, just call for EqW ... that is the biggest error source
        #flux_array_range = [lineFlux]

        posterior_odds_list = []
        prob_lae_given_data_list = []

        setup = {} #first run setup date for the LineClassifierPro ... will be populated by source_prob on first call
                   #then passed in on subsequent calls to speed up processing
        #ct = 0
        #for lf,ew in zip(lineflux_array,ew_obs_array):
        for i in range(num_mc):
            tryagain = 0
            lf = 0
            cn = 0
            ew = 0
            while tryagain < _max_sample_retry:
                lf = np.random.normal(lineFlux, lineFlux_err)
                cn = np.random.normal(continuum, continuum_err)
                if lf > 0 and cn > 0:
                    ew  = lf / cn
                    break
                else:
                    tryagain += 1

            if not (tryagain < _max_sample_retry):
                log.info("Failed to properly sample lineflux and/or continuum. Cannot continue.")
                break

            try:
                #ct += 1
                #log.debug("%d"%ct)
                posterior_odds, prob_lae_given_data,setup  = LineClassifierPro.source_prob(UNIVERSE_CONFIG,
                                                                                    np.array([ra]), np.array([dec]),
                                                                                    np.array([z_LyA]),
                                                                                    np.array([lf]),
                                                                                    np.array([0.0]),
                                                                                    np.array([ew]), np.array([0.0]),
                                                                                    c_obs=None, which_color=None,
                                                                                    addl_fluxes=np.array(extra_fluxes),
                                                                                    addl_fluxes_error=np.array(
                                                                                        extra_fluxes_err),
                                                                                    addl_line_names=np.array(
                                                                                        extra_fluxes_name),
                                                                                    flim_file=FLUX_LIMIT_FN,
                                                                                    extended_output=False,
                                                                                    setup=setup)


                if isinstance(posterior_odds,list) or isinstance(posterior_odds,np.ndarray):
                    if len(posterior_odds) == 1:
                        posterior_odds = posterior_odds[0]
                    else:
                        log.info("Weird. posterior_odds %s" %(posterior_odds))

                if isinstance(prob_lae_given_data,list) or isinstance(prob_lae_given_data,np.ndarray):
                    if len(prob_lae_given_data) == 1:
                        prob_lae_given_data = prob_lae_given_data[0]
                    else:
                        log.info("Weird. prob_lae_given_data %s" %(prob_lae_given_data))

                if (posterior_odds is not None) and (posterior_odds != 0):
                    pogd = np.float(prob_lae_given_data) / posterior_odds
                else:
                    pogd = 0.

                plgd = np.float(prob_lae_given_data)
                pogd = np.float(pogd)

                #the base code can limit this to 1000.0 (explicitly) if P(OII|Data) == 0,
                #so we DO need to force these to the max of 1000.0 (which could otherwise be exceeded
                #if P(OII|data) > 0 but very small)

                posterior_odds = float(posterior_odds)
                posterior_odds = max(MIN_PLAE_POII,min(MAX_PLAE_POII,posterior_odds))

                lae_oii_ratio_list.append(float(posterior_odds))
                p_lae_list.append(plgd)
                p_oii_list.append(pogd)

            except:
                log.debug("Exception calling prob_LAE in mc_prob_LAE",exc_info=True)

        #we were unable to get a sampling, so just call once with the exact values
        if len(lae_oii_ratio_list) == 0:
            try:
                lf = lineFlux
                ew = lineFlux/continuum
                posterior_odds, prob_lae_given_data,setup  = LineClassifierPro.source_prob(UNIVERSE_CONFIG,
                                                                                    np.array([ra]), np.array([dec]),
                                                                                    np.array([z_LyA]),
                                                                                    np.array([lf]),
                                                                                    np.array([0.0]),
                                                                                    np.array([ew]), np.array([0.0]),
                                                                                    c_obs=None, which_color=None,
                                                                                    addl_fluxes=np.array(extra_fluxes),
                                                                                    addl_fluxes_error=np.array(
                                                                                        extra_fluxes_err),
                                                                                    addl_line_names=np.array(
                                                                                        extra_fluxes_name),
                                                                                    flim_file=FLUX_LIMIT_FN,
                                                                                    extended_output=False,
                                                                                    setup=setup)


                if isinstance(posterior_odds,list) or isinstance(posterior_odds,np.ndarray):
                    if len(posterior_odds) == 1:
                        posterior_odds = posterior_odds[0]
                    else:
                        log.info("Weird. posterior_odds %s" %(posterior_odds))

                if isinstance(prob_lae_given_data,list) or isinstance(prob_lae_given_data,np.ndarray):
                    if len(prob_lae_given_data) == 1:
                        prob_lae_given_data = prob_lae_given_data[0]
                    else:
                        log.info("Weird. prob_lae_given_data %s" %(prob_lae_given_data))

                if (posterior_odds is not None) and (posterior_odds != 0):
                    pogd = np.float(prob_lae_given_data) / posterior_odds
                else:
                    pogd = 0.

                plgd = np.float(prob_lae_given_data)
                pogd = np.float(pogd)

                log.debug("Sampling (%d) PLAE/POII ... done. Unable to sample. No details returned." % (num_mc))
                return float(posterior_odds), plgd, pogd, None

            except:
                log.debug("Exception calling prob_LAE in mc_prob_LAE",exc_info=True)

        try:
            #lae_oii_ratio_list = np.array(lae_oii_ratio_list)
            #using biweight
            log.debug("Biweight ...")

            try:
                loc = biweight.biweight_location(lae_oii_ratio_list)
                hi,lo = conf_interval_asym(lae_oii_ratio_list,loc)
                #the actual std can be huge and is dodgy to compute since we are capped 1000 - 0.001
                #so, use the quantiles in the middle 0.68 (rouhgly +/- 1sd IF this were normal distro
                adj_std = 0.5 * (np.quantile(lae_oii_ratio_list,0.16) + np.quantile(lae_oii_ratio_list,0.84))
                if (hi is None) or (lo is None):
                    log.debug("Unable to perform direct asym confidence interval. Reverting to old method.")
                    loc = biweight.biweight_location(lae_oii_ratio_list)  # the "average"
                    scale = biweight.biweight_scale(lae_oii_ratio_list)
                    ci = conf_interval(len(lae_oii_ratio_list), scale * np.sqrt(num_mc), conf=confidence)
                    if ci is not None:
                        ratio_LAE_list = [loc, loc - ci, loc + ci, adj_std]  # np.nanstd(lae_oii_ratio_list)]
                    else:
                        log.warning("Confidence Interval is None in line_prob::mc_prob_LAE (p1)")
                        ratio_LAE_list = [loc, 0.001, 1000.0, adj_std]
                else:
                    ratio_LAE_list = [loc, lo, hi,adj_std] #np.nanstd(lae_oii_ratio_list)]
            except:
                log.debug("Unable to perform direct asym confidence interval. Reverting to old method.")
                loc = biweight.biweight_location(lae_oii_ratio_list)  # the "average"
                scale = biweight.biweight_scale(lae_oii_ratio_list)
                ci = conf_interval(len(lae_oii_ratio_list), scale * np.sqrt(num_mc), conf=confidence)
                adj_std = 0.5 * (np.quantile(lae_oii_ratio_list, 0.16) + np.quantile(lae_oii_ratio_list, 0.84))
                if ci is not None:
                    ratio_LAE_list = [loc, loc - ci, loc + ci,adj_std]#np.nanstd(lae_oii_ratio_list)]
                else:
                    log.warning("Confidence Interval is None in line_prob::mc_prob_LAE (p2)")
                    ratio_LAE_list = [loc,0.001,1000.0,adj_std]

            if False:
                try: #this data is often skewed, so run bootstraps to normalize and take the confidence interval there
                    loc,ci = elixer_biweight.bootstrap_confidence_interval(lae_oii_ratio_list,confidence=confidence)
                    if (loc is None) or (ci is None):
                        log.debug("Unable to perform confidence interval via bootstrap. Reverting to old method.")
                        loc = biweight.biweight_location(lae_oii_ratio_list)  # the "average"
                        scale = biweight.biweight_scale(lae_oii_ratio_list)
                        ci = conf_interval(len(lae_oii_ratio_list), scale * np.sqrt(num_mc), conf=confidence)
                        ratio_LAE_list = [loc, loc - ci, loc + ci,np.nanstd(lae_oii_ratio_list)]
                except: #if it fails, fall back to the old way (and assume a normal distribution)
                    loc = biweight.biweight_location(lae_oii_ratio_list)  # the "average"
                    scale = biweight.biweight_scale(lae_oii_ratio_list)
                    ci = conf_interval(len(lae_oii_ratio_list), scale * np.sqrt(num_mc), conf=confidence)
                    ratio_LAE_list = [loc, loc - ci, loc + ci,np.nanstd(lae_oii_ratio_list)]


            #??? should the 'scale' by multiplied by sqrt(# samples) to be consistent?
            #??? I think the sigma_mc == true sigma / sqrt(# samples) (kind of backward from sample vs population)
            #ci = conf_interval(len(lae_oii_ratio_list), scale, conf=confidence)
            #ci = conf_interval(len(lae_oii_ratio_list),scale*np.sqrt(num_mc),conf=confidence)

            log.debug("Raw Biweight: %0.4g (%0.4g, %0.4g), min (%0.4g) max (%0.4g) std (%0.4g), Q1 (%0.4g) Q2 (%0.4g) Q3 (%0.4g)"
                      % (ratio_LAE_list[0], ratio_LAE_list[1], ratio_LAE_list[2], min(lae_oii_ratio_list), max(lae_oii_ratio_list),ratio_LAE_list[3],
                         np.quantile(lae_oii_ratio_list,0.25),np.quantile(lae_oii_ratio_list,0.50),np.quantile(lae_oii_ratio_list,0.75))
                      )

            try:
                mean_cntr, var_cntr, std_cntr = bayes_mvs(lae_oii_ratio_list, alpha=confidence)
                log.debug("Bayes MVS: %0.4g (%0.4g, %0.4g), min (%0.4g) max (%0.4g) std(%0.4g), Q1 (%0.4g) Q2 (%0.4g) Q3 (%0.4g)"
                          % (mean_cntr[0], mean_cntr[1][0], mean_cntr[1][1], min(lae_oii_ratio_list), max(lae_oii_ratio_list), std_cntr[0],
                             np.quantile(lae_oii_ratio_list,0.25),np.quantile(lae_oii_ratio_list,0.50),np.quantile(lae_oii_ratio_list,0.75))
                          )
            except:
                pass

            # for i in range(len(ratio_LAE_list)): #force the range to be between MIN_PLAE_POII and MAX_PLAE_POII
            #     ratio_LAE_list[i] = max(min(MAX_PLAE_POII,ratio_LAE_list[i]),MIN_PLAE_POII)

            # log.debug("Limited Biweight: %0.3g (%0.3g, %0.3g) min (%0.3g) max (%0.3g)"
            #           % (ratio_LAE_list[0], ratio_LAE_list[1], ratio_LAE_list[2], min(lae_oii_ratio_list),
            #              max(lae_oii_ratio_list)))

            # temp:
            if False:
                log.debug("plotting ..." )
                plt.close('all')
                #plt.figure()
                vals, bins, _ = plt.hist(lae_oii_ratio_list, bins="auto")
                plt.title("%0.3g (%0.3g, %0.3g) bins (%d)\n min (%0.3g) max (%0.3g) "
                          % (ratio_LAE_list[0], ratio_LAE_list[1], ratio_LAE_list[2], len(bins) - 1, min(vals), max(vals)))
                plt.savefig("plae_bw_hist_%d.png" %num_mc)
        except:
            log.debug("Exception calling biweight or conf_interval in mc_prob_LAE", exc_info=True)
            return None, None, None, None

        log.debug("Sampling (%d) PLAE/POII ... done" % (num_mc))
        return ratio_LAE_list[0], p_lae_list[0], p_oii_list[0], {'ratio':ratio_LAE_list,'plgd':p_lae_list,'pogd':p_oii_list}
    except:
        log.debug("Exception calling mc_prob_LAE", exc_info=True)
        return None, None, None, None



