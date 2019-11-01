#import Bayes.bayesian
#import Bayes.nb
import math
try:
    from ConfigParser import RawConfigParser
except:
    from configparser import RawConfigParser
import os
import sys

try:
    from elixer import global_config as G
    import elixer.line_classifier.probs.classification_prob as LineClassifierPro
#    import elixer.line_classifier.probs.classification_prob_leung as LineClassifierPro_Leung
except:
    import global_config as G
    import line_classifier.probs.classification_prob as LineClassifierPro
#    import line_classifier.probs.classification_prob_leung as LineClassifierPro_Leung

import numpy as np
#numpy import array
#don't need to do this ... is performed in source_prob call
#from line_classifier.misc.tools import generate_cosmology_from_config, read_flim_file


MAX_PLAE_POII = 1000
MIN_PLAE_POII = 0.001
UNIVERSE_CONFIG = None
FLUX_LIMIT_FN = None
COSMOLOGY = None

#log = G.logging.getLogger('line_prob_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('line_prob_logger')
log.setlevel(G.logging.DEBUG)



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

def prob_LAE(wl_obs,lineFlux,lineFlux_err=None, ew_obs=None, ew_obs_err=None, c_obs=None, which_color=None, addl_wavelengths=None,
             addl_fluxes=None,addl_errors=None, sky_area=None, cosmo=None, lae_priors=None, ew_case=None, W_0=None,
             z_OII=None, sigma=None):


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

    posterior_odds = 0.0
    prob_lae_given_data = 0.0

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
        return 0,0,0


    try:
        posterior_odds, prob_lae_given_data = LineClassifierPro.source_prob(UNIVERSE_CONFIG,
                                                      np.array([ra]), np.array([dec]), np.array([z_LyA]),
                                                      np.array([lineFlux]), np.array([lineFlux_err]),
                                                      np.array([ew_obs]), np.array([ew_obs_err]),
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


    if (posterior_odds is not None) and (posterior_odds != 0):
        pogd = prob_lae_given_data / posterior_odds
    else:
        pogd = 0.

    plgd = np.float(prob_lae_given_data)
    ratio_LAE = np.float(min(MAX_PLAE_POII, posterior_odds))
    ratio_LAE = np.float(max(ratio_LAE,MIN_PLAE_POII))

    #temporary -- compare results and note if the new method disagrees with the old
    # if old_ratio_LAE + ratio_LAE > 0.2: #if they are both small, don't bother
    #     if abs((old_ratio_LAE - ratio_LAE)/(0.5*(old_ratio_LAE+ratio_LAE))) > 0.01:
    #         msg = "Warning! Difference in P(LAE)/P(OII).\n  Original: P(LAE|data) = %f, P(OII|data) = %f, ratio = %f" \
    #               "\n  New: P(LAE|data) = %f, P(OII|data) = %f, ratio = %f" \
    #               %(old_plgd,old_pogd,old_ratio_LAE,plgd,pogd,ratio_LAE)
    #
    #         log.warning("***" + msg)
    #         #print(msg)


    return ratio_LAE, plgd, pogd
