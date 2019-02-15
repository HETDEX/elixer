#API Wrapper for classification (mostly P(LAE)/P(OII))
import global_config as G
import line_prob

log = G.Global_Logger('classify_logger')
log.setlevel(G.logging.DEBUG)

G.logging.basicConfig(filename="elixer_classify.log", level=G.LOG_LEVEL, filemode='w')

def plae_poii(line_wave,line_flux,line_flux_err, eqw_obs, eqw_obs_err,
              addl_wave = [], addl_flux = [], addl_flux_err = []):
    """
    Returns the Bayesian P(LAE)/P(OII) from EW distributions of LAEs and OII galaxies from
    Leung+ 2017

    Maximum P(LAE)/P(OII) is capped at 999. There is no minimum value but 0.0 represents an
    error or inability to calcluate the ratio.


    :param line_wave:  observed wavelength of the emission line in angstroms
    :param line_flux:  observed line flux (cgs)
    :param line_flux_err: observed line flux error (cgs)
    :param eqw_obs: observed equivalent width
    :param eqw_obs_err: observed equivalent width error
    :param addl_wave: array of additional emission line wavelengths in angstroms
    :param addl_flux: array of additional emission line fluxes (cgs)
    :param addl_flux_err: array of additional emission line flux errors (cgs)
    :return: P(LAE)/P(OII), P(LAE|data), P(OII|data) a zero value is an unknown
    """

    ratio, p_lae, p_oii = line_prob.prob_LAE(wl_obs=line_wave,
                                             lineFlux=line_flux,
                                             lineFlux_err=line_flux_err,
                                             ew_obs=eqw_obs,
                                             ew_obs_err=eqw_obs_err,
                                             addl_wavelengths=addl_wave,
                                             addl_fluxes=addl_flux,
                                             addl_errors=addl_flux_err,
                                             c_obs=None, which_color=None,
                                             sky_area=None, cosmo=None, lae_priors=None,
                                             ew_case=None, W_0=None, z_OII=None, sigma=None)


    #todo: any logging, etc?
    return ratio, p_lae, p_oii