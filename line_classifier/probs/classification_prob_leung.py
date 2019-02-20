"""

Carry out a Bayesian classification of a line source based off
of Andrew Leung's paper (Leung+ 2017) and code.

Author: Daniel Farrow 2018 (parts adapted from Andrew Leung's code)


"""

from __future__ import absolute_import, print_function


from numpy import pi, square, exp, array, power, zeros, ones, isnan, sqrt, abs, errstate, divide
from numpy import any as nany
from astropy.table import Table
from scipy.stats import norm
from scipy.special import gammainc, gammaincc, gamma
from line_classifier.lfs_ews.luminosity_function import LuminosityFunction, gamma_integral_limits
from line_classifier.lfs_ews.equivalent_width import EquivalentWidthAssigner
from line_classifier.misc.tools import read_flim_file, generate_cosmology_from_config

import global_config as G
_logger = G.Global_Logger("prob")
_logger.setlevel(G.logging.INFO)

#import logging
#_logger = logging.getLogger("prob")
#logging.basicConfig(level=logging.INFO)

class TooFaintForLimitsException(Exception):
    pass

class NegativeProbException(Exception):
    pass

class UnrecognizedSourceException(Exception):
    pass

class LFIntegrator(object):
    """
    Class to integrate the luminosity function,
    either analytically or numerically when 
    using detection curve

    TODO: Replace flim file with sensitivity cube?

    Parameters
    ----------
    lf : line_classification.lfs_ews.luminosity_function:LuminosityFunction
        the luminosity function to integrate
    flim_file : str
        the filename of the file containing
        flux limits
        integrate up to a flux limit
    flux_errors : pyhetdex.selfunc.flux_errors:FluxErrors
        Returns flux errors for a given ra, dec, 
        lambda_. Pass to apply Eddington bias to
        luminosity function

    """

    def __init__(self, lf, flim_file, flux_errors=None): 

      
       self.lf = lf
       self.flims = read_flim_file(flim_file)
       self.flux_errors = flux_errors

    def integrate_lf_limits(self, config, ra, dec, zs, lmins, lmaxes, lambda_, cosmo):
        """
        Integrate the luminosity between specific
        flux limit. Sets huge normalisation value for non-physical 
        redshifts (i.e. deals with low redshift limits
        for OII and LAE)

        Parameter
        ---------
        config : ConfigParser
            configuration object
        ra, dec, zs : array
            locations to integrate the LF, redshift
            to set the LF parameters
        lmins, lmaxes : float array
            arrays of limits, each row corresponds
            to ra, dec and zs
        lambda_ : float
            wavelength of line (for z -> wl conversion)

        Returns
        -------
        range_integrals : array
            Integral of LF between lmin and lmax
        norms : array
            Integral of LF from flux limit up to
            infinity
        """

        # Luminosity limits 
        wls = (1.0 + array(zs))*lambda_

        # For flux limit need zs assuming LAE
        zlaes = wls/config.getfloat("wavelengths", "LAE") - 1.0

        llims = self.flims(zlaes)*(4.0*pi*square(cosmo.luminosity_distance(zs).to('cm').value)) 
 
        if len(llims) == 1: 
            if lmins < llims:
                raise TooFaintForLimitsException("There are Lmin values less than the limiting L")
        else:
            if nany(lmins < llims):
                _logger.error(len(lmins[lmins < llims]), lmins[lmins < llims], llims[lmins < llims])
                raise TooFaintForLimitsException("There are Lmin values less than the limiting L")
 
        # If no flux errors included just integrate the luminosity function analytically
        if not self.flux_errors:
         
            # No need to do phi* and stuff here
            phistars = self.lf.phi_star_func(zs) 
            lstars = self.lf.Lstars_func(zs)
            alphas = self.lf.alphas_func(zs)

            # Integrate gamma function
            range_integrals = zeros(len(alphas))
            norms = -99.0*ones(len(alphas))

            # Sources out of range
            undetectable = (llims < 0.1) | (zs < 0.0)
            
            # Sources that are too bright to be likely candidates
            # (otherwise Gamma integral fails)
            undetectable = undetectable | (lmins/lstars >= 25.0)
            detectable = (llims >= 0.1) & (zs >= 0.0)

            # Get lmax from lf
            lmaxes_for_norm = self.lf.return_lmax_at_z(zs)

            # LMAX CHOICE NOT TESTED!!!
            norms[detectable] = phistars[detectable]*gamma_integral_limits(alphas[detectable] + 1,
                                                                           (llims/lstars)[detectable], 
                                                                           (1000*lmaxes_for_norm/lstars)[detectable])
            # Too bright to be likely 
            not_too_bright_det =  (lmins/lstars < 25.0) & detectable
               
            range_integrals[not_too_bright_det] = phistars[not_too_bright_det]*gamma_integral_limits(alphas[not_too_bright_det] + 1.0,  
                                                                                     (lmins/lstars)[not_too_bright_det], 
                                                                                     (lmaxes/lstars)[not_too_bright_det])
            # Correct n-density for the EW
            if self.lf.ew_assigner:
                norms[detectable] *= self.lf.ew_assigner.classification_correction(zs[detectable])
                range_integrals[not_too_bright_det] *= self.lf.ew_assigner.classification_correction(zs[not_too_bright_det])

            return range_integrals, norms
        else:
            raise NotImplementedError()           

def luminosity_likelihoods(config, ras, decs, zs, fluxes, lf, lambda_, flim_file, cosmo, delta_l = 0.05):
    """
    Return likelihoods based off of the flux and redshift of an object. Also return
    the expected number density of objects at that redshift (+/- 4AA).

    Parameters
    ----------
    config : ConfigParser
        configuration object
    ras, decs, zs, fluxes : array
        Source properties. zs are true redshifts 
        (i.e. not the inferred LAE ones)
    lf : line_classification.lfs_ews.luminosity_function:LuminosityFunction
        a luminosity function to integrate
    flim_file : str
        path to a file containing the flux limits. Used
        to set luminosity function integral faint limits
    cosmo : astropy.cosmology:FLRW
        an astropy cosmology object to deal with
        cosmology
    delta_l : float (Optional)
        percentage range of luminosity integral (set
        to +/-5% based off Leung+ 2016

    Returns
    -------
    prob_flux : array
        P(flux|LF) the probability that an
        observed flux in the range 1 +- delta_l 
        was drawn from a LF
    n_expected : array
        The expected number of sources in a wavelength 
        slice +/- 4A around its wavelength (value to roughly match
        what Andrew Leung used) (assuming 1 steradian of sky)
    """

    # Grab cosmology from luminosity function
    wl = lambda_*(1.0 + array(zs))
    zlaes = wl/config.getfloat("wavelengths", "LAE") - 1.0

    # Has to be LAE redshift
    Ls = 4.0*pi*square(cosmo.luminosity_distance(zs).to('cm').value)*fluxes

    # Class to integrate the luminosity function  
    lf_inter = LFIntegrator(lf, flim_file)

    # 5% range from Andrew Leung's work
    lupper = (1.0 + delta_l)*Ls
    llower = (1.0 - delta_l)*Ls

    # Don't let range drop below Lmin
    llims = lf_inter.flims(zlaes)*(4.0*pi*square(cosmo.luminosity_distance(zs).to('cm').value))
    out_of_range_is = llower < llims

    # Set lower limit to flux limit and add missing range to upper limit
    if nany(out_of_range_is):
        lrange = lupper[out_of_range_is] - llower[out_of_range_is]
        lupper[out_of_range_is] = llims[out_of_range_is] + lrange
        llower[out_of_range_is] = 1.0001*llims[out_of_range_is]

    lf_ints, ns = lf_inter.integrate_lf_limits(config, ras, decs, zs, llower, 
                                              lupper, lambda_, cosmo)  

    # P(flux|LF)
    prob_flux = lf_ints/ns

    # Undetectable stuff
    prob_flux[ns < -98] = 0.0
    ns[ns < -98] = 0.0

    if nany(prob_flux < 0.0):
        dodges_is = prob_flux < 0.0
        _logger.error("zs, L_lower  L_upper  LF_Integral  Norm")
        _logger.error(zs[dodges_is], llower[dodges_is], lupper[dodges_is], lf_ints[dodges_is], ns[dodges_is])
        raise NegativeProbException("The probability here is negative!")

    # Now derive the expected number of sources in the wavelength slices
    zmins = (wl - 4.0)/lambda_ - 1.0
    zmaxes = (wl + 4.0)/lambda_ - 1.0
    vols = cosmo.comoving_volume(zmaxes).to('Mpc3').value - cosmo.comoving_volume(zmins).to('Mpc3').value

    n_expected = vols*ns

    # for testing
    #for a, b, c, d in zip(ns, vols, zmins, zmaxes)[:10]:
    #    print(a, b, c, d, lambda_)

    return prob_flux, n_expected

def ew_prob(ews, zs, ew_assigner):
    """
    Return P(EW|EW_FUNCTION), the likelihood
    an equivalent width +- 5% 
    was drawn from an particular EW 
    distribution

    Parameters
    ----------
    ews, zs : array
        the equivalent widthd of the
        and redshifts of the objects
    ew_assigner : simcat.equivalent_width:EquivalentWidthAssigner 
        the distibution of equivalent widths

    Returns
    -------
    prob : array
        P(EW|EW_FUNCTION)

    """
    w0s = ew_assigner.w0_func(zs)
    ews_rest = ews/(1.0 + zs)
    prob_ew = exp(-0.95*ews_rest/w0s) - exp(-1.05*ews_rest/w0s)

    # Deal with negative EW as in Leung+ , negative EW only for
    # really noisey continuum values which should be for LAEs
    prob_ew[ews_rest < 0.0] = 1.0
   
    return prob_ew


def prob_additional_line(name, line_fluxes, line_flux_errors, addl_fluxes, addl_fluxes_error, rel_line_strength):
    """
    Return the probability a line flux from a different line in the spectrum is consistent 
    with the line being classified being LyA or OII

    Parameters
    ----------
    name : str
        name of line
    line_fluxes, line_flux_errors : array or float
        flux(es) and error(s) of the LAE or OII candidate,
        Errors not used but here to have similar API
        Farrow+ (in prep) classification method
    addl_fluxes, addl_fluxes_error : array or float
        flux(es) and errors of the named line
    rel_line_strength : float
        relative strength of the other line
        wrt to OII 

    Returns
    -------
    pdata_lae, pdata_oii : array
        P(DATA|LAE), P(DATA|OII) for the
        emission line fluxes
    """ 

    # Leung+ 2016 doesn't account for the noise on the OII candidate line
    stddevs = addl_fluxes_error

    # The Leung+ 2016 5% range again
    max_ = 1.05*addl_fluxes
    min_ = 0.95*addl_fluxes

    # If negative, swap limits
    try:
        len(addl_fluxes)
    except TypeError as e:
        # If float passed, sort it here
        if addl_fluxes < 0:
            min_ = 1.05*addl_fluxes
            max_ = 0.95*addl_fluxes
    else:
        ltzero = addl_fluxes < 0.0
        min_[ltzero] = 1.05*addl_fluxes[ltzero]
        max_[ltzero] = 0.95*addl_fluxes[ltzero]

    # Probability line arose due to chance, i.e. source is LAE
    pdata_lae = norm.cdf(max_, loc=0.0, scale=stddevs) - norm.cdf(min_, loc=0.0, scale=stddevs) 

    # Probability consistent with OII 
    expected_flux = rel_line_strength*line_fluxes
    pdata_oii = norm.cdf(max_, loc=expected_flux,  scale=stddevs) - norm.cdf(min_, loc=expected_flux, scale=stddevs)

    # Leave overall probability unchanged if line out of range or not measured
    out_of_range = (stddevs < 1e-30) | (addl_fluxes < -98)

    # Array or floats passed?
    try:
        len(pdata_lae)
    except TypeError as e:
        if out_of_range:
            pdata_lae = 1.0
            pdata_oii = 1.0
    else:
        pdata_lae[out_of_range] = 1.0 
        pdata_oii[out_of_range] = 1.0

    return pdata_lae, pdata_oii


def source_prob(config, ra, dec, zs, fluxes, flux_errs, ews_obs, ew_err, c_obs, which_color, addl_fluxes, 
                addl_fluxes_error, addl_line_names, flim_file, extended_output=False):
    """
    Return P(LAE) = P(LAE|DATA)/P(DATA) and evidence ratio P(DATA|LAE)P(LAE)/(P(DATA|OII)P(OII)) 
    given input information about the sources

    Parameters
    ----------
    config : ConfigParser
        configuration object
    ra, dec, zs, fluxes, flux_errs, ews_obs, ew_err, c_obs : array
        positions, redshifts (assuming LAE), line fluxes, errors on
        line fluxes, equivalent widths, errors on EW and 
        colours of the sources (latter two not used!)
    which_color : str
        which colour is given 
    addl_fluxes, addl_fluxes_error : 2D array
        each i of addl_fluxes[i, :] should correspond,
        in order, to the fluxes measured for each source
        for the emission line named at position i in
        addl_line_names. To not use set to None (not an
        array of None!)
    addl_line_names : array
        names of emission lines stored in addl_fluxes, in the 
        correct order (see above). To not use set to None (not an array of None!)
    flim_file : str
        file containing the flux limits 
    h : float
        Hubbles constant/100
    extended_output : bool
        Return extended output

    Returns
    -------
    posterior_odds, prob_lae_given_data : float arrays
        posterior_odds = P(DATA|LAE)P(LAE)/(P(DATA|OII)P(OII))
        P(LAE|DATA) = P(DATA|LAE)*P(LAE)/(P(DATA|LAE)*P(LAE) + P(DATA|OII)*P(OII))

    """
    lae_ew = EquivalentWidthAssigner.from_config(config, 'LAE_EW')
    oii_ew = EquivalentWidthAssigner.from_config(config, "OII_EW")
    lf_lae = LuminosityFunction.from_config(config, "LAE_LF", ew_assigner=lae_ew)
    lf_oii = LuminosityFunction.from_config(config, "OII_LF")
    cosmo = generate_cosmology_from_config(config)

    oii_zlim = config.getfloat("General", "oii_zlim")

    _logger.info("Using Hubbles Constant of {:f}".format(cosmo.H0))

    # Cast everything to arrays
    ra = array(ra)
    dec = array(dec)
    zs = array(zs)
    fluxes = array(fluxes)
    ews_obs = array(ews_obs)
    c_obs = array(c_obs)

    zs_oii = ((zs + 1.0)*config.getfloat("wavelengths","LAE"))/config.getfloat("wavelengths", "OII") - 1.0

    # Probability of equivalent widths 
    prob_ew_lae = ew_prob(ews_obs, zs, lae_ew)
    prob_ew_oii = ew_prob(ews_obs, zs_oii, oii_ew)

    # Deal with negative and huge EW as in Leung+ , negative or huge EW only for
    # really noisey continuum values which should be for LAEs
    prob_ew_lae[(ews_obs < 0.0) | (ews_obs > 5000.0)] = 1.0
    prob_ew_oii[(ews_obs < 0.0) | (ews_obs > 5000.0)] = 0.0  

    # Add additional lines to classification probability
    prob_lines_lae = 1.0
    prob_lines_oii = 1.0
    if type(addl_line_names) != type(None):
        for line_name, taddl_fluxes, taddl_fluxes_errors in zip(addl_line_names, addl_fluxes[:], addl_fluxes_error[:]):

            rlstrgth = config.getfloat("RelativeLineStrengths", line_name)
            tprob_lines_lae, tprob_lines_oii = prob_additional_line(line_name, fluxes, flux_errs, taddl_fluxes, taddl_fluxes_errors, rlstrgth)

            if nany(tprob_lines_lae < 0.0) or nany(tprob_lines_oii < 0.0):
                _logger.warning("Negative probability for line {:s}".format(line_name))
                #dodgy_is = tprob_lines_lae < 0.0
                #_logger.error(tprob_lines_lae[dodgy_is], fluxes[dodgy_is], taddl_fluxes[dodgy_is], zs_oii[dodgy_is], line_name)
                #raise NegativeProbException("The probability here is negative!")

            # Not an LAE or an OII?
            neither = (tprob_lines_lae + tprob_lines_oii) < 1e-30
            if nany(neither):
                _logger.warning("Emission line {:s} doesn't look like it's from OII or LAE!")
                #_logger.error(fluxes[neither], flux_errs[neither], taddl_fluxes[neither], taddl_fluxes_errors[neither], zs_oii[neither], line_name)
                #_logger.error(flim_file)
                #raise UnrecognizedSourceException("Neither OII or LAE")

            prob_lines_lae *= tprob_lines_lae
            prob_lines_oii *= tprob_lines_oii

    # Carry out integrals of the luminosity function
    _logger.info('Computing OII posteriors')
    prob_flux_oii, noiis = luminosity_likelihoods(config, ra, dec, zs_oii, fluxes, lf_oii, config.getfloat("wavelengths", "OII"), flim_file, cosmo, delta_l=0.05)
    _logger.info('Computing LAE posteriors')
    prob_flux_lae, nlaes = luminosity_likelihoods(config, ra, dec, zs, fluxes, lf_lae, config.getfloat("wavelengths","LAE"), flim_file, cosmo, delta_l=0.05)

    # Compute the LAE/OII priors
    prior_oii = noiis/(nlaes + noiis)
    prior_lae = nlaes/(nlaes + noiis)

    # P(DATA|LAE), P(DATA|OII)
    prob_data_lae = prob_ew_lae*prob_flux_lae*prob_lines_lae
    prob_data_oii = prob_flux_oii*prob_ew_oii*prob_lines_oii
 
    # This section for test output 
    #table = Table([prob_ew_lae, prob_flux_lae, prob_lines_lae, prob_ew_oii, prob_flux_oii, prob_lines_oii],
    #              names=["prob_ew_lae", "prob_flux_lae", "prob_lines_lae", "prob_ew_oii", 
    #                     "prob_flux_oii", "prob_lines_oii"])
    #table.write("probs_lae.fits")
 
    # Remove anything with too low an OII redshift
    prob_data_oii[zs_oii < oii_zlim] = 0.0
    prior_oii[zs_oii < oii_zlim] = 0.0
    prior_lae[zs_oii < oii_zlim] = 1.0    

    prob_data = prob_data_lae*prior_lae + prob_data_oii*prior_oii 

    #print(prior_lae, prior_oii, prob_data_lae, prob_data_oii)

    #Ignore div0 errors
    # with errstate(divide='ignore'):
    #     posterior_odds = (prob_data_lae*prior_lae)/(prob_data_oii*prior_oii)
    #
    # prob_lae_given_data = prob_data_lae*prior_lae/prob_data

    with errstate(divide='ignore',invalid='ignore'):
        posterior_odds = divide( (prob_data_lae*prior_lae), (prob_data_oii*prior_oii) )
        prob_lae_given_data = divide( prob_data_lae * prior_lae,  prob_data )

    if nany(prob_lae_given_data < 0.0) or nany(isnan(prob_lae_given_data)):
        #dodgy_is = (prob_lae_given_data < 0.0) | isnan(prob_lae_given_data)
        #print(prob_lae_given_data[dodgy_is], prob_data_lae[dodgy_is], prob_data_oii[dodgy_is], 
        #      prior_lae[dodgy_is], prior_oii[dodgy_is], prob_ew_lae[dodgy_is], prob_ew_oii[dodgy_is],
        #      ews_obs[dodgy_is], prob_lines_lae[dodgy_is], prob_lines_oii[dodgy_is], zs_oii[dodgy_is])
        #raise NegativeProbException("""The probability here is negative or NAN! Could be low-z OII (z<0.05) or weird 
        #                               source neither OII or LAE!""")
        _logger.warning("Some sources appear to be neither LAE or OII!")

    
    # Not a chance it's OII
    posterior_odds[(prior_oii < 1e-80) | (prob_data_oii < 1e-80)] = 1e32
    prob_lae_given_data[(prior_oii < 1e-80) | (prob_data_oii < 1e-80)] = 1.0

    if not extended_output:
        return posterior_odds, prob_lae_given_data
    else:
        return posterior_odds, prob_lae_given_data, prob_data_lae, prob_data_oii, prior_lae, prior_oii


