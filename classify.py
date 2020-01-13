#API Wrapper for classification (mostly P(LAE)/P(OII))
try:
    from elixer import global_config as G
    from elixer import line_prob
    from elixer import spectrum as elixer_spectrum
except:
    import global_config as G
    import line_prob
    import spectrum as elixer_spectrum

log = G.Global_Logger('classify_logger')
log.setlevel(G.logging.DEBUG)

G.logging.basicConfig(filename="elixer_classify.log", level=G.LOG_LEVEL, filemode='w')

def plae_poii(line_wave,line_flux,line_flux_err, eqw_obs, eqw_obs_err,
              addl_wave = [], addl_flux = [], addl_flux_err = [], estimate_error=False):
    """
    Returns the Bayesian P(LAE)/P(OII) from EW distributions of LAEs and OII galaxies from
    Leung+ 2017

    Maximum P(LAE)/P(OII) is capped at 999. There is no minimum value but 0.0 represents an
    error or inability to calcluate the ratio.

    Uses ELiXer standard options and cosmology

    :param line_wave:  observed wavelength of the emission line in angstroms
    :param line_flux:  observed line flux (cgs)
    :param line_flux_err: observed line flux error (cgs)
    :param eqw_obs: observed equivalent width
    :param eqw_obs_err: observed equivalent width error
    :param addl_wave: array of additional emission line wavelengths in angstroms
    :param addl_flux: array of additional emission line fluxes (cgs)
    :param addl_flux_err: array of additional emission line flux errors (cgs)
    :param estimate_error: optional boolean. If TRUE, an error estimate on the PLAE/POII ratio will be made
                           and an extra return will be populated (plae_details)
    :return: P(LAE)/P(OII), P(LAE|data), P(OII|data) a zero value is an unknown
             plae_details (optional) dictionary if estimate_error is True
                key:"ratio"  value: list of floats [PLAE/POII, Minimum PLAE/POII, Maximum PLAE/POII]
                key: "plgd"  value: list of floats [P(LAE|Data), Minimum P(LAE|Data), Maximum P(LAE|Data)]
                key: "pogd"  value: list of floats [P(OII|Data), Minimum P(OII|Data), Maximum P(OII|Data)]
    """


    if estimate_error:
        ratio, p_lae, p_oii, plae_details = line_prob.prob_LAE(wl_obs=line_wave,
                                                 lineFlux=line_flux,
                                                 lineFlux_err=line_flux_err,
                                                 ew_obs=eqw_obs,
                                                 ew_obs_err=eqw_obs_err,
                                                 addl_wavelengths=addl_wave,
                                                 addl_fluxes=addl_flux,
                                                 addl_errors=addl_flux_err,
                                                 estimate_error=True)
        return ratio, p_lae, p_oii, plae_details
    else:
        ratio, p_lae, p_oii = line_prob.prob_LAE(wl_obs=line_wave,
                                                 lineFlux=line_flux,
                                                 lineFlux_err=line_flux_err,
                                                 ew_obs=eqw_obs,
                                                 ew_obs_err=eqw_obs_err,
                                                 addl_wavelengths=addl_wave,
                                                 addl_fluxes=addl_flux,
                                                 addl_errors=addl_flux_err,
                                                 estimate_error=False)

        return ratio, p_lae, p_oii



#todo: ELiXer additional line finder
def solution_finder(waves, flux, flux_err, line_wave=None):
    """
    Scan input spectrum and return dictonary containing extracted info and lists of lines and possible solutions

    :param waves: wavelength array in angstroms
    :param flux: flux array (in cgs e-17 units). These are the default HETDEX spectra units
                 (flux density x2AA bin)
    :param flux_err: flux_err array (in cgs e-17 units)
    :param line_wave: optional wavelength of indentified main emission line
    :return: spectrum dictionary containing:
                'primary_wave': (float) the observed wavelength of the primary emission line
                'eqw_obs': (float,float) (observed equivalent width, error)
                'plae_poii': (float,(float,float)) (PLAE/POII, (mininum PLAE/POII, maximum PLAE/POII))
                'emission_lines': list of all posssible lines found as dictionaries
                'solutions': list of all solutions as dictionaries

            'emission_lines' dictionary contains:
                'wave': (float) observed wavelength,
                'continuum': (float) observed estimated continuum
                'fwhm': (float) observed fwhm
                'int_flux': (float) observed integrated line flux
                'snr': (float) S/N
                'line_score': (float) ELiXer line score
                'aborption': (bool) True if this is an absorption feature

            'solutions' dictionary contains:
                'wave_rest': (float) rest-frame wavelegth
                'name':(string) symbolic name,
                'z': (float) redshift,
                'score': (float) ELiXer solution score,
                'frac_score': (float) fractional score = this solution score / sum of all solution scores,
                'lines': list of lines included in the solution,
                'rejected_lines': list of found lines removed from the solution (too poor, too close to
                                  another line)

            'lines' and 'rejected_lines' dictionaries contain:
                 'wave_obs': (float) observed wavelength,
                 'wave_rest': (float) rest wavelength,
                 'name': (string) sybolic name,
                 'z': (float) redshift,
                 'line_score': (float) ELiXer line score,
                 'snr': (float) S/N
                 'int_flux':(float) observed integrated line flux
                 'fwhm': (float) observed fwhm
                 'aborption': (bool) True if this is an absorption feature

    """

    spec = elixer_spectrum.Spectrum()
    spec.set_spectra(wavelengths=waves,values=flux,errors=flux_err,central=line_wave,values_units=-17)

    if line_wave is None or line_wave == 0.0:
        line_wave = spec.find_central_wavelength(waves,flux,flux_err,values_units=-17)
        spec.set_spectra(wavelengths=waves, values=flux, errors=flux_err, central=line_wave, values_units=-17)

    solutions = spec.classify()

    spec_dict = {'primary_wave': spec.central,
                 'eqw_obs': (spec.eqw_obs, spec.eqw_obs_unc),
                 'plae_poii': (spec.p_lae_oii_ratio_range[0],
                               (spec.p_lae_oii_ratio_range[1], spec.p_lae_oii_ratio_range[2])),
                 'emission_lines': [],
                 'solutions': []
                 }

    if solutions is not None and len(solutions) > 0:
        for line in spec.all_found_lines:
            line_dict = {'wave': line.mcmc_x0 if line.mcmc_x0 else line.fit_x0,
                         'continuum':line.mcmc_continuum if line.mcmc_continuum else line.fit_continuum,
                         'fwhm':line.fwhm,
                         'int_flux':line.mcmc_line_flux if line.mcmc_line_flux else line.fit_line_flux,
                         'snr':line.mcmc_snr if line.mcmc_snr > -1 else line.snr,
                         'line_score':line.line_score,
                         'aborption': line.absorber}
            spec_dict['emission_lines'].append(line_dict)

        for sol in solutions:
            sol_dict = {'wave_rest':sol.central_rest,
                        'name':sol.name,
                        'z':sol.z,
                        'score': sol.score,
                        'frac_score':sol.frac_score,
                        'lines': [],
                        'rejected_lines': []
                        }
            for line in sol.lines:
                line_dict = {'wave_obs': line.w_obs,
                             'wave_rest': line.w_rest,
                             'name': line.name,
                             'z': line.z,
                             'line_score': line.line_score,
                             'snr':line.snr,
                             'int_flux':line.flux,
                             'fwhm':line.sigma * 2.355,
                             'aborption': line.absorber}
                sol_dict['lines'].append(line_dict)
            for line in sol.rejected_lines:
                line_dict = {'wave_obs': line.w_obs,
                             'wave_rest': line.w_rest,
                             'name': line.name,
                             'z': line.z,
                             'line_score': line.line_score,
                             'snr':line.snr,
                             'int_flux':line.flux,
                             'fwhm':line.sigma * 2.355,
                             'aborption': line.absorber}
                sol_dict['rejected_lines'].append(line_dict)

            spec_dict['solutions'].append(sol_dict)

    return spec_dict