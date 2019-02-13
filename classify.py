#API Wrapper for classification (mostly P(LAE)/P(OII))
import global_config as G
import line_prob

log = G.Global_Logger('classify_logger')
log.setlevel(G.logging.DEBUG)



#    Bayesian probability ratio of P(LAE)/P(OII) based predominantly on equivalent width (assuming a redshift of
#    LyA) and luminosity functions for LAEs and OII from (Leung+ 2017)
def plae_poii():


    ratio, self.p_lae, self.p_oii = line_prob.prob_LAE(wl_obs=self.w,
                                                       lineFlux=self.estflux,
                                                       lineFlux_err=self.estflux_unc,
                                                       ew_obs=self.eqw_obs,
                                                       ew_obs_err=self.eqw_obs_unc,
                                                       c_obs=None, which_color=None,
                                                       addl_fluxes=[], addl_wavelengths=[],
                                                       sky_area=None,
                                                       cosmo=None, lae_priors=None,
                                                       ew_case=None, W_0=None,
                                                       z_OII=None, sigma=None)
    if (self.p_lae is not None) and (self.p_lae > 0.0):
        if (self.p_oii is not None) and (self.p_oii > 0.0):
            self.p_lae_oii_ratio = self.p_lae / self.p_oii
        else:
            self.p_lae_oii_ratio = float('inf')
    else:
        self.p_lae_oii_ratio = 0.0

    self.p_lae_oii_ratio = min(line_prob.MAX_PLAE_POII, self.p_lae_oii_ratio)  # cap to MAX