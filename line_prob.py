import Bayes.bayesian
import Bayes.nb


Cosmology = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.70,'omega_k_0':0}
LAE_Priors=1 #case 1

Bayes.nb.init(_alpha_LAE,_mult_LStar_LAE,_mult_PhiStar_LAE,_mult_w0_LAE,
              _alpha_OII,_mult_LStar_OII,_mult_PhiStar_OII,_mult_w0_OII)

Bayes.bayesian.prob_ratio(wl_obs=3900.0, lineFlux=1.6e-17, ew_obs=60.0,
           c_obs=None, which_color=None, addl_fluxes=0,
           sky_area=2.0, cosmo=Cosmology, LAE_priors=LAE_Priors,
           EW_case=1, z_OII=0.1, W_0=1216, sigma=0.1)