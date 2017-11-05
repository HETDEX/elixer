import Bayes.bayesian
import Bayes.nb
import math
import global_config as G


MAX_PLAE_POII = 99.9

# test call
#addl_fluxes is an array one for each = ['[NeIII]','H_beta','[OIII]','[OIII]']

#turn addl_fluxes into dictionary of wavelength and flux


#print(ratio_LAE, plgd, pogd)

def example_prob_LAE():
    Cosmology = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 'h': 0.70, 'omega_k_0': 0}
    LAE_Priors = 1  # case 1
    alpha_LAE = -1.65  # but can have other values based on LAE_Priors case adopted (-1.36, -1.3, etc?)

    # def main():

    # initialization taken directly from Andrew's code, but I don't know what they mean
    Bayes.nb.init(_alpha_LAE=-1.65, _mult_LStar_LAE=1, _mult_PhiStar_LAE=1, _mult_w0_LAE=1,
                  _alpha_OII=-1.2, _mult_LStar_OII=1, _mult_PhiStar_OII=1, _mult_w0_OII=1)

    ratio_LAE, plgd, pogd = Bayes.bayesian.prob_ratio(wl_obs=3900.0, lineFlux=1.6e-17, ew_obs=-20.0,
                                                      c_obs=None, which_color=None, addl_fluxes=None,
                                                      sky_area=5.5e-07, cosmo=Cosmology, LAE_priors=LAE_Priors,
                                                      EW_case=1, W_0=None, z_OII=None, sigma=None)

    print(ratio_LAE, plgd, pogd)


def fiber_area_in_sqdeg(num_fibers=1):
    #assumes no overlap
    return num_fibers*(math.pi*G.Fiber_Radius**2)/(3600.)**2


#wrapper for Andrew Leung's base code
def prob_LAE(wl_obs,lineFlux,ew_obs,c_obs, which_color=None, addl_fluxes=None,
            sky_area=None, cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None, sigma=None):

    #sanity check
    if (ew_obs == -300) or (ew_obs == 666):
        #bsically, sentinel (null) values from Karl's input file
        return 0.0,0.0,0.0

    #what about different equivalent width calculations for LAE vs OII (different rest wavelength ... or is the ew_obs
    #not rest_frame --- the more likely ... in which case, don't divide by the ratio of observed wavelength/rest)?

    #addl_fluxes should be a dictionary: {wavlength,flux} (need to update downstream code though)

    #sky_area in square degrees (should this be the overlapped sum of all fibers? each at 0.75" radius?
                #makes very little difference in the output

    #LAE call needs LAE_Priors, but OII does not
    #OII calls need the last four  parameters (EW_case, z_OII, W_0, sigma) but the LAE calls do not

    Cosmology_default = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 'h': 0.70, 'omega_k_0': 0}
    LAE_Priors_default = 1  # case 1
    EW_case_default = 1
    alpha_LAE_default = -1.65  # but can have other values based on LAE_Priors case adopted (-1.36, -1.3, etc?)
    sky_area_default = 1.36e-7 # one fiber in square degrees (has very little effect on probabilities anyway and
                               # really is only used in the simulation
    #looks like z_OII is meant to be a list that is used by the simulation, so not needed here
    #the actual z_OII that is used is calculated as you would expect from the wavelength, assuming it is OII

    # def main():
    if cosmo is None:
        cosmo = Cosmology_default

    if lae_priors is None:
        lae_priors = LAE_Priors_default

    if ew_case is None:
        ew_case = EW_case_default

    if sky_area is None:
        sky_area = sky_area_default


    #suppress sign of EW (always wants positive)
    ew_obs = abs(ew_obs)

    # initialization taken directly from Andrew's code, but I don't know what they mean
    Bayes.nb.init(_alpha_LAE=alpha_LAE_default, _mult_LStar_LAE=1, _mult_PhiStar_LAE=1, _mult_w0_LAE=1,
                  _alpha_OII=-1.2, _mult_LStar_OII=1, _mult_PhiStar_OII=1, _mult_w0_OII=1)

    #plgd == Probability of lae given the data?
    #pogd = Probability of OII given the data ?
    ratio_LAE, plgd, pogd = Bayes.bayesian.prob_ratio(wl_obs=wl_obs, lineFlux=lineFlux, ew_obs=ew_obs,
                                                      c_obs=c_obs, which_color=which_color, addl_fluxes=addl_fluxes,
                                                      sky_area=sky_area, cosmo=cosmo, LAE_priors=lae_priors,
                                                      EW_case=ew_case, W_0=W_0, z_OII=z_OII,  sigma=sigma)

    #ratio_LAE is plgd/pogd
    #slightly different representation of ratio_LAE (so recomputed for voltron use)
    if (plgd is not None) and (plgd > 0.0):
        if (pogd is not None) and (pogd > 0.0):
            ratio_LAE = float(plgd) / pogd
        else:
            ratio_LAE = float('inf')
    else:
        ratio_LAE = 0.0

    ratio_LAE = min(MAX_PLAE_POII,ratio_LAE)

    return ratio_LAE, plgd, pogd

# end main
#if __name__ == '__main__':
#    main()