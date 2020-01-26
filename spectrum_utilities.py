"""
Set of simple functions that act on numpy arrays mostly
Keep this simple ... no complex stucture, not a lot of error control
"""

from __future__ import print_function

try:
    from elixer import global_config as G
except:
    import global_config as G

import numpy as np
import astropy.constants
import astropy.units as U
import astropy.cosmology as Cosmo
import astropy.stats.biweight as biweight
import weighted_biweight_git as weighted_biweight

#SU = Simple Universe (concordance)
SU_H0 = 70.
SU_Omega_m0 = 0.3
SU_T_CMB = 2.73

log = G.Global_Logger('spectrum_utils')
log.setlevel(G.logging.DEBUG)

def ujy2cgs(ujy,lam): #micro-jansky to erg/s/cm2/AA
    conv = None
    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        conv =  ujy * 1e-29 * c / (lam*lam)
    except:
        log.info("Exception! in ujy2cgs.",exc_info=True)
    return conv

def cgs2ujy(cgs,lam): #erg/s/cm2/AA to micro-jansky
    conv = None
    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        conv = cgs * 1e29 / c * lam*lam
    except:
        log.info("Exception! in cgs2ujy.", exc_info=True)
    return conv

def getnearpos(array,value):
    """
    Nearest, but works best (with less than and greater than) if monotonically increasing. Otherwise,
    lt and gt are (almost) meaningless

    :param array:
    :param value:
    :return: nearest index, nearest index less than the value, nearest index greater than the value
            None if there is no less than or greater than
    """
    idx = (np.abs(array-value)).argmin()

    if array[idx] == value:
        lt = idx
        gt = idx
    elif array[idx] < value:
        lt = idx
        gt = idx + 1
    else:
        lt = idx - 1
        gt = idx

    if lt < 0:
        lt = None

    if gt > len(array) -1:
        gt = None


    return idx, lt, gt


def chi_sqr(obs, exp, error=None, c=None):
    """

    :param obs: (data)
    :param exp: (model)
    :param error: (error on the data)
    :param c: can pass in a fixed c (in most cases, should just be 1.0)
    :return: chi2 and c (best level)
    """

    obs = np.array(obs) #aka data
    exp = np.array(exp) #aka model

    x = len(obs)

    if error is not None:
        error = np.array(error)

    if (error is not None) and (c is None):
        c = np.sum((obs*exp)/(error*error)) / np.sum((exp*exp)/(error*error))
    elif c is None:
        c = 1.0

    chisqr = 0
    if error is None:
        error=np.zeros(np.shape(obs))
        error += 1.0

    chisqr = np.sum(((obs - c*exp)/(error))**2)

    # for i in range(x):
    #         #chisqr = chisqr + ((obs[i]-c*exp[i])**2)/(error[i]**2)
    #         chisqr = chisqr + ((obs[i] - c * exp[i]) ** 2) / (exp[i])

    return chisqr,c

def build_cosmology(H0 = SU_H0, Omega_m0 = SU_Omega_m0, T_CMB = SU_T_CMB):
    cosmo = Cosmo.FlatLambdaCDM(H0=H0,Om0=Omega_m0,Tcmb0=T_CMB)
    return cosmo


def luminosity_distance(z,cosmology=None):
    """
    :param cosmology:
    :param z:
    :return:  lumiosity distance as astropy units
    """

    if cosmology is None:
        cosmology = build_cosmology() #use the defaults

    return cosmology.luminosity_distance(z)

def shift_to_restframe(z, flux, wave, ez=0.0, eflux=None, ewave=None):
    """
    We are assuming no (or insignificant) error in wavelength and z ...

    All three numpy arrays must be of the same length and same indexing
    :param z:
    :param flux: numpy array of fluxes ... units don't matter (but assumed to be a flux NOT flux denisty)
    :param wave: numpy array of wavelenths ... units don't matter
    :param ez:
    :param eflux:  numpy array of flux errors ... again, units don't matter, but same as the flux
                   (note: the is NOT the wavelengths of the flux error)
    :param ewave: numpy array of wavelenth errors ... units don't matter
    :return:
    """

    #todo: how are the wavelength bins defined? edges? center of bin?
    #that is; 4500.0 AA bin ... is that 4490.0 - 4451.0  or 4500.00 - 4502.0 .....

    #rescale the wavelengths
    wave /= (1.+z) #shift to lower z and squeeze bins

    flux *= (1.+z) #shift up flux (shorter wavelength = higher energy)


    #todo: deal with wavelength bin compression (change in counting error?)


    #todo: deal with the various errors (that might not have been passed in)
    # if eflux is None or len(eflux) == 0:
    #     eflux = np.zeros(np.shape(flux))
    #
    # if ewave is None or len(ewave) == 0:
    #     ewave = np.zeros(np.shape(flux))

    #todo: deal with luminosity distance (undo dimming?) and turn into luminosity??
    #
    #lum = flux*4.*np.pi*(luminosity_distance(z))**2
    #
    # or just boost the flux to zero distance point source (don't include the 4*pi)
    #
    #flux = flux * (luminosity_distance(z))**2




    return flux, wave, eflux, ewave


def make_grid(all_waves):
    """
    Takes in all the wavelength arrays to figure the best grid so that
    the interpolation only happens on a single grid

    The grid is on the step size of the spectrum with the smallest steps and the range is
    between the shortest and longest wavelengths that are in all spectra.

    :param all_waves: 2d array of all wavelengths
    :return: grid (1d array) of wavelengths

    """

    all_waves = np.array(all_waves)
    #set the range to the maximum(minimum) to the minimum(maximum)
    mn = np.max(np.amin(all_waves,axis=1)) #maximum of the minimums of each row
    mx = np.min(np.amax(all_waves,axis=1)) #minimum of the maximums of each row

    # the step is the smallest of the step sizes
    # assume (within each wavelength array) the stepsize is uniform
    step = np.min(all_waves[:,1] - all_waves[:,0])

    #return the grid
    return np.arange(mn, mx + step, step)


def interpolate(flux,wave,grid,eflux=None,ewave=None):
    """

    :param flux:
    :param wave:
    :param grid:
    :param eflux:
    :param ewave:
    :return: interpolated flux and interpolated flux error
            note: does not return the wavelengths as that is the grid that was passed in
    """

    #todo: how do we really want to handle interpolating the noise (and how much new noise does
    #interpolation add)?

    interp_flux = np.interp(grid, wave, flux)
    if (eflux is not None) and (len(eflux)==len(wave)):
        interp_eflux = np.interp(grid, wave, eflux)
    else:
        interp_eflux = np.zeros(np.shape(interp_flux))

    #todo: dividing by constant with an error, so adjust the error here too
    interp_ewave = ewave

    return interp_flux, interp_eflux, interp_ewave

def add_spectra(flux1,flux2,wave1,wave2,grid=None,eflux1=None,eflux2=None,ewave1=None,ewave2=None):

    """
    Add two spectra, assumed to be already in the same (e.g. rest) frame
    Does not assume they are on the same grid, but the grid is supplied
    Assumes each spectra's grid is fixed width and is strictly increasing to higher indices

    :param flux1:
    :param flux2:
    :param wave1:
    :param wave2:
    :param grid:
    :param eflux1:
    :param eflux2:
    :param ewave1: error in wavelength measures for 1st spectra
                    (note: NOT the wavelengths of the flux error)
    :param ewave2: error in wavelength measures for 2nd spectra
                    (note: NOT the wavelengths of the flux error)
    :return:
    """

    #todo: deal with errers in flux and in wavelength

    #get the maximum overlap between the two arrays
    #not going to add extrapolated values (i.e. outside this minimum overlap)
    w_min = max(wave1[0],wave2[0])
    w_max = min(wave1[-1],wave2[-1])

    #how many datapoints?

    #if grid is not supplied, find which is smaller and add on its grid
    if (grid is None) or (len(grid) == 0):
        step =  min(wave1[1]-wave1[0],wave2[1]-wave2[0])
        #different min, max ... want the grid to cover the maximums
        #when summing, though, the extrapolated values, outside the original range
        #   will not be added
        g_min = min(wave1[0],wave2[0])
        g_max = max(wave1[-1],wave2[-1])
        grid = np.arange(g_min,g_max+step,step)

    #(linear) interpolate each onto the same grid
    #todo: not sure linear interpolation of error is the best .. maybe should take larger of two nearest?
    # ... what does this do to the noise?
    interp_flux1 = np.interp(grid, wave1, flux1)
    if (eflux1 is not None) and (len(eflux1)==len(wave1)):
        interp_eflux1 = np.interp(grid, wave1, eflux1)
    else:
        interp_eflux1 = np.zeros(np.shape(interp_flux1))

    interp_flux2 = np.interp(grid, wave2, flux2)
    if (eflux2 is not None) and (len(eflux2) == len(wave2)):
        interp_eflux2 = np.interp(grid, wave2, eflux2)
    else:
        interp_eflux2 = np.zeros(np.shape(interp_flux2))

    #then add
    flux = interp_flux1 + interp_flux2
    eflux = interp_eflux1 + interp_eflux2

    #todo: how to deal with any errors in the wavelength determination (bin)?
    #z is determined from the wavelength, which has error and then the shift is determined from z

    #then crop to the overlap region
    i, mn, _ = getnearpos(grid,w_min) #want the nearest position greater than the minimum overlap position

    if mn is None:
        mn = i

    i, _,mx = getnearpos(grid, w_max) #want the nearest position less than the maximum overlap position

    if mx is None:
        mx = i

    flux = flux[mn:mx+1]
    eflux = eflux[mn:mx+1]
    wave = grid[mn:mx+1]
    ewave = np.zeros(np.shape(wave))  #for now, no error


    return flux,wave,eflux,ewave


def red_vs_blue(cwave,wave,flux,flux_err,fwhm=None):
    """
    Stack the spectra on the red side and blue side of the emission line (cwave) and check for consistency vs zero continuum.

    LAE should be very red color by this (little contiuum to blue , more to the red side of LyA)
    but OII should be red also (less to blue, a bit more to red, approaching the Balmer break),
    so this may be a matter of degree OR detection to red but not to blue?)

    :param cwave:
    :param flux: in CGS as flux density (erg/s/cm2/AA)
    :param waves: in AA
    :param flux_err:
    :return: dictionary
    """
    min_bins = 20 #have to have at least min_bins on each size to compute
    min_detect_sigma = 1.0 #what multiple above err must the detection be to be used?

    try:
        #assume waves of same step size (2AA for HETDEX)
        if (cwave is None) or (flux is None) or (wave is None) or (flux_err is None) \
            or (len(flux) != len(wave)) or (len(flux) != len(flux_err)) or (len(flux) < 100):
            log.debug("RvB Invalid parameters")
            return None

        step = wave[1]-wave[0]
        length = len(wave)

        #split the spectrum into a red side (cwave + 20 AA to the end) and a blue side (end to cwave-20AA)
        idx,lt,rt = getnearpos(wave,cwave)

        if fwhm is not None and fwhm > 0.0:
            blue_idx,_,_ = getnearpos(wave,cwave-2*fwhm)
            red_idx,_,_ = getnearpos(wave,cwave+2*fwhm)
        else: #if no fwhm go +/- 20AA from central wave
            blue_idx,_,_ = getnearpos(wave,cwave-20.0)
            red_idx,_,_ = getnearpos(wave,cwave+20.0)

        if (blue_idx <= 0) or (red_idx >= length):
            log.debug("RvB color, unusable index (b: %d), (r: %d)" %(blue_idx,red_idx))
            return None

        if ((blue_idx+1) < min_bins) or ((len(wave)-red_idx) < min_bins):
            log.debug("RvB color, insufficient bins. Indicies: (b: %d), (r: %d)" % (blue_idx, red_idx))
            return None

        #sum up flux
        #sum up flux_err in quadrature

        blue_bins = blue_idx + 1
        blue_width = blue_bins * step
        blue_mid_wave = wave[int(blue_idx/2)]
        blue_side = np.array(flux[0:blue_idx+1])
        blue_flux = np.sum(blue_side)
        blue_err = np.sqrt(np.sum(np.array(flux_err[0:blue_idx+1])**2))

        # #as mean (with error)
        # blue_flux_density = blue_flux / blue_bins
        # blue_flux_density_err = blue_err / blue_bins
        #
        # #or as median
        # blue_flux_density = np.median(blue_side)
        # blue_flux_density_err = np.std(blue_side)/np.sqrt(blue_bins)
        #
        # #or as biweight
        # blue_flux_density = biweight.biweight_location(blue_side)
        # blue_flux_density_err = biweight.biweight_scale(blue_side)/np.sqrt(blue_bins)

        #todo: probably won't matter much, but should we force blue and red to both be weighted or un-weighted?
        #should be a very rare (if ever) case
        #or as weighted biweight
        try:
            blue_flux_density = weighted_biweight.biweight_location_weights(blue_side,weights=flux_err[0:blue_idx+1])
            blue_flux_density_err = biweight.biweight_scale(blue_side)/np.sqrt(blue_bins)
        except:
            log.info("Weighted_biweight failed. Switching to normal biweight")
            blue_flux_density = biweight.biweight_location(blue_side)
            blue_flux_density_err = biweight.biweight_scale(blue_side)/np.sqrt(blue_bins)


        #now to jansky
        blue_flux_density_ujy = cgs2ujy(blue_flux_density,blue_mid_wave)
        blue_flux_density_err_ujy = cgs2ujy(blue_flux_density_err,blue_mid_wave)


        red_bins = (len(wave) - red_idx)
        red_width = red_bins * step
        red_mid_wave = wave[int((len(wave)+red_idx) / 2)]
        red_side = np.array(flux[red_idx:])
        red_flux = np.sum(red_side)
        red_err = np.sqrt(np.sum(np.array(flux_err[red_idx:]) ** 2))

        # #as mean with error
        # red_flux_density = red_flux / red_bins
        # red_flux_density_err = red_err / red_bins
        #
        # #or as median
        # red_flux_density = np.median(red_side)
        # red_flux_density_err = np.std(red_side)/np.sqrt(red_bins)
        #
        # # or as biweight
        # red_flux_density = biweight.biweight_location(red_side)
        # red_flux_density_err = biweight.biweight_scale(red_side) / np.sqrt(red_bins)

        # or as weighted biweight
        try:
            red_flux_density = weighted_biweight.biweight_location_weights(red_side,weights=flux_err[red_idx:])
            red_flux_density_err = biweight.biweight_scale(red_side) / np.sqrt(red_bins)
        except:
            log.info("Weighted_biweight failed. Switching to normal biweight")
            red_flux_density = biweight.biweight_location(red_side)
            red_flux_density_err = biweight.biweight_scale(red_side) / np.sqrt(red_bins)

        # now to jansky
        red_flux_density_ujy = cgs2ujy(red_flux_density, red_mid_wave)
        red_flux_density_err_ujy = cgs2ujy(red_flux_density_err, red_mid_wave)

        rvb = {}
        rvb['blue_flux_density'] = blue_flux_density
        rvb['blue_flux_density_err'] = blue_flux_density_err
        rvb['blue_width'] = blue_width
        rvb['blue_flux_density_ujy'] = blue_flux_density_ujy
        rvb['blue_flux_density_err_ujy'] = blue_flux_density_err_ujy


        rvb['red_flux_density'] = red_flux_density
        rvb['red_flux_density_err'] = red_flux_density_err
        rvb['red_width'] = red_width
        rvb['red_flux_density_ujy'] = red_flux_density_ujy
        rvb['red_flux_density_err_ujy'] = red_flux_density_err_ujy

        # if False: #as CGS ... remember though, these are per AA
        # ...so would need to then have (x_flux_denisty * x_wave / (y_flux_density * y_wave)) * (x_wave/y_wave)
        # or the ratio * (x_wave/y_wave)**2   (remember per lambda is inverse to energy where frequency is not)
        #     if (red_flux_density != 0) and (blue_flux_density != 0):
        #         ratio = blue_flux_density / red_flux_density
        #         ratio_err = abs(ratio) * np.sqrt((blue_flux_density_err/blue_flux_density)**2 +
        #                                     (red_flux_density_err/red_flux_density)**2)
        #     else:
        #         ratio = 1.0
        #         ratio_err = 0.0
        # else: #as uJy


        #can not have nan's (all log arguments here are stictly positive?)
        rvb['blue'] = -2.5*np.log10(blue_flux_density_ujy)
        rvb['blue_err'] = -2.5*np.log10(blue_flux_density_err_ujy)
        rvb['red'] = -2.5*np.log10(red_flux_density_ujy)
        rvb['red_err'] = -2.5*np.log10(red_flux_density_err_ujy)

        #check for detection ... if signal is below the error (min_detect_sigma), replace
        #the signal with the 1-sigma error (max value) and treat as a limit on the color

        log.debug("Spectrum (pseudo) blue = (%f,+/-%f) uJy" % (blue_flux_density_ujy,blue_flux_density_err_ujy) )
        log.debug("Spectrum (pseudo)  red = (%f,+/-%f) uJy" % (red_flux_density_ujy, red_flux_density_err_ujy))

        #blue side detection
        flag = 0 #bit-wise:   01 = blue limit, 10 = red limit, 11 = both (meaningless then?)
        if blue_flux_density_ujy / blue_flux_density_err_ujy < min_detect_sigma:
            blue_flux_density_ujy = blue_flux_density_err_ujy #1-sigma limit
            flag = 1 #blue-side sigma limit

        #red side detection
        if red_flux_density_ujy / red_flux_density_err_ujy < min_detect_sigma:
            red_flux_density_ujy = red_flux_density_err_ujy #1-sigma limit
            flag += 2 #red-side sigma limit


        rvb['flag'] = flag
        if flag == 0:
            rvb['flag_str'] = "good"
        elif flag == 1:
            rvb['flag_str'] = "lower limit"
        elif flag == 2:
            rvb['flag_str'] = "upper limit"
        else: #flag == 3
            rvb['flag_str'] = "non-detect"


        if (red_flux_density_ujy != 0) and (blue_flux_density_ujy != 0):
            ratio = red_flux_density_ujy / blue_flux_density_ujy
            ratio_err = abs(ratio) * np.sqrt((blue_flux_density_err_ujy/blue_flux_density_ujy)**2 +
                                        (red_flux_density_err_ujy/red_flux_density_ujy)**2)
        else:
            ratio = 1.0
            ratio_err = 0.0


        rvb['ratio'] = ratio #blue over red so the color is correct
        rvb['ratio_err'] = ratio_err

        #todo: non detections ... enforce at least 30 datapoints to either side
        #todo: if one side is detected, use the error of the other as a limit
        #i.e. blue = 0 +/- 0.1  red = 10 +/-1  use red/blue as 10/0.1 as 1 sigma limit
        #todo: color 2.5 log (r/b)   (or -2.5 log(b/r)

        rvb['color'] = 2.5 *np.log10(ratio)
        #error is +/- ... so add 1st index (gets more red, i.e the reported color is a lower limit)
        #                    add 2nd index (gets more blue, i.e. the reported color is an upper limit)

        if flag: #any problems
            if flag == 1:
                max_red  = 999
                max_blue = rvb['color']
            elif flag == 2:
                max_red = rvb['color']
                max_blue = -999
            else:
                max_red = 999
                max_blue = -999
        else: #all good
            max_red  = np.nan_to_num(2.5 * np.log10(ratio - ratio_err))
            max_blue = np.nan_to_num(2.5 * np.log10(ratio + ratio_err))


        rvb['color_err'] = (max_red-rvb['color'],max_blue-rvb['color']) #or (redder, bluer) or (larger number, smaller number)
        rvb['color_range'] = (max_blue, max_red)

        log.debug("Spectrum (pseudo) color = %0.03g (%0.3g,%0.3g) [%0.3g,%0.3g] flag: %d" \
                             %(rvb['color'],rvb['color_err'][0],rvb['color_err'][1],
                               rvb['color_range'][0],rvb['color_range'][1],flag))

        # rvb['color_err_b'] = -2.5 *np.log10(ratio + ratio_err) #more on red or less on blue ... color gets more RED
        # rvb['color_err_r'] = -2.5 * np.log10(ratio - ratio_err)#less on red or more on blue ... color gets more BLUE

    except:
        log.info("Exception! in spectrum_utilities red_vs_blue.",exc_info=True)
        return None

    return rvb
