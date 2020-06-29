"""
Set of simple functions that act on numpy arrays mostly
Keep this simple ... no complex stucture, not a lot of error control
"""

from __future__ import print_function

try:
    from elixer import global_config as G
    from elixer import weighted_biweight as weighted_biweight
except:
    import global_config as G
    import weighted_biweight as weighted_biweight

import numpy as np
import pickle
import os.path as op
import astropy.constants
import astropy.units as U
from astropy.coordinates import SkyCoord
import astropy.cosmology as Cosmo
import astropy.stats.biweight as biweight

from scipy.optimize import curve_fit

from hetdex_tools.get_spec import get_spectra as hda_get_spectra
from hetdex_api import survey as hda_survey
#from hetdex_api.shot import get_fibers_table as hda_get_fibers_table

import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter,MaxNLocator

#SU = Simple Universe (concordance)
SU_H0 = 70.
SU_Omega_m0 = 0.3
SU_T_CMB = 2.73

log = G.Global_Logger('spectrum_utils')
log.setlevel(G.LOG_LEVEL)


filter_iso_dict = {'u': 3650.0,
                   'b': 4450.0,
                   'g': 4640.0,
                   'v': 5510.0,
                   'r': 6580.0,
                   'i': 8060.0,
                   'z': 9000.0,
                   'y': 10200.0,
                   'f606w': 6000.0,
                   'acs_f606w_flux':6000.0, #name sometimes set this way from catalog
                   }

def filter_iso(filtername, lam):
    """
    Approximate iso wavelength lookup by filter. If not found, just returns the wavelength passed in.
    :param filtername:
    :param lam:
    :return:
    """
    global filter_iso_dict
    try:
        if filtername.lower() in filter_iso_dict.keys():
            return filter_iso_dict[filtername.lower()]
        else:
            log.info(f"Unable to match filter {filtername} to iso wavelength")
            return lam
    except:
        log.debug("Exception is spectrum_utilities::filter_iso",exc_info=True)
        return lam


def mag2cgs(mag,lam):
    """
    :param mag:   AB mag
    :param lam: central wavelength or (better) f_lam_iso
    :return:
    """

    try:
        c = (astropy.constants.c * (1e10 * U.AA / U.m)).value
        return 3631. * 1e-23 * 10**(-0.4 * mag) * c / (lam * lam)
    except:
        return 0

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


def ujy2mag(ujy): #micro-jansky to erg/s/cm2/AA
    conv = None
    try:
        conv = -2.5 * np.log10(ujy / 1e6 / 3631.)
    except:
        log.info("Exception! in ujy2mag.",exc_info=True)
    return conv


def cgs2mag(cgs,lam):
    """
    :param cgs:   erg/s/cm2/AA
    :return:
    """
    try:
       return ujy2mag(cgs2ujy(cgs,lam))
    except:
        return 0


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


def physical_diameter(z,a,cosmology=None):
    """
    :param cosmology:
    :param z:
    :param a: size (diameter) in arcsec
    :return:  lumiosity distance as astropy units
    """

    if cosmology is None:
        cosmology = build_cosmology() #use the defaults

    apk = cosmology.arcsec_per_kpc_proper(z)

    if apk > 0:
        return  (a / apk).value #strip units?
    else:
        return None

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
            blue_flux_density = weighted_biweight.biweight_location_errors(blue_side,errors=flux_err[0:blue_idx+1])
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
            red_flux_density = weighted_biweight.biweight_location_errors(red_side,errors=flux_err[red_idx:])
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


        rvb['ratio'] = ratio
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


def velocity_offset(wave1,wave2):
    """
    Return the velocity offset from wave1 to wave2
    (negative values = blue shift = wave2 is blue of wave1)
    :param wave1: wavelength 1 (unitless or units must match)
    :param wave2: wavelength 2 (unitless or units must match)
    :return: velocity offset as quantity
    """

    try:
        return astropy.constants.c.to(U.km/U.s) * (1 - wave1/wave2)
    except:
        return None

def wavelength_offset(wave,velocity):
    """
    Return change in wavelength (observered) for a given base wavelength and velocity offset
    (reminder, negative velocity is toward or blue shifted and the return will be negative)
    :param wave: wavelength 1 (unitless or units must match)
    :param velocity: in km/s
    :return: wavelength (same units as wave)
    """
    try:
        if isinstance(velocity, U.Quantity):
            return wave - wave * (1.0 - velocity/astropy.constants.c.to(U.km/U.s))
        else:
            return wave - wave * (1.0 - velocity / astropy.constants.c.to(U.km / U.s).value)
    except:
        return None



def extract_at_position(ra,dec,aperture,shotid,ffsky=False):
    """

    :param ra:
    :param dec:
    :param aperture:
    :param shotid:
    :param ffsky:
    :return:
    """

    return_dict = {}
    return_dict['flux'] = None
    return_dict['fluxerr'] = None
    return_dict['wave'] = None
    return_dict['ra'] = ra
    return_dict['dec'] = dec
    return_dict['shot'] = shotid
    return_dict['aperture'] = aperture
    return_dict['ffsky'] = ffsky

    try:
        coord = SkyCoord(ra=ra * U.deg, dec=dec * U.deg)
        apt = hda_get_spectra(coord, survey=f"hdr{G.HDR_Version}", shotid=shotid,ffsky=ffsky,
                          multiprocess=False, rad=aperture,tpmin=0.0)

        if len(apt) == 0:
            #print(f"No spectra for ra ({self.ra}) dec ({self.dec})")
            log.info(f"No spectra for ra ({ra}) dec ({dec})")
            return return_dict

        # returned from get_spectra as flux density (per AA), so multiply by wavebin width to match the HDF5 reads
        return_dict['flux'] = np.nan_to_num(apt['spec'][0], nan=0.000) * G.FLUX_WAVEBIN_WIDTH   #in 1e-17 units (like HDF5 read)
        return_dict['fluxerr'] = np.nan_to_num(apt['spec_err'][0], nan=0.000) * G.FLUX_WAVEBIN_WIDTH
        return_dict['wave'] = np.array(apt['wavelength'][0])
        return_dict['ra'] = ra
        return_dict['dec'] = dec
    except Exception as E:
        print(f"Exception in Elixer::spectrum_utilities::extract_at_position",E)
        log.info(f"Exception in Elixer::spectrum_utilities::extract_at_position",exc_info=True)

    return return_dict

def rms(data, fit,cw_pix=None,hw_pix=None,norm=True):
    """

    :param data: (raw) data
    :param fit:  fitted data (on the same scale)
    :param cw_pix: (nearest) pixel (index) of the central peak (could be +/- 1 pix (bin)
    :param hw_pix: half-width (in pixels from the cw_pix) overwhich to calculate rmse (i.e. cw_pix +/- hw_pix)
    :param norm: T/F whether or not to divide by the peak of the raw data
    :return:
    """
    #sanity check
    min_pix = 5  # want at least 5 pix (bins) to left and right

    if (data is None):
        log.warning("Invalid data (None) supplied for rms.")
        return -999
    elif (fit is None):
        log.warning("Invalid data (fit=None) supplied for rms.")
        return -999
    elif (len(data) != len(fit)):
        log.warning("Invalid data supplied for rms, length of fit <> data.")
        return -999
    elif any(np.isnan(data)):
        log.warning("Invalid data supplied for rms, NaNs in data.")
        return -999
    elif any(np.isnan(fit)):
        log.warning("Invalid data supplied for rms, NaNs in fit.")
        return -999
    elif not (min_pix < cw_pix < (len(data) - min_pix)):
        # could be highly skewed (esp think of large asym in LyA, with big velocity disp. (booming AGN)
        log.warning("Invalid data supplied for rms. Minimum distance from array edge not met.")
        return -999

    if norm:
        mx = max(data)
        if mx < 0:
            log.warning("Invalid data supplied for rms. max data < 0")
            return -999
    else:
        mx = 1.0

    d = np.array(data)/mx
    f = np.array(fit)/mx

    if ((cw_pix is not None) and (hw_pix is not None)):
        left = max(cw_pix - hw_pix,0)
        right = min(cw_pix + hw_pix,len(data))

        #due to rounding of pixels (bins) from the caller (the central index +/- 2 and the half-width to either side +/- 2)
        # either left or right can be off by a max total of 4 pix
        # rounding_error = 4
        # if -1*rounding_error <= left < 0:
        #     left = 0
        #
        # if len(data) < right <= (len(data) +rounding_error):
        #     right = len(data)


        if (left < 0) or (right > len(data)):
            log.warning("Invalid range supplied for rms. Data len = %d. Central Idx = %d , Half-width= %d"
                      % (len(data),cw_pix,hw_pix))
            return -999

        d = d[left:right+1]
        f = f[left:right+1]

    return np.sqrt(((f - d) ** 2).mean())

def gaussian(x, x0, sigma, a=1.0, y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None

    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y



def simple_fit_slope (wavelengths, values, errors=None,trim=True):
    """
    Just a least squares fit, no MCMC
    Trim off the blue and red-most regions that are a bit dodgy

    :param wavelengths: unitless floats (but generally in AA)
    :param values: unitless floats (but generally in erg/s/cm2 over 2AA e-17) (HETDEX standard)
    :param errors: unitless floats (same as values)
    :param trim: Trim off the blue and red-most regions that are a bit dodgy
    :return: slope, slope_error
    """

    slope = None
    slope_error = None
    if (wavelengths is None) or (values is None) or (len(wavelengths)==0) or (len(values)==0) \
            or (len(wavelengths) != len(values)):
        log.warning("Zero length (or None) spectrum passed to simple_fit_slope().")
        return slope, slope_error

    try:
        if trim  and (len(wavelengths) == 1036): #assumes HETDEX standard rectified 2AA wide bins 3470-5540
            idx_lt = 65  #3600AA
            idx_rt = 966 #5400AA (technically 965+1 so it is included in the slice)
        else:
            idx_lt = 0
            idx_rt = -1

        #check the slope
        if errors is not None and len(errors)==len(values):
            weights = 1./np.array(errors[idx_lt:idx_rt])
        else:
            weights = None

        coeff, cov  = np.polyfit(wavelengths[idx_lt:idx_rt], values[idx_lt:idx_rt],
                                 w=weights,cov=True,deg=1)
        if coeff is not None:
            slope = coeff[0]

        if cov is not None:
            slope_error = np.sqrt(np.diag(cov))[0]

    except:
        log.debug("Exception in simple_fit_slope() ", exc_info=True)

    return slope, slope_error


def simple_fit_wave(values,errors,wavelengths,central,wave_slop_kms=1000.0):
    """
    Simple curve_fit to gaussian; lsq "best"

    :param values:
    :param errors:
    :param wavelengths:
    :param central:
    :param wave_slop_kms: +/- from central wavelength in km/s (default of 1000 km/s ~ +/-13AA at 4500AA)
    :return:
    """

    return_dict = {}
    return_dict['x0'] = None
    return_dict['fitflux'] = 0.0
    return_dict['continuum_level'] = 0.0
    return_dict['velocity_offset'] = 0.0
    return_dict['sigma'] = 0.0
    return_dict['rmse'] = 0.0
    return_dict['snr'] = 0.0
    return_dict['meanflux_density'] = 0.0
    return_dict['velocity_offset_limit'] = wave_slop_kms  # not memory efficient to store the same value
    # repeatedly, but, thinking ahead to maybe allowing the max velocity limit to vary by coordinate
    #return_dict['score'] = 0.0

    if (values is None) or (len(values) == 0):
        return return_dict



    try:
        min_sigma = 1.5 #FWHM  ~ 3.5AA (w/o error, best measure would be about 5AA)
        wave_slop = wavelength_offset(central,wave_slop_kms) #for HETDEX, in AA
        wave_side = int(round(max(40,2*wave_slop) / G.FLUX_WAVEBIN_WIDTH)) #at least 40AA to either side or twice the slop
        idx,_,_ = getnearpos(wavelengths,central)
        min_idx = max(0,idx-wave_side)
        max_idx = min(len(values),idx+wave_side)

        narrow_wave_x = wavelengths[min_idx:max_idx+1]
        narrow_wave_counts = values[min_idx:max_idx+1]

        return_dict['meanflux_density'] = np.nansum(values[min_idx:max_idx+1])/\
                                         (wavelengths[max_idx+1]-wavelengths[min_idx])
        #reminder, max_idx+1 since we want to include the WIDTH of the last bin in the sum
        #also, assumes input is in flux units and the wavebins are the same width

        if (errors is not None) and (len(errors) == len(wavelengths)):
            narrow_wave_errors = errors[min_idx:max_idx+1]
            #replace any 0 with 1
            narrow_wave_errors[np.where(narrow_wave_errors == 0)] = 1
            narrow_wave_err_sigma = 1. / (narrow_wave_errors * narrow_wave_errors)
        else:
            narrow_wave_errors = None
            narrow_wave_err_sigma = None

        try:

            parm, pcov = curve_fit(gaussian, np.float64(narrow_wave_x), np.float64(narrow_wave_counts),
                                   p0=(central, 1.5, 1.0, 0.0),
                                   bounds=((central - wave_slop, min_sigma, 0.0, -100.0),
                                           (central + wave_slop, np.inf, np.inf, np.inf)),
                                   # sigma=1./(narrow_wave_errors*narrow_wave_errors)
                                   sigma=narrow_wave_err_sigma  # , #handles the 1./(err*err)
                                   # note: if sigma == None, then curve_fit uses array of all 1.0
                                   # method='trf'
                                   )

            try:
                if not np.any(pcov):  # all zeros ... something wrong
                    log.info("Something very wrong with curve_fit")
            except Exception as E:
                print(E)
                return return_dict

            perr = np.sqrt(np.diag(pcov))

            x0 = parm[0]
            sigma = parm[1]
            A = parm[2]
            Y = parm[3]
            fitflux = A / G.FLUX_WAVEBIN_WIDTH
            continuum_level = Y
            vel_offset = velocity_offset(central, x0).value

            #sn_pix to either side of the peak
            num_sn_pix = int(round(min(20, len(narrow_wave_counts)// 2 - 1)))  # don't go larger than the actual array

            rms_wave = gaussian(narrow_wave_x, parm[0], parm[1], parm[2], parm[3])
            if (2 * sigma * 2.355) > (len(narrow_wave_counts)):
                #could be very skewed and broad, so don't center on the emission peak, but center on the array
                fit_rmse = rms(narrow_wave_counts, rms_wave, cw_pix=len(narrow_wave_counts) // 2, hw_pix=num_sn_pix,
                           norm=False)
            else:
                fit_rmse = rms(narrow_wave_counts, rms_wave, cw_pix=getnearpos(narrow_wave_x, x0 )[0], hw_pix=num_sn_pix,
                                 norm=False)

            num_sn_pix = num_sn_pix * 2 + 1 #update to full counting of sn_pix (each side + 1 for the peak bin)
            snr = A / (np.sqrt(num_sn_pix) * fit_rmse)

            return_dict['x0'] = x0
            return_dict['fitflux'] = fitflux
            return_dict['continuum_level'] = continuum_level
            return_dict['velocity_offset'] = vel_offset
            return_dict['sigma']=sigma
            return_dict['rmse'] = fit_rmse
            return_dict['snr'] = snr
            #return_dict['score'] = line_score


            #assume fit line is LyA and get the sum around rest 880-910
            try:
                lyc_obs = (880.0 * (1.+x0/G.LyA_rest), 910.0 * (1.+x0/G.LyA_rest))
                lyc_idx = blue,_,_ = getnearpos(wavelengths,lyc_obs[0])
                lyc_idx = red, _, _ = getnearpos(wavelengths, lyc_obs[1])
                return_dict['f900'] = cgs2ujy(np.sum(values[blue,red+1])/(lyc_obs[1]-lyc_obs[0]),900.0)
                return_dict['f900e'] = cgs2ujy(np.sum(errors[blue,red+1])/(lyc_obs[1]-lyc_obs[0]),900.0)
            except:
                return_dict['f900'] = -999
                return_dict['f900e'] = -999

        except Exception as E:
            print(E)

    except Exception as E:
        print(E)

    return return_dict


def make_raster_grid(ra,dec,width,resolution):
    """
    Make the RA and Dec meshgrid for rastering. Centered on provided ra, dec.
    :param ra: decimal degrees
    :param dec:  decimal degrees
    :param width: arcsec
    :param resolution:arcsec
    :return: ra_meshgrid  (RAs for each Dec) and dec_meshgrid (Decs for each RA)
    """
    try:
        width /= 3600.
        resolution /= 3600.

        # todo note: generally expected to be small enough at low enough DEC that curvature is ignored?
        #  or should I adjust the RA accordingly?

        ra_grid = np.concatenate((np.flip(np.arange(ra-resolution,ra -width-resolution,-1*resolution)),
                                  np.arange(ra, ra + width + resolution, resolution)))

        dec_grid = np.concatenate((np.flip(np.arange(dec-resolution,dec-width-resolution, -1*resolution)),
                                   np.arange(dec, dec + width + resolution, resolution)))
        ra_meshgrid, dec_meshgrid = np.meshgrid(ra_grid, dec_grid)
        return ra_meshgrid, dec_meshgrid
    except:
        log.error("Exception in spectrum_utilities::make_raster_grid",exc_info=True)
        return None, None


def raster_search(ra_meshgrid,dec_meshgrid,shotlist,cw,aperture=3.0,max_velocity=1500.0):
    """
    Iterate over the supplied meshgrids and force extract at each point
    :param ra_meshgrid: (see make_raster_grid)
    :param dec_meshgrid:
    :param shotlist: (must be a list, so if just one shot, pass as [shot]
    :param cw: central wavelength (search/extract around this wavelength
    :param aperture: in arcsec
    :param max_velocity: in km/s (max allowance to either side of the specified <cw> to fit)
    :return: meshgrid of (extraction) dictionaries with location and extraction info
    """

    try:
        edict = np.transpose(np.zeros_like(ra_meshgrid, dtype=dict)) #need to transpose since addressing as dec,ra
        ct = 0
        wct = 0

        #for s in [shotlist[1]]:
        # bw_events = 0
        # max_bw_ct = 0
        # min_bw_ct = 9999
        # wbw_events = 0

        msg = f"Raster extracting emission. RA x Dec x Shots " \
                 f"({np.shape(ra_meshgrid)[1]}x{np.shape(ra_meshgrid)[0]}x{len(shotlist)}) = " \
                 f"{np.shape(ra_meshgrid)[1] * np.shape(ra_meshgrid)[0] * len(shotlist)} extractions."
        log.info(msg)
        print(msg)

        for r in range(np.shape(ra_meshgrid)[1]): #columns (x or RA values)
            for d in range(np.shape(ra_meshgrid)[0]): #rows (y or Dec values)
                exlist = []
                for s in shotlist:
                    #print("Shot = ", s)
                    ct += 1
                    #print(ct)
                    ex = extract_at_position(ra_meshgrid[d,r], dec_meshgrid[d,r], aperture, s)

                    if ex['flux'] is None:
                        continue

                    exlist.append(ex)

                if len(exlist) == 0:
                    continue
                elif len(exlist) == 1:
                    avg_f = exlist[0]['flux']
                    avg_fe = exlist[0]['fluxerr']

                elif len(exlist) > 1:
                    #make a new ex dictionary with the weighted biweight average of the data and run simple fit
                    flux = []
                    fluxe = []
                    #waves are all the same
                    for ex in exlist:
                        flux.append(ex['flux'])
                        fluxe.append(ex['fluxerr'])

                    flux = np.array(flux)
                    fluxe = np.array(fluxe)

                    #now weighted biweight
                    avg_f  = np.zeros(len(flux[0]))
                    avg_fe = np.zeros(len(flux[0]))

                    # if len(flux[0]) > max_bw_ct:
                    #     max_bw_ct = len(flux[0])
                    #
                    # if len(flux[0]) < min_bw_ct:
                    #     min_bw_ct = len(flux[0])

                    for i in range(len(flux[0])):

                        wslice_err = np.array([m[i] for m in fluxe])  # all rows, ith index
                        wslice = np.array([m[i] for m in flux])  # all rows, ith index

                        wslice_err = wslice_err[np.where(wslice != -999)]  # so same as wslice, since these need to line up
                        wslice = wslice[np.where(wslice != -999)]  # git rid of the out of range interp values

                        # git rid of any nans
                        wslice_err = wslice_err[~np.isnan(wslice)]  # err first so the wslice is not modified
                        wslice = wslice[~np.isnan(wslice)]

                        try:
                            f = weighted_biweight.biweight_location_errors(wslice, errors=wslice_err)
                            #not really using the errors, but this is good enough?
                            fe = weighted_biweight.biweight_scale(wslice) / np.sqrt(len(wslice))
                            #wbw_events += 1
                        except:
                            #log.info("Weighted_biweight failed. Switching to normal biweight")
                            f = weighted_biweight.biweight_location(wslice)
                            fe = weighted_biweight.biweight_scale(wslice) /np.sqrt(len(wslice))
                            #bw_events += 1

                        if np.isnan(f):
                            avg_f[i] = 0
                            avg_fe[i] = 0
                        elif np.isnan(fe):
                            avg_f[i] = f
                            avg_fe[i] = 0
                        else:
                            avg_f[i] = f
                            avg_fe[i] = fe
                else:
                    #something very wrong
                    #print("********** unexpectedly zero spectra???????")
                    continue
                #end if

                #now, fit to Gaussian
                ex['fit'] = simple_fit_wave(avg_f,
                                               avg_fe,
                                               exlist[0]['wave'],
                                               cw,
                                               wave_slop_kms=max_velocity)

                # kill off bad fits based on snr, rmse, sigma, continuum
                # overly? generous sigma ... maybe make similar to original?
                if (1.0 < ex['fit']['sigma'] < 20.0) and \
                        (3.0 < ex['fit']['snr'] < 1000.0):
                    if ex['fit']['fitflux'] > 0:
                        #print("Winner")
                        wct += 1
                else:  # is bad, wipe out
                    #print("bad fit")
                    ex['bad_fit'] = copy.copy(ex['fit'])
                    ex['fit']['snr'] = 0
                    ex['fit']['fitflux'] = 0
                    ex['fit']['continuum_level'] = 0
                    ex['fit']['velocity_offset'] = 0
                    ex['fit']['rmse'] = 0
                    ex['fit']['x0'] = 0
                    ex['fit']['sigma'] = 0

                edict[r, d] = ex

        log.info(f"Raster 'good' emission fits ({wct}) / ({ct})")
        #print("Good fits:", wct)
        #print(f"Biweights: max_ct ({max_bw_ct}), min_ct ({min_bw_ct}), wbw ({wbw_events}), bw ({bw_events})")

        return edict
    except:
        log.error("Exception in spectrum_utilities::raster_search", exc_info=True)
        return None



def make_raster_plots(dict_meshgrid,ra_meshgrid,dec_meshgrid,cw,key,colormap=cm.coolwarm,save=None,savepy=None,show=False):
    """
    Make 3 plots (an interactive (if show is true) 3D plot, a contour plot and a color meshplot (like a heatmap)
    :param dict_meshgrid: meshgrid of dictionaries for the RA, Dec
    :param ra_meshgrid:
    :param dec_meshgrid:
    :param cw: central wavelength
    :param key: one of "fitflux" (integrated line flux), "f900", "velocity_offset", "continuum_level"
    :param colormap: matplotlib color map to use for scale
    ##:param lyc: if True, assume the cw is LyA and produce an extra fitflux plot around rest-900AA
    :param save: filename base to save the plots (the files as saved as <save>_3d.png, <save>_contour.png, <save>_mesh.png
    :param savepy: save a pickle and a python file to load it and launche interactive plots
    :param show: if True, display the plots interactively
    :return: returns a mesh grid of the values selected by <key>
    """

    #3D mesh apperently does not support masked arrays correctly, so cannot mask bad values
    # colomap.set_bad(color=[0.2, 1.0, 0.23]) #green

    try:
        #set RA, DEC as central position of the meshgrids
        r,c = np.shape(ra_meshgrid)
        RA = ra_meshgrid[0,(c-1)//2]

        r, c = np.shape(dec_meshgrid)
        DEC = dec_meshgrid[(r-1)//2,0]

        plt.close('all')
        old_backend = None
        if show:
            old_backend = plt.get_backend()
            plt.switch_backend('TkAgg')

        ########################
        #as a contour
        ########################

        #z for integrated flux
        #colomap.set_bad(color=[0.2, 1.0, 0.23]) #green
        colormap.set_bad(color=[1.0, 1.0, 1.0]) #white
        bad_value = -1e9
        z = np.full(ra_meshgrid.shape,bad_value,dtype=float)
        for r in range(np.shape(ra_meshgrid)[1]):  # columns (x or RA values)
            for d in range(np.shape(ra_meshgrid)[0]):  # rows (y or Dec values)
                try:
                    if key != 'meanflux_density' and 'bad_fit' in dict_meshgrid[d, r].keys():
                        pass # no fit, assume same as "no-data"
                    else:
                        z[d,r] = dict_meshgrid[d, r]['fit'][key]
                except:
                    pass  # nothing there, so leave the 'bad value' in place

        z = np.ma.masked_where(z == bad_value, z)

        try:
            max_aa_offset = dict_meshgrid[0][0]['fit']['velocity_offset_limit'] #in km/s
            max_aa_offset = cw * max_aa_offset / (astropy.constants.c.to('km/s')).value
        except:
           max_aa_offset = None
           # max_aa_offset = cw * 1500.00 / (astropy.constants.c.to('km/s')).value

        fig = plt.figure()
        ax = plt.gca()
        num_levels = np.max(ra_meshgrid.shape)
        levels = MaxNLocator(nbins=num_levels).tick_values(z.min(), z.max())
        surf = ax.contourf((dec_meshgrid-DEC)*3600.0,(ra_meshgrid-RA)*3600.0, z, cmap=colormap, levels=levels)

        #get the x-range, then reverse so East is to the left
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[1],xlim[0])

        if max_aa_offset is not None:
            info_title = fr"$\alpha$ ({RA:0.5f}) $\delta$ ({DEC:0.5f}) $\lambda$ ({cw:0.2f} +/- {max_aa_offset:0.2f})"
        else:
            info_title = fr"$\alpha$ ({RA:0.5f}) $\delta$ ({DEC:0.5f}) $\lambda$ ({cw:0.2f})"

        ax.set_title(info_title)
        ax.set_xlabel(r'$\Delta$RA"')
        ax.set_ylabel(r'$\Delta$Dec"')
        fig.colorbar(surf, shrink=0.5, aspect=5,label=key)

        if save:
            plt.savefig(save +  "_" + key + "_contour.png")

        if show:
            fig.show()


        ######################
        # as a color mesh
        ######################
        fig = plt.figure()
        ax = plt.gca()
        #notice here, ra & dec are in the other order vs 3D plot
        surf = ax.pcolormesh((dec_meshgrid-DEC)*3600.0, (ra_meshgrid-RA)*3600.0, z, cmap=colormap)
        #get the x-range, then reverse so East is to the left
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[1],xlim[0])

        ax.set_title(info_title)
        ax.set_xlabel(r'$\Delta$RA"')
        ax.set_ylabel(r'$\Delta$Dec"')
        fig.colorbar(surf, shrink=0.5, aspect=5,label=key)

        if save:
            plt.savefig(save +  "_" + key + "_mesh.png")

        if show:
            fig.show()

        ##########################
        # as interactive 3D
        # do this one last so can interact and see the other two
        ##########################
        bad_value = 0
        z = np.full(ra_meshgrid.shape, bad_value, dtype=float)
        for r in range(np.shape(ra_meshgrid)[1]):  # columns (x or RA values)
            for d in range(np.shape(ra_meshgrid)[0]):  # rows (y or Dec values)
                try:
                    z[d, r] = dict_meshgrid[d, r]['fit'][key]
                except:
                    z[d, r] = 0  # 3D plot does not handle masked arrays #nothing there, so leave the 'bad value' in place

        # 3D Plot (interactive)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # because this plots as X,Y,Z, and we're using a meshgrid, the x-coords (the RA's are in the dec_meshgrid,
        # that is, all the RAs  (X's) for each given Dec)
        # in otherwords, though, the RAs are values of X coords, but first parameter are the ROWS (row-major)
        # which are along the vertical (each row contains the list of all RAs (x's) for that row
        surf = ax.plot_surface((dec_meshgrid - DEC) * 3600.0, (ra_meshgrid - RA) * 3600.0, z, cmap=colormap,
                               linewidth=0, antialiased=False)

        # get the x-range, then reverse so East is to the left
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[1], xlim[0])

        ax.set_title(info_title)
        ax.set_xlabel(r'$\Delta$RA"')
        ax.set_ylabel(r'$\Delta$Dec"')
        ax.set_zlabel(key)
        fig.colorbar(surf, shrink=0.5, aspect=5, label=key)

        if save is not None:  # interactive does not work
            # pickle.dump(plt.gcf(),open(save+".fig.pickle","wb"))
            plt.savefig(save + "_" + key + "_3d.png")

        if show:
            plt.show()

        #restore the original backend
        if old_backend:
            plt.switch_backend(old_backend)


        if savepy is not None:
            try:
                #save a pickle of the parameters and a python file to run?
                parms_dict = {}

                #make these small?? (drop the details?)
                # cp_dict = copy.copy(dict_meshgrid)
                # for c in np.ndarray.flatten(cp_dict):
                #     try:
                #         c['flux'] =[]
                #         c['fluxerr'] = []
                #         c['wave'] = []
                #     except: #entry might be empty (0) and not hold a dictionary
                #         pass
                # parms_dict['dict_meshgrid'] = cp_dict

                #or dump the whole data (keeping the spectra at each point?) could be usefull?
                parms_dict['dict_meshgrid'] = dict_meshgrid

                parms_dict['ra_meshgrid'] = ra_meshgrid
                parms_dict['dec_meshgrid'] = dec_meshgrid
                parms_dict['cw'] = cw
                parms_dict['colormap'] = colormap

                fn = savepy + "_grid"
                #subsequent calls to other plots will overwrite the pickle, but it is the same data
                pickle.dump(parms_dict,open(fn+".pickle","wb"))

                pyfilestr = "from elixer import spectrum_utilities as SU\n"
                pyfilestr += "import pickle\n"
                pyfilestr += "import sys\n"
                pyfilestr += "import numpy as np\n"
                pyfilestr += "args = list(map(str.lower, sys.argv))\n"
                pyfilestr += f"parms_dict = pickle.load(open('{op.basename(fn)+'.pickle'}','rb'))\n"
                pyfilestr += "data = np.ndarray.flatten(parms_dict['dict_meshgrid'])\n"
                pyfilestr += "idx = np.where(data != 0)[0]\n"
                pyfilestr += "if (idx is None) or len(idx)==0:\n"
                pyfilestr += "  print('No emission line was able to be fit with provided parameters.')\n"
                pyfilestr += "  exit(0)\n"
                pyfilestr += "try:\n"
                pyfilestr += "    allkeys = data[idx[0]]['fit'].keys()\n"
                pyfilestr += "except:\n"
                pyfilestr += "    allkeys = data[idx[0]]['bad_fit'].keys()\n"
                pyfilestr += "if (len(args) > 1):\n"
                pyfilestr += "    if args[1] in allkeys:\n"
                pyfilestr += "        key = args[1] \n"
                pyfilestr += "    else:\n"
                pyfilestr += "        print(f'Available keys to plot: {allkeys}')\n"
                pyfilestr += "        exit(0)\n"
                pyfilestr += "else:\n"
                pyfilestr += "    print('Assuming fitflux plot')\n"
                pyfilestr += "    print(f'Available keys to plot: {allkeys}')\n"
                pyfilestr += "    key = 'fitflux'\n"
                pyfilestr += "SU.make_raster_plots(parms_dict['dict_meshgrid']," \
                             "parms_dict['ra_meshgrid']," \
                             "parms_dict['dec_meshgrid']," \
                             "parms_dict['cw']," \
                             "key," \
                             "parms_dict['colormap']," \
                             "show=True,save=None,savepy=None)\n"


                with open(fn+".py","w") as f:
                    f.write(pyfilestr)
            except Exception as e:
                print("Unable to save interactive py plots",e)

        return z
    except:
        log.error("Exception in spectrum_utilities::make_raster_plots",exc_info=True)
        return None

def get_shotids(ra,dec,hdr_version=G.HDR_Version):
    """

    :param ra: decimal degrees
    :param dec: decimal degrees
    :param hdr_version: 1,2, etc. default is the current ELiXer data release default
    :return: list of integer shotids that overlap the provided ra and dec
    """

    try:
        survey = hda_survey.Survey(survey="hdr%d" % hdr_version)
        if not survey:
            log.info(f"Cannot build hetdex_api survey object to determine shotid")
            return []

        # this is only looking at the pointing, not checking individual fibers, so
        # need to give it a big radius to search that covers the focal plane
        # if centered (and it should be) no ifu edge is more than 12 acrmin away
        shotlist = survey.get_shotlist(SkyCoord(ra, dec, unit='deg', frame='icrs'),
                                           radius=G.FOV_RADIUS_DEGREE * U.deg)

        log.debug(f"{len(shotlist)} shots near ({ra},{dec})")

        return shotlist
    except:
        log.error("Exception in spectrum_utilities::get_shotids", exc_info=True)
        return []

####################################
# example of using raster search
###################################
# from elixer import spectrum_utilities as SU
# RA = 214.629608
# DEC  =  52.544327
# cw = 3819.0
# shotlist = SU.get_shotids(RA,DEC)
# ra_meshgrid, dec_meshgrid = SU.make_raster_grid(RA,DEC,3.0,0.4)
# edict = SU.raster_search(ra_meshgrid,dec_meshgrid,shotlist,cw,max_velocity=2000,aperture=3.0)
# z = SU.make_raster_plots(edict,ra_meshgrid,dec_meshgrid,cw,"fitflux",show=True)
# notice:  in the above call, to save static images, specify a filename base in the "save" parameter
#          or for interactive images, in the "savepy" parameter