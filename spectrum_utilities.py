"""
Set of simple functions that act on numpy arrays mostly
Keep this simple ... no complex stucture, not a lot of error control
"""

from __future__ import print_function

import global_config as G
import numpy as np
import astropy.units as U
import astropy.cosmology as Cosmo

#SU = Simple Universe (concordance)
SU_H0 = 70.
SU_Omega_m0 = 0.3
SU_T_CMB = 2.73


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
    wave /= (1.+z) #slide to lower z and squeeze bins

    #no ... only do this if these were flux denisities
    #flux *= (1.+z) #shift up flux (shorter wavelength = higher energy)


    #todo: deal with wavelength bin compression (change in counting error?)


    #todo: deal with the various errors (that might not have been passed in)
    # if eflux is None or len(eflux) == 0:
    #     eflux = np.zeros(np.shape(flux))
    #
    # if ewave is None or len(ewave) == 0:
    #     ewave = np.zeros(np.shape(flux))

    #todo: deal with luminosity distance (undo dimming?)



    return flux, wave, eflux, ewave


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
    interp_eflux1 = np.interp(grid, wave1, eflux1)

    interp_flux2 = np.interp(grid, wave2, flux2)
    interp_eflux2 = np.interp(grid, wave2, eflux2)

    #then add
    flux = interp_flux1 + interp_flux2
    eflux = interp_eflux1 + interp_eflux2

    #todo: how to deal with any errors in the wavelength determination (bin)?
    #z is determined from the wavelength, which has error and then the shift is determined from z

    #then crop to the overlap region
    i, _, gt = getnearpos(grid,w_min)

    if gt is None:
        gt = i

    i, lt, _ = getnearpos(grid, w_max)

    if lt is None:
        lt = i

    flux = flux[lt:gt+1]
    eflux = eflux[lt:gt+1]
    wave = grid[lt:gt+1]
    ewave = np.zeros(np.shape(wave))  #for now, no error


    return flux,wave,eflux,ewave