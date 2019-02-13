from __future__ import (absolute_import, print_function)

import json
from scipy.integrate import quad
from scipy.interpolate import interp1d
from numpy import (seterr, linspace, power, exp, log10, trapz, array, digitize, 
                   cumsum, pi, array, insert, arange, median)
import numpy.random as rand
from astropy.table import Table
from scipy.special import gammainc, gammaincc, gamma
from line_classifier.misc.tools import generate_cosmology_from_config 

class TooLargeForGammaException(Exception):
    pass

def gamma_integral_limits(s, min_, max_):
    """
    Compute the gamma function integral between  
    max_ and min_

    Parameters
    ----------
    s : array
        int_a_b t^{s - 1} e^{-t}    

    max_, min_ : array
        the limits of the integration

    """

    try:
        iter(s < -1.0)
    except TypeError:
        if (max_ <= 0.0) | (min_ <= 0.0):
            raise Exception("Limits must both be positive and greater than 0!")
        if s < -1.0:
            raise Exception("Input s must be > -1.0")
    else:
        if any(array(max_) <= 0.0) | any(array(min_) <= 0.0):
            raise Exception("Limits must both be positive and greater than 0!")
        if any(s < -1.0):
            raise Exception("Input s must be > -1.0")

    try:
        len(s)
    except TypeError: 
        s = [s]
        max_ = [max_]
        min_ = [min_]
 
    s = array(s)
    max_ = array(max_)
    min_ = array(min_)

    if any(min_ > 25.0):
        raise TooLargeForGammaException("This routine only works for min < 25.0")

    # Use integration by parts to prove this
    gplus1 = array(gamma(s+1))*(gammainc(s + 1, max_) - gammainc(s + 1, min_))
    gplus1 += power(max_, s)*exp(-1.0*array(max_))
    gplus1 -= power(min_, s)*exp(-1.0*array(min_))

    return gplus1/s


def schechter_factory(Lstar, alpha, phi_star):
    """ Return a function that evaluates a Schecter function"""

    def schec(L):
        phi_L = phi_star*power(L/Lstar, alpha)*exp(-1.0*L/Lstar)
        return phi_L

    return schec


class LuminosityFunction(object):
    """
    Class to deal with luminosity functions

    TODO: Better way of choosing Lmax, make
    sure always past knee even when Lmin small?

    Parameters
    ----------
    Lstar, alpha, phi_stars : floats
        the luminosity function 
        parameters and different redshifts
    zs : floats
        the redshifts corresponding to the
        LF parameters
    zmin, zmax : float
       the range over which a LF is valid
    lmin_lf : float
       the phistar values are expected to be integrated
       down to this luminosity, this is used to convert
       them into regular phi*
    lf_h : float
       the H/100 of the cosmology appropriate for the
       luminosity function (only passed to store here, not
       used in class)
    ew_assigner : simcat.equivalent_width:EquivalentWidthAssigner
        If passed, increase the predicted n-density values to 
        account for LAEs with EW<20
    interpolate_log : bool (optional)
       interpolate the lumonisity in log10 (default: True)
    """

    def __init__(self, Lstars, alphas, phi_stars, zs, zmin, zmax, lmin_lf,
                 cosmo, flim, lf_h=1.0, det_frac_v_z = None, rseed=None, 
                 ew_assigner=None, interpolate_log=True):

        if len(Lstars) != len(alphas) or len(zs) != len(alphas):
            raise ValueError("Lstar, alpha or zs are not all of the same length")

        self.lf_h = lf_h 
        self.zs = zs
        self.iclfs = []
        self.nzs = len(self.zs)
        self.cosmo = cosmo
        self.flim = flim
        self.zmin = zmin
        self.zmax = zmax
        self.fourpi = 4.0*pi
        self.ew_assigner = ew_assigner

        # Note: Robin Ciardullo thinks a better form of evolution would be one 
        # following the star formation rate density of the Universe
        if len(zs) > 1:
            self.alphas_func = interp1d(zs, alphas, fill_value="extrapolate") 
            if interpolate_log:
                self.interpo_func = interp1d(zs, log10(Lstars), fill_value="extrapolate")
                self.Lstars_func = lambda x : power(10, self.interpo_func(x))
                self.interpo_func_phi = interp1d(zs, log10(phi_stars), fill_value="extrapolate")
                # Divide by this integral as expecting phi_star values integrated up to a minimum L
                self.phi_star_func = lambda x : power(10, self.interpo_func_phi(x))/gamma_integral_limits(self.alphas_func(x) + 1.0,
                                                                                                          (lmin_lf)/self.Lstars_func(x),
                                                                                                          (10**80.0)/self.Lstars_func(x))
            else:
                self.Lstars_func = interp1d(zs, Lstars, fill_value="extrapolate")
                self.phi_star_func = interp1d(zs, phi_stars, fill_value="extrapolate")

        else:
            self.alphas_func = lambda x : alphas[0]        
            self.Lstars_func = lambda x : Lstars[0]        
            self.phi_star_func = lambda x : phi_stars[0]

    @classmethod
    def from_config(cls, config, section, lf_h=1.0, **kwargs): 
        """
        Return a LuminosityFunction object based off 
        of information in a config file

        Parameters
        ----------
        config : ConfigParser
            a ConfigParser object
        section : str
            name of section to extract 
            parameters from
        """

        rlf_h = config.getfloat(section, "rlf_h")
        Lstars = array(json.loads(config.get(section, "Lstars")))
        alphas = array(json.loads(config.get(section, "alphas")))
        phi_stars = array(json.loads(config.get(section, "phi_stars")))
        zs = array(json.loads(config.get(section, "zs")))
        zmin = config.getfloat(section, "zmin")
        zmax = config.getfloat(section, "zmax")
        lmin_lf = config.getfloat(section, "lmin_lf")
        flim = config.getfloat(section, "flim")

        cosmo = generate_cosmology_from_config(config) 
        # Don't want this having units as rlf_h doesn't have any
        lf_h = cosmo.H0.value/100.0

        # Rescale to the target cosmology
        phi_stars *= lf_h/rlf_h * lf_h/rlf_h * lf_h/rlf_h
        Lstars *= rlf_h/lf_h * rlf_h/lf_h
        lmin_lf *= rlf_h/lf_h * rlf_h/lf_h

        return cls(Lstars, alphas, phi_stars, zs, zmin, zmax, lmin_lf, cosmo, 
                   flim, lf_h=lf_h, **kwargs)

      
    def return_lf_at_z(self, zin, normed=False):
        """
        Return a LF at the given redshift. The normed
        option sets the value at Lmin to 1.

        Parameters
        ----------
        zin : float
            the redshift you want a luminosity function 
            for
        normed : bool (optional)
            if True, return function normalised
            such that at LF(Lmin)=1.0
        """
        thisLstar = self.Lstars_func(zin)
        thisAlpha = self.alphas_func(zin)
        Lmin = self.return_lmin_at_z(zin)

        if normed: 
            sf = schechter_factory(thisLstar, thisAlpha, 1.0)
            return schechter_factory(thisLstar, thisAlpha, 1.0/sf(Lmin))
        else:
            this_phi = self.phi_star_func(zin)
            return schechter_factory(thisLstar, thisAlpha, this_phi)

    def return_lmin_at_z(self, z):
        """ Convert flux limit to minimum L """
    
        d = self.cosmo.luminosity_distance(z).to('cm').value 
        Lmin = self.flim*self.fourpi*d*d
   
        return Lmin

    def return_lmax_at_z(self, z):
        """ Assume Lmax of 6000*Lmin - tested n-density to 1%"""

        lmin = self.return_lmin_at_z(z)

        return lmin*6000
 
