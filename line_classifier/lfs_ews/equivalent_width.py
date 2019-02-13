"""

Module to simulate LAE and OII
equivalent widths based off of Leung+ 2017
(and some of his code as well). Interpolated 
EW approach "InterpolatedEW" developed and
presented in Farrow+ (in prep).


Daniel Farrow 2018 (MPE)

"""
from __future__ import absolute_import

import json
from numpy import log, exp, sqrt, square, copy, log10, isfinite, digitize, array, zeros, trapz
from numpy.random import uniform, seed
from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator 
from scipy.special import erfc
from astropy.io import fits

class NonFiniteInterpolationCube(Exception):
    pass 


class InterpolatedEW(object):
     """
     Return the EW function by interpolating
     an input table

     Parameters
     ----------
     filename : str
         a file containing the z, EW 
         and n values
     normalize : bool (Optional)
         optionally normalize the input 
         distributions. Off by default as
         need to account for EWs outside
         of the range of the cube in norm
         (default: False)

     """

     def __init__(self, filename, normalize=False):

         hdus = fits.open(filename)
         self.zbcens = hdus["REDSHIFT"].data
         ew = hdus["EW_BCENS"].data
         data = hdus["Primary"].data

         # Optionally normalize the distributions
         if normalize:
             for i in range(len(self.zbcens)):
                 norm = trapz(data[i, :], x=ew)
                 print(norm) 
                 if norm > 0.0:                    
                     data[i, :] /= norm

         self.minz = min(self.zbcens)
         self.maxz = max(self.zbcens)
         self.minew = min(ew)
         self.maxew = max(ew)

         if not all(isfinite(data).flatten()):
            raise NonFiniteInterpolationCube("Non finite values in the EW interpolation cube: {:s}".format(filename))

         self.interpolator = RegularGridInterpolator((self.zbcens, log10(ew)), data, method="linear", bounds_error=False)

     def return_new(self, z, ew):
         """
         Return the interpolated values 
         at z and EW. If outside of interpolation
         range set to last value in interpolation
         range (not sure why this isn't default
         for nearest method).

         """
         znew = copy(z)
         ewnew = copy(ew)

         znew[z < self.minz] = self.minz
         znew[z > self.maxz] = self.maxz

         ewnew[ew < self.minew] = self.minew
         ewnew[ew > self.maxew] = self.maxew

         interp_vals = self.interpolator((znew, log10(ewnew)))

         return interp_vals

class EquivalentWidthAssigner(object):
    """
    Generate rest frame equivalent widths from
    an analytical perscription. Does
    linear interpolation with redshift
    with the different w0 values

    Parameters
    ----------
    w0s : float array
        the w0 of exponential
        form fits to the EW distribution
    zs : float array
        the redshifts corresponding to
        the w0 values passed
    seed : integer (optional)
        an integer seed for the RNG

    Attributes
    ----------
    w0_func : callable
        interpolation function over
        w0 and z

    """
    def __init__(self, zs, w0s, seed_=None):

        self.w0s = w0s
        self.zs = zs
        if len(self.zs) > 1:
            self.w0_func = interp1d(self.zs, self.w0s, fill_value="extrapolate")
        else:
            self.w0_func = lambda x : self.w0s[0]

        seed(seed=seed_)

    @classmethod
    def from_config(cls, config, section, **kwargs):
        """
        Generate a EquivalentWidthAssigner object
        from a config file

        Parameters
        ----------
        config : ConfigParser
            a ConfigParser object
        section : str
            name of section to extract 
            parameters from
        """

        zs = json.loads(config.get(section, "zs"))
        w0s = json.loads(config.get(section, "w0s"))

        return cls(zs, w0s, **kwargs)


    def classification_correction(self, zin, ew_cut=20.0):
        """
        Increase number density in mocks 
        to account for the fact the measured
        LFs we use assume that EWs<20A are OII
        emitters (see Leung+ 2016)

        Parameters
        ----------
        zin : float or array
            redshifts to return boost factor
            at
        ew_cut : float (optional)
            the EW cut used for the LF
   
        Returns
        -------
        boost : float or array
            the correction factor 
            to multiply n with i.e. the
            fraction of LAEs with EWs
            less than the cut         
        """

        ws = self.w0_func(zin)
        boost = 1.0/(exp(-1.0*ew_cut/ws))

        return boost

    def ew_function(self, z):
        """
        Return the functional
        fit to the EW distribution
        at redshift z

        Parameters
        ----------
        z : float
            redshift of EW function
      
        Returns
        -------
        ew_func : callable
            the EW function

        """

        w0 = self.w0_func(z)

        def ew_func(w):
            return exp(-1.0*w/w0)/w0

        return ew_func

    def ew_func_convolved(self, z):
        """
        Return a function describing
        the EW distribution convolved
        with flux error

        Parameters
        ----------
        z : float
            redshift of EW function
      
        Returns
        -------
        ew_func_conv : callable
            the EW function. Two arguments
            EW and EW_ERROR

        """

        w0 = self.w0_func(z)

        def ew_func_conv(w, werr):
            """
            The convolution of an error function
            and a Gaussian, see e.g.

            http://www.np.ph.bham.ac.uk/research_resources/programs/halflife/gauss_exp_conv.pdf

            (accessed 09/11/2018)

            or

            Eli Grushka 1972 

            https://pubs.acs.org/doi/abs/10.1021/ac60319a011


            Parameters
            ----------
            w : array 
               the measured EW
            werr : array
               the standard deviation of the 
               distribution (i.e. the error on w)
            """
            norm = 1.0/(2.0*w0)
            werr2 = square(werr)
            conv = norm*exp((werr2 - 2.0*w*w0)/(2.0*w0*w0))
            conv = conv*erfc((werr2/w0 - w)/(sqrt(2.0)*werr))

            return conv

        return ew_func_conv


    def gen_ews(self, zin):
        """
        Generate rest-frame EWs for input
        redshifts

        Parameters
        ----------
        zin : array
            the redshifts of the 
            emission lines

        Returns
        -------
        ews : array
            noise free equivalent
            widths in units of 1/w0
        """

        # Random values to map to EWs (I think 1 results in ew=inf)
        rans = uniform(0.0, 0.999999, size=len(zin))
 
        # w values for the redshifts
        ws = self.w0_func(zin)
        
        # inverse of CDF for each source
        ews = -1.0*ws*log(1.0 - rans)

        return ews



