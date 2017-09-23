'''
   Last updated: 9 Sep 2015
   
   Andrew Leung
   Rutgers University
   
   Bayesian classification for LAEs and [O II] emitters
   see http://arxiv.org/abs/1510.07043
   
'''

from pylab import *
import numpy as np
import math
import bisect
import random
import time
import collections
from cosmolopy import distance as cd
import mpmath as mpm
import matplotlib.pyplot as plt
import laegen as lg
import oiigen as og
import nb
import scipy.stats as stats
import scipy.special as ssp
import fit_lognorm as fln

reload(lg)
reload(og)



def init(_alpha_LAE,_mult_LStar_LAE,_mult_phiStar_LAE,_mult_w0_LAE,_alpha_OII,_mult_LStar_OII,_mult_phiStar_OII,_mult_w0_OII):
   
   global alpha_LAE,mult_LStar_LAE,mult_phiStar_LAE,mult_w0_LAE,alpha_OII,mult_LStar_OII,mult_phiStar_OII,mult_w0_OII
   alpha_LAE,mult_LStar_LAE,mult_phiStar_LAE,mult_w0_LAE,alpha_OII,mult_LStar_OII,mult_phiStar_OII,mult_w0_OII = _alpha_LAE,_mult_LStar_LAE,_mult_phiStar_LAE,_mult_w0_LAE,_alpha_OII,_mult_LStar_OII,_mult_phiStar_OII,_mult_w0_OII



def comoving_volume(z,cosmo):         ## function returns comoving volume out to specified redshift in Mpc^3
   Vc = cd.comoving_volume(z, **cosmo)
   return Vc



def prob_ratio(wl_obs,lineFlux,ew_obs,c_obs,which_color,addl_fluxes,sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma):
   
   a, phiStar_LAE, LStar_LAE, z_LAE, L_min_LAE = prob_data_given_LAE(wl_obs,lineFlux,ew_obs,c_obs,which_color,
                                                                     addl_fluxes,sky_area,cosmo,LAE_priors)
   b, phiStar_OII, LStar_OII, z_OII, L_min_OII = prob_data_given_OII(wl_obs,lineFlux,ew_obs,c_obs,which_color,
                                                                     addl_fluxes,sky_area,cosmo,EW_case,z_OII,W_0,sigma)

   c = prob_LAE(wl_obs, phiStar_LAE, LStar_LAE, z_LAE, L_min_LAE, phiStar_OII, LStar_OII, z_OII, L_min_OII, cosmo)

   if b <= 0: ratioLAE = 1e32
   else: ratioLAE = a * c / b / (1-c)

   if a <= 0: ratioOII = 1e32
   else: ratioOII = b * (1-c) / a / c

   '''
      bayesian.prob_ratio(), a function of wavelength observed for the emission line, flux from the line, and equivalent
       width observed for the line, returns seven quantities
      
         1. the ratio of the probability that the observed object is an LAE and the probabitility that it is an 
         [O II] emitter
      
         2. the ratio of the probability that the observed object is an [O II] emitter and the probabitility that it is
          an LAE
         
         3. P(data|LAE) is given by the function prob_data_given_LAE(), a function of wavelength observed for the
          emission line, flux from the line, and equivalent width observed for the line. The wavelength observed allows
           us to select the corresponding luminosity function and EW distribution, which allows us to find the 
           probability that an LAE observed at that wavelength (binned into intervals of 200 angstroms) has line flux 
           that is within 5 percent of that observed and EW that is within 5 percent of that observed. Luminosity 
           functions and EW distributions for LAEs are interpolated and extrapolated from the Schechter and exponential
            function parameters for z~2.1 and z~3.1 LAEs (Ciardullo et al. 2012)
         
         4. P(data|OII) is given by the function prob_data_given_OII(), exactly analogous to prob_data_given_LAE(). 
         Luminosity functions and EW distributions for [O II] emitters are interpolated and extrapolated from 
         Schechter and exponential function parameters given in four redshift bins 0<z<0.56 (Ciardullo et al. 2013)
         
         5. P(LAE) = 1-P(OII); the 'prior' probability that is an object is an LAE given only wavelength observed for
          the emission line (binned into intervals of 200 angstroms), computed using the number density of LAEs and
           [O II] emitters given by their luminosity functions (Ci12, Ci13), and the cosmological volume of the
            respective redshift bins using cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.70}
         
         6. P(data|LAE)*P(LAE) = P(LAE|data)*P(data)
         
         7. P(data|OII)*P(OII) = P(OII|data)*P(data)
         
   '''

   #return ratioLAE, ratioOII, a, b, c, (a*c), (b*(1-c))
   return ratioLAE, (a*c), (b*(1-c))



def prob_data_given_LAE(wl_obs,lineFlux,ew_obs,c_obs,which_color,addl_fluxes,sky_area,cosmo,LAE_priors):
   t0 = time.time()
   
   z_LAE = wl_obs/1215.668-1
   L = 4*np.pi*( 3.08567758e24 * nb.lumDist(z_LAE,cosmo) )**2 * lineFlux
   norm_to_1, phiStar_LAE, LStar, z_LAE, L_min = LAE_LF(wl_obs,sky_area,LAE_priors,cosmo)
   ### third argument '1' denotes Ci12, evolving LF between z = 2.1 and z = 3.1
   
   LStar = LStar * mult_LStar_LAE
   phiStar_LAE = phiStar_LAE * mult_phiStar_LAE
   
   prob_lineFlux = norm_to_1 * phiStar_LAE * ( float(mpm.gammainc(alpha_LAE+1,0.95*L/LStar)) - float(mpm.gammainc(alpha_LAE+1,1.05*L/LStar)) )

   ### same thing; chain rule + recusive definition of gamma function (15-08-17)
   #prob_lineFlux = norm_to_1 * phiStar_LAE / (alpha_LAE+1) * ( float(mpm.gammainc(alpha_LAE+2,0.95*L/LStar)-(0.95*L/LStar)**(alpha_LAE+1)*np.exp(-0.95*L/LStar)) - float(mpm.gammainc(alpha_LAE+2,1.05*L/LStar)-(1.05*L/LStar)**(alpha_LAE+1)*np.exp(-1.05*L/LStar)) )
   
   '''
      probability that line flux is within +/- 5 percent of observation
      
      alpha_LAE = -1.65 (Ciardullo+ 12)
      by definition, the parameter t in gamma(t) is alpha+1
      need to use recursion relation for the incomplete gamma function to integral under the schechter function
      
   '''

   if not which_color == 'no_imaging':
      
      w_0 = LAE_EW(wl_obs,LAE_priors)
      ### last argument '1' denotes Ci12, evolving LF between z = 2.1 and z = 3.1

      w_0 = w_0 * mult_w0_LAE
      ew_rest = ew_obs/(1+z_LAE)

      if ew_obs <= 0:         ### (10/14/14) allow negative EW (only LAEs affected because [O II]s are bright in the continuum)
         prob_EW = 1.
      
      else:
         prob_EW = -exp(-1.05*ew_rest/w_0) - ( -exp(-0.95*ew_rest/w_0) )
      
   ###### (04-24-15)
   addl_em_lines  = ['[NeIII]','H_beta','[OIII]','[OIII]']
   addl_lambda_rf = np.array([3869.00, 4861.32, 4958.91, 5006.84])
   
   inf_OII_redshift = wl_obs/3727.45-1
   
   addl_lambda_ob = addl_lambda_rf * (1+inf_OII_redshift)
   prob_NeIII3869, prob_Hb4861, prob_OII4959, prob_OII5007 = 1.,1.,1.,1.
   
   f_obs = addl_fluxes
   
   for i in range(len(addl_em_lines)):
      if 3500. <= addl_lambda_ob[i] <= 5500.:
         mean = 0.
         stdev = 0.2*nb.lineSens(addl_lambda_ob[i])*np.sqrt(sky_area/300.)
         if i == 0:
            prob_NeIII3869 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )
         elif i == 1:
            prob_Hb4861 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )
         elif i == 2:
            prob_OII4959 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )
         elif i == 3:
            prob_OII5007 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )

   prob_addl_lines = prob_NeIII3869 * prob_Hb4861 * prob_OII4959 * prob_OII5007
   
   if not (which_color == 'g-r') and not (which_color == 'g-i') and not (which_color == 'g-z') and not (which_color == 'r-i') and not (which_color == 'r-z') and not (which_color == 'i-z') and not (which_color == 'no_imaging'):
      return (prob_lineFlux * prob_EW * prob_addl_lines), phiStar_LAE, LStar, z_LAE, L_min

   elif which_color == 'no_imaging':
      return (prob_lineFlux * prob_addl_lines), phiStar_LAE, LStar, z_LAE, L_min
   
   else:
      if which_color == 'g-r':
         c_mean = lg.g_r_mean ### -0.15 (AL 02-27-14) undo fudge factor to offset LAE distribution from OII now that r'-z' colors were used for power-law slope   ### (12-01-14) see EG email: "In other words, your simulation implies that observed LAE g-r colors are 0.15 mags bluer than the real ones (due mostly to Lya affecting g-band at most redshifts).  Which means we need to shift the observed HPS distribuiton bluer as a "best guess" at reality."
         c_stdev = lg.g_r_stdev
      elif which_color == 'g-i':
         c_mean = lg.g_i_mean
         c_stdev = lg.g_i_stdev
      elif which_color == 'g-z':
         c_mean = lg.g_z_mean
         c_stdev = lg.g_z_stdev
      elif which_color == 'r-i':
         c_mean = lg.r_i_mean
         c_stdev = lg.r_i_stdev
      elif which_color == 'r-z':
         c_mean = lg.r_z_mean
         c_stdev = lg.r_z_stdev
      elif which_color == 'i-z':
         c_mean = lg.i_z_mean
         c_stdev = lg.i_z_stdev

      prob_color = stats.norm.cdf(c_obs+0.05,c_mean,c_stdev) - stats.norm.cdf(c_obs-0.05,c_mean,c_stdev)

      return (prob_lineFlux * prob_EW * prob_color * prob_addl_lines), phiStar_LAE, LStar, z_LAE, L_min



def prob_data_given_OII(wl_obs,lineFlux,ew_obs,c_obs,which_color,addl_fluxes,sky_area,cosmo,EW_case,z_OII_list,W_0_list,sigma_list):
   t0 = time.time()
   
   z_OII = wl_obs/3727.45-1
   L = 4*np.pi*( 3.08567758e24 * nb.lumDist(z_OII,cosmo) )**2 * lineFlux
   norm_to_1, phiStar_OII, LStar, z_OII, L_min = OII_LF(wl_obs,sky_area,cosmo)
   
   LStar = LStar * mult_LStar_OII
   phiStar_OII = phiStar_OII * mult_phiStar_OII
   
   
   if wl_obs < 3727.45: prob_lineFlux = 0         ## wavelength observed shorter than [O II] rest frame
   else:
      prob_lineFlux = norm_to_1 * phiStar_OII * ( float(mpm.gammainc(alpha_OII+1,0.95*L/LStar)) - float(mpm.gammainc(alpha_OII+1,1.05*L/LStar)) )
      ### same thing; chain rule + recusive definition of gamma function (15-08-17)
      #prob_lineFlux = norm_to_1 * phiStar_OII / (alpha_OII+1) * ( float(mpm.gammainc(alpha_OII+2,0.95*L/LStar)-(0.95*L/LStar)**(alpha_OII+1)*np.exp(-0.95*L/LStar)) - float(mpm.gammainc(alpha_OII+2,1.05*L/LStar)-(1.05*L/LStar)**(alpha_OII+1)*np.exp(-1.05*L/LStar)) )
      
   '''
      probability that line flux is within +/- 5 percent of observation
      
      alpha_OII = -1.2 (Ciardullo+ 13)
      by definition, the parameter t in gamma(t) is alpha+1
      need to use recursion relation for the incomplete gamma function to integral under the schechter function
      
   '''

   if not which_color == 'no_imaging':

      ew_rest = ew_obs/(1+z_OII)
      
      if ew_obs <= 0:         ### (10/14/14) allow negative EW (only LAEs affected because [O II]s are bright in the continuum)
         prob_EW = 0.
      
      else:
         if (EW_case == 'lognormal' and ew_rest <= 10) or EW_case == 'fully_lognormal':
            ### (09/03/15) re-fitting for lognormal parameters makes for a good slow-down!
            ###            0.53 second per object to run prob_ratio() this way
            ###            0.0096 second per object by pre-fitting lognormal in nb.prob_ratio()
            ###            0.0036 second per object on grad box
            #fitParam, fitCovar = fln.run(z_OII,'base')
            #W_0, sigma = fitParam[0], fitParam[1]
            
            W_0   = np.interp(z_OII,z_OII_list,W_0_list)
            sigma = np.interp(z_OII,z_OII_list,sigma_list)
            prob_EW = 0.5 * ( ssp.erf((np.log((1.05*ew_rest)/W_0)-0.5*sigma**2)/(np.sqrt(2)*sigma)) - ssp.erf((np.log((0.95*ew_rest)/W_0)-0.5*sigma**2)/(np.sqrt(2)*sigma)) )
      
         else:
            w_0 = OII_EW(wl_obs)
            w_0 = w_0 * mult_w0_OII
            prob_EW = -np.exp(-1.05*ew_rest/w_0) - ( -np.exp(-0.95*ew_rest/w_0) )
   
   ###### (04-24-15)
   addl_em_lines  = ['[NeIII]','H_beta','[OIII]','[OIII]']
   addl_lambda_rf = np.array([3869.00, 4861.32, 4958.91, 5006.84])
   rel_strength   = np.array([0.416, 1., 1.617, 4.752])/1.791      ## Anders_Fritze_2003.dat, metallicity one-fifth solar
   #rel_strength   = np.array([0.300, 1., 1.399, 4.081])/3.010      ## (05-06-15) Anders_Fritze_2003.dat, metallicity 0.5-2 in solar units

   inf_OII_redshift = wl_obs/3727.45-1
   addl_lambda_ob = []
   for i in range(len(addl_em_lines)):
      this_redshifted_line = addl_lambda_rf[i] * (1+inf_OII_redshift)
      addl_lambda_ob.append(this_redshifted_line)

   addl_lambda_ob = np.array(addl_lambda_ob)
   prob_NeIII3869, prob_Hb4861, prob_OII4959, prob_OII5007 = 1.,1.,1.,1.
   
   f_obs = addl_fluxes
   
   for i in range(len(addl_em_lines)):
      if 3500. <= addl_lambda_ob[i] <= 5500.:
         mean = rel_strength[i]*lineFlux
         stdev = 0.2*nb.lineSens(addl_lambda_ob[i])*np.sqrt(sky_area/300.)
         if i == 0:
            prob_NeIII3869 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )
         elif i == 1:
            prob_Hb4861 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )
         elif i == 2:
            prob_OII4959 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )
         elif i == 3:
            prob_OII5007 = np.abs( stats.norm.cdf(f_obs[i]*1.05,mean,stdev) - stats.norm.cdf(f_obs[i]*0.95,mean,stdev) )

   prob_addl_lines = prob_NeIII3869 * prob_Hb4861 * prob_OII4959 * prob_OII5007


   if not (which_color == 'g-r') and not (which_color == 'g-i') and not (which_color == 'g-z') and not (which_color == 'r-i') and not (which_color == 'r-z') and not (which_color == 'i-z') and not (which_color == 'no_imaging'):
      return (prob_lineFlux * prob_EW * prob_addl_lines), phiStar_OII, LStar, z_OII, L_min
   
   elif which_color == 'no_imaging':
      return (prob_lineFlux * prob_addl_lines), phiStar_OII, LStar, z_OII, L_min
   
   else:
      if which_color == 'g-r':
         c_mean = og.g_r_mean
         c_stdev = og.g_r_stdev
      elif which_color == 'g-i':
         c_mean = og.g_i_mean
         c_stdev = og.g_i_stdev
      elif which_color == 'g-z':
         c_mean = og.g_z_mean
         c_stdev = og.g_z_stdev
      elif which_color == 'r-i':
         c_mean = og.r_i_mean
         c_stdev = og.r_i_stdev
      elif which_color == 'r-z':
         c_mean = og.r_z_mean
         c_stdev = og.r_z_stdev
      elif which_color == 'i-z':
         c_mean = og.i_z_mean
         c_stdev = og.i_z_stdev

      prob_color = stats.norm.cdf(c_obs+0.05,c_mean,c_stdev) - stats.norm.cdf(c_obs-0.05,c_mean,c_stdev)

      return (prob_lineFlux * prob_EW * prob_color * prob_addl_lines), phiStar_OII, LStar, z_OII, L_min



def prob_LAE(wl_obs, phiStar_LAE, LStar_LAE, z_LAE, L_min_LAE, phiStar_OII, LStar_OII, z_OII, L_min_OII, cosmo):
   t0 = time.time()
   
   if wl_obs < 3727.45:
      a = 1.
      b = 0.
   
   else:
      if bisect.bisect(nb.LyaWL,wl_obs) > 249:
         LAE_redshift_bin_upper_limit = nb.LyaWL[249]/1215.668-1
         LAE_redshift_bin_lower_limit = nb.LyaWL[248]/1215.668-1
         OII_redshift_bin_upper_limit = nb.LyaWL[249]/3727.45-1
         OII_redshift_bin_lower_limit = nb.LyaWL[248]/3727.45-1

      else:
         LAE_redshift_bin_upper_limit = nb.LyaWL[bisect.bisect(nb.LyaWL,wl_obs)]/1215.668-1
         LAE_redshift_bin_lower_limit = nb.LyaWL[bisect.bisect(nb.LyaWL,wl_obs)-1]/1215.668-1
         OII_redshift_bin_upper_limit = nb.LyaWL[bisect.bisect(nb.LyaWL,wl_obs)]/3727.45-1
         OII_redshift_bin_lower_limit = nb.LyaWL[bisect.bisect(nb.LyaWL,wl_obs)-1]/3727.45-1


      volume_if_LAE = comoving_volume(LAE_redshift_bin_upper_limit,cosmo)-comoving_volume(LAE_redshift_bin_lower_limit,cosmo)
      volume_if_OII = comoving_volume(OII_redshift_bin_upper_limit,cosmo)-comoving_volume(OII_redshift_bin_lower_limit,cosmo)


      alpha_LAE = -1.65
      alpha_OII = -1.2
      
      LOverLStar_LAE = L_min_LAE/LStar_LAE
      LOverLStar_OII = L_min_OII/LStar_OII

      a = phiStar_LAE * float(mpm.gammainc(alpha_LAE+1,LOverLStar_LAE)) * volume_if_LAE
      b = phiStar_OII * float(mpm.gammainc(alpha_OII+1,LOverLStar_OII)) * volume_if_OII

      ### same thing; chain rule + recusive definition of gamma function (15-08-17)
      #a = phiStar_LAE * (1/(alpha_LAE+1)) * ( float(mpm.gammainc(alpha_LAE+2,LOverLStar_LAE)-(LOverLStar_LAE)**(alpha_LAE+1)*np.exp(-LOverLStar_LAE)) ) * volume_if_LAE
      #b = phiStar_OII * (1/(alpha_OII+1)) * ( float(mpm.gammainc(alpha_OII+2,LOverLStar_OII)-(LOverLStar_OII)**(alpha_OII+1)*np.exp(-LOverLStar_OII)) ) * volume_if_OII

   return a/(a+b)



def LAE_LF(wl_obs,sky_area,case,cosmo):
   ## returns LAE Schechter function parameters for given wavelength bin
   t0 = time.time()

   '''
   if case == 1 or case == 4 or case == 5:
      ## default faint-end slope, Ciardullo 2012
      alpha_LAE =   -1.65
   
   if case == 2:
      ## Gronwall 2007 faint-end slope
      alpha_LAE = -1.36
   
   if case == 3:
      ## Ouchi 2008 faint-end slope
      alpha_LAE = -1.5
   
   '''


   z_LAE = wl_obs/1215.668-1
   
   phiStar_LAE = lg.phiStarExt(z_LAE,case)
   
   logLStar = lg.LStarExt(z_LAE,case)
   LStar = 10**logLStar
      

   L_min = 4*np.pi*( 3.08567758e24 * nb.lumDist(z_LAE,cosmo) )**2 * nb.lineSens(wl_obs) / np.sqrt(300./sky_area)
   
   
   '''
      number of objects between the flux cut (function of wavelength) and the specified luminosity
      
   '''
   
   norm_to_1 = ( phiStar_LAE * float(mpm.gammainc(alpha_LAE+1,L_min/LStar)) )**-1
   ### same thing; chain rule + recusive definition of gamma function (15-08-17)
   #norm_to_1 = ( phiStar_LAE / (alpha_LAE+1) * float(mpm.gammainc(alpha_LAE+2,L_min/LStar)-(L_min/LStar)**(alpha_LAE+1)*np.exp(-L_min/LStar)) )**-1
   
   return norm_to_1, phiStar_LAE, LStar, z_LAE, L_min



def LAE_EW(wl_obs,case):
   ## returns e-folding scale length of LAE EW distribution for given wavelength bin
   t0 = time.time()

   z_LAE = wl_obs/1215.668-1
   w_0 = lg.w_0Ext(z_LAE,case)
   
   return w_0



def OII_LF(wl_obs,sky_area,cosmo):
   ## returns [O II] Schechter function parameters for given wavelength bin
   t0 = time.time()
   
   '''
   alpha_OII = -1.2

   '''

   z_OII = wl_obs/3727.45-1
   
   #phiStar_OII = og.phiStarFactor(z_OII,'base')      ### holy shit bug (discovered 09-09-10)
   
   logLStar = og.LStarExt(z_OII,'base')
   LStar = 10**logLStar
   
   phiStar_OII = float(og.phiStarFactor(z_OII,'base') / mpm.gammainc(-1.2+1,10**40.5/LStar))
   
   L_min = 4*np.pi*( 3.08567758e24 * nb.lumDist(z_OII,cosmo) )**2 * nb.lineSens(wl_obs) / np.sqrt(300./sky_area)

   '''
      number of objects between the flux cut (function of wavelength) and the specified luminosity
   
   '''

   norm_to_1 = ( phiStar_OII * float(mpm.gammainc(alpha_OII+1,L_min/LStar)) )**-1
   ### same thing; chain rule + recusive definition of gamma function (15-08-17)
   #norm_to_1 = ( phiStar_OII / (alpha_OII+1) * float(mpm.gammainc(alpha_OII+2,L_min/LStar)-(L_min/LStar)**(alpha_OII+1)*np.exp(-L_min/LStar)) )**-1

   return norm_to_1, phiStar_OII, LStar, z_OII, L_min



def OII_EW(wl_obs):
   ## returns e-folding scale length of [O II] EW distribution for given wavelength bin
   t0 = time.time()
   
   z_OII = wl_obs/3727.45-1
   
   '''
   if EW_case == 'lognormal': 
   ### (09-03-15) this part should never be called to run,
                  i.e., always assume exponential form in Bayesian method
      fitParam, fitCovar = fln.run(z_OII,EW_case)
      W_0, sigma = fitParam[0], fitParam[1]
      return W_0, sigma
      
   '''
      
   w_0 = og.w_0Ext(z_OII,'base')
   return w_0


