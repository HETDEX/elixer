'''
   Last updated: 6 Oct 2015
   
   Andrew Leung
   Rutgers University
   
   Wrapper for bayesian.py
     - HETDEX flux detection limits
   
'''

from datapath import *
import numpy as np
import bisect
import time
import matplotlib.pyplot as plt
import cosmolopy.distance as cd
import bayesian as b
import random
import madau
import gc
import scipy.special as ssp
import collections
import imaging as im
import sigma_dA as sda
import fit_lognorm as fln

reload(sda)



'''
   Bayes' theorem: P(A|B) = [P(B|A)*P(A)]/P(B)

   P(A) - probability that object is an LAE

   P(B) - probability that the following conditions are simultaneously true
            #equivalent width (observed) is smaller than observed for object
            #wavelength (observed) is shorter than observed for object
            #continuum flux is smaller than observed for object
            #emission line flux is smaller than observed for object

   P(A|B) - probability that object is an LAE given that conditions in B are true

   P(B|A) - probability that conditions in B are true given that object is an LAE

'''


def assembleData(sc,sa,x,y,isb,lim,runname,selectplots,genrand,filter,meta_trans_factor,cc):
   t0 = time.time()
   
   global scale
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global true_ew_inf,true_ew_obs,true_em_line_flux,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n
   global true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global f_nu_cont_slope
   global sigma_line_flux,sigma_f_nu,sigma_ew_obs,sigma_g_minus_r
   
   scale = sc
   sky_area = sa
   lenx,leny = len(x[2]),len(y[2])
   
   ## index
   d00 = np.arange(1,lenx+leny+1)
   
   ## redshift
   #d01 = np.append(x[2],y[2])
   d01 = np.append(np.array(x[6])/1215.668-1,np.array(y[6])/3727.45-1)
   
   ## redshift bin
   d02 = np.append(x[3],y[3])
   
   ## luminosity [erg/s]
   d03 = np.append(x[4],y[4])

   ## luminosity distance [Mpc]
   t1 = time.time()
   d04 = lumDist(d01,cc)
   print('nb.assembleData(), %.2f seconds to compute luminosity distances' % (time.time()-t1))
   
   ## wavelength (observed) [angstroms]
   d05 = np.append(x[6],y[6])
   
   ## equivalent width (rest frame) [angstroms]
   d06 = np.append(x[5],y[5])
   
   ## equivalent width (observed) [angstroms]
   d07 = d06*(1.+d01)
   
   ## flux (emission line) [erg/s/cm^2]
   d08 = d03/(4*np.pi*(3.085678e24*d04)**2)

   ## continuum flux density (f_nu) [microjanskys]
   d09 = (1e6)*(1e23)*(d08)/(d07)*(d05**2)/(2.997925e18)
   
   ## inferred Lyman-alpha redshift
   d10 = np.append(np.array(x[6])/1215.668-1,np.array(y[6])/1215.668-1)
   '''
   zinf = []
   for i in range(leny): zinf.append((y[6][i]/1215.668)-1)
   d10 = np.append(x[2],zinf)
   
   '''
   
   ## inferred Lyman-alpha equivalent width [angstroms]
   d11 = d07/(1.+d10)                              ## W_Lya = EW_obs / (1+z_Lya)
   
   '''
      inferred EW calculated by 'de-redshifting' the observed EW using the Lyman-alpha redshift, i.e., as if the object were an LAE
      
   '''
      
   ## signal-to-noise
   five_sigma_cont = 10**(-0.4*(25.1-23.9))         ## in uJy
   d12 = d09/(0.2*five_sigma_cont)
   
   ## object type
   d13 = ['']*(lenx+leny)
   for i in range(lenx): d13[i] = 'LAE'
   for i in range(lenx,lenx+leny): d13[i] = 'OII'
   
   ## AB magnitude of continuum flux
   d14 = np.ones(lenx+leny) * -999
   for i in range(len(d14)):
      if d09[i] > 0:
         d14[i] = 23.9-2.5*(np.log10(d09[i]))
      else:
         d14[i] = 999
   
   #global true_ew_obs_, true_ew_inf_, true_em_line_flux_, true_cont_flux_dens_, true_AB_cont_mag_
   true_ew_obs_ = np.copy(d07)
   true_ew_inf_ = np.copy(d11)
   true_em_line_flux_ = np.copy(d08)
   true_cont_flux_dens_ = np.copy(d09)
   true_AB_cont_mag_ = np.copy(d14)

   ## g minus r color index
   d15 = np.append(x[8],y[8])

   ## g minus i color index
   d16 = np.append(x[9],y[9])

   ## g minus z color index
   d17 = np.append(x[10],y[10])

   ## r minus i color index
   d18 = np.append(x[11],y[11])

   ## r minus z color index
   d19 = np.append(x[12],y[12])

   ## i minus z color index
   d20 = np.append(x[13],y[13])


   '''
      Noise is added to emission line flux; this noisified flux simulates the quantity that the detector would record. The 'true' line flux is used for the extrapolation of f_nu to band of the imaging survey, but since the survey is line flux limited, we can reduce the number of galaxies for which we have to extrapolate f_nu (a time-consuming process at ~0.06 s per object) by first cutting the sample based on the noisified flux limit. The 'true' line flux is preserved as a separate variable.
      
   '''

   ## temporarily replaced to debug imaging survey measurement noise (10-21-14)
   ## restored; save vector of random values (10-22-14)
   t1 = time.time()

   if genrand:
      rd_lineFlux = np.random.rand(len(d00))
      data = open(str(rstrpath)+'rd_lineFlux_full_'+str(runname)+'.dat','w')
      data.write('# \n')
      data.write('# Array of random numbers \n')
      data.write('#   * used to simulate measurement noise for line flux \n')
      data.write('#   * full simulated sample (before applying line flux limit) \n')
      if scale == 0.1:    data.write('#   * one-tenth scale \n')
      elif scale == 0.25: data.write('#   * one-quarter scale \n')
      elif scale == 1.:   data.write('#   * full scale \n')
      else: data.write('#   * %.2f scale \n'%scale)
      data.write('#   *   '+str(int(lenx))+' LAEs \n')
      data.write('#   *   '+str(int(leny))+' [O II] emitters \n')
      data.write('# \n')
      data.write('# Column 1:  index number \n')
      data.write('# Column 2:  random number \n')
      data.write('# \n')
      data.write('# \n')
      for i in range(len(d00)): data.write(str(d00[i])+'\t'+str(rd_lineFlux[i])+'\n')
      data.write('# \n')
      data.close()
   
   elif (runname[:6] == '150717')\
         or (runname[:7] == '150903_')\
         or (runname[:6] == '150906')\
         or (runname[:6] == '150908')\
         or (runname[:6] == '151006'):
   
      rd_LAE_lineFlux,index_rd_LAE = [],[]
      if runname[:6] == '150717': data = open(str(rstrpath)+'rd_lineFlux_full_150618_g0522_25.10.dat','r')
      elif (runname[:7] == '150903_') or (runname[:6] == '150906'): data = open(str(rstrpath)+'rd_lineFlux_full_150903_g0522_25.10.dat','r')
      elif runname[:6] == '150908': data = open(str(rstrpath)+'rd_lineFlux_full_150908_g0522_25.10.dat','r')
      elif runname[:6] == '151006': data = open(str(rstrpath)+'rd_lineFlux_full_151006_g0522_25.10.dat','r')


      ln,LAEsimct_LAE = 0,0
      for line in data.readlines():
         ln += 1
         thisline = line.split()
         if runname[:6] == '150717' and ln == 5:
            LAEsimct_LAE = int(thisline[2])
         elif ((runname[:7] == '150903_') or (int(runname[:6]) >= 150906)) and ln == 6:
            LAEsimct_LAE = int(thisline[2])
         if not line.startswith('#'):
            index_rd_LAE.append(int(thisline[0]))
            rd_LAE_lineFlux.append(float(thisline[1]))
      data.close()
      print('LAE count from LAE file: '+str(LAEsimct_LAE))

      rd_OII_lineFlux,index_rd_OII = [],[]
      if runname[:6] == '150717': data = open(str(rstrpath)+'rd_lineFlux_full_150525_1.0_g0522_25.10.dat','r')
      elif (runname[:7] == '150903_') or (runname[:6] == '150906'): data = open(str(rstrpath)+'rd_lineFlux_full_150903_g0522_25.10.dat','r')
      elif runname[:6] == '150908': data = open(str(rstrpath)+'rd_lineFlux_full_150908_g0522_25.10.dat','r')
      elif runname[:6] == '151006': data = open(str(rstrpath)+'rd_lineFlux_full_151006_g0522_25.10.dat','r')

      ln,LAEsimct_OII = 0,0
      for line in data.readlines():
         ln += 1
         thisline = line.split()
         if runname[:6] == '150717' and ln == 5:
            LAEsimct_OII = int(thisline[2])
         elif ((runname[:7] == '150903_') or (int(runname[:6]) >= 150906)) and ln == 6:
            LAEsimct_OII = int(thisline[2])
         if not line.startswith('#'):
            index_rd_OII.append(int(thisline[0]))
            rd_OII_lineFlux.append(float(thisline[1]))
      data.close()
      print('LAE count from OII file: '+str(LAEsimct_OII))

      index_rd = np.append(index_rd_LAE[:LAEsimct_LAE],index_rd_OII[LAEsimct_OII:])
      rd_lineFlux = np.append(rd_LAE_lineFlux[:LAEsimct_LAE],rd_OII_lineFlux[LAEsimct_OII:])

      print('***')
      print('*** nb.assembleData(): check list match at import of vector of random numbers for line flux measurement noise ***')
      print(len(rd_lineFlux), (lenx+leny))
      print(len(rd_lineFlux) == (lenx+leny))
      print(np.array(index_rd) == d00)
      print('***')
      print('***')

      rd_LAE_lineFlux,index_rd_LAE = [],[]
      rd_OII_lineFlux,index_rd_OII = [],[]

   else:
      rd_lineFlux,index_rd = [],[]
      if (runname[:6] == '140724'):
         data = open(str(rstrpath)+'rd_lineFlux_full_140724_r1102_24.47.dat','r')
      elif (runname[8:12] == '0121') \
        or (runname[8:12] == '0122') \
        or (runname[8:12] == '0128') \
        or (runname[8:12] == '0214') \
        or (runname[8:12] == '0217') \
        or (runname[8:12] == '0219') \
        or True:      ## default to this file, never 'else'
         data = open(str(rstrpath)+'rd_lineFlux_full_'+runname[:6]+'_r1102_24.85.dat','r')
      else:
         data = open(str(rstrpath)+'rd_lineFlux_full_'+runname[:6]+'_r'+runname[8:12]+'_24.85.dat','r')

      ## added column, old format obsolete
      ##if runname[:6] == '140724': data = open(str(rstrpath)+'rd_lineFlux_full_140724_r1022.dat','r')
      ##elif runname[:6] == '141025': data = open(str(rstrpath)+'rd_lineFlux_full_141025_r1025.dat','r')
      
      for line in data.readlines():
         if not line.startswith('#'):
            a = line.split()
            index_rd.append(float(a[0]))
            rd_lineFlux.append(float(a[1]))
      data.close()

      print('***')
      print('*** nb.assembleData(): check list match at import of vector of random numbers for line flux measurement noise ***')
      print(len(rd_lineFlux), (lenx+leny))
      print(len(rd_lineFlux) == (lenx+leny))
      print(np.array(index_rd) == d00)
      print('***')
      print('***')

   sigma_line_flux = 0.2 * lineSens(d05) * np.sqrt(sky_area/300.)
   for i in range(len(d00)):
      d08[i] = float('%.4e'%(d08[i] + ncdfinv(rd_lineFlux[i],0,sigma_line_flux[i])))
                  
      if d08[i] < 0:   d08[i] = 0.            ## if line flux with noise added is < 0, set line flux = 0
            
      if i%(1e5) == 0: print('nb.assembleData(), '+str(i+1)+'st loop to add noise to emission line flux: %.2f seconds' % (time.time()-t1))

   wl = d05
   fline = d08
   rd_lineFlux,index_rd = [],[]         ## clear arrays of random numbers and object indices from memory
   deleteIndex = []
   for i in range(lenx+leny):
      ''' ### turned off for 0214 (02-15-15)
      ### (GZ 01/28 request: test 4 times deeper)
      if (fline[i] < 0.25 * lineSens(wl[i])) or (wl[i] < 3500.) or (wl[i] > 5500.):
      '''
      ###### (05-28-15) spectroscopic depth depends on area surveyed per unit observing time
      if (fline[i] < lineSens(wl[i])*np.sqrt(sky_area/300.) ) or (wl[i] < 3500.) or (wl[i] > 5500.):
         deleteIndex.append(i)
      if i%(1e5) == 0: print('nb.assembleData(), '+str(i+1)+'st loop to apply line flux detection limit: %.2f seconds' % (time.time()-t1))
   print('len(deleteIndex) = '+str(len(deleteIndex)))

   ## create new dataset ("observable sample") consisting only of objects whose emission line flux is greater than HETDEX's line detection sensitivity
   c00 = np.delete(d00,deleteIndex)         ## index
   c01 = np.delete(d01,deleteIndex)         ## redshift
   c02 = np.delete(d02,deleteIndex)         ## redshift bin
   c03 = np.delete(d03,deleteIndex)         ## luminosity [erg/s]
   c04 = np.delete(d04,deleteIndex)         ## luminosity distance [Mpc]
   c05 = np.delete(d05,deleteIndex)         ## wavelength (observed) [angstroms]
   c06 = np.delete(d06,deleteIndex)         ## equivalent width (rest frame) [angstroms]
   c07 = np.delete(d07,deleteIndex)         ## equivalent width (observed) [angstroms]
   c08 = np.delete(d08,deleteIndex)         ## flux (emission line) [erg/s/cm^2]
   c09 = np.delete(d09,deleteIndex)         ## flux (continuum) [microjanskys]
   c10 = np.delete(d10,deleteIndex)         ## inferred Lyman-alpha redshift
   c11 = np.delete(d11,deleteIndex)         ## inferred Lyman-alpha equivalent width [angstroms]
   c12 = np.delete(d12,deleteIndex)         ## signal-to-noise
   c13 = np.delete(d13,deleteIndex)         ## object type
   c14 = np.delete(d14,deleteIndex)         ## AB magnitude of continuum flux
   c15 = np.delete(d15,deleteIndex)         ## g minus r color index
   c16 = np.delete(d16,deleteIndex)         ## g minus i color index
   c17 = np.delete(d17,deleteIndex)         ## g minus z color index
   c18 = np.delete(d18,deleteIndex)         ## r minus i color index
   c19 = np.delete(d19,deleteIndex)         ## r minus z color index
   c20 = np.delete(d20,deleteIndex)         ## i minus z color index

   true_em_line_flux = np.delete(true_em_line_flux_,deleteIndex)
   sigma_line_flux = np.delete(sigma_line_flux,deleteIndex)

   true_ew_rest_fs = np.copy(c06)
   true_ew_obs_fs = np.copy(c07)
   true_ew_inf_fs = np.copy(c11)
   true_cont_flux_dens_fs = np.copy(c09)
   true_AB_cont_mag_fs = np.copy(c14)
   true_s_to_n_fs = np.copy(c12)

   true_g_minus_r = np.copy(c15)
   true_g_minus_i = np.copy(c16)
   true_g_minus_z = np.copy(c17)
   true_r_minus_i = np.copy(c18)
   true_r_minus_z = np.copy(c19)
   true_i_minus_z = np.copy(c20)

   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   ## clear memory of arrays with no future use
   d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
   gc.collect()


   '''
      AL (08-14-14): Use g-r colors (drawn randomly from Gaussian-fitted g-r color distributions of LAEs and [O II] emitters in HPS) to extrapolate imaging survey f_nu value (which is computed under the assumption that all galaxies have flat spectra in f_nu) from lambda_EL to g-band effective wavelength and r-band effective wavelength.
      
      AL (08-19-14): g-r color index of an object tells us the relative brightness of its continuum emission at the effective wavelength of the g-band and r-band filters. f_nu at lambda_EL is given as a relationship of two simulated quantities: flux of the emission line and the equivalent width of the line. To extrapolate f_nu to the effective wavelength of the g-band and r-band filters, I suppose the slope of the continuum is a power law. (I examined the SEDs that I fitted for two z~0 galaxies this summer... the slope between g' and r' is more or less a straight line on a log-log plot, and the two bands are very close together relative to the full range of wavelengths for the SED that approximating with anything but a straight line seems inappropriate.)
      
      We have four equations:
      [1]  log(f_nu_at_EL) = m * log(lambda_EL) + b
      [2]  log(f_nu_at_g) = m * log(lambda_eff_g) + b
      [3]  log(f_nu_at_r) = m * log(lambda_eff_r) + b
      [4]  f_nu_at_r = 10^(0.4*[g-r]) * f_nu_at_g
      
      Constants:
      lambda_eff_g = 4750 angstroms
      lambda_eff_r = 6220 angstroms
      
      Simulated quantities:
      f_nu_at_EL, lambda_EL, and [g-r]
      
      So, for each simulated galaxy, we have four equations and four unknowns: m, b, f_nu_at_g, and f_nu_at_r.
      Hooray! Now crank out the algebra.
      
      (determine the power law coefficient and slope)
      
   '''

   z = c01
   zinf = c10
   ewinf = c11
   f_nu_at_EL = c09                           ## f_nu (at lambda_EL) assuming flat spectra

   lambda_EL = c05

   if True:   #(runname[8:12] == '0214') or (runname[8:12] == '0217') or (runname[8:12] == '0219'):
      lambda_eff_r = 6292.
      lambda_eff_z = 9400.
      ## reference: http://www.noao.edu/kpno/mosaic/filters/

      r_minus_z = c19

      m = 0.4 * r_minus_z / np.log10(lambda_eff_z/lambda_eff_r)

   ''' (04-24-15) commented out snipets that uses g-r color for spectral slope, since replaced by r-z
   else:
      lambda_eff_g = 4750.         #4686. #(g0115)      #4750. #(g1102,g0119)
      lambda_eff_r = 6220.         #6165. #(r0115)      #6220. #(r1102,r0119)

      g_minus_r = c15

      m = 0.4 * g_minus_r / np.log10(lambda_eff_r/lambda_eff_g)

   '''

   f_nu_cont_slope = np.copy(m)

   # b = np.log10(f_nu_at_EL) - 0.4 * g_minus_r * np.log10(lambda_EL) / np.log10(lambda_eff_r/lambda_eff_g)
   ## TypeError: can't multiply sequence by non-int of type 'float'
   b = np.ones(len(z)) * -999

   if True:   #(runname[8:12] == '0214') or (runname[8:12] == '0217') or (runname[8:12] == '0219'):
      for i in range(len(b)):
         b[i] = np.log10(f_nu_at_EL[i]) - 0.4 * r_minus_z[i] * np.log10(lambda_EL[i]) / np.log10(lambda_eff_z/lambda_eff_r)

   ''' (04-24-15) commented out snipets that uses g-r color for spectral slope, since replaced by r-z
   else:
      for i in range(len(b)):
         b[i] = np.log10(f_nu_at_EL[i]) - 0.4 * g_minus_r[i] * np.log10(lambda_EL[i]) / np.log10(lambda_eff_r/lambda_eff_g)
         
   '''

   '''
      add emission line to galaxy spectra
         FWHM of a gaussian = 2 * sqrt(2*ln(2)*(sigma^2)) = 2.35 * sigma
      
      attenuate by Madau prescription for IGM absorption
      
      convolve with g' or r' filter
      
   '''

   FWHM = 5.                     ## 5 angstrom wide line (see EG 8/25/14 email)
   f_from_EL = true_em_line_flux
   objtype = c13

   if True:   #(runname[8:12] == '0214') or (runname[8:12] == '0217') or (runname[8:12] == '0219'):
      if isb == 'u':
         bandpass_min     = 302
         bandpass_max     = 403
      
      elif isb == 'g':
         bandpass_min     = 386
         bandpass_max     = 576

      elif isb == 'r':
         bandpass_min     = 531
         bandpass_max     = 719

      elif isb == 'i':
         bandpass_min     = 675
         bandpass_max     = 867

      elif isb == 'z':
         bandpass_min     = 808
         bandpass_max     = 1099

      ## source of data files: http://www.aip.de/en/research/facilities/stella/instruments/data/sloanugriz-filter-curves

      wl_over_bandpass = 10*np.array(range(bandpass_min,bandpass_max))
      f_nu_at_wl_inc = np.ones(len(z)) * -999
         
      if filter == 'sdss':
         if   isb == 'u': data = open(fltrpath+'Sloan_u.txt','r')
         elif isb == 'g': data = open(fltrpath+'Sloan_g.txt','r')
         elif isb == 'r': data = open(fltrpath+'Sloan_r.txt','r')
         elif isb == 'i': data = open(fltrpath+'Sloan_i.txt','r')
         elif isb == 'z': data = open(fltrpath+'Sloan_z.txt','r')
      
      elif filter == 'hsc':
         if   isb == 'g': data = open(fltrpath+'HSC-g.dat','r')
         elif isb == 'r': data = open(fltrpath+'HSC-r.dat','r')

      wl_filter,transmission = [],[]
      for line in data.readlines():
         if not line.startswith('#') and not line.startswith('<'):
            thisLine = line.split()
            if float(thisLine[0]) >= bandpass_min-0.5 and float(thisLine[0]) <= bandpass_max-0.5:
               wl_filter.append(10.*float(thisLine[0]))
               transmission.append(float(thisLine[1])/100.)
      data.close()

      if filter == 'sdss':
         wl_filter = meta_trans_factor * np.array(wl_filter[::-1])
         transmission = np.array(transmission[::-1])
      
      data = open('QE_mosaic.dat','r')
      wl_QE, quantum_eff = [],[]
      for line in data.readlines():
         thisLine = line.split()
         wl_QE.append(float(thisLine[0]))
         quantum_eff.append(float(thisLine[1]))
      data.close()
         
      t1 = time.time()
      for i in range(len(z)):            ## for each simulated galaxy above the flux limit
         f_nu_over_bandpass = 10 ** (m[i] * np.log10(wl_over_bandpass) + b[i])                     ## extrapolate f_nu to specified survey band
         f_lambda_over_bandpass = (2.997925e18)/(wl_over_bandpass**2) * (1e-29)*f_nu_over_bandpass         ## convert to f_lambda
         ref_spectrum = (2.997925e18)/(wl_over_bandpass**2) * (1e-29)*np.ones(len(wl_over_bandpass))      ## 1 uJy reference spectrum converted to f_lambda
               
         taueff = []
         for j in range(len(wl_over_bandpass)):
            taueff.append(madau.mktaueff(wl_over_bandpass[j],z[i]))
         taueff = np.array(taueff)
               
         ## attenuate measured flux at each wavelength interval with tau_effective calculated according to Madau (1995) prescription
         f_nu_over_bandpass  *= np.exp(-1*taueff)
         f_lambda_over_bandpass *= np.exp(-1*taueff)
         ref_spectrum *= np.exp(-1*taueff)

         if True:      ## (04-24-15) always correct for CCD quantum efficiency
            trans_factor = np.interp(wl_over_bandpass,wl_filter,transmission) * np.interp(wl_over_bandpass,wl_QE,quantum_eff)
         
         ## convolve flux with filter transmission curve
         f_nu_over_bandpass *= trans_factor
         f_lambda_over_bandpass *= trans_factor
         ref_spectrum *= trans_factor

         if 10*(bandpass_min-0.5) <= lambda_EL[i] <= 10*(bandpass_max-0.5):      ## add emission line if within filter bandpass
            wl_with_EL = lambda_EL[i]+0.01*np.array(range(-1500,1501))
            truncIndex = []
            for j in range(len(wl_with_EL)):
               if wl_with_EL[j] < 10*(bandpass_min-0.5) or wl_with_EL[j] > 10*(bandpass_max-0.5): truncIndex.append(j)
            wl_with_EL = np.delete(wl_with_EL,truncIndex)      ## truncate portion of emission line that falls outside of bandpass
            
            f_lambda_of_EL = f_from_EL[i] * 2/FWHM*np.sqrt(np.log(2)/np.pi) * np.exp(-4*np.log(2)*(np.delete(0.01*np.array(range(-1500,1501)),truncIndex)/FWHM)**2)               ## gaussian line profile
            f_nu_of_EL = (1e29)*f_lambda_of_EL * (wl_with_EL**2)/(2.997925e18)
            
            if True:      ## (04-24-15) always correct for CCD quantum efficiency
               trans_factor = np.interp(wl_with_EL,wl_filter,transmission) * np.interp(wl_with_EL,wl_QE,quantum_eff)

            ## convolve emission line flux with filter transmission (including CCD quantum efficiency)
            f_nu_of_EL *= trans_factor
            f_lambda_of_EL *= trans_factor
               
            int_flux_bandpass = sum(f_lambda_over_bandpass)*10 + sum(f_lambda_of_EL)*0.01
            
         else: int_flux_bandpass = sum(f_lambda_over_bandpass)*10

         if objtype[i] == 'OII':
         ## (04-24-15) add emission lines (if OII, add others)
         
            addl_em_lines  = ['[NeIII]','H_beta','[OIII]','[OIII]']
            addl_lambda_rf = np.array([3869.00, 4861.32, 4958.91, 5006.84])
            addl_lambda_ob = addl_lambda_rf * (1+z[i])

            rel_strength   = np.array([0.416, 1., 1.617, 4.752])/1.791      ## Anders_Fritze_2003.dat, metallicity one-fifth solar
            #rel_strength   = np.array([0.300, 1., 1.399, 4.081])/3.010      ## (05-06-15) Anders_Fritze_2003.dat, metallicity 0.5-2 in solar units
            addl_fluxes    = rel_strength * f_from_EL[i]
            
            if selectplots and (np.mod(zinf[i],0.05) <= (1.2e-4) or ewinf[i] > 20):
            
               plt.close()
               if objtype[i] == 'LAE': linecolor,text = 'red','LAE spectrum'   ## buried in OII-only loop so this should never happen
               else: linecolor,text = 'blue','[O II] emitter'
            
               plt.scatter(wl_over_bandpass,f_nu_over_bandpass,s=2,lw=0,c=linecolor)
            
               if 10*(bandpass_min-0.5) <= lambda_EL[i] <= 10*(bandpass_max-0.5):
                  f_nu_with_EL = np.interp(wl_with_EL,wl_over_bandpass,f_nu_over_bandpass)      ## initialize, without emission line
                  f_nu_with_EL += f_nu_of_EL
                  plt.scatter(wl_with_EL,f_nu_with_EL,s=0.5,lw=0,c=linecolor)
                  plt.ylim(0,1.1*max(max(f_nu_with_EL),max(f_nu_over_bandpass)))
               else: plt.ylim(0,1.1*max(f_nu_over_bandpass))
            
            for k in range(len(addl_em_lines)):
               if 10*(bandpass_min-0.5) <= addl_lambda_ob[k] <= 10*(bandpass_max-0.5):
                  wl_with_line = addl_lambda_ob[k]+0.01*np.array(range(-1500,1501))
                  truncIndex = []
                  for j in range(len(wl_with_line)):
                     if wl_with_line[j] < 10*(bandpass_min-0.5) or wl_with_line[j] > 10*(bandpass_max-0.5): truncIndex.append(j)
                  wl_with_line = np.delete(wl_with_line,truncIndex)      ## truncate portion of emission line that falls outside of bandpass
                  
                  f_lambda_of_line = addl_fluxes[k] * 2/FWHM*np.sqrt(np.log(2)/np.pi) * np.exp(-4*np.log(2)*(np.delete(0.01*np.array(range(-1500,1501)),truncIndex)/FWHM)**2)               ## gaussian line profile
                  f_nu_of_line = (1e29)*f_lambda_of_line * (wl_with_line**2)/(2.997925e18)
                  
                  trans_factor = np.interp(wl_with_line,wl_filter,transmission) * np.interp(wl_with_line,wl_QE,quantum_eff)
                  
                  ## convolve emission line flux with filter transmission curve
                  f_nu_of_line *= trans_factor
                  f_lambda_of_line *= trans_factor
                  
                  if selectplots and (np.mod(zinf[i],0.05) <= (1.2e-4) or ewinf[i] > 20):
                  
                     f_nu_with_line = np.interp(wl_with_line,wl_over_bandpass,f_nu_over_bandpass)      ## initialize, without emission line
                     f_nu_with_line += f_nu_of_line
                     plt.scatter(wl_with_line,f_nu_with_line,s=0.5,lw=0,c=linecolor)
                  
                  int_flux_bandpass += sum(f_lambda_of_line)*0.01
            
            if selectplots and (np.mod(zinf[i],0.05) <= (1.2e-4) or ewinf[i] > 20):
               
               plt.grid()
               plt.xlabel('observed-frame wavelength (\AA)')
               plt.ylabel('flux density (uJy)')

               int_flux_ref = sum(ref_spectrum)*10
               f_nu = int_flux_bandpass/int_flux_ref
               plt.title('z = %.3f '%(z[i])+text+' convolved with '+filter+' '+isb+'\' filter, '+'f_nu = %.2f'%(f_nu)+' uJy')
                  
               plt.savefig(specpath+str(runname)+'_'+isb+'-band_spectrum_%08d' % int(c00[i]) +'_f_nu.pdf')
               plt.close()
      
         int_flux_ref = sum(ref_spectrum)*10            ## interval of wavelength for integration = 10 A
         
         f_nu = int_flux_bandpass/int_flux_ref
         f_nu_at_wl_inc[i] = round(f_nu,5)
         
         if i%(1e3)==0 or i<=10:
            print('nb.assembleData(), %.3f seconds since ' %(time.time()-t1) +filter+' '+isb+'\' filter file loaded')
            print('  i = '+str(i)+', redshift of galaxy is %.4f, f_nu = %.4f uJy' % (z[i],f_nu_at_wl_inc[i]))

         if i%(1e3)==0:
            gc.collect()

      c09 = np.array(f_nu_at_wl_inc)
      c07 = (1e6)*(1e23)*(true_em_line_flux)/(f_nu_at_wl_inc)*((lambda_EL)**2)/(2.997925e18)      ## recalculate ew_obs based on f_nu at bandpass
   

   else:   ## (04-24-15) soon to comment out, uses g-r color for spectral slope, since replaced by r-z

      if isb == 'g':                                                ## if imaging survey is in g-band
         print('****** assuming g-band imaging survey ******')
         
         wl_over_g_band = 10*np.array(range(386,576))
         f_nu_at_g = np.ones(len(z)) * -999
         
         if filter == 'sdss':   data = open(fltrpath+'Sloan_g.txt','r')
         elif filter == 'hsc': data = open(fltrpath+'HSC-g.dat','r')
         
         wl_filter,transmission = [],[]
         for line in data.readlines():
            if not line.startswith('#'):
               thisLine = line.split()
               if float(thisLine[0]) >= 385.5 and float(thisLine[0]) <= 575.5:
                     wl_filter.append(10.*float(thisLine[0]))
                     transmission.append(float(thisLine[1])/100.)
         data.close()

         wl_filter = meta_trans_factor * np.array(wl_filter[::-1])
         transmission = np.array(transmission[::-1])

         t1 = time.time()
         for i in range(len(z)):            ## for each simulated galaxy above the flux limit
            f_nu_over_g_band = 10 ** (m[i] * np.log10(wl_over_g_band) + b[i])                     ## extrapolate f_nu to g-band
            f_lambda_over_g_band = (2.997925e18)/(wl_over_g_band**2) * (1e-29)*f_nu_over_g_band         ## convert to f_lambda
            ref_spectrum = (2.997925e18)/(wl_over_g_band**2) * (1e-29)*np.ones(len(wl_over_g_band))      ## 1 uJy reference spectrum converted to f_lambda

            taueff = []
            for j in range(len(wl_over_g_band)):
               taueff.append(madau.mktaueff(wl_over_g_band[j],z[i]))
            taueff = np.array(taueff)
            
            ## attenuate g-band at each wavelength interval with tau_effective calculated according to Madau (1995) prescription
            f_nu_over_g_band  *= np.exp(-1*taueff)
            f_lambda_over_g_band *= np.exp(-1*taueff)
            ref_spectrum *= np.exp(-1*taueff)

            trans_factor = np.interp(wl_over_g_band,wl_filter,transmission)

            ## convolve flux with g-band transmission curve
            f_nu_over_g_band *= trans_factor
            f_lambda_over_g_band *= trans_factor
            ref_spectrum *= trans_factor
            
            if lambda_EL[i] >= 3855 and lambda_EL[i] <= 5755:            ## add emission line if within filter bandpass
               wl_with_EL = lambda_EL[i]+0.01*np.array(range(-1500,1501))
               truncIndex = []
               for j in range(len(wl_with_EL)):
                  if wl_with_EL[j] < 3855 or wl_with_EL[j] > 5755: truncIndex.append(j)
               wl_with_EL = np.delete(wl_with_EL,truncIndex)               ## truncate portion of emission line that falls outside of bandpass
               
               f_lambda_of_EL = f_from_EL[i] * 2/FWHM*np.sqrt(np.log(2)/np.pi) * np.exp(-4*np.log(2)*(np.delete(0.01*np.array(range(-1500,1501)),truncIndex)/FWHM)**2)               ## gaussian line profile
               f_nu_of_EL = (1e29)*f_lambda_of_EL * (wl_with_EL**2)/(2.997925e18)
               
               trans_factor = np.interp(wl_with_EL,wl_filter,transmission)
                           
               ## convolve flux with g-band transmission curve
               f_nu_of_EL *= trans_factor
               f_lambda_of_EL *= trans_factor
               
               '''
                  flux density of emission line integrated (0.01A intervals) -> emission line contribution
                  flux density of continnum integrated (10A intervals) -> continuum contribution
                  sum of the two = total flux in g-band
               
               '''
               int_flux_g_band = sum(f_lambda_over_g_band)*10 + sum(f_lambda_of_EL)*0.01
            
            else: int_flux_g_band = sum(f_lambda_over_g_band)*10

            int_flux_ref = sum(ref_spectrum)*10            ## interval of wavelength for integration = 10 A

            f_nu = int_flux_g_band/int_flux_ref
            f_nu_at_g[i] = round(f_nu,4)
            
            if i%(1e3)==0 or i<=10:
               print('nb.assembleData(), %.3f seconds since g\' filter loaded' % (time.time()-t1))
               print('  i = '+str(i)+', redshift of galaxy is %.4f, f_nu = %.4f uJy' % (z[i],f_nu_at_g[i]))

            if selectplots:
               if np.abs(zinf[i]-1.9)<=(1.2e-4) or \
                  np.abs(zinf[i]-1.95)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.0)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.05)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.1)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.15)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.2)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.25)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.3)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.35)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.4)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.45)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.5)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.55)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.6)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.65)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.7)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.75)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.8)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.85)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.9)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.95)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.0)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.05)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.1)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.15)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.2)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.25)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.3)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.35)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.4)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.45)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.5)<=(1.2e-4) or \
                  (ewinf[i] > 20 and objtype[i] == 'OII'):
               
                  if lambda_EL[i] >= 3855 and lambda_EL[i] <= 5755:
                     f_lambda_with_EL = np.interp(wl_with_EL,wl_over_g_band,f_lambda_over_g_band)      ## initialize, without emission line
                     f_lambda_with_EL += f_lambda_of_EL
                     f_nu_with_EL = np.interp(wl_with_EL,wl_over_g_band,f_nu_over_g_band)
                     f_nu_with_EL += f_nu_of_EL
                  
                  plt.close()
                  if objtype[i] == 'LAE': linecolor,text = 'red','LAE spectrum'
                  else: linecolor,text = 'blue','[O II] emitter'
                  
                  plt.scatter(wl_over_g_band,f_nu_over_g_band,s=2,lw=0,c=linecolor)
                  if lambda_EL[i] >= 3855 and lambda_EL[i] <= 5755:
                     plt.scatter(wl_with_EL,f_nu_with_EL,s=0.5,lw=0,c=linecolor)
                     plt.ylim(0,1.1*max(max(f_nu_with_EL),max(f_nu_over_g_band)))
                  else: plt.ylim(0,1.1*max(f_nu_over_g_band))
                  plt.xlabel(r'$\lambda$'+' ($\mathrm{\AA}$)')
                  plt.ylabel('$f$'+r'$_{\nu}$'+' $(\mu \mathrm{Jy})$')
                  plt.grid()
                  plt.title('simulated $z$=%.5f '%(z[i])+text+' convolved with $g\'$ filter, '+r'$f_{\nu}$'+' $=$ %.3f'%(f_nu)+' $\mu \mathrm{Jy}$')
                  plt.savefig(str(runname)+'_g-band_spectrum_%08d' % int(c00[i]) +'_f_nu.pdf')
                  plt.close()
                  
                  plt.scatter(wl_over_g_band,1e17*(f_lambda_over_g_band),s=2,lw=0,c=linecolor)
                  if lambda_EL[i] >= 3855 and lambda_EL[i] <= 5755:
                     plt.scatter(wl_with_EL,1e17*(f_lambda_with_EL),s=0.5,lw=0,c=linecolor)
                     plt.ylim(0,1.1e17*max(max(f_lambda_with_EL),max(f_lambda_over_g_band)))
                  else: plt.ylim(0,1.1e17*max(f_lambda_over_g_band))
                  plt.xlabel(r'$\lambda$'+' ($\mathrm{\AA}$)')
                  plt.ylabel('$f$'+r'$_{\lambda}$'+' $(10^{-17} \mathrm{erg/cm^{2}/s/\AA})$')
                  plt.grid()
                  ewobs = (1e6)*(1e23)*(true_em_line_flux[i])/(f_nu)*((lambda_EL[i])**2)/(2.997925e18)
                  plt.title('simulated $z$=%.5f '%(z[i])+text+' convolved with $g\'$ filter, $EW$'+r'$_{\mathrm{obs}}$'+' $=$ %.1f'%(ewobs)+' $\mathrm{\AA}$')
                  plt.savefig(str(runname)+'_g-band_spectrum_%08d' % int(c00[i]) +'_f_lambda.pdf')
                  plt.close()
                  
            if i%(1e3)==0:
               gc.collect()
         
         c09 = np.array(f_nu_at_g)
         c07 = (1e6)*(1e23)*(true_em_line_flux)/(f_nu_at_g)*((lambda_EL)**2)/(2.997925e18)      ## recalculate ew_obs based on f_nu at g-band


      elif isb == 'r':                                             ## if imaging survey is in r-band
         print('****** assuming r-band imaging survey ******')
         
         wl_over_r_band = 10*np.array(range(531,719))
         f_nu_at_r = np.ones(len(z)) * -999
         
         if filter == 'sdss':   data = open(fltrpath+'Sloan_r.txt','r')
         elif filter == 'hsc':    data = open(fltrpath+'HSC-r.dat','r')

         wl_filter,transmission = [],[]
         for line in data.readlines():
            if not line.startswith('#'):
               thisLine = line.split()
               if float(thisLine[0]) >= 530.5 and float(thisLine[0]) <= 718.5:
                  wl_filter.append(10.*float(thisLine[0]))
                  transmission.append(float(thisLine[1])/100.)
         data.close()

         wl_filter = meta_trans_factor * np.array(wl_filter[::-1])
         transmission = transmission[::-1]

         t1 = time.time()
         for i in range(len(z)):            ## for each simulated galaxy above the flux limit
            f_nu_over_r_band = 10 ** (m[i] * np.log10(wl_over_r_band) + b[i])                     ## extrapolate f_nu to r-band
            f_lambda_over_r_band = (2.997925e18)/(wl_over_r_band**2) * (1e-29)*f_nu_over_r_band         ## convert to f_lambda
            ref_spectrum = (2.997925e18)/(wl_over_r_band**2) * (1e-29)*np.ones(len(wl_over_r_band))      ## 1 uJy reference spectrum

            taueff = []
            for j in range(len(wl_over_r_band)):
               taueff.append(madau.mktaueff(wl_over_r_band[j],z[i]))
            taueff = np.array(taueff)

            ## attenuate r-band at each wavelength interval with tau_effective calculated according to Madau (1995) prescription (no effect on r-band except for z>3.36 LAEs)
            f_nu_over_r_band  *= np.exp(-1*taueff)
            f_lambda_over_r_band *= np.exp(-1*taueff)
            ref_spectrum *= np.exp(-1*taueff)
            
            trans_factor = np.interp(wl_over_r_band,wl_filter,transmission)

            ## convolve flux with r-band transmission curve
            f_nu_over_r_band *= trans_factor
            f_lambda_over_r_band *= trans_factor
            ref_spectrum *= trans_factor
            
            if lambda_EL[i] >= 5305 and lambda_EL[i] <= 7185:            ## add emission line if within filter bandpass
               wl_with_EL = lambda_EL[i]+0.01*np.array(range(-1500,1501))
               truncIndex = []
               for j in range(len(wl_with_EL)):
                  if wl_with_EL[j] < 5305 or wl_with_EL[j] > 7185: truncIndex.append(j)
               wl_with_EL = np.delete(wl_with_EL,truncIndex)               ## truncate portion of emission line that falls outside of bandpass
               
               f_lambda_of_EL = f_from_EL[i] * 2/FWHM*np.sqrt(np.log(2)/np.pi) * np.exp(-4*np.log(2)*(np.delete(0.01*np.array(range(-1500,1501)),truncIndex)/FWHM)**2)               ## gaussian line profile
               f_nu_of_EL = (1e29)*f_lambda_of_EL * (wl_with_EL**2)/(2.997925e18)
               
               trans_factor = np.interp(wl_with_EL,wl_filter,transmission)
               
               ## convolve flux with g-band transmission curve
               f_nu_of_EL *= trans_factor
               f_lambda_of_EL *= trans_factor
            
               '''
                  flux density of emission line integrated (0.01A intervals) -> emission line contribution
                  flux density of continnum integrated (10A intervals) -> continuum contribution
                  sum of the two = total flux in r-band
               
               '''
               int_flux_r_band = sum(f_lambda_over_r_band)*10 + sum(f_lambda_of_EL)*0.01
            
            else: int_flux_r_band = sum(f_lambda_over_r_band)*10
            
            int_flux_ref = sum(ref_spectrum)*10            ## interval of wavelength for integration = 10 A
            
            f_nu = int_flux_r_band/int_flux_ref
            f_nu_at_r[i] = round(f_nu,4)
            
            if i%(1e3)==0 or i<=10:
               print('nb.assembleData(), %.3f seconds since r\' filter loaded' % (time.time()-t1))
               print('  i = '+str(i)+', redshift of galaxy is %.4f, f_nu = %.4f uJy' % (z[i],f_nu_at_r[i]))
            
            if selectplots:
               if np.abs(zinf[i]-1.9)<=(1.2e-4) or \
                  np.abs(zinf[i]-1.95)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.0)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.05)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.1)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.15)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.2)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.25)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.3)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.35)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.4)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.45)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.5)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.55)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.6)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.65)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.7)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.75)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.8)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.85)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.9)<=(1.2e-4) or \
                  np.abs(zinf[i]-2.95)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.0)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.05)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.1)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.15)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.2)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.25)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.3)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.35)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.4)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.45)<=(1.2e-4) or \
                  np.abs(zinf[i]-3.5)<=(1.2e-4) or \
                  (ewinf[i] > 20 and objtype[i] == 'OII'):
                  
                  if lambda_EL[i] >= 5305 and lambda_EL[i] <= 7185:
                     f_lambda_with_EL = np.interp(wl_with_EL,wl_over_r_band,f_lambda_over_r_band)      ## initialize, without emission line
                     f_lambda_with_EL += f_lambda_of_EL
                     f_nu_with_EL = np.interp(wl_with_EL,wl_over_r_band,f_nu_over_r_band)
                     f_nu_with_EL += f_nu_of_EL
                  
                  plt.close()
                  if objtype[i] == 'LAE': linecolor,text = 'red','LAE spectrum'
                  else: linecolor,text = 'blue','[O II] emitter'

                  plt.scatter(wl_over_r_band,f_nu_over_r_band,s=2,lw=0,c=linecolor)
                  if lambda_EL[i] >= 5305 and lambda_EL[i] <= 7185:
                     plt.scatter(wl_with_EL,f_nu_with_EL,s=0.5,lw=0,c=linecolor)
                     plt.ylim(0,1.1*max(max(f_nu_with_EL),max(f_nu_over_r_band)))
                  else: plt.ylim(0,1.1*max(f_nu_over_r_band))
                  plt.xlabel(r'$\lambda$'+' ($\mathrm{\AA}$)')
                  plt.ylabel('$f$'+r'$_{\nu}$'+' $(\mu \mathrm{Jy})$')
                  plt.grid()
                  plt.title('simulated $z$=%.5f '%(z[i])+text+' convolved with $r\'$ filter, '+r'$f_{\nu}$'+' $=$ %.3f'%(f_nu)+' $\mu \mathrm{Jy}$')
                  plt.savefig(str(runname)+'_r-band_spectrum_%08d' % int(c00[i]) +'_f_nu.pdf')
                  plt.close()

                  plt.scatter(wl_over_r_band,1e17*(f_lambda_over_r_band),s=2,lw=0,c=linecolor)
                  if lambda_EL[i] >= 5305 and lambda_EL[i] <= 7185:
                     plt.scatter(wl_with_EL,1e17*(f_lambda_with_EL),s=0.5,lw=0,c=linecolor)
                     plt.ylim(0,1.1e17*max(max(f_lambda_with_EL),max(f_lambda_over_r_band)))
                  else: plt.ylim(0,1.1e17*max(f_lambda_over_r_band))
                  plt.xlabel(r'$\lambda$'+' ($\mathrm{\AA}$)')
                  plt.ylabel('$f$'+r'$_{\lambda}$'+' $(10^{-17} \mathrm{erg/cm^{2}/s/\AA})$')
                  plt.grid()
                  ewobs = (1e6)*(1e23)*(true_em_line_flux[i])/(f_nu)*((lambda_EL[i])**2)/(2.997925e18)
                  plt.title('simulated $z$=%.5f '%(z[i])+text+' convolved with $r\'$ filter, $EW$'+r'$_{\mathrm{obs}}$'+' $=$ %.1f'%(ewobs)+' $\mathrm{\AA}$')
                  plt.savefig(str(runname)+'_r-band_spectrum_%08d' % int(c00[i]) +'_f_lambda.pdf')
                  plt.close()

            if i%(1e3)==0:
               gc.collect()

         c09 = np.array(f_nu_at_r)
         c07 = (1e6)*(1e23)*(true_em_line_flux)/(f_nu_at_r)*((lambda_EL)**2)/(2.997925e18)      ## recalculate ew_obs based on f_nu at r-band


      else:
         print('****** assuming all galaxies have flat spectra in f_nu ******')


   true_cont_flux_dens = np.copy(c09)            ## f_nu, extrapolated to imaging survey bandpass, noiseless
   true_ew_obs = np.copy(c07)                     ## EW_obs = (line flux)/(f_nu) * (lambda^2)/c
   true_ew_inf = np.copy(c07/(1.+c10))            ## W_Lya = EW_obs / (1+z_Lya)
   for i in range(len(z)):
      if c09[i] > 0:
         c14[i] = 23.9-2.5*(np.log10(c09[i]))
      else:
         c14[i] = 999
   true_AB_cont_mag = np.copy(c14)


   '''
      noise is added for continuum flux and propagated to equivalent width
      c09 (f_nu) has noise added
      c14 (AB_mag) f_nu is converted to continuum magnitude
      c07 (ew_obs) is recalculated to include noise, line flux (noisified) / f_nu (noisified)
      c11 (ew_inf) is then calculated based on this 'noisefied' EW

      c06 (ew_rest), simulated by sampling the exponential distribution with the e-fold specified for observed wavelength, is not used in the Bayesian method
      
   '''
   
   ## (10/6/14): moved 'lim' variable for 5 sigma survey limit to method input
   '''
   ## five sigma AB continuum sensitivity is 25.1 at all wavelengths
   lim = 25.1

   ## comment out next line for assumption of equal depth in equal time
   #if isb == 'r': lim -= 0.25         ## assuming equal time and equal seeing, r-band 0.25 less deep (see EG emails on 8/29/14 and 9/11/14)
   if isb == 'r': lim -= 2.            ## debug: test what happens if r-band were 2 mags shallower
   
   '''
   
   five_sigma_cont = 10**(-0.4*(lim-23.9))                     ## uJy
   sigma_f_nu = 0.2 * five_sigma_cont
   true_s_to_n = true_cont_flux_dens / sigma_f_nu

   if True:      ### imaging surveys are independent realizations
      rd_f_nu = np.random.rand(len(z))
      data = open(str(rstrpath)+'rd_f_nu_'+str(runname)+'.dat','w')
      data.write('# \n')
      data.write('# Array of random numbers \n')
      data.write('#   * used to simulate measurement noise for f_nu \n')
      data.write('#   * observable sample (after applying line flux limit) \n')
      if scale == 0.1:    data.write('#   * one-tenth scale \n')
      elif scale == 0.25: data.write('#   * one-quarter scale \n')
      elif scale == 1.:   data.write('#   * full scale \n')
      else: data.write('#   * %.2f scale \n'%scale)
      data.write('#   *   '+str(int(cLAE))+' LAEs \n')
      data.write('#   *   '+str(int(cOII))+' [O II] emitters \n')
      data.write('# \n')
      data.write('# Column 1:  index number \n')
      data.write('# Column 2:  random number \n')
      data.write('# \n')
      data.write('# \n')
      for i in range(len(z)): data.write(str(c00[i])+'\t'+str(rd_f_nu[i])+'\n')
      data.write('# \n')
      data.close()

   else:
      rd_f_nu = []
      if runname[:6] == '140724': data = open(str(rstrpath)+'rd_f_nu_140724_r1102_24.47.dat','r')
      elif runname[:6] == '141025': data = open(str(rstrpath)+'rd_f_nu_141025_r1103_24.47.dat','r')
      
      for line in data.readlines():
         if not line.startswith('#'):
            a = line.split()
            index_rd.append(float(a[0]))
            rd_f_nu.append(float(a[1]))
      data.close()

      print('***')
      print('*** nb.assembleData(): check list match at import of vector of random numbers for f_nu measurement noise ***')
      print(len(rd_f_nu), len(c00))
      print(len(rd_f_nu) == len(c00))
      print(np.array(rd_f_nu) == c00)
      print('***')
      print('***')

   t1 = time.time()
   sigma_ew_obs = [0.]*len(z)
   for i in range(len(z)):            ## for each simulated object
      ## replaced; save vector of random values (10-22-14)
      #c09[i] += random.gauss(0,sigma_f_nu)         ## add noise to f_nu
      c09[i] = round(c09[i] + ncdfinv(rd_f_nu[i],0,sigma_f_nu), 4)
      
      ## ew_obs recalculated to include noise in line flux and f_nu
      
      '''
      ## allow negative noisified equivalent width (10/14/14)
      c07[i] = (1e6)*(1e23)*(c08[i])/(c09[i])*((c05[i])**2)/(2.997925e18)

      '''
      
      if c09[i] <= 0:               ## if noisified f_nu < 0
         #c09[i] = 0                  ## leave continuum flux as negative (05-16-14)
                                    ## ew_obs randomly drawn from a gaussian centered at 5000 angstrom (arbitrary); previously set equal to 10000 angstrom
         c07[i] = 1e4
         #while c07[i] <= 0: c07[i] = random.gauss(5000,1000)

      else:
         c07[i] = round((1e6)*(1e23)*(c08[i])/(c09[i])*((c05[i])**2)/(2.997925e18), 4)      ## ew_obs recalculated to include noise
      
      sigma_ew_obs[i] = round(true_ew_obs[i]*np.sqrt((sigma_line_flux[i]/true_em_line_flux[i])**2+(sigma_f_nu/true_cont_flux_dens[i])**2), 4)

      if i%100000 == 0: print('nb.assembleData(), '+str(i+1)+'st loop to add noise to f_nu and EW: %.2f seconds' % (time.time()-t1))

   rd_f_nu,index_rd = [],[]         ## clear arrays of random numbers and object indices from memory


   ## recalculate signal-to-noise
   c12 = c09/sigma_f_nu

   ## recalculate AB magnitude of continuum flux
   c14 = []
   for i in range(len(z)):
      if c09[i] > 0:
         c14.append(23.9-2.5*(np.log10(c09[i])))
      else:
         c14.append(999)

   c14 = np.array(c14)


   '''
      where is the regime in which noise in background dominated?
   
   '''


   ## inferred Lyman-alpha equivalent width [angstroms]
   c11 = c07/(1.+c10)         ## calculated based on noisified ew_obs (c07)

   '''
      inferred EW calculated by 'de-redshifting' the observed EW using the Lyman-alpha redshift, i.e., as if the object were an LAE
      
   '''


   lineFlux,f_nu,ew_obs,ew_inf,cont_mag = c08,c09,c07,c11,c14
   
   print('')
   print('***')
   print('*********************************************')
   print('*********************************************')
   print('***        run name: '+str(runname))
   print('***    imaging band: '+str(isb))
   print('***   5 sigma depth: '+str(lim))
   print('*********************************************')
   print('***')
   print('*** ****** LAEs ******')
   print('***')
   print('*** median line flux (noiseless): '+str(np.median(true_em_line_flux[:cLAE]))+' erg/cm^2/s')
   print('*** median line flux (noisified): '+str(np.median(lineFlux[:cLAE]))+' erg/cm^2/s')
   print('***')
   print('*** median f_nu (noiseless): '+str(np.median(true_cont_flux_dens[:cLAE]))+' uJy')
   print('*** median f_nu (noisified): '+str(np.median(f_nu[:cLAE]))+' uJy')
   print('***')
   print('*** median continuum magnitude (noiseless): '+str(np.median(true_AB_cont_mag[:cLAE])))
   print('*** median continuum magnitude (noisified): '+str(np.median(cont_mag[:cLAE])))
   print('***')
   print('*** median photometric EW, observed frame (noiseless): '+str(np.median(true_ew_obs[:cLAE]))+' angstroms')
   print('*** median photometric EW, observed frame (noisified): '+str(np.median(ew_obs[:cLAE]))+' angstroms')
   print('***')
   print('*** median photometric EW, inferred Lyman-alpha (noiseless): '+str(np.median(true_ew_inf[:cLAE]))+' angstroms')
   print('*** median photometric EW, inferred Lyman-alpha (noisified): '+str(np.median(ew_inf[:cLAE]))+' angstroms')
   print('***')
   print('***')
   print('*********************************************')
   print('*********************************************')
   print('***        run name: '+str(runname))
   print('***    imaging band: '+str(isb))
   print('***   5 sigma depth: '+str(lim))
   print('*********************************************')
   print('***')
   print('*** ****** [O II] emitters ******')
   print('***')
   print('*** median line flux (noiseless): '+str(np.median(true_em_line_flux[cLAE:]))+' erg/cm^2/s')
   print('*** median line flux (noisified): '+str(np.median(lineFlux[cLAE:]))+' erg/cm^2/s')
   print('***')
   print('*** median f_nu (noiseless): '+str(np.median(true_cont_flux_dens[cLAE:]))+' uJy')
   print('*** median f_nu (noisified): '+str(np.median(f_nu[cLAE:]))+' uJy')
   print('***')
   print('*** median continuum magnitude (noiseless): '+str(np.median(true_AB_cont_mag[cLAE:])))
   print('*** median continuum magnitude (noisified): '+str(np.median(cont_mag[cLAE:])))
   print('***')
   print('*** median photometric EW, observed frame (noiseless): '+str(np.median(true_ew_obs[cLAE:]))+' angstroms')
   print('*** median photometric EW, observed frame (noisified): '+str(np.median(ew_obs[cLAE:]))+' angstroms')
   print('***')
   print('*** median photometric EW, inferred Lyman-alpha (noiseless): '+str(np.median(true_ew_inf[cLAE:]))+' angstroms')
   print('*** median photometric EW, inferred Lyman-alpha (noisified): '+str(np.median(ew_inf[cLAE:]))+' angstroms')
   print('***')
   
   
   cLAEinBox_noiseless,cOIIinBox_noiseless,cLAEinBox_noisified,cOIIinBox_noisified = 0,0,0,0
   for i in range(len(z)):
      if (objtype[i] == 'LAE') and (true_ew_inf[i] > 5) and (true_ew_inf[i] <= 20) and (true_AB_cont_mag[i] > 20) and (true_AB_cont_mag[i] < 22):
         cLAEinBox_noiseless += 1
      if (objtype[i] == 'OII') and (true_ew_inf[i] > 5) and (true_ew_inf[i] <= 20) and (true_AB_cont_mag[i] > 20) and (true_AB_cont_mag[i] < 22):
         cOIIinBox_noiseless += 1
      if (objtype[i] == 'LAE') and (ew_inf[i] > 5) and (ew_inf[i] <= 20) and (cont_mag[i] > 20) and (cont_mag[i] < 22):
         cLAEinBox_noisified += 1
      if (objtype[i] == 'OII') and (ew_inf[i] > 5) and (ew_inf[i] <= 20) and (cont_mag[i] > 20) and (cont_mag[i] < 22):
         cOIIinBox_noisified += 1

   print('***')
   print('*********************************************')
   print('*********************************************')
   print('***        run name: '+str(runname))
   print('***    imaging band: '+str(isb))
   print('***   5 sigma depth: '+str(lim))
   print('*********************************************')
   print('***')
   print('*********************************************')
   print('*************** galaxy count ****************')
   print('*********************************************')
   print('*** simulated sample ************************')
   print('***    '+str(int(lenx/scale))+' LAEs')
   print('***    '+str(int(leny/scale))+' [O II] emitters')
   print('*********************************************')
   print('*********************************************')
   print('*** observable sample ***********************')
   print('***    '+str(int(cLAE/scale))+' LAEs')
   print('***    '+str(int(cOII/scale))+' [O II] emitters')
   print('*********************************************')
   print('*********************************************')
   print('*** observable sample in box (noiseless) ****')
   print('***    '+str(int(cLAEinBox_noiseless/scale))+' LAEs')
   print('***    '+str(int(cOIIinBox_noiseless/scale))+' [O II] emitters')
   print('*********************************************')
   print('*********************************************')
   print('*** observable sample in box (noisified) ****')
   print('***    '+str(int(cLAEinBox_noisified/scale))+' LAEs')
   print('***    '+str(int(cOIIinBox_noisified/scale))+' [O II] emitters')
   print('*********************************************')
   print('*********************************************')
   print('***')
   print('***')
         
   
   '''
      add noise to color
      
      EG (05-19-14 email): So for our case where we assume a single continuum magnitude value for all filters for a given object, you get dmag by comparing that to the 5sigma_limiting magnitude assumed for the imaging survey using the above formula and then multiply by sqrt(2) to get dcolor.   We are being aggressive in assuming that the imaging survey reaches the same limit (roughly 25.1) in all filters, but we can improve that in the future with a trade study that properly assumes fixed observing time and decides how many (and which) bands to split it between.
   
   '''

   five_sigma_lim = lim
   mag = true_AB_cont_mag

   #rd_g_minus_r = np.random.rand(len(z))
   
   t1 = time.time()
   sigma_g_minus_r = [0.]*len(z)
   for i in range(len(z)):
      dmag = -0.5/np.log(10.0) * 10**(-0.4*(five_sigma_lim-mag[i]))
      ## to be ; save vector of random values (10-22-14)
      #c15[i] += ncdfinv(rd_g_minus_r[i],0,np.sqrt(2)*dmag)
      c15[i] += random.gauss(0, np.sqrt(2)*dmag)
      c16[i] += random.gauss(0, np.sqrt(2)*dmag)
      c17[i] += random.gauss(0, np.sqrt(2)*dmag)
      c18[i] += random.gauss(0, np.sqrt(2)*dmag)
      c19[i] += random.gauss(0, np.sqrt(2)*dmag)
      c20[i] += random.gauss(0, np.sqrt(2)*dmag)
      sigma_g_minus_r[i] = np.abs(np.sqrt(2)*dmag)
   
      if i%100000 == 0: print('nb.assembleData(), '+str(i+1)+'st loop to add noise to colors: %.2f seconds' % (time.time()-t1))
   
   '''
      AL (07-25-14): [O II] emitters are brighter in the continua (smaller magnitude), and so the value of the Gaussian sigma is smaller and so color is less noisy. Does this make sense? Color is difference in magnitude, which is ratio of fluxes in the specified bands.
      
   '''

   sigma_f_nu = sigma_f_nu * np.ones(len(z))
   
   print('nb.assembleData() required %.2f seconds wall time' % (time.time()-t0))
   print('')
   print('###')
   print('### nb.assembleData('+runname+', genrand='+str(genrand)+') finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('###')
   print('###')
   print('')



def prob_ratio(run,isb,which_color,scale,sky_area,cosmo,LAE_priors,EW_case):
   t0 = time.time()
   
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global true_ew_inf,true_ew_obs,true_em_line_flux,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n
   global true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global f_nu_cont_slope
   global sigma_line_flux,sigma_f_nu,sigma_ew_obs,sigma_g_minus_r
   global addl_lines
   
   wl_obs = c05
   lineFlux = c08
   ew_obs = c07
   zinf = c10
   ewinf = c11
   objtype = c13
   g_minus_r = c15
   g_minus_i = c16
   g_minus_z = c17
   r_minus_i = c18
   r_minus_z = c19
   i_minus_z = c20
   
   if len(EW_case) >= 9 and EW_case[len(EW_case)-9:] == 'lognormal':
      z_OII, W_0, sigma = fln.lognorm_params()         ### three lists
   else:
      z_OII, W_0, sigma = [],[],[]
   
   out_file = open(path+str(run)+'_select_objects.dat','w')
   a1,b1,c1,d1,e1,f1,g1 = [],[],[],[],[],[],[]
   
   for i in range(len(wl_obs)):
      
      ###### (05-28-15) added spectroscopic depth as function of survey area in fixed observing time
      if which_color == 'g-r':
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],g_minus_r[i],which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      elif which_color == 'g-i':
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],g_minus_i[i],which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      elif which_color == 'g-z':
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],g_minus_z[i],which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      elif which_color == 'r-i':
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],r_minus_i[i],which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      elif which_color == 'r-z':
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],r_minus_z[i],which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      elif which_color == 'i-z':
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],i_minus_z[i],which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      elif which_color == 'no_imaging':   ### (08-24-15)
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],'','',which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      else:
         ratLAE_, plgd_, pogd_ = b.prob_ratio(wl_obs[i],lineFlux[i],ew_obs[i],'',which_color,addl_lines[i],sky_area,cosmo,LAE_priors,EW_case,z_OII,W_0,sigma)
      
      a1.append(ratLAE_)
      f1.append(plgd_)
      g1.append(pogd_)

      if np.abs(zinf[i]-1.9)<=(1.2e-4) or \
         np.abs(zinf[i]-1.95)<=(1.2e-4) or \
         np.abs(zinf[i]-2.0)<=(1.2e-4) or \
         np.abs(zinf[i]-2.05)<=(1.2e-4) or \
         np.abs(zinf[i]-2.1)<=(1.2e-4) or \
         np.abs(zinf[i]-2.15)<=(1.2e-4) or \
         np.abs(zinf[i]-2.2)<=(1.2e-4) or \
         np.abs(zinf[i]-2.25)<=(1.2e-4) or \
         np.abs(zinf[i]-2.3)<=(1.2e-4) or \
         np.abs(zinf[i]-2.35)<=(1.2e-4) or \
         np.abs(zinf[i]-2.4)<=(1.2e-4) or \
         np.abs(zinf[i]-2.45)<=(1.2e-4) or \
         np.abs(zinf[i]-2.5)<=(1.2e-4) or \
         np.abs(zinf[i]-2.55)<=(1.2e-4) or \
         np.abs(zinf[i]-2.6)<=(1.2e-4) or \
         np.abs(zinf[i]-2.65)<=(1.2e-4) or \
         np.abs(zinf[i]-2.7)<=(1.2e-4) or \
         np.abs(zinf[i]-2.75)<=(1.2e-4) or \
         np.abs(zinf[i]-2.8)<=(1.2e-4) or \
         np.abs(zinf[i]-2.85)<=(1.2e-4) or \
         np.abs(zinf[i]-2.9)<=(1.2e-4) or \
         np.abs(zinf[i]-2.95)<=(1.2e-4) or \
         np.abs(zinf[i]-3.0)<=(1.2e-4) or \
         np.abs(zinf[i]-3.05)<=(1.2e-4) or \
         np.abs(zinf[i]-3.1)<=(1.2e-4) or \
         np.abs(zinf[i]-3.15)<=(1.2e-4) or \
         np.abs(zinf[i]-3.2)<=(1.2e-4) or \
         np.abs(zinf[i]-3.25)<=(1.2e-4) or \
         np.abs(zinf[i]-3.3)<=(1.2e-4) or \
         np.abs(zinf[i]-3.35)<=(1.2e-4) or \
         np.abs(zinf[i]-3.4)<=(1.2e-4) or \
         np.abs(zinf[i]-3.45)<=(1.2e-4) or \
         np.abs(zinf[i]-3.5)<=(1.2e-4) or \
         (ewinf[i] > 20 and objtype[i] == 'OII'):
         
         out_file.write('\n')
         out_file.write('simulation run id: '+str(run)+'\n')
         out_file.write('object index: '+str(c00[i])+'\n')
         out_file.write('\n')
         out_file.write('   object type:                    '+str(c13[i])+'\n')
         out_file.write('   emission line wavelength:       '+'%.3f'%(c05[i])+' angstroms\n')
         out_file.write('   redshift:                       '+'%.6f'%(c01[i])+'\n')
         out_file.write('   redshift, inferred Lyman-alpha: '+'%.6f'%(c10[i])+'\n')
         out_file.write('   luminosity distance:            '+'%.2f'%(c04[i])+' Mpc\n')
         out_file.write('\n')
         out_file.write('   emission line luminosity:               '+'%.5e'%(c03[i])+' erg/s\n')
         out_file.write('   emission line flux (simulated):         '+'%.5e'%(true_em_line_flux[i])+' erg/s/cm^2\n')
         out_file.write('   emission line flux (noisified):         '+'%.5e'%(c08[i])+' erg/s/cm^2\n')
         out_file.write('   HETDEX line flux sensitivity limit:     '+'%.5e'%(lineSens(c05[i])*np.sqrt(sky_area/300.))+' erg/s/cm^2\n')
         out_file.write('   sigma uncertainty, emission line flux:  '+'%.5e'%(sigma_line_flux[i])+' erg/s/cm^2\n')
         out_file.write('   fractional error, emission line flux:   '+'%.5f'%(sigma_line_flux[i]/true_em_line_flux[i])+'\n')
         out_file.write('   signal-to-noise, line flux (simulated): '+'%.5f'%(true_em_line_flux[i]/sigma_line_flux[i])+'\n')
         out_file.write('-> signal-to-noise, line flux (noisified): '+'%.5f'%(c08[i]/sigma_line_flux[i])+'\n')
         out_file.write('\n')
         out_file.write('   EW, rest frame (flat spectrum):           '+'%.3f'%(true_ew_rest_fs[i])+' angstroms\n')
         out_file.write('   EW, observed frame (flat spectrum):       '+'%.3f'%(true_ew_obs_fs[i])+' angstroms\n')
         out_file.write('   EW, inferred Lyman-alpha (flat spectrum): '+'%.3f'%(true_ew_inf_fs[i])+' angstroms\n')
         out_file.write('\n')
         out_file.write('   continuum flux density, f_nu (flat spectrum): '+'%.3f'%(true_cont_flux_dens_fs[i])+' uJy\n')
         out_file.write('   AB continuum magnitude (flat spectrum):       '+'%.3f'%(true_AB_cont_mag_fs[i])+'\n')
         out_file.write('\n')
         out_file.write('   g-r color (simulated):        '+'%.4f'%(true_g_minus_r[i])+'\n')
         out_file.write('   g-r color (noisified):        '+'%.4f'%(c15[i])+'\n')
         out_file.write('   sigma uncertainty, g-r color: '+'%.4f'%(sigma_g_minus_r[i])+'\n')
         out_file.write('\n')
         out_file.write('   power law slope in f_nu:     '+'%.4f'%(f_nu_cont_slope[i])+'\n')
         out_file.write('   power law slope in f_lambda: '+'%.4f'%(f_nu_cont_slope[i]-2)+'\n')
         out_file.write('\n')
         out_file.write('   imaging survey band:                                 '+str(isb)+'\'\n')
         out_file.write('     photometric EW, observed frame (simulated):       '+'%.3f'%(true_ew_obs[i])+' angstroms\n')
         out_file.write('     photometric EW, observed frame (noisified):       '+'%.3f'%(c07[i])+' angstroms\n')
         out_file.write('     sigma uncertainty, photometric EW:                '+'%.3f'%(sigma_ew_obs[i])+' angstroms\n')
         out_file.write('     fractional error, photometric EW:                 '+'%.5f'%(sigma_ew_obs[i]/true_ew_obs[i])+'\n')
         out_file.write('     signal-to-noise, photometric EW (simulated):      '+'%.5f'%(true_ew_obs[i]/sigma_ew_obs[i])+'\n')
         out_file.write('  -> signal-to-noise, photometric EW (noisified):      '+'%.5f'%(c07[i]/sigma_ew_obs[i])+'\n')
         out_file.write('     photometric EW, inferred Lyman-alpha (simulated): '+'%.3f'%(true_ew_inf[i])+' angstroms\n')
         out_file.write('     photometric EW, inferred Lyman-alpha (noisified): '+'%.3f'%(c11[i])+' angstroms\n')
         out_file.write('\n')
         out_file.write('   imaging survey band:                         '+str(isb)+'\'\n')
         out_file.write('     continuum flux density, f_nu (simulated): '+'%.3f'%(true_cont_flux_dens[i])+' uJy\n')
         out_file.write('     continuum flux density, f_nu (noisified): '+'%.3f'%(c09[i])+' uJy\n')
         out_file.write('     sigma uncertainty, f_nu:                  '+'%.5f'%(sigma_f_nu[i])+' uJy\n')
         out_file.write('     fractional error, f_nu:                   '+'%.5f'%(sigma_f_nu[i]/true_cont_flux_dens[i])+'\n')
         out_file.write('     signal-to-noise, f_nu (simulated):        '+str(true_s_to_n[i])+'\n')
         out_file.write('  -> signal-to-noise, f_nu (noisified):        '+str(c12[i])+'\n')
         out_file.write('     AB continuum magnitude (simulated):       '+'%.3f'%(true_AB_cont_mag[i])+'\n')
         out_file.write('     AB continuum magnitude (noisified):       '+'%.3f'%(c14[i])+'\n')
         out_file.write('\n')
         out_file.write('  probability ratio: '+'%.5e'%(float(ratLAE_))+'\n')
         out_file.write('  p(LAE):            '+'%.7f'%(float(ratLAE_)/(1+ratLAE_))+'\n')
         out_file.write('\n')
      
      if i%(1e4)==0: print('nb.prob_ratio(), '+str(i+1)+'st loop: %.2f seconds' % (time.time()-t0))
      
   out_file.close()
   
   global ratioLAE_, ratioOII_, prob_data_given_LAE, prob_data_given_OII, prob_LAE, prob_LAE_given_data, prob_OII_given_data
   ratioLAE_ = a1
   prob_LAE_given_data = f1
   prob_OII_given_data = g1
   
   print('nb.prob_ratio() finished in %.0f seconds' % (time.time()-t0))



def get_data_from_imaging(isb, g_dep, r_dep, sc):
   reload(im)
   
   global scale
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global true_ew_inf,true_ew_obs,true_em_line_flux,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n
   global true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global sigma_line_flux, sigma_f_nu, sigma_ew_obs, sigma_g_minus_r
   global f_nu_cont_slope
   global ratioLAE_
   
   scale                  = sc
   c00                    = im.c00                     ## object index
   c13                    = im.c13                     ## object type
   c10                    = im.c10                    ## redshift
   c05                    = im.c05                     ## wavelength observed
   c08                    = im.c08                     ## emission line flux
   c15                    = im.noisified_g_minus_r      ## g-r calculated from simulated g' and r'
   c16                    = im.c16                     ## g-i simulated from distribution
   c17                    = im.c17                     ## g-z simulated from distribution
   c18                    = im.c18                     ## r-i simulated from distribution
   c19                    = im.c19                     ## r-z simulated from distribution
   c20                    = im.c20                     ## i-z simulated from distribution
   
   true_em_line_flux      = im.true_em_line_flux
   true_cont_flux_dens_fs = im.true_cont_flux_dens_fs
   true_ew_obs_fs         = im.true_ew_obs_fs
   true_g_minus_r         = im.noiseless_g_minus_r      ## g-r calculated from simulated g' and r'
   true_g_minus_i         = im.true_g_minus_i
   true_g_minus_z         = im.true_g_minus_z
   true_r_minus_i         = im.true_r_minus_i
   true_r_minus_z         = im.true_r_minus_z
   true_i_minus_z         = im.true_i_minus_z
   sigma_line_flux        = im.sigma_line_flux
   sigma_g_minus_r        = im.calculated_dcolor
   f_nu_cont_slope        = im.f_nu_cont_slope         ## power-law slope in f_nu corresponding to g-r simulated from distribution
      
   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   if isb == 'g\'' or isb == 'g':
      sigma_f_nu            = im.g_sigma_f_nu
      sigma_ew_obs          = im.g_sigma_ew_obs
      true_ew_obs           = im.g_true_ew_obs
      true_ew_inf           = im.g_true_ew_inf
      true_cont_flux_dens   = im.g_true_cont_flux_dens
      true_s_to_n           = im.g_true_s_to_n
      true_AB_cont_mag      = im.g_true_AB_cont_mag
      c07                   = im.g_c07                  ## photometric equivalent width observed in g' band
      c09                   = im.g_c09                  ## f_nu in g' band
      c11                   = im.g_c11                  ## inferred Lyman-alpha equivalent width in g' band
      c12                   = im.g_c12                  ## signal-to-noise of g' band flux
      c14                   = im.g_c14                  ## continuum magnitude in g' band
      ratioLAE_             = im.prob_LAE_over_prob_OII[0]
   
   elif isb == 'r\'' or isb == 'r':
      sigma_f_nu            = im.r_sigma_f_nu
      sigma_ew_obs          = im.r_sigma_ew_obs
      true_ew_obs           = im.r_true_ew_obs
      true_ew_inf           = im.r_true_ew_inf
      true_cont_flux_dens   = im.r_true_cont_flux_dens
      true_s_to_n           = im.r_true_s_to_n
      true_AB_cont_mag      = im.r_true_AB_cont_mag
      c07                   = im.r_c07            ## photometric equivalent width observed in r' band
      c09                   = im.r_c09            ## f_nu in r' band
      c11                   = im.r_c11            ## inferred Lyman-alpha equivalent width in r' band
      c12                   = im.r_c12            ## signal-to-noise of r' band flux
      c14                   = im.r_c14            ## continuum magnitude in r' band
      ratioLAE_             = im.prob_LAE_over_prob_OII[1]


   elif isb == 'g-r' or isb == 'power-law-interpolated to lambda_EL':
      sigma_f_nu_g = 0.2* 10**(-0.4*(g_dep-23.9))
      sigma_f_nu_r = 0.2* 10**(-0.4*(r_dep-23.9))
      
      true_ew_obs           = im.phot_EW_obs_at_EL[0]
      true_ew_inf           = true_ew_obs/(1+c10)
      true_cont_flux_dens   = im.f_nu_at_EL[0]
      
      sigma_f_nu            = true_cont_flux_dens * np.sqrt( (sigma_f_nu_g/im.g_true_cont_flux_dens)**2 +(sigma_f_nu_r/im.r_true_cont_flux_dens)**2 )
      sigma_ew_obs          = true_ew_obs * np.sqrt( (sigma_line_flux/true_em_line_flux)**2 +(sigma_f_nu/true_cont_flux_dens)**2 )
      
      true_s_to_n           = true_cont_flux_dens / sigma_f_nu
      true_AB_cont_mag      = 23.9-2.5*(np.log10(true_cont_flux_dens))
      c07                   = im.phot_EW_obs_at_EL[1]                     ## photometric equivalent width observed at lambda_EL
      c09                   = im.f_nu_at_EL[1]                           ## power-law-interpolated f_nu at lambda_EL
      c11                   = c07/(1+c10)                                 ## inferred Lyman-alpha equivalent width at lambda_EL
      c12                   = c09 / sigma_f_nu                           ## signal-to-noise
      c14                   = 23.9-2.5*(np.log10(c09))                  ## continuum magnitude at lambda_EL
      ratioLAE_             = im.prob_LAE_over_prob_OII[2]

      for i in range(len(c14)):
         if c07[i] == 1e4:
            c14[i] = 999



def contam_and_incomp(run,low,spec,high,plots,ver_sigma_dA):
   t0 = time.time()
   
   global scale
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global ratioLAE_

   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   wl_obs = c05
   wlbin = []
   for i in range(len(c13)):
      if wl_obs[i] < 3700: wlbin.append(0)
      elif wl_obs[i] >= 3700 and wl_obs[i] < 3900: wlbin.append(1)         ## no [O II] interlopers at wl < 3727
      elif wl_obs[i] >= 3900 and wl_obs[i] < 4100: wlbin.append(2)
      elif wl_obs[i] >= 4100 and wl_obs[i] < 4300: wlbin.append(3)
      elif wl_obs[i] >= 4300 and wl_obs[i] < 4500: wlbin.append(4)
      elif wl_obs[i] >= 4500 and wl_obs[i] < 4700: wlbin.append(5)
      elif wl_obs[i] >= 4700 and wl_obs[i] < 4900: wlbin.append(6)
      elif wl_obs[i] >= 4900 and wl_obs[i] < 5100: wlbin.append(7)
      elif wl_obs[i] >= 5100 and wl_obs[i] < 5300: wlbin.append(8)
      elif wl_obs[i] >= 5300: wlbin.append(9)
   
   global trueLAEcount, classLAEcount
   trueLAEcount = [0]*10
   classLAEcount = [0]*10
   contamCount = [0]*10
   incompCount = [0]*10
   
   for i in range(len(c13)):
      if objtype[i] == 'LAE': trueLAEcount[wlbin[i]] += 1
      if ratioLAE_[i] > spec: classLAEcount[wlbin[i]] += 1
      if ratioLAE_[i] > spec and objtype[i] == 'OII': contamCount[wlbin[i]] += 1
      if ratioLAE_[i] <= spec and objtype[i] == 'LAE': incompCount[wlbin[i]] += 1

   contamFrac,contamFrac_low,contamFrac_high = [],[],[]
   incompFrac,incompFrac_low,incompFrac_high = [],[],[]
   
   for i in range(max(wlbin)+1):
      
      if classLAEcount[i] == 0: contamFrac.append(0.)
      else: contamFrac.append(contamCount[i]/float(classLAEcount[i]))
      
      if trueLAEcount[i] == 0: incompFrac.append(0.)
      else: incompFrac.append(incompCount[i]/float(trueLAEcount[i]))
   
   contamFrac = np.array(contamFrac)
   incompFrac = np.array(incompFrac)

   if plots:
      plt.close()
      fig = plt.figure()
      ax1 = fig.add_subplot(111)
      ax2 = ax1.twiny()
      
      x = [3600,3800,4000,4200,4400,4600,4800,5000,5200,5400]
      x_ticks = [3500,3700,3900,4100,4300,4500,4700,4900,5100,5300,5500]
      
      plt.scatter(x,contamFrac,marker='o',c='blue',lw=0,s=20,label='contamination',rasterized=True)
      plt.scatter(x,incompFrac,marker='^',c='red',lw=0,s=20,label='incompleteness',rasterized=True)
      
      plt.errorbar(x,contamFrac,yerr=[contam_err_low,contam_err_high],fmt='o',lw=2,c='blue')
      plt.errorbar(x,incompFrac,yerr=[incomp_err_low,incomp_err_high],fmt='^',lw=1,c='red')
      
      plt.xlim(3450,5550)
      
      ax1.set_xticks(x_ticks)
      ax1.set_xlim([3450,5550])
      ax1.set_xlabel('observed wavelength ($\mathrm{\AA}$)')
      ax1.set_ylim([-0.01,0.45])
      ax1.set_ylabel('contamination fraction; incompleteness')
      
      z_tick_labels = []
      for i in range(11):
         z_tick_labels.append(z_LAE(3500+200*i))
      ax2.set_xticks(x_ticks)
      ax2.set_xticklabels(z_tick_labels)
      ax2.set_xlabel('Lyman-alpha redshift')

      plt.plot([3500,3500],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([3700,3700],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([3900,3900],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([4100,4100],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([4300,4300],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([4500,4500],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([4700,4700],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([4900,4900],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([5100,5100],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([5300,5300],[-0.1,1],'k-',ls='dotted',lw=0.25)
      plt.plot([5500,5500],[-0.1,1],'k-',ls='dotted',lw=0.25)
      
      plt.plot([3350,5650],[0.0,0.0],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.05,0.05],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.1,0.1],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.15,0.15],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.2,0.2],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.25,0.25],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.3,0.3],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.35,0.35],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.4,0.4],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.45,0.45],'k-',ls='dotted',lw=0.25)
      plt.plot([3350,5650],[0.5,0.5],'k-',ls='dotted',lw=0.25)


      legend = plt.legend(loc='upper left',scatterpoints=1,shadow=True)
      frame = legend.get_frame()
      frame.set_facecolor('0.95')
      for label in legend.get_texts(): label.set_fontsize('medium')
      for label in legend.get_lines(): label.set_linewidth(1)
      
      plt.savefig(str(run)+'-contam_and_incomp-spec'+str(round(spec,5))+'.pdf',dpi=200)
      plt.close()

   baserun = run[:len(run)-12]
   
   sigmaDA,sigmaDA_bin = [],[]
   prtContamFrac,prtClassLAEcount,prtIncompFrac,prtTrueLAEcount,prtCntLAErecovered = [0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]

   cntLAErecovered = []
   for i in range(max(wlbin)+1):
      cntLAErecovered.append(classLAEcount[i] - contamCount[i])

   for i in range(4):
      prtContamFrac[0] += contamFrac[i]*classLAEcount[i]
      prtClassLAEcount[0] += classLAEcount[i]
      prtIncompFrac[0] += incompFrac[i]*trueLAEcount[i]
      prtTrueLAEcount[0] += trueLAEcount[i]
      prtCntLAErecovered[0] += cntLAErecovered[i]
   #sigmaDA.append(np.sqrt(foo**-1))                           ### (11-26-14) Skype call w/ EG and VA: use sigma_bin formula for 1.9 < z < 2.5
   prtContamFrac[0] = prtContamFrac[0]/prtClassLAEcount[0]
   prtIncompFrac[0] = prtIncompFrac[0]/prtTrueLAEcount[0]
   sigmaDA.append(sda.sigma_dA(prtContamFrac[0],prtIncompFrac[0],prtCntLAErecovered[0],scale,0,ver_sigma_dA,baserun))
   #sigmaDA.append( np.sqrt( (prtContamFrac[0]/0.025)**2 + (270000*scale)/prtCntLAErecovered[0] ) )
   ### (04-22-15) replaced by previous line with option to use 'old' or 'new' formula

   for i in range(4,10):                                       ### (12-03-14) telecon with PSU and JJF: second bin is 2.5 < z < 3.5
      #bar += ((sigmaDA_bin[i])**2)**-1
      prtContamFrac[1] += contamFrac[i]*classLAEcount[i]
      prtClassLAEcount[1] += classLAEcount[i]
      prtIncompFrac[1] += incompFrac[i]*trueLAEcount[i]
      prtTrueLAEcount[1] += trueLAEcount[i]
      prtCntLAErecovered[1] += cntLAErecovered[i]
   #sigmaDA.append(np.sqrt(bar**-1))                           ### (11-26-14) Skype call w/ EG and VA: use sigma_bin formula for 1.9 < z < 2.5
   prtContamFrac[1] = prtContamFrac[1]/prtClassLAEcount[1]
   prtIncompFrac[1] = prtIncompFrac[1]/prtTrueLAEcount[1]
   sigmaDA.append(sda.sigma_dA(prtContamFrac[1],prtIncompFrac[1],prtCntLAErecovered[1],scale,1,ver_sigma_dA,baserun))
   #sigmaDA.append( np.sqrt( (prtContamFrac[1]/0.05)**2 + (360000*scale)/prtCntLAErecovered[1] ) )
   ### (02-27-15) Skype call with EG and VA: modified formula for high-z bin
   ### (04-22-15) replaced by previous line with option to use 'old' or 'new' formula

   for i in range(10):
      prtContamFrac[2] += contamFrac[i]*classLAEcount[i]
      prtClassLAEcount[2] += classLAEcount[i]
      prtIncompFrac[2] += incompFrac[i]*trueLAEcount[i]
      prtTrueLAEcount[2] += trueLAEcount[i]
      prtCntLAErecovered[2] += cntLAErecovered[i]
   prtContamFrac[2] = prtContamFrac[2]/prtClassLAEcount[2]
   prtIncompFrac[2] = prtIncompFrac[2]/prtTrueLAEcount[2]
   sigmaDA.append(sda.sigma_dA(prtContamFrac[2],prtIncompFrac[2],prtCntLAErecovered[2],scale,2,ver_sigma_dA,baserun))

   ### each output element is a length-3 list
   return np.array(sigmaDA), np.array(prtContamFrac), np.array(prtClassLAEcount)/scale, np.array(prtIncompFrac), np.array(prtTrueLAEcount)/scale, np.array(prtCntLAErecovered)/scale



def write_sim_data(run,band,depth,sc,sa):
   t0 = time.time()
   
   print('nb.write_sim_data() began at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')
   
   scale, sky_area = sc, sa
   
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global true_ew_inf,true_ew_obs,true_em_line_flux,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n
   global true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global f_nu_cont_slope
   
   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   out_file = open(str(path)+'LAE_simulation_flat_spectra_'+str(run)+'.dat','w')
   out_file.write('# \n')
   out_file.write('# Simulated HETDEX catalog of Lyman-alpha emitters \n')
   if scale == 0.1:  out_file.write('#   * one-tenth scale Monte Carlo simulation \n')
   elif scale == 0.25: out_file.write('#   * one-quarter scale Monte Carlo simulation \n')
   elif scale == 1.: out_file.write('#   * full scale Monte Carlo simulation \n')
   else: out_file.write('#   * %.2f scale Monte Carlo simulation \n'%scale)
   out_file.write('#   * line flux and EW simulated at emission line wavelength \n')
   out_file.write('#   * f_nu calculated assuming a flat spectrum \n')
   out_file.write('# \n')
   out_file.write('# Column 1:  object type \n')
   out_file.write('# Column 2:  wavelength of emission line (angstroms) \n')
   out_file.write('# Column 3:  emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 4:  continuum flux density (microjanskys) \n')
   out_file.write('# Column 5:  equivalent width observed (angstroms) \n')
   out_file.write('# Column 6:  g-r color index \n')
   out_file.write('# Column 7:  g-i color index \n')
   out_file.write('# Column 8:  g-z color index \n')
   out_file.write('# Column 9:  r-i color index \n')
   out_file.write('# Column 10: r-z color index \n')
   out_file.write('# Column 11: i-z color index \n')
   out_file.write('# Column 12: simulation index \n')
   out_file.write('# \n')
   out_file.write('# \n')
   
   for i in range(cLAE):
      out_file.write('%6s'%(str(c13[i])) \
                     +'%10s'%('%.2f'%(c05[i]))
                     +'%15s'%('%.4e'%(true_em_line_flux[i])) \
                     +'%13s'%('%.4f'%(true_cont_flux_dens_fs[i])) \
                     +'%13s'%('%.4f'%(true_ew_obs_fs[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_r[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_i_minus_z[i])) \
                     +'%10s'%(str(c00[i])) \
                     +'\n')
   out_file.close()


   out_file = open(str(path)+'LAE_simulation_before_noise_'+str(run)+'.dat','w')
   out_file.write('# \n')
   out_file.write('# Simulated HETDEX catalog of Lyman-alpha emitters \n')
   if scale == 0.1:    out_file.write('#   * one-tenth scale simulation \n')
   elif scale == 0.25: out_file.write('#   * one-quarter scale simulation \n')
   elif scale == 1.:   out_file.write('#   * full scale simulation \n')
   else: out_file.write('#   * %.2f scale Monte Carlo simulation \n'%scale)
   out_file.write('#   * Monte Carlo simulated quantities prior to addition of noise \n')
   out_file.write('#   *    imaging survey band: '+str(band)+'\n')
   out_file.write('# \n')
   out_file.write('# Column 1:  object type \n')
   out_file.write('# Column 2:  wavelength of emission line (angstroms) \n')
   out_file.write('# Column 3:  emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 4:  continuum flux density (microjanskys) \n')
   out_file.write('# Column 5:  equivalent width observed (angstroms) \n')
   out_file.write('# Column 6:  g-r color index \n')
   out_file.write('# Column 7:  g-i color index \n')
   out_file.write('# Column 8:  g-z color index \n')
   out_file.write('# Column 9:  r-i color index \n')
   out_file.write('# Column 10: r-z color index \n')
   out_file.write('# Column 11: i-z color index \n')
   out_file.write('# \n')
   out_file.write('# \n')

   for i in range(cLAE):
      out_file.write('%6s'%(str(c13[i])) \
                     +'%10s'%('%.2f'%(c05[i]))
                     +'%15s'%('%.4e'%(true_em_line_flux[i])) \
                     +'%13s'%('%.4f'%(true_cont_flux_dens[i])) \
                     +'%13s'%('%.4f'%(true_ew_obs[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_r[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_i_minus_z[i])) \
                     +'\n')
   out_file.close()


   out_file = open(str(path)+'LAE_simulation_with_noise_'+str(run)+'.dat','w')
   out_file.write('# \n')
   out_file.write('# Simulated HETDEX catalog of Lyman-alpha emitters \n')
   if scale == 0.1:    out_file.write('#   * one-tenth scale simulation \n')
   elif scale == 0.25: out_file.write('#   * one-quarter scale simulation \n')
   elif scale == 1.:   out_file.write('#   * full scale simulation \n')
   else: out_file.write('#   * %.2f scale Monte Carlo simulation \n'%scale)
   out_file.write('#   * Monte Carlo simulated quantities with Gaussian noise added \n')
   out_file.write('#   *    imaging survey band: '+str(band)+'\n')
   out_file.write('#   *    5 sigma depth (mag): '+str(depth)+'\n')
   out_file.write('# \n')
   out_file.write('# Column 1:  object type \n')
   out_file.write('# Column 2:  wavelength of emission line (angstroms) \n')
   out_file.write('# Column 3:  emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 4:  continuum flux density (microjanskys) \n')
   out_file.write('# Column 5:  equivalent width observed (angstroms) \n')
   out_file.write('# Column 6:  g-r color index \n')
   out_file.write('# Column 7:  g-i color index \n')
   out_file.write('# Column 8:  g-z color index \n')
   out_file.write('# Column 9:  r-i color index \n')
   out_file.write('# Column 10: r-z color index \n')
   out_file.write('# Column 11: i-z color index \n')
   out_file.write('# Column 12: sigma uncertainty, emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 13: sigma uncertainty, continuum flux density (microjanskys) \n')
   out_file.write('# Column 14: sigma uncertainty, equivalent width observed (angstroms) \n')
   out_file.write('# Column 15: sigma uncertainty, g-r color index \n')
   out_file.write('# Column 16: f_nu continuum slope  \n')
   out_file.write('# Column 17: [Ne III] 3869 flux (erg/cm^2/s) \n')
   out_file.write('# Column 18: sigma uncertainty, [Ne III] 3869 flux (erg/cm^2/s) \n')
   out_file.write('# Column 19: H-beta   4861 flux (erg/cm^2/s) \n')
   out_file.write('# Column 20: sigma uncertainty, H-beta   4861 flux (erg/cm^2/s) \n')
   out_file.write('# Column 21: [O III]  4959 flux (erg/cm^2/s) \n')
   out_file.write('# Column 22: sigma uncertainty, [O III]  4959 flux (erg/cm^2/s) \n')
   out_file.write('# Column 23: [O III]  5007 flux (erg/cm^2/s) \n')
   out_file.write('# Column 24: sigma uncertainty, [O III]  5007 flux (erg/cm^2/s) \n')
   out_file.write('# \n')
   out_file.write('# \n')
   
   if band == 'g':
      bandpass_min     = 386
      bandpass_max     = 576
      
   elif band == 'r':
      bandpass_min     = 531
      bandpass_max     = 719

   addl_em_lines  = ['[NeIII]','H_beta','[OIII]','[OIII]']
   addl_lambda_rf = np.array([3869.00, 4861.32, 4958.91, 5006.84])

   for i in range(cLAE):
      out_file.write('%6s'%(str(c13[i])) \
                     +'%10s'%('%.2f'%(c05[i]))
                     +'%15s'%('%.4e'%(c08[i])) \
                     +'%12s'%('%.3f'%(c09[i])) \
                     +'%12s'%('%.3f'%(c07[i])) \
                     +'%10s'%('%.3f'%(c15[i])) \
                     +'%10s'%('%.3f'%(c16[i])) \
                     +'%10s'%('%.3f'%(c17[i])) \
                     +'%10s'%('%.3f'%(c18[i])) \
                     +'%10s'%('%.3f'%(c19[i])) \
                     +'%10s'%('%.3f'%(c20[i])) \
                     +'%15s'%('%.4e'%(sigma_line_flux[i])) \
                     +'%12s'%('%.3f'%(sigma_f_nu[i])) \
                     +'%12s'%('%.3f'%(sigma_ew_obs[i])) \
                     +'%10s'%('%.3f'%(sigma_g_minus_r[i])) \
                     +'%10s'%('%.3f'%(f_nu_cont_slope[i])) \
                     )
   
      inf_OII_redshift = c05[i]/3727.45-1
      addl_lambda_ob = addl_lambda_rf * (1+inf_OII_redshift)
      for k in range(len(addl_em_lines)):
         if addl_lambda_ob[k] <= 5500.:      #10*(bandpass_max-0.5):
            sigma_addl_line = 0.2 * lineSens(addl_lambda_ob[k]) * np.sqrt(sky_area/300.)
            addl_flux = ncdfinv(random.random(),0.,sigma_addl_line)
            out_file.write('%15s'%('%.4e'%(addl_flux)))
            out_file.write('%15s'%('%.4e'%(sigma_addl_line)))
         else:
            out_file.write('%15s'%('0.'))
            out_file.write('%15s'%('0.'))

      out_file.write('\n')
   
   out_file.close()

   out_file = open(str(path)+'OII_simulation_flat_spectra_'+str(run)+'.dat','w')
   out_file.write('# \n')
   out_file.write('# Simulated HETDEX catalog of [O II] emitters \n')
   if scale == 0.1:  out_file.write('#   * one-tenth scale Monte Carlo simulation \n')
   elif scale == 0.25: out_file.write('#   * one-quarter scale Monte Carlo simulation \n')
   elif scale == 1.: out_file.write('#   * full scale Monte Carlo simulation \n')
   else: out_file.write('#   * %.2f scale Monte Carlo simulation \n'%scale)
   out_file.write('#   * line flux and EW simulated at emission line wavelength \n')
   out_file.write('#   * f_nu calculated assuming a flat spectrum \n')
   out_file.write('# \n')
   out_file.write('# Column 1:  object type \n')
   out_file.write('# Column 2:  wavelength of emission line (angstroms) \n')
   out_file.write('# Column 3:  emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 4:  continuum flux density (microjanskys) \n')
   out_file.write('# Column 5:  equivalent width observed (angstroms) \n')
   out_file.write('# Column 6:  g-r color index \n')
   out_file.write('# Column 7:  g-i color index \n')
   out_file.write('# Column 8:  g-z color index \n')
   out_file.write('# Column 9:  r-i color index \n')
   out_file.write('# Column 10: r-z color index \n')
   out_file.write('# Column 11: i-z color index \n')
   out_file.write('# Column 12: simulation index \n')
   out_file.write('# \n')
   out_file.write('# \n')
   
   for i in range(cLAE,cLAE+cOII):
      out_file.write('%6s'%(str(c13[i])) \
                     +'%10s'%('%.2f'%(c05[i]))
                     +'%15s'%('%.4e'%(true_em_line_flux[i])) \
                     +'%13s'%('%.4f'%(true_cont_flux_dens_fs[i])) \
                     +'%13s'%('%.4f'%(true_ew_obs_fs[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_r[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_i_minus_z[i])) \
                     +'%10s'%(str(c00[i])) \
                     +'\n')
   out_file.close()

   out_file = open(str(path)+'OII_simulation_before_noise_'+str(run)+'.dat','w')
   out_file.write('# \n')
   out_file.write('# Simulated HETDEX catalog of [O II] emitters \n')
   if scale == 0.1:    out_file.write('#   * one-tenth scale simulation \n')
   elif scale == 0.25: out_file.write('#   * one-quarter scale simulation \n')
   elif scale == 1.:   out_file.write('#   * full scale simulation \n')
   else: out_file.write('#   * %.2f scale Monte Carlo simulation \n'%scale)
   out_file.write('#   * Monte Carlo simulated quantities prior to addition of noise \n')
   out_file.write('#   *    imaging survey band: '+str(band)+'\n')
   out_file.write('# \n')
   out_file.write('# Column 1:  object type \n')
   out_file.write('# Column 2:  wavelength of emission line (angstroms) \n')
   out_file.write('# Column 3:  emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 4:  continuum flux density (microjanskys) \n')
   out_file.write('# Column 5:  equivalent width observed (angstroms) \n')
   out_file.write('# Column 6:  g-r color index \n')
   out_file.write('# Column 7:  g-i color index \n')
   out_file.write('# Column 8:  g-z color index \n')
   out_file.write('# Column 9:  r-i color index \n')
   out_file.write('# Column 10: r-z color index \n')
   out_file.write('# Column 11: i-z color index \n')
   out_file.write('# Column 12: [Ne III] 3869 flux (erg/cm^2/s) \n')
   out_file.write('# Column 13: H-beta   4861 flux (erg/cm^2/s) \n')
   out_file.write('# Column 14: [O III]  4959 flux (erg/cm^2/s) \n')
   out_file.write('# Column 15: [O III]  5007 flux (erg/cm^2/s) \n')
   out_file.write('# \n')
   out_file.write('# \n')

   f_from_EL = true_em_line_flux
   rel_strength   = np.array([0.416, 1., 1.617, 4.752])/1.791      ## (04-24-15) Anders_Fritze_2003.dat, metallicity one-fifth solar
   #rel_strength = np.array([0.300, 1., 1.399, 4.081])/3.010      ## (05-06-15) Anders_Fritze_2003.dat, metallicity 0.5-2 solar

   for i in range(cLAE,cLAE+cOII):
      out_file.write('%6s'%(str(c13[i])) \
                     +'%10s'%('%.2f'%(c05[i]))
                     +'%15s'%('%.4e'%(true_em_line_flux[i])) \
                     +'%13s'%('%.4f'%(true_cont_flux_dens[i])) \
                     +'%13s'%('%.4f'%(true_ew_obs[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_r[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_g_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_i[i])) \
                     +'%10s'%('%.3f'%(true_r_minus_z[i])) \
                     +'%10s'%('%.3f'%(true_i_minus_z[i])) \
                     )
      
      addl_lambda_ob = addl_lambda_rf * (1+c01[i])
      for k in range(len(addl_em_lines)):
         if addl_lambda_ob[k] <= 5500.:      #10*(bandpass_max-0.5):
            addl_flux = rel_strength[k] * f_from_EL[i]
            out_file.write('\t'+'%.4e'%(addl_flux))
         else:
            out_file.write('\t'+'0.')

      out_file.write('\n')

   out_file.close()

   out_file = open(str(path)+'OII_simulation_with_noise_'+str(run)+'.dat','w')
   out_file.write('# \n')
   out_file.write('# Simulated HETDEX catalog of [O II] emitters \n')
   if scale == 0.1:    out_file.write('#   * one-tenth scale simulation \n')
   elif scale == 0.25: out_file.write('#   * one-quarter scale simulation \n')
   elif scale == 1.:   out_file.write('#   * full scale simulation \n')
   else: out_file.write('#   * %.2f scale Monte Carlo simulation \n'%scale)
   out_file.write('#   * Monte Carlo simulated quantities with Gaussian noise added \n')
   out_file.write('#   *    imaging survey band: '+str(band)+'\n')
   out_file.write('#   *    5 sigma depth (mag): '+str(depth)+'\n')
   out_file.write('# \n')
   out_file.write('# Column 1:  object type \n')
   out_file.write('# Column 2:  wavelength of emission line (angstroms) \n')
   out_file.write('# Column 3:  emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 4:  continuum flux density (microjanskys) \n')
   out_file.write('# Column 5:  equivalent width observed (angstroms) \n')
   out_file.write('# Column 6:  g-r color index \n')
   out_file.write('# Column 7:  g-i color index \n')
   out_file.write('# Column 8:  g-z color index \n')
   out_file.write('# Column 9:  r-i color index \n')
   out_file.write('# Column 10: r-z color index \n')
   out_file.write('# Column 11: i-z color index \n')
   out_file.write('# Column 12: sigma uncertainty, emission line flux (erg/cm^2/s) \n')
   out_file.write('# Column 13: sigma uncertainty, continuum flux density (microjanskys) \n')
   out_file.write('# Column 14: sigma uncertainty, equivalent width observed (angstroms) \n')
   out_file.write('# Column 15: sigma uncertainty, g-r color index \n')
   out_file.write('# Column 16: f_nu continuum slope  \n')
   out_file.write('# Column 17: [Ne III] 3869 flux (erg/cm^2/s) \n')
   out_file.write('# Column 18: sigma uncertainty, [Ne III] 3869 flux (erg/cm^2/s) \n')
   out_file.write('# Column 19: H-beta   4861 flux (erg/cm^2/s) \n')
   out_file.write('# Column 20: sigma uncertainty, H-beta   4861 flux (erg/cm^2/s) \n')
   out_file.write('# Column 21: [O III]  4959 flux (erg/cm^2/s) \n')
   out_file.write('# Column 22: sigma uncertainty, [O III]  4959 flux (erg/cm^2/s) \n')
   out_file.write('# Column 23: [O III]  5007 flux (erg/cm^2/s) \n')
   out_file.write('# Column 24: sigma uncertainty, [O III]  5007 flux (erg/cm^2/s) \n')
   out_file.write('# \n')
   out_file.write('# \n')
   
   for i in range(cLAE,cLAE+cOII):
      out_file.write('%6s'%(str(c13[i])) \
                     +'%10s'%('%.2f'%(c05[i]))
                     +'%15s'%('%.4e'%(c08[i])) \
                     +'%12s'%('%.3f'%(c09[i])) \
                     +'%12s'%('%.3f'%(c07[i])) \
                     +'%10s'%('%.3f'%(c15[i])) \
                     +'%10s'%('%.3f'%(c16[i])) \
                     +'%10s'%('%.3f'%(c17[i])) \
                     +'%10s'%('%.3f'%(c18[i])) \
                     +'%10s'%('%.3f'%(c19[i])) \
                     +'%10s'%('%.3f'%(c20[i])) \
                     +'%15s'%('%.4e'%(sigma_line_flux[i])) \
                     +'%12s'%('%.3f'%(sigma_f_nu[i])) \
                     +'%12s'%('%.3f'%(sigma_ew_obs[i])) \
                     +'%10s'%('%.3f'%(sigma_g_minus_r[i])) \
                     +'%10s'%('%.3f'%(f_nu_cont_slope[i])) \
                     )

      addl_lambda_ob = addl_lambda_rf * (1+c01[i])
      for k in range(len(addl_em_lines)):
         if addl_lambda_ob[k] <= 5500.:      #10*(bandpass_max-0.5):
            sigma_addl_line = 0.2 * lineSens(addl_lambda_ob[k]) * np.sqrt(sky_area/300.)
            addl_flux = rel_strength[k] * f_from_EL[i]
            addl_flux += ncdfinv(random.random(),0.,sigma_addl_line)
            out_file.write('%15s'%('%.4e'%(addl_flux)))
            out_file.write('%15s'%('%.4e'%(sigma_addl_line)))
         else:
            out_file.write('%15s'%('0.'))
            out_file.write('%15s'%('0.'))

      out_file.write('\n')

   out_file.close()

   print('nb.write_sim_data() required %.2f seconds' % (time.time()-t0))
   print('nb.write_sim_data() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')



def get_sim_data(run,isb,sc,cc):
   t0 = time.time()
   global scale
   global sigma_line_flux, sigma_f_nu, sigma_ew_obs, sigma_g_minus_r, f_nu_cont_slope
   
   scale = sc
   noise = ['flat_spectra','before_noise','with_noise']
   objlabel = ['LAE','OII']
   
   sigma_line_flux, sigma_f_nu, sigma_ew_obs, sigma_g_minus_r, f_nu_cont_slope = [],[],[],[],[]
   objtype, wl_obs, lineFlux, f_nu, ew_obs, g_minus_r, g_minus_i, g_minus_z, r_minus_i, r_minus_z, i_minus_z, index = [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[]
   NeIII3869_flux, HB4861_flux, OIII4959_flux, OIII5007_flux = [],[],[],[]
   
   for i in range(len(noise)):
      for j in range(len(objlabel)):
         data = open(str(path)+str(objlabel[j])+'_simulation_'+str(noise[i])+'_'+str(run)+'.dat','r')
         for line in data.readlines():
            if not line.startswith('#'):
               simobject = line.split()
               objtype[i].append(str(simobject[0]))
               wl_obs[i].append(float(simobject[1]))
               lineFlux[i].append(float(simobject[2]))
               f_nu[i].append(float(simobject[3]))
               ew_obs[i].append(float(simobject[4]))
               g_minus_r[i].append(float(simobject[5]))
               g_minus_i[i].append(float(simobject[6]))
               g_minus_z[i].append(float(simobject[7]))
               r_minus_i[i].append(float(simobject[8]))
               r_minus_z[i].append(float(simobject[9]))
               i_minus_z[i].append(float(simobject[10]))
               if noise[i] == 'with_noise':
                  sigma_line_flux.append(float(simobject[11]))
                  sigma_f_nu.append(float(simobject[12]))
                  sigma_ew_obs.append(float(simobject[13]))
                  sigma_g_minus_r.append(float(simobject[14]))
                  f_nu_cont_slope.append(float(simobject[15]))
                  NeIII3869_flux.append(float(simobject[16]))
                  HB4861_flux.append(float(simobject[18]))
                  OIII4959_flux.append(float(simobject[20]))
                  OIII5007_flux.append(float(simobject[22]))
               if noise[i] == 'flat_spectra':
                  index.append(str(simobject[11]))
         data.close()
         print('nb.get_sim_data(), done reading in simulated '+str(objlabel[j])+' '+str(noise[i])+' data ('+str(run)+'): %.1f seconds' % (time.time()-t0))

   global c00,c13,c05,c08,c09,c07,c15,c16,c17,c18,c19,c20,c10,c12,c14,c11,c01,c04,c03
   global true_ew_rest,true_ew_obs,true_ew_inf,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n,true_em_line_flux,true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs

   c00 = np.array(index)
   c13 = np.array(objtype[0])
   c05 = np.array(wl_obs[0])

   c08 = np.array(lineFlux[2])
   true_em_line_flux = np.array(lineFlux[1])

   c09 = np.array(f_nu[2])
   true_cont_flux_dens = np.array(f_nu[1])
   true_cont_flux_dens_fs = np.array(f_nu[0])

   c07 = np.array(ew_obs[2])
   true_ew_obs = np.array(ew_obs[1])
   true_ew_obs_fs = np.array(ew_obs[0])

   c15 = np.array(g_minus_r[2])
   c16 = np.array(g_minus_i[2])
   c17 = np.array(g_minus_z[2])
   c18 = np.array(r_minus_i[2])
   c19 = np.array(r_minus_z[2])
   c20 = np.array(i_minus_z[2])
   true_g_minus_r = np.array(g_minus_r[1])
   true_g_minus_i = np.array(g_minus_i[1])
   true_g_minus_z = np.array(g_minus_z[1])
   true_r_minus_i = np.array(r_minus_i[1])
   true_r_minus_z = np.array(r_minus_z[1])
   true_i_minus_z = np.array(i_minus_z[1])

   sigma_line_flux = np.array(sigma_line_flux)
   sigma_f_nu = np.array(sigma_f_nu)
   sigma_ew_obs = np.array(sigma_ew_obs)
   sigma_g_minus_r = np.array(sigma_g_minus_r)
   f_nu_cont_slope = np.array(f_nu_cont_slope)

   c10 = c05/1215.668 -1      # inferred Lyman-alpha redshift

   c11 = c07/(1.+c10)
   true_ew_inf = true_ew_obs/(1+c10)
   true_ew_inf_fs = true_ew_obs_fs/(1+c10)

   c12 = c09/sigma_f_nu
   true_s_to_n = true_cont_flux_dens/sigma_f_nu
   true_s_to_n_fs = true_cont_flux_dens_fs/sigma_f_nu

   c14 = []
   true_AB_cont_mag = 23.9-2.5*(np.log10(true_cont_flux_dens))
   true_AB_cont_mag_fs = 23.9-2.5*(np.log10(true_cont_flux_dens_fs))

   for i in range(len(index)):
      if c09[i] > 0: c14.append(23.9-2.5*(np.log10(c09[i])))
      else: c14.append(999)
   c14 = np.array(c14)

   cLAE = objtype[0].count('LAE')
   cOII = objtype[0].count('OII')
   c01 = []
   for i in range(cLAE): c01.append(c05[i]/1215.668 -1)
   for i in range(cLAE,cLAE+cOII): c01.append(c05[i]/3727.45 -1)
   c01 = np.array(c01)

   c04 = lumDist(c01,cc)

   c03 = c08 * (4*np.pi*(3.08567758e24*c04)**2)

   true_ew_rest_fs = true_ew_obs_fs/(1+c01)

   global addl_lines
   addl_lines = []
   print(len(NeIII3869_flux),len(HB4861_flux),len(OIII4959_flux),len(OIII5007_flux))
   for i in range(len(index)):
      addl_lines.append([NeIII3869_flux[i], HB4861_flux[i], OIII4959_flux[i], OIII5007_flux[i]])
   print(len(addl_lines),len(addl_lines[0]))



def write_prob_ratio(run):
   t = time.time()
   
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global ratioLAE_
   
   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   wl_obs = c05
   redshift = []
   for i in range(cLAE): redshift.append(wl_obs[i]/1215.668-1)
   for i in range(cLAE,cLAE+cOII): redshift.append(wl_obs[i]/3727.45-1)
   redshift = np.array(redshift)
   probratio = np.array(ratioLAE_)
   probLAE = probratio/(1+probratio)
   for i in range(cLAE):
      if probratio[i] == 100: probLAE[i] = 1.
   for i in range(cLAE,cLAE+cOII):
      if probLAE[i] < 0:
         probLAE[i] = 0.
         probratio[i] = 0.

   out_file = open(str(path)+str(run)+'_bayesian_results.dat','w')
   out_file.write('# index'+'\t'+'redshift'+'\t'+'prob_ratio'+'\t'+'prob_LAE'+'\t'+'object type'+'\n')
   for i in range(cLAE+cOII):
      out_file.write(str(int(c00[i]))+'\t'+'%.4f'%(redshift[i])+'\t'+'%.6e'%(probratio[i])+'\t'+'%.5f'%(probLAE[i])+'\t'+str(objtype[i])+'\n')
   out_file.close()

   print('nb.write_prob_ratio() required %.2f seconds' % (time.time()-t))
   print('nb.write_prob_ratio() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')



def get_prob_ratio(run):
   
   global _redshift, _probratio, _probLAE, _objtype
   _redshift, _probratio, _probLAE, _objtype = [],[],[],[]
   
   data = open(str(path)+str(run)+'_bayesian_results.dat','r')
   
   for line in data.readlines():
      if not line.startswith('#'):
         index, redshift, probratio, probLAE, objtype = map(str, line.split())
         _redshift.append(float(redshift))
         _probratio.append(float(probratio))
         _probLAE.append(float(probLAE))
         _objtype.append(str(objtype))
   
   data.close()

   global ratioLAE_, c05, c13
   ratioLAE_ = np.copy(_probratio)
   c13 = np.array(_objtype)
   c05 = []
   for i in range(len(_redshift)):
      if c13[i] == 'LAE': c05.append(1215.668*(1+_redshift[i]))
      elif c13[i] == 'OII': c05.append(3727.45*(1+_redshift[i]))
   c13 = np.array(c13)



def ncdf(x,mu,sigma):
   return 0.5 * ( 1 + ssp.erf((x-mu)/sigma/(2**0.5)) )

def ncdfinv(y,mu,sigma):
   return ssp.erfinv(2*y-1) * np.sqrt(2) * sigma + mu



def lumDist(z,cosmo):                        ## returns luminosity distance in Mpc
   #global cosmo
   d_a = cd.angular_diameter_distance(z, **cosmo)
   d_l = d_a * (1+z)**2
   return d_l



def lineSens(WL):                        ## returns HETDEX line flux sensitivity limit in erg/cm^2/s
   #global LyaWL, LFlim
   return np.interp(WL,LyaWL,LFlim)
   


def plotLineSens():
   z, z_int = [1.878], 0.0001
   #while z[len(z)-1] < 2.5:
   while z[len(z)-1] < 3.5:
      z.append(z[len(z)-1]+z_int)

   fluxlim = []
   for i in range(len(z)):
      fluxlim.append(lineSens((1+z[i])*1215.668))

   plt.close()
   plt.scatter(z,fluxlim,c='red',lw=0,s=6,rasterized=True)
   plt.ylabel('line flux limit (erg cm$^{-2}$ s$^{-1}$)')
   plt.xlabel('Lyman-alpha redshift')
   #plt.yscale('log')
   plt.yscale('linear')
   #plt.xlim(1.85,2.501)
   plt.xlim(1.87,3.5)
   plt.ylim(10**-17,10**-16)

   for i in range(20):
      z_inf = 1.9+0.1*i
      plt.plot([z_inf,z_inf],[10**-17,10**-16],'k-',ls='dashed',lw=0.5)


   plt.savefig('line_flux_limit_linear_interpolate.pdf',dpi=450)
   plt.close()



def init(_alpha_LAE,_mult_LStar_LAE,_mult_PhiStar_LAE,_mult_w0_LAE,_alpha_OII,_mult_LStar_OII,_mult_PhiStar_OII,_mult_w0_OII):
   
   reload(b)
   b.init(_alpha_LAE,_mult_LStar_LAE,_mult_PhiStar_LAE,_mult_w0_LAE,_alpha_OII,_mult_LStar_OII,_mult_PhiStar_OII,_mult_w0_OII)
   
   ## Greg's middle baseline for line flux sensitivity, see Line_flux_limit_5_sigma_baseline.dat
   global LyaWL, LFlim
   LyaWL, LFlim = [0]*250, [0]*250
   LyaWL[0], LFlim[0] = 3500.00064, 8.58779E-17
   LyaWL[1], LFlim[1] = 3508.02624, 8.47950E-17
   LyaWL[2], LFlim[2] = 3516.06400, 8.37411E-17
   LyaWL[3], LFlim[3] = 3524.10176, 8.27151E-17
   LyaWL[4], LFlim[4] = 3532.12736, 8.17159E-17
   LyaWL[5], LFlim[5] = 3540.16512, 8.07423E-17
   LyaWL[6], LFlim[6] = 3548.19072, 7.97935E-17
   LyaWL[7], LFlim[7] = 3556.22848, 7.88684E-17
   LyaWL[8], LFlim[8] = 3564.25408, 7.79662E-17
   LyaWL[9], LFlim[9] = 3572.29184, 7.70862E-17
   LyaWL[10], LFlim[10] = 3580.31744, 7.37038E-17
   LyaWL[11], LFlim[11] = 3588.35520, 7.04207E-17
   LyaWL[12], LFlim[12] = 3596.38080, 7.30409E-17
   LyaWL[13], LFlim[13] = 3604.41856, 7.01981E-17
   LyaWL[14], LFlim[14] = 3612.44416, 6.96233E-17
   LyaWL[15], LFlim[15] = 3620.48192, 7.74228E-17
   LyaWL[16], LFlim[16] = 3628.51968, 7.76095E-17
   LyaWL[17], LFlim[17] = 3636.54528, 8.58089E-17
   LyaWL[18], LFlim[18] = 3644.58304, 8.03009E-17
   LyaWL[19], LFlim[19] = 3652.60864, 7.80095E-17
   LyaWL[20], LFlim[20] = 3660.64640, 7.78460E-17
   LyaWL[21], LFlim[21] = 3668.67200, 7.24132E-17
   LyaWL[22], LFlim[22] = 3676.70976, 6.81157E-17
   LyaWL[23], LFlim[23] = 3684.73536, 6.48053E-17
   LyaWL[24], LFlim[24] = 3692.77312, 6.56293E-17
   LyaWL[25], LFlim[25] = 3700.79872, 7.27374E-17
   LyaWL[26], LFlim[26] = 3708.83648, 6.92679E-17
   LyaWL[27], LFlim[27] = 3716.86208, 6.66188E-17
   LyaWL[28], LFlim[28] = 3724.89984, 6.30892E-17
   LyaWL[29], LFlim[29] = 3732.93760, 6.90976E-17
   LyaWL[30], LFlim[30] = 3740.96320, 7.21361E-17
   LyaWL[31], LFlim[31] = 3749.00096, 6.46427E-17
   LyaWL[32], LFlim[32] = 3757.02656, 6.35535E-17
   LyaWL[33], LFlim[33] = 3765.06432, 5.89598E-17
   LyaWL[34], LFlim[34] = 3773.08992, 6.00240E-17
   LyaWL[35], LFlim[35] = 3781.12768, 6.01406E-17
   LyaWL[36], LFlim[36] = 3789.15328, 5.85113E-17
   LyaWL[37], LFlim[37] = 3797.19104, 5.83368E-17
   LyaWL[38], LFlim[38] = 3805.21664, 5.77919E-17
   LyaWL[39], LFlim[39] = 3813.25440, 5.94066E-17
   LyaWL[40], LFlim[40] = 3821.28000, 5.92455E-17
   LyaWL[41], LFlim[41] = 3829.31776, 6.26535E-17
   LyaWL[42], LFlim[42] = 3837.34336, 6.33413E-17
   LyaWL[43], LFlim[43] = 3845.38112, 6.13060E-17
   LyaWL[44], LFlim[44] = 3853.41888, 5.80438E-17
   LyaWL[45], LFlim[45] = 3861.44448, 5.43236E-17
   LyaWL[46], LFlim[46] = 3869.48224, 5.35939E-17
   LyaWL[47], LFlim[47] = 3877.50784, 5.24846E-17
   LyaWL[48], LFlim[48] = 3885.54560, 4.93688E-17
   LyaWL[49], LFlim[49] = 3893.57120, 5.04683E-17
   LyaWL[50], LFlim[50] = 3901.60896, 5.19567E-17
   LyaWL[51], LFlim[51] = 3909.63456, 5.81161E-17
   LyaWL[52], LFlim[52] = 3917.67232, 5.48940E-17
   LyaWL[53], LFlim[53] = 3925.69792, 4.97027E-17
   LyaWL[54], LFlim[54] = 3933.73568, 4.43705E-17
   LyaWL[55], LFlim[55] = 3941.76128, 4.92324E-17
   LyaWL[56], LFlim[56] = 3949.79904, 5.05659E-17
   LyaWL[57], LFlim[57] = 3957.83680, 4.85842E-17
   LyaWL[58], LFlim[58] = 3965.86240, 4.36797E-17
   LyaWL[59], LFlim[59] = 3973.90016, 4.33257E-17
   LyaWL[60], LFlim[60] = 3981.92576, 4.82361E-17
   LyaWL[61], LFlim[61] = 3989.96352, 4.76576E-17
   LyaWL[62], LFlim[62] = 3997.98912, 4.71065E-17
   LyaWL[63], LFlim[63] = 4006.02688, 4.79427E-17
   LyaWL[64], LFlim[64] = 4014.05248, 5.09209E-17
   LyaWL[65], LFlim[65] = 4022.09024, 4.92586E-17
   LyaWL[66], LFlim[66] = 4030.11584, 4.79038E-17
   LyaWL[67], LFlim[67] = 4038.15360, 4.75190E-17
   LyaWL[68], LFlim[68] = 4046.17920, 5.14693E-17
   LyaWL[69], LFlim[69] = 4054.21696, 4.92683E-17
   LyaWL[70], LFlim[70] = 4062.25472, 4.83870E-17
   LyaWL[71], LFlim[71] = 4070.28032, 5.00043E-17
   LyaWL[72], LFlim[72] = 4078.31808, 4.87716E-17
   LyaWL[73], LFlim[73] = 4086.34368, 4.75618E-17
   LyaWL[74], LFlim[74] = 4094.38144, 4.58229E-17
   LyaWL[75], LFlim[75] = 4102.40704, 4.32813E-17
   LyaWL[76], LFlim[76] = 4110.44480, 4.37177E-17
   LyaWL[77], LFlim[77] = 4118.47040, 4.46614E-17
   LyaWL[78], LFlim[78] = 4126.50816, 4.60225E-17
   LyaWL[79], LFlim[79] = 4134.53376, 4.57953E-17
   LyaWL[80], LFlim[80] = 4142.57152, 4.54652E-17
   LyaWL[81], LFlim[81] = 4150.59712, 4.40922E-17
   LyaWL[82], LFlim[82] = 4158.63488, 4.62173E-17
   LyaWL[83], LFlim[83] = 4166.67264, 4.82653E-17
   LyaWL[84], LFlim[84] = 4174.69824, 4.75369E-17
   LyaWL[85], LFlim[85] = 4182.73600, 4.55771E-17
   LyaWL[86], LFlim[86] = 4190.76160, 4.35319E-17
   LyaWL[87], LFlim[87] = 4198.79936, 4.21368E-17
   LyaWL[88], LFlim[88] = 4206.82496, 4.14461E-17
   LyaWL[89], LFlim[89] = 4214.86272, 4.19800E-17
   LyaWL[90], LFlim[90] = 4222.88832, 4.16962E-17
   LyaWL[91], LFlim[91] = 4230.92608, 4.07434E-17
   LyaWL[92], LFlim[92] = 4238.95168, 4.12555E-17
   LyaWL[93], LFlim[93] = 4246.98944, 4.18439E-17
   LyaWL[94], LFlim[94] = 4255.01504, 4.19093E-17
   LyaWL[95], LFlim[95] = 4263.05280, 4.12194E-17
   LyaWL[96], LFlim[96] = 4271.07840, 4.19028E-17
   LyaWL[97], LFlim[97] = 4279.11616, 4.19328E-17
   LyaWL[98], LFlim[98] = 4287.15392, 4.14909E-17
   LyaWL[99], LFlim[99] = 4295.17952, 4.11614E-17
   LyaWL[100], LFlim[100] = 4303.21728, 3.91361E-17
   LyaWL[101], LFlim[101] = 4311.24288, 3.95501E-17
   LyaWL[102], LFlim[102] = 4319.28064, 4.31839E-17
   LyaWL[103], LFlim[103] = 4327.30624, 3.99921E-17
   LyaWL[104], LFlim[104] = 4335.34400, 4.08349E-17
   LyaWL[105], LFlim[105] = 4343.36960, 3.93918E-17
   LyaWL[106], LFlim[106] = 4351.40736, 4.11580E-17
   LyaWL[107], LFlim[107] = 4359.43296, 4.54893E-17
   LyaWL[108], LFlim[108] = 4367.47072, 4.21084E-17
   LyaWL[109], LFlim[109] = 4375.49632, 4.11251E-17
   LyaWL[110], LFlim[110] = 4383.53408, 4.09289E-17
   LyaWL[111], LFlim[111] = 4391.57184, 4.00564E-17
   LyaWL[112], LFlim[112] = 4399.59744, 4.08841E-17
   LyaWL[113], LFlim[113] = 4407.63520, 4.32614E-17
   LyaWL[114], LFlim[114] = 4415.66080, 4.43877E-17
   LyaWL[115], LFlim[115] = 4423.69856, 4.41704E-17
   LyaWL[116], LFlim[116] = 4431.72416, 4.30561E-17
   LyaWL[117], LFlim[117] = 4439.76192, 4.24060E-17
   LyaWL[118], LFlim[118] = 4447.78752, 4.22086E-17
   LyaWL[119], LFlim[119] = 4455.82528, 4.07700E-17
   LyaWL[120], LFlim[120] = 4463.85088, 4.00664E-17
   LyaWL[121], LFlim[121] = 4471.88864, 4.03750E-17
   LyaWL[122], LFlim[122] = 4479.91424, 4.10311E-17
   LyaWL[123], LFlim[123] = 4487.95200, 4.13301E-17
   LyaWL[124], LFlim[124] = 4495.98976, 4.11781E-17
   LyaWL[125], LFlim[125] = 4504.01536, 4.11622E-17
   LyaWL[126], LFlim[126] = 4512.05312, 4.15699E-17
   LyaWL[127], LFlim[127] = 4520.07872, 4.11544E-17
   LyaWL[128], LFlim[128] = 4528.11648, 4.16200E-17
   LyaWL[129], LFlim[129] = 4536.14208, 4.28884E-17
   LyaWL[130], LFlim[130] = 4544.17984, 4.31325E-17
   LyaWL[131], LFlim[131] = 4552.20544, 4.24052E-17
   LyaWL[132], LFlim[132] = 4560.24320, 4.18982E-17
   LyaWL[133], LFlim[133] = 4568.26880, 4.12283E-17
   LyaWL[134], LFlim[134] = 4576.30656, 4.12378E-17
   LyaWL[135], LFlim[135] = 4584.33216, 4.15980E-17
   LyaWL[136], LFlim[136] = 4592.36992, 4.09836E-17
   LyaWL[137], LFlim[137] = 4600.40768, 4.01482E-17
   LyaWL[138], LFlim[138] = 4608.43328, 3.99451E-17
   LyaWL[139], LFlim[139] = 4616.47104, 3.99664E-17
   LyaWL[140], LFlim[140] = 4624.49664, 4.00349E-17
   LyaWL[141], LFlim[141] = 4632.53440, 3.97906E-17
   LyaWL[142], LFlim[142] = 4640.56000, 3.92588E-17
   LyaWL[143], LFlim[143] = 4648.59776, 3.87472E-17
   LyaWL[144], LFlim[144] = 4656.62336, 3.96056E-17
   LyaWL[145], LFlim[145] = 4664.66112, 4.11128E-17
   LyaWL[146], LFlim[146] = 4672.68672, 4.02669E-17
   LyaWL[147], LFlim[147] = 4680.72448, 4.04600E-17
   LyaWL[148], LFlim[148] = 4688.75008, 4.02924E-17
   LyaWL[149], LFlim[149] = 4696.78784, 3.98659E-17
   LyaWL[150], LFlim[150] = 4704.81344, 3.90662E-17
   LyaWL[151], LFlim[151] = 4712.85120, 3.97629E-17
   LyaWL[152], LFlim[152] = 4720.88896, 3.99433E-17
   LyaWL[153], LFlim[153] = 4728.91456, 3.95446E-17
   LyaWL[154], LFlim[154] = 4736.95232, 3.93130E-17
   LyaWL[155], LFlim[155] = 4744.97792, 3.97511E-17
   LyaWL[156], LFlim[156] = 4753.01568, 3.91101E-17
   LyaWL[157], LFlim[157] = 4761.04128, 3.88355E-17
   LyaWL[158], LFlim[158] = 4769.07904, 3.84494E-17
   LyaWL[159], LFlim[159] = 4777.10464, 3.90188E-17
   LyaWL[160], LFlim[160] = 4785.14240, 3.87821E-17
   LyaWL[161], LFlim[161] = 4793.16800, 3.87975E-17
   LyaWL[162], LFlim[162] = 4801.20576, 3.86451E-17
   LyaWL[163], LFlim[163] = 4809.23136, 3.85878E-17
   LyaWL[164], LFlim[164] = 4817.26912, 3.97090E-17
   LyaWL[165], LFlim[165] = 4825.30688, 4.03886E-17
   LyaWL[166], LFlim[166] = 4833.33248, 4.03393E-17
   LyaWL[167], LFlim[167] = 4841.37024, 4.01690E-17
   LyaWL[168], LFlim[168] = 4849.39584, 3.93792E-17
   LyaWL[169], LFlim[169] = 4857.43360, 3.78358E-17
   LyaWL[170], LFlim[170] = 4865.45920, 3.81016E-17
   LyaWL[171], LFlim[171] = 4873.49696, 3.76625E-17
   LyaWL[172], LFlim[172] = 4881.52256, 3.90232E-17
   LyaWL[173], LFlim[173] = 4889.56032, 3.87568E-17
   LyaWL[174], LFlim[174] = 4897.58592, 3.91195E-17
   LyaWL[175], LFlim[175] = 4905.62368, 3.89577E-17
   LyaWL[176], LFlim[176] = 4913.64928, 3.83832E-17
   LyaWL[177], LFlim[177] = 4921.68704, 3.74830E-17
   LyaWL[178], LFlim[178] = 4929.72480, 3.86967E-17
   LyaWL[179], LFlim[179] = 4937.75040, 3.86864E-17
   LyaWL[180], LFlim[180] = 4945.78816, 3.86326E-17
   LyaWL[181], LFlim[181] = 4953.81376, 3.92888E-17
   LyaWL[182], LFlim[182] = 4961.85152, 3.83594E-17
   LyaWL[183], LFlim[183] = 4969.87712, 3.91876E-17
   LyaWL[184], LFlim[184] = 4977.91488, 3.92309E-17
   LyaWL[185], LFlim[185] = 4985.94048, 3.81904E-17
   LyaWL[186], LFlim[186] = 4993.97824, 3.91887E-17
   LyaWL[187], LFlim[187] = 5002.00384, 3.83307E-17
   LyaWL[188], LFlim[188] = 5010.04160, 3.82223E-17
   LyaWL[189], LFlim[189] = 5018.06720, 3.76832E-17
   LyaWL[190], LFlim[190] = 5026.10496, 3.76001E-17
   LyaWL[191], LFlim[191] = 5034.13056, 3.78771E-17
   LyaWL[192], LFlim[192] = 5042.16832, 3.68258E-17
   LyaWL[193], LFlim[193] = 5050.20608, 3.73250E-17
   LyaWL[194], LFlim[194] = 5058.23168, 3.77358E-17
   LyaWL[195], LFlim[195] = 5066.26944, 3.75363E-17
   LyaWL[196], LFlim[196] = 5074.29504, 3.73396E-17
   LyaWL[197], LFlim[197] = 5082.33280, 3.74794E-17
   LyaWL[198], LFlim[198] = 5090.35840, 3.80544E-17
   LyaWL[199], LFlim[199] = 5098.39616, 3.73037E-17
   LyaWL[200], LFlim[200] = 5106.42176, 3.72491E-17
   LyaWL[201], LFlim[201] = 5114.45952, 3.74600E-17
   LyaWL[202], LFlim[202] = 5122.48512, 3.72336E-17
   LyaWL[203], LFlim[203] = 5130.52288, 3.75199E-17
   LyaWL[204], LFlim[204] = 5138.54848, 3.83346E-17
   LyaWL[205], LFlim[205] = 5146.58624, 3.83748E-17
   LyaWL[206], LFlim[206] = 5154.62400, 3.83509E-17
   LyaWL[207], LFlim[207] = 5162.64960, 3.79870E-17
   LyaWL[208], LFlim[208] = 5170.68736, 3.59924E-17
   LyaWL[209], LFlim[209] = 5178.71296, 3.73076E-17
   LyaWL[210], LFlim[210] = 5186.75072, 3.74566E-17
   LyaWL[211], LFlim[211] = 5194.77632, 3.85782E-17
   LyaWL[212], LFlim[212] = 5202.81408, 4.01949E-17
   LyaWL[213], LFlim[213] = 5210.83968, 3.77919E-17
   LyaWL[214], LFlim[214] = 5218.87744, 3.83645E-17
   LyaWL[215], LFlim[215] = 5226.90304, 3.81564E-17
   LyaWL[216], LFlim[216] = 5234.94080, 3.83984E-17
   LyaWL[217], LFlim[217] = 5242.96640, 3.84863E-17
   LyaWL[218], LFlim[218] = 5251.00416, 3.83138E-17
   LyaWL[219], LFlim[219] = 5259.04192, 3.86892E-17
   LyaWL[220], LFlim[220] = 5267.06752, 3.62419E-17
   LyaWL[221], LFlim[221] = 5275.10528, 3.81221E-17
   LyaWL[222], LFlim[222] = 5283.13088, 3.73345E-17
   LyaWL[223], LFlim[223] = 5291.16864, 3.78526E-17
   LyaWL[224], LFlim[224] = 5299.19424, 3.81800E-17
   LyaWL[225], LFlim[225] = 5307.23200, 3.83290E-17
   LyaWL[226], LFlim[226] = 5315.25760, 3.87418E-17
   LyaWL[227], LFlim[227] = 5323.29536, 3.84026E-17
   LyaWL[228], LFlim[228] = 5331.32096, 3.84439E-17
   LyaWL[229], LFlim[229] = 5339.35872, 3.84157E-17
   LyaWL[230], LFlim[230] = 5347.38432, 3.86638E-17
   LyaWL[231], LFlim[231] = 5355.42208, 3.88684E-17
   LyaWL[232], LFlim[232] = 5363.45984, 3.82820E-17
   LyaWL[233], LFlim[233] = 5371.48544, 3.84686E-17
   LyaWL[234], LFlim[234] = 5379.52320, 3.86777E-17
   LyaWL[235], LFlim[235] = 5387.54880, 3.85325E-17
   LyaWL[236], LFlim[236] = 5395.58656, 3.81814E-17
   LyaWL[237], LFlim[237] = 5403.61216, 3.79271E-17
   LyaWL[238], LFlim[238] = 5411.64992, 3.82173E-17
   LyaWL[239], LFlim[239] = 5419.67552, 3.84565E-17
   LyaWL[240], LFlim[240] = 5427.71328, 3.83676E-17
   LyaWL[241], LFlim[241] = 5435.73888, 3.86560E-17
   LyaWL[242], LFlim[242] = 5443.77664, 3.89290E-17
   LyaWL[243], LFlim[243] = 5451.80224, 3.97822E-17
   LyaWL[244], LFlim[244] = 5459.84000, 4.22912E-17
   LyaWL[245], LFlim[245] = 5467.86560, 4.03826E-17
   LyaWL[246], LFlim[246] = 5475.90336, 3.94643E-17
   LyaWL[247], LFlim[247] = 5483.94112, 3.97560E-17
   LyaWL[248], LFlim[248] = 5491.96672, 3.97984E-17
   LyaWL[249], LFlim[249] = 5500.00448, 4.00431E-17



def plot_prob_LAE_vs(run):
   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   zbin = c02
   fline = c08
   fcont = c09
      
   print('P(LAE) mean max min:',mean(prob_LAE),max(prob_LAE),min(prob_LAE))
   
   fig, ax = plt.subplots()
   
   ## prior probability that an observed object is an LAE vs. wavelength (observed)
   wl = c05
   plt.scatter(wl[0:cLAE],prob_LAE[0:cLAE],c='red',alpha=0.03,lw=0,s=12,label='LAE (N='+str(cLAE)+')')
   plt.scatter(wl[cLAE:cLAE+cOII],prob_LAE[cLAE:cLAE+cOII],alpha=0.03,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')')
   plt.ylabel('$P$ ( LAE )')
   plt.xlabel('observed wavelength ($\mathrm{\AA}$)')
   plt.xlim(3250,5750)
   plt.ylim(-0.05,1.05)
   legend = plt.legend(loc='upper right',scatterpoints=90,shadow=True,title='post-detection limit cut')
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('medium')
   for label in legend.get_lines(): label.set_linewidth(1)
   plt.grid()
   plt.savefig(str(run)+'-P(LAE)-vs-WL.pdf')
   plt.close()
   
   ## prior probability that an observed object is an LAE vs. equivalent width (rest frame)
   ew = c06
   plt.scatter(ew[0:cLAE],prob_LAE[0:cLAE],c='red',alpha=0.03,lw=0,s=12,label='LAE (N='+str(cLAE)+')')
   plt.scatter(ew[cLAE:cLAE+cOII],prob_LAE[cLAE:cLAE+cOII],alpha=0.03,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')')
   plt.ylabel('$P$ ( LAE )')
   plt.xlabel('$EW_{rest frame}$ ($\mathrm{\AA}$)')
   plt.xlim(-50,425)
   plt.ylim(-0.05,1.05)
   legend = plt.legend(loc='upper right',scatterpoints=90,shadow=True,title='post-detection limit cut')
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('medium')
   for label in legend.get_lines(): label.set_linewidth(1)
   plt.grid()
   plt.savefig(str(run)+'-P(LAE)-vs-EW.pdf')
   plt.close()
   
   ## prior probability that an observed object is an LAE vs. emission line flux
   fline = c08
   plt.scatter(fline[0:cLAE],prob_LAE[0:cLAE],c='red',alpha=0.03,lw=0,s=12,label='LAE (N='+str(cLAE)+')')
   plt.scatter(fline[cLAE:cLAE+cOII],prob_LAE[cLAE:cLAE+cOII],alpha=0.03,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')')
   plt.ylabel('$P$ ( LAE )')
   plt.xlabel('emission line flux (erg/sec/cm$^2$)')
   plt.xscale('log')
   plt.xlim(4*10**-18,4*10**-11)
   plt.ylim(-0.05,1.05)
   legend = plt.legend(loc='upper right',scatterpoints=90,shadow=True,title='post-detection limit cut')
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('medium')
   for label in legend.get_lines(): label.set_linewidth(1)
   plt.grid()
   plt.savefig(str(run)+'-P(LAE)-vs-ELF.pdf')
   plt.close()
   
   ## prior probability that an observed object is an LAE vs. continuum flux density
   fcont = c09
   plt.scatter(fcont[0:cLAE],prob_LAE[0:cLAE],c='red',alpha=0.03,lw=0,s=12,label='LAE (N='+str(cLAE)+')')
   plt.scatter(fcont[cLAE:cLAE+cOII],prob_LAE[cLAE:cLAE+cOII],alpha=0.03,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')')
   plt.ylabel('$P$ ( LAE )')
   plt.xlabel('continuum flux density ($\mu$Jy)')
   plt.xscale('log')
   plt.xlim(5*10**-3,7*10**7)
   plt.ylim(-0.05,1.05)
   legend = plt.legend(loc='upper right',scatterpoints=90,shadow=True,title='post-detection limit cut')
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('medium')
   for label in legend.get_lines(): label.set_linewidth(0.2)
   plt.grid()
   plt.savefig(str(run)+'-P(LAE)-vs-CFD.pdf')
   plt.close()



def noise_debug_plots(run):
   fline = c08
   fcont = c09
   ewinf = c11
   
   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   ## plot emission line flux
   fig = plt.figure()
   ax1 = fig.add_subplot(221)
   
   ax1.scatter(true_em_line_flux[cLAE:cLAE+cOII],fline[cLAE:cLAE+cOII],alpha=0.01,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')',rasterized=True)
   ax1.scatter(true_em_line_flux[0:cLAE],fline[0:cLAE],c='red',alpha=0.02,lw=0,s=12,label='LAE (N='+str(cLAE)+')',rasterized=True)
   
   ax1.set_xlabel('true emission line flux (erg/sec/cm$^2$)',fontsize=9)
   ax1.set_ylabel('recorded emission line flux (erg/sec/cm$^2$)',fontsize=8)
   
   ax1.set_xscale('log')
   ax1.set_yscale('log')
   
   ax1.set_xlim(4*10**-18,4*10**-12)
   ax1.set_ylim(10**-17,4*10**-12)
   
   ax1.set_title('emission line flux',fontsize=10)
   
   legend = plt.legend(loc='upper left',scatterpoints=200,shadow=True)
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('x-small')
   for label in legend.get_lines(): label.set_linewidth(1)
   plt.grid()
   
   ## plot continuum flux density
   ax2 = fig.add_subplot(222)

   ax2.scatter(true_cont_flux_dens[cLAE:cLAE+cOII],fcont[cLAE:cLAE+cOII],alpha=0.01,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')',rasterized=True)
   ax2.scatter(true_cont_flux_dens[0:cLAE],fcont[0:cLAE],c='red',alpha=0.02,lw=0,s=12,label='LAE (N='+str(cLAE)+')',rasterized=True)

   ax2.set_xlabel('true continuum flux density ($\mu$Jy)',fontsize=9)
   ax2.set_ylabel('recorded continuum flux density ($\mu$Jy)',fontsize=8)

   ax2.set_xscale('log')
   ax2.set_yscale('log')

   ax2.set_xlim(10**-3,6*10**6)
   ax2.set_ylim(10**-4,9*10**6)
   
   ax2.set_title('continuum flux density',fontsize=10)
   
   
   legend = plt.legend(loc='upper left',scatterpoints=200,shadow=True)
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('x-small')
   for label in legend.get_lines(): label.set_linewidth(1)
   plt.grid()

   ## plot inferred EW
   ax3 = fig.add_subplot(223)

   ax3.scatter(true_ew_inf[cLAE:cLAE+cOII],ewinf[cLAE:cLAE+cOII],alpha=0.01,c='blue',lw=0,s=12,label='[O II] (N='+str(cOII)+')',rasterized=True)
   ax3.scatter(true_ew_inf[0:cLAE],ewinf[0:cLAE],c='red',alpha=0.02,lw=0,s=12,label='LAE (N='+str(cLAE)+')',rasterized=True)

   ax3.set_xlabel('true inferred EW$_{rest}$ ($\mathrm{\AA}$)',fontsize=9)
   ax3.set_ylabel('recorded inferred EW$_{rest}$ ($\mathrm{\AA}$)',fontsize=8)

   ax3.set_xscale('log')
   ax3.set_yscale('log')

   ax3.set_xlim(10**-2,9*10**2)
   ax3.set_ylim(10**-2,5*10**3)
   
   ax3.set_title('inferred equivalent width',fontsize=10)
   
   legend = plt.legend(loc='upper left',scatterpoints=200,shadow=True)
   frame = legend.get_frame()
   frame.set_facecolor('0.95')
   for label in legend.get_texts(): label.set_fontsize('x-small')
   for label in legend.get_lines(): label.set_linewidth(1)

   plt.tight_layout()
   plt.grid()
   plt.savefig(str(run)+'-noise-debug.pdf',dpi=450)
   plt.close()



def count_sim_obj():
   numLAEsim = d13.tolist().count('LAE')
   numOIIsim = d13.tolist().count('OII')

   wl_obs = d05
   LAEsim0,LAEsim1,LAEsim2,LAEsim3,LAEsim4 = 0,0,0,0,0
   OIIsim0,OIIsim1,OIIsim2,OIIsim3,OIIsim4 = 0,0,0,0,0

   for i in range(numLAEsim):
      if wl_obs[i] > 3500 and wl_obs[i] < 3700:
         LAEsim0 += 1
         LAEsim1 += 1
      elif wl_obs[i] > 3700 and wl_obs[i] < 3900:
         LAEsim0 += 1
         LAEsim2 += 1
      elif wl_obs[i] > 3900 and wl_obs[i] < 4100:
         LAEsim0 += 1
         LAEsim3 += 1
      elif wl_obs[i] > 4100 and wl_obs[i] < 4300:
         LAEsim0 += 1
         LAEsim4 += 1

   for i in range(numLAEsim,len(d13)):
      if wl_obs[i] > 3500 and wl_obs[i] < 3700:
         OIIsim0 += 1
         OIIsim1 += 1
      elif wl_obs[i] > 3700 and wl_obs[i] < 3900:
         OIIsim0 += 1
         OIIsim2 += 1
      elif wl_obs[i] > 3900 and wl_obs[i] < 4100:
         OIIsim0 += 1
         OIIsim3 += 1
      elif wl_obs[i] > 4100 and wl_obs[i] < 4300:
         OIIsim0 += 1
         OIIsim4 += 1

   print('*** 1.88 < z < 2.54 ***')
   print('LAEs simulated = '+str(10*LAEsim0))
   print('OIIs simulated = '+str(10*OIIsim0))
   print('')
   print('*** 1.88 < z < 2.04 ***')
   print('LAEs simulated = '+str(10*LAEsim1))
   print('OIIs simulated = '+str(10*OIIsim1))
   print('')
   print('*** 2.04 < z < 2.21 ***')
   print('LAEs simulated = '+str(10*LAEsim2))
   print('OIIs simulated = '+str(10*OIIsim2))
   print('')
   print('*** 2.21 < z < 2.37 ***')
   print('LAEs simulated = '+str(10*LAEsim3))
   print('OIIs simulated = '+str(10*OIIsim3))
   print('')
   print('*** 2.37 < z < 2.54 ***')
   print('LAEs simulated = '+str(10*LAEsim4))
   print('OIIs simulated = '+str(10*OIIsim4))
   print('')



def z_LAE(wl_obs):
   return round(wl_obs/1215.668-1,2)



def contam_and_incomp_ew20(run,ver_sigma_dA):
   t0 = time.time()
   
   global scale,c13,c11,c05
   
   objtype = c13.tolist()
   ew_inf = c11
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')
   
   wl_obs = c05
   wlbin = []
   for i in range(len(c13)):
      if wl_obs[i] < 3700: wlbin.append(0)
      elif wl_obs[i] >= 3700 and wl_obs[i] < 3900: wlbin.append(1)         ## no [O II] interlopers at wl < 3727
      elif wl_obs[i] >= 3900 and wl_obs[i] < 4100: wlbin.append(2)
      elif wl_obs[i] >= 4100 and wl_obs[i] < 4300: wlbin.append(3)
      elif wl_obs[i] >= 4300 and wl_obs[i] < 4500: wlbin.append(4)
      elif wl_obs[i] >= 4500 and wl_obs[i] < 4700: wlbin.append(5)
      elif wl_obs[i] >= 4700 and wl_obs[i] < 4900: wlbin.append(6)
      elif wl_obs[i] >= 4900 and wl_obs[i] < 5100: wlbin.append(7)
      elif wl_obs[i] >= 5100 and wl_obs[i] < 5300: wlbin.append(8)
      elif wl_obs[i] >= 5300: wlbin.append(9)

   global trueLAEcount, classLAEcount
   trueLAEcount = [0]*10
   classLAEcount = [0]*10
   contamCount = [0]*10
   incompCount = [0]*10

   for i in range(len(c13)):
      if objtype[i] == 'LAE': trueLAEcount[wlbin[i]] += 1
      if ew_inf[i] > 20: classLAEcount[wlbin[i]] += 1
      
      if ew_inf[i] > 20 and objtype[i] == 'OII': contamCount[wlbin[i]] += 1
      
      if ew_inf[i] <= 20 and objtype[i] == 'LAE' and wl_obs[i] > 3727.45: incompCount[wlbin[i]] += 1
   
   global contamFrac, incompFrac
   contamFrac = []
   incompFrac = []
   
   for i in range(max(wlbin)+1):
      if classLAEcount[i] == 0: contamFrac.append(0.)
      else: contamFrac.append(contamCount[i]/float(classLAEcount[i]))
      
      if trueLAEcount[i] == 0: incompFrac.append(0.)
      else: incompFrac.append(incompCount[i]/float(trueLAEcount[i]))
   
   contamFrac = np.array(contamFrac)

   incompFrac = np.array(incompFrac)
   
   out_file = open(path+str(run)+'_bin_ew20.dat','w')
   out_file.write(''+'\n')
   out_file.write('run name = '+str(run)+'\n')
   out_file.write('******'+'\n')
   out_file.write('EW_inf >= 20 angstroms'+'\n')
   out_file.write('******'+'\n')
   out_file.write(''+'\n')
   out_file.write('=== first bin (3500-3700 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[0])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[0])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[0])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[0])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== second bin (3700-3900 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[1])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[1])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[1])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[1])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== third bin (3900-4100 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[2])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[2])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[2])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[2])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== fourth bin (4100-4300 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[3])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[3])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[3])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[3])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== fifth bin (4300-4500 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[4])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[4])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[4])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[4])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== sixth bin (4500-4700 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[5])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[5])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[5])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[5])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== seventh bin (4700-4900 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[6])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[6])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[6])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[6])+'\n')
   out_file.write('')
   out_file.write('=== eighth bin (4900-5100 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[7])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[7])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[7])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[7])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== ninth bin (5100-5300 angstrom) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[8])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[8])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[8])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[8])+'\n')
   out_file.write(''+'\n')
   out_file.write('=== tenth bin (5300-5500 angstroms) ==='+'\n')
   out_file.write('  %.9f percent false positives (contamination)' % (100*contamFrac[9])+'\n')
   out_file.write('    out of %.0f "classified" LAEs in bin' % (classLAEcount[9])+'\n')
   out_file.write('  %.9f percent LAEs lost (incompleteness)' % (100*incompFrac[9])+'\n')
   out_file.write('    out of %.0f "true" LAEs in bin' % (trueLAEcount[9])+'\n')
   out_file.write(''+'\n')
   out_file.close()

   baserun = run[:len(run)-12]

   sigmaDA,sigmaDA_bin = [],[]
   prtContamFrac,prtClassLAEcount,prtIncompFrac,prtTrueLAEcount,prtCntLAErecovered = [0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]
   
   cntLAErecovered = []
   for i in range(max(wlbin)+1):
      cntLAErecovered.append(classLAEcount[i] - contamCount[i])
      #sigmaDA_thisbin = np.sqrt( (contamFrac[i]/0.025)**2 + (270000*scale)/cntLAErecovered[i] )
      #sigmaDA_bin.append(sigmaDA_thisbin)
   
   #foo = 0
   for i in range(4):
      #foo += ((sigmaDA_bin[i])**2)**-1
      prtContamFrac[0] += contamFrac[i]*classLAEcount[i]
      prtClassLAEcount[0] += classLAEcount[i]
      prtIncompFrac[0] += incompFrac[i]*trueLAEcount[i]
      prtTrueLAEcount[0] += trueLAEcount[i]
      prtCntLAErecovered[0] += cntLAErecovered[i]
   #sigmaDA.append(np.sqrt(foo**-1))                           ### (11-26-14) Skype call w/ EG and VA: use sigma_bin formula for 1.9 < z < 2.5
   prtContamFrac[0] = prtContamFrac[0]/prtClassLAEcount[0]
   prtIncompFrac[0] = prtIncompFrac[0]/prtTrueLAEcount[0]
   sigmaDA.append(sda.sigma_dA(prtContamFrac[0],prtIncompFrac[0],prtCntLAErecovered[0],scale,0,ver_sigma_dA,baserun))
   #sigmaDA.append( np.sqrt( (prtContamFrac[0]/0.025)**2 + (270000*scale)/prtCntLAErecovered[0] ) )
   ### (04-22-15) replaced by previous line with option to use 'old' or 'new' formula

   #bar = 0
   #for i in range(10):
   for i in range(4,10):                                       ### (12-03-14) telecon with PSU and JJF: second bin is 2.5 < z < 3.5
      #bar += ((sigmaDA_bin[i])**2)**-1
      prtContamFrac[1] += contamFrac[i]*classLAEcount[i]
      prtClassLAEcount[1] += classLAEcount[i]
      prtIncompFrac[1] += incompFrac[i]*trueLAEcount[i]
      prtTrueLAEcount[1] += trueLAEcount[i]
      prtCntLAErecovered[1] += cntLAErecovered[i]
   #sigmaDA.append(np.sqrt(bar**-1))                           ### (11-26-14) Skype call w/ EG and VA: use sigma_bin formula for 1.9 < z < 2.5
   prtContamFrac[1] = prtContamFrac[1]/prtClassLAEcount[1]
   prtIncompFrac[1] = prtIncompFrac[1]/prtTrueLAEcount[1]
   sigmaDA.append(sda.sigma_dA(prtContamFrac[1],prtIncompFrac[1],prtCntLAErecovered[1],scale,1,ver_sigma_dA,baserun))
   #sigmaDA.append( np.sqrt( (prtContamFrac[1]/0.05)**2 + (360000*scale)/prtCntLAErecovered[1] ) )
   ### (02-27-15) Skype call with EG and VA: modified formula for high-z bin
   ### (04-22-15) replaced by previous line with option to use 'old' or 'new' formula


   for i in range(10):
      prtContamFrac[2] += contamFrac[i]*classLAEcount[i]
      prtClassLAEcount[2] += classLAEcount[i]
      prtIncompFrac[2] += incompFrac[i]*trueLAEcount[i]
      prtTrueLAEcount[2] += trueLAEcount[i]
      prtCntLAErecovered[2] += cntLAErecovered[i]
   prtContamFrac[2] = prtContamFrac[2]/prtClassLAEcount[2]
   prtIncompFrac[2] = prtIncompFrac[2]/prtTrueLAEcount[2]
   sigmaDA.append(sda.sigma_dA(prtContamFrac[2],prtIncompFrac[2],prtCntLAErecovered[2],scale,2,ver_sigma_dA,baserun))

   ### each output is a length-3 list
   return np.array(sigmaDA), np.array(prtContamFrac), np.array(prtClassLAEcount)/scale, np.array(prtIncompFrac), np.array(prtTrueLAEcount)/scale, np.array(prtCntLAErecovered)/scale


