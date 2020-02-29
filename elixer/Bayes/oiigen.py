'''
   Last updated: 9 Sep 2015
   
   Andrew Leung
   Rutgers University
   
   [O II] emitters simulation
     - Luminosity function from Ciardullo+ (2013)
     - Possible forms of equivalent width distribution:
         exponential
         lognormal
     - HETDEX survey design (1/4.5 fill factor)
   
'''

from pylab import *
import random
import time
import bisect
import numpy as np
import collections
from cosmolopy import distance as cd
import matplotlib.pyplot as plt
import mpmath as mpm
import fit_lognorm as fln

from numpy.random import rand


def redshift_bins_vol():         ## define centers and upper and lower limits of redshift bins for calculation of cosmological volume and object count
   
   z_int = 5e-4   #1e-4
   v_binll,v_binul = [0.],[0.+z_int]
   
   while v_binll[len(v_binll)-1]+z_int <= 0.476:
      v_binll.append(float(round(v_binll[len(v_binll)-1]+z_int,int(np.log10(1/z_int))+1)))
      v_binul.append(float(round(v_binul[len(v_binul)-1]+z_int,int(np.log10(1/z_int))+1)))
   
   v_bin = 0.5 * (np.array(v_binll)+np.array(v_binul))
   return v_bin,v_binll,v_binul


def redshift_bins_sim():         ## define centers and upper and lower limits of redshift bins for simulation of line luminosity and equivalent width
   
   z_int = 5e-3
   ## (12-03-14) was 0.01, tried smaller
   ## (12-11-14) remember lesson about convergence of riemann sum
   binll,binul = [0.],[0.+z_int]
   
   while binll[len(binll)-1]+z_int <= 0.476:
      binll.append(float(round(binll[len(binll)-1]+z_int,int(np.log10(1/z_int))+2)))
      binul.append(float(round(binul[len(binul)-1]+z_int,int(np.log10(1/z_int))+2)))

   bin = 0.5 * (np.array(binll)+np.array(binul))
   return bin,binll,binul
   


def comoving_volume(z,cosmo):      ## function returns comoving volume out to specified redshift in Mpc^3
   Vc = cd.comoving_volume(z, **cosmo)
   return Vc


def bin_volume(sky_area,v_bin,v_binll,v_binul,cosmo):
   t0 = time.time()
   
   dV = [0]*len(v_bin)
   for i in range(len(v_bin)):
      dV[i] = sky_area*(4*180**2*(math.pi)**-1)**-1 * (comoving_volume(v_binul[i],cosmo)-comoving_volume(v_binll[i],cosmo))
      
      if i%10000 == 0: print('simOII().bin_volume(), finished calculating for '+str(i+1)+'st bin: %.2f seconds' % (time.time()-t0))
   print('simOII().bin_volume() required %.2f seconds' % (time.time()-t0))

   return dV


def object_count(scale,fillFactor,v_bin,dV,OII_case):
   ## numbers of objects to be generated in each bin
   t0 = time.time()
   
   v_phiStar,v_logLStar,v_LStar,v_w_0 = [],[],[],[]
   for i in range(len(v_bin)):
      x = phiStarFactor(v_bin[i],OII_case)
      y = LStarExt(v_bin[i],OII_case)
      v_phiStar.append(x)
      v_logLStar.append(y)
      v_LStar.append(10**y)
      v_w_0.append(w_0Ext(v_bin[i],OII_case))
   
   alpha = -1.2
   L_min = 10**38.5   #1e39
   
   v_rawCDF = []
   for i in range(len(v_bin)):
      v_rawCDF.append( v_phiStar[i] * float(mpm.gammainc(alpha+1,L_min/v_LStar[i]) ) )
      ### same thing; chain rule + recusive definition of gamma function (15-08-17)
      #v_rawCDF.append( v_phiStar[i] * (1/(alpha+1)) * float(mpm.gammainc(alpha+2,L_min/v_LStar[i])-(L_min/v_LStar[i])**(alpha+1)*exp(-L_min/v_LStar[i]) ) )
      
      if i%10000 == 0: print('simOII().object_count(), finished calculating LF integral for '+str(i+1)+'st bin: %.2f seconds' % (time.time()-t0))
   
   v_N = []
   for i in range(len(v_bin)):
      ### EW < 5A excluded from EW distribution fits in Ciardullo+13 (15-08-17)
      v_N.append(v_rawCDF[i] * dV[i] * scale * fillFactor)   #/ exp(-5/v_w_0[i]) )
   
   print('simOII().object_count() required %.2f seconds' % (time.time()-t0))
   return v_N



### no longer specifying an absolute number to generate; rather, relative numbers of objects to be generated in each redshift bin according to comoving volume differential, and must be consistent with number of generated LAEs

def getRedshift(v_N,v_bin,v_binll,v_binul):
   t0 = time.time()
   
   x = []
   for i in range(len(v_bin)):
      x = np.append(x, v_binll[i] + (v_binul[i]-v_binll[i]) * rand(v_N[i]) )
      x = np.round(x, 8)
      
      if i%50 == 0: print('simOII().getRedshift(), loop for '+str(i+1)+'st volume bin: %.2f seconds' % (time.time()-t0))
   
   return np.array(x)



def binSort(z,bin,binll,binul):         ## returns index number corresponding to selected bin ('bin 0' has no [O II] emitters)
   
   x = -999
   for i in range(len(bin)):
      if z >= binll[i] and z < binul[i]: x = i
   return x



def phiStarFactor(z,case):      ### (as of 09-09-15) actually returns phiStar
   '''
      linear inter/extrapolation to generate phiStar normalization factors that is equivalent to the cumulative density for log L > 40.5 (ergs/sec) given in Ciardullo 2013 (Table 1)
      
   '''
   
   bin_ci13 = [0.1000,0.2625,0.3875,0.5050]
   logPhi405 = [-2.30,-2.12,-2.07,-2.07]
   
   if case == 'base':
      errbar = [0.,0.,0.,0.]
   elif case == 'low':         ### best case for LAE cosmology
      errbar = [-0.11,-0.06,-0.05,-0.08]
   elif case == 'high':         ### worst case for LAE cosmology
      errbar = [0.09,0.05,0.04,0.03]
   else:
      errbar = [0.,0.,0.,0.]

   logPhi405 = np.array(logPhi405) + np.array(errbar)

   if z <= bin_ci13[1]: i = 0
   elif z > bin_ci13[1] and z <= bin_ci13[2]: i = 1
   elif z > bin_ci13[2]: i = 2
   else: print('oiigen.phiStarFactor(): bin assignment error')
   
   x = logPhi405[i] + (logPhi405[i+1]-logPhi405[i]) / (bin_ci13[i+1]-bin_ci13[i]) * (z-bin_ci13[i])
   
   return float(10**x / mpm.gammainc(-1.2+1,10**40.5/10**LStarExt(z,case)))
   


def LStarExt(z,case):
   '''
      linear inter/extrapolation of log(L*) given in Ciardullo 2013 for 
         bin1 is 0.000 < z < 0.200, average value of lower and upper limits is 0.10
         bin2 is 0.200 < z < 0.325, average value 0.2625
         bin3 is 0.325 < z < 0.450, average value 0.3875
         bin4 is 0.450 < z < 0.560, average value 0.5050
   
      at z < 0.0512, an [O II] emitting galaxy that is 5 kpc across in size will subtend more than 5 arcseconds and be resolved. therefore, we will not simulate z < 0.0512 [O II] emitters since they will not be mistakened for LAEs
      
      z > 0.476 [O II] emitters fall outside of HETDEX's spectral range, not simulated
      
   '''
   
   bin_ci13 = [0.1000,0.2625,0.3875,0.5050]
   logLStar = [41.07,41.29,41.50,41.68]
   
   if case == 'base':
      errbar = [0.,0.,0.,0.]
   elif case == 'low':         ### best case for LAE cosmology
      errbar = [-0.16,-0.11,-0.10,-0.12]
   elif case == 'high':         ### worst case for LAE cosmology
      errbar = [0.18,0.11,0.08,0.08]
   else:
      errbar = [0.,0.,0.,0.]

   logLStar = np.array(logLStar) + np.array(errbar)
   
   if z <= bin_ci13[1]: i = 0
   elif z > bin_ci13[1] and z <= bin_ci13[2]: i = 1
   elif z > bin_ci13[2]: i = 2
   else: print('oiigen.LStarExt(): bin assignment error')

   x = logLStar[i] + (logLStar[i+1]-logLStar[i]) / (bin_ci13[i+1]-bin_ci13[i]) * (z-bin_ci13[i])
   return float(x)



def LF(case,L_list,thisbin):               ## returns i by j matrix (actually i lists of size j); i redshift bins and j luminosity intervals
   t0 = time.time()
   alpha = -1.2
   
   phiStar = phiStarFactor(thisbin,case)
   LStar = 10**(LStarExt(thisbin,case))
   LOverLStar = L_list/LStar
   
   rawCDF = []
   for L in range(len(L_list)):
      rawCDF.append( phiStar * ( float(mpm.gammainc(alpha+1,LOverLStar[0])) - float(mpm.gammainc(alpha+1,LOverLStar[L])) ) )
      ### same thing; chain rule + recusive definition of gamma function (15-08-17)
      #rawCDF.append( phiStar * (1/(alpha+1)) * ( float(mpm.gammainc(alpha+2,LOverLStar[0])-(LOverLStar[0])**(alpha+1)*exp(-LOverLStar[0])) - float(mpm.gammainc(alpha+2,LOverLStar[L])-(LOverLStar[L])**(alpha+1)*exp(-LOverLStar[L])) ) )
   
   normalization = (rawCDF[len(rawCDF)-1])**(-1)
   normalizedCDF = normalization * np.array(rawCDF)

   return normalizedCDF.tolist()



### here we have to modify to generate objects according to number density (per comoving volume) given by the Schechter function, replacing the arbitrarily specified N

def getLuminosity(case,bin,N):
   t0 = time.time()
   
   logLmin, logLmax = 38.5, 43.0
   L_list = 10**(np.linspace(logLmin,logLmax,181))
   thisList = []

   for i in range(len(bin)):
      chosenLF = LF(case,L_list,bin[i])   ###(05-25-15)
      #chosenLF = LFTable[i]      ## luminosity function corresponding to the redshift bin (actually the cumulative distribution function for luminosity from 2E42 to 1E44 ergs/sec)
      R = rand(N[i])                           ## R is a numpy.array; not sure why it is more efficient (discovered experimentally) to do this here [as opposed to making N calls to random.random()] but not in the case of the simple getRedshift() and getSN()
      
      for j in range(int(N[i])):
         r = R[j]                           ## random number between 0 and 1 used for this instance of the loop
         
         thisList.append(np.interp(r,chosenLF,L_list))
      
      if i%10 == 0: print('simOII.getLuminosity(), finished loop '+str(i+1)+': %.2f seconds' % (time.time()-t0))

   print('getLuminosity() required %.2f seconds wall time' % (time.time()-t0))
   return np.array(thisList)
                     

                     
def w_0Ext(z,case):      ## linear inter/extrapolation of w_0 given in Ciardullo 2013 (Table 1)
   bin_ci13 = [0.1000,0.2625,0.3875,0.5050]
   _w_0 = [8.0,11.5,16.6,21.5]
   
   errbar = [1.6,1.6,2.4,3.3]
   if case == 'low':
      ### best case for LAE cosmology (small w_0 leads to large correction factor for simulated [OII] counts, but 5-sigma counts should end up being be lower)
      _w_0 = np.array(_w_0) - np.array(errbar)
   elif case == 'high':
      ### worst case for LAE cosmology (large w_0 leads to small correction factor for simulated [OII] counts, but 5-sigma counts should end up being be higher)
      _w_0 = np.array(_w_0) + np.array(errbar)

   if z <= bin_ci13[1]: i = 0
   elif z > bin_ci13[1] and z <= bin_ci13[2]: i = 1
   elif z > bin_ci13[2]: i = 2
   else: print('oiigen.w_0Ext(): bin assignment error')

   x = _w_0[i] + (_w_0[i+1]-_w_0[i]) / (bin_ci13[i+1]-bin_ci13[i]) * (z-bin_ci13[i])
   return float(x)      # in angstroms



'''
   simulate equivalent width (rest frame)

'''
def EW(case,w_list,thisbin):         ## returns i by j matrix (actually i lists of size j); i redshift bins and j wavelength intervals
   t0 = time.time()
   rawCDF = []

   if not case[:9] == 'lognormal':
      w_0 = w_0Ext(thisbin,case)
   
      for w in range(len(w_list)):
         rawCDF.append( np.exp((-1)*w_list[w]/w_0) )

      normalizedCDF = 1-np.array(rawCDF)
   
   else:
      if case == 'lognormal_EW':        EW_case = 'base'
      elif case == 'lognormal_EW_low':  EW_case = 'low'
      elif case == 'lognormal_EW_high': EW_case = 'high'
      fitParam, fitCovar = fln.run(thisbin,EW_case)               ###(08-18-15)
      W_0, sigma = fitParam[0], fitParam[1]
         
      for w in range(len(w_list)):
         rawCDF.append( sum(rawCDF) + fln.lognorm_pdf_mod1_BL00(w_list[w],W_0,sigma) )

      normalization = (rawCDF[len(rawCDF)-1])**(-1)
      normalizedCDF = normalization * np.array(rawCDF)
   
   return normalizedCDF.tolist()



def getEW(case,bin,N):
   t0 = time.time()
   
   logwmin, logwmax = -3., 3.
   w_list = 10**(np.linspace(logwmin,logwmax,121))
   thisList = []
   
   for i in range(len(bin)):
      
      if case == 'lognormal_EW' or case == 'lognormal_EW_low' or case == 'lognormal_EW_high':
         if case == 'lognormal_EW':
            fitParam, fitCovar = fln.run(bin[i],'base')               ###(08-18-15)
         elif case == 'lognormal_EW_low':
            fitParam, fitCovar = fln.run(bin[i],'low')
         elif case == 'lognormal_EW_high':
            fitParam, fitCovar = fln.run(bin[i],'high')

         W_0, sigma = fitParam[0], fitParam[1]
         mu = np.log(W_0)
         
         for j in range(int(N[i])):
            thisList.append(np.random.lognormal(mu,sigma))

      else:
         chosenDist = EW(case,w_list,bin[i])      ###(05-25-15)
         #chosenDist = EWTable[i]     ## cumulative distribution function for EW from 20 to 500 angstroms corresponding to redshift bin
         R = rand(N[i])
         
         for j in range(int(N[i])):
            r = R[j]                              ## random number between 0 and 1 used for this instance of the loop
            
            thisList.append(np.interp(r,chosenDist,w_list))
      
      if i%10 == 0: print('simOII.getEW(), finished loop '+str(i+1)+': %.2f seconds' % (time.time()-t0))

   print('getEW() required %.2f seconds wall time' % (time.time()-t0))
   return np.array(thisList)



'''
   simulate color
   
'''
g_r_mean = 0.787817129993
g_r_stdev = 0.305763193382

g_i_mean = 1.17335352866
g_i_stdev = 0.420496614516

g_z_mean = 1.35090342418
g_z_stdev = 0.518779035927

r_i_mean = 0.354144442043
r_i_stdev = 0.168842129176

r_z_mean = 0.532128339139
r_z_stdev = 0.279919009704

i_z_mean = 0.161533452684
i_z_stdev = 0.129429958359


def getColor(realizedZ):
   t0 = time.time()
   
   g_r, g_i, g_z, r_i, r_z, i_z = [],[],[],[],[],[]
   
   for i in range(len(realizedZ)):
      g_r.append(random.gauss(g_r_mean,g_r_stdev))
      g_i.append(random.gauss(g_i_mean,g_i_stdev))
      g_z.append(random.gauss(g_z_mean,g_z_stdev))
      r_i.append(random.gauss(r_i_mean,r_i_stdev))
      r_z.append(random.gauss(r_z_mean,r_z_stdev))
      i_z.append(random.gauss(i_z_mean,i_z_stdev))
   
      if i%(1e5) == 0: print('simOII().getColor(), finished simulating color for '+str(i+1)+'st object: %.2f seconds' % (time.time()-t0))
   
   return np.array(g_r), np.array(g_i), np.array(g_z), np.array(r_i), np.array(r_z), np.array(i_z)



'''
   calculate observed wavelength

'''
def getLambda(realizedZ):
   return 3727.45*(1+realizedZ)



def simOII(scale,case,sky_area,cosmo):                     ## simulate OII population
   t0 = time.time()
   
   v_bin,v_binll,v_binul = redshift_bins_vol()
   print('simOII().redshift_bins_vol(), finished: %.2f seconds' % (time.time()-t0))
   
   dV = bin_volume(sky_area,v_bin,v_binll,v_binul,cosmo)
   print('simOII().bin_volume(), finished: %.2f seconds' % (time.time()-t0))
   
   fillFactor = 1/4.5
   v_N = object_count(scale,fillFactor,v_bin,dV,case)
   print('simOII().object_count(), finished: %.2f seconds' % (time.time()-t0))

   realizedZ = getRedshift(v_N,v_bin,v_binll,v_binul)
   print('simOII().getRedshift(), finished: %.2f seconds' % (time.time()-t0))

   v_bin,v_binll,v_binul,dV,v_N = [],[],[],[],[]

   bin,binll,binul = redshift_bins_sim()
   
   objBin = []
   for i in range(len(realizedZ)):
      objBin.append(binSort(realizedZ[i],bin,binll,binul))
      if i%500000 == 0: print('simOII(), assigning redshift bin, finished '+str(i+1)+'st loop: %.2f seconds' % (time.time()-t0))
   
   N = []
   for i in range(len(bin)):
      N.append(objBin.count(i))
      if i%10 == 0: print('simOII(), computing number count for simulation redshift bin no. '+str(i+1)+': %.2f seconds' % (time.time()-t0))
   
   realizedWL = getLambda(realizedZ)             ## in angstroms; function of redshift

   #LFTable = LF()   ###(05-25-15)
   realizedL = getLuminosity(case,bin,N)         ## in ergs per second

   #EWTable = EW()   ###(05-25-15)
   realizedEW = getEW(case,bin,N)                ## in angstroms

   realizedg_r,realizedg_i,realizedg_z,realizedr_i,realizedr_z,realizedi_z = getColor(realizedZ)         ## color indices
   
   realizedSNr = []   #getSNr()
   realizedRA = []    #getRA()
   realizedDec = []   #getDec()

   labels = '[realizedRA,realizedDec,realizedZ,objBin,realizedL,realizedEW,realizedWL,realizedSNr,realizedg_r,realizedg_i,realizedg_z,realizedr_i,realizedr_z,realizedi_z]'

   print('simOII() required %.2f seconds wall time' % (time.time()-t0))
   print(str(len(realizedZ))+' [O II]s simulated')
   return [np.array(realizedRA),np.array(realizedDec),realizedZ,np.array(objBin),realizedL,realizedEW,realizedWL,realizedSNr,realizedg_r,realizedg_i,realizedg_z,realizedr_i,realizedr_z,realizedi_z]


