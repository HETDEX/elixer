'''
   Last updated: 22 Aug 2015
   
   Andrew Leung
   Rutgers University
   
   LAE simulation
     - Possible luminosity functions:
         Gronwall+  (2016)
         Ciardullo+ (2012)
         Gronwall+  (2007)
     - HETDEX survey design (1/4.5 fill factor)

'''

from pylab import *
import random
import time
import bisect
import numpy as np
import collections
from cosmolopy import distance as cd
import mpmath as mpm
import gc
from numpy.random import rand



def redshift_bins_vol():         ## define centers and upper and lower limits of redshift bins for calculation of cosmological volume and object count

   z_int = 5e-4      #1e-3      #1e-4      #1e-5
   v_bin,v_binll,v_binul = [],[1.878],[1.878+z_int]
   
   while v_binll[len(v_binll)-1]+z_int <= 3.525:
      v_binll.append(float(round(v_binll[len(v_binll)-1]+z_int,int(np.log10(1/z_int))+1)))
      v_binul.append(float(round(v_binul[len(v_binul)-1]+z_int,int(np.log10(1/z_int))+1)))
   #v_binul[len(v_binul)-1] = 3.525

   #for i in range(len(v_binll)):
      #v_bin.append(float(round(0.5*(v_binll[i]+v_binul[i]),int(np.log10(1/z_int))+2)))
   v_bin = 0.5 * (np.array(v_binll)+np.array(v_binul))
   return v_bin,v_binll,v_binul


def redshift_bins_sim():         ## define centers and upper and lower limits of redshift bins for simulation of line luminosity and equivalent width
   #global bin,binll,binul

   z_int = 1e-2      ## (12-03-14) #0.025
   bin,binll,binul = [],[1.878],[1.878+z_int]
   
   while binll[len(binll)-1]+z_int <= 3.525:
      binll.append(float(round(binll[len(binll)-1]+z_int,int(np.log10(1/z_int))+1)))
      binul.append(float(round(binul[len(binul)-1]+z_int,int(np.log10(1/z_int))+1)))

   bin = 0.5 * (np.array(binll)+np.array(binul))
   return bin,binll,binul


def binSort(z,bin,binll,binul):            ## returns index number corresponding to selected bin
   
   x = -999
   for i in range(len(bin)):
      if z >= binll[i] and z < binul[i]: x = i
   return x



def comoving_volume(z,cosmo):      ## function returns comoving volume out to specified redshift in Mpc^3
   Vc = cd.comoving_volume(z, **cosmo)
   return Vc


def bin_volume(sky_area,v_bin,v_binll,v_binul,cosmo):
   t0 = time.time()
   
   dV = [0]*len(v_bin)
   for i in range(len(v_bin)):
      dV[i] = sky_area*(4*180**2*(math.pi)**-1)**-1 * (comoving_volume(v_binul[i],cosmo)-comoving_volume(v_binll[i],cosmo))
      
      if i%10000 == 0: print('simLAE().bin_volume(), finished calculating for '+str(i+1)+'st bin: %.2f seconds' % (time.time()-t0))
   print('simLAE().bin_volume() required %.2f seconds' % (time.time()-t0))
   return dV


def object_count(scale,fillFactor,v_bin,dV,LAE_case):
   ## numbers of objects to be generated in each bin
   t0 = time.time()
   
   v_logPhiStar,v_phiStar,v_logLStar,v_LStar,v_w_0 = [],[],[],[],[]
   for i in range(len(v_bin)):
      x = phiStarExt(v_bin[i],LAE_case)
      y = LStarExt(v_bin[i],LAE_case)
      v_phiStar.append(x)
      v_logLStar.append(y)
      v_LStar.append(10**y)
      v_w_0.append(w_0Ext(v_bin[i],LAE_case))

   if LAE_case == 2: alpha = -1.36
   elif LAE_case == 3: alpha = -1.5
   else: alpha = -1.65
   L_min = 1e41
   
   
   '''
      object count 'inflated' by the fraction of simulated LAEs that would have EW < 20 angstroms, since Lyman-alpha luminosity functions are measured for LAEs that have EW > 20 angstroms
      
   '''

   #global v_rawCDF
   v_rawCDF = []
   for i in range(len(v_bin)):
      v_rawCDF.append( v_phiStar[i] * float(mpm.gammainc(alpha+1,L_min/v_LStar[i]) ) )
      ### same thing; chain rule + recusive definition of gamma function (15-08-17)
      #v_rawCDF.append( v_phiStar[i] * (1/(alpha+1)) * float(mpm.gammainc(alpha+2,L_min/v_LStar[i])-(L_min/v_LStar[i])**(alpha+1)*exp(-L_min/v_LStar[i]) ) )
   
      if i%10000 == 0: print('simLAE().object_count(), finished calculating LF integral for '+str(i+1)+'st bin: %.2f seconds' % (time.time()-t0))

   v_N = []
   for i in range(len(v_bin)):
      v_N.append(int(round( v_rawCDF[i] * dV[i] * scale * fillFactor / exp(-20/v_w_0[i]) )))

   print('simLAE().object_count() required %.2f seconds' % (time.time()-t0))
   return v_N



def getRedshift(v_N,v_bin,v_binll,v_binul):
   t0 = time.time()
   
   x = []
   for i in range(len(v_bin)):
      x = np.append(x, v_binll[i] + (v_binul[i]-v_binll[i]) * rand(v_N[i]) )
      x = np.round(x, 8)

      if i%20 == 0:
         print('simLAE.getRedshift(), \n ### loop for '+str(i+1)+'st volume bin: %.2f seconds' % (time.time()-t0))
         print(' ### '+'{:,}'.format(v_N[i])+' LAEs in '+str(v_binll[i])+' < z < '+str(v_binul[i]))
         print(' ### current clock time: '+time.strftime("%a, %d %b %Y %H:%M:%S",time.localtime())+'\n')
      gc.collect()

   return np.array(x)
   


'''
   simulate line flux

'''
def phiStarExt(z,case):
   ## linear inter/extrapolation of log(phi*) given in Table 5 of Ciardullo 2011 for z=3.1 and z=2.1
   
   if case == 1:
      ## z = 2.1 and z = 3.1 from Ciardullo+ 2012, inter/extrapolated for 1.9 < z < 3.5
      logPhiStar31 = -3.17
      logPhiStar21 = -2.86
   
   elif case == 2:
      ## z = 3.1 parameters from Gronwall+ 2007 applied to 1.9 < z < 3.5
      logPhiStar31 = np.log10(1.28e-3)
      logPhiStar21 = np.log10(1.28e-3)
   
   elif case == 3:
      ## z = 3.1 parameters from Ouchi+ 2008 applied to 1.9 < z < 3.5
      logPhiStar31 = np.log10(9.2e-4)
      logPhiStar21 = np.log10(9.2e-4)
      
   elif case == 4:
      ## z = 3.1 parameters from Ciardullo+ 2012 applied to 1.9 < z < 3.5
      logPhiStar31 = -3.155
      logPhiStar21 = -3.155
   
   elif case == 5:
      ## z = 2.1 parameters from Ciardullo+ 2012 applied to 1.9 < z < 3.5
      logPhiStar31 = -2.86
      logPhiStar21 = -2.86
   
   elif case == 6:
      ## z = 2.1 and z = 3.1 from Gronwall+ 2015, inter/extrapolated for 1.9 < z < 3.5
      logPhiStar31 = np.log10(10**-2.98)
      logPhiStar21 = np.log10(10**-3.08)
   
   x = logPhiStar21 + (logPhiStar31-logPhiStar21) / (3.104-2.063) * (z-2.063)
   #x = logPhiStar21 + (logPhiStar31-logPhiStar21) / (3.100-2.100) * (z-2.100)

   if case < 6:
      return float(10**x)
      #return float(x)                  ## log(phi*) replaced (15-08-17)
      #return (0.096*z-3.278)            ## Greg's
      
   else:      ### inter/extrapolated parameter actually log(phi_tot)(>1.5e42) (15-08-17)
      numDens = 10**x
      alpha = -1.65
      L_min = 1.5e42
      LStar = 10**LStarExt(z,case)
      y = numDens / float(mpm.gammainc(alpha+1,L_min/LStar))
      return float(y)



def LStarExt(z,case):
   ## linear inter/extrapolation of log(L*) given in Table 5 of Ciardullo 2011 for z=3.1 and z=2.1
   
   if case == 1:
      ## z = 2.1 and z = 3.1 from Ciardullo+ 2012, inter/extrapolated for 1.9 < z < 3.5
      logLStar31 = 42.76
      logLStar21 = 42.33
   
   elif case == 2:
      ## z = 3.1 parameters from Gronwall+ 2007 applied to 1.9 < z < 3.5
      logLStar31 = 42.66
      logLStar21 = 42.66
   
   elif case == 3:
      ## z = 3.1 parameters from Ouchi+ 2008 applied to 1.9 < z < 3.5
      logLStar31 = np.log10(5.8e42)
      logLStar21 = np.log10(5.8e42)
   
   elif case == 4:
      ## z = 3.1 parameters from Ciardullo+ 2012 applied to 1.9 < z < 3.5
      logLStar31 = 42.76
      logLStar21 = 42.76
   
   elif case == 5:
      ## z = 2.1 parameters from Ciardullo+ 2012 applied to 1.9 < z < 3.5
      logLStar31 = 42.33
      logLStar21 = 42.33
   
   elif case == 6:
      ## z = 2.1 and z = 3.1 from Gronwall+ 2015, inter/extrapolated for 1.9 < z < 3.5
      logLStar31 = 42.77
      logLStar21 = 42.61
   

   x = logLStar21 + (logLStar31-logLStar21) / (3.104-2.063) * (z-2.063)
   #x = logLStar21 + (logLStar31-logLStar21) / (3.100-2.100) * (z-2.100)
   return float(x)

   #return (0.154*z+42.293)         ## Greg's



def LF(case,L_list,thisbin):
   ## returns i by j matrix (actually i lists of size j); i redshift bins and j luminosity intervals
   t0 = time.time()
   
   if case == 1 or case == 4 or case == 5 or case == 6:
      ## Ciardullo+12, Gronwall+15 faint-end slope
      alpha = -1.65
   
   elif case == 2:
      ## Gronwall+07 faint-end slope
      alpha = -1.36
   
   elif case == 3:
      ## Ouchi+08 faint-end slope
      alpha = -1.5
   
   phiStar = phiStarExt(thisbin,case)
   LStar = 10**(LStarExt(thisbin,case))
   LOverLStar = L_list/LStar
   
   rawCDF = []
   for L in range(len(L_list)):
      rawCDF.append( phiStar * ( float(mpm.gammainc(alpha+1,LOverLStar[0])) - float(mpm.gammainc(alpha+1,LOverLStar[L])) ) )
      ### same thing; chain rule + recusive definition of gamma function (15-08-17)
      #rawCDF.append( phiStar * (1/(alpha+1)) * ( float(mpm.gammainc(alpha+2,LOverLStar[0])-(LOverLStar[0])**(alpha+1)*exp(-LOverLStar[0])) - float(mpm.gammainc(alpha+2,LOverLStar[L])-(LOverLStar[L])**(alpha+1)*exp(-LOverLStar[L])) ) )
   
   normalization = (rawCDF[len(rawCDF)-1])**(-1)
   normalizedCDF = normalization * np.array(rawCDF)
   normalizedCDF = normalizedCDF.tolist()
   #normalizedCDF.append(1.0)

   return normalizedCDF



def getLuminosity(case,bin,N):
   t0 = time.time()
   
   logLmin, logLmax = 41.0, 44.0
   L_list = 10**(np.linspace(logLmin,logLmax,121))
   thisList = []
   
   for i in range(len(bin)):
      chosenLF = LF(case,L_list,bin[i])   ###(05-25-15)
      #chosenLF = LFTable[i]      ## luminosity function corresponding to the redshift bin (actually the cumulative distribution function for luminosity from 2E42 to 1E44 ergs/sec)
      R = rand(N[i])                           ## R is a numpy.array; not sure why it is more efficient (discovered experimentally) to do this here [as opposed to making N calls to random.random()] but not in the case of the simple getRedshift() and getSN()
      
      for j in range(0,int(N[i])):
         r = R[j]                           ## random number between 0 and 1 used for this instance of the loop
         
         thisList.append(np.interp(r,chosenLF,L_list))
      
      if i%10 == 0: print('simLAE(), getLuminosity(), finished loop '+str(i+1)+': %.2f seconds' % (time.time()-t0))
      gc.collect()

   print('getLuminosity() required %.2f seconds' % (time.time()-t0))
   return np.array(thisList)



'''
   simulate equivalent width (rest frame)

'''
def w_0Ext(z,case):
   ## returns EWD scale length w_0 in angstroms

   if case == 2:
      ## Gronwall+ 2007
      x = 76.
   
   elif case == 5:
      ## z = 2.1 parameters from Ciardullo+ 2012 applied to 1.9 < z < 3.5
      x = 50.

   else:
      if case == 1:
      ## linear inter/extrapolation of w_0 given in Figure 9 by Ciardullo+ 2012 for z = 2.1 and z = 3.1
         w_031 = 64.
         w_021 = 50.
      elif case == 6:
      ## linear inter/extrapolation of w_0 given in Table 6 by Gronwall+ 2015 for z = 2.1 and z = 3.1
         w_031 = 100.
         w_021 = 50.
      x = w_021 + (w_031-w_021) / (3.104-2.063) * (z-2.063)

   return float(x)


def EW(case,w_list,thisbin):
   ## returns i by j matrix (actually i lists of size j); i redshift bins and j wavelength intervals
   t0 = time.time()
   
   w_0 = w_0Ext(thisbin,case)
   
   eta = []
   for w in range(len(w_list)):
      eta.append( np.exp((-1)*w_list[w]/w_0) )
   
   normalizedCDF = 1-np.array(eta)

   return normalizedCDF



def getEW(case,bin,N):
   t0 = time.time()
   
   logwmin, logwmax = -3., 3.
   w_list = 10**(np.linspace(logwmin,logwmax,121))
   thisList = []
   
   for i in range(len(bin)):
      chosenDist = EW(case,w_list,bin[i])      ###(05-25-15)
      #chosenDist = EWTable[i]   ## cumulative distribution function for EW from 20 to 500 angstroms corresponding to redshift bin
      R = rand(N[i])
      
      for j in range(0,int(N[i])):
         r = R[j]                           ## random number between 0 and 1 used for this instance of the loop
         
         thisList.append(np.interp(r,chosenDist,w_list))
      
      if i%10 == 0: print('simLAE(), getEW(), finished loop '+str(i+1)+': %.2f seconds' % (time.time()-t0))
      gc.collect()

   print('getEW() required %.2f seconds wall time' % (time.time()-t0))
   return np.array(thisList)



'''
   simulate color

'''
g_r_mean = 0.492203039845
g_r_stdev = 0.430859427828

g_i_mean = 0.64475640562
g_i_stdev = 0.539096543074

g_z_mean = 0.635911790846
g_z_stdev = 0.815492546678

r_i_mean = 0.151534169339
r_i_stdev = 0.263085812023

r_z_mean = 0.107353674735
r_z_stdev = 0.551370100988

i_z_mean = -0.0310937402259
i_z_stdev = 0.518062564639


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
   
      if i%(5e5) == 0: print('simLAE().getColor(), finished simulating color for '+str(i+1)+'st object: %.2f seconds' % (time.time()-t0))

   return np.array(g_r), np.array(g_i), np.array(g_z), np.array(r_i), np.array(r_z), np.array(i_z)



'''
   calculate observed wavelength

'''
def getLambda(realizedZ):
   return 1215.668*(1+realizedZ)




def simLAE(scale,case,sky_area,cosmo):                  ## simulate LAE population
   t0 = time.time()

   v_bin,v_binll,v_binul = redshift_bins_vol()
   print('simLAE().redshift_bins_vol(), finished: %.2f seconds' % (time.time()-t0))
   
   dV = bin_volume(sky_area,v_bin,v_binll,v_binul,cosmo)
   print('simLAE().bin_volume(), finished: %.2f seconds' % (time.time()-t0))
   
   fillFactor = 1/4.5
   v_N = object_count(scale,fillFactor,v_bin,dV,case)
   print('simLAE().object_count(), finished: %.2f seconds' % (time.time()-t0))

   realizedZ = getRedshift(v_N,v_bin,v_binll,v_binul)
   print('simLAE().getRedshift(), finished: %.2f seconds' % (time.time()-t0))
   
   v_bin,v_binll,v_binul,dV,v_N = [],[],[],[],[]
   
   bin,binll,binul = redshift_bins_sim()

   objBin = []
   print(len(realizedZ))
   for i in range(len(realizedZ)):
      objBin.append(binSort(realizedZ[i],bin,binll,binul))
      if i%500000 == 0: print('simLAE(), assigning redshift bin, finished '+str(i+1)+'st loop: %.2f seconds' % (time.time()-t0))

   N = []
   for i in range(len(bin)):
      N.append(objBin.count(i))
      print('simLAE(), computing number count for simulation redshift bin no. '+str(i+1)+': %.2f seconds' % (time.time()-t0))

   realizedWL = getLambda(realizedZ)            ## in angstroms; function of redshift

   #LFTable = LF(case)   ###(05-25-15)
   realizedL = getLuminosity(case,bin,N)         ## in ergs per second

   #EWTable = EW()   ###(05-25-15)
   realizedEW = getEW(case,bin,N)               ## in angstroms

   realizedg_r,realizedg_i,realizedg_z,realizedr_i,realizedr_z,realizedi_z = getColor(realizedZ)         ## color indices

   realizedSNr = []    #getSNr()
   realizedRA = []    #getRA()
   realizedDec = []    #getDec()

   labels = '[realizedRA,realizedDec,realizedZ,objBin,realizedL,realizedEW,realizedWL,realizedSNr,realizedg_r,realizedg_i,realizedg_z,realizedr_i,realizedr_z,realizedi_z]'

   print('simLAE() required %.2f seconds wall time' % (time.time()-t0))
   print(str(len(realizedZ))+' LAEs simulated')
   return [np.array(realizedRA),np.array(realizedDec),realizedZ,np.array(objBin),realizedL,realizedEW,realizedWL,realizedSNr,realizedg_r,realizedg_i,realizedg_z,realizedr_i,realizedr_z,realizedi_z]


