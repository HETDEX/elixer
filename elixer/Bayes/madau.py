'''
   Last updated: 4 Sep 2014

   Andrew Leung
   Rutgers University
   
   mktaueff() adapted from Fortran subroutine Madau provided by Viviana Acquaviva
   realizetau() based on Madau (1995)
   
'''

import time
import numpy as np
from pylab import *
from scipy import special as sp



def mktaueff(lambdaobsframe,z):         ### make taueff for equation (14) in Madau (1995)

   y = 1.+z

   thfluxes = 1.
   taueff = 0.

   if ( (1025.71*y<=lambdaobsframe) and (lambdaobsframe<=1215.66*y) ):
      taueff = 0.0036*(lambdaobsframe/1215.66)**3.46

   if ( (972.53*y<=lambdaobsframe) and (lambdaobsframe<=1025.71*y) ):
      taueff = 0.0036*(lambdaobsframe/1215.66)**3.46 +0.0017*(lambdaobsframe/1025.71)**3.46

   if ( (949.73*y<=lambdaobsframe) and (lambdaobsframe<=972.53*y) ):
      taueff = 0.0036*(lambdaobsframe/1215.66)**3.46 +0.0017*(lambdaobsframe/1025.71)**3.46 +0.0012*(lambdaobsframe/972.53)**3.46

   if ( (911.74*y<=lambdaobsframe) and (lambdaobsframe<=949.73*y) ):
      taueff = 0.0036*(lambdaobsframe/1215.66)**3.46 +0.0017*(lambdaobsframe/1025.71)**3.46 +0.0012*(lambdaobsframe/972.53)**3.46 +0.00093*(lambdaobsframe/949.73)**3.46            ### 5n -> 1n

   if ( (911.740<=lambdaobsframe) and (lambdaobsframe<=911.74*y) ):
      x = lambdaobsframe/911.74
      taueff = 0.0036*(lambdaobsframe/1215.66)**3.46 +0.0017*(lambdaobsframe/1025.71)**3.46 +0.0012*(lambdaobsframe/972.53)**3.46 +0.00093*(lambdaobsframe/949.73)**3.46 +0.25*x**3.*(y**0.46-x**0.46) +9.4*x**1.5*(y**0.18-x**0.18) +0.7*x**3.*(y**(-1.32)-x**(-1.32)) -0.023*(y**1.68-x**1.68)

   ### we have assumed no photons with lambda_rest < 912 A can exist in galaxies (galaxies are their own Lyman limit)

   if (lambdaobsframe<=911.740*y):
      taueff = 1e32

   thfluxes = thfluxes*np.exp(-taueff)
   lambdarestframe = lambdaobsframe/y

   return taueff



def realizetau(lambdaobsframe,z):         ### cumulative probability of p(tau) is P(>tau) = erf(0.5*taueff/tau**0.5)
   t0 = time.time()
   
   thislist = []
   r = rand(len(z))

   for i in range(len(z)):                           ### Monte Carlo simulation of optical depth tau realized from cumulative distribution
      taueff = mktaueff(lambdaobsframe,z[i])
      arg = sp.erfinv(r[i])
      tau = (0.5*taueff/arg)**2
      thislist.append(tau)
   
      if i%100000==0: print('madau.realizetau(), '+str(i+1)+'st loop to realize optical depth tau at lambda_EL: %.2f seconds' % (time.time()-t0))

   return thislist





'''
   subroutine Madau(Th,z,writeout)
   ! does not include effect of dimming by distance!
   implicit none
   Type(TheoryFluxes) Th
   
   real z, x, y
   real, dimension(:), allocatable :: negtaueff
   integer i
   logical writeout
   
   allocate(negtaueff(Th%nofluxes))
   
   y = (1.0+z)
   if (writeout) then
   open(20,file='IGMdustcorrectedfluxes.dat')
   end if
   do i = 1, Th%nofluxes
   
   negtaueff(i) = 0. !this is - taueff
   
   if ((1026.0*y.le.(Th%lambdaobsframe(i))).and.((Th%lambdaobsframe(i)).le.1216.0*y)) then
   negtaueff(i) = -0.0036*(Th%lambdaobsframe(i)/1216.0)**3.46
   end if
   
   if (973.0*y.le.(Th%lambdaobsframe(i)).and.((Th%lambdaobsframe(i)).le.1026.0*y)) then
   negtaueff(i) = -0.0036*(Th%lambdaobsframe(i)/1216.0)**3.46 -0.0017*(Th%lambdaobsframe(i)/1026.0)**3.46
   end if
   
   if (950.0*y.le.(Th%lambdaobsframe(i)).and.((Th%lambdaobsframe(i)).le.973.0*y)) then
   negtaueff(i) = -0.0036*(Th%lambdaobsframe(i)/1216.0)**3.46 -0.0017*(Th%lambdaobsframe(i)/1026.0)**3.46 -0.0012*(Th%lambdaobsframe(i)/973.0)**3.46
   end if
   
   if (912.0*y.le.(Th%lambdaobsframe(i)).and.((Th%lambdaobsframe(i)).le.950.0*y)) then
   negtaueff(i) = -0.0036*(Th%lambdaobsframe(i)/1216.0)**3.46 -0.0017*(Th%lambdaobsframe(i)/1026.0)**3.46 -0.0012*(Th%lambdaobsframe(i)/973.0)**3.46 -0.00093*(Th%lambdaobsframe(i)/950.0)**3.46 !5n -> 1n
   end if
   
   if (912.0.le.(Th%lambdaobsframe(i)).and.((Th%lambdaobsframe(i)).le.912.0*y)) then
   x = Th%lambdaobsframe(i)/912.0
   negtaueff(i) = -0.0036*(Th%lambdaobsframe(i)/1216.0)**3.46 -0.0017*(Th%lambdaobsframe(i)/1026.0)**3.46 -0.0012*(Th%lambdaobsframe(i)/973.0)**3.46 -0.00093*(Th%lambdaobsframe(i)/950.0)**3.46 -0.25*x**3.*(y**0.46 - x**0.46) - 9.4*x**1.5*(y**0.18-x**0.18) &
   - 0.7*x**3.*(y**(-1.32) -x**(-1.32)) + 0.023*(y**1.68 - x**1.68)
   end if
   
   !We actually assume that no photons with lambda(rest frame) < 912 A can exist in galaxies (galaxies are their own Lyman limit)
   
   if (Th%lambdaobsframe(i).le.912.0*y) then
   negtaueff(i) = -1e32
   end if
   
   Th%thfluxes(i) = Th%thfluxes(i)*exp(negtaueff(i))
   
   if (writeout) then
   write(20,*), Th%lambda(i), Th%lambdaobsframe(i), Th%thfluxes(i), exp(negtaueff(i))
   end if
   
   end do
   
   if (writeout) then
   close(20)
   end if
   end subroutine Madau

'''