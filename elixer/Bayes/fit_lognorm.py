'''
   Last updated: 19 Sep 2015
   
   Andrew Leung
   Rutgers University
   
'''

from datapath import *
import oiigen as og
import laegen as lg
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import time



def lognorm_pdf(x,mu,sigma):
   return np.exp(-(np.log(x)-mu)**2/(2*sigma**2)) / (x*sigma*np.sqrt(2*np.pi))

def lognorm_pdf_nomu(x,sigma):
   return np.exp(-(np.log(x))**2/(2*sigma**2))

def lognorm_pdf_musq(x,mu,sigma):
   return np.exp(-(np.log(x)-0.5*mu**2)**2/(2*sigma**2))

def lognorm_pdf_musq_norm(x,mu,sigma):
   return np.exp(-(np.log(x)-0.5*mu**2)**2/(2*sigma**2)) / (x*sigma*np.sqrt(2*np.pi))

def lognorm_pdf_musq_fixsigma(x,mu):         ### see Blanton & Lin (2000), Table 1
   return np.exp(-(np.log(x)-0.5*mu**2)**2/(2*0.77**2))

def lognorm_pdf_mod1_BL00(x,W_0,sigma):      ### see Blanton & Lin (2000), Formula 5
   return np.exp(-(np.log(x/W_0)-0.5*sigma**2)**2/(2*sigma**2)) / (x*sigma*np.sqrt(2*np.pi))

def lognorm_pdf_mod2_BL00(x,mu,sigma):         ### see Blanton & Lin (2000), Formula 5
   return np.exp(-(np.log(x)-0.5*mu**2-0.5*sigma**2)**2/(2*sigma**2)) / (x*sigma*np.sqrt(2*np.pi))

def lognorm_pdf_BL00_fixmu(x,sigma):         ### see Blanton & Lin (2000), Table 1
   return np.exp(-(np.log(x/10.14)-0.5*sigma**2)**2/(2*sigma**2))

def decay_exp(x,w_0):
   return np.exp(-x/w_0)

def decay_exp_norm(x,w_0):
   return np.exp(-x/w_0) / w_0


def run(z_OII,case):
   w_0 = og.w_0Ext(z_OII,case)
   #logwmin, logwmax = 0.5*np.log10(w_0), np.log10(100)
   logwmin, logwmax = np.log10(5), np.log10(100)
   w_list = 10**(np.linspace(logwmin,logwmax,1e5))
   #w_list = np.append(w_list,np.linspace(25,1000,1e5))
   #w_list = np.linspace(5,1000,6000)
   
   exp_dist = []
   for w in range(len(w_list)):
      exp_dist.append( decay_exp_norm(w_list[w],w_0) )
      #exp_dist.append( decay_exp(w_list[w],w_0) )

   #fitParam, fitCovar = curve_fit(lognorm_pdf,w_list,exp_dist)
   #fitParam, fitCovar = curve_fit(lognorm_pdf_nomu,w_list,exp_dist)
   #fitParam, fitCovar = curve_fit(lognorm_pdf_musq,w_list,exp_dist)
   #fitParam, fitCovar = curve_fit(lognorm_pdf_musq_fixsigma,w_list,exp_dist)
   fitParam, fitCovar = curve_fit(lognorm_pdf_mod1_BL00,w_list,exp_dist)
   #fitParam, fitCovar = curve_fit(lognorm_pdf_mod2_BL00,w_list,exp_dist)
   #fitParam, fitCovar = curve_fit(lognorm_pdf_BL00_fixmu,w_list,exp_dist)
   #print(fitParams2)
   #print(fitCovariances2)
   return fitParam, fitCovar


def run_LAE(z_LAE,case):
   w_0 = lg.w_0Ext(z_LAE,case)
   logwmin, logwmax = np.log10(20), np.log10(1000)
   w_list = 10**(np.linspace(logwmin,logwmax,1e5))

   exp_dist = []
   for w in range(len(w_list)):
      exp_dist.append( decay_exp_norm(w_list[w],w_0) )

   fitParam, fitCovar = curve_fit(lognorm_pdf_mod1_BL00,w_list,exp_dist)
   return fitParam, fitCovar




def lognorm_params():
   t0 = time.time()
   redshifts = np.linspace(0.05,0.5,451)
   W_0, sigma = [],[]
   for i in range(len(redshifts)):
      a,b = run(redshifts[i],'base')
      W_0.append(a[0])
      sigma.append(a[1])
      if i%10 == 0: print('fit_lognorm.lognorm_params()', i, '%.0f seconds'%(time.time()-t0), time.strftime("%d %b %H:%M:%S", time.localtime()))
   
   return redshifts.tolist(), W_0, sigma         ### three lists


def plot(z_OII,case):
   fitParam, fitCovar = run(z_OII,case)
   print(fitParam)   #,fitCovar)
   fig, ax = plt.subplots()
   
   w_list = 10**(np.linspace(-3,2,1e5))
   
   plt.plot(w_list, decay_exp_norm(w_list,og.w_0Ext(z_OII,'base')), color='red',  marker='',ls='-', ms=1,lw=1.6,label='exponential ($z$ = '+str(z_OII)+')')
   plt.plot(w_list, lognorm_pdf_mod1_BL00(w_list,fitParam[0],fitParam[1]), color='blue',  marker='',ls='-', ms=1,lw=1.6,label='lognormal fit ($z$ = '+str(z_OII)+')')
   
   #plt.xscale('log')
   #plt.yscale('log')
   legend = plt.legend(loc='upper right',scatterpoints=100,shadow=False)
   frame = legend.get_frame()
   for label in legend.get_texts(): label.set_fontsize('large')
   for label in legend.get_lines(): label.set_linewidth(0.96)
   
   plt.xlabel('rest-frame EW (\AA)')
   plt.ylabel('probability density')
   
   plt.text(53*1.25, 0.06*(1-0.65*(z_OII-0.05)/0.45), r'$W_{_{0}}$ = '+str(round(fitParam[0],4)), fontsize='xx-large', fontweight='bold', color='blue')
   plt.text(53*1.25, 0.05*(1-0.65*(z_OII-0.05)/0.45), r'$\sigma_{_{W}}$ = '+str(round(fitParam[1],4)), fontsize='xx-large', fontweight='bold', color='blue')
   plt.text(54*1.25, 0.08*(1-0.65*(z_OII-0.05)/0.45), '$w_{_{0}}$ = '+str(round(og.w_0Ext(z_OII,'base'),2)), fontsize='xx-large', fontweight='bold', color='red')
   
   fig.tight_layout()
   #plt.savefig('normalized_lognormal_z='+str(z_OII)+'.pdf')
   #plt.savefig('test_lognormal_z='+str(z_OII)+'.pdf')
   plt.savefig(case+'_lognormal_z='+str(z_OII)+'.pdf')
   plt.close()


def plotLAE(z_LAE,refcase):
   
   if refcase == 'Gr15': case = 6
   elif refcase == 'Ci12': case = 1
   
   fitParam, fitCovar = run_LAE(z_LAE,case)
   print(fitParam)   #,fitCovar)
   fig, ax = plt.subplots()
   
   w_list = 10**(np.linspace(-3,3,1e5))
   
   plt.plot(w_list, decay_exp_norm(w_list,lg.w_0Ext(z_LAE,case)), color='red',  marker='',ls='-', ms=1,lw=1.6,label='exponential ($z$ = '+str(z_LAE)+')')
   plt.plot(w_list, lognorm_pdf_mod1_BL00(w_list,fitParam[0],fitParam[1]), color='blue',  marker='',ls='-', ms=1,lw=1.6,label='lognormal fit ($z$ = '+str(z_LAE)+')')
   
   plt.plot([20,20],[0,1],'k-',ls='dashed',lw=0.6)
   
   plt.xscale('log')
   plt.xlim(1e-1,1e3)
   #plt.yscale('log')
   plt.ylim(0,0.02)
   
   legend = plt.legend(loc='upper right',scatterpoints=100,shadow=False)
   frame = legend.get_frame()
   for label in legend.get_texts(): label.set_fontsize('large')
   for label in legend.get_lines(): label.set_linewidth(0.96)
   
   plt.xlabel('rest-frame EW (\AA)')
   plt.ylabel('probability density')
   
   plt.title('exponential:\,\{$w_{_{0}}$\,=\,'+str(round(lg.w_0Ext(z_LAE,case),2))+'\}, '+r'lognormal:\,\{$W_{_{0}}$\,=\,'+str(round(fitParam[0],4))+', '+r'$\sigma_{_{W}}$\,=\,'+str(round(fitParam[1],4))+'\}')
   
   fig.tight_layout()
   #plt.savefig('normalized_lognormal_z='+str(z_OII)+'.pdf')
   #plt.savefig('test_lognormal_z='+str(z_OII)+'.pdf')
   plt.savefig(refcase+'_lognormal_z='+str(z_LAE)+'.pdf')
   plt.close()


if __name__ == '__main__':
   
   z = [2.1,3.1,3.5]
   y = ['Gr15','Ci12']
   
   for redshift in z:
      for refcase in y:
         plotLAE(redshift,refcase)
   
   '''
   bin_ci13 = [0.1000,0.2625,0.3875,0.5050]
   z_list = np.linspace(0.15,0.55,9)
   z_list = np.append(z_list,bin_ci13)
   
   #z_list = np.linspace(0.05,0.50,46)
   for i in range(len(z_list)):
      plot(z_list[i],'base')
      plot(z_list[i],'low')
      plot(z_list[i],'high')
      
   '''

