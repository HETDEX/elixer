'''
   Last updated: 6 Oct 2015
   
   Andrew Leung
   Rutgers University
   
   Simulate splitting observing time to obtain color
   Bayesian method tuning buried in this file as optimize_bayesian()
   
'''

from datapath import *
import time
import numpy as np
import bayesian as b
import nb
import gc



def runid(runid_1,runid_2):
   global g_band, r_band, phot_EW

   g_band = [round(float(runid_1[13:]),1),  runid_1[7]+'$\'$ (5$\sigma$ = '+str(round(float(runid_1[13:]),1))+')']
   r_band = [round(float(runid_2[13:]),1),  runid_2[7]+'$\'$ (5$\sigma$ = '+str(round(float(runid_2[13:]),2))+')']

   phot_EW = [runid_1[7]+'\' ('+runid_1[13:]+'), '+runid_2[7]+'\' ('+runid_2[13:]+')']



def sim_color(runid_1,runid_2,fromScratch,scale):
   print('imaging.sim_color() began at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')
   t0 = time.time()
   
   reload(nb)
   nb.init(-1.65,1,1,1,-1.2,1,1,1)
   
   ## initial simulation run with g-r color obtained from subtracting r' mag from g' mag done with the same vector of random seed for measurement noise in each imaging survey
   
   global c00,c13,c05,c08,c09,c07,c15,c16,c17,c18,c19,c20,c10,c12,c14,c11,c01,c04,c03
   global true_ew_rest,true_ew_obs,true_ew_inf,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n,true_em_line_flux
   global true_g_minus_r, true_g_minus_i, true_g_minus_z, true_r_minus_i, true_r_minus_z, true_i_minus_z
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global f_nu_cont_slope
   global sigma_line_flux, sigma_f_nu, sigma_ew_obs, sigma_g_minus_r
   global b_prob_ratio  ## (AL 02-27-15)
   
   global g_band,r_band
   runid(runid_1,runid_2)
   
   runname = runid_1 + runid_2[6:]
   
   global g_sigma_f_nu,g_sigma_ew_obs,g_true_ew_obs,g_true_ew_inf,g_true_cont_flux_dens,g_true_s_to_n,g_true_AB_cont_mag,g_c07,g_c09,g_c11,g_c12,g_c14

   get_data(runid_1)
   g_sigma_f_nu          = np.copy(sigma_f_nu)
   g_sigma_ew_obs        = np.copy(sigma_ew_obs)
   g_true_ew_obs         = np.copy(true_ew_obs)
   g_true_ew_inf         = np.copy(true_ew_inf)
   g_true_cont_flux_dens = np.copy(true_cont_flux_dens)
   g_true_s_to_n         = np.copy(true_s_to_n)
   g_true_AB_cont_mag    = np.copy(true_AB_cont_mag)
   g_c07                 = np.copy(c07)
   g_c09                 = np.copy(c09)
   g_c11                 = np.copy(c11)
   g_c12                 = np.copy(c12)
   g_c13                 = np.copy(c13)
   g_c14                 = np.copy(c14)
   g_f_nu                = g_c09
   g_mag                 = g_c14
   g_index               = np.copy(c00).tolist()
   
   global g_prob_ratio  ## (AL 02-27-15)
   g_prob_ratio          = np.copy(b_prob_ratio)
   
   global r_sigma_f_nu,r_sigma_ew_obs,r_true_ew_obs,r_true_ew_inf,r_true_cont_flux_dens,r_true_s_to_n,r_true_AB_cont_mag,r_c07,r_c09,r_c11,r_c12,r_c14

   get_data(runid_2)
   r_sigma_f_nu          = np.copy(sigma_f_nu)
   r_sigma_ew_obs        = np.copy(sigma_ew_obs)
   r_true_ew_obs         = np.copy(true_ew_obs)
   r_true_ew_inf         = np.copy(true_ew_inf)
   r_true_cont_flux_dens = np.copy(true_cont_flux_dens)
   r_true_s_to_n         = np.copy(true_s_to_n)
   r_true_AB_cont_mag    = np.copy(true_AB_cont_mag)
   r_c07                 = np.copy(c07)
   r_c09                 = np.copy(c09)
   r_c11                 = np.copy(c11)
   r_c12                 = np.copy(c12)
   r_c13                 = np.copy(c13)
   r_c14                 = np.copy(c14)
   r_f_nu                = r_c09
   r_mag                 = r_c14
   r_index               = np.copy(c00).tolist()
   
   global r_prob_ratio  ## (AL 02-27-15)
   r_prob_ratio          = np.copy(b_prob_ratio)
   
   print('*** check list match ***')
   print(len(redshift) == len(c01))
   print(redshift[88] == c01[88])
   print(float(redshift[88]),float(c01[88]))
   print('*** check data match ***')
   print(len(cont_mag[2]) == len(c14))
   print(cont_mag[2] == c14)
   print(float(cont_mag[2][88]),float(r_c14[88]))
   
   
   global noiseless_g_minus_r, noisified_g_minus_r, calculated_dmag_g, calculated_dmag_r, calculated_dcolor
   noiseless_g_minus_r = g_true_AB_cont_mag - r_true_AB_cont_mag
   noisified_g_minus_r = g_mag - r_mag
   calculated_dmag_g = -0.5/np.log(10.0) * 10**(-0.4*(g_band[0] -g_true_AB_cont_mag))
   calculated_dmag_r = -0.5/np.log(10.0) * 10**(-0.4*(r_band[0] -r_true_AB_cont_mag))

   calculated_dcolor = np.sqrt(calculated_dmag_g**2 +calculated_dmag_r**2)
   
   
   '''
      interpolate or extrapolate to obtain f_nu at lambda_EL, which yields a photometric EW of the line computed at lambda_EL
      EG email (11-11-14)
      
   '''
   
   global f_nu_at_EL, phot_EW_obs_at_EL
   
   '''
      use appropriate variable names for color (soon)
      AL (02-27-15)
      
   '''
   
   if runid_1[7] == 'g' and runid_2[7] == 'r':
      lambda_eff_g = 4813.         #4686. #(g0115)      #4750. #(g1102,g0119)
      lambda_eff_r = 6287.         #6165. #(r0115)      #6220. #(r1102,r0119)
   
   elif runid_1[7] == 'g' and runid_2[7] == 'i':
      lambda_eff_g = 4813.
      lambda_eff_r = 7732.

   elif runid_1[7] == 'g' and runid_2[7] == 'z':
      lambda_eff_g = 4813.
      lambda_eff_r = 9400.
   
   elif runid_1[7] == 'r' and runid_2[7] == 'i':
      lambda_eff_g = 6287.
      lambda_eff_r = 7732.
   
   elif runid_1[7] == 'r' and runid_2[7] == 'z':
      lambda_eff_g = 6287.
      lambda_eff_r = 9400.
   
   #noisified_g_minus_r = noisified_g_minus_r.tolist()
   ### [noiseless, noisified]
   g_minus_r = [noiseless_g_minus_r, noisified_g_minus_r]
   f_nu_at_g = [g_true_cont_flux_dens, g_c09]
   f_nu_at_r = [r_true_cont_flux_dens, r_c09]
   lineFlux  = [true_em_line_flux, c08]
   lambda_EL = c05
   m                 = [[],[]]
   b                 = [[0]*len(g_minus_r[0]),[0]*len(g_minus_r[1])]
   f_nu_at_EL        = [[0]*len(g_minus_r[0]),[0]*len(g_minus_r[1])]
   phot_EW_obs_at_EL = [[0]*len(g_minus_r[0]),[0]*len(g_minus_r[1])]
   
   for i in range(len(g_minus_r)):
      ### (12-01-14) bug fix -> always use noiseless color to interpolate
      ### (12-17-14) wasn't a bug -> use appropriate color to interpolate
      m[i] = 0.4 * g_minus_r[i] / np.log10(lambda_eff_r/lambda_eff_g)
   
      for j in range(len(g_minus_r[i])):            ### (12-10-14 post-PSU telecon meeting)
         if (f_nu_at_g[i][j] < 0) or (f_nu_at_r[i][j] < 0):
            phot_EW_obs_at_EL[i][j] = 1e5      ## (04-24-15)1e4
            f_nu_at_EL[i][j] = 1e-4            ### just off the plot for EW_inf vs mag (33.9 AB)
            noisified_g_minus_r[j] = -999
         else:
            b1 = np.log10(f_nu_at_g[i][j]) - m[i][j] * np.log10(lambda_eff_g)
            b2 = np.log10(f_nu_at_r[i][j]) - m[i][j] * np.log10(lambda_eff_r)
            b[i][j] = (0.5 * (b1+b2))         ### estimated power-law slope, an average
            f_nu_at_EL[i][j] = 10** (m[i][j] * np.log10(lambda_EL[j]) + b[i][j])
            phot_EW_obs_at_EL[i][j] = (1e6)*(1e23)*(lineFlux[i][j])/(f_nu_at_EL[i][j])*((lambda_EL[j])**2)/(2.997925e18)

   global phot_sigma_f_nu, phot_sigma_ew_obs
   
   phot_sigma_f_nu   = f_nu_at_EL[0] * np.sqrt( (g_sigma_f_nu/g_true_cont_flux_dens)**2 +(r_sigma_f_nu/r_true_cont_flux_dens)**2 )
   phot_sigma_ew_obs = phot_EW_obs_at_EL[0] * np.sqrt( (sigma_line_flux/true_em_line_flux)**2 +(phot_sigma_f_nu/f_nu_at_EL[0])**2 )

   optreq0      = [-1,-1,-1]
   optreq1      = [-1,-1,-1]
   optsigmaDA0  = [-1,-1,-1]
   optsigmaDA1  = [-1,-1,-1]
   sigmaDA_ew20 = [-1,-1,-1]

   if fromScratch:
      bayesian_separation(runname,'g',g_band[0],'r',r_band[0],scale)
      write_results(runname,scale)

   global phot_EW
   reload(nb)

   band = 'power-law-interpolated to lambda_EL'
   run = runname+'_lambda_EL'

   if fromScratch:
      nb.get_data_from_imaging(band, g_band[0], r_band[0], scale)
      nb.write_sim_data(run, band, phot_EW[0])               ## write out data in format used by plots.py
      nb.write_prob_ratio(run)
   else:
      nb.get_sim_data(run, band, scale)
      nb.get_prob_ratio(run)

   optreq0[2],optsigmaDA0[2],sigmaDA_ew20[2] = optimize_bayesian(runname+'_lambda_EL', band, phot_EW[0], '1.9-2.5')
   temp = np.copy(sigmaDA_ew20[2])
   optreq1[2],optsigmaDA1[2],sigmaDA_ew20[2] = optimize_bayesian(runname+'_lambda_EL', band, phot_EW[0], '2.5-3.5')
   print('*** check match of sigma_ew20 lists for two Bayesian optimization bins ***')
   print('*** run name: '+run)
   print(temp == np.array(sigmaDA_ew20[2]))

   if not np.all(temp == np.array(sigmaDA_ew20[2])):
      print (temp)
      print (np.array(sigmaDA_ew20[2]))


   ''' (AL 02-27-15)
   band = 'g\''
   run = runname+'_g_f_nu'

   if fromScratch:
      nb.get_data_from_imaging(band, g_band[0], r_band[0], scale)
      nb.write_sim_data(run, band, g_band[0])               ## write out data in format used by plots.py
      nb.write_prob_ratio(run)
   else:
      nb.get_sim_data(run, band, scale)
      nb.get_prob_ratio(run)

   optreq0[0],optsigmaDA0[0],sigmaDA_ew20[0] = optimize_bayesian(runname+'_g_f_nu', band, g_band[0], '1.9-2.5')
   temp = np.copy(sigmaDA_ew20[0])
   optreq1[0],optsigmaDA1[0],sigmaDA_ew20[0] = optimize_bayesian(runname+'_g_f_nu', band, g_band[0], '2.5-3.5')
   print('*** check match of sigma_ew20 lists for two Bayesian optimization bins ***')
   print('*** run name: '+run)
   print(temp == np.array(sigmaDA_ew20[0]))


   band = 'r\''
   run = runname+'_r_f_nu'

   if fromScratch:
      nb.get_data_from_imaging(band, g_band[0], r_band[0], scale)
      nb.write_sim_data(run, band, r_band[0])               ## write out data in format used by plots.py
      nb.write_prob_ratio(run)
   else:
      nb.get_sim_data(run, band, scale)
      nb.get_prob_ratio(run)

   optreq0[1],optsigmaDA0[1],sigmaDA_ew20[1] = optimize_bayesian(runname+'_r_f_nu', band, r_band[0], '1.9-2.5')
   temp = np.copy(sigmaDA_ew20[1])
   optreq1[1],optsigmaDA1[1],sigmaDA_ew20[1] = optimize_bayesian(runname+'_r_f_nu', band, r_band[0], '2.5-3.5')
   print('*** check match of sigma_ew20 lists for two Bayesian optimization bins ***')
   print('*** run name: '+run)
   print(temp == np.array(sigmaDA_ew20[1]))
   
   '''


   print('imaging.sim_color() required %.1f s for '% (time.time()-t0))

   gc.collect()
   print('imaging.sim_color() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')

   return optreq0, optreq1, optsigmaDA0, optsigmaDA1, sigmaDA_ew20         ## [ optreq_g, optreq_r, optreq_phot ], each list member is length-2 for results in 1.9<z<2.5 (index-0) and 1.9<z<3.5 (index-1)



def optimize_bayesian(scale,run,band,depth,bin,ver_sigma_dA):
   print('imaging.optimize_bayesian() began at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('    run = '+str(run))
   print('   band = '+str(band))
   print('  depth = '+str(depth))
   print('  scale = '+str(scale))
   print('')
   
   t0 = time.time()
   if bin == '1.9-2.5': optbin = 0
   elif bin == '2.5-3.5': optbin = 1
   elif bin == '1.9-3.5': optbin = 2
   
   out_file = open(str(writepath)+str(run)+'_separation_results_opt'+str(optbin)+'_sda'+str(ver_sigma_dA)+'.dat','w')
   out_file.write(''+'\n')
   
   spec = ['ew20',1e5,2e5,3e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6]
   sigmaDA,contamFrac,classLAEcount,incompFrac,trueLAEcount,cntLAErecovered = [],[],[],[],[],[]
   
   reload(nb)
   
   sigmaDA_,contamFrac_,classLAEcount_,incompFrac_,trueLAEcount_,cntLAErecovered_ = nb.contam_and_incomp_ew20(run,ver_sigma_dA)
   sigmaDA.append(sigmaDA_)                  ### each output from nb.contam_and_incomp_ew20() is a length-2 list
   contamFrac.append(contamFrac_)
   classLAEcount.append(classLAEcount_)
   incompFrac.append(incompFrac_)
   trueLAEcount.append(trueLAEcount_)
   cntLAErecovered.append(cntLAErecovered_)
   
   sigmaDA_ew20 = sigmaDA[0]         ### first (and for now only) member of list sigmaDA is a length-2 list, in which optbin=0 is 1.9<z<2.5, optbin=1 is 1.9<z<3.5 -> sigmaDA_ew20 is a length-2 list (to be outputed)
   
   for i in range(1,len(spec)):
      sigmaDA_,contamFrac_,classLAEcount_,incompFrac_,trueLAEcount_,cntLAErecovered_ = nb.contam_and_incomp(run,0.5,float(spec[i]),10.0,False,ver_sigma_dA)
      sigmaDA.append(sigmaDA_)               ### again, each output from nb.contam_and_incomp() is a length-2 list
      contamFrac.append(contamFrac_)
      classLAEcount.append(classLAEcount_)
      incompFrac.append(incompFrac_)
      trueLAEcount.append(trueLAEcount_)
      cntLAErecovered.append(cntLAErecovered_)
   
      print('imaging.optimize_bayesian() %.2f s' % (time.time()-t0))
      print('****** run: '+run)
      print('****   optimized for: '+bin)
      print('****   specified requirement: '+str(spec[i]))
      print('****   sigma(d_A): '+str(sigmaDA[i][optbin]))
      print('****')

   sigmaDA_opt, optreq = [], 0.
   for i in range(len(spec)):
      if i == 0:
         j = i
         out_file.write('******************************************************************'+'\n')
         out_file.write('***        run name: '+str(run)+'\n')
         out_file.write('***    imaging band: '+str(band)+'\n')
         out_file.write('***   5 sigma depth: '+str(depth)+'\n')
         out_file.write('***           scale: '+str(scale)+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** specified separation requirement: '+str(spec[i])+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** ****** 1.9<z<2.5 ******'+'\n')
         out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][0]))+'\n')
         out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][0]))+'\n')
         out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][0]-cntLAErecovered[j][0]))+'\n')
         out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][0]/float(trueLAEcount[j][0]))+'\n')
         out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][0]))+'\n')
         out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][0]-cntLAErecovered[j][0]))+'\n')
         out_file.write('***               contamination: '+str(contamFrac[j][0])+'\n')
         out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][0])+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** ****** 2.5<z<3.5 ******'+'\n')
         out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][1]))+'\n')
         out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][1]))+'\n')
         out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][1]-cntLAErecovered[j][1]))+'\n')
         out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][1]/float(trueLAEcount[j][1]))+'\n')
         out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][1]))+'\n')
         out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][1]-cntLAErecovered[j][1]))+'\n')
         out_file.write('***               contamination: '+str(contamFrac[j][1])+'\n')
         out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][1])+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** ****** 1.9<z<3.5 ******'+'\n')
         out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][2]))+'\n')
         out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][2]))+'\n')
         out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][2]-cntLAErecovered[j][2]))+'\n')
         out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][2]/float(trueLAEcount[j][2]))+'\n')
         out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][2]))+'\n')
         out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][2]-cntLAErecovered[j][2]))+'\n')
         out_file.write('***               contamination: '+str(contamFrac[j][2])+'\n')
         out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][2])+'\n')
         out_file.write('***'+'\n')
         out_file.write('***'+'\n\n\n')

      
      if not i == 0:
         sigmaDA_opt.append(sigmaDA[i][optbin])

   optreq = float(spec[np.argmin(sigmaDA_opt)+1])

   
   '''
      ##   iterate to find optimal Bayesian requirement
      
   '''

   iterations = 12   #8      #16
   
   for i in range(iterations):
      spec,sigmaDA = [optreq-9*10**(-i+4),optreq-8*10**(-i+4),optreq-7*10**(-i+4),optreq-6*10**(-i+4),optreq-5*10**(-i+4),optreq-4*10**(-i+4),optreq-3*10**(-i+4),optreq-2*10**(-i+4),optreq-1*10**(-i+4),optreq,optreq+1*10**(-i+4),optreq+2*10**(-i+4),optreq+3*10**(-i+4),optreq+4*10**(-i+4),optreq+5*10**(-i+4),optreq+6*10**(-i+4),optreq+7*10**(-i+4),optreq+8*10**(-i+4),optreq+9*10**(-i+4)],[]
      
      sigmaDA,contamFrac,classLAEcount,incompFrac,trueLAEcount,cntLAErecovered = [],[],[],[],[],[]
      sigmaDA_opt = []
      
      for j in range(len(spec)):
         sigmaDA_,contamFrac_,classLAEcount_,incompFrac_,trueLAEcount_,cntLAErecovered_ = nb.contam_and_incomp(run,0.5,float(spec[j]),10.0,False,ver_sigma_dA)
         sigmaDA.append(sigmaDA_)
         contamFrac.append(contamFrac_)
         classLAEcount.append(classLAEcount_)
         incompFrac.append(incompFrac_)
         trueLAEcount.append(trueLAEcount_)
         cntLAErecovered.append(cntLAErecovered_)
         
         sigmaDA_opt.append(sigmaDA[j][optbin])
         print('imaging.optimize_bayesian() %.2f s' % (time.time()-t0))
         print('****** run: '+run)
         print('****   optimizing for: '+bin)
         print('****   specified requirement: '+str(spec[j]))
         print('****   sigma(d_A): '+str(sigmaDA[j][optbin]))
         print('****')

      optreq = float(spec[np.argmin(sigmaDA_opt)])
      if i+1 == iterations: optindex = np.argmin(sigmaDA_opt)

   j = optindex
   optsigmaDA = sigmaDA[optindex]               ## optsigmaDA is a length-2 list: i=0 is 1.9<z<2.5, i=1 is 1.9<z<3.5, in which optbin=0 is 1.9<z<2.5, optbin=1 is 1.9<z<3.5 -> optsigmaDA is a length-2 list (to be outputed)
   
   out_file.write('******************************************************************'+'\n')
   out_file.write('***        run name: '+str(run)+'\n')
   out_file.write('***    imaging band: '+str(band)+'\n')
   out_file.write('***   5 sigma depth: '+str(depth)+'\n')
   out_file.write('***           scale: '+str(scale)+'\n')
   out_file.write('***'+'\n')
   out_file.write('***       optimized for redshift bin: '+str(bin)+'\n')
   out_file.write('*** specified separation requirement: %.5f' % (spec[j])+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** ****** 1.9<z<2.5 ******'+'\n')
   out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][0]))+'\n')
   out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][0]))+'\n')
   out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][0]-cntLAErecovered[j][0]))+'\n')
   out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][0]/float(trueLAEcount[j][0]))+'\n')
   out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][0]))+'\n')
   out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][0]-cntLAErecovered[j][0]))+'\n')
   out_file.write('***               contamination: '+str(contamFrac[j][0])+'\n')
   out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][0])+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** ****** 2.5<z<3.5 ******'+'\n')
   out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][1]))+'\n')
   out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][1]))+'\n')
   out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][1]-cntLAErecovered[j][1]))+'\n')
   out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][1]/float(trueLAEcount[j][1]))+'\n')
   out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][1]))+'\n')
   out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][1]-cntLAErecovered[j][1]))+'\n')
   out_file.write('***               contamination: '+str(contamFrac[j][1])+'\n')
   out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][1])+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** ****** 1.9<z<3.5 ******'+'\n')
   out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][2]))+'\n')
   out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][2]))+'\n')
   out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][2]-cntLAErecovered[j][2]))+'\n')
   out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][2]/float(trueLAEcount[j][2]))+'\n')
   out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][2]))+'\n')
   out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][2]-cntLAErecovered[j][2]))+'\n')
   out_file.write('***               contamination: '+str(contamFrac[j][2])+'\n')
   out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][2])+'\n')
   out_file.write('***'+'\n')
   out_file.write('***'+'\n')

   
   out_file.write(''+'\n')
   out_file.write('***'+'\n')
   out_file.write('******************************************************************'+'\n')
   out_file.write('***        run name: '+str(run)+'\n')
   out_file.write('***    imaging band: '+str(band)+'\n')
   out_file.write('***   5 sigma depth: '+str(depth)+'\n')
   out_file.write('***           scale: '+str(scale)+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** EW>20A limit for classification as LAE'+'\n')
   out_file.write('***   1.9<z<2.5: sigma(D_A) = %.5f' % float(sigmaDA_ew20[0])+'\n')
   out_file.write('***   2.5<z<3.5: sigma(D_A) = %.5f' % float(sigmaDA_ew20[1])+'\n')
   out_file.write('***   1.9<z<3.5: sigma(D_A) = %.5f' % float(sigmaDA_ew20[2])+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** optimal Bayesian ratio requirement is %.5f' % float(optreq)+'\n')
   out_file.write('*** optimal Bayesian probability requirement is %.5f' % float(optreq/(1+optreq))+'\n')
   out_file.write('*** optimized for redshift bin: '+str(bin)+'\n')
   out_file.write('***'+'\n')
   out_file.write('***   1.9<z<2.5: sigma(D_A) = %.5f' % float(optsigmaDA[0])+'\n')
   out_file.write('***   1.9<z<2.5: increase in effective data = %.3f percent' % float(100*((sigmaDA_ew20[0]/optsigmaDA[0])**2-1))+'\n')
   out_file.write('***'+'\n')
   out_file.write('***   2.5<z<3.5: sigma(D_A) = %.5f' % float(optsigmaDA[1])+'\n')
   out_file.write('***   2.5<z<3.5: increase in effective data = %.3f percent' % float(100*((sigmaDA_ew20[1]/optsigmaDA[1])**2-1))+'\n')
   out_file.write('***'+'\n')
   out_file.write('***   1.9<z<3.5: sigma(D_A) = %.5f' % float(optsigmaDA[2])+'\n')
   out_file.write('***   1.9<z<3.5: increase in effective data = %.3f percent' % float(100*((sigmaDA_ew20[2]/optsigmaDA[2])**2-1))+'\n')
   out_file.write('***'+'\n')
   out_file.write('***'+'\n')
   out_file.write('\n'+'\n'+'\n')
   
   ### append results for probability ratio = 1
   sigmaDA,contamFrac,classLAEcount,incompFrac,trueLAEcount,cntLAErecovered = [],[],[],[],[],[]
   sigmaDA_opt = []
   spec = [1.]   #,1.45779,3.76608,3.03136]

   if run[:7] == '150817t':
      spec = [1.,1.45779,3.76608,3.03136]

   elif run[:7] == '150902_' or run[:7] == '150903_' or run[:7] == '150903p':
      spec = [1.,0.88804,5.50589,3.49331]

   else:
      ### define case for Bayesian method optimization
      optcase = '151006'      
      #optcase = '150909e'
      #optcase = '150903'
      #optcase = '150902'
      #optcase = '150817t'      ### Ci12, Ci13 simulation and Bayesian method
      
      for i in range(3):
         data = open(writepath+optcase+'_g0522_25.10_separation_results_opt'+str(i)+'_sda4.dat','r')
         ln = 0
         for line in data.readlines():
            ln += 1
            thisline = line.split()
            if ln == 50: spec.append(float(thisline[4]))
         data.close()

   for j in range(len(spec)):
      sigmaDA_,contamFrac_,classLAEcount_,incompFrac_,trueLAEcount_,cntLAErecovered_ = nb.contam_and_incomp(run,0.5,float(spec[j]),10.0,False,ver_sigma_dA)
      sigmaDA.append(sigmaDA_)
      contamFrac.append(contamFrac_)
      classLAEcount.append(classLAEcount_)
      incompFrac.append(incompFrac_)
      trueLAEcount.append(trueLAEcount_)
      cntLAErecovered.append(cntLAErecovered_)
      sigmaDA_opt.append(sigmaDA[j][optbin])
   
      if j == 0 or optbin+1 == j:

         out_file.write('******************************************************************'+'\n')
         out_file.write('***        run name: '+str(run)+'\n')
         out_file.write('***    imaging band: '+str(band)+'\n')
         out_file.write('***   5 sigma depth: '+str(depth)+'\n')
         out_file.write('***           scale: '+str(scale)+'\n')
         out_file.write('***'+'\n')
         out_file.write('***       optimized for redshift bin: '+str(bin)+'\n')
         out_file.write('*** specified separation requirement: %.5f' % (spec[j])+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** ****** 1.9<z<2.5 ******'+'\n')
         out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][0]))+'\n')
         out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][0]))+'\n')
         out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][0]-cntLAErecovered[j][0]))+'\n')
         out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][0]/float(trueLAEcount[j][0]))+'\n')
         out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][0]))+'\n')
         out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][0]-cntLAErecovered[j][0]))+'\n')
         out_file.write('***               contamination: '+str(contamFrac[j][0])+'\n')
         out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][0])+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** ****** 2.5<z<3.5 ******'+'\n')
         out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][1]))+'\n')
         out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][1]))+'\n')
         out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][1]-cntLAErecovered[j][1]))+'\n')
         out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][1]/float(trueLAEcount[j][1]))+'\n')
         out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][1]))+'\n')
         out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][1]-cntLAErecovered[j][1]))+'\n')
         out_file.write('***               contamination: '+str(contamFrac[j][1])+'\n')
         out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][1])+'\n')
         out_file.write('***'+'\n')
         out_file.write('*** ****** 1.9<z<3.5 ******'+'\n')
         out_file.write('***     \'true\' observable LAEs: '+'{:,}'.format(int(trueLAEcount[j][2]))+'\n')
         out_file.write('***         classified as LAEs : '+'{:,}'.format(int(classLAEcount[j][2]))+'\n')
         out_file.write('*** missed observable as LAEs : '+'{:,}'.format(int(trueLAEcount[j][2]-cntLAErecovered[j][2]))+'\n')
         out_file.write('***       sample incompleteness: '+str(1-cntLAErecovered[j][2]/float(trueLAEcount[j][2]))+'\n')
         out_file.write('***         true LAEs recovered: '+'{:,}'.format(int(cntLAErecovered[j][2]))+'\n')
         out_file.write('***      misidentified [O II]\'s: '+'{:,}'.format(int(classLAEcount[j][2]-cntLAErecovered[j][2]))+'\n')
         out_file.write('***               contamination: '+str(contamFrac[j][2])+'\n')
         out_file.write('***                  sigma(D_A): '+str(sigmaDA[j][2])+'\n')
         out_file.write('***'+'\n')
         out_file.write('***'+'\n')


   optsigmaDA = sigmaDA[optbin+1]
   optreq = spec[optbin+1]

   out_file.write(''+'\n')
   out_file.write('***'+'\n')
   out_file.write('******************************************************************'+'\n')
   out_file.write('***        run name: '+str(run)+'\n')
   out_file.write('***    imaging band: '+str(band)+'\n')
   out_file.write('***   5 sigma depth: '+str(depth)+'\n')
   out_file.write('***           scale: '+str(scale)+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** EW>20A limit for classification as LAE'+'\n')
   out_file.write('***   1.9<z<2.5: sigma(D_A) = %.5f' % float(sigmaDA_ew20[0])+'\n')
   out_file.write('***   2.5<z<3.5: sigma(D_A) = %.5f' % float(sigmaDA_ew20[1])+'\n')
   out_file.write('***   1.9<z<3.5: sigma(D_A) = %.5f' % float(sigmaDA_ew20[2])+'\n')
   out_file.write('***'+'\n')
   out_file.write('*** optimal Bayesian ratio requirement is %.5f' % float(optreq)+'\n')
   out_file.write('*** optimal Bayesian probability requirement is %.5f' % float(optreq/(1+optreq))+'\n')
   out_file.write('*** optimized (Ci12) for redshift bin: '+str(bin)+'\n')
   out_file.write('***'+'\n')
   out_file.write('***   1.9<z<2.5: sigma(D_A) = %.5f' % float(optsigmaDA[0])+'\n')
   out_file.write('***   1.9<z<2.5: increase in effective data = %.3f percent' % float(100*((sigmaDA_ew20[0]/optsigmaDA[0])**2-1))+'\n')
   out_file.write('***'+'\n')
   out_file.write('***   2.5<z<3.5: sigma(D_A) = %.5f' % float(optsigmaDA[1])+'\n')
   out_file.write('***   2.5<z<3.5: increase in effective data = %.3f percent' % float(100*((sigmaDA_ew20[1]/optsigmaDA[1])**2-1))+'\n')
   out_file.write('***'+'\n')
   out_file.write('***   1.9<z<3.5: sigma(D_A) = %.5f' % float(optsigmaDA[2])+'\n')
   out_file.write('***   1.9<z<3.5: increase in effective data = %.3f percent' % float(100*((sigmaDA_ew20[2]/optsigmaDA[2])**2-1))+'\n')
   out_file.write('***'+'\n')
   out_file.write('***'+'\n')

   out_file.close()



   print('imaging.optimize_bayesian() required %.1f seconds' % (time.time()-t0))
   print('imaging.optimize_bayesian() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('    run = '+str(run))
   print('   band = '+str(band))
   print('  depth = '+str(depth))
   print('')

   gc.collect()
   print('imaging.optimize_bayesian() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')
   
   return optreq, optsigmaDA, sigmaDA_ew20

   '''
      ### debug
      ### if bin == '1.9-2.5': optbin = 0
      ### elif bin == '1.9-3.5' or bin == 'all': optbin = 1
      
      ### optreq (float)
      ### optsigmaDA (list of length 2)
      ### sigmaDA_ew20 (list of length 2)

   '''


def bayesian_separation(run, isb1,isd1, isb2,isd2, scale):
   print('imaging.bayesian_separation() began at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')
   t0 = time.time()
   
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global true_ew_inf,true_ew_obs,true_em_line_flux,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n
   global true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global f_nu_cont_slope
   global sigma_line_flux,sigma_f_nu,sigma_ew_obs,sigma_g_minus_r
   
   global g_sigma_f_nu,g_sigma_ew_obs,g_true_ew_obs,g_true_ew_inf,g_true_cont_flux_dens,g_true_s_to_n,g_true_AB_cont_mag,g_c07,g_c09,g_c11,g_c12,g_c14
   global r_sigma_f_nu,r_sigma_ew_obs,r_true_ew_obs,r_true_ew_inf,r_true_cont_flux_dens,r_true_s_to_n,r_true_AB_cont_mag,r_c07,r_c09,r_c11,r_c12,r_c14
   global noiseless_g_minus_r, noisified_g_minus_r, calculated_dmag_g, calculated_dmag_r, calculated_dcolor
   global f_nu_at_EL, phot_EW_obs_at_EL
   global phot_sigma_f_nu, phot_sigma_ew_obs

   objtype = c13
   wl_obs = c05
   zinf = c10
   lineFlux = c08
   g_minus_r = noisified_g_minus_r
   g_band_ew_obs = g_c07
   g_band_ew_inf = g_c11
   r_band_ew_obs = r_c07
   r_band_ew_inf = r_c11
   phot_EW_obs = phot_EW_obs_at_EL[1]
   phot_EW_inf = phot_EW_obs_at_EL[1]/(1+c10)
   
   nb.init(-1.65,1,1,1,-1.2,1,1,1)
   b.init(-1.65,1,1,1,-1.2,1,1,1)
   which_color = isb1+'-'+isb2
   
   out_file = open(str(run)+'_select_objects.dat','w')
   
   global prob_LAE_over_prob_OII, prob_LAE_given_data, prob_OII_given_data
   prob_LAE_over_prob_OII, prob_LAE_given_data, prob_OII_given_data = [[],[],[]], [[],[],[]], [[],[],[]]
   
   global g_prob_ratio, r_prob_ratio
   
   for i in range(len(wl_obs)):
      
      #if which_color == 'g-r':
      if not g_minus_r[i] == -999:
         p_ratLAE_, p_plgd_, p_pogd_  = b.prob_ratio(wl_obs[i], lineFlux[i], phot_EW_obs[i], g_minus_r[i], which_color)
      elif g_minus_r[i] == -999:
         p_ratLAE_, p_plgd_, p_pogd_  = b.prob_ratio(wl_obs[i], lineFlux[i], phot_EW_obs[i], '', '')
      
      g_ratLAE_ = g_prob_ratio[i]   ## (AL 02-27-15)
      prob_LAE_over_prob_OII[0].append(g_ratLAE_)
      
      r_ratLAE_ = r_prob_ratio[i]   ## (AL 02-27-15)
      prob_LAE_over_prob_OII[1].append(r_ratLAE_)
      
      prob_LAE_over_prob_OII[2].append(p_ratLAE_)
      prob_LAE_given_data[2].append(p_plgd_)
      prob_OII_given_data[2].append(p_pogd_)
      
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
         (((g_band_ew_inf[i] > 20) or (r_band_ew_inf[i] > 20)) and objtype[i] == 'OII'):
         
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
         out_file.write('   HETDEX line flux sensitivity limit:     '+'%.5e'%(nb.lineSens(c05[i]))+' erg/s/cm^2\n')
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
         out_file.write('   '+isb1+'-'+isb2+' color (simulated):        '+'%.4f'%(true_g_minus_r[i])+'\n')
         out_file.write('   '+isb1+'-'+isb2+' color (noisified):        '+'%.4f'%(c15[i])+'\n')
         out_file.write('   sigma uncertainty, g-r color: '+'%.4f'%(sigma_g_minus_r[i])+'\n')
         out_file.write('\n')
         out_file.write('   power law slope in f_nu:      '+'%.4f'%(f_nu_cont_slope[i])+'\n')
         out_file.write('   power law slope in f_lambda:  '+'%.4f'%(f_nu_cont_slope[i]-2)+'\n')
         out_file.write('\n')
         out_file.write('\n')
         out_file.write('   '+isb1+'-'+isb2+' color (calculated, noiseless):  '+'%.4f'%(noiseless_g_minus_r[i])+'\n')
         out_file.write('   '+isb1+'-'+isb2+' color (calculated, noisified):  '+'%.4f'%(noisified_g_minus_r[i])+'\n')
         out_file.write('   dcolor:                             '+'%.4f'%(calculated_dcolor[i])+'\n')
         out_file.write('\n')
         out_file.write('   power-law-interpolated (noiseless): \n')
         out_file.write('     interpolated f_nu, at lambda_EL (noiseless): '+'%.4f'%(f_nu_at_EL[0][i])+' uJy\n')
         out_file.write('       sigma uncertainty:                         '+'%.4f'%(phot_sigma_f_nu[i])+' uJy\n')
         out_file.write('       signal-to-noise:                           '+'%.5f'%(f_nu_at_EL[0][i]/phot_sigma_f_nu[i])+'\n')
         out_file.write('     photometric EW at lambda_EL (noiseless): '+'%.4f'%(phot_EW_obs_at_EL[0][i])+' angstroms\n')
         out_file.write('       sigma uncertainty:                         '+'%.4f'%(phot_sigma_ew_obs[i])+' angstroms\n')
         out_file.write('       signal-to-noise:                           '+'%.5f'%(phot_EW_obs_at_EL[0][i]/phot_sigma_ew_obs[i])+'\n')
         out_file.write('\n')
         out_file.write('   power-law-interpolated (noisified): \n')
         out_file.write('     interpolated f_nu, at lambda_EL (noisified): '+'%.4f'%(f_nu_at_EL[1][i])+' uJy\n')
         out_file.write('       sigma uncertainty:                         '+'%.4f'%(phot_sigma_f_nu[i])+' uJy\n')
         out_file.write('    -> signal-to-noise:                           '+'%.5f'%(f_nu_at_EL[0][1]/phot_sigma_f_nu[i])+'\n')
         out_file.write('     photometric EW at lambda_EL (noisified): '+'%.4f'%(phot_EW_obs_at_EL[1][i])+' angstroms\n')
         out_file.write('       sigma uncertainty:                         '+'%.4f'%(phot_sigma_ew_obs[i])+' angstroms\n')
         out_file.write('    -> signal-to-noise:                           '+'%.5f'%(phot_EW_obs_at_EL[1][i]/phot_sigma_ew_obs[i])+'\n')
         out_file.write('\n')
         out_file.write('       probability ratio: '+'%.5e'%(float(p_ratLAE_))+'\n')
         out_file.write('       p(LAE):            '+'%.7f'%(float(p_ratLAE_)/(1+p_ratLAE_))+'\n')
         out_file.write('\n')
         out_file.write('\n')
         
         out_file.write('   imaging survey band:                                 '+isb1+'\'\n')
         out_file.write('     photometric EW, observed frame (simulated):       '+'%.3f'%(g_true_ew_obs[i])+' angstroms\n')
         out_file.write('     photometric EW, observed frame (noisified):       '+'%.3f'%(g_c07[i])+' angstroms\n')
         out_file.write('     sigma uncertainty, photometric EW:                '+'%.3f'%(g_sigma_ew_obs[i])+' angstroms\n')
         out_file.write('     fractional error, photometric EW:                 '+'%.5f'%(g_sigma_ew_obs[i]/g_true_ew_obs[i])+'\n')
         out_file.write('     signal-to-noise, photometric EW (simulated):      '+'%.5f'%(g_true_ew_obs[i]/g_sigma_ew_obs[i])+'\n')
         out_file.write('  -> signal-to-noise, photometric EW (noisified):      '+'%.5f'%(g_c07[i]/g_sigma_ew_obs[i])+'\n')
         out_file.write('     photometric EW, inferred Lyman-alpha (simulated): '+'%.3f'%(g_true_ew_inf[i])+' angstroms\n')
         out_file.write('     photometric EW, inferred Lyman-alpha (noisified): '+'%.3f'%(g_c11[i])+' angstroms\n')
         out_file.write('\n')
         out_file.write('   imaging survey band:                         '+isb1+'\'\n')
         out_file.write('     continuum flux density, f_nu (simulated): '+'%.3f'%(g_true_cont_flux_dens[i])+' uJy\n')
         out_file.write('     continuum flux density, f_nu (noisified): '+'%.3f'%(g_c09[i])+' uJy\n')
         out_file.write('     sigma uncertainty, f_nu:                  '+'%.5f'%(g_sigma_f_nu[i])+' uJy\n')
         out_file.write('     fractional error, f_nu:                   '+'%.5f'%(g_sigma_f_nu[i]/g_true_cont_flux_dens[i])+'\n')
         out_file.write('     signal-to-noise, f_nu (simulated):        '+str(g_true_s_to_n[i])+'\n')
         out_file.write('  -> signal-to-noise, f_nu (noisified):        '+str(g_c12[i])+'\n')
         out_file.write('     AB continuum magnitude (simulated):       '+'%.3f'%(g_true_AB_cont_mag[i])+'\n')
         out_file.write('     AB continuum magnitude (noisified):       '+'%.3f'%(g_c14[i])+'\n')
         out_file.write('\n')
         out_file.write('   imaging survey band: '+str('g')+'\'\n')
         out_file.write('     depth:             '+str(isd1)+'\'\n')
         out_file.write('     probability ratio: '+'%.5e'%(float(g_ratLAE_))+'\n')
         out_file.write('     p(LAE):            '+'%.7f'%(float(g_ratLAE_)/(1+g_ratLAE_))+'\n')
         out_file.write('\n')
         out_file.write('\n')
         out_file.write('   imaging survey band:                                 '+isb2+'\'\n')
         out_file.write('     photometric EW, observed frame (simulated):       '+'%.3f'%(r_true_ew_obs[i])+' angstroms\n')
         out_file.write('     photometric EW, observed frame (noisified):       '+'%.3f'%(r_c07[i])+' angstroms\n')
         out_file.write('     sigma uncertainty, photometric EW:                '+'%.3f'%(r_sigma_ew_obs[i])+' angstroms\n')
         out_file.write('     fractional error, photometric EW:                 '+'%.5f'%(r_sigma_ew_obs[i]/r_true_ew_obs[i])+'\n')
         out_file.write('     signal-to-noise, photometric EW (simulated):      '+'%.5f'%(r_true_ew_obs[i]/r_sigma_ew_obs[i])+'\n')
         out_file.write('  -> signal-to-noise, photometric EW (noisified):      '+'%.5f'%(r_c07[i]/r_sigma_ew_obs[i])+'\n')
         out_file.write('     photometric EW, inferred Lyman-alpha (simulated): '+'%.3f'%(r_true_ew_inf[i])+' angstroms\n')
         out_file.write('     photometric EW, inferred Lyman-alpha (noisified): '+'%.3f'%(r_c11[i])+' angstroms\n')
         out_file.write('\n')
         out_file.write('   imaging survey band:                         '+isb2+'\'\n')
         out_file.write('     continuum flux density, f_nu (simulated): '+'%.3f'%(r_true_cont_flux_dens[i])+' uJy\n')
         out_file.write('     continuum flux density, f_nu (noisified): '+'%.3f'%(r_c09[i])+' uJy\n')
         out_file.write('     sigma uncertainty, f_nu:                  '+'%.5f'%(r_sigma_f_nu[i])+' uJy\n')
         out_file.write('     fractional error, f_nu:                   '+'%.5f'%(r_sigma_f_nu[i]/r_true_cont_flux_dens[i])+'\n')
         out_file.write('     signal-to-noise, f_nu (simulated):        '+str(r_true_s_to_n[i])+'\n')
         out_file.write('  -> signal-to-noise, f_nu (noisified):        '+str(r_c12[i])+'\n')
         out_file.write('     AB continuum magnitude (simulated):       '+'%.3f'%(r_true_AB_cont_mag[i])+'\n')
         out_file.write('     AB continuum magnitude (noisified):       '+'%.3f'%(r_c14[i])+'\n')
         out_file.write('\n')
         out_file.write('   imaging survey band: '+str('r')+'\'\n')
         out_file.write('     depth:             '+str(isd2)+'\'\n')
         out_file.write('     probability ratio: '+'%.5e'%(float(r_ratLAE_))+'\n')
         out_file.write('     p(LAE):            '+'%.7f'%(float(r_ratLAE_)/(1+r_ratLAE_))+'\n')
         out_file.write('\n')
         out_file.write('\n')

      if i%(1e4)==0: print('imaging.bayesian_separation(), '+str(i+1)+'st loop: %.2f seconds' % (time.time()-t0))

   out_file.close()

   print('imaging.bayesian_separation() required %.1f seconds' % (time.time()-t0))

   print('imaging.bayesian_separation() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')



def write_results(run,scale):
   print('imaging.write_results() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')
   t0 = time.time()
   
   global c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global true_ew_inf,true_ew_obs,true_em_line_flux,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n
   global true_g_minus_r,true_g_minus_i,true_g_minus_z,true_r_minus_i,true_r_minus_z,true_i_minus_z
   global f_nu_cont_slope
   global sigma_line_flux,sigma_f_nu,sigma_ew_obs,sigma_g_minus_r
   global g_sigma_f_nu,g_sigma_ew_obs,g_true_ew_obs,g_true_ew_obs,g_true_cont_flux_dens,g_true_s_to_n,g_true_AB_cont_mag,g_c07,g_c09,g_c11,g_c12,g_c14
   global r_sigma_f_nu,r_sigma_ew_obs,r_true_ew_obs,r_true_ew_obs,r_true_cont_flux_dens,r_true_s_to_n,r_true_AB_cont_mag,r_c07,r_c09,r_c11,r_c12,r_c14
   global noiseless_g_minus_r, noisified_g_minus_r, calculated_dmag_g, calculated_dmag_r, calculated_dcolor
   global prob_LAE_over_prob_OII
   global f_nu_at_EL, phot_EW_obs_at_EL
   global phot_sigma_f_nu, phot_sigma_ew_obs
   

   objtype = c13.tolist()
   cLAE = objtype.count('LAE')
   cOII = objtype.count('OII')

   noise = ['noiseless','noisified']
   objlabel = ['LAE','OII']
   band1 = run[7]
   band2 = run[18]

   for k in range(len(noise)):
      for j in range(len(objlabel)):
         out_file = open(str(path)+str(objlabel[j])+'_simulation_'+str(noise[k])+'_'+str(run)+'.dat','w')
         out_file.write('# \n')
         if objlabel[j] == 'LAE':      out_file.write('# Simulated HETDEX catalog of Lyman-alpha emitters \n')
         elif objlabel[j] == 'OII':    out_file.write('# Simulated HETDEX catalog of [O II] emitters \n')
         if scale == 0.1:              out_file.write('#   * one-tenth scale simulation \n')
         elif scale == 0.25:           out_file.write('#   * one-quarter scale simulation \n')
         elif scale == 1.:             out_file.write('#   * full scale simulation \n')
         if noise[k] == 'noisified':   out_file.write('#   * Monte Carlo simulated quantities with Gaussian noise added \n')
         elif noise[k] == 'noiseless': out_file.write('#   * Monte Carlo simulated quantities prior to addition of noise \n')
         out_file.write('#   *    imaging survey bands: '+band1+'\', '+band2+'\' \n')
         g_5sigma_half = 23.9-2.5*np.log10(np.sqrt(2)*10**(-0.4*(g_band[0]-23.9)))
         r_5sigma_half = 23.9-2.5*np.log10(np.sqrt(2)*10**(-0.4*(r_band[0]-23.9)))
         out_file.write('#   *   5 sigma survey depths: %.2f mag, %.2f mag \n' %(g_5sigma_half,r_5sigma_half) )
         out_file.write('# \n')
         out_file.write('# Column 1:  object type \n')
         out_file.write('# Column 2:  wavelength of emission line      (angstroms) \n')
         out_file.write('# Column 3:  emission line flux               (erg/cm^2/s) \n')
         out_file.write('# Column 4:  continuum flux density       [lambda_EL] (uJy) \n')
         out_file.write('# Column 5:  photometric EW observed      [lambda_EL] (angstroms) \n')
         out_file.write('# Column 6:  continuum flux density       ['+band1+'\'] (uJy) \n')
         out_file.write('# Column 7:  equivalent width observed    ['+band1+'\'] (angstroms) \n')
         out_file.write('# Column 8:  continuum flux density       ['+band2+'\'] (uJy) \n')
         out_file.write('# Column 9:  equivalent width observed    ['+band2+'\'] (angstroms) \n')
         out_file.write('# Column 10: '+band1+'\'-'+band2+'\' color index  (simulated from HPS observed distributions)\n')
         out_file.write('# Column 11: '+band1+'\'-'+band2+'\' color index  (calculated from simulated imaging surveys) \n')
         out_file.write('# Column 12: signal-to-noise, emission line flux               (erg/cm^2/s) \n')
         out_file.write('# Column 13: signal-to-noise, continuum flux density     [lambda_EL] (uJy) \n')
         out_file.write('# Column 14: signal-to-noise, equivalent width observed  [lambda_EL] (angstroms) \n')
         out_file.write('# Column 15: signal-to-noise, continuum flux density     ['+band1+'\'] (uJy) \n')
         out_file.write('# Column 16: signal-to-noise, equivalent width observed  ['+band1+'\'] (angstroms) \n')
         out_file.write('# Column 17: signal-to-noise, continuum flux density     [r\'] (uJy) \n')
         out_file.write('# Column 18: signal-to-noise, equivalent width observed  [r\'] (angstroms) \n')
         out_file.write('# Column 19: sigma uncertainty, emission line flux               (erg/cm^2/s) \n')
         out_file.write('# Column 20: sigma uncertainty, continuum flux density     [lambda_EL] (uJy) \n')
         out_file.write('# Column 21: sigma uncertainty, equivalent width observed  [lambda_EL] (angstroms) \n')
         out_file.write('# Column 22: sigma uncertainty, continuum flux density     ['+band1+'\'] (uJy) \n')
         out_file.write('# Column 23: sigma uncertainty, equivalent width observed  ['+band1+'\'] (angstroms) \n')
         out_file.write('# Column 24: sigma uncertainty, continuum flux density     ['+band2+'\'] (uJy) \n')
         out_file.write('# Column 25: sigma uncertainty, equivalent width observed  ['+band2+'\'] (angstroms) \n')
         out_file.write('# Column 26: sigma uncertainty, '+band1+'\'-'+band2+'\' color index  (uncertainty in HPS observed distributions) \n')
         out_file.write('# Column 27: d('+band1+'\'-'+band2+'\' color index)                  (uncertainty propagated from simulated imaging surveys) \n')
         if noise[k] == 'noisified':
            out_file.write('# Column 28: P(LAE)  [lambda_EL] photometric EW + '+band1+'\'-'+band2+'\' color \n')
            out_file.write('# Column 29: P(LAE)  ['+band1+'\'] equvalent width \n')
            out_file.write('# Column 30: P(LAE)  ['+band2+'\'] equvalent width \n')
         out_file.write('# \n')
         out_file.write('# \n')

         if objlabel[j] == 'LAE': start,end = 0,cLAE
         elif objlabel[j] == 'OII': start,end = cLAE,cLAE+cOII

         if noise[k] == 'noiseless':
            for i in range(start,end):
               out_file.write(str(c13[i])+'\t'
                              +'%.3f'%(c05[i])+'\t'
                              +'%.5e'%(true_em_line_flux[i])+'\t'
                              +'%.3f'%(f_nu_at_EL[0][i])+'\t'
                              +'%.3f'%(phot_EW_obs_at_EL[0][i])+'\t'
                              +'%.3f'%(g_true_cont_flux_dens[i])+'\t'
                              +'%.3f'%(g_true_ew_obs[i])+'\t'
                              +'%.3f'%(r_true_cont_flux_dens[i])+'\t'
                              +'%.3f'%(r_true_ew_obs[i])+'\t'
                              +'%.4f'%(true_g_minus_r[i])+'\t'
                              +'%.4f'%(noiseless_g_minus_r[i])+'\t'
                              +'%.5f'%(true_em_line_flux[i]/sigma_line_flux[i])+'\t'
                              +'%.3f'%(f_nu_at_EL[0][i]/phot_sigma_f_nu[i])+'\t'
                              +'%.5f'%(phot_EW_obs_at_EL[0][i]/phot_sigma_ew_obs[i])+'\t'
                              +'%.3f'%(g_true_s_to_n[i])+'\t'
                              +'%.5f'%(g_true_ew_obs[i]/g_sigma_ew_obs[i])+'\t'
                              +'%.3f'%(r_true_s_to_n[i])+'\t'
                              +'%.5f'%(r_true_ew_obs[i]/r_sigma_ew_obs[i])+'\t'
                              +'%.5e'%(sigma_line_flux[i])+'\t'
                              +'%.5f'%(phot_sigma_f_nu[i])+'\t'
                              +'%.3f'%(phot_sigma_ew_obs[i])+'\t'
                              +'%.5f'%(g_sigma_f_nu[i])+'\t'
                              +'%.3f'%(g_sigma_ew_obs[i])+'\t'
                              +'%.5f'%(r_sigma_f_nu[i])+'\t'
                              +'%.3f'%(r_sigma_ew_obs[i])+'\t'
                              +'%.4f'%(sigma_g_minus_r[i])+'\t'
                              +'%.4f'%(calculated_dcolor[i])+'\n')
                              
         elif noise[k] == 'noisified':
            for i in range(start,end):
               out_file.write(str(c13[i])+'\t'
                              +'%.3f'%(c05[i])+'\t'
                              +'%.5e'%(c08[i])+'\t'
                              +'%.3f'%(f_nu_at_EL[1][i])+'\t'
                              +'%.3f'%(phot_EW_obs_at_EL[1][i])+'\t'
                              +'%.3f'%(g_c09[i])+'\t'
                              +'%.3f'%(g_c07[i])+'\t'
                              +'%.3f'%(r_c09[i])+'\t'
                              +'%.3f'%(r_c07[i])+'\t'
                              +'%.4f'%(g_minus_r[i])+'\t'
                              +'%.4f'%(noisified_g_minus_r[i])+'\t'
                              +'%.5f'%(c08[i]/sigma_line_flux[i])+'\t'
                              +'%.3f'%(f_nu_at_EL[1][i]/phot_sigma_f_nu[i])+'\t'
                              +'%.5f'%(phot_EW_obs_at_EL[1][i]/phot_sigma_ew_obs[i])+'\t'
                              +'%.3f'%(g_c12[i])+'\t'
                              +'%.5f'%(g_c07[i]/g_sigma_ew_obs[i])+'\t'
                              +'%.3f'%(r_c12[i])+'\t'
                              +'%.5f'%(r_c07[i]/r_sigma_ew_obs[i])+'\t'
                              +'%.5e'%(sigma_line_flux[i])+'\t'
                              +'%.5f'%(phot_sigma_f_nu[i])+'\t'
                              +'%.3f'%(phot_sigma_ew_obs[i])+'\t'
                              +'%.5f'%(g_sigma_f_nu[i])+'\t'
                              +'%.3f'%(g_sigma_ew_obs[i])+'\t'
                              +'%.5f'%(r_sigma_f_nu[i])+'\t'
                              +'%.3f'%(r_sigma_ew_obs[i])+'\t'
                              +'%.4f'%(sigma_g_minus_r[i])+'\t'
                              +'%.4f'%(calculated_dcolor[i])+'\t'
                              +'%.4f'%(float(prob_LAE_over_prob_OII[2][i])/float(1+float(prob_LAE_over_prob_OII[2][i])))+'\t'
                              +'%.4f'%(float(prob_LAE_over_prob_OII[0][i])/float(1+float(prob_LAE_over_prob_OII[0][i])))+'\t'
                              +'%.4f'%(float(prob_LAE_over_prob_OII[1][i])/float(1+float(prob_LAE_over_prob_OII[1][i])))+'\n')
         
         out_file.close()
         print('imaging.write_results(), done writing out '+str(objlabel[j])+'_simulation_'+str(noise[k])+'_'+str(run)+'.dat: %.1f seconds' % (time.time()-t0))

   print('imaging.write_results() required %.1f seconds' % (time.time()-t0))
   gc.collect()
   
   print('imaging.write_results() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')


def get_data(run):
   print('imaging.get_data() began at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')
   t0 = time.time()
   
   global index, objtype, wl_obs, lineFlux, f_nu, ew_obs
   global g_minus_r, g_minus_i, g_minus_z, r_minus_i, r_minus_z, i_minus_z
   global sigma_line_flux, sigma_f_nu, sigma_ew_obs, sigma_g_minus_r
   global b_index, redshift, b_prob_ratio, b_prob_LAE, b_objtype
   global cLAE, cOII
   global zinf, ewinf, s_to_n, cont_mag, redshift, wlbin, ewinf
   
   objtype, wl_obs, lineFlux, f_nu, ew_obs, g_minus_r, g_minus_i, g_minus_z, r_minus_i, r_minus_z, i_minus_z, index = [[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]],[]
   sigma_line_flux, sigma_f_nu, sigma_ew_obs, sigma_g_minus_r = [],[],[],[]
   
   noise = ['flat_spectra','before_noise','with_noise']
   objlabel = ['LAE','OII']
   
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
               if noise[i] == 'flat_spectra':
                  index.append(str(simobject[11]))
         data.close()
         print('imaging.get_data(), done reading in simulated '+str(objlabel[j])+' '+str(noise[i])+' data: %.1f seconds' % (time.time()-t0))

   print(len(sigma_f_nu),len(sigma_ew_obs))
   
   cLAE = objtype[0].count('LAE')
   cOII = objtype[0].count('OII')
   print('check that lists match')
   print(cLAE == objtype[1].count('LAE'),cOII == objtype[1].count('OII'))
   
   b_index, redshift, b_prob_ratio, b_prob_LAE, b_objtype = [],[],[],[],[]
   
   data = open(str(path)+str(run)+'_bayesian_results.dat','r')
   for line in data.readlines():
      if not line.startswith('#'):
         _index, _redshift, _prob_ratio, _prob_LAE, _objtype = map(str, line.split())
         b_index.append(int(_index))
         redshift.append(float(_redshift))
         b_prob_ratio.append(float(_prob_ratio))
         b_prob_LAE.append(float(_prob_LAE))
         b_objtype.append(str(_objtype))
   data.close()
   print('imaging.get_data(), done reading in Bayesian separation results: %.1f seconds' % (time.time()-t0))
   
   if not len(objtype[0])==len(b_objtype): print('*** ERROR: data mismatch ***')
   gc.collect()

   sigma_line_flux = np.array(sigma_line_flux)
   sigma_f_nu = np.array(sigma_f_nu)
   sigma_ew_obs = np.array(sigma_ew_obs)
   sigma_g_minus_r = np.array(sigma_g_minus_r)
   ew_obs = np.array(ew_obs)
   wl_obs = np.array(wl_obs)
   f_nu = np.array(f_nu)
   
   zinf = wl_obs/1215.668 -1      # inferred Lyman-alpha redshift
   ewinf = ew_obs/(1.+zinf)
   s_to_n = f_nu/sigma_f_nu
   
   cont_mag = [[],[],[]]
   cont_mag[0] = 23.9-2.5*(np.log10(f_nu[0]))      ## flat-spectra mag
   cont_mag[1] = 23.9-2.5*(np.log10(f_nu[1]))      ## noiseless mag
   
   foo = []
   for i in range(cLAE+cOII):
      if f_nu[2][i] > 0: foo.append(23.9-2.5*(np.log10(f_nu[2][i])))
      else: foo.append(999)
   cont_mag[2] = foo                                 ## noisified mag

   cont_mag = np.array(cont_mag)

   redshift = np.array(redshift)
   
   wlbin = []
   for i in range(cLAE+cOII):
      if wl_obs[0][i] < 3700: wlbin.append(0)
      elif wl_obs[0][i] >= 3700 and wl_obs[0][i] < 3900: wlbin.append(1)         ## no [O II] interlopers at wl < 3727
      elif wl_obs[0][i] >= 3900 and wl_obs[0][i] < 4100: wlbin.append(2)
      elif wl_obs[0][i] >= 4100 and wl_obs[0][i] < 4300: wlbin.append(3)
      elif wl_obs[0][i] >= 4300 and wl_obs[0][i] < 4500: wlbin.append(4)
      elif wl_obs[0][i] >= 4500 and wl_obs[0][i] < 4700: wlbin.append(5)
      elif wl_obs[0][i] >= 4700 and wl_obs[0][i] < 4900: wlbin.append(6)
      elif wl_obs[0][i] >= 4900 and wl_obs[0][i] < 5100: wlbin.append(7)
      elif wl_obs[0][i] >= 5100 and wl_obs[0][i] < 5300: wlbin.append(8)
      elif wl_obs[0][i] >= 5300: wlbin.append(9)
      else: print('### error: lambda out of spectral range ###')


   global scale
   global c00,c13,c05,c08,c09,c07,c15,c16,c17,c18,c19,c20,c10,c12,c14,c11,c01,c04,c03
   global true_ew_rest,true_ew_obs,true_ew_inf,true_cont_flux_dens,true_AB_cont_mag,true_s_to_n,true_em_line_flux
   global true_g_minus_r, true_g_minus_i, true_g_minus_z, true_r_minus_i, true_r_minus_z, true_i_minus_z
   global true_ew_rest_fs,true_ew_obs_fs,true_ew_inf_fs,true_cont_flux_dens_fs,true_AB_cont_mag_fs,true_s_to_n_fs
   global f_nu_cont_slope
   
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

   lambda_eff_g = 4750.
   lambda_eff_r = 6220.
   g_minus_r = c15
   m = 0.4 * g_minus_r / np.log10(lambda_eff_r/lambda_eff_g)
   f_nu_cont_slope = np.copy(m)
   
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
   
   c04 = nb.lumDist(c01)
   
   c03 = c08 * (4*np.pi*(3.085678e24*c04)**2)
   
   true_ew_rest_fs = true_ew_obs_fs/(1+c01)

   print('imaging.get_data() finished at: '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
   print('')



if __name__ == '__main__': sim_color()

