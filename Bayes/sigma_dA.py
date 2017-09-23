from datapath import *
import numpy as np


def sigma_dA ( fContam, fIncomp, cntLAErecovered, scale, bin, version, baserun ) :

	if 0 <= version <= 2 :

		if version == 'old' or version == 0 :
			
			if bin == '1.9-2.5' or bin == 0 :
				sigma2_dA = (fContam/0.025)**2 + (270000*scale)/cntLAErecovered
			
			elif bin == '2.5-3.5' or bin == 1 :
				sigma2_dA = (fContam/0.05)**2 + (360000*scale)/cntLAErecovered

		elif 1 <= version <= 2 :
		
			purity = 1.-fContam
			comp   = 1.-fIncomp
			x = [purity,comp]
			
			if version == 1:
		
				if bin == '1.9-2.5' or bin == 0 :
					a,b,c,d,e,f,g = 0.681689624681, -2.16250317052, 0.740562744622, -0.668443232827, 2.08179253546, -5.13850338411, -1.6775132763
				
				elif bin == '2.5-3.5' or bin == 1 :
					a,b,c,d,e,f,g = 0.951538667763, -1.93045963513, -4.5814059474, -1.84896416802, 8.03559001831, -3.46946410056, -1.73147001525
		
			elif version == 2:
				
				if bin == '1.9-2.5' or bin == 0 :
					a,b,c,d,e,f,g = 0.604236809478, -1.19216195965, 0.674273808159, -0.744444591131, 2.20603037196, -3.43169450808, -1.62080260745
				
				elif bin == '2.5-3.5' or bin == 1 :
					a,b,c,d,e,f,g = 0.918252305828, -0.559265810365, -28.9396959949, -1.70351167331, 32.4206021953, -0.725856758197, -1.68961729049
		
			sigma2_dA = a*x[0]**b + c*x[1]**d + e*x[0]**f*x[1]**g

		return np.sqrt(sigma2_dA)

	elif version >= 3 :
		
		x = [fContam,fIncomp]

		if version == 3 :

			if bin == '1.9-2.5' or bin == 0 :
				floor = 1.86481
				a,b,c,d,e,f,g = 3.32245291712, 1.08726762021, 1.82220092906, 1.1720357335, 12.4199946023, 0.845825026382, 2.37163507386
		
			elif bin == '2.5-3.5' or bin == 1 :
				floor = 2.13722
				a,b,c,d,e,f,g = 14.749980163, 1.40431000771, 2.80882389269, 1.30706702444, 21.4442583753, 1.57984728605, 0.930866040527

		elif version == 4:
			
			if baserun == '150525_34.38': survey_area = int(41253./2**2)
			elif baserun == '150525' or baserun[:7] == '150824_' : survey_area = 300 * float(baserun[7:])
			else: survey_area = 300.
			
			if   bin == '1.9-2.5' or bin == 0: opt = 'bin1'
			elif bin == '2.5-3.5' or bin == 1: opt = 'bin2'
			elif bin == '1.9-3.5' or bin == 2: opt = 'all'
			
			floor,a,b,c,d,e,f,g, = 0,0,0,0,0,0,0,0
			
			data = open(sudepath+'%.1f'%(survey_area)+'_coefficients_'+opt+'.dat','r')
			for line in data.readlines():
				thisline = line.split()
				if thisline[0] == 'floor':
					floor = float(thisline[2])
				if thisline[0]+' '+thisline[1]+' '+thisline[2] == 'With 7 params,':
					a = float(thisline[3])
					b = float(thisline[4])
					c = float(thisline[5])
					d = float(thisline[6])
					e = float(thisline[7])
					f = float(thisline[8])
					g = float(thisline[9])
			data.close()

		sigma_dA = floor + a*x[0]**b + c*x[1]**d + e*x[0]**f*x[1]**g

		return sigma_dA


