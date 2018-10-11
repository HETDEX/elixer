#explore probabilities

RUN = False #if false, reads in the file, if True runs samples and makes a new file
FILE = "/home/dustin/code/python/elixer/exp_per_wave_gauss_out.txt"
SAMPLES = 100  #now, as samples PER wavelength

#todo: still need to update the plotting to be per-wavelength
#todo: change trial to just do a fit at the cw ... do not run peakdet

import numpy as np
import scipy as sp
import imp
import sys
import os.path as op
import spectrum
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import collections
from scipy.optimize import curve_fit


plt.switch_backend('QT4Agg')

SHOW_SPEC = False
SPECTRUM_WIDTH = 1000 #in 2AA steps
UNC_PARAMS_FILE = "/home/dustin/code/python/elixer/noise/unc_fit_parms.txt"
FLUX_PARAMS_FILE = "/home/dustin/code/python/elixer/noise/flux_fit_parms.txt"
NOISE_SAMPLES_FILE = "noise_lines.txt"


# N = noise , f = flux or score or whatever criteria
# want to know P(N|f) so ... P(N|f) = P(f|N)*P(N)/p(f)
#
# P(f|N) is what we find here .... really P(f|N)*P(N) since P(N) here is 1.0 (all samples are only noise, no signal)
# but is it okay that P(N) = 1.0 ... that feels not right (while true for this set, it would not be true for all spectra)
# what is p(f)?
# there is a wrinkle, of course, since I am only considering those lines that pass the "is_good" test
# so this is more like P(N|f,g) where g is passing the good test
# so ...
# P(N|f,g) = P(f,g|N)*P(N) / {P(f|g)*p(g)}

# could get at p(g) by removing the restriction that it must pass the is_good test (e.g. # passing / # total)
# then p(f|g) would be fraction of each score bin out of those that were good
# and p(f,g|N) ... well, f,g are correlated, but letting p(f,g|N) --> just p(f,g) = p(f|g)*p(g) leaves us with
#     P(N|f,g) == 1 (which is kind of true in here since ALL samples are noise)

# ?? how to objectively determine total P(N)
#
#


#Basically, generate full (random noise) spectra and scan for "peaks" using the elixer peak finder
#to produce xxx (say 100,000) identifiable peaks (ie. that a gaussian fits)
#For each peak, get its properties (SNR, EW, etc) and produce its line_score (per elixer)
#Then bin up those line_scores and sum to the right (such that the bin width does not matter much)
#The prob that a line_score is random is that line_score's bin / total number of scores
#  so, if there are 1537 line_score in the 9.0 line_score center bin with another 523 in bins 10.0 to inf,
# the prob of random is (1537+523)/100000 = 0.02060



#6-18-2018 update
#slightly different logic .... still get num_trials, but only accept IF is_good(), so the resulting set is all peaks
#by score that elixer would classify as good. The fraction then, is by score/total ... out of the universe of noise,
#the fraction (by score) that are considered good == probability by score that the fit is just noise


#these are for the resulting PDF, not for fitting the emission line itself
def polynomial(x,coeff):
    return np.poly1d(coeff)(x)

def gaussian(x,x0,sigma,a=1.0,y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None
    #have the / np.sqrt(...) part so the basic shape is normalized to 1 ... that way the 'a' becomes the area
    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y


def exponential(x,A,r):
    if (x is None) or (A is None) or (r is None):
        return None
    return A*np.exp(r*x)

def reload():
    imp.reload(spectrum)
    imp.reload(sys.modules[__name__])


class SpectrumDistro:
    def __init__(self,filename):
        self.parms = [] #array of parameter dicts
        self.waves = [] #just the wavelengths to index
        self.sample = {}

        self.read_file(filename)
        sp.random.seed(seed=1138) #fixed for now so should get same sequnce


    def read_file(self,filename):
        try:
            #with open(filename, "r") as f:
            #int(wm[0, c]), parm[0], parm[1], parm[2], parm[3], sk
            #wavelength, x0, sigma, Area, y-offset, skew
            #self.waves = []
            self.parms = []
            out = np.loadtxt(filename, dtype=float) # x by 6

            for p in out:
                self.parms.append({"wave":p[0],"mu":p[1],"sigma":p[2],"area":p[3],"y":p[4],"skew":p[5]})
                self.waves.append(p[0])

            self.waves = np.array(self.waves)

        except:
            print("Exception reading file in SpectrumDistro")

    #
    # def distro_spectrum(self,center,wings):
    #     #center = center wavelength
    #     #wings = how many indicies to either side of the center to sample from the respective noise distro
    #     #(each wavelength bin or index has its own noise PDF distribution from which to sample)
    #
    #     #find the parameters nearest to center
    #     idx = (np.abs(self.waves - center)).argmin()
    #
    #     left = max(0,idx-wings)
    #     right = min(SPECTRUM_WIDTH,idx+wings+1)
    #     spec = np.zeros(SPECTRUM_WIDTH)
    #
    #     for i in np.arange(left,right):
    #         spec[i] = sp.stats.skewnorm.rvs(loc=self.parms[i]['mu'],
    #                                             scale=self.parms[i]['sigma'],
    #                                             a=self.parms[i]['skew'],size=1)
    #     return spec


    def distro_spectrum(self,wavelengths):
        #center = center wavelength
        #wings = how many indicies to either side of the center to sample from the respective noise distro
        #(each wavelength bin or index has its own noise PDF distribution from which to sample)

        #find the parameters nearest to center
        spec = np.zeros(len(wavelengths))

        for w in wavelengths:
            i = spectrum.getnearpos(self.waves,w)

            spec[i] = sp.stats.skewnorm.rvs(loc=self.parms[i]['mu'],
                                                scale=self.parms[i]['sigma'],
                                                a=self.parms[i]['skew'],size=1)

        return spec




class GaussFit:

    #all class vars
    num_trials = SAMPLES #start with 1000 just to test
    #note: because an entire spectrum is generated and many peaks are evaluated per spectrum
    #      the actualy 'number of trials' is much larger where the meaning is per peak not per spectrum
    num_pix = 1001
    w_min = 3500.0
    pix_size = 2.0 #AA
    score_bin_centers = np.arange(0.5,100.0,1.0)
    score_bin_edges = score_bin_centers - 0.5
    score_bin_edges = np.append(score_bin_edges, 100.0)

    start_time = None


    def __init__(self):
        pass

    def rand_spec(self):
        #return a random (noise only) 'spectrum'
        return sp.stats.skewnorm.rvs(loc=0, scale=7.0, a=1.5, size=self.num_pix)
        #loc = mu, scale = standard deviation, a = skew parameter (alpha)
        #return sp.stats.skewnorm.rvs(loc=0, scale=7.0, a=0.0, size=self.num_pix)


    #skewnormal
    # 2 * norm.pdf(x) * norm.cdf(a*x)
    #def make_spectrum(self,):




    def trial(self,cw,sd_flux,sd_unc): #cw is central wavelength

        so = spectrum.Spectrum()

        wavelengths = np.arange(self.w_min,self.w_min+(self.num_pix)*self.pix_size,self.pix_size) # actual wavelengths do not actaully matter
        #values = sd_.distro_spectrum(center=cw,wings=25)*10.0  #*10.0 so can keep the -18 units
        #values = self.rand_spec() #old, not PER wavelength

        #values = sd_flux.distro_spectrum(center=cw, wings=1000) * 10.0  # *10.0 so can keep the -18 units
        #errors = sd_unc.distro_spectrum(center=cw,wings=1000)*10.0  #*10.0 so can keep the -18 units
        values = sd_flux.distro_spectrum(wavelengths) * 10.0  # *10.0 so can keep the -18 units
        errors = sd_unc.distro_spectrum(wavelengths)*10.0  #*10.0 so can keep the -18 units
        # 6-18-2018 additional condition, enforce_good = True, min_sigma = 1.0 ... just as per normal
        peaks = spectrum.peakdet(wavelengths, values, errors, values_units=-18,enforce_good=True,min_sigma=1.0)

        score = []
        line_flux = []
        snr = []
        good = []
        eli = []

        num_good = 0

        #plot = False
        for p in peaks:
            #if p.snr > 40.0:
            #    plot = True
            #    print("Big SNR ...")
            #    xsnr = p.snr
            if p.is_good(): #6-18-2018 additional condition
                score.append(p.line_score)
                line_flux.append(p.line_flux)
                snr.append(p.snr)
                good.append(p.is_good())
                num_good += 1
                eli.append(p)

        # p = spectrum.signal_score(wavelengths, values, errors, central=cw,
        #                             values_units=-18, min_sigma=1.0, absorber=False)
        #
        # if p is not None:
        #     if p.is_good():
        #         score.append(p.line_score)
        #         line_flux.append(p.line_flux)
        #         snr.append(p.snr)
        #         good.append(p.is_good())
        #         num_good += 1
        #         eli.append(p)

        # plt.close()
        # plt.figure(figsize=(8, 2), frameon=False)
        # plt.plot(wavelengths, values)
        # plt.show()


        if SHOW_SPEC and p is not None: #and (len(peaks) > 0):# and plot:
            plt.close()
            plt.figure(figsize=(8, 2), frameon=False)
            plt.plot(wavelengths,values)
            x = [p.raw_x0 for p in peaks]
            y = [p.raw_h for p in peaks]  # np.array(peaks)[:, 2]

            plt.scatter(x, y, facecolors='none', edgecolors='r', zorder=99)
            plt.xlim((min(wavelengths),max(wavelengths)))

            for i in range(len(peaks)):
                h = peaks[i].raw_h
                plt.annotate("%0.1f" % peaks[i].line_score, xy=(peaks[i].fit_x0, h), xytext=(peaks[i].fit_x0, h),
                                  fontsize=6, zorder=99)

            #plt.savefig("exp_score_%0.2g.png" % (xscore))
           # plt.close()
            plt.show()

        return score,line_flux, snr, good, eli, num_good

        # a_central = wavelengths[len(wavelengths)/2]
        # central_z = 0.0
        # values_units = -18
        #
        # eli = spectrum.signal_score(wavelengths, values, errors, a_central,
        #                    central_z=central_z, values_units=values_units, spectrum=None)
        #
        # if eli is not None:
        #     return eli.snr, eli.is_good(z=0.0)
        # else:
        #     return 0.,False

    def go(self):

        all_wave = [] #wavelengths
        all_score = []
        all_line_flux = []
        all_snr  = []
        all_good = []
        all_eli = []
        all_num_good = [] #the number of "good" peaks found in a single spectrum

        total_samples = 0
        total_trials = 0
        sd_unc = SpectrumDistro(filename=UNC_PARAMS_FILE)
        sd_flux = SpectrumDistro(filename=FLUX_PARAMS_FILE)

        wave_range = np.arange(3500.0, 5500.0, 2.0)
        wave_bins = np.zeros(len(wave_range))

        self.start_time = timer()
        #for i in range(self.num_trials):



        #for debugging

        #for cw in np.arange(3510.00, 5490.00,2.0):
        #    total_samples = 0 #total samples is not per wavelength
       # while total_samples < self.num_trials:

        exists = False
        if op.isfile(NOISE_SAMPLES_FILE):  # alread exists
            exists = True

        with open(NOISE_SAMPLES_FILE,"a+") as out:
            if exists:#alread exists
                txt = np.loadtxt(NOISE_SAMPLES_FILE,skiprows=1)

                all_score = txt[:,8]
                all_line_flux = txt[:,4]
                all_snr = txt[:,2]
                all_good = txt[:,1]
                total_trials = txt[:,9][-1]
                total_samples = txt[:, 10][-1]


                bins = txt[:,0]
                for b in bins:
                    i = spectrum.getnearpos(wave_range,b)
                    wave_bins[i] += 1

            else:
                out.write("#Wavelength Good  SNR  Sigma  LineFlux  Cont  SBR  EW_Obs  Score Trials Samples\n")

            done = False

            try:
                while (np.any([(w<self.num_trials) for w in wave_bins])) and (not done):

                    score,line_flux, snr, good, eli, num_good = self.trial(4500.0,sd_flux,sd_unc)
                    total_trials += 1

                    samples = len(score)

                    #print (iter, total_samples)
                    if samples > 0:
                        total_samples += samples
                        #percent = float(total_samples)/float(self.num_trials)
                        #sys.stdout.write('\r')
                        #sys.stdout.write("[%-20s] %g%% (N=%d)" % ('=' * int(percent / 5.0), percent, total_samples))
                        #sys.stdout.flush()

                 #       print(snr,good)
                        #all_wave = np.concatenate((all_wave,cw))
                        for e in eli:
                            #round to nearest step
                            w = spectrum.getnearpos(wave_range,e.fit_x0)
                            all_wave.append(wave_range[w])
                            wave_bins[w] += 1

                            #print(wave_range[w])

                        plt.clf()
                        #plt.title(title)
                        plt.plot(wave_range, wave_bins)
                        # plt.ion()
                        plt.pause(0.01)
                        plt.show(block=False)

                        all_score = np.concatenate((all_score,score))
                        all_line_flux = np.concatenate((all_line_flux,line_flux))
                        all_snr = np.concatenate((all_snr, snr))
                        all_good = np.concatenate((all_good, good))
                        all_eli = np.concatenate((all_eli, eli))
                        #all_num_good = np.concatenate(all_num_good,num_good)
                        all_num_good.append(num_good)
                      #  for s,g in zip(snr,good):

                        #rate = total_samples/(timer() - self.start_time)
                        #tot_sec = int((self.num_trials - total_samples)/rate)
                        tot_sec = int(timer()-self.start_time)

                        h = tot_sec / 3600
                        m = (tot_sec - h*3600)/60
                        s = (tot_sec - h*3600 - m*60)

                        print("Trials: %d Samples: %d  ELAPSED: %02d:%02d:%02d  min(%d) max(%d)" % (total_trials, total_samples,h,m,s,np.min(wave_bins),np.max(wave_bins)) )

                        for e in eli:
                            e.build(values_units=-18)
                            if e.line_flux == -999:
                                e.line_flux = 0.0
                            out.write("%d  %f  %f  %f  %f  %f  %f  %f %f %d %d\n"
                                      % (
                                      e.fit_x0, e.is_good(), e.snr, e.fit_sigma, e.line_flux, e.cont, e.sbr,
                                      e.eqw_obs, e.line_score, total_trials, total_samples))
                            out.flush()

                        #todo: plot histogram of bins so can watch them fill in
                        #todo: remove clock estimate (no longer valid) (or make s|t it is based on the current smallest bin)

            except (KeyboardInterrupt, SystemExit):
                done = True

            except:
               print("Some exception ... keep going")

        #print (len(all_snr),len(np.where(all_good)[0]))

        if False:
            with open("exp_gauss_out.txt","w") as out:
                out.write("#Wavelength Good  SNR  Sigma  LineFlux  Cont  SBR  EW_Obs  Score\n")
                for eli in all_eli:
                    eli.build(values_units=-18)
                    if eli.line_flux == -999:
                        eli.line_flux = 0.0
                    out.write("%d  %f  %f  %f  %f  %f  %f  %f %d\n"
                          % (eli.fit_x0 , eli.is_good(), eli.snr, eli.fit_sigma, eli.line_flux, eli.cont , eli.sbr, eli.eqw_obs,
                             eli.line_score))



        return all_score,all_line_flux, all_snr, all_good, all_eli, all_num_good


    def load(self,filename): #no error control

        score = []
        line_flux = []
        snr = []
        good = []
        eli = []
        #num_good = [] #only known at original run time

        #sort of a dumb way to do this, but convenient to put in the ELI objects
        out =  np.loadtxt(filename,dtype=np.float64)
        wave = out[:,0].astype(int)
        score = out[:,8]
        line_flux = out[:,4]
        snr = out[:,2]
        good = out[:,1].astype(int)

        total_trials = out[:,9].astype(int)[-1]
        total_samples = out[:, 10].astype(int)[-1]

        # Good  SNR  Sigma  LineFlux  Cont  SBR  EW_Obs  Score
        for i in range(len(out)):
            e = spectrum.EmissionLineInfo()
            e.snr = out[i,2]
            e.fit_sigma = out[i,3]
            e.line_flux = out[i,4]
            if e.line_flux == -999:
                e.line_flux = 0.0
            e.cont = out[i,5]
            e.sbr = out[i,6]
            e.eqw_obs = out[i,7]
            e.line_score  = out[i,8]

            eli.append(e)

        return np.array(wave), np.array(score), np.array(line_flux),np.array(snr),np.array(good), np.array(eli), \
               total_trials, total_samples


def main():
    gf = GaussFit()

    if (RUN):
        score, line_flux, snr, good, eli, num_good = gf.go()
    #else:
        #wave, score, line_flux, snr, good, eli = gf.load(FILE)
        #num_good = None #can only know at original runtime (at least as currently implemented)

    wave, score, line_flux, snr, good, eli, trials, samples = gf.load(NOISE_SAMPLES_FILE)

    all_noise_count = len(good)

    #todo: break these up by wavelength bin but do essentially the same thing
    #todo: want a file as output with wavelength then tuples (score,fraction) (score,fraction) ...
    #todo: -OR- can we reliably fit a curve? then it would be wavelength then params
    #todo: -OR- wavelengths (score, count) (score, count) .... and figure precentages on the read in side
    #todo:              this lets us deal with very small counts where maybe only 1 or 2 instances occur at even lowest score
    #todo:      if only one or few? what is a minimum number for statistics
    #todo:         implies it is difficult to get a random line even at lowest score


    #get the unique wavelengths (note: on the read side, will find the nearest wavelength to the target wavelength
        # and if that distance in wavelength is greater than say 2, then assume there are zero hits at that wavelength


    #save as an array of dictionaries
    # the dictionary has two keys: "scores" and "frac" whose values are the arrays

    out = {}


    #with open("random_signal_by_wavebin.txt","w+") as f:
    if True:
        set_of_waves = sorted(set(wave))

        for w in set_of_waves:
            #center on the wavelength, +/- 1 AA  e.g. 3560 == 3559 + 3560 + 3561, so 2AA wide bin centered on w
            #so most wavelengths get counted in more than one bin, but that is okay and what I want to smooth it out
            # e.g. 3561 = 3560 + 3561 + 3562
            #      3562 = 3561 + 3562 + 3563  ... so 3561 appears in three bins
            # in many cases though, there is not a 1 AA adjacent wave-bin, so we just add in zero
            a = np.where( abs(wave-w) <= 1.0)[0] #list of indicies that match this wavelength (techinally a tuple, hence the [0])

            all_score = np.histogram(score[a], bins=gf.score_bin_edges)
            all_score = all_score[0].astype(float)
            pop_frac = all_score / len(a)
            trial_frac = all_score / trials
            sample_frac = all_score / samples


            #rather hand-wavy, but thinking about counting or binning errors,
            #force this to be decreasing or level
            #sometimes get a low bin in-between two higher bins ,so set the low middle bin = max of what remains
            # usually, the very next bin (sometimes two bins to the right, for very low counts)
            # and force a minimum of 0.05
            first = np.nonzero(pop_frac)[0][0]
            last = np.nonzero(pop_frac)[0][-1]

            for i in range(first, last):
                pop_frac[i] = max(max(pop_frac[i:]),0.05)

            indicies = np.where(pop_frac != 0)[0]

            #now, turn into cumulative sum for trial and sample fractions
            trial_frac = np.flip(np.cumsum(np.flip(trial_frac, axis=0)), axis=0)
            sample_frac = np.flip(np.cumsum(np.flip(sample_frac, axis=0)), axis=0)

            d = {'scores':gf.score_bin_centers[indicies],'counts':all_score[indicies],'fracs':pop_frac[indicies],
                 'trial_fracs':trial_frac, 'sample_fracs':sample_frac}
            out[w] = d

          #   #visualize
          #   # plt.xlim(xmax=20) #essentially down to zero by SNR = 20
          #   plt.plot(gf.score_bin_centers, pop_frac,
          #            zorder=2)  # distro of ALL possible signals (e.g. a gaussian was fit, though may not be good)
          #   plt.bar(gf.score_bin_centers, pop_frac, color='g', alpha=0.5)
          #   plt.ylabel("fraction")
          #   plt.xlabel("line score")
          #   plt.xlim(0, 20)
          #   plt.title("Pseudo-PDF (MDF) of Random Emission Line Scores for %d +/- 1 AA" %w)
          #   # plt.savefig("random_emission_line_score_pdf.png")
          #
          #
          #
          #   #want to fit a curve ... exponential decay maybe? or just a bin count (set minimum to the rightmost value
          #   #regardless of the actually minimum??) For small counts have a binarization (counting) problem where there
          #   #can be multiple zero counts between non-zero counts just by random chance
          #   #start from the score = 2.5 bin (or the maximum count)
          #
          #   #
          #   # the exponential fit just does not do well enough ... will just record the (adjusted) bin values
          #   #
          #   x_start = gf.score_bin_centers[np.argmax(pop_frac)]
          #   x_stop = gf.score_bin_centers[ np.where(pop_frac != 0)[0][-1] ] #last non-zero element
          #   x_grid = np.linspace(x_start,x_stop,1000)
          #   x = gf.score_bin_centers[ np.where(pop_frac != 0)[0]]
          #   y = pop_frac[np.where(pop_frac != 0)[0]]
          #   unc =  np.sqrt(y * len(a))/len(a) #treat as Poissin with noise = sqrt(Num samples) at each bin
          #   plt.errorbar(x,y,yerr=unc)
          #   try:
          #       parm, pcov = curve_fit(exponential, x, y,sigma=unc)
          #
          #       plt.title("wave = %d, exp=%f, A=%f\nunc_exp=%f, unc_A=%f" %(int(w), parm[0], parm[1], pcov[0][0], pcov[1][1]))
          #       plt.plot(x_grid,exponential(x_grid,parm[0],parm[1]),color='r')
          #       plt.plot(x_grid,exponential(x_grid,parm[0]+pcov[0][0],parm[1]+pcov[1][1]),color='g')
          #   except:
          #       plt.title("wave = %d, Fit Failed" %(int(w)))
          #
          #
          #   plt.ylim(0,1)
          #   plt.savefig("noise_fig_w%d.png" %(int(w)))
          #
          # #  plt.show()
          #   plt.close()

    np.save('random_signal_dict',out)

    o2 = np.load('random_signal_dict.npy').item()

    exit(0)

    all_score = np.histogram(score, bins=gf.score_bin_edges)

    # really almost meaningless ... as the score increases it will necessarily be marked as "good" so
    # the fraction of "good" scores rises with the scores
   # good_score = np.histogram(score[np.where(good)[0]], binso2.shape=gf.score_bin_edges)

    all_score = all_score[0].astype(float)
    good_score = good_score[0].astype(float)
    frac_good = good_score / all_score #may be some 0/0
    frac_good = np.nan_to_num(frac_good)

    #this is really all we want ... what fraction does each score bin represents of the total scores
    #pop_frac = all_score   # fraction of the population in each snr bin
    pop_frac = all_score / gf.num_trials #6-18-2018 version

    #6-18-2018 remove this condition
    #truncate to the bin where the cumulative fraction drops below about 0.0002
    #for i in range(len(pop_frac)):
    #    pop_frac[i] = np.sum(pop_frac[i:]) / gf.num_trials

    # create two arrays ... one of the score bins and one with the fraction of total (equiv. to prob of noise)
    keep_idx = np.where(pop_frac >= 0.0002)[0]
    keep_bins = gf.score_bin_centers[keep_idx]
    #want to sum to the right (s|t lowest bin is sum of all higher bins)
    keep_frac = pop_frac[keep_idx]

    print("Keep Bin Centers:")
    print(repr(keep_bins))
    print("\nKeep frac:")
    print(repr(keep_frac))



    #plt.xlim(xmax=20) #essentially down to zero by SNR = 20
    plt.plot(gf.score_bin_centers,pop_frac,zorder=2) #distro of ALL possible signals (e.g. a gaussian was fit, though may not be good)
    plt.bar(gf.score_bin_centers,pop_frac,color='g',alpha=0.5)
    plt.ylabel("fraction")
    plt.xlabel("line score")
    plt.xlim(0,20)
    plt.title("Pseudo-PDF (MDF) of Random Emission Line Scores")
    #plt.savefig("random_emission_line_score_pdf.png")

    plt.show()


    #look at SNR
    plt.close()
    plt.title("SNR")
    plt.hist([ob.snr for ob in eli],bins=gf.score_bin_edges) #all
    plt.hist([ob.snr for ob in eli[np.where(good)[0]]],bins=gf.score_bin_edges) #just the good ones
    plt.show()

    # look at SBR
    plt.close()
    plt.title("SBR")
    vals,bins,_ = plt.hist([ob.sbr for ob in eli],bins=50)  # all
    plt.hist([ob.sbr for ob in eli[np.where(good)[0]]],bins=bins)  # just the good ones
    plt.show()

    # look at sigma
    plt.close()
    plt.title("Sigma")
    vals, bins, _ = plt.hist([ob.fit_sigma for ob in eli],bins=50)  # all
    plt.hist([ob.fit_sigma for ob in eli[np.where(good)[0]]],bins=bins)  # just the good ones
    plt.show()

    # look at EW
    #clean up
    plt.close()
    plt.title("EW_obs")
    vals, bins, _ = plt.hist(np.clip([ob.eqw_obs for ob in eli],0,1000),bins=50)  # all
    plt.hist([ob.eqw_obs for ob in eli[np.where(good)[0]]],bins=bins)  # just the good ones
    plt.show()


    # look at line_flux
    plt.close()
    plt.title("line_flux")
    vals, bins, _ =plt.hist(np.clip([ob.line_flux for ob in eli],0,np.inf),bins=50)  # all
    plt.hist([ob.line_flux for ob in eli[np.where(good)[0]]],bins=bins)  # just the good ones
    plt.show()


    # look at line_flux
    plt.close()
    plt.title("DX0")
    vals, bins, _ =plt.hist(np.clip([ob.fit_dx0 for ob in eli],0,np.inf),bins=50)  # all
    plt.hist([ob.line_flux for ob in eli[np.where(good)[0]]],bins=bins)  # just the good ones
    plt.show()

    # look at raw_score
    # plt.close()
    # plt.title=("raw_score")
    # vals, bins, _ =plt.hist([ob.raw_score for ob in eli],bins=50)  # all
    # plt.hist([ob.raw_score for ob in eli[np.where(good)[0]]],bins=bins)  # just the good ones
    # plt.show()



    #look at possible correlations

    #SNR vs SBR (there isn't one)
    plt.close()
    plt.title("SNR vs SBR")
    plt.xlabel("SNR")
    plt.ylabel("SBR")
    plt.scatter([ob.snr for ob in eli], [ob.sbr for ob in eli])
    plt.show()


    #SNR vs sigma (there isn't one)
    plt.close()
    plt.title("SNR vs Sigma")
    plt.xlabel("SNR")
    plt.ylabel("sigma")
    plt.scatter([ob.snr for ob in eli], [ob.fit_sigma for ob in eli])
    plt.show()


    #SNR vs sigma (a little, weak)
    plt.close()
    plt.title("SNR vs EW")
    plt.xlabel("SNR")
    plt.ylabel("EW")
    plt.scatter([ob.snr for ob in eli], [ob.eqw_obs for ob in eli])
    plt.show()


    #SNR vs line flux (a little, weak)
    plt.close()
    plt.title("SNR vs line_flux")
    plt.xlabel("SNR")
    plt.ylabel("Line Flux")
    plt.scatter(np.array([ob.snr for ob in eli]), np.array([ob.line_flux for ob in eli])*1e18)
    plt.show()



    #SNR vs line flux (a little, weak)
    plt.close()
    plt.title("SNR vs line_score")
    plt.xlabel("SNR")
    plt.ylabel("Line Score")
    plt.scatter(np.array([ob.snr for ob in eli]), np.array([ob.line_score for ob in eli]))
    plt.show()


    plt.close()
    plt.title("EW vs sigma")
    plt.xlabel("EW")
    plt.ylabel("sigma")
    plt.scatter(np.array([ob.eqw_obs for ob in eli]), np.array([ob.fit_sigma for ob in eli]))
    plt.show()


    plt.close()
    plt.title("EW vs Line Flux")
    plt.xlabel("EW")
    plt.ylabel("Line Flux")
    plt.scatter(np.array([ob.eqw_obs for ob in eli]), np.array([ob.line_flux for ob in eli])*1e17)
    plt.show()

    #histogram
    #plt.hist(good_snr,bins=gf.snr_bin_edges)
    #plt.savefig("hist.png")
    #plt.show()



if __name__ == '__main__':
    main()
