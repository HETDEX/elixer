#explore probabilities

RUN = False #if false, reads in the file, if True runs samples and makes a new file
FILE = "/home/dustin/code/python/elixer/exp_gauss_out.txt"
SAMPLES = 100000

import numpy as np
import scipy as sp
import imp
import sys
import spectrum
import matplotlib.pyplot as plt
from timeit import default_timer as timer


plt.switch_backend('QT4Agg')

SHOW_SPEC = False


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


def reload():
    imp.reload(spectrum)
    imp.reload(sys.modules[__name__])

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
        #return sp.stats.skewnorm.rvs(loc=0, scale=7.0, a=0.0, size=self.num_pix)


    def trial(self):
        so = spectrum.Spectrum()
        wavelengths = np.arange(self.w_min,self.w_min+(self.num_pix)*self.pix_size,self.pix_size) # actual wavelengths do not actaully matter
        values = self.rand_spec()

        errors = None

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


        if SHOW_SPEC and (len(peaks) > 0):# and plot:
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
            #plt.show()

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

        all_score = []
        all_line_flux = []
        all_snr  = []
        all_good = []
        all_eli = []
        all_num_good = [] #the number of "good" peaks found in a single spectrum

        total_samples = 0

        self.start_time = timer()
        #for i in range(self.num_trials):
        while total_samples < self.num_trials:
            score,line_flux, snr, good, eli, num_good = self.trial()

            samples = len(score)

            if samples > 0:
                total_samples += samples
                #percent = float(total_samples)/float(self.num_trials)
                #sys.stdout.write('\r')
                #sys.stdout.write("[%-20s] %g%% (N=%d)" % ('=' * int(percent / 5.0), percent, total_samples))
                #sys.stdout.flush()

         #       print(snr,good)
                all_score = np.concatenate((all_score,score))
                all_line_flux = np.concatenate((all_line_flux,line_flux))
                all_snr = np.concatenate((all_snr, snr))
                all_good = np.concatenate((all_good, good))
                all_eli = np.concatenate((all_eli, eli))
                all_num_good = np.concatenate(all_num_good,num_good)
              #  for s,g in zip(snr,good):

                rate = total_samples/(timer() - self.start_time)
                tot_sec = int((self.num_trials - total_samples)/rate)
                h = tot_sec / 3600
                m = (tot_sec - h*3600)/60
                s = (tot_sec - h*3600 - m*60)

                print("Samples: %d  ETA: %02d:%02d:%02d" % (total_samples,h,m,s) )



        #print (len(all_snr),len(np.where(all_good)[0]))

        with open("exp_gauss_out.txt","w") as out:
            out.write("#Good  SNR  Sigma  LineFlux  Cont  SBR  EW_Obs  Score\n")
            for eli in all_eli:
                eli.build(values_units=-18)
                if eli.line_flux == -999:
                    eli.line_flux = 0.0
                out.write("%d  %f  %f  %f  %f  %f  %f  %f %d\n"
                      % (eli.is_good(), eli.snr, eli.fit_sigma, eli.line_flux, eli.cont , eli.sbr, eli.eqw_obs,
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
        score = out[:,7]
        line_flux = out[:,3]
        snr = out[:,1]
        good = out[:,0].astype(int)
        # Good  SNR  Sigma  LineFlux  Cont  SBR  EW_Obs  Score
        for i in range(len(out)):
            e = spectrum.EmissionLineInfo()
            e.snr = out[i,1]
            e.fit_sigma = out[i,2]
            e.line_flux = out[i,3]
            if e.line_flux == -999:
                e.line_flux = 0.0
            e.cont = out[i,4]
            e.sbr = out[i,5]
            e.eqw_obs = out[i,6]
            e.line_score  = out[i,7]

            eli.append(e)

        return np.array(score), np.array(line_flux),np.array(snr),np.array(good), np.array(eli)


def main():
    gf = GaussFit()

    if (RUN):
        score, line_flux, snr, good, eli, num_good = gf.go()
    else:
        score, line_flux, snr, good, eli = gf.load(FILE)
        num_good = None #can only know at original runtime (at least as currently implemented)


    all_score = np.histogram(score, bins=gf.score_bin_edges)

    # really almost meaningless ... as the score increases it will necessarily be marked as "good" so
    # the fraction of "good" scores rises with the scores
    good_score = np.histogram(score[np.where(good)[0]], bins=gf.score_bin_edges)

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
