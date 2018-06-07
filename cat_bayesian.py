## Bayesian priors, etc used to help quantify posterior probability that a catalog object is a match for our observation
from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import global_config as G
import os.path as op
import numpy as np
from scipy.optimize import curve_fit

log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)


#todo: read in distance information, for now convert to mdf
#todo: later fit sigmoid and estimate pdf from the mdf

DISTANCE_PRIOR_FILENAME ="distance_list_bool.txt"# "distance_list.txt"
MIRROR = False #should probably stay false ... mirror to 3rd quadrant for sigmoid seems to make fit worse

PDF_FUNC = "poly"
#PDF_FUNC = "sigmoid"
POLY_DEG = 3


# def sigmoid(x, x0, k): #generalized logistic (Richard's curve)
#     #A = 0
#     #K = 1
#     #Q ~ 0.001
#     return  1. / (1. + 0.001 * np.exp(-k * (x - x0)))

def sigmoid(x, x0, k): #logistic
    return  1. / (1. + np.exp(-k * (x - x0)))

def polynomial(x,coeff):
    return np.poly1d(coeff)(x)

def gaussian(x,x0,sigma,a=1.0,y=0.0):
    if (x is None) or (x0 is None) or (sigma is None):
        return None
    #have the / np.sqrt(...) part so the basic shape is normalized to 1 ... that way the 'a' becomes the area
    return a * (np.exp(-np.power((x - x0) / sigma, 2.) / 2.) / np.sqrt(2 * np.pi * sigma ** 2)) + y






#distance prior needs as input the distance and the magnitude of the bid object
class DistancePrior:

    #all class vars
    num_trials = 1000
    #annuli_bin_centers = np.arange(0.0,30.2,0.2) #uncomment if NOT trimming last row
    annuli_bin_centers = np.arange(0.0, 30.0, 0.2) #does not include last row (radii bin center = 30.0")

    annuli_bin_edges = annuli_bin_centers - 0.1
    annuli_bin_edges[0] = 0.0
    annuli_bin_edges  = np.append(annuli_bin_edges,30.0)

    mag_bin_centers = np.arange(21.,29.,1.)
    mag_bin_edges = mag_bin_centers - 0.5
    mag_bin_edges[0] = 21.0
    mag_bin_edges = np.append(mag_bin_edges, 28.0)

    def __init__(self):
        self.mdf_matrix = None
        self.pdf_sigmoid_parms = None
        self.pdf_poly_parms = None
        self.build_mdfs()
        self.build_pdfs()

    def plot_sigmoid_fits(self):
        import pylab
        #assumes only 8 magnitude bins
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'teal']
        x = self.mdf_matrix[:,0]
        for i in range(len(self.pdf_sigmoid_parms)):
            y = sigmoid(x, *(self.pdf_sigmoid_parms[i]))
            pylab.plot(x, y, label=str(21 + i), c=color[i])
            pylab.plot(x, self.mdf_matrix[:, i + 1], '.', c=color[i])

        pylab.legend(loc='best')
        pylab.show()

    def plot_poly_fits(self):
        import pylab
        #assumes only 8 magnitude bins
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'teal']
        x = self.mdf_matrix[:,0]
        for i in range(len(self.pdf_poly_parms)):
            y = polynomial(x, self.pdf_poly_parms[i])
            pylab.plot(x, y, label=str(21 + i), c=color[i])
            pylab.plot(x, self.mdf_matrix[:, i + 1], '.', c=color[i])

        pylab.legend(loc='best')
        pylab.show()


    def build_mdfs(self):
        file = op.join(op.dirname(G.__file__),DISTANCE_PRIOR_FILENAME)
        try:
            out = np.genfromtxt(file, dtype=None,comments ="#")
        except:
            log.error("Cannot read distance prior file: %s" % file, exc_info=True)

        try:
            #reduce 2nd through nth column values by divid by num_trials, cap at 1.0
            #note: for the _bool.txt version, the cap of 1.0 is not necessary as they are already
            #      in number of successes (not number of objects)
            out[:,1:] /= self.num_trials
            out[:, 1:] = np.clip(out[:,1:],0.0,1.0)
            #cut the last row (1/2 sized bin)
            out = out[:-1,:]
            #note: these are not all strictly increasing and the last bin is only 1/2 sized

            #find the first = 1.0 and set all the rest
            for m in range(1,len(self.mag_bin_centers)+1):
                i = np.where(out[:,m]==1.0)[0] #array of all
                if len(i) > 0:
                    out[i[0]:,m] = 1.0

            self.mdf_matrix = out
        except:
            log.error("Cannot build mdf from distance prior file: %s" % file, exc_info=True)


    def build_pdfs(self):
        if self.mdf_matrix is None:
            self.build_mdfs()

            if self.mdf_matrix is None:
                log.error("Cannot build pdfs for distance priors")


        if PDF_FUNC == "sigmoid":
            if self.pdf_sigmoid_parms is not None:
                del self.pdf_sigmoid_parms[:]

            self.pdf_sigmoid_parms = []
            #to make sigmoid fit better, mirror the data to the 3rd quadrant??
            #NO ... fits better w/o that
            if MIRROR:
                xdata = np.concatenate((-1 * np.flip(self.annuli_bin_centers[1:],0), self.annuli_bin_centers))
            else:
                xdata = self.annuli_bin_centers

            #brightest to dimmest
            for mag_idx in range(len(self.mag_bin_centers)):
                ydata = self.mdf_matrix[:, mag_idx + 1]  # +1 since first column is just the distance
                if MIRROR:
                    ydata = np.concatenate((-1 * np.flip(ydata[1:],0), ydata))


                #No ... truncating to the first "1" does not really improve the fit
                #one_idx = np.argmax(ydata)
                try:
                    popt, pcov = curve_fit(sigmoid, xdata, ydata,
                                           p0=(5.0, 0.2),
                                           bounds=((-1000.0,0.0),(1000.0,10.0))
                                           )
                except:
                    log.error("Failure to fit sigmoid to mag = %g" %(self.mag_bin_centers[mag_idx]), exc_info=True)
                    print("Failure to fit sigmoid to mag = %g" % (self.mag_bin_centers[mag_idx]))
                    self.pdf_sigmoid_parms.append(None)
                    continue

                self.pdf_sigmoid_parms.append(popt)
        elif PDF_FUNC == "poly":
            if self.pdf_poly_parms is not None:
                del self.pdf_poly_parms[:]

            self.pdf_poly_parms = []

            xdata = self.annuli_bin_centers

            # brightest to dimmest
            for mag_idx in range(len(self.mag_bin_centers)):
                ydata = self.mdf_matrix[:, mag_idx + 1]  # +1 since first column is just the distance

                try:
                    popt = np.polyfit(xdata,ydata,POLY_DEG)
                except:
                    log.error("Failure to fit polynomial to mag = %g" %(self.mag_bin_centers[mag_idx]), exc_info=True)
                    print("Failure to fit polynomial to mag = %g" % (self.mag_bin_centers[mag_idx]))
                    self.pdf_poly_parms.append(None)
                    continue

                self.pdf_poly_parms.append(popt)


    def get_prior(self,dist,mag): #distance in arcsec
        #find in mdf the likelihood of a RANDOM match
        #return 1-likelihood(random)
        p_rand_match = 0.0
        try:
            #round dist to nearest 0.2" (get an index)
            #round mag to nearest 1.0 mag (get an index)

            # technically this would put the right edge in with the lower bin
            # but for an early, quick and dirty experiment, is okay
            dist_bin = round(float(dist) / 0.2) * 0.2
            mag_bin = round(float(mag))

            #find the corresponding indicies
            #should always have an entry (unless > 30" or mag is out of range)
            # in either case, will just return 1.0
            if (mag_bin < 21.0) and (dist_bin < 5.0): #super-bright and close
                p_rand_match = 0.0
            elif (dist_bin > 30.0): #way too far (outside of the distribution, so call it random)
                p_rand_match = 1.0
            elif (mag_bin < 21.0): #bright (so outside the distribution, but still far-ish away)
                p_rand_match = 1.0
            else:
                p_rand_match = -1
                imag = np.where(self.mag_bin_centers == mag_bin)[0][0]
                if PDF_FUNC == "sigmoid":
                    #use the pdf if available
                    if (self.pdf_sigmoid_parms is not None) and (len(self.pdf_sigmoid_parms) >  imag) and \
                            (self.pdf_sigmoid_parms[imag] is not None):

                        p_rand_match = sigmoid(dist,*(self.pdf_sigmoid_parms[imag]))
                        log.info("Using sigmoid PDF for distance prior: mag = %f , dist = %f, 1-p(rand) = %f" %
                                 (mag, dist,1. - p_rand_match))
                elif PDF_FUNC == "poly":
                    if (self.pdf_poly_parms is not None) and (len(self.pdf_poly_parms) >  imag) and \
                            (self.pdf_poly_parms[imag] is not None):

                        p_rand_match = polynomial(dist,self.pdf_poly_parms[imag])
                        log.info("Using polynomial PDF for distance prior: mag = %f , dist = %f, 1-p(rand) = %f" %
                                 (mag, dist,1. - p_rand_match))

                if p_rand_match == -1 :  # use the mdf_matrix
                    idist = np.where(self.annuli_bin_centers == dist_bin)[0][0]
                    p_rand_match = self.mdf_matrix[idist, imag]
                    log.info("Using MDF for distance prior: mag = %f , dist = %f, 1-p(rand) = %f" %
                             (mag, dist, 1. - p_rand_match))


        except:
            log.warning("Cannot sample distance mdf. Will return p = 1.0",exc_info=True)

        return 1. - p_rand_match



class RedshiftPrior:
    #prior based on the redshift (spec_z ... and maybe phot_z) of the bid object and the
    #possible redshift of the emission line (at least as LyA or [OII])
    pass

class MagnitudePrior:
    #prior based on the magnitude of the bid object and the characteristic of the emission line
    pass
