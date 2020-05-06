## Bayesian priors, etc used to help quantify posterior probability that a catalog object is a match for our observation
from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

try:
    from elixer import global_config as G
except:
    import global_config as G

import os.path as op
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)


#todo: read in distance information, for now convert to mdf
#todo: later fit sigmoid and estimate pdf from the mdf

DISTANCE_PRIOR_FILENAME ="distance_list_bool.txt"# "distance_list.txt"
MIRROR = False #should probably stay false ... mirror to 3rd quadrant for sigmoid seems to make fit worse

#PDF_MODEL = "sigmoid"
PDF_MODEL = "poly" #DEFAULT ... better fit than sigmoid in these cases
POLY_DEG = 3 #cubic is somewhat better than quadratic, but the relative improvements fall off rapidly after
             #each higher order does do a little better for the fainter mags, but the improvements are
             #increasingly small


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

#just a simple diagnostic for comparing polynomial fits ... not intended to be used at runtime
def chi2 (obs, model, error=None):

    obs = np.array(obs)
    model = np.array(model)

    if error is not None:
        error = np.array(error)

    #trim non-values
    if 0:
        remove = []
        for i in range(len(obs)):
            if obs[i] > 98.0:
                remove.append(i)

        obs = np.delete(obs, remove)
        model = np.delete(model, remove)
        if error is not None:
            error = np.delete(error, remove)

    if error is not None:
        c = np.sum((obs*model)/(error*error)) / np.sum((model*model)/(error*error))
    else:
        c = 1.0

    chisqr = 0
    if error is None:
        error=np.zeros(len(obs))
        error += 1.0

    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-c*model[i])**2)/(error[i])
    return chisqr,c



#distance prior needs as input the distance and the magnitude of the bid object
class DistancePrior:

    #all class vars
    num_trials = 1000
    #annuli_bin_centers = np.arange(0.0,30.2,0.2) #uncomment if NOT trimming last row
    dist_bin_width = 0.2
    annuli_bin_centers = np.arange(0.0, 30.0, dist_bin_width) #does not include last row (radii bin center = 30.0")

    annuli_bin_edges = annuli_bin_centers - 0.1
    annuli_bin_edges[0] = 0.0
    annuli_bin_edges  = np.append(annuli_bin_edges,30.0)

    mag_bin_centers = np.arange(21.,29.,1.)
    mag_bin_edges = mag_bin_centers - 0.5
    mag_bin_edges[0] = 21.0
    mag_bin_edges = np.append(mag_bin_edges, 28.0)

    min_mag = min(mag_bin_centers)
    max_mag = max(mag_bin_centers)
    max_radius = max(annuli_bin_centers)

    def __init__(self,pdf_model=PDF_MODEL, poly_deg=POLY_DEG, dist_err = 0.5):
        #dist_err = astrometry accuracy error in arcsec

        self.pdf_model = pdf_model
        self.poly_deg = poly_deg
        self.mdf_matrix = None
        self.pdf_sigmoid_parms = None
        self.pdf_poly_parms = None
        # dist_err = astrometry accuracy error in arcsec + annuli_bin_width
        self.dist_err = dist_err + self.dist_bin_width
        self.build_mdfs()
        self.build_pdfs()

    def plot_fits(self):
        if self.pdf_model == "sigmoid":
            self.plot_sigmoid_fits()
        elif self.pdf_model == "poly":
            self.plot_poly_fits()

    def plot_sigmoid_fits(self):
        # just a simple diagnostic for visualizing fits ... not intended to be used at runtime
        import pylab
        #assumes only 8 magnitude bins
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'teal']
        pylab.title("Sigmoid")
        pylab.xlabel("Arcsec")
        pylab.ylabel("P(1+ in 1\" wide annulus)")
        x = self.mdf_matrix[:,0]
        for i in range(len(self.pdf_sigmoid_parms)):
            y = sigmoid(x, *(self.pdf_sigmoid_parms[i]))

            c2, c = chi2(obs=self.mdf_matrix[:, i + 1], model=y, error=None)

            pylab.plot(x, y, label="%d(%f)"%(21 + i,c2), c=color[i])
            pylab.plot(x, self.mdf_matrix[:, i + 1], '.', c=color[i])

        pylab.legend(loc='upper left', title=r"mag ($\chi{^2}$)") #loc="best"
        pylab.show()

    def plot_poly_fits(self):
        # just a simple diagnostic for visualizing fits ... not intended to be used at runtime
        import pylab
        #assumes only 8 magnitude bins
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'teal']
        x = self.mdf_matrix[:,0]
        pylab.title("Polynomial, Degree %d" %(self.poly_deg))
        pylab.xlabel("Arcsec")
        pylab.ylabel("P(1+ in 1\" wide annulus)")
        for i in range(len(self.pdf_poly_parms)):
            y = polynomial(x, self.pdf_poly_parms[i])

            c2,c = chi2(obs=self.mdf_matrix[:, i + 1],model=y,error=None)

            pylab.plot(x, y, label="%d(%0.3f)"%(21 + i,c2), c=color[i])
            pylab.plot(x, self.mdf_matrix[:, i + 1], '.', c=color[i])

        pylab.legend(loc='upper left',title=r"mag ($\chi{^2}$)") #loc="best"
        #pylab.savefig("poly_%d.png" % (self.poly_deg))
        pylab.show()



    def build_mdfs(self):
        file = op.join(op.dirname(G.__file__),DISTANCE_PRIOR_FILENAME)
        try:
            if G.python2():
                out = np.genfromtxt(file, dtype=None,comments ="#")
            else:
                out = np.genfromtxt(file, dtype=None, comments="#",encoding=None)
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


    #though, not really a pdf ... does not sum to 1.0
    def build_pdfs(self):
        if self.mdf_matrix is None:
            self.build_mdfs()

            if self.mdf_matrix is None:
                log.error("Cannot build pdfs for distance priors")


        if self.pdf_model == "sigmoid":
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
                    popt, pcov = curve_fit(sigmoid, np.float64(xdata), np.float64(ydata),
                                           p0=(5.0, 0.2),
                                           bounds=((-1000.0,0.0),(1000.0,10.0))
                                           )
                except:
                    log.error("Failure to fit sigmoid to mag = %g" %(self.mag_bin_centers[mag_idx]), exc_info=True)
                    print("Failure to fit sigmoid to mag = %g" % (self.mag_bin_centers[mag_idx]))
                    self.pdf_sigmoid_parms.append(None)
                    continue

                self.pdf_sigmoid_parms.append(popt)
        elif self.pdf_model == "poly":
            if self.pdf_poly_parms is not None:
                del self.pdf_poly_parms[:]

            self.pdf_poly_parms = []

            xdata = self.annuli_bin_centers

            # brightest to dimmest
            for mag_idx in range(len(self.mag_bin_centers)):
                ydata = self.mdf_matrix[:, mag_idx + 1]  # +1 since first column is just the distance

                try:
                    popt = np.polyfit(xdata,ydata,self.poly_deg)
                except:
                    log.error("Failure to fit polynomial to mag = %g" %(self.mag_bin_centers[mag_idx]), exc_info=True)
                    print("Failure to fit polynomial to mag = %g" % (self.mag_bin_centers[mag_idx]))
                    self.pdf_poly_parms.append(None)
                    continue

                self.pdf_poly_parms.append(popt)

    def get_mdf_prior(self,dist,dist_bin,mag,mag_bin,integrate):

        p_rand_match = 0.0
        min_dist_bin = round(float(max(0, dist - self.dist_err)) / self.dist_bin_width) * self.dist_bin_width
        max_dist_bin = round(float(dist + self.dist_err) / self.dist_bin_width) * self.dist_bin_width
        idist = np.where(self.annuli_bin_centers == dist_bin)[0][0]
        imag = np.where(self.mag_bin_centers == mag_bin)[0][0]

        if integrate:
            min_idist = np.where(self.annuli_bin_centers == min_dist_bin)[0][0]
            max_idist = np.where(self.annuli_bin_centers == max_dist_bin)[0][0]

            for i in range(min_idist, max_idist + 1):
                p_rand_match += self.mdf_matrix[i, imag]
        else:
            p_rand_match = self.mdf_matrix[idist, imag]

        log.info("Using MDF for distance prior: mag = %f , dist = %f, Match score = %f" %
                 (mag, dist, 1. - p_rand_match))

        return p_rand_match

    def get_prior(self,dist,mag,integrate=True,mdf=False): #distance in arcsec
        #find in mdf the likelihood of a RANDOM match
        #return 1-likelihood(random)
        p_rand_match = 0.0
        try:
            #round dist to nearest 0.2" (get an index)
            #round mag to nearest 1.0 mag (get an index)

            # technically this would put the right edge in with the lower bin
            # but for an early, quick and dirty experiment, is okay
            dist_bin = round(float(dist) / self.dist_bin_width) * self.dist_bin_width
            mag_bin = round(float(mag))

            #find the corresponding indicies
            #should always have an entry (unless > 30" or mag is out of range)
            # in either case, will just return 1.0


            if (dist_bin > self.max_radius):  # way too far (outside of the distribution, so call it random)
                p_rand_match = 1.0
            else:

                if (mag_bin < self.min_mag) and (dist_bin > 5.0): #too bright, but not super close
                    #bright usually means big, larger dist_bin
                    mag_bin = self.min_mag #just set to the minimum mag and use that
                elif (mag_bin > self.max_mag) and (dist_bin < 3.0): #faint, but fairly close
                    mag_bin = self.max_mag

                if (mag_bin < self.min_mag): #super-bright and close ... very likely to be the real match
                    p_rand_match = 0.0
                elif (mag_bin > self.max_mag): #too faint. No data but assume very many so max out random likelihood
                    p_rand_match = 1.0
                elif mdf:
                    p_rand_match = self.get_mdf_prior(dist,dist_bin,mag,mag_bin,integrate)
                else:
                    p_rand_match = -1
                    imag = np.where(self.mag_bin_centers == mag_bin)[0][0]
                    if self.pdf_model == "sigmoid":
                        #use the pdf if available
                        if (self.pdf_sigmoid_parms is not None) and (len(self.pdf_sigmoid_parms) >  imag) and \
                                (self.pdf_sigmoid_parms[imag] is not None):

                            if integrate:
                                min_dist = max(0, dist - self.dist_err)
                                max_dist = dist + self.dist_err
                                p_rand_match = quad(sigmoid, min_dist, max_dist, *(self.pdf_sigmoid_parms[imag]))[0]
                            else:
                                # simple function call (lookup)
                                p_rand_match = sigmoid(dist, *(self.pdf_sigmoid_parms[imag]))

                            log.info("Using sigmoid PDF for distance prior: mag = %f , dist = %f, Match score = %f" %
                                     (mag, dist,1. - p_rand_match))
                    elif self.pdf_model == "poly":
                        if (self.pdf_poly_parms is not None) and (len(self.pdf_poly_parms) >  imag) and \
                                (self.pdf_poly_parms[imag] is not None):

                            if integrate:
                                #integrate for area under the curve and deal with astrometry error
                                min_dist = max(0, dist - self.dist_err)
                                max_dist = dist + self.dist_err
                                p_rand_match = quad(polynomial, min_dist, max_dist, self.pdf_poly_parms[imag])[0]
                            else:
                                #simple function call (lookup)
                                p_rand_match = polynomial(dist,self.pdf_poly_parms[imag])

                            log.info("Using polynomial PDF for distance prior: mag = %f , dist = %f, Match score = %f" %
                                     (mag, dist,1. - p_rand_match))

                    if p_rand_match == -1 :  # use the mdf_matrix
                        idist = np.where(self.annuli_bin_centers == dist_bin)[0][0]

                        if integrate:
                            min_idist = np.where(self.annuli_bin_centers == min_dist_bin)[0][0]
                            max_idist = np.where(self.annuli_bin_centers == max_dist_bin)[0][0]

                            for i in range(min_idist,max_idist+1):
                                p_rand_match += self.mdf_matrix[i, imag]
                        else:
                            p_rand_match = self.mdf_matrix[idist, imag]

                        log.info("Using MDF for distance prior: mag = %f , dist = %f, Match score = %f" %
                                 (mag, dist, 1. - p_rand_match))
        except:
            log.warning("Cannot sample distance mdf. Will return p = 1.0",exc_info=True)
            p_rand_match = 1.0


        p_rand_match = min(1.0,abs(p_rand_match))

        return 1. - p_rand_match



class RedshiftPrior:
    #prior based on the redshift (spec_z ... and maybe phot_z) of the bid object and the
    #possible redshift of the emission line (at least as LyA or [OII])
    pass

class MagnitudePrior:
    #prior based on the magnitude of the bid object and the characteristic of the emission line
    pass
