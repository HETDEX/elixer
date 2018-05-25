## Bayesian priors, etc used to help quantify posterior probability that a catalog object is a match for our observation
from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import global_config as G
import os.path as op
import numpy as np

log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)


#todo: read in distance information, for now convert to mdf
#todo: later fit sigmoid and estimate pdf from the mdf

DISTANCE_PRIOR_FILENAME = "distance_list.txt"

#distance prior needs as input the distance and the magnitude of the bid object
class DistancePrior:

    #all class vars
    num_trials = 1000
    annuli_bin_centers = np.arange(0.0,30.2,0.2)

    annuli_bin_edges = annuli_bin_centers - 0.1
    annuli_bin_edges[0] = 0.0
    annuli_bin_edges  = np.append(annuli_bin_edges,30.0)

    mag_bin_centers = np.arange(21.,29.,1.)
    mag_bin_edges = mag_bin_centers - 0.5
    mag_bin_edges[0] = 21.0
    mag_bin_edges = np.append(mag_bin_edges, 28.0)

    def __init__(self):
        self.mdf_matrix = None
        self.build_mdfs()


    def build_mdfs(self):
        file = op.join(op.dirname(G.__file__),DISTANCE_PRIOR_FILENAME)
        try:
            out = np.genfromtxt(file, dtype=None,comments ="#")
        except:
            log.error("Cannot read distance prior file: %s" % file, exc_info=True)

        try:
            #reduce 2nd through nth column values by divid by num_trials, cap at 1.0
            out[:,1:] /= self.num_trials
            out[:, 1:] = np.clip(out[:,1:],0.0,1.0)
            #note: these are not all strictly increasing and the last bin is only 1/2 sized

            #find the first = 1.0 and set all the rest
            for m in range(1,len(self.mag_bin_centers)+1):
                i = np.where(out[:,m]==1.0)[0] #array of all
                if len(i) > 0:
                    out[i[0]:,m] = 1.0

            self.mdf_matrix = out
        except:
            log.error("Cannot build mdf from distance prior file: %s" % file, exc_info=True)


    def get_prior(self,dist,mag): #distance in arcsec
        #find in mdf the likelihood of a RANDOM match
        #return 1-likelihood(random)
        p_rand_match = 0.0

        #round dist to nearest 0.2" (get an index)
        #round mag to nearest 1.0 mag (get an index)

        try:
            # technically this would put the right edge in with the lower bin
            # but for an early, quick and dirty experiment, is okay
            dist = round(float(dist) / 0.2) * 0.2
            mag = round(float(mag))

            #find the corresponding indicies
            #should always have an entry (unless > 30" or mag is out of range)
            # in either case, will just return 1.0
            if (dist > 30.0) or (mag < 21.0):
                p_rand_match = 1.0
            else:
                idist = np.where(self.annuli_bin_centers == dist)[0][0]
                imag =  np.where(self.mag_bin_centers == mag)[0][0]
                p_rand_match = self.mdf_matrix[idist,imag]

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
