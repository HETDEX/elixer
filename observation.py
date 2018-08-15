#for consuming rsp data

import global_config as G
import numpy as np
import os.path as op

import fiber as elixer_fiber
import spectrum as elixer_spectrum

log = G.Global_Logger('obs_logger')
log.setlevel(G.logging.DEBUG)


def angular_distance(ra1,dec1,ra2,dec2):
    #distances are expected to be relatively small, so will use the median between the decs for curvature
    dist = -1.
    try:
        dec_avg = 0.5*(dec1 + dec2)
        dist = np.sqrt((np.cos(np.deg2rad(dec_avg)) * (ra2 - ra1)) ** 2 + (dec2 - dec1) ** 2)
    except:
        log.warning("Invalid angular distance.",exc_info=True)

    return dist * 3600. #in arcsec

class SyntheticObservation():
    #is one 'observation', and contains all fibers (regardless of IFU or exposure) for that obervation
    #the observation is synthetic in that it may not correspond to an actual, HET observation and is only
    #defined by a set of fibers select by their RA, Dec

    def __init__(self):
        path = None #e.g. to a base rsp output dir
        file = None

        self.ra = None #central (anchor) RA for the "observation"
        self.dec = None #central (anchor) Dec for the "observation"
        self.target_wavelength = None
        self.annulus = None
        self.best_radius = None

        self.fibers_all = [] #one fiber object for each
        self.eli_dict = {} #everytime we call signal_score on a fiber, it should be added here so as to not repeat
                           #keys are fiber objects themselves, values are EmissionLineInfo objects

        self.units = 1 #assumed to be 10**-17 cgs (but being read-in in full notation, e.g. '1')
        self.fibers_work = []  # working set of fibers (subset of fibers_all)

        self.sum_wavelengths = []
        self.sum_values = [] #could be flux or counts, but at this point, should be calibrated flux
        self.sum_errors = []
        self.sum_count = 0 #generally should be == len(self.fibers_work)

        self.w = 0
        self.fwhm = 0
        self.estflux = 0
        self.snr = 0

    def build_complete_emission_line_info_dict(self):
        for f in self.fibers_all:
            if not (f in self.eli_dict):
                #can be None and that is okay
                try:
                    self.eli_dict[f] = elixer_spectrum.signal_score(wavelengths=f.fluxcal_central_emis_wavelengths,
                               values=f.fluxcal_central_emis_flux, errors=f.fluxcal_central_emis_fluxerr,
                               central=self.w, values_units=self.units, sbr=None,
                               show_plot=False,plot_id=None, plot_path=None,do_mcmc=False,
                                                                    force_score=True)

                    #build up the full list of peaks
                    f.is_empty(wavelengths=f.fluxcal_central_emis_wavelengths,
                               values=f.fluxcal_central_emis_flux, errors=f.fluxcal_central_emis_fluxerr,
                               units=self.units, max_score=2.0, max_snr=2.0, max_val=5.0e-17, force=False)
                except:
                    log.error("Error! Could not get signal_score for fiber. %s" %(str(f)), exc_info=True)


    def annulus_fibers(self,inner_radius=None,outer_radius=None,ra=None,dec=None,empty=False):
        '''
        Build subset of fibers that are between the inner and outer radius.
        If outer radius is larger than maximum fiber distance, only populate as much as is possible. No error.
        If inner radius is larger than maximum fiber distance, then get an empty set. No error.

        :param inner_radius:
        :param outer_radius:
        :param ra: optional ... if not specified, use the observations center RA
        :param dec: optional ... if not specified, use the observations center Dec
        :return:
        '''
        self.fibers_work = []

        if (inner_radius is None) or (outer_radius is None):
            if self.annulus is not None:
                inner_radius = self.annulus[0]
                outer_radius = self.annulus[1]
            else:
                log.warning("SyntheticObsercation::annulus_fibers invalid radii (None)")
                return self.fibers_work


        if ra is None and dec is None:
            ra = self.ra
            dec = self.dec

        #having problems with np.where ... so just do this for now
        #self.fibers_work = all[np.where(inner_radius < angular_distance(ra,dec,f.ra,f.dec) < outer_radius)]

        if inner_radius < outer_radius:
            for f in self.fibers_all:
                if inner_radius < angular_distance(ra, dec, f.ra, f.dec) < outer_radius:
                    if (not empty) or (empty and f.is_empty(wavelengths=f.fluxcal_central_emis_wavelengths,
                               values=f.fluxcal_central_emis_flux, errors=f.fluxcal_central_emis_fluxerr,
                                units=self.units, max_score=2.0,max_snr=2.0,force=False)):
                        self.fibers_work.append(f)
        else:
            log.warning("Observation::annulus_fibers Invalid radii (inner = %f, outer = %f)" % (inner_radius, outer_radius))

        for f in self.fibers_work:
            if not (f in self.eli_dict):
                #can be None and that is okay
                try:
                    self.eli_dict[f] = elixer_spectrum.signal_score(wavelengths=f.fluxcal_central_emis_wavelengths,
                               values=f.fluxcal_central_emis_flux, errors=f.fluxcal_central_emis_fluxerr,
                               central=self.w, values_units=self.units, sbr=None,
                               show_plot=False,plot_id=None, plot_path=None,do_mcmc=False,
                                                                    force_score=True)
                except:
                    log.error("Error! Could not get signal_score for fiber. %s" %(str(f)), exc_info=True)

        return self.fibers_work

    def nearest_fiber(self,ra,dec):
        '''
        return nearest fiber to the provided ra and dec
        :param ra:
        :param dec:
        :return: fiber, distance (float)
        '''

        best_fiber = None
        best_dist = 999.9

        for f in self.fibers_all:
            dist = angular_distance(ra, dec, f.ra, f.dec)
            if dist < best_dist:
                best_fiber = f

        return best_fiber,best_dist


    def sum_fibers(self):
        """
        iterate over all fibers (use work fibers if not empty set)

        using interp_ values so the wavelengths should be aligned

        straight, unweighted sum ... intended for diffuse emission ... would expect stronger signal closer to the
            center of annulus (assuming it is centered on an object, like AGN) so might consider a weight at some point

        :return: count of summed fibers
        """

        self.sum_wavelengths = []
        self.sum_values = []
        self.sum_errors = []
        self.sum_count = 0

        #self.sum_values = np.zeros()

        if (self.fibers_work is None) or (len(self.fibers_work) == 0):
            fibers = self.fibers_all
        else:
            fibers = self.fibers_work

        for f in fibers:
            if len(f.interp_spectra_wavelengths)==0:
                f.interpolate()

            if len(f.interp_spectra_wavelengths)==0:
                log.warning("Cannot interpolate (to sum) fiber (%f,%f)" %(f.ra,f.dec))
                continue

            self.sum_count += 1
            if len(self.sum_wavelengths) == 0:
                self.sum_wavelengths = np.array(f.interp_spectra_wavelengths)
                self.sum_values = np.array(f.interp_spectra_flux)
                self.sum_errors = np.array(f.interp_spectra_errors)
            else:
                self.sum_values += np.array(f.interp_spectra_flux)
                self.sum_errors += np.array(f.interp_spectra_errors)

        if self.sum_count == 0:
            try:
                self.sum_wavelengths = self.fibers_all[0].fluxcal_central_emis_wavelengths
            except:
                self.sum_wavelengths = np.arange(3500.0,5500.0,2.0)

            self.sum_values = np.zeros(len(self.sum_wavelengths))
            self.sum_errors = np.zeros(len(self.sum_wavelengths))

        return self.sum_count
    #
    # take a look at the exp files ... did something like this already




#end RSP class