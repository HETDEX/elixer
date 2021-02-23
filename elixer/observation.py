#for consuming rsp data

try:
    from elixer import global_config as G
    from elixer import fiber as elixer_fiber
    from elixer import spectrum as elixer_spectrum
    from elixer import utilities as utils
    from elixer import hetdex_fits
except:
    import global_config as G
    import fiber as elixer_fiber
    import spectrum as elixer_spectrum
    import utilities as utils
    import hetdex_fits

import numpy as np
import os.path as op
from astropy.coordinates import SkyCoord
from astropy import units as U
from astropy.stats import sigma_clipped_stats
try:
    from hetdex_api.shot import get_fibers_table as hda_get_fibers_table
    from hetdex_tools.get_spec import get_spectra as hda_get_spectra
except Exception as e:
    print("WARNING!!!! CANNOT IMPORT hetdex_api tools: ",e)
#import copy

log = G.Global_Logger('obs_logger')
log.setlevel(G.LOG_LEVEL)


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
        self.extraction_ffsky = None

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

        self.calfib_noise_estimate = np.zeros(len(G.CALFIB_WAVEGRID))
        self.survey_shotid = None


    def get_aperture_fibers(self):
        """
        Get all the fibers that contribute to the *aperture* (so fiber centers at a distance of 0.0 out to the outer radius + 0.75")
        :param basic_only:
        :return:
        """
        try:
            coord = SkyCoord(ra=self.ra * U.deg, dec=self.dec * U.deg)

            if self.survey_shotid is None:
                #get shot for RA, Dec
                log.error("Required survey_shotid is None in SyntheticObservation::get_aperture_fibers")
                self.status = -1
                return

            ftb = hda_get_fibers_table(self.survey_shotid,coord,
                                       radius=(self.annulus[1] + G.Fiber_Radius) * U.arcsec, #use the outer radius
                                       survey=f"hdr{G.HDR_Version}")

            #build list of fibers and sort by distance (as proxy for weight)
            count = 0
            duplicate_count = 0
            num_fibers = len(ftb)
            for row in ftb:
                count += 1
                specid = row['specid']
                ifuslot = row['ifuslot']
                ifuid = row['ifuid']
                amp = row['amp']
                #date = str(row['date'])

                # expected to be "20180320T052104.2"
                #time_ex = row['timestamp'][9:]
                #time = time_ex[0:6]  # hhmmss

                #yyyymmddxxx
                date = str(self.survey_shotid//1000) #chop off the shotid part and keep the date part
                time = "000000"
                time_ex = "000000.0"


                mfits_name = row['multiframe']  # similar to multi*fits style name
                #fiber_index is a string here: '20190104025_3_multi_315_021_073_RL_030'
                fiber_index = row['fibidx'] #int(row['fiber_id'][-3:])  (fiber_id is a number, fibidx is index)
                obsid = row['fiber_id'][8:11]

                idstring = date + "v" + time_ex + "_" + specid + "_" + ifuslot + "_" + ifuid + "_" + amp + "_" #leave off the fiber for the moment

                log.debug("Building fiber %d of %d (%s e%d) ..." % (count, num_fibers,idstring + str(fiber_index+1),int(row['expnum'])))
                idstring += str(fiber_index) #add the fiber index (zero based)

                fiber = elixer_fiber.Fiber(idstring=idstring,specid=specid,ifuslot=ifuslot,ifuid=ifuid,amp=amp,
                                           date=date,time=time,time_ex=time_ex, panacea_fiber_index=fiber_index,
                                           detect_id=0)

                if fiber is not None:
                    duplicate = False
                    fiber.ra = row['ra']
                    fiber.dec = row['dec']
                    fiber.obsid = obsid #int(row['obsind']) #notice: obsind vs obsid
                    fiber.expid = int(row['expnum'])  # integer now
                    fiber.detect_id = 0
                    fiber.center_x = row['ifux']
                    fiber.center_y = row['ifuy']

                    #don't have weights used, so use distance to the provided RA, Dec as a sorting substitute
                    #fiber.raw_weight = row['weight']
                    fiber.distance = utils.angular_distance(fiber.ra,fiber.dec,self.ra,self.dec)

                    # check that this is NOT a duplicate
                    for i in range(len(self.fibers_all)):
                        if fiber == self.fibers_all[i]:
                            log.warning(
                                "Warning! Duplicate Fiber : %s . idx %d == %d. Duplicate will not be processed." %
                                (fiber.idstring, i, count - 1))
                            duplicate = True
                            duplicate_count += 1
                            break

                    if duplicate:
                        continue  # continue on to next fiber

                    #don't need this HERE
                    #we have all the data we need in the ftb (fiber table) from above
                    #todo: need to load up the fiber data (wavelengths, flux, etc)
                    # ['wavelength']  ['calfib'] OR ['spec_fullsky_sub'] (need to know the --ffsky parameter) and ['calfibe']

                    fiber.d_aperture_local_calfib = row['calfib']
                    fiber.d_aperture_calfibe =  row['calfibe']
                    fiber.d_aperture_ffsky_calfib =  row['spec_fullsky_sub']




                    if False:
                        #todo: later? extract exactly on each fiber center individually
                        #  this really is not right. ... this would be as if there were a single point source at the center
                        #  of each fiber (and nothing else around it) would be the line flux of that point source (spread out
                        #  over the PSF)
                        fcoord = SkyCoord(ra=fiber.ra * U.deg, dec=fiber.dec * U.deg)
                        #technically, this is grabbing the specific fiber and its 6 adjacent neighbors (but with PSF convulution
                        #with an average PSF around 2", this is necessary and correct to get the light in this fiber)
                        #BUT this returns MUCH bigger values (sum) than the other two
                        apt = hda_get_spectra(fcoord, survey=f"hdr{G.HDR_Version}", shotid=self.survey_shotid,
                                              ffsky=self.extraction_ffsky, multiprocess=G.GET_SPECTRA_MULTIPROCESS, rad=2.25*G.Fiber_Radius,
                                              fiberweights=True)
                                                #2.25 so fiber centers can be off a little and still get the necessary 7 fibers

                        if len(apt) == 1: #should be a single fiber
                            # print(f"No spectra for ra ({self.ra}) dec ({self.dec})")
                            fiber.d_fiber_sumspec_flux = np.nan_to_num(apt['spec'][0],
                                                              nan=0.000) * G.FLUX_WAVEBIN_WIDTH  # in 1e-17 units (like HDF5 read)
                            fiber.d_fiber_sumspec_fluxerr = np.nan_to_num(apt['spec_err'][0], nan=0.000) * G.FLUX_WAVEBIN_WIDTH

                           # self.sumspec_wavelength = np.array(apt['wavelength'][0])
                        else:
                            log.info(f"Unexpected number of spectra ({len(apt)}) returned for fiber at ra ({fiber.ra}) dec ({fiber.dec})")



                    if False:
                        # fiber.relative_weight = row['weight']
                        # add the fiber (still needs to load its fits file)
                        # we already know the path to it ... so do that here??

                        # full path to the HDF5 fits equivalent (or failing that the panacea fits file?)
                        fiber.fits_fn = fiber.find_hdf5_multifits()

                        # now, get the corresponding FITS or FITS equivalent (HDF5)
                        fits = hetdex_fits.HetdexFits(empty=True)
                        # populate the data we need to read the HDF5 file
                        fits.filename = fiber.fits_fn  # mfits_name #todo: fix to the corect path
                        fits.multiframe = mfits_name
                        fits.panacea = True
                        fits.hdf5 = True

                        fits.obsid = str(fiber.obsid).zfill(3)
                        fits.expid = int(fiber.expid)
                        fits.specid = str(fiber.specid).zfill(3)
                        fits.ifuslot = str(fiber.ifuslot).zfill(3)
                        fits.ifuid = str(fiber.ifuid).zfill(3)
                        fits.amp = fiber.amp
                        fits.side = fiber.amp[0]

                        fits.obs_date = fiber.dither_date
                        fits.obs_ymd = fits.obs_date

                        # now read the HDF5 equivalent
                        fits.read_hdf5()
                        # check if it is okay

                        if fits.okay:
                            fiber.fits = fits
                        else:
                            log.error("HDF5 multi-fits equivalent is not okay ...")

                    self.fibers_all.append(fiber)

            #build a noise estimate over the top 4 fibers (amps)?
            # try:
            #     good_idx = np.where([x.fits for x in self.fibers_all])[0]  # some might be None, so get those that are not
            #
            #     all_calfib = np.concatenate([self.fibers_all[i].fits.calfib for i in good_idx], axis=0)
            #
            #     # use the std dev of all "mostly empty" (hence sigma=3.0) or "sky" fibers as the error
            #     mean, median, std = sigma_clipped_stats(all_calfib, axis=0, sigma=3.0)
            #     self.calfib_noise_estimate = std
            #
            # except:
            #     log.info("Could not build SyntheticObservation calfib_noise_estimate", exc_info=True)
        except:
            log.info("Could not get fibers in SyntheticObservation::get_aperture_fibers()", exc_info=True)

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
                               units=self.units, max_score=10.0, max_snr=5.0, max_val=9.0e-17, force=False,
                               central_wavelength=self.target_wavelength)
                             #max_score = 2.0, max_snr = 2.0, max_val = 5.0e-17
                except:
                    log.error("Error! Could not get signal_score for fiber. %s" %(str(f)), exc_info=True)


    def annulus_fibers(self,inner_radius=None,outer_radius=None,ra=None,dec=None,empty=False,central_wavelength=None):
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

        central_wavelength = None #force to not use right now
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
            for f in self.fibers_all: #any portion of the fiber between (not just the center)
                if (inner_radius - G.Fiber_Radius) < angular_distance(ra, dec, f.ra, f.dec) < (outer_radius + G.Fiber_Radius):
                    self.fibers_work.append(f)
                    # if (not empty) or (empty and f.is_empty(wavelengths=f.fluxcal_central_emis_wavelengths,
                    #            values=f.fluxcal_central_emis_flux, errors=f.fluxcal_central_emis_fluxerr,
                    #             units=self.units, max_score=10.0,max_snr=5.0,force=False,
                    #                                         central_wavelength=central_wavelength)):
                    #     #previous max_score = 2.0 , max_snr = 2.0 in effort to keep out signal
                    #     self.fibers_work.append(f)
        else:
            log.warning("Observation::annulus_fibers Invalid radii (inner = %f, outer = %f)" % (inner_radius, outer_radius))

        #todo: how best to figure the contribution from these fibers....
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