
try:
    from elixer import global_config as G
    from elixer import spectrum as elixer_spectrum
except:
    import global_config as G
    import spectrum as elixer_spectrum


import numpy as np
import os.path as op

#log = G.logging.getLogger('fiber_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('fiber_logger')
log.setlevel(G.LOG_LEVEL)

SIDE = ["L", "R"]

#!!! REMEBER, Y-axis runs 'down':  Python 0,0 is top-left, DS9 is bottom-left
#!!! so in DS9 LU is 'above' LL and RL is 'above' RU
AMP  = ["LU","LL","RL","RU"] #in order from bottom to top
AMP_OFFSET = {"LU":1,"LL":113,"RL":225,"RU":337}

MIN_WAVELENGTH = 3500.0
MAX_WAVELENGTH = 5500.0
INTERPOLATION_AA_PER_PIX = 2.0

def fit_line(wavelengths,values,errors=None):
#super simple line fit ... very basic
#rescale x so that we start at x = 0
    coeff = np.polyfit(wavelengths,values,deg=1)
    #flip the array so [0] = 0th, [1] = 1st ...
    coeff = np.flip(coeff,0)
    return coeff

def parse_fiber_idstring(idstring):
    if idstring is None:
        return None
    # 20170326T105655.6_032_094_028_LU_032

    toks = idstring.split("_")

    if len(toks) != 6:
        if (len(toks) == 1) and (toks[0] == "666"):
            return True  # this is an "ignore" flag, but still continue as if it were a fiber
        else:
            pass  # stop bothering with this ... it is always there
            # log.warning("Unexpected fiber id string: %s" % fiber)
        return False

    #idstring = fiber_idstring  # toks[0] #ie. 20170326T105655.6

    date = idstring[0:8]
    # next should be 'T'
    time = idstring[9:15]  # not the .# not always there
    if idstring[15] == ".":
        time_ex = idstring[9:17]
    else:
        time_ex = None

    specid = toks[1]
    ifuslot = toks[2]
    ifuid = toks[3]
    amp = toks[4]
    # fiber_idx = toks[5] #note: this is the INDEX from panacea, not the relative fiberm but karl adds 1 to it
    # (ie. fiber #1 = index 111, fiber #2 = index 110 ... fiber #112 = index 0)
    panacea_fiber_idx = int(toks[5]) - 1

    fiber_dict = dict(zip(["idstring", "specid", "ifuslot", "ifuid", "amp", "date", "time",
                           "time_ex", "panacea_fiber_idx"],
                          [idstring, specid, ifuslot, ifuid, amp, date, time,
                           time_ex, panacea_fiber_idx]))

    return fiber_dict

class Fiber:
    #todo: if needed allow fiber number (in amp or side or ccd) to be passed in instead of panacea index
    def __init__(self,idstring=None,specid=None,ifuslot=None,ifuid=None,amp=None,date=None,time=None,time_ex=None,
                 panacea_fiber_index=-1, detect_id = -1):

        if idstring is None:
            #must specify something
            if specid is None:
                return None
            idstring = ""
        elif specid is None:
            try:
                dict_args = parse_fiber_idstring(idstring)
                if dict_args is not None:
                    idstring = dict_args["idstring"]
                    specid = dict_args["specid"]
                    ifuslot = dict_args["ifuslot"]
                    ifuid = dict_args["ifuid"]
                    if amp is None:
                        amp = dict_args["amp"]
                    date = dict_args["date"]
                    time = dict_args["time"]
                    time_ex = dict_args["time_ex"]
                    if panacea_fiber_index == -1:
                        panacea_fiber_index = dict_args["panacea_fiber_idx"]
            except:
                log.error("Exception: Cannot parse fiber string.",exc_info=True)

        self.bad = False
        self.detect_id = detect_id
        self.idstring = idstring #whole string
        #scifits_idstring ... just the first part that IDs the file
        self.scifits_idstring = idstring.split("_")[0] #todo: if cure, strip off leading non-numeric characters
        self.specid = specid
        self.ifuslot = ifuslot
        self.ifuid = ifuid
        self.amp = amp
        if self.amp and (len(self.amp) == 2):
            self.side = amp[0]
        else:
            self.side = ""
            self.amp = ""
        self.dither_date = date #or obsid
        self.dither_time = time #or observation time
        self.dither_time_extended = time_ex
        self.obsid = None
        self.expid = None
        self.sn = None
        self.fits_fn = None #full path to the fits file
        self.fits = None #HetdexFits object that includes this fiber

        self.dither_idx = None
        self.center_x = None
        self.center_y = None

        self.emis_x = -1 #x,y coords on the amp of the emission line peak
        self.emis_y = -1

        self.panacea_idx = -1 #0 to 111
        self.number_in_amp = -1 #1 to 112
        self.number_in_side = -1 #1 to 224
        self.number_in_ccd = -1 #1 to 448

        self.ra = None
        self.dec = None

        #as of 1.4.0a11+ no longer using dqs
        # self.dqs = None  # detection quality score for this fiber (as part of the DetObj owner)
        # self.dqs_raw = None  # unweighted score
        # self.dqs_dist = None  # distance from source
        # self.dqs_w = None #dqs weight
        # self.dqs_bad = False

        #these are from panacea or cure directly , the flux-calibrated data is farther down
        self.central_wave_pixels_bad = 0
        self.central_emis_counts = [] #from fiber-extracted ... the few pixels around the peak
        self.central_emis_wavelengths = []
        self.central_emis_errors = []


        #full length   (NOT CALIBRATED)
        self.data_spectra_wavelengths=[]
        self.data_spectra_counts=[]
        self.data_spectra_errors = []
        self.max_spectra_count = None

        #interpolated onto a 1 angstrom grid
        self.interp_spectra_wavelengths = []
        self.interp_spectra_flux = []
        self.interp_spectra_counts = []
        self.interp_spectra_errors = []

        self.multi = None
        self.raw_weight = 1.0 #raw weight from Karl
        self.relative_weight = 1.0 #spatially calculated, so one for entire fiber
        self.pixel_flat_center_ratio = 1.0 #ratio of the position of emission line to the rest of the pixel flat
        # to sum fibers need relative_weight*thuput at each wavelength

        #full length CALIBRATED (labeled 'central' but this is the whole length)
        self.fluxcal_central_emis_wavelengths = []
        self.fluxcal_central_emis_counts = []
        self.fluxcal_central_emis_flux = []
        self.fluxcal_central_emis_fluxerr = []
        self.fluxcal_central_emis_thru = []
        self.fluxcal_emis_cont = []

        self.empty_status = None #meaning, no strong signals (anywhere in spectrum)
        self.peaks = None #if populated is an array of peaks from spectrum::peakdet (may be empty if no strong signal)

        #info about weak (diffuse) signal:  d for diffuse
        self.d_wavelength = None #line center attempting to find (e.g. th observed, LyA line of central object)
        self.d_units = -17 #expected to be 10^-17 cgs
        self.d_flux = None #approximate flux (area of gaussian)
        self.d_flux_err = None #error on the flux estimate
        self.d_snr = None #approximate signal to noise
        self.d_prob_noise = None #probability that this "signal" is just noise (generally will be high, given, by
                                 #definition, the low flux (and correlated low SNR) we are seeking

        try:
            self.panacea_idx = int(panacea_fiber_index)
            self.number_in_amp = 112 - self.panacea_idx
            self.number_in_ccd = AMP_OFFSET[self.amp] + self.number_in_amp - 1
            if self.number_in_ccd > 224:
                self.number_in_side = self.number_in_ccd - 224
            else:
                self.number_in_side = self.number_in_ccd

            #panacea_idx +1 since Karl starts at 1 and panacea at 0
            self.multi = "multi_%s_%s_%s_%s_%s" % \
                        ( str(self.specid).zfill(3), str(self.ifuslot).zfill(3),str(self.ifuid).zfill(3),
                          self.amp, str(self.panacea_idx+1).zfill(3))
        except:
            log.error("Unable to map fiber index (%d) to fiber number(s)" % int(panacea_fiber_index), exc_info=True)

    def clear(self,bad=False):
        self.central_emis_counts = []  # from fiber-extracted ... the few pixels around the peak
        self.central_emis_wavelengths = []
        self.central_emis_errors = []

        # full length   (NOT CALIBRATED)
        self.data_spectra_wavelengths = []
        self.data_spectra_counts = []
        self.data_spectra_errors = []
        self.max_spectra_count = None

        # interpolated onto a 1 angstrom grid
        self.interp_spectra_wavelengths = []
        self.interp_spectra_flux = []
        self.interp_spectra_counts = []
        self.interp_spectra_errors = []

        self.multi = None
        self.relative_weight = 1.0  # spatially calculated, so one for entire fiber
        # to sum fibers need relative_weight*thuput at each wavelength

        # full length CALIBRATED (labeled 'central' but this is the whole length)
        self.fluxcal_central_emis_wavelengths = []
        self.fluxcal_central_emis_counts = []
        self.fluxcal_central_emis_flux = []
        self.fluxcal_central_emis_fluxerr = []
        self.fluxcal_central_emis_thru = []
        self.fluxcal_emis_cont = []

        self.empty_status = None  # meaning, no strong signals (anywhere in spectrum)
        self.peaks = None  # if populated is an array of peaks from spectrum::peakdet (may be empty if no strong signal)

        # info about weak (diffuse) signal:  d for diffuse
        self.d_wavelength = None  # line center attempting to find (e.g. th observed, LyA line of central object)
        self.d_units = -17  # expected to be 10^-17 cgs
        self.d_flux = None  # approximate flux (area of gaussian)
        self.d_flux_err = None  # error on the flux estimate
        self.d_snr = None  # approximate signal to noise
        self.d_prob_noise = None  # probability that this "signal" is just noise (generally will be high, given, by
        # definition, the low flux (and correlated low SNR) we are seeking

        self.bad = bad #mark as bad fiber

    def __str__(self):
        date = ""
        time = ""
        ra = 0
        dec = 0
        if self.ra is not None:
            ra = self.ra

        if self.dec is not None:
            dec = self.dec

        if self.dither_date is not None:
            date = self.dither_date

        if self.dither_time is not None:
            time = self.dither_time

        return "(ra=%f, dec=%f, datetime=%sT%s)" %(ra, dec, date, time)

    @property
    def fluxcal_wavelengths(self):
        return self.fluxcal_central_emis_wavelengths

    @property
    def fluxcal_flux(self):
        return self.fluxcal_central_emis_flux

    @property
    def fluxcal_values(self):
        return self.fluxcal_central_emis_flux

    @property
    def fluxcal_err(self):
        return self.fluxcal_central_emis_fluxerr

    @property
    def sky_x(self):
        return self.center_x

    @property
    def sky_y(self):
        return self.center_y

    @property
    def max_count(self):
        #todo: better way to find max?
        if (self.max_spectra_count is None) and (len(self.data_spectra_counts) > 1):
            self.max_spectra_count = np.max(self.data_spectra_counts)
        return self.max_spectra_count


    @property
    def ds9_x(self):
        """return the translated emis_x coordinate in terms of ds9 indexing"""
        #ds9 starts with 1, python with 0
        if (self.emis_y is not None) and (self.emis_y != -1):
            return self.emis_x + 1
        else:
            return -1

    @property
    def ds9_y(self):
        """return the translated emis_y coordinate in terms of ds9 indexing"""
        #ds9 starts bottom, left as 1,1
        #python starts top, left as 0,0
        #assume 1032 amp height
        #panacea already has the correct (python) indexing, so the indexing is correct except for the 1 base vs 0 base
        if (self.emis_y is not None) and (self.emis_y != -1):
            #return AMP_HEIGHT_Y - self.emis_y
            return self.emis_y + 1
        else:
            return -1

    #for dictionary comparisions to use as keys
    def __eq__(self,other):
        return (self.ra, self.dec, self.dither_date, self.dither_time) == \
               (other.ra, other.dec, other.dither_date,other.dither_time)

    # for dictionary comparisions to use as keys
    def __ne__(self,other):
        return not (self == other)

    # for dictionary comparisions to use as keys
    def __hash__(self):
        return hash((self.ra, self.dec, self.dither_date, self.dither_time))


    def find_hdf5_multifits(self,loc=None):
        """
        Find the parent HDF5 multi-fits equivalent file that houses THIS fiber
        ... depends on date, time, shot, etc
        :return:
        """
         #build up path and see if it exists
        #print("******* !!!!!! take this out ... dev only !!!!!!! *******")
        #return "/home/dustin/code/python/hdf5_learn/cache/test_new.h5"
        #return "/home/dustin/code/python/hdf5_learn/cache/20180123v009.h5"

        try:
            fn = None
            #todo: not sure what the naming schema is going to be, but will assume stops at a SHOT
            name = self.dither_date + "v" + str(self.obsid).zfill(3) + ".h5"

            if loc is None:
                path = G.PANACEA_HDF5_BASEDIR #start here
                fn = op.join(path, name)
                #/work/03261/polonius/hdr1/reduction/data  ... all at top level?
                #name == DateVShot ... i.e.: 20180123v009.h5

                #todo: add to the path ... not sure yet if all the HDF5 files will be at one level or if they
                #todo: will follow the multi*fits directory organization
                #<BASEDIR>/20180123/virus/virus0000009/exp01/virus/<multi_xxx.fits>
            else:
                path = loc
                fn = op.join(path, name)
                if not op.isfile(fn):
                    path = G.PANACEA_HDF5_BASEDIR  # start here
                    fn = op.join(path, name)

            if not op.isfile(fn):
                log.info("Could not locate HDF5 multi-fits equivalent file: %s" %fn)
                fn = None
        except:
            log.error("Exception attempting to locate HDF5 multi-fits equivalent file", exc_info=True)
            fn = None

        return fn



    def is_empty(self, wavelengths, values, errors=None, units=None, max_score=2.0, max_snr=2.0,
                 max_val=5.0e-17, force=False, central_wavelength=None):
        '''
        Basically, is the fiber free from any overt signals (real emission line(s), continuum, sky, etc)
        Values and errors must have the same units
        Values and errors must already be normalized (e.g. values/fiber_to_fiber, errors*fiber_to_fiber, etc)
        Would be best (downstream) if they were also flux calibtrated, but does not matter much in this function

        '''

        #return True #always add, just for now just to count the fibers

        central_wavelength = None

        if not force and (self.empty_status is not None):
            return self.empty_status

        self.empty_status = False
        if (wavelengths is None) or (values is None) or (len(wavelengths)==0) or (len(values)==0):
            log.warning("Zero length (or None) spectrum passed to fiber::is_empty(). Treating as NOT empty.")
            return self.empty_status

        narrow_values = values
        narrow_wavelengths = wavelengths
        narrow_errors = errors
        if central_wavelength is not None:
            min_idx = G.getnearpos(wavelengths,central_wavelength - 20.0)
            max_idx = G.getnearpos(wavelengths,central_wavelength + 20.0)
            narrow_values = values[min_idx:max_idx]
            narrow_wavelengths = wavelengths[min_idx:max_idx]
            if errors is not None:
                narrow_errors = errors[min_idx:max_idx]
            log.debug("Fiber::is_empty subselect wavelengths [%g:%g]" %(wavelengths[min_idx], wavelengths[max_idx]))
            #subselect around the central wavelength +/- 50A
        else:
            log.debug("Fiber::is_empty central wavelgnth not provided.")


        #first check the slope .... if there is an overall slope there must be continuum and that == signal
        try:
            #simplest check first
            #take out any large values (could be just a stuck pixel, but a gaussian would not fit and thus no peaks)
            mx = max(narrow_values)
            mn = min(narrow_values)

            if mx > max_val:
                log.info("Maximum value exceeded (%0.3g < %0.3g). Fiber not empty." %(max_val,mx))
                self.empty_status = False
                return self.empty_status

            if mn < (-1.*max_val):
                log.info("Minium value exceeded (%0.3g > %0.3g). Fiber not empty." %(-1.*max_val,mn))
                self.empty_status = False
                return self.empty_status



            #next check the slope

            coeff = fit_line(wavelengths, values, errors)  # flipped order ... coeff[0] = 0th, coeff[1]=1st
            self.spectrum_linear_coeff = coeff

            #todo: not sure I can trust this ... can have a slope and still no signal due to
            #todo:   wavelength bin dependent uncertainty
            #todo: for now, just report it in the log
            # if abs(coeff[1]) > 0.001:  # todo .. what is a good value here?
            #     log.info("Maximum slope exceeded (|%0.3g| > 0.001). Fiber not empty." % (coeff[1]))
            #     self.empty_status = False
            #     return self.empty_status


            #now check for any "good" peaks (most costly check)
            self.peaks = elixer_spectrum.peakdet(narrow_wavelengths, narrow_values, narrow_errors,values_units=units)  # , h, dh, zero)

            num_peaks = -1
            if (self.peaks is not None):
                num_peaks = len(self.peaks)

            log.info("(%f,%f) spectrum basic info. Peaks (%d), continuum (mx+b): %f(x) + %f, min (%f), max(%f)"
                     %(self.ra, self.dec, num_peaks, coeff[1], coeff[0], mn, mx))

            #could have 1+ peaks, if all line_score below max_score and all snr below max_snr
            self.empty_status = True
            if num_peaks > 0:
                for p in self.peaks:
                    if (p.line_score is not None and p.snr is not None) and \
                        (p.line_score > max_score or p.snr > max_snr):
                        self.empty_status = False
                        break

            if self.empty_status == False:
                log.debug("Fiber may not be empty.")
        except:
            log.debug("Exception in fiber::is_empty() ", exc_info=True)
            self.empty_status = False

        return self.empty_status

    def interpolate(self,grid_size=INTERPOLATION_AA_PER_PIX):

        self.interp_spectra_wavelengths = np.arange(MIN_WAVELENGTH, MAX_WAVELENGTH + grid_size, grid_size)
        self.interp_spectra_flux = np.interp(self.interp_spectra_wavelengths, self.fluxcal_wavelengths,
                                              self.fluxcal_flux)

        #should not need this one...
        self.interp_spectra_counts = np.interp(self.interp_spectra_wavelengths, self.fluxcal_wavelengths,
                                             self.fluxcal_central_emis_counts)

        #not sure this is the best approach, but the errors are pretty consistent (slowly varying),
        #so hopefully this is okay
        self.interp_spectra_errors = np.interp(self.interp_spectra_wavelengths,
                                               self.fluxcal_wavelengths,
                                               self.fluxcal_err)





    # def dqs_weight(self,ra,dec):
    #     weight = 0.0
    #     #specifically None ... a 0.0 RA or Dec is possible
    #     if (ra is None) or (dec is None) or (self.ra is None) or (self.dec is None):
    #         self.dqs_dist = 999.9
    #         return weight
    #
    #     dist = np.sqrt( (np.cos(np.deg2rad(dec))*(ra-self.ra)) ** 2 + (dec - self.dec) ** 2) * 3600.
    #
    #     if dist > G.FULL_WEIGHT_DISTANCE:
    #         if dist > G.ZERO_WEIGHT_DISTANCE:
    #             weight = 0.0
    #         else:
    #             weight = G.QUAD_A*dist**2 + G.QUAD_B*dist + G.QUAD_C
    #     else:
    #         weight = 1.0
    #
    #     self.dqs_dist = dist
    #     self.dqs_w = weight
    #     log.debug("Line (%f,%f), Fiber (%f,%f), dist = %f, weight = %f" %(ra,dec,self.ra,self.dec,dist,weight))
    #     return weight

    # def dqs_score(self,ra,dec,force_recompute=False): #yeah, redundantly named ...
    #     if self.dqs_bad:
    #         return 0.0
    #
    #     if (self.dqs is not None) and not force_recompute:
    #         return self.dqs
    #
    #     self.dqs = 0.0
    #     self.dqs_raw = 0.0
    #     if (ra is None) or (dec is None) or (self.ra is None) or (self.dec is None):
    #         return self.dqs
    #
    #     weight = self.dqs_weight(ra,dec)
    #     score = 0.0
    #     sqrt_sn = 100.0 #above linear_sn
    #     linear_sn = 3.0 #above sq_sn
    #     sq_sn = 2.0
    #     base_sn = 3.0
    #     #build score (additive only)
    #     if self.sn:
    #         if self.sn < base_sn:
    #             score += 0.0
    #         elif self.sn < (base_sn + sq_sn):
    #             score += (self.sn - base_sn)**2
    #         elif self.sn < (base_sn + sq_sn + linear_sn): #linear growth
    #             #square growth part
    #             score += sq_sn**2
    #             #linear part
    #             score += (self.sn - (base_sn + sq_sn))
    #         elif self.sn < (base_sn + sq_sn + linear_sn + sqrt_sn): #sqrt growth
    #             # square growth part
    #             score += sq_sn ** 2
    #             # linear part
    #             score += linear_sn
    #             #sqrt growth part
    #             score += np.sqrt(1. + self.sn-(base_sn+sq_sn+linear_sn))-1
    #         else:
    #             log.info("Unexpected, really large S/N (%f) for %s" % (self.sn,self.idstring))
    #             score += -1.0 #same as low score ... something really wrong sn 100+ is nonsense (thinking cosmic ray?)
    #
    #     self.dqs_raw = score
    #     self.dqs = weight * score
    #
    #     log.debug("DetID # %d , Fiber: %s , Dist = %g , Raw Score = %g , Weighted Score = %g"
    #               %(self.detect_id,self.idstring, self.dqs_dist, self.dqs_raw, self.dqs))
    #
    #     return self.dqs
    #
