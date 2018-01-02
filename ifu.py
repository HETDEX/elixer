#todo: contain list of fibers (fiber.py) that belong to the ifu
#todo: ifu has specification as to its IDentity and data/time/exposure
#todo: provide sub-selection of its fibers (i.e. annulus, max value cut, etc)

#todo: can find itself from id info (e.g. find the corresponding panacea fits? or should that be in another object?)
#has 4 amps (LL,LU, RL,RU)
#do we care about the amps??? or just merge them all together into an IFU

import global_config as G
import numpy as np
import os
import fnmatch
import fiber as ifu_fiber
import hetdex_fits


AMP = ifu_fiber.AMP
AMP_OFFSET = ifu_fiber.AMP_OFFSET

MIN_WAVELENGTH = 3500.0
MAX_WAVELENGTH = 5500.0

log = G.logging.getLogger('ifu_logger')
log.setLevel(G.logging.DEBUG)


def find_first_file(pattern, path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None


class IFU_EXPOSURE:
#holding class for fits files and fibers
    def __init__(self,exp):
        self.expid = exp

        self.fits = []  # list of panacea fits as Hetdex Fits objects
        self.fibers = [None] * 448  # list of all fibers [448]


class IFU:

    #is one IFU for one observation, and contains all exposures for that obervation
    #the fits files and fibers are stored under each IFU_EXPOSURE

    def __init__(self,idstring):

        #self.path = None #maybe multiples? 3x dithers?
        #self.filename = None #maybe multiples? 3x dithers and 4x amps

        self.basepath = None #excludes the last "/exp01/virus/"
        self.basename = None #excludes the "_LU.fits"

        self.idstring = None
        self.scifits_idstring = None #self.idstring.split("_")[0]
        self.specid = None
        self.ifuslot = None
        self.ifuid = None
        self.expid = None
        self.obsid = None

        self.date = None
        self.time = None
        self.time_ex = None
        self.dithernum = None #1,2,3

        #self.add_fiber_ids = []
        #self.add_values = []
        #self.add_wavelengths = []

        if idstring is not None:
            try:
                dict_args = ifu_fiber.parse_fiber_idstring(idstring)
                if dict_args is not None:
                    self.idstring = dict_args["idstring"]
                    self.scifits_idstring = self.idstring.split("_")[0]
                    self.specid = dict_args["specid"]
                    self.ifuslot = dict_args["ifuslot"]
                    self.ifuid = dict_args["ifuid"]
                    #self.amp = dict_args["amp"]
                    self.date = dict_args["date"]
                    self.time = dict_args["time"]
                    self.time_ex = dict_args["time_ex"]
            except:
                log.error("Exception: Cannot parse fiber string.", exc_info=True)

        self.exposures = [] #list of exposures (that contain the fits files and fibers)

        #self.fits = [] #list of panacea fits as Hetdex Fits objects
        #self.fibers = [None]*448  # list of all fibers [448]


    #@property
    def exposure(self,expid):
        for exp in self.exposures:
            if exp.expid == expid:
                return exp

    def get_absolute_fiber_index(self,amp,fiber_number): #returns absolute index (0-447)
        abs_idx = -1

        try:
            #reminder, panacea is indexed backward (idx=0 --> fiber #112 for that amp, idx=111 --> fiber #1)
            amp = amp.upper()
            offset = AMP_OFFSET[amp]
            abs_idx = offset + (112-fiber_number) -1
        except:
            pass

        return abs_idx

    def build_from_files(self,expid=-1):
        #find the path and filename based on other info

        try:
            path = os.path.join(G.PANACEA_RED_BASEDIR, self.date, "virus")
        except:
            log.error("Cannot build path to panacea file.",exc_info=True)
            return None

        if not os.path.exists(path):
            log.error("Cannot locate reduced data for %s" % (self.idstring))
            return None

        if self.scifits_idstring is None:
            self.scifits_idstring = self.idstring.split("_")[0]

        scifile = find_first_file("*" + self.scifits_idstring + "*", path)
        if not scifile:
            log.error("Cannot locate reduction data for %s" % (self.idstring))
            return None
        else:
            log.debug("Found reduction folder for file: " + scifile)

        try:
            obsid = scifile.split("virus/virus")[1].split("/")[0]
            if expid < 1:
                #expid = scifile.split("/exp")[1].split("/")[0]
                min_exp = 1
                max_exp = 99
            else:
                min_exp = expid
                max_exp = expid+1

            self.expid = int(expid)
            self.obsid = int(obsid)
        except:
            log.error("Cannot locate reduction data for %s" % (self.idstring))
            return None


        for exp in range(min_exp,max_exp,1):
            # now build the panace fits path
            path = os.path.join(G.PANACEA_RED_BASEDIR, self.date, "virus", "virus" + obsid, "exp" + str(exp).zfill(2),
                           "virus")

            if not os.path.exists(path):
                #if this was the only exposure to load, log an error
                if (min_exp+1 == max_exp):
                    log.error("Cannot locate panacea reduction data for %s" % (self.idstring))
                    return None
                #else, we were building all exposures and finally hit the last one, so just break out
                else:
                    break

            ifu_exp = IFU_EXPOSURE(exp)

            # now build the path to the multi_*.fits and the file basename
            # leaves off the  LL.fits etc
            multi_fits_basename = "multi_" + self.specid + "_" + self.ifuslot + "_" + self.ifuid + "_"
            # leaves off the exp01/virus/
            multi_fits_basepath = os.path.join(G.PANACEA_RED_BASEDIR, self.date, "virus",
                                          "virus" + str(self.obsid).zfill(7))

            self.basepath = multi_fits_basepath
            self.basename = multi_fits_basename

            # see if path is good and read in the panacea fits
            path = os.path.join(multi_fits_basepath, "exp" + str(exp).zfill(2), "virus")
            if os.path.isdir(path):
                for amp in AMP:
                    fn = os.path.join(path, multi_fits_basename + amp + ".fits")

                    if os.path.isfile(fn):  # or op.islink(fn):
                        log.debug("Found reduced panacea file: " + fn)

                        fits = hetdex_fits.HetdexFits(fn, None, None, -1, panacea=True)
                        fits.obs_date = self.date
                        fits.obs_ymd = fits.obs_date
                        fits.obsid = self.obsid
                        #fits.expid = self.expid
                        fits.amp = amp
                        fits.side = amp[0]


                        ifu_exp.fits.append(fits)


                       # self.fits.append(fits)
                    elif os.path.islink(fn):
                        log.error("Cannot open <" + fn + ">. Currently do not properly handle files as soft-links. "
                                                         "The path, however, can contain soft-linked directories.")
                    else:
                        log.error("Designated reduced panacea file does not exist: " + fn)

                self.exposures.append(ifu_exp)

            else:
                log.error("Cannot locate panacea reduction data for %s" % (path))
                return False


    def build_fibers(self):

        #want the fe_data (fiber extracted data)
        #maybe give ifu_fiber.fiber a HETDEX fits object to pull the data? or pull it here and feed it to fiber?

        for exp in self.exposures:
            del exp.fibers[:]
            exp.fibers = [None]*448  # list of all fibers [448]

            for fits in exp.fits:
                for i in range(len(fits.fe_data)):
                    fib = ifu_fiber.Fiber(self.idstring,amp=fits.amp,panacea_fiber_index=i) #this will have the original amp, side

                    #update with THIS file's amp and side
                    fib.amp = fits.amp
                    fib.side = fits.side
                    fib.expid = fits.expid

                    #fiber sky position (center of fiber)
                    fib.center_x = fits.fiber_centers[i][0]
                    fib.center_y = fits.fiber_centers[i][1]

                    #get the recorded data (counts and corresponding wavelengths)
                    fib.data_spectra_counts = fits.fe_data[i]
                    fib.data_spectra_wavelengths =  fits.wave_data[i]

                    #interpolate onto a fixed length, 1 Angstrom grid
                    fib.interp_spectra_wavelengths = np.arange(MIN_WAVELENGTH,MAX_WAVELENGTH+1.0,1.0)
                    fib.interp_spectra_counts = np.interp(fib.interp_spectra_wavelengths,fib.data_spectra_wavelengths,
                                                          fib.data_spectra_counts)

                    # remember panacea idx number backward so idx=0 is fiber #112, idx=1 is fiber #111 ... idx=336 = #448
                    # so, fill in accordingly
                    exp.fibers[fib.number_in_ccd-1] = fib


        #end class IFU
