#todo: contain list of fibers (fiber.py) that belong to the ifu
#todo: ifu has specification as to its IDentity and data/time/exposure
#todo: provide sub-selection of its fibers (i.e. annulus, max value cut, etc)

#todo: can find itself from id info (e.g. find the corresponding panacea fits? or should that be in another object?)
#has 4 amps (LL,LU, RL,RU)
#do we care about the amps??? or just merge them all together into an IFU

import global_config as G
import numpy as np
import os
import glob
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

    def __init__(self,idstring,ifuslot=None,ifuid=None,specid=None,build=True):

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
                    if dict_args == False: #failed to parse ... might just be the date-time part

                        if len(idstring) == 17:
                            self.idstring = idstring
                            self.date = idstring[0:8]
                            # next should be 'T'
                            self.time = idstring[9:15]  # not the .# not always there
                            if idstring[15] == ".":
                                self.time_ex = idstring[9:17]
                            else:
                                self.time_ex = None
                        elif len(idstring) > 9:
                            if idstring[8] == 'v': #might be of form 20171014v004 or 20171014v004_107
                                try:
                                    self.idstring = idstring
                                    self.date = idstring[0:8]
                                    self.obsid = int((idstring.split('v')[1]).split('_')[0])
                                except:
                                    log.error("Unable to parse idstring: %s" %idstring)
                                    return
                        else:
                            log.error("Unable to parse idstring: %s" % idstring)
                            return
                    else:
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

        if ifuslot is not None:
            self.ifuslot = str(ifuslot).zfill(3)

        if ifuid is not None:
            self.ifuid = str(ifuid).zfill(3)

        if specid is not None:
            self.specid = str(specid).zfill(3)


        self.exposures = [] #list of exposures (that contain the fits files and fibers)

        #self.fits = [] #list of panacea fits as Hetdex Fits objects
        #self.fibers = [None]*448  # list of all fibers [448]


        if build:
            self.build_from_files()
            self.build_fibers()


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

        if  (self.specid is None) and (self.ifuslot is None) and (self.ifuid is None):
            log.error("Cannot build IFU data. Missing required info: specid, ifuslotid, and ifuid")
            return None

        try:
            path = os.path.join(G.PANACEA_RED_BASEDIR, self.date, "virus")
            if (self.obsid is not None):
                path = os.path.join(path, "virus" + str(self.obsid).zfill(7))
        except:
            log.error("Cannot build path to panacea file.",exc_info=True)
            return None

        if not os.path.exists(path):
            log.error("Cannot locate reduced data for %s" % (self.idstring))
            return None

        if self.scifits_idstring is None:
                #had enough to build the basics, but not a full string
            if (self.obsid is not None) and (self.date is not None):
                self.scifits_idstring = self.date
            else:
                self.scifits_idstring = self.idstring.split("_")[0]

        scifile = find_first_file("*" + self.scifits_idstring + "*", path)

        if not scifile:
            log.error("Cannot locate reduction data for %s" % (self.idstring))
            return None
        else:
            log.debug("Found reduction folder for file: " + scifile)

        try:
            if self.obsid is None:
                obsid = scifile.split("virus/virus")[1].split("/")[0]
            else:
                obsid = str(self.obsid).zfill(7)
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

            # leaves off the exp01/virus/
            multi_fits_basepath = os.path.join(G.PANACEA_RED_BASEDIR, self.date, "virus",
                                               "virus" + str(self.obsid).zfill(7))

            #at least one is not set, find an example file and populate the others
            if (self.specid is None) or (self.ifuslot is None) or (self.ifuid is None):
                check_file = "multi_"
                if self.specid is None:
                    check_file += "???_"
                else:
                    check_file += self.specid + "_"

                if self.ifuslot is None:
                    check_file += "???_"
                else:
                    check_file += self.ifuslot + "_"

                if self.ifuid is None:
                    check_file += "???_"
                else:
                    check_file += self.ifuid + "_"

                check_file += "*"

                path = os.path.join(multi_fits_basepath, "exp" + str(exp).zfill(2), "virus")
                names = glob.glob(os.path.join(path, check_file))

                #actually expect there to be 4 ... one for each amp
                if len(names) > 0:
                    toks = os.path.basename(names[0]).split("_")
                    self.specid = toks[1]
                    self.ifuslot = toks[2]
                    self.ifuid = toks[3]

                    #also need to update the idstring with this additional information (the full string
                    #is expected downstream)
                    self.idstring = self.idstring + "_" + self.specid + "_" + self.ifuslot + "_" + self.ifuid + "_LL_001"

            # now build the path to the multi_*.fits and the file basename
            # leaves off the  LL.fits etc
            multi_fits_basename = "multi_" + self.specid + "_" + self.ifuslot + "_" + self.ifuid + "_"


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

    # version 1.3.5
    # each row contains one emission line with accompanying fiber information
    # 1 input (entry) ID
    # 2 detect ID
    # 3 detection quality score
    # 4 emission line RA (decimal degrees)
    # 5 emission line Dec (decimal degrees)
    # 6 emission line wavelength (AA)
    # 7 emission line sky X
    # 8 emission line sky Y
    # 9 emission line sigma (significance) for cure or S/N for panacea
    # 10 emission line chi2 (point source fit) (cure)
    # 11 emission line estimated fraction of recovered flux
    # 12 emission line flux (electron counts)
    # 13 emission line flux (cgs)
    # 14 emission line continuum flux (electron counts)
    # 15 emission line continuum flux (cgs)
    # 16 emission line equivalent width (observed) [estimated]
    # 17 P(LAE)/P(OII),number of fiber records to follow (each consists of the following columns)
    # 18   fiber_id string (panacea) or reduced science fits filename (cure)
    # 19   observation date YYYYMMDD
    # 20   observation ID (for that date)
    # 21   exposure ID
    # 22   fiber number on full CCD (1-448)
    # 23   RA of fiber center
    # 24   Dec of fiber center
    # 25   S/N of emission line in this fiber
    # 26   weighted quality score
    # 27   X coord on the CCD for the amp of this emission line in this fiber (as shown in ds9)
    # 28   Y coord on the CCD for the amp of this emission line in this fiber (as shown in ds9)
    # 29   the next fiber_id string and so on ...
    ###############
    #  0     1   2   3           4           5       6      7        8   9   10  11        12                 13
    # 240	240	5.0	214.92604	52.57537	5378.4	24.15	-10.29	6.4	666	1.0	508.9	9.3383363738e-17	15.03
    #
    # 14                 15                16           17
    # 2.7580113126e-18	33.8589487691	0.284801251776	5
    #
    #18                                     19         20  21   22   23             24      25  26   27 28
    # 20170603T065529.2_051_105_051_LL_089	20170603	2	3	136	214.92616	52.57556	6.4	5.4	955	818
    #  0                  1  2   3  4  5
    #
    # 29
    # 20170603T064844.9_051_105_051_LL_089	20170603	2	2	136	214.92589	52.57519	6.2	5.2	955	818
    # 20170603T065529.2_051_105_051_LL_109	20170603	2	3	116	214.92627	52.57486	5.1	3.02202414323	954	994
    # 20170603T064844.9_051_105_051_LL_090	20170603	2	2	135	214.92694	52.57489	2.8	0.0	955	827
    # 20170603T064844.9_051_105_051_LL_070	20170603	2	2	155	214.92683	52.5756	2.6	0.0	957	643


    #cw=5378.4

    #x = i.exposure(3).fibers[i.get_absolute_fiber_index("LL",89)].interp_spectra_wavelengths

    #v  = i.exposure(3).fibers[i.get_absolute_fiber_index("LL",89)].interp_spectra_counts
    #v += i.exposure(2).fibers[i.get_absolute_fiber_index("LL",89)].interp_spectra_counts


    def sum_fibers_from_fib_file(self,file,id):
        """

        :param file:  full path to the voltron *_fib.txt file
        :param id: the id number of the line to use
        :return: two arrays and a value: (1) wavelengths ,  (2) summed and averaged values and (3) central w
        """

        #make sure file exists
        #find the line
        #call sum_fibers...

        line = None
        version = None

        if not os.path.exists(file):
            log.error("Provided voltron fiber file does not exist: %s" %file)
            return None, None, None

        try:
            with open(file, 'r') as f:
                for line in f:
                    if line[0] == '#' or len(line) < 100: #a commonent
                        if (version is None) and ("version" in line):
                            # version 1.3.5
                            toks = line.split()
                            version = toks[2]
                    else: #data line
                        toks = line.split()
                        if int(id) == int(toks[0]): #we found the line we want
                            break



        except:
            log.error("Unable to parse voltron fiber file: %s" %file)


        return self.sum_fibers_from_voltron(line,version)


    def sum_fibers_from_voltron(self,line,version=None):
        """

        :param line: the string (line) from a voltron *_fib.txt file
        :param version: the version number of the *_fib.txt file if known
        :return: two arrays and a value: (1) wavelengths ,  (2) summed and averaged values and (3) central wavelength
        """

        if (line is None) or (len(line) < 20):
            return None, None, None

        toks = line.split()
        len_toks = len(toks)

        x = []
        v = []
        cw = 0.0

        #todo: replace as needed with different parsing depending on version number
        if True:
            advance = 11
            input_id = int(toks[0])
            detect_id = int(toks[1])

            cw = float(toks[5])
            number_of_fibers = int(toks[17])

            idx = 18
            count_of_fibers = 0 #validate against number_of_fibers
            try:
                while idx < len_toks:
                    idstr = toks[idx]
                    #get the amp and the fiber id
                    id_toks = idstr.split("_")
                    if len(id_toks) != 6:
                        log.error("Invalid fiber id string: %s" % (idstr))
                        idx += advance
                        continue

                    amp = id_toks[4]
                    fib = int(id_toks[5])
                    exp = int(toks[idx+3])

                    if len(x) == 0:
                        x = self.exposure(exp).fibers[self.get_absolute_fiber_index(amp,fib)].interp_spectra_wavelengths
                        x = np.array(x)

                    if len(v) == 0:
                        v = self.exposure(exp).fibers[self.get_absolute_fiber_index(amp, fib)].interp_spectra_counts
                        v = np.array(v)
                    else:
                        v += self.exposure(exp).fibers[self.get_absolute_fiber_index(amp, fib)].interp_spectra_counts

                    count_of_fibers += 1


                    #advance to next entry
                    idx += advance

                if count_of_fibers != number_of_fibers:
                    log.warning(
                        "Fiber counts do not match. (expected %d , got %d)" % (number_of_fibers, count_of_fibers))

                if count_of_fibers > 0:
                    log.info("Summed %d fibers" % count_of_fibers)
                    v = v / float(count_of_fibers)
                else:
                    v = []

            except:
                pass #we're done


        return x,v,cw

#end class IFU
