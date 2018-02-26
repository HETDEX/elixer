
#collection of a few functions to find and load panacea files
#and an IFU class that holds all exposures (which in turn hold the fits files and the fibers)
#for an ifu for one observation
#and provides some basic operations (like summing up fibers and interpolating)

import global_config as G
import numpy as np
import os
import glob
import fnmatch
import fiber as voltron_fiber
import hetdex_fits
from astropy.io import fits as pyfits

import spectrum as voltron_spectrum


UNITS = ['counts','cgs-17']

def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


AMP = voltron_fiber.AMP
AMP_OFFSET = voltron_fiber.AMP_OFFSET

MIN_WAVELENGTH = 3500.0
MAX_WAVELENGTH = 5500.0
INTERPOLATION_AA_PER_PIX = 2.0

log = G.logging.getLogger('ifu_logger')
log.setLevel(G.logging.DEBUG)


def find_first_file(pattern, path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None


def find_panacea_path_info(date,time_ex=None,time=None,basepath=G.PANACEA_RED_BASEDIR):
    """

    :param date: date string YYYYMMDD
    :param time: time string hhmmss
    :param time_ex: extended time string hhmmss.s
    :return: None or dictionary with path, obsid, and expid
    """

    #get the basedir for the date to the reduction data
    info = {}
    ex = True
    if time_ex is not None: #use it vs time
        target_time = time_ex
    elif time is not None:
        target_time = time
        ex = False
    else:
        log.error("Invalid parameters passed to find_panacea_path_info()")
        return info

    try:
        path = os.path.join(basepath, date, "virus")
    except:
        log.error("Cannot build path to panacea file.",exc_info=True)
        return info

    if not os.path.exists(path):
        log.error("Cannot locate reduced data for %s %s" % ())
        return info

    #path set above looks like:
    #../red1/reductions/20170603/virus/

    #complete path looks like:
    #../red1/reductions/20170603/virus/virus0000002/exp01/virus
    #../red1/reductions/20180113/virus/virus0000011/exp01/virus

    #get all the subdirectories
    subdirs = glob.glob(os.path.join(path, "virus*/exp*/virus"))

    #walk each sub directory for observation id and exp id and find any
    #multi*fits file, query its header and check for a time match
    #if the time does match, pull the observation id and exp id from the path
    #and return those (with the path)

    for s in subdirs: #order here does not matter
        #already know the subdirs exist (though, technically could be a filename rather than a
        #directory, but will trust in the fixed naming convention
        if len(info) > 0:
            break

        multi = glob.glob(os.path.join(s,"multi*"))
        for m in multi:
            try:
                f = pyfits.open(m)
            except:
                log.error("could not open file " + m, exc_info=True)
                continue

            try:
                ut = f[0].header['UT']  #format like = '08:57:00.902'
                if ex:
                    tx = ut[0:2]+ut[3:5]+ut[6:10] #if ut is bad, let the try block deal with it
                else:
                    tx = ut[0:2]+ut[3:5]+ut[6:8]

                    try: #either way, we are done with this file
                        f.close()
                    except:
                        log.error("could not close file " + m, exc_info=True)

                    if target_time == tx:
                        info['obsid'] = s.split("virus/virus")[1].split("/")[0]
                        info['expid'] = s.split("/exp")[1].split("/")[0]
                        info['path'] = s #overwrite path defined above

                        log.info("Found panacea mult* files: %s" %s)

                    break

            except:
                log.error("Could not read header value [UT]. Will try [RAWFN] for file: " + m, exc_info=False)

            #older panacea multi files did not have the UT value, but did have RAWFN that contains it in
            #the path
            try:
                #looks like
                # '/work/03946/hetdex/maverick/20180113/virus/virus0000015/exp02/virus&/20180113T085700.9_035RU_sci.fits'
                rawfn = os.path.basename(f[0].header['RAWFN']) #20180113T085700.9_035RU_sci.fits
                #want the time part
                if ex:
                    tx = (rawfn.split('T')[1]).split('_')[0] #if ut is bad, let the try block deal with it
                else:
                    tx = (rawfn.split('T')[1]).split('.')[0]

                try: #either way, we are done with this file
                    f.close()
                except:
                    log.error("could not close file " + m, exc_info=True)

                if target_time == tx:
                    info['obsid'] = s.split("virus/virus")[1].split("/")[0]
                    info['expid'] = s.split("/exp")[1].split("/")[0]
                    info['path'] = s #overwrite path defined above

                    log.info("Found panacea mult* files: %s" %s)

                break

            except:
                log.error("Could not read header values for file: " + m, exc_info=False)
                continue


    if len(info) == 0:
        log.info("Unable to locate panacea multi* files for %sT%s" %(date,target_time))

    return info

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

        self.sum_wavelengths = []
        self.sum_values = []
        self.sum_errors = []
        self.sum_count = 0

        if idstring is not None:
            try:
                dict_args = voltron_fiber.parse_fiber_idstring(idstring)

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

        #fast way (old way) first?
        if self.scifits_idstring is None:
                #had enough to build the basics, but not a full string
            if (self.obsid is not None) and (self.date is not None):
                self.scifits_idstring = self.date
            else:
                self.scifits_idstring = self.idstring.split("_")[0]

        scifile = find_first_file("*" + self.scifits_idstring + "*", path)

        if scifile:
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
        else:
            log.info("Filename method failed. Will try fits headers. Could not locate reduction data for %s" % (self.idstring))

            info = find_panacea_path_info(self.date,self.time_ex,self.time)

            if len(info) > 0:
                obsid = info['obsid']
                self.obsid = int(obsid)

                if expid < 1: #passed in value
                    min_exp = 1
                    max_exp = 99
                else:
                    min_exp = expid
                    max_exp = expid+1

                self.expid = int(expid)
            else:
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


    def build_fibers(self,grid_size=INTERPOLATION_AA_PER_PIX):

        #want the fe_data (fiber extracted data)
        #maybe give voltron_fiber.fiber a HETDEX fits object to pull the data? or pull it here and feed it to fiber?

        #grid_size = 2.0 #AA per pixel (seems to be okay ... no real difference in flux (area under the spectra))
        for exp in self.exposures:
            del exp.fibers[:]
            exp.fibers = [None]*448  # list of all fibers [448]

            for fits in exp.fits:
                for i in range(len(fits.fe_data)):
                    fib = voltron_fiber.Fiber(self.idstring,amp=fits.amp,panacea_fiber_index=i) #this will have the original amp, side

                    #update with THIS file's amp and side
                    fib.amp = fits.amp
                    fib.side = fits.side
                    fib.expid = fits.expid

                    #fiber sky position (center of fiber)
                    fib.center_x = fits.fiber_centers[i][0]
                    fib.center_y = fits.fiber_centers[i][1]

                    #get the recorded data (counts and corresponding wavelengths)
                    #divide by the fiber_to_fiber
                    #fiber_to_fiber is on the same grid as fe_data
                    fib.data_spectra_counts = fits.fe_data[i] / fits.fiber_to_fiber[i]
                    fib.data_spectra_wavelengths = fits.wave_data[i]

                    #errors are the average for all fibers (so multiply by specific fiber to fiber)
                    #[0] is the wavelength, [1] is either empirical or estimated error and [2] is the other
                    #[1] and [2] are fairly close and neither is exactly right, so it does not matter which to use
                    fib.data_spectra_errors = fits.fiber_to_fiber[i] * np.interp(fib.data_spectra_wavelengths,
                                                        fits.error_analysis[0],fits.error_analysis[1])

                    #interpolate onto a fixed length, 1 Angstrom grid
                    fib.interp_spectra_wavelengths = np.arange(MIN_WAVELENGTH,MAX_WAVELENGTH+grid_size,grid_size)
                    fib.interp_spectra_counts = np.interp(fib.interp_spectra_wavelengths,fib.data_spectra_wavelengths,
                                                          fib.data_spectra_counts)

                    #already adjusted the errors to the data grid AND adjusted for fiber to fiber, so just interpolate
                    fib.interp_spectra_errors = np.interp(fib.interp_spectra_wavelengths,
                                                          fib.data_spectra_wavelengths,
                                                          fib.data_spectra_errors)

                    #test: testing that interpolotion of 1 AA per pix prodcues the same area under the spectra
                    #as the original raw data
                    #
                    #####################################
                    #seems that any interpolation (even 2.0 or 1.9 vs 1.0 AA per pix) gives no difference
                    #however, raw data vs interpolation (say at 1.0) is always a bit different, but usually around
                    # 0.05% different and almost always less than 0.1% different. In the extremely few cases that are greater
                    # than that (even up to 75% different, there is one (or a few?) maxed out pixel(s) that is causing
                    # the problem
                    ######################################
                    #
                    #print("***** Temporary Test ******")
                    #
                    ##temp ...testing interpolation
                    # left = getnearpos(fib.data_spectra_wavelengths,MIN_WAVELENGTH)
                    # right = getnearpos(fib.data_spectra_wavelengths,MAX_WAVELENGTH)
                    # raw_counts = fib.data_spectra_counts[left:right+1]
                    # raw_x = fib.data_spectra_wavelengths[left:right+1]
                    #
                    # mn = np.min(raw_counts)
                    # if mn < 0:
                    #     fib.data_spectra_counts += -1*mn
                    # rc = np.trapz(raw_counts, raw_x)
                    #
                    # grid_size = 1.0  # AA per pixel
                    # fib.interp_spectra_wavelengths = np.arange(MIN_WAVELENGTH, MAX_WAVELENGTH + grid_size, grid_size)
                    # fib.interp_spectra_counts = np.interp(fib.interp_spectra_wavelengths, fib.data_spectra_wavelengths,
                    #                                       fib.data_spectra_counts)
                    # ic = np.trapz(fib.interp_spectra_counts,fib.interp_spectra_wavelengths)
                    # #rd = abs( (rc-ic)/(0.5*(rc+ic)))
                    # rd =  (rc-ic)/rc
                    # if abs(rd) > 0:
                    #     print("****** rc(%f), ic(%f), diff=%f" %(rc,ic,rd))
                    # else:
                    #     print("$$$$$$ same")
                    #
                    # if (abs(rd) < 0.001) and (abs(rd) > 0.0001):
                    #     import matplotlib.pyplot as plt
                    #     plt.close()
                    #     plt.xlim((3495,5505))
                    #     plt.plot(raw_x, raw_counts)
                    #     plt.plot(fib.interp_spectra_wavelengths,fib.interp_spectra_counts,c='r',ls=":")
                    #     plt.savefig("what.png")
                    #     print("!!!!!! rc(%f), ic(%f), diff=%f" % (rc, ic, rd))


                    # remember panacea idx number backward so idx=0 is fiber #112, idx=1 is fiber #111 ... idx=336 = #448
                    # so, fill in accordingly
                    exp.fibers[fib.number_in_ccd-1] = fib

    def sum_fibers_from_fib_file(self,file,line_id,fiber_indicies=None):
        """

        :param file:  full path to the voltron *_fib.txt file
        :param line_id: the id number of the line to use (1st column in *_fib.txt or the parenthetical value in pdf)
        :param fiber_indicies: zero based index of the fibers (as listed in t#cut or similar to add (adds all if is None or if empty)
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
                        if int(line_id) == int(toks[0]): #we found the line we want
                            break
        except:
            log.error("Unable to parse voltron fiber file: %s" %file)

        return self.sum_fibers_from_voltron(line,version,fiber_indicies)


    def sum_fibers_from_voltron(self,line,version=None,fiber_indicies=None):
        """
        :param line: the string (line) from a voltron *_fib.txt file
        :param version: the version number of the *_fib.txt file if known
        :param fiber_indicies: zero based index of the fibers (as listed in t#cut or similar to add (adds all if is None or if empty)
        :return: three arrays and a value: (1) wavelengths ,  (2) summed  values, (3) summed errors and (4) central wavelength
        """

        if (line is None) or (len(line) < 20):
            return None, None, None

        toks = line.split()
        len_toks = len(toks)

        x = []
        v = []
        e = []
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
            fiber_idx = 0
            try:
                while idx < len_toks:

                    if (not fiber_indicies) or (len(fiber_indicies) == 0) or (fiber_idx in fiber_indicies):
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

                            e = self.exposure(exp).fibers[self.get_absolute_fiber_index(amp, fib)].interp_spectra_errors
                            e = np.array(e)
                        else:
                            v += self.exposure(exp).fibers[self.get_absolute_fiber_index(amp, fib)].interp_spectra_counts
                            e += self.exposure(exp).fibers[self.get_absolute_fiber_index(amp, fib)].interp_spectra_errors


                        count_of_fibers += 1


                    #advance to next entry
                    idx += advance
                    fiber_idx += 1

                if count_of_fibers != number_of_fibers:
                    log.warning(
                        "Fiber counts do not match. (expected %d , got %d)" % (number_of_fibers, count_of_fibers))

                if count_of_fibers < 1:
                   v = []
                else:
                    log.info("Summed %d fibers" % count_of_fibers)
                    #v = v / float(count_of_fibers) #keep all, do not normalize to fibers

            except:
                pass #we're done


        return x,v,e,cw


    def is_fiber_empty(self,wavelengths,values,errors=None, units=None,max_score=10.0, max_snr = 2.0):
        '''
        Basically, is the fiber free from any overt signals (real emission line(s), continuum, sky, etc)
        Values and errors must have the same units
        Values and errors must already be normalized (e.g. values/fiber_to_fiber, errors*fiber_to_fiber, etc)
        Would be best (downstream) if they were also flux calibtrated, but does not matter much in this function

        '''
        # todo: reject if any signal found
        # could find peaks and kick out anything with S/N > 3?
        # what about contiuum??
        #  could average over chunks of, say, 100 pixels and reject if any chunk is above a threshold
        #   but then would need to know if these are counts, ergs, or what units???

        #what about a relative distance between min and max as a simple check??
        # (like    (max-min)/(.5*abs(max+min)) ...not that ... basically will always be 2
        # maybe (max-min)/(mean(values)) should not be more than 2 or 3?
        rc = False
        try:

#            extrema_ratio = (max(values) - min(values))/np.mean(values)
#            print (extrema_ratio)

            if (units is None) or (units == 'counts'):
                if min(values) < -50.0 or max(values) > 50.0:
                    log.debug("Fiber rejected for addition. Large extrema.")
                    return False

            peaks = voltron_spectrum.peakdet(wavelengths, values, dw=5.0)#, h, dh, zero)
            #signal = list(filter(lambda x: (x[5] > max_score) or (x[6] > max_snr),peaks))
            signal = list(filter(lambda x: x[6] > max_snr,peaks))

            if len(signal) == 0:
                rc = True
            else:
                log.debug("Fiber rejected for addition. Peaks found.")
        except:
            log.debug("Exception in ifu::is_fiber_empty() ", exc_info=True)

        return rc

    def sum_empty_fibers(self):
        """
        iterate over all fibers in this IFU (over all exposures for this shot)
        sum up those that are apparently empty

        :return: count of summed fibers
        """

        del self.sum_wavelengths[:]
        del self.sum_values[:]
        del self.sum_errors[:]
        self.sum_count = 0

        for e in self.exposures:
            for f in e.fibers:
                if self.is_fiber_empty(f.interp_spectra_wavelengths,f.interp_spectra_counts,f.interp_spectra_errors):
                    self.sum_count += 1
                    if len(self.sum_wavelengths) == 0:
                        self.sum_wavelengths = np.array(f.interp_spectra_wavelengths)
                        self.sum_values = np.array(f.interp_spectra_counts)
                        self.sum_errors = np.array(f.interp_spectra_errors)
                    else:
                        self.sum_values += np.array(f.interp_spectra_counts)
                        self.sum_errors += np.array(f.interp_spectra_errors)

        return self.sum_count


#end class IFU
