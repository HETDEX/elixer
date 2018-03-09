import global_config as G
import numpy as np

log = G.logging.getLogger('fiber_logger')
log.setLevel(G.logging.DEBUG)

SIDE = ["L", "R"]

#!!! REMEBER, Y-axis runs 'down':  Python 0,0 is top-left, DS9 is bottom-left
#!!! so in DS9 LU is 'above' LL and RL is 'above' RU
AMP  = ["LU","LL","RL","RU"] #in order from bottom to top
AMP_OFFSET = {"LU":1,"LL":113,"RL":225,"RU":337}


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
            # log.warn("Unexpected fiber id string: %s" % fiber)
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

        self.dqs = None  # detection quality score for this fiber (as part of the DetObj owner)
        self.dqs_raw = None  # unweighted score
        self.dqs_dist = None  # distance from source
        self.dqs_w = None #dqs weight
        self.dqs_bad = False
        self.central_wave_pixels_bad = 0
        self.central_emis_counts = [] #from fiber-extracted ... the few pixels around the peak
        self.central_emis_wavelengths = []
        self.central_emis_errors = []


        #full length    
        self.data_spectra_wavelengths=[]
        self.data_spectra_counts=[]
        self.data_spectra_errors = []
        self.max_spectra_count = None

        #interpolated onto a 1 angstrom grid
        self.interp_spectra_wavelengths = []
        self.interp_spectra_counts = []
        self.interp_spectra_errors = []

        self.multi = None
        self.relative_weight = 1.0
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


    def dqs_weight(self,ra,dec):
        weight = 0.0
        #specifically None ... a 0.0 RA or Dec is possible
        if (ra is None) or (dec is None) or (self.ra is None) or (self.dec is None):
            self.dqs_dist = 999.9
            return weight

        dist = np.sqrt( (np.cos(np.deg2rad(dec))*(ra-self.ra)) ** 2 + (dec - self.dec) ** 2) * 3600.

        if dist > G.FULL_WEIGHT_DISTANCE:
            if dist > G.ZERO_WEIGHT_DISTANCE:
                weight = 0.0
            else:
                weight = G.QUAD_A*dist**2 + G.QUAD_B*dist + G.QUAD_C
        else:
            weight = 1.0

        self.dqs_dist = dist
        self.dqs_w = weight
        log.debug("Line (%f,%f), Fiber (%f,%f), dist = %f, weight = %f" %(ra,dec,self.ra,self.dec,dist,weight))
        return weight

    def dqs_score(self,ra,dec,force_recompute=False): #yeah, redundantly named ...
        if self.dqs_bad:
            return 0.0

        if (self.dqs is not None) and not force_recompute:
            return self.dqs

        self.dqs = 0.0
        self.dqs_raw = 0.0
        if (ra is None) or (dec is None) or (self.ra is None) or (self.dec is None):
            return self.dqs

        weight = self.dqs_weight(ra,dec)
        score = 0.0
        sqrt_sn = 100.0 #above linear_sn
        linear_sn = 3.0 #above sq_sn
        sq_sn = 2.0
        base_sn = 3.0
        #build score (additive only)
        if self.sn:
            if self.sn < base_sn:
                score += 0.0
            elif self.sn < (base_sn + sq_sn):
                score += (self.sn - base_sn)**2
            elif self.sn < (base_sn + sq_sn + linear_sn): #linear growth
                #square growth part
                score += sq_sn**2
                #linear part
                score += (self.sn - (base_sn + sq_sn))
            elif self.sn < (base_sn + sq_sn + linear_sn + sqrt_sn): #sqrt growth
                # square growth part
                score += sq_sn ** 2
                # linear part
                score += linear_sn
                #sqrt growth part
                score += np.sqrt(1. + self.sn-(base_sn+sq_sn+linear_sn))-1
            else:
                log.info("Unexpected, really large S/N (%f) for %s" % (self.sn,self.idstring))
                score += -1.0 #same as low score ... something really wrong sn 100+ is nonsense (thinking cosmic ray?)

        self.dqs_raw = score
        self.dqs = weight * score

        log.debug("DetID # %d , Fiber: %s , Dist = %g , Raw Score = %g , Weighted Score = %g"
                  %(self.detect_id,self.idstring, self.dqs_dist, self.dqs_raw, self.dqs))

        return self.dqs

