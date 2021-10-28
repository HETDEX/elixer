#science image (usually very large) FITS file
#represents overall large image: provides small cutouts for display

#needs:
# image location/filename
# decriptive image name
# catalog name (to match back to catalog?)
# filter name
# wavelength(s) covered

#methods
# load file
# get status
# get cutout (takes ra,dec,error) returns cutout_image (the cutout is just another science image)

try:
    from elixer import global_config as G
    from elixer import utilities
except:
    import global_config as G
    import utilities

from hetdex_tools import phot_tools

import gc
from time import sleep
import mmap

import numpy as np
from astropy.visualization import ZScaleInterval
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
#from astropy.coordinates import match_coordinates_sky
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from astropy.wcs import Wcsprm
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import NoOverlapError
from astropy import units as ap_units
from photutils import CircularAperture #pixel coords
from photutils import SkyCircularAperture, SkyCircularAnnulus,SkyEllipticalAperture #sky coords
from photutils import aperture_photometry
from photutils import centroid_2dg
from astropy.stats import sigma_clipped_stats, sigma_clip
import astropy.stats.biweight as biweight
import astropy.wcs #need constants
import sep #source extractor python module

import scipy.stats as stats

import copy as cp

PIXEL_APERTURE_METHOD='exact' #'exact' 'center' 'subpixel'

#log = G.logging.getLogger('sciimg_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('sciimg_logger')
log.setlevel(G.LOG_LEVEL)



def get_line_image(friendid=None, detectid=None, coords=None, shotid=None, subcont=True, convolve_image=False,
                   pixscale=0.25, imsize=9.0, wave_range=None, return_coords=False):
    """
    Wrapper for hetdex_api.hetdex_tools.phot_tools get_line_image()
    A synthetic image from fibers using region around the emission line


    :return: a cutout like the science cutouts (mostly an astropy HDU)
    """
    cutout = None
    try:

        hdu = phot_tools.get_line_image(friendid=friendid,
                                        detectid=detectid,
                                        coords=coords,
                                        shotid=shotid,
                                        subcont=subcont,
                                        convolve_image=convolve_image,
                                        pixscale=pixscale,
                                        imsize=imsize,
                                        wave_range=wave_range,
                                        return_coords=return_coords)

        #there are 4 extensions in the HDU .. the 0th is the image we want
        cutout = cp.deepcopy(hdu[0])

    except:
        log.error("Exception calling hetdex_api's get_line_image(): ", exc_info=True)

    return cutout

def is_cutout_empty(cutout,check_unique_fraction=False):
    """
    Either all values the same or a simple horizontal or vertical gradient
    (currently, any other gradient would be missed ... assumes perpendicular to gradient is constant, min==max)
    :param cutout:
    :return:
    """

    if cutout is None:
        return True

    rc = False

    try:
        #std = np.nanstd(cutout.data)
        # mean = np.nanmean(cutout.data) #no NOT mean (could legit be 0 (average of + and - values))
        # med = np.nanmedian(cutout.data) #sort of a typical value to set a scale
        # sm = np.sum(cutout.data)
        # log.debug("Checking for emtpy cutout: mean (%f), median (%f), sum (%f), std (%f)" % (mean, med, sm, std))
        #
        # if std < 1e-5: #empty
        #     rc = True
        # elif (med / std) > 1e4:#prob empty or gradient
        #     rc = True

        # if (sm == 0) or (std == 0):
        #     rc = True

        #run through the middle:
        sp = np.shape(cutout.data)
        hz = cutout.data[:,int(sp[1]/2)]
        vt = cutout.data[int(sp[0]/2),:]

        # std = np.nanstd(cutout.data)
        # mx = np.max(cutout.data)
        # mn = np.min(cutout.data)

        #auto-correlation
        lag = max(1,int(sp[0]/10)) #say, 1/10 of the cutout size?
        #todo: what if we are on some large object that legitimately grows in a correlated fashion (centered on sphere?)
        #or the wings of a galaxy ... must be a really high correlation
        hzc = np.corrcoef(([hz[:-lag], hz[lag:]]))[0,1] #one of the off-diagonal values
        vtc = np.corrcoef(([vt[:-lag], vt[lag:]]))[0,1]

        #are either all the same
        if (min(hz) == max(hz)) or (min(vt) == max(vt)):
            #gradient or all same
            rc = True
            log.warning("Gradient or all same pixels found in cutout.")
        elif (hzc > 0.99) and (vtc > 0.99):
            rc = True
            log.warning("Autocorrelation in cutout implies empty.")
        # elif (std > 0.0) and (mx-mn)/std < 5.0: #complete width ... like saying all of one side is less than 2.5 std
        #     rc = True
        #     log.info("Too little variation in cutout. Assume empty.")

        #check unique values
        #u,uidx,ucts = np.unique(detectids,return_index=True,return_counts=True)
        if not rc:
            try:
                flat = cutout.data.flatten()
                frac_nonzero = len(np.where(flat!=0)[0])/len(flat)
                #sat = np.where(flat==) #what is saturated? (depends on each cutout)
                uniq_array, uniq_counts = np.unique(flat,return_counts=True)
                frac_uniq = len(uniq_array) / len(flat)

                #uniq_max = np.max(uniq_array)

                if len(uniq_counts) > 10:
                    #sum of top 3 counts
                    top_duplicates = np.sum(sorted(uniq_counts,reverse=True)[0:3])
                 #   uniq_max_sum = top_duplicates
                else:
                    #just the single top count
                    top_duplicates = np.max(uniq_counts)

                frac_top_duplicates = top_duplicates/len(flat)

                skew = stats.skew(flat)
                # very non normal (alone) is great (means signal)
                # skew could be + or -  and still have signal?
                # remember: skew is from the mean, not zero
                skew_zscore, skew_pval = stats.skewtest(flat) 
                mean = np.mean(flat)
                median = np.median(flat)
                std = np.std(flat)
                bwloc = biweight.biweight_location(flat)
                bwscale = biweight.biweight_scale(flat)

                zero_std = median / std # looking for a negative value, more negative than -1 (s|t median is more than
                                        # one std dev less than zero



                log.debug(f"mean ({mean}), median ({median}), std ({std}), bwloc ({bwloc}), bwscale ({bwscale}), "
                          f"skew ({skew}), skew_pval ({skew_pval}), zero_std ({zero_std})")

                sc_mean, sc_median, sc_std = sigma_clipped_stats(flat, sigma=3.0)
                clip = sigma_clip(flat,sigma=3.0,masked=False)
                sc_skew = stats.skew(clip)
                sc_kurt = stats.kurtosis(clip)
                sc_skew_zscore, sc_skew_pval = stats.skewtest(clip)
                sc_zero_std = sc_median / sc_std
                sc_run = np.max(clip)-np.min(clip)
                log.debug(f"sc_mean ({sc_mean}), sc_median ({sc_median}), sc_std ({sc_std})")
                log.debug (f"sc_skew ({sc_skew}), sc_skew_pval ({sc_skew_pval}), sc_zero_std ({sc_zero_std}), sc_kurt ({sc_kurt}), sc_run ({sc_run})")

                if frac_uniq < G.FRAC_UNIQUE_PIXELS_MINIMUM:
                    log.warning(f"Fraction of (minimum) unique pixels ({frac_uniq}) < ({G.FRAC_UNIQUE_PIXELS_MINIMUM}) "
                                f" Assume cutout is empty or simple pattern.")
                    rc = True
                elif (frac_uniq < G.FRAC_UNIQUE_PIXELS_NOT_EMPTY) and (frac_top_duplicates > G.FRAC_DUPLICATE_PIXELS):
                        #and (uniq_array[np.argmax(uniq_counts)] < 5000.0):
                    log.warning(f"Fraction of unique pixels ({frac_uniq}) < ({G.FRAC_UNIQUE_PIXELS_NOT_EMPTY}) "
                                f"and fraction of top duplicates ({frac_top_duplicates}) > ({G.FRAC_DUPLICATE_PIXELS}) "
                             f" Assume cutout is empty or simple pattern.")
                    rc = True
                elif ((hzc > 0.99) or (vtc > 0.99)) and (frac_uniq < G.FRAC_UNIQUE_PIXELS_AUTOCORRELATE):
                    log.warning(f"Fraction of unique pixels ({frac_uniq}) small AND horizontal or vertical "
                             f"auto-correlaction. Assume cutout is empty or simple pattern.")
                    rc = True
                elif frac_nonzero < G.FRAC_NONZERO_PIXELS:
                    log.warning(f"Fraction of zero pixels ({frac_nonzero}) < ({G.FRAC_NONZERO_PIXELS}). "
                                f"Assume cutout is empty or simple pattern.")
                    rc = True
                elif (sc_zero_std < -2.5) and (sc_skew_pval < 1e-20):
                    log.warning(f"Negative median ({sc_median},  ({sc_zero_std}) from zero,"
                                f"and very inconsistent with normal (pval {sc_skew_pval}) post sigma-clip. "
                               f"Assume cutout is empty or simple pattern or junk.")
                    rc = True
                elif sc_zero_std < -10.0 : #absurdly low (basically, small std dev and all values are very negative)
                    log.warning(f"Very negative values, ({sc_zero_std}) std from zero,"
                               f"Assume cutout is empty or simple pattern or junk.")
                    rc = True


                #no ... these can trap signal (real) sources and good images
                # elif (mean < 0) and (skew < -1.0) and (skew_pval < 1e-20):
                #     log.warning(f"Negative mean ({mean}, very negative skew ({skew}), and very inconsistent with normal (pval {skew_pval}). "
                #                f"Assume cutout is empty or simple pattern or junk.")
                #     rc = True
                # elif (median < mean) and (mean < 0) and (skew < 0.0) and (skew_pval < 1e-20):
                #     log.warning(f"Negative median ({median} < negative mean ({mean}), very negative skew ({skew}), "
                #                 f"and very inconsistent with normal (pval {skew_pval}). "
                #                f"Assume cutout is empty or simple pattern or junk.")
                #     rc = True
                elif frac_uniq < 0.9:
                    log.info(f"Low fraction of unique pixels ({frac_uniq}). Image may be bad.")
                    if check_unique_fraction:
                        rc = True
                    #print(f"Low fraction of unique pixels ({frac_uniq}). Image may be bad.")
            except:
                log.debug("*** Exception! Exception in science_image::is_cutout_empty()", exc_info=True)
    except:
        log.debug("*** Exception! Exception in science_image::is_cutout_empty()", exc_info=True)

    return rc



class science_image():

    def __init__(self, wcs_manual=False, image_location=None,frame=None, wcs_idx=0, hdulist=None):
        self.image_location = None
        self.image_name = None
        self.catalog_name = None
        self.filter_name = None
        self.wavelength_aa_min = 0
        self.wavelength_aa_max = 0
        self.last_x_center = None #lower left corner is 0,0
        self.last_y_center = None
        self.last_x0_center = None #center is 0,0
        self.last_y0_center = None

        self.ra_center = 0.0
        self.dec_center = 0.0

        #self.fits = None # fits handle

        self.hdulist = hdulist
        self.headers = None #array of hdulist headers

        self.wcs = None
        self.vmin = None
        self.vmax = None
        self.pixel_size = None #arcsec/pixel
        self.window = None
        self.exptime = None

        self.image_buffer = None

        self.wcs_manual = wcs_manual
        self.wcs_idx = wcs_idx #usually on hdu 0 but sometimes not (i.e. HyperSuprimeCam on 1)
        self.footprint = None #on sky footprint, decimal degrees Ra,Dec as 4x2 array (with North up: LL, UL, UR, LR)

        #todo: do I need to worry about this?
        if frame is not None:
            self.frame = frame
        else:
            self.frame = 'icrs' #todo: try icrs or fk5 (older)

        if (image_location is not None) and (len(image_location) > 0):
            self.image_location = image_location
            self.load_image(wcs_manual=wcs_manual)
        elif hdulist is not None:
            self.headers = []
            for i in range(len(hdulist)):
                self.headers.append(hdulist[i].header)

            self.wcs = WCS(hdulist[self.wcs_idx].header)
            try:
                self.footprint = WCS.calc_footprint(self.wcs)
            except:
                log.error("Unable to get on-sky footprint")

            try:
                self.exptime = float(self.hdulist[self.wcs_idx].header['EXPTIME'])
            except:
                try:  # if not with the wcs header, check the main (0) header
                    self.exptime = float(self.hdulist[0].header['EXPTIME'])
                except:

                    log.warning('Warning. Could not load exposure time from %s' % self.image_location)#, exc_info=True)
                    self.exptime = None
            try:
                self.pixel_size = self.calc_pixel_size(
                    self.wcs)  # np.sqrt(self.wcs.wcs.cd[0, 0] ** 2 + self.wcs.wcs.cd[0, 1] ** 2) * 3600.0  # arcsec/pixel
                log.debug("Pixel Size = %f asec/pixel" % self.pixel_size)
            except:
                log.error("Unable to build pixel size", exc_info=True)



    # def cycle_fits(self):
    #     if self.fits is not None:
    #         log.info("Closing fits and forcing garbage collection")
    #         self.fits.close() # should free the memory (i.e. for large files, even with memmap)
    #         del self.fits[self.wcs_idx].data
    #         del self.fits
    #         self.fits = None
    #
    #         gc.collect()
    #         log.info("Reopening fits")
    #         self.fits = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
    #         #self.load_image(self.wcs_manual)#

    def load_image(self,wcs_manual=False):
        if (self.image_location is None) or (len(self.image_location) == 0):
            return -1

        if self.hdulist is not None:
            try:
                self.hdulist.close()
            except:
                log.info("Unable to close fits file.")

        if self.headers is not None:
            del self.headers[:]

        try:
            log.info("Loading fits %s ..." % self.image_location)
            self.hdulist = fits.open(self.image_location,memmap=True,lazy_load_hdus=True)
        except:
            log.error("Unable to open science image file: %s" %self.image_location)
            return -1

        self.headers = []
        for i in range(len(self.hdulist)):
            self.headers.append(fits.getheader(self.image_location,i))


        if wcs_manual:
            self.build_wcs_manually()
        else:
            try:
                if (self.wcs_idx is not None) and (self.wcs_idx > 0):
                    f = fits.open(self.image_location)
                    self.wcs = WCS(f[self.wcs_idx].header,relax = astropy.wcs.WCSHDR_CD00i00j | astropy.wcs.WCSHDR_PC00i00j)
                    f.close()
                else:
                    #self.wcs = WCS(header=self.hdulist[self.wcs_idx].header,fobj=self.image_location)
                    self.wcs = WCS(self.image_location,relax = astropy.wcs.WCSHDR_CD00i00j | astropy.wcs.WCSHDR_PC00i00j)
            except:
                log.error("Unable to use WCS constructor. Will attempt to build manually.", exc_info=True)
                self.build_wcs_manually()

        if self.wcs is None: #must have WCS
            self.hdulist.close()
            return -1

        try:
            self.footprint = WCS.calc_footprint(self.wcs)
        except:
            log.error("Unable to get on-sky footprint")
            self.footprint = None

        try:
            if 'EXPTIME' in self.hdulist[self.wcs_idx].header:
                self.exptime = float(self.hdulist[self.wcs_idx].header['EXPTIME'])
            #if not with the wcs header, check the main (0) header
            elif 'EXPTIME' in self.hdulist[0].header:
                self.exptime = float(self.hdulist[0].header['EXPTIME'])
            else:
                self.exptime = None
                log.warning('Warning. [EXPTIME] not found in %s' % self.image_location)
        except:
            log.warning('Warning. Could not load exposure time from %s' %self.image_location, exc_info=True)
            self.exptime = None

        try:
            self.pixel_size = self.calc_pixel_size(self.wcs)#np.sqrt(self.wcs.wcs.cd[0, 0] ** 2 + self.wcs.wcs.cd[0, 1] ** 2) * 3600.0  # arcsec/pixel
            log.debug("Pixel Size = %f asec/pixel" %self.pixel_size)
        except:
            log.error("Unable to build pixel size", exc_info=True)


        #check the footprint
        try:
            if (self.footprint is not None) and (self.pixel_size is not None):
                ra_range = (max(self.footprint[:,0])-min(self.footprint[:,0])) * 3600.0
                dec_range = (max(self.footprint[:,1])-min(self.footprint[:,1])) * 3600.0
                #ignore keystoning from dec for now
                footprint_area = ra_range * dec_range / (self.pixel_size**2)
                pixel_area = self.wcs.pixel_shape[0] * self.wcs.pixel_shape[1]

                #should be close in size:
                if not (0.8 < footprint_area/pixel_area < 1.2):
                    log.error("Significant error in footprint ( %0.2f vs %0.2f sq.pixels)" %(footprint_area,pixel_area))
                    self.footprint = None
        except:
            log.error("Unable to verify footprint. Will set to None.")
            self.footprint = None

        # 02-13-2020 ... is this is no longer necessary to close the list  (the memory problem was resovled with Python3)
        self.hdulist.close() #don't keep it open ... can be a memory problem with wrangler
        self.hdulist = None
        return 0
    #end load_image


    def build_wcs_manually(self):

        close = False
        if self.hdulist is None:
            try:
                log.info("Loading (local scope) fits %s ..." % self.image_location)
                hdulist = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
                close = True
            except:
                log.error("Unable to open science image file: %s" % self.image_location)
                return -1
        else:
            log.info("Using outer scope fits %s ..." % self.image_location)
            hdulist = self.hdulist

        try:
            #astropy 3.1 makes _naxis? private so this no longer works
            #instead use updated constructor that builds from header

            # self.wcs = WCS(naxis=hdulist[self.wcs_idx].header['NAXIS'])
            # self.wcs.wcs.crpix = [hdulist[self.wcs_idx].header['CRPIX1'], hdulist[self.wcs_idx].header['CRPIX2']]
            # self.wcs.wcs.crval = [hdulist[self.wcs_idx].header['CRVAL1'], hdulist[self.wcs_idx].header['CRVAL2']]
            # self.wcs.wcs.ctype = [hdulist[self.wcs_idx].header['CTYPE1'], hdulist[self.wcs_idx].header['CTYPE2']]
            # #self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
            # self.wcs.wcs.cd = [[hdulist[self.wcs_idx].header['CD1_1'], hdulist[self.wcs_idx].header['CD1_2']],
            #                    [hdulist[self.wcs_idx].header['CD2_1'], hdulist[self.wcs_idx].header['CD2_2']]]
            # self.wcs._naxis1 = hdulist[self.wcs_idx].header['NAXIS1']
            # self.wcs._naxis2 = hdulist[self.wcs_idx].header['NAXIS2']

            #since is tuple, can't do the following
            #self.wcs.pixel_shape[0] = hdulist[self.wcs_idx].header['NAXIS1']
            #self.wcs.pixel_shape[1] = hdulist[self.wcs_idx].header['NAXIS2']

            self.wcs = WCS(naxis=hdulist[self.wcs_idx].header['NAXIS'])
            self.wcs.wcs.crpix = [hdulist[self.wcs_idx].header['CRPIX1'], hdulist[self.wcs_idx].header['CRPIX2']]
            self.wcs.wcs.crval = [hdulist[self.wcs_idx].header['CRVAL1'], hdulist[self.wcs_idx].header['CRVAL2']]
            self.wcs.wcs.ctype = [hdulist[self.wcs_idx].header['CTYPE1'], hdulist[self.wcs_idx].header['CTYPE2']]
            # self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
            try:
                self.wcs.wcs.cd = [[hdulist[self.wcs_idx].header['CD1_1'], hdulist[self.wcs_idx].header['CD1_2']],
                                   [hdulist[self.wcs_idx].header['CD2_1'], hdulist[self.wcs_idx].header['CD2_2']]]
            except:
                log.info("Missing common CDx_x keys. Assume just CD1_1 and CD2_2")
                try:
                    self.wcs.wcs.cd = [[hdulist[self.wcs_idx].header['CD1_1'], 0],
                                       [0, hdulist[self.wcs_idx].header['CD2_2']]]
                except:
                    log.error("Failed to build WCS manually.", exc_info=True)
                    self.wcs = None
            # self.wcs._naxis1 = hdulist[self.wcs_idx].header['NAXIS1']
            # self.wcs._naxis2 = hdulist[self.wcs_idx].header['NAXIS2']

            self.wcs.pixel_shape = (hdulist[self.wcs_idx].header['NAXIS1'], hdulist[self.wcs_idx].header['NAXIS2'])

            #can't trust this next line ... it fails with SIP sometimes, esp in GOODS N, so just leave as full manual
            #self.wcs = WCS(header=hdulist[self.wcs_idx].header,fobj=hdulist)

        except:
            log.error("Failed to build WCS manually.", exc_info=True)
            self.wcs = None


            # log.error("Failed to build WCS from header. Trying alternate manual load ....", exc_info=True)
            #
            # try: #really manual
            #     self.wcs = WCS(naxis=hdulist[self.wcs_idx].header['NAXIS'])
            #     self.wcs.wcs.crpix = [hdulist[self.wcs_idx].header['CRPIX1'], hdulist[self.wcs_idx].header['CRPIX2']]
            #     self.wcs.wcs.crval = [hdulist[self.wcs_idx].header['CRVAL1'], hdulist[self.wcs_idx].header['CRVAL2']]
            #     self.wcs.wcs.ctype = [hdulist[self.wcs_idx].header['CTYPE1'], hdulist[self.wcs_idx].header['CTYPE2']]
            #     #self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
            #     self.wcs.wcs.cd = [[hdulist[self.wcs_idx].header['CD1_1'], hdulist[self.wcs_idx].header['CD1_2']],
            #                        [hdulist[self.wcs_idx].header['CD2_1'], hdulist[self.wcs_idx].header['CD2_2']]]
            #     # self.wcs._naxis1 = hdulist[self.wcs_idx].header['NAXIS1']
            #     # self.wcs._naxis2 = hdulist[self.wcs_idx].header['NAXIS2']
            #
            #     self.wcs.pixel_shape = (hdulist[self.wcs_idx].header['NAXIS1'],hdulist[self.wcs_idx].header['NAXIS2'])
            # except:
            #     log.error("Failed to build WCS manually.",exc_info=True)
            #     self.wcs = None

        if close:
            hdulist.close()

    def contains_position(self,ra,dec,verify=True):
        if self.footprint is not None: #do fast check first
            if (ra  > np.max(self.footprint[:,0])) or (ra  < np.min(self.footprint[:,0])) or \
               (dec > np.max(self.footprint[:,1])) or (dec < np.min(self.footprint[:,1])):
                #can't be inside the rectangle
                log.debug("position (%f, %f) is not in image max rectangle." % (ra, dec), exc_info=False)
                return False

        close = False
        rc = True
        if self.hdulist is None:
            try:
                log.info("Loading (local scope) fits %s ..." % self.image_location)
                hdulist = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
                close = True
            except:
                log.error("Unable to open science image file: %s" % self.image_location)
                return False
        else:
            log.info("Using outer scope fits %s ..." % self.image_location)
            hdulist = self.hdulist

        #now, it could be, so actually try a cutout to see if it will work
        if rc and verify:
            try:
                cutout = Cutout2D(hdulist[self.wcs_idx].data, SkyCoord(ra, dec, unit="deg", frame=self.frame), (1, 1),
                                  wcs=self.wcs, copy=False,mode="partial",fill_value=0)#,mode='partial')
            except:
                log.debug("position (%f, %f) is not in image." % (ra,dec), exc_info=False)
                rc = False

        if close:
            hdulist.close()

        return rc

    def calc_pixel_size(self,wcs):

        if hasattr(wcs.wcs,'cd'):
            return np.sqrt(wcs.wcs.cd[0, 0] ** 2 + wcs.wcs.cd[0, 1] ** 2) * 3600.0
        elif hasattr(wcs.wcs,'cdelt'): #like Pan-STARRS (assume both directions are the same)
            return wcs.wcs.cdelt[0] * 3600.0
        else: #we have a problem
            log.warning("Warning! Unable to determine pixel scale in science_image::calc_pixel_size. WCS does not have cd or cdelt keywords.")
            return None

    def get_vrange(self,vals,contrast=0.25):
        self.vmin = None
        self.vmax = None

        cpvals = cp.copy(vals)

        #use the interior ~2/3
        try:
            y,x = np.shape(cpvals)
            cpvals = cpvals[int(0.17*y):int(0.83*y),int(0.17*x):int(0.83*x)]
        except:
            pass #just use the whole width

        try:
            zscale = ZScaleInterval(contrast=contrast,krej=2.5) #nsamples=len(vals)
            self.vmin, self.vmax = zscale.get_limits(values=cpvals)
            log.info("Vrange = %f, %f" %(self.vmin,self.vmax))
        except:
            log.info("Exception in science_image::get_vrange:",exc_info =True)

        return self.vmin,self.vmax


    def update_center(self, cutout, radius, play=None):
        """
        Internal (to science_image) use only
        :param cutout:
        :param radius: aperture radius (radius inside which to search for the 2D gaussian centroid ... it can fall
        outside the radius but the region outside the rectangle circumscribed about the radius is masked out)
        :param play: how much play (in arcsec) to allow in x,y ... that is, how far are we allowed to let the center
        shift from the geomtric center of the cutout (if None, use the radius)
        :return:
        """

        cx = -1
        cy = -1
        if play is None:
            play = min(radius,G.NUDGE_MAG_APERTURE_CENTER)

        try:
            cx, cy = cutout.center_cutout

            if play <= 0.0:
                log.debug("No aperture play allowed.")
                return cy,cy

            pix = (radius / self.pixel_size) #arcsec to pixels
            dpix = (play / self.pixel_size)

            pix = max(pix,dpix) #for masking use the larger area of the aperture (pix) or the play radius (dpix)
            mask = np.full(cutout.shape,True)
            mask[int(round(cx-pix-1)):int(round(cx+pix+1)),int(round(cy-pix-1)):int(round(cy+pix+1))] = False

            gx, gy = centroid_2dg(cutout.data,mask=mask)

            dist = -1.0
            try:
                dist = np.sqrt((gx - cx)*(gx - cx) + (gy - cy)*(gy - cy))
                dist = dist * self.pixel_size
            except:
                pass

            if (abs(gx - cx) < dpix) and (abs(gy - cy) < dpix):
                log.info("Centroid (%f,%f) found within acceptable range of geometric center (%f,%f). Dist = %f arcsec, Allowed = %f arcsec"
                         % (gx, gy, cx, cy, dist, play))
            else:
                log.info("Centroid (%f,%f) too far from geometric center (%f,%f). Dist = %f arcsec, Allowed = %f arcsec" %(gx,gy,cx,cy,dist,play))
                gx = cx
                gy = cy

        except:
            gx = cx
            gy = cy

        return gx, gy

    def find_sep_objects(self,cutout,max_dist=None):
        """

        :param cutout:
        :param max_dist: in arcsec (max distance to the ellipse allowed) ... does not apply if INSIDE an ellipse
        :return: ##array of source extractor objects and (selected_idx, flux(cts), fluxerr(cts), flag)
                array of dictionary img_objects translated to ELiXer units, index of selected object or None
        """

        def initialize_dict():
            d = {}
            d['idx'] = None
            d['x'] = None
            d['y'] = None
            d['ra'] = None
            d['dec'] = None
            d['a'] = None
            d['b'] = None
            d['theta'] = None
            d['background'] = None
            d['background_rms'] = None
            d['dist_baryctr'] = None
            d['dist_curve'] = None
            d['flux_cts'] = None
            d['flux_cts_err'] = None
            d['flags'] = None
            d['mag'] = None
            d['mag_faint'] = None
            d['mag_bright'] = None
            d['mag_err'] = None
            d['selected'] = False

            return d

        log.debug("Scanning cutout with source extractor ...")
        #todo:
        # details['area_pix'] = None
        # details['sky_area_pix'] = None
        # details['sky_average'] = None
        # details['sky_counts'] = None

        img_objects = [] #array of dictionaries ... a translation from sep objects to what ELiXer wants
        objects = None
        try:
            if (cutout is None) or (cutout.data is None):
                return img_objects, None

            cx, cy = cutout.center_cutout

            if not G.BIG_ENDIAN:
                data = cutout.data.byteswap().newbyteorder()
            else:
                data = cutout.data
            try:
                bkg = sep.Background(data)
            except Exception as e:
                if type(e) == ValueError:
                    log.debug("sep.Background() value error. May be ENDIAN issue. Swapping...")
                    try:
                        if not G.BIG_ENDIAN:
                            #the inverse of the above assignment (for zipped data the decompression may already handle the
                            #flip so doing it again would have put it in the wrong ENDIAN order
                            data =  cutout.data
                        else:
                            data = cutout.data.byteswap().newbyteorder()

                        bkg = sep.Background(data)

                    except:
                        log.warning("Exception in science_image::find_sep_objects",exc_info=True)
                        return img_objects, None

            data_sub = data - bkg
            data_err = bkg.globalrms #use the background RMS as the error (assume sky dominated)
            objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)

            selected_idx = -1
            inside_objs = []  # (index, dist_to_barycenter)
            outside_objs = []  # (index, dist_to_barycenter, dist_to_curve)

            map_idx = np.full(len(objects),-1) #holds the index of img_objects for each objects entry (if no image object, the value is -1)
            idx = -1
            for obj in objects:
                idx += 1
                # NOTE: #3.* applied for the same reason as above ... a & b are given in kron isophotal diameters
                # so 6a/2 == 3a == radius needed for function
                success, dist2curve, dist2bary, pt = utilities.dist_to_ellipse(cx, cy, obj['x'], obj['y'], 3. * obj['a'],
                                                                         3. * obj['b'], obj['theta'])
                #copy to ELiXer img_objects
                d = initialize_dict()
                d['idx'] = idx
                # convert to image center as 0,0 (needed later in plotting) and to arcsecs
                d['x'] = (obj['x'] - cx) * self.pixel_size  #want as distance in arcsec so pixels * arcsec/pixel
                d['y'] = (obj['y'] - cy) * self.pixel_size
                # the 6.* factor is from source extractor using 6 isophotal diameters
                d['a'] = 6. * obj['a'] * self.pixel_size
                d['b'] = 6. * obj['b'] * self.pixel_size
                d['theta'] = obj['theta']
                d['background'] = bkg.globalback
                d['background_rms'] = bkg.globalrms
                d['dist_baryctr'] = dist2bary * self.pixel_size
                if success:
                    d['dist_curve'] = dist2curve * self.pixel_size
                else:
                    d['dist_curve'] = -1.0

                try:
                    # now, get the flux
                    kronrad, krflag = sep.kron_radius(data_sub, obj['x'], obj['y'],
                                                      obj['a'], obj['b'], obj['theta'], r=6.0)
                    # r=6 == 6 isophotal radii ... source extractor always uses 6
                    # minimum diameter = 3.5 (1.75 radius)
                    radius = kronrad * np.sqrt(obj['a'] * obj['b'])
                    if radius < 1.75:
                        radius = 1.75
                        flux, fluxerr, flag = sep.sum_circle(data_sub, obj['x'], obj['y'],
                                                             radius, subpix=1,err=data_err)
                    else:
                        flux, fluxerr, flag = sep.sum_ellipse(data_sub, obj['x'], obj['y'],
                                                              obj['a'], obj['b'], obj['theta'],
                                                              2.5 * kronrad, subpix=1,err=data_err)
                except Exception as e:
                    try:
                        if e.args[0] == "invalid aperture parameters":
                            pass #do nothing ... not important
                    except:
                        log.warning(f"Exception with source extractor. {e}")
                    continue

                try:  # flux, fluxerr, flag may be ndarrays but of size zero (a bit weird)
                    flux = float(flux)
                    fluxerr = float(fluxerr)
                    flag = int(flag)
                except:
                    log.debug("Exception casting results from sep.sum", exc_info=True)

                d['flux_cts'] = flux
                d['flux_cts_err'] = fluxerr
                d['flags'] = flag
                d['selected'] = False

                #And now get the flux for a fixed 2" diameter aperture
                try:
                    # SEP_FIXED_APERTURE_RADIUS is in arcsec and we need pixels, so divide by pixel_size in arcsec/pixel
                    radius = G.SEP_FIXED_APERTURE_RADIUS / self.pixel_size
                    # now, get the flux
                    flux, fluxerr, flag = sep.sum_circle(data_sub, obj['x'], obj['y'],
                                                         radius, subpix=1,err=data_err)
                except:
                    log.warning("Exception with source extractor",exc_info=True)
                    continue

                try:  # flux, fluxerr, flag may be ndarrays but of size zero (a bit weird)
                    flux = float(flux)
                    fluxerr = float(fluxerr)
                    flag = int(flag)
                except:
                    log.debug("Exception casting results from sep.sum", exc_info=True)

                d['fixed_aper_radius'] = G.SEP_FIXED_APERTURE_RADIUS
                d['fixed_aper_flux_cts'] = flux
                d['fixed_aper_flux_cts_err'] = fluxerr
                d['fixed_aper_flags'] = flag
                d['fixed_aper_mag'] = None
                d['fixed_aper_mag_bright'] = None
                d['fixed_aper_mag_faint'] = None
                d['fixed_aper_mag_err'] = None

                img_objects.append(d)
                map_idx[idx]=len(img_objects) -1

                if success:  # this is outside
                    outside_objs.append((idx, dist2bary, dist2curve))
                elif dist2bary is not None:
                    inside_objs.append((idx, dist2bary))

            # now, choose
            dist_to_curve_aa = 0.0
            if len(inside_objs) != 0:  # there are objects to which the HETDEX point is interior
                # sort by distance to barycenter
                inside_objs = sorted(inside_objs, key=lambda x: x[1])
                selected_idx = inside_objs[0][0]
            elif len(outside_objs) != 0:  # all outside
                # sort by distance to the curve
                outside_objs = sorted(outside_objs, key=lambda x: x[2])
                selected_idx = outside_objs[0][0]
                dist_to_curve_aa = outside_objs[0][2] * self.pixel_size #need to covert to arcsec
            else:  # none found at all, so we would use the old-style cicular aperture
                # todo: aperture stuff
                log.info("No (source extractor) objects found")
                return img_objects, None

            if selected_idx >= 0:
                obj = objects[selected_idx]
            #check max distance
            #todo: incorporate the effective radius of the ellipse? s\t large ellipse gets a little larger max_dist?
            # and a very small ellipse gets (maybe) a little shorted max_dist?
            #effective_radius = 0.5*np.sqrt(d['a']*d['b'])
            if dist_to_curve_aa > max_dist:
                log.info("Dist to nearest source extractor oject (%f) exceeds max allowed (%f)"
                         %(dist_to_curve_aa,max_dist))
                return img_objects, None



            #selected_idx applies to the objects list
            #IT IS NOT NECESSARILY THE SAME SIZE as img_objects

            #mark selected item
            if selected_idx >= 0:
                img_object_idx = map_idx[selected_idx] #xlat to img_objects index
                img_objects[img_object_idx]['selected'] = True
                return img_objects, img_object_idx
            else:
                return img_objects, None

            # # for obj in objects:
            # # now, get the flux
            #
            # kronrad, krflag = sep.kron_radius(data_sub, obj['x'], obj['y'],
            #                                   obj['a'], obj['b'], obj['theta'], r=6.0)
            # # r=6 == 6 isophotal radii ... source extractor always uses 6
            #
            # # minimum diameter = 3.5 (1.75 radius)
            # radius = kronrad * np.sqrt(obj['a'] * obj['b'])
            # if radius < 1.75:
            #     radius = 1.75
            #     flux, fluxerr, flag = sep.sum_circle(data_sub, obj['x'], obj['y'],
            #                                          radius, subpix=1)
            # else:
            #     flux, fluxerr, flag = sep.sum_ellipse(data_sub, obj['x'], obj['y'],
            #                                           obj['a'], obj['b'], obj['theta'],
            #                                           2.5 * kronrad, subpix=1)
            #
            # try: #flux, fluxerr, flag may be ndarrays but of size zero (a bit weird)
            #     flux = float(flux)
            #     fluxerr = float(fluxerr)
            #     flag = int(flag)
            # except:
            #     log.debug("Exception casting results from sep.sum",exc_info=True)
            #
            #
            # for obj in objects:
            #     # convert to image center as 0,0 (needed later in plotting) and to arcsecs
            #     obj['x'] = (obj['x'] - cx) * self.pixel_size
            #     obj['y'] = (obj['y'] - cy) * self.pixel_size
            #     # the 6.* factor is from source extractor using 6 isophotal diameters
            #     obj['a'] = 6. * obj['a'] * self.pixel_size
            #     obj['b'] = 6. * obj['b'] * self.pixel_size

            #return objects,(selected_idx,flux,fluxerr,flag)
        except:
            log.error("Source Extractor call failed.",exc_info=True)

        return img_objects, None


    def get_mask_cutout(self,ra,dec,error,path):
        """

        :param ra:
        :param dec:
        :param error:
        :return:
        """
        cutout = None
        try:
            position = SkyCoord(ra, dec, unit="deg", frame=self.frame)
            pix_window = int(np.ceil(error / self.pixel_size))
            hdulist = fits.open(path, memmap=False, lazy_load_hdus=True)
            cutout = Cutout2D(hdulist[self.wcs_idx].data, position, (pix_window, pix_window),
                                   wcs=self.wcs, copy=False,mode="partial",fill_value=0)
            hdulist.close()
        except:
            log.info(f"Could not get mask cutout", exc_info=True)

        return cutout

    def get_cutout(self,ra,dec,error,window=None,image=None,copy=False,aperture=0,mag_func=None,
                   do_sky_subtract=True,return_details=False,reset_center=True):
        '''ra,dec in decimal degrees. error and window in arcsecs'''
        #error is central box (+/- from ra,dec)
        #window is the size of the entire coutout
        #return a cutout


        details = {'catalog_name':None,'filter_name':None,'ra':None,'dec':None,
                   'radius':None, #effective radius, sqrt(a*b) IF ellipse
                   'mag':None,'mag_err':None, 'mag_bright':None,'mag_faint':None,
                   'pixel_scale':None, #pixel_scale = arcsex/pixel
                   'area_pix':None,'sky_area_pix':None,
                   'aperture_counts':None, #adjusted ... sky subtracted If applicable
                   'sky_counts':None, 'sky_average':None,
                   'aperture_eqw_rest_lya':None,'aperture_eqw_rest_lya_err':None,'aperture_plae':None,
                   'elixer_apertures':None,'elixer_aper_idx':None,
                   'sep_objects':None,'sep_obj_idx':None,
                   'fail_mag_limit':False,'raw_mag':None,'raw_mag_bright':None,'raw_mag_faint':None,'raw_mag_err':None,
                   'exptime':None}


        self.window = None
        if reset_center:
            self.last_x_center = None
            self.last_y_center = None
            self.last_x0_center = None
            self.last_y0_center = None

        cutout = None
        counts = None #raw data counts in aperture
        mag = 10000. #aperture converted to mag_AB

        if (aperture is not None) and (mag_func is not None):
            if aperture == -1:
                #in most cases this should be set to the calling catalog's best starting aperture
                #but as s safety, reset here to at least 0.5"
                if G.DYNAMIC_MAG_APERTURE:
                    aperture = 0.5
                else:
                    aperture= G.FIXED_MAG_APERTURE

            radius = aperture

            # aperture-radius is not allowed to grow past the error-radius in the dynamic case
            if G.DYNAMIC_MAG_APERTURE:
                max_aperture = max(0, error, radius)
            else:
                max_aperture = G.FIXED_MAG_APERTURE

            if max_aperture is None: #can happen if called from catalogs and the defaults get overwritten
                max_aperture = 1.5 #safety check (arcsec)

            sky_outer_radius = max_aperture * 3.0#10. #this is the maximum it can be
            sky_inner_radius = max_aperture * 2.0#5.
            #so ... 4**2 - 2** = 16-4 = 12x sky pixels than aperture pixels at reasonably localized (assuming
            #point sources only)
        else:
            radius = 0.0
            sky_outer_radius = 0.
            sky_inner_radius = 0.

        if aperture is None:
            aperture = 0.


        if (error is None or error == 0) and (window is None or window == 0):
            log.info("inavlid error box and window box")
            if return_details:
                return cutout, counts, mag, radius, details
            else:
                return cutout, counts, mag, radius

        if window is None or window < error:
            window = float(max(2.0*error,5.0)) #should be at least 5 arcsecs

        self.window = window
        #sanity check (shouold be of order few to maybe 100 arcsec, certainly no more than a few arcmin
        #so, greater than that implies a bad degree to arcsec conversion (degree assumed when arcsec passed)

        #e.g. arcsecs are expected here, but from another location degrees are expected. IF the initial paramter was
        #passed as arcsec where degrees were expected, and then converted to arcsecs there would be an extra
        #factor of x3600 and this number would be huge, so, divide by 3600 if that seems to be the case
        #THAT is:  degrees wanted but arcsec passed in -> "degree"x3600 to get to arcsec (so now was arcsec x3600) ->
        #the number is huge now, but really was arcsec to begin with, so /3600
        if window > 1000: #so the max expected is 999 arcsec (given that the window is 3x the request, 333 arcsec is max)
            msg = f"Unexpectedly high cutout size requested ({window} ). Assume arcsec passed instead of degrees " \
                        f"and will convert back to degrees. Changing to ({window/3600.0}) degrees"
            log.warning(msg)
            print(msg)
            window /= 3600.0
            self.window = window

        #data = None
        #pix_size = None
        #wcs = None
        position = None
        if image is None:

            close = False
            if self.hdulist is None:
                try:
                    log.info("Loading (local scope) fits %s ..." % self.image_location)
                    hdulist = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
                    close = True
                except:
                    log.error("Unable to open science image file: %s" % self.image_location)
                    if return_details:
                        return cutout, counts, mag, radius, details
                    else:
                        return cutout, counts, mag, radius
            else:
                log.info("Using outer scope fits %s ..." % self.image_location)
                hdulist = self.hdulist

            #may have been freed to help with wrangler memory
            if hdulist is not None:
                #need to keep memory use down ... the statements below (esp. data = self.fit[0].data)
                #cause a copy into memory ... want to avoid it so, leave the code fragmented a bit as is
               # data = self.hdulist[0].data
               # pix_size = self.pixel_size
               # wcs = self.wcs

                try:
                    position = SkyCoord(ra, dec, unit="deg", frame=self.frame)
                    image = hdulist[self.wcs_idx] #?should auto-close when it falls out of scope

                    #sanity check
                    #x, y = skycoord_to_pixel(position, wcs=self.wcs, mode='all')
                    #x = x * self.pix_size
                    #y = y * self.pix_size

                    pix_window = int(np.ceil(window / self.pixel_size))  # now in pixels
                    log.debug("Collecting cutout size = %d square at RA,Dec = (%f,%f)" %(pix_window,ra,dec))

                    #test for wrangler

                    retries = 0
                    total_sleep = 0.0
                    max_retries = 30
                    while retries < max_retries:
                        try:
                            cutout = Cutout2D(hdulist[self.wcs_idx].data, position, (pix_window, pix_window),
                                              wcs=self.wcs, copy=copy,mode="partial",fill_value=0)

                            image = cutout  # now that we have a cutout, the rest of this func works on it

                            #get a larger cutout to accomodate the sky subtraction outer radius
                            if sky_outer_radius > 0:
                                sky_annulus = SkyCircularAnnulus(position, r_in=sky_inner_radius * ap_units.arcsec,
                                                    r_out=sky_outer_radius * ap_units.arcsec).to_pixel(self.wcs)

                                sky_pix_window = 2*(int(sky_annulus.r_out)+ 1)

                                if sky_pix_window > pix_window:
                                    #if the sky window is larger than the original cutout, get a larger cutout for the sky
                                    #else, just use the original cutout (e.g. if the window is large, but the aperture is small
                                    try:
                                        sky_image = Cutout2D(hdulist[self.wcs_idx].data, position, (sky_pix_window, sky_pix_window),
                                                      wcs=self.wcs, copy=False,mode="partial",fill_value=0) #don't need a copy, will not persist beyond
                                                                                #this call
                                    except:
                                        log.warning("Exception attempting to get larger sky_image. science_image::get_cutout",
                                                    exc_info=True)
                                        sky_image = image
                                else:
                                    sky_image = image
                            else:
                                sky_image = image

                            break

                        except (MemoryError, mmap.error):

                            if close: #we are using a local hdulist
                                del cutout
                                cutout = None
                                if close:
                                    try:
                                        hdulist.close()
                                    except:
                                        log.warning("Exception attempting to close hdulist. science_image::get_cutout",
                                                    exc_info=True)

                                retries += 1
                                if retries >= max_retries:
                                    break

                                gc.collect() #try to force an immediate clean up

                                t2sleep = np.random.random_integers(0,5000)/1000. #sleep up to 5 sec
                                total_sleep += t2sleep
                                log.info("+++++ Memory issue? Random sleep (%d / %d) and retry (%f)s" %
                                         (retries,max_retries,t2sleep),exc_info=True)
                                sleep(t2sleep)

                                try:
                                    log.info("Loading (local scope) fits %s ..." % self.image_location)
                                    hdulist = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
                                    close = True
                                except:
                                    log.error("Unable to open science image file: %s" % self.image_location)
                                    if return_details:
                                        return cutout, counts, mag, radius, details
                                    else:
                                        return cutout, counts, mag, radius
                            else:
                                log.error("Unable to open science image file: %s" % self.image_location)
                                retries = max_retries

                        except NoOverlapError:
                            log.info("Unable to load cutout (NoOverlapError).", exc_info=False)
                            retries = max_retries
                        except Exception as ex:
                            if "Arrays do not overlap" in str(ex):
                                log.info("Unable to load cutout (NoOverlapError")
                            else:
                                log.error("Exception. Unable to load cutout.",exc_info=True)
                            retries = max_retries



                    if close:
                        try:
                            hdulist.close()
                            gc.collect()  # try to force an immediate clean up
                        except:
                            log.warning("Exception attempting to close hdulist. science_image::get_cutout", exc_info=True)

                    if retries >= max_retries:
                        log.info("+++++ giving up (%d,%f) ...." % (retries,total_sleep))
                        if return_details:
                            return None, counts, mag, radius, details
                        else:
                            return None, counts, mag, radius
                    elif retries > 0:
                        log.info("+++++ it worked (%d,%f) ...." % (retries,total_sleep))



                    #before = float(G.HPY.heap().size) /(2**20)


                    #potential "cheat" for wrangler
                    # tempfits = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
                    # cutout = Cutout2D(tempfits[self.wcs_idx].data, position, (pix_window, pix_window), wcs=self.wcs,
                    #                   copy=copy)
                    # tempfits.close()
                    # del tempfits
                    #


                    #attempt to work arond wrangler memory problems
                    #self.hdulist.close()
                    #self.hdulist = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)

                    #cutout = Cutout2D(hdulist[self.wcs_idx].data, position, (pix_window, pix_window), wcs=self.wcs,
                    #                  copy=copy)

                    #image = cutout #now that we have a cutout, the rest of this func works on it

                    #del self.hdulist[self.wcs_idx].data
                    #self.hdulist[self.wcs_idx].data = []




                    #self.cycle_fits()
                    #self.hdulist.close()

                    #after = float(G.HPY.heap().size) /(2**20)

                    #msg = "+++++ heap before (%0.2f), after (%0.2f) MB" %(before,after)
                    #print(msg)
                    #log.debug(msg)
                    self.get_vrange(cutout.data)
                except NoOverlapError:
                    log.info("Error (possible NoOverlapError) in science_image::get_cutout(). *** Did more than one catalog match the coordinates? ***"
                             "Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                    print("Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                    if return_details:
                        return cutout, counts, mag, radius, details
                    else:
                        return cutout, counts, mag, radius
                except:
                    #after = float(G.HPY.heap().size) /(2**20)

                    #msg = "+++++ heap before (%0.2f), after (%0.2f) MB (exception case)" %(before,after)
                    #print(msg)
                    #log.debug(msg)

                    log.error("Exception in science_image::get_cutout (%s):" %self.image_location, exc_info=True)
                    if return_details:
                        return cutout, counts, mag, radius, details
                    else:
                        return cutout, counts, mag, radius

                if not (self.contains_position(ra,dec)):
                    log.info("science image (%s) does not contain requested position: RA=%f , Dec=%f"
                             %(self.image_location,ra,dec))
                    if return_details:
                        return None, counts, mag, radius, details
                    else:
                        return None, counts, mag, radius
            else:
                log.error("No fits or passed image from which to make cutout.")
                if return_details:
                    return cutout, counts, mag, radius, details
                else:
                    return cutout, counts, mag, radius
        else:
            #data = image.data
            #pix_size = self.calc_pixel_size(image.wcs)
            #wcs = image.wcs
            try:
                position = SkyCoord(ra, dec, unit="deg")#, frame='fk5')
                #self.pixel_size = self.calc_pixel_size(image.wcs)
                pix_window = float(window) / self.calc_pixel_size(image.wcs)  # now in pixels
                cutout = Cutout2D(image.data, position, (pix_window, pix_window), wcs=image.wcs, copy=copy,mode="partial",fill_value=0)
                self.get_vrange(cutout.data)
            except NoOverlapError:
                log.info("Warning (NoOverlapError) in science_image::get_cutout(). "
                        "Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                #print("Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                if cutout is not None:
                    try:
                        self.get_vrange(cutout.data)
                    except:
                        pass
            except:
                log.error("Exception in science_image::get_cutout ():" , exc_info=True)
                if return_details:
                    return cutout, counts, mag, radius, details
                else:
                    return cutout, counts, mag, radius


        details['pixel_scale'] = self.pixel_size
        details['exptime'] = self.exptime

        #We have the cutout info, now get aperture photometry

        if is_cutout_empty(cutout):
            if (G.ALLOW_EMPTY_IMAGE):
                pass
                # some cases seem to create problems, but forcing different values
                # or forcing all the same or all zero does not seem to matter
                # for matplotlib plot to PDF
                # it is an empty image anyway, and under normal configuration
                # this will get the default empty plot
                # if np.min(cutout.data) == np.max(cutout.data):
                #     cutout.data *= 0. #set all to exactly zero
            else:
                print(
                    f"Cutout is empty or simple gradient. Will deliberately fail cutout request. {self.image_location}")
                log.warning(f"Cutout is empty or simple gradient. Will deliberately fail cutout request. {self.image_location}")
                if return_details:
                    details['elixer_apertures'] = []
                    details['elixer_aper_idx'] = None
                    return None, 0, 99.99, 0, details
                else:
                    return None, 0, 99.99, 0

        #put down aperture on cutout at RA,Dec and get magnitude
        if (position is not None) and (cutout is not None) and (image is not None) \
                and (mag_func is not None) and (aperture > 0):

            #if source extractor works, use it else, proceed as before with circular aperture photometry

            return_mag = None #the configured magnitude source (ELiXer Ap at a radius or SEP, etc)
            return_radius = None #the effective radius for that magnitude

            if G.USE_SOURCE_EXTRACTOR:
                source_objects,selected_obj_idx = self.find_sep_objects(cutout,G.NUDGE_SEP_MAX_DIST)

                if (source_objects is not None) and (len(source_objects) > 0):

                    #get the mag for all
                    for sobj in source_objects:

                        #start with the the fixed aperture (since we re-use the varaiables: counts, etc later
                        #from the elliptical apertures
                        if 'fixed_aper_flux_cts' in sobj.keys():
                            counts = sobj['fixed_aper_flux_cts'] #sep_info[1]
                            count_err = sobj['fixed_aper_flux_cts_err'] #sep_info[2]

                            mag, mag_faint, mag_bright,mag_err = None, None, None, None
                            try:
                                mag = mag_func(counts, cutout, self.headers)
                                mag_faint = mag_func(counts-count_err, cutout, self.headers)
                                mag_bright = mag_func(counts+count_err, cutout, self.headers)
                                if mag_faint < 99:
                                    mag_err = max(mag_faint - mag, mag - mag_bright)
                                else:
                                    mag_err = mag - mag_bright
                            except:
                                log.error("Exception calling mag_func.",exc_info=True)

                            sobj['fixed_aper_mag'] = mag
                            sobj['fixed_aper_mag_faint'] = mag_faint
                            sobj['fixed_aper_mag_bright'] = mag_bright
                            sobj['fixed_aper_mag_err'] = mag_err


                        #and now the elliptical, fitted aperture
                        #selected_obj_idx = sep_info[0]
                        counts = sobj['flux_cts'] #sep_info[1]
                        count_err = sobj['flux_cts_err'] #sep_info[2]

                        mag, mag_faint, mag_bright,mag_err = None, None, None, None
                        try:
                            mag = mag_func(counts, cutout, self.headers)
                            mag_faint = mag_func(counts-count_err, cutout, self.headers)
                            mag_bright = mag_func(counts+count_err, cutout, self.headers)
                            if mag_faint < 99:
                                mag_err = max(mag_faint - mag, mag - mag_bright)
                            else:
                                mag_err = mag - mag_bright
                        except:
                            log.error("Exception calling mag_func.",exc_info=True)

                        sobj['mag'] = mag
                        sobj['mag_faint'] = mag_faint
                        sobj['mag_bright'] = mag_bright
                        sobj['mag_err'] = mag_err





                        try:
                            #this assumes lower-left is 0,0 but the object x,y uses center as 0,0
                            #sobj['x'] and y ARE IN ARCSEC ... need to be in pixels for this cal
                            sc = wcs_utils.pixel_to_skycoord(sobj['x']/self.pixel_size + cutout.center_cutout[0],
                                                             sobj['y']/self.pixel_size + cutout.center_cutout[1],
                                                             cutout.wcs, origin=0)
                            sobj['ra'] = sc.ra.value
                            sobj['dec'] = sc.dec.value
                        except:
                            log.debug("Exception converting source extrator x,y to RA, Dec", exc_info=True)

                        if sobj['selected']:
                            # the shift in AA from center
                            self.last_x0_center = sobj['x'] #* self.pixel_size
                            self.last_y0_center = sobj['y'] #* self.pixel_size
                            # the shift in AA from lower left
                            self.last_x_center = (sobj['x']/self.pixel_size  + cutout.center_cutout[0]) * self.pixel_size
                            self.last_y_center = (sobj['y']/self.pixel_size  + cutout.center_cutout[1]) * self.pixel_size

                            #details['radius'] = radius
                            try:
                                details['radius'] = 0.5*np.sqrt(sobj['a']*sobj['b']) #0.5 * because a,b are diameters, not radii
                            except:
                                details['radius'] = -1.0

                            details['aperture_counts'] = counts #Already Sky subtracted
                            #todo: modify find_sep_objects to get this extra info
                            details['area_pix'] = None
                            details['sky_area_pix'] = None
                            details['sky_average'] = None
                            details['sky_counts'] = None
                            details['mag'] = mag
                            details['mag_err'] = mag_err
                            details['mag_bright'] = mag_bright
                            details['mag_faint'] = mag_faint
                            details['ra'] = sobj['ra']
                            details['dec'] = sobj['dec']

                            #matplotlib plotting later needs these in sky units (arcsec) not pixels
                    details['sep_objects']  = source_objects
                    details['sep_obj_idx'] = selected_obj_idx

                    if selected_obj_idx is not None:
                        return_counts = details['aperture_counts']
                        return_mag = details['sep_objects'][selected_obj_idx]['mag']
                        return_radius = np.sqrt(details['sep_objects'][selected_obj_idx]['a']*
                                               details['sep_objects'][selected_obj_idx]['b'])*0.5

                    # todo: DO NOT RETURN HERE, we ALSO want to get the ELiXer aperture photometry
                    # if selected_obj_idx is not None:
                    #     if return_details:
                    #         return cutout, counts, return_mag, return_radius, details
                    #     else:
                    #         return cutout, counts, return_mag, return_radius



            cutout, counts, mag, radius, details = self.get_circular_aperture_photometry(cutout,ra,dec,error,mag_func,
                                                    position,image,do_sky_subtract,sky_image,
                                                    sky_inner_radius,sky_outer_radius,aperture,
                                                    details,return_details,check_cutout_empty=False)
                                                     #cutout empty already checked above

            #now which one (counts, mag, radius) to use???
            #todo: do we need to specify if it was from elixer_apertures or sep_objects?

            if return_mag is None: #it was not set by SEP objects
                log.info("Using ELiXer circular aperture as reported aperture.")
                return_mag = mag
                return_radius = radius
                return_counts = counts

                try:
                    if details['elixer_aper_idx'] is None: #not set above? probably mag = 99.9
                        if (details['elixer_apertures']) is not None and len(details['elixer_apertures']) > 0:
                            details['elixer_aper_idx'] = len(details['elixer_apertures'])-1

                    if details['elixer_aper_idx'] is not None:
                        ap = details['elixer_apertures'][details['elixer_aper_idx']]
                        details['radius'] = ap['radius']
                        details['aperture_counts'] = ap['aperture_counts']
                        details['area_pix'] = ap['area_pix']
                        details['sky_area_pix'] = ap['sky_area_pix']
                        details['sky_average'] = ap['sky_average']
                        details['sky_counts'] = ap['sky_counts']
                        details['mag'] = ap['mag']
                        details['mag_err'] = ap['mag_err']
                        details['mag_bright'] = ap['mag_bright']
                        details['mag_faint'] = ap['mag_faint']
                        details['ra'] = ap['ra']
                        details['dec'] = ap['dec']
                except:
                    log.info("Exception.",exc_info=True)

            else: #the SEP object will be the "aperture" chosen ... so clear the selected flag from the circular aperture
                log.info("Using SEP selected object as reported aperture.")
                try:
                    details['elixer_apertures'][details['elixer_aper_idx']]['selected'] = False
                except:
                    pass

            if return_details:
                return cutout, return_counts, return_mag, return_radius, details
            else:
                return cutout, return_counts, return_mag, return_radius

        # DEAD CODE (1)

        if return_details:
            return cutout, counts, mag, radius, details
        else:
            return cutout, counts, mag, radius

    def get_circular_aperture_photometry(self,cutout,ra,dec,error,mag_func,position,image,do_sky_subtract,
                                         sky_image,sky_inner_radius,sky_outer_radius,aperture,
                                         details,return_details,check_cutout_empty=True):
        """

        :param position:
        :param image:
        :param sky_image:
        :param radius: (set already from calling func, get_cutout
        :param check_cutout_empty (can set to False IF already checked in caller)
        :return:

        radius, mag, counts (that will be vs return_mag, radius, counts
        """
        def initialize_dict():
            return {'idx':None,'ra':None,'dec':None,'radius':None,'area_pix':None,'aperture_counts':None,
                           'sky_area_pix':None,'sky_average':None,'sky_counts':None,'sky_err':None,
                           'mag':None,'mag_err':None,'mag_bright':None,'mag_faint':None,'selected':None }

        x_center, y_center = self.update_center(cutout, aperture, play=G.NUDGE_MAG_APERTURE_CENTER)
        self.last_x_center = x_center * self.pixel_size
        self.last_y_center = y_center * self.pixel_size
        self.last_x0_center = (x_center - cutout.center_cutout[0]) * self.pixel_size  # the shift in AA from center
        self.last_y0_center = (y_center - cutout.center_cutout[1]) * self.pixel_size
        source_aperture_area = 0.0


        elixer_aperture_list = []
        elixer_aper_idx = None #the selected index


        if check_cutout_empty and is_cutout_empty(cutout):
            if (G.ALLOW_EMPTY_IMAGE):
                pass
                # some cases seem to create problems, but forcing different values
                # or forcing all the same or all zero does not seem to matter
                # for matplotlib plot to PDF
                # it is an empty image anyway, and under normal configuration
                # this will get the default empty plot
                # if np.min(cutout.data) == np.max(cutout.data):
                #     cutout.data *= 0. #set all to exactly zero
            else:
                log.info("Cutout is empty or simple gradient. Will deliberately fail cutout request.")
                if return_details:
                    details['elixer_apertures'] = elixer_aperture_list
                    details['elixer_aper_idx'] = elixer_aper_idx
                    return None, 0, 99.99, 0, details
                else:
                    return None, 0, 99.99, 0


        #todo: for now, there is only ONE, but this is the basic framework to allow multiples (esp for dynamic aperture)

        try:
            sky_coord_center = wcs_utils.pixel_to_skycoord(x_center, y_center, cutout.wcs, origin=0)
        except:
            log.warning("Exception! getting aperture RA,Dec.", exc_info=True)

        try:
            distance_to_center = utilities.angular_distance(ra,dec,sky_coord_center.ra.value,sky_coord_center.dec.value)
        except:
            try:
                distance_to_center = np.sqrt(self.last_x0_center*self.last_x0_center + self.last_y0_center*self.last_y0_center)
            except:
                distance_to_center = 0

        if G.DYNAMIC_MAG_APERTURE:
            if aperture and (aperture > 0.0):
                radius = aperture
            elif G.MIN_DYNAMIC_MAG_RADIUS is not None:
                radius = G.MIN_DYNAMIC_MAG_RADIUS
            else:
                radius = 1.5  # failsafe

            radius = min(radius, G.MAX_DYNAMIC_MAG_APERTURE, error)
            step = 0.1

            sky_avg, sky_err, sky_pix, sd_sky_pix = self.get_local_sky(sky_image, position,
                                                                       G.MAX_DYNAMIC_MAG_APERTURE * 2.0,
                                                                       G.MAX_DYNAMIC_MAG_APERTURE * 4.0)

            log.debug("science_image::get_cutout() mag aperture radius step size = %f" % step)

            max_radius = radius
            max_bright = 99.9
            max_counts = 0
            max_area_pix = 0
            expected_count_growth = 0.0  # number of new pixels * sky

            mag_list = [99.9]
            rad_list = [0.0]
            elixer_aper_idx = -1
            while radius <= min(error, G.MAX_DYNAMIC_MAG_APERTURE):
                elixer_aper_idx += 1
                source_aperture_area = 0.
                try:
                    # use the cutout first if possible (much smaller than image and faster)
                    # note: net difference is less than 0.1 mag at 0.5" and less than 0.01 at 1.5"
                    # I believe the source of the difference is the center position ... in the pixel mode
                    # we center on the center pixel as reported by the center of the cutout. In sky mode
                    # we center on an RA, Dec position and the slight difference between the two centers
                    # (itself, less than one pixel) yields slightly different counts as the area covered is slightly
                    # offset (fraction of a pixel or, typically, small fractions of an arcsec)
                    try:
                        pix_aperture = CircularAperture((x_center, y_center), r=radius / self.pixel_size)
                        phot_table = aperture_photometry(cutout.data, pix_aperture, method=PIXEL_APERTURE_METHOD)
                        counts = phot_table['aperture_sum'][0]
                        try:
                            source_aperture_area = pix_aperture.area()  # older version(s) of photutils
                        except:
                            source_aperture_area = pix_aperture.area
                    except:
                        log.info("Pixel based aperture photometry failed. Attemping sky based ... ", exc_info=True)

                        try:
                            sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
                            phot_table = aperture_photometry(image, sky_aperture, method=PIXEL_APERTURE_METHOD)
                            counts = phot_table['aperture_sum'][0]
                            try:
                                source_aperture_area = sky_aperture.area()  # older version(s) of photutils
                            except:
                                source_aperture_area = sky_aperture.area
                        except:
                            log.info("Sky based aperture photometry failed. Will skip aperture photometery.",
                                     exc_info=True)
                            break

                    try:
                        delta_pix = source_aperture_area - max_area_pix
                        expected_count_growth = delta_pix * sky_avg
                        # assume sd as error and add in quadrature
                        expected_count_growth += np.sqrt(delta_pix) * sky_err
                    except:
                        expected_count_growth = 0.0

                    if not isinstance(counts, float):
                        log.info("Attempting to strip units from counts (%s) ..." % (type(counts)))
                        try:
                            counts = counts.value

                            if not isinstance(counts, float):
                                log.warning(
                                    "Cannot cast counts as float. Will not attempt aperture magnitude calculation")
                                break

                        except:
                            log.info("Failed to strip units. Cannot cast to float. "
                                     "Will not attempt aperture magnitude calculation", exc_info=True)
                            break

                    mag = mag_func(counts, cutout, self.headers)

                    log.info(
                        "Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %g mag = %g dmag = %g"
                        % (radius, ra, dec, counts, mag, mag - mag_list[-1]))

                    mag_list.append(mag)
                    rad_list.append(radius)

                    # todo: if mag == 99.9 at radius == 0.5" maybe just stop or limit to 1" total?
                    # todo: don't want to catch somthing on the edge and then expand
                    # todo: plus our astrometry accuracy is ~ 0.5"

                    # intermediate steps
                    elixer_aperture = initialize_dict()
                    elixer_aperture['idx'] = elixer_aper_idx
                    elixer_aperture['selected'] = False #will be set to True later if appropriate
                    elixer_aperture['ra'] = sky_coord_center.ra.value
                    elixer_aperture['dec'] = sky_coord_center.dec.value
                    elixer_aperture['radius'] = radius
                    elixer_aperture['mag'] = mag
                    elixer_aperture['aperture_counts'] = counts
                    elixer_aperture['area_pix'] = source_aperture_area
                    elixer_aperture['dist_to_center'] = distance_to_center
                    elixer_aperture_list.append(elixer_aperture)
                    #don't set the elixer_aper_idx yet, this one might not be accepted

                    if mag < 99:
                        # if now growing fainter OR the brightness increase is small, we are done
                        if expected_count_growth is not None:
                            if (mag > max_bright) or (counts - max_counts) < expected_count_growth:
                                break
                        elif (mag > max_bright) or (abs(mag - max_bright) < 0.01):  # < 0.0005):
                            break

                    # elif (counts <= 0) and (radius < G.MAX_DYNAMIC_MAG_APERTURE):
                    #     pass #ignore this and keep going
                    elif (radius >= aperture) or (abs(radius - aperture) < 1e-5) or (
                            radius > G.MAX_DYNAMIC_MAG_APERTURE):
                        # weirdness in floats, difference when "==" is non-zero ~ 1e-16
                        if max_bright > mag:
                            max_bright = mag
                            max_counts = counts
                            max_radius = radius
                            max_area_pix = source_aperture_area
                        break

                    max_bright = mag
                    max_counts = counts
                    max_radius = radius
                    max_area_pix = source_aperture_area
                    details['elixer_aper_idx'] = elixer_aperture['idx'] #so far, this one is still okay
                except:
                    log.error("Exception in science_image::get_cutout () using dynamic aperture", exc_info=True)

                radius += step
                # end while loop

            if max_bright > 99 and len(mag_list) == 2:
                #this only had one aperture, but the max did not get updated as was an immediate exit
                try:
                    mag = elixer_aperture_list[-1]['mag']
                    counts = elixer_aperture_list[-1]['aperture_counts']
                    radius = elixer_aperture_list[-1]['radius']
                    source_aperture_area = elixer_aperture_list[-1]['area_pix']
                    details['elixer_aper_idx']=elixer_aperture_list[-1]['idx']

                except:
                    mag = max_bright
                    counts = max_counts
                    radius = max_radius
                    source_aperture_area = max_area_pix
            else:
                mag = max_bright
                counts = max_counts
                radius = max_radius
                source_aperture_area = max_area_pix
            # selected from HERE, but might not be the final reported aperture from the outer caller
            try:
                elixer_aperture_list[details['elixer_aper_idx']]['selected'] = True
            except:
                pass
            details['elixer_apertures'] = elixer_aperture_list
            #elixer_aper_idx is already set

        else: #NOT allowing Dynamic Aperture growth
            try:
                if aperture and (aperture > 0.0):
                    radius = aperture
                elif G.FIXED_MAG_APERTURE is not None:
                    radius = G.FIXED_MAG_APERTURE
                else:
                    radius = 1.5  # failsafe

                elixer_aperture = initialize_dict()
                elixer_aperture['idx'] = 0
                elixer_aperture['selected'] = True
                elixer_aperture['ra'] = sky_coord_center.ra.value
                elixer_aperture['dec'] = sky_coord_center.dec.value
                elixer_aperture['radius'] = radius

                source_aperture_area = 0.

                try:
                    pix_aperture = CircularAperture((x_center, y_center), r=radius / self.pixel_size)
                    phot_table = aperture_photometry(cutout.data, pix_aperture, method=PIXEL_APERTURE_METHOD)
                    try:
                        source_aperture_area = pix_aperture.area()
                    except:
                        source_aperture_area = pix_aperture.area
                except:
                    log.info("Pixel based aperture photometry failed. Attemping sky based ... ", exc_info=True)
                    # note: if we do this, photutils loads the entire fits image and that can be costly
                    # in terms of time and memory
                    try:
                        sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
                        phot_table = aperture_photometry(image, sky_aperture, method=PIXEL_APERTURE_METHOD)
                        try:
                            source_aperture_area = sky_aperture.area()
                        except:
                            source_aperture_area = sky_aperture.area
                    except:
                        log.info("Sky based aperture photometry failed. Will skip aperture photometery.",
                                 exc_info=True)
                        if return_details:
                            details['elixer_apertures'] = elixer_aperture_list
                            details['elixer_aper_idx'] = elixer_aperture['idx']
                            return cutout, counts, mag, radius, details
                        else:
                            return cutout, counts, mag, radius

                counts = phot_table['aperture_sum'][0]

                if not isinstance(counts, float):
                    log.info("Attempting to strip units from counts (%s) ..." % (type(counts)))
                    try:
                        counts = counts.value
                    except:
                        log.info("Failed to strip units. Cannot cast to float. "
                                 "Will not attempt aperture magnitude calculation", exc_info=True)

                mag = mag_func(counts, cutout, self.headers)

                elixer_aperture['mag'] = mag
                elixer_aperture['aperture_counts'] = counts
                elixer_aperture['area_pix'] = source_aperture_area
                details['elixer_aper_idx'] = elixer_aperture['idx']
                elixer_aperture_list.append(elixer_aperture)
                details['elixer_apertures'] = elixer_aperture_list

                log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %s Mag_AB = %g"
                         % (radius, ra, dec, str(counts), mag))
                # print ("Counts = %s Mag %f" %(str(counts),mag))
            except:
                log.error("Exception in science_image::get_cutout () using aperture", exc_info=True)


        #if we have a magnitude and it is fainter than a minimum, subtract the sky from a surrounding annulus
        #s|t we have ~ 3x pixels in the sky annulus as in the source aperture, so 2x the radius
        if do_sky_subtract and (mag > G.SKY_ANNULUS_MIN_MAG): #and (mag < 99)
            #do it anyway, even if mag = 99.9 as the counts might be negative, but the local sky might be negative too
            try:

                #todo: note in photutils, pixel x,y is the CENTER of the pixel and [0,0] is the center of the
                #todo: lower-left pixel
                #it should not really matter if the pixel position is the center of the pixel or at a corner
                #given the typically small pixels the set of pixels will not change much one way or the other
                #and we are going to take a median average (and not use partial pixels), at least not in this 1st
                #revision

                # to take a median value and subtract off the counts from the original aperture
                # yes, against the new cutout (which is just a super set of the smaller cutout
                #source_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec).to_pixel(sky_image.wcs)

                sky_outer_radius = radius * 3.0  # 10. #this is the maximum it can be
                sky_inner_radius = radius * 2.0  # 5.
                sky_annulus = SkyCircularAnnulus(position, r_in=sky_inner_radius * ap_units.arcsec,
                                                 r_out=sky_outer_radius * ap_units.arcsec).to_pixel(sky_image.wcs)

                #made an annulus in pixel, set to a "mask" where 0 = pixel not in annulus, 1 = pixel in annulus
                try:
                    sky_mask = sky_annulus.to_mask(method='center')[0]
                except:
                    sky_mask = sky_annulus.to_mask(method='center')


                sky_mask = np.array(sky_mask)[0:sky_image.data.shape[0],0:sky_image.data.shape[1]]
                #select all pixels from the cutout that are in the annulus
                #this sometimes gets off by 1-pixel, so trim off if out of range of the array
                # search = np.where(sky_mask.data > 0)

                #the sky_mask and the sky_image share a common origin (0,0) so they have the same indexing
                #the sky_mask may be larger than the sky_image, so turn the mask into a 2D array (so we can slice
                #and slice away the mask outside the sky_image)

                search = np.where(sky_mask > 0)
                annulus_data_1d = sky_image.data[search]
                sky_pix = len(annulus_data_1d)

                #and take the median average from a 3-sigma clip
                #if going to sigma clip, probably should iterate (clip, cut out pixels | | > clip, clip again ... ) until stable?
                # bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d,sigma=3.0) #this is the average sky per pixel
                # sky_avg = bkg_median
                # # #bkg_median * source_aperture.area() #total (sky) to subtract is averager sky per pix * number pix in source aperture
                # log.debug("Sigma clipping sky pixels (%f). Inner (%f), outer (%f). Sum (%f) Avg (%f) "
                #           % (sky_pix, sky_inner_radius, sky_outer_radius, np.sum(annulus_data_1d), sky_avg))
                #
                # annulus_data_1d = annulus_data_1d[np.where(abs(annulus_data_1d-bkg_median) < 3.0*bkg_std)]

                #replace with biweight "average"
                bw_cen = biweight.biweight_location(annulus_data_1d)
                bw_scale = biweight.biweight_scale(annulus_data_1d)
                sky_avg = bw_cen
                #set sky N to number of pixels within 1 sd (or bw_scale, in this case)
                #not strictly correct, but we are not including other error sources so this
                #will nudge toward a larger error (which would be more appropriate)
                N = len(np.where(abs(annulus_data_1d-bw_cen) < bw_scale)[0])
                sky_err = bw_scale/np.sqrt(N)

                log.debug("Sky pixels (%f). Inner (%f), outer (%f). Sum (%f) Avg (%f) "
                          %(sky_pix,sky_inner_radius,sky_outer_radius,np.sum(annulus_data_1d),sky_avg))

                #todo: here loop over all elixer_apertures (may be only one if not Dynamic, or may be many)
                for elixer_aperture in elixer_aperture_list:
                    source_aperture_area = elixer_aperture['area_pix']
                    counts = elixer_aperture['aperture_counts']
                    sky_cts = sky_avg * source_aperture_area
                    base_counts = counts
                    counts -= sky_cts
                    #re-compute the magnitude
                    base_mag = mag
                    sky_mag = mag_func(counts,cutout,self.headers)

                    sky_mag_bright = mag_func(base_counts - (sky_avg - sky_err) * source_aperture_area,
                             cutout, self.headers)

                    sky_mag_faint = mag_func(base_counts - (sky_avg + sky_err) * source_aperture_area,
                             cutout, self.headers)

                    if sky_mag_faint < 99:
                        mag_err = max((sky_mag_faint-sky_mag),(sky_mag-sky_mag_bright))
                    elif sky_mag < 99:
                        mag_err = sky_mag-sky_mag_bright
                    elif sky_mag_bright < 99:
                        mag_err = abs(base_mag-sky_mag_bright)
                    else: #can't get mag on the sky only ... below limit
                        #todo: this should be related to the mag limit of the imaging
                        mag_err = 0.0 #something kind of reasonable, 100x in flux?

                    if not (sky_mag < 99):
                        if sky_mag_bright < 99:
                            sky_mag = sky_mag_bright
                        elif sky_mag_faint < 99: #odd case if sky_avg is negative (and sky_err is positive)
                            sky_mag = sky_mag_faint

                    #mag should now be fainter (usually ... could have slightly negative sky?)
                    #the photometry should have pretty good sky subtration ... but what if we are on a faint object
                    #that is near large object ... could be we don't get enough sky pixels or the average is skewed high
                    #so if we make much of a change, at least log a warning

                    mag = sky_mag #1.11.0a9 2021-04-13 ... do it anyway, even if we go negative or below limit
                                  # it can mean there is something wrong
                    if (base_mag < 99.9) and (abs(sky_mag - base_mag) > G.MAX_SKY_SUBTRACT_MAG):
                       # print("Warning! Unexepectedly large sky subtraction impact to magnitude: %0.2f to %0.2f at (%f,%f)"
                       #             %(base_mag,sky_mag,ra,dec))
                        log.warning("Warning! Unexepectedly large sky subtraction impact to magnitude: %0.2f to %0.2f at (%f,%f)"
                                    %(base_mag,sky_mag,ra,dec))
                        elixer_aperture['warn_sky'] = 1
                    # elif sky_mag < 99.9:
                    #     mag = sky_mag
                    #else the mag remains unchanged

                    log.info("Sky subtracted imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Sky = (%f/pix %f tot). Counts = %s Mag_AB = %g"
                             % (elixer_aperture['radius'],ra,dec,sky_avg, sky_cts,str(counts),mag))
                    #print ("Counts = %s Mag %f" %(str(counts),mag))

                    elixer_aperture['aperture_counts'] = counts #update to sky subtracted counts to be consistent
                    elixer_aperture['sky_area_pix'] = sky_pix
                    elixer_aperture['sky_average'] = sky_avg
                    elixer_aperture['sky_counts'] = sky_cts #sky_average * aperture area (in pixels)
                    elixer_aperture['sky_err'] = sky_err
                    elixer_aperture['mag'] = mag
                    elixer_aperture['mag_err'] = mag_err
                    elixer_aperture['mag_bright'] = min(sky_mag_bright,mag)
                    elixer_aperture['mag_faint'] = max(sky_mag_faint,mag)

            except:
                #print("Sky Mask Problem ....")
                log.error("Exception in science_image::get_cutout () figuring sky subtraction aperture", exc_info=True)

        if return_details: #elixer_aperture in details is ALREADY set
            return cutout, counts, mag, radius, details
        else:
            return cutout, counts, mag, radius
    #end get_circular_aperture_photometry

    def get_local_sky(self,image,position,inner_radius,outer_radius):

        try:
            # todo: note in photutils, pixel x,y is the CENTER of the pixel and [0,0] is the center of the
            # todo: lower-left pixel
            # it should not really matter if the pixel position is the center of the pixel or at a corner
            # given the typically small pixels the set of pixels will not change much one way or the other
            # and we are going to take a median average (and not use partial pixels), at least not in this 1st
            # revision

            # to take a median value and subtract off the counts from the original aperture
            # yes, against the new cutout (which is just a super set of the smaller cutout
            # source_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec).to_pixel(sky_image.wcs)

            sky_annulus = SkyCircularAnnulus(position, r_in=inner_radius * ap_units.arcsec,
                                             r_out=outer_radius * ap_units.arcsec).to_pixel(image.wcs)

            # made an annulus in pixel, set to a "mask" where 0 = pixel not in annulus, 1 = pixel in annulus
            try:
                sky_mask = sky_annulus.to_mask(method='center')[0]
            except:
                sky_mask = sky_annulus.to_mask(method='center')

            sky_mask = np.array(sky_mask)[0:image.data.shape[0], 0:image.data.shape[1]]
            # select all pixels from the cutout that are in the annulus
            # this sometimes gets off by 1-pixel, so trim off if out of range of the array
            # search = np.where(sky_mask.data > 0)

            # the sky_mask and the sky_image share a common origin (0,0) so they have the same indexing
            # the sky_mask may be larger than the sky_image, so turn the mask into a 2D array (so we can slice
            # and slice away the mask outside the sky_image)

            search = np.where(sky_mask > 0)
            annulus_data_1d = image.data[search]

            # and take the median average from a 3-sigma clip
            # bkg_mean, bkg_median, _ = sigma_clipped_stats(annulus_data_1d,sigma=3.0) #this is the average sky per pixel
            # sky_avg = bkg_median
            # #bkg_median * source_aperture.area() #total (sky) to subtract is averager sky per pix * number pix in source aperture

            # replace with biweight "average"
            bw_cen = biweight.biweight_location(annulus_data_1d)
            bw_scale = biweight.biweight_scale(annulus_data_1d)
            sky_pix = len(annulus_data_1d)
            sky_avg = bw_cen
            # set sky N to number of pixels within 1 sd (or bw_scale, in this case)
            # not strictly correct, but we are not including other error sources so this
            # will nudge toward a larger error (which would be more appropriate)
            N = len(np.where(abs(annulus_data_1d - bw_cen) < bw_scale)[0])
            sky_err = bw_scale / np.sqrt(N)


        except:
            log.error("Exception! ",exc_info=True)
            return None,None,None,None

        return sky_avg, sky_err, sky_pix, N



    def get_pixels_under_aperture(self,cutout,ra,dec,semi_major,semi_minor,angle,north_angle=np.pi/2.):
        """
        Get the pixels under an aperture (ellipse ... for circle major=minor)

        :param cutout:
        :param ra: decimal degrees (of center of aperture) (unitless float)
        :param dec: decimal degrees (of center of aperture) (unitless float)
        :param major: semi major axis ('radius') in arcsec (unitless float)
        :param minor: semi minor axis ('radius') in arcsec (unitless float)
        :param angle: rotation from 0, counter clockwise in radians (unitless float)
        :param north_angle: angle to north, counter clockwise from positive x-axsis (unitless float)
        :return: array of pixels (values)
        """

        #notice. these are from source extractor which does not use the WCS and its zero angle is the positive x-axis

        pixels = None
        try:
            ap_center = SkyCoord(ra, dec, unit="deg")
            a = semi_major*ap_units.arcsec
            b = semi_minor*ap_units.arcsec
            theta = ((angle - north_angle) * 180. / np.pi)*ap_units.deg

            #test use to_pixel to convert to pixel coords and overplot just to be sure
            aperture = SkyEllipticalAperture(ap_center,a,b,theta).to_pixel(cutout.wcs)

            try:
                sky_mask = aperture.to_mask(method='center')[0]
            except:
                sky_mask = aperture.to_mask(method='center')

            sky_mask = np.array(sky_mask)[0:cutout.data.shape[0], 0:cutout.data.shape[1]]
            # select all pixels from the cutout that are in the annulus
            # this sometimes gets off by 1-pixel, so trim off if out of range of the array
            # search = np.where(sky_mask.data > 0)

            # the sky_mask and the sky_image share a common origin (0,0) so they have the same indexing
            # the sky_mask may be larger than the sky_image, so turn the mask into a 2D array (so we can slice
            # and slice away the mask outside the sky_image)

            search = np.where(sky_mask > 0)
            pixels = cutout.data[search].astype(int) #want as integer (bitmapped mask), not float

        except:
            log.info("Exception in science_image::get_pixels_under_aperture",exc_info=True)

        return pixels



    def get_position(self,ra,dec,cutout):
        #this is not strictly a pixel x and y but is a meta position based on the wcs
        #remember pixel (0,0) is not necessarily x=0,y=0 from this call
        if cutout is None:
            log.warning("Supplied cutout is None (science_image::get_position)")
            return None,None
        x,y = None,None
        try:
            pix_size = self.calc_pixel_size(cutout.wcs)
            position = SkyCoord(ra, dec, unit="deg", frame='icrs') #,obstime="2017-05-02T08:39:48")
            x,y = skycoord_to_pixel(position, wcs=cutout.wcs,mode='all')
            x = x*pix_size
            y = y*pix_size
        except:
            log.info("Exception in science_image:get_position:", exc_info=True)

        return x,y

    def get_rotation_to_celestrial_north(self,cutout):
        #counterclockwise angle in radians to celestrial north from the x-axis (so north up would be pi/2)
        try:
            if hasattr(cutout.wcs.wcs, 'cd'):
                theta = np.arctan2(cutout.wcs.wcs.cd[0, 1], cutout.wcs.wcs.cd[0, 0]) - np.pi/2.
            elif hasattr(cutout.wcs.wcs,'pc'):
                theta = np.arctan2(cutout.wcs.wcs.pc[0, 1], cutout.wcs.wcs.pc[0, 0]) - np.pi/2.

            #theta = np.pi/2. - np.arctan2(cutout.wcs.wcs.cd[0, 1], cutout.wcs.wcs.cd[0, 0])

            if theta < 0: # clockwise rotation
                theta += 2*np.pi
            log.debug("Rotation (radians) = %g" % theta)
            return theta
        except:
            log.error("Unable to calculate rotation.",exc_info=True)
            return None

    def get_rect_parms(self,cutout,x,y,rot=None): #side is length of whole side ... 2*error window
        #rotation passed in radians, returned in degrees as wanted by rect call
        if rot is None: #the - pi/2 is so we get rotation from y-axis instead of x-axis
            rot = self.get_rotation_to_celestrial_north(cutout) - np.pi/2.0

        rect_rot = rot * 180./np.pi

        coords = np.dot(self.rotation_matrix(rot,deg=False), np.array([x, y]).transpose())
        x = coords[0]
        y = coords[1]

        return x,y,rect_rot

    def rotation_matrix(self,theta=0.0,deg=True):
        #Returns a rotation matrix for CCW rotation
        #if deg is False, theta is in radians
        if deg:
            rad = theta*np.pi/180.0
        else:
            rad = theta
        s = np.sin(rad)
        c = np.cos(rad)
        return np.array([[c, -s], [s, c]])

    def is_cutout_blank(self,cutout):
        #check the variance if zero, cutout is empty
        try:
            if np.var(cutout.data) == 0.0:
                return True
        except:
            return False
        return False


#DEAD CODE (1)
#           #DEAD CODE (1)
#             x_center, y_center = self.update_center(cutout,radius,play=G.NUDGE_MAG_APERTURE_CENTER)
#             self.last_x_center = x_center*self.pixel_size
#             self.last_y_center = y_center*self.pixel_size
#             self.last_x0_center = (x_center - cutout.center_cutout[0])*self.pixel_size #the shift in AA from center
#             self.last_y0_center = (y_center - cutout.center_cutout[1])*self.pixel_size
#             source_aperture_area = 0.0
#
#             try:
#                 sc = wcs_utils.pixel_to_skycoord(x_center,y_center,cutout.wcs,origin=0)
#                 details['ra'] = sc.ra.value
#                 details['dec']= sc.dec.value
#             except:
#                 log.warning("Exception! getting aperture RA,Dec.", exc_info=True)
#
#             if is_cutout_empty(cutout):
#                 if (G.ALLOW_EMPTY_IMAGE):
#                     pass
#                     #some cases seem to create problems, but forcing different values
#                     #or forcing all the same or all zero does not seem to matter
#                     #for matplotlib plot to PDF
#                     #it is an empty image anyway, and under normal configuration
#                     #this will get the default empty plot
#                     # if np.min(cutout.data) == np.max(cutout.data):
#                     #     cutout.data *= 0. #set all to exactly zero
#                 else:
#                     log.info("Cutout is empty or simple gradient. Will deliberately fail cutout request.")
#                     if return_details:
#                         return None, 0, 99.99, 0, details
#                     else:
#                         return None, 0, 99.99, 0
#
#             if False: #test out photutils
#                 from photutils import find_peaks, segmentation
#                 mean, median, std = sigma_clipped_stats(cutout.data, sigma=3.0)
#                 threshold = median + (5. * std)
#
#                 cx, cy = cutout.center_cutout
#                 pix = (radius / self.pixel_size)  # arcsec to pixels
#
#                 #positive mask (search where TRUE)
#                 mask = np.full(cutout.shape, False)
#                 mask[int(round(cx - pix)):int(round(cx + pix)), int(round(cy - pix)):int(round(cy + pix))] = True
#
#                 peaks_table = find_peaks(cutout.data,threshold,footprint=mask)
#
#                 connected_pixels = int(min((0.5 / self.pixel_size)**2,24))+1
#                 seg_img = segmentation.detect_sources(cutout.data,threshold,npixels=connected_pixels,mask=np.logical_not(mask))
#
#                 if False: #reminder, this will wipe out the imaging for the ELiXer Plot
#                     import matplotlib.pyplot as plt
#                     plt.close('all')
#                     plt.imshow(seg_img,origin='lower')
#                     plt.savefig("segimg.png")
#
#             if G.DYNAMIC_MAG_APERTURE:
#                 if aperture and (aperture > 0.0):
#                     radius = aperture
#                 elif G.MIN_DYNAMIC_MAG_RADIUS is not None:
#                     radius = G.MIN_DYNAMIC_MAG_RADIUS
#                 else:
#                     radius = 1.5  # failsafe
#
#                 radius = min(radius,G.MAX_DYNAMIC_MAG_APERTURE,error)
#                 step = 0.1
#
#                 sky_avg, sky_err, sky_pix, sd_sky_pix = self.get_local_sky(sky_image,position,G.MAX_DYNAMIC_MAG_APERTURE*2.0,G.MAX_DYNAMIC_MAG_APERTURE*4.0)
#
#                 # try:
#                 #     ps = cutout.wcs.pixel_scale_matrix[0][0] * 3600.0 #approx pix-scale
#                 #     step = ps/2. #half-pixel steps in terms of arcsec
#                 #     if step > radius:
#                 #         radius = step
#                 # except:
#                 #     step = 0.1
#
#                 log.debug("science_image::get_cutout() mag aperture radius step size = %f" %step)
#
#                 max_radius = radius
#                 max_bright = 99.9
#                 max_counts = 0
#                 max_area_pix = 0
#                 expected_count_growth = 0.0 #number of new pixels * sky
#
#                 #last_slope = 0. #not really a slope but looking to see if there is a rapid upturn and then stop
#                 #last_count = 0.
#
#                 mag_list = [99.9]
#                 rad_list = [0.0]
#
#                 while radius <= min(error,G.MAX_DYNAMIC_MAG_APERTURE):
#                     source_aperture_area = 0.
#                     try:
#                         #use the cutout first if possible (much smaller than image and faster)
#                         #note: net difference is less than 0.1 mag at 0.5" and less than 0.01 at 1.5"
#                         #I believe the source of the difference is the center position ... in the pixel mode
#                         #we center on the center pixel as reported by the center of the cutout. In sky mode
#                         #we center on an RA, Dec position and the slight difference between the two centers
#                         # (itself, less than one pixel) yields slightly different counts as the area covered is slightly
#                         # offset (fraction of a pixel or, typically, small fractions of an arcsec)
#                         try:
#                             #pix_aperture = CircularAperture(cutout.center_cutout,r=radius/self.pixel_size)
#                             pix_aperture = CircularAperture((x_center,y_center), r=radius / self.pixel_size)
#                             phot_table = aperture_photometry(cutout.data, pix_aperture,method=PIXEL_APERTURE_METHOD)
#                             counts = phot_table['aperture_sum'][0]
#                             try:
#                                 source_aperture_area = pix_aperture.area() #older version(s) of photutils
#                             except:
#                                 source_aperture_area = pix_aperture.area
#                         except:
#                             log.info("Pixel based aperture photometry failed. Attemping sky based ... ",exc_info=True)
#
#                             try:
#                                 sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
#                                 phot_table = aperture_photometry(image, sky_aperture,method=PIXEL_APERTURE_METHOD)
# #                                pix_aperture = CircularAperture(cutout.center_cutout, r=radius / self.pixel_size)
# #                                phot_table = aperture_photometry(cutout.data, pix_aperture)
#                                 counts = phot_table['aperture_sum'][0]
#                                 try:
#                                     source_aperture_area = sky_aperture.area() #older version(s) of photutils
#                                 except:
#                                     source_aperture_area = sky_aperture.area
#                             except:
#                                 log.info("Sky based aperture photometry failed. Will skip aperture photometery.",
#                                          exc_info=True)
#                                 break
#
#                         try:
#                             delta_pix = source_aperture_area - max_area_pix
#                             expected_count_growth = delta_pix*sky_avg
#                             #assume sd as error and add in quadrature
#                             expected_count_growth += np.sqrt(delta_pix)*sky_err
#                         except:
#                             expected_count_growth = 0.0
#
#
#                         #log.info("+++++ %s, %s" %(counts.__repr__(), type(counts)))
#                         #log.info("+++++\n %s" %(phot_table.__repr__()))
#                         #counts might now be a quantity type and not just a float
#                         #if isinstance(counts,astropy.units.quantity.Quantity):
#                         if not isinstance(counts, float):
#                             log.info("Attempting to strip units from counts (%s) ..." %(type(counts)))
#                             try:
#                                 counts = counts.value
#
#                                 if not isinstance(counts, float):
#                                     log.warning(
#                                         "Cannot cast counts as float. Will not attempt aperture magnitude calculation")
#                                     break
#
#                             except:
#                                 log.info("Failed to strip units. Cannot cast to float. "
#                                          "Will not attempt aperture magnitude calculation",exc_info=True)
#                                 break
#
#                         mag = mag_func(counts, cutout, self.headers)
#
#                         # pix_mag = mag_func(pix_counts, cutout, self.fits)
#                         # log.info("++++++ pix_mag (%f) sky_mag(%f)" %(pix_mag,mag))
#                         # log.info("++++++ pix_radius (%f)  sky_radius (%f)" % (radius / self.pixel_size, radius))
#                         # log.info("++++++ pix_counts (%f)  sky_counts (%f)" % (pix_counts, counts))
#
#                         log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %g mag = %g dmag = %g"
#                                  % (radius, ra, dec, counts, mag, mag - mag_list[-1]))
#
#                         mag_list.append(mag)
#                         rad_list.append(radius)
#
#                         #todo: if mag == 99.9 at radius == 0.5" maybe just stop or limit to 1" total?
#                         #todo: don't want to catch somthing on the edge and then expand
#                         #todo: plus our astrometry accuracy is ~ 0.5"
#
#                         if mag < 99:
#                             #if now growing fainter OR the brightness increase is small, we are done
#                             if expected_count_growth is not None:
#                                 if (mag > max_bright) or (counts-max_counts) < expected_count_growth:
#                                     break
#                             elif (mag > max_bright) or (abs(mag-max_bright) < 0.01):#< 0.0005):
#                                 break
#                             # elif last_count != 0:
#                             #     count_slope = (counts - max_counts) / last_count  # or current annulus/previous annulus
#                             #     area_slope = (2.*radius+step)/(2.*radius-step)
#                             #     if count_slope > (3. * area_slope): #should be stricly positive
#                             #         log.info("Aperture growth aborted. Unexpectedly large increase in counts.")
#                             #         break # (though, technically can have negative counts in some imaging)
#                         elif (radius >= aperture) or (abs(radius-aperture) <1e-5) or (radius > G.MAX_DYNAMIC_MAG_APERTURE):
#                             #weirdness in floats, difference when "==" is non-zero ~ 1e-16
#                             if max_bright > mag:
#                                 max_bright = mag
#                                 max_counts = counts
#                                 max_radius = radius
#                                 max_area_pix = source_aperture_area
#                             break
#
#                         max_bright = mag
#                         max_counts = counts
#                         max_radius = radius
#                         max_area_pix = source_aperture_area
#                         #last_count = counts - last_count #keeping just the counts in the current annulus
#                     except:
#                         log.error("Exception in science_image::get_cutout () using dynamic aperture", exc_info=True)
#
#
#                     radius += step
#                     #end while loop
#
#                 mag = max_bright
#                 counts = max_counts
#                 radius = max_radius
#                 source_aperture_area = max_area_pix
#
#                 details['radius'] = radius
#                 details['mag'] = mag
#                 details['aperture_counts'] = counts
#                 details['area_pix'] = source_aperture_area
#
#             else:
#
#                 try:
#                     if aperture and (aperture > 0.0):
#                         radius = aperture
#                     elif G.FIXED_MAG_APERTURE is not None:
#                         radius = G.FIXED_MAG_APERTURE
#                     else:
#                         radius = 1.5 #failsafe
#
#                     details['radius'] = radius
#                     source_aperture_area = 0.
#
#                     try:
#
#                         # try:
#                         #     cx, cy = cutout.center_cutout
#                         #     pix = (radius / self.pixel_size)
#                         #
#                         #     mask = np.full(cutout.shape,True)
#                         #     mask[int(round(cx-pix)):int(round(cx+pix)),int(round(cy-pix)):int(round(cy+pix))] = False
#                         #
#                         #     gx, gy = centroid_2dg(cutout.data,mask=mask)
#                         #
#                         #
#                         #     # import matplotlib.pyplot as plt
#                         #     # plt.close('all')
#                         #     # plt.imshow(cutout.data,origin="lower")
#                         #     # plt.plot(cx,cy,marker="+",color="r")
#                         #     # plt.plot(gx,gy,marker="x",color="y")
#                         #     # plt.plot(mask,alpha=0.5)
#                         #     # #plt.show()
#                         #     # plt.savefig("pos.png")
#                         #
#                         #     #allow the center to shift by up to 50% of the radius of the aperture
#                         #     dpix = pix #* 0.5
#                         #
#                         #     log.debug("Delta X,Y Pixels (centroid - center) = (%0.2f,%0.2f), arcsec = (%0.2f,%0.2f)"
#                         #               % (gx - cy, gy - cy, (gx - cy)*self.pixel_size, (gy - cy)*self.pixel_size))
#                         #
#                         #     #if the centroid is very near the center, use it,
#                         #     #but if we are off by more than 2pix, use the geomtric center of the cutout
#                         #     #basically, can nudge it a bit to the 2D Gaussian center, but don't stray too far
#                         #     if (abs(gx-cx) < dpix) and (abs(gy-cy) < dpix):
#                         #         log.info("Using shifted (2D Gaussian centroid) center for circular aperture ...")
#                         #         pix_aperture = CircularAperture((gx,gy), r=radius / self.pixel_size)
#                         #     else:
#                         #         log.info("Using geometric center for circular aperture ...")
#                         #         pix_aperture = CircularAperture((cx,cy), r=radius / self.pixel_size)
#                         # except:
#                         #     pix_aperture = CircularAperture(cutout.center_cutout, r=radius / self.pixel_size)
#
#                         #pix_aperture = CircularAperture(cutout.center_cutout, r=radius / self.pixel_size)
#                         pix_aperture = CircularAperture((x_center, y_center), r=radius / self.pixel_size)
#                         phot_table = aperture_photometry(cutout.data, pix_aperture,method=PIXEL_APERTURE_METHOD)
#                         try:
#                             source_aperture_area = pix_aperture.area()
#                         except:
#                             source_aperture_area = pix_aperture.area
#                     except:
#                         log.info("Pixel based aperture photometry failed. Attemping sky based ... ", exc_info=True)
#                         #note: if we do this, photutils loads the entire fits image and that can be costly
#                         #in terms of time and memory
#                         try:
#                             sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
#                             phot_table = aperture_photometry(image, sky_aperture,method=PIXEL_APERTURE_METHOD)
#                             try:
#                                 source_aperture_area = sky_aperture.area()
#                             except:
#                                 source_aperture_area = sky_aperture.area
#                         except:
#                             log.info("Sky based aperture photometry failed. Will skip aperture photometery.",
#                                      exc_info=True)
#                             if return_details:
#                                 return cutout, counts, mag, radius, details
#                             else:
#                                 return cutout, counts, mag, radius
#
#                     counts = phot_table['aperture_sum'][0]
#
#                     if not isinstance(counts, float):
#                         log.info("Attempting to strip units from counts (%s) ..." % (type(counts)))
#                         try:
#                             counts = counts.value
#                         except:
#                             log.info("Failed to strip units. Cannot cast to float. "
#                                      "Will not attempt aperture magnitude calculation", exc_info=True)
#
#                     mag = mag_func(counts,cutout,self.headers)
#
#                     details['mag'] = mag
#                     details['aperture_counts'] = counts
#                     details['area_pix'] = source_aperture_area
#
#                     log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %s Mag_AB = %g"
#                              % (radius,ra,dec,str(counts),mag))
#                     #print ("Counts = %s Mag %f" %(str(counts),mag))
#                 except:
#                     log.error("Exception in science_image::get_cutout () using aperture", exc_info=True)
#
#
#             #if we have a magnitude and it is fainter than a minimum, subtract the sky from a surrounding annulus
#             #s|t we have ~ 3x pixels in the sky annulus as in the source aperture, so 2x the radius
#             if do_sky_subtract and (mag < 99) and (mag > G.SKY_ANNULUS_MIN_MAG):
#
#                 try:
#
#                     #todo: note in photutils, pixel x,y is the CENTER of the pixel and [0,0] is the center of the
#                     #todo: lower-left pixel
#                     #it should not really matter if the pixel position is the center of the pixel or at a corner
#                     #given the typically small pixels the set of pixels will not change much one way or the other
#                     #and we are going to take a median average (and not use partial pixels), at least not in this 1st
#                     #revision
#
#                     # to take a median value and subtract off the counts from the original aperture
#                     # yes, against the new cutout (which is just a super set of the smaller cutout
#                     #source_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec).to_pixel(sky_image.wcs)
#
#                     sky_annulus = SkyCircularAnnulus(position, r_in=sky_inner_radius * ap_units.arcsec,
#                                                      r_out=sky_outer_radius * ap_units.arcsec).to_pixel(sky_image.wcs)
#
#                     #made an annulus in pixel, set to a "mask" where 0 = pixel not in annulus, 1 = pixel in annulus
#                     try:
#                         sky_mask = sky_annulus.to_mask(method='center')[0]
#                     except:
#                         sky_mask = sky_annulus.to_mask(method='center')
#
#
#                     sky_mask = np.array(sky_mask)[0:sky_image.data.shape[0],0:sky_image.data.shape[1]]
#                     #select all pixels from the cutout that are in the annulus
#                     #this sometimes gets off by 1-pixel, so trim off if out of range of the array
#                     # search = np.where(sky_mask.data > 0)
#
#                     #the sky_mask and the sky_image share a common origin (0,0) so they have the same indexing
#                     #the sky_mask may be larger than the sky_image, so turn the mask into a 2D array (so we can slice
#                     #and slice away the mask outside the sky_image)
#
#                     search = np.where(sky_mask > 0)
#                     # safe_search = [None]*2
#                     # safe_search[0] = search[0][np.where(search[0] < sky_image.data.shape[0])]
#                     # safe_search[1] = search[1][np.where(search[1] < sky_image.data.shape[1])]
#
#                     # #todo: lets matplotlib here ... the sky_mask and the sky_image.data
#                     # import matplotlib.pyplot as plt
#                     # plt.close('all')
#                     # plt.imshow(sky_image.data,origin='lower')
#                     # plt.imshow(sky_mask,origin='lower',alpha=0.5)
#                     # plt.savefig("mask.png")
#
#             #        safe_search = np.where((search[0] < sky_image.data.shape[0]) & (search[1] < sky_image.data.shape[1]))
#
#                     annulus_data_1d = sky_image.data[search]
#
#                     #and take the median average from a 3-sigma clip
#                     #bkg_mean, bkg_median, _ = sigma_clipped_stats(annulus_data_1d,sigma=3.0) #this is the average sky per pixel
#                     #sky_avg = bkg_median
#                     # #bkg_median * source_aperture.area() #total (sky) to subtract is averager sky per pix * number pix in source aperture
#
#                     #replace with biweight "average"
#                     bw_cen = biweight.biweight_location(annulus_data_1d)
#                     bw_scale = biweight.biweight_scale(annulus_data_1d)
#                     sky_pix = len(annulus_data_1d)
#                     sky_avg = bw_cen
#                     #set sky N to number of pixels within 1 sd (or bw_scale, in this case)
#                     #not strictly correct, but we are not including other error sources so this
#                     #will nudge toward a larger error (which would be more appropriate)
#                     N = len(np.where(abs(annulus_data_1d-bw_cen) < bw_scale)[0])
#                     sky_err = bw_scale/np.sqrt(N)
#
#
#                     sky_cts = sky_avg * source_aperture_area
#                     base_counts = counts
#                     counts -= sky_cts
#                     #re-compute the magnitude
#                     base_mag = mag
#                     sky_mag = mag_func(counts,cutout,self.headers)
#
#                     sky_mag_bright = mag_func(base_counts - (sky_avg - sky_err) * source_aperture_area,
#                              cutout, self.headers)
#
#                     sky_mag_faint = mag_func(base_counts - (sky_avg + sky_err) * source_aperture_area,
#                              cutout, self.headers)
#
#                     if sky_mag_faint < 99:
#                         mag_err = max((sky_mag_faint-sky_mag),(sky_mag-sky_mag_bright))
#                     else:
#                         mag_err = sky_mag-sky_mag_bright
#
#                     #todo: temporary
#                     if False:
#                         with open("check_mag_log.txt","a+") as logfile:
#                             marker = " "
#                             if abs(sky_mag - base_mag) > 1.0:
#                                 marker = '*'
#
#                             logfile.write("%s %s %s %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %s"
#                                 %(marker, self.catalog_name, self.filter_name, base_mag-sky_mag,ra,dec,
#                                   cutout.center_cutout[0],cutout.center_cutout[1], self.last_x_center, self.last_y_center,self.pixel_size,
#                                   base_mag,sky_mag,source_aperture_area,base_counts,len(annulus_data_1d),bw_cen,
#                                   np.sum(annulus_data_1d),np.mean(annulus_data_1d),np.median(annulus_data_1d),np.std(annulus_data_1d),"\n"))
#
#
#
#
#
#                     #mag should now be fainter (usually ... could have slightly negative sky?)
#                     #the photometry should have pretty good sky subtration ... but what if we are on a faint object
#                     #that is near large object ... could be we don't get enough sky pixels or the average is skewed high
#                     #so if we make much of a change, at least log a warning
#                     if abs(sky_mag - base_mag) > G.MAX_SKY_SUBTRACT_MAG:
#                        # print("Warning! Unexepectedly large sky subtraction impact to magnitude: %0.2f to %0.2f at (%f,%f)"
#                        #             %(base_mag,sky_mag,ra,dec))
#                         log.warning("Warning! Unexepectedly large sky subtraction impact to magnitude: %0.2f to %0.2f at (%f,%f)"
#                                     %(base_mag,sky_mag,ra,dec))
#                     else: #todo: !!!! temporary ... just to see what happens if we keep the original base mag
#                         mag = sky_mag
#
#                     log.info("Sky subtracted imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Sky = (%f/pix %f tot). Counts = %s Mag_AB = %g"
#                              % (radius,ra,dec,sky_avg, sky_cts,str(counts),mag))
#                     #print ("Counts = %s Mag %f" %(str(counts),mag))
#
#                     details['aperture_counts'] = counts #update to sky subtracted counts to be consistent
#                     details['sky_area_pix'] = sky_pix
#                     details['sky_average'] = sky_avg
#                     details['sky_counts'] = sky_cts
#                     details['mag'] = mag
#                     details['mag_err'] = mag_err
#                     details['mag_bright'] = sky_mag_bright
#                     details['mag_faint'] = sky_mag_faint
#
#                 except:
#                     #print("Sky Mask Problem ....")
#                     log.error("Exception in science_image::get_cutout () figuring sky subtraction aperture", exc_info=True)
