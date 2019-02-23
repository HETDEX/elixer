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

import global_config as G
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
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import NoOverlapError
from astropy import units as ap_units
from photutils import CircularAperture #pixel coords
from photutils import SkyCircularAperture, SkyCircularAnnulus #sky coords
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats


#log = G.logging.getLogger('sciimg_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('sciimg_logger')
log.setlevel(G.logging.DEBUG)

class science_image():

    def __init__(self, wcs_manual=False, image_location=None,frame=None, wcs_idx=0):
        self.image_location = None
        self.image_name = None
        self.catalog_name = None
        self.filter_name = None
        self.wavelength_aa_min = 0
        self.wavelength_aa_max = 0

        self.ra_center = 0.0
        self.dec_center = 0.0

        #self.fits = None # fits handle

        self.hdulist = None
        self.headers = None #array of hdulist headers

        self.wcs = None
        self.vmin = None
        self.vmax = None
        self.pixel_size = None
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
                self.wcs = WCS(self.image_location)
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

        try:
            self.exptime = self.hdulist[self.wcs_idx].header['EXPTIME']
        except:
            try:#if not with the wcs header, check the main (0) header
                self.exptime = self.hdulist[0].header['EXPTIME']
            except:

                log.warning('Warning. Could not load exposure time from %s' %self.image_location, exc_info=True)
                self.exptime = None

        try:
            self.pixel_size = self.calc_pixel_size(self.wcs)#np.sqrt(self.wcs.wcs.cd[0, 0] ** 2 + self.wcs.wcs.cd[0, 1] ** 2) * 3600.0  # arcsec/pixel
            log.debug("Pixel Size = %f asec/pixel" %self.pixel_size)
        except:
            log.error("Unable to build pixel size", exc_info=True)

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
            self.wcs = WCS(naxis=hdulist[self.wcs_idx].header['NAXIS'])
            self.wcs.wcs.crpix = [hdulist[self.wcs_idx].header['CRPIX1'], hdulist[self.wcs_idx].header['CRPIX2']]
            self.wcs.wcs.crval = [hdulist[self.wcs_idx].header['CRVAL1'], hdulist[self.wcs_idx].header['CRVAL2']]
            self.wcs.wcs.ctype = [hdulist[self.wcs_idx].header['CTYPE1'], hdulist[self.wcs_idx].header['CTYPE2']]
            #self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
            self.wcs.wcs.cd = [[hdulist[self.wcs_idx].header['CD1_1'], hdulist[self.wcs_idx].header['CD1_2']],
                               [hdulist[self.wcs_idx].header['CD2_1'], hdulist[self.wcs_idx].header['CD2_2']]]
            self.wcs._naxis1 = hdulist[self.wcs_idx].header['NAXIS1']
            self.wcs._naxis2 = hdulist[self.wcs_idx].header['NAXIS2']
        except:
            log.error("Failed to build WCS manually.",exc_info=True)
            self.wcs = None

        if close:
            hdulist.close()

    def contains_position(self,ra,dec):
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
                return -1
        else:
            log.info("Using outer scope fits %s ..." % self.image_location)
            hdulist = self.hdulist

        #now, it could be, so actually try a cutout to see if it will work
        try:
            cutout = Cutout2D(hdulist[self.wcs_idx].data, SkyCoord(ra, dec, unit="deg", frame=self.frame), (1, 1),
                              wcs=self.wcs, copy=False)#,mode='partial')
        except:
            log.debug("position (%f, %f) is not in image." % (ra,dec), exc_info=False)
            rc = False

        if close:
            hdulist.close()

        return rc

    def calc_pixel_size(self,wcs):
        return np.sqrt(wcs.wcs.cd[0, 0] ** 2 + wcs.wcs.cd[0, 1] ** 2) * 3600.0

    def get_vrange(self,vals,contrast=0.25):
        self.vmin = None
        self.vmax = None

        try:
            zscale = ZScaleInterval(contrast=contrast,krej=2.5) #nsamples=len(vals)
            self.vmin, self.vmax = zscale.get_limits(values=vals )
            log.info("Vrange = %f, %f" %(self.vmin,self.vmax))
        except:
            log.info("Exception in science_image::get_vrange:",exc_info =True)

        return self.vmin,self.vmax

    def get_cutout(self,ra,dec,error,window=None,image=None,copy=False,aperture=0,mag_func=None):
        '''ra,dec in decimal degrees. error and window in arcsecs'''
        #error is central box (+/- from ra,dec)
        #window is the size of the entire coutout
        #return a cutout

        self.window = None
        cutout = None
        counts = None #raw data counts in aperture
        mag = 999.9 #aperture converted to mag_AB
        if (aperture is not None) and (mag_func is not None):
            radius = aperture

            # aperture-radius is not allowed to grow past the error-radius in the dynamic case
            if G.DYNAMIC_MAG_APERTURE:
                max_aperture = max(0, error, radius)
            else:
                max_aperture = G.FIXED_MAG_APERTURE

            sky_outer_radius = max_aperture * 10. #this is the maximum it can be
            sky_inner_radius = max_aperture * 5.
        else:
            radius = 0.0
            sky_outer_radius = 0.
            sky_inner_radius = 0.


        if (error is None or error == 0) and (window is None or window == 0):
            log.info("inavlid error box and window box")
            return cutout, counts, mag, radius

        if window is None or window < error:
            window = float(max(2.0*error,5.0)) #should be at least 5 arcsecs

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
                    return -1
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
                    max_retries = 30
                    while retries < max_retries:
                        try:
                            cutout = Cutout2D(hdulist[self.wcs_idx].data, position, (pix_window, pix_window),
                                              wcs=self.wcs, copy=copy)

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
                                                      wcs=self.wcs, copy=False) #don't need a copy, will not persist beyond
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

                                t2sleep = np.random.random_integers(0,5000)/1000. #sleep up to 5 sec
                                log.info("+++++ Memory issue? Random sleep (%d / %d) and retry (%f)s" %
                                         (retries,max_retries,t2sleep),exc_info=True)
                                sleep(t2sleep)

                                try:
                                    log.info("Loading (local scope) fits %s ..." % self.image_location)
                                    hdulist = fits.open(self.image_location, memmap=True, lazy_load_hdus=True)
                                    close = True
                                except:
                                    log.error("Unable to open science image file: %s" % self.image_location)
                                    return -1
                            else:
                                log.error("Unable to open science image file: %s" % self.image_location)
                                retries = max_retries
                        except:
                            log.error("Exception. Unable to load cutout.",exc_info=True)
                            retries = max_retries


                    if close:
                        try:
                            hdulist.close()
                        except:
                            log.warning("Exception attempting to close hdulist. science_image::get_cutout", exc_info=True)

                    if retries >= max_retries:
                        log.info("+++++ giving up ....")
                        return None, counts, mag, radius
                    elif retries > 0:
                        log.info("+++++ it worked (%d) ...." % retries)



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
                    return cutout, counts, mag, radius
                except:
                    #after = float(G.HPY.heap().size) /(2**20)

                    #msg = "+++++ heap before (%0.2f), after (%0.2f) MB (exception case)" %(before,after)
                    #print(msg)
                    #log.debug(msg)

                    log.error("Exception in science_image::get_cutout (%s):" %self.image_location, exc_info=True)
                    return cutout, counts, mag, radius

                if not (self.contains_position(ra,dec)):
                    log.info("science image (%s) does not contain requested position: RA=%f , Dec=%f"
                             %(self.image_location,ra,dec))
                    return cutout, counts, mag, radius
            else:
                log.error("No fits or passed image from which to make cutout.")
                return cutout, counts, mag, radius
        else:
            #data = image.data
            #pix_size = self.calc_pixel_size(image.wcs)
            #wcs = image.wcs
            try:
                position = SkyCoord(ra, dec, unit="deg")#, frame='fk5')
                #self.pixel_size = self.calc_pixel_size(image.wcs)
                pix_window = float(window) / self.calc_pixel_size(image.wcs)  # now in pixels
                cutout = Cutout2D(image.data, position, (pix_window, pix_window), wcs=image.wcs, copy=copy)
                self.get_vrange(cutout.data)
            except NoOverlapError as e:
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
                return cutout, counts, mag, radius

        #put down aperture on cutout at RA,Dec and get magnitude
        if (position is not None) and (cutout is not None) and (image is not None) \
                and (mag_func is not None) and (aperture > 0):

            if G.DYNAMIC_MAG_APERTURE:
                radius = max(0.5,aperture)
                step = 0.1

                # try:
                #     ps = cutout.wcs.pixel_scale_matrix[0][0] * 3600.0 #approx pix-scale
                #     step = ps/2. #half-pixel steps in terms of arcsec
                #     if step > radius:
                #         radius = step
                # except:
                #     step = 0.1

                log.debug("science_image::get_cutout() mag aperture radius step size = %f" %step)

                max_radius = radius
                max_bright = 99.9
                max_counts = 0

                mag_list = [99.9]
                rad_list = [0.0]

                while radius <= error:
                    try:
                        #use the cutout first if possible (much smaller than image and faster)
                        #note: net difference is less than 0.1 mag at 0.5" and less than 0.01 at 1.5"
                        #I believe the source of the difference is the center position ... in the pixel mode
                        #we center on the center pixel as reported by the center of the cutout. In sky mode
                        #we center on an RA, Dec position and the slight difference between the two centers
                        # (itself, less than one pixel) yields slightly different counts as the area covered is slightly
                        # offset (fraction of a pixel or, typically, small fractions of an arcsec)
                        try:
                            pix_aperture = CircularAperture(cutout.center_cutout,r=radius/self.pixel_size)
                            phot_table = aperture_photometry(cutout.data, pix_aperture)
                            counts = phot_table['aperture_sum'][0]
                        except:
                            log.info("Pixel based aperture photometry failed. Attemping sky based ... ",exc_info=True)

                            try:
                                sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
                                phot_table = aperture_photometry(image, sky_aperture)
#                                pix_aperture = CircularAperture(cutout.center_cutout, r=radius / self.pixel_size)
#                                phot_table = aperture_photometry(cutout.data, pix_aperture)
                                counts = phot_table['aperture_sum'][0]
                            except:
                                log.info("Sky based aperture photometry failed. Will skip aperture photometery.",
                                         exc_info=True)
                                break

                        #log.info("+++++ %s, %s" %(counts.__repr__(), type(counts)))
                        #log.info("+++++\n %s" %(phot_table.__repr__()))
                        #counts might now be a quantity type and not just a float
                        #if isinstance(counts,astropy.units.quantity.Quantity):
                        if not isinstance(counts, float):
                            log.info("Attempting to strip units from counts (%s) ..." %(type(counts)))
                            try:
                                counts = counts.value

                                if not isinstance(counts, float):
                                    log.warning(
                                        "Cannot cast counts as float. Will not attempt aperture magnitude calculation")
                                    break

                            except:
                                log.info("Failed to strip units. Cannot cast to float. "
                                         "Will not attempt aperture magnitude calculation",exc_info=True)
                                break

                        mag = mag_func(counts, cutout, self.headers)

                        # pix_mag = mag_func(pix_counts, cutout, self.fits)
                        # log.info("++++++ pix_mag (%f) sky_mag(%f)" %(pix_mag,mag))
                        # log.info("++++++ pix_radius (%f)  sky_radius (%f)" % (radius / self.pixel_size, radius))
                        # log.info("++++++ pix_counts (%f)  sky_counts (%f)" % (pix_counts, counts))

                        log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %g mag = %g dmag = %g"
                                 % (radius, ra, dec, counts, mag, mag - mag_list[-1]))

                        mag_list.append(mag)
                        rad_list.append(radius)

                        #todo: if mag == 99.9 at radius == 0.5" maybe just stop or limit to 1" total?
                        #todo: don't want to catch somthing on the edge and then expand
                        #todo: plus our astrometry accuracy is ~ 0.5"

                        if mag < 99:
                            if (mag > max_bright) or (abs(mag-max_bright) < 0.01):#< 0.0005):
                                break
                        elif (radius >= aperture) or (abs(radius-aperture) <1e-5):
                            #weirdness in floats, difference when "==" is non-zero ~ 1e-16
                            if max_bright > mag:
                                max_bright = mag
                                max_counts = counts
                                max_radius = radius
                            break

                        max_bright = mag
                        max_counts = counts
                        max_radius = radius
                    except:
                        log.error("Exception in science_image::get_cutout () using dynamic aperture", exc_info=True)

                    radius += step
                    #end while loop

                mag = max_bright
                counts = max_counts
                radius = max_radius

            else:

                try:
                    if G.FIXED_MAG_APERTURE is not None:
                        radius = G.FIXED_MAG_APERTURE
                    else:
                        if (type(aperture) is float) or (type(aperture) is int):
                            radius = aperture
                        else:
                            radius = 1.

                    try:
                        pix_aperture = CircularAperture(cutout.center_cutout, r=radius / self.pixel_size)
                        phot_table = aperture_photometry(cutout.data, pix_aperture)
                    except:
                        log.info("Pixel based aperture photometry failed. Attemping sky based ... ", exc_info=True)
                        #note: if we do this, photutils loads the entire fits image and that can be costly
                        #in terms of time and memory
                        try:
                            sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
                            phot_table = aperture_photometry(image, sky_aperture)
                        except:
                            log.info("Sky based aperture photometry failed. Will skip aperture photometery.",
                                     exc_info=True)
                            return cutout, counts, mag, radius

                    counts = phot_table['aperture_sum'][0]

                    if not isinstance(counts, float):
                        log.info("Attempting to strip units from counts (%s) ..." % (type(counts)))
                        try:
                            counts = counts.value
                        except:
                            log.info("Failed to strip units. Cannot cast to float. "
                                     "Will not attempt aperture magnitude calculation", exc_info=True)

                    mag = mag_func(counts,cutout,self.headers)

                    log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %s Mag_AB = %g"
                             % (radius,ra,dec,str(counts),mag))
                    #print ("Counts = %s Mag %f" %(str(counts),mag))
                except:
                    log.error("Exception in science_image::get_cutout () using aperture", exc_info=True)


            #if we have a magnitude and it is fainter than a minimum, subtract the sky from a surrounding annulus
            #s|t we have ~ 3x pixels in the sky annulus as in the source aperture, so 2x the radius
            if (mag < 99) and (mag > G.SKY_ANNULUS_MIN_MAG):

                try:

                    # we know position, image, are good or could not have gotten here
                    # if self.pixel_size is not None:
                    #     pix_window = 2. * float(sky_outer_radius * 1.1) / self.pixel_size #length of size, so 2x radius
                    # #    sky_cutout = Cutout2D(image.data, position, (pix_window, pix_window), wcs=self.wcs)
                    # #    wcs = self.wcs
                    # else:
                    #     pix_window = 2. * float(sky_outer_radius*1.1) / self.calc_pixel_size(sky_image.wcs)  # now in pixels
                    #  #   sky_cutout = Cutout2D(image.data, position, (pix_window, pix_window), wcs=image.wcs)
                    #  #   wcs = image.wcs
                    #
                    # #is there anyway we don't have image.data and image.wcs?
                    # sky_cutout = Cutout2D(sky_image.data, position, (pix_window, pix_window), wcs=sky_image.wcs)


                    #todo: note in photutils, pixel x,y is the CENTER of the pixel and [0,0] is the center of the
                    #todo: lower-left pixel
                    #it should not really matter if the pixel position is the center of the pixel or at a corner
                    #given the typically small pixels the set of pixels will not change much one way or the other
                    #and we are going to take a median average (and not use partial pixels), at least not in this 1st
                    #revision

                    # to take a median value and subtract off the counts from the original aperture
                    # yes, against the new cutout (which is just a super set of the smaller cutout
                    source_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec).to_pixel(sky_image.wcs)

                    sky_annulus = SkyCircularAnnulus(position, r_in=sky_inner_radius * ap_units.arcsec,
                                                     r_out=sky_outer_radius * ap_units.arcsec).to_pixel(sky_image.wcs)

                    #made an annulus in pixel, set to a "mask" where 0 = pixel not in annulus, 1 = pixel in annulus
                    sky_mask = sky_annulus.to_mask(method='center')[0]

                    #print("+++++", np.shape(sky_image.data),pix_window)

                    #select all pixels from the cutout that are in the annulus
                    annulus_data_1d = sky_image.data[np.where(sky_mask.data > 0)]

                    #print("+++++ total sky pixels",len(annulus_data_1d))

                    #and take the median average from a 3-sigma clip
                    bkg_mean, bkg_median, _ = sigma_clipped_stats(annulus_data_1d,sigma=3.0) #this is the average sky per pixel
                    #bkg_median * source_aperture.area() #total (sky) to subtract is averager sky per pix * number pix in source aperture

                    #print("+++++", bkg_mean, bkg_median)

                    counts -= bkg_median * source_aperture.area()

                    #re-compute the magnitude
                    mag = mag_func(counts,cutout,self.headers)

                    log.info("Sky subtracted imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Sky = (%f/pix %f tot). Counts = %s Mag_AB = %g"
                             % (radius,ra,dec,bkg_median, bkg_median * source_aperture.area(),str(counts),mag))
                    #print ("Counts = %s Mag %f" %(str(counts),mag))


                except:
                    log.error("Exception in science_image::get_cutout () figuring sky subtraction aperture", exc_info=True)

        return cutout, counts, mag, radius

    def get_position(self,ra,dec,cutout):
        #this is not strictly a pixel x and y but is a meta position based on the wcs
        #remember pixel (0,0) is not necessarily x=0,y=0 from this call
        x,y = None,None
        try:
            pix_size = self.calc_pixel_size(cutout.wcs)
            position = SkyCoord(ra, dec, unit="deg", frame='fk5') #,obstime="2017-05-02T08:39:48")
            x,y = skycoord_to_pixel(position, wcs=cutout.wcs,mode='all')
            x = x*pix_size
            y = y*pix_size
        except:
            log.info("Exception in science_image:get_position:", exc_info=True)

        return x,y

    def get_rotation_to_celestrial_north(self,cutout):
        #counterclockwise angle in radians to celestrial north from the x-axis (so north up would be pi/2)
        try:
            theta = np.arctan2(cutout.wcs.wcs.cd[0, 1], cutout.wcs.wcs.cd[0, 0]) - np.pi/2.
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