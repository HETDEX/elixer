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
from photutils import SkyCircularAperture #sky coords
from photutils import aperture_photometry

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
        self.fits = None # fits handle
        self.wcs = None
        self.vmin = None
        self.vmax = None
        self.pixel_size = None
        self.window = None
        self.exptime = None

        self.image_buffer = None

        self.wcs_idx = wcs_idx #usually on hdu 0 but sometimes not (i.e. HyperSuprimeCam on 1)
        self.footprint = None #on sky footprint, decimal degrees Ra,Dec as 4x2 array (with North up: LL, UL, UR, LR)

        #todo: do I need to worry about this?
        if frame is not None:
            self.frame = frame
        else:
            self.frame = 'fk5'

        if (image_location is not None) and (len(image_location) > 0):
            self.image_location = image_location
            self.load_image(wcs_manual=wcs_manual)

    def load_image(self,wcs_manual=False):
        if (self.image_location is None) or (len(self.image_location) == 0):
            return -1

        if self.fits is not None:
            try:
                self.fits.close()
            except:
                log.info("Unable to close fits file.")

        try:
            log.info("Loading fits %s ..." % self.image_location)
            self.fits = fits.open(self.image_location)
        except:
            log.error("Unable to open science image file: %s" %self.image_location)
            return -1

        if wcs_manual:
            self.build_wcs_manually()
        else:
            try:
                self.wcs = WCS(self.image_location)
            except:
                log.error("Unable to use WCS constructor. Will attempt to build manually.", exc_info=True)
                self.build_wcs_manually()

        if self.wcs is None: #must have WCS
            return -1

        try:
            self.footprint = WCS.calc_footprint(self.wcs)
        except:
            log.error("Unable to get on-sky footprint")

        try:
            self.exptime = self.fits[self.wcs_idx].header['EXPTIME']
        except:
            try:#if not with the wcs header, check the main (0) header
                self.exptime = self.fits[0].header['EXPTIME']
            except:

                log.warning('Warning. Could not load exposure time from %s' %self.image_location, exc_info=True)
                self.exptime = None

        try:
            self.pixel_size = self.calc_pixel_size(self.wcs)#np.sqrt(self.wcs.wcs.cd[0, 0] ** 2 + self.wcs.wcs.cd[0, 1] ** 2) * 3600.0  # arcsec/pixel
            log.debug("Pixel Size = %f asec/pixel" %self.pixel_size)
        except:
            log.error("Unable to build pixel size", exc_info=True)

    def build_wcs_manually(self):
        try:
            self.wcs = WCS(naxis=self.fits[self.wcs_idx].header['NAXIS'])
            self.wcs.wcs.crpix = [self.fits[self.wcs_idx].header['CRPIX1'], self.fits[self.wcs_idx].header['CRPIX2']]
            self.wcs.wcs.crval = [self.fits[self.wcs_idx].header['CRVAL1'], self.fits[self.wcs_idx].header['CRVAL2']]
            self.wcs.wcs.ctype = [self.fits[self.wcs_idx].header['CTYPE1'], self.fits[self.wcs_idx].header['CTYPE2']]
            #self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
            self.wcs.wcs.cd = [[self.fits[self.wcs_idx].header['CD1_1'], self.fits[self.wcs_idx].header['CD1_2']],
                               [self.fits[self.wcs_idx].header['CD2_1'], self.fits[self.wcs_idx].header['CD2_2']]]
            self.wcs._naxis1 = self.fits[self.wcs_idx].header['NAXIS1']
            self.wcs._naxis2 = self.fits[self.wcs_idx].header['NAXIS2']
        except:
            log.error("Failed to build WCS manually.",exc_info=True)
            self.wcs = None

    def contains_position(self,ra,dec):
        if self.footprint is not None: #do fast check first
            if (ra  > np.max(self.footprint[:,0])) or (ra  < np.min(self.footprint[:,0])) or \
               (dec > np.max(self.footprint[:,1])) or (dec < np.min(self.footprint[:,1])):
                #can't be inside the rectangle
                log.debug("position (%f, %f) is not in image max rectangle." % (ra, dec), exc_info=False)
                return False

        #now, it could be, so actually try a cutout to see if it will work
        try:
            cutout = Cutout2D(self.fits[self.wcs_idx].data, SkyCoord(ra, dec, unit="deg", frame=self.frame), (1, 1),
                              wcs=self.wcs, copy=True)#,mode='partial')
        except:
            log.debug("position (%f, %f) is not in image." % (ra,dec), exc_info=False)
            return False
        return True

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
        radius = aperture

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
            if self.fits is not None:
                #need to keep memory use down ... the statements below (esp. data = self.fit[0].data)
                #cause a copy into memory ... want to avoid it so, leave the code fragmented a bit as is
               # data = self.fits[0].data
               # pix_size = self.pixel_size
               # wcs = self.wcs

                try:
                    position = SkyCoord(ra, dec, unit="deg", frame=self.frame)
                    image = self.fits[self.wcs_idx]

                    #sanity check
                    #x, y = skycoord_to_pixel(position, wcs=self.wcs, mode='all')
                    #x = x * self.pix_size
                    #y = y * self.pix_size

                    pix_window = int(np.ceil(window / self.pixel_size))  # now in pixels
                    log.debug("Collecting cutout size = %d square at RA,Dec = (%f,%f)" %(pix_window,ra,dec))
                    cutout = Cutout2D(self.fits[self.wcs_idx].data, position, (pix_window, pix_window), wcs=self.wcs, copy=copy)
                    self.get_vrange(cutout.data)
                except NoOverlapError:
                    log.info("Error (possible NoOverlapError) in science_image::get_cutout(). *** Did more than one catalog match the coordinates? ***"
                             "Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                    print("Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                    return cutout, counts, mag, radius
                except:
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
                radius = 0.5
                step = 0.1
                max_radius = radius
                max_bright = 99.9
                max_counts = 0

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


                        mag = mag_func(counts, cutout, self.fits)

                        # pix_mag = mag_func(pix_counts, cutout, self.fits)
                        # log.info("++++++ pix_mag (%f) sky_mag(%f)" %(pix_mag,mag))
                        # log.info("++++++ pix_radius (%f)  sky_radius (%f)" % (radius / self.pixel_size, radius))
                        # log.info("++++++ pix_counts (%f)  sky_counts (%f)" % (pix_counts, counts))

                        log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %g mag = %g"
                                 % (radius, ra, dec, counts, mag))

                        #todo: if mag == 99.9 at radius == 0.5" maybe just stop or limit to 1" total?
                        #todo: don't want to catch somthing on the edge and then expand
                        #todo: plus our astrometry accuracy is ~ 0.5"

                        if mag < 99:
                            if (mag > max_bright) or (abs(mag-max_bright) < 0.05):
                                break
                        elif (radius >= aperture) or (abs(radius-aperture) <1e-5):
                            #weirdness in floats, difference when "==" is non-zero ~ 1e-16
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
                    if (type(aperture) is float) or (type(aperture) is int):
                        radius = aperture
                    else:
                        radius = 1.
                    sky_aperture = SkyCircularAperture(position, r=radius * ap_units.arcsec)
                    phot_table = aperture_photometry(image,sky_aperture)
                    counts = phot_table['aperture_sum'][0]
                    mag = mag_func(counts,cutout,self.fits)

                    log.info("Imaging circular aperture radius = %g\" at RA, Dec = (%g,%g). Counts = %g Mag_AB = %g"
                             % (radius,ra,dec,counts,mag))
                    print ("Counts = %f Mag %f" %(counts,mag))
                except:
                    log.error("Exception in science_image::get_cutout () using aperture", exc_info=True)

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