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

import global_config
import numpy as np
from astropy.visualization import ZScaleInterval
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
#from astropy.coordinates import match_coordinates_sky
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import NoOverlapError

log = global_config.logging.getLogger('sciimg_logger')
log.setLevel(global_config.logging.DEBUG)

class science_image():

    def __init__(self, wcs_manual=False, image_location=None,frame=None):
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

        try:
            self.exptime = self.fits[0].header['EXPTIME']
        except:
            log.warning('Warning. Could not load exposure time from %s' %self.image_location, exc_info=True)

        try:
            self.pixel_size = self.calc_pixel_size(self.wcs)#np.sqrt(self.wcs.wcs.cd[0, 0] ** 2 + self.wcs.wcs.cd[0, 1] ** 2) * 3600.0  # arcsec/pixel
            log.debug("Pixel Size = %f asec/pixel" %self.pixel_size)
        except:
            log.error("Unable to build pixel size", exc_info=True)

    def build_wcs_manually(self):
        self.wcs = WCS(naxis=self.fits[0].header['NAXIS'])
        self.wcs.wcs.crpix = [self.fits[0].header['CRPIX1'], self.fits[0].header['CRPIX2']]
        self.wcs.wcs.crval = [self.fits[0].header['CRVAL1'], self.fits[0].header['CRVAL2']]
        self.wcs.wcs.ctype = [self.fits[0].header['CTYPE1'], self.fits[0].header['CTYPE2']]
        #self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
        self.wcs.wcs.cd = [[self.fits[0].header['CD1_1'], self.fits[0].header['CD1_2']],
                           [self.fits[0].header['CD2_1'], self.fits[0].header['CD2_2']]]
        self.wcs._naxis1 = self.fits[0].header['NAXIS1']
        self.wcs._naxis2 = self.fits[0].header['NAXIS2']

    def contains_position(self,ra,dec):
        try:
            cutout = Cutout2D(self.fits[0].data, SkyCoord(ra, dec, unit="deg", frame=self.frame), (1, 1),
                              wcs=self.wcs, copy=True)#,mode='partial')
        except:
            log.debug("position (%f, %f) is not in image." % (ra,dec), exc_info=False)
            return False
        return True

    def calc_pixel_size(self,wcs):
        return np.sqrt(wcs.wcs.cd[0, 0] ** 2 + wcs.wcs.cd[0, 1] ** 2) * 3600.0

    def get_vrange(self,vals):
        self.vmin = None
        self.vmax = None

        try:
            zscale = ZScaleInterval(contrast=0.25,krej=2.5) #nsamples=len(vals)
            self.vmin, self.vmax = zscale.get_limits(values=vals )
        except:
            log.info("Exception in science_image::get_vrange:",exc_info =True)

        return self.vmin,self.vmax

    def get_cutout(self,ra,dec,error,window=None,image=None,copy=False):
        '''ra,dec in decimal degrees. error and window in arcsecs'''
        #error is central box (+/- from ra,dec)
        #window is the size of the entire coutout
        #return a cutout

        self.window = None
        if (error is None or error == 0) and (window is None or window == 0):
            log.info("inavlid error box and window box")
            return None

        if window is None or window < error:
            window = float(max(2.0*error,5.0)) #should be at least 5 arcsecs

        self.window = window

        #data = None
        #pix_size = None
        #wcs = None
        if image is None:
            if self.fits is not None:
                #need to keep memory use down ... the statements below (esp. data = self.fit[0].data)
                #cause a copy into memory ... want to avoid it so, leave the code fragmented a bit as is
               # data = self.fits[0].data
               # pix_size = self.pixel_size
               # wcs = self.wcs

                try:
                    position = SkyCoord(ra, dec, unit="deg", frame=self.frame)

                    #sanity check
                    #x, y = skycoord_to_pixel(position, wcs=self.wcs, mode='all')
                    #x = x * self.pix_size
                    #y = y * self.pix_size

                    pix_window = int(np.ceil(window / self.pixel_size))  # now in pixels
                    log.debug("Collecting cutout size = %d square at RA,Dec = (%f,%f)" %(pix_window,ra,dec))
                    cutout = Cutout2D(self.fits[0].data, position, (pix_window, pix_window), wcs=self.wcs, copy=copy)
                    self.get_vrange(cutout.data)
                except NoOverlapError:
                    log.info("Exception (possible NoOverlapError) in science_image::get_cutout(). "
                             "Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                    print("Target is not in range of image. RA,Dec = (%f,%f) Window = %d" % (ra, dec, pix_window))
                    return None
                except:
                    log.error("Exception in science_image::get_cutout (%s):" %self.image_location, exc_info=True)
                    return None

                if not (self.contains_position(ra,dec)):
                    log.info("science image (%s) does not contain requested position: RA=%f , Dec=%f"
                             %(self.image_location,ra,dec))
                    return None
            else:
                log.error("No fits or passed image from which to make cutout.")
                return None
        else:
            #data = image.data
            #pix_size = self.calc_pixel_size(image.wcs)
            #wcs = image.wcs
            try:
                position = SkyCoord(ra, dec, unit="deg", frame='fk5')
                pix_window = window / self.calc_pixel_size(image.wcs)  # now in pixels
                cutout = Cutout2D(image.data, position, (pix_window, pix_window), wcs=image.wcs, copy=copy)
                self.get_vrange(cutout.data)
            except:
                log.error("Exception in science_image::get_cutout ():" , exc_info=True)
                return None

        return cutout

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