"""
Basic source extraction, derived from ELiXer implementation, but completely independent of ELiXer

Based on source extraction for python (sep)

Primarily two user interfaces (last functions defined):
  get_cutout(...)   : optional for the user to provide a fits from which to make a smaller 2D cutout and then pass
                    that cutout to find_objects()

  find_objects(...) : basically source extraction with a few limited options. Works on a 2D cutout and returns
                      a list of dictionaries that define positions, extents, orientations, and counts for detections.
                      The caller is responsible for converting counts to mags or fluxes as per the 2DCutout or FITS.
                      (note: for multiple catalogs, ELiXer does already have the conversions, so take a look at the
                       various cat_*.py files in the ELiXer project.)

"""

import sys
import numpy as np
import math

import sep #source extractor python module

import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.wcs


#######################################
# Helper functions
#######################################

def dist_to_ellipse(xp,yp,xc,yc,a,b,angle):
    """
    Find the distance to the nearest point ON the ellipse for point OUTSIDE the ellipse.

    NOTICE: this is COORDINATE DISTANCE ... does not attempt to adjust for underlying metric (i.e does not adjust
    for projection on sphere with RA distances as function of cos(declination).).

    :param xp: x coord of point
    :param yp: y coord of point
    :param xc: x coord of ellipse center
    :param yc: y coord of ellipse center
    :param a: major axis (radius)
    :param b: minor axis (radius)
    :param angle: rotation angle in radians from positive x axis (counter clockwise)
    Assumes lower left is 0,0
    :return: outside (True/False), distance to curve, distance to barycenter, nearest point on curve as (x,y)
            where outside also proxies for success ... if True, this should have worked, if False, not
    """

    try:
        #translate to center ellipse at 0,0
        original_xc = xc
        original_yc = yc
        xp = xp - xc
        yp = yp - yc
        xc,yc = 0,0

        #rotate to major axis along x
        angle = 2.*math.pi - angle   #want in clock-wise from positive x axis (rotation matrix below is clock-wise)
        cosa = math.cos(angle)
        sina = math.sin(angle)

        #xt = transformed xp coord where major axis is positive x-axis and minor is positive y-axis
        #yt = transformed yp coord
        xt = xp*cosa-yp*sina
        yt = xp*sina+yp*cosa


        dist_to_barycenter = np.sqrt(xt*xt+yt*yt)
        dist_to_curve = None
        inside = (((xt * xt) / (a * a)) + ((yt * yt) / (b * b))) <= 1

        if inside:
            #log.info("Point to ellipse, point is INSIDE the ellipse.")
            return not inside, None,dist_to_barycenter,(None,None)

        #calculate and return distance
        #no analytical solution? so approximate
        #modified from Johannes Peter (https://gist.github.com/JohannesMP)
        px = abs(xt)
        py = abs(yt)

        tx = 0.707
        ty = 0.707

        for x in range(0, 3):
            x = a * tx
            y = b * ty

            ex = (a * a - b * b) * tx ** 3 / a
            ey = (b * b - a * a) * ty ** 3 / b

            rx = x - ex
            ry = y - ey

            qx = px - ex
            qy = py - ey

            r = math.hypot(rx, ry)
            q = math.hypot(qx, qy)

            tx = min(1, max(0, (qx * r / q + ex) / a))
            ty = min(1, max(0, (qy * r / q + ey) / b))
            t = math.hypot(tx, ty)
            tx /= t
            ty /= t

        pt_on_curve =  (math.copysign(a * tx, xt), math.copysign(b * ty, yt))
        curve_to_barycenter = np.sqrt(pt_on_curve[0] * pt_on_curve[0] + pt_on_curve[1] * pt_on_curve[1])
        dist_to_curve = dist_to_barycenter - curve_to_barycenter

        #transform pt_on_curve back to ORIGINAL coordinate system
        #inverse rotation
        xt = pt_on_curve[0]*cosa+pt_on_curve[1]*sina
        yt = -1*pt_on_curve[0]*sina+pt_on_curve[1]*cosa
        #translate back
        xt += original_xc
        yt += original_yc
        pt_on_curve = (xt,yt)

        return not inside, dist_to_curve,dist_to_barycenter,pt_on_curve
    except Exception as E:
        #print(f"Exception in utilities. {E}")
        return False,None, None, None


def calc_pixel_size(wcs):
    if hasattr(wcs.wcs, 'cd'):
        return np.sqrt(wcs.wcs.cd[0, 0] ** 2 + wcs.wcs.cd[0, 1] ** 2) * 3600.0, []
    elif hasattr(wcs.wcs, 'cdelt'):  # like Pan-STARRS (assume both directions are the same)
        return abs(wcs.wcs.cdelt[0]) * 3600.0, []
    else:  # we have a problem
        return None, ["Warning! Unable to determine pixel scale. WCS does not have cd or cdelt keywords."]




#######################################
# Primary User Functions
#######################################

def get_cutout(coord, side, image, imgidx=0, wcsidx=None, wcs=None):
    """

    :param coord: astropy SkyCoord
    :param side: width (of one full side) in arcsec (float)
    :param image: fits filename or hdu
    :param imgidx: hdu index for the image you want to cutout (e.g. if multiframe fits)
    :param wcsidx: the hdu index that holds the WCS info; if None assumes the imgidx
    :param wcs: WCS if provided by the caller
    :return: astropy Cutout2D, array of status messages
    """

    try:
        cutout = None
        status = []

        # todo: extra validation of parameters
        if coord is None:
            status.append(f"Invalid coord.")
            return None, status

        if side is None or side < 0:
            status.append(f"Invalid side. {side}")
            return None, status

        if imgidx is None:
            imgidx = 0

        if isinstance(image, str):
            hdulist = fits.open(image)
        elif isintance(image, isinstance(hdu, fits.hdu.hdulist.HDUList)):
            hdulist = image
        else:
            status.append(f"Invalid image type passed: {type(image)}")
            return None, status

        if imgidx > len(hdulist):
            status.append(f"Invalid imgidx {imgidx}. HDU has only {len(hdulist)} elements.")
            return None, status

        if wcsidx is None:
            wcsidx = imgidx

        # get the wcs (automatically)
        try:
            wcs = astropy.wcs.WCS(hdulist[wcsidx].header, relax=astropy.wcs.WCSHDR_CD00i00j | astropy.wcs.WCSHDR_PC00i00j)
        except:
            wcs = None

        if wcs is None:
            # todo: if fails, load manually?

            if wcs is None:  # still None
                status.append("Unable to load WCS")
                return None, status

        pixel_size, *_ = calc_pixel_size(wcs)  # this is pixels per arcsec
        pix_window = int(np.ceil(side / pixel_size))

        # get the actual 2D Cutout
        cutout = Cutout2D(hdulist[imgidx].data, coord, (pix_window, pix_window),
                          wcs=wcs, copy=True, mode="partial", fill_value=0)

        return cutout, status

    except Exception as e:
        status.append(f"Exception in get_cutout(): {e}")
        return None, status


def find_objects(cutout, fixed_radius=None, det_thresh= 1.5, kron_mux = 2.5, correct_loss = False):
    """
    Basically, Source extractor.

    :param cutout: this is an astropy Cutout2D object
    :param fixed_radius: radius (in arcsec, as float) for a fixed circular aperture under which to take counts centered
                         at each detected object's barycenter
    :param det_thresh: multiple of the global background rms as minimum detection threshold trigger
    :param kron_mux: number of kron radii for the aperture; explicitly suported 2.0 and 2.5 (common to source extractor)
    :param correct_loss: adjust counts to approximate for aperture lost light. Optional ONLY for kron_mux 2.5 and 2.0.
                         otherwise is off. Always off for fixed radius (forced aperture).
    :return: array of source extractor dictionary objects, array of status messages
    """

    def initialize_dict():
        d = {}
        d['idx'] = None  # an index
        d['x'] = None  # x position in pixels (center is 0,0)
        d['y'] = None  # y position
        d['ra'] = None  # decimal degrees
        d['dec'] = None  # decimal degrees
        d['a'] = None  # major axis (arcsec)
        d['b'] = None
        d['theta'] = None
        d['background'] = None
        d['background_rms'] = None
        d['dist_baryctr'] = None  # distance in arcsec from the center of the image to the center of object ellipse
        d['dist_curve'] = None  # distance in arcsec from the center of the image to the nearest point on object ellipse
        d['flux_cts'] = None
        d['flux_cts_err'] = None
        d['flags'] = None

        return d



    img_objects = []  # array of dictionaries
    objects = None
    status = []
    # 2.5 should capture 94%+ of galaxy light with possibly larger error for faint sources
    # 2.0 should capture 90%+ of galaxy light with less errror

    if correct_loss:
        if kron_mux == 2.0:
            lost_light_correction = 1.0 / 0.90 #written this way to be clear and so can multiply later
        elif kron_mux == 2.5:
            lost_light_correction = 1.0 / 0.94 #written this way to be clear and so can multiply later
        else:
            lost_light_correction = 1.0
            status.append(f"Unsupported kron_mux ({kron_mux}). Lost light correction turned off.")
    else:
        lost_light_correction = 1.0
    # #status.append("Scanning cutout with source extractor ...")

    try:
        if (cutout is None) or (cutout.data is None):
            status.append("bad cutout")
            return img_objects, status

        cx, cy = cutout.center_cutout

        # the endianness of THIS system ... may not
        # necessarily be the endian encoding of the cutout data
        if sys.byteorder == 'big':
            data = cutout.data
            big_endian = True
        else:
            data = cutout.data.byteswap().newbyteorder()
            big_endian = False

        try:
            bkg = sep.Background(data)
        except Exception as e:
            if type(e) == ValueError:
                #status.append("sep.Background() value error. May be ENDIAN issue. Swapping...")
                try:
                    if not big_endian:
                        # the inverse of the above assignment (for zipped data the decompression may already handle the
                        # flip so doing it again would have put it in the wrong ENDIAN order
                        data = cutout.data
                    else:
                        data = cutout.data.byteswap().newbyteorder()

                    bkg = sep.Background(data)

                except Exception as e:
                    status.append(f"Exception {e}")
                    return img_objects, status

        data_sub = data - bkg
        data_err = bkg.globalrms  # use the background RMS as the error (assume sky dominated)
        #using 1.5x globalrms
        objects = sep.extract(data_sub, det_thresh, err=bkg.globalrms)

        selected_idx = -1
        inside_objs = []  # (index, dist_to_barycenter)
        outside_objs = []  # (index, dist_to_barycenter, dist_to_curve)

        map_idx = np.full(len(objects), -1)
        # holds the index of img_objects for each object's entry (if no image object, the value is -1)

        idx = -1

        pixel_size,*_ = calc_pixel_size(cutout.wcs)

        for obj in objects:
            idx += 1
            # NOTE: #3.* applied for the same reason as above ... a & b are given in kron isophotal diameters
            # so 6a/2 == 3a == radius needed for function

            success, dist2curve, dist2bary, pt = dist_to_ellipse(cx, cy, obj['x'], obj['y'],
                                                                 3. * obj['a'], 3. * obj['b'], obj['theta'])

            # copy to ELiXer img_objects
            d = initialize_dict()
            d['idx'] = idx
            # convert to image center as 0,0 (needed later in plotting) and to arcsecs
            d['x'] = (obj['x'] - cx) * pixel_size  # want as distance in arcsec so pixels * arcsec/pixel
            d['y'] = (obj['y'] - cy) * pixel_size
            # the 6.* factor is from source extractor using 6 isophotal diameters
            d['a'] = 6. * obj['a'] * pixel_size
            d['b'] = 6. * obj['b'] * pixel_size
            d['theta'] = obj['theta']
            d['background'] = bkg.globalback
            d['background_rms'] = bkg.globalrms
            d['dist_baryctr'] = dist2bary * pixel_size
            if success:
                d['dist_curve'] = dist2curve * pixel_size
            else:
                d['dist_curve'] = -1.0
            try:
                # now, get the flux
                # dd 2023-09-07 incompatibility with sep version 1.1+ and numpy means the
                # parameters need to all go in as arrays or lists
                kronrad, krflag = sep.kron_radius(data=data_sub,
                                                  x=[obj['x']], y=[obj['y']],
                                                  a=[obj['a']], b=[obj['b']], theta=[obj['theta']],
                                                  r=[6.0])  # ,
                # mask=None, maskthresh=0.0, seg_id=None, segmap=None)
                kronrad = kronrad[0]
                # not using krflag
                # r=6 == 6 isophotal radii ... source extractor always uses 6
                # minimum diameter = 3.5 (1.75 radius)
                radius = kronrad * np.sqrt(obj['a'] * obj['b'])
                if radius < 1.75:
                    radius = 1.75
                    flux, fluxerr, flag = sep.sum_circle(data_sub, [obj['x']], [obj['y']],
                                                         [radius], subpix=1, err=data_err)

                    lost_light_correction = 1.0
                    status.append(f"idx [{idx}] minimum radius set to {radius}. Lost light correction turned off.")
                else:
                    flux, fluxerr, flag = sep.sum_ellipse(data_sub, [obj['x']], [obj['y']],
                                                          [obj['a']], [obj['b']], [obj['theta']],
                                                          [kron_mux * kronrad], subpix=1, err=data_err)

                flux = flux[0] * lost_light_correction
                fluxerr = fluxerr[0] * lost_light_correction
                flag = flag[0]

            except Exception as e:
                try:
                    if e.args[0] == "invalid aperture parameters":
                        # log.debug(f"+++++ invalid aperture parameters")
                        pass  # do nothing ... not important
                    else:
                        status.append(f"idx [{idx}] Exception! {e}")
                except Exception as e:
                    status.append(f"idx [{idx}] Exception with source extractor. {e}")
                continue

            # flux, fluxerr, flag may be ndarrays but of size zero (a bit weird)
            flux = float(flux)
            fluxerr = float(fluxerr)
            flag = int(flag)

            d['flux_cts'] = flux  #lost light correction already applied above
            d['flux_cts_err'] = fluxerr
            d['flags'] = flag

            if fixed_radius is not None:
                # And now get the flux for a fixed radius aperture
                try:
                    # radius is in arcsec and we need pixels, so divide by pixel_size in arcsec/pixel
                    radius = fixed_radius / pixel_size
                    # now, get the flux
                    flux, fluxerr, flag = sep.sum_circle(data_sub, [obj['x']], [obj['y']],
                                                         [radius], subpix=1, err=data_err)

                    # reminder, you don't do lost light correction for forced aperture (that is up to the caller)
                    flux = flux[0]
                    fluxerr = fluxerr[0]
                    flag = flag[0]

                except Exception as e:
                    status.append("idx [{idx}] Exception with source extractor {e}")
                    continue

                flux = float(flux)
                fluxerr = float(fluxerr)
                flag = int(flag)

                d['fixed_aper_radius'] = fixed_radius
                d['fixed_aper_flux_cts'] = flux
                d['fixed_aper_flux_cts_err'] = fluxerr
                d['fixed_aper_flags'] = flag

            img_objects.append(d)
            map_idx[idx] = len(img_objects) - 1

            if success:  # this is outside
                outside_objs.append((idx, dist2bary, dist2curve))
            elif dist2bary is not None:
                inside_objs.append((idx, dist2bary))

    except Exception as e:
        status.append(f"Source Extractor call failed {e}")

    return img_objects, status