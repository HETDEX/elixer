"""
Check the galaxy mask to see if the target is part of a large galaxy
Based on work from John Feldmeier and Erin Cooper in HETDEX_API

"""

try:
    from elixer import global_config as G
    from elixer import utilities
except:
    import global_config as G
    import utilities

log = G.Global_Logger('galaxy_mask')
log.setlevel(G.LOG_LEVEL)

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
import astropy.table
from regions import EllipseSkyRegion, EllipsePixelRegion
import numpy as np
#from math import ceil


def create_dummy_wcs(coords, pixscale=0.5*u.arcsec, imsize=60.*u.arcmin):
    """
    From HETDEX_API:
    Create a simple fake WCS in order to use the regions subroutine.
    Adapted from John Feldmeiers galmask.py

    Parameters
    ----------
    coords: a SkyCoord object
        center coordinates of WCS
    pixscale: astropy quantity
        pixel scale of WCS in astropy angle quantity units
    imsize: astropy quantity
        size of WCS in astropy angle quanity units
    """

    gridsize = imsize.to_value('arcsec')
    gridstep = pixscale.to_value('arcsec')

    # Create coordinate center
    ra_cen = coords.ra.deg
    dec_cen = coords.dec.deg

    ndim = int(2 * gridsize / gridstep + 1)
    center = ndim / 2
    w = wcs.WCS(naxis=2)
    w.wcs.crval = [ra_cen, dec_cen]
    w.wcs.crpix = [center, center]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ['deg','deg']
    w.wcs.cdelt = [-gridstep / gridsize, gridstep / gridsize]
    w.array_shape = [ndim, ndim]

    return w

def create_ellreg(t, index, d25scale=1.0):
    """
    From HETDEX_API:
    Creates an elliptical sky region from astropy.regions, with info from RC3 catalog

    t - a table of galaxy regions, similar to that found in read_rc3_tables
    index - the value of the table to be used
    scale - a scaling factor.  Leaving at 1.0 means to use the D25 isophote,
    and it's likely we will want to scale this larger.
    """

    coords = SkyCoord(t["Coords"][index], frame="icrs")

    # The ellipse region uses the major and minor axes, so we have to multiply by
    # two first, before applying any user scaling.

    major = (t["SemiMajorAxis"][index]) * np.float64(2.0) * d25scale * u.arcmin
    minor = (t["SemiMinorAxis"][index]) * np.float64(2.0) * d25scale * u.arcmin
    pa = (t["PositionAngle"][index]) * u.deg
    ellipse_reg = EllipseSkyRegion(center=coords, height=major, width=minor, angle=pa)

    return ellipse_reg

class GalaxyMask():
    """
    wrapper for galaxy mask function
    hangs on to the astropy table of galaxy data
    """

    galaxy_table = None
    last_entry_idx = None

    def __init__(self):
        galmask_fn = None
        try:
            self.galaxy_table = astropy.table.Table.read( G.HETDEX_API_CONFIG.rc3cat)
            self.galaxy_table["SkyCoords"] = SkyCoord(self.galaxy_table["Coords"]) #build out column as SkyCoord for easier searching
        except:

            try:
                if 'hdr2.1' in G.HETDEX_API_CONFIG.rc3cat:
                    self.galaxy_table = astropy.table.Table.read(G.HETDEX_API_CONFIG.rc3cat.replace("hdr2.1","hdr4"))
                    self.galaxy_table["SkyCoords"] = SkyCoord(
                        self.galaxy_table["Coords"])  # build out column as SkyCoord for easier searching
                else:
                    #what to replace
                    self.galaxy_table = astropy.table.Table.read(G.HETDEX_API_CONFIG.rc3cat.replace("hdr"+G.HDR_Version,G.HDR_LAST_GOOD_Latest_Str))
                    self.galaxy_table["SkyCoords"] = SkyCoord(
                        self.galaxy_table["Coords"])  # build out column as SkyCoord for easier searching
            except:
                log.error("Unable to open galaxy mask table file. Galaxy mask unavailable.", exc_info=True)
                self.galaxy_table = None


    def redshift(self,ra,dec,d25scale=G.GALAXY_MASK_D25_SCALE):
        """
        From HETDEX_API:

        Find the nearest galaxy in the table in which the coordinates are inside the galaxy ellipse and return the
        redshift of that galaxy.
        If the ra,dec do not lie within any galaxy, return None
        If the ra,dec lie within multiple galaxies, return all unique redshifts (could be same entry more than once)

        :param ra:  in decimal degrees
        :param dec: in decimal degrees
        :param d25scale:  The scaling of ellipses.  1.0 means use the ellipse for D25.
                                Experimentation showed a value of 1.75 might be more appropriate
        :return: list of redshifts and the correspoding list of minimum D25 (integer) values
        """

        try:
            self.last_entry_idx = None
            target_coord = SkyCoord(ra,dec,unit='deg')
            target_wcs = create_dummy_wcs(target_coord)

            #no tight fixed distance as these are not point sources and can be big (e.g. M31 is about 5 degrees across)
            seps = target_coord.separation(self.galaxy_table['SkyCoords']).deg
            bids = np.argsort(seps)

            z = []
            min_scales = []
            scales = np.arange(1.0,int(d25scale)+1.0,1.0)

            for i in bids:
                if seps[i] > 5.0:
                    #we're done ... the rest are too far away for this to make any sense
                    break

                #using the HETDEX_API scaling ... could re-write and use a distance, but this is simple and fast
                #enough for now
                ellipse = create_ellreg(self.galaxy_table, i, d25scale=d25scale)
                if ellipse.contains(target_coord, target_wcs):
                    z.append(self.galaxy_table["NEDRedshift"][i])
                    min_scale = d25scale
                    #now, what is lowest integer of D25 scale that still hits
                    for scale in scales:
                        ellipse = create_ellreg(self.galaxy_table, i, d25scale=scale)
                        if ellipse.contains(target_coord, target_wcs):
                            min_scale = scale
                            break
                    min_scales.append(min_scale)

            if len(z) > 0:
                z = np.array(z)
                min_scales = np.array(min_scales)
                _, idx = np.unique(z, return_index=True)
                scales = []

                #return the smallest scale for each of the unique redshifts
                for i in idx:
                    try:
                        scales.append(min(min_scales[z==z[i]]))
                    except:
                        pass #might not be any

                self.last_entry_idx = idx
                return z[idx],scales

            else:
                return None, None
        except:
            log.warning("Exception in galaxy_mask::redshift()",exc_info=True)
            return None, None

