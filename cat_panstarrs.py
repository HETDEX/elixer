from __future__ import print_function


try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_base
    from elixer import match_summary
    from elixer import line_prob
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import line_prob


import os.path as op
import copy
import os
import ssl


import matplotlib
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import astropy.table
from astropy import coordinates as coords
from astropy import units as u
from astropy.table import Table
from astropy.io import fits

#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field



def panstarrs_count_to_mag(count,cutout=None,headers=None):
    #see https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images
    """
    Photometric calibration
    At the time of stacking, the pixel values in stacks are rescaled to a zero-point of 25 + 2.5* log10(exposure time),
    based on the input warp calibration (see keyword HIERARCH FPA.ZP in the FITS header - the individual warp
    zero-points are also in the header, ZPT_nnnn, as are the relative scaling factors applied to each, SCL_nnnn).
    However, as the final ubercalibration of the data has not taken place at this time, this zero-point may be
    slightly incorrect (usually at the hundredths of a magnitude level). In contrast, the stack fluxes/magnitudes in
    the PSPS catalog database have undergone the final calibration process and are more accurate.

    So to measure your own magnitudes off a stack image, you need to apply the following formula:

        MAG=-2.5*log10(sum(data-units)) +25+2.5*log10(EXPTIME)

    Note that due to the complicated nature of the stacking process, the data-units in a stack are not easily
    related to actual photons from the target.

    :param count:
    :param cutout:
    :param headers:
    :return:
    """

    try:
        exptime = headers[0]['EXPTIME']
        mag = -2.5*np.log10(count) + 25.0 + 2.5*np.log10(exptime)
    except:
        mag = 99.9

    if np.isnan(mag):
        mag = 99.9

    return mag



#helper fuctions from Pan-STARRS example: https://ps1images.stsci.edu/ps1image.html
import requests

def arcsec2pix(arcsec):
    #Pan-STARRS wants pixels for the size and FITS platescale is 0.25"/pix
    return int(arcsec/0.25) #must be an integer

def getimages(ra, dec, size=240, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, size=size, filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase + filename)
    return url


def getcolorim(ra, dec, size=240, output_size=None, filters="grizy", format="jpg"):
    """Get color image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    url = geturl(ra, dec, size=size, filters=filters, output_size=output_size, format=format, color=True)
    r = requests.get(url)
    im = Image.open(BytesIO(r.content))
    return im


def getgrayim(ra, dec, size=240, output_size=None, filter="g", format="jpg"):
    """Get grayscale image at a sky position

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filter = string with filter to extract (one of grizy)
    format = data format (options are "jpg", "png")
    Returns the image
    """

    if format not in ("jpg", "png"):
        raise ValueError("format must be jpg or png")
    if filter not in list("grizy"):
        raise ValueError("filter must be one of grizy")
    url = geturl(ra, dec, size=size, filters=filter, output_size=output_size, format=format)
    r = requests.get(url[0])
    im = Image.open(BytesIO(r.content))
    return im

def get_image(ra,dec,radius,filters):
    """

    :param ra: in decimal degrees (float)
    :param dec: in decimal degrees (float)
    :param radius:  in arcsec (float)
    :param filters: any of "grizy"
    :return:
    """

    hdulist = None

    #having some issues with PanSTARRS certificate, so just ignore and do not attempt to validate the certificate
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    try:
        size = arcsec2pix(radius)
        fitsurl = geturl(ra, dec, size=size, filters=filters, format="fits")
        hdulist = fits.open(fitsurl[0])
    except:
        log.error("Exception in cat_panstarrs.py::get_image",exc_info=True)

    return hdulist

class PANSTARRS(cat_base.Catalog):#Pan-STARRS
    """
    Online, on-demand only ... There is no direct archival catalog or imaging

PanSTARRS1 Quick Facts
Location	Haleakala, Hawaii
Telescope	1.8 m diameter
Field of view	3 degree diameter, 7 square degree FOV
Filters	g, r, i, z, y
Detectors	60 orthogonal transfer arrays
Surveys
3pi Steradian Survey,

Medium Deep Survey

Sky coverage	North of declination -30 degree
3pi stack
5s depth	grizy < 23.3, 23.2, 23.1, 22.3, 21.3
Single epoch
5s depth	grizy < 22.0, 21.8, 21.5, 20.9, 19.7
Saturation	12-14 mag, depends on seeing
Median seeing	grizy = 1.31, 1.19, 1.11, 1.07, 1.02 arcsec

    """


    # class variables
    CONT_EST_BASE = None

    MainCatalog = None #there is no Main Catalog ... must load individual catalog tracts
    Name = "Pan-STARRS"
    Filters = ['g','r','i','z','y'] #case is important ... needs to be lowercase
    WCS_Manual = True

    def __init__(self):
        super(PANSTARRS, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_unique = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0
        self.master_cutout = None

    def get_filter_flux(self, df):

        #todo:
        print("get_filter_flux not defined yet")
        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str


    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''
        #todo:
        print("build_list_of_bid_targets not defined yet")
        #todo: PANSTARRS may not really support targets anyway

        return 0, None, None



    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None,detobj=None):

        self.clear_pages()
        ras = []
        decs = []

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            entry = self.build_cat_summary_figure(cat_match,target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                                  detobj=detobj)

            if entry is not None:
                self.add_bid_entry(entry)
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")


        if G.SINGLE_PAGE_PER_DETECT: # and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
            entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                    target_ra=target_ra, target_dec=target_dec,
                                                                    target_w=target_w, target_flux=target_flux,
                                                                    detobj=detobj)
            if entry is not None:
                self.add_bid_entry(entry)
            else:
                return None
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")

        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):  # each bid taget gets its own line
            log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

        return self.pages


    def get_stacked_cutout(self,ra,dec,window):

        stacked_cutout = None
        error = window

        # for a given Tile, iterate over all filters
        tile, tract = self.find_target_tile(ra, dec)
        if tile is None:
            # problem
            print("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        for f in self.Filters:
            try:
                i = self.CatalogImages[
                    next(i for (i, d) in enumerate(self.CatalogImages)
                         if ((d['filter'] == f) and (d['tile'] == tile)))]
            except:
                i = None

            if i is None:
                continue

            try:
                wcs_manual = i['wcs_manual']
            except:
                wcs_manual = self.WCS_Manual

            try:
                if i['image'] is None:
                    i['image'] = science_image.science_image(wcs_manual=wcs_manual,wcs_idx=1,
                                                             image_location=op.join(i['path'], i['name']))
                sci = i['image']

                cutout, _, _, _ = sci.get_cutout(ra, dec, error, window=window, aperture=None, mag_func=None)
                #don't need pix_counts or mag, etc here, so don't pass aperture or mag_func

                if cutout is not None:  # construct master cutout
                    if stacked_cutout is None:
                        stacked_cutout = copy.deepcopy(cutout)
                        ref_exptime = sci.exptime
                        total_adjusted_exptime = 1.0
                    else:
                        stacked_cutout.data = np.add(stacked_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
            except:
                log.error("Error in get_stacked_cutout.",exc_info=True)

        return stacked_cutout

    def build_cat_summary_figure (self, cat_match, ra, dec, error, bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        query_radius = max(error*1.5,30.0) #query larger than strictly needed
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

        rows = 10
        #cols = 1 + len(self.CatalogImages)/len(self.Tiles)
        #note: setting size to 7 from 6 so they will be the right size (the 7th position will not be populated)
        cols = 7 # 1 for the fiber position and up to 5 filters for any one tile (u,g,r,i,z)

        fig_sz_x = 18 #cols * 3
        fig_sz_y = 3 #ows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        # All on one line now across top of plots
        # if G.ZOO:
        #     title = "Possible Matches = %d (within +/- %g\")" \
        #             % (len(self.dataframe_of_bid_targets_unique), error)
        # else:
        #     title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
        #             % (len(self.dataframe_of_bid_targets_unique), error)

        title = "Pan-STARRS imaging only. (mag limit g,r ~ 23.2) "

        cont_est = -1
        # if target_flux and self.CONT_EST_BASE:
        #     title += "  Minimum (no match) 3$\sigma$ rest-EW: "
        #     cont_est = self.CONT_EST_BASE*3
        #     if cont_est != -1:
        #         title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
        #         if target_w >= G.OII_rest:
        #             title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
        #         else:
        #             title = title + "  OII = N/A"
        #     else:
        #         title += "  LyA = N/A  OII = N/A"



        plt.subplot(gs[0, :])
        text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        ref_exptime = 1.0
        total_adjusted_exptime = 1.0
        bid_colors = self.get_bid_colors(len(bid_ras))
        exptime_cont_est = -1
        index = 0 #images go in positions 1+ (0 is for the fiber positions)

        best_plae_poii = None
        best_plae_poii_filter = '-'
        bid_target = None

        #pos = coords.SkyCoord(ra,dec,unit="deg",frame='icrs')

        for f in self.Filters:
            index += 1

            if index > cols:
                log.warning("Exceeded max number of grid spec columns.")
                break #have to be done

            try:
                wcs_manual = self.WCS_Manual
                aperture = 2.0
                mag_func = panstarrs_count_to_mag
            except:
                wcs_manual = self.WCS_Manual
                aperture = 0.0
                mag_func = None

            log.info("Pan-STARRS query (%f,%f) at %f arcsec for band %s ..." % (ra, dec, query_radius, f))
            hdulist = get_image(ra,dec,query_radius,f)

            if hdulist is None:
                log.info("Pan-STARRS query (%f,%f) at %f arcsec for band %s returned None" %(ra,dec,query_radius,f))
                continue

            sci = science_image.science_image(wcs_manual=wcs_manual, wcs_idx=0,
                                        image_location=None,hdulist=hdulist)

            #the filters are in order, use r if g is not there
            if (f == 'r') and (sci.exptime is not None) and (exptime_cont_est == -1):
                exptime_cont_est = sci.exptime

            # the filters are in order, so this will overwrite r
            if (f == 'g') and (sci.exptime is not None):
                exptime_cont_est = sci.exptime

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag, mag_radius = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func)
            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            try:  # update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None) and (f in 'gr'):
                    # make a "blank" catalog match (e.g. at this specific RA, Dec (not actually from catalog)
                    bid_target = match_summary.BidTarget()
                    bid_target.catalog_name = self.Name
                    bid_target.bid_ra = 666  # nonsense RA
                    bid_target.bid_dec = 666  # nonsense Dec
                    bid_target.distance = 0.0
                    bid_target.bid_filter = f
                    bid_target.bid_mag = mag
                    bid_target.bid_mag_err_bright = 0.0 #todo: right now don't have error on aperture mag
                    bid_target.bid_mag_err_faint = 0.0
                    bid_target.bid_flux_est_cgs_unc = 0.0

                    if mag < 99:
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag, target_w)
                    else:
                        bid_target.bid_flux_est_cgs = cont_est

                    bid_target.add_filter("Pan-STARRS", f, bid_target.bid_flux_est_cgs, -1)

                    addl_waves = None
                    addl_flux = None
                    addl_ferr = None
                    try:
                        addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                        addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                        addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                    except:
                        pass

                    lineFlux_err = 0.
                    if detobj is not None:
                        try:
                            lineFlux_err = detobj.estflux_unc
                        except:
                            lineFlux_err = 0.

                    #build EW error from lineFlux_err and aperture estimate error
                    ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                    try:
                        ew_obs_err =  abs(ew_obs * np.sqrt(
                                        (lineFlux_err / target_flux) ** 2 +
                                        (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                    except:
                        ew_obs_err = 0.

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = \
                        line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                                           ew_obs=ew_obs,
                                           lineFlux_err= lineFlux_err,
                                           ew_obs_err= ew_obs_err,
                                           c_obs=None, which_color=None, addl_fluxes=addl_flux,
                                           addl_wavelengths=addl_waves,addl_errors=addl_ferr,sky_area=None,
                                           cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                                           sigma=None)


                    if best_plae_poii is None or f == 'r':
                        best_plae_poii = bid_target.p_lae_oii_ratio
                        best_plae_poii_filter = f

                    cat_match.add_bid_target(bid_target)
            except:
                log.debug('Could not build exact location photometry info.', exc_info=True)
            #
            # if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
            #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g (%s)" % (bid_target.p_lae_oii_ratio, f))
            #




            if cutout is not None:  # construct master cutout

                # 1st cutout might not be what we want for the master (could be a summary image from elsewhere)
                if self.master_cutout:
                    if self.master_cutout.shape != cutout.shape:
                        del self.master_cutout
                        self.master_cutout = None

                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True)
                    if sci.exptime:
                        ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    try:
                        self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
                    except:
                        log.warning("Unexpected exception.", exc_info=True)

                _ = plt.subplot(gs[1:, index])

                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])

                plt.title("Pan-STARRS " + f)
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.plot(0, 0, "r+")

                if pix_counts is not None:
                    cx = sci.last_x0_center
                    cy = sci.last_y0_center
                    self.add_aperture_position(plt,mag_radius,mag,cx,cy)


                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))


        if (not G.ZOO) and (bid_target is not None) and (best_plae_poii is not None):
            text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g (%s)" % (best_plae_poii, best_plae_poii_filter))

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            return None  # empty (catch_all) will produce fiber locations
            # still need to plot relative fiber positions here
            # plt.subplot(gs[1:, 0])
            # return self.build_empty_cat_summary_figure(ra, dec, error, bid_ras, bid_decs, target_w=target_w,
            #                                            fiber_locs=fiber_locs)
        else:
            self.master_cutout.data /= total_adjusted_exptime

        plt.subplot(gs[1:, 0])
        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)

        # complete the entry
        plt.close()
        return fig


    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None,detobj=None):

        #todo: this is effectively NOT used for Pan-STARRS at this point as we are only grabbing imaging, not a catalog
        #todo: we will leave early with ras < 1 (adds the "row intentionally blank" line)
        rows = 1
        cols = 6

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)

        #col(0) = "labels", 1..3 = bid targets, 4..5= Zplot
        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        # entry text
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        #row labels
        plt.subplot(gs[0, 0])
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if len(ras) < 1:
            # per Karl insert a blank row
            text = "No matching targets in catalog.\nRow intentionally blank."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            plt.close()
            return fig
        elif (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):
            text = "Too many matching targets in catalog.\nIndividual target reports on followin pages."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            plt.close()
            return fig


        bid_colors = self.get_bid_colors(len(ras))

        if G.ZOO:
            text = "Separation\n" + \
                   "1-p(rand)\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "Est OII rest-EW\n" + \
                   "mag\n"
        else:
            text = "Separation\n" + \
                   "1-p(rand)\n" + \
                   "RA, Dec\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "Est OII rest-EW\n" + \
                   "mag\n" + \
                   "P(LAE)/P(OII)\n"


        plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)

        col_idx = 0
        target_count = 0
        # targets are in order of increasing distance
        for r, d in zip(ras, decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break
            col_idx += 1
            try: #DO NOT WANT _unique as that has wiped out the filters
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'r')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                text = ""

                if G.ZOO:
                    text = text + "%g\"\n%0.3f\n" \
                                  % (df['distance'].values[0] * 3600.,df['dist_prior'].values[0])
                else:
                    text = text + "%g\"\n%0.3f\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600.,df['dist_prior'].values[0],
                                    df['RA'].values[0], df['DEC'].values[0])

                text += "N/A\nN/A\n"  #dont have specz or photoz for HSC

                #todo: add flux (cont est)
                try:
                    filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = self.get_filter_flux(df)
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0
                    filter_mag = 0.0
                    filter_mag_bright = 0.0
                    filter_mag_faint = 0.0
                    filter_str = "NA"

                bid_target = None

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,target_w) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

                        if target_w >= G.OII_rest:
                            text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.OII_rest))
                        else:
                            text = text + "N/A\n"
                        try:
                            bid_target = match_summary.BidTarget()
                            bid_target.catalog_name = self.Name
                            bid_target.bid_ra = df['RA'].values[0]
                            bid_target.bid_dec = df['DEC'].values[0]
                            bid_target.distance = df['distance'].values[0] * 3600
                            bid_target.bid_flux_est_cgs = filter_fl_cgs
                            bid_target.bid_filter = filter_str
                            bid_target.bid_mag = filter_mag
                            bid_target.bid_mag_err_bright = filter_mag_bright
                            bid_target.bid_mag_err_faint = filter_mag_faint
                            bid_target.bid_flux_est_cgs_unc = filter_fl_err

                            addl_waves = None
                            addl_flux = None
                            addl_ferr = None
                            try:
                                addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                                addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                                addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                            except:
                                pass

                            lineFlux_err = 0.
                            if detobj is not None:
                                try:
                                    lineFlux_err = detobj.estflux_unc
                                except:
                                    lineFlux_err = 0.

                            # build EW error from lineFlux_err and aperture estimate error
                            ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                            try:
                                ew_obs_err = abs(ew_obs * np.sqrt(
                                    (lineFlux_err / target_flux) ** 2 +
                                    (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                            except:
                                ew_obs_err = 0.

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = \
                                line_prob.prob_LAE(wl_obs=target_w,
                                                   lineFlux=target_flux,
                                                   ew_obs=ew_obs,
                                                   lineFlux_err=lineFlux_err,
                                                   ew_obs_err=ew_obs_err,
                                                   c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                                                   addl_fluxes=addl_flux, addl_errors=addl_ferr, sky_area=None,
                                                   cosmo=None, lae_priors=None,
                                                   ew_case=None, W_0=None,
                                                   z_OII=None, sigma=None)
                            #dfx = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                            #                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

                            try:
                                bid_target.add_filter('HSC','R',filter_fl_cgs,filter_fl_err)
                            except:
                                log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                            cat_match.add_bid_target(bid_target)
                        except:
                            log.debug('Unable to build bid_target.',exc_info=True)


                else:
                    text += "N/A\nN/A\n"

                text = text + "%0.2f(%0.2f,%0.2f)\n" % (filter_mag, filter_mag_bright, filter_mag_faint)

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    text += "%0.3g\n" % (bid_target.p_lae_oii_ratio)
                else:
                    text += "\n"
            else:
                text = "%s\n%f\n%f\n" % ("--",r, d)

            plt.subplot(gs[0, col_idx])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font,color=bid_colors[col_idx-1])

            # fig holds the entire page

            #todo: photo z plot if becomes available
            plt.subplot(gs[0, 4:])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            text = "Photo z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        plt.close()
        return fig

    def get_single_cutout(self, ra, dec, window, catalog_image,aperture=None,filter=None):


        d = {'cutout':None,
             'hdu':None,
             'path':None,
             'filter':catalog_image['filter'],
             'instrument':catalog_image['instrument'],
             'mag':None,
             'aperture':None,
             'ap_center': None}

        try:
            wcs_manual = self.WCS_Manual
            aperture = 2.0
            mag_func = panstarrs_count_to_mag
        except:
            wcs_manual = self.WCS_Manual
            aperture = 0.0
            mag_func = None

        try:

            log.info("Pan-STARRS query (%f,%f) at %f arcsec for band %s ..." % (ra, dec, query_radius, f))
            hdulist = get_image(ra,dec,query_radius,f)

            if hdulist is None:
                log.info("Pan-STARRS query (%f,%f) at %f arcsec for band %s returned None" % (ra, dec, query_radius, f))
            else:
                # todo: choose the best image?
                sci = science_image.science_image(wcs_manual=wcs_manual, wcs_idx=0,
                                                  image_location=None, hdulist=hdulist)
                sci.catalog_name = "Pan-STARRS"
                sci.filter_name = filter

                d['path'] = "Pan-STARRS Online"
                d['hdu'] = sci.headers

                # to here, window is in degrees so ...
                window = 3600. * window

                cutout,pix_counts, mag, mag_radius = sci.get_cutout(ra, dec, error=window, window=window, aperture=aperture,
                                                 mag_func=mag_func,copy=True)
                # don't need pix_counts or mag, etc here, so don't pass aperture or mag_func

                if cutout is not None:  # construct master cutout
                    d['cutout'] = cutout
                    if (mag is not None) and (mag < 999):
                        d['mag'] = mag
                        d['aperture'] = mag_radius
                        d['ap_center'] = (sci.last_x0_center, sci.last_y0_center)
        except:
            log.error("Error in get_single_cutout.", exc_info=True)

        return d

    def get_cutouts(self,ra,dec,window,aperture=None):
        l = list()

        for f in self.Filters:
            l.append(self.get_single_cutout(ra,dec,window,i,aperture,filter=f))

        return l
