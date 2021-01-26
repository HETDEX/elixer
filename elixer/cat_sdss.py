from __future__ import print_function


try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_base
    from elixer import match_summary
    from elixer import line_prob
    from elixer import utilities
    from elixer import spectrum_utilities as SU
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import line_prob
    import utilities
    import spectrum_utilities as SU

from astropy.coordinates import SkyCoord
import os.path as op
import copy
import io

import matplotlib
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import astropy.table
from astroquery.sdss import SDSS as SDSS_API
from astropy import coordinates as coords
from astropy import units as u


#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field



def sdss_count_to_mag(count,cutout=None,headers=None):

    try:
        bunit = str(headers[0]['BUNIT'])
        # nmgy = float(headers[0]['NMGY'])
        # nmgyivar = float(headers[0]['NMGYIVAR'])

        if bunit.lower() != "nanomaggy":
            log.warning("Unexpected BUNIT for SDSS: %s" %(bunit))
            return 99.9
    except:
        log.error("Exception in sdss_count_to_mag", exc_info=True)
        return 99.9

    if count > 0:

        #I would have expected to 1e-9 nanoJy to Jy then divide by 3631, but it looks like they (SDSS) have
        #already done that and wrapped it into the nmgy conversion, so we use the nmgy convesion and do NOT
        #divide by 3631Jy in the log?
        # if isinstance(count, float):
        #     jy = nmgy * count * 3.361e-6
        # else:
        #     jy = nmgy * count.value * 3.361e-6
        #return -2.5 * np.log10(jy)

        #NOPE: ... SDSS is using weird units ... nanomaggy and the counts are already IN nanomaggy so
        #you do not need to convert
        #Per their documentation, this is the correct conversion
        if isinstance(count, float):
            ct_nmgy = count
        else:
            ct_nmgy = count.value

        return 22.5 - 2.5 * np.log10(ct_nmgy)
    else:
        return 99.9  # need a better floor

    return 99.9

class SDSS(cat_base.Catalog):#SDSS
    """
    Online, on-demand only ... There is no direct archival catalog or imaging
    """


    #maglimits to 95% completeness for point sources
# u
#     3551AA
#     22.0
# g
#     4686AA
#     22.2
# r
#     6165AA
#     22.2
# i
#     7481AA
#     21.3
# z
#     8931AA
#     20.5


    MAG_LIMIT = 22.5

    # class variables
    CONT_EST_BASE = None

    mean_FWHM = 1.67 #at 75% quartile for g-band ... varies also by filter, but this is good middle of the road limit
    MainCatalog = None #there is no Main Catalog ... must load individual catalog tracts
    Name = "SDSS"
    Filters = ['u','g','r','i','z'] #case is important ... needs to be lowercase
    WCS_Manual = True

    def __init__(self):
        super(SDSS, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_unique = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0
        self.master_cutout = None

    def get_filters(self,ra=None,dec=None):
        return ['u','g', 'r', 'i', 'z']

    def get_filter_flux(self, df):
        #todo:
        print("get_filter_flux not defined yet")
        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str


    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''
        #todo:
        print("build_list_of_bid_targets not defined yet")
        #todo: SDSS may not really support targets anyway

        return 0, None, None



    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None,detobj=None):

        self.clear_pages()
        ras = []
        decs = []

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            if G.BUILD_REPORT_BY_FILTER:
                #here we return a list of dictionaries (the "cutouts" from this catalog)
                return self.build_cat_summary_details(cat_match,target_ra, target_dec, error, ras, decs,
                                              target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                              detobj=detobj)
            else:
                entry = self.build_cat_summary_figure(cat_match,target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                                  detobj=detobj)


        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")
            return None

        if entry is not None:
            self.add_bid_entry(entry)

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

    # def get_stacked_cutout(self, ra, dec, window):
    #
    #     stacked_cutout = None
    #     error = window
    #
    #     cutouts = self.get_cutouts(ra, dec, window)
    #
    #     stacked_cutout = None
    #
    #     for c in cutouts:
    #         cutout = c['cutout']
    #         try:
    #             exptime = c['hdu']['exptime']
    #         except:
    #             exptime = 1.0
    #         if cutout is not None:  # construct master cutout
    #             if stacked_cutout is None:
    #                 stacked_cutout = copy.deepcopy(cutout)
    #                 ref_exptime = exptime
    #                 total_adjusted_exptime = 1.0
    #             else:
    #                 stacked_cutout.data = np.add(stacked_cutout.data, cutout.data * exptime / ref_exptime)
    #                 total_adjusted_exptime += exptime / ref_exptime
    #
    #     return stacked_cutout

    def build_cat_summary_figure (self, cat_match, ra, dec, error, bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        query_radius = max(error*1.5,30.0) #need to query at least 30" else SDSS is cranky
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

        title = "SDSS imaging only. (mag limit g,r ~ 22.2) "

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

        pos = coords.SkyCoord(ra,dec,unit="deg",frame='icrs')

        for f in self.Filters:
            index += 1

            if index > cols:
                log.warning("Exceeded max number of grid spec columns.")
                break #have to be done

            try:
                wcs_manual = self.WCS_Manual
                aperture = self.mean_FWHM*0.5 + 0.5 # since a radius, half the FWHM + 0.5" for astrometric error
                mag_func = sdss_count_to_mag
            except:
                wcs_manual = self.WCS_Manual
                aperture = 0.0
                mag_func = None

            log.info("SDSS query (%f,%f) at %f arcsec for band %s ..." % (ra, dec, query_radius, f))
            hdulist_array = SDSS_API.get_images(coordinates=pos, radius=query_radius*u.arcsec,band=f)

            if hdulist_array is None:
                log.info("SDSS query (%f,%f) at %f arcsec for band %s returned None" %(ra,dec,query_radius,f))
                continue

            #todo: choose the best image? longest exp time or most NMGY?
            best_idx = 0
            #for i in range(len(hdulist_array)):



            sci = science_image.science_image(wcs_manual=wcs_manual, wcs_idx=0,
                                        image_location=None,hdulist=hdulist_array[best_idx])

            #the filters are in order, use r if g is not there
            if (f == 'r') and (sci.exptime is not None) and (exptime_cont_est == -1):
                exptime_cont_est = sci.exptime

            # the filters are in order, so this will overwrite r
            if (f == 'g') and (sci.exptime is not None):
                exptime_cont_est = sci.exptime

            # sci.load_image(wcs_manual=True)

            # if we are down to SDSS, take what you can get (plus the SDSS resolution is low to the point
            # the we can trip the empty image condition when it is not warranted
            save_ALLOW_EMPTY_IMAGE = G.ALLOW_EMPTY_IMAGE
            G.ALLOW_EMPTY_IMAGE = True

            cutout, pix_counts, mag, mag_radius, details = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func,return_details=True)

            G.ALLOW_EMPTY_IMAGE = save_ALLOW_EMPTY_IMAGE

            if (self.MAG_LIMIT < mag < 100) and (mag_radius > 0):
                details['fail_mag_limit'] = True
                details['raw_mag'] = mag
                details['raw_mag_bright'] = details['mag_bright']
                details['raw_mag_faint'] = details['mag_faint']
                details['raw_mag_err'] = details['mag_err']
                log.warning(f"Cutout mag {mag} greater than limit {self.MAG_LIMIT}. Setting to limit.")
                mag = self.MAG_LIMIT
                if details:
                    details['mag'] = mag
                    try:
                        details['mag_bright'] = min(mag,details['mag_bright'])
                    except:
                        details['mag_bright'] = mag
                    try:
                        details['mag_faint'] = max(mag,G.MAX_MAG_FAINT)
                    except:
                        details['mag_faint'] = G.MAX_MAG_FAINTs

            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            bid_target = None
            cutout_ewr = None
            cutout_ewr_err = None
            cutout_plae = None

            try:  # update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None) and (f == 'r'):
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
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag, SU.filter_iso(f,target_w))
                        try:
                            flux_faint = None
                            flux_bright = None

                            if details['mag_faint'] < 99:
                                flux_faint = self.obs_mag_to_cgs_flux(details['mag_faint'], SU.filter_iso(f,target_w))

                            if details['mag_bright'] < 99:
                                flux_bright = self.obs_mag_to_cgs_flux(details['mag_bright'], SU.filter_iso(f,target_w))

                            if flux_bright and flux_faint:
                                bid_target.bid_flux_est_cgs_unc = max((bid_target.bid_flux_est_cgs - flux_faint),
                                                                      (flux_bright - bid_target.bid_flux_est_cgs))
                            elif flux_bright:
                                bid_target.bid_flux_est_cgs_unc = flux_bright - bid_target.bid_flux_est_cgs

                        except:
                            pass

                    else:
                        bid_target.bid_flux_est_cgs = cont_est

                    try:
                        bid_target.bid_mag_err_bright = mag - details['mag_bright']
                        bid_target.bid_mag_err_faint = details['mag_faint'] - mag
                    except:
                        pass

                    bid_target.add_filter("SDSS", f, bid_target.bid_flux_est_cgs, -1)

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

                    # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                    #     line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                    #                        ew_obs=ew_obs,
                    #                        lineFlux_err= lineFlux_err,
                    #                        ew_obs_err= ew_obs_err,
                    #                        c_obs=None, which_color=None, addl_fluxes=addl_flux,
                    #                        addl_wavelengths=addl_waves,addl_errors=addl_ferr,sky_area=None,
                    #                        cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                    #                        sigma=None,estimate_error=True)

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                        line_prob.mc_prob_LAE(
                            wl_obs=target_w,
                            lineFlux=target_flux,
                            lineFlux_err=lineFlux_err,
                            continuum=bid_target.bid_flux_est_cgs,
                            continuum_err=bid_target.bid_flux_est_cgs_unc,
                            c_obs=None, which_color=None,
                            addl_wavelengths=addl_waves,
                            addl_fluxes=addl_flux,
                            addl_errors=addl_ferr,
                            sky_area=None,
                            cosmo=None, lae_priors=None,
                            ew_case=None, W_0=None,
                            z_OII=None, sigma=None)

                    try:
                        if plae_errors:
                            bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
                            bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
                    except:
                        pass

                    cutout_plae = bid_target.p_lae_oii_ratio
                    cutout_ewr = ew_obs / (1. + target_w / G.LyA_rest)
                    cutout_ewr_err = ew_obs_err / (1. + target_w / G.LyA_rest)

                    # if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)" % (bid_target.p_lae_oii_ratio,f))

                    if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                        try:
                            text.set_text(
                                text.get_text() + "  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$ (%s)" %
                                (round(bid_target.p_lae_oii_ratio, 3),
                                 round(bid_target.p_lae_oii_ratio_max, 3),
                                 round(bid_target.p_lae_oii_ratio_min, 3),
                                 f))
                        except:
                            log.debug("Exception adding PLAE with range", exc_info=True)
                            try:
                                text.set_text(
                                    text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)" % (bid_target.p_lae_oii_ratio, f))
                            except:
                                text.set_text(
                                    text.get_text() + "  P(LAE)/P(OII): (%s) (%s)" % ("---", f))

                    cat_match.add_bid_target(bid_target)
                    try:  # no downstream edits so they can both point to same bid_target
                        detobj.bid_target_list.append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to detobj.", exc_info=True)
            except:
                log.debug('Could not build exact location photometry info.', exc_info=True)



            if cutout is not None:  # construct master cutout
                # 1st cutout might not be what we want for the master (could be a summary image from elsewhere)
                if self.master_cutout:
                    if self.master_cutout.shape != cutout.shape:
                        del self.master_cutout
                        self.master_cutout = None

                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    # if we are down to SDSS, take what you can get (plus the SDSS resolution is low to the point
                    # the we can trip the empty image condition when it is not warranted
                    save_ALLOW_EMPTY_IMAGE = G.ALLOW_EMPTY_IMAGE
                    G.ALLOW_EMPTY_IMAGE = True

                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True)

                    G.ALLOW_EMPTY_IMAGE = save_ALLOW_EMPTY_IMAGE

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

                plt.title("SDSS " + f)
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                #plt.plot(0, 0, "r+")
                self.add_zero_position(plt)

                if pix_counts is not None:
                    details['catalog_name'] = self.name
                    details['filter_name'] = f
                    details['aperture_eqw_rest_lya'] = cutout_ewr
                    details['aperture_eqw_rest_lya_err'] = cutout_ewr_err
                    details['aperture_plae'] = cutout_plae
                    try:
                        if plae_errors:
                            details['aperture_plae_min'] = plae_errors['ratio'][1]
                            details['aperture_plae_max'] = plae_errors['ratio'][2]
                    except:
                        details['aperture_plae_min'] = None
                        details['aperture_plae_max'] = None

                    cx = sci.last_x0_center
                    cy = sci.last_y0_center
                    if (details['sep_objects'] is not None): # and (details['sep_obj_idx'] is not None):
                        self.add_elliptical_aperture_positions(plt,details['sep_objects'],details['sep_obj_idx'],
                                                               mag_radius,mag,cx,cy,cutout_ewr,cutout_plae)
                    else:
                        self.add_aperture_position(plt, mag_radius, mag, cx, cy, cutout_ewr, cutout_plae)


                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    self.add_catalog_position(plt,
                                              x=(fx-x)-target_box_side / 2.0,
                                              y=(fy-y)-target_box_side / 2.0,
                                              size=target_box_side, color=bc)
                    # plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                    #                                   width=target_box_side, height=target_box_side,
                    #                                   angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))

            if (details is not None) and (detobj is not None):
                detobj.aperture_details_list.append(details)
        #if False:
        #    if target_flux is not None:
        #        #todo: get exptime from the tile (science image has it)
        #        cont_est = self.get_f606w_max_cont(exptime_cont_est, 3,self.CONT_EST_BASE)
        #        if cont_est != -1:
        #            title += "Minimum (no match)\n  3$\sigma$ rest-EW:\n"
        #            title += "  LyA = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.LyA_rest))
        #            if target_w >= G.OII_rest:
        #                title = title + "  OII = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.OII_rest))
        #            else:
        #                title = title + "  OII = N/A\n"

            #plt.subplot(gs[0, 0])
            #plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
            #plt.gca().set_frame_on(False)
            #plt.gca().axis('off')

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

        # get zoo style cutout as png
        if G.ZOO_MINI and (detobj is not None):
            plt.figure()
            self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout, unlabeled=True)

            plt.gca().set_axis_off()

            box_ratio = 1.0#0.99
            # add window outline
            xl, xr = plt.gca().get_xlim()
            yb, yt = plt.gca().get_ylim()
            zero_x = (xl + xr) / 2.
            zero_y = (yb + yt) / 2.
            rx = (xr - xl) * box_ratio / 2.0
            ry = (yt - yb) * box_ratio / 2.0

            plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2 , height=ry * 2,
                                              angle=0, color='red', fill=False,linewidth=8))

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300,transparent=True)
            detobj.image_cutout_fiber_pos = buf
            plt.close()

        return fig


    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None,detobj=None):

        #todo: this is effectively NOT used for SDSS at this point as we are only grabbing imaging, not a catalog
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
                   "Match score\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "mag\n\n"
        else:
            text = "Separation\n" + \
                   "Match score\n" + \
                   "RA, Dec\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
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
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'r')]
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
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        #text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
                        # assumes no error in wavelength or c

                        # try:
                        #     ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        #     ew_u = abs(ew * np.sqrt(
                        #                 (detobj.estflux_unc / target_flux) ** 2 +
                        #                 (filter_fl_err / filter_fl) ** 2))
                        #     text = text + utilities.unc_str((ew,ew_u)) + "$\AA$\n"
                        # except:
                        #     log.debug("Exception computing catalog EW: ",exc_info=True)
                        #     text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        # if target_w >= G.OII_rest:
                        #     text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.OII_rest))
                        # else:
                        #     text = text + "N/A\n"
                        try:
                            bid_target = match_summary.BidTarget()
                            bid_target.catalog_name = self.Name
                            bid_target.bid_ra = df['RA'].values[0]
                            bid_target.bid_dec = df['DEC'].values[0]
                            bid_target.distance = df['distance'].values[0] * 3600
                            bid_target.prob_match = df['dist_prior'].values[0]
                            bid_target.bid_flux_est_cgs = filter_fl_cgs
                            bid_target.bid_filter = filter_str
                            bid_target.bid_mag = filter_mag
                            bid_target.bid_mag_err_bright = filter_mag_bright
                            bid_target.bid_mag_err_faint = filter_mag_faint
                            bid_target.bid_flux_est_cgs_unc = filter_fl_cgs_unc

                            try:
                                ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                                ew_u = abs(ew * np.sqrt(
                                    (detobj.estflux_unc / target_flux) ** 2 +
                                    (filter_fl_err / filter_fl) ** 2))

                                bid_target.bid_ew_lya_rest = ew
                                bid_target.bid_ew_lya_rest_err = ew_u

                                text = text + utilities.unc_str((ew, ew_u)) + "$\AA$\n"
                            except:
                                log.debug("Exception computing catalog EW: ", exc_info=True)
                                text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

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

                            # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                            #     line_prob.prob_LAE(wl_obs=target_w,
                            #                        lineFlux=target_flux,
                            #                        ew_obs=ew_obs,
                            #                        lineFlux_err=lineFlux_err,
                            #                        ew_obs_err=ew_obs_err,
                            #                        c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                            #                        addl_fluxes=addl_flux, addl_errors=addl_ferr, sky_area=None,
                            #                        cosmo=None, lae_priors=None,
                            #                        ew_case=None, W_0=None,
                            #                        z_OII=None, sigma=None, estimate_error=True)
                            #dfx = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                            #                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                                line_prob.mc_prob_LAE(
                                    wl_obs=target_w,
                                    lineFlux=target_flux,
                                    lineFlux_err=lineFlux_err,
                                    continuum=bid_target.bid_flux_est_cgs,
                                    continuum_err=bid_target.bid_flux_est_cgs_unc,
                                    c_obs=None, which_color=None,
                                    addl_wavelengths=addl_waves,
                                    addl_fluxes=addl_flux,
                                    addl_errors=addl_ferr,
                                    sky_area=None,
                                    cosmo=None, lae_priors=None,
                                    ew_case=None, W_0=None,
                                    z_OII=None, sigma=None)

                            try:
                                if plae_errors:
                                    bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
                                    bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
                            except:
                                pass

                            try:
                                bid_target.add_filter('HSC','R',filter_fl_cgs,filter_fl_err)
                            except:
                                log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                            cat_match.add_bid_target(bid_target)
                            try:  # no downstream edits so they can both point to same bid_target
                                if detobj is not None:
                                    detobj.bid_target_list.append(bid_target)
                            except:
                                log.warning("Unable to append bid_target to detobj.", exc_info=True)
                        except:
                            log.debug('Unable to build bid_target.',exc_info=True)


                else:
                    text += "N/A\nN/A\n"


                try:
                    text = text + "%0.2f(%0.2f,%0.2f)\n" % (filter_mag, filter_mag_bright, filter_mag_faint)
                except:
                    log.warning("Magnitude info is none: mag(%s), mag_bright(%s), mag_faint(%s)"
                                % (filter_mag, filter_mag_bright, filter_mag_faint))
                    text += "No mag info\n"

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    try:
                        text += r"$%0.4g\ ^{%.4g}_{%.4g}$" % (utilities.saferound(bid_target.p_lae_oii_ratio, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_max, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_min, 3))
                        text += "\n"
                    except:
                        text += "%0.4g\n" % (bid_target.p_lae_oii_ratio)
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

    def get_single_cutout(self, ra, dec, window, catalog_image,aperture=None,filter=None,error=None,do_sky_subtract=True):


        d = {'cutout':None,
             'hdu':None,
             'path':None,
             'filter':filter,
             'instrument':'SDSS',
             'mag':None,
             'aperture':None,
             'ap_center': None,
             'mag_limit':None,
             'details': None}

        try:
            wcs_manual = self.WCS_Manual
            if aperture is None:
                aperture = self.mean_FWHM * 0.5 + 0.5 # since a radius, half the FWHM + 0.5" for astrometric error
            mag_func = sdss_count_to_mag
        except:
            wcs_manual = self.WCS_Manual
            aperture = 0.0
            mag_func = None

        query_radius = max(window*1.5, 30.0)

        pos = SkyCoord(ra,dec,unit='deg')

        try:

            log.info("SDSS query (%f,%f) at %f arcsec for band %s ..." % (ra, dec, query_radius, filter))
            hdulist_array = SDSS_API.get_images(coordinates=pos, radius=query_radius * u.arcsec, band=filter)

            if hdulist_array is None:
                log.info("SDSS query (%f,%f) at %f arcsec for band %s returned None" % (ra, dec, query_radius, filter))
            else:
                # todo: choose the best image?
                sci = science_image.science_image(wcs_manual=wcs_manual, wcs_idx=0,
                                                  image_location=None, hdulist=hdulist_array[0])
                sci.catalog_name = "SDSS"
                sci.filter_name = filter

                d['path'] = "SDSS Online"
                d['hdu'] = sci.headers

                # to here, window is in degrees so ...
                window = 3600. * window
                if not error:
                    error = window

                #if we are down to SDSS, take what you can get (plus the SDSS resolution is low to the point
                #the we can trip the empty image condition when it is not warranted
                save_ALLOW_EMPTY_IMAGE = G.ALLOW_EMPTY_IMAGE
                G.ALLOW_EMPTY_IMAGE = True

                cutout, pix_counts, mag, mag_radius, details = sci.get_cutout(ra, dec, error=error, window=window,
                                                                              aperture=aperture,
                                                                              mag_func=mag_func, copy=True,
                                                                              return_details=True)

                G.ALLOW_EMPTY_IMAGE = save_ALLOW_EMPTY_IMAGE
                # don't need pix_counts or mag, etc here, so don't pass aperture or mag_func

                if cutout is not None:  # construct master cutout
                    d['cutout'] = cutout
                    details['catalog_name']=self.name
                    details['filter_name']=catalog_image['filter']
                    d['mag_limit']=self.get_mag_limit(None,mag_radius*2.)
                    if (mag is not None) and (mag < 999):
                        d['mag'] = mag
                        d['aperture'] = mag_radius
                        d['ap_center'] = (sci.last_x0_center, sci.last_y0_center)
                        d['details'] = details
        except:
            log.error("Error in get_single_cutout.", exc_info=True)

        return d

    def get_cutouts(self,ra,dec,window,aperture=None,filter=None,first=None,error=None,do_sky_subtract=True):
        l = list()

        #filters are fixed

        if filter:
            outer = filter
            inner = self.Filters
        else:
            outer = self.Filters
            inner = None

        wild_filters = iter(self.Filters)

        if outer:
            for f in outer:
                try:
                    if f == '*':
                        f = next(wild_filters, None)
                        if f is None:
                            break
                    elif inner and (f not in inner):
                        # if filter list provided but the image is NOT in the filter list go to next one
                        continue

                    cutout = self.get_single_cutout(ra, dec, window, None, aperture,filter=f,error=error)
                    if first:
                        if cutout['cutout'] is not None:
                                l.append(cutout)
                                break
                    else:
                        # if we are not escaping on the first hit, append ALL cutouts (even if no image was collected)
                        l.append(cutout)
                except:
                    log.error("Exception! collecting image cutouts.", exc_info=True)
        else:
            for f in self.Filters:
                try:
                    l.append(self.get_single_cutout(ra,dec,window,None,aperture,filter=f))
                except:
                    log.error("Exception! collecting image cutouts.", exc_info=True)


        return l

    # def get_cutouts(self,ra,dec,window,aperture=None):
    #     l = list()
    #
    #     for f in self.Filters:
    #         l.append(self.get_single_cutout(ra,dec,window,i,aperture,filter=f))
    #
    #     return l
