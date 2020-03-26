from __future__ import print_function

from elixer import global_config as G
import os.path as op
import copy



import matplotlib
matplotlib.use('agg')

from elixer import science_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from elixer import line_prob


#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

#pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

from elixer import cat_base
from elixer import match_summary


RA_NUDGE =  -2.5/3600.#-25./3600.
DEC_NUDGE = -2.0/3600. #-7./3600.


def shela_count_to_mag(count,cutout=None,sci_image=None):
    if count is not None:
        if sci_image is not None:
            #get the conversion factor, each tile is different
            try:
                fluxcon = float(sci_image[0].header['fluxcon'])
                magadd = float(sci_image[0].header['magadd'])
            except:
                #gain = 1.0
                log.error("Exception in shela_count_to_mag",exc_info=True)
                return 99.9

        if count > 0:

            #note: either should give the exact same result as fluxcon is calculated from magadd

            if magadd != 0:
                return -2.5 * np.log10(count) + magadd
            else:
                return -2.5 * np.log10(count * fluxcon)
        else:
            return 99.9  # need a better floor

class AST376_SHELA(cat_base.Catalog):

    # class variables
    MainCatalog = None
    Name = "AST376_SHELA"

    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min':36.373517 , 'RA_max': 37.035425, 'Dec_min': -0.002608, 'Dec_max': 0.756604}

    WCS_Manual = False

    CatalogImages = [
        {'path': G.AST376_PATH,
         'name': 'stack_crop_SHELA.fits',
         'filter': 'r',
         'instrument': 'LF1',
         'cols': None,
         'labels':None,
         'image': None,
         'expanded': False,
         'wcs_manual': False,
         'aperture': 5.0,
         'mag_func': shela_count_to_mag
         }
    ]


    def __init__(self):
        super(AST376_SHELA, self).__init__()

        # self.table_of_bid_targets = None
        self.num_targets = 0

        # do this only as needed
        # self.read_main_catalog()
        # self.read_photoz_catalog()
        # self.build_catalog_images() #will just build on demand

        self.master_cutout = None

    # todo: is this more efficient? garbage collection does not seem to be running
    # so building as needed does not seem to help memory
    def build_catalog_images(self):
        for i in self.CatalogImages:  # i is a dictionary
            i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                     image_location=op.join(i['path'], i['name']))

    @classmethod
    def read_photoz_catalog(cls):
        pass

    @classmethod
    def read_catalog(cls, catalog_loc, name):
        pass
        return None


    def build_list_of_bid_targets(self, ra, dec, error):

        return 0,None,None

        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''
        # NO dataframes for this catalog ... AST376 assignment
        # if self.df is None:
        #     self.read_main_catalog()
        # if self.df_photoz is None:
        #     self.read_photoz_catalog()

        error_in_deg = np.float64(error) / 3600.0

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0

        coord_scale = np.cos(np.deg2rad(dec))

        #can't actually happen for this catalog
        if coord_scale < 0.1: #about 85deg
            print("Warning! Excessive declination (%f) for this method of defining error window. Not supported" %(dec))
            log.error("Warning! Excessive declination (%f) for this method of defining error window. Not supported" %(dec))
            return 0,None,None

        ra_min = np.float64(ra - error_in_deg/coord_scale)
        ra_max = np.float64(ra + error_in_deg/coord_scale)
        dec_min = np.float64(dec - error_in_deg)
        dec_max = np.float64(dec + error_in_deg)

        log.info(self.Name + " searching for bid targets in range: RA [%f +/- %f], Dec [%f +/- %f] ..."
                 % (ra, error_in_deg, dec, error_in_deg))

        # try:
        #     self.dataframe_of_bid_targets = \
        #         self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
        #                 (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)].copy()
        #
        #     # ID matches between both catalogs
        #     self.dataframe_of_bid_targets_photoz = \
        #         self.df_photoz[(self.df_photoz['ID'].isin(self.dataframe_of_bid_targets['ID']))].copy()
        # except:
        #     log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.dataframe_of_bid_targets is not None:
            self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra, dec)

            log.info(self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
                     ". Found = %d" % (self.num_targets))

        return self.num_targets, self.dataframe_of_bid_targets, self.dataframe_of_bid_targets_photoz

    # column names are catalog specific, but could map catalog specific names to generic ones and produce a dictionary?
    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None,target_flux=None):

        self.clear_pages()

        #there is no matching catalog ... this is just imaging so don't bother with an empty catalog

        ras = []
        decs = []

        self.dataframe_of_bid_targets = []

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            entry = self.build_cat_summary_figure(cat_match, target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux, )
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")

        if entry is not None:
            self.add_bid_entry(entry)

        if G.SINGLE_PAGE_PER_DETECT:
            entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                    target_ra=target_ra, target_dec=target_dec,
                                                                    target_w=target_w, target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)

        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):  # each bid taget gets its own line
            log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

        return self.pages


    def get_stacked_cutout(self,ra,dec,window):

        stacked_cutout = None
        error = window

        for i in self.CatalogImages:  # i is a dictionary
            try:
                wcs_manual = i['wcs_manual']
            except:
                wcs_manual = self.WCS_Manual

            try:
                if i['image'] is None:
                    i['image'] = science_image.science_image(wcs_manual=wcs_manual,
                                                             image_location=op.join(i['path'], i['name']))
                sci = i['image']

                cutout, _, _, _ = sci.get_cutout(ra + RA_NUDGE, dec + DEC_NUDGE, error, window=window, aperture=None, mag_func=None)
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

    def build_cat_summary_figure (self, cat_match,ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

        # set a minimum window size?
        # if window < 8:
        #    window = 8

        rows = 10 #2 (use 0 for text and 1: for plots)
        #7 is the current miniumum to make the spacing look right
        cols = 7 # 1+ len(self.CatalogImages) #(use 0 for master_stacked and 1 - N for filters)

        fig_sz_x = 18 #cols * 3 # was 6 cols
        fig_sz_y = 3 #rows * 3 # was 1 or 2 rows

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.05)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        #All on one line now across top of plots
        if G.ZOO:
            title = "Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets), error)
        else:
            title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets), error)

        cont_est = -1
        if target_flux is not None:
           # cont_est = self.CONT_EST_BASE*3 #self.get_f606w_max_cont(self.EXPTIME_F606W, 3, self.CONT_EST_BASE)
            if cont_est != -1:
                title += "  Minimum (no match) 3$\sigma$ rest-EW: "
                title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A"

       # title += "\n"

        plt.subplot(gs[0, :])
        text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        ref_exptime = None
        total_adjusted_exptime = None

        # add the bid targets
        bid_colors = self.get_bid_colors(len(bid_ras))

        index = 0 #start in the 2nd box which is index 1 (1st box is for the fiber position plot)
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

            try:
                wcs_manual = i['wcs_manual']
                aperture = i['aperture']
                mag_func = i['mag_func']
            except:
                wcs_manual = self.WCS_Manual
                aperture = 1.0
                mag_func = None


            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag, mag_radius = sci.get_cutout(ra + RA_NUDGE, dec + DEC_NUDGE, error, window=window,
                                                     aperture=aperture, mag_func=mag_func)

            try: #update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None):
                    #make a "blank" catalog match (e.g. at this specific RA, Dec (not actually from catalog)
                    bid_target = match_summary.BidTarget()
                    bid_target.catalog_name = self.Name
                    bid_target.bid_ra = 666 #nonsense RA
                    bid_target.bid_dec = 666 #nonsense Dec
                    bid_target.distance = 0.0
                    if mag < 99:
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag,target_w)
                    else:
                        bid_target.bid_flux_est_cgs = cont_est

                    bid_target.add_filter(i['instrument'],i['filter'],bid_target.bid_flux_est_cgs,-1)

                    addl_waves = None
                    addl_flux = None
                    addl_ferr = None
                    try:
                        addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                        addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                        addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                    except:
                        pass

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = \
                        line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                                           ew_obs=(target_flux / bid_target.bid_flux_est_cgs),
                                           c_obs=None, which_color=None, addl_fluxes=addl_flux,
                                           addl_wavelengths=addl_waves,addl_errors=addl_ferr,sky_area=None,
                                           cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                                           sigma=None)

                    if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                        text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g" % (bid_target.p_lae_oii_ratio))

                    cat_match.add_bid_target(bid_target)
            except:
                log.debug('Could not build exact location photometry info.',exc_info=True)


            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            # 1st cutout might not be what we want for the master (could be a summary image from elsewhere)
            if self.master_cutout:
                if self.master_cutout.shape != cutout.shape:
                    del self.master_cutout
                    self.master_cutout = None

            if cutout is not None:  # construct master cutout
                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout,_,_, _ = sci.get_cutout(ra + RA_NUDGE, dec + DEC_NUDGE, error, window=window, copy=True)
                    ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                    total_adjusted_exptime += sci.exptime / ref_exptime

                plt.subplot(gs[1:, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.plot(0, 0, "r+")
                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))

                if pix_counts is not None:
                    self.add_aperture_position(plt,mag_radius,mag)

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            #still need to plot relative fiber positions here
            plt.subplot(gs[1:, 0])
            return self.build_empty_cat_summary_figure(ra, dec, error, bid_ras, bid_decs, target_w=target_w,
                                                       fiber_locs=fiber_locs)
        else:
            self.master_cutout.data = self.master_cutout.data.astype(float) / total_adjusted_exptime

        plt.subplot(gs[1:, 0])

        if fiber_locs is not None:
            nudged_fiber_locs = []
            for f in fiber_locs:
                nudged_fiber_locs.append( (f[0] + RA_NUDGE, f[1] + DEC_NUDGE, f[2], f[3], f[4], f[5]) )

            self.add_fiber_positions(plt, ra + RA_NUDGE, dec + DEC_NUDGE, nudged_fiber_locs, error, ext, self.master_cutout)

        # complete the entry
        plt.close()
        return fig



    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None):


        window = error * 2.
        photoz_file = None
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        z_photoz_weighted = None

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
            text = "Too many matching targets. Individual reports on following pages.\n\nMORE PAGES ..."
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
            spec_z = 0.0

            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0])]

                idnum = df['ID'].values[0]  # to matchup in photoz catalog
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            try:
                # note cannot dirctly use RA,DEC as the recorded precission is different (could do a rounded match)
                # but the idnums match up, so just use that
                df_photoz = self.dataframe_of_bid_targets_photoz.loc[
                    self.dataframe_of_bid_targets_photoz['ID'] == idnum]

                if len(df_photoz) == 0:
                    log.debug("No conterpart found in photoz catalog; RA=%f , Dec =%f" % (r[0], d[0]))
                    df_photoz = None
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                df_photoz = None

            if df_photoz is not None:
                photoz_file = df_photoz['file'].values[0]
                z_best = df_photoz['z_best'].values[0]
                z_best_type = df_photoz['z_best_type'].values[0]  # s = spectral , p = photometric?
                z_photoz_weighted = df_photoz['mFDa4_z_weight']

            if df is not None:
                text = ""

                if G.ZOO:
                    text = text + "%g\"\n%0.3f\n" \
                                  % (df['distance'].values[0] * 3600.,df['dist_prior'].values[0])
                else:
                    text = text + "%g\"\n%0.3f\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600.,df['dist_prior'].values[0],
                                    df['RA'].values[0], df['DEC'].values[0])

                if z_best_type is not None:
                    if (z_best_type.lower() == 'p'):
                        text = text + "N/A\n" + "%g\n" % z_best
                    elif (z_best_type.lower() == 's'):
                        text = text + "%g (circle)\n" % z_best
                        spec_z = z_best
                        if z_photoz_weighted is not None:
                            text = text + "%g\n" % z_photoz_weighted
                        else:
                            text = text + "N/A\n"
                    else:
                        text = text + "N/A\n"
                else:
                    text = text + "N/A\nN/A\n"

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
                        filter_fl_cgs = self.micro_jansky_to_cgs(filter_fl,target_w)#filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

                        if target_w >= G.OII_rest:
                            text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.OII_rest))
                        else:
                            text = text + "N/A\n"

                        # bid target info is only of value if we have a flux from the emission line
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

                        addl_waves = None
                        addl_flux = None
                        addl_ferr = None
                        try:
                            addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                            addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                            addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                        except:
                            pass

                        bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = line_prob.prob_LAE(wl_obs=target_w,
                                                                                       lineFlux=target_flux,
                                                                                       ew_obs=(target_flux / filter_fl_cgs),
                                                                                       c_obs=None, which_color=None,
                                                                                       addl_wavelengths=addl_waves,
                                                                                       addl_fluxes=addl_flux,
                                                                                       addl_errors=addl_ferr,
                                                                                       sky_area=None,
                                                                                       cosmo=None, lae_priors=None,
                                                                                       ew_case=None, W_0=None,
                                                                                       z_OII=None, sigma=None)

                        for c in self.CatalogImages:
                            try:
                                bid_target.add_filter(c['instrument'], c['filter'],
                                                      self.micro_jansky_to_cgs(df[c['cols'][0]].values[0],
                                                                               target_w),
                                                      self.micro_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                               target_w))
                            except:
                                log.debug('Could not add filter info to bid_target.')

                        cat_match.add_bid_target(bid_target)
                else:
                    text += "N/A\nN/A\n"

                # if filter_mag != 0:
                text = text + "%0.2f(%0.2f,%0.2f)\n" % (filter_mag, filter_mag_bright, filter_mag_faint)
                # else:
                #    text = text + "%g(%g) $\\mu$Jy\n" % (filter_fl, filter_fl_err)

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    text += "%0.3g\n" % (bid_target.p_lae_oii_ratio)
                else:
                    text += "\n"
            else:
                text = "%s\n%f\n%f\n" % ("--", r, d)

            plt.subplot(gs[0, col_idx])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font,color=bid_colors[col_idx-1])


            # fig holds the entire page
        plt.close()
        return fig

#######################################
# end class GOODS_N
#######################################
