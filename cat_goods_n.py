from __future__ import print_function

import global_config as G
import os.path as op


GOODS_N_BASE_PATH = G.GOODS_N_BASE_PATH
GOODS_N_CAT = op.join(G.GOODS_N_CAT_PATH, "GOODSN_HST_Finkelstein.cat")
GOODS_N_IMAGES_PATH = GOODS_N_BASE_PATH


import matplotlib
matplotlib.use('agg')

import pandas as pd
import science_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec


log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

import cat_base
import match_summary

class GOODS_N(cat_base.Catalog):

    # class variables
    # 1 RA (J2000)
    # 2 Dec (J2000)
    # 3 Best-fit photo-z
    # 4 Photoz lower 68% CL
    # 5 Photoz upper 68% CL
    # 6 F435W (B) flux (nJy)
    # 7 F435W (B) flux error (nJy)
    # 8 F606W (V) flux (nJy)
    # 9 F606W (V) flux error (nJy)
    # 10 F775W (i) flux (nJy)
    # 11 F775W (i) flux error (nJy)
    # 12 F814W (I) flux (nJy)
    # 13 F814W (I) flux error (nJy)
    # 14 F850LP (z) flux (nJy)
    # 15 F850LP (z) flux error (nJy)
    # 16 F105W (Y) flux (nJy)
    # 17 F105W (Y) flux error (nJy)
    # 18 F125W (J) flux (nJy)
    # 19 F125W (J) flux error (nJy)
    # 20 F160W (H) flux (nJy)
    # 21 F160W (H) flux error (nJy)

    MainCatalog = GOODS_N_CAT
    Name = "GOODS_N"
    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min': 188.91, 'RA_max': 189.55, 'Dec_min': 62.09, 'Dec_max': 62.39}
    Cat_Coord_Range = {'RA_min': 188.915588, 'RA_max': 189.543671, 'Dec_min': 62.091625, 'Dec_max': 62.385319}
    WCS_Manual = True
    EXPTIME_F606W = 16950.0
    CONT_EST_BASE = 2.8e-21

    CatalogImages = [
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'gn_acs_old_f435w_060mas_v2_drz.fits',
         'filter': 'f435w',
         'instrument': 'ACS',
         'cols': ['F435W (V) flux (nJy)', 'F435W (V) flux error (nJy)'],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'gn_acs_old_f606w_060mas_v2_drz.fits',
         'filter': 'f606w',
         'instrument': 'ACS',
         'cols': ['F606W (V) flux (nJy)','F606W (V) flux error (nJy)'],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'gn_acs_old_f775w_060mas_v2_drz.fits',
         'filter': 'f775w',
         'instrument': 'ACS',
         'cols': ['F775W (V) flux (nJy)', 'F775W (V) flux error (nJy)'],
         'labels': ["Flux", "Err"],
         'image': None
         },
    #    {'path': GOODS_N_IMAGES_PATH,
    #     'name': 'gn_acs_old_f850l_060mas_v2_drz.fits',
    #     'filter': 'f850l',
    #     'instrument': 'ACS',
    #     'cols': ['F850L (V) flux (nJy)', 'F850L (V) flux error (nJy)'],
    #     'labels': ["Flux", "Err"],
    #     'image': None
    #     },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'gn_all_candels_acs_f814w_060mas_v0.9_drz.fits',
         'filter': 'f814w',
         'instrument': 'ACS',
         'cols': ['F814W (I) flux (nJy)','F814W (I) flux error (nJy)'],
         'labels': ["Flux", "Err"],
         'image': None
         },
     #   {'path': GOODS_N_IMAGES_PATH,
     #    'name': 'gn_all_candels_wfc3_f105w_060mas_v0.8_drz.fits',
     #    'filter': 'f105w',
     #    'instrument': 'WFC3',
     #    'cols': ['F105W (J) flux (nJy)', 'F105W (J) flux error (nJy)'],
     #    'labels': ["Flux", "Err"],
     #    'image': None
     #    },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'gn_all_candels_wfc3_f125w_060mas_v0.8_drz.fits',
         'filter': 'f125w',
         'instrument': 'WFC3',
         'cols': ['F125W (J) flux (nJy)','F125W (J) flux error (nJy)'],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'gn_all_candels_wfc3_f160w_060mas_v0.8_drz.fits',
         'filter': 'f160w',
         'instrument': 'WFC3',
         'cols': ['F160W (H) flux (nJy)','F160W (H) flux error (nJy)'],
         'labels': ["Flux", "Err"],
         'image': None
         }
    ]

    def __init__(self):
        super(GOODS_N, self).__init__()

        self.num_targets = 0
        self.master_cutout = None

    # so building as needed does not seem to help memory
    def build_catalog_images(self):
        for i in self.CatalogImages:  # i is a dictionary
            i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                     image_location=op.join(i['path'], i['name']))

    @classmethod
    def read_catalog(cls, catalog_loc, name):

        log.debug("Building " + name + " dataframe...")
        header = []
        skip = 0
        try:
            f = open(catalog_loc, mode='r')
        except:
            log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
            return None

        line = f.readline()
        while '#' in line:
            skip += 1
            toks = line.split()
            if (len(toks) > 2): # and toks[1].isdigit():  # format:   #<id number> <column name (multiple tokens)>
                label = " ".join(toks[1:])
                #translastion
                if label == 'RA (J2000)':
                    label = 'RA'
                elif label == 'Dec (J2000)':
                    label = 'DEC'
                header.append(label)
            line = f.readline()

        f.close()

        try:
            df = pd.read_csv(catalog_loc, names=header,
                             delim_whitespace=True, header=None, index_col=None, skiprows=skip)
        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return df

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        if self.df is None:
            self.read_main_catalog()

        error_in_deg = np.float64(error) / 3600.0

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0

        coord_scale = np.cos(np.deg2rad(dec))

        # can't actually happen for this catalog
        if coord_scale < 0.1:  # about 85deg
            print("Warning! Excessive declination (%f) for this method of defining error window. Not supported" % (dec))
            log.error(
                "Warning! Excessive declination (%f) for this method of defining error window. Not supported" % (dec))
            return 0, None, None

        ra_min = np.float64(ra - error_in_deg)
        ra_max = np.float64(ra + error_in_deg)
        dec_min = np.float64(dec - error_in_deg)
        dec_max = np.float64(dec + error_in_deg)

        log.info(self.Name + " searching for bid targets in range: RA [%f +/- %f], Dec [%f +/- %f] ..."
                 % (ra, error_in_deg, dec, error_in_deg))

        try:
            self.dataframe_of_bid_targets = \
                self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                        (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)].copy()

        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

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
        self.build_list_of_bid_targets(target_ra, target_dec, error)

        ras = self.dataframe_of_bid_targets.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            entry = self.build_cat_summary_figure(target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux)
        else:
            entry = self.build_exact_target_location_figure(target_ra, target_dec, error, section_title=section_title,
                                                        target_w=target_w, fiber_locs=fiber_locs,
                                                        target_flux=target_flux)
        if entry is not None:
            self.add_bid_entry(entry)

        if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
            entry = self.build_multiple_bid_target_figures_one_line(cat_match,ras, decs, error,
                                                               target_ra=target_ra, target_dec=target_dec,
                                                               target_w=target_w, target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)
        else: #each bid taget gets its own line

            bid_colors = self.get_bid_colors(len(ras))
            number = 0
            for r, d in zip(ras, decs):
                number += 1
                try:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0])]

                except:
                    log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                    continue  # this must be here, so skip to next ra,dec

                print("Building report for bid target %d in %s" % (base_count + number, self.Name))

                if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
                    entry = self.build_bid_target_figure_one_line(cat_match,r[0], d[0], error=error, df=df, df_photoz=None,
                                                     target_ra=target_ra, target_dec=target_dec,
                                                     section_title=section_title,
                                                     bid_number=number, target_w=target_w, of_number=num_hits-base_count,
                                                     target_flux=target_flux,color=bid_colors[number-1])
                else:
                    entry = self.build_bid_target_figure(cat_match,r[0], d[0], error=error, df=df, df_photoz=None,
                                                     target_ra=target_ra, target_dec=target_dec,
                                                     section_title=section_title,
                                                     bid_number=number, target_w=target_w, of_number=num_hits-base_count,
                                                     target_flux=target_flux)
                if entry is not None:
                    self.add_bid_entry(entry)

        return self.pages

    def build_exact_target_location_figure(self, ra, dec, error, section_title="", target_w=0, fiber_locs=None,
                                           target_flux=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 4

        # set a minimum window size?
        # if window < 8:
        #    window = 8

        rows = 2
        cols = len(self.CatalogImages)

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        title = "Catalog: %s\n" % self.Name + section_title + "\nPossible Matches = %d (within +/- %g\")\n" \
                                                              "RA = %f    Dec = %f\n" % (
                                                                  len(self.dataframe_of_bid_targets), error, ra, dec)
        if target_w > 0:
            title = title + "Wavelength = %g $\AA$\n" % target_w
        else:
            title = title + "\n"

        if target_flux is not None:
            cont_est = self.CONT_EST_BASE*3 #self.get_f606w_max_cont(self.EXPTIME_F606W, 3, self.CONT_EST_BASE)
            if cont_est != -1:
                title += "Minimum (no match)\n  3$\sigma$ rest-EW:\n"
                title += "  LyA = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A\n"

        plt.subplot(gs[0, 0])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        index = -1
        ref_exptime = 1.0
        total_adjusted_exptime = 1.0
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            if cutout is not None:  # construct master cutout
                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout = sci.get_cutout(ra, dec, error, window=window, copy=True)
                    if sci.exptime:
                        ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    try:
                        self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
                    except:
                        log.warn("Unexpected exception.", exc_info=True)

                plt.subplot(gs[rows - 1, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([ext, ext / 2., 0, -ext / 2., -ext])
                plt.yticks([ext, ext / 2., 0, -ext / 2., -ext])

                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            return None
        else:
            self.master_cutout.data /= total_adjusted_exptime

        # plot the master cutout
        empty_sci = science_image.science_image()
        plt.subplot(gs[0, cols - 1])
        vmin, vmax = empty_sci.get_vrange(self.master_cutout.data)
        plt.imshow(self.master_cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                   vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
        plt.title("Master Cutout (Stacked)")
        plt.xlabel("arcsecs")
        plt.xticks([ext, ext / 2., 0, -ext / 2., -ext])
        plt.yticks([ext, ext / 2., 0, -ext / 2., -ext])

        # only show this lable if there is not going to be an adjacent fiber plot
        if (fiber_locs is None) or (len(fiber_locs) == 0):
            plt.ylabel("arcsecs")
        plt.plot(0, 0, "r+")

        theta = empty_sci.get_rotation_to_celestrial_north(self.master_cutout)

        self.add_north_box(plt, sci, self.master_cutout, error, 0, 0, theta)
        # self.add_north_arrow(plt, sci, self.master_cutout, theta)

        plt.subplot(gs[0, cols - 2])
        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)

        # complete the entry
        plt.close()
        return fig


    def build_bid_target_figure(self, cat_match, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                section_title="", bid_number=1, target_w=0, of_number=0, target_flux=None):
        '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
        photometry images, Z_PDF, etc

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 2.
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        z_photoz_weighted = None

        rows = 2
        cols = len(self.CatalogImages)

        if df_photoz is not None:
           pass #photoz is None at this time

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        if df is not None:
            title = "%s  Possible Match #%d" % (section_title, bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

            title = title + "\nRA = %f    Dec = %f\nSeparation    = %g\"" \
                            % (df['RA'].values[0], df['DEC'].values[0],
                               df['distance'].values[0] * 3600)

            if z_best_type is not None:
                if (z_best_type.lower() == 'p'):
                    title = title + "\nPhoto Z       = %g (blue)" % z_best
                elif (z_best_type.lower() == 's'):
                    title = title + "\nSpec Z        = %g (gold)" % z_best
                    spec_z = z_best
                    if z_photoz_weighted is not None:
                        title = title + "\nPhoto Z       = %g (blue)" % z_photoz_weighted

            if target_w > 0:
                la_z = target_w / G.LyA_rest - 1.0
                oii_z = target_w / G.OII_rest - 1.0
                title = title + "\nLyA Z (virus) = %g " % la_z
                if (oii_z > 0):
                    title = title + "\nOII Z (virus) = %g " % oii_z
                else:
                    title = title + "\nOII Z (virus) = N/A"

            if target_flux is not None:

                best_fit_photo_z = 0.0
                try:
                    best_fit_photo_z = float(df['Best-fit photo-z'].values[0])
                except:
                    pass

                title += "\nSpec Z = N/A\n" + "Photo Z = %g\n" % best_fit_photo_z

                try:
                    filter_fl = float(
                        df['F606W (V) flux (nJy)'].values[0])  # in nano-jansky or 1e-32  erg s^-1 cm^-2 Hz^-2
                    filter_fl_err = float(df['F606W (V) flux error (nJy)'].values[0])
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None) and (filter_fl > 0):
                        filter_fl_adj = self.nano_jansky_to_cgs(filter_fl,target_w)# * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        title = title + "Est LyA rest-EW = %g $\AA$\n" \
                                        % (target_flux / filter_fl_adj / (target_w / G.LyA_rest))

                        if target_w >= G.OII_rest:
                            title = title + "Est OII rest-EW = %g $\AA$\n" \
                                            % (target_flux / filter_fl_adj / (target_w / G.OII_rest))
                        else:
                            title = title + "Est OII rest-EW = N/A\n"

                        # bid target info is only of value if we have a flux from the emission line
                        bid_target = match_summary.BidTarget()
                        bid_target.bid_ra = df['RA'].values[0]
                        bid_target.bid_dec = df['DEC'].values[0]
                        bid_target.distance = df['distance'].values[0] * 3600
                        bid_target.bid_flux_est_cgs = filter_fl

                        for c in self.CatalogImages:
                            try:
                                bid_target.add_filter(c['instrument'], c['filter'],
                                                      self.nano_jansky_to_cgs(df[c['cols'][0]].values[0],
                                                                               target_w),
                                                      self.nano_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                               target_w))
                            except:
                                log.debug('Could not add filter info to bid_target.')

                        cat_match.add_bid_target(bid_target)
                else:
                    title += "Est LyA rest-EW = N/A\nEst OII rest-EW = N/A\n"

                title = title + "F606W Flux = %g(%g) nJy\n" % (filter_fl, filter_fl_err)
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

        plt.subplot(gs[0, 0])
        plt.text(0, 0, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        index = -1
        # iterate over all filter images
        for i in self.CatalogImages:  # i is a dictionary
            index += 1  # for subplot ... is 1 based
            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.

            if cutout is not None:
                plt.subplot(gs[1, index])

                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

                # add (+) to mark location of Target RA,DEC
                # we are centered on ra,dec and target_ra, target_dec belong to the HETDEX detect
                if cutout and (target_ra is not None) and (target_dec is not None):
                    px, py = sci.get_position(target_ra, target_dec, cutout)
                    x, y = sci.get_position(ra, dec, cutout)

                    plt.plot((px - x), (py - y), "r+")

                    plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2., height=error * 2.,
                                                      angle=0.0, color='yellow', fill=False, linewidth=5.0, zorder=1))
                    # set the diameter of the cirle to half the error (radius error/4)
                    plt.gca().add_patch(plt.Circle((0, 0), radius=error / 4.0, color='yellow', fill=False))

                    self.add_north_arrow(plt, sci, cutout, theta=None)

                # iterate over all filters for this image and print values
                font.set_size(10)
                if df is not None:
                    s = ""
                    for f, l in zip(i['cols'], i['labels']):
                        # print (f)
                        try:
                            v = float(df[f].values[0])
                        except:
                            v = 0.0

                        s = s + "%-8s = %.5f\n" % (l, v)

                    plt.xlabel(s, multialignment='left', fontproperties=font)

        # add photo_z plot
        if df_photoz is not None:
            pass
        else:
            plt.subplot(gs[0, 2])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            text = "Photo Z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        # master cutout (0,0 is the observered (exact) target RA, DEC)
        if (self.master_cutout) and (target_ra) and (target_dec):
            ext = error * 1.5 #to be consistent with self.master_cutout scale set to error window *3 and ext = /2
            # resizing (growing) is a problem since the master_cutout is stacked
            # (could shrink (cutout smaller section) but not grow without re-stacking larger cutouts of filters)
            plt.subplot(gs[0, cols - 1])
            empty_sci = science_image.science_image()
            # need a new cutout since we rescaled the ext (and window) size
            #cutout = empty_sci.get_cutout(target_ra, target_dec, error, window=ext * 2, image=self.master_cutout)
            cutout = self.master_cutout
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            plt.imshow(cutout.data, origin='lower', interpolation='none',
                       cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
            plt.title("Master Cutout (Stacked)")
            plt.xlabel("arcsecs")

            plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
            plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

            # mark the bid target location on the master cutout
            # note: poor naming on my part, but target_ra and target_dec are the emission line (so the center)
            px, py = empty_sci.get_position(target_ra, target_dec, cutout)
            x, y = empty_sci.get_position(ra, dec, cutout)

            # set the diameter of the cirle to half the error (radius error/4)
            plt.gca().add_patch(
                plt.Circle(((x - px), (y - py)), radius=error / 4.0, color='yellow', fill=False))

            # this is correct, do not rotate the yellow rectangle (it is a zoom window only)
         #   x = (x - px) - error
         #   y = (y - py) - error
         #   plt.gca().add_patch(plt.Rectangle((x, y), width=error * 2, height=error * 2,
         #                                     angle=0.0, color='yellow', fill=False))

            plt.plot(0, 0, "r+")
            self.add_north_box(plt, empty_sci, cutout, error, 0, 0, theta=None)

        # fig holds the entire page
        plt.close()
        return fig


    def build_cat_summary_figure (self, ra, dec, error,bid_ras, bid_decs, target_w=0,
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

        rows = 10 #2
        cols = 1+ len(self.CatalogImages)

        fig_sz_x = 18 #cols * 3
        fig_sz_y = 3 #rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        # All on one line now across top of plots
        title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                            % (len(self.dataframe_of_bid_targets), error)

        if target_flux is not None:
            cont_est = self.CONT_EST_BASE * 3
            if cont_est != -1:
                title += "  Minimum (no match) 3$\sigma$ rest-EW: "
                title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A"

        plt.subplot(gs[0, :])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None


        ref_exptime = None
        total_adjusted_exptime = None

        # add the bid targets
        #norm = plt.Normalize()
        #bid_colors = plt.cm.brg(norm(np.arange(len(bid_ras))))
        bid_colors = self.get_bid_colors(len(bid_ras))

        index = 0 #start in the 2nd box (1 + 1 = 2, zero based count)
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            if cutout is not None:  # construct master cutout
                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout = sci.get_cutout(ra, dec, error, window=window, copy=True)
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

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            return None
        else:
            self.master_cutout.data /= total_adjusted_exptime

        plt.subplot(gs[1:, 0])

        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)

        # complete the entry
        plt.close()
        return fig

    def build_bid_target_figure_one_line (self,cat_match, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                section_title="", bid_number=1, target_w=0, of_number=0, target_flux=None, color="k"):
        '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
        photometry images, Z_PDF, etc

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 2.
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        z_photoz_weighted = None

        rows = 1
        cols = 6

        if df_photoz is not None:
           pass

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

        #cols*2 to leave one column for the color coded rectange ... all other are made at 2x columns
        gs = gridspec.GridSpec(rows, cols*2, wspace=0.25, hspace=0.5)

        #use col = 0 for color coded rectangle
        plt.subplot(gs[0, 0])
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')
        #2:1 so height should be 1/2 width
        plt.gca().add_patch(plt.Rectangle((0.25,0.25), width=0.5, height=0.25, angle=0.0, color=color,
                                          fill=False,linewidth=5,zorder=1))

        #entry text
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)
        spec_z = 0.0

        if df is not None:
            title = "%s  Possible Match #%d" % (section_title, bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

            title = title + "\nRA = %f    Dec = %f\nSeparation    = %g\"" \
                            % (df['RA'].values[0], df['DEC'].values[0],df['distance'].values[0] * 3600)

            if z_best_type is not None:
                if (z_best_type.lower() == 'p'):
                    title = title + "\nPhoto Z       = %g (blue)" % z_best
                elif (z_best_type.lower() == 's'):
                    title = title + "\nSpec Z        = %g (gold)" % z_best
                    spec_z = z_best
                    if z_photoz_weighted is not None:
                        title = title + "\nPhoto Z       = %g (blue)" % z_photoz_weighted

            if target_w > 0:
                la_z = target_w / G.LyA_rest - 1.0
                oii_z = target_w / G.OII_rest - 1.0
                title = title + "\nLyA Z (virus) = %g" % la_z
                if (oii_z > 0):
                    title = title + "\nOII Z (virus) = %g" % oii_z
                else:
                    title = title + "\nOII Z (virus) = N/A"

            if target_flux is not None:
                filter_fl = df['ACS_F606W_FLUX'].values[0]  # in nano-jansky or 1e-32  erg s^-1 cm^-2 Hz^-2
                if (filter_fl is not None) and (filter_fl > 0):
                    filter_fl = self.nano_jansky_to_cgs(filter_fl,target_w) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                    title = title + "\nEst LyA rest-EW = %g $\AA$" \
                                    % (target_flux / filter_fl / (target_w / G.LyA_rest))

                    if target_w >= G.OII_rest:
                        title = title + "\nEst OII rest-EW = %g $\AA$" \
                                        % (target_flux / filter_fl / (target_w / G.OII_rest))
                    else:
                        title = title + "\nEst OII rest-EW = N/A"

                    # bid target info is only of value if we have a flux from the emission line
                    bid_target = match_summary.BidTarget()
                    bid_target.bid_ra = df['RA'].values[0]
                    bid_target.bid_dec = df['DEC'].values[0]
                    bid_target.distance = df['distance'].values[0] * 3600
                    bid_target.bid_flux_est_cgs = filter_fl

                    for c in self.CatalogImages:
                        try:
                            bid_target.add_filter(c['instrument'], c['filter'],
                                                  self.nano_jansky_to_cgs(df[c['cols'][0]].values[0],
                                                                          target_w),
                                                  self.nano_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                          target_w))
                        except:
                            log.debug('Could not add filter info to bid_target.')

                    cat_match.add_bid_target(bid_target)
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

        plt.subplot(gs[0, 1:4])
        plt.text(0, 0, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        #add flux values
        if df is not None:
            # iterate over all filter images
            title = ""
            for i in self.CatalogImages:  # i is a dictionary
                # iterate over all filters for this image and print values
                #font.set_size(10)

                #not all filters have entries ... note 'cols'[0] is flux, [1] is the error
                if df[i['cols']].empty :
                    title = title + "%7s %s %s = -- (--)\n" % (i['instrument'], i['filter'], "Flux")
                else:
                    try:
                        title = title + "%7s %s %s = %.5f (%.5f)\n" % (i['instrument'], i['filter'], "Flux",
                                float(df[i['cols'][0]].values[0]), float(df[i['cols'][1]].values[0]))
                    except:
                        title = title + "%7s %s %s = -- (--)\n" % (i['instrument'], i['filter'], "Flux")


        plt.subplot(gs[0, 4])
        plt.text(0, 0, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        # todo: photo z plot if becomes available
        # add photo_z plot
        if df_photoz is not None:
            pass
        else:
            plt.subplot(gs[0, -4:])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            text = "Photo Z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        # fig holds the entire page
        plt.close()
        return fig

    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None):
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


        bid_colors = self.get_bid_colors(len(ras))

        text = "Separation\n" + \
               "RA, Dec\n" + \
               "Spec Z\n" + \
               "Photo Z\n" + \
               "Est LyA rest-EW\n" + \
               "Est OII rest-EW\n" + \
               "ACS WFC f606W Flux\n"

        plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)

        col_idx = 0
        for r, d in zip(ras, decs):
            col_idx += 1
            spec_z = 0.0

            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0])]

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                text = ""
                text = text + "%g\"\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600,df['RA'].values[0], df['DEC'].values[0])

                best_fit_photo_z = 0.0
                try:
                    best_fit_photo_z = float(df['Best-fit photo-z'].values[0])
                except:
                    pass

                text += "N/A\n" + "%g\n" % best_fit_photo_z

                try:
                    filter_fl = float(df['F606W (V) flux (nJy)'].values[0])  # in nano-jansky or 1e-32  erg s^-1 cm^-2 Hz^-2
                    filter_fl_err = float(df['F606W (V) flux error (nJy)'].values[0])
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_adj = self.nano_jansky_to_cgs(filter_fl,target_w)#filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        text = text + "%g $\AA$\n" % (target_flux / filter_fl_adj / (target_w / G.LyA_rest))

                        if target_w >= G.OII_rest:
                            text = text + "%g $\AA$\n" % (target_flux / filter_fl_adj / (target_w / G.OII_rest))
                        else:
                            text = text + "N/A\n"

                        # bid target info is only of value if we have a flux from the emission line
                        bid_target = match_summary.BidTarget()
                        bid_target.bid_ra = df['RA'].values[0]
                        bid_target.bid_dec = df['DEC'].values[0]
                        bid_target.distance = df['distance'].values[0] * 3600
                        bid_target.bid_flux_est_cgs = filter_fl

                        for c in self.CatalogImages:
                            try:
                                bid_target.add_filter(c['instrument'], c['filter'],
                                                      self.nano_jansky_to_cgs(df[c['cols'][0]].values[0],
                                                                              target_w),
                                                      self.nano_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                              target_w))
                            except:
                                log.debug('Could not add filter info to bid_target.')

                        cat_match.add_bid_target(bid_target)
                else:
                    text += "N/A\nN/A\n"

                if filter_fl < 0:
                    text = text + "%g(%g) nJy !?\n" % (filter_fl, filter_fl_err)
                else:
                    text = text + "%g(%g) nJy\n" % (filter_fl, filter_fl_err)
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
            text = "Photo Z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        plt.close()
        return fig

#######################################
# end class GOODS_N
#######################################
