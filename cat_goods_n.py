from __future__ import print_function

import global_config as G
import os.path as op


GOODS_N_BASE_PATH = G.GOODS_N_BASE_PATH
GOODS_N_CAT = op.join(GOODS_N_BASE_PATH,"photometry/CANDELS.GOODSN.F160W.v1_1.photom.cat")
GOODS_N_IMAGES_PATH = op.join(GOODS_N_BASE_PATH, "images")
GOODS_N_PHOTOZ_CAT = op.join(GOODS_N_BASE_PATH , "photoz/zcat_GOODSN_v2.0.cat")
GOODS_N_PHOTOZ_ZPDF_PATH = op.join(GOODS_N_BASE_PATH, "photoz/zPDFs/")

import matplotlib
matplotlib.use('agg')

import pandas as pd
import science_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
#import matplotlib.patches as mpatches
#import mpl_toolkits.axisartist.floating_axes as floating_axes
#from matplotlib.transforms import Affine2D
import line_prob


log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

import cat_base
import match_summary


class GOODS_N(cat_base.Catalog):

    # class variables
    MainCatalog = GOODS_N_CAT
    Name = "GOODS-N"

    # if multiple images, the composite broadest range (filled in by hand)
    Cat_Coord_Range = {'RA_min': 188.915597, 'RA_max': 189.563471, 'Dec_min': 62.091438, 'Dec_max': 62.388316}
    Image_Coord_Range = {'RA_min':188.862 , 'RA_max': 189.605, 'Dec_min': 62.066, 'Dec_max': 62.409}

    WCS_Manual = True
    EXPTIME_F606W = 236118.0 #289618.0
    CONT_EST_BASE = 2.8e-21#3.3e-21
    BidCols = ["ID", "IAU_designation", "RA", "DEC",
               "CFHT_U_FLUX", "CFHT_U_FLUXERR",
               "IRAC_CH1_FLUX", "IRAC_CH1_FLUXERR", "IRAC_CH2_FLUX", "IRAC_CH2_FLUXERR",
               "ACS_F606W_FLUX", "ACS_F606W_FLUXERR",
               "ACS_F814W_FLUX", "ACS_F814W_FLUXERR",
               "WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR",
               "WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR",
               "WC3_F160W_FLUX", "WFC3_F160W_FLUXERR",
               "DEEP_SPEC_Z"]  # NOTE: there are no F105W values

    CatalogImages = [
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f435w_060mas_v1.3_drz.fits',
         'filter': 'f435w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F435W_FLUX", "ACS_F435W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f606w_060mas_v2.0_drz.fits',
         'filter': 'f606w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F606W_FLUX", "ACS_F606W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f775w_060mas_v2.0_drz.fits',
         'filter': 'f775w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F775W_FLUX", "ACS_F775W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f814w_060mas_v2.0_drz.fits',
         'filter': 'f814w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F814W_FLUX", "ACS_F814W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        #omit 850LP (per Steve) ... long exposure due to low sensitivity
       # {'path': GOODS_N_IMAGES_PATH,
       #  'name': 'goodsn_all_acs_wfc_f850l_060mas_v2.0_drz.fits',
       #  'filter': 'f850lp',
       #  'instrument': 'ACS WFC',
       #  'cols': ["ACS_F850LP_FLUX", "ACS_F850LP_FLUXERR"],
       #  'labels': ["Flux", "Err"],
       #  'image': None
       #  },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_wfc3_ir_f105w_060mas_v1.0_drz.fits',
         'filter': 'f105w',
         'instrument': 'WFC3',
         'cols': [],
         'labels': [],
         'image': None
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_wfc3_ir_f125w_060mas_v1.0_drz.fits',
         'filter': 'f125w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        #omit 140w per Steve
        #{'path': GOODS_N_IMAGES_PATH,
        # 'name': 'goodsn_all_wfc3_ir_f140w_060mas_v1.0_drz.fits',
        # 'filter': 'f140w',
        # 'instrument': 'WFC3',
        # 'cols': ["WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR"],
        # 'labels': ["Flux", "Err"],
        # 'image': None
        # },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_wfc3_ir_f160w_060mas_v1.0_drz.fits',
         'filter': 'f160w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F160W_FLUX", "WFC3_F160W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         }
    ]

    PhotoZCatalog = GOODS_N_PHOTOZ_CAT
    SupportFilesLocation = GOODS_N_PHOTOZ_ZPDF_PATH

    def __init__(self):
        super(GOODS_N, self).__init__()

        # self.dataframe_of_bid_targets = None #defined in base class
        self.dataframe_of_bid_targets_photoz = None
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
        if cls.df_photoz is not None:
            log.debug("Already built df_photoz")
        else:
            try:
                print("Reading photoz catalog for ", cls.Name)
                cls.df_photoz = cls.read_catalog(cls.PhotoZCatalog, cls.Name)
            except:
                print("Failed")

        return

    @classmethod
    def read_catalog(cls, catalog_loc, name):

        log.debug("Building " + name + " dataframe...")
        idx = []
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
            if (len(toks) > 2) and toks[1].isdigit():  # format:   # <id number> <column name>
                idx.append(toks[1])
                header.append(toks[2])
            line = f.readline()

        f.close()

        try:
            df = pd.read_csv(catalog_loc, names=header,
                             delim_whitespace=True, header=None, index_col=None, skiprows=skip)

            old_names = ['ID (H-band SExtractor ID)', 'IAU_Name','RA (J2000, H-band)', 'DEC (J2000, H-band)']
            new_names = ['ID', 'IAU_designation','RA', 'DEC']
            df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return df

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        if self.df is None:
            self.read_main_catalog()
        if self.df_photoz is None:
            self.read_photoz_catalog()

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

        try:
            self.dataframe_of_bid_targets = \
                self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                        (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)].copy()

            # ID matches between both catalogs
            self.dataframe_of_bid_targets_photoz = \
                self.df_photoz[(self.df_photoz['ID'].isin(self.dataframe_of_bid_targets['ID']))].copy()
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

        if G.SINGLE_PAGE_PER_DETECT: # and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
            entry = self.build_multiple_bid_target_figures_one_line(cat_match,ras, decs, error,
                                                               target_ra=target_ra, target_dec=target_dec,
                                                               target_w=target_w, target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)
        #else: #each bid taget gets its own line
        if len(ras) > G.MAX_COMBINE_BID_TARGETS:  # each bid taget gets its own line

            bid_colors = self.get_bid_colors(len(ras))
            number = 0
            for r, d in zip(ras, decs):
                number += 1
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

                print("Building report for bid target %d in %s" % (base_count + number, self.Name))

                if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
                    entry = self.build_bid_target_figure_one_line(cat_match,r[0], d[0], error=error, df=df, df_photoz=df_photoz,
                                                     target_ra=target_ra, target_dec=target_dec,
                                                     section_title=section_title,
                                                     bid_number=number, target_w=target_w, of_number=num_hits-base_count,
                                                     target_flux=target_flux,color=bid_colors[number-1])
                else:
                    entry = self.build_bid_target_figure(cat_match,r[0], d[0], error=error, df=df, df_photoz=df_photoz,
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

        if G.ZOO:
            title = "Catalog: %s\n" % self.Name + section_title + "\nPossible Matches = %d (within +/- %g\")\n" % \
                                                                  (len(self.dataframe_of_bid_targets), error)
        else:
            title = "Catalog: %s\n" % self.Name + section_title + "\nPossible Matches = %d (within +/- %g\")\n" \
                                                                  "RA = %f    Dec = %f\n" % (
                                                                      len(self.dataframe_of_bid_targets), error, ra,
                                                                      dec)
        if target_w > 0:
            title = title + "Wavelength = %g $\AA$\n" % target_w
        else:
            title = title + "\n"

        if target_flux is not None:
            cont_est = self.CONT_EST_BASE*3 #self.get_f606w_max_cont(self.EXPTIME_F606W, 3, self.CONT_EST_BASE)
            if cont_est != -1:
                title += "Minimum (no match)\n  3$\sigma$ rest-EW:\n"
                title += "  LyA = %g $\AA$\n" % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$\n" % ((target_flux / cont_est) / (target_w / G.OII_rest))
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

                #1st cutout might not be what we want for the master (could be a summary image from elsewhere)
                if self.master_cutout:
                    if self.master_cutout.shape != cutout.shape:
                        del self.master_cutout
                        self.master_cutout = None

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
        photoz_file = None
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        # z_spec = None
        # z_spec_ref = None
        z_photoz_weighted = None

        rows = 2
        cols = len(self.CatalogImages)

        if df_photoz is not None:
            photoz_file = df_photoz['file'].values[0]
            z_best = df_photoz['z_best'].values[0]
            z_best_type = df_photoz['z_best_type'].values[0]  # s = spectral , p = photometric?
            z_photoz_weighted = df_photoz['mFDa4_z_weight']
            # z_spec = df_photoz['z_spec'].values[0]
            # z_spec_ref = df_photoz['z_spec_ref'].values[0]
            # rows = rows + 1

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        spec_z = 0.0

        if df is not None:
            title = "%s  Possible Match #%d" % (section_title, bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

            if G.ZOO:
                title = title + "\nSeparation    = %g\"" \
                                % (df['distance'].values[0] * 3600)
            else:
                title = title + "\n%s\nRA = %f    Dec = %f\nSeparation    = %g\"" \
                            % (df['IAU_designation'].values[0], df['RA'].values[0], df['DEC'].values[0],
                               df['distance'].values[0] * 3600)

            # do not use DEEP SPEC Z, just use spec Z below
            # z = df['DEEP_SPEC_Z'].values[0]
            # if z >= 0.0:
            #     if (z_best_type is not None) and (z_best_type.lower() == 's'):
            #         title = title + "\nDEEP SPEC Z = %g" % z
            #     else:
            #         title = title + "\nDEEP SPEC Z = %g (gold)" % z
            #         spec_z = z

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
                title = title + "\nLyA Z (virus) = %g (red)" % la_z
                if (oii_z > 0):
                    title = title + "\nOII Z (virus) = %g (green)" % oii_z
                else:
                    title = title + "\nOII Z (virus) = N/A"

            if target_flux is not None:
                filter_fl = df['ACS_F606W_FLUX'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                if (filter_fl is not None) and (filter_fl > 0):
                    filter_fl = filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
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

                    # todo: add call to line_probabilities:
                    ratio, bid_target.p_lae, bid_target.p_oii = line_prob.prob_LAE(wl_obs=target_w,
                                                                                   lineFlux=target_flux,
                                                                                   ew_obs=(target_flux / filter_fl),
                                                                                   c_obs=None, which_color=None,
                                                                                   addl_fluxes=None, sky_area=None,
                                                                                   cosmo=None, lae_priors=None,
                                                                                   ew_case=None, W_0=None, z_OII=None,
                                                                                   sigma=None)
                    if not G.ZOO:
                        if bid_target.p_lae is not None:
                            title += "\nP(LAE) = %0.3g" % bid_target.p_lae
                            if bid_target.p_oii is not None:
                                title += "  P(OII) = %0.3g" % bid_target.p_oii
                                if bid_target.p_oii > 0.0:
                                    title += "\nP(LAE)/P(OII) = %0.3g" % (bid_target.p_lae / bid_target.p_oii)

                    for c in self.CatalogImages:
                        try:
                           bid_target.add_filter(c['instrument'], c['filter'],
                                              self.micro_jansky_to_cgs(df[c['cols'][0]].values[0], target_w),
                                              self.micro_jansky_to_cgs(df[c['cols'][1]].values[0], target_w))
                        except:
                            log.debug('Could not add filter info to bid_target.')

                    cat_match.add_bid_target(bid_target)
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
                        v = df[f].values[0]
                        s = s + "%-8s = %.5f\n" % (l, v)

                    plt.xlabel(s, multialignment='left', fontproperties=font)

        # add photo_z plot
        # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
        # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
        # get 'file'
        # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
        if df_photoz is not None:
            z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
            if z_cat is not None:
                x = z_cat['z'].values
                y = z_cat['mFDa4'].values
                plt.subplot(gs[0, 3:5])
                plt.plot(x, y, zorder=1)
                plt.xlim([0, 3.6])
                # trim axis to 0 to 3.6

                if spec_z > 0:
                    plt.axvline(x=spec_z, color='gold', linestyle='solid', linewidth=3, zorder=0)

                if target_w > 0:
                    la_z = target_w / G.LyA_rest - 1.0
                    oii_z = target_w / G.OII_rest - 1.0
                    plt.axvline(x=la_z, color='r', linestyle='--', zorder=2)
                    if (oii_z > 0):
                        plt.axvline(x=oii_z, color='g', linestyle='--', zorder=2)

                plt.title("Photo Z PDF")
                plt.gca().yaxis.set_visible(False)
                plt.xlabel("Z")


        # master cutout (0,0 is the observered (exact) target RA, DEC)
        #reminder target_ra and target_dec are the emission line RA, Dec (inconsistent naming on my part)
        if (self.master_cutout) and (target_ra) and (target_dec):
            # window=error*4
            ext = error * 1.5 #to be consistent with self.master_cutout scale set to error window *3 and ext = /2
            # resizing (growing) is a problem since the master_cutout is stacked
            # (could shrink (cutout smaller section) but not grow without re-stacking larger cutouts of filters)
            plt.subplot(gs[0, cols - 1])

            empty_sci = science_image.science_image()
            # need a new cutout since we rescaled the ext (and window) size
            cutout = empty_sci.get_cutout(target_ra, target_dec, error, window=ext * 2, image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            vmin, vmax = empty_sci.get_vrange(cutout.data)
            plt.imshow(cutout.data, origin='lower', interpolation='none',
                       cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
            plt.title("Master Cutout (Stacked)")
            plt.xlabel("arcsecs")
            # plt.ylabel("arcsecs")

            # plt.set_xticklabels([str(ext), str(ext / 2.), str(0), str(-ext / 2.), str(-ext)])
            plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
            plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

            # mark the bid target location on the master cutout
            px, py = empty_sci.get_position(target_ra, target_dec, cutout)
            x, y = empty_sci.get_position(ra, dec, cutout)

            # set the diameter of the cirle to half the error (radius error/4)
            plt.gca().add_patch(plt.Circle(((x - px), (y - py)), radius=error / 4.0, color='yellow', fill=False))

            # this is correct, do not rotate the yellow rectangle (it is a zoom window only)
            #x = (x - px) - error
            #y = (y - py) - error
            #plt.gca().add_patch(plt.Rectangle((x, y), width=error * 2, height=error * 2,
            #                                  angle=0.0, color='yellow', fill=False))

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

        rows = 10 #2 (use 0 for text and 1: for plots)
        cols = 1+ len(self.CatalogImages) #(use 0 for master_stacked and 1 - N for filters)

        fig_sz_x = 18 #cols * 3 # was 6 cols
        fig_sz_y = 3 #rows * 3 # was 1 or 2 rows

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
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

        if target_flux is not None:
            cont_est = self.CONT_EST_BASE*3 #self.get_f606w_max_cont(self.EXPTIME_F606W, 3, self.CONT_EST_BASE)
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
        bid_colors = self.get_bid_colors(len(bid_ras))

        index = 0 #start in the 2nd box which is index 1 (1st box is for the fiber position plot)
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=window)
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
            plt.close()
            #still need to plot relative fiber positions here
            plt.subplot(gs[1:, 0])
            return self.build_empty_cat_summary_figure(ra, dec, error, bid_ras, bid_decs, target_w=target_w,
                                                       fiber_locs=fiber_locs)
        else:
            self.master_cutout.data /= total_adjusted_exptime

        plt.subplot(gs[1:, 0])
        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)

        # complete the entry
        plt.close()
        return fig

    def build_bid_target_figure_one_line (self, cat_match, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                section_title="", bid_number=1, target_w=0, of_number=0, target_flux=None, color="k"):
        '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
        photometry images, Z_PDF, etc

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 2.
        photoz_file = None
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        # z_spec = None
        # z_spec_ref = None
        z_photoz_weighted = None

        rows = 1
        cols = 6

        if df_photoz is not None:
            photoz_file = df_photoz['file'].values[0]
            z_best = df_photoz['z_best'].values[0]
            z_best_type = df_photoz['z_best_type'].values[0]  # s = spectral , p = photometric?
            z_photoz_weighted = df_photoz['mFDa4_z_weight']

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

            if G.ZOO:
                title = title + "\nSeparation    = %g\"" \
                                % (df['distance'].values[0] * 3600)
            else:
                title = title + "\n%s\nRA = %f    Dec = %f\nSeparation    = %g\"" \
                            % (df['IAU_designation'].values[0], df['RA'].values[0], df['DEC'].values[0],
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
                title = title + "\nLyA Z (virus) = %g (red)" % la_z
                if (oii_z > 0):
                    title = title + "\nOII Z (virus) = %g (green)" % oii_z
                else:
                    title = title + "\nOII Z (virus) = N/A"

            if target_flux is not None:
                filter_fl = df['ACS_F606W_FLUX'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                if (filter_fl is not None) and (filter_fl > 0):
                    #filter_fl = filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                    filter_fl = self.micro_jansky_to_cgs(filter_fl,target_w)
                    title = title + "\nEst LyA rest-EW = %g $\AA$" \
                                    % (target_flux / filter_fl / (target_w / G.LyA_rest))

                    if target_w >= G.OII_rest:
                        title = title + "\nEst OII rest-EW = %g $\AA$" \
                                        % (target_flux / filter_fl / (target_w / G.OII_rest))
                    else:
                        title = title + "\nEst OII rest-EW = N/A"

                    #bid target info is only of value if we have a flux from the emission line
                    bid_target = match_summary.BidTarget()
                    bid_target.bid_ra = df['RA'].values[0]
                    bid_target.bid_dec = df['DEC'].values[0]
                    bid_target.distance = df['distance'].values[0] * 3600
                    bid_target.bid_flux_est_cgs = filter_fl
                    # todo: add call to line_probabilities:
                    ratio, bid_target.p_lae, bid_target.p_oii = line_prob.prob_LAE(wl_obs=target_w,
                                                                                   lineFlux=target_flux,
                                                                                   ew_obs=(target_flux / filter_fl),
                                                                                   c_obs=None, which_color=None,
                                                                                   addl_fluxes=None, sky_area=None,
                                                                                   cosmo=None, lae_priors=None,
                                                                                   ew_case=None, W_0=None, z_OII=None,
                                                                                   sigma=None)
                    if not G.ZOO:
                        if bid_target.p_lae is not None:
                            title += "\nP(LAE) = %0.3g" % bid_target.p_lae
                            if bid_target.p_oii is not None:
                                title += "  P(OII) = %0.3g" % bid_target.p_oii
                                if bid_target.p_oii > 0.0:
                                    title += "\nP(LAE)/P(OII) = %0.3g" % (bid_target.p_lae / bid_target.p_oii)

                    for c in self.CatalogImages:
                        try:
                            bid_target.add_filter(c['instrument'], c['filter'],
                                                  self.micro_jansky_to_cgs(df[c['cols'][0]].values[0], target_w),
                                                  self.micro_jansky_to_cgs(df[c['cols'][1]].values[0], target_w))
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
                    title = title + "%7s %s %s = %.5f (%.5f)\n" % (i['instrument'], i['filter'], "Flux",
                            df[i['cols'][0]].values[0], df[i['cols'][1]].values[0])


        plt.subplot(gs[0, 4])
        plt.text(0, 0, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')


        # add photo_z plot
        # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
        # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
        # get 'file'
        # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
        if df_photoz is not None:
            z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
            if z_cat is not None:
                x = z_cat['z'].values
                y = z_cat['mFDa4'].values
                plt.subplot(gs[0, -4:])
                plt.plot(x, y, zorder=1)
                plt.xlim([0, 3.6])
                # trim axis to 0 to 3.6

                if spec_z > 0:
                    plt.axvline(x=spec_z, color='gold', linestyle='solid', linewidth=3, zorder=0)

                if target_w > 0:
                    la_z = target_w / G.LyA_rest - 1.0
                    oii_z = target_w / G.OII_rest - 1.0
                    plt.axvline(x=la_z, color='r', linestyle='--', zorder=2)
                    if (oii_z > 0):
                        plt.axvline(x=oii_z, color='g', linestyle='--', zorder=2)

                plt.title("Photo Z PDF")
                plt.gca().yaxis.set_visible(False)
                plt.xlabel("Z")

        # fig holds the entire page
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
        elif len(ras) > G.MAX_COMBINE_BID_TARGETS:
            text = "Too many matching targets. Individual reports on following pages.\n\nMORE PAGES ..."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            plt.close()
            return fig


        bid_colors = self.get_bid_colors(len(ras))

        if G.ZOO:
            text = "Separation\n" + \
                   "Spec Z\n" + \
                   "Photo Z\n" + \
                   "Est LyA rest-EW\n" + \
                   "Est OII rest-EW\n" + \
                   "ACS WFC f606W Flux\n"
        else:
            text = "Separation\n" + \
                   "RA, Dec\n" + \
                   "Spec Z\n" + \
                   "Photo Z\n" + \
                   "Est LyA rest-EW\n" + \
                   "Est OII rest-EW\n" + \
                   "ACS WFC f606W Flux\n" + \
                   "P(LAE), P(OII)\n" + \
                   "P(LAE)/P(OII)\n"


        plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)

        col_idx = 0
        for r, d in zip(ras, decs):
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
                    text = text + "%g\"\n" \
                                  % (df['distance'].values[0] * 3600)
                else:
                    text = text + "%g\"\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600,df['RA'].values[0], df['DEC'].values[0])

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
                    filter_fl = df['ACS_F606W_FLUX'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                    filter_fl_err = df['ACS_F606W_FLUXERR'].values[0]
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0

                bid_target = None
                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_adj = filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
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
                        # todo: add call to line_probabilities:
                        ratio, bid_target.p_lae, bid_target.p_oii = line_prob.prob_LAE(wl_obs=target_w,
                                                                                       lineFlux=target_flux,
                                                                                       ew_obs=(target_flux / filter_fl_adj),
                                                                                       c_obs=None, which_color=None,
                                                                                       addl_fluxes=None, sky_area=None,
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

                if filter_fl < 0:
                    text = text + "%g(%g) $\\mu$Jy !?\n" % (filter_fl, filter_fl_err)
                else:
                    text = text + "%g(%g) $\\mu$Jy\n" % (filter_fl, filter_fl_err)

                if (not G.ZOO) and (bid_target is not None):
                    if bid_target.p_lae is not None:
                        text = text + "%0.3g" % bid_target.p_lae
                        if bid_target.p_oii is not None:
                            text = text + ", %0.3g" % bid_target.p_oii
                            if bid_target.p_oii > 0.0:
                                text = text + "\n%0.3g\n" % (bid_target.p_lae / bid_target.p_oii)
                            else:
                                text = text + "\nN\A\n"
                        else:
                            text += ", N/A\nN/A\n"


            else:
                text = "%s\n%f\n%f\n" % ("--",r, d)

            plt.subplot(gs[0, col_idx])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font,color=bid_colors[col_idx-1])

            # add photo_z plot
            # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
            # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
            # get 'file'
            # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
            #overplot photo Z lines

            if df_photoz is not None:
                z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
                if z_cat is not None:
                    x = z_cat['z'].values
                    y = z_cat['mFDa4'].values
                    plt.subplot(gs[0, 4:])
                    plt.plot(x, y, color=bid_colors[col_idx-1])
                    plt.xlim([0, 3.6])
                    # trim axis to 0 to 3.6

                    if spec_z > 0:
                        #plt.axvline(x=spec_z, color='gold', linestyle='solid', linewidth=3, zorder=0)
                        plt.scatter([spec_z,],[plt.gca().get_ylim()[1]*0.9,],zorder=9,
                                 marker="o",s=80,facecolors='none',edgecolors=bid_colors[col_idx-1])

                    if col_idx == 1:
                        legend = []
                        if target_w > 0:
                            la_z = target_w / G.LyA_rest - 1.0
                            oii_z = target_w / G.OII_rest - 1.0
                            if (oii_z > 0):
                                h = plt.axvline(x=oii_z, color='g', linestyle='--', zorder=9,
                                                label="OII Z(virus) = % g" % oii_z)
                                legend.append(h)
                            h = plt.axvline(x=la_z, color='r', linestyle='--', zorder=9,
                                label="LyA Z (VIRUS) = %g" % la_z)
                            legend.append(h)

                            plt.gca().legend(handles=legend, loc='lower center', ncol=len(legend), frameon=False,
                                                 fontsize='small', borderaxespad=0, bbox_to_anchor=(0.5, -0.25))

                    plt.title("Photo Z PDF")
                    plt.gca().yaxis.set_visible(False)
                    #plt.xlabel("Z")

                  #  if len(legend) > 0:
                  #      plt.gca().legend(handles=legend, loc = 'lower center', ncol=len(legend), frameon=False,
                  #                      fontsize='small', borderaxespad=0,bbox_to_anchor=(0.5,-0.25))


            # fig holds the entire page
        plt.close()
        return fig

#######################################
# end class GOODS_N
#######################################
