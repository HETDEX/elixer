from __future__ import print_function

import global_config as G
import os.path as op


CANDELS_EGS_Stefanon_2016_BASE_PATH = G.CANDELS_EGS_Stefanon_2016_BASE_PATH
CANDELS_EGS_Stefanon_2016_CAT = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH,
                                        "photometry/CANDELS.EGS.F160W.v1_1.photom.cat")
CANDELS_EGS_Stefanon_2016_IMAGES_PATH = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH, "images")
CANDELS_EGS_Stefanon_2016_PHOTOZ_CAT = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH , "photoz/zcat_EGS_v2.0.cat")
CANDELS_EGS_Stefanon_2016_PHOTOZ_ZPDF_PATH = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH, "photoz/zPDF/")

import matplotlib
matplotlib.use('agg')

import pandas as pd
import science_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.transforms import Affine2D


log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

import cat_base


class CANDELS_EGS_Stefanon_2016(cat_base.Catalog):
    # RA,Dec in decimal degrees

    # photometry catalog
    #  1 ID #  2 IAU_designation  #  3 RA  #  4 DEC #  5 RA_Lotz2008 (RA in AEGIS ACS astrometric system)  #  6 DEC_Lotz2008 (DEC in AEGIS ACS astrometric system)
    #  7 FLAGS #  8 CLASS_STAR #  9 CFHT_U_FLUX  # 10 CFHT_U_FLUXERR # 11 CFHT_g_FLUX # 12 CFHT_g_FLUXERR # 13 CFHT_r_FLUX
    # 14 CFHT_r_FLUXERR  # 15 CFHT_i_FLUX # 16 CFHT_i_FLUXERR # 17 CFHT_z_FLUX # 18 CFHT_z_FLUXERR # 19 ACS_F606W_FLUX
    # 20 ACS_F606W_FLUXERR # 21 ACS_F814W_FLUX # 22 ACS_F814W_FLUXERR # 23 WFC3_F125W_FLUX # 24 WFC3_F125W_FLUXERR
    # 25 WFC3_F140W_FLUX # 26 WFC3_F140W_FLUXERR # 27 WFC3_F160W_FLUX # 28 WFC3_F160W_FLUXERR # 29 WIRCAM_J_FLUX
    # 30 WIRCAM_J_FLUXERR  # 31 WIRCAM_H_FLUX # 32 WIRCAM_H_FLUXERR # 33 WIRCAM_K_FLUX # 34 WIRCAM_K_FLUXERR
    # 35 NEWFIRM_J1_FLUX # 36 NEWFIRM_J1_FLUXERR # 37 NEWFIRM_J2_FLUX # 38 NEWFIRM_J2_FLUXERR # 39 NEWFIRM_J3_FLUX # 40 NEWFIRM_J3_FLUXERR
    # 41 NEWFIRM_H1_FLUX # 42 NEWFIRM_H1_FLUXERR # 43 NEWFIRM_H2_FLUX # 44 NEWFIRM_H2_FLUXERR # 45 NEWFIRM_K_FLUX# 46 NEWFIRM_K_FLUXERR
    # 47 IRAC_CH1_FLUX # 48 IRAC_CH1_FLUXERR # 49 IRAC_CH2_FLUX # 50 IRAC_CH2_FLUXERR # 51 IRAC_CH3_FLUX # 52 IRAC_CH3_FLUXERR # 53 IRAC_CH4_FLUX
    # 54 IRAC_CH4_FLUXERR # 55 ACS_F606W_V08_FLUX # 56 ACS_F606W_V08_FLUXERR # 57 ACS_F814W_V08_FLUX # 58 ACS_F814W_V08_FLUXERR
    # 59 WFC3_F125W_V08_FLUX # 60 WFC3_F125W_V08_FLUXERR # 61 WFC3_F160W_V08_FLUX # 62 WFC3_F160W_V08_FLUXERR
    # 63 IRAC_CH3_V08_FLUX # 64 IRAC_CH3_V08_FLUXERR # 65 IRAC_CH4_V08_FLUX # 66 IRAC_CH4_V08_FLUXERR # 67 DEEP_SPEC_Z

    # class variables
    MainCatalog = CANDELS_EGS_Stefanon_2016_CAT
    Name = "CANDELS_EGS_Stefanon_2016"
    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}
    Cat_Coord_Range = {'RA_min': 214.576759, 'RA_max': 215.305229, 'Dec_min': 52.677569, 'Dec_max': 53.105756}
    WCS_Manual = True
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
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits',
         'filter': 'f606w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F606W_FLUX", "ACS_F606W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_acs_wfc_f814w_060mas_v1.1_drz.fits',
         'filter': 'f814w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F814W_FLUX", "ACS_F814W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f105w_060mas_v1.5_drz.fits',
         'filter': 'f105w',
         'instrument': 'WFC3',
         'cols': [],
         'labels': [],
         'image': None
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f125w_060mas_v1.1_drz.fits',
         'filter': 'f125w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f140w_060mas_v1.1_drz.fits',
         'filter': 'f140w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f160w_060mas_v1.1_drz.fits',
         'filter': 'f160w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F160W_FLUX", "WFC3_F160W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None
         }
    ]

    # 1 file # 2 ID (CANDELS.EGS.F160W.v1b_1.photom.cat) # 3 RA (CANDELS.EGS.F160W.v1b_1.photom.cat) # 4 DEC (CANDELS.EGS.F160W.v1b_1.photom.cat)
    # 5 z_best # 6 z_best_type # 7 z_spec # 8 z_spec_ref # 9 z_grism # 10 mFDa4_z_peak # 11 mFDa4_z_weight # 12 mFDa4_z683_low
    # 13 mFDa4_z683_high # 14 mFDa4_z954_low # 15 mFDa4_z954_high # 16 HB4_z_peak # 17 HB4_z_weight # 18 HB4_z683_low
    # 19 HB4_z683_high # 20 HB4_z954_low # 21 HB4_z954_high # 22 Finkelstein_z_peak # 23 Finkelstein_z_weight
    # 24 Finkelstein_z683_low # 25 Finkelstein_z683_high # 26 Finkelstein_z954_low # 27 Finkelstein_z954_high
    # 28 Fontana_z_peak # 29 Fontana_z_weight # 30 Fontana_z683_low # 31 Fontana_z683_high # 32 Fontana_z954_low
    # 33 Fontana_z954_high # 34 Pforr_z_peak # 35 Pforr_z_weight # 36 Pforr_z683_low # 37 Pforr_z683_high
    # 38 Pforr_z954_low # 39 Pforr_z954_high # 40 Salvato_z_peak # 41 Salvato_z_weight # 42 Salvato_z683_low
    # 43 Salvato_z683_high # 44 Salvato_z954_low # 45 Salvato_z954_high # 46 Wiklind_z_peak # 47 Wiklind_z_weight
    # 48 Wiklind_z683_low  # 49 Wiklind_z683_high # 50 Wiklind_z954_low # 51 Wiklind_z954_high # 52 Wuyts_z_peak
    # 53 Wuyts_z_weight  # 54 Wuyts_z683_low # 55 Wuyts_z683_high # 56 Wuyts_z954_low # 57 Wuyts_z954_high

    PhotoZCatalog = CANDELS_EGS_Stefanon_2016_PHOTOZ_CAT
    SupportFilesLocation = CANDELS_EGS_Stefanon_2016_PHOTOZ_ZPDF_PATH












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

        title = "Catalog: %s\n" % self.Name + section_title + "\nPossible Matches = %d (within %g\")\n" \
                                                              "RA = %f    Dec = %f\n" % (
                                                              len(self.dataframe_of_bid_targets), error, ra, dec)
        if target_w > 0:
            title = title + "Wavelength = %g $\AA$\n" % target_w
        else:
            title = title + "\n"

        if target_flux is not None:
            title = title + "Min (no match) 3$\sigma$ LyA rest-EW = %g $\AA$\n" % \
                            ( -1 * (target_flux / 9.9e-21) / (target_w / G.LyA_rest))
            if target_w >= G.OII_rest:
                title = title + "Min (no match) 3$\sigma$ OII rest-EW = %g $\AA$\n" % \
                            (-1 *(target_flux / 9.9e-21) / (target_w / G.OII_rest))
            else:
                title = title + "Min (no match) 3$\sigma$ OII rest-EW = N/A\n"

        plt.subplot(gs[0, 0])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        index = -1
        ref_exptime = None
        total_adjusted_exptime = None
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
                    self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime/ref_exptime)
                    total_adjusted_exptime += sci.exptime/ref_exptime

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

        self.add_north_box(plt, sci, self.master_cutout, error, 0,0, theta)
        #self.add_north_arrow(plt, sci, self.master_cutout, theta)


        # plot the fiber cutout
        if (fiber_locs is not None) and (len(fiber_locs) > 0):
            plt.subplot(gs[0, cols - 2])

            plt.title("Fiber Positions")
            plt.xlabel("arcsecs")
            plt.ylabel("arcsecs")

            plt.plot(0, 0, "r+")

            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')

            x, y = empty_sci.get_position(ra, dec, self.master_cutout)  # zero (absolute) position

            for r, d, c, i, dist in fiber_locs:
                # print("+++++ Cutout RA,DEC,ID,COLOR", r,d,i,c)
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx, fy = empty_sci.get_position(r, d, self.master_cutout)

                xmin = min(xmin, fx - x)
                xmax = max(xmax, fx - x)
                ymin = min(ymin, fy - y)
                ymax = max(ymax, fy - y)


                plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius, color=c, fill=False))
                plt.text((fx - x), (fy - y), str(i), ha='center', va='center', fontsize='x-small', color=c)


            ext_base = max (abs(xmin),abs(xmax),abs(ymin),abs(ymax))
            #larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
            ext = ext_base + G.Fiber_Radius

            # need a new cutout since we rescaled the ext (and window) size
            cutout = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            self.add_north_arrow(plt, sci, cutout, theta=None)

            plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

        # complete the entry
        plt.close()
        return fig

    def build_bid_target_figure(self, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                section_title="", bid_number=1, target_w=0, of_number=0,target_flux=None):
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
            title = "%s  Possible Match #%d" % (section_title,bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

            title = title + "\n%s\nRA = %f    Dec = %f\nSeparation    = %g\"" \
                    % (df['IAU_designation'].values[0], df['RA'].values[0], df['DEC'].values[0],
                    df['distance'].values[0] * 3600)
           #do not use DEEP SPEC Z, just use spec Z below
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
                    filter_fl = filter_fl * 1e-29 * 3e18 / (target_w ** 2) #3e18 ~ c in angstroms/sec
                    title = title +"\nEst LyA rest-EW = %g $\AA$" % (-1*target_flux / filter_fl / (target_w / G.LyA_rest))

                    if target_w >= G.OII_rest:
                        title = title + "\nEst OII rest-EW = %g $\AA$" % (-1*target_flux / filter_fl /(target_w / G.OII_rest))
                    else:
                        title = title + "\nEst OII rest-EW = N/A"
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

        plt.subplot(gs[0, 0])
        plt.text(0, 0.20, title, ha='left', va='bottom', fontproperties=font)
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
                plt.xticks([ext, ext / 2., 0, -ext / 2., -ext])
                plt.yticks([ext, ext / 2., 0, -ext / 2., -ext])

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

                    self.add_north_arrow(plt, sci, cutout,theta=None)

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
                plt.xlim([0,3.6])
                #trim axis to 0 to 3.6

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

        empty_sci = science_image.science_image()
        # master cutout (0,0 is the observered (exact) target RA, DEC)
        if self.master_cutout is not None:
            # window=error*4
            ext = error * 2.
            plt.subplot(gs[0, cols - 1])
            vmin, vmax = empty_sci.get_vrange(self.master_cutout.data)
            plt.imshow(self.master_cutout.data, origin='lower', interpolation='none',
                       cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
            plt.title("Master Cutout (Stacked)")
            plt.xlabel("arcsecs")
           # plt.ylabel("arcsecs")

            #plt.set_xticklabels([str(ext), str(ext / 2.), str(0), str(-ext / 2.), str(-ext)])
            plt.xticks([ext, ext / 2., 0, -ext / 2., -ext])
            plt.yticks([ext, ext / 2., 0, -ext / 2., -ext])

            # mark the bid target location on the master cutout
            if (target_ra is not None) and (target_dec is not None):
                px, py = empty_sci.get_position(target_ra, target_dec, self.master_cutout)
                x, y = empty_sci.get_position(ra, dec, self.master_cutout)


                # set the diameter of the cirle to half the error (radius error/4)
                plt.gca().add_patch(plt.Circle(((x - px), (y - py)), radius=error / 4.0, color='yellow', fill=False))

                #this is correct, do not rotate the yellow rectangle (it is a zoom window only)
                x = (x - px) - error
                y = (y - py) - error
                plt.gca().add_patch(plt.Rectangle((x, y), width=error * 2, height=error * 2,
                                                  angle=0.0, color='yellow', fill=False))

                plt.plot(0, 0, "r+")
                self.add_north_box(plt, empty_sci, self.master_cutout, error, 0, 0, theta=None)

        #fig holds the entire page
        plt.close()
        return fig












    def __init__(self):
        super(CANDELS_EGS_Stefanon_2016, self).__init__()

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
    def build_bid_target_reports(self, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None,target_flux=None):

        self.clear_pages()
        self.build_list_of_bid_targets(target_ra, target_dec, error)

        ras = self.dataframe_of_bid_targets.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:

            entry = self.build_cat_summary_figure(target_ra, target_dec, error, ras, decs,
                                                  section_title=section_title, target_w=target_w,
                                                  fiber_locs=fiber_locs, target_flux=target_flux)
        else:
            entry = self.build_exact_target_location_figure(target_ra, target_dec, error, section_title=section_title,
                                                        target_w=target_w, fiber_locs=fiber_locs,
                                                        target_flux=target_flux)
        if entry is not None:
            self.add_bid_entry(entry)

        number = 0
        # display each bid target
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

            if G.SINGLE_PAGE_PER_DETECT:
                entry = self.build_bid_target_figure_one_line(r[0], d[0], error=error, df=df, df_photoz=df_photoz,
                                                 target_ra=target_ra, target_dec=target_dec,
                                                 section_title=section_title,
                                                 bid_number=number, target_w=target_w, of_number=num_hits-base_count,
                                                 target_flux=target_flux)
            else:
                entry = self.build_bid_target_figure(r[0], d[0], error=error, df=df, df_photoz=df_photoz,
                                                 target_ra=target_ra, target_dec=target_dec,
                                                 section_title=section_title,
                                                 bid_number=number, target_w=target_w, of_number=num_hits-base_count,
                                                 target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)

        return self.pages



    def build_cat_summary_figure (self, ra, dec, error,bid_ras, bid_decs, section_title="", target_w=0,
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

        title = "Catalog: %s\n" % self.Name + section_title + "\nPossible Matches = %d (within %g\")\n" \
                                                              % (len(self.dataframe_of_bid_targets), error)

        if target_flux is not None:
            title = title + "Min (no match) 3$\sigma$ LyA rest-EW = %g $\AA$\n" % \
                            (-1 * (target_flux / 9.9e-21) / (target_w / G.LyA_rest))
            if target_w >= G.OII_rest:
                title = title + "Min (no match) 3$\sigma$ OII rest-EW = %g $\AA$\n" % \
                                (-1 * (target_flux / 9.9e-21) / (target_w / G.OII_rest))
            else:
                title = title + "Min (no match) 3$\sigma$ OII rest-EW = N/A\n"

        plt.subplot(gs[0, 0])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        index = -1
        ref_exptime = None
        total_adjusted_exptime = None

        # add the bid targets
        norm = plt.Normalize()
        bid_colors = plt.cm.brg(norm(np.arange(len(bid_ras))))

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

                plt.subplot(gs[rows - 1, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

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

        # plot the master cutout
        empty_sci = science_image.science_image()
        plt.subplot(gs[0, cols - 1])
        vmin, vmax = empty_sci.get_vrange(self.master_cutout.data)
        plt.imshow(self.master_cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                   vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
        plt.title("Master Cutout (Stacked)")
        plt.xlabel("arcsecs")
        plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

        # only show this lable if there is not going to be an adjacent fiber plot
        if (fiber_locs is None) or (len(fiber_locs) == 0):
            plt.ylabel("arcsecs")
        plt.plot(0, 0, "r+")

        theta = empty_sci.get_rotation_to_celestrial_north(self.master_cutout)
        self.add_north_box(plt, sci, self.master_cutout, error, 0, 0, theta)


        #add the bid targets
        x, y = empty_sci.get_position(ra, dec, self.master_cutout)  # zero (absolute) position
        for br,bd,bc in zip(bid_ras,bid_decs,bid_colors):
            fx, fy = empty_sci.get_position(br, bd, self.master_cutout)
            plt.gca().add_patch(plt.Rectangle(( (fx - x) - target_box_side/2.0, (fy - y) - target_box_side/2.0),
                                              width=target_box_side, height=target_box_side,
                                              angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))

        # plot the fiber cutout
        if (fiber_locs is not None) and (len(fiber_locs) > 0):
            plt.subplot(gs[0, cols - 3])

            plt.title("Fiber Positions")
            plt.xlabel("arcsecs")
            plt.ylabel("arcsecs")

            plt.plot(0, 0, "r+")

            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')

            x, y = empty_sci.get_position(ra, dec, self.master_cutout)  # zero (absolute) position

            for r, d, c, i, dist in fiber_locs:
                # print("+++++ Cutout RA,DEC,ID,COLOR", r,d,i,c)
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx, fy = empty_sci.get_position(r, d, self.master_cutout)

                xmin = min(xmin, fx - x)
                xmax = max(xmax, fx - x)
                ymin = min(ymin, fy - y)
                ymax = max(ymax, fy - y)

                plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius, color=c, fill=False))
                plt.text((fx - x), (fy - y), str(i), ha='center', va='center', fontsize='x-small', color=c)

            ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
            scale = (ext_base + G.Fiber_Radius) / ext
            ext = scale * ext

            # need a new cutout since we rescaled the ext (and window) size
            cutout = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            if scale > 1.0: #e.g. we zoomed in
                self.add_north_arrow(plt, sci, cutout, theta=None, scale=scale)
            else: #did not zoom in, so position based on original cutout
                self.add_north_arrow(plt, sci, cutout, theta=None, scale=1.0)

            plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

        # complete the entry
        plt.close()
        return fig

    def build_bid_target_figure_one_line (self, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
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

        rows = 1
        cols = 6

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
        plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

        spec_z = 0.0

        if df is not None:
            title = "%s  Possible Match #%d" % (section_title, bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

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
                    filter_fl = filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                    title = title + "\nEst LyA rest-EW = %g $\AA$" % (
                    -1 * target_flux / filter_fl / (target_w / G.LyA_rest))

                    if target_w >= G.OII_rest:
                        title = title + "\nEst OII rest-EW = %g $\AA$" % (
                        -1 * target_flux / filter_fl / (target_w / G.OII_rest))
                    else:
                        title = title + "\nEst OII rest-EW = N/A"
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

        plt.subplot(gs[0, 0])
        plt.text(0, 0.20, title, ha='left', va='bottom', fontproperties=font)
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


        plt.subplot(gs[0, 2])
        plt.text(0, 0.20, title, ha='left', va='bottom', fontproperties=font)
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
                plt.subplot(gs[0, -2:])
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

#######################################
# end class CANDELS_EGS_Stefanon_2016
#######################################
