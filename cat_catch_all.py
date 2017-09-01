from __future__ import print_function

import global_config as G
import os.path as op


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


class CATCH_ALL(cat_base.Catalog):

    # class variables
    MainCatalog = None
    Name = "CATCH-ALL"
    # if multiple images, the composite broadest range (filled in by hand)
    #cover all sphere in image (will be blank) but invalid range in actual catalog
    Cat_Coord_Range = {'RA_min':361.0 , 'RA_max': 361.1, 'Dec_min': 91.0, 'Dec_max': 91.1}
    Image_Coord_Range = {'RA_min': 0.0, 'RA_max': 360.0, 'Dec_min': -90.0, 'Dec_max': 90.0}


    WCS_Manual = False
    EXPTIME_F606W = 1.0 #289618.0
    CONT_EST_BASE = 1.0
    BidCols = []  # NOTE: there are no F105W values

    CatalogImages = [ ]

    PhotoZCatalog = None
    SupportFilesLocation = None

    def __init__(self):
        super(CATCH_ALL, self).__init__()

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
        pass

    @classmethod
    def read_photoz_catalog(cls):
        pass

    @classmethod
    def read_catalog(cls, catalog_loc, name):
        pass

    # column names are catalog specific, but could map catalog specific names to generic ones and produce a dictionary?
    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None,target_flux=None):

        self.clear_pages()
        self.build_list_of_bid_targets(target_ra, target_dec, error)

        ras = []
        decs = []

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

        return self.pages

    def build_exact_target_location_figure(self, ra, dec, error, section_title="", target_w=0, fiber_locs=None,
                                           target_flux=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 4

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


    def build_cat_summary_figure (self, ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

        rows = 10 #2 (use 0 for text and 1: for plots)
        cols = 7 #just going to use the first, but this sets about the right size

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
        title = "No overlapping imaging catalog."

        plt.subplot(gs[0, :])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        plt.subplot(gs[1:, 0])
        self.add_empty_catalog_fiber_positions(plt, ra, dec, fiber_locs)

        # complete the entry
        plt.close()
        return fig


#######################################
# end class CATCH_ALL
#######################################
