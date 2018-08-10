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


#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

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

        ras = []
        decs = []

        # display the exact (target) location
        entry = self.build_cat_summary_figure(target_ra, target_dec, error, ras, decs,
                                            target_w=target_w, fiber_locs=fiber_locs)

        if entry is not None:
            self.add_bid_entry(entry)

        entry = self.build_multiple_bid_target_figures_one_line()

        if entry is not None:
            self.add_bid_entry(entry)

        return self.pages


    def get_stacked_cutout(self,ra,dec,window):
        return None #no cutout available

    def build_cat_summary_figure (self, ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None):
        return self.build_empty_cat_summary_figure(ra,dec,error,bid_ras, bid_decs, target_w=target_w,
                                  fiber_locs=fiber_locs)

    def build_multiple_bid_target_figures_one_line(self):

        rows = 1
        cols = 6

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)

        # col(0) = "labels", 1..3 = bid targets, 4..5= Zplot
        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        # entry text
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        # row labels
        plt.subplot(gs[0, 0])
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')


        # per Karl insert a blank row
        text = "No overlapping imagaing catalog.\nRow intentionally blank."
        plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
        plt.close()
        return fig

#######################################
# end class CATCH_ALL
#######################################
