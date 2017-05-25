from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import global_config as G
import os.path as op

import matplotlib
matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field


#todo: future ... see if can reorganize and use this as a wrapper and maintain only one Figure per report
class Page:
    def __init__(self,num_entries):
        self.num_entries = num_entries
        self.rows_per_entry = 0
        self.cols_per_entry = 0
        self.grid = None
        self.current_entry = 0
        self.current_row = 0
        self.current_col = 0
        self.figure = None


    @property
    def col(self):
        return self.current_col
    @property
    def row(self):
        return self.current_row

    @property
    def entry(self):
        return self.current_entry

    @property
    def gs(self):
        return self.grid

    @property
    def fig(self):
        return self.figure

    def build_grid(self,rows,cols):
        self.grid = gridspec.GridSpec(self.num_entries * rows, cols, wspace=0.25, hspace=0.5)
        self.rows_per_entry = rows
        self.cols_per_entry = cols
        self.figure = plt.figure(figsize=(self.num_entries*rows*3,cols*3))

    def next_col(self):
        if (self.current_col == self.cols_per_entry):
            self.current_col = 0
            self.current_row += 1
        else:
            self.current_col += 1

    def next_row(self):
        self.current_row += 1

    def next_entry(self):
        self.current_entry +=1
        self.current_row +=1
        self.current_col = 0

__metaclass__ = type
class Catalog:
    MainCatalog = None
    Name = "Generic Catalog (Base)"
    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min':None,'RA_max':None,'Dec_min':None, 'Dec_max':None}
    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    df = None  # pandas dataframe ... all instances share the same frame
    df_photoz = None
    status = -1

    def __init__(self):
        self.pages = None #list of bid entries (rows in the pdf)
        self.dataframe_of_bid_targets = None

    @property
    def ok(self):
        return (self.status == 0)

    @property
    def name(self):
        return (self.Name)

    @classmethod
    def position_in_cat(cls, ra, dec, error = 0.0):  # error assumed to be small and this is approximate anyway
        """Simple check for ra and dec within a rectangle defined by the min/max of RA,DEC for the catalog.
        RA and Dec in decimal degrees.
        """
        result = False

        if (cls.Cat_Coord_Range['RA_min'] is None) and (cls.Image_Coord_Range['RA_min'] is None) and (cls.df is None):
            cls.read_main_catalog()

        try:
            #either in the catalog range OR in the Image range
            if (cls.Cat_Coord_Range['RA_min'] is not None):
                result = (ra >=  (cls.Cat_Coord_Range['RA_min'] - error)) and \
                         (ra <=  (cls.Cat_Coord_Range['RA_max'] + error)) and \
                         (dec >= (cls.Cat_Coord_Range['Dec_min'] - error)) and \
                         (dec <= (cls.Cat_Coord_Range['Dec_max'] + error))
        except:
            result = False

        try:
            if (not result) and (cls.Image_Coord_Range['RA_min'] is not None):
                result = (ra >=  (cls.Image_Coord_Range['RA_min'] - error)) and \
                         (ra <=  (cls.Image_Coord_Range['RA_max'] + error)) and \
                         (dec >= (cls.Image_Coord_Range['Dec_min'] - error)) and \
                         (dec <= (cls.Image_Coord_Range['Dec_max'] + error))
        except:
            pass #keep the result as is

        return result

    @classmethod
    def read_main_catalog(cls):
        if cls.df is not None:
            log.debug("Already built df")
        elif cls.MainCatalog is not None:
            try:
                print("Reading main catalog for ", cls.Name)
                cls.df = cls.read_catalog(cls.MainCatalog, cls.Name)
                cls.status = 0

                #also check vs. by hand
                ra_min = cls.Cat_Coord_Range['RA_min']
                ra_max = cls.Cat_Coord_Range['RA_max']
                dec_min = cls.Cat_Coord_Range['Dec_min']
                dec_max = cls.Cat_Coord_Range['Dec_max']

                cls.Cat_Coord_Range['RA_min'] = cls.df['RA'].min()
                cls.Cat_Coord_Range['RA_max'] = cls.df['RA'].max()
                cls.Cat_Coord_Range['Dec_min'] = cls.df['DEC'].min()
                cls.Cat_Coord_Range['Dec_max']= cls.df['DEC'].max()

                if ra_min is not None:
                    if  (abs(ra_min - cls.Cat_Coord_Range['RA_min']) > 1e-6 ) or \
                        (abs(ra_max - cls.Cat_Coord_Range['RA_max']) > 1e-6 ) or \
                        (abs(dec_min - cls.Cat_Coord_Range['Dec_min']) > 1e-6 )or \
                        (abs(dec_max - cls.Cat_Coord_Range['Dec_max']) > 1e-6 ):
                        print("Warning! Pre-defined catalog coordinate ranges may have changed. Please check class "
                              "definitions for %s" %(cls.Name))
                        log.info("Warning! Pre-defined catalog coordinate ranges may have changed. Please check class "
                              "definitions for %s.\nPre-defined ranges: RA [%f - %f], Dec [%f -%f]\n"
                                 "Runtime ranges: RA [%f - %f], Dec [%f -%f]"
                                 %(cls.Name,ra_min,ra_max,dec_min,dec_max,cls.Cat_Coord_Range['RA_min'],
                                   cls.Cat_Coord_Range['RA_max'],cls.Cat_Coord_Range['Dec_min'],
                                   cls.Cat_Coord_Range['Dec_max']))

                log.debug(cls.Name + " Coordinate Range: RA: %f to %f , Dec: %f to %f"
                          % (cls.Cat_Coord_Range['RA_min'], cls.Cat_Coord_Range['RA_max'],
                             cls.Cat_Coord_Range['Dec_min'], cls.Cat_Coord_Range['Dec_max']))
            except:
                print("Failed")
                cls.status = -1
                log.error("Exception in read_main_catalog for %s" % cls.Name, exc_info=True)

        else:
            print("No catalog defined for ", cls.Name)

        if cls.df is None:
            cls.status = -1
        return

    def sort_bid_targets_by_likelihood(self,ra,dec):
        #right now, just by euclidean distance (ra,dec are of target)
        #todo: if radial separation is greater than the error, remove this bid target?
        #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)
        self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra)**2 +
                                                            (self.dataframe_of_bid_targets['DEC'] - dec)**2)
        self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)


    #must be defined in child class
    def build_bid_target_reports(self, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None):
        return None

    def clear_pages(self):
        if self.pages is None:
            self.pages = []
        elif len(self.pages) > 0:
            del self.pages[:]

    def add_bid_entry(self, entry):
        if self.pages is None:
            self.clear_pages()
        self.pages.append(entry)

