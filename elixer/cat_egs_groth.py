from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_base
    from elixer import match_summary
    from elixer import utilities
    from elixer import spectrum_utilities as SU
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import utilities
    import spectrum_utilities as SU

import os.path as op

EGS_GROTH_BASE_PATH = G.EGS_GROTH_BASE_PATH
EGS_GROTH_CAT_PATH = G.EGS_GROTH_CAT_PATH
#EGS_GROTH_IMAGE = op.join(G.EGS_GROTH_BASE_PATH,"groth.fits")

import matplotlib
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import astropy.io.fits as fits
from astropy.table import Table
#from astropy.io import ascii #note: this works, but pandas is much faster


#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field


class EGS_GROTH(cat_base.Catalog):
    #class variables
    EGS_GROTH_BASE_PATH = G.EGS_GROTH_BASE_PATH
    EGS_GROTH_CAT = None #op.join(G.EGS_GROTH_CAT_PATH, "")
    EGS_GROTH_IMAGE_PATH = G.EGS_GROTH_BASE_PATH
    EGS_GROTH_IMAGE = op.join(G.EGS_GROTH_BASE_PATH,"groth.fits")

    #there is no catalog??
    MainCatalog = EGS_GROTH_CAT
    Name = "EGS_GROTH"
    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}
    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}
    WCS_Manual = False

    AstroTable = None

    BidCols = [ ]

    CatalogImages = [
        {'path': EGS_GROTH_IMAGE_PATH,
         'name': 'groth.fits',
         'filter': 'unknown',
         'instrument': 'unknown',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 1.
         }]


    def __init__(self):
        super(EGS_GROTH, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

        #only on demand
        #self.read_main_catalog()

        self.master_cutout = None

    @classmethod
    def read_catalog(cls, catalog_loc, name):
        "This catalog is in a fits file"

        log.debug("Building " + name + " dataframe...")
        try:
            #f = fits.open(catalog_loc)[1].data
            table = Table.read(catalog_loc)
        except:
            log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
            return None

        #convert into a pandas dataframe ... cannot convert directly to pandas because of the [25] lists
        #so build a pandas df with just the few columns we need for searching
        #then pull data from full astro table as needed

        try:
            lookup_table = Table([table['NUMBER'], table['ALPHA_J2000'], table['DELTA_J2000']])
            df = lookup_table.to_pandas()
            old_names = ['NUMBER','ALPHA_J2000','DELTA_J2000']
            new_names = ['ID','RA','DEC']
            df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
            cls.AstroTable = table
        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return df


#######################################
#end class EGS_GROTH
#######################################
