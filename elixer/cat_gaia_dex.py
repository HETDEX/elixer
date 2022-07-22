"""
GAIA HETDEX value added catalog from Keith Hawkins and Greg Zeimann

gmag limit <~ 21

"""

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

import os.path as op
import copy
import io

#STACK_COSMOS_BASE_PATH = G.STACK_COSMOS_BASE_PATH
#STACK_COSMOS_IMAGE = G.STACK_COSMOS_BASE_PATH#op.join(G.STACK_COSMOS_BASE_PATH,"COSMOS_g_sci.fits")
#STACK_COSMOS_CAT = G.STACK_COSMOS_CAT_PATH#op.join(G.STACK_COSMOS_CAT_PATH,"cat_g.fits")


import matplotlib
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import astropy.io.fits as fits
from astropy.table import Table
from astropy.table import Table

log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

class GAIA_DEX(cat_base.Catalog):

    # class variables
    GAIA_DEX_BASE_PATH = G.GAIA_DEX_BASE_PATH
    GAIA_DEX_CAT_PATH = G.GAIA_DEX_BASE_PATH

    MainCatalog = GAIA_DEX_CAT_PATH
    Name = "GAIA_DEX"

    # if multiple images, the composite broadest range (filled in by hand)
    Cat_Coord_Range = {'RA_min': 5.875, 'RA_max': 335.046, 'Dec_min':-1.105, 'Dec_max':67.861}
    WCS_Manual = False

    CONT_EST_BASE = 0.0

    CAT_Table = None

    def __init__(self):
        super(GAIA_DEX, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

    @classmethod
    def read_catalog(cls):

        log.debug("Building " + cls.Name + " dataframe...")
        cls.CAT_Table = None
        try:
            cls.CAT_Table = Table.read(cls.GAIA_DEX_CAT_PATH,format='fits')

            #keep only the columns we want
            cls.CAT_Table.keep_columns(['source_id','ra','ra_error','dec','dec_error','pmra','pmra_error','pmdec',
                                        'pmdec_error','phot_g_mean_mag','phot_rp_mean_mag'])#,'radial_velocity','radial_velocity_error'])

        except:
            log.error(cls.Name + " Exception attempting to open catalog file: " + cls.GAIA_DEX_CAT_PATH, exc_info=True)

    def query_catalog(self,ra,dec,error):
        """

        :param ra: in decimal degrees
        :param dec:  in decimal degrees
        :param error: in arcsec
        :return: astropy table (with subset of columns),
                 list of arrays where [0] of each array is the 'ID' or 'NUMBER' matching the table
        """

        try:
            err = error/3600.
            ids = []

            #find the entry IDs
            if self.CAT_Table is not None:
                sel = ((self.CAT_Table["ra"] > ra - err) & (self.CAT_Table["ra"] < ra + err) &
                       (self.CAT_Table["dec"] > dec - err) & (self.CAT_Table["dec"] < dec + err))
                ids = self.CAT_Table["source_id"][sel]
            else:
                #problem
                log.error("Cannot query GAIA_DEX catalog. Tables are None.")


            #so ... what to return? and in what format (list of dictionaries?)
            # will want to ADD a distance to ra, dec for each found entry
            idx = np.array(ids)-1
            cat_rows = []
            pdz_rows = []

            if self.CAT_Table:
                cat_rows = self.CAT_Table[idx]

                if cat_rows is not None:
                    cat_rows.rename_column('ra','RA')
                    cat_rows.rename_column('dec','DEC')
                    cat_rows.rename_column("source_id","ID")
                    if len(cat_rows) > 0:
                        cat_rows['distance'] = 999.9 #needs to be a float type
                        cat_rows['nonzero_pm'] = -1 # 0 = False, 1= True
                        for i in range(len(cat_rows)):
                            cat_rows['distance'][i] = utilities.angular_distance(ra,dec,
                                                        cat_rows["RA"][i],cat_rows["DEC"][i])

                            #if the proper motion in RA or Dec is more than 1.3 times the error
                            #todo: should add a minimum total proper motion to trigger?
                            pm = (abs(cat_rows['pmra'][i]) > 1.3 * cat_rows['pmra_error'][i]) or \
                                 (abs(cat_rows['pmdec'][i]) > 1.3 * cat_rows['pmdec_error'][i])
                            cat_rows['nonzero_pm'][i] = 1 if pm else 0

            return cat_rows

        except:
            log.error("Exception in cat_gaia_dex.query_catalog",exc_info=True)

        return None

    # def sort_bid_targets_by_likelihood(self, ra, dec):
    #     # right now, just by euclidean distance (ra,dec are of target)
    #     #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)
    #     self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra) ** 2 +
    #                                                         (self.dataframe_of_bid_targets['DEC'] - dec) ** 2)
    #     self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)

    def get_mag(self,df,filter='g'):
        """

        :param df:
        :param filter:
        :return: mag, mag_bright, mag_faint
        """

        #don't have errors right now (could use fluxes and find the band pass definition for f_iso)
        try:
            if filter.lower()=='g':
                return df['phot_g_mean_mag'],None,None
            elif filter.lower()=='r':
                return df['phot_rp_mean_mag'],None,None
            else:
                return  None,None,None
        except:
            return  None,None,None


    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        if self.df is None:
            self.read_main_catalog()

        error_in_deg = np.float64(error) / 3600.0

        self.dataframe_of_bid_targets = None
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

            log.info(
                self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
                ". Found = %d" % (self.num_targets))

        # extra None for compatibility with catalogs that have photoZ
        return self.num_targets, self.dataframe_of_bid_targets, None

#######################################
#end class GAIA_DEX
#######################################