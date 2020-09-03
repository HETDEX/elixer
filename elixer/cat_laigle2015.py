"""
Laigle+2015
in COSMOS
RA: ~ 149.0 - 151.few
DEC: ~ 1.4 - 3.0
and a weird handfull around RA, Dec (3.0,1.0)

This is not a full ELiXer style catalog but is a suplement to cat_stack_cosmos

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

class LAIGLE2015(cat_base.Catalog):

    # class variables
    LAIGLE2015_BASE_PATH = G.COSMOS_LAIGLE_BASE_PATH

    #these catalog files are synchronized s\t the pdz "ID" == cat "NUMBER"
    #and each == entry's index+1
    LAIGLE2015_PDZ_PATH = op.join(G.COSMOS_LAIGLE_CAT_PATH, "pdz_cosmos2015_v1.3.fits")
    LAIGLE2015_CAT_PATH = op.join(G.COSMOS_LAIGLE_CAT_PATH, "COSMOS2015_Laigle+_v1.1.fits")

    MainCatalog = LAIGLE2015_CAT_PATH
    Name = "Laigle+2015"

    # if multiple images, the composite broadest range (filled in by hand)
    Cat_Coord_Range = {'RA_min': 149.28424, 'RA_max': 150.99477, 'Dec_min': 1.4354018, 'Dec_max':3.0448198}
    WCS_Manual = False

    CONT_EST_BASE = 0.0

    PDZ_Table = None
    CAT_Table = None

    def __init__(self):
        super(LAIGLE2015, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

    @classmethod
    def read_catalog(cls):

        log.debug("Building " + cls.Name + " dataframe...")
        cls.CAT_Table = None
        cls.PDZ_Table = None
        try:
            cls.CAT_Table = Table.read(cls.LAIGLE2015_CAT_PATH)
        except:
            log.error(cls.Name + " Exception attempting to open catalog file: " + cls.LAIGLE2015_CAT_PATH, exc_info=True)

        try:
            cls.PDZ_Table = Table.read(cls.LAIGLE2015_PDZ_PATH)
        except:
            log.error(cls.Name + " Exception attempting to open catalog file: " + cls.LAIGLE2015_PDZ_PATH, exc_info=True)


    def query_catalog(self,ra,dec,error):
        """

        :param ra: in decimal degrees
        :param dec:  in decimal degrees
        :param error: in arcsec
        :return:
        """

        try:
            err = error/3600.
            ids = []

            #find the entry IDs
            if self.PDZ_Table is not None:
                sel = ( (self.PDZ_Table["RA"] > ra - err) & (self.PDZ_Table["RA"] < ra + err) &
                        (self.PDZ_Table["DEC"] > dec - err) & (self.PDZ_Table["DEC"] < dec + err))
                ids = self.PDZ_Table["ID"][sel]
            elif self.CAT_Table is not None:
                sel = ((self.CAT_Table["ALPHA_J2000"] > ra - err) & (self.CAT_Table["ALPHA_J2000"] < ra + err) &
                       (self.CAT_Table["DELTA_J2000"] > dec - err) & (self.CAT_Table["DELTA_J2000"] < dec + err))
                ids = self.CAT_Table["NUMBER"][sel]
            else:
                #problem
                log.error("Cannot query LAIGLE+2015 catalog. Tables are None.")


            #so ... what to return? and in what format (list of dictionaries?)
            # will want to ADD a distance to ra, dec for each found entry
            idx = np.array(ids)-1
            cat_rows = []
            pdz_rows = []

            if self.CAT_Table:
                cat_rows = self.CAT_Table["NUMBER","ALPHA_J2000","DELTA_J2000","B_MAG_AUTO","B_MAGERR_AUTO",
                                      "V_MAG_AUTO","V_MAGERR_AUTO","r_MAG_AUTO","r_MAGERR_AUTO",
                                      "PHOTOZ"][idx]
                cat_rows['distance'] = 999.9 #needs to be a float type
                for i in range(len(cat_rows)):
                    cat_rows['distance'][i] = utilities.angular_distance(ra,dec,
                                                cat_rows["ALPHA_J2000"][i],cat_rows["DELTA_J2000"][i])

            if self.PDZ_Table:
                for i in idx: #a list of 400 columns per row (z0.00 to z4.00)
                    pdz_rows.append(np.array(list(self.PDZ_Table[i])[74:474]))

            #now compbine cat_rows, and pdz_pdfs and return

            return cat_rows, pdz_rows #just for the moment

        except:
            log.error("Exception in cat_laigle2015.query_catalog",exc_info=True)

        return None

    # def sort_bid_targets_by_likelihood(self, ra, dec):
    #     # right now, just by euclidean distance (ra,dec are of target)
    #     #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)
    #     self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra) ** 2 +
    #                                                         (self.dataframe_of_bid_targets['DEC'] - dec) ** 2)
    #     self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)

    def get_filter_flux(self,df): #right now, only g-band catalog

        filter_fl = None
        filter_fl_err = None
        mag = None
        mag_bright = None
        mag_faint = None
        filter_str = None
        try:

            filter_str = 'g'
            dfx = df
            #dfx = df.loc[df['FILTER']==filter_str]
            #
            #
            # if (dfx is None) or (len(dfx)==0):
            #     filter_str = 'r'
            #     dfx = df.loc[df['FILTER'] == filter_str]
            #
            # if (dfx is None) or (len(dfx)==0):
            #     filter_str = '?'
            #     log.error("Neither g-band nor r-band filter available.")
            #     return filter_fl, filter_fl_err, mag, mag_plus, mag_minus, filter_str

            filter_fl = dfx['FLUX_AUTO'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
            filter_fl_err = dfx['FLUXERR_AUTO'].values[0]

            mag = dfx['MAG_AUTO'].values[0]
            mag_faint = dfx['MAGERR_AUTO'].values[0]
            mag_bright = -1 * mag_faint

            #something is way wrong with the MAG_AUTO
            mag, mag_bright, mag_faint= self.micro_jansky_to_mag(filter_fl,filter_fl_err)

        except: #not the EGS df, try the CFHTLS
            pass

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

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
#end class LAIGLE2015
#######################################