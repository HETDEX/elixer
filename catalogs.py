#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import pandas as pd
import global_config
from astropy.coordinates import Angle



log = global_config.logging.getLogger('Cat_logger')
log.setLevel(global_config.logging.DEBUG)

#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:

class Catalog:

    CatalogLocation = None
    Name = "Generic Catalog (Base)"
    df = None  # pandas dataframe ... all instances share the same frame
    RA_min = None
    RA_max = None
    Dec_min = None
    Dec_max = None
    status = -1

    def __init__(self):
        pass

    @property
    def ok(self):
        return (self.status == 0)

    @property
    def name(self):
        return (self.Name)

    @classmethod
    def position_in_cat(cls, ra, dec):  # error assumed to be small and this is approximate anyway
        '''Simple check for ra and dec within a rectangle defined by the min/max of RA,DEC for the catalog.
        RA and Dec in decimal degrees.
        '''
        if cls.ok:
            try:
                result = (ra >= cls.RA_min) and (ra <= cls.RA_max) and (dec >= cls.Dec_min) and (dec <= cls.Dec_max)
            except:
                result = False
        else:
            result = False
        return result


#specific implementation of The CANDELS-EGS Multi-wavelength catalog Stefanon et al., 2016
#CandlesEgsStefanon2016

class CANDELS_EGS_Stefanon_2016(Catalog):
#RA,Dec in decimal degrees

    #class variables
    CatalogLocation = "/home/dustin/code/python/voltron/data/EGS/photometry/CANDELS.EGS.F160W.v1_1.photom.cat"
    Name = "CANDELS_EGS_Stefanon_2016"


    def __init__(self):
        self.dataframe_of_bid_targets = None
        self.read_catalog()

    @classmethod
    def read_catalog(cls):

        if cls.df is not None:
            log.debug("Already built df")
            return

        log.debug("Building " + cls.Name + " dataframe...")
        idx = []
        header = []
        skip = 0
        try:
            f = open(cls.CatalogLocation, mode='r')
        except:
            log.error(cls.Name + " Exception attempting to open catalog file: " + cls.CatalogLocation)
            cls.status = -1
            return


        line = f.readline()
        while '#' in line:
            skip += 1
            toks = line.split()
            idx.append(toks[1])
            header.append(toks[2])
            line = f.readline()

        f.close()

        try:
            cls.df = pd.read_csv(cls.CatalogLocation, names=header,
                delim_whitespace=True, header=None, index_col=0, skiprows=skip)
        except:
            log.error(cls.Name + " Exception attempting to build pandas dataframe")
            cls.status = -1
            return

        cls.status = 0

        cls.RA_min = cls.df['RA'].min()
        cls.RA_max = cls.df['RA'].max()
        cls.Dec_min = cls.df['DEC'].min()
        cls.Dec_max = cls.df['DEC'].max()

    @classmethod
    def coordinate_range(cls,echo=False):
        if echo:
            msg = "RA (%f, %f)" % (cls.RA_min, cls.RA_max) + "Dec(%f, %f)" % (cls.Dec_min, cls.Dec_max)
            print( msg )
        log.debug(cls.Name + " Simple Coordinate Box: " + msg )
        return (cls.RA_min, cls.RA_max, cls.Dec_min, cls.Dec_max)

    def build_list_of_bid_targets(self,ra,dec,error):
        '''ra and dec in decimal degress. error in arcsec.
        returns a pandas dataframe'''
        self.dataframe_of_bid_targets = None

        e = float(error)/3600.0
        ra_min = float(ra - e)
        ra_max = float(ra + e)
        dec_min = float(dec - e)
        dec_max = float(dec + e)

        try:
            self.dataframe_of_bid_targets = self.df[(self.df['RA'] > ra_min) & (self.df['RA'] < ra_max) &
                                                (self.df['DEC'] > dec_min) & (self.df['DEC'] < dec_max)]
        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets")

        if self.dataframe_of_bid_targets is not None:
            log.debug(self.Name + " searching for objects in [%f - %f, %f - %f] " %(ra_min,ra_max,dec_min,dec_max) +
                  ". Found = %d" % (self.dataframe_of_bid_targets['RA'].count()))

        return self.dataframe_of_bid_targets