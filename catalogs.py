import global_config as G

#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:

import cat_candles_egs_stefanon_2016
import cat_goods_n
#import cat_goods_n_finkelstein
import cat_egs_groth
import cat_stack_cosmos
import cat_shela
import cat_hsc
import cat_catch_all
#import cat_ast376_shela


class CatalogLibrary:

    def __init__(self):
        self.cats = None
        self.build_catalog_list()

        try:
            log.debug("Built catalog library")
        except:
            # first time we need to log anything
            G.logging.basicConfig(filename=G.LOG_FILENAME, level=G.LOG_LEVEL, filemode='w')

    def build_catalog_list(self):
        # build list of all catalogs below
        if (self.cats is not None) and (len(self.cats) !=0):
            del self.cats[:]

        self.cats = list()
        self.cats.append(cat_candles_egs_stefanon_2016.CANDELS_EGS_Stefanon_2016())
        self.cats.append(cat_goods_n.GOODS_N())
        # self.cats.append(cat_goods_n_finkelstein.GOODS_N_FINKELSTEIN())
        # self.cats.append(EGS_GROTH()) #this is of no value right now
        self.cats.append(cat_stack_cosmos.STACK_COSMOS())
        self.cats.append(cat_shela.SHELA())
        self.cats.append(cat_hsc.HSC())
        # self.cats.append(cat_ast376_shela.AST376_SHELA())


    def get_full_catalog_list(self):
        if self.cats is None:
            self.build_catalog_list()

        return self.cats

    def get_catch_all(self):
        return cat_catch_all.CATCH_ALL()


    def find_catalogs(self,position):
        '''
        Build a list of catalogs whose footprint contains the position

        :param position: astropy SkyCoord
        :param radius: in deccimal degrees at the moment
        :return: list of catalog objects
        '''

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()
        matched_cats = list()

        for c in self.cats:
            if c.position_in_cat(ra=ra, dec=dec, error=0):
                matched_cats.append(c)

        return matched_cats



    def get_cutouts(self,position,radius,catalogs=None):
        '''
        Return a list of dictionaries of the FITS cutouts from imaging catalogs
        (does not include objects in those catalogs, just the images)

        :param position: astropy SkyCoord
        :param radius:
        :param catalogs: optional list of catalogs to search (if not provided, searches all)
        :return: list of dictionaries of cutouts and info
        '''

        if catalogs is None:
            catalogs = self.find_catalogs(position)

        if (catalogs is None) or (len(catalogs) == 0):
            return []

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()

        l = list()

        for c in catalogs:
            cutouts = c.get_cutouts(ra, dec, radius)
            if (cutouts is not None) and (len(cutouts) > 0):
                l.extend(cutouts)

        return l


    def get_catalog_objects(self,position,radius,catalogs=None):
        '''
         Return a list of dictionaries of pandas dataframes of objects in imaging catalogs

        :param position: astropy SkyCoord
        :param radius:
        :param catalogs: optional list of catalogs to search (if not provided, searches all)
        :return: list of dictionaries of cutouts and info
        '''

        if catalogs is None:
            catalogs = self.find_catalogs(position)

        if (catalogs is None) or (len(catalogs) == 0):
            return []

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()

        l = list()

        for c in catalogs:
            df = c.get_catalog_objects(ra, dec, radius)
            if (df is not None) and (len(df) > 0):
                l.append(df)

        return l