#import global_config as G




#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:



try:
    from elixer import global_config as G
    from elixer import cat_candles_egs_stefanon_2016
    from elixer import cat_goods_n
    # from elixer import cat_goods_n_finkelstein
    from elixer import cat_egs_groth
    from elixer import cat_stack_cosmos
    from elixer import cat_shela
    from elixer import cat_hsc
    from elixer import cat_catch_all
    # from elixer import cat_ast376_shela
except:
    import global_config as G
    import cat_candles_egs_stefanon_2016
    import cat_goods_n
    #import cat_goods_n_finkelstein
    import cat_egs_groth
    import cat_stack_cosmos
    import cat_shela
    import cat_hsc
    import cat_catch_all
    # from elixer import cat_ast376_shela

# log = G.logging.getLogger('Cat_logger')
# log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

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



    def get_cutouts(self,position,radius,catalogs=None,aperture=None,dynamic=False):
        '''
        Return a list of dictionaries of the FITS cutouts from imaging catalogs
        (does not include objects in those catalogs, just the images)

        :param position: astropy SkyCoord
        :param radius: half-side of square cutout in arcsecs
        :param catalogs: optional list of catalogs to search (if not provided, searches all)
        :param aperture: optional aperture radius in arcsecs inside which to calcuate an AB magnitude
                          note: only returned IF the associated image has a magnitude function defined
                          note: will be forced to radius if radius is smaller than aperture
        :param dynamic: optional - if True, the aperture provided will grow in 0.1 arcsec steps until the magnitude
                        stabalizes (similar to, but not curve of growth)
        :return: list of dictionaries of cutouts and info
        '''

        if catalogs is None:
            catalogs = self.find_catalogs(position)

        if (catalogs is None) or (len(catalogs) == 0):
            return []

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()

        if radius > 0.5: #assume then that radius is in arcsecs
            radius /= 3600. #downstream calls expect this in degrees

        #sanity check, aperture cannot be larger than the cutout
        if aperture is not None:
            aperture = min(aperture, radius*3600.)

        l = list()

        #override the default ELiXer behavior
        G.DYNAMIC_MAG_APERTURE = dynamic
        G.FIXED_MAG_APERTURE = aperture

        for c in catalogs:
            cutouts = c.get_cutouts(ra, dec, window=radius*2.,aperture=aperture)
            # since this is a half-length of a side, the window (or side) is 2x radius
            if (cutouts is not None) and (len(cutouts) > 0):
                l.extend(cutouts)

        return l


    def get_catalog_objects(self,position,radius,catalogs=None):
        '''
         Return a list of dictionaries of pandas dataframes of objects in imaging catalogs

        :param position: astropy SkyCoord
        :param radius: distance in arcsecs from the provided position in which to search for catalog objects
        :param catalogs: optional list of catalogs to search (if not provided, searches all)
        :return: list of dictionaries of cutouts and info
        '''

        if catalogs is None:
            catalogs = self.find_catalogs(position)

        if (catalogs is None) or (len(catalogs) == 0):
            return []

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()

        if radius > 0.5: #assume then that radius is in arcsecs
            radius /= 3600. #downstream calls expect this in degrees

        l = list()

        for c in catalogs:
            df = c.get_catalog_objects(ra, dec, radius)
            if (df is not None) and (len(df) > 0):
                l.append(df)

        return l