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
    from elixer import cat_kpno
    from elixer import cat_sdss
    from elixer import cat_panstarrs
    from elixer import cat_catch_all
    from elixer import cat_decals_web
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
    import cat_kpno
    import cat_catch_all
    import cat_sdss
    import cat_panstarrs
    import cat_decals_web
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
        self.cats.append(cat_kpno.KPNO())
        # self.cats.append(cat_decals_web.DECaLS())
        # self.cats.append(cat_ast376_shela.AST376_SHELA())


    def get_full_catalog_list(self):
        if self.cats is None:
            self.build_catalog_list()

        return self.cats

    def get_catch_all(self):
        return cat_catch_all.CATCH_ALL()

    def get_sdss(self):
        return cat_sdss.SDSS()

    def get_panstarrs(self):
        return cat_panstarrs.PANSTARRS()

    def get_decals_web(self):
        return cat_decals_web.DECaLS()


    def find_catalogs(self,position,verify=False):
        '''
        Build a list of catalogs whose footprint contains the position

        :param position: astropy SkyCoord
        :param radius: in deccimal degrees at the moment
        :param verify: if False, checks only that the position is in the broad catalog footprint
                       if True, verifies access to imaging files and checks for available cutouts at the
                       position
                       (True option can be slow depending on file size, network speed, etc)
        :return: list of catalog objects
        '''

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()
        matched_cats = list()

        for c in self.cats:
            if c.position_in_cat(ra=ra, dec=dec, error=0):
                if verify: #check that there actually exists an accessible image
                    if self.verify_cutouts(position,c):
                        matched_cats.append(c)
                else: #go ahead and append
                    matched_cats.append(c)

        return matched_cats

    def verify_cutouts(self,position,catalog):
        """
        Verify that a cutout can be made in a catalog at the given position

        :param position:  astropy SkyCoord
        :param catalog: ELiXer catalog object
        :return: True or False
        """

        if catalog is None:
            return False

        result = False
        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()

        radius = 0.0003 #downstream calls expect this in degrees #approx 1"
        aperture = 1.0 #arcsec

        r_dict = catalog.get_cutouts(ra,dec,window=radius*2.,aperture=aperture)

        if (r_dict is not None) and (len(r_dict) > 0):
            #if any cutouts are returned
            for d in r_dict:
                if d['cutout'] is not None and (sum(d['cutout'].shape) > 10):
                    result = True
                    break

        return result

    def get_filters(self,position=None,catalogs=None):
        '''
        Get dictionary of available filters per catalog.

        :param position: used to select catalog(s) IF catalogs is not specified
        :param catalogs: look up only those catalogs in this list
        :return: dictionary of catalog name [key] and list of available catalog filters [value]
                 (at this time there is no check to see if the position is covered by each filter,
                 only that the catalog has the filter)
        '''

        filters = {}  # name is the key, filter_list is the value

        try:
            if catalogs is None:
                catalogs = self.find_catalogs(position)

            if (catalogs is None) or (len(catalogs) == 0):
                return []

            # if position:
            #     ra = position.ra.to_value() #decimal degrees
            #     dec = position.dec.to_value()
            # else:
            #     ra = None
            #     dec = None

            for c in catalogs:
                filters[c.name] = c.get_filters(ra=None,dec=None)

        except:
            log.error("Exception! Unable to get filters.",exc_info=True)

        return filters

    def get_cutouts(self,position,radius=None,side=None,catalogs=None,aperture=None,dynamic=False,
                    nudge=None,filter=None,first=False):
        '''
        Return a list of dictionaries of the FITS cutouts from imaging catalogs
        (does not include objects in those catalogs, just the images).

        position and one of radius or side MUST be provided

        0.5 arcsecs to 0.5 degrees

        :param position: astropy SkyCoord
        :param radius: half-side of square cutout x1.5 (returned (square) cutout size is radius x3)
                       units are assumed to be arcsecs if the value is greater than 0.5 and in decimal
                       degrees otherwise
        :param side: may be used instead of (takes priority over) radius and is the width of the side of
                     square cutout requested. Units are assumed to be arcsecs if the value is greater than
                     0.5 and in decimal degrees otherwise
        :param catalogs: optional list of catalogs to search (if not provided, searches all)
        :param aperture: optional aperture radius in arcsecs inside which to calcuate an AB magnitude
                          note: only returned IF the associated image has a magnitude function defined
                          (None, otherwise)
                          note: will be forced to radius if radius is smaller than aperture
                          note: a value of 99.9 means the magnitude could not be calculated (usually an
                            error with the pixel counts or photutils)
        :param dynamic: optional - if True, the aperture provided will grow in 0.1 arcsec steps until the
                        magnitude stabalizes (similar to, but not curve of growth)
        :param nudge: optional - if not None, specifies the amount of drift (in x and y in arcsecs) allowed
                      for the center of the aperture to align it with the local 2D Gaussian centroid of the
                      pixel counts. If None or 0.0, the center is not allowed to move and stays on the
                      supplied RA and Dec (position).
        :param filter: optional - if not None is a LIST of filter name(s) (as strings), specifying which
                       cutouts to get (can be used with the catalogs parameter). The '*' is the wildcard
                       and will match ANY catalog filter.
        :param first: optional - if True and filter is specified, return only the first cutout found. If
                      filter is specified, return the first cutout found that matches a filter in the filter
                      list parameter in the order specified in the list
        :return: list of dictionaries of cutouts and info,
                one for each matching catalog FITS image that contains the requested coordinate.
                The dictionary contains the following keys:

                'cutout' = an astropy Cutout2D object (or None if there was an error)
                'path' = the full path to the FITS image from which the cutout was made
                'hdu' = the HDU list from the FITS image
                'instrument' = the instrument name (like DECAM, or HSC, or HST-WFC-3)
                'filter' = the filter name
                'instrument' = the instrument name
                'mag' = the calculated magnitude within the aperture radius if a conversion is available
                        and aperture was specified
                'aperture' = the aperture radius for the magnitude
                'ap_center' = the displacment of the center of the aperture from the center of the image
                              (if 'nudge' was specified)
                'details' = dictionary of detailed information about the aperture and source extractor
                            (if available) photometry

        '''

        if catalogs is None:
            catalogs = self.find_catalogs(position)

        if (catalogs is None) or (len(catalogs) == 0):
            log.error("No catalogs available.")
            return []

        if not (side or radius):
            log.error("Insufficient information to process. Neither side nor radius specified.")
            return []

        if side:
            radius = side/3.0

        ra = position.ra.to_value() #decimal degrees
        dec = position.dec.to_value()

        if radius > 0.5: #assume then that radius is in arcsecs
            radius /= 3600. #downstream calls expect this in degrees

        #sanity check, aperture cannot be larger than the cutout
        if aperture is not None:
            aperture = min(aperture, radius*3600.)

        l = list()

        #override the default ELiXer behavior

        saved_DYNAMIC_MAG_APERTURE = G.DYNAMIC_MAG_APERTURE
        saved_FIXED_MAG_APERTURE = G.FIXED_MAG_APERTURE
        saved_NUDGE_MAG_APERTURE_CENTER = G.NUDGE_MAG_APERTURE_CENTER

        G.DYNAMIC_MAG_APERTURE = dynamic
        G.FIXED_MAG_APERTURE = aperture
        if (nudge is None) or (nudge < 0):
            nudge = 0.0
        G.NUDGE_MAG_APERTURE_CENTER = nudge

        if filter: #filternames are are lowercase, so force input to lowercase for matching
            try:
                filter = [x.lower() for x in filter]
            except:
                pass

        for c in catalogs:
            cutouts = c.get_cutouts(ra, dec, window=radius*2.,aperture=aperture,filter=filter,first=first)
            # since this is a half-length of a side, the window (or side) is 2x radius
            if (cutouts is not None) and (len(cutouts) > 0):
                l.extend(cutouts)

        #restore
        G.DYNAMIC_MAG_APERTURE = saved_DYNAMIC_MAG_APERTURE
        G.FIXED_MAG_APERTURE = saved_FIXED_MAG_APERTURE
        G.NUDGE_MAG_APERTURE_CENTER = saved_NUDGE_MAG_APERTURE_CENTER

        return l


    def get_catalog_objects(self,position,radius,catalogs=None):
        '''
         Return a list of dictionaries of pandas dataframes of objects in imaging catalogs

        :param position: astropy SkyCoord
        :param radius: distance in arcsecs from the provided position in which to search for catalog objects
        :param catalogs: optional list of catalogs to search (if not provided, searches all)
        :return: list of dictionaries (one for each catalog containing the target position; usually just one)
                 Each dictionary contains the following keys:

                'count' = the number of catalog objects within the search radius
                'name' = the name of the catalog
                'dataframe' = pandas dataframe containing the catalog detections. Available columns depend
                              on the catalog but will always contain at least: 'RA','DEC', and 'distance'
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