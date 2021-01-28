from __future__ import print_function

"""
originally a duplicate of cat_candles_egs_stefanon_2016 which includes a section of CFHTLS

This now has the D3 (deep) single tile (center roughly RA 210, Dec 52) and the W3 (wide) tiles in the same region. 
This covers a section of the HETDEX spring field. 
"""

try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import line_prob
    from elixer import cat_base
    from elixer import match_summary
    from elixer import utilities
    from elixer import sqlite_utils as sql
    from elixer import spectrum_utilities as SU
    from elixer import cfhtls_meta
except:
    import global_config as G
    import science_image
    import line_prob
    import cat_base
    import match_summary
    import utilities
    import sqlite_utils as sql
    import spectrum_utilities as SU
    import cfhtls_meta

import os.path as op
import copy
import io



CFHTLS_BASE_PATH = G.CFHTLS_BASE_PATH
CFHTLS_CAT_PATH = G.CFHTLS_BASE_PATH
CFHTLS_IMAGE_PATH = G.CFHTLS_BASE_PATH

import matplotlib
#matplotlib.use('agg')

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
#import matplotlib.patches as mpatches
#import mpl_toolkits.axisartist.floating_axes as floating_axes
#from matplotlib.transforms import Affine2D




#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field


def cfhtls_count_to_mag(count,cutout=None,sci_image=None):
    if count is not None:
        zero_point = 30.0

        if sci_image:
            try:
                #try to read the zero point MZP_AB, but they are all 30.0
                if 'MZP_AB' in sci_image[0]:
                    zero_point = float(headers[0]['MZP_AB'])
            except:
                pass

        if count > 0:
            return -2.5 * np.log10(count) + zero_point #PHOTZP in FITS header or MZP_AB
        else:
            return 99.9  # need a better floor

class CFHTLS(cat_base.Catalog):
    # RA,Dec in decimal degrees

    # class variables
    MainCatalog = None
    Name = "MegaPrime/CFHTLS"

    #Image_Coord_Range = {'RA_min':208.529266, 'RA_max':220.420734, 'Dec_min':51.205958, 'Dec_max':57.805087}
    Image_Coord_Range = cfhtls_meta.Image_Coord_Range
    Cat_Coord_Range = Image_Coord_Range
    Tile_Dict = cfhtls_meta.CFHTLS_META_DICT
    Filters = [] #are populated when the CatalogImages are built
    CatalogImages = [] #built in constructor

    #fix the Tile_Dict
    #correct the paths
    for k in Tile_Dict.keys():
        Tile_Dict[k]['path'] = G.CFHTLS_BASE_PATH #op.join(G.CFHTLS_BASE_PATH,op.basename(Tile_Dict[k]['path']))
        Tile_Dict[k]['filter'] = Tile_Dict[k]['filter'][0] #all of form r.MP9601

    mean_FWHM = 0.6 #just a rough value (seems to be 0.6 - 1.0)
    #checking just the W_r with 1" and 2" 2000x at 5sigma, consistently about 26.2 (25.3)
    MAG_LIMIT = 26.0 #for the _W_?_ images; rough estimate, needs to be done per tile (about the same as HSC)
    MAG_LIMIT_DEEP = 27.3 # for the D3.xx.xx images

    WCS_Manual = False
    CONT_EST_BASE = None

    CFHTLS_PhotoZCatalog = G.CFHTLS_PHOTOZ_CAT #limited photz catalog
    df = None  # cat_base has df
    loaded_tiles = []
    photz_df = None

    #for the CFHTLS_W_* catalogs
    BidCols = [
        'NUMBER',
        'X_IMAGE',
        'Y_IMAGE',
        'ERRA_IMAGE',
        'ERRB_IMAGE',
        'ERRTHETA_IMAGE',
        'A_IMAGE',
        'B_IMAGE',
        'POLAR_IMAGE',
        'THETA_IMAGE',
        'X_WORLD',
        'Y_WORLD',
        'ERRA_WORLD',
        'ERRB_WORLD',
        'ERRTHETA_WORLD',
        'A_WORLD',
        'B_WORLD',
        'POLAR_WORLD',
        'THETA_WORLD',
        'ALPHA_J2000',
        'DELTA_J2000',
        'ERRTHETA_J2000',
        'THETA_J2000',
        'XWIN_IMAGE',
        'YWIN_IMAGE',
        'ERRAWIN_IMAGE',
        'ERRBWIN_IMAGE',
        'ERRTHETAWIN_IMAGE',
        'AWIN_IMAGE',
        'BWIN_IMAGE',
        'POLARWIN_IMAGE',
        'THETAWIN_IMAGE',
        'XWIN_WORLD',
        'YWIN_WORLD',
        'ERRAWIN_WORLD',
        'ERRBWIN_WORLD',
        'ERRTHETAWIN_WORLD',
        'AWIN_WORLD',
        'BWIN_WORLD',
        'POLARWIN_WORLD',
        'THETAWIN_WORLD',
        'ALPHAWIN_J2000',
        'DELTAWIN_J2000',
        'ERRTHETAWIN_J2000',
        'THETAWIN_J2000',
        'FLUX_ISO',
        'FLUXERR_ISO',
        'MAG_ISO',
        'MAGERR_ISO',
        'FLUX_APER',
        'FLUXERR_APER',
        'MAG_APER',
        'MAGERR_APER',
        'FLUX_AUTO',
        'FLUXERR_AUTO',
        'MAG_AUTO',
        'MAGERR_AUTO',
        'FLUX_PETRO',
        'FLUXERR_PETRO',
        'MAG_PETRO',
        'MAGERR_PETRO',
        'FLUX_RADIUS',
        'KRON_RADIUS',
        'PETRO_RADIUS',
        'BACKGROUND',
        'THRESHOLD',
        'MU_THRESHOLD',
        'FLUX_MAX',
        'MU_MAX',
        'ISOAREA_IMAGE',
        'ISOAREAF_IMAGE',
        'ISOAREA_WORLD',
        'ISOAREAF_WORLD',
        'FLAGS',
        'CLASS_STAR'
    ]
    #   1 NUMBER                 Running object number
    #   2 X_IMAGE                Object position along x                                    [pixel]
    #   3 Y_IMAGE                Object position along y                                    [pixel]
    #   4 ERRA_IMAGE             RMS position error along major axis                        [pixel]
    #   5 ERRB_IMAGE             RMS position error along minor axis                        [pixel]
    #   6 ERRTHETA_IMAGE         Error ellipse position angle (CCW/x)                       [deg]
    #   7 A_IMAGE                Profile RMS along major axis                               [pixel]
    #   8 B_IMAGE                Profile RMS along minor axis                               [pixel]
    #   9 POLAR_IMAGE            (A_IMAGE^2 - B_IMAGE^2)/(A_IMAGE^2 + B_IMAGE^2)
    #  10 THETA_IMAGE            Position angle (CCW/x)                                     [deg]
    #  11 X_WORLD                Barycenter position along world x axis                     [deg]
    #  12 Y_WORLD                Barycenter position along world y axis                     [deg]
    #  13 ERRA_WORLD             World RMS position error along major axis                  [deg]
    #  14 ERRB_WORLD             World RMS position error along minor axis                  [deg]
    #  15 ERRTHETA_WORLD         Error ellipse pos. angle (CCW/world-x)                     [deg]
    #  16 A_WORLD                Profile RMS along major axis (world units)                 [deg]
    #  17 B_WORLD                Profile RMS along minor axis (world units)                 [deg]
    #  18 POLAR_WORLD            (A_WORLD^2 - B_WORLD^2)/(A_WORLD^2 + B_WORLD^2)
    #  19 THETA_WORLD            Position angle (CCW/world-x)                               [deg]
    #  20 ALPHA_J2000            Right ascension of barycenter (J2000)                      [deg]
    #  21 DELTA_J2000            Declination of barycenter (J2000)                          [deg]
    #  22 ERRTHETA_J2000         J2000 error ellipse pos. angle (east of north)             [deg]
    #  23 THETA_J2000            Position angle (east of north) (J2000)                     [deg]
    #  24 XWIN_IMAGE             Windowed position estimate along x                         [pixel]
    #  25 YWIN_IMAGE             Windowed position estimate along y                         [pixel]
    #  26 ERRAWIN_IMAGE          RMS windowed pos error along major axis                    [pixel]
    #  27 ERRBWIN_IMAGE          RMS windowed pos error along minor axis                    [pixel]
    #  28 ERRTHETAWIN_IMAGE      Windowed error ellipse pos angle (CCW/x)                   [deg]
    #  29 AWIN_IMAGE             Windowed profile RMS along major axis                      [pixel]
    #  30 BWIN_IMAGE             Windowed profile RMS along minor axis                      [pixel]
    #  31 POLARWIN_IMAGE         (AWIN^2 - BWIN^2)/(AWIN^2 + BWIN^2)
    #  32 THETAWIN_IMAGE         Windowed position angle (CCW/x)                            [deg]
    #  33 XWIN_WORLD             Windowed position along world x axis                       [deg]
    #  34 YWIN_WORLD             Windowed position along world y axis                       [deg]
    #  35 ERRAWIN_WORLD          World RMS windowed pos error along major axis              [deg]
    #  36 ERRBWIN_WORLD          World RMS windowed pos error along minor axis              [deg]
    #  37 ERRTHETAWIN_WORLD      Windowed error ellipse pos. angle (CCW/world-x)            [deg]
    #  38 AWIN_WORLD             Windowed profile RMS along major axis (world units)        [deg]
    #  39 BWIN_WORLD             Windowed profile RMS along minor axis (world units)        [deg]
    #  40 POLARWIN_WORLD         (AWIN^2 - BWIN^2)/(AWIN^2 + BWIN^2)
    #  41 THETAWIN_WORLD         Windowed position angle (CCW/world-x)                      [deg]
    #  42 ALPHAWIN_J2000         Windowed right ascension (J2000)                           [deg]
    #  43 DELTAWIN_J2000         windowed declination (J2000)                               [deg]
    #  44 ERRTHETAWIN_J2000      J2000 windowed error ellipse pos. angle (east of north)    [deg]
    #  45 THETAWIN_J2000         Windowed position angle (east of north) (J2000)            [deg]
    #  46 FLUX_ISO               Isophotal flux                                             [count]
    #  47 FLUXERR_ISO            RMS error for isophotal flux                               [count]
    #  48 MAG_ISO                Isophotal magnitude                                        [mag]
    #  49 MAGERR_ISO             RMS error for isophotal magnitude                          [mag]
    #  50 FLUX_APER              Flux vector within fixed circular aperture(s)              [count]
    #  77 FLUXERR_APER           RMS error vector for aperture flux(es)                     [count]
    # 104 MAG_APER               Fixed aperture magnitude vector                            [mag]
    # 131 MAGERR_APER            RMS error vector for fixed aperture mag.                   [mag]
    # 158 FLUX_AUTO              Flux within a Kron-like elliptical aperture                [count]
    # 159 FLUXERR_AUTO           RMS error for AUTO flux                                    [count]
    # 160 MAG_AUTO               Kron-like elliptical aperture magnitude                    [mag]
    # 161 MAGERR_AUTO            RMS error for AUTO magnitude                               [mag]
    # 162 FLUX_PETRO             Flux within a Petrosian-like elliptical aperture           [count]
    # 163 FLUXERR_PETRO          RMS error for PETROsian flux                               [count]
    # 164 MAG_PETRO              Petrosian-like elliptical aperture magnitude               [mag]
    # 165 MAGERR_PETRO           RMS error for PETROsian magnitude                          [mag]
    # 166 FLUX_RADIUS            Fraction-of-light radii                                    [pixel]
    # 169 KRON_RADIUS            Kron apertures in units of A or B
    # 170 PETRO_RADIUS           Petrosian apertures in units of A or B
    # 171 BACKGROUND             Background at centroid position                            [count]
    # 172 THRESHOLD              Detection threshold above background                       [count]
    # 173 MU_THRESHOLD           Detection threshold above background                       [mag * arcsec**(-2)]
    # 174 FLUX_MAX               Peak flux above background                                 [count]
    # 175 MU_MAX                 Peak surface brightness above background                   [mag * arcsec**(-2)]
    # 176 ISOAREA_IMAGE          Isophotal area above Analysis threshold                    [pixel**2]
    # 177 ISOAREAF_IMAGE         Isophotal area (filtered) above Detection threshold        [pixel**2]
    # 178 ISOAREA_WORLD          Isophotal area above Analysis threshold                    [deg**2]
    # 179 ISOAREAF_WORLD         Isophotal area (filtered) above Detection threshold        [deg**2]
    # 180 FLAGS                  Extraction flags
    # 181 CLASS_STAR             S/G classifier output

    #for the D3.x catalogs
    BidCols_Deep = [
        'NUMBER',
        'X_IMAGE',
        'Y_IMAGE',
        'ALPHA_J2000',
        'DELTA_J2000',
        'MAG_AUTO',
        'MAGERR_AUTO',
        'MAG_BEST',
        'MAGERR_BEST',
        'MAG_APER',
        'A_WORLD',
        'ERRA_WORLD',
        'B_WORLD',
        'ERRB_WORLD',
        'THETA_J2000',
        'ERRTHETA_J2000',
        'ISOAREA_IMAGE',
        'MU_MAX',
        'FLUX_RADIUS',
        'FLAGS',
    ]

    #deep imaging
    #   1 NUMBER          Running object number
    #   2 X_IMAGE         Object position along x                         [pixel]
    #   3 Y_IMAGE         Object position along y                         [pixel]
    #   4 ALPHA_J2000     Right ascension of barycenter (J2000)           [deg]
    #   5 DELTA_J2000     Declination of barycenter (J2000)               [deg]
    #   6 MAG_AUTO        Kron-like elliptical aperture magnitude         [mag]
    #   7 MAGERR_AUTO     RMS error for AUTO magnitude                    [mag]
    #   8 MAG_BEST        Best of MAG_AUTO and MAG_ISOCOR                 [mag]
    #   9 MAGERR_BEST     RMS error for MAG_BEST                          [mag]
    #  10 MAG_APER        Fixed aperture magnitude vector                 [mag]
    #  11 MAGERR_APER     RMS error vector for fixed aperture mag.        [mag]
    #  12 A_WORLD         Profile RMS along major axis (world units)      [deg]
    #  13 ERRA_WORLD      World RMS position error along major axis       [pixel]
    #  14 B_WORLD         Profile RMS along minor axis (world units)      [deg]
    #  15 ERRB_WORLD      World RMS position error along minor axis       [pixel]
    #  16 THETA_J2000     Position angle (east of north) (J2000)          [deg]
    #  17 ERRTHETA_J2000  J2000 error ellipse pos. angle (east of north)  [deg]
    #  18 ISOAREA_IMAGE   Isophotal area above Analysis threshold         [pixel**2]
    #  19 MU_MAX          Peak surface brightness above background        [mag * arcsec**(-2)]
    #  20 FLUX_RADIUS     Fraction-of-light radii                         [pixel]
    #  21 FLAGS           Extraction flags



    def __init__(self):
        super(CFHTLS, self).__init__()

        # self.dataframe_of_bid_targets = None #defined in base class
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

        # do this only as needed
        # self.read_main_catalog()
        # self.read_photoz_catalog()
        # self.build_catalog_images() #will just build on demand

        self.master_cutout = None
        self.build_catalog_of_images()

    @classmethod
    def read_main_catalog(cls):
        #super(CFHTLS, cls).read_main_catalog()
        #there is no "main" catalog for CFHTLS, but there is a large photz catalog which we read just below
        #there are individual catalogs for each tile that are read later, as needed
        return
        #the photz catalog will be read in as needed (not at init time)
        # try:
        #     print("Reading CFHTLS photz catalog for ", cls.Name)
        #     cls.photz_df = cls.read_photz_catalog(cls.CFHTLS_PhotoZCatalog, cls.Name)
        #
        # except:
        #     print("Failed")
        #     cls.status = -1
        #     log.error("Exception in read_main_catalog for %s" % cls.Name, exc_info=True)

    @classmethod
    def read_tile_catalog(cls, tile):
        """

        Read as r band first then g band if r not avaialble

        :param catalog_loc:
        :param name:
        :param tract: list of string string as the HSC track id, ie. ['16814']
        :param position: a tuple or array with exactly 2 elements as integers 0-9, i.e. (0,1) or (2,8), etc
        :return:
        """

        name = cls.Name
        #fully qualified track (as a partial path)
        filter = 'r'
        fqtile = op.join(CFHTLS_BASE_PATH,tile.replace('?',filter,1).rstrip("fits") + "cat")
        if not op.exists(fqtile):
            #try g band
            filter = 'g'
            fqtile = op.join(CFHTLS_BASE_PATH,tile.replace('?',filter,1).rstrip("fits") + "cat")
            if not op.exists(fqtile):
                #there is something wrong, don't bother checking the other filters
                log.warning(f"Unable to find catalogs for: {tile} under {CFHTLS_BASE_PATH}")
                return None

        if set(fqtile).issubset(cls.loaded_tiles):
            log.info("Catalog tract (%s) already loaded." %fqtract)
            return cls.df

        #todo: future more than just the R filter if any are ever added
        if tile[0:2]=="D3":
            header=cls.BidCols_Deep
        else:
            header = cls.BidCols

        log.debug("Building " + cls.Name + " " + tile + " dataframe...")

        try:
            df = pd.read_csv(fqtile, names=header,comment='#',
                             delim_whitespace=True, header=None, index_col=False, skiprows=0)

            old_names = ['ALPHA_J2000','DELTA_J2000']
            new_names = ['RA','DEC']
            df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

            df['FILTER'] = filter #add the FILTER to the dataframe !!! case is important. must be lowercase

            if cls.df is not None:
                cls.df = pd.concat([cls.df, df])
            else:
                cls.df = df
            cls.loaded_tiles.append(tile)

        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return cls.df

    @classmethod
    def read_photoz_catalog(cls):
        if cls.df_photoz is not None:
            log.debug("Already built df_photoz")
        else:
            try:
                print("Reading photoz catalog for ", cls.Name)
                cls.df_photoz = cls.read_chftls_photz_catalog(cls.CFHTLS_PhotoZCatalog, cls.Name)
            except:
                print("Failed")

        return

    @classmethod
    def read_chftls_photz_catalog(cls, catalog_loc, name):

        log.debug("Building " + name + " dataframe...")

        #names slightly changed (0,1,2,6) to match with the EGS header names
        #each line is 10 index wide
        header = ['ID','RA','DEC','flag','StarGal','r2','z_best','zPDF','zPDF_l68','zPDF_u68',
                  'chi2_zPDF','mod','ebv','NbFilt','zMin','zl68','zu68','chi2_best','zp_2','chi2_2',
                  'mods','chis','zq','chiq','modq','U','G','R','I','Y',
                  'Z','eU','eG','eR','eI','eY','eZ','MU','MG','MR',
                  'MI','MY','MZ']

        dtypes = {'ID':str,'RA':np.float64,'DEC':np.float64,'flag':int,'StarGal':int,'r2':float,'z_best':float,
                  'zPDF':float,'zPDF_l68':float,'zPDF_u68':float,
                  'chi2_zPDF':float,'mod':int,'ebv':float,'NbFilt':int,'zMin':float,'zl68':float,'zu68':float,
                  'chi2_best':float,'zp_2':float,'chi2_2':float,
                  'mods':int,'chis':float,'zq':float,'chiq':float,'modq':int,
                  'U':float,'G':float,'R':float,'I':float,'Y':float,'Z':float,
                  'eU':float,'eG':float,'eR':float,'eI':float,'eY':float,'eZ':float,
                  'MU':float,'MG':float,'MR':float,'MI':float,'MY':float,'MZ':float}

        #this has no header comments

        #note z_best = -99.9 is the non-value?

        try: #for now, just use the few columns needed (ID,RA,DEC,z_best,G,eG)
            df = pd.read_csv(catalog_loc, names=header,dtype=dtypes,usecols = [0,1,2,6,26,32],
                             na_values= ['*********'],
                             delim_whitespace=True, header=None, index_col=False, skiprows=0)

            #for cloned compatibility with CANDELS phot-z catalog
            df['file'] = None
            df['z_best_type'] = 'p'
            df['mFDa4_z_weight'] = None
        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return df

    def build_catalog_of_images(self):

        for t in self.Tile_Dict.keys(): #tile is the key (the filename)
            try:
                path = self.Tile_Dict[t]['path']
                name = t
                wcs_manual = False
                filter = self.Tile_Dict[t]['filter']

                if filter in self.Filters:
                    pass
                else:
                    self.Filters.append(filter)

                self.CatalogImages.append(
                    {'path': path,
                     'name': name, #filename is the tilename
                     'tile': t,
                     'pos': None, #not using positoin (but kept for compatibility)
                     'filter': filter,
                     'instrument': "CFHTLS/MegaPrime",
                     'cols': [],
                     'labels': [],
                     'image': None,
                     'expanded': False,
                     'wcs_manual': wcs_manual,
                     'aperture': self.mean_FWHM * 0.5 + 0.5, #since a radius, half the FWHM + 0.5" for astrometric error
                     'mag_func': cfhtls_count_to_mag
                     })

            except:
                log.error("Exception building CatalogImages in cat_cfhtls.",exc_info=True)

    def get_mag_limit(self,image_identification=None,aperture_diameter=None):
        """
            to be overwritten by subclasses to return their particular format of maglimit

            :param image_identification: some way (sub-class specific) to identify which image
                    HERE we want a tuple ... [0] = tile name and [1] = filter name
            :param aperture_diameter: in arcsec
            :return:
        """

        try:
            if image_identification and image_identification[0:2] == "D3":
                return self.MAG_LIMIT_DEEP
            else:
                return self.MAG_LIMIT
        except:
            log.warning("cat_cfhtls.py get_mag_limit fail.",exc_info=True)
            try:
                return self.MAG_LIMIT
            except:
                return 99.9

    def get_filter_flux(self,df):
        """

        :param df:
        :return: flux in uJy and mags
        """

        filter_fl = None
        filter_fl_err = None
        mag = None
        mag_bright = None
        mag_faint = None
        filter_str = None
        try:

            #there seem to be issues in the catalogs with the _AUTO versions of flux and mag (where MAG_AUTO is still
            # in counts, not mags and the errors on the flux are huge, so use the _ISO as it seems more stable and
            # in the expected units
            # return -2.5 * np.log10(count) + 30.0 #PHOTZP in FITS header or MZP_AB

            #filter_fl = df['FLUX_ISO'].values[0]  #(counts)
            #filter_fl_err = df['FLUXERR_ISO'].values[0]
            filter_str = df['FILTER'].values[0]

            #turn the counts into uJy? don't have a conversion from counts, but that is wrapped in the mags, so
            #just use them
            try:
                mag = df['MAG_ISO'].values[0]
                mag_bright = mag - df['MAGERR_ISO'].values[0]
                mag_faint = mag + df['MAGERR_ISO'].values[0]

                filter_fl = 10**(-0.4*mag) * 3631.0e6 #uJy
                filter_fl_err = df['FLUXERR_ISO'].values[0]/df['FLUX_ISO'].values[0] * filter_fl
            except:
                try:
                    mag = df['MAG_BEST'].values[0]
                    mag_bright = mag - df['MAGERR_BEST'].values[0]
                    mag_faint = mag + df['MAGERR_BEST'].values[0]

                    filter_fl = 10**(-0.4*mag) * 3631.0e6 #uJy
                    filter_fl_err = 0.5 * (10**(-0.4*mag_bright) - 10**(-0.4*mag_faint)) * 3631.0e6
                except:
                    pass
        except:
            pass

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str


    def find_target_tile(self,ra,dec):
        #assumed to have already confirmed this target is at least in coordinate range of this catalog
        #return at most one tile, but maybe more than one tract (for the catalog ... HSC does not completely
        #   overlap the tracts so if multiple tiles are valid, depending on which one is selected, you may
        #   not find matching objects for the associated tract)
        tile = None
        keys = []
        for k in self.Tile_Dict.keys():
            # don't bother to load if ra, dec not in range
            try:
                if not ((ra >= self.Tile_Dict[k]['RA_min']) and (ra <= self.Tile_Dict[k]['RA_max']) and
                        (dec >= self.Tile_Dict[k]['Dec_min']) and (dec <= self.Tile_Dict[k]['Dec_max'])) :
                    continue
                else:
                    #adjust out the filter
                    # removed# two unique names: D3.I.1_20558_1_21553.fits and CFHTLS_D-25_g_141927+524056_T0007_SIGWEI.fits
                    #D3.?.fits
                    #otherwise all are like: 'CFHTLS_W_u_141123+562231_T0007_MEDIAN.fits'
                    if k[0:2] == "D3":
                        cur_tile = "D3.?.fits"
                    else:
                        toks = k.split("_")
                        if toks[1] == "D-25":
                            cur_tile = "CFHTLS_D-25_g_141927+524056_T0007_SIGWEI.fits"
                        else:
                            toks[2] = '?' #replace the filter character with 'x'
                            cur_tile = "_".join(toks)

                    if cur_tile in keys:
                        pass
                    else:
                        keys.append(cur_tile) #multiple tiles should overlap because of filters
            except:
                pass

        if len(keys) == 0: #we're done ... did not find any
            return None
        elif len(keys) == 1: #found exactly one
            tile = keys[0] #remember tile is a string ... there can be only one
        elif len(keys) > 1: #find the best one
            log.info("Multiple overlapping tiles %s. Sub-selecting tile with maximum angular coverage around target." %keys)
            min = 9e9
            #we don't have the actual corners anymore, so just assume a rectangle
            #so there are 2 of each min, max coords. Only need the smallest distance so just sum one
            #these still have the '?' for the filter

            #use the deep imaging if that is available, otherwise, choose the more central tile
            if "D3.?.fits" in keys:
                tile = "D3.?.fits"
            else:
                for k in keys:
                    adjusted_key = k.replace('?','r')
                    sqdist = (ra-self.Tile_Dict[adjusted_key]['RA_min'])**2 + (dec-self.Tile_Dict[adjusted_key]['Dec_min'])**2 + \
                             (ra-self.Tile_Dict[adjusted_key]['RA_max'])**2 + (dec-self.Tile_Dict[adjusted_key]['Dec_max'])**2
                    if sqdist < min:
                        min = sqdist
                        tile = k

        else: #really?? len(keys) < 0 : this is just a sanity catch
            log.error("ERROR! len(keys) < 0 in cat_cfhtls::find_target_tile.")
            return None

        log.info("Selected tile: %s" % tile)
        #now we have the tile key (filename)
        #do we want to find the matching catalog and see if there is an entry in it?

        #sanity check the image
        # try:
        #     image = science_image.science_image(wcs_manual=self.WCS_Manual,wcs_idx=0,
        #                                         image_location=op.join(self.HSC_IMAGE_PATH,tile))
        #     if image.contains_position(ra, dec):
        #         pass
        #     else:
        #         log.debug("position (%f, %f) is not in image. %s" % (ra, dec,tile))
        #         tile = None
        # except:
        #     pass

        return tile

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        #even if not None, could be we need a different catalog, so check and append
        tile  = self.find_target_tile(ra,dec)

        if tile is None:
            log.info("Could not locate tile for CFHTLS. Discontinuing search of this catalog.")
            return -1,None,None

        #could be none or could be not loaded yet
        if self.df is None or not (set([tile]).issubset(self.loaded_tiles)):
            self.read_tile_catalog(tile)


        #don't need phot_z yet
        # if self.df_photoz is None:
        #     self.read_photoz_catalog()

        error_in_deg = np.float64(error) / 3600.0

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0

        coord_scale = np.cos(np.deg2rad(dec))

        #can't actually happen for this catalog
        if coord_scale < 0.1: #about 85deg
            print("Warning! Excessive declination (%f) for this method of defining error window. Not supported" %(dec))
            log.error("Warning! Excessive declination (%f) for this method of defining error window. Not supported" %(dec))
            return 0,None,None

        ra_min = np.float64(ra - error_in_deg/coord_scale)
        ra_max = np.float64(ra + error_in_deg/coord_scale)
        dec_min = np.float64(dec - error_in_deg)
        dec_max = np.float64(dec + error_in_deg)

        log.info(self.Name + " searching for bid targets in range: RA [%f +/- %f], Dec [%f +/- %f] ..."
                 % (ra, error_in_deg, dec, error_in_deg))

        try:
            self.dataframe_of_bid_targets = \
                self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                        (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)].copy()

            if self.dataframe_of_bid_targets is not None:
                self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()

            #we will deal with possible phot-z later
            # if self.num_targets > 0:
            #     # ID matches between both catalogs
            #     self.dataframe_of_bid_targets_photoz = \
            #         self.df_photoz[(self.df_photoz['ID'].isin(self.dataframe_of_bid_targets['ID']))].copy()
        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)


        if self.num_targets > 0:
            self.sort_bid_targets_by_likelihood(ra, dec)

            log.info(self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
                     ". Found = %d" % (self.num_targets))

        return self.num_targets, self.dataframe_of_bid_targets, self.dataframe_of_bid_targets_photoz

    # column names are catalog specific, but could map catalog specific names to generic ones and produce a dictionary?
    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None,target_flux=None,detobj=None):

        self.clear_pages()
        num_targets, _, _ = self.build_list_of_bid_targets(target_ra, target_dec, error)

        if (self.dataframe_of_bid_targets is None):
            return None

        ras = self.dataframe_of_bid_targets.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            if G.BUILD_REPORT_BY_FILTER:
                if G.BUILD_REPORT_BY_FILTER:
                    #here we return a list of dictionaries (the "cutouts" from this catalog)
                    return self.build_cat_summary_details(cat_match,target_ra, target_dec, error, ras, decs,
                                                          target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                                      detobj=detobj)
                else:
                    log.error("ERROR! CFHTLS catalog does not support build_cat_summary_figure. "
                              "global_config BUILD_REPORT_BY_FILTER MUST be set to True.")
                    print("ERROR! CFHTLS catalog does not support build_cat_summary_figure. "
                          "global_config BUILD_REPORT_BY_FILTER MUST be set to True.")
                    return None
            else:
                log.error("ERROR! CFHTLS catalog does not support build_cat_summary_figure. "
                          "global_config BUILD_REPORT_BY_FILTER MUST be set to True.")
                print("ERROR! CFHTLS catalog does not support build_cat_summary_figure. "
                      "global_config BUILD_REPORT_BY_FILTER MUST be set to True.")
                return None
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")
            return None

        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS): # each bid taget gets its own line
            log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

        return None


    def get_stacked_cutout(self,ra,dec,window):

        stacked_cutout = None
        error = window

        for i in self.CatalogImages:  # i is a dictionary
            try:
                wcs_manual = i['wcs_manual']
            except:
                wcs_manual = self.WCS_Manual

            try:
                if i['image'] is None:
                    i['image'] = science_image.science_image(wcs_manual=wcs_manual,
                                                             image_location=op.join(i['path'], i['name']))
                sci = i['image']

                cutout, _, _, _ = sci.get_cutout(ra, dec, error, window=window, aperture=None, mag_func=None)
                #don't need pix_counts or mag, etc here, so don't pass aperture or mag_func

                if cutout is not None:  # construct master cutout
                    if stacked_cutout is None:
                        stacked_cutout = copy.deepcopy(cutout)
                        ref_exptime = sci.exptime
                        total_adjusted_exptime = 1.0
                    else:
                        stacked_cutout.data = np.add(stacked_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
            except:
                log.error("Error in get_stacked_cutout.",exc_info=True)

        return stacked_cutout



    def get_single_cutout(self, ra, dec, window, catalog_image,aperture=None,error=None,do_sky_subtract=True):
        """

        :param ra:
        :param dec:
        :param window:
        :param catalog_image:
        :param aperture:
        :return:
        """
        d = {'cutout':None,
             'hdu':None,
             'path':None,
             'filter':catalog_image['filter'],
             'instrument':catalog_image['instrument'],
             'mag':None,
             'aperture':None,
             'ap_center':None,
             'mag_limit':None,
             'details': None}

        try:
            wcs_manual = catalog_image['wcs_manual']
            mag_func = catalog_image['mag_func']
        except:
            wcs_manual = self.WCS_Manual
            mag_func = None

        wcs_idx = 0
        try:
            if catalog_image['image'] is None:
                catalog_image['image'] =  science_image.science_image(wcs_manual=wcs_manual,wcs_idx=wcs_idx,
                                                                      image_location=op.join(catalog_image['path'],
                                                                                             catalog_image['name']))
                catalog_image['image'].catalog_name = catalog_image['name']
                catalog_image['image'].filter_name = catalog_image['filter']

            sci = catalog_image['image']

            if (sci.headers is None) or (len(sci.headers) == 0): #the catalog_image['image'] is no good? reload?
                sci.load_image(wcs_manual=wcs_manual)

            d['path'] = sci.image_location
            d['hdu'] = sci.headers

            # to here, window is in degrees so ...
            window = 3600. * window

            if not error:
                error = window

            cutout,pix_counts, mag, mag_radius,details = sci.get_cutout(ra, dec, error=error, window=window, aperture=aperture,
                                                                        mag_func=mag_func,copy=True,return_details=True)
            # don't need pix_counts or mag, etc here, so don't pass aperture or mag_func
            if cutout is not None:  # construct master cutout
                d['cutout'] = cutout
                details['catalog_name']=self.name
                details['filter_name']=catalog_image['filter']
                d['mag_limit']=self.get_mag_limit(catalog_image['name'],mag_radius*2.)
                if (mag is not None) and (mag < 999):
                    if d['mag_limit'] and (d['mag_limit'] < mag < 100):
                        log.warning(f"Cutout mag {mag} greater than limit {d['mag_limit']}. Setting to limit.")
                        details['fail_mag_limit'] = True
                        details['raw_mag'] = mag
                        details['raw_mag_bright'] = details['mag_bright']
                        details['raw_mag_faint'] = details['mag_faint']
                        details['raw_mag_err'] = details['mag_err']
                        mag = d['mag_limit']
                        details['mag'] = mag

                        try:
                            details['mag_bright'] = min(mag,details['mag_bright'])
                        except:
                            details['mag_bright'] = mag
                        try:
                            details['mag_faint'] = max(mag,G.MAX_MAG_FAINT)
                        except:
                            details['mag_faint'] = G.MAX_MAG_FAINT

                    d['mag'] = mag
                    d['aperture'] = mag_radius
                    d['ap_center'] = (sci.last_x0_center, sci.last_y0_center)
                    d['details'] = details
        except:
            log.error("Error in get_single_cutout.", exc_info=True)

        return d

    def get_cutouts(self,ra,dec,window,aperture=None,filter=None,first=False,error=None,do_sky_subtract=True):
        l = list()

        tile = self.find_target_tile(ra, dec)

        if tile is None:
            # problem
            log.error("No appropriate tile found in CFHTLS for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        if aperture == -1:
            aperture = self.mean_FWHM * 0.5 + 0.5

        if filter:
            outer = filter
            inner = [x.lower() for x in self.Filters]
        else:
            outer = [x.lower() for x in self.Filters]
            inner = None

        wild_filters = iter([x.lower() for x in self.Filters])

        if outer:
            for f in outer:
                try:
                    if f == '*':
                        f = next(wild_filters, None)
                        if f is None:
                            break
                    elif inner and (f not in inner):
                        # if filter list provided but the image is NOT in the filter list go to next one
                        continue

                    lookup_tile = tile.replace('?',f,1)

                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == lookup_tile)))]
                    if i is not None:
                        cutout = self.get_single_cutout(ra, dec, window, i, aperture,error)

                        if first:
                            if cutout['cutout'] is not None:
                                l.append(cutout)
                                break
                        else:
                            # if we are not escaping on the first hit, append ALL cutouts (even if no image was collected)
                            l.append(cutout)
                except:
                    log.error("Exception! collecting image cutouts.", exc_info=True)
        else:
            for f in self.Filters:
                try:
                    lookup_tile = tile.replace('?',f,1)
                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == lookup_tile)))]
                except:
                    i = None

                if i is None:
                    continue

                l.append(self.get_single_cutout(ra,dec,window,i,aperture))

        return l



    def build_cat_summary_details(self,cat_match, ra, dec, error, bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        """
        similar to build_cat_summary_figure, but rather than build up an image section to be displayed in the
        elixer report, this builds up a dictionary of information to be aggregated later over multiple catalogs

        ***note: here we call the base class implementation to get the cutouts and then update those cutouts with
        any catalog specific changes

        :param cat_match: a match summary object (contains info about the PDF location, etc)
        :param ra:  the RA of the HETDEX detection
        :param dec:  the Dec of the HETDEX detection
        :param error: radius (or half-side of a box) in which to search for matches (the cutout is 3x this on a side)
        :param bid_ras: RAs of potential catalog counterparts
        :param bid_decs: Decs of potential catalog counterparts
        :param target_w: observed wavelength (from HETDEX)
        :param fiber_locs: array (or list) of 6-tuples that describe fiber locations (which fiber, position, color, etc)
        :param target_flux: HETDEX integrated line flux in CGS flux units (erg/s/cm2)
        :param detobj: the DetObj instance
        :return: cutouts list of dictionaries with bid-target objects as well
        """

        cutouts = super().build_cat_summary_details(cat_match, ra, dec, error, bid_ras, bid_decs, target_w,
                                                    fiber_locs, target_flux,detobj)

        if not cutouts:
            return cutouts

        for c in cutouts:
            try:
                details = c['details']
            except:
                pass


        #####################################################
        # BidTarget format is Unique to each child catalog
        #####################################################
        #now the bid targets
        #2. catalog entries as a new key under cutouts (like 'details') ... 'counterparts'
        #    this should be similar to the build_multiple_bid_target_figures_one_line()

        if len(bid_ras) > 0:
            #if there are no cutouts (but we do have a catalog), create a cutouts list of dictionries to hold the
            #counterparts
            if not cutouts or len(cutouts) == 0:
                cutouts = [{}]

            cutouts[0]['counterparts'] = []
            #create an empty list of counterparts under the 1st cutout
            #counterparts are not filter specific, so we will just keep one list under the 1st cutout

            #go ahead and get the photz catalog now if within rough range

            if self.df_photoz is None and (208.559 < ra < 220.391) and (51.211 < dec < 57.804):
                self.read_photoz_catalog()

            #and try to match up bid_ras and bid_decs



        target_count = 0
        # targets are in order of increasing distance
        for r, d in zip(bid_ras, bid_decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break
            df = None
            df_photz=None
            phot_z = None
            try: #DO NOT WANT _unique as that has wiped out the filters
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0])]
                #multiple filters
                #try to match up with photz
                if df is not None:
                    coord_error = 0.5/3600. #arcsec
                    #todo: allow for RA adjustment based on dec (but at 0.5" is pretty small error even at higher dec)
                    ra_min = r[0] - coord_error
                    ra_max = r[0] + coord_error
                    dec_min = d[0] - coord_error
                    dec_max = d[0] + coord_error
                    try:
                        df_photz= self.df_photoz[(self.df_photoz['RA'] >= ra_min) & (self.df_photoz['RA'] <= ra_max) &
                                                 (self.df_photoz['DEC'] >= dec_min) & (self.df_photoz['DEC'] <= dec_max)].copy()

                        #todo: check for number of hits and select best if more than one?
                        if (df_photz is not None) and len(df_photz) == 1: #singular match ... if more than one, can I actually narrow it down?
                            phot_z = df_photz['z_best'].values[0]

                    except:
                        log.info(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                #add flux (cont est)
                try:
                    #fluxes for HSC NEP are in micro-Jansky
                    filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = self.get_filter_flux(df)
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0
                    filter_mag = 0.0
                    filter_mag_bright = 0.0
                    filter_mag_faint = 0.0
                    filter_str = "NA"

                bid_target = None

                if (target_flux is not None) and (filter_fl):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        #fluxes for HSC NEP are in micro-Jansky
                        filter_fl_cgs = self.micro_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        filter_fl_cgs_unc = self.micro_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
                        # assumes no error in wavelength or c

                        try:
                            bid_target = match_summary.BidTarget()
                            bid_target.catalog_name = self.Name
                            bid_target.bid_ra = df['RA'].values[0]
                            bid_target.bid_dec = df['DEC'].values[0]
                            bid_target.distance = df['distance'].values[0] * 3600
                            bid_target.prob_match = df['dist_prior'].values[0]
                            bid_target.bid_flux_est_cgs = filter_fl_cgs
                            bid_target.bid_filter = filter_str
                            bid_target.bid_mag = filter_mag
                            bid_target.bid_mag_err_bright = filter_mag_bright
                            bid_target.bid_mag_err_faint = filter_mag_faint
                            bid_target.bid_flux_est_cgs_unc = filter_fl_cgs_unc
                            if phot_z is not None:
                                bid_target.phot_z = phot_z

                            if target_w:
                                try:

                                    ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                                    ew_u = abs(ew * np.sqrt(
                                        (detobj.estflux_unc / target_flux) ** 2 +
                                        (filter_fl_err / filter_fl) ** 2))

                                    bid_target.bid_ew_lya_rest = ew
                                    bid_target.bid_ew_lya_rest_err = ew_u

                                except:
                                    log.debug("Exception computing catalog EW: ", exc_info=True)

                                addl_waves = None
                                addl_flux = None
                                addl_ferr = None
                                try:
                                    addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                                    addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                                    addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                                except:
                                    pass

                                lineFlux_err = 0.
                                if detobj is not None:
                                    try:
                                        lineFlux_err = detobj.estflux_unc
                                    except:
                                        lineFlux_err = 0.

                                # build EW error from lineFlux_err and aperture estimate error
                                ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                                try:
                                    ew_obs_err = abs(ew_obs * np.sqrt(
                                        (lineFlux_err / target_flux) ** 2 +
                                        (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                                except:
                                    ew_obs_err = 0.

                                bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                                    line_prob.mc_prob_LAE(
                                        wl_obs=target_w,
                                        lineFlux=target_flux,
                                        lineFlux_err=lineFlux_err,
                                        continuum=bid_target.bid_flux_est_cgs,
                                        continuum_err=bid_target.bid_flux_est_cgs_unc,
                                        c_obs=None, which_color=None,
                                        addl_wavelengths=addl_waves,
                                        addl_fluxes=addl_flux,
                                        addl_errors=addl_ferr,
                                        sky_area=None,
                                        cosmo=None, lae_priors=None,
                                        ew_case=None, W_0=None,
                                        z_OII=None, sigma=None)

                                try:
                                    if plae_errors:
                                        bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
                                        bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
                                except:
                                    pass

                            try:
                                bid_target.add_filter('HSC NEP',filter_str,filter_fl_cgs,filter_fl_err)
                            except:
                                log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                            cat_match.add_bid_target(bid_target)
                            #you can get here w/o a detect object (i.e. if this is just a catalog hit w/o HETDEX)
                            try:  # no downstream edits so they can both point to same bid_target
                                if detobj is not None:
                                    detobj.bid_target_list.append(bid_target)
                            except:
                                log.warning("Unable to append bid_target to detobj.", exc_info=True)

                            try:
                                cutouts[0]['counterparts'].append(bid_target)
                            except:
                                log.warning("Unable to append bid_target to cutouts.", exc_info=True)
                        except:
                            log.debug('Unable to build bid_target.',exc_info=True)



        return cutouts

#######################################
# end class CFHTLS
#######################################
