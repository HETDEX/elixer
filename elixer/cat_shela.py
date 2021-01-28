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


import matplotlib
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.MatplotlibDeprecationWarning)
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
#import astropy.io.fits as fits
import astropy.table
#import astropy.utils.exceptions
#import warnings
#warnings.filterwarnings('ignore', category=astropy.utils.exceptions.AstropyUserWarning, append=True)


#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field


def shela_count_to_mag(count,cutout=None,headers=None):
    if count is not None:
        #if cutout is not None:
        #get the conversion factor, each tile is different
        try:
            #gain = float(sci_image[0].header['GAIN'])
            #nanofact = float(sci_image[0].header['NANOFACT'])
            magzero = float(headers[0]['MAGZERO'])
        except:
            #gain = 1.0
            nanofact = 0.0
            log.error("Exception in shela_count_to_mag",exc_info=True)
            return 99.9

        if count > 0:
            #return -2.5 * np.log10(count*nanofact) + magzero
            # counts for SHELA  ALREADY in nanojansky
            if isinstance(count,float):
                return -2.5 * np.log10(count) + magzero
            else:
                return -2.5 * np.log10(count.value) + magzero
        else:
            return 99.9  # need a better floor

class SHELA(cat_base.Catalog):
    # class variables
    SHELA_BASE_PATH = G.SHELA_BASE_PATH
    SHELA_CAT_PATH = G.SHELA_CAT_PATH
    SHELA_IMAGE_PATH = G.DECAM_IMAGE_PATH#G.SHELA_BASE_PATH
    MAG_LIMIT = 25.0 #closer to 24.7

    #not all tiles have all filters
    Filters = ['u','g','r','i','z']
    #Tiles = ['3','4','5','6']
    SHELA_Tiles = ['B3','B4','B5','B6']
    Tiles = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10',
             'B1','B2','B3','B4','B5','B6','B7','B8','B9','B10',
             'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']
    Img_ext = ['psfsci.fits','sci.fits'] #was psfsci.fits for just SHELA
    Cat_ext = ['dualcat.fits','cat.fits'] #was 'dualgcat.fits'
    loaded_cat_tiles = [] #tile lables, like "A1","C9", etc for which the catalog has already been loaded

    CONT_EST_BASE = None

    PhotoZ_combined_cat = op.join(G.SHELA_PHOTO_Z_COMBINED_PATH,"shela_decam_irac_vista_combined_catalog.fits")
    PhotoZ_master_cat = op.join(G.SHELA_PHOTO_Z_MASTER_PATH,"photz_master.zout.FITS")

    df = None
    df_photoz = None

    MainCatalog = "SHELA" #while there is no main catalog, this needs to be not None
    Name = "DECAM/SHELA"

    mean_FWHM = 1.0  # average 0.7 to about 1.0

    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min': 8.50, 'RA_max': 36.51, 'Dec_min': -4.0, 'Dec_max': 4.0}
    #approximate
    Tile_Coord_Range = {
                        'A1': {'RA_min':  8.50, 'RA_max': 11.31, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A2': {'RA_min': 11.30, 'RA_max': 14.11, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A3': {'RA_min': 14.10, 'RA_max': 16.91, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A5': {'RA_min': 19.70, 'RA_max': 22.51, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A6': {'RA_min': 22.50, 'RA_max': 25.31, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A7': {'RA_min': 25.30, 'RA_max': 28.11, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A8': {'RA_min': 28.10, 'RA_max': 30.91, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A9': {'RA_min': 30.90, 'RA_max': 33.71, 'Dec_min': 1.32, 'Dec_max': 4.0},
                        'A10':{'RA_min': 33.70, 'RA_max': 36.51, 'Dec_min': 1.32, 'Dec_max': 4.0},

                        'B1': {'RA_min':  8.50, 'RA_max': 11.31, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B2': {'RA_min': 11.30, 'RA_max': 14.11, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B3': {'RA_min': 14.10, 'RA_max': 16.91, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B5': {'RA_min': 19.70, 'RA_max': 22.51, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B6': {'RA_min': 22.50, 'RA_max': 25.31, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B7': {'RA_min': 25.30, 'RA_max': 28.11, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B8': {'RA_min': 28.10, 'RA_max': 30.91, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B9': {'RA_min': 30.90, 'RA_max': 33.71, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        'B10':{'RA_min': 33.70, 'RA_max': 36.51, 'Dec_min': -1.35, 'Dec_max': 1.34},

                        'C1': {'RA_min':  8.50, 'RA_max': 11.31, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C2': {'RA_min': 11.30, 'RA_max': 14.11, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C3': {'RA_min': 14.10, 'RA_max': 16.91, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C5': {'RA_min': 19.70, 'RA_max': 22.51, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C6': {'RA_min': 22.50, 'RA_max': 25.31, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C7': {'RA_min': 25.30, 'RA_max': 28.11, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C8': {'RA_min': 28.10, 'RA_max': 30.91, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C9': {'RA_min': 30.90, 'RA_max': 33.71, 'Dec_min': -4.0, 'Dec_max': -1.32},
                        'C10':{'RA_min': 33.70, 'RA_max': 36.51, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    }


    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    WCS_Manual = False

    AstroTable = None

# ColDefs(
#     name = 'NUMBER'; format = '1J'; disp = 'I10'
#     name = 'FLUXERR_ISO'; format = '1E'; unit = 'count'; disp = 'G12.7'
#     name = 'MAG_APER'; format = '25E'; unit = 'mag'; disp = 'F8.4'
#     name = 'MAGERR_APER'; format = '25E'; unit = 'mag'; disp = 'F8.4'
#     name = 'FLUX_AUTO'; format = '1E'; unit = 'count'; disp = 'G12.7'
#     name = 'FLUXERR_AUTO'; format = '1E'; unit = 'count'; disp = 'G12.7'
#     name = 'MAG_AUTO'; format = '1E'; unit = 'mag'; disp = 'F8.4'
#     name = 'MAGERR_AUTO'; format = '1E'; unit = 'mag'; disp = 'F8.4'
#     name = 'KRON_RADIUS'; format = '1E'; disp = 'F5.2'
#     name = 'THRESHOLD'; format = '1E'; unit = 'count'; disp = 'G12.7'
#     name = 'X_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F11.4'
#     name = 'Y_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F11.4'
#     name = 'ALPHA_J2000'; format = '1D'; unit = 'deg'; disp = 'F11.7'
#     name = 'DELTA_J2000'; format = '1D'; unit = 'deg'; disp = 'F11.7'
#     name = 'A_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
#     name = 'B_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
#     name = 'FLUX_RADIUS'; format = '1E'; unit = 'pixel'; disp = 'F10.3'
#     name = 'THETA_J2000'; format = '1E'; unit = 'deg'; disp = 'F6.2'
#     name = 'FWHM_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F8.2'
#     name = 'FWHM_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
#     name = 'FLAGS'; format = '1I'; disp = 'I3'
#     name = 'IMAFLAGS_ISO'; format = '1J'; disp = 'I9'
#     name = 'NIMAFLAGS_ISO'; format = '1J'; disp = 'I9'
#     name = 'CLASS_STAR'; format = '1E'; disp = 'F6.3'
# )

    # photz_master.zout.FITS fields
    # 'id',
    # 'z_spec',
    # 'z_a',
    # 'z_m1',
    # 'chi_a',
    # 'l68',
    # 'u68',
    # 'l95',
    # 'u95',
    # 'l99',
    # 'u99',
    # 'nfilt',
    # 'q_z',
    # 'z_peak',
    # 'peak_prob',
    # 'z_mc'


    #shela_decam_irac_vista_combined_catalog.fits fields
    # 'X_IMAGE',
    # 'Y_IMAGE',
    # 'RA',
    # 'DEC',
    # 'FWHM_IMAGE',
    # 'A_IMAGE',
    # 'B_IMAGE',
    # 'THETA_IMAGE',
    # 'ISOAREA_FROM_SEGMAP',
    # 'FLAGS',
    # 'FLUX_APER_1_u',
    # 'FLUX_APER_2_u',
    # 'SIGMA_APER_1_u',
    # 'SIGMA_APER_2_u',
    # 'FLUX_AUTO_u',
    # 'SIGMA_AUTO_u',
    # 'FLUX_ISO_u',
    # 'SIGMA_ISO_u',
    # 'FLUX_RADIUS_u',
    # 'KRON_RADIUS_u',
    # 'EXP_CENTER_PIXEL_u',
    # 'CLASS_STAR_u',
    # 'IMAFLAGS_ISO_u',
    # 'FLUX_APER_1_g',
    # 'FLUX_APER_2_g',
    # 'SIGMA_APER_1_g',
    # 'SIGMA_APER_2_g',
    # 'FLUX_AUTO_g',
    # 'SIGMA_AUTO_g',
    # 'FLUX_ISO_g',
    # 'SIGMA_ISO_g',
    # 'FLUX_RADIUS_g',
    # 'KRON_RADIUS_g',
    # 'EXP_CENTER_PIXEL_g',
    # 'CLASS_STAR_g',
    # 'IMAFLAGS_ISO_g',
    # 'FLUX_APER_1_r',
    # 'FLUX_APER_2_r',
    # 'SIGMA_APER_1_r',
    # 'SIGMA_APER_2_r',
    # 'FLUX_AUTO_r',
    # 'SIGMA_AUTO_r',
    # 'FLUX_ISO_r',
    # 'SIGMA_ISO_r',
    # 'FLUX_RADIUS_r',
    # 'KRON_RADIUS_r',
    # 'EXP_CENTER_PIXEL_r',
    # 'CLASS_STAR_r',
    # 'IMAFLAGS_ISO_r',
    # 'FLUX_APER_1_i',
    # 'FLUX_APER_2_i',
    # 'SIGMA_APER_1_i',
    # 'SIGMA_APER_2_i',
    # 'FLUX_AUTO_i',
    # 'SIGMA_AUTO_i',
    # 'FLUX_ISO_i',
    # 'SIGMA_ISO_i',
    # 'FLUX_RADIUS_i',
    # 'KRON_RADIUS_i',
    # 'EXP_CENTER_PIXEL_i',
    # 'CLASS_STAR_i',
    # 'IMAFLAGS_ISO_i',
    # 'FLUX_APER_1_z',
    # 'FLUX_APER_2_z',
    # 'SIGMA_APER_1_z',
    # 'SIGMA_APER_2_z',
    # 'FLUX_AUTO_z',
    # 'SIGMA_AUTO_z',
    # 'FLUX_ISO_z',
    # 'SIGMA_ISO_z',
    # 'FLUX_RADIUS_z',
    # 'KRON_RADIUS_z',
    # 'EXP_CENTER_PIXEL_z',
    # 'CLASS_STAR_z',
    # 'IMAFLAGS_ISO_z',
    # 'EBV',
    # 'EBV_STDDEV',
    # 'DETECT_IMG_FLUX_AUTO',
    # 'DETECT_IMG_FLUXERR_AUTO',
    # 'CATALOG_ID_A',
    # 'CATALOG_ID_B',
    # 'ch1_trflux_uJy',
    # 'ch2_trflux_uJy',
    # 'ch1_fluxvar_uJy',
    # 'ch2_fluxvar_uJy',
    # 'ch1_aper_errflux_uJy',
    # 'ch2_aper_errflux_uJy',
    # 'ch1_logprob',
    # 'ch2_logprob',
    # 'ch1_tractorflag',
    # 'ch2_tractorflag',
    # 'ch1_optpsf_arcs',
    # 'ch2_optpsf_arcs',
    # 'ch1_psfvar_arcs',
    # 'ch2_psfvar_arcs',
    # 'irac_ch1weightvalues',
    # 'irac_ch2weightvalues',
    # 'DECAM_FIELD_ID',
    # 'MASTER_ID',
    # 'FLUX_AUTO_K',
    # 'SIGMA_AUTO_K',
    # 'FLUX_AUTO_J',
    # 'SIGMA_AUTO_J',
    # 'VISTA_d2d_arcsec'

    #NUMBER is NOT unique across Tiles (as expected)
    #NUMBER is NOT unique within a Tile either (bummer)
    #so ... it is essentially useless, just an index-1
    #plus they have different entries per filter, so must match by exact RA, DEC? are they consistent??

    BidCols = ['NUMBER',  # int32
               'FLUXERR_ISO',  # ct float32
               'MAG_APER',  # [25] mag float32
               'MAGERR_APER',  # [25] mag float32
               'FLUX_AUTO',  # ct float32
               'FLUXERR_AUTO',  # ct float32
               'MAG_AUTO',  # mag float32
               'MAGERR_AUTO',  # mag float32
               'KRON-RADIUS', #
               'THRESHOLD', # ct float32
               'X_IMAGE',  # pix float32
               'Y_IMAGE',  # pix float32
               'ALPHA_J2000',  # deg float64
               'DELTA_J2000',  # deg float64
               'A_WORLD',  # deg float32
               'B_WORLD',  # deg float32
               'FLUX_RADIUS',  # pix float32
               'THETA_J2000',  #
               'FWHM_IMAGE',  # pix float32
               'FWHM_WORLD',  # deg float32
               'FLAGS',  # int16
               'IMAFLAGS_ISO',  # int32
               'NIMAFLAGS_ISO',  # int32
               'CLASS_STAR']  # float32

    CatalogImages = [] #built in constructor

    def __init__(self):
        super(SHELA, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_unique = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0
        self.master_cutout = None
        self.build_catalog_of_images()


    @classmethod
    def read_catalog(cls, catalog_loc=None, name=None, tile=None): #tile is a single tile like "A3"
        "This catalog is in a fits file"

        #ignore catalog_loc and name. Must use class defined.
        #build each tile and filter and concatenate into a single pandas dataframe

        if tile is not None:
            if tile in cls.loaded_cat_tiles:
                log.info("Catalog tile (%s) already loaded." %tile)
                return cls.df
        else:
            log.error("Cannot load catalog tile for DECAM/SHELA. No tile provided.")
            return None

        if name is None:
            name = cls.Name

        df_master = pd.DataFrame()

        for f in cls.Filters:
            for ext in cls.Cat_ext:
                cat_name = tile +'_'+f+'_'+ ext
                cat_loc = op.join(cls.SHELA_CAT_PATH, cat_name)

                if not op.exists(cat_loc):
                    continue

                log.debug("Building " + cls.Name + " " + cat_name + " dataframe...")

                try:
                    table = astropy.table.Table.read(cat_loc)#,format='fits')
                except Exception as e:
                    if type(e) is astropy.io.registry.IORegistryError:
                        log.error(name + " Exception attempting to open catalog file: (IORegistryError, bad format)" + cat_loc, exc_info=False)
                    else:
                        log.error(name + " Exception attempting to open catalog file: " + cat_loc, exc_info=True)
                    continue #try the next one  #exc_info = sys.exc_info()

                # convert into a pandas dataframe ... cannot convert directly to pandas because of the [25] lists
                # so build a pandas df with just the few columns we need for searching
                # then pull data from full astro table as needed

                try:
                    lookup_table = astropy.table.Table([table['NUMBER'], table['ALPHA_J2000'], table['DELTA_J2000'],
                                          table['FLUX_AUTO'],table['FLUXERR_AUTO'],table['MAG_AUTO'],table['MAGERR_AUTO']])
                    pddf = lookup_table.to_pandas()
                    old_names = ['NUMBER', 'ALPHA_J2000', 'DELTA_J2000']
                    new_names = ['ID', 'RA', 'DEC']
                    pddf.rename(columns=dict(zip(old_names, new_names)), inplace=True)
                    pddf['TILE'] = tile
                    pddf['FILTER'] = f

                    df_master = pd.concat([df_master,pddf])

                    cls.loaded_cat_tiles.append(tile)

                   # cls.AstroTable = table
                except:
                    log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
                    return None

                break #keep only the first found matching extensions

        cls.df = df_master
        return df_master

    @classmethod
    def merge_photoz_catalogs(cls, combined_cat_file=PhotoZ_combined_cat, master_cat_file=PhotoZ_master_cat):
        "This catalog is in a fits file"

        # if (0):
        #     print("!!!SKIPPING PHOTOZ CATALOGS ... BE SURE TO RESTORE!!!")
        #     return

        log.info("Merging DECAM/SHELA PhotoZ Catalogs (this may take a while) ...")
        print("Merging DECAM/SHELA PhotoZ Catalogs (this may take a while) ...")

        try:
            combined_table = astropy.table.Table.read(combined_cat_file)
            master_table = astropy.table.Table.read(master_cat_file)
            master_table.rename_column('id', 'MASTER_ID')
            cls.df_photoz = astropy.table.join(master_table, combined_table, ['MASTER_ID'])

            #error with .to_pandas(), so have to leave as astropy table (not as fast, but then
            #we are just doing direct lookup, not a search)
        except Exception as e:
            if type(e) is MemoryError:
                log.error("Exception (Memory) attempting to open and compbine photoz catalog files: \n%s\n%s"
                          %(combined_cat_file, master_cat_file), exc_info=False)
            else:
                log.error("Exception attempting to open and compbine photoz catalog files: \n%s\n%s"
                          %(combined_cat_file, master_cat_file), exc_info=True)
            return None


        return cls.df_photoz

    def build_catalog_of_images(self):
        for t in self.Tiles:
            for f in self.Filters:
                #if the file exists, add it
                for ext in self.Img_ext:
                    name = t+'_'+f+'_'+ ext
                    if op.exists(op.join(self.SHELA_IMAGE_PATH,name)):
                        self.CatalogImages.append(
                            {'path': self.SHELA_IMAGE_PATH,
                             'name': name, #'B'+t+'_'+f+'_'+self.Img_ext,
                             'tile': t,
                             'filter': f,
                             'instrument': "DECAM",
                             'cols': [],
                             'labels': [],
                             'image': None,
                             'expanded': False,
                             'wcs_manual': False,
                             'aperture': self.mean_FWHM*0.5 + 0.5, # since a radius, half the FWHM + 0.5" for astrometric error
                             'mag_func': shela_count_to_mag
                             })
                        break #(out of ext in self.Img_ext) keep the first name that is found

    def find_target_tile(self,ra,dec,verify=True):
        #assumed to have already confirmed this target is at least in coordinate range of this catalog
        tile = None
        for t in self.Tiles:
            # don't bother to load if ra, dec not in range
            try:
                coord_range = self.Tile_Coord_Range[t]
                # {'RA_min': 14.09, 'RA_max': 16.91, 'Dec_min': -1.35, 'Dec_max': 1.34}
                if not ((ra >= coord_range['RA_min']) and (ra <= coord_range['RA_max']) and
                        (dec >= coord_range['Dec_min']) and (dec <= coord_range['Dec_max'])) :
                    continue
            except:
                pass

            for c in self.CatalogImages:
                if c['tile'] != t:
                    continue

                try: #really a redundant check
                    if ((ra < self.Tile_Coord_Range[c['tile']]['RA_min']) or \
                       (ra > self.Tile_Coord_Range[c['tile']]['RA_max']) or \
                       (dec < self.Tile_Coord_Range[c['tile']]['Dec_min']) or \
                       (dec > self.Tile_Coord_Range[c['tile']]['Dec_max'])):
                        continue
                except:
                    log.warning("Minor Exception in cat_shela.py:find_target_tile ", exc_info=True)

                if verify: #collect image and confirm position is there
                    try:
                        image = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                            image_location=op.join(self.SHELA_IMAGE_PATH, c['name']))
                        if image.contains_position(ra, dec):
                            tile = t
                        else:
                            log.debug("position (%f, %f) is not in image. %s" % (ra, dec, c['name']))

                    except:
                        pass
                else:
                    tile = t

                if tile is not None: #break out of catalog search
                    break

            if tile is not None: #break out of tile search
                break

        return tile


    def get_filter_flux(self,df):

        filter_fl = None
        filter_fl_err = None
        mag = None
        mag_bright = None
        mag_faint = None
        filter_str = None
        try:

            filter_str = 'r'
            dfx = df.loc[df['FILTER'] == filter_str]

            if (dfx is None) or (len(dfx) == 0):
                filter_str = 'g'
                dfx = df.loc[df['FILTER'] == filter_str]

            if (dfx is None) or (len(dfx) == 0):
                filter_str = '?'
                log.error("Neither g-band nor r-band filter available.")
                return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

            filter_fl = dfx['FLUX_AUTO'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
            filter_fl_err = dfx['FLUXERR_AUTO'].values[0]
            mag = dfx['MAG_AUTO'].values[0]
            mag_faint = dfx['MAGERR_AUTO'].values[0]
            mag_bright = -1 * mag_faint

        except: #not the EGS df, try the CFHTLS
            pass

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        #if self.df is None:
        #    self.read_main_catalog()

        #even if not None, could be we need a different catalog, so check and append
        tile = self.find_target_tile(ra,dec)

        if tile is None:
            log.info("Could not locate tile for DECAM/SHELA. Discontinuing search of this catalog.")
            return -1,None,None

        #could be none or could be not loaded yet
        if self.df is None or not (set(tile).issubset(self.loaded_cat_tiles)):
            self.read_catalog(tile=tile)

        if self.df_photoz is None:
            self.merge_photoz_catalogs()

        error_in_deg = np.float64(error) / 3600.0

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
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
            #may contain duplicates (across tiles)
            #remove duplicates (assuming same RA,DEC between tiles has same data)
            #so, different tiles that have the same ra,dec and filter get dropped (keep only 1)
            #but if the filter is different, it is kept

            #this could be done at construction time, but given the smaller subset I think
            #this is faster here
            self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.drop_duplicates(
                subset=['RA','DEC','FILTER']) #keeps one of each filter


            #relying on auto garbage collection here ...
            #want to keep FILTER='g' or FILTER='r' if possible (r is better)
            self.dataframe_of_bid_targets_unique = \
                self.dataframe_of_bid_targets[self.dataframe_of_bid_targets['FILTER']=='r']

            if len(self.dataframe_of_bid_targets_unique) == 0:
                self.dataframe_of_bid_targets_unique = \
                    self.dataframe_of_bid_targets[self.dataframe_of_bid_targets['FILTER'] == 'g']

            if len(self.dataframe_of_bid_targets_unique) == 0:
                self.dataframe_of_bid_targets_unique = \
                    self.dataframe_of_bid_targets_unique.drop_duplicates(subset=['RA','DEC'])#,'FILTER'])

            self.num_targets = self.dataframe_of_bid_targets_unique.iloc[:,0].count()

        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.dataframe_of_bid_targets_unique is not None:
            #self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra, dec)

            log.info(self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
                     ". Found = %d" % (self.num_targets))

        return self.num_targets, self.dataframe_of_bid_targets_unique, None



    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None,detobj=None):

        self.clear_pages()
        self.build_list_of_bid_targets(target_ra, target_dec, error)

        if (self.dataframe_of_bid_targets_unique is None):
            return None

        ras = self.dataframe_of_bid_targets_unique.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets_unique.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            if G.BUILD_REPORT_BY_FILTER:
                #here we return a list of dictionaries (the "cutouts" from this catalog)
                return self.build_cat_summary_details(cat_match,target_ra, target_dec, error, ras, decs,
                                              target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                              detobj=detobj)
            else:
                entry = self.build_cat_summary_figure(cat_match,target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                                  detobj=detobj)
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")
            return None

        if entry is not None:
            self.add_bid_entry(entry)

            if G.SINGLE_PAGE_PER_DETECT: # and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
                entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                        target_ra=target_ra, target_dec=target_dec,
                                                                        target_w=target_w, target_flux=target_flux,
                                                                        detobj=detobj)
                if entry is not None:
                    self.add_bid_entry(entry)
        else:
            return None

#        else:  # each bid taget gets its own line
        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):  # each bid taget gets its own line
            log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

        return self.pages

    def get_stacked_cutout(self, ra, dec, window):

        stacked_cutout = None
        error = window

        # for a given Tile, iterate over all filters
        tile = self.find_target_tile(ra, dec)
        if tile is None:
            # problem
            print("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        for f in self.Filters:
            try:
                i = self.CatalogImages[
                    next(i for (i, d) in enumerate(self.CatalogImages)
                         if ((d['filter'] == f) and (d['tile'] == tile)))]
            except:
                i = None

            if i is None:
                continue

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
                log.error("Error in get_stacked_cutout.", exc_info=True)

        return stacked_cutout



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
                                                    fiber_locs, target_flux,detobj,do_sky_subtract=True)

        if not cutouts:
            return cutouts

        #####################################################
        # Nothing unique for the imaging needed here for candels
        #####################################################

        # for c in cutouts:
        #     try:
        #         details = c['details']
        #     except:
        #         pass


        #####################################################
        # BidTarget format is Unique to each child catalog
        #####################################################
        #now the bid targets
        #2. catalog entries as a new key under cutouts (like 'details') ... 'counterparts'
        #    this should be similar to the build_multiple_bid_target_figures_one_line()

        if not cutouts or len(cutouts) == 0:
            cutouts = [{}]

        cutouts[0]['counterparts'] = []

        target_count = 0
        # targets are in order of increasing distance
        for r, d in zip(bid_ras, bid_decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break

            try: #DO NOT WANT _unique (since that has wiped out the filters)
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'r')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                           (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0])]

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets_unique", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                best_fit_photo_z = 0.0 #SHELA has no photoz right now

                try:
                    filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = self.get_filter_flux(df)
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0
                    filter_mag = 0.0
                    filter_mag_bright = 0.0
                    filter_mag_faint = 0.0
                    filter_str = "NA"

                bid_target = None

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        #text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
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

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
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

                            dfx = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                                    (self.dataframe_of_bid_targets['DEC'] == d[0])]

                            for flt,flux,err in zip(dfx['FILTER'].values,
                                                    dfx['FLUX_AUTO'].values,
                                                    dfx['FLUXERR_AUTO'].values):
                                try:
                                    bid_target.add_filter('NA',flt,
                                                          self.nano_jansky_to_cgs(flux,SU.filter_iso(filter_str,target_w)),
                                                          self.nano_jansky_to_cgs(err,SU.filter_iso(filter_str,target_w)))
                                except:
                                    log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                        except:
                            log.debug('Unable to build bid_target.',exc_info=True)

                if bid_target:
                    cat_match.add_bid_target(bid_target)
                    try: # no downstream edits so they can both point to same bid_target
                        detobj.bid_target_list.append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to detobj.",exc_info=True)
                    try:
                        cutouts[0]['counterparts'].append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to cutouts.", exc_info=True)

        return cutouts

    def build_cat_summary_figure (self, cat_match, ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

        rows = 10
        #cols = 1 + len(self.CatalogImages)/len(self.Tiles)
        #note: setting size to 7 from 6 so they will be the right size (the 7th position will not be populated)
        cols = 7 # 1 for the fiber position and up to 5 filters for any one tile (u,g,r,i,z)

        fig_sz_x = 18 #cols * 3
        fig_sz_y = 3 #ows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        # for a given Tile, iterate over all filters
        tile = self.find_target_tile(ra, dec)

        if tile is None:
            # problem
            print("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        # All on one line now across top of plots
        if G.ZOO:
            title = "Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets_unique), error)
        else:
            title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets_unique), error)

        # title += "  Minimum (no match) 3$\sigma$ rest-EW: "
        # cont_est  = -1
        # if target_flux  and self.CONT_EST_BASE:
        #     cont_est = self.CONT_EST_BASE*3
        #     if cont_est != -1:
        #         title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
        #         if target_w >= G.OII_rest:
        #             title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
        #         else:
        #             title = title + "  OII = N/A"
        #     else:
        #         title += "  LyA = N/A  OII = N/A"
        # else:
        #     title += "  LyA = N/A  OII = N/A"


        plt.subplot(gs[0, :])
        text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        ref_exptime = 1.0
        total_adjusted_exptime = 1.0
        bid_colors = self.get_bid_colors(len(bid_ras))
        exptime_cont_est = -1
        index = 0 #images go in positions 1+ (0 is for the fiber positions)

        best_plae_poii = None
        best_plae_poii_filter = '-'
        best_plae_range = None

        for f in self.Filters:
            try:
                i = self.CatalogImages[
                    next(i for (i, d) in enumerate(self.CatalogImages)
                         if ((d['filter'] == f) and (d['tile'] == tile)))]
            except:
                i = None

            if i is None:
                continue

            index += 1

            if index > cols:
                log.warning("Exceeded max number of grid spec columns.")
                break #have to be done

            try:
                wcs_manual = i['wcs_manual']
                aperture = i['aperture']
                mag_func = i['mag_func']
            except:
                wcs_manual = self.WCS_Manual
                aperture = 0.0
                mag_func = None

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=wcs_manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            #the filters are in order, use g if f is not there
            if (f == 'g') and (sci.exptime is not None) and (exptime_cont_est == -1):
                exptime_cont_est = sci.exptime

            # the filters are in order, so this will overwrite g
            if (f == 'r') and (sci.exptime is not None):
                exptime_cont_est = sci.exptime

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag, mag_radius,details = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func,return_details=True)

            if (self.MAG_LIMIT < mag < 100) and (mag_radius > 0):
                details['fail_mag_limit'] = True
                details['raw_mag'] = mag
                details['raw_mag_bright'] = details['mag_bright']
                details['raw_mag_faint'] = details['mag_faint']
                details['raw_mag_err'] = details['mag_err']
                log.warning(f"Cutout mag {mag} greater than limit {self.MAG_LIMIT}. Setting to limit.")
                mag = self.MAG_LIMIT
                if details:
                    details['mag'] = mag
                    try:
                        details['mag_bright'] = min(mag,details['mag_bright'])
                    except:
                        details['mag_bright'] = mag
                    try:
                        details['mag_faint'] = max(mag,G.MAX_MAG_FAINT)
                    except:
                        details['mag_faint'] = G.MAX_MAG_FAINT

            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            bid_target = None
            cutout_ewr = None
            cutout_ewr_err = None
            cutout_plae = None

            try:  # update non-matched source line with PLAE()
                if (mag < 99)  and (target_flux is not None) and (i['filter'] in 'gr'):
                    # make a "blank" catalog match (e.g. at this specific RA, Dec (not actually from catalog)
                    bid_target = match_summary.BidTarget()
                    bid_target.catalog_name = self.Name
                    bid_target.bid_ra = 666  # nonsense RA
                    bid_target.bid_dec = 666  # nonsense Dec
                    bid_target.distance = 0.0
                    bid_target.bid_filter = i['filter']
                    bid_target.bid_mag = mag
                    bid_target.bid_mag_err_bright = 0.0 #todo: right now don't have error on aperture mag
                    bid_target.bid_mag_err_faint = 0.0
                    bid_target.bid_flux_est_cgs_unc = 0.0

                    if mag < 99:
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag, SU.filter_iso(i['filter'],target_w))
                        try:
                            flux_faint = None
                            flux_bright = None

                            if details['mag_faint'] < 99:
                                flux_faint = self.obs_mag_to_cgs_flux(details['mag_faint'], SU.filter_iso(i['filter'],target_w))

                            if details['mag_bright'] < 99:
                                flux_bright = self.obs_mag_to_cgs_flux(details['mag_bright'], SU.filter_iso(i['filter'],target_w))

                            if flux_bright and flux_faint:
                                bid_target.bid_flux_est_cgs_unc = max((bid_target.bid_flux_est_cgs - flux_faint),
                                                                      (flux_bright - bid_target.bid_flux_est_cgs))
                            elif flux_bright:
                                bid_target.bid_flux_est_cgs_unc = flux_bright - bid_target.bid_flux_est_cgs

                        except:
                            pass

                    else:
                        bid_target.bid_flux_est_cgs = cont_est

                    try:
                        bid_target.bid_mag_err_bright = mag - details['mag_bright']
                        bid_target.bid_mag_err_faint = details['mag_faint'] - mag
                    except:
                        pass

                    bid_target.add_filter(i['instrument'], i['filter'], bid_target.bid_flux_est_cgs, -1)

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

                    #build EW error from lineFlux_err and aperture estimate error
                    ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                    try:
                        ew_obs_err =  abs(ew_obs * np.sqrt(
                                        (lineFlux_err / target_flux) ** 2 +
                                        (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                    except:
                        ew_obs_err = 0.

                    # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                    #     line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                    #                        ew_obs=ew_obs,
                    #                        lineFlux_err= lineFlux_err,
                    #                        ew_obs_err= ew_obs_err,
                    #                        c_obs=None, which_color=None, addl_fluxes=addl_flux,
                    #                        addl_wavelengths=addl_waves,addl_errors=addl_ferr,sky_area=None,
                    #                        cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                    #                        sigma=None,estimate_error=True)

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
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

                    cutout_plae = bid_target.p_lae_oii_ratio
                    cutout_ewr = ew_obs / (1. + target_w / G.LyA_rest)
                    cutout_ewr_err = ew_obs_err / (1. + target_w / G.LyA_rest)

                    if best_plae_poii is None or i['filter'] == 'g': #favor g over r
                        best_plae_poii = bid_target.p_lae_oii_ratio
                        best_plae_poii_filter = i['filter']
                        if plae_errors:
                            try:
                                best_plae_range = plae_errors['ratio']
                            except:
                                pass

                    #if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    #    text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g" % (bid_target.p_lae_oii_ratio))

                    cat_match.add_bid_target(bid_target)
                    try:  # no downstream edits so they can both point to same bid_target
                        if detobj is not None:
                            detobj.bid_target_list.append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to detobj.", exc_info=True)
            except:
                log.debug('Could not build exact location photometry info.', exc_info=True)

            if (not G.ZOO) and (bid_target is not None) and (i['filter'] in 'gr'):
                try:
                    text.set_text(
                        text.get_text() + "  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$ (%s)" %
                            (round(bid_target.p_lae_oii_ratio, 3),
                             round(bid_target.p_lae_oii_ratio_max, 3),
                             round(bid_target.p_lae_oii_ratio_min, 3),
                             i['filter']))
                except:
                    log.debug("Exception adding PLAE with range",exc_info=True)
                    try:
                        text.set_text(text.get_text() + "  P(LAE)/P(OII): %0.4g (%s)" %
                                  (bid_target.p_lae_oii_ratio,i['filter']))
                    except:
                        text.set_text(
                            text.get_text() + "  P(LAE)/P(OII): (%s) (%s)" % ("---", i['filter']))


            if cutout is not None:  # construct master cutout
                # 1st cutout might not be what we want for the master (could be a summary image from elsewhere)
                if self.master_cutout:
                    if self.master_cutout.shape != cutout.shape:
                        del self.master_cutout
                        self.master_cutout = None

                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True,reset_center=False)
                    #self.master_cutout,_,_,_ = sci.get_cutout(ra, dec, error, window=window, copy=True)
                    if sci.exptime:
                        ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    try:
                        self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
                    except:
                        log.warning("Unexpected exception.", exc_info=True)

                plt.subplot(gs[1:, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                #plt.plot(0, 0, "r+")
                self.add_zero_position(plt)

                if pix_counts is not None:
                    details['catalog_name'] = self.name
                    details['filter_name'] = i['filter']
                    details['aperture_eqw_rest_lya'] = cutout_ewr
                    details['aperture_eqw_rest_lya_err'] = cutout_ewr_err
                    details['aperture_plae'] = cutout_plae
                    try:
                        if plae_errors:
                            details['aperture_plae_min'] = plae_errors['ratio'][1]
                            details['aperture_plae_max'] = plae_errors['ratio'][2]
                    except:
                        details['aperture_plae_min'] = None
                        details['aperture_plae_max'] = None

                    cx = sci.last_x0_center
                    cy = sci.last_y0_center
                    if (details['sep_objects'] is not None): # and (details['sep_obj_idx'] is not None):
                        self.add_elliptical_aperture_positions(plt,details['sep_objects'],details['sep_obj_idx'],
                                                               mag_radius,mag,cx,cy,cutout_ewr,cutout_plae)
                    else:
                        self.add_aperture_position(plt, mag_radius, mag, cx, cy, cutout_ewr, cutout_plae)

                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    self.add_catalog_position(plt,
                                              x=(fx-x)-target_box_side / 2.0,
                                              y=(fy-y)-target_box_side / 2.0,
                                              size=target_box_side, color=bc)
                    # plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                    #                                   width=target_box_side, height=target_box_side,
                    #                                   angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))

            if (details is not None) and (detobj is not None):
                detobj.aperture_details_list.append(details)

        #if False:
        #    if target_flux is not None:
        #        #todo: get exptime from the tile (science image has it)
        #        cont_est = self.get_f606w_max_cont(exptime_cont_est, 3,self.CONT_EST_BASE)
        #        if cont_est != -1:
        #            title += "Minimum (no match)\n  3$\sigma$ rest-EW:\n"
        #            title += "  LyA = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.LyA_rest))
        #            if target_w >= G.OII_rest:
        #                title = title + "  OII = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.OII_rest))
        #            else:
        #                title = title + "  OII = N/A\n"

            #plt.subplot(gs[0, 0])
            #plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
            #plt.gca().set_frame_on(False)
            #plt.gca().axis('off')

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            return None #empty (catch_all) will produce fiber locations
            # still need to plot relative fiber positions here
            # plt.subplot(gs[1:, 0])
            # return self.build_empty_cat_summary_figure(ra, dec, error, bid_ras, bid_decs, target_w=target_w,
            #                                            fiber_locs=fiber_locs)
        else:
            self.master_cutout.data /= total_adjusted_exptime

        plt.subplot(gs[1:, 0])
        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)
        # complete the entry
        plt.close()

        # get zoo style cutout as png
        if G.ZOO_MINI and (detobj is not None):
            plt.figure()
            self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout, unlabeled=True)

            plt.gca().set_axis_off()

            box_ratio = 1.0#0.99
            # add window outline
            xl, xr = plt.gca().get_xlim()
            yb, yt = plt.gca().get_ylim()
            zero_x = (xl + xr) / 2.
            zero_y = (yb + yt) / 2.
            rx = (xr - xl) * box_ratio / 2.0
            ry = (yt - yb) * box_ratio / 2.0

            plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2 , height=ry * 2,
                                              angle=0, color='red', fill=False,linewidth=8))

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300,transparent=True)
            detobj.image_cutout_fiber_pos = buf
            plt.close()

        return fig


    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None,detobj=None):

        rows = 1
        cols = 6

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)

        #col(0) = "labels", 1..3 = bid targets, 4..5= Zplot
        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        # entry text
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        #row labels
        plt.subplot(gs[0, 0])
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if len(ras) < 1:
            # per Karl insert a blank row
            text = "No matching targets in catalog.\nRow intentionally blank."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            plt.close()
            return fig
        elif (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):
            text = "Too many matching targets in catalog.\nIndividual target reports on followin pages."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            plt.close()
            return fig


        bid_colors = self.get_bid_colors(len(ras))

        if G.ZOO:
            text = "Separation\n" + \
                   "Match score\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "mag\n\n"
        else:
            text = "Separation\n" + \
                   "Match score\n" + \
                   "RA, Dec\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "mag\n" + \
                   "P(LAE)/P(OII)\n"


        plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)

        col_idx = 0
        target_count = 0
        # targets are in order of increasing distance
        for r, d in zip(ras, decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break
            col_idx += 1
            try: #DO NOT WANT _unique (since that has wiped out the filters)
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'r')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets_unique", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                text = ""

                if G.ZOO:
                    text = text + "%g\"\n%0.3f\n" \
                                  % (df['distance'].values[0] * 3600.,df['dist_prior'].values[0])
                else:
                    text = text + "%g\"\n%0.3f\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600.,df['dist_prior'].values[0],
                                    df['RA'].values[0], df['DEC'].values[0])

                best_fit_photo_z = 0.0 #SHELA has no photoz right now
                #try:
                #    best_fit_photo_z = float(df['Best-fit photo-z'].values[0])
                #except:
                #    pass

                #text += "N/A\n" + "%g\n" % best_fit_photo_z
                text += "N/A\nN/A\n"

                try:
                    filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = self.get_filter_flux(df)
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0
                    filter_mag = 0.0
                    filter_mag_bright = 0.0
                    filter_mag_faint = 0.0
                    filter_str = "NA"

                bid_target = None

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        #text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
                        # assumes no error in wavelength or c

                        # try:
                        #     ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        #     ew_u = abs(ew * np.sqrt(
                        #                 (detobj.estflux_unc / target_flux) ** 2 +
                        #                 (filter_fl_err / filter_fl) ** 2))
                        #     text = text + utilities.unc_str((ew,ew_u)) + "$\AA$\n"
                        # except:
                        #     log.debug("Exception computing catalog EW: ",exc_info=True)
                        #     text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

                        # if target_w >= G.OII_rest:
                        #     text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.OII_rest))
                        # else:
                        #     text = text + "N/A\n"
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

                            try:
                                ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                                ew_u = abs(ew * np.sqrt(
                                    (detobj.estflux_unc / target_flux) ** 2 +
                                    (filter_fl_err / filter_fl) ** 2))

                                bid_target.bid_ew_lya_rest = ew
                                bid_target.bid_ew_lya_rest_err = ew_u

                                text = text + utilities.unc_str((ew, ew_u)) + "$\AA$\n"
                            except:
                                log.debug("Exception computing catalog EW: ", exc_info=True)
                                text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

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

                            # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                            #     line_prob.prob_LAE(wl_obs=target_w,
                            #                        lineFlux=target_flux,
                            #                        ew_obs=ew_obs,
                            #                        lineFlux_err=lineFlux_err,
                            #                        ew_obs_err=ew_obs_err,
                            #                        c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                            #                        addl_fluxes=addl_flux, addl_errors=addl_ferr, sky_area=None,
                            #                        cosmo=None, lae_priors=None,
                            #                        ew_case=None, W_0=None,
                            #                        z_OII=None, sigma=None, estimate_error=True)

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
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

                            dfx = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                                    (self.dataframe_of_bid_targets['DEC'] == d[0])]

                            for flt,flux,err in zip(dfx['FILTER'].values,
                                                    dfx['FLUX_AUTO'].values,
                                                    dfx['FLUXERR_AUTO'].values):
                                try:
                                    bid_target.add_filter('NA',flt,
                                                          self.nano_jansky_to_cgs(flux,SU.filter_iso(filter_str,target_w)),
                                                          self.nano_jansky_to_cgs(err,SU.filter_iso(filter_str,target_w)))
                                except:
                                    log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                            cat_match.add_bid_target(bid_target)
                            try:  # no downstream edits so they can both point to same bid_target
                                detobj.bid_target_list.append(bid_target)
                            except:
                                log.warning("Unable to append bid_target to detobj.", exc_info=True)
                        except:
                            log.debug('Unable to build bid_target.',exc_info=True)


                else:
                    text += "N/A\nN/A\n"

                try:
                    text = text + "%0.2f(%0.2f)%s\n" % (filter_mag, filter_mag_faint,filter_str)
                except:
                    log.warning("Magnitude info is none: mag(%s), mag_bright(%s), mag_faint(%s)"
                                % (filter_mag, filter_mag_bright, filter_mag_faint))
                    text += "No mag info\n"

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    try:
                        text += r"$%0.4g\ ^{%.4g}_{%.4g}$" % (utilities.saferound(bid_target.p_lae_oii_ratio, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_max, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_min, 3))
                        text += "\n"
                    except:
                        text += "%0.4g\n" % (bid_target.p_lae_oii_ratio)
                else:
                    text += "\n\n"
            else:
                text = "%s\n%f\n%f\n" % ("--",r, d)

            plt.subplot(gs[0, col_idx])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font,color=bid_colors[col_idx-1])

            # fig holds the entire page

            #todo: photo z plot if becomes available
            plt.subplot(gs[0, 4:])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            text = "Photo z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        plt.close()
        return fig



    def get_single_cutout(self, ra, dec, window, catalog_image,aperture=None,error=None,do_sky_subtract=True):

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

        try:
            if catalog_image['image'] is None:
                catalog_image['image'] =  science_image.science_image(wcs_manual=wcs_manual,
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

            cutout, pix_counts, mag, mag_radius, details = sci.get_cutout(ra, dec, error=error, window=window,
                                                                          aperture=aperture,
                                                                          mag_func=mag_func, copy=True,
                                                                          return_details=True)
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
                    else:
                        d['mag'] = mag

                    d['aperture'] = mag_radius
                    d['ap_center'] = (sci.last_x0_center, sci.last_y0_center)
                    d['details'] = details
        except:
            log.error("Error in get_single_cutout.", exc_info=True)

        return d

    def get_cutouts(self,ra,dec,window,aperture=None,filter=None,first=False,error=None,do_sky_subtract=True):
        l = list()

        tile = self.find_target_tile(ra, dec,verify=False)

        if tile is None:
            # problem
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        if aperture == -1:
            try:
                aperture = self.mean_FWHM * 0.5 + 0.5
            except:
                pass


        if filter:
            outer = filter
            inner = [x.lower() for x in self.Filters]
        else:
            outer = self.Filters
            inner = None

        wild_filters = iter([x.lower() for x in self.Filters])

        for f in outer:
            try:
                if f == '*':
                    f = next(wild_filters, None)
                    if f is None:
                        break
                elif inner and (f not in inner):
                    # if filter list provided but the image is NOT in the filter list go to next one
                    continue

                try:
                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == tile)))]
                except:
                    i = None

                if i is None:
                    continue

                cutout = self.get_single_cutout(ra, dec, window, i, aperture,error)

                if first:
                    if cutout['cutout'] is not None:
                        l.append(cutout)
                        break
                else:
                    # if we are not escaping on the first hit, append ALL cutouts (even if no image was collected)
                    l.append(cutout)

            except:
                log.error("Exception! collecting filter cutout.",exc_info=True)

        return l
