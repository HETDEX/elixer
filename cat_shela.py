from __future__ import print_function

import global_config as G
import os.path as op

STACK_COSMOS_BASE_PATH = G.STACK_COSMOS_BASE_PATH
STACK_COSMOS_IMAGE = op.join(G.STACK_COSMOS_BASE_PATH,"COSMOS_g_sci.fits")
STACK_COSMOS_CAT = op.join(G.STACK_COSMOS_CAT_PATH,"cat_g.fits")


import matplotlib
matplotlib.use('agg')

import pandas as pd
import science_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
#import astropy.io.fits as fits
import astropy.table
#import astropy.utils.exceptions
#import warnings
#warnings.filterwarnings('ignore', category=astropy.utils.exceptions.AstropyUserWarning, append=True)


log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

import cat_base
import match_summary

class SHELA(cat_base.Catalog):
    # class variables
    SHELA_BASE_PATH = G.SHELA_BASE_PATH
    SHELA_IMAGE_PATH = G.SHELA_BASE_PATH

    Filters = ['u','g','r','i','z']
    Tiles = ['3','4','5','6']
    Img_ext = 'psfsci.fits'
    Cat_ext = 'dualgcat.fits'

    CONT_EST_BASE = None

    PhotoZ_combined_cat = op.join(G.SHELA_PHOTO_Z_COMBINED_PATH,"shela_decam_irac_vista_combined_catalog.fits")
    PhotoZ_master_cat = op.join(G.SHELA_PHOTO_Z_MASTER_PATH,"photz_master.zout.FITS")

    df = None
    df_photoz = None

    MainCatalog = "SHELA" #while there is no main catalog, this needs to be not None
    Name = "SHELA"

    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min': 14.09, 'RA_max': 25.31, 'Dec_min': -1.35, 'Dec_max': 1.34}
    #approximate
    Tile_Coord_Range = {'3': {'RA_min': 14.09, 'RA_max': 16.91, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        '4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        '5': {'RA_min': 19.70, 'RA_max': 22.52, 'Dec_min': -1.35, 'Dec_max': 1.34},
                        '6': {'RA_min': 22.51, 'RA_max': 25.31, 'Dec_min': -1.35, 'Dec_max': 1.34},
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
    def read_catalog(cls, catalog_loc=None, name=None):
        "This catalog is in a fits file"

        #ignore catalog_loc and name. Must use class defined.
        #build each tile and filter and concatenate into a single pandas dataframe

        df_master = pd.DataFrame()

        for t in cls.Tiles:
            for f in cls.Filters:
                cat_name = 'B'+t+'_'+f+'_'+cls.Cat_ext
                log.debug("Building " + cls.Name + " " + cat_name + " dataframe...")

                cat_loc = op.join(cls.SHELA_BASE_PATH,cat_name)

                try:
                    table = astropy.table.Table.read(cat_loc)
                except:
                    log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
                    return None

                # convert into a pandas dataframe ... cannot convert directly to pandas because of the [25] lists
                # so build a pandas df with just the few columns we need for searching
                # then pull data from full astro table as needed

                try:
                    lookup_table = astropy.table.Table([table['NUMBER'], table['ALPHA_J2000'], table['DELTA_J2000'],
                                          table['FLUX_AUTO'],table['FLUXERR_AUTO']])
                    pddf = lookup_table.to_pandas()
                    old_names = ['NUMBER', 'ALPHA_J2000', 'DELTA_J2000']
                    new_names = ['ID', 'RA', 'DEC']
                    pddf.rename(columns=dict(zip(old_names, new_names)), inplace=True)
                    pddf['TILE'] = t
                    pddf['FILTER'] = f

                    df_master = pd.concat([df_master,pddf])

                   # cls.AstroTable = table
                except:
                    log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
                    return None

        cls.df = df_master
        return df_master

    @classmethod
    def merge_photoz_catalogs(cls, combined_cat_file=PhotoZ_combined_cat, master_cat_file=PhotoZ_master_cat):
        "This catalog is in a fits file"

        try:
            combined_table = astropy.table.Table.read(combined_cat_file)
            master_table = astropy.table.Table.read(master_cat_file)
            master_table.rename_column('id', 'MASTER_ID')
            cls.df_photoz = astropy.table.join(master_table, combined_table, ['MASTER_ID'])

            #error with .to_pandas(), so have to leave as astropy table (not as fast, but then
            #we are just doing direct lookup, not a search)
        except:
            log.error("Exception attempting to open and compbine photoz catalog files: \n%s\n%s"
                      %(combined_cat_file, master_cat_file), exc_info=True)
            return None


        return cls.df_photoz

    def build_catalog_of_images(self):
        for t in self.Tiles:
            for f in self.Filters:
                self.CatalogImages.append(
                    {'path': self.SHELA_IMAGE_PATH,
                     'name': 'B'+t+'_'+f+'_'+self.Img_ext,
                     'tile': t,
                     'filter': f,
                     'instrument': "",
                     'cols': [],
                     'labels': [],
                     'image': None
                     })

    def find_target_tile(self,ra,dec):
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

            for f in ['g']: #self.Filters:
                #can we assume the filters all have the same coord range?
                #see if can get a cutout?
                img_name = 'B' + t + '_' + f + '_' + self.Img_ext
                try:
                    image = science_image.science_image(wcs_manual=self.WCS_Manual,
                            image_location=op.join(self.SHELA_IMAGE_PATH, img_name))
                    if image.contains_position(ra,dec):
                        tile = t
                    else:
                        log.debug("position (%f, %f) is not in image. %s" % (ra, dec,img_name))

                except:
                    pass

                if tile is not None:
                    break

        return tile

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        if self.df is None:
            self.read_main_catalog()

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
                subset=['RA','DEC','FILTER'])


            #relying on auto garbage collection here ...
            self.dataframe_of_bid_targets_unique = self.dataframe_of_bid_targets.copy()
            self.dataframe_of_bid_targets_unique = \
                self.dataframe_of_bid_targets_unique.drop_duplicates(subset=['RA','DEC'])
            self.num_targets = self.dataframe_of_bid_targets_unique.iloc[:,0].count()

        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.dataframe_of_bid_targets is not None:
            #self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra, dec)

            log.info(self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
                     ". Found = %d" % (self.num_targets))

        return self.num_targets, self.dataframe_of_bid_targets, None



    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None):

        self.clear_pages()
        self.build_list_of_bid_targets(target_ra, target_dec, error)

        ras = self.dataframe_of_bid_targets_unique.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets_unique.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
            entry = self.build_cat_summary_figure(target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux)
        else:
            entry = self.build_exact_target_location_figure(target_ra, target_dec, error, section_title=section_title,
                                                            target_w=target_w, fiber_locs=fiber_locs,
                                                            target_flux=target_flux)
        if entry is not None:
            self.add_bid_entry(entry)

        if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
            entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                    target_ra=target_ra, target_dec=target_dec,
                                                                    target_w=target_w, target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)
        else:  # each bid taget gets its own line

            bid_colors = self.get_bid_colors(len(ras))
            number = 0
            for r, d in zip(ras, decs):
                number += 1
                try:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                           (self.dataframe_of_bid_targets['FILTER'] == 'r')]
                    if df is None:
                        df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                               (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                               (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                    if df is None:
                        df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                               (self.dataframe_of_bid_targets['DEC'] == d[0])]

                except:
                    log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                    continue  # this must be here, so skip to next ra,dec

                print("Building report for bid target %d in %s" % (base_count + number, self.Name))

                if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
                    entry = self.build_bid_target_figure_one_line(cat_match, r[0], d[0], error=error, df=df,
                                                                  df_photoz=None,
                                                                  target_ra=target_ra, target_dec=target_dec,
                                                                  section_title=section_title,
                                                                  bid_number=number, target_w=target_w,
                                                                  of_number=num_hits - base_count,
                                                                  target_flux=target_flux, color=bid_colors[number - 1])
                else:
                    entry = self.build_bid_target_figure(cat_match, r[0], d[0], error=error, df=df, df_photoz=None,
                                                         target_ra=target_ra, target_dec=target_dec,
                                                         section_title=section_title,
                                                         bid_number=number, target_w=target_w,
                                                         of_number=num_hits - base_count,
                                                         target_flux=target_flux)
                if entry is not None:
                    self.add_bid_entry(entry)

        return self.pages

    def build_exact_target_location_figure(self, ra, dec, error, section_title="", target_w=0, fiber_locs=None,
                                           target_flux=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 4

        rows = 2
        cols = len(self.Filters)

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        title = "Catalog: %s\n" % self.Name + section_title + "\nPossible Matches = %d (within %g\")\n" \
                                                              "RA = %f    Dec = %f\n" % (
                                                                  len(self.dataframe_of_bid_targets), error, ra, dec)
        if target_w > 0:
            title = title + "Wavelength = %g $\AA$\n" % target_w
        else:
            title = title + "\n"
        plt.subplot(gs[0, 0])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        #for a given Tile, iterate over all filters
        tile = self.find_target_tile(ra,dec)

        if tile is None:
            #problem
            print("+++++ No appropriate tile found in SHELA for RA,DEC = [%f,%f]" %(ra,dec))
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" %(ra,dec))
            return None

        index = -1
        for f in self.Filters:
            index += 1

            i = self.CatalogImages[
                next(i for (i,d) in enumerate(self.CatalogImages)
                     if ((d['filter'] == f) and (d['tile'] == tile)))]

            if i is None:
                continue

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            if cutout is not None:  # construct master cutout
                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout = sci.get_cutout(ra, dec, error, window=window, copy=True)
                else:
                    self.master_cutout.data = np.add(self.master_cutout.data, cutout.data)

                plt.subplot(gs[rows - 1, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])

                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            return None

        # plot the master cutout
        empty_sci = science_image.science_image()
        plt.subplot(gs[0, cols - 1])
        vmin, vmax = empty_sci.get_vrange(self.master_cutout.data)
        plt.imshow(self.master_cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                   vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
        plt.title("Master Cutout (Stacked)")
        plt.xlabel("arcsecs")
        plt.xticks([ext, ext / 2., 0, -ext / 2., -ext])
        plt.yticks([ext, ext / 2., 0, -ext / 2., -ext])
        # only show this lable if there is not going to be an adjacent fiber plot
        if (fiber_locs is None) or (len(fiber_locs) == 0):
            plt.ylabel("arcsecs")
        plt.plot(0, 0, "r+")

        self.add_north_box(plt, empty_sci, self.master_cutout, error, 0, 0, theta=None)

        # plot the fiber cutout
        if (fiber_locs is not None) and (len(fiber_locs) > 0):
            plt.subplot(gs[0, cols - 2])

            plt.title("Fiber Positions")
            plt.xlabel("arcsecs")
            plt.ylabel("arcsecs")

            plt.plot(0, 0, "r+")

            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')

            x, y = empty_sci.get_position(ra, dec, self.master_cutout)  # zero (absolute) position

            self.add_north_box(plt, empty_sci, self.master_cutout, error, 0, 0, theta=None)

            for r, d, c, i in fiber_locs:
                # print("+++++ Cutout RA,DEC,ID,COLOR", r,d,i,c)
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx, fy = empty_sci.get_position(r, d, self.master_cutout)

                xmin = min(xmin, fx - x)
                xmax = max(xmax, fx - x)
                ymin = min(ymin, fy - y)
                ymax = max(ymax, fy - y)

                plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius, color=c, fill=False))
                plt.text((fx - x), (fy - y), str(i), ha='center', va='center', fontsize='x-small', color=c)

            ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
            ext = ext_base + G.Fiber_Radius

            # need a new cutout since we rescalled the ext (and window) size
            cutout = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            self.add_north_arrow(plt, sci, cutout, theta=None)

            plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

        # complete the entry
        plt.close()
        return fig

    def build_bid_target_figure(self, cat_match,ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                section_title="", bid_number=1, target_w=0, of_number=0,target_flux=None):
        '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
        photometry images, Z_PDF, etc

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 2.
        photoz_file = None

        rows = 2
        cols = len(self.Filters)

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        spec_z = 0.0

        #todo: find photo z or spec z ... right now SHELA cat does not seem to have them

        #todo: get bid target flux ... again, don't have that ... but maybe convert magnitude into flux?

        if df is not None:
            title = "%s  Possible Match #%d" % (section_title, bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

            title = title + "\nRA = %f    Dec = %f\nSeparation  = %g\"" \
                    % (df['RA'].values[0], df['DEC'].values[0],df['distance'].values[0] * 3600)
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

        plt.subplot(gs[0, 0])
        plt.text(0, 0.20, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        # for a given Tile, iterate over all filters
        tile = self.find_target_tile(ra, dec)

        if tile is None:
            # problem
            print("+++++ No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        index = -1
        for f in self.Filters:
            index += 1

            i = self.CatalogImages[
                next(i for (i, d) in enumerate(self.CatalogImages)
                     if ((d['filter'] == f) and (d['tile'] == tile)))]

            if i is None:
                continue

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.

            if cutout is not None:
                plt.subplot(gs[1, index])

                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])

                # add (+) to mark location of Target RA,DEC
                # we are centered on ra,dec and target_ra, target_dec belong to the HETDEX detect
                if cutout and (target_ra is not None) and (target_dec is not None):
                    px, py = sci.get_position(target_ra, target_dec, cutout)
                    x, y = sci.get_position(ra, dec, cutout)

                    plt.plot((px - x), (py - y), "r+")

                    self.add_north_arrow(plt, sci, cutout, theta=None)
                    plt.gca().add_patch(plt.Circle((0, 0), radius=error / 4.0, color='yellow', fill=False))

                # iterate over all filters for this image and print values
                font.set_size(10)
                if df is not None:
                    s = ""
                    for f, l in zip(i['cols'], i['labels']):
                        # print (f)
                        v = df[f].values[0]
                        s = s + "%-8s = %.5f\n" % (l, v)

                    plt.xlabel(s, multialignment='left', fontproperties=font)

        # add photo_z plot
        # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
        # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
        # get 'file'
        # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
        if df_photoz is not None:
            z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
            if z_cat is not None:
                x = z_cat['z'].values
                y = z_cat['mFDa4'].values
                plt.subplot(gs[0, 3])
                plt.plot(x, y, zorder=1)

                if spec_z > 0:
                    plt.axvline(x=spec_z, color='gold', linestyle='solid', linewidth=3, zorder=0)

                if target_w > 0:
                    la_z = target_w / G.LyA_rest - 1.0
                    oii_z = target_w / G.OII_rest - 1.0
                    plt.axvline(x=la_z, color='r', linestyle='--', zorder=2)
                    if (oii_z > 0):
                        plt.axvline(x=oii_z, color='g', linestyle='--', zorder=2)

                plt.title("Photo Z PDF")
                plt.gca().yaxis.set_visible(False)
                plt.xlabel("Z")

        # master cutout (0,0 is the observered (exact) target RA, DEC)
        if self.master_cutout is not None:
            # window=error*4
            ext = error * 2.
            plt.subplot(gs[0, cols - 1])
            # todo: rebuild pixel scale
            empty_sci = science_image.science_image()
            # need a new cutout since we rescaled the ext (and window) size
            cutout = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            vmin, vmax = empty_sci.get_vrange(cutout.data)
            plt.imshow(cutout.data, origin='lower', interpolation='none',
                       cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
            plt.title("Master Cutout (Stacked)")
            plt.xlabel("arcsecs")
            # plt.ylabel("arcsecs")

            # plt.set_xticklabels([str(ext), str(ext / 2.), str(0), str(-ext / 2.), str(-ext)])
            plt.xticks([ext, ext / 2., 0, -ext / 2., -ext])
            plt.yticks([ext, ext / 2., 0, -ext / 2., -ext])

            # mark the bid target location on the master cutout
            if (target_ra is not None) and (target_dec is not None):
                px, py = empty_sci.get_position(target_ra, target_dec, cutout)
                x, y = empty_sci.get_position(ra, dec, cutout)

                # set the diameter of the cirle to half the error (radius error/4)
                plt.gca().add_patch(
                    plt.Circle(((x - px), (y - py)), radius=error / 4.0, color='yellow', fill=False))

                # this is correct, do not rotate the yellow rectangle (it is a zoom window only)
                x = (x - px) - error
                y = (y - py) - error
                plt.gca().add_patch(plt.Rectangle((x, y), width=error * 2, height=error * 2,
                                                  angle=0.0, color='yellow', fill=False))

                plt.plot(0, 0, "r+")
                self.add_north_box(plt, empty_sci, cutout, error, 0, 0, theta=None)

        plt.close()
        return fig



    def build_cat_summary_figure (self, ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

        rows = 10
        cols = 1 + len(self.CatalogImages)/len(self.Tiles)

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
            print("+++++ No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        # All on one line now across top of plots
        title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                                % (len(bid_ras), error)

        title += "  Minimum (no match) 3$\sigma$ rest-EW: "
        if target_flux  and self.CONT_EST_BASE:
            cont_est = self.CONT_EST_BASE*3
            if cont_est != -1:
                title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A"
            else:
                title += "  LyA = N/A  OII = N/A"
        else:
            title += "  LyA = N/A  OII = N/A"


        plt.subplot(gs[0, :])
        plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        ref_exptime = None
        total_adjusted_exptime = None
        bid_colors = self.get_bid_colors(len(bid_ras))
        exptime_cont_est = -1
        index = 0 #images go in positions 1+ (0 is for the fiber positions)
        for f in self.Filters:
            index += 1

            i = self.CatalogImages[
                next(i for (i, d) in enumerate(self.CatalogImages)
                     if ((d['filter'] == f) and (d['tile'] == tile)))]

            if i is None:
                continue

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # todo: is this okay? using 'r' as rough equivalent of f606w
            #the filters are in order, use g if r is not there
            if (f == 'g') and (sci.exptime is not None) and (exptime_cont_est == -1):
                exptime_cont_est = sci.exptime

            # the filters are in order, so this will overwrite g
            if (f == 'r') and (sci.exptime is not None):
                exptime_cont_est = sci.exptime

            # sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            if cutout is not None:  # construct master cutout
                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout = sci.get_cutout(ra, dec, error, window=window, copy=True)
                    ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                    total_adjusted_exptime += sci.exptime / ref_exptime

                plt.subplot(gs[1:, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.plot(0, 0, "r+")
                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))


        if target_flux is not None:
            #todo: is this okay? using 'r' filter
            #todo: get exptime from the tile (science image has it)
            cont_est = self.get_f606w_max_cont(exptime_cont_est, 3,self.CONT_EST_BASE)
            if cont_est != -1:
                title += "Minimum (no match)\n  3$\sigma$ rest-EW:\n"
                title += "  LyA = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A\n"

        #plt.subplot(gs[0, 0])
        #plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        #plt.gca().set_frame_on(False)
        #plt.gca().axis('off')

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            return None
        else:
            self.master_cutout.data /= total_adjusted_exptime

        empty_sci = science_image.science_image()


        # plot the fiber cutout
        if (fiber_locs is not None) and (len(fiber_locs) > 0):
            plt.subplot(gs[1:, 0])

            plt.title("Fiber Positions")
            plt.xlabel("arcsecs")
            plt.gca().xaxis.labelpad = 0

            plt.plot(0, 0, "r+")

            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')

            x, y = empty_sci.get_position(ra, dec, self.master_cutout)  # zero (absolute) position

            for r, d, c, i, dist in fiber_locs:
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx, fy = empty_sci.get_position(r, d, self.master_cutout)

                xmin = min(xmin, fx - x)
                xmax = max(xmax, fx - x)
                ymin = min(ymin, fy - y)
                ymax = max(ymax, fy - y)

                plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius, color=c, fill=False))
                plt.text((fx - x), (fy - y), str(i), ha='center', va='center', fontsize='x-small', color=c)

            # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
            ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            ext = max(ext_base + G.Fiber_Radius,ext)

            # need a new cutout since we rescaled the ext (and window) size
            cutout = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)

            plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

            plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
            plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

        # complete the entry
        plt.close()
        return fig

    def build_bid_target_figure_one_line (self,cat_match, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                section_title="", bid_number=1, target_w=0, of_number=0, target_flux=None, color="k"):
        '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
        photometry images, Z_PDF, etc

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 2.
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        z_photoz_weighted = None

        rows = 1
        cols = 6

        if df_photoz is not None:
           pass

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2)

        #cols*2 to leave one column for the color coded rectange ... all other are made at 2x columns
        gs = gridspec.GridSpec(rows, cols*2, wspace=0.25, hspace=0.5)

        #use col = 0 for color coded rectangle
        plt.subplot(gs[0, 0])
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')
        #2:1 so height should be 1/2 width
        plt.gca().add_patch(plt.Rectangle((0.25,0.25), width=0.5, height=0.25, angle=0.0, color=color,
                                          fill=False,linewidth=5,zorder=1))

        #entry text
        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        if df is not None:
            title = "%s  Possible Match #%d" % (section_title, bid_number)
            if of_number > 0:
                title = title + " of %d" % of_number

            title = title + "\nRA = %f    Dec = %f\nSeparation    = %g\"" \
                            % (df['RA'].values[0], df['DEC'].values[0],df['distance'].values[0] * 3600)

            if z_best_type is not None:
                if (z_best_type.lower() == 'p'):
                    title = title + "\nPhoto Z       = %g (blue)" % z_best
                elif (z_best_type.lower() == 's'):
                    title = title + "\nSpec Z        = %g (gold)" % z_best
                    spec_z = z_best
                    if z_photoz_weighted is not None:
                        title = title + "\nPhoto Z       = %g (blue)" % z_photoz_weighted

            if target_w > 0:
                la_z = target_w / G.LyA_rest - 1.0
                oii_z = target_w / G.OII_rest - 1.0
                title = title + "\nLyA Z (virus) = %g" % la_z
                if (oii_z > 0):
                    title = title + "\nOII Z (virus) = %g" % oii_z
                else:
                    title = title + "\nOII Z (virus) = N/A"

            if (target_flux is not None) and (df['FILTER'].values[0] in 'rg'):
                #here ...
                filter_fl = df['FLUX_AUTO'].values[0]  #?? in nano-jansky or 1e-32  erg s^-1 cm^-2 Hz^-2
                if (filter_fl is not None) and (filter_fl > 0):
                    filter_fl = self.nano_jansky_to_cgs(filter_fl,target_w) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                    title = title + "\nEst LyA rest-EW = %g $\AA$" % \
                                    (target_flux / filter_fl / (target_w / G.LyA_rest))

                    if target_w >= G.OII_rest:
                        title = title + "\nEst OII rest-EW = %g $\AA$" % \
                                        (target_flux / filter_fl / (target_w / G.OII_rest))
                    else:
                        title = title + "\nEst OII rest-EW = N/A"

                    # bid target info is only of value if we have a flux from the emission line
                    bid_target = match_summary.BidTarget()
                    bid_target.bid_ra = df['RA'].values[0]
                    bid_target.bid_dec = df['DEC'].values[0]
                    bid_target.distance = df['distance'].values[0] * 3600
                    bid_target.bid_flux_est_cgs = filter_fl
                    bid_target.bid_flux_f606w_cgs = filter_fl
                    #todo: build these filters
                    #bid_target.bid_flux_f814w_cgs = self.nano_jansky_to_cgs(df['F814W (I) flux (nJy)'].values[0],
                    #                                                        target_w)
                    #bid_target.bid_flux_f125w_cgs = self.nano_jansky_to_cgs(df['F125W (J) flux (nJy)'].values[0],
                    #                                                        target_w)
                    #bid_target.bid_flux_f160w_cgs = self.nano_jansky_to_cgs(df['F160W (H) flux (nJy)'].values[0],
                    #                                                        target_w)
                    cat_match.add_bid_target(bid_target)
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

        plt.subplot(gs[0, 1:4])
        plt.text(0, 0, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        #add flux values
        if df is not None:
            # iterate over all filter images
            title = ""
            for i in self.CatalogImages:  # i is a dictionary
                # iterate over all filters for this image and print values
                #font.set_size(10)

                #not all filters have entries ... note 'cols'[0] is flux, [1] is the error
                if df[i['cols']].empty :
                    title = title + "%7s %s %s = -- (--)\n" % (i['instrument'], i['filter'], "Flux")
                else:
                    try:
                        title = title + "%7s %s %s = %.5f (%.5f)\n" % (i['instrument'], i['filter'], "Flux",
                                float(df[i['cols'][0]].values[0]), float(df[i['cols'][1]].values[0]))
                    except:
                        title = title + "%7s %s %s = -- (--)\n" % (i['instrument'], i['filter'], "Flux")


        plt.subplot(gs[0, 4])
        plt.text(0, 0, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        # todo: photo z plot if becomes available
        # add photo_z plot
        if df_photoz is not None:
            pass
        else:
            plt.subplot(gs[0, -4:])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            text = "Photo Z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        # fig holds the entire page
        plt.close()
        return fig

    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None):

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


        bid_colors = self.get_bid_colors(len(ras))

        text = "Separation\n" + \
               "RA, Dec\n" + \
               "Spec Z\n" + \
               "Photo Z\n" + \
               "Est LyA rest-EW\n" + \
               "Est OII rest-EW\n" #+ \
               #"ACS WFC f606W Flux\n"

        plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)

        col_idx = 0
        for r, d in zip(ras, decs):
            col_idx += 1
            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'r')]
                if df is None:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                if df is None:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                text = ""
                text = text + "%g\"\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600,df['RA'].values[0], df['DEC'].values[0])

                best_fit_photo_z = 0.0
                try:
                    best_fit_photo_z = float(df['Best-fit photo-z'].values[0])
                except:
                    pass

                text += "N/A\n" + "%g\n" % best_fit_photo_z

                filter_fl = 0.0
                filter_fl_err = 0.0

                #todo: add flux (cont est)
                try:
                    if (df['FILTER'].values[0] in 'rg'):
                      filter_fl = float(df['FLUX_AUTO'].values[0])  #?? in nano-jansky or 1e-32  erg s^-1 cm^-2 Hz^-2
                      filter_fl_err = float(df['FLUXERR_AUTO'].values[0])
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_adj = self.nano_jansky_to_cgs(filter_fl,target_w) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        text = text + "%g $\AA$\n" % (target_flux / filter_fl_adj / (target_w / G.LyA_rest))

                        if target_w >= G.OII_rest:
                            text = text + "%g $\AA$\n" % (target_flux / filter_fl_adj / (target_w / G.OII_rest))
                        else:
                            text = text + "N/A\n"
                            # bid target info is only of value if we have a flux from the emission line
                        bid_target = match_summary.BidTarget()
                        bid_target.bid_ra = df['RA'].values[0]
                        bid_target.bid_dec = df['DEC'].values[0]
                        bid_target.distance = df['distance'].values[0] * 3600
                        bid_target.bid_flux_est_cgs = filter_fl

                        #todo: build these filters
                        #bid_target.bid_flux_f606w_cgs = filter_fl
                        #bid_target.bid_flux_f814w_cgs = self.nano_jansky_to_cgs(df['F814W (I) flux (nJy)'].values[0],
                        #    target_w)
                        #bid_target.bid_flux_f125w_cgs = self.nano_jansky_to_cgs(df['F125W (J) flux (nJy)'].values[0],
                        #    target_w)
                        #bid_target.bid_flux_f160w_cgs = self.nano_jansky_to_cgs(df['F160W (H) flux (nJy)'].values[0],
                        #    target_w)
                        cat_match.add_bid_target(bid_target)
                else:
                    text += "N/A\nN/A\n"

            #todo: add flux (cont est)
               # if filter_fl < 0:
               #     text = text + "%g(%g) nJy !?\n" % (filter_fl, filter_fl_err)
               # else:
               #     text = text + "%g(%g) nJy\n" % (filter_fl, filter_fl_err)
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
            text = "Photo Z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        plt.close()
        return fig
