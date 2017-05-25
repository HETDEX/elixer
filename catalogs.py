from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import global_config as G
import os.path as op


CANDELS_EGS_Stefanon_2016_BASE_PATH = G.CANDELS_EGS_Stefanon_2016_BASE_PATH
CANDELS_EGS_Stefanon_2016_CAT = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH,
                                        "photometry/CANDELS.EGS.F160W.v1_1.photom.cat")
CANDELS_EGS_Stefanon_2016_IMAGES_PATH = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH, "images")
CANDELS_EGS_Stefanon_2016_PHOTOZ_CAT = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH , "photoz/zcat_EGS_v2.0.cat")
CANDELS_EGS_Stefanon_2016_PHOTOZ_ZPDF_PATH = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH, "photoz/zPDF/")


EGS_GROTH_BASE_PATH = G.EGS_GROTH_BASE_PATH
EGS_GROTH_CAT_PATH = G.EGS_GROTH_CAT_PATH
#EGS_GROTH_IMAGE = op.join(G.EGS_GROTH_BASE_PATH,"groth.fits")

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
import astropy.io.fits as fits
from astropy.table import Table
#from astropy.io import ascii #note: this works, but pandas is much faster


log = G.logging.getLogger('Cat_logger')
log.setLevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:


def get_catalog_list():
    #build list of all catalogs below
    cats = list()
    cats.append(CANDELS_EGS_Stefanon_2016())
    cats.append(EGS_GROTH())
    cats.append(STACK_COSMOS())

  #  cats.append(CANDELS_EGS_Stefanon_2016())
  #  cats[1].Name = "Duplicate CANDELS"
    return cats


#todo: future ... see if can reorganize and use this as a wrapper and maintain only one Figure per report
class Page:
    def __init__(self,num_entries):
        self.num_entries = num_entries
        self.rows_per_entry = 0
        self.cols_per_entry = 0
        self.grid = None
        self.current_entry = 0
        self.current_row = 0
        self.current_col = 0
        self.figure = None


    @property
    def col(self):
        return self.current_col
    @property
    def row(self):
        return self.current_row

    @property
    def entry(self):
        return self.current_entry

    @property
    def gs(self):
        return self.grid

    @property
    def fig(self):
        return self.figure

    def build_grid(self,rows,cols):
        self.grid = gridspec.GridSpec(self.num_entries * rows, cols, wspace=0.25, hspace=0.5)
        self.rows_per_entry = rows
        self.cols_per_entry = cols
        self.figure = plt.figure(figsize=(self.num_entries*rows*3,cols*3))

    def next_col(self):
        if (self.current_col == self.cols_per_entry):
            self.current_col = 0
            self.current_row += 1
        else:
            self.current_col += 1

    def next_row(self):
        self.current_row += 1

    def next_entry(self):
        self.current_entry +=1
        self.current_row +=1
        self.current_col = 0

__metaclass__ = type
class Catalog:
    MainCatalog = None
    Name = "Generic Catalog (Base)"
    df = None  # pandas dataframe ... all instances share the same frame
    df_photoz = None
    #tbl = None # astropy.io.table
    RA_min = None
    RA_max = None
    Dec_min = None
    Dec_max = None
    status = -1

    def __init__(self):
        self.pages = None #list of bid entries (rows in the pdf)
        self.dataframe_of_bid_targets = None

    @property
    def ok(self):
        return (self.status == 0)

    @property
    def name(self):
        return (self.Name)

    @classmethod
    def position_in_cat(cls, ra, dec, error = 0.0):  # error assumed to be small and this is approximate anyway
        """Simple check for ra and dec within a rectangle defined by the min/max of RA,DEC for the catalog.
        RA and Dec in decimal degrees.
        """
        if cls.ok:
            try:
                result = (ra >= (cls.RA_min - error)) and (ra <= (cls.RA_max + error))\
                         and (dec >= (cls.Dec_min - error)) and (dec <= (cls.Dec_max + error))
            except:
                result = False
        else:
            result = False
        return result

    @classmethod
    def coordinate_range(cls,echo=False):
        if echo:
            msg = "RA (%f, %f)" % (cls.RA_min, cls.RA_max) + "Dec(%f, %f)" % (cls.Dec_min, cls.Dec_max)
            print( msg )
        log.debug(cls.Name + " Simple Coordinate Box: " + msg )
        return (cls.RA_min, cls.RA_max, cls.Dec_min, cls.Dec_max)

    @classmethod
    def get_dict(cls,id,cols):
        """returns a (nested) dictionary of desired cols for a single row from the full dataframe
            form {col_name : {id : value}}
        """
        try:
            bid_dict = cls.df.loc[id,cols].to_dict()
        except:
            log.error("Exception attempting to build dictionary for id %d" %id)
            return None
        return bid_dict

    @classmethod
    def read_main_catalog(cls):
        if cls.df is not None:
            log.debug("Already built df")
        elif cls.MainCatalog is not None:
            try:
                print("Reading main catalog for ", cls.Name)
                cls.df = cls.read_catalog(cls.MainCatalog, cls.Name)
                cls.status = 0
                cls.RA_min = cls.df['RA'].min()
                cls.RA_max = cls.df['RA'].max()
                cls.Dec_min = cls.df['DEC'].min()
                cls.Dec_max = cls.df['DEC'].max()

                log.debug(cls.Name + " Coordinate Range: RA: %f to %f , Dec: %f to %f" % (cls.RA_min, cls.RA_max,
                                                                                          cls.Dec_min, cls.Dec_max))
            except:
                print("Failed")
                cls.status = -1
                log.error("Exception in read_main_catalog for %s" % cls.Name, exc_info=True)

        else:
            print("No catalog defined for ", cls.Name)

        if cls.df is None:
            cls.status = -1
        return

    def sort_bid_targets_by_likelihood(self,ra,dec):
        #right now, just by euclidean distance (ra,dec are of target)
        #todo: if radial separation is greater than the error, remove this bid target?
        #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)
        self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra)**2 +
                                                            (self.dataframe_of_bid_targets['DEC'] - dec)**2)
        self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)


    #must be defined in child class
    def build_bid_target_reports(self, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None):
        return None

    def clear_pages(self):
        if self.pages is None:
            self.pages = []
        elif len(self.pages) > 0:
            del self.pages[:]

    def add_bid_entry(self, entry):
        if self.pages is None:
            self.clear_pages()
        self.pages.append(entry)


#specific implementation of The CANDELS-EGS Multi-wavelength catalog Stefanon et al., 2016
#CandlesEgsStefanon2016

class CANDELS_EGS_Stefanon_2016(Catalog):
#RA,Dec in decimal degrees

#photometry catalog
#  1 ID #  2 IAU_designation  #  3 RA  #  4 DEC #  5 RA_Lotz2008 (RA in AEGIS ACS astrometric system)  #  6 DEC_Lotz2008 (DEC in AEGIS ACS astrometric system)
#  7 FLAGS #  8 CLASS_STAR #  9 CFHT_U_FLUX  # 10 CFHT_U_FLUXERR # 11 CFHT_g_FLUX # 12 CFHT_g_FLUXERR # 13 CFHT_r_FLUX
# 14 CFHT_r_FLUXERR  # 15 CFHT_i_FLUX # 16 CFHT_i_FLUXERR # 17 CFHT_z_FLUX # 18 CFHT_z_FLUXERR # 19 ACS_F606W_FLUX
# 20 ACS_F606W_FLUXERR # 21 ACS_F814W_FLUX # 22 ACS_F814W_FLUXERR # 23 WFC3_F125W_FLUX # 24 WFC3_F125W_FLUXERR
# 25 WFC3_F140W_FLUX # 26 WFC3_F140W_FLUXERR # 27 WFC3_F160W_FLUX # 28 WFC3_F160W_FLUXERR # 29 WIRCAM_J_FLUX
# 30 WIRCAM_J_FLUXERR  # 31 WIRCAM_H_FLUX # 32 WIRCAM_H_FLUXERR # 33 WIRCAM_K_FLUX # 34 WIRCAM_K_FLUXERR
# 35 NEWFIRM_J1_FLUX # 36 NEWFIRM_J1_FLUXERR # 37 NEWFIRM_J2_FLUX # 38 NEWFIRM_J2_FLUXERR # 39 NEWFIRM_J3_FLUX # 40 NEWFIRM_J3_FLUXERR
# 41 NEWFIRM_H1_FLUX # 42 NEWFIRM_H1_FLUXERR # 43 NEWFIRM_H2_FLUX # 44 NEWFIRM_H2_FLUXERR # 45 NEWFIRM_K_FLUX# 46 NEWFIRM_K_FLUXERR
# 47 IRAC_CH1_FLUX # 48 IRAC_CH1_FLUXERR # 49 IRAC_CH2_FLUX # 50 IRAC_CH2_FLUXERR # 51 IRAC_CH3_FLUX # 52 IRAC_CH3_FLUXERR # 53 IRAC_CH4_FLUX
# 54 IRAC_CH4_FLUXERR # 55 ACS_F606W_V08_FLUX # 56 ACS_F606W_V08_FLUXERR # 57 ACS_F814W_V08_FLUX # 58 ACS_F814W_V08_FLUXERR
# 59 WFC3_F125W_V08_FLUX # 60 WFC3_F125W_V08_FLUXERR # 61 WFC3_F160W_V08_FLUX # 62 WFC3_F160W_V08_FLUXERR
# 63 IRAC_CH3_V08_FLUX # 64 IRAC_CH3_V08_FLUXERR # 65 IRAC_CH4_V08_FLUX # 66 IRAC_CH4_V08_FLUXERR # 67 DEEP_SPEC_Z

    #class variables
    MainCatalog = CANDELS_EGS_Stefanon_2016_CAT
    Name = "CANDELS_EGS_Stefanon_2016"
    WCS_Manual = True
    BidCols = ["ID","IAU_designation","RA","DEC",
               "CFHT_U_FLUX","CFHT_U_FLUXERR",
               "IRAC_CH1_FLUX","IRAC_CH1_FLUXERR","IRAC_CH2_FLUX","IRAC_CH2_FLUXERR",
               "ACS_F606W_FLUX","ACS_F606W_FLUXERR",
               "ACS_F814W_FLUX","ACS_F814W_FLUXERR",
               "WFC3_F125W_FLUX","WFC3_F125W_FLUXERR",
               "WFC3_F140W_FLUX","WFC3_F140W_FLUXERR",
               "WC3_F160W_FLUX","WFC3_F160W_FLUXERR",
               "DEEP_SPEC_Z"]  #NOTE: there are no F105W values

    CatalogImages = [
                {'path':CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
                 'name':'egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits',
                 'filter':'f606w',
                 'instrument':'ACS WFC',
                 'cols':["ACS_F606W_FLUX","ACS_F606W_FLUXERR"],
                 'labels':["Flux","Err"],
                 'image':None
                },
                {'path':CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
                 'name':'egs_all_acs_wfc_f814w_060mas_v1.1_drz.fits',
                 'filter':'f814w',
                 'instrument':'ACS WFC',
                 'cols':["ACS_F814W_FLUX","ACS_F814W_FLUXERR"],
                 'labels':["Flux","Err"],
                 'image':None
                },
                {'path':CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
                 'name':'egs_all_wfc3_ir_f105w_060mas_v1.5_drz.fits',
                 'filter':'f105w',
                 'instrument':'WFC3',
                 'cols':[],
                 'labels':[],
                 'image':None
                },
                {'path':CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
                 'name':'egs_all_wfc3_ir_f125w_060mas_v1.1_drz.fits',
                 'filter':'f125w',
                 'instrument':'WFC3',
                 'cols':["WFC3_F125W_FLUX","WFC3_F125W_FLUXERR"],
                 'labels':["Flux","Err"],
                 'image':None
                },
                {'path':CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
                 'name':'egs_all_wfc3_ir_f140w_060mas_v1.1_drz.fits',
                 'filter':'f140w',
                 'instrument':'WFC3',
                 'cols':["WFC3_F140W_FLUX","WFC3_F140W_FLUXERR"],
                 'labels':["Flux","Err"],
                 'image':None
                },
                {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
                 'name': 'egs_all_wfc3_ir_f160w_060mas_v1.1_drz.fits',
                 'filter': 'f160w',
                 'instrument': 'WFC3',
                 'cols':["WFC3_F160W_FLUX","WFC3_F160W_FLUXERR"],
                 'labels':["Flux","Err"],
                 'image':None
                }
               ]

# 1 file # 2 ID (CANDELS.EGS.F160W.v1b_1.photom.cat) # 3 RA (CANDELS.EGS.F160W.v1b_1.photom.cat) # 4 DEC (CANDELS.EGS.F160W.v1b_1.photom.cat)
# 5 z_best # 6 z_best_type # 7 z_spec # 8 z_spec_ref # 9 z_grism # 10 mFDa4_z_peak # 11 mFDa4_z_weight # 12 mFDa4_z683_low
# 13 mFDa4_z683_high # 14 mFDa4_z954_low # 15 mFDa4_z954_high # 16 HB4_z_peak # 17 HB4_z_weight # 18 HB4_z683_low
# 19 HB4_z683_high # 20 HB4_z954_low # 21 HB4_z954_high # 22 Finkelstein_z_peak # 23 Finkelstein_z_weight
# 24 Finkelstein_z683_low # 25 Finkelstein_z683_high # 26 Finkelstein_z954_low # 27 Finkelstein_z954_high
# 28 Fontana_z_peak # 29 Fontana_z_weight # 30 Fontana_z683_low # 31 Fontana_z683_high # 32 Fontana_z954_low
# 33 Fontana_z954_high # 34 Pforr_z_peak # 35 Pforr_z_weight # 36 Pforr_z683_low # 37 Pforr_z683_high
# 38 Pforr_z954_low # 39 Pforr_z954_high # 40 Salvato_z_peak # 41 Salvato_z_weight # 42 Salvato_z683_low
# 43 Salvato_z683_high # 44 Salvato_z954_low # 45 Salvato_z954_high # 46 Wiklind_z_peak # 47 Wiklind_z_weight
# 48 Wiklind_z683_low  # 49 Wiklind_z683_high # 50 Wiklind_z954_low # 51 Wiklind_z954_high # 52 Wuyts_z_peak
# 53 Wuyts_z_weight  # 54 Wuyts_z683_low # 55 Wuyts_z683_high # 56 Wuyts_z954_low # 57 Wuyts_z954_high

    PhotoZCatalog = CANDELS_EGS_Stefanon_2016_PHOTOZ_CAT
    SupportFilesLocation = CANDELS_EGS_Stefanon_2016_PHOTOZ_ZPDF_PATH

    def __init__(self):
        super(CANDELS_EGS_Stefanon_2016, self).__init__()

        #self.dataframe_of_bid_targets = None #defined in base class
        self.dataframe_of_bid_targets_photoz = None
        #self.table_of_bid_targets = None
        self.num_targets = 0

        self.read_main_catalog()
        self.read_photoz_catalog()
        #self.build_catalog_images() #will just build on demand

        self.master_cutout= None


    #todo: is this more efficient? garbage collection does not seem to be running
    #so building as needed does not seem to help memory
    def build_catalog_images(self):
        for i in self.CatalogImages:  # i is a dictionary
            i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual, image_location=op.join(i['path'],i['name']))


    @classmethod
    def read_main_catalog(cls):
        if cls.df is not None:
            log.debug("Already built df")
        else:
            try:
                print("Reading main catalog for ", cls.Name)
                cls.df = cls.read_catalog(cls.MainCatalog,cls.Name)
                cls.status = 0
                cls.RA_min = cls.df['RA'].min()
                cls.RA_max = cls.df['RA'].max()
                cls.Dec_min = cls.df['DEC'].min()
                cls.Dec_max = cls.df['DEC'].max()

                log.debug(cls.Name + " Coordinate Range: RA: %f to %f , Dec: %f to %f" % (cls.RA_min, cls.RA_max,
                                                                                          cls.Dec_min, cls.Dec_max))
            except:
                print("Failed")
                cls.status = -1

            if cls.df is None:
                cls.status = -1
        return

    @classmethod
    def read_photoz_catalog(cls):
        if cls.df_photoz is not None:
            log.debug("Already built df_photoz")
        else:
            try:
                print("Reading photoz catalog for ", cls.Name)
                cls.df_photoz = cls.read_catalog(cls.PhotoZCatalog,cls.Name)
            except:
                print("Failed")

        return


    @classmethod
    def read_catalog(cls,catalog_loc,name):

        log.debug("Building " + name + " dataframe...")
        idx = []
        header = []
        skip = 0
        try:
            f = open(catalog_loc, mode='r')
        except:
            log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
            return None


        line = f.readline()
        while '#' in line:
            skip += 1
            toks = line.split()
            if (len(toks) > 2) and toks[1].isdigit():   #format:   # <id number> <column name>
                idx.append(toks[1])
                header.append(toks[2])
            line = f.readline()

        f.close()

        try:
            df = pd.read_csv(catalog_loc, names=header,
                delim_whitespace=True, header=None, index_col=None, skiprows=skip)
        except:
            log.error(name + " Exception attempting to build pandas dataframe",exc_info=True)
            return None

        return df


    def build_list_of_bid_targets(self,ra,dec,error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        error_in_deg = float(error) / 3600.0

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0

        ra_min = float(ra - error_in_deg)
        ra_max = float(ra + error_in_deg)
        dec_min = float(dec - error_in_deg)
        dec_max = float(dec + error_in_deg)

        log.info(self.Name + " searching for bid targets in range: RA [%f +/- %f], Dec [%f +/- %f] ..."
                % (ra,error_in_deg,dec,error_in_deg))

        try:
            self.dataframe_of_bid_targets = self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                                                (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)]

            #ID matches between both catalogs
            self.dataframe_of_bid_targets_photoz = \
                self.df_photoz[(self.df_photoz['ID'].isin(self.dataframe_of_bid_targets['ID']))]
        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets",exc_info=True)

        if self.dataframe_of_bid_targets is not None:
            self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra,dec)

            log.info (self.Name + " searching for objects in [%f - %f, %f - %f] " %(ra_min,ra_max,dec_min,dec_max) +
                  ". Found = %d" % (self.num_targets ))

        return self.num_targets, self.dataframe_of_bid_targets, self.dataframe_of_bid_targets_photoz


    #column names are catalog specific, but could map catalog specific names to generic ones and produce a dictionary?
    def build_bid_target_reports(self,target_ra, target_dec, error, num_hits=0,section_title="",base_count=0,
                                 target_w=0,fiber_locs=None):

        self.clear_pages()
        self.build_list_of_bid_targets(target_ra,target_dec,error)

        ras = self.dataframe_of_bid_targets.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets.loc[:, ['DEC']].values

        #display the exact (target) location
        entry = self.build_exact_target_location_figure(target_ra,target_dec,error,section_title=section_title,
                                                        target_w=target_w,fiber_locs=fiber_locs)

        if entry is not None:
            self.add_bid_entry(entry)

        number = 0
        #display each bid target
        for r,d in zip(ras,decs):
            number+=1
            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                   (self.dataframe_of_bid_targets['DEC'] == d[0])]

                idnum = df['ID'].values[0] #to matchup in photoz catalog
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue #this must be here, so skip to next ra,dec

            try:
                #note cannot dirctly use RA,DEC as the recorded precission is different (could do a rounded match)
                #but the idnums match up, so just use that
                df_photoz = self.dataframe_of_bid_targets_photoz.loc[self.dataframe_of_bid_targets_photoz['ID'] == idnum ]

                if len(df_photoz) == 0:
                    log.debug("No conterpart found in photoz catalog; RA=%f , Dec =%f" %(r[0],d[0] ))
                    df_photoz = None
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets",exc_info=True)
                df_photoz = None

            print("Building report for bid target %d in %s" % (base_count + number,self.Name))
            entry = self.build_bid_target_figure(r[0],d[0],error=error,df=df,df_photoz=df_photoz,
                                                 target_ra=target_ra,target_dec=target_dec,section_title=section_title,
                                                 bid_number=number,target_w=target_w)
            if entry is not None:
                self.add_bid_entry(entry)

        return self.pages



    def build_exact_target_location_figure(self, ra, dec, error,section_title="",target_w=0,fiber_locs=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...
        
        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        #not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error*4

        #set a minimum window size?
        #if window < 8:
        #    window = 8

        rows = 2
        cols = len(self.CatalogImages)

        fig_sz_x = cols * 3
        fig_sz_y = rows * 3

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)
        #reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        title = "Catalog: %s\n" %self.Name + section_title + "\nTarget Location\nPossible Matches=%d (within %g\")\n" \
                "RA = %f    Dec = %f\n" % (len(self.dataframe_of_bid_targets),error,ra, dec)
        if target_w > 0:
            title = title + "Wavelength = %g $\AA$\n" % target_w
        else:
            title = title + "\n"
        plt.subplot(gs[0,0])
        plt.text(0, 0.3, title,ha='left',va='bottom',fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')



        if self.master_cutout is not None:
            del(self.master_cutout)
            self.master_cutout = None


        index = -1
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2. #extent is from the 0,0 center, so window/2

            if cutout is not None: #construct master cutout
                #master cutout needs a copy of the data since it is going to be modified  (stacked)
                #repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout = sci.get_cutout(ra, dec, error, window=window,copy=True)
                else:
                    self.master_cutout.data = np.add(self.master_cutout.data, cutout.data)

                plt.subplot(gs[rows-1, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])

                plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                                  angle=0.0, color='red', fill=False))


        if self.master_cutout is None:
            #cannot continue
            print("No catalog image available in %s" %self.Name)
            return None


        #plot the master cutout
        empty_sci = science_image.science_image()
        plt.subplot( gs[0, cols-1])
        vmin,vmax = empty_sci.get_vrange(self.master_cutout.data)
        plt.imshow(self.master_cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                   vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
        plt.title("Master Cutout (Stacked)")
        plt.xlabel("arcsecs")
        # only show this lable if there is not going to be an adjacent fiber plot
        if (fiber_locs is None) or (len(fiber_locs) == 0):
            plt.ylabel("arcsecs")
        plt.plot(0,0, "r+")
        plt.gca().add_patch(plt.Rectangle( (-error,-error), width=error*2, height=error*2,
                angle=0.0, color='red', fill=False ))


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

            x, y = empty_sci.get_position(ra, dec, self.master_cutout) #zero (absolute) position

            plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                              angle=0.0, color='red', fill=False))

            for r,d,c,i in fiber_locs:
                #print("+++++ Cutout RA,DEC,ID,COLOR", r,d,i,c)
                #fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx, fy = empty_sci.get_position(r, d, self.master_cutout)

                xmin = min(xmin,fx-x)
                xmax = max(xmax,fx-x)
                ymin = min(ymin,fy-y)
                ymax = max(ymax,fy-y)

                plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius, color=c, fill=False))
                plt.text((fx - x), (fy - y), str(i), ha='center', va='center', fontsize='x-small', color=c)


            ext = max(abs(-xmin-2*G.Fiber_Radius),abs(xmax+2*G.Fiber_Radius),
                      abs(-ymin-2*G.Fiber_Radius),abs(ymax+2*G.Fiber_Radius),2.0)

            #need a new cutout since we rescalled the ext (and window) size
            cutout = empty_sci.get_cutout(ra, dec, error, window=ext*2,image=self.master_cutout)
            vmin, vmax = empty_sci.get_vrange(cutout.data)

            plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                   vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

        # complete the entry
        plt.close()
        return fig



    def build_bid_target_figure(self,ra,dec,error,df=None,df_photoz=None,target_ra=None,target_dec=None,
                                section_title="",bid_number = 1,target_w = 0):
        '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
        photometry images, Z_PDF, etc
        
        Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
        window = error * 2.
        photoz_file = None
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        #z_spec = None
        #z_spec_ref = None

        rows = 2
        cols = len(self.CatalogImages)

        if df_photoz is not None:
            photoz_file = df_photoz['file'].values[0]
            z_best = df_photoz['z_best'].values[0]
            z_best_type = df_photoz['z_best_type'].values[0] #s = spectral , p = photometric?
            #z_spec = df_photoz['z_spec'].values[0]
            #z_spec_ref = df_photoz['z_spec_ref'].values[0]
            #rows = rows + 1

        fig_sz_x = cols*3
        fig_sz_y = rows*3

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        fig = plt.figure(figsize=(fig_sz_x,fig_sz_y))

        if df is not None:
            title = "%s\nPossible Match #%d\n%s\n\nRA = %f    Dec = %f\nSeparation = %g\"" \
                    % (section_title, bid_number,df['IAU_designation'].values[0], df['RA'].values[0], df['DEC'].values[0],
                       df['distance'].values[0] * 3600)
            z = df['DEEP_SPEC_Z'].values[0]
            if z >= 0.0:
                title = title + "\nDEEP SPEC Z = %g" % z

            if z_best_type is not None:
                if (z_best_type.lower() == 'p'):
                    title = title + "\nPhoto Z     = %g (blue)" % z_best
                elif (z_best_type.lower() == 's'):
                    title = title + "\nSpec Z      = %g (blue)" % z_best
            if target_w > 0:
                la_z = target_w / G.LyA_rest - 1.0
                oii_z = target_w / G.OII_rest - 1.0
                title = title + "\nLyA Z       = %g (red)" % la_z
                if (oii_z > 0):
                    title = title + "\nOII Z       = %g (green)" % oii_z
                else:
                    title = title + "\nOII Z       = N/A"
        else:
            title = "%s\nRA=%f    Dec=%f" % (section_title,ra, dec)

        plt.subplot(gs[0, 0])
        plt.text(0,0.20,title,ha='left',va='bottom',fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        index = -1
        #iterate over all filter images
        for i in self.CatalogImages: # i is a dictionary
            index+= 1 #for subplot ... is 1 based
            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            cutout = sci.get_cutout(ra, dec, error, window=window)
            ext = sci.window / 2.

            if cutout is not None:
                plt.subplot(gs[1, index])

                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent= [-ext,ext,-ext,ext])
                plt.title(i['instrument']+" "+i['filter'])

                #add (+) to mark location of Target RA,DEC
                #we are centered on ra,dec and target_ra, target_dec belong to the HETDEX detect
                if cutout and (target_ra is not None) and (target_dec is not None):
                    px, py = sci.get_position(target_ra, target_dec, cutout)
                    x,y = sci.get_position(ra, dec, cutout)

                    plt.plot((px-x),(py-y),"r+")

                    plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2., height=error * 2.,
                                                  angle=0.0, color='yellow', fill=False,linewidth=5.0,zorder=1))
                    # set the diameter of the cirle to half the error (radius error/4)
                    plt.gca().add_patch(plt.Circle((0,0), radius=error / 4.0, color='yellow', fill=False))

                #iterate over all filters for this image and print values
                font.set_size(10)
                if df is not None:
                    s = ""
                    for f,l in zip(i['cols'],i['labels']):
                        #print (f)
                        v = df[f].values[0]
                        s = s + "%-8s = %.5f\n" %(l,v)

                    plt.xlabel(s,multialignment='left',fontproperties=font)


        #add photo_z plot
        # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
        # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
        #get 'file'
        # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
        if df_photoz is not None:
            z_cat = self.read_catalog(op.join(self.SupportFilesLocation,photoz_file),"z_cat")
            if z_cat is not None:
                x = z_cat['z'].values
                y = z_cat['mFDa4'].values
                plt.subplot(gs[0, 3])
                plt.plot(x,y)
                if target_w > 0:
                    la_z = target_w / G.LyA_rest - 1.0
                    oii_z = target_w / G.OII_rest - 1.0
                    plt.axvline(x=la_z,color='r', linestyle='--')
                    if (oii_z > 0):
                        plt.axvline(x=oii_z, color='g', linestyle='--')
                plt.title("Photo Z PDF")
                plt.gca().yaxis.set_visible(False)
                plt.xlabel("Z")

        empty_sci = science_image.science_image()
        #master cutout (0,0 is the observered (exact) target RA, DEC)
        if self.master_cutout is not None:
            #window=error*4
            ext = error*2.
            plt.subplot(gs[0, cols - 1])
            vmin, vmax = empty_sci.get_vrange(self.master_cutout.data)
            plt.imshow(self.master_cutout.data, origin='lower', interpolation='none',
                       cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
            plt.title("Master Cutout (Stacked)")
            plt.xlabel("arcsecs")
            plt.ylabel("arcsecs")

            #mark the bid target location on the master cutout
            if  (target_ra is not None) and (target_dec is not None):
                px, py = empty_sci.get_position(target_ra, target_dec, self.master_cutout)
                x, y   = empty_sci.get_position(ra, dec, self.master_cutout)
                plt.plot(0, 0, "r+")

                #set the diameter of the cirle to half the error (radius error/4)
                plt.gca().add_patch(plt.Circle(((x-px), (y-py)), radius=error/4.0, color='yellow', fill=False))
                plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                                  angle=0.0, color='red', fill=False))
                x = (x-px)  - error
                y = (y-py)  - error
                plt.gca().add_patch(plt.Rectangle((x, y), width=error * 2, height=error * 2,
                                                  angle=0.0, color='yellow', fill=False))

        plt.close()
        return fig

#######################################
#end class CANDELS_EGS_Stefanon_2016
#######################################


class EGS_GROTH(Catalog):
    #class variables
    EGS_GROTH_BASE_PATH = G.EGS_GROTH_BASE_PATH
    EGS_GROTH_CAT = None #op.join(G.EGS_GROTH_CAT_PATH, "")
    EGS_GROTH_IMAGE_PATH = G.EGS_GROTH_BASE_PATH
    EGS_GROTH_IMAGE = op.join(G.EGS_GROTH_BASE_PATH,"groth.fits")

    #there is no catalog??
    MainCatalog = EGS_GROTH_CAT
    Name = "EGS_GROTH"
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
         'image': None
         }]


    def __init__(self):
        super(EGS_GROTH, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

        self.read_main_catalog()

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




class STACK_COSMOS(Catalog):

        # name = 'NUMBER'; format = '1J'; disp = 'I10'
        # name = 'FLUXERR_ISO'; format = '1E'; unit = 'count'; disp = 'G12.7'
        # name = 'MAG_APER'; format = '25E'; unit = 'mag'; disp = 'F8.4'
        # name = 'MAGERR_APER'; format = '25E'; unit = 'mag'; disp = 'F8.4'
        # name = 'FLUX_APER'; format = '25E'; unit = 'count'; disp = 'G12.7'
        # name = 'FLUXERR_APER'; format = '25E'; unit = 'count'; disp = 'G12.7'
        # name = 'FLUX_AUTO'; format = '1E'; unit = 'count'; disp = 'G12.7'
        # name = 'FLUXERR_AUTO'; format = '1E'; unit = 'count'; disp = 'G12.7'
        # name = 'MAG_AUTO'; format = '1E'; unit = 'mag'; disp = 'F8.4'
        # name = 'MAGERR_AUTO'; format = '1E'; unit = 'mag'; disp = 'F8.4'
        # name = 'THRESHOLD'; format = '1E'; unit = 'count'; disp = 'G12.7'
        # name = 'X_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F11.4'
        # name = 'Y_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F11.4'
        # name = 'ALPHA_J2000'; format = '1D'; unit = 'deg'; disp = 'F11.7'
        # name = 'DELTA_J2000'; format = '1D'; unit = 'deg'; disp = 'F11.7'
        # name = 'A_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
        # name = 'B_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
        # name = 'FLUX_RADIUS'; format = '1E'; unit = 'pixel'; disp = 'F10.3'
        # name = 'THETA_IMAGE'; format = '1E'; unit = 'deg'; disp = 'F6.2'
        # name = 'FWHM_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F8.2'
        # name = 'FWHM_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
        # name = 'FLAGS'; format = '1I'; disp = 'I3'
        # name = 'IMAFLAGS_ISO'; format = '1J'; disp = 'I9'
        # name = 'NIMAFLAGS_ISO'; format = '1J'; disp = 'I9'
        # name = 'CLASS_STAR'; format = '1E'; disp = 'F6.3'


        # class variables
        STACK_COSMOS_BASE_PATH = G.STACK_COSMOS_BASE_PATH
        STACK_COSMOS_CAT = op.join(G.STACK_COSMOS_CAT_PATH, "cat_g.fits")
        STACK_COSMOS_IMAGE_PATH = G.STACK_COSMOS_BASE_PATH
        STACK_COSMOS_IMAGE = op.join(STACK_COSMOS_IMAGE_PATH, "COSMOS_g_sci.fits")

        MainCatalog = STACK_COSMOS_CAT
        Name = "STACK_COSMOS"
        WCS_Manual = False

        AstroTable = None

        BidCols = ['NUMBER',  # int32
                   'FLUXERR_ISO',  # ct float32
                   'MAG_APER',  # [25] mag float32
                   'MAGERR_APER',  # [25] mag float32
                   'FLUX_APER',  # [25] ct float32
                   'FLUXERR_APER',  # [25] ct float32
                   'FLUX_AUTO',  # ct float32
                   'FLUXERR_AUTO',  # ct float32
                   'MAG_AUTO',  # mag float32
                   'MAGERR_AUTO',  # mag float32
                   'THRESHOLD',  # ct float32
                   'X_IMAGE',  # pix float32
                   'Y_IMAGE',  # pix float32
                   'ALPHA_J2000',  # deg float64
                   'DELTA_J2000',  # deg float64
                   'A_WORLD',  # deg float32
                   'B_WORLD',  # deg float32
                   'FLUX_RADIUS',  # pix float32
                   'THETA_IMAGE',  # deg float32
                   'FWHM_IMAGE',  # pix float32
                   'FWHM_WORLD',  # deg float32
                   'FLAGS',  # int16
                   'IMAFLAGS_ISO',  # int32
                   'NIMAFLAGS_ISO',  # int32
                   'CLASS_STAR']  # float32

        CatalogImages = [
            {'path': STACK_COSMOS_IMAGE_PATH,
             'name': 'COSMOS_g_sci.fits',
             'filter': 'unknown',
             'instrument': 'unknown',
             'cols': [],
             'labels': [],
             'image': None
             #'frame': 'icrs'
            }]

        def __init__(self):
            super(STACK_COSMOS, self).__init__()

            self.dataframe_of_bid_targets = None
            self.dataframe_of_bid_targets_photoz = None
            # self.table_of_bid_targets = None
            self.num_targets = 0

            self.read_main_catalog()

            self.master_cutout = None

        @classmethod
        def read_catalog(cls, catalog_loc, name):
            "This catalog is in a fits file"

            log.debug("Building " + name + " dataframe...")
            try:
                # f = fits.open(catalog_loc)[1].data
                table = Table.read(catalog_loc)
            except:
                log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
                return None

            # convert into a pandas dataframe ... cannot convert directly to pandas because of the [25] lists
            # so build a pandas df with just the few columns we need for searching
            # then pull data from full astro table as needed

            try:
                lookup_table = Table([table['NUMBER'], table['ALPHA_J2000'], table['DELTA_J2000']])
                df = lookup_table.to_pandas()
                old_names = ['NUMBER', 'ALPHA_J2000', 'DELTA_J2000']
                new_names = ['ID', 'RA', 'DEC']
                df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
                cls.AstroTable = table
            except:
                log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
                return None

            return df

        def sort_bid_targets_by_likelihood(self, ra, dec):
            # right now, just by euclidean distance (ra,dec are of target)
            #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)
            self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra) ** 2 +
                                                                (self.dataframe_of_bid_targets['DEC'] - dec) ** 2)
            self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)


        def build_list_of_bid_targets(self, ra, dec, error):
            '''ra and dec in decimal degrees. error in arcsec.
            returns a pandas dataframe'''

            error_in_deg = float(error) / 3600.0

            self.dataframe_of_bid_targets = None
            self.num_targets = 0

            ra_min = float(ra - error_in_deg)
            ra_max = float(ra + error_in_deg)
            dec_min = float(dec - error_in_deg)
            dec_max = float(dec + error_in_deg)

            log.info(self.Name + " searching for bid targets in range: RA [%f +/- %f], Dec [%f +/- %f] ..."
                     % (ra, error_in_deg, dec, error_in_deg))

            try:
                self.dataframe_of_bid_targets = self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                                                        (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)]

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



        def build_bid_target_reports(self, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                     target_w=0, fiber_locs=None):
            self.clear_pages()
            self.build_list_of_bid_targets(target_ra, target_dec, error)

            if self.num_targets == 0:
                return None

            ras = self.dataframe_of_bid_targets.loc[:, ['RA']].values
            decs = self.dataframe_of_bid_targets.loc[:, ['DEC']].values

            # display the exact (target) location
            entry = self.build_exact_target_location_figure(target_ra, target_dec, error, section_title=section_title,
                                                            target_w=target_w, fiber_locs=fiber_locs)

            if entry is not None:
                self.add_bid_entry(entry)

            number = 0
            # display each bid target
            for r, d in zip(ras, decs):
                number += 1
                try:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0])]

                    idnum = df['ID'].values[0]  # to matchup in photoz catalog
                except:
                    log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                    continue  # this must be here, so skip to next ra,dec

                #todo: is there photometry???

                print("Building report for bid target %d in %s" % (base_count + number, self.Name))
                entry = self.build_bid_target_figure(r[0], d[0], error=error, df=df, df_photoz=None,
                                                     target_ra=target_ra, target_dec=target_dec,
                                                     section_title=section_title,
                                                     bid_number=number, target_w=target_w)
                if entry is not None:
                    self.add_bid_entry(entry)

            return self.pages

        def build_exact_target_location_figure(self, ra, dec, error, section_title="", target_w=0, fiber_locs=None):
            '''Builds the figure (page) the exact target location. Contains just the filter images ...

            Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

            #there is just one image

            # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
            # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS
            window = error * 4

            num_cat_images = len(self.CatalogImages)
            cols = max(num_cat_images,6)
            if num_cat_images > 1:
                rows = 2
            else:
                rows = 1

            fig_sz_x = cols * 3
            fig_sz_y = rows * 3

            fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))

            gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)
            # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

            font = FontProperties()
            font.set_family('monospace')
            font.set_size(12)

            title = "Catalog: %s\n" % self.Name + section_title + "\nTarget Location\nPossible Matches=%d (within %g\")\n" \
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

            #there is (at this time) just the one image, but leave in the loop in-case we change that?
            index = -1
            for i in self.CatalogImages:  # i is a dictionary
                index += 1

                if i['image'] is None:
                    i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                             image_location=op.join(i['path'], i['name'])) #,frame=i['frame'])
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

                    if rows > 1:
                        plt.subplot(gs[rows - 1, index])
                        plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                                   vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                        plt.title(i['instrument'] + " " + i['filter'])

                        plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                                          angle=0.0, color='red', fill=False))

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
            # only show this lable if there is not going to be an adjacent fiber plot
            if (fiber_locs is None) or (len(fiber_locs) == 0):
                plt.ylabel("arcsecs")
            plt.plot(0, 0, "r+")
            plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                              angle=0.0, color='red', fill=False))

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

                plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                                  angle=0.0, color='red', fill=False))

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

                ext = max(abs(-xmin - 2 * G.Fiber_Radius), abs(xmax + 2 * G.Fiber_Radius),
                          abs(-ymin - 2 * G.Fiber_Radius), abs(ymax + 2 * G.Fiber_Radius), 2.0)

                # need a new cutout since we rescalled the ext (and window) size
                cutout = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
                vmin, vmax = empty_sci.get_vrange(cutout.data)

                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

            # complete the entry
            plt.close()
            return fig



        def build_bid_target_figure(self, ra, dec, error, df=None, df_photoz=None, target_ra=None, target_dec=None,
                                    section_title="", bid_number=1, target_w=0):
            '''Builds the entry (e.g. like a row) for one bid target. Includes the target info (name, loc, Z, etc),
            photometry images, Z_PDF, etc

            Returns the matplotlib figure. Due to limitations of matplotlib pdf generateion, each figure = 1 page'''

            # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
            # not the middle, so need the total length of each side to be twice translated error or 2*2*errorS

            window = error * 2.

            num_cat_images = len(self.CatalogImages)
            cols = max(num_cat_images, 6)
            if num_cat_images > 1:
                rows = 2
            else:
                rows = 1

            fig_sz_x = cols * 3
            fig_sz_y = rows * 3

            gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

            font = FontProperties()
            font.set_family('monospace')
            font.set_size(12)

            fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))

            if df is not None:
                title = "%s\nPossible Match #%d\n\nRA = %f    Dec = %f\nSeparation = %g\"" \
                        % (section_title, bid_number, df['RA'].values[0],
                           df['DEC'].values[0],
                           df['distance'].values[0] * 3600)

                if target_w > 0:
                    la_z = target_w / G.LyA_rest - 1.0
                    oii_z = target_w / G.OII_rest - 1.0
                    title = title + "\nLyA Z   = %g (red)" % la_z
                    if (oii_z > 0):
                        title = title + "\nOII Z   = %g (green)" % oii_z
                    else:
                        title = title + "\nOII Z   = N/A"
            else:
                title = "%s\nRA=%f    Dec=%f" % (section_title, ra, dec)

            plt.subplot(gs[0, 0])
            plt.text(0, 0.20, title, ha='left', va='bottom', fontproperties=font)
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')

            index = -1
            # iterate over all filter images
            for i in self.CatalogImages:  # i is a dictionary
                index += 1  # for subplot ... is 1 based
                if i['image'] is None:
                    i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                             image_location=op.join(i['path'], i['name']))
                sci = i['image']

                cutout = sci.get_cutout(ra, dec, error, window=window)
                ext = sci.window / 2.

                if cutout is not None:

                    if rows == 1:
                        plt.subplot(gs[rows-1, cols-2])
                    else:
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

                        plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2., height=error * 2.,
                                                          angle=0.0, color='yellow', fill=False, linewidth=5.0,
                                                          zorder=1))
                        # set the diameter of the cirle to half the error (radius error/4)
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



            empty_sci = science_image.science_image()
            # master cutout (0,0 is the observered (exact) target RA, DEC)
            if self.master_cutout is not None:
                # window=error*4
                ext = error * 2.
                plt.subplot(gs[0, cols - 1])
                vmin, vmax = empty_sci.get_vrange(self.master_cutout.data)
                plt.imshow(self.master_cutout.data, origin='lower', interpolation='none',
                           cmap=plt.get_cmap('gray_r'),
                           vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
                plt.title("Master Cutout (Stacked)")
                plt.xlabel("arcsecs")
                plt.ylabel("arcsecs")

                # mark the bid target location on the master cutout
                if (target_ra is not None) and (target_dec is not None):
                    px, py = empty_sci.get_position(target_ra, target_dec, self.master_cutout)
                    x, y = empty_sci.get_position(ra, dec, self.master_cutout)
                    plt.plot(0, 0, "r+")

                    # set the diameter of the cirle to half the error (radius error/4)
                    plt.gca().add_patch(
                        plt.Circle(((x - px), (y - py)), radius=error / 4.0, color='yellow', fill=False))
                    plt.gca().add_patch(plt.Rectangle((-error, -error), width=error * 2, height=error * 2,
                                                      angle=0.0, color='red', fill=False))
                    x = (x - px) - error
                    y = (y - py) - error
                    plt.gca().add_patch(plt.Rectangle((x, y), width=error * 2, height=error * 2,
                                                      angle=0.0, color='yellow', fill=False))

            plt.close()
            return fig


#######################################
#end class STACK_COSMOS
#######################################
