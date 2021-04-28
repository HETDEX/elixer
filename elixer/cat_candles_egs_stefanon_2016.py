from __future__ import print_function

try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import line_prob
    from elixer import cat_base
    from elixer import match_summary
    from elixer import utilities
    from elixer import sqlite_utils as sql
    from elixer import spectrum_utilities as SU
except:
    import global_config as G
    import science_image
    import line_prob
    import cat_base
    import match_summary
    import utilities
    import sqlite_utils as sql
    import spectrum_utilities as SU

import os.path as op
import copy
import io



CANDELS_EGS_Stefanon_2016_BASE_PATH = G.CANDELS_EGS_Stefanon_2016_BASE_PATH
CANDELS_EGS_Stefanon_2016_CAT = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH,
                                        "photometry/CANDELS.EGS.F160W.v1_1.photom.cat")
CANDELS_EGS_Stefanon_2016_IMAGES_PATH = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH, "images")
CANDELS_EGS_Stefanon_2016_PHOTOZ_CAT = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH , "photoz/zcat_EGS_v2.0.cat")
CANDELS_EGS_Stefanon_2016_PHOTOZ_ZPDF_PATH = op.join(CANDELS_EGS_Stefanon_2016_BASE_PATH, "photoz/zPDF/")

EXPANDED_IMAGES_PATH = G.EGS_CFHTLS_PATH

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


# # filter transforms: (see https://ui.adsabs.harvard.edu/abs/2018AJ....156..296H/abstract)
# Filter	Identifier	t(s)	ZP (mag)	lambda_p (nm)	Deltalambda (nm)
# F438W	ibcd40011	1600	24.886	432.5	61.8
# F475W	ibcd40khq	465	25.697	477.3	134.4
# F475X	ibcd40kbq	275	26.107	493.9	205.6
# F555W	ibcd40021	1160	25.728	530.8	156.2
# F606W	ibcd40031	900	25.902	588.7	218.2
# F814W	ibcd40061	1300	24.586	805.4	153.6

def cfhtls_count_to_mag(count,cutout=None,sci_image=None):
    if count is not None:
        if count > 0:
            return -2.5 * np.log10(count) + 30.0 #PHOTZP in FITS header
        else:
            return 99.9  # need a better floor


def acs_wfc_f606w_count_to_mag(count,cutout=None,headers=None):
    if count is not None:
        #if cutout is not None:
        #get the conversion factor, each tile is different
        try:
            photoflam = float(headers[0]['PHOTFLAM']) #inverse sensitivity, ergs / cm2 / Ang / electron
            photozero = float(headers[0]['PHOTZPT']) #/ ST magnitude zero point

            if not isinstance(count, float):
                count = count.value

            if count > 0:
                flux = photoflam * count
                return -2.5 * np.log10(flux) + photozero
            else:
                return 99.9  # need a better floor

        except:
            log.warning("Exception in acs_wfc_f606w_count_to_mag",exc_info=True)
            return 99.9




class CANDELS_EGS_Stefanon_2016(cat_base.Catalog):
    # RA,Dec in decimal degrees

    # photometry catalog
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

    # class variables
    MainCatalog = CANDELS_EGS_Stefanon_2016_CAT
    Name = "CANDELS/EGS"
    # if multiple images, the composite broadest range (filled in by hand)
    #includes the expanded catalog
    Image_Coord_Range = {'RA_min': 214.0, 'RA_max': 215.7, 'Dec_min': 52.15, 'Dec_max': 53.21}
#    Cat_Coord_Range = {'RA_min': 214.576759, 'RA_max': 215.305229, 'Dec_min': 52.677569, 'Dec_max': 53.105756}
    #updated with CFHTLS extended coverage
    Cat_Coord_Range = {'RA_min': 208.559, 'RA_max': 220.391, 'Dec_min': 51.2113, 'Dec_max': 57.8033}

    mean_FWHM = 0.15 #typical use for photometric aperture, but is too good here ... objects that are point
                    #sources may be resolved with HST

    MAG_LIMIT = 30.0 #associated catalog goes well into 32+, but this is an overall limit
    WCS_Manual = False#True
    EXPTIME_F606W = 289618.0
    CONT_EST_BASE = 3.3e-21 #about 30 mag
    BidCols = ["ID", "IAU_designation", "RA", "DEC",
               "CFHT_U_FLUX", "CFHT_U_FLUXERR",
               "IRAC_CH1_FLUX", "IRAC_CH1_FLUXERR", "IRAC_CH2_FLUX", "IRAC_CH2_FLUXERR",
               "ACS_F606W_FLUX", "ACS_F606W_FLUXERR",
               "ACS_F814W_FLUX", "ACS_F814W_FLUXERR",
               "WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR",
               "WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR",
               "WC3_F160W_FLUX", "WFC3_F160W_FLUXERR",
               "DEEP_SPEC_Z"]  # NOTE: there are no F105W values


    #todo: add 4x2 array of corners (ra,dec) to each catalogimage so can quickly check if target ra,dec is contained

    CatalogImages = [
        # {'path': EXPANDED_IMAGES_PATH,
        #  'name': 'D3.I.1_20558_1_21553.fits', #or: CFHTLS_D-25_g_141927+524056_T0007_SIGWEI.fits
        #  'filter': 'g',
        #  'instrument': 'CFHTLS',
        #  'cols': [],
        #  'labels': [],
        #  'image': None,
        #  'expanded': True,
        #  'wcs_manual': False,
        #  'sky_subtract': True,
        #  'aperture': 0.9*0.5 + 0.5,  #if non-zero, use an aperture of this radius in arcsecs to find image based mag
        #                              # since a radius, half the FWHM + 0.5" for astrometric error
        #  'mag_func': cfhtls_count_to_mag,
        #  'footprint': [[215.72182464, 52.12056378],[215.74413972, 53.23356319],
        #                [213.97033408, 53.23356319],[213.99264916, 52.12056378]],
        #  'RA_min': 213.970334,
        #  'RA_max': 215.744140,
        #  'Dec_min': 52.120564,
        #  'Dec_max': 53.233563
        #  }, #mean FWHM for CFHTLS is better than 0.9"
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits',
         'filter': 'f606w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F606W_FLUX", "ACS_F606W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': mean_FWHM * 0.5 + 0.5, # since a radius, half the FWHM + 0.5" for astrometric error
                             # , #if zero, there is something odd with the header and can't use astropy apertures
                             #normally would be the FWHM/2, but these tend to be resolved and NOT point sources
         'mag_func': acs_wfc_f606w_count_to_mag,
         'sky_subtract': False
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_acs_wfc_f814w_060mas_v1.1_drz.fits',
         'filter': 'f814w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F814W_FLUX", "ACS_F814W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 0,
         'mag_func': None,
         'sky_subtract': False
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f105w_060mas_v1.5_drz.fits',
         'filter': 'f105w',
         'instrument': 'WFC3',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 0,
         'mag_func': None,
         'sky_subtract': False
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f125w_060mas_v1.1_drz.fits',
         'filter': 'f125w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 0,
         'mag_func': None,
         'sky_subtract': False
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f140w_060mas_v1.1_drz.fits',
         'filter': 'f140w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 0,
         'mag_func': None,
         'sky_subtract': False
         },
        {'path': CANDELS_EGS_Stefanon_2016_IMAGES_PATH,
         'name': 'egs_all_wfc3_ir_f160w_060mas_v1.1_drz.fits',
         'filter': 'f160w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F160W_FLUX", "WFC3_F160W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 0,
         'mag_func': None,
         'sky_subtract': False
         }
    ]

    if not G.BUILD_REPORT_BY_FILTER:
        CatalogImages = [{'path': EXPANDED_IMAGES_PATH,
                           'name': 'D3.I.1_20558_1_21553.fits', #or: CFHTLS_D-25_g_141927+524056_T0007_SIGWEI.fits
                           'filter': 'g',
                           'instrument': 'CFHTLS',
                           'cols': [],
                           'labels': [],
                           'image': None,
                           'expanded': True,
                           'wcs_manual': False,
                           'sky_subtract': True,
                           'aperture': 0.9*0.5 + 0.5,  #if non-zero, use an aperture of this radius in arcsecs to find image based mag
                                                       # since a radius, half the FWHM + 0.5" for astrometric error
                           'mag_func': cfhtls_count_to_mag,
                           'footprint': [[215.72182464, 52.12056378],[215.74413972, 53.23356319],
                                         [213.97033408, 53.23356319],[213.99264916, 52.12056378]],
                           'RA_min': 213.970334,
                           'RA_max': 215.744140,
                           'Dec_min': 52.120564,
                           'Dec_max': 53.233563
                           },] + CatalogImages

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

    CFHTLS_PhotoZCatalog = G.CFHTLS_PHOTOZ_CAT
    cfhtls_df = None  # cat_base has df
    cfhtls_df_photoz = None

    def __init__(self):
        super(CANDELS_EGS_Stefanon_2016, self).__init__()

        # self.dataframe_of_bid_targets = None #defined in base class
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

        # do this only as needed
        # self.read_main_catalog()
        # self.read_photoz_catalog()
        # self.build_catalog_images() #will just build on demand

        self.master_cutout = None


    # todo: is this more efficient? garbage collection does not seem to be running
    # so building as needed does not seem to help memory
    def build_catalog_images(self):
        for i in self.CatalogImages:  # i is a dictionary
            i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                     image_location=op.join(i['path'], i['name']))

    @classmethod
    def read_main_catalog(cls):
        super(CANDELS_EGS_Stefanon_2016, cls).read_main_catalog()

        if cls.cfhtls_df is not None:
            return #already read in

        try:
            print("Reading CFHTLS catalog for ", cls.Name)
            cls.cfhtls_df = cls.read_chftls_catalog(cls.CFHTLS_PhotoZCatalog, cls.Name)

        except:
            print("Failed")
            cls.status = -1
            log.error("Exception in read_main_catalog for %s" % cls.Name, exc_info=True)




    @classmethod
    def read_photoz_catalog(cls):
        if cls.df_photoz is not None:
            log.debug("Already built df_photoz")
        else:
            try:
                print("Reading photoz catalog for ", cls.Name)
                cls.df_photoz = cls.read_catalog(cls.PhotoZCatalog, cls.Name)
            except:
                print("Failed")

        return

    @classmethod
    def read_catalog(cls, catalog_loc, name):

        log.debug("Building " + name + " dataframe...")
        idx = []
        header = []
        skip = 0
        keep_f = False

        f = None
        if op.exists(catalog_loc):
            try:
                f = open(catalog_loc, mode='r')
            except:
                log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
                return None
        else: #see if sql db is there
            db_loc = op.join(op.dirname(catalog_loc),"zPDF.db")
            log.debug(f"Checking zPDF database {db_loc} ...")
            if op.exists(db_loc):
                try:
                    f = sql.fetch_zpdf(db_loc, fn=op.basename(catalog_loc))
                    f = io.StringIO(f.decode()) #treat as a text stream (but still has the \t and \n un-translated
                    keep_f = True
                except:
                    log.error(name + " Exception attempting to open catalog zPDF Db: " + db_loc, exc_info=True)
                    return None
            else:
                log.debug(f"zPDF database {db_loc} does not exist")

        #build up the header for Pandas
        if f is None:
            log.debug("Could not open database file")
            return None

        line = f.readline()
        while '#' in line:
            skip += 1
            toks = line.split()
            if (len(toks) > 2) and toks[1].isdigit():  # format:   # <id number> <column name>
                idx.append(toks[1])
                header.append(toks[2])
            line = f.readline()

        if not keep_f:
            f.close()

        try:
            if keep_f:
                df = pd.read_csv(f, names=header,
                             delim_whitespace=True, header=None, index_col=False, skiprows=0)
            else:
                df = pd.read_csv(catalog_loc, names=header,
                             delim_whitespace=True, header=None, index_col=False, skiprows=skip)
        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            if keep_f:
                f.close()
            return None
        if keep_f:
            f.close()

        return df

    @classmethod
    def read_chftls_photoz_catalog(cls):
        if cls.df_cfhtls_photoz is not None:
            log.debug("Already built df_photoz")
        else:
            try:
                print("Reading photoz catalog for ", cls.Name)
                cls.df_cfhtls_photoz = cls.read_chftls_catalog(cls.CFHTLS_PhotoZCatalog, cls.Name)
            except:
                print("Failed")

        return

    @classmethod
    def read_chftls_catalog(cls, catalog_loc, name):

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
        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return df

    def get_filter_flux(self,df):

        filter_fl = None
        filter_fl_err = None
        mag = None
        mag_plus = None
        mag_minus = None
        filter_str = 'ACS_F606W_FLUX' #used to lookup
        filter_name= 'f606w'
        try:
            filter_fl = df[filter_str].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
            filter_fl_err = df['ACS_F606W_FLUXERR'].values[0]
            mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl,filter_fl_err)
        except: #not the EGS df, try the CFHTLS
            try: # try f435 (~ g-band)
                filter_str = 'ACS_F435W_FLUX' #used to lookup
                filter_name= 'f435w'
                filter_fl = df[filter_str].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                filter_fl_err = df['ACS_F435W_FLUXERR'].values[0]
                mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl,filter_fl_err)
            except:
                filter_str = None
                filter_name = None

        if filter_fl is None:
            try:
                filter_fl = self.obs_mag_to_micro_Jy(df['G'].values[0])
                filter_fl_err = abs(filter_fl - self.obs_mag_to_micro_Jy(df['G'].values[0] - df['eG'].values[0]))
                mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl, filter_fl_err)
            except:
                pass

        return filter_fl, filter_fl_err, mag, mag_plus, mag_minus, filter_name


    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        if self.df is None:
            self.read_main_catalog()
        if self.df_photoz is None:
            self.read_photoz_catalog()

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

            if self.num_targets > 0:
                # ID matches between both catalogs
                self.dataframe_of_bid_targets_photoz = \
                    self.df_photoz[(self.df_photoz['ID'].isin(self.dataframe_of_bid_targets['ID']))].copy()
        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.num_targets == 0:  # try the larger CFHTLS
            try:

                self.dataframe_of_bid_targets = \
                    self.cfhtls_df[(self.cfhtls_df['RA'] >= ra_min) & (self.cfhtls_df['RA'] <= ra_max) &
                                   (self.cfhtls_df['DEC'] >= dec_min) & (self.cfhtls_df['DEC'] <= dec_max)].copy()

                # the CFHTLS dataframe is combined with photo-z (at least for now)
                self.dataframe_of_bid_targets_photoz = self.dataframe_of_bid_targets

                if self.dataframe_of_bid_targets is not None:
                    self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
                    # todo: re-name column headers for compatibility?
                    # todo: maybe add a column to identify which catalog this is and wrap the pull of data
                    # todo:      for printing?

                    self.dataframe_of_bid_targets_photoz['file'] = None
                    self.dataframe_of_bid_targets_photoz['z_best_type'] = 'p'
                    self.dataframe_of_bid_targets_photoz['mFDa4_z_weight'] = None

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

            if G.SINGLE_PAGE_PER_DETECT:
                entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                            target_ra=target_ra, target_dec=target_dec,
                                                                            target_w=target_w, target_flux=target_flux,
                                                                            detobj=detobj)
                if entry is not None:
                    self.add_bid_entry(entry)
        else:
            return None

        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS): # each bid taget gets its own line
            log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

        return self.pages


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
                                                    fiber_locs, target_flux,detobj,do_sky_subtract=False)

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
        for r, d in zip(bid_ras, bid_decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break

            spec_z = -1.0

            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0])]

                idnum = df['ID'].values[0]  # to matchup in photoz catalog
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            try:
                # note cannot dirctly use RA,DEC as the recorded precission is different (could do a rounded match)
                # but the idnums match up, so just use that
                df_photoz = self.dataframe_of_bid_targets_photoz.loc[
                    self.dataframe_of_bid_targets_photoz['ID'] == idnum]

                if len(df_photoz) == 0:
                    log.debug("No conterpart found in photoz catalog; RA=%f , Dec =%f" % (r[0], d[0]))
                    df_photoz = None
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                df_photoz = None

            z_photoz_weighted = -1
            if df_photoz is not None:
                try:
                    photoz_file = df_photoz['file'].values[0]
                    z_best = df_photoz['z_best'].values[0]
                    z_best_type = df_photoz['z_best_type'].values[0]  # s = spectral , p = photometric?
                    z_photoz_weighted = df_photoz['mFDa4_z_weight'].values[0]
                except:
                    z_photoz_weighted = -1
                    log.error("Exception!", exc_info=True)

            if z_photoz_weighted is None:
                z_photoz_weighted = -1

            if df is not None:

                if z_best_type is not None and z_best_type.lower() == 's':
                    spec_z = z_best

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
                        filter_fl_cgs = self.micro_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w))
                        # filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        filter_fl_cgs_unc = self.micro_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
                        # assumes no error in wavelength or c

                        # bid target info is only of value if we have a flux from the emission line
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
                        if spec_z >= 0.0:
                            bid_target.spec_z = spec_z

                        if z_photoz_weighted >= 0.0:
                            bid_target.phot_z = z_photoz_weighted

                        lineFlux_err = 0.
                        if detobj is not None:
                            try:
                                lineFlux_err = detobj.estflux_unc
                            except:
                                lineFlux_err = 0.

                        try:
                            # ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                            # ew_u = abs(ew * np.sqrt(
                            #     (detobj.estflux_unc / target_flux) ** 2 +
                            #     (filter_fl_err / filter_fl) ** 2))
                            #
                            # bid_target.bid_ew_lya_rest = ew
                            # bid_target.bid_ew_lya_rest_err = ew_u

                            bid_target.bid_ew_lya_rest, bid_target.bid_ew_lya_rest_err = \
                                SU.lya_ewr(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                           bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

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



                        # build EW error from lineFlux_err and aperture estimate error
                        #ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                        # try:
                        #     ew_obs_err = abs(ew_obs * np.sqrt(
                        #         (lineFlux_err / target_flux) ** 2 +
                        #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                        # except:
                        #     ew_obs_err = 0.

                        ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                                       bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

                        bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                            line_prob.mc_prob_LAE(
                                wl_obs=target_w,
                                lineFlux=target_flux,
                                lineFlux_err=lineFlux_err,
                                continuum=bid_target.bid_flux_est_cgs * SU.continuum_band_adjustment(target_w,bid_target.bid_filter),
                                continuum_err=bid_target.bid_flux_est_cgs_unc * SU.continuum_band_adjustment(target_w,bid_target.bid_filter),
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

                        for c in self.CatalogImages:
                            try:
                                bid_target.add_filter(c['instrument'], c['filter'],
                                                      self.micro_jansky_to_cgs(df[c['cols'][0]].values[0],
                                                                               SU.filter_iso(c['filter'],target_w)),
                                                      self.micro_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                               SU.filter_iso(c['filter'],target_w)))
                            except:
                                log.debug('Could not add filter info to bid_target.')


                # add photo_z plot
                # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
                # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
                # get 'file'
                # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
                #overplot photo Z lines

                if bid_target and photoz_file is not None:
                    z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
                    if z_cat is not None:
                        bid_target.phot_z_pdf_z = z_cat['z'].values
                        bid_target.phot_z_pdf_pz = z_cat['mFDa4'].values
                        #(spec_z already assigned)


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

    def build_cat_summary_figure (self, cat_match,ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

        cont_est = -1

        # set a minimum window size?
        # if window < 8:
        #    window = 8

        rows = 10 #2 (use 0 for text and 1: for plots)
        cols = 1+ len(self.CatalogImages) #(use 0 for master_stacked and 1 - N for filters)

        fig_sz_x = 18 #cols * 3 # was 6 cols
        fig_sz_y = 3 #rows * 3 # was 1 or 2 rows

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        #All on one line now across top of plots
        if G.ZOO:
            title = "Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets), error)
        else:
            title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets), error)

        # if target_flux is not None:
        #     cont_est = self.CONT_EST_BASE*3 #self.get_f606w_max_cont(self.EXPTIME_F606W, 3, self.CONT_EST_BASE)
        #     if cont_est != -1:
        #         title += "  Minimum (no match) 3$\sigma$ rest-EW: "
        #         title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
        #         if target_w >= G.OII_rest:
        #             title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
        #         else:
        #             title = title + "  OII = N/A"

        plt.subplot(gs[0, :])
        #text may be updated below with PLAE()
        text = plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        ref_exptime = None
        total_adjusted_exptime = None

        # add the bid targets
        bid_colors = self.get_bid_colors(len(bid_ras))

        best_plae_poii = None
        best_plae_poii_filter = '-'
        best_plae_range = None
        bid_target = None
        index = 0 #start in the 2nd box which is index 1 (1st box is for the fiber position plot)
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

            try:
                wcs_manual = i['wcs_manual']
                aperture = i['aperture']
                mag_func = i['mag_func']
                do_sky_subtract = i['sky_subtract']
            except:
                wcs_manual = self.WCS_Manual
                aperture = 0.0
                mag_func = None

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=wcs_manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag, mag_radius,details = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func,
                                                     do_sky_subtract=do_sky_subtract,return_details=True)

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

            bid_target = None
            cutout_ewr = None
            cutout_ewr_err = None
            cutout_plae = None

            try: #update non-matched source line with PLAE()
                #if (mag < 99) and (target_flux is not None) and (i['instrument'] == 'CFHTLS') and (i['filter'] == 'g'):
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None) \
                            and (((i['instrument'] == 'CFHTLS') and (i['filter'] == 'g')) or (i['filter'] == 'f606w')):
                    #make a "blank" catalog match (e.g. at this specific RA, Dec (not actually from catalog)
                    bid_target = match_summary.BidTarget()
                    bid_target.catalog_name = self.Name
                    bid_target.bid_ra = 666 #nonsense RA
                    bid_target.bid_dec = 666 #nonsense Dec
                    bid_target.distance = 0.0
                    bid_target.bid_filter = i['filter']
                    bid_target.bid_mag = mag
                    bid_target.bid_mag_err_bright = 0.0 #todo: right now don't have error on aperture mag
                    bid_target.bid_mag_err_faint = 0.0

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

                    bid_target.add_filter(i['instrument'],i['filter'],bid_target.bid_flux_est_cgs,-1)

                    addl_waves = None
                    addl_flux = None
                    addl_ferr = None
                    try:
                        addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                        addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                        addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                    except:
                        pass


                    #todo: calculate error on EW as flux from HETDEX (so flux_err from HETDEX) and continuum
                    #todo: from catalog (so, continuum error from catalog)

                    lineFlux_err = 0.
                    if detobj is not None:
                        try:
                            lineFlux_err = detobj.estflux_unc
                        except:
                            lineFlux_err = 0.

                    #build EW error from lineFlux_err and aperture estimate error
                    # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                    # try:
                    #     ew_obs_err =  abs(ew_obs * np.sqrt(
                    #                     (lineFlux_err / target_flux) ** 2 +
                    #                     (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                    # except:
                    #     ew_obs_err = 0.

                    # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                    #     line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                    #                        ew_obs=ew_obs,
                    #                        lineFlux_err= lineFlux_err,
                    #                        ew_obs_err= ew_obs_err,
                    #                        c_obs=None, which_color=None, addl_fluxes=addl_flux,
                    #                        addl_wavelengths=addl_waves,addl_errors=addl_ferr,sky_area=None,
                    #                        cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                    #                        sigma=None,estimate_error=True)


                    ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                                   bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                        line_prob.mc_prob_LAE(
                            wl_obs=target_w,
                            lineFlux=target_flux,
                            lineFlux_err=lineFlux_err,
                            continuum=bid_target.bid_flux_est_cgs* SU.continuum_band_adjustment(target_w,bid_target.bid_filter),
                            continuum_err=bid_target.bid_flux_est_cgs_unc* SU.continuum_band_adjustment(target_w,bid_target.bid_filter),
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

                    if best_plae_poii is None or i['filter'] == 'f606w':
                        best_plae_poii = bid_target.p_lae_oii_ratio
                        best_plae_poii_filter = i['filter']
                        if plae_errors:
                            try:
                                best_plae_range = plae_errors['ratio']
                            except:
                                pass

                    # #if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    # #    text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g" % (bid_target.p_lae_oii_ratio))
                    # if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g (%s)"
                    #     % (bid_target.p_lae_oii_ratio, i['filter']))

                    cat_match.add_bid_target(bid_target)
                    try:  # no downstream edits so they can both point to same bid_target
                        detobj.bid_target_list.append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to detobj.", exc_info=True)
            except:
                log.debug('Could not build exact location photometry info.',exc_info=True)

            # if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
            #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g (%s)"
            #                   % (best_plae_poii, best_plae_poii_filter))


            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            if cutout is not None:  # construct master cutout
                #1st cutout might not be what we want for the master (could be a summary image from elsewhere)
                if self.master_cutout:
                    if self.master_cutout.shape != cutout.shape:
                        del self.master_cutout
                        self.master_cutout = None


                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True,reset_center=False)
                    #self.master_cutout,_,_,_ = sci.get_cutout(ra, dec, error, window=window, copy=True)
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

        # if (not G.ZOO) and (best_plae_poii is not None):
        #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)"
        #                   % (best_plae_poii, best_plae_poii_filter))

        if (not G.ZOO) and (best_plae_poii is not None):
            try:
                text.set_text(
                    text.get_text() + "  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$ (%s)" %
                    (round(best_plae_poii, 3),
                     round(best_plae_range[2], 3),
                     round(best_plae_range[1], 3),
                     best_plae_poii_filter))
            except:
                log.debug("Exception adding PLAE with range", exc_info=True)
                try:
                    text.set_text(text.get_text() + "  P(LAE)/P(OII): %0.4g (%s)" % (best_plae_poii, best_plae_poii_filter))
                except:
                    text.set_text(
                        text.get_text() + "  P(LAE)/P(OII): (%s) (%s)" % ("---", best_plae_poii_filter))

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            return None  # empty (catch_all) will produce fiber locations
            #still need to plot relative fiber positions here
            # plt.subplot(gs[1:, 0])
            # return self.build_empty_cat_summary_figure(ra, dec, error, bid_ras, bid_decs, target_w=target_w,
            #                                fiber_locs=fiber_locs)
        else:
            self.master_cutout.data /= total_adjusted_exptime




        plt.subplot(gs[1:, 0])
        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)
        # complete the entry
        plt.close()

        #get zoo style cutout as png
        if G.ZOO_MINI and (detobj is not None):
            plt.figure()
            self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout,unlabeled=True)

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


        window = error * 2.
        photoz_file = None
        z_best = None
        z_best_type = None  # s = spectral , p = photometric?
        z_photoz_weighted = None

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
            text = "Too many matching targets. Individual reports on following pages.\n\nMORE PAGES ..."
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
        #targets are in order of increasing distance
        for r, d in zip(ras, decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break

            col_idx += 1
            spec_z = -1.0

            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0])]

                idnum = df['ID'].values[0]  # to matchup in photoz catalog
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            try:
                # note cannot dirctly use RA,DEC as the recorded precission is different (could do a rounded match)
                # but the idnums match up, so just use that
                df_photoz = self.dataframe_of_bid_targets_photoz.loc[
                    self.dataframe_of_bid_targets_photoz['ID'] == idnum]

                if len(df_photoz) == 0:
                    log.debug("No conterpart found in photoz catalog; RA=%f , Dec =%f" % (r[0], d[0]))
                    df_photoz = None
            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                df_photoz = None

            if df_photoz is not None:
                try:
                    photoz_file = df_photoz['file'].values[0]
                    z_best = df_photoz['z_best'].values[0]
                    z_best_type = df_photoz['z_best_type'].values[0]  # s = spectral , p = photometric?
                    z_photoz_weighted = df_photoz['mFDa4_z_weight'].values[0]
                except:
                    log.error("Exception!", exc_info=True)

            if df is not None:
                text = ""

                if G.ZOO:
                    text = text + "%g\"\n%0.3f\n" \
                                  % (df['distance'].values[0] * 3600.,df['dist_prior'].values[0])
                else:
                    text = text + "%g\"\n%0.3f\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600.,df['dist_prior'].values[0],
                                    df['RA'].values[0], df['DEC'].values[0])
                if z_best_type is not None:
                    if (z_best_type.lower() == 'p'):
                        text = text + "N/A\n" + "%g\n" % z_best
                    elif (z_best_type.lower() == 's'):
                        text = text + "%g (circle)\n" % z_best
                        spec_z = z_best
                        if z_photoz_weighted is not None:
                            text = text + "%g\n" % z_photoz_weighted
                        else:
                            text = text + "N/A\n"
                    else:
                        text = text + "N/A\n"
                else:
                    text = text + "N/A\nN/A\n"

                try:
                   # filter_fl = df['ACS_F606W_FLUX'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                   # filter_fl_err = df['ACS_F606W_FLUXERR'].values[0]
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
                        filter_fl_cgs = self.micro_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w))# filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        #text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                        filter_fl_cgs_unc = self.micro_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
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

                        # bid target info is only of value if we have a flux from the emission line
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
                        if spec_z >= 0.0:
                            bid_target.spec_z = spec_z

                        lineFlux_err = 0.
                        if detobj is not None:
                            try:
                                lineFlux_err = detobj.estflux_unc
                            except:
                                lineFlux_err = 0.
                        try:
                            # ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                            # ew_u = abs(ew * np.sqrt(
                            #     (detobj.estflux_unc / target_flux) ** 2 +
                            #     (filter_fl_err / filter_fl) ** 2))
                            #
                            # bid_target.bid_ew_lya_rest = ew
                            # bid_target.bid_ew_lya_rest_err = ew_u

                            bid_target.bid_ew_lya_rest, bid_target.bid_ew_lya_rest_err = \
                                SU.lya_ewr(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                           bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

                            text = text + utilities.unc_str((bid_target.bid_ew_lya_rest, bid_target.bid_ew_lya_rest_err)) + "$\AA$\n"
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



                        # build EW error from lineFlux_err and aperture estimate error
                        # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                        # try:
                        #     ew_obs_err = abs(ew_obs * np.sqrt(
                        #         (lineFlux_err / target_flux) ** 2 +
                        #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                        # except:
                        #     ew_obs_err = 0.

                        # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                        #     line_prob.prob_LAE(wl_obs=target_w,
                        #                        lineFlux=target_flux,
                        #                        ew_obs=ew_obs,
                        #                        lineFlux_err=lineFlux_err,
                        #                        ew_obs_err=ew_obs_err,
                        #                        c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                        #                        addl_fluxes=addl_flux, addl_errors=addl_ferr,sky_area=None,
                        #                        cosmo=None, lae_priors=None,
                        #                        ew_case=None, W_0=None,
                        #                        z_OII=None, sigma=None,estimate_error=True)

                        ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                                       bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)


                        bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                            line_prob.mc_prob_LAE(
                                wl_obs=target_w,
                                lineFlux=target_flux,
                                lineFlux_err=lineFlux_err,
                                continuum=bid_target.bid_flux_est_cgs * SU.continuum_band_adjustment(target_w,bid_target.bid_filter),
                                continuum_err=bid_target.bid_flux_est_cgs_unc * SU.continuum_band_adjustment(target_w,bid_target.bid_filter),
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

                        for c in self.CatalogImages:
                            try:
                                bid_target.add_filter(c['instrument'], c['filter'],
                                                      self.micro_jansky_to_cgs(df[c['cols'][0]].values[0],
                                                                               SU.filter_iso(c['filter'],target_w)),
                                                      self.micro_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                               SU.filter_iso(c['filter'],target_w)))
                            except:
                                log.debug('Could not add filter info to bid_target.')

                        cat_match.add_bid_target(bid_target)
                        try: # no downstream edits so they can both point to same bid_target
                            detobj.bid_target_list.append(bid_target)
                        except:
                            log.warning("Unable to append bid_target to detobj.",exc_info=True)

                else:
                    text += "N/A\nN/A\n"

                #if filter_mag != 0:
                try:
                    text += "%0.2f(%0.2f,%0.2f)\n" % (filter_mag, filter_mag_bright,filter_mag_faint)
                except:
                    log.warning("Magnitude info is none: mag(%s), mag_bright(%s), mag_faint(%s)"
                                %(filter_mag, filter_mag_bright,filter_mag_faint))
                    text += "No mag info\n"
                #else:
                #    text = text + "%g(%g) $\\mu$Jy\n" % (filter_fl, filter_fl_err)

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    try:
                        text += r"$%0.4g\ ^{%.4g}_{%.4g}$" % (utilities.saferound(bid_target.p_lae_oii_ratio, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_max, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_min, 3))
                        text += "\n"
                    except:
                        text += "%0.4g\n" % (utilities.saferound(bid_target.p_lae_oii_ratio, 3))
                else:
                    text += "\n"
            else:
                text = "%s\n%f\n%f\n" % ("--",r, d)

            plt.subplot(gs[0, col_idx])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font,color=bid_colors[col_idx-1])

            # add photo_z plot
            # if the z_best_type is 'p' call it photo-Z, if s call it 'spec-Z'
            # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
            # get 'file'
            # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
            #overplot photo Z lines

            if photoz_file is not None:
                z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
                if z_cat is not None:
                    x = z_cat['z'].values
                    y = z_cat['mFDa4'].values
                    plt.subplot(gs[0, 4:])
                    plt.plot(x, y, color=bid_colors[col_idx-1])
                    plt.xlim([0, 3.6])
                    # trim axis to 0 to 3.6

                    if spec_z >= 0.0:
                        #plt.axvline(x=spec_z, color='gold', linestyle='solid', linewidth=3, zorder=0)
                        plt.scatter([spec_z,],[plt.gca().get_ylim()[1]*0.9,],zorder=9,
                                 marker="o",s=80,facecolors='none',edgecolors=bid_colors[col_idx-1])

                    if col_idx == 1:
                        legend = []
                        if target_w > 0:
                            la_z = target_w / G.LyA_rest - 1.0
                            oii_z = target_w / G.OII_rest - 1.0
                            if (oii_z > 0):
                                h = plt.axvline(x=oii_z, color='g', linestyle='--', zorder=9,
                                                label="OII z(virus) = % g" % oii_z)
                                legend.append(h)
                            h = plt.axvline(x=la_z, color='r', linestyle='--', zorder=9,
                                label="LyA z (VIRUS) = %g" % la_z)
                            legend.append(h)

                            plt.gca().legend(handles=legend, loc='lower center', ncol=len(legend), frameon=False,
                                                 fontsize='small', borderaxespad=0, bbox_to_anchor=(0.5, -0.25))

                    plt.title("Photo z PDF")
                    plt.gca().yaxis.set_visible(False)
                    #plt.xlabel("z")

                  #  if len(legend) > 0:
                  #      plt.gca().legend(handles=legend, loc = 'lower center', ncol=len(legend), frameon=False,
                  #                      fontsize='small', borderaxespad=0,bbox_to_anchor=(0.5,-0.25))


            # fig holds the entire page
        plt.close()
        return fig

#######################################
# end class CANDELS_EGS_Stefanon_2016
#######################################
