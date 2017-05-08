#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import pandas as pd
import global_config
import science_image
import numpy as np
import math
#from astropy.coordinates import Angle
import matplotlib.pyplot as plt
#from astropy.io import ascii #note: this works, but pandas is much faster


log = global_config.logging.getLogger('Cat_logger')
log.setLevel(global_config.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

#base class for catalogs (essentially an interface class)
#all Catalogs classes must implement:


def get_catalog_list():
    #build list of all catalogs below
    cats = list()
    cats.append(CANDELS_EGS_Stefanon_2016())
   # cats.append(dummy_cat())

    return cats


class Catalog:

    CatalogLocation = None
    Name = "Generic Catalog (Base)"
    df = None  # pandas dataframe ... all instances share the same frame
    #tbl = None # astropy.io.table
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
    CatalogLocation = "/home/dustin/code/python/voltron/data/EGS/photometry/CANDELS.EGS.F160W.v1_1.photom.cat"
    Name = "CANDELS_EGS_Stefanon_2016"
    BidCols = ["ID","IAU_designation","RA","DEC",
               "CFHT_U_FLUX","CFHT_U_FLUXERR",
               "IRAC_CH1_FLUX","IRAC_CH1_FLUXERR","IRAC_CH2_FLUX","IRAC_CH2_FLUXERR",
               "ACS_F606W_FLUX","ACS_F606W_FLUXERR","ACS_F606W_V08_FLUX","ACS_F606W_V08_FLUXERR",
               "ACS_F814W_FLUX","ACS_F814W_FLUXERR","ACS_F814W_V08_FLUX","ACS_F814W_V08_FLUXERR",
               "WFC3_F125W_FLUX","WFC3_F125W_FLUXERR","WFC3_F125W_V08_FLUX","WFC3_F125W_V08_FLUXERR",
               "WFC3_F140W_FLUX","WFC3_F140W_FLUXERR",
               "WC3_F160W_FLUX","WFC3_F160W_FLUXERR","WFC3_F160W_V08_FLUX","WFC3_F160W_V08_FLUXERR",
               "DEEP_SPEC_Z"]  #NOTE: there are no F105W values

    Images = [  {'path':"/home/dustin/code/python/voltron/data/EGS/images/",
                 'name':'egs_all_acs_wfc_f606w_060mas_v1.1_drz.fits',
                 'filter':'f606w',
                 'instrument':'ACS WFC',
                 'cols':["ACS_F606W_FLUX","ACS_F606W_FLUXERR","ACS_F606W_V08_FLUX","ACS_F606W_V08_FLUXERR"],
                 'labels':["Flux","Err","V08 Flux", "V08 Err"]
                },
                {'path':"/home/dustin/code/python/voltron/data/EGS/images/",
                 'name':'egs_all_acs_wfc_f814w_060mas_v1.1_drz.fits',
                 'filter':'f814w',
                 'instrument':'ACS WFC',
                 'cols':["ACS_F814W_FLUX","ACS_F814W_FLUXERR","ACS_F814W_V08_FLUX","ACS_F814W_V08_FLUXERR"],
                 'labels':["Flux","Err","V08 Flux","V08 Err"]
                },
                {'path':"/home/dustin/code/python/voltron/data/EGS/images/",
                 'name':'egs_all_wfc3_ir_f105w_060mas_v1.5_drz.fits',
                 'filter':'f105w',
                 'instrument':'WFC3',
                 'cols':[],
                 'labels':[]
                },
                {'path':"/home/dustin/code/python/voltron/data/EGS/images/",
                 'name':'egs_all_wfc3_ir_f125w_060mas_v1.1_drz.fits',
                 'filter':'f125w',
                 'instrument':'WFC3',
                 'cols':["WFC3_F125W_FLUX","WFC3_F125W_FLUXERR","WFC3_F125W_V08_FLUX","WFC3_F125W_V08_FLUXERR"],
                 'labels':["Flux","Err","V08 Flux","V08 Err"]
                },
                {'path':"/home/dustin/code/python/voltron/data/EGS/images/",
                 'name':'egs_all_wfc3_ir_f140w_060mas_v1.1_drz.fits',
                 'filter':'f140w',
                 'instrument':'WFC3',
                 'cols':["WFC3_F140W_FLUX","WFC3_F140W_FLUXERR"],
                 'labels':["Flux","Err"]
                },
                {'path': "/home/dustin/code/python/voltron/data/EGS/images/",
                 'name': 'egs_all_wfc3_ir_f160w_060mas_v1.1_drz.fits',
                 'filter': 'f160w',
                 'instrument': 'WFC3',
                 'cols':["WFC3_F160W_FLUX","WFC3_F160W_FLUXERR","WFC3_F160W_V08_FLUX","WFC3_F160W_V08_FLUXERR"],
                 'labels':["Flux","Err","V08 Flux","V08 Err"]
                }
               ]

    def __init__(self):
      #  super(CANDELS_EGS_Stefanon_2016, self).__init__()

        self.dataframe_of_bid_targets = None
        #self.table_of_bid_targets = None
        self.num_targets = 0
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
        log.debug(cls.Name + " Coordinate Range: RA: %f to %f , Dec: %f to %f" % (cls.RA_min,cls.RA_max,
                                                                                  cls.Dec_min,cls.Dec_max ))


    # @classmethod
    # def read_catalog_ascii(cls):
    #     if cls.tbl is not None:
    #         log.debug("Already built table")
    #         return
    #
    #     log.debug("Building " + cls.Name + " table ...")
    #
    #     try:
    #         cls.tbl = ascii.read(cls.CatalogLocation)
    #     except:
    #         log.error(cls.Name + " Exception attempting to build astropy.io.table")
    #         cls.status = -1
    #         return
    #
    #     cls.status = 0
    #
    #     cls.RA_min = cls.tbl['RA'].min()
    #     cls.RA_max = cls.tbl['RA'].max()
    #     cls.Dec_min = cls.tbl['DEC'].min()
    #     cls.Dec_max = cls.tbl['DEC'].max()

    # def build_list_of_bid_targets_ascii(self,ra,dec,error):
    #     '''ra and dec in decimal degress. error in arcsec.
    #     returns a pandas dataframe'''
    #     self.dataframe_of_bid_targets = None
    #     self.num_targets = 0
    #
    #     ra_min = float(ra - error)
    #     ra_max = float(ra + error)
    #     dec_min = float(dec - error)
    #     dec_max = float(dec + error)
    #
    #     try:
    #         self.table_of_bid_targets = self.tbl[(self.tbl['RA'] > ra_min) & (self.tbl['RA'] < ra_max) &
    #                                             (self.tbl['DEC'] > dec_min) & (self.tbl['DEC'] < dec_max)]
    #     except:
    #         log.error(self.Name + " Exception in build_list_of_bid_targets")
    #
    #     if self.table_of_bid_targets is not None:
    #         self.num_targets = self.table_of_bid_targets.__len__()
    #         log.debug(self.Name + " searching for objects in [%f - %f, %f - %f] " %(ra_min,ra_max,dec_min,dec_max) +
    #               ". Found = %d" % (self.num_targets ))
    #
    #     return self.num_targets, self.table_of_bid_targets



    @classmethod
    def coordinate_range(cls,echo=False):
        if echo:
            msg = "RA (%f, %f)" % (cls.RA_min, cls.RA_max) + "Dec(%f, %f)" % (cls.Dec_min, cls.Dec_max)
            print( msg )
        log.debug(cls.Name + " Simple Coordinate Box: " + msg )
        return (cls.RA_min, cls.RA_max, cls.Dec_min, cls.Dec_max)


    def sort_bid_targets_by_likelihood(self,ra,dec):
        #right now, just by euclidean distance (ra,dec are of target)
        self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra)**2 +
                                                            (self.dataframe_of_bid_targets['DEC'] - dec)**2)
        self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)

    def build_list_of_bid_targets(self,ra,dec,error):
        '''ra and dec in decimal degress. error in arcsec.
        returns a pandas dataframe'''
        self.dataframe_of_bid_targets = None
        self.num_targets = 0

        ra_min = float(ra - error)
        ra_max = float(ra + error)
        dec_min = float(dec - error)
        dec_max = float(dec + error)

        try:
            self.dataframe_of_bid_targets = self.df[(self.df['RA'] > ra_min) & (self.df['RA'] < ra_max) &
                                                (self.df['DEC'] > dec_min) & (self.df['DEC'] < dec_max)]
        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets")

        if self.dataframe_of_bid_targets is not None:
            self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra,dec)

            log.debug(self.Name + " searching for objects in [%f - %f, %f - %f] " %(ra_min,ra_max,dec_min,dec_max) +
                  ". Found = %d" % (self.num_targets ))

        return self.num_targets, self.dataframe_of_bid_targets



    #todo: select columns for each matching record
    #i.e.n = egs.df.loc[0:2,['ACS_F606W_FLUX','ACS_F606W_V08_FLUX']].values
    #   give back a numpy 2D array in this case (here selected the first 2 records, but will want to do this
    #   for one record at a time

    def get_bid_dict(self,id,cols):
        """returns a (nested) dictionary of desired cols for a single row from the full bid dataframe
        form {col_name : {id : value}} where id is 1-based
        """
        try:
            bid_dict = self.dataframe_of_bid_targets.loc[id,cols].to_dict()
            log.debug(str(bid_dict))
        except:
            log.error("Exception attempting to build dictionary for %s : id %d" % (self.name, id))
            return None
        return bid_dict




#########################################
## testing
#########################################
    #for testing only
    def display_all_bid_images(self,target_ra, target_dec, error):
        ras = self.dataframe_of_bid_targets.loc[:,['RA']].values
        decs = self.dataframe_of_bid_targets.loc[:,['DEC']].values
        #dist = self.dataframe_of_bid_targets.loc[:,['distance']].values
        #get back an array of arrays ([[value],[value],...[value]] )

        for r,d in zip(ras,decs):
            df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                   (self.dataframe_of_bid_targets['DEC'] == d[0])]
            self.display_bid_image(r[0],d[0],error,100,df)


    def display_bid_image(self,ra,dec,error,window=100,df=None):

        #num  = len(self.Images)
        #rows = math.trunc(math.sqrt(num))
        #cols = int(math.ceil(float(num)/rows))

        rows = 1
        cols = len(self.Images)

        fig_sz_x = cols*3
        fig_sz_y = rows*5

        index = 0
        plt.figure(figsize=(fig_sz_x,fig_sz_y))
        for i in self.Images: # i is a dictionary
            index+= 1
            sci = science_image.science_image()
            sci.image_location = i['path']+i['name']

            sci.load_image(wcs_manual=True)
            cutout = sci.get_cutout(ra, dec, error, window=8) #8 arcsec
            ext = int(sci.window / 2)

            if cutout is not None:
                plt.subplot(rows,cols,index)
                #plt.axis('equal')
                plt.imshow(cutout.data, origin='lower', interpolation='nearest', cmap=plt.get_cmap('gray'),
                           vmin=sci.vmin, vmax=sci.vmax, extent= [-ext,ext,-ext,ext])
                plt.title(i['instrument']+" "+i['filter'])
                #todo: iterate over all fields for this image and print values
                if df is not None:
                   # print(len(df))
                   # print(df)
                    s = ""
                    for f,l in zip(i['cols'],i['labels']):
                        #print (f)
                        v = df[f].values[0]
                        s = s + l + " = " + str(v) + "\n"
                    plt.xlabel(s,multialignment='left')

        plt.tight_layout()
        plt.show()

#######################################
#end class CANDELS_EGS_Stefanon_2016
#######################################



class dummy_cat(Catalog):
#RA,Dec in decimal degrees

    #class variables
    CatalogLocation = "nowhere"
    Name = "Dummy Cat"


    def __init__(self):
    #    super(dummy_cat, self).__init__()
        self.dataframe_of_bid_targets = None
        self.read_catalog()

    @classmethod
    def read_catalog(cls):
        pass

    @classmethod
    def coordinate_range(cls,echo=False):
        if echo:
            msg = "RA (%f, %f)" % (cls.RA_min, cls.RA_max) + "Dec(%f, %f)" % (cls.Dec_min, cls.Dec_max)
            print( msg )
        log.debug(cls.Name + " Simple Coordinate Box: " + msg )
        return (cls.RA_min, cls.RA_max, cls.Dec_min, cls.Dec_max)

    def build_list_of_bid_targets(self,ra,dec,error):
       return 0,None




