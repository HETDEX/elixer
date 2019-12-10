from __future__ import print_function


"""
2019-09-19
This HSC imaging data set is consist of HSC r-band data in the HETDEX spring field
taken in the semesters of S15A+S17A+S18A (= a total of 2 nights).
This amounts to 150 deg2 (= 2 times wider than the S15A HSC data available in HDR1).


The sky distribution of the reduced data is shown as green filled circles in 
/data/04094/mshiro/HSC/dr2/check_tract.png (/work/03649/hetdex/hdr2/imaging/hsc/check_tract.png)

The seeing size is =0.6-1.0 arcsec in FWHM.
The depth is 5 sigma limiting magnitude of r = 25.5 mag for 2''.0 aperture.

***note: ELiXer calculate corners are slightly different than HSC reported corners (ELiXer are
slightly wider ... assume HSC does not include the very edge/perimeter that is in the image). This
does not matter as ELiXer only uses this to find the correct tile and that is the tile(s) for which
the provided target RA, Dec are maximally inside the tile).

"""

try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_base
    from elixer import match_summary
    from elixer import line_prob
    from elixer import hsc_meta
    from elixer import utilities
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import line_prob
    import hsc_meta
    import utilities

import os.path as op
import copy
import io

import matplotlib
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import astropy.table

#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field



def hsc_count_to_mag(count,cutout=None,headers=None):
   # return 999.9

    #We can convert the counts into flux
    # with a keyword in the header of the imaging data;
    # FLUXMAG0=     63095734448.0194
    #
    # Because the HSC pipeline uses the zeropoint value (corresponds to 27 mag) to all filters,
    # we can convert the counts values in the R-band imaging data as follows:
    # -2.5*log(flux_R) -48.6 = -2.5*log(count_R / FLUXMAG0)
    # --> flux_R = count_R / ( 10^(30.24) )

    #note: zero point is 27 mag, tiles are no longer different?
#release 2:
#The magnitude zero point is R_ZP = 27.0 mag.
# You can convert the count values in the imaging data into flux as follows:
# -2.5*log(flux_R) -48.6 = -2.5*log(count_R) + R_ZP
# --> flux_R = count_R / (10^(30.24))

    fluxmag0 = None
    try:
        if 'FLUXMAG0' in headers[0]:
            fluxmag0 = float(headers[0]['FLUXMAG0'])
    except:
        fluxmag0 = None

    if count is not None:
        if count > 0:
            if fluxmag0 is not None:
                return -2.5 * np.log10(count/fluxmag0) #+ 48.6
            else:
                return -2.5 * np.log10(count) + 27.0

            return
        else:
            return 99.9  # need a better floor

class HSC(cat_base.Catalog):#Hyper Suprime Cam
    # class variables
    HSC_BASE_PATH = G.HSC_BASE_PATH
    HSC_CAT_PATH = G.HSC_CAT_PATH
    HSC_IMAGE_PATH = G.HSC_IMAGE_PATH

    mean_FWHM = 1.0 #average 0.6 to 1.0

    CONT_EST_BASE = None

    df = None
    loaded_tracts = []

    MainCatalog = None #there is no Main Catalog ... must load individual catalog tracts
    Name = "HyperSuprimeCam"

    Image_Coord_Range = hsc_meta.Image_Coord_Range
    Tile_Dict = hsc_meta.HSC_META_DICT
    Filters = ['r'] #case is important ... needs to be lowercase
    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    WCS_Manual = False

    AstroTable = None

    ##----------------------------
    ## Catalog Format
    ##----------------------------

    # 1   unique IDs
    # 2   position in X [pixel]
    # 3   position in Y [pixel]
    # 4   position in RA [degree]
    # 5   position in Dec [degree]

    # 6   flux within 3.000000-pixel aperture
    # 7   1-sigma flux uncertainty
    # 8   General Failure Flag
    # 9   magnitude within 3.000000-pixel aperture
    # 10  1-sigma magnitude uncertainty

    # 11  flux within 4.500000-pixel aperture
    # 12  1-sigma flux uncertainty
    # 13  General Failure Flag
    # 14  magnitude within 4.500000-pixel aperture
    # 15  1-sigma magnitude uncertainty

    # 16  flux within 6.000000-pixel aperture
    # 17  1-sigma flux uncertainty
    # 18  General Failure Flag
    # 19  magnitude within 6.000000-pixel aperture
    # 20  1-sigma magnitude uncertainty

    # 21  flux within 9.000000-pixel aperture
    # 22  1-sigma flux uncertainty
    # 23  General Failure Flag
    # 24  magnitude within 9.000000-pixel aperture
    # 25  1-sigma magnitude uncertainty

    # 26  flux within 12.000000-pixel aperture
    # 27  1-sigma flux uncertainty
    # 28  General Failure Flag
    # 29  magnitude within 12.000000-pixel aperture
    # 30  1-sigma magnitude uncertainty

    # 31  flux within 17.000000-pixel aperture
    # 32  1-sigma flux uncertainty
    # 33  General Failure Flag
    # 34  magnitude within 17.000000-pixel aperture
    # 35  1-sigma magnitude uncertainty

    # 36  flux within 25.000000-pixel aperture
    # 37  1-sigma flux uncertainty
    # 38  General Failure Flag
    # 39  magnitude within 25.000000-pixel aperture
    # 40  1-sigma magnitude uncertainty

    # 41  flux within 35.000000-pixel aperture
    # 42  1-sigma flux uncertainty
    # 43  General Failure Flag
    # 44  magnitude within 35.000000-pixel aperture
    # 45  1-sigma magnitude uncertainty

    # 46  flux within 50.000000-pixel aperture'
    # 47  1-sigma flux uncertainty
    # 48  General Failure Flag
    # 49  magnitude within 50.000000-pixel aperture
    # 50  1-sigma magnitude uncertainty

    # 51  flux within 70.000000-pixel aperture
    # 52  1-sigma flux uncertainty
    # 53  General Failure Flag
    # 54  magnitude within 70.000000-pixel aperture
    # 55  1-sigma magnitude uncertainty

    # 56  flux derived from linear least-squares fit of PSF model
    # 57  1-sigma flux uncertainty
    # 58  General Failure Flag
    # 59  magnitude derived from linear least-squares fit of PSF model
    # 60  1-sigma magnitude uncertainty

    # 61  convolved Kron flux: seeing 3.500000
    # 62  1-sigma flux uncertainty
    # 63  convolved Kron flux failed: seeing 3.500000
    # 64  convolved magnitude: seeing 3.500000
    # 65  1-sigma magnitude uncertainty

    # 66  convolved Kron flux: seeing 5.000000
    # 67  1-sigma flux uncertainty
    # 68  convolved Kron flux failed: seeing 5.000000
    # 69  convolved magnitude: seeing 5.00000
    # 70  1-sigma magnitude uncertainty

    # 71  convolved Kron flux: seeing 6.500000
    # 72  1-sigma flux uncertainty
    # 73  convolved Kron flux failed: seeing 6.500000
    # 74  convolved magnitude: seeing 6.500000
    # 75  1-sigma magnitude uncertainty

    # 76  convolved Kron flux: seeing 8.000000
    # 77  1-sigma flux uncertainty
    # 78  convolved Kron flux failed: seeing 8.000000
    # 79  convolved magnitude: seeing 8.000000
    # 80  1-sigma magnitude uncertainty

    # 81  flux from the final cmodel fit
    # 82  flux uncertainty from the final cmodel fit
    # 83  flag set if the final cmodel fit (or any previous fit) failed
    # 84  magnitude from the final cmodel fit
    # 85  magnitude uncertainty from the final cmodel fit

    # 86  Number of children this object has (defaults to 0)
    # 87  Source is outside usable exposure region (masked EDGE or NO_DATA)
    # 88  Interpolated pixel in the Source center
    # 89  Saturated pixel in the Source center
    # 90  Cosmic ray in the Source center

    # 91  Bad pixel in the Source footprint
    # 92  Source center is close to BRIGHT_OBJECT pixels
    # 93  Source footprint includes BRIGHT_OBJECT pixels
    # 94  General Failure Flag

    # 95  true if source is in the inner region of a coadd tract
    # 96  true if source is in the inner region of a coadd patch
    # 97  Number of images contributing at center, not including anyclipping
    # 98  original seeing (Gaussian sigma) at position [pixel]

    BidCols = [
        'sourceID',
        'X', # on fits
        'Y', # on fits
        'RA',
        'Dec', #reanamed 'DEC'

        'flux3.0',
        'flux3.0_err',
        'flux3.0_flags',#("False" means no problems)
        'mag3.0',
        'mag3.0_err',

        'flux4.5',
        'flux4.5_err',
        'flux4.5_flags',  # ("False" means no problems)
        'mag4.5',
        'mag4.5_err',

        'flux6.0',
        'flux6.0_err',
        'flux6.0_flags',  # ("False" means no problems)
        'mag6.0',
        'mag6.0_err',

        'flux9.0',
        'flux9.0_err',
        'flux9.0_flags',  # ("False" means no problems)
        'mag9.0',
        'mag9.0_err',

        'flux12.0',
        'flux12.0_err',
        'flux12.0_flags',  # ("False" means no problems)
        'mag12.0',
        'mag12.0_err',

        'flux17.0',
        'flux17.0_err',
        'flux17.0_flags',  # ("False" means no problems)
        'mag17.0',
        'mag17.0_err',

        'flux25.0',
        'flux25.0_err',
        'flux25.0_flags',  # ("False" means no problems)
        'mag25.0',
        'mag25.0_err',

        'flux35.0',
        'flux35.0_err',
        'flux35.0_flags',  # ("False" means no problems)
        'mag35.0',
        'mag35.0_err',

        'flux50.0',
        'flux50.0_err',
        'flux50.0_flags',  # ("False" means no problems)
        'mag50.0',
        'mag50.0_err',

        'flux70.0',
        'flux70.0_err',
        'flux70.0_flags',  # ("False" means no problems)
        'mag70.0',
        'mag70.0_err',

        'fluxlsq',
        'fluxlsq_err',
        'fluxlsq_flags',  # ("False" means no problems)
        'maglsq',
        'maglsq_err',

        'flux.kron3.5',
        'flux.kron3.5_err',
        'flux.kron3.5_flags',# ("False" means no problems)
        'mag.kron3.5',
        'mag.kron3.5_err',

        'flux.kron5.0',
        'flux.kron5.0_err',
        'flux.kron5.0_flags',  # ("False" means no problems)
        'mag.kron5.0',
        'mag.kron5.0_err',

        'flux.kron6.5',
        'flux.kron6.5_err',
        'flux.kron6.5_flags',  # ("False" means no problems)
        'mag.kron6.5',
        'mag.kron6.5_err',

        'flux.kron8.0',
        'flux.kron8.0_err',
        'flux.kron8.0_flags',  # ("False" means no problems)
        'mag.kron8.0',
        'mag.kron8.0_err',

        'flux.cmodel',
        'flux.cmodel_err',
        'flux.cmodel_flags',# ("False" means no problems)
        'mag.cmodel',
        'mag.cmodel_err',

        'children', #86
        'outside', #87
        'interpix_center', #88
        'saturatedpix_center', #89
        'cosmic_center', #90

        'bad_pix', #91
        'near_bright_obj', #92
        'footprint_bright_obj', #93
        'general_flag', #94

        'inner_coadd_tract', #95
        'inner_coadd_patch', #96
        'num_images', #97
        'orig_seeing' #Gaussian Sigma
        ]

    CatalogImages = [] #built in constructor

    def __init__(self):
        super(HSC, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_unique = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0
        self.master_cutout = None
        self.build_catalog_of_images()

    @classmethod
    def read_catalog(cls, catalog_loc=None, name=None,tract=None,position=None):
        """

        :param catalog_loc:
        :param name:
        :param tract: list of string string as the HSC track id, ie. ['16814']
        :param position: a tuple or array with exactly 2 elements as integers 0-9, i.e. (0,1) or (2,8), etc
        :return:
        """

        if name is None:
            name = cls.Name

        fqtract =[] #fully qualified track (as a partial path)

        if (tract is not None) and (len(tract) > 0) and (position is not None) and (len(position) == len(tract)): #should be a list of positions and the same length as tract
            for i in range(len(tract)):
                fqtract.append(op.join(tract[i],"R_P%d_%d.cat" %(position[i][0],position[i][1])))
        else:
            log.warning("Unexpected tract and positions in cat_hsc::read_catalogs: %s, %s" %(str(tract),str(position)))
            return None


        if set(fqtract).issubset(cls.loaded_tracts):
            log.info("Catalog tract (%s) already loaded." %fqtract)
            return cls.df



        #todo: future more than just the R filter if any are ever added
        for t in fqtract:
            if t in cls.loaded_tracts: #skip if already loaded
                continue

            cat_name = t
            cat_loc = op.join(cls.HSC_CAT_PATH, cat_name)
            header = cls.BidCols

            if not op.exists(cat_loc):
                log.error("Cannot load catalog tract for HSC. File does not exist: %s" %cat_loc)

            log.debug("Building " + cls.Name + " " + cat_name + " dataframe...")

            try:
                df = pd.read_csv(cat_loc, names=header,
                                 delim_whitespace=True, header=None, index_col=None, skiprows=0)

                old_names = ['Dec']
                new_names = ['DEC']
                df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

                df['FILTER'] = 'r' #add the FILTER to the dataframe !!! case is important. must be lowercase

                if cls.df is not None:
                    cls.df = pd.concat([cls.df, df])
                else:
                    cls.df = df
                cls.loaded_tracts.append(t)

            except:
                log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
                continue

        return cls.df

    def build_catalog_of_images(self):

        for t in self.Tile_Dict.keys(): #tile is the key (the filename)
            for f in self.Filters:
                self.CatalogImages.append(
                    {'path': op.join(self.HSC_IMAGE_PATH,self.Tile_Dict[t]['tract']),
                     'name': t, #filename is the tilename
                     'tile': t,
                     'pos': self.Tile_Dict[t]['pos'], #the position tuple i.e. (0,3) or (2,8) ... in the name as 03 or 28
                     'filter': f,
                     'instrument': "HSC",
                     'cols': [],
                     'labels': [],
                     'image': None,
                     'expanded': False,
                     'wcs_manual': False,
                     'aperture': self.mean_FWHM * 0.5 + 0.5, #since a radius, half the FWHM + 0.5" for astrometric error
                     'mag_func': hsc_count_to_mag
                     })

    def find_target_tile(self,ra,dec):
        #assumed to have already confirmed this target is at least in coordinate range of this catalog
        #return at most one tile, but maybe more than one tract (for the catalog ... HSC does not completely
        #   overlap the tracts so if multiple tiles are valid, depending on which one is selected, you may
        #   not find matching objects for the associated tract)
        tile = None
        tracts = []
        positions = []
        keys = []
        for k in self.Tile_Dict.keys():
            # don't bother to load if ra, dec not in range
            try:
                if not ((ra >= self.Tile_Dict[k]['RA_min']) and (ra <= self.Tile_Dict[k]['RA_max']) and
                        (dec >= self.Tile_Dict[k]['Dec_min']) and (dec <= self.Tile_Dict[k]['Dec_max'])) :
                    continue
                else:
                    keys.append(k)
            except:
                pass

        if len(keys) == 0: #we're done ... did not find any
            return None, None, None
        elif len(keys) == 1: #found exactly one
            tile = keys[0] #remember tile is a string ... there can be only one
            positions.append(self.Tile_Dict[tile]['pos'])
            tracts.append(self.Tile_Dict[tile]['tract']) #remember, tract is a list (there can be more than one)
        elif len(keys) > 1: #find the best one
            log.info("Multiple overlapping tiles %s. Sub-selecting tile with maximum angular coverage around target." %keys)
            min = 9e9
            #we don't have the actual corners anymore, so just assume a rectangle
            #so there are 2 of each min, max coords. Only need the smallest distance so just sum one
            for k in keys:
                tracts.append(self.Tile_Dict[k]['tract'])
                positions.append(self.Tile_Dict[k]['pos'])
                sqdist = (ra-self.Tile_Dict[k]['RA_min'])**2 + (dec-self.Tile_Dict[k]['Dec_min'])**2 + \
                         (ra-self.Tile_Dict[k]['RA_max'])**2 + (dec-self.Tile_Dict[k]['Dec_max'])**2
                if sqdist < min:
                    min = sqdist
                    tile = k

        else: #really?? len(keys) < 0 : this is just a sanity catch
            log.error("ERROR! len(keys) < 0 in cat_hsc::find_target_tile.")
            return None, None, None

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

        return tile, tracts, positions

    def get_filter_flux(self, df):

        filter_fl = None
        filter_fl_err = None
        filter_flag = None
        mag = None
        mag_bright = None
        mag_faint = None
        filter_str = 'R'

        #filter_fl now reported in counts_R, so ...
        #per HSC documentation, counts / 10**(30.24) = flux in ergs/s/cm2/Hz then *10**23 to get to Jy and *10**9 to nJy
        #so ...( cts / 10**(30.24) ) * 10**23 * 10**9 = cts * 10**(32-30.24) == cts * 10**(1.76)
        cts2njy = 57.54399373 #10**1.76


        method  = None

        try:
            if df['fluxlsq_flags'].values[0]:  # there is a problem
                try:
                    log.debug("HSC fluxlsq flagged. mag = %f" %df["maglsq"].values[0])
                except:
                    pass
                if df['flux3.0_flags'].values[0]:
                    try:
                        log.debug("HSC flux3.0 flagged. mag = %f" % df["mag3.0"].values[0])
                    except:
                        pass
                    if df['flux.cmodel_flags'].values[0]:  # there is a problem
                    #if df['flux.cmodel_flags'].values[0]:  # there is a problem
                        try:
                            log.debug("HSC cmodel flagged. mag = %f" % df["mag.cmodel"].values[0])
                        except:
                            pass
                        log.info("Flux/Mag unreliable due to errors.")
                        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str
                    else:
                        method = 'cmodel'
                else:
                    method = '3.0'
            else:
                method = 'lsq'
        except:
            log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)
            return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

        try:

            if method == 'lsq':# or method == 'kron':
                filter_fl = df['fluxlsq'].values[0]  * cts2njy
                filter_fl_err = df["fluxlsq_err"].values[0] * cts2njy
                mag = df["maglsq"].values[0]
                # mag_bright = mag - df["magerr."+method].values[0]
                mag_faint = df['maglsq_err'].values[0]
                mag_bright = -1 * mag_faint
            elif method == '3.0':
                filter_fl = df['flux3.0'].values[0]  * cts2njy
                filter_fl_err = df["flux3.0_err"].values[0] * cts2njy
                mag = df["mag3.0"].values[0]
                # mag_bright = mag - df["magerr."+method].values[0]
                mag_faint = df['mag3.0_err'].values[0]
                mag_bright = -1 * mag_faint
            else:  # cmodel
                filter_fl = df['flux.cmodel'].values[0] * cts2njy
                filter_fl_err = df['flux.cmodel_err'].values[0] * cts2njy
                mag = df['mag.cmodel'].values[0]
                # mag_bright = mag - df["magerr."+method].values[0]
                mag_faint = df['mag.cmodel_err'].values[0]
                mag_bright = -1 * mag_faint

            # mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl, filter_fl_err)
        except:  # not the EGS df, try the CFHTLS
            log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)

            # it is unclear what unit flux is in (but it is not nJy or uJy), so lets back convert from the magnitude
            # this is used as a continuum estimate

        #with updated HSC data release 2 (Sept.2019) this is not needed
        #filter_fl = self.obs_mag_to_nano_Jy(mag)
        #filter_fl_err = 0.0  # set to 0 so not to be trusted

        try:
            log.debug("HSC selected method = %s , mag = %f" %(method,mag))
        except:
            pass

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str


    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        #even if not None, could be we need a different catalog, so check and append
        tile, tracts, positions = self.find_target_tile(ra,dec)

        if tile is None:
            log.info("Could not locate tile for HSC. Discontinuing search of this catalog.")
            return -1,None,None

        #could be none or could be not loaded yet
        #if self.df is None or not (self.Tile_Dict[tile]['tract'] in self.loaded_tracts):
        if self.df is None or not (set(tracts).issubset(self.loaded_tracts)):
            #self.read_main_catalog()
            #self.read_catalog(tract=self.Tile_Dict[tile]['tract'])
            self.read_catalog(tract=tracts,position=positions)

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
                self.df[  (self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max)
                        & (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)
                        & (self.df['children'] == 0)
                        & (self.df['outside'] == False)
                        & (self.df['interpix_center'] == False)
                        & (self.df['saturatedpix_center'] == False)
                        & (self.df['cosmic_center'] == False)
                        & (self.df['bad_pix'] == False)
                        & (self.df['near_bright_obj'] == False)
                        & (self.df['footprint_bright_obj'] == False)
                        & (self.df['general_flag'] == False)
                        & (self.df['inner_coadd_tract'] == True)
                        & (self.df['inner_coadd_patch'] == True)
                        & (self.df['num_images'] > 2)
                        ].copy()
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
        num_targets, _, _ = self.build_list_of_bid_targets(target_ra, target_dec, error)
        #could be there is no matching tile, if so, the dataframe will be none

        #if (num_targets == 0) or
        if (self.dataframe_of_bid_targets_unique is None):
            return None

        ras = self.dataframe_of_bid_targets_unique.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets_unique.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:
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


    def get_stacked_cutout(self,ra,dec,window):

        stacked_cutout = None
        error = window

        # for a given Tile, iterate over all filters
        tile, tracts, positions = self.find_target_tile(ra, dec)
        if tile is None:
            # problem
            print("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
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
                    i['image'] = science_image.science_image(wcs_manual=wcs_manual,wcs_idx=0,
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
        tile, tracts, positions = self.find_target_tile(ra, dec)

        if tile is None:
            # problem
            print("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        # All on one line now across top of plots
        if G.ZOO:
            title = "Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets_unique), error)
        else:
            title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets_unique), error)

        cont_est = -1
        if target_flux and self.CONT_EST_BASE:
            title += "  Minimum (no match) 3$\sigma$ rest-EW: "
            cont_est = self.CONT_EST_BASE*3
            if cont_est != -1:
                title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A"
            else:
                title += "  LyA = N/A  OII = N/A"



        plt.subplot(gs[0, :])
        text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        ref_exptime = 1.0
        total_adjusted_exptime = 1.0
        bid_colors = self.get_bid_colors(len(bid_ras))
        exptime_cont_est = -1
        index = 0 #images go in positions 1+ (0 is for the fiber positions)

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
                i['image'] = science_image.science_image(wcs_manual=wcs_manual,wcs_idx=0,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            #the filters are in order, use r if g is not there
            if (f == 'r') and (sci.exptime is not None) and (exptime_cont_est == -1):
                exptime_cont_est = sci.exptime

            # the filters are in order, so this will overwrite r
            if (f == 'g') and (sci.exptime is not None):
                exptime_cont_est = sci.exptime

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag, mag_radius,details = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func,return_details=True)

            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            bid_target = None
            cutout_ewr = None
            cutout_ewr_err = None
            cutout_plae = None

            try:  # update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None) and (i['filter'] == 'r'):
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
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag, target_w)
                        try:
                            flux_faint = None
                            flux_bright = None

                            if details['mag_faint'] < 99:
                                flux_faint = self.obs_mag_to_cgs_flux(details['mag_faint'], target_w)

                            if details['mag_bright'] < 99:
                                flux_bright = self.obs_mag_to_cgs_flux(details['mag_bright'], target_w)

                            if flux_bright and flux_faint:
                                bid_target.bid_flux_est_cgs_unc = max((bid_target.bid_flux_est_cgs - flux_faint),
                                                                      (flux_bright -bid_target.bid_flux_est_cgs))
                            elif flux_bright:
                                bid_target.bid_flux_est_cgs_unc = flux_bright -bid_target.bid_flux_est_cgs

                        except:
                            pass


                    else:
                        bid_target.bid_flux_est_cgs = cont_est

                    try:
                        bid_target.bid_mag_err_bright = mag - details['mag_bright']
                        bid_target.bid_mag_err_faint =  details['mag_faint'] - mag
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

                    # build EW error from lineFlux_err and aperture estimate error
                    ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                    try:
                        ew_obs_err = abs(ew_obs * np.sqrt(
                            (lineFlux_err / target_flux) ** 2 +
                            (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                    except:
                        ew_obs_err = 0.

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                        line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                                           ew_obs=ew_obs,
                                           lineFlux_err=lineFlux_err,
                                           ew_obs_err=ew_obs_err,
                                           c_obs=None, which_color=None, addl_fluxes=addl_flux,
                                           addl_wavelengths=addl_waves, addl_errors=addl_ferr, sky_area=None,
                                           cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                                           sigma=None,estimate_error=True)

                    try:
                        if plae_errors:
                            bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
                            bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
                    except:
                        pass

                    cutout_plae = bid_target.p_lae_oii_ratio
                    cutout_ewr = ew_obs / (1. + target_w / G.LyA_rest)
                    cutout_ewr_err = ew_obs_err / (1. + target_w / G.LyA_rest)
                    if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                        text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)" % (bid_target.p_lae_oii_ratio,i['filter']))
                        # text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g [%0.4g:%0.4g] (%s)"
                        #               %(utilities.saferound(bid_target.p_lae_oii_ratio,3),
                        #                 utilities.saferound(bid_target.p_lae_oii_ratio_min,3),
                        #                 utilities.saferound(bid_target.p_lae_oii_ratio_max,3),i['filter']))

                    cat_match.add_bid_target(bid_target)
                    try:  # no downstream edits so they can both point to same bid_target
                        detobj.bid_target_list.append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to detobj.", exc_info=True)
            except:
                log.debug('Could not build exact location photometry info.', exc_info=True)



            if cutout is not None:  # construct master cutout

                # 1st cutout might not be what we want for the master (could be a summary image from elsewhere)
                if self.master_cutout:
                    if self.master_cutout.shape != cutout.shape:
                        del self.master_cutout
                        self.master_cutout = None

                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True)
                    if sci.exptime:
                        ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    try:
                        self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
                    except:
                        log.warning("Unexpected exception.", exc_info=True)

                _ = plt.subplot(gs[1:, index])

                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])

                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                #plt.plot(0, 0, "r+")
                self.add_zero_position(plt)

                if pix_counts is not None:
                    details['catalog_name'] = self.name
                    details['filter_name'] = f
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
                    if details['sep_objects'] is not None:
                        self.add_elliptical_aperture_positions(plt,details['sep_objects'],details['sep_obj_idx'],
                                                               mag,cx,cy,cutout_ewr,cutout_plae)
                    else:
                        self.add_aperture_position(plt,mag_radius,mag,cx,cy,cutout_ewr,cutout_plae)


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


        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()

            return None
        else:
            self.master_cutout.data /= total_adjusted_exptime

        plt.subplot(gs[1:, 0])

        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)
        #self.add_zero_position(plt)
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

            plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2, height=ry * 2,
                                              angle=0, color='red', fill=False,linewidth=3))

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
                   "1-p(rand)\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "mag\n\n"
        else:
            text = "Separation\n" + \
                   "1-p(rand)\n" + \
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
            try: #DO NOT WANT _unique as that has wiped out the filters
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
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
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

                text += "N/A\nN/A\n"  #dont have specz or photoz for HSC

                #todo: add flux (cont est)
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
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,target_w) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, target_w)
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
                        #

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

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                                line_prob.prob_LAE(wl_obs=target_w,
                                                   lineFlux=target_flux,
                                                   ew_obs=ew_obs,
                                                   lineFlux_err=lineFlux_err,
                                                   ew_obs_err=ew_obs_err,
                                                   c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                                                   addl_fluxes=addl_flux, addl_errors=addl_ferr, sky_area=None,
                                                   cosmo=None, lae_priors=None,
                                                   ew_case=None, W_0=None,
                                                   z_OII=None, sigma=None,estimate_error=True)
                            #dfx = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                            #                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

                            try:
                                if plae_errors:
                                    bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
                                    bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
                            except:
                                pass

                            try:
                                bid_target.add_filter('HSC','R',filter_fl_cgs,filter_fl_err)
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
                    text = text + "%0.2f(%0.2f,%0.2f)\n" % (filter_mag, filter_mag_bright, filter_mag_faint)
                except:
                    log.warning("Magnitude info is none: mag(%s), mag_bright(%s), mag_faint(%s)"
                                % (filter_mag, filter_mag_bright, filter_mag_faint))
                    text += "No mag info\n"

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    text += "%0.4g\n" % ( utilities.saferound(bid_target.p_lae_oii_ratio,3))
                    # text += "%0.4g [%0.4g:%0.4g]\n" % ( utilities.saferound(bid_target.p_lae_oii_ratio,3),
                    #                                       utilities.saferound(bid_target.p_lae_oii_ratio_min,3),
                    #                                       utilities.saferound(bid_target.p_lae_oii_ratio_max,3))
                else:
                    text += "\n"
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

    def get_single_cutout(self, ra, dec, window, catalog_image,aperture=None):


        d = {'cutout':None,
             'hdu':None,
             'path':None,
             'filter':catalog_image['filter'],
             'instrument':catalog_image['instrument'],
             'mag':None,
             'aperture':None,
             'ap_center':None}

        try:
            wcs_manual = catalog_image['wcs_manual']
            mag_func = catalog_image['mag_func']
        except:
            wcs_manual = self.WCS_Manual
            mag_func = None

        try:
            if catalog_image['image'] is None:
                catalog_image['image'] =  science_image.science_image(wcs_manual=wcs_manual,wcs_idx=0,
                                                        image_location=op.join(catalog_image['path'],
                                                                        catalog_image['name']))
                catalog_image['image'].catalog_name = catalog_image['name']
                catalog_image['image'].filter_name = catalog_image['filter']

            sci = catalog_image['image']

            if sci.hdulist is None:
                sci.load_image(wcs_manual=wcs_manual)

            d['path'] = sci.image_location
            d['hdu'] = sci.headers

            # to here, window is in degrees so ...
            window = 3600. * window

            cutout,pix_counts, mag, mag_radius = sci.get_cutout(ra, dec, error=window, window=window, aperture=aperture,
                                             mag_func=mag_func,copy=True)
            # don't need pix_counts or mag, etc here, so don't pass aperture or mag_func

            if cutout is not None:  # construct master cutout
                d['cutout'] = cutout
                if (mag is not None) and (mag < 999):
                    d['mag'] = mag
                    d['aperture'] = mag_radius
                    d['ap_center'] = (sci.last_x0_center, sci.last_y0_center)
        except:
            log.error("Error in get_single_cutout.", exc_info=True)

        return d

    def get_cutouts(self,ra,dec,window,aperture=None,filter=None,first=False):
        l = list()

        tile, tracts, positions = self.find_target_tile(ra, dec)

        if tile is None:
            # problem
            log.error("No appropriate tile found in HSC for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        # try:
        #     #cat_filters = list(set([x['filter'].lower() for x in self.CatalogImages]))
        #     cat_filters = list(dict((x['filter'], {}) for x in self.CatalogImages).keys())
        # except:
        #     cat_filters = None

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

                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == tile)))]
                    if i is not None:
                        cutout = self.get_single_cutout(ra, dec, window, i, aperture)

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
                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == tile)))]
                except:
                    i = None

                if i is None:
                    continue

                l.append(self.get_single_cutout(ra,dec,window,i,aperture))

        return l
