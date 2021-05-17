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
    from elixer import spectrum_utilities as SU
    from elixer import cat_kpno
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import line_prob
    import hsc_meta
    import utilities
    import spectrum_utilities as SU
    import cat_kpno

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
log.setlevel(G.LOG_LEVEL)

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

    INCLUDE_KPNO_G = True

    #Shiro's DR3 document now says 25.5, website says 26.1 (27 for deep, 28 for ultra deep)
    #I have been nudging by ~ 0.5 mag (since was 5sigma, and I am using my own apertures)
    #originally had 26.5 (but that may have been a mis-copy for wide-g band at 26.5)
    #and this is a final floor limit
    #Updated 2020-12-17 ... was using data from HSC-SSP (10 minute exp) vs HETDEX specific
    #survey from HSC which is 6 min. and 25.5 is correct, so set full limit to 26.0
    # other HSC may be as deep as 27.2
    MAG_LIMIT = 26.2 #(my overall average)

    mean_FWHM = 1.0 #average 0.6 to 1.0

    CONT_EST_BASE = None

    df = None
    loaded_tracts = []

    MainCatalog = None #there is no Main Catalog ... must load individual catalog tracts
    Name = "HyperSuprimeCam"

    Image_Coord_Range = hsc_meta.Image_Coord_Range
    Tile_Dict = hsc_meta.HSC_META_DICT
    #correct the basepaths
    for k in Tile_Dict.keys():
        Tile_Dict[k]['path'] = op.join(G.HSC_IMAGE_PATH,Tile_Dict[k]['tract'],op.basename(Tile_Dict[k]['path']))


    Filters = ['r'] #case is important ... needs to be lowercase
    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    WCS_Manual = False

    AstroTable = None

    #Masks (bitmapped) (from *_mask.fits headers)
    MP_BAD = 0  #2**0
    MP_SAT = 2  #2**1
    MP_INTRP = 4  #2**2
    MP_CR = 8
    MP_EDGE = 16
    MP_DETECTED = 32
    MP_DETECTED_NEGATIVE = 64
    MP_SUSPECT = 128
    MP_NO_DATA = 256
    MP_BRIGHT_OBJECT = 512
    MP_CROSSTALK = 1024
    MP_NOT_DEBLENDED = 2048
    MP_UNMASKEDNAN = 4096
    MP_REJECTED = 8192
    MP_CLIPPED = 16384
    MP_SENSOR_EDGE = 32768
    MP_INEXACT_PSF = 65536  #2**16
    MASK_LENGTH = 17

    #
    # Notice: sizes in pixels at 0.168" and are diameters
    # so 17.0 is 17 pixel diameter aperture (~ 2.86" diamter or 1.4" radius (about 2 touching HETDEX fibers))
    #

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


    try: #old columns
        if G.HSC_S15A:
            BidCols = [
                'sourceID',
                'X', # on fits
                'Y', # on fits
                'RA',
                'Dec', #reanamed 'DEC'
                'flux.psf',
                'flux.psf.err',
                'flux.psf.flags',#("False" means no problems)
                'mag.psf',
                'magerr.psf',
                'flux.kron',
                'flux.kron.err',
                'flux.kron.flags',# ("False" means no problems)
                'mag.kron',
                'magerr.kron',
                'cmodel.flux',
                'cmodel.flux.err',
                'cmodel.flux.flags',# ("False" means no problems)
                'cmodel.mag',
                'cmodel.magerr'
                ]
    except:
        pass


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

        s15a = False
        try:
            if G.HSC_S15A:
                s15a = True
        except:
            pass

        fqtract =[] #fully qualified track (as a partial path)

        if (tract is not None) and (len(tract) > 0) and (position is not None) and (len(position) == len(tract)): #should be a list of positions and the same length as tract
            if s15a:
                for i in range(len(tract)):
                  #cat_name = 'R_' + t + ".dat"
                  fqtract.append(op.join("R_%s.dat" % (tract[i])))
            else:
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
                continue

            log.debug("Building " + cls.Name + " " + cat_name + " dataframe...")

            try:
                df = pd.read_csv(cat_loc, names=header,
                                 delim_whitespace=True, header=None, index_col=False, skiprows=0)

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
                try:
                    if G.HSC_S15A:
                        path = self.HSC_IMAGE_PATH
                        wcs_manual = True
                        toks = t.split('-')
                        t1,t2 = toks[4][0],toks[4][1]
                        name = toks[0] + "-" + toks[1] + "-" + toks[2] + "-" + toks[3] + "-" + t1 + "," + t2 +".fits"
                    else:
                        path = op.join(self.HSC_IMAGE_PATH,self.Tile_Dict[t]['tract'])
                        name = t
                        wcs_manual = False
                except:
                    path = op.join(self.HSC_IMAGE_PATH,self.Tile_Dict[t]['tract'])
                    name = t
                    wcs_manual = False

                self.CatalogImages.append(
                    {'path': path,
                     'name': name, #filename is the tilename
                     'tile': t,
                     'pos': self.Tile_Dict[t]['pos'], #the position tuple i.e. (0,3) or (2,8) ... in the name as 03 or 28
                     'filter': f,
                     'instrument': "HSC",
                     'cols': [],
                     'labels': [],
                     'image': None,
                     'expanded': False,
                     'wcs_manual': wcs_manual,
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
            min_dist = 9e9
            max_dist = 0
            #we don't have the actual corners anymore, so just assume a rectangle
            #so there are 2 of each min, max coords. Only need the smallest distance so just sum one
            for k in keys:
                tracts.append(self.Tile_Dict[k]['tract'])
                positions.append(self.Tile_Dict[k]['pos'])

                # sqdist = (ra-self.Tile_Dict[k]['RA_min'])**2 + (dec-self.Tile_Dict[k]['Dec_min'])**2 + \
                #          (ra-self.Tile_Dict[k]['RA_max'])**2 + (dec-self.Tile_Dict[k]['Dec_max'])**2
                #
                # if sqdist < min_dist and op.exists(self.Tile_Dict[k]['path']):
                #     min_dist = sqdist
                #     tile = k

                #should not be negative, but could be?
                #in any case, the min is the smallest distance to an edge in RA and Dec
                inside_ra = min((ra-self.Tile_Dict[k]['RA_min']),(self.Tile_Dict[k]['RA_max']-ra))
                inside_dec = min((dec-self.Tile_Dict[k]['Dec_min']),(self.Tile_Dict[k]['Dec_max']-dec))

                edge_dist = min(inside_dec,inside_ra)
                #we want the tile with the largest minium edge distance

                if edge_dist > max_dist and op.exists(self.Tile_Dict[k]['path']):
                    max_dist = edge_dist
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


    def get_mask_cutout(self,tile,ra,dec,error):
        """
        Given an image tile, get the corresponding mask tile

        :param tile:
        :return:
        """

        mask_cutout = None
        try:
            #modify name for mask
            mask_tile_name = tile.rstrip(".fits") +"_mask.fits"
            # Find in Tile Dict for the path
            path = op.join(G.HSC_IMAGE_PATH,self.Tile_Dict[tile]['tract'],op.basename(self.Tile_Dict[tile]['path']))
            path = path.replace('image_tract_patch','mask_tract_patch')
            path = path.replace(tile,mask_tile_name)

            #now get the fits image cutout (don't use get_cutout as that is strictly for images)
            mask_image = science_image.science_image(wcs_manual=self.WCS_Manual,wcs_idx=0, image_location=path)

            mask_cutout = mask_image.get_mask_cutout(ra,dec,error,path)

        except:
            log.info(f"Could not get mask cutout for tile ({tile})",exc_info=True)

        return mask_cutout


    def get_filter_flux_s15a(self, df):

        filter_fl = None
        filter_fl_err = None
        filter_flag = None
        mag = None
        mag_bright = None
        mag_faint = None
        filter_str = 'R'

        method  = None

        try:
            if df['flux.psf.flags'].values[0]:  # there is a problem
                if df['flux.kron.flags'].values[0]:  # there is a problem
                    if df['cmodel.flux.flags'].values[0]:  # there is a problem
                        log.info("Flux/Mag unreliable due to errors.")
                        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str
                    else:
                        method = 'cmodel'
                else:
                    method = 'kron'
            else:
                method = 'psf'
        except:
            log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)
            return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

        try:

            if method == 'psf' or method == 'kron':
                filter_fl = df["flux." + method].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                filter_fl_err = df["flux." + method + ".err"].values[0]
                mag = df["mag." + method].values[0]
                # mag_bright = mag - df["magerr."+method].values[0]
                mag_faint = df["magerr." + method].values[0]
                mag_bright = -1 * mag_faint
            else:  # cmodel
                filter_fl = df[method + ".flux"].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                filter_fl_err = df[method + ".flux.err"].values[0]
                mag = df[method + ".mag"].values[0]
                # mag_bright = mag - df["magerr."+method].values[0]
                mag_faint = df[method + ".magerr"].values[0]
                mag_bright = -1 * mag_faint

            # mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl, filter_fl_err)
        except:  # not the EGS df, try the CFHTLS
            log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)

            # it is unclear what unit flux is in (but it is not nJy or uJy), so lets back convert from the magnitude
            # this is used as a continuum estimate

        filter_fl = self.obs_mag_to_nano_Jy(mag)
        filter_fl_err = 0.0  # set to 0 so not to be trusted

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str


    # def old_get_filter_flux(self, df):
    #
    #     try:
    #         if G.HSC_S15A:
    #             return self.get_filter_flux_s15a(df)
    #     except:
    #         pass
    #
    #     filter_fl = None
    #     filter_fl_err = None
    #     filter_flag = None
    #     mag = None
    #     mag_bright = None
    #     mag_faint = None
    #     filter_str = 'R'
    #
    #     #filter_fl now reported in counts_R, so ...
    #     #per HSC documentation, counts / 10**(30.24) = flux in ergs/s/cm2/Hz then *10**23 to get to Jy and *10**9 to nJy
    #     #so ...( cts / 10**(30.24) ) * 10**23 * 10**9 = cts * 10**(32-30.24) == cts * 10**(1.76)
    #     cts2njy = 57.54399373 #10**1.76
    #
    #
    #     method  = None
    #
    #
    #     # NOTICE: HSC apertures defined by num of pixels at 0.168 "/pix scale
    #     # so flux3.0 is not 3" but 3 pixels or ~ 0.5"
    #     # so for 3", want 17.8" or flux17 (that is about a 3" DIAMETER aperture)
    #     try:
    #         if df['fluxlsq_flags'].values[0]:  # there is a problem
    #             try:
    #                 log.debug("HSC fluxlsq flagged. mag = %f" %df["maglsq"].values[0])
    #             except:
    #                 pass
    #             if df['flux17.0_flags'].values[0]:
    #                 try:
    #                     log.debug("HSC flux17.0 flagged. mag = %f" % df["mag17.0"].values[0])
    #                 except:
    #                     pass
    #                 if df['flux.cmodel_flags'].values[0]:  # there is a problem
    #                 #if df['flux.cmodel_flags'].values[0]:  # there is a problem
    #                     try:
    #                         log.debug("HSC cmodel flagged. mag = %f" % df["mag.cmodel"].values[0])
    #                     except:
    #                         pass
    #                     log.info("Flux/Mag unreliable due to errors.")
    #                     return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str
    #                 else:
    #                     method = 'cmodel'
    #             else:
    #                 method = '17.0'  #17 pixels, ~ 3"
    #         else:
    #             method = 'lsq'
    #     except:
    #         log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)
    #         return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str
    #
    #     try:
    #
    #         if method == 'lsq':# or method == 'kron':
    #             filter_fl = df['fluxlsq'].values[0]  * cts2njy
    #             filter_fl_err = df["fluxlsq_err"].values[0] * cts2njy
    #             mag = df["maglsq"].values[0]
    #             # mag_bright = mag - df["magerr."+method].values[0]
    #             mag_faint = df['maglsq_err'].values[0]
    #             mag_bright = -1 * mag_faint
    #         elif method == '17.0': #17 pixels ~ 3"
    #             filter_fl = df['flux17.0'].values[0]  * cts2njy
    #             filter_fl_err = df["flux17.0_err"].values[0] * cts2njy
    #             mag = df["mag17.0"].values[0]
    #             # mag_bright = mag - df["magerr."+method].values[0]
    #             mag_faint = df['mag17.0_err'].values[0]
    #             mag_bright = -1 * mag_faint
    #         else:  # cmodel
    #             filter_fl = df['flux.cmodel'].values[0] * cts2njy
    #             filter_fl_err = df['flux.cmodel_err'].values[0] * cts2njy
    #             mag = df['mag.cmodel'].values[0]
    #             # mag_bright = mag - df["magerr."+method].values[0]
    #             mag_faint = df['mag.cmodel_err'].values[0]
    #             mag_bright = -1 * mag_faint
    #
    #         # mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl, filter_fl_err)
    #     except:  # not the EGS df, try the CFHTLS
    #         log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)
    #
    #         # it is unclear what unit flux is in (but it is not nJy or uJy), so lets back convert from the magnitude
    #         # this is used as a continuum estimate
    #
    #     #with updated HSC data release 2 (Sept.2019) this is not needed
    #     #filter_fl = self.obs_mag_to_nano_Jy(mag)
    #     #filter_fl_err = 0.0  # set to 0 so not to be trusted
    #
    #     try:
    #         log.debug("HSC selected method = %s , mag = %f" %(method,mag))
    #     except:
    #         pass
    #
    #     return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str
    #
    #

    #reworked as average of 3" (17pix), lsq and model mags
    def get_filter_flux(self, df):

        try:
            if G.HSC_S15A:
                return self.get_filter_flux_s15a(df)
        except:
            pass

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

        flux_list = []
        flux_err_list = []
        # mag_list = []
        # mag_err_list = []
        methods = []

        # NOTICE: HSC apertures defined by num of pixels at 0.168 "/pix scale
        # so flux3.0 is not 3" but 3 pixels or ~ 0.5"
        # so for 3", want 17.8" or flux17 (that is about a 3" DIAMETER aperture)
        # flux35.0 would be roughtly 3" RADIUS aperture
        #

        try:
            if df['fluxlsq_flags'].values[0] == 0: #no flags, is good:
                #still check for nan
                if not (np.isnan(df['fluxlsq'].values[0]) or df['fluxlsq'].values[0] < 0) :
                    flux_list.append(df['fluxlsq'].values[0] * cts2njy)
                    flux_err_list.append(df["fluxlsq_err"].values[0] * cts2njy)
                    #mag_list.append(df["maglsq"].values[0])
                    #mag_err_list.append(df['maglsq_err'].values[0])
                    methods.append('lsq')

            #about 3" DIAMETER (1.5" RADIUS ... pretty common for the ELiXer Aperture)
            if df['flux17.0_flags'].values[0] == 0: #no flags, is good:
                if not (np.isnan(df['flux17.0'].values[0]) or df['flux17.0'].values[0] < 0) :
                    flux_list.append(df['flux17.0'].values[0] * cts2njy)
                    flux_err_list.append(df["flux17.0_err"].values[0] * cts2njy)
                    # mag_list.append(df["mag17.0"].values[0])
                    # mag_err_list.append(df['mag17.0_err'].values[0])
                    methods.append('flux17.0')

            #todo: do I actually want to go out this wide?
            # this would cover around 15 HETDEX fibers (3" radius, 6" diameter)
            # or most of the default search box
            # if df['flux35.0_flags'].values[0] == 0: #no flags, is good:
            #     flux_list.append(df['flux35.0'].values[0] * cts2njy)
            #     flux_err_list.append(df["flux35.0_err"].values[0] * cts2njy)
            #     # mag_list.append(df["mag35.0"].values[0])
            #     # mag_err_list.append(df['mag35.0_err'].values[0])
            #     methods.append('flux35.0')

            if df['flux.cmodel_flags'].values[0] == 0: #no flags, is good:
                if not (np.isnan(df['flux.cmodel'].values[0]) or df['flux.cmodel'].values[0] < 0) :
                    flux_list.append(df['flux.cmodel'].values[0] * cts2njy)
                    flux_err_list.append(df['flux.cmodel_err'].values[0] * cts2njy)
                    # mag_list.append(df['mag.cmodel'].values[0])
                    # mag_err_list.append(df['mag.cmodel_err'].values[0])
                    methods.append('cmodel')

            if df['flux.kron3.5_flags'].values[0] == 0: #no flags, is good:
                if not (np.isnan(df['flux.kron3.5'].values[0]) or df['flux.kron3.5'].values[0] < 0) :
                    flux_list.append(df['flux.kron3.5'].values[0] * cts2njy)
                    flux_err_list.append(df['flux.kron3.5_err'].values[0] * cts2njy)
                    # mag_list.append(df['mag.cmodel'].values[0])
                    # mag_err_list.append(df['mag.cmodel_err'].values[0])
                    methods.append('kron3.5')

            #flux_list = np.array(flux_list)
            #should not be any nan but, just in case
            flux_list = np.ma.masked_where(np.isnan(flux_list), flux_list, copy=False)
            flux_err_list = np.array(flux_err_list)
            avg_method = "none"

            if len(flux_list) == 0:
                log.info("Unable to get non-flagged catalog flux for HSC object.")
            else:
                if len(flux_list) < 3:
                    filter_fl = np.mean(flux_list)
                    filter_fl_err = np.sqrt(np.sum(flux_err_list*flux_err_list))
                    avg_method = "mean"
                else:
                    try: #weighted biweight
                        filter_fl = SU.weighted_biweight.biweight_location_errors(flux_list, errors=flux_err_list)
                        filter_fl_err = SU.weighted_biweight.biweight_scale(flux_list) / np.sqrt(len(flux_err_list))
                        avg_method = "weighted biweight"
                    except:
                        try: #standard biweigth
                            filter_fl = SU.weighted_biweight.biweight_location(flux_list)
                            filter_fl_err = SU.weighted_biweight.biweight_scale(flux_list) / np.sqrt(len(flux_err_list))
                            avg_method = "biweight"
                        except:
                            # lets average (inverse variance)
                            variance_list = flux_err_list ** 2  # std dev
                            filter_fl = np.sum(flux_list / variance_list) / np.sum(1 / variance_list)
                            filter_fl_err = np.sqrt(
                                np.sum(flux_err_list * flux_err_list) / (len(flux_list) * len(flux_list)))
                            avg_method = "inverse variance"


                mag = -2.5*np.log10(filter_fl*1e-9/3631.)
                mag_bright = -2.5*np.log10((filter_fl+filter_fl_err)*1e-9/3631.)
                mag_faint = -2.5 * np.log10((filter_fl - filter_fl_err) * 1e-9 / 3631.)

        except:
            log.error("Exception in cat_hsc.get_filter_flux", exc_info=True)
            return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

        try:
            log.debug("HSC averaged (%s) mag = %f. Methods used = %s" %(avg_method,mag,str(methods)))
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

        s15a = False
        try:
            if G.HSC_S15A:
                s15a = True
        except:
            pass

        try:

            if s15a:
                self.dataframe_of_bid_targets = \
                    self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                        (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)].copy()
            else:
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
                except:
                    self.dataframe_of_bid_targets = []
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
        #called from elixer.py

        self.clear_pages()
        num_targets, _, _ = self.build_list_of_bid_targets(target_ra, target_dec, error)
        #could be there is no matching tile, if so, the dataframe will be none

        #if (num_targets == 0) or
        if (self.dataframe_of_bid_targets_unique is None):
            ras = []
            decs = []
        else:
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
                #here entry is a matplotlib image to be added to the PDF report
                entry = self.build_cat_summary_figure(cat_match,target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux,
                                                  detobj=detobj)
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")
            return None


        if entry is not None:
            self.add_bid_entry(entry)

            if G.SINGLE_PAGE_PER_DETECT: # and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
                #here entry is a matplotlib image to be added to the PDF report
                entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                        target_ra=target_ra, target_dec=target_dec,
                                                                    target_w=target_w, target_flux=target_flux,
                                                                    detobj=detobj)
            if entry is not None:
                self.add_bid_entry(entry)
        else:
            return None

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
                if G.HSC_S15A:
                    wcs_idx = 1
                else:
                    wcs_idx = 0
            except:
                wcs_idx = 0

            try:
                if i['image'] is None:
                    i['image'] = science_image.science_image(wcs_manual=wcs_manual,wcs_idx=wcs_idx,
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
                                          fiber_locs, target_flux,detobj)

        if not cutouts:
            return cutouts

        for c in cutouts:
            try:
                details = c['details']
            except:
                pass

            #####################################################
            # Unique to HSC: check for mask flags on the tile
            #####################################################
            if (details is not None) and (detobj is not None):
                #check for flags
                tile, tracts, positions = self.find_target_tile(ra, dec)
                #get the mask cutout
                mask_cutout = self.get_mask_cutout(tile,ra,dec,error)
                sci = science_image.science_image() #empty sci for work
                if mask_cutout is not None:
                    #iterate over the Elixer Apertures and the SEP apertures
                    if details['elixer_apertures']:
                        for a in details['elixer_apertures']:
                            # get the masks under the aperture
                            # do this for each step (increase in size)
                            pixels = sci.get_pixels_under_aperture(mask_cutout,a['ra'],a['dec'],
                                                                   a['radius'],a['radius'],
                                                                   angle= 0., north_angle=np.pi/2.)

                            mask_frac = self.update_mask_counts(pixels,None) / len(pixels)

                            #check the mask for any counts > 10% of total pixels
                            trip_mask = np.where(mask_frac > 0.10)[0]
                            # not all flags are 'bad' (i.e. 32 = detection)
                            a['image_flags'] = np.sum([2**x for x in trip_mask])

                    if details['sep_objects']:
                        for a in details['sep_objects']:
                            mask_frac = None
                            pixels = sci.get_pixels_under_aperture(mask_cutout, a['ra'], a['dec'],
                                                                   a['a'], a['b'],
                                                                   angle=a['theta'], north_angle=np.pi / 2.)

                            mask_frac = self.update_mask_counts(pixels, None)
                            # check the mask for any counts > 10% of total pixels
                            trip_mask = np.where(mask_frac > 0.10)[0]
                            #not all flags are 'bad' (i.e. 32 = detection)
                            a['image_flags'] = np.sum([2 ** x for x in trip_mask])


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

        target_count = 0
        # targets are in order of increasing distance
        for r, d in zip(bid_ras, bid_decs):
            target_count += 1
            if target_count > G.MAX_COMBINE_BID_TARGETS:
                break

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
                #add flux (cont est)
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
                        filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
                        # assumes no error in wavelength or c

                        try:

                            lineFlux_err = 0.
                            if detobj is not None:
                                try:
                                    lineFlux_err = detobj.estflux_unc
                                except:
                                    lineFlux_err = 0.

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
                                # if target_w:
                                #     ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                                #     ew_u = abs(ew * np.sqrt(
                                #         (detobj.estflux_unc / target_flux) ** 2 +
                                #         (filter_fl_err / filter_fl) ** 2))
                                # else:
                                #     ew = np.nan
                                #     ew_u = np.nan
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
                            # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                            # try:
                            #     ew_obs_err = abs(ew_obs * np.sqrt(
                            #         (lineFlux_err / target_flux) ** 2 +
                            #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                            # except:
                            #     ew_obs_err = 0.

                            ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                                           bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
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

                            try:
                                bid_target.add_filter('HSC','R',filter_fl_cgs,filter_fl_err)
                            except:
                                log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                            cat_match.add_bid_target(bid_target)
                            try:  # no downstream edits so they can both point to same bid_target
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

    # def defunct_hsc_build_cat_summary_details(self,cat_match, ra, dec, error, bid_ras, bid_decs, target_w=0,
    #                               fiber_locs=None, target_flux=None,detobj=None):
    #     """
    #     similar to build_cat_summary_figure, but rather than build up an image section to be displayed in the
    #     elixer report, this builds up a dictionary of information to be aggregated later over multiple catalogs
    #
    #     :param cat_match: a match summary object (contains info about the PDF location, etc)
    #     :param ra:  the RA of the HETDEX detection
    #     :param dec:  the Dec of the HETDEX detection
    #     :param error: radius (or half-side of a box) in which to search for matches (the cutout is 3x this on a side)
    #     :param bid_ras: RAs of potential catalog counterparts
    #     :param bid_decs: Decs of potential catalog counterparts
    #     :param target_w: observed wavelength (from HETDEX)
    #     :param fiber_locs: array (or list) of 6-tuples that describe fiber locations (which fiber, position, color, etc)
    #     :param target_flux: HETDEX integrated line flux in CGS flux units (erg/s/cm2)
    #     :param detobj: the DetObj instance
    #     :return: cutouts list of dictionaries with bid-target objects as well
    #     """
    #
    #
    #     window = error * 3
    #     cutouts = self.get_cutouts(ra,dec,window/3600.,aperture=-1,filter=None,first=False,error=error)
    #
    #     #do other stuff
    #     #1 get PLAE/POII for "best" aperture in each cutout image with details
    #     for c in cutouts:
    #         if c['details'] is None:
    #             continue
    #
    #         if (not target_flux) or (c['details']['filter_name'].lower() not in ['r','g','f606w']):
    #             continue
    #
    #         mag = c['details']['mag']
    #         filter = c['details']['filter_name'].lower()
    #         details = c['details']
    #
    #         #set the continuum estimate from the broadband filter
    #         #if no aperture magnitude was calculated, set it to the mag-limit
    #         if not (c['details']['mag'] < 99): #no mag could be calculated
    #             non_detect = min(self.get_mag_limit(),33.0) #33 is just there for some not 99 limit
    #             cont_est = self.obs_mag_to_cgs_flux(non_detect,SU.filter_iso(filter,target_w))
    #         else:
    #             cont_est = self.obs_mag_to_cgs_flux(mag,SU.filter_iso(filter,target_w))
    #
    #         bid_target = match_summary.BidTarget()
    #         bid_target.catalog_name = self.Name
    #         bid_target.bid_ra = 666  # nonsense RA
    #         bid_target.bid_dec = 666  # nonsense Dec
    #         bid_target.distance = 0.0
    #         bid_target.bid_filter = filter
    #         bid_target.bid_mag = mag
    #         bid_target.bid_mag_err_bright = 0.0 #todo: right now don't have error on aperture mag
    #         bid_target.bid_mag_err_faint = 0.0
    #         bid_target.bid_flux_est_cgs_unc = 0.0
    #
    #         bid_target.bid_flux_est_cgs = cont_est
    #         try:
    #             flux_faint = None
    #             flux_bright = None
    #
    #             if details['mag_faint'] < 99:
    #                 flux_faint = self.obs_mag_to_cgs_flux(details['mag_faint'], SU.filter_iso(filter,target_w))
    #
    #             if details['mag_bright'] < 99:
    #                 flux_bright = self.obs_mag_to_cgs_flux(details['mag_bright'], SU.filter_iso(filter,target_w))
    #
    #             if flux_bright and flux_faint:
    #                 bid_target.bid_flux_est_cgs_unc = max((bid_target.bid_flux_est_cgs - flux_faint),
    #                                                       (flux_bright -bid_target.bid_flux_est_cgs))
    #             elif flux_bright:
    #                 bid_target.bid_flux_est_cgs_unc = flux_bright - bid_target.bid_flux_est_cgs
    #             else: #neither faint nor bright, so at the limit
    #                 pass #anything we can reasonably do here? fully below flux limit?
    #
    #         except:
    #             pass
    #
    #         try:
    #             bid_target.bid_mag_err_bright = mag - details['mag_bright']
    #             bid_target.bid_mag_err_faint =  details['mag_faint'] - mag
    #         except:
    #             pass
    #
    #         bid_target.add_filter(c['instrument'], filter, bid_target.bid_flux_est_cgs, -1)
    #
    #         addl_waves = None
    #         addl_flux = None
    #         addl_ferr = None
    #         try:
    #             addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
    #             addl_flux = cat_match.detobj.spec_obj.addl_fluxes
    #             addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
    #         except:
    #             pass
    #
    #         lineFlux_err = 0.
    #         if detobj is not None:
    #             try:
    #                 lineFlux_err = detobj.estflux_unc
    #             except:
    #                 lineFlux_err = 0.
    #
    #         # build EW error from lineFlux_err and aperture estimate error
    #         ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
    #         try:
    #             ew_obs_err = abs(ew_obs * np.sqrt(
    #                 (lineFlux_err / target_flux) ** 2 +
    #                 (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
    #         except:
    #             ew_obs_err = 0.
    #
    #         bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
    #             line_prob.mc_prob_LAE(
    #                 wl_obs=target_w,
    #                 lineFlux=target_flux,
    #                 lineFlux_err=lineFlux_err,
    #                 continuum=bid_target.bid_flux_est_cgs,
    #                 continuum_err=bid_target.bid_flux_est_cgs_unc,
    #                 c_obs=None, which_color=None,
    #                 addl_wavelengths=addl_waves,
    #                 addl_fluxes=addl_flux,
    #                 addl_errors=addl_ferr,
    #                 sky_area=None,
    #                 cosmo=None, lae_priors=None,
    #                 ew_case=None, W_0=None,
    #                 z_OII=None, sigma=None)
    #
    #         try:
    #             if plae_errors:
    #                 bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
    #                 bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
    #                 c['aperture_plae_min'] = plae_errors['ratio'][1] #new key
    #                 c['aperture_plae_max'] = plae_errors['ratio'][2] #new key
    #         except:
    #             pass
    #
    #         c['aperture_plae'] = bid_target.p_lae_oii_ratio
    #         c['aperture_eqw_rest_lya'] = ew_obs / (1. + target_w / G.LyA_rest)
    #         c['aperture_eqw_rest_lya_err'] = ew_obs_err / (1. + target_w / G.LyA_rest)
    #
    #         #also goes into the details
    #         if c['details']:
    #             c['details']['aperture_plae'] = bid_target.p_lae_oii_ratio
    #             c['details']['aperture_eqw_rest_lya'] = ew_obs / (1. + target_w / G.LyA_rest)
    #             c['details']['aperture_eqw_rest_lya_err'] = ew_obs_err / (1. + target_w / G.LyA_rest)
    #             try:
    #                 if plae_errors:
    #                     c['details']['aperture_plae_min'] = plae_errors['ratio'][1] #new key
    #                     c['details']['aperture_plae_max'] = plae_errors['ratio'][2] #new key
    #             except:
    #                 pass
    #
    #         cat_match.add_bid_target(bid_target)
    #         try:  # no downstream edits so they can both point to same bid_target
    #             if detobj is not None:
    #                 detobj.bid_target_list.append(bid_target)
    #         except:
    #             log.warning("Unable to append bid_target to detobj.", exc_info=True)
    #
    #
    #
    #         #####################################################
    #         # Unique to HSC: check for mask flags on the tile
    #         #####################################################
    #         if (details is not None) and (detobj is not None):
    #             #check for flags
    #             tile, tracts, positions = self.find_target_tile(ra, dec)
    #             #get the mask cutout
    #             mask_cutout = self.get_mask_cutout(tile,ra,dec,error)
    #             sci = science_image.science_image() #empty sci for work
    #             if mask_cutout is not None:
    #                 #iterate over the Elixer Apertures and the SEP apertures
    #                 if details['elixer_apertures']:
    #                     for a in details['elixer_apertures']:
    #                         # get the masks under the aperture
    #                         # do this for each step (increase in size)
    #                         pixels = sci.get_pixels_under_aperture(mask_cutout,a['ra'],a['dec'],
    #                                                                a['radius'],a['radius'],
    #                                                                angle= 0., north_angle=np.pi/2.)
    #
    #                         mask_frac = self.update_mask_counts(pixels,None) / len(pixels)
    #
    #                         #check the mask for any counts > 10% of total pixels
    #                         trip_mask = np.where(mask_frac > 0.10)[0]
    #                         # not all flags are 'bad' (i.e. 32 = detection)
    #                         a['image_flags'] = np.sum([2**x for x in trip_mask])
    #
    #                 if details['sep_objects']:
    #                     for a in details['sep_objects']:
    #                         mask_frac = None
    #                         pixels = sci.get_pixels_under_aperture(mask_cutout, a['ra'], a['dec'],
    #                                                                a['a'], a['b'],
    #                                                                angle=a['theta'], north_angle=np.pi / 2.)
    #
    #                         mask_frac = self.update_mask_counts(pixels, None)
    #                         # check the mask for any counts > 10% of total pixels
    #                         trip_mask = np.where(mask_frac > 0.10)[0]
    #                         #not all flags are 'bad' (i.e. 32 = detection)
    #                         a['image_flags'] = np.sum([2 ** x for x in trip_mask])
    #
    #
    #         #####################################################
    #         #Common to all catalogs; add the aperture details
    #         #####################################################
    #
    #         if (details is not None) and (detobj is not None):
    #             detobj.aperture_details_list.append(details)
    #
    #     #2. catalog entries as a new key under cutouts (like 'details') ... 'counterparts'
    #     #    this should be similar to the build_multiple_bid_target_figures_one_line()
    #
    #     if len(bid_ras) > 0:
    #         #if there are no cutouts (but we do have a catalog), create a cutouts list of dictionries to hold the
    #         #counterparts
    #         if not cutouts or len(cutouts) == 0:
    #             cutouts = [{}]
    #
    #         cutouts[0]['counterparts'] = []
    #         #create an empty list of counterparts under the 1st cutout
    #         #counterparts are not filter specific, so we will just keep one list under the 1st cutout
    #
    #     target_count = 0
    #     # targets are in order of increasing distance
    #     for r, d in zip(bid_ras, bid_decs):
    #         target_count += 1
    #         if target_count > G.MAX_COMBINE_BID_TARGETS:
    #             break
    #
    #         try: #DO NOT WANT _unique as that has wiped out the filters
    #             df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
    #                                                    (self.dataframe_of_bid_targets['DEC'] == d[0]) &
    #                                                    (self.dataframe_of_bid_targets['FILTER'] == 'r')]
    #             if (df is None) or (len(df) == 0):
    #                 df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
    #                                                        (self.dataframe_of_bid_targets['DEC'] == d[0]) &
    #                                                        (self.dataframe_of_bid_targets['FILTER'] == 'g')]
    #             if (df is None) or (len(df) == 0):
    #                 df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
    #                                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]
    #
    #         except:
    #             log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
    #             continue  # this must be here, so skip to next ra,dec
    #
    #         if df is not None:
    #             #add flux (cont est)
    #             try:
    #                 filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = self.get_filter_flux(df)
    #             except:
    #                 filter_fl = 0.0
    #                 filter_fl_err = 0.0
    #                 filter_mag = 0.0
    #                 filter_mag_bright = 0.0
    #                 filter_mag_faint = 0.0
    #                 filter_str = "NA"
    #
    #             bid_target = None
    #
    #             if (target_flux is not None) and (filter_fl != 0.0):
    #                 if (filter_fl is not None):# and (filter_fl > 0):
    #                     filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
    #                     filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
    #                     # assumes no error in wavelength or c
    #
    #                     try:
    #                         bid_target = match_summary.BidTarget()
    #                         bid_target.catalog_name = self.Name
    #                         bid_target.bid_ra = df['RA'].values[0]
    #                         bid_target.bid_dec = df['DEC'].values[0]
    #                         bid_target.distance = df['distance'].values[0] * 3600
    #                         bid_target.prob_match = df['dist_prior'].values[0]
    #                         bid_target.bid_flux_est_cgs = filter_fl_cgs
    #                         bid_target.bid_filter = filter_str
    #                         bid_target.bid_mag = filter_mag
    #                         bid_target.bid_mag_err_bright = filter_mag_bright
    #                         bid_target.bid_mag_err_faint = filter_mag_faint
    #                         bid_target.bid_flux_est_cgs_unc = filter_fl_cgs_unc
    #
    #                         try:
    #                             ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
    #                             ew_u = abs(ew * np.sqrt(
    #                                 (detobj.estflux_unc / target_flux) ** 2 +
    #                                 (filter_fl_err / filter_fl) ** 2))
    #
    #                             bid_target.bid_ew_lya_rest = ew
    #                             bid_target.bid_ew_lya_rest_err = ew_u
    #
    #                         except:
    #                             log.debug("Exception computing catalog EW: ", exc_info=True)
    #
    #                         addl_waves = None
    #                         addl_flux = None
    #                         addl_ferr = None
    #                         try:
    #                             addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
    #                             addl_flux = cat_match.detobj.spec_obj.addl_fluxes
    #                             addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
    #                         except:
    #                             pass
    #
    #                         lineFlux_err = 0.
    #                         if detobj is not None:
    #                             try:
    #                                 lineFlux_err = detobj.estflux_unc
    #                             except:
    #                                 lineFlux_err = 0.
    #
    #                         # build EW error from lineFlux_err and aperture estimate error
    #                         ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
    #                         try:
    #                             ew_obs_err = abs(ew_obs * np.sqrt(
    #                                 (lineFlux_err / target_flux) ** 2 +
    #                                 (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
    #                         except:
    #                             ew_obs_err = 0.
    #
    #                         bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
    #                             line_prob.mc_prob_LAE(
    #                                 wl_obs=target_w,
    #                                 lineFlux=target_flux,
    #                                 lineFlux_err=lineFlux_err,
    #                                 continuum=bid_target.bid_flux_est_cgs,
    #                                 continuum_err=bid_target.bid_flux_est_cgs_unc,
    #                                 c_obs=None, which_color=None,
    #                                 addl_wavelengths=addl_waves,
    #                                 addl_fluxes=addl_flux,
    #                                 addl_errors=addl_ferr,
    #                                 sky_area=None,
    #                                 cosmo=None, lae_priors=None,
    #                                 ew_case=None, W_0=None,
    #                                 z_OII=None, sigma=None)
    #
    #                         try:
    #                             if plae_errors:
    #                                 bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
    #                                 bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
    #                         except:
    #                             pass
    #
    #                         try:
    #                             bid_target.add_filter('HSC','R',filter_fl_cgs,filter_fl_err)
    #                         except:
    #                             log.debug('Unable to build filter entry for bid_target.',exc_info=True)
    #
    #                         cat_match.add_bid_target(bid_target)
    #                         try:  # no downstream edits so they can both point to same bid_target
    #                             detobj.bid_target_list.append(bid_target)
    #                         except:
    #                             log.warning("Unable to append bid_target to detobj.", exc_info=True)
    #
    #                         try:
    #                             cutouts[0]['counterparts'].append(bid_target)
    #                         except:
    #                             log.warning("Unable to append bid_target to cutouts.", exc_info=True)
    #                     except:
    #                         log.debug('Unable to build bid_target.',exc_info=True)
    #
    #
    #     #once all done, pass cutouts to a new function to create the summary figure ... SHOULD be only in the base class
    #     #as all should be identical and based only on the cutouts object data
    #     # this include
    #     #1. stack cutouts to make a master cutout ...
    #     #2. add fibers, North box to master cutout ...
    #
    #     return cutouts


    def build_cat_summary_figure (self, cat_match, ra, dec, error, bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        """
            Builds the figure (page) the exact target location. Contains just the filter images ...
            Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page

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
        :return:
        """

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

        if self.INCLUDE_KPNO_G:
            try:

                #IF KPNO g-band is available,
                #   advance the index by 1 and insert the KPNO g-band image in 1st position AFTER the "master" cutout (made
                #   from only HSC)

                kpno = cat_kpno.KPNO()
                kpno_cuts = kpno.get_cutouts(ra,dec,window/3600.0,aperture=kpno.mean_FWHM * 0.5 + 0.5,filter='g',first=True)

                if (kpno_cuts is not None) and (len(kpno_cuts) == 1):
                    index += 1
                    _ = plt.subplot(gs[1:, index])

                    sci = science_image.science_image()

                    vmin, vmax = sci.get_vrange(kpno_cuts[0]['cutout'].data)
                    pix_size = sci.calc_pixel_size(kpno_cuts[0]['cutout'].wcs)
                    ext = kpno_cuts[0]['cutout'].shape[0] * pix_size / 2.

                    plt.imshow(kpno_cuts[0]['cutout'].data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                               vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

                    #plt.title(i['instrument'] + " " + i['filter'])
                    plt.title("(" + kpno.name + " g)")
                    plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    # plt.plot(0, 0, "r+")
                    self.add_zero_position(plt)

                    if kpno_cuts[0]['details'] is not None:
                        try:
                            if (kpno.MAG_LIMIT < kpno_cuts[0]['details']['mag'] < 100) and (kpno_cuts[0]['details']['radius'] > 0):
                                kpno_cuts[0]['details']['fail_mag_limit'] = True
                                kpno_cuts[0]['details']['raw_mag'] = kpno_cuts[0]['details']['mag']
                                kpno_cuts[0]['details']['raw_mag_bright'] = kpno_cuts[0]['details']['mag_bright']
                                kpno_cuts[0]['details']['raw_mag_faint'] = kpno_cuts[0]['details']['mag_faint']
                                kpno_cuts[0]['details']['raw_mag_err'] = kpno_cuts[0]['details']['mag_err']
                                log.warning(f"Cutout mag {kpno_cuts[0]['details']['mag']} greater than limit {kpno.MAG_LIMIT}. Setting to limit.")

                                kpno_cuts[0]['details']['mag'] = kpno.MAG_LIMIT
                                try:
                                    kpno_cuts[0]['details']['mag_bright'] = min(kpno.MAG_LIMIT, kpno_cuts[0]['details']['mag_bright'])
                                except:
                                    kpno_cuts[0]['details']['mag_bright'] = kpno.MAG_LIMIT
                                try:
                                    kpno_cuts[0]['details']['mag_faint'] = max(kpno.MAG_LIMIT, G.MAX_MAG_FAINT)
                                except:
                                    kpno_cuts[0]['details']['mag_faint'] = G.MAX_MAG_FAINT
                        except:
                            pass
                        #this will happen anyway under KPNO itself
                        # (note: the "details" are actually populated in the independent KPNO catalog calls)

                        #build up the needed parameters from the kpno_cuts
                        cx = kpno_cuts[0]['ap_center'][0]
                        cy = kpno_cuts[0]['ap_center'][1]

                        #and need to get an EW and PLAE/POII

                        if detobj is not None:
                            try:
                                lineFlux_err = detobj.estflux_unc
                            except:
                                lineFlux_err = 0.

                        try:
                            flux_faint = None
                            flux_bright = None
                            bid_flux_est_cgs_unc = None
                            bid_flux_est_cgs = None

                            if kpno_cuts[0]['details']['mag'] < 99:
                                bid_flux_est_cgs = self.obs_mag_to_cgs_flux(kpno_cuts[0]['details']['mag'],
                                                                            SU.filter_iso('g', target_w))

                                if kpno_cuts[0]['details']['mag_faint'] < 99:
                                    flux_faint = self.obs_mag_to_cgs_flux(kpno_cuts[0]['details']['mag_faint'],
                                                                          SU.filter_iso('g', target_w))

                                if kpno_cuts[0]['details']['mag_bright'] < 99:
                                    flux_bright = self.obs_mag_to_cgs_flux(kpno_cuts[0]['details']['mag_bright'],
                                                                           SU.filter_iso('g', target_w))
                                if flux_bright and flux_faint:
                                    bid_flux_est_cgs_unc = max((bid_flux_est_cgs - flux_faint),
                                                                          (flux_bright - bid_flux_est_cgs))
                                elif flux_bright:
                                    bid_flux_est_cgs_unc = flux_bright - bid_flux_est_cgs

                        except:
                            pass

                        addl_waves = None
                        addl_flux = None
                        addl_ferr = None
                        try:
                            addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                            addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                            addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                        except:
                            pass

                        p_lae_oii_ratio, p_lae, p_oii, plae_errors = \
                            line_prob.mc_prob_LAE(
                                wl_obs=target_w,
                                lineFlux=target_flux,
                                lineFlux_err=lineFlux_err,
                                continuum=bid_flux_est_cgs,
                                continuum_err=bid_flux_est_cgs_unc,
                                c_obs=None, which_color=None,
                                addl_wavelengths=addl_waves,
                                addl_fluxes=addl_flux,
                                addl_errors=addl_ferr,
                                sky_area=None,
                                cosmo=None, lae_priors=None,
                                ew_case=None, W_0=None,
                                z_OII=None, sigma=None)

                        ew_obs = (target_flux / bid_flux_est_cgs)
                        cutout_ewr = ew_obs / (1. + target_w / G.LyA_rest)
                        cutout_plae = p_lae_oii_ratio

                        if (kpno_cuts[0]['details']['sep_objects'] is not None):  # and (details['sep_obj_idx'] is not None):
                            self.add_elliptical_aperture_positions(plt, kpno_cuts[0]['details']['sep_objects'],
                                                                   kpno_cuts[0]['details']['sep_obj_idx'],
                                                                   kpno_cuts[0]['details']['radius'],
                                                                   kpno_cuts[0]['details']['mag'],
                                                                   cx, cy, cutout_ewr, cutout_plae)
                        else:
                            self.add_aperture_position(plt, kpno_cuts[0]['details']['radius'],
                                                       kpno_cuts[0]['details']['mag'],
                                                       cx, cy, cutout_ewr, cutout_plae)

                    self.add_north_box(plt, sci, kpno_cuts[0]['cutout'], error, 0, 0, theta=None)
                    #don't want KPNO catalog objects, just the HSC ones

            except:
                log.warning("Exception adding KPNO to HSC report",exc_info=True)

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

            try:
                if G.HSC_S15A:
                    wcs_idx = 1
                else:
                    wcs_idx = 0
            except:
                wcs_idx = 0

            if i['image'] is None:
                i['image'] = science_image.science_image(wcs_manual=wcs_manual,wcs_idx=wcs_idx,
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

            if (self.MAG_LIMIT < mag < 100) and (mag_radius > 0):
                log.warning(f"Cutout mag {mag} greater than limit {self.MAG_LIMIT}. Setting to limit.")
                details['fail_mag_limit'] = True
                details['raw_mag'] = mag
                details['raw_mag_bright'] = details['mag_bright']
                details['raw_mag_faint'] = details['mag_faint']
                details['raw_mag_err'] = details['mag_err']
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
                        #bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag, target_w)
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag,SU.filter_iso(i['filter'],target_w))
                        try:
                            flux_faint = None
                            flux_bright = None

                            if details['mag_faint'] < 99:
                                flux_faint = self.obs_mag_to_cgs_flux(details['mag_faint'], SU.filter_iso(i['filter'],target_w))

                            if details['mag_bright'] < 99:
                                flux_bright = self.obs_mag_to_cgs_flux(details['mag_bright'], SU.filter_iso(i['filter'],target_w))

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
                    # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                    # try:
                    #     ew_obs_err = abs(ew_obs * np.sqrt(
                    #         (lineFlux_err / target_flux) ** 2 +
                    #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                    # except:
                    #     ew_obs_err = 0.

                    ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                                   bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

                    # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                    #     line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                    #                        ew_obs=ew_obs,
                    #                        lineFlux_err=lineFlux_err,
                    #                        ew_obs_err=ew_obs_err,
                    #                        c_obs=None, which_color=None, addl_fluxes=addl_flux,
                    #                        addl_wavelengths=addl_waves, addl_errors=addl_ferr, sky_area=None,
                    #                        cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                    #                        sigma=None,estimate_error=True)

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

                    cutout_plae = bid_target.p_lae_oii_ratio
                    cutout_ewr = ew_obs / (1. + target_w / G.LyA_rest)
                    cutout_ewr_err = ew_obs_err / (1. + target_w / G.LyA_rest)
                    # if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)" % (bid_target.p_lae_oii_ratio,i['filter']))

                    if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                        try:
                            text.set_text(
                                text.get_text() + "  P(LAE)/P(OII): $%.4g\ ^{%.4g}_{%.4g}$ (%s)" %
                                (round(bid_target.p_lae_oii_ratio, 3),
                                 round(bid_target.p_lae_oii_ratio_max, 3),
                                 round(bid_target.p_lae_oii_ratio_min, 3),
                                 f))
                        except:
                            log.debug("Exception adding PLAE with range", exc_info=True)
                            try:
                                text.set_text(
                                    text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)" % (bid_target.p_lae_oii_ratio, f))
                            except:
                                text.set_text(
                                    text.get_text() + "  P(LAE)/P(OII): (%s) (%s)" % ("---", f))
                        # text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g [%0.4g:%0.4g] (%s)"
                        #               %(utilities.saferound(bid_target.p_lae_oii_ratio,3),
                        #                 utilities.saferound(bid_target.p_lae_oii_ratio_min,3),
                        #                 utilities.saferound(bid_target.p_lae_oii_ratio_max,3),i['filter']))

                    cat_match.add_bid_target(bid_target)
                    try:  # no downstream edits so they can both point to same bid_target
                        if detobj is not None:
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
                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True,reset_center=False)
                    #self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True)
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
                    if (details['sep_objects'] is not None): # and (details['sep_obj_idx'] is not None):
                        self.add_elliptical_aperture_positions(plt,details['sep_objects'],details['sep_obj_idx'],
                                                               mag_radius,mag,cx,cy,cutout_ewr,cutout_plae)
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
                #check for flags
                #get the mask cutout
                mask_cutout = self.get_mask_cutout(tile,ra,dec,error)
                if mask_cutout is not None:
                    #iterate over the Elixer Apertures and the SEP apertures
                    if details['elixer_apertures']:
                        for a in details['elixer_apertures']:
                            # get the masks under the aperture
                            # do this for each step (increase in size)
                            pixels = sci.get_pixels_under_aperture(mask_cutout,a['ra'],a['dec'],
                                                               a['radius'],a['radius'],
                                                               angle= 0., north_angle=np.pi/2.)

                            mask_frac = self.update_mask_counts(pixels,None) / len(pixels)

                            #check the mask for any counts > 10% of total pixels
                            trip_mask = np.where(mask_frac > 0.10)[0]
                            # not all flags are 'bad' (i.e. 32 = detection)
                            a['image_flags'] = np.sum([2**x for x in trip_mask])

                    if details['sep_objects']:
                        for a in details['sep_objects']:
                            mask_frac = None
                            pixels = sci.get_pixels_under_aperture(mask_cutout, a['ra'], a['dec'],
                                                                   a['a'], a['b'],
                                                                   angle=a['theta'], north_angle=np.pi / 2.)

                            mask_frac = self.update_mask_counts(pixels, None)
                            # check the mask for any counts > 10% of total pixels
                            trip_mask = np.where(mask_frac > 0.10)[0]
                            #not all flags are 'bad' (i.e. 32 = detection)
                            a['image_flags'] = np.sum([2 ** x for x in trip_mask])

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

            plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2 , height=ry * 2,
                                              angle=0, color='red', fill=False,linewidth=8))

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300,transparent=True)
            detobj.image_cutout_fiber_pos = buf
            plt.close()

        return fig


    def update_mask_counts(self,pixels=None,mask_counts=None):
        """

        :param pixels:
        :param mask:
        :return:
        """

        try:
            if mask_counts is None:
                mask_counts = np.zeros(self.MASK_LENGTH,dtype=int)

            if pixels is None:
                return mask_counts

            if type(pixels) is list:
                pixels = np.array(pixels)

            for i in range(len(mask_counts)):
                try:
                    num_pix = len(np.where(pixels & 2**i)[0])
                    mask_counts[i] += num_pix
                except:
                    pass
        except:
            pass

        return mask_counts

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
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
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

                            lineFlux_err = 0.
                            if detobj is not None:
                                try:
                                    lineFlux_err = detobj.estflux_unc
                                except:
                                    lineFlux_err = 0.

                            # build EW error from lineFlux_err and aperture estimate error
                            # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                            # try:
                            #     ew_obs_err = abs(ew_obs * np.sqrt(
                            #         (lineFlux_err / target_flux) ** 2 +
                            #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                            # except:
                            #     ew_obs_err = 0.

                            ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w, bid_target.bid_filter,
                                                           bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

                            # bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                            #     line_prob.prob_LAE(wl_obs=target_w,
                            #                        lineFlux=target_flux,
                            #                        ew_obs=ew_obs,
                            #                        lineFlux_err=lineFlux_err,
                            #                        ew_obs_err=ew_obs_err,
                            #                        c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                            #                        addl_fluxes=addl_flux, addl_errors=addl_ferr, sky_area=None,
                            #                        cosmo=None, lae_priors=None,
                            #                        ew_case=None, W_0=None,
                            #                        z_OII=None, sigma=None,estimate_error=True)

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
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
                    try:
                        text += r"$%0.4g\ ^{%.4g}_{%.4g}$" % (utilities.saferound(bid_target.p_lae_oii_ratio, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_max, 3),
                                                              utilities.saferound(bid_target.p_lae_oii_ratio_min, 3))
                        text += "\n"
                    except:
                        text += "%0.4g\n" % ( utilities.saferound(bid_target.p_lae_oii_ratio,3))

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

        try:
            if G.HSC_S15A:
                wcs_idx = 1
            else:
                wcs_idx = 0
        except:
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
                try:
                    if d['mag_limit']:
                        details['mag_limit']=d['mag_limit']
                    else:
                        details['mag_limit'] = None
                except:
                    details['mag_limit'] = None

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

                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == tile)))]
                    if i is not None:
                        cutout = self.get_single_cutout(ra, dec, window, i, aperture,error)

                        if first:
                            if cutout['cutout'] is not None:
                                l.append(cutout)
                                break
                        else:
                            # if we are not escaping on the first hit, append ALL cutouts (even if no image was collected)
                            l.append(cutout)

                except Exception as e:
                    if type(e) is StopIteration:
                        #just did not find any more catalog images to use that match the criteria
                        pass
                    else:
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
