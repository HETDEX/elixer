from __future__ import print_function

import scipy.interpolate

"""
This is for the HSC SSP data. Cloned originally from hsc_nep.py code, but modified extensively
"""

try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_base
    from elixer import match_summary
    from elixer import line_prob
    from elixer import utilities
    from elixer import spectrum_utilities as SU
    from elixer import hsc_ssp_meta
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import line_prob
    import utilities
    import spectrum_utilities as SU
    import hsc_ssp_meta

import os.path as op
import glob
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
import astropy.io.fits as astropyFITS

#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

def hsc_count_to_mag(count,cutout=None,headers=None):
    """
    SSP is using the same zero point as the HETDEX HSC, so this is still valid

    :param count:
    :param cutout:
    :param headers:
    :return:
    """


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
        # if 'FLUXMAG0' in headers[0]:
        #     fluxmag0 = float(headers[0]['FLUXMAG0'])
        #this can come from a compressed .fz fits which may have added a new header at [0]
        for h in headers:
            if 'FLUXMAG0' in h:
                fluxmag0 = float(h['FLUXMAG0'])
                break
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

class HSC_SSP(cat_base.Catalog):#Hyper Suprime Cam, North Ecliptic Pole
    # class variables

    HSC_BASE_PATH = G.HSC_SSP_BASE_PATH
    HSC_CAT_PATH = G.HSC_SSP_CAT_PATH
    HSC_IMAGE_PATH = G.HSC_SSP_IMAGE_PATH
    HSC_PHOTZ_PATH = G.HSC_SSP_PHOTO_Z_PATH#"/scratch/03261/polonius/hsc_photz"

    INCLUDE_KPNO_G = False


    MAG_LIMIT = 26.2 #mostly care about r (this give a little slop for error and for smaller aperture before the limit kicks in)

    mean_FWHM = 0.75 #average: g=0.77, r=0.76, i = 0.58, z = 0.68, y = 0.68

    CONT_EST_BASE = None

    df = None
    loaded_tracts = []

    MainCatalog = None #there is no Main Catalog ... must load individual catalog tracts
    Name = "HSC-SSP" #"HyperSuprimeCam_SSP"

    #todo: HERE ... just define the coordrange, etc rather than load a meta file
    #using the HETDEX HSC dictionary format, but filter, tract, and pos are not relevant
    #the path is updated just below
    #Image_Coord_Range = {'RA_min':270.3579555, 'RA_max':270.84872930, 'Dec_min':67.5036877, 'Dec_max':67.8488075}

    Image_Coord_Range = hsc_ssp_meta.Image_Coord_Range
    Tile_Dict = hsc_ssp_meta.HSC_META_DICT

    #correct the basepaths
    for k in Tile_Dict.keys():
        Tile_Dict[k]['path'] = op.join(HSC_IMAGE_PATH,Tile_Dict[k]['path'].split("hsc_ssp/")[1])


    Filters = ['g','r','i','z','y'] #case is important ... needs to be lowercase
    Filter_HDU_Image_Idx = 1 #idx 0 is a generic header

    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    WCS_Manual = False

    AstroTable = None

    #HETDEX HSC values
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

    BidCols = []


    CatalogImages = [] #built in constructor (like the Tile_Dict, but is populated with actual cutouts as they are made

    def __init__(self):
        super(HSC_SSP, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_unique = None
        self.dataframe_of_bid_targets_photoz = None
        self.num_targets = 0
        self.master_cutout = None
        self.build_catalog_of_images()

    @classmethod
    def read_catalog(cls, catalog_loc=None, name=None,tract=None,position=None, tile=None):
        """

        There is not a singluar catalog for HSC-SSP ... each tile has its own. Mostly (maybe all?) only have entries
        where there is i-band coverage. Photometry is definitely limited to i-band coverage targets.

        :param catalog_loc:
        :param name:
        :param tract: list of string string as the HSC track id, ie. ['16814']
        :param position: a tuple or array with exactly 2 elements as integers 0-9, i.e. (0,1) or (2,8), etc
        :return:
        """

        if name is None:
            name = cls.Name


        if G.BANDPASS_PREFER_G:
            first = 'G'
            second = 'R'
        else:
            first = 'R'
            second = 'G'

        try:
            filter_str = 'x'
            if tile is not None and '?' in tile:
                tileX = tile.replace("-?-",f"-{first}-") #get the 'R' filter catalog
                if tileX not in cls.Tile_Dict.keys():
                    tileX = tile.replace("-?-",f"-{second}-")
                    if tileX not in cls.Tile_Dict.keys():
                        log.info("Unable to locate suitable photometric counterpart catalog.")
                        return None
                    else:
                        filter_str = second.lower()
                else:
                    filter_str = first.lower()
            else:
                return None


            #turn tileX into the catalog name
            #ie.g srcMatchFull-HSC-G-10056-0,0.fits
            cat_name = tileX.replace("calexp","srcMatchFull")

            #get the path
            fqtract = [op.join(op.dirname(cls.Tile_Dict[tileX]['path']),cat_name),]

            if set(fqtract).issubset(cls.loaded_tracts):
                log.info("Catalog tract (%s) already loaded." %fqtract)
                return cls.df

            #todo: future more than just the R filter if any are ever added
            for t in fqtract:
                if t in cls.loaded_tracts: #skip if already loaded
                    continue

                cat_name = op.basename(t)
                cat_loc = t
                #header = cls.BidCols

                if not op.exists(cat_loc):
                    log.error("Cannot load catalog tract for HSC. File does not exist: %s" %cat_loc)

                log.debug("Building " + cls.Name + " " + cat_name + " dataframe...")

                try:
                    table = astropy.table.Table.read(cat_loc,hdu=1)#,format='fits'# )

                    cls.flags_table = astropy.table.Table([table['src_id'],table['flags']]) #of limited use, need to grow when adding more tiles

                    #warning! don't keep the HSC "distance" ... we overwrite later with our own value and theirs has
                    #a different meaning
                    table.keep_columns(['src_id','src_coord_ra','src_coord_dec',   #'distance',
                                        'src_base_CircularApertureFlux_3_0_instFlux','src_base_CircularApertureFlux_3_0_instFluxErr',
                                        'src_ext_photometryKron_KronFlux_instFlux','src_ext_photometryKron_KronFlux_instFluxErr',
                                        'src_ext_photometryKron_KronFlux_radius','src_ext_photometryKron_KronFlux_psf_radius'])

                    #flags is an array of 246 booleans, numbering starts with 1
                    #as is we cannot keep the flags and translate to a pandas dataframe as it does not support multi-dimensional columns
                    #want to check 'src_detect_isPrimary' (216 or index 215),'src_base_PixellFags_flag' ()

                    #from HSC documentatin
                    #You should first select primary objects (i.e., objects in the inner patch and inner tract with no
                    # children) by applying isprimary=True.   It will be a good practice to then apply pixel flags to
                    # make sure that objects do not suffer from problematic pixels; {filter}_pixelflags_saturated_{any, center},
                    # {filter}_pixelflags_interpolated_{any,center}, etc, except for the UD region (see the Known Problems
                    # page for details).  There are two separate pixel flags depending on which part of objects is concerned.
                    # Those with "any" are set True when any pixel of the object's footprint are affected.  Those with
                    # "center" are  set True when any of the central 3×3 pixels are affected.  For most cases, the latter
                    # should be fine as problematic pixels are interpolated reasonably well in the outer part.  Finally,
                    # one may want to make sure that the object centroiding is OK with {filter}_sdsscentroid_flags
                    # (if the centroiding is bad, photometry is likely bad), but faint objects tend to have these flags on.
                    # One should check if scientific results are sensitive to this flag.  It is also important to make
                    # sure that the photometry is OK using the flags associated with the measurement such as {filter}_psfflux_flags.

                    #since the faint objects are of most concern to us and the documentation warns that they tend to have
                    # the centroiding flag set, we will have to ignore that flag. But we may want to pay attention to "center"

                    #the RA and Dec are in RADIANS
                    table['src_coord_ra'] *= 180.0/np.pi
                    table['src_coord_dec'] *= 180.0/np.pi

                    #add the filter so we can identify it later
                    table['filter_str'] = filter_str

                    #table['tract'] = np.tile(np.array(tract),(len(table),1))  #need this later to get the zPDF
                    #pandas is a problem and won't let me use this as a column of arrays, so will make into a comma
                    #separate string
                    table['tract'] = ';'.join(tract)  # need this later to get the zPDF (use ; as some tracts could have a comma)

                except KeyError:
                    log.error(name + " Exception attempting to open catalog file. KeyError." + cat_loc)
                    continue
                except Exception as e:
                    if type(e) is astropy.io.registry.IORegistryError:
                        log.error(name + " Exception attempting to open catalog file: (IORegistryError, bad format)" + cat_loc, exc_info=False)
                    else:
                        log.error(name + " Exception attempting to open catalog file: " + cat_loc, exc_info=True)
                    continue #try the next one  #exc_info = sys.exc_info()


                try:
                    df = table.to_pandas()
                    #df = pd.read_csv(cat_loc, names=header,
                    #                 delim_whitespace=True, header=None, index_col=None, skiprows=0)

                    old_names = ['src_coord_ra','src_coord_dec']
                    new_names = ['RA','DEC']
                    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

                   # df['FILTER'] = 'r' #add the FILTER to the dataframe !!! case is important. must be lowercase

                    if cls.df is not None:
                        cls.df = pd.concat([cls.df, df])
                    else:
                        cls.df = df
                    cls.loaded_tracts.append(t)

                except:
                    log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
                    continue

            return cls.df
        except:
            log.error("Exception!",exc_info=True)
            return None


    def get_mag_limit(self,image_identification=None,aperture_diameter=None):
        """
            to be overwritten by subclasses to return their particular format of maglimit

            :param image_identification: some way (sub-class specific) to identify which image
                    HERE we want a tuple ... [0] = tile name and [1] = filter name
            :param aperture_diameter: in arcsec
            :return:
        """

        try:
            #0.2 ~= 2.5 * log(1.2) ... or a 20% error
            return self.Tile_Dict[image_identification[0]]['depth'] + 0.2

        except:
            log.warning("cat_hsc_nep.py get_mag_limit fail.",exc_info=True)
            try:
                return self.MAG_LIMIT
            except:
                return 99.9

    def build_catalog_of_images(self):

        for t in self.Tile_Dict.keys(): #tile is the key (the filename)
            #for f in self.Filters:
            f = self.Tile_Dict[t]['filter'].lower()
            #path = self.HSC_IMAGE_PATH #op.join(self.HSC_IMAGE_PATH,self.Tile_Dict[t]['tract'])
            #path is already corrected in the constructor
            name = t
            #wcs_manual = self.WCS_Manual

            self.CatalogImages.append(
                {'path': op.dirname(self.Tile_Dict[t]['path']),
                 'name': name, #filename is the tilename
                 'tile': t,
                 'pos': self.Tile_Dict[t]['pos'], #the position tuple i.e. (0,3) or (2,8) ... in the name as 03 or 28
                 'filter': f,
                 'instrument': "HSC SSP",
                 'cols': [],
                 'labels': [],
                 'image': None,
                 'expanded': False,
                 'wcs_manual': self.WCS_Manual,
                 'aperture': self.mean_FWHM * 0.5 + 0.5, #since a radius, half the FWHM + 0.5" for astrometric error
                 'mag_func': hsc_count_to_mag,
                 'mag_depth': self.Tile_Dict[t]['depth'],
                 })

    def find_target_tile(self,ra,dec,return_all_matched_tiles=False):
        #assumed to have already confirmed this target is at least in coordinate range of this catalog
        #return at most one tile, but maybe more than one tract (for the catalog ... HSC does not completely
        #   overlap the tracts so if multiple tiles are valid, depending on which one is selected, you may
        #   not find matching objects for the associated tract)
        tile = None
        all_matched_tiles  = []
        tracts = []
        positions = []
        keys = []
        #note: there will usually be "duplicates" as each tile may have multiple filters
        #however, they all have 'r'
        for k in self.Tile_Dict.keys():
            # don't bother to load if ra, dec not in range
            try:
                #only check the 'r' fitler
                #NOTE this is just finding the tile, there is not a need for g vs r preference here
                if self.Tile_Dict[k]['filter'].lower() != 'r':
                    continue

                if self.Tile_Dict[k]['RA_max'] - self.Tile_Dict[k]['RA_min'] < 30: #30 deg as a big value
                                                                                   #ie. we are NOT crossing the 0 or 360 deg line
                    if not ((ra >= self.Tile_Dict[k]['RA_min']) and (ra <= self.Tile_Dict[k]['RA_max']) and
                            (dec >= self.Tile_Dict[k]['Dec_min']) and (dec <= self.Tile_Dict[k]['Dec_max'])) :
                        continue
                    else:
                        keys.append(k)
                else: # we are crossing the 0/360 boundary, so we need to be greater than the max (ie. between max and 360)
                      # OR less than the min (between 0 and the minimum)
                    if not ((ra <= self.Tile_Dict[k]['RA_min']) or (ra >= self.Tile_Dict[k]['RA_max']) and
                            (dec >= self.Tile_Dict[k]['Dec_min']) and (dec <= self.Tile_Dict[k]['Dec_max'])) :
                        continue
                    else:
                        keys.append(k)

            except:
                pass

        if len(keys) == 0: #we're done ... did not find any
            if return_all_matched_tiles:
                return None, None, None, None
            else:
                return None, None, None
        elif len(keys) == 1: #found exactly one
            tile = keys[0] #remember tile is a string ... there can be only one
            if return_all_matched_tiles:
                all_matched_tiles = copy.copy(keys)
            positions.append(self.Tile_Dict[tile]['pos'])
            tracts.append(self.Tile_Dict[tile]['tract']) #remember, tract is a list (there can be more than one)
        elif len(keys) > 1: #find the best one
            log.info("Multiple overlapping tiles %s. Sub-selecting tile with maximum angular coverage around target." %keys)
            # min = 9e9
            # #we don't have the actual corners anymore, so just assume a rectangle
            # #so there are 2 of each min, max coords. Only need the smallest distance so just sum one
            # for k in keys:
            #     tracts.append(self.Tile_Dict[k]['tract'])
            #     positions.append(self.Tile_Dict[k]['pos'])
            #     sqdist = (ra-self.Tile_Dict[k]['RA_min'])**2 + (dec-self.Tile_Dict[k]['Dec_min'])**2 + \
            #              (ra-self.Tile_Dict[k]['RA_max'])**2 + (dec-self.Tile_Dict[k]['Dec_max'])**2
            #     if sqdist < min:
            #         min = sqdist
            #         tile = k
            #
            #         min_dist = 9e9
            max_dist = 0
            tile = keys[0] #start with the first one
            if return_all_matched_tiles:
                all_matched_tiles = copy.copy(keys)
            for k in keys:
                tracts.append(self.Tile_Dict[k]['tract'])
                positions.append(self.Tile_Dict[k]['pos'])

                #should not be negative, but could be?
                #in any case, the min is the smallest distance to an edge in RA and Dec
                inside_ra = abs(min((ra-self.Tile_Dict[k]['RA_min']),(self.Tile_Dict[k]['RA_max']-ra)))
                inside_dec = min((dec-self.Tile_Dict[k]['Dec_min']),(self.Tile_Dict[k]['Dec_max']-dec))

                edge_dist = min(inside_dec,inside_ra)
                #we want the tile with the largest minium edge distance

                if edge_dist > max_dist and op.exists(self.Tile_Dict[k]['path']):
                    max_dist = edge_dist
                    tile = k

        else: #really?? len(keys) < 0 : this is just a sanity catch
            log.error("ERROR! len(keys) < 0 in cat_hsc::find_target_tile.")
            if return_all_matched_tiles:
                return None, None, None, None
            else:
                return None, None, None

        tile = tile.replace("-R-","-?-")
        if return_all_matched_tiles and all_matched_tiles is not None:
            for i in range(len(all_matched_tiles)):
                all_matched_tiles[i] = all_matched_tiles[i].replace("-R-","-?-")

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

        if return_all_matched_tiles:
            return tile, tracts, positions, all_matched_tiles
        else:
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

            mask_cutout = mask_image.get_mask_cutout(ra,dec,error,mask_image.image_location)

        except:
            log.info(f"Could not get mask cutout for tile ({tile})",exc_info=True)

        return mask_cutout


    #reworked as average of 3" (17pix), lsq and model mags
    def get_filter_flux(self, df):
        """

        :param df:
        :return:  flux in uJy
        """

        filter_fl = None
        filter_fl_err = None

        mag = None
        mag_bright = None
        mag_faint = None
        #filter_str = 'r' # has grizy, but we will use r (preferred) then g

        flux_list = []
        flux_err_list = []
        mag_list = []
        mag_err_list = []

        filter_str=None

        # if G.BANDPASS_PREFER_G:
        #     first = 'g'
        #     second = 'r'
        # else:
        #     first = 'r'
        #     second = 'g'

        try:
            filter_str = df['filter_str'].values[0]
            #the values here are in counts and I need to know how to convert ....
            filter_fl = df['src_ext_photometryKron_KronFlux_instFlux'].values[0]
            filter_fl_err = df['src_ext_photometryKron_KronFlux_instFluxErr'].values[0]

            #have to turn into mag ... need to know the filter
            mag = hsc_count_to_mag(filter_fl)
            #actual mag with the error, not just the error as a +/-
            mag_bright = hsc_count_to_mag(filter_fl+filter_fl_err)
            mag_faint = hsc_count_to_mag(filter_fl-filter_fl_err)

            try:
                sel = np.array(self.flags_table['src_id'] == df['src_id'].values[0])
                log.debug(f"*** reported flags: {np.array(np.where(self.flags_table['flags'][sel])[1])+1}")
                #+1 since the TFLAGSn are 1 based, [1] is just to deal with the array shape
            except:
                pass

        except:
            log.error("Exception in cat_hsc_nep.get_filter_flux", exc_info=True)
            return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

        try:
            log.debug(f"HSC SSP {filter_str} mag {mag},{mag_bright},{mag_faint}")
        except:
            pass

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str


    def get_zPDF(self,tract_ids, src_id,type=None):
        """
        scan the PDF root directory for all matches to th tile_id (there may be 0 to 3)
        take the mean if more than one and return the zPDF

        :param tract_ids: semicolon separated string of the numeric tile identifier (usually just one but could be more)
        :param src_id: the uniuqe identifier
        :param type: to be implemented later ... specify DNNZ, DEMP, or MIZUKI
        :return: zPDF and zbins or None, None
        """

        try:

            #scan for matching files
            log.info(f"Searching for zPDFs for src_id {src_id} in tracts {tract_ids}: {self.HSC_PHOTZ_PATH}")
            files = []

            if op.isdir(self.HSC_PHOTZ_PATH):
                for tid in tract_ids.split(";"):
                    fn = op.join(self.HSC_PHOTZ_PATH,f"**/*{tid}*fits")
                    #limit to just the mizuki ones for now
                    #all use different z ranges, z steps and can be wildly different in results
                    #fn = op.join(self.HSC_PHOTZ_PATH,f"**/*mizuki*/*{tid}*fits")
                    files = files + [f for f in glob.iglob(fn,recursive=True)]

            if len(files) == 0:
                log.info(f"No matching zPDF files found for tract(s) {tract_ids}.")
                return None, [], []

            #read each one in turn
            z_bins_list = []
            z_pdf_list = []
            #min_bins = np.inf
            log.info(f"Found {len(files)} zPDF files: {files}")
            for f in files:
                try: #is there any way to speed this up? and HDF5
                    idx = None
                    #first see if there is an .sid version ... faster to check
                    sid = f[:-5]+".sid"
                    if op.exists(sid):
                        log.debug(f"Checking {sid}")
                        src_ids = np.fromfile(sid, dtype=np.int64)
                        idx = np.where(src_ids==src_id)[0]
                        if len(idx)==0:
                            log.debug(f"{src_id} not found.")
                            continue
                        elif len(idx) > 1: #safety check
                            log.warning(f"{src_id} found {len(idx)} times in {sid}")
                            idx = None
                    elif G.LOCAL_DEVBOX:
                        log.info("***** Skipping HSC-SSP zPDFs due to large size and removet connection from local dev box. *****")
                        print("***** Skipping HSC-SSP zPDFs due to large size and removet connection from local dev box. *****")
                        return None,[],[]


                    #either no .sid file or we found the src_id
                    #turn off lazy loading ... unfortunately, to search, we need it all
                    #log.debug(f"*** {f}")
                    hdulist = astropyFITS.open(f, mode="readonly",memmap=True, lazy_load_hdus=True,
                                               ignore_missing_simple=True)
                    #log.debug("*** hdulist open complete")
                    if idx is not None and len(idx)==1: #idx is still an array here, with one element
                        # confirm the src_id
                        if src_id != hdulist[1].data[idx[0]][0]:
                            log.info(f"*.sid file index mismatch, will check *.fits file")
                            idx = None

                    if idx is None:
                        idx = np.where(np.array([x[0] for x in np.array(hdulist[1].data)]) == src_id)[0]
                        log.info("*** hdulist search complete")

                    if idx is None or len(idx) == 0:
                        hdulist.close()
                        continue
                    elif len(idx) > 1:  # safety check
                        log.warning(f"{src_id} found {len(idx)} times in {f}. Will not use.")
                        hdulist.close()
                        continue

                    idx = idx[0]
                    z_bins = [x[0] for x in np.array(hdulist[2].data)]
                    z_pdf = hdulist[1].data[idx][1:][0]
                    z_bins_list.append(np.array(z_bins))
                    z_pdf_list.append(np.array(z_pdf))
                    #min_bins = min(min_bins,len(z_bins))
                    hdulist.close()
                    log.info(f"Found matching src_id in {f}")
                    #break #just use the first one for now
                except: #this is due to idx being empty (can't index [0][0] on an empty array ... no matching idx were found)
                    log.info("***** Exception!",exc_info=True)
                    try:
                        hdulist.close()
                    except:
                        pass

                #alternate with Tables direct read
                # try: #is there any way to speed this up? and HDF5
                #     #turn off lazy loading ... unfortunately, to search, we need it all
                #     log.info(f"*** {f}")
                #
                #     t1 = astropy.table.Table.read(f,format="fits",hdu=1)
                #     log.info("*** table 1 load complete")
                #     idx = np.where(t1['ID']==src_id)
                #     if len(idx[0]) == 0:
                #         del t1
                #         continue
                #     idx = idx[0][0]
                #     log.info("*** search complete")
                #     z_pdf = t1['PDF'][idx] #hdulist[1].data[idx][1:][0]
                #
                #     t2 = astropy.table.Table.read(f,format="fits",hdu=2)
                #     z_bins = np.array(t2['BINS'])
                #
                #     z_bins_list.append(np.array(z_bins))
                #     z_pdf_list.append(np.array(z_pdf))
                #     #min_bins = min(min_bins,len(z_bins))
                #     del t1
                #     del t2
                #     log.info(f"Found matching src_id in {f}")
                #     #break #just use the first one for now
                # except: #this is due to idx being empty (can't index [0][0] on an empty array ... no matching idx were found)
                #     log.info("***** Exception!",exc_info=True)
                #     try:
                #         del t1
                #         del t2 #might not exist
                #     except:
                #         pass

            #average together
            #for HSC SSP the z bin scales are all the same (0.01 steps in z), but some go to z = 6 and some to z = 7
            #!!!NOPE this is NOT TRUE!!! some use steps of 0.01 and some are other steps
            #so need to rework this to interpolate to same grid and then average

            #interpolate and average
            if len(z_pdf_list) > 1:
                z_pdf_interp_list = []
                z_bins = np.arange(0.0,6.01,0.01)
                for pdf,bins in zip(z_pdf_list,z_bins_list):
                    try:
                        z_pdf_interp_list.append(np.clip(np.interp(z_bins,bins,pdf),0,1.0))
                    except:
                        log.debug("Exception.",exc_info=True)

                if len(z_pdf_interp_list) > 1:
                    z_pdf_avg = np.nanmean(z_pdf_interp_list, axis=0)
                    z_pdf_max = z_bins[np.argmax(z_pdf_avg)]
                elif len(z_pdf_interp_list) == 1:
                    #something went wrong
                    z_pdf_avg = z_pdf_interp_list[0]
                    z_pdf_max = z_bins[np.argmax(z_pdf_avg)]
                    log.debug("Partial failure in hsc_ssp get_zPDF(). Too few zPDFs in interpolation.")
                else:
                    #something went very wrong
                    z_pdf_max = None
                    z_pdf_avg = []
                    z_bins = []
                    log.debug("Failure in hsc_ssp get_zPDF(). No zPDFs in interpolation.")

                #testing only'
                # if True:
                #     try:
                #         plt.close('all')
                #         plt.figure(figsize=(8,3))
                #         for pdf,bins in zip(z_pdf_list,z_bins_list):
                #             plt.plot(bins,pdf,alpha=0.5)
                #
                #         plt.plot(z_bins,z_pdf_avg,label="avg")
                #         plt.legend()
                #         plt.tight_layout()
                #         plt.savefig("hsc_ssp_zpdf.png")
                #
                #     except:
                #         log.debug("+++++ Exception",exc_info=True)

            elif len(z_pdf_list) == 1 :
                z_pdf_avg = z_pdf_list[0]
                z_bins = z_bins_list[0]
                z_pdf_max = z_bins[np.argmax(z_pdf_avg)]
            else:
                log.info("No matching src_id found.")
                z_pdf_max = None
                z_pdf_avg = []
                z_bins = []

            return z_pdf_max, z_pdf_avg, z_bins

            #z_pdf_sum = np.sum([p[0:min_bins+1] for p in z_pdf_list],axis=0)
            #z_pdf_avg = z_pdf_sum/np.sum(z_pdf_sum)
            #z_bins = np.arange(0.0,(min_bins+1)/100.0,0.01)
            #z_pdf_max = z_bins[np.argmax(z_pdf_avg)]

            #we, however, limit to z = 3.6 ?? (do that later)
            return z_pdf_max, z_pdf_avg, z_bins

        except:
            log.info("Exception in cat_hsc_ssp HSC_SSP::get_zPDF()", exc_info=True)
            return None, [], []


    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        #even if not None, could be we need a different catalog, so check and append
        tile, tracts, positions, all_tiles  = self.find_target_tile(ra,dec, return_all_matched_tiles=True)

        if tile is None:
            log.info("Could not locate tile for HSC_SSP. Discontinuing search of this catalog.")
            return -1,None,None

        #could be none or could be not loaded yet
        #if self.df is None or not (self.Tile_Dict[tile]['tract'] in self.loaded_tracts):
        if self.df is None or not (set(tracts).issubset(self.loaded_tracts)):
            #self.read_main_catalog()
            #self.read_catalog(tract=self.Tile_Dict[tile]['tract'])
            if all_tiles is not None and len(all_tiles) > 0:
                for one_tile in all_tiles:
                    self.read_catalog(tile=one_tile,tract=tracts,position=positions)
            else:
                self.read_catalog(tile=tile, tract=tracts, position=positions)

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

        if self.df is not None:
            try:
                self.dataframe_of_bid_targets = \
                    self.df[  (self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max)
                        & (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)
                        ].copy()
                #may contain duplicates (across tiles)
                #remove duplicates (assuming same RA,DEC between tiles has same data)
                #so, different tiles that have the same ra,dec and filter get dropped (keep only 1)
                #but if the filter is different, it is kept

                #this could be done at construction time, but given the smaller subset I think
                #this is faster here
                try:
                    self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.drop_duplicates(
                        subset=['RA','DEC'])
                except:
                    pass


                #relying on auto garbage collection here ...
                try:
                    self.dataframe_of_bid_targets_unique = self.dataframe_of_bid_targets.copy()
                    self.dataframe_of_bid_targets_unique = \
                        self.dataframe_of_bid_targets_unique.drop_duplicates(subset=['RA','DEC'])#,'FILTER'])
                    self.num_targets = self.dataframe_of_bid_targets_unique.iloc[:,0].count()
                except:
                    self.num_targets = 0

            except:
                log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.dataframe_of_bid_targets_unique is not None:
            #self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra, dec)

            #log.info(self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
            #         ". Found = %d" % (self.num_targets))
            if self.num_targets > 0:
                log.info(f"{self.Name} searching for objects in [{ra_min} - {ra_max}, {dec_min} - {dec_max}]. "
                         f"Found = {self.num_targets}. srcID = {list(self.dataframe_of_bid_targets_unique['src_id'].values)}")
            else:
                log.info(f"{self.Name} searching for objects in [{ra_min} - {ra_max}, {dec_min} - {dec_max}]. "
                         f"Found = {self.num_targets}.")

        return self.num_targets, self.dataframe_of_bid_targets_unique, None



    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None,detobj=None):

        self.clear_pages()
        num_targets, _, _ = self.build_list_of_bid_targets(target_ra, target_dec, error)
        #could be there is no matching tile, if so, the dataframe will be none

        #if (num_targets == 0) or
        if (self.dataframe_of_bid_targets_unique is None):
            ras = []
            decs = []
        else:
            try:
                ras = self.dataframe_of_bid_targets_unique.loc[:, ['RA']].values
                decs = self.dataframe_of_bid_targets_unique.loc[:, ['DEC']].values
            except:
                ras = []
                dec = []

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

            wcs_idx = 1

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
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0])]
                #multiple filters

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                #add flux (cont est)
                try:
                    #fluxes for HSC SSP are in micro-Jansky
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
                        #fluxes for HSC SSP are in micro-Jansky
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

                            if G.CHECK_ALL_CATALOG_BID_Z: #only load if we are going to use it
                                bid_target.phot_z, bid_target.phot_z_pdf_pz, bid_target.phot_z_pdf_z = \
                                    self.get_zPDF(tract_ids=df['tract'].values[0], src_id=df['src_id'].values[0])
                                #notice for the 'tract' we want it as the array

                            if target_w:

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
                                    #
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


                                # # build EW error from lineFlux_err and aperture estimate error
                                # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                                # try:
                                #     ew_obs_err = abs(ew_obs * np.sqrt(
                                #         (lineFlux_err / target_flux) ** 2 +
                                #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                                # except:
                                #     ew_obs_err = 0.

                                ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w,
                                                               filter_str, filter_fl_cgs,filter_fl_cgs_unc)

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
                                bid_target.add_filter('HSC SSP',filter_str,filter_fl_cgs,filter_fl_err)
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

        if self.INCLUDE_KPNO_G:
            try:

                #IF KPNO g-band is available,
                #   advance the index by 1 and insert the KPNO g-band image in 1st position AFTER the "master" cutout (made
                #   from only HSC)

                kpno = cat_kpno.KPNO()
                kpno_cuts = kpno.get_cutouts(ra,dec,window/3600.0,aperture=kpno.mean_FWHM * 0.5 + 0.5,filter='g',first=True,detobj=detobj)

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

            wcs_idx = 1

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
                                                     aperture=aperture,mag_func=mag_func,return_details=True,detobj=detobj)

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
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None) and \
                        ((i['filter'] == 'r') or (i['filter']=='g')):
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
                    self.master_cutout,_,_, _ = sci.get_cutout(ra, dec, error, window=window, copy=True,reset_center=False,detobj=detobj)
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
                if G.BANDPASS_PREFER_G:
                    first = 'g'
                    second = 'r'
                else:
                    first = 'r'
                    second = 'g'

                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == first)]
                if (df is None) or (len(df) == 0):
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == second)]
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

    def get_single_cutout(self, ra, dec, window, catalog_image,aperture=None,error=None,do_sky_subtract=True,detobj=None):
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

        # try:
        #     wcs_idx = self.Filter_HDU_Image_Idx[catalog_image['filter']]
        # except:
        #     wcs_idx = 0

        wcs_idx = 1

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
                                             mag_func=mag_func,copy=True,return_details=True,detobj=detobj)
            # don't need pix_counts or mag, etc here, so don't pass aperture or mag_func
            if cutout is not None:  # construct master cutout
                d['cutout'] = cutout
                details['catalog_name']=self.name
                details['filter_name']=catalog_image['filter']
                d['mag_limit']=self.get_mag_limit([catalog_image['name'],catalog_image['filter']],mag_radius*2.)
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

    def get_cutouts(self,ra,dec,window,aperture=None,filter=None,first=False,error=None,do_sky_subtract=True,detobj=None):
        l = list()

        tile, tracts, positions = self.find_target_tile(ra, dec)

        #tile returns with just the R band label: i.e. calexp-HSC-R-572-7,0.fits
        #so have to swap out 'R' for each of the other bands

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


        if aperture == -1:
            try:
                aperture = self.mean_FWHM * 0.5 + 0.5
            except:
                pass


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

                    #swap out the base tile name for the other filters (f)

                    i = self.CatalogImages[
                        next(i for (i, d) in enumerate(self.CatalogImages)
                             if ((d['filter'] == f) and (d['tile'] == tile.replace("?",f.upper()))))]
                    if i is not None:
                        cutout = self.get_single_cutout(ra, dec, window, i, aperture,error,detobj=detobj)

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

                l.append(self.get_single_cutout(ra,dec,window,i,aperture,detobj=detobj))

        return l
