from __future__ import print_function

try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_base
    from elixer import match_summary
    from elixer import line_prob
    from elixer import utilities
except:
    import global_config as G
    import science_image
    import cat_base
    import match_summary
    import line_prob
    import utilities

import os.path as op
import copy
import scipy
import io


GOODS_N_BASE_PATH = G.GOODS_N_BASE_PATH
GOODS_N_CAT = op.join(GOODS_N_BASE_PATH,"photometry/CANDELS.GOODSN.F160W.v1_1.photom.cat")
GOODS_N_IMAGES_PATH = op.join(GOODS_N_BASE_PATH, "images")
GOODS_N_PHOTOZ_CAT = op.join(GOODS_N_BASE_PATH , "photoz/zcat_GOODSN_v2.0.cat")
GOODS_N_PHOTOZ_ZPDF_PATH = op.join(GOODS_N_BASE_PATH, "photoz/zPDFs/")

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
log.setlevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field


#todo: update with aperture on photometry
#todo: currently astropy does not like the drz fits files and throws exception with the aperture

def goodsn_f606w_count_to_mag(count,cutout=None,headers=None):
    if count is not None:
        #if cutout is not None:
        #get the conversion factor, each tile is different
        try:
            photoflam = float(headers[0]['PHOTFLAM']) #inverse sensitivity, ergs / cm2 / Ang / electron
            photozero = float(headers[0]['PHOTZPT']) #/ ST magnitude zero point


            if not isinstance(count, float):
                count = count.value

            if count > 0:
                flux = photoflam*count
                # convert from per Angstrom to per Hertz
                # F_v  = F_l * (l^2)/c ... but which lambda to use? center of the filter? ... what is the iso-lambda
                # iso = 5778.3 AA?
                # c = scipy.constants.c * 1e10 #in AA
                # flux = flux * (5778.3**2.)/c * 1e-23 #to Jansky
                #then
                return  -2.5 * np.log10(flux) + photozero
               # return -2.5 * np.log10(flux/3631.0)
            else:
                return 99.9  # need a better floor
        except:
            log.warning("Exception in goodsn_count_to_mag",exc_info=True)
            return 99.9


class GOODS_N(cat_base.Catalog):

    # class variables
    MainCatalog = GOODS_N_CAT
    Name = "GOODS-N"
    mean_FWHM = 0.15 #typical use for photometric aperture, but is too good here ... objects that are point
                    #sources may be resolved with HST

    # if multiple images, the composite broadest range (filled in by hand)
    Cat_Coord_Range = {'RA_min': 188.915597, 'RA_max': 189.563471, 'Dec_min': 62.091438, 'Dec_max': 62.388316}
    Image_Coord_Range = {'RA_min':188.862 , 'RA_max': 189.605, 'Dec_min': 62.066, 'Dec_max': 62.409}

    WCS_Manual = True
    EXPTIME_F606W = 236118.0 #289618.0
    CONT_EST_BASE = 2.8e-21#3.3e-21
    BidCols = ["ID", "IAU_designation", "RA", "DEC",
               "CFHT_U_FLUX", "CFHT_U_FLUXERR",
               "IRAC_CH1_FLUX", "IRAC_CH1_FLUXERR", "IRAC_CH2_FLUX", "IRAC_CH2_FLUXERR",
               "ACS_F606W_FLUX", "ACS_F606W_FLUXERR",
               "ACS_F814W_FLUX", "ACS_F814W_FLUXERR",
               "WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR",
               "WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR",
               "WC3_F160W_FLUX", "WFC3_F160W_FLUXERR",
               "DEEP_SPEC_Z"]  # NOTE: there are no F105W values

    CatalogImages = [
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f435w_060mas_v1.3_drz.fits',
         'filter': 'f435w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F435W_FLUX", "ACS_F435W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': 0.0,
         'mag_func': None,
         'sky_subtract': False
         },
        # / PHOTOMETRY    KEYWORDS
        # PHOTMODE = 'ACS WFC1 F606W MJD#52599.1628' / observation con
        # PHOTFLAM = 7.7265099E-20 / inverse sensitivity, ergs / cm2 / Ang / electron
        # PHOTZPT = -2.1100000E+01 / ST magnitude zero point
        # PHOTPLAM = 5.9194604E+03 / Pivot wavelength(Angstroms)
        # PHOTBW = 6.7240521E+02 / RMS bandwidth of filter plus detector
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f606w_060mas_v2.0_drz.fits',
         'filter': 'f606w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F606W_FLUX", "ACS_F606W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture': mean_FWHM* 0.5 + 0.5, # since a radius, half the FWHM + 0.5" for astrometric error
         'mag_func': goodsn_f606w_count_to_mag,
         'sky_subtract': False
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f775w_060mas_v2.0_drz.fits',
         'filter': 'f775w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F775W_FLUX", "ACS_F775W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture':0.0,
         'mag_func': None,
         'sky_subtract': False
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_acs_wfc_f814w_060mas_v2.0_drz.fits',
         'filter': 'f814w',
         'instrument': 'ACS WFC',
         'cols': ["ACS_F814W_FLUX", "ACS_F814W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture':0.0,
         'mag_func': None,
         'sky_subtract': False
         },
        #omit 850LP (per Steve) ... long exposure due to low sensitivity
       # {'path': GOODS_N_IMAGES_PATH,
       #  'name': 'goodsn_all_acs_wfc_f850l_060mas_v2.0_drz.fits',
       #  'filter': 'f850lp',
       #  'instrument': 'ACS WFC',
       #  'cols': ["ACS_F850LP_FLUX", "ACS_F850LP_FLUXERR"],
       #  'labels': ["Flux", "Err"],
       #  'image': None,
       #  },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_wfc3_ir_f105w_060mas_v1.0_drz.fits',
         'filter': 'f105w',
         'instrument': 'WFC3',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture':0.0,
         'mag_func': None,
         'sky_subtract': False
         },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_wfc3_ir_f125w_060mas_v1.0_drz.fits',
         'filter': 'f125w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F125W_FLUX", "WFC3_F125W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture':0.0,
         'mag_func': None,
         'sky_subtract': False
         },
        #omit 140w per Steve
        #{'path': GOODS_N_IMAGES_PATH,
        # 'name': 'goodsn_all_wfc3_ir_f140w_060mas_v1.0_drz.fits',
        # 'filter': 'f140w',
        # 'instrument': 'WFC3',
        # 'cols': ["WFC3_F140W_FLUX", "WFC3_F140W_FLUXERR"],
        # 'labels': ["Flux", "Err"],
        # 'image': None
        # },
        {'path': GOODS_N_IMAGES_PATH,
         'name': 'goodsn_all_wfc3_ir_f160w_060mas_v1.0_drz.fits',
         'filter': 'f160w',
         'instrument': 'WFC3',
         'cols': ["WFC3_F160W_FLUX", "WFC3_F160W_FLUXERR"],
         'labels': ["Flux", "Err"],
         'image': None,
         'expanded': False,
         'wcs_manual': True,
         'aperture':0.0,
         'mag_func': None,
         'sky_subtract': False
         }
    ]

    PhotoZCatalog = GOODS_N_PHOTOZ_CAT
    SupportFilesLocation = GOODS_N_PHOTOZ_ZPDF_PATH

    def __init__(self):
        super(GOODS_N, self).__init__()

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
        try:
            f = open(catalog_loc, mode='r')
        except:
            log.error(name + " Exception attempting to open catalog file: " + catalog_loc, exc_info=True)
            return None

        line = f.readline()
        while '#' in line:
            skip += 1
            toks = line.split()
            if (len(toks) > 2) and toks[1].isdigit():  # format:   # <id number> <column name>
                idx.append(toks[1])
                header.append(toks[2])
            line = f.readline()

        f.close()

        try:
            df = pd.read_csv(catalog_loc, names=header,
                             delim_whitespace=True, header=None, index_col=None, skiprows=skip)

            old_names = ['ID (H-band SExtractor ID)', 'IAU_Name','RA (J2000, H-band)', 'DEC (J2000, H-band)']
            new_names = ['ID', 'IAU_designation','RA', 'DEC']
            df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

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
        filter_str = 'ACS_F606W_FLUX'
        try:
            filter_fl = df[filter_str].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
            filter_fl_err = df['ACS_F606W_FLUXERR'].values[0]
            mag, mag_plus, mag_minus = self.micro_jansky_to_mag(filter_fl, filter_fl_err)
        except:  # not the EGS df, try the CFHTLS
            pass

        if filter_fl is None:
            try:
                filter_fl = self.obs_mag_to_micro_Jy(df['G'].values[0])
                filter_fl_err = abs(filter_fl - self.obs_mag_to_micro_Jy(df['G'].values[0] - df['eG'].values[0]))
            except:
                pass

        return filter_fl, filter_fl_err, mag, mag_plus, mag_minus, filter_str


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

            # ID matches between both catalogs
            self.dataframe_of_bid_targets_photoz = \
                self.df_photoz[(self.df_photoz['ID'].isin(self.dataframe_of_bid_targets['ID']))].copy()
        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.dataframe_of_bid_targets is not None:
            self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
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

            entry = self.build_cat_summary_figure(cat_match, target_ra, target_dec, error, ras, decs,
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
                                                                        target_w=target_w, target_flux=target_flux,detobj=detobj)
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

    def build_cat_summary_figure (self, cat_match,ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...

        Returns the matplotlib figure. Due to limitations of matplotlib pdf generation, each figure = 1 page'''

        # note: error is essentially a radius, but this is done as a box, with the 0,0 position in lower-left
        # not the middle, so need the total length of each side to be twice translated error or 2*2*error
        # ... change to 1.5 times twice the translated error (really sqrt(2) * 2* error, but 1.5 is close enough)
        window = error * 3
        target_box_side = error/4.0 #basically, the box is 1/32 of the window size

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

        cont_est = -1
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
        text = plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

        ref_exptime = None
        total_adjusted_exptime = None

        best_plae_poii = None
        best_plae_poii_filter = '-'

        # add the bid targets
        bid_colors = self.get_bid_colors(len(bid_ras))

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
                i['image'] = science_image.science_image(wcs_manual=self.WCS_Manual,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            # sci.load_image(wcs_manual=True)
            #log.info("Reminder: aperture issue with .drz fits file, so no forced aperture magnitude.")
            cutout, pix_counts, mag, mag_radius, details = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture, mag_func=mag_func,
                                                    do_sky_subtract=do_sky_subtract,return_details=True)

            bid_target = None
            cutout_ewr = None
            cutout_ewr_err = None
            cutout_plae = None

            try: #update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None)\
                        and (((i['instrument'] == 'CFHTLS') and (i['filter'] == 'g')) or (i['filter'] == 'f606w')) :
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

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
                        line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                                           ew_obs=ew_obs,
                                           lineFlux_err= lineFlux_err,
                                           ew_obs_err= ew_obs_err,
                                           c_obs=None, which_color=None, addl_fluxes=addl_flux,
                                           addl_wavelengths=addl_waves,addl_errors=addl_ferr,sky_area=None,
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

                    if best_plae_poii is None or i['filter'] == 'f606w':
                        best_plae_poii = bid_target.p_lae_oii_ratio
                        best_plae_poii_filter = i['filter']



                    cat_match.add_bid_target(bid_target)

                    try:  # no downstream edits so they can both point to same bid_target
                        detobj.bid_target_list.append(bid_target)
                    except:
                        log.warning("Unable to append bid_target to detobj.", exc_info=True)
            except:
                log.debug('Could not build exact location photometry info.',exc_info=True)

            #move outside loop so only happens once
            # if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
            #     text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g (%s)"
            #                   % (best_plae_poii, best_plae_poii_filter))

            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2



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
                    ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                    total_adjusted_exptime += sci.exptime / ref_exptime

                plt.subplot(gs[1:, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])

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
                    if (details['sep_objects'] is not None) and (details['sep_obj_idx'] is not None):
                        self.add_elliptical_aperture_positions(plt, details['sep_objects'], details['sep_obj_idx'],
                                                               mag, cx, cy, cutout_ewr, cutout_plae)
                    else:
                        self.add_aperture_position(plt, mag_radius, mag, cx, cy, cutout_ewr, cutout_plae)

                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                #plt.plot(0, 0, "r+")
                self.add_zero_position(plt)
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


        if (not G.ZOO) and (best_plae_poii is not None):
            text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.4g (%s)"
                          % (best_plae_poii, best_plae_poii_filter))

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            return None  # empty (catch_all) will produce fiber locations
            #still need to plot relative fiber positions here
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

            plt.gca().add_patch(plt.Rectangle((zero_x - rx,  zero_y - ry), width=rx * 2, height=ry * 2,
                                              angle=0, color='red', fill=False,linewidth=3))

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
                    log.error("Exception!",exc_info=True)

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

                # try:
                #     filter_fl = df['ACS_F606W_FLUX'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
                #     filter_fl_err = df['ACS_F606W_FLUXERR'].values[0]
                # except:
                #     filter_fl = 0.0
                #     filter_fl_err = 0.0


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
                        filter_fl_cgs = self.micro_jansky_to_cgs(filter_fl,target_w)
                        filter_fl_cgs_unc = self.micro_jansky_to_cgs(filter_fl_err, target_w)
                        #assumes no error in wavelength or c




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
                        bid_target.bid_flux_est_cgs_unc = filter_fl_cgs_unc
                        bid_target.bid_filter = filter_str
                        bid_target.bid_mag = filter_mag
                        bid_target.bid_mag_err_bright = filter_mag_bright
                        bid_target.bid_mag_err_faint = filter_mag_faint
                        if spec_z >= 0.0:
                            bid_target.spec_z = spec_z
                        if (z_photoz_weighted is not None) and (z_photoz_weighted >= 0.0):
                            bid_target.phot_z = z_photoz_weighted


                        try:
                            ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
                            ew_u = abs(ew * np.sqrt(
                                        (detobj.estflux_unc / target_flux) ** 2 +
                                        (filter_fl_err / filter_fl) ** 2))

                            bid_target.bid_ew_lya_rest = ew
                            bid_target.bid_ew_lya_rest_err = ew_u

                            text = text + utilities.unc_str((ew,ew_u)) + "$\AA$\n"
                        except:
                            log.debug("Exception computing catalog EW: ",exc_info=True)
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

                        bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii, plae_errors = \
                            line_prob.prob_LAE(wl_obs=target_w,
                                               lineFlux=target_flux,
                                               ew_obs=ew_obs,
                                               lineFlux_err=lineFlux_err,
                                               ew_obs_err=ew_obs_err,
                                               c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                                               addl_fluxes=addl_flux, addl_errors=addl_ferr, sky_area=None,
                                               cosmo=None, lae_priors=None,
                                               ew_case=None, W_0=None,
                                               z_OII=None, sigma=None, estimate_error=True)

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
                                                                               target_w),
                                                      self.micro_jansky_to_cgs(df[c['cols'][1]].values[0],
                                                                               target_w))
                            except:
                                log.debug('Could not add filter info to bid_target.')

                        cat_match.add_bid_target(bid_target)
                        try: # no downstream edits so they can both point to same bid_target
                            detobj.bid_target_list.append(bid_target)
                        except:
                            log.warning("Unable to append bid_target to detobj.",exc_info=True)
                else:
                    text += "N/A\nN/A\n"

                # if filter_mag != 0:
                try:
                    text = text + "%0.2f(%0.2f,%0.2f)\n" % (filter_mag, filter_mag_bright, filter_mag_faint)
                except:
                    log.warning("Magnitude info is none: mag(%s), mag_bright(%s), mag_faint(%s)"
                                % (filter_mag, filter_mag_bright, filter_mag_faint))
                    text += "No mag info\n"
                # else:
                #    text = text + "%g(%g) $\\mu$Jy\n" % (filter_fl, filter_fl_err)

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    text += "%0.4g\n" % (bid_target.p_lae_oii_ratio)
                else:
                    text += "\n"
            else:
                text = "%s\n%f\n%f\n" % ("--", r, d)

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

            if df_photoz is not None:
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
# end class GOODS_N
#######################################
