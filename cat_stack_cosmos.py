from __future__ import print_function

import global_config as G
import os.path as op

#STACK_COSMOS_BASE_PATH = G.STACK_COSMOS_BASE_PATH
#STACK_COSMOS_IMAGE = G.STACK_COSMOS_BASE_PATH#op.join(G.STACK_COSMOS_BASE_PATH,"COSMOS_g_sci.fits")
#STACK_COSMOS_CAT = G.STACK_COSMOS_CAT_PATH#op.join(G.STACK_COSMOS_CAT_PATH,"cat_g.fits")


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

import line_prob

#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

import cat_base
import match_summary

def cosmos_g_count_to_mag(count,cutout=None,sci_image=None):
    #nanofact = 334.116462522 #counts to nano-janksy from g fits header [NANOFACT]
    magzero = 31.4
    if count is not None:

        try:
            # gain = float(sci_image[0].header['GAIN'])
            #nanofact = float(sci_image[0].header['NANOFACT'])
            magzero = float(sci_image[0].header['MAGZERO'])
            #exptime = float(sci_image[0].header['EXPTIME'])
        except:
            # gain = 1.0
            nanofact = 0.0
            log.error("Exception in shela_count_to_mag", exc_info=True)
            return 99.9

        if count > 0:
            #return -2.5 * np.log10(count*nanofact) + magzero
            #counts for cosmos ALREADY in nanojansky
            return -2.5 * np.log10(count) + magzero
        else:
            return 99.9  # need a better floor

class STACK_COSMOS(cat_base.Catalog):

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
    #only use g-band catalog (best overlap with HETDEX wavelengths)
    STACK_COSMOS_CAT = op.join(G.STACK_COSMOS_CAT_PATH, "cat_g.fits")
    STACK_COSMOS_IMAGE_PATH = G.STACK_COSMOS_BASE_PATH
    #STACK_COSMOS_IMAGE = op.join(STACK_COSMOS_IMAGE_PATH, "COSMOS_g_sci.fits")

    MainCatalog = STACK_COSMOS_CAT
    Name = "STACK_COSMOS"
    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}
    Cat_Coord_Range = {'RA_min': 149.005021, 'RA_max': 151.275747, 'Dec_min': 1.150460, 'Dec_max': 3.242518}
    WCS_Manual = False

    CONT_EST_BASE = 0.0

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
         'name': 'COSMOS_u_sci.fits',
         'filter': 'u',
         'instrument': '',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': False,
         'aperture': 1.0
         # 'frame': 'icrs'
         },
        {'path': STACK_COSMOS_IMAGE_PATH,
         'name': 'COSMOS_g_sci.fits',
         'filter': 'g',
         'instrument': '',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': False,
         'aperture': 1.0,
         'mag_func': cosmos_g_count_to_mag
         #'frame': 'icrs'
        },
        {'path': STACK_COSMOS_IMAGE_PATH,
         'name': 'COSMOS_r_sci.fits',
         'filter': 'r',
         'instrument': '',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': False,
         'aperture': 1.0
         # 'frame': 'icrs'
         },
        {'path': STACK_COSMOS_IMAGE_PATH,
         'name': 'COSMOS_i_sci.fits',
         'filter': 'i',
         'instrument': '',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': False,
         'aperture': 1.0
         # 'frame': 'icrs'
         },
        {'path': STACK_COSMOS_IMAGE_PATH,
         'name': 'COSMOS_z_sci.fits',
         'filter': 'z',
         'instrument': '',
         'cols': [],
         'labels': [],
         'image': None,
         'expanded': False,
         'wcs_manual': False,
         'aperture': 1.0
         # 'frame': 'icrs'
         }
    ]

    def __init__(self):
        super(STACK_COSMOS, self).__init__()

        self.dataframe_of_bid_targets = None
        self.dataframe_of_bid_targets_photoz = None
        # self.table_of_bid_targets = None
        self.num_targets = 0

        #now only on demand
        #self.read_main_catalog()

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
            lookup_table = Table([table['NUMBER'], table['ALPHA_J2000'], table['DELTA_J2000'],
                                  table['FLUX_AUTO'],table['FLUXERR_AUTO'],
                                  table['MAG_AUTO'],table['MAGERR_AUTO']])
            df = lookup_table.to_pandas()
            old_names = ['NUMBER', 'ALPHA_J2000', 'DELTA_J2000']
            new_names = ['ID', 'RA', 'DEC']
            df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
            cls.AstroTable = table
        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return df

    # def sort_bid_targets_by_likelihood(self, ra, dec):
    #     # right now, just by euclidean distance (ra,dec are of target)
    #     #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)
    #     self.dataframe_of_bid_targets['distance'] = np.sqrt((self.dataframe_of_bid_targets['RA'] - ra) ** 2 +
    #                                                         (self.dataframe_of_bid_targets['DEC'] - dec) ** 2)
    #     self.dataframe_of_bid_targets = self.dataframe_of_bid_targets.sort_values(by='distance', ascending=True)

    def get_filter_flux(self,df): #right now, only g-band catalog

        filter_fl = None
        filter_fl_err = None
        mag = None
        mag_bright = None
        mag_faint = None
        filter_str = None
        try:

            filter_str = 'g'
            dfx = df
            #dfx = df.loc[df['FILTER']==filter_str]
            #
            #
            # if (dfx is None) or (len(dfx)==0):
            #     filter_str = 'r'
            #     dfx = df.loc[df['FILTER'] == filter_str]
            #
            # if (dfx is None) or (len(dfx)==0):
            #     filter_str = '?'
            #     log.error("Neither g-band nor r-band filter available.")
            #     return filter_fl, filter_fl_err, mag, mag_plus, mag_minus, filter_str

            filter_fl = dfx['FLUX_AUTO'].values[0]  # in micro-jansky or 1e-29  erg s^-1 cm^-2 Hz^-2
            filter_fl_err = dfx['FLUXERR_AUTO'].values[0]

            mag = dfx['MAG_AUTO'].values[0]
            mag_faint = dfx['MAGERR_AUTO'].values[0]
            mag_bright = -1 * mag_faint

            #something is way wrong with the MAG_AUTO
            mag, mag_bright, mag_faint= self.micro_jansky_to_mag(filter_fl,filter_fl_err)

        except: #not the EGS df, try the CFHTLS
            pass

        return filter_fl, filter_fl_err, mag, mag_bright, mag_faint, filter_str

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        if self.df is None:
            self.read_main_catalog()

        error_in_deg = np.float64(error) / 3600.0

        self.dataframe_of_bid_targets = None
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

    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None):

        self.clear_pages()
        self.build_list_of_bid_targets(target_ra, target_dec, error)

        if (self.dataframe_of_bid_targets is None):
            return None

        ras = self.dataframe_of_bid_targets.loc[:, ['RA']].values
        decs = self.dataframe_of_bid_targets.loc[:, ['DEC']].values

        # display the exact (target) location
        if G.SINGLE_PAGE_PER_DETECT:

            entry = self.build_cat_summary_figure(cat_match, target_ra, target_dec, error, ras, decs,
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux, )
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")

        if entry is not None:
            self.add_bid_entry(entry)

        # if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
        #    entry = self.build_multiple_bid_target_figures_one_line(cat_match,ras, decs, error,
        #                                                       target_ra=target_ra, target_dec=target_dec,
        #                                                       target_w=target_w, target_flux=target_flux)
        #    if entry is not None:
        #        self.add_bid_entry(entry)
        # else: #each bid taget gets its own line

        if G.SINGLE_PAGE_PER_DETECT:
            entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                    target_ra=target_ra, target_dec=target_dec,
                                                                    target_w=target_w, target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)

        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):  # each bid taget gets its own line

            bid_colors = self.get_bid_colors(len(ras))
            number = 0
            for r, d in zip(ras, decs):
                number += 1
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

                print("Building report for bid target %d in %s" % (base_count + number, self.Name))

                if G.SINGLE_PAGE_PER_DETECT and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
                    log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

                if entry is not None:
                    self.add_bid_entry(entry)

        return self.pages

    def build_cat_summary_figure (self, cat_match,ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None):
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
        if cols < 7: cols = 7

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

        if target_flux is not None:
            cont_est = self.CONT_EST_BASE*3 #self.get_f606w_max_cont(self.EXPTIME_F606W, 3, self.CONT_EST_BASE)
            if cont_est != 0:
                title += "  Minimum (no match) 3$\sigma$ rest-EW: "
                title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A"
            else:
                title += "  No continuum floor baseline defined."

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

        index = 0 #start in the 2nd box which is index 1 (1st box is for the fiber position plot)
        for i in self.CatalogImages:  # i is a dictionary
            index += 1

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

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag, mag_radius = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func)

            try: #update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None)\
                        and (i['instrument'] == 'CFHTLS') and (i['filter'] == 'g'):
                    #make a "blank" catalog match (e.g. at this specific RA, Dec (not actually from catalog)
                    bid_target = match_summary.BidTarget()
                    bid_target.catalog_name = self.Name
                    bid_target.bid_ra = 666 #nonsense RA
                    bid_target.bid_dec = 666 #nonsense Dec
                    bid_target.distance = 0.0
                    if mag < 99:
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag,target_w)
                    else:
                        bid_target.bid_flux_est_cgs = cont_est

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

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = \
                        line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                                           ew_obs=(target_flux / bid_target.bid_flux_est_cgs),
                                           c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                                           addl_fluxes=addl_flux, addl_errors=addl_ferr,sky_area=None,
                                           cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                                           sigma=None)

                    if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                        text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g" % (bid_target.p_lae_oii_ratio))

                    cat_match.add_bid_target(bid_target)
            except:
                log.debug('Could not build exact location photometry info.',exc_info=True)


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
                    self.master_cutout,_,_,_ = sci.get_cutout(ra, dec, error, window=window, copy=True)
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
                if pix_counts is not None:
                    self.add_aperture_position(plt,mag_radius,mag)
                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            #still need to plot relative fiber positions here
            plt.subplot(gs[1:, 0])
            return self.build_empty_cat_summary_figure(ra, dec, error, bid_ras, bid_decs, target_w=target_w,
                                           fiber_locs=fiber_locs)
        else:
            self.master_cutout.data /= total_adjusted_exptime

        plt.subplot(gs[1:, 0])
        self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, self.master_cutout)

        # complete the entry
        plt.close()
        return fig


    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None,
                                                       target_dec=None,
                                                       target_w=0, target_flux=None):

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

            # col(0) = "labels", 1..3 = bid targets, 4..5= Zplot
            gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.5)

            # entry text
            font = FontProperties()
            font.set_family('monospace')
            font.set_size(12)

            # row labels
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
                       "Est OII rest-EW\n" + \
                       "mag\n"
            else:
                text = "Separation\n" + \
                       "1-p(rand)\n" + \
                       "RA, Dec\n" + \
                       "Spec z\n" + \
                       "Photo z\n" + \
                       "Est LyA rest-EW\n" + \
                       "Est OII rest-EW\n" + \
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
                spec_z = 0.0

                try:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                           (self.dataframe_of_bid_targets['DEC'] == d[0])]

                    idnum = df['ID'].values[0]  # to matchup in photoz catalog
                except:
                    log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                    continue  # this must be here, so skip to next ra,dec

                df_photoz = None
                # try: #don't have photoz???
                #     # note cannot dirctly use RA,DEC as the recorded precission is different (could do a rounded match)
                #     # but the idnums match up, so just use that
                #     df_photoz = self.dataframe_of_bid_targets_photoz.loc[
                #         self.dataframe_of_bid_targets_photoz['ID'] == idnum]
                #
                #     if len(df_photoz) == 0:
                #         log.debug("No conterpart found in photoz catalog; RA=%f , Dec =%f" % (r[0], d[0]))
                #         df_photoz = None
                # except:
                #     log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                #     df_photoz = None
                #
                # if df_photoz is not None:
                #     photoz_file = df_photoz['file'].values[0]
                #     z_best = df_photoz['z_best'].values[0]
                #     z_best_type = df_photoz['z_best_type'].values[0]  # s = spectral , p = photometric?
                #     z_photoz_weighted = df_photoz['mFDa4_z_weight']

                if df is not None:
                    text = ""

                    if G.ZOO:
                        text = text + "%g\"\n%0.3f\n" \
                               % (df['distance'].values[0] * 3600., df['dist_prior'].values[0])
                    else:
                        text = text + "%g\"\n%0.3f\n%f, %f\n" \
                               % (df['distance'].values[0] * 3600., df['dist_prior'].values[0],
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
                        filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = \
                            self.get_filter_flux(df)
                    except:
                        filter_fl = 0.0
                        filter_fl_err = 0.0
                        filter_mag = 0.0
                        filter_mag_bright = 0.0
                        filter_mag_faint = 0.0
                        filter_str = "NA"

                    bid_target = None
                    if (target_flux is not None) and (filter_fl != 0.0):
                        if (filter_fl is not None):  # and (filter_fl > 0):
                            filter_fl_cgs = self.micro_jansky_to_cgs(filter_fl,
                                                                     target_w)  # filter_fl * 1e-29 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                            text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

                            if target_w >= G.OII_rest:
                                text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.OII_rest))
                            else:
                                text = text + "N/A\n"

                            # bid target info is only of value if we have a flux from the emission line
                            bid_target = match_summary.BidTarget()
                            bid_target.catalog_name = self.Name
                            bid_target.bid_ra = df['RA'].values[0]
                            bid_target.bid_dec = df['DEC'].values[0]
                            bid_target.distance = df['distance'].values[0] * 3600
                            bid_target.bid_flux_est_cgs = filter_fl_cgs
                            bid_target.bid_filter = filter_str
                            bid_target.bid_mag = filter_mag
                            bid_target.bid_mag_err_bright = filter_mag_bright
                            bid_target.bid_mag_err_faint = filter_mag_faint

                            addl_waves = None
                            addl_flux = None
                            addl_ferr = None
                            try:
                                addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                                addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                                addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                            except:
                                pass

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = line_prob.prob_LAE(
                                wl_obs=target_w,
                                lineFlux=target_flux,
                                ew_obs=(target_flux / filter_fl_cgs),
                                c_obs=None, which_color=None,
                                addl_wavelengths=addl_waves,
                                addl_fluxes=addl_flux, addl_errors=addl_ferr,sky_area=None,
                                cosmo=None, lae_priors=None,
                                ew_case=None, W_0=None,
                                z_OII=None, sigma=None)

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
                    else:
                        text += "N/A\nN/A\n"

                    text = text + "%0.2f(%0.2f,%0.2f)%s\n" % (filter_mag, filter_mag_bright,filter_mag_faint, filter_str)

                    if (not G.ZOO):
                        if (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                            text += "%0.3g\n" % (bid_target.p_lae_oii_ratio)
                        else:
                            text += "\n"

                else:
                    text = "%s\n%f\n%f\n" % ("--", r, d)

                plt.subplot(gs[0, col_idx])
                plt.gca().set_frame_on(False)
                plt.gca().axis('off')
                plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font, color=bid_colors[col_idx - 1])

                # add photo_z plot
                # if the z_best_type is 'p' call it photo-z, if s call it 'spec-z'
                # alwasy read in file for "file" and plot column 1 (z as x) vs column 9 (pseudo-probability)
                # get 'file'
                # z_best  # 6 z_best_type # 7 z_spec # 8 z_spec_ref
                # overplot photo z lines

                if df_photoz is not None:
                    z_cat = self.read_catalog(op.join(self.SupportFilesLocation, photoz_file), "z_cat")
                    if z_cat is not None:
                        x = z_cat['z'].values
                        y = z_cat['mFDa4'].values
                        plt.subplot(gs[0, 4:])
                        plt.plot(x, y, color=bid_colors[col_idx - 1])
                        plt.xlim([0, 3.6])
                        # trim axis to 0 to 3.6

                        if spec_z > 0:
                            # plt.axvline(x=spec_z, color='gold', linestyle='solid', linewidth=3, zorder=0)
                            plt.scatter([spec_z, ], [plt.gca().get_ylim()[1] * 0.9, ], zorder=9,
                                        marker="o", s=80, facecolors='none', edgecolors=bid_colors[col_idx - 1])

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
                        # plt.xlabel("z")

                        #  if len(legend) > 0:
                        #      plt.gca().legend(handles=legend, loc = 'lower center', ncol=len(legend), frameon=False,
                        #                      fontsize='small', borderaxespad=0,bbox_to_anchor=(0.5,-0.25))


                        # fig holds the entire page
            plt.close()
            return fig

#######################################
#end class STACK_COSMOS
#######################################