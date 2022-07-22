from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package


try:
    from elixer import global_config as G
    from elixer import science_image
    from elixer import cat_bayesian
    from elixer import observation as elixer_observation
    from elixer import utilities
    from elixer import spectrum_utilities as SU
    from elixer import match_summary
    from elixer import line_prob
except:
    import global_config as G
    import science_image
    import cat_bayesian
    import observation as elixer_observation
    import utilities
    import spectrum_utilities as SU
    import match_summary
    import line_prob

import os.path as op
import copy
from astropy.coordinates import SkyCoord

import matplotlib
#matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from matplotlib.font_manager import FontProperties
import scipy.constants
import io


#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.LOG_LEVEL)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field



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
    # if multiple images, the composite broadest range (filled in by hand)
    Image_Coord_Range = {'RA_min':None,'RA_max':None,'Dec_min':None, 'Dec_max':None}
    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    df = None  # pandas dataframe ... all instances share the same frame
    df_photoz = None
    status = -1
    mean_FWHM = 1.5 #sort of generic seeing ... each implemented catalog should overwrite this
    MAG_LIMIT = 99.9

    def __init__(self):
        self.pages = None #list of bid entries (rows in the pdf)
        self.dataframe_of_bid_targets = None
        self.master_cutout = None
        self.distance_prior = cat_bayesian.DistancePrior()

        #blue, red, green, white
        self.colormap = [[0, 0, 1,1], [1, 0, 0,1], [0, .85, 0,1], [1, 1, 1,0.7]]

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
        result = False
        error = error / 3600.00 #passed in as arcsec but coords below are in degrees

        if (cls.Cat_Coord_Range['RA_min'] is None) and (cls.Image_Coord_Range['RA_min'] is None) and (cls.df is None):
            cls.read_main_catalog()

        #use the imaging first (it is most important)
        #some catalogs have entries outside of the imageing field and, now, that is not useful
        try:
            if (cls.Image_Coord_Range['RA_min'] is not None):
                result = (ra >= (cls.Image_Coord_Range['RA_min'] - error)) and \
                         (ra <= (cls.Image_Coord_Range['RA_max'] + error)) and \
                         (dec >= (cls.Image_Coord_Range['Dec_min'] - error)) and \
                         (dec <= (cls.Image_Coord_Range['Dec_max'] + error))

               # if result: #yes it is in range, lets see if there is actually a non-empty cutout for it??
               #     cutout,counts, ... this is a complication. Needs to know which image(s) to load

            elif (cls.Cat_Coord_Range['RA_min'] is not None):
                result = (ra >=  (cls.Cat_Coord_Range['RA_min'] - error)) and \
                         (ra <=  (cls.Cat_Coord_Range['RA_max'] + error)) and \
                         (dec >= (cls.Cat_Coord_Range['Dec_min'] - error)) and \
                         (dec <= (cls.Cat_Coord_Range['Dec_max'] + error))
        except:
            result = False

        # try:
        #     #either in the catalog range OR in the Image range
        #     if (cls.Cat_Coord_Range['RA_min'] is not None):
        #         result = (ra >=  (cls.Cat_Coord_Range['RA_min'] - error)) and \
        #                  (ra <=  (cls.Cat_Coord_Range['RA_max'] + error)) and \
        #                  (dec >= (cls.Cat_Coord_Range['Dec_min'] - error)) and \
        #                  (dec <= (cls.Cat_Coord_Range['Dec_max'] + error))
        # except:
        #     result = False
        #
        # try:
        #     if (not result) and (cls.Image_Coord_Range['RA_min'] is not None):
        #         result = (ra >=  (cls.Image_Coord_Range['RA_min'] - error)) and \
        #                  (ra <=  (cls.Image_Coord_Range['RA_max'] + error)) and \
        #                  (dec >= (cls.Image_Coord_Range['Dec_min'] - error)) and \
        #                  (dec <= (cls.Image_Coord_Range['Dec_max'] + error))
        # except:
        #     pass #keep the result as is

        return result

    @classmethod
    def read_main_catalog(cls):
        if cls.df is not None:
            log.debug("Already built df")
        elif cls.MainCatalog is not None:
            try:
                print("Reading main catalog for ", cls.Name)
                cls.df = cls.read_catalog(cls.MainCatalog, cls.Name)
                cls.status = 0

                #if True:
                #    print(cls.df.columns.values)

                #with the extended catalogs, this is no-longer appropriate
                if False:
                    #also check vs. by hand
                    ra_min = cls.Cat_Coord_Range['RA_min']
                    ra_max = cls.Cat_Coord_Range['RA_max']
                    dec_min = cls.Cat_Coord_Range['Dec_min']
                    dec_max = cls.Cat_Coord_Range['Dec_max']

                    cls.Cat_Coord_Range['RA_min'] = cls.df['RA'].min()
                    cls.Cat_Coord_Range['RA_max'] = cls.df['RA'].max()
                    cls.Cat_Coord_Range['Dec_min'] = cls.df['DEC'].min()
                    cls.Cat_Coord_Range['Dec_max']= cls.df['DEC'].max()

                    if ra_min is not None:
                        if  (abs(ra_min - cls.Cat_Coord_Range['RA_min']) > 1e-6 ) or \
                            (abs(ra_max - cls.Cat_Coord_Range['RA_max']) > 1e-6 ) or \
                            (abs(dec_min - cls.Cat_Coord_Range['Dec_min']) > 1e-6 )or \
                            (abs(dec_max - cls.Cat_Coord_Range['Dec_max']) > 1e-6 ):
                            print("Warning! Pre-defined catalog coordinate ranges may have changed. Please check class "
                                  "definitions for %s" %(cls.Name))
                            log.info("Warning! Pre-defined catalog coordinate ranges may have changed. Please check class "
                                  "definitions for %s.\nPre-defined ranges: RA [%f - %f], Dec [%f -%f]\n"
                                     "Runtime ranges: RA [%f - %f], Dec [%f -%f]"
                                     %(cls.Name,ra_min,ra_max,dec_min,dec_max,cls.Cat_Coord_Range['RA_min'],
                                       cls.Cat_Coord_Range['RA_max'],cls.Cat_Coord_Range['Dec_min'],
                                       cls.Cat_Coord_Range['Dec_max']))

                    log.debug(cls.Name + " Coordinate Range: RA: %f to %f , Dec: %f to %f"
                              % (cls.Cat_Coord_Range['RA_min'], cls.Cat_Coord_Range['RA_max'],
                                 cls.Cat_Coord_Range['Dec_min'], cls.Cat_Coord_Range['Dec_max']))
            except:
                print("Failed")
                cls.status = -1
                log.error("Exception in read_main_catalog for %s" % cls.Name, exc_info=True)

        else:
            print("No Main Catalog defined for ", cls.Name)

        if cls.df is None:
            cls.status = -1
        return



    # def sort_bid_targets_by_aperture(self, ra, dec, xc, yc, a, b, angle ):
    # PROBLEM: as currently organized, sorting by "likelihood" occurs well before SEP is run and is static
    # (and needs to be static) for all filters and only some filters will get apertures (and the apertures from SEP
    # can change filter to filter) so ... say, u-band may have no apertures and has the sorted bid-targets plotted,
    # then g-band gets apertures and would re-sort, but this throws off what has already gone before ...
    # This has to be abandoned UNTIL/UNLESS a significant re-organization is done
    #     """
    #     Similar to sort_bid_targets_by_likelihood .... this resorts to select the single best match
    #     based on the bid-target being within the selected imaging aperture
    #     """
    #
    #
    #     try:
    #         if hasattr(self, 'dataframe_of_bid_targets_unique'):
    #             pass
    #
    #         # YES, both need to have this performed (this one always) as they are used for different purposes later
    #         if hasattr(self, 'dataframe_of_bid_targets') and (
    #                 self.dataframe_of_bid_targets is not None):  # sanity check ... all cats have this
    #             pass
    #     except:
    #         log.warning("Exception in cat_base::Catalog::sort_bid_targets_by_aperture()", exc_info=True)

    def get_mag_limit(self,image_identification=None,aperture_diameter=None):
        """
        to be overwritten by subclasses to return their particular format of maglimit

        :param image_identification: some way (sub-class specific) to identify which image
        :param aperture_diameter: in arcsec
        :return:
        """

        try:
            return self.MAG_LIMIT
        except:
            return 99.9

    def sort_bid_targets_by_likelihood(self,ra,dec):
        #right now, just by euclidean distance (ra,dec are of target) (don't forget to adjust RA coord difference
        #for the declination ... just use the target dec, the difference to the bid dec is negligable)
        #if radial separation is greater than the error, remove this bid target?
        #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)

        #if _unique exists, the child catalog could have duplicates (stitched together from other filters, etc)
        #   so, use the _unique version to sort by distance, otherwise the same object will have multiple entries
        #   and the sort is messed up

        try:
            if hasattr(self,'dataframe_of_bid_targets_unique'):
                self.dataframe_of_bid_targets_unique['dist_prior'] = 1.0
                df = self.dataframe_of_bid_targets_unique
                df['distance'] = np.sqrt(
                    (np.cos(np.deg2rad(dec)) * (df['RA'] - ra)) ** 2 + (df['DEC'] - dec) ** 2)

                df['dist_prior'] = 1.0
                df['catalog_mag'] = 99.9
                df['catalog_filter'] = '-'
                df['catalog_flux'] = -1.0
                df['catalog_flux_err'] = -1.0

                pidx = df.columns.get_loc('dist_prior')
                midx = df.columns.get_loc('catalog_mag')
                fidx = df.columns.get_loc('catalog_filter')
                fl_idx = df.columns.get_loc('catalog_flux')
                fle_idx = df.columns.get_loc('catalog_flux_err')

                for i in range(len(df)):
                    filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = \
                        self.get_filter_flux(df.iloc[[i]]) #note the .iloc[[i]] so we get a dataframe not a series
                    if filter_mag is not None:
                        p = self.distance_prior.get_prior(df.iloc[i]['distance'] * 3600.0, filter_mag)
                        df.iat[i, pidx] = p
                        df.iat[i, midx] = filter_mag
                        df.iat[i, fidx] = filter_str
                        df.iat[i, fl_idx] = filter_fl
                        df.iat[i, fle_idx] = filter_fl_err

                self.dataframe_of_bid_targets_unique = df.sort_values(by=['dist_prior','distance'], ascending=[False,True])


            #YES, both need to have this performed (this one always) as they are used for different purposes later
            if hasattr(self,'dataframe_of_bid_targets') and (self.dataframe_of_bid_targets is not None): #sanity check ... all cats have this
                df = self.dataframe_of_bid_targets
                df['distance'] = np.sqrt(
                    (np.cos(np.deg2rad(dec)) * (df['RA'] - ra)) ** 2 + (df['DEC'] - dec) ** 2)

                df['dist_prior'] = 1.0
                df['catalog_mag'] = 99.9
                df['catalog_filter'] = '-'
                df['catalog_flux'] = -1.0
                df['catalog_flux_err'] = -1
                pidx = df.columns.get_loc('dist_prior')
                midx = df.columns.get_loc('catalog_mag')
                fidx = df.columns.get_loc('catalog_filter')
                fl_idx = df.columns.get_loc('catalog_flux')
                fle_idx = df.columns.get_loc('catalog_flux_err')
                #note: if _unique exists, this sort here is redudant, otherwise it is needed
                for i in range(len(df)):
                    filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = \
                        self.get_filter_flux(df.iloc[[i]]) #note the .iloc[[i]] so we get a dataframe not a series
                    if filter_mag is not None:
                        p = self.distance_prior.get_prior(df.iloc[i]['distance']*3600.0,filter_mag)
                        df.iat[i,pidx] = p
                        df.iat[i, midx] = float(99 if np.isnan(filter_mag) else filter_mag)
                        df.iat[i, fidx] = filter_str
                        df.iat[i, fl_idx] = float(0 if np.isnan(filter_fl) else filter_fl)
                        try:
                            df.iat[i, fle_idx] =float(0 if np.isnan(filter_fl_err) else filter_fl_err) #pandas bug on insert
                        except:
                            df.iat[i, fle_idx] = 0.0
                        #have to explicitly cast away from np.float32 to just a float


                self.dataframe_of_bid_targets = df.sort_values(by=['dist_prior','distance'], ascending=[False,True])
        except: #could be it exists but is an empty list
            _log = False
            try:
                if hasattr(self,'dataframe_of_bid_targets_unique') and \
                        (self.dataframe_of_bid_targets_unique is not None) and \
                        (len(self.dataframe_of_bid_targets_unique)==0):
                    self.dataframe_of_bid_targets_unique = None
            except:
                _log = True

            try:
                if hasattr(self,'dataframe_of_bid_targets') and \
                        (self.dataframe_of_bid_targets is not None) and \
                        (len(self.dataframe_of_bid_targets)==0):
                    self.dataframe_of_bid_targets = None
            except:
                _log = True
            if _log:
                log.warning("Exception in cat_base::Catalog::sort_bid_targets_by_likelihood()",exc_info=True)


    def clear_pages(self):
        if self.pages is None:
            self.pages = []
        elif len(self.pages) > 0:
            del self.pages[:]

        if self.master_cutout is not None:
            del (self.master_cutout)
            self.master_cutout = None

    def add_bid_entry(self, entry):
        if self.pages is None:
            self.clear_pages()
        self.pages.append(entry)

    def add_north_box(self,plt,sci,cutout,half_side,zero_x = 0,zero_y = 0,theta = None,box=True):
        #theta is angle in radians counter-clockwise from x-axis to the north celestrial pole

        if (plt is None) or (sci is None) or (cutout is None) or (half_side is None):
            return

        try:
            if theta is None:
                theta = sci.get_rotation_to_celestrial_north(cutout)


            rx, ry, rrot = sci.get_rect_parms(cutout, -half_side, -half_side, theta - np.pi / 2.)

            if box:
                plt.gca().add_patch(plt.Rectangle((zero_x + rx, zero_y + ry), width=half_side * 2, height=half_side * 2,
                                                  angle=rrot, color='red', fill=False))

                t_dx = half_side * np.cos(theta)  # * sci.pixel_size
                t_dy = half_side * np.sin(theta)  # * sci.pixel_size
                plt.text(zero_x + t_dx * 1.3, zero_y + t_dy * 1.3, 'N', rotation=rrot,
                         fontsize=10, color="red", verticalalignment='center', horizontalalignment='center')

                t_dx = half_side * np.cos(theta + np.pi / 2.)  # * sci.pixel_size
                t_dy = half_side * np.sin(theta + np.pi / 2.)  # * sci.pixel_size
                plt.text(zero_x + t_dx * 1.3, zero_y + t_dy * 1.3, 'E', rotation=rrot,
                         fontsize=10, color="red", verticalalignment='center', horizontalalignment='center')
            else:
                t_dx = half_side * np.cos(theta)  # * sci.pixel_size
                t_dy = half_side * np.sin(theta)  # * sci.pixel_size
                plt.text(zero_x + t_dx * 0.95, zero_y + t_dy * 0.95, 'N', rotation=rrot,
                         fontsize=10, color="red", verticalalignment='center', horizontalalignment='center')

                t_dx = half_side * np.cos(theta + np.pi / 2.)  # * sci.pixel_size
                t_dy = half_side * np.sin(theta + np.pi / 2.)  # * sci.pixel_size
                plt.text(zero_x + t_dx * 0.95, zero_y + t_dy * 0.95, 'E', rotation=rrot,
                         fontsize=10, color="red", verticalalignment='center', horizontalalignment='center')

        except:
            log.error("Exception bulding celestrial north box.", exc_info=True)


    def add_north_arrow (self, plt, sci, cutout, theta=None,scale=1.0):
        # theta is angle in radians counter-clockwise from x-axis to the north celestrial pole

        if (plt is None) or (sci is None) or (cutout is None):
            return

        try:
            #arrow_color = [0.2, 1.0, 0.23]
            arrow_color = "red"

            if theta is None:
                theta = sci.get_rotation_to_celestrial_north(cutout)

            t_rot = (theta - np.pi/2.) * 180./np.pi

            arrow_len = 0.03 * (cutout.xmax_cutout + cutout.ymax_cutout)
            arrow_x = cutout.xmax_cutout * 0.4 * sci.pixel_size * scale
            arrow_y = cutout.ymax_cutout * 0.4 * sci.pixel_size * scale
            arrow_dx = arrow_len * np.cos(theta) * sci.pixel_size * scale
            arrow_dy = arrow_len * np.sin(theta) * sci.pixel_size * scale
            plt.gca().add_patch(plt.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, color=arrow_color, linewidth=1.0))
            plt.text(arrow_x + arrow_dx * 1.5, arrow_y + arrow_dy * 1.5, 'N', rotation=t_rot,
                     fontsize=8, color=arrow_color, verticalalignment='center', horizontalalignment='center')

            arrow_dx = arrow_len * np.cos(theta + np.pi / 2.) * sci.pixel_size * scale
            arrow_dy = arrow_len * np.sin(theta + np.pi / 2.) * sci.pixel_size * scale
            plt.gca().add_patch(plt.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, color=arrow_color, linewidth=1.0))
            plt.text(arrow_x + arrow_dx * 1.5, arrow_y + arrow_dy * 1.5, 'E',rotation=t_rot,
                     fontsize=8, color=arrow_color, verticalalignment='center', horizontalalignment='center')
        except:
            log.error("Exception bulding celestrial north arrow.", exc_info=True)

    def get_bid_colors(self,count=1):
        #adjust so colors always in the same sequence regardless of the number
        #ie. 1 = blue, 2 = red, 3 = green, then other colors
        #norm = plt.Normalize()
        #map = plt.cm.tab10(norm(np.arange(10)))
        # gist_rainbow or brg or hsv

        #any extras get the last color

        if count > len(self.colormap):
            map = self.colormap[:]
            elem = self.colormap[-1]
            map = map + [elem]*(count-len(self.colormap))
        else:
            map = self.colormap[:count]

        return map

    #caller might send in flux and.or wavelength as strings, so protect there
    #also, might not have valid flux

    def nano_jansky_to_mag(self,flux,err=None):#,wavelength):
        if flux <= 0.0:
            log.info("Cannot use negative flux (cat_base::*_janksy_to_mag()). flux =  %f" %flux)
            return 99.9, 0., 0.

        if err is None:
            return -2.5*np.log10(flux) + 31.4,0,0
        else:
            cn = self.nano_jansky_to_cgs(flux)[0]
            mx = self.nano_jansky_to_cgs(flux+err)[0] - cn #reminder, this will be negative because magnitudes
            mn = self.nano_jansky_to_cgs(flux-err)[0] - cn #reminder, this will be positive because magnitudes
            return cn, mx, mn

    def micro_jansky_to_mag(self,flux,err=None):#,wavelength):
        if flux <= 0.0:
            log.info("Cannot use negative flux (cat_base::*_janksy_to_mag()). flux =  %f" %flux)
            return 99.9, 0., 0.
        if err is None:
            return -2.5*np.log10(flux) + 23.9, 0, 0
        else:
            cn = self.micro_jansky_to_mag(flux)[0]
            mx = self.micro_jansky_to_mag(flux+err)[0] - cn
            mn = self.micro_jansky_to_mag(flux-err)[0] - cn
            return cn, mx, mn

    def micro_jansky_to_cgs(self,flux,wavelength):
        return self.nano_jansky_to_cgs(flux*1000.0,wavelength)

    def nano_jansky_to_cgs(self,flux,wavelength):
        c = scipy.constants.c * 1e10
        try:
            return float(flux) * 1e-32 * c / (float(wavelength) ** 2)  # 3e18 ~ c in angstroms/sec
        except:
            return 0.

    def obs_mag_to_Jy(self,mag):
        # essentially f_v
        try:
            return 3631.0 * 10 ** (-0.4 * mag)
        except:
            return 0.

    def obs_mag_to_micro_Jy(self,mag):
        try:
            return self.obs_mag_to_Jy(mag) * 1e6
        except:
            return 0.

    def obs_mag_to_nano_Jy(self,mag):
        return self.obs_mag_to_Jy(mag) * 1e9

    def obs_mag_to_cgs_flux(self,mag, wavelength):
        # approximate, but good enough?
        # should be the filter iso wavelength or iso frequency, not the line
        # hz = (3e18)/float(wavelength)
        try:
            return self.obs_mag_to_Jy(mag) * 1e-23 * (scipy.constants.c * 1e10) / (wavelength ** 2)
        except:
            return 0.

    def get_f606w_max_cont(self,exp_time,sigma=3,base=None):
        #note:this goes as the sqrt of exp time, but even so, this is not a valid use
        #each field has its own value (and may have several)
        candles_egs_baseline_exp_time = 289618.0
        candles_egs_baseline_cont = 3.3e-21

        try:
            if base is None:
                cont = sigma * candles_egs_baseline_cont * np.sqrt(candles_egs_baseline_exp_time / exp_time)
            else:
                cont = sigma * base
        except:
            log.error("Error in cat_base:get_f606w_max_cont ", exc_info=True)
            cont = -1

        return cont

    def build_annulus_report(self, obs, cutout, section_title=""):

        MAX_LINE_SCORE = 20.0
        MAX_ALPHA = 0.5 #still want to be able to see through it
        #BUILD HERE ... we have everything we need now
        #base this off of build_bid_target_reports, but use the attached cutout and just make the single image
        print("Building ANNULUS  REPORT")

       # if (cutout is None) or (obs is None):
        if obs is None: #must have obs, but cutout could be None
            log.error("Invalid parameters passed to build_annulus_report")
            return None

        self.clear_pages()

        #ra,dec,annulus should already be set as part of the obs (SyntheticObservation)

        if (obs.fibers_work is None) or (len(obs.fibers_work) == 0):
            #make sure the eli_dict is fully populated
            if len(obs.eli_dict) == 0:
                obs.build_complete_emission_line_info_dict()
                if len(obs.eli_dict) == 0: #problem, we're done
                    log.error("Problem building complete EmissionLineInfo dictionary.")
                    return None

            #now sub select the annulus fibers (without signal) (populate obs.fibers_work
            obs.annulus_fibers(empty=True)
            if len(obs.fibers_work) == 0:
                log.warning("Warning. No fibers found inside the annulus.")
        elif obs.best_radius != obs.annulus[1]: #get the full set (maximum included)
            #*** note: this does NOT rebuild the spectrum or the mcmc fit, but does rebuild the list of fibers
            #    if you re-run set_spectrum after this it WILL rebuild the spectrum and fit
            obs.annulus_fibers(empty=True)
            if len(obs.fibers_work) == 0:
                log.warning("Warning. No fibers found inside the annulus.")

        rows = 1
        cols = 1

        fig_sz_x = G.ANNULUS_FIGURE_SZ_X #initial guess
        fig_sz_y = G.ANNULUS_FIGURE_SZ_X #7 #want square

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.axes().set_aspect('equal')
       # plt.subplots_adjust(left=0.1, right=0.90, top=0.90, bottom=0.05)

       # gs = gridspec.GridSpec(rows, cols, wspace=0.0, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        title = "Diffuse Emission"

        #plt.subplot(gs[:, :]) #right now, only the one (if in future want more this needs to change)
        #text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        #plt.gca().set_frame_on(False)
        #plt.gca().axis('off')

        signal_color = 'r'
        empty_color = 'y'
        ext = obs.annulus[1]

        if cutout is not None:
            empty_sci = science_image.science_image()

            vmin, vmax = empty_sci.get_vrange(cutout.data)

            plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                       vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

            self.add_north_box(plt, empty_sci, cutout, obs.annulus[1], 0, 0, theta=None, box=False)

            x, y = empty_sci.get_position(obs.ra, obs.dec, cutout)  # zero (absolute) position
        else:

            # gray background
            rec = plt.Rectangle((-ext, -ext), width=ext * 2., height=ext * 2., fill=True, lw=1,
                                color='gray', zorder=0, alpha=0.3)
            plt.gca().add_patch(rec)

            x = y = 0


        #same with or without cutout
        plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        plt.title(title)
        plt.ylabel('arcsec')
        plt.xlabel('arcsec')


        #put in the fibers ... this is very similar to, but not the same as add_fiber_positions
        try:

            #plot annulus
            #inner radius
            plt.gca().add_patch(plt.Circle((0, 0), radius=obs.annulus[0],
                                           facecolor='none', fill=False, alpha=1,
                                           edgecolor='k', linestyle="dashed"))

            #outer radius
            plt.gca().add_patch(plt.Circle((0, 0), radius=obs.annulus[1],
                                           facecolor='none', fill=False, alpha=1,
                                           edgecolor='k', linestyle="dashed"))

            #best
            if obs.best_radius is not None:
                plt.gca().add_patch(plt.Circle((0, 0), radius=obs.best_radius,
                                               facecolor='none', fill=False, alpha=1,
                                               edgecolor='r', linestyle="solid"))


            #go over ALL fibers for the fill color, but only add edge to work fibers
            for f in obs.fibers_all:
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                if cutout:
                    fx, fy = empty_sci.get_position(f.ra, f.dec, cutout)
                else:
                    fx = (f.ra - obs.ra) * np.cos(np.deg2rad(obs.dec)) * 3600.
                    fy = (f.dec - obs.dec) * 3600.

                #for now, set alpha as a fraction of a max line score? cap at 1.0
                # not eli_dict ... want strongest signal at ANY wavelength, not just the central one


                # if obs.eli_dict[f] is not None:
                #     alpha = obs.eli_dict[f].line_score
                #     if alpha is None:
                #         alpha = 0.0
                #     else:
                #         alpha = min(MAX_ALPHA, (alpha / MAX_LINE_SCORE * MAX_ALPHA))
                # else:
                #     alpha = 0.0

                if (f.peaks is None) or (len(f.peaks) == 0):
                    alpha = 0.0
                else:
                    alpha = max(p.line_score for p in f.peaks)
                    if alpha is None:
                        alpha = 0.0
                    else:
                        alpha = min(MAX_ALPHA, (alpha / MAX_LINE_SCORE * MAX_ALPHA))


                if alpha > 0: #has signal
                    plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius,
                                               facecolor=signal_color, fill=True, alpha=alpha,
                                               edgecolor=signal_color,linestyle="solid",linewidth=1))
                elif not(f in obs.fibers_work): #empty and NOT in working set
                    #maybe there was a problem if this is inside the outer radius
                    #or if outside the radius, this is expected
                    plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius,
                                               facecolor='none', fill=False, alpha=1,
                                               edgecolor='grey',linestyle="solid",linewidth=1))

                if f in obs.fibers_work:
                    # todo:
                    # could be in work AND have signal (depending on options)
                    # don't want to color twice so, if colored just above, don't re-fill here, just add the edge

                    plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius,
                                                   facecolor=empty_color, fill=True, alpha=0.25,
                                                   edgecolor=empty_color, linestyle="solid", linewidth=1))



        except:
            log.error("Unable to overplot gradient (all) fiber positions.", exc_info=True)

        #plt.axes().set_aspect('equal')
        #plt.tight_layout()

        plt.close()
        self.add_bid_entry(fig)

        return self.pages

    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None,target_flux=None,detobj=None):
        #implement in child class
        #called from elixer.py to get the list of bid targets for each catalog as part of a pre-report search
        pass


    def stack_image_cutouts(self,cutouts):
        """
        given a set of cutouts (list of dictionaries)
        stack the filters and return a single FITS cutout image

        Notice: in an excpetion case, there is no imaging and "cutouts" is actually a matplotlib figure
        for the empty data

        :param cutouts:
        :return:
        """

        stacked_cutout = None
        if cutouts is not None and len(cutouts) > 0:
            try:
                total_adjusted_exptime = 1.0
                ref_exptime = 0.0
                for c in cutouts:
                    if c and isinstance(c,dict) and c['cutout']:
                        if stacked_cutout is None:
                            stacked_cutout = copy.deepcopy(c['cutout'])
                            try:
                                ref_exptime = c['details']['exptime']
                                if not ref_exptime:
                                    ref_exptime = 1.0
                            except:
                                ref_exptime = 1.0
                            total_adjusted_exptime = 1.0
                        else:
                            try:
                                #log.debug(f"{np.shape(stacked_cutout.data)}, {np.shape(c['cutout'].data)}, {c['details']['exptime']}, {ref_exptime}")
                                this_exptime = 1.0 if c['details']['exptime'] is None else c['details']['exptime']
                                stacked_cutout.data = np.add(stacked_cutout.data, c['cutout'].data * this_exptime / ref_exptime)
                                total_adjusted_exptime += c['details']['exptime'] / ref_exptime
                            except:
                                pass
                if stacked_cutout and total_adjusted_exptime:
                    stacked_cutout.data /= total_adjusted_exptime
            except:
                log.warning("Exception in cat_base.py stack_image_cutouts",exc_info=True)
        return stacked_cutout


    def revisit_sep_selection(self,detobj):
        """
        Double check the SEP selection on a per aperture list basis. Could be due to astrometric or other error, the
        selected SEP object might not be correct even though it is the closest to the HETDEX position. We want to also
        check the g-mag for compatibility.

        :param detobj:
        :return:
        """

        try:
            if detobj.best_gmag is None or detobj.best_gmag <= 0:
                return #cannot check the gmag

            for d in detobj.aperture_details_list:
                if 'sep_obj_idx' in d.keys() and d['sep_obj_idx'] != None:
                    #there is an sep object selected as best
                    #is the selected object a good mag match?
                    idx = d['sep_obj_idx']
                    if  (abs(detobj.best_gmag - d['sep_objects'][idx]['mag']) < 0.5) or \
                        ( (detobj.best_gmag < 22) and ( d['sep_objects'][idx]['mag'] < 22) ) or \
                        ( ( d['sep_objects'][idx]['mag'] > detobj.best_gmag) and (detobj.best_gmag > G.HETDEX_CONTINUUM_MAG_LIMIT)):
                        #yep, compatible, so keep this one
                        pass
                    else: #not compatible ... is there a better one?
                        target_dist = d['sep_objects'][idx]['dist_curve'] if d['sep_objects'][idx]['dist_curve'] > 0 else d['sep_objects'][idx]['dist_baryctr']
                        target_dist += 0.5 #allow up to 0.5"
                        best_idx = idx
                        best_dist = 999.9
                        best_dmag = 999.9

                        for i,s in enumerate(d['sep_objects']):
                            bid_dist = s['dist_curve'] if s['dist_curve'] > 0 else s['dist_baryctr']
                            bid_dmag = abs(detobj.best_gmag - s['mag'])
                            if bid_dist <= target_dist: #sufficiently close
                                if (bid_dmag < 0.5) or \
                                   ((detobj.best_gmag < 22) and (s['mag'] < 22)) or \
                                   ((s['mag'] > detobj.best_gmag) and (detobj.best_gmag > G.HETDEX_CONTINUUM_MAG_LIMIT)):
                                    #they are compatible
                                    if (bid_dist < best_dist) and (bid_dmag < best_dmag):
                                        #de-select the old one
                                        d['sep_objects'][best_idx]['selected'] = False

                                        #set the new one
                                        best_idx = i
                                        best_dist = bid_dist
                                        best_dmag = bid_dmag
                                        d['sep_objects'][best_idx]['selected'] = True


                        if best_idx != idx:
                            log.info(f"[{detobj.entry_id}] - updated selected SEP object from "
                                     f"{d['sep_obj_idx'] } to {best_idx} for {d['catalog_name']}-{d['filter_name']}")
                            d['sep_obj_idx'] = best_idx

                            #update the other relevant data
                            d['ra'] = d['sep_objects'][best_idx]['ra']
                            d['dec'] = d['sep_objects'][best_idx]['dec']
                            d['radius'] = 0.5*np.sqrt(d['sep_objects'][best_idx]['a']*d['sep_objects'][best_idx]['b'])
                            d['mag'] = d['sep_objects'][best_idx]['mag']
                            d['mag_faint'] = d['sep_objects'][best_idx]['mag_faint']
                            d['mag_bright'] = d['sep_objects'][best_idx]['mag_bright']
                            d['aperture_counts'] = d['sep_objects'][best_idx]['flux_cts']



                            # d['aperture_eqw_rest_lya']
                            # d['aperture_eqw_rest_lya_err']
                            # d['aperture_plae']
                            # d['aperture_plae_min']
                            # d['aperture_plae_max']

        except:
            log.warning("Exception in cat_base.py revisit_sep_selection", exc_info=True)

    def build_cat_summary_pdf_section(self,list_of_cutouts, cat_match, ra, dec, error, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None):
        """
        Scans cutouts to build the optimal catalog summary section, prioritizing the "best" (deepest) survey for
        g and r equivalent filters and filling in other filters with the deepest equivalent

        :param cutouts:
        :param cat_match:
        :param ra:
        :param dec:
        :param error:
        :param target_w:
        :param fiber_locs:
        :param target_flux:
        :param detobj:
        :return: matplotlib image section(s)
        """

        #begin by picking the "best" deepest g or r band as the primary catalog
        #this will be the master images for the fiber positions

        if len(list_of_cutouts) == 0:
            return None

        #self.revisit_sep_selection(detobj)

        #allow master (fiber locations) and up to 5 (or 6) filters?
        the_best_cat_idx = 0
        the_best_cutout_idx = 0
        list_of_counterparts = [] #these are all BidTarget objects

        best_dict = {}
        best_dict['u'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['u',]}
        best_dict['g'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['g','v','f435w']}
        best_dict['r'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['r','f606w']}
        #for HST may have both f775w and f814w, but only show the "best"
        best_dict['i'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['i','f775w','f814w']}
        best_dict['z'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['z',]}
        best_dict['y'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['y','f105w']}
        best_dict['j'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['j','f125w']}
        best_dict['jh'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['f140w',]}
        best_dict['h'] = {'depth': 0,'cat_idx': 0,'cutout_idx': 0,'filters':['h','f160w']}

        #other WFC3 filters, f140w, between j and h
        #u ~ 3543 AA
        #g ~ 4770 AA
        #r ~ 6231 AA
        #i ~ 7625 AA
        #z ~ 9134 AA
        #y ~ 10,200 AA
        #J 12,200 AA
        #H 16,300 AA
        #K 21,900 AA
        #L 34,500 AA
        #M 47,500AA
        #N 105,000AA
        #Q 210,000AA

        all_filter_names = []
        #find best g or r (preferred) depths
        for cat_idx, cutouts in enumerate(list_of_cutouts):
            for cutout_idx, c in enumerate(cutouts):
                try:
                    filter = c['filter'].lower()
                    all_filter_names.append(filter)

                    for k in best_dict.keys(): #in order
                        if filter in best_dict[k]['filters']:
                            if best_dict[k]['depth'] < c['mag_limit'] < 99.9:
                                best_dict[k]['depth'] = c['mag_limit']
                                best_dict[k]['cat_idx']  = cat_idx
                                best_dict[k]['cutout_idx'] = cutout_idx
                            break

                    #else: not one of the filters we care about or beyond what we normally access
                except:
                    pass

        #the top (fiber locations) will be 'r' (preferred) or 'g'
        if best_dict['r']['depth']:
            the_best_cat_idx = best_dict['r']['cat_idx']
            the_best_cutout_idx =  best_dict['r']['cutout_idx']
        elif best_dict['g']['depth']:
            the_best_cat_idx = best_dict['g']['cat_idx']
            the_best_cutout_idx =  best_dict['g']['cutout_idx']
        else: #just use the 0th idex
            the_best_cat_idx = 0

        #for later to fill out the set of filters if there is any room left
        all_filter_names = np.unique(all_filter_names)

        log.info(f"All reported filters (up to 6 shown in report): {all_filter_names}")

        try:
            stacked_cutout = self.stack_image_cutouts(list_of_cutouts[the_best_cat_idx])
        except:
            log.debug("Minor exception", exc_info=True)
            stacked_cutout = None

        if stacked_cutout is None:
            fig = self.build_empty_cat_summary_figure(ra,dec,error,None,None,target_w,fiber_locs)
        else:
            #now turn this into a plot object and start adding North Box and Fibers
            rows = 10
            #note: setting size to 7 from 6 so they will be the right size (the 7th position will not be populated)
            cols = 7 # 1 for the fiber position and up to 6 filters for any one tile (u,g,r,i,z,y)
            fig_sz_x = 18 #cols * 3
            fig_sz_y = 3 #ows * 3

            fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

            gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
            # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

            font = FontProperties()
            font.set_family('monospace')
            font.set_size(12)

            #this is the single line between the 1D spectrum above and the cutouts below
            the_entry = list_of_cutouts[the_best_cat_idx][the_best_cutout_idx]

            if the_entry['details'] and the_entry['details']['catalog_name']:
                name = the_entry['details']['catalog_name']
            elif the_entry['instrument']:
                name = the_entry['instrument']
            else:
                name = "---"

            if the_entry['details'] and the_entry['details']['filter_name']:
                filter = the_entry['details']['filter_name']
            elif the_entry['filter']:
                filter = the_entry['filter']
            else:
                filter = "-"

            #couterparts, though, are only listed in the zeroth position
            # list_of_counterparts  = []
            # try:
            #     if 'counterparts' in the_entry.keys():
            #         list_of_counterparts = the_entry['counterparts']
            #     elif 'counterparts' in list_of_cutouts[the_best_cat_idx][0]:
            #         list_of_counterparts = list_of_cutouts[the_best_cat_idx][0]['counterparts']
            # except:
            #     pass
            #


            #todo: rethink this .... the "the_best_cat_idx" is the deepest imaging, but maybe not the best for the
            # catalog ... catalog may be poorly aligned with selection criteria (example: 2100059570) where CANDLES detections
            #  are just the faint ones and not the big galaxy right in the center that we have selected
            # MAYBE!!! merge ALL the counter_parts, toss out duplicates (RA,Dec separation < 0.1" ?
            # NOT all [0] have ['counterparts']

            try:
                all_counterparts = np.concatenate([x[0]['counterparts'] for x in list_of_cutouts if 'counterparts' in x[0].keys()])
            except:
                all_counterparts = []

            for cp in all_counterparts:
                add = True
                for i,uc in enumerate(list_of_counterparts):
                    if abs(cp.distance - uc.distance) < 0.1: #less than 0.1" from the center of image
                        if utilities.angular_distance(cp.bid_ra,cp.bid_dec,uc.bid_ra,uc.bid_dec) < 0.1: #duplicate
                            choice = None
                            if (cp.spec_z is None) and (uc.spec_z is None):
                                if (cp.phot_z is None) and (uc.phot_z is None):
                                    pass #leave as None selected
                                else:
                                    if cp.phot_z is None:
                                        choice = uc
                                    elif uc.phot_z is None:
                                        choice = cp
                                    else: #both have spec_z
                                        pass #leave the choice at none
                            else: # at least one as spec_z
                                if cp.spec_z is None:
                                    choice = uc
                                elif uc.spec_z is None:
                                    choice = cp
                                else: #both have spec_z
                                    pass #leave the choice at none

                            if choice is None: #neither has spec_z or phot_z
                                if cp.bid_flux_est_cgs_unc < uc.bid_flux_est_cgs_unc:
                                    choice = cp
                                else:
                                    choice = uc

                            if choice is cp: #otherwise, keep what is there
                                #need to swap
                                list_of_counterparts[i] = cp
                                add = False

                if add:
                    list_of_counterparts.append(cp)

            #now resort by likelihood of match (new method)
            list_of_counterparts.sort(key=lambda x: x.distance, reverse=False)
            selected_idx = None
            alternate_idx = None
            try:
                for idx, cp in enumerate(list_of_counterparts):
                    #walk in order of distance and check
                    #if inside the ellipse for the selected SEP (if there is one)
                    #is counterpart mag consistent with HETDEX SDSS-g (similar or fainter if HETDEX g > 24)
                    if cp.bid_filter is not None:
                        cp_filter = cp.bid_filter.lower()
                    else:
                        cp_filter = 'x'
                    selected_sep = None
                    #find the selected SEP
                    for d in detobj.aperture_details_list:
                        if (d['filter_name'].lower() != cp_filter) or (d['sep_objects'] is None):
                            continue
                        try:
                            selected_sep = d['sep_objects'][np.where([x['selected'] for x in d['sep_objects']])[0][0]]

                            if  (abs(detobj.best_gmag - selected_sep['mag']) < 0.5) or \
                                ((detobj.best_gmag < 22) and (selected_sep['mag'] < 22)) or \
                                ((selected_sep['mag'] > detobj.best_gmag) and (detobj.best_gmag > G.HETDEX_CONTINUUM_MAG_LIMIT)):
                                pass
                            else:
                                selected_sep = None #the mag is not consistent with HETDEX

                        except:
                            pass #usually just means that no SEP object is a selected object

                    #This is checking the catalog object vs the selected SEP object BUT the SEP object might not match
                    #the HETDEX position?? This can be a problem if there is an astrometric offset between HETDEX and the
                    #underlying imaging and the catalog which can cause the selection of the wrong SEP object
                    if selected_sep: #there is a selected SEP, but it might not correspond to THIS catalog object
                        #are the magnitudes similar and is it inside the ellipse?
                        if utilities.is_in_ellipse(cp.bid_ra,cp.bid_dec,selected_sep['ra'],selected_sep['dec'],
                                                selected_sep['a'],selected_sep['b'],selected_sep['theta'])[0]:
                            #are the mags similar (or SEP is fainter than limit?)
                            if (abs(selected_sep['mag'] - cp.bid_mag) < 0.5) or \
                                ( (selected_sep['mag'] < 22) and (cp.bid_mag < 22) ) or \
                                ( (cp.bid_mag > selected_sep['mag']) and (d['fail_mag_limit'])):
                                    selected_idx = idx
                                    break #this is the one .. the selected SEP matches the catalog object, but they might not match HETDEX g
                            # elif  (abs(detobj.best_gmag - cp.bid_mag) < 0.5) or \
                            #     ( (detobj.best_gmag < 22) and (cp.bid_mag < 22) ) or \
                            #     ( (cp.bid_mag > detobj.best_gmag) and (detobj.best_gmag > G.HETDEX_CONTINUUM_MAG_LIMIT)):
                            #         #did not match the SEP selected object but DOES match with HETDEX g
                            #
                            else:
                                if alternate_idx is None:
                                    alternate_idx = idx
                                elif list_of_counterparts[alternate_idx].distance > list_of_counterparts[idx].distance:
                                    alternate_idx = idx
                    else:
                        #Really ... if no selected SEP, then these must all be faint and we just want the closest one
                        #IF its mag is roughly compatible with DEX-g (or fainter if at the limit) and not far away
                        #note that alternate_idx does NOT come into play here
                        try:
                            if detobj.best_gmag < G.HETDEX_CONTINUUM_MAG_LIMIT:
                                if (abs(detobj.best_gmag - cp.bid_mag) < 0.5) or \
                                    ( (detobj.best_gmag < 22) and (cp.bid_mag < 22) ) or \
                                    ( (cp.bid_mag > detobj.best_gmag) and (cp.distance < 0.75)): #allow a little slop for Ra, Dec differences
                                       selected_idx = idx
                            else: #really faint
                                if (cp.distance < 1.0):
                                    if (abs(detobj.best_gmag - cp.bid_mag) < 0.5) or (cp.bid_mag > detobj.best_gmag) :
                                        selected_idx = idx
                                else:
                                    selected_idx = None
                        except:
                            selected_idx = None
                        break
            except:
                log.warning("Exception attempting to identify best counterpart match in cat_base::build_cat_summary_pdf_section()",
                            exc_info=True)

            if selected_idx is not None:
                if selected_idx != 0:
                    list_of_counterparts.insert(0,list_of_counterparts.pop(selected_idx))
                    selected_idx = 0 #since we moved it to the front of the list

                if detobj is not None:
                    detobj.best_counterpart = list_of_counterparts[0]

                list_of_counterparts[0].selected = True #make the top of the list THE selected catalog match
            elif alternate_idx is not None:
                if alternate_idx != 0:
                    list_of_counterparts.insert(0,list_of_counterparts.pop(alternate_idx))
                    alternate_idx = 0 #since we moved it to the front of the list

                if detobj is not None:
                    detobj.best_counterpart = list_of_counterparts[0]

                list_of_counterparts[0].selected = True #make the top of the list THE selected catalog match


            #old method of just selecting the list of counterparts from the single deepest catalog
            #but this is problematic as that deepest catalog may not be selected in a compatibile way and
            #our galaxy might not be in it
            #counterparts are always just on the [0]th entry for the catalog
            try:
                if len(list_of_counterparts) == 0: #something went wrong
                    counterpart_cat_idx = the_best_cat_idx
                    if 'counterparts' in list_of_cutouts[counterpart_cat_idx][0].keys():# and list_of_cutouts[counterpart_cat_idx][0]['counterparts']:
                        list_of_counterparts = list_of_cutouts[counterpart_cat_idx][0]['counterparts']
                    else:
                        counterpart_cat_idx = 0
                        best_len = 0
                        for idx,cat in enumerate(list_of_cutouts):
                            if 'counterparts' in cat[0].keys():
                                if len(cat[0]['counterparts']) > best_len:
                                    counterpart_cat_idx = idx
                                    best_len = len(cat[0]['counterparts'])

                        if 'counterparts' in list_of_cutouts[counterpart_cat_idx][0].keys():
                            list_of_counterparts = list_of_cutouts[counterpart_cat_idx][0]['counterparts']
                        else:
                            list_of_counterparts = []
            except:
                log.debug("Minor exception",exc_info=True)

            try:
                possible_matches = len(list_of_counterparts)
            except:
                possible_matches = '--'

            title = f"{name} : Possible Matches = {possible_matches} (within +/- {error:g}\")  P(LAE)/P(OII): "
            try:
                title +=  r'$%.4g\ ^{%.4g}_{%.4g}$' % (round(the_entry['aperture_plae'], 3),
                                                   round(the_entry['aperture_plae_max'], 3),
                                                   round(the_entry['aperture_plae_min'], 3))

                title += f" ({filter})"
            except:
                title += "N/A"


            #todo: HERE - resort list_of_counterparts to find the most likely match(es)
            # based on nearerst to select SEP barycenter with similar mag
            # or nearest to DEX center (again, with similar mag to aperture)
            #todo: ALSO the place to choose different list_of_counterparts?
            #  catalog may be poorly aligned with selection criteria (example: 2100059570) where CANDLES detections
            #  are just the faint ones and not the big galaxy right in the center that we have selected

            # list_of_counterparts are the catalog matches
            # detobj.aperture_details_list has imaging info

            bid_ras = [x.bid_ra for x in list_of_counterparts]
            bid_decs = [x.bid_dec for x in list_of_counterparts]
            bid_colors = self.get_bid_colors(len(bid_ras))

            target_box_side = error/4.0

            plt.subplot(gs[0, :])
            text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')

            sci = science_image.science_image() #empty science image to use for functions


            #
            # Add the left-most (stacked) image with the fiber positions
            #
            index = 0 #images go in positions 1+ (0 is for the fiber positions stacked cutout)

            _ = plt.subplot(gs[1:, index])
            pix_size = sci.calc_pixel_size(stacked_cutout.wcs)
            ext = stacked_cutout.shape[0] * pix_size / 2.
            #add_fiber_positions also takes care of the north box and the center
            self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, stacked_cutout)




            #
            # Add the line image if we can get it
            # subcont=True, convolve_image=False,
            #pixscale=0.25, imsize=9.0, wave_range=None, return_coords=False):

            try:
                line_image = science_image.get_line_image(plt,friendid=None,detectid=None,
                                                      coords=SkyCoord(ra=ra,dec=dec,frame='icrs',unit='deg'),
                                                      shotid=detobj.survey_shotid, subcont=True, convolve_image=False,
                                                      pixscale=0.25, imsize=3*error,
                                                      wave_range=[detobj.w - 3.0/2.355*detobj.fwhm, detobj.w + 3.0/2.355*detobj.fwhm],
                                                      sigma=detobj.fwhm/2.355,
                                                      return_coords=False)

                if line_image is not None:
                    index += 1

                    im_ax = plt.subplot(gs[1:, index])#,projection=stacked_cutout.wcs)
                    pix_size =0.25 #make sure to match vs pixscale in above call
                    ext = line_image.shape[0] * pix_size / 2.
                    #without fibers

                    im = plt.imshow(line_image.data, origin='lower', interpolation='none', extent=[-ext, ext, -ext, ext],
                                    vmin=line_image.vmin,vmax=line_image.vmax)#,
                                    #transform=im_ax.get_transform(line_image.wcs))#,cmap=plt.get_cmap('gray_r'))

                    #trying to get the color bar to occupy the axis label space does not seem to work
                    #and the bar and labels just don't seem important here anyway
                    #_ = plt.colorbar(im, orientation="horizontal",fraction=0.07)#,anchor=(0.3,0.0))
                    self.add_north_box(plt, sci, line_image, error, 0, 0, theta=None)#np.pi/2.0)

                    #self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, line_image,use_gray_cmap=False)
                    #add_fiber_positions also takes care of the north box and the center
                    plt.title(f"Lineflux Map")
                    plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    self.add_zero_position(plt)

                    try:
                        plt.xlabel(f"s/b: {line_image.flux/line_image.bkg_stddev:0.2f} +/- {line_image.flux_err/line_image.bkg_stddev:0.3f}")
                               #f"\n{line_image.bkg_stddev:0.2f}, {line_image.apcor:0.2f}")
                    except: #these might be None
                        plt.xlabel(f"sn: undef")


                    plt.gca().xaxis.labelpad = 0
                    plt.subplots_adjust(bottom=0.1)
                else:
                    log.info("Unable to build emission line image postage stamp.")


            except:
                log.warning("Exception building line image",exc_info=True)


            #
            #Now add the other images (in filter order) until we run out of spaces
            #
            for k in best_dict.keys():
                if best_dict[k]['depth']: # a best depth is set
                    index += 1
                    if index >= cols:
                        log.info("Reached maximum number of filters to display. Skipping remainder.")
                        break

                    _ = plt.subplot(gs[1:, index])
                    the_entry = list_of_cutouts[best_dict[k]['cat_idx']][best_dict[k]['cutout_idx']]

                    vmin, vmax = sci.get_vrange(the_entry['cutout'].data)
                    pix_size = sci.calc_pixel_size(the_entry['cutout'].wcs)
                    ext = the_entry['cutout'].shape[0] * pix_size / 2.

                    plt.imshow(the_entry['cutout'].data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

                    if the_entry['instrument']:
                        name = the_entry['instrument'][0:10]
                    elif the_entry['details'] and the_entry['details']['catalog_name']:
                        name = the_entry['details']['catalog_name'][0:10]
                    else:
                        name = "---"

                    if the_entry['mag_limit'] and the_entry['mag_limit'] < 99.9:
                        name += f"({the_entry['mag_limit']:0.1f})"

                    if the_entry['filter']:
                        filter = the_entry['filter']
                    elif the_entry['details'] and the_entry['details']['filter_name']:
                        filter = the_entry['details']['filter_name']
                    else:
                        filter = "-"

                    plt.title(f"{name} {filter}")
                    plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                    plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

                    self.add_zero_position(plt)

                    #build up the needed parameters
                    #if there is an aperture (from source extractor ellipse or elixer circular aperture(s)), draw them
                    if the_entry['ap_center']:
                        cx = the_entry['ap_center'][0]
                        cy = the_entry['ap_center'][1]

                        if the_entry['details']:
                            cutout_ewr = the_entry['details']['aperture_eqw_rest_lya']
                            cutout_plae = the_entry['details']['aperture_plae']

                            if ((the_entry['details']['sep_objects'] is not None) and \
                                    (the_entry['details']['sep_obj_idx'] is not None)):  # and (details['sep_obj_idx'] is not None):
                                self.add_elliptical_aperture_positions(plt, the_entry['details']['sep_objects'],
                                                                       the_entry['details']['sep_obj_idx'],
                                                                       the_entry['details']['radius'],
                                                                       the_entry['details']['mag'],
                                                                       cx, cy, cutout_ewr, cutout_plae)

                                if detobj is not None:
                                    try:
                                        #using the fixed aperture (usually either 2" diam or 3" diam)
                                        #on the seleccted sep obj
                                        sep_idx = the_entry['details']['sep_obj_idx']
                                        sep_obj = the_entry['details']['sep_objects'][sep_idx]

                                        detobj.set_best_filter_mag(band = filter,
                                                                   mag = sep_obj['fixed_aper_mag'],
                                                                   mag_bright = sep_obj['fixed_aper_mag_bright'],
                                                                   mag_faint = sep_obj['fixed_aper_mag_faint'],
                                                                   ra= sep_obj['ra'],
                                                                   dec = sep_obj['dec'],
                                                                   catalog=the_entry['details']['catalog_name']
                                                                   )
                                    except:
                                        log.info(f"Unable to set best filter mag for fixed filter")


                                    # detobj.set_best_filter_mag(filter,the_entry['details']['mag'],
                                    #                            the_entry['details']['mag_bright'],the_entry['details']['mag_faint'])
                            else:

                                try:
                                    distance_to_center = the_entry['details']['elixer_apertures'][the_entry['details']['elixer_aper_idx']]['dist_to_center']
                                except:
                                    distance_to_center = None

                                self.add_aperture_position(plt, the_entry['details']['radius'],
                                                           the_entry['details']['mag'],
                                                           cx, cy, cutout_ewr, cutout_plae,distance_to_center)

                                if the_entry['details']['sep_objects'] is not None:
                                    #still add SEP apertures even though none were selected
                                    self.add_elliptical_aperture_positions(plt, the_entry['details']['sep_objects'],
                                                                           -1,
                                                                           -1,
                                                                           -1,
                                                                           cx, cy, cutout_ewr, cutout_plae)

                                #no sep detection so no reliable color
                                # if detobj is not None:
                                #     detobj.set_best_filter_mag(filter,the_entry['details']['mag'],
                                #                        the_entry['details']['mag_bright'],the_entry['details']['mag_faint'])
                        else:
                            log.warning("No cutout details ...")

                    self.add_north_box(plt, sci, the_entry['cutout'], error, 0, 0, theta=None)

                    x, y = sci.get_position(ra, dec, the_entry['cutout'])  # zero (absolute) position
                    for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                        fx, fy = sci.get_position(br, bd, the_entry['cutout'])

                        self.add_catalog_position(plt,
                                                  x=(fx-x)-target_box_side / 2.0,
                                                  y=(fy-y)-target_box_side / 2.0,
                                                  size=target_box_side, color=bc)

            # complete the entry
            plt.close()

        self.clear_pages()
        self.add_bid_entry(fig)

        # get zoo style cutout as png
        if G.ZOO_MINI and (detobj is not None) and (stacked_cutout is not None):
            plt.figure()

            try:
                if sci is None:
                    sci = science_image.science_image()
            except:
                sci = science_image.science_image()

            pix_size = sci.calc_pixel_size(stacked_cutout.wcs)
            ext = stacked_cutout.shape[0] * pix_size / 2.
            #add_fiber_positions also takes care of the north box and the center
            self.add_fiber_positions(plt, ra, dec, fiber_locs, error, ext, stacked_cutout, unlabeled=True)

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


        #################################################
        #now build up the couterparts from the catalog
        #################################################

        #want to use the catalog associated with the_best_cat_idx, if there is one.
        #If there is not a catalog, then use the r or g band catalog with the most hits?

        if not list_of_counterparts:
            counterpart_cat_idx = the_best_cat_idx
            list_of_counterparts = [] #these are all BidTarget objects
            #counterparts are always just on the [0]th entry for the catalog
            try:
                if list_of_cutouts[counterpart_cat_idx][0]['counterparts']:
                    list_of_counterparts = list_of_cutouts[counterpart_cat_idx][0]['counterparts']
                else:
                    counterpart_cat_idx = 0
                    best_len = 0
                    for idx,cat in enumerate(list_of_cutouts):
                        if cat[0]['counterparts']:
                            if len(cat[0]['counterparts']) > best_len:
                                counterpart_cat_idx = idx
                                best_len = len(cat[0]['counterparts'])

                    list_of_counterparts = list_of_cutouts[counterpart_cat_idx][0]['counterparts']
            except:
                pass

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

        if len(list_of_counterparts) < 1:
            # per Karl insert a blank row
            text = "No matching targets in catalog.\nRow intentionally blank."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            plt.close()
            self.add_bid_entry(fig)
            return self.pages
        elif (not G.FORCE_SINGLE_PAGE) and (len(list_of_counterparts) > G.MAX_COMBINE_BID_TARGETS):
            text = "Too many matching targets in catalog.\nIndividual target reports on followin pages."
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font)
            self.add_bid_entry(fig)
            return self.pages


        bid_colors = self.get_bid_colors(len(list_of_counterparts))

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
        phot_z_plotted = False
        phot_z_reference_lines_plotted = False
        make_phot_z_plot = False
        # targets are in order of increasing likelihood


        #iterate over the bid targets (list_of_counterparts) and build up the text
        for counterpart in list_of_counterparts:
            col_idx += 1
            if col_idx >= cols:
                log.info("cat_base::build_cat_summary_pdf_section. Number of bid targets exceeds number of columns.")
                break
            target_count += 1
            text = ""

            if G.ZOO:
                text = text + "%g\"\n%0.3f\n" \
                       % (counterpart.distance,counterpart.prob_match)
            else:
                text = text + "%g\"\n%0.3f\n%f, %f\n" \
                       % ( counterpart.distance,counterpart.prob_match,
                           counterpart.bid_ra, counterpart.bid_dec)

            #spec_z and phot_z
            if counterpart.spec_z is not None:
                text += "%g\n" % counterpart.spec_z
            else:
                text += "N/A\n"

            if counterpart.phot_z is not None:
                text += "%g\n" % counterpart.phot_z
            else:
                text += "N/A\n"

            #EW
            if counterpart.bid_ew_lya_rest is not None:
                if counterpart.bid_ew_lya_rest_err is not None:
                    text = text + utilities.unc_str((counterpart.bid_ew_lya_rest, counterpart.bid_ew_lya_rest_err)) + "$\AA$\n"
                else:
                    text = text + utilities.unc_str((counterpart.bid_ew_lya_rest, 0.0)) + "$\AA$\n"
            else:
                text += "---\n"

            #mag
            if counterpart.bid_mag is not None:
                text += "%0.2f" % counterpart.bid_mag
            if counterpart.bid_mag_err_bright is not None:
                text += "(%0.2f," % counterpart.bid_mag_err_bright
            else:
                text += "(--.--,"
            if counterpart.bid_mag_err_faint is not None:
                text += "%0.2f)" % counterpart.bid_mag_err_faint
            else:
                text += "--.--)"

            if counterpart.bid_filter is not None:
                text += counterpart.bid_filter
            text += "\n"

            if not make_phot_z_plot:
                if (counterpart.phot_z_pdf_pz is not None) and (counterpart.phot_z_pdf_z is not None) and \
                        (len(counterpart.phot_z_pdf_pz) == len(counterpart.phot_z_pdf_z ) != 0):
                    make_phot_z_plot = True
                elif counterpart.spec_z is not None and counterpart.spec_z >= 0.0:
                    make_phot_z_plot = True

            #plae/poii
            if (not G.ZOO) and (counterpart.p_lae_oii_ratio is not None):
                try:
                    text += r"$%0.4g\ ^{%.4g}_{%.4g}$" % (utilities.saferound(counterpart.p_lae_oii_ratio, 3),
                                                          utilities.saferound(counterpart.p_lae_oii_ratio_max, 3),
                                                          utilities.saferound(counterpart.p_lae_oii_ratio_min, 3))
                    text += "\n"
                except:
                    text += "%0.4g\n" % ( utilities.saferound(counterpart.p_lae_oii_ratio,3))

            else:
                text += "---\n"

            plt.subplot(gs[0, col_idx])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            plt.text(0, 0, text, ha='left', va='bottom', fontproperties=font,color=bid_colors[col_idx-1])

        #now the phot z plot (if there are any to plot)


        if make_phot_z_plot:
            col_idx = 0 #reset the col_idx for colors
            plt.subplot(gs[0, 4:])
            plt.xlim([0, 3.6])

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
                phot_z_reference_lines_plotted = True

            plt.title("Phot z PDF")
            plt.gca().yaxis.set_visible(False)

            for counterpart in list_of_counterparts:
                col_idx += 1
                if (counterpart.phot_z_pdf_pz is not None) and (counterpart.phot_z_pdf_z is not None) and \
                        (len(counterpart.phot_z_pdf_pz) == len(counterpart.phot_z_pdf_z ) != 0):
                    x = counterpart.phot_z_pdf_z
                    y = counterpart.phot_z_pdf_pz
                    plt.plot(x, y, color=bid_colors[col_idx-1])

                if counterpart.spec_z is not None and counterpart.spec_z >= 0.0:
                    plt.scatter([counterpart.spec_z,],[plt.gca().get_ylim()[1]*0.9,],zorder=9,
                                    marker="o",s=80,facecolors='none',edgecolors=bid_colors[col_idx-1])

        if not make_phot_z_plot:
            plt.subplot(gs[0, 4:])
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')
            text = "Phot z plot not available."
            plt.text(0, 0.5, text, ha='left', va='bottom', fontproperties=font)

        self.add_bid_entry(fig)

        #end build_cat_summary_pdf_section()
        return self.pages

    def build_cat_summary_details(self,cat_match, ra, dec, error, bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None,detobj=None,do_sky_subtract=True):
        """
        similar to build_cat_summary_figure, but rather than build up an image section to be displayed in the
        elixer report, this builds up a dictionary of information to be aggregated later over multiple catalogs

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
        :param do_sky_subtract: normally True (but HST might not want to perform additional sky subtraction)
        :return: cutouts list of dictionaries with bid-target objects as well
        """


        window = error * 3
        cutouts = self.get_cutouts(ra,dec,window/3600.,aperture=-1,filter=None,first=False,error=error,
                                   do_sky_subtract=do_sky_subtract,detobj=detobj)

        #do other stuff
        #1 get PLAE/POII for "best" aperture in each cutout image with details
        for c in cutouts:
            if c['details'] is None:
                continue

            #if (not target_flux) or (c['details']['filter_name'].lower() not in ['r','g','f606w']):
            #    continue

            mag = c['details']['mag']
            filter = c['details']['filter_name'].lower()
            details = c['details']

            #set the continuum estimate from the broadband filter
            #if no aperture magnitude was calculated, set it to the mag-limit
            cont_est = -999
            try:
                if (not details['mag']) or not (details['mag'] < 99): #no mag could be calculated
                    non_detect = min(self.get_mag_limit(details['filter_name'],details['radius']),33.0) #33 is just there for some not 99 limit
                    cont_est = self.obs_mag_to_cgs_flux(non_detect,SU.filter_iso(filter,target_w))
                else:
                    #+0.2 for about a 20% error on the mag_limit (allowing for measurement error and often
                    #smaller aperture size than what was used to measure the limit)
                    if c['mag_limit'] and (c['mag_limit'] < 99.0) and c['mag'] and (c['mag'] > (c['mag_limit'] + 0.2)):
                        details['fail_mag_limit']=True
                        details['raw_mag'] = mag
                        details['raw_mag_bright'] = details['mag_bright']
                        details['raw_mag_faint'] = details['mag_faint']

                        mag = c['mag_limit']
                        c['mag'] = mag
                        details['mag'] = mag
                        details['mag_bright'] = mag
                        details['mag_faint'] = 99.9

                        log.info(f"Mag {details['raw_mag']} below limit {c['mag_limit']}. Setting to limit. {details['catalog_name']} {details['filter_name']}")

                    cont_est = self.obs_mag_to_cgs_flux(mag,SU.filter_iso(filter,target_w))
            except:
                try:
                    log.info(f"Warning. Unable to get continuum estimate from bandpass. {details['catalog_name']} {details['filter_name']}")
                except:
                    log.info("Warning. Unable to get continuum estimate from bandpass.")

            bid_target = match_summary.BidTarget()
            bid_target.catalog_name = self.Name
            bid_target.bid_ra = 666  # nonsense RA
            bid_target.bid_dec = 666  # nonsense Dec
            bid_target.distance = 0.0
            bid_target.bid_filter = filter
            bid_target.bid_mag = mag
            bid_target.bid_mag_err_bright = 0.0 #todo: right now don't have error on aperture mag
            bid_target.bid_mag_err_faint = 0.0
            bid_target.bid_flux_est_cgs_unc = 0.0

            bid_target.bid_flux_est_cgs = cont_est
            try:
                flux_faint = None
                flux_bright = None

                if details['mag_faint'] < 99:
                    flux_faint = self.obs_mag_to_cgs_flux(details['mag_faint'], SU.filter_iso(filter,target_w))

                if details['mag_bright'] < 99:
                    flux_bright = self.obs_mag_to_cgs_flux(details['mag_bright'], SU.filter_iso(filter,target_w))

                if flux_bright and flux_faint:
                    bid_target.bid_flux_est_cgs_unc = max((bid_target.bid_flux_est_cgs - flux_faint),
                                                          (flux_bright -bid_target.bid_flux_est_cgs))
                elif flux_bright:
                    bid_target.bid_flux_est_cgs_unc = flux_bright - bid_target.bid_flux_est_cgs
                else: #neither faint nor bright, so at the limit
                    pass #anything we can reasonably do here? fully below flux limit?

            except:
                pass

            try:
                bid_target.bid_mag_err_bright = mag - details['mag_bright']
                bid_target.bid_mag_err_faint =  details['mag_faint'] - mag
            except:
                pass

            bid_target.add_filter(c['instrument'], filter, bid_target.bid_flux_est_cgs, -1)

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

            #only run PLAE/POII for g or r bands and if we have a line flux
            if target_flux and (c['details']['filter_name'].lower() in ['r','g','f606w']):
                # build EW error from lineFlux_err and aperture estimate error
                # ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
                # try:
                #     ew_obs_err = abs(ew_obs * np.sqrt(
                #         (lineFlux_err / target_flux) ** 2 +
                #         (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
                # except:
                #     ew_obs_err = 0.

                ew_obs, ew_obs_err = SU.ew_obs(target_flux,lineFlux_err,target_w,
                                                   filter, bid_target.bid_flux_est_cgs,bid_target.bid_flux_est_cgs_unc)

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
                        c['aperture_plae_min'] = plae_errors['ratio'][1] #new key
                        c['aperture_plae_max'] = plae_errors['ratio'][2] #new key
                except:
                    pass

                c['aperture_plae'] = bid_target.p_lae_oii_ratio
                c['aperture_eqw_rest_lya'] = ew_obs / (1. + target_w / G.LyA_rest)
                c['aperture_eqw_rest_lya_err'] = ew_obs_err / (1. + target_w / G.LyA_rest)

                #also goes into the details
                if c['details']:
                    c['details']['aperture_plae'] = bid_target.p_lae_oii_ratio
                    c['details']['aperture_eqw_rest_lya'] = ew_obs / (1. + target_w / G.LyA_rest)
                    c['details']['aperture_eqw_rest_lya_err'] = ew_obs_err / (1. + target_w / G.LyA_rest)
                    try:
                        if plae_errors:
                            c['details']['aperture_plae_min'] = plae_errors['ratio'][1] #new key
                            c['details']['aperture_plae_max'] = plae_errors['ratio'][2] #new key
                    except:
                        pass

            cat_match.add_bid_target(bid_target)
            try:  # no downstream edits so they can both point to same bid_target
                if detobj is not None:
                    detobj.bid_target_list.append(bid_target)
            except:
                log.warning("Unable to append bid_target to detobj.", exc_info=True)



            #####################################################
            #Common to all catalogs; add the aperture details
            #####################################################

            if (details is not None) and (detobj is not None):
                detobj.aperture_details_list.append(details)


            #the bid targets (conterparts) are added back in the child class

        return cutouts

        # #2. catalog entries as a new key under cutouts (like 'details') ... 'counterparts'
        # #    this should be similar to the build_multiple_bid_target_figures_one_line()
        #
        # if len(bid_ras) > 0:
        #     #if there are no cutouts (but we do have a catalog), create a cutouts list of dictionries to hold the
        #     #counterparts
        #     if not cutouts or len(cutouts) == 0:
        #         cutouts = [{}]
        #
        #     cutouts[0]['counterparts'] = []
        #     #create an empty list of counterparts under the 1st cutout
        #     #counterparts are not filter specific, so we will just keep one list under the 1st cutout
        #
        # target_count = 0
        # # targets are in order of increasing distance
        # for r, d in zip(bid_ras, bid_decs):
        #     target_count += 1
        #     if target_count > G.MAX_COMBINE_BID_TARGETS:
        #         break
        #
        #     try: #DO NOT WANT _unique as that has wiped out the filters
        #         df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
        #                                                (self.dataframe_of_bid_targets['DEC'] == d[0]) &
        #                                                (self.dataframe_of_bid_targets['FILTER'] == 'r')]
        #         if (df is None) or (len(df) == 0):
        #             df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
        #                                                    (self.dataframe_of_bid_targets['DEC'] == d[0]) &
        #                                                    (self.dataframe_of_bid_targets['FILTER'] == 'g')]
        #         if (df is None) or (len(df) == 0):
        #             df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
        #                                                    (self.dataframe_of_bid_targets['DEC'] == d[0])]
        #
        #     except:
        #         log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
        #         continue  # this must be here, so skip to next ra,dec
        #
        #     if df is not None:
        #         #add flux (cont est)
        #         try:
        #             filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = self.get_filter_flux(df)
        #         except:
        #             filter_fl = 0.0
        #             filter_fl_err = 0.0
        #             filter_mag = 0.0
        #             filter_mag_bright = 0.0
        #             filter_mag_faint = 0.0
        #             filter_str = "NA"
        #
        #         bid_target = None
        #
        #         if (target_flux is not None) and (filter_fl != 0.0):
        #             if (filter_fl is not None):# and (filter_fl > 0):
        #                 filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,SU.filter_iso(filter_str,target_w)) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
        #                 filter_fl_cgs_unc = self.nano_jansky_to_cgs(filter_fl_err, SU.filter_iso(filter_str,target_w))
        #                 # assumes no error in wavelength or c
        #
        #                 try:
        #                     bid_target = match_summary.BidTarget()
        #                     bid_target.catalog_name = self.Name
        #                     bid_target.bid_ra = df['RA'].values[0]
        #                     bid_target.bid_dec = df['DEC'].values[0]
        #                     bid_target.distance = df['distance'].values[0] * 3600
        #                     bid_target.prob_match = df['dist_prior'].values[0]
        #                     bid_target.bid_flux_est_cgs = filter_fl_cgs
        #                     bid_target.bid_filter = filter_str
        #                     bid_target.bid_mag = filter_mag
        #                     bid_target.bid_mag_err_bright = filter_mag_bright
        #                     bid_target.bid_mag_err_faint = filter_mag_faint
        #                     bid_target.bid_flux_est_cgs_unc = filter_fl_cgs_unc
        #
        #                     try:
        #                         ew = (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))
        #                         ew_u = abs(ew * np.sqrt(
        #                             (detobj.estflux_unc / target_flux) ** 2 +
        #                             (filter_fl_err / filter_fl) ** 2))
        #
        #                         bid_target.bid_ew_lya_rest = ew
        #                         bid_target.bid_ew_lya_rest_err = ew_u
        #
        #                     except:
        #                         log.debug("Exception computing catalog EW: ", exc_info=True)
        #
        #                     addl_waves = None
        #                     addl_flux = None
        #                     addl_ferr = None
        #                     try:
        #                         addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
        #                         addl_flux = cat_match.detobj.spec_obj.addl_fluxes
        #                         addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
        #                     except:
        #                         pass
        #
        #                     lineFlux_err = 0.
        #                     if detobj is not None:
        #                         try:
        #                             lineFlux_err = detobj.estflux_unc
        #                         except:
        #                             lineFlux_err = 0.
        #
        #                     # build EW error from lineFlux_err and aperture estimate error
        #                     ew_obs = (target_flux / bid_target.bid_flux_est_cgs)
        #                     try:
        #                         ew_obs_err = abs(ew_obs * np.sqrt(
        #                             (lineFlux_err / target_flux) ** 2 +
        #                             (bid_target.bid_flux_est_cgs_unc / bid_target.bid_flux_est_cgs) ** 2))
        #                     except:
        #                         ew_obs_err = 0.
        #
        #                     bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii,plae_errors = \
        #                         line_prob.mc_prob_LAE(
        #                             wl_obs=target_w,
        #                             lineFlux=target_flux,
        #                             lineFlux_err=lineFlux_err,
        #                             continuum=bid_target.bid_flux_est_cgs,
        #                             continuum_err=bid_target.bid_flux_est_cgs_unc,
        #                             c_obs=None, which_color=None,
        #                             addl_wavelengths=addl_waves,
        #                             addl_fluxes=addl_flux,
        #                             addl_errors=addl_ferr,
        #                             sky_area=None,
        #                             cosmo=None, lae_priors=None,
        #                             ew_case=None, W_0=None,
        #                             z_OII=None, sigma=None)
        #
        #                     try:
        #                         if plae_errors:
        #                             bid_target.p_lae_oii_ratio_min = plae_errors['ratio'][1]
        #                             bid_target.p_lae_oii_ratio_max = plae_errors['ratio'][2]
        #                     except:
        #                         pass
        #
        #                     try:
        #                         bid_target.add_filter('HSC','R',filter_fl_cgs,filter_fl_err)
        #                     except:
        #                         log.debug('Unable to build filter entry for bid_target.',exc_info=True)
        #
        #                     cat_match.add_bid_target(bid_target)
        #                     try:  # no downstream edits so they can both point to same bid_target
        #                         detobj.bid_target_list.append(bid_target)
        #                     except:
        #                         log.warning("Unable to append bid_target to detobj.", exc_info=True)
        #
        #                     try:
        #                         cutouts[0]['counterparts'].append(bid_target)
        #                     except:
        #                         log.warning("Unable to append bid_target to cutouts.", exc_info=True)
        #                 except:
        #                     log.debug('Unable to build bid_target.',exc_info=True)
        #

        #once all done, pass cutouts to a new function to create the summary figure ... SHOULD be only in the base class
        #as all should be identical and based only on the cutouts object data
        # this include
        #1. stack cutouts to make a master cutout ...
        #2. add fibers, North box to master cutout ...

        #return cutouts




    def build_cat_summary_figure (self,ra,dec,error,bid_ras,bid_decs, target_w=0, fiber_locs=None, target_flux=None,detobj=None):
        #implement in child class
        pass


    def build_bid_target_figure_one_line (self,cat_match, ra, dec, error, df=None, df_photoz=None, target_ra=None,
                                          target_dec=None, section_title="", bid_number=1, target_w=0, of_number=0,
                                          target_flux=None, color="k",detobj=None):
        # implement in child class
        pass

    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None,detobj=None):
        # implement in child class
        pass


    def is_edge_fiber(self,absolute_fiber_num,ifux=None,ifuy=None):
        """
        fiber_num is the ABSOLUTE fiber number 1-448
        NOT the per amp number (1-112)

        or use fiber center IFUx and IFUy as the fiber may be in a non-standard location
        but the IFUx and IFUy should be correct

        # -22.88 < x < 22.88 ,  -24.24 < y < 24.24

        :param fiber_num:
        :return:
        """

        if ifux is None or ifuy is None:
            return absolute_fiber_num in G.CCD_EDGE_FIBERS_ALL
        else:
            #back off just a bit for some slop
            if (-22.5 < ifux < 22.5) and (-24.0 < ifuy < 24.0):
                return False
            else:
                return True




    def add_fiber_positions(self,plt,ra,dec,fiber_locs,error,ext,cutout,unlabeled=False,use_gray_cmap=True):
            # plot the fiber cutout
            log.debug("Plotting fiber positions...")

            #temporary for Milos (examples for segementation)
            #np.save("r"+str(ra)+"d"+str(dec), cutout.data)

            #test
            #ar = np.load("r"+str(ra)+"d"+str(dec)+".npy")
            #plt.imshow(ar, origin='lower')

            try:
                empty_sci = science_image.science_image()

                if (fiber_locs is None) or (len(fiber_locs) == 0):
                    fiber_locs = []
                    plt.title("")
                else:
                    if unlabeled:
                        plt.title("")
                    else:
                        plt.title("Fiber Positions")
                if not unlabeled:
                    plt.xlabel("arcsecs")
                    plt.gca().xaxis.labelpad = 0

                #plt.plot(0, 0, "r+")
                xmin = float('inf')
                xmax = float('-inf')
                ymin = float('inf')
                ymax = float('-inf')

                x, y = empty_sci.get_position(ra, dec, cutout)  # zero (absolute) position

                for r, d, c, i, dist,fn, ifux,ifuy in fiber_locs:
                    # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                    fx, fy = empty_sci.get_position(r, d, cutout)

                    xmin = min(xmin, fx - x)
                    xmax = max(xmax, fx - x)
                    ymin = min(ymin, fy - y)
                    ymax = max(ymax, fy - y)

                    plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius, color=c, fill=False,
                                                   linestyle='solid'))
                   #stop displaying the 1-5 fiber number
                   # plt.text((fx - x), (fy - y), str(i), ha='center', va='center', fontsize='x-small', color=c)

                    if self.is_edge_fiber(fn,ifux,ifuy):
                        plt.gca().add_patch(
                            plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius+0.1, color=c, fill=False,
                                       linestyle='dashed'))

                # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
                #can't do this ... if you just rescale for the fibers, the underlying coordinate system becomes invalid
                # ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
                # if ext_base != np.inf:
                #     ext = max(ext_base + G.Fiber_Radius, ext)


                # need a new cutout since we rescaled the ext (and window) size
                if self.master_cutout:
                    cutout,_,_,_ = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
                    if cutout is None:
                        log.warning("Cannot obtain new cutout from master_cutout in cat_base::add_fiber_positions")#,exc_info=True)
                        cutout = self.master_cutout

                vmin, vmax = empty_sci.get_vrange(cutout.data)

                if not unlabeled:
                    self.add_north_box(plt, empty_sci, cutout, error, 0, 0, theta=None)
                if use_gray_cmap:
                    plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])
                else:
                    plt.imshow(cutout.data, origin='lower', interpolation='none',
                               vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])


                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])

                #needs to be here at the end due to rescaling
                #self.add_zero_position(plt)
                plt.plot(0, 0, "r+")

                if unlabeled:
                    plt.axis("off")
                    plt.gca().set_yticklabels([])
                    plt.gca().set_xticklabels([])

            except:
                log.error("Unable to overplot fiber positions.",exc_info=True)


    def add_zero_position(self,plt):

        try:
            # over plot a reticle for the 0,0 position
            line_frac = 0.2 #length of the line (each arm) as a fraction of the image size
            open_frac = 0.1 #length of the open area (as a fraction of the image size) in between the arms
            _color='r'
            _zorder=1

            #get the dimensions
            xl,xr = plt.gca().get_xlim()
            yl,yr = plt.gca().get_ylim()

            cx = (xl + xr)/2.0
            cy = (yl + yr)/2.0

            dx = xr - xl
            dy = yr - yl

            #left arm
            plt.hlines(cy,xmin=cx-(dx*open_frac/2.0)-dx*line_frac, xmax=cx-(dx*open_frac/2.0),colors=_color,
                       linestyles='solid',linewidth=1.0,zorder=_zorder)
            #right arm
            plt.hlines(cy, xmin=cx+(dx*open_frac/2.0), xmax=cx+(dx*open_frac/2.0)+dx*line_frac, colors=_color,
                       linestyles='solid', linewidth=1.0,zorder=_zorder)

            #bottom arm
            plt.vlines(cx,ymin=cy-(dy*open_frac/2.0)-dy*line_frac, ymax=cy-(dy*open_frac/2.0),colors=_color,
                       linestyles='solid',linewidth=1.0,zorder=_zorder)
            # top arm
            plt.vlines(cx, ymin=cy+(dy*open_frac/2.0), ymax=cy+(dy*open_frac/2.0)+dy*line_frac, colors=_color,
                       linestyles='solid', linewidth=1.0,zorder=_zorder)
        except:
            log.error("Exception! in cat_base::add_zero_position()",exc_info=True)


    def add_catalog_position(self,plt,x,y,size,color):
        try:
            plt.gca().add_patch(plt.Rectangle((x,y), width=size, height=size,
                                              angle=0.0, color=color, fill=False, linewidth=1.0, zorder=3))
        except:
            log.info("Exception!",exc_info=True)



    def add_aperture_position(self,plt,radius,mag=None,cx=0,cy=0,ew=None,plae=None,distance_to_center=None):
            # over plot a circle of radius on the center of the image (assumed to be the photo-aperture)
            try:
                if radius:
                    log.debug("Plotting imaging aperture position...")

                    if (cx is None) or (cy is None):
                        cx = 0
                        cy = 0

                    if distance_to_center is None:
                        distance_to_center = np.sqrt(cx*cx+cy*cy)

                    try:
                        plt.gca().add_patch(plt.Circle((cx,cy), radius=radius, color='gold', fill=False,
                                                       linestyle='solid'))

                        #temporary
                        if mag is not None:
                            label = "m:%0.1f rc:%0.1f\"  s:%0.1f\"" % (mag,radius,distance_to_center)

                            if ew is not None:
                                label += "\n EWr: %0.0f" %(ew)
                                if plae is not None:
                                    label += ", PLAE: %0.4g" %(round(plae, 3))

                            plt.xlabel(label)
                            plt.gca().xaxis.labelpad = 0
                            plt.subplots_adjust(bottom=0.1)
                            #plt.tight_layout()

                    except:
                        log.error("Unable to overplot aperture position.",exc_info=True)
            except:
                log.error("Unable to overplot aperture position.",exc_info=True)


    def add_elliptical_aperture_positions(self,plt,ellipse_objs,selected_idx=None,radius=None,mag=None,cx=0,cy=0,ew=None,plae=None):

            try:
                log.debug("Plotting imaging (elliptical) aperture position...")

                if (cx is None) or (cy is None):
                    cx = 0
                    cy = 0

                image_width = plt.gca().get_xlim()[1]*2.0 #center is at 0,0; this is a square so x == y

                for eobj in ellipse_objs:
                    use_circle = False
                    a = eobj['a'] #major axis diameter in arcsec
                    b = eobj['b']
                    ellipse_radius = 0.5*np.sqrt(a*b) #approximate radius (treat ellipse like a circle)

                    if eobj['selected']:
                        color = 'gold'
                        alpha = 1.0
                        zorder = 2
                        ls='solid'
                    else:
                        color = 'white'
                        alpha = 0.8
                        zorder = 1
                        ls = '--'

                    if ellipse_radius/image_width < 0.1: #1/9" ... typical ... want close to 1"
                        log.debug("Ellipse too small (r ~ %0.2g). Using larger circle to highlight." %(ellipse_radius))
                        a = b = 0.1 * image_width * 2. #a,b are diameters so 2x
                        ls = '--'

                    plt.gca().add_artist(Ellipse(xy=(eobj['x'], eobj['y']),
                                width=a,  # diameter with (*6 is for *6 kron isophotal units)?
                                height=b,
                                angle=eobj['theta'] * 180. / np.pi,
                                facecolor='none',
                                edgecolor=color, alpha=alpha,zorder=zorder,linestyle=ls))

                    if (eobj['selected']) and (mag is not None):
                        label = "m:%0.1f  re:%0.1f\"  s:%0.1f\"" % (mag,ellipse_radius,eobj['dist_baryctr'])

                        if ew is not None:
                            label += "\n EWr: %0.0f" %(ew)
                            if plae is not None:
                                label += ", PLAE: %0.4g" %(round(plae, 3))

                        plt.xlabel(label)
                        plt.gca().xaxis.labelpad = 0
                        plt.subplots_adjust(bottom=0.1)
            except:
                log.error("Unable to overplot (elliptical) aperture position.",exc_info=True)

            if (selected_idx is None) and (radius is not None):
                return self.add_aperture_position(plt,radius,mag,cx,cy,ew,plae)


    def add_empty_catalog_fiber_positions(self, plt,fig,ra,dec,fiber_locs):
        '''used if there is no catalog. Just plot relative positions'''

        if fiber_locs is None: #there are no fiber locations (could be a specific RA,Dec search)
            return None

        try:
            plt.title("Relative Fiber Positions")
            #plt.plot(0, 0, "r+")

            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')

            for r, d, c, i, dist, fn, ifux,ifuy in fiber_locs:
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx = (r - ra) * np.cos(np.deg2rad(dec)) * 3600.
                fy = (d - dec) * 3600.

                xmin = min(xmin, fx)
                xmax = max(xmax, fx)
                ymin = min(ymin, fy)
                ymax = max(ymax, fy)

                plt.gca().add_patch(plt.Circle((fx, fy), radius=G.Fiber_Radius, color=c, fill=False,
                                             linestyle='solid', zorder=9))
                plt.text(fx, fy, str(i), ha='center', va='center', fontsize='x-small', color=c)

                if fn in G.CCD_EDGE_FIBERS_ALL:
                    plt.gca().add_patch(
                        plt.Circle((fx, fy), radius=G.Fiber_Radius + 0.1, color=c, fill=False,
                                   linestyle='dashed', zorder=9))

            # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
            ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            ext = np.ceil(ext_base + G.Fiber_Radius)

            plt.xlim(-ext,ext)
            plt.ylim(-ext,ext)
            plt.xlabel('arcsecs')
            plt.axis('equal')

            rec = plt.Rectangle((-ext,-ext), width=ext*2., height=ext*2., fill=True, lw=1,
                                color='gray', zorder=0, alpha=0.5)
            plt.gca().add_patch(rec)

            plt.xticks([int(-ext), int(-ext / 2.), 0, int(ext / 2.), int(ext)])
            plt.yticks([int(-ext), int(-ext / 2.), 0, int(ext / 2.), int(ext)])

            #needs to be at the end
            #self.add_zero_position(plt)
            plt.plot(0, 0, "r+")
        except:
            log.error("Exception in cat_base::add_empty_catalog_fiber_positions.",exc_info=True)


    def edge_compass(self,fiber_num):
        # return 8 point compass direction if edge (East is left)
        if fiber_num in G.CCD_EDGE_FIBERS_ALL:
            if fiber_num in G.CCD_EDGE_FIBERS_TOP:
                if fiber_num in G.CCD_EDGE_FIBERS_RIGHT:
                    return 'NW'
                elif fiber_num in G.CCD_EDGE_FIBERS_LEFT:
                    return 'NE'
                else:
                    return 'N'
            elif fiber_num in G.CCD_EDGE_FIBERS_BOTTOM:
                if fiber_num in G.CCD_EDGE_FIBERS_RIGHT:
                    return 'SW'
                elif fiber_num in G.CCD_EDGE_FIBERS_LEFT:
                    return 'SE'
                else:
                    return 'S'
            elif fiber_num in G.CCD_EDGE_FIBERS_RIGHT:
                return 'W'
            elif fiber_num in G.CCD_EDGE_FIBERS_LEFT:
                return 'E'
            else:
                return ''
        else:
            return ''

    #todo: note cannot do this with the new t5cut that has multiple observations and specifies the RA and Dec for
    #each fiber since we no longer know the rotation (and there might be multiple rotations, since multiple observations)
    #so we cannot correctly find the corner fibers relative to the target fiber directly, or build a
    #tangentplane(s) (need rotation(s) and fplane center(s) RA, Dec)
    #If there are two or more fibers from the same exposure, I could still figure it out, but that is going to be
    #less accurate? and certainly unreliable (that there will be 2 or more fibers)
    def add_ifu_box(self,plt,sci,cutout,corner_ras,corner_decs, color=None):
        #theta is angle in radians counter-clockwise from x-axis to the north celestrial pole

        if (plt is None) or (sci is None) or (cutout is None):
            return

        try:
            #todo: just call plt with arrays of ra and dec, no marker, dashed line, color = color (None is okay)
            #todo: I don't think we need to translate the ra and decs ... just send in as is
            pass


        except:
            log.error("Exception bulding ifu footprint box.", exc_info=True)


    #really should not be necessary (all uses actually get each position from RA, and Dec
    #and not a coordinate scale or angular distance scale
    def scale_dx_dy_for_dec(self,dec,dx,dy,sci,cutout):
        scale = np.cos(np.deg2rad(dec))

        #rotation is from the x-axis, so to make from vertical, - np.pi/2.
        rad = sci.get_rotation_to_celestrial_north(cutout) - np.pi/2.

        dx -= scale*np.cos(rad) #negative because RA increases East and East is -x
        dy += scale*np.sin(rad)

        return dx,dy

    def build_empty_cat_summary_figure(self, ra, dec, error, bid_ras, bid_decs, target_w=0,
                                 fiber_locs=None):
        '''Builds the figure (page) the exact target location. Contains just the filter images ...'''

        rows = 10  # 2 (use 0 for text and 1: for plots)
        cols = 6  # just going to use the first, but this sets about the right size

        fig_sz_x = 18  # cols * 3 # was 6 cols
        fig_sz_y = 3  # rows * 3 # was 1 or 2 rows

        try:
            fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)

            gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
            # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

            font = FontProperties()
            font.set_family('monospace')
            font.set_size(12)

            # All on one line now across top of plots
            title = "No overlapping imaging catalog (or image is blank)."

            plt.subplot(gs[0, :])
            plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
            plt.gca().set_frame_on(False)
            plt.gca().axis('off')

            plt.subplot(gs[2:, 0])
            self.add_empty_catalog_fiber_positions(plt, fig, ra, dec, fiber_locs)

            # complete the entry
            plt.close()
            return fig
        except:
            log.error("Exception in cat_base::build_empty_cat_summary_figure.",exc_info=True)
            return None


    def get_single_cutout(self,ra,dec,window,catalog_image,aperture=None,error=None,do_sky_subtract=True,detobj=None):
        #window is in DEGREES
        #error in arcsec and is for internal ELiXer use

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
                catalog_image['image'] = science_image.science_image(wcs_manual=wcs_manual,
                                                         image_location=op.join(catalog_image['path'],
                                                                                catalog_image['name']))
                catalog_image['image'].catalog_name = catalog_image['name']
                catalog_image['image'].filter_name = catalog_image['filter']

            sci = catalog_image['image']


            if (sci.headers is None) or (len(sci.headers) == 0): #the catalog_image['image'] is no good? reload?
                sci.load_image(wcs_manual=wcs_manual)

            d['path'] = sci.image_location
            d['hdu'] = sci.headers

            #to here, window is in degrees so ...
            window = 3600.*window
            if not error:
                error = window

            if aperture == -1:
                try:
                    aperture = catalog_image['aperture']
                except:
                    pass

            cutout,pix_counts, mag, mag_radius,details = sci.get_cutout(ra, dec, error=error, window=window, aperture=aperture,
                                             mag_func=mag_func,copy=True,return_details=True,do_sky_subtract=do_sky_subtract,
                                             detobj=detobj)
            #don't need pix_counts or mag, etc here, so don't pass aperture or mag_func

            if cutout is not None:  # construct master cutout
               d['cutout'] = cutout
               details['catalog_name']=self.name
               details['filter_name']=catalog_image['filter']
               d['mag_limit']=self.get_mag_limit(catalog_image['name'],mag_radius*2.)
               if details:
                   d['details'] = details

               try:
                   if d['mag_limit']:
                       details['mag_limit']=d['mag_limit']
                   else:
                       details['mag_limit'] = None
               except:
                   details['mag_limit'] = None

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
            log.error("Error in get_single_cutout.",exc_info=True)

        return d

    # def contains_position(self,ra,dec,window,catalog_image,aperture=None,verify=True):
    #     #window is in DEGREES
    #
    #     d = {'cutout':None,
    #          'hdu':None,
    #          'path':None,
    #          'filter':catalog_image['filter'],
    #          'instrument':catalog_image['instrument'],
    #          'mag':None,
    #          'aperture':None,
    #          'ap_center':None,
    #          'found':None}
    #     try:
    #         wcs_manual = catalog_image['wcs_manual']
    #         mag_func = catalog_image['mag_func']
    #     except:
    #         wcs_manual = self.WCS_Manual
    #         mag_func = None
    #
    #     try:
    #         if catalog_image['image'] is None:
    #             catalog_image['image'] = science_image.science_image(wcs_manual=wcs_manual,
    #                                                      image_location=op.join(catalog_image['path'],
    #                                                                             catalog_image['name']))
    #             catalog_image['image'].catalog_name = catalog_image['name']
    #             catalog_image['image'].filter_name = catalog_image['filter']
    #
    #         sci = catalog_image['image']
    #
    #         if sci.hdulist is None:
    #             sci.load_image(wcs_manual=wcs_manual)
    #
    #         d['path'] = sci.image_location
    #         d['hdu'] = sci.headers
    #
    #         rc = sci.contains_position(ra,dec,verify)
    #         d['found'] = rc
    #     except:
    #         log.error("Error in contains_position.",exc_info=True)
    #
    #     return d


    #generic, can be used by most catalogs (like CANDELS), only override if necessary
    def get_filters(self,ra=None,dec=None):
        '''
        Return list of (unique) filters included in this catalog (regardless of RA, Dec)

        :param ra: not used at this time
        :param dec:
        :return:
        '''
        #todo: future ... see if ra, dec are in the filter? that could be very time consuming

        try:
            cat_filters = list(set([x['filter'].lower() for x in self.CatalogImages]))
        except:
            cat_filters = None

        return list(set(cat_filters))

    #generic, can be used by most catalogs (like CANDELS), only override if necessary
    def get_cutouts(self,ra,dec,window,aperture=None,filter=None,first=False,error=None,do_sky_subtract=True,detobj=None):
        """

        :param ra:
        :param dec:
        :param window:
        :param aperture:
        :param filter:
        :param first:
        :param error:
        :param do_sky_subtract:
       # :param mag_check: usually the HETDEX-gmag, if known. selected SEP objects should also match the mag
        :return:
        """
        l = list()

        #not every catalog has a list of filters, and some contain multiples
        try:
            cat_filters = list(dict((x['filter'], {}) for x in self.CatalogImages).keys())
            #cat_filters = list(set([x['filter'].lower() for x in self.CatalogImages]))
        except:
            cat_filters = None

        if filter:
            outer = filter
            inner = cat_filters
        else:
            outer = cat_filters
            inner = None

        wild_filters = iter(cat_filters)

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
                             if ((d['filter'] == f)))]

                    cutout = self.get_single_cutout(ra, dec, window, i, aperture,error,do_sky_subtract,detobj)

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
            for i in self.CatalogImages:  # i is a dictionary
                # note: this works, but can be grossly inefficient and
                # should be overwritten by the child-class
                try:
                    l.append(self.get_single_cutout(ra, dec, window, i, aperture,detobj=detobj))
                    if first:
                        break
                except:
                    log.error("Exception! collecting image cutouts.", exc_info=True)
        return l

    #todo:
    def get_catalog_objects(self,ra,dec,window):
        #needs to be overwritten by each catalog

        d = {'count':-1,'name':None,'dataframe':None,'get_filter_flux':None}

        window = window * 3600. #came in as degress, want arsec
        num, df, photoz = self.build_list_of_bid_targets(ra,dec,window)

        if df is not None:
            d['count'] = num
            d['dataframe'] = df
            d['name'] = self.name
            d['get_filter_flux'] = self.get_filter_flux
            #very few have photoz, so lets just drop it?

        return d

    def get_stacked_cutout(self, ra, dec, window):

        stacked_cutout = None
        error = window

        cutouts = self.get_cutouts(ra, dec, window)

        stacked_cutout = None

        for c in cutouts:
            cutout = c['cutout']
            try:
                exptime = c['hdu']['exptime']
            except:
                exptime = 1.0
            if cutout is not None:  # construct master cutout
                if stacked_cutout is None:
                    stacked_cutout = copy.deepcopy(cutout)
                    ref_exptime = exptime
                    total_adjusted_exptime = 1.0
                else:
                    stacked_cutout.data = np.add(stacked_cutout.data, cutout.data * exptime / ref_exptime)
                    total_adjusted_exptime += exptime / ref_exptime

        return stacked_cutout


    # def write_cutout_as_fits(self,cutout,filename):
    #     '''write a cutout as a fits file'''
    #     pass


    # def estimate_aperture_flux_uncertainty(self,mag_func,counts,cutout,fits):
    # #sky has been subtracted or is not significant
    # #assume only Poisson noise in the counts ... need to know the count in EACH pixel though, to do that
    #     mx_mag = mag_func(counts + np.sqrt(counts), cutout, self.fits)
    #     mn_mag = mag_func(counts - np.sqrt(counts), cutout, self.fits)