from __future__ import print_function
#keep it simple for now. Put base class and all children in here.
#Later, create a proper package

import global_config as G
import os.path as op

import matplotlib
matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import science_image
from matplotlib.font_manager import FontProperties
import scipy.constants

import cat_bayesian
import observation as elixer_observation
#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

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

    def sort_bid_targets_by_likelihood(self,ra,dec):
        #right now, just by euclidean distance (ra,dec are of target) (don't forget to adjust RA coord difference
        #for the declination ... just use the target dec, the difference to the bid dec is negligable)
        #if radial separation is greater than the error, remove this bid target?
        #  remember we are looking in a box (error x error) so radial can be greater than errro (i.e. in a corner)

        #if _unique exists, the child catalog could have duplicates (stitched together from other filters, etc)
        #   so, use the _unique version to sort by distance, otherwise the same object will have multiple entries
        #   and the sort is messed up

        if hasattr(self,'dataframe_of_bid_targets_unique'):
            self.dataframe_of_bid_targets_unique['dist_prior'] = 1.0
            df = self.dataframe_of_bid_targets_unique
            df['distance'] = np.sqrt(
                (np.cos(np.deg2rad(dec)) * (df['RA'] - ra)) ** 2 + (df['DEC'] - dec) ** 2)

            df['dist_prior'] = 1.0
            pidx = df.columns.get_loc('dist_prior')

            for i in range(len(df)):
                filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = \
                    self.get_filter_flux(df.iloc[[i]]) #note the .iloc[[i]] so we get a dataframe not a series
                if filter_mag is not None:
                    p = self.distance_prior.get_prior(df.iloc[i]['distance'] * 3600.0, filter_mag)
                    df.iat[i, pidx] = p

            self.dataframe_of_bid_targets_unique = df.sort_values(by=['dist_prior','distance'], ascending=[False,True])


        #YES, both need to have this performed (this one always) as they are used for different purposes later
        if hasattr(self,'dataframe_of_bid_targets'): #sanity check ... all cats have this
            df = self.dataframe_of_bid_targets
            df['distance'] = np.sqrt(
                (np.cos(np.deg2rad(dec)) * (df['RA'] - ra)) ** 2 + (df['DEC'] - dec) ** 2)

            df['dist_prior'] = 1.0
            pidx = df.columns.get_loc('dist_prior')
            #note: if _unique exists, this sort here is redudant, otherwise it is needed
            for i in range(len(df)):
                filter_fl, filter_fl_err, filter_mag, filter_mag_bright, filter_mag_faint, filter_str = \
                    self.get_filter_flux(df.iloc[[i]]) #note the .iloc[[i]] so we get a dataframe not a series
                if filter_mag is not None:
                    p = self.distance_prior.get_prior(df.iloc[i]['distance']*3600.0,filter_mag)
                    df.iat[i,pidx] = p


            self.dataframe_of_bid_targets = df.sort_values(by=['dist_prior','distance'], ascending=[False,True])



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

        if (cutout is None) or (obs is None):
            log.error("Invalid parameters passed to build_annulus_report")
            return None

        self.clear_pages()

        #ra,dec,annulus should already be set as part of the obs (SyntheticObservation)

        #make sure the eli_dict is fully populated
        obs.build_complete_emission_line_info_dict()
        if len(obs.eli_dict) == 0: #problem, we're done
            log.error("Problem building complete EmissionLineInfo dictionary.")
            return None

        #now sub select the annulus fibers (without signal) (populate obs.fibers_work
        obs.annulus_fibers()
        if len(obs.fibers_work) == 0:
            log.warning("Warning. No fibers found inside the annulus.")

        rows = 1
        cols = 1

        fig_sz_x = G.ANNULUS_FIGURE_SZ_X #initial guess
        fig_sz_y = 10

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
       # plt.subplots_adjust(left=0.1, right=0.90, top=0.90, bottom=0.05)

        gs = gridspec.GridSpec(rows, cols, wspace=0.0, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        title = "Diffuse Emission"

        plt.subplot(gs[:, :]) #right now, only the one (if in future want more this needs to change)
        #text = plt.text(0, 0.7, title, ha='left', va='bottom', fontproperties=font)
        #plt.gca().set_frame_on(False)
        #plt.gca().axis('off')

        empty_sci = science_image.science_image()

        vmin, vmax = empty_sci.get_vrange(cutout.data)
        ext = obs.annulus[1]

        plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                   vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])

        plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
        plt.title(title)
        plt.ylabel('arcsec')
        plt.xlabel('arcsec')

        self.add_north_box(plt, empty_sci, cutout, obs.annulus[1], 0, 0, theta=None,box=False)

        #put in the fibers ... this is very similar to, but not the same as add_fiber_positions
        try:
            xmin = float('inf')
            xmax = float('-inf')
            ymin = float('inf')
            ymax = float('-inf')

            x, y = empty_sci.get_position(obs.ra, obs.dec, cutout)  # zero (absolute) position


            #plot annulus
            plt.gca().add_patch(plt.Circle((0, 0), radius=obs.annulus[0],
                                           facecolor='none', fill=False, alpha=1,
                                           edgecolor='k', linestyle="dashed"))

            plt.gca().add_patch(plt.Circle((0, 0), radius=obs.annulus[1],
                                           facecolor='none', fill=False, alpha=1,
                                           edgecolor='k', linestyle="dashed"))


            #go over ALL fibers for the fill color, but only add edge to work fibers
            signal_color = 'r'
            empty_color = 'y'
            for f in obs.fibers_all:
                # fiber absolute position ... need relative position to plot (so fiber - zero pos)
                fx, fy = empty_sci.get_position(f.ra, f.dec, cutout)

                xmin = min(xmin, fx - x)
                xmax = max(xmax, fx - x)
                ymin = min(ymin, fy - y)
                ymax = max(ymax, fy - y)

                #for now, set alpha as a fraction of a max line score? cap at 1.0
                if obs.eli_dict[f] is not None:
                    alpha = obs.eli_dict[f].line_score
                    if alpha is None:
                        alpha = 0.0
                    else:
                        alpha = min(MAX_ALPHA, (alpha / MAX_LINE_SCORE * MAX_ALPHA))
                else:
                    alpha = 0.0

                plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius,
                                               facecolor=signal_color, fill=True, alpha=alpha,
                                               edgecolor='none',linestyle=None))

                #over plot an edge to the fiber if it is a work fiber
                if f in obs.fibers_work:
                    plt.gca().add_patch(plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius,
                                                   facecolor='none', fill=True, alpha=0.5,
                                                   edgecolor=empty_color, linestyle="solid",linewidth=1))


        except:
            log.error("Unable to overplot gradient (all) fiber positions.", exc_info=True)

        #plot all fibers (with signal info for gradient fill but no edge color)
        #then (over) plot fibers_work with edge color
        #plt.tight_layout()
        plt.close()
        self.add_bid_entry(fig)

        return self.pages

    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="", base_count=0,
                                 target_w=0, fiber_locs=None,target_flux=None):
        #implement in child class
        pass


    def build_cat_summary_figure (self,ra,dec,error,bid_ras,bid_decs, target_w=0, fiber_locs=None, target_flux=None):
        #implement in child class
        pass


    def build_bid_target_figure_one_line (self,cat_match, ra, dec, error, df=None, df_photoz=None, target_ra=None,
                                          target_dec=None, section_title="", bid_number=1, target_w=0, of_number=0,
                                          target_flux=None, color="k"):
        # implement in child class
        pass

    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None):
        # implement in child class
        pass


    def is_edge_fiber(self,fiber_num):
        return fiber_num in G.CCD_EDGE_FIBERS_ALL


    def add_fiber_positions(self,plt,ra,dec,fiber_locs,error,ext,cutout):
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
                    plt.title("Fiber Positions")
                plt.xlabel("arcsecs")
                plt.gca().xaxis.labelpad = 0

                plt.plot(0, 0, "r+")

                xmin = float('inf')
                xmax = float('-inf')
                ymin = float('inf')
                ymax = float('-inf')

                x, y = empty_sci.get_position(ra, dec, cutout)  # zero (absolute) position

                for r, d, c, i, dist,fn in fiber_locs:
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

                    if self.is_edge_fiber(fn):
                        plt.gca().add_patch(
                            plt.Circle(((fx - x), (fy - y)), radius=G.Fiber_Radius+0.1, color=c, fill=False,
                                       linestyle='dashed'))

                # larger of the spread of the fibers or the maximum width (in non-rotated x-y plane) of the error window
                ext_base = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
                if ext_base != np.inf:
                    ext = max(ext_base + G.Fiber_Radius, ext)


                # need a new cutout since we rescaled the ext (and window) size
                cutout,_,_,_ = empty_sci.get_cutout(ra, dec, error, window=ext * 2, image=self.master_cutout)
                if cutout is None:
                    log.warning("Cannot obtain new cutout from master_cutout in cat_base::add_fiber_positions")#,exc_info=True)
                    cutout = self.master_cutout

                vmin, vmax = empty_sci.get_vrange(cutout.data)

                self.add_north_box(plt, empty_sci, cutout, error, 0, 0, theta=None)
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=vmin, vmax=vmax, extent=[-ext, ext, -ext, ext])


                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
            except:
                log.error("Unable to overplot fiber positions.",exc_info=True)



    def add_aperture_position(self,plt,radius,mag=None):
            # over plot a circle of radius on the center of the image (assumed to be the photo-aperture)
            if radius > 0:
                log.debug("Plotting imaging aperture position...")

                try:
                    plt.gca().add_patch(plt.Circle((0,0), radius=radius, color='gold', fill=False,
                                                   linestyle='solid'))

                    #temporary
                    if mag is not None:
                        plt.xlabel("mag = %0.1f, %0.1f\"" % (mag,radius))
                        plt.gca().xaxis.labelpad = 0
                        #plt.tight_layout()

                except:
                    log.error("Unable to overplot aperture position.",exc_info=True)


    def add_empty_catalog_fiber_positions(self, plt,fig,ra,dec,fiber_locs):
        '''used if there is no catalog. Just plot relative positions'''

        if fiber_locs is None: #there are no fiber locations (could be a specific RA,Dec search)
            return None

        plt.title("Relative Fiber Positions")
        plt.plot(0, 0, "r+")

        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')

        for r, d, c, i, dist, fn in fiber_locs:
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

        fig = plt.figure(figsize=(fig_sz_x, fig_sz_y))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)

        gs = gridspec.GridSpec(rows, cols, wspace=0.25, hspace=0.0)
        # reminder gridspec indexing is 0 based; matplotlib.subplot is 1-based

        font = FontProperties()
        font.set_family('monospace')
        font.set_size(12)

        # All on one line now across top of plots
        title = "No overlapping imaging catalog."

        plt.subplot(gs[0, :])
        plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
        plt.gca().set_frame_on(False)
        plt.gca().axis('off')

        plt.subplot(gs[2:, 0])
        self.add_empty_catalog_fiber_positions(plt, fig, ra, dec, fiber_locs)

        # complete the entry
        plt.close()
        return fig


    def get_stacked_cutout(self,ra,dec,window):
        #implement in child class
        pass

    # def write_cutout_as_fits(self,cutout,filename):
    #     '''write a cutout as a fits file'''
    #     pass
