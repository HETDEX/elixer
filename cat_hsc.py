from __future__ import print_function

import global_config as G
import os.path as op


import matplotlib
matplotlib.use('agg')

import pandas as pd
import science_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import astropy.table
import line_prob

#log = G.logging.getLogger('Cat_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('cat_logger')
log.setlevel(G.logging.DEBUG)

pd.options.mode.chained_assignment = None  #turn off warning about setting the distance field

import cat_base
import match_summary
import hsc_meta

def hsc_count_to_mag(count,cutout=None,sci_image=None):
   # return 999.9

    #We can convert the counts into flux
    # with a keyword in the header of the imaging data;
    # FLUXMAG0=     63095734448.0194
    #
    # Because the HSC pipeline uses the zeropoint value (corresponds to 27 mag) to all filters,
    # we can convert the counts values in the R-band imaging data as follows:
    # -2.5*log(flux_R) -48.6 = -2.5*log(count_R / FLUXMAG0)
    # --> flux_R = count_R / ( 10^(30.24) )

    #note: zero point is 27 mag
    if count is not None:
        if sci_image is not None:
            #get the conversion factor, each tile is different
            try:
                fluxmag0 = float(sci_image[0].header['FLUXMAG0'])
            except:
                fluxmag0 = 0.0
                log.error("Exception in hsc_count_to_mag",exc_info=True)
                return 99.9

        if count > 0:
            return -2.5 * np.log10(count/fluxmag0) #+ 48.6
        else:
            return 99.9  # need a better floor

class HSC(cat_base.Catalog):#Hyper Suprime Cam
    # class variables
    HSC_BASE_PATH = G.HSC_BASE_PATH
    HSC_CAT_PATH = G.HSC_CAT_PATH
    HSC_IMAGE_PATH = G.HSC_IMAGE_PATH

    CONT_EST_BASE = None

    df = None
    loaded_tracts = []

    MainCatalog = None #there is no Main Catalog ... must load individual catalog tracts
    Name = "HyperSuprimeCam"

    Image_Coord_Range = hsc_meta.Image_Coord_Range
    Tile_Dict = hsc_meta.HSC_META_DICT
    Filters = ['r'] #case is important ... needs to be lowercase
    Cat_Coord_Range = {'RA_min': None, 'RA_max': None, 'Dec_min': None, 'Dec_max': None}

    WCS_Manual = True

    AstroTable = None

    # 1 sourceID
    # 2 X on fits
    # 3 Y on fits
    # 4 RA
    # 5 Dec
    # 6 flux.psf
    # 7 flux.psf.err
    # 8 flux.psf.flags ("False" means no problems)
    # 9 mag.psf
    # 10 magerr.psf
    # 11 flux.kron
    # 12 flux.kron.err
    # 13 flux.kron.flags ("False" means no problems)
    # 14 mag.kron
    # 15 magerr.kron
    # 16 cmodel.flux
    # 17 cmodel.flux.err
    # 18 cmodel.flux.flags ("False" means no problems)
    # 19 cmodel.mag
    # 20 cmodel.magerr
    #
    #
    # SOME TRACTS below have no detections,
    # because these tract have almost no overlap with HSC pointings.
    # Please check the tract-pointing distributions at
    # /work/04094/mshiro/maverick/HSC/S15A/reduced/tract_pointing/tract_pointing.png
    #
    # --no detection tracts--
    # R_16174.dat
    # R_16645.dat
    # R_16805.dat
    # R_16963.dat
    # R_16964.dat
    # R_16965.dat
    # R_16969.dat
    # R_16972.dat
    # R_16974.dat

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
    def read_catalog(cls, catalog_loc=None, name=None,tract=None):

        if name is None:
            name = cls.Name

        if tract is not None:
            if tract in cls.loaded_tracts:
                log.info("Catalog tract (%s) already loaded." %tract)
                return cls.df

        if tract is None:
            log.error("Cannot load catalog tract for HSC. No tract provided.")
            return None


        #todo: future more than just the R filter if any are ever added
        cat_name = 'R_' + tract + ".dat"
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
            cls.loaded_tracts.append(tract)

        except:
            log.error(name + " Exception attempting to build pandas dataframe", exc_info=True)
            return None

        return cls.df

    def build_catalog_of_images(self):

        for t in self.Tile_Dict.keys(): #tile is the key (the filename)
            for f in self.Filters:
                self.CatalogImages.append(
                    {'path': self.HSC_IMAGE_PATH,
                     'name': t, #filename is the tilename
                     'tile': t,
                     'filter': f,
                     'instrument': "",
                     'cols': [],
                     'labels': [],
                     'image': None,
                     'expanded': False,
                     'wcs_manual': True,
                     'aperture': 1.0,
                     'mag_func': hsc_count_to_mag
                     })

    def find_target_tile(self,ra,dec):
        #assumed to have already confirmed this target is at least in coordinate range of this catalog
        tile = None
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
            return None
        elif len(keys) == 1: #found exactly one
            tile = keys[0]
        elif len(keys) > 1: #find the best one
            min = 9e9
            #we don't have the actual corners anymore, so just assume a rectangle
            #so there are 2 of each min, max coords. Only need the smallest distance so just sum one
            for k in keys:
                sqdist = (ra-self.Tile_Dict[k]['RA_min'])**2 + (dec-self.Tile_Dict[k]['Dec_min'])**2 + \
                         (ra-self.Tile_Dict[k]['RA_max'])**2 + (dec-self.Tile_Dict[k]['Dec_max'])**2
                if sqdist < min:
                    min = sqdist
                    tile = k
        else: #really?? len(keys) < 0 : this is just a sanity catch
            log.error("ERROR! len(keys) < 0 in cat_hsc::find_target_tile.")
            return None

        #now we have the tile key (filename)
        #do we want to find the matching catalog and see if there is an entry in it?

        #sanity check the image
        try:
            image = science_image.science_image(wcs_manual=self.WCS_Manual,wcs_idx=1,
                                                image_location=op.join(self.HSC_IMAGE_PATH,tile))
            if image.contains_position(ra, dec):
                pass
            else:
                log.debug("position (%f, %f) is not in image. %s" % (ra, dec,tile))
                tile = None
        except:
            pass

        return tile

    def build_list_of_bid_targets(self, ra, dec, error):
        '''ra and dec in decimal degrees. error in arcsec.
        returns a pandas dataframe'''

        #even if not None, could be we need a different catalog, so check and append
        tile = self.find_target_tile(ra,dec)

        if tile is None:
            log.info("Could not locate tile for HSC. Discontinuing search of this catalog.")
            return 0,None,None

        #could be none or could be not loaded yet
        if self.df is None or not (self.Tile_Dict[tile]['tract'] in self.loaded_tracts):
            #self.read_main_catalog()
            self.read_catalog(tract=self.Tile_Dict[tile]['tract'])

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
                self.df[(self.df['RA'] >= ra_min) & (self.df['RA'] <= ra_max) &
                        (self.df['DEC'] >= dec_min) & (self.df['DEC'] <= dec_max)].copy()
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
                self.dataframe_of_bid_targets_unique.drop_duplicates(subset=['RA','DEC'])
            self.num_targets = self.dataframe_of_bid_targets_unique.iloc[:,0].count()

        except:
            log.error(self.Name + " Exception in build_list_of_bid_targets", exc_info=True)

        if self.dataframe_of_bid_targets is not None:
            #self.num_targets = self.dataframe_of_bid_targets.iloc[:, 0].count()
            self.sort_bid_targets_by_likelihood(ra, dec)

            log.info(self.Name + " searching for objects in [%f - %f, %f - %f] " % (ra_min, ra_max, dec_min, dec_max) +
                     ". Found = %d" % (self.num_targets))

        return self.num_targets, self.dataframe_of_bid_targets, None



    def build_bid_target_reports(self, cat_match, target_ra, target_dec, error, num_hits=0, section_title="",
                                 base_count=0,
                                 target_w=0, fiber_locs=None, target_flux=None):

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
                                                  target_w=target_w, fiber_locs=fiber_locs, target_flux=target_flux)
        else:
            log.error("ERROR!!! Unexpected state of G.SINGLE_PAGE_PER_DETECT")


        if entry is not None:
            self.add_bid_entry(entry)

        if G.SINGLE_PAGE_PER_DETECT: # and (len(ras) <= G.MAX_COMBINE_BID_TARGETS):
            entry = self.build_multiple_bid_target_figures_one_line(cat_match, ras, decs, error,
                                                                    target_ra=target_ra, target_dec=target_dec,
                                                                    target_w=target_w, target_flux=target_flux)
            if entry is not None:
                self.add_bid_entry(entry)
#        else:  # each bid taget gets its own line
        if (not G.FORCE_SINGLE_PAGE) and (len(ras) > G.MAX_COMBINE_BID_TARGETS):  # each bid taget gets its own line
            log.error("ERROR!!! Unexpected state of G.FORCE_SINGLE_PAGE")

        return self.pages


    def build_cat_summary_figure (self, cat_match, ra, dec, error,bid_ras, bid_decs, target_w=0,
                                  fiber_locs=None, target_flux=None):
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
        tile = self.find_target_tile(ra, dec)

        if tile is None:
            # problem
            print("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            log.error("No appropriate tile found in SHELA for RA,DEC = [%f,%f]" % (ra, dec))
            return None

        # All on one line now across top of plots
        if G.ZOO:
            title = "Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets), error)
        else:
            title = self.Name + " : Possible Matches = %d (within +/- %g\")" \
                    % (len(self.dataframe_of_bid_targets), error)

        title += "  Minimum (no match) 3$\sigma$ rest-EW: "
        cont_est  = -1
        if target_flux  and self.CONT_EST_BASE:
            cont_est = self.CONT_EST_BASE*3
            if cont_est != -1:
                title += "  LyA = %g $\AA$ " % ((target_flux / cont_est) / (target_w / G.LyA_rest))
                if target_w >= G.OII_rest:
                    title = title + "  OII = %g $\AA$" % ((target_flux / cont_est) / (target_w / G.OII_rest))
                else:
                    title = title + "  OII = N/A"
            else:
                title += "  LyA = N/A  OII = N/A"
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
                i['image'] = science_image.science_image(wcs_manual=wcs_manual,wcs_idx=1,
                                                         image_location=op.join(i['path'], i['name']))
            sci = i['image']

            #the filters are in order, use r if g is not there
            if (f == 'r') and (sci.exptime is not None) and (exptime_cont_est == -1):
                exptime_cont_est = sci.exptime

            # the filters are in order, so this will overwrite r
            if (f == 'g') and (sci.exptime is not None):
                exptime_cont_est = sci.exptime

            # sci.load_image(wcs_manual=True)
            cutout, pix_counts, mag = sci.get_cutout(ra, dec, error, window=window,
                                                     aperture=aperture,mag_func=mag_func)
            ext = sci.window / 2.  # extent is from the 0,0 center, so window/2

            try:  # update non-matched source line with PLAE()
                if ((mag < 99) or (cont_est != -1)) and (target_flux is not None) and (i['filter'] == 'g'):
                    # make a "blank" catalog match (e.g. at this specific RA, Dec (not actually from catalog)
                    bid_target = match_summary.BidTarget()
                    bid_target.catalog_name = self.Name
                    bid_target.bid_ra = 666  # nonsense RA
                    bid_target.bid_dec = 666  # nonsense Dec
                    bid_target.distance = 0.0
                    if mag < 99:
                        bid_target.bid_flux_est_cgs = self.obs_mag_to_cgs_flux(mag, target_w)
                    else:
                        bid_target.bid_flux_est_cgs = cont_est

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

                    bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = \
                        line_prob.prob_LAE(wl_obs=target_w, lineFlux=target_flux,
                                           ew_obs=(target_flux / bid_target.bid_flux_est_cgs),
                                           c_obs=None, which_color=None, addl_wavelengths=addl_waves,
                                           addl_fluxes=addl_flux,addl_errors=addl_ferr, sky_area=None,
                                           cosmo=None, lae_priors=None, ew_case=None, W_0=None, z_OII=None,
                                           sigma=None)

                    if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                        text.set_text(text.get_text() + "  P(LAE)/P(OII) = %0.3g" % (bid_target.p_lae_oii_ratio))

                    cat_match.add_bid_target(bid_target)
            except:
                log.debug('Could not build exact location photometry info.', exc_info=True)

            # 1st cutout might not be what we want for the master (could be a summary image from elsewhere)
            if self.master_cutout:
                if self.master_cutout.shape != cutout.shape:
                    del self.master_cutout
                    self.master_cutout = None

            if cutout is not None:  # construct master cutout
                # master cutout needs a copy of the data since it is going to be modified  (stacked)
                # repeat the cutout call, but get a copy
                if self.master_cutout is None:
                    self.master_cutout,_,_ = sci.get_cutout(ra, dec, error, window=window, copy=True)
                    if sci.exptime:
                        ref_exptime = sci.exptime
                    total_adjusted_exptime = 1.0
                else:
                    try:
                        self.master_cutout.data = np.add(self.master_cutout.data, cutout.data * sci.exptime / ref_exptime)
                        total_adjusted_exptime += sci.exptime / ref_exptime
                    except:
                        log.warn("Unexpected exception.", exc_info=True)

                plt.subplot(gs[1:, index])
                plt.imshow(cutout.data, origin='lower', interpolation='none', cmap=plt.get_cmap('gray_r'),
                           vmin=sci.vmin, vmax=sci.vmax, extent=[-ext, ext, -ext, ext])
                plt.title(i['instrument'] + " " + i['filter'])
                plt.xticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.yticks([int(ext), int(ext / 2.), 0, int(-ext / 2.), int(-ext)])
                plt.plot(0, 0, "r+")

                if pix_counts is not None:
                    self.add_aperture_position(plt,aperture,mag)

                self.add_north_box(plt, sci, cutout, error, 0, 0, theta=None)
                x, y = sci.get_position(ra, dec, cutout)  # zero (absolute) position
                for br, bd, bc in zip(bid_ras, bid_decs, bid_colors):
                    fx, fy = sci.get_position(br, bd, cutout)
                    plt.gca().add_patch(plt.Rectangle(((fx - x) - target_box_side / 2.0, (fy - y) - target_box_side / 2.0),
                                                      width=target_box_side, height=target_box_side,
                                                      angle=0.0, color=bc, fill=False, linewidth=1.0, zorder=1))



        #if False:
        #    if target_flux is not None:
        #        #todo: get exptime from the tile (science image has it)
        #        cont_est = self.get_f606w_max_cont(exptime_cont_est, 3,self.CONT_EST_BASE)
        #        if cont_est != -1:
        #            title += "Minimum (no match)\n  3$\sigma$ rest-EW:\n"
        #            title += "  LyA = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.LyA_rest))
        #            if target_w >= G.OII_rest:
        #                title = title + "  OII = %g $\AA$\n" %  ((target_flux / cont_est) / (target_w / G.OII_rest))
        #            else:
        #                title = title + "  OII = N/A\n"

            #plt.subplot(gs[0, 0])
            #plt.text(0, 0.3, title, ha='left', va='bottom', fontproperties=font)
            #plt.gca().set_frame_on(False)
            #plt.gca().axis('off')

        if self.master_cutout is None:
            # cannot continue
            print("No catalog image available in %s" % self.Name)
            plt.close()
            # still need to plot relative fiber positions here
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


    def build_multiple_bid_target_figures_one_line(self, cat_match, ras, decs, error, target_ra=None, target_dec=None,
                                         target_w=0, target_flux=None):

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
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "Est OII rest-EW\n"
        else:
            text = "Separation\n" + \
                   "RA, Dec\n" + \
                   "Spec z\n" + \
                   "Photo z\n" + \
                   "Est LyA rest-EW\n" + \
                   "Est OII rest-EW\n" + \
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
            try:
                df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'r')]
                if df is None:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                       (self.dataframe_of_bid_targets['DEC'] == d[0]) &
                                                       (self.dataframe_of_bid_targets['FILTER'] == 'g')]
                if df is None:
                    df = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                        (self.dataframe_of_bid_targets['DEC'] == d[0])]

            except:
                log.error("Exception attempting to find object in dataframe_of_bid_targets", exc_info=True)
                continue  # this must be here, so skip to next ra,dec

            if df is not None:
                text = ""

                if G.ZOO:
                    text = text + "%g\"\n" \
                                  % (df['distance'].values[0] * 3600)
                else:
                    text = text + "%g\"\n%f, %f\n" \
                                % ( df['distance'].values[0] * 3600,df['RA'].values[0], df['DEC'].values[0])


                best_fit_photo_z = 0.0
                try:
                    best_fit_photo_z = float(df['Best-fit photo-z'].values[0])
                except:
                    pass

                text += "N/A\n" + "%g\n" % best_fit_photo_z

                filter_fl = 0.0
                filter_fl_err = 0.0

                #todo: add flux (cont est)
                try:
                    if (df['FILTER'].values[0] in 'rg'):
                      filter_fl = float(df['FLUX_AUTO'].values[0])  #?? in nano-jansky or 1e-32  erg s^-1 cm^-2 Hz^-2
                      filter_fl_err = float(df['FLUXERR_AUTO'].values[0])
                except:
                    filter_fl = 0.0
                    filter_fl_err = 0.0

                bid_target = None

                if (target_flux is not None) and (filter_fl != 0.0):
                    if (filter_fl is not None):# and (filter_fl > 0):
                        filter_fl_cgs = self.nano_jansky_to_cgs(filter_fl,target_w) #filter_fl * 1e-32 * 3e18 / (target_w ** 2)  # 3e18 ~ c in angstroms/sec
                        text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.LyA_rest))

                        if target_w >= G.OII_rest:
                            text = text + "%g $\AA$\n" % (target_flux / filter_fl_cgs / (target_w / G.OII_rest))
                        else:
                            text = text + "N/A\n"
                        try:
                            bid_target = match_summary.BidTarget()
                            bid_target.catalog_name = self.Name
                            bid_target.bid_ra = df['RA'].values[0]
                            bid_target.bid_dec = df['DEC'].values[0]
                            bid_target.distance = df['distance'].values[0] * 3600
                            bid_target.bid_flux_est_cgs = filter_fl

                            addl_waves = None
                            addl_flux = None
                            addl_ferr = None
                            try:
                                addl_waves = cat_match.detobj.spec_obj.addl_wavelengths
                                addl_flux = cat_match.detobj.spec_obj.addl_fluxes
                                addl_ferr = cat_match.detobj.spec_obj.addl_fluxerrs
                            except:
                                pass

                            bid_target.p_lae_oii_ratio, bid_target.p_lae, bid_target.p_oii = line_prob.prob_LAE(wl_obs=target_w,
                                                                                           lineFlux=target_flux,
                                                                                           ew_obs=(
                                                                                           target_flux / filter_fl_cgs),
                                                                                           c_obs=None, which_color=None,
                                                                                           addl_wavelengths=addl_waves,
                                                                                           addl_fluxes=addl_flux,
                                                                                           addl_errors=addl_ferr,
                                                                                           sky_area=None, cosmo=None,
                                                                                           lae_priors=None,
                                                                                           ew_case=None, W_0=None,
                                                                                           z_OII=None, sigma=None)

                            dfx = self.dataframe_of_bid_targets.loc[(self.dataframe_of_bid_targets['RA'] == r[0]) &
                                                                    (self.dataframe_of_bid_targets['DEC'] == d[0])]

                            for flt,flux,err in zip(dfx['FILTER'].values,
                                                    dfx['FLUX_AUTO'].values,
                                                    dfx['FLUXERR_AUTO'].values):
                                try:
                                    bid_target.add_filter('NA',flt,
                                                          self.nano_jansky_to_cgs(flux,target_w),
                                                          self.nano_jansky_to_cgs(err,target_w))
                                except:
                                    log.debug('Unable to build filter entry for bid_target.',exc_info=True)

                            cat_match.add_bid_target(bid_target)
                        except:
                            log.debug('Unable to build bid_target.',exc_info=True)


                else:
                    text += "N/A\nN/A\n"

                #todo: add flux (cont est)
               # if filter_fl < 0:
               #     text = text + "%g(%g) nJy !?\n" % (filter_fl, filter_fl_err)
               # else:
               #     text = text + "%g(%g) nJy\n" % (filter_fl, filter_fl_err)

                if (not G.ZOO) and (bid_target is not None) and (bid_target.p_lae_oii_ratio is not None):
                    text += "%0.3g\n" % (bid_target.p_lae_oii_ratio)
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
