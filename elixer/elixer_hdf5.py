"""
ELiXer HDF5 utilities ...
create ELiXer catalog(s) as HDF5
merge existing ELiXer catalogs
"""


__version__ = '0.6.1' #catalog version ... can merge if major and minor version numbers are the same or in special circumstances

try:
    from elixer import hetdex
    from elixer import match_summary
    from elixer import global_config as G
except:
    import hetdex
    import match_summary
    import global_config as G

import numpy as np
import tables
import os
import time


UNSET_FLOAT = -999.999
UNSET_INT = -99999
UNSET_STR = ""
UNSET_NAN = np.nan

log = G.Global_Logger('hdf5_logger')
log.setlevel(G.LOG_LEVEL)


#make a class for each table
class Version(tables.IsDescription):
    #version table, very basic info
    version = tables.StringCol(itemsize=16, dflt='',pos=0)
    version_pytables = tables.StringCol(itemsize=16, dflt='',pos=1)


class Detections(tables.IsDescription):
#top level detections summary, one row for each ELiXer/HETDEX detection
    detectid = tables.Int64Col(pos=0) #unique HETDEX detection ID 1e9+
    detectname = tables.StringCol(itemsize=64,pos=1)

    elixer_version = tables.StringCol(itemsize=16,pos=2) #version of elixer that generated this detection report
    elixer_datetime = tables.StringCol(itemsize=21,pos=3) #YYYY-MM-DD hh:mm:ss

    #of the primary fiber ... typically, just three dithers and
    #all from the same observation, so this would apply to all but
    #it can be that this is built up from observations over different date/times
    #this is mostly or entirely redundant with HETDEX HDF5 data (detections or survey)
    shotid = tables.Int64Col()
    obsid = tables.Int32Col()
    specid = tables.StringCol(itemsize=3)
    ifuslot = tables.StringCol(itemsize=3)
    ifuid = tables.StringCol(itemsize=3)
    if G.HDR_Version_float == 1:
        seeing_gaussian = tables.Float32Col(dflt=UNSET_FLOAT)
        seeing_moffat = tables.Float32Col(dflt=UNSET_FLOAT)
    else:
        seeing_fwhm = tables.Float32Col(dflt=UNSET_FLOAT)
    response = tables.Float32Col(dflt=UNSET_FLOAT)
    fieldname = tables.StringCol(itemsize=32)

    #about the detection
    ra = tables.Float32Col(dflt=UNSET_FLOAT,pos=4)
    dec = tables.Float32Col(dflt=UNSET_FLOAT,pos=5)
    wavelength_obs = tables.Float32Col(dflt=UNSET_FLOAT,pos=6)
    wavelength_obs_err = tables.Float32Col(dflt=UNSET_FLOAT,pos=7)

    best_z = tables.Float32Col(dflt=-1.0,pos=8)
    best_pz = tables.Float32Col(dflt=0.0,pos=9)

    flags = tables.Int32Col(dflt=0,pos=10)
    review = tables.Int8Col(dflt=0,pos=11)
    cluster_parent = tables.Int64Col(dflt=0,pos=12)

    flux_line = tables.Float32Col(dflt=UNSET_FLOAT) #actual flux not flux density
    flux_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    fwhm_line_aa = tables.Float32Col(dflt=UNSET_FLOAT)
    fwhm_line_aa_err = tables.Float32Col(dflt=UNSET_FLOAT)
    sn = tables.Float32Col(dflt=UNSET_FLOAT)
    sn_err = tables.Float32Col(dflt=UNSET_FLOAT)
    chi2 = tables.Float32Col(dflt=UNSET_FLOAT)
    chi2_err = tables.Float32Col(dflt=UNSET_FLOAT)

    continuum_line = tables.Float32Col(dflt=UNSET_FLOAT) #continuum (y-offset) from Gaussian fit to the line
    continuum_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_line = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_line = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_line_max = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_line_min = tables.Float32Col(dflt=UNSET_FLOAT)

    #new
    continuum_wide = tables.Float32Col(dflt=UNSET_FLOAT) #continuum from (mostly) the full spectrum width, from pseudo g-band magnitude
    continuum_wide_err = tables.Float32Col(dflt=UNSET_FLOAT)
    mag_g_wide = tables.Float32Col(dflt=UNSET_FLOAT) #the pseudo g-band magnitude from the spectrum
    mag_g_wide_err = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_wide = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_wide_err = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_wide = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_wide_max = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_wide_min = tables.Float32Col(dflt=UNSET_FLOAT)

    #more new
    continuum_masked = tables.Float32Col(dflt=UNSET_FLOAT) #continuum from (mostly) the full spectrum width, from pseudo g-band magnitude
    continuum_masked_err = tables.Float32Col(dflt=UNSET_FLOAT)


    # continuum_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # continuum_sdss_g_err = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # mag_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # mag_sdss_g_err = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # eqw_rest_lya_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # eqw_rest_lya_sdss_g_err = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # plae_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # plae_sdss_g_max = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # plae_sdss_g_min = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    #
    #
    # # from the full width spectrum (but not passed through SDSS filter)
    # continuum_full_spec = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # continuum_full_spec_err = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # mag_full_spec = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # mag_full_spec_err = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # eqw_rest_lya_full_spec = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # eqw_rest_lya_full_spec_err = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # plae_full_spec = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # plae_full_spec_max = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")
    # plae_full_spec_min = tables.Float32Col(dflt=UNSET_FLOAT) #todo: remove (replace with a "best")


    #ELiXer solution based on extra lines
    multiline_flag = tables.BoolCol(dflt=False) #True if s a single "good" solution
    multiline_z = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_rest_w = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_prob = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_raw_score = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_frac_score = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_name = tables.StringCol(itemsize=16)

    # pseudo_color = tables.Float32Col(dflt=UNSET_FLOAT)
    # pseudo_color_min = tables.Float32Col(dflt=UNSET_FLOAT)
    # pseudo_color_max = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_flag = tables.Int64Col(dflt=0)

    pseudo_color_blue_flux = tables.Float32Col(dflt=UNSET_FLOAT) #all un uJy
    pseudo_color_blue_flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_red_flux = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_red_flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_rvb_ratio = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_rvb_ratio_err = tables.Float32Col(dflt=UNSET_FLOAT)

    #ELiXer combined (rules, inv variance, weights and Bayes) classification info
    combined_plae = tables.Float32Col(dflt=UNSET_FLOAT)   #combination of all PLAE/POII
    #combined_plae_err = tables.Float32Col(dflt=UNSET_FLOAT)

    combined_plae_lo = tables.Float32Col(dflt=UNSET_FLOAT)
    combined_plae_hi = tables.Float32Col(dflt=UNSET_FLOAT)
    plya_classification = tables.Float32Col(dflt=UNSET_FLOAT) #final, combine P(LAE) (0.0 - 1.0) #fomerly plae_classification

    spurious_reason = tables.StringCol(itemsize=32,dflt=UNSET_STR)
    combined_continuum = tables.Float32Col(dflt=UNSET_FLOAT)   #combination of all continuum estimates
    combined_continuum_err = tables.Float32Col(dflt=UNSET_FLOAT)

    combined_eqw_rest_lya = tables.Float32Col(dflt=UNSET_FLOAT)
    combined_eqw_rest_lya_err = tables.Float32Col(dflt=UNSET_FLOAT)

    spectral_slope = tables.Float32Col(dflt=UNSET_FLOAT)
    spectral_slope_err = tables.Float32Col(dflt=0.0)

    ccd_adjacent_mag = tables.Float32Col(dflt=99.9) #ccd adjacent, single fiber brightest mag
    central_single_fiber_mag = tables.Float32Col(dflt=99.9) #ccd adjacent, single fiber brightest mag
    ffsky_subtraction = tables.BoolCol(dflt=False) #if true, this detection used the full-field sky subtraction

    #added version 0.2.4
    classification_labels = tables.StringCol(itemsize=32,dflt="")

    #add filter colors
    color_ug = tables.Float32Col(shape=(3,),dflt=[np.nan,np.nan,np.nan] ) #as color, blue max, red_max
    color_ur = tables.Float32Col(shape=(3,),dflt=[np.nan,np.nan,np.nan] ) #as color, blue max, red_max
    color_gr = tables.Float32Col(shape=(3,),dflt=[np.nan,np.nan,np.nan] ) #as color, blue max, red_max

class SpectraLines(tables.IsDescription):
    detectid = tables.Int64Col(pos=0)  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(dflt=UNSET_FLOAT,pos=1)
    type = tables.Int32Col(dflt=UNSET_INT,pos=2) # 1 = emission, 0 = unknown, -1 = absorbtion
    score = tables.Float32Col(dflt=UNSET_FLOAT,pos=3)
    sn = tables.Float32Col(dflt=UNSET_FLOAT,pos=4)
    chi2 = tables.Float32Col(dflt=UNSET_FLOAT,pos=5)
    flux_line = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    used = tables.BoolCol(dflt=False) #True if used in the reported multiline solution
    sol_num = tables.Int32Col(dflt=-1) #-1 is unset, 0 is simple scan, no solution, 1+ = solution number in
                                       #decreasing score order


class ClassificationExtraFeatures(tables.IsDescription):
    """
    One per detectid
    Debugging table with specific P(LyA) votes, (though not all votes)
    """
    detectid = tables.Int64Col(pos=0)
    #already have PLAE/POII aggregrate high and low
    plae_poii_combined = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_poii_combined_hi = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_poii_combined_lo = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_poii_combined_midpoint = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_poii_combined_vote = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_poii_combined_weight = tables.Float32Col(dflt=UNSET_FLOAT)

    ew_rest_lya_combined = tables.Float32Col(dflt=UNSET_FLOAT)
    ew_rest_lya_combined_err = tables.Float32Col(dflt=UNSET_FLOAT)
    ew_rest_lya_combined_vote = tables.Float32Col(dflt=UNSET_FLOAT)
    ew_rest_lya_combined_weight = tables.Float32Col(dflt=UNSET_FLOAT)

    dex_gmag = tables.Float32Col(dflt=UNSET_FLOAT)
    dex_gmag_bright = tables.Float32Col(dflt=UNSET_FLOAT)
    dex_gmag_faint = tables.Float32Col(dflt=UNSET_FLOAT)
    gmag_thresh_bright = tables.Float32Col(dflt=UNSET_FLOAT)
    gmag_thresh_faint = tables.Float32Col(dflt=UNSET_FLOAT)
    dex_gmag_vote = tables.Float32Col(dflt=UNSET_FLOAT)
    dex_gmag_weight = tables.Float32Col(dflt=UNSET_FLOAT)

    size_in_psf = tables.Float32Col(dflt=UNSET_FLOAT)
    diam_in_arcsec = tables.Float32Col(dflt=UNSET_FLOAT)
    oii_size_in_kpc = tables.Float32Col(dflt=UNSET_FLOAT)
    lya_size_in_kpc = tables.Float32Col(dflt=UNSET_FLOAT)
    size_in_psf_vote = tables.Float32Col(dflt=UNSET_FLOAT)
    size_in_psf_weight = tables.Float32Col(dflt=UNSET_FLOAT)

    rb_flux_asym = tables.Float32Col(dflt=UNSET_FLOAT)
    rb_flux_asym_err = tables.Float32Col(dflt=UNSET_FLOAT)
    rb_flux_asym_vote = tables.Float32Col(dflt=UNSET_FLOAT)
    rb_flux_asym_weight = tables.Float32Col(dflt=UNSET_FLOAT)

    line_sigma = tables.Float32Col(dflt=UNSET_FLOAT)
    line_sigma_err = tables.Float32Col(dflt=UNSET_FLOAT)
    line_sigma_vote = tables.Float32Col(dflt=UNSET_FLOAT)
    line_sigma_weight = tables.Float32Col(dflt=UNSET_FLOAT)





#Only used when G.LyC is True ... special table for Lyman Continuum project
#there is some duplicated data, but this is meant to be self contained
class NeighborSpectra(tables.IsDescription):

    detectid = tables.Int64Col(pos=0)
    candidate_ra = tables.Float32Col(pos=1,dflt=UNSET_FLOAT) #was aperture_ra
    candidate_dec = tables.Float32Col(pos=2,dflt=UNSET_FLOAT) #was aperture_dec

    shotid = tables.Int64Col(pos=3)
    seeing_fwhm = tables.Float32Col(pos=4,dflt=UNSET_FLOAT)
    neighbor_ra = tables.Float32Col(pos=5,dflt=UNSET_FLOAT) #was aperture_ra
    neighbor_dec = tables.Float32Col(pos=6,dflt=UNSET_FLOAT) #was aperture_dec

    dist_curve = tables.Float32Col(pos=7,dflt=UNSET_FLOAT)
    dist_baryctr = tables.Float32Col(pos=8,dflt=UNSET_FLOAT)
    major = tables.Float32Col(pos=9,dflt=UNSET_FLOAT) #major axis (diameter) 'a' in arcsec
    minor = tables.Float32Col(pos=10,dflt=UNSET_FLOAT) #'b'
    theta = tables.Float32Col(pos=11,dflt=0.0) #radians counter-clockwise from x-axis

    seeing_size = tables.Float32Col(pos=12,dflt=0.0) #major / seeing fwhm ... unresolved is < 1 (makes for easier query)
    seeing_dist = tables.Float32Col(pos=13,dflt=0.0) #dist_curve / seeing fwhm   ... only meaningful if resovled (again for easy query)

    sky_type = tables.Int32Col(pos=14,dflt=UNSET_INT) # local vs fullfield, but leave room for other type
    aperture_radius = tables.Float32Col(pos=15,dflt=UNSET_FLOAT)

    filter = tables.StringCol(pos=16,itemsize=16)
    mag = tables.Float32Col(pos=17,dflt=UNSET_FLOAT)
    mag_err = tables.Float32Col(pos=18,dflt=UNSET_FLOAT)
    mag_limit = tables.Float32Col(pos=19,dflt=UNSET_FLOAT)

    dex_g_mag = tables.Float32Col(pos=20,dflt=UNSET_FLOAT)
    dex_g_mag_err = tables.Float32Col(pos=21,dflt=UNSET_FLOAT)

    # wavelength = tables.Float32Col(pos=22,shape=(1036,)) #fixed wavelength, so just use the one from another table
    spec_type = tables.Int32Col(pos=22,dflt=UNSET_INT) # 0 = not populated, 1 = Flux density (cgs), 2 = Flux (cgs)
    spec_scale = tables.Int32Col(pos=23,dflt=UNSET_INT) # i.e. -17 = 1e-17 ., 0 == in zeroed units , so values like few x 10^-17
    spec_val = tables.Float32Col(pos=24,shape=(1036,)) #
    spec_err = tables.Float32Col(pos=25,shape=(1036,))

    # flat_flux = tables.Float32Col(pos=25,shape=(1036,)) #lets just calculate this on the fly and not waste the space
    # flat_flux_err = tables.Float32Col(pos=26,shape=(1036,))


class CalibratedSpectra(tables.IsDescription):
    detectid = tables.Int64Col(pos=0)  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(shape=(1036,),pos=1)
    flux = tables.Float32Col(shape=(1036,),pos=2 )
    flux_err = tables.Float32Col(shape=(1036,),pos=3)
    aperture_radius = tables.Float32Col(dflt=UNSET_FLOAT,pos=4)
    sky_background = tables.Int32Col(dflt=UNSET_INT,pos=5)
    num_fibers = tables.Int32Col(dflt=UNSET_INT,pos=6)

class Aperture(tables.IsDescription):
    #one entry per aperture photometry collected
    detectid = tables.Int64Col(pos=0)
    ra = tables.Float32Col(pos=1,dflt=UNSET_FLOAT) #was aperture_ra
    dec = tables.Float32Col(pos=2,dflt=UNSET_FLOAT) #was aperture_dec
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16)
    image_depth_mag = tables.Float32Col(dflt=99.9)
    pixel_scale = tables.Float32Col(dflt=UNSET_FLOAT)
    radius = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec , #was aperture_radius
    mag = tables.Float32Col(dflt=UNSET_FLOAT) #was aperture_mag
    mag_err = tables.Float32Col(dflt=UNSET_FLOAT) #was  aperture_mag_err
    aperture_area_pix = tables.Float32Col(dflt=UNSET_FLOAT) #pixels
    sky_area_pix = tables.Float32Col(dflt=UNSET_FLOAT) #pixels
    eqw_rest_lya = tables.Float32Col(dflt=UNSET_FLOAT) #was  aperture_eqw_rest_lya
    eqw_rest_lya_err = tables.Float32Col(dflt=UNSET_FLOAT) #was  aperture_eqw_rest_lya_err
    plae = tables.Float32Col(dflt=UNSET_NAN) #was  aperture_plae
    plae_max = tables.Float32Col(dflt=UNSET_NAN) #was  aperture_plae_max
    plae_min = tables.Float32Col(dflt=UNSET_NAN) #was  aperture_plae_min
    aperture_cts = tables.Float32Col(dflt=UNSET_FLOAT) #was aperture_counts
    sky_cts = tables.Float32Col(dflt=UNSET_FLOAT)
    sky_average = tables.Float32Col(dflt=UNSET_FLOAT)

class ElixerApertures(tables.IsDescription):
    #one entry per aperture photometry collected
    detectid = tables.Int64Col(pos=0)
    ra = tables.Float32Col(pos=1,dflt=UNSET_FLOAT) #decimal degrees of center
    dec = tables.Float32Col(pos=2,dflt=UNSET_FLOAT)
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16)
    image_depth_mag = tables.Float32Col(dflt=99.9)
    pixel_scale = tables.Float32Col(dflt=UNSET_FLOAT) #arcsec/pixel
    selected = tables.BoolCol(dflt=False) #if True this is the object used for the aperture PLAE/OII, etc (see above table)
    radius = tables.Float32Col(dflt=UNSET_FLOAT) #major axis (diameter) 'a' in arcsec
    mag = tables.Float32Col(dflt=UNSET_FLOAT)
    mag_err = tables.Float32Col(dflt=UNSET_FLOAT)

    # sky_total_cts = tables.Float32Col(dflt=UNSET_FLOAT) #sky_counts
    # sky_total_pix = tables.Float32Col(dflt=UNSET_FLOAT) #sky_average
    # sky_cts = tables.Float32Col(dflt=UNSET_FLOAT) #sky_average
    # sky_err = tables.Float32Col(dflt=UNSET_FLOAT)
    # aperture_cts = tables.Float32Col(dflt=UNSET_FLOAT) #aperture_counts
    # aperture_cts_err = tables.Float32Col(dflt=UNSET_FLOAT)  #
    #
    #
    background_cts = tables.Float32Col(dflt=UNSET_FLOAT) #sky_average
    background_err = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_cts = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    flags = tables.Int32Col(dflt=0) #aperture flags
    image_flags = tables.Int64Col(dflt=0) #separate from the aperture flags, these are ties to the image reduction pipeline


class ExtractedObjects(tables.IsDescription):
    #one entry per aperture photometry collected
    detectid = tables.Int64Col(pos=0)
    ra = tables.Float32Col(pos=1,dflt=UNSET_FLOAT) #decimal degrees of center
    dec = tables.Float32Col(pos=2,dflt=UNSET_FLOAT)
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16)
    image_depth_mag = tables.Float32Col(dflt=99.9)
    pixel_scale = tables.Float32Col(dflt=UNSET_FLOAT) #arcsec/pixel
    selected = tables.BoolCol(dflt=False) #if True this is the object used for the aperture PLAE/OII, etc (see above table)
    major = tables.Float32Col(dflt=UNSET_FLOAT) #major axis (diameter) 'a' in arcsec
    minor = tables.Float32Col(dflt=UNSET_FLOAT) #'b'
    theta = tables.Float32Col(dflt=0.0) #radians counter-clockwise from x-axis
    mag = tables.Float32Col(dflt=UNSET_FLOAT)
    mag_err = tables.Float32Col(dflt=UNSET_FLOAT)
    background_cts = tables.Float32Col(dflt=UNSET_FLOAT)
    background_err = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_cts = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    flags = tables.Int32Col(dflt=0)
    dist_curve = tables.Float32Col(dflt=UNSET_FLOAT)
    dist_baryctr = tables.Float32Col(dflt=UNSET_FLOAT)
    image_flags = tables.Int64Col(dflt=0) #separate from the aperture flags, these are ties to the image reduction pipeline

    fixed_aper_radius = tables.Float32Col(dflt=UNSET_FLOAT)
    fixed_aper_mag = tables.Float32Col(dflt=UNSET_FLOAT)
    fixed_aper_mag_err = tables.Float32Col(dflt=UNSET_FLOAT)
    fixed_aper_flux_cts = tables.Float32Col(dflt=UNSET_FLOAT)
    fixed_aper_flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    fixed_aper_flags = tables.Int32Col(dflt=0)

    # flag ... bit mask
    # 01 sep.OBJ_MERGED	      object is result of deblending
    # 02 sep.OBJ_TRUNC	      object is truncated at image boundary
    # 08 sep.OBJ_SINGU	      x, y fully correlated in object
    # 10 sep.APER_TRUNC	      aperture truncated at image boundary
    # 20 sep.APER_HASMASKED	  aperture contains one or more masked pixels
    # 40 sep.APER_ALLMASKED	  aperture contains only masked pixels
    # 80 sep.APER_NONPOSITIVE aperture sum is negative in kron_radius



class CatalogMatch(tables.IsDescription):
    # one entry per catalog bid target
    detectid = tables.Int64Col(pos=0)
    ra = tables.Float32Col(pos=1,dflt=UNSET_FLOAT) #was cat_ra
    dec = tables.Float32Col(pos=2,dflt=UNSET_FLOAT) #was cat_dec
    selected = tables.BoolCol(pos=3,dflt=False)
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16)
    match_num = tables.Int32Col(dflt=-1)
    separation = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec
    prob_match = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec
    specz = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_specz
    photz = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_photz
    flux = tables.Float32Col(dflt=UNSET_FLOAT) #was  cat_flux
    flux_err = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_flux_err
    mag = tables.Float32Col(dflt=UNSET_FLOAT) #was  cat_mag
    mag_err = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_mag_err
    eqw_rest_lya = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_eqw_rest_lya
    eqw_rest_lya_err = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_eqw_rest_lya_err
    plae = tables.Float32Col(dflt=UNSET_FLOAT) #was  cat_plae
    plae_max = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_plae_max
    plae_min = tables.Float32Col(dflt=UNSET_FLOAT) #was cat_plae_min

    #maybe add in the PDF of the photz ... not sure how big
    #to make the columns ... needs to be fixed, but might
    #vary catalog to catalog
    #cat_photz_pdf_z = tables.Float32Col(1036, )
    #cat_photz_pdf_p = tables.Float32Col(1036, )


def version_match(fileh):
    """
    Checks an existing HDF5 file to see if the version is compatible for appending

    :param fileh: file handle to HDF5 file to be appended
    :return: T/F (if versions match), None or version for the version in the HDF5 file
    """
    if fileh is None:
        return False, None
    try:
        vtbl = fileh.root.Version
        # should be exactly one row
        rows = vtbl.read()
        if (rows is None) or (rows.size != 1):
            log.error("Problem loading Version table ...")
            return False, None

        existing_version = rows[0]['version'].decode()
        if existing_version == __version__:
            return True, existing_version
        else: #three decimal strings
            try:

                if upgrade(fileh,existing_version,__version__):
                    return True, __version__
                else:
                    ex_version = existing_version.split(".")
                    this_version = __version__.split(".")

                    if ex_version[0] == this_version[0]:
                        if ex_version[1] == this_version[1]:
                            return True, existing_version #only differ in engineering version
            except:
                return False, existing_version

            return False, existing_version #differ in major or minor version
    except:
        log.error("Exception! in elixer_hdf5::version_match().",exc_info=True)
        return False, None

    return False, None



def flush_all(fileh,reindex=True):

    if fileh is not None:
        #iterate over all tables and issue flush
        vtb = fileh.root.Version
        dtb = fileh.root.Detections
        ltb = fileh.root.SpectraLines
        stb = fileh.root.CalibratedSpectra
        atb = fileh.root.Aperture
        ctb = fileh.root.CatalogMatch

        vtb.flush()
        dtb.flush()
        ltb.flush()
        stb.flush()
        atb.flush()
        ctb.flush()

        try:
            etb = fileh.root.ExtractedObjects
            etb.flush()
        except:
            pass

        try:
            xtb = fileh.root.ElixerApertures
            xtb.flush()
        except:
            pass

        try: #this table is not always create, so may not exist
            ntb = fileh.root.NeighborSpectra
            ntb.flush()
        except:
            ntb = None

        try:
            vote_tb = fileh.root.ClassificationExtraFeatures
            vote_tb.flush()
        except:
            vote_tb = None

        if not reindex:
            return #we're done

        #remove (old) index if exists
        #vtb does not have or need an index
        try:
            dtb.cols.detectid.remove_index()
        except:
            log.debug("Failed to remove detectid index on detections table",exc_info=True)

        try:
            dtb.cols.ra.remove_index()
            dtb.cols.dec.remove_index()
        except:
            log.debug("Failed to remove ra, dec index on detections table",exc_info=True)

        try:
            ltb.cols.detectid.remove_index()
            stb.cols.detectid.remove_index()
            atb.cols.detectid.remove_index()
            ctb.cols.detectid.remove_index()
        except:
            log.debug("Failed to remove detectid index on multiple tables",exc_info=True)


        dtb.flush()
        ltb.flush()
        stb.flush()
        atb.flush()
        ctb.flush()

        #create (new) index
        # vtb does not have or need an index
        try:
            dtb.cols.detectid.create_csindex()
        except:
            log.debug("Index fail on detections table: detectid",exc_info=True)

        try:
            dtb.cols.ra.create_csindex()
            dtb.cols.dec.create_csindex()
        except:
            log.debug("Index fail on detections table: ra and/or dec",exc_info=True)


        try:
            ltb.cols.detectid.create_csindex()
        except:
            log.debug("Index fail on lines table",exc_info=True)

        try:
            stb.cols.detectid.create_csindex()
        except:
            log.debug("Index fail on spectra table",exc_info=True)

        try:
            atb.cols.detectid.create_csindex()
        except:
            log.debug("Index fail on apertures table",exc_info=True)

        try:
            ctb.cols.detectid.create_csindex()
        except:
            log.debug("Index fail on catalog match table",exc_info=True)


        #vtb.flush() # no need to re-flush vtb
        dtb.flush()
        ltb.flush()
        stb.flush()
        atb.flush()
        ctb.flush()

        try:
            etb.cols.detectid.remove_index()
            etb.cols.detectid.create_csindex()
            etb.flush()
        except:
            log.debug("Index fail on extracted objects table")

        try:
            xtb.cols.detectid.remove_index()
            xtb.cols.detectid.create_csindex()
            xtb.flush()
        except:
            log.debug("Index fail on elixer apertures table")


        try:
            if ntb is not None:
                ntb.cols.detectid.remove_index()
                ntb.cols.detectid.create_csindex()
                ntb.flush()
        except:
            log.debug("Index fail on neighbor spectra table")

        try:
            if vote_tb is not None:
                vote_tb.cols.detectid.remove_index()
                vote_tb.cols.detectid.create_csindex()
                vote_tb.flush()
        except:
            log.debug("Index fail on vote table")

    return



def get_hdf5_filehandle(fname,append=False,allow_overwrite=True,must_exist=False,
                        estimated_dets=tables.parameters.EXPECTED_ROWS_TABLE):
    """
    Return a file handle to work on. Create if does not exist, return existing handle if already present and versions
    are compatible (and append is requested).

    :param fname:
    :param append:
    :return:
    """

    if estimated_dets < tables.parameters.EXPECTED_ROWS_TABLE:
        estimated_dets = tables.parameters.EXPECTED_ROWS_TABLE
    fileh = None
    make_new = False
    try:
        if os.path.exists(fname):
            if append or must_exist:
                #log.debug("ELiXer HDF5 exists (%s). Will append if versions allow." %(fname))
                fileh = tables.open_file(fname, 'a', 'ELiXer Detection Catalog')
                #check the version

                if must_exist and not append:
                    return fileh

                version_okay, existing_version = version_match(fileh)
                if not version_okay:
                    if existing_version is None:
                        existing_version  = 'unknown'

                    log.error('ELiXer HDF5 Catalog (%s) already exists and does not match the current version. (%s != %s)'
                              %(fname,existing_version,__version__))
                    fileh.close()

                    #make under a different name?
                    make_new = True
                    fname += "_" + str(int(time.time())) + ".h5"
                    log.error('Making alternate ELiXer HDF5 Catalog (%s).' %(fname))
                else:
                    log.debug("ELiXer HDF5 exists (%s). Versions match (%s), will append." %(fname,__version__))
                    return fileh
            else:
                if allow_overwrite:
                    make_new = True
                    log.info("ELiXer HDF5 exists (%s). Append not requested. Will overwrite." % (fname))
                else:
                    make_new = False
                    log.info("ELiXer HDF5 exists (%s). Append not requested. New not allowd. Will fail." % (fname))
        else:
            if must_exist:
                make_new = False
                log.info("ELiXer HDF5 does not exist (%s). Must-exist enforced. Will fail." % (fname))
            else:
                make_new = True

        if make_new:
            log.debug("Creating new ELiXer HDF5 catalog (%s)" % (fname))

            fileh = tables.open_file(fname, 'w', 'ELiXer Detection Catalog')

            vtb = fileh.create_table(fileh.root, 'Version', Version,
                               'ELiXer Detection Version Table')
            #vtbl = fileh.root.Version
            row = vtb.row
            row['version'] = __version__
            row['version_pytables'] = tables.__version__
            row.append()
            vtb.flush()

            fileh.create_table(fileh.root, 'Detections', Detections,
                               'ELiXer Detection Summary Table',
                               expectedrows=estimated_dets)

            fileh.create_table(fileh.root, 'SpectraLines', SpectraLines,
                               'ELiXer Identified SpectraLines Table',
                               expectedrows=estimated_dets*2)

            fileh.create_table(fileh.root, 'CalibratedSpectra', CalibratedSpectra,
                               'HETDEX Flux Calibrated, PSF Weighted Summed Spectra Table',
                               expectedrows=estimated_dets)

            fileh.create_table(fileh.root, 'Aperture', Aperture,
                               'ELiXer Aperture Photometry Table',
                               expectedrows=estimated_dets*3) #mostly a g and r aperture, sometimes more

            fileh.create_table(fileh.root, 'CatalogMatch', CatalogMatch,
                               'ELiXer Catalog Matched Objected Table',
                               expectedrows=estimated_dets*3)

            fileh.create_table(fileh.root, 'ExtractedObjects',ExtractedObjects,
                               'ELiXer Image Extracted Objects Table',
                               expectedrows=estimated_dets*30) #multiple filters, many objects

            fileh.create_table(fileh.root, 'ElixerApertures', ElixerApertures,
                               'ELiXer Image Circular Apertures Table',
                               expectedrows=estimated_dets*3) #mostly a g and r aperture, sometimes more

            if G.LyC: #special Lyman Continuum project handling (this IS called from elixer so G.LyC is set)
                fileh.create_table(fileh.root, 'NeighborSpectra', NeighborSpectra,
                                   'NeighborSpectra Table',
                                   expectedrows=estimated_dets*3) #mostly a g and r aperture, sometimes more

            if G.VoteFeaturesTable:
                fileh.create_table(fileh.root,'ClassificationExtraFeatures',ClassificationExtraFeatures,'ClassificationExtraFeatures Table',
                                   expectedrows=estimated_dets)


    except:
        log.error("Exception! in elixer_hdf5::get_hdf5_filehandle().",exc_info=True)

    return fileh


def append_entry(fileh,det,overwrite=False):
    """

    :param fileh: file handle to the HDF5 file
    :param det: ELiXer DetObj
    :return:
    """
    try:
        #get tables
        dtb = fileh.root.Detections
        stb = fileh.root.CalibratedSpectra
        ltb = fileh.root.SpectraLines
        atb = fileh.root.Aperture
        ctb = fileh.root.CatalogMatch
        try:
            etb = fileh.root.ExtractedObjects
        except:
            fileh.create_table(fileh.root, 'ExtractedObjects',ExtractedObjects,
                               'ELiXer Image Extracted Objects Table')
            etb = fileh.root.ExtractedObjects

        try:
            xtb = fileh.root.ElixerApertures
        except:
            fileh.create_table(fileh.root, 'ElixerApertures', ElixerApertures,
                               'ELiXer Image Circular Apertures Table')
            xtb = fileh.root.ElixerApertures

        list_tables = [dtb,stb,ltb,atb,ctb,etb,xtb]


        try:
            ntb = fileh.root.NeighborSpectra
            list_tables.append(ntb)
            LyC = True
        except:
            LyC = False

        try:
            vote_tb = fileh.root.ClassificationExtraFeatures
            list_tables.append(vote_tb)
            Vote_Table = True
        except:
            Vote_Table = False

        q_detectid = det.hdf5_detectid
        rows = dtb.read_where("detectid==q_detectid")

        if rows.size > 0:
            if overwrite:
                log.info("Detection (%d) already exists in HDF5. Will replace." % (q_detectid))
                try:
                    for t in list_tables:
                        idx = t.get_where_list("detectid==q_detectid")
                        for i in np.flip(idx): #index gets updated, so delete in reverse order
                            t.remove_row(i)
                except:
                    log.info("Remove row(s) failed for %d. Skipping remainder." %(q_detectid),exc_info=True)
                    return
            else:
                log.info("Detection (%d) already exists in HDF5. Skipping." %(q_detectid))
                return

        #############################
        #Detection (summary) table
        #############################
        row = dtb.row
        #row[''] =
        row['detectid'] = det.hdf5_detectid
        row['detectname'] = det.hdf5_detectname
        row['elixer_version'] = det.elixer_version
        row['elixer_datetime'] = det.elixer_datetime #timestamp when base DetObj is built (note: is different than
                                                     #the timestamp in the PDF which is set when the PDF is built)

        row['shotid'] = det.survey_shotid  #this is int64 YYYYMMDDsss  where sss is the 3 digit shot (observation) ID
        row['obsid'] = det.fibers[0].obsid
        row['specid'] = det.fibers[0].specid
        row['ifuslot'] = det.fibers[0].ifuslot
        row['ifuid'] = det.fibers[0].ifuid

        if G.HDR_Version_float == 1:
            try:
                row['seeing_gaussian'] = det.survey_fwhm_gaussian
            except:
                row['seeing_gaussian'] = UNSET_FLOAT
            try:
                row['seeing_moffat'] = det.survey_fwhm_moffat
            except:
                row['seeing_moffat'] = UNSET_FLOAT
        else:
            try:
                row['seeing_fwhm'] = det.survey_fwhm
            except:
                row['seeing_fwhm'] = UNSET_FLOAT

        row['response'] = det.survey_response
        row['fieldname'] = det.survey_fieldname

        if det.wra is not None: #reminder, the displayed representation may not be full precision
            row['ra'] = det.wra
            row['dec'] = det.wdec
        elif det.ra is not None:
            row['ra'] = det.ra
            row['dec'] = det.dec
        row['wavelength_obs'] = det.w
        row['wavelength_obs_err'] = det.w_unc

        if (det.estflux_h5 is not None) and (det.estflux_h5 != det.estflux):
            #usually does not have the 10^-17 scale
            #sanity check
            if det.estflux_h5 > 1e-10:
                row['flux_line'] = det.estflux_h5 * G.HETDEX_FLUX_BASE_CGS
                row['flux_line_err'] = det.estflux_h5_unc * G.HETDEX_FLUX_BASE_CGS
            else:
                row['flux_line'] = det.estflux_h5
                row['flux_line_err'] = det.estflux_h5_unc
        else:
            row['flux_line'] = det.estflux
            row['flux_line_err'] = det.estflux_unc

        row['fwhm_line_aa'] = det.fwhm
        row['fwhm_line_aa_err'] = det.fwhm_unc
        if det.snr is not None:
            row['sn'] = det.snr
            row['sn_err'] = det.snr_unc
        elif det.sn is not None:
            row['sn'] = det.sn
        row['chi2'] = det.chi2
        row['chi2_err'] = det.chi2_unc

        if det.hetdex_cont_cgs is not None:
            row['continuum_line'] = det.hetdex_cont_cgs
            row['continuum_line_err'] = det.hetdex_cont_cgs_unc
        elif not det.using_sdss_gmag_ew:
            row['continuum_line'] = det.cont_cgs
            row['continuum_line_err'] = det.cont_cgs_unc

        _lya_1pz = det.w / G.LyA_rest #no sense is doing -1.0 then +1.0

        try:
            if det.eqw_line_obs is not None:
                row['eqw_rest_lya_line'] = det.eqw_line_obs / _lya_1pz
                row['eqw_rest_lya_line_err'] = det.eqw_line_obs_unc / _lya_1pz
            else:
                row['eqw_rest_lya_line'] = det.eqw_obs / _lya_1pz
                row['eqw_rest_lya_line_err'] = det.eqw_obs_unc / _lya_1pz
        except:
            pass

        row['plae_line'] = det.p_lae_oii_ratio
        try:
            if det.p_lae_oii_ratio_range:
                row['plae_line_min'] = det.p_lae_oii_ratio_range[1]
                row['plae_line_max'] = det.p_lae_oii_ratio_range[2]
        except:
            pass





        row['continuum_wide'] = det.best_gmag_cgs_cont
        row['continuum_wide_err'] = det.best_gmag_cgs_cont_unc
        row['mag_g_wide'] = det.best_gmag
        if det.best_gmag_unc is not None:
            row['mag_g_wide_err'] = det.best_gmag_unc

        try:
            row['continuum_masked'] = det.best_masked_cgs_cont
            row['continuum_masked_err'] = det.best_masked_cgs_cont_unc
        except:
            pass

        #hetdex line flux / sdss continuum flux
        try: #it is odd, but possible to have eqw_sdss_obs but NOT the _unc
            if det.best_eqw_gmag_obs is not None:
                row['eqw_rest_lya_wide'] = det.best_eqw_gmag_obs / _lya_1pz

            if det.best_eqw_gmag_obs_unc is not None:
                row['eqw_rest_lya_wide_err'] = det.best_eqw_gmag_obs_unc / _lya_1pz
        except:
            pass

        row['plae_wide'] = det.best_gmag_p_lae_oii_ratio
        try:
            if det.best_gmag_p_lae_oii_ratio_range:
                row['plae_wide_min'] = det.best_gmag_p_lae_oii_ratio_range[1]
                row['plae_wide_max'] = det.best_gmag_p_lae_oii_ratio_range[2]
        except:
            pass

        #
        # #full width (hetdex) specturm mag and related
        # try:
        #     row['continuum_full_spec'] = det.hetdex_gmag_cgs_cont
        #     row['continuum_full_spec_err'] = det.hetdex_gmag_cgs_cont_unc
        #     row['mag_full_spec'] = det.hetdex_gmag
        #
        #     row['plae_full_spec'] = det.hetdex_gmag_p_lae_oii_ratio
        #     if det.hetdex_gmag_unc is not None:
        #         row['mag_full_spec_err'] = det.hetdex_gmag_unc
        #
        #     # hetdex line flux / sdss continuum flux
        #     try:  # it is odd, but possible to have eqw_sdss_obs but NOT the _unc
        #         if det.eqw_hetdex_gmag_obs is not None:
        #             row['eqw_rest_lya_full_spec'] = det.eqw_hetdex_gmag_obs / _lya_1pz
        #
        #         if det.eqw_hetdex_gmag_obs_unc / _lya_1pz is not None:
        #             row['eqw_rest_lya_full_spec_err'] = det.eqw_hetdex_gmag_obs_unc / _lya_1pz
        #     except:
        #         pass
        #
        #     try:
        #         if det.hetdex_gmag_p_lae_oii_ratio:
        #             row['plae_full_spec_min'] = det.hetdex_gmag_p_lae_oii_ratio_range[1]
        #             row['plae_full_spec_max'] = det.hetdex_gmag_p_lae_oii_ratio_range[2]
        #     except:
        #         pass
        #
        # except:
        #     pass

        #
        if (det.spec_obj is not None) and (det.spec_obj.solutions is not None) and (len(det.spec_obj.solutions) > 0):
            row['multiline_flag'] = det.multiline_z_minimum_flag
            row['multiline_z'] = det.spec_obj.solutions[0].z
            row['multiline_rest_w'] = det.spec_obj.solutions[0].emission_line.w_rest
            row['multiline_name'] = det.spec_obj.solutions[0].emission_line.name

            row['multiline_raw_score'] = det.spec_obj.solutions[0].score
            row['multiline_frac_score'] = det.spec_obj.solutions[0].frac_score
            try:
                _,_,row['multiline_prob'] = det.multiline_solution_score()
                #row['multiline_prob'] = det.spec_obj.solutions[0].prob_real
            except:
                row['multiline_prob'] = -999.999
            #?? other lines ... other solutions ... move into a separate table ... SpectraLines table

        if (det.rvb is not None):
            # row['pseudo_color'] = det.rvb['color']
            # row['pseudo_color_min'] = det.rvb['color_range'][0]
            # row['pseudo_color_max'] = det.rvb['color_range'][1]

            row['pseudo_color_blue_flux'] = det.rvb['blue_flux_density_ujy']
            row['pseudo_color_blue_flux_err'] = det.rvb['blue_flux_density_err_ujy']
            row['pseudo_color_red_flux'] = det.rvb['red_flux_density_ujy']
            row['pseudo_color_red_flux_err'] = det.rvb['red_flux_density_err_ujy']
            row['pseudo_color_rvb_ratio'] = det.rvb['ratio']
            row['pseudo_color_rvb_ratio_err'] = det.rvb['ratio_err']

            row['pseudo_color_flag'] = det.rvb['flag']

        if (det.classification_dict is not None):
            try:
                row['combined_plae'] = det.classification_dict['plae_hat']
                # try:
                #     row['combined_plae_err'] = det.classification_dict['plae_hat_err']
                # except:
                #     row['combined_plae_err'] = det.classification_dict['plae_hat_sd']
                try:
                    row['combined_plae_lo'] = max(0.001,det.classification_dict['plae_hat_lo'])
                except:
                    row['combined_plae_lo'] = -1

                try:
                    row['combined_plae_hi'] = min(1000.0,det.classification_dict['plae_hat_hi'])
                except:
                    row['combined_plae_hi'] = -1

                row['plya_classification'] = det.classification_dict['scaled_plae']
                #last two are new
                row['combined_continuum'] = det.classification_dict['continuum_hat']
                row['combined_continuum_err'] = det.classification_dict['continuum_hat_err']

                row['combined_eqw_rest_lya'] = det.classification_dict['combined_eqw_rest_lya']
                row['combined_eqw_rest_lya_err'] = det.classification_dict['combined_eqw_rest_lya_err']

            except:
                pass

            try:
                if det.classification_dict['spurious_reason'] is not None:
                    row['spurious_reason'] = det.classification_dict['spurious_reason']
            except:
                pass

        if (det.spec_obj is not None) and (det.spec_obj.spectrum_slope is not None):
            try:
                #note: y-axis is 2AA wide flux (per standard HETDEX)
                #so .. units would be (rise/run) erg/s/cm^2 / AA  [units flux / units of AA == units of flux density ]
                #not an actual flux density [i.e. just a change in flux per change in wavelength]
                row['spectral_slope'] = det.spec_obj.spectrum_slope
                row['spectral_slope_err'] = det.spec_obj.spectrum_slope_err
            except:
                pass

        if det.ccd_adjacent_single_fiber_brightest_mag is not None:
            try:
                row['ccd_adjacent_mag'] = det.ccd_adjacent_single_fiber_brightest_mag
            except:
                pass

        if det.central_single_fiber_mag is not None:
            try:
                row['central_single_fiber_mag'] = det.central_single_fiber_mag
            except:
                pass

        try:
            row['ffsky_subtraction'] = det.extraction_ffsky #False by default, only true on re-extraction if --ffsky
        except:
            pass

        try: #added in version 0.2.4
            row['classification_labels'] = det.spec_obj.classification_label.rstrip(",")
        except:
            pass

        try:
            if det.best_z is not None:
                row['best_z'] = det.best_z
            if det.best_p_of_z:
                row['best_pz'] = det.best_p_of_z
        except:
            pass

        try:
            if det.flags is not None:
                row['flags'] = det.flags
            else:
                row['flags'] = 0
        except:
            try:
                row['flags'] = 0
            except:
                pass

        try:
            if det.needs_review is not None and (-128 <= det.needs_review <= 127) :
                row['review'] = np.int8(det.needs_review)
            else:
                row['review'] = 0
        except:
            try:
                row['review'] = 0
            except:
                pass


        try:
            if det.cluster_parent != 0:
                row['cluster_parent'] = det.cluster_parent
            else:
                row['cluster_parent'] = 0
        except:
            pass

        try:
            if det.color_ur is not None and det.color_ur[0] is not None:
                row['color_ur'] = det.color_ur[:]
        except:
            row['color_ur'] = [np.nan,np.nan,np.nan]

        try:
            if det.color_ug is not None and det.color_ug[0] is not None:
                row['color_ug'] = det.color_ug[:]
        except:
            row['color_ug'] = [np.nan,np.nan,np.nan]

        try:
            if det.color_gr is not None and det.color_gr[0] is not None:
                row['color_gr'] = det.color_gr[:]
        except:
            row['color_gr'] = [np.nan,np.nan,np.nan]

        row.append()
        dtb.flush()


        #############################
        #Calibrated Spectra Table
        #############################
        row = stb.row

        row['detectid'] = det.hdf5_detectid
        row['wavelength'] = det.sumspec_wavelength[:]
        row['flux'] = det.sumspec_flux[:]
        row['flux_err'] = det.sumspec_fluxerr[:]

        try:
            if det.extraction_aperture is not None:
                row['aperture_radius'] = det.extraction_aperture

                #yes, only trust if extraction_aperture is also set
                if det.extraction_ffsky is not None:
                    row['sky_background'] = 1 if det.extraction_ffsky else 0

            if det.fibers is not None:
                row['num_fibers'] = len(det.fibers)
        except:
            pass


        row.append()
        stb.flush()


        #############################
        #ELiXer Found Spectral Lines Table
        #############################

        try:
            if det.spec_obj.all_found_lines is None:
                l1 = []
            else:
                l1 = det.spec_obj.all_found_lines

            if det.spec_obj.all_found_absorbs is None:
                l2 = []
            else:
                l2 = det.spec_obj.all_found_absorbs

            for line in np.concatenate((l1,l2)):
                row = ltb.row
                row['detectid'] = det.hdf5_detectid
                row['sol_num'] = 0
                if line.absorber:
                    row['type'] = -1
                else:
                    row['type'] = 1

                row['wavelength'] = line.fit_x0
                # if line.mcmc_line_flux_tuple is not None:
                #     row['flux_line'] = line.mcmc_line_flux
                #     row['flux_line_err']  = 0.5 * (abs(line.mcmc_line_flux_tuple[1]) + abs(line.mcmc_line_flux_tuple[2]))
                #
                # else:
                row['flux_line'] = line.fit_line_flux
                row['flux_line_err'] = line.line_flux_err #line.fit_line_flux_err   #line.fit_rmse

                row['score'] = line.line_score
                if line.mcmc_snr is not None and line.mcmc_snr > 0:
                    row['sn'] = line.mcmc_snr
                else:
                    row['sn'] = line.snr

                try:
                    if line.mcmc_chi2:
                        row['chi2'] = line.mcmc_chi2
                    elif line.fit_chi2:
                        row['chi2'] = line.fit_chi2
                except:
                    pass


                row['used'] = False #these are all found lines, may or may not be in solution

                row.append()
                ltb.flush()
        except:
            log.debug(f"Exception.",exc_info=True)


        sol_num = 0
        if (det.spec_obj is not None) and (det.spec_obj.solutions is not None):
            for sol in det.spec_obj.solutions:
                sol_num += 1
                for line in sol.lines:
                    #THIS is an EmissinLine object NOT a fitted EmissionLineInfo object
                    row = ltb.row
                    row['detectid'] = det.hdf5_detectid
                    row['sol_num'] = sol_num
                    if line.absorber:
                        row['type'] = -1
                    else:
                        row['type'] = 1

                    row['wavelength'] = line.w_obs
                    row['flux_line'] = line.flux
                    row['flux_line_err'] = line.flux_err #should find the "found" line version to get this

                    row['score'] = line.line_score
                    row['sn'] = line.snr
                    try:
                        row['chi2'] = line.chi2
                    except:
                        pass

                    row['used'] = True  # these are all found lines, may or may not be in solution

                    row.append()
                    ltb.flush()

        #############################
        #Aperture Table
        #############################

        for d in det.aperture_details_list:
            if d['mag'] is not None:
                row = atb.row
                row['detectid'] = det.hdf5_detectid
                row['catalog_name'] = d['catalog_name']
                row['filter_name'] = d['filter_name'].lower()
                #row['image_depth_mag'] = ??
                row['ra'] = d['ra']
                row['dec'] = d['dec']
                row['radius'] = d['radius']

                try:
                    if d['fail_mag_limit']:
                        row['mag'] = d['raw_mag']
                        row['mag_err'] = d['raw_mag_err']
                    else:
                        row['mag'] = s['mag']
                        row['mag_err'] = s['mag_err']
                except:
                    row['mag']=d['mag']
                    row['mag_err'] = d['mag_err']
                row['aperture_area_pix'] = d['area_pix']
                row['sky_area_pix'] = d['sky_area_pix']
                row['aperture_cts'] = d['aperture_counts']
                row['sky_cts'] = d['sky_counts']
                row['sky_average'] = d['sky_average']
                row['eqw_rest_lya'] = d['aperture_eqw_rest_lya']
                try:
                    row['eqw_rest_lya_err'] = d['aperture_eqw_rest_lya_err']
                except:
                    pass
                row['plae'] = d['aperture_plae']
                try: #key might not exist
                    row['plae_max'] = d['aperture_plae_max']
                    row['plae_min'] = d['aperture_plae_min']
                except:
                    pass

                try: #added in 0.0.5
                    row['pixel_scale'] = d['pixel_scale']
                except:
                    pass

                try:
                    if d['mag_limit'] is not None:
                        row['image_depth_mag'] = d['mag_limit']
                except:
                    pass



                row.append()
                atb.flush()

        ################################
        # ElixerApertures
        ###############################
        for d in det.aperture_details_list:
            if d['elixer_apertures'] is None:
                continue
            for s in d['elixer_apertures']:
                row = xtb.row
                row['detectid'] = det.hdf5_detectid
                row['catalog_name'] = d['catalog_name']
                row['filter_name'] = d['filter_name'].lower()
                row['pixel_scale'] = d['pixel_scale']
                row['selected'] = s['selected']
                row['ra'] = s['ra']
                row['dec'] = s['dec']
                row['radius'] = s['radius']
                try:
                    if s['fail_mag_limit']:
                        row['mag'] = s['raw_mag']
                        row['mag_err'] = s['raw_mag_err']
                    else:
                        row['mag'] = s['mag']
                        row['mag_err'] = s['mag_err']
                except:
                    row['mag']=s['mag']
                    row['mag_err'] = s['mag_err']
                row['background_cts'] = s['sky_average']
                row['background_err'] = s['sky_err']
                row['flux_cts'] = s['aperture_counts']#/s['area_pix']
                #row['flux_err'] = s['flux_cts_err']
                #row['flags'] = s['flags']

                try:
                    row['image_flags'] = s['image_flags']
                except:
                    row['image_flags'] = 0

                try:
                    if d['mag_limit'] is not None:
                        row['image_depth_mag'] = d['mag_limit']
                except:
                    pass

                row.append()
                xtb.flush()

        ################################
        # ExtractedObjects
        ###############################
        for d in det.aperture_details_list:
            if d['sep_objects'] is None:
                continue
            for s in d['sep_objects']:
                row = etb.row
                row['detectid'] = det.hdf5_detectid
                row['catalog_name'] = d['catalog_name']
                row['filter_name'] = d['filter_name'].lower()
                row['pixel_scale'] = d['pixel_scale']
                row['selected'] = s['selected']
                row['ra'] = s['ra']
                row['dec'] = s['dec']
                row['major'] = s['a']
                row['minor'] = s['b']
                row['theta'] = s['theta']
                try:
                    if s['fail_mag_limit']:
                        row['mag'] = s['raw_mag']
                        row['mag_err'] = s['raw_mag_err']
                    else:
                        row['mag'] = s['mag']
                        row['mag_err'] = s['mag_err']
                except:
                    row['mag']=s['mag']
                    row['mag_err'] = s['mag_err']

                row['background_cts'] = s['background']
                row['background_err'] = s['background_rms']
                row['flux_cts'] = s['flux_cts']
                row['flux_err'] = s['flux_cts_err']
                row['flags'] = s['flags']
                try:
                    row['dist_curve'] = s['dist_curve']
                    row['dist_baryctr'] = s['dist_baryctr']
                except:
                    pass

                try:
                    row['image_flags'] = s['image_flags']
                except:
                    row['image_flags'] = 0

                try:
                    if d['mag_limit'] is not None:
                        row['image_depth_mag'] = d['mag_limit']
                except:
                    pass

                try:
                    if 'fixed_aper_flux_cts' in s.keys():
                        row['fixed_aper_radius'] = s['fixed_aper_radius']
                        row['fixed_aper_flux_cts'] = s['fixed_aper_flux_cts']
                        row['fixed_aper_flux_err'] = s['fixed_aper_flux_cts_err']
                        row['fixed_aper_mag']=s['fixed_aper_mag']
                        row['fixed_aper_mag_err'] = s['fixed_aper_mag_err']
                        row['fixed_aper_flags'] = s['fixed_aper_flags']
                except:
                    pass

                row.append()
                etb.flush()

        #################################
        #Catalog Match table
        ################################
        match_num = 0
        for d in det.bid_target_list:
            try:
                if -90.0 < d.bid_dec < 90.0: #666 or 181 is the aperture info
                    match_num += 1

                    #for f in d.filters: #just using selected "best" filter
                    row = ctb.row
                    row['detectid'] = det.hdf5_detectid
                    row['match_num'] = match_num
                    row['catalog_name'] = d.catalog_name
                    if d.bid_filter is not None:
                        row['filter_name'] = d.bid_filter.lower()
                    row['separation'] = d.distance
                    row['prob_match'] = d.prob_match
                    row['ra'] = d.bid_ra
                    row['dec'] = d.bid_dec
                    if (d.spec_z is not None) and (d.spec_z >= 0.0):
                        row['specz'] = d.spec_z
                    if (d.phot_z is not None) and (d.phot_z >= 0.0):
                        row['photz'] = d.phot_z
                    row['flux'] = d.bid_flux_est_cgs
                    row['flux_err'] = d.bid_flux_est_cgs_unc
                    row['mag'] = d.bid_mag

                    #some report as a +/- error and some report as the mag-error, mag+error
                    if d.bid_mag_err_bright / d.bid_mag > 0.5: #just a hard value
                        #this "error" is the mag with the error, not the error along
                        row['mag_err'] = 0.5 * (abs(d.bid_mag - d.bid_mag_err_bright) + abs(d.bid_mag - d.bid_mag_err_faint))
                    else:
                        #yes, this dilutes the error a bit, smaller to the bright, larger to the faint
                        row['mag_err'] = 0.5 * (abs(d.bid_mag_err_bright) + abs(d.bid_mag_err_faint))
                    row['plae'] = d.p_lae_oii_ratio

                    try: #var might not exist
                        row['plae_max'] = d.p_lae_oii_ratio_max
                        row['plae_min'] = d.p_lae_oii_ratio_min
                    except:
                        pass

                    if d.bid_ew_lya_rest is not None:
                        row['eqw_rest_lya'] = d.bid_ew_lya_rest
                        if d.bid_ew_lya_rest_err is not None:
                            row['eqw_rest_lya_err'] = d.bid_ew_lya_rest_err

                    try:
                        row['selected'] = d.selected
                    except:
                        pass

                    row.append()
                    ctb.flush()
            except:
                log.error("Exception! in elixer_hdf5::append_entry",exc_info=True)


        #################################
        #NeighborSpectra table
        ################################
        #todo: fill in table
        if LyC and (det.neighbors_sep is not None) and (det.neighbors_sep['sep_objects'] is not None):
            for n in det.neighbors_sep['sep_objects']:
                try:
                    if 'flux' in n.keys() and len(n['flux']>0) and np.any(n['flux']):
                        row = ntb.row
                        row['detectid'] = det.hdf5_detectid
                        if det.wra is not None: #reminder, the displayed representation may not be full precision
                            row['candidate_ra'] = det.wra
                            row['candidate_dec'] = det.wdec
                        elif det.ra is not None:
                            row['candidate_ra'] = det.ra
                            row['candidate_dec'] = det.dec

                        row['shotid'] = det.survey_shotid
                        try:
                            row['seeing_fwhm'] = det.survey_fwhm
                        except:
                            row['seeing_fwhm'] = UNSET_FLOAT

                        row['neighbor_ra'] = n['ra']
                        row['neighbor_dec'] = n['dec']

                        row['dist_curve'] = n['dist_curve']
                        row['dist_baryctr'] = n['dist_baryctr']
                        row['major'] = n['a']
                        row['minor'] = n['b']
                        row['theta'] = n['theta']

                        row['seeing_size'] = row['major']/row['seeing_fwhm']
                        row['seeing_dist'] = row['dist_curve']/row['seeing_fwhm']

                        row['sky_type'] = 1 if det.extraction_ffsky else 0 #1 = ff, 0 = lo
                        row['aperture_radius'] = det.extraction_aperture if det.extraction_aperture else 3.0
                        #row['spec_type'] = 0 if n['spec_type'] == 'dex' else 1 #0=DEX spectrum, 1= flat in fnu

                        #row['wavelength'] = n['wave'][:]

                        row['spec_val'] = n['flux'][:]
                        row['spec_err'] = n['flux_err'][:]
                        row['spec_type'] = 2 if np.any(row['spec_val']) else 0  #0 = not set, 1 = flux density (not used in elixer)
                                                                                #2 = flux
                        row['spec_scale'] = -17

                        # row['flat_flux'] = n['flat_flux'][:]
                        # row['flat_flux_err'] = n['flat_flux_err'][:]

                        row['filter'] = det.neighbors_sep['filter_name'].lower()

                        row['mag']=n['mag']
                        row['mag_err'] = n['mag_err']
                        row['mag_limit'] = det.neighbors_sep['mag_limit']

                        row['dex_g_mag'] = n['dex_g_mag']
                        row['dex_g_mag_err'] = n['dex_g_mag_err']

                        row.append()
                        ntb.flush()

                except:
                    log.error("Exception! in elixer_hdf5::append_entry",exc_info=True)


        ###################################
        # ClassificationExtraFeatures
        ###################################
        if Vote_Table:
            try:
                row = vote_tb.row
                row['detectid'] = det.hdf5_detectid

                #not always vote with info

                try:
                    row['plae_poii_combined_vote'] = det.vote_info['plae_poii_combined_vote']
                    row['plae_poii_combined_weight'] = det.vote_info['plae_poii_combined_weight']
                    row['plae_poii_combined_midpoint'] = det.vote_info['plae_poii_combined_midpoint']

                    row['plae_poii_combined'] = det.vote_info['plae_poii_combined']
                    row['plae_poii_combined_hi'] = det.vote_info['plae_poii_combined_hi']
                    row['plae_poii_combined_lo'] = det.vote_info['plae_poii_combined_lo']
                except:
                    pass


                try:
                    row['ew_rest_lya_combined'] = det.vote_info['ew_rest_lya_combined']
                    row['ew_rest_lya_combined_err'] = det.vote_info['ew_rest_lya_combined_err']
                except:
                    pass

                try:
                    row['ew_rest_lya_combined_vote'] = det.vote_info['ew_rest_lya_combined_vote']
                    row['ew_rest_lya_combined_weight'] = det.vote_info['ew_rest_lya_combined_weight']
                except:
                    pass


                try:
                    row['dex_gmag'] = det.vote_info['dex_gmag']
                    row['dex_gmag_bright'] = det.vote_info['dex_gmag_bright']
                    row['dex_gmag_faint'] = det.vote_info['dex_gmag_faint']
                    row['gmag_thresh_bright'] = det.vote_info['gmag_thresh_bright']
                    row['gmag_thresh_faint'] = det.vote_info['gmag_thresh_faint']
                except:
                    pass

                try:
                    row['dex_gmag_vote'] = det.vote_info['dex_gmag_vote']
                    row['dex_gmag_weight'] = det.vote_info['dex_gmag_weight']
                except:
                    pass

                try:
                    row['size_in_psf'] = det.vote_info['size_in_psf']
                    row['diam_in_arcsec'] = det.vote_info['diam_in_arcsec']
                    row['oii_size_in_kpc'] = det.vote_info['oii_size_in_kpc']
                    # row['lya_size_in_kpc'] = det.vote_info['lya_size_in_kpc']# is not one
                except:
                    pass

                try:
                    row['size_in_psf_vote'] = det.vote_info['size_in_psf_vote']
                    row['size_in_psf_weight'] = det.vote_info['size_in_psf_weight']
                except:
                    pass

                try:
                    row['rb_flux_asym'] = det.vote_info['rb_flux_asym']
                    row['rb_flux_asym_err'] = det.vote_info['rb_flux_asym_err']
                except:
                    pass

                try:
                    row['rb_flux_asym_vote'] = det.vote_info['rb_flux_asym_vote']
                    row['rb_flux_asym_weight']= det.vote_info['rb_flux_asym_weight']
                except:
                    pass

                try:
                    row['line_sigma'] = det.vote_info['line_sigma']
                    row['line_sigma_err']  = det.vote_info['line_sigma_err']
                except:
                    pass

                try:
                    row['line_sigma_vote'] = det.vote_info['line_sigma_vote']
                    row['line_sigma_weight'] = det.vote_info['line_sigma_weight']
                except:
                    pass



                row.append()
                vote_tb.flush()

            except:
                pass

    except:
        log.error("Exception! in elixer_hdf5::append_entry",exc_info=True)

    return


def build_elixer_hdf5(fname,hd_list=[],overwrite=False):
    """

    :param fname:
    :param hd_list:
    :param overwrite:  if TRUE, "remove" matching entries and replace with new ones per detectid
    :return:
    """
    #build a new HDF5 file from the current active run
    #this is like the old ELiXer creating _cat.txt and _fib.txt

    fileh = get_hdf5_filehandle(fname,append=False)

    if fileh is None:
        print("Unable to build ELiXer catalog.")
        log.error("Unable to build ELiXer catalog.")
        return

    for h in hd_list:  # iterate over all hetdex (hd) collections
        for e in h.emis_list: #for each detection in each hd collection
            if e.status >= 0:
                append_entry(fileh,e,overwrite)


    flush_all(fileh)
    fileh.close()
    print("File written: %s" %fname)
    log.info("File written: %s" %fname)

def extend_elixer_hdf5(fname,hd_list=[],overwrite=False):
    """

    :param fname:
    :param hd_list:
    :param overwrite: if TRUE, "remove" matching entries and replace with new ones per detectid
    :return:
    """
    #build a new HDF5 file from the current active run
    #this is like the old ELiXer creating _cat.txt and _fib.txt

    fileh = get_hdf5_filehandle(fname,append=True)

    if fileh is None:
        print("Unable to build ELiXer catalog.")
        log.error("Unable to build ELiXer catalog.")
        return

    for h in hd_list:  # iterate over all hetdex (hd) collections
        for e in h.emis_list: #for each detection in each hd collection
            if e.status >= 0:
                append_entry(fileh,e,overwrite)


    flush_all(fileh)
    fileh.close()
    print("File written: %s" %fname)
    log.info("File written: %s" %fname)

def detectid_in_file(fname,det_id):
    """

    :param fname:
    :param det_id:
    :return:
    """
    try:
        fileh = get_hdf5_filehandle(fname, must_exist=True, allow_overwrite=False, append=False)
        if fileh:
            dtb = fileh.root.Detections
            rows = dtb.read_where("detectid==det_id")
            if rows is None or len(rows) == 0:
                return 0
            else:
                return len(rows)
        else:
            return -1
    except:
        log.error("Exception in elixer_hdf5::detectid_in_file",exc_info=True)
        return -1

def remove_duplicates(file):
    """
    Scan the Detections table for duplicate detectIDs and then remove the duplicate rows in all tables.

    :param file:
    :return:
    """

    try:
        h5 = get_hdf5_filehandle(file, append=True, allow_overwrite=True, must_exist=True)

        if h5 is None:
            log.info("Unable to open source file for remove_duplicates.")
            return False

    except:
        log.error("Exception! in elixer_hdf5::remove_duplicates", exc_info=True)


    try:
        dtb = h5.root.Detections
        stb = h5.root.CalibratedSpectra
        ltb = h5.root.SpectraLines
        atb = h5.root.Aperture
        ctb = h5.root.CatalogMatch
        etb = h5.root.ExtractedObjects #new MUST have this table
        xtb = h5.root.ElixerApertures

        try:
            ntb = h5.root.NeighborSpectra
            LyC = True
        except:
            LyC = False

        try:
            vote_tb = h5.root.ClassificationExtraFeatures
            Vote_Table = True
        except:
            Vote_Table = False

        detectids = dtb.read(field='detectid')

        #identify the duplicates
        u, uidx, ucts = np.unique(detectids, return_index=True, return_counts=True)
        sel = np.where(ucts > 1)
        dups = u[sel][:]
        cts = ucts[sel][:]

        if len(dups)==0:
            log.info("No duplicates found.")
            h5.close()
            return True

        log.info(f"Removing duplicates for {len(dups)} detections ...")

        #find the rows in each table for the duplicates
        for d,c in zip(dups,cts):
            try:

                log.info(f"Removing {c-1} duplicates for {d} ...")

                #assuming the rows are in order, but may not be continguous
                #that is, all the rows belonging to the first instance of the detectID appear before any other
                #duplicate rows, BUT, there could be different detectID rows interspersed, so we delete one row at a time

                #maybe it is reindexing after each removal, so I guess we have to do this with remove_rows
                #Detections table is one row per (one per detectid)
                rows = dtb.get_where_list("detectid==d")
                if rows.size > 1:
                    dtb.remove_rows(rows[1],rows[-1]+1)
                    #dtb.flush()

                #CalibratedSpectra (one per detectid)
                rows = stb.get_where_list("detectid==d")
                if rows.size > 1:
                    stb.remove_rows(rows[1],rows[-1]+1)
                    #stb.flush()

                #SpectraLines
                rows = ltb.get_where_list("detectid==d")
                if rows.size > 1:
                    start = rows.size / c
                    if start.is_integer():
                        start = int(start)
                        ltb.remove_rows(rows[start],rows[-1]+1)
                        #ltb.flush()

                #Aperture
                rows = atb.get_where_list("detectid==d")
                if rows.size > 1:
                    start = rows.size / c
                    if start.is_integer():
                        start = int(start)
                        atb.remove_rows(rows[start],rows[-1]+1)
                        #atb.flush()

                #CatalogMatch
                rows = ctb.get_where_list("detectid==d")
                if rows.size > 1:
                    start = rows.size / c
                    if start.is_integer():
                        start = int(start)
                        ctb.remove_rows(rows[start], rows[-1] + 1)
                        #ctb.flush()

                #ExtractedObjects
                rows = etb.get_where_list("detectid==d")
                if rows.size > 1:
                    start = rows.size / c
                    if start.is_integer():
                        start = int(start)
                        etb.remove_rows(rows[start], rows[-1] + 1)
                        #etb.flush()

                #ElixerApertures
                rows = xtb.get_where_list("detectid==d")
                if rows.size > 1:
                    start = rows.size / c
                    if start.is_integer():
                        start = int(start)
                        xtb.remove_rows(rows[start], rows[-1] + 1)
                        #xtb.flush()

                if LyC:
                    rows = ntb.get_where_list("detectid==d")
                    if rows.size > 1:
                        start = rows.size / c
                        if start.is_integer():
                            start = int(start)
                            ntb.remove_rows(rows[start], rows[-1] + 1)


                if Vote_Table:
                    rows = vote_tb.get_where_list("detectid==d")
                    if rows.size > 1:
                        start = rows.size / c
                        if start.is_integer():
                            start = int(start)
                            vote_tb.remove_rows(rows[start], rows[-1] + 1)

            except:
                log.error(f"Exception removing rows for {d}",exc_info=True)

        log.info("Remove duplicate rows complete. Flushing the file ...")
        flush_all(h5)
        h5.close()
        log.info("Remove duplicates complete. Done.")
        return True

    except:
        log.error("Exception! conducting merge in elixer_hdf5::remove_duplicates", exc_info=True)
        return False


#
#
# def delete_entries(file,delete_list):
#     """
#     Mark detectIDs (in all tables) from the delete_list as deleted (remove_row)
#
#     :param file:
#     :return:
#     """
#
#     try:
#         h5 = get_hdf5_filehandle(file, append=True, allow_overwrite=True, must_exist=True)
#
#         if h5 is None:
#             log.info("Unable to open source file for remove_duplicates.")
#             return False
#
#     except:
#         log.error("Exception! in elixer_hdf5::delete_entries", exc_info=True)
#
#
#     try:
#         dtb = h5.root.Detections
#         stb = h5.root.CalibratedSpectra
#         ltb = h5.root.SpectraLines
#         atb = h5.root.Aperture
#         ctb = h5.root.CatalogMatch
#         etb = h5.root.ExtractedObjects #new MUST have this table
#         xtb = h5.root.ElixerApertures
#
#         dtb.cols.detectid.remove_index()
#         ltb.cols.detectid.remove_index()
#         stb.cols.detectid.remove_index()
#         atb.cols.detectid.remove_index()
#         ctb.cols.detectid.remove_index()
#
#         dtb.flush()
#         ltb.flush()
#         stb.flush()
#         atb.flush()
#         ctb.flush()
#
#         detectids = delete_list
#
#         log.info(f"Removing entries for {len(detectids)} detections ...")
#
#         #find the rows in each table for the duplicates
#         for c, d in enumerate(detectids):
#             try:
#
#                 log.info(f"Removing #{c+1} ({d}) ...")
#
#                 #assuming the rows are in order, but may not be continguous
#                 #that is, all the rows belonging to the first instance of the detectID appear before any other
#                 #duplicate rows, BUT, there could be different detectID rows interspersed, so we delete one row at a time
#
#                 #maybe it is reindexing after each removal, so I guess we have to do this with remove_rows
#                 #Detections table is one row per (one per detectid)
#                 rows = dtb.get_where_list("detectid==d")
#                 for r in rows:
#                     dtb.remove_row(r)
#                 # if rows.size > 1:
#                 #     dtb.remove_rows(rows[1],rows[-1]+1)
#                 #     #dtb.flush()
#
#                 #CalibratedSpectra (one per detectid)
#                 rows = stb.get_where_list("detectid==d")
#                 for r in rows:
#                     stb.remove_row(r)
#                 # if rows.size > 1:
#                 #     stb.remove_rows(rows[1],rows[-1]+1)
#                 #     #stb.flush()
#
#                 #SpectraLines
#                 rows = ltb.get_where_list("detectid==d")
#                 for r in rows:
#                     ltb.remove_row(r)
#                 # if rows.size > 1:
#                 #     start = rows.size / c
#                 #     if start.is_integer():
#                 #         start = int(start)
#                 #         ltb.remove_rows(rows[start],rows[-1]+1)
#                         #ltb.flush()
#
#                 #Aperture
#                 rows = atb.get_where_list("detectid==d")
#                 for r in rows:
#                     atb.remove_row(r)
#                 # if rows.size > 1:
#                 #     start = rows.size / c
#                 #     if start.is_integer():
#                 #         start = int(start)
#                 #         atb.remove_rows(rows[start],rows[-1]+1)
#                 #         #atb.flush()
#
#                 #CatalogMatch
#                 rows = ctb.get_where_list("detectid==d")
#                 for r in rows:
#                     ctb.remove_row(r)
#                 # if rows.size > 1:
#                 #     start = rows.size / c
#                 #     if start.is_integer():
#                 #         start = int(start)
#                 #         ctb.remove_rows(rows[start], rows[-1] + 1)
#                         #ctb.flush()
#
#                 #ExtractedObjects
#                 rows = etb.get_where_list("detectid==d")
#                 for r in rows:
#                     etb.remove_row(r)
#                 # if rows.size > 1:
#                 #     start = rows.size / c
#                 #     if start.is_integer():
#                 #         start = int(start)
#                 #         etb.remove_rows(rows[start], rows[-1] + 1)
#                 #         #etb.flush()
#
#                 #ElixerApertures
#                 rows = xtb.get_where_list("detectid==d")
#                 for r in rows:
#                     xtb.remove_row(r)
#                 # if rows.size > 1:
#                 #     start = rows.size / c
#                 #     if start.is_integer():
#                 #         start = int(start)
#                 #         xtb.remove_rows(rows[start], rows[-1] + 1)
#                 #         #xtb.flush()
#             except:
#                 log.error(f"Exception removing rows for {d}",exc_info=True)
#
#         log.info("Remove duplicate rows complete. Flushing the file ...")
#         flush_all(h5)
#         h5.close()
#         log.info("Remove duplicates complete. Done.")
#         return True
#
#     except:
#         log.error("Exception! conducting merge in elixer_hdf5::merge_unique", exc_info=True)
#         return False


def merge_unique(newfile,file1,file2):
    """
    Merge, detectID by detectID file1 and file2 into newfile, keeping only the most recent detectID if
    there are duplicates.

    :param newfile: file name of new file to create as merge of file1 and file2
    :param file1:  either of the two files to merge
    :param file2:  other file to merge
    :return:
    """
    import glob

    chunk_size = 25000 #25k
    try:
        file1_handle = get_hdf5_filehandle(file1,append=False,allow_overwrite=False,must_exist=True)
        file2_handle = get_hdf5_filehandle(file2, append=False, allow_overwrite=False, must_exist=True)


        if (file1_handle is None) or (file2_handle is None):
            print("Unable to open source file(s) for merge_unique.")
            log.info("Unable to open source file(s) for merge_unique.")
            return False

        # max_dets = len(file1_handle.root.Detections) + len(file2_handle.root.Detections)
        #
        # newfile_handle = get_hdf5_filehandle(newfile,append=False,allow_overwrite=False,must_exist=False,
        #                                      estimated_dets=max_dets)
        #
        # if newfile_handle is None:
        #     print("Unable to create destination file for merge_unique. File may already exist.")
        #     log.info("Unable to create destination file for merge_unique.")
        #     return False

    except:
        log.error("Exception! in elixer_hdf5::merge_unique",exc_info=True)

    #todo: enforce version matching?? since creating new file, these should be backward compatible with defaults for
    #todo: missing columns

    try:

        dtb1 = file1_handle.root.Detections
        dtb2 = file2_handle.root.Detections

        detectids = dtb1.read()['detectid']
        detectids = np.concatenate((detectids,dtb2.read()['detectid']))

        detectids = np.array(sorted(set(detectids))) #'set' so they are unique

        #break into chunks of 100,000
        num_chunks = int(len(detectids)/chunk_size)+1
        detect_chunks = np.array_split(detectids,num_chunks)

        log.info("Merging %d detections ..." %len(detectids))

        for chunk in detect_chunks:
            #make a new receiving h5 file
            log.info(f"Merging for chunk starting at {chunk[0]}")
            newfile_chunk = newfile + f".chunk{chunk[0]}"
            newfile_handle = get_hdf5_filehandle(newfile_chunk, append=False, allow_overwrite=True, must_exist=False,
                                                 estimated_dets=chunk_size)

            if newfile_handle is None:
                print(f"Unable to create destination file {newfile_chunk} for merge_unique. File may already exist.")
                log.info(f"Unable to create destination file {newfile_chunk} for merge_unique.")
                return False

            dtb_new = newfile_handle.root.Detections
            stb_new = newfile_handle.root.CalibratedSpectra
            ltb_new = newfile_handle.root.SpectraLines
            atb_new = newfile_handle.root.Aperture
            ctb_new = newfile_handle.root.CatalogMatch
            etb_new = newfile_handle.root.ExtractedObjects  # new MUST have this table
            xtb_new = newfile_handle.root.ElixerApertures

            try:
                ntb_new = newfile_handle.root.NeighborSpectra
                LyC = True
            except:
                LyC = False

            try:
                vote_tb_new = newfile_handle.root.ClassificationExtraFeatures
                Vote_Table = True
            except:
                Vote_Table = False

            for d in chunk:
                try:
                    log.debug(f"Merging {d}")

                    source_h = None

                    date1 = dtb1.read_where('detectid==d')['elixer_datetime']
                    date2 = dtb2.read_where('detectid==d')['elixer_datetime']
                    date_new = dtb_new.read_where('detectid==d')['elixer_datetime']
                    q_date = None

                    #temporary
                    # if (date1.size > 0) and (date2.size > 0):
                    #     print("Duplicates",d)

                    #choose nearest date
                    if date1.size == 0:
                        if date2.size == 0: #this is impossible for both
                            log.error("Impossible ... both dates returned no rows: detectid (%d)" %d)
                            continue
                        elif date2.size > 1: #file2 to be used, file1 has not entry
                            #pick newest date
                            source_h = file2_handle
                            q_date = max(date2)
                        else:
                            source_h = file2_handle
                            q_date = date2[0]
                    elif date2.size == 0:  #file1 to be used, file2 has no entry
                        if date1.size > 1:
                            source_h = file1_handle
                            q_date = max(date1)
                        else:
                            source_h = file1_handle
                            q_date = date1[0]
                    else: #both have entries
                        best_date1 = max(date1)
                        best_date2 = max(date2)

                        if best_date1 > best_date2:
                            source_h = file1_handle
                            q_date = best_date1
                        else:
                            source_h = file2_handle
                            q_date = best_date2

                    #now check the that NEW file does not already have this
                    if date_new.size == 0: #it does not, so proceed
                        pass
                    elif date_new.size == 1:
                        if date_new < q_date:
                            #the "new" file is already out of date (from a previous trip through this loop)
                            #really, this should not happen either and for now, just alarm and move on
                            print(f"Elixer merge_unique, new file already found for {d}")
                            log.error(f"Elixer merge_unique, new file already found for {d}")
                            continue
                        else: #already good
                            print(f"Elixer merge_unique, new file already found for {d}. Date is good. Keeping ...")
                            log.info(f"Elixer merge_unique, new file already found for {d}. Date is good. Keeping ...")
                            continue
                    else: #this should be impossible
                        print(f"Elixer merge_unique, multiple entries ({date_new.size}) in new file already found for {d}")
                        log.error(f"Elixer merge_unique, multiple entries ({date_new.size}) in new file already found for {d}")
                        continue


                    if source_h is None:
                        continue

                    dtb_src = source_h.root.Detections
                    stb_src = source_h.root.CalibratedSpectra
                    ltb_src = source_h.root.SpectraLines
                    atb_src = source_h.root.Aperture
                    ctb_src = source_h.root.CatalogMatch

                    dtb_new.append(dtb_src.read_where("(detectid==d) & (elixer_datetime==q_date)"))
                    #################################
                    #manual merge of defunct version
                    #################################
                    #if False:
                    #   old_row = dtb_src.read_where("(detectid==d) & (elixer_datetime==q_date)")[0]
                    #   new_row = dtb_new.row
                    #   temp_append_dtb_002_to_003(new_row,old_row)

                    #unfortunately, have to assume following data is unique
                    stb_new.append(stb_src.read_where("(detectid==d)"))
                    ltb_new.append(ltb_src.read_where("(detectid==d)"))
                    atb_new.append(atb_src.read_where("(detectid==d)"))
                    ctb_new.append(ctb_src.read_where("(detectid==d)"))
                    try:
                        etb_src = source_h.root.ExtractedObjects
                        etb_new.append(etb_src.read_where("(detectid==d)"))
                    except Exception as e:
                        print(f"ExtractedObjects merge failed {d}")
                        print(e)

                    try:
                        xtb_src = source_h.root.ElixerApertures
                        xtb_new.append(xtb_src.read_where("(detectid==d)"))
                    except Exception as e:
                        print(f"ElixerApertures merge failed {d}")
                        print(e)

                    if LyC:
                        try:
                            ntb_src = source_h.root.NeighborSpectra
                            ntb_new.append(ntb_src.read_where("(detectid==d)"))
                        except Exception as e:
                            print(f"NeighborSpectra merge failed {d}")
                            print(e)

                    if Vote_Table:
                        try:
                            vote_tb_src = source_h.root.ClassificationExtraFeatures
                            vote_tb_new.append(vote_tb_src.read_where("(detectid==d)"))
                        except Exception as e:
                            print(f"ClassificationExtraFeatures merge failed {d}")
                            print(e)

                    #flush_all(newfile_handle) #don't think we need to flush every time
                    #definitely DO NOT reindex each time ... super costly; flush each time is okay, but does increase
                    #overall time a bit and is not necessary

                except Exception as e:
                    print(f"Exception! merging detectid {d} : {e}")
                    log.error("Exception! merging detectid (%d): (%s)" %(d,e))
             # end for loop
            flush_all(newfile_handle,reindex=True) #re-index is important for the final merge
            newfile_handle.close()

        #end for loop (chunks)
        file2_handle.close()
        file1_handle.close()

        #now glob all the chunks and regular merge (already know they are unique)
        log.info("Chunking done. Calling merge_elixer_hdf5_files ...")
        chunk_files = glob.glob(newfile + ".chunk*")
        merge_elixer_hdf5_files(newfile,chunk_files)

        #now, cleanup the chunks
        for cf in chunk_files:
            os.remove(cf)

    except:
        log.error("Exception! conducting merge in elixer_hdf5::merge_unique", exc_info=True)
        return False

    return True


def merge_chunks(newfile):
    """
    merge newfile + ".chunk*" into new file

    :param newfile:
    :return:
    """

    try:
        #now glob all the chunks and regular merge (already know they are unique)
        log.info("Merging chunks. Calling merge_elixer_hdf5_files ...")
        chunk_files = glob.glob(newfile + ".chunk*")
        merge_elixer_hdf5_files(newfile,chunk_files)
        return True
    except:
        log.error("Exception! conducting merge in elixer_hdf5::merge_chunks", exc_info=True)
        return False


def merge_elixer_hdf5_files(fname,flist=[]):
    """

    :param fname: the output (final/merged) HDF5 file
    :param flist:  list of all files to merge
    :return: None or filename
    """

    #first, estimate the number of detections
    max_dets = 0
    for f in flist:
        if f == fname: #could be the output file is one of those to merge
            continue #just skip and move on

        fh = get_hdf5_filehandle(f,append=False,allow_overwrite=False,must_exist=True)
        try:
            _ = fh.root.NeighborSpectra
            G.LyC = True
        except:
            pass

        try:
            _ = fh.root.ClassificationExtraFeatures
            G.VoteFeaturesTable = True
        except:
            pass

        if fh is None:
            continue
        else:
            max_dets += len(fh.root.Detections)

    #merging existing distinct HDF5 files w/o new additions from an active run
    fileh = get_hdf5_filehandle(fname,append=True,estimated_dets=max_dets)

    if fileh is None:
        log.error("Unable to merge ELiXer catalogs.")
        return None

    #set up new HDF5 tables (into which to append)
    dtb = fileh.root.Detections

    stb = fileh.root.CalibratedSpectra
    ltb = fileh.root.SpectraLines
    atb = fileh.root.Aperture
    ctb = fileh.root.CatalogMatch
    etb = fileh.root.ExtractedObjects
    xtb = fileh.root.ElixerApertures

    try: #only exist if elixer was run with --lyc
        ntb = fileh.root.NeighborSpectra
        LyC = True
    except:
        LyC = False

    try: #only exist if elixer was run with --lyc
        vote_tb = fileh.root.ClassificationExtraFeatures
        Vote_Table = True
    except:
        Vote_Table = False


    log.info(f"Merging approximately {max_dets} in {len(flist)} files ...")

    for f in flist:
        if f == fname: #could be the output file is one of those to merge
            continue #just skip and move on

        merge_fh = get_hdf5_filehandle(f,append=True)

        if merge_fh is None:
            log.error("Unable to merge: %s" %(f))
            continue

        m_dtb = merge_fh.root.Detections
        m_stb = merge_fh.root.CalibratedSpectra
        m_ltb = merge_fh.root.SpectraLines
        m_atb = merge_fh.root.Aperture
        m_ctb = merge_fh.root.CatalogMatch


        #now merge
        dtb.append(m_dtb.read())
        stb.append(m_stb.read())
        ltb.append(m_ltb.read())
        atb.append(m_atb.read())
        ctb.append(m_ctb.read())

        try: #might not have ExtractedObjects table
            m_etb = merge_fh.root.ExtractedObjects
            etb.append(m_etb.read())
        except:
            pass

        try: #might not have ElixerApertures table
            m_xtb = merge_fh.root.ElixerApertures
            xtb.append(m_xtb.read())
        except:
            pass

        if LyC:
            try: #might not have ElixerApertures table
                m_ntb = merge_fh.root.NeighborSpectra
                ntb.append(m_ntb.read())
            except:
                pass

        if Vote_Table:
            try: #might not have ElixerApertures table
                m_vote_tb = merge_fh.root.ClassificationExtraFeatures
                vote_tb.append(m_vote_tb.read())
            except:
                pass

        flush_all(fileh,reindex=True) #for now, just to be safe, reindex anyway
        #close the merge input file
        merge_fh.close()

    flush_all(fileh,reindex=True)
    fileh.close()
    return fname



#######################################
# Version migrations
#######################################

def add_column(table, ColClass, collabel, colvalues):
    # Step 1: Adjust table description
    d = table.description._v_colobjects.copy()  # original description
    d[collabel] = ColClass#()  # add column

    # Step 2: Create new temporary table:
    newtable = tables.Table(table._v_file.root, '_temp_table', d, filters=table.filters)  # new table
    ###Step 2a: Copy attributes:
    table.attrs._f_copy(newtable)
    ###Step 2b: Copy table rows, also add new column values:
    for row, value in zip(table, colvalues):
        newtable.append([tuple(list(row[:]) + [value])])
    newtable.flush()

    # Step 3: Move temporary table to original location:
    parent = table._v_parent  # original table location
    name = table._v_name  # original table name
    table.remove()  # remove original table
    newtable.move(parent, name)  # move temporary table to original location

    return newtable

def upgrade(fileh,old_version, new_version):
    #is there an upgrade for fileh to version?
    done = False

    func_list = []
    max_version = old_version

    while not done:
        if max_version == '0.0.4':
            func_list.append(upgrade_0p0p4_to_0p1p0)
            max_version = "0.1.0"
        elif max_version == '0.0.5': #either way go to 0.1.0
            func_list.append(upgrade_0p0p4_to_0p1p0)
            max_version = "0.1.0"
        elif max_version == "0.2.3":
            func_list.append(upgrade_0p2p3_to_0p2p4)
            max_version = "0.2.4"

        else:
            done = True

    if max_version == new_version:
        for f in func_list:
            result = f(fileh)
            if not result:
                return False

        return True
    else:
        return False



def temp_append_dtb_002_to_003(row,old_row):
    #############################
    # Detection (summary) table
    #############################
    # row[''] =
    row['detectid'] = old_row['detectid']
    row['detectname'] = old_row['detectname'].tostring()
    row['elixer_version'] = old_row['elixer_version'].tostring()
    row['elixer_datetime'] = old_row['elixer_datetime'].tostring() # timestamp when base DetObj is built (note: is different than
    # the timestamp in the PDF which is set when the PDF is built)

    row['shotid'] = old_row['shotid']  # this is int64 YYYYMMDDsss  where sss is the 3 digit shot (observation) ID
    row['obsid'] = old_row['obsid']
    row['specid'] = old_row['specid'].tostring()
    row['ifuslot'] = old_row['ifuslot'].tostring()
    row['ifuid'] = old_row['ifuid'].tostring()
    row['seeing_gaussian'] = old_row['seeing_gaussian']
    row['seeing_moffat'] = old_row['seeing_moffat']
    row['response'] = old_row['response']
    row['fieldname'] =  old_row['fieldname'].tostring()

    row['ra'] = old_row['ra']
    row['dec'] = old_row['dec']
    row['wavelength_obs'] = old_row['wavelength_obs']
    row['wavelength_obs_err'] = old_row['wavelength_obs_err']


    row['flux_line'] = old_row['flux_line']
    row['flux_line_err'] =  old_row['flux_line_err']

    row['fwhm_line_aa'] =old_row['fwhm_line_aa']
    row['fwhm_line_aa_err'] = old_row['fwhm_line_aa_err']
    row['sn'] = old_row['sn']
    row['sn_err'] = old_row['sn_err']

    row['chi2'] = old_row['chi2']
    row['chi2_err'] = old_row['chi2_err']

    row['continuum_line'] = old_row['continuum_line']
    row['continuum_line_err'] = old_row['continuum_line_err']

    row['continuum_sdss_g'] = old_row['continuum_sdss_g']
    row['continuum_sdss_g_err'] = old_row['continuum_sdss_g_err']
    row['mag_sdss_g'] = old_row['mag_sdss_g']
    row['mag_sdss_g_err'] = old_row['mag_sdss_g_err']

    row['eqw_rest_lya_line'] = old_row['eqw_rest_lya_line']
    row['eqw_rest_lya_line_err'] = old_row['eqw_rest_lya_line_err']

    row['eqw_rest_lya_sdss_g'] = old_row['eqw_rest_lya_sdss_g']
    row['eqw_rest_lya_sdss_g_err'] = old_row['eqw_rest_lya_sdss_g_err']

    row['plae_line'] = old_row['plae_line']
    row['plae_sdss_g'] = old_row['plae_sdss_g']

    try:
        row['plae_line_max'] = old_row['plae_line_max']
        row['plae_line_min'] = old_row['plae_line_min']
        row['plae_sdss_g_max'] = old_row['plae_sdss_g_max']
        row['plae_sdss_g_min'] = old_row['plae_sdss_g_min']
    except:
        pass

    row['multiline_flag'] = old_row['multiline_flag']
    row['multiline_z'] = old_row['multiline_z']
    row['multiline_rest_w'] = old_row['multiline_rest_w']
    row['multiline_name'] = old_row['multiline_name'].tostring()

    row['multiline_raw_score'] = old_row['multiline_raw_score']
    row['multiline_frac_score'] =old_row['multiline_frac_score']
    row['multiline_prob'] =old_row['multiline_prob']
        # ?? other lines ... other solutions ... move into a separate table ... SpectraLines table

    try:
        #row['pseudo_color'] = det.rvb['color']
        #row['pseudo_color_min'] = det.rvb['color_range'][0]
        #row['pseudo_color_max'] = det.rvb['color_range'][1]

        row['pseudo_color_blue_flux'] = old_row['pseudo_color_blue_flux']
        row['pseudo_color_blue_flux_err'] = old_row['pseudo_color_blue_flux_err']
        row['pseudo_color_red_flux'] = old_row['pseudo_color_red_flux']
        row['pseudo_color_red_flux_err'] = old_row['pseudo_color_red_flux_err']
        row['pseudo_color_rvb_ratio'] = old_row['pseudo_color_rb_ratio']
        row['pseudo_color_rvb_ratio_err'] = old_row['pseudo_color_rb_ratio_err']

        row['pseudo_color_flag'] = old_row['pseudo_color_flag']
    except:
        pass

    try:
        row['combined_plae'] = old_row['combined_plae']
        row['combined_plae_err'] = old_row['combined_plae_err']
        row['plae_classification'] = old_row['plae_classification']
    except:
        pass

    try:
        row['continuum_full_spec'] = old_row['continuum_full_spec']
        row['continuum_full_spec_err'] = old_row['continuum_full_spec_err']
        row['mag_full_spec'] = old_row['mag_full_spec']
        row['mag_full_spec_err'] = old_row['mag_full_spec_err']
        row['eqw_rest_lya_full_spec'] = old_row['eqw_rest_lya_full_spec']
        row['eqw_rest_lya_full_spec_err'] = old_row['eqw_rest_lya_full_spec_err']
        row['plae_full_spec'] = old_row['plae_full_spec']
        row['plae_full_spec_max'] = old_row['plae_full_spec_max']
        row['plae_full_spec_min'] = old_row['plae_full_spec_min']

    except:
        pass

    row.append()


def upgrade_0p2p3_to_0p2p4(fileh):
    from_version = "0.2.3"
    to_version = "0.2.4"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))


        table = fileh.root.Detections  # table to which column is to be appended
        ColClass =  tables.StringCol(itemsize=32, dflt="")  # new column class
        collabel = 'classification_labels'  # new column label
        colvalues = np.full(len(table),"")  # new column values
        newtable = add_column(table, ColClass, collabel, colvalues)
        #print(fileh.root.mytable[:])  # "table" is gone and "newtable" is now fileh.root.mytable

        #lastly update the version
        vtb = fileh.root.Version
        for row in vtb:  # should be only one
            row['version'] = to_version
            row['version_pytables'] = tables.__version__
            row.update()
        vtb.flush()
        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False


# def upgrade_0p0p4_to_0p0p5(fileh):
#     from_version = "0.0.4"
#     to_version = "0.0.5"
#
#     try:
#         log.info("Upgrading %s to %s ..." %(from_version,to_version))
#
#
#
#
#         #lastly update the version
#         vtb = fileh.root.Version
#         for row in vtb:  # should be only one
#             row['version'] = to_version
#             row['version_pytables'] = tables.__version__
#             row.update()
#         vtb.flush()
#         return True
#     except:
#         log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
#         return False

def upgrade_0p0p4_to_0p0p5(fileh):
    from_version = "0.0.4"
    to_version = "0.0.5"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))




        #lastly update the version
        vtb = fileh.root.Version
        for row in vtb:  # should be only one
            row['version'] = to_version
            row['version_pytables'] = tables.__version__
            row.update()
        vtb.flush()
        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False



def upgrade_0p0px_to_0p1p0(oldfile_handle,newfile_handle):
    from_version = "0.0.x"
    to_version = "0.1.0"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))

        dtb_new = newfile_handle.root.Detections
        stb_new = newfile_handle.root.CalibratedSpectra
        ltb_new = newfile_handle.root.SpectraLines
        atb_new = newfile_handle.root.Aperture
        ctb_new = newfile_handle.root.CatalogMatch
        etb_new = newfile_handle.root.ExtractedObjects
        xtb_new = newfile_handle.root.ElixerApertures

        dtb_old = oldfile_handle.root.Detections
        stb_old = oldfile_handle.root.CalibratedSpectra
        ltb_old = oldfile_handle.root.SpectraLines
        atb_old = oldfile_handle.root.Aperture
        ctb_old = oldfile_handle.root.CatalogMatch
        try:
            etb_old = oldfile_handle.root.ExtractedObjects #this is a new table
        except:
            pass

        try:
            xtb_old = oldfile_handle.root.ElixerApertures #this is a new table
        except:
            pass

        #new columns in Detections
        for old_row in dtb_old.read():
            new_row = dtb_new.row
            for n in dtb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Detections column failed (%s)"%n)
            new_row.append()
            dtb_new.flush()


        # old_rows = dtb_old.read()
        # for old_row in old_rows:
        #     new_row = dtb_new.row
        #     new_row['detectid'] = old_row['detectid']
        #     new_row['detectname'] =
        #     new_row['elixer_version'] =
        #     new_row['elixer_datetime'] =
        #     new_row['shotid'] =
        #     new_row['obsid'] =
        #     new_row['specid'] =
        #     new_row['ifuslot'] =
        #     new_row['ifuid'] =
        #     new_row['seeing_gaussian'] =
        #     new_row['seeing_moffat'] =
        #     new_row['response'] =
        #     new_row['fieldname'] =
        #     new_row['ra'] =
        #     new_row['dec'] =
        #     new_row['wavelength_obs'] =
        #     new_row['wavelength_obs_err'] =
        #     new_row['flux_line'] =
        #     new_row['flux_line_err'] =
        #     new_row['fwhm_line_aa'] =
        #     new_row['fwhm_line_aa_err'] =
        #     new_row['sn'] =
        #     new_row['sn_err'] =
        #     new_row['chi2'] =
        #     new_row['chi2_err'] =
        #     new_row['continuum_line'] =
        #     new_row['continuum_line_err'] =
        #     new_row['continuum_sdss_g'] =
        #     new_row['continuum_sdss_g_err'] =
        #     new_row['mag_sdss_g'] =
        #     new_row['mag_sdss_g_err'] =
        #     new_row['eqw_rest_lya_line'] =
        #     new_row['eqw_rest_lya_line_err'] =
        #     new_row['eqw_rest_lya_sdss_g'] =
        #     new_row['eqw_rest_lya_sdss_g_err'] =
        #     new_row['plae_line'] =
        #     new_row['plae_line_max'] =
        #     new_row['plae_line_min'] =
        #     new_row['plae_sdss_g'] =
        #     new_row['plae_sdss_g_max'] =
        #     new_row['plae_sdss_g_min'] =
        #     new_row['multiline_flag'] =
        #     new_row['multiline_z'] =
        #     new_row['multiline_rest_w'] =
        #     new_row['multiline_prob'] =
        #     new_row['multiline_raw_score'] =
        #     new_row['multiline_frac_score'] =
        #     new_row['multiline_name'] =
        #
        #
        #     new_row['pseudo_color_flag'] =
        #     new_row['pseudo_color_blue_flux'] =
        #     new_row['pseudo_color_blue_flux_err'] =
        #     new_row['pseudo_color_red_flux'] =
        #     new_row['pseudo_color_red_flux_err'] =
        #     new_row['pseudo_color_rvb_ratio'] =
        #     new_row['pseudo_color_rvb_ratio_err'] =

        #no change to CalibratedSpectra
        stb_new.append(stb_old.read())
        stb_new.flush()

        #no change to SpectraLines
        ltb_new.append(ltb_old.read())
        ltb_new.flush()

        #Aperture ... renames
        old_rows = atb_old.read()
        for old_row in old_rows:
            new_row = atb_new.row
            new_row['detectid'] = old_row['detectid']
            new_row['ra'] = old_row['aperture_ra']
            new_row['dec'] = old_row['aperture_dec']
            new_row['catalog_name'] = old_row['catalog_name']
            new_row['filter_name'] = old_row['filter_name']
            new_row['image_depth_mag'] = old_row['image_depth_mag']
            try: #might not have pixel scale
                new_row['pixel_scale'] = old_row['pixel_scale']
            except:
                pass
            new_row['radius'] = old_row['aperture_radius']
            new_row['mag'] = old_row['aperture_mag']
            new_row['mag_err'] = old_row['aperture_mag_err']
            new_row['aperture_area_pix'] = old_row['aperture_area_pix']
            new_row['sky_area_pix'] = old_row['sky_area_pix']
            new_row['eqw_rest_lya'] = old_row['aperture_eqw_rest_lya']
            new_row['eqw_rest_lya_err'] = old_row['aperture_eqw_rest_lya_err']
            new_row['plae'] = old_row['aperture_plae']
            try: #old record may not have these
                new_row['plae_max'] = old_row['aperture_plae_max']
                new_row['plae_min'] = old_row['aperture_plae_min']
            except:
                pass
            new_row['aperture_cts'] = old_row['aperture_counts']
            new_row['sky_cts'] = old_row['sky_counts']
            new_row['sky_average'] = old_row['sky_average']

            new_row.append()
            atb_new.flush()


        #catalog match
        old_rows = ctb_old.read()
        for old_row in old_rows:
            new_row = ctb_new.row
            new_row['detectid'] = old_row['detectid']
            new_row['ra'] = old_row['cat_ra']
            new_row['dec'] = old_row['cat_dec']
            new_row['catalog_name'] = old_row['catalog_name']
            new_row['filter_name'] = old_row['filter_name']
            new_row['match_num'] = old_row['match_num']
            new_row['separation'] = old_row['separation']
            new_row['prob_match'] = old_row['prob_match']
            new_row['specz'] = old_row['cat_specz']
            new_row['photz'] = old_row['cat_photz']
            new_row['flux'] = old_row['cat_flux']
            new_row['flux_err'] = old_row['cat_flux_err']
            new_row['mag'] = old_row['cat_mag']
            new_row['mag_err'] = old_row['cat_mag_err']
            new_row['eqw_rest_lya'] = old_row['cat_eqw_rest_lya']
            new_row['eqw_rest_lya_err'] = old_row['cat_eqw_rest_lya_err']
            new_row['plae'] = old_row['cat_plae']
            try: #old record may not have these
                new_row['plae_max'] = old_row['cat_plae_max']
                new_row['plae_min'] = old_row['cat_plae_min']
            except:
                pass


            new_row.append()
            ctb_new.flush()


        #ExtractedObjects might not exist
        try:
            etb_new.append(etb_old.read())
            etb_new.flush()
        except:
            pass

        try:
            xtb_new.append(xtb_old.read())
            xtb_new.flush()
        except:
            pass


        flush_all(newfile_handle)
        # close the merge input file
        newfile_handle.close()
        oldfile_handle.close()

        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False




def upgrade_0p1px_to_0p2p0(oldfile_handle,newfile_handle):
    """
    new column in Detections (spurious_reason)
    new column in ExtractedObjects (dist_curve)
    new column in ExtractedObjects (dist_barycntr)
    :param oldfile_handle:
    :param newfile_handle:
    :return:
    """
    from_version = "0.1.x"
    to_version = "0.2.0"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))

        dtb_new = newfile_handle.root.Detections
        stb_new = newfile_handle.root.CalibratedSpectra
        ltb_new = newfile_handle.root.SpectraLines
        atb_new = newfile_handle.root.Aperture
        ctb_new = newfile_handle.root.CatalogMatch
        etb_new = newfile_handle.root.ExtractedObjects
        xtb_new = newfile_handle.root.ElixerApertures

        dtb_old = oldfile_handle.root.Detections
        stb_old = oldfile_handle.root.CalibratedSpectra
        ltb_old = oldfile_handle.root.SpectraLines
        atb_old = oldfile_handle.root.Aperture
        ctb_old = oldfile_handle.root.CatalogMatch
        etb_old = oldfile_handle.root.ExtractedObjects
        xtb_old = oldfile_handle.root.ElixerApertures

        #new columns in Detections
        for old_row in dtb_old.read():
            new_row = dtb_new.row
            for n in dtb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Detections column failed (%s). Default set."%n)
            new_row.append()
            dtb_new.flush()

        #new columns in Detections
        for old_row in etb_old.read():
            new_row = etb_new.row
            for n in etb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Extracted objects column failed (%s). Default set."%n)
            new_row.append()
            dtb_new.flush()


        #no change to CalibratedSpectra
        stb_new.append(stb_old.read())
        stb_new.flush()

        #no change to SpectraLines
        ltb_new.append(ltb_old.read())
        ltb_new.flush()

        #no change to Aperture
        atb_new.append(atb_old.read())
        atb_new.flush()

        #no change to CatalogMatch
        ctb_new.append(ctb_old.read())
        ctb_new.flush()

        #no change to ElixerApertures
        xtb_new.append(xtb_old.read())
        xtb_new.flush()

        flush_all(newfile_handle)
        # close the merge input file
        newfile_handle.close()
        oldfile_handle.close()

        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False




def upgrade_0p2px_to_0p3p0(oldfile_handle,newfile_handle):
    """
    new column in Detections (spurious_reason)
    new column in ExtractedObjects (dist_curve)
    new column in ExtractedObjects (dist_barycntr)
    :param oldfile_handle:
    :param newfile_handle:
    :return:
    """
    from_version = "0.2.x"
    to_version = "0.3.0"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))

        dtb_new = newfile_handle.root.Detections
        stb_new = newfile_handle.root.CalibratedSpectra
        ltb_new = newfile_handle.root.SpectraLines
        atb_new = newfile_handle.root.Aperture
        ctb_new = newfile_handle.root.CatalogMatch
        etb_new = newfile_handle.root.ExtractedObjects
        xtb_new = newfile_handle.root.ElixerApertures

        dtb_old = oldfile_handle.root.Detections
        stb_old = oldfile_handle.root.CalibratedSpectra
        ltb_old = oldfile_handle.root.SpectraLines
        atb_old = oldfile_handle.root.Aperture
        ctb_old = oldfile_handle.root.CatalogMatch
        etb_old = oldfile_handle.root.ExtractedObjects
        xtb_old = oldfile_handle.root.ElixerApertures

        #new columns in Detections
        for old_row in dtb_old.read():
            new_row = dtb_new.row
            for n in dtb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Detections column failed (%s). Default set."%n)
            new_row.append()
            dtb_new.flush()

        #new column added (image_depth_mag)
        for old_row in etb_old.read():
            new_row = etb_new.row
            for n in etb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Extracted Object column failed (%s). Default set."%n)
            new_row.append()
            etb_new.flush()

        #no change to CalibratedSpectra
        stb_new.append(stb_old.read())
        stb_new.flush()

        #no change to SpectraLines
        ltb_new.append(ltb_old.read())
        ltb_new.flush()

        #new column added (image_depth_mag)
        for old_row in atb_old.read():
            new_row = atb_new.row
            for n in atb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Extracted Object column failed (%s). Default set."%n)
            new_row.append()
            atb_new.flush()

        #no change to CatalogMatch
        ctb_new.append(ctb_old.read())
        ctb_new.flush()

        #no change to ElixerApertures
        xtb_new.append(xtb_old.read())
        xtb_new.flush()

        flush_all(newfile_handle)
        # close the merge input file
        newfile_handle.close()
        oldfile_handle.close()

        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False


def upgrade_0p3p0_to_0p3p1(oldfile_handle,newfile_handle):
    """
    #add flags column to Detections
    #add review column to Detections
    #add selected column to CatalogMatch
    #add several columns to ExtracteObjects (fixed_aper_xxx)

    :param oldfile_handle:
    :param newfile_handle:
    :return:
    """

    from_version = "0.3.0"
    to_version = "0.3.1"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))

        dtb_new = newfile_handle.root.Detections
        stb_new = newfile_handle.root.CalibratedSpectra
        ltb_new = newfile_handle.root.SpectraLines
        atb_new = newfile_handle.root.Aperture
        ctb_new = newfile_handle.root.CatalogMatch
        etb_new = newfile_handle.root.ExtractedObjects
        xtb_new = newfile_handle.root.ElixerApertures

        dtb_old = oldfile_handle.root.Detections
        stb_old = oldfile_handle.root.CalibratedSpectra
        ltb_old = oldfile_handle.root.SpectraLines
        atb_old = oldfile_handle.root.Aperture
        ctb_old = oldfile_handle.root.CatalogMatch
        etb_old = oldfile_handle.root.ExtractedObjects
        xtb_old = oldfile_handle.root.ElixerApertures

        #new columns in Detections
        for old_row in dtb_old.read():
            new_row = dtb_new.row
            for n in dtb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("Detections column failed (%s). Default set."%n)
            new_row.append()
            dtb_new.flush()

        #CalibratedSpectra
        for old_row in stb_old.read():
            new_row = stb_new.row
            for n in stb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("CalibratedSpectra column failed (%s). Default set."%n)
            new_row.append()
            stb_new.flush()

        #no change to SpectraLines
        ltb_new.append(ltb_old.read())
        ltb_new.flush()

        #no change to Aperture
        atb_new.append(atb_old.read())
        atb_new.flush()

        #CatalogMatch
        for old_row in ctb_old.read():
            new_row = ctb_new.row
            for n in ctb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("CatalogMatch column failed (%s). Default set."%n)
            new_row.append()
            ctb_new.flush()


        #no change to ExtractedObjects

        for old_row in etb_old.read():
            new_row = etb_new.row
            for n in etb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("ExtractedObjects column failed (%s). Default set."%n)
            new_row.append()

        etb_new.flush()

        #no change to ElixerApertures
        xtb_new.append(xtb_old.read())
        xtb_new.flush()

        flush_all(newfile_handle)

        # close the merge input file
        newfile_handle.close()
        oldfile_handle.close()

        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False


def upgrade_0p3px_to_0p4p0(oldfile_handle,newfile_handle):
    """
    #add flags column to Detections
    #add review column to Detections
    #add selected column to CatalogMatch
    #add several columns to ExtracteObjects (fixed_aper_xxx)

    :param oldfile_handle:
    :param newfile_handle:
    :return:
    """

    from_version = "0.3.1"
    to_version = "0.4.0"

    try:
        log.info("Upgrading %s to %s ..." %(from_version,to_version))

        dtb_new = newfile_handle.root.Detections
        stb_new = newfile_handle.root.CalibratedSpectra
        ltb_new = newfile_handle.root.SpectraLines
        atb_new = newfile_handle.root.Aperture
        ctb_new = newfile_handle.root.CatalogMatch
        etb_new = newfile_handle.root.ExtractedObjects
        xtb_new = newfile_handle.root.ElixerApertures

        dtb_old = oldfile_handle.root.Detections
        stb_old = oldfile_handle.root.CalibratedSpectra
        ltb_old = oldfile_handle.root.SpectraLines
        atb_old = oldfile_handle.root.Aperture
        ctb_old = oldfile_handle.root.CatalogMatch
        etb_old = oldfile_handle.root.ExtractedObjects
        xtb_old = oldfile_handle.root.ElixerApertures

        #no change in Detections
        dtb_new.append(dtb_old.read())
        dtb_new.flush()

        #no change to Calibrated Spectra
        stb_new.append(stb_old.read())
        stb_new.flush()


        #SpectraLines
        for old_row in ltb_old.read():
            new_row = ltb_new.row
            for n in ltb_new.colnames:
                try: #can be missing name (new columns)
                    new_row[n] = old_row[n]
                except:
                    log.debug("SpectraLines column failed (%s). Default set."%n)
            new_row.append()
            ltb_new.flush()


        #no change to Aperture
        atb_new.append(atb_old.read())
        atb_new.flush()

        #CatalogMatch
        ctb_new.append(ctb_old.read())
        ctb_new.flush()

        #no change to ExtractedObjects
        etb_new.append(etb_old.read())
        etb_new.flush()

        #no change to ElixerApertures
        xtb_new.append(xtb_old.read())
        xtb_new.flush()

        flush_all(newfile_handle)

        # close the merge input file
        newfile_handle.close()
        oldfile_handle.close()

        return True
    except:
        log.error("Upgrade failed %s to %s:" %(from_version,to_version),exc_info=True)
        return False




def upgrade_hdf5(oldfile,newfile):
    """
    Primarily here because pytables does not allow for renaming of column names

    :param oldfile:
    :param newfile:
    :return:
    """

    try:
        newfile_handle = get_hdf5_filehandle(newfile,append=False,allow_overwrite=False,must_exist=False)

        if newfile_handle is None:
            print("Unable to create destination file for upgrade_hdf5. File may alread exist.")
            log.info("Unable to create destination file for upgrade_hdf5.")
            return False

        oldfile_handle = get_hdf5_filehandle(oldfile,append=False,allow_overwrite=False,must_exist=True)

        if (oldfile_handle is None):
            print("Unable to source file(s) for upgrade_hdf5.")
            log.info("Unable to open source file(s) for upgrade_hdf5.")
            return False

        old_version = oldfile_handle.root.Version.read()['version'][0].decode()
        if old_version == __version__:
            print("Already at latest version (%s)." %old_version)
            log.info("Already at latest version (%s)." %old_version)
            return False
    except:
        log.error("Exception! in elixer_hdf5::upgrade_hdf5",exc_info=True)


    try:
        done = False
        func_list = []
        max_version = old_version

        while not done:
            if (max_version == '0.0.3') or (max_version == '0.0.4') or (max_version == '0.0.5'):
                func_list.append(upgrade_0p0px_to_0p1p0)
                max_version = "0.1.0"
            elif (max_version == '0.1.0') or (max_version == '0.1.1') or (max_version == '0.1.2'):
                func_list.append(upgrade_0p1px_to_0p2p0)
                max_version = "0.2.0"
            elif (max_version == '0.3.1') or (max_version == '0.3.2'):
                func_list.append(upgrade_0p3px_to_0p4p0)
                max_version = "0.4.0"
            else:
                done = True

        if max_version == __version__:
            for f in func_list:
                result = f(oldfile_handle,newfile_handle)
                if not result:
                    return False

            return True
        else:
            print("No viable upgrade path from %s to %s" %(old_version,__version__))
            log.info("No viable upgrade path from %s to %s" %(old_version,__version__))

    except:
        log.error("Exception! in elixer_hdf5::upgrade_hdf5",exc_info=True)

    return True