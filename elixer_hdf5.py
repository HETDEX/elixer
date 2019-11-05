"""
ELiXer HDF5 utilities ...
create ELiXer catalog(s) as HDF5
merge existing ELiXer catalogs
"""


__version__ = '0.0.2' #catalog version ... can merge if version numbers are the same or in special circumstances

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

log = G.Global_Logger('hdf5_logger')
log.setlevel(G.logging.DEBUG)


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
    seeing_gaussian = tables.Float32Col(dflt=UNSET_FLOAT)
    seeing_moffat = tables.Float32Col(dflt=UNSET_FLOAT)
    response = tables.Float32Col(dflt=UNSET_FLOAT)
    fieldname = tables.StringCol(itemsize=32)

    #about the detection
    ra = tables.Float32Col(dflt=UNSET_FLOAT,pos=4)
    dec = tables.Float32Col(dflt=UNSET_FLOAT,pos=5)
    wavelength_obs = tables.Float32Col(dflt=UNSET_FLOAT,pos=6)
    wavelength_obs_err = tables.Float32Col(dflt=UNSET_FLOAT,pos=7)
    flux_line = tables.Float32Col(dflt=UNSET_FLOAT) #actual flux not flux density
    flux_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    fwhm_line_aa = tables.Float32Col(dflt=UNSET_FLOAT)
    fwhm_line_aa_err = tables.Float32Col(dflt=UNSET_FLOAT)
    sn = tables.Float32Col(dflt=UNSET_FLOAT)
    sn_err = tables.Float32Col(dflt=UNSET_FLOAT)
    chi2 = tables.Float32Col(dflt=UNSET_FLOAT)
    chi2_err = tables.Float32Col(dflt=UNSET_FLOAT)

    continuum_line = tables.Float32Col(dflt=UNSET_FLOAT) #continuum from near the line
    continuum_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    continuum_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT)
    continuum_sdss_g_err = tables.Float32Col(dflt=UNSET_FLOAT)
    mag_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT)
    mag_sdss_g_err = tables.Float32Col(dflt=UNSET_FLOAT)

    eqw_rest_lya_line = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT)
    eqw_rest_lya_sdss_g_err = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_line = tables.Float32Col(dflt=UNSET_FLOAT)
    plae_sdss_g = tables.Float32Col(dflt=UNSET_FLOAT)

    #ELiXer solution based on extra lines
    multiline_flag = tables.BoolCol(dflt=False) #True if s a single "good" solution
    multiline_z = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_rest_w = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_prob = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_raw_score = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_frac_score = tables.Float32Col(dflt=UNSET_FLOAT)
    multiline_name = tables.StringCol(itemsize=16)

    pseudo_color = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_min = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_max = tables.Float32Col(dflt=UNSET_FLOAT)
    pseudo_color_flag = tables.Int64Col(dflt=0)


class SpectraLines(tables.IsDescription):
    detectid = tables.Int64Col(pos=0)  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(dflt=UNSET_FLOAT)
    type = tables.Int32Col(dflt=UNSET_INT) # 1 = emission, 0 = unknown, -1 = absorbtion
    flux_line = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    score = tables.Float32Col(dflt=UNSET_FLOAT)
    sn = tables.Float32Col(dflt=UNSET_FLOAT)
    used = tables.BoolCol(dflt=False) #True if used in the reported multiline solution
    sol_num = tables.Int32Col(dflt=-1) #-1 is unset, 0 is simple scan, no solution, 1+ = solution number in
                                       #decreasing score order




class CalibratedSpectra(tables.IsDescription):
    detectid = tables.Int64Col(pos=0)  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(shape=(1036,) )
    flux = tables.Float32Col(shape=(1036,) )
    flux_err = tables.Float32Col(shape=(1036,) )

class Aperture(tables.IsDescription):
    #one entry per aperture photometry collected
    detectid = tables.Int64Col(pos=0)
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16)
    image_depth_mag = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_ra = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_dec = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_radius = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec
    aperture_mag = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_mag_err = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_area_pix = tables.Float32Col(dflt=UNSET_FLOAT) #pixels
    sky_area_pix = tables.Float32Col(dflt=UNSET_FLOAT) #pixels
    aperture_eqw_rest_lya = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_eqw_rest_lya_err = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_plae = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_counts = tables.Float32Col(dflt=UNSET_FLOAT)
    sky_counts = tables.Float32Col(dflt=UNSET_FLOAT)
    sky_average = tables.Float32Col(dflt=UNSET_FLOAT)


class CatalogMatch(tables.IsDescription):
    # one entry per catalog bid target
    detectid = tables.Int64Col(pos=0)
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16)
    match_num = tables.Int32Col(dflt=-1)
    separation = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec
    prob_match = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec
    cat_ra = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_dec = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_specz = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_photz = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_flux = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_mag = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_mag_err = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_eqw_rest_lya = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_eqw_rest_lya_err = tables.Float32Col(dflt=UNSET_FLOAT)
    cat_plae = tables.Float32Col(dflt=UNSET_FLOAT)

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
        if existing_version != __version__:
            return False, existing_version
        else:
            return True, existing_version
    except:
        log.error("Exception! in elixer_hdf5::version_match().",exc_info=True)
        return False, None

    return False, None



def flush_all(fileh):

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

        #remove (old) index if exists
        #vtb does not have or need an index
        dtb.cols.detectid.remove_index()
        ltb.cols.detectid.remove_index()
        stb.cols.detectid.remove_index()
        atb.cols.detectid.remove_index()
        ctb.cols.detectid.remove_index()

        #create (new) index
        # vtb does not have or need an index
        dtb.cols.detectid.create_csindex()
        ltb.cols.detectid.create_csindex()
        stb.cols.detectid.create_csindex()
        atb.cols.detectid.create_csindex()
        ctb.cols.detectid.create_csindex()

        #vtb.flush() # no need to re-flush vtb
        dtb.flush()
        ltb.flush()
        stb.flush()
        atb.flush()
        ctb.flush()

    return



def get_hdf5_filehandle(fname,append=False,allow_overwrite=True,must_exist=False):
    """
    Return a file handle to work on. Create if does not exist, return existing handle if already present and versions
    are compatible (and append is requested).

    :param fname:
    :param append:
    :return:
    """

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
                               'ELiXer Detection Summary Table')

            fileh.create_table(fileh.root, 'SpectraLines', SpectraLines,
                               'ELiXer Identified SpectraLines Table')

            fileh.create_table(fileh.root, 'CalibratedSpectra', CalibratedSpectra,
                               'HETDEX Flux Calibrated, PSF Weighted Summed Spectra Table')

            fileh.create_table(fileh.root, 'Aperture', Aperture,
                               'ELiXer Aperture Photometry Table')

            fileh.create_table(fileh.root, 'CatalogMatch', CatalogMatch,
                               'ELiXer Catalog Matched Objected Table')

            #todo: any actual images tables? (imaging cutouts, 2D fibers, etc)??

    except:
        log.error("Exception! in elixer_hdf5::get_hdf5_filehandle().",exc_info=True)

    return fileh


def append_entry(fileh,det):
    """

    :param fileh: file handle to the HDF5 file
    :param det: ELiXer DetObj
    :return:
    """
    try:
        #get tables
        dtb = fileh.root.Detections

        q_detectid = det.hdf5_detectid
        rows = dtb.read_where("detectid==q_detectid")

        if rows.size > 0:
            log.info("Detection (%d) already exists in HDF5. Skipping." %(q_detectid))
            return

        stb = fileh.root.CalibratedSpectra
        ltb = fileh.root.SpectraLines
        atb = fileh.root.Aperture
        ctb = fileh.root.CatalogMatch


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
        row['seeing_gaussian'] = det.survey_fwhm_gaussian
        row['seeing_moffat'] = det.survey_fwhm_moffat
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

        if det.estflux_h5 is not None:
            row['flux_line'] = det.estflux_h5 * G.HETDEX_FLUX_BASE_CGS
            row['flux_line_err'] = det.estflux_h5_unc * G.HETDEX_FLUX_BASE_CGS
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


        row['continuum_sdss_g'] = det.sdss_cgs_cont
        row['continuum_sdss_g_err'] = det.sdss_cgs_cont_unc
        row['mag_sdss_g'] = det.sdss_gmag
        if det.sdss_gmag_unc is not None:
            row['mag_sdss_g_err'] = det.sdss_gmag_unc

        _lya_1pz = det.w / G.LyA_rest #no sense is doing -1.0 then +1.0

        if det.eqw_line_obs is not None:
            row['eqw_rest_lya_line'] = det.eqw_line_obs / _lya_1pz
            row['eqw_rest_lya_line_err'] = det.eqw_line_obs_unc / _lya_1pz
        else:
            row['eqw_rest_lya_line'] = det.eqw_obs / _lya_1pz
            row['eqw_rest_lya_line_err'] = det.eqw_obs_unc / _lya_1pz
        #hetdex line flux / sdss continuum flux

        if det.eqw_sdss_obs is not None:
            row['eqw_rest_lya_sdss_g'] = det.eqw_sdss_obs / _lya_1pz
        if det.eqw_sdss_obs_unc / _lya_1pz is not None:
            row['eqw_rest_lya_sdss_g_err'] = det.eqw_sdss_obs_unc / _lya_1pz

        row['plae_line'] = det.p_lae_oii_ratio
        row['plae_sdss_g'] = det.sdss_gmag_p_lae_oii_ratio
        #
        if (det.spec_obj is not None) and (det.spec_obj.solutions is not None) and (len(det.spec_obj.solutions) > 0):
            row['multiline_flag'] = det.multiline_z_minimum_flag
            row['multiline_z'] = det.spec_obj.solutions[0].z
            row['multiline_rest_w'] = det.spec_obj.solutions[0].emission_line.w_rest
            row['multiline_name'] = det.spec_obj.solutions[0].emission_line.name

            row['multiline_raw_score'] = det.spec_obj.solutions[0].score
            row['multiline_frac_score'] = det.spec_obj.solutions[0].frac_score
            row['multiline_prob'] = det.spec_obj.solutions[0].prob_real
            #?? other lines ... other solutions ... move into a separate table ... SpectraLines table

        if (det.rvb is not None):
            row['pseudo_color'] = det.rvb['color']
            row['pseudo_color_min'] = det.rvb['color_range'][0]
            row['pseudo_color_max'] = det.rvb['color_range'][1]
            row['pseudo_color_flag'] = det.rvb['flag']

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
        row.append()
        stb.flush()


        #############################
        #ELiXer Found Spectral Lines Table
        #############################


        for line in det.spec_obj.all_found_lines:
            row = ltb.row
            row['detectid'] = det.hdf5_detectid
            row['sol_num'] = 0
            if line.absorber:
                row['type'] = -1
            else:
                row['type'] = 1

            row['wavelength'] = line.fit_x0
            if line.mcmc_line_flux_tuple is not None:
                row['flux_line'] = line.mcmc_line_flux
                row['flux_line_err']  = 0.5 * (abs(line.mcmc_line_flux_tuple[1]) + abs(line.mcmc_line_flux_tuple[2]))

            else:
                row['flux_line'] = line.fit_line_flux
                row['flux_line_err'] = line.fit_rmse

            row['score'] = line.line_score
            if line.mcmc_snr > 0:
                row['sn'] = line.mcmc_snr
            else:
                row['sn'] = line.snr
            row['used'] = False #these are all found lines, may or may not be in solution

            row.append()
            ltb.flush()

        sol_num = 0
        for sol in det.spec_obj.solutions:
            for line in sol.lines:
                row = ltb.row
                row['detectid'] = det.hdf5_detectid
                sol_num += 1
                row['sol_num'] = sol_num
                if line.absorber:
                    row['type'] = -1
                else:
                    row['type'] = 1

                row['wavelength'] = line.w_obs
                row['flux_line'] = line.flux
                row['flux_line_err'] = 0 #should find the "found" line version to get this

                row['score'] = line.line_score
                row['sn'] = line.snr
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
                row['filter_name'] = d['filter_name']
                #row['image_depth_mag'] = ??
                row['aperture_ra'] = d['ra']
                row['aperture_dec'] = d['dec']
                row['aperture_radius'] = d['radius']
                row['aperture_mag']=d['mag']
                #row['aperture_mag_err'] = d['']
                row['aperture_area_pix'] = d['area_pix']
                row['sky_area_pix'] = d['sky_area_pix']
                row['aperture_counts'] = d['aperture_counts']
                row['sky_counts'] = d['sky_counts']
                row['sky_average'] = d['sky_average']
                row['aperture_eqw_rest_lya'] = d['aperture_eqw_rest_lya']
                row['aperture_plae'] = d['aperture_plae']

                row.append()
                atb.flush()

        #################################
        #Catalog Match table
        ################################
        match_num = 0
        for d in det.bid_target_list:
            if -90.0 < d.bid_dec < 90.0: #666 or 181 is the aperture info
                match_num += 1
                #todo: multiple filters
                #for f in d.filters: #just using selected "best" filter
                row = ctb.row
                row['detectid'] = det.hdf5_detectid
                row['match_num'] = match_num
                row['catalog_name'] = d.catalog_name
                row['filter_name'] = d.bid_filter
                row['separation'] = d.distance
                row['prob_match'] = d.prob_match
                row['cat_ra'] = d.bid_ra
                row['cat_dec'] = d.bid_dec
                if (d.spec_z is not None) and (d.spec_z >= 0.0):
                    row['cat_specz'] = d.spec_z
                if (d.phot_z is not None) and (d.phot_z >= 0.0):
                    row['cat_photz'] = d.phot_z
                row['filter_name'] = d.bid_filter
                row['cat_flux'] = d.bid_flux_est_cgs
                row['cat_flux_err'] = d.bid_flux_est_cgs_unc
                row['cat_mag'] = d.bid_mag
                row['cat_mag_err'] = 0.5 * (abs(d.bid_mag_err_bright) + abs(d.bid_mag_err_faint))
                row['cat_plae'] = d.p_lae_oii_ratio

                if d.bid_ew_lya_rest is not None:
                    row['cat_eqw_rest_lya'] = d.bid_ew_lya_rest
                    if d.bid_ew_lya_rest_err is not None:
                        row['cat_eqw_rest_lya_err'] = d.bid_ew_lya_rest_err

                row.append()
                ctb.flush()

    except:
        log.error("Exception! in elixer_hdf5::append_entry",exc_info=True)

    return


def build_elixer_hdf5(fname,hd_list=[]):
    #build a new HDF5 file from the current active run
    #this is like the old ELiXer creating _cat.txt and _fib.txt

    fileh = get_hdf5_filehandle(fname,append=False)

    if fileh is None:
        print("Unable to build ELiXer catalog.")
        log.error("Unable to build ELiXer catalog.")
        return

    for h in hd_list:  # iterate over all hetdex (hd) collections
        for e in h.emis_list: #for each detection in each hd collection
            append_entry(fileh,e)


    flush_all(fileh)
    fileh.close()
    print("File written: %s" %fname)
    log.info("File written: %s" %fname)

def extend_elixer_hdf5(fname,hd_list=[]):
    #build a new HDF5 file from the current active run
    #this is like the old ELiXer creating _cat.txt and _fib.txt

    fileh = get_hdf5_filehandle(fname,append=True)

    if fileh is None:
        print("Unable to build ELiXer catalog.")
        log.error("Unable to build ELiXer catalog.")
        return

    for h in hd_list:  # iterate over all hetdex (hd) collections
        for e in h.emis_list: #for each detection in each hd collection
            append_entry(fileh,e)


    flush_all(fileh)
    fileh.close()
    print("File written: %s" %fname)
    log.info("File written: %s" %fname)



def merge_unique(newfile,file1,file2):
    """
    Merge, detectID by detectID file1 and file2 into newfile, keeping only the most recent detectID if
    there are duplicates.

    :param newfile: file name of new file to create as merge of file1 and file2
    :param file1:  either of the two files to merge
    :param file2:  other file to merge
    :return:
    """

    try:
        newfile_handle = get_hdf5_filehandle(newfile,append=False,allow_overwrite=False,must_exist=False)

        if newfile_handle is None:
            log.info("Unable to create destination file for merge_unique.")
            return False

        file1_handle = get_hdf5_filehandle(file1,append=False,allow_overwrite=False,must_exist=True)
        file2_handle = get_hdf5_filehandle(file2, append=False, allow_overwrite=False, must_exist=True)

        if (file1_handle is None) or (file2_handle is None):
            log.info("Unable to open source file(s) for merge_unique.")
            return False
    except:
        log.error("Exception! in elixer_hdf5::merge_unique",exc_info=True)

    #todo: enforce version matching?? since creating new file, these should be backward compatible with defaults for
    #todo: missing columns

    try:
        dtb_new = newfile_handle.root.Detections
        stb_new = newfile_handle.root.CalibratedSpectra
        ltb_new = newfile_handle.root.SpectraLines
        atb_new = newfile_handle.root.Aperture
        ctb_new = newfile_handle.root.CatalogMatch

        dtb1 = file1_handle.root.Detections
        dtb2 = file2_handle.root.Detections

        detectids = dtb1.read()['detectid']
        detectids = np.concatenate((detectids,dtb2.read()['detectid']))

        detectids = sorted(detectids)

        log.debug("Merging %d detections ..." %len(detectids))

        for d in detectids:
            try:
                source_h = None

                date1 = dtb1.read_where('detectid==d')['elixer_datetime']
                date2 = dtb2.read_where('detectid==d')['elixer_datetime']
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


                if source_h is None:
                    continue

                dtb_src = source_h.root.Detections
                stb_src = source_h.root.CalibratedSpectra
                ltb_src = source_h.root.SpectraLines
                atb_src = source_h.root.Aperture
                ctb_src = source_h.root.CatalogMatch

                dtb_new.append(dtb_src.read_where("(detectid==d) & (elixer_datetime==q_date)"))
                #unfortunately, have to assume following data is unique
                stb_new.append(stb_src.read_where("(detectid==d)"))
                ltb_new.append(ltb_src.read_where("(detectid==d)"))
                atb_new.append(atb_src.read_where("(detectid==d)"))
                ctb_new.append(ctb_src.read_where("(detectid==d)"))

                #flush_all(newfile_handle) #don't think we need to flush every time

            except:
                log.error("Exception! merging detectid (%d)" %d,exc_info=True)
        # end for loop

        flush_all(newfile_handle)
        newfile_handle.close()
        file2_handle.close()
        file1_handle.close()

    except:
        log.error("Exception! conducting merge in elixer_hdf5::merge_unique", exc_info=True)
        return False

    return True


def merge_elixer_hdf5_files(fname,flist=[]):
    """

    :param fname: the output (final/merged) HDF5 file
    :param flist:  list of all files to merge
    :return: None or filename
    """
    #merging existing distinct HDF5 files w/o new additions from an active run
    fileh = get_hdf5_filehandle(fname,append=True)

    if fileh is None:
        log.error("Unable to merge ELiXer catalogs.")
        return None

    #set up new HDF5 tables (into which to append)
    dtb = fileh.root.Detections

    stb = fileh.root.CalibratedSpectra
    ltb = fileh.root.SpectraLines
    atb = fileh.root.Aperture
    ctb = fileh.root.CatalogMatch

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

        flush_all(fileh)
        #close the merge input file
        merge_fh.close()

    flush_all(fileh)
    fileh.close()
    return fname