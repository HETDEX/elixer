"""
ELiXer HDF5 utilities ...
create ELiXer catalog(s) as HDF5
merge existing ELiXer catalogs
"""


__version__ = '0.0.1' #catalog version ... can merge if version numbers are the same or in special circumstances

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

class SpectraLines(tables.IsDescription):
    detectid = tables.Int64Col(pos=0)  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(dflt=UNSET_FLOAT)
    type = tables.Int32Col(dflt=UNSET_INT) # 1 = emission, 0 = unknown, -1 = absorbtion
    flux_line = tables.Float32Col(dflt=UNSET_FLOAT)
    flux_line_err = tables.Float32Col(dflt=UNSET_FLOAT)
    score = tables.Float32Col(dflt=UNSET_FLOAT)
    sn = tables.Float32Col(dflt=UNSET_FLOAT)
    used = tables.BoolCol(dflt=False) #True if used in the reported multiline solution
    #todo: solutionID ? (zero indexed [0].[1] ...



class CalibratedSpectra(tables.IsDescription):
    detectid = tables.Int64Col(pos=0)  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(shape=(1036,) )
    flux = tables.Float32Col(shape=(1036,) )
    flux_err = tables.Float32Col(shape=(1036,) )

class Aperture(tables.IsDescription):
    #one entry per aperture photometry collected
    detectid = tables.Int64Col(pos=0)
    catalog_name = tables.StringCol(itemsize=16)
    filter_name = tables.StringCol(itemsize=16 )
    image_depth_mag = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_ra = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_dec = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_radius = tables.Float32Col(dflt=UNSET_FLOAT) #in arcsec
    aperture_flux = tables.Float32Col(dflt=UNSET_FLOAT) #with sky already subtracted
    aperture_flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_mag = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_mag_err = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_area_pix = tables.Float32Col(dflt=UNSET_FLOAT) #pixels
    sky_flux = tables.Float32Col(dflt=UNSET_FLOAT)
    sky_flux_err = tables.Float32Col(dflt=UNSET_FLOAT)
    sky_area_pix = tables.Float32Col(dflt=UNSET_FLOAT) #pixels
    aperture_eqw_rest_lya = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_eqw_rest_lya_err = tables.Float32Col(dflt=UNSET_FLOAT)
    aperture_plae = tables.Float32Col(dflt=UNSET_FLOAT)


class CatalogMatch(tables.IsDescription):
    # one entry per catalog bid target
    detectid = tables.Int64Col(pos=0)
    catalog_name = tables.StringCol(itemsize=16)
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
            self.status = -1
            log.error("Problem loading Version table ...")
            return False, None

        existing_version = rows[0]['version'].decode()
        if existing_version != __version__:
            return False, __version__
        else:
            return True, __version__
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

    return



def get_hdf5_filehandle(fname,append=False):
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
            if append:
                #log.debug("ELiXer HDF5 exists (%s). Will append if versions allow." %(fname))
                fileh = tables.open_file(fname, 'a', 'ELiXer Detection Catalog')
                #check the version

                version_okay, existing_version = version_match(fileh)
                if not version_okay:
                    if existing_version is None:
                        existing_version  = 'unknown'

                    log.error('ELiXer HDF5 Catalog (%s) already exists and does not match the current version. (%s != %s)'
                              %(fname,existing_version,__version__))
                    fileh.close()
                    return None
                else:
                    log.debug("ELiXer HDF5 exists (%s). Versions match (%s), will append." %(fname,__version__))
                    return fileh
            else:
                make_new = True
                log.info("ELiXer HDF5 exists (%s). Append not requested. Will overwrite." % (fname))
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

        row.append()


        #############################
        #Calibrated Spectra Table
        #############################
        row = stb.row

        row['detectid'] = det.hdf5_detectid
        row['wavelength'] = det.sumspec_wavelength[:]
        row['flux'] = det.sumspec_flux[:]
        row['flux_err'] = det.sumspec_fluxerr[:]
        row.append()


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
           #todo: build up the tables here
            append_entry(fileh,e)


    flush_all(fileh)
    fileh.close()
    print("File written: %s" %fname)
    log.info("File written: %s" %fname)


def merge_elixer_hdf5_files(fname,flist=[]):
    """

    :param fname: the output (final/merged) HDF5 file
    :param flist:  list of all files to merge
    :return:
    """
    #merging existing distinct HDF5 files w/o new additions from an active run
    fileh = get_hdf5_filehandle(fname,append=True)

    if fileh is None:
        log.error("Unable to merge ELiXer catalogs.")
        return

    for f in flist:
        if f == fname: #could be the output file is one of those to merge
            continue #just skip and move on

        merge_fh = get_hdf5_filehandle(f,append=True)

        if merge_fh is None:
            log.error("Unable to merge: %s" %(f))
            continue

        #todo: merge stuff ... explicit reads then writes?
        #todo: ???? can we load an entire table from merge_fh as an object and append to fileh??

        #example:
        # elif args.mergedir:
        # files = sorted(glob.glob(op.join(args.mergedir, 'detect*.h5')))
        #
        # detectid_max = 1
        #
        # for file in files:
        #     fileh_i = tb.open_file(file, 'r')
        #     tableMain_i = fileh_i.root.Detections.read()
        #     tableFibers_i = fileh_i.root.Fibers.read()
        #     tableSpectra_i = fileh_i.root.Spectra.read()
        #
        #     tableMain_i['detectid'] += detectid_max
        #     tableFibers_i['detectid'] += detectid_max
        #     tableSpectra_i['detectid'] += detectid_max
        #
        #     tableMain.append(tableMain_i)
        #     tableFibers.append(tableFibers_i)
        #     tableSpectra.append(tableSpectra_i)
        #
        #     detectid_max = np.max(tableMain.cols.detectid[:]) - index_buff
        #
        #     fileh_i.close()





        flush_all(fileh)
        #close the merge input file
        merge_fh.close()

    flush_all(fileh)
    fileh.close()