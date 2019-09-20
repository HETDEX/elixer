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


log = G.Global_Logger('hdf5_logger')
log.setlevel(G.logging.DEBUG)


#make a class for each table
class Version(tables.IsDescription):
    #version table, very basic info
    version = tables.StringCol((16),pos=0) #this version
    version_pytables = tables.StringCol((16),pos=1)


class Detections(tables.IsDescription):
#top level detections summary, one row for each ELiXer/HETDEX detection
    p = 0
    detectid = tables.Int64Col(pos=p); p+=1 #unique HETDEX detection ID 1e9+
    elixer_version = tables.StringCol((16),pos=p); p+=1 #version of elixer that generated this detection report
    elixer_datetime = tables.StringCol((21),pos=p); p+=1 #YYYY-MM-DD hh:mm:ss

    #of the primary fiber ... typically, just three dithers and
    #all from the same observation, so this would apply to all but
    #it can be that this is built up from observations over different date/times
    #this is mostly or entirely redundant with HETDEX HDF5 data (detections or survey)
    shotid = tables.Int64Col(pos=p); p+=1
    obsid = tables.Int32Col(pos=p); p+=1
    specid = tables.StringCol((3),pos=p); p+=1
    ifuslot = tables.StringCol((3),pos=p); p+=1
    ifuid = tables.StringCol((3),pos=p); p+=1
    seeing_gaussian = tables.Float32Col(pos=p); p+=1
    seeing_moffat = tables.Float32Col(pos=p); p+=1
    response = tables.Float32Col(pos=p); p+=1

    #about the detection
    ra = tables.Float32Col(pos=p); p+=1
    dec = tables.Float32Col(pos=p); p+=1
    wavelength_obs = tables.Float32Col(pos=p); p+=1
    wavelength_obs_err = tables.Float32Col(pos=p); p+=1
    flux_line = tables.Float32Col(pos=p); p+=1 #actual flux not flux density
    flux_line_err = tables.Float32Col(pos=p); p+=1
    fwhm_line = tables.Float32Col(pos=p); p+=1
    fwhm_line_err = tables.Float32Col(pos=p); p+=1
    sn = tables.Float32Col(pos=p); p+=1
    sn_err = tables.Float32Col(pos=p); p+=1
    chi2 = tables.Float32Col(pos=p); p+=1
    chi2_err = tables.Float32Col(pos=p); p+=1

    continuum_line = tables.Float32Col(pos=p); p+=1 #continuum from near the line
    continuum_line_err = tables.Float32Col(pos=p); p+=1
    continuum_sdss_g = tables.Float32Col(pos=p); p+=1
    continuum_sdss_g_err = tables.Float32Col(pos=p); p+=1
    mag_sdss_g = tables.Float32Col(pos=p); p+=1
    mag_sdss_g_err = tables.Float32Col(pos=p); p+=1

    eqw_rest_lya_line = tables.Float32Col(pos=p); p+=1
    eqw_rest_lya_line_err = tables.Float32Col(pos=p); p+=1
    eqw_rest_lya_sdss_g = tables.Float32Col(pos=p);p += 1
    eqw_rest_lya_sdss_g_err = tables.Float32Col(pos=p);p += 1
    plae_line = tables.Float32Col(pos=p); p+=1
    plae_sdss_g = tables.Float32Col(pos=p); p+=1

    #ELiXer solution based on extra lines
    multiline_z = tables.Float32Col(pos=p); p += 1
    multiline_rest_w = tables.Float32Col(pos=p); p += 1
    multiline_prob = tables.Float32Col(pos=p); p += 1
    multiline_score = tables.Float32Col(pos=p); p += 1

class SpectraLines(tables.IsDescription):
    p = 0
    detectid = tables.Int64Col(pos=p);p += 1  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(pos=p); p+=1
    type = tables.Int32Col(pos=p); p+=1 # 1 = emission, 0 = unknown, -1 = absorbtion
    flux_line = tables.Float32Col(pos=p); p+=1
    flux_line_err = tables.Float32Col(pos=p); p+=1
    score = tables.Float32Col(pos=p); p+=1
    sn = tables.Float32Col(pos=p); p+=1
    used = tables.BoolCol(pos=p); p+=1 #True if used in the reported multiline solution



class CalibratedSpectra(tables.IsDescription):
    p = 0
    detectid = tables.Int64Col(pos=p); p += 1  # unique HETDEX detection ID 1e9+
    wavelength = tables.Float32Col(1036, pos=p); p+=1
    flux = tables.Float32Col(1036, pos=p); p += 1
    flux_err = tables.Float32Col(1036, pos=p); p+=1

class Aperture(tables.IsDescription):
    #one entry per aperture photometry collected
    p = 0
    detectid = tables.Int64Col(pos=p); p += 1
    catalog_name = tables.StringCol((16),pos=p); p+=1
    filter_name = tables.StringCol((16), pos=p); p+=1
    image_depth_mag = tables.Float32Col(pos=p); p+=1
    aperture_ra = tables.Float32Col(pos=p); p+=1
    aperture_dec = tables.Float32Col(pos=p); p+=1
    aperture_radius = tables.Float32Col(pos=p); p+=1 #in arcsec
    aperture_flux = tables.Float32Col(pos=p); p+=1 #with sky already subtracted
    aperture_flux_err = tables.Float32Col(pos=p);p += 1
    aperture_mag = tables.Float32Col(pos=p); p+=1
    aperture_mag_err = tables.Float32Col(pos=p); p+=1
    aperture_area_pix = tables.Float32Col(pos=p); p+=1 #pixels
    sky_flux = tables.Float32Col(pos=p); p+=1
    sky_flux_err = tables.Float32Col(pos=p); p += 1
    sky_area_pix = tables.Float32Col(pos=p); p+=1 #pixels
    aperture_eqw_rest_lya = tables.Float32Col(pos=p); p+=1
    aperture_eqw_rest_lya_err = tables.Float32Col(pos=p); p += 1
    aperture_plae = tables.Float32Col(pos=p); p+=1


class CatalogMatch(tables.IsDescription):
    # one entry per catalog bid target
    p = 0

    detectid = tables.Int64Col(pos=p);    p += 1
    catalog_name = tables.StringCol((16), pos=p);    p += 1
    separation = tables.Float32Col(pos=p); p+=1 #in arcsec
    prob_match = tables.Float32Col(pos=p); p+=1 #in arcsec
    cat_ra = tables.Float32Col(pos=p); p+=1
    cat_dec = tables.Float32Col(pos=p); p+=1
    cat_specz = tables.Float32Col(pos=p); p+=1
    cat_photz = tables.Float32Col(pos=p); p+=1
    cat_flux = tables.Float32Col(pos=p); p+=1
    cat_flux_err = tables.Float32Col(pos=p); p+=1
    cat_mag = tables.Float32Col(pos=p); p+=1
    cat_mag_err = tables.Float32Col(pos=p); p+=1
    cat_eqw_rest_lya = tables.Float32Col(pos=p); p+=1
    cat_eqw_rest_lya_err = tables.Float32Col(pos=p); p+=1
    cat_plae = tables.Float32Col(pos=p); p+=1

    #maybe add in the PDF of the photz ... not sure how big
    #to make the columns ... needs to be fixed, but might
    #vary catalog to catalog
    #cat_photz_pdf_z = tables.Float32Col(1036, pos=p); p+=1
    #cat_photz_pdf_p = tables.Float32Col(1036, pos=p); p+=1




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

        vtbl = fileh.root.Version
        dtbl = fileh.root.Detections


        vtbl.flush()
        dtbl.flush()

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

            fileh.create_table(fileh.root, 'Version', Version,
                               'ELiXer Detection Version Table')

            vtbl = fileh.root.Version
            row = vtbl.row
            row['version'] = __version__
            row['version_pytables'] = tables.__version__
            row.append()

            vtbl.flush()

            fileh.create_table(fileh.root, 'Detections', Detections,
                               'ELiXer Detection Summary Table')

            #todo: create all other tables

    except:
        log.error("Exception! in elixer_hdf5::get_hdf5_filehandle().",exc_info=True)

    return fileh



def build_elixer_hdf5(fname,hd_list=[]):
    #build a new HDF5 file from the current active run
    #this is like the old ELiXer creating _cat.txt and _fib.txt

    fileh = get_hdf5_filehandle(fname,append=False)

    if fileh is None:
        log.error("Unable to build ELiXer catalog.")
        return


    for h in hd_list:  # iterate over all hetdex (hd) collections
        for e in h.emis_list: #for each detection in each hd collection
            pass

    flush_all(fileh)
    fileh.close()


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


        flush_all(fileh)
        #close the merge input file
        merge_fh.close()

    flush_all(fileh)
    fileh.close()