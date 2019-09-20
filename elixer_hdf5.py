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
    detectid = tables.Int64Col(pos=0) #unique HETDEX detection ID 1e9+
    elixer_version = tables.StringCol((16),pos=1) #version of elixer that generated this detection report
    elixer_datetime = tables.StringCol((21),pos=1) #YYYY-MM-DD hh:mm:ss



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