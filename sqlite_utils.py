try:
    from elixer import global_config as G
except:
    import global_config as G

import sqlite3
import os.path as op
import gzip

log = G.Global_Logger('sqlite_logger')
log.setlevel(G.logging.DEBUG)


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    #if you pass the file name as :memory: to the connect() function of the sqlite3 module,
    # it will create a new database that resides in the memory (RAM) instead of a database file on dis
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except:
        log.info("Exception creating SQLite connection",exc_info=True)

    return conn

def fetch_blob(db_file,key,tablename="blobtable",keyname="blobname"):
    """

    :param conn:
    :param tablename
    :param keyname:
    :param key:
    :return:
    """
    try:
        conn = create_connection(db_file)

        if conn is None:
            return None

        cursor = conn.cursor()

        sql_read_blob = f"""SELECT * from {tablename} where {keyname} = ?"""

        cursor.execute(sql_read_blob, (str(key),))
        blob = cursor.fetchall()
        cursor.close()
        conn.close()

        if blob is not None:
            if len(blob) == 1:
                return gzip.decompress(blob[0][1]) #dictionary [0][0] is the key, [0][1] is the payload
            elif len(blob) == 0:
                log.info("No matching blob found")
                return None
            else:
                log.info(f"Unexpected number of blobs ({len(blob)}) returned")
                return None
        return blob
    except:
        log.info("Exception fetching SQLite blob", exc_info=True)
        return False



def fetch_zpdf(db_file,key=None,fn=None):
    """

    :param conn:
    :param key: integer , zPDF ID number
    :return:
    """
    try:

        if key is None and fn is None:
            log.debug("Invalid call to fetch_zpdf. Neither key nor fn specified")
            return None
        elif key is None:
            key = int(op.basename(fn).split('_')[-1].split('.')[0].split('ID')[-1])

        conn = create_connection(db_file)

        if conn is None:
            return None

        cursor = conn.cursor()

        sql_read_blob = f"""SELECT zpdf_blob from zpdf_table where zpdf_id = ?"""

        cursor.execute(sql_read_blob, (str(key),))
        blob = cursor.fetchall()
        #get back a list of tuples (each list entry is one row, the tuple is the row, so
        #we want the 1st entry (detectid is unique, so should be one or none, and the second column which is the image)
        cursor.close()
        conn.close()

        if blob is not None:
            if len(blob) == 1:
                return gzip.decompress(blob[0][0])
            elif len(blob) == 0:
                log.info("No matching blob found")
                return None
            else:
                log.info(f"Unexpected number of blobs ({len(blob)}) returned")
                return None

        return blob
    except:
        log.info("Exception fetching SQLite blob", exc_info=True)
        return False

