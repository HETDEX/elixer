"""
open all 3 db files
get list of inserted detectIDs that are in all 3
move those corresponding files out of all_pngs to done_pngs

"""
#todo: right now this is just a clone of make_report_db

#from hetdex_api import sqlite_utils as sql
import sqlite3 as sql
import numpy as np
#import argparse
import shutil
import os
import glob


# def parse_commandline(auto_force=False):
#     desc = "move already processed image files"
#     #parser = argparse.ArgumentParser(description=desc)
#     #parser.add_argument('--prefix', help="report prefix", required=True, type=str)
#     #parser.add_argument('--img_dir', help="Directory with the images", required=True, type=str)
#     #parser.add_argument('--img_name', help="Wildcard image name", required=True, type=str)
#     #parser.add_argument('--move_dir', help="Directory to move processed images", required=True, type=str)
#     #parser.add_argument('--mv2dir', help="mv to directory (leaves in cwd if not provided)",required=False,type=str)
#
#     args = parser.parse_args()
#
#     return args

def get_db_connection(fn,readonly=True):
    """
    return a SQLite3 databse connection object for the provide databse filename

    assumes file exists (will trap exception if not and return None)
    :param fn:
    :return: None or connection object
    """

    conn = None
    try:
        if fn is not None:
            if readonly:
                conn = sqlite3.connect("file:" +fn + "?mode=ro",uri=True)
            else:
                conn = sqlite3.connect(fn)
    except Error as e:
        print(e)

    return conn

def fetch_all_detectids(conn):
    """
    wrapper just to make code cleaner

    :param detectid:
    :param report_type:
    :return:
    """

    try:
        if type(conn) == sqlite3.Connection:
            cursor = conn.cursor()
            #there should be exactly one row
            sql_statement = """SELECT detectid from report"""
            cursor.execute(sql_statement)
            rows_detections = cursor.fetchall()
            cursor.close()
            return [r[0] for r in rows_detections]
    except Exception as e:
        print(e)

def main():
    #go blind?
    #args = parse_commandline()

    #make sure directories exist
    if os.path.exists("all_pngs"):
        if os.path.exists("done_pngs"):
            pass #all good
        else:
            print("done_pngs not found")
            exit(-1)
    else:
        print("all_pngs not found")
        exit(-1)

    dets_std = [] #standard reports
    dets_nei = [] #neighbors
    dets_mini = [] #mini
    dets_intersect = []

    db_std = glob.glob("elixer_reports_*[0-9].db") #should be exactly one
    db_nei = glob.glob("elixer_reports_[0-9]*_nei.db") #should be exactly one
    db_mini = glob.glob("elixer_reports_[0-9]*_mini.db") #should be exactly one

    if len(db_std) != len(db_nei) != len(db_mini) != 1:
        print("Error locating databases")
        exit(-1)

    db_std = db_std[0]
    db_nei = db_nei[0]
    db_mini = db_mini[0]

    #open the (3) databases? (one at a time is fine) and get lists of detectIDs
    conn = get_db_connection(db_std)
    dets_std = fetch_all_detectids(conn)
    conn.close()

    conn = get_db_connection(db_nei)
    dets_nei = fetch_all_detectids(conn)
    conn.close()

    conn = get_db_connection(db_mini)
    dets_mini = fetch_all_detectids(conn)
    conn.close()

    dets_std = np.array(dets_std)
    dets_nei = np.array(dets_nei)
    dets_mini = np.array(dets_mini)

    #keep those in all 3 lists
    dets_intersect = np.intersect1d(dets_std,dets_nei)
    dets_intersect = np.intersect1d(dets_intersect,dets_mini)

    #move those files
    exception_count = 0
    print(f"Moving {len(dets_intersect)} x3 detection files ... ")
    for d in dets_intersect:
        srcfn = f"all_pngs/{d}.png"
        destfn = f"done_pngs/{d}.png"

        try:
            shutil.move(srcfn, destfn)
        except Exception as e:
            print(e)
            exception_count += 1

        srcfn = f"all_pngs/{d}_nei.png"
        destfn = f"done_pngs/{d}_nei.png"

        try:
            shutil.move(srcfn, destfn)
        except Exception as e:
            print(e)
            exception_count += 1

        srcfn = f"all_pngs/{d}_mini.png"
        destfn = f"done_pngs/{d}_mini.png"

        try:
            shutil.move(srcfn, destfn)
        except Exception as e:
            print(e)
            exception_count += 1

        if exception_count > 100:
            print("too many exceptions, something wrong")
            exit(-1)

    print(f"complete")

if __name__ == '__main__':
    main()