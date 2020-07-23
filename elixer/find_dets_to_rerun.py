"""

Similar to clean_for_recovery, but this script does not delete anything.
It looks for detections in the entire data release and returns a list of detections that need to be re-run because they are:
1) missing (no entry in the report db)
2) have no imaging (empty Aperture table in the Elixer HDF5)
3) have no neighborhood map (no entry in the *_nei.db)
4) have no mini png (no entry in the *_mini.db)

"""

import sys
import numpy as np
import tables
import glob
import os
from hetdex_api.config import HDRconfig
import sqlite3
#from hetdex_api import sqlite_utils as sql
HDRVERSION = "hdr2.1"
which_catalog = 0 #0 = standard, 6 = broad, 9 = continuum
db_path = "/data/03261/polonius/hdr2.1.run/detect/image_db/"

if which_catalog == 0:
    STARTID = 2100000000
    STOPID =  2103000000  #3 million (overkill)
    report_prefix = "210"
elif  which_catalog == 6:
    STARTID = 2160000000
    STOPID =  2161000000 #1 million (overkill)
    report_prefix = "216"
elif  which_catalog == 9:
    STARTID = 2190000000
    STOPID =  2191000000 #1 million (overkill)
    report_prefix = "219"
else:
    print("Bad catalog selected",which_catalog)

AUTO = False

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

if "--start" in args: #overide default if specified on command line
    try:
        i = args.index("--start")
        if i != -1:
            STARTID = int(sys.argv[i + 1])
            AUTO = True
    except:
        pass

if "--stop" in args: #overide default if specified on command line
    try:
        i = args.index("--stop")
        if i != -1:
            STOPID = int(sys.argv[i + 1])
            AUTO = True
    except:
        pass




#todo: make configurable (which hdr version)
cfg = HDRconfig(survey=HDRVERSION)

check_nei = False
check_mini = False
check_imaging = False


if not AUTO:
    i = input("Check for no imaging (y/n)?")
    if len(i) > 0 and i.upper() == "Y":
        check_imaging = True

    i = input("Check for nei.png (y/n)?")
    if len(i) > 0 and i.upper() == "Y":
        check_nei = True

    i = input("Check for mini.png (y/n)?")
    if len(i) > 0 and i.upper() == "Y":
        check_mini = True
else:
    check_nei = True
    check_mini = True
    check_imaging = True


print("Reading detecth5 file ...")
if which_catalog == 0:
    hetdex_h5 = tables.open_file(cfg.detecth5,"r")
elif which_catalog == 6:
    hetdex_h5 = tables.open_file(cfg.detectbroadh5,"r")
elif which_catalog == 9:
    hetdex_h5 = tables.open_file(cfg.contsourceh5,"r")
else:
    print("Bad catalog selected:",which_catalog)
dtb = hetdex_h5.root.Detections
alldets = dtb.read(field="detectid")
hetdex_h5.close()

#this is slow enough, that the prints don't make an impact
#and they are a good progress indicator
sel = np.where(alldets >= STARTID)
alldets = alldets[sel]
sel = np.where(alldets <= STOPID)
alldets = alldets[sel]

#get a unique set of prefixes for the start-stop range
#and later only read in the dbs that correspond to this list
prefix = list(set((np.array(alldets)/1e5).astype(int)))


ct_no_imaging = 0
ct_no_png = 0
ct_no_nei = 0
ct_no_mini = 0

missing = []
# no_nei = []
# no_mini = []
# no_imaging = []

all_rpts = []
all_nei = []
all_mini = []

#todo: open the various sqlite dbs and get the list of all detectids they have
#todo: make file locations automatic
#todo: add some error control

#main reports
SQL_QUERY = "SELECT detectid from report;"

dbs = sorted(glob.glob(os.path.join(db_path,"elixer_reports_" + report_prefix + "*[0-9].db")))
for db in dbs:
    okay = True #only read in those dbs that have a prefix that is in our start-stop range
    try:
        pr = int(os.path.basename(db).split('_')[2].split('.')[0])
        if not (pr in prefix):
            okay = False
    except:
        pass

    if not okay:
        continue

    conn = sqlite3.connect("file:" + db + "?mode=ro",uri=True)
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)
    dets = cursor.fetchall()
    cursor.close()
    conn.close()
    all_rpts.extend([x[0] for x in dets])

dbs = sorted(glob.glob(os.path.join(db_path,"elixer_reports_" + report_prefix + "*nei.db")))
for db in dbs:
    okay = True
    try:
        pr = int(os.path.basename(db).split('_')[2].split('.')[0])
        if not (pr in prefix):
            okay = False
    except:
        pass

    if not okay:
        continue

    conn = sqlite3.connect("file:" + db + "?mode=ro",uri=True)
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)
    dets = cursor.fetchall()
    cursor.close()
    conn.close()
    all_nei.extend([x[0] for x in dets])

dbs = sorted(glob.glob(os.path.join(db_path,"elixer_reports_" + report_prefix + "*mini.db")))
for db in dbs:
    okay = True
    try:
        pr = int(os.path.basename(db).split('_')[2].split('.')[0])
        if not (pr in prefix):
            okay = False
    except:
        pass

    if not okay:
        continue

    conn = sqlite3.connect("file:" + db + "?mode=ro",uri=True)
    cursor = conn.cursor()
    cursor.execute(SQL_QUERY)
    dets = cursor.fetchall()
    cursor.close()
    conn.close()
    all_mini.extend([x[0] for x in dets])

#todo: is it worth the overhead to remove 'd' from the various all_xxx lists
# once it has been checked (so that the next check has a shorter list?)
# maybe run d in alldets backward (or the all_xxx lists backward) so always
# removing from the end?
for d in alldets:
    with open("dets.progress", "a+") as progress:
        progress.write(f"{d}\n")

    #check if exists
    if not (d in all_rpts):
        #does not exist
        print(f"{d} missing report: png ({ct_no_png}), nei ({ct_no_nei}), mini ({ct_no_mini}), img ({ct_no_imaging})")
        missing.append(d)
        ct_no_png += 1
        with open("dets.rerun","a+") as f:
            f.write(f"{d}\n")
        continue #already added so no need to check further

    if check_nei:
        if not (d in all_nei):
            # does not exist
            print(f"{d} missing neighborhood: png ({ct_no_png}), nei ({ct_no_nei}), mini ({ct_no_mini}), img ({ct_no_imaging})")
            missing.append(d)
            ct_no_nei += 1
            with open("dets.rerun", "a+") as f:
                f.write(f"{d}\n")
            continue  # already added so no need to check further

    if check_mini:
        if not (d in all_mini):
            # does not exist
            print(f"{d} missing mini: png ({ct_no_png}), nei ({ct_no_nei}), mini ({ct_no_mini}), img ({ct_no_imaging})")
            missing.append(d)
            ct_no_mini += 1
            with open("dets.rerun", "a+") as f:
                f.write(f"{d}\n")
            continue  # already added so no need to check further

    #most involved, so do this one last (since one of the above checks may have
    #already marked this one to be rerun)
    if check_imaging:
        # elixer_h5 = tables.open_file(cfg.elixerh5,"r") #"elixer_merged_cat.h5","r")
        elixer_h5 = tables.open_file("/data/03261/polonius/hdr2/detect/elixer.h5", "r")
        apt = elixer_h5.root.Aperture
        rows = apt.read_where("detectid==d",field="detectid")
        if rows.size==0:
            print(f"{d} missing imaging: png ({ct_no_png}), nei ({ct_no_nei}), mini ({ct_no_mini}), img ({ct_no_imaging})")
            missing.append(d)
            with open("dets.rerun", "a+") as f:
                f.write(f"{d}\n")
            ct_no_imaging += 1
        elixer_h5.close()


print(f"{len(missing)} to be re-run")
print(f"{ct_no_png} no png")
print(f"{ct_no_nei} no nei")
print(f"{ct_no_mini} no mini")
print(f"{ct_no_imaging} no imaging")
np.savetxt("dets_all"+ report_prefix +".rerun",missing,fmt="%d")

