"""
Iterate through all expected detections and check that:
1. there is an entry in the elixer h5 file
2. the elixer h5 has imaging recorded if there is imaging available
3. there are entry(ies) in the resport database(s)


report the detections that fail the above tests

"""



import tables
import glob
import os
import numpy as np
import sys
from tqdm import tqdm
import hetdex_api.sqlite_utils as sql



alldets = None
args = list(map(str.lower,sys.argv))
if "--dets" in args: #overide default if specified on command line
    try:
        i = args.index("--dets")
        if i != -1:
            dets_file = str(sys.argv[i + 1])
            alldets = np.loadtxt(dets_file,dtype=int)
    except:
        pass

if alldets is None:
    print("Failed to load detections list. Exiting.")
    exit(-1)

h5 = None
if "--h5" in args:
    try:
        i = args.index("--h5")
        if i != -1:
            h5_file = str(sys.argv[i + 1])
            h5 = tables.open_file(h5_file)
    except:
        pass

if h5 is None:
    print("Failed to load h5 file. Exiting.")
    exit(-1)



imgdb_dir = None

if "--imgdb_dir" in args:
    try:
        i = args.index("--imgdb_dir")
        if i != -1:
            imgdb_dir = str(sys.argv[i + 1])
    except:
        pass

if imgdb_dir is None or not os.path.exists(imgdb_dir):
    print("Failed to load imgdb_dir. Exiting.")
    exit(-1)

#instantiate imaging db (add path first)
for key in sql.DICT_DB_PATHS.keys():
    #set to only check this path
    sql.DICT_DB_PATHS[key] = [imgdb_dir]
    #sql.DICT_DB_PATHS[key].insert(0, imgdb_dir)

elixer_conn_mgr = sql.ConnMgr()




#first check just for entries in h5
h5_dets = h5.root.Detections.read(field="detectid")
missing_dets = np.setdiff1d(alldets,h5_dets)
print(f"Number of missing detections: {len(missing_dets)}")
np.savetxt("consistency_missing_from_h5.dets",missing_dets,fmt="%d")



#check for no imaging
all_apt_dets = np.unique(h5.root.Aperture.read(field="detectid"))  # detectids that have an aperture entry
missing_imaging_dets = np.setdiff1d(h5_dets,all_apt_dets)
print(f"Number of missing imaging: {len(missing_imaging_dets)}")
np.savetxt("consistency_missing_imaging_h5.dets",missing_imaging_dets,fmt="%d")



#todo: add a different interface to either just query (one at a time) the existence of the entry
#todo: OR a new interface to retrieve the list of all entries and then just compare to the expected list
#check for database entries
missing_db_report = []
missing_db_nei  = []
missing_db_mini = []
for d in tqdm(h5_dets):
    try:
        img = elixer_conn_mgr.fetch_image(d, report_type="report")
        if img is None:
            missing_db_report.append(d)
        else:
            del img



        img = elixer_conn_mgr.fetch_image(d, report_type="nei")
        if img is None:
            missing_db_nei.append(d)
        else:
            del img

        img = elixer_conn_mgr.fetch_image(d, report_type="mini")
        if img is None:
            missing_db_mini.append(d)
        else:
            del img
    except:
        pass


print(f"Number of missing database report: {len(missing_db_report)}")
np.savetxt("consistency_missing_db_report.dets",missing_db_report,fmt="%d")


print(f"Number of missing database nei: {len(missing_db_nei)}")
np.savetxt("consistency_missing_db_nei.dets",missing_db_nei,fmt="%d")

print(f"Number of missing database mini: {len(missing_db_mini)}")
np.savetxt("consistency_missing_db_mini.dets",missing_db_mini,fmt="%d")


h5.close()

print("Done.")

