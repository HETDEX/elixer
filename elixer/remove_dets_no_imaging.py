"""
Looks explicitly at elixer_merged_cat.h5 in the current directory and then removes down-stream
matched detection reporst so elixer in recovery mode will recreate
all detectids for which there appears to be no imaging.
"""

import tables
import glob
import os

h5 = tables.open_file("elixer_merged_cat.h5","r")

dtb = h5.root.Detections
apt = h5.root.Aperture

alldets = dtb.read(field="detectid")
missing = []

for d in alldets:
    rows = apt.read_where("detectid==d",field="detectid")
    if rows.size==0:
        missing.append(d)

h5.close()


for d in missing:
    files = glob.glob("dispatch_*/*/"+str(d)+"*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass

