"""
Looks explicitly at elixer_merged_cat.h5 in the current directory and prints to screen
all detectids for which there appears to be no imaging.
"""

import tables


h5 = tables.open_file("elixer_merged_cat.h5","r")

dtb = h5.root.Detections
apt = h5.root.Aperture

alldets = dtb.read(field="detectid")
#missing = []

for d in alldets:
    rows = apt.read_where("detectid==d",field="detectid")
    if rows.size==0:
        print(d)

h5.close()
