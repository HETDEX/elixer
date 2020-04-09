"""
edit as needed, mostly here for reference purposes

Use to update columns on specific rows an existing elixer.h5 file
"""

import tables
import numpy as np

h5file = "/home/dustin/code/python/lycon.hdr2.randomsky/elixer_hdr2.h5"

h5 = tables.open_file(h5file,"r+")

atb = h5.root.Aperture
eotb = h5.root.ExtractedObjects

rows = eotb.read_where('(selected==True)')
detid = rows['detectid']
filter = rows['filter_name']
major_d = rows['major']
minor_d = rows['minor']

for d,f,a,b in zip(detid,filter,major_d,minor_d):
    try:
        i = atb.get_where_list("(detectid==d) & (filter_name==f)")
        if i.size == 1:
            i = i[0]
            re = 0.5*np.sqrt(a*b)
            atb.modify_column(i,i+1,1,re,'radius')
    except Exception as e:
        print(f"{d} Failure",e)
