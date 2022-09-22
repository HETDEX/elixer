"""
The HSC-SSP photz catalogs are as ~2500 FITS files. The main HDU is [1] and has a single "column" that is the
unique ID followed by the zPDF values in 100, 600, or 700 bins (the bins are defined in HDU[2]).
This requires a full read of the entire HDU[1] just to see if the unique ID is present and that is costly, as it has
to pull in all the zPDFs. This is due to an organizational constraint in FITS format which requires row reads (cannot
column read ... though HSC puts these all as one column anyway, so that may be moot).

This file will read in a FITS, strip off the effected 1st column (the unique src_id) and save that back as binary file.
This allows for MUCH faster read and check (to see if the src_id) is present. Will still have to read the full zPDF in the
cases where the src_id IS found, but in the other cases (the majority of cases), no ID is found and we skip the full read.
"""

import numpy as np
#import glob
import astropy.io.fits as astropyFITS
import os.path as op
from datetime import datetime
import sys

def main():

    args = list(map(str.lower, sys.argv)) #args[1] is the file to process

    print("using: ",args[1])

    files = np.loadtxt(args[1],dtype=str)
    tag = args[1][-2:]
    sz = len(files)

    #need to define files
    for i,f in enumerate(files):
        try:
            path = op.dirname(f)
            name = op.basename(f)[:-5] #strop off the ".fits"
            outfile = op.join(path,name+".sid")

            if op.exists(outfile):
                print(f"({i}) exists ",f)
                continue

            print(f"{i} {datetime.now().strftime('%H:%M:%S')} reading {f}")
            hdulist = astropyFITS.open(f, mode="readonly", memmap=True, lazy_load_hdus=True,
                                   ignore_missing_simple=True)

            print(f"{i}  {datetime.now().strftime('%H:%M:%S')} preparing array ...")
            src_ids = np.array([x[0] for x in np.array(hdulist[1].data)])

            src_ids.tofile(outfile)
            print(f"({i}/{sz}) {tag} {datetime.now().strftime('%H:%M:%S')} done ", f)
            hdulist.close()

        except Exception as e:
            print(f"({i}) failed {f}",e)
            try:
                hdulist.close()
            except:
                pass


    print(f"*** {tag} *** all complete")

if __name__ == '__main__':
    main()
