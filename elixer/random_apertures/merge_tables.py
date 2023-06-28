# combine (stack) the tables
import glob
from astropy.table import Table,vstack
import sys

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here
#"random_apertures_"
#"fiber_summary_"
#"coord_apertures_"
if "--prefix" in args:
    i = args.index("--prefix")
    try:
        prefix = sys.argv[i + 1]
    except:
        print("bad --prefix specified")
        exit(-1)
else:
    print("no prefix specified")
    exit(-1)

table_outname = prefix

files = glob.glob(table_outname +"*.fits")
#T = Table.read(files[0],format="fits")
T = None
T2 = None
write_every = 100
for i,f in enumerate(files[1:]):
    print(i+1,f)
    t = Table.read(f, format="fits")

    if "_fibers.fits" in f:
        if T2 is None:
            T2 = Table.read(f,format="fits")
        else:
            T2 = vstack([T2, t])
    else:
        if T is None:
            T = Table.read(f, format="fits")
        else:
            T = vstack([T,t])

    if (i+1) % write_every == 0:
        if T is not None:
            T.write(table_outname+"_all.fits",format='fits',overwrite=True)
        if T2 is not None:
            T2.write(table_outname+"_fibers_all.fits",format='fits',overwrite=True)

if T is not None:
    T.write(table_outname+"_all.fits",format='fits',overwrite=True)

if T2 is not None:
    T2.write(table_outname + "_all_fibers.fits", format='fits', overwrite=True)