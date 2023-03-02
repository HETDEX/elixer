# combine (stack) the tables
import glob
from astropy.table import Table,vstack


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
T = Table.read(files[0],format="fits")
write_every = 100
for i,f in enumerate(files[1:]):
    print(i+1,f)
    t = Table.read(f,format="fits")
    T = vstack([T,t])

    if (i+1) % write_every == 0:
        T.write(table_outname+"_all.fits",format='fits',overwrite=True)

T.write(table_outname+"_all.fits",format='fits',overwrite=True)