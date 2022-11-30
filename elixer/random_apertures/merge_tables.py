# combine (stack) the tables
import glob
from astropy.table import Table,vstack

table_outname = "random_apertures_"

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