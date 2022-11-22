# combine (stack) the tables
import glob
from astropy.table import Table,vstack

table_outname = "random_apertures_"

files = glob.glob(table_outname +"*.fits")
T = Table.read(files[0],format="fits")
for f in tqdm(files[1:]):
    t = Table.read(f,format="fits")
    T = vstack([T,t])

T.write(table_outname+"_all.fits",format='fits',overwrite=True)