"""
merge the per-shot detections h5 files into a single (fits) file (should be around 20GB)
NOTICE: the fiber files need to stay as per-shot and as h5 files (would be around 400GB if we did it)

!!! NOTICE !!! edit the table_outname to select what you want (e.g. the *_dets_ll  vs *_dets_ff vs *dets_ffrc

!!! NOTiCE !!! this needs a lot of memory, so vm-small may NOT be sufficient !!!

"""


# combine (stack) the tables
import glob
from astropy.table import Table,vstack
import tables
from tqdm import tqdm
import astropy.io.misc.hdf5 as hdf5

table_outname = "hsc_sep_rex_flags_nei1p5_dets_ll"
match_name = "*_hsc_sep_rex_flags_nei1p5*.h5"
files = glob.glob(match_name)

print(len(files),"found")

T = None
write_every = 100
ct = 0
for i,f in tqdm(enumerate(files),total=len(files)):
    #print(i+1,f)
    t = Table.read(f, path='Detections')

    if T is None:
        T = t
    else:
        T = vstack([T,t])

    if (i+1) % write_every == 0:
        if T is not None:
            ct += 1
            T.write(table_outname+f"_{str(ct).zfill(3)}.fits",format='fits',overwrite=True)
            del T
            T = None

if T is not None:
    ct += 1
    T.write(table_outname + f"_{str(ct).zfill(3)}.fits", format='fits', overwrite=True)
    del T
    T = None

#now merge the intermediates
files = glob.glob(table_outname + "*.fits")
print(len(files),"merging ....")
T = None
for i,f in tqdm(enumerate(files),total=len(files)):
    t = Table.read(f)

    if T is None:
        T = t
    else:
        T = vstack([T,t])

#print("Writing final fits ...")
#T.write(table_outname+"_all.fits",format='fits',overwrite=True)
#print("Done")

print("Writing final hdf5 file ...")
hdf5.write_table_hdf5(T,table_outname+"_all.h5",path="Detections",overwrite=True)

#set the index
h5 = tables.open_file(table_outname+"_all.h5",mode='r+')
h5.root.Detections.cols.detectid.create_csindex()
h5.root.Detections.flush()

h5.close()
print("Done")