# combine (stack) the tables
import glob
from astropy.table import Table,vstack
import sys

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here
#"random_apertures_"
#"fiber_summary_"
#"coord_apertures_"
# if "--prefix" in args:
#     i = args.index("--prefix")
#     try:
#         prefix = sys.argv[i + 1]
#     except:
#         print("bad --prefix specified")
#         exit(-1)
# else:
#     print("no prefix specified")
#     exit(-1)




if False: #LL table
    #run twice ... just in case something is weird and there are differnt numbers of ll and ff fits
    table_outname = "empty_fibers_ll_"#prefix
    files = glob.glob("empty_fibers_*ll.fits")
    files = sorted(files)

    #T = Table.read(files[0],format="fits")
    T = None
    write_every = 100
    for i,f in enumerate(files):
        print(i+1,f)
        t = Table.read(f, format="fits")

        if T is None:
            T = Table.read(f, format="fits")
        else:
            T = vstack([T,t])

        if (i+1) % write_every == 0:
            if T is not None:
                T.write(table_outname+"_all.fits",format='fits',overwrite=True)

    if T is not None:
        T.write(table_outname+"_all.fits",format='fits',overwrite=True)

    del T

if False: #FF Table
    # now the ff
    table_outname = "empty_fibers_ff_"#prefix
    files = glob.glob("empty_fibers_*ff.fits")
    files = sorted(files)

    #T = Table.read(files[0],format="fits")
    T = None
    write_every = 100
    for i,f in enumerate(files):
        print(i+1,f)
        t = Table.read(f, format="fits")

        if T is None:
            T = Table.read(f, format="fits")
        else:
            T = vstack([T,t])

        if (i+1) % write_every == 0:
            if T is not None:
                T.write(table_outname+"_all.fits",format='fits',overwrite=True)

    if T is not None:
        T.write(table_outname+"_all.fits",format='fits',overwrite=True)

    del T

if True: #FFrc
    # now the ffrc
    table_outname = "empty_fibers_ffrc_"#prefix
    files = glob.glob("empty_fibers_*ffrc.fits")
    files = sorted(files)

    #T = Table.read(files[0],format="fits")
    T = None
    write_every = 100
    for i,f in enumerate(files):
        print(i+1,f)
        t = Table.read(f, format="fits")

        if T is None:
            T = Table.read(f, format="fits")
        else:
            T = vstack([T,t])

        if (i+1) % write_every == 0:
            if T is not None:
                T.write(table_outname+"_all.fits",format='fits',overwrite=True)

    if T is not None:
        T.write(table_outname+"_all.fits",format='fits',overwrite=True)