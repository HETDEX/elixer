# combine (stack) the tables
import glob
from astropy.table import Table,vstack
import sys

SAVE_AS_H5 = True


if SAVE_AS_H5:
    import astropy.io.misc.hdf5 as hdf5
    import numpy as np


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


if False: #older way ... works, but less efficient
    write_every = 100

    if True: #LL table
        table_outname = "empty_fibers_ll_"#prefix
        files = glob.glob("empty_fibers_*ll.fits")
        files = sorted(files)

        #T = Table.read(files[0],format="fits")
        T = None
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

    if True: #FF Table
        # now the ff
        table_outname = "empty_fibers_ff_"#prefix
        files = glob.glob("empty_fibers_*ff.fits")
        files = sorted(files)

        #T = Table.read(files[0],format="fits")
        T = None
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

else: #newer way, stack in a single call?
    if True:  # LL table
        table_outname = "empty_fibers_ll_"  # prefix
        files = glob.glob("empty_fibers_*ll.fits")
        files = sorted(files)

        T = vstack([Table.read(f,format="fits") for f in files])

        if T is not None:
            T.write(table_outname + "_all.fits", format='fits', overwrite=True)

            if SAVE_AS_H5:
                for c in tab.colnames:
                    if isinstance(tab[c], astropy.table.column.MaskedColumn):
                        print(f"Converting masked column: {c}")
                        tab[c] = np.array(tab[c])

                hdf5.write_table_hdf5(tab, fn + ".h5", path="Table", overwrite=False)

                h5 = tables.open_file(fn + ".h5", mode='r+')

                try:
                    h5.root.Table.cols.shotid.create_csindex()
                    h5.root.Table.flush()
                except Exception as E:
                    print(f"Could not create index on {index}. {E}")

        del T

    if True:  # ff table
        table_outname = "empty_fibers_ff_"  # prefix
        files = glob.glob("empty_fibers_*ff.fits")
        files = sorted(files)

        T = vstack([Table.read(f, format="fits") for f in files])

        if T is not None:
            T.write(table_outname + "_all.fits", format='fits', overwrite=True)



        del T

    if True:  # rescor table
        table_outname = "empty_fibers_ffrc_"  # prefix
        files = glob.glob("empty_fibers_*ffrc.fits")
        files = sorted(files)

        T = vstack([Table.read(f, format="fits") for f in files])

        if T is not None:
            T.write(table_outname + "_all.fits", format='fits', overwrite=True)

        del T