# combine (stack) the tables
# each table file is bound to a single shot and has a single seeing FWHM
# since there are (usually) 200 apertures request per shot, each with aronud 20 fibers per aperture
#  a table for 1200 shots gets very big very quick
# here we reduce each shot to two stacks
#  1) the aperture stack
#  2) the fibers stack
# and save those as tables over all shots


import glob
import numpy as np
from astropy.table import Table,vstack
import sys
import os.path as op

from elixer import global_config as G
from elixer import spectrum_utilities as SU

average = "biweight"


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

if "--path" in args:
    i = args.index("--path")
    try:
        path = sys.argv[i + 1]
    except:
        print("bad --path specified")
        exit(-1)
else:
    path = "./"

table_outname = prefix

aper_files = sorted(glob.glob(op.join(path,prefix +"*[0-9].fits")))
fiber_files = sorted(glob.glob(op.join(path,prefix +"*_fibers.fits")))

#these SHOULD be in the same order due to the shotid
if len(fiber_files) == 0:
    fiber_files = np.full(len(aper_files),None)
elif len(fiber_files) != len(aper_files):
    #we have a problem
    print(f"Inconsistent aperture and fiber table file counts. Exiting!")
    exit(-1)

T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int),
                 ('seeing',float),
                 ('aper_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_ct', int),
                 ('aper_fluxd_corrected', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_fluxd_err_corrected', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_corr_ct', int),
                 ('fiber_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_ct', int),
                 ])


write_every = 100

for i,files in enumerate(zip(aper_files,fiber_files)):
    print(i+1,op.basename(files[0]))
    f1 = files[0]
    f2 = files[1]

    #anity check
    if f2 is not None and op.basename(f2) != str(op.basename(f1)[:-5])+"_fibers.fits":
        print(f"Problem. Aperture and fiber files out of sync. Exiting!")
        exit(-1)

    t1 = Table.read(f1, format="fits")
    if f2 is not None:
        t2 = Table.read(f2, format="fits")
    else:
        t2 = None


    #stack the apertures
    try:
        aper_flux_stack, aper_fluxe_stack, aper_grid, aper_contributions = SU.stack_spectra(
                                                                                        t1['fluxd'],
                                                                                        t1['fluxd_err'],
                                                                                        np.tile(G.CALFIB_WAVEGRID,(len(t1),1)),
                                                                                        grid=G.CALFIB_WAVEGRID,
                                                                                        avg_type=average,
                                                                                        straight_error=False)
        aper_contributions = int(np.nanmean(aper_contributions))  # should actually be a constant
    except Exception as e:
        print(e)
        aper_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        aper_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        aper_contributions = 0


    try:
        #stack the correceted apertures
        corr_flux_stack, corr_fluxe_stack, corr_grid, corr_contributions = SU.stack_spectra(
                                                                            t1['fluxd_zero'],
                                                                            t1['fluxd_zero_err'],
                                                                            np.tile(G.CALFIB_WAVEGRID, (len(t1), 1)),
                                                                            grid=G.CALFIB_WAVEGRID,
                                                                            avg_type=average,
                                                                            straight_error=False)
        corr_contributions = int(np.nanmean(corr_contributions))  # should actually be a constant
    except Exception as e:
        print(e)
        corr_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        corr_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        corr_contributions = 0


    if t2 is not None:
        # stack the fibers
        try:
            fiber_flux_stack, fiber_fluxe_stack, fiber_grid, fiber_contributions = SU.stack_spectra(
                                                                                                t2['fluxd'],
                                                                                                t2['fluxd_err'],
                                                                                                np.tile(G.CALFIB_WAVEGRID,
                                                                                                        (len(t2), 1)),
                                                                                                grid=G.CALFIB_WAVEGRID,
                                                                                                avg_type=average,
                                                                                                straight_error=False)
            fiber_contributions = int(np.nanmean(fiber_contributions)) #should actually be a constant
        except Exception as e:
            print(e)
            fiber_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
            fiber_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
            fiber_contributions =0
    else:
        fiber_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        fiber_fluxe_stack= np.full(len(G.CALFIB_WAVEGRID), np.nan)
        fiber_contributions = 0
        #fiber_grid, fiber_contributions



   #add to the table:
    T.add_row([t1['ra'][0], t1['dec'][0], t1['shotid'][0], t1['seeing'][0],
               aper_flux_stack, aper_fluxe_stack,aper_contributions,
               corr_flux_stack, corr_fluxe_stack,corr_contributions,
               fiber_flux_stack, fiber_fluxe_stack,fiber_contributions])

    if (i) % write_every == 0:
        if T is not None:
            T.write(table_outname+"_stacks.fits",format='fits',overwrite=True)

    if t1 is not None:
        del t1
    if t2 is not None:
        del t2

if T is not None:
    T.write(table_outname+"_stacks.fits",format='fits',overwrite=True)
