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
import os
import os.path as op
import copy
import shutil
from datetime import datetime

from elixer import global_config as G
from elixer import spectrum_utilities as SU

average = "biweight"

tmppath = "." #"/tmp/hx/"

if not os.path.exists(tmppath):
    os.makedirs(tmppath, mode=0o755)
    if not os.access(tmppath, os.W_OK):
        print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
        exit(-1)

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



if "--ffsky" in args:
    #
    # changes the files to use from *[0-9].fits and *[0-9]_fibers.fits
    # to *[0-9]_ff.fits and *[0-9]_ff_fibers.fits
    # does not apply if --aperfile and --fiberfile are used
    print("using ff sky subtraction")
    ffsky = True
else:
    print("using local sky subtraction")
    ffsky = False

if "--aperfile" in args:
    i = args.index("--aperfile")
    try:
        aperfile = sys.argv[i + 1]
    except:
        print("bad --aperfile specified")
        exit(-1)
else:
    aperfile =None

if "--fiberfile" in args:
    i = args.index("--fiberfile")
    try:
        fiberfile = sys.argv[i + 1]
    except:
        print("bad --fiberfile specified")
        exit(-1)
else:
    fiberfile = None

if ffsky:
    table_outname = prefix + "per_shot_ff_"
else:
    table_outname = prefix + "per_shot_"

if aperfile is None:
    if ffsky:
        aper_files = sorted(glob.glob(op.join(path, prefix + "*[0-9]_ff.fits")))
    else:
        aper_files = sorted(glob.glob(op.join(path,prefix +"*[0-9].fits")))
else:
    aper_files = [op.join(path,aperfile)]
    table_outname += f"_{aperfile[:-5]}"

if fiberfile is None:
    if ffsky:
        fiber_files = sorted(glob.glob(op.join(path, prefix + "*[0-9]_ff_fibers.fits")))
    else:
        fiber_files = sorted(glob.glob(op.join(path,prefix +"*[0-9]_fibers.fits")))
else:
    fiber_files = [op.join(path, fiberfile)]

#these SHOULD be in the same order due to the shotid
if len(fiber_files) == 0:
    fiber_files = np.full(len(aper_files),None)
elif len(fiber_files) != len(aper_files):
    #we have a problem
    print(f"Inconsistent aperture and fiber table file counts. Exiting!")
    exit(-1)

existing_shots = None
try:
    T = Table.read(op.join(tmppath, table_outname + "_stacks.fits"), format='fits')
    existing_shots = np.array(T['shotid'])
except:
    T = Table(dtype=[('ra', float), ('dec', float), ('shotid', int),
                 ('seeing',float), ('response',float),
                 ('aper_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_ct', int),
                 ('aper_fluxd_corrected', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_fluxd_err_corrected', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_corr_ct', int),
                 ('fiber_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_ct', int),
                 ('dust_corr', (float, len(G.CALFIB_WAVEGRID))),
                 ('trim_aper_5bin_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('trim_aper_5bin_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('trim_fiber_5bin_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('trim_fiber_5bin_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_dered_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('aper_dered_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_dered_fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_dered_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ])


write_every = 100




#
# todo: over all shots (not really set up this way) AND over each shot (as currently set up)
# todo: order by blue and red flux separately and stack blue and red ends, keeping the bottom 2/3 of each set
# todo: then join the blue + red ends
#
# todo: actually use 5 bins:
# [3500-3860),[3860-4270),[4270-4680),[4680-5090),[5090-5500)
# or indicies: [15-195),[195-400),[400-605),[605-810),[810-1015)
#


def sort_by_wavelength(fluxd_2d, fluxd_err_2d): #,trim=0.95):
    """
    sort down each wavelength bin,

    :param fluxd_2d:
    :param fluxd_err_2d:
    :param sort:
    :param trim:
    :return:
    """

    #need the fluxd_err_2d to go with it?
    try:
        F = np.array(fluxd_2d)
        E = np.array(fluxd_err_2d)
        #sort down each column (that is, down each wavelength bin)
        #get the sort as 2D indicies so can apply to the flux and the flux_err
        i2d = F.argsort(axis=0)
        sF = np.take_along_axis(F,i2d,0)
        sE = np.take_along_axis(E,i2d,0)


        #more efficient to perform this sort and return
        #let the caller then sort at different trims
        return sF, sE

        # #keep the top trim (sorted from smallest flux at the top to largest)
        # last_row = int(trim * len(F))
        # sF = sF[0:last_row+1]
        # sE = sE[0:last_row + 1]

        #stack (use elixer is fine)

    except Exception as Exc:
        print(Exc)
        return None, None




def split_spectra_into_bins(fluxd_2d, fluxd_err_2d,sort=True,trim=0.666):
    """
    given 2d array of spectra (G.CALFIB_WAVEGRID wavelengths assumed) in flux_density,
    split and retun as a dictionary of 5 bins (10 entries ... 5bins x 2 for fluxd and fluxd_error)
    :param fluxd_2d:
    :param fluxd_err_2d:
    :param sort: if True, sort under each bin (independently)
    :param trim: if 0 to 1.0, keep the bottom "trim" fraction of the sorted sample (only if sort is True)
    :return:
    """

    rd = {} #return dict

    bin_idx_ranges = [(15,195),(195,400),(400,605),(605,810),(810,1015)] #(inclusive-exclusive)
    full_idx_ranges = [(0, 195), (195, 400), (400, 605), (605, 810), (810, 1036)] #1st and last different to cover full 1036


    #todo: when appending, though need the full width, so 1st bin needs to be (0,195) and the last (810,1036)

    for i,((left,right),(full_left,full_right)) in enumerate(zip(bin_idx_ranges,full_idx_ranges)):
        f = fluxd_2d[:,left:right]
        e = fluxd_err_2d[:,left:right]
        r = (full_left,full_right)
        ff = fluxd_2d[:,full_left:full_right]
        ef = fluxd_err_2d[:,full_left:full_right]

        if sort:
            md = np.nanmedian(fluxd_2d,axis=1)
            idx = np.argsort(md) #ascending order, smallest average flux to largest

            if trim is not None and ( 0 < trim < 1.0):
                max_idx = int(len(md)*trim)
            else:
                max_idx = len(md)
            f = f[idx[0:max_idx]]
            e = e[idx[0:max_idx]]
            ff = ff[idx[0:max_idx]]
            ef = ef[idx[0:max_idx]]

        rd[f"f{i + 1}"] = copy.copy(ff) #flux density in bin range
        rd[f"e{i + 1}"] = copy.copy(ef) #flux density error in bin range
        rd[f"r{i + 1}"] = copy.copy(r) #bin range indicies (inclusive index, exclusive index)

    return rd






for enum_i,files in enumerate(zip(aper_files,fiber_files)):
    print("stack_per_shot:", enum_i+1,op.basename(files[0]),datetime.now(),flush=True)

    f1 = files[0]
    f2 = files[1]

    #oddly, if this fails with an exception, I want the whole script to stop
    if existing_shots is not None:
        #get the shotid and see if we already have it
        the_shot = int(os.path.basename(f1).split("_")[2].split(".")[0])
        if the_shot in existing_shots:
            print(f"stack_per_shot: {the_shot} #{enum_i+1} already in table. Skipping.")
            continue

    #sanity check
    if f2 is not None and op.basename(f2) != str(op.basename(f1)[:-5])+"_fibers.fits":
        print(f"Problem. Aperture and fiber files out of sync. Exiting!")
        exit(-1)

    t1 = Table.read(f1, format="fits")
    if f2 is not None:
        t2 = Table.read(f2, format="fits")
    else:
        t2 = None

    try:
        dust_corr = np.nanmedian(t1['dust_corr'],axis=0)
    except:
        dust_corr = np.full(len(G.CALFIB_WAVEGRID), np.nan)

    # stack apertures by trimmed data
    ad = split_spectra_into_bins(t1['fluxd'], t1['fluxd_err'], sort=True, trim=0.666)
    i = 0
    while f"f{i + 1}" in ad.keys():
        fkey = f"f{i + 1}"
        ekey = f"e{i + 1}"
        rkey = f"r{i + 1}"
        try:
            aper_flux_stack, aper_fluxe_stack, aper_grid, aper_contributions = SU.stack_spectra(
                ad[fkey],
                ad[ekey],
                np.tile(G.CALFIB_WAVEGRID[ad[rkey][0]:ad[rkey][1]], (len(ad[fkey]), 1)),
                grid=G.CALFIB_WAVEGRID[ad[rkey][0]:ad[rkey][1]],
                avg_type=average,
                straight_error=False)
            aper_contributions = int(np.nanmean(aper_contributions))  # should actually be a constant

            ad[f"fs{i + 1}"] = aper_flux_stack
            ad[f"es{i + 1}"] = aper_fluxe_stack

        except Exception as e:
            print(e,flush=True)
            ad[f"fs{i + 1}"] = None
            ad[f"es{i + 1}"] = None

        i += 1

    # now stitch them all together into a single spectrum and error
    try:
        ad["fluxd_stack"] = np.concatenate([ad[x] for x in ad.keys() if x[0:2] == "fs"])
        ad["fluxe_stack"] = np.concatenate([ad[x] for x in ad.keys() if x[0:2] == "es"])
    except Exception as e:
        print(e,flush=True)
        print(f"Failing {f1} and continue on ...")
        continue



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
        print(e,flush=True)
        aper_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        aper_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        aper_contributions = 0



    #stack the de-reddened apertures
    try:
        aper_dered_flux_stack, aper_dered_fluxe_stack, aper_dered_grid, aper_dered_contributions = \
                                                                                    SU.stack_spectra(
                                                                                        t1['fluxd']*t1['dust_corr'],
                                                                                        t1['fluxd_err'],
                                                                                        np.tile(G.CALFIB_WAVEGRID,(len(t1),1)),
                                                                                        grid=G.CALFIB_WAVEGRID,
                                                                                        avg_type=average,
                                                                                        straight_error=False)
        aper_dered_contributions = int(np.nanmean(aper_dered_contributions))  # should actually be a constant
    except Exception as e:
        print(e,flush=True)
        aper_dered_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        aper_dered_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        aper_dered_contributions = 0


    #
    # #stack apertures by trimmed data
    # ad = split_spectra_into_bins(t1['fluxd'],t1['fluxd_err'],sort=True,trim=0.666)
    # i = 0
    # while f"f{i+1}" in ad.keys():
    #     fkey = f"f{i+1}"
    #     ekey = f"e{i+1}"
    #     rkey = f"r{i+1}"
    #     try:
    #         aper_flux_stack, aper_fluxe_stack, aper_grid, aper_contributions = SU.stack_spectra(
    #             ad[fkey],
    #             ad[ekey],
    #             np.tile(G.CALFIB_WAVEGRID[ad[rkey][0]:ad[rkey][1]], (len(ad[fkey]), 1)),
    #             grid=G.CALFIB_WAVEGRID[ad[rkey][0]:ad[rkey][1]],
    #             avg_type=average,
    #             straight_error=False)
    #         aper_contributions = int(np.nanmean(aper_contributions))  # should actually be a constant
    #
    #         ad[f"fs{i+1}"] = aper_flux_stack
    #         ad[f"es{i+1}"] = aper_fluxe_stack
    #
    #     except Exception as e:
    #         print(e)
    #         ad[f"fs{i+1}"] = None
    #         ad[f"es{i+1}"] = None
    #
    #     i += 1
    #
    # #now stitch them all together into a single spectrum and error
    # ad["fluxd_stack"] = np.concatenate([ad[x] for x in ad.keys() if x[0:2] =="fs"])
    # ad["fluxe_stack"] = np.concatenate([ad[x] for x in ad.keys() if x[0:2] == "es"])
    #



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
        print(e,flush=True)
        corr_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        corr_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        corr_contributions = 0




    ###############################################
    # per wavelength bin fibers sort, trim, stack
    # This does NOT alter t2[] columns
    ###############################################
    if t2 is not None:
        # sort
        sF, sE = sort_by_wavelength(t2['fluxd'], t2['fluxd_err'])
        #sort with dust correction
        sFdust, sEdust = sort_by_wavelength(t2['fluxd']*t2['dust_corr'], t2['fluxd_err'])


    ########################
    #Fibers data (5 bins)
    #######################
    if t2 is not None:

        # stack fibers by trimmed data
        fd = split_spectra_into_bins(t2['fluxd'], t2['fluxd_err'], sort=True, trim=0.666)
        i = 0
        while f"f{i + 1}" in fd.keys():
            fkey = f"f{i + 1}"
            ekey = f"e{i + 1}"
            rkey = f"r{i + 1}"
            try:
                fiber_flux_stack, fiber_fluxe_stack, fiber_grid, fiber_contributions = SU.stack_spectra(
                    fd[fkey],
                    fd[ekey],
                    np.tile(G.CALFIB_WAVEGRID[fd[rkey][0]:fd[rkey][1]], (len(fd[fkey]), 1)),
                    grid=G.CALFIB_WAVEGRID[fd[rkey][0]:fd[rkey][1]],
                    avg_type=average,
                    straight_error=False)
                fiber_contributions = int(np.nanmean(fiber_contributions))  # should actually be a constant

                fd[f"fs{i + 1}"] = fiber_flux_stack
                fd[f"es{i + 1}"] = fiber_fluxe_stack

            except Exception as e:
                print(e,flush=True)
                fd[f"fs{i + 1}"] = None
                fd[f"es{i + 1}"] = None

            i += 1

        # now stitch them all together into a single spectrum and error
        fd["fluxd_stack"] = np.concatenate([fd[x] for x in fd.keys() if x[0:2] == "fs"])
        fd["fluxe_stack"] = np.concatenate([fd[x] for x in fd.keys() if x[0:2] == "es"])

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
            print(e,flush=True)
            fiber_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
            fiber_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
            fiber_contributions =0



        # stack the fibers dereddened
        try:
            fiber_dered_flux_stack, fiber_dered_fluxe_stack, fiber_dered_grid, fiber_dered_contributions = SU.stack_spectra(
                                                                                                t2['fluxd']*t2['dust_corr'],
                                                                                                t2['fluxd_err'],
                                                                                                np.tile(G.CALFIB_WAVEGRID,
                                                                                                        (len(t2), 1)),
                                                                                                grid=G.CALFIB_WAVEGRID,
                                                                                                avg_type=average,
                                                                                                straight_error=False)
            fiber_dered_contributions = int(np.nanmean(fiber_dered_contributions)) #should actually be a constant
        except Exception as e:
            print(e,flush=True)
            fiber_dered_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
            fiber_dered_fluxe_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
            fiber_dered_contributions =0
    else:
        fiber_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        fiber_fluxe_stack= np.full(len(G.CALFIB_WAVEGRID), np.nan)
        fiber_contributions = 0

        fiber_dered_flux_stack = np.full(len(G.CALFIB_WAVEGRID), np.nan)
        fiber_dered_fluxe_stack= np.full(len(G.CALFIB_WAVEGRID), np.nan)
        fiber_dered_contributions = 0
        #fiber_grid, fiber_contributions







   #add to the table:
    try:
        T.add_row([t1['ra'][0], t1['dec'][0], t1['shotid'][0], t1['seeing'][0],t1['response'][0],
                   aper_flux_stack, aper_fluxe_stack,aper_contributions,
                   corr_flux_stack, corr_fluxe_stack,corr_contributions,
                   fiber_flux_stack, fiber_fluxe_stack,fiber_contributions,
                   dust_corr,
                   ad['fluxd_stack'],ad['fluxe_stack'],
                   fd['fluxd_stack'],fd['fluxe_stack'],
                   aper_dered_flux_stack, aper_dered_fluxe_stack,
                   fiber_dered_flux_stack, fiber_dered_fluxe_stack])
    except:
        print(f"[{enum_i}] Unable to add row.")

    if enum_i % write_every == 0:
        print("intermediate write ...",flush=True)
        if T is not None:
            T.write(op.join(tmppath,table_outname+"_stacks.fits"),format='fits',overwrite=True)
        else:
            print("Table is None",flush=True)

    if t1 is not None:
        del t1
    if t2 is not None:
        del t2

print("final write ...",flush=True)
if T is not None:
    T.write(op.join(tmppath,table_outname+"_stacks.fits"),format='fits',overwrite=True)
else:
    print("Table is None",flush=True)

if tmppath is not None and tmppath != ".":
    print("Copying from /tmp",flush=True)
    shutil.copy2(op.join(tmppath,table_outname+"_stacks.fits"),table_outname+"_stacks.fits")  # ,copy_function=copy)
    os.remove(op.join(tmppath,table_outname+"_stacks.fits"))