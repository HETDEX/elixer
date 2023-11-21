"""
simple script to collect the "empty" fibers for a single shot (all dithers) validating each against acceptance critieria,
and save to an astropy table

intended for use with SLURM

caller specifies only the shotid

loosely based on hetdex_all_shots_random_empty_apertures notebooks on /work/03261/polonius/notebooks

There is a basic check for continuum and emission lines at an absolute level and then variations on cuts/clips
overall, per wavelength bin, and/or per wavelength range

Then under those cuts, the fibers are stacked into a single average and recorded for the shot

"""
import sys
import os
import os.path as op
import datetime


##################################
# should we even run this one?
##################################

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#form is integer dateobs (no 'v')
if "--shot" in args:
    i = args.index("--shot")
    try:
        shotid = int(sys.argv[i + 1])
    except:
        print("bad shotid specified")
        exit(-1)
else:
    print("no shotid specified")
    exit(-1)
shot = shotid

if "--ffsky" in args:
    print("using ff sky subtraction")
    ffsky = True
else:
    print("using local sky subtraction")
    ffsky = False

if "--bothsky" in args:
    print("Using local sky subtraction selection. Recording both local and ffsky extractions.")
    print("Overriding --ffsky")
    bothsky = True
    ffsky = False
else:
    print("using local sky subtraction")
    bothsky = False

if "--fiber_corr" in args:
    print("Apply per fiber residual correction")
    per_fiber_corr = True
else:
    print("Do Not apply per fiber residual correction")
    per_fiber_corr = False


#dust is a bit dodgy here since it is RA, Dec and thus per fiber specific and IT DOES VARY from fiber to fiber
#albeit rather slowly
# if "--dust" in args:
#     print("recording dust correction")
#     dust = True
# else:
#     print("NO dust correction")
#     dust = False


print(f"{shotid} starting: {datetime.datetime.now()}")

table_outname = "empty_fibers_"+str(shotid) + ".fits"
table_outname2 = "empty_fibers_"+str(shotid) + "_ff.fits"


#maybe this one was already done?
# which files to check really depends on --ffsky and --bothsky
    #did they all copy or did it timeout during a copy?

if bothsky:
    if op.exists(table_outname) and op.exists(table_outname2):
        #all good
        print(f"{shotid} --bothsky outputs already exist. Exiting. {datetime.datetime.now()}")
        exit(0)
elif ffsky:
    if op.exists(table_outname2):
        #all good
        print(f"{shotid} --ffsky output already exists. Exiting. {datetime.datetime.now()}")
        exit(0)
else:
    if op.exists(table_outname):
        #all good
        print(f"{shotid} output already exist. Exiting. {datetime.datetime.now()}")
        exit(0)

if op.exists(table_outname) or op.exists(table_outname2):
    print(f"{shotid} incomplete. Reset and retry. {datetime.datetime.now()}")

    try:
        if not ffsky or bothsky:
            os.remove(table_outname)
    except:
        pass

    try:
        if ffsky or bothsky:
            os.remove(table_outname2)
    except:
        pass



#####################################
# OK. This one needs to be run, so
# continue loading packages, etc
#####################################

print(f"{shotid} load ....  {datetime.datetime.now()}")

import numpy as np
import tables
from astropy.coordinates import SkyCoord
from astropy.table import Table,join,vstack
import astropy.units as u
import copy
import glob
import shutil

from hetdex_api.config import HDRconfig
from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import Survey,FiberIndex
from hetdex_tools.get_spec import get_spectra
from hetdex_api.extinction import *  #includes deredden_spectra
from hetdex_api.extract import Extract

from elixer import spectrum as elixer_spectrum
from elixer import spectrum_utilities as SU
from elixer import global_config as G
from elixer import catalogs

survey_name = "hdr4" #"hdr4" #"hdr2.1"

tmppath = "/tmp/hx/"
#tmppath = "/home/dustin/temp/random_apertures/hx/"

if not os.path.exists(tmppath):
    try:
        os.makedirs(tmppath, mode=0o755)
        if not os.access(tmppath, os.W_OK):
            print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
            exit(-1)
    except: #can catch timing on creation if multiples are trying to make it
        if not os.access(tmppath, os.W_OK):
            print(f"Warning! --tmp path does not exist, cannot be created, or is not writable: {tmppath}")
            exit(-1)

#########################################
# Helper Funcs
#########################################

def linear_g(seeing): #estimate of depth from the seeing (based on HDR 3.0.3 detections)
    # return seeing * (-31./70.) + 25.76  # middle of the y_err
    #return seeing * (-31. / 70.) + 25.56  # bottom of the y_err
    return SU.estimated_depth(seeing)


def split_spectra_into_bins(fluxd_2d, fluxd_err_2d, sort=True, trim=0.9):
    """
    given 2d array of spectra (G.CALFIB_WAVEGRID wavelengths assumed) in flux_density,
    split and retun as a dictionary of 5 bins (10 entries ... 5bins x 2 for fluxd and fluxd_error)
    :param fluxd_2d:
    :param fluxd_err_2d:
    :param sort: if True, sort under each bin (independently)
    :param trim: if 0 to 1.0, keep the bottom "trim" fraction of the sorted sample (only if sort is True)
    :return:
    """

    rd = {}  # return dict

    bin_idx_ranges = [(15, 195), (195, 400), (400, 605), (605, 810), (810, 1015)]  # (inclusive-exclusive)
    full_idx_ranges = [(0, 195), (195, 400), (400, 605), (605, 810),
                       (810, 1036)]  # 1st and last different to cover full 1036

    # todo: when appending, though need the full width, so 1st bin needs to be (0,195) and the last (810,1036)

    for i, ((left, right), (full_left, full_right)) in enumerate(zip(bin_idx_ranges, full_idx_ranges)):
        f = fluxd_2d[:, left:right]
        e = fluxd_err_2d[:, left:right]
        r = (full_left, full_right)
        ff = fluxd_2d[:, full_left:full_right]
        ef = fluxd_err_2d[:, full_left:full_right]

        sel_e = np.isnan(e)
        sel_ef = np.isnan(ef)
        f[sel_e] = np.nan
        ff[sel_ef] = np.nan

        if sort:
            md = np.nanmedian(fluxd_2d, axis=1)
            idx = np.argsort(md)  # ascending order, smallest average flux to largest

            if trim is not None and (0 < trim < 1.0):
                max_idx = int(len(md) * trim)
            else:
                max_idx = len(md)
            f = f[idx[0:max_idx]]
            e = e[idx[0:max_idx]]
            ff = ff[idx[0:max_idx]]
            ef = ef[idx[0:max_idx]]

        rd[f"f{i + 1}"] = copy.copy(ff)  # flux density in bin range
        rd[f"e{i + 1}"] = copy.copy(ef)  # flux density error in bin range
        rd[f"r{i + 1}"] = copy.copy(r)  # bin range indicies (inclusive index, exclusive index)

    return rd


def stack_by_wavebin_bw(fluxd_2d, fluxd_err_2d, trim=1.00, sc=None, ir=None):
    """
    stack down each wavelength bin, trimming under each

    sc = sigma_clip (symmetric)
    ir = internal ratio (e.g. interior 2/3)

    """

    stack = np.zeros(len(G.CALFIB_WAVEGRID))
    stack_err = np.zeros(len(G.CALFIB_WAVEGRID))
    contrib = np.zeros(len(G.CALFIB_WAVEGRID))
    N = len(fluxd_2d)

    for i in range(1036):
        # only consider those that have an associated error
        # errors of < 0 are flagged as bad pixels, cosmic strikes, etc
        sel = np.array(fluxd_err_2d[:, i] > 0) & np.array(~np.isnan(fluxd_err_2d[:, i])) & np.array(
            ~np.isnan(fluxd_2d[:, i]))
        if np.count_nonzero(sel) == 0:
            stack[i] = 0
            stack_err[i] = 0
            continue

        if trim < 1.0:
            idx = np.argsort(fluxd_2d[:, i][sel])
            max_idx = int(len(idx) * trim)
            column = fluxd_2d[:, i][sel][idx][0:max_idx]
            column_err = fluxd_err_2d[:, i][sel][idx][0:max_idx]
        elif sc is not None:
            column = fluxd_2d[:, i][sel]
            column_err = fluxd_err_2d[:, i][sel]
            mask = sigma_clip(column, sigma=sc)
            column = column[~mask.mask]
            column_err = column_err[~mask.mask]
        elif ir is not None:
            if ir >= 1.0:
                column = fluxd_2d[:, i][sel]
                column_err = fluxd_err_2d[:, i][sel]
            else:
                er = (1.0 - ir) / 2.0  # exterior ratio, e.g. if ir = 2/3 then er = 1/6 ... trim away the upper and lower 1/6
                idx = np.argsort(fluxd_2d[:, i][sel])
                low_idx = int(len(idx) * er)
                high_idx = int(len(idx) * (ir + er))
                column = fluxd_2d[:, i][sel][idx][low_idx:high_idx]
                column_err = fluxd_err_2d[:, i][sel][idx][low_idx:high_idx]
        # print(ir,low_idx,high_idx)

        elif trim == 1.0:
            column = fluxd_2d[:, i][sel]
            column_err = fluxd_err_2d[:, i][sel]
        else:
            return None, None, None

        try:
            stack[i] = biweight.biweight_location(column)
            stack_err[i] = biweight.biweight_scale(column)/np.sqrt(len(column))
            contrib[i] = len(column)
        except:
            stack[i] = 0
            stack_err[i] = 0

    return stack, stack_err, contrib


#########################
# Main Logic
#########################


hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
print("Reading survey table ...")
survey_table=survey.return_astropy_table()
print("Getting FiberIndex ...")
FibIndex = FiberIndex(survey_name)

negative_flux_limit = -0.4e-17  # erg/s/cm2/AA ... more negative than this, assume something is wrong
hetdex_nearest_detection = 2.0 #in arcsec

wave_continuum_chunks_idx = [np.arange(15, 166, 1),  # 3500-3800
                             np.arange(215, 616, 1),  # 3900-4700
                             np.arange(615, 966, 1),  # 4700-5400
                             ]
# array of tuples that set the minimum and maximum average flux density value (e-17) that are loosely okay.
# these are not based on seeing FWHM.
# aligns with wave_continuum_chunks_idx
#aperture values: note 0.1e-17 is about 24.2 at 4700AA

#based on the aperture values; for a typical high weighted (central) fiber this is about a factor of 7x high
#so, for a rough 1st estimate, divide by 7
#(though this is seeing dependent)
aper2fiber = 7.0
acceptable_fluxd = [(-0.1/aper2fiber, 0.5/aper2fiber), #swing in the blue ... really should never be negative, roughly 5-10x is maybe okay
                    (-0.1/aper2fiber, 0.1/aper2fiber),
                    (-0.1/aper2fiber, 0.1/aper2fiber),]



#############################################
# out put tables
#############################################

#local sky, ALL stacking statistics are biweight
T = Table(dtype=[('shotid', int),

                 ('raw_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep all AFTER flagged fibers removed
                 ('raw_fluxd_err', (float, len(G.CALFIB_WAVEGRID))), #NO OTHER TRIMMING
                 ('raw_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### fixed continuum trim (applies to ALL below) ###
                 #this might be the biggest differentiator ... what do we define as continuum level?
                 #and should this be shot variable?

                 ('trim_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #keep all AFTER flag trim and absolute flux cut
                 ('trim_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('trim_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### top percent trim (applies only to this section) ###

                 ('t01_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 1% per wavelength bin
                 ('t01_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t01_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t02_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 2% per wavelength bin
                 ('t02_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t02_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t03_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 3% per wavelength bin
                 ('t03_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t03_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t04_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 4% per wavelength bin
                 ('t04_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t04_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('t05_fluxd', (float, len(G.CALFIB_WAVEGRID))),  #trim off top 5% per wavelength bin
                 ('t05_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('t05_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### sigma clips (applies only to this section) ###
                 #sigma clip of 1 is WAAAY too agreesive, 2 is maybe okay, but we'll just use 3,4,5

                 ('sc3_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 3-sigm clip per wavelength bin
                 ('sc3_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('sc3_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('sc4_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 4-sigm clip per wavelength bin
                 ('sc4_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('sc4_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('sc5_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # 5-sigm clip per wavelength bin
                 ('sc5_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('sc5_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ### interior fraction (applies only to this section) ###

                 #these are all extememly similar; after trimming for continuum, the distributions
                 #are extemely Gaussian, so this particular variation makes almost no difference

                 # ('ir50_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 50% (trim off top and bottom 25%)
                 # ('ir50_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 # ('ir50_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 #roughly interior +/- 1 std (assuming a normal distro)
                 ('ir67_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 2/3 (trim off top and bottom 1/6)
                 ('ir67_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ir67_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 #roughly interior +/- 2 std (assuming a normal distro)
                 ('ir95_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 95% (trim off top and bottom 2.5%)
                 ('ir95_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ir95_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 # roughly interior +/- 3 std (assuming a normal distro)
                 ('ir99_fluxd', (float, len(G.CALFIB_WAVEGRID))),  # keep innermost 99% (trim off top and bottom 0.5%)
                 ('ir99_fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('ir99_contrib', (float, len(G.CALFIB_WAVEGRID))),

                 ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_weights',(float,32)),
                 ('fiber_weights_norm',(float,32)),
                 ('fluxd_zero', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_zero_err', (float, len(G.CALFIB_WAVEGRID))),

                 ])

#ffsky (if both)
T2 = Table(dtype=[('shotid', int),
                 ('seeing',float),('response',float),
                 ('dex_g',float),('dex_g_err',float),('dex_cont',float),('dex_cont_err',float),
                 ('fluxd', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('fiber_weights',(float,32)),
                 ('fiber_weights_norm',(float,32)),
                 ('fluxd_zero', (float, len(G.CALFIB_WAVEGRID))),
                 ('fluxd_zero_err', (float, len(G.CALFIB_WAVEGRID))),
                 ('dust_corr', (float, len(G.CALFIB_WAVEGRID))),
                 ])






fibers_table = get_fibers_table(shot)
#drop the columns we don't care about to save memory
fibers_table.keep_columns(['fiber_id','calfib','calfib_ffsky','calfibe','fiber_to_fiber','trace','chi2']) #have to keep the fiber_id for the moment

start = f'{shot}_0'
stop = f'{shot}_9'
mask_table = Table(FibIndex.fibermaskh5.root.Flags.read_where("(fiber_id > start) & (fiber_id < stop)"))
super_tab = join(fibers_table, mask_table, "fiber_id")
del fibers_table #don't need it anymore
del mask_table
super_tab.remove_columns(['fiber_id']) #don't need it anymore


#######################################################################
# first, always cut all fibers (entire fibers) that are flagged
# remove any that are bad (flag = True is actually good)
#######################################################################
sel = super_tab['flag'] & super_tab['amp_flag'] & super_tab['meteor_flag'] & super_tab['gal_flag'] & super_tab['shot_flag'] & super_tab['throughput_flag']
super_tab = super_tab[sel]
#and we do not need the flag columns anymore
super_tab.remove_columns(['flag','amp_flag','meteor_flag','gal_flag','shot_flag','throughput_flag'])

flux_stack, fluxe_stack = stack_by_wavebin_bw(super_tab[col], super_tab[cole], trim=1.00, sc=None, ir=None)

T['raw_fluxd'] = flux_stack
T['raw_fluxd_err'] = fluxe_stack

############################################################################
# next, cut all fibers with obvious continuum or deeply negative problems
############################################################################
rd = split_spectra_into_bins(super_tab['calfib'],super_tab['calfibe'],sort=False,trim=None)

super_tab['avg1'] = np.nanmedian(rd['f1'],axis=1) #3500-3860
super_tab['avg2'] = np.nanmedian(rd['f2'],axis=1) #3860-4270
super_tab['avg3'] = np.nanmedian(rd['f3'],axis=1) #4270-4860
super_tab['avg4'] = np.nanmedian(rd['f4'],axis=1) #4860-5090
super_tab['avg5'] = np.nanmedian(rd['f5'],axis=1) #5090-5500


norm_min = -0.05
norm_max = 0.05
#first bin, at exteme blue is different
sel =       np.array(super_tab['avg1'] > norm_min) & np.array(super_tab['avg1'] < 0.2) #3500-3860

sel = sel & np.array(super_tab['avg2'] > norm_min) & np.array(super_tab['avg2'] < norm_max) #3860-4270
sel = sel & np.array(super_tab['avg3'] > norm_min) & np.array(super_tab['avg3'] < norm_max) #4270-4860
sel = sel & np.array(super_tab['avg4'] > norm_min) & np.array(super_tab['avg4'] < norm_max) #4860-5090
sel = sel & np.array(super_tab['avg5'] > norm_min) & np.array(super_tab['avg5'] < norm_max) #5090-5500
super_tab = super_tab[sel]

#######################################################################
# next, cut all fibers with emission lines??? (this could be trixy)
# maybe a median filter that is 3 or 5 wide and if there are any peaks above
# some level, then cut the whole fiber?
# any faint emission line will not repeat over more than a handfull of fibers
#   and when averaged into 60-70K fibers, will be meaningless
#######################################################################



# NOW apply the variable cuts:
#  entire fiber, in multi-wavelength bins, per wavelength bin
#  by sigma clip?, by top xx where the xx% could vary by bin
#





if np.count_nonzero(sel) != 1:
    print(f"problem: {shotid} has {np.count_nonzero(sel)} hits")
    exit(0)



del survey_table

#elix_h5  = tables.open_file("/scratch/03946/hetdex/hdr3/detect/elixer.h5")
#elix_h5  = tables.open_file("/scratch/projects/hetdex/hdr3/detect/elixer.h5")
#the RA, Dec of HETDEX detections, used as a check against detections being too close
#these CAN BE NOISE though
elix_h5  = tables.open_file("/scratch/projects/hetdex/hdr4/detect/elixer_hdr3_hdr4_all_cat.h5")
dex_ra = elix_h5.root.Detections.read_where("sn > 5.5",field='ra')
dex_dec = elix_h5.root.Detections.read_where("sn > 5.5",field='dec')
#dex_snr = elix_h5.root.Detections.read(field='sn')
elix_h5.close()

print(f"{shotid} main loop ....  {datetime.datetime.now()}")

aperture_sky_subtraction_residual = None #only one per shot
aperture_sky_subtraction_residual_err = None

catlib = catalogs.CatalogLibrary()

for f in super_tab: #these fibers are in a random order so just iterating over them is random
                    #though it is possible to get two apertures at the same location (adjacent fibers and random
                    #ra, dec shifts could line them up ... would be exceptionally unlikely, so we will ignore)
                    #It is more likely to pick up one or a few that have overlapping apertures, but since this is
                    #PSF weighted, the effect of the overlap is limited and quickly falls off over 1-2"

    if sequential_fails > fail_rate_max:
        if aper_ct > 50:
            break
        elif aper_ct > 10 and short_array:
            break



    if aper_ct > empty_apertures_per_shot:
        # we have enough, move on to the next shot
        break

    sequential_fails += 1

    if not (f['flag']):#" and f['amp_flag'] and f['meteor_flag'] and f['gal_flag']):
        # if any of the three are false, skip this fiber and move on
        pass


    ra = np.random.uniform(-ra_nudge, ra_nudge) / 3600. + f['ra']
    dec = np.random.uniform(-dec_nudge, dec_nudge) / 3600. + f['dec']
    coord = SkyCoord(ra, dec, unit="deg")

    #none can be within 1.5" of another
    if len(accepted_coords) > 0:
        #just iterate over the accepted_coords as there aren't that many
        #rather than keep rebuilding a SkyCoord object that is arrays of RA, Dec
        #noting you cannot use an array of SkyCoords here
        #separations = np.array([coord.separation(c).arcsec for c in accepted_coords])
        if np.count_nonzero(np.array([coord.separation(c).arcsec for c in accepted_coords]) < 1.5) > 0:
            continue


    if dust:
        dust_corr = deredden_spectra(G.CALFIB_WAVEGRID, coord)
    else:
        dust_corr = np.ones(len(G.CALFIB_WAVEGRID))



    #first grab the imaging, see if there is anything close
    #reject if nearby and large or really nearby and bright
    img_mag_fail = False
    img_sep_fail = False
    try:
        cutouts = catlib.get_cutouts(position=coord, radius=5., aperture=1.5, dynamic=False, first=True, nudge=False,
                                         filter=['r','g'],allow_bad_image='False',allow_web='False')

        if cutouts is not None and len(cutouts) > 0:
            for c in cutouts:
                if c['details'] is not None:
                    if c['details']['sep_objects'] is not None and len(c['details']['sep_objects']) > 0:
                        #there are SEP objects
                        for sep in c['details']['sep_objects']:
                            if sep['dist_curve'] == -1: #we are inside
                                img_sep_fail = True
                                break
                            elif sep['dist_curve'] <= 1.5 and sep['mag_bright'] < 25.5:
                                img_sep_fail = True
                                break
                            elif sep['dist_curve'] <= 2.5 and sep['mag_bright'] < 24.0:
                                img_sep_fail = True
                                break
                            elif sep['a'] > 3.0 and sep['dist_curve'] < sep['a']: #already know we are outside the ellipse, so dist != -1
                                img_sep_fail = True
                                break

                    if not img_sep_fail and c['details']['elixer_apertures'] is not None and len(c['details']['elixer_apertures']) > 0:
                        #there are apertures at the exact position
                        c_idx = c['details']['elixer_aper_idx']
                        elix_img_mag = c['details']['elixer_apertures'][c_idx]['mag_bright']

                        #treating r and g as roughly equivalent here
                        #also just assuming that the imaging depth is sufficient
                        if elix_img_mag < min_gmag:
                            img_mag_fail = True
                            break


    except Exception as e:
        #but keep going anyway
        print(e)

    if img_mag_fail or img_sep_fail:
        continue

    try:

        #this one does NOT use the fiber_flux_offset, but if it passes, we will call again with one that does use it
        apt = get_spectra(coord, survey=survey_name, shotid=shot,
                          ffsky=ffsky, multiprocess=True, rad=aper,
                          tpmin=0.0, fiberweights=True, loglevel="ERROR",
                          fiber_flux_offset = None)
        try:
            if apt['fiber_weights'][0].shape[0] < min_fibers:
                continue  # too few fibers, probably on the edge or has dead fibers
        except Exception as e:
            continue


        len_spec = len(apt['spec'][0])
        #check for overly negative spectrum (number of bins)
        try:
            negsel = np.where(np.array(apt['spec'][0]) > 0)[0]
            #would normally expect about half to be negative is this is an "empty" fiber
            if len(negsel) < (0.25 * len_spec):  # pretty weak check, but if it fails something is very wrong
                continue

            #more than 10% are exactly zero or NaN, there is something very wrong
            if np.count_nonzero(apt['spec'][0])/len_spec < 0.9 or np.count_nonzero(apt['spec_err'][0])/len_spec < 0.9 or \
               np.count_nonzero(np.isnan(apt['spec'][0]))/len_spec > 0.1 or \
               np.count_nonzero(np.isnan(apt['spec_err'][0]))/len_spec > 0.1:
                continue
        except:
            pass


        # fluxd = np.nan_to_num(apt['spec'][0]) * 1e-17
        # fluxd_err = np.nan_to_num(apt['spec_err'][0]) * 1e-17
        # lets leave the nans in place
        fluxd = np.array(apt['spec'][0]) * 1e-17
        fluxd_err = np.array(apt['spec_err'][0]) * 1e-17
        wavelength = np.array(apt['wavelength'][0])



#####   #it is convenient to do this HERE since we apply this to the original aperture,
        #the per_fiber_corr (if set) is done later
        if per_aper_corr:
            if aperture_sky_subtraction_residual is None:
                aperture_sky_subtraction_residual, aperture_sky_subtraction_residual_err = \
                    SU.get_background_residual(shotid=shot, rtype="aper3.5", persist=True,
                                               dered=False, ffsky=ffsky)
                aperture_sky_subtraction_residual *= 1e-17
                aperture_sky_subtraction_residual_err *= 1e-17

            if aperture_sky_subtraction_residual is not None:
                fluxd_offset = fluxd - aperture_sky_subtraction_residual
                fluxd_offset_err = np.sqrt(fluxd_err**2 + aperture_sky_subtraction_residual_err**2)

                if applydust:
                    # dust_corr = deredden_spectra(wavelength, coord)
                    fluxd_offset *= dust_corr
                    fluxd_offset_err *= dust_corr
            else:
                fluxd_offset = np.full(len(fluxd), np.nan)
                fluxd_offset_err = np.full(len(fluxd), np.nan)

#########

        if applydust:
            #dust_corr = deredden_spectra(wavelength, coord)
            fluxd *= dust_corr
            fluxd_err *= dust_corr

        ##
        ## check the extracted spectrum for evidence of continuum or emission lines
        ##


        g, c, ge, ce = elixer_spectrum.get_hetdex_gmag(fluxd, wavelength, fluxd_err)
        # g,ge,_,_ = elixer_spectrum.get_best_gmag(fluxd,fluxd_err,wavelength) #weird astropy issue, just use hetdex_gmag
        # if g is None:
        #    print(f"bad g {ra},{dec}")
        #    continue #point source mag is too bright, likely has something in it
        # g == None for negative continuum or below limit, so None itself is okay, need to check continuum
        if g is not None and g < min_gmag:
            #reason 1 ... too bright
            reject_file.write(f"{ra}  {dec}  {shot} 1\n")
            reject_file.flush()
            continue  # point source mag is too bright, likely has something in it
        elif c < negative_flux_limit: #g might be None, if continuum is negative, then we don't get a magnitude
            #reason 2 ... too negative
            reject_file.write(f"{ra}  {dec}  {shot} 2\n")
            reject_file.flush()
            continue  # point source mag is too bright, likely has something in it

        dex_cont = c
        dex_cont_err = ce

        if g is None or np.isnan(g):
            dex_g = 99.0
            dex_g_err = 0
        else:
            dex_g = g
            dex_g_err = ge
        ##
        ## check for excess in chunks of wavelength range, particularly in the blue, say 3500-3800
        ##
        fail = False
        for chunk, ok_fluxd in zip(wave_continuum_chunks_idx, acceptable_fluxd):
            # can't use the normal get_hetdex_gmag as it has safeties on the width of the spectrum and the covered wavelengths
            if ok_fluxd[0] < np.nanmedian(fluxd[chunk]) < ok_fluxd[1]:
                # acceptable
                pass
            else:
                fail = True
                # reason 3 ... chunk out of range
                reject_file.write(f"{ra}  {dec}  {shot} 3\n")
                reject_file.flush()
                break

        if fail:
            continue

        # check for any emission lines ... simple scan? or should we use elixer's method?
        #2022-12-09 should re-run and use this with fluxd*2.0 since that is how it is calibrated
        pos, status = elixer_spectrum.sn_peakdet_no_fit(wavelength, fluxd*2.0, fluxd_err*2.0, dx=3, dv=3, dvmx=4.0,
                                                        absorber=False,
                                                        spec_obj=None, return_status=True)
        if status != 0:
            # reason 4 ... emission line found
            reject_file.write(f"{ra}  {dec}  {shot} 4\n")
            reject_file.flush()
            continue  # some failure or 1 or more possible lines


        #todo: check for HETDEX detections? could check imaging, but would want to limit to anything brigher than 25.0 or 25.5
        fail = False
        sel_coord = np.array(abs(dex_ra - ra) < 5.0/3600) & np.array(abs(dex_dec-dec) < 5.0/3600)
        if np.count_nonzero(sel_coord) > 0:
            c1 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            carray = SkyCoord(ra=dex_ra[sel_coord] * u.deg, dec=dex_dec[sel_coord] * u.deg)
            dist = c1.separation(carray).value * 3600.
            if np.any(dist <= hetdex_nearest_detection):
                fail = True

        # elif len(pos) > 1:
        #    continue #possible emission lines

        if fail:
            continue

        accepted_coords.append(coord)
        sequential_fails = 0

        fiber_weights = sorted(apt['fiber_weights'][0][:,2])[::-1]
        norm_weights = fiber_weights/np.sum(fiber_weights)
        fiber_weights = np.pad(fiber_weights,(0,32-len(fiber_weights)))
        norm_weights = np.pad(norm_weights, (0, 32 - len(norm_weights)))

        if False:
            f50, apcor = SU.get_fluxlimits(ra, dec, [3800.0, 5000.0], shot)
            if f50 is None:
                f50 = [-1,-1]
                print("Error. f50 is None.")

            if apcor is None:
                apcor = [-1,-1]
                print("Error. apcor is None.")
        else:
            f50 = [-1, -1]
            apcor = [-1, -1]

        # FUTURE/OPTIONAL: Should we check the imaging?? run a cutout and SEP check and only accept if nothing found?

        #
        # will run elixer on all these positions as well to make sure there are not other issues
        #
        # this is acceptable coords, so save:

        fluxd_sum = np.nansum(fluxd[215:966])
        fluxd_sum_wide = np.nansum(fluxd[65:966])
        fluxd_median = np.nanmedian(fluxd[215:966])
        fluxd_median_wide = np.nanmedian(fluxd[65:966])

        outfile.write(f"{ra}  {dec}  {shot}\n")
        outfile.flush()

        if per_fiber_corr:
            try:
                #sky_subtraction_residual = SU.fetch_universal_single_fiber_sky_subtraction_residual(
                #    ffsky=ffsky, hdr="3")
                #adjust_type  0 = default (none), 1 = multiply  2 = add, 3 = None
                #fiber_flux_offset = -1 * SU.adjust_fiber_correction_by_seeing(sky_subtraction_residual, seeing, adjust_type=3)

                if per_fiber_corr:
                    sky_subtraction_residual = SU.interpolate_universal_single_fiber_sky_subtraction_residual(
                        seeing, ffsky=ffsky, hdr="3",zeroflat=False,response=response, xfrac=1.0)

                    fiber_flux_offset = -1 * sky_subtraction_residual

                    apt_offset = get_spectra(coord, survey=survey_name, shotid=shot,
                                             ffsky=ffsky, multiprocess=True, rad=aper,
                                             tpmin=0.0, fiberweights=True, loglevel="ERROR",
                                             fiber_flux_offset=fiber_flux_offset)

                    fluxd_offset = np.array(apt_offset['spec'][0]) * 1e-17
                    fluxd_offset_err = np.array(apt_offset['spec_err'][0]) * 1e-17
                    #wavelength = np.array(apt['wavelength'][0])

                if applydust:
                    #dust_corr = deredden_spectra(wavelength, coord)
                    fluxd_offset *= dust_corr
                    fluxd_offset_err *= dust_corr

            except Exception as e:
                print(e)
                fluxd_offset = np.full(len(fluxd), np.nan)
                fluxd_offset_err = np.full(len(fluxd),np.nan)
        elif not per_aper_corr: #only set to nans if we did not already do the per apeture version near the top
            fluxd_offset = np.full(len(fluxd), np.nan)
            fluxd_offset_err = np.full(len(fluxd), np.nan)

        #aperture level
        T.add_row([ra, dec, shotid, seeing, response, apcor[1], f50[0], f50[1],
                   dex_g, dex_g_err, dex_cont, dex_cont_err,
                   fluxd_sum,fluxd_sum_wide,fluxd_median,fluxd_median_wide,
                   fluxd, fluxd_err,fiber_weights,norm_weights,fluxd_offset, fluxd_offset_err,dust_corr])


        if bothsky:
            apt = get_spectra(coord, survey=survey_name, shotid=shot,
                              ffsky=True, multiprocess=True, rad=aper,
                              tpmin=0.0, fiberweights=True, loglevel="ERROR",
                              fiber_flux_offset=None)
            try:
                if apt['fiber_weights'][0].shape[0] < min_fibers:
                    continue  # too few fibers, probably on the edge or has dead fibers
            except Exception as e:
                continue

            # fluxd = np.nan_to_num(apt['spec'][0]) * 1e-17
            # fluxd_err = np.nan_to_num(apt['spec_err'][0]) * 1e-17
            # lets leave the nans in place
            fluxd = np.array(apt['spec'][0]) * 1e-17
            fluxd_err = np.array(apt['spec_err'][0]) * 1e-17
            wavelength = np.array(apt['wavelength'][0])

            if applydust:
                #dust_corr = deredden_spectra(wavelength, coord)
                fluxd *= dust_corr
                fluxd_err *= dust_corr

            gff, cff, geff, ceff = elixer_spectrum.get_hetdex_gmag(fluxd, wavelength, fluxd_err)
            dex_cont_ff = cff
            dex_cont_err_ff = ceff

            if gff is None or np.isnan(gff):
                dex_g_ff = 99.0
                dex_g_err_ff = 0
            else:
                dex_g_ff = gff
                dex_g_err_ff = geff



            #aperture level FFsky (Some of this is repeat from the local sky)
            T3.add_row([ra, dec, shotid, seeing, response, apcor[1], f50[0], f50[1],
                   dex_g, dex_g_err, dex_cont, dex_cont_err,
                   dex_g_ff, dex_g_err_ff, dex_cont_ff, dex_cont_err_ff,
                   fluxd_sum,fluxd_sum_wide,fluxd_median,fluxd_median_wide,
                   fluxd, fluxd_err,fiber_weights,norm_weights,fluxd_offset, fluxd_offset_err,dust_corr])


        try:
             #fiber level
            #get the fibers at the coord for the aperture size
            _, _, _, _, fiber_ra, fiber_dec, spec, spece, _, _, _ = E.get_fiberinfo_for_coord(coord,
                                                    radius=aper,ffsky=ffsky,return_fiber_info=True,
                                                    fiber_lower_limit=3, verbose=False, fiber_flux_offset=None)

            if applydust:
                # dust_corr = deredden_spectra(wavelength, coord)
                spec *= dust_corr
                spece *= dust_corr

            for i in range(len(fiber_ra)):
                T2.add_row([ra,dec,fiber_ra[i], fiber_dec[i],shotid,seeing,spec[i],spece[i],dust_corr])
        except Exception as e:
            print("Exception (1b) !", e)
            continue


        if bothsky:
            try:
                #fiber level ffsky
                #get the fibers at the coord for the aperture size
                _, _, _, _, fiber_ra, fiber_dec, spec, spece, _, _, _ = E.get_fiberinfo_for_coord(coord,
                                                        radius=aper,ffsky=True,return_fiber_info=True,
                                                        fiber_lower_limit=3, verbose=False, fiber_flux_offset=None)

                if applydust:
                    # dust_corr = deredden_spectra(wavelength, coord)
                    spec *= dust_corr
                    spece *= dust_corr

                for i in range(len(fiber_ra)):
                    T4.add_row([ra,dec,fiber_ra[i], fiber_dec[i],shotid,seeing,spec[i],spece[i],dust_corr])
            except Exception as e:
                print("Exception (1d) !", e)
                continue

        aper_ct += 1


        if aper_ct % write_every == 0:
            T.write(op.join(tmppath,table_outname), format='fits', overwrite=True)
            T2.write(op.join(tmppath,table_outname2), format='fits', overwrite=True)
            if bothsky:
                T3.write(op.join(tmppath,table_outname3), format='fits', overwrite=True)
                T4.write(op.join(tmppath,table_outname4), format='fits', overwrite=True)

    except Exception as e:
        print("Exception (2) !", e)
        continue

T.write(op.join(tmppath,table_outname), format='fits', overwrite=True)
T2.write(op.join(tmppath,table_outname2), format='fits', overwrite=True)
if bothsky:
    T3.write(op.join(tmppath,table_outname3), format='fits', overwrite=True)
    T4.write(op.join(tmppath,table_outname4), format='fits', overwrite=True)


print(f"{shotid} Copying from /tmp : {datetime.datetime.now()}")
shutil.copy2(op.join(tmppath,table_outname),table_outname)
shutil.copy2(op.join(tmppath,table_outname2),table_outname2)
shutil.copy2(op.join(tmppath,table_outname3),table_outname3)
shutil.copy2(op.join(tmppath,table_outname4),table_outname4)
shutil.copy2(op.join(tmppath,reject_outname),reject_outname)

print(f"{shotid} Done copying: {datetime.datetime.now()}")

os.remove(op.join(tmppath,table_outname))
os.remove(op.join(tmppath,table_outname2))
os.remove(op.join(tmppath,table_outname3))
os.remove(op.join(tmppath,table_outname4))
os.remove(op.join(tmppath,reject_outname))