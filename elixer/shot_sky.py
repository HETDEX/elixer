"""
Figure the sky residual for a single shot
"""
try:
    from elixer import global_config as G
except:
    import global_config as G

import numpy as np
import os
import tables
from astropy.table import Table

log = G.Global_Logger('shot_sky')
log.setlevel(G.LOG_LEVEL)
#shot_path = G.PANACEA_HDF5_BASEDIR #"/data/05350/ecooper/hdr2.1/reduction/data/"
AMP_FLAG_TABLE = None

STACKING_AVG_METHOD = "biweight" #"mean_68" "mean_95", "mean_std", "median" "biweight" "weighted_biweight"

class DEX_Fiber:
    """
    A variation on Fiber, but closely associated with the HETDEX representation / storage of fiber data
    """
    def __init__(self,h5_path=None,fiberid=None):

        self.fiberid = None #aka full "address" string like: '20170221v016e001_multi_016_104_026_RU_n001'
        self.shotid = 0 #as integer
        self.datevshot = ""
        self.expnum = -1 #as integer 1-3
        self.multiframe = ""
        self.fiber_num = -1 #1-112

        self.h5_path = h5_path

        self.calfib = []
        self.calfibe = []
        self.spec_fullsky_sub = []

        self.lo_sum = 0 #sum of values in calfib 3600AA-5400AA
        self.ff_sum = 0 #ditto for spec_fullsky_sub
        self.er_sum = 0 #calfibe in quadrature

        if fiberid is not None:
            self.parse_fiberid(fiberid)


    def parse_fiberid(self,fiberid=None):
        if fiberid is not None:
            self.fiberid = fiberid

        toks = self.fiberid.split("_")
        no_v = toks[0].replace("v", "")
        self.shotid = int(no_v) #v may not be there
        self.datevshot = no_v[0:8] + 'v' + no_v[8:]
        self.expnum = int(toks[1]) #1-3
        self.multiframe = "_".join(toks[2:7])
        self.fiber_num = int(toks[7])  # 1-112


    def load_spectra(self,ff,er):
        """
        takes the full fiberid (or address) and loads the spectra for it
        :param fiberid:
        :return:
        """
        #'"20170303014_3_multi_025_076_032_LU_003"'
        try:
            self.calfibe = er
            self.spec_fullsky_sub =ff
            self.sum()

        except:
            log.info("Exception in shot_sky",exc_info=True)

    def sum(self): #3600 t0 5400
        #self.lo_sum = np.sum(self.calfib[65:966]) #avoid the ends, and bad skylines
        self.ff_sum = np.sum(self.spec_fullsky_sub[65:966]) #avoid the ends, and bad skylines
        self.er_sum = np.sqrt(np.sum(self.calfibe[65:966] ** 2))


def load_amp_flag_table(path=G.BAD_AMP_TABLE):
    global AMP_FLAG_TABLE
    try:
        AMP_FLAG_TABLE = Table.read(path)
        return AMP_FLAG_TABLE
    except:
        log.info("Exception in shot_sky", exc_info=True)


def amp_okay(mf,shotid,bad_amp_table=AMP_FLAG_TABLE):
    try:
        if bad_amp_table is None:
            bad_amp_table = load_amp_flag_table()

        t = bad_amp_table
        search = ((t['shotid'] == shotid) & (t['multiframe'] == mf.decode()))
        flag = t[search]['flag']
        if len(flag) == 1:
            if flag[0] == 0:
                return False
        elif len(flag) > 1:
            return False
    except:
        log.info("Exception in shot_sky", exc_info=True)
        return False

    return True


def get_multiframes_from_shot(shotid):
    """
    checks to make sure the returned multiframes are NOT on the bad list
    :param shotid:
    :return:
    """
    global AMP_FLAG_TABLE
    mf_list = []
    try:
        filename = os.path.join(G.PANACEA_HDF5_BASEDIR ,str(shotid)[0:8] + "v"+str(shotid)[8:] + ".h5")
        sh5 = tables.open_file(filename)
        fitb = sh5.root.Data.FiberIndex

        multiframes = np.unique(fitb.read(field="multiframe"))
        sh5.close()
        for m in multiframes:
            if amp_okay(m,shotid,AMP_FLAG_TABLE):
                mf_list.append(m)
    except:
        log.info("Exception in shot_sky",exc_info=True)

    return mf_list




def is_good_sky(flux,err,throughput,chi2):
    """
    Assumes on CALFIB wave grid (3470-5540 in steps of 2)
    Assumes in flux units (erg/s/cm^2 over 2AA ... not flux density)
    :param waves:
    :param flux:
    :param err:
    :return:
    """

    try:

        flux = np.array(flux)
        err = np.array(err)
        chi2 = np.array(chi2)
        throughput = np.array(throughput)

        #ignore zeros in throughput as meaningless
        throughput[np.where(throughput == 0)[0]] = 1.0

        if not (len(np.where(flux[25:1001] == 0)[0]) < 100):
            #print(f"zero flux {len(np.where(flux[25:1001] == 0)[0])}")
            return False

        sel = np.where(flux != 0)[0]
        if not (len(np.where(err[sel] == 0)[0]) < 10): #these should never be zero
            #print(f"zero err {len(np.where(err[25:1001] == 0)[0])}")
            return False

        if np.nanmin(throughput[25:1001]) < 0.08: #reduce a bit
            #print(f"throughput {np.nanmin(throughput)}")
            return False

        if np.nanmax(chi2[25:1001]) > 6.0:
            #print(f"chi2 {np.nanmax(chi2)}")
            return False


        max_flux = flux + err
        min_flux = flux - err

        # -0.5, 0.5, and 1.5 yeild about 10% as sky
        # -0.3, 0.3, 4.0 too strict 0 too strict
        # -0.2, 0.2, 4.0 too strict 0
        # -0.4, 0.4, 3.0 yields about 7% (most lost to the wavebin)
        # -0.3, 0.3, 3.0 too strict
        # -0.2, 0.2, 3.0 too strict (most lost to qsum)
        # -0.2, 0.2, 2.0 too strict


        #combo full, narrow, line (577174640 total fibers not in bad amps)
        #0.2   0.2    2.0  - too strict     1.34e-4  (.013 %)          ~     77K fibers
        #0.2   0.3    2.0   about right?    0.022    (2.2%) [12610202] ~ 12.6 million fibers
        #0.2   0.3    3.0                   0.027    (2.7%)            ~ 15.7 million fibers
        full_cont_floor = -0.4  #
        full_cont_ceil = 0.4  #
        narrow_cont_floor = -0.5
        narrow_cont_ceil = 0.5
        single_line_ceil = 5.0  # 2.0 #units of cgs e^-17
        adj_line_ceil = 4.0  # 2.0 #units of cgs e^-17

        sel = np.where(max_flux > single_line_ceil)[0]
        if (sel is not None) and (len(sel) > 0):
            #print(f"wavebin single fails")
            return False

        #maybe change this: find wavebins above the limit and check the bins to either side
        #if they are back below the limit, accept and move so ... need 2 continguous bins?

        sel = np.where(max_flux > adj_line_ceil)[0]
        fail = False
        if (sel is not None) and (len(sel) > 0):
            #for i in range(len(sel)-2):
            #    if (sel[i]+1 == sel[i+1]) and (sel[i]+2 == sel[i+2]): #adjacent
            for i in range(len(sel)-1):
                if sel[i]+1 == sel[i+1]: #adjacent
                    fail = True
                    #print(f"wavebin adjacent fails")
                    break
        if fail:
            return False


        #no worst case wavebin out of range
        # if (np.nanmax(max_flux) > line_ceil) or \
        #    (abs(np.nanmin(min_flux)) > line_ceil):
        #     print(f"wavebin {np.nanmax(max_flux)} {np.nanmin(min_flux)}")
        #     return False

        #overall continuum (3520-5472]
        sum_flux = np.nansum(flux[25:1001]) / 1952.
        sum_err = np.nansum(err[25:1001]) / 1952.
        if ((sum_flux + sum_err) > full_cont_ceil) or \
           ((sum_flux - sum_err) < full_cont_floor):
            #print(f"overall cont {sum_flux + sum_err} {sum_flux - sum_err}")
            return False

        # scan all 500AA wide sums:
        # [65] = 3600   [716] = 4902
        # [25] = 3520 (for 880AA*(z=3 + 1) = 3520
        # [1001] = 5472
        for i in range(65, 716, 1):
            qsum = np.nansum(flux[i:i + 500]) / 500.0
            esum = np.nansum(err[i:i + 500]) / 500.0

            if ((qsum + esum) > narrow_cont_ceil) or ((qsum - esum) < narrow_cont_floor):
                #print(f"qsum {qsum + esum} {qsum - esum} ")
                return False  #

        return True
    except:
        log.info("Exception in shot_sky",exc_info=True)
        return False




def get_all_fibers(shotid,mf_list):
    """
    gets all fibers (and all exposures) from list of multiframes within a single shot
    :param shotid:
    :return:
    """
    log.info(f"Collecting all fibers for shot {shotid}")
    fiber_list = []

    try:
        filename = os.path.join(G.PANACEA_HDF5_BASEDIR,str(shotid)[0:8] + "v"+str(shotid)[8:] + ".h5")
        sh5 = tables.open_file(filename)
        ftb = sh5.root.Data.Fibers

        for count_m, m in enumerate(mf_list):
            #print(f"     {count_m+1}/{len(mf_list)} {m.decode()} at {datetime.now().strftime('%H:%M:%S')}")
            rows = ftb.read_where("multiframe==m")

            for count_row, row in enumerate(rows):
                #lo = row['calfib']
                er = row['calfibe']
                ff = row['spec_fullsky_sub']
                c2 = row['chi2']
                tp = row['Throughput']

                #if is_good_sky(lo,er,tp,c2):
                if is_good_sky(ff, er, tp, c2):
                    #don't care anymore about identifying info
                    #just want the flux and err
                    #all are on the 1036 x2AA spacing

                    fiber = DEX_Fiber() #temporary storages
                    #20170221v016e001_multi_016_104_026_RU_n001

                    fiber.parse_fiberid(row['fiber_id'].decode())
                    fiber.load_spectra(ff,er)
                    fiber_list.append(fiber)

        sh5.close()
    except:
        log.info("Exception in shot_sky", exc_info=True)
        try:
            sh5.close()
        except:
            pass

    return fiber_list


def stack_sky_spectra(spectra, ffsky=True,straight_error=False):
    """

    Here, we stack first in observed frame, then shift to a specified "rest" frame (as
    set by the caller ... expected to be the rest frame of the detection for which this
    sky is relevant)

    :param spectra: list of Spectra objects (*note: not spectra bundle objects) to stack
    :param z: rest-z of the detection for which this is relevant (0 = observed frame)
    :param straight_error: if true, just use the error as is ... if False, divide by sqrt(N)
    :return: single StackedSpectra object
    """

    try:
        with open("count.log","w") as f:
            f.write(f"{len(spectra)}\n")

    except:
        log.info("Exception in shot_sky", exc_info=True)

    long_grid = G.CALFIB_WAVEGRID[:] #this is in the rest frame

    if ffsky:
        matrix = [x.spec_fullsky_sub for x in spectra]
    else:
        matrix = [x.calfib for x in spectra]

    er_matrix = [x.calfibe for x in spectra]

    stack_flux = np.zeros(len(long_grid))
    stack_err = np.zeros(len(long_grid))

    for i in range(len(long_grid)):
        #build error first since wslice is modified and re-assigned
        #build the error on the flux (wslice) so they stay lined up


        #have to remove the quantity here, just use the float values
        wslice_err = np.array([m[i] for m in er_matrix])  # all rows, ith index
        wslice = np.array([m[i] for m in matrix])   # all rows, ith index

        wslice_err = wslice_err[np.where(wslice > -999)]  # so same as wslice, since these need to line up
        wslice = wslice[np.where(wslice > -999)]  # git rid of the out of range interp values

        # git rid of any nans
        wslice_err = wslice_err[~np.isnan(wslice)] #err first so the wslice is not modified
        wslice_lo = wslice[~np.isnan(wslice)]


        if len(wslice_err) > 0:
            if straight_error or (STACKING_AVG_METHOD == 'mean_95'):
                try:
                    mean_cntr, var_cntr, std_cntr = scipy.stats.bayes_mvs(wslice, alpha=0.95)
                    if np.isnan(mean_cntr[0]):
                        raise( Exception('mean_ctr is nan'))
                    stack_flux[i] = mean_cntr[0]
                    #an average error
                    stack_err[i] = 0.5 * (abs(mean_cntr[0]-mean_cntr[1][0]) + abs(mean_cntr[0]-mean_cntr[1][1]))
                except:
                    log.info("Exception in shot_sky. Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..." %(i,long_grid[i]),
                             exc_info=True)
                    try:
                        stack_flux[i] = SU.biweight_location(wslice)
                        stack_err[i] = SU.biweight_scale(wslice)#* 2. #2 sigma ~ 95%
                    except:
                        log.info("Exception in shot_sky",exc_info=True)
            elif straight_error or (STACKING_AVG_METHOD == 'mean_68'):
                try:
                    mean_cntr, var_cntr, std_cntr = scipy.stats.bayes_mvs(wslice, alpha=0.68)
                    if np.isnan(mean_cntr[0]):
                        raise (Exception('mean_ctr is nan'))
                    stack_flux[i] = mean_cntr[0]
                    # an average error
                    stack_err[i] = 0.5 * (abs(mean_cntr[0] - mean_cntr[1][0]) + abs(mean_cntr[0] - mean_cntr[1][1]))
                except:
                    log.info("Exception in shot_sky Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..."
                             % (i, long_grid[i]), exc_info=True)
                    try:
                        stack_flux[i] = SU.biweight_location(wslice)
                        stack_err[i] = SU.biweight_scale(wslice)   # * 2. #2 sigma ~ 95%
                    except:
                        log.info("Exception in shot_sky", exc_info=True)
            elif (STACKING_AVG_METHOD == 'mean_std'):
                try:
                    stack_flux[i] = np.nanmean(wslice)
                    # an average error
                    stack_err[i] = np.nanstd(wslice) / np.sqrt(len(wslice))
                except:
                    log.info("Exception in shot_sky. Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..."
                             %(i,long_grid[i]),exc_info=True)
                    try:
                        stack_flux[i] = SU.biweight_location(wslice)
                        stack_err[i] = SU.biweight_scale(wslice)#* 2. #2 sigma ~ 95%
                    except:
                        log.info("Exception in shot_sky", exc_info=True)

            elif (STACKING_AVG_METHOD == 'median'):
                try:
                    stack_flux[i] = np.nanmedian(wslice)
                    #an average error
                    stack_err[i] =np.nanstd(wslice)/np.sqrt(len(wslice))
                except Exception as e:
                    log.info("Exception in shot_sky. Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..."
                             %(i,long_grid[i]),exc_info=True)

                    try:
                        stack_flux[i] = SU.biweight_location(wslice)
                        stack_err[i] = SU.biweight_scale(wslice) #* 2. #2 sigma ~ 95%
                    except:
                        log.info("Exception in shot_sky", exc_info=True)

            elif (STACKING_AVG_METHOD == 'biweight'):
                try:
                    stack_flux[i] = SU.biweight_location(wslice)
                    stack_err[i] = SU.biweight_scale(wslice) / np.sqrt(len(wslice))
                except Exception as e:
                    log.info("Exception in shot_sky.Straight Error failed (iter=%d,wave=%f). Switching to biweight at 2 sigma  ..."
                             %(i,long_grid[i]),exc_info=True)
                    try:
                        stack_flux[i] = SU.biweight_location(wslice)
                        stack_err[i] = SU.biweight_scale(wslice) #* 2. #2 sigma ~ 95%
                    except:
                        log.info("Exception in shot_sky", exc_info=True)
            else: #weighted_biweight
            #definitely keep the scale defaults (c=6,c=9) per Karl, etc .. these best wieght for Gaussian limits
                try:
                    stack_flux[i] = SU.biweight_location_errors(wslice, errors=wslice_err)
                    stack_err[i] = SU.biweight_scale(wslice) / np.sqrt(len(wslice))
                except:
                    log.info("Exception in shot_sky", exc_info=True)
                    stack_flux[i] = SU.biweight_location(wslice)
                    stack_err[i] = SU.biweight_scale(wslice) / np.sqrt(len(wslice))

    return stack_flux, stack_err




def get_shot_sky_residual(shotid):
    """
    Really only make sense for the ffsky subtracted data, so that is the assumption

    :param shotid:
    :return:
    """

    log.info(f"Building {shotid} sky residual ... ")
    try:
        ff_stack, ff_er_stack = None, None
        #check every multiframe in shot to see if is good
        mf_list = get_multiframes_from_shot(shotid)

        if mf_list is None or len(mf_list) == 0:
            return None, None

        #collect all the fibers
        fiber_list = get_all_fibers(shotid,mf_list)

        if fiber_list is None or len(fiber_list) == 0:
            return None, None

        #now perform the sky-subselection (interior 2/3) on lo and ff (separetly)
        fiber_list.sort(key=lambda x: x.ff_sum, reverse=True)
        ff_selected_fiberids = [x.fiberid for x in fiber_list[int(0.17 * len(fiber_list)):-int(0.17 * len(fiber_list))]]
        ff_selected_fibers = fiber_list[int(0.17 * len(fiber_list)):-int(0.17 * len(fiber_list))]

        #now stack (biweight) in observed frame
        ff_stack, ff_er_stack = stack_sky_spectra(ff_selected_fibers, ffsky=True)  # the er_stacks should be the same
    except:
        log.info("Exception in shot_sky", exc_info=True)

    return ff_stack, ff_er_stack





