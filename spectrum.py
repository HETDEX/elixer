import global_config as G
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
#import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import io
from scipy.stats import gmean
from scipy import signal


log = G.logging.getLogger('spectrum_logger')
log.setLevel(G.logging.DEBUG)

MIN_FWHM = 5
MIN_HEIGHT = 20
MIN_DELTA_HEIGHT = 2 #to be a peak, must be at least this high above next adjacent point to the left

def peakdet(x,v,dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0):

    #peakind = signal.find_peaks_cwt(v, [2,3,4,5],min_snr=4.0) #indexes of peaks

    #emis = zip(peakind,x[peakind],v[peakind])
    #emistab.append((pi, px, pv, pix_width, centroid))
    #return emis



    #dh (formerly, delta)
    #dw (minimum width (as a fwhm) for a peak, else is noise and is ignored) IN PIXELS
    # todo: think about jagged peaks (e.g. a wide peak with many subpeaks)
    #zero is the count level zero (nominally zero, but with noise might raise or lower)
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []
    emistab = []
    delta = dh

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)
    num_pix = len(v)

    if num_pix != len(x):
        log.warning('peakdet: Input vectors v and x must have same length')
        return None,None

    if not np.isscalar(dh):
        log.warning('peakdet: Input argument delta must be a scalar')
        return None, None

    if dh <= 0:
        log.warning('peakdet: Input argument delta must be positive')
        return None, None

    minv, maxv = np.Inf, -np.Inf
    minpos, maxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        thisv = v[i]
        if thisv > maxv:
            maxv = thisv
            maxpos = x[i]
            maxidx = i
        if thisv < minv:
            minv = thisv
            minpos = x[i]
            minidx = i
        if lookformax:
            if (thisv >= h) and (thisv < maxv - delta):
                #i-1 since we are now on the right side of the peak and want the index associated with max
                maxtab.append((maxidx,maxpos, maxv))
                minv = thisv
                minpos = x[i]
                lookformax = False
        else:
            if thisv > minv + delta:
                mintab.append((minidx,minpos, minv))
                maxv = thisv
                maxpos = x[i]
                lookformax = True


    #make an array, slice out the 3rd column
    #gm = gmean(np.array(maxtab)[:,2])
    peaks = np.array(maxtab)[:, 2]
    gm = np.mean(peaks)
    std = np.std(peaks)

    #now, throw out anything waaaaay above the mean (toss out the outliers and recompute mean)
    sub = peaks[np.where(peaks < (5.0*std))[0]]
    gm = np.mean(sub)

    for pi,px,pv in maxtab:
        #check fwhm (assume 0 is the continuum level)

        #minium height above the mean of the peaks (w/o outliers)
        if (pv < 1.333 * gm):
            continue

        hm = float((pv - zero) / 2.0)
        pix_width = 0

        #for centroid (though only down to fwhm)
        sum_pos_val = x[pi] * v[pi]
        sum_pos = x[pi]
        sum_val = v[pi]

        #check left
        pix_idx = pi -1

        try:
            while (pix_idx >=0) and (v[pix_idx] >= hm):
                sum_pos += x[pix_idx]
                sum_pos_val += x[pix_idx] * v[pix_idx]
                sum_val += v[pix_idx]
                pix_width += 1
                pix_idx -= 1

        except:
            pass

        #check right
        pix_idx = pi + 1

        try:
            while (pix_idx < num_pix) and (v[pix_idx] >= hm):
                sum_pos += x[pix_idx]
                sum_pos_val += x[pix_idx] * v[pix_idx]
                sum_val += v[pix_idx]
                pix_width += 1
                pix_idx += 1
        except:
            pass

        #check local region around centroid
        centroid_pos = sum_pos_val / sum_val #centroid is an index

        #what is the average value in the vacinity of the peak (exlcuding the area under the peak)
        side_pix = max(20,pix_width)
        left = max(0,(pi - pix_width)-side_pix)
        sub_left = v[left:(pi - pix_width)]
        gm_left = np.mean(v[left:(pi - pix_width)])

        right = min(num_pix,pi+pix_width+side_pix+1)
        sub_right = v[(pi + pix_width):right]
        gm_right = np.mean(v[(pi + pix_width):right])

        #minimum height above the local gm_average
        #note: can be a problem for adjacent peaks?
        if pv < (2.0 * np.mean(np.concatenate((sub_left,sub_right)))):
            continue

        #check vs minimum width
        if not (pix_width < dw):
            #see if too close to prior peak (these are in increasing wavelength order)
            if len(emistab) > 0:
                if (px - emistab[-1][1]) > 6.0:
                    emistab.append((pi, px, pv,pix_width,centroid_pos))
                else: #too close ... keep the higher peak
                    if pv > emistab[-1][2]:
                        emistab.pop()
                        emistab.append((pi, px, pv, pix_width, centroid_pos))
            else:
                emistab.append((pi, px, pv, pix_width, centroid_pos))


    #return np.array(maxtab), np.array(mintab)
    return emistab


class EmissionLine():
    def __init__(self,name,w_rest,plot_color,solution=True,z=0):
        self.name = name
        self.w_rest = w_rest
        self.w_obs = w_rest * (1.0 + z)
        self.z = z
        self.color = plot_color
        self.solution = solution #True = can consider this as the target line

    def redshift(self,z):
        self.z = z
        self.w_obs = self.w_rest * (1.0 + z)
        return self.w_obs


class Spectrum:

    def __init__(self):

        self.emission_lines = [EmissionLine("Ly$\\alpha$ ", 1216, 'red'),
                               EmissionLine("OII ", 3727, 'green'),
                               EmissionLine("OIII", 4959, "lime"), EmissionLine("OIII", 5007, "lime"),
                               EmissionLine("CIII", 1909, "purple"),
                               EmissionLine("CIV ", 1549, "black"),
                               EmissionLine("H$\\beta$ ", 4861, "blue"),
                               EmissionLine("HeII", 1640, "orange"),
                               EmissionLine("MgII", 2798, "magenta", solution=False),
                               EmissionLine("H$\\gamma$ ", 4341, "royalblue", solution=False),
                               EmissionLine("NV ", 1240, "teal", solution=False),
                               EmissionLine("SiII", 1260, "gray", solution=False)]


    def build_full_width_spectrum(self, counts, wavelengths, central_wavelength = 0,
                                  show_skylines=True, show_peaks = True, name=None,
                                  dw=MIN_FWHM,h=MIN_HEIGHT,dh=MIN_DELTA_HEIGHT,zero=0.0):

        # fig = plt.figure(figsize=(5, 6.25), frameon=False)
        fig = plt.figure(figsize=(8, 3), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        dy = 1.0 / 5.0  # + 1 skip for legend, + 2 for double height spectra + 2 for double height labels

        # this is the 1D averaged spectrum
        #textplot = plt.axes([0.025, .6, 0.95, dy * 2])
        specplot = plt.axes([0.05, 0.20, 0.90, 0.40])
        #specplot = plt.axes([0.025, 0.20, 0.95, 0.40])

        # they should all be the same length
        # yes, want round and int ... so we get nearest pixel inside the range)
        left = wavelengths[0]
        right = wavelengths[-1]

        try:
            mn = np.min(counts)
            mn = max(mn, -20)  # negative flux makes no sense (excepting for some noise error)
            mx = np.max(counts)
            ran = mx - mn
            specplot.step(wavelengths, counts,  where='mid', lw=1)

            specplot.axis([left, right, mn - ran / 20, mx + ran / 20])

            specplot.locator_params(axis='y', tight=True, nbins=4)


            if show_peaks:
                #emistab.append((pi, px, pv,pix_width,centroid))
                peaks = peakdet(wavelengths, counts,dw,h,dh,zero)
                if (peaks is not None) and (len(peaks) > 0):
                    specplot.scatter(np.array(peaks)[:, 1], np.array(peaks)[:, 2], facecolors='none', edgecolors='r')

            #textplot = plt.axes([0.025, .6, 0.95, dy * 2])
            textplot = plt.axes([0.05, .6, 0.90, dy * 2])
            textplot.set_xticks([])
            textplot.set_yticks([])
            textplot.axis(specplot.axis())
            textplot.axis('off')

            if central_wavelength > 0:
                wavemin = specplot.axis()[0]
                wavemax = specplot.axis()[1]
                legend = []
                name_waves = []
                obs_waves = []

                for e in self.emission_lines:
                    if not e.solution:
                        continue

                    z = central_wavelength / e.w_rest - 1.0

                    if (z < 0):
                        continue

                    count = 0
                    for f in self.emission_lines:
                        if (f == e) or not (wavemin <= f.redshift(z) <= wavemax):
                            continue

                        count += 1
                        y_pos = textplot.axis()[2]
                        for w in obs_waves:
                            if abs(f.w_obs - w) < 20:  # too close, shift one vertically
                                y_pos = (textplot.axis()[3] - textplot.axis()[2]) / 2.0 + textplot.axis()[2]
                                break

                        obs_waves.append(f.w_obs)
                        textplot.text(f.w_obs, y_pos, f.name + " {", rotation=-90, ha='center', va='bottom',
                                      fontsize=12, color=e.color)  # use the e color for this family

                    if (count > 0) and not (e.name in name_waves):
                        legend.append(mpatches.Patch(color=e.color, label=e.name))
                        name_waves.append(e.name)

            # make a legend ... this won't work as is ... need multiple colors
            skipplot = plt.axes([.025,0.0, 0.95, dy])
            skipplot.set_xticks([])
            skipplot.set_yticks([])
            skipplot.axis(specplot.axis())
            skipplot.axis('off')
            skipplot.legend(handles=legend, loc='center', ncol=len(legend), frameon=False,
                            fontsize='small', borderaxespad=0)

        except:
            log.warning("Unable to build full width spec plot.", exc_info=True)

        if show_skylines:
            try:
                yl, yh = specplot.get_ylim()

                central_w = 3545
                half_width = 10
                rec = plt.Rectangle((central_w - half_width, yl), 2 * half_width, yh - yl, fill=True, lw=1,
                                    color='gray', alpha=0.5, zorder=1)
                specplot.add_patch(rec)

                central_w = 5462
                half_width = 5
                rec = plt.Rectangle((central_w - half_width, yl), 2 * half_width, yh - yl, fill=True, lw=1,
                                    color='gray', alpha=0.5, zorder=1)
                specplot.add_patch(rec)
            except:
                log.warning("Unable add skylines.", exc_info=True)

        if name is not None:
            try:
                plt.savefig(name+".png", format='png', dpi=300)
            except:
                log.warning("Unable save plot to file.", exc_info=True)


        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf


