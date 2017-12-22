import global_config as G
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import io


log = G.logging.getLogger('spectrum_logger')
log.setLevel(G.logging.DEBUG)

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

    # 2d spectra cutouts (one per fiber)
    def build_full_width_spectrum(self, counts, wavelengths, central_wavelength = 0, show_skylines=True, name=None):

        # fig = plt.figure(figsize=(5, 6.25), frameon=False)
        fig = plt.figure(figsize=(8, 3), frameon=False)
        plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)

        dy = 1.0 / 5.0  # + 1 skip for legend, + 2 for double height spectra + 2 for double height labels

        # this is the 1D averaged spectrum
        specplot = plt.axes([0.05, 0.10, 0.90, 0.80])

        # they should all be the same length
        # yes, want round and int ... so we get nearest pixel inside the range)
        left = wavelengths[0]
        right = wavelengths[-1]

        try:

            mn = np.min(counts)
            mn = max(mn, -20)  # negative flux makes no sense (excepting for some noise error)
            mx = np.max(counts)
            ran = mx - mn
            specplot.step(wavelengths, counts, c='b', where='mid', lw=1)

            specplot.axis([left, right, mn - ran / 20, mx + ran / 20])

            specplot.locator_params(axis='y', tight=True, nbins=4)

            textplot = plt.axes([0.025, 3.0 * dy, 0.95, dy * 2])
            textplot.set_xticks([])
            textplot.set_yticks([])
            textplot.axis(specplot.axis())
            textplot.axis('off')

            # iterate over all emission lines ... assume the cwave is that line and plot the additional lines

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
            skipplot = plt.axes([.025, 1.0*dy, 0.95, dy])
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
                pass

        if name is not None:
            plt.savefig(name+".png", format='png', dpi=300)
            buf = None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)

        plt.close(fig)
        return buf


