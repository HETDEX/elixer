from __future__ import print_function
import global_config as G

log = G.logging.getLogger('match_summary_logger')
log.setLevel(G.logging.DEBUG)

class BidTarget:
    def __init__(self):
        self.bid_ra = 361.00
        self.bid_dec = 181.00
        self.distance = 0.0
        self.bid_flux_est_cgs = 0.0
        self.bid_flux_f606w_cgs = 0.0
        self.bid_flux_f814w_cgs = 0.0
        # self.bid_flux_f105w_cgs = 0.0
        self.bid_flux_f125w_cgs = 0.0
        # self.bid_flux_f140w_cgs = 0.0
        self.bid_flux_f160w_cgs = 0.0
        # todo: flux_errors??
        # todo: other filters (depends on catalog)
        # todo: filter exposure time??

class Match:
    def __init__(self,emis=None):
        #self.match_num = 0
        if emis is None:
            self.detect_id = 0
            self.emis_ra = 361.00
            self.emis_dec = 181.00
            self.emis_wavelength = 0.0
            self.emis_sigma = 0.0
            self.emis_chi2 = 0.0
            self.emis_sn = 0.0
            self.fluxfrac = 1.0
            self.emis_flux_count = 0.0
            self.emis_flux_cgs = 0.0
            #todo: self.emis_peak_fwhm ??
            #todo:
            self.emis_cont_flux_count = 0.0
            self.emis_cont_flux_cgs = 0.0
            self.emis_obs_eqw = 0.0
        else:
            self.detect_id = emis.id
            if emis.wra is not None:
                self.emis_ra = emis.wra
                self.emis_dec = emis.wdec
            else:
                self.emis_ra = emis.ra
                self.emis_dec = emis.dec

            self.emis_wavelength = emis.w
            if emis.panacea:
                self.emis_sn = emis.sigma
                self.emis_sigma = -99999
                self.emis_chi2 = -99999
            else:
                self.emis_sigma = emis.sigma
                self.emis_chi2 = emis.chi2
                self.emis_sn = -99999
            self.emis_flux_frac = emis.fluxfrac
            self.emis_flux_count = emis.dataflux
            self.emis_flux_cgs = emis.estflux #emis.dataflux * G.FLUX_CONVERSION / emis.fluxfrac
            self.emis_cont_flux_count = emis.cont
            self.emis_cont_flux_cgs = self.emis_cont_flux_count * G.FLUX_CONVERSION
            if self.emis_cont_flux_count != 0:
                self.emis_obs_eqw = -1 * self.emis_flux_count / self.emis_cont_flux_count
            else:
                self.emis_obs_eqw = -99999

        self.bid_targets = []

    def add_bid_target(self,bid):
        if bid is not None:
            self.bid_targets.append(bid)


class MatchSet:
    headers = [
        "entry number",
        "detect ID",
        "emission line RA (decimal degrees)",
        "emission line Dec (decimal degrees)",
        "emission line wavelength (AA)",
        "emission line sigma (significance)",
        "emission line chi2 (point source fit)",
        "emission line S/N",
        "emission line estimated fraction of recovered flux",
        "emission line flux (electron counts)",
        "emission line flux (cgs)",
        "emission line continuum flux (electron counts)",
        "emission line continuum flux (cgs)",
        "emission line observed (estimated) equivalent width (not restframe)",
        "catalog object RA (decimal degrees)",
        "catalog object Dec (decimal degrees)",
        "catalog object separation (arcsecs)",
        "catalog object continuum flux est at emission line wavelength (cgs)",
        "catalog object flux f606w at emission line wavelength (cgs)",
        "catalog object flux f814w at emission line wavelength (cgs)",
        "catalog object flux f160w at emission line wavelength (cgs)",
        "catalog object flux f125w at emission line wavelength (cgs)"
    ]

    def __init__(self):
        self.match_set = [] #list of all Match objects

    def add(self,match):
        if match is not None:
            self.match_set.append(match)

    @property
    def size(self):
        return len(self.match_set)

    def write_file(self,filename):

        if filename is not None:
            sep = "\t"
            try:
                f = open(filename,'w')
            except:
                log.error("Exception create match summary file: %s" %filename, exc_info=True)
                return None

            # write help (header) part
            f.write("# each row contains one emission line and one matched imaging catalog counterpart\n")
            f.write("# the same emission line may repeat with additional possible imaging catalog counterparts\n")
            col_num = 0
            for h in self.headers:
                col_num += 1
                f.write("# %d %s\n" % (col_num, h))

            entry_num = 0
            for m in self.match_set:
                for b in m.bid_targets:
                    entry_num += 1
                    f.write(str(entry_num))
                    f.write(sep + str(m.detect_id))
                    f.write(sep + str(m.emis_ra))
                    f.write(sep + str(m.emis_dec))
                    f.write(sep + str(m.emis_wavelength))
                    f.write(sep + str(m.emis_sigma))
                    f.write(sep + str(m.emis_chi2))
                    f.write(sep + str(m.emis_sn))
                    f.write(sep + str(m.emis_flux_frac))
                    f.write(sep + str(m.emis_flux_count))
                    f.write(sep + str(m.emis_flux_cgs))
                    f.write(sep + str(m.emis_cont_flux_count))
                    f.write(sep + str(m.emis_cont_flux_cgs))
                    f.write(sep + str(m.emis_obs_eqw))
                    f.write(sep + str(b.bid_ra))
                    f.write(sep + str(b.bid_dec))
                    f.write(sep + str(b.distance))
                    f.write(sep + str(b.bid_flux_est_cgs))
                    f.write(sep + str(b.bid_flux_f606w_cgs))
                    f.write(sep + str(b.bid_flux_f814w_cgs))
                    f.write(sep + str(b.bid_flux_f125w_cgs))
                    f.write(sep + str(b.bid_flux_f160w_cgs))
                    f.write("\n")

            f.close()


