from __future__ import print_function
import global_config as G

#log = G.logging.getLogger('match_summary_logger')
#log.setLevel(G.logging.DEBUG)
log = G.Global_Logger('match_summary_logger')
log.setlevel(G.logging.DEBUG)


class Filter:
    def __init__(self,instrument,filter,flux,err):
        #don't want spaces
        self.instrument = instrument.replace(" ","_")
        self.filter = filter.replace(" ","_")
        self.flux = flux
        self.err = err
        #note: units always cgs


class BidTarget:
    def __init__(self):
        self.bid_ra = 361.00
        self.bid_dec = 181.00
        self.distance = 0.0
        self.bid_flux_est_cgs = 0.0
        self.p_lae = None
        self.p_oii = None
        self.p_lae_oii_ratio = None
        self.catalog_name = None
        # todo: filter exposure time??

        self.filters = []

    def add_filter(self,instrument,filter,flux,err):
        self.filters.append(Filter(instrument,filter,flux,err))


class Match:
    def __init__(self,emis=None):
        #self.match_num = 0
        if emis is None:
            self.detobj = None
            self.input_id = 0
            self.detect_id = 0
            self.emis_ra = 361.00
            self.emis_dec = 181.00
            self.emis_wavelength = 0.0
            self.emis_fwhm = -1.0
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
            # this is a bit kludgey but now need info on spectra data continained within emis (v1.4.0)
            # and that also makes most of this data redundant
            self.detobj = emis
            self.input_id = emis.entry_id
            self.detect_id = emis.id
            if emis.wra is not None:
                self.emis_ra = emis.wra
                self.emis_dec = emis.wdec
            else:
                self.emis_ra = emis.ra
                self.emis_dec = emis.dec

            self.emis_wavelength = emis.w

            if emis.fwhm is not None:
                self.emis_fwhm= emis.fwhm
            else:
                self.emis_fwhm = -1.0

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
            self.emis_cont_flux_cgs = emis.cont_cgs #* G.FLUX_CONVERSION
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
        "input (entry) ID",
        "detect ID",
        "emission line RA (decimal degrees)",
        "emission line Dec (decimal degrees)",
        "number of possible catalog matches",
        "emission line wavelength (AA)",
        "emission line FWHM (AA)",
        "emission line sigma (significance)",
        "emission line chi2 (point source fit)",
        "emission line S/N",
        "emission line estimated fraction of recovered flux",
        "emission line flux (electron counts)",
        "emission line flux (cgs)",
        "emission line continuum flux (electron counts)",
        "emission line continuum flux (cgs)",
        "emission line observed (estimated) equivalent width (not restframe)",
        "catalog name",
        "catalog object RA (decimal degrees) [for the exact emission position, not matched to a catalog object, will = 361.0]",
        "catalog object Dec (decimal degrees)[for the exact emission position, not matched to a catalog object, will = 181.0]",
        "catalog object separation (arcsecs) [for the exact emission position, not matched to a catalog object, will = 0.0]",
        "catalog object continuum flux est at emission line wavelength (cgs) [for the exact emission position, from aperture on catalog image]",
        "P(LAE)/P(OII)",
        "number of filters to follow (variable)",
        "  instrument name (if available)",
        "  filter name",
        "  flux in cgs units",
        "  flux error in cgs units",
        "  the next instrument name and so on ..."
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
            f.write("# version " + str(G.__version__) + "\n")
            f.write("# each row contains one emission line and one matched imaging catalog counterpart\n")
            f.write("# the same emission line may repeat with additional possible imaging catalog counterparts\n")
            col_num = 0
            for h in self.headers:
                col_num += 1
                f.write("# %d %s\n" % (col_num, h))

            #need to get number of potential matches

            #entry_num = 0
            for m in self.match_set:
                dummy_bid = 0
                if len(m.bid_targets) == 0:
                    #make a dummy one
                    m.add_bid_target(BidTarget())
                    dummy_bid = 1

                for b in m.bid_targets:
             #       entry_num += 1
                    #f.write(str(entry_num))
                    f.write(str(m.input_id))
                    f.write(sep + str(m.detect_id))
                    f.write(sep + str(m.emis_ra))
                    f.write(sep + str(m.emis_dec))
                    f.write(sep + str(len(m.bid_targets)-dummy_bid))
                    f.write(sep + str(m.emis_wavelength))
                    f.write(sep + str(m.emis_fwhm))
                    f.write(sep + str(m.emis_sigma))
                    f.write(sep + str(m.emis_chi2))
                    f.write(sep + str(m.emis_sn))
                    f.write(sep + str(m.emis_flux_frac))
                    f.write(sep + str(m.emis_flux_count))
                    f.write(sep + str(m.emis_flux_cgs))
                    f.write(sep + str(m.emis_cont_flux_count))
                    f.write(sep + str(m.emis_cont_flux_cgs))
                    f.write(sep + str(m.emis_obs_eqw))

                    f.write(sep + str(b.catalog_name))
                    f.write(sep + str(b.bid_ra))
                    f.write(sep + str(b.bid_dec))
                    f.write(sep + str(b.distance))
                    f.write(sep + str(b.bid_flux_est_cgs))
                    f.write(sep + str(b.p_lae_oii_ratio))
                    f.write(sep + str(len(b.filters)))
                    for flt in b.filters:
                        f.write(sep + str(flt.instrument))
                        f.write(sep + str(flt.filter))
                        f.write(sep + str(flt.flux))
                        f.write(sep + str(flt.err))

                    f.write("\n")

            f.close()
            msg = "File written: " + filename
            log.info(msg)
            print(msg)