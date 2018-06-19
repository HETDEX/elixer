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
        self.bid_filter = ""
        self.bid_flux_est_cgs = 0.0
        self.bid_mag = 0.0
        self.bid_mag_err_bright = 0.0
        self.bid_mag_err_faint = 0.0
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
        self.addtl_line_names = "None"
        self.addtl_line_obs = "None"
        self.addtl_line_rest = "None"
        self.all_line_obs = "None"
        self.best_z = -1
        self.best_confidence = -1

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
        self.update_classification(emis)

    def add_bid_target(self,bid):
        if bid is not None:
            self.bid_targets.append(bid)

    def update_classification(self,emis): #emis is a DetObj
        if emis.spec_obj is None:
            self.addtl_line_names = "None"
            self.addtl_line_obs = "None"
            self.addtl_line_rest = "None"
            self.all_line_obs = "None"
            self.best_z = -1
            self.best_confidence = -1
            return

        #at the moment, this is all that is in place
        #todo: replace with a call to the DetObj that aggregates all the info and establishes a best_z
        good, confidence = emis.multi_line_solution_score()
        if good:
            self.addtl_line_names = ""
            self.addtl_line_obs = ""
            self.addtl_line_rest = ""
            self.all_line_obs = ""

            # strong solution
            sol = emis.spec_obj.solutions[0]

            self.best_z = sol.z
            self.best_confidence = confidence

            for el in sol.lines:
                self.addtl_line_names += "%s," % el.name.rstrip(" ")
                self.addtl_line_obs += "%0.2f," % el.w_obs
                self.addtl_line_rest += "%0.2f," % el.w_rest

            self.addtl_line_names = self.addtl_line_names.rstrip(",")
            self.addtl_line_obs = self.addtl_line_obs.rstrip(",")
            self.addtl_line_rest = self.addtl_line_rest.rstrip(",")

            for el in emis.spec_obj.all_found_lines:
                self.all_line_obs += "%0.2f," % el.fit_x0

            self.all_line_obs = self.all_line_obs.rstrip(",")
        else:
            self.addtl_line_names = "None"
            self.addtl_line_obs = "None"
            self.addtl_line_rest = "None"
            self.all_line_obs = "None"



class MatchSet:
    headers = [
        "input (entry) ID",
        "detect ID",
        "emission line RA (decimal degrees)",
        "emission line Dec (decimal degrees)",
        "number of possible catalog matches",
        "emission line wavelength (AA)",
        "best z (highest confidence redshift based on all information) [-1 if there is none]",
	    "best confidence (measure of the confidence of the best z) [-1 if there is none or is not calculated, 0.0 to 1.0 otherwise]",
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
        "comma separated list of observed wavelengths of ALL potential emission lines (whether or not they are used in the multi-line classification) [\"None\" if empty]",
	    "comma separated list of the additional line names used in the multi-line match [\"None\" if empty] ***Does NOT include the main (central) line***",
	    "comma separated list of the additional rest-frame wavelengths corresponding to emission line names [\"None\" if empty] ***Does NOT include the main (central) line***",
	    "comma separated list of the additional observed wavelengths corresponding to emission line names [\"None\" if empty] ***Does NOT include the main (central) line***",
        "catalog name",
        "catalog object RA (decimal degrees) [for the exact emission position, not matched to a catalog object, will = 361.0]",
        "catalog object Dec (decimal degrees)[for the exact emission position, not matched to a catalog object, will = 181.0]",
        "catalog object separation (arcsecs) [for the exact emission position, not matched to a catalog object, will = 0.0]",
        "catalog object continuum flux est at emission line wavelength (cgs) [for the exact emission position, from aperture on catalog image]",
        "catalog object filter used for magnitudes,"
        "catalog object magnitude,"
        "catalog object magnitude error (brighter),"
        "catalog object magnitude error (fainter)",
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

                    f.write(sep + str(m.best_z))
                    f.write(sep + str(m.best_confidence))

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

                    f.write(sep + str(m.all_line_obs))
                    f.write(sep + str(m.addtl_line_names))
                    f.write(sep + str(m.addtl_line_rest))
                    f.write(sep + str(m.all_line_obs))

                    f.write(sep + str(b.catalog_name))
                    f.write(sep + str(b.bid_ra))
                    f.write(sep + str(b.bid_dec))
                    f.write(sep + str(b.distance))
                    f.write(sep + str(b.bid_flux_est_cgs))

                    f.write(sep + str(b.bid_filter))
                    f.write(sep + str(b.bid_mag))
                    f.write(sep + str(b.bid_mag_err_bright))
                    f.write(sep + str(b.bid_mag_err_faint))

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