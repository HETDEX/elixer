from __future__ import print_function
import catalogs
import argparse
import hetdex
import global_config
from astropy.coordinates import Angle
from matplotlib.backends.backend_pdf import PdfPages
from distutils.version import LooseVersion
import sys

VERSION = sys.version.split()[0]


def get_input(prompt):
    if LooseVersion(VERSION) >= LooseVersion('3.0'):
        i = input(prompt)
    else:
        i = raw_input(prompt)
    return i



#todo: alternate use: provide dither file, detect_line.dat file and specify sigma and chi2 limits or a
#todo: list of detect IDs for which to run a catalog match. Would then not need an explicit RA/DEC, but
#todo: would still need an error

def parse_commandline():
    desc = "Search multiple catalogs for possible object(s) at specified coordinates."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--force', help='Do not prompt for confirmation.', required=False,
                        action='store_true', default=False)

    #todo: change to optional when dither is supported
    parser.add_argument('-r', '--ra', help='Target RA as decimal degrees or h:m:s.as (end with \'h\')'
                                           'or d:m:s.as (end with \'d\') '
                                           'Examples: --ra 214.963542  or --ra 14:19:51.250h or --ra 214:57:48.7512d'
                                            , required=False)
    # todo: change to optional when dither is supported
    parser.add_argument('-d', '--dec', help='Target Dec (as decimal degrees or d:m:s.as (end with \'d\') '
                                            'Examples: --dec 52.921167    or  --dec 52:55:16.20d', required=False)
    parser.add_argument('-e', '--error', help="Error (+/-) in RA and Dec in arcsecs.", required=True, type=float)
    parser.add_argument('-n','--name', help="PDF report filename",required=True)

    parser.add_argument('--dither', help="HETDEX Dither file", required=False)
    parser.add_argument('--path', help="Override path to science fits in dither file", required=False)
    parser.add_argument('--line', help="HETDEX (Cure) detect line file", required=False)
    parser.add_argument('--id', help="ID or list of IDs from detect line file for which to search", required=False)
    parser.add_argument('--sigma', help="Minimum sigma threshold to meet in selecting detections", required=False,
                        type=float,default=0.0)
    parser.add_argument('--chi2', help="Maximum chi2 threshold to meet in selecting detections", required=False,
                        type=float,default=1e9)

    args = parser.parse_args()

    if args.ra is not None:
        if ":" in args.ra:
            try:
                args.ra = float(Angle(args.ra).degree)
            except:
                print("Error. Cannot determine format of RA")
                exit(-1)
        else:
            args.ra = float(args.ra)

    if args.dec is not None:
        if ":" in args.dec:
            try:
                args.dec = float(Angle(args.dec).degree)
            except:
                print("Error. Cannot determine format of DEC")
                exit(-1)
        else:
            args.dec = float(args.dec)

    if args.error < 0:
        print("Invalid error. Must be non-negative.")
        exit(0)

    if not args.force:
        prompt = ""
        if (args.ra is not None) and (args.dec is not None):
            prompt = "Looking for targets +/- %f\" from RA=%f DEC=%f\nProceed (y/n ENTER=YES)?" \
                          % (args.error, args.ra, args.dec)
        else:
            prompt = "Looking for targets +/- %f\" from detections listed in file.\nProceed (y/n ENTER=YES)?" \
                        % args.error

        i = get_input(prompt)

        if len(i) > 0 and i.upper() !=  "Y":
            print ("Cancelled.")
            exit(0)
        else:
            print()

    if valid_parameters(args):
        return args
    else:
        print("Invalid command line parameters. Cancelled.")
        exit(-1)

def valid_parameters(args):

    #must have ra and dec -OR- dither and (ID or (chi2 and sigma))
    if (args.ra is None) or (args.dec is None):
        if (args.dither is None) or (args.line is None):
            print("Invalid parameters. Must specify either (--ra and --dec) or detect parameters (--dither, --line, --id, "
                  "--sigma, --chi2)")
            return False
        else:
            return True
    else: #chi2 and sigma have default values, so don't use here
        if (args.dither is None) and (args.line is None) and (args.id is None):# and (args.chi2 is None) and (args.sigma is None)
            return True
        else:
            print("Confusing parameters. Pass either (--ra and --dec) or detect parameters (--dither, --line, --id, "
                  "--sigma, --chi2)")
            return False

        #we have an  RA and Dec
    #error is already checked as required by argsparse




def build_hd(args):
    if (args.dither is not None):
        if (args.line is not None) or (args.id is not None):
            return True

    return False

def build_pages (ra,dec,error,cats,pages,idstring):
    # next, build reports for each catalog
    #todo: build a section header for THIS ra + dec
    #include RA, DEC, ENUMERATION (i.e. Report ID#1, 2, 3 ...)
    section_title = "Inspection ID: " + idstring
    for c in cats:
        r = c.build_bid_target_reports(ra, dec, error,section_title=section_title)
        if r is not None:
            pages = pages + r

    return pages


def build_report(pages,report_name):
    print("Finalizing report ...")
    pdf = PdfPages(report_name)
    rows = len(pages)

    for r in range(rows):
        pdf.savefig(pages[r])

    pdf.close()
    print("File written: " + report_name)
    return


def confirm(hits,force):
    if hits == 0:
        print("No possible matches found. Exiting")
        return False

    if not force:
        i = get_input("%d total possible matches found.\nProceed (y/n ENTER=YES)?" % hits)

        if len(i) > 0 and i.upper() != "Y":
            print("Cancelled.")
            return False
        else:
            print()
    else:
        print("%d possible matches found. Building report..." % hits)

    return True

def main():

    args = parse_commandline()

    cats = catalogs.get_catalog_list()

    #convert error to decimal degrees for consistency
    error_in_deg = float(args.error)/3600.0

    pages = []
    hd = None

    # first, if hetdex info provided, build the hetdex part of the report
    # hetedex part
    if build_hd(args):
        hd = hetdex.HETDEX(args)

    if hd.status != 0:
        #fatal
        print("Fatal error. Cannot build HETDEX working object.")
        exit (-1)

    # todo: iterate over all emission line detections
    if (hd is not None) and (len(hd.emis_list) > 0):
        #todo: first see if there are any possible matches anywhere
        num_hits = 0
        for e in hd.emis_list:
            ra = e.ra
            dec = e.dec
            print ("Checking RA=%f, Dec=%f" % (ra,dec))
            for c in cats:
                if c.position_in_cat(ra=ra, dec=dec, error=error_in_deg):
                    hits, _, _ = c.build_list_of_bid_targets(ra=ra, dec=dec, error=error_in_deg)
                    num_hits += hits
                    if hits > 0:
                        print("%d hits in %s for Detect ID %d" % (hits, c.name, e.id))

        if not confirm(num_hits,args.force):
            exit(0)

        section_id = 0
        total = len(hd.emis_list)
        for e in hd.emis_list:
            section_id += 1
            id = "# " + str(section_id) + " of " + str(total) + ": Detect ID# " + str(e.id)
            ra = e.ra
            dec = e.dec
            pages = build_pages(ra, dec, args.error, cats, pages, idstring=id)

    else:
        num_hits = 0
        for c in cats:
            if c.position_in_cat(ra=args.ra,dec=args.dec,error=error_in_deg):
                hits,_,_ = c.build_list_of_bid_targets(ra=args.ra,dec=args.dec,error=error_in_deg)
                num_hits += hits
                if hits > 0:
                    print ("%d hits in %s" %(hits,c.name))

        if not confirm(num_hits,args.force):
            exit(0)

        pages = build_pages(args.ra, args.dec, args.error, cats, pages, idstring="# 1 of 1")







    build_report(pages,args.name)

    exit(0)

# end main


if __name__ == '__main__':
    main()