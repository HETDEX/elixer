from __future__ import print_function
import catalogs
import argparse
from astropy.coordinates import Angle
from matplotlib.backends.backend_pdf import PdfPages

#todo: parse line
#if ra, dec have ":" and "h" or "d" (hours or degrees for ra (hours implied?), degrees for dec)
#    use astropy to convert to decimal degrees

#add parameter -f (force) to not prompt for confirmations


#print confirmation line ("looking for targets at RA=(decimal degrees) Dec=  +/- (arcsecs). Proceed?
#if yes, find number of targets
#print xxx targets found, proceed with plots?



def parse_commandline():
    desc = "Search multiple catalogs for possible object(s) at specified coordinates."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--force', help='Do not prompt for confirmation.', required=False,
                        action='store_true', default=False)
    parser.add_argument('-r', '--ra', help='Target RA as decimal degrees or h:m:s.as (end with \'h\')'
                                           'or d:m:s.as (end with \'d\') '
                                           'Examples: --ra 214.963542  or --ra 14:19:51.250h or --ra 214:57:48.7512d'
                                            , required=True)
    parser.add_argument('-d', '--dec', help='Target Dec (as decimal degrees or d:m:s.as (end with \'d\') '
                                            'Examples: --dec 52.921167    or  --dec 52:55:16.20d', required=True)
    parser.add_argument('-e', '--error', help="Error (+/-) in RA and Dec in arcsecs.", required=True, type=float)
    parser.add_argument('-n','--name', help="PDF report filename",required=True)


    args = parser.parse_args()

    if ":" in args.ra:
        try:
            args.ra = float(Angle(args.ra).degree)
        except:
            print("Error. Cannot determine format of RA")
            exit(-1)
    else:
        args.ra = float(args.ra)

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
        i = raw_input("Looking for targets +/- %f\" from RA=%f DEC=%f\nProceed (y/n ENTER=YES)?"
                      % (args.error, args.ra, args.dec))

        if len(i) > 0 and i.upper() !=  "Y":
            print ("Cancelled.")
            exit(0)
        else:
            print()

    return args



def main():

    args = parse_commandline()

    cats = catalogs.get_catalog_list()

    #convert error to decimal degrees for consistency
    error_in_deg = float(args.error)/3600.0

    num_hits = 0
    for c in cats:
        if c.position_in_cat(ra=args.ra,dec=args.dec,error=error_in_deg):
            hits,_,_ = c.build_list_of_bid_targets(ra=args.ra,dec=args.dec,error=error_in_deg)
            num_hits += hits
            if hits > 0:
                print ("%d hits in %s" %(hits,c.name))

    if num_hits == 0:
        print ("No possible matches found. Exiting")
        exit(0)

    if not args.force:
        i = raw_input("%d total possible matches found.\nProceed (y/n ENTER=YES)?" % num_hits)

        if len(i) > 0 and i.upper() !=  "Y":
            print ("Cancelled.")
            exit(0)
        else:
            print()
    else:
        print ("%d possible matches found. Building report..." % num_hits)


    #for test
    reports = []
    for c in cats:
        r = c.build_bid_target_reports(args.ra, args.dec,args.error)
        if r is not None:
            reports = reports + r

    print("Finalizing report ...")
    pdf = PdfPages(args.name)
    rows = len(reports)

    for r in range(rows):
        pdf.savefig(reports[r])
    pdf.close()
    print("File written: " + args.name)


    exit(0)

# end main


if __name__ == '__main__':
    main()