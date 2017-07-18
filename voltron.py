from __future__ import print_function
import catalogs
import argparse
import hetdex
import match_summary
import global_config as G
from astropy.coordinates import Angle
from matplotlib.backends.backend_pdf import PdfPages
from distutils.version import LooseVersion
import pyhetdex.tools.files.file_tools as ft
import sys
import glob
import os
import errno
import re
from PIL import Image



#try:
#    import PyPDF2 as PyPDF
#except ImportError:
#    PyPDF = None

try:
    import pdfrw as PyPDF
except ImportError:
    pdfrw = None


VERSION = sys.version.split()[0]
#import random

G_PDF_FILE_NUM = 0

log = G.logging.getLogger('main_logger')
log.setLevel(G.logging.DEBUG)

def get_input(prompt):
    if LooseVersion(VERSION) >= LooseVersion('3.0'):
        i = input(prompt)
    else:
        i = raw_input(prompt)
    return i


def parse_astrometry(file):
    ra = None
    dec = None
    par = None
    try:
        with open(file, 'r') as f:
            f = ft.skip_comments(f)
            for l in f:
                if len(l) > 10: #some reasonable minimum
                    toks = l.split()
                    #todo: some sanity checking??
                    ra = float(toks[0])
                    dec = float(toks[1])
                    par = float(toks[2])
    except:
        log.error("Cannot read astrometry file: %s" % file, exc_info=True)

    return ra,dec,par


class pdf_file():
    def __init__(self,basename,id):
        self.filename = '%s' % basename
        self.id = id
        self.bid_count = 0 #rough number of bid targets included
        if self.id > 0: #i.e. otherwise, just building a single pdf file
            #make the directory
            if not os.path.isdir(self.filename):
                try:
                    os.makedirs(self.filename)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        print ("Fatal. Cannot create pdf output directory: %s" % self.filename)
                        log.critical("Fatal. Cannot create pdf output directory: %s" % self.filename,exc_info=True)
                        exit(-1)
            # have to leave files there ... this is called per internal iteration (not just once0
           # else: #already exists
           #     try: #empty the directory of any previous runs
           #         regex = re.compile(self.filename + '_.*')
           #         [os.remove(os.path.join(self.filename,f)) for f in os.listdir(self.filename) if re.match(regex,f)]
           #     except:
           #         log.error("Unable to clean output directory: " + self.filename,exc_info=True)

            filename = os.path.basename(self.filename) + "_" + str(id).zfill(3) + ".pdf"
            self.filename = os.path.join(self.filename,filename)
        else:
            pass #keep filename as is

        self.pages = None



def parse_commandline():
    desc = "Search multiple catalogs for possible object matches.\n\nNote: if (--ra), (--dec), (--par) supplied in " \
           "addition to (--dither),(--line), the supplied RA, Dec, and Parangle will be used instead of the " \
           "TELERA, TELEDEC, and PARANGLE from the science FITS files."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--force', help='Do not prompt for confirmation.', required=False,
                        action='store_true', default=False)
    parser.add_argument('-c', '--cure', help='Use Cure processed fits (instead of Panacea).', required=False,
                        action='store_true', default=False)

    parser.add_argument('--ra', help='Target RA as decimal degrees or h:m:s.as (end with \'h\')'
                                           'or d:m:s.as (end with \'d\') '
                                           'Examples: --ra 214.963542  or --ra 14:19:51.250h or --ra 214:57:48.7512d'
                                            , required=False)
    parser.add_argument('--dec', help='Target Dec (as decimal degrees or d:m:s.as (end with \'d\') '
                                            'Examples: --dec 52.921167    or  --dec 52:55:16.20d', required=False)
    #parser.add_argument('--rot',help="Rotation (as decimal degrees). NOT THE PARANGLE.",required=False,type=float)
    parser.add_argument('--par', help="The Parangle in decimal degrees.", required=False, type=float)
    parser.add_argument('--rot', help="The rotation in decimal degrees. (Superceeds use of --par)", required=False, type=float)

    parser.add_argument('--ast', help="Astrometry coordinates file. Use instead of --ra, --dec, --par", required=False)

    parser.add_argument('--obsdate', help="Observation Date. Must be YYYYMMDD.  "
                                          "Must provide for Panacea.", required=False)
    parser.add_argument('--obsid', help="Observation ID (integer). "
                                        "Must provide for Panacea.", required=False, type=int)
    parser.add_argument('--specid', help="SpecID aka CAM (integer) i.e. --specid 13 or --specid 13,14,19  "
                                         "If not specified, all are used. (may be restricted by --ifuid or --ifuslot)", required=False, type=int)
    parser.add_argument('--ifuid', help="IFU ID (integer) *** NOTICE. This is the cable ID.  "
                                        "If not specified, all are used (may be restricted by --specid or --ifuslot)", required=False, type=int)
    parser.add_argument('--ifuslot', help="IFU SLOT ID (integer)  "
                                          "If not specified, all are used (may be restricted by --specid or --ifusid)", required=False, type=int)

    parser.add_argument('-e', '--error', help="Error (+/-) in RA and Dec in arcsecs.", required=True, type=float)

    parser.add_argument('--fibers', help="Number of fibers to plot in 1D spectra cutout."
                                         "If present, also turns off weighted average.", required=False, type=int)

    parser.add_argument('-n','--name', help="Report filename or directory name (if HETDEX emission lines supplied)",required=True)
    parser.add_argument('--multi', help='*Mandatory. Switch remains only for compatibility. Cannot be turned off.*'
                                        'Produce one PDF file per emission line (in folder from --name).', required=False,
                        action='store_true', default=False)

    parser.add_argument('--dither', help="HETDEX Dither file", required=False)
    parser.add_argument('--path', help="Override path to science fits in dither file", required=False)
    parser.add_argument('--line', help="HETDEX detect line file", required=False)
    parser.add_argument('--ifu', help="HETDEX IFU (Cure) file", required=False)
    parser.add_argument('--dist', help="HETDEX Distortion (Cure) file base (i.e. do not include trailing _L.dist or _R.dist)",
                        required=False)
    parser.add_argument('--id', help="ID or list of IDs from detect line file for which to search", required=False)
    parser.add_argument('--sigma', help="Minimum sigma threshold (Cure) to meet in selecting detections", required=False,
                        type=float,default=0.0)
    parser.add_argument('--chi2', help="Maximum chi2 threshold (Cure) to meet in selecting detections", required=False,
                        type=float,default=1e9)
    parser.add_argument('--sn', help="Minimum fiber signal/noise threshold (Panacea) to plot in spectra cutouts",
                        required=False, type=float, default=0.0)

    args = parser.parse_args()

    #reminder to self ... this is pointless with SLURM given the bash wraper (which does not know about the
    #speccific dir name and just builds voltron.run

    #if args.name is not None:
    #    G.logging.basicConfig(filename="voltron."+args.name+".log", level=G.LOG_LEVEL, filemode='w')
    #else:
    #    print("Missing mandatory paramater --name.")
    #    exit(-1)

    #regardless of setting, --multi must now always be true
    args.multi = True

    log.info(args)

    if args.ra is not None:
        if ":" in args.ra:
            try:
                args.ra = float(Angle(args.ra).degree)
            except:
                print("Error. Cannot determine format of RA")
                log.critical("Main exit. Invalid command line parameters.")
                exit(-1)
        else:
            args.ra = float(args.ra)

    if args.dec is not None:
        if ":" in args.dec:
            try:
                args.dec = float(Angle(args.dec).degree)
            except:
                print("Error. Cannot determine format of DEC")
                log.critical("Main exit. Invalid command line parameters.")
                exit(-1)
        else:
            args.dec = float(args.dec)

    if args.ast is not None:
        r,d,p = parse_astrometry(args.ast)
        if r is not None:
            args.ra = r
            args.dec = d
            args.par = p


    if args.error < 0:
        print("Invalid error. Must be non-negative.")
        log.critical("Main exit. Invalid command line parameters.")
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
            log.critical("Main exit. User cancel.")
            exit(0)
        else:
            print()

    if valid_parameters(args):
        return args
    else:
        print("Invalid command line parameters. Cancelled.")
        log.critical("Main exit. Invalid command line parameters.")
        exit(-1)

def valid_parameters(args):

    #must have ra and dec -OR- dither and (ID or (chi2 and sigma))
    if (args.ra is None) or (args.dec is None):
        if (args.line is None):
            print("Invalid parameters. Must specify either (--ra and --dec) or detect parameters (--dither, --line, --id, "
                  "--sigma, --chi2)")
            return False
        elif args.cure:
            if (args.ifu is None):
                print("Warning. IFU file not provided. Report might not contain spectra cutouts. Will search for IFU file "
                      "in the config directory.")
            if (args.dist is None):
                print("Warning. Distortion file (base) not provided. Report might not contain spectra cutouts. "
                      "Will search for Distortion files in the config directory.")

            return True
        else:
            return True
    else: #chi2 and sigma have default values, so don't use here

        return True

        #if (args.dither is None) and (args.line is None) and (args.id is None):# and (args.chi2 is None) and (args.sigma is None)
        #    return True
        #else:
        #    print("Confusing parameters. Pass either (--ra and --dec) or detect parameters (--dither, --line, --id, "
        #          "--sigma, --chi2)")
        #    return False


def build_hd(args):
    #if (args.dither is not None):
    if (args.line is not None): #or (args.id is not None):
            return True

    return False


def build_hetdex_section(pdfname, hetdex, detect_id = 0,pages=None):
    #detection ids are unique (for the single detect_line.dat file we are using)
    if pages is None:
        pages = []
    pages = hetdex.build_hetdex_data_page(pages,detect_id)

    if PyPDF is not None:
        build_report_part(pdfname,pages)
        pages = None

    return pages


def build_pages (pdfname,match,ra,dec,error,cats,pages,num_hits=0,idstring="",base_count = 0,target_w=0,fiber_locs=None,
                 target_flux=None):
    #if a report object is passed in, immediately append to it, otherwise, add to the pages list and return that
    section_title = idstring
    count = 0
    for c in cats:
        r = c.build_bid_target_reports(match,ra, dec, error,num_hits=num_hits,section_title=section_title,
                                       base_count=base_count,target_w=target_w,fiber_locs=fiber_locs,
                                       target_flux=target_flux)
        count = 0
        if r is not None:
            if PyPDF is not None:
                build_report_part(pdfname,r)
            else:
                pages = pages + r
            count = max(0,len(r)-1) #1st page is the target page

    return pages, count


def open_report(report_name):
    return PdfPages(report_name)

def close_report(report):
    if report is not None:
        report.close()

def add_to_report(pages,report):
    if (pages is None) or (len(pages) == 0):
        return

    print("Adding to report ...")
    rows = len(pages)

    for r in range(rows):
        report.savefig(pages[r])
    return


def build_report(pages,report_name):
    if (pages is None) or (len(pages) == 0):
        return

    print("Finalizing report ...")

    pdf = PdfPages(report_name)
    rows = len(pages)

    for r in range(rows):
        pdf.savefig(pages[r])

    pdf.close()
    print("File written: " + report_name)
    return



def build_report_part(report_name,pages):
    if (pages is None) or (len(pages) == 0):
        return

    global G_PDF_FILE_NUM

    G_PDF_FILE_NUM += 1
    part_name = report_name + ".part%s" % (str(G_PDF_FILE_NUM).zfill(4))

    pdf = PdfPages(part_name)
    rows = len(pages)

    for r in range(rows):
        pdf.savefig(pages[r])

    pdf.close()

    return


def join_report_parts(report_name, bid_count=0):

    if PyPDF is None:
        return
    print("Finalizing report ...")

    metadata = PyPDF.IndirectPdfDict(
        Title='Voltron Emission Line Report',
        Author="HETDEX, Univ. of Texas",
        Keywords='Voltron Version = ' + G.__version__)

    if G.SINGLE_PAGE_PER_DETECT:
        if (bid_count <= G.MAX_COMBINE_BID_TARGETS):
            log.info("Creating single page report for %s. Bid count = %d" % (report_name, bid_count))
            list_pages = []

            for i in range(G_PDF_FILE_NUM):
                #use this rather than glob since glob sometimes messes up the ordering
                #and this needs to be in the correct order
                #(though this is a bit ineffecient since we iterate over all the parts every time)
                part_name = report_name+".part%s" % str(i+1).zfill(4)
                if os.path.isfile(part_name):
                    pages = PyPDF.PdfReader(part_name).pages
                    for p in pages:
                        #merge_page += p
                        list_pages.append(p)

            merge_page = PyPDF.PageMerge() + list_pages

            scale = 1.0 #full scale
            y_offset = 0
            #need to count backward ... position 0,0 is the bottom of the page
            #each additional "page" is advanced in y by the y height of the previous "page"
            for i in range(len(merge_page) - 1, -1, -1):
                page = merge_page[i]
                page.scale(scale)
                page.x = 0
                page.y = y_offset
                y_offset = scale* page.box[3] #box is [x0,y0,x_top, y_top]

 #           if os.path.isdir(report_name):  # often just the base name is given
            if not report_name.endswith(".pdf"):
                report_name += ".pdf"
            writer = PyPDF.PdfWriter(report_name)

            try:
                writer.addPage(merge_page.render())
                writer.trailer.Info = metadata
                writer.write()
            except:
                log.error("Error writing out pdf: " + report_name, exc_info = True)
        else: #want a single page, but there are just too many sub-pages
            list_pages = []
            log.info("Single page report not possible for %s. Bid count = %d" %(report_name,bid_count))
            part_num = 0
            for i in range(G_PDF_FILE_NUM):
                # use this rather than glob since glob sometimes messes up the ordering
                # and this needs to be in the correct order
                # (though this is a bit ineffecient since we iterate over all the parts every time)
                part_name = report_name + ".part%s" % str(i + 1).zfill(4)
                if os.path.isfile(part_name):
                    pages = PyPDF.PdfReader(part_name).pages
                    part_num = i + 1
                    for p in pages:
                        # merge_page += p
                        list_pages.append(p)
                    break # just merge the first part

            merge_page = PyPDF.PageMerge() + list_pages

            scale = 1.0  # full scale
            y_offset = 0
            # need to count backward ... position 0,0 is the bottom of the page
            # each additional "page" is advanced in y by the y height of the previous "page"
            for i in range(len(merge_page) - 1, -1, -1):
                page = merge_page[i]
                page.scale(scale)
                page.x = 0
                page.y = y_offset
                y_offset = scale * page.box[3]  # box is [x0,y0,x_top, y_top]

#            if os.path.isdir(report_name):  # often just the base name is given
            if not report_name.endswith(".pdf"):
                report_name += ".pdf"
            writer = PyPDF.PdfWriter(report_name)
            writer.addPage(merge_page.render())

            #now, add (but don't merge) the other parts
            for i in range(part_num,G_PDF_FILE_NUM):
                # use this rather than glob since glob sometimes messes up the ordering
                # and this needs to be in the correct order
                # (though this is a bit ineffecient since we iterate over all the parts every time)
                part_name = report_name + ".part%s" % str(i + 1).zfill(4)
                if os.path.isfile(part_name):
                    writer.addpages(PyPDF.PdfReader(part_name).pages)

            writer.trailer.Info = metadata
            writer.write()
    else:
        log.info("Creating multi-page report for %s. Bid count = %d" % (report_name, bid_count))
        writer = PyPDF.PdfWriter()
        #for --multi the file part numbers are unique. Only the first file starts with 001. The second starts with
        #where the first left off
        for i in range(G_PDF_FILE_NUM):
            #use this rather than glob since glob sometimes messes up the ordering
            #and this needs to be in the correct order
            #(though this is a bit ineffecient since we iterate over all the parts every time)
            part_name = report_name+".part%s" % str(i+1).zfill(4)
            if os.path.isfile(part_name):
                writer.addpages(PyPDF.PdfReader(part_name).pages)

        writer.trailer.Info = metadata
        writer.write(report_name)

    print("File written: " + report_name)


def delete_report_parts(report_name):
    for f in glob.glob(report_name+".part*"):
        os.remove(f)

def confirm(hits,force):

    if not force:
        i = get_input("\n%d total possible matches found.\nProceed (y/n ENTER=YES)?" % hits)

        if len(i) > 0 and i.upper() != "Y":
            print("Cancelled.")
            return False
        else:
            print()
    else:
        print("%d possible matches found. Building report..." % hits)

    return True


def ifulist_from_detect_file(args):
    ifu_list = []
    if args.line is not None:
        try:
            with open(args.line, 'r') as f:
                f = ft.skip_comments(f)
                for l in f:
                    toks = l.split()
                    if len(toks) > 17: #this may be an aggregate line file (last token = ifuxxx)
                        if "ifu" in toks[17]:
                            ifu = str(toks[17][-3:])  # ifu093 -> 093
                            if ifu in ifu_list:
                                continue
                            else:
                                ifu_list.append(ifu)
        except:
            log.info("Exception checking detection file for ifu list", exc_info=True)
    return ifu_list

def main():
    global  G_PDF_FILE_NUM

    G.gc.enable()
    #G.gc.set_debug(G.gc.DEBUG_LEAK)
    args = parse_commandline()
    cats = catalogs.get_catalog_list()
    pages = []

    #if a specific observation date was specified, build one hd object per ifu, otherwise they all go to one
    if (args.obsdate is not None):
        ifu_list = ifulist_from_detect_file(args)
    else:
        ifu_list = []

    hd_list = [] #one entry for each amp (or side) and dither
    file_list = []
    match_list = match_summary.MatchSet()

    # first, if hetdex info provided, build the hetdex part of the report
    # hetedex part
    if build_hd(args):
        if (len(ifu_list) > 0) and ((args.ifuslot is None) and (args.ifuid is None) and (args.specid is None)):

            #sort so easier to find
            ifu_list.sort()

            for ifu in ifu_list:
                args.ifuslot = int(ifu)
                hd = hetdex.HETDEX(args)
                if hd is not None:
                    hd_list.append(hd)
        else:
            hd = hetdex.HETDEX(args)
            if hd is not None:
                if hd.status == 0:
                    hd_list.append(hd)

    if len(hd_list) > 0:
        for hd in hd_list:
            if hd.status != 0:
                #fatal
                print("Fatal error. Cannot build HETDEX working object.")
                log.critical("Fatal error. Cannot build HETDEX working object.")
                if len(hd_list) > 1:
                    continue
                else:
                    log.critical("Main exit. Fatal error.")
                    exit (-1)

            #iterate over all emission line detections
            if len(hd.emis_list) > 0:
                print()
                #first see if there are any possible matches anywhere
                matched_cats = []
                num_hits = 0
                for c in cats:
                    for e in hd.emis_list:
                        if (e.wra is not None) and (e.wdec is not None):  # weighted RA and Dec
                            ra = e.wra
                            dec = e.wdec
                        else:
                            ra = e.ra
                            dec = e.dec
                        if c.position_in_cat(ra=ra, dec=dec, error=args.error):
                            hits, _, _ = c.build_list_of_bid_targets(ra=ra, dec=dec, error=args.error)
                            num_hits += hits

                            if c not in matched_cats:
                                matched_cats.append(c)
                            print("%d hits in %s for Detect ID #%d" % (hits, c.name, e.id))
                        else: #todo: don't bother printing the negative case
                            print("Coordinates not in range of %s for Detect ID #%d" % (c.name,e.id))

                if not confirm(num_hits,args.force):
                    log.critical("Main exit. User cancel.")
                    exit(0)

                #now build the report for each emission detection
                for e in hd.emis_list:
                    pdf = pdf_file(args.name, e.id)

                    id = "Detect ID #" + str(e.id)
                    if (e.wra is not None) and (e.wdec is not None): #weighted RA and Dec
                        ra = e.wra
                        dec = e.wdec
                    else:
                        ra = e.ra
                        dec = e.dec
                    pdf.pages = build_hetdex_section(pdf.filename,hd,e.id,pdf.pages) #this is the fiber, spectra cutouts for this detect

                    match = match_summary.Match(e)

                    pdf.pages,pdf.bid_count = build_pages(pdf.filename, match, ra, dec, args.error, matched_cats, pdf.pages,
                                                  num_hits=num_hits, idstring=id,base_count=0,target_w=e.w,
                                                  fiber_locs=e.fiber_locs,target_flux=e.estflux)

                    #only add if there is at least one imaging catalog counterpart
                    if len(match.bid_targets) > 0:
                        match_list.add(match)

                    file_list.append(pdf)

           # else: #for multi calls (which are common now) this is of no use
           #     print("\nNo emission detections meet minimum criteria for specified IFU. Exiting.\n"
           #     log.warning("No emission detections meet minimum criteria for specified IFU. Exiting.")
    elif (args.ra and args.dec):
        num_hits = 0
        matched_cats = []
        for c in cats:
            if c.position_in_cat(ra=args.ra,dec=args.dec,error=args.error):
                if c not in matched_cats:
                    matched_cats.append(c)
                hits,_,_ = c.build_list_of_bid_targets(ra=args.ra,dec=args.dec,error=args.error)
                num_hits += hits
                if hits > 0:
                    print ("%d hits in %s" %(hits,c.name))

        if not confirm(num_hits,args.force):
            log.critical("Main exit. User cancel.")
            exit(0)

        pages,_ = build_pages(args.name,None,args.ra, args.dec, args.error, matched_cats, pages, idstring="# 1 of 1")
    else:
        print("Invalid command line call. Insufficient information to execute.")
        exit(-1)

    if len(file_list) > 0:
        for f in file_list:
            build_report(f.pages,f.filename)
    else:
        build_report(pages,args.name)

    if PyPDF is not None:
        if len(file_list) > 0:
            for f in file_list:
                join_report_parts(f.filename,f.bid_count)
                delete_report_parts(f.filename)
        else:
            join_report_parts(args.name)
            delete_report_parts(args.name)

    if match_list.size > 0:
        filename = os.path.join(args.name,args.name+"_cat.txt")
        match_list.write_file(filename)

    log.critical("Main complete.")

    exit(0)

# end main


if __name__ == '__main__':
    main()