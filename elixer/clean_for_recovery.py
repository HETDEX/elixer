"""
Looks explicitly at elixer_merged_cat.h5 in the current directory and then removes down-stream
matched detection reporst so elixer in recovery mode will recreate
all detectids for which there appears to be no imaging.

Then also rebuilds .png for any .pdf without a .png
if checking for nei and/or mini and either is not found, remove the detection for full recreation
else just make a system call to make the png
"""



import tables
import glob
import os
import numpy as np
import sys

MINIMUM_PDF_FILESIZE = 100000 #100k bytes
MINIMUM_PNG_FILESIZE = 430000 #43k bytes

alldets = None
args = list(map(str.lower,sys.argv))
if "--dets" in args: #overide default if specified on command line
    try:
        i = args.index("--dets")
        if i != -1:
            dets_file = str(sys.argv[i + 1])
            alldets = np.loadtxt(dets_file,dtype=int)
    except:
        pass


check_nei = False
check_mini = False
remove_no_imaging = False
remove_no_png = False
remove_pdf_too_small = False
remove_files = False #set to false if only want to list the files that would be removed

if os.path.exists("elixer_merged_cat.h5"):
    print("elixer_merged_cat.h5 exists ... will compare with PDFs")


i = input("Remove Files (Y)  or List only (N)?")
if len(i) > 0 and i.upper() == "Y":
    remove_files = True

i = input("Remove if PDF too small (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    remove_pdf_too_small = True

i = input("Remove if no report png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    remove_no_png = True

i = input("Remove if PNG too small (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    remove_png_too_small = True
    remove_no_png = True #force to check if removing if too small

i = input("Remove if no imaging (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    i = input("**** Are you sure? This version of imaging check is not reliable (yes/no) [full word response]?")
    if len(i) > 0 and i.upper() == "YES":
        remove_no_imaging = True

i = input("Check for nei.png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    check_nei = True

i = input("Check for mini.png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    check_mini = True


globdets = glob.glob("dispatch_*/*/*.pdf")
allpdf_dets = []
for g in globdets:
    allpdf_dets.append(np.int64(g.rstrip(".pdf").split("/")[-1]))
allpdf_dets = np.array(allpdf_dets)

all_h5_dets = None
if remove_no_imaging:
    try:
        h5 = tables.open_file("elixer_merged_cat.h5","r")
        dtb = h5.root.Detections
        apt = h5.root.Aperture

        alldets = dtb.read(field="detectid")
        all_h5_dets = dtb.read(field="detectid")
    except Exception as e:
        print(e)
        print("elixer_merged_cat.h5 is needed. You must run elixer --merge first")
        exit(-1)
elif os.path.exists("elixer_merged_cat.h5"):
    try:
        h5 = tables.open_file("elixer_merged_cat.h5", "r")
        dtb = h5.root.Detections
        if (alldets is None) or len(alldets) == 0:
            alldets = dtb.read(field="detectid")
        all_h5_dets = dtb.read(field="detectid")
    except Exception as e:
        print(e)
        print("elixer_merged_cat.h5 error reading. You must run elixer --merge first or remove elixer_mergded_cat.h5")
        exit(-1)
else:
    alldets = allpdf_dets
#is there another way to get the alldets?

if np.array_equal(alldets, allpdf_dets):
    alldets = np.unique(np.concatenate((alldets,allpdf_dets)))

if all_h5_dets is not None:
    missing_h5_entries = np.setdiff1d(alldets,all_h5_dets)
else:
    missing_h5_entries = []


missing = []

ct_no_imaging = 0
ct_no_png = 0
ct_no_nei = 0
ct_no_mini = 0
ct_no_pdf = 0

would_be_removed = [] #list of detectIDs that would be removed (or are to be removed)

if remove_no_imaging:
    for d in alldets:
        rows = apt.read_where("detectid==d",field="detectid")
        #hmm ... that is not 100% accurate ... there are reasons this might be zero even with imaging
        if rows.size==0:
            missing.append(d)

    ct_no_imaging = len(missing)
    h5.close()

    would_be_removed += missing
    if remove_files:
        for d in missing:
            files = glob.glob("dispatch_*/*/"+str(d)+"*")
            if len(files) > 0:
                print("Removing " + str(d) + "...")
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        pass
# else:
#     print(f"{len(missing)} reports without imaging ... ")
#     for d in missing:
#         files = glob.glob("dispatch_*/*/" + str(d) + ".pdf")
#         if len(files) == 1:
#             print(d,files[0])
#         else:
#             print(d)
#     print(f"{len(missing)} reports without imaging removed")



#find pdfs without pngs
print("Checking for missing .png files ...")

all_nei = glob.glob("dispatch_*/*/*_nei.png")
all_mini = glob.glob("dispatch_*/*/*_mini.png")
all_rpt = glob.glob("dispatch_*/*/*[0-9].png")
all_pdf = glob.glob("dispatch_*/*/*.pdf")

names_nei = [os.path.basename(x) for x in all_nei]
names_mini = [os.path.basename(x) for x in all_mini]
names_rpt = [os.path.basename(x) for x in all_rpt]
names_pdf = [os.path.basename(x) for x in all_pdf]

regen_png = []

for d in alldets:
    if d in missing:
        continue #skip it, already marked for re-creation

    mini_okay = True #was found or don't care
    nei_okay = True #was found or don't care
    png_okay = True
    h5_okay = True #was found in h5 file
    pdf_okay = True
    pdf_file = None

    mini_idx = -1
    nei_idx = -1
    rpt_idx = -1
    pdf_idx = -1

    mini_path = None
    nei_path = None
    rpt_path = None
    pdf_path = None

    try:
        pdf_idx = names_pdf.index(str(d) + ".pdf")
        pdf_path = all_pdf[pdf_idx]
    except:
        pdf_idx = -1
        ct_no_pdf += 1
        #the pdf is missing, so no need to go further for this one
        continue


    #checking for no h5 entry
    try:
        if d in missing_h5_entries:
            h5_okay = False
    except:
        pass

    #check the PDF filesize ... if too small, it was generated but missing data
    #is that ever okay?
    pdf_sz = None
    if remove_pdf_too_small:
        try:
            pdf_sz = os.path.getsize(pdf_path)
            if pdf_sz < MINIMUM_PDF_FILESIZE:
                #this is a problem ... the main reports should be 500k-1000k or so
                pdf_okay = False
        except:
            pdf_idx = -1
            ct_no_pdf += 1
            #the pdf is already missing, so no need to go further for this one
            continue

    try:
        mini_idx = names_mini.index(str(d)+"_mini.png")
        mini_path = all_mini[mini_idx]
    except:
        ct_no_mini += 1
        mini_idx = -1
        if check_mini:
            mini_okay = False

    try:
        nei_idx = names_nei.index(str(d)+"_nei.png")
        nei_path = all_nei[nei_idx]
    except:
        nei_idx = -1
        ct_no_nei += 1
        if check_nei:
            nei_okay = False

    try:
        rpt_idx = names_rpt.index(str(d) + ".png")
        rpt_path = all_rpt[rpt_idx]

        if remove_png_too_small:
            try:
                png_sz = os.path.getsize(rpt_path)
                if png_sz < MINIMUM_PNG_FILESIZE and pdf_sz and pdf_sz > MINIMUM_PNG_FILESIZE * 0.8:
                    # this is a problem ... the main reports should be 43k+ or so
                    rpt_idx = -1
                    ct_no_png += 1
                    pdf_okay = False  # technically, the PDF is fine, it is the PNG that has a problem
                    # todo:
                    # try:
                    #
                    #
                    # except: #try to regenerate the PNG?
                    #     pdf_okay = False  # technically, the PDF is fine, it is the PNG that has a problem
            except Exception as e:
                print(e)
    except:
        rpt_idx = -1
        ct_no_png += 1
        png_okay = False


    if not (mini_okay and nei_okay and pdf_okay and h5_okay):
        #remove the report for recovery
        if remove_files:
            would_be_removed.append(d)
            print("Removing " + str(d) + " ...")
            try:
                if pdf_path:
                    os.remove(pdf_path) # this is the only one that really matters (the others will be overwritten)
                else:
                    print(f"Warning! No pdf path for {d}: {pdf_path}")

                if rpt_path:
                    os.remove(rpt_path)
                if nei_path:
                    os.remove(nei_path)
                if mini_path:
                    os.remove(mini_path)

            except:
               pass

    elif not png_okay and (pdf_idx > -1):
        if remove_no_png:
            would_be_removed.append(d)
            if remove_files:
                print("Removing " + str(d) + " ...")
                try:
                    os.remove(pdf_path)
                except:
                    pass
        else:
            #try to build png from os call
            print("OS call to pdftoppm for " + str(d) + "...")
            try:
                pdf_file = all_pdf[pdf_idx]
                os.system("pdftoppm %s %s -png -singlefile" % (pdf_file, pdf_file.rstrip(".pdf")))
            except Exception as e:
                print(e)
    elif (pdf_idx == -1) and png_okay:
        #the pdf was not found, but there is a png? weird, should not happen:
        #there is a png BUT it is of the wrong PDF?
        would_be_removed.append(d)
        print(f"PDF/PNG mismatch: {d}")
        try:
            os.remove(rpt_path)
        except:
            pass

        #todo: should we add these to a list to check again at the end (after a pause to complete?)

    #regardles, we can now remove this detection from the list so next searches are faster
    try:
        if pdf_idx > -1:
            del all_pdf[pdf_idx]
            del names_pdf[pdf_idx]

        if mini_idx > -1:
            del all_mini[mini_idx]
            del names_mini[mini_idx]

        if nei_idx > -1:
            del all_nei[nei_idx]
            del names_nei[nei_idx]

        if rpt_idx > -1:
            del all_rpt[rpt_idx]
            del names_rpt[rpt_idx]
    except Exception as e:
        print(e)

would_be_removed = np.unique(would_be_removed)
print(f"Missing h5 entry: {len(missing_h5_entries)}")
print(f"Missing PDF: {ct_no_pdf}")
print(f"Missing imaging: {ct_no_imaging}")
print(f"Missing report png: {ct_no_png}")
print(f"Missing nei png: {ct_no_nei}")
print(f"Missing mini png: {ct_no_mini}")
print(f"Total # DetectIDs that should be re-run: {len(would_be_removed)}")

try:
    np.savetxt("elixer_recovery_list.dets",would_be_removed,fmt="%d")
    print("Wrote: elixer_recovery_list.dets")
except:
    pass


