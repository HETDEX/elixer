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

i = input("Remove if no imaging (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    remove_no_imaging = True

i = input("Check for nei.png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    check_nei = True

i = input("Check for mini.png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    check_mini = True

if remove_no_imaging:
    h5 = tables.open_file("elixer_merged_cat.h5","r")

    dtb = h5.root.Detections
    apt = h5.root.Aperture

    alldets = dtb.read(field="detectid")

missing = []

ct_no_imaging = 0
ct_no_png = 0
ct_no_nei = 0
ct_no_mini = 0

if remove_no_imaging:
    for d in alldets:
        rows = apt.read_where("detectid==d",field="detectid")
        if rows.size==0:
            missing.append(d)

    ct_no_imaging = len(missing)
    h5.close()

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

all_nei = glob.glob("dispatch_*/*/*nei.png")
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
        mini_idx = names_mini.index(str(d)+"_mini.png")
        mini_path = all_mini[mini_idx]
    except:
        ct_no_mini += 1
        mini_idx = -1
        if check_mini:
            mini_okay = False

    try:
        nei_idx = names_nei.index(str(d)+"nei.png")
        nei_path = all_nei[nei_idx]
    except:
        nei_idx = -1
        ct_no_nei += 1
        if check_nei:
            nei_okay = False

    try:
        rpt_idx = names_rpt.index(str(d) + ".png")
        rpt_path = all_rpt[rpt_idx]
    except:
        rpt_idx = -1
        ct_no_png += 1
        png_okay = False

    try:
        pdf_idx = names_pdf.index(str(d) + ".pdf")
        pdf_path = all_pdf[pdf_idx]
    except:
        pdf_idx = -1



    if not (mini_okay and nei_okay):
        #remove the report for recovery
        print("Removing " + str(d) + " ...")
        try:
            if pdf_path:
                os.remove(pdf_path) # this is the only one that really matters (the others will be overwritten)
            else:
                print(f"Warning! No pdf path for {d}: {pdf_path}")

            if rpt_path:
                os.remove((rpt_path))
            if nei_path:
                os.remove(nei_path)
            if mini_path:
                os.remove(mini_path)

        except:
           pass

    elif not png_okay and (pdf_idx > -1):
        #try to build png from os call
        print("OS call to pdftoppm for " + str(d) + "...")
        try:
            pdf_file = all_pdf[pdf_idx]
            os.system("pdftoppm %s %s -png -singlefile" % (pdf_file, pdf_file.rstrip(".pdf")))
        except Exception as e:
            print(e)

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


print(f"Missing imaging: {ct_no_imaging}")
print(f"Missing report png: {ct_no_png}")
print(f"Missing nei png: {ct_no_nei}")
print(f"Missing mini png: {ct_no_mini}")
