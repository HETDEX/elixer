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


check_nei = False
check_mini = False


i = input("Check for nei.png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    check_nei = True

i = input("Check for mini.png (y/n)?")
if len(i) > 0 and i.upper() == "Y":
    check_mini = True

h5 = tables.open_file("elixer_merged_cat.h5","r")

dtb = h5.root.Detections
apt = h5.root.Aperture

alldets = dtb.read(field="detectid")
missing = []

for d in alldets:
    rows = apt.read_where("detectid==d",field="detectid")
    if rows.size==0:
        missing.append(d)

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

#find pdfs without pngs
print("Checking for missing .png files ...")
for d in alldets:
    if d in missing:
        continue #skip it, already marked for re-creation

    mini_okay = True #was found or don't care
    nei_okay = True #was found or don't care
    png_okay = True
    pdf_file = None
    files = glob.glob("dispatch_*/*/"+str(d)+"*")

    justfn = []
    for f in files:
        justfn.append(os.path.basename(f))
        if f[-3:] == 'pdf':
            pdf_file = f

    if check_mini and (str(d)+"mini.png" not in justfn):
        mini_okay = False

    if check_nei and (str(d)+"nei.png" not in justfn):
        nei_okay = False

    if str(d)+".png" not in justfn:
        png_okay = False

    if not (mini_okay and nei_okay):
        #remove the report for recovery
        if len(files) > 0:
            print("Removing " + str(d) + " ...")
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass
    elif not png_okay:
        #try to build png from os call
        print("OS call to pdftoppm for " + str(d) + "...")
        try:
            os.system("pdftoppm %s %s -png -singlefile" % (pdf_file, pdf_file.rstrip(".pdf")))
        except Exception as e:
            print(e)


