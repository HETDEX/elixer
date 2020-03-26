import sys
import glob
import os
# import global_config as G
from pdf2image import convert_from_path

RESOLUTION=150 #DPI

# from PIL import Image as PIL_Image
#
# try:
#     import elixer.pdfrw as PyPDF
# except:
#     try:
#         import pdfrw as PyPDF
#     except ImportError:
#         pdfrw = None


# log = G.Global_Logger('converter')
# log.setlevel(G.LOG_LEVEL)

def convert_pdf(filename, resolution=150, jpeg=True, png=False):

    #file might not exist, but this will just trap an execption
    if filename is None:
        return

    try:
        ext = filename[-4:]
        if ext.lower() != ".pdf":
            try:
                print("Invalid filename passed to elixer::convert_pdf(%s)" % filename)
            except:
                return
    except:
        try:
            print("Invalid filename passed to elixer::convert_pdf(%s)" %filename)
        except:
            return


    try:
        #check that the file exists (can be a timing issue on tacc)
        if not os.path.isfile(filename):
            print("Error converting (%s) to image type. File not found (may be filesystem lag. Will sleep and retry."
                     %(filename) )
            time.sleep(5.0) #5 sec should be plenty

            if not os.path.isfile(filename):
                print(
                    "Error converting (%s) to image type. File still not found. Aborting conversion."
                    % (filename))
                return

        pages = convert_from_path(filename,resolution)
        if png:
            for i in range(len(pages)):
                if i > 0:
                    image_name = filename.rstrip(".pdf") + "_p%02d.png" %i
                else:
                    image_name = filename.rstrip(".pdf") + ".png"
                pages[i].save(image_name,"PNG")
                print("File written: " + image_name)

        if jpeg:
            for i in range(len(pages)):
                if i > 0:
                    image_name = filename.rstrip(".pdf") + "_p%02d.jpg" %i
                else:
                    image_name = filename.rstrip(".pdf") + ".jpg"
                pages[i].save(image_name,"JPEG")
                print("File written: " + image_name)

    except:
        print("Error (1) converting pdf to image type: " + filename, sys.exc_info())
        return


def main():

    #get the pdfs
    pdfs = glob.glob("dispatch_*/*/*.pdf")

    if len(pdfs) == 0: # we may bin in a dispactch folder
        pdfs = glob.glob("*/*.pdf")

        #get the (existing) pngs (not the _mini.png or nei.png)
        pngs = glob.glob("*/*[0-9].png")
    else:
        #get the (existing) pngs (not the _mini.png or nei.png)
        pngs = glob.glob("dispatch_*/*/*[0-9].png")

    num_to_convert = len(pdfs) - len(pngs)

    #for each pdf, if not found in png, convert it
    if num_to_convert > 0:
        ct = 0
        for pdf in pdfs:
            base = pdf.rstrip(".pdf")
            png = pdf.rstrip(".pdf") + ".png"

            if png in pngs:
                continue
            else:
                ct += 1
                print("Converting (%d of %d) %s"%(ct,num_to_convert,pdf))
                #odd ... only system call version works from command line
                #convert_pdf(pdf,png=True,jpeg=False,resolution=RESOLUTION)
                os.system("pdftoppm %s %s -png -singlefile" %(pdf,base))
    else:
        print("Nothing to convert")


if __name__ == '__main__':
    main()