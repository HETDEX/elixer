#to use, pipe the output to some text file, then repalce hsc_meta.py with that text file
#NOTE: the last entry in the dictionary has a trailing comma ... need to remove that

from astropy.wcs import WCS
import astropy
import astropy.io.fits as fits
import argparse
import numpy as np
import os
import glob2 #easier to search for files


basepath = "/scratch/03261/polonius/hdr2.1.2/imaging/cfhtls/"

def build_wcs_automatically(fname):
    wcs = wcs = WCS(fname,relax = astropy.wcs.WCSHDR_CD00i00j | astropy.wcs.WCSHDR_PC00i00j)
    return wcs



def main():

    print("CFHTLS_META_DICT = {")

    min_ra = 361.0
    max_ra = 0.0
    min_dec = 90.0
    max_dec = -90.0

    img_path = basepath
    #img_path = os.path.join(basepath, "image_tract_patch")

    #most are "wide" files
    #there are two D-25
    #and one D3 which will be handled separately

    #can be one or more directories between the image files
    #files = os.listdir(img_path)
    files = glob2.glob(img_path + "D3.?.fits")
    #files += glob2.glob(img_path + "CFHTLS_D*.fits")
    files += glob2.glob(img_path + "CFHTLS_W*.fits")


    for f in files:
        #img = fits.open(os.path.join(img_path,f))
        #footprint = WCS.calc_footprint(build_wcs_manually_1(img))
        fname = os.path.basename(f)
        footprint = WCS.calc_footprint(build_wcs_automatically(f))

        ra_lo = np.min(footprint[:, 0]) #min ra
        ra_hi = np.max(footprint[:, 0])  # max ra
        dec_lo = np.min(footprint[:, 1])  # min dec
        dec_hi = np.max(footprint[:, 1])  # max dec
        if min_ra > ra_lo:
            min_ra = ra_lo

        if max_ra < ra_hi:
            max_ra = ra_hi

        if min_dec > dec_lo:
            min_dec = dec_lo

        if max_dec < dec_hi:
            max_dec = dec_hi

        header = fits.getheader(f)
        filter = header['filter']
        try:
            instrument = header['instrume']
        except:
            instrument = "MegaPrime"

        print(
            "'%s': {'RA_min':%f,'RA_max':%f,'Dec_min':%f,'Dec_max':%f,"
            "'instrument':'%s','filter':'%s','path':'%s'},"
            % (fname, ra_lo, ra_hi, dec_lo, dec_hi, instrument, filter,f))

        #img.close()
    print("}")

    print("Image_Coord_Range = {'RA_min':%f, 'RA_max':%f, 'Dec_min':%f, 'Dec_max':%f}" %(min_ra,max_ra,min_dec,max_dec))


if __name__ == '__main__':
    main()