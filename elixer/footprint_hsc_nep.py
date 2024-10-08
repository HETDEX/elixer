#to use, pipe the output to some text file, then repalce hsc_meta.py with that text file
#NOTE: the last entry in the dictionary has a trailing comma ... need to remove that


"""
HSC_NEP

"""
from astropy.wcs import WCS
import astropy
import astropy.io.fits as fits
import argparse
import numpy as np
import os
import glob2 #easier to search for files

#basepath = "/home1/07446/astroboi/work2/project/Project_NEP/data/processed/HSC_Files/current_overlap"
#basepath = "/scratch/03261/polonius/nep/"
basepath = "/scratch/07446/astroboi/H20_Mosiacs"


mag_depth = {'g':26.6,
             'r':26.2,
             'i':26.2,
             'z':25.3,
             'y':24.5}

subdirs = [
           ]

def build_wcs_automatically(fname):
    f = fits.open(fname)
    wcs = WCS(f[1].header)
    #wcs = WCS(fname,relax = astropy.wcs.WCSHDR_CD00i00j | astropy.wcs.WCSHDR_PC00i00j)
    f.close()
    return wcs

def build_wcs_manually(img):
    wcs = WCS(naxis=img[0].header['NAXIS'])
    wcs.wcs.crpix = [img[0].header['CRPIX1'], img[0].header['CRPIX2']]
    wcs.wcs.crval = [img[0].header['CRVAL1'], img[0].header['CRVAL2']]
    wcs.wcs.ctype = [img[0].header['CTYPE1'], img[0].header['CTYPE2']]
    # self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
    wcs.wcs.cd = [[img[0].header['CD1_1'], img[0].header['CD1_2']],
                       [img[0].header['CD2_1'], img[0].header['CD2_2']]]
    # wcs._naxis1 = img[0].header['NAXIS1']
    # wcs._naxis2 = img[0].header['NAXIS2']

    wcs.pixel_shape = (img[0].header['NAXIS1'], img[0].header['NAXIS2'])

    return wcs


def build_wcs_manually_1(img):
    wcs = WCS(naxis=img[1].header['NAXIS'])
    wcs.wcs.crpix = [img[1].header['CRPIX1'], img[1].header['CRPIX2']]
    wcs.wcs.crval = [img[1].header['CRVAL1'], img[1].header['CRVAL2']]
    wcs.wcs.ctype = [img[1].header['CTYPE1'], img[1].header['CTYPE2']]
    # self.wcs.wcs.cdelt = [None,None]#[hdu1[1].header['CDELT1O'],hdu1[1].header['CDELT2O']]
    wcs.wcs.cd = [[img[1].header['CD1_1'], img[1].header['CD1_2']],
                       [img[1].header['CD2_1'], img[1].header['CD2_2']]]
    # wcs._naxis1 = img[1].header['NAXIS1']
    # wcs._naxis2 = img[1].header['NAXIS2']
    wcs.pixel_shape = (img[1].header['NAXIS1'], img[1].header['NAXIS2'])

    return wcs


def parse_hsc_image_name(name):
    #calexp-HSC-R-16666-4,3.fits
    #cosmo: calexp-HSC-G-9569-6,7.fits
    #   parse out the filter (R) and the tile position (4,3) and catalog_tract (16666)

    #H20_NEP_T18212_B83_MULTIBAND.fits
    #H20_EDFN_v2.3_HSC-R.fits

    toks = name.split("_")
    instrument = "HSC NEP" #toks[1] #'HSC'
    #old  H20_NEP_T18212_B83_MULTIBAND.fits  style
    #filter = toks[2] #'R'
    #cat_tract = toks[2][1:] #'16666'
    #tile_pos = eval(toks[4].split('.')[0]) #so we get a tuple of ints

    #newer H20_EDFN_v2.3_HSC-R.fits style
    filter =toks[3][4].lower()
    cat_tract = "xxx"
    tile_pos = "xxx"
    pos = (0,0) #toks[4].split('.')[0].split(",") #ie. "16.fits" --> 1 6   # so we get a tuple of ints

    return instrument,filter,cat_tract,tile_pos






def main():

    #parser = argparse.ArgumentParser()
    #parser.add_argument('-f','--file', help='Fits filename',required=True)

    #args = parser.parse_args()
#    try:
#        footprint = WCS.calc_footprint(WCS(args.file))
#    except:
#        img = fits.open(args.file)
#        footprint = WCS.calc_footprint(build_wcs_manually(img))
#        img.close()

    #open each file under basepath/images
    #get the range
    #append to dictionary like Tile_Coord_Range where the key is the filename
    #   e.g. calexp-HSC-R-16666-4,3.fits
    #   parse out the filter (R) and the tile position (4,3) and catalog_tract (16666)


    print("HSC_META_DICT = {")

    min_ra = 361.0
    max_ra = 0.0
    min_dec = 90.0
    max_dec = -90.0

    #img_path = os.path.join(basepath,"images")

    img_path = basepath #os.path.join(basepath, sub)


    #can be one or more directories between the image files
    #files = os.listdir(img_path)
    files = sorted(glob2.glob(img_path + "/**/*.fits"))
    exclude_files = sorted(glob2.glob(img_path + "/**/*.var.fits"))

    files = np.setdiff1d(files,exclude_files)
    #any tiles that are bad and should not be loaded go in this list
    bad_list = []

    for f in files:
        #img = fits.open(os.path.join(img_path,f))
        #footprint = WCS.calc_footprint(build_wcs_manually_1(img))
        fname = os.path.basename(f)
        if fname in bad_list:
            continue

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

        instrument,filter,cat_tract,tile_pos = parse_hsc_image_name(os.path.basename(f))

        #20240212 DD: older verion had multiple extensions with mulitple filters per Tile
        #newer version has single large images per filter
        #print("'%s': {'RA_min':%f,'RA_max':%f,'Dec_min':%f,'Dec_max':%f,'instrument':'HSC NEP','filter':'xx','tract':'%s','pos':'%s','path':'%s'},"
        #      %(fname,ra_lo,ra_hi,dec_lo,dec_hi,cat_tract,tile_pos,f))

        print("'%s': {'RA_min':%f,'RA_max':%f,'Dec_min':%f,'Dec_max':%f,'instrument':'HSC NEP','filter':'%s','tract':'%s','pos':'%s','path':'%s'},"
              %(fname,ra_lo,ra_hi,dec_lo,dec_hi,filter,cat_tract,tile_pos,f))


            #img.close()
    print("}")

    print("Image_Coord_Range = {'RA_min':%f, 'RA_max':%f, 'Dec_min':%f, 'Dec_max':%f}" %(min_ra,max_ra,min_dec,max_dec))


if __name__ == '__main__':
    main()