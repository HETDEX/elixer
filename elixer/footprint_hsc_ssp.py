#to use, pipe the output to some text file, then repalce hsc_meta.py with that text file
#NOTE: the last entry in the dictionary has a trailing comma ... need to remove that


"""
hsc_ssp : all wide so: (2" ??)
g: 26.6
r: 26.2
i: 26.2
z: 25.3
y: 24.5


cosmos:  all pdr2_wide/deepCoadd_-results
w01: all pdr2_wide/deepCoadd_-results/HSC-X
w02: ditto


"""
from astropy.wcs import WCS
import astropy
import astropy.io.fits as fits
import argparse
import numpy as np
import os
import glob2 #easier to search for files


basepath = "/scratch/03946/hetdex/imaging/hsc_ssp"

#subdirs:
# cosmos\
#   g
#   r
# w01
#   g
#   r
#   .... i, z, y ....
# w02

#magdepths for the "wide" exposures
mag_depth = {'g':26.6,
             'r':26.2,
             'i':26.2,
             'z':25.3,
             'y':24.5}

subdirs = ["cosmos/g",
           "cosmos/r",
           "w01/g",
           "w01/r",
           "w01/i",
           "w01/z",
           "w01/y",
           "w02/g",
           "w02/r",
           "w02/i",
           "w02/z",
           "w02/y",
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

    toks = name.split("-")
    instrument = toks[1] #'HSC'
    filter = toks[2] #'R'
    cat_tract = toks[3] #'16666'
    #tile_pos = eval(toks[4].split('.')[0]) #so we get a tuple of ints

    pos = toks[4].split('.')[0].split(",") #ie. "16.fits" --> 1 6   # so we get a tuple of ints

    tile_pos = [int(pos[0]),int(pos[1])]

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

    for sub in subdirs:
        img_path = os.path.join(basepath, sub)


        #can be one or more directories between the image files
        #files = os.listdir(img_path)
        files = glob2.glob(img_path + "/**/*.fits")

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

            print("'%s': {'RA_min':%f,'RA_max':%f,'Dec_min':%f,'Dec_max':%f,'instrument':'%s','filter':'%s','depth':%f,'tract':'%s','pos':(%d,%d),'path':'%s'},"
                  %(fname,ra_lo,ra_hi,dec_lo,dec_hi,instrument,filter.lower(),mag_depth[filter.lower()],cat_tract,tile_pos[0],tile_pos[1],f))


            #img.close()
    print("}")

    print("Image_Coord_Range = {'RA_min':%f, 'RA_max':%f, 'Dec_min':%f, 'Dec_max':%f}" %(min_ra,max_ra,min_dec,max_dec))


if __name__ == '__main__':
    main()