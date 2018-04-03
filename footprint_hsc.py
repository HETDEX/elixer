from astropy.wcs import WCS
import astropy.io.fits as fits
import argparse
import numpy as np

basepath = "/work/04094/mshiro/maverick/HSC/S15A/reduced"


#super simple ... load a FITS file and print its footprint


def build_wcs_manually(img):
    wcs = WCS(naxis=img[0].header['NAXIS'])
    wcs.wcs.crpix = [img[0].header['CRPIX1'], img[0].header['CRPIX2']]
    wcs.wcs.crval = [img[0].header['CRVAL1'], img[0].header['CRVAL2']]
    wcs.wcs.ctype = [img[0].header['CTYPE1'], img[0].header['CTYPE2']]
    # self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
    wcs.wcs.cd = [[img[0].header['CD1_1'], img[0].header['CD1_2']],
                       [img[0].header['CD2_1'], img[0].header['CD2_2']]]
    wcs._naxis1 = img[0].header['NAXIS1']
    wcs._naxis2 = img[0].header['NAXIS2']

    return wcs


def build_wcs_manually_1(img):
    wcs = WCS(naxis=img[1].header['NAXIS'])
    wcs.wcs.crpix = [img[1].header['CRPIX1'], img[1].header['CRPIX2']]
    wcs.wcs.crval = [img[1].header['CRVAL1'], img[1].header['CRVAL2']]
    wcs.wcs.ctype = [img[1].header['CTYPE1'], img[1].header['CTYPE2']]
    # self.wcs.wcs.cdelt = [None,None]#[hdu1[1].header['CDELT1O'],hdu1[1].header['CDELT2O']]
    wcs.wcs.cd = [[img[1].header['CD1_1'], img[1].header['CD1_2']],
                       [img[1].header['CD2_1'], img[1].header['CD2_2']]]
    wcs._naxis1 = img[1].header['NAXIS1']
    wcs._naxis2 = img[1].header['NAXIS2']

    return wcs


#SHELA example
# if multiple images, the composite broadest range (filled in by hand)
Image_Coord_Range = {'RA_min': 8.50, 'RA_max': 36.51, 'Dec_min': -4.0, 'Dec_max': 4.0}
#approximate
Tile_Coord_Range = {
                    'A1': {'RA_min':  8.50, 'RA_max': 11.31, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A2': {'RA_min': 11.30, 'RA_max': 14.11, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A3': {'RA_min': 14.10, 'RA_max': 16.91, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A5': {'RA_min': 19.70, 'RA_max': 22.51, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A6': {'RA_min': 22.50, 'RA_max': 25.31, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A7': {'RA_min': 25.30, 'RA_max': 28.11, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A8': {'RA_min': 28.10, 'RA_max': 30.91, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A9': {'RA_min': 30.90, 'RA_max': 33.71, 'Dec_min': 1.32, 'Dec_max': 4.0},
                    'A10':{'RA_min': 33.70, 'RA_max': 36.51, 'Dec_min': 1.32, 'Dec_max': 4.0},

                    'B1': {'RA_min':  8.50, 'RA_max': 11.31, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B2': {'RA_min': 11.30, 'RA_max': 14.11, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B3': {'RA_min': 14.10, 'RA_max': 16.91, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B5': {'RA_min': 19.70, 'RA_max': 22.51, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B6': {'RA_min': 22.50, 'RA_max': 25.31, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B7': {'RA_min': 25.30, 'RA_max': 28.11, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B8': {'RA_min': 28.10, 'RA_max': 30.91, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B9': {'RA_min': 30.90, 'RA_max': 33.71, 'Dec_min': -1.35, 'Dec_max': 1.34},
                    'B10':{'RA_min': 33.70, 'RA_max': 36.51, 'Dec_min': -1.35, 'Dec_max': 1.34},

                    'C1': {'RA_min':  8.50, 'RA_max': 11.31, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C2': {'RA_min': 11.30, 'RA_max': 14.11, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C3': {'RA_min': 14.10, 'RA_max': 16.91, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C4': {'RA_min': 16.90, 'RA_max': 19.71, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C5': {'RA_min': 19.70, 'RA_max': 22.51, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C6': {'RA_min': 22.50, 'RA_max': 25.31, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C7': {'RA_min': 25.30, 'RA_max': 28.11, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C8': {'RA_min': 28.10, 'RA_max': 30.91, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C9': {'RA_min': 30.90, 'RA_max': 33.71, 'Dec_min': -4.0, 'Dec_max': -1.32},
                    'C10':{'RA_min': 33.70, 'RA_max': 36.51, 'Dec_min': -4.0, 'Dec_max': -1.32},
                }




def parse_hsc_image_name(name):
    #calexp-HSC-R-16666-4,3.fits
    #   parse out the filter (R) and the tile position (4,3) and catalog_tract (16666)

    toks = name.split("-")
    instrument = toks[1] #'HSC'
    filter = toks[2] #'R'
    cat_tract = toks[3] #'16666'
    tile_pos = eval(toks[4].split('.')[0]) #so we get a tuple of ints

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

    #here for f in files ...

    img = fits.open(f)
    footprint = WCS.calc_footprint(build_wcs_manually_1(img))

    img.close()


    print("UL", footprint[1],"  UR", footprint[2])
    print("LL", footprint[0],"  LR", footprint[3])

    print("copy form")
    print("[[%f,%f],[%f,%f],[%f,%f],[%f,%f]]" %
          (footprint[0][0], footprint[0][1],
           footprint[1][0], footprint[1][1],
           footprint[2][0], footprint[2][1],
           footprint[3][0], footprint[3][1],
           ))

    print("Min Max")
    print("{'RA_min': %f, 'RA_max': %f, 'Dec_min': %f, 'Dec_max': %f}" %
          (np.min(footprint[:,0]),np.max(footprint[:,0]),
           np.min(footprint[:,1]), np.max(footprint[:,1])))

    print("Min Max linear dict")
    print("'RA_min': %f," % np.min(footprint[:, 0]))
    print("'RA_max': %f," % np.max(footprint[:, 0]))
    print("'Dec_min': %f," % np.min(footprint[:, 1]))
    print("'Dec_max': %f," % np.max(footprint[:, 1]))

if __name__ == '__main__':
    main()