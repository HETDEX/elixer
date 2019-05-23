from astropy.wcs import WCS
import astropy.io.fits as fits
import argparse
import numpy as np




#super simple ... load a FITS file and print its footprint


# def build_wcs_manually(img):
#     wcs = WCS(naxis=img[0].header['NAXIS'])
#     wcs.wcs.crpix = [img[0].header['CRPIX1'], img[0].header['CRPIX2']]
#     wcs.wcs.crval = [img[0].header['CRVAL1'], img[0].header['CRVAL2']]
#     wcs.wcs.ctype = [img[0].header['CTYPE1'], img[0].header['CTYPE2']]
#     # self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
#     wcs.wcs.cd = [[img[0].header['CD1_1'], img[0].header['CD1_2']],
#                        [img[0].header['CD2_1'], img[0].header['CD2_2']]]
#     wcs._naxis1 = img[0].header['NAXIS1']
#     wcs._naxis2 = img[0].header['NAXIS2']
#
#     return wcs


# def build_wcs_manually_1(img):
#     wcs = WCS(naxis=img[1].header['NAXIS'])
#     wcs.wcs.crpix = [img[1].header['CRPIX1'], img[1].header['CRPIX2']]
#     wcs.wcs.crval = [img[1].header['CRVAL1'], img[1].header['CRVAL2']]
#     wcs.wcs.ctype = [img[1].header['CTYPE1'], img[1].header['CTYPE2']]
#     # self.wcs.wcs.cdelt = [None,None]#[hdu1[1].header['CDELT1O'],hdu1[1].header['CDELT2O']]
#     wcs.wcs.cd = [[img[1].header['CD1_1'], img[1].header['CD1_2']],
#                        [img[1].header['CD2_1'], img[1].header['CD2_2']]]
#     wcs._naxis1 = img[1].header['NAXIS1']
#     wcs._naxis2 = img[1].header['NAXIS2']
#
#     return wcs

def build_wcs_manually(img, idx=0):
    wcs = WCS(naxis=img[idx].header['NAXIS'])
    wcs.wcs.crpix = [img[idx].header['CRPIX1'], img[idx].header['CRPIX2']]
    wcs.wcs.crval = [img[idx].header['CRVAL1'], img[idx].header['CRVAL2']]
    wcs.wcs.ctype = [img[idx].header['CTYPE1'], img[idx].header['CTYPE2']]
    # self.wcs.wcs.cdelt = [None,None]#[hdu1[0].header['CDELT1O'],hdu1[0].header['CDELT2O']]
    wcs.wcs.cd = [[img[idx].header['CD1_1'], img[idx].header['CD1_2']],
                       [img[0].header['CD2_1'], img[idx].header['CD2_2']]]
    #wcs._naxis1 = img[idx].header['NAXIS1']
    #wcs._naxis2 = img[idx].header['NAXIS2']
    wcs.pixel_shape = (img[idx].header['NAXIS1'], img[idx].header['NAXIS2'])

    return wcs



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help='Fits filename',required=True)

    args = parser.parse_args()
#    try:
#        footprint = WCS.calc_footprint(WCS(args.file))
#    except:
#        img = fits.open(args.file)
#        footprint = WCS.calc_footprint(build_wcs_manually(img))
#        img.close()

    img = fits.open(args.file)

    for idx in range(len(img)):
        try:
            footprint = WCS.calc_footprint(build_wcs_manually(img,idx))
            break
        except:
            print("Img[%d] failed. Trying next ..." %idx)

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