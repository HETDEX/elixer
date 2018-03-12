from astropy.wcs import WCS
import astropy.io.fits as fits
import argparse




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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help='Fits filename',required=True)

    args = parser.parse_args()
    try:
        footprint = WCS.calc_footprint(WCS(args.file))
    except:
        img = fits.open(args.file)
        footprint = WCS.calc_footprint(build_wcs_manually(img))
        img.close()


    print("UL", footprint[1],"  UR", footprint[2])
    print("LL", footprint[0],"  LR", footprint[3])


if __name__ == '__main__':
    main()