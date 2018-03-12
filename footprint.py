from astropy.wcs import WCS
import astropy.io.fits as fits
import argparse




#super simple ... load a FITS file and print its footprint






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help='Fits filename',required=True)

    args = parser.parse_args()
    footprint = WCS.calc_footprint(WCS(args.file))

    print("UL", footprint[1],"  UR", footprint[2])
    print("LL", footprint[0],"  LR", footprint[3])


if __name__ == '__main__':
    main()