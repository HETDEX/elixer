import os.path

from astropy.wcs import WCS
import astropy.io.fits as fits
import astropy
import argparse
import numpy as np
import sys




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


def build_wcs_automatically(fname):
    wcs = wcs = WCS(fname,relax = astropy.wcs.WCSHDR_CD00i00j | astropy.wcs.WCSHDR_PC00i00j)
    return wcs


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


def find_target_image(image_dict,ra,dec):
    """

    :param image_dict:
    :param ra:
    :param dec:
    :param basepath:
    :return:
    """

    keys = []

    for k in image_dict:
        # don't bother to load if ra, dec not in range
        try:

            if image_dict[k]['RA_max'] - image_dict[k]['RA_min'] < 30:  # 30 deg as a big value
                # ie. we are NOT crossing the 0 or 360 deg line
                if not ((ra >= image_dict[k]['RA_min']) and (ra <= image_dict[k]['RA_max']) and
                        (dec >= image_dict[k]['Dec_min']) and (dec <= image_dict[k]['Dec_max'])):
                    continue
                else:
                    keys.append(k)
            else:  # we are crossing the 0/360 boundary, so we need to be greater than the max (ie. between max and 360)
                # OR less than the min (between 0 and the minimum)
                if not ((ra <= image_dict[k]['RA_min']) or (ra >= image_dict[k]['RA_max']) and
                        (dec >= image_dict[k]['Dec_min']) and (dec <= image_dict[k]['Dec_max'])):
                    continue
                else:
                    keys.append(k)

        except:
            pass


    if len(keys) == 0:  # we're done ... did not find any
        return None
    elif len(keys) == 1:  # found exactly one
        tile = keys  # remember tile is a string ... there can be only one
    elif len(keys) > 1:  # find the best one
        max_dist = 0
        tiles_dist = []
        for k in keys:
            # should not be negative, but could be?
            # in any case, the min is the smallest distance to an edge in RA and Dec
            inside_ra = abs(min((ra - image_dict[k]['RA_min']), (image_dict[k]['RA_max'] - ra)))
            inside_dec = min((dec - image_dict[k]['Dec_min']), (image_dict[k]['Dec_max'] - dec))

            tiles_dist.append(min(inside_dec, inside_ra))
            # we want the tile with the largest minium edge distance


        idx = np.argsort(tiles_dist)[::-1]
        tile = np.array(keys)[idx]

    else:
        return None

    return tile

def build_simple_dictionary(filenames):
    """
    iterate over filenames and return a dictionary keyed by the filename (just the basename) with the min,max of RA, Dec

    :param filenames: an array of filenames or fqpn
    :return: dictionary
    """

    try:

        the_dict={}
        for fn in filenames:
            try:
                bn = os.path.basename(fn)

                img = fits.open(fn)
                footprint = None

                for idx in range(len(img)):
                    try:
                        # footprint = WCS.calc_footprint(build_wcs_manually(img,idx))
                        footprint = WCS.calc_footprint(build_wcs_automatically(fn))
                        break
                    except:
                        print("Img[%d] failed. Trying next ..." % idx)
                        print(sys.exc_info())

                img.close()

                if footprint is None:
                    the_dict[bn] = {'RA_min': None,
                                    'RA_max': None,
                                    'Dec_min': None,
                                    'Dec_max': None}
                else:
                    the_dict[bn] = {'RA_min': np.min(footprint[:, 0]),
                                    'RA_max': np.max(footprint[:, 0]),
                                    'Dec_min': np.min(footprint[:, 1]),
                                    'Dec_max': np.max(footprint[:, 1])}


            except Exception as e:
                print(e)


        return the_dict
    except Exception as e:
        print(e)


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
    footprint = None

    for idx in range(len(img)):
        try:
            #footprint = WCS.calc_footprint(build_wcs_manually(img,idx))
            footprint = WCS.calc_footprint(build_wcs_automatically(args.file))
            break
        except:
            print("Img[%d] failed. Trying next ..." %idx)
            print(sys.exc_info())

    img.close()

    if footprint is None:
        print("Unable to compute footprint")
        exit()

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