import sys
try:
    from elixer import catalogs
except:
    import catalogs

from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.io.fits as fits
import logging

logging.basicConfig(filename="cut2fits.log", level=logging.DEBUG, filemode='w')

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#keep it simple, don't bother with args.parser
#need --ra --dec --radius

ra = None
dec = None
radius = None
name = None

if "--ra" in args:
    i = args.index("--ra")
    ra = float(sys.argv[i + 1])
else: #we're done
    print("Missing mandatory parameter: --ra")
    exit(0)

if "--dec" in args:
    i = args.index("--dec")
    dec = float(sys.argv[i + 1])
else: #we're done
    print("Missing mandatory parameter: --dec")
    exit(0)


if "--radius" in args:
    i = args.index("--radius")
    radius = float(sys.argv[i + 1])
else: #we're done
    print("Missing mandatory parameter: --radius")
    exit(0)

if "--name" in args:
    i = args.index("--name")
    name = sys.argv[i + 1]
else:  # we're done
    print("Missing mandatory parameter: --name")
    exit(0)

catlib = catalogs.CatalogLibrary()

coord = SkyCoord(ra*u.degree,dec*u.degree)
#filters = catlib.get_filters(coord,catalogs=None)
#cutouts =  catlib.get_cutouts(coord,side=radius*3.,filter=['*','r','g','F606W','*'], first=False,nudge=0.0,aperture=1.5)#[0]
cutouts =  catlib.get_cutouts(coord,side=radius*3.,filter=['*'], first=False,nudge=0.0,aperture=1.5)#[0]

#cutouts = catlib.get_cutouts(coord,radius*2./3600.*u.degree)
#radius*2. s|t the given value becomes a side of the square
#/3600.0 because the bash wrapper assumes arcesec but we pass in as decimal degreees

#print(cutouts)

for i in range(len(cutouts)):
    fn = '%s_%d.fits' % (name, i)
    co = cutouts[i]['cutout']
    if co is None or co.data is None:
        print("Warning. Bad data for (%s [%s])" %(cutouts[i]['instrument'],cutouts[i]['filter']))
        continue
    hdu = fits.PrimaryHDU(co.data) #essentially empty header
    hdu.header.update(co.wcs.to_header()) #insert the cutout's WCS
    print("Writing (%s [%s]) %s ..." %(cutouts[i]['instrument'],cutouts[i]['filter'],fn))
    hdu.writeto(fn, overwrite=True)