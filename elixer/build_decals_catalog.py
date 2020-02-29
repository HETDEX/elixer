import glob
from astropy.table import Table, vstack
from astropy.io import fits, ascii
import requests
import io
import os
import datetime

#load the list of bricks
# BRICKNAME	char[8]	Name of the brick.
# BRICKID	int32	A unique integer with 1-to-1 mapping to brickname.
# BRICKQ	int16	A "priority" factor used for processing.
# BRICKROW	int32	Dec row number.
# BRICKCOL	int32	Number of the brick within a Dec row.
# RA	float64	RA of the center of the brick.
# DEC	float64	Dec of the center of the brick.
# RA1	float64	Lower RA boundary.
# RA2	float64	Upper RA boundary.
# DEC1	float64	Lower Dec boundary.
# DEC2	float64	Upper Dec boundary.
#bricks_fn = "/home/dustin/temp/survey-bricks.fits"
#ra_str = 'RA'
#dec_str = 'DEC'
#brickname_str = 'BRICKNAME'
#bricks_fn = "/home/dustin/temp/survey-bricks-dr8-south.fits" #DR8 (south only) ... the spring field should be empty
bricks_fn = "survey-bricks-dr8-south.fits"
ra_str = 'ra'
dec_str = 'dec'
brickname_str = 'brickname'

base_url = "https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/south/tractor/"
bricks = Table.read(bricks_fn)


spring_bricks = bricks[(bricks[ra_str] > 155.) & (bricks[ra_str] < 230.) & (bricks[dec_str] > 40.) & (bricks[dec_str]< 60.)]
fall_bricks = bricks[(bricks[ra_str] > 5.) & (bricks[ra_str] < 40.) & (bricks[dec_str] > -7.) & (bricks[dec_str]< 7.)]
#again if DR8 south, expect only in fall field ... the spring should be empty
#go get them all from web calls

outfile = "decals_dr8_master_catalog.fits"
keep_columns = ['release','brickid','brickname','objid','brick_primary','brightblob',
                'type','ra','dec','ref_cat','ref_id','flux_g','flux_r','flux_z',
                'psfsize_g','psfsize_r','psfsize_z','psfdepth_g','psfdepth_r','psfdepth_z' ]
master_table = Table()

total_sz = len(spring_bricks) + len(fall_bricks)
ct = 0
start_time = datetime.datetime.now()
last_iter_time = start_time

for field in [spring_bricks,fall_bricks]:
    for bn in field[brickname_str]: #like : 0051m067
        try:
            split_time = datetime.datetime.now()
            elapsed_time = split_time-start_time
            split_time = split_time-last_iter_time
            remaining_time = datetime.timedelta(seconds=(total_sz-ct)*split_time.total_seconds())

            last_iter_time = datetime.datetime.now() #mark this time for the next loop

            ct += 1
            dir1 = bn[0:3] #this is the 000 to 359 (the first 3 chars of the brickname
            fn = "tractor-%s.fits" %bn
            url = base_url + dir1 + "/" + fn

            file_obj = None

            try:
                if (os.path.isfile(fn)):
                    print("%d / %d : %s ***ALREADY DOWNLOADED*** split %f  elapsed %f  remaining %s" %
                          (ct,total_sz,fn,split_time.total_seconds(),elapsed_time.total_seconds(),str(remaining_time)))
                    file_obj = open(fn,"rb")
                else:
                    print("%d / %d : %s  split %f  elapsed %f  remaining %s" %
                          (ct, total_sz, fn, split_time.total_seconds(), elapsed_time.total_seconds(),str(remaining_time)))

                    response = requests.get(url, allow_redirects=True)

                    if response.status_code == 200:  # "OK" response
                        # rather than write out the whole catalog file, lets trim to the columns I want to save ?
                        # there are a lot of files, so,maybe go ahead and (at least temporarily write out)
                        open(fn, 'wb').write(response.content)
                        file_obj = io.BytesIO(response.content)
                    else:
                        print("DECaLS http response code = %d (%s)" % (response.status_code, response.reason))
                        file_obj = None

            except:
                print("%d / %d : %s ***EXCEPTION***" %(ct,total_sz,fn))

            if file_obj:
                t = Table.read(file_obj)
                file_obj.close()
                for name in t.colnames:
                    if name not in keep_columns:
                        t.remove_column(name)

                #write back out with reduced columns
                t.write(fn,overwrite=True, format="fits")

                #todo: what about checking for duplicates?
                master_table = vstack([master_table, t])
                #write at each step?? #too costly, but we will have the files, so can recreate if this fails along the way
                #master_table.write(outfile, overwrite=True, format="fits")

        except Exception as e:
            print(e)

#write once at the end? #should be of order 4-5 GB
master_table.write(outfile,overwrite=True,format="fits")



if False:
    #iterate over all Tables and concatenat (vstack) over rows
    outfile = "decals_dr8_master_catalog.fits"
    keep_columns = ['release','brickid','brickname','objid','brick_primary','brightblob',
                    'type','ra','dec','ref_cat','ref_id','flux_g','flux_r','flux_z',
                    'psfsize_g','psfsize_r','psfsize_z','psfdepth_g','psfdepth_r','psfdepth_z',
                    ]
    table_names = glob.glob("/home/dustin/temp/tractor*.fits")
    sz = len(table_names)
    master_table = Table()
    for ct,fn in enumerate(table_names):
        t = Table.read(fn)
        print("%d/%d %s" %(ct,sz,fn))
        for name in t.colnames:
            if name not in keep_columns:
                t.remove_column(name)

        master_table = vstack([master_table,t])


    #now we have master_table all stacked with just the columns we want, so save it
    master_table.write(outfile,format="fits")