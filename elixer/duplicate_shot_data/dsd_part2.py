#part two
#checks for duplicate amp+exposure data in a single shotfile


import sys
import os.path as op
import tables
import numpy as np

shot_file = sys.argv[1]
h5 = tables.open_file(shot_file,"r")
itb = h5.root.Data.Images
shot = op.basename(shot_file)[:-3]
dupfile = open(shot +".dup","w")
errfile = open(shot +".err","w")

multiframes = itb.read(field='multiframe')

for mf in multiframes:
    try:
        images = itb.read_where("multiframe==mf", field='image')
        exposures = itb.read_where("multiframe==mf", field='expnum')

        line = f"{shot} {mf}"
        has_dups = False

        for i in range(len(exposures)):
            if np.any(images[i]) and (np.shape(images[i]) == (1032,1032)) \
                    and (len(np.unique(images[i])) > 500):
                for j in range(i+1,len(exposures),1):
                    if np.any(images[j]) and (np.shape(images[j]) == (1032,1032))  \
                    and (len(np.unique(images[j])) > 500):
                        if np.any(images[i]-images[j]):
                            pass #all good, they are not identical
                        else:
                            has_dups = True
                            line += f" ({exposures[i]},{exposures[j]})"

        if has_dups:
            line += "\n"
            dupfile.write(line)

    except Exception as e:
        errfile.write(e)
        errfile.write("\n")


dupfile.close()
errfile.close()