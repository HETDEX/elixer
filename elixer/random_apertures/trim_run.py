"""
clean up the rand_ap.run file (or other similar) based on tasks already completed

get list of completed tasks by shotid and remove those matching lined from the .run file

"""

import sys
import os
import glob
import numpy as np


#get list of already done (change as needed)
#basepath = "/scratch/03261/polonius/random_apertures/hdr4/fixed24.3_all/"
#os.chdir(basepath)
basepath = "./"
pattern = "random_apertures_*_ff_fibers.fits"
done_fn = glob.glob(basepath+pattern)

#extract the shotids
done_shots = [f.split("_")[2] for f in done_fn]


#get the run file
run_file_toks = np.loadtxt(basepath+"rand_ap.run",dtype=str,unpack=False)
#find the shotid
shot_idx = np.argwhere(run_file_toks[0]=="--shot")[0][0] + 1
run_file_shots = [x[shot_idx] for x in run_file_toks]
missing_shots = np.setdiff1d(run_file_shots,done_shots)

with open("rand_ap.run.new","w+") as f:
    for i in range(len(run_file_shots)):
        if run_file_shots[i] in missing_shots:
            f.write(' '.join(run_file_toks[i])+"\n")

