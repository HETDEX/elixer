######!/usr/bin/env python

"""
find the dispatch_xxxx that are incomplete from their list of detections
"""

import glob
import os
import numpy as np
from tqdm import tqdm

workdir = "./"


print_missing_dets = False

#get all the dispatches
dispatches = sorted(glob.glob(os.path.join(workdir,"dispatch_*")))
basename = os.path.basename(os.getcwd())

missing_dispatches = []

for dispatch in tqdm(dispatches):
    detectids = np.loadtxt(os.path.join(dispatch,os.path.basename(dispatch)),dtype=int)
    detectids = detectids[::-1] #most likely to be the one at the end

    #it may not necessarily be the last one, if there was a hiccup on another one
    #so need to check them all, but start at the end which is more likely to not have been completed
    #(could also check on the _nei.png or _mini.png)
    for detectid in detectids:
        if not os.path.exists(os.path.join(dispatch,basename,f"{detectid}.png")):
            if print_missing_dets:
              print(detectid)
              if detectid not in missing_dispatches:
                  missing_dispatches.append(os.path.basename(dispatch))

            else:
              print(dispatch)
              missing_dispatches.append(os.path.basename(dispatch))
              break


if len(missing_dispatches) > 0:
    print(f"Incomplete Dispatches: ")
    for md in missing_dispatches:
        print(f"{md}")

#now update the elixer.run
_,run_dispatches = np.loadtxt(os.path.join(workdir,"elixer.run"),dtype=str,usecols=[0,1],unpack=True)


tasks = []
with open(os.path.join(workdir,"elixer.run"),"r") as f:
    while line := f.readline():
        tasks.append(line)

_,_, idx = np.intersect1d(missing_dispatches,run_dispatches,return_indices=True)

print(f"Found {len(idx)} matching task. Writing out sweep file ...")

#print(missing_dispatches)
with open("elixer_sweep.run","w+") as f:
    for i in idx:
        #f.write(' '.join(tasks[i])+"\n")
        #print(' '.join(tasks[i])+"\n")
        f.write(tasks[i])
        #print(tasks[i])

