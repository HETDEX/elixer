"""
find the dispatch_xxxx that are incomplete from their list of detections
"""

import glob
import os
import numpy as np



#get all the dispatches

dispatches = sorted(glob.glob("dispatch_*"))
basename = os.path.basename(os.getcwd())

missing_dispatches = []

for dispatch in dispatches:
    detectids = np.loadtxt(os.path.join(dispatch,dispatch),dtype=int)
    detectids = detectids[::-1] #most likely to be the one at the end

    #it may not necessarily be the last one, if there was a hiccup on another one
    #so need to check them all, but start at the end which is more likely to not have been completed
    #(could also check on the _nei.png or _mini.png)
    for detectid in detectids:
        if not os.path.exists(os.path.join(dispatch,basename,f"{detectid}.png")):
            print(dispatch)
            missing_dispatches.append(dispatch)
            break


#now update the elixer.run
_,run_dispatches = np.loadtxt("elixer.run",dtype=str,usecols=[0,1],unpack=True)

tasks = np.loadtxt("elixer.run",dtype=str,unpack=False) #want each row as one unit, do not unpack

_,_, idx = np.intersect1d(missing_dispatches,run_dispatches,return_indices=True)

with open("elixer_sweep.run","w+") as f:
    for i in idx:
        f.write(' '.join(tasks[i])+"\n")


#print(missing_dispatches)
