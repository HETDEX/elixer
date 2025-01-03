#!/usr/bin/env python

#this supports the full re-run of ELiXer on all HETDEX data
#Since only a subset of the shot.h5 files are on /scratch/projects/hetdex and to avoid heavy access of
#/corral-repl/utexes/Hobby-Eberly-Telescope/, we want to copy over the shot.h5 files needed for the next
#set of detections to /scratch and run from there. After that is done, we want to remove the shot.h5 files
#from /scratch and resotre the soft-links (that point to /corral-repl)

#The oragnization of detections is in batches of 100,000 in (roughly) date order

#associated are three sets of files
# dXX_hdrX_100000.dets (or cXX for continuum and 100000 may be smaller for the last sets)
#     these are the detections
# sdXX_hdrX_1000000.dets (matching shotid for the same row in the detections file above)
# su_dXX_hdrX_1000000.dets (the unique shotids for the matching file) .. top row is always 0

# to use this script, you must be the hetdex user and work under /scratch/projects/hetdex/hdr<X>/reduction/data
# you must pre-create the ./symlinks direcotry to hold the smymlinks that are being swapped/restored
# you must pre-create the ./used_shots directory to (temporarily) hold the shot.h5 to delete locally AFTER
#     they have been used (that is, after the elixer run that needs them is done)

#NOTICE: the user is then resoponsible for deleting h5 files from ./used_shots once happy that this
# behaved correctly


#take two parameters ... current file (su_xxx) and previous (su_xxx  or None)
#need to think about HDR3 vs HDR4 or just deal with manually being in the right directory

#import glob
import os
import shutil
import sys
import numpy as np
from tqdm import tqdm

#safety check, make sure we are where we should be
working_root = "/home/dustin/test/"
#working_root = "/scratch/projects/hetdex/hdr3/"
if os.getcwd()[:len(working_root)] != working_root:
    print(f"Wrong directory. Must only run under {working_root}, under hdr<x>/reductions/data/")
    exit(-1)

if not os.path.isdir(f"{os.getcwd()}/symlinks"):
    print(f"symlinks subdir does not exist. Exiting.")
    exit(-1)

if not os.path.isdir(f"{os.getcwd()}/used_shots"):
    print(f"used_shots subdir does not exist. Exiting.")
    exit(-1)

cl_args = list(map(str,sys.argv))
if cl_args[1] != "skip":
  reset_only = False
  curr_shots = np.loadtxt(cl_args[1],dtype=int)
  if curr_shots[0] == 0:
      curr_shots = np.delete(curr_shots,0)
else:
  reset_only = True

fail = False
try:
    if cl_args[2] == cl_args[1]:
        print("Cannot use the same file for current and previous")
        fail = True
except:
    pass

if fail:
    exit(-1)

try:
    prev_fn = np.loadtxt(cl_args[2],dtype=int)
    if prev_fn[0] == 0:
        prev_fn = np.delete(prev_fn,0)
except:
    prev_fn = None


if prev_fn is not None:
    mv_ct = 0
    print("Restoring symlinks from previous run ...")
    for shot in tqdm(prev_fn):
        if shot not in curr_shots:
            shotfn = f"{str(shot)[0:8]}v{str(shot)[8:]}.h5"
            if os.path.isfile(shotfn) and not os.path.islink(shotfn):
                ##delete the file
                #os.remove(shotfn)
                if os.path.islink(os.path.join(f"./symlinks/{shotfn}")):
                  #IF the link exists, mv the shot file and restore the link
                  #no ... just move it to a temp location for safety ... user should manually delete from there
                  shutil.move(shotfn,os.path.join(f"./used_shots/{shotfn}"))

                  #restore the link
                  shutil.copy2(os.path.join(f"./symlinks/{shotfn}"),shotfn,follow_symlinks=False)
                  mv_ct += 1
                #else: #this link did not exist, this was a flat file already, so ignore and move on
                #  pass

    if mv_ct == 0:
        print("Warning! No files moved/restored. Are the <current> <previous> inputs correct?")
else:
    print("No previous file provided")


if reset_only:
  print("Reset Only. Exiting.")
  exit(0)


for shot in tqdm(curr_shots):
    shotfn = f"{str(shot)[0:8]}v{str(shot)[8:]}.h5"
    if os.path.islink(shotfn):
        #backup the link
        if not os.path.islink(os.path.join(f"./symlinks/{shotfn}")): #yes, just want to see if it exists already
#            print(f"{os.path.join(f'./symlinks/{shotfn}')} does not exist? ")
            shutil.copy2(shotfn,os.path.join(f"./symlinks/{shotfn}"),follow_symlinks=False)
        shutil.copy2(shotfn,f"{shotfn}.ln",follow_symlinks=False)

        #remove the link
        os.unlink(shotfn)

        #copy real file
        shutil.copy2(f"{shotfn}.ln",shotfn,follow_symlinks=True)

        #remove temp link
        os.unlink(f"{shotfn}.ln")

