"""
This is a wrapper around reextract_hsc.py, which prepares the detection and fiber tables as fits files for the
  provided input coordinates.
This wrapper reads in a list of coordinate files and calls into reextract_hsc.py for each of those files.

The intent/assumption here is that the coordinates are broken in to files with one per shotid and the output, then, is
  per shotid.

You need to look into reextract_hsc.py for the input definitions, etc

You need to edit the coord_files as needed to select what you want
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm


args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

#form is integer dateobs (no 'v')
if "--start":
    i = args.index("--start")
    try:
        start_index = int(sys.argv[i + 1])
    except:
        print("no start")
        exit(-1)
else:
    print("no start")
    exit(-1)

if "--step":
    i = args.index("--step")
    try:
        step = int(sys.argv[i + 1])
    except:
        print("no step")
        exit(-1)
else:
    print("no step")
    exit(-1)

if "--sky":
    i = args.index("--sky")
    try:
        skychoice = int(sys.argv[i + 1])
    except:
        print("sky fail")
        exit(-1)
else:
    print("default sky (0 = ll)")
    skychoice = 0  # 0 - local, 1 = ffsky, 2 = ffsky + rescor

#read in all the shot files and remove those that already have output
coord_files = sorted(glob.glob("coords/*_hsc_sep_rex_flags_nei1p5.coords"))
last_file = coord_files[-1]
last_file_idx = len(coord_files) - 1
print(f"{len(coord_files)} coord files found, starting at index {start_index} and using every {step}th ... ")
coord_files = coord_files[start_index:-1:step]

#due to the slicing, we will miss the very last file, so, if the sliced list plus the next step would have hit
#the last file, append that last file
if len(coord_files)*step + step  == last_file_idx:
    coord_files.append(last_file)


for inputfile in tqdm(coord_files):
    #check if the expected output already exists (match what is expected from reextract_hsc.py)
    basename = os.path.basename(inputfile).split(".")[0]
    dT_name = f"{basename}_dets"  # _ffrc"#.fits"
    fT_name = f"{basename}_fibers"  # _ffrc" #.fits"

    bin_ctstr = ""
    if skychoice == 0:  # local
        ffsky = False
        rescor = False
        sky_ext = "_ll"
        dT_name += sky_ext
        fT_name += sky_ext
    elif skychoice == 1:  # local
        ffsky = True
        rescor = False
        sky_ext = "_ff"
        dT_name += sky_ext
        fT_name += sky_ext
    elif skychoice == 2:  # local
        ffsky = True
        rescor = True
        sky_ext = "_rc"
        dT_name += sky_ext
        fT_name += sky_ext
    else:
        print("invalid sky")
        continue

    if os.path.exists(basename + ".h5") or os.path.exists(basename + sky_ext + ".h5"):
    #if os.path.exists(dT_name + bin_ctstr + ".fits") and os.path.exists(fT_name + bin_ctstr + ".fits"):
        print(f"{basename} already done, skipping ...")
        continue

    #good to go?
    os.system(f"python reextract_hsc.py --input {inputfile} --sky {skychoice}")