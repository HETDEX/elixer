import os
import sys
import numpy as np
import tables

args = list(map(str.lower,sys.argv))

mergelist_file = args[1]

with open(mergelist_file,'r') as f:
    lines = f.readlines()

for line in lines:
    fn = line[:-1] #strip the trailing \n
    if not os.path.isfile(fn):
        print(line, "No file.")
    else:
        print(line,end=" ")
       # print(" ... ",end="")
    #try to open and close the h5
    try:
        h5 = tables.open_file(fn)
        #do something
        rows = h5.root.Version.read()
        if len(rows) < 1:
            print("Fail Version", len(rows))
            continue

        rows = h5.root.Detections.read()
        if len(rows) < 1:
            print("Fail Detections", len(rows))
            continue

        rows = h5.root.CalibratedSpectra.read()
        if len(rows) < 1:
            print("Fail CalibratedSpectra", len(rows))
            continue

        rows = h5.root.ElixerApertures.read()
        # if len(rows) < 1:
        #     print("Fail ElixerApertures", len(rows))
        #     continue

        rows = h5.root.CatalogMatch.read()
        #this one can be zero
        # if len(rows) < 1:
        #     print(line, "CatalogMatch", len(rows))
        #     continue

        rows = h5.root.SpectraLines.read()
        # if len(rows) < 1:
        #     print(line, "SpectraLines", len(rows))
        #     continue

        rows = h5.root.ExtractedObjects.read()
        # if len(rows) < 1:
        #     print(line, "ExtractedObjects", len(rows))
        #     continue

        rows = h5.root.Aperture.read()
        # if len(rows) < 1:
        #     print(line, "Aperture", len(rows))
        #     continue

        print("OK")
        h5.close()
    except:
        print("Fail", line)

