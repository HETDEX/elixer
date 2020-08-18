#get the list of shot files to check and divide up into a SLURM-style run file
#(modify an elixer.slurm file to point to dsd.run and use all the cores on a node)


import glob
import os.path as op
shot_path = "/data/05350/ecooper/hdr2.1/reduction/data/20*.h5"


shot_files = glob.glob(shot_path)

with open("dsd.run","w") as f:
    for s in shot_files:
        shot = op.basename(s).rstrip(".h5")
        f.write(f"mkdir {shot}; cd {shot}; python3 ../dsd_part2.py {s}\n")

