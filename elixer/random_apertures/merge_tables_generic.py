"""
    input a wildcard enabled query for glob
    vstack all matches
    assumes (requires) tables be the same structure
"""

# combine (stack) the tables
import glob
from astropy.table import Table,vstack
import sys
from tqdm import tqdm

write_every = 100

args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

if "--search" in args:
    i = args.index("--search")
    try:
        search = sys.argv[i + 1]
    except:
        print("bad --search specified")
        exit(-1)
else:
    print("no search specified")
    exit(-1)


if "--out" in args:
    i = args.index("--out")
    try:
        outfn = sys.argv[i + 1]
    except:
        print("bad --out specified")
        exit(-1)
else:
    print("no --out specified")
    exit(-1)

if "--fmt" in args:
    i = args.index("--fmt")
    try:
        fmt = sys.argv[i + 1]
    except:
        print("bad --fmt specified")
        exit(-1)
else:
    fmt = None #try to guess



if "--index" in args:
    i = args.index("--index")
    try:
        set_index = sys.argv[i + 1]
    except:
        print("bad --index specified")
        exit(-1)
else:
    set_index = None #try to guess

table_outname = outfn

files = sorted(glob.glob(search))

if len(files) == 0:
    print(f"No merging to run. 0 files.")
    exit(0)
elif len(files) == 1:
    print(f"No merging to run. 1 file. {files}")
    exit(0)


print(f"Stacking {len(files)} files...")

T = Table.read(files[0], format=fmt)
#todo: this is NOT efficient for large tables ... astropy loads the whole thing then "stacks"
# so you use 2x the memory
# but for my immediate use this is sufficient

for i,f in enumerate(tqdm(files[1:])):
    #print(i+1,f)
    T = vstack([T, Table.read(f, format=fmt)])

    if (i+1) % write_every == 0:
        if T is not None:
            T.write(outfn,format=fmt,overwrite=True)

if set_index is not None:
    T.add_index([set_index])

if T is not None:
    T.write(outfn,format=fmt,overwrite=True)




