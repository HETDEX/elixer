# combine (stack) the tables
from astropy.table import Table, Column
from elixer import spectrum_utilities as SU
from elixer import global_config as G
import sys
import numpy as np

avg = 'biweight'
args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

if "--all" in args:
    i = args.index("--all")
    try:
        all = sys.argv[i + 1]
    except:
        print("bad --all specified")
        exit(-1)
else:
    print("no --all specified")
    exit(-1)

i = all.find("_all.fits")
if i < 0:
    print("invalid '*_all.fits' provided")
    exit(-1)
table_outname = all[0:i]

#could still throw exception if this does not exist, but that is okay .. will see the fail and try again
print(f"Reading {all} ...")
T = Table.read(all,format="fits")

columns = T.columns
out_column = []
out_column_err = []
out_stack = []
out_stacke = []
for col in T.columns:
    if "_stack_" in col: #just the _stack_ and _stack_ct

        err_col = col.replace("stack","stacke")
        print(f"Stacking [{col}] ... ")
        stack, stacke, _, _ = SU.stack_spectra(T[col],
                                                             T[err_col],
                                                             np.tile(G.CALFIB_WAVEGRID, (len(T), 1)),
                                                             grid=G.CALFIB_WAVEGRID,
                                                             avg_type=avg,
                                                             straight_error=False)

        if stack is not None and len(stack) != 0:
            out_column.append(col)
            out_column_err.append(err_col)
            out_stack.append(stack)
            out_stacke.append(stacke)

del T
row = []
type_row = []

for col,err_col,stack,stacke in zip (out_column,out_column_err,out_stack,out_stacke):
    row.append(stack)
    row.append(stacke)
    type_row.append((col,(float,len(stack))))
    type_row.append((err_col,(float,len(stacke))))

#just the one row:
T_default = Table(dtype=type_row)
T_default.add_row(row)
T_default.write(table_outname + "_default.fits",format='fits',overwrite=True)
