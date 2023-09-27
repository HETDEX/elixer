import os
import numpy as np

out = np.loadtxt("elixer.run",dtype=str)

sel = np.full(len(out),False)

for i in range(len(out)):
    if not os.path.exists(os.path.join("./",out[i][1],"copy.done")):
        sel[i] = True

with open("elixer.run.clean","w+") as f:
    for i in range(np.count_nonzero(sel)):
        ln = " ".join(out[sel][i])
        f.write(f"{ln}\n")