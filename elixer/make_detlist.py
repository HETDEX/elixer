import numpy as np
import tables
from hetdex_api.config import HDRconfig

cfg = HDRconfig("hdr2.1")

#####################
# Standard Emission
#####################
try:
    h5 = tables.open_file(cfg.detecth5,"r")
    dtb = h5.root.Detections
    alldets = dtb.read(field="detectid")
    #allchi2 = dtb.read(field="chi2")
    #allsn = dtb.read(field="sn")
    mx_prefix = int(max(alldets)/1e5)
    mn_prefix = int(min(alldets)/1e5)

    for i in range(mn_prefix,mx_prefix+1):
        sel = np.where((alldets < (i+1)*1e5) & (alldets >= (i)*1e5))
        #sel = np.where((alldets < (i+1)*1e5) & (alldets >= (i)*1e5) & (allchi2 < 1.3) & (allsn > 5.5))
        np.savetxt(f"dets{i}",alldets[sel],fmt="%d")
        print(f"dets{i} : entries = {len(sel[0])}")
    h5.close()
except Exception as e:
    print(e)


#####################
# Broadline
#####################
try:
    h5 = tables.open_file(cfg.detectbroadh5,"r")
    dtb = h5.root.Detections
    alldets = dtb.read(field="detectid")
    mx_prefix = int(max(alldets)/1e5)
    mn_prefix = int(min(alldets)/1e5)

    for i in range(mn_prefix,mx_prefix+1):
        sel = np.where((alldets < (i+1)*1e5) & (alldets >= (i)*1e5))
        np.savetxt(f"dets{i}",alldets[sel],fmt="%d")
        print(f"dets{i} : entries = {len(sel[0])}")
    h5.close()
except Exception as e:
    print(e)


#####################
# Continuum
#####################
try:
    h5 = tables.open_file(cfg.contsourceh5,"r")
    dtb = h5.root.Detections
    alldets = dtb.read(field="detectid")
    mx_prefix = int(max(alldets)/1e5)
    mn_prefix = int(min(alldets)/1e5)

    for i in range(mn_prefix,mx_prefix+1):
        sel = np.where((alldets < (i+1)*1e5) & (alldets >= (i)*1e5))
        np.savetxt(f"dets{i}",alldets[sel],fmt="%d")
        print(f"dets{i} : entries = {len(sel[0])}")
    h5.close()
except Exception as e:
    print(e)

