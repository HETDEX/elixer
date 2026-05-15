"""
DD 20260513
heavily based on Shiro Mukae's model_fitting_hdr543.py
but modified to act on a single detection and keep it in memory, rather than iterate over many detections from flat files

This is a test case ...

"""


import model_fitting_config as CNN
#from model_fitting_config import *
# import elixer.cnn.model_fitting_hdr543
# from elixer.cnn.model.TDSA_LeakyGAP_logit import TDSA_LeakyGAP_logit
# import elixer.cnn.model_fitting_config

import tables


h5 = tables.open_file("test_cat.h5")

# #read the 2D cutout we want
# qdet = 3004650406
# rows = h5.root.Fiber2DCutouts.read_where("detectid==qdet")
#
# specs = rows['img_sum']
# dets = rows['detectid']
# label = -1
#
# cnn_t = process_detections(specs, dets)
# #dataset = SpectraDataset(specs, dets, label)
# print("*** DONE ***")
# print(len(cnn_t))
# print(cnn_t)
#

#read the 2D cutout we want
#qdet = 3004650406
rows = h5.root.Fiber2DCutouts.read()

specs = rows['img_sum']
dets = rows['detectid']
label = -1

cnn_t = CNN.process_detections(specs, dets)
#dataset = SpectraDataset(specs, dets, label)
print("*** DONE ***")
print(len(cnn_t))
print(cnn_t)

for t in cnn_t:
    print(f"{t['detectid']} {t['CNN_Score_2D_Spectra']}")

h5.close()


