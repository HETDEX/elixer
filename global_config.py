from __future__ import print_function
import logging
import gc

#catalogs are defined at top of catalogs.py

#CONFIG_BASEDIR = "/work/03946/hetdex/maverick/"
#CANDELS_EGS_Stefanon_2016_BASE_PATH = "/work/03564/stevenf/maverick/EGS"

CONFIG_BASEDIR = "/home/dustin/code/python/voltron/data/config/"
CANDELS_EGS_Stefanon_2016_BASE_PATH = "/home/dustin/code/python/voltron/data/EGS"

LOG_FILENAME = "voltron.log"
LOG_LEVEL = logging.DEBUG

logging.basicConfig(filename=LOG_FILENAME,level=LOG_LEVEL,filemode='w')
#.debug(), .info(), .warning(), .error(), .critical()

LyA_rest = 1216. #A 1215.668 and 1215.674
OII_rest = 3727.

Fiber_Radius = 0.75 #arcsec