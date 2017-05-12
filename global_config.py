from __future__ import print_function
import logging

#catalogs are defined at top of catalogs.py

LOG_FILENAME = "voltron.log"
LOG_LEVEL = logging.DEBUG

logging.basicConfig(filename=LOG_FILENAME,level=LOG_LEVEL,filemode='w')
#.debug(), .info(), .warning(), .error(), .critical()