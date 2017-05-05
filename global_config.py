import logging

LOG_FILENAME = "voltron.log"
LOG_LEVEL = logging.DEBUG

logging.basicConfig(filename=LOG_FILENAME,level=LOG_LEVEL,filemode='w')
#.debug(), .info(), .warning(), .error(), .critical()