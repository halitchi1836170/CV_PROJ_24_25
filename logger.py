import logging
from colorlog import ColoredFormatter
from Globals import folders_and_files,logger_config
import os

#---------------------------------------------------------------------------------------------------------------------#
#                                                   LOGGER SETUP
#---------------------------------------------------------------------------------------------------------------------#

LOG_LEVEL = logger_config["log_level"]
LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname)s - %(funcName)s%(reset)s | %(log_color) - s%(message) - s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)



log = logging.getLogger("CV_PROJ_MAIN")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)