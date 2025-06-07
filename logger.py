import logging
from colorlog import ColoredFormatter
from config import folders_and_files

#---------------------------------------------------------------------------------------------------------------------#
#                                                   LOGGER SETUP
#---------------------------------------------------------------------------------------------------------------------#

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname) - s%(reset)s | %(log_color) - s%(message) - s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

streamFile = logging.FileHandler(filename=f"{folders_and_files["log_file"]}", mode="w", encoding="utf-8")
streamFile.setLevel(LOG_LEVEL)

log = logging.getLogger("CV_PROJ_MAIN")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)
log.addHandler(streamFile)