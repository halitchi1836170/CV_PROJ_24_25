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

FILE_LOGFORMAT = "%(asctime)s - %(levelname)s - %(funcName)s | %(message)s"
file_formatter  = logging.Formatter(FILE_LOGFORMAT)

generic_log_folder = folders_and_files["log_folder"]
genericStreamFile = logging.FileHandler(filename=f"{generic_log_folder}/{'training_'+folders_and_files['log_file']}", mode="w", encoding="utf-8")
genericStreamFile.setLevel(logging.DEBUG)
genericStreamFile.setFormatter(file_formatter)

log = logging.getLogger("CV_PROJ_MAIN")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)
log.addHandler(genericStreamFile)