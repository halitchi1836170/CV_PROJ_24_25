from logger import log
from config import *

def print_params(configuration):
    for (key, value) in configuration.items():
        log.info("%s = %s", key, value)

def get_header_title(string, new_line=False):
    str_result=""
    if new_line :
        str_result = int((header_length - len(string)) / 2) * "-" + string + int((header_length - len(string)) / 2) * "-" + "\n"
    else:
        str_result = int((header_length - len(string)) / 2) * "-" + string + int((header_length - len(string)) / 2) * "-"
    return str_result