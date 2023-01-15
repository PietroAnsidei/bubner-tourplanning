"""Setup functions."""
import logging
import sys
import time

from src import constants

logging.Formatter.converter = time.gmtime

# Set standard output and standard error for the logs
logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
log_format = "%(asctime)s [%(levelname)s] %(message)s"
formatter = logging.Formatter(log_format)

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.INFO)
h1.addFilter(lambda record: record.levelno <= logging.WARNING)
h1.setFormatter(formatter)

h2 = logging.StreamHandler()
h2.setLevel(logging.ERROR)
h2.setFormatter(formatter)
logger.addHandler(h1)
logger.addHandler(h2)


def setup_params():
    """Set project up."""
    return constants.params
