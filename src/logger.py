"""Shared logger. Import as: from src.logger import logger"""

import logging

from src.config import LOG_FILE

logger = logging.getLogger("listgen")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

stream = logging.StreamHandler()
stream.setFormatter(formatter)
logger.addHandler(stream)

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
file = logging.FileHandler(LOG_FILE, encoding="utf-8")
file.setFormatter(formatter)
logger.addHandler(file)
