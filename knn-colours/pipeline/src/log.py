import os
import sys
from pathlib import Path

from loguru import logger

log_dir = Path("data/logs").absolute()
log_dir.mkdir(exist_ok=True)

logger.remove()
logger.add(
    log_dir / "{time}.log",
    format="{time} | {level} | {message} | {extra}",
    level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
)
logger.add(
    sys.stdout,
    format="{message}",
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
)


def get_logger():
    return logger
