
import os
import logging
from logstash import LogstashHandler

logger = logging.getLogger('python-logstash-logger')
logger.setLevel(logging.ERROR)
logger.addHandler(
    LogstashHandler(os.environ['LOGSTASH_HOST'], 514, version=1)
)
