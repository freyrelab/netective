__all__ = []

from logging import getLogger, Formatter, StreamHandler, DEBUG, ERROR, WARNING, CRITICAL, INFO
import logging

class LeveledFormatter(Formatter):
    _formats = {}

    def __init__(self, *args, **kwargs):
        super(LeveledFormatter, self).__init__(*args, **kwargs)

    def set_formatter(self, level, formatter):
        self._formats[level] = formatter

    def format(self, record):
        f = self._formats.get(record.levelno)

        if f is None:
            f = super(LeveledFormatter, self)

        return f.format(record)
    
def get_logger(logger_name):
   
   formatter = LeveledFormatter('??? %(message)s')
   formatter.set_formatter(DEBUG, Formatter('[%(asctime)s] [%(module)s] [%(funcName)s] [%(lineno)d] [%(message)s]', datefmt='%m/%d/%Y %I:%M:%S %p'))
   formatter.set_formatter(INFO, Formatter('[%(levelname)s] %(message)s'))
   formatter.set_formatter(WARNING, Formatter('[%(funcName)s] [%(levelname)s] [%(message)s]'))
   formatter.set_formatter(ERROR, Formatter('[%(levelname)s] [%(message)s]'))
   formatter.set_formatter(CRITICAL, Formatter('[%(module)s] [%(funcName)s] [%(levelname)s] [%(message)s]'))

   handler = StreamHandler()
   handler.setFormatter(formatter)

   logger = getLogger(logger_name)
   logger.setLevel(WARNING)
   logger.addHandler(handler)
   logger.propagate = False

   return logger

def set_log_level(logger= logging.Logger, verbose: str = 'CRITICAL'):
    if isinstance(verbose, str):
        numeric_level = getattr(logging, verbose.upper(), None)
    else:
        numeric_level = verbose
    
    if not isinstance(numeric_level, int):
        logger.critical('Invalid verbose level: %s' % verbose)
    
    logger.setLevel(numeric_level)