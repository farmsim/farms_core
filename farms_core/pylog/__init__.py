"""Log module init including LOGGER and logstr"""

from .log import Logger

LOGGER = Logger()
debug = LOGGER.debug
info = LOGGER.info
warning = LOGGER.warning
error = LOGGER.error
critical = LOGGER.critical
exception = LOGGER.exception


def set_name(name: str):
    """Set logger name"""
    LOGGER.name = name


def set_level(level: str):
    """Set the level logging

    Parameters
    ----------
    level: <str>
        Level of the logging required. ex : level='debug'

    """
    levels = {
        'info': LOGGER.INFO, 'debug': LOGGER.DEBUG,
        'warning': LOGGER.WARNING, 'error': LOGGER.ERROR,
        'critical': LOGGER.CRITICAL}

    _level = levels.get(level, None)

    if _level:
        LOGGER.set_level(_level)
    else:
        error('Invalid logger level')
        raise AttributeError
