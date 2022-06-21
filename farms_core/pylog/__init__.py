"""Python logger for FARMS"""

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


def get_level():
    """Get the current level"""
    levels = {
        LOGGER.DEBUG: 'debug',
        LOGGER.INFO: 'info',
        LOGGER.WARNING: 'warning',
        LOGGER.ERROR: 'error',
        LOGGER.CRITICAL: 'critical',
    }
    return levels.get(LOGGER.level, 'debug')


def set_level(level: str):
    """Set the level logging

    Parameters
    ----------
    level: <str>
        Level of the logging required. ex : level='debug'

    """
    levels = {
        'debug': LOGGER.DEBUG,
        'info': LOGGER.INFO,
        'warning': LOGGER.WARNING,
        'error': LOGGER.ERROR,
        'critical': LOGGER.CRITICAL
    }

    _level = levels.get(level, None)

    if _level:
        LOGGER.set_level(_level)
    else:
        error('Invalid logger level')
        raise AttributeError
