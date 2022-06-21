"""Python logger for FARMS"""

import sys
import logging

from colorama import Fore


class LogFormatter(logging.Formatter):
    """Project custom logging format"""

    HEADER = "[%(name)s-%(process)d] %(asctime)s - [%(levelname)s]"
    HEADER += """ - %(filename)s::%(lineno)s::%(funcName)s():\n"""
    MESSAGE = "%(message)s\n"
    END = "-"

    COLOR = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def __init__(self, color=False):
        self.color: bool = color
        self._fmt: str = self.HEADER + self.MESSAGE + self.END
        super().__init__(self._fmt)

    def format(self, record):
        if self.color:  # Add color to format based on level
            format_orig = self._get_fmt()
            message = self.HEADER + Fore.RESET + self.MESSAGE
            color = self.COLOR[record.levelno]
            self._set_fmt(color + message + color + self.END + Fore.RESET)
        result = logging.Formatter.format(self, record)
        if self.color:  # Reset format
            self._set_fmt(format_orig)
        return result

    def _get_fmt(self):
        """ Get format """
        return self._style._fmt if hasattr(self, "_style") else self._fmt

    def _set_fmt(self, fmt):
        """ Set format """
        if hasattr(self, "_style"):
            self._style._fmt = fmt
        else:
            self._fmt = fmt


class Logger(logging.Logger):
    """Project custom logger"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __init__(self, name="PYLOG", level=logging.DEBUG, file_path=None):
        super().__init__(name)
        if file_path is None:
            self.fh = None
        else:
            self.fh = self.init_handler(logging.FileHandler(file_path))
        self.ch = self.init_handler(
            logging.StreamHandler(sys.stdout),
            level=level,
            color=True
        )

    def init_handler(self, handling=None, level=logging.DEBUG, color=False):
        """Init logging"""
        handler = handling
        handler.setLevel(level)
        handler.setFormatter(LogFormatter(color=color))
        self.addHandler(handler)
        return handler

    def log2file(self, file_path):
        """Log to a file with with path 'file_path'"""
        self.removeHandler(self.fh)
        self.fh = self.init_handler(logging.FileHandler(file_path))

    def set_level(self, level):
        """Set level function"""
        self.ch.setLevel(level)

    def test(self):
        """Test all logging types"""
        self.info("LOGGING: Testing log messages")
        self.debug("This is a debugging message")
        self.info("This is an informational message")
        self.warning("This is a warning message")
        self.error("This is an error message")
        self.critical("This is a critical message")
        self.info("LOGGING: Testing log messages COMPLETE")


def logstr(*args, **kwargs):
    """Log info with print()-like function arguments"""
    msg = ""
    # Seperation
    sep = kwargs.pop("sep", " ")
    # Get all arguments
    for arg in args:
        msg += str(arg) + sep
    # Remove final seperation of necessary
    if sep:
        msg = msg[:-len(sep)]
    return msg + kwargs.pop("endl", "\n")
