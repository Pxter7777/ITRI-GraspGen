"""Custom logging formatter with colored output."""

# Referenced from https://stackoverflow.com/a/56944256
import logging
from typing import ClassVar


class CustomLoggingFormatter(logging.Formatter):
    """Color-coded logging formatter for console output.

    Attributes:
        FORMATS (ClassVar[dict[int, str]]): Mapping of log levels to format strings.

    Example:
        handler = logging.StreamHandler()
        handler.setFormatter(CustomLoggingFormatter())
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
    """

    _grey: str = "\x1b[38;20m"
    _green: str = "\x1b[32;20m"
    _yellow: str = "\x1b[33;20m"
    _red: str = "\x1b[31;20m"
    _bold_red: str = "\x1b[31;1m"
    _reset: str = "\x1b[0m"
    _fmt_str: str = (
        "[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
    )

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: _grey + _fmt_str + _reset,
        logging.INFO: _green + _fmt_str + _reset,
        logging.WARNING: _yellow + _fmt_str + _reset,
        logging.ERROR: _red + _fmt_str + _reset,
        logging.CRITICAL: _bold_red + _fmt_str + _reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color based on severity level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log string with ANSI color codes.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
