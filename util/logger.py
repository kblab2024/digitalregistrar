import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(
    name: str = "app",
    level: int = logging.INFO,
    log_file: str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    json_format: bool = False,
):
    """
    Create and configure a logger.

    Args:
        name: Logger name (use __name__ usually)
        level: Logging level (e.g. logging.DEBUG)
        log_file: Optional file to write logs
        max_bytes: Max size per log file before rotation
        backup_count: How many rotated files to keep
        json_format: If True, outputs logs as JSON lines
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root handler exists

    if logger.handlers:
        return logger  # already configured

    # --- formatter setup ---
    if json_format:
        import json
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "level": record.levelname,
                    "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "name": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_record["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_record, ensure_ascii=False)
        formatter = JsonFormatter()
    else:
        # colorful console formatter
        COLORS = {
            "DEBUG": "\033[36m",   # Cyan
            "INFO": "\033[32m",    # Green
            "WARNING": "\033[33m", # Yellow
            "ERROR": "\033[31m",   # Red
            "CRITICAL": "\033[41m" # Red background
        }
        RESET = "\033[0m"
        class ColorFormatter(logging.Formatter):
            def format(self, record):
                color = COLORS.get(record.levelname, "")
                msg = super().format(record)
                return f"{color}{msg}{RESET}"
        formatter = ColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S",
        )

    # --- console handler ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    # --- file handler (rotating) ---
    if log_file:
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger
