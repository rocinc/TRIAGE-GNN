import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file.

    Args:
        name:    Logger name, also used as the log filename stem.
        log_dir: If given, a file handler is added at log_dir/name.log.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers when called multiple times.
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
