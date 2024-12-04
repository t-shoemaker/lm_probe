import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    logfile: Optional[str | Path] = None,
) -> logging.Logger:
    """Set up a logger with console and optional file output.

    Parameters
    ----------
    name: str, optional
        Logger name (defaults to root logger)
    level: int
        Logging level (defaults to INFO)
    logfile: str or Path, optional
        Optional path to log file

    Returns
    -------
    Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logging

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if logfile:
        log_path = Path(logfile)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
