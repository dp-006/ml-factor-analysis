import logging
from datetime import datetime
from pathlib import Path

# Mapping of string log levels to logging constants
LOG_LEVEL_MAPPING = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

def setup_logger(
    name: str = __name__,
    level: str = "info",
    log_to_file: bool = False,
    log_file_path: Path = None,
    log_mode: str = 'a',
    timestamp: str = None,
    runid: str = None
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name (str): Logger name.
        level (str): Log level as string (debug, info, etc.).
        log_to_file (bool): Whether to log into a file.
        log_file_path (Path): Custom log file path.
        log_mode (str): File write mode ('a' for append, 'w' for overwrite).
        timestamp (str): Optional timestamp to include in log file name.
        runid (str): Optional run ID to include in log file name.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger

    # Set log level
    log_level = LOG_LEVEL_MAPPING.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)

    # Define log format
    log_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s"
    )

    # Console handler (logs to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler (logs to file)
    if log_to_file:
        if log_file_path is None:
            # Create logs directory at project root: mlops/{timestamp}/{runid}/logs
            project_root = Path(__file__).parent.parent
            log_dir = project_root / f"mlops/{timestamp}/{runid}/logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Generate log file name with timestamp
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = log_dir / f"log_{date_str}.log"

        # File handler mode options:
        #   'a' = Append mode (preserve previous runs, default)
        #   'w' = Overwrite mode (only keep current run)
        file_handler = logging.FileHandler(
            filename=log_file_path,
            mode=log_mode,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # Prevent logs from propagating to parent loggers
    logger.propagate = False

    return logger

def get_logger(
    name: str = None,
    log_file_name: str = None,
    log_to_file: bool = False,
    log_mode: str = 'a',
    timestamp: str = None,
    runid: str = None
) -> logging.Logger:
    """
    Helper function to quickly get a configured logger.

    Args:
        name (str): Logger name (defaults to module name).
        log_file_name (str): Custom log file name.
        log_to_file (bool): Whether to log to file. Defaults to False.
        log_mode (str): File write mode ('a' for append, 'w' for overwrite). Defaults to 'a'.
        timestamp (str): Optional timestamp to include in log file name.
        runid (str): Optional run ID to include in log file name.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Use module name as logger name if not provided
    if name is None:
        name = __name__

    log_file_path = None

    # Create custom log file path if provided
    if log_file_name:
        # Create mlops/{timestamp}/{runid}/logs structure at project root
        project_root = Path(__file__).parent.parent
        log_dir = project_root / f"mlops/{timestamp}/{runid}/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_file_name

    return setup_logger(name=name, log_to_file=log_to_file, log_file_path=log_file_path, log_mode=log_mode, timestamp=timestamp, runid=runid)


# Example usage
if __name__ == "__main__":
    logger = get_logger("my_app", "my_app.log")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")