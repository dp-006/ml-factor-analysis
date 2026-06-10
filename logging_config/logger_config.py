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

# Logger configuration mapping: {logger_name: log_file_name}
LOGGER_CONFIG = {
    "mlops": "mlops.log",
    "mlops.factor_analysis": "factor_analysis.log",
    "mlops.descriptive_analysis": "descriptive_analysis.log",
    "mlops.utils": "utils.log",
    "mlops.io_operations": "io_operations.log",
    "mlops.column_operations": "column_operations.log",
}

def setup_logger(
    name: str = __name__,
    level: str = "info",
    log_to_file: bool = False,
    log_file_path: Path = None,
    log_file_name: str = None,
    log_mode: str = 'a',
    timestamp: str = None,
    runid: str = None,
    propagate: bool = False
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
        propagate (bool): Whether to propagate logs to parent loggers.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    print(f"Setting up logger '{name}' with level '{level.upper()}' and log_to_file={log_to_file}")

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    print(f"Cleared existing handlers for logger '{name}'")

    # Set log level
    log_level = LOG_LEVEL_MAPPING.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)
    print(f"Set log level for logger '{name}' to '{logging.getLevelName(log_level)}'")

    # Define log format
    log_format = logging.Formatter(
        f"%(asctime)s | [{runid}] | %(name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s"
    )
    print(f"Defined log format for logger '{name}'")

    # Console handler (logs to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    print(f"Added console handler for logger '{name}'")

    # File handler (logs to file)
    if log_to_file:
        if log_file_path is None:
            # Create logs directory at project root: mlops/logs
            project_root = Path(__file__).parent.parent
            print(f"Project root directory: {project_root}")
            log_dir = project_root / f"mlops/logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created log directory: {log_dir}")

            if log_file_name is None:
                raise ValueError("log_file_name must be provided if log_to_file is True and log_file_path is not specified.")

            # Generate log file path with log file name
            log_file_path = log_dir / f"{log_file_name}"
            print(f"Generated log file path for logger '{name}': {log_file_path}")

        # File handler mode options:
        #   'a' = Append mode (preserve previous runs, default)
        #   'w' = Overwrite mode (only keep current run)
        file_handler = logging.FileHandler(
            filename=log_file_path,
            mode=log_mode,
            encoding="utf-8"
        )
        print(f"Created file handler for logger '{name}' with mode '{log_mode}'")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        print(f"Added file handler for logger '{name}' with path '{log_file_path}'")
    
    # Set propagate to control whether logs are passed to parent loggers
    logger.propagate = propagate
    print(f"Set propagate for logger '{name}' to {propagate}")

    return logger

def setup_multiple_loggers(
    level: str = "info",
    log_mode: str = 'w',
    timestamp: str = None,
    runid: str = None,
    propagate: bool = False,
    logger_config: dict = None
) -> dict:
    """
    Set up multiple loggers with shared format and level configuration.

    Args:
        level (str): Log level as string (debug, info, etc.). Applied to all loggers.
        log_mode (str): File write mode ('a' for append, 'w' for overwrite). Defaults to 'w'.
        timestamp (str): Optional timestamp to include in log file path.
        runid (str): Optional run ID to include in log file path.
        propagate (bool): Whether to propagate logs to parent loggers. Defaults to False.
        logger_config (dict): Custom logger config mapping {logger_name: log_file_name}.
                             Defaults to LOGGER_CONFIG if not provided.

    Returns:
        dict: Dictionary of configured logger instances {logger_name: logger}.
    """
    if logger_config is None:
        logger_config = LOGGER_CONFIG

    configured_loggers = {}

    for logger_name, log_file_name in logger_config.items():
        print(f"Configuring logger '{logger_name}' with log file '{log_file_name}'")
        # Create log file path
        project_root = Path(__file__).parent.parent
        log_dir = project_root / f"mlops/{timestamp}/{runid}/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / log_file_name
        print(f"Log file path for logger '{logger_name}': {log_file_path}")

        # Setup logger with shared configuration
        logger = setup_logger(
            name=logger_name,
            level=level,
            log_to_file=True,
            log_file_path=log_file_path,
            log_mode=log_mode,
            timestamp=timestamp,
            runid=runid,
            propagate=propagate
        )
        print(f"Configured logger '{logger_name}' with level '{level.upper()}' and log file '{log_file_path}'")
        configured_loggers[logger_name] = logger
    return configured_loggers

def get_logger(logger_name: str, logger_file_name: str) -> logging.Logger:
    '''
    Get a logger instance by name. If the logger is already configured, it will be returned.
    If not, a new logger will be set up with the provided name and file name.

    Args:
        logger_name (str): Name of the logger to retrieve.
        logger_file_name (str): Log file name to use if a new logger needs to be created.
    
    Returns:
        logging.Logger: Logger instance corresponding to the provided name.
    
    Exceptions:
        If logger setup fails, a basic logger will be created and an error message will be printed
        to the console. The fallback logger will log to the console with a warning about using the fallback.

    Note:
        This function assumes that the logger configuration is managed through the logging library's loggerDict. 
        If the logger is not found, it will attempt to set up a new logger with the provided name and file name.    
    '''
    try:
        # Check if logger is already configured in logging library
        if logger_name in logging.Logger.manager.loggerDict:
            # Logger exists, use it
            logger = logging.getLogger(logger_name)
        else:
            # Logger doesn't exist, setup new one with proper configuration
            logger = setup_logger(
                name=logger_name,
                level="info",
                log_to_file=True,
                log_file_path=None,
                log_file_name=logger_file_name,
                log_mode="w",
                propagate=False
            )
    except Exception as e:
        # Fallback: if anything fails, create basic logger
        print(f"[ERROR] Logger setup failed: {e}")
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.warning(f"Using fallback logger: {e}")
    
    return logger