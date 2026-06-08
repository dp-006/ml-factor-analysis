'''
Accelera Consulting - 2026

Main module for the ML. 
This module initializes the logger and sets up the main execution flow for the project.

'''
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
import uuid
from logging_config.logger_config import setup_multiple_loggers

timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S")
runid = uuid.uuid4().hex[:8]

# Set up loggers for different modules with consistent configuration
loggers = setup_multiple_loggers(
    level="info",
    log_mode="w",
    timestamp=timestamp,
    runid=runid,
    propagate=False
)

logger = loggers["mlops"]  # Use the setup logger from dict, not logging.getLogger() 

# Log Possible Loggers for Factor Analysis Module
logger.info("Available loggers for factor analysis module:")
for logger_name in loggers.keys():
    logger.info(f"Logger Name: {logger_name}") 
    logger.info(f"Logger Instance: {loggers[logger_name]}")
    logger.info(f"Logger Handlers: {loggers[logger_name].handlers}")
    logger.info(f"Logger Level: {logging.getLevelName(loggers[logger_name].level)}")
    logger.info(f"Logger Propagate: {loggers[logger_name].propagate}")
    log_filepath = None
    for handler in loggers[logger_name].handlers:
        if isinstance(handler, logging.FileHandler):
            log_filepath = handler.baseFilename
            break
    logger.info(
        "\n"
        "Log Folder Structure\n"
        "mlops\n"
        f"└── {timestamp}\n"
        f"    └── {runid}\n"
        "        └── logs\n"
        f"            └── {Path(log_filepath).name if log_filepath else 'N/A'}"
    )
    logger.info("-" * 50)  # Separator for readability

from factor_analysis import run_factor_analysis

# Get Sampe Data
# Get Metadata as json
metadata_path = "inputs/sample/datatypes.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)
column_dtypes = metadata.get("column_dtypes", {})
# Get the sample data
input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
df = pd.read_csv(input_csv_path, dtype=column_dtypes)
# Log Column names and data types
for column in df.columns:
    logger.info(f"Column: {column}: {column_dtypes.get(column, 'Unknown')} | Unique Values: {df[column].nunique()}")
logger.info("Sample data loaded successfully.")

# Get Factor Analysis
factor_analysis_results = run_factor_analysis(
    df=df,
    target_variable="TARGET",
    drop_last=True,
    fill_strategy_numeric="mean",
    encoding_strategy_categorical="ordinal",
    rotation="varimax",
    eigenvalue_threshold=1.0,
    loading_threshold=0.50,
    output_dir="outputs/factor_analysis"
    )
