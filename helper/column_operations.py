'''
Accelera Consulting

Helper functions for column operations, such as renaming columns, dropping columns, and selecting columns.
'''

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
import numpy as np
from logging_config.logger_config import get_logger

logger_name = "mlops.column_operations"
logger_file_name = "column_operations.log"
logger = get_logger(logger_name, logger_file_name)


# Quality Function for given a dataframe or series
def quality_function_for_pandas(df_or_series):
    '''
    Purpose
    -------
    This function performs quality checks on a given pandas DataFrame or Series and returns a report of the quality issues detected. The checks include:

    1. Missing Values: Counts the number of missing values in each column (for DataFrame) or in the Series.
    2. Infinite Values: Counts the number of infinite values in each column (for DataFrame) or in the Series, but only for numeric columns. For non-numeric columns, it returns "N/A".
    3. Zero Variance: Checks if any column (for DataFrame) or the Series has zero variance (i.e., all values are the same).
    4. Duplicate Columns: Counts the number of duplicate columns in the DataFrame. This check is not applicable for Series, so it will return "N/A".
    5. Special Characters: Counts the number of entries in each column (for DataFrame) or in the Series that contain special characters (non-alphanumeric and non-space characters), but only for object type columns. For non-object columns, it returns "N/A".
    Example of special characters check: 
    If a column contains the value "Hello@World", it would be counted as containing special characters because of the "@" symbol.
    Special Chars: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~

    Parameters
    ----------
    df_or_series : pandas.DataFrame or pandas.Series
        The input DataFrame or Series to be checked for quality issues.

    Raises
    ------
    If the input is not a pandas DataFrame or Series, a ValueError is raised.
    If dataFrame can not passthe quality checks, it will log the quality report for the problematic columns and raise error. 
    If series can not passthe quality checks, it will log the quality report for the series and raise error.

    Returns
    -------
    dict
        A dictionary containing the quality report for each column (for DataFrame) or for the Series, with the following structure:
        {
            "column_name": {
                "missing_values": int,
                "infinite_values": int or "N/A",
                "zero_variance": bool,
                "duplicate_columns": int or "N/A",
                "special_characters": int or "N/A"
            },
            ...
        }
        or for Series:
        {
            "missing_values": int,
            "infinite_values": int or "N/A",
            "zero_variance": bool,
            "special_characters": int or "N/A"
        }

    '''
    if isinstance(df_or_series, pd.DataFrame):
        quality_report = {}
        for column in df_or_series.columns:
            quality_report[column] = {
                "missing_values": df_or_series[column].isnull().sum(),
                "infinite_values": np.isinf(df_or_series[column]).sum() if is_numeric_dtype(df_or_series[column]) else "N/A",
                "zero_variance": df_or_series[column].nunique() == 1,
                "duplicate_columns": df_or_series.columns.duplicated().sum(),
                "special_characters": df_or_series[column].apply(lambda x: isinstance(x, str) and any(not c.isalnum() and not c.isspace() for c in x)).sum() if is_object_dtype(df_or_series[column]) else "N/A"
            }
            #Log only proplematic columns (columns with missing values, infinite values, zero variance, duplicate columns, or special characters)
            if (quality_report[column]["missing_values"] > 0 or
                (isinstance(quality_report[column]["infinite_values"], int) and quality_report[column]["infinite_values"] > 0) or
                quality_report[column]["zero_variance"] or
                (isinstance(quality_report[column]["duplicate_columns"], int) and quality_report[column]["duplicate_columns"] > 0) or
                (isinstance(quality_report[column]["special_characters"], int) and quality_report[column]["special_characters"] > 0)):
                logger.info(f"Quality report for column '{column}': {quality_report[column]}")
        # All columns with no quality issues will be logged as "No quality issues detected in column 'column_name'."
        if not any(quality_report[column]["missing_values"] > 0 or
                   (isinstance(quality_report[column]["infinite_values"], int) and quality_report[column]["infinite_values"] > 0) or
                   quality_report[column]["zero_variance"] or
                   (isinstance(quality_report[column]["duplicate_columns"], int) and quality_report[column]["duplicate_columns"] > 0) or
                   (isinstance(quality_report[column]["special_characters"], int) and quality_report[column]["special_characters"] > 0)
                   for column in df_or_series.columns):
            logger.info("No quality issues detected in any columns.")
        # Raise error if there are quality issues in any column
        if any(quality_report[column]["missing_values"] > 0 or
               (isinstance(quality_report[column]["infinite_values"], int) and quality_report[column]["infinite_values"] > 0) or
               quality_report[column]["zero_variance"] or
               (isinstance(quality_report[column]["duplicate_columns"], int) and quality_report[column]["duplicate_columns"] > 0) or
               (isinstance(quality_report[column]["special_characters"], int) and quality_report[column]["special_characters"] > 0)
               for column in df_or_series.columns):
            raise ValueError("DataFrame contains quality issues. Please check the logs for details.")
        return quality_report
    elif isinstance(df_or_series, pd.Series):
        quality_report = {}
        quality_report = {
            "missing_values": df_or_series.isnull().sum(),
            "infinite_values": np.isinf(df_or_series).sum() if is_numeric_dtype(df_or_series) else "N/A",
            "zero_variance": df_or_series.nunique() == 1,
            "special_characters": df_or_series.apply(lambda x: isinstance(x, str) and any(not c.isalnum() and not c.isspace() for c in x)).sum() if is_object_dtype(df_or_series) else "N/A"
        }
        # Log only if there are quality issues (missing values, infinite values, zero variance, or special characters)
        if (quality_report["missing_values"] > 0 or
            (isinstance(quality_report["infinite_values"], int) and quality_report["infinite_values"] > 0) or
            quality_report["zero_variance"] or
            (isinstance(quality_report["special_characters"], int) and quality_report["special_characters"] > 0)):
            logger.info(f"Quality report for series: {quality_report}")
        else:
            logger.info("No quality issues detected in the series.")
        # Raise error if there are quality issues in the series
        if (quality_report["missing_values"] > 0 or
            (isinstance(quality_report["infinite_values"], int) and quality_report["infinite_values"] > 0) or
            quality_report["zero_variance"] or
            (isinstance(quality_report["special_characters"], int) and quality_report["special_characters"] > 0)):
            raise ValueError("Series contains quality issues. Please check the logs for details.")
        return quality_report
    else:
        raise ValueError("Input must be a pandas DataFrame or Series")

# Convert pandas Interval to string representation
def convert_pd_interval_to_str(
        interval: pd.Interval,
        round_apply: bool = True
        ) -> str:
    """
    Purpose
    -------
    Convert a pandas Interval object to a string representation.

    This function is used to convert the bin labels generated by pd.qcut or pd.cut,
    which are pandas Interval objects, into a more readable string format for reporting.
    What is pd.interval:
    Definition of ( and ] in pandas Interval:
    - ( means the interval does not include the left endpoint (open on the left).
    - ] means the interval includes the right endpoint (closed on the right).
        Therefore,
        [18, 25) means the interval includes the left endpoint but not the right endpoint

    Parameters
    ----------
    interval : pd.Interval
        The pandas Interval object to convert.

    Returns
    -------
    str
        A string representation of the interval in the format "(lower_bound, upper_bound]".
        example: pd.Interval(18, 25, closed="right") will be converted to "from_18_to_25".

    Note
    ----
    Called in:
    - Metadata returned by calculate_woe_iv_table function.
    """
    if not isinstance(interval, pd.Interval):
        logger.error("Input is not a pandas Interval.")
        raise ValueError("Input must be a pandas Interval.")
    
    lower_bound = interval.left
    upper_bound = interval.right
    closed = interval.closed
    logger.info(f"Converting Interval: {interval} with lower_bound={lower_bound}, upper_bound={upper_bound}, closed='{closed}' to string representation.")

    # Get Rounds of lower_bound and upper_bound if round_apply is True, otherwise keep them as they are
    if round_apply:
        lower_bound = int(round(lower_bound))
        upper_bound = int(round(upper_bound))
        logger.info(f"Rounded bounds: lower_bound={lower_bound}, upper_bound={upper_bound}")

    if closed == "right":
        return f"from_{lower_bound}_to_{upper_bound}"
    elif closed == "left":
        return f"from_{lower_bound}_to_{upper_bound}"
    elif closed == "both":
        return f"from_{lower_bound}_to_{upper_bound}"
    # if closed is neither, raise error because it is an invalid state for a pandas Interval
    elif closed == "neither":
        error_message = f"Invalid 'closed' attribute in Interval: {closed}. 'neither' is not a valid state for a pandas Interval."
        logger.error(error_message)
        raise ValueError(error_message)
    else:
        logger.error(f"Invalid 'closed' attribute in Interval: {closed}")
        raise ValueError(f"Invalid 'closed' attribute in Interval: {closed}")

# Function to convert binary columns to object type
def convert_binary_columns(
        df: pd.DataFrame,
        binary_rule: list | None = [0, 1]
        ) -> tuple:
    '''
    Purpose
    -------
    Convert binary columns in a DataFrame to object type. 
    A binary column is defined as a column that contains only two unique values, which can be 0 and 1 or any other pair of values. 
    This function checks each column in the DataFrame to see if it meets this criterion and converts it to object type if it does.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    binary_rule : list or None, optional
        If not None, list of values to consider as binary (default is [0, 1]). The column should have exactly 2 unique values and be a subset of the binary rule set. 
        If None, all columns with exactly 2 unique values are considered binary.

    Returns
    -------
    tuple
        Tuple containing the modified DataFrame and a list of binary column names.
    '''
    # Raise Error if the input is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Raise Error if binary_rule is not a list or None
    if binary_rule is not None and not isinstance(binary_rule, list):
        raise ValueError("binary_rule must be a list or None. Example: [0, 1] or None")
    
    # Raise Error if binary rule is list and does not contain exactly 2 unique values
    if binary_rule is not None and len(set(binary_rule)) != 2:
        raise ValueError("binary_rule must contain exactly 2 unique numeric values. Example: [0, 1]")
    
    # Detect numeric columns in the DataFrame
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    metadata = {}
    for col in numeric_cols:
        # Detect Unique Values and Number of Unique Values in the Column
        unique_values = df[col].dropna().unique()
        number_of_unique_values = len(unique_values)
        logger.info(f"Column '{col}' has {number_of_unique_values} unique values: {unique_values}")
        
        # Check if column is binary
        is_binary = False
        if binary_rule is None:
            # If no rule specified, any column with exactly 2 unique values is binary
            is_binary = number_of_unique_values == 2
        else:
            # Check if column values are subset of binary_rule and has exactly 2 unique values
            is_binary = set(unique_values) <= set(binary_rule) and number_of_unique_values == 2
        
        if is_binary:
            # Append the column name to the list of binary columns
            metadata[col] = unique_values
            # Change the data type of the column to object
            df[col] = df[col].astype('object')
            logger.info(f"Column '{col}' is detected as a binary column and converted to object type.")
    
    logger.info(f"Number of binary columns detected and converted to object type: {len(metadata.keys())}")
    return df, metadata

def detect_missing_columns_with_treshold(df: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """
    Purpose
    -------
    Detect columns in a DataFrame that have a proportion of missing values
    exceeding a specified threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for missing values.

    threshold : float, optional
        Proportion of missing values above which a column is considered to have
        too many missing values. Default ``0.5``.
    
    Returns
    -------
    list[str]
        List of column names that have a proportion of missing values above the
        specified threshold.
    """
    # Raise Error if the input is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    missing_columns = [col for col in df.columns if df[col].isna().mean() > threshold]
    logger.info(f"Number of columns with missing values above threshold {threshold}: {len(missing_columns)}")
    return missing_columns

# Detect Zero Variance Columns
def detect_zero_variance_columns(df: pd.DataFrame) -> list[str]:
    """
    Purpose
    -------
    Detect columns in a DataFrame that have zero variance, meaning all values
    in the column are the same.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for zero variance columns.
    """
    # Raise Error if the input is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    zero_variance_columns = [col for col in df.columns if df[col].nunique() <= 1]
    logger.info(f"Number of zero variance columns detected: {len(zero_variance_columns)}")
    return zero_variance_columns

# Detect Duplicate Columns
def detect_duplicate_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Purpose
    -------
    Detect duplicate columns in a DataFrame, meaning columns that have the same
    values across all rows. Returns a mapping of retained column names to lists
    of duplicate column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for duplicate columns.
    
    Returns
    -------
    dict[str, list[str]]
        Dictionary where keys are retained column names and values are lists
        of duplicate column names. If no duplicates found, returns empty dict.
        Example: {'col1': ['col2', 'col3']} means col2 and col3 are duplicates of col1.
    """
    # Raise Error if the input is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Get all columns that are duplicated (not just the ones marked as duplicates)
    duplicated_mask = df.T.duplicated(keep=False)
    
    if not duplicated_mask.any():
        logger.info("No duplicate columns detected.")
        return {}
    
    # Group duplicate columns
    duplicate_groups = {}
    seen = set()
    
    for col in df.columns:
        if col in seen:
            continue
        
        # Find all columns identical to this one
        identical_cols = [c for c in df.columns if df[col].equals(df[c])]
        
        if len(identical_cols) > 1:
            # First column is retained, rest are duplicates
            retained = identical_cols[0]
            duplicates = identical_cols[1:]
            duplicate_groups[retained] = duplicates
            seen.update(identical_cols)
    
    logger.info(f"Number of duplicate column groups detected: {len(duplicate_groups)}")
    for retained, dups in duplicate_groups.items():
        logger.info(f"Retained: '{retained}' | Duplicates: {dups}")
    
    return duplicate_groups

# Detect Infinite Values in Columns
def detect_infinite_columns(df: pd.DataFrame) -> list[str]:
    """
    Purpose
    -------
    Detect columns in a DataFrame that contain infinite values (np.inf or -np.inf).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for infinite values.

    Returns
    -------
    list[str]
        List of column names that contain infinite values.
    """
    # Raise Error if the input is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Select only numeric columns and check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    infinite_columns = [col for col in numeric_df.columns 
                        if numeric_df[col].apply(np.isinf).any()]
    
    logger.info(f"Number of columns with infinite values detected: {len(infinite_columns)}")
    return infinite_columns

# Detect Numeric Columns with Number of Unique Values Below a Threshold and convert them to object type
def detect_low_cardinality_numeric_columns_to_object(df: pd.DataFrame, threshold: int = 10) -> tuple:
    """
    Purpose
    -------
    Detect numeric columns in a DataFrame that have a number of unique values
    below a specified threshold and convert them to object type.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to check for low cardinality numeric columns.

    threshold : int, optional
        Number of unique values below which a numeric column is considered low cardinality (default is 10).

    Returns
    -------
    dict
        key: column name, value: number of unique values in the column that was converted to object type.
    """
    # Raise Error if the input is not a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    low_cardinality_columns = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        num_unique = df[col].nunique()
        if num_unique <= threshold:
            logger.info(f"Numeric column '{col}' has {num_unique} unique values.")
            low_cardinality_columns[col] = num_unique
            logger.info(f"Column '{col}' is detected as low cardinality and must be converted to object type.")
    
    logger.info(f"Number of low cardinality numeric columns detected and converted: {len(low_cardinality_columns)}")
    return low_cardinality_columns

# Detect Outlier Indicator Columns
def detect_outlier_indicator_columns(
        columns: list[str],
        suffixes: tuple[str, ...] = ('_right', '_left')
        ) -> list[str]:
    """
    Purpose
    -------
    Detect columns in a list of column names that are outlier indicator columns.
    Outlier indicator columns are typically created by Winsorizer or similar transformers
    and are identified by a suffix such as '_right' or '_left'.
    These columns are binary indicators (0/1) and should not be processed by WOE binning.

    Parameters
    ----------
    columns : list[str]
        List of column names to check.

    suffixes : tuple[str, ...], optional
        Tuple of suffixes that identify outlier indicator columns.
        Default is ('_right', '_left').

    Returns
    -------
    list[str]
        List of column names that are detected as outlier indicator columns.
    """
    if not isinstance(columns, list):
        raise ValueError("Input 'columns' must be a list of strings.")

    if not isinstance(suffixes, tuple) or not all(isinstance(s, str) for s in suffixes):
        raise ValueError("Input 'suffixes' must be a tuple of strings. Example: ('_right', '_left')")

    indicator_columns = [col for col in columns if any(col.endswith(suffix) for suffix in suffixes)]
    logger.info(
        f"Outlier indicator columns detected ({len(indicator_columns)}) "
        f"using suffixes {suffixes}: {indicator_columns}"
    )
    return indicator_columns