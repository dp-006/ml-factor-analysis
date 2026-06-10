'''
Accelera Consulting

IO operations helper functions.
'''

import os
import json
import pandas as pd
import numpy as np

from logging_config.logger_config import get_logger

logger_name = "mlops.io_operations"
logger_file_name = "io_operations.log"
logger = get_logger(logger_name, logger_file_name)

# Custom JSON encoder to handle pandas objects
class CustomJSONEncoder(json.JSONEncoder):
    '''
    Purpose
    -------
    Custom JSON encoder to handle pandas objects.
    '''
    def default(self, obj):
        '''
        Purpose
        -------
        Override the default method to convert pandas objects to JSON serializable formats.

        Parameters
        ----------
        obj : any
            The object to be converted to a JSON serializable format.
        
        Returns
        -------
        any
            The JSON serializable format of the input object.
        
        Examples for each type of pandas object:
        1. pd.Index: pd.Index([1, 2, 3]) -> [1, 2, 3]
        2. pd.Series: pd.Series([1, 2, 3]) -> [1, 2, 3]
        3. np.ndarray: np.array([1, 2, 3]) -> [1, 2, 3]
        4. np.integer: np.int64(1) -> 1
        5. np.floating: np.float64(1.0) -> 1
        6. np.bool_: np.bool_(True) -> True
        7. Other types: The default JSON encoder will handle other types as usual.
        '''
        if isinstance(obj, (pd.Index, pd.Series)):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Save JSON file to the given path.
def io_save_json(data:str, path:str, indent:int=4) -> str:
    '''
    Purpose
    -------
    Save JSON file to the given path.

    Parameters
    ----------
    data : str
        The data to be saved in JSON format.
    path : str
        The path where the JSON file will be saved.
        Use relative paths (e.g., "outputs/report.json") for saving within the project.
        Avoid absolute paths with leading "/" on Windows as they resolve to the drive root (C:\).
    
    Returns
    -------
    str
        The path where the JSON file was saved.
    
    Notes
    -----
    Path Formatting Guidelines:
    - Relative path (RECOMMENDED): "outputs/report.json" or "./outputs/report.json" 
      Saves relative to current working directory (best for projects)
    - Dynamic path (RECOMMENDED): os.path.join(os.getcwd(), "outputs", "report.json") 
      Flexible and cross-platform compatible
    - AVOID: "/outputs/report.json" 
      On Windows, resolves to C:\outputs\ (drive root), not the project directory
    '''
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    logger.info(f"Ensuring directory exists: {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory created: {directory}")
    try:
        logger.info(f"Saving JSON file to {path}")
        with open(path, 'w') as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=indent)
        logger.info(f"JSON file saved successfully to {path}")
        return path
    except Exception as e:
        error_message = f"Error saving JSON file to {path}: {e}"
        logger.error(error_message)
        raise Exception(error_message)

# Load JSON file from the given path.
def io_load_json(path:str):
    '''
    Purpose
    -------
    Load JSON file from the given path.

    Parameters
    ----------
    path : str
        The path to the JSON file to be loaded.

    Returns
    -------
    dict
        The data loaded from the JSON file.
    '''
    # Check if the file exists
    logger.info(f"Checking if file exists: {path}")
    if not os.path.exists(path):
        error_message = f"File not found: {path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)
    try:
        logger.info(f"Loading JSON file from {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON file loaded successfully from {path}")
        return data
    except Exception as e:
        error_message = f"Error loading JSON file from {path}: {e}"
        logger.error(error_message)
        raise Exception(error_message)

# Read CSV file from the given path and return a DataFrame.
def io_read_csv_as_df(path:str, dtype:dict=None):
    '''
    Purpose
    -------
    Read CSV file from the given path and return a DataFrame.

    Parameters
    ----------
    path : str
        The path to the CSV file to be read.
    dtype : dict, optional
        A dictionary specifying the data types for the columns in the CSV file (default is None).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data from the CSV file.
    '''
    # Check if the file exists
    logger.info(f"Checking if file exists: {path}")
    if not os.path.exists(path):
        error_message = f"File not found: {path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)
    try:
        logger.info(f"Reading CSV file from {path} with dtype={dtype}")
        df = pd.read_csv(path, dtype=dtype)
        logger.info(f"CSV file read successfully from {path}")
        # Log the shape of the DataFrame
        logger.info(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
        return df
    except Exception as e:
        error_message = f"Error reading CSV file from {path}: {e}"
        logger.error(error_message)
        raise Exception(error_message)

# Check Quality of the DataFrame
def io_check_dataframe_quality(
        df:pd.DataFrame,
        check_categorical:bool=True,
        check_missing:bool=True,
        check_inf:bool=True,
        check_special_characters:bool=True,
        check_constant:bool=True,
        check_high_cardinality:bool=True,
        check_numeric_like_object:bool=True,
        check_zero_variance:bool=True,
        check_high_corr:bool=True,
        check_empty_strings:bool=True,
        check_duplicates:bool=True
        ) -> dict:
    '''
    Purpose
    -------
    Check Quality of the DataFrame.

    Check Rules:
    1: Checking for Ctegroical and String Columns.
    2: Checking for Missing Values.
    3: Checking for Inf Values.
    4: Checking for Special Characters in Object Columns.
    5: Checking for Constant Columns.
    6: Checking for High Cardinality Columns.
    7: Checking for Numeric Columns stored as Object.
    8: Checking for Numeric Columns with Zero Variance.
    9: Checking for Highly Correlated Numeric Columns.
    10: Checking for Empty Strings in Object Columns.
    11: Checking for Duplicate Rows in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked for quality.
    
    check_categorical : bool, optional
        Whether to check for categorical and string columns (default is True).

    check_missing : bool, optional
        Whether to check for missing values (default is True).

    check_inf : bool, optional
        Whether to check for Inf values (default is True).

    check_special_characters : bool, optional
        Whether to check for special characters in object columns (default is True).

    check_constant : bool, optional
        Whether to check for constant columns (default is True).

    check_high_cardinality : bool, optional
        Whether to check for high cardinality columns (default is True).

    check_numeric_like_object : bool, optional
        Whether to check for numeric columns stored as object (default is True).

    check_zero_variance : bool, optional
        Whether to check for numeric columns with zero variance (default is True).

    check_high_corr : bool, optional
        Whether to check for highly correlated numeric columns (default is True).

    check_empty_strings : bool, optional
        Whether to check for empty strings in object columns (default is True).

    check_duplicates : bool, optional
        Whether to check for duplicate rows in the DataFrame (default is True).

    Returns
    -------
    dict
        A dictionary containing the quality metrics of the DataFrame.
    '''
    try:
        logger.info("Checking quality of the DataFrame")
        # Number of rows and columns in the DataFrame
        n_rows, n_columns = df.shape
        logger.info(f"Number of rows: {n_rows}, Number of columns: {n_columns}")
        
        # Raise an error if the DataFrame is empty
        if n_rows == 0:
            error_message = "DataFrame is empty."
            logger.error(error_message)
            raise ValueError(error_message)

        # Detect Number of Object, String and Categorical Columns in the DataFrame and log the names of such columns
        if check_categorical:
            categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
            object_columns = df.select_dtypes(include=['object']).columns.tolist()
            string_columns = df.select_dtypes(include=['string']).columns.tolist()
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Object columns: {object_columns}")
            logger.info(f"String columns: {string_columns}")
            # Log a warning if there are any string or categorical columns in the DataFrame, as the feature engine library accepts only numeric and object columns
            if categorical_columns or string_columns:
                warning_message = "Feature engine library accepts ONLY numeric and object columns. " \
                    f"Detected: {len(categorical_columns)} categorical, " \
                    f"{len(object_columns)} object, {len(string_columns)} string."
                logger.warning(warning_message)
            else:
                logger.info("No categorical or string columns types detected in the DataFrame.")

        # Detect Number of Missing Values in Each Column and log name, percentage of missing values for each column with missing values
        if check_missing:
            missing_values = {}
            for column in df.columns:
                count = df[column].isna().sum()
                if count > 0:
                    missing_values[column] = count / n_rows * 100
            if missing_values:
                logger.warning("Missing values detected in columns:")
                for column, percentage in missing_values.items():
                    logger.warning(f"{column}: {percentage:.2f}% missing values")
            else:
                logger.info("No missing values detected in any column.")

        # Detect -Inf or +Inf values in the DataFrame and log column names and percentage of such values for each column with Inf values
        if check_inf:
            inf_values = {}
            for column in df.columns:
                count = df[column].isin([float('inf'), float('-inf')]).sum()
                if count > 0:
                    inf_values[column] = count / n_rows * 100
            if inf_values:
                logger.warning("Inf values detected in columns:")
                for column, percentage in inf_values.items():
                    logger.warning(f"{column}: {percentage:.2f}% Inf values")
            else:
                logger.info("No Inf values detected in any column.")

        # Detect any special characters in the DataFrame Object columns and log only columns with such characters
        if check_special_characters:
            special_characters = {}
            for column in df.select_dtypes(include=['object', 'string', 'category']).columns:
                count = df[column].str.contains(r'[^\w\s]', regex=True).sum()
                if count > 0:
                    special_characters[column] = count / n_rows * 100
            if special_characters:
                logger.warning("Special characters detected in columns:")
                for column, percentage in special_characters.items():
                    logger.warning(f"{column}: {percentage:.2f}% values with special characters")
            else:
                logger.info("No special characters detected in any column.")
        
        # Detect Constant Columns in the DataFrame and log the names of such columns
        if check_constant:
            constant_columns = df.columns[df.nunique() <= 1].tolist()
            if constant_columns:
                logger.warning("Constant columns detected:")
                for column in constant_columns:
                    logger.warning(f"{column} is a constant column")
            else:
                logger.info("No constant columns detected in the DataFrame.")
        
        # Detect High Cardinality Columns in the DataFrame with more than 50 unique values and log names and number of unique values of such columns
        if check_high_cardinality:
            high_cardinality_columns = {}
            object_columns_for_cardinality = df.select_dtypes(include=['object', 'string', 'category']).columns
            for column in object_columns_for_cardinality:
                unique_values = df[column].nunique()
                if unique_values > 50:  # Threshold for high cardinality can be adjusted as needed
                    high_cardinality_columns[column] = unique_values
            if high_cardinality_columns:
                logger.warning("High cardinality columns detected:")
                for column, unique_count in high_cardinality_columns.items():
                    logger.warning(f"{column}: {unique_count} unique values")
            else:
                logger.info("No high cardinality columns detected in the DataFrame.")
        
        # Numeric Columns stored as Object: Detect numeric columns stored as object and log the names of such columns
        if check_numeric_like_object:
            numeric_like_object_cols = {}
            for col in df.select_dtypes(include=["object"]):
                ratio = pd.to_numeric(
                    df[col],
                    errors="coerce"
                    ).notna().mean()
                numeric_like_object_cols[col] = ratio
            if numeric_like_object_cols:
                logger.warning("Numeric-like object columns detected:")
                for column, ratio in numeric_like_object_cols.items():
                    if ratio > 0.5:  # Threshold can be adjusted as needed
                        logger.warning(f"{column}: {ratio:.2%} numeric-like values")
            else:
                logger.info("No numeric-like object columns detected in the DataFrame.")
        
        # Detect Numeric Columns with Zero Variance and log the names of such columns
        if check_zero_variance:
            zero_variance_columns = {}
            for col in df.select_dtypes(include=["number"]):
                if df[col].var() == 0:
                    zero_variance_columns[col] = df[col].nunique()
            if zero_variance_columns:
                logger.warning("Numeric columns with zero variance detected:")
                for column in zero_variance_columns:
                    logger.warning(f"{column} is a zero variance column")
            else:
                logger.info("No numeric columns with zero variance detected in the DataFrame.")
        
        # Detect Highly Correlated Numeric Columns: Detect pairs of numeric columns with a correlation coefficient greater than 0.9 and log the names of such columns
        if check_high_corr:
            high_corr_pairs = []
            numeric_cols = df.select_dtypes(include=["number"]).columns
            corr_matrix = df[numeric_cols].corr().abs()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > 0.9:  # Threshold can be adjusted as needed
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            if high_corr_pairs:
                logger.warning("Highly correlated numeric columns detected:")
                for col1, col2, corr_value in high_corr_pairs:
                    logger.warning(f"{col1} and {col2} have a correlation of {corr_value:.2f}")
            else:
                logger.info("No highly correlated numeric columns detected in the DataFrame.")
        
        # Detect Empty Strings in Object Columns: Detect object columns with empty strings and log the names of such columns along with the count of empty strings
        if check_empty_strings:
            empty_strings = {}
            for col in df.select_dtypes(include=["object","string"]):
                count = (df[col].astype(str).str.strip() == "").sum()
                if count > 0:
                    empty_strings[col] = count
            if empty_strings:
                logger.warning("Object columns with empty strings detected:")
                for column, count in empty_strings.items():
                    logger.warning(f"{column}: {count} empty strings")
            else:
                logger.info("No object columns with empty strings detected in the DataFrame.")
        
        # Detect duplicate rows in the DataFrame and log the number of duplicate rows
        if check_duplicates:
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                logger.warning(f"Duplicate rows detected: {duplicate_rows} duplicate rows")
            else:
                logger.info("No duplicate rows detected in the DataFrame.")

        # Prepare the quality metrics dictionary to be returned
        quality_metrics = {
            "numRows": n_rows,
            "numColumns": n_columns,
            "categoricalColumns": categorical_columns if check_categorical else 'not checked',
            "objectColumns": object_columns if check_categorical else 'not checked',
            "stringColumns": string_columns if check_categorical else 'not checked',
            "missingValues": missing_values if check_missing else 'not checked',
            "infValues": inf_values if check_inf else 'not checked',
            "specialCharacters": special_characters if check_special_characters else 'not checked',
            "constantColumns": constant_columns if check_constant else 'not checked',
            "highCardinalityColumns": high_cardinality_columns if check_high_cardinality else 'not checked',
            "numericLikeObjectCols": numeric_like_object_cols if check_numeric_like_object else 'not checked',
            "zeroVarianceColumns": zero_variance_columns if check_zero_variance else 'not checked',
            "highCorrPairs": high_corr_pairs if check_high_corr else 'not checked',
            "emptyStrings": empty_strings if check_empty_strings else 'not checked',
            "duplicateRows": int(duplicate_rows) if check_duplicates else 'not checked'
            }

        logger.info(f"DataFrame quality metrics: {quality_metrics}")
        return quality_metrics
    except Exception as e:
        error_message = f"Error checking quality of the DataFrame: {e}"
        logger.error(error_message)
        raise Exception(error_message) from e