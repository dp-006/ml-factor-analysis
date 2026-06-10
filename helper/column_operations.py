'''
Accelera Consulting

Helper functions for column operations, such as renaming columns, dropping columns, and selecting columns.
'''

import pandas as pd

from logging_config.logger_config import get_logger

logger_name = "mlops.column_operations"
logger_file_name = "column_operations.log"
logger = get_logger(logger_name, logger_file_name)

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