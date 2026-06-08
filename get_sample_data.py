"""
Accelera Consulting

Script 1: Data Preparation
Load the credit card clean dataset and save it to the inputs folder as CSV.
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""
import os
import pandas as pd
import json 

# We need it to load the dataset from UCI repository, so we will use ucimlrepo library to fetch the dataset directly. 
# This allows us to get the original data along with its metadata. 
# We can also log the dataset information for better understanding and documentation.
from ucimlrepo import fetch_ucirepo

import logging
from logging_config.logger_config import setup_logger

logger_name = "mlops.utils"

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
            log_mode="w",
            timestamp="test_timestamp",
            runid="test_run",
            propagate=False
        )
except Exception as e:
    # Fallback: if anything fails, create basic logger
    print(f"[ERROR] Logger setup failed for {logger_name}: {e}")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.warning(f"Using fallback logger: {e}")


def save_metadata_to_json(metadata, output_dir, file_name):
    """
    Save metadata dictionary to JSON file.
    
    Parameters
    ----------
    metadata : dict
        Metadata dictionary to save
    output_dir : str
        Output directory to save the JSON file
    file_name : str
        Name of the JSON file
    
    Returns
    -------
    json_path : str
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    json_path = os.path.join(output_dir, file_name)
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"Metadata saved successfully to: {json_path}")
        return json_path
    except Exception as e:
        logger.error(f"Failed to save metadata to JSON: {e}")
        raise

def rename_uci_credit_card_columns(df):
    """
    Rename UCI credit card dataset columns to meaningful names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with original column names (X1, X2, X3, etc.)
    
    Returns
    -------
    df_renamed : pd.DataFrame
        DataFrame with renamed columns
    column_mapping : dict
        Mapping of original to new column names
    """
    # Define column mapping based on UCI dataset documentation
    column_mapping = {
        'Y': 'TARGET',
        'X1': 'credit_limit',
        'X2': 'gender',
        'X3': 'education',
        'X4': 'marital_status',
        'X5': 'age',
        'X6': 'repayment_status_sep_2005',
        'X7': 'repayment_status_aug_2005',
        'X8': 'repayment_status_jul_2005',
        'X9': 'repayment_status_jun_2005',
        'X10': 'repayment_status_may_2005',
        'X11': 'repayment_status_apr_2005',
        'X12': 'bill_amount_sep_2005',
        'X13': 'bill_amount_aug_2005',
        'X14': 'bill_amount_jul_2005',
        'X15': 'bill_amount_jun_2005',
        'X16': 'bill_amount_may_2005',
        'X17': 'bill_amount_apr_2005',
        'X18': 'payment_amount_sep_2005',
        'X19': 'payment_amount_aug_2005',
        'X20': 'payment_amount_jul_2005',
        'X21': 'payment_amount_jun_2005',
        'X22': 'payment_amount_may_2005',
        'X23': 'payment_amount_apr_2005'
    }
    
    # Rename columns
    df_renamed = df.copy()
    
    # Handle both X1-X23 format and any other format
    rename_dict = {}
    for old_col in df_renamed.columns:
        if old_col in column_mapping:
            rename_dict[old_col] = column_mapping[old_col]
    
    df_renamed = df_renamed.rename(columns=rename_dict)
    
    logger.info(f"Renamed {len(rename_dict)} columns")
    logger.info(f"Column mapping:")
    for old_name, new_name in rename_dict.items():
        logger.info(f"  {old_name:6s} -> {new_name}")
    
    logger.info(f"\nNew column names: {list(df_renamed.columns)}")
    
    return df_renamed, column_mapping

def add_column_descriptions(df):
    """
    Add descriptive information for each column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Renamed DataFrame
    
    Returns
    -------
    column_descriptions : dict
        Dictionary with column descriptions
    """
    column_descriptions = {
        'credit_limit': 'Amount of given credit (NT dollar)',
        'gender': 'Gender (1=male, 2=female)',
        'education': 'Education (1=graduate, 2=university, 3=high school, 4=others)',
        'marital_status': 'Marital status (1=married, 2=single, 3=others)',
        'age': 'Age (year)',
        'repayment_status_sep_2005': 'Repayment status in September 2005 (-1=pay duly, 1-9=payment delay)',
        'repayment_status_aug_2005': 'Repayment status in August 2005 (-1=pay duly, 1-9=payment delay)',
        'repayment_status_jul_2005': 'Repayment status in July 2005 (-1=pay duly, 1-9=payment delay)',
        'repayment_status_jun_2005': 'Repayment status in June 2005 (-1=pay duly, 1-9=payment delay)',
        'repayment_status_may_2005': 'Repayment status in May 2005 (-1=pay duly, 1-9=payment delay)',
        'repayment_status_apr_2005': 'Repayment status in April 2005 (-1=pay duly, 1-9=payment delay)',
        'bill_amount_sep_2005': 'Bill statement amount in September 2005 (NT dollar)',
        'bill_amount_aug_2005': 'Bill statement amount in August 2005 (NT dollar)',
        'bill_amount_jul_2005': 'Bill statement amount in July 2005 (NT dollar)',
        'bill_amount_jun_2005': 'Bill statement amount in June 2005 (NT dollar)',
        'bill_amount_may_2005': 'Bill statement amount in May 2005 (NT dollar)',
        'bill_amount_apr_2005': 'Bill statement amount in April 2005 (NT dollar)',
        'payment_amount_sep_2005': 'Previous payment amount in September 2005 (NT dollar)',
        'payment_amount_aug_2005': 'Previous payment amount in August 2005 (NT dollar)',
        'payment_amount_jul_2005': 'Previous payment amount in July 2005 (NT dollar)',
        'payment_amount_jun_2005': 'Previous payment amount in June 2005 (NT dollar)',
        'payment_amount_may_2005': 'Previous payment amount in May 2005 (NT dollar)',
        'payment_amount_apr_2005': 'Previous payment amount in April 2005 (NT dollar)'
    }
    
    # Filter descriptions to only include columns present in df
    available_descriptions = {k: v for k, v in column_descriptions.items() if k in df.columns}
    
    return available_descriptions

def get_column_dtypes():
    """
    Define data types for all columns based on their descriptions.
    
    Returns
    -------
    dtypes_dict : dict
        Dictionary mapping column names to pandas data types
    """
    dtypes_dict = {
        'TARGET': 'int32',  # Binary target variable (0 or 1)
        'credit_limit': 'int32',  # Numerical amount
        'gender': 'object',  # 1=male, 2=female
        'education': 'object',  # 1=graduate, 2=university, 3=high school, 4=others
        'marital_status': 'object',  # 1=married, 2=single, 3=others
        'age': 'int32',  # Age in years
        'repayment_status_sep_2005': 'object',  # -1=pay duly, 1-9=payment delay
        'repayment_status_aug_2005': 'object',
        'repayment_status_jul_2005': 'object',
        'repayment_status_jun_2005': 'object',
        'repayment_status_may_2005': 'object',
        'repayment_status_apr_2005': 'object',
        'bill_amount_sep_2005': 'float32',  # Bill amounts in NT dollar
        'bill_amount_aug_2005': 'float32',
        'bill_amount_jul_2005': 'float32',
        'bill_amount_jun_2005': 'float32',
        'bill_amount_may_2005': 'float32',
        'bill_amount_apr_2005': 'float32',
        'payment_amount_sep_2005': 'float32',  # Payment amounts in NT dollar
        'payment_amount_aug_2005': 'float32',
        'payment_amount_jul_2005': 'float32',
        'payment_amount_jun_2005': 'float32',
        'payment_amount_may_2005': 'float32',
        'payment_amount_apr_2005': 'float32'
    }
    
    return dtypes_dict

def load_uci_credit_card_dataset(dataset_id=350, output_dir='inputs/sample'):
    """
    Load UCI credit card dataset using ucimlrepo library and save to CSV.
    
    Parameters
    ----------
    dataset_id : int
        UCI dataset ID (default=350 for Default of Credit Card Clients)
    output_dir : str
        Output directory to save the CSV file (default='inputs/sample')
    
    Returns
    -------
    df : pd.DataFrame
        Complete dataset with features and target
    csv_path : str
        Path to the saved CSV file
    metadata : dict
        Dataset metadata with column information and data types
    """
    logger.info(f"Fetching UCI dataset with ID: {dataset_id}")
    
    try:
        # Fetch dataset from UCI repository
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Extract data
        X = dataset.data.features
        y = dataset.data.targets
        
        logger.info(f"Dataset loaded successfully")
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Log metadata
        logger.info(f"Dataset metadata: {dataset.metadata}")
        
        # Log variable information
        logger.info(f"Number of variables: {len(dataset.variables)}")
        
        # Combine features and target
        df = X.copy()
        df[y.columns[0]] = y.iloc[:, 0]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Rename columns to meaningful names
        df, column_mapping = rename_uci_credit_card_columns(df)

        # Fix Data Types before saving
        dtypes = get_column_dtypes()
        for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    logger.info(f"Converted column '{col}' to {dtype}")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to {dtype}: {e}")
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'uci_credit_card_dataset.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved successfully to: {csv_path}")
        logger.info(f"Final dataset shape: {df.shape}")
        
        # Get column descriptions
        column_descriptions = add_column_descriptions(df)
        
        # Create metadata focused on data types for CSV reading
        metadata = {
            "filename": "uci_credit_card_dataset.csv",
            "shape": list(df.shape),
            "column_dtypes": dtypes,  # Data types for reading CSV
            "column_descriptions": column_descriptions,
            "columns": list(df.columns),
            "source": {
                "dataset_id": dataset_id,
                "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
            }
        }
        
        # Save metadata as JSON
        metadata_json_path = save_metadata_to_json(
            metadata,
            output_dir,
            'datatypes.json'
        )
        logger.info(f"Data types metadata saved to: {metadata_json_path}")
        
        return df, csv_path, metadata
        
    except Exception as e:
        logger.error(f"Failed to load UCI dataset: {e}")
        raise Exception(f"Failed to load UCI dataset: {e}") 

# Main execution
if __name__ == "__main__":
    df_uci, csv_path_uci, metadata = load_uci_credit_card_dataset(
        dataset_id=350,
        output_dir='inputs/sample'
    )
    

