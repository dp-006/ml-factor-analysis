"""
Accelera Consulting

Custom Sklearn Transformer for Data Type Conversion
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_datatypes"
logger_file_name = "sklearn_wrapper_datatypes.log"
logger = get_logger(logger_name, logger_file_name)


class DataTypeConverter(BaseEstimator, TransformerMixin):
    """
    Custom transformer that converts and filters columns based on their data types.
    
    During fit:
    - Numeric columns (int, float) are marked to be converted to float
    - Boolean columns are marked to be converted to float (0/1)
    - String/object/categorical columns are marked to be kept as object
    - All other columns are marked for dropping
    - Stores the drop_columns_ list for reference
    
    During transform:
    - Converts numeric columns to float
    - Converts boolean columns to float (0/1)
    - Keeps object/categorical columns as object
    - Drops all other column types (datetime, timedelta, etc.)
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    dtype_mapping_ : dict
        Dictionary mapping column names to their target data types ('float' or 'object')
        learned during the fit process.
        Example: {'col1': 'float', 'col2': 'object', ...}
    
    drop_columns_ : list
        List of columns to be dropped (columns that are neither numeric, boolean, string, nor categorical)
        Example: ['col3', 'col4', ...]
    
    dtype_details_ : dict
        Detailed information about each column's data type and transformation action.
        Contains a key for each column with info about original dtype, target dtype, category, and action.
        Also contains '_summary' key with statistics about the transformation.
        Example: {
            'col1': {'original_dtype': 'int64', 'target_dtype': 'float', 'category': 'numeric', 'action': 'convert to float'},
            'col2': {'original_dtype': 'object', 'target_dtype': 'object', 'category': 'string', 'action': 'keep as object'},
            'col3': {'original_dtype': 'datetime64[ns]', 'target_dtype': None, 'category': 'datetime', 'action': 'drop (unsupported type)'},
            '_summary': {'total_columns': 3, 'float_columns': 1, 'object_columns': 1, 'drop_columns': 1, 'retained_columns': 2}
        }
    
    Example
    -------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn_wrapper_datatypes import DataTypeConverter
    >>> import pandas as pd
    >>> 
    >>> X_train = pd.DataFrame({
    ...     'numeric_col': [1, 2, 3],
    ...     'string_col': ['a', 'b', 'c'],
    ...     'datetime_col': pd.date_range('2020-01-01', periods=3)
    ... })
    >>> 
    >>> converter = DataTypeConverter()
    >>> converter.fit(X_train)
    >>> X_transformed = converter.transform(X_train)
    """
    
    def __init__(self):
        self.dtype_mapping_ = {}
        self.drop_columns_ = []
        self.dtype_details_ = {}
    
    def _detect_column_type(self, series):
        """
        Detect the type of a pandas Series and categorize it.
        
        Returns 'float', 'object', or 'drop'
        """
        dtype = series.dtype
        dtype_name = str(dtype).lower()
        
        # ============================================================================
        # NUMERIC TYPES -> CONVERT TO FLOAT
        # ============================================================================
        # Integer types
        if dtype in [np.int8, np.int16, np.int32, np.int64, 
                     np.uint8, np.uint16, np.uint32, np.uint64]:
            return 'float'
        
        # Float types
        if dtype in [np.float16, np.float32, np.float64]:
            return 'float'
        
        # Check for any numeric dtype
        if pd.api.types.is_numeric_dtype(series):
            return 'float'
        
        # ============================================================================
        # STRING/OBJECT TYPES -> KEEP AS OBJECT
        # ============================================================================
        # Object dtype (usually strings and mixed types)
        if pd.api.types.is_object_dtype(series):
            return 'object'
        
        # StringDtype (pandas 1.0+)
        if pd.api.types.is_string_dtype(series):
            return 'object'
        
        # Categorical dtype -> Convert to object
        if pd.api.types.is_categorical_dtype(series):
            return 'object'
        
        # ============================================================================
        # UNSUPPORTED TYPES -> DROP
        # ============================================================================
        # DateTime types
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'drop'
        
        # TimeDelta types
        if pd.api.types.is_timedelta64_dtype(series):
            return 'drop'
        
        # Boolean dtype -> Convert to 0/1 (float)
        if pd.api.types.is_bool_dtype(series):
            return 'float'
        
        # Complex numbers
        if np.issubdtype(dtype, np.complexfloating):
            return 'drop'
        
        # Sparse arrays
        if pd.api.types.is_sparse(series):
            return 'drop'
        
        # Period dtype
        if pd.api.types.is_period_dtype(series):
            return 'drop'
        
        # Interval dtype
        if pd.api.types.is_interval_dtype(series):
            return 'drop'
        
        # Unknown types default to drop
        logger.warning(f"Unknown dtype '{dtype}' detected. Will be dropped.")
        return 'drop'
    
    def fit(self, X, y=None):
        """
        Learn the data types for all columns and identify which columns to drop.
        
        This method analyzes all columns in X, categorizes them, and populates:
        - dtype_mapping_: mapping of column names to target data types
        - drop_columns_: list of columns to be dropped
        - dtype_details_: detailed information about each column including original dtype, target dtype, category, and action taken
        
        During fit, this method inspects each column in X and categorizes it as:
        - 'float': numeric columns to be converted to float (includes boolean -> 0/1)
        - 'object': string/object/categorical columns to be kept as object
        - 'drop': all other types (datetime, timedelta, etc.)

        Mapping:
        Numeric Types (→ float):
            ├── int8, int16, int32, int64
            ├── uint8, uint16, uint32, uint64
            ├── float16, float32, float64
            └── boolean (converted to 0/1)

            String/Object Types (→ object):
            ├── object dtype
            ├── StringDtype
            └── categorical

            Unsupported Types (→ DROP):
            ├── datetime64
            ├── timedelta64
            ├── complex64/128
            ├── sparse arrays
            ├── period & interval dtypes
            └── unknown types
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input training data to learn data types from
        
        y : Ignored
            Not used, present here for API consistency by convention
        
        Returns
        -------
        self : DataTypeConverter
            Fitted transformer
        """
        logger.info("Fitting DataTypeConverter...")
        self.dtype_mapping_ = {}
        self.drop_columns_ = []
        self.dtype_details_ = {}
        
        for col in X.columns:
            original_dtype = str(X[col].dtype)
            col_type = self._detect_column_type(X[col])
            
            if col_type == 'float':
                self.dtype_mapping_[col] = 'float'
                action = 'convert to float'
                logger.info(f"Column '{col}' ({original_dtype}) -> CONVERT TO FLOAT")
                self.dtype_details_[col] = {
                    'original_dtype': original_dtype,
                    'target_dtype': 'float',
                    'category': 'numeric',
                    'action': action
                }
            elif col_type == 'object':
                self.dtype_mapping_[col] = 'object'
                # Determine if it's categorical, string, or object
                if pd.api.types.is_categorical_dtype(X[col]):
                    category = 'categorical'
                elif pd.api.types.is_string_dtype(X[col]):
                    category = 'string'
                else:
                    category = 'object'
                action = 'keep as object'
                logger.info(f"Column '{col}' ({original_dtype}) -> KEEP AS OBJECT")
                self.dtype_details_[col] = {
                    'original_dtype': original_dtype,
                    'target_dtype': 'object',
                    'category': category,
                    'action': action
                }
            elif col_type == 'drop':
                self.drop_columns_.append(col)
                # Determine the type of unsupported column
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    category = 'datetime'
                elif pd.api.types.is_timedelta64_dtype(X[col]):
                    category = 'timedelta'
                elif np.issubdtype(X[col].dtype, np.complexfloating):
                    category = 'complex'
                elif pd.api.types.is_sparse(X[col]):
                    category = 'sparse'
                elif pd.api.types.is_period_dtype(X[col]):
                    category = 'period'
                elif pd.api.types.is_interval_dtype(X[col]):
                    category = 'interval'
                else:
                    category = 'unknown'
                action = 'drop (unsupported type)'
                logger.info(f"Column '{col}' ({original_dtype}) -> DROP ({category})")
                self.dtype_details_[col] = {
                    'original_dtype': original_dtype,
                    'target_dtype': None,
                    'category': category,
                    'action': action
                }
        
        # Create summary statistics
        self.dtype_details_['_summary'] = {
            'total_columns': len(X.columns),
            'float_columns': len([c for c in self.dtype_mapping_ if self.dtype_mapping_[c] == 'float']),
            'object_columns': len([c for c in self.dtype_mapping_ if self.dtype_mapping_[c] == 'object']),
            'drop_columns': len(self.drop_columns_),
            'retained_columns': len(self.dtype_mapping_)
        }

        logger.info(f"Columns to drop: {self.drop_columns_}")
        return self
    
    def transform(self, X):
        """
        Convert columns according to learned mapping and drop unsupported columns.
        
        This method:
        1. Converts numeric columns to float
        2. Converts boolean columns to float (0/1)
        3. Keeps object/categorical columns as object
        4. Drops all other column types
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform
        
        Returns
        -------
        X_transformed : pandas.DataFrame
            Transformed data with:
            - Numeric columns converted to float
            - Boolean columns converted to float (0/1)
            - Object/categorical columns kept as object
            - Unsupported columns dropped
        
        Raises
        ------
        ValueError
            If transform is called before fit, or if mapping is not available
        """
        if not self.dtype_mapping_:
            error_message = "Transformer must be fitted before calling transform. Call fit() first."
            logger.error(error_message)
            raise ValueError(error_message)
        
        X_transformed = X.copy()
        logger.info("Transforming data types according to learned mapping...")
        
        # Step 1: Drop unsupported columns
        columns_to_drop = [col for col in self.drop_columns_ if col in X_transformed.columns]
        if columns_to_drop:
            logger.info(f"Dropping {len(columns_to_drop)} unsupported columns: {columns_to_drop}")
            X_transformed = X_transformed.drop(columns=columns_to_drop)
        
        # Step 2: Convert columns according to mapping
        for col, target_dtype in self.dtype_mapping_.items():
            if col in X_transformed.columns:
                if target_dtype == 'float':
                    # Handle boolean columns separately -> convert to 0/1
                    if pd.api.types.is_bool_dtype(X_transformed[col]):
                        X_transformed[col] = X_transformed[col].astype(int).astype('float')
                        logger.info(f"Converted boolean column '{col}' to 0/1 (float)")
                    else:
                        # Convert to float, coercing errors to NaN
                        X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce').astype('float')
                        logger.info(f"Converted column '{col}' to float")
                elif target_dtype == 'object':
                    # Handle categorical columns separately
                    if pd.api.types.is_categorical_dtype(X_transformed[col]):
                        X_transformed[col] = X_transformed[col].astype('object')
                        logger.info(f"Converted categorical column '{col}' to object")
                    else:
                        # Convert to object
                        X_transformed[col] = X_transformed[col].astype('object')
                        logger.info(f"Converted column '{col}' to object")
            else:
                logger.warning(f"Column '{col}' from mapping not found in input data")
        
        logger.info(f"Data type conversion completed. Shape: {X_transformed.shape}")
        logger.info(f"Total columns before transformation: {len(X.columns)}")
        logger.info(f"Number of columns dropped: {len(columns_to_drop)}")
        logger.info(f"Number of retained columns: {len(X_transformed.columns)}")
        return X_transformed
