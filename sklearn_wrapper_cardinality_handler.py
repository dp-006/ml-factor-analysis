"""
Accelera Consulting

sklearn custom transformer for handling low cardinality numeric columns.

This module provides a sklearn-compatible transformer that detects and converts
numeric columns with low cardinality (few unique values) to object type.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from helper.column_operations import detect_low_cardinality_numeric_columns_to_object, detect_outlier_indicator_columns
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_cardinality_handler"
logger_file_name = "sklearn_wrapper_cardinality_handler.log"
logger = get_logger(logger_name, logger_file_name)


class LowCardinalityHandler(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn custom transformer that detects and handles numeric columns with 
    low cardinality (few unique values).
    
    This transformer wraps the detect_low_cardinality_numeric_columns_to_object 
    function from the column_operations module to provide sklearn-compatible 
    pipeline functionality. Low cardinality numeric columns are converted to 
    object type as they are better treated as categorical features.
    
    Attributes
    ----------
    threshold : int, optional
        Number of unique values below which a numeric column is considered low 
        cardinality (default is 10). Columns with unique values <= threshold will 
        be converted to object type.
    
    low_cardinality_columns_ : dict
        Mapping of low cardinality column names to their unique value counts 
        (learned during fit). Format: {col_name: unique_count}
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn_wrapper_cardinality_handler import LowCardinalityHandler
    >>> 
    >>> # Create sample data with low cardinality numeric columns
    >>> df = pd.DataFrame({
    ...     'age': [20, 25, 30, 35, 40],
    ...     'status': [1, 1, 2, 2, 1],  # Only 2 unique values
    ...     'rating': [1, 2, 3, 4, 5, 1, 2, 3]  # Only 5 unique values (cardinality=8)
    ... })
    >>> 
    >>> # Create and fit transformer with threshold=10
    >>> handler = LowCardinalityHandler(threshold=10, add_suffix=True, suffix='s')
    >>> handler.fit(df)
    >>> 
    >>> # Transform data - 'status' and 'rating' will be converted to object with suffix
    >>> df_transformed = handler.transform(df)
    >>> print(df_transformed.dtypes)  # status and rating will be 'object'
    """
    
    def __init__(self, threshold: int = 10, add_suffix: bool = False, suffix: str = "s"):
        """
        Initialize the LowCardinalityHandler transformer.
        
        Parameters
        ----------
        threshold : int, optional
            Number of unique values below which a numeric column is considered 
            low cardinality. Default is 10. Must be >= 1.
        
        add_suffix : bool, optional
            If True, adds a suffix to each value in converted columns. This can help 
            avoid issues with WOE transformation. Default is False.
        
        suffix : str, optional
            The suffix string to add to each value when add_suffix=True. Default is "s".
            Example: with suffix="s", value 1 becomes "1s".
        
        Raises
        ------
        ValueError
            If threshold is not a positive integer, or if add_suffix is not a boolean, or if suffix is not a string.
        """
        if not isinstance(threshold, int):
            raise ValueError("threshold must be an integer.")
        if threshold < 1:
            raise ValueError("threshold must be >= 1.")
        if not isinstance(add_suffix, bool):
            raise ValueError("add_suffix must be a boolean.")
        if not isinstance(suffix, str):
            raise ValueError("suffix must be a string.")
        
        self.threshold = threshold
        self.add_suffix = add_suffix
        self.suffix = suffix
        self.low_cardinality_columns_ = None
        logger.info(f"LowCardinalityHandler initialized with threshold={threshold}, add_suffix={add_suffix}, suffix='{suffix}'")
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by detecting low cardinality numeric columns.
        
        Purpose
        -------
        Learns which numeric columns have a number of unique values below or equal 
        to the specified threshold by analyzing the input DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to analyze for low cardinality numeric columns.
        y : None, optional
            Target variable (not used, present for sklearn API consistency).
        
        Returns
        -------
        self
            Returns self for method chaining in sklearn pipelines.
        
        Raises
        ------
        ValueError
            If input X is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        
        logger.info(f"Fitting LowCardinalityHandler on DataFrame with shape {X.shape}")
        
        # Detect low cardinality columns using the helper function
        self.low_cardinality_columns_ = detect_low_cardinality_numeric_columns_to_object(
            X, self.threshold
        )

        # Exclude outlier indicator columns (_right, _left) — these are binary flags
        # created by Winsorizer and must not be converted to object type.
        indicator_cols = detect_outlier_indicator_columns(list(self.low_cardinality_columns_.keys()))
        for col in indicator_cols:
            del self.low_cardinality_columns_[col]
        if indicator_cols:
            logger.info(
                f"Excluded {len(indicator_cols)} outlier indicator columns from low cardinality "
                f"conversion: {indicator_cols}"
            )

        logger.info(
            f"Fit complete. Detected {len(self.low_cardinality_columns_)} low cardinality "
            f"numeric columns (unique values <= {self.threshold}): "
            f"{dict(self.low_cardinality_columns_)}"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by handling low cardinality numeric columns.
        
        Purpose
        -------
        Applies the learned transformation by either converting detected low 
        cardinality columns to object type or keeping them as numeric based on 
        the action parameter set during initialization. Optionally adds a suffix 
        to each value to help avoid issues with WOE transformation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
            - If action='convert': Numeric columns with unique values <= threshold 
              (as detected during fit) are converted to object type. If add_suffix=True, 
              each value will have the suffix appended.
            - If action='keep': Returns original DataFrame with all numeric columns 
              unchanged.
        
        Raises
        ------
        ValueError
            If input X is not a pandas DataFrame, or if transformer has not been 
            fitted yet.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        
        if self.low_cardinality_columns_ is None:
            raise ValueError(
                "Transformer has not been fitted yet. Call fit() first before "
                "calling transform()."
            )
        
        logger.info(f"Transforming DataFrame with shape {X.shape}")
        X_transformed = X.copy()
        
        # Get columns that exist in both X and low_cardinality_columns_
        cols_to_convert = [
            col for col in self.low_cardinality_columns_.keys() 
            if col in X_transformed.columns
        ]
        
        if cols_to_convert:
            logger.info(
                f"Converting {len(cols_to_convert)} low cardinality columns to object type: "
                f"{cols_to_convert}"
            )
            for col in cols_to_convert:
                X_transformed[col] = X_transformed[col].astype('object')
                logger.info(
                    f"Converted column '{col}' with {self.low_cardinality_columns_[col]} "
                    f"unique values to object type"
                )
                
                # Add suffix to values if requested
                if self.add_suffix:
                    X_transformed[col] = X_transformed[col].apply(
                        lambda x: str(x) + self.suffix if pd.notna(x) else x
                    ).astype('object')
                    logger.info(
                        f"Added suffix '{self.suffix}' to all values in column '{col}'"
                    )
        else:
            logger.info("No low cardinality columns found in current data.")
        
        logger.info(f"Transform complete. Returned DataFrame with shape {X_transformed.shape}")
        return X_transformed
    
