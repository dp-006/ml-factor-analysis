"""
Accelera Consulting

sklearn custom transformer for handling missing columns.

This module provides a sklearn-compatible transformer that detects and removes
columns with a proportion of missing values exceeding a specified threshold.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from helper.column_operations import detect_missing_columns_with_treshold
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_missing_handler"
logger_file_name = "sklearn_wrapper_missing_handler.log"
logger = get_logger(logger_name, logger_file_name)


class MissingColumnsHandler(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn custom transformer that detects and handles columns with high 
    proportion of missing values.
    
    This transformer wraps the detect_missing_columns_with_treshold function 
    from column_operations module to provide sklearn-compatible pipeline functionality.
    
    Attributes
    ----------
    threshold : float, optional
        Proportion of missing values above which a column is considered to have
        too many missing values. Default is 0.5 (50%). Must be between 0 and 1.
    
    missing_columns_ : list
        List of column names with missing values above threshold (learned during fit).
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn_wrapper_missing_handler import MissingColumnsHandler
    >>> 
    >>> # Create sample data with missing values
    >>> df = pd.DataFrame({
    ...     'col1': [1, 2, None, 4],
    ...     'col2': [None, None, None, 1],  # 75% missing
    ...     'col3': [1, 2, 3, 4]
    ... })
    >>> 
    >>> # Create and fit transformer
    >>> handler = MissingColumnsHandler(threshold=0.5, action='remove')
    >>> handler.fit(df)
    >>> 
    >>> # Transform data
    >>> df_cleaned = handler.transform(df)
    >>> print(df_cleaned.columns)  # 'col2' will be removed
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the MissingColumnsHandler transformer.
        
        Parameters
        ----------
        threshold : float, optional
            Proportion of missing values above which a column is considered problematic.
            Default is 0.5 (50%). Must be between 0 and 1.
        
        Raises
        ------
        ValueError
            If threshold is not between 0 and 1.
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a numeric value.")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be a float between 0 and 1.")
        
        self.threshold = threshold
        self.missing_columns_ = None
        logger.info(f"MissingColumnsHandler initialized with threshold={threshold}")
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by detecting columns with missing values above threshold.
        
        Purpose
        -------
        Learns which columns have a proportion of missing values exceeding the 
        specified threshold by analyzing the input DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to analyze for missing values.
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
        
        logger.info(f"Fitting MissingColumnsHandler on DataFrame with shape {X.shape}")
        
        # Detect missing columns using the helper function
        self.missing_columns_ = detect_missing_columns_with_treshold(X, self.threshold)
        
        logger.info(
            f"Fit complete. Detected {len(self.missing_columns_)} columns with "
            f"missing values > {self.threshold*100}%: {self.missing_columns_}"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by removing columns with high missing values.
        
        Purpose
        -------
        Removes columns that have a proportion of missing values exceeding the 
        specified threshold (as detected during fit).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame without columns that have missing values above 
            the specified threshold (as detected during fit).
        
        Raises
        ------
        ValueError
            If input X is not a pandas DataFrame, or if transformer has not been 
            fitted yet.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        
        if self.missing_columns_ is None:
            raise ValueError(
                "Transformer has not been fitted yet. Call fit() first before "
                "calling transform()."
            )
        
        logger.info(f"Transforming DataFrame with shape {X.shape}")
        
        # Get columns that exist in both X and missing_columns_
        cols_to_drop = [col for col in self.missing_columns_ if col in X.columns]
        
        if cols_to_drop:
            logger.info(
                f"Removing {len(cols_to_drop)} columns with high missing values: "
                f"{cols_to_drop}"
            )
            X_transformed = X.drop(columns=cols_to_drop)
        else:
            logger.info("No columns from missing_columns_ found in current data.")
            X_transformed = X.copy()
        
        logger.info(f"Transform complete. Returned DataFrame with shape {X_transformed.shape}")
        return X_transformed
    