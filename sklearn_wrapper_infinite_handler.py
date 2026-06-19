"""
Accelera Consulting

sklearn custom transformer for handling infinite columns.

This module provides a sklearn-compatible transformer that detects and removes
or replaces columns containing infinite values (np.inf or -np.inf).
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from helper.column_operations import detect_infinite_columns
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_infinite_handler"
logger_file_name = "sklearn_wrapper_infinite_handler.log"
logger = get_logger(logger_name, logger_file_name)


class InfiniteColumnsHandler(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn custom transformer that detects and handles columns with 
    infinite values (np.inf or -np.inf).
    
    This transformer wraps the detect_infinite_columns function from the 
    column_operations module to provide sklearn-compatible pipeline functionality.
    
    Attributes
    ----------
    infinite_columns_ : list
        List of column names with infinite values (learned during fit).
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn_wrapper_infinite_handler import InfiniteColumnsHandler
    >>> 
    >>> # Create sample data with infinite values
    >>> df = pd.DataFrame({
    ...     'col1': [1.0, 2.0, np.inf, 4.0],
    ...     'col2': [1.0, 2.0, 3.0, 4.0],
    ...     'col3': [-np.inf, 2.0, 3.0, 4.0]
    ... })
    >>> 
    >>> # Create and fit transformer to remove infinite columns
    >>> handler = InfiniteColumnsHandler()
    >>> handler.fit(df)
    >>> 
    >>> # Transform data
    >>> df_cleaned = handler.transform(df)
    >>> print(df_cleaned.columns)  # 'col1' and 'col3' will be removed
    """
    
    def __init__(self):
        """
        Initialize the InfiniteColumnsHandler transformer.
        
        This transformer removes all columns that contain infinite values.
        """
        self.infinite_columns_ = None
        logger.info(f"InfiniteColumnsHandler initialized")
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by detecting columns with infinite values.
        
        Purpose
        -------
        Learns which columns contain infinite values (np.inf or -np.inf) by 
        analyzing the input DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to analyze for infinite values.
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
        
        logger.info(f"Fitting InfiniteColumnsHandler on DataFrame with shape {X.shape}")
        
        # Detect infinite columns using the helper function
        self.infinite_columns_ = detect_infinite_columns(X)
        
        logger.info(
            f"Fit complete. Detected {len(self.infinite_columns_)} columns with "
            f"infinite values: {self.infinite_columns_}"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by removing columns with infinite values.
        
        Purpose
        -------
        Removes columns that contain infinite values (np.inf or -np.inf) as detected 
        during fit.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed DataFrame without columns that have infinite values 
            (as detected during fit).
        
        Raises
        ------
        ValueError
            If input X is not a pandas DataFrame, or if transformer has not been 
            fitted yet.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        
        if self.infinite_columns_ is None:
            raise ValueError(
                "Transformer has not been fitted yet. Call fit() first before "
                "calling transform()."
            )
        
        logger.info(f"Transforming DataFrame with shape {X.shape}")
        X_transformed = X.copy()
        
        # Get columns that exist in both X and infinite_columns_
        cols_to_drop = [col for col in self.infinite_columns_ if col in X_transformed.columns]
        
        if cols_to_drop:
            logger.info(
                f"Removing {len(cols_to_drop)} columns with infinite values: "
                f"{cols_to_drop}"
            )
            X_transformed = X_transformed.drop(columns=cols_to_drop)
        else:
            logger.info("No columns from infinite_columns_ found in current data.")
        
        logger.info(f"Transform complete. Returned DataFrame with shape {X_transformed.shape}")
        return X_transformed
    