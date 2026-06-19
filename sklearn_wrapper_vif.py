"""
Sklearn-compatible custom transformer for VIF-based feature selection.

This module provides a VIFTransformer class that integrates with sklearn pipelines
for automated feature selection based on Variance Inflation Factor analysis.

Usage in pipelines:
    from sklearn.pipeline import Pipeline
    from sklearn_wrapper_vif import VIFTransformer
    
    pipeline = Pipeline([
        ('vif_selector', VIFTransformer(vif_threshold=5.0)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from vif_analysis import variance_inflation_factor, iterative_feature_selector_with_vif
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_vif"
logger_file_name = "sklearn_wrapper_vif.log"
logger = get_logger(logger_name, logger_file_name)


class VIFTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for feature selection using Variance Inflation Factor (VIF).
    
    This transformer removes features with high multicollinearity (VIF above threshold) during fitting,
    and then removes those same features during transformation. It can be used in sklearn pipelines
    and GridSearchCV for automated feature selection.
    
    Parameters
    ----------
    vif_threshold : float, default=5.0
        The VIF threshold above which features will be removed. Features with VIF >= threshold
        are considered to have high multicollinearity.
    
    max_iterations : int, default=100
        The maximum number of iterations to perform during the iterative feature removal process.
        If the algorithm doesn't converge before this limit, it will return the remaining features.
    
    Attributes
    ----------
    features_to_keep_ : list
        List of feature names to retain after fitting. These are features with VIF below the threshold.
        Example: ['feature_1', 'feature_3', 'feature_5']
    
    vif_results_ : dict
        Dictionary containing the VIF values for each feature before removal.
        Example:
        {
            "feature_1": {"vif": 10.5, "interpretation": "High multicollinearity"},
            "feature_2": {"vif": 3.2, "interpretation": "Acceptable multicollinearity"},
            ...
        }
    
    iteration_steps_ : dict
        Dictionary tracking the iterative removal process, showing which features were removed at each step.
        Example:
        {
            "iteration_1": {"removed_feature": "feature_1", "vif_values": {"feature_1": 10.5, "feature_2": 3.2, ...}},
            "iteration_2": {"removed_feature": "feature_3", "vif_values": {"feature_2": 3.2, "feature_4": 6.8, ...}},
            ...
        }
    
    Examples
    --------
    >>> from sklearn_wrapper_vif import VIFTransformer
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 4, 6], 'C': [1.5, 3, 4.5]})
    >>> y = [0, 1, 0]
    >>> 
    >>> # Use in a pipeline
    >>> pipeline = Pipeline([
    ...     ('vif', VIFTransformer(vif_threshold=5.0)),
    ...     ('model', LogisticRegression())
    ... ])
    >>> pipeline.fit(X, y)
    >>> predictions = pipeline.predict(X)
    """
    
    def __init__(self, vif_threshold=5.0, max_iterations=100):
        """
        Initialize the VIFTransformer.
        
        Parameters
        ----------
        vif_threshold : float, default=5.0
            The VIF threshold above which features will be removed.
        max_iterations : int, default=100
            The maximum number of iterations for feature removal.
        """
        self.vif_threshold = vif_threshold
        self.max_iterations = max_iterations
        self.features_to_keep_ = None
        self.vif_results_ = None
        self.iteration_steps_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer by identifying features to keep based on VIF analysis.
        
        This method performs iterative feature selection, removing features with high VIF
        until all remaining features have VIF below the threshold.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features. If numpy array, features will be converted to DataFrame with
            default column names (feature_0, feature_1, etc.).
        y : array-like, optional
            Target variable. Not used, present for sklearn compatibility.
        
        Returns
        -------
        self : VIFTransformer
            Returns self for method chaining in sklearn pipelines.
        
        Raises
        ------
        ValueError
            If X is empty, contains non-numeric columns, NaN values, or infinite values.
        """
        logger.info("=" * 80)
        logger.info("VIFTransformer.fit() - Starting feature selection with VIF analysis")
        logger.info(f"VIF Threshold: {self.vif_threshold}, Max Iterations: {self.max_iterations}")
        logger.info("=" * 80)
        
        # Convert numpy array to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        # Validate input data
        if X.empty:
            error_message = "Input DataFrame is empty. Cannot perform VIF analysis on empty data."
            logger.error(error_message)
            raise ValueError(error_message)
        
        if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
            error_message = "Input DataFrame contains non-numeric columns. VIF can only be calculated for numeric features."
            logger.error(error_message)
            raise ValueError(error_message)
        
        if X.isnull().values.any():
            error_message = "Input DataFrame contains NaN values. Please remove or impute NaN values before fitting."
            logger.error(error_message)
            raise ValueError(error_message)
        
        if np.isinf(X.values).any():
            error_message = "Input DataFrame contains infinite values. Please remove or impute infinite values before fitting."
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Perform iterative VIF-based feature selection
        selected_features, steps = iterative_feature_selector_with_vif(
            X,
            vif_treshold=self.vif_threshold,
            maxiterations=self.max_iterations,
            output_json_path=None,
            output_csv_path=None
        )
        
        # Store the results
        self.features_to_keep_ = selected_features
        self.iteration_steps_ = steps
        
        # Calculate VIF for all features in the final set for reference
        X_final = X[self.features_to_keep_]

        # Vif Results store steps that comes from iterative_feature_selector_with_vif, which already contains VIF values for all features at each step.
        self.vif_results_ = steps
        
        logger.info(f"VIFTransformer fitting completed. "
                    f"Number of features to keep: {len(self.features_to_keep_)} "
                    f"Number of features removed: {X.shape[1] - len(self.features_to_keep_)}")
        
        # Log removed features and their VIF values from steps
        for step, details in steps.items():
            removed_feature = details.get("removedFeatureName")
            if removed_feature:
                vif_value = details.get("removedFeatureVIF", "N/A")
                logger.info(f"Removed feature: {removed_feature}, VIF: {vif_value}, Removed in step: {step}")
        logger.info("=" * 80)
        
        return self
    
    def transform(self, X):
        """
        Transform the data by removing features not in the retained set.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features to transform. Must have the same features as the data used in fit().
        
        Returns
        -------
        X_transformed : pd.DataFrame or np.ndarray
            Transformed data with only the retained features.
        
        Raises
        ------
        ValueError
            If transformer hasn't been fitted yet or if X contains unexpected features.
        """
        if self.features_to_keep_ is None:
            error_message = "VIFTransformer must be fitted before calling transform(). Call fit() first."
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Handle numpy array input
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        logger.info(f"Transforming data: selecting {len(self.features_to_keep_)} features from {X_df.shape[1]} total features")
        
        # Select only the retained features
        X_transformed = X_df[self.features_to_keep_]
        
        # Return in the same format as input
        if is_numpy:
            return X_transformed.values
        else:
            return X_transformed

    

