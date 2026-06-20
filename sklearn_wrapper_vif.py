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
        List of feature names to retain after fitting (all features with final VIF below the threshold).

        Example:
            ['credit_limit_woe', 'age_woe', 'payment_amount_sep_2005_woe', 'repayment_status_sep_2005_woe', ...]

    vif_results_ : dict
        Alias for iteration_steps_. Contains the same step-by-step removal history. See iteration_steps_.

    iteration_steps_ : dict
        Integer-keyed dict tracking each removal step. Each step records all feature VIF values at that
        point in the iteration, which feature was removed, and its VIF value.

        Structure:
            {
                <step: int>: {
                    'vifResults': {
                        <feature_name: str>: {
                            'vif': <float>,
                            'interpretation': <str>   # human-readable severity label
                        },
                        ...                           # one entry per feature still present at this step
                    },
                    'removedFeatureName': <str>,       # feature eliminated in this step
                    'removedFeatureVIF':  <np.float64> # VIF value that triggered its removal
                },
                ...
            }

        Example (first 2 steps):
            {
                1: {
                    'vifResults': {
                        'bill_amount_jun_2005_woe': {
                            'vif': 8.007729933441384,
                            'interpretation': 'Moderate to high correlation with other predictors. Consider checking for multicollinearity.'
                        },
                        'bill_amount_jul_2005_woe': {
                            'vif': 7.786321474925436,
                            'interpretation': 'Moderate to high correlation with other predictors. Consider checking for multicollinearity.'
                        },
                        'credit_limit_woe': {
                            'vif': 1.5706809870938874,
                            'interpretation': 'Mild to moderate correlation with other predictors (usually fine).'
                        },
                        ...
                    },
                    'removedFeatureName': 'bill_amount_jun_2005_woe',
                    'removedFeatureVIF':  np.float64(8.007729933441384)
                },
                2: {
                    'vifResults': {
                        'bill_amount_aug_2005_right': {
                            'vif': 6.1405193680979115,
                            'interpretation': 'Moderate to high correlation with other predictors. Consider checking for multicollinearity.'
                        },
                        'credit_limit_woe': {
                            'vif': 1.5703265761121037,
                            'interpretation': 'Mild to moderate correlation with other predictors (usually fine).'
                        },
                        ...
                    },
                    'removedFeatureName': 'bill_amount_aug_2005_right',
                    'removedFeatureVIF':  np.float64(6.1405193680979115)
                },
                ...
            }

        Note: The loop terminates when all remaining features have VIF < vif_threshold,
        so the final kept features are NOT recorded as a step — only removals are logged.
    
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
    
    def __init__(self, vif_threshold=5.0, max_iterations=100, output_dir=None):
        """
        Initialize the VIFTransformer.
        
        Parameters
        ----------
        vif_threshold : float, default=5.0
            The VIF threshold above which features will be removed.
        max_iterations : int, default=100
            The maximum number of iterations for feature removal.
        output_dir : str or None, default=None
            Directory to save VIF results. If provided, two files are written:
            - vif_iterative_steps.json  : full step-by-step removal history
            - vif_iterative_steps.csv   : tabular view of the same steps
            If None, no files are saved.
        """
        self.vif_threshold = vif_threshold
        self.max_iterations = max_iterations
        self.output_dir = output_dir
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
        
        # Resolve output paths if an output directory is provided
        import os
        output_json_path = os.path.join(self.output_dir, "vif_iterative_steps.json") if self.output_dir else None
        output_csv_path  = os.path.join(self.output_dir, "vif_iterative_steps.csv")  if self.output_dir else None

        # Perform iterative VIF-based feature selection
        selected_features, steps = iterative_feature_selector_with_vif(
            X,
            vif_treshold=self.vif_threshold,
            maxiterations=self.max_iterations,
            output_json_path=output_json_path,
            output_csv_path=output_csv_path
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

    

