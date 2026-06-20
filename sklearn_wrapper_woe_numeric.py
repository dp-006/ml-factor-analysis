'''
Accelera Consulting
Author: Accelera Team

Custom sklearn transformer for Weight of Evidence (WOE) transformation of numeric features.
Computes WOE binning during fit() and applies the mapping during transform().
'''

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from auto_binning_woe_numeric import auto_woe_binning_numeric
from helper import detect_outlier_indicator_columns
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_woe_numeric"
logger_file_name = "sklearn_wrapper_woe_numeric.log"
logger = get_logger(logger_name, logger_file_name)


class WOETransformerNumeric(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn-compatible custom transformer that computes and applies Weight of Evidence (WOE)
    transformation for numeric features during fit() and transform() operations.
    
    Unlike the previous implementation that read from pre-computed JSON files, this transformer
    computes WOE binning on-the-fly during fitting using the auto_woe_binning_numeric algorithm.
    
    Parameters
    ----------
    target_col : str
        Name of the binary target column in the training data. Default is 'TARGET'.
    
    initial_bins : int, optional
        Number of initial quantile-based bins for the algorithm. Default is 20.
    
    min_bin_pct : float, optional
        Minimum allowed observation ratio per bin. Default is 0.05.
    
    max_final_bins : int, optional
        Maximum allowed number of final bins after merging. Default is 10.
    
    min_final_bins : int, optional
        Minimum allowed number of final bins. Default is 3.
    
    min_iv : float, optional
        Minimum acceptable Information Value. Default is 0.02.
    
    max_iv : float, optional
        Maximum acceptable Information Value. Default is 0.50.
    
    max_iter : int, optional
        Maximum number of merge iterations. Default is 25.
    
    unseen : float, optional
        WOE value to assign to unseen/out-of-range values. Default is 0.
    
    Attributes
    ----------
    bins_dict_ : dict
        Maps each fitted feature name to a pd.IntervalIndex of its final bins.
        Used during transform() to assign each observation to a bin.

        Example::

            bins_dict_ = {
                'credit_limit': IntervalIndex([
                    (9999.999, 30000.0],
                    (30000.0,  70000.0],
                    (70000.0, 100000.0],
                    ...
                ], dtype='interval[float64, right]')
            }

    woe_dict_ : dict
        Maps each fitted feature name to a dict of {bin_index (int): woe_value (float)}.
        Bin indices correspond positionally to the intervals in bins_dict_.

        Example::

            woe_dict_ = {
                'credit_limit': {
                    0: -0.7246,   # bin (9999.999, 30000.0]
                    1: -0.2811,   # bin (30000.0,  70000.0]
                    2: -0.1241,   # ...
                    3: -0.0561,
                    4:  0.2117,
                    5:  0.3084,
                    6:  0.5362,
                    7:  0.7722
                }
            }

    feature_names_ : list
        Names of features that were successfully fitted (in fit order).

        Example::

            feature_names_ = ['credit_limit', 'age', 'bill_amount_sep_2005', ...]

    binning_results_ : dict
        Maps each fitted feature name to the full result dict returned by
        auto_woe_binning_numeric. Useful for auditing bin quality.

        Example::

            binning_results_ = {
                'credit_limit': {
                    'feature':       'credit_limit',
                    'totalIv':       0.1973,
                    'interpretIv':   'Medium Predictive Power',
                    'numberofBins':  8,
                    'status':        'PASS',          # PASS | REVIEW_* | REJECTED
                    'converged':     True,
                    'stopIteration': 5,
                    'woe_table':     [...],           # list of bin-level dicts
                    'finalIntervals': ['(9999.999, 30000.0]', ...],
                    'checks':        {...}
                }
            }
    """
    
    def __init__(self,
                 target_col: str = 'TARGET',
                 initial_bins: int = 20,
                 min_bin_pct: float = 0.05,
                 max_final_bins: int = 10,
                 min_final_bins: int = 3,
                 min_iv: float = 0.02,
                 max_iv: float = 0.50,
                 max_iter: int = 25,
                 unseen: float = 0):
        self.target_col = target_col
        self.initial_bins = initial_bins
        self.min_bin_pct = min_bin_pct
        self.max_final_bins = max_final_bins
        self.min_final_bins = min_final_bins
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.max_iter = max_iter
        self.unseen = unseen
        
        self.bins_dict_ = {}
        self.woe_dict_ = {}
        self.feature_names_ = []
        self.binning_results_ = {}
    
    def fit(self, X, y=None):
        """
        Purpose
        -------
        Compute WOE binning for each numeric feature in X using auto_woe_binning_numeric.
        Store the bin definitions and WOE mappings for use in transform().
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing numeric features (target column should NOT be included).
        
        y : pd.Series or array-like
            Binary target variable. Required for WOE computation.
            Will be concatenated with X internally with column name 'TARGET'.
        
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        if y is None:
            raise ValueError("y (target variable) is required for WOE fitting")
        
        # Convert y to Series if it's not already
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        
        # Concatenate X with target internally
        X_with_target = pd.concat([X, y.rename('TARGET')], axis=1)
        
        # Identify numeric features
        numeric_features = X_with_target.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target column from features
        numeric_features = [f for f in numeric_features if f != 'TARGET']
        # Skip outlier indicator columns (_right, _left) — they are binary flags, not suitable for WOE binning
        indicator_cols = detect_outlier_indicator_columns(numeric_features)
        numeric_features = [f for f in numeric_features if f not in indicator_cols]
        
        if not numeric_features:
            raise ValueError("No numeric features found in X (excluding target column)")
        
        logger.info(f"Starting WOE fitting for {len(numeric_features)} numeric features: {numeric_features}")
        
        self.feature_names_ = []
        self.bins_dict_ = {}
        self.woe_dict_ = {}
        self.binning_results_ = {}
        
        for feature in numeric_features:
            try:
                logger.info(f"Computing WOE for feature: {feature}")
                
                # Call auto_woe_binning_numeric to compute binning
                binning_result = auto_woe_binning_numeric(
                    df=X_with_target,
                    feature=feature,
                    target='TARGET',
                    initial_bins=self.initial_bins,
                    min_bin_pct=self.min_bin_pct,
                    max_final_bins=self.max_final_bins,
                    min_final_bins=self.min_final_bins,
                    min_iv=self.min_iv,
                    max_iv=self.max_iv,
                    max_iter=self.max_iter
                )
                
                # Store the complete binning result
                self.binning_results_[feature] = binning_result
                
                # Extract bin intervals from final_intervals
                # final_intervals are pandas Interval objects
                final_intervals = binning_result.get("final_intervals", [])
                
                if not final_intervals:
                    logger.warning(f"No final intervals found for feature '{feature}'. Skipping.")
                    continue
                
                # Create IntervalIndex from the final intervals
                interval_index = pd.IntervalIndex(final_intervals)
                self.bins_dict_[feature] = interval_index
                
                # Extract WOE values from woe_table (top-level key in binning_result)
                woe_table = binning_result.get("woe_table", [])
                
                if not woe_table:
                    logger.warning(f"No WOE table found for feature '{feature}'. Skipping.")
                    continue
                
                # Create bin_index -> woe_value mapping
                woe_mapping = {}
                for idx, item in enumerate(woe_table):
                    woe_value = float(item.get("woe", 0.0))
                    woe_mapping[idx] = woe_value
                
                self.woe_dict_[feature] = woe_mapping
                self.feature_names_.append(feature)
                
                # Log bin information
                logger.info(f"Feature '{feature}' fitted with {len(woe_mapping)} bins and IV={binning_result.get('totalIv', 0):.4f}")
                for idx, woe_val in woe_mapping.items():
                    logger.info(f"  Bin {idx}: WOE = {woe_val:.6f}")
                
            except Exception as e:
                error_message = f"Error computing WOE for feature '{feature}': {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)
        
        if not self.feature_names_:
            raise ValueError("No features were successfully fitted for WOE transformation")
        
        logger.info(f"WOE fitting completed for {len(self.feature_names_)} features: {self.feature_names_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose
        -------
        Transform numeric features to WOE values using the fitted bin definitions and mappings.
        Original feature columns are replaced with WOE-transformed versions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with numeric feature columns to transform.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with original numeric features replaced by WOE-transformed columns.

        Notes
        -----
        Unseen / out-of-range values
            During transform, each observation is assigned to a bin using
            pd.IntervalIndex.get_indexer(). Two unseen cases are handled silently:

            1. Value falls outside all fitted intervals (get_indexer returns -1):
               Assigned ``self.unseen`` (default 0).

            2. Value falls inside an interval but the bin index is not in woe_dict_
               (should not occur in normal usage):
               Assigned ``self.unseen`` (default 0).

            A WOE of 0 is neutral — it carries no good/bad signal and does not
            bias the model in either direction.
        """
        if not self.bins_dict_:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X_transformed = X.copy()
        
        for feature in self.feature_names_:
            if feature not in X.columns:
                logger.warning(f"Feature '{feature}' not found in input dataframe during transform")
                continue
            
            bins = self.bins_dict_[feature]
            woe_mapping = self.woe_dict_[feature]
            
            # Map each observation to its bin index
            positions = bins.get_indexer(X[feature].to_numpy())
            
            # Convert bin indices to WOE values
            woe_values = np.array([
                round(woe_mapping.get(pos, self.unseen), 6) if pos >= 0 else self.unseen
                for pos in positions
            ])
            
            # Create WOE column and drop original
            woe_col_name = f"{feature}_woe"
            X_transformed[woe_col_name] = woe_values
            X_transformed = X_transformed.drop(columns=[feature])
            
            logger.info(f"Transformed feature '{feature}' to WOE and dropped original column")
        
        logger.info(f"Transform completed for {len(self.feature_names_)} features")
        return X_transformed
    
    def get_feature_info(self, feature_name: str) -> dict:
        """
        Purpose
        -------
        Retrieve detailed information about a specific fitted feature's WOE binning.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature to retrieve information for.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'bins': IntervalIndex of bins
            - 'woe_mapping': {bin_index: woe_value}
            - 'binning_result': complete result dict from auto_woe_binning_numeric
        
        Raises
        ------
        ValueError
            If the feature was not fitted or not found.
        """
        if feature_name not in self.feature_names_:
            raise ValueError(f"Feature '{feature_name}' was not fitted")
        
        return {
            'bins': self.bins_dict_[feature_name],
            'woe_mapping': self.woe_dict_[feature_name],
            'binning_result': self.binning_results_[feature_name]
        }
