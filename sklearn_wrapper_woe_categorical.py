'''
Accelera Consulting
Author: Accelera Team

Custom sklearn transformer for Weight of Evidence (WOE) transformation of categorical features.
Computes WOE binning during fit() and applies the mapping during transform().
'''

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from auto_binning_woe_categorical import auto_woe_binning_categorical
from helper import detect_outlier_indicator_columns
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_woe_categorical"
logger_file_name = "sklearn_wrapper_woe_categorical.log"
logger = get_logger(logger_name, logger_file_name)


class WOETransformerCategorical(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn-compatible custom transformer that computes and applies Weight of Evidence (WOE)
    transformation for categorical (object dtype) features during fit() and transform() operations.
    
    Unlike the previous implementation that read from pre-computed JSON files, this transformer
    computes WOE binning on-the-fly during fitting using the auto_woe_binning_categorical algorithm.
    
    Parameters
    ----------
    target_col : str
        Name of the binary target column in the training data. Default is 'TARGET'.
    
    min_bin_pct : float, optional
        Minimum allowed observation ratio per bin. Default is 0.05.
    
    max_final_bins : int, optional
        Maximum allowed number of final bins after merging. Default is 6.
    
    min_final_bins : int, optional
        Minimum allowed number of final bins. Default is 2.
    
    min_iv : float, optional
        Minimum acceptable Information Value. Default is 0.02.
    
    max_iv : float, optional
        Maximum acceptable Information Value. Default is 0.50.
    
    max_iter : int, optional
        Maximum number of merge iterations. Default is 20.
    
    unseen : float, optional
        WOE value to assign to unseen/unknown categories. Default is 0.
    
    Attributes
    ----------
    category_woe_dict_ : dict
        Maps each fitted feature name to a dict of {category_value (str): woe_value (float)}.
        Category values are stored as strings regardless of their original dtype.
        Used during transform() to look up the WOE for each observed category.

        Example::

            category_woe_dict_ = {
                'education': {
                    '0':  1.9607,   # category 0 -> group with highest WOE
                    '1':  0.1673,
                    '2': -0.0812,
                    'Rare': -0.3401
                },
                'gender': {
                    '1': -0.1255,
                    '2':  0.0880
                }
            }

    feature_names_ : list
        Names of features that were successfully fitted (in fit order).

        Example::

            feature_names_ = ['gender', 'education', 'marital_status', ...]

    binning_results_ : dict
        Maps each fitted feature name to the full result dict returned by
        auto_woe_binning_categorical. Useful for auditing bin quality.

        Example::

            binning_results_ = {
                'education': {
                    'feature':        'education',
                    'totalIv':        0.0357,
                    'interpretIv':    'Weak Predictive Power',
                    'numberofBins':   7,
                    'status':         'REVIEW_NOT_CONVERGED',  # PASS | REVIEW_* | REJECTED
                    'converged':      False,
                    'stopIteration':  20,
                    'valuesToTheGroup': {
                        'education_grp0': ['0'],
                        'education_grp1': ['1', '2'],
                        ...
                    },
                    'woe_table': [...],   # list of bin-level dicts
                    'checks':    {...}
                }
            }
    """
    
    def __init__(self,
                 target_col: str = 'TARGET',
                 min_bin_pct: float = 0.05,
                 max_final_bins: int = 6,
                 min_final_bins: int = 2,
                 min_iv: float = 0.02,
                 max_iv: float = 0.50,
                 max_iter: int = 20,
                 unseen: float = 0):
        self.target_col = target_col
        self.min_bin_pct = min_bin_pct
        self.max_final_bins = max_final_bins
        self.min_final_bins = min_final_bins
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.max_iter = max_iter
        self.unseen = unseen
        
        self.category_woe_dict_ = {}
        self.feature_names_ = []
        self.binning_results_ = {}
    
    def fit(self, X, y=None):
        """
        Purpose
        -------
        Compute WOE binning for each categorical feature in X using auto_woe_binning_categorical.
        Store the category-to-WOE mappings for use in transform().
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing categorical features (target column should NOT be included).
        
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
        
        # Identify categorical (object dtype) features
        categorical_features = X_with_target.select_dtypes(include=['object']).columns.tolist()
        # Remove target column from features
        categorical_features = [f for f in categorical_features if f != 'TARGET']
        # Skip outlier indicator columns (_right, _left) — they are binary flags, not suitable for WOE binning
        indicator_cols = detect_outlier_indicator_columns(categorical_features)
        categorical_features = [f for f in categorical_features if f not in indicator_cols]
        
        if not categorical_features:
            raise ValueError("No categorical (object dtype) features found in X (excluding target column)")
        
        logger.info(f"Starting WOE fitting for {len(categorical_features)} categorical features: {categorical_features}")
        
        self.feature_names_ = []
        self.category_woe_dict_ = {}
        self.binning_results_ = {}
        
        for feature in categorical_features:
            try:
                logger.info(f"Computing WOE for categorical feature: {feature}")
                
                # Call auto_woe_binning_categorical to compute binning
                binning_result = auto_woe_binning_categorical(
                    df=X_with_target,
                    feature=feature,
                    target='TARGET',
                    min_bin_pct=self.min_bin_pct,
                    max_final_bins=self.max_final_bins,
                    min_final_bins=self.min_final_bins,
                    min_iv=self.min_iv,
                    max_iv=self.max_iv,
                    max_iter=self.max_iter
                )
                
                # Store the complete binning result
                self.binning_results_[feature] = binning_result
                
                # Extract valuesToTheGroup and woe_table
                values_to_group = binning_result.get("valuesToTheGroup", {})
                woe_table = binning_result.get("woe_table", [])
                
                if not values_to_group or not woe_table:
                    logger.warning(f"No mapping found for feature '{feature}'. Skipping.")
                    continue
                
                # Build bin_name -> woe mapping from woe_table
                # woe_table format: [{'_bin': 'feature_grp0', 'woe': 0.123, ...}, ...]
                bin_woe = {}
                for item in woe_table:
                    bin_name = item.get("_bin")
                    woe_value = float(item.get("woe", 0.0))
                    bin_woe[bin_name] = woe_value
                
                # Build category_value -> woe mapping using valuesToTheGroup
                # valuesToTheGroup format: {'feature_grp0': ['cat1', 'cat2'], 'feature_grp1': ['cat3'], ...}
                category_woe = {}
                for bin_name, categories in values_to_group.items():
                    woe_value = bin_woe.get(bin_name)
                    if woe_value is None:
                        logger.warning(f"No WOE value found for bin '{bin_name}' in feature '{feature}'")
                        continue
                    for category in categories:
                        # Store category -> woe_value mapping
                        # Category values are stored as strings (as they appear in JSON)
                        category_woe[str(category)] = woe_value
                
                if not category_woe:
                    logger.warning(f"No category WOE mapping built for feature '{feature}'. Skipping.")
                    continue
                
                self.category_woe_dict_[feature] = category_woe
                self.feature_names_.append(feature)
                
                # Log mapping information
                total_iv = binning_result.get("totalIv", 0)
                logger.info(f"Feature '{feature}' fitted with {len(category_woe)} category mappings and IV={total_iv:.4f}")
                for bin_name, categories in values_to_group.items():
                    woe_val = bin_woe.get(bin_name, 0)
                    logger.info(f"  {bin_name} ({len(categories)} categories): WOE = {woe_val:.6f}")
                
            except Exception as e:
                error_message = f"Error computing WOE for categorical feature '{feature}': {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)
        
        if not self.feature_names_:
            raise ValueError("No categorical features were successfully fitted for WOE transformation")
        
        logger.info(f"WOE fitting completed for {len(self.feature_names_)} categorical features: {self.feature_names_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose
        -------
        Transform categorical features to WOE values using the fitted category-to-WOE mappings.
        Original feature columns are replaced with WOE-transformed versions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with categorical feature columns to transform.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with original categorical features replaced by WOE-transformed columns.

        Notes
        -----
        Unseen categories
            During transform, each observation value is cast to str and looked up
            in ``category_woe_dict_`` (built from ``valuesToTheGroup`` during fit).

            If a category was not present in the training data — or was not assigned
            to any bin (e.g. appeared after RareLabelEncoder threshold changes) —
            the dict lookup returns ``self.unseen`` (default 0).

            A WOE of 0 is neutral — it carries no good/bad signal and does not
            bias the model in either direction.
        """
        if not self.category_woe_dict_:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X_transformed = X.copy()
        
        for feature in self.feature_names_:
            if feature not in X.columns:
                logger.warning(f"Feature '{feature}' not found in input dataframe during transform")
                continue
            
            category_woe = self.category_woe_dict_[feature]
            
            # Convert feature values to string (to match the stored mapping keys)
            feature_values = X[feature].astype(str)
            
            # Map each category to its WOE value
            # For unknown categories, use self.unseen value
            woe_values = np.array([
                round(category_woe.get(cat, self.unseen), 6) for cat in feature_values
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
            - 'category_woe_mapping': {category_value: woe_value}
            - 'binning_result': complete result dict from auto_woe_binning_categorical
        
        Raises
        ------
        ValueError
            If the feature was not fitted or not found.
        """
        if feature_name not in self.feature_names_:
            raise ValueError(f"Feature '{feature_name}' was not fitted")
        
        return {
            'category_woe_mapping': self.category_woe_dict_[feature_name],
            'binning_result': self.binning_results_[feature_name]
        }
