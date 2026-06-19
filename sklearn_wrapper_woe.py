'''
Accelera Consulting
Author: Accelera Team

Standalone helper to transform a numeric feature into its Weight of Evidence (WOE)
representation using predefined pandas Interval bins.
'''

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_woe"
logger_file_name = "sklearn_wrapper_woe.log"
logger = get_logger(logger_name, logger_file_name)


class WOETransformerNumeric(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn-compatible custom transformer that converts numeric features to 
    Weight of Evidence (WOE) values using predefined numeric bins from JSON metadata files.
    
    The transformer reads bin definitions from JSON files located at:
    outputs/auto_binning_woe/{feature_name}/final_auto_binning_woe.json
    
    This transformer is designed for numeric features with interval bins like:
    (-inf, 20.999], (20.999, 35.0], (35.0, 50.0], etc.
    
    Parameters
    ----------
    base_path : str, optional
        Base directory path for WOE metadata. Default is 'outputs/auto_binning_woe'
    
    unseen : float, optional
        WOE value to assign to unseen/out-of-range values (not found in any bin during transform).
        Default is 0 (neutral value: has no effect when multiplied by model coefficients).
        Pass None to use np.nan for unseen values instead.
    
    Attributes
    ----------
    bins_dict_ : dict
        Dictionary storing pd.IntervalIndex for each feature after fit()
    feature_names_ : list
        List of feature names after fit()
    """
    
    def __init__(self, base_path: str = "outputs/auto_binning_woe", unseen: float = 0):
        self.base_path = base_path
        self.unseen = unseen
        self.bins_dict_ = {}          # {feature_name: pd.IntervalIndex}
        self.woe_dict_ = {}           # {feature_name: {bin_index: woe_value}}
        self.feature_names_ = []
    
    def _parse_bin_string(self, bin_string: str) -> tuple:
        """
        Convert bin string like '(20.999, 35.0]' to tuple (20.999, 35.0)

        Parameters
        ----------
        bin_string : str
            Bin string from JSON metadata, e.g., '(20.999, 35.0]'
        
        Returns
        -------
        tuple
            Tuple of floats representing the bin edges, e.g., (20.999, 35.0)
        """
        # Strip whitespace, for example: ' (20.999, 35.0] ' -> '(20.999, 35.0]'
        bin_string = bin_string.strip()
        
        # Remove brackets and split
        inner = bin_string[1:-1] # removes the first and last character (the brackets), for example: '(20.999, 35.0]' -> '20.999, 35.0'
        
        # Split by comma and strip whitespace from each part
        parts = [p.strip() for p in inner.split(',')] # for example: '20.999, 35.0' -> ['20.999', '35.0']

        return (float(parts[0]), float(parts[1]))
    
    def fit(self, X, y=None):
        """
        Purpose
        -------
        Load predefined bins and WOE values from JSON metadata files for each feature in X.
        Only features with available JSON metadata will be transformed.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe. Feature names are extracted from columns.
        
        y : None
            Ignored. Present for sklearn compatibility.

        Note:
        -----
        self.feature_names_:
        List of feature names for which WOE metadata was successfully loaded.
        Example: ['feature1', 'feature2', 'feature3']

        self.bins_dict_:
        Dictionary mapping feature names to their corresponding pd.IntervalIndex of bins.
        Example:
        {
            'feature1': IntervalIndex([(-inf, 20.999], (20.999, 35.0], (35.0, 50.0], (50.0, inf)], dtype='interval[float64, right]'),
            'feature2': IntervalIndex([(-inf, 10.0], (10.0, 25.0], (25.0, 40.0], (40.0, inf)], dtype='interval[float64, right]')
        }
        self.woe_dict_:
        Dictionary mapping feature names to their corresponding bin index -> WOE value mapping.
        Example:
        {
            'feature1': {0: -0.4054651081081644, 1: 0.4054651081081644, 2: 0.0, 3: 0.0},
            'feature2': {0: -0.2231435513142097, 1: 0.2231435513142097, 2: 0.0, 3: 0.0}
        }
        
        How to detect woe from the JSON metadata:
        ============================================
        Step 1: Load the JSON metadata file
        Example: metadata = json.load(f) -> {'feature': 'age', 'woe_table': [...], ...}
        
        Step 2: Extract the woe_table list
        Example: woe_table = metadata.get("woe_table", [])
                 Result: [{'_bin': '(20.999, 35.0]', 'woe': 0.0425, ...}, 
                          {'_bin': '(35.0, 40.0]', 'woe': 0.0293, ...}, ...]
        
        Step 3: Iterate through woe_table with index
        Example: for idx, item in enumerate(woe_table):
                 idx = 0, item = {'_bin': '(20.999, 35.0]', 'woe': 0.0425, ...}
                 idx = 1, item = {'_bin': '(35.0, 40.0]', 'woe': 0.0293, ...}
        
        Step 4: Extract WOE value for each bin
        Example: woe_value = float(item["woe"])
                 idx=0 -> woe_value = 0.0425
                 idx=1 -> woe_value = 0.0293
        
        Step 5: Create mapping from bin_index to woe_value
        Example: woe_mapping = {0: 0.0425, 1: 0.0293, 2: 0.0001, ...}
        
        Step 6: During transform, map each observation to bin_index first, then lookup WOE
        Example: positions = bins.get_indexer([25.5, 37.2, 50.1])  # Get bin indices
                 Result: positions = [0, 1, -1]  # -1 means not found
                 
        Step 7: Convert bin indices to WOE values
        Example: woe_values = [woe_mapping.get(0, nan), woe_mapping.get(1, nan), nan]
                 Result: woe_values = [0.0425, 0.0293, nan]

        
        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Get all column names as feature names
        feature_names = X.columns.tolist()
        logger.info(f"Fitting WOETransformer - Features in DataFrame: {feature_names}")
        
        self.feature_names_ = []
        self.bins_dict_ = {}
        self.woe_dict_ = {}
        
        for feature_name in feature_names:
            json_path = Path(self.base_path) / feature_name / "final_auto_binning_woe.json"
            logger.info(f"Attempting to load WOE metadata for feature {feature_name} from {json_path}")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Extract bins and WOE values from woe_table
                bin_strings = [item["_bin"] for item in metadata.get("woe_table", [])]
                bin_tuples = [self._parse_bin_string(b) for b in bin_strings]
                # Example:
                # bin_strings = ['(-inf, 20.999]', '(20.999, 35.0]', '(35.0, 50.0]', '(50.0, inf)']
                # bin_tuples = [(-np.inf, 20.999), (20.999, 35.0), (35.0, 50.0), (50.0, np.inf)]
                
                # Create IntervalIndex
                interval_index = pd.IntervalIndex.from_tuples(bin_tuples)
                # Store the IntervalIndex for this feature
                # Example of interval_index: 
                # IntervalIndex([(-inf, 20.999], (20.999, 35.0], (35.0, 50.0], (50.0, inf)], dtype='interval[float64, right]')
                self.bins_dict_[feature_name] = interval_index
                
                # Create bin_index -> woe_value mapping (using woe for actual WOE values)
                # woe_mapping example:
                # {0: -0.4054651081081644, 1: 0.4054651081081644, 2: 0.0, 3: 0.0}
                woe_mapping = {}
                for idx, item in enumerate(metadata.get("woe_table", [])):
                    woe_mapping[idx] = float(item["woe"])
                
                self.woe_dict_[feature_name] = woe_mapping
                # Add feature name to the list of features that have WOE metadata
                # Example of woe dict:
                # woe_dict_ = {
                #     'feature1': {0: -0.4054651081081644, 1: 0.4054651081081644, 2: 0.0, 3: 0.0},
                #     'feature2': {0: -0.2231435513142097, 1: 0.2231435513142097, 2: 0.0, 3: 0.0}
                # }
                self.feature_names_.append(feature_name)
                
                logger.info(f"Loaded {len(bin_tuples)} bins and WOE values for feature '{feature_name}' from {json_path}")
                for idx, woe_val in woe_mapping.items():
                    logger.info(f"  Bin {idx}: WOE = {woe_val:.6f}")
                
            except FileNotFoundError:
                error_message = f"JSON metadata not found for feature '{feature_name}': {json_path}. This feature will not be transformed."
                logger.warning(error_message)
            except Exception as e:
                error_message = f"Error loading bins for feature '{feature_name}': {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)
        
        # If no features have WOE metadata, raise an error
        if not self.feature_names_:
            raise ValueError(f"No features with WOE metadata found in {self.base_path}")
        
        logger.info(f"WOETransformer fitted for {len(self.feature_names_)} features: {self.feature_names_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose
        -------
        Transform numeric features to WOE values using predefined bins and WOE mappings from JSON.
        Original feature columns are removed and replaced with WOE-transformed versions.
        Values outside any bin range are assigned the `unseen` value (default: 0).
        
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with feature columns to transform
        
        Returns
        -------
        pd.DataFrame
            DataFrame with original features replaced by WOE-transformed columns for fitted features
        """
        if not self.bins_dict_:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X_transformed = X.copy()
        
        for feature_name in self.feature_names_:
            if feature_name not in X.columns:
                logger.warning(f"Feature '{feature_name}' not found in input dataframe during transform")
                continue
            
            bins = self.bins_dict_[feature_name]
            woe_mapping = self.woe_dict_[feature_name]
            
            # Map each observation to bin index
            positions = bins.get_indexer(X[feature_name].to_numpy())
            
            # Convert bin indices to WOE values using mapping and round to 6 decimals
            woe_values = np.array([round(woe_mapping.get(pos, self.unseen), 6) if pos >= 0 else self.unseen for pos in positions])
            
            # Create WOE column with actual WOE values from JSON
            woe_col_name = f"{feature_name}_woe"
            X_transformed[woe_col_name] = woe_values
            
            # Drop original feature column
            X_transformed = X_transformed.drop(columns=[feature_name])
            
            logger.info(f"Transformed feature '{feature_name}' to WOE representation using JSON metadata and dropped original column")
        
        logger.info(f"Transform completed for {len(self.feature_names_)} features")
        return X_transformed


class WOETransformerCategorical(BaseEstimator, TransformerMixin):
    """
    Purpose
    -------
    A sklearn-compatible custom transformer that converts categorical (object dtype)
    features to Weight of Evidence (WOE) values using predefined category groups
    from JSON metadata files.

    The transformer reads bin definitions from JSON files located at:
    outputs/auto_binning_woe/{feature_name}/final_auto_binning_woe.json

    These JSON files are produced by auto_woe_binning_categorical and contain:
    - "valuesToTheGroup": mapping of generated bin name (e.g. "feature_grp0") to
      the list of original categories that belong to that group.
    - "woe_table": list of records where "_bin" is the generated bin name
      (e.g. "feature_grp0") and "woe" is the WOE value of that bin.

    Parameters
    ----------
    base_path : str, optional
        Base directory path for WOE metadata. Default is 'outputs/auto_binning_woe'
    
    unseen : float, optional
        WOE value to assign to unseen/out-of-range categories (not found in metadata during transform).
        Default is 0 (neutral value: has no effect when multiplied by model coefficients).
        Pass None to use np.nan for unseen categories instead.

    Attributes
    ----------
    category_woe_dict_ : dict
        Dictionary mapping each feature to a {category_value: woe_value} mapping.
        Example:
        {
            'marital_status': {'single': 0.42, 'married': -0.31, 'divorced': -0.31}
        }
    feature_names_ : list
        List of feature names for which WOE metadata was successfully loaded.
    """

    def __init__(self, base_path: str = "outputs/auto_binning_woe", unseen: float = 0):
        self.base_path = base_path
        self.unseen = unseen
        self.category_woe_dict_ = {}   # {feature_name: {category_value: woe_value}}
        self.feature_names_ = []

    def fit(self, X, y=None):
        """
        Purpose
        -------
        Load predefined category groups and WOE values from JSON metadata files for
        each feature in X. Only features with available JSON metadata will be
        transformed.

        For each feature, the method builds a direct mapping from each original
        category value to the WOE value of the group it belongs to.

        How the mapping is built from the JSON metadata:
        ================================================
        Step 1: Load the JSON metadata file
        Example: metadata = json.load(f)

        Step 2: Build bin_name -> woe mapping from woe_table
        Example: woe_table = [
                     {'_bin': 'feature_grp2', 'woe': 1.3839, ...},
                     {'_bin': 'feature_grp1', 'woe': 0.1748, ...}, ...
                 ]
                 bin_woe = {'feature_grp2': 1.3839, 'feature_grp1': 0.1748, ...}

        Step 3: Use valuesToTheGroup to map each category to its bin's WOE
        Example: valuesToTheGroup = {'feature_grp2': ['-1', '2'], 'feature_grp1': ['1'], ...}
                 category_woe = {'-1': 1.3839, '2': 1.3839, '1': 0.1748, ...}

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe. Feature names are extracted from columns.

        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        feature_names = X.columns.tolist()
        logger.info(f"Fitting WOETransformerCategorical - Features in DataFrame: {feature_names}")

        self.feature_names_ = []
        self.category_woe_dict_ = {}

        for feature_name in feature_names:
            json_path = Path(self.base_path) / feature_name / "final_auto_binning_woe.json"
            logger.info(f"Attempting to load categorical WOE metadata for feature '{feature_name}' from {json_path}")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Build bin_name -> woe mapping from woe_table
                # Example: {'feature_grp2': 1.3839, 'feature_grp1': 0.1748, ...}
                bin_woe = {
                    item["_bin"]: float(item["woe"])
                    for item in metadata.get("woe_table", [])
                }

                # Build category_value -> woe mapping using valuesToTheGroup
                # Example: {'-1': 1.3839, '2': 1.3839, '1': 0.1748, ...}
                values_to_group = metadata.get("valuesToTheGroup", {})
                category_woe = {}
                for bin_name, categories in values_to_group.items():
                    woe_value = bin_woe.get(bin_name)
                    if woe_value is None:
                        logger.warning(f"No WOE value found for bin '{bin_name}' in feature '{feature_name}'")
                        continue
                    for category in categories:
                        # Store the category exactly as it appears in the JSON
                        category_woe[category] = woe_value

                if not category_woe:
                    logger.warning(f"No category WOE mapping built for feature '{feature_name}'. Skipping.")
                    continue

                self.category_woe_dict_[feature_name] = category_woe
                self.feature_names_.append(feature_name)

                logger.info(f"Loaded {len(category_woe)} category WOE mappings for feature '{feature_name}' from {json_path}")
                for category, woe_val in category_woe.items():
                    logger.info(f"  Category '{category}': WOE = {woe_val:.6f}")

            except FileNotFoundError:
                error_message = f"JSON metadata not found for feature '{feature_name}': {json_path}. This feature will not be transformed."
                logger.warning(error_message)
            except Exception as e:
                error_message = f"Error loading categorical bins for feature '{feature_name}': {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)

        if not self.feature_names_:
            raise ValueError(f"No categorical features with WOE metadata found in {self.base_path}")

        logger.info(f"WOETransformerCategorical fitted for {len(self.feature_names_)} features: {self.feature_names_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Purpose
        -------
        Transform categorical features to WOE values using predefined category-to-WOE
        mappings from JSON. Original feature columns are removed and replaced with
        WOE-transformed versions.

        Categories that were not seen during fit (i.e. not present in the JSON
        metadata) are mapped to the `unseen` value (default: 0).

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with feature columns to transform

        Returns
        -------
        pd.DataFrame
            DataFrame with original features replaced by WOE-transformed columns
            for fitted features.
        """
        if not self.category_woe_dict_:
            raise ValueError("Transformer not fitted. Call fit() first.")

        X_transformed = X.copy()

        for feature_name in self.feature_names_:
            if feature_name not in X.columns:
                logger.warning(f"Feature '{feature_name}' not found in input dataframe during transform")
                continue

            category_woe = self.category_woe_dict_[feature_name]
            logger.info(f"Transforming feature '{feature_name}' using category WOE mapping with {len(category_woe)} categories")

            # Map each observation's category to its WOE value.
            # Cast to str so that the lookup matches the JSON keys regardless of
            # the original dtype (the JSON stores categories as written).
            woe_values = []
            unseen_categories = []
            for value in X[feature_name].astype(str):
                if value in category_woe:
                    woe_values.append(round(category_woe[value], 6))
                else:
                    woe_values.append(self.unseen)
                    if value not in unseen_categories:
                        unseen_categories.append(value)
            woe_values = pd.Series(woe_values, index=X.index)

            # Log about unseen categories
            if len(unseen_categories) > 0:
                if self.unseen is None or (isinstance(self.unseen, float) and np.isnan(self.unseen)):
                    logger.warning(f"Feature '{feature_name}' has unseen categories mapped to NaN: {unseen_categories}")
                else:
                    logger.info(f"Feature '{feature_name}' has unseen categories mapped to {self.unseen}: {unseen_categories}")

            woe_col_name = f"{feature_name}_woe"
            X_transformed[woe_col_name] = woe_values

            # Drop original feature column
            X_transformed = X_transformed.drop(columns=[feature_name])

            logger.info(f"Transformed categorical feature '{feature_name}' to WOE representation using JSON metadata and dropped original column")

        logger.info(f"Transform completed for {len(self.feature_names_)} categorical features")
        return X_transformed


