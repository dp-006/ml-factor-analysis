'''
Accelera Consulting

Main Training Pipeline for MLOps Project
'''

# Close some specific warnings that are not relevant for this pipeline
import warnings

# Ignore warnings about the number of unique categories in feature-engine encoding, as we are handling it with our custom LowCardinalityHandler
warnings.filterwarnings(
    action='ignore', 
    message='The number of unique categories', 
    category=UserWarning, 
    module='feature_engine.encoding'
    )

# feature-engine preprocessing
from feature_engine.selection import DropFeatures
from feature_engine.selection import DropConstantFeatures
from feature_engine.selection import DropDuplicateFeatures
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer

# feature-engine selection
from feature_engine.selection import RecursiveFeatureAddition
from feature_engine.selection import DropHighPSIFeatures

# sklearn models and pipeline tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Custom Missing Columns Handler
from sklearn_wrapper_missing_handler import MissingColumnsHandler

# Custom Infinite Columns Handler
from sklearn_wrapper_infinite_handler import InfiniteColumnsHandler

# Custom Low Cardinality Handler
from sklearn_wrapper_cardinality_handler import LowCardinalityHandler

# Custom Data Type Handler
from sklearn_wrapper_datatypes import DataTypeConverter

# Custom WOE Transformers (computes WOE during fit)
from sklearn_wrapper_woe_numeric import WOETransformerNumeric
from sklearn_wrapper_woe_categorical import WOETransformerCategorical

# Custom VIF Transformer
from sklearn_wrapper_vif import VIFTransformer

# Custom Logistic Regression Predictor
from sklearn_wrapper_logistic import LogisticRegressionWrapper

# Helper functions
from helper import (
    split_train_test,
    generate_sample_data,
)

# Logging
from logging_config.logger_config import get_logger
logger_name = "mlops.ml_training"
logger_file_name = "ml_training.log"
logger = get_logger(logger_name, logger_file_name)


# ============================================================================
# DEFAULT CONFIGURATIONS
# ============================================================================

DEFAULT_CONFIG = {
    'winsorizer': {
        'capping_method': 'quantiles',
        'tail': 'right',
        'fold': 0.01,
        'add_indicators': True,
        'missing_values': 'ignore'
    },
    'rare_label_encoder': {
        'tol': 0.01,
        'n_categories': 10,
        'replace_with': 'Rare',
        'missing_values': 'ignore'
    },
    'arbitrary_number_imputer': {
        'arbitrary_number': -999999
    },
    'categorical_imputer': {
        'imputation_method': 'missing',
        'fill_value': 'Missing'
    },
    'woe_numeric': {
        'target_col': 'TARGET',
        'initial_bins': 20,
        'min_bin_pct': 0.05,
        'max_final_bins': 10,
        'min_final_bins': 3,
        'min_iv': 0.02,
        'max_iv': 0.50,
        'max_iter': 25
    },
    'woe_categorical': {
        'target_col': 'TARGET',
        'min_bin_pct': 0.05,
        'max_final_bins': 6,
        'min_final_bins': 2,
        'min_iv': 0.02,
        'max_iv': 0.50,
        'max_iter': 20
    },
    'psi_filter': {
        'threshold': 0.25
    },
    'vif_filter': {
        'vif_threshold': 5.0,
        'output_dir': 'outputs/vif_analysis'
    },
    'rfa_filter': {
        'cv': 2,
        'threshold': 0.0001,
        'scoring': 'roc_auc'
    },
    'logistic_regression': {
        'disp': False,
        'method': 'bfgs',
        'maxiter': 500,
        'model_dir': 'outputs/logitmodel',
        'metrics_dir': 'outputs/logitmodel/metrics',
        'save_results': True,
        'save_model': True,
        'evaluate_model': True,
        'evaluate_bins': 20,
        'threshold': 0.5
    }
}

# ============================================================================
#  SUB-PIPELINES
# ============================================================================

def data_cleaning_pipeline():
    """
    Purpose
    -------
    Creates a data cleaning pipeline.
    Handles missing columns, constant features, duplicate columns, and infinite values.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Data cleaning pipeline ready to fit and transform
    """
    
    pipeline = Pipeline(steps=[
        ('missing_handler', MissingColumnsHandler(threshold=0.5)),
        ('constant_handler', DropConstantFeatures(tol=1.0, missing_values='raise')),
        ('duplicate_handler', DropDuplicateFeatures()),
        ('infinite_handler', InfiniteColumnsHandler())
    ])
    
    return pipeline

def low_cardinality_pipeline():
    """
    Purpose
    -------
    Creates a low cardinality handling pipeline for Step 2.
    Converts numeric columns with few unique values to object type.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Low cardinality pipeline ready to fit and transform
    """
    
    pipeline = Pipeline(steps=[
        ('low_cardinality_handler', LowCardinalityHandler(threshold=10, add_suffix=True, suffix="s")),
        ('data_type_converter', DataTypeConverter())
    ])
    
    return pipeline

def outlier_and_missing_pipeline(config):
    """
    Purpose
    -------
    Creates a pipeline for Steps 3 and 4: Outlier Detection and Missing Value Imputation.
    - Detect outliers in numeric columns using Winsorizer and in categorical columns using RareLabelEncoder.
    - Impute missing values in numeric columns using ArbitraryNumberImputer and in categorical columns using CategoricalImputer.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing parameters for each step:
        - 'winsorizer': dict with capping_method, tail, fold, add_indicators, missing_values
        - 'rare_label_encoder': dict with tol, n_categories, replace_with, missing_values
        - 'arbitrary_number_imputer': dict with arbitrary_number
        - 'categorical_imputer': dict with imputation_method, fill_value
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Outlier and missing value imputation pipeline ready to fit and transform

    """
    pipeline = Pipeline(steps=[
        # OUTLIER DETECTION FOR NUMERIC COLUMNS
        ('winsorizer', Winsorizer(
            capping_method=config['winsorizer'].get('capping_method', 'quantiles'),
            tail=config['winsorizer'].get('tail', 'right'),
            fold=config['winsorizer'].get('fold', 0.01),
            add_indicators=config['winsorizer'].get('add_indicators', True),
            missing_values=config['winsorizer'].get('missing_values', 'ignore')
        )),
        # OUTLIER DETECTION FOR CATEGORICAL COLUMNS
        ('rare_label_encoder', RareLabelEncoder(
            tol=config['rare_label_encoder'].get('tol', 0.01),
            n_categories=config['rare_label_encoder'].get('n_categories', 10),
            max_n_categories=None,
            replace_with=config['rare_label_encoder'].get('replace_with', 'Rare'),
            missing_values=config['rare_label_encoder'].get('missing_values', 'ignore')
        )),
        # MISSING VALUE IMPUTATION FOR NUMERIC COLUMNS
        ('arbitrary_number_imputer', ArbitraryNumberImputer(
            arbitrary_number=config['arbitrary_number_imputer'].get('arbitrary_number', -999999)
        )),
        # MISSING VALUE IMPUTATION FOR CATEGORICAL COLUMNS
        ('categorical_imputer', CategoricalImputer(
            imputation_method=config['categorical_imputer'].get('imputation_method', 'missing'),
            fill_value=config['categorical_imputer'].get('fill_value', 'Missing')
        ))
    ])
    
    return pipeline

def woe_transformers_pipeline(config):
    """
    Purpose
    -------
    Creates WOE transformers for numeric and categorical features.
    These transformers will accept X and y separately during fit().
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary containing 'woe_numeric' and 'woe_categorical' keys.
        If None, uses DEFAULT_CONFIG.
    
    Returns
    -------
    tuple
        (WOETransformerNumeric, WOETransformerCategorical)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    woe_numeric_config = config.get('woe_numeric', {})
    woe_categorical_config = config.get('woe_categorical', {})
    
    woe_numeric = WOETransformerNumeric(
        target_col=woe_numeric_config.get('target_col', 'TARGET'),
        initial_bins=woe_numeric_config.get('initial_bins', 20),
        min_bin_pct=woe_numeric_config.get('min_bin_pct', 0.05),
        max_final_bins=woe_numeric_config.get('max_final_bins', 10),
        min_final_bins=woe_numeric_config.get('min_final_bins', 3),
        min_iv=woe_numeric_config.get('min_iv', 0.02),
        max_iv=woe_numeric_config.get('max_iv', 0.50),
        max_iter=woe_numeric_config.get('max_iter', 25)
    )
    
    woe_categorical = WOETransformerCategorical(
        target_col=woe_categorical_config.get('target_col', 'TARGET'),
        min_bin_pct=woe_categorical_config.get('min_bin_pct', 0.05),
        max_final_bins=woe_categorical_config.get('max_final_bins', 6),
        min_final_bins=woe_categorical_config.get('min_final_bins', 2),
        min_iv=woe_categorical_config.get('min_iv', 0.02),
        max_iv=woe_categorical_config.get('max_iv', 0.50),
        max_iter=woe_categorical_config.get('max_iter', 20)
    )

    # Each transformer internally selects its own dtypes and returns a DataFrame,
    # so a plain Pipeline suffices — no ColumnTransformer needed.
    sklearn_pipeline = Pipeline(steps=[
        # WOE TRANSFORMERS FOR NUMERIC FEATURES
        ('woe_numeric', woe_numeric),
        # WOE TRANSFORMERS FOR CATEGORICAL FEATURES
        ('woe_categorical', woe_categorical)
    ])
    
    return sklearn_pipeline

def feature_filtering_pipeline(config):
    """
    Purpose
    -------
    Creates a feature filtering pipeline for Steps 6, 7 and 8:
    - Drop unstable features using PSI (DropHighPSIFeatures).
    - Remove multicollinear features using VIF (VIFTransformer).
    - Select features using Recursive Feature Addition with RandomForestClassifier.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - 'psi_filter': dict with threshold
        - 'vif_filter': dict with vif_threshold
        - 'rfa_filter': dict with cv, threshold, scoring

    Returns
    -------
    sklearn.pipeline.Pipeline
        Feature filtering pipeline ready to fit and transform
    """
    psi_config = config.get('psi_filter', {})
    vif_config = config.get('vif_filter', {})
    rfa_config = config.get('rfa_filter', {})

    pipeline = Pipeline(steps=[
        # DROP HIGH PSI FEATURES
        ('psi_filter', DropHighPSIFeatures(
            threshold=psi_config.get('threshold', 0.25)
        )),
        # DECREASE MULTICOLLINEARITY WITH VIF
        ('vif_filter', VIFTransformer(
            vif_threshold=vif_config.get('vif_threshold', 5.0),
            output_dir=vif_config.get('output_dir', None)
        )),
        # RECURSIVE FEATURE ADDITION
        ('rfa_filter', RecursiveFeatureAddition(
            RandomForestClassifier(random_state=42),
            cv=rfa_config.get('cv', 2),
            threshold=rfa_config.get('threshold', 0.0001),
            scoring=rfa_config.get('scoring', 'roc_auc')
        ))
    ])

    return pipeline

def logistic_regression_pipeline(config):
    """
    Purpose
    -------
    Creates a logistic regression estimator for Step 9: Model Training.
    Uses LogisticRegressionWrapper (statsmodels backend) as the final estimator
    in a single-step sklearn Pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing a 'logistic_regression' key with:
        - 'disp'          : bool   — show convergence messages
        - 'method'        : str    — optimization method ('bfgs', 'newton', ...)
        - 'maxiter'       : int    — max iterations
        - 'model_dir'     : str    — directory to save model artifacts
        - 'save_results'  : bool   — save JSON result files
        - 'save_model'    : bool   — save pickled model
        - 'evaluate_model': bool   — run post-fit evaluation
        - 'evaluate_bins' : int    — decile bins for evaluation
        - 'threshold'     : float  — classification threshold

    Returns
    -------
    sklearn.pipeline.Pipeline
        Single-step pipeline wrapping LogisticRegressionWrapper, ready to fit.
    """
    lr_config = config.get('logistic_regression', {})

    pipeline = Pipeline(steps=[
        ('logistic_regression', LogisticRegressionWrapper(
            disp=lr_config.get('disp', False),
            method=lr_config.get('method', 'bfgs'),
            maxiter=lr_config.get('maxiter', 500),
            model_dir=lr_config.get('model_dir', 'outputs/logitmodel'),
            metrics_dir=lr_config.get('metrics_dir', None),
            save_results=lr_config.get('save_results', True),
            save_model=lr_config.get('save_model', True),
            evaluate_model=lr_config.get('evaluate_model', True),
            evaluate_bins=lr_config.get('evaluate_bins', 20),
            threshold=lr_config.get('threshold', 0.5)
        ))
    ])

    return pipeline

# ============================================================================
#  ONE PIPELINE TO RUN ALL SUB-PIPESLINES IN SEQUENCE
# ============================================================================

def full_training_pipeline(config):
    """
    Purpose
    -------
    Creates a full training pipeline that sequentially executes all the sub-pipelines:
    1. Data Cleaning Pipeline
    2. Low Cardinality Pipeline
    3. Outlier Detection and Missing Value Imputation Pipeline
    4. WOE Feature Engineering Pipeline
    5. Feature Filtering Pipeline
    6. Logistic Regression Pipeline

    Parameters
    ----------
    config : dict
        Configuration dictionary containing parameters for each sub-pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Full training pipeline ready to fit and transform

    Pipeline Structure
    ------------------
    Pipeline(steps=[
        ('data_cleaning', Pipeline(steps=[
            ('missing_handler',   MissingColumnsHandler()),
            ('constant_handler',  DropConstantFeatures()),
            ('duplicate_handler', DropDuplicateFeatures()),
            ('infinite_handler',  InfiniteColumnsHandler())
        ])),
        ('low_cardinality', Pipeline(steps=[
            ('low_cardinality_handler', LowCardinalityHandler()),
            ('data_type_converter',     DataTypeConverter())
        ])),
        ('outlier_and_missing', Pipeline(steps=[
            ('winsorizer',               Winsorizer()),
            ('rare_label_encoder',       RareLabelEncoder()),
            ('arbitrary_number_imputer', ArbitraryNumberImputer()),
            ('categorical_imputer',      CategoricalImputer())
        ])),
        ('woe_transformers', Pipeline(steps=[
            ('woe_numeric',      WOETransformerNumeric()),
            ('woe_categorical',  WOETransformerCategorical())
        ])),
        ('feature_filtering', Pipeline(steps=[
            ('psi_filter', DropHighPSIFeatures()),
            ('vif_filter', VIFTransformer()),
            ('rfa_filter', RecursiveFeatureAddition())
        ])),
        ('logistic_regression', Pipeline(steps=[
            ('logistic_regression', LogisticRegressionWrapper())
        ]))
    ])
    """

    full_pipeline = Pipeline(steps=[
        ('data_cleaning', data_cleaning_pipeline()),
        ('low_cardinality', low_cardinality_pipeline()),
        ('outlier_and_missing', outlier_and_missing_pipeline(config)),
        ('woe_transformers', woe_transformers_pipeline(config)),
        ('feature_filtering', feature_filtering_pipeline(config)),
        ('logistic_regression', logistic_regression_pipeline(config))
    ])

    logger.info("Full training pipeline created with all sub-pipelines in sequence")
    logger.info("=" * 80)
    logger.info(f"\n\nFull pipeline\n\n{full_pipeline}\n")
    logger.info("=" * 80)
    
    return full_pipeline




if __name__ == "__main__":

    # ============================================================================
    # FUNCTION BASED PIPELINE EXECUTION
    # ============================================================================

    X, y = generate_sample_data()

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.33, random_state=42, stratify=True)

    # Build and run the full training pipeline
    full_pipeline = full_training_pipeline(DEFAULT_CONFIG)
    full_pipeline.fit(X_train, y_train)

    # Predictions on test set with one sample of output review
    sample = X_test.iloc[0:1]
    sample_pred = full_pipeline.predict(sample)
    sample_prob = full_pipeline.predict_proba(sample)
    logger.info(f"Sample prediction for first test row: {sample_pred[0]} with probability {sample_prob[0][1]:.4f}")

    # # STEP 1: Data Cleaning Pipeline
    # pipeline_1 = data_cleaning_pipeline()
    # X_train = pipeline_1.fit_transform(X_train)
    # X_test = pipeline_1.transform(X_test)
    # list(map(lambda x: quality_function_for_pandas(x), [X_train, X_test, y_train, y_test]))

    # # STEP 2: Low Cardinality Pipeline
    # pipeline_2 = low_cardinality_pipeline()
    # X_train = pipeline_2.fit_transform(X_train)
    # X_test = pipeline_2.transform(X_test)
    # list(map(lambda x: quality_function_for_pandas(x), [X_train, X_test, y_train, y_test]))

    # # STEPS 3 & 4: Outlier Detection and Missing Value Imputation Pipeline
    # pipeline_3 = outlier_and_missing_pipeline(DEFAULT_CONFIG)
    # X_train = pipeline_3.fit_transform(X_train)
    # X_test = pipeline_3.transform(X_test)
    # list(map(lambda x: quality_function_for_pandas(x), [X_train, X_test, y_train, y_test]))

    # # STEP 5: WOE Feature Engineering
    # woe_pipeline = woe_transformers_pipeline(DEFAULT_CONFIG)
    # X_train = woe_pipeline.fit_transform(X_train, y_train)
    # X_test = woe_pipeline.transform(X_test)
    # list(map(lambda x: quality_function_for_pandas(x), [X_train, X_test, y_train, y_test]))

    # # print woe fit attributes for numeric and categorical features to review the binning results and WOE values
    # woe_numeric = woe_pipeline.named_steps['woe_numeric']
    # woe_categorical = woe_pipeline.named_steps['woe_categorical']

    # # --- Numeric WOE review ---
    # logger.info(f"WOE numeric features fitted ({len(woe_numeric.feature_names_)}): {woe_numeric.feature_names_}")
    # logger.info("WOE numeric bins_dict_ (first 2 features):")
    # for feature, intervals in list(woe_numeric.bins_dict_.items())[:2]:
    #     logger.info(f"  {feature}: {intervals}")
    # logger.info("WOE numeric woe_dict_ (first 2 features):")
    # for feature, woe_map in list(woe_numeric.woe_dict_.items())[:2]:
    #     logger.info(f"  {feature}: {woe_map}")
    # logger.info("WOE numeric binning_results_ (first 2 features, selected keys):")
    # for feature, result in list(woe_numeric.binning_results_.items())[:2]:
    #     logger.info(f"  {feature} -> status={result.get('status')} | totalIv={result.get('totalIv')} | numberofBins={result.get('numberofBins')}")

    # # --- Categorical WOE review ---
    # logger.info(f"WOE categorical features fitted ({len(woe_categorical.feature_names_)}): {woe_categorical.feature_names_}")
    # logger.info("WOE categorical category_woe_dict_ (first 2 features):")
    # for feature, cat_map in list(woe_categorical.category_woe_dict_.items())[:2]:
    #     preview = dict(list(cat_map.items())[:2])
    #     logger.info(f"  {feature} (showing 2 of {len(cat_map)} categories): {preview}")
    # logger.info("WOE categorical binning_results_ (first 2 features, selected keys):")
    # for feature, result in list(woe_categorical.binning_results_.items())[:2]:
    #     logger.info(f"  {feature} -> status={result.get('status')} | totalIv={result.get('totalIv')} | numberofBins={result.get('numberofBins')}")

    # # STEPS 6, 7 & 8: Feature Filtering Pipeline
    # pipeline_5 = feature_filtering_pipeline(DEFAULT_CONFIG)
    # X_train = pipeline_5.fit_transform(X_train, y_train)
    # X_test = pipeline_5.transform(X_test)
    # list(map(lambda x: quality_function_for_pandas(x), [X_train, X_test, y_train, y_test]))

    # # --- VIF detailed review ---
    # vif_filter = pipeline_5.named_steps['vif_filter']

    # logger.info(f"VIF features_to_keep_: {vif_filter.features_to_keep_}")

    # # iteration_steps_ first entry (full raw structure)
    # first_step_key = list(vif_filter.iteration_steps_.keys())[0]
    # logger.info(f"VIF iteration_steps_ (first): {{{first_step_key}: {vif_filter.iteration_steps_[first_step_key]}}}")

    # # vif_results_ first entry (full raw structure)
    # first_result_key = list(vif_filter.vif_results_.keys())[0]
    # logger.info(f"VIF vif_results_ (first): {{{first_result_key}: {vif_filter.vif_results_[first_result_key]}}}")

    # # STEP 9: Logistic Regression Pipeline
    # pipeline_6 = logistic_regression_pipeline(DEFAULT_CONFIG)
    # pipeline_6.fit(X_train, y_train)

    # # --- Logistic Regression review ---
    # log_reg = pipeline_6.named_steps['logistic_regression']
    # logger.info(f"Logistic Regression model fitted: {log_reg.model_fit_.summary()}")
    # logger.info(f"Logistic Regression n_features_in_: {log_reg.n_features_in_}")
    # logger.info(f"Logistic Regression classes_: {log_reg.classes_}")
