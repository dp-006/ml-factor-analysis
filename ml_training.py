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

# Ignore pandas datetime format inference warning triggered by feature-engine's internal variable type detection
warnings.filterwarnings(
    action='ignore',
    message='Could not infer format',
    category=UserWarning,
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
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.selection import SelectByTargetMeanPerformance

# sklearn models and pipeline tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# Custom Missing Columns Handler
from sklearn_wrapper_missing_handler import MissingColumnsHandler

# Custom Infinite Columns Handler
from sklearn_wrapper_infinite_handler import InfiniteColumnsHandler

# Custom Low Cardinality Handler
from sklearn_wrapper_cardinality_handler import LowCardinalityHandler

# Custom Data Type Handler
from sklearn_wrapper_datatypes import DataTypeConverter

# Custom Binary Column Handler
from sklearn_wrapper_binary import BinaryColumnConverter

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
    apply_random_undersampling,
    quality_function_for_pandas
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
    'missing_handler': {
        'threshold': 0.25
    },
    'constant_handler': {
        'tol': 1.0,
        'missing_values': 'ignore'
    },
    'winsorizer': {
        'capping_method': 'quantiles',
        'tail': 'right',
        'fold': 0.01,
        'add_indicators': False,
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
        'max_iter': 25,
        'skip_indicator': False
    },
    'woe_categorical': {
        'target_col': 'TARGET',
        'min_bin_pct': 0.05,
        'max_final_bins': 6,
        'min_final_bins': 2,
        'min_iv': 0.02,
        'max_iv': 0.50,
        'max_iter': 20,
        'skip_indicator': False
    },
    'sfp_filter': {
        'cv': 2,
        'threshold': 0.55,
        'scoring': 'roc_auc'
    },
    'stmp_filter': {
        'bins': 5,
        'strategy': 'equal_width',
        'scoring': 'roc_auc',
        'cv': 3,
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

def data_cleaning_pipeline(config=None):
    """
    Purpose
    -------
    Creates a data cleaning pipeline.
    - Binary Column Conversion: Converts binary columns (0/1) to categorical (YES/NO).
    - Missing Columns Handler: Removes columns with a high percentage of missing values.
    - Drop Constant Features: Removes columns with constant values.
    - Drop Duplicate Features: Removes duplicate columns.
    - Drop Infinite Columns: Removes columns with infinite values.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Data cleaning pipeline ready to fit and transform
    """
    if config is None:
        config = DEFAULT_CONFIG

    missing_config = config.get('missing_handler', {})
    constant_config = config.get('constant_handler', {})

    pipeline = Pipeline(steps=[
        # Binary Handler
        ('binary_handler', BinaryColumnConverter()),

        # Missing Columns Handler
        ('missing_handler', MissingColumnsHandler(
            threshold=missing_config.get('threshold', 0.5)
        )),
        # Drop Constant Features (tolerance = 1.0 means all values are the same, ignoring NaNs)
        ('constant_handler', DropConstantFeatures(
            tol=constant_config.get('tol', 1.0),
            missing_values=constant_config.get('missing_values', 'ignore')
        )),

        # Drop Duplicate Features (keeps the first occurrence of each duplicate)
        ('duplicate_handler', DropDuplicateFeatures()),

        # Drop Infinite Columns (removes columns with any infinite values)
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
        ('low_cardinality_handler', LowCardinalityHandler(threshold=10, add_suffix=False, suffix="s")),
        ('data_type_converter', DataTypeConverter())
    ])
    
    return pipeline

def outlier_and_missing_pipeline(config):
    """
    Purpose
    -------
    Creates a pipeline for Steps 3 and 4: Outlier Detection and Missing Value Imputation.
    Uses a ColumnTransformer to explicitly route numeric and object columns to
    their respective sub-pipelines, preventing dtype ambiguity issues with
    feature-engine's auto-detection.

    Numeric branch  : Winsorizer → ArbitraryNumberImputer
    Categorical branch: RareLabelEncoder → CategoricalImputer
    
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
    numeric_pipeline = Pipeline(steps=[
        # OUTLIER DETECTION FOR NUMERIC COLUMNS
        ('winsorizer', Winsorizer(
            capping_method=config['winsorizer'].get('capping_method', 'quantiles'),
            tail=config['winsorizer'].get('tail', 'right'),
            fold=config['winsorizer'].get('fold', 0.01),
            add_indicators=config['winsorizer'].get('add_indicators', True),
            missing_values=config['winsorizer'].get('missing_values', 'ignore')
        )),
        # MISSING VALUE IMPUTATION FOR NUMERIC COLUMNS
        # ('arbitrary_number_imputer', ArbitraryNumberImputer(
        #     arbitrary_number=config['arbitrary_number_imputer'].get('arbitrary_number', -999999)
        # )),
    ])

    categorical_pipeline = Pipeline(steps=[
        # OUTLIER DETECTION FOR CATEGORICAL COLUMNS
        ('rare_label_encoder', RareLabelEncoder(
            tol=config['rare_label_encoder'].get('tol', 0.01),
            n_categories=config['rare_label_encoder'].get('n_categories', 10),
            max_n_categories=None,
            replace_with=config['rare_label_encoder'].get('replace_with', 'Rare'),
            missing_values=config['rare_label_encoder'].get('missing_values', 'ignore')
        )),
        # MISSING VALUE IMPUTATION FOR CATEGORICAL COLUMNS
        # ('categorical_imputer', CategoricalImputer(
        #     imputation_method=config['categorical_imputer'].get('imputation_method', 'missing'),
        #     fill_value=config['categorical_imputer'].get('fill_value', 'Missing')
        # )),
    ])

    ct = ColumnTransformer(
        transformers=[
            ('numeric_pipeline', numeric_pipeline, make_column_selector(dtype_exclude=object)),
            ('categorical_pipeline', categorical_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipeline = Pipeline(steps=[
        ('column_transformer', ct)
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
        max_iter=woe_numeric_config.get('max_iter', 25),
        skip_indicator=woe_numeric_config.get('skip_indicator', False)
    )
    
    woe_categorical = WOETransformerCategorical(
        target_col=woe_categorical_config.get('target_col', 'TARGET'),
        min_bin_pct=woe_categorical_config.get('min_bin_pct', 0.05),
        max_final_bins=woe_categorical_config.get('max_final_bins', 6),
        min_final_bins=woe_categorical_config.get('min_final_bins', 2),
        min_iv=woe_categorical_config.get('min_iv', 0.02),
        max_iv=woe_categorical_config.get('max_iv', 0.50),
        max_iter=woe_categorical_config.get('max_iter', 20),
        skip_indicator=woe_categorical_config.get('skip_indicator', False)
    )

    # Each transformer internally selects its own dtypes and returns a DataFrame,
    # so a plain Pipeline suffices — no ColumnTransformer needed.
    sklearn_pipeline = Pipeline(steps=[
        # Binary Conversion (Because Indicators from Winsorizer may create new binary columns)
        ('binary_handler', BinaryColumnConverter()),

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

def sfp_feature_filtering_pipeline(config):
    """
    Purpose
    -------
    Creates a feature filtering pipeline that only applies SelectBySingleFeaturePerformance.
    This is used for the Stability Feature Performance (SFP) analysis in Step 7.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'rfa_filter' key with:
        - cv: int (cross-validation folds)
        - threshold: float (performance improvement threshold)
        - scoring: str (scoring metric)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with SelectBySingleFeaturePerformance ready to fit and transform
    """
    sfp_config = config.get('sfp_filter', {})

    pipeline = Pipeline(steps=[
        ('sfp_filter', SelectBySingleFeaturePerformance(
            estimator=RandomForestClassifier(random_state=42),
            cv=sfp_config.get('cv', 2),
            threshold=sfp_config.get('threshold', 0.0001),
            scoring=sfp_config.get('scoring', 'roc_auc')
        ))
    ])

    return pipeline

def target_mean_performance_pipeline(config):
    """
    Purpose
    -------
    Creates a pipeline that filters categorical features using SelectByTargetMeanPerformance.
    Only categorical (object dtype) columns are evaluated and potentially dropped;
    all numeric columns are passed through unchanged.
    Uses ColumnTransformer with make_column_selector to restrict selection to categoricals only.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'stmp_filter' key with:
        - bins      : int          (number of intervals for target mean encoding)
        - strategy  : str          ('equal_width' or 'equal_frequency')
        - scoring   : str          (scoring metric, e.g. 'roc_auc')
        - cv        : int          (cross-validation folds)
        - threshold : float | None (min performance to keep; None = mean of all features)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline wrapping a ColumnTransformer that applies SelectByTargetMeanPerformance
        to categorical columns and passes numeric columns through unchanged.
    """
    stmp_config = config.get('stmp_filter', {})

    ct = ColumnTransformer(
        transformers=[
            ('stmp_filter', SelectByTargetMeanPerformance(
                bins=stmp_config.get('bins', 5),
                strategy=stmp_config.get('strategy', 'equal_width'),
                scoring=stmp_config.get('scoring', 'roc_auc'),
                cv=stmp_config.get('cv', 3),
                regression=False
            ), make_column_selector(dtype_include=object))
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipeline = Pipeline(steps=[
        ('stmp_filter', ct)
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
    4. Stability Feature Performance (SFP) Filtering Pipeline
    5. SelectByTargetMeanPerformance (STMP) Filtering Pipeline
    6. WOE Feature Engineering Pipeline
    7. Feature Filtering Pipeline
    8. Logistic Regression Pipeline

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
            ('binary_handler', BinaryColumnConverter()),
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
        ('sfp_filter', Pipeline(steps=[
            ('sfp_filter', SelectBySingleFeaturePerformance())
        ])),
        ('stmp_filter', ColumnTransformer(transformers=[
            ('stmp_filter', SelectByTargetMeanPerformance(), make_column_selector(dtype_include=object))
        ], remainder='passthrough')),
        ('woe_transformers', Pipeline(steps=[
            ('woe_numeric',      WOETransformerNumeric()),
            ('woe_categorical',  WOETransformerCategorical())
        ])),
        ('feature_filtering', Pipeline(steps=[
            ('psi_filter', DropHighPSIFeatures()),
            ('vif_filter', VIFTransformer()),
            ('rfa_filter', RecursiveFeatureAddition())
        ])),
        ('ml_model', Pipeline(steps=[
            ('logistic_regression', LogisticRegressionWrapper())
        ]))
    ])
    """

    full_pipeline = Pipeline(steps=[
        ('data_cleaning', data_cleaning_pipeline(config)),
        ('low_cardinality', low_cardinality_pipeline()),
        ('outlier_and_missing', outlier_and_missing_pipeline(config)),
        ('sfp_filter', sfp_feature_filtering_pipeline(config)),
        ('stmp_filter', target_mean_performance_pipeline(config)),
        ('woe_transformers', woe_transformers_pipeline(config)),
        ('feature_filtering', feature_filtering_pipeline(config)),
        ('ml_model', logistic_regression_pipeline(config))
    ], verbose=True)

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
    # Random make %1 of the payment_amount_apr_2005 column null
    null_idx = X['payment_amount_apr_2005'].sample(frac=0.01, random_state=42).index
    X.loc[null_idx, 'payment_amount_apr_2005'] = None

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.33, random_state=42, stratify=True)

    # Sample data quality check before training
    logger.info("=" * 80)
    X_train, y_train = apply_random_undersampling(X_train, y_train, sampling_strategy='auto', random_state=42)
    logger.info("=" * 80)

    # # Build and run the full training pipeline
    # full_pipeline = full_training_pipeline(DEFAULT_CONFIG)
    # full_pipeline.fit(X_train, y_train)

    # # Log removed features after each filtering step
    # logger.info("=" * 80)
    # logger.info("Logging removed features after each filtering step:")
    # for step_name, step in full_pipeline.named_steps.items():
    #     if step_name == 'data_cleaning':
    #         missing_removed   = step.named_steps['missing_handler'].missing_columns_
    #         constant_removed  = step.named_steps['constant_handler'].features_to_drop_
    #         duplicate_removed = step.named_steps['duplicate_handler'].features_to_drop_
    #         infinite_removed  = step.named_steps['infinite_handler'].infinite_columns_
    #         logger.info(f"Data Cleaning | missing_handler  removed: {len(missing_removed)} features")
    #         for col in missing_removed:
    #             logger.info(f"  - {col} (missing values above threshold)")
    #         logger.info(f"Data Cleaning | constant_handler  removed: {len(constant_removed)} features")
    #         for col in constant_removed:
    #             logger.info(f"  - {col} (constant feature)")
    #         logger.info(f"Data Cleaning | duplicate_handler removed: {len(duplicate_removed)} features")
    #         for col in duplicate_removed:
    #             logger.info(f"  - {col} (duplicate feature)")
    #         logger.info(f"Data Cleaning | infinite_handler  removed: {len(infinite_removed)} features")
    #         for col in infinite_removed:
    #             logger.info(f"  - {col} (infinite values)")
    #     elif step_name == 'sfp_filter':
    #         removed = step.named_steps['sfp_filter'].features_to_drop_
    #         logger.info(f"SFP Filter removed features: {len(removed)} features")
    #         for feature in removed:
    #             logger.info(f"  - {feature} (single feature performance below threshold)")
    #     elif step_name == 'stmp_filter':
    #         stmp_transformer = step.named_steps['stmp_filter'].named_transformers_['stmp_filter']
    #         removed = stmp_transformer.features_to_drop_
    #         logger.info(f"STMP Filter removed features: {len(removed)} features")
    #         for feature in removed:
    #             logger.info(f"  - {feature} (STMP filter criteria)")
    #     elif step_name == 'woe_transformers':
    #         woe_num = step.named_steps['woe_numeric']
    #         woe_cat = step.named_steps['woe_categorical']
    #         num_dropped = [f for f in woe_num.binning_results_ if f not in woe_num.feature_names_]
    #         cat_dropped = [f for f in woe_cat.binning_results_ if f not in woe_cat.feature_names_]
    #         logger.info(f"WOE Numeric     dropped features (low/high IV): {len(num_dropped)} features")
    #         for feature in num_dropped:
    #             logger.info(f"  - {feature} (low/high IV)")
    #         logger.info(f"WOE Categorical dropped features (low/high IV): {len(cat_dropped)} features")
    #         for feature in cat_dropped:
    #             logger.info(f"  - {feature} (low/high IV)")
    #     elif step_name == 'feature_filtering':
    #         psi_removed = step.named_steps['psi_filter'].features_to_drop_
    #         vif_step    = step.named_steps['vif_filter']
    #         vif_removed = [
    #             details.get('removedFeatureName')
    #             for details in vif_step.iteration_steps_.values()
    #             if details.get('removedFeatureName')
    #         ]
    #         rfa_removed = step.named_steps['rfa_filter'].features_to_drop_
    #         logger.info(f"PSI Filter removed features: {len(psi_removed)} features")
    #         for feature in psi_removed:
    #             logger.info(f"  - {feature} (PSI filter criteria)")
    #         logger.info(f"VIF Filter removed features: {len(vif_removed)} features")
    #         for feature in vif_removed:
    #             logger.info(f"  - {feature} (VIF filter criteria)")
    #         logger.info(f"RFA Filter removed features: {len(rfa_removed)} features")
    #         for feature in rfa_removed:
    #             logger.info(f"  - {feature} (RFA filter criteria)")
    # logger.info("=" * 80)

    # logger.info("=" * 80)
    # logger.info("Model Features:")
    # model_features = full_pipeline.named_steps['ml_model'].named_steps['logistic_regression'].model_fit_.model.exog_names
    # final_features = [f for f in model_features]
    # logger.info(f"Model features: {len(final_features)} features")
    # for feature in final_features:
    #     logger.info(f"  - {feature}")
    # logger.info("=" * 80)

    # # Predictions on test set with one sample of output review
    # sample = X_test.iloc[0:1]
    # sample_pred = full_pipeline.predict(sample)
    # sample_prob = full_pipeline.predict_proba(sample)
    # logger.info(f"Sample prediction for first test row: {sample_pred[0]} with probability {sample_prob[0][1]:.4f}")

    # STEP 1: Data Cleaning Pipeline
    pipeline_1 = data_cleaning_pipeline()
    X_train = pipeline_1.fit_transform(X_train)
    X_test = pipeline_1.transform(X_test)

    # STEP 2: Low Cardinality Pipeline
    pipeline_2 = low_cardinality_pipeline()
    X_train = pipeline_2.fit_transform(X_train)
    X_test = pipeline_2.transform(X_test)

    # STEPS 3 & 4: Outlier Detection and Missing Value Imputation Pipeline
    pipeline_3 = outlier_and_missing_pipeline(DEFAULT_CONFIG)
    X_train = pipeline_3.fit_transform(X_train)
    X_test = pipeline_3.transform(X_test)

    # STEP 5: WOE Feature Engineering
    woe_pipeline = woe_transformers_pipeline(DEFAULT_CONFIG)
    X_train = woe_pipeline.fit_transform(X_train, y_train)
    X_test = woe_pipeline.transform(X_test)
    list(map(lambda x: quality_function_for_pandas(x), [X_train, X_test, y_train, y_test]))
    print(X_train.columns)
    print(X_train['low_cardinality_numeric_woe'].value_counts())
    print((X_test['low_cardinality_numeric_woe']).dtype)

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

