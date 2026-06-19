'''
Accelera Consultin

Main Training Pipeline for MLOps Project
'''


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

# sklearn models
from sklearn.ensemble import RandomForestClassifier

# Auto WOE binning functions
from auto_binning_woe import auto_woe_binning_numeric
from auto_binning_woe_categorical import auto_woe_binning_categorical

# Custom Missing Columns Handler
from sklearn_wrapper_missing_handler import MissingColumnsHandler

# Custom Infinite Columns Handler
from sklearn_wrapper_infinite_handler import InfiniteColumnsHandler

# Custom Low Cardinality Handler
from sklearn_wrapper_cardinality_handler import LowCardinalityHandler

# Custom WOE Transformer
from sklearn_wrapper_woe import WOETransformerNumeric, WOETransformerCategorical

# Custom VIF Transformer
from sklearn_wrapper_vif import VIFTransformer

# Custom Logistic Regression Predictor
from sklearn_wrapper_logistic import LogisticRegressionWrapper

# pandas
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_object_dtype

# numpy
import numpy as np
import json

# Helper functions
from helper import (
    io_save_dataframe_as_csv,
    split_train_test
)

# Logging
from logging_config.logger_config import get_logger
logger_name = "mlops.ml_training"
logger_file_name = "ml_training.log"
logger = get_logger(logger_name, logger_file_name)


if __name__ == "__main__":
    # Test with UCI Credit Card Dataset
    metadata_path = "inputs/sample/datatypes.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    column_dtypes = metadata.get("column_dtypes", {})
    # Get the sample data
    input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
    df = pd.read_csv(input_csv_path, dtype=column_dtypes)

    X = df.drop(columns=["TARGET"])
    X["null_column"] = None  # Add a column with all null values to test missing column detection
    X["zero_variance_column_float"] = 1  # Add a column with zero variance to test zero variance detection
    X["zero_variance_column_object"] = "same"  # Add a column with zero variance to test zero variance detection
    X["duplicate_1"] = X["payment_amount_sep_2005"]  # Add a duplicate column to test duplicate detection
    X["duplicate_2"] = X["payment_amount_sep_2005"]  # Add another duplicate column to test duplicate detection
    X["duplicate_3"] = X["payment_amount_sep_2005"]  # Add another duplicate column to test duplicate detection
    X["duplicate_4"] = X["age"]  # Add another duplicate column to test duplicate detection
    X["infinite_column"] = np.inf  # Add a column with infinite values to test infinite value detection
    X["low_cardinality_numeric"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * (len(X) // 10) + [1] * (len(X) % 10)  # Add a low cardinality numeric column to test conversion to object
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.33, random_state=42, stratify=True)

    # START TRAINING PIPELINE
    training_metadata = {}

    # STEP 1: DATA QUALITY CHECKS

    # 1.1 Remove Missing Columns in training
    missing_handler = MissingColumnsHandler(threshold=0.5)
    X_train_cleaned = missing_handler.fit_transform(X_train)
    X_test_cleaned = missing_handler.transform(X_test)

    # 1.2 Detect Zero variance columns in training
    constant_handler = DropConstantFeatures(tol=1.0, missing_values='raise')
    X_train_cleaned = constant_handler.fit_transform(X_train_cleaned)
    X_test_cleaned = constant_handler.transform(X_test_cleaned)
    logger.info(f"Zero variance columns detected and removed: {constant_handler.features_to_drop_}")

    # 1.3 Detect Duplicate columns in training
    duplicate_handler = DropDuplicateFeatures()
    X_train_cleaned = duplicate_handler.fit_transform(X_train_cleaned)
    X_test_cleaned = duplicate_handler.transform(X_test_cleaned)

    # 1.4 Detect Infinite values in training
    infinite_handler = InfiniteColumnsHandler()
    X_train_cleaned = infinite_handler.fit_transform(X_train_cleaned)
    X_test_cleaned = infinite_handler.transform(X_test_cleaned)


    # Append the results to training metadata
    training_metadata["data_quality_checks"] = {
        "missingColumns": missing_handler.missing_columns_,
        "zeroVarianceColumns": constant_handler.features_to_drop_,
        "duplicateColumns": duplicate_handler.features_to_drop_,
        "infiniteColumns": infinite_handler.infinite_columns_
    }

    # STEP 2: COLUMN TYPES CHECKS

    # 2.1 Check Numeric columns in training with only number of unique values less than 10
    low_cardinality_handler = LowCardinalityHandler(threshold=10, action='convert', add_suffix=True, suffix="s")
    X_train_cleaned = low_cardinality_handler.fit_transform(X_train_cleaned)
    X_test_cleaned = low_cardinality_handler.transform(X_test_cleaned)

    # Append the results to training metadata
    training_metadata["column_type_checks"] = {
        "lowCardinalityColumns": low_cardinality_handler.low_cardinality_columns_
    }

    # STEP 3: REPLICAMENT (OUTLIER) DETECTION

    # 3.1 Detect outliers in numeric columns in training using Winsorizer
    numeric_cols = X_train_cleaned.select_dtypes(include=['number']).columns.tolist()

    wz = Winsorizer(
        capping_method='quantiles', 
        tail='right', 
        fold=0.01, 
        add_indicators=True, 
        variables=numeric_cols, 
        missing_values='ignore'
        )
    
    # indicators columns ends with _left and _right
    indicators_cols = [numeric_col + "_left" for numeric_col in numeric_cols] + [numeric_col + "_right" for numeric_col in numeric_cols]

    X_train_cleaned= wz.fit_transform(X_train_cleaned)
    X_test_cleaned = wz.transform(X_test_cleaned)

    # 3.2 Detect outliers in categorical columns (object type) in training using value counts and capping rare categories
    categorical_cols = X_train_cleaned.select_dtypes(include=['object']).columns.tolist()
    rle = RareLabelEncoder(
        tol=0.01, 
        n_categories=10, 
        max_n_categories=None, 
        replace_with='Rare', 
        missing_values='ignore',
        variables=categorical_cols
    )
    X_train_cleaned = rle.fit_transform(X_train_cleaned)
    X_test_cleaned = rle.transform(X_test_cleaned)

    # Append the results to training metadata
    training_metadata["outlier_detection"] = {
        "winsorizer": {
            "capping_method": 'quantiles',
            "tail": 'right',
            "fold": 0.01,
            "variables": numeric_cols
        },
        "rare_label_encoder": {
            "tol": 0.01,
            "n_categories": 10,
            "replace_with": 'Rare',
            "variables": categorical_cols
        }
    }

    # STEP 4: MISSING VALUE IMPUTATION

    # 4.1 Impute missing values in numeric columns in training using ArbitraryNumberImputer
    numeric_cols = X_train_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    arbitrary_number = -999999  # You can choose any arbitrary number that is not present in the data
    # Check if the arbitrary number is already present in the data to avoid confusion
    if (X_train_cleaned[numeric_cols] == arbitrary_number).any().any() or (X_test_cleaned[numeric_cols] == arbitrary_number).any().any():
        logger.warning(f"The arbitrary number {arbitrary_number} is already present in the data. Consider choosing a different number for imputation to avoid confusion.")
   
    imputer = ArbitraryNumberImputer(
        arbitrary_number=arbitrary_number, 
        variables=numeric_cols
    )

    X_train_cleaned = imputer.fit_transform(X_train_cleaned)
    X_test_cleaned = imputer.transform(X_test_cleaned)

    # 4.2 Impute missing values in categorical columns in training using ArbitraryNumberImputer
    categorical_cols = X_train_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    fill_value = 'Missing'  # You can choose any fill value that is not present in the data
    # Check if the fill value is already present in the data to avoid confusion
    if (X_train_cleaned[categorical_cols] == fill_value).any().any() or (X_test_cleaned[categorical_cols] == fill_value).any().any():
        logger.warning(f"The fill value '{fill_value}' is already present in the data. Consider choosing a different value for imputation to avoid confusion.")
   
    imputer_cat = CategoricalImputer(
        imputation_method='missing', # Can be 'frequent' for frequent category imputation or 'missing' to impute with an arbitrary value. 
        fill_value=fill_value, 
        variables=categorical_cols
    )

    X_train_cleaned = imputer_cat.fit_transform(X_train_cleaned)
    X_test_cleaned = imputer_cat.transform(X_test_cleaned)

    # Save data as csv
    saved_path = io_save_dataframe_as_csv(X_train_cleaned, "outputs/training/step5_X_train_cleaned.csv")
    logger.info(f"Cleaned training data saved to: {saved_path}")

    # STEP 5:   FEATURE ENGINEERING (WOE TRANSFORMATION)

    # 5.0 RUN WOE AUTO BINNING fro numeric and categorical columns
    for col in X_train_cleaned.columns:
        # Skip indicaotor columns created by Winsorizer
        if col not in indicators_cols:
            if is_numeric_dtype(X_train_cleaned[col]):
                result = auto_woe_binning_numeric(
                    df=pd.concat([X_train_cleaned, y_train], axis=1), # concat X_train_cleaned and y_train to pass the target variable
                    feature=col,
                    target="TARGET",
                    initial_bins=20,
                    min_bin_pct=0.05,
                    max_final_bins=10,
                    min_final_bins=3,
                    min_iv=0.02,
                    max_iv=0.50,
                    max_iter=25
                )
            # if data type is object, we will use auto_woe_binning_categorical
            elif is_object_dtype(X_train_cleaned[col]):
                result = auto_woe_binning_categorical(
                    df=pd.concat([X_train_cleaned, y_train], axis=1), # concat X_train_cleaned and y_train to pass the target variable
                    feature=col,
                    target="TARGET",
                    min_bin_pct=0.05,
                    max_final_bins=6,
                    min_final_bins=2,
                    min_iv=0.02,
                    max_iv=0.50,
                    max_iter=20
                )

    numeric_features = X_train_cleaned.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"Numeric features for WOE transformation: {numeric_features}")
    logger.info(f"Categorical features for WOE transformation: {categorical_features}")
    
    # 5.2 Apply WOE transformation for numeric columns
    if numeric_features:
        woe_transformer_numeric = WOETransformerNumeric(base_path="outputs/auto_binning_woe")
        X_train_numeric = woe_transformer_numeric.fit_transform(X_train_cleaned[numeric_features])
        X_test_numeric = woe_transformer_numeric.transform(X_test_cleaned[numeric_features])
        logger.info(f"Successfully transformed {len(numeric_features)} numeric features to WOE")
    
    # 5.3 Apply WOE transformation for categorical columns
    if categorical_features:
        woe_transformer_categorical = WOETransformerCategorical(base_path="outputs/auto_binning_woe")
        X_train_categorical = woe_transformer_categorical.fit_transform(X_train_cleaned[categorical_features])
        X_test_categorical = woe_transformer_categorical.transform(X_test_cleaned[categorical_features])
        logger.info(f"Successfully transformed {len(categorical_features)} categorical features to WOE")

    
    # 5.4 Combine transformed numeric and categorical features
    X_train_cleaned = pd.concat([X_train_numeric, X_train_categorical], axis=1)
    X_test_cleaned = pd.concat([X_test_numeric, X_test_categorical], axis=1)
    
    logger.info(f"Final training set shape after WOE transformation: {X_train_cleaned.shape}")
    logger.info(f"Final test set shape after WOE transformation: {X_test_cleaned.shape}")

    # Save data as csv
    saved_path = io_save_dataframe_as_csv(X_train_cleaned, "outputs/training/step5_X_train_cleaned.csv")
    logger.info(f"Cleaned training data saved to: {saved_path}")

    # STEP 6: Drop High PSI Features
    # To compute the PSI, DropHighPSIFeatures() splits the dataset in two: a basis and a test set. 
    # Then, it compares the distribution of each feature between those sets.
    psi_transformer = DropHighPSIFeatures(threshold=0.25)
    X_train_psi = psi_transformer.fit_transform(X_train_cleaned, X_test_cleaned)
    X_test_psi = psi_transformer.transform(X_test_cleaned)
    logger.info(f"Final training set shape after PSI transformation: {X_train_psi.shape}")
    logger.info(f"Final test set shape after PSI transformation: {X_test_psi.shape}")

    # STEP 7: Decrease Multicollinearity with VIF
    # Variance Inflation Factor (VIF) is a measure of how much the variance of a regression coefficient is inflated due to multicollinearity with other features.
    # A VIF value greater than 5 or 10 indicates high multicollinearity.
    vif_transformer = VIFTransformer(vif_threshold=5.0)
    X_train_vif = vif_transformer.fit_transform(X_train_psi, y_train)
    X_test_vif = vif_transformer.transform(X_test_psi)
    logger.info(f"Final training set shape after VIF transformation: {X_train_vif.shape}")
    logger.info(f"Final test set shape after VIF transformation: {X_test_vif.shape}")

    # STEP 8: Feature Selection with Recursive Feature Addition (RFA) using RandomForestClassifier
    rfa = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=42), 
        cv=2,
        threshold=0.0001,
        scoring='roc_auc',
        )
    X_train_rfa = rfa.fit_transform(X_train_vif, y_train)
    X_test_rfa = rfa.transform(X_test_vif)
    logger.info(f"Final training set shape after RFA transformation: {X_train_rfa.shape}")
    logger.info(f"Final test set shape after RFA transformation: {X_test_rfa.shape}")

    # STEP 9: Custom Logistic Regression Model Training
    log_reg = LogisticRegressionWrapper(
        disp=False,
        method='bfgs',
        maxiter=500,
        model_dir="outputs/logitmodel",
        save_results=True,
        save_model=True,
        evaluate_model=True,
        evaluate_bins=20,
        threshold=0.5
    )
    fitted_model = log_reg.fit(X_train_rfa, y_train)

