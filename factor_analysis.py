'''
Accelera Consulting

Factor Analysis Module

Based on the following sources:
- https://metricgate.com/docs/kaiser-meyer-olkin/
- https://metricgate.com/docs/bartlett-sphericity-test/
- https://metricgate.com/docs/eigenvalue-calculator/


For JSON outputs, we are applying CamelCase naming convention for keys to be consistent with other outputs in the project. 
(Snake case is used for variable names in the code, but JSON outputs use CamelCase for better readability and consistency with other outputs in the project.)

Main Steps:
1: KMO Test of Sampling Adequacy -> Question: Factor Analysis can be performed Decision: if KMO > 0.50
2: Bartlett's Test of Sphericity -> Question: Are the variables sufficiently correlated to justify factor analysis? Decision: if p-value < 0.05
3: Eigenvalues and Eigenvectors Calculation -> Question: How many factors to retain? Decision: Kaiser criterion (eigenvalue > 1) and explained variance thresholds (e.g. 80% cumulative variance)
'''

import json
import os
from pathlib import Path
import uuid
from datetime import datetime, timezone, timedelta

import logging
from logging_config import get_logger # there is a logging_config.py file with a get_logger function to set up logging for the project.

import pandas as pd
# Set pandas display options for better readability
pd.set_option('display.float_format', '{:.6f}'.format) # Display floats with 6 decimal places
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', 150) # Set display width for better formatting

import numpy as np
# Set numpy print options for better readability
np.set_printoptions(precision=6, suppress=True, linewidth=150)

from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.selection import DropDuplicateFeatures

from sklearn.datasets import load_iris # Used for testing and demonstration purposes. Not part of the main factor analysis code.

timestamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S")
runid = uuid.uuid4().hex[:8]

# Step Up Logger for StreamHandler and FileHandler with timestamp and run_id in the log file name and log directory structure
logger = get_logger(
    name="mlops.factor_analysis", 
    log_file_name=f"factor_analysis.log", 
    log_to_file=True, 
    log_mode="w",
    timestamp=timestamp,
    runid=runid
)

# Log properties of the logger for debugging purposes
logger.info("|||| STARTING FACTOR ANALYSIS MODULE ||||")
logger.info(f"Logger Name: {logger.name}")
logger.info(f"Logger Level: {logging.getLevelName(logger.getEffectiveLevel())}")
logger.info(f"Logger has handlers: {logger.handlers}")
for handler in logger.handlers:
    logger.info(f"Handler Type: {type(handler).__name__}")
logger.info(
    "\n"
    "Log Folder Structure\n"
    "mlops\n"
    f"└── {timestamp}\n"
    f"    └── {runid}\n"
    "        └── logs\n"
    f"            └── {Path(logger.handlers[1].baseFilename).name}"
)
logger.info("Logger initialized successfully for factor analysis module.")
logger.info("|"*41)

def get_iris_dataset(target_variable: str = "TARGET") -> pd.DataFrame:
    '''
    Purpose
    -------
    Loads the Iris dataset and returns it as a pandas DataFrame with the specified target variable name.

    Parameters
    ----------
    target_variable : str, default="TARGET"
        Name of the target variable column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the Iris dataset with the target variable.
    '''
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    target_iris = pd.Series(iris.target, name=target_variable)
    df_iris[target_variable] = target_iris
    return df_iris

def check_data_quality(df_work: pd.DataFrame):
    '''
    Purpose
    -------
    Checks the data quality of the dataframe by identifying zero variance columns,
    columns with infinite values and duplicate columns.

    Parameters
    ----------
    df_work : pd.DataFrame
        Input dataframe to check for data quality issues.
    
    Returns
    -------
    df_work : pd.DataFrame
        Dataframe after dropping zero variance columns, columns with infinite values and duplicate columns.
    zero_variance_cols : list
        List of zero variance columns that were dropped.
    infinite_cols : list
        List of columns with infinite values that were dropped.
    duplicate_cols : list
        List of duplicate columns that were dropped.    
    '''
    # Drop Zero Variance Columns and log the dropped columns
    zero_variance_cols = df_work.columns[df_work.nunique() <= 1].tolist()
    if zero_variance_cols:
        logger.info(f"Dropping zero variance columns: {zero_variance_cols}")
        df_work = df_work.drop(columns=zero_variance_cols)
    else:
        logger.info("No zero variance columns found.")
    
    # Drop infinite values columns and log the dropped columns
    numeric_cols = df_work.select_dtypes(include=[np.number])
    infinite_cols = numeric_cols.columns[np.isinf(numeric_cols.values).any(axis=0)].tolist()
    if infinite_cols:
        logger.info(f"Dropping columns with infinite values: {infinite_cols}")
        df_work = df_work.drop(columns=infinite_cols)
    else:
        logger.info("No columns with infinite values found.")
    
    # Drop Duplicate Columns and log the dropped columns
    duplicate_dropper = DropDuplicateFeatures()
    df_work = duplicate_dropper.fit_transform(df_work)
    duplicate_cols = duplicate_dropper.features_to_drop_
    # Conver to list
    duplicate_cols = list(duplicate_cols)
    if duplicate_cols:
        logger.info(f"Dropping duplicate columns: {duplicate_cols}")
    else:
        logger.info("No duplicate columns found.")

    return df_work, zero_variance_cols, infinite_cols, duplicate_cols

def check_target_is_binary(
        df_work: pd.DataFrame, 
        target_variable: str,
        target_mapping: dict | None = {'yes': 1, 'no': 0}):
    '''
    Purpose
    -------
    Checks if the target variable is binary.

    Parameters
    ----------
    df_work : pd.DataFrame
        Input dataframe containing the target variable.
    target_variable : str
        Name of the target variable to check.
    target_mapping : dict | None, default={'yes': 1, 'no': 0}
        Mapping of target variable values to binary values.

    Returns
    -------
    is_binary : bool
        True if the target variable is binary, False otherwise.
    df_work : pd.DataFrame
        Dataframe with the target variable mapped to binary values if it is binary and target_mapping is provided.
    
    Raises
    ------
    ValueError
        If the target variable is not found in the dataframe columns.
    '''
    if target_variable not in df_work.columns:
        error_message = f"Target variable '{target_variable}' not found in the dataframe columns."
        logger.error(error_message)
        raise ValueError(error_message)

    unique_values = df_work[target_variable].dropna().unique()
    is_binary = len(unique_values) == 2

    if is_binary:
        logger.info(f"Target variable '{target_variable}' is binary with values: {unique_values}")
    else:
        logger.info(f"Target variable '{target_variable}' is not binary. Unique values: {unique_values}")
    
    # Convert Target Variable 1 and 0 accordingly to target_mapping if it is binary and log the mapping
    if is_binary and target_mapping and not set(unique_values) == {1, 0}:
        df_work[target_variable] = df_work[target_variable].map(target_mapping)
        logger.info(f"Target variable '{target_variable}' mapped to binary values: {target_mapping}")
    else:
        logger.info(f"No mapping applied to target variable '{target_variable}' as it is already in binary format: 1 and 0.")

    return is_binary, df_work

def prepare_factor_analysis_data(
    df: pd.DataFrame,
    target_variable: str | None = None,
    drop_last: bool = True,
    fill_strategy_numeric: str = "median",
    encoding_strategy_categorical: str = "ordinal"
) -> pd.DataFrame:
    """
    Purpose
    -------
    Prepares a raw dataframe for factor analysis.

    This function performs the preprocessing steps required before
    calculating KMO, Bartlett test, eigenvalues and factor loadings.

    Steps
    -----
    1. Check data quality and drop zero variance columns, columns with infinite values and duplicate columns.
    2. Drop target variable if provided.
    3. Fill null values in numeric columns with mean, median or zero according to fill_strategy_numeric.
    4. Fill null values in categorical columns with the most frequent category.
    5. One-hot encode categorical variables.
    6. Standardize all variables using StandardScaler.
    7. Add target variable back to the beginning of dataframe if it was dropped and check if it is binary.
    8. Log the target distribution if target variable is provided.
    9. Return the prepared dataframe and metadata of the preparation steps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    target_variable : str, default=None
        The name of the target variable. If provided, it will be excluded from the factor analysis.

    drop_last : bool, default=True
        Whether to drop the last dummy category during one-hot encoding.

    Example Input
    -------------
        AGE  TENURE  INCOME CITY
    0    25       1   30000    A
    1    30       3   40000    A
    2    35       5   50000    B

    Returns
    -------
    df_work : pd.DataFrame
        Encoded and standardized dataframe.

    Example Output
    --------------
            AGE    TENURE    INCOME    CITY_A
    0   -1.2247   -1.1355   -1.0000    0.7071
    1    0.0000   -0.1622    0.0000    0.7071
    2    1.2247    1.2977    1.0000   -1.4142

    Notes
    -----
    Factor analysis is based on correlations.
    Standardization is used so variables with large scales do not dominate the analysis.
    """
    # Raise error if any column is not numeric or object data type
    non_supported_cols = df.select_dtypes(exclude=["number", "object"]).columns.tolist()
    if non_supported_cols:
        error_message = f"Unsupported data types in columns: {non_supported_cols}. " \
                        "Only numeric and object data types are supported."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise error if fill_strategy_numeric is not supported
    if fill_strategy_numeric not in ["mean", "median", "zero"]:
        error_message = f"Unsupported fill strategy for numeric variables: {fill_strategy_numeric}. " \
                        "Use 'mean', 'median' or 'zero'."
        logger.error(error_message)
        raise ValueError(error_message)

    # Create a copy of the dataframe to work on and drop target variable if provided
    df_work = df.copy()

    # Check data quality and drop zero variance columns, columns with infinite values and duplicate columns
    df_work, zero_variance_cols, infinite_cols, duplicate_cols = check_data_quality(df_work)

    # Log the target variable and whether it will be dropped
    if target_variable:
        logger.info(f"Target variable provided: {target_variable}. It will be dropped from the factor analysis.")
        # Check if target variable exists in the dataframe if not raise error
        if target_variable not in df_work.columns:
            error_message = f"Target variable '{target_variable}' not found in the dataframe columns."
            logger.error(error_message)
            raise ValueError(error_message)
        df_work = df_work.drop(columns=[target_variable])

    # Numeric Features: Fill Null values with mean, median or zero according to fill_strategy_numeric
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    
    # Log the numeric columns and their fill strategy
    for col in numeric_cols:
        logger.info(f"Numeric Column: {col} | Fill Strategy: {fill_strategy_numeric}")
    
    # Call proper imputer based on fill_strategy_numeric
    if fill_strategy_numeric == "mean":
        numeric_imputer = MeanMedianImputer(
            imputation_method="mean",
            variables=numeric_cols
        )
    if fill_strategy_numeric == "median":
        numeric_imputer = MeanMedianImputer(
            imputation_method="median",
            variables=numeric_cols
        )
    if fill_strategy_numeric == "zero":
        numeric_imputer = ArbitraryNumberImputer(
            arbitrary_number=0,
            variables=numeric_cols
        )
    # Fit and transform the numeric imputer on the numeric columns
    df_work = numeric_imputer.fit_transform(df_work)
    logger.info(f"Numeric columns imputed using {fill_strategy_numeric} strategy.")

    # Categorical Features: Fill Null values with RareLabelImputer
    categorical_cols = df_work.select_dtypes(include=["object"]).columns.tolist()
    # Log the categorical columns and their imputation strategy
    for col in categorical_cols:
        logger.info(f"Categorical Column: {col} | Imputation Strategy: frequent")

    if categorical_cols:
        categorical_imputer = CategoricalImputer(
            imputation_method="frequent",
            variables=categorical_cols
        )
        df_work = categorical_imputer.fit_transform(df_work)
        logger.info("Categorical columns imputed using frequent strategy.")
    
    # Raise error if there are still missing values after imputation
    if df_work.isnull().sum().sum() > 0:
        missing_values_count_by_column = df_work.isnull().sum()
        error_message = "There are still missing values after imputation. Please check the imputation steps."
        error_message += f"\nMissing values count by column:\n{missing_values_count_by_column}"
        logger.error(error_message)
        raise ValueError(error_message)

    # Set up encoder based on encoding_strategy_categorical and log the encoding strategy
    if encoding_strategy_categorical == "onehot":
        encoder = OneHotEncoder(
            variables=categorical_cols,
            drop_last=drop_last
        )
    elif encoding_strategy_categorical == "ordinal":
        encoder = OrdinalEncoder(
            variables=categorical_cols,
            encoding_method="arbitrary", # categories are numbered arbitrarily.
        )

    # Fit and transform the encoder on the dataframe
    df_work = encoder.fit_transform(df_work)
    logger.info(f"Categorical variables encoded using {encoding_strategy_categorical} strategy.")

    # Raise error if there are still non numeric data type columns after encoding
    non_numeric_cols_after_encoding = df_work.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols_after_encoding:
        error_message = f"Columns with non-numeric data type after encoding: {non_numeric_cols_after_encoding}. "
        error_message += "Please check the encoding steps."
        logger.error(error_message)
        raise ValueError(error_message)

    # Use SklearnTransformerWrapper to apply StandardScaler to all columns
    scaler = SklearnTransformerWrapper(
        transformer=StandardScaler(),
        variables=df_work.columns.tolist()
    )

    df_work = scaler.fit_transform(df_work)
    logger.info("All variables standardized using StandardScaler.")

    # Check if there are any infinite values after scaling and raise error if there are
    if np.isinf(df_work.values).sum() > 0:
        inf_columns = df_work.columns[np.isinf(df_work.values).any(axis=0)].tolist()
        error_message = f"There are infinite values in the dataframe after scaling. \
            Please check the scaling steps. Columns with infinite values: {inf_columns}"
        logger.error(error_message)
        raise ValueError(error_message)

    # Add Target back to the beginning of dataframe if it was dropped and log the target distribution
    if target_variable:
        df_work.insert(loc=0, column=target_variable, value=df[target_variable])
        logger.info(f"Target variable '{target_variable}' added back to the dataframe after preprocessing.")
        # Check Target is Binary and Convert to 1 and 0 if it is binary and log the target distribution
        is_binary, df_work = check_target_is_binary(df_work, target_variable)
        logger.info(f"Binary Check Completed for Target Variable '{target_variable}'. Is Binary: {is_binary}")

    if target_variable:
        target_dist = df_work[target_variable].value_counts(normalize=True).mul(100).round(2)
        logger.info(f"Target distribution (%):\n{target_dist}")
        target_value_counts = df_work[target_variable].value_counts()
        logger.info(f"Target distribution values:\n{target_value_counts}")
    
    metadata = {
        "originalColumns": df.columns.tolist(),
        "processedColumns": df_work.columns.tolist(),
        "numericColumns": numeric_cols,
        "categoricalColumns": categorical_cols,
        "zeroVarianceColumnsDropped": zero_variance_cols,
        "infiniteValueColumnsDropped": infinite_cols,
        "duplicateColumnsDropped": duplicate_cols,
        "targetVariable": target_variable,
        "fillStrategyNumeric": fill_strategy_numeric,
        "dropLastCategory": drop_last,
    }
    
    logger.info("Data preparation for factor analysis completed successfully.")
    return df_work, metadata

def interpret_kmo(kmo_value: float):
    """
    Purpose
    -------
    Interprets the KMO value based on commonly accepted thresholds.

    Parameters
    ----------
    kmo_value : float
        The KMO statistic to interpret.

    Returns
    -------
    interpretation : str
        Interpretation of the KMO value.

    Example Input
    -------------
    kmo_value = 0.7069

    Example Output
    --------------
    interpretation:

    "Good"

    Interpretation
    --------------
    KMO < 0.50
        Not suitable.

    0.50 <= KMO < 0.60
        Weak.

    0.60 <= KMO < 0.70
        Moderate.

    0.70 <= KMO < 0.80
        Good.

    0.80 <= KMO < 0.90
        Very good.

    KMO >= 0.90
        Excellent.
    """
    
    if kmo_value < 0.50:
        return "Not suitable"
    elif kmo_value < 0.60:
        return "Weak"
    elif kmo_value < 0.70:
        return "Moderate"
    elif kmo_value < 0.80:
        return "Good"
    elif kmo_value < 0.90:
        return "Very good"
    else:
        return "Excellent"

def calculate_kmo_manual(
        df: pd.DataFrame, 
        target_variable: str | None = None,
        output_dir: str | None = None):
    """
    Purpose
    -------
    Calculates the Kaiser-Meyer-Olkin (KMO) statistic manually.

    KMO measures whether the correlation structure of the variables
    is suitable for factor analysis.

    Reference
    ---------
    https://metricgate.com/docs/kaiser-meyer-olkin/

    Steps
    -------------
    1. Calculate the correlation matrix of the variables.
    2. Calculate the partial correlation matrix by taking the inverse of the correlation matrix.
        Inverse of Correlation Matrix is Precision Matrix(P).
    3. Calculate KMO for each variable and overall KMO for the dataset.

    Formula Logic
    -------------
    KMO compares squared correlations with squared partial correlations:

        sum(r_ij^2)
        --------------------------------
        sum(r_ij^2) + sum(p_ij^2)

    where:

    r_ij
        Correlation between variable i and variable j.

    p_ij
        Partial correlation between variable i and variable j,
        after controlling for all other variables.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded and standardized numeric dataframe.
    target_variable : str, optional
        The name of the target variable to be excluded from the KMO calculation.
    output_dir : str, optional
        Directory to save KMO output as JSON file. If None, output will not be saved to a file.

    Example Input
    -------------
            AGE    TENURE    INCOME
    0   -1.2247   -1.1355   -1.0000
    1    0.0000   -0.1622    0.0000
    2    1.2247    1.2977    1.0000

    Returns
    -------
    kmo_per_variable : pd.Series
        KMO value for each variable.

    kmo_model : float
        Overall KMO value for the full dataset.

    Example Output
    --------------
    kmo_per_variable:

    AGE       0.7423
    TENURE    0.6641
    INCOME    0.7288
    Name: kmo, dtype: float64

    kmo_model:

    0.7069

    Interpretation
    --------------
    KMO < 0.50
        Not suitable.

    0.50 <= KMO < 0.60
        Weak.

    0.60 <= KMO < 0.70
        Moderate.

    0.70 <= KMO < 0.80
        Good.

    0.80 <= KMO < 0.90
        Very good.

    KMO >= 0.90
        Excellent.

    Notes
    -----
    This function is used as a quality gate before factor extraction.

    Factor analysis is generally not recommended when:

        kmo_model < 0.50
    """
    logger.info("Kaiser-Meyer-Olkin (KMO) Test of Sampling Adequacy.")

    # Copy df
    df_work = df.copy()

    # if target_variable is provided, drop it from the dataframe before calculating KMO and log the dropped target variable
    # if target_variable is not provided, log a warning that target variable is not provided and KMO will be calculated for all variables in the dataframe
    if target_variable is None:
        logger.warning("Target variable is not provided. KMO will be calculated for all variables in the dataframe.")
    elif target_variable in df_work.columns:
        df_work = df_work.drop(columns=[target_variable])
        logger.info(f"Dropped target variable '{target_variable}' from dataframe for KMO calculation.")
    else:
        error_message = f"Target variable '{target_variable}' not found in the dataframe columns. " \
                        "Please check the target variable name and try again."
        logger.error(error_message)
        raise ValueError(error_message)
        
    # Calculate the correlation matrix with Pearson method and convert it to numpy array
    corr_matrix = df_work.corr(method='pearson').round(6).values
    # Log the correlation matrix and its properties
    logger.info(f"Shape of Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")
    logger.info(f"Min correlation in Corr Matrix: {corr_matrix.min():.6f}")
    logger.info(f"Max correlation in Corr Matrix: {corr_matrix.max():.6f}")
    logger.info(f"Mean abs correlation in Corr Matrix: {np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])):.6f}")

    inv_corr_matrix = np.linalg.inv(corr_matrix)
    logger.info("Inverse of Correlation matrix (Precision Matrix) prepared for Partial Correlation calculation")
    # Log Properties of Inverse Correlation Matrix
    logger.info(f"Shape of Inverse Correlation matrix: {inv_corr_matrix.shape[0]} x {inv_corr_matrix.shape[1]}")

    partial_corr = np.zeros_like(corr_matrix)
    logger.info("partial_corr matrix initialized with zeros")
    # Log the initial state of partial_corr matrix
    logger.info(f"Shape of Partial Correlation matrix: {partial_corr.shape[0]} x {partial_corr.shape[1]}")

    # Partial Correlation Calculation Logic without Matrix Inversion for only 3 variables:
    # PartialCorr(X, Y | Z)
    # r_xy = Correlation(X, Y) r_xz = Correlation(X, Z) r_yz = Correlation(Y, Z)
    # p_xy = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    logger.info("Calculating partial correlation matrix")
    logger.info("-" * 50)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            col_i = df_work.columns[i]
            col_j = df_work.columns[j]
            if i == j:
                partial_corr[i, j] = 0
            else:
                pij = inv_corr_matrix[i, j]
                pii = inv_corr_matrix[i, i]
                pjj = inv_corr_matrix[j, j]
                partial_corr[i, j] = (
                    -pij
                    / np.sqrt(pii * pjj)
                )
                logger.info(f"PartialCorr({col_i} (idx{i}), {col_j} (idx{j}) | others) = -({pij:.6f}) / sqrt({pii:.6f} * {pjj:.6f}) = {partial_corr[i, j]:.6f}")

    # KMO Formula Logic:
    # KMO works with squared correlations and squared partial correlations.
    # We are getting squared because KMO compares the magnitude of correlations to partial correlations 
    # without regard to their direction (positive or negative).
    # rij  = correlation pij  = partial correlation
    #               Σ(rij²)
    # KMO = ------------------
    #        Σ(rij²) + Σ(pij²)
    # KMO prefers higher correlations and lower partial correlations.
    # It means that variables share common factors (high correlations) and do not have unique variance (low partial correlations).
    # So you can group variables together based on their shared variance, which is the essence of factor analysis.
    # For example, if we have 3 variables X, Y and Z:
    # Corr(X,Y)=0.80
    # PartialCorr(X,Y|others)=0.10
    # 0.80² = 0.64
    # 0.10² = 0.01
    #     0.64
    # --------------
    #   0.64 + 0.01
    # = 0.984


    # Get Squared Correlations
    corr_squared = corr_matrix ** 2
    logger.info("Squared Correlation matrix calculated")

    # Get Squared Partial Correlations
    partial_corr_squared = partial_corr ** 2
    logger.info("Squared Partial Correlation matrix calculated")

    # Fill diagonal with zeros because we are only interested in correlations and partial correlations between different variables, 
    # not the correlation of a variable with itself. (Converting them 1 to 0)
    np.fill_diagonal(corr_squared, 0)
    np.fill_diagonal(partial_corr_squared, 0)

    # ========================================================================
    # KMO PER VARIABLE (Variable-Level Analysis)
    # ========================================================================
    # PURPOSE:
    #   Calculates an individual KMO value for each variable by comparing
    #   its squared correlations with all other variables against its
    #   squared partial correlations. This reveals which variables are
    #   most suitable for factor analysis.
    #
    # FORMULA (for variable i):
    #   KMO(i) = Σ(r_ij²) / [Σ(r_ij²) + Σ(p_ij²)]
    #   where j ≠ i (all other variables)
    #
    # CALCULATION EXAMPLE:
    #   Given 3 variables (X, Y, Z):
    #
    #   Squared Correlations Matrix:     Squared Partial Correlations Matrix:
    #         X       Y       Z               X       Y       Z
    #   X    0.00    0.64    0.49      X    0.00    0.04    0.01
    #   Y    0.64    0.00    0.36      Y    0.04    0.00    0.09
    #   Z    0.49    0.36    0.00      Z    0.01    0.09    0.00
    #
    #   For variable X:
    #   - Sum of squared correlations: 0.64 + 0.49 = 1.13
    #   - Sum of squared partial correlations: 0.04 + 0.01 = 0.05
    #   - KMO(X) = 1.13 / (1.13 + 0.05) = 0.958
    #
    # INTERPRETATION:
    #   Each variable gets its own KMO score, repeated for all variables.
    #   Higher variable-level KMO indicates strong correlation with other
    #   variables and low unique variance—ideal for factor analysis.
    #
    # COMPUTATION NOTE:
    #   Using axis=0 sums across rows for each column (variable), giving
    #   each variable's aggregated correlation/partial correlation strength.
    # ======================================================================
    kmo_per_variable = corr_squared.sum(axis=0) / (
        corr_squared.sum(axis=0) + partial_corr_squared.sum(axis=0)
    )
    # Log KMO per variable
    logger.info("KMO per variable:")
    logger.info("-" * 50)
    for i, col in enumerate(df_work.columns):
        kmo_value = kmo_per_variable[i]
        interpretation = interpret_kmo(kmo_value)
        logger.info(f"KMO for variable '{col}': {kmo_value:.6f} ({interpretation})")
    logger.info("-" * 50)

    # ========================================================================
    # OVERALL KMO MODEL (Dataset-Level Summary)
    # ========================================================================
    # PURPOSE:
    #   Calculates a single, aggregate KMO value for the entire dataset by
    #   summing all squared correlations and partial correlations across all
    #   variables. This summarizes overall suitability for factor analysis.
    #
    # FORMULA:
    #   KMO_Model = Σ(r_ij²) / [Σ(r_ij²) + Σ(p_ij²)]
    #
    # CALCULATION EXAMPLE:
    #   Given 3 variables (X, Y, Z):
    #
    #   Squared Correlations:      Squared Partial Correlations:
    #   - X↔Y: 0.64               - X↔Y: 0.04
    #   - X↔Z: 0.49               - X↔Z: 0.01
    #   - Y↔Z: 0.36               - Y↔Z: 0.09
    #   ────────────              ────────────
    #   Sum = 2.98                 Sum = 0.28
    #
    #   KMO_Model = 2.98 / (2.98 + 0.28) = 0.914
    #
    # INTERPRETATION SCALE:
    #   < 0.50      → Not suitable (inadequate correlation structure)
    #   0.50–0.60   → Weak (marginal for factor analysis)
    #   0.60–0.70   → Moderate (acceptable correlation structure)
    #   0.70–0.80   → Good (strong correlation structure)
    #   0.80–0.90   → Very Good (very strong correlations)
    #   ≥ 0.90      → Excellent (ideal for factor analysis)
    #
    # KEY INSIGHT:
    #   Higher KMO indicates variables share more common variance and have
    #   less unique variance—the essence of effective factor grouping.
    # ======================================================================
    kmo_model = corr_squared.sum() / (
        corr_squared.sum() + partial_corr_squared.sum()
    )
    interpretation = interpret_kmo(kmo_model)
    logger.info(f"Overall KMO for the dataset: {kmo_model:.6f} ({interpretation})")

    # Convert kmo_per_variable to pandas Series with variable names as index and name it "kmo"
    kmo_per_variable = pd.Series(
        kmo_per_variable,
        index=df_work.columns,
        name="kmo"
    )

    # Convert 6 decimal places for kmo_per_variable and kmo_model for better readability in logs and outputs
    kmo_per_variable = kmo_per_variable.round(6)
    kmo_model = round(kmo_model, 6)

    # Prepare metadata for KMO output
    metadata = {
        "kmoPerVariable": kmo_per_variable.round(6).to_dict(),
        "kmoModel": {"value": kmo_model, "interpretation": interpret_kmo(kmo_model)},
        "kmoInterpretationThresholds": {
            "Not suitable": "< 0.50",
            "Weak": "0.50 - 0.60",
            "Moderate": "0.60 - 0.70",
            "Good": "0.70 - 0.80",
            "Very Good": "0.80 - 0.90",
            "Excellent": "> 0.90"
        },
        "whatIsKmo": "KMO (Kaiser-Meyer-Olkin) measures the suitability of the dataset for factor analysis based on the correlation structure of the variables.\
Higher KMO values indicate that the variables share common variance and are suitable for factor analysis.\
KMO prefers higher correlations and lower partial correlations.\
It means that variables share common factors (high correlations) and do not have unique variance (low partial correlations)."
        }
    logger.info("Metadata for KMO output prepared.")

    # Save metadata to JSON file if output_dir is provided and log the saved file path
    if output_dir:
        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        # Save KMO Metadata output to JSON file
        with open(f"{output_dir}/kmo_output.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"KMO Metadata output saved to {output_dir}/kmo_output.json")
    
    logger.info("KMO (Kaiser-Meyer-Olkin) calculation completed successfully.")
    return kmo_per_variable, kmo_model, metadata

def interpret_bartlett(p_value: float, reject_threshold: float = 0.05):
    """
    Purpose
    -------
    Interprets the p-value from Bartlett's Test of Sphericity.

    Parameters
    ----------
    p_value : float
        The p-value from Bartlett's test.
    reject_threshold : float, default=0.05
        The threshold for rejecting the null hypothesis. Commonly set at 0.05.

    Returns
    -------
    interpretation : str
        Interpretation of the p-value.
    suggested_action : str
        Suggested action based on the interpretation.

    Example Input
    -------------
    p_value = 0.000001
    reject_threshold = 0.05

    Example Output
    --------------
    interpretation:

    "Reject H0. Factor analysis can continue."

    Interpretation
    --------------
    p_value < 0.05
        Reject H0.
        Factor analysis can continue.

    p_value >= 0.05
        Fail to reject H0.
        Factor analysis is not recommended.
    """
    logger.info(f"Interpreting Bartlett's Test of Sphericity results. For p-value: {p_value:.8f} with reject threshold: {reject_threshold:.2f}")
    if p_value < reject_threshold:
        interpretation = f"Reject H0 with p-value={p_value:.8f}. The correlation matrix significantly differs from an identity matrix. Factor analysis can continue."
        suggested_action = "Factor analysis can continue."
    else:
        interpretation = f"Fail to reject H0 with p-value={p_value:.8f}. The correlation matrix does not significantly differ from an identity matrix. Factor analysis is not recommended."
        suggested_action = "Factor analysis is not recommended."
    logger.info(interpretation)
    return interpretation, suggested_action

def calculate_bartlett_manual(
        df: pd.DataFrame,
        target_variable: str | None = None,
        output_dir: str | None = None):
    """
    Purpose
    -------
    Performs Bartlett's Test of Sphericity manually.

    Bartlett's test evaluates whether the correlation matrix
    significantly differs from an identity matrix.

    Example of Identity Matrix:
            AGE  TENURE  INCOME
        AGE     1      0       0
        TENURE  0      1       0
        INCOME  0      0       1

        -> Correlation between different variables is zero.
        -> Variables do not share common factors.
    
    Reference
    ---------
    https://metricgate.com/docs/bartlett-sphericity-test/

    Hypothesis
    ----------
    H0:
        The correlation matrix is an identity matrix.
        Variables are not sufficiently correlated.

    H1:
        The correlation matrix is not an identity matrix.
        Variables are sufficiently correlated.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded and standardized numeric dataframe.
    target_variable : str, optional
        The name of the target variable to be excluded from the Bartlett's test calculation.
    output_dir : str, optional
        Directory to save Bartlett's test output as JSON file. If None, output will not be

    Example Input
    -------------
            AGE    TENURE    INCOME
    0   -1.2247   -1.1355   -1.0000
    1    0.0000   -0.1622    0.0000
    2    1.2247    1.2977    1.0000

    Returns
    -------
    chi_square_value : float
        Bartlett chi-square statistic.

    p_value : float
        Bartlett test p-value.

    degrees_of_freedom : int
        Test degrees of freedom.

    Example Output
    --------------
    chi_square_value: 85.1200 
        definition: Bartlett chi-square statistic. Measures the overall significance of the correlation matrix.
        range: [0, ∞)
        interpretation: Higher values indicate that the correlation matrix significantly differs from an identity matrix, suggesting that

    p_value: 0.000001
        definition: Bartlett test p-value. Indicates the probability of observing the chi-square statistic under the null hypothesis.
        range: [0, 1]
        interpretation: A low p-value (typically < 0.05) indicates that we can reject the null hypothesis, suggesting that the correlation matrix is not an identity matrix and that factor analysis may be appropriate.
    
    degrees_of_freedom: 15
        definition: The degrees of freedom for the test. DF is calculated based on the number of variables and is used to determine the critical value for the chi-square distribution when interpreting the test results.
        range: [0, ∞)
        interpretation: Higher degrees of freedom indicate more variables in the analysis, which can affect the chi-square statistic and p-value. The degrees of freedom are used in conjunction with the chi-square value to assess the statistical significance of the test results.
    
    Interpretation
    --------------
    p_value < 0.05
        Reject H0.
        Factor analysis can continue.

    p_value >= 0.05
        Fail to reject H0.
        Factor analysis is not recommended.

    Notes
    -----
    This function is used as a quality gate before factor extraction.

    Factor analysis is generally not recommended when:

        p_value >= 0.05
        Fail to reject H0.
        Factor analysis is not recommended.
    """
    logger.info("Calculating Bartlett's Test of Sphericity manually.")

    # Copy df
    df_work = df.copy()

    # if target_variable is provided, drop it from the dataframe before calculating Bartlett test and log the dropped target variable
    # if target_variable is not provided, log a warning that target variable is not provided and Bartlett test will be calculated for all variables in the dataframe
    if target_variable is None:
        logger.warning("Target variable is not provided. Bartlett test will be calculated for all variables in the dataframe.")
    elif target_variable in df_work.columns:
        df_work = df_work.drop(columns=[target_variable])
        logger.info(f"Dropped target variable '{target_variable}' from dataframe for Bartlett test calculation.")
    else:
        error_message = f"Target variable '{target_variable}' not found in the dataframe columns. " \
                        "Please check the target variable name and try again."
        logger.error(error_message)
        raise ValueError(error_message)
        
    # Number of samples (n) and number of variables (p) are needed to 
    # calculate the chi-square statistic and degrees of freedom for Bartlett's test.
    n_samples = df_work.shape[0]
    n_variables = df_work.shape[1]
    logger.info(f"Number of samples: {n_samples}")
    logger.info(f"Number of variables: {n_variables}")

    # Calculate the correlation matrix and its determinant. 
    # The correlation matrix is used to assess the relationships between variables, 
    # and its determinant is a key component in calculating the chi-square statistic for Bartlett's test.
    corr_matrix = df_work.corr().values
    logger.info("Correlation matrix (R) calculated for Bartlett's test.")

    # The determinant of the correlation matrix is used in the formula for the chi-square statistic in Bartlett's test.
    det_corr = np.linalg.det(corr_matrix)
    logger.info(f"Determinant of the correlation matrix (R): {det_corr:.8f}")

    # Check the determinant of the correlation matrix to ensure it is valid for Bartlett's test.
    # Range of determinant: [0, ∞)
    if det_corr > 0.70:
        logger.info("Determinant is close to 1. Correlation matrix may be close to an identity matrix !!!")

    if det_corr <= 0:
        error_message = f"Correlation matrix determinant is zero or negative. Det = {det_corr:.8f}."
        error_message += " Variables may be highly collinear, which can invalidate Bartlett's test."
        error_message += " Or, there may be an error in the data or preprocessing steps leading to an invalid correlation matrix."
        error_message += " Please check the correlation matrix for multicollinearity issues."
        error_message += "YOU CAN NOT TAKE NATURAL LOGARITHM OF ZERO OR NEGATIVE NUMBER !!!"
        raise ValueError(error_message)

    # Calculate the natural logarithm of the determinant of the correlation matrix, 
    # which is used in the formula for the chi-square statistic in Bartlett's test.
    ln_det_corr = np.log(det_corr)
    logger.info(f"ln(Det(R)) = {ln_det_corr:.8f}")


    # Calculation of the chi-square statistic for Bartlett's test:
    # The chi-square statistic is calculated using the formula:
    # Formula:
    # χ² = - (n - 1 - (2p + 5) / 6) * ln(det(R))
    # where:
    # n = number of samples
    # p = number of variables
    # correction_factor = (n - 1 - (2p + 5) / 6) is used to adjust the chi-square statistic for sample size and number of variables, which helps to improve the accuracy of the test results, especially in cases with small sample sizes or a large number of variables.
    
    # Calculate the correction factor
    correction_factor = (n_samples - 1 - ((2 * n_variables + 5) / 6))
    logger.info(f"Correction factor = {correction_factor:.4f}")

    # Calculate the chi-square statistic
    chi_square_value = -correction_factor * ln_det_corr
    logger.info(f"Chi-square statistic for Bartlett's test: {chi_square_value:.4f}")

    # Calculate DF for Bartlett's test:
    degrees_of_freedom = n_variables * (n_variables - 1) / 2
    logger.info(f"Degrees of freedom for Bartlett's test: {int(degrees_of_freedom)}")

    p_value = chi2.sf(chi_square_value, degrees_of_freedom)
    logger.info(f"P-value for Bartlett's test: {p_value:.8f}")

    # Interpretation of the test results is based on the p-value:
    interpretation, suggested_action = interpret_bartlett(p_value)
    logger.info("Interpretation of Bartlett's test completed.")

    # Prepare metadata for output
    metadata = {
        "testName": "Bartlett's Test of Sphericity",
        "testDefinition": "Bartlett's test evaluates whether the correlation matrix significantly differs from an identity matrix. A low p-value indicates that we can reject the null hypothesis, suggesting that the correlation matrix is not an identity matrix and that factor analysis may be appropriate.",
        "numSamples": f"{n_samples}",
        "numVariables": f"{n_variables}",
        "determinantCorrelation": f"{det_corr:.8f}",
        "logDeterminantCorrelation": f"{ln_det_corr:.8f}",
        "correctionFactor": f"{correction_factor:.4f}",
        "chiSquareValue": f"{chi_square_value:.4f}",
        "degreesOfFreedom": f"{int(degrees_of_freedom)}",
        "pValue": f"{p_value:.8f}",
        "interpretationThresholds": {
            "pValue < 0.05": "Reject H0. Factor analysis can continue.",
            "pValue >= 0.05": "Fail to reject H0. Factor analysis is not recommended."
        },
        "suggestedAction": suggested_action,
        "interpretation": interpretation
    }
    logger.info("Bartlett's test metadata prepared for output.")

    if output_dir:
        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        # Save Bartlett's test output to JSON file
        with open(f"{output_dir}/bartlett_test_output.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Bartlett's test output saved to {output_dir}/bartlett_test_output.json")

    logger.info("Bartlett's Test of Sphericity calculation completed successfully.")
    return chi_square_value, p_value, int(degrees_of_freedom), metadata

def validate_factor_analysis_suitability(
    kmo_model: float,
    bartlett_p_value: float,
    min_kmo: float = 0.50,
    max_bartlett_p_value: float = 0.05
):
    """
    Purpose
    -------
    Validates whether the dataset is suitable for factor analysis.

    This function acts as a quality gate using KMO and Bartlett test results.

    Parameters
    ----------
    kmo_model : float
        Overall KMO value.

    bartlett_p_value : float
        Bartlett test p-value.

    min_kmo : float, default=0.50
        Minimum acceptable KMO value.

    max_bartlett_p_value : float, default=0.05
        Maximum acceptable Bartlett p-value.

    Example Input
    -------------
    kmo_model = 0.7069
    bartlett_p_value = 0.000001
    min_kmo = 0.50
    max_bartlett_p_value = 0.05

    Returns
    -------
    None
        Returns nothing if validation passes.

    Example Output
    --------------
    None

    Interpretation
    --------------
    If this function raises no error, factor extraction can continue.

    Notes
    -----
    This function prevents factor analysis from continuing when
    the dataset does not have a suitable correlation structure.
    """

    if kmo_model < min_kmo:
        logger.warning("=" * 80)
        logger.warning("[WARNING] KMO TEST FAILED - INADEQUATE SAMPLING ADEQUACY")
        logger.warning("=" * 80)
        warning_message = f"KMO={kmo_model:.4f} is BELOW the minimum threshold of {min_kmo:.2f}.\n"
        warning_message += "Factor analysis is NOT RECOMMENDED due to inadequate sampling adequacy.\n"
        warning_message += "Proceed with caution or consider alternative dimensionality reduction methods."
        logger.warning(warning_message)
        logger.warning("=" * 80)

    if bartlett_p_value >= max_bartlett_p_value:
        logger.warning("=" * 80)
        logger.warning("[WARNING] BARTLETT'S TEST FAILED - INSUFFICIENT CORRELATION")
        logger.warning("=" * 80)
        warning_message = f"Bartlett's p-value={bartlett_p_value:.6f} is ABOVE the maximum threshold of {max_bartlett_p_value:.2f}.\n"
        warning_message += "Factor analysis is NOT RECOMMENDED due to insufficient correlation among variables.\n"
        warning_message += "Variables do not appear to be sufficiently correlated to justify factor analysis.\n"
        warning_message += "Proceed with caution or consider alternative dimensionality reduction methods."
        logger.warning(warning_message)
        logger.warning("=" * 80)
    
    if kmo_model >= min_kmo and bartlett_p_value < max_bartlett_p_value:
        logger.info("=" * 80)
        logger.info("[VALID] DATASET IS SUITABLE FOR FACTOR ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"KMO Model: {kmo_model:.6f} (meets minimum requirement of {min_kmo:.2f})")
        logger.info(f"Bartlett's p-value: {bartlett_p_value:.8f} (below maximum threshold of {max_bartlett_p_value:.2f})")
        logger.info("Factor extraction can proceed with confidence.")
        logger.info("=" * 80)

def calculate_eigenvalues(
        df: pd.DataFrame,
        target_variable: str | None = None,
        output_dir: str | None = None):
    """
    Purpose
    -------
    Calculates eigenvalues and eigenvectors from the correlation matrix.

    Eigenvalues are used to determine how many factors should be retained for Factor Analysis.

    Reference
    ---------
    https://metricgate.com/docs/eigenvalue-calculator/

    What is Eigenvalue?
    ---------------
    Eigenvalue represents the amount of variance in the original variables that is captured by each factor. 
    Higher eigenvalues indicate that the factor explains more variance.

    Factor Selection Rule
    ---------------------
    Kaiser criterion:

        retain factors with eigenvalue > 1

    Parameters
    ----------
    df : pd.DataFrame
        Encoded and standardized numeric dataframe.

    Returns
    -------
    eigen_table : pd.DataFrame
        Table containing factor number, eigenvalue,
        explained variance ratio and cumulative variance ratio.

    eigenvalues : np.ndarray
        Sorted eigenvalues in descending order.

    eigenvectors : np.ndarray
        Eigenvectors sorted according to eigenvalues.

    Example Output
    --------------
    eigen_table:

         factor  eigenvalue  explained_variance_ratio  cumulative_variance_ratio
    0  Factor_1      3.4200                    0.5700                     0.5700
    1  Factor_2      1.4100                    0.2350                     0.8050
    2  Factor_3      0.6200                    0.1033                     0.9083

    eigenvalues:

    [3.42, 1.41, 0.62]


    Interpretation
    --------------
    In the example above, Factor_1 and Factor_2 are retained
    because their eigenvalues are greater than 1.

    Notes
    -----
    The eigenvectors are later used to calculate factor loadings.
    """
    logger.info("Calculating eigenvalues and eigenvectors from the correlation matrix.")

    # Copy df
    df_work = df.copy()
    logger.info("Dataframe copied for eigenvalue calculation.")

    # if target_variable is provided, drop it from the dataframe before calculating eigenvalues and log the dropped target variable
    # if target_variable is not provided, log a warning that target variable is not provided and eigenvalues will be calculated for all variables in the dataframe
    if target_variable is None:
        logger.warning("Target variable is not provided. Eigenvalues will be calculated for all variables in the dataframe.")
    elif target_variable in df_work.columns:
        df_work = df_work.drop(columns=[target_variable])
        logger.info(f"Dropped target variable '{target_variable}' from dataframe for eigenvalue calculation.")
    else:
        error_message = f"Target variable '{target_variable}' not found in the dataframe columns. " \
                        "Please check the target variable name and try again."
        logger.error(error_message)
        raise ValueError(error_message)

    # Calculate the correlation matrix and convert it to numpy array
    corr_matrix = df_work.corr().values
    logger.info("Correlation matrix calculated for eigenvalue decomposition.")

    # ========================================================================
    # EIGENVALUE AND EIGENVECTOR CALCULATION
    # ========================================================================
    # PURPOSE:
    #   Extract eigenvalues and eigenvectors from the correlation matrix.
    #   These represent the variance explained by each principal component
    #   and the direction of maximum variance respectively.
    #
    # WHY CORRELATION MATRIX?
    #   Using the correlation matrix (instead of covariance) standardizes
    #   the variables, ensuring the factor structure is not dominated by
    #   variables with large scales. This allows fair comparison across
    #   all variables regardless of their measurement units.
    #
    # MATHEMATICAL FORMULATION:
    #   For a correlation matrix R, we solve the characteristic equation:
    #
    #       |R - λI| = 0
    #
    #   where:
    #       λ  = eigenvalue (variance explained by this principal component)
    #       I  = identity matrix
    #       |·| = determinant operator
    #
    #   Each eigenvalue λ has a corresponding eigenvector v that satisfies:
    #
    #       (R - λI)v = 0
    #
    #   or equivalently:
    #
    #       Rv = λv
    # LOGICAL NOTE:
    #   - We are finding smaller matrix repressenting input R Matrix with less dimensions.
    #   - Main idea is representing the original data with fewer dimensions while retaining as much variance as possible.
    #
    # SHAPES
    #   - Eigenvalues: (n,)
    #   - Eigenvectors: (n, n)
    #   - R: (n, n)
    #   where n: number of variables
    #
    # INTERPRETATION:
    #   - Eigenvalue magnitude: amount of variance captured by the factor
    #   - Eigenvector: loadings/weights indicating variable contributions
    #   - Larger eigenvalues indicate more important factors
    #   - Sum of eigenvalues equals the number of variables (for correlation matrix)
    #   - Sum of explained variance ratios equals 1 (or 100%) when all factors are retained
    #
    # IMPLEMENTATION:
    #   NumPy's linear algebra module (linalg.eigh) efficiently computes
    #   eigenvalues and eigenvectors for symmetric matrices like correlations.
    # ======================================================================

    # Calculate eigenvalues and eigenvectors using NumPy's linear algebra function for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    logger.info("Eigenvalues calculated:")
    for i, eigenvalue in enumerate(eigenvalues):
        logger.info(f"Eigenvalue {i + 1}: {eigenvalue:.8f}")
    logger.info(f"Eigenvectors calculated. {eigenvectors.shape[0]} variables and {eigenvectors.shape[1]} factors.")
    
    # Take real parts of eigenvalues
    eigenvalues = np.real(eigenvalues)
    logger.info("Real parts of eigenvalues extracted:")
    for i, eigenvalue in enumerate(eigenvalues):
        logger.info(f"Eigenvalue {i + 1}: {eigenvalue:.8f}")

    # ========================================================================
    # SORT EIGENVALUES AND EIGENVECTORS IN DESCENDING ORDER
    # ========================================================================
    # WHY SORTING IS NECESSARY:
    #   NumPy's eigh() returns eigenvalues in ascending order by default.
    #   We need descending order to:
    #   1. Rank factors by explained variance (largest first)
    #   2. Apply Kaiser criterion (retain eigenvalue > 1) on important factors
    #   3. Improve interpretability of factor analysis results
    # ========================================================================
    sorted_index = np.argsort(eigenvalues)[::-1]
    logger.info(f"Eigenvalues sorted in descending order. Sorted indices: {sorted_index}")

    # This is very IMPORTANT: 
    # We must sort both eigenvalues and eigenvectors according to the same indices to maintain the correct correspondence between them.
    # Never FORGET THIS STEP !!! must sprt both eigenvalues and eigenvectors according to the same sorted indices.
    # Rows of eigenvectors correspond to variables, and columns correspond to factors.
    # We just rearrange the columns of eigenvectors to match the sorted order of eigenvalues.
    # NOT CHANGE THE ROWS OF EIGENVECTORS, ONLY CHANGE THE COLUMNS TO MATCH THE SORTED EIGENVALUES.
    eigenvalues = eigenvalues[sorted_index]
    eigenvectors = eigenvectors[:, sorted_index]
    logger.info("Eigenvalues and eigenvectors sorted according to eigenvalues.")

    # Distribution of eigenvalues can be informative about the factor structure.
    for i, eigenvalue in enumerate(eigenvalues):
        # Log Table Formatted Eigenvalues with Explained Variance Ratios
        logger.info(f"Eigenvalue {i + 1}: {eigenvalue:.8f} | Variance Explained: {eigenvalue / eigenvalues.sum():.4f} | Cumulative Variance Explained: {np.cumsum(eigenvalues / eigenvalues.sum())[i]:.4f}")

    eigen_table = pd.DataFrame({
        "factor": [f"Factor_{i + 1}" for i in range(len(eigenvalues))],
        "eigenvalue": np.round(eigenvalues, 8),
        "explainedVarianceRatio": np.round(eigenvalues / eigenvalues.sum(), 4),
        "cumulativeVarianceRatio": np.round(np.cumsum(eigenvalues / eigenvalues.sum()), 4)
    })
    logger.info("Eigenvalues table created with factor names, eigenvalues, explained variance ratios, and cumulative variance ratios.")

    # Get Sum and Cum Sum of Eigenvalues
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance_ratio = explained_variance_ratio.cumsum()

    # Prepare metadata for output
    metadata = {
        "eigenvalueDefinition": "Eigenvalue represents the amount of variance in the original variables that is captured by each factor.\
Higher eigenvalues indicate that the factor explains more variance.",
        "factorSelectionRule": "Kaiser criterion: retain factors with eigenvalue > 1",
        "numFactors": len(eigenvalues), # Total number of factors (equal to number of variables)
        "numberOfFactorsByKaiserCriterion": int((eigenvalues > 1).sum()),
        "explainedVarianceOfKaiserFactors": round(explained_variance_ratio[eigenvalues > 1].sum(), 4),
        "eigenvalues": eigenvalues.round(8).tolist(),
        "eigenvectorsShape": {
            "rows": eigenvectors.shape[0], 
            "columns": eigenvectors.shape[1]
            },
        "eigenTable": eigen_table.to_dict(orient="records"),
        "numberOfFactorExplained50PercentVariance": int((cumulative_variance_ratio).searchsorted(0.50) + 1),
        "numberOfFactorExplained75PercentVariance": int((cumulative_variance_ratio).searchsorted(0.75) + 1),
        "numberOfFactorExplained80PercentVariance": int((cumulative_variance_ratio).searchsorted(0.80) + 1),
        "numberOfFactorExplained90PercentVariance": int((cumulative_variance_ratio).searchsorted(0.90) + 1),
        "numberOfFactorExplained95PercentVariance": int((cumulative_variance_ratio).searchsorted(0.95) + 1),
        "explainedVarianceOfKaiserThreshold_1_0": round(explained_variance_ratio[eigenvalues > 1.0].sum(), 4),
        "explainedVarianceOfKaiserThreshold_1_2": round(explained_variance_ratio[eigenvalues > 1.2].sum(), 4),
        "explainedVarianceOfKaiserThreshold_1_5": round(explained_variance_ratio[eigenvalues > 1.5].sum(), 4)
    }
    logger.info("EigenValues metadata prepared for output.")

    if output_dir:
        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        # Save EigenValues output to JSON file
        with open(f"{output_dir}/eigenvalues_output.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"EigenValues output saved to {output_dir}/eigenvalues_output.json")

    logger.info("EigenValues calculation completed successfully.")
    return eigen_table, eigenvalues, eigenvectors

def varimax(
    loadings: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-6
):
    """
    Purpose
    -------
    Applies Varimax rotation to factor loadings.

    Varimax rotation makes factor loadings easier to interpret
    by pushing loadings closer to either high or low values.

    Parameters
    ----------
    loadings : np.ndarray
        Unrotated factor loading matrix.

    gamma : float, default=1.0
        Varimax parameter.

    max_iter : int, default=100
        Maximum number of iterations.

    tolerance : float, default=1e-6
        Convergence tolerance.

    Example Input
    -------------
    loadings:

    [[0.62, 0.48],
     [0.60, 0.51],
     [0.55, 0.58],
     [0.49, 0.61]]

    Returns
    -------
    rotated_loadings : np.ndarray
        Rotated factor loading matrix.

    Example Output
    --------------
    rotated_loadings:

    [[0.90, 0.10],
     [0.85, 0.15],
     [0.10, 0.88],
     [0.15, 0.84]]

    Interpretation
    --------------
    Before rotation, variables may load moderately on multiple factors.

    After rotation, each variable tends to load strongly on one factor,
    which makes factor grouping easier.

    Notes
    -----
    Rotation does not change the total explanatory power.
    It only improves interpretability.
    """

    n_rows, n_cols = loadings.shape

    rotation_matrix = np.eye(n_cols)

    previous_value = 0

    for _ in range(max_iter):
        rotated_loadings = np.dot(loadings, rotation_matrix)

        u, s, vh = np.linalg.svd(
            np.dot(
                loadings.T,
                rotated_loadings ** 3
                - (gamma / n_rows)
                * np.dot(
                    rotated_loadings,
                    np.diag(
                        np.diag(
                            np.dot(rotated_loadings.T, rotated_loadings)
                        )
                    )
                )
            )
        )

        rotation_matrix = np.dot(u, vh)

        current_value = s.sum()

        if previous_value != 0 and current_value / previous_value < 1 + tolerance:
            break

        previous_value = current_value

    return np.dot(loadings, rotation_matrix)

def calculate_factor_loadings(
    df: pd.DataFrame,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    rotation: str = "varimax",
    eigenvalue_threshold: float = 1.0,
    target_variable: str | None = None,
    output_dir: str | None = None
):
    """
    Purpose
    -------
    Calculates factor loadings using eigenvalues and eigenvectors.

    Factor loadings show how strongly each variable is associated
    with each factor.

    Calculation Logic
    -----------------
    For each retained factor:

        loading = eigenvector * sqrt(eigenvalue)

    Factor Selection Rule
    ---------------------
    Retain factors with:

        eigenvalue > 1

    Parameters
    ----------
    df : pd.DataFrame
        Encoded and standardized numeric dataframe.

    eigenvalues : np.ndarray
        Sorted eigenvalues.

    eigenvectors : np.ndarray
        Sorted eigenvectors.

    rotation : str, default="varimax"
        Rotation method.

        Supported values:
        - "varimax"
        - None
    
    eigenvalue_threshold : float, default=1.0
        Minimum eigenvalue for factor retention.

    target_variable : str | None, default=None
        Specific variable to focus on.

    output_dir : str | None, default=None
        Directory to save output files.

    Example Input
    -------------
    eigenvalues:

    [3.42, 1.41, 0.62]

    rotation:

    "varimax"

    Returns
    -------
    loadings_df : pd.DataFrame
        Factor loading table.

    n_factors : int
        Number of retained factors.

    Example Output
    --------------
    loadings_df:

              Factor_1  Factor_2
    AGE          0.9100    0.1200
    TENURE       0.8800    0.1000
    INCOME       0.1800    0.8600
    BALANCE      0.1500    0.8200

    n_factors:

    2

    Interpretation
    --------------
    AGE and TENURE are mainly associated with Factor_1.

    INCOME and BALANCE are mainly associated with Factor_2.

    Notes
    -----
    This implementation uses a principal-factor-like approach based on the correlation matrix decomposition.
    """
    logger.info("Starting factor loadings calculation.")

    # Check if eigenvalues are provided and sorted in descending order
    if eigenvalues is None or len(eigenvalues) == 0:
        error_message = "Eigenvalues array is empty. Please provide valid eigenvalues for factor loadings calculation."
        logger.error(error_message)
        raise ValueError(error_message)
    if not np.all(np.diff(eigenvalues) <= 0):
        error_message = "Eigenvalues are not sorted in descending order. Please sort eigenvalues before calculating factor loadings."
        logger.error(error_message)
        raise ValueError(error_message)
    else:
        # Log the provided eigenvalues for factor loadings calculation
        logger.info("GIVEN EIGEN VALUES DESCENDING ORDER")
        for i, eigenvalue in enumerate(eigenvalues):
            logger.info(f"Eigenvalue {i + 1}: {eigenvalue:.8f} | Var Explained: {eigenvalue / eigenvalues.sum():.4f} | Cum Var Explained: {np.cumsum(eigenvalues / eigenvalues.sum())[i]:.4f}")

    # Copy df
    df_work = df.copy()
    logger.info("Dataframe copied for factor loadings calculation.")

    # if target_variable is provided, drop it from the dataframe before calculating factor loadings and log the dropped target variable
    # if target_variable is not provided, log a warning that target variable is not provided and factor loadings will be calculated for all variables in the dataframe
    if target_variable is None:
        logger.warning("Target variable is not provided. Factor loadings will be calculated for all variables in the dataframe.")
    elif target_variable in df_work.columns:
        df_work = df_work.drop(columns=[target_variable])
        logger.info(f"Dropped target variable '{target_variable}' from dataframe for factor loadings calculation.")
    else:
        error_message = f"Target variable '{target_variable}' not found in the dataframe columns. " \
                        "Please check the target variable name and try again."
        logger.error(error_message)
        raise ValueError(error_message)

    # Determine the number of factors to retain based on the eigenvalue threshold
    n_factors = int((eigenvalues > eigenvalue_threshold).sum())
    logger.info(f"Number of factors retained based on eigenvalue threshold > {eigenvalue_threshold}: {n_factors}")

    if n_factors < 1:
        error_message = f"No factors retained. No eigenvalues greater than {eigenvalue_threshold}. "\
                        "Please consider lowering the eigenvalue threshold or reviewing the eigenvalues for factor retention."
        logger.error(error_message)
        raise ValueError(error_message)

    selected_eigenvalues = eigenvalues[:n_factors]
    # Log the selected eigenvalues for retained factors
    logger.info("-" * 50)
    logger.info("Selected eigenvalues for retained factors:")
    for i, eigenvalue in enumerate(selected_eigenvalues):
        logger.info(f"Selected Eigenvalue for Factor_{i + 1}: {eigenvalue:.8f} | Var Explained: {eigenvalue / eigenvalues.sum():.4f} | Cum Var Explained: {np.cumsum(eigenvalues / eigenvalues.sum())[i]:.4f}")
    logger.info(f"Total variance explained by retained factors ({n_factors}): {selected_eigenvalues.sum() / eigenvalues.sum():.4f}")
    logger.info("-" * 50)
    selected_eigenvectors = eigenvectors[:, :n_factors]
    logger.info(f"Shape of selected eigenvectors for retained factors: {selected_eigenvectors.shape[0]} variables and {selected_eigenvectors.shape[1]} factors.")

    # ========================================================================
    # FACTOR LOADINGS CALCULATION
    # ========================================================================
    # FORMULA:
    #   loading = eigenvector * sqrt(eigenvalue)
    #
    # MATHEMATICAL INTERPRETATION:
    #   Eigenvector:   Direction/orientation of the factor in variable space
    #   Eigenvalue:    Magnitude of variance explained by that factor
    #   sqrt(λ):       Scaling factor that converts direction to strength
    #
    # WHY MULTIPLY BY SQRT(EIGENVALUE)?
    #   - Eigenvectors are unit vectors (length = 1) showing pure direction
    #   - Eigenvalues indicate variance magnitude but are unscaled
    #   - Multiplying combines both: direction × strength = association
    #   - Result: loadings reflect how strongly each variable relates to each factor
    #
    # INTERPRETATION:
    #   Higher |loading| = variable is important for this factor
    #   Lower |loading| = variable has weak association with this factor
    #   Sign = direction of association (+ or -)
    #
    # RESULT PROPERTIES:
    #   - Each loading value in [-1, 1] range (approximately)
    #   - Represents correlation-like measure between variable and factor
    #   - Sum of squared loadings per factor ≈ eigenvalue of that factor
    # ======================================================================
    loadings = selected_eigenvectors * np.sqrt(selected_eigenvalues)
    logger.info("Factor loadings calculated using eigenvectors and eigenvalues.")
    # Log the calculated factor loadings before rotation
    logger.info("Calculated factor loadings before rotation:")
    logger.info("-" * 50)
    for i in range(loadings.shape[1]):
        logger.info(f"\tFactor_{i + 1} loadings:")
        for j in range(loadings.shape[0]):
            logger.info(f"\t\tVariable: {df_work.columns[j]}: {loadings[j, i]:.4f}")
    logger.info("-" * 50)

    # Apply rotation if specified
    if rotation == "varimax":
        logger.info("Applying Varimax rotation to factor loadings.")
        loadings = varimax(loadings)
    elif rotation is None:
        logger.info("No rotation applied to factor loadings.")
        pass
    else:
        error_message = f"Unsupported rotation method: {rotation}. " \
                        "Use rotation='varimax' or rotation=None."
        logger.error(error_message)
        raise ValueError(error_message)

    loadings_df = pd.DataFrame(
        loadings,
        index=df_work.columns,
        columns=[f"Factor_{i + 1}" for i in range(n_factors)]
    )
    logger.info("Factor loadings calculated and stored in DataFrame.")

    # Prepare metadata for output
    metadata = {
        "factorLoadingDefinition": "Factor loadings show how strongly each variable is associated with each factor. They are calculated using the formula: loading = eigenvector * sqrt(eigenvalue). Higher absolute loadings indicate stronger associations between variables and factors.",
        "factorSelectionRule": f"Retain factors with eigenvalue > {eigenvalue_threshold}",
        "rotationMethod": rotation,
        "numberOfFactorsRetained": n_factors,
        "loadingsShape": {
            "rows": loadings_df.shape[0],
            "columns": loadings_df.shape[1]
        },
        "loadingsTable": (loadings_df.round(4).reset_index(names="variable").to_dict(orient="records"))
    }
    logger.info("Factor loadings metadata prepared for output.")

    if output_dir:
        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        # Save factor loadings metadata to JSON file
        with open(f"{output_dir}/factor_loadings_output.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Factor loadings metadata saved to {output_dir}/factor_loadings_output.json")

    logger.info("Factor loadings calculation completed.")
    return loadings_df, n_factors

def create_factor_groups(
    loadings_df: pd.DataFrame,
    loading_threshold: float = 0.50,
    output_dir: str | None = None
):
    """
    Purpose
    -------
    Assigns each variable to the factor where it has the highest
    absolute loading.

    This function converts factor loading results into variable groups.

    Parameters
    ----------
    loadings_df : pd.DataFrame
        Factor loading table.

    loading_threshold : float, default=0.50
        Minimum absolute loading required for a variable to be considered
        a strong member of a factor group.
    
    output_dir : str | None, default=None
        Directory to save output files.

    Example Input
    -------------
              Factor_1  Factor_2
    AGE          0.9100    0.1200
    TENURE       0.8800    0.1000
    INCOME       0.1800    0.8600
    BALANCE      0.1500    0.8200
    CITY_A       0.1100    0.6400

    Returns
    -------
    grouping_table : pd.DataFrame
        Variable-level factor assignment table.

    grouped_summary : pd.DataFrame
        Factor-level grouped variable summary.

    Example Output
    --------------
    grouping_table:

       variable assigned_factor  max_abs_loading  loading_value  group_status
    0       AGE        Factor_1           0.9100         0.9100  STRONG_GROUP
    1    TENURE        Factor_1           0.8800         0.8800  STRONG_GROUP
    2    INCOME        Factor_2           0.8600         0.8600  STRONG_GROUP
    3   BALANCE        Factor_2           0.8200         0.8200  STRONG_GROUP
    4    CITY_A        Factor_2           0.6400         0.6400  STRONG_GROUP

    grouped_summary:

      assigned_factor        variables_in_group
    0        Factor_1             [AGE, TENURE]
    1        Factor_2  [INCOME, BALANCE, CITY_A]

    Interpretation
    --------------
    Variables assigned to the same factor are interpreted as belonging
    to the same underlying concept.

    Notes
    -----
    After grouping, a representative variable can be selected from
    each group based on:

    - Information Value
    - Business meaning
    - Missing ratio
    - Stability
    - Model performance
    """
    logger.info("Creating factor groups based on factor loadings.")

    abs_loadings = loadings_df.abs()
    logger.info("Absolute values of factor loadings calculated for group assignment.")
    # log the absolute loadings for debugging
    logger.info("Absolute factor loadings:")
    for i in range(abs_loadings.shape[1]):
        logger.info(f"\tFactor_{i + 1} absolute loadings:")
        for j in range(abs_loadings.shape[0]):
            logger.info(f"\t\tVariable: {abs_loadings.index[j]}: {abs_loadings.iloc[j, i]:.4f}")
    logger.info("-" * 50)

    grouping_table = pd.DataFrame({
        "variable": abs_loadings.index,
        "assignedFactor": abs_loadings.idxmax(axis=1),
        "maxAbsLoading": abs_loadings.max(axis=1)
    })
    logger.info("Created grouping table (dataframe) with variable names, assigned factors, and maximum absolute loadings.")
    logger.info("First 5 rows of the grouping table before adding loading values:")
    logger.info(grouping_table.head(5))
    logger.info("-" * 50)

    # Retrieve the actual loading value (with sign) for each variable's assigned factor
    # Example: If AGE has maxAbsLoading=0.91, loadingValue could be +0.91 or -0.91 based on direction
    grouping_table["loadingValue"] = grouping_table.apply(
        lambda row: loadings_df.loc[row["variable"], row["assignedFactor"]],
        axis=1
    )
    logger.info("Added actual loading values (with signs) to the grouping table.")
    logger.info("First 5 rows of the grouping table after adding loading values:")
    logger.info(grouping_table.head(5))
    logger.info("-" * 50)

    # ========================================================================
    # ASSIGN GROUP STATUS BASED ON LOADING THRESHOLD
    # ========================================================================
    # Classifies each variable as either STRONG_GROUP or WEAK_LOADING based on
    # whether its maximum absolute loading meets the threshold.
    # Example: If loading_threshold=0.50 and maxAbsLoading=0.65 → STRONG_GROUP
    # ========================================================================
    grouping_table["groupStatus"] = np.where(
        grouping_table["maxAbsLoading"] >= loading_threshold,
        "STRONG_GROUP",
        "WEAK_LOADING"
    )
    logger.info(f"Assigned group status based on loading threshold of {loading_threshold}.")
    logger.info("First 5 rows of the grouping table after assigning group status:")
    logger.info(grouping_table.head(5))
    logger.info("-" * 50)

    # ========================================================================
    # SORTING GROUPING TABLE FOR BETTER INTERPRETATION
    # ========================================================================
    # Sorts the grouping table by assigned factor and maximum absolute loading
    # in descending order for easier interpretation.
    # Example: Variables with higher loadings appear first within each factor.
    # ========================================================================
    grouping_table = grouping_table.sort_values(
        by=["assignedFactor", "maxAbsLoading"],
        ascending=[True, False]
    ).reset_index(drop=True)
    logger.info("First 5 rows of the grouping table after sorting:")
    logger.info(grouping_table.head(5))
    logger.info("-" * 50)

    # ========================================================================
    # CREATE GROUPED SUMMARY TABLE
    # ========================================================================
    # Creates a summary table that lists the variables assigned to each factor group.
    # Example: Factor_1 → [AGE, TENURE], Factor_2 → [INCOME, BALANCE, CITY_A]
    # ========================================================================
    grouped_summary = (
        grouping_table[grouping_table["groupStatus"] == "STRONG_GROUP"]
        .groupby("assignedFactor")["variable"]
        .apply(list)
        .reset_index()
        .rename(columns={"variable": "variablesInGroup"})
    )
    logger.info("Created grouped summary table with assigned factors and their corresponding variables.")
    logger.info("First 5 rows of the grouped summary table:")
    logger.info(grouped_summary.head(5))
    logger.info("-" * 50)

    # Prepare metadata for output
    metadata = {
        "groupingLogic": "Each variable is assigned to the factor where it has the highest absolute loading. Variables with max absolute loading above the specified threshold are classified as STRONG_GROUP, while those below are classified as WEAK_LOADING.",
        "loadingThreshold": loading_threshold,
        "groupStatusDefinition": {
            "STRONG_GROUP": f"Variable has a maximum absolute loading >= {loading_threshold} on its assigned factor, indicating a strong association with that factor.",
            "WEAK_LOADING": f"Variable has a maximum absolute loading < {loading_threshold} on its assigned factor, indicating a weak association with that factor."
        },
        "groupingTableSample": grouping_table.head(5).to_dict(orient="records"),
        "groupedSummarySample": grouped_summary.head(5).to_dict(orient="records")
    }

    logger.info("Factor grouping metadata prepared for output.")

    # Save grouping results to JSON file if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/factor_grouping_output.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Factor grouping metadata saved to {output_dir}/factor_grouping_output.json")
    
    # Save grouping table and grouped summary to CSV files if output directory is specified
    if output_dir:
        grouping_table.to_csv(f"{output_dir}/grouping_table.csv", index=False)
        grouped_summary.to_csv(f"{output_dir}/grouped_summary.csv", index=False)
        logger.info(f"Grouping table saved to {output_dir}/grouping_table.csv")
        logger.info(f"Grouped summary saved to {output_dir}/grouped_summary.csv")

    return grouping_table, grouped_summary, metadata

def run_factor_analysis(
    df: pd.DataFrame,
    target_variable: str | None = None,
    drop_last: bool = True,
    fill_strategy_numeric: str = "mean",
    encoding_strategy_categorical: str = "ordinal",
    rotation: str = "varimax",
    eigenvalue_threshold: float = 1.0,
    loading_threshold: float = 0.50,
    output_dir: str | None = None
) -> dict:
    """
    Purpose
    -------
    Executes a complete factor analysis pipeline and returns all outputs as a dictionary.

    This function consolidates all factor analysis steps:
    1. Data preparation (encoding, filling missing values)
    2. KMO (Kaiser-Meyer-Olkin) test
    3. Bartlett's Test of Sphericity
    4. Eigenvalue calculation
    5. Factor loadings calculation
    6. Factor grouping

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe for factor analysis.

    target_variable : str | None, default=None
        Name of the target variable to exclude from analysis.

    drop_last : bool, default=True
        Whether to drop the last category when encoding categorical variables.

    fill_strategy_numeric : str, default="mean"
        Strategy for filling missing numeric values: "mean", "median", etc.

    encoding_strategy_categorical : str, default="ordinal"
        Strategy for encoding categorical variables.

    rotation : str, default="varimax"
        Rotation method for factor loadings: "varimax" or None.

    eigenvalue_threshold : float, default=1.0
        Minimum eigenvalue for retaining factors (Kaiser criterion).

    loading_threshold : float, default=0.50
        Minimum absolute loading for classifying as STRONG_GROUP.

    output_dir : str | None, default=None
        Directory to save output files. If None, outputs are not saved to files.

    Returns
    -------
    dict
        Dictionary containing all factor analysis results:
        {
            "prepared_data": pd.DataFrame,
            "preparation_metadata": dict,
            "kmo_per_variable": pd.Series,
            "kmo_model": float,
            "kmo_metadata": dict,
            "bartlett_chi_square": float,
            "bartlett_p_value": float,
            "bartlett_df": int,
            "bartlett_metadata": dict,
            "eigen_table": pd.DataFrame,
            "eigenvalues": np.ndarray,
            "eigenvectors": np.ndarray,
            "loadings_df": pd.DataFrame,
            "n_factors": int,
            "factor_loadings_metadata": dict,
            "grouping_table": pd.DataFrame,
            "grouped_summary": pd.DataFrame,
            "grouping_metadata": dict
        }

    Example
    -------
    >>> result = run_factor_analysis(
    ...     df=df,
    ...     target_variable="TARGET",
    ...     output_dir="outputs/factor_analysis"
    ... )
    >>> kmo_value = result["kmo_model"]
    >>> factor_groups = result["grouped_summary"]

    Notes
    -----
    - All steps are executed in sequence
    - Quality gates (KMO, Bartlett) are checked; errors are raised if thresholds not met
    - Outputs can optionally be saved to JSON and CSV files
    """
    # Display pipeline startup information with ASCII frame
    logger.info("")
    logger.info("+" + "=" * 78 + "+")
    logger.info("|" + " " * 78 + "|")
    logger.info("|" + "COMPREHENSIVE FACTOR ANALYSIS PIPELINE STARTED".center(78) + "|")
    logger.info("|" + " " * 78 + "|")
    logger.info("+" + "=" * 78 + "+")
    
    # Display pipeline parameters
    logger.info("")
    logger.info("[PIPELINE PARAMETERS]")
    logger.info(f"  Target Variable ..................... {target_variable if target_variable else 'None (all variables)'}")
    logger.info(f"  Drop Last Category .................. {drop_last}")
    logger.info(f"  Fill Strategy (Numeric) ............ {fill_strategy_numeric}")
    logger.info(f"  Encoding Strategy (Categorical) ... {encoding_strategy_categorical}")
    logger.info(f"  Rotation Method ..................... {rotation if rotation else 'None'}")
    logger.info(f"  Eigenvalue Threshold ............... {eigenvalue_threshold}")
    logger.info(f"  Loading Threshold .................. {loading_threshold}")
    logger.info(f"  Output Directory ................... {output_dir if output_dir else 'None (outputs not saved to files)'}")
    logger.info("")
    
    results = {}

    try:
        # Step 1: Data Preparation
        logger.info("\n[STEP 1/6] Preparing data for factor analysis...")
        df_prepared, metadata_of_preparation = prepare_factor_analysis_data(
            df=df,
            target_variable=target_variable,
            drop_last=drop_last,
            fill_strategy_numeric=fill_strategy_numeric,
            encoding_strategy_categorical=encoding_strategy_categorical
        )
        results["prepared_data"] = df_prepared
        results["preparation_metadata"] = metadata_of_preparation
        logger.info("Data preparation completed.")

        # Step 2: Calculate KMO
        logger.info("\n[STEP 2/6] Calculating Kaiser-Meyer-Olkin (KMO) test...")
        kmo_per_variable, kmo_model, kmo_metadata = calculate_kmo_manual(
            df_prepared,
            target_variable=target_variable,
            output_dir=output_dir
        )
        results["kmo_per_variable"] = kmo_per_variable
        results["kmo_model"] = kmo_model
        results["kmo_metadata"] = kmo_metadata
        logger.info("KMO calculation completed.")

        # Step 3: Calculate Bartlett's Test of Sphericity
        logger.info("\n[STEP 3/6] Calculating Bartlett's Test of Sphericity...")
        bartlett_chi_square, bartlett_p_value, bartlett_df, bartlett_metadata = calculate_bartlett_manual(
            df_prepared,
            target_variable=target_variable,
            output_dir=output_dir
        )
        results["bartlett_chi_square"] = bartlett_chi_square
        results["bartlett_p_value"] = bartlett_p_value
        results["bartlett_df"] = bartlett_df
        results["bartlett_metadata"] = bartlett_metadata
        logger.info("Bartlett's test completed.")

        # Step 4: Validate Factor Analysis Suitability
        logger.info("\n[VALIDATION] Checking factor analysis suitability...")
        validate_factor_analysis_suitability(
            kmo_model=kmo_model,
            bartlett_p_value=bartlett_p_value,
            min_kmo=0.50,
            max_bartlett_p_value=0.05
        )
        logger.info("Dataset is validated for factor analysis suitability.")

        # Step 5: Calculate Eigenvalues
        logger.info("\n[STEP 4/6] Calculating eigenvalues and eigenvectors...")
        eigen_table, eigenvalues, eigenvectors = calculate_eigenvalues(
            df_prepared,
            target_variable=target_variable,
            output_dir=output_dir
        )
        results["eigen_table"] = eigen_table
        results["eigenvalues"] = eigenvalues
        results["eigenvectors"] = eigenvectors
        logger.info("Eigenvalue calculation completed.")

        # Step 6: Calculate Factor Loadings
        logger.info("\n[STEP 5/6] Calculating factor loadings...")
        loadings_df, n_factors = calculate_factor_loadings(
            df_prepared,
            eigenvalues,
            eigenvectors,
            rotation=rotation,
            eigenvalue_threshold=eigenvalue_threshold,
            target_variable=target_variable,
            output_dir=output_dir
        )
        results["loadings_df"] = loadings_df
        results["n_factors"] = n_factors
        results["factor_loadings_metadata"] = {
            "factorLoadingDefinition": "Factor loadings show how strongly each variable is associated with each factor.",
            "factorSelectionRule": f"Retain factors with eigenvalue > {eigenvalue_threshold}",
            "rotationMethod": rotation,
            "numberOfFactorsRetained": n_factors,
            "loadingsShape": {"rows": loadings_df.shape[0], "columns": loadings_df.shape[1]}
        }
        logger.info("Factor loadings calculation completed.")

        # Step 7: Create Factor Groups
        logger.info("\n[STEP 6/6] Creating factor groups...")
        grouping_table, grouped_summary, grouping_metadata = create_factor_groups(
            loadings_df,
            loading_threshold=loading_threshold,
            output_dir=output_dir
        )
        results["grouping_table"] = grouping_table
        results["grouped_summary"] = grouped_summary
        results["grouping_metadata"] = grouping_metadata
        logger.info("Factor grouping completed.")

        logger.info("\n" + "=" * 80)
        logger.info("FACTOR ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("")
        logger.info("[SUMMARY]")
        logger.info(f"  KMO Model Score .................... {results['kmo_model']:.6f}")
        logger.info(f"  Bartlett p-value .................. {results['bartlett_p_value']:.8f}")
        logger.info(f"  Number of Factors Retained ....... {results['n_factors']}")
        logger.info(f"  Number of Factor Groups Created .. {len(results['grouped_summary'])}")
        logger.info("")

    except Exception as e:
        logger.error(f"Error in factor analysis pipeline: {str(e)}", exc_info=True)
        raise

    return results


if __name__ == "__main__":
    # Get Metadata as json
    metadata_path = "inputs/sample/datatypes.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    column_dtypes = metadata.get("column_dtypes", {})
    # Get the sample data
    input_csv_path = "inputs/sample/uci_credit_card_dataset.csv"
    df = pd.read_csv(input_csv_path, dtype=column_dtypes)
    # Log Column names and data types
    for column in df.columns:
        logger.info(f"Column: {column}: {column_dtypes.get(column, 'Unknown')} | Unique Values: {df[column].nunique()}")
    logger.info("Sample data loaded successfully.")

    run_method = "full_pipeline"
    logger.info(f"Running factor analysis with method: {run_method}")

    if run_method == "full_pipeline":
        run_factor_analysis(
            df=df,
            target_variable="TARGET",
            drop_last=True,
            fill_strategy_numeric="mean",
            encoding_strategy_categorical="ordinal",
            rotation="varimax",
            eigenvalue_threshold=1.0,
            loading_threshold=0.50,
            output_dir="outputs/factor_analysis"
        )
    
    else:
        logger.info("Manual run method for review.")

        # Data Preparation and Factor Analysis Grouping
        df_prepared, metadata_of_preparation = prepare_factor_analysis_data(
            df=df, 
            target_variable="TARGET", 
            drop_last=True, 
            fill_strategy_numeric="mean", 
            encoding_strategy_categorical="ordinal")
        logger.info("Data prepared for factor analysis.")

        # Calculate KMO
        kmo_per_variable, kmo_model, kmo_metadata = calculate_kmo_manual(
            df_prepared, 
            target_variable="TARGET", 
            output_dir="outputs/factor_analysis"
            )

        # Calculate Bartlett's Test of Sphericity
        bartlett_chi_square, bartlett_p_value, bartlett_df, bartlett_metadata = calculate_bartlett_manual(
            df_prepared, 
            target_variable="TARGET", 
            output_dir="outputs/factor_analysis"
            )
        
        # Calculate Eigenvalues and Eigenvectors
        eigen_table, eigenvalues, eigenvectors = calculate_eigenvalues(
            df_prepared, 
            target_variable="TARGET", 
            output_dir="outputs/factor_analysis"
            )
        
        # Calculate Factor Loadings
        loadings_df, n_factors = calculate_factor_loadings(
            df_prepared, 
            eigenvalues, 
            eigenvectors, 
            rotation="varimax", 
            eigenvalue_threshold=1.0, 
            target_variable="TARGET", 
            output_dir="outputs/factor_analysis"
            )
        
        # Create Factor Groups
        grouping_table, grouped_summary, grouping_metadata = create_factor_groups(
            loadings_df, 
            loading_threshold=0.50, 
            output_dir="outputs/factor_analysis"
            )