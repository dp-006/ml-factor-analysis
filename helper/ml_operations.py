'''
Accelera Consulting

This module provides utility functions for interpreting statistical results in machine learning operations (MLOps). 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split 
from logging_config.logger_config import get_logger
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


logger_name = "mlops.ml_operations"
logger_file_name = "ml_operations.log"
logger = get_logger(logger_name, logger_file_name)


def apply_random_oversampling(
    X,
    y,
    sampling_strategy="auto",
    random_state=42,
    shrinkage=None,
):
    """
    Apply RandomOverSampler.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.

    y : pd.Series or np.ndarray
        Target variable.

    sampling_strategy : str, float, dict, or callable, default='auto'
        Resampling strategy passed to ``RandomOverSampler``.

        - **float** — Desired ratio of minority samples to majority samples
          *after* resampling (``n_minority / n_majority``).
          Only valid for **binary** classification.

          Example::

              sampling_strategy=0.5
              # If majority class has 1000 samples, minority is oversampled
              # to 500 (ratio = 500/1000 = 0.5).

        - **str** — Class(es) to resample so that all targeted classes
          reach equal sample counts. Accepted values:

          - 'minority'``     resample only the minority class.
          - 'not minority'`` resample all classes except the minority.
          - 'not majority'`` resample all classes except the majority
                                 *(default when* 'auto'`` *is used)*.
          - 'all'``          resample every class.
          - 'auto'``         equivalent to 'not majority'.

          Example::

              sampling_strategy='minority'
              # Only the minority class is oversampled to match the majority.

        - **dict** — Keys are class labels; values are the desired number
          of samples for each class after resampling.

          Example::

              sampling_strategy={0: 900, 1: 900}
              # Class 0 and class 1 will each have 900 samples after
              # resampling (values must be >= current class count).

        - **callable** — A function that receives ``y`` and returns a
          ``dict`` in the same format as the dict case above.

          Example::

              def my_strategy(y):
                  counts = Counter(y)
                  majority_n = max(counts.values())
                  return {cls: majority_n for cls in counts}

              sampling_strategy=my_strategy
              # Every class is oversampled to match the majority class.

    random_state : int, default=42
        Random seed.

    shrinkage : float, dict, None, default=None
        Smoothed bootstrap parameter.

    Returns
    -------
    X_resampled
        Resampled features.

    y_resampled
        Resampled target.
    """

    # Get RandomOverSampler instance with specified parameters
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        shrinkage=shrinkage,
    )

    # Fit and resample the data
    X_resampled, y_resampled = ros.fit_resample(X, y)

    logger.info(f"Random oversampling applied with sampling_strategy={sampling_strategy}, random_state={random_state}, shrinkage={shrinkage}")

    # Log X and y shapes before and after resampling
    logger.info(f"Original X shape: {X.shape}, Original y shape: {y.shape}")
    logger.info(f"Resampled X shape: {X_resampled.shape}, Resampled y shape: {y_resampled.shape}")
    # Number of samples per class before and after resampling
    original_class_counts = pd.Series(y).value_counts()
    logger.info(f"Original class distribution:\n{original_class_counts}")
    resampled_class_counts = pd.Series(y_resampled).value_counts()
    logger.info(f"Resampled class distribution:\n{resampled_class_counts}")

    return X_resampled, y_resampled

def apply_random_undersampling(
    X,
    y,
    sampling_strategy="auto",
    random_state=42,
):
    """
    Apply RandomUnderSampler — keeps ALL minority-class samples and randomly
    draws the same number of samples from each majority class.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.

    y : pd.Series or np.ndarray
        Target variable.

    sampling_strategy : str, float, dict, or callable, default='auto'
        Resampling strategy passed to ``RandomUnderSampler``.
        ``'auto'`` (equivalent to ``'not minority'``) undersamples every
        majority class down to the size of the minority class.

        Example with ``'auto'`` on a binary target::

            # Original : {0: 15654, 1: 4446}
            # Resampled: {0: 4446,  1: 4446}  <- all 1s kept, 0s sampled

    random_state : int, default=42
        Random seed.

    Returns
    -------
    X_resampled
        Resampled features.

    y_resampled
        Resampled target.
    """

    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    X_resampled, y_resampled = rus.fit_resample(X, y)

    logger.info(f"Random undersampling applied with sampling_strategy={sampling_strategy}, random_state={random_state}")
    logger.info(f"Original X shape: {X.shape}, Original y shape: {y.shape}")
    logger.info(f"Resampled X shape: {X_resampled.shape}, Resampled y shape: {y_resampled.shape}")
    original_class_counts = pd.Series(y).value_counts()
    logger.info(f"Original class distribution:\n{original_class_counts}")
    resampled_class_counts = pd.Series(y_resampled).value_counts()
    logger.info(f"Resampled class distribution:\n{resampled_class_counts}")

    return X_resampled, y_resampled

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.33,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Purpose
    -------
    Split data into train and test sets while preserving pandas DataFrame/Series
    structure and column names.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (features).

    y : pd.Series
        Target series.

    test_size : float, optional
        Proportion of data for testing. Default ``0.33``.

    random_state : int, optional
        Random seed for reproducibility. Default ``42``.

    stratify : bool, optional
        If True, stratify split by target variable (useful for classification).
        Default ``False``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        - X_train: Training features (DataFrame)
        - X_test: Testing features (DataFrame)
        - y_train: Training target (Series)
        - y_test: Testing target (Series)
    """
    stratify_arg = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )
    logger.info(f"Data split into train and test sets with test_size={test_size}, random_state={random_state}, stratify={stratify}")
    logger.info(f"Number of training samples: {len(X_train)}")
    logger.info(f"Training set class distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Number of testing samples: {len(X_test)}")
    logger.info(f"Testing set class distribution:\n{y_test.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test

def interpret_p_value(p_value: float) -> dict:
    """
    Interpret statistical p-value.

    Parameters
    ----------
    p_value : float
        P-value from a statistical test.

    Returns
    -------
    dict
        Statistical significance interpretation.
    """

    # Raise Error for not numeric p-value
    if not isinstance(p_value, (int, float)):
        error_message = f"Invalid p-value type: {type(p_value)}. Expected numeric type."
        logger.error(error_message)
        raise TypeError(error_message)

    # Raise Error for invalid p-value
    if not (0 <= p_value <= 1):
        error_message = f"Invalid p-value: {p_value}. Must be between 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error for p-value is NaN
    if np.isnan(p_value):
        error_message = f"Invalid p-value: {p_value}. P-value cannot be NaN."
        logger.error(error_message)
        raise ValueError(error_message)

    if p_value < 0.001:
        significance = "Highly Significant"
    elif p_value < 0.01:
        significance = "Very Significant"
    elif p_value < 0.05:
        significance = "Significant"
    else:
        significance = "Not Significant"

    logger.info(f"P-value interpretation: {significance} for p-value {p_value}")

    return {
        "pValue": float(p_value),
        "significant": p_value < 0.05,
        "significanceLevel": significance
    }

def interpret_odds_ratio(odds_ratio: float) -> dict:
    """
    Interpret Odds Ratio.

    Odds Ratio > 1:
        Increases odds of positive class.

    Odds Ratio < 1:
        Decreases odds of positive class.

    Odds Ratio = 1:
        No effect.
    """
    # IMPORTANT NOTE:
    #
    # Odds Ratio describes the relative change in ODDS, not the absolute change
    # in probability.
    #
    # Example:
    #
    # Assume:
    #     P(Positive Class) = 90%
    #     P(Negative Class) = 10%
    #
    # Initial Odds:
    #     Odds = 0.90 / 0.10 = 9.0
    #
    # If Odds Ratio = 0.7836:
    #     New Odds = 9.0 × 0.7836 = 7.0524
    #
    # Convert new odds back to probability:
    #     P = Odds / (1 + Odds)
    #     P = 7.0524 / (1 + 7.0524)
    #     P = 87.58%
    #
    # Result:
    #     Odds decrease by 21.64%
    #     Probability decreases from 90.00% to 87.58%
    #
    # Therefore:
    #     Odds Ratio does NOT represent a direct percentage point change
    #     in probability.
    #
    # The relationship between odds and probability is non-linear.
    # For direct probability interpretation, use Marginal Effects instead.

    # Raise Error for not numeric odds ratio
    if not isinstance(odds_ratio, (int, float)):
        error_message = f"Invalid odds ratio type: {type(odds_ratio)}. Expected numeric type."
        logger.error(error_message)
        raise TypeError(error_message)
    
    # Raise Error for inf 
    if np.isinf(odds_ratio):
        error_message = f"Invalid odds ratio: {odds_ratio}. Odds ratio cannot be infinite."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error for odds ratio is NaN
    if np.isnan(odds_ratio):
        error_message = f"Invalid odds ratio: {odds_ratio}. Odds ratio cannot be NaN."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error for odds ratio is negative
    if odds_ratio < 0:
        error_message = f"Invalid odds ratio: {odds_ratio}. Odds ratio cannot be negative."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error for odds ratio is zero
    if odds_ratio == 0:
        error_message = f"Invalid odds ratio: {odds_ratio}. Odds ratio cannot be zero."
        logger.error(error_message)
        raise ValueError(error_message)

    # If odds ratio is approximately 1, interpret as no effect
    if np.isclose(odds_ratio, 1.0):
        logger.info("Odds Ratio is approximately 1. No meaningful change in odds.")
        return {
            "direction": "No Effect",
            "percentChangeInOdds": 0.0,
            "magnitude": "None",
            "interpretation": "No meaningful change in odds."
        }

    if odds_ratio > 1:
        percent_change = (odds_ratio - 1) * 100
        direction = "Increases"
        logger.info(f"Odds Ratio > 1: {direction} the odds of the positive class by {percent_change:.2f}%.")

    else:
        percent_change = (1 - odds_ratio) * 100
        direction = "Decreases"
        logger.info(f"Odds Ratio < 1: {direction} the odds of the positive class by {percent_change:.2f}%.")

    if percent_change >= 100:
        magnitude = "Very Large"
    elif percent_change >= 50:
        magnitude = "Large"
    elif percent_change >= 20:
        magnitude = "Moderate"
    elif percent_change >= 5:
        magnitude = "Small"
    else:
        magnitude = "Very Small"

    return {
        "direction": direction,
        "percentChangeInOdds": round(percent_change, 2),
        "magnitude": magnitude,
        "interpretation":
            f"A one-unit increase in the variable {direction.lower()} the odds of the positive class by {percent_change:.2f}%."
    }

def interpret_marginal_effect(marginal_effect: float) -> dict:
    """
    Interpret marginal effect.

    Marginal Effect represents the approximate change in predicted probability
    for a one-unit increase in the variable.

    Example:
        marginal_effect = -0.025
        means probability decreases by 2.5 percentage points.
    """

    # Raise Error for not numeric marginal effect
    if not isinstance(marginal_effect, (int, float)):
        error_message = f"Invalid marginal effect type: {type(marginal_effect)}. Expected numeric type."
        logger.error(error_message)
        raise TypeError(error_message)
    
    # Raise Error for marginal effect infinity
    if np.isinf(marginal_effect):
        error_message = f"Invalid marginal effect: {marginal_effect}. Marginal effect cannot be infinite."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error for marginal effect is NaN
    if np.isnan(marginal_effect):
        error_message = f"Invalid marginal effect: {marginal_effect}. Marginal effect cannot be NaN."
        logger.error(error_message)
        raise ValueError(error_message)

    probability_point_change = marginal_effect * 100
    abs_change = abs(probability_point_change)

    if np.isclose(marginal_effect, 0.0):
        direction = "No Effect"
        magnitude = "None"
        interpretation = "No meaningful change in predicted probability."
    else:
        direction = "Increases" if marginal_effect > 0 else "Decreases"

        if abs_change >= 10:
            magnitude = "Very Large"
        elif abs_change >= 5:
            magnitude = "Large"
        elif abs_change >= 2:
            magnitude = "Moderate"
        elif abs_change >= 1:
            magnitude = "Small"
        else:
            magnitude = "Very Small"

        interpretation = (
            f"A one-unit increase in the variable {direction.lower()} "
            f"the predicted probability of the positive class by "
            f"{abs_change:.2f} percentage points."
        )

        logger.info(f"Marginal Effect: {marginal_effect:.6f} - {magnitude} --> {interpretation}")

    return {"interpretation": interpretation, "magnitude": magnitude, "direction": direction}

def interpret_roc_auc(roc_auc: float) -> dict:
    """
    Interpret ROC-AUC score.

    ROC-AUC measures the model's ability to discriminate
    between positive and negative classes.

    Interpretation
    --------------
    ROC-AUC = 0.50
        Random prediction.

    ROC-AUC = 1.00
        Perfect discrimination.

    Higher values indicate better discriminatory power.

    Return
    ------
    dict
        Dictionary containing quality and interpretation of ROC-AUC score.
    """

    # Raise Error for not numeric ROC-AUC
    if not isinstance(roc_auc, (int, float)):
        error_message = (
            f"Invalid ROC-AUC type: {type(roc_auc)}. "
            "Expected numeric type."
        )
        logger.error(error_message)
        raise TypeError(error_message)

    # Raise Error for ROC-AUC is NaN
    if np.isnan(roc_auc):
        error_message = (
            f"Invalid ROC-AUC: {roc_auc}. "
            "ROC-AUC cannot be NaN."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error for ROC-AUC is infinite
    if np.isinf(roc_auc):
        error_message = (
            f"Invalid ROC-AUC: {roc_auc}. "
            "ROC-AUC cannot be infinite."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error for ROC-AUC outside valid range
    if not (0 <= roc_auc <= 1):
        error_message = (
            f"Invalid ROC-AUC: {roc_auc}. "
            "ROC-AUC must be between 0 and 1."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if roc_auc < 0.50:
        quality = "Worse Than Random"
        interpretation = (
            "Model performs worse than random guessing."
        )

    elif roc_auc < 0.60:
        quality = "Poor"
        interpretation = (
            "Model has poor discriminatory power."
        )

    elif roc_auc < 0.70:
        quality = "Fair"
        interpretation = (
            "Model has fair discriminatory power."
        )

    elif roc_auc < 0.80:
        quality = "Good"
        interpretation = (
            "Model has good discriminatory power."
        )

    elif roc_auc < 0.90:
        quality = "Very Good"
        interpretation = (
            "Model has very good discriminatory power."
        )

    else:
        quality = "Excellent"
        interpretation = (
            "Model has excellent discriminatory power."
        )

    logger.info(
        f"ROC-AUC Interpretation: "
        f"{roc_auc:.6f} -> "
        f"{quality} -> "
        f"{interpretation}"
    )

    return {
        "quality": quality,
        "interpretation": interpretation
    }

def interpret_gini(gini: float) -> dict:
    """
    Interpret Gini coefficient.

    Gini is derived from ROC-AUC:

        Gini = 2 * ROC-AUC - 1

    Higher values indicate better discriminatory power.

    Return
    ------
    dict
        Dictionary containing quality and interpretation of Gini coefficient.
    """

    if not isinstance(gini, (int, float)):
        error_message = (
            f"Invalid Gini type: {type(gini)}. "
            "Expected numeric type."
        )
        logger.error(error_message)
        raise TypeError(error_message)

    if np.isnan(gini):
        error_message = f"Invalid Gini: {gini}. Gini cannot be NaN."
        logger.error(error_message)
        raise ValueError(error_message)

    if np.isinf(gini):
        error_message = f"Invalid Gini: {gini}. Gini cannot be infinite."
        logger.error(error_message)
        raise ValueError(error_message)

    if not (-1 <= gini <= 1):
        error_message = (
            f"Invalid Gini: {gini}. "
            "Gini must be between -1 and 1."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if gini < 0:
        quality = "Worse Than Random"
        interpretation = "Model performs worse than random guessing."
    elif gini < 0.20:
        quality = "Poor"
        interpretation = "Model has poor discriminatory power."
    elif gini < 0.40:
        quality = "Fair"
        interpretation = "Model has fair discriminatory power."
    elif gini < 0.60:
        quality = "Good"
        interpretation = "Model has good discriminatory power."
    elif gini < 0.80:
        quality = "Very Good"
        interpretation = "Model has very good discriminatory power."
    else:
        quality = "Excellent"
        interpretation = "Model has excellent discriminatory power."

    logger.info(
        f"Gini Interpretation: "
        f"{gini:.6f} -> {quality} -> {interpretation}"
    )

    return {
        "quality": quality,
        "interpretation": interpretation
    }

def interpret_ks(ks: float) -> dict:
    """
    Interpret Kolmogorov-Smirnov (KS) statistic.

    Higher KS values indicate better separation between
    positive and negative classes.

    Return
    ------
    dict
        Dictionary containing quality and interpretation of KS statistic.
    """

    # Raise Error for not numeric KS
    if not isinstance(ks, (int, float)):
        error_message = (
            f"Invalid KS type: {type(ks)}. "
            "Expected numeric type."
        )
        logger.error(error_message)
        raise TypeError(error_message)

    # Raise Error for KS is NaN
    if np.isnan(ks):
        error_message = f"Invalid KS: {ks}. KS cannot be NaN."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error for KS is infinite
    if np.isinf(ks):
        error_message = f"Invalid KS: {ks}. KS cannot be infinite."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error for invalid KS range
    if not (0 <= ks <= 1):
        error_message = (
            f"Invalid KS: {ks}. "
            "KS must be between 0 and 1."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    ks_percentage = ks * 100

    if ks_percentage < 20:
        quality = "Poor"
        interpretation = (
            "Model has poor discriminatory power."
        )

    elif ks_percentage < 30:
        quality = "Fair"
        interpretation = (
            "Model has fair discriminatory power."
        )

    elif ks_percentage < 40:
        quality = "Good"
        interpretation = (
            "Model has good discriminatory power."
        )

    elif ks_percentage < 50:
        quality = "Very Good"
        interpretation = (
            "Model has very good discriminatory power."
        )

    else:
        quality = "Excellent"
        interpretation = (
            "Model has excellent discriminatory power."
        )

    logger.info(
        f"KS Interpretation: "
        f"{ks:.6f} "
        f"({ks_percentage:.2f}%) -> "
        f"{quality} -> "
        f"{interpretation}"
    )

    return {
        "quality": quality,
        "interpretation": interpretation
    }

def from_np_to_list_converter(input_np: any) -> list:
    """
    Convert input array to a standard format for processing.

    Parameters
    ----------
    input_np : any
        Input array to be converted.

    Returns
    -------
    list
        Converted list.
    """
    if isinstance(input_np, np.ndarray):
        converted = input_np.tolist()
        logger.info(
            f"converted from numpy array to list | np.shape={input_np.shape}"
        )
        return converted

    if isinstance(input_np, list):
        logger.info("Input is already a list. No conversion needed.")
        return input_np

    error_message = f"Invalid input type: {type(input_np)}. Expected numpy array or list."
    logger.error(error_message)
    raise TypeError(error_message)

def from_pdseries_to_list_converter(input_series: any) -> list:
    """
    Convert input pandas Series to a standard format for processing.

    Parameters
    ----------
    input_series : any
        Input pandas Series to be converted.

    Returns
    -------
    list
        Converted list.
    """
    if isinstance(input_series, pd.Series):
        converted = input_series.tolist()
        logger.info(
            f"converted from pandas Series to list | pd.shape={input_series.shape}"
        )
        return converted

    if isinstance(input_series, list):
        logger.info("Input is already a list. No conversion needed.")
        return input_series

    error_message = f"Invalid input type: {type(input_series)}. Expected pandas Series or list."
    logger.error(error_message)
    raise TypeError(error_message)

def from_list_to_np_converter(input_list: any) -> np.ndarray:
    """
    Convert input list to a standard format for processing.

    Parameters
    ----------
    input_list : list
        Input list to be converted.
    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    if isinstance(input_list, list):
        converted = np.asarray(input_list)
        logger.info(
            f"converted from list to numpy array | np.shape={converted.shape}"
        )
        return converted

    if isinstance(input_list, np.ndarray):
        logger.info(
            f"Input is already a numpy array. No conversion needed. | np.shape={input_list.shape}"
        )
        return input_list

    error_message = f"Invalid input type: {type(input_list)}. Expected list or numpy array."
    logger.error(error_message)
    raise TypeError(error_message)

def from_list_to_pdseries_converter(input_list: any) -> pd.Series:
    """
    Convert input list to a standard format for processing.

    Parameters
    ----------
    input_list : any
        Input list to be converted.
    
    Returns
    -------
    pd.Series
        Converted pandas Series.
    """
    if isinstance(input_list, list):
        converted = pd.Series(input_list)
        logger.info(
            f"converted from list to pandas Series for accuracy calculation. | pd.shape={converted.shape}"
        )
        return converted

    if isinstance(input_list, pd.Series):
        logger.info(
            f"Input is already a pandas Series. No conversion needed. | pd.shape={input_list.shape}"
        )
        return input_list

    error_message = f"Invalid input type: {type(input_list)}. Expected list or pandas Series."
    logger.error(error_message)
    raise TypeError(error_message)

def calculate_confusion_matrix(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray | pd.Series
) -> dict:
    """
    Purpose
    ----------
    Calculate binary classification confusion matrix.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values.
        Expected values: 0 and 1.

    y_pred : list
        Predicted class labels.
        Expected values: 0 and 1.

    Returns
    -------
    dict
        Dictionary containing confusion matrix metrics.

        Example:
        {
            "trueNegative": 850,
            "falsePositive": 50,
            "falseNegative": 30,
            "truePositive": 70,
            "totalObservations": 1000
        }
    """
    # Raise Error if y_true or y_pred is empty
    if len(y_true) == 0 or len(y_pred) == 0:
        error_message = "y_true and y_pred must not be empty."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error if y_true and y_pred have different lengths
    if len(y_true) != len(y_pred):
        error_message = f"Length mismatch: y_true has length {len(y_true)}, but y_pred has length {len(y_pred)}."
        logger.error(error_message)
        raise ValueError(error_message)

    # Convert to numpy arrays for validation
    y_true_arr = from_list_to_np_converter(y_true)
    y_pred_arr = from_list_to_np_converter(y_pred)
    logger.info("Input is converted to numpy array for validation.")
    
    # Raise Error if y_true or y_pred contains NaN values
    if np.isnan(y_true_arr).any() or np.isnan(y_pred_arr).any():
        error_message = "y_true and y_pred must not contain NaN values."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error if y_true or y_pred contains infinite values
    if np.isinf(y_true_arr).any() or np.isinf(y_pred_arr).any():
        error_message = "y_true and y_pred must not contain infinite values."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error if y_true contains values other than 0 and 1
    if not np.isin(y_true_arr, [0, 1]).all():
        error_message = f"y_true contains invalid values: {np.unique(y_true_arr)}. Expected only 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if y_pred contains values other than 0 and 1
    if not np.isin(y_pred_arr, [0, 1]).all():
        error_message = f"y_pred contains invalid values: {np.unique(y_pred_arr)}. Expected only 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)
    
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    logger.info("Confusion matrix calculated successfully.")

    if cm.shape != (2, 2):
        error_message = "Only binary classification is supported."
        logger.error(error_message)
        raise ValueError(error_message)

    tn, fp, fn, tp = cm.ravel()

    results = {
        "trueNegative": int(tn),
        "falsePositive": int(fp),
        "falseNegative": int(fn),
        "truePositive": int(tp),
        "totalObservations": int(tn + fp + fn + tp)
    }

    logger.info("Confusion Matrix:")
    logger.info(f"\tTrue Negative  (TN): {tn}")
    logger.info(f"\tFalse Positive (FP): {fp}")
    logger.info(f"\tFalse Negative (FN): {fn}")
    logger.info(f"\tTrue Positive  (TP): {tp}")

    return results

def calculate_accuracy(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray | pd.Series,
    confusion_matrix_dict: dict | None = None
) -> dict:
    """
    Calculate classification accuracy.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values.

    y_pred : list | np.ndarray | pd.Series
        Predicted class labels.

    Returns
    -------
    dict
        Accuracy statistics.

        Example:
        {
            "accuracy": 0.9234,
            "accuracyPercentage": 92.34
        }
    """

    if confusion_matrix_dict is None:
        y_true = from_list_to_np_converter(y_true)
        y_pred = from_list_to_np_converter(y_pred)

        confusion_matrix_dict = calculate_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred
        )

    tn = confusion_matrix_dict["trueNegative"]
    fp = confusion_matrix_dict["falsePositive"]
    fn = confusion_matrix_dict["falseNegative"]
    tp = confusion_matrix_dict["truePositive"]

    total_observations = confusion_matrix_dict["totalObservations"]

    accuracy = (tp + tn) / total_observations

    logger.info(
        f"Accuracy Calculation --> "
        f"(TP + TN) / Total = "
        f"({tp} + {tn}) / {total_observations} = "
        f"{accuracy:.6f}"
    )

    return {
        "accuracy": float(accuracy),
        "accuracyPercentage": round(accuracy * 100, 2)
    }

def calculate_precision(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray | pd.Series,
    confusion_matrix_dict: dict | None = None
) -> dict:
    """
    Calculate classification precision.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values.

    y_pred : list | np.ndarray | pd.Series
        Predicted class labels.

    Returns
    -------
    dict
        Precision statistics.

        Example:
        {
            "precision": 0.875,
            "precisionPercentage": 87.50
        }
    """

    if confusion_matrix_dict is None:
        y_true = from_list_to_np_converter(y_true)
        y_pred = from_list_to_np_converter(y_pred)

        confusion_matrix_dict = calculate_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred
        )

    tp = confusion_matrix_dict["truePositive"]
    fp = confusion_matrix_dict["falsePositive"]

    # Prevent division by zero
    if (tp + fp) == 0:
        error_message = (
            "Precision is undefined because "
            "(TP + FP) equals zero."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    precision = tp / (tp + fp)

    logger.info(
        f"Precision Calculation --> "
        f"TP / (TP + FP) = "
        f"{tp} / ({tp} + {fp}) = "
        f"{precision:.6f}"
    )

    return {
        "precision": float(precision),
        "precisionPercentage": round(precision * 100, 2)
    }

def calculate_recall(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray | pd.Series,
    confusion_matrix_dict: dict | None = None
) -> dict:
    """
    Calculate classification recall.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values.

    y_pred : list | np.ndarray | pd.Series
        Predicted class labels.

    Returns
    -------
    dict
        Recall statistics.

        Example:
        {
            "recall": 0.8235,
            "recallPercentage": 82.35
        }
    """

    if confusion_matrix_dict is None:
        y_true = from_list_to_np_converter(y_true)
        y_pred = from_list_to_np_converter(y_pred)

        confusion_matrix_dict = calculate_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred
        )

    tp = confusion_matrix_dict["truePositive"]
    fn = confusion_matrix_dict["falseNegative"]

    # Prevent division by zero
    if (tp + fn) == 0:
        error_message = (
            "Recall is undefined because "
            "(TP + FN) equals zero."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    recall = tp / (tp + fn)

    logger.info(
        f"Recall Calculation --> "
        f"TP / (TP + FN) = "
        f"{tp} / ({tp} + {fn}) = "
        f"{recall:.6f}"
    )

    return {
        "recall": float(recall),
        "recallPercentage": round(recall * 100, 2)
    }

def calculate_f1_score(
    y_true: list | np.ndarray | pd.Series,
    y_pred: list | np.ndarray | pd.Series,
    confusion_matrix_dict: dict | None = None
) -> dict:
    """
    Calculate F1 Score.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values.

    y_pred : list | np.ndarray | pd.Series
        Predicted class labels.

    Returns
    -------
    dict
        F1 score statistics.

        Example:
        {
            "f1Score": 0.8421,
            "f1ScorePercentage": 84.21
        }
    """

    precision_dict = calculate_precision(
        y_true=y_true,
        y_pred=y_pred,
        confusion_matrix_dict=confusion_matrix_dict
    )

    recall_dict = calculate_recall(
        y_true=y_true,
        y_pred=y_pred,
        confusion_matrix_dict=confusion_matrix_dict
    )

    precision = precision_dict["precision"]
    recall = recall_dict["recall"]

    if (precision + recall) == 0:
        error_message = (
            "F1 Score is undefined because "
            "(Precision + Recall) equals zero."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    f1_score = (
        2 * precision * recall
    ) / (
        precision + recall
    )

    logger.info(
        f"F1 Score Calculation --> "
        f"2 * Precision * Recall / (Precision + Recall) = "
        f"2 * {precision:.6f} * {recall:.6f} "
        f"/ ({precision:.6f} + {recall:.6f}) = "
        f"{f1_score:.6f}"
    )

    return {
        "f1Score": float(f1_score),
        "f1ScorePercentage": round(f1_score * 100, 2)
    }

def calculate_roc_auc(
    y_true: list | np.ndarray | pd.Series,
    y_prob: list | np.ndarray | pd.Series
) -> dict:
    """
    Calculate ROC-AUC score.

    y_prob must be predicted probabilities, not class labels.
    """

    y_true = from_list_to_np_converter(y_true)
    y_prob = from_list_to_np_converter(y_prob)

    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must not be empty.")

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}"
        )

    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("y_true and y_prob must not contain NaN values.")

    if np.isinf(y_true).any() or np.isinf(y_prob).any():
        raise ValueError("y_true and y_prob must not contain infinite values.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            f"y_true contains invalid values: {np.unique(y_true)}. Expected only 0 and 1."
        )

    if not ((0 <= y_prob) & (y_prob <= 1)).all():
        raise ValueError("y_prob must contain probability values between 0 and 1.")

    roc_auc = roc_auc_score(y_true, y_prob)

    logger.info(
        f"ROC-AUC Calculation --> ROC-AUC = {roc_auc:.6f}"
    )

    return {
        "rocAuc": float(roc_auc),
        "rocAucPercentage": round(roc_auc * 100, 2)
    }

def calculate_gini(
    y_true: list | np.ndarray | pd.Series,
    y_prob: list | np.ndarray | pd.Series,
    roc_auc: float | None = None
) -> dict:
    """
    Calculate Gini coefficient from ROC-AUC.

    Formula
    -------
    Gini = 2 * ROC-AUC - 1
    """

    if roc_auc is None:
        roc_auc_dict = calculate_roc_auc(
            y_true=y_true,
            y_prob=y_prob
        )
        roc_auc = roc_auc_dict["rocAuc"]

    gini = 2 * roc_auc - 1

    logger.info(
        f"Gini Calculation --> "
        f"2 * ROC-AUC - 1 = "
        f"2 * {roc_auc:.6f} - 1 = {gini:.6f}"
    )

    return {
        "gini": float(gini),
        "giniPercentage": round(gini * 100, 2)
    }

def calculate_ks(
    y_true: list | np.ndarray | pd.Series,
    y_prob: list | np.ndarray | pd.Series
) -> dict:
    """
    Purpose
    ----------
    Calculate Kolmogorov-Smirnov (KS) statistic for binary classification.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values (0 or 1).
    y_prob : list | np.ndarray | pd.Series
        Predicted probabilities for the positive class (1).
    
    Returns
    -------
    dict        
        Dictionary containing KS statistic and related information.
    """
    # =============================================================================
    # KOLMOGOROV-SMIRNOV (KS) STATISTIC
    # =============================================================================
    #
    # Purpose
    # -------
    # KS measures the maximum separation between the cumulative distributions
    # of the positive class (Bad) and the negative class (Good).
    #
    # In credit risk modeling, KS is one of the most commonly used metrics for
    # evaluating how well a model distinguishes risky customers from non-risky
    # customers.
    #
    #
    # Concept
    # -------
    # For every possible probability threshold:
    #
    #     1. Calculate cumulative Bad rate
    #     2. Calculate cumulative Good rate
    #     3. Compute the difference
    #
    #         KS = Cumulative Bad Rate - Cumulative Good Rate
    #
    # The maximum difference observed across all thresholds is reported as the
    # model's KS statistic.
    #
    #
    # Mathematical Definition
    # -----------------------
    #
    #     KS = max(
    #         Cumulative Bad Distribution
    #         -
    #         Cumulative Good Distribution
    #     )
    #
    #
    # Example
    # -------
    #
    # Threshold    Cum Bad %    Cum Good %    KS
    # ------------------------------------------------
    # 0.90            15            2         13
    # 0.80            35            8         27
    # 0.70            55           20         35
    # 0.60            70           35         35
    # 0.50            85           60         25
    #
    # Maximum KS:
    #
    #     KS = 35
    #
    #
    # Interpretation
    # --------------
    # KS identifies the threshold where the model best separates the positive
    # and negative classes.
    #
    # Example:
    #
    # At threshold = 0.70:
    #
    #     - 55% of all Bad customers are captured
    #     - Only 20% of all Good customers are captured
    #
    # Therefore:
    #
    #     KS = 55% - 20% = 35%
    #
    # Larger KS values indicate stronger discriminatory power.
    #
    #
    # Relationship to ROC Curve
    # -------------------------
    # KS is closely related to the ROC Curve.
    #
    # ROC uses:
    #
    #     TPR (True Positive Rate)
    #     FPR (False Positive Rate)
    #
    # KS can also be expressed as:
    #
    #     KS = max(TPR - FPR)
    #
    # Therefore, KS and ROC-AUC both measure model discrimination,
    # but from different perspectives.
    #
    #
    # Practical Interpretation
    # ------------------------
    #
    # ROC-AUC:
    #     Measures overall discriminatory power.
    #
    # KS:
    #     Measures the maximum separation achieved at a specific threshold.
    #
    #
    # Common Industry Guidelines
    # --------------------------
    #
    # KS < 20
    #     Poor
    #
    # 20 <= KS < 30
    #     Fair
    #
    # 30 <= KS < 40
    #     Good
    #
    # 40 <= KS < 50
    #     Very Good
    #
    # KS >= 50
    #     Excellent
    #
    #
    # Recommended Output
    # ------------------
    # In addition to the KS value itself, it is often useful to return:
    #
    #     - KS Statistic
    #     - Optimal Threshold
    #     - Cumulative Bad Rate at KS
    #     - Cumulative Good Rate at KS
    #
    # Example:
    #
    # {
    #     "ks": 41.27,
    #     "optimalThreshold": 0.382,
    #     "cumulativeBadRate": 0.71,
    #     "cumulativeGoodRate": 0.297
    # }
    #
    # This threshold often serves as a candidate operational cutoff in
    # credit scoring and risk decision systems.
    # =============================================================================

    y_true = from_list_to_np_converter(y_true)
    y_prob = from_list_to_np_converter(y_prob)

    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError("y_true and y_prob must not be empty.")

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true has length {len(y_true)}, "
            f"but y_prob has length {len(y_prob)}."
        )

    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError("y_true and y_prob must not contain NaN values.")

    if np.isinf(y_true).any() or np.isinf(y_prob).any():
        raise ValueError("y_true and y_prob must not contain infinite values.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            f"y_true contains invalid values: {np.unique(y_true)}. "
            "Expected only 0 and 1."
        )

    if not ((0 <= y_prob) & (y_prob <= 1)).all():
        raise ValueError("y_prob must contain probability values between 0 and 1.")

    df_ks = pd.DataFrame({
        "yTrue": y_true,
        "yProb": y_prob
    })

    # Sort descending: highest predicted risk first
    df_ks = df_ks.sort_values(
        by="yProb",
        ascending=False
    ).reset_index(drop=True)

    total_bad = int((df_ks["yTrue"] == 1).sum())
    total_good = int((df_ks["yTrue"] == 0).sum())

    if total_bad == 0 or total_good == 0:
        raise ValueError(
            "KS cannot be calculated because both classes must exist in y_true."
        )

    df_ks["bad"] = (df_ks["yTrue"] == 1).astype(int)
    df_ks["good"] = (df_ks["yTrue"] == 0).astype(int)

    df_ks["cumulativeBad"] = df_ks["bad"].cumsum()
    df_ks["cumulativeGood"] = df_ks["good"].cumsum()

    df_ks["cumulativeBadRate"] = df_ks["cumulativeBad"] / total_bad
    df_ks["cumulativeGoodRate"] = df_ks["cumulativeGood"] / total_good

    df_ks["ks"] = (
        df_ks["cumulativeBadRate"] -
        df_ks["cumulativeGoodRate"]
    )

    ks_idx = df_ks["ks"].idxmax()
    ks_row = df_ks.loc[ks_idx]

    ks_value = float(ks_row["ks"])
    optimal_threshold = float(ks_row["yProb"])

    logger.info(f"KS Calculation for Best Threshold -->")
    logger.info(f"\tKS={ks_value:.6f}")
    logger.info(f"\tThreshold={optimal_threshold:.6f}")
    logger.info(f"\tCumBad={ks_row['cumulativeBadRate']:.6f}")
    logger.info(f"\tCumGood={ks_row['cumulativeGoodRate']:.6f}")
    logger.info(f"\tDifference={ks_row['ks']:.6f}")

    return {
        "ks": ks_value,
        "ksPercentage": round(ks_value * 100, 2),
        "optimalThreshold": optimal_threshold,
        "cumulativeBadRate": float(ks_row["cumulativeBadRate"]),
        "cumulativeGoodRate": float(ks_row["cumulativeGoodRate"]),
        "cumulativeBadPercentage": round(float(ks_row["cumulativeBadRate"]) * 100, 2),
        "cumulativeGoodPercentage": round(float(ks_row["cumulativeGoodRate"]) * 100, 2),
        "thresholdIndex": int(ks_idx),
        "totalBad": total_bad,
        "totalGood": total_good,
        "totalObservations": int(len(df_ks))
    }

def calculate_decile_analysis(
    y_true: list | np.ndarray | pd.Series,
    y_prob: list | np.ndarray | pd.Series,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Purpose
    ----------
    Calculate decile analysis table for binary classification risk model.
    Higher probabilities are assumed to represent higher risk / positive class.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values (0 or 1).
    y_prob : list | np.ndarray | pd.Series
        Predicted probabilities for the positive class (1).
    n_bins : int, optional
        Number of bins (deciles) to create. Default is 10.
    
    Returns
    -------
    dict
        Dictionary containing decile analysis results, including:
            - numberOfBins
            - totalObservations
            - totalGood
            - totalBad
            - deciles: list of dictionaries with decile statistics.
    """
    # Convert inputs to numpy arrays for validation
    y_true = from_list_to_np_converter(y_true)
    y_prob = from_list_to_np_converter(y_prob)
    
    # Raise Error if y_true or y_prob is empty
    if len(y_true) == 0 or len(y_prob) == 0:
        error_message = "y_true and y_prob must not be empty."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if y_true and y_prob have different lengths
    if len(y_true) != len(y_prob):
        error_message = (
            f"Length mismatch: y_true has length {len(y_true)}, "
            f"but y_prob has length {len(y_prob)}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if y_true or y_prob contains NaN values
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        error_message = "y_true and y_prob must not contain NaN values."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if y_true or y_prob contains infinite values
    if np.isinf(y_true).any() or np.isinf(y_prob).any():
        error_message = "y_true and y_prob must not contain infinite values."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if y_true contains values other than 0 and 1
    if not np.isin(y_true, [0, 1]).all():
        error_message = f"y_true contains invalid values: {np.unique(y_true)}. Expected only 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if y_prob contains values other than 0 and 1
    if not ((0 <= y_prob) & (y_prob <= 1)).all():
        error_message = "y_prob must contain probability values between 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if n_bins is less than 2
    if n_bins < 2:
        error_message = "n_bins must be at least 2."
        logger.error(error_message)
        raise ValueError(error_message)

    # Generate decile table as DataFrame for easier calculations
    df = pd.DataFrame({
        "yTrue": y_true,
        "yProb": y_prob
    })

    # Sort descending: highest predicted risk first
    df = df.sort_values(
        by="yProb",
        ascending=False # From highest to lowest predicted probability
    ).reset_index(drop=True)

    # Assign decile labels based on the sorted index
    df["decile"] = pd.qcut(
        df.index + 1,
        q=n_bins,
        labels=False
    ) + 1

    total_observations = len(df)
    total_bad = int((df["yTrue"] == 1).sum())
    total_good = int((df["yTrue"] == 0).sum())
    logger.info(f"Number of observations: {total_observations}")
    logger.info(f"Number of good observations: {total_good}")
    logger.info(f"Number of bad observations: {total_bad}")

    # Raise Error if total_bad or total_good is zero, as decile analysis cannot be performed
    if total_bad == 0 or total_good == 0:
        error_message = "Decile analysis cannot be calculated because both classes must exist in y_true."
        logger.error(error_message)
        raise ValueError(error_message)

    # Group by decile and calculate statistics
    decile_table = (
        df
        .groupby("decile", observed=True)
        .agg(
            minProbability=("yProb", "min"),
            maxProbability=("yProb", "max"),
            observations=("yTrue", "count"),
            bad=("yTrue", "sum")
        )
        .reset_index()
    )

    # Number of good observations in each decile
    decile_table["good"] = decile_table["observations"] - decile_table["bad"]

    # Population rate: share of total observations in each decile
    decile_table["populationRate"] = (
        decile_table["observations"] / total_observations
        )

    # Bad rate: share of bad observations in each decile
    decile_table["badRate"] = (
        decile_table["bad"] / decile_table["observations"]
        )

    # Good rate: share of good observations in each decile
    decile_table["goodRate"] = (
        decile_table["good"] / decile_table["observations"]
        )

    # Bad Distribution: share of total bad observations in each decile
    decile_table["badDistribution"] = (
        decile_table["bad"] / total_bad
        )

    # Good Distribution: share of total good observations in each decile
    decile_table["goodDistribution"] = (
        decile_table["good"] / total_good
        )

    # Cumulative Bad and Good counts
    decile_table["cumulativeBad"] = decile_table["bad"].cumsum()
    decile_table["cumulativeGood"] = decile_table["good"].cumsum()

    # Cumulative Bad and Good rates
    decile_table["cumulativeBadRate"] = (
        decile_table["cumulativeBad"] / total_bad
        )

    decile_table["cumulativeGoodRate"] = (
        decile_table["cumulativeGood"] / total_good
        )

    # Cumulative Population and Population Rate
    decile_table["cumulativePopulation"] = decile_table["observations"].cumsum()

    decile_table["cumulativePopulationRate"] = (
        decile_table["cumulativePopulation"] / total_observations
        )

    # Cumulative Lift: ratio of cumulative bad rate to cumulative population rate
    decile_table["cumulativeLift"] = (
        decile_table["cumulativeBadRate"] /
        decile_table["cumulativePopulationRate"]
        )

    # KS statistic at decile level: difference between cumulative bad rate and cumulative good rate
    decile_table["ks"] = (
        decile_table["cumulativeBadRate"] -
        decile_table["cumulativeGoodRate"]
    )

    # Basis for Lift: overall bad rate across the entire dataset
    overall_bad_rate = total_bad / total_observations

    # Lift at decile level: ratio of bad rate in each decile to overall bad rate
    decile_table["decileLift"] = (
        decile_table["badRate"] / overall_bad_rate
    )

    # For KS Curve, we need to calculate cumulative distributions across all observations, not just deciles.
    curve_df = df.sort_values(
        by="yProb",
        ascending=False
    ).reset_index(drop=True)

    curve_df["cumulativeBad"] = (curve_df["yTrue"] == 1).cumsum()
    curve_df["cumulativeGood"] = (curve_df["yTrue"] == 0).cumsum()

    curve_df["cumulativeBadRate"] = curve_df["cumulativeBad"] / total_bad
    curve_df["cumulativeGoodRate"] = curve_df["cumulativeGood"] / total_good

    curve_df["cumulativePopulationRate"] = (
        (curve_df.index + 1) / total_observations
    )

    curve_df["ks"] = (
        curve_df["cumulativeBadRate"] - curve_df["cumulativeGoodRate"]
    )

    ks_curve = curve_df[
        [
            "yProb",
            "cumulativeBadRate",
            "cumulativeGoodRate",
            "cumulativePopulationRate",
            "ks"
        ]
    ].rename(columns={"yProb": "threshold"})

    logger.info(
        f"Decile analysis completed with n_bins={n_bins}. "
        f"Total={total_observations}, Bad={total_bad}, Good={total_good}"
    )

    # Log table 
    logger.info("Decile table:\n%s", decile_table)

    return {
        "numberOfBins": n_bins,
        "totalObservations": total_observations,
        "totalGood": total_good,
        "totalBad": total_bad,
        "deciles": decile_table.to_dict(orient="records"),
        "ksCurve": ks_curve.to_dict(orient="records")
        }

def plot_ks_v0(
    decile_analysis: dict,
    figsize: tuple = (10, 6),
):
    """
    Plot KS Curve using decile analysis output.

    Parameters
    ----------
    decile_analysis : dict
        Output of calculate_decile_analysis().

    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        KS plot figure.
    """

    decile_df = pd.DataFrame(
        decile_analysis["ksCurve"]
    )

    ks_value = decile_df["ks"].max()
    ks_threshold = decile_df.loc[
        decile_df["ks"].idxmax(),
        "threshold"
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        decile_df["threshold"],
        decile_df["cumulativeGoodRate"],
        label="Class 0 (Good)"
    )

    ax.plot(
        decile_df["threshold"],
        decile_df["cumulativeBadRate"],
        label="Class 1 (Bad)"
    )

    ax.axvline(
        ks_threshold,
        linestyle=":",
        color="black",
        label=f"KS Statistic: {ks_value:.3f} at {ks_threshold:.3f}"
    )

    ax.annotate(
        f"Threshold = {ks_threshold:.4f}",
        xy=(ks_threshold, 0.5),
        xytext=(5, 0),
        textcoords="offset points",
        rotation=90,
        va="center",
        ha="left"
    )

    ax.set_title("KS Statistic Plot")

    ax.set_xlabel("Threshold")

    ax.set_ylabel("Cumulative Rate (Good and Bad)")

    ax.legend()

    ax.grid(False)

    logger.info(
        f"KS Plot created. "
        f"KS={ks_value:.6f}, "
        f"Threshold={ks_threshold:.6f}"
    )

    return fig

def plot_lift(
    decile_analysis: dict,
    lift_type: str = "cumulative",
    figsize: tuple = (10, 6),
):
    """
    Plot Lift Chart using decile analysis output.

    Parameters
    ----------
    decile_analysis : dict
        Output of calculate_decile_analysis().

    lift_type : str, optional
        Type of lift chart to plot.
        - "cumulative": plots cumulative lift by cumulative population rate.
        - "decile": plots decile lift by decile.
        Default is "cumulative".

    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Lift plot figure.
    """

    decile_df = pd.DataFrame(decile_analysis["deciles"])

    if lift_type not in ["cumulative", "decile"]:
        error_message = (
            f"Invalid lift_type: {lift_type}. "
            "Expected 'cumulative' or 'decile'."
        )
        raise ValueError(error_message)


    fig, ax = plt.subplots(figsize=figsize)

    if lift_type == "cumulative":
        max_lift = decile_df["cumulativeLift"].max()
        max_lift_pop = decile_df.loc[
            decile_df["cumulativeLift"].idxmax(),
            "cumulativePopulationRate"
        ]

        ax.plot(
            decile_df["cumulativePopulationRate"],
            decile_df["cumulativeLift"],
            marker="o",
            label="Cumulative Lift"
        )

        ax.axhline(
            y=1,
            linestyle="--",
            color="black",
            label="Baseline Lift = 1"
        )

        ax.set_title(f"Cumulative Lift Chart (Max Lift = {max_lift:.2f})")
        ax.set_xlabel("Cumulative Population Rate")
        ax.set_ylabel("Cumulative Lift")

        ax.set_xticks(decile_df["cumulativePopulationRate"])
        ax.tick_params(axis="x", labelrotation=90)

        ax.xaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0)
        )

    else:
        ax.bar(
            decile_df["decile"],
            decile_df["decileLift"],
            label="Decile Lift"
        )

        ax.axhline(
            y=1,
            linestyle="--",
            color="black",
            label="Baseline Lift = 1"
        )

        ax.set_title("Decile Lift Chart")
        ax.set_xlabel("Decile")
        ax.set_ylabel("Lift")

        ax.set_xticks(decile_df["decile"])

    ax.legend()
    ax.grid(False)

    logger.info(
        f"Lift Plot created. "
        f"lift_type={lift_type}"
    )

    return fig

def plot_lorenz(
    decile_analysis: dict,
    figsize: tuple = (10, 6),
):
    """
    Plot Lorenz (Cumulative Gains) Curve using decile analysis output.

    Parameters
    ----------
    decile_analysis : dict
        Output of calculate_decile_analysis().

    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Lorenz curve figure.
    """

    decile_df = pd.DataFrame(
        decile_analysis["deciles"]
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        decile_df["cumulativePopulationRate"],
        decile_df["cumulativeBadRate"],
        marker="o",
        linewidth=2,
        label="Model"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Random Model"
    )

    ax.set_title("Lorenz Curve")

    ax.set_xlabel("Cumulative Population")
    ax.set_ylabel("Cumulative Bads Captured")

    ax.legend()

    ax.grid(False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(
        decile_df["cumulativePopulationRate"]
    )

    ax.set_xticklabels(
        [f"{x:.0%}" for x in decile_df["cumulativePopulationRate"]]
    )

    ax.set_yticks(
        decile_df["cumulativeBadRate"]
    )

    ax.set_yticklabels(
        [f"{y:.0%}" for y in decile_df["cumulativeBadRate"]]
    )

    logger.info("Lorenz Curve created.")

    return fig

def plot_roc_auc(
    y_true: list | np.ndarray | pd.Series,
    y_prob: list | np.ndarray | pd.Series,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot ROC AUC Curve.

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values (0 or 1).
    y_prob : list | np.ndarray | pd.Series
        Predicted probabilities for the positive class (1).
    figsize : tuple, optional
        Figure size. Default is (10, 6).

    Returns
    -------
    matplotlib.figure.Figure
        ROC AUC plot figure.
    """
    y_true = from_list_to_np_converter(y_true)
    y_prob = from_list_to_np_converter(y_prob)

    if len(y_true) == 0 or len(y_prob) == 0:
        error_message = "y_true and y_prob must not be empty."
        logger.error(error_message)
        raise ValueError(error_message)

    if len(y_true) != len(y_prob):
        error_message = (
            f"Length mismatch: y_true has length {len(y_true)}, "
            f"but y_prob has length {len(y_prob)}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        error_message = "y_true and y_prob must not contain NaN values."
        logger.error(error_message)
        raise ValueError(error_message)

    if np.isinf(y_true).any() or np.isinf(y_prob).any():
        error_message = "y_true and y_prob must not contain infinite values."
        logger.error(error_message)
        raise ValueError(error_message)

    if not np.isin(y_true, [0, 1]).all():
        error_message = (
            f"y_true contains invalid values: {np.unique(y_true)}. "
            "Expected only 0 and 1."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if not ((0 <= y_prob) & (y_prob <= 1)).all():
        error_message = "y_prob must contain probability values between 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC = {auc_score:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Random Classifier"
    )

    ax.fill_between(
        fpr,
        tpr,
        fpr,
        alpha=0.2,
        color="orange"
    )

    ax.fill_between(
        [0, 1],
        [0, 1],
        0,
        alpha=0.2,
        color="orange"
    )

    ax.set_title("ROC AUC Curve")
    ax.set_xlabel("False Positive Rate (FP / (FP + TN))") # means that the model is incorrectly classifying negative instances as positive
    ax.set_ylabel("True Positive Rate (TP / (TP + FN))") # means that the model is correctly classifying positive instances

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(False)

    logger.info(
        f"ROC AUC Plot created. AUC={auc_score:.6f}"
    )

    return fig

def calculate_threshold_analysis(
    y_true: list | np.ndarray | pd.Series,
    y_prob: list | np.ndarray | pd.Series,
    thresholds: list[float],
) -> dict:
    """
    Calculate classification metrics for each given threshold.

    For each threshold, predicted class is determined as:
        y_pred = 1 if y_prob >= threshold else 0

    Parameters
    ----------
    y_true : list | np.ndarray | pd.Series
        Actual target values (0 or 1).
    y_prob : list | np.ndarray | pd.Series
        Predicted probabilities for the positive class (1).
    thresholds : list[float]
        List of probability thresholds to evaluate.

    Returns
    -------
    dict
        Dictionary with key "rows" containing a list of per-threshold
        metric dictionaries. Convert to DataFrame with:
            pd.DataFrame(result["rows"])

        Each row contains:
            - threshold
            - truePositive (TP)
            - trueNegative (TN)
            - falsePositive (FP)
            - falseNegative (FN)
            - accuracy
                Proportion of correct predictions (both true positives and true negatives) among all predictions.
                Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
            - precision:
                Proportion of positive predictions that are correct.
                Formula: Precision = TP / (TP + FP)
            - recall:
                Proportion of actual positives that are correctly identified.
                Formula: Recall = TP / (TP + FN)
            - specificity:
                Proportion of actual negatives that are correctly identified.
                Formula: Specificity = TN / (TN + FP)
            - falsePositiveRate:
                Proportion of actual negatives that are incorrectly identified as positives.
                (Assuming that that is 0 but the model predicts 1)
                Formula: FPR = FP / (FP + TN)
            - f1Score: 
                Harmonic mean of precision and recall, providing a single metric that balances both.
                Ranges from 0 to 1, where 1 indicates perfect precision and recall, and 0 indicates the worst performance.
            - mcc: Matthews Correlation Coefficient,
                which is a measure of the quality of binary classifications, taking into account true and false positives and negatives. 
                It returns a value between -1 and +1, where +1 indicates perfect prediction, 0 indicates no better than random prediction, 
                and -1 indicates total disagreement between prediction and observation.

    """
    y_true = from_list_to_np_converter(y_true)
    y_prob = from_list_to_np_converter(y_prob)

    if len(y_true) == 0 or len(y_prob) == 0:
        error_message = "y_true and y_prob must not be empty."
        logger.error(error_message)
        raise ValueError(error_message)

    if len(y_true) != len(y_prob):
        error_message = (
            f"Length mismatch: y_true has length {len(y_true)}, "
            f"but y_prob has length {len(y_prob)}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        error_message = "y_true and y_prob must not contain NaN values."
        logger.error(error_message)
        raise ValueError(error_message)

    if np.isinf(y_true).any() or np.isinf(y_prob).any():
        error_message = "y_true and y_prob must not contain infinite values."
        logger.error(error_message)
        raise ValueError(error_message)

    if not np.isin(y_true, [0, 1]).all():
        error_message = (
            f"y_true contains invalid values: {np.unique(y_true)}. "
            "Expected only 0 and 1."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if not ((0 <= y_prob) & (y_prob <= 1)).all():
        error_message = "y_prob must contain probability values between 0 and 1."
        logger.error(error_message)
        raise ValueError(error_message)

    if not thresholds:
        error_message = "thresholds list must not be empty."
        logger.error(error_message)
        raise ValueError(error_message)

    invalid_thresholds = [t for t in thresholds if not (0 <= t <= 1)]
    if invalid_thresholds:
        error_message = (
            f"All thresholds must be between 0 and 1. "
            f"Invalid values: {invalid_thresholds}"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total = int(tn + fp + fn + tp)
        accuracy = (tp + tn) / total

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        mcc_denom = np.sqrt(
            float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        mcc = float((tp * tn - fp * fn) / mcc_denom) if mcc_denom > 0 else 0.0

        rows.append({
            "threshold": float(threshold),
            "truePositive": int(tp),
            "trueNegative": int(tn),
            "falsePositive": int(fp),
            "falseNegative": int(fn),
            "accuracy": round(float(accuracy), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "specificity": round(specificity, 6),
            "falsePositiveRate": round(fpr_value, 6),
            "f1Score": round(float(f1), 6),
            "mcc": round(mcc, 6),
        })

        logger.info(
            f"Threshold={threshold:.4f} | "
            f"TP={tp} TN={tn} FP={fp} FN={fn} | "
            f"Accuracy={accuracy:.4f} Precision={precision:.4f} "
            f"Recall={recall:.4f} Specificity={specificity:.4f} "
            f"FPR={fpr_value:.4f} F1={f1:.4f} MCC={mcc:.4f}"
        )

    return {"rows": rows}

def interpret_vif(vif_value: float) -> dict:
    """
    Interpret the Variance Inflation Factor (VIF) value.

    VIF = 1: No correlation with other predictors
    1 < VIF ≤ 5: Mild to moderate correlation (usually fine)
    VIF > 10: Strong multicollinearity -> take corrective steps

    Formula for VIF is: VIF = 1 / (1 - R²)

    Where R² is the coefficient of determination of the regression of the variable against all other variables.
    R² ranges from 0 to 1.
    A higher R² means the variable is highly predictable from other variables -> higher VIF.
    If R² is close to 1, the variable is almost fully explained by others -> strong multicollinearity.

    Parameters
    ----------
    vif_value : float
        The VIF value to interpret.

    Returns
    -------
    dict
        A dictionary containing the VIF value and its interpretation.
    """

    if vif_value < 1:
        logger.warning(f"VIF value {vif_value} is less than 1, which is unexpected. VIF should be >= 1.")
        return {
            "vif": vif_value,
            "interpretation": "Unexpected VIF value. VIF should be >= 1."
        }
    elif vif_value == 1:
        return {
            "vif": vif_value,
            "interpretation": "No correlation with other predictors."
        }
    elif 1 < vif_value <= 5:
        return {
            "vif": vif_value,
            "interpretation": "Mild to moderate correlation with other predictors (usually fine)."
        }
    elif 5 < vif_value <= 10:
        return {
            "vif": vif_value,
            "interpretation": "Moderate to high correlation with other predictors. Consider checking for multicollinearity."
        }
    else:  # vif_value > 10
        return {
            "vif": vif_value,
            "interpretation": "Strong multicollinearity detected. Consider taking corrective steps (e.g., removing variables, regularization)."
        }

def plot_ks(
    decile_analysis: dict,
    figsize: tuple = (10, 6),
):
    """
    Plot KS Curve in scikit-plot style.

    X-axis: Threshold
    Y-axis: Percentage below threshold

    Returns
    -------
    matplotlib.figure.Figure
        KS plot figure.
    """

    decile_df = pd.DataFrame(
        decile_analysis["ksCurve"]
    ).copy()

    # Your ksCurve is calculated on descending scores:
    # cumulativeRate = P(score >= threshold | class)
    #
    # Scikit-plot style needs:
    # percentageBelowThreshold = P(score <= threshold | class)
    decile_df["goodBelowThreshold"] = 1 - decile_df["cumulativeGoodRate"]
    decile_df["badBelowThreshold"] = 1 - decile_df["cumulativeBadRate"]

    # For plotting by threshold from low to high
    decile_df = decile_df.sort_values(
        by="threshold",
        ascending=True
    ).reset_index(drop=True)

    # KS is absolute distance between the two curves
    decile_df["ksPlot"] = (
        decile_df["goodBelowThreshold"] -
        decile_df["badBelowThreshold"]
    ).abs()

    ks_idx = decile_df["ksPlot"].idxmax()
    ks_value = float(decile_df.loc[ks_idx, "ksPlot"])
    ks_threshold = float(decile_df.loc[ks_idx, "threshold"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        decile_df["threshold"],
        decile_df["goodBelowThreshold"],
        label="Class 0 (Good)"
    )

    ax.plot(
        decile_df["threshold"],
        decile_df["badBelowThreshold"],
        label="Class 1 (Bad)"
    )

    ax.axvline(
        ks_threshold,
        linestyle=":",
        color="black",
        label=f"KS Statistic: {ks_value:.3f} at {ks_threshold:.3f}"
    )

    ax.annotate(
        f"Threshold = {ks_threshold:.4f}",
        xy=(ks_threshold, 0.5),
        xytext=(5, 0),
        textcoords="offset points",
        rotation=90,
        va="center",
        ha="left"
    )

    ax.set_title("KS Statistic Plot")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Percentage below threshold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(False)

    logger.info(
        f"KS Plot created. "
        f"KS={ks_value:.6f}, "
        f"Threshold={ks_threshold:.6f}"
    )

    return fig