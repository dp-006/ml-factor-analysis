'''
Accelera Consulting

Reference:
https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html#

https://www.statsmodels.org/dev/stats.html
https://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#variance_inflation_factor

'''
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from helper import interpret_vif, io_save_json, io_save_dataframe_as_csv
from logging_config.logger_config import get_logger

logger_name = "mlops.vif_analysis"
logger_file_name = "vif_analysis.log"
logger = get_logger(logger_name, logger_file_name)

def variance_inflation_factor(
        data, 
        feature_idx,
        standardize=True
        ):
    """
    Variance inflation factor, VIF, for one feature.

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    feature_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, data.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by feature_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    For j = 1, ..., k, the variance inflation factor for the j-th variable is defined as:
    
    X_j = β_0 + β_1*X_1 + ... + β_k*X_k
    
    where R² is the R-squared value from the regression of X_j on all other explanatory
    variables. Then:
    
    VIF_j = 1 / (1 - R²_j)

    Parameters
    ----------
    data : {ndarray, DataFrame}
        design matrix with all explanatory variables, as for example used in
        regression
    feature_idx : int
        index of the exogenous variable in the columns of data
    standardize : bool, optional
        If True, standardizes the design matrix columns to mean 0 and standard
        deviation 1 before computing VIF. This ensures numerical stability
        for non-linear transformations or micro-scale data. Default is True.

    Returns
    -------
    float
        variance inflation factor

    interpretation : dict
        A dictionary containing the VIF value and its interpretation.

    References
    ----------
    https://en.wikipedia.org/wiki/Variance_inflation_factor
    """

    # Convert data to a numpy array of type float for numerical stability in calculations.
    data = np.asarray(data, dtype=float)
    
    # Raise Error if data contains Infinite values, as this would lead to unreliable VIF calculations.
    if np.isinf(data).any():
        error_message = ("Input data contains infinite values, which can lead to unreliable VIF calculations. "
        "Please remove or impute infinite values before calling this function.")
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error if data contains NaN values, as this would lead to unreliable VIF calculations.
    if np.isnan(data).any():
        error_message = ("Input data contains NaN values, which can lead to unreliable VIF calculations. "
        "Please remove or impute NaN values before calling this function.")
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info(f"Calculating VIF for feature index {feature_idx} with standardize={standardize}")

    # Check the number of variables in the design matrix and log it for informational purposes.
    k_vars = data.shape[1]
    logger.info(f"Number of variables in the design matrix: {k_vars}")

    if standardize:
        logger.info("Standardizing the design matrix for numerical stability.")
        stds = np.std(data, axis=0)
        means = np.mean(data, axis=0)
        safe_mask = stds > 1e-10
        working = data.copy()
        working[:, safe_mask] = (data[:, safe_mask] - means[safe_mask]) / stds[
            safe_mask
        ]
    else:
        logger.info("Using the original design matrix without standardization.")
        working = data

    # NOTE: Check the condition number of the design matrix to warn about potential numerical instability in VIF calculations.
    # A high condition number indicates that the matrix is close to singular, which can lead to unreliable VIF estimates.
    cond_num = np.linalg.cond(working)
    logger.info(f"Condition number of the design matrix: {cond_num:.2e}")
    if cond_num > 1e4:
        warning_message = (
            f"The design matrix is poorly conditioned (condition number={cond_num:.2e}). "
            "VIF calculations may be numerically unstable."
        )
        logger.warning(warning_message)

    # Check if the feature index is within the valid range of the design matrix columns.
    if feature_idx < 0 or feature_idx >= k_vars:
        error_message = (
            f"Feature index {feature_idx} is out of bounds for the design matrix with {k_vars} variables."
        )
        logger.error(error_message)
        raise IndexError(error_message)
    
    # Retrieve the column of the design matrix corresponding to the feature index and the remaining columns for the auxiliary regression.
    x_i = working[:, feature_idx]
    mask = np.arange(k_vars) != feature_idx
    x_noti = working[:, mask]

    # Fit an OLS regression of the feature against the other features and calculate the R-squared value.
    r_sq = OLS(x_i, x_noti).fit().rsquared
    logger.info(f"R-squared for feature index {feature_idx}: {r_sq:.4f}")

    # Clip R-squared to the range [0, 1 - 1e-15] to prevent numerical issues in VIF calculation when R-squared is very close to 1.
    r_sq = np.clip(r_sq, 0.0, 1.0 - 1e-15)
    logger.info(f"Clipped R-squared for feature index {feature_idx}: {r_sq:.4f}")

    # Calculate VIF using the formula VIF = 1 / (1 - R-squared) and log the result along with an interpretation of the VIF value.
    vif = 1.0 / (1.0 - r_sq)
    logger.info(f"Calculated VIF for feature index {feature_idx}: {vif:.4f}")

    # Interpret the VIF value using the helper function and log the interpretation for informational purposes.
    interpretation = interpret_vif(vif)
    logger.info(f"VIF Interpretation for feature index {feature_idx}: {interpretation['interpretation']}")

    return vif, interpretation

def convert_steps_to_dataframe(steps, remaining_features):
    """
    Convert iterative VIF selection steps to a structured DataFrame.
    
    Parameters
    ----------
    steps : dict
        Dictionary containing VIF calculation results for each step
    remaining_features : list or Index
        List of remaining feature names after iterative selection
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Feature, Step, VIF, Removed
    """
    csv_data = []
    
    # Process each step and extract VIF results
    for step_num, step_data in steps.items():
        vif_results = step_data["vifResults"]
        removed_feature = step_data["removedFeatureName"]
        
        for feature, details in vif_results.items():
            csv_data.append({
                "Feature": feature,
                "Step": step_num,
                "VIF": details["vif"],
                "Removed": "Yes" if feature == removed_feature else "No"
            })
    
    # Add final selected features with step after last iteration
    final_step = int(max(steps.keys())) + 1 if steps else 1
    for feature in remaining_features:
        # Find the last VIF for this feature
        last_vif = None
        for step_num in sorted(steps.keys(), reverse=True):
            if feature in steps[step_num]["vifResults"]:
                last_vif = steps[step_num]["vifResults"][feature]["vif"]
                break
        
        if last_vif is not None:
            csv_data.append({
                "Feature": feature,
                "Step": final_step,
                "VIF": last_vif,
                "Removed": "No"
            })
    
    return pd.DataFrame(csv_data)

def iterative_feature_selector_with_vif(
        X, 
        vif_treshold=5.0, 
        maxiterations=100, 
        output_json_path=None,
        output_csv_path=None
        ):
    """
    Iteratively removes features with the highest VIF until all remaining features have VIF below the specified threshold.

    Parameters
    ----------
    X : DataFrame
        The input features for which to calculate VIF.
    threshold : float, optional
        The VIF threshold above which features will be removed. Default is 5.0.
    maxiterations : int, optional
        The maximum number of iterations to perform. Default is 100.
    
    Steps:
    1. Calculate VIF for all features in the input DataFrame.
    2. Identify the feature with the highest VIF.
    2.1. If the highest VIF is above the specified threshold, remove that feature from the DataFrame.
    2.2. If the highest VIF is below the threshold, stop the iteration.
    3. Repeat steps 1-2 until all remaining features have VIF below the threshold.

    Returns
    -------
    DataFrame
        A DataFrame containing the remaining features after iterative removal based on VIF.
    """
    logger.info(f"Starting iterative feature selection with VIF threshold: {vif_treshold} and maximum iterations: {maxiterations}")

    # Raise Error if the input DataFrame is empty, as this would lead to unreliable VIF calculations.
    if X.empty:
        error_message = "Input DataFrame is empty. Cannot perform VIF analysis on empty data."
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise Error if the input DataFrame contains non-numeric columns, as VIF can only be calculated for numeric features.
    if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
        error_message = "Input DataFrame contains non-numeric columns. VIF can only be calculated for numeric features."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error if the input DataFrame contains NaN values, as this would lead to unreliable VIF calculations.
    if X.isnull().values.any():
        error_message = "Input DataFrame contains NaN values. Please remove or impute NaN values before performing VIF analysis."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise Error if the input DataFrame contains Infinite values, as this would lead to unreliable VIF calculations.
    if np.isinf(X.values).any():
        error_message = "Input DataFrame contains infinite values. Please remove or impute infinite values before performing VIF analysis."
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Create a copy of the input DataFrame to avoid modifying the original data.
    X_remaining = X.copy()
    
    step = 1
    steps = {}
    while True:
        # if not convergence after maxiterations, break the loop and return the remaining features.
        if step > maxiterations:
            warning_message = f"Reached maximum iterations ({maxiterations}). Stopping the iterative feature selection process."
            logger.warning(warning_message)

            # Save the final selected features and the steps taken during the iterative process to a JSON file if an output path is provided.
            if output_json_path:
                saved_path = io_save_json({"selected_features": X_remaining.columns.tolist(), "steps": steps}, output_json_path)
                logger.info(f"Saved iterative feature selection results to JSON file: {saved_path}")
            # Save steps to a CSV file if an output path is provided, with each step's VIF results and removed feature information.
            if output_csv_path:
                steps_df = convert_steps_to_dataframe(steps, X_remaining.columns)
                saved_path = io_save_dataframe_as_csv(steps_df, output_csv_path)
                logger.info(f"Saved iterative feature selection steps to CSV file: {saved_path}")

            selected_features = X_remaining.columns.tolist()
            return selected_features, steps
        
        logger.info(f"Step {step}: Calculating VIF for remaining features.")
        # Calculate VIF for all remaining features and store the results in a dictionary.
        vif_results = {}
        for i in range(X_remaining.shape[1]):
            vif, interpretation = variance_inflation_factor(X_remaining, i)
            # Append the VIF value and its interpretation to the results dictionary for the current feature.
            vif_results[X_remaining.columns[i]] = {
                "vif": vif,
                "interpretation": interpretation["interpretation"]
            }
        
        # Convert the VIF results dictionary to a DataFrame for easier analysis and logging.
        vif_df = pd.DataFrame(vif_results).T
        logger.info(f"Current VIF values:\n{vif_df}")

        # Identify the feature with the maximum VIF value and log it for informational purposes.
        max_vif_feature = vif_df["vif"].idxmax()
        max_vif_value = vif_df.loc[max_vif_feature, "vif"]
        logger.info(f"Feature with maximum VIF: {max_vif_feature} (VIF={max_vif_value:.4f})")

        # If the maximum VIF value is below the threshold, break the loop as all remaining features are acceptable.
        if max_vif_value < vif_treshold:
            logger.info("All remaining features have VIF below the threshold. Stopping iteration.")
            break
        
        # Remove the feature with the highest VIF from the remaining features and log the action.
        logger.info(f"Removing feature {max_vif_feature} due to high VIF ({max_vif_value:.4f}).")
        X_remaining.drop(columns=[max_vif_feature], inplace=True)

        # Append the current step's VIF results to the steps dictionary for tracking the iterative process.
        steps[step] = {"vifResults": vif_df.to_dict(orient="index"), "removedFeatureName": max_vif_feature, "removedFeatureVIF": max_vif_value}
        step += 1

    # Save the final selected features and the steps taken during the iterative process to a JSON file if an output path is provided.
    if output_json_path:
        saved_path = io_save_json({"selected_features": X_remaining.columns.tolist(), "steps": steps}, output_json_path)
        logger.info(f"Saved iterative feature selection results to JSON file: {saved_path}")
    # Save steps to a CSV file if an output path is provided, with each step's VIF results and removed feature information.
    if output_csv_path:
        steps_df = convert_steps_to_dataframe(steps, X_remaining.columns)
        saved_path = io_save_dataframe_as_csv(steps_df, output_csv_path)
        logger.info(f"Saved iterative feature selection steps to CSV file: {saved_path}")

    selected_features = X_remaining.columns.tolist()

    return selected_features, steps



