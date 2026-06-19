"""
Auto WoE Binning for Categorical Features

This module contains the auto_woe_binning_categorical function for automatic
binning and Weight of Evidence (WoE) transformation of categorical features.

The algorithm follows the same rule-based structure as auto_woe_binning_numeric
but adapted for categorical/object dtype columns.
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
from helper import io_save_json, io_save_dataframe_as_csv
from logging_config.logger_config import get_logger
from auto_binning_woe import (
    is_monotonic,
    calculate_woe_iv_table,
    check_binning_quality,
    interpret_iv,
    find_closest_neighbor,
    find_most_similar_adjacent_pair,
    find_monotonicity_violation_pair
)

logger_name = "mlops.auto_binning_woe_categorical"
logger_file_name = "auto_binning_woe_categorical.log"
logger = get_logger(logger_name, logger_file_name)


def convert_group_id_to_str(group_id: int, group_names: dict) -> str:
    """
    Purpose
    -------
    Convert a group ID to a human-readable group name string.
    
    Parameters
    ----------
    group_id : int
        The group ID.
    
    group_names : dict
        Mapping from group_id to group_name (e.g., {0: 'A', 1: 'B_C', 2: 'D'}).
    
    Returns
    -------
    str
        Human-readable group name.
    
    Note
    ----
    Called in:
    - Converting bin column values to readable format for categorical binning.
    """
    return group_names.get(group_id, str(group_id))

def calculate_woe_iv_table_categorical(
    df: pd.DataFrame,
    bin_col: str,
    target: str,
    eps: float = 0.5
) -> tuple:
    """
    Purpose
    -------
    Calculate bin-level WoE/IV statistics for categorical binning.

    This computes Good, Bad, Bad Rate, WoE and IV for a bin column that
    contains integer group IDs (instead of pandas Interval objects).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with bin assignments.

    bin_col : str
        Column name containing integer group/bin assignments.

    target : str
        Binary target variable.

    eps : float, optional
        Smoothing constant for WoE calculation.

    Returns
    -------
    tuple
        (summary_table, metadata)

    Note
    ----
    The bin_col stays as integer group IDs so the iterative merge logic keeps
    working. Renaming to feature_grpN names is only done at the final output.
    """
    logger.info(f"Calculating WoE / IV table for categorical binning. bin_col='{bin_col}'")

    tmp = df[[bin_col, target]].copy()

    # Group by bin and calculate statistics
    summary = (
        tmp.groupby(bin_col, observed=False)[target]
        .agg(total="count", bad="sum")
        .reset_index()
    )

    summary["good"] = summary["total"] - summary["bad"]
    summary["bad_rate"] = summary["bad"] / summary["total"]
    summary["bin_pct"] = summary["total"] / summary["total"].sum()

    total_good = summary["good"].sum()
    total_bad = summary["bad"].sum()

    logger.info(f"Total good: {total_good}, Total bad: {total_bad}")

    # Apply smoothing
    summary["good_dist"] = (summary["good"] + eps) / (total_good + eps * len(summary))
    summary["bad_dist"] = (summary["bad"] + eps) / (total_bad + eps * len(summary))
    summary["odds_ratio"] = summary["good_dist"] / summary["bad_dist"]
    summary["woe"] = np.log(summary["odds_ratio"])
    summary["woe_display"] = summary["woe"] * 100
    summary["good_dist_minus_bad_dist"] = summary["good_dist"] - summary["bad_dist"]
    summary["iv"] = (summary["good_dist"] - summary["bad_dist"]) * summary["woe"]

    column_order = [
        bin_col,
        "total",
        "good",
        "bad",
        "bad_rate",
        "bin_pct",
        "good_dist",
        "bad_dist",
        "good_dist_minus_bad_dist",
        "odds_ratio",
        "woe",
        "woe_display",
        "iv"
    ]

    summary_reordered = summary[column_order]

    total_iv = summary["iv"].sum()
    total_iv_interpretation = interpret_iv(total_iv)

    logger.info(f"WoE / IV table calculated. Bin count={len(summary)}, Total IV={total_iv:.6f}")

    metadata = {
        "binCol": bin_col,
        "target": target,
        "eps": eps,
        "totalIv": total_iv,
        "interpretIv": total_iv_interpretation,
        "numberofBins": len(summary_reordered)
    }

    return summary_reordered, metadata

def create_initial_category_groups(
    categories: list
) -> dict:
    """
    Purpose
    -------
    Create initial category groups where each distinct category is its own bin.
    
    Parameters
    ----------
    categories : list
        List of distinct categories.
    
    Returns
    -------
    dict
        Dictionary mapping category -> group_id.
        Example: {'A': 0, 'B': 1, 'C': 2}
    
    Note
    ----
    Called in:
    - Initial category group creation before Rule 1.
    """
    logger.info(f"Creating initial category groups. Category count={len(categories)}")
    
    category_groups = {}
    for idx, category in enumerate(categories):
        category_groups[category] = idx
    
    logger.info(f"Initial category groups created:")
    for cat, group_id in category_groups.items():
        logger.info(f"  - Category: '{cat}' -> Group: {group_id}")
    
    return category_groups

def assign_bins_from_category_groups(
    x: pd.Series,
    category_groups: dict
) -> pd.Series:
    """
    Purpose
    -------
    Assign categorical values to the current category groups.
    
    This function maps each category to its current group assignment
    based on the category_groups mapping.
    
    Parameters
    ----------
    x : pd.Series
        Categorical feature values.
    
    category_groups : dict
        Mapping from category to group_id.
    
    Returns
    -------
    pd.Series
        Group assignments for each observation.
    
    Note
    ----
    Called in:
    - After every merge operation.
    - Before recalculating WoE / IV.
    - Final bin assignment step.
    """
    logger.info(f"Assigning observations to category groups. Group count={len(set(category_groups.values()))}")
    
    # Map categories to group IDs
    result = x.map(category_groups)
    
    logger.info("Observations assigned to groups:")
    for group_id in sorted(set(category_groups.values())):
        count_in_group = (result == group_id).sum()
        categories_in_group = [cat for cat, gid in category_groups.items() if gid == group_id]
        logger.info(f"\tGroup {group_id}: {count_in_group} observations, Categories: {categories_in_group}")
    
    return result

def merge_category_groups(
    category_groups: dict,
    idx1: int,
    idx2: int
) -> dict:
    """
    Purpose
    -------
    Merge two category groups by reassigning all categories from one group to another.
    
    Parameters
    ----------
    category_groups : dict
        Current mapping from category to group_id.
    
    idx1 : int
        Source group ID (categories from this group will be merged).
    
    idx2 : int
        Target group ID (categories will be assigned to this group).
    
    Returns
    -------
    dict
        Updated category_groups mapping with merged groups.
    
    Note
    ----
    - Groups are renumbered after merge to maintain sequential IDs.
    - Called after finding a pair to merge.
    """
    logger.info(f"Merging category groups. Merging group {idx1} into group {idx2}")
    
    # Create new mapping
    new_groups = {}
    for cat, group_id in category_groups.items():
        if group_id == idx1:
            new_groups[cat] = idx2
        else:
            new_groups[cat] = group_id
    
    # Renumber groups to be sequential
    unique_groups = sorted(set(new_groups.values()))
    group_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_groups)}
    
    renumbered_groups = {cat: group_mapping[gid] for cat, gid in new_groups.items()}
    
    logger.info(f"Groups merged and renumbered. New group count={len(set(renumbered_groups.values()))}")
    
    return renumbered_groups

def get_group_names(
    category_groups: dict
) -> dict:
    """
    Purpose
    -------
    Create human-readable names for each group based on its member categories.
    
    Parameters
    ----------
    category_groups : dict
        Mapping from category to group_id.
    
    Returns
    -------
    dict
        Mapping from group_id to group_name.
        Example: {0: 'A', 1: 'B_C', 2: 'D'}
    
    Note
    ----
    Group names are created by joining category names with underscores.
    """
    logger.info("Creating human-readable group names")
    
    group_to_categories = {}
    for cat, group_id in category_groups.items():
        if group_id not in group_to_categories:
            group_to_categories[group_id] = []
        group_to_categories[group_id].append(str(cat))
    
    # Sort categories within each group for consistent naming
    group_names = {}
    for group_id in sorted(group_to_categories.keys()):
        sorted_cats = sorted(group_to_categories[group_id])
        group_names[group_id] = "_".join(sorted_cats)
    
    logger.info("Group names created:")
    for group_id, group_name in group_names.items():
        logger.info(f"  - Group {group_id}: '{group_name}'")
    
    return group_names

def auto_woe_binning_categorical(
    df: pd.DataFrame,
    feature: str,
    target: str,
    min_bin_pct: float = 0.05,
    max_final_bins: int = 10,
    min_final_bins: int = 3,
    min_iv: float = 0.02,
    max_iv: float = 0.50,
    max_iter: int = 20
) -> dict:
    """
    Purpose
    -------
    Automatically perform monotonic WoE binning for a categorical variable.

    The algorithm starts by creating a separate bin for each distinct category,
    then iteratively merges adjacent categories based on bad_rate or WoE similarity
    until the required binning rules are satisfied.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    feature : str
        Categorical feature (object dtype) to be binned.

    target : str
        Binary target variable.
        Expected convention:
        - 0 = Good
        - 1 = Bad

    min_bin_pct : float, optional
        Minimum allowed observation ratio per bin.
        Default is 0.05.

    max_final_bins : int, optional
        Maximum allowed number of final bins.
        Default is 10.

    min_final_bins : int, optional
        Minimum allowed number of final bins.
        Default is 3.

    min_iv : float, optional
        Minimum acceptable Information Value.
        Default is 0.02.

    max_iv : float, optional
        Maximum acceptable Information Value.
        Values above this level may indicate leakage.
        Default is 0.50.

    max_iter : int, optional
        Maximum number of merge iterations.
        Default is 20.

    Returns
    -------
    dict
        Dictionary containing:
        - feature
        - woe_table
        - total_iv
        - checks
        - status
        - category_groups
        - group_names

    Raises
    ------
    ValueError
        If the supplied feature is not categorical (object dtype).
    """
    logger.info(f"Starting categorical WoE auto binning. Feature:{feature}, Target:{target}")
    
    # Raise error if Target column is not present in the dataframe
    if target not in df.columns:
        error_message = (f"Target column: {target} not found in the dataframe. Please provide a valid target column.")
        logger.error(error_message)
        raise ValueError(error_message)
    
    # Raise error if feature and target are the same
    if feature == target:
        error_message = (f"Feature Name: {feature} and Target Name: {target} cannot be the same. Please provide different feature and target columns.")
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise error if the feature is not object/categorical type
    if not is_object_dtype(df[feature]):
        error_message = (f"Feature: {feature} is not object/categorical dtype. auto_woe_binning_categorical only supports object dtype features.")
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise error if feature contains NULL values
    if df[feature].isnull().any():
        error_message = (f"Feature: {feature} contains NULL values, which cannot be binned. Please handle NULL values before applying auto_woe_binning_categorical.")
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise error if feature is constant
    if df[feature].nunique() == 1:
        error_message = (f"Feature: {feature} is constant, which cannot be binned. Please remove or handle constant features before applying auto_woe_binning_categorical.")
        logger.error(error_message)
        raise ValueError(error_message)

    # NOTE: Features with exactly 2 unique values are treated as flag variables.
    # They are not rejected. Instead, each category becomes its own group, WoE is
    # computed directly, and the rule-based merging is skipped (handled below).

    # Raise error if target variable is not binary
    if df[target].nunique() != 2:
        error_message = (f"Target variable: {target} is not binary, WoE binning requires a binary target variable.")
        logger.error(error_message)
        raise ValueError(error_message)

    # Raise error if target is not in {0, 1} format
    if set(df[target].unique()) != {0, 1}:
        error_message = (f"Target variable: {target} has unique values {set(df[target].unique())}, but expected {{0, 1}} with convention 0=Good and 1=Bad.")
        logger.error(error_message)
        raise ValueError(error_message)

    # Prepare data
    data = df[[feature, target]].copy()
    before_drop = len(data)
    data = data.dropna(subset=[feature, target])
    after_drop = len(data)

    logger.info(f"Missing rows dropped. Before={before_drop}, After={after_drop}, Dropped={before_drop - after_drop}")

    data[target] = data[target].astype(int)
    logger.info(f"Target variable {target} converted to integer type.")

    # Get distinct categories and sort them by bad_rate for initial ordering
    distinct_categories = sorted(data[feature].unique())
    logger.info(f"Distinct categories found: {distinct_categories}")

    # Create initial category groups (each category is its own group initially)
    category_groups = create_initial_category_groups(distinct_categories)

    steps_metadata = {}
    converged = False
    step = 0

    # Flag variable handling: a feature with exactly 2 unique values is treated as
    # a flag. Each category stays as its own group, WoE is computed directly, and
    # the rule-based merging loop is skipped entirely.
    is_flag = len(distinct_categories) == 2
    if is_flag:
        logger.info(f"Feature '{feature}' is a flag variable (2 unique values). "
                    f"Creating 2 groups, computing WoE and skipping rule-based merging.")
        converged = True

    # Start the iterative merging process (skipped entirely for flag variables)
    iteration_range = [] if is_flag else range(1, max_iter + 1)
    for step in iteration_range:
        logger.info(f"--- Iteration {step} ---")

        triggered_rule = None

        # Assign bins based on current category groups
        data["_bin"] = assign_bins_from_category_groups(
            x=data[feature],
            category_groups=category_groups
        )
        logger.info(f"STEP: {step} - Bins assigned based on current category groups.")

        # Calculate WoE / IV summary table
        summary, _ = calculate_woe_iv_table_categorical(
            df=data,
            bin_col="_bin",
            target=target,
            eps=0.5
        )
        logger.info(f"STEP: {step} - WoE / IV summary table calculated.")

        # Sort summary by bad_rate for ordering (since categories don't have natural order)
        summary = summary.sort_values("bad_rate").reset_index(drop=True)
        logger.info(f"STEP: {step} - Summary table sorted by bad_rate.")

        current_iv = summary["iv"].sum()
        current_bin_count = len(set(category_groups.values()))

        logger.info(f"STEP: {step} - Current bin count={current_bin_count}, Current IV={current_iv:.6f}")

        # Convert bin column to string for JSON serialization
        summary_for_metadata = summary.copy()
        summary_for_metadata["_bin"] = summary_for_metadata["_bin"].astype(str)

        steps_metadata[step] = {
            "bin_count": current_bin_count,
            "iv": current_iv,
            "triggered_rule": triggered_rule,
            "woe": summary_for_metadata.to_dict(orient="records")
        }

        logger.info(f"STEP: {step} - RULES CHECK START")

        # Rule 0: Minimum bin count stop condition
        if current_bin_count <= min_final_bins:
            triggered_rule = "STOPPED: Minimum bin count reached"
            steps_metadata[step]["triggered_rule"] = triggered_rule
            converged = True
            logger.info(f"STEP: {step} - Minimum bin count ({min_final_bins}) reached. Stopping with current bins.")
            break

        # Rule 1: Merge bins with insufficient observation ratio
        if (summary["bin_pct"] < min_bin_pct).any() and current_bin_count > 1:
            idx = summary["bin_pct"].idxmin()
            problem_bin = summary.loc[idx, "_bin"]

            # Find closest neighbor by bad_rate similarity
            merge_idx = find_closest_neighbor(
                summary=summary,
                idx=idx,
                metric="bad_rate"
            )
            merge_bin = summary.loc[merge_idx, "_bin"]

            triggered_rule = "Rule 1: Small bin merge"
            logger.info(f"Rule 1 triggered: small bin merge.")
            logger.info(f"STEP: {step} - Problem bin={problem_bin}, Merge with bin={merge_bin}")

            # Merge the bins
            category_groups = merge_category_groups(
                category_groups=category_groups,
                idx1=merge_bin,
                idx2=problem_bin
            )
            logger.info(f"STEP: {step} - Category groups merged based on Rule 1.")
            steps_metadata[step]["triggered_rule"] = triggered_rule
            continue

        # Rule 2: Merge bins with zero good or zero bad
        zero_mask = (summary["good"] == 0) | (summary["bad"] == 0)
        if zero_mask.any() and current_bin_count > 1:
            idx = zero_mask.idxmax()
            problem_bin = summary.loc[idx, "_bin"]

            merge_idx = find_closest_neighbor(
                summary=summary,
                idx=idx,
                metric="bad_rate"
            )
            merge_bin = summary.loc[merge_idx, "_bin"]

            triggered_rule = "Rule 2: Zero good/bad merge"
            logger.info(f"Rule 2 triggered: zero good/bad merge.")
            logger.info(f"STEP: {step} - Problem bin={problem_bin}, Merge with bin={merge_bin}")

            category_groups = merge_category_groups(
                category_groups=category_groups,
                idx1=merge_bin,
                idx2=problem_bin
            )
            logger.info(f"STEP: {step} - Category groups merged based on Rule 2.")
            steps_metadata[step]["triggered_rule"] = triggered_rule
            continue

        # Rule 3: Bad Rate monotonicity (for sorted categories by bad_rate)
        if not is_monotonic(summary["bad_rate"]) and current_bin_count > 1:
            merge_idx = find_monotonicity_violation_pair(
                summary=summary,
                metric="bad_rate"
            )
            logger.info(f"Monotonicity violation detected in bad_rate. Merge pair index identified: {merge_idx}")

            if merge_idx is None:
                merge_idx = find_most_similar_adjacent_pair(
                    summary=summary,
                    metric="bad_rate"
                )
                logger.info(f"No monotonicity violation pair found. Fallback to most similar adjacent pair. Merge pair index identified: {merge_idx}")

            bin1 = summary.loc[merge_idx, "_bin"]
            bin2 = summary.loc[merge_idx + 1, "_bin"]

            triggered_rule = "Rule 3: Bad Rate monotonicity merge"
            logger.info(f"Rule 3 triggered: Bad Rate monotonicity merge.")
            logger.info(f"STEP: {step} - Merge pair=({bin1}, {bin2})")

            category_groups = merge_category_groups(
                category_groups=category_groups,
                idx1=bin2,
                idx2=bin1
            )
            logger.info(f"STEP: {step} - Category groups merged based on Rule 3.")
            steps_metadata[step]["triggered_rule"] = triggered_rule
            continue

        # Rule 4: WoE monotonicity
        if not is_monotonic(summary["woe"]) and current_bin_count > 1:
            merge_idx = find_monotonicity_violation_pair(
                summary=summary,
                metric="woe"
            )

            if merge_idx is None:
                merge_idx = find_most_similar_adjacent_pair(
                    summary=summary,
                    metric="woe"
                )

            bin1 = summary.loc[merge_idx, "_bin"]
            bin2 = summary.loc[merge_idx + 1, "_bin"]

            triggered_rule = "Rule 4: WoE monotonicity merge"
            logger.info(f"Rule 4 triggered: WoE monotonicity merge.")
            logger.info(f"STEP: {step} - Merge pair=({bin1}, {bin2})")

            category_groups = merge_category_groups(
                category_groups=category_groups,
                idx1=bin2,
                idx2=bin1
            )
            logger.info(f"STEP: {step} - Category groups merged based on Rule 4.")
            steps_metadata[step]["triggered_rule"] = triggered_rule
            continue

        # Rule 5: Reduce bin count if exceeds maximum
        if current_bin_count > max_final_bins:
            merge_idx = find_most_similar_adjacent_pair(
                summary=summary,
                metric="bad_rate"
            )

            bin1 = summary.loc[merge_idx, "_bin"]
            bin2 = summary.loc[merge_idx + 1, "_bin"]

            triggered_rule = "Rule 5: Reducing bin count"
            logger.info(f"Rule 5 triggered: reducing bin count.")
            logger.info(f"STEP: {step} - Merge pair=({bin1}, {bin2})")

            category_groups = merge_category_groups(
                category_groups=category_groups,
                idx1=bin2,
                idx2=bin1
            )
            logger.info(f"STEP: {step} - Category groups merged based on Rule 5.")
            steps_metadata[step]["triggered_rule"] = triggered_rule
            continue

        # All rules satisfied - stop iteration
        triggered_rule = "STOPPED: All rules satisfied"
        steps_metadata[step]["triggered_rule"] = triggered_rule
        converged = True
        logger.info(f"No more merge required. Stopping at iteration={step}")
        break

    # Final check if converged
    if not converged:
        logger.warning(f"Auto binning did NOT converge for feature '{feature}'. "
                       f"Reached max_iter={max_iter} while rule violations were still present.")

    logger.info(f"Iterative merging process completed after {step} iterations.")
    logger.info("----FINAL BINNING RESULTS----")

    # Final bin assignment
    data["_bin"] = assign_bins_from_category_groups(
        x=data[feature],
        category_groups=category_groups
    )

    # Final WoE / IV calculation
    woe_table, metadata_of_woe = calculate_woe_iv_table_categorical(
        df=data,
        bin_col="_bin",
        target=target
    )

    # Sort by bad_rate for consistency
    woe_table = woe_table.sort_values("bad_rate").reset_index(drop=True)

    # Calculate total IV
    total_iv = woe_table["iv"].sum()

    # Build the final bin name mapping (group_id -> feature_grpN) and
    # the mapping that stores which category belongs to which group.
    final_group_ids = sorted(set(category_groups.values()))
    generated_bin_names = {gid: f"{feature}_grp{gid}" for gid in final_group_ids}

    # values_to_the_group: feature_grpN -> list of original categories
    values_to_the_group = {}
    for category, group_id in category_groups.items():
        bin_name = generated_bin_names[group_id]
        values_to_the_group.setdefault(bin_name, []).append(category)
    for bin_name in values_to_the_group:
        values_to_the_group[bin_name] = sorted(values_to_the_group[bin_name])

    # Rename the integer group IDs in the final woe_table to feature_grpN names
    woe_table["_bin"] = woe_table["_bin"].map(generated_bin_names)

    # Perform quality checks
    checks, checks_formatted = check_binning_quality(
        summary=woe_table,
        min_bin_pct=min_bin_pct,
        max_bins=max_final_bins,
        min_bins=min_final_bins
    )

    # Final status assignment
    if is_flag:
        # Flag variables are not subject to the standard binning rules. We only
        # flag IV-based data quality concerns, otherwise pass them through.
        if total_iv < min_iv:
            status = "REVIEW_WEAK_VARIABLE"
            logger.warning(f"Flag variable final status: {status} (IV={total_iv:.6f} < {min_iv})")
        elif total_iv > max_iv:
            status = "REVIEW_POSSIBLE_LEAKAGE"
            logger.warning(f"Flag variable final status: {status} (IV={total_iv:.6f} > {max_iv})")
        else:
            status = "PASS_FLAG"
            logger.info(f"Flag variable final status: {status}")
    elif not converged:
        status = "REVIEW_NOT_CONVERGED"
        logger.warning(f"Final status: {status} (max_iter reached)")
    elif total_iv < min_iv:
        status = "REVIEW_WEAK_VARIABLE"
        logger.warning(f"Final status: {status} (IV={total_iv:.6f} < {min_iv})")
    elif total_iv > max_iv:
        status = "REVIEW_POSSIBLE_LEAKAGE"
        logger.warning(f"Final status: {status} (IV={total_iv:.6f} > {max_iv})")
    elif all(checks.values()):
        status = "PASS"
        logger.info(f"Final status: {status} (all checks passed)")
    else:
        status = "REVIEW"
        logger.warning(f"Final status: {status} (some checks failed)")

    logger.info(f"Feature: {feature}, Final bins={len(woe_table)}, Total IV={total_iv:.6f}, Status={status}")

    logger.info("Values to Group mapping:")
    for bin_name, categories in values_to_the_group.items():
        logger.info(f"  - {bin_name}: {categories}")

    # Prepare output
    woe_table_for_json = woe_table.copy()
    woe_table_for_json["_bin"] = woe_table_for_json["_bin"].astype(str)

    auto_binning_result = {
        "feature": feature,
        "totalIv": metadata_of_woe.get("totalIv", "not available"),
        "interpretIv": metadata_of_woe.get("interpretIv", "not available"),
        "numberofBins": metadata_of_woe.get("numberofBins", "not available"),
        "status": status,
        "valuesToTheGroup": values_to_the_group,
        "woe_table": woe_table_for_json.to_dict(orient="records"),
        "checks": checks_formatted
    }

    # Save results
    output_path = f"./outputs/auto_binning_woe/{feature}/final_auto_binning_woe.json"
    saved_path = io_save_json(auto_binning_result, output_path)
    logger.info(f"Auto binning result saved to {saved_path}")

    csv_output_path = f"./outputs/auto_binning_woe/{feature}/final_woe_table.csv"
    saved_path = io_save_dataframe_as_csv(woe_table, csv_output_path)
    logger.info(f"Final WOE table saved to {saved_path}")

    # Log summary
    logger.info("=" * 50)
    logger.info(f"INITIALIZE PARAMETERS")
    logger.info(f"Feature:..................... {feature}")
    logger.info(f"Target:...................... {target}")
    logger.info(f"Minimum Bin Percentage:...... {min_bin_pct}")
    logger.info(f"Maximum Final Bins:.......... {max_final_bins}")
    logger.info(f"Minimum Final Bins:.......... {min_final_bins}")
    logger.info(f"Minimum IV:.................. {min_iv}")
    logger.info(f"Maximum IV:.................. {max_iv}")
    logger.info(f"Maximum Iterations:.......... {max_iter}")
    logger.info("=" * 50)

    logger.info("=" * 50)
    logger.info("AUTO BINNING RESULTS")
    logger.info(f"Stopping Iteration:............... {step}")
    logger.info(f"Feature Name:..................... {feature}")
    logger.info(f"Final Bin Count:.................. {len(woe_table)}")
    logger.info(f"Total Information Value (IV):..... {total_iv:.6f}")
    logger.info(f"IV Interpretation:................ {interpret_iv(total_iv)}")
    logger.info(f"Binning Status:................... {status}")
    logger.info("-" * 50)
    logger.info("WoE Summary Table:")
    logger.info(f"\n{woe_table.to_string(index=False)}\n")
    logger.info("-" * 50)
    logger.info("Values to Group:")
    for bin_name, categories in values_to_the_group.items():
        logger.info(f"  {bin_name}: {categories}")
    logger.info("-" * 50)
    logger.info("Quality Check Results:")
    for check_name, check_result in checks_formatted.items():
        logger.info(f"  {check_name}: {check_result}")
    logger.info("-" * 50)

    logger.info(f"Auto binning completed successfully for feature: '{feature}'")
    logger.info("=" * 50)

    return auto_binning_result
