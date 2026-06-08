import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from logging_config.logger_config import get_logger

logger_name = "mlops.utils"
logger_file_name = "utils.log"
logger = get_logger(logger_name, logger_file_name)

def _sanitize(obj):
    """Recursively replace NaN/Inf float values with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and math.isinf(obj):
        return "infinite"
    if isinstance(obj, np.floating) and np.isnan(obj):
        return None
    if isinstance(obj, np.floating) and np.isinf(obj):
        return "infinite"
    return obj

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def shapiro_wilk(s: pd.Series) -> dict:
    """Shapiro-Wilk normality test. Best for n < 5000."""
    try:
        stat, p = stats.shapiro(s.dropna())
        stat_f, p_f = float(stat), float(p)
        if math.isnan(p_f) or math.isnan(stat_f):
            return {"statistic": None, "p_value": None, "interpretation": "not applicable"}
        return {
            "statistic": round(stat_f, 6),
            "p_value": round(p_f, 6),
            "interpretation": "normal (fail to reject H0)" if p_f > 0.05 else "not normal (reject H0)",
        }
    except Exception as e:
        logger.error(f"Shapiro-Wilk test failed for column '{s.name}': {e}")
        return {"statistic": None, "p_value": None, "interpretation": "test failed"}

def dagostino_pearson(s: pd.Series) -> dict:
    """D'Agostino-Pearson normality test. Based on skewness and kurtosis."""
    try:
        stat, p = stats.normaltest(s.dropna())
        stat_f, p_f = float(stat), float(p)
        if math.isnan(p_f) or math.isnan(stat_f):
            return {"statistic": None, "p_value": None, "interpretation": "not applicable"}
        return {
            "statistic": round(stat_f, 6),
            "p_value": round(p_f, 6),
            "interpretation": "normal (fail to reject H0)" if p_f > 0.05 else "not normal (reject H0)",
        }
    except Exception as e:
        logger.error(f"D'Agostino-Pearson test failed for column '{s.name}': {e}")
        return {"statistic": None, "p_value": None, "interpretation": "test failed"}

def kolmogorov_smirnov(s: pd.Series) -> dict:
    """Kolmogorov-Smirnov test against a normal distribution."""
    try:
        clean = s.dropna()
        stat, p = stats.kstest(clean, "norm", args=(clean.mean(), clean.std()))
        stat_f, p_f = float(stat), float(p)
        if math.isnan(p_f) or math.isnan(stat_f):
            return {"statistic": None, "p_value": None, "interpretation": "not applicable"}
        return {
            "statistic": round(stat_f, 6),
            "p_value": round(p_f, 6),
            "interpretation": "normal (fail to reject H0)" if p_f > 0.05 else "not normal (reject H0)",
        }
    except Exception as e:
        logger.error(f"Kolmogorov-Smirnov test failed for column '{s.name}': {e}")
        return {"statistic": None, "p_value": None, "interpretation": "test failed"}

def percentile_distribution(s: pd.Series, quantiles: list) -> list:
    """Count values per bucket using pd.cut with quantile boundaries."""
    logger.info(f"Computing percentile distribution for column '{s.name}' with quantiles: {quantiles}")
    try:
        clean = s.dropna()
        if len(clean) == 0:
            logger.info(f"No valid data for column '{s.name}' to compute percentile distribution.")
            return [{"error": f"No data available for column '{s.name}'"}]
        
        # Check if feature is constant
        if clean.std() == 0 or clean.min() == clean.max():
            logger.info(f"Feature '{s.name}' has constant value - percentile distribution not applicable.")
            return [{
                "bucket": "constant_feature",
                "note": "Feature has constant value - percentile distribution not applicable",
                "std": round(float(clean.std()), 6),
                "count": len(clean),
                "unique_count": clean.nunique()
            }]
        
        n = len(clean)
        boundaries = [clean.quantile(q) for q in quantiles]
        labels = [f"P{int(q * 100):02d}" for q in quantiles]

        bin_edges = [clean.min()] + boundaries + [clean.max()]
        # Remove duplicate bin edges
        bin_edges = sorted(list(set(bin_edges)))
        logger.info("Removed duplicate bin edges if any.")

        bin_labels = (
            [f"below_{labels[0]}"] +
            [f"{labels[i]}_{labels[i+1]}" for i in range(len(labels) - 1)] +
            [f"above_{labels[-1]}"]
        )[:len(bin_edges) - 1]

        thresholds = [
            {"lower": round(float(bin_edges[i]), 6), "upper": round(float(bin_edges[i+1]), 6)}
            for i in range(len(bin_edges) - 1)
        ]

        binned = pd.cut(clean, bins=bin_edges, labels=bin_labels, include_lowest=True)
        counts = binned.value_counts().reindex(bin_labels, fill_value=0)

        # log bin edges, labels, thresholds and counts for interpretability with loop
        for i in range(len(bin_edges) - 1):
            logger.info(f"Bin {bin_labels[i]}: {round(float(bin_edges[i]), 6)} to {round(float(bin_edges[i+1]), 6)}, Count: {counts[bin_labels[i]]}")

        return [
            {
                "bucket": str(bucket),
                "lower": thresholds[i]["lower"],
                "upper": thresholds[i]["upper"],
                "count": int(count),
                "ratio": round(count / n, 6),
            }
            for i, (bucket, count) in enumerate(counts.items())
            if bucket is not None
        ]
    except Exception as e:
        error_msg = f"Percentile distribution failed for column '{s.name}': {str(e)}"
        logger.error(error_msg)
        return [{
            "bucket": "error",
            "note": "Percentile distribution calculation failed",
            "error": error_msg
        }]

def save_json(data: dict | list, output_path: str, indent: int = 4) -> None:
    """
    Save a dictionary or list as a JSON file.

    Args:
        data (dict | list): Data to serialize.
        output_path (str): Full file path including filename (e.g. 'outputs/summary.json').
        indent (int): JSON indentation level.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If the data is not JSON serializable.
    """
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(data), f, indent=indent, ensure_ascii=False, cls=_NumpyEncoder)
            logger.info(f"Data successfully saved to {output_path}.")
    except (OSError, TypeError) as e:
        logger.error(f"Failed to save data to {output_path}: {e}")
        raise

def plot_boxplots(df: pd.DataFrame, output_dir: str = "outputs/boxplots") -> None:
    """
    Draw a boxplot for each numeric column in the given DataFrame.

    Each plot includes mean, median, min, max, Q1, Q3, IQR, and outlier
    counts in the legend. Files are saved as SVG under ``output_dir`` with
    the naming pattern ``<column>_bb.svg``.

    Args:
        df (pd.DataFrame): Source data. Must contain at least one numeric column.
        output_dir (str): Directory where SVG files are written.
            Created automatically if it does not exist.

    Raises:
        Exception: Re-raises any per-column plotting error after logging it.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        try:
            s = df[col].dropna()
            col_mean    = s.mean()
            col_median  = s.median()
            col_min     = s.min()
            col_max     = s.max()
            col_q1      = s.quantile(0.25)
            col_q3      = s.quantile(0.75)
            iqr         = col_q3 - col_q1
            lower_fence = col_q1 - 1.5 * iqr
            upper_fence = col_q3 + 1.5 * iqr
            lower_outliers = s[s < lower_fence]
            upper_outliers = s[s > upper_fence]

            fig, ax = plt.subplots(figsize=(6, 8))
            ax.boxplot(s, patch_artist=True)
            ax.set_title(col)
            ax.set_ylabel("Value")
            ax.set_xticks([])

            ax.plot(1, col_mean, marker="D", color="red", markersize=8, zorder=5, label=f"Mean={col_mean:.2f}")
            ax.plot([], [], color="blue",   linestyle="none", label=f"Median={col_median:.2f}")
            ax.axhline(col_min, color="green",  linestyle=":", linewidth=1, label=f"Min={col_min:.2f}")
            ax.axhline(col_max, color="orange", linestyle=":", linewidth=1, label=f"Max={col_max:.2f}")
            ax.plot([], [], color="purple", linestyle="none", label=f"Q1 (25%)={col_q1:.2f}")
            ax.plot([], [], color="brown",  linestyle="none", label=f"Q3 (75%)={col_q3:.2f}")
            ax.plot([], [], color="gray",   linestyle="none", label=f"IQR={iqr:.2f}")
            ax.plot([], [], color="black",  linestyle="none", label=f"Lower outliers (<{lower_fence:.2f}): {len(lower_outliers)}")
            ax.plot([], [], color="black",  linestyle="none", label=f"Upper outliers (>{upper_fence:.2f}): {len(upper_outliers)}")
            ax.legend(fontsize=8, loc="upper right")

            file_path = os.path.join(output_dir, f"{col}_bb.svg")
            fig.savefig(file_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Boxplot saved: {file_path}")
        except Exception as e:
            error_message = f"Failed to plot boxplot for column '{col}': {e}"
            logger.error(error_message)
            # Create a text file instead of raising exception
            txt_path = os.path.join(output_dir, f"{col}_bb.txt")
            with open(txt_path, "w") as f:
                f.write(error_message)

def plot_histograms(df: pd.DataFrame, output_dir: str = "outputs/histograms") -> None:
    """
    Draw a histogram overlaid with a KDE and a theoretical normal curve for
    each numeric column in the given DataFrame.

    The y-axis uses probability density so all three layers share the same
    scale. Mean, median, and standard deviation are displayed in the legend.
    Files are saved as SVG under ``output_dir`` with the naming pattern
    ``<column>_hist.svg``.

    Args:
        df (pd.DataFrame): Source data. Must contain at least one numeric column.
        output_dir (str): Directory where SVG files are written.
            Created automatically if it does not exist.

    Raises:
        Exception: Re-raises any per-column plotting error after logging it.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        try:
            s = df[col].dropna()
            col_mean   = s.mean()
            col_median = s.median()
            col_std    = s.std()
            x = np.linspace(s.min(), s.max(), 300)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(s, bins="auto", density=True, alpha=0.4, color="steelblue", label="Histogram")
            kde = stats.gaussian_kde(s)
            ax.plot(x, kde(x), color="steelblue", linewidth=2, label="KDE")
            ax.plot(x, stats.norm.pdf(x, col_mean, col_std),
                    color="red", linewidth=2, linestyle="--",
                    label=f"Normal (μ={col_mean:.2f}, σ={col_std:.2f})")
            
            # Add mean and ±3σ lines
            ax.axvline(col_mean, color="green", linestyle="--", linewidth=0.8, label=f"Mean = {col_mean:.4f}")
            ax.axvline(col_mean + 3*col_std, color="orange", linestyle="--", linewidth=0.7, label=f"Mean ± 3σ")
            ax.axvline(col_mean - 3*col_std, color="orange", linestyle="--", linewidth=0.7)
            
            # Calculate and display outliers
            left_outliers = (s < col_mean - 3*col_std).sum()
            right_outliers = (s > col_mean + 3*col_std).sum()
            total_outliers = left_outliers + right_outliers
            outlier_text = f"Outliers (±3σ):\nLeft: {left_outliers}\nRight: {right_outliers}\nTotal: {total_outliers}"
            ax.text(0.02, 0.98, outlier_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            ax.plot([], [], color="none", label=f"Median = {col_median:.4f}")
            ax.plot([], [], color="none", label=f"Std    = {col_std:.4f}")

            ax.set_title(col)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8, loc="upper right")

            file_path = Path(output_dir, f"{col}_hist.svg")
            fig.savefig(file_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Histogram saved: {file_path}")
        except Exception as e:
            error_message = f"Failed to plot histogram for column '{col}': {e}"
            logger.error(error_message)
            # Create a text file instead of raising exception
            txt_path = os.path.join(output_dir, f"{col}_hist.txt")
            with open(txt_path, "w") as f:
                f.write(error_message)

def plot_qq_plots(df: pd.DataFrame, output_dir: str = "outputs/qq_plots") -> None:
    """
    Draw a Q-Q (quantile-quantile) plot for each numeric column in the given
    DataFrame, comparing sample quantiles against a theoretical normal
    distribution.

    Each plot includes the data quantiles as scatter points, the fitted
    normal reference line, and the R² of that fit in the legend.
    Files are saved as SVG under ``output_dir`` with the naming pattern
    ``<column>_qq.svg``.

    Args:
        df (pd.DataFrame): Source data. Must contain at least one numeric column.
        output_dir (str): Directory where SVG files are written.
            Created automatically if it does not exist.

    Raises:
        Exception: Re-raises any per-column plotting error after logging it.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        try:
            s = df[col].dropna()

            fig, ax = plt.subplots()
            (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
            ax.scatter(osm, osr, color="steelblue", s=10, alpha=0.6, label="Data quantiles")
            x_line = np.array([osm[0], osm[-1]])
            ax.plot(x_line, slope * x_line + intercept,
                    color="red", linewidth=1.5, linestyle="--", label="Normal reference line")
            ax.plot([], [], color="none", label=f"R² = {r**2:.4f}")

            ax.set_title(f"Q-Q Plot — {col}")
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Sample quantiles")
            ax.legend(fontsize=8, loc="upper left")

            file_path = Path(output_dir, f"{col}_qq.svg")
            fig.savefig(file_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Q-Q plot saved: {file_path}")
        except Exception as e:
            error_message = f"Failed to plot Q-Q plot for column '{col}': {e}"
            logger.error(error_message)
            # Create a text file instead of raising exception
            txt_path = Path(output_dir, f"{col}_qq.txt")
            with open(txt_path, "w") as f:
                f.write(error_message)

def plot_bar_charts(
    df: pd.DataFrame,
    top_n: int = 5,
    output_dir: str = "outputs/bar_charts",
) -> None:
    """
    Draw a bar chart for each non-numeric column in the given DataFrame.

    The top ``top_n`` categories (by frequency) are shown individually;
    all remaining categories are grouped into a single "Other" bar.
    Frequency count and ratio are displayed on each bar. Files are saved
    as SVG under ``output_dir`` with the naming pattern
    ``<column>_bar.svg``.

    Args:
        df (pd.DataFrame): Source data. Must contain at least one non-numeric column.
        top_n (int): Number of top categories to show individually before
            grouping the rest as "Other". Defaults to 5.
        output_dir (str): Directory where SVG files are written.
            Created automatically if it does not exist.

    Raises:
        Exception: Re-raises any per-column plotting error after logging it.
    """
    os.makedirs(output_dir, exist_ok=True)
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

    for col in non_numeric_cols:
        try:
            s = df[col].dropna()
            value_counts = s.value_counts()
            total = len(s)

            top = value_counts.iloc[:top_n]
            other_count = value_counts.iloc[top_n:].sum()

            # Sort top categories by count (ascending)
            top_sorted = top.sort_values(ascending=True)
            labels = list(top_sorted.index.astype(str))
            counts = list(top_sorted.values)
            
            if other_count > 0:
                labels.append("Other")
                counts.append(other_count)

            ratios = [c / total for c in counts]
            # Colors: steelblue for top_n categories, orange for "Other"
            colors = ["steelblue"] * len(top) + (["orange"] if other_count > 0 else [])

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(labels, counts, color=colors)

            for bar, count, ratio in zip(bars, counts, ratios):
                # Position labels at the end of the bar (outside)
                x_pos = bar.get_width() + max(counts) * 0.02  # Slightly to the right
                y_pos = bar.get_y() + bar.get_height() / 2
                ax.text(
                    x_pos,
                    y_pos,
                    f"{count} ({ratio:.1%})",
                    ha="left", va="center", fontsize=10, color="black", weight="bold",
                )

            ax.set_title(col)
            ax.set_xlabel("Count")
            ax.set_ylabel("Category")
            
            # Add some margin to the right for labels
            ax.set_xlim(0, max(counts) * 1.25)

            file_path = Path(output_dir, f"{col}_bar.svg")
            fig.savefig(file_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Bar chart saved: {file_path}")
        except Exception as e:
            error_message = f"Failed to plot bar chart for column '{col}': {e}"
            logger.error(error_message)
            # Create a text file instead of raising exception
            txt_path = Path(output_dir, f"{col}_bar.txt")
            with open(txt_path, "w") as f:
                f.write(error_message)
