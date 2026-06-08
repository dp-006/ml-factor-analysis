<<<<<<< HEAD
'''
Accelera Consulting - 2026

Descriptive Analysis Module

Based on the following sources:
- https://metricgate.com/docs/kaiser-meyer-olkin/
- https://metricgate.com/docs/bartlett-sphericity-test/
- https://metricgate.com/docs/eigenvalue-calculator/


For JSON outputs, we are applying CamelCase naming convention for keys to be consistent with other outputs in the project. 
(Snake case is used for variable names in the code, but JSON outputs use CamelCase for better readability and consistency with other outputs in the project.)

'''
import json
from typing import Union
import pandas as pd
from logging_config.logger_config import get_logger

from helper import (
    save_json,
    shapiro_wilk,
    dagostino_pearson,
    kolmogorov_smirnov,
    percentile_distribution,
    plot_boxplots,
    plot_histograms,
    plot_qq_plots,
    plot_bar_charts,
)


logger_name = "mlops.descriptive_analysis"
logger_file_name = "descriptive_analysis.log"
logger = get_logger(logger_name, logger_file_name)


def load_data_schema(data_schema: Union[dict, str] = None) -> dict:
    """
    Load data schema from a dictionary or JSON file.

    Args:
        data_schema (dict or str, optional): Data schema as a dictionary or path to a JSON file.
            If a string path is provided, it will be loaded from the file. Defaults to None.

    Returns:
        dict: The loaded schema dictionary, or None if no schema was provided.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
        Exception: For other errors during loading.
    """
    if data_schema is None:
        return None

    if isinstance(data_schema, dict):
        logger.info("Data schema provided as dictionary.")
        return data_schema

    if isinstance(data_schema, str):
        try:
            with open(data_schema, 'r') as f:
                loaded_schema = json.load(f)
            logger.info(f"Data schema loaded from {data_schema}")
            return loaded_schema
        except FileNotFoundError:
            error_msg = f"Schema file not found: {data_schema}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON in schema file: {data_schema}"
            logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, doc="", pos=0)
        except Exception as e:
            error_msg = f"Error loading schema file: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    return None

def get_descriptive_statistics(
    df: pd.DataFrame = None,
    plot_boxplot: bool = True,
    plot_histogram: bool = True,
    plot_qq: bool = True,
    plot_bar: bool = True,
    bar_top_n: int = 5,
    subfolder: str = "."
) -> pd.DataFrame:
    """
    Return the number of variables and their data types.

    Args:
        plot_boxplot (bool): If True, generates and saves a boxplot for each
            numeric column. Defaults to True.
        plot_histogram (bool): If True, generates and saves a histogram + KDE +
            normal curve for each numeric column. Defaults to True.
        plot_qq (bool): If True, generates and saves a Q-Q plot for each
            numeric column. Defaults to True.
        plot_bar (bool): If True, generates and saves a bar chart for each
            non-numeric column. Defaults to True.
        bar_top_n (int): Number of top categories shown individually in the
            bar chart; the rest are grouped as "Other". Defaults to 5.
        subfolder (str): Sub-directory under ``outputs/descriptive_analysis/`` where all outputs
            (JSON and plots) are saved. Defaults to ``"train"``.

    Returns:
        pd.DataFrame: DataFrame with columns ['variable', 'dtype'] and a
                        total variable count logged.

    Raises:
        RuntimeError: If data has not been loaded yet.
    """
    if df is None:
        raise RuntimeError("No data available. Please load data before calling get_descriptive_statistics.")

    try:
        QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        summary = df.dtypes.reset_index()
        summary.columns = ["variable", "dtype"]
        summary["dtype"] = summary["dtype"].astype(str)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        numeric_stats = {}
        for col in numeric_cols:
            s = df[col]
            stats = {
                "count": int(s.count()),
                "null_count": int(s.isna().sum()),
                "null_ratio": round(s.isna().sum() / len(s), 6),
                "mean": round(s.mean(), 6),
                "median": round(s.median(), 6),
                "std": round(s.std(), 6),
                "variance": round(s.var(), 6),
                "min": round(s.min(), 6),
                "max": round(s.max(), 6),
                "IQR": round(s.quantile(0.75) - s.quantile(0.25), 6),
                "skewness": round(s.skew(), 6),
                "skew_interpretation": (
                    "highly negatively skewed" if s.skew() < -1 else
                    "moderately negatively skewed" if s.skew() < -0.5 else
                    "approximately symmetric" if s.skew() <= 0.5 else
                    "moderately positively skewed" if s.skew() <= 1 else
                    "highly positively skewed"
                ),
                "kurtosis": round(s.kurt(), 6),
                "kurtosis_interpretation": (
                    "platykurtic (flat, light-tailed)" if s.kurt() < -1 else
                    "mesokurtic (normal-like)" if s.kurt() <= 1 else
                    "leptokurtic (peaked, heavy-tailed)"
                ),
            }
            for q in QUANTILES:
                label = f"P{int(q * 100):02d}"
                stats[label] = round(s.quantile(q), 6)
            stats["normality_tests"] = {
                "shapiro_wilk": shapiro_wilk(s),
                "dagostino_pearson": dagostino_pearson(s),
                "kolmogorov_smirnov": kolmogorov_smirnov(s),
            }
            stats["percentile_distribution"] = percentile_distribution(s, QUANTILES)
            numeric_stats[col] = stats

        categorical_stats = {}
        for col in non_numeric_cols:
            s = df[col]
            value_counts = s.value_counts(dropna=False)
            mode_val = s.mode()
            categorical_stats[col] = {
                "count": int(s.count()),
                "null_count": int(s.isna().sum()),
                "null_ratio": round(s.isna().sum() / len(s), 6),
                "unique_count": int(s.nunique()),
                "mode": str(mode_val.iloc[0]) if not mode_val.empty else None,
                "mode_frequency": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "mode_ratio": round(value_counts.iloc[0] / len(s), 6) if not value_counts.empty else 0.0,
                "frequency": [
                    {
                        "value": str(val),
                        "count": int(cnt),
                        "ratio": round(cnt / len(s), 6),
                    }
                    for val, cnt in value_counts.items()
                ],
            }

        variable_summary = {}
        for row in summary.to_dict(orient="records"):
            col = row["variable"]
            entry = {"dtype": row["dtype"]}
            if col in numeric_stats:
                entry["statistics"] = numeric_stats[col]
            elif col in categorical_stats:
                entry["statistics"] = categorical_stats[col]
            variable_summary[col] = entry

        n_vars = len(summary)
        logger.info(f"Total variables: {n_vars}")
        logger.info(f"\n{summary.to_string(index=False)}")

        save_json({"features": variable_summary}, f"outputs/descriptive_analysis/{subfolder}/summary_stats.json")
        logger.info(f"Variable summary saved to outputs/descriptive_analysis/{subfolder}/summary_stats.json.")

        if plot_boxplot:
            plot_boxplots(df, output_dir=f"outputs/descriptive_analysis/{subfolder}/boxplots")
        if plot_histogram:
            plot_histograms(df, output_dir=f"outputs/descriptive_analysis/{subfolder}/histograms")
        if plot_qq:
            plot_qq_plots(df, output_dir=f"outputs/descriptive_analysis/{subfolder}/qq_plots")
        if plot_bar:
            plot_bar_charts(df, top_n=bar_top_n, output_dir=f"outputs/descriptive_analysis/{subfolder}/bar_charts")

        return summary
    except Exception as e:
        error_message = f"Error computing variable summary: {e}"
        logger.error(error_message)
        raise Exception(error_message) from e

if __name__ == "__main__":
    input_data_path = "inputs/sample/uci_credit_card_dataset.csv"
    input_schema_path = "inputs/sample/datatypes.json"
    data_schema = load_data_schema(input_schema_path).get('column_dtypes')
    X_train = pd.read_csv(input_data_path, dtype=data_schema)   
    _ = get_descriptive_statistics(df=X_train)
=======
'''
Accelera Consulting - 2026

Descriptive Analysis Module

Based on the following sources:
- https://metricgate.com/docs/kaiser-meyer-olkin/
- https://metricgate.com/docs/bartlett-sphericity-test/
- https://metricgate.com/docs/eigenvalue-calculator/


For JSON outputs, we are applying CamelCase naming convention for keys to be consistent with other outputs in the project. 
(Snake case is used for variable names in the code, but JSON outputs use CamelCase for better readability and consistency with other outputs in the project.)

'''
import json
from typing import Union
import pandas as pd
import logging
from logging_config.logger_config import setup_logger

from helper import (
    save_json,
    shapiro_wilk,
    dagostino_pearson,
    kolmogorov_smirnov,
    percentile_distribution,
    plot_boxplots,
    plot_histograms,
    plot_qq_plots,
    plot_bar_charts,
)


logger_name = "mlops.descriptive_analysis"

try:
    # Check if logger is already configured in logging library
    if logger_name in logging.Logger.manager.loggerDict:
        # Logger exists, use it
        logger = logging.getLogger(logger_name)
    else:
        # Logger doesn't exist, setup new one with proper configuration
        logger = setup_logger(
            name=logger_name,
            level="info",
            log_to_file=True,
            log_mode="w",
            timestamp="test_timestamp",
            runid="test_run",
            propagate=False
        )
except Exception as e:
    # Fallback: if anything fails, create basic logger
    print(f"[ERROR] Logger setup failed for {logger_name}: {e}")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.warning(f"Using fallback logger: {e}")


def load_data_schema(data_schema: Union[dict, str] = None) -> dict:
    """
    Load data schema from a dictionary or JSON file.

    Args:
        data_schema (dict or str, optional): Data schema as a dictionary or path to a JSON file.
            If a string path is provided, it will be loaded from the file. Defaults to None.

    Returns:
        dict: The loaded schema dictionary, or None if no schema was provided.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
        Exception: For other errors during loading.
    """
    if data_schema is None:
        return None

    if isinstance(data_schema, dict):
        logger.info("Data schema provided as dictionary.")
        return data_schema

    if isinstance(data_schema, str):
        try:
            with open(data_schema, 'r') as f:
                loaded_schema = json.load(f)
            logger.info(f"Data schema loaded from {data_schema}")
            return loaded_schema
        except FileNotFoundError:
            error_msg = f"Schema file not found: {data_schema}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON in schema file: {data_schema}"
            logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, doc="", pos=0)
        except Exception as e:
            error_msg = f"Error loading schema file: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    return None

def get_descriptive_statistics(
    df: pd.DataFrame = None,
    plot_boxplot: bool = True,
    plot_histogram: bool = True,
    plot_qq: bool = True,
    plot_bar: bool = True,
    bar_top_n: int = 5,
    subfolder: str = "."
) -> pd.DataFrame:
    """
    Return the number of variables and their data types.

    Args:
        plot_boxplot (bool): If True, generates and saves a boxplot for each
            numeric column. Defaults to True.
        plot_histogram (bool): If True, generates and saves a histogram + KDE +
            normal curve for each numeric column. Defaults to True.
        plot_qq (bool): If True, generates and saves a Q-Q plot for each
            numeric column. Defaults to True.
        plot_bar (bool): If True, generates and saves a bar chart for each
            non-numeric column. Defaults to True.
        bar_top_n (int): Number of top categories shown individually in the
            bar chart; the rest are grouped as "Other". Defaults to 5.
        subfolder (str): Sub-directory under ``outputs/descriptive_analysis/`` where all outputs
            (JSON and plots) are saved. Defaults to ``"train"``.

    Returns:
        pd.DataFrame: DataFrame with columns ['variable', 'dtype'] and a
                        total variable count logged.

    Raises:
        RuntimeError: If data has not been loaded yet.
    """
    if df is None:
        raise RuntimeError("No data available. Please load data before calling get_descriptive_statistics.")

    try:
        QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        summary = df.dtypes.reset_index()
        summary.columns = ["variable", "dtype"]
        summary["dtype"] = summary["dtype"].astype(str)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        numeric_stats = {}
        for col in numeric_cols:
            s = df[col]
            stats = {
                "count": int(s.count()),
                "null_count": int(s.isna().sum()),
                "null_ratio": round(s.isna().sum() / len(s), 6),
                "mean": round(s.mean(), 6),
                "median": round(s.median(), 6),
                "std": round(s.std(), 6),
                "variance": round(s.var(), 6),
                "min": round(s.min(), 6),
                "max": round(s.max(), 6),
                "IQR": round(s.quantile(0.75) - s.quantile(0.25), 6),
                "skewness": round(s.skew(), 6),
                "skew_interpretation": (
                    "highly negatively skewed" if s.skew() < -1 else
                    "moderately negatively skewed" if s.skew() < -0.5 else
                    "approximately symmetric" if s.skew() <= 0.5 else
                    "moderately positively skewed" if s.skew() <= 1 else
                    "highly positively skewed"
                ),
                "kurtosis": round(s.kurt(), 6),
                "kurtosis_interpretation": (
                    "platykurtic (flat, light-tailed)" if s.kurt() < -1 else
                    "mesokurtic (normal-like)" if s.kurt() <= 1 else
                    "leptokurtic (peaked, heavy-tailed)"
                ),
            }
            for q in QUANTILES:
                label = f"P{int(q * 100):02d}"
                stats[label] = round(s.quantile(q), 6)
            stats["normality_tests"] = {
                "shapiro_wilk": shapiro_wilk(s),
                "dagostino_pearson": dagostino_pearson(s),
                "kolmogorov_smirnov": kolmogorov_smirnov(s),
            }
            stats["percentile_distribution"] = percentile_distribution(s, QUANTILES)
            numeric_stats[col] = stats

        categorical_stats = {}
        for col in non_numeric_cols:
            s = df[col]
            value_counts = s.value_counts(dropna=False)
            mode_val = s.mode()
            categorical_stats[col] = {
                "count": int(s.count()),
                "null_count": int(s.isna().sum()),
                "null_ratio": round(s.isna().sum() / len(s), 6),
                "unique_count": int(s.nunique()),
                "mode": str(mode_val.iloc[0]) if not mode_val.empty else None,
                "mode_frequency": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "mode_ratio": round(value_counts.iloc[0] / len(s), 6) if not value_counts.empty else 0.0,
                "frequency": [
                    {
                        "value": str(val),
                        "count": int(cnt),
                        "ratio": round(cnt / len(s), 6),
                    }
                    for val, cnt in value_counts.items()
                ],
            }

        variable_summary = {}
        for row in summary.to_dict(orient="records"):
            col = row["variable"]
            entry = {"dtype": row["dtype"]}
            if col in numeric_stats:
                entry["statistics"] = numeric_stats[col]
            elif col in categorical_stats:
                entry["statistics"] = categorical_stats[col]
            variable_summary[col] = entry

        n_vars = len(summary)
        logger.info(f"Total variables: {n_vars}")
        logger.info(f"\n{summary.to_string(index=False)}")

        save_json({"features": variable_summary}, f"outputs/descriptive_analysis/{subfolder}/summary_stats.json")
        logger.info(f"Variable summary saved to outputs/descriptive_analysis/{subfolder}/summary_stats.json.")

        if plot_boxplot:
            plot_boxplots(df, output_dir=f"outputs/descriptive_analysis/{subfolder}/boxplots")
        if plot_histogram:
            plot_histograms(df, output_dir=f"outputs/descriptive_analysis/{subfolder}/histograms")
        if plot_qq:
            plot_qq_plots(df, output_dir=f"outputs/descriptive_analysis/{subfolder}/qq_plots")
        if plot_bar:
            plot_bar_charts(df, top_n=bar_top_n, output_dir=f"outputs/descriptive_analysis/{subfolder}/bar_charts")

        return summary
    except Exception as e:
        error_message = f"Error computing variable summary: {e}"
        logger.error(error_message)
        raise Exception(error_message) from e

if __name__ == "__main__":
    input_data_path = "inputs/sample/uci_credit_card_dataset.csv"
    input_schema_path = "inputs/sample/datatypes.json"
    data_schema = load_data_schema(input_schema_path).get('column_dtypes')
    X_train = pd.read_csv(input_data_path, dtype=data_schema)   
    _ = get_descriptive_statistics(df=X_train)
>>>>>>> d438e88be478c4c02e1e05ccc3e82a42615bffa9
