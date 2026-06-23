"""
Accelera Consulting

Custom Sklearn Transformer for Binary Column Detection and Conversion
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from logging_config.logger_config import get_logger

logger_name = "mlops.sklearn_wrapper_binary"
logger_file_name = "sklearn_wrapper_binary.log"
logger = get_logger(logger_name, logger_file_name)


class BinaryColumnConverter(BaseEstimator, TransformerMixin):
    """
    Custom transformer that detects binary columns (containing only 0 and 1,
    optionally with nulls) and converts their values to 'YES' / 'NO' strings
    (object dtype).

    During fit:
    - Scans all columns and identifies those whose non-null unique values are
      exactly {0, 1} (or subsets thereof, i.e. only 0 or only 1 are also
      considered binary).
    - Stores the list of binary columns in ``binary_columns_``.

    During transform:
    - Replaces 1  → 'YES'
    - Replaces 0  → 'NO'
    - Leaves NaN/None values unchanged.
    - Casts the resulting column to ``object`` dtype.

    Parameters
    ----------
    None

    Attributes
    ----------
    binary_columns_ : list of str
        Column names identified as binary during fit.

    binary_columns_details_ : dict
        Detailed information about each binary column detected during fit, plus
        a ``'_summary'`` key with aggregate statistics.

        Per-column keys:

        - ``'original_dtype'`` (*str*) — pandas dtype of the column before any
          transformation (e.g. ``'float64'``, ``'int64'``, ``'object'``).
        - ``'null_count'`` (*int*) — number of null/NaN values in the column.
        - ``'null_pct'`` (*float*) — percentage of null values (0–100, rounded
          to 2 decimal places).
        - ``'unique_non_null_values'`` (*list*) — sorted list of non-null unique
          values found (subset of ``[0, 1]``).
        - ``'row_count'`` (*int*) — total number of rows in the fitted data.
        - ``'action'`` (*str*) — always ``'convert 1→YES / 0→NO'``.

        Summary key ``'_summary'``:

        - ``'total_columns_scanned'`` (*int*)
        - ``'binary_columns_detected'`` (*int*)

        Example::

            {
                'flag_a': {
                    'original_dtype': 'float64',
                    'null_count': 1,
                    'null_pct': 20.0,
                    'unique_non_null_values': [0, 1],
                    'row_count': 5,
                    'action': 'convert 1->YES / 0->NO'
                },
                '_summary': {
                    'total_columns_scanned': 3,
                    'binary_columns_detected': 1
                }
            }

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn_wrapper_binary import BinaryColumnConverter
    >>>
    >>> df = pd.DataFrame({
    ...     'flag_a': [1, 0, 1, None, 0],
    ...     'flag_b': [0, 0, 1, 1, 0],
    ...     'amount': [100, 200, 300, 400, 500],
    ... })
    >>>
    >>> converter = BinaryColumnConverter()
    >>> converter.fit(df)
    >>> df_transformed = converter.transform(df)
    >>> print(df_transformed[['flag_a', 'flag_b']].dtypes)
    flag_a    object
    flag_b    object
    >>> print(df_transformed['flag_a'].tolist())
    ['YES', 'NO', 'YES', None, 'NO']
    """

    def __init__(self):
        self.binary_columns_ = []
        self.binary_columns_details_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Detect binary columns in *X*.

        A column is considered binary when its non-null unique values are a
        non-empty subset of {0, 1}.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to analyse.
        y : ignored

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        logger.info(f"Fitting BinaryColumnConverter on DataFrame with shape {X.shape}")

        binary_cols = []
        details = {}
        n_rows = len(X)

        for col in X.columns:
            non_null_values = X[col].dropna().unique()
            if len(non_null_values) == 0:
                # All-null column — skip
                continue
            unique_set = set(non_null_values)
            if unique_set <= {0, 1}:
                null_count = int(X[col].isna().sum())
                null_pct = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0.0
                binary_cols.append(col)
                details[col] = {
                    'original_dtype': str(X[col].dtype),
                    'null_count': null_count,
                    'null_pct': null_pct,
                    'unique_non_null_values': sorted(unique_set),
                    'row_count': n_rows,
                    'action': 'convert 1->YES / 0->NO',
                }
                logger.debug(
                    f"Column '{col}' identified as binary "
                    f"(dtype={X[col].dtype}, unique non-null values: {unique_set}, "
                    f"null count: {null_count} ({null_pct}%))"
                )

        details['_summary'] = {
            'total_columns_scanned': len(X.columns),
            'binary_columns_detected': len(binary_cols),
        }

        self.binary_columns_ = binary_cols
        self.binary_columns_details_ = details
        logger.info(
            f"Fit complete. Detected {len(self.binary_columns_)} binary column(s): "
            f"{self.binary_columns_}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert binary columns to 'YES' / 'NO' (object dtype).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Copy of *X* with binary columns converted to object dtype where
            1 → 'YES', 0 → 'NO', and NaN/None remain unchanged.

        Raises
        ------
        ValueError
            If *X* is not a pandas DataFrame or if the transformer has not
            been fitted yet.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        if self.binary_columns_ is None:
            raise ValueError(
                "Transformer has not been fitted yet. "
                "Call fit() before transform()."
            )

        logger.info(f"Transforming DataFrame with shape {X.shape}")
        X_transformed = X.copy()

        cols_present = [col for col in self.binary_columns_ if col in X_transformed.columns]

        if cols_present:
            logger.info(
                f"Converting {len(cols_present)} binary column(s) to YES/NO: "
                f"{cols_present}"
            )
            for col in cols_present:
                X_transformed[col] = X_transformed[col].map(
                    lambda x: "YES" if x == 1 else ("NO" if x == 0 else np.nan)
                ).astype(object)
                logger.debug(f"Column '{col}' converted: 1→'YES', 0→'NO', nulls preserved.")
        else:
            logger.info("No binary columns found in the current DataFrame.")

        return X_transformed
