# ML Logistic Regression Module Documentation

This module implements Logistic Regression using Statsmodels. It includes detailed logging, model summary extraction, and saving results to a model directory. The class `LogisticRegression` provides methods for fitting the model, making predictions, and extracting summaries with interpretations of coefficients and fit quality metrics.

---

## Table of Contents
1. [Function Call Hierarchy](#function-call-hierarchy)
2. [Class Overview](#class-overview)
3. [Function Detailed Documentation](#function-detailed-documentation)
4. [Usage Examples](#usage-examples)
5. [References](#references)

---

## Function Call Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Main Execution Flow                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  fit(X, y)  [PUBLIC METHOD]                                                 │
│  ├─ sm.add_constant(X) - Add intercept term                                 │
│  ├─ sm.Logit(y, X_with_const) - Initialize model                            │
│  ├─ self.model.fit() - Train model                                          │
│  ├─┬─────────────────────────────────────────────────┐                      │
│  │ │ Call: _extract_summary()  ◄───────────┐         │                      │
│  │ └─────────────────────────────────────────────────┘                      │
│  │         │                                          │                     │
│  │         ├─> _get_fit_quality()                     │                     │
│  │         │       ├─> _interpret_pseudo_rsquared()   │                     │
│  │         │       └─> _interpret_p_value()           │ Saves JSON/CSV      │
│  │         │                                          │                     │
│  │         ├─> _generate_linear_equation()            │                     │
│  │         │                                          │                     │
│  │         ├─> _format_linear_equation_for_logging()  │                     │
│  │         │                                          │                     │
│  │         ├─> _get_coeff_details()                   │                     │
│  │         │       ├─> _interpret_odds_ratio()        │                     │
│  │         │       ├─> _interpret_p_value()           │                     │
│  │         │       └─> _format_odds_ratio_summary()   │                     │
│  │         │                                          │                     │
│  │         ├─> _get_marginal_effects()                │                     │
│  │         │       └─> _interpret_marg_effect()       │                     │
│  │         │                                          │                     │
│  │         ├─> _format_marginal_effect()              │                     │
│  │         │       └─> _interpret_marg_effect()       │                     │
│  │         │                                          │                     │
│  │         └─> Returns summary_dict                   │                     │
│  │                                                    │                     │
│  └────────────────────────────────────────────────────┘                     │
│                                                                             │
│  ├─┬─────────────────────────────────────────────────┐                      │
│  │ │ Call: _create_model_details_dataframe()         │                      │
│  │ └─────────────────────────────────────────────────┘                      │
│  │         │                                         │                      │
│  │         ├─> _get_marginal_effects()               │                      │
│  │         │       └─> _interpret_marg_effect()      │                      │
│  │         │                                         │                      │
│  │         ├─> _interpret_odds_ratio()               │                      │
│  │         │                                         │                      │
│  │         ├─> _interpret_marg_effect()              │ Saves CSV            │
│  │         │                                         │                      │
│  │         └─> Returns pd.DataFrame                  │                      │
│  │                                                   │                      │
│  └───────────────────────────────────────────────────┘                      │
│                                                                             │
│  └─ io_save_json() - Save summary as JSON                                   │
│  └─ io_save_dataframe_as_csv() - Save details as CSV                        │
│  └─ Return self.model_fit                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                          Returns LogitResults object
                          ├─ model_details.json saved
                          └─ model_details.csv saved


┌──────────────────────────────────────────────────────────────────────────────┐
│  predict(X, include_const=True)  [PUBLIC METHOD - INDEPENDENT]               │
│  ├─ Check if model_fit exists                                               │
│  ├─ sm.add_constant(X) if include_const=True                                │
│  └─ Return self.model_fit.predict(X_pred)                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Helper Interpretation Functions (Called by other methods)                   │
│  ├─ _interpret_p_value()           - Classify statistical significance       │
│  ├─ _interpret_pseudo_rsquared()   - Assess model fit quality               │
│  ├─ _interpret_odds_ratio()        - Classify effect direction/magnitude    │
│  └─ _interpret_marg_effect()       - Classify probability change            │
└──────────────────────────────────────────────────────────────────────────────┘

```

---

## Class Overview

### LogisticRegression

Main class for logistic regression model implementation using statsmodels.

**Initialization Parameters:**
- `disp` (bool, default=False): Display convergence messages during model fitting
- `method` (str, default="bfgs"): Optimization method for model fitting
- `class_labels` (tuple, default=("Good (0)", "Default (1)")): Labels for binary classes

**Instance Attributes:**
- `self.model`: The statsmodels Logit model object
- `self.model_fit`: Fitted model results (LogitResults object)
- `self.disp`: Display setting for convergence messages
- `self.method`: Optimization method
- `self.class_labels`: Binary class labels

---

## Function Detailed Documentation

### 1. `__init__(disp=False, method="bfgs", class_labels=("Good (0)", "Default (1)"))`

**Purpose:** Initialize LogisticRegression instance with configuration settings.

**Signature:**
```python
def __init__(self, disp=False, method="bfgs", class_labels=("Good (0)", "Default (1)"))
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disp` | bool | False | Set True to print convergence messages during model fitting |
| `method` | str | "bfgs" | Optimization method (alternatives: newton, lbfgs, powell, cg, ncg) |
| `class_labels` | tuple | ("Good (0)", "Default (1)") | Labels for binary target classes |

**Returns:**
- None (constructor)

**Behavior:**
- Initializes instance attributes
- Logs configuration information

**Example:**
```python
log_reg = LogisticRegression(disp=False, method="bfgs")
```

---

### 2. `fit(X, y)`

**Purpose:** Train logistic regression model using statsmodels Logit. Fits the model, extracts summaries, and saves results to output directory.

**Signature:**
```python
def fit(self, X, y)
```

**Parameters:**
| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `X` | pd.DataFrame | (n_samples, n_features) | Features matrix with independent variables |
| `y` | pd.Series | (n_samples,) | Target series with binary dependent variable (0 or 1) |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| result | statsmodels.discrete.discrete_model.LogitResults | Fitted model statistics and coefficients |

**Behavior:**
- Saves model summary as JSON: `./outputs/model/model_details.json`
- Saves model details as CSV: `./outputs/model/model_details.csv`
- Logs comprehensive model training information
- Stores fitted model in `self.model_fit`

**Process Flow:**
1. Adds constant term for intercept
2. Initializes Logit model
3. Fits model with specified optimization method
4. Calls `_extract_summary()` to compile results
5. Calls `_create_model_details_dataframe()` to create detailed table
6. Saves results to output files

**Example:**
```python
X_train = df.drop(columns=['TARGET'])
y_train = df['TARGET']
log_reg = LogisticRegression()
result = log_reg.fit(X_train, y_train)
```

---

### 3. `_extract_summary()`

**Purpose:** Extract and compile all model summary data into a structured dictionary combining model info, fit quality metrics, coefficients, and marginal effects.

**Signature:**
```python
def _extract_summary(self)
```

**Parameters:**
- None

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| summary_dict | dict | Comprehensive summary with keys: `modelInfo`, `fitQuality`, `linearEquation`, `coefficientsDetails`, `marginalEffects` |

**Return Structure:**
```python
{
    'modelInfo': {
        'dependentVar': str,           # Target variable name
        'numObs': int,                 # Total observations
        'numOfTrue': int,              # Positive class count
        'numOfFalse': int,             # Negative class count
        'percentOfTrue': str,          # Positive class percentage
        'percentOfFalse': str,         # Negative class percentage
        'dfResid': int,                # Degrees of freedom (residuals)
        'dfModel': int                 # Degrees of freedom (model)
    },
    'fitQuality': {                    # See _get_fit_quality() for details
        'llf': {...},                  # Log-Likelihood Function
        'aic': {...},                  # Akaike Information Criterion
        'bic': {...},                  # Bayesian Information Criterion
        'log_likelihood': {...},       # Log-Likelihood
        'pseudo_r_squared': {...},     # McFadden's Pseudo R-squared
        'llr_pvalue': {...}            # Likelihood Ratio Test p-value
    },
    'linearEquation': {                # See _generate_linear_equation() for details
        'equation_full_precision': str,
        'equation_rounded': str,
        'description': str
    },
    'coefficientsDetails': {...},      # See _get_coeff_details() for details
    'marginalEffects': {               # See _get_marginal_effects() for details
        'effectsSummary': {...},
        'statistics': {...}
    }
}
```

**Behavior:**
- Logs full model summary
- Logs fit quality metrics
- Logs linear equation
- Logs odds ratio interpretations
- Logs marginal effects

**Called By:**
- `fit()` - Main entry point for model training

**Example:**
```python
summary = log_reg._extract_summary()
print(summary['fitQuality']['pseudo_r_squared'])
```

---

### 4. `_get_fit_quality(model_fit)`

**Purpose:** Extract and interpret model fit quality metrics (AIC, BIC, pseudo R-squared, log-likelihood, LLR p-value).

**Signature:**
```python
def _get_fit_quality(self, model_fit)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_fit` | statsmodels.discrete.discrete_model.LogitResults | Fitted logistic regression model |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| fit_quality_summary | dict | Dictionary with metrics: `llf`, `aic`, `bic`, `log_likelihood`, `pseudo_r_squared`, `llr_pvalue` |

**Return Structure (per metric):**
```python
{
    'value': float,              # Numeric metric value
    'metric': str,               # Metric name
    'description': str,          # What metric measures
    'range': str,                # Acceptable range
    'interpretation': str        # Current value interpretation
}
```

**Metric Interpretations:**

| Metric | Value Type | Interpretation Range |
|--------|-----------|----------------------|
| Pseudo R-squared | 0.0-1.0 | 0.0-0.1: Poor, 0.1-0.3: Acceptable, 0.3-0.5: Good, 0.5-1.0: Excellent |
| LLR p-value | 0.0-1.0 | <0.001: Highly Sig., <0.01: Very Sig., <0.05: Sig., ≥0.05: Not Sig. |
| AIC/BIC | Relative | Lower is better (for model comparison) |

**Called By:**
- `_extract_summary()` - Gathers fit quality metrics

**Behavior:**
- Extracts statistical metrics (AIC, BIC, pseudo R²)
- Computes log-likelihood and likelihood ratio tests
- Interprets each metric with practical guidance

**Helper Methods Called:**
- `_interpret_pseudo_rsquared()` - Assess R-squared quality
- `_interpret_p_value()` - Interpret p-value significance

**Example:**
```python
fit_quality = log_reg._get_fit_quality(log_reg.model_fit)
print(fit_quality['pseudo_r_squared']['interpretation'])
```

---

### 5. `_interpret_p_value(p_value)`

**Purpose:** Interpret statistical p-value and classify into significance levels following standard conventions.

**Signature:**
```python
def _interpret_p_value(self, p_value)
```

**Parameters:**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `p_value` | float | 0.0-1.0 | P-value from statistical test |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| interpretation | dict | Contains `text`, `textForCoefficient`, `is_significant` |

**Return Structure:**
```python
{
    'text': str,                          # Significance level classification
    'textForCoefficient': str,            # Effect description for coefficients
    'is_significant': str                 # 'Yes' if p<0.05, else 'No'
}
```

**Significance Thresholds:**
| P-value | Classification | Effect |
|---------|-----------------|--------|
| < 0.001 | Highly Significant | Strong effect |
| < 0.01 | Very Significant | Strong effect |
| < 0.05 | Significant | Moderate effect |
| ≥ 0.05 | Not Significant | No meaningful effect |

**Called By:**
- `_get_fit_quality()` - Interpret LLR p-value
- `_get_coeff_details()` - Interpret coefficient p-values
- `_get_marginal_effects()` - Interpret marginal effect p-values

**Example:**
```python
interp = log_reg._interpret_p_value(0.003)
print(interp['text'])  # Output: "Highly Significant"
```

---

### 6. `_interpret_pseudo_rsquared(pseudo_rsquared)`

**Purpose:** Interpret McFadden's pseudo R-squared value and provide model fit quality assessment.

**Signature:**
```python
def _interpret_pseudo_rsquared(self, pseudo_rsquared)
```

**Parameters:**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `pseudo_rsquared` | float | 0.0-1.0 | McFadden's pseudo R-squared value |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| interpretation | str | Human-readable quality assessment |

**Quality Assessment Scale:**

| Pseudo R² Range | Assessment |
|-----------------|-----------|
| > 0.5 | Excellent - Model explains substantial variation |
| 0.3 - 0.5 | Good - Model explains moderate variation |
| 0.1 - 0.3 | Acceptable - Model explains some variation |
| < 0.1 | Poor - Model explains little variation |

**Called By:**
- `_get_fit_quality()` - Assess model fit quality

**Example:**
```python
assessment = log_reg._interpret_pseudo_rsquared(0.35)
print(assessment)  # Output: "Good - Model explains moderate variation in target"
```

---

### 7. `_interpret_odds_ratio(odds_ratio)`

**Purpose:** Interpret odds ratio (exp(coefficient)) and classify effect direction and magnitude. Odds ratio represents multiplicative change in odds for unit increase in feature.

**Signature:**
```python
def _interpret_odds_ratio(self, odds_ratio)
```

**Parameters:**
| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `odds_ratio` | float | > 0 | Exponentiated coefficient value |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| interpretation | dict | Contains direction, magnitude, percentage change, and interpretation |

**Return Structure:**
```python
{
    'odds_ratio': float,             # Original input value
    'direction': str,                # 'INCREASES' or 'DECREASES'
    'direction_text': str,           # lowercase version
    'magnitude': str,                # 'LARGE', 'MODERATE', 'SMALL', 'VERY SMALL'
    'magnitude_text': str,           # lowercase version
    'percentage_change': float,      # Percent change in odds for unit increase
    'interpretation': str            # Human-readable interpretation
}
```

**Magnitude Classification (Percentage Change):**
| % Change | Magnitude |
|----------|-----------|
| ≥ 50% | LARGE |
| ≥ 20% | MODERATE |
| ≥ 5% | SMALL |
| < 5% | VERY SMALL |

**Odds Ratio Examples:**
- OR = 1.5 → Unit increase increases odds by 50%
- OR = 0.8 → Unit increase decreases odds by 20%
- OR = 1.0 → Unit increase has no effect

**Called By:**
- `_get_coeff_details()` - Interpret coefficient effects
- `_create_model_details_dataframe()` - Include in detailed results

**Behavior:**
- Classifies effect direction (increases/decreases)
- Determines magnitude based on percentage change
- Returns comprehensive interpretation dictionary

**Example:**
```python
interp = log_reg._interpret_odds_ratio(1.35)
print(interp['interpretation'])  # "Unit increase increases odds by 35.00%"
```

---

### 8. `_interpret_marg_effect(marginal_effect, abs_effect)`

**Purpose:** Interpret marginal effect (change in predicted probability for unit increase) and classify effect direction and magnitude.

**Signature:**
```python
def _interpret_marg_effect(self, marginal_effect, abs_effect)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `marginal_effect` | float | Change in predicted probability for unit increase |
| `abs_effect` | float | Absolute value of marginal effect |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| interpretation | dict | Contains direction, magnitude, and probability change |

**Return Structure:**
```python
{
    'direction': str,                    # 'INCREASES' or 'DECREASES'
    'effect_interpretation': str,        # 'Positive Effect' or 'Negative Effect'
    'magnitude': str,                    # 'LARGE', 'MODERATE', 'SMALL', 'VERY SMALL'
    'direction_text': str,               # lowercase direction
    'magnitude_text': str,               # lowercase magnitude
    'probability_change_percent': float  # Abs effect * 100
}
```

**Magnitude Classification (Absolute Effect):**
| Abs Effect | Magnitude |
|-----------|-----------|
| ≥ 0.10 | LARGE |
| ≥ 0.05 | MODERATE |
| ≥ 0.01 | SMALL |
| < 0.01 | VERY SMALL |

**Example Interpretation:**
- Credit Score +1 → Default probability -0.25% (small negative effect)
- Age +1 year → Default probability +0.05% (small positive effect)

**Called By:**
- `_get_marginal_effects()` - Interpret each feature's effect
- `_format_marginal_effect()` - Format for display
- `_create_model_details_dataframe()` - Include in details

**Example:**
```python
interp = log_reg._interpret_marg_effect(-0.0025, 0.0025)
print(interp['direction'])  # "DECREASES"
```

---

### 9. `_get_marginal_effects()`

**Purpose:** Extract marginal effects of each feature and compile into structured JSON format. Marginal effects show change in predicted probability for 1-unit increase in feature (evaluated at mean values).

**Signature:**
```python
def _get_marginal_effects(self)
```

**Parameters:**
- None

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| marginal_effects_json | dict | Comprehensive marginal effects data |

**Return Structure:**
```python
{
    'marginalEffects': object,           # Raw statsmodels marginal effects object
    'marginalEffectsSummary': object,    # Summary table from statsmodels
    'effectsSummary': {                  # Per-feature interpretation
        'feature_name': {
            'marginalEffect': float,
            'probabilityChange': float,
            'probabilityChangePercent': float,
            'direction': str,
            'magnitude': str,
            'pValue': float,
            'significant': str,
            'significanceLevel': str,
            'ciLower': float,
            'ciUpper': float,
            'standardError': float,
            'tStatistic': float
        }
    },
    'effectsSorted': [                   # List sorted by absolute effect (descending)
        {
            'feature': str,
            'marginalEffect': float,
            'absEffect': float,
            'stdErr': float,
            'tStat': float,
            'pValue': float,
            'ciLower': float,
            'ciUpper': float
        }
    ],
    'statistics': {
        'totalFeatures': int,            # Total number of features
        'significantFeatures': int,      # Features with p<0.05
        'insignificantFeatures': int,    # Features with p≥0.05
        'averageAbsoluteEffect': float   # Mean abs marginal effect
    }
}
```

**Called By:**
- `_extract_summary()` - Compile summary data
- `_format_marginal_effect()` - Format for display
- `_create_model_details_dataframe()` - Include in detailed results

**Helper Methods Called:**
- `_interpret_marg_effect()` - Interpret each effect

**Example:**
```python
marg_effects = log_reg._get_marginal_effects()
print(marg_effects['statistics']['significantFeatures'])
```

---

### 10. `_format_marginal_effect(marginal_effects_json=None)`

**Purpose:** Format marginal effects interpretation into human-readable string for logging display.

**Signature:**
```python
def _format_marginal_effect(self, marginal_effects_json=None)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `marginal_effects_json` | dict | None | Pre-computed marginal effects data (optional) |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| formatted_result | dict | Contains formatted string and original data |

**Return Structure:**
```python
{
    'formattedString': str,              # Human-readable formatted output
    'marginalEffectsData': dict          # Original marginal effects data
}
```

**Formatted Output Includes:**
- Definition of marginal effects
- Detailed feature interpretations (sorted by impact)
- Summary statistics
- Strongest impact feature
- Average absolute effect

**Called By:**
- `_extract_summary()` - Format for logging
- `_interpret_marginal_effect()` - Format for display

**Helper Methods Called:**
- `_get_marginal_effects()` - Extract data if not provided
- `_interpret_p_value()` - Interpret significance
- `_interpret_marg_effect()` - Interpret effects

**Example:**
```python
formatted = log_reg._format_marginal_effect()
print(formatted['formattedString'])
```

---

### 11. `_get_coeff_details()`

**Purpose:** Extract coefficient details including significance, confidence intervals, and odds ratio interpretations for each feature.

**Signature:**
```python
def _get_coeff_details(self)
```

**Parameters:**
- None

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| coeff_summary | dict | Dictionary with feature names as keys |

**Return Structure:**
```python
{
    'feature_name': {
        'coefficient': float,                    # Original coefficient value
        'pValue': float,                         # P-value from t-test
        'significantOfCoefficient': str,         # 'Yes' or 'No'
        'interpretationOfCoefficient': str,      # Significance and effect description
        'ciLower': float,                        # Lower 95% confidence interval
        'ciUpper': float,                        # Upper 95% confidence interval
        'oddsRatio': float,                      # exp(coefficient)
        'oddsRatioInterpretation': {
            'odds_ratio': float,
            'direction': str,
            'direction_text': str,
            'magnitude': str,
            'magnitude_text': str,
            'percentage_change': float,
            'interpretation': str
        }
    },
    'const': {...}                              # Intercept term
}
```

**Called By:**
- `_extract_summary()` - Compile summary data
- `_create_model_details_dataframe()` - Include in details

**Helper Methods Called:**
- `_interpret_p_value()` - Interpret p-values
- `_interpret_odds_ratio()` - Interpret odds ratios
- `_format_odds_ratio_summary()` - Format for display

**Behavior:**
- Extracts coefficients, p-values, and confidence intervals
- Computes odds ratios for each feature
- Logs formatted odds ratio interpretations

**Example:**
```python
coeff_details = log_reg._get_coeff_details()
print(coeff_details['credit_limit']['oddsRatio'])
```

---

### 12. `_format_odds_ratio_summary(coeff_summary)`

**Purpose:** Format odds ratio summary for each coefficient into comprehensive human-readable string for logging.

**Signature:**
```python
def _format_odds_ratio_summary(self, coeff_summary)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `coeff_summary` | dict | Dictionary from `_get_coeff_details()` output |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| formatted_string | str | Multi-line formatted output for logging |

**Formatted Output Includes:**
- Definition of odds ratios
- Interpretation examples
- Feature-by-feature analysis (sorted by impact)
- Summary statistics:
  - Total features
  - Features increasing/decreasing odds
  - Statistically significant features
  - Strongest positive/negative effects

**Called By:**
- `_get_coeff_details()` - Format results for logging

**Example:**
```python
formatted = log_reg._format_odds_ratio_summary(coeff_details)
print(formatted)
```

---

### 13. `_generate_linear_equation()`

**Purpose:** Generate linear regression equation string from coefficients in logit form: z = β₀ + β₁*x₁ + β₂*x₂ + ...

**Signature:**
```python
def _generate_linear_equation(self)
```

**Parameters:**
- None

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| equation_dict | dict | Contains full precision and rounded equation strings |

**Return Structure:**
```python
{
    'equation_full_precision': str,      # Equation with full coefficient precision
    'equation_rounded': str,             # Equation with rounded coefficients (4 decimals)
    'description': str                   # Description of equation form
}
```

**Example Output:**
```
equation_full_precision: "z = 0.123456 + 0.234567*feature_1 - 0.345678*feature_2"
equation_rounded: "z = 0.1235 + 0.2346*feature_1 - 0.3457*feature_2"
description: "Logit linear equation (z). Probability = exp(z) / (1 + exp(z))"
```

**Called By:**
- `_extract_summary()` - Include in summary dict

**Helper Methods Called:**
- `_format_linear_equation_for_logging()` - Format for display

**Example:**
```python
equation = log_reg._generate_linear_equation()
print(equation['equation_rounded'])
```

---

### 14. `_format_linear_equation_for_logging()`

**Purpose:** Format the linear equation into human-readable string for logging display.

**Signature:**
```python
def _format_linear_equation_for_logging(self)
```

**Parameters:**
- None

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| formatted_string | str | Multi-line formatted equation for display |

**Formatted Output Includes:**
- Linear equation formula with intercept
- Feature coefficients sorted by absolute value
- Interpretation guide:
  - What z represents (log-odds)
  - Probability conversion formula
  - Positive/negative coefficient meanings
  - Magnitude interpretation

**Called By:**
- `_extract_summary()` - Format for logging

**Example:**
```python
formatted_eq = log_reg._format_linear_equation_for_logging()
print(formatted_eq)
```

---

### 15. `predict(X, include_const=True)`

**Purpose:** Generate predictions from the fitted logistic regression model. Returns predicted probabilities for positive class.

**Signature:**
```python
def predict(self, X, include_const=True)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | pd.DataFrame or np.ndarray | - | Input features for prediction (must match training features) |
| `include_const` | bool | True | Whether to add constant term to features |

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| predictions | np.ndarray | Array of predicted probabilities (0.0 to 1.0) |

**Shape:**
- Input: (n_samples, n_features)
- Output: (n_samples,)

**Raises:**
- ValueError: If model not fitted yet

**Behavior:**
- Adds constant term if include_const is True
- Generates predictions using fitted model
- Logs prediction information (sample count, mean probability)

**Example:**
```python
X_test = test_df.drop(columns=['TARGET'])
predictions = log_reg.predict(X_test)
print(predictions.mean())  # Mean predicted probability
```

---

### 16. `_create_model_details_dataframe()`

**Purpose:** Create comprehensive pandas DataFrame with all model coefficient information for analysis and reporting.

**Signature:**
```python
def _create_model_details_dataframe(self)
```

**Parameters:**
- None

**Returns:**
| Return Value | Type | Description |
|--------------|------|-------------|
| detailed_df | pd.DataFrame | DataFrame with detailed model information |

**DataFrame Columns:**

| Column | Type | Description |
|--------|------|-------------|
| Variable | str | Feature name |
| Coefficient | float | Model coefficient value |
| Std_Error | float | Standard error of coefficient |
| T_Statistic | float | T-test statistic (coeff / std_err) |
| P_Value | float | P-value from t-test |
| Significant | str | 'Yes' if p<0.05, else 'No' |
| CI_Lower_95 | float | Lower 95% confidence interval |
| CI_Upper_95 | float | Upper 95% confidence interval |
| Odds_Ratio | float | Exponentiated coefficient |
| Odds_Change_Percent | float | Percentage change in odds for unit increase |
| Marginal_Effect | float | Change in predicted probability for unit increase |
| Magnitude | str | Effect size magnitude |
| Direction | str | Direction of effect |

**DataFrame Structure:**
- First row: Constant term
- Remaining rows: Features sorted by absolute coefficient value (descending)

**Called By:**
- `fit()` - Create and save to CSV

**Helper Methods Called:**
- `_get_marginal_effects()` - Extract marginal effects
- `_interpret_odds_ratio()` - Interpret odds ratios
- `_interpret_marg_effect()` - Interpret effects

**Behavior:**
- Compiles all coefficient data into DataFrame format
- Sorts features by absolute coefficient value
- Logs DataFrame creation information

**Example:**
```python
details_df = log_reg._create_model_details_dataframe()
print(details_df.to_string())
details_df.to_csv('coefficients.csv', index=False)
```

---

## Usage Examples

### Example 1: Basic Model Training and Prediction

```python
import pandas as pd
from ml_logistic import LogisticRegression
from factor_analysis import prepare_factor_analysis_data

# Load data
df = pd.read_csv('uci_credit_card_dataset.csv')

# Prepare data for factor analysis
df_prepared, metadata = prepare_factor_analysis_data(
    df=df,
    target_variable="TARGET"
)

# Split features and target
X_train = df_prepared.drop(columns=["TARGET"])
y_train = df_prepared["TARGET"]

# Create and fit model
log_reg = LogisticRegression(disp=False, method="bfgs")
result = log_reg.fit(X_train, y_train)

# Make predictions
predictions = log_reg.predict(X_train)
print(f"Average predicted probability: {predictions.mean():.4f}")
```

### Example 2: Accessing Model Results

```python
# Access coefficients
coeff_details = log_reg._get_coeff_details()
for feature, details in coeff_details.items():
    if feature != 'const':
        print(f"{feature}:")
        print(f"  Coefficient: {details['coefficient']:.4f}")
        print(f"  Odds Ratio: {details['oddsRatio']:.4f}")
        print(f"  Significant: {details['significantOfCoefficient']}")

# Access marginal effects
marg_effects = log_reg._get_marginal_effects()
print(f"Significant features: {marg_effects['statistics']['significantFeatures']}")

# Access fit quality
fit_quality = log_reg._get_fit_quality(log_reg.model_fit)
print(f"Pseudo R²: {fit_quality['pseudo_r_squared']['value']:.4f}")
print(f"Interpretation: {fit_quality['pseudo_r_squared']['interpretation']}")
```

### Example 3: Accessing Saved Results

```python
import json

# Load model summary JSON
with open('./outputs/model/model_details.json', 'r') as f:
    model_summary = json.load(f)

print("Model Info:")
print(f"  Observations: {model_summary['modelInfo']['numObs']}")
print(f"  Target Distribution: {model_summary['modelInfo']['percentOfTrue']}")

print("\nFit Quality:")
print(f"  AIC: {model_summary['fitQuality']['aic']['value']:.2f}")
print(f"  BIC: {model_summary['fitQuality']['bic']['value']:.2f}")

# Load model details CSV
details_df = pd.read_csv('./outputs/model/model_details.csv')
print(details_df.to_string())
```

### Example 4: Getting Model Summary Dictionary

```python
# Get comprehensive summary
summary = log_reg._extract_summary()

# Access different components
print("Model Information:")
print(f"  Dependent Variable: {summary['modelInfo']['dependentVar']}")
print(f"  Observations: {summary['modelInfo']['numObs']}")

print("\nLinear Equation:")
print(f"  {summary['linearEquation']['equation_rounded']}")

print("\nCoefficients Details:")
for feature, details in summary['coefficientsDetails'].items():
    print(f"  {feature}: {details['coefficient']:.4f}")
```

---

## Key Concepts

### Logistic Regression Equation

The logistic regression model predicts the probability of a binary outcome using:

**Logit Form (Linear Scale):**
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$$

**Probability Form (Probability Scale):**
$$P(Y=1|X) = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}$$

Where:
- $z$ = log-odds (logit value)
- $\beta_j$ = coefficient for feature $x_j$
- $P(Y=1|X)$ = predicted probability of positive class

### Odds Ratio Interpretation

**Odds Ratio = exp(β)** represents the multiplicative change in odds for a one-unit increase in the feature:

- **OR > 1**: Feature INCREASES odds of positive class
  - Example: OR=1.5 means 50% increase in odds
- **OR < 1**: Feature DECREASES odds of positive class
  - Example: OR=0.8 means 20% decrease in odds
- **OR = 1**: Feature has NO effect

### Marginal Effect Interpretation

**Marginal Effect** = change in predicted probability for a one-unit increase in feature (evaluated at mean values).

- Can range from -1.0 to +1.0
- Represents the most intuitive interpretation for non-statisticians
- Depends on the current probability level (non-linear effect)

### Model Fit Metrics

| Metric | Interpretation |
|--------|-----------------|
| **Pseudo R²** | Proportion of variation explained (0-1 scale) |
| **AIC/BIC** | Model comparison criterion (lower is better) |
| **LLR p-value** | Test if model is better than null model |
| **Coefficients p-value** | Statistical significance of each feature |

---

## References

### Statsmodels Documentation
- [Logit Model Overview](https://www.statsmodels.org/stable/examples/notebooks/generated/discrete_choice_overview.html)
- [Logit Fit Method](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html)
- [LogitResults](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.LogitResults.html)

### Related Modules
- `factor_analysis`: Data preparation for logistic regression
- `helper.io_operations`: Input/output utilities
- `logging_config.logger_config`: Logging configuration

### Optimization Methods
Available methods: `bfgs`, `newton`, `lbfgs`, `powell`, `cg`, `ncg`, `basinhopping`, `minimize`

---

## Notes

1. **Constant Term**: The model automatically adds a constant term for the intercept
2. **Binary Target**: Target variable must contain only 0 and 1 values
3. **Feature Scaling**: No automatic feature scaling (consider preprocessing if needed)
4. **Missing Values**: Remove NaN values before calling `fit()`
5. **Output Directory**: Ensure `./outputs/model/` directory exists or will be created
6. **Logging**: All detailed information is logged to `mlops/ml_logistic.log`

