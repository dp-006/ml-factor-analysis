## Automatic Binning and Weight of Evidence (WOE) Transformation

Weight of Evidence (WOE) binning is a statistical technique used in credit risk modeling and other binary classification problems to transform continuous numerical variables into categorical bins that maximize their predictive power for a binary target variable. This project implements a comprehensive automatic WOE binning pipeline with quality gates and detailed statistical interpretations.

### Pipeline Overview

The automatic WOE binning pipeline consists of 6 steps:

1. **Initial Bin Creation** - Creating quantile-based bins using pd.qcut
2. **WOE/IV Calculation** - Computing Weight of Evidence and Information Value statistics
3. **Small Bin Merging (Rule 1)** - Enforcing minimum bin size requirements
4. **Zero Distribution Merging (Rule 2)** - Ensuring good and bad cases in every bin
5. **Monotonicity Enforcement (Rules 3-4)** - Ensuring monotonic trends in bad rate and WOE
6. **Bin Count Reduction (Rule 5)** - Reducing final bins to meet business constraints

---

### Key Concepts

#### Weight of Evidence (WOE)

WOE measures the separation between good and bad cases in each bin. Higher absolute WOE values indicate stronger separation.

$$\text{WOE} = \ln\left(\frac{\text{Distribution of Good}}{\text{Distribution of Bad}}\right)$$

Where:
- **Distribution of Good** = (Good in Bin / Total Good) + Smoothing
- **Distribution of Bad** = (Bad in Bin / Total Bad) + Smoothing

#### Information Value (IV)

IV quantifies the total predictive power of a binned feature. It's the sum of each bin's contribution:

$$\text{IV} = \sum \left(\text{Distribution of Good} - \text{Distribution of Bad}\right) \times \text{WOE}$$

**IV Interpretation:**
- **IV < 0.02**: Not Predictive
- **0.02 ≤ IV < 0.1**: Weak Predictive Power
- **0.1 ≤ IV < 0.3**: Medium Predictive Power
- **0.3 ≤ IV < 0.5**: Strong Predictive Power
- **IV ≥ 0.5**: Suspiciously High (potential data leakage or overfitting)

---

### Step 1: Initial Bin Creation (`create_initial_bins`)

#### Purpose

This function creates initial quantile-based bins for a numeric feature using `pd.qcut`. This ensures roughly equal number of observations in each bin (equi-frequency binning).

#### Why Initial Binning Matters

- Provides a data-driven starting point for bin boundaries
- Roughly equal observation counts per bin enable better statistical stability
- Sets the stage for iterative refinement based on predictive power

#### Processing Steps

```
Numeric Feature Data
   ↓
1. Apply pd.qcut (Quantile-based Binning)
   - Creates bins with approximately equal frequencies
   - Produces pandas Interval objects as bin labels
   ↓
2. Fallback to pd.cut
   - If qcut fails due to duplicate bin edges, use equal-width binning
   ↓
Initial Bins Created (pandas Categorical)
```

#### Function Signature

```python
create_initial_bins(
    df: pd.DataFrame,
    feature: str,
    series: pd.Series = None,
    initial_bins: int = 20
) -> pd.Series
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input dataframe |
| `feature` | str | - | Numeric feature column name |
| `series` | Series or None | None | Optional series to use instead of df[feature] |
| `initial_bins` | int | 20 | Requested number of initial quantile-based bins |

#### Example Output

```
Feature: AGE
Initial Bins Created:
  - (18.0, 25.0]: 500 observations
  - (25.0, 35.0]: 495 observations
  - (35.0, 50.0]: 502 observations
  - (50.0, 65.0]: 498 observations
  - (65.0, 80.0]: 505 observations
```

---

### Step 2: WOE/IV Calculation (`calculate_woe_iv_table`)

#### Purpose

This function calculates the Weight of Evidence (WOE) and Information Value (IV) for each bin. It's the core analytical engine that quantifies the predictive power of each bin.

#### Function Signature

```python
calculate_woe_iv_table(
    df: pd.DataFrame,
    bin_col: str,
    target: str,
    eps: float = 0.5,
    output_dir: str = None
) -> (pd.DataFrame, dict)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input dataframe with bins and target |
| `bin_col` | str | - | Column name containing bin assignments |
| `target` | str | - | Binary target column (0=Good, 1=Bad) |
| `eps` | float | 0.5 | Smoothing constant to prevent division by zero |
| `output_dir` | str or None | None | Directory to save results as JSON and CSV |

#### Output Table Columns

| Column | Description |
|--------|-------------|
| `bin` | Bin interval (e.g., (18, 25]) |
| `human_readable_bin` | Readable bin label (e.g., from_18_to_25) |
| `total` | Total observations in bin |
| `good` | Count of good cases (target=0) |
| `bad` | Count of bad cases (target=1) |
| `bad_rate` | Bad case ratio in bin |
| `bin_pct` | Bin size as % of total observations |
| `good_dist` | Good distribution (with smoothing) |
| `bad_dist` | Bad distribution (with smoothing) |
| `good_dist_minus_bad_dist` | Difference between good and bad distributions |
| `odds_ratio` | Good Distribution / Bad Distribution |
| `woe` | Weight of Evidence (natural logarithm of odds ratio) |
| `woe_display` | WOE × 100 (for reporting purposes) |
| `iv` | Bin's contribution to total Information Value |

#### Example Output

| bin | human_readable_bin | total | good | bad | bad_rate | bin_pct | good_dist | bad_dist | good_dist_minus_bad_dist | odds_ratio | woe | woe_display | iv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (18, 25] | from_18_to_25 | 500 | 480 | 20 | 0.04 | 0.20 | 0.195 | 0.095 | 0.100 | 2.053 | 0.735 | 73.5 | 0.0735 |
| (25, 35] | from_25_to_35 | 495 | 450 | 45 | 0.09 | 0.19 | 0.183 | 0.214 | -0.031 | 0.854 | -0.161 | -16.1 | -0.0050 |
| (35, 50] | from_35_to_50 | 502 | 420 | 82 | 0.16 | 0.20 | 0.171 | 0.310 | -0.139 | 0.552 | -0.603 | -60.3 | -0.0840 |

#### Smoothing Explanation

Smoothing prevents infinite WOE values when a bin has zero good or zero bad cases:

$$\text{Good\_Dist}_{smoothed} = \frac{\text{Good}_{bin} + \epsilon}{\text{Good}_{total} + \epsilon \times n_{bins}}$$

Where:
- eps = 0.5 (smoothing constant)
- n_bins = number of bins

---

### Step 3: Iterative Bin Merging

The algorithm iteratively merges adjacent bins to satisfy binning rules while preserving predictive power.

#### Helper Functions

**`merge_intervals(intervals, i)`** - Merges two adjacent pandas Interval bins
```
Example:
Before: [(18, 25], (25, 35], (35, 50]]
Merge indices 0 and 1:
After: [(18, 35], (35, 50]]
```

**`assign_bins_from_intervals(x, intervals)`** - Reassigns observations to updated intervals
```
After each merge, observations are re-assigned to the new interval structure
without re-running pd.qcut
```

**`find_closest_neighbor(summary, idx, metric)`** - Finds the closest neighbor to a problematic bin
```
If bin A is problematic, find whether bin A-B or B-C are more similar
based on a metric (bad_rate, woe, etc.)
```

**`find_most_similar_adjacent_pair(summary, metric)`** - Finds the most similar adjacent bin pair
```
Scans entire bin table to find the pair with smallest metric difference
Used when one merge is needed for bin count reduction
```

**`find_monotonicity_violation_pair(summary, metric)`** - Finds bins violating monotonicity
```
Detects adjacent bins where metric trend is not consistently increasing or decreasing
Returns the pair with smallest metric difference to minimize information loss
```

---

### Step 4: Quality Rules

#### Rule 1: Minimum Bin Size

**Objective**: Ensure each bin contains sufficient observations

**Threshold**: `min_bin_pct = 0.05` (default 5% of total observations)

**Action**: If a bin has fewer observations than threshold, merge it with its closest neighbor based on bad_rate

---

#### Rule 2: Good and Bad Requirement

**Objective**: Ensure every bin contains both good and bad cases

**Threshold**: Good > 0 AND Bad > 0 in all bins

**Action**: If a bin has zero good or zero bad cases, merge it with its closest neighbor

---

#### Rule 3: Bad Rate Monotonicity

**Objective**: Ensure bad rate increases or decreases consistently across ordered bins

**Why**: Ensures logical progression of risk levels across bins

**Example (Monotonic Increasing):**
```
Bin 1: Bad Rate = 5%
Bin 2: Bad Rate = 10%
Bin 3: Bad Rate = 15%
Status: Passes
```

**Example (Violation):**
```
Bin 1: Bad Rate = 5%
Bin 2: Bad Rate = 15%
Bin 3: Bad Rate = 10%
Status: Fails (Bin 2 → Bin 3 violates trend)
Action: Merge Bin 2 and Bin 3
```

---

#### Rule 4: WOE Monotonicity

**Objective**: Ensure WOE (separating power) increases or decreases consistently

**Why**: Provides logical ordering of predictive power across bins

**Algorithm**:
1. Check both increasing and decreasing monotonicity directions
2. Select the direction with fewer violations (minimizes merges)
3. Merge the violating pair with smallest metric difference

**Important Distinction: `is_monotonic()` vs `find_monotonicity_violation_pair()`**

The algorithm uses two different functions for monotonicity, each serving a distinct purpose:

| Function | Purpose | Output | Usage |
|----------|---------|--------|-------|
| **`is_monotonic(series)`** | **Validation/Check**: Determines if a complete series is monotonic increasing OR decreasing | `True` / `False` | Final quality check to validate if the entire binned feature satisfies monotonicity |
| **`find_monotonicity_violation_pair(summary, metric)`** | **Action/Correction**: Identifies specific adjacent bin pairs that violate monotonicity and decides which pair to merge | Index (int) or None | During iterative binning process to fix violations by merging problematic pairs |

**Example with Bad Rates: [5%, 10%, 8%, 20%]**

1. **`is_monotonic()` check**:
   - Returns `False` (the series is NOT monotonic)
   - Tells us: *"There's a problem, monotonicity is violated"*

2. **`find_monotonicity_violation_pair()` action**:
   - Finds increasing monotonicity violations: [1] (10% → 8%)
   - Finds decreasing monotonicity violations: [0, 2] (5% → 10% and 8% → 20%)
   - Selects increasing direction (fewer violations)
   - Returns `1` (merge index)
   - Tells us: *"Merge pair at index (1,2) to fix the violation at 10% → 8%"*

---

#### Rule 5: Maximum Bin Count

**Objective**: Limit final bins to a business-acceptable number

**Threshold**: `max_final_bins = 10` (default)

**Action**: While bin count exceeds threshold, merge the most similar adjacent pair

**Metric**: Based on bad_rate (or woe if no monotonicity violations)

---

#### Rule 6: Information Value Range

**Objective**: Verify final IV is within acceptable range

**Range**: `0.02 ≤ IV ≤ 0.50`

**Actions**:
- If IV < 0.02: Feature marked as "Not Predictive" (status: WARNING)
- If IV > 0.50: Feature marked as "Suspiciously High" (status: WARNING)
- If 0.02 ≤ IV ≤ 0.50: Feature is acceptable (status: SUCCESS)

---

### Merging Strategy & Helper Functions

The core of the algorithm is **intelligent bin merging** based on different criteria depending on which rule is violated:

#### Key Merging Strategies

1. **`find_closest_neighbor(summary, idx, metric="bad_rate")`**
   - Used in Rules 1 and 2 when a specific bin is problematic
   - Compares metric value of left neighbor vs. right neighbor
   - Merges with the neighbor that is **closest in metric value** (minimizes information loss)
   - Example: If bin B (bad_rate=11%) is too small, and bin A (bad_rate=10%) and bin C (bad_rate=30%) are neighbors:
     - Distance to A: |11% - 10%| = 1%
     - Distance to C: |11% - 30%| = 19%
     - Decision: Merge with A (closer)

2. **`find_monotonicity_violation_pair(summary, metric="bad_rate")`**
   - Used in Rules 3 and 4 for monotonicity violations
   - Detects which direction (increasing or decreasing) has fewer violations
   - Among violations, finds pair with **smallest metric difference**
   - This greedy approach minimizes impact of each merge on predictive power
   - Example: If bad_rate = [5%, 10%, 8%, 20%]:
     - Increasing violations: [1] (10% → 8% is a decrease)
     - Decreasing violations: [0, 2] (5% → 10% and 8% → 20% are increases)
     - Choose increasing direction (fewer violations)
     - Merge pair at index 1 (10%, 8%) with difference = 2%

   **Detailed Step-by-Step Execution (Algorithm Walk-through)**
   
   The function performs the following steps:
   
   **Step 1: Extract Values**
   ```python
   values = summary[metric].values  # [5, 10, 8, 20]
   inc_violations = []
   dec_violations = []
   ```
   
   **Step 2: Loop Through Adjacent Pairs**
   
   The algorithm loops through `i = 0, 1, 2` (total pairs = len(values) - 1):
   
   | Iteration | i | values[i] | values[i+1] | Condition | Result | Reason |
   |-----------|---|-----------|-------------|-----------|--------|--------|
   | 1 | 0 | 5 | 10 | `5 < 10` | dec_violations=[0] | 5→10 increases, violates decreasing trend |
   | 2 | 1 | 10 | 8 | `10 > 8` | inc_violations=[1] | 10→8 decreases, violates increasing trend |
   | 3 | 2 | 8 | 20 | `8 < 20` | dec_violations=[0,2] | 8→20 increases, violates decreasing trend |
   
   **Step 3: Direction Selection**
   ```
   len(inc_violations) = 1  ← fewer violations
   len(dec_violations) = 2
   
   Decision: Select increasing direction (1 < 2)
   violations = [1]
   ```
   
   **Step 4: Find Closest Violating Pair**
   ```python
   # Among selected violations, find pair with smallest metric difference
   merge_idx = min(
       violations=[1],
       key=lambda i: abs(summary.loc[i, metric] - summary.loc[i + 1, metric])
   )
   
   # For i=1: |10 - 8| = 2  ← this is the pair to merge
   merge_idx = 1
   ```
   
   **Step 5: Return Merge Index**
   ```python
   return 1  # Merge bins at indices (1, 2): combine 10% and 8% into (~9%)
   ```
   
   **Integration with Main Loop**
   
   The function returns only the index. The actual merging happens in the calling function:
   ```python
   # In auto_woe_binning_numeric():
   for step in range(1, max_iter + 1):
       # ... assign bins and calculate WOE/IV ...
       
       if not is_monotonic(summary["bad_rate"]):
           merge_idx = find_monotonicity_violation_pair(summary, metric="bad_rate")
           intervals = merge_intervals(intervals, i=merge_idx)  # MERGE HERE
           
           # IMPORTANT: WOE/IV is recalculated in next iteration!
           continue  # ← Goes back to top of loop
   ```
   
   This ensures that after each merge, the WOE/IV statistics are recalculated with the new bin structure, potentially revealing new violations to address in subsequent iterations.

3. **`find_most_similar_adjacent_pair(summary, metric="bad_rate")`**
   - Used in Rule 5 for bin count reduction
   - Scans entire table for the pair of adjacent bins with **smallest metric difference**
   - Merging similar bins has minimal impact on model performance
   - Example: If bad_rate = [5%, 10%, 11%, 30%]:
     - Pair (0,1): difference = 5%
     - Pair (1,2): difference = 1% ← smallest
     - Pair (2,3): difference = 19%
     - Decision: Merge pair (1,2)

#### Bin Merging Mechanics

When bins are merged:
```
Before: [(18, 25], (25, 35], (35, 50]]
        Bin 0      Bin 1      Bin 2

Merge Bin 0 and Bin 1 (i=0):
  - New interval: (18, 35]  (take left of Bin 0, right of Bin 1)
  - Remove boundary at 25
  - New structure: [(18, 35], (35, 50]]

After reassignment:
  - All observations in (18, 35] stay there
  - All observations in (35, 50] stay there
  - Recompute WOE/IV for merged bins
```

#### Why Merge Order Matters

Rules are checked in order (1→5). This ensures:
- **Priority 1-2**: Eliminate statistical anomalies (empty or very small bins)
- **Priority 3-4**: Preserve logical monotonic trends in risk
- **Priority 5**: Finally reduce to business-acceptable bin count

**IMPORTANT**: This ordering prevents creating bins that satisfy early rules but violate later ones.

---

### Main Function: `auto_woe_binning_numeric`

#### Purpose

Automatically performs monotonic WOE binning for a numeric variable following all six rules.

#### Function Signature

```python
auto_woe_binning_numeric(
    df: pd.DataFrame,
    feature: str,
    target: str,
    initial_bins: int = 20,
    min_bin_pct: float = 0.05,
    max_final_bins: int = 10,
    min_final_bins: int = 3,
    min_iv: float = 0.02,
    max_iv: float = 0.50,
    max_iter: int = 100
) -> dict
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Input dataframe |
| `feature` | str | - | Numeric feature to bin |
| `target` | str | - | Binary target (0=Good, 1=Bad) |
| `initial_bins` | int | 20 | Number of initial quantile bins |
| `min_bin_pct` | float | 0.05 | Minimum bin size as % of total |
| `max_final_bins` | int | 10 | Maximum allowed final bins |
| `min_final_bins` | int | 3 | Minimum allowed final bins |
| `min_iv` | float | 0.02 | Minimum acceptable IV |
| `max_iv` | float | 0.50 | Maximum acceptable IV |
| `max_iter` | int | 100 | Maximum merge iterations |

#### Output Structure

```python
{
    "feature": "AGE",
    "woe_table": [...],                # Final WOE/IV table as list of dicts
    "checks": {                        # Quality check results
        "min_bin_pct_ok": "True with min_bin_pct=0.05",
        "good_bad_exist_ok": "True (good > 0 and bad > 0 in every bin)",
        "bad_rate_monotonic_ok": "True (checked on 'bad_rate' column)",
        "woe_monotonic_ok": "True (checked on 'woe' column)",
        "bin_count_ok": "True (min_bins=3, max_bins=10)",
        "iv_reasonable_ok": "True (min_iv=0.02, max_iv=0.50, total_iv=0.245)"
    },
    "status": "PASS",                  # PASS, REVIEW, REVIEW_WEAK_VARIABLE, REVIEW_POSSIBLE_LEAKAGE, REVIEW_NOT_CONVERGED
    "converged": True,                 # True if all rules satisfied, False if max_iter reached first
    "stopIteration": 5,                # Which iteration the algorithm stopped at
    "maxIter": 100,                    # Maximum iterations allowed
    "totalIv": 0.245,                  # Total IV for feature
    "interpretIv": "Strong Predictive Power",
    "finalIntervals": [                # Final bin boundaries as strings
        "(18.0, 25.0]",
        "(25.0, 35.0]",
        "(35.0, 50.0]"
    ],
    "stepsMetadata": {                 # Detailed merge history
        "1": {
            "bin_count": 20,
            "iv": 0.123,
            "triggered_rule": "Rule 1: Small bin merge",
            "summary": [...]
        },
        ...
    }
}
```

#### Output Status Codes

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| `PASS` | All rules satisfied, IV in range | Use binning directly |
| `REVIEW` | Some quality checks failed | Manually review violations |
| `REVIEW_WEAK_VARIABLE` | IV < 0.02 | Feature has weak predictive power, consider excluding |
| `REVIEW_POSSIBLE_LEAKAGE` | IV > 0.50 | Possible data leakage or overfitting, investigate |
| `REVIEW_NOT_CONVERGED` | max_iter reached before convergence | Algorithm could not satisfy all rules; inspect stepsMetadata |

#### Algorithm Flow

```
Input Data (df, feature, target)
   ↓
1. Data Validation & Cleaning
   - Check feature is numeric
   - Drop missing values
   ↓
2. Create Initial Bins
   - Use pd.qcut for quantile-based binning (default: 20 bins)
   ↓
3. Iterative Quality Refinement Loop (up to max_iter=100 iterations)
   │
   ├─ Calculate WOE/IV for current bins
   │
   ├─ Rule 1: Fix small bins?
   │  └─ If bin_pct < 0.05 → Merge with closest neighbor → Continue loop
   │
   ├─ Rule 2: Fix zero good/bad bins?
   │  └─ If good=0 OR bad=0 → Merge with closest neighbor → Continue loop
   │
   ├─ Rule 3: Fix bad_rate monotonicity?
   │  └─ If bad_rate not monotonic → Merge violating pair → Continue loop
   │
   ├─ Rule 4: Fix WOE monotonicity?
   │  └─ If WOE not monotonic → Merge violating pair → Continue loop
   │
   ├─ Rule 5: Too many bins?
   │  └─ If bin_count > max_final_bins → Merge most similar pair → Continue loop
   │
   └─ All rules satisfied? → Exit loop (converged=True)
   │
   ├─ Loop exhausted (max_iter reached)? → Exit loop (converged=False)
   └─ Log warning if not converged
   ↓
4. Final WOE/IV Calculation
   - Recompute table with final intervals
   ↓
5. Quality Check & Status Assignment
   ├─ If NOT converged → status = "REVIEW_NOT_CONVERGED"
   ├─ If total_iv < 0.02 → status = "REVIEW_WEAK_VARIABLE"
   ├─ If total_iv > 0.50 → status = "REVIEW_POSSIBLE_LEAKAGE"
   ├─ If all checks pass → status = "PASS"
   └─ If some checks fail → status = "REVIEW"
   ↓
Output: Final binning results with all metadata and iteration history
```

#### Iteration Logic Explained

Each iteration follows this sequence:

**Step 1: Compute Current Binning Statistics**
```
For each bin:
  - Count total observations
  - Count good cases (target=0) and bad cases (target=1)
  - Calculate bad_rate = bad / total
  - Calculate bin_pct = total / sum_of_all_totals
  - Apply smoothing and calculate good_dist, bad_dist
  - Compute WOE = ln(good_dist / bad_dist)
  - Compute IV contribution = (good_dist - bad_dist) × WOE
```

**Step 2: Check Rule 1 - Minimum Bin Size**
```
IF any bin has bin_pct < min_bin_pct (default 0.05):
  1. Identify problematic bin with smallest bin_pct
  2. Find closest neighbor:
     - If first bin → merge with right neighbor
     - If last bin → merge with left neighbor
     - Else → merge with neighbor having closer bad_rate
  3. Merge the pair and continue to next iteration
ELSE:
  Proceed to Rule 2
```

**Step 3: Check Rule 2 - Good and Bad Cases**
```
IF any bin has good=0 OR bad=0:
  1. Identify problematic bin
  2. Find closest neighbor (same logic as Rule 1)
  3. Merge the pair and continue to next iteration
ELSE:
  Proceed to Rule 3
```

**Step 4: Check Rule 3 - Bad Rate Monotonicity**
```
IF bad_rate is NOT monotonic increasing or decreasing:
  1. Detect violations:
     - Increasing violations: indices where bad_rate[i] > bad_rate[i+1]
     - Decreasing violations: indices where bad_rate[i] < bad_rate[i+1]
  2. Select direction with FEWER violations (to minimize information loss)
  3. Among violations, find pair with smallest bad_rate difference
  4. Merge this pair and continue to next iteration
ELSE:
  Proceed to Rule 4
```

**Step 5: Check Rule 4 - WOE Monotonicity**
```
IF WOE is NOT monotonic increasing or decreasing:
  1. Same logic as Rule 3, but applied to WOE values
  2. Select direction with fewer violations
  3. Find violating pair with smallest WOE difference
  4. Merge this pair and continue to next iteration
ELSE:
  Proceed to Rule 5
```

**Step 6: Check Rule 5 - Maximum Bin Count**
```
IF bin_count > max_final_bins (default 10):
  1. Find the most similar adjacent pair in the entire table
  2. Similarity metric: smallest bad_rate difference
  3. Merge this pair and continue to next iteration
ELSE:
  All rules satisfied → Exit loop with converged=True
```

**Convergence Check:**
```
If loop completes all iterations (max_iter=100) without converging:
  → Set converged=False
  → Log warning: "Auto binning did NOT converge..."
  → Proceed to status assignment with REVIEW_NOT_CONVERGED status
```

---

---

### Usage Example

#### Basic Usage

```python
import pandas as pd
from auto_binnig_woe import auto_woe_binning_numeric

# Load data
df = pd.read_csv("data.csv")

# Run automatic binning
result = auto_woe_binning_numeric(
    df=df,
    feature="AGE",
    target="default",  # 0=Good (no default), 1=Bad (default)
    initial_bins=20,
    min_bin_pct=0.05,
    max_final_bins=8
)

# Access results
print(f"Feature: {result['feature']}")
print(f"Status: {result['status']}")
print(f"Total IV: {result['total_iv']:.4f}")
print("\nFinal WOE Table:")
print(result['summary'])
print("\nQuality Checks:")
for check_name, check_result in result['checks'].items():
    print(f"  {check_name}: {check_result}")
```

#### With Output Files

```python
# Save WOE/IV results to files
result['summary'].to_csv("outputs/woe_binning_age.csv", index=False)

# Access metadata
metadata = result.get('metadata', {})
print(f"Total IV: {metadata.get('totalIv', result['total_iv'])}")
print(f"IV Interpretation: {metadata.get('interpretIv')}")
```

---

### Helper Functions

#### `convert_pd_interval_to_str(interval, round_apply=True)`

Converts pandas Interval objects to human-readable strings

**Example:**
```python
interval = pd.Interval(18, 25, closed="right")
result = convert_pd_interval_to_str(interval)
# Output: "from_18_to_25"
```

#### `interpret_iv(iv)`

Returns human-readable IV interpretation

**Example:**
```python
interpret_iv(0.15)  # Returns: "Medium Predictive Power"
interpret_iv(0.45)  # Returns: "Strong Predictive Power"
```

#### `is_monotonic(s)`

Checks if a numeric Series is monotonic increasing or decreasing

**Example:**
```python
bad_rates = pd.Series([0.05, 0.10, 0.15, 0.20])
is_monotonic(bad_rates)  # Returns: True

bad_rates = pd.Series([0.05, 0.10, 0.08, 0.20])
is_monotonic(bad_rates)  # Returns: False (violation at index 1→2)
```

---

### Common Use Cases

#### Case 1: Basic Credit Risk Binning

```python
# Bin age for credit risk scoring
result = auto_woe_binning_numeric(
    df=credit_data,
    feature="age",
    target="default",
    initial_bins=20
)

if result['status'] == 'SUCCESS':
    print("Age binning successful!")
    print(f"Created {len(result['summary'])} bins with IV = {result['total_iv']:.4f}")
```

#### Case 2: Strict Business Requirements

```python
# Maximum 5 bins, minimum 10% bin size
result = auto_woe_binning_numeric(
    df=credit_data,
    feature="income",
    target="default",
    initial_bins=20,
    min_bin_pct=0.10,
    max_final_bins=5
)
```

#### Case 3: Exploratory Analysis

```python
# Start with more bins, relaxed constraints
result = auto_woe_binning_numeric(
    df=credit_data,
    feature="utilization_ratio",
    target="default",
    initial_bins=30,
    min_bin_pct=0.02,
    max_final_bins=10
)
```

---

### Important Notes

**Target Convention**: The target variable must follow the convention:
- 0 = Good (no event of interest)
- 1 = Bad (event occurred)

**Pandas Interval Format**: Initial bins are pandas Interval objects with the format:
- `(left, right]` - interval includes right endpoint, excludes left
- Example: `(18, 25]` includes 25 but excludes 18

**Smoothing in WOE**: The `eps` parameter (default 0.5) prevents infinite WOE when a bin has zero good or bad cases. This is a statistical best practice.

**Monotonicity Direction**: The algorithm automatically selects whether to enforce increasing or decreasing monotonicity based on which requires fewer merges. This minimizes information loss.

**IV Range Warning**: An IV greater than 0.5 may indicate data leakage or overfitting. Investigate feature definition and data quality in such cases.

---

### Output Files

When `output_dir` is specified in `calculate_woe_iv_table`, the following files are saved:

1. **`woe_iv_metadata.json`** - Summary statistics and metadata
2. **`woe_iv_summary_table.csv`** - Detailed WOE/IV table for reporting

Example metadata structure:
```json
{
  "binCol": "_bin",
  "target": "default",
  "eps": 0.5,
  "totalIv": 0.245,
  "interpretIv": "Medium Predictive Power",
  "summaryTable": [
    {
      "bin": "(18, 25]",
      "total": 500,
      "good": 480,
      "bad": 20,
      "bad_rate": 0.04,
      "woe": 0.735,
      "iv": 0.031
    }
  ]
}
```

---

### Understanding Bin Selection Strategies: `find_closest_neighbor` vs `find_most_similar_adjacent_pair`

The binning algorithm uses two different strategies for selecting which bins to merge. While both functions compare adjacent bins, they serve different purposes and follow different prioritization logic.

#### `find_closest_neighbor(summary, idx, metric="bad_rate")`

**Purpose**: Find the optimal neighbor to merge with a **specific problematic bin**

**Use Cases**:
- Rule 1: When a bin is too small (less than `min_bin_pct`)
- Rule 2: When a bin has zero good cases or zero bad cases

**Selection Logic**:

This function focuses on a single problematic bin and compares only its two neighbors (left and right):

1. **For the first bin** (idx=0): Can only merge with right neighbor → returns 0
2. **For the last bin** (idx=len-1): Can only merge with left neighbor → returns idx-1
3. **For middle bins**: Compares both neighbors and selects the one with **smallest metric difference**

The metric difference is calculated as:
$$\text{Distance} = |\text{metric}_{problematic} - \text{metric}_{neighbor}|$$

**Priority/Tie-Breaking**: When comparing left vs. right neighbors with `<=` operator, the **left neighbor is preferred** if both neighbors have equal metric distance.

**Example**:

```
Problematic Bin Scenario:
Bin A: bad_rate = 0.10
Bin B: bad_rate = 0.11 ← PROBLEMATIC (too small)
Bin C: bad_rate = 0.30

Metric Distances:
- Left neighbor (A):  |0.11 - 0.10| = 0.01
- Right neighbor (C): |0.11 - 0.30| = 0.19

Decision: Merge B with A (smaller distance minimizes information loss)
```

**When Executed**: Early in the algorithm when specific data quality issues are detected (Rules 1-2)

---

#### `find_most_similar_adjacent_pair(summary, metric="bad_rate")`

**Purpose**: Find the most similar pair of adjacent bins **across the entire table** without focusing on a specific problem

**Use Cases**:
- Rule 5: When too many bins remain and reduction is needed
- Fallback: When no specific violations are found in Rules 3-4

**Selection Logic**:

This function scans ALL adjacent pairs in the summary table and calculates their metric differences:

$$\text{Difference}_i = |\text{metric}_{bin_i} - \text{metric}_{bin_{i+1}}|$$

Then selects the pair with the **globally smallest difference**:

$$\text{Selected Pair} = \arg\min_i \left( \text{Difference}_i \right)$$

**Priority/Tie-Breaking**: If multiple pairs have identical minimum differences, the function returns the **first occurrence** (lowest index pair) due to how Python's `min()` function works.

**Example**:

```
Entire Summary Table:
Bin 1: bad_rate = 0.05
Bin 2: bad_rate = 0.10
Bin 3: bad_rate = 0.11  ← Very similar to Bin 2
Bin 4: bad_rate = 0.35

Metric Differences for Adjacent Pairs:
Pair (0,1): |0.05 - 0.10| = 0.05
Pair (1,2): |0.10 - 0.11| = 0.01 ← SMALLEST
Pair (2,3): |0.11 - 0.35| = 0.24

Decision: Merge bins 2 and 3 (smallest difference)
```

---

#### How the Loop Works (Step-by-Step Execution)

The function builds a `diffs` list by iterating through all adjacent pairs and calculating their differences. Here's the exact flow:

**Step 0: Initialize**
```python
diffs = []  # Empty list to store (index, difference) tuples
```

**Step 1-4: Loop through each adjacent pair**

Using the example data:
```python
summary = pd.DataFrame({
    "bad_rate": [0.10, 0.15, 0.14, 0.30, 0.28]
})
# 5 bins → 4 adjacent pairs
```

**Iteration 1 (i=0):**
```python
diff = abs(summary.loc[0, "bad_rate"] - summary.loc[1, "bad_rate"])
     = abs(0.10 - 0.15) = 0.05
diffs.append((0, 0.05))
# Log: "Index 0 <-> 1 | bad_rate: 0.100000 → 0.150000 | Difference = 0.050000"
# diffs = [(0, 0.05)]
```

**Iteration 2 (i=1):**
```python
diff = abs(summary.loc[1, "bad_rate"] - summary.loc[2, "bad_rate"])
     = abs(0.15 - 0.14) = 0.01  ← SMALL!
diffs.append((1, 0.01))
# Log: "Index 1 <-> 2 | bad_rate: 0.150000 → 0.140000 | Difference = 0.010000"
# diffs = [(0, 0.05), (1, 0.01)]
```

**Iteration 3 (i=2):**
```python
diff = abs(summary.loc[2, "bad_rate"] - summary.loc[3, "bad_rate"])
     = abs(0.14 - 0.30) = 0.16
diffs.append((2, 0.16))
# Log: "Index 2 <-> 3 | bad_rate: 0.140000 → 0.300000 | Difference = 0.160000"
# diffs = [(0, 0.05), (1, 0.01), (2, 0.16)]
```

**Iteration 4 (i=3):**
```python
diff = abs(summary.loc[3, "bad_rate"] - summary.loc[4, "bad_rate"])
     = abs(0.30 - 0.28) = 0.02
diffs.append((3, 0.02))
# Log: "Index 3 <-> 4 | bad_rate: 0.300000 → 0.280000 | Difference = 0.020000"
# diffs = [(0, 0.05), (1, 0.01), (2, 0.16), (3, 0.02)]
```

**Step 5: Find Minimum**

After loop completes, find the pair with smallest difference:

```python
diffs = [(0, 0.05), (1, 0.01), (2, 0.16), (3, 0.02)]

merge_idx = min(diffs, key=lambda x: x[1])[0]
#           ↓ 
#  Compare all x[1] values: 0.05, 0.01, 0.16, 0.02
#                                 ↑ MINIMUM
#  Returns: (1, 0.01)[0] = 1

min_diff = min(diffs, key=lambda x: x[1])[1]
#        = 0.01
```

**Result:**
```python
return merge_idx  # Returns: 1

# This means:
# - Merge pair at index (1, 2)
# - Call merge_intervals(intervals, i=1)
# - Combine Bin 1 and Bin 2 together
```

**Key Behavior:**
- **Deterministic**: Always returns the FIRST pair with minimum difference
- **Greedy**: Only one merge per call (no multiple suggestions)
- **Tie-breaking**: If multiple pairs have same minimum, first index wins

**Iteration Summary Table:**

| i | Pair | bad_rate | Difference | diffs List |
|---|------|----------|------------|------------|
| 0 | 0-1 | 0.10→0.15 | 0.05 | [(0, 0.05)] |
| 1 | 1-2 | 0.15→0.14 | **0.01** | [(0, 0.05), (1, 0.01)] |
| 2 | 2-3 | 0.14→0.30 | 0.16 | [..., (2, 0.16)] |
| 3 | 3-4 | 0.30→0.28 | 0.02 | [..., (3, 0.02)] |

`min()` selected: `(1, 0.01)` → **merge_idx = 1** 

---

#### Key Differences Summary

| Aspect | `find_closest_neighbor` | `find_most_similar_adjacent_pair` |
|--------|------------------------|----------------------------------|
| **Scope** | Compares 2 neighbors of a specific bin | Compares ALL adjacent pairs |
| **Input** | Problem bin index (idx) | Entire summary table |
| **When Used** | Problem-driven (Rules 1-2) | Global optimization (Rule 5) |
| **Selection** | Closest neighbor to problem bin | Most similar pair anywhere in table |
| **Tie-Breaking** | Left neighbor preferred (`<=`) | First occurrence (lowest index) |
| **Information Loss** | Minimized by matching closest neighbor | Minimized by merging most similar bins |

---

#### Priority Execution Order

The algorithm processes both functions in a specific order within each iteration:

```
Iteration Loop:
  1. Calculate WOE/IV for current bins
  
  2. Rule 1 (Small Bins)
     └─ find_closest_neighbor() → Merge problematic small bin with closest neighbor
     
  3. Rule 2 (Zero Good/Bad)
     └─ find_closest_neighbor() → Merge problematic zero-distribution bin
     
  4. Rule 3 (Bad Rate Monotonicity)
     └─ find_monotonicity_violation_pair() → Find violating pair
         └─ Uses logic similar to find_closest_neighbor for violation pair
     
  5. Rule 4 (WOE Monotonicity)
     └─ find_monotonicity_violation_pair() → Same as Rule 3
     
  6. Rule 5 (Max Bin Count)
     └─ find_most_similar_adjacent_pair() → Global search for most similar pair
     
  7. All rules satisfied?
     └─ YES → Converged (exit loop)
     └─ NO → Continue to next iteration
```

**Why This Order Matters**:
- **Early rules (1-2)** fix critical data quality issues with targeted merges
- **Middle rules (3-4)** enforce logical trends with minimal disruption
- **Late rule (5)** reduces bin count globally while maintaining quality
- **Result**: Each phase builds on previous quality improvements

---

#### Practical Implications

**Use Case 1: Feature with Quality Issues**
```
Initial State: 20 bins, some with < 5% observations

Execution:
  Rule 1 → find_closest_neighbor() identifies small bins
          Merges each with its closest similar neighbor
          Result: 18 bins, all > 5%
          
  Rule 2 → find_closest_neighbor() checks for zero distributions
          (None found)
          Result: 18 bins
          
  Rule 3 → Checks monotonicity (satisfied)
  Rule 4 → Checks WOE (satisfied)
  
  Rule 5 → Max bins = 10, current = 18
          find_most_similar_adjacent_pair() globally scans
          Repeatedly finds and merges most similar pairs
          Result: 10 final bins
```

**Use Case 2: Clean Feature Needing Only Bin Reduction**
```
Initial State: 25 bins, all have good data quality

Execution:
  Rule 1 → All bins > 5% (skip)
  Rule 2 → All bins have good and bad (skip)
  Rule 3 → Bad rate is monotonic (skip)
  Rule 4 → WOE is monotonic (skip)
  
  Rule 5 → Max bins = 10, current = 25
          find_most_similar_adjacent_pair() runs repeatedly
          Iteration 1: Finds pair with min difference, merges → 24 bins
          Iteration 2: Finds next min pair, merges → 23 bins
          ... (continues)
          Iteration 15: Final merge → 10 bins (converged)
```

---

#### Function Behavior Comparison

```python
# Example data for both functions
summary = pd.DataFrame({
    "bin": ["A", "B", "C", "D", "E"],
    "bad_rate": [0.10, 0.11, 0.30, 0.025, 0.32]
})

# Scenario 1: Using find_closest_neighbor on problematic bin
# Bin B (index=1) is problematic
result_closest = find_closest_neighbor(summary, idx=1, metric="bad_rate")
# Returns: 0 (left neighbor A is closer: |0.11-0.10|=0.01 vs |0.11-0.30|=0.19)

# Scenario 2: Using find_most_similar_adjacent_pair globally
result_global = find_most_similar_adjacent_pair(summary, metric="bad_rate")
# Scans all pairs:
#   Pair (0,1): |0.10-0.11| = 0.01
#   Pair (1,2): |0.11-0.30| = 0.19
#   Pair (2,3): |0.30-0.025| = 0.275
#   Pair (3,4): |0.025-0.32| = 0.295
# Returns: 0 (first pair with 0.01 difference is smallest)

# Note: Both return 0 in this case, but for different reasons!
# - find_closest_neighbor: Because bin A is closer to problem bin B
# - find_most_similar_adjacent_pair: Because pair (A,B) is globally most similar
```

---

#### Design Philosophy

The dual-function approach reflects a **graduated problem-solving strategy**:

1. **Targeted fixes** (`find_closest_neighbor`): Solve specific identified problems with minimal disruption
2. **Global optimization** (`find_most_similar_adjacent_pair`): Once problems are fixed, optimize overall bin structure

This ensures the algorithm:
- Doesn't over-merge early (which could destroy predictive power)
- Addresses quality issues first (data-driven approach)
- Only applies broad reductions when necessary (business constraints)
- Preserves information throughout the process
