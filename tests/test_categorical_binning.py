"""
Test script for categorical WoE binning function
"""

import pandas as pd
import numpy as np
from auto_binning_woe_categorical import auto_woe_binning_categorical

# Create sample categorical data
np.random.seed(42)

# Generate sample data with categorical features
n_samples = 1000

# Create a categorical feature with different bad rates for each category
# categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
categories = [-1, 2, -3, 1, 2, 3, 4]
data = []

for category in categories:
    # Different bad rates for different categories
    if category in [-1, 2]:
        bad_rate = 0.1
    elif category in [-3, 1]:
        bad_rate = 0.3
    else:
        bad_rate = 0.6
    
    # Generate samples for this category
    n_cat_samples = n_samples // len(categories)
    for _ in range(n_cat_samples):
        is_bad = np.random.random() < bad_rate
        data.append({
            'feature': category,
            'target': 1 if is_bad else 0
        })

df = pd.DataFrame(data)

# Add some more samples to ensure we have exactly n_samples
while len(df) < n_samples:
    cat = np.random.choice(categories)
    bad_rate = 0.1 if cat in [-1, 2] else (0.3 if cat in [-3, 1] else 0.6)
    is_bad = np.random.random() < bad_rate
    df = pd.concat([df, pd.DataFrame({'feature': [cat], 'target': [1 if is_bad else 0]})])

df = df.reset_index(drop=True)[:n_samples]
df["feature"] = df["feature"].astype(object)

print("Sample Data:")
print(df.head(10))
print(f"\nData shape: {df.shape}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
print(f"\nFeature categories:\n{df['feature'].value_counts()}")

# Run categorical binning
print("\n" + "="*50)
print("Running Categorical WoE Binning")
print("="*50 + "\n")

try:
    result = auto_woe_binning_categorical(
        df=df,
        feature='feature',
        target='target',
        min_bin_pct=0.05,
        max_final_bins=5,
        min_final_bins=2,
        min_iv=0.02,
        max_iv=0.50,
        max_iter=20
    )
    
    print("\n" + "="*50)
    print("BINNING RESULTS")
    print("="*50)
    print(f"Status: {result['status']}")
    print(f"Total IV: {result['totalIv']}")
    print(f"Number of Final Bins: {result['numberofBins']}")
    
    print("\nValues to the Group:")
    for bin_name, categories in result['valuesToTheGroup'].items():
        print(f"  {bin_name}: {categories}")
    
    print("\nWoE Table:")
    woe_table_df = pd.DataFrame(result['woe_table'])
    print(woe_table_df[['_bin', 'total', 'good', 'bad', 'bad_rate', 'woe', 'iv']].to_string(index=False))
    
    print("\nQuality Checks:")
    for check_name, check_result in result['checks'].items():
        print(f"  {check_name}: {check_result}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
