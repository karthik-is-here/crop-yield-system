import pandas as pd

# ── Load datasets ──────────────────────────────────────────
crop_df = pd.read_csv("data/crop_recommendation.csv")
yield_df = pd.read_csv("data/yield_df.csv")

# ── Basic shape ────────────────────────────────────────────
print("=" * 50)
print("CROP RECOMMENDATION DATASET")
print("=" * 50)

print(f"\nShape: {crop_df.shape}")
print(f"\nColumns: {list(crop_df.columns)}")
print(f"\nFirst 5 rows:")
print(crop_df.head())
print(f"\nData types:")
print(crop_df.dtypes)
print(f"\nMissing values:")
print(crop_df.isnull().sum())
print(f"\nBasic statistics:")
print(crop_df.describe())
print(f"\nUnique crops: {crop_df['label'].unique()}")
print(f"Number of crop types: {crop_df['label'].nunique()}")

print("\n")
print("=" * 50)
print("YIELD PREDICTION DATASET")
print("=" * 50)

print(f"\nShape: {yield_df.shape}")
print(f"\nColumns: {list(yield_df.columns)}")
print(f"\nFirst 5 rows:")
print(yield_df.head())
print(f"\nData types:")
print(yield_df.dtypes)
print(f"\nMissing values:")
print(yield_df.isnull().sum())
print(f"\nBasic statistics:")
print(yield_df.describe())
print(f"\nUnique crops: {yield_df['Item'].unique()}")
print(f"Number of crop types: {yield_df['Item'].nunique()}")