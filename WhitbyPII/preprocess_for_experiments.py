# preprocess_for_experiments.py
import pandas as pd
import os

# Load tab-separated data and skip problem rows
df = pd.read_csv("data/train_with_canaries_cleaned.csv", sep="\t", on_bad_lines="skip")

# Compute length bins
df["length"] = df["text"].astype(str).apply(lambda x: len(x.split()))
df["length_bin"] = pd.qcut(df["length"], q=3, labels=["short", "medium", "long"])

# Extract year from date
df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

# Drop rows with invalid or missing years
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# Save cleaned and enriched version (overwrite or update existing)
os.makedirs("data", exist_ok=True)
df.to_csv("data/train_with_canaries_cleaned.csv", sep="\t", index=False)

