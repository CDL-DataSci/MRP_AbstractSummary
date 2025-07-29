# step_split_data

import pandas as pd
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
input_csv = "step1a.csv"
train_output = "train.csv"
val_output = "val.csv"
test_output = "test.csv"
random_seed = 42  # for reproducibility

# === LOAD DATA ===
df = pd.read_csv(input_csv)

# === INITIAL SPLIT: TRAIN (70%) and TEMP (30%) ===
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=random_seed,
    shuffle=True
)

# === SPLIT TEMP INTO VALIDATION (15%) AND TEST (15%) ===
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=random_seed,
    shuffle=True
)

# === SAVE SPLITS ===
train_df.to_csv(train_output, index=False, sep="\t")
val_df.to_csv(val_output, index=False, sep="\t")
test_df.to_csv(test_output, index=False, sep="\t")

print(f"✅ Data split complete!")
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")