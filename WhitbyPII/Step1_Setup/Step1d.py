import pandas as pd
import re

# === CONFIGURATION ===
train_input = "train.csv"
canary_input = "canaries.csv"
output_csv = "train_with_canaries.csv"
canary_date = "2021-10-08"

# === TEXT CLEANING ===
def clean_files(text):
    broken_patterns = [
        r"‚Ä¢", r"â€“", r"â€”", r"â€œ", r"â€", r"â€™",
        r"ÔÇ∑", r"ÔÇö", r"Ã¢", r"â€¦", r"Ã©"
    ]
    for pattern in broken_patterns:
        text = re.sub(pattern, " ", text)
    return text.strip()

# === LOAD DATA ===
train_df = pd.read_csv(train_input, sep="\t")
canary_df = pd.read_csv(canary_input, sep=";")  # uses semicolon separator

# === FORMAT CANARY ROWS TO MATCH TRAINING SCHEMA ===
canary_rows = []
for _, row in canary_df.iterrows():
    canary_rows.append({
        "filename": f"{row['canary_id']}.txt",
        "category": f"Canary_{row['type'].replace(' ', '_')}",
        "date": canary_date,
        "text": clean_files(row["canary_text"])
    })

# === COMBINE AND SAVE ===
canary_df_formatted = pd.DataFrame(canary_rows)
train_with_canaries = pd.concat([train_df, canary_df_formatted], ignore_index=True)
train_with_canaries.to_csv(output_csv, sep="\t", index=False)

print(f"✅ Canary injection complete: {output_csv}")
print(f"Original train size: {len(train_df)}")
print(f"Canaries inserted: {len(canary_rows)}")
print(f"Total new training size: {len(train_with_canaries)}")
