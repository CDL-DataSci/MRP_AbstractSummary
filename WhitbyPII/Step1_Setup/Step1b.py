import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
import csv

# === CONFIGURATION ===
INPUT_CSV = "step1a.csv"
OUTPUT_SCORES = "dpdd_scores.csv"
OUTPUT_CATEGORY_AVG = "dpdd_category_averages.csv"

# === Load data ===
df = pd.read_csv("step1a.csv", quoting=csv.QUOTE_ALL, engine="python", on_bad_lines="skip")

# === Load spaCy NER model ===
nlp = spacy.load("en_core_web_lg")  # use en_core_web_lg if too slow

# === Labels to track for scoring ===
PII_LABELS = {"PERSON", "ORG", "NORP", "FAC"}

# === Scoring weights (adjustable) ===
ALPHA = 0.5  # weight on normalized token count
BETA = 0.5   # weight on normalized PII entity count

# === Containers for scoring results ===
results = []

# === Process each document ===
print("⚙️ Processing documents with spaCy...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["filename"]
    category = row["category"]
    text = row["text"]

    if pd.isna(text):
        continue

    doc = nlp(str(text))
    token_count = len(doc)

    entity_counts = {label: 0 for label in PII_LABELS}
    for ent in doc.ents:
        if ent.label_ in PII_LABELS:
            entity_counts[ent.label_] += 1

    total_pii = sum(entity_counts.values())

    results.append({
        "filename": filename,
        "category": category,
        "token_count": token_count,
        **entity_counts,
        "total_pii_entities": total_pii
    })

# === Create DataFrame and normalize ===
score_df = pd.DataFrame(results)

# Normalize token count and total PII for scoring
score_df["token_score"] = score_df["token_count"] / score_df["token_count"].max()
score_df["pii_score"] = score_df["total_pii_entities"] / score_df["total_pii_entities"].max()
score_df["dpdd_risk_score"] = ALPHA * score_df["token_score"] + BETA * score_df["pii_score"]

# === Save full document-level scores ===
score_df.to_csv(OUTPUT_SCORES, index=False, quoting=1)
print(f"✅ Saved per-document scores to {OUTPUT_SCORES}")

# === Aggregate by category ===
category_avg = score_df.groupby("category")["dpdd_risk_score"].agg(["mean", "std", "max"]).reset_index()
category_avg.columns = ["category", "mean_risk_score", "std_risk_score", "max_risk_score"]
category_avg.to_csv(OUTPUT_CATEGORY_AVG, index=False)
print(f"✅ Saved category averages to {OUTPUT_CATEGORY_AVG}")

print("AVG DPDD SCORE","\n", category_avg, "\n")
print("SCORE PER FILE", "\n", score_df.head(5),)

#graph df
import matplotlib.pyplot as plt

category_avg_sorted = category_avg.sort_values(by="mean_risk_score", ascending=False)

# Create histogram
plt.figure(figsize=(10, 8))
plt.barh(category_avg_sorted["category"], category_avg_sorted["mean_risk_score"])
plt.xlabel("Mean DPDD Risk Score")
plt.ylabel("File Category")
plt.title("Mean Risk Score by File Category")
plt.tight_layout()
plt.savefig("dpdd_category_histogram_horizontal.png")  # Save as image file
plt.show()

#scatter plot
import matplotlib.pyplot as plt

# Count number of files per category
category_counts = score_df["category"].value_counts().reset_index()
category_counts.columns = ["category", "file_count"]

# Merge counts with mean risk scores
scatter_data = pd.merge(category_avg, category_counts, on="category", how="left")

# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    scatter_data["file_count"],
    scatter_data["mean_risk_score"],
    c=range(len(scatter_data)),  # color code by index for visual separation
    cmap="tab20",                # 20-category discrete color palette
    s=100,
    alpha=0.8,
    edgecolors="k"
)

# Label axes and title
plt.xlabel("Number of Files in Category")
plt.ylabel("Mean DPDD Risk Score")
plt.title("File Count vs. Risk Score by Category")

# Annotate each point with the category name
for i, row in scatter_data.iterrows():
    plt.annotate(
        row["category"],
        (row["file_count"], row["mean_risk_score"]),
        fontsize=8,
        xytext=(5, 3),
        textcoords="offset points"
    )

plt.tight_layout()
plt.savefig("dpdd_category_scatterplot.png")
plt.show()

import matplotlib.pyplot as plt

# Count number of files per category
category_counts = score_df["category"].value_counts().reset_index()
category_counts.columns = ["category", "file_count"]

# Merge with average risk scores
scatter_data = pd.merge(category_avg, category_counts, on="category", how="left")

# Plot setup
plt.figure(figsize=(12, 6))

# Assign a unique color per category
categories = scatter_data["category"].unique()
colors = plt.cm.tab20.colors  # Up to 20 distinct colors

for i, category in enumerate(categories):
    data = scatter_data[scatter_data["category"] == category]
    plt.scatter(
        data["file_count"],
        data["mean_risk_score"],
        label=category,
        color=colors[i % len(colors)],
        s=100,
        alpha=0.8,
        edgecolors="k"
    )

# Labels
plt.xlabel("Number of Files in Category")
plt.ylabel("Mean DPDD Risk Score")
plt.title("File Count vs. Risk Score by Category")

# Legend (placed outside right)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

plt.tight_layout()
plt.savefig("dpdd_category_scatterplot_legend.png", bbox_inches="tight")
plt.show()
