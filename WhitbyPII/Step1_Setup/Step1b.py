# Step 1B: NER and Proxy DPDD

import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
import csv

INPUT_CSV = "step1a.csv"
OUTPUT_SCORES = "dpdd_scores.csv"
OUTPUT_CATEGORY_AVG = "dpdd_category_averages.csv"

#Load data
df = pd.read_csv("step1a.csv", quoting=csv.QUOTE_ALL, engine="python", on_bad_lines="skip")

#Load NER
nlp = spacy.load("en_core_web_lg") 
PII_LABELS = {"PERSON", "ORG", "NORP", "FAC"}
ALPHA = 0.5 
BETA = 0.5 

results = []

# Process each document
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

score_df = pd.DataFrame(results)

score_df["token_score"] = score_df["token_count"] / score_df["token_count"].max()
score_df["pii_score"] = score_df["total_pii_entities"] / score_df["total_pii_entities"].max()
score_df["dpdd_risk_score"] = ALPHA * score_df["token_score"] + BETA * score_df["pii_score"]
score_df.to_csv(OUTPUT_SCORES, index=False, quoting=1)

# Avg by category
category_avg = score_df.groupby("category")["dpdd_risk_score"].agg(["mean", "std", "max"]).reset_index()
category_avg.columns = ["category", "mean_risk_score", "std_risk_score", "max_risk_score"]
category_avg.to_csv(OUTPUT_CATEGORY_AVG, index=False)

print("AVG DPDD SCORE","\n", category_avg, "\n")
print("SCORE PER FILE", "\n", score_df.head(5),)

#graph df
import matplotlib.pyplot as plt

category_avg_sorted = category_avg.sort_values(by="mean_risk_score", ascending=False)

#histogram
plt.figure(figsize=(10, 8))
plt.barh(category_avg_sorted["category"], category_avg_sorted["mean_risk_score"])
plt.xlabel("Mean DPDD Risk Score")
plt.ylabel("File Category")
plt.title("Mean Risk Score by File Category")
plt.tight_layout()
plt.savefig("dpdd_category_histogram_horizontal.png")  
plt.show()


#scatter plot
category_counts = score_df["category"].value_counts().reset_index()
category_counts.columns = ["category", "file_count"]
scatter_data = pd.merge(category_avg, category_counts, on="category", how="left")

plt.figure(figsize=(12, 6))

categories = scatter_data["category"].unique()
colors = plt.cm.tab20.colors 

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

plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

plt.tight_layout()
plt.savefig("dpdd_category_scatterplot_legend.png", bbox_inches="tight")
plt.show()
