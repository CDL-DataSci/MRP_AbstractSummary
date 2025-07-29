# CESR_metrics.py

import os
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from rouge_score import rouge_scorer
import bert_score
import re

GENERATED_DIR = "outputs" 

#Load Canaries
def load_canaries(path="canaries.csv"):
    df_canary = pd.read_csv(path, sep=";")
    canary_patterns = [re.escape(text.strip()) for text in df_canary["canary_text"].dropna().unique()]
    return canary_patterns

CANARY_PATTERNS = load_canaries()

#Load all outputs
def load_all_outputs(output_dir):
    all_results = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            with open(os.path.join(output_dir, filename)) as f:
                data = json.load(f)
                all_results.extend(data)
    return pd.DataFrame(all_results)

#ROUGE Scores
def evaluate_rouge(df):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1s, rouge2s, rougels = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing ROUGE"):
        scores = scorer.score(str(row["reference"]), str(row["generated_summary"]))
        rouge1s.append(scores['rouge1'].fmeasure)
        rouge2s.append(scores['rouge2'].fmeasure)
        rougels.append(scores['rougeL'].fmeasure)

    return {
        "rouge1": sum(rouge1s)/len(rouge1s),
        "rouge2": sum(rouge2s)/len(rouge2s),
        "rougeL": sum(rougels)/len(rougels),
    }

#BERTScores
def evaluate_bertscore(df):
    P, R, F1 = bert_score.score(
        df["generated_summary"].tolist(),
        df["reference"].tolist(),
        lang="en",
        verbose=True
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }

#Canary Extraction Score (CESR)
def evaluate_cesr(df, canary_patterns):
    total = 0
    hits = 0

    for canary in canary_patterns:
        regex = re.compile(canary)
        for summary in df["generated_summary"]:
            if regex.search(str(summary)):
                hits += 1
                break 
        total += 1

    return {
        "cesr": hits / total if total > 0 else 0
    }

#Precision & Recall
def evaluate_precision_recall(df):
    if "predicted_label" in df.columns and "true_label" in df.columns:
        return {
            "precision": precision_score(df["true_label"], df["predicted_label"], average='macro'),
            "recall": recall_score(df["true_label"], df["predicted_label"], average='macro'),
        }
    return {}

#Main
if __name__ == "__main__":
    df = load_all_outputs(GENERATED_DIR)

    print("✔ Loaded", len(df), "generated samples")

    rouge_scores = evaluate_rouge(df)
    bert_scores = evaluate_bertscore(df)
    cesr_score = evaluate_cesr(df, CANARY_PATTERNS)
    pr_scores = evaluate_precision_recall(df)

    results = {
        **rouge_scores,
        **bert_scores,
        **cesr_score,
        **pr_scores
    }

    print("\n📊 Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    with open("evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
