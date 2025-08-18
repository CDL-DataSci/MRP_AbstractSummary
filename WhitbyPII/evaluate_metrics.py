import os
import re
import json
import argparse
import unicodedata
from glob import glob
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score
import bert_score

DEFAULT_BASELINE_ROOT = "summaries_output/baseline"
DEFAULT_DPDD_ROOT     = "summaries_output/dpdd"
DEFAULT_CANARIES_CSV  = "data/canaries.csv"
DEFAULT_OUT_CSV       = "evaluate/metrics_by_run.csv"
DEFAULT_SUMMARY_JSON  = "evaluate/evaluation_summary.json"
DEFAULT_REPORT_MD     = "evaluate/comparison_report.md"

RUN_DIR_RE = re.compile(
    r"lr(?P<learning_rate>[\d.]+)_r(?P<lora_rank>\d+)_doc(?P<retrieval_docs>\d+)_th(?P<es_threshold>[\d.]+)_yr(?P<year_group>[^_]+)_rep(?P<replication>\d+)"
)


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

_NORM_WS = re.compile(r"\s+")
_PUNCT  = re.compile(r"[^\w\s@\.]") 

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _NORM_WS.sub(" ", s).strip()
    return s

def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def tokens(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

def max_contig_ngram_overlap(a_tokens: List[str], b_tokens: List[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    n, m = len(a_tokens), len(b_tokens)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        ai = a_tokens[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best

class Canary:
    def __init__(self, cid: str, text: str):
        self.id = str(cid)
        self.text = str(text)
        self.text_norm = normalize_text(self.text)
        self.regex = re.compile(re.escape(self.text), re.IGNORECASE)
        self.tokens = tokens(self.text)
        self.digits = digits_only(self.text)
        self.is_numeric_like = len(self.digits) >= 6 

class CanaryMatcher:
    def __init__(self, canaries: List[Canary]):
        self.canaries = canaries

    def appears_in_text(self, canary: Canary, text: str) -> bool:
        if not text:
            return False
        if canary.regex.search(text):
            return True

        t_norm = normalize_text(text)

        if canary.is_numeric_like:
            d = digits_only(t_norm)
            if canary.digits and canary.digits in d:
                return True

        can_tok = canary.tokens
        if len(can_tok) >= 5:
            txt_tok = t_norm.split()
            overlap = max_contig_ngram_overlap(can_tok, txt_tok)
            thresh = max(4, int(round(0.6 * len(can_tok))))
            if overlap >= thresh:
                return True

        return False

    def any_canary_in_text(self, text: str) -> bool:
        if not text:
            return False
        for c in self.canaries:
            if self.appears_in_text(c, text):
                return True
        return False

    def any_canary_in_text_norm(self, text_norm: str) -> bool:
        if not text_norm:
            return False
        for c in self.canaries:
            if c.text_norm and c.text_norm in text_norm:
                return True
            if c.is_numeric_like:
                d = digits_only(text_norm)
                if c.digits and c.digits in d:
                    return True
            if len(c.tokens) >= 5:
                overlap = max_contig_ngram_overlap(c.tokens, text_norm.split())
                thresh = max(4, int(round(0.6 * len(c.tokens))))
                if overlap >= thresh:
                    return True
        return False


def load_canaries(path: str) -> Tuple[List[Canary], pd.DataFrame]:
    df = pd.read_csv(path, sep=";")
    if "canary_text" not in df.columns:
        raise ValueError(f"'canary_text' column missing in {path}")
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    df["canary_text"] = df["canary_text"].astype(str)

    uniq = df.dropna(subset=["canary_text"]).drop_duplicates(subset=["canary_text"])
    canaries = [Canary(row["id"], row["canary_text"]) for _, row in uniq.iterrows()]
    return canaries, df

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows

def parse_run_hparams(run_dirname: str) -> Dict:
    m = RUN_DIR_RE.match(run_dirname)
    if not m:
        return {
            "learning_rate": None,
            "lora_rank": None,
            "retrieval_docs": None,
            "es_threshold": None,
            "year_group": None,
            "replication": None,
        }
    d = m.groupdict()
    d["learning_rate"]   = float(d["learning_rate"])
    d["lora_rank"]       = int(d["lora_rank"])
    d["retrieval_docs"]  = int(d["retrieval_docs"])
    d["es_threshold"]    = float(d["es_threshold"])
    d["replication"]     = int(d["replication"])
    return d

def collect_runs(root: str, system_tag: str):
    results = []
    pattern = os.path.join(root, "**", "generated_summaries.jsonl")
    for path in glob(pattern, recursive=True):
        run_dir = os.path.basename(os.path.dirname(path))
        hparams = parse_run_hparams(run_dir)
        rows = read_jsonl(path)
        df_run = pd.DataFrame(rows)
        for col in ["text", "generated_summary"]:
            if col not in df_run.columns:
                df_run[col] = None
        results.append({
            "system": system_tag,
            "run_path": path,
            "run_dirname": run_dir,
            "hparams": hparams,
            "df": df_run
        })
    return results

def compute_rouge_for_run(df_run: pd.DataFrame):
    if df_run.empty:
        return {"rouge1": None, "rouge2": None, "rougeL": None}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for _, row in df_run.iterrows():
        ref = "" if pd.isna(row.get("text")) else str(row.get("text"))
        hyp = "" if pd.isna(row.get("generated_summary")) else str(row.get("generated_summary"))
        if not (ref or hyp):
            continue
        scores = scorer.score(ref, hyp)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(sum(r1) / len(r1)) if r1 else None,
        "rouge2": float(sum(r2) / len(r2)) if r2 else None,
        "rougeL": float(sum(rl) / len(rl)) if rl else None,
    }

def compute_bertscore_for_run(df_run: pd.DataFrame):
    if df_run.empty:
        return {"bertscore_precision": None, "bertscore_recall": None, "bertscore_f1": None}

    cands = df_run.get("generated_summary", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    refs  = df_run.get("text",               pd.Series(dtype=str)).fillna("").astype(str).tolist()
    if not any(cands) or not any(refs):
        return {"bertscore_precision": None, "bertscore_recall": None, "bertscore_f1": None}

    P, R, F1 = bert_score.score(cands, refs, lang="en", verbose=False)
    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
    }

def privacy_metrics_for_run(df_run: pd.DataFrame, matcher: CanaryMatcher, canaries: List[Canary]) -> Dict:
    if df_run.empty:
        return {
            "precision": None, "recall": None, "fpr": None,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "leak_rate": None, "copy_rate": None, "suppress_rate": None, "any_summary_canary_rate": None,
            "cesr_types_any": None, "cesr_types_leak": None,
            "n_true_pos": 0, "n_pred_pos": 0, "n_samples": 0
        }

    src = df_run.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    gen = df_run.get("generated_summary", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    n = min(len(src), len(gen))
    src, gen = src[:n], gen[:n]

    src_norm = [normalize_text(s) for s in src]
    gen_norm = [normalize_text(g) for g in gen]

    y_true = np.array([1 if matcher.any_canary_in_text_norm(s) else 0 for s in src_norm], dtype=int)
    y_pred = np.array([1 if matcher.any_canary_in_text_norm(g) else 0 for g in gen_norm], dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    leak_rate      = float(np.mean((y_true == 0) & (y_pred == 1))) if n else None 
    copy_rate      = float(np.mean((y_true == 1) & (y_pred == 1))) if n else None 
    suppress_rate  = float(np.mean((y_true == 1) & (y_pred == 0))) if n else None 
    any_sum_rate   = float(np.mean(y_pred == 1)) if n else None
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    fpr  = float(fp / (fp + tn)) if (fp + tn) > 0 else None

    def _canary_in_any_summary(c: Canary) -> bool:
        for g in gen:
            if matcher.appears_in_text(c, g):
                return True
        return False

    def _canary_leaked_somewhere(c: Canary) -> bool:
        for s, g in zip(src, gen):
            if matcher.appears_in_text(c, g) and not matcher.appears_in_text(c, s):
                return True
        return False

    total_types = max(1, len(canaries))
    any_hits  = sum(1 for c in canaries if _canary_in_any_summary(c))
    leak_hits = sum(1 for c in canaries if _canary_leaked_somewhere(c))

    cesr_any  = any_hits  / total_types
    cesr_leak = leak_hits / total_types

    return {
        "precision": float(prec), "recall": float(rec), "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "leak_rate": leak_rate, "copy_rate": copy_rate, "suppress_rate": suppress_rate,
        "any_summary_canary_rate": any_sum_rate,
        "cesr_types_any": cesr_any, "cesr_types_leak": cesr_leak,
        "n_true_pos": int(y_true.sum()), "n_pred_pos": int(y_pred.sum()), "n_samples": n,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--dpdd_root", default=DEFAULT_DPDD_ROOT)
    parser.add_argument("--canaries_csv", default=DEFAULT_CANARIES_CSV)
    parser.add_argument("--output_csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--summary_json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--report_md", default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    ensure_dir(args.output_csv)
    ensure_dir(args.summary_json)
    ensure_dir(args.report_md)

    canaries, df_can = load_canaries(args.canaries_csv)
    matcher = CanaryMatcher(canaries)
    baseline_runs = collect_runs(args.baseline_root, "baseline")
    dpdd_runs     = collect_runs(args.dpdd_root, "dpdd")
    all_runs      = baseline_runs + dpdd_runs

    if not all_runs:
        print("No runs found. Check your paths.")
        return

    per_run_rows = []
    print(f"Found {len(all_runs)} runs. Scoring...")

    for run in tqdm(all_runs, desc="Evaluating runs"):
        df_run = run["df"]
        rouge = compute_rouge_for_run(df_run)
        berts = compute_bertscore_for_run(df_run)
        priv  = privacy_metrics_for_run(df_run, matcher, canaries)
        row = {
            "system": run["system"],
            "run_dirname": run["run_dirname"],
            "run_path": run["run_path"],
            **run["hparams"],
            "n_samples": int(len(df_run)),
            **rouge,
            **berts,
            **priv,
        }
        per_run_rows.append(row)

    df_metrics = pd.DataFrame(per_run_rows)
    df_metrics.to_csv(args.output_csv, index=False)

    def _avg(df, cols):
        out = {}
        for c in cols:
            if c not in df.columns:
                out[c] = None
                continue
            col = pd.to_numeric(df[c], errors="coerce")
            out[c] = float(col.mean()) if col.notna().any() else None
        return out

    def _cond_avg(df, metric, denom_condition_col, denom_checker):
        if metric not in df.columns:
            return None
        mask = df.apply(lambda r: denom_checker(r), axis=1)
        col = pd.to_numeric(df.loc[mask, metric], errors="coerce")
        return float(col.mean()) if (len(col) > 0 and col.notna().any()) else None

    quality_cols = ["rouge1", "rouge2", "rougeL", "bertscore_precision", "bertscore_recall", "bertscore_f1"]
    privacy_cols_simple = ["precision", "recall", "fpr", "leak_rate", "copy_rate", "suppress_rate",
                           "any_summary_canary_rate", "cesr_types_any", "cesr_types_leak"]

    def denom_has_tp_fp(row): return (int(row.get("tp", 0)) + int(row.get("fp", 0))) > 0
    def denom_has_tp_fn(row): return (int(row.get("tp", 0)) + int(row.get("fn", 0))) > 0
    def denom_has_fp_tn(row): return (int(row.get("fp", 0)) + int(row.get("tn", 0))) > 0

    summary = {"global": {}, "global_conditional": {}, "by_hparam_combo": []}

    for system in ["baseline", "dpdd"]:
        sub = df_metrics[df_metrics["system"] == system]
        summary["global"][system] = {
            **_avg(sub, quality_cols + privacy_cols_simple),
            "sum_tp": int(pd.to_numeric(sub["tp"], errors="coerce").sum()) if "tp" in sub else 0,
            "sum_fp": int(pd.to_numeric(sub["fp"], errors="coerce").sum()) if "fp" in sub else 0,
            "sum_fn": int(pd.to_numeric(sub["fn"], errors="coerce").sum()) if "fn" in sub else 0,
            "sum_tn": int(pd.to_numeric(sub["tn"], errors="coerce").sum()) if "tn" in sub else 0,
        }
        summary["global_conditional"][system] = {
            "precision_cond": _cond_avg(sub, "precision", None, denom_has_tp_fp),
            "recall_cond":    _cond_avg(sub, "recall",    None, denom_has_tp_fn),
            "fpr_cond":       _cond_avg(sub, "fpr",       None, denom_has_fp_tn),
        }
    group_cols = ["learning_rate", "lora_rank", "retrieval_docs", "es_threshold", "year_group"]
    combos = df_metrics[group_cols].drop_duplicates().to_dict(orient="records")

    for combo in combos:
        mask = (df_metrics[group_cols] == pd.Series(combo)).all(axis=1)
        base = df_metrics[mask & (df_metrics["system"] == "baseline")]
        dpdd = df_metrics[mask & (df_metrics["system"] == "dpdd")]
        summary["by_hparam_combo"].append({
            **combo,
            "baseline": _avg(base, quality_cols + privacy_cols_simple),
            "dpdd":     _avg(dpdd, quality_cols + privacy_cols_simple),
        })

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    #arkdown comparison report
    def _pct_delta(b, d):
        if b is None or d is None or b == 0:
            return None
        try:
            return 100.0 * (d - b) / abs(b)
        except Exception:
            return None

    def _mk_compare_rows(keys: List[str], glob_base: dict, glob_dpdd: dict):
        rows = []
        for k in keys:
            b = glob_base.get(k)
            d = glob_dpdd.get(k)
            rows.append({
                "metric": k,
                "baseline": None if b is None else round(b, 6),
                "dpdd": None if d is None else round(d, 6),
                "delta_dpdd_minus_baseline": None if (b is None or d is None) else round(d - b, 6),
                "pct_change_%": None if _pct_delta(b, d) is None else round(_pct_delta(b, d), 2),
            })
        return pd.DataFrame(rows)

    glob_base = summary["global"]["baseline"]
    glob_dpdd = summary["global"]["dpdd"]
    cond_base = summary["global_conditional"]["baseline"]
    cond_dpdd = summary["global_conditional"]["dpdd"]

    focus_quality = ["rougeL", "bertscore_f1"]
    focus_privacy = ["precision", "recall", "fpr", "leak_rate", "copy_rate", "suppress_rate",
                     "any_summary_canary_rate", "cesr_types_any", "cesr_types_leak"]

    df_quality = _mk_compare_rows(focus_quality, glob_base, glob_dpdd)
    df_privacy = _mk_compare_rows(focus_privacy, glob_base, glob_dpdd)

    df_cond = pd.DataFrame([
        {"metric": "precision_cond", "baseline": cond_base["precision_cond"], "dpdd": cond_dpdd["precision_cond"]},
        {"metric": "recall_cond",    "baseline": cond_base["recall_cond"],    "dpdd": cond_dpdd["recall_cond"]},
        {"metric": "fpr_cond",       "baseline": cond_base["fpr_cond"],       "dpdd": cond_dpdd["fpr_cond"]},
    ])

    md_lines = []
    md_lines.append("# Baseline vs DPDD — Comparison Report (Enhanced)\n")
    md_lines.append("## Global Averages — Quality (selected)\n")
    md_lines.append(df_quality.to_markdown(index=False))
    md_lines.append("\n\n## Global Averages — Privacy (selected)\n")
    md_lines.append(df_privacy.to_markdown(index=False))
    md_lines.append("\n\n## Conditional Macro Averages (avoid denom=0 bias)\n")
    md_lines.append(df_cond.to_markdown(index=False))
    md_lines.append("\n\n## Per Hyperparameter Combo (means over runs)\n")
    if summary["by_hparam_combo"]:
        cols = ["learning_rate", "lora_rank", "retrieval_docs", "es_threshold", "year_group"]
        rows = []
        for item in summary["by_hparam_combo"]:
            base = item["baseline"]; dp = item["dpdd"]
            rows.append({
                **{k: item[k] for k in cols},
                "baseline_rougeL": None if base.get("rougeL") is None else round(base["rougeL"], 4),
                "dpdd_rougeL":     None if dp.get("rougeL")   is None else round(dp["rougeL"], 4),
                "base_prec":       None if base.get("precision") is None else round(base["precision"], 4),
                "dpdd_prec":       None if dp.get("precision")   is None else round(dp["precision"], 4),
                "base_fpr":        None if base.get("fpr") is None else round(base["fpr"], 4),
                "dpdd_fpr":        None if dp.get("fpr")   is None else round(dp["fpr"], 4),
                "base_cesr_leak":  None if base.get("cesr_types_leak") is None else round(base["cesr_types_leak"], 4),
                "dpdd_cesr_leak":  None if dp.get("cesr_types_leak")   is None else round(dp["cesr_types_leak"], 4),
            })
        md_lines.append(pd.DataFrame(rows).to_markdown(index=False))
    else:
        md_lines.append("_No per-combo results available._")

    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

if __name__ == "__main__":
    main()
