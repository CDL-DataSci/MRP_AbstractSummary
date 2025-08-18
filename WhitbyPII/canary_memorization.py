"""
Evaluate memorization by matching injected canaries.
- calculates per-run and global memorization metrics:
    * any_summary_canary_rate
    * n_summaries_any_hit
    * cesr_types_any
    * n_canary_types_hit
    * src_has_canary_rate
"""

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

DEFAULT_BASELINE_ROOT = "summaries_output/baseline"
DEFAULT_DPDD_ROOT     = "summaries_output/dpdd"
DEFAULT_CANARIES_CSV  = "data/canaries.csv"
DEFAULT_OUTPUT_DIR    = "evaluate"

RUN_DIR_RE = re.compile(
    r"lr(?P<learning_rate>[\d.]+)_r(?P<lora_rank>\d+)_doc(?P<retrieval_docs>\d+)_th(?P<es_threshold>[\d.]+)_yr(?P<year_group>[^_]+)_rep(?P<replication>\d+)"
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

#Text normalization & matching

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

def collect_clean_runs(root: str, system_tag: str):
    """
    Collect ONLY regenerated 'clean' summaries:
      <root>/**/_clean/generated_summaries.jsonl
    """
    results = []
    pattern = os.path.join(root, "**", "_clean", "generated_summaries.jsonl")
    for path in glob(pattern, recursive=True):
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
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

#Metrics

def run_memorization_metrics(df_run: pd.DataFrame,
                             matcher: CanaryMatcher,
                             canaries: List[Canary],
                             max_examples: int = 0):
    
    # Lists
    src = df_run.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    gen = df_run.get("generated_summary", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    n = min(len(src), len(gen))
    src, gen = src[:n], gen[:n]

    
    src_flags = [1 if matcher.any_canary_in_text(s) else 0 for s in src]
    src_has_canary_rate = float(np.mean(src_flags)) if n else None

#find any canary
    gen_flags = [1 if matcher.any_canary_in_text(g) else 0 for g in gen]
    any_summary_canary_rate = float(np.mean(gen_flags)) if n else None
    n_summaries_any_hit = int(np.sum(gen_flags))

#CESR
    def _canary_in_any_summary(c: Canary) -> bool:
        for g in gen:
            if matcher.appears_in_text(c, g):
                return True
        return False

    any_hits = []
    for c in canaries:
        if _canary_in_any_summary(c):
            any_hits.append(c.id)

    n_canary_types_hit = len(set(any_hits))
    total_canary_types = max(1, len(canaries))
    cesr_types_any = n_canary_types_hit / total_canary_types

    examples = []
    if max_examples > 0 and n > 0:
        count = 0
        for i, (s, g) in enumerate(zip(src, gen)):
            if matcher.any_canary_in_text(g):
                examples.append({
                    "idx": i,
                    "source_excerpt": s[:400],
                    "summary_excerpt": g[:400]
                })
                count += 1
                if count >= max_examples:
                    break

    metrics = {
        "n_samples": n,
        "any_summary_canary_rate": any_summary_canary_rate,
        "n_summaries_any_hit": n_summaries_any_hit,
        "cesr_types_any": cesr_types_any,
        "n_canary_types_hit": n_canary_types_hit,
        "total_canary_types": total_canary_types,
        "src_has_canary_rate": src_has_canary_rate,
    }
    return metrics, examples

#Pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--dpdd_root",     default=DEFAULT_DPDD_ROOT)
    parser.add_argument("--canaries_csv",  default=DEFAULT_CANARIES_CSV)
    parser.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_examples_per_run", type=int, default=5,
                        help="Write up to N example hits per run (0 disables).")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "examples"))

    #Load canaries & matcher
    canaries, df_can = load_canaries(args.canaries_csv)
    matcher = CanaryMatcher(canaries)

    baseline_runs = collect_clean_runs(args.baseline_root, "baseline")
    dpdd_runs     = collect_clean_runs(args.dpdd_root, "dpdd")
    all_runs      = baseline_runs + dpdd_runs

    if not all_runs:
        print("No clean runs found. Check your paths.")
        return

    per_run_rows = []
    print(f"Found {len(all_runs)} clean runs. Scoring memorization...")

    for run in tqdm(all_runs, desc="Evaluating (clean)"):
        df_run = run["df"]

        metrics, examples = run_memorization_metrics(
            df_run, matcher, canaries, max_examples=args.max_examples_per_run
        )

        row = {
            "system": run["system"],
            "run_dirname": run["run_dirname"],
            "run_path": run["run_path"],
            **run["hparams"],
            **metrics,
        }
        per_run_rows.append(row)

        if args.max_examples_per_run > 0 and examples:
            ex_path = os.path.join(args.output_dir, "examples", f"{run['run_dirname']}.jsonl")
            with open(ex_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    df_metrics = pd.DataFrame(per_run_rows)
    out_csv   = os.path.join(args.output_dir, "clean_canary_metrics_by_run.csv")
    df_metrics.to_csv(out_csv, index=False)
    print(f"✅ Wrote per-run clean memorization metrics to {out_csv}")

    def _avg(df, cols):
        out = {}
        for c in cols:
            if c not in df.columns:
                out[c] = None
                continue
            col = pd.to_numeric(df[c], errors="coerce")
            out[c] = float(col.mean()) if col.notna().any() else None
        return out

    summary = {"global": {}, "by_hparam_combo": []}
    key_cols = ["any_summary_canary_rate", "cesr_types_any", "src_has_canary_rate"]

    for system in ["baseline", "dpdd"]:
        sub = df_metrics[df_metrics["system"] == system]
        summary["global"][system] = {
            **_avg(sub, key_cols),
            "sum_n_samples": int(pd.to_numeric(sub["n_samples"], errors="coerce").sum()) if "n_samples" in sub else 0,
            "sum_n_summaries_any_hit": int(pd.to_numeric(sub["n_summaries_any_hit"], errors="coerce").sum()) if "n_summaries_any_hit" in sub else 0,
            "avg_n_canary_types_hit": float(pd.to_numeric(sub["n_canary_types_hit"], errors="coerce").mean()) if "n_canary_types_hit" in sub else None,
        }

    group_cols = ["learning_rate", "lora_rank", "retrieval_docs", "es_threshold", "year_group"]
    combos = df_metrics[group_cols].drop_duplicates().to_dict(orient="records")

    for combo in combos:
        mask = (df_metrics[group_cols] == pd.Series(combo)).all(axis=1)
        base = df_metrics[mask & (df_metrics["system"] == "baseline")]
        dpdd = df_metrics[mask & (df_metrics["system"] == "dpdd")]
        summary["by_hparam_combo"].append({
            **combo,
            "baseline": _avg(base, key_cols),
            "dpdd":     _avg(dpdd, key_cols),
        })

    summary_json = os.path.join(args.output_dir, "clean_canary_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote global summary to {summary_json}")

#Markdown Report
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

    glob_base = summary["global"].get("baseline", {})
    glob_dpdd = summary["global"].get("dpdd",    {})

    import pandas as _pd 
    focus_privacy = ["any_summary_canary_rate", "cesr_types_any", "src_has_canary_rate"]
    df_privacy = _mk_compare_rows(focus_privacy, glob_base, glob_dpdd)

    md_lines = []
    md_lines.append("# Clean-Set Memorization — Canary Matching Report\n")
    md_lines.append("This report checks **only clean generations** (`_clean/`) for any injected canary matches.\n")
    md_lines.append("## Global Averages — Memorization (selected)\n")
    md_lines.append(df_privacy.to_markdown(index=False))

    md_lines.append("\n\n## Per Hyperparameter Combo (means over runs)\n")
    if summary["by_hparam_combo"]:
        rows = []
        for item in summary["by_hparam_combo"]:
            base = item["baseline"]; dp = item["dpdd"]
            rows.append({
                "learning_rate": item.get("learning_rate"),
                "lora_rank": item.get("lora_rank"),
                "retrieval_docs": item.get("retrieval_docs"),
                "es_threshold": item.get("es_threshold"),
                "year_group": item.get("year_group"),
                "baseline_any_rate": None if base.get("any_summary_canary_rate") is None else round(base["any_summary_canary_rate"], 6),
                "dpdd_any_rate":     None if dp.get("any_summary_canary_rate")   is None else round(dp["any_summary_canary_rate"], 6),
                "baseline_cesr_any": None if base.get("cesr_types_any") is None else round(base["cesr_types_any"], 6),
                "dpdd_cesr_any":     None if dp.get("cesr_types_any")   is None else round(dp["cesr_types_any"], 6),
            })
        md_lines.append(_pd.DataFrame(rows).to_markdown(index=False))
    else:
        md_lines.append("_No per-combo results available._")

    report_md = os.path.join(args.output_dir, "clean_canary_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

if __name__ == "__main__":
    main()
