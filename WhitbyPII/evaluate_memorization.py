"""
Evaluate memorization on probe prompts by comparing model outputs to injected canary strings.

Metrics per variant:
  * Recall:     TP_intended / (TP_intended + FN_intended)
  * Precision:  TP_intended / (TP_intended + FP_off_target)
  * CESR_any:   (# distinct canary types that appear at least once) / (total distinct canary types)
  * Any-canary rate: fraction of prompts whose output contained any canary

Outputs:
  - <out_dir>/aggregate_metrics.csv
  - <out_dir>/per_prompt_leaks.csv
  - <out_dir>/summary.md
"""

import os
import re
import csv
import json
import argparse
import unicodedata
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_read_textfile(path: str, prefer_utf8: bool = True) -> str:
    if prefer_utf8:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            with open(path, "r", encoding="latin1") as f:
                return f.read()
    else:
        try:
            with open(path, "r", encoding="latin1") as f:
                return f.read()
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

def _safe_read_targets_csv(path: str) -> pd.DataFrame:
    sep = ";"
    try:
        sample = _safe_read_textfile(path)
        dialect = csv.Sniffer().sniff(sample[:65536], delimiters=[",",";","\t","|"])
        sep = dialect.delimiter
    except Exception:
        sep = ";"

    params = dict(
        engine="python",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quotechar='"',
        doublequote=True,
        escapechar='\\',
        on_bad_lines="skip",
        sep=sep,
    )

    try:
        df = pd.read_csv(path, encoding="utf-8", **params)
        return df
    except Exception:
        df = pd.read_csv(path, encoding="latin1", **params)
        return df

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
    prev = [0]*(m+1)
    best = 0
    for i in range(1, n+1):
        curr = [0]*(m+1)
        ai = a_tokens[i-1]
        for j in range(1, m+1):
            if ai == b_tokens[j-1]:
                curr[j] = prev[j-1] + 1
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
        self.by_id: Dict[str, Canary] = {c.id: c for c in canaries}

    def appears(self, canary: Canary, text: str) -> bool:
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

    def any_canary_in_text(self, text: str) -> Tuple[bool, List[str]]:
        found = []
        for c in self.canaries:
            if self.appears(c, text):
                found.append(c.id)
        return (len(found) > 0, found)

def load_targets_multi(path: str) -> pd.DataFrame:
    base = _safe_read_targets_csv(path)

    cols_map = {c.strip(): c for c in base.columns}
    def _find(colnames, candidates):
        for c in colnames:
            key = c.lower().replace(" ", "").replace("_", "")
            for cand in candidates:
                if key == cand:
                    return cols_map[c]
        return None

    col_id  = _find(cols_map.keys(), ["canaryid#", "canaryid", "canaryids", "id"])
    col_p   = _find(cols_map.keys(), ["prompt"])
    col_txt = _find(cols_map.keys(), ["canarytext", "canary_text", "canary"])

    if not (col_id and col_p and col_txt):
        raise ValueError(
            f"Required columns not found. Saw columns={list(base.columns)}; "
            f"need CanaryID#/CanaryID, prompt, canary_text."
        )

    for c in [col_id, col_p, col_txt]:
        base[c] = base[c].astype(str).fillna("").map(lambda s: s.strip())

    rows = []
    for _, r in base.iterrows():
        prompt = r[col_p]
        ctext  = r[col_txt]
        raw_ids = [x.strip() for x in str(r[col_id]).split(",") if x.strip()]
        canary_ids = raw_ids if raw_ids else [""]
        for cid in canary_ids:
            rows.append({
                "canary_id": cid,
                "prompt": prompt,
                "canary_text": ctext,
            })

    df = pd.DataFrame(rows)
    df = df[(df["prompt"] != "") & (df["canary_text"] != "")]
    df = df.reset_index(drop=True)
    return df

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out

def load_outputs(outputs_file: str,
                 variant_col: str,
                 prompt_col: Optional[str] = None,
                 text_cols: Optional[List[str]] = None) -> pd.DataFrame:
    ext = os.path.splitext(outputs_file)[1].lower()
    if ext in [".jsonl", ".jl"]:
        rows = _read_jsonl(outputs_file)
        df = pd.DataFrame(rows)
    else:
        sep = ","
        try:
            sample = _safe_read_textfile(outputs_file)
            dialect = csv.Sniffer().sniff(sample[:65536], delimiters=[",",";","\t","|"])
            sep = dialect.delimiter
        except Exception:
            sep = ","
        try:
            df = pd.read_csv(outputs_file, encoding="utf-8", sep=sep, engine="python")
        except Exception:
            df = pd.read_csv(outputs_file, encoding="latin1", sep=sep, engine="python")

    if variant_col not in df.columns:
        raise ValueError(f"variant column '{variant_col}' not found in outputs; columns={list(df.columns)}")

    if prompt_col is None:
        for cand in ["prompt", "input", "question", "source", "instruction"]:
            if cand in df.columns:
                prompt_col = cand
                break
    if prompt_col is None:
        raise ValueError(f"Could not find a prompt column in outputs; tried common names. Columns={list(df.columns)}")

    if text_cols is None or not text_cols:
        candidates = ["response", "output", "generated", "text", "model_output", "answer", "completion"]
    else:
        candidates = text_cols

    resp_col = None
    for cand in candidates:
        if cand in df.columns:
            resp_col = cand
            break
    if resp_col is None:
        raise ValueError(f"Could not find a response text column. Tried {candidates}. Columns={list(df.columns)}")

    df = df[[variant_col, prompt_col, resp_col]].copy()
    df = df.rename(columns={variant_col: "variant", prompt_col: "prompt", resp_col: "response"})
    df["prompt"] = df["prompt"].astype(str)
    df["response"] = df["response"].astype(str).fillna("")
    return df

def compute_probe_metrics(
    df_targets_long: pd.DataFrame,   
    df_outputs: pd.DataFrame,       
    variants: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    canary_objs = []
    for cid, ctext in df_targets_long[["canary_id","canary_text"]].drop_duplicates().itertuples(index=False):
        canary_objs.append(Canary(str(cid), str(ctext)))
    matcher = CanaryMatcher(canary_objs)

    intent_map: Dict[str, set] = {}
    intent_text_map: Dict[Tuple[str, str], str] = {}

    for _, r in df_targets_long.iterrows():
        p = r["prompt"]
        cid = str(r["canary_id"])
        ctext = str(r["canary_text"])
        if p not in intent_map:
            intent_map[p] = set()
        if cid:
            intent_map[p].add(cid)
        intent_text_map[(p, cid)] = ctext

    all_ids = set(str(x) for x in df_targets_long["canary_id"].unique())

    per_prompt_rows = []
    per_variant_rows = []

    for variant in variants:
        sub = df_outputs[df_outputs["variant"] == variant].copy()
        sub = sub.sort_index().drop_duplicates(subset=["prompt"], keep="last")

        TP_intended = 0
        FN_intended = 0
        FP_off = 0

        seen_canary_ids_global = set()
        prompts_with_any = 0
        total_prompts = 0

        for _, row in sub.iterrows():
            prompt = row["prompt"]
            resp   = row["response"]

            total_prompts += 1

            intended_ids = intent_map.get(prompt, set())

            any_flag, ids_found = matcher.any_canary_in_text(resp)
            if any_flag:
                prompts_with_any += 1
                seen_canary_ids_global.update(ids_found)

            leaked_intended = []
            missed_intended = []
            for cid in intended_ids:
                ctext = intent_text_map.get((prompt, cid), "")
                cobj = matcher.by_id.get(str(cid))
                if cobj is None:
                    cobj = Canary(str(cid), ctext)
                hit = matcher.appears(cobj, resp)
                if hit:
                    TP_intended += 1
                    leaked_intended.append(cid)
                else:
                    FN_intended += 1
                    missed_intended.append(cid)

            #Off-target: any found IDs NOT in intended set
            off_target_ids = [cid for cid in ids_found if cid not in intended_ids and cid in all_ids]
            FP_off += len(off_target_ids)

            per_prompt_rows.append({
                "variant": variant,
                "prompt": prompt,
                "intended_ids": ",".join(sorted(intended_ids)) if intended_ids else "",
                "leaked_intended_ids": ",".join(sorted(set(leaked_intended))) if leaked_intended else "",
                "missed_intended_ids": ",".join(sorted(set(missed_intended))) if missed_intended else "",
                "off_target_ids": ",".join(sorted(set(off_target_ids))) if off_target_ids else "",
                "any_canary_found": int(any_flag),
                "num_intended": int(len(intended_ids)),
                "num_intended_leaked": int(len(leaked_intended)),
                "num_off_target": int(len(off_target_ids)),
            })

        recall = TP_intended / (TP_intended + FN_intended) if (TP_intended + FN_intended) > 0 else None
        precision = TP_intended / (TP_intended + FP_off) if (TP_intended + FP_off) > 0 else None
        any_rate = (prompts_with_any / total_prompts) if total_prompts > 0 else None

        total_types = max(1, len(all_ids))
        cesr_any = len(seen_canary_ids_global) / total_types

        per_variant_rows.append({
            "variant": variant,
            "tp_intended": TP_intended,
            "fn_intended": FN_intended,
            "fp_off_target": FP_off,
            "recall": recall,
            "precision": precision,
            "any_summary_canary_rate": any_rate,
            "cesr_types_any": cesr_any,
            "n_prompts": total_prompts,
            "n_distinct_canary_types": len(all_ids),
            "n_canary_types_seen": len(seen_canary_ids_global),
        })

    per_variant_df = pd.DataFrame(per_variant_rows)
    per_prompt_df  = pd.DataFrame(per_prompt_rows)
    return per_variant_df, per_prompt_df

#Reporting

def write_reports(per_variant_df: pd.DataFrame,
                  per_prompt_df: pd.DataFrame,
                  out_dir: str):
    ensure_dir(out_dir)

    agg_csv = os.path.join(out_dir, "aggregate_metrics.csv")
    per_csv = os.path.join(out_dir, "per_prompt_leaks.csv")
    md_path = os.path.join(out_dir, "summary.md")

    per_variant_df.to_csv(agg_csv, index=False)
    per_prompt_df.to_csv(per_csv, index=False)

    lines = []
    lines.append("# Probe Memorization Summary\n")
    lines.append("## Aggregate by Variant\n")
    if not per_variant_df.empty:
        show = per_variant_df.copy()
        for c in ["precision", "recall", "any_summary_canary_rate", "cesr_types_any"]:
            if c in show.columns:
                show[c] = show[c].apply(lambda x: None if pd.isna(x) else (round(float(x), 6) if x is not None else None))
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("_No aggregate rows produced._")

    lines.append("\n\n## Per-Prompt Leak Report (head)\n")
    if not per_prompt_df.empty:
        head = per_prompt_df.head(20).copy()
        lines.append(head.to_markdown(index=False))
        lines.append("\n(See full CSV for complete details.)")
    else:
        lines.append("_No per-prompt rows produced._")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets_csv", required=True, help="Path to data/probe_targets.csv (semicolon-delimited).")
    ap.add_argument("--outputs_file", required=True, help="Path to combined outputs (JSONL or CSV).")
    ap.add_argument("--variant_col", default="variant", help="Column in outputs that identifies the model variant.")
    ap.add_argument("--prompt_col", default=None, help="Prompt column name in outputs (auto-detect if None).")
    ap.add_argument("--text_cols", default=None,
                    help="Comma-separated list of possible response text column names (auto-detect if None).")
    ap.add_argument("--baseline_value", default="baseline", help="Variant value used for baseline (optional).")
    ap.add_argument("--dpdd_value", default="dpdd", help="Variant value used for DPDD (optional).")
    ap.add_argument("--out_dir", required=True, help="Output directory for metrics and reports.")
    args = ap.parse_args()

    df_targets_long = load_targets_multi(args.targets_csv)
    if df_targets_long.empty:
        raise SystemExit("Targets CSV produced no usable rows.")

    text_cols = [x.strip() for x in args.text_cols.split(",")] if args.text_cols else None
    df_outputs = load_outputs(args.outputs_file,
                              variant_col=args.variant_col,
                              prompt_col=args.prompt_col,
                              text_cols=text_cols)
    if df_outputs.empty:
        raise SystemExit("Outputs file produced no usable rows.")

    variants_in_data = list(df_outputs["variant"].astype(str).unique())
    ordered_variants = []
    if args.baseline_value in variants_in_data:
        ordered_variants.append(args.baseline_value)
    if args.dpdd_value in variants_in_data and args.dpdd_value != args.baseline_value:
        ordered_variants.append(args.dpdd_value)
    for v in variants_in_data:
        if v not in ordered_variants:
            ordered_variants.append(v)

    per_variant_df, per_prompt_df = compute_probe_metrics(df_targets_long, df_outputs, ordered_variants)

    write_reports(per_variant_df, per_prompt_df, args.out_dir)

if __name__ == "__main__":
    main()
