
"""
Create 3 visuals from Training_Log.csv:
1) Scatter: Validation Loss (mean) vs DPDD Score (mean), colored by ID#
2) Heatmap: Validation Loss (mean) with rows = "LoRA-Year Group", columns = Learning Rate
3) Heatmap: Validation Loss (mean) by full Run Configuration (sorted ascending)

Columns expected:
  Model | ID# | Run desc | Rep | Epoch | DPDD score | Loss (validation) | Data Dropped

"""

import os
import re
import argparse
import warnings
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


RUN_RE = re.compile(
    r"lr(?P<lr>[\d.]+)_r(?P<r>\d+)_doc(?P<doc>\d+)_th(?P<th>[\d.]+)_yr(?P<year>[^_]+)"
)

EXPECTED_COLS = [
    "Model", "ID#", "Run desc", "Rep", "Epoch", "DPDD score", "Loss (validation)", "Data Dropped"
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_training_csv(path: str) -> pd.DataFrame:
    
    def _read(sep):
        return pd.read_csv(path, sep=sep, engine="python")

    # try , then ;
    for sep in [",", ";"]:
        try:
            df = _read(sep)
            cols = [c.strip() for c in df.columns]
            df.columns = cols
            if set(EXPECTED_COLS).issubset(set(cols)):
                return df
        except Exception:
            pass

    df = pd.read_csv(path, engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    raise ValueError(f"Missing required columns in {path}: {missing}. Detected columns: {list(df.columns)}")

def parse_run_desc(s: str) -> Tuple[float, int, int, float, str]:
    """
    Extract lr, r, doc, th, year from 'Run desc'
    """
    if not isinstance(s, str):
        return np.nan, np.nan, np.nan, np.nan, None
    m = RUN_RE.match(s.strip())
    if not m:
        return np.nan, np.nan, np.nan, np.nan, None
    d = m.groupdict()
    try:
        lr = float(d["lr"])
    except Exception:
        lr = np.nan
    try:
        r = int(d["r"])
    except Exception:
        r = np.nan
    try:
        doc = int(d["doc"])
    except Exception:
        doc = np.nan
    try:
        th = float(d["th"])
    except Exception:
        th = np.nan
    year = d.get("year", None)
    return lr, r, doc, th, year

def make_scatter(df_cfg: pd.DataFrame, outpath: str):
    """
    Scatter: x = mean validation loss, y = mean DPDD score, one point per config.
    Points are colored by the most frequent ID# seen for that config.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    df_cfg = df_cfg.copy()
    df_cfg["id_mode"] = df_cfg["id_mode"].astype(str)

    unique_ids = sorted(df_cfg["id_mode"].dropna().unique())
    cmap = plt.get_cmap("tab20") 
    color_map = {uid: cmap(i % cmap.N) for i, uid in enumerate(unique_ids)}

    for uid, g in df_cfg.groupby("id_mode"):
        ax.scatter(
            g["loss_mean"],
            g["dpdd_mean"],
            s=70,
            alpha=0.9,
            color=color_map.get(uid, "gray"),
            edgecolors="black",
            linewidths=0.3,
            label=uid,
        )

    ax.set_title("Model Configurations: Loss vs DPDD Score", fontsize=22, pad=14)
    ax.set_xlabel("Validation Loss (mean)", fontsize=14)
    ax.set_ylabel("DPDD Score (mean)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)

    n = len(unique_ids)
    ncols = 1 if n <= 12 else 2 if n <= 24 else 3
    ax.legend(
        title="ID#",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        ncol=ncols,
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def make_heatmap_lora_year_lr(df_cfg: pd.DataFrame, outpath: str):
    """
    Heatmap of Validation Loss (mean) with rows = "LoRA-Year Group", columns = Learning Rate
    Values are aggregated mean across configs that share those three axes.
    """
    # Build row label "r-year"
    df_cfg = df_cfg.copy()
    df_cfg["row_label"] = df_cfg["lora"].astype(int).astype(str) + "-" + df_cfg["year_group"].astype(str)
    df_cfg["lr_str"] = df_cfg["learning_rate"].apply(lambda v: f"{v:.4f}".rstrip("0").rstrip(".") if pd.notna(v) else "nan")

    pivot = df_cfg.pivot_table(
        index="row_label",
        columns="lr_str",
        values="loss_mean",
        aggfunc="mean"
    )

    # order rows by lora then year
    pivot = pivot.sort_index()

    data = pivot.values
    plt.figure(figsize=(12, 7))
    im = plt.imshow(data, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Validation Loss")

    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=0)
    plt.yticks(range(pivot.shape[0]), pivot.index)

    mean_val = np.nanmean(data)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                txt = ""
                color = "black"
            else:
                txt = f"{val:.3f}"
                color = "white" if val > mean_val else "black"
            plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    plt.title("Validation Loss Heatmap (LoRA × Year Group × Learning Rate)", fontsize=16, pad=12)
    plt.xlabel("Learning Rate")
    plt.ylabel("LoRA × Year Group")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def make_heatmap_by_run(df_cfg: pd.DataFrame, outpath: str):
    """
    Heatmap with a single column (Validation Loss mean) and one row per full run configuration,
    sorted ascending by loss.
    """
    df_sorted = df_cfg.sort_values("loss_mean", ascending=True).reset_index(drop=True)
    labels = df_sorted["Run desc"].tolist()
    vals = df_sorted["loss_mean"].values.reshape(-1, 1)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(vals, cmap="viridis", aspect="auto")

    plt.yticks(range(len(labels)), labels)
    plt.xticks([0], ["Validation Loss"])

    mean_val = np.nanmean(vals)
    for i in range(len(labels)):
        v = vals[i, 0]
        txt = f"{v:.3f}" if not np.isnan(v) else ""
        color = "white" if (not np.isnan(v) and v > mean_val) else "black"
        plt.text(0, i, txt, ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, label="Validation Loss")
    plt.title("Validation Loss by Run Configuration", fontsize=18, pad=12)
    plt.xlabel("Validation Loss")
    plt.ylabel("Run Configuration")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Training_Log.csv", help="Path to Training_Log.csv")
    ap.add_argument("--outdir", default="evaluate/trainlog_viz", help="Directory to save figures")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load
    df = read_training_csv(args.csv)

    for col in ["DPDD score", "Loss (validation)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = df["Run desc"].apply(parse_run_desc)
    df["learning_rate"] = parsed.apply(lambda t: t[0])
    df["lora"]          = parsed.apply(lambda t: t[1])
    df["doc"]           = parsed.apply(lambda t: t[2])
    df["th"]            = parsed.apply(lambda t: t[3])
    df["year_group"]    = parsed.apply(lambda t: t[4])

    grp_cols = ["Run desc", "learning_rate", "lora", "doc", "th", "year_group"]

    def _mode_id(series: pd.Series):
        s = series.astype(str).dropna()
        return s.mode().iat[0] if not s.mode().empty else None

    df_cfg = (
        df.groupby(grp_cols, dropna=False)
          .agg(loss_mean=("Loss (validation)", "mean"),
               dpdd_mean=("DPDD score", "mean"),
               n=("ID#", "count"),
               id_mode=("ID#", _mode_id))  
          .reset_index()
    )

    make_scatter(df_cfg, os.path.join(args.outdir, "scatter_loss_vs_dpdd.png"))
    make_heatmap_lora_year_lr(df_cfg, os.path.join(args.outdir, "heatmap_lora_year_lr.png"))
    make_heatmap_by_run(df_cfg, os.path.join(args.outdir, "heatmap_by_run_config.png"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
