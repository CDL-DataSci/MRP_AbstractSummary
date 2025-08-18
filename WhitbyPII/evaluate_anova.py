import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def effect_sizes(anova_df: pd.DataFrame) -> pd.DataFrame:
    df = anova_df.copy()
    if "sum_sq" not in df.columns or "df" not in df.columns:
        df["eta_sq"] = np.nan
        df["omega_sq"] = np.nan
        return df

    total_ss = df["sum_sq"].sum(skipna=True)
    resid_row = "Residual" if "Residual" in df.index else df.index[-1]
    ms_error = (
        df.loc[resid_row, "mean_sq"]
        if "mean_sq" in df.columns
        else df.loc[resid_row, "sum_sq"] / df.loc[resid_row, "df"]
    )

    df["eta_sq"] = df["sum_sq"] / total_ss
    df["omega_sq"] = (df["sum_sq"] - df["df"] * ms_error) / (total_ss + ms_error)
    return df


def list_levels(df: pd.DataFrame, factor: str):
    vals = df[factor].dropna().astype(str)
    return sorted(vals.unique())


def build_formula(active_factors, add_interactions=True):
    main = [f"C({f})" for f in active_factors]

    interactions = []
    if add_interactions:
        pairs = [
            ("learning_rate", "lora_rank"),
            ("retrieval_docs", "es_threshold"),
            ("year_group", "system"),
        ]
        for a, b in pairs:
            if a in active_factors and b in active_factors:
                interactions.append(f"C({a}):C({b})")

    rhs = " + ".join(main + interactions) if (main or interactions) else "1"
    return rhs, interactions


def fit_anova_adaptive(df: pd.DataFrame, dv: str, prefer_type: int = 3, add_interactions=True):
    candidate_factors = ["system", "learning_rate", "lora_rank", "retrieval_docs", "es_threshold", "year_group"]
    keep_cols = [dv] + [c for c in candidate_factors if c in df.columns]
    work = df[keep_cols].dropna().copy()
    if work.empty:
        raise ValueError("No rows left after dropping NA in DV/factors.")

    if "system" in work.columns:
        work["system"] = work["system"].astype(str).str.strip().str.lower()
    if "year_group" in work.columns:
        work["year_group"] = work["year_group"].astype(str).str.strip()

#Determine which factors actually vary (>=2 levels)
    active = []
    for fac in candidate_factors:
        if fac in work.columns:
            levels = list_levels(work, fac)
            if len(levels) >= 2:
                active.append(fac)

    if not active:
        raise ValueError("No factors vary. Cannot run factorial ANOVA.")

    if work[dv].nunique(dropna=True) <= 1:
        raise ValueError(f"DV '{dv}' has no variance after filtering.")

    rhs, inter_terms = build_formula(active, add_interactions=add_interactions)
    formula = f"{dv} ~ {rhs}"

    model = ols(formula, data=work).fit()

    for typ in [prefer_type, 2] if prefer_type == 3 else [2, 3]:
        try:
            tbl = anova_lm(model, typ=typ)
            tbl["mean_sq"] = tbl["sum_sq"] / tbl["df"]
            tbl = effect_sizes(tbl)
            return model, tbl, formula, work, typ, active, inter_terms
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Error: {last_err}")


def robust_ols_fallback(df: pd.DataFrame, dv: str, active_factors, add_interactions=True):
    rhs, _ = build_formula(active_factors, add_interactions=add_interactions)
    formula = f"{dv} ~ {rhs}"
    model = ols(formula, data=df).fit(cov_type="HC3")  # robust SE
    coef = model.summary2().tables[1]
    return model, coef, formula


def summarize_anova_md(metric: str, anova_tbl: pd.DataFrame, formula: str, typ: int,
                       out_csv: str, active_factors, interactions):
    tbl = anova_tbl.copy()
    for col in ["sum_sq", "mean_sq", "F", "PR(>F)", "eta_sq", "omega_sq"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].astype(float).round(6)
    md = []
    md.append(f"# ANOVA for {metric}\n")
    md.append(f"**ANOVA type:** Type-{typ}\n")
    md.append("**Model formula**  \n")
    md.append(f"`{formula}`\n")
    md.append("\n**Active factors (varying in data):** " +
              ", ".join(active_factors) + "\n")
    if interactions:
        md.append("**Included interactions:** " + ", ".join(interactions) + "\n")
    else:
        md.append("**Included interactions:** (none)\n")
    md.append("\n**ANOVA table with effect sizes**\n")
    md.append(tbl.to_markdown())
    md.append("\n\n**Notes**\n")
    md.append("- η² (`eta_sq`) = proportion of total variance explained (biased upward).")
    md.append("- ω² (`omega_sq`) = less biased effect size; prefer for reporting.")
    md.append(f"\n\n*Full CSV saved to:* `{out_csv}`\n")
    return "\n".join(md)


def main_effects_plot(df: pd.DataFrame, dv: str, out_png: str):
    factors = ["system", "learning_rate", "lora_rank", "retrieval_docs", "es_threshold", "year_group"]
    # Only plot factors that exist
    factors = [f for f in factors if f in df.columns]
    if not factors:
        return
    n = len(factors)
    plt.figure(figsize=(min(16, 4 * n), 4))
    for i, fac in enumerate(factors, start=1):
        plt.subplot(1, n, i)
        grp = df.groupby(fac)[dv].agg(["mean", "count", "std"])
        grp["se"] = grp["std"] / np.sqrt(grp["count"])
        grp = grp.sort_index(key=lambda x: x.map(str))
        plt.errorbar(range(len(grp)), grp["mean"], yerr=grp["se"], fmt='o')
        plt.xticks(range(len(grp)), grp.index.astype(str), rotation=45, ha="right")
        plt.title(f"{fac}\nmean ± 1 SE")
        plt.xlabel("")
        plt.ylabel(dv)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive factorial ANOVA over WhitbyPII experiment factors.")
    p.add_argument("--csv_path", default="evaluate/metrics_by_run.csv", help="Path to metrics_by_run.csv")
    p.add_argument("--out_dir", default="evaluate/anova", help="Directory to write outputs")
    p.add_argument("--metrics", nargs="+",
                   default=["rougeL", "bertscore_f1", "cesr_types_any", "any_summary_canary_rate"],
                   help="Dependent variables to analyze")
    p.add_argument("--anova_type", choices=[2,3], type=int, default=3,
                   help="Preferred ANOVA type (falls back automatically on failure)")
    p.add_argument("--no_interactions", action="store_true", help="Disable interaction terms")
    p.add_argument("--no_plots", action="store_true", help="Skip main-effects plots")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    df = pd.read_csv(args.csv_path)

    #Minimal sanity: expected columns
    candidate_factors = ["system", "learning_rate", "lora_rank", "retrieval_docs", "es_threshold", "year_group"]
    for c in candidate_factors:
        if c not in df.columns:
            print(f"[WARN] Column '{c}' missing in {args.csv_path}")

    df = coerce_numeric(df, ["learning_rate", "lora_rank", "retrieval_docs", "es_threshold"])

    if "system" in df.columns:
        df["system"] = df["system"].astype(str).str.strip().str.lower()
    if "year_group" in df.columns:
        df["year_group"] = df["year_group"].astype(str).str.strip()

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[WARN] Metric '{metric}' not found in CSV. Skipping.")
            continue

        print(f"\n=== {metric} ===")
        try:
            model, anova_tbl, formula, used_df, used_type, active_factors, interactions = fit_anova_adaptive(
                df, dv=metric, prefer_type=args.anova_type, add_interactions=not args.no_interactions
            )
            out_csv = os.path.join(args.out_dir, f"{metric}_anova.csv")
            out_md  = os.path.join(args.out_dir, f"{metric}_anova.md")
            anova_tbl.to_csv(out_csv)
            md_text = summarize_anova_md(metric, anova_tbl, formula, used_type, out_csv, active_factors, interactions)
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"✅ ANOVA ({'Type-III' if used_type==3 else 'Type-II'}) written:\n  - {out_csv}\n  - {out_md}")

            if not args.no_plots:
                out_png = os.path.join(args.out_dir, f"{metric}_effects.png")
                try:
                    main_effects_plot(used_df, metric, out_png)
                    print(f"🖼  Saved main-effects plot: {out_png}")
                except Exception as e:
                    print(f"[WARN] Plotting failed for '{metric}': {e}")

        except Exception as e:
            print(f"[WARN] ANOVA failed for '{metric}' ({e}).")
            try:
                work = df[[metric] + [c for c in candidate_factors if c in df.columns]].dropna().copy()
                if "system" in work.columns:
                    work["system"] = work["system"].astype(str).str.strip().str.lower()
                if "year_group" in work.columns:
                    work["year_group"] = work["year_group"].astype(str).str.strip()
                active = []
                for fac in candidate_factors:
                    if fac in work.columns:
                        if len(list_levels(work, fac)) >= 2:
                            active.append(fac)
                if not active:
                    raise ValueError("No varying factors available.")

                model, coef, formula = robust_ols_fallback(work, metric, active, add_interactions=not args.no_interactions)
                out_md  = os.path.join(args.out_dir, f"{metric}_robust_ols.md")
                out_csv = os.path.join(args.out_dir, f"{metric}_robust_ols.csv")
                coef.to_csv(out_csv)
                md = []
                md.append(f"# Robust OLS (HC3) for {metric}\n")
                md.append("**Model formula**  \n")
                md.append(f"`{formula}`\n")
                md.append("\n**Coefficient table (robust SE)**\n")
                md.append(coef.round(6).to_markdown())
                with open(out_md, "w", encoding="utf-8") as f:
                    f.write("\n".join(md))
            except Exception as e2:
                print(f"[ERROR] Could not fit robust OLS for '{metric}': {e2}")


if __name__ == "__main__":
    main()
