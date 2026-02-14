import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GROUP_ORDER = ["CU", "MCI", "Dementia"]
GENDER_ORDER = ["Men", "Women"]


def ensure_results_dir(path="results"):
    os.makedirs(path, exist_ok=True)


def standardize_gender(x):
    """
    Accepts common gender encodings:
    - 1/2 (ADNI-style) -> Men/Women
    - 'Male'/'Female' -> Men/Women
    - already 'Men'/'Women' -> unchanged
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and x in [1, 2]:
        return "Men" if int(x) == 1 else "Women"
    s = str(x).strip().lower()
    if s in ["1", "m", "male", "man", "men"]:
        return "Men"
    if s in ["2", "f", "female", "woman", "women"]:
        return "Women"
    # if it's something else, keep it as-is (could be "Unknown")
    return str(x)


def map_cdr_to_group(cdr):
    """
    Common mapping:
      0   -> CU
      0.5 -> MCI
      >=1 -> Dementia
    """
    if pd.isna(cdr):
        return np.nan
    try:
        c = float(cdr)
    except Exception:
        return np.nan
    if c == 0:
        return "CU"
    elif c == 0.5:
        return "MCI"
    else:
        return "Dementia"


def permutation_test_3groups(values, labels, n_perm=10000, seed=42):
    """
    Nonparametric permutation test across 3 groups using:
      statistic = variance of group means
    One-sided: p = P(stat_perm >= stat_obs)
    """
    rng = np.random.default_rng(seed)

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    # drop NaNs
    ok = ~np.isnan(values)
    values = values[ok]
    labels = labels[ok]

    # ensure all groups exist
    groups = {}
    for g in GROUP_ORDER:
        groups[g] = values[labels == g]
        if len(groups[g]) == 0:
            raise ValueError(f"Group '{g}' has 0 samples. Cannot run 3-group permutation test.")

    n_sizes = [len(groups[g]) for g in GROUP_ORDER]
    pooled = np.concatenate([groups[g] for g in GROUP_ORDER])

    obs_means = [np.mean(groups[g]) for g in GROUP_ORDER]
    obs_stat = np.var(obs_means)

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        a = 0
        fake_means = []
        for size in n_sizes:
            fake_means.append(np.mean(shuffled[a:a+size]))
            a += size
        perm_stats[i] = np.var(fake_means)

    p_value = np.mean(perm_stats >= obs_stat)

    return {
        "obs_means": obs_means,
        "obs_stat": float(obs_stat),
        "p_value": float(p_value),
        "perm_stats": perm_stats,
        "n_sizes": n_sizes,
    }


def save_perm_hist(perm_stats, obs_stat, title, out_png):
    plt.figure(figsize=(7, 4))
    plt.hist(perm_stats, bins=35)
    plt.axvline(obs_stat, linewidth=2)
    plt.title(title)
    plt.xlabel("Permutation statistic (variance of group means)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_group_box(df, value_col, out_png, title):
    # simple matplotlib boxplot by group (no seaborn dependency)
    data = [df.loc[df["cdr_group"] == g, value_col].dropna().values for g in GROUP_ORDER]

    plt.figure(figsize=(7, 4))
    plt.boxplot(data, tick_labels=GROUP_ORDER, showfliers=False)
    plt.title(title)
    plt.xlabel("Cognitive group")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/adni_participants_summary_modified.csv",
                        help="Path to the summary CSV (not included in repo).")
    parser.add_argument("--n_perm", type=int, default=10000, help="Number of permutations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--baseline_one_per_rid", action="store_true",
                        help="If set, keep one row per RID (first occurrence). Use if your file has repeats.")
    args = parser.parse_args()

    ensure_results_dir("results")

    # -----------------------------
    # Load
    # -----------------------------
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Data file not found: {args.data}\n"
            "Place your ADNI-derived summary CSV in the data/ folder (not tracked by git)."
        )

    df = pd.read_csv(args.data)

    # expected columns
    required = ["RID", "wmh", "cdr", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            "Your file must include: RID, wmh, cdr, gender"
        )

    # -----------------------------
    # Clean + features
    # -----------------------------
    if args.baseline_one_per_rid:
        df = df.drop_duplicates(subset="RID", keep="first").copy()

    df["wmh"] = pd.to_numeric(df["wmh"], errors="coerce")
    df["cdr"] = pd.to_numeric(df["cdr"], errors="coerce")
    df["gender"] = df["gender"].apply(standardize_gender)

    df["cdr_group"] = df["cdr"].apply(map_cdr_to_group)
    df = df.dropna(subset=["wmh", "cdr_group", "gender"]).copy()
    df = df[df["wmh"] >= 0].copy()

    # log transform (common for skewed volumes)
    df["log_wmh"] = np.log1p(df["wmh"])

    # keep only Men/Women for stratified part
    df_g = df[df["gender"].isin(GENDER_ORDER)].copy()

    # -----------------------------
    # Descriptives
    # -----------------------------
    desc = (
        df.groupby("cdr_group")["wmh"]
        .agg(["count", "mean", "median", "std"])
        .reindex(GROUP_ORDER)
    )

    desc_gender = (
        df_g.groupby(["cdr_group", "gender"])["wmh"]
        .agg(["count", "mean", "median", "std"])
        .reindex(pd.MultiIndex.from_product([GROUP_ORDER, GENDER_ORDER], names=["cdr_group", "gender"]))
    )

    # save descriptive tables
    desc.to_csv("results/descriptives_overall.csv")
    desc_gender.to_csv("results/descriptives_by_gender.csv")

    # quick figure: boxplot of log_wmh by group
    plot_group_box(
        df,
        value_col="log_wmh",
        out_png="results/box_log_wmh_by_group.png",
        title="log(WMH + 1) by Cognitive Group"
    )

    # -----------------------------
    # Permutation overall (on log_wmh)
    # -----------------------------
    overall = permutation_test_3groups(
        values=df["log_wmh"].values,
        labels=df["cdr_group"].values,
        n_perm=args.n_perm,
        seed=args.seed
    )
    save_perm_hist(
        overall["perm_stats"],
        overall["obs_stat"],
        title=f"Permutation test (overall) — log(WMH+1), n_perm={args.n_perm}",
        out_png="results/permutation_overall_hist.png"
    )

    # -----------------------------
    # Permutation by gender
    # -----------------------------
    gender_results = {}
    for g in GENDER_ORDER:
        sub = df_g[df_g["gender"] == g].copy()
        # if any group missing, skip with note
        if any(len(sub[sub["cdr_group"] == grp]) == 0 for grp in GROUP_ORDER):
            gender_results[g] = {"skipped": True, "reason": "One group has 0 samples."}
            continue

        res = permutation_test_3groups(
            values=sub["log_wmh"].values,
            labels=sub["cdr_group"].values,
            n_perm=args.n_perm,
            seed=args.seed + (1 if g == "Men" else 2)
        )
        gender_results[g] = res
        save_perm_hist(
            res["perm_stats"],
            res["obs_stat"],
            title=f"Permutation test ({g}) — log(WMH+1), n_perm={args.n_perm}",
            out_png=f"results/permutation_{g.lower()}_hist.png"
        )

    # -----------------------------
    # Write report
    # -----------------------------
    with open("results/run_report.txt", "w", encoding="utf-8") as f:
        f.write("WMH Dementia Permutation Analysis\n")
        f.write("=================================\n\n")
        f.write(f"Data file: {args.data}\n")
        f.write(f"Rows after cleaning: {len(df)}\n")
        f.write(f"Permutation count: {args.n_perm}\n\n")

        f.write("Overall descriptives (WMH):\n")
        f.write(desc.to_string())
        f.write("\n\n")

        f.write("Overall permutation test (log_wmh):\n")
        f.write(f"  Group sizes (CU, MCI, Dementia): {overall['n_sizes']}\n")
        f.write(f"  Observed group means: {overall['obs_means']}\n")
        f.write(f"  Observed statistic: {overall['obs_stat']:.6g}\n")
        f.write(f"  p-value: {overall['p_value']:.6g}\n\n")

        f.write("Permutation test by gender (log_wmh):\n")
        for g in GENDER_ORDER:
            f.write(f"- {g}:\n")
            gr = gender_results.get(g, {})
            if gr.get("skipped"):
                f.write(f"  Skipped: {gr.get('reason')}\n")
            else:
                f.write(f"  Group sizes: {gr['n_sizes']}\n")
                f.write(f"  Observed means: {gr['obs_means']}\n")
                f.write(f"  Observed stat: {gr['obs_stat']:.6g}\n")
                f.write(f"  p-value: {gr['p_value']:.6g}\n")
            f.write("\n")

        f.write("Outputs:\n")
        f.write("- results/box_log_wmh_by_group.png\n")
        f.write("- results/permutation_overall_hist.png\n")
        f.write("- results/permutation_men_hist.png\n")
        f.write("- results/permutation_women_hist.png\n")
        f.write("- results/descriptives_overall.csv\n")
        f.write("- results/descriptives_by_gender.csv\n")
        f.write("- results/run_report.txt\n")

    print("Done. See results/ folder.")


if __name__ == "__main__":
    main()