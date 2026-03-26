import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score
RDLogger.DisableLog("rdApp.*")

def load_compounds(path):
    df = pd.read_csv(path)
    records, n_bad = [], 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None: n_bad += 1; continue
        try:
            pic50 = float(row["pic50"])
        except (KeyError, ValueError):
            continue
        if np.isnan(pic50): continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        records.append({"pic50": pic50, "fp": list(fp)})
    print(f"  {len(records)} valid ({n_bad} skipped)")
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_compounds(args.input)
    X = np.array(df["fp"].tolist())
    y = df["pic50"].values

    rf_base = RandomForestRegressor(n_estimators=200, n_jobs=-1)

    # LOO-CV (single, deterministic)
    loo = LeaveOneOut()
    y_loo = np.zeros(len(y))
    rf_loo = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    for tr, te in loo.split(X):
        rf_loo.fit(X[tr], y[tr])
        y_loo[te] = rf_loo.predict(X[te])
    r2_loo = r2_score(y, y_loo)

    # K-fold with varying seeds
    results = []
    seeds = range(args.n_repeats)
    for k in [3, 5, 10]:
        for seed in seeds:
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            y_pred = np.zeros(len(y))
            for tr, te in kf.split(X):
                rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                rf.fit(X[tr], y[tr])
                y_pred[te] = rf.predict(X[te])
            results.append({"k": k, "seed": seed, "r2": r2_score(y, y_pred)})

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(args.output_dir, "cv_stability.csv"), index=False)
    print(f"Saved: {args.output_dir}/cv_stability.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_k = {3: "#4C72B0", 5: "#DD8452", 10: "#55A868"}
    for k in [3, 5, 10]:
        vals = res_df[res_df["k"] == k]["r2"].values
        ax.scatter([str(k)] * len(vals), vals, alpha=0.6, s=40, color=colors_k[k], label=f"{k}-fold")
        ax.plot([str(k)-0.3 if False else str(k)], [vals.mean()], "D", color=colors_k[k], markersize=10, zorder=5)
    # LOO line
    ax.axhline(r2_loo, color="k", linestyle="--", lw=2, label=f"LOO-CV (R²={r2_loo:.3f})")
    ax.set_xlabel("K (number of folds)", fontsize=11)
    ax.set_ylabel("R² (cross-validated)", fontsize=11)
    ax.set_title(f"CV Stability ({args.n_repeats} random seeds per K)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cv_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output_dir}/cv_stability.png")

    print(f"\n--- CV Stability Results ({args.n_repeats} seeds) ---")
    print(f"  LOO-CV R²: {r2_loo:.4f} (deterministic)")
    for k in [3, 5, 10]:
        vals = res_df[res_df["k"] == k]["r2"].values
        print(f"  {k}-fold:    R²={vals.mean():.3f} ± {vals.std():.3f}  [min={vals.min():.3f}, max={vals.max():.3f}]")
    print("\nDone.")

if __name__ == "__main__":
    main()
