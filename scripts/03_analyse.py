"""Step 3: gather the FEP results and compare them with experiment and with Project 1.

`openfe gather` gives MLE ΔG per ligand (centred on 0) and ΔΔG per edge. ΔG values are
shifted to the experimental mean before computing MAE/RMSE (correlations are unaffected).
Writes results/fep_summary.csv, results/fep_stats.csv, results/edges_ddg.csv and
figures/fig_methods_comparison.png (docking vs MM/GBSA vs FEP if Project 1 results exist).
"""
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROJECT1 = ROOT.parent / "tyk2-docking-md-mmgbsa"
R_KCAL = 0.0019872043
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [("vina_score", "Docking (Vina)", "#2a78d6", "Vina score (kcal/mol)"),
          ("dG_mmgbsa", "MD + MM/GBSA", "#eb6834", "MM/GBSA ΔG (kcal/mol)"),
          ("dG_fep", "Alchemical FEP (OpenFE)", "#1baf7a", "FEP ΔG (kcal/mol)")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def experimental_dg(temperature):
    data = yaml.safe_load((ROOT / "inputs" / "benchmark" / "ligands.yml").read_text())
    scale = {"uM": 1e-6, "nM": 1e-9}
    return {n: R_KCAL * temperature * math.log(e["measurement"]["value"] * scale[e["measurement"]["unit"]])
            for n, e in data.items()}


def gather(report, out):
    subprocess.run(["openfe", "gather", str(ROOT / "work" / "results"), "--report", report,
                    "--allow-partial", "-o", str(out)], check=True)
    return pd.read_csv(out, sep="\t")


def bootstrap(x, y, n_boot=10000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    rs, rhos = [], []
    for idx in rng.integers(0, len(x), size=(n_boot, len(x))):
        if np.ptp(x[idx]) and np.ptp(y[idx]):
            rs.append(stats.pearsonr(x[idx], y[idx])[0])
            rhos.append(stats.spearmanr(x[idx], y[idx])[0])
    return {"n": len(x), "pearson_r": stats.pearsonr(x, y)[0],
            "r_ci_low": np.percentile(rs, 2.5), "r_ci_high": np.percentile(rs, 97.5),
            "spearman_rho": stats.spearmanr(x, y)[0],
            "rho_ci_low": np.percentile(rhos, 2.5), "rho_ci_high": np.percentile(rhos, 97.5)}


def cycle_closures(edges):
    """Signed sum of ΔΔG around each independent cycle of the network (ideal: 0)."""
    g = nx.DiGraph()
    for e in edges.itertuples():
        g.add_edge(e.ligand_i, e.ligand_j, ddg=e.ddg_fep)
    closures = []
    for cycle in nx.cycle_basis(g.to_undirected()):
        total = 0.0
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            total += g[a][b]["ddg"] if g.has_edge(a, b) else -g[b][a]["ddg"]
        closures.append(abs(total))
    return closures


def analyse(dg, ddg, exp):
    dg = dg.iloc[:, :3].set_axis(["ligand", "dG_fep_raw", "dG_fep_err"], axis=1)
    dg["dG_exp"] = dg["ligand"].map(exp)
    dg["dG_fep"] = dg["dG_fep_raw"] - dg["dG_fep_raw"].mean() + dg["dG_exp"].mean()
    err = dg["dG_fep"] - dg["dG_exp"]
    st = bootstrap(dg["dG_exp"], dg["dG_fep"])
    st.update(mae=err.abs().mean(), rmse=math.sqrt((err ** 2).mean()))

    edges = ddg.iloc[:, :4].set_axis(["ligand_i", "ligand_j", "ddg_fep", "ddg_err"], axis=1)
    edges["ddg_exp"] = edges["ligand_j"].map(exp) - edges["ligand_i"].map(exp)
    st["edge_mae"] = (edges["ddg_fep"] - edges["ddg_exp"]).abs().mean()
    closures = cycle_closures(edges)
    st["cycles"], st["mean_cycle_closure"] = len(closures), (np.mean(closures) if closures else np.nan)
    return dg, edges, st


def panel(ax, x, y, yerr, title, color, ylabel, st, diagonal):
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=INK_2, elinewidth=1, zorder=2)
    ax.scatter(x, y, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    lo, hi = min(x.min(), y.min() if diagonal else x.min()) - 0.5, max(x.max(), y.max() if diagonal else x.max()) + 0.5
    if diagonal:  # same scale: show y = x and a ±1 kcal/mol band
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color=GRID, alpha=0.6, zorder=0)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        slope, icpt = np.polyfit(x, y, 1)
        ax.plot([x.min() - .2, x.max() + .2], [slope * (x.min() - .2) + icpt, slope * (x.max() + .2) + icpt],
                color=INK_2, linewidth=1, linestyle="--", zorder=1)
    extra = f"\nMAE = {st['mae']:.2f}, RMSE = {st['rmse']:.2f} kcal/mol" if "mae" in st else ""
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=42)  # same height in every panel
    ax.text(0, 1.02, f"Pearson r = {st['pearson_r']:.2f}  [{st['r_ci_low']:.2f}, {st['r_ci_high']:.2f}]\n"
            f"Spearman ρ = {st['spearman_rho']:.2f}  [{st['rho_ci_low']:.2f}, {st['rho_ci_high']:.2f}]"
            + extra, transform=ax.transAxes, va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Experimental ΔG (kcal/mol)")
    ax.set_ylabel(ylabel)


def comparison_figure(dg, st_fep, out):
    df = dg.set_index("ligand")
    p1 = PROJECT1 / "results" / "summary.csv"
    if p1.exists():
        df = df.join(pd.read_csv(p1).set_index("ligand")[["vina_score", "dG_mmgbsa", "dG_mmgbsa_sem"]])
    cols = [p for p in PANELS if p[0] in df and df[p[0]].notna().sum() >= 3]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.8), squeeze=False)
    for ax, (col, title, color, ylabel) in zip(axes[0], cols):
        sub = df.dropna(subset=[col])
        st = st_fep if col == "dG_fep" else bootstrap(sub["dG_exp"], sub[col])
        yerr = sub.get("dG_fep_err") if col == "dG_fep" else sub.get("dG_mmgbsa_sem") if col == "dG_mmgbsa" else None
        panel(ax, sub["dG_exp"], sub[col], yerr, title, color, ylabel, st, diagonal=col == "dG_fep")
    fig.suptitle(f"TYK2, {len(df)} ligands: three ways to rank binders", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    exp = experimental_dg(cfg["temperature_K"])
    (ROOT / "results").mkdir(exist_ok=True)
    dg = gather("dg", ROOT / "results" / "openfe_dg.tsv")
    ddg = gather("ddg", ROOT / "results" / "openfe_ddg.tsv")
    dg, edges, st = analyse(dg, ddg, exp)
    dg.to_csv(ROOT / "results" / "fep_summary.csv", index=False, float_format="%.3f")
    edges.to_csv(ROOT / "results" / "edges_ddg.csv", index=False, float_format="%.3f")
    pd.DataFrame([st]).to_csv(ROOT / "results" / "fep_stats.csv", index=False, float_format="%.3f")
    (ROOT / "figures").mkdir(exist_ok=True)
    comparison_figure(dg, st, ROOT / "figures" / "fig_methods_comparison.png")
    print(f"FEP vs experiment: r = {st['pearson_r']:.2f} [{st['r_ci_low']:.2f}, {st['r_ci_high']:.2f}], "
          f"MAE = {st['mae']:.2f}, RMSE = {st['rmse']:.2f} kcal/mol; edge MAE = {st['edge_mae']:.2f}; "
          f"{st['cycles']} cycles, mean |closure| = {st['mean_cycle_closure']:.2f} kcal/mol")


if __name__ == "__main__":
    main()
