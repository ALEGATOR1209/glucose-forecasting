"""Generate publication-quality figures for RoBioinfo2026 presentation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "sugar": "#0891b2",
    "glumind_orig": "#059669",
    "nhits": "#f97316",
    "gluformer": "#8b5cf6",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# =============================================================================
# FIGURE 1: Training curve (loss vs epoch)
# =============================================================================
def parse_training_log(path):
    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []
    best_epochs = []
    with open(path) as f:
        for line in f:
            m = re.search(r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss=([\d.]+)\s+\|\s+val_loss=([\d.]+|SKIP)", line)
            if m:
                ep = int(m.group(1))
                tl = float(m.group(2))
                train_epochs.append(ep)
                train_losses.append(tl)
                vl = m.group(3)
                if vl != "SKIP":
                    val_epochs.append(ep)
                    val_losses.append(float(vl))
            if "New best" in line:
                bm = re.search(r"epoch (\d+)", line)
                if bm:
                    best_epochs.append(int(bm.group(1)))
    return (np.array(train_epochs), np.array(train_losses),
            np.array(val_epochs), np.array(val_losses), best_epochs)


def fig_training_curve():
    log_path = os.path.join(os.path.dirname(OUT_DIR), "..", "test_model", "tuning.txt")
    log_path = os.path.normpath(log_path)
    te, tl, ve, vl, best = parse_training_log(log_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(te, tl, color=COLORS["sugar"], linewidth=1.5, alpha=0.8, label="Train loss")
    ax.plot(ve, vl, color=COLORS["gluformer"], linewidth=2, marker="o", markersize=3, label="Val loss")

    for be in best:
        idx = np.where(ve == be)[0]
        if len(idx):
            ax.plot(be, vl[idx[0]], "v", color=COLORS["glumind_orig"], markersize=8, zorder=5)
    ax.plot([], [], "v", color=COLORS["glumind_orig"], markersize=8, label="New best checkpoint")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized space)")
    ax.set_title("Sugar I Training Convergence (AI-READI + T1DM, 4.45M windows)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, te[-1] + 1)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "fig_training_curve.png"))
    plt.close(fig)
    print("  -> fig_training_curve.png")


# =============================================================================
# FIGURE 2: Per-cohort MAE comparison (grouped bar chart)
# =============================================================================
def fig_per_cohort_mae():
    groups = ["Healthy", "Pre-T2DM", "T2DM\nOral", "T2DM\nInsulin", "T1DM"]
    sugar = [9.57, 9.89, 12.35, 13.58, 15.06]
    nhits = [16.86, 14.00, 19.97, 28.31, 15.53]
    gluformer = [17.08, 14.21, 19.36, 26.36, 15.46]
    glumind_orig = [10.58, 11.08, 13.74, 16.41, None]

    x = np.arange(len(groups))
    w = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - 1.5*w, sugar, w, color=COLORS["sugar"], label="Sugar I (Ours)", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x - 0.5*w, [v if v else 0 for v in glumind_orig], w, color=COLORS["glumind_orig"],
                   label="GluMind (orig. paper)", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + 0.5*w, nhits, w, color=COLORS["nhits"], label="NHITS", edgecolor="white", linewidth=0.5)
    bars4 = ax.bar(x + 1.5*w, gluformer, w, color=COLORS["gluformer"], label="GluFormer", edgecolor="white", linewidth=0.5)

    # hide the glumind_orig T1DM bar (no data)
    bars2[4].set_visible(False)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            if bar.get_visible() and bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("Clinical Cohort")
    ax.set_ylabel("MAE (mg/dL)  — lower is better")
    ax.set_title("Per-Cohort Prediction Error: Sugar I vs. Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 32)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)

    # add severity arrow
    ax.annotate("", xy=(4.6, 1.5), xytext=(-0.6, 1.5),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2))
    ax.text(2, 0.3, "increasing glycemic variability  →", ha="center", fontsize=10, color="#94a3b8", style="italic")

    fig.savefig(os.path.join(OUT_DIR, "fig_per_cohort_mae.png"))
    plt.close(fig)
    print("  -> fig_per_cohort_mae.png")


# =============================================================================
# FIGURE 3: Overall metrics comparison (T2DM — 3 metrics side by side)
# =============================================================================
def fig_overall_t2dm():
    metrics = ["MAE (mg/dL)", "RMSE (mg/dL)", "MARD (%)"]
    sugar_vals = [11.33, 17.73, 8.25]
    orig_vals = [12.95, 18.19, None]
    nhits_vals = [20.60, 34.45, 13.33]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        models = ["Sugar I\n(Ours)", "GluMind\n(orig.)", "NHITS"]
        vals = [sugar_vals[i], orig_vals[i], nhits_vals[i]]
        colors = [COLORS["sugar"], COLORS["glumind_orig"], COLORS["nhits"]]

        if vals[1] is None:
            models = [models[0], models[2]]
            vals = [vals[0], vals[2]]
            colors = [colors[0], colors[2]]

        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.2f}" if i < 2 else f"{val:.2f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("")

    fig.suptitle("AI-READI (T2DM + Prediabetes): 60-min Glucose Prediction", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "lower is better for all metrics", ha="center", fontsize=11, color="#94a3b8", style="italic")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overall_t2dm.png"))
    plt.close(fig)
    print("  -> fig_overall_t2dm.png")


# =============================================================================
# FIGURE 4: Overall metrics comparison (T1DM)
# =============================================================================
def fig_overall_t1dm():
    metrics = ["MAE (mg/dL)", "RMSE (mg/dL)", "MARD (%)"]
    sugar_vals = [14.51, 23.00, 10.99]
    nhits_vals = [15.11, 21.05, 11.24]
    gluformer_vals = [19.53, 33.28, 15.10]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        models = ["Sugar I\n(Ours)", "NHITS", "GluFormer"]
        vals = [sugar_vals[i], nhits_vals[i], gluformer_vals[i]]
        colors = [COLORS["sugar"], COLORS["nhits"], COLORS["gluformer"]]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.2f}" if i < 2 else f"{val:.2f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=0.3)

        # highlight RMSE where NHITS wins
        if i == 1:
            bars[1].set_edgecolor("#dc2626")
            bars[1].set_linewidth(2)

    fig.suptitle("Type 1 Diabetes (GlucoBench): 60-min Glucose Prediction", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "lower is better — red border = NHITS wins RMSE on this cohort", ha="center", fontsize=11, color="#94a3b8", style="italic")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overall_t1dm.png"))
    plt.close(fig)
    print("  -> fig_overall_t1dm.png")


# =============================================================================
# FIGURE 5: Error gradient — MAE increases with disease severity
# =============================================================================
def fig_error_gradient():
    groups = ["Healthy", "Pre-T2DM", "T2DM Oral", "T2DM Insulin", "T1DM"]
    mae = [9.57, 9.89, 12.35, 13.58, 15.06]
    gradient_colors = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(groups, mae, color=gradient_colors, edgecolor="white", linewidth=1, width=0.6)

    for bar, val in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Sugar I: Prediction Error Tracks Disease Severity", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 18)
    ax.grid(axis="y", alpha=0.3)

    ax.annotate("", xy=(4.4, 16.5), xytext=(-0.4, 16.5),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.5))
    ax.text(2, 17, "increasing glycemic variability", ha="center", fontsize=11, color="#64748b", style="italic")

    # target range band
    ax.axhspan(0, 10, alpha=0.05, color="green")

    fig.savefig(os.path.join(OUT_DIR, "fig_error_gradient.png"))
    plt.close(fig)
    print("  -> fig_error_gradient.png")


# =============================================================================
# FIGURE 6: Improvement heatmap — Sugar I vs baselines per cohort
# =============================================================================
def fig_improvement_heatmap():
    groups = ["Healthy", "Pre-T2DM", "Oral-T2DM", "Insulin-T2DM", "T1DM"]
    baselines = ["NHITS", "GluFormer", "GluMind (orig.)"]

    sugar_mae = np.array([9.57, 9.89, 12.35, 13.58, 15.06])
    nhits_mae = np.array([16.86, 14.00, 19.97, 28.31, 15.53])
    gluformer_mae = np.array([17.08, 14.21, 19.36, 26.36, 15.46])
    glumind_orig_mae = np.array([10.58, 11.08, 13.74, 16.41, np.nan])

    improvement = np.zeros((3, 5))
    improvement[0] = (nhits_mae - sugar_mae) / nhits_mae * 100
    improvement[1] = (gluformer_mae - sugar_mae) / gluformer_mae * 100
    improvement[2] = (glumind_orig_mae - sugar_mae) / glumind_orig_mae * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(improvement, cmap="YlGn", aspect="auto", vmin=0, vmax=55)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baselines)

    for i in range(len(baselines)):
        for j in range(len(groups)):
            val = improvement[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="#94a3b8")
            else:
                color = "white" if val > 35 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    ax.set_title("Sugar I: MAE Improvement Over Baselines (per cohort)", fontsize=14, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% MAE improvement", fontsize=11)

    fig.savefig(os.path.join(OUT_DIR, "fig_improvement_heatmap.png"))
    plt.close(fig)
    print("  -> fig_improvement_heatmap.png")


# =============================================================================
# FIGURE 7: Architecture summary diagram (simplified)
# =============================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#2563eb", linewidth=1.5)
    proc_style = dict(boxstyle="round,pad=0.4", facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.5)
    out_style = dict(boxstyle="round,pad=0.4", facecolor="#d1fae5", edgecolor="#059669", linewidth=1.5)
    block_style = dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1)

    # Input boxes
    inputs = [("CGM\n(5-min)", 1, 5), ("HR\n(1-min)", 3.5, 5), ("Steps\n(1-min)", 6, 5)]
    for txt, x, y in inputs:
        ax.text(x, y, txt, ha="center", va="center", fontsize=11, fontweight="bold", bbox=box_style)

    # Embedding
    ax.text(3.5, 3.8, "Per-channel Linear Embedding\n+ Positional Encoding", ha="center", va="center",
            fontsize=10, bbox=block_style)

    # Arrows from inputs to embedding
    for x, _ in [(1, 5), (3.5, 5), (6, 5)]:
        ax.annotate("", xy=(3.5, 4.15), xytext=(x, 4.65),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

    # Parallel blocks
    ax.text(2, 2.5, "Cross-Attention\n(sensor fusion)", ha="center", va="center",
            fontsize=10, fontweight="bold", bbox=proc_style)
    ax.text(5.5, 2.5, "Multi-Scale Attention\n(DS=1, 2, 4)", ha="center", va="center",
            fontsize=10, fontweight="bold", bbox=proc_style)

    ax.annotate("", xy=(2, 3.1), xytext=(3, 3.5), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
    ax.annotate("", xy=(5.5, 3.1), xytext=(4, 3.5), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

    ax.text(3.75, 2.5, "+", ha="center", va="center", fontsize=20, fontweight="bold", color="#64748b")
    ax.text(3.75, 2.05, "IN PARALLEL", ha="center", va="center", fontsize=8, color="#dc2626", fontweight="bold")

    # Repeat block indicator
    ax.text(8.5, 2.5, "× 3 blocks", ha="center", va="center", fontsize=14, color="#64748b",
            fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cbd5e1"))

    # Output
    ax.text(3.75, 1.0, "Output Head (MLP)", ha="center", va="center", fontsize=11, bbox=block_style)
    ax.annotate("", xy=(3.75, 1.35), xytext=(3.75, 1.95), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

    ax.text(8.5, 1.0, "12 glucose predictions\n(next 60 min)", ha="center", va="center",
            fontsize=11, fontweight="bold", bbox=out_style)
    ax.annotate("", xy=(6.8, 1.0), xytext=(5.1, 1.0), arrowprops=dict(arrowstyle="->", color="#059669", lw=2))

    # Config text
    ax.text(11, 5, "Config:\nd_model = 32\nn_heads = 4\nn_blocks = 3\nff_units = 128\nparams = 197K",
            ha="center", va="top", fontsize=9, color="#64748b", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    ax.set_title("Sugar I Architecture (GluMind variant)", fontsize=15, fontweight="bold", pad=10)
    fig.savefig(os.path.join(OUT_DIR, "fig_architecture.png"))
    plt.close(fig)
    print("  -> fig_architecture.png")


# =============================================================================
# FIGURE 8: Checkpoint metrics evolution (MAE over training)
# =============================================================================
def fig_checkpoint_metrics():
    # From VAL_EPOCH checkpoints in tuning.txt
    epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mae =    [11.55, 11.44, 11.93, 11.75, 11.65, 11.50, 11.60, 11.43, 11.52, 11.48]
    rmse =   [18.24, 17.88, 18.09, 18.12, 18.14, 17.95, 18.07, 17.87, 17.97, 17.91]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, mae, "o-", color=COLORS["sugar"], linewidth=2, markersize=7, label="MAE (mg/dL)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE (mg/dL)", color=COLORS["sugar"])
    ax1.tick_params(axis="y", labelcolor=COLORS["sugar"])
    ax1.set_ylim(11.0, 12.2)

    ax2 = ax1.twinx()
    ax2.plot(epochs, rmse, "s--", color=COLORS["gluformer"], linewidth=2, markersize=6, label="RMSE (mg/dL)")
    ax2.set_ylabel("RMSE (mg/dL)", color=COLORS["gluformer"])
    ax2.tick_params(axis="y", labelcolor=COLORS["gluformer"])
    ax2.set_ylim(17.5, 18.5)

    # Best checkpoint marker
    best_idx = np.argmin(mae)
    ax1.plot(epochs[best_idx], mae[best_idx], "*", color="#dc2626", markersize=18, zorder=10)
    ax1.annotate(f"Best: {mae[best_idx]:.2f}", xy=(epochs[best_idx], mae[best_idx]),
                 xytext=(epochs[best_idx]+10, mae[best_idx]+0.15),
                 fontsize=10, color="#dc2626", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#dc2626"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Validation Metrics During Training (every 10 epochs)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, "fig_checkpoint_metrics.png"))
    plt.close(fig)
    print("  -> fig_checkpoint_metrics.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_training_curve()
    fig_per_cohort_mae()
    fig_overall_t2dm()
    fig_overall_t1dm()
    fig_error_gradient()
    fig_improvement_heatmap()
    fig_architecture()
    fig_checkpoint_metrics()
    print("Done! All figures saved to", OUT_DIR)
