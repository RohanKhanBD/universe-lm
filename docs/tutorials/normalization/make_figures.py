from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

COLORS = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "grid": "#d9e2ec",
    "blue": "#2f80ed",
    "green": "#2f9e44",
    "amber": "#f59f00",
    "red": "#c92a2a",
    "paper": "#fbfaf7",
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


def barh(ax, labels, values, baseline, title, subtitle, colors):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.68)
    ax.axvline(baseline, color=COLORS["red"], lw=2, ls="--", alpha=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("validation loss, lower is better")
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color=COLORS["ink"])
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=COLORS["muted"], fontsize=10)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9, color=COLORS["ink"])


def plot_full_stack():
    labels = ["pnorm1.5", "pnorm1", "channelscale", "LayerNorm", "RMSNorm"]
    values = [6.3063, 6.3156, 6.3253, 6.3487, 6.3584]
    colors = [COLORS["green"], COLORS["green"], COLORS["green"], COLORS["amber"], COLORS["red"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor(COLORS["paper"])
    barh(
        ax,
        labels,
        values,
        baseline=6.3584,
        title="Full-stack norm result",
        subtitle="V-embed + q-gain + SWA384 + RoPE250k. Mild robustness wins here.",
        colors=colors,
    )
    ax.set_xlim(6.29, 6.37)
    save(fig, "full_stack_results.png")


def plot_clean_baseline():
    labels = [
        "LayerNorm",
        "body + QK pnorm1.5",
        "body + V pnorm1.5",
        "pnorm1.75",
        "pnorm1.5",
        "RMSNorm",
    ]
    values = [6.3628, 6.3922, 6.4025, 6.4088, 6.4387, 6.4516]
    colors = [COLORS["green"], COLORS["blue"], COLORS["blue"], COLORS["amber"], COLORS["amber"], COLORS["red"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.patch.set_facecolor(COLORS["paper"])
    barh(
        ax,
        labels,
        values,
        baseline=6.4516,
        title="Clean-baseline norm result",
        subtitle="Plain full attention, no V-embed, no q-gain, no SWA. The winner moves.",
        colors=colors,
    )
    ax.set_xlim(6.34, 6.47)
    save(fig, "clean_baseline_results.png")


def plot_mechanism():
    channels = np.arange(1, 17)
    x = np.array([0.8, -0.7, 0.6, -0.9, 0.5, 0.7, -0.6, 0.8, 0.4, -0.5, 0.6, -0.7, 0.5, 0.6, -0.4, 8.0])
    rms_weights = x**2 / np.sum(x**2)
    p15_weights = np.abs(x) ** 1.5 / np.sum(np.abs(x) ** 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    fig.patch.set_facecolor(COLORS["paper"])
    axes[0].bar(channels, rms_weights, color=COLORS["red"], width=0.72)
    axes[1].bar(channels, p15_weights, color=COLORS["green"], width=0.72)

    for ax, title in zip(axes, ["RMSNorm denominator", "p-norm 1.5 denominator"]):
        ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["ink"])
        ax.set_xlabel("channel")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("share of scale estimate")
    axes[0].text(0.03, 0.92, "one huge channel dominates", transform=axes[0].transAxes, color=COLORS["red"], fontsize=10)
    axes[1].text(0.03, 0.92, "outlier still matters, but less", transform=axes[1].transAxes, color=COLORS["green"], fontsize=10)
    fig.suptitle("Why mild robustness can help", x=0.03, ha="left", fontsize=17, fontweight="bold", color=COLORS["ink"])
    save(fig, "mechanism.png")


def plot_workflow():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("1", "Try one norm", "same model, one change"),
        ("2", "Beat RMSNorm", "keep the real default"),
        ("3", "Strip the stack", "remove helper tricks"),
        ("4", "Pair seeds", "check noise"),
        ("5", "Teach the rule", "state the boundary"),
    ]
    xs = np.linspace(0.8, 9.2, len(steps))
    for i, (num, title, sub) in enumerate(steps):
        x = xs[i]
        circle = plt.Circle((x, 1.8), 0.38, color=COLORS["blue"], alpha=0.95)
        ax.add_patch(circle)
        ax.text(x, 1.8, num, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.text(x, 1.12, title, ha="center", va="center", color=COLORS["ink"], fontsize=11, fontweight="bold")
        ax.text(x, 0.78, sub, ha="center", va="center", color=COLORS["muted"], fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.45, 1.8), xytext=(x + 0.45, 1.8), arrowprops=dict(arrowstyle="->", lw=1.8, color=COLORS["muted"]))
    ax.text(0.2, 2.72, "A safer ablation workflow", fontsize=17, fontweight="bold", color=COLORS["ink"])
    save(fig, "workflow.png")


if __name__ == "__main__":
    plot_full_stack()
    plot_clean_baseline()
    plot_mechanism()
    plot_workflow()
