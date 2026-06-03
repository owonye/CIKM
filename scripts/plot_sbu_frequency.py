from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["HotpotQA", "MuSiQue", "NQ", "TriviaQA"]
STOP_RATE = np.array([38.2, 33.7, 49.7, 59.4])
SBU_RATES = {
    "GPT-4.1-mini": np.array([31.7, 36.5, 41.9, 17.2]),
    "Gemma-4-E4B-it": np.array([38.5, 26.4, 45.1, 27.6]),
    "Qwen-2.5-7B-Inst.": np.array([34.0, 14.8, 40.6, 23.9]),
}


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(DATASETS))
    width = 0.22
    offsets = [-width, 0.0, width]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    hatches = ["", "///", "\\\\\\"]

    fig, ax = plt.subplots(figsize=(3.35, 2.25))

    for (label, values), offset, color, hatch in zip(SBU_RATES.items(), offsets, colors, hatches):
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch=hatch,
            zorder=3,
        )

    ax.plot(
        x,
        STOP_RATE,
        color="#555555",
        marker="D",
        markersize=3.2,
        linewidth=1.0,
        label="Stop rate",
        zorder=4,
    )

    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 65)
    ax.set_yticks(np.arange(0, 70, 10))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        handlelength=1.2,
        columnspacing=0.55,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "sbu_frequency.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "sbu_frequency.png", dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(3.35, 2.05))

    for (label, values), offset, color, hatch in zip(SBU_RATES.items(), offsets, colors, hatches):
        ax2.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch=hatch,
            zorder=3,
        )

    ax2.set_ylabel("SBU among stop cases (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASETS)
    ax2.set_ylim(0, 50)
    ax2.set_yticks(np.arange(0, 55, 10))
    ax2.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", length=0)

    ax2.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        handlelength=1.2,
        columnspacing=0.6,
    )

    fig2.tight_layout(pad=0.3)
    fig2.savefig(out_dir / "sbu_frequency_bars_only.pdf", bbox_inches="tight")
    fig2.savefig(out_dir / "sbu_frequency_bars_only.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
