from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


POINTS = [
    {
        "label": "Diagnose->Expand",
        "short": "Diagnose\nExpand",
        "recovery": 0.508,
        "deficit": 0.273,
        "color": "#4C78A8",
        "marker": "o",
        "size": 42,
        "zorder": 4,
    },
    {
        "label": "Random candidate",
        "short": "Random",
        "recovery": 0.335,
        "deficit": 0.177,
        "color": "#9E9E9E",
        "marker": "o",
        "size": 34,
        "zorder": 3,
    },
    {
        "label": "Next-ranked candidate",
        "short": "Next",
        "recovery": 0.329,
        "deficit": 0.176,
        "color": "#BDBDBD",
        "marker": "^",
        "size": 42,
        "zorder": 3,
    },
    {
        "label": "Max sufficiency gain",
        "short": "Max suff.\ngain",
        "recovery": 0.401,
        "deficit": 0.205,
        "color": "#F58518",
        "marker": "s",
        "size": 40,
        "zorder": 4,
    },
    {
        "label": "Max query overlap",
        "short": "Max query\noverlap",
        "recovery": 0.524,
        "deficit": 0.284,
        "color": "#ECA82C",
        "marker": "s",
        "size": 40,
        "zorder": 4,
    },
    {
        "label": "Proposed selection",
        "short": "Proposed",
        "recovery": 0.675,
        "deficit": 0.367,
        "color": "#B279A2",
        "marker": "*",
        "size": 115,
        "zorder": 5,
    },
]


OFFSETS = {
    "Diagnose\nExpand": (-0.015, -0.010),
    "Random": (0.012, -0.002),
    "Next": (-0.004, -0.012),
    "Max suff.\ngain": (0.012, -0.014),
    "Max query\noverlap": (0.016, 0.000),
    "Proposed": (-0.020, 0.007),
}


ALIGN = {
    "Diagnose\nExpand": "right",
    "Random": "left",
    "Next": "right",
    "Max suff.\ngain": "left",
    "Max query\noverlap": "left",
    "Proposed": "right",
}


def main() -> None:
    out_dir = Path("figures")
    asset_dir = Path("assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    for point in POINTS:
        ax.scatter(
            point["recovery"],
            point["deficit"],
            s=point["size"],
            color=point["color"],
            marker=point["marker"],
            edgecolor="black",
            linewidth=0.45,
            zorder=point["zorder"],
        )
        dx, dy = OFFSETS[point["short"]]
        ax.text(
            point["recovery"] + dx,
            point["deficit"] + dy,
            point["short"],
            fontsize=6.6,
            va="center",
            ha=ALIGN[point["short"]],
            linespacing=0.9,
        )

    max_overlap = next(p for p in POINTS if p["short"] == "Max query\noverlap")
    proposed = next(p for p in POINTS if p["short"] == "Proposed")
    ax.annotate(
        "",
        xy=(proposed["recovery"] - 0.018, proposed["deficit"] - 0.006),
        xytext=(max_overlap["recovery"] + 0.018, max_overlap["deficit"] + 0.006),
        arrowprops={
            "arrowstyle": "->",
            "color": "#777777",
            "lw": 0.8,
            "linestyle": "--",
        },
        zorder=2,
    )
    ax.text(
        0.585,
        0.337,
        "+15.1 percentage points in recovery\n+0.083 deficit reduction",
        fontsize=5.8,
        color="#555555",
        ha="center",
        va="center",
    )

    ax.set_xlabel("Recovery rate")
    ax.set_ylabel("Anchor-deficit reduction")
    ax.set_xlim(0.29, 0.71)
    ax.set_ylim(0.15, 0.39)
    ax.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70])
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35])
    ax.grid(color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.295,
        0.384,
        "Higher is better",
        fontsize=6.0,
        color="#555555",
        ha="left",
        va="top",
    )

    fig.tight_layout(pad=0.35)
    for target_dir in (out_dir, asset_dir):
        fig.savefig(target_dir / "repair_tradeoff.pdf", bbox_inches="tight")
        fig.savefig(target_dir / "repair_tradeoff.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
