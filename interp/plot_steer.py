"""The steering control, in one figure: phantom pull tracks model damage,
not the direction we steered."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessbench.analysis import GRID, INK, INK_2, SURFACE, VARIANT_COLOR  # noqa: E402

REAL = VARIANT_COLOR["chess960"]
RAND = INK_2


def series(path: str):
    rows = json.load(open(path))
    out = []
    for a in sorted({r["alpha"] for r in rows}):
        s = [r for r in rows if r["alpha"] == a]
        out.append((a, statistics.mean(r["best_legal_lp"] for r in s),
                    sum(r["book_wins"] for r in s) / len(s)))
    return out


def main() -> int:
    real = series("interp/steer_small.json")
    rand = series("interp/steer_random.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=200)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        fig.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=9)

    # A: the sweeps look different...
    for data, color, lab in ((real, REAL, "'standard back rank' direction"),
                             (rand, RAND, "random direction (control)")):
        ax1.plot([d[0] for d in data], [d[2] for d in data], "o-", color=color,
                 linewidth=2, markersize=6, label=lab)
    ax1.set_title("Steering sweeps disagree on sign", color=INK, fontsize=11, loc="left")
    ax1.set_xlabel("steering coefficient α", color=INK_2, fontsize=9)
    ax1.set_ylabel("phantom win rate", color=INK_2, fontsize=9)
    ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    leg = ax1.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)

    # B: ...but collapse onto one damage curve
    for data, color, lab in ((real, REAL, "'standard back rank' direction"),
                             (rand, RAND, "random direction (control)")):
        ax2.plot([d[1] for d in data], [d[2] for d in data], "o", color=color,
                 markersize=7, label=lab)
    allpts = sorted(real + rand, key=lambda d: d[1])
    ax2.plot([d[1] for d in allpts], [d[2] for d in allpts], "-", color=GRID,
             linewidth=2, zorder=0)
    ax2.invert_xaxis()
    ax2.set_title("…but both lie on one damage curve (r = −0.98)", color=INK,
                  fontsize=11, loc="left")
    ax2.set_xlabel("model health  →  worse  (log-prob of best LEGAL move)",
                   color=INK_2, fontsize=9)
    ax2.set_ylabel("phantom win rate", color=INK_2, fontsize=9)
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    fig.text(0.005, 0.005,
             "A random direction reproduces the effect as strongly as the semantic one: "
             "the phantom pull tracks degradation, not the steered direction. "
             "No causal support for a 'standard back rank' feature.",
             fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("interp/steering_control.png")
    print("wrote interp/steering_control.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
